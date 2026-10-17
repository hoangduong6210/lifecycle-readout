"""Audit saved validation controls; no fitting, candidate replacement or test data."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import tomllib
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference_scores(data, candidates, anchors):
    """Independent sparse row replay for declared score anchors, including ties.

    Keep each source's current timestamp minimum and previous timestamp minimum.
    This avoids the production grouped sorting/dense transition implementation.
    Only transition/pair counts needed by the anchors are retained.
    """
    u, v, t = (data[k] for k in ('src', 'dst', 'time'))
    contexts, wanted_transitions, wanted_pairs = {}, set(), set()
    for row in anchors:
        src, timestamp = candidates['sources'][row], candidates['timestamps'][row]
        history = (u == src) & (t < timestamp)
        latest = t[history].max() if history.any() else None
        context = int(v[history & (t == latest)].min()) if latest is not None else -1
        contexts[row] = context
        dsts = np.r_[candidates['positive_destinations'][row], candidates['negative_destinations'][row]]
        wanted_pairs.update((int(src), int(dst)) for dst in dsts)
        wanted_transitions.update((context, int(dst)) for dst in dsts if context >= 0)
    cutoffs = {}
    for row in anchors:
        cutoffs.setdefault(int(np.searchsorted(t, candidates['timestamps'][row], side='left')), []).append(row)
    source_state, seen, transitions = {}, set(), Counter()
    pop = np.zeros(int(v.max()) + 1, dtype=np.int64)
    catalog_size = len(np.unique(v[data['split'] == 0]))
    expected = {}
    for index in range(max(cutoffs) + 1):
        for row in cutoffs.get(index, []):
            src = int(candidates['sources'][row])
            context = source_state[src][1] if src in source_state else -1
            assert context == contexts[row], 'Anchor context differs from strict-past lookup'
            dsts = np.r_[candidates['positive_destinations'][row], candidates['negative_destinations'][row]]
            popularity = (pop[dsts] + 1.) / (index + catalog_size)
            expected[row] = dict(
                edge_memory=np.array([float((src, int(dst)) in seen) for dst in dsts]),
                popularity=popularity,
                pooled_transition=np.array([transitions[context, int(dst)] for dst in dsts]) + popularity)
        if index == max(cutoffs):
            break
        src, dst, timestamp = int(u[index]), int(v[index]), float(t[index])
        state = source_state.get(src)
        if state is None:
            previous = -1
            source_state[src] = (timestamp, dst, previous)
        elif timestamp > state[0]:
            previous = state[1]
            source_state[src] = (timestamp, dst, previous)
        else:
            assert timestamp == state[0]
            previous = state[2]
            source_state[src] = (timestamp, min(dst, state[1]), previous)
        if (src, dst) in wanted_pairs:
            seen.add((src, dst))
        if (previous, dst) in wanted_transitions:
            transitions[previous, dst] += 1
        pop[dst] += 1
    return expected


def audit(attempt):
    attempt = attempt.resolve()
    folder = attempt / 'artifacts'
    job = json.loads((attempt / 'job.json').read_text())
    assert job['status'] == 'completed' and job['cleanup_verified']
    assert digest(attempt / 'source.zip') == job['source_bundle_sha256']
    archives = list(attempt.glob('*.tar.gz'))
    assert len(archives) == 1 and digest(archives[0]) == job['archive_sha256']
    for line in (folder / 'checksums.sha256').read_text().splitlines():
        sha, rel = line.split(maxsplit=1)
        path = (folder / rel).resolve()
        assert path.is_relative_to(folder) and digest(path) == sha, rel
    with zipfile.ZipFile(attempt / 'source.zip') as archive:
        manifest = json.loads(archive.read('source_manifest.json'))
        assert manifest == json.loads((folder / 'source_manifest.json').read_text())
        for rel, sha in manifest.items():
            assert hashlib.sha256(archive.read(rel)).hexdigest() == sha, rel
        settings = tomllib.loads(archive.read('configs/tgbseq_baseline_screen_v1.toml').decode())
        assert manifest[settings['protocol']] == job['protocol_sha256']
        assert manifest['configs/tgbseq_baseline_screen_v1.toml'] == job['config_sha256']
    result = json.loads((folder / 'results.json').read_text())
    assert result['settings'] == settings and not result['test_scored'] and not result['smoke']
    data_path = ROOT / settings['prepared_path']
    assert digest(data_path) == result['input_sha256'] == manifest[settings['prepared_path']]
    with np.load(data_path) as stored:
        data = {k: stored[k] for k in stored.files}
    assert set(np.unique(data['split'])) == {0, 1}
    val = data['split'] == 1
    with np.load(folder / 'validation_candidates.npz') as stored:
        candidates = {k: stored[k] for k in stored.files}
    for saved, source in [('event_ids', 'event_ids'), ('sources', 'src'),
                          ('positive_destinations', 'dst'), ('timestamps', 'time')]:
        np.testing.assert_array_equal(candidates[saved], data[source][val])
    negatives = candidates['negative_destinations']
    n = int(val.sum())
    assert negatives.shape == (n, settings['negative_count']) and result['query_count'] == n
    catalog = np.unique(data['dst'][data['split'] == 0])
    assert np.isin(negatives, catalog).all()
    # Every negative checked against all same-source/time positives, not just its row.
    positive_keys = np.rec.fromarrays([candidates[k] for k in ['timestamps', 'sources', 'positive_destinations']])
    positive_keys.sort()
    for start in range(0, n, 4096):
        stop = min(n, start + 4096)
        keys = np.rec.fromarrays([np.repeat(candidates['timestamps'][start:stop], negatives.shape[1]),
                                 np.repeat(candidates['sources'][start:stop], negatives.shape[1]),
                                 negatives[start:stop].reshape(-1)])
        locations = np.searchsorted(positive_keys, keys)
        assert not (positive_keys[np.minimum(locations, n - 1)] == keys).any()
    anchors = np.unique(np.r_[0, n - 1, np.linspace(0, n - 1, 24, dtype=int)])
    oracle = reference_scores(data, candidates, anchors)
    metrics = {}
    for kind in settings['models']:
        path = folder / f'{kind}_predictions.npz'
        assert digest(path) == result['models'][kind]['prediction_sha256']
        with np.load(path) as saved:
            pos, neg, rr = saved['positive'], saved['negative'], saved['reciprocal_rank']
        assert pos.shape == (n,) and neg.shape == negatives.shape and rr.shape == (n,)
        assert np.isfinite(pos).all() and np.isfinite(neg).all()
        for start in range(0, n, 4096):
            p, q = pos[start:start + 4096], neg[start:start + 4096]
            # Independent expression of the official optimistic/pessimistic midpoint.
            rank = (1. + (q >= p[:, None]).sum(axis=1) + 1. + (q > p[:, None]).sum(axis=1)) / 2.
            np.testing.assert_array_equal(rr[start:start + 4096], 1. / rank)
        assert float(rr.mean()) == result['models'][kind]['mrr']
        for row in anchors:
            np.testing.assert_allclose(np.r_[pos[row], neg[row]], oracle[row][kind], rtol=1e-13, atol=1e-15)
        metrics[kind] = float(rr.mean())
    return dict(status='PASS', attempt=str(attempt.relative_to(ROOT)), query_count=n,
        full_candidate_and_metric_checks=True, score_anchor_rows=anchors.tolist(),
        score_anchor_count=len(anchors), score_anchor_method='Independent sparse chronological replay',
        full_score_replay=False, test_scored=False, publication_eligible=False, metrics=metrics,
        source_bundle_sha256=job['source_bundle_sha256'], result_sha256=digest(folder / 'results.json'),
        auditor_sha256=digest(Path(__file__)),
        limitations='24 deterministic score anchors; all candidates and reciprocal ranks audited. One candidate draw; not official test or scientific superiority evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Audit output must be fresh')
    report = audit(args.attempt)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
