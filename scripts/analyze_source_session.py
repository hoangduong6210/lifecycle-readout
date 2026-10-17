"""Posthoc session-gap stratification of retained validation reciprocal ranks."""
from pathlib import Path
import argparse
import hashlib
import json
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'protocols/source_session_diagnostic_v1.toml'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_gaps(sources, timestamps):
    """Return elapsed time since the source's prior distinct timestamp group."""
    sources, timestamps = np.asarray(sources), np.asarray(timestamps)
    if sources.ndim != 1 or timestamps.shape != sources.shape or not len(sources):
        raise ValueError('Expected nonempty one-dimensional aligned arrays')
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError('Timestamps must be finite and globally nondecreasing')
    # Stable sorting by source preserves the original chronological order.
    order = np.argsort(sources, kind='stable')
    u, t = sources[order], timestamps[order].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, (u[1:] != u[:-1]) | (t[1:] != t[:-1])])
    gu, gt = u[starts], t[starts]
    previous = np.r_[np.nan, gt[:-1]]
    previous[np.r_[True, gu[1:] != gu[:-1]]] = np.nan
    groups = gt - previous
    result = np.empty(len(sources), dtype=np.float64)
    result[order] = np.repeat(groups, np.diff(np.r_[starts, len(sources)]))
    assert np.all(result[np.isfinite(result)] > 0)
    return result


def training_cuts(gaps, split):
    values = gaps[(split == 0) & np.isfinite(gaps)]
    if not len(values):
        raise ValueError('No available training gaps')
    return np.quantile(values, [.25, .5, .75], method='linear')


def summarize(gaps, ranks, cuts):
    labels = np.searchsorted(cuts, gaps, side='right')
    labels[~np.isfinite(gaps)] = -1
    rows = []
    for label in [-1, 0, 1, 2, 3]:
        mask = labels == label
        count = int(mask.sum())
        metrics = {key: float(value[mask].mean()) if count else None
                   for key, value in ranks.items()}
        rows.append(dict(stratum='unavailable' if label == -1 else f'interval_{label+1}',
                         count=count, fraction=count/len(gaps),
                         mean_gap_seconds=float(gaps[mask].mean()) if count and label != -1 else None,
                         median_gap_seconds=float(np.median(gaps[mask])) if count and label != -1 else None,
                         mrr=metrics,
                         transition_minus_popularity=metrics['pooled_transition']-metrics['popularity'] if count else None))
    for key, value in ranks.items():
        reconstructed = sum(row['count']*row['mrr'][key] for row in rows if row['count'])/len(gaps)
        np.testing.assert_allclose(reconstructed, value.mean(), rtol=1e-14, atol=1e-15)
    assert sum(row['count'] for row in rows) == len(gaps)
    return rows, labels


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    attempt = root/'results/audit/lcr-ML-20M-ba89635570'
    index_path = root/'evidence/indexes/ml20m_screen_v1.toml'
    index = tomllib.loads(index_path.read_text())
    indexed = {entry['path']: entry['sha256'] for entry in index['artifact']}
    verified = {}

    def verify(path):
        relative = path.relative_to(root).as_posix()
        actual = digest(path)
        assert indexed[relative] == actual, relative
        verified[relative] = actual

    for path in [attempt/'job.json', attempt/'independent_audit.json',
                 attempt/'artifacts/results.json', attempt/'artifacts/validation_candidates.npz',
                 attempt/'artifacts/source_manifest.json']:
        verify(path)
    job = json.loads((attempt/'job.json').read_text())
    assert job['status'] == 'completed' and job['cleanup_verified']
    prior = json.loads((attempt/'independent_audit.json').read_text())
    assert prior['status'] == 'PASS' and prior['full_candidate_and_metric_checks']
    result = json.loads((attempt/'artifacts/results.json').read_text())
    assert not result['test_scored']
    settings = result['settings']
    prepared = root/settings['prepared_path']
    actual = digest(prepared)
    manifest = json.loads((attempt/'artifacts/source_manifest.json').read_text())
    assert actual == result['input_sha256'] == manifest[settings['prepared_path']]
    verified[settings['prepared_path']] = actual
    with np.load(prepared) as saved:
        data = {key: saved[key] for key in ['src','dst','time','split','event_ids']}
    assert set(np.unique(data['split'])) == {0, 1}
    val = data['split'] == 1
    with np.load(attempt/'artifacts/validation_candidates.npz') as candidates:
        for key, raw in [('event_ids','event_ids'),('sources','src'),
                         ('timestamps','time'),('positive_destinations','dst')]:
            np.testing.assert_array_equal(candidates[key], data[raw][val])
    gaps = source_gaps(data['src'], data['time'])
    cuts = training_cuts(gaps, data['split'])
    anchors = np.unique(np.r_[np.linspace(0, int((~val).sum())-1, 12, dtype=int),
                             np.flatnonzero(val)[np.linspace(0, int(val.sum())-1, 12, dtype=int)]])
    for row in anchors:
        past = data['time'][(data['src'] == data['src'][row]) & (data['time'] < data['time'][row])]
        expected = float(data['time'][row] - past.max()) if len(past) else np.nan
        np.testing.assert_equal(gaps[row], expected)
    ranks = {}
    for model in ['edge_memory','popularity','pooled_transition']:
        path = attempt/f'artifacts/{model}_predictions.npz'
        verify(path)
        with np.load(path) as saved:
            rr = saved['reciprocal_rank']
        assert rr.shape == (int(val.sum()),) and np.isfinite(rr).all()
        assert ((rr > 0) & (rr <= 1)).all()
        assert float(rr.mean()) == result['models'][model]['mrr']
        ranks[model] = rr
    rows, labels = summarize(gaps[val], ranks, cuts)
    np.savez_compressed(output/'query_diagnostic.npz', event_ids=data['event_ids'][val],
                        source_gap_seconds=gaps[val], interval=labels, **ranks)
    report = dict(status='PASS', protocol_id='LCR-P-SESSION-DIAG-001',
                  protocol_sha256=digest(root/PROTOCOL), analyzer_sha256=digest(Path(__file__)),
                  parent_index_sha256=digest(index_path), verified_inputs=verified,
                  training_cutoffs_seconds=cuts.tolist(), training_gap_count=int(((data['split']==0)&np.isfinite(gaps)).sum()),
                  query_count=int(val.sum()), gap_anchor_rows=anchors.tolist(),
                  overall_mrr={key:float(value.mean()) for key,value in ranks.items()}, strata=rows,
                  test_read=False, fitting=False, gpu_allocated=False, publication_eligible=False,
                  limitations='Posthoc descriptive heterogeneity on one shared candidate draw. Not a decay-gate experiment or causal effect. Original prior audit covers all ranks and 24 score anchors, not full score replay.')
    (output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(ROOT, args.output.resolve())
    print(json.dumps({key:report[key] for key in ['status','training_cutoffs_seconds','query_count','strata']},indent=2))
