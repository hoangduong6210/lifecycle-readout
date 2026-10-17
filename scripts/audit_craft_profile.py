"""Check author-profile provenance, raw history inputs and GPU/CPU roundtrip."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import sha256, save_json
from lifecycle_readout.craft_adapter import make_author_model


def audit(attempt):
    attempt = attempt.resolve()
    folder = attempt / 'artifacts'
    job = json.loads((attempt / 'job.json').read_text())
    assert job['status'] == 'completed' and job['cleanup_verified']
    assert sha256(attempt / 'source.zip') == job['source_bundle_sha256']
    assert sha256(attempt / 'artifacts.tar.gz') == job['archive_sha256']
    for line in (folder / 'checksums.sha256').read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        p = (folder / rel).resolve()
        assert p.is_relative_to(folder) and sha256(p) == expected, rel
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    with zipfile.ZipFile(attempt / 'source.zip') as archive:
        assert manifest == json.loads(archive.read('source_manifest.json'))
        for rel, expected in manifest.items():
            import hashlib
            assert hashlib.sha256(archive.read(rel)).hexdigest() == expected, rel
    result = json.loads((folder / 'results.json').read_text())
    settings = result['settings']
    for item in settings['input_files']:
        assert sha256(ROOT / item['path']) == item['sha256'] == manifest[item['path']]
    for rel in ['src/lifecycle_readout/craft_adapter.py', 'scripts/profile_craft.py']:
        assert sha256(ROOT / rel) == manifest[rel], 'Replay archived source after source drift'
    assert not result['test_scored'] and not result['mrr_computed'] and not result['publication_eligible']
    assert [p['train_batch_size'] for p in result['training_profiles']] == settings['train_batch_sizes']
    with np.load(ROOT / settings['prepared_path']) as stored:
        data = {k: stored[k] for k in stored.files}
    assert set(np.unique(data['split'])) == {0, 1}
    with np.load(ROOT / settings['candidates_path']) as stored:
        candidates = {k: stored[k] for k in stored.files}
    identity = json.loads((ROOT / 'results/audit/q1_preflight_v1/craft_profile/anchor_identity.json').read_text())
    assert sha256(ROOT / identity['parent_path']) == identity['parent_sha256']
    assert sha256(ROOT / identity['output_path']) == identity['output_sha256']
    with np.load(ROOT / identity['parent_path']) as parent:
        for key in parent.files:
            np.testing.assert_array_equal(candidates[key], parent[key][candidates['validation_rows']])
    with np.load(folder / 'roundtrip.npz') as stored:
        saved = {k: stored[k] for k in stored.files}
    rows = saved['validation_rows']
    assert len(rows) == settings['validation_query_count']
    np.testing.assert_array_equal(rows, candidates['validation_rows'])
    global_rows = np.flatnonzero(data['split'] == 1)[rows]
    np.testing.assert_array_equal(data['event_ids'][global_rows], candidates['event_ids'])
    np.testing.assert_array_equal(saved['input_4'], np.c_[candidates['positive_destinations'], candidates['negative_destinations']])
    np.testing.assert_array_equal(saved['input_3'], data['time'][global_rows])
    # Raw row filtering verifies all saved source contexts. Destination last
    # times are independently reduced across the complete strictly past prefix.
    for row, event in enumerate(global_rows):
        t, u = data['time'][event], data['src'][event]
        past = np.flatnonzero((data['src'] == u) & (data['time'] < t))[-settings['history_length']:]
        seq = np.zeros(settings['history_length'], dtype=np.int64)
        seq_t = np.zeros(settings['history_length'], dtype=np.float64)
        seq[:len(past)], seq_t[:len(past)] = data['dst'][past], data['time'][past]
        np.testing.assert_array_equal(saved['input_0'][row], seq)
        np.testing.assert_array_equal(saved['input_2'][row], seq_t)
        # Original forward changes zero sequence lengths to one in place.
        assert saved['input_1'][row] == max(1, len(past))
        cutoff = np.searchsorted(data['time'], t, side='left')
        last = np.full(int(data['dst'].max()) + 1, -100000., dtype=np.float64)
        np.maximum.at(last, data['dst'][:cutoff], data['time'][:cutoff])
        np.testing.assert_array_equal(saved['input_5'][row], last[saved['input_4'][row]])
    for profile in result['training_profiles']:
        samples = profile['seconds_per_step_samples']
        assert len(samples) == settings['measured_steps'] and min(samples) > 0
        assert float(np.mean(samples)) == profile['seconds_per_step_mean']
        expected = float(np.mean(samples) * np.ceil(np.count_nonzero(data['split'] == 0) / profile['train_batch_size']))
        assert expected == profile['projected_train_epoch_seconds']
    torch.set_num_threads(1)
    model = make_author_model(ROOT / settings['author_path'], settings, int(data['dst'].max()))
    model.load_state_dict(torch.load(folder / 'roundtrip_state.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        positive, negative = model.predict(*(torch.from_numpy(saved[f'input_{i}']) for i in range(6)))
    np.testing.assert_allclose(positive.numpy(), saved['positive'], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(negative.numpy(), saved['negative'], rtol=2e-4, atol=2e-5)
    return dict(status='PASS', attempt=str(attempt.relative_to(ROOT)),
        scope='All 20 saved source histories, candidate identities and global update times; short timing arithmetic; author GPU/CPU roundtrip.',
        test_scored=False, mrr_computed=False, publication_eligible=False,
        complete_epoch_measured=False, author_accuracy_reproduced=False,
        max_positive_roundtrip_error=float(np.max(np.abs(positive.numpy() - saved['positive']))),
        max_negative_roundtrip_error=float(np.max(np.abs(negative.numpy() - saved['negative']))),
        auditor_sha256=sha256(Path(__file__)), result_sha256=sha256(folder / 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Audit output must be fresh')
    report = audit(args.attempt)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
