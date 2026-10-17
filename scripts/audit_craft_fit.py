"""Audit full validation endpoints and independently rebuild raw-history anchors."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import tomllib
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import save_json, sha256
from lifecycle_readout.craft_adapter import make_author_model


def verify_archived_contract(archive, job, manifest, result):
    """Bind reported settings to the pre-execution configuration and protocol."""
    settings = result['settings']
    config_raw = archive.read(settings['config_path'])
    config_hash = hashlib.sha256(config_raw).hexdigest()
    assert config_hash == job['config_sha256'] == manifest[settings['config_path']]
    config = tomllib.loads(config_raw.decode())
    assert settings == config, 'Reported settings differ from sealed configuration'
    protocol_raw = archive.read(config['protocol'])
    assert hashlib.sha256(protocol_raw).hexdigest() == job['protocol_sha256'] == manifest[config['protocol']]
    protocol = tomllib.loads(protocol_raw.decode())
    assert result['protocol_id'] == config['protocol_id'] == protocol['protocol_id']
    assert result['dataset'] in config['datasets']
    assert settings['seeds'] == protocol['seeds']
    for key in ['max_epochs', 'patience', 'training_budget_seconds']:
        assert settings[key] == protocol[key], key
    assert result['convergence_demonstrated'] is False
    return config


def verify_epoch_selection(folder, run, settings, train_count, query_count, elapsed):
    """Audit complete epochs and the first declared stopping opportunity."""
    epochs = run['validation_history']
    assert run['completed_epochs'] == len(epochs) and 0 < len(epochs) <= settings['max_epochs']
    assert [e['epoch'] for e in epochs] == list(range(1, len(epochs) + 1))
    best, selected, stale, previous_elapsed, selected_rr = -np.inf, None, 0, -np.inf, None
    for position, epoch in enumerate(epochs):
        assert epoch['train_count'] == train_count
        assert re.fullmatch(r'[0-9a-f]{64}', epoch['candidate_trace_sha256'])
        assert math.isfinite(epoch['training_seconds']) and epoch['training_seconds'] > 0
        assert math.isfinite(epoch['train_loss'])
        assert previous_elapsed < epoch['elapsed_seconds'] <= elapsed
        previous_elapsed = epoch['elapsed_seconds']
        with np.load(folder / f'seed{run["seed"]}_epoch{epoch["epoch"]}_validation_rr.npz') as saved:
            rr = saved['reciprocal_rank']
        assert rr.shape == (query_count,) and np.isfinite(rr).all()
        assert (rr >= 1. / (settings['negative_count'] + 1)).all() and (rr <= 1).all()
        np.testing.assert_allclose(2. / rr, np.round(2. / rr), rtol=0, atol=1e-10)
        assert float(rr.mean()) == epoch['validation_mrr']
        improved = epoch['validation_mrr'] > best
        assert improved == epoch['selected_at_this_epoch']
        if improved:
            best, selected, stale, selected_rr = epoch['validation_mrr'], epoch['epoch'], 0, rr.copy()
        else:
            stale += 1
        if stale >= settings['patience']:
            assert position == len(epochs) - 1 and run['stop_reason'] == 'validation_patience', 'Training continued past patience'
    assert selected == run['selected_epoch']
    reason = run['stop_reason']
    if reason == 'max_epochs':
        assert len(epochs) == settings['max_epochs']
    elif reason == 'validation_patience':
        assert stale >= settings['patience']
    elif reason in {'runtime_cap_between_epochs', 'runtime_cap_partial_epoch'}:
        assert len(epochs) < settings['max_epochs'] and elapsed >= settings['training_budget_seconds']
        if reason == 'runtime_cap_partial_epoch':
            partial = json.loads((folder / f'partial_seed{run["seed"]}_epoch{len(epochs)+1}.json').read_text())
            assert partial['epoch_completed'] is False
            assert 0 <= partial['rows'] < train_count and partial['total_epoch_rows'] == train_count
    else:
        raise AssertionError('Undeclared stopping reason')
    return best, selected, selected_rr


def audit(attempt):
    attempt = attempt.resolve(); folder = attempt / 'artifacts'
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
            assert hashlib.sha256(archive.read(rel)).hexdigest() == expected, rel
    result = json.loads((folder / 'results.json').read_text())
    with zipfile.ZipFile(attempt / 'source.zip') as archive:
        settings = verify_archived_contract(archive, job, manifest, result)
    assert not result['test_scored'] and not result['publication_eligible'] and not result['smoke']
    assert result['evaluation_split'] == 'validation'
    for item in settings['input_files']:
        assert sha256(ROOT / item['path']) == item['sha256'] == manifest[item['path']]
    for rel in ['src/lifecycle_readout/craft_adapter.py', 'src/lifecycle_readout/craft_training.py']:
        assert sha256(ROOT / rel) == manifest[rel], 'Use archived source after drift'
    with np.load(ROOT / settings['prepared_path']) as stored:
        data = {k: stored[k] for k in stored.files}
    with np.load(ROOT / settings['candidates_path']) as stored:
        candidates = {k: stored[k] for k in stored.files}
    assert set(np.unique(data['split'])) == {0, 1}
    train, val = data['split'] == 0, data['split'] == 1
    n = int(val.sum()); assert result['train_count'] == int(train.sum()) and result['validation_count'] == n
    for saved, raw in [('event_ids', 'event_ids'), ('sources', 'src'),
                       ('timestamps', 'time'), ('positive_destinations', 'dst')]:
        np.testing.assert_array_equal(candidates[saved], data[raw][val])
    assert [r['seed'] for r in result['per_seed_runs']] == settings['seeds']
    assert result['complete_declared_seeds']
    torch.set_num_threads(1)
    anchors = np.linspace(0, n - 1, 20, dtype=int)
    # Construct model inputs from raw events, without invoking the production adapter.
    length = settings['history_length']
    seq = np.zeros((len(anchors), length), dtype=np.int64)
    seq_time = np.zeros_like(seq, dtype=np.float64); counts = np.zeros(len(anchors), dtype=np.int64)
    query_time = candidates['timestamps'][anchors]
    items = np.c_[candidates['positive_destinations'][anchors], candidates['negative_destinations'][anchors]]
    updates = np.full(items.shape, -100000., dtype=np.float64)
    for i, row in enumerate(anchors):
        src, t = candidates['sources'][row], query_time[i]
        past = np.flatnonzero((data['src'] == src) & (data['time'] < t))[-length:]
        counts[i] = len(past); seq[i, :len(past)] = data['dst'][past]
        seq_time[i, :len(past)] = data['time'][past]
        cutoff = np.searchsorted(data['time'], t, side='left')
        last = np.full(int(data['dst'].max()) + 1, -100000., dtype=np.float64)
        np.maximum.at(last, data['dst'][:cutoff], data['time'][:cutoff])
        updates[i] = last[items[i]]
    report = []
    for run in result['per_seed_runs']:
        seed = run['seed']; epochs = run['validation_history']
        assert set(run['models']) == set(settings['models'])
        best, selected, selected_rr = verify_epoch_selection(folder, run, settings, int(train.sum()), n, result['elapsed_seconds'])
        reason = run['stop_reason']
        checkpoint = folder / f'best_seed{seed}.pt'
        prediction = folder / f'CRAFT_seed{seed}_validation_predictions.npz'
        model_record = run['models']['CRAFT']
        assert model_record['query_count'] == n
        assert sha256(checkpoint) == model_record['checkpoint_sha256']
        assert sha256(prediction) == model_record['prediction_sha256']
        with np.load(prediction) as saved:
            positive, negative, rr = saved['positive'], saved['negative'], saved['reciprocal_rank']
            np.testing.assert_array_equal(saved['event_ids'], candidates['event_ids'])
        np.testing.assert_array_equal(rr, selected_rr, err_msg='Final ranks differ from the selected epoch')
        assert positive.shape == (n,) and negative.shape == (n, settings['negative_count'])
        assert np.isfinite(positive).all() and np.isfinite(negative).all()
        for start in range(0, n, 4096):
            p, q = positive[start:start+4096], negative[start:start+4096]
            expected = 1. / ((1. + (q >= p[:, None]).sum(axis=1) + 1. + (q > p[:, None]).sum(axis=1)) / 2.)
            np.testing.assert_array_equal(rr[start:start+4096], expected)
        assert float(rr.mean()) == best == model_record['validation_mrr']
        model = make_author_model(ROOT / settings['author_path'], settings, int(data['dst'].max()))
        assert sum(p.numel() for p in model.parameters()) == model_record['parameter_count']
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)); model.eval()
        with torch.no_grad():
            p, q = model.predict(*(torch.from_numpy(x.copy()) for x in [seq, counts, seq_time, query_time, items, updates]))
        np.testing.assert_allclose(p.numpy(), positive[anchors], rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(q.reshape(len(anchors), -1).numpy(), negative[anchors], rtol=2e-4, atol=2e-5)
        report.append(dict(seed=seed, completed_epochs=len(epochs), selected_epoch=selected,
                           stop_reason=reason, validation_mrr=best))
    return dict(status='PASS', attempt=str(attempt.relative_to(ROOT)), seeds=report,
        full_candidate_identity_and_mrr_checks=True, raw_history_score_anchor_count=len(anchors),
        archived_configuration_and_protocol_binding=True, first_stopping_opportunity_checked=True,
        selected_epoch_per_query_ranks_equal=True,
        full_training_negative_replay=False, full_training_reproduction=False,
        test_scored=False, publication_eligible=False, auditor_sha256=sha256(Path(__file__)),
        limitations='Single-seed validation development; complete epoch counts and trace identities recorded, not independently replayed. CPU/GPU scores checked at 20 raw-history anchors only.')


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
