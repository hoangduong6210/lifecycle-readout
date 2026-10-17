"""Replay retained hard-control checkpoints, without fitting or selecting models.

Imports the exact archived model source in a fresh process. History inputs are
rebuilt below from prepared events rather than the archived stream builder.
This checks inference portability, not clean-environment training reproduction.
"""
from collections import defaultdict, deque
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rebuild_inputs(data, candidates, budget, decay):
    """Read an entire time group before observing its allowed events."""
    ids = candidates['event_ids'][candidates['split'] == 2]
    rows = {int(event): row for row, event in enumerate(ids)}
    negative = candidates['negative_destinations'][candidates['split'] == 2]
    excluded = set(candidates['excluded_training_ids'].tolist())
    neighbors = defaultdict(lambda: deque(maxlen=budget))
    pairs = {}
    edges = np.full((len(ids), 3, budget), -1, dtype=np.int64)
    phi = np.empty((len(ids), 2, 6), dtype=np.float32)
    elapsed = np.empty((len(ids), 2), dtype=np.float32)
    src, dst, times = (data[k] for k in ['sources', 'destinations', 'timestamps'])
    start = 0
    while start < len(times):
        t = float(times[start])
        stop = int(np.searchsorted(times, t, side='right'))
        for event in range(start, stop):
            if event not in rows:
                continue
            assert event not in excluded
            row = rows[event]
            u, v, q = int(src[event]), int(dst[event]), int(negative[row])
            for slot, node in enumerate([u, v, q]):
                history = list(neighbors[node])
                edges[row, slot, :len(history)] = history
            for slot, node in enumerate([v, q]):
                last, count = pairs.get((u, node), (None, 0))
                dt = 0. if last is None else t - last
                elapsed[row, slot] = dt
                phi[row, slot] = [
                    .1 if last is None else .1 + .9 * np.exp(-decay * dt),
                    np.log1p(dt), np.log1p(count), count / np.sqrt(max(1, event)),
                    np.sin(2 * np.pi * (t % 86400) / 86400),
                    np.cos(2 * np.pi * (t % 86400) / 86400)]
        for event in range(start, stop):
            if event in excluded:
                continue
            u, v = int(src[event]), int(dst[event])
            neighbors[u].append(event)
            neighbors[v].append(event)
            pairs[(u, v)] = (t, pairs.get((u, v), (None, 0))[1] + 1)
        start = stop
    mask = edges >= 0
    assert (times[np.maximum(edges, 0)][mask] < np.broadcast_to(times[ids, None, None], edges.shape)[mask]).all()
    return ids, edges, phi, elapsed


def scalar_log_score(logits, raw_rate, elapsed, time_unit):
    """Stable mathematical score before probability clipping; no new fitting."""
    import torch.nn.functional as F
    return F.logsigmoid(logits.double()) - F.softplus(raw_rate.double()) * elapsed.double() / time_unit


def replay(attempt, output, scalar_unclipped=False):
    started = time.monotonic()
    attempt = attempt.resolve()
    job = json.loads((attempt / 'job.json').read_text())
    assert job['status'] == 'completed' and job['cleanup_verified']
    assert sha256(attempt / 'source.zip') == job['source_bundle_sha256']
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'sealed_source'
    with zipfile.ZipFile(attempt / 'source.zip') as bundle:
        manifest = json.loads(bundle.read('source_manifest.json'))
        for name, expected in manifest.items():
            if name.startswith('src/') and name.endswith('.py'):
                target = (source / name).resolve()
                assert target.is_relative_to(source.resolve())
                raw = bundle.read(name)
                assert hashlib.sha256(raw).hexdigest() == expected
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    assert not any(k == 'lifecycle_readout' or k.startswith('lifecycle_readout.') for k in sys.modules)
    sys.path.insert(0, str(source / 'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    import lifecycle_readout.hard_controls as archived
    assert Path(archived.__file__).resolve().is_relative_to(source.resolve())
    artifacts = attempt / 'artifacts'
    result = json.loads((artifacts / 'results.json').read_text())
    assert result['protocol_id'] == 'LCR-P-HARD-001' and not result['smoke']
    assert result['complete_declared_seeds'] and not result['publication_eligible']
    settings = result['settings']
    assert [r['seed'] for r in result['per_seed_runs']] == settings['seeds']
    prepared = ROOT / 'resources/corpora' / f"{result['dataset']}_causal_v2.npz"
    assert sha256(prepared) == result['data']['derived_sha256']
    with np.load(prepared, allow_pickle=False) as saved:
        data = {key: saved[key] for key in ['sources', 'destinations', 'timestamps', 'features']}
    torch.set_num_threads(1)
    records = []
    for run in result['per_seed_runs']:
        seed = run['seed']
        folder = artifacts / f'seed{seed}'
        with np.load(folder / 'candidate_identity.npz', allow_pickle=False) as z:
            candidates = {key: z[key] for key in z.files}
        ids, edges, phi, elapsed = rebuild_inputs(data, candidates, settings['max_neighbors'], settings['history_decay'])
        test = candidates['split'] == 2
        np.testing.assert_array_equal(ids, candidates['event_ids'][test])
        np.testing.assert_allclose(phi, candidates['pair_features'][test], rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(elapsed, candidates['elapsed_times'][test])
        train_phi = candidates['pair_features'][candidates['split'] == 0].reshape(-1, 6)
        assert list(run['models']) == settings['models']
        for kind in (['scalar_decay'] if scalar_unclipped else settings['models']):
            info = run['models'][kind]
            checkpoint = folder / f'{kind}_seed{seed}.pt'
            prediction = folder / f'{kind}_seed{seed}_predictions.npz'
            assert sha256(checkpoint) == info['checkpoint_sha256']
            assert sha256(prediction) == info['predictions_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert saved['kind'] == kind and saved['seed'] == seed
            assert saved['settings'] == settings
            best = max(info['validation_history'], key=lambda e: e['validation']['ap'])
            assert saved['selected_epoch'] == info['selected_epoch'] == best['epoch']
            np.testing.assert_allclose(saved['phi_mean'], train_phi.mean(0))
            np.testing.assert_allclose(saved['phi_std'], train_phi.std(0).clip(min=1e-5))
            model = HistoryControl(kind, data['features'].shape[1], settings, saved['phi_mean'], saved['phi_std'])
            assert sum(p.numel() for p in model.parameters()) == info['parameter_count']
            model.load_state_dict(saved['model'], strict=True)
            model.eval()
            positive, negative, log_positive, log_negative = [], [], [], []
            with torch.inference_mode():
                for start in range(0, len(ids), settings['batch_size']):
                    batch = slice(start, start + settings['batch_size'])
                    indices = edges[batch]
                    valid = indices >= 0
                    safe = np.maximum(indices, 0)
                    ages = np.where(valid, data['timestamps'][ids[batch], None, None] - data['timestamps'][safe], 0).astype(np.float32)
                    features = data['features'][safe] * valid[..., None]
                    inputs = [features, ages, valid, phi[batch], elapsed[batch]]
                    p, q = model(*(torch.from_numpy(x.copy()) for x in inputs))
                    positive.append(p['prob'].numpy())
                    negative.append(q['prob'].numpy())
                    if scalar_unclipped:
                        for slot, value, target in [(0, p, log_positive), (1, q, log_negative)]:
                            target.append(scalar_log_score(value['base_logits'], model.raw_rate,
                                          torch.from_numpy(elapsed[batch, slot].copy()),
                                          model.time_unit).numpy())
            p, q = np.concatenate(positive), np.concatenate(negative)
            with np.load(prediction, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['event_ids'], ids)
                np.testing.assert_allclose(p, z['positive'], rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(q, z['negative'], rtol=2e-4, atol=2e-5)
                error = float(max(np.max(np.abs(p - z['positive'])), np.max(np.abs(q - z['negative']))))
            records.append(dict(seed=seed, model=kind, query_count=len(ids), max_absolute_probability_error=error,
                                checkpoint_sha256=sha256(checkpoint), prediction_sha256=sha256(prediction)))
            if scalar_unclipped:
                from sklearn.metrics import average_precision_score, roc_auc_score
                lp, lq = np.concatenate(log_positive), np.concatenate(log_negative)
                np.testing.assert_allclose(np.exp(lp).clip(1e-6, 1-1e-6), p, rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(np.exp(lq).clip(1e-6, 1-1e-6), q, rtol=2e-4, atol=2e-5)
                assert np.isfinite(lp).all() and np.isfinite(lq).all()
                with np.load(prediction, allow_pickle=False) as z:
                    retained_p, retained_q = z['positive'].copy(), z['negative'].copy()
                tied = retained_p == retained_q
                labels = np.r_[np.ones(len(ids)), np.zeros(len(ids))]
                log_scores = np.r_[lp, lq]
                detail = dict(
                    original_clipped_pair_accuracy=float(np.mean((retained_p > retained_q)+.5*tied)),
                    replayed_clipped_pair_accuracy=float(np.mean((p > q)+.5*(p == q))),
                    stable_log_pair_accuracy=float(np.mean((lp > lq)+.5*(lp == lq))),
                    stable_log_pair_ties=float(np.mean(lp == lq)),
                    original_tied_queries=int(tied.sum()),
                    stable_log_accuracy_on_original_ties=(float(np.mean((lp[tied] > lq[tied])+.5*(lp[tied] == lq[tied]))) if tied.any() else None),
                    stable_log_pooled_ap=float(average_precision_score(labels, log_scores)),
                    stable_log_pooled_auc=float(roc_auc_score(labels, log_scores)),
                    preclamp_probability_brier=float(np.mean((np.exp(log_scores)-labels)**2)))
                raw_file = output/f'scalar_seed{seed}_log_scores.npz'
                np.savez_compressed(raw_file, event_ids=ids, positive_log_score=lp, negative_log_score=lq,
                                    original_positive=retained_p, original_negative=retained_q)
                records[-1].update(unclipped_diagnostic=detail, log_scores_sha256=sha256(raw_file))
            print(f"Verified {result['dataset']} seed={seed} model={kind} queries={len(ids)}", flush=True)
    report = dict(status='PASS', dataset=result['dataset'], source_bundle_sha256=sha256(attempt / 'source.zip'),
        independent_full_test_history_reconstruction=True, full_retained_test_checkpoint_inference=True,
        full_training_reproduction=False, fresh_dependency_environment=False, new_model_fit=False,
        publication_eligible=False, rtol=2e-4, atol=2e-5, elapsed_seconds=time.monotonic() - started,
        local_environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device='cpu'),
        original_environment=result['environment'], per_model=records, auditor_sha256=sha256(Path(__file__)),
        limitations='Prepared bytes and retained checkpoints reused. Archived model inference and independent input reconstruction; no optimization replay, fresh evaluation cohort or scientific admission.')
    if scalar_unclipped:
        report.update(protocol_id='LCR-P-POOL-DIAG-002', posthoc=True, checked_models=['scalar_decay'],
                      stable_log_float64_diagnostic=True, original_primary_endpoint_replaced=False,
                      diagnostic_protocol_sha256=sha256(ROOT/'protocols/pooling_unclipped_diagnostic_v1.toml'))
        report['limitations'] += ' Stable log scores evaluate learned logits in float64; unavailable original GPU pre-clamp floats are not reproduced.'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scalar-unclipped', action='store_true')
    args = parser.parse_args()
    replay(args.attempt, args.output, scalar_unclipped=args.scalar_unclipped)
