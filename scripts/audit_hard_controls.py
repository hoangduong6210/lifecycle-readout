"""Recompute endpoint values and independently replay hard-negative eligibility."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import zipfile
import tomllib
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_replay(data, candidates, unit):
    sources, dests, times = (data[k] for k in ('sources', 'destinations', 'timestamps'))
    event_ids = candidates['event_ids']
    row_map = {int(i): row for row, i in enumerate(event_ids)}
    skipped = set(candidates['skipped_no_hard_support_ids'].tolist())
    excluded = set(candidates['excluded_training_ids'].tolist())
    assert len(row_map) == len(event_ids)
    assert not (set(row_map) & skipped or set(row_map) & excluded or skipped & excluded)
    assert len(row_map) + len(skipped) + len(excluded) == len(times)
    unseen = set(candidates['unseen_nodes'].tolist())
    past = defaultdict(dict)
    selected, no_support = 0, 0
    start = 0
    while start < len(times):
        t = float(times[start]);stop = int(np.searchsorted(times, t, side='right'))
        allowed = [i for i in range(start, stop) if i not in excluded]
        positives = defaultdict(set)
        for i in allowed:
            positives[int(sources[i])].add(int(dests[i]))
        for i in allowed:
            u, v = int(sources[i]), int(dests[i])
            previous = past[u].get(v)
            pool = []
            if previous:
                last, count = previous
                count_bin = int(math.log2(count))
                time_bin = int(math.log2(1 + (t - last) / unit))
                pool = [d for d, (last_q, count_q) in past[u].items()
                        if d not in positives[u] and last_q < t
                        and int(math.log2(count_q)) == count_bin
                        and int(math.log2(1 + (t - last_q) / unit)) == time_bin]
            if not pool:
                assert i in skipped, ('missing exclusion', i)
                no_support += 1
                continue
            assert i in row_map and i not in skipped, ('eligible event was omitted', i)
            row = row_map[i]
            negative = int(candidates['negative_destinations'][row])
            assert negative in pool, ('unmatched or future negative', i)
            for k, d in enumerate([v, negative]):
                last, count = past[u][d]
                np.testing.assert_allclose(candidates['elapsed_times'][row, k], t - last, rtol=2e-6)
                np.testing.assert_allclose(candidates['pair_features'][row, k, 2], np.log1p(count), rtol=2e-6)
            if candidates['split'][row] == 0:
                assert u not in unseen and v not in unseen and negative not in unseen
            selected += 1
        # Observe all real events, even when their supervision lacked support.
        for i in allowed:
            u, v = int(sources[i]), int(dests[i])
            count = past[u].get(v, (0., 0))[1]
            past[u][v] = (t, count + 1)
        start = stop
    assert selected == len(event_ids) and no_support == len(skipped)
    return dict(matched_rows=selected, unsupported_rows=no_support, excluded_training_rows=len(excluded))


def recompute(pos, neg):
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    if not len(pos):
        return dict(ap=None, auc=None, brier=None)
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    return dict(ap=float(average_precision_score(labels, scores)), auc=float(roc_auc_score(labels, scores)),
                brier=float(np.mean((scores - labels) ** 2)))


def audit(attempt, *, study="hard-v1"):
    if study == "hard-v1":
        expected_protocol, expected_seeds = "LCR-P-HARD-001", [1, 7, 42]
        expected_models = ["base", "base_stats", "scalar_decay", "mlp_gate", "ordered_gate"]
        evaluation_split = "test"
    elif study == "q1-dev-v1":
        expected_protocol, expected_seeds = "LCR-P-Q1-DEV-001", [101, 202, 303]
        expected_models = ["base_stats", "scalar_decay", "ordered_gate", "residual_scalar", "residual_mixture", "mlp_residual"]
        evaluation_split = "validation"
    elif study == "cadence-v1":
        expected_protocol, expected_seeds = "LCR-P-CADENCE-DEV-001", [101, 202, 303]
        preliminary = json.loads((attempt / 'artifacts/results.json').read_text())
        name = preliminary['settings']['config_path']
        allowed = {f'configs/pair_cadence_{arm}_v1.toml': arm for arm in ['original', 'features', 'relative']}
        assert name in allowed
        arm = allowed[name]
        expected_models = (["residual_scalar", "residual_mixture"] if arm == 'relative'
                           else ["base_stats", "residual_scalar", "residual_mixture", "mlp_residual"])
        evaluation_split = "validation"
    else:
        raise ValueError("Unknown study")
    endpoint_key = "all_test" if evaluation_split == "test" else "all_validation"
    job = json.loads((attempt / 'job.json').read_text())
    assert job['cleanup_verified']
    if study == 'cadence-v1':
        assert job['status'] == 'completed'
        assert digest(attempt / 'artifacts.tar.gz') == job['archive_sha256']
    if job['status'] != 'completed':
        # Preserve transport failure. Recovered completed training is auditable
        # only with an explicit record pinned to the unmodified failed job.
        assert study == 'q1-dev-v1' and job['status'] == 'failed'
        recovery = json.loads((attempt / 'transport_reconciliation.json').read_text())
        assert recovery['original_job_sha256'] == digest(attempt / 'job.json')
        assert recovery['training_outcome'] == 'completed'
        assert recovery['transport_outcome'] == 'failed_client_interrupted'
        assert job['artifact_checksums_verified'] and job['remote_outcome_verified']
        assert digest(attempt / 'artifacts.tar.gz') == job['archive_sha256']
    artifacts = attempt / 'artifacts'
    # Check every retrieved artifact, including checkpoints and predictions.
    for line in (artifacts / 'checksums.sha256').read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        path = (artifacts / rel).resolve()
        assert path.is_relative_to(artifacts.resolve()) and path.is_file()
        assert digest(path) == expected, rel
    results = json.loads((artifacts / 'results.json').read_text())
    assert results['protocol_id'] == expected_protocol and not results['smoke']
    if evaluation_split == 'validation':
        assert results['evaluation_split'] == 'validation' and results['test_scored'] is False
    assert [r['seed'] for r in results['per_seed_runs']] == expected_seeds
    assert results['complete_declared_seeds'] and not results['publication_eligible']
    assert digest(attempt / 'source.zip') == job['source_bundle_sha256']
    with zipfile.ZipFile(attempt / 'source.zip') as bundle:
        source_manifest = json.loads(bundle.read('source_manifest.json'))
        for rel, expected in source_manifest.items():
            assert hashlib.sha256(bundle.read(rel)).hexdigest() == expected, rel
        assert source_manifest[results['settings']['protocol']] == job['protocol_sha256']
        assert source_manifest[results['settings']['config_path']] == job['config_sha256']
        for rel, expected in results['source_files_sha256'].items():
            assert source_manifest[rel] == expected
        if study == 'cadence-v1':
            settings = results['settings']
            assert settings == tomllib.loads(bundle.read(settings['config_path']).decode())
            assert digest(ROOT / settings['config_path']) == job['config_sha256']
            cadence_protocol = tomllib.loads(bundle.read(settings['protocol']).decode())
            assert digest(ROOT / settings['protocol']) == job['protocol_sha256']
            assert cadence_protocol['seeds'] == expected_seeds == settings['seeds']
            assert settings['models'] == expected_models
            assert settings['pair_feature_dim'] == 9
            assert settings['pair_clock_mode'] == ('relative' if arm == 'relative' else 'absolute')
            assert settings.get('mask_pair_clock_features', False) == (arm == 'original')
            assert cadence_protocol['test_scoring_permitted'] is False
            for rel in ['src/lifecycle_readout/hard_controls.py',
                        'src/lifecycle_readout/modeling/monotone_residual.py',
                        'src/lifecycle_readout/modeling/temporal_mixer.py',
                        'src/lifecycle_readout/modeling/latent_lifecycle.py']:
                assert digest(ROOT / rel) == source_manifest[rel], 'Use archived model source after drift'
    dataset = results['dataset']
    prepared = ROOT / 'resources/corpora' / f'{dataset}_causal_v2.npz'
    assert digest(prepared) == results['data']['derived_sha256']
    with np.load(prepared, allow_pickle=False) as stored:
        data = {k: stored[k] for k in stored.files}
    output = dict(dataset=dataset, protocol_id=results['protocol_id'], attempt=str(attempt.relative_to(ROOT)),
                  source_bundle_sha256=job['source_bundle_sha256'], cleanup_verified=True,
                  checksums_verified=True, per_seed=[], publication_eligible=False,
                  evaluation_split=evaluation_split, test_scored=evaluation_split == 'test')
    for run in results['per_seed_runs']:
        assert list(run['models']) == expected_models
        seed_path = artifacts / f"seed{run['seed']}"
        with np.load(seed_path / 'candidate_identity.npz', allow_pickle=False) as z:
            candidates = {k: z[k] for k in z.files}
        replay = independent_replay(data, candidates, results['settings']['hard_time_unit'])
        if study == 'cadence-v1':
            from audit_pair_clock_inputs import verify_saved_clock
            replay['cadence'] = verify_saved_clock(data, candidates, seed_path, results['settings'], cadence_protocol)
        test = candidates['split'] == (2 if evaluation_split == 'test' else 1)
        model_metrics = {}
        for kind in expected_models:
            report = run['models'][kind]
            if evaluation_split == 'validation':
                assert 'all_test' not in report and report['evaluation_split'] == 'validation'
                train_times = candidates['elapsed_times'][candidates['split'] == 0].reshape(-1)
                if study == 'cadence-v1' and results['settings']['pair_clock_mode'] == 'relative':
                    assert report['effective_decay_time_unit'] is None and report['decay_clock_mode'] == 'relative'
                else:
                    np.testing.assert_allclose(report['effective_decay_time_unit'], np.median(train_times[train_times > 0]))
            pfile = seed_path / f"{kind}_seed{run['seed']}_predictions.npz"
            assert digest(pfile) == report['predictions_sha256']
            checkpoint = seed_path / f"{kind}_seed{run['seed']}.pt"
            assert digest(checkpoint) == report['checkpoint_sha256']
            if evaluation_split == 'validation':
                import torch
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert saved['kind'] == kind and saved['seed'] == run['seed']
                assert saved['settings']['evaluation_split'] == 'validation'
                train_phi = candidates['pair_features'][candidates['split'] == 0].reshape(-1, results['settings'].get('pair_feature_dim', 6))
                np.testing.assert_allclose(saved['phi_mean'], train_phi.mean(0))
                np.testing.assert_allclose(saved['phi_std'], train_phi.std(0).clip(min=1e-5))
                if kind.startswith('residual_'):
                    assert (saved['model']['residual.amplitude'] >= 0).all()
                if study == 'cadence-v1':
                    effective = dict(results['settings'], decay_time_unit=float(np.median(train_times[train_times > 0])))
                    assert saved['settings'] == effective
                    assert saved['selected_epoch'] == report['selected_epoch']
                    assert [e['epoch'] for e in report['validation_history']] == list(range(1, effective['epochs']+1))
                    from lifecycle_readout.hard_controls import HistoryControl
                    model = HistoryControl(kind, data['features'].shape[1], effective, saved['phi_mean'], saved['phi_std'])
                    model.load_state_dict(saved['model'], strict=True)
                    assert sum(p.numel() for p in model.parameters()) == report['parameter_count']
                    assert all(torch.isfinite(v).all() for v in saved['model'].values())
            with np.load(pfile, allow_pickle=False) as p:
                if study == 'cadence-v1':
                    assert p['positive'].shape == p['negative'].shape == (int(test.sum()),)
                    assert report[endpoint_key]['positive_count'] == report[endpoint_key]['negative_count'] == int(test.sum())
                for key in ['event_ids', 'negative_destinations', 'pair_features', 'elapsed_times', 'inductive']:
                    np.testing.assert_array_equal(p[key], candidates[key][test])
                recalculated = recompute(p['positive'], p['negative'])
                for k, v in recalculated.items():
                    np.testing.assert_allclose(v, report[endpoint_key][k], rtol=1e-6, atol=1e-8)
                masks = {'both_silence_ge_1000': (p['elapsed_times'] >= 1000).all(1),
                         'inductive': p['inductive'], 'transductive': ~p['inductive']}
                for name, mask in masks.items():
                    actual = recompute(p['positive'][mask], p['negative'][mask])
                    for k, v in actual.items():
                        if v is None:
                            assert report['strata'][name][k] is None
                        else:
                            np.testing.assert_allclose(v, report['strata'][name][k], rtol=1e-6, atol=1e-8)
            best = max(report['validation_history'], key=lambda x: x['validation']['ap'])
            assert report['selected_epoch'] == best['epoch']
            model_metrics[kind] = recalculated
        output['per_seed'].append(dict(seed=run['seed'], replay=replay, models=model_metrics))
    for kind in expected_models:
        for metric in ['ap', 'auc', 'brier']:
            values = [r['models'][kind][metric] for r in output['per_seed']]
            np.testing.assert_allclose(np.mean(values), results['aggregate'][kind][metric]['mean'], rtol=1e-6)
            np.testing.assert_allclose(np.std(values, ddof=1), results['aggregate'][kind][metric]['std_ddof1'], rtol=1e-5, atol=1e-7)
    output['all_checks_pass'] = True
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=['hard-v1', 'q1-dev-v1', 'cadence-v1'], default='hard-v1')
    args = parser.parse_args()
    output = audit(args.attempt.resolve(), study=args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'per_seed'}, indent=2))
