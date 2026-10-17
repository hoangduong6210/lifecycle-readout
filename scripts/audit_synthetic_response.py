"""Independently verify the saved synthetic diagnostic without model fitting."""
from pathlib import Path
import argparse
import json
import sys
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import sha256
from lifecycle_readout.synthetic_response import FlexibleResidual, tensors
from lifecycle_readout.modeling.monotone_residual import NestedMonotoneResidual


def independent_kl(predicted, truth):
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    p = np.exp(-np.logaddexp(0., -truth))
    q = np.exp(-np.logaddexp(0., -pred))
    kl = p * (np.logaddexp(0., -pred) - np.logaddexp(0., -truth))
    kl += (1. - p) * (np.logaddexp(0., pred) - np.logaddexp(0., truth))
    return {'excess_expected_bce': float(kl.mean()), 'probability_mse': float(np.mean((q - p) ** 2))}


def audit(folder):
    torch.set_num_threads(1)
    folder = folder.resolve()
    for line in (folder / 'checksums.sha256').read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        p = (folder / rel).resolve()
        assert p.is_relative_to(folder) and p.is_file() and sha256(p) == expected, rel
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    for rel, expected in manifest.items():
        assert sha256(folder / 'source' / rel) == expected, rel
    # Inference uses the inspected current implementation only if the source
    # matches the retained run; source drift requires the archived environment.
    for rel in ['src/lifecycle_readout/synthetic_response.py',
                'src/lifecycle_readout/modeling/monotone_residual.py']:
        assert sha256(ROOT / rel) == manifest[rel], 'Use the archived source: ' + rel
    results = json.loads((folder / 'results.json').read_text())
    config = tomllib.loads((folder / 'source/configs/q1_synthetic_v1.toml').read_text())
    protocol = tomllib.loads((folder / 'source' / config['protocol']).read_text())
    assert results['settings'] == config and results['protocol_id'] == protocol['protocol_id']
    assert not results['publication_eligible']
    expected_runs = [(seed, scenario) for seed in config['seeds'] for scenario in config['scenarios']]
    assert [(run['seed'], run['scenario']) for run in results['runs']] == expected_runs
    verified = 0
    for run in results['runs']:
        location = folder / f'{run["scenario"]}_seed{run["seed"]}'
        parts = {}
        for split in ('train', 'validation', 'test', 'tail'):
            with np.load(location / f'{split}_data.npz') as z:
                parts[split] = {k: z[k] for k in z.files}
            part = parts[split]; c = part['context'][:, 0].astype(np.float64); t = part['elapsed'].astype(np.float64)
            base = 1. + .4 * c
            scenario = run['scenario']
            if scenario == 'null': expected_truth = base
            elif scenario == 'homogeneous_decay': expected_truth = base - 1.8 * np.log1p(t / 1.5)
            elif scenario == 'contextual_decay': expected_truth = base - (1.2 + c) * np.log1p(t / (1.7 - 1.3 * c))
            elif scenario == 'increasing': expected_truth = base + .8 * np.log1p(t)
            else: raise AssertionError('Unknown scenario')
            np.testing.assert_allclose(part['base'], base, atol=1e-15)
            np.testing.assert_allclose(part['truth'], expected_truth, atol=1e-15)
            low, high = ((config['train_time_max'], config['tail_time_max']) if split == 'tail'
                         else (0., config['train_time_max']))
            assert ((t >= low) & (t <= high)).all()
            count = config['train_count'] if split == 'train' else config['validation_count'] if split == 'validation' else config['evaluation_count']
            assert len(t) == count and ((c >= -1) & (c <= 1)).all()
        assert list(run['models']) == ['base', *config['models']]
        for kind, record in run['models'].items():
            if kind != 'base':
                checkpoint = location / f'{kind}.pt'
                assert sha256(checkpoint) == record['checkpoint_sha256']
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert saved['kind'] == kind and saved['settings'] == config
                chosen = min(record['validation_history'], key=lambda x: x['excess_expected_bce'])
                assert record['selected_step'] == saved['selected_step'] == chosen['step']
                assert len(record['validation_history']) == config['steps']
                model = (FlexibleResidual(config['hidden_dim']) if kind == 'flexible' else
                         NestedMonotoneResidual(1, n_components=1 if kind == 'scalar' else config['components'],
                                                hidden_dim=config['hidden_dim'], rate_init=config['rate_init']))
                model.load_state_dict(saved['model']); model.eval()
                if kind != 'flexible': assert (model.amplitude >= 0).all()
            for split in ('validation', 'test', 'tail'):
                with np.load(location / f'{kind}_{split}.npz') as z: logits = z['logits']
                part = parts[split]
                if kind == 'base': np.testing.assert_array_equal(logits, part['base'])
                else:
                    with torch.no_grad(): rebuilt = model(*tensors(part))['logits'].numpy()
                    np.testing.assert_array_equal(logits, rebuilt)
                actual = independent_kl(logits, part['truth'])
                assert logits.shape == part['truth'].shape and np.isfinite(logits).all()
                for metric, value in actual.items():
                    np.testing.assert_allclose(value, record['endpoints'][split][metric], rtol=1e-7, atol=1e-14)
                if split == 'validation' and kind != 'base':
                    np.testing.assert_allclose(actual['excess_expected_bce'], chosen['excess_expected_bce'], atol=1e-14)
                verified += 1
    for scenario, models in results['aggregate'].items():
        for kind, record in models.items():
            values = [run['models'][kind]['endpoints']['test']['excess_expected_bce']
                      for run in results['runs'] if run['scenario'] == scenario]
            np.testing.assert_allclose(record['excess_expected_bce_mean'], np.mean(values), atol=1e-14)
            np.testing.assert_allclose(record['std_ddof1'], np.std(values, ddof=1), atol=1e-14)
    return {'protocol_id': protocol['protocol_id'], 'all_checks_pass': True,
            'results_sha256': sha256(folder / 'results.json'), 'saved_endpoint_arrays_verified': verified,
            'scenario_seed_runs': len(results['runs']), 'trained_models_verified': len(results['runs']) * len(config['models']),
            'publication_eligible': False, 'graph_data_evaluated': False, 'model_training_in_audit': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.attempt)
    if args.output.exists(): parser.error('Use a fresh audit output path')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
