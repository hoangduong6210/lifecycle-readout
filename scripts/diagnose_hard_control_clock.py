"""Cheap train/validation diagnosis of preserved runs; never scores test data."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def diagnose(attempt: Path):
    artifacts = attempt / 'artifacts'
    source = artifacts / 'source_manifest.json'
    output = {'attempt': attempt.name, 'scope': 'training clock and recorded validation only',
              'causal_attribution': False, 'test_metrics_used': False, 'per_seed': [],
              'input_sha256': {str(source.relative_to(attempt)): hashlib.sha256(source.read_bytes()).hexdigest()}}
    for folder in sorted(artifacts.glob('seed*')):
        path = folder / 'candidate_identity.npz'
        output['input_sha256'][str(path.relative_to(attempt))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as candidates:
            times = candidates['elapsed_times'][candidates['split'] == 0].reshape(-1)
        if not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError('Training elapsed values must be finite and nonnegative')
        # Reference the declared v1 initialization, not a fitted parameter.
        initial_rate, time_unit = .1, 1000.
        seed = {'seed': int(folder.name.removeprefix('seed')),
                'train_candidate_count': len(times),
                'train_elapsed_quantiles_seconds': dict(zip(['min', 'median', 'q90', 'q99', 'max'],
                    map(float, np.quantile(times, [0., .5, .9, .99, 1.])))),
                'reference_scalar_initial_rate': initial_rate,
                'reference_clock_seconds': time_unit,
                'median_initial_scalar_multiplier': float(np.median(np.exp(-initial_rate * times / time_unit))),
                'validation': {}}
        for kind in ['base', 'base_stats', 'scalar_decay', 'mlp_gate', 'ordered_gate']:
            path = folder / f'{kind}_seed{seed["seed"]}.json'
            record = json.loads(path.read_text())
            output['input_sha256'][str(path.relative_to(attempt))] = hashlib.sha256(path.read_bytes()).hexdigest()
            chosen = next(e for e in record['validation_history'] if e['epoch'] == record['selected_epoch'])
            seed['validation'][kind] = {'selected_epoch': chosen['epoch'],
                'last_recorded_epoch': record['validation_history'][-1]['epoch'],
                'ap': chosen['validation']['ap'], 'brier': chosen['validation']['brier']}
        output['per_seed'].append(seed)
    if not output['per_seed']:
        raise ValueError('No retained seed candidates found')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite a diagnostic; use a fresh output path')
    result = diagnose(args.attempt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Training/validation diagnostic saved; no model training or test scoring.')
