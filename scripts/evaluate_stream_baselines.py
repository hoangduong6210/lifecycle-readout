"""Validation-only streaming count screen on a prepared official split."""
from pathlib import Path
import argparse
import sys
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.stream_baselines import evaluate_counts
from lifecycle_readout.benchmark import save_json, sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    if args.dataset not in settings['datasets']: parser.error('Undeclared dataset')
    if args.seeds and args.seeds != settings['seeds']: parser.error('Undeclared candidate seed')
    if not args.smoke and (args.device != 'cuda' or not torch.cuda.is_available()):
        parser.error('Full screen requires the bounded Colab runtime; no local heavy fallback')
    data_path = ROOT / settings['prepared_path']
    expected = next(item['sha256'] for item in settings['input_files'] if item['path'] == settings['prepared_path'])
    if sha256(data_path) != expected: parser.error('Prepared input checksum mismatch')
    with np.load(data_path) as stored: data = {k: stored[k] for k in stored.files}
    if args.smoke:
        rows = np.r_[np.flatnonzero(data['split'] == 0)[:2000], np.flatnonzero(data['split'] == 1)[:100]]
        data = {k: v[rows] for k, v in data.items()}
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    assert protocol['models'] == settings['models']
    result = evaluate_counts(data, settings, args.output_dir)
    result.update(protocol_id=protocol['protocol_id'], dataset=args.dataset, smoke=args.smoke,
        input_sha256=expected, complete_declared_seeds=True,
        per_seed_runs=[{'seed': settings['candidate_seed'], 'seed_role': 'validation_candidate_sampling',
                        'models': result['models']}])
    save_json(args.output_dir / 'results.json', result)
    (args.output_dir / 'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n'
        for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.name != 'checksums.sha256'))
