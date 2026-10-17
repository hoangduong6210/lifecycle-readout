"""Audit cached upstream identities and evaluator parity; no dataset scoring."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.ranking import reciprocal_ranks

PINNED_COMMIT = 'c1ee801ea4301c2f944f5b3da10cbc5310fa4f68'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    identity = json.loads((folder / 'upstream_identity.json').read_text())
    assert identity['commit'] == PINNED_COMMIT
    for rel, record in identity['files'].items():
        path = (folder / 'upstream' / rel).resolve()
        assert path.is_relative_to((folder / 'upstream').resolve())
        assert digest(path) == record['sha256']
        assert record['url'] == f'https://raw.githubusercontent.com/TGB-Seq/TGB-Seq/{PINNED_COMMIT}/{rel}'
    # This small evaluator was inspected before first execution; execute only
    # those exact pinned bytes rather than an arbitrary fetched module.
    evaluator = folder / 'upstream/tgb_seq/LinkPred/evaluator.py'
    assert digest(evaluator) == 'bd6986205c07c5fa3313717873eded687c9a2d1d6c63df681985864ca37ad076'
    spec = importlib.util.spec_from_file_location('pinned_tgbseq_evaluator', evaluator)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rng = np.random.RandomState(139)
    cases = {'all_ties': (np.zeros(3), np.zeros((3, 100))),
             'all_win': (np.ones(3), np.zeros((3, 100))),
             'all_loss': (np.zeros(3), np.ones((3, 100))),
             'integer_ties': (rng.randint(-2, 3, 500), rng.randint(-2, 3, (500, 100))),
             'continuous': (rng.normal(size=500), rng.normal(size=(500, 100)))}
    checks = []
    for name, (pos, neg) in cases.items():
        expected, actual = module.Evaluator().eval(pos, neg), reciprocal_ranks(pos, neg)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
        checks.append({'case': name, 'queries': len(pos), 'negative_candidates': neg.shape[1],
                       'max_absolute_error': float(np.max(np.abs(actual - expected)))})
    datasets = json.loads((folder / 'dataset_preflight.json').read_text())
    sizes = {}
    for dataset in datasets:
        name = dataset['dataset']
        metadata = folder / f'{name}_metadata.json'
        assert digest(metadata) == dataset['metadata_sha256']
        assert json.loads(metadata.read_text())['sha'] == dataset['revision']
        assert dataset['dataset_bytes_downloaded'] is False
        entries = json.loads((folder / f'{name}_tree.json').read_text())
        sizes[name] = {e['path']: e['size'] for e in entries if e['path'].endswith(('.csv', '.npy'))}
    return {'all_checks_pass': True, 'upstream_commit': PINNED_COMMIT,
            'upstream_evaluator_sha256': digest(evaluator),
            'adapter_sha256': digest(ROOT / 'src/lifecycle_readout/ranking.py'),
            'cases': checks, 'dataset_file_sizes_bytes': sizes,
            'real_dataset_scored': False, 'dataset_bytes_downloaded': False,
            'underlying_dataset_rights_closed': False, 'publication_eligible': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output path')
    result = audit(args.folder)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Pinned evaluator parity and metadata verified; no graph test scored.')
