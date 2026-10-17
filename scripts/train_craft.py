"""Thin validation-only author-model entry point; no official test is loaded."""
from pathlib import Path
import argparse
import sys
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import sha256
from lifecycle_readout.craft_training import fit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    if args.dataset != 'ML-20M' or (args.seeds is not None and args.seeds != settings['seeds']):
        parser.error('Undeclared dataset or seed set')
    if args.device != 'cuda' or not torch.cuda.is_available():
        parser.error('Full author baseline requires bounded Colab CUDA')
    for item in settings['input_files']:
        if sha256(ROOT / item['path']) != item['sha256']:
            parser.error('Input identity mismatch: ' + item['path'])
    with np.load(ROOT / settings['prepared_path']) as stored:
        data = {k: stored[k] for k in stored.files}
    with np.load(ROOT / settings['candidates_path']) as stored:
        candidates = {k: stored[k] for k in stored.files}
    fit(data, candidates, settings, ROOT / settings['author_path'], args.output_dir, args.device)
