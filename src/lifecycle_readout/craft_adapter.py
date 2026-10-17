"""Strict-past development adapter for a hash-pinned CRAFT author model.

No author loader, candidate sampler, training runner or test data is imported.
Author bytes remain unchanged; only their internal import namespace is relocated.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


CRAFT_COMMIT = '81590a78dddf1f0f589bc710fec2effd83f6cd5a'


def load_author_class(folder):
    folder = Path(folder)
    identity = json.loads((folder / 'source_identity.json').read_text())
    if identity['commit'] != CRAFT_COMMIT:
        raise ValueError('Unreviewed CRAFT source revision')
    sources = {}
    for name in ['models/modules.py', 'models/CRAFT.py', 'LICENSE']:
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity['files'][name]['sha256']:
            raise ValueError('Author source checksum mismatch: ' + name)
        sources[name] = raw
    # The source record itself must be pinned by the execution input manifest.
    # A private namespace avoids replacing another package named "models".
    prefix = 'lifecycle_readout._craft_' + CRAFT_COMMIT
    module_name = prefix + '_modules'
    dependencies = importlib.util.module_from_spec(importlib.util.spec_from_loader(module_name, loader=None))
    exec(compile(sources['models/modules.py'], str(folder / 'models/modules.py'), 'exec'), dependencies.__dict__)
    sys.modules[module_name] = dependencies
    parsed = ast.parse(sources['models/CRAFT.py'])
    relocated = 0
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'models.modules':
            node.module = module_name
            relocated += 1
    if relocated != 2:
        raise ValueError('Author import layout changed')
    model_module = importlib.util.module_from_spec(importlib.util.spec_from_loader(prefix, loader=None))
    exec(compile(parsed, str(folder / 'models/CRAFT.py'), 'exec'), model_module.__dict__)
    sys.modules[prefix] = model_module
    return model_module.CRAFT


class _GroupedTimes:
    def __init__(self, ids, timestamps, values=None):
        order = np.argsort(ids, kind='stable')
        self.times = timestamps[order]
        self.values = values[order] if values is not None else None
        self.offsets = np.r_[0, np.bincount(ids).cumsum()]

    def before(self, ids, times):
        """Absolute grouped-array boundaries; no same-time event is eligible."""
        result = np.zeros(len(ids), dtype=np.int64)
        order = np.argsort(ids, kind='stable')
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(ids)]
        for begin, finish in zip(starts, stops):
            if begin == finish:
                continue
            node = sorted_ids[begin]
            rows = order[begin:finish]
            if node < 0:
                raise ValueError('Negative node ID')
            if node + 1 >= len(self.offsets):
                continue
            start, stop = self.offsets[node:node + 2]
            result[rows] = start + np.searchsorted(self.times[start:stop], times[rows], side='left')
        return result


class StrictPastCraftHistory:
    """Recent source destinations plus destination-global last visit time.

    An immutable supplied train/validation stream is safe to index only because
    every query truncates strictly by its own timestamp. Node IDs reserve zero
    for padding. Past timestamp ties retain the supplied event order.
    """
    def __init__(self, data):
        u, v, t, split = (np.asarray(data[k]) for k in ['src', 'dst', 'time', 'split'])
        if not len(u) or any(x.shape != u.shape for x in [v, t, split]):
            raise ValueError('Nonempty aligned event arrays required')
        if u.ndim != 1 or not np.isfinite(t).all() or (np.diff(t) < 0).any():
            raise ValueError('Finite chronological one-dimensional events required')
        if not np.isin(split, [0, 1]).all() or (np.diff(split) < 0).any():
            raise ValueError('Only ordered train/validation inputs permitted')
        if min(u.min(), v.min()) <= 0:
            raise ValueError('Positive IDs required; zero is padding')
        self.source = _GroupedTimes(u, t, v)
        self.destination = _GroupedTimes(v, t)

    def batch(self, sources, times, candidates, length, device='cpu'):
        sources, times, candidates = np.asarray(sources), np.asarray(times), np.asarray(candidates)
        if length <= 0 or sources.ndim != 1 or times.shape != sources.shape:
            raise ValueError('Invalid query shape or history length')
        if candidates.ndim != 2 or candidates.shape[0] != len(sources):
            raise ValueError('Candidates must have shape [queries, candidates]')
        if not np.isfinite(times).all() or (candidates <= 0).any():
            raise ValueError('Finite query times and positive candidate IDs required')
        end = self.source.before(sources, times)
        seq = np.zeros((len(sources), length), dtype=np.int64)
        seq_times = np.zeros_like(seq, dtype=np.float64)
        counts = np.zeros(len(sources), dtype=np.int64)
        for row, src in enumerate(sources):
            if src + 1 >= len(self.source.offsets):
                continue
            begin = max(int(self.source.offsets[src]), int(end[row]) - length)
            count = int(end[row]) - begin
            counts[row] = count
            seq[row, :count] = self.source.values[begin:end[row]]
            seq_times[row, :count] = self.source.times[begin:end[row]]
        flat = candidates.reshape(-1)
        flat_times = np.repeat(times, candidates.shape[1])
        ends = self.destination.before(flat, flat_times)
        last = np.full(len(flat), -100000., dtype=np.float64)
        known = flat + 1 < len(self.destination.offsets)
        valid = np.zeros(len(flat), dtype=bool)
        valid[known] = ends[known] > self.destination.offsets[flat[known]]
        last[valid] = self.destination.times[ends[valid] - 1]
        # Keep absolute timestamps in float64 until after subtraction inside
        # CRAFT. Casting UNIX seconds to float32 here can erase short intervals.
        arrays = [seq, counts, seq_times, times.copy(), candidates.copy(), last.reshape(candidates.shape)]
        return tuple(torch.from_numpy(x).to(device) for x in arrays)


def make_author_model(folder, settings, item_count, device='cpu'):
    cls = load_author_class(folder)
    model = cls(n_layers=settings['layers'], n_heads=settings['heads'],
        hidden_size=settings['embedding_size'], hidden_dropout_prob=settings['hidden_dropout'],
        attn_dropout_prob=settings['attention_dropout'], hidden_act='gelu', layer_norm_eps=1e-12,
        initializer_range=.02, n_nodes=item_count, max_seq_length=settings['history_length'],
        device=device, loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
        output_cat_time_intervals=True, output_cat_repeat_times=False, num_output_layer=2,
        emb_dropout_prob=settings['embedding_dropout'], skip_connection=False).to(device)
    model.set_min_idx(src_min_idx=1, dst_min_idx=1)
    return model
