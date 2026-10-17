"""Development-only streaming count controls with strictly past updates.

These are explicit simple baselines, not author Base3/EdgeBank reproductions
or a proposed novel method. Validation candidates are generated once and shared.
"""
from pathlib import Path
import json
import platform
import time

import numpy as np

from .benchmark import save_json, sha256
from .ranking import reciprocal_ranks


def training_predecessors(source, destination, timestamps):
    """Smallest destination in the source's previous distinct-time event set."""
    order = np.argsort(source, kind='stable')
    u, v, t = source[order], destination[order], timestamps[order]
    starts = np.r_[True, (u[1:] != u[:-1]) | (t[1:] != t[:-1])]
    idx = np.flatnonzero(starts)
    minimum = np.minimum.reduceat(v, idx)
    previous = np.r_[-1, minimum[:-1]]
    previous[np.r_[True, u[idx][1:] != u[idx][:-1]]] = -1
    sorted_prev = previous[np.cumsum(starts) - 1]
    result = np.empty(len(source), dtype=np.int64); result[order] = sorted_prev
    last = np.full(int(source.max()) + 1, -1, dtype=np.int64)
    final_groups = np.r_[u[idx][1:] != u[idx][:-1], True]
    last[u[idx][final_groups]] = minimum[final_groups]
    return result, last


def draw_candidates(rng, source, positives, catalog, count):
    """Uniform with replacement, rejecting every simultaneous true edge."""
    positive_keys = (source.astype(np.uint64) << np.uint64(32)) | positives.astype(np.uint64)
    positive_keys = np.unique(positive_keys)
    banned = {int(u): set(map(int, positives[source == u])) for u in np.unique(source)}
    if any(len(blocked) >= len(catalog) and all(int(v) in blocked for v in catalog)
           for blocked in banned.values()):
        raise ValueError('No available negative in the fixed training catalogue')
    values = rng.choice(catalog, (len(source), count), replace=True)
    while True:
        keys = (source[:, None].astype(np.uint64) << np.uint64(32)) | values.astype(np.uint64)
        location = np.searchsorted(positive_keys, keys)
        bad = (location < len(positive_keys)) & (positive_keys[np.minimum(location, len(positive_keys) - 1)] == keys)
        if not bad.any():
            return values
        values[bad] = rng.choice(catalog, int(bad.sum()), replace=True)


def evaluate_counts(data, settings, output):
    """Heavy full-dataset screens belong on Colab; no test input is accepted."""
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    u, v, t, split = (data[k] for k in ('src', 'dst', 'time', 'split'))
    if not np.isin(split, [0, 1]).all() or not (split == 0).any() or not (split == 1).any():
        raise ValueError('Only provided training and validation rows are allowed')
    if (np.diff(t) < 0).any() or (np.diff(split) < 0).any():
        raise ValueError('Chronological masks required')
    if min(u.min(), v.min()) < 0 or max(u.max(), v.max()) > np.iinfo(np.uint32).max:
        raise ValueError('Unsupported node IDs')
    nsrc, ndst = int(u.max()) + 1, int(v.max()) + 1
    required_bytes = nsrc * ndst + ndst * ndst * np.dtype(np.int32).itemsize
    if required_bytes > settings['max_state_bytes']:
        raise ValueError('Dense baseline state exceeds declared memory budget')
    train = split == 0; val = np.flatnonzero(split == 1)
    if t[train].max() >= t[val].min():
        raise ValueError('Train/validation boundary must be strictly separated')
    validation_times = t[val]
    previous, last = training_predecessors(u[train], v[train], t[train])
    last = np.pad(last, (0, nsrc - len(last)), constant_values=-1)
    seen = np.zeros((nsrc, ndst), dtype=bool)
    seen[u[train], v[train]] = True
    popularity = np.bincount(v[train], minlength=ndst).astype(np.int64)
    catalog = np.flatnonzero(popularity)
    transitions = np.zeros((ndst, ndst), dtype=np.int32)
    valid = previous >= 0
    np.add.at(transitions, (previous[valid], v[train][valid]), 1)
    del previous
    rng = np.random.RandomState(settings['candidate_seed'])
    negatives = np.empty((len(val), settings['negative_count']), dtype=np.int64)
    scores = {kind: np.empty((len(val), settings['negative_count'] + 1), dtype=np.float64)
              for kind in settings['models']}
    if list(scores) != ['edge_memory', 'popularity', 'pooled_transition']:
        raise ValueError('Unexpected baseline model list')
    observed = 0
    start = 0
    while start < len(val):
        stop = int(np.searchsorted(validation_times, validation_times[start], side='right'))
        rows = val[start:stop]; query_u, query_v = u[rows], v[rows]
        neg = draw_candidates(rng, query_u, query_v, catalog, settings['negative_count'])
        candidates = np.c_[query_v, neg]; negatives[start:stop] = neg
        scores['edge_memory'][start:stop] = seen[query_u[:, None], candidates]
        pop = (popularity[candidates] + settings['popularity_pseudocount']) / (
            popularity.sum() + settings['popularity_pseudocount'] * len(catalog))
        scores['popularity'][start:stop] = pop
        contexts = last[query_u]
        supported = contexts >= 0
        transition = np.zeros_like(pop)
        transition[supported] = transitions[contexts[supported, None], candidates[supported]]
        # Query-wise denominator is constant across candidates and omitted for
        # ranking. Unknown contexts use the same popularity backoff.
        scores['pooled_transition'][start:stop] = transition + settings['transition_prior_mass'] * pop
        # Observe all events at this timestamp only after all their predictions.
        seen[query_u, query_v] = True
        np.add.at(popularity, query_v, 1)
        np.add.at(transitions, (contexts[supported], query_v[supported]), 1)
        order = np.argsort(query_u, kind='stable')
        group_start = np.r_[True, query_u[order][1:] != query_u[order][:-1]]
        group_idx = np.flatnonzero(group_start)
        last[query_u[order][group_idx]] = np.minimum.reduceat(query_v[order], group_idx)
        observed += len(rows)
        if observed // 50000 > (observed - len(rows)) // 50000:
            print(f'Scored validation queries: {observed}/{len(val)}', flush=True)
        start = stop
    identity = output / 'validation_candidates.npz'
    np.savez_compressed(identity, event_ids=data['event_ids'][val], sources=u[val],
                        positive_destinations=v[val], timestamps=t[val], negative_destinations=negatives)
    result = {'evaluation_split': 'validation', 'test_scored': False, 'publication_eligible': False,
              'query_count': len(val), 'settings': settings, 'candidate_sha256': sha256(identity),
              'state_bytes': required_bytes, 'models': {},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__}}
    for kind, score in scores.items():
        rr = reciprocal_ranks(score[:, 0], score[:, 1:])
        path = output / f'{kind}_predictions.npz'
        np.savez_compressed(path, positive=score[:, 0], negative=score[:, 1:], reciprocal_rank=rr)
        result['models'][kind] = {'mrr': float(rr.mean()), 'query_count': len(rr),
                                  'prediction_sha256': sha256(path)}
    result['elapsed_seconds'] = time.monotonic() - started
    save_json(output / 'results.json', result)
    return result
