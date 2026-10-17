"""Build candidate inputs from strictly earlier events, independent of labels.

Equal-timestamp events are queried together and only then enter history.
Frozen pre-event snapshots permit shuffled optimization without history leakage.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass
class CandidateStream:
    event_ids: np.ndarray
    negative_destinations: np.ndarray
    neighbor_edges: np.ndarray  # [event, source/positive/negative, neighbor]
    pair_features: np.ndarray   # [event, positive/negative, feature]
    elapsed_times: np.ndarray
    split: np.ndarray          # train=0, validation=1, test=2
    inductive: np.ndarray
    excluded_training_ids: np.ndarray
    skipped_no_negative_ids: np.ndarray
    unseen_nodes: np.ndarray
    train_end: float
    validation_end: float

    def digest(self) -> str:
        h = hashlib.sha256()
        for name in ("event_ids", "negative_destinations", "neighbor_edges",
                     "pair_features", "elapsed_times", "split", "inductive",
                     "excluded_training_ids", "skipped_no_negative_ids", "unseen_nodes"):
            a = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()


def validate_events(sources, destinations, timestamps, features) -> None:
    n = len(sources)
    if n < 10 or any(len(a) != n for a in (destinations, timestamps, features)):
        raise ValueError("Incomplete event arrays or insufficient chronological coverage")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Event features must be a finite matrix")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("Timestamps must be finite and nondecreasing")
    if np.min(sources) < 0 or np.min(destinations) < 0:
        raise ValueError("Node IDs must be nonnegative")
    if np.intersect1d(sources, destinations).size:
        raise ValueError("Source and destination node namespaces must be disjoint")


def build_candidate_stream(sources, destinations, timestamps, features, *, seed=42,
                           split=(0.70, 0.15, 0.15), unseen_ratio=0.10,
                           max_neighbors=20, history_decay=0.01, split_seed=42,
                           negative_sampling="past_uniform", hard_time_unit=1000.0) -> CandidateStream:
    validate_events(sources, destinations, timestamps, features)
    if len(split) != 3 or not math.isclose(sum(split), 1.0) or min(split) <= 0:
        raise ValueError("Three positive chronological proportions must sum to one")
    if not 0 <= unseen_ratio < 1 or max_neighbors < 1:
        raise ValueError("Invalid held-out ratio or neighbor budget")
    if negative_sampling not in ("past_uniform", "matched_historical") or hard_time_unit <= 0:
        raise ValueError("Invalid negative sampling contract")
    n = len(sources)
    train_end, validation_end = np.quantile(timestamps, [split[0], split[0] + split[1]])
    if train_end >= validation_end:
        raise ValueError("Tied timestamps do not permit distinct train/validation boundaries")
    test_nodes = np.unique(np.concatenate((sources[timestamps > validation_end],
                                          destinations[timestamps > validation_end])))
    rng = np.random.RandomState(seed)
    unseen = set(np.random.RandomState(split_seed).choice(
        test_nodes, int(len(test_nodes) * unseen_ratio), replace=False).tolist())
    neighbors = defaultdict(lambda: deque(maxlen=max_neighbors))
    source_pairs = defaultdict(dict)
    pairs = {}  # (u,v) -> (last timestamp, event count)
    observed_destinations, observed_set = [], set()
    ids, negatives, snapshots, phi_rows, dt_rows, splits, ind = [], [], [], [], [], [], []
    excluded, skipped = [], []

    def query(u, v, t, index):
        previous, count = pairs.get((int(u), int(v)), (None, 0))
        dt = 0.0 if previous is None else t - previous
        if previous is not None and dt <= 0:
            raise AssertionError("Candidate history must be strictly earlier")
        intensity = 0.1 if previous is None else 0.1 + 0.9 * math.exp(-history_decay * dt)
        return [intensity, math.log1p(dt), math.log1p(count),
                count / math.sqrt(max(1, index)),
                math.sin(2 * math.pi * (t % 86400) / 86400),
                math.cos(2 * math.pi * (t % 86400) / 86400)], dt

    start = 0
    while start < n:
        stop = int(np.searchsorted(timestamps, timestamps[start], side="right"))
        t = float(timestamps[start])
        phase = 0 if t <= train_end else 1 if t <= validation_end else 2
        allowed = [i for i in range(start, stop)
                   if phase != 0 or (int(sources[i]) not in unseen and int(destinations[i]) not in unseen)]
        allowed_set = set(allowed)
        excluded.extend(i for i in range(start, stop) if i not in allowed_set)
        current_positive = defaultdict(set)
        for i in allowed:
            current_positive[int(sources[i])].add(int(destinations[i]))
        for i in allowed:
            u, v = int(sources[i]), int(destinations[i])
            forbidden = current_positive[u]
            if negative_sampling == "matched_historical":
                previous, count = pairs.get((u, v), (None, 0))
                if not count:
                    skipped.append(i)
                    continue
                count_bin = int(math.floor(math.log2(count)))
                time_bin = int(math.floor(math.log2(1.0 + (t - previous) / hard_time_unit)))
                pool = [d for d, (last, c) in source_pairs[u].items()
                        if d not in forbidden and last < t
                        and int(math.floor(math.log2(c))) == count_bin
                        and int(math.floor(math.log2(1.0 + (t - last) / hard_time_unit))) == time_bin]
                if not pool:
                    skipped.append(i)
                    continue
                negative = pool[rng.randint(len(pool))]
            else:
                if not observed_destinations or len(forbidden.intersection(observed_set)) == len(observed_destinations):
                    skipped.append(i)
                    continue
                negative = None
                for _ in range(32):
                    candidate = observed_destinations[rng.randint(len(observed_destinations))]
                    if candidate not in forbidden:
                        negative = candidate
                        break
                if negative is None:
                    pool = [d for d in observed_destinations if d not in forbidden]
                    negative = pool[rng.randint(len(pool))]
            snapshot = np.full((3, max_neighbors), -1, dtype=np.int32)
            for row, node in enumerate((u, v, negative)):
                history = list(neighbors[node])
                snapshot[row, :len(history)] = history
                if history and np.any(timestamps[history] >= t):
                    raise AssertionError("Neighbor history contains current or future features")
            p, dp = query(u, v, t, i)
            q, dq = query(u, negative, t, i)
            ids.append(i); negatives.append(negative); snapshots.append(snapshot)
            phi_rows.append((p, q)); dt_rows.append((dp, dq)); splits.append(phase)
            ind.append(u in unseen or v in unseen)
        # No event at t enters any query at t.
        for i in allowed:
            u, v = int(sources[i]), int(destinations[i])
            neighbors[u].append(i); neighbors[v].append(i)
            previous, count = pairs.get((u, v), (None, 0))
            pairs[(u, v)] = (t, count + 1)
            source_pairs[u][v] = (t, count + 1)
            if v not in observed_set:
                observed_set.add(v); observed_destinations.append(v)
        start = stop
    if not ids:
        raise ValueError("No eligible negatives from past destination history")
    return CandidateStream(np.array(ids, dtype=np.int64), np.array(negatives, dtype=np.int64),
                           np.stack(snapshots), np.array(phi_rows, dtype=np.float32),
                           np.array(dt_rows, dtype=np.float32), np.array(splits, dtype=np.int8),
                           np.array(ind, dtype=bool), np.array(excluded, dtype=np.int64),
                           np.array(skipped, dtype=np.int64), np.array(sorted(unseen), dtype=np.int64),
                           float(train_end), float(validation_end))


def neighbor_batch(stream, rows, timestamps, features):
    """Gather only pre-event features; padding cannot reveal candidate labels."""
    edges = stream.neighbor_edges[rows]
    mask = edges >= 0
    safe = np.maximum(edges, 0)
    times = timestamps[stream.event_ids[rows]][:, None, None]
    dt = np.where(mask, times - timestamps[safe], 0).astype(np.float32)
    if np.any(dt[mask] <= 0):
        raise AssertionError("Non-past feature in model input")
    x = features[safe] * mask[..., None]
    return x, dt, mask
