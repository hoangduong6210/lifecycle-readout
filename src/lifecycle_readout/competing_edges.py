"""Strict-past competing-destination exposure; exploratory behavioral profiles.

Profiles are hashed source-interaction counts, not observed topic semantics.
All queries at a timestamp precede all updates at that timestamp. Exposure uses
the candidate profile at query time and competitor profiles at their event times.
"""
from dataclasses import replace
import math

import numpy as np


def source_bucket(source: int, dimensions: int, seed: int) -> int:
    """Stable integer mixing, independent of Python's randomized string hash."""
    x = (int(source) + int(seed) + 0x9E3779B97F4A7C15) & ((1 << 64)-1)
    x = ((x ^ (x >> 30))*0xBF58476D1CE4E5B9) & ((1 << 64)-1)
    x = ((x ^ (x >> 27))*0x94D049BB133111EB) & ((1 << 64)-1)
    return (x ^ (x >> 31)) % dimensions


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else np.zeros_like(vector)


def competing_exposure(data, stream, *, dimensions: int, profile_seed: int):
    """Extract raw [competitor count, exposure, profile available, usable count].

    For candidate (u,v), integrate other-source-u destinations strictly after
    the last (u,v) timestamp and strictly before the query. All events at the
    last timestamp are excluded, independent of file ordering within ties.
    Unsupported allowed events enter history; excluded training events do not.
    Test features remain zero and no test feature query is made.
    """
    if not isinstance(dimensions, int) or dimensions < 2:
        raise ValueError("Profile dimensions must be an integer at least two")
    if stream.pair_features.shape[-1] != 6:
        raise ValueError("Expected original six-feature hard stream")
    times, sources, destinations = (data[k] for k in ("timestamps", "sources", "destinations"))
    eligible = stream.split <= 1
    rows = {int(event): row for row, event in enumerate(stream.event_ids) if eligible[row]}
    excluded = set(map(int, stream.excluded_training_ids))
    raw = np.zeros((len(stream.event_ids), 2, 4), dtype=np.float64)
    shuffled = np.zeros(raw.shape[:2], dtype=np.float64)
    permutation = np.random.RandomState(profile_seed).permutation(dimensions)
    if np.array_equal(permutation, np.arange(dimensions)):
        permutation = np.roll(permutation, 1)
    profiles, source_totals, pair_reset = {}, {}, {}
    buckets = {}
    counts = np.zeros(raw.shape[:2], dtype=np.int64)
    elapsed = np.zeros(raw.shape[:2], dtype=np.float64)
    # Save resets only for queried pairs; this changes storage, never history.
    requested = set()
    for event, row in rows.items():
        u = int(sources[event])
        requested.add((u, int(destinations[event])))
        requested.add((u, int(stream.negative_destinations[row])))
    pair_history = {}
    start = 0
    while start < len(times) and times[start] <= stream.validation_end:
        t = float(times[start])
        stop = int(np.searchsorted(times, t, side="right"))
        for event in range(start, stop):
            row = rows.get(event)
            if row is None:
                continue
            u = int(sources[event])
            total, n, usable = source_totals.get(u, (np.zeros(dimensions), 0, 0))
            for slot, v in enumerate((int(destinations[event]), int(stream.negative_destinations[row]))):
                key = (u, v)
                if key not in pair_history or key not in pair_reset:
                    raise AssertionError("Recurrent candidate lacks causal reset")
                last, count = pair_history[key]
                previous, n0, usable0 = pair_reset[key]
                vector = _unit(profiles.get(v, np.zeros(dimensions)))
                recent = total - previous
                exposure = float(vector @ recent)
                shuffled[row, slot] = max(0., float(vector[permutation] @ recent))
                raw[row, slot] = (n-n0, max(0., exposure), float(vector.any()), usable-usable0)
                counts[row, slot] = count
                elapsed[row, slot] = t-last
        allowed = [i for i in range(start, stop) if i not in excluded]
        event_vectors = {int(destinations[i]): _unit(profiles.get(int(destinations[i]), np.zeros(dimensions)))
                         for i in allowed}
        # Competitor vectors all come from the state before this timestamp.
        touched = set()
        for event in allowed:
            u, v = int(sources[event]), int(destinations[event])
            total, n, usable = source_totals.get(u, (np.zeros(dimensions), 0, 0))
            vector = event_vectors[v]
            source_totals[u] = (total + vector, n+1, usable+int(vector.any()))
            last, count = pair_history.get((u, v), (t, 0))
            pair_history[(u, v)] = (t, count+1)
            touched.add((u, v))
        for key in touched & requested:
            total, n, usable = source_totals[key[0]]
            pair_reset[key] = (total.copy(), n, usable)
        for event in allowed:
            u, v = int(sources[event]), int(destinations[event])
            if u not in buckets:
                buckets[u] = source_bucket(u, dimensions, profile_seed)
            if v not in profiles:
                profiles[v] = np.zeros(dimensions)
            profiles[v][buckets[u]] += 1
        start = stop
    np.testing.assert_allclose(elapsed[eligible], stream.elapsed_times[eligible], rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(np.log1p(counts[eligible]), stream.pair_features[eligible, :, 2], rtol=2e-6, atol=1e-7)
    if not np.isfinite(raw).all() or (raw < 0).any() or not np.isfinite(shuffled).all():
        raise AssertionError("Invalid competition features")
    return dict(raw=raw, shuffled_exposure=shuffled, queried=eligible,
                coordinate_permutation=permutation, prior_count=counts, elapsed=elapsed)


def competition_stream(stream, arrays, arm):
    """Same feature dimensions/capacity for every arm; only declared masks differ."""
    raw = arrays["raw"].copy()
    if arm == "masked_mlp":
        raw[:] = 0
    elif arm == "activity_mlp":
        raw[:, :, 1] = 0  # Same availability/usable-count channels as profile.
    elif arm == "shuffled_mlp":
        raw[:, :, 1] = arrays["shuffled_exposure"]
    elif arm not in ("profile_mlp", "activity_displacement", "profile_displacement"):
        raise ValueError("Unknown competition arm")
    for index in (0, 1, 3):
        raw[:, :, index] = np.log1p(raw[:, :, index])
    raw[~arrays["queried"]] = 0
    return replace(stream, pair_features=np.concatenate((stream.pair_features, raw.astype(np.float32)), axis=-1))


def oracle_exposure(data, stream, rows, *, dimensions, profile_seed):
    """Independent query-wise history reconstruction, without prefix subtraction.

    Intended for small audit anchor sets. Destination profiles are rebuilt by
    direct counting of strictly earlier allowed raw events for each requested
    timestamp, rather than sharing extractor state or reset logic.
    """
    times, sources, destinations = (data[k] for k in ("timestamps", "sources", "destinations"))
    allowed = np.ones(len(times), dtype=bool)
    allowed[stream.excluded_training_ids] = False
    buckets = np.array([source_bucket(u, dimensions, profile_seed) for u in sources])
    indices = {int(v): np.flatnonzero(allowed & (destinations == v)) for v in np.unique(destinations)}
    permutation = np.random.RandomState(profile_seed).permutation(dimensions)
    if np.array_equal(permutation, np.arange(dimensions)):
        permutation = np.roll(permutation, 1)
    cache = {}

    def profile(v, time):
        key = (int(v), float(time))
        if key not in cache:
            history = indices.get(int(v), np.empty(0, dtype=int))
            end = np.searchsorted(times[history], time, side="left")
            cache[key] = _unit(np.bincount(buckets[history[:end]], minlength=dimensions).astype(float))
        return cache[key]

    raw = np.zeros((len(rows), 2, 4))
    shuffled = np.zeros((len(rows), 2))
    for anchor, row in enumerate(rows):
        if stream.split[row] > 1:
            raise ValueError("Test anchors are prohibited")
        event = stream.event_ids[row]
        u, time = sources[event], times[event]
        for slot, v in enumerate((destinations[event], stream.negative_destinations[row])):
            past = np.flatnonzero(allowed & (sources == u) & (destinations == v) & (times < time))
            if not len(past):
                raise AssertionError("Oracle candidate has no history")
            recent = np.flatnonzero(allowed & (sources == u) & (times > times[past[-1]]) & (times < time))
            vector = profile(v, time)
            other = [profile(destinations[j], times[j]) for j in recent]
            exposure = sum(float(vector @ other_vector) for other_vector in other)
            shuffled[anchor, slot] = sum(float(vector[permutation] @ other_vector) for other_vector in other)
            raw[anchor, slot] = (len(recent), exposure, float(vector.any()), sum(int(x.any()) for x in other))
    return raw, shuffled
