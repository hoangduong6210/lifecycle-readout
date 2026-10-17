from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifecycle_readout.causal_stream import build_candidate_stream, neighbor_batch


def events():
    n = 80
    return dict(sources=np.arange(n) % 4,
                destinations=10 + (np.arange(n) % 7),
                timestamps=np.repeat(np.arange(n // 2, dtype=float), 2),
                features=np.arange(n, dtype=np.float32).reshape(-1, 1))


def test_strict_past_neighbors_and_negative_destinations():
    data = events(); stream = build_candidate_stream(**data, unseen_ratio=0)
    for row, i in enumerate(stream.event_ids):
        t = data["timestamps"][i]
        assert stream.negative_destinations[row] in data["destinations"][data["timestamps"] < t]
        assert stream.negative_destinations[row] not in data["destinations"][(data["timestamps"] == t) & (data["sources"] == data["sources"][i])]
        edges = stream.neighbor_edges[row]
        assert np.all(data["timestamps"][edges[edges >= 0]] < t)
    assert stream.skipped_no_negative_ids.size > 0


def test_positive_and_negative_unobserved_pair_features_are_identical():
    data = events(); stream = build_candidate_stream(**data, unseen_ratio=0)
    for row in range(len(stream.event_ids)):
        p, q = stream.pair_features[row]
        if p[2] == 0 and q[2] == 0:
            np.testing.assert_array_equal(p, q)
            assert stream.elapsed_times[row, 0] == stream.elapsed_times[row, 1] == 0


def test_current_and_future_event_features_cannot_enter_earlier_inputs():
    data = events(); stream = build_candidate_stream(**data, unseen_ratio=0)
    cutoff = 30
    rows = np.flatnonzero(stream.event_ids < cutoff)
    changed = dict(data, features=data["features"].copy())
    changed["features"][cutoff:] = 999999
    a = neighbor_batch(stream, rows, data["timestamps"], data["features"])
    b = neighbor_batch(stream, rows, changed["timestamps"], changed["features"])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    # Candidate's own event features must also be absent from its input.
    row = 12; i = stream.event_ids[row]
    changed["features"][i] = -100
    np.testing.assert_array_equal(neighbor_batch(stream, [row], data["timestamps"], data["features"])[0],
                                  neighbor_batch(stream, [row], changed["timestamps"], changed["features"])[0])


def test_held_out_training_edges_absent_from_every_history():
    data = events(); stream = build_candidate_stream(**data, unseen_ratio=0.3)
    assert stream.excluded_training_ids.size
    assert not np.intersect1d(stream.excluded_training_ids, stream.neighbor_edges).size
    train_ids = stream.event_ids[stream.split == 0]
    assert not np.intersect1d(data["sources"][train_ids], stream.unseen_nodes).size
    assert not np.intersect1d(data["destinations"][train_ids], stream.unseen_nodes).size
    assert not np.intersect1d(stream.negative_destinations[stream.split == 0], stream.unseen_nodes).size


def test_reproducible_fixed_split_with_distinct_negative_seeds():
    data = events()
    a = build_candidate_stream(**data, seed=1, split_seed=42)
    b = build_candidate_stream(**data, seed=7, split_seed=42)
    c = build_candidate_stream(**data, seed=1, split_seed=42)
    np.testing.assert_array_equal(a.unseen_nodes, b.unseen_nodes)
    assert a.digest() == c.digest()
    assert a.digest() != b.digest()


def test_validation_and_test_share_complete_history_stream():
    data = events(); stream = build_candidate_stream(**data, unseen_ratio=0)
    first_test = np.flatnonzero(stream.split == 2)[0]
    neighbors = stream.neighbor_edges[first_test]
    valid_edges = neighbors[neighbors >= 0]
    assert np.any((data["timestamps"][valid_edges] > stream.train_end) &
                  (data["timestamps"][valid_edges] <= stream.validation_end))


def test_invalid_chronology_fails_instead_of_clamping_future_history():
    data = events(); data["timestamps"][5] = -1
    with pytest.raises(ValueError, match="nondecreasing"):
        build_candidate_stream(**data)
