"""A tied-event fixture for the independent checkpoint replay history."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('hard_checkpoint_auditor', Path(__file__).resolve().parents[1] / 'scripts/audit_hard_checkpoint_replay.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_replay_history_excludes_heldout_training_and_entire_current_group():
    data = dict(sources=np.array([0, 0, 0, 1, 0, 0, 1, 0]),
                destinations=np.array([10, 11, 10, 12, 10, 11, 11, 10]),
                timestamps=np.array([1., 1., 2., 2., 3., 3., 3., 4.]))
    candidates = dict(event_ids=np.array([4, 5, 7]), split=np.array([2, 2, 2]),
                      negative_destinations=np.array([11, 10, 11]), excluded_training_ids=np.array([2]))
    ids, edges, phi, elapsed = auditor.rebuild_inputs(data, candidates, 3, .01)
    np.testing.assert_array_equal(ids, [4, 5, 7])
    # Earlier events without supervision still appear; excluded event 2 never does.
    np.testing.assert_array_equal(edges[0], [[0, 1, -1], [0, -1, -1], [1, -1, -1]])
    np.testing.assert_array_equal(edges[1], [[0, 1, -1], [1, -1, -1], [0, -1, -1]])
    np.testing.assert_array_equal(edges[2], [[1, 4, 5], [0, 4, -1], [1, 5, 6]])
    np.testing.assert_array_equal(elapsed, [[2., 2.], [2., 2.], [1., 1.]])
    np.testing.assert_allclose(phi[:, :, 2], [[np.log(2)]*2, [np.log(2)]*2, [np.log(3)]*2])
    np.testing.assert_allclose(phi[:, :, 3], [[.5]*2, [1/np.sqrt(5)]*2, [2/np.sqrt(7)]*2])


def test_stable_scalar_log_score_preserves_order_below_probability_floor():
    logits = torch.tensor([0., 1.], dtype=torch.float32)
    rate = torch.tensor(0., dtype=torch.float32)
    elapsed = torch.tensor([100., 100.], dtype=torch.float32)
    raw = auditor.scalar_log_score(logits, rate, elapsed, 1.)
    assert raw.dtype == torch.float64 and torch.isfinite(raw).all()
    assert raw[1] > raw[0]
    clipped = raw.exp().clamp(1e-6, 1-1e-6)
    assert clipped[0] == clipped[1]
    # Moderate values match direct probability multiplication without clipping.
    moderate = auditor.scalar_log_score(logits, rate, elapsed*.01, 1.)
    expected = torch.sigmoid(logits.double()) * torch.exp(-torch.nn.functional.softplus(rate.double()))
    torch.testing.assert_close(moderate.exp(), expected)
