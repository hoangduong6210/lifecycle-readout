"""Independent small examples for pooling claims and exact tie handling."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('pool_diagnostic', Path(__file__).resolve().parents[1]/'scripts/analyze_hard_pooling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cartesian_decomposition_matches_explicit_pairs_with_ties():
    p, q = np.array([.8, .5, .2]), np.array([.8, .3, .1])
    bins = np.array([[0, 0], [0, 0], [1, 2]])
    r = module.diagnose(p, q, bins)
    wins = (p[:, None] > q[None, :]) + .5*(p[:, None] == q[None, :])
    same = np.all(bins[:, None, :] == bins[None, :, :], axis=-1)
    assert r['pooled_auc'] == pytest.approx(wins.mean())
    assert r['same_bin_cartesian_auc'] == pytest.approx(wins[same].mean())
    assert r['cross_bin_cartesian_auc'] == pytest.approx(wins[~same].mean())
    assert r['half_credit_pair_accuracy'] == pytest.approx(5/6)
    assert r['auc_reconstruction'] == pytest.approx(wins.mean())


def test_balanced_bin_only_score_has_chance_auc_ap():
    r = module.diagnose(np.array([.9, .1]), np.array([.9, .1]), np.array([[0, 0], [0, 1]]))
    assert r['pooled_ap'] == .5
    assert r['pooled_auc'] == .5
    assert r['half_credit_pair_accuracy'] == .5
    assert r['pooled_brier'] > .25


def test_strict_silence_decrease_is_compatible_with_perfect_pooled_ranking():
    # logit = 10*(2*h-1) - .01*t, h is a predictive covariate.
    # Derivative in t is strictly negative at fixed h, yet all positives beat all negatives.
    t = np.array([1., 100.])
    sigmoid = lambda x: 1/(1+np.exp(-x))
    p, q = sigmoid(10-.01*t), sigmoid(-10-.01*t)
    r = module.diagnose(p, q, np.array([[0, 0], [0, 1]]))
    assert r['pooled_ap'] == r['pooled_auc'] == r['half_credit_pair_accuracy'] == 1
    assert r['pooled_brier'] < 1e-6
    assert np.all(sigmoid(10-.01*(t+1)) < p)


def test_pooled_loss_can_coexist_with_perfect_within_query_order():
    r = module.diagnose(np.array([.9, .2]), np.array([.8, .1]), np.array([[0, 0], [0, 1]]))
    assert r['half_credit_pair_accuracy'] == 1
    assert r['same_bin_cartesian_auc'] == 1
    assert r['cross_bin_cartesian_auc'] == .5
    assert r['pooled_auc'] == .75
    assert r['pooled_ap'] < 1


def test_unmatched_stored_bins_are_rejected():
    phi = np.zeros((1, 2, 6)); phi[:, :, 2] = np.log1p([[1, 4]])
    with pytest.raises(AssertionError):
        module.stored_bins(phi, np.ones((1, 2)), 1000.)


def test_completely_ordered_balanced_bins_match_closed_form_auc():
    bins = np.array([[0, 0], [0, 0], [0, 1]])
    r = module.diagnose(np.array([.9, .85, .2]), np.array([.8, .7, .1]), bins)
    expected = .5 + ((2/3)**2 + (1/3)**2)*.5
    assert r['same_bin_cartesian_auc'] == 1
    assert r['cross_bin_cartesian_auc'] == .5
    assert r['pooled_auc'] == pytest.approx(expected)


def test_original_float32_floor_and_ties_are_reported_separately():
    p = np.array([1e-6, .8], dtype=np.float32)
    q = np.array([1e-6, .3], dtype=np.float32)
    r = module.diagnose(p, q, np.zeros((2, 2), dtype=int))
    assert r['paired_ties'] == r['paired_ties_at_original_clip'] == .5
    assert r['positive_exact_zero'] == r['negative_exact_zero'] == 0
    assert r['half_credit_pair_accuracy'] == .75
    assert r['cross_bin_cartesian_auc'] is None
