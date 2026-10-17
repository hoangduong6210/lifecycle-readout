from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.ranking import reciprocal_ranks


def test_midrank_handles_wins_losses_and_ties():
    positive = np.array([1., 0., .5, .2])
    negatives = np.array([[0., 0., 0.], [1., 1., 1.], [.5, .5, .5], [.3, .2, .1]])
    np.testing.assert_allclose(reciprocal_ranks(positive, negatives), [1., .25, .4, .4])


def test_full_tie_matches_fixed_multi_negative_evaluation():
    result = reciprocal_ranks(np.zeros(2), np.zeros((2, 100)))
    np.testing.assert_allclose(result, [1 / 51, 1 / 51])


@pytest.mark.parametrize('pos,neg', [([], [[]]), ([1.], [[]]), ([1.], [[float('nan')]]),
                                    ([[1.]], [[0.]]), ([1., 2.], [[0.]])])
def test_invalid_ranking_inputs_are_rejected(pos, neg):
    with pytest.raises(ValueError): reciprocal_ranks(pos, neg)
