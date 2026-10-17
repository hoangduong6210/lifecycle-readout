"""Ranking under supplied official candidates; never generate replacements.

Midrank matches TGB-Seq evaluator at upstream commit
c1ee801ea4301c2f944f5b3da10cbc5310fa4f68. Inputs may be logits or scores;
only ordering and exact ties matter. Return per-query values for aggregation.
"""
import numpy as np


def reciprocal_ranks(positive, negatives):
    positive, negatives = np.asarray(positive), np.asarray(negatives)
    if positive.ndim != 1 or negatives.ndim != 2 or negatives.shape[0] != len(positive):
        raise ValueError('Expected positive [queries] and negative [queries, candidates]')
    if not len(positive) or not negatives.shape[1]:
        raise ValueError('Empty queries or candidate sets')
    if not np.isfinite(positive).all() or not np.isfinite(negatives).all():
        raise ValueError('Finite scores required')
    higher = np.count_nonzero(negatives > positive[:, None], axis=1)
    tied = np.count_nonzero(negatives == positive[:, None], axis=1)
    rank = 1. + higher + .5 * tied
    return 1. / rank
