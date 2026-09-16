"""
Evaluation metrics for temporal link prediction.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_ap_auc(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """
    Compute Average Precision and AUC-ROC.
    pos_scores: (N,) scores for positive edges
    neg_scores: (N,) scores for negative edges
    """
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # Guard against degenerate cases
    if len(np.unique(labels)) < 2:
        return 0.5, 0.5

    ap  = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return ap, auc


class RunningMetrics:
    """Accumulates batch metrics across an epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._pos = []
        self._neg = []
        self._losses = []
        self._extras = {}

    def update(self, pos_scores, neg_scores, loss: float, extras: dict = None):
        self._pos.append(pos_scores)
        self._neg.append(neg_scores)
        self._losses.append(loss)
        if extras:
            for k, v in extras.items():
                self._extras.setdefault(k, []).append(v)

    def compute(self):
        pos = np.concatenate(self._pos)
        neg = np.concatenate(self._neg)
        ap, auc = compute_ap_auc(pos, neg)
        result = {
            "AP": ap,
            "AUC": auc,
            "Loss": np.mean(self._losses),
        }
        for k, vals in self._extras.items():
            result[k] = np.mean(vals)
        return result
