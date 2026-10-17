"""Fixed-model paired user-cluster uncertainty for a single retained cohort."""
import numpy as np


class WeightedPairedAP:
    """Tie-aware AP for paired candidates, reusing one fixed score ordering."""
    def __init__(self, positive, negative):
        positive, negative = np.asarray(positive), np.asarray(negative)
        if positive.ndim != 1 or negative.shape != positive.shape or not len(positive):
            raise ValueError('Nonempty paired vectors required')
        scores = np.r_[positive, negative]
        if not np.isfinite(scores).all():
            raise ValueError('Finite scores required')
        self.n = len(positive)
        self.order = np.argsort(-scores, kind='stable')
        sorted_scores = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores)-1]
        self.positive = (self.order < self.n).astype(np.float64)

    def __call__(self, query_weights):
        weights = np.asarray(query_weights, dtype=np.float64)
        if weights.shape != (self.n,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Finite nonnegative nonzero paired weights required')
        sorted_weights = np.r_[weights, weights][self.order]
        tp = np.cumsum(sorted_weights*self.positive)[self.ends]
        count = np.cumsum(sorted_weights)[self.ends]
        precision = np.divide(tp, count, out=np.zeros_like(tp), where=count > 0)
        return float(np.sum(np.diff(np.r_[0.,tp]) * precision) / weights.sum())


def paired_cluster_bootstrap(users, seed_pairs, *, replicates, rng_seed):
    """Same sampled user multiplicities for every model and frozen seed pair.

    seed_pairs contains (phase positive, phase negative, MLP positive, MLP negative).
    Queries are shared across seeds; user is the positive-query source, with its
    weight assigned to both candidates. This is conditional on models/month.
    """
    users = np.asarray(users)
    if users.ndim != 1 or not len(users) or replicates < 1 or not seed_pairs:
        raise ValueError('Users, model pairs and positive replicate count required')
    active, inverse = np.unique(users, return_inverse=True)
    evaluators = []
    for pair in seed_pairs:
        if len(pair) != 4 or any(np.asarray(a).shape != users.shape for a in pair):
            raise ValueError('Every seed must cover the same paired query population')
        evaluators.append((WeightedPairedAP(*pair[:2]), WeightedPairedAP(*pair[2:])))
    rng = np.random.default_rng(rng_seed)
    multiplicities = np.empty((replicates,len(active)),dtype=np.int64)
    differences = np.empty((replicates,len(evaluators)),dtype=np.float64)
    for b in range(replicates):
        multiplicities[b] = np.bincount(rng.integers(0,len(active),size=len(active)),minlength=len(active))
        weights = multiplicities[b,inverse]
        for j,(phase,control) in enumerate(evaluators):
            differences[b,j] = phase(weights)-control(weights)
    means = differences.mean(axis=1)
    interval = np.quantile(means,[.025,.975],method='linear')
    return dict(active_users=active,user_multiplicities=multiplicities,
                seed_differences=differences,mean_differences=means,interval=interval)
