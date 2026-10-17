from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_holdout_evaluation import WeightedPairedAP,paired_cluster_bootstrap


def oracle(p,n,weights):
    return average_precision_score(np.r_[np.ones(len(p)),np.zeros(len(p))],
                                   np.r_[p,n],sample_weight=np.r_[weights,weights])


def test_weighted_ties_and_zero_weight_queries_match_sklearn():
    rng=np.random.default_rng(17)
    for n in (1,7,57):
        for _ in range(12):
            p,nscore=rng.integers(0,4,size=(2,n))/3
            weights=rng.integers(0,5,size=n);weights[0]+=1
            assert WeightedPairedAP(p,nscore)(weights)==pytest.approx(oracle(p,nscore,weights),abs=2e-15)
    with pytest.raises(ValueError):WeightedPairedAP([np.nan],[0])
    with pytest.raises(ValueError):WeightedPairedAP([1],[0])([0])


def test_shared_cluster_multiplicities_every_seed_and_every_replicate():
    users=np.array([4,4,4,7,9,9]);rng=np.random.default_rng(19)
    pairs=[tuple(rng.integers(0,8,size=(4,len(users)))/7) for _ in range(3)]
    result=paired_cluster_bootstrap(users,pairs,replicates=37,rng_seed=20261016)
    assert (result['user_multiplicities'].sum(axis=1)==3).all()
    for i,multiplicity in enumerate(result['user_multiplicities']):
        weight=multiplicity[np.searchsorted(result['active_users'],users)]
        expected=[oracle(*pair[:2],weight)-oracle(*pair[2:],weight) for pair in pairs]
        np.testing.assert_allclose(result['seed_differences'][i],expected,atol=2e-15,rtol=0)
    np.testing.assert_array_equal(result['interval'],np.quantile(result['seed_differences'].mean(1),[.025,.975],method='linear'))
    again=paired_cluster_bootstrap(users,pairs,replicates=37,rng_seed=20261016)
    np.testing.assert_array_equal(result['user_multiplicities'],again['user_multiplicities'])
    with pytest.raises(ValueError):paired_cluster_bootstrap(users,[(np.ones(2),)*4],replicates=2,rng_seed=1)
