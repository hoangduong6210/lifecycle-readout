import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('session_diagnostic', Path(__file__).parents[1]/'scripts/analyze_source_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_spec = importlib.util.spec_from_file_location('session_audit', Path(__file__).parents[1]/'scripts/audit_source_session.py')
audit_module = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit_module)


def test_tied_events_use_previous_distinct_timestamp_and_same_source_only():
    u = np.array([3, 4, 3, 3, 4, 3, 4])
    t = np.array([1., 1., 2., 2., 3., 5., 5.])
    np.testing.assert_equal(module.source_gaps(u,t), [np.nan,np.nan,1,1,2,3,2])


def test_future_validation_rows_do_not_change_training_gaps_or_cuts():
    u = np.array([0,1,0,1,0,1,0,1])
    t = np.arange(8,dtype=float)
    first = module.source_gaps(u,t)
    later = module.source_gaps(np.r_[u,[0,1]],np.r_[t,[20.,30.]])
    np.testing.assert_equal(first,later[:8])
    np.testing.assert_equal(module.training_cuts(first,np.zeros(8)),
                            module.training_cuts(later,np.r_[np.zeros(8),[1,1]]))


def test_duplicate_train_cutoffs_keep_empty_intervals_and_missing_history():
    gaps = np.array([np.nan,2.,2.,2.])
    ranks = dict(edge_memory=np.array([.1,.2,.3,.4]),
                 popularity=np.array([.2,.3,.4,.5]),pooled_transition=np.array([.3,.4,.5,.6]))
    rows, labels = module.summarize(gaps,ranks,np.array([2.,2.,2.]))
    np.testing.assert_array_equal(labels,[-1,3,3,3])
    assert [row['count'] for row in rows] == [1,0,0,0,3]
    assert rows[1]['transition_minus_popularity'] is None


@pytest.mark.parametrize('times', [[1.,0.],[0.,np.nan]])
def test_invalid_chronology_fails(times):
    with pytest.raises(ValueError):
        module.source_gaps(np.array([1,1]),np.array(times))


def test_empty_training_history_fails():
    with pytest.raises(ValueError):
        module.training_cuts(np.array([np.nan,1.]),np.array([0,1]))


def test_two_algorithms_match_direct_strict_past_lookup_with_ties():
    rng = np.random.default_rng(1583)
    u = rng.integers(0,11,200)
    t = np.sort(rng.integers(0,35,200)).astype(float)
    expected = np.array([at-t[(u==src)&(t<at)].max() if ((u==src)&(t<at)).any() else np.nan
                         for src,at in zip(u,t)])
    np.testing.assert_equal(module.source_gaps(u,t),expected)
    np.testing.assert_equal(audit_module.chronological_reference(u,t),expected)
