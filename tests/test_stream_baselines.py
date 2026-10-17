"""Independent small-stream replay of count controls and timestamp ties."""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.stream_baselines import training_predecessors, evaluate_counts, draw_candidates


def fixture():
    return dict(src=np.array([1,1,2,1,2,3,1,1,2,3]),
                dst=np.array([5,6,6,7,5,7,5,6,7,5]),
                time=np.array([1.,1.,1.,2.,2.,2.,3.,3.,3.,4.]),
                split=np.array([0,0,0,0,0,0,1,1,1,1]),event_ids=np.arange(10))


def previous(history, source, time):
    pairs=[(t,v) for u,v,t in history if u==source and t<time]
    if not pairs:return -1
    latest=max(t for t,v in pairs)
    return min(v for t,v in pairs if t==latest)


def test_predecessor_uses_prior_timestamp_group_and_deterministic_minimum():
    d=fixture();prev,last=training_predecessors(d['src'],d['dst'],d['time'])
    history=list(zip(d['src'],d['dst'],d['time']))
    np.testing.assert_array_equal(prev,[previous(history,u,t) for u,t in zip(d['src'],d['time'])])
    assert last[1]==5


def test_all_control_scores_match_independent_strict_past_reference(tmp_path):
    d=fixture();settings=dict(max_state_bytes=1000000,models=['edge_memory','popularity','pooled_transition'],
        candidate_seed=42,negative_count=5,popularity_pseudocount=1.,transition_prior_mass=1.)
    result=evaluate_counts(d,settings,tmp_path/'out')
    history=list(zip(d['src'],d['dst'],d['time']))
    catalog=set(d['dst'][d['split']==0])
    with np.load(tmp_path/'out/validation_candidates.npz') as c:
        for kind in settings['models']:
            with np.load(tmp_path/'out'/f'{kind}_predictions.npz') as saved:
                for row,event in enumerate(c['event_ids']):
                    source,time=d['src'][event],d['time'][event]
                    past=[x for x in history if x[2]<time]
                    seen={(u,v) for u,v,t in past};pop=Counter(v for u,v,t in past)
                    trans=Counter((previous(history,u,t),v) for u,v,t in past if previous(history,u,t)>=0)
                    context=previous(history,source,time)
                    candidates=np.r_[d['dst'][event],c['negative_destinations'][row]]
                    forbidden={v for u,v,t in history if u==source and t==time}
                    assert not forbidden.intersection(c['negative_destinations'][row])
                    expected=[]
                    for candidate in candidates:
                        p=(pop[candidate]+1.)/(len(past)+len(catalog))
                        expected.append(float((source,candidate) in seen) if kind=='edge_memory' else p if kind=='popularity' else trans[context,candidate]+p)
                    np.testing.assert_allclose(np.r_[saved['positive'][row],saved['negative'][row]],expected)
    assert not result['test_scored'] and result['query_count']==4


def test_exhausted_catalogue_is_an_explicit_failure():
    with pytest.raises(ValueError,match='No available'):
        draw_candidates(np.random.RandomState(1),np.array([1,1]),np.array([2,3]),np.array([2,3]),5)
