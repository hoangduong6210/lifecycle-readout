"""Exact-ID witnesses and every-query raw-history decomposition checks."""
from pathlib import Path
import itertools
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.competition_attribution import kernel_attribution,exposure_attribution
from lifecycle_readout.competing_edges import source_bucket
from lifecycle_readout.competitive_event_stream import event_frozen_exposure
from lifecycle_readout.causal_stream import build_candidate_stream


def test_own_source_alone_produces_maximal_similarity_without_other_users():
    x=kernel_attribution(0,{0:2},{0:7},dimensions=8,profile_seed=17)
    np.testing.assert_allclose(x['parts'],[1,0,0,0])
    assert x['same_other']==0 and x['collision']==0


def test_collision_alone_can_produce_maximal_similarity():
    ids={}
    for user in range(100):
        bucket=source_bucket(user,8,17)
        if bucket in ids:break
        ids[bucket]=user
    other=ids[bucket];assert other!=user
    x=kernel_attribution(1000,{user:1},{other:1},dimensions=8,profile_seed=17)
    assert x['parts'].sum()==1 and x['collision']==1 and x['same_other']==0


def test_permutation_expectation_and_uniform_component_identity():
    a=np.array([1.,2.,4.]);a/=np.linalg.norm(a)
    b=np.array([3.,1.,2.]);b/=np.linalg.norm(b)
    values=[a[list(p)]@b for p in itertools.permutations(range(3))]
    np.testing.assert_allclose(np.mean(values),a.sum()*b.sum()/3,atol=1e-15)
    np.testing.assert_allclose(a@b,a.sum()*b.sum()/3+(a-a.mean())@(b-b.mean()),atol=1e-15)
    uniform=np.ones(3)/np.sqrt(3)
    for p in itertools.permutations(range(3)):np.testing.assert_allclose(uniform[list(p)]@uniform,1)


@pytest.mark.parametrize('unseen',[0.,.2])
def test_every_synthetic_query_against_direct_raw_id_reconstruction(unseen):
    n=480
    data=dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,timestamps=np.floor(np.arange(n)*11./30)*30,features=np.zeros((n,2),dtype=np.float32))
    stream=build_candidate_stream(**data,unseen_ratio=unseen,negative_sampling='matched_historical',max_neighbors=3)
    out=exposure_attribution(data,stream,dimensions=8,profile_seed=17)
    original=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    np.testing.assert_allclose(out['parts'].sum(-1),original['raw'][:,:,1],atol=1e-11)
    np.testing.assert_allclose(out['shuffled_exposure'],original['shuffled_exposure'],atol=1e-11)
    allowed=np.ones(n,dtype=bool);allowed[stream.excluded_training_ids]=False
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    def counts(dest,time):
        users,c=np.unique(u[allowed & (v==dest) & (t<time)],return_counts=True)
        return dict(zip(map(int,users),map(int,c)))
    for row in np.flatnonzero(out['queried']):
        event=stream.event_ids[row];source=int(u[event]);time=t[event]
        for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
            last=t[np.flatnonzero(allowed & (u==source) & (v==dest) & (t<time))[-1]]
            ref=counts(dest,last);parts=np.zeros(4);background=0.
            for k in np.flatnonzero(allowed & (u==source) & (t>last) & (t<time)):
                result=kernel_attribution(source,ref,counts(v[k],t[k]),dimensions=8,profile_seed=17)
                parts+=result['parts'];background+=result['uniform_background']
                np.testing.assert_allclose(result['parts'].sum(),result['parts'][0]+result['same_other']+result['collision'],atol=1e-13)
            np.testing.assert_allclose(out['parts'][row,slot],parts,atol=1e-11)
            np.testing.assert_allclose(out['uniform_background'][row,slot],background,atol=1e-11)
    assert not out['parts'][stream.split==2].any()
