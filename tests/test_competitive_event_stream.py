"""Direct raw-history oracle; no shared production prefix/reset implementation."""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competing_edges import source_bucket,competing_exposure
from lifecycle_readout.competitive_event_stream import event_frozen_exposure
from lifecycle_readout.competitive_lifecycle import competitive_step


def fixture(unseen=0.):
    n=480
    data=dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,
              timestamps=np.floor(np.arange(n)*11./30)*30,
              features=np.zeros((n,2),dtype=np.float32))
    stream=build_candidate_stream(**data,unseen_ratio=unseen,negative_sampling='matched_historical',max_neighbors=3)
    return data,stream


def oracle(data,stream,dimensions=8,seed=17):
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    allowed=np.ones(len(t),dtype=bool);allowed[stream.excluded_training_ids]=False
    bucket=np.array([source_bucket(x,dimensions,seed) for x in u])
    def profile(dest,time):
        counts=np.bincount(bucket[allowed & (v==dest) & (t<time)],minlength=dimensions).astype(float)
        norm=np.linalg.norm(counts)
        return counts/norm if norm else counts
    raw=np.zeros((len(stream.event_ids),2,4));ref=np.zeros((len(stream.event_ids),2,dimensions))
    doses={}
    for row in np.flatnonzero(stream.split<=1):
        event=stream.event_ids[row];time=t[event];source=u[event]
        for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
            own=np.flatnonzero(allowed & (u==source) & (v==dest) & (t<time))
            last=t[own[-1]];reference=profile(dest,last)
            between=np.flatnonzero(allowed & (u==source) & (t>last) & (t<time))
            assert not (v[between]==dest).any()
            kernels=[];usable=0
            for k in between:
                vector=profile(v[k],t[k]);usable+=bool(vector.any())
                kernels.append(float(reference@vector))
            raw[row,slot]=(len(between),sum(kernels),bool(reference.any()),usable)
            ref[row,slot]=reference;doses[(row,slot)]=(last,[(t[k],x) for k,x in zip(between,kernels)])
    return raw,ref,doses


@pytest.mark.parametrize('unseen',[0.,.2])
def test_all_synthetic_queries_against_direct_raw_history(unseen):
    data,stream=fixture(unseen)
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    expected,refs,_=oracle(data,stream)
    assert out['queried'].any()
    np.testing.assert_allclose(out['raw'],expected,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(out['reference'],refs,atol=1e-14)
    assert not out['raw'][stream.split==2].any()
    assert not out['reference'][stream.split==2].any()
    assert not out['elapsed'][stream.split==2].any()
    assert not out['reset_time'][stream.split==2].any()


def test_all_query_states_equal_independent_event_by_event_replay():
    data,stream=fixture()
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    _,_,events=oracle(data,stream)
    initial=np.array([.2,.3,.5]);d=np.array([.003,.01]);g=np.array([.8,.1])
    rows=np.flatnonzero(out['queried'])
    state=torch.tensor(np.tile(initial,(2*len(rows),1)),dtype=torch.float64)
    query=competitive_step(state,torch.tensor(out['elapsed'][rows].ravel()),
                           torch.tensor(out['raw'][rows,:,1].ravel()),torch.tensor(d),torch.tensor(g)).numpy()
    for n,(row,slot) in enumerate(( (row,slot) for row in rows for slot in (0,1))):
        last,steps=events[(row,slot)];active=initial[1:].copy();clock=last
        for time,kernel in steps:
            active*=np.exp(-d*(time-clock))
            active*=np.exp(-g*kernel);clock=time
        active*=np.exp(-d*(data['timestamps'][stream.event_ids[row]]-clock))
        # Independent NumPy replay and exact mass completion, not torch operator.
        np.testing.assert_allclose(query[n],np.r_[1-active.sum(),active],atol=2e-14)


def test_future_and_excluded_events_and_payload_do_not_change_features():
    data,stream=fixture(.2)
    assert len(stream.excluded_training_ids)>0
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    altered={k:x.copy() for k,x in data.items()}
    mask=(data['timestamps']>stream.validation_end)
    mask[stream.excluded_training_ids]=True
    altered['sources'][mask]+=100;altered['destinations'][mask]+=1000
    altered['features'][:]=123
    again=event_frozen_exposure(altered,stream,dimensions=8,profile_seed=17)
    for key in ('raw','reference','elapsed','prior_count','shuffled_exposure'):
        np.testing.assert_array_equal(out[key],again[key])


def test_frozen_reference_actually_differs_from_query_time_profile():
    data,stream=fixture()
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    old=competing_exposure(data,stream,dimensions=8,profile_seed=17)
    assert np.max(np.abs(out['raw'][:,:,1]-old['raw'][:,:,1]))>1e-3
    np.testing.assert_array_equal(out['raw'][:,:,0],old['raw'][:,:,0])


def test_duplicate_queries_and_unsupported_dimensions_rejected():
    data,stream=fixture()
    with pytest.raises(ValueError):event_frozen_exposure(data,stream,dimensions=True,profile_seed=17)
    ids=stream.event_ids.copy();ids[1]=ids[0]
    with pytest.raises(ValueError):event_frozen_exposure(data,replace(stream,event_ids=ids),dimensions=8,profile_seed=17)


def test_timestamp_group_permutations_do_not_change_candidate_exposure():
    data,stream=fixture(.2)
    original=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    groups=[np.flatnonzero(data['timestamps']==time)[::-1] for time in np.unique(data['timestamps'])]
    order=np.concatenate(groups);inverse=np.argsort(order)
    changed={k:x[order] for k,x in data.items()}
    remapped=replace(stream,event_ids=inverse[stream.event_ids],
                     excluded_training_ids=inverse[stream.excluded_training_ids])
    replay=event_frozen_exposure(changed,remapped,dimensions=8,profile_seed=17)
    for key in ('raw','reference','prior_count','elapsed','reset_time','shuffled_exposure'):
        np.testing.assert_allclose(original[key],replay[key],atol=1e-12)


def test_exposure_does_not_decrease_between_own_edge_resets():
    data,stream=fixture()
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    records={}
    for row in np.flatnonzero(out['queried']):
        event=stream.event_ids[row]
        for slot,v in enumerate((data['destinations'][event],stream.negative_destinations[row])):
            key=(data['sources'][event],v,out['reset_time'][row,slot])
            records.setdefault(key,[]).append((data['timestamps'][event],out['raw'][row,slot,1]))
    count=0
    for values in records.values():
        values=sorted(values)
        if len(values)>1:
            count+=1
            assert (np.diff([x[1] for x in values])>=-1e-12).all()
    assert count>0


def test_real_auditor_oracle_on_every_synthetic_query():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/audit_competitive_event_history.py'
    spec=importlib.util.spec_from_file_location('history_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data,stream=fixture(.2)
    rows=np.flatnonzero(stream.split<=1)
    raw,refs,_=module.direct_history(data,stream,rows,8,17)
    out=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    np.testing.assert_allclose(raw,out['raw'],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(refs,out['reference'],atol=1e-14)
