"""All validation slots against direct counts, shared-stream prefixes and masks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.competition_exact_overlap import validation_overlap
from lifecycle_readout.competition_attribution import kernel_attribution
from lifecycle_readout.competitive_event_stream import event_frozen_exposure
from lifecycle_readout.causal_stream import build_candidate_stream


def fixture():
    n=480
    return dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,timestamps=np.floor(np.arange(n)*11./30)*30,features=np.zeros((n,2),dtype=np.float32))


def test_every_validation_slot_two_streams_independent_counts():
    data=fixture()
    streams=[build_candidate_stream(**data,seed=s,unseen_ratio=.2,negative_sampling='matched_historical',max_neighbors=3) for s in (101,202)]
    outputs,meta=validation_overlap(data,streams,dimensions=8,profile_seed=17)
    assert meta['exact_prefix_entries']>0
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    allowed=np.ones(len(t),dtype=bool);allowed[streams[0].excluded_training_ids]=False
    def counts(dest,time):
        ids,c=np.unique(u[allowed & (v==dest) & (t<time)],return_counts=True)
        return dict(zip(map(int,ids),map(int,c)))
    for stream,out in zip(streams,outputs):
        original=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
        selected=stream.split==1
        np.testing.assert_allclose(out['components'][selected].sum(-1),original['raw'][selected,:,1],atol=1e-11)
        assert not out['components'][~selected].any()
        for row in np.flatnonzero(selected):
            event=stream.event_ids[row];source=int(u[event]);time=t[event]
            for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
                last=t[np.flatnonzero(allowed & (u==source) & (v==dest) & (t<time))[-1]]
                ref=counts(dest,last);expected=np.zeros(3)
                for k in np.flatnonzero(allowed & (u==source) & (t>last) & (t<time)):
                    x=kernel_attribution(source,ref,counts(v[k],t[k]),dimensions=8,profile_seed=17)
                    expected+=np.array([x['parts'][0],x['same_other'],x['collision']])
                np.testing.assert_allclose(expected,out['components'][row,slot],atol=1e-11)


def test_memory_guard_stops_instead_of_truncating():
    data=fixture();stream=build_candidate_stream(**data,unseen_ratio=0,negative_sampling='matched_historical')
    with pytest.raises(RuntimeError,match='memory budget'):validation_overlap(data,[stream],dimensions=8,profile_seed=17,max_prefix_entries=0)
