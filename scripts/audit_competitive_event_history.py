"""Prospective input audit: fixed causal profiles, no model fitting or scoring."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import time
import tomllib
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competing_edges import source_bucket,competing_exposure
from lifecycle_readout.competitive_event_stream import event_frozen_exposure
from lifecycle_readout.competitive_lifecycle import competitive_step
from lifecycle_readout.benchmark import save_json,sha256


def direct_history(data,stream,rows,dimensions,seed):
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    allowed=np.ones(len(t),dtype=bool);allowed[stream.excluded_training_ids]=False
    bucket=np.array([source_bucket(x,dimensions,seed) for x in u])
    def profile(dest,time):
        counts=np.bincount(bucket[allowed & (v==dest) & (t<time)],minlength=dimensions).astype(float)
        norm=np.linalg.norm(counts)
        return counts/norm if norm else counts
    raw=np.zeros((len(stream.event_ids),2,4));ref=np.zeros((len(stream.event_ids),2,dimensions))
    doses={}
    for row in rows:
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



def run(root,output):
    if output.exists():raise FileExistsError('Fresh input-audit output required')
    protocol_path=root/'protocols/competitive_event_history_v1.toml'
    p=tomllib.loads(protocol_path.read_text())
    files=[root/'src/lifecycle_readout/competitive_event_stream.py',root/'src/lifecycle_readout/competitive_lifecycle.py',root/'src/lifecycle_readout/competing_edges.py',root/'src/lifecycle_readout/causal_stream.py',Path(__file__),protocol_path,root/'tests/test_competitive_event_stream.py']
    pins={str(f.relative_to(root)):sha256(f) for f in files}
    parent=root/p['parent_candidates'];parent_results=json.loads((parent/'artifacts/results.json').read_text())
    assert [x['seed'] for x in parent_results['per_seed_runs']]==p['seeds']
    archive=root/p['parent_data_archive']
    output.mkdir(parents=True)
    pre=dict(protocol_id=p['protocol_id'],source_hashes=pins,parent_source_sha256=sha256(archive),parent_results_sha256=sha256(parent/'artifacts/results.json'),started_unix=time.time(),model_fit=False,test_scored=False,gpu_allocated=False)
    pre['parent_input_hashes']={str(f.relative_to(root)):sha256(f) for seed in p['seeds'] for f in (parent/'artifacts'/f'seed{seed}'/'candidate_identity.npz',parent/'artifacts'/f'seed{seed}'/'competition_features.npz')}
    pre['environment']=dict(numpy=np.__version__,torch=torch.__version__,python=sys.version)
    save_json(output/'preanalysis.json',pre)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('source_manifest.json'))
        rel='resources/corpora/wikipedia_causal_v2.npz';raw=z.read(rel)
        assert hashlib.sha256(raw).hexdigest()==manifest[rel]==parent_results['data']['derived_sha256']
        data=dict(np.load(io.BytesIO(raw),allow_pickle=False))
    results=[]
    for seed in p['seeds']:
        settings=parent_results['settings']
        stream=build_candidate_stream(**data,seed=seed,split=(.7,.15,.15),unseen_ratio=.1,split_seed=settings['split_seed'],max_neighbors=settings['max_neighbors'],history_decay=settings['history_decay'],negative_sampling='matched_historical',hard_time_unit=settings['hard_time_unit'])
        folder=parent/'artifacts'/f'seed{seed}'
        prior=dict(np.load(folder/'candidate_identity.npz'))
        for key in ('event_ids','negative_destinations','split','pair_features','elapsed_times','inductive','excluded_training_ids','unseen_nodes'):
            np.testing.assert_array_equal(getattr(stream,key),prior[key])
        out=event_frozen_exposure(data,stream,dimensions=p['profile_dimensions'],profile_seed=p['profile_seed'])
        anchors=[]
        for phase in (0,1):
            rows=np.flatnonzero(stream.split==phase)
            assert len(rows)>0
            anchors.extend(rows[np.linspace(0,len(rows)-1,min(p['anchors_per_split'],len(rows)),dtype=int)].tolist())
        expected,refs,events=direct_history(data,stream,anchors,p['profile_dimensions'],p['profile_seed'])
        np.testing.assert_allclose(out['raw'][anchors],expected[anchors],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(out['reference'][anchors],refs[anchors],rtol=1e-12,atol=1e-12)
        initial=np.array([.2,.3,.5]);d=np.array([.2,.9]);g=np.array([.8,.1]);unit=1000.
        states=competitive_step(torch.tensor(np.tile(initial,(2*len(anchors),1))),torch.tensor(out['elapsed'][anchors].ravel()/unit),torch.tensor(out['raw'][anchors,:,1].ravel()),torch.tensor(d),torch.tensor(g)).numpy()
        state_error=0.
        for index,(row,slot) in enumerate(( (row,slot) for row in anchors for slot in (0,1))):
            last,steps=events[(row,slot)];active=initial[1:].copy();clock=last
            for event_time,kernel in steps:
                active*=np.exp(-d*(event_time-clock)/unit);active*=np.exp(-g*kernel);clock=event_time
            active*=np.exp(-d*(data['timestamps'][stream.event_ids[row]]-clock)/unit)
            ref=np.r_[1-active.sum(),active]
            state_error=max(state_error,float(np.max(np.abs(states[index]-ref))))
            np.testing.assert_allclose(states[index],ref,rtol=1e-12,atol=1e-12)
        old=dict(np.load(folder/'competition_features.npz'))
        np.testing.assert_array_equal(out['raw'][:,:,0],old['raw'][:,:,0])
        assert not out['raw'][stream.split==2].any()
        record=dict(seed=seed,anchor_rows=anchors,anchor_candidate_count=2*len(anchors),raw_max_absolute_error=float(np.max(np.abs(out['raw'][anchors]-expected[anchors]))),state_max_absolute_error=state_error,splits={})
        for phase,name in ((0,'train'),(1,'validation')):
            rows=np.flatnonzero(stream.split==phase);raw=out['raw'][rows];diff=raw[:,0,1]-raw[:,1,1]
            record['splits'][name]=dict(queries=len(rows),reference_available_fraction=float(raw[:,:,2].mean()),nonzero_exposure_fraction=float((raw[:,:,1]>0).mean()),candidate_exact_exposure_tie_fraction=float((diff==0).mean()),mean_absolute_candidate_exposure_difference=float(np.abs(diff).mean()),mean_absolute_old_profile_exposure_difference=float(np.abs(raw[:,:,1]-old['raw'][rows,:,1]).mean()))
        np.savez_compressed(output/f'seed{seed}_features.npz',**out,event_ids=stream.event_ids,negative_destinations=stream.negative_destinations,split=stream.split)
        save_json(output/f'seed{seed}_audit.json',record);results.append(record)
        print('EVENT_HISTORY_SEED',seed,'anchors',len(anchors),'PASS',flush=True)
    assert pins=={str(f.relative_to(root)):sha256(f) for f in files}
    result=dict(status='PASS',protocol_id=p['protocol_id'],dataset='wikipedia',data_sha256=parent_results['data']['derived_sha256'],per_seed=results,source_guard_passed=True,model_fit=False,test_scored=False,gpu_allocated=False,publication_eligible=False,scope='All supported feature rows extracted; fixed direct raw-history/state anchors only, not all-real-row independent replay',elapsed_seconds=time.time()-pre['started_unix'])
    save_json(output/'results.json',result)
    print('EVENT_HISTORY_COMPLETE',len(results),'seeds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(ROOT,args.output.resolve())
