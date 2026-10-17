"""Posthoc decomposition of fixed input features; no prediction/model fitting."""
from pathlib import Path
import argparse,hashlib,io,json,sys,time,tomllib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.benchmark import save_json,sha256
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competition_attribution import exposure_attribution,kernel_attribution


def run(output):
    if output.exists():raise FileExistsError('Fresh anatomy output required')
    protocol=ROOT/'protocols/competition_attribution_v1.toml';p=tomllib.loads(protocol.read_text())
    files=[protocol,Path(__file__),ROOT/'src/lifecycle_readout/competition_attribution.py',ROOT/'src/lifecycle_readout/competitive_event_stream.py',ROOT/'src/lifecycle_readout/competing_edges.py',ROOT/'src/lifecycle_readout/causal_stream.py',ROOT/'tests/test_competition_attribution.py']
    inputs=[ROOT/p['data_archive']]
    inputs.extend(ROOT/p['candidate_parent']/f'seed{s}/candidate_identity.npz' for s in p['seeds'])
    inputs.extend(ROOT/p['feature_parent']/f'seed{s}_features.npz' for s in p['seeds'])
    pins={str(f.relative_to(ROOT)):sha256(f) for f in files+inputs}
    output.mkdir(parents=True)
    save_json(output/'preanalysis.json',dict(protocol_id=p['protocol_id'],source_input_hashes=pins,posthoc_after_model_outcomes=True,real_data_component_outcomes_previously_seen=False,numpy=np.__version__,python=sys.version,started_unix=time.time()))
    with zipfile.ZipFile(ROOT/p['data_archive']) as archive:
        rel='resources/corpora/wikipedia_causal_v2.npz';raw=archive.read(rel)
        manifest=json.loads(archive.read('source_manifest.json'))
        assert hashlib.sha256(raw).hexdigest()==manifest[rel]
        data=dict(np.load(io.BytesIO(raw),allow_pickle=False))
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    result=dict(protocol_id=p['protocol_id'],dataset='wikipedia',data_sha256=hashlib.sha256(raw).hexdigest(),posthoc=True,per_seed=[],model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
    for seed in p['seeds']:
        stream=build_candidate_stream(**data,seed=seed,split_seed=42,max_neighbors=20,history_decay=.01,negative_sampling='matched_historical',hard_time_unit=1000.)
        prior=dict(np.load(ROOT/p['candidate_parent']/f'seed{seed}/candidate_identity.npz'))
        for key in ('event_ids','negative_destinations','split','pair_features','elapsed_times','excluded_training_ids'):
            np.testing.assert_array_equal(getattr(stream,key),prior[key])
        frozen=dict(np.load(ROOT/p['feature_parent']/f'seed{seed}_features.npz'))
        out=exposure_attribution(data,stream,dimensions=p['profile_dimensions'],profile_seed=p['profile_seed'])
        np.testing.assert_allclose(out['parts'].sum(-1),frozen['raw'][:,:,1],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(out['shuffled_exposure'],frozen['shuffled_exposure'],rtol=1e-8,atol=1e-8)
        assert not out['parts'][stream.split==2].any()
        allowed=np.ones(len(t),dtype=bool);allowed[stream.excluded_training_ids]=False
        selected=[]
        for phase in (0,1):
            rows=np.flatnonzero(stream.split==phase)
            selected.extend(rows[np.linspace(0,len(rows)-1,min(8,len(rows)),dtype=int)].tolist())
        cache={}
        def exact_counts(dest,time):
            key=(int(dest),float(time))
            if key not in cache:
                users,counts=np.unique(u[allowed & (v==dest) & (t<time)],return_counts=True)
                cache[key]=dict(zip(map(int,users),map(int,counts)))
            return cache[key]
        exact=np.zeros((len(selected),2,3));max_part_error=0.;max_background_error=0.
        for index,row in enumerate(selected):
            event=stream.event_ids[row];source=int(u[event]);query_time=t[event]
            for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
                own=np.flatnonzero(allowed & (u==source) & (v==dest) & (t<query_time));last=t[own[-1]]
                reference=exact_counts(dest,last);parts=np.zeros(4);background=0.
                for k in np.flatnonzero(allowed & (u==source) & (t>last) & (t<query_time)):
                    q=kernel_attribution(source,reference,exact_counts(v[k],t[k]),dimensions=p['profile_dimensions'],profile_seed=p['profile_seed'])
                    parts+=q['parts'];background+=q['uniform_background']
                    exact[index,slot]+=np.array([q['parts'][0],q['same_other'],q['collision']])
                np.testing.assert_allclose(parts,out['parts'][row,slot],rtol=1e-8,atol=1e-8)
                np.testing.assert_allclose(background,out['uniform_background'][row,slot],rtol=1e-8,atol=1e-8)
                np.testing.assert_allclose(exact[index,slot].sum(),frozen['raw'][row,slot,1],rtol=1e-8,atol=1e-8)
                max_part_error=max(max_part_error,float(np.max(np.abs(parts-out['parts'][row,slot]))))
                max_background_error=max(max_background_error,float(abs(background-out['uniform_background'][row,slot])))
        entry=dict(seed=seed,anchor_rows=selected,anchor_exact_components=exact.sum((0,1)).tolist(),max_anchor_component_error=max_part_error,max_anchor_background_error=max_background_error,max_full_exposure_error=float(np.max(np.abs(out['parts'].sum(-1)-frozen['raw'][:,:,1]))),splits={})
        for phase,name in ((0,'train'),(1,'validation')):
            rows=np.flatnonzero(stream.split==phase);parts=out['parts'][rows];exposure=frozen['raw'][rows,:,1];total=float(exposure.sum());sums=parts.sum((0,1))
            entry['splits'][name]=dict(queries=len(rows),total_exposure=total,component_sums=sums.tolist(),component_fractions=(sums/total).tolist() if total else None,uniform_background_over_total=float(out['uniform_background'][rows].sum()/total) if total else None,mean_absolute_centered_alignment=float(np.abs(exposure-out['uniform_background'][rows]).mean()))
        np.savez_compressed(output/f'seed{seed}_anatomy.npz',**out,anchor_rows=np.array(selected),anchor_exact_components=exact,event_ids=stream.event_ids,negative_destinations=stream.negative_destinations,split=stream.split)
        save_json(output/f'seed{seed}_audit.json',entry);result['per_seed'].append(entry)
        print('ANATOMY_SEED',seed,'PASS',flush=True)
    assert pins=={str(f.relative_to(ROOT)):sha256(f) for f in files+inputs}
    result['status']='PASS';result['source_guard_passed']=True
    save_json(output/'results.json',result);print('ANATOMY_COMPLETE',len(result['per_seed']),'seeds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output.resolve())
