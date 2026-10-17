"""Full validation input census after fixed-anchor anatomy, no fitting/scoring."""
from pathlib import Path
import argparse,hashlib,io,json,sys,time,tomllib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.benchmark import save_json,sha256
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competition_exact_overlap import validation_overlap


def run(output):
    if output.exists():raise FileExistsError('Fresh output required')
    p=tomllib.loads((ROOT/'protocols/competition_attribution_full_validation_v1.toml').read_text())
    prior=ROOT/'results/audit/competition_attribution_execution_v1'
    pins=[ROOT/'protocols/competition_attribution_full_validation_v1.toml',Path(__file__),ROOT/'src/lifecycle_readout/competition_exact_overlap.py',ROOT/'src/lifecycle_readout/competing_edges.py',ROOT/'src/lifecycle_readout/causal_stream.py',ROOT/'tests/test_competition_exact_overlap.py',prior/'results.json',ROOT/'results/audit/lcr-wikipedia-256094e9e7/source.zip']
    for seed in p['seeds']:
        pins.extend([ROOT/f'results/audit/lcr-wikipedia-f83b3eafa4/artifacts/seed{seed}/candidate_identity.npz',ROOT/f'results/audit/competitive_event_history_execution_v1/seed{seed}_features.npz',prior/f'seed{seed}_anatomy.npz'])
    identities={str(f.relative_to(ROOT)):sha256(f) for f in pins};output.mkdir(parents=True)
    save_json(output/'preanalysis.json',dict(protocol_id=p['protocol_id'],source_input_hashes=identities,posthoc_after_anchor_outcomes=True,full_validation_component_outcomes_previously_seen=False,started_unix=time.time(),python=sys.version,numpy=np.__version__))
    with zipfile.ZipFile(ROOT/'results/audit/lcr-wikipedia-256094e9e7/source.zip') as archive:
        raw=archive.read('resources/corpora/wikipedia_causal_v2.npz');data=dict(np.load(io.BytesIO(raw),allow_pickle=False))
    assert hashlib.sha256(raw).hexdigest()=='64ad05c6114e592b594e0f0b48f842993d04996d860805821c18c190cbfb3f83'
    streams=[]
    for seed in p['seeds']:
        stream=build_candidate_stream(**data,seed=seed,split_seed=42,max_neighbors=20,history_decay=.01,negative_sampling='matched_historical',hard_time_unit=1000.)
        parent=dict(np.load(ROOT/f'results/audit/lcr-wikipedia-f83b3eafa4/artifacts/seed{seed}/candidate_identity.npz'))
        for key in ('event_ids','negative_destinations','split','pair_features','elapsed_times','excluded_training_ids'):
            np.testing.assert_array_equal(getattr(stream,key),parent[key])
        streams.append(stream)
    outputs,metadata=validation_overlap(data,streams,dimensions=p['profile_dimensions'],profile_seed=p['profile_seed'],max_prefix_entries=2000000)
    result=dict(status='PASS',protocol_id=p['protocol_id'],posthoc=True,per_seed=[],algorithm_metadata=metadata,model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
    for seed,stream,out in zip(p['seeds'],streams,outputs):
        fixed=dict(np.load(ROOT/f'results/audit/competitive_event_history_execution_v1/seed{seed}_features.npz'))
        old=dict(np.load(prior/f'seed{seed}_anatomy.npz'));selected=stream.split==1
        q=out['components'][selected];exposure=fixed['raw'][selected,:,1]
        np.testing.assert_allclose(q.sum(-1),exposure,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(q[:,:,0],old['parts'][selected,:,0],rtol=1e-8,atol=1e-8)
        assert (q[:,:,1]<=old['parts'][selected,:,3]+1e-8).all()
        assert (q[:,:,2]>=old['parts'][selected,:,1:3].sum(-1)-1e-8).all()
        assert not out['components'][~selected].any()
        anchor_rows=old['anchor_rows'];keep=stream.split[anchor_rows]==1
        np.testing.assert_allclose(out['components'][anchor_rows[keep]],old['anchor_exact_components'][keep],rtol=1e-8,atol=1e-8)
        sums=q.sum((0,1));total=float(exposure.sum())
        entry=dict(seed=seed,validation_queries=int(selected.sum()),candidate_slots=int(2*selected.sum()),total_exposure=total,component_sums=sums.tolist(),component_fractions=(sums/total).tolist(),max_total_reconstruction_error=float(np.max(np.abs(q.sum(-1)-exposure))),max_anchor_exact_component_error=float(np.max(np.abs(out['components'][anchor_rows[keep]]-old['anchor_exact_components'][keep]))))
        result['per_seed'].append(entry)
        np.savez_compressed(output/f'seed{seed}_exact_overlap.npz',**out,event_ids=stream.event_ids,negative_destinations=stream.negative_destinations,split=stream.split)
        print('FULL_OVERLAP_SEED',seed,'PASS',flush=True)
    assert identities=={str(f.relative_to(ROOT)):sha256(f) for f in pins}
    result['source_guard_passed']=True;save_json(output/'results.json',result)
    print('FULL_OVERLAP_COMPLETE',len(result['per_seed']),'seeds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output.resolve())
