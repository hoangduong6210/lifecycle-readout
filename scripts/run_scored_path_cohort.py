"""Colab-only complete supported-validation path diagnostic; no optimizer."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
import tomllib
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def support(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(dataset,settings,output):
    if not torch.cuda.is_available():raise RuntimeError('Full-cohort intervention requires Colab CUDA')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh output required')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for item in settings['input_files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Input identity mismatch')
    protocol=tomllib.loads((ROOT/settings['protocol']).read_text())
    parent=ROOT/settings['parent_attempt'];parent_results=json.loads((parent/'artifacts/results.json').read_text())
    assert parent_results['dataset']==dataset and parent_results['protocol_id']=='LCR-P-HARD-001'
    assert [entry['seed'] for entry in parent_results['per_seed_runs']]==protocol['seeds']
    source=output/'sealed_parent_source';cache=output.parent/'scored_path_parent_cache'/dataset;cache.mkdir(parents=True,exist_ok=False)
    prepared=cache/f'{dataset}.npz'
    with zipfile.ZipFile(parent/'source.zip') as archive:
        manifest=json.loads(archive.read('source_manifest.json'))
        for rel,digest in manifest.items():
            if rel.startswith('src/') and rel.endswith('.py'):
                raw=archive.read(rel);assert hashlib.sha256(raw).hexdigest()==digest
                target=(source/rel).resolve();assert target.is_relative_to(source.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        prepared.write_bytes(archive.read(f'resources/corpora/{dataset}_causal_v2.npz'))
    assert sha(prepared)==parent_results['data']['derived_sha256']
    data=dict(np.load(prepared))
    history=support('raw_history_support',ROOT/'scripts/diagnose_scored_silence.py')
    engine=support('cohort_engine',ROOT/'src/lifecycle_readout/scored_path_cohort.py')
    direction=support('direction_engine',ROOT/'src/lifecycle_readout/scored_path.py')
    assert not any(k=='lifecycle_readout' or k.startswith('lifecycle_readout.') for k in sys.modules)
    sys.path.insert(0,str(source/'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    import lifecycle_readout.hard_controls as archived
    assert Path(archived.__file__).resolve().is_relative_to(source.resolve())
    grid=torch.logspace(protocol['normalized_times_log10_min'],protocol['normalized_times_log10_max'],protocol['normalized_times_count'],dtype=torch.double)
    result=dict(protocol_id=protocol['protocol_id'],dataset=dataset,parent_attempt=settings['parent_attempt'],
        data=parent_results['data'],settings=settings,per_seed_runs=[],model_fit=False,test_scored=False,publication_eligible=False,
        environment=dict(torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0)))
    started=time.monotonic()
    for entry in parent_results['per_seed_runs']:
        seed=entry['seed'];folder=parent/'artifacts'/f'seed{seed}';candidates=dict(np.load(folder/'candidate_identity.npz'))
        selected=np.flatnonzero(candidates['split']==1);ids=candidates['event_ids'][selected]
        edges,phi,elapsed=history.selected_history(data,candidates,ids,parent_results['settings']['max_neighbors'],parent_results['settings']['history_decay'])
        np.testing.assert_allclose(phi,candidates['pair_features'][selected],rtol=2e-6,atol=1e-7);np.testing.assert_array_equal(elapsed,candidates['elapsed_times'][selected])
        checkpoint=folder/f'ordered_gate_seed{seed}.pt';assert sha(checkpoint)==entry['models']['ordered_gate']['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=HistoryControl('ordered_gate',data['features'].shape[1],saved['settings'],saved['phi_mean'],saved['phi_std']).cuda().eval()
        model.load_state_dict(saved['model']);parts=[dict(initial=[],context=[],base=[],fixed=[]) for _ in range(2)];anchor_error=0.
        for start in range(0,len(ids),settings['backbone_batch_size']):
            stop=min(start+settings['backbone_batch_size'],len(ids));idx=edges[start:stop];mask=idx>=0;safe=np.maximum(idx,0)
            x=data['features'][safe]*mask[...,None]
            ages=np.where(mask,data['timestamps'][ids[start:stop],None,None]-data['timestamps'][safe],0).astype(np.float32)
            assert (ages[mask]>0).all()
            inputs=[torch.from_numpy(v.copy()).cuda() for v in (x,ages,mask,phi[start:stop],elapsed[start:stop])]
            x,dt,mask,p,e=inputs
            with torch.no_grad():
                original=model(*inputs)
                for slot,dest in enumerate((1,2)):
                    h,b=model.base(x[:,0],dt[:,0],mask[:,0],x[:,dest],dt[:,dest],mask[:,dest])
                    context=(p[:,slot]-model.phi_mean)/model.phi_std
                    b=b+model.history_head(torch.cat((h,context),-1)).squeeze(-1)
                    initial=model.lifecycle.encoder(h,context);q,matrix=model.lifecycle.kernel(initial,context,e[:,slot]/model.time_unit)
                    reconstructed=torch.sigmoid(b)*model.lifecycle.survival_gate(q)
                    torch.testing.assert_close(reconstructed.clamp(1e-6,1-1e-6),original[slot]['prob'],rtol=2e-4,atol=2e-5)
                    anchor_error=max(anchor_error,float((reconstructed.clamp(1e-6,1-1e-6)-original[slot]['prob']).abs().max()))
                    for key,value in dict(initial=initial,context=context,base=b,fixed=matrix).items():parts[slot][key].append(value.cpu())
        anchors={key:torch.cat([torch.cat(part[key]) for part in parts]) for key in parts[0]}
        double=copy.deepcopy(model.lifecycle).double();all_arrays={};certificate_errors=[];boundaries=0;max_precision_error=0.
        for path in protocol['paths']:
            all_arrays[path+'_native_scores']=[];all_arrays[path+'_double_scores']=[];all_arrays[path+'_derivatives']=[]
        total=len(anchors['initial'])
        for start in range(0,total,settings['candidate_microbatch']):
            stop=min(start+settings['candidate_microbatch'],total)
            tensors={key:value[start:stop].cuda() for key,value in anchors.items()}
            with torch.no_grad():native=engine.evaluate_paths(model.lifecycle,**tensors,grid=grid.float().cuda(),paths=protocol['paths'])
            lifted=engine.evaluate_paths(double,**{key:value.double() for key,value in tensors.items()},grid=grid.cuda(),paths=protocol['paths'],derivative_support=direction.entropic_direction)
            for path in protocol['paths']:
                left,right=native[path]['scores'],lifted[path]['scores'];np.testing.assert_allclose(left,right,rtol=2e-4,atol=2e-5)
                max_precision_error=max(max_precision_error,float(np.max(np.abs(left-right))))
                all_arrays[path+'_native_scores'].append(left);all_arrays[path+'_double_scores'].append(right);all_arrays[path+'_derivatives'].append(lifted[path]['derivatives'])
                error=lifted[path]['certificate_max_error']
                if error is not None:certificate_errors.append(error)
                boundaries+=lifted[path].get('certificate_boundary_exclusions',0)
        arrays={key:np.concatenate(value) for key,value in all_arrays.items()}
        arrays.update({key:value.numpy() for key,value in anchors.items()});arrays.update(event_ids=ids,negative_destinations=candidates['negative_destinations'][selected],normalized_times=grid.numpy())
        np.savez_compressed(output/f'seed{seed}.npz',**arrays)
        paths=[]
        for path in protocol['paths']:
            scores=arrays[path+'_native_scores'];delta=np.diff(scores,axis=-1);double_delta=np.diff(arrays[path+'_double_scores'],axis=-1)
            flags=(delta>protocol['adjacent_increase_threshold']).any(-1);double_flags=(double_delta>protocol['adjacent_increase_threshold']).any(-1)
            paths.append(dict(path=path,candidate_trajectories=total,violation_count=int(flags.sum()),violation_fraction=float(flags.mean()),
                clipped_violation_count=int((np.diff(scores.clip(1e-6,1-1e-6),axis=-1)>protocol['adjacent_increase_threshold']).any(-1).sum()),
                positive_derivative_trajectories=int((arrays[path+'_derivatives']>protocol['positive_derivative_threshold']).any(-1).sum()),
                native_double_flag_disagreements=int((flags!=double_flags).sum()),maximum_adjacent_increase=float(delta.max())))
        result['per_seed_runs'].append(dict(seed=seed,query_count=len(ids),candidate_trajectories=total,checkpoint_sha256=sha(checkpoint),candidate_identity_sha256=sha(folder/'candidate_identity.npz'),
            path_results=paths,anchor_max_error=anchor_error,native_double_max_error=max_precision_error,certificate_max_error=max(certificate_errors),nonsmooth_boundary_exclusions=boundaries))
        (output/'partial_results.json').write_text(json.dumps(result,indent=2)+'\n')
        print('COHORT_SEED_COMPLETE',dataset,seed,'queries',len(ids),'dynamic_violations',paths[-1]['violation_count'],flush=True)
    result.update(complete_declared_seeds=[e['seed'] for e in result['per_seed_runs']]==protocol['seeds'],elapsed_seconds=time.monotonic()-started)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['wikipedia','reddit'],required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--output_dir',type=Path,required=True)
    args=p.parse_args();run(args.dataset,tomllib.loads(args.config.read_text()),args.output_dir)
