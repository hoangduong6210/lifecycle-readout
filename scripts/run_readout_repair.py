"""Colab-only no-refit repair and ranking-distortion evaluation."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,time,tomllib,zipfile
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(dataset,settings,output):
    if not torch.cuda.is_available():raise RuntimeError('Real repair evaluation requires Colab CUDA')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh output required')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for item in settings['input_files']:assert sha(ROOT/item['path'])==item['sha256']
    protocol=tomllib.loads((ROOT/settings['protocol']).read_text());variants=protocol['variants']
    folder=ROOT/settings['anchor_directory'];source=output/'sealed_parent_source'
    with zipfile.ZipFile(folder/'parent_code.zip') as z:
        for rel,digest in json.loads(z.read('source_manifest.json')).items():
            raw=z.read(rel);assert hashlib.sha256(raw).hexdigest()==digest
            path=(source/rel).resolve();assert path.is_relative_to(source.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    spec=importlib.util.spec_from_file_location('repair_engine',ROOT/'src/lifecycle_readout/readout_repair.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    assert not any(k=='lifecycle_readout' or k.startswith('lifecycle_readout.') for k in sys.modules)
    sys.path.insert(0,str(source/'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    parent=ROOT/settings['parent_attempt'];parent_results=json.loads((parent/'artifacts/results.json').read_text())
    result=dict(protocol_id=protocol['protocol_id'],dataset=dataset,settings=settings,model_fit=False,test_scored=False,publication_eligible=False,per_seed_runs=[],environment=dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)))
    started=time.monotonic()
    for parent_entry in parent_results['per_seed_runs']:
        seed=parent_entry['seed'];anchors=dict(np.load(folder/f'seed{seed}_anchors.npz'));n=len(anchors['event_ids']);total=2*n
        checkpoint=parent/'artifacts'/f'seed{seed}/ordered_gate_seed{seed}.pt';assert sha(checkpoint)==parent_entry['models']['ordered_gate']['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=HistoryControl('ordered_gate',parent_results['data']['feature_dim'],saved['settings'],saved['phi_mean'],saved['phi_std']).cuda().eval();model.load_state_dict(saved['model'])
        lifecycle=model.lifecycle;double=copy.deepcopy(lifecycle).double();scores={v:[] for v in variants};curves={v:[] for v in variants};slopes={v:[] for v in variants};double_curves={v:[] for v in variants};potentials=[];joint=[];coordinate=[];max_error=0.
        grid=torch.from_numpy(anchors['normalized_times']);steps=len(grid)
        for start in range(0,total,settings['candidate_microbatch']):
            stop=min(start+settings['candidate_microbatch'],total);length=stop-start
            initial,context,base=[torch.from_numpy(anchors[key][start:stop].copy()).cuda() for key in ['initial','context','base']]
            actual=torch.from_numpy(anchors['elapsed_times'][start:stop].copy()).cuda()/model.time_unit
            with torch.no_grad():
                _,reference=lifecycle.kernel(initial,context,torch.zeros_like(actual))
                potential=engine.gradient_potential(reference);cap=engine.readout_caps(lifecycle.survival_gate.survival_weights)
                potentials.append(potential.cpu().numpy());joint.append(engine.project_potential(potential,cap).cpu().numpy());coordinate.append(engine.project_potential(potential,cap.new_zeros(len(cap))).cpu().numpy())
                for variant in variants:scores[variant].append(engine.repair_score(lifecycle,initial,context,base,actual,variant).cpu().numpy())
            for variant in variants:
                repeated=[value.repeat_interleave(steps,0) for value in [initial,context,base]]
                with torch.no_grad():native=engine.repair_score(lifecycle,*repeated,grid.float().cuda().repeat(length),variant)
                time_grid=grid.double().cuda().repeat(length).requires_grad_()
                lifted=engine.repair_score(double,*[value.double() for value in repeated],time_grid,variant)
                derivative,=torch.autograd.grad(lifted.sum(),time_grid)
                left=native.reshape(length,steps).cpu().numpy();right=lifted.detach().reshape(length,steps).cpu().numpy()
                np.testing.assert_allclose(left,right,rtol=2e-4,atol=2e-5);max_error=max(max_error,float(np.max(np.abs(left-right))))
                curves[variant].append(left);double_curves[variant].append(right);slopes[variant].append(derivative.detach().reshape(length,steps).cpu().numpy())
        arrays=dict(event_ids=anchors['event_ids'],negative_destinations=anchors['negative_destinations'],normalized_times=anchors['normalized_times'],raw_potential=np.concatenate(potentials),readout_potential=np.concatenate(joint),coordinate_potential=np.concatenate(coordinate))
        entry=dict(seed=seed,query_count=n,checkpoint_sha256=sha(checkpoint),variants={},native_double_max_error=max_error)
        original=np.concatenate(scores['original_dynamic']).clip(1e-6,1-1e-6);y=np.r_[np.ones(n),np.zeros(n)]
        original_order=np.sign(original[:n]-original[n:])
        for variant in variants:
            raw=np.concatenate(scores[variant]);clipped=raw.clip(1e-6,1-1e-6);trajectory=np.concatenate(curves[variant]);high=np.concatenate(double_curves[variant]);derivative=np.concatenate(slopes[variant])
            flags=(np.diff(trajectory,axis=-1)>1e-5).any(-1);double_flags=(np.diff(high,axis=-1)>1e-5).any(-1)
            entry['variants'][variant]=dict(ap=float(average_precision_score(y,clipped)),raw_ap=float(average_precision_score(y,raw)),auc=float(roc_auc_score(y,clipped)),brier=float(np.mean((clipped-y)**2)),pair_accuracy=float(np.mean((clipped[:n]>clipped[n:])+.5*(clipped[:n]==clipped[n:]))),mean_absolute_probability_shift=float(np.mean(np.abs(clipped-original))),paired_order_flips=int((np.sign(clipped[:n]-clipped[n:])!=original_order).sum()),violation_count=int(flags.sum()),candidate_trajectories=total,positive_derivative_trajectories=int((derivative>1e-8).any(-1).sum()),native_double_flag_disagreements=int((flags!=double_flags).sum()))
            arrays[variant+'_scores']=raw;arrays[variant+'_curves']=trajectory;arrays[variant+'_double_curves']=high;arrays[variant+'_derivatives']=derivative
        entry['paired_ap_difference']=entry['variants']['readout_projected']['ap']-entry['variants']['original_dynamic']['ap']
        best=parent_entry['models']['ordered_gate']['selected_epoch'];old=next(x for x in parent_entry['models']['ordered_gate']['validation_history'] if x['epoch']==best)['validation']['ap']
        entry['original_validation_ap_reconstruction_difference']=entry['variants']['original_dynamic']['ap']-old
        entry['readout_projection_changed_candidates']=int((np.max(np.abs(arrays['readout_potential']-arrays['raw_potential']),axis=-1)>1e-7).sum())
        np.savez_compressed(output/f'seed{seed}.npz',**arrays);result['per_seed_runs'].append(entry)
        (output/'partial_results.json').write_text(json.dumps(result,indent=2)+'\n')
        print('REPAIR_SEED_COMPLETE',dataset,seed,'AP_delta',entry['paired_ap_difference'],'repair_violations',entry['variants']['readout_projected']['violation_count'],flush=True)
    delta=[e['paired_ap_difference'] for e in result['per_seed_runs']]
    result.update(complete_declared_seeds=[e['seed'] for e in result['per_seed_runs']]==protocol['seeds'],primary_ap=dict(values=delta,mean=float(np.mean(delta)),std_ddof1=float(np.std(delta,ddof=1))),elapsed_seconds=time.monotonic()-started)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['wikipedia','reddit'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--output_dir',type=Path,required=True)
    args=p.parse_args();run(args.dataset,tomllib.loads(args.config.read_text()),args.output_dir)
