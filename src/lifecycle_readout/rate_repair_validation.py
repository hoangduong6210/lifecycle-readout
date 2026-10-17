"""No-refit fixed-rate versus universal-rate repair with a common x metric."""
from pathlib import Path
import json,hashlib,time,shutil
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from .modeling.inverse_tilt_projection import project_inverse_tilts,all_rate_inverse_projection
from .modeling.fixed_rate_tilt import fixed_rate_check


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tilt_score(initial,base,time,potential,rates,gaps,w0,bias):
    argument=-time[:,None]*rates
    flow=np.c_[initial[:,0]+np.sum(initial[:,1:]*(-np.expm1(argument)),axis=1),initial[:,1:]*np.exp(argument)]
    weight=np.exp(potential-potential.max(axis=1,keepdims=True));weighted=flow*weight;q=weighted/weighted.sum(axis=1,keepdims=True)
    gate_logit=w0+q[:,1:]@gaps+bias;gate=expit(gate_logit);score=expit(base)*gate
    ratio=np.exp(potential[:,:1]-potential[:,1:]);derivative=-q[:,1:]@(gaps*rates)+(q[:,1:]@gaps)*np.sum(q[:,1:]*rates*(1-ratio),axis=1)
    return score,expit(base)*gate*(1-gate)*derivative


def metrics(raw,n,clip):
    p=raw.clip(clip,1-clip);labels=np.r_[np.ones(n),np.zeros(n)]
    return dict(ap=float(average_precision_score(labels,p)),raw_ap=float(average_precision_score(labels,raw)),auc=float(roc_auc_score(labels,p)),pair_accuracy=float(np.mean((p[:n]>p[n:])+.5*(p[:n]==p[n:]))))


def run_study(root,settings,protocol,out):
    root=Path(root);out=Path(out);pre=json.loads((out/'preanalysis.json').read_text());assert settings['seeds']==protocol['seeds'] and [x['name'] for x in settings['dataset']]==protocol['datasets'];assert not (out/'results.json').exists()
    for name,h in pre['source_input_hashes'].items():assert sha(root/name)==h
    execution=dict(status='running',started_unix=time.time());(out/'progress.json').write_text(json.dumps(execution,indent=2)+'\n');result=dict(protocol_id=protocol['protocol_id'],model_fit=False,test_scored=False,publication_eligible=False,settings=settings,datasets={})
    try:
        for item in settings['dataset']:
            name=item['name'];parent=json.loads((root/item['repair_attempt']/'artifacts/results.json').read_text());assert [e['seed'] for e in parent['per_seed_runs']]==settings['seeds'];entries=[]
            for old in parent['per_seed_runs']:
                seed=old['seed'];a=dict(np.load(root/item['repair_attempt']/f'artifacts/seed{seed}.npz',allow_pickle=False));anchor=dict(np.load(root/item['anchors']/f'seed{seed}_anchors.npz',allow_pickle=False));np.testing.assert_array_equal(a['event_ids'],anchor['event_ids']);np.testing.assert_array_equal(a['negative_destinations'],anchor['negative_destinations']);n=len(a['event_ids']);total=2*n;assert a['raw_potential'].shape==(total,5)
                path=root/item['parent_attempt']/f'artifacts/seed{seed}/ordered_gate_seed{seed}.pt';assert sha(path)==old['checkpoint_sha256'];saved=torch.load(path,map_location='cpu',weights_only=True);weights=saved['model'];rates=np.logaddexp(0,weights['lifecycle.kernel.decay_param'].numpy().astype(float));gaps=np.cumsum(np.logaddexp(0,weights['lifecycle.survival_gate.deltas'].numpy().astype(float)));w0=float(weights['lifecycle.survival_gate.w_base']);bias=float(weights['lifecycle.survival_gate.b_surv'][0]);potential=a['raw_potential'].astype(float);assert np.all(potential[:,0]==0);raw_x=np.exp(-potential[:,1:]/2)
                fixed=project_inverse_tilts(raw_x,gaps,rates,tolerance=settings['solver_tolerance'],max_cycles=settings['solver_max_cycles']);xf=fixed['inverse_tilt'];xu=all_rate_inverse_projection(raw_x,gaps);df=np.square(xf-raw_x).sum(1);du=np.square(xu-raw_x).sum(1);assert np.all(df<=du+settings['distance_tolerance'])
                potentials={'gradient_unconstrained':potential.copy()}
                for kind,x in [('all_rate_x_projected',xu),('fixed_rate_x_projected',xf)]:
                    new=potential.copy();changed=x!=raw_x;new[:,1:]=np.where(changed,-2*np.log(x),new[:,1:]);potentials[kind]=new
                initial=anchor['initial'].astype(float);initial/=initial.sum(axis=1,keepdims=True);base=anchor['base'].astype(float);actual=anchor['elapsed_times'].astype(float)/float(saved['settings']['decay_time_unit']);grid=anchor['normalized_times'].astype(float);assert len(grid)==25
                arrays=dict(event_ids=a['event_ids'],negative_destinations=a['negative_destinations'],normalized_times=grid,raw_potential=potential,raw_inverse=raw_x,all_rate_inverse=xu,fixed_rate_inverse=xf,dual=fixed['dual'],A=fixed['A'],b=fixed['b'],rates=rates,gaps=gaps,w0=np.array(w0),bias=np.array(bias));entry=dict(seed=seed,query_count=n,candidate_count=total,checkpoint_sha256=sha(path),solver_cycles=fixed['cycles'],kkt=fixed['kkt'],constraint_count=len(fixed['b']),variants={},mean_squared_distance_fixed=float(df.mean()),mean_squared_distance_all_rate=float(du.mean()),strict_distance_reduction_count=int((df<du-settings['distance_tolerance']).sum()),fixed_changed_candidates=int((np.max(np.abs(xf-raw_x),axis=1)>settings['solver_tolerance']).sum()),all_rate_changed_candidates=int((np.max(np.abs(xu-raw_x),axis=1)>settings['solver_tolerance']).sum()))
                for kind in protocol['variants']:
                    pot=potentials[kind];raw,_=tilt_score(initial,base,actual,pot,rates,gaps,w0,bias);curves=[];derivatives=[]
                    for start in range(0,total,settings['candidate_microbatch']):
                        stop=min(total,start+settings['candidate_microbatch']);length=stop-start;score,derivative=tilt_score(np.repeat(initial[start:stop],25,axis=0),np.repeat(base[start:stop],25),np.tile(grid,length),np.repeat(pot[start:stop],25,axis=0),rates,gaps,w0,bias);curves.append(score.reshape(length,25));derivatives.append(derivative.reshape(length,25))
                    curves=np.concatenate(curves);derivative=np.concatenate(derivatives);flags=(np.diff(curves,axis=1)>settings['grid_increase_threshold']).any(1);slope_flags=(derivative>settings['derivative_threshold']).any(1);report=metrics(raw,n,settings['probability_clip']);report.update(violation_count=int(flags.sum()),positive_derivative_trajectories=int(slope_flags.sum()),mean_absolute_shift_from_raw=None)
                    if kind!='gradient_unconstrained':assert not flags.any() and not slope_flags.any()
                    entry['variants'][kind]=report;arrays[kind+'_scores']=raw;arrays[kind+'_curves']=curves;arrays[kind+'_derivatives']=derivative
                for kind in protocol['variants']:entry['variants'][kind]['mean_absolute_shift_from_raw']=float(np.abs(arrays[kind+'_scores']-arrays['gradient_unconstrained_scores']).mean())
                expected=a['gradient_unconstrained_scores'];np.testing.assert_allclose(arrays['gradient_unconstrained_scores'],expected,rtol=settings['parent_replay_rtol'],atol=settings['parent_replay_atol']);entry['parent_gradient_max_probability_error']=float(np.abs(arrays['gradient_unconstrained_scores']-expected).max());entry['parent_gradient_ap_difference']=entry['variants']['gradient_unconstrained']['ap']-old['variants']['gradient_unconstrained']['ap'];entry['original_dynamic_reference_ap']=old['variants']['original_dynamic']['ap'];entry['primary_ap_difference']=entry['variants']['fixed_rate_x_projected']['ap']-entry['variants']['all_rate_x_projected']['ap'];entry['fixed_minus_raw_ap']=entry['variants']['fixed_rate_x_projected']['ap']-entry['variants']['gradient_unconstrained']['ap'];entry['fixed_minus_original_dynamic_reference_ap']=entry['variants']['fixed_rate_x_projected']['ap']-entry['original_dynamic_reference_ap']
                folder=out/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'seed{seed}.npz',**arrays);entries.append(entry);result['datasets'][name]=dict(per_seed=entries);(out/'partial_results.json').write_text(json.dumps(result,indent=2)+'\n');print('RATE_REPAIR_SEED_COMPLETE',name,seed,flush=True)
            values=[e['primary_ap_difference'] for e in entries];result['datasets'][name].update(primary_ap=dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1))))
        for name,h in pre['source_input_hashes'].items():assert sha(root/name)==h
        result.update(status='PASS_EXECUTION_PENDING_INDEPENDENT_AUDIT',elapsed_seconds=time.time()-execution['started_unix']);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');execution.update(status='completed',ended_unix=time.time())
    except BaseException as exc:
        execution.update(status='failed',error_type=type(exc).__name__,ended_unix=time.time());raise
    finally:(out/'progress.json').write_text(json.dumps(execution,indent=2)+'\n')
