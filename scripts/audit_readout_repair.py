"""Independent NumPy lifecycle-score, projection KKT and artifact audit."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.special import expit,ndtr,softmax
from sklearn.metrics import average_precision_score,roc_auc_score
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(initial,context,base,time,parameters,variant,potential):
    def parameter(key):return parameters['lifecycle.'+key].numpy().astype(np.float64)
    rates=np.logaddexp(0,parameter('kernel.decay_param'))
    active=initial[:,1:]*np.exp(-time[:,None]*rates)
    if variant in ('original_dynamic','legacy_reference'):
        flow=np.c_[np.clip(1-active.sum(-1),1e-6,1),active];flow/=flow.sum(-1,keepdims=True)
        logits=np.log(np.maximum(flow,1e-6))
    else:
        dormant=initial[:,0]+np.sum(initial[:,1:]*(-np.expm1(-time[:,None]*rates)),axis=-1)
        flow=np.c_[dormant,active]
        with np.errstate(divide='ignore'):logits=np.log(flow)
    if variant in ('original_dynamic','legacy_reference','safe_reference'):
        reference_time=time if variant=='original_dynamic' else np.zeros_like(time)
        inputs=np.c_[context,np.log1p(reference_time)]
        hidden=inputs@parameter('kernel.trans_net.0.weight').T+parameter('kernel.trans_net.0.bias')
        hidden=hidden*ndtr(hidden)
        matrix=hidden@parameter('kernel.trans_net.2.weight').T+parameter('kernel.trans_net.2.bias')
        k=initial.shape[-1];matrix=matrix.reshape(-1,k,k);matrix*=1-np.eye(k)[None]
        logits+=np.einsum('bi,bij->bj',flow,matrix)
    else:logits+=potential
    weights=np.r_[parameter('survival_gate.w_base'),parameter('survival_gate.w_base')+np.cumsum(np.logaddexp(0,parameter('survival_gate.deltas')))]
    gate=expit(softmax(logits,axis=-1)@weights+parameter('survival_gate.b_surv'))
    return expit(base)*gate


def audit(attempt,output):
    if output.exists():raise FileExistsError('Fresh repair audit required')
    job=json.loads((attempt/'job.json').read_text());assert job['status']=='completed' and job['cleanup_verified'] and job['artifact_checksums_verified']
    result=json.loads((attempt/'artifacts/results.json').read_text());settings=result['settings']
    assert result['protocol_id']=='LCR-P-REPAIR-VAL-001' and result['complete_declared_seeds'] and not result['model_fit'] and not result['test_scored']
    assert [e['seed'] for e in result['per_seed_runs']]==[1,7,42]
    records=[];maximum=0.
    for entry in result['per_seed_runs']:
        seed=entry['seed'];arrays=dict(np.load(attempt/'artifacts'/f'seed{seed}.npz'))
        anchors=dict(np.load(ROOT/settings['anchor_directory']/f'seed{seed}_anchors.npz'))
        np.testing.assert_array_equal(arrays['event_ids'],anchors['event_ids']);np.testing.assert_array_equal(arrays['negative_destinations'],anchors['negative_destinations'])
        n=len(arrays['event_ids']);assert entry['query_count']==n;truth=np.r_[np.ones(n),np.zeros(n)]
        checkpoint=ROOT/settings['parent_attempt']/f'artifacts/seed{seed}/ordered_gate_seed{seed}.pt'
        assert sha(checkpoint)==entry['checkpoint_sha256'];saved=torch.load(checkpoint,map_location='cpu',weights_only=True);parameters=saved['model']
        weights=np.r_[parameters['lifecycle.survival_gate.w_base'].numpy(),parameters['lifecycle.survival_gate.w_base'].numpy()+np.cumsum(np.logaddexp(0,parameters['lifecycle.survival_gate.deltas'].numpy()))].astype(float)
        gaps=weights[1:]-weights[0];remaining=gaps.max()-gaps;caps=np.full_like(gaps,np.inf);mask=remaining>0;caps[mask]=np.log(gaps.max())-np.log(remaining[mask])
        for kind,bounds in [('readout_potential',caps),('coordinate_potential',np.zeros_like(caps))]:
            b=arrays[kind][:,1:];raw=arrays['raw_potential'][:,1:];assert (b<=bounds+2e-5).all()
            difference=b-raw;gradient=(len(caps)+1)*difference-difference.sum(-1,keepdims=True)
            free=b<bounds-2e-5
            assert np.max(np.abs(gradient[free]))<2e-4
            assert (gradient[~free]<=2e-4).all()
        original=arrays['original_dynamic_scores'].clip(1e-6,1-1e-6);order=np.sign(original[:n]-original[n:])
        for variant,report in entry['variants'].items():
            raw=arrays[variant+'_scores'];clipped=raw.clip(1e-6,1-1e-6);curves=arrays[variant+'_curves'];double=arrays[variant+'_double_curves'];slopes=arrays[variant+'_derivatives']
            assert raw.shape==(2*n,) and curves.shape==double.shape==slopes.shape==(2*n,25)
            values=dict(ap=float(average_precision_score(truth,clipped)),raw_ap=float(average_precision_score(truth,raw)),auc=float(roc_auc_score(truth,clipped)),brier=float(np.mean((clipped-truth)**2)),pair_accuracy=float(np.mean((clipped[:n]>clipped[n:])+.5*(clipped[:n]==clipped[n:]))),mean_absolute_probability_shift=float(np.mean(np.abs(clipped-original))),paired_order_flips=int((np.sign(clipped[:n]-clipped[n:])!=order).sum()),violation_count=int((np.diff(curves,axis=-1)>1e-5).any(-1).sum()),positive_derivative_trajectories=int((slopes>1e-8).any(-1).sum()),native_double_flag_disagreements=int(((np.diff(curves,axis=-1)>1e-5).any(-1)!=(np.diff(double,axis=-1)>1e-5).any(-1)).sum()))
            for key,value in values.items():np.testing.assert_allclose(value,report[key],rtol=1e-6,atol=1e-8)
            potential=arrays.get({'gradient_unconstrained':'raw_potential','readout_projected':'readout_potential','coordinate_projected':'coordinate_potential'}.get(variant,''),np.zeros((2*n,5)))
            for start in range(0,2*n,128):
                stop=min(start+128,2*n);x=anchors['initial'][start:stop].astype(float);c=anchors['context'][start:stop].astype(float);b=anchors['base'][start:stop].astype(float)
                time=(anchors['elapsed_times'][start:stop]/float(saved['settings']['decay_time_unit'])).astype(np.float32).astype(float)
                actual=replay(x,c,b,time,parameters,variant,potential[start:stop]);np.testing.assert_allclose(actual,raw[start:stop],rtol=2e-4,atol=2e-5)
                maximum=max(maximum,float(np.max(np.abs(actual-raw[start:stop]))))
                repeated=[np.repeat(v,25,axis=0) for v in [x,c,b]];t=np.tile(anchors['normalized_times'].astype(np.float32).astype(float),stop-start)
                expected=replay(*repeated,t,parameters,variant,np.repeat(potential[start:stop],25,axis=0)).reshape(stop-start,25)
                np.testing.assert_allclose(expected,curves[start:stop],rtol=2e-4,atol=2e-5);maximum=max(maximum,float(np.max(np.abs(expected-curves[start:stop]))))
            records.append(dict(dataset=result['dataset'],seed=seed,variant=variant,metrics=values))
        np.testing.assert_allclose(entry['paired_ap_difference'],entry['variants']['readout_projected']['ap']-entry['variants']['original_dynamic']['ap'],atol=1e-15,rtol=0)
    delta=[e['paired_ap_difference'] for e in result['per_seed_runs']];np.testing.assert_array_equal(delta,result['primary_ap']['values']);np.testing.assert_allclose(np.mean(delta),result['primary_ap']['mean'],atol=1e-15,rtol=0)
    report=dict(status='PASS_NUMPY_SCORE_KKT_AND_ARTIFACT_AUDIT',dataset=result['dataset'],records=records,maximum_numpy_probability_error=maximum,runtime_released=True,scope='All actual validation scores and full conditional grid scores independently reconstructed using NumPy/SciPy lifecycle equations from frozen anchors; projection feasibility/KKT, metrics and flags checked. No backbone or optimizer replay, fresh dataset confirmation or novelty clearance.',model_fit=False,test_scored=False,publication_eligible=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],result['dataset'],'maximum_error',maximum)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.attempt,a.output)
