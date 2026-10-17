"""CPU replay of cadence validation checkpoints using independently rebuilt inputs.

Run each attempt in a fresh process so only its verified archived model source
is imported. This does not rerun optimization or create a fresh evaluation set.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import platform
import sys
import time
import zipfile

import numpy as np
import torch

from audit_hard_checkpoint_replay import rebuild_inputs, sha256

ROOT=Path(__file__).resolve().parents[1]


def independent_clock(data,candidates,tau0):
    """Separate timestamp lists and binary searches, including train-only fallback."""
    eligible=candidates['split']<=1
    end=float(data['timestamps'][candidates['event_ids'][eligible]].max())
    excluded=set(candidates['excluded_training_ids'].tolist())
    history=defaultdict(list)
    for event,(u,v,t) in enumerate(zip(data['sources'],data['destinations'],data['timestamps'])):
        if t>end:break
        if event not in excluded:history[int(u),int(v)].append(float(t))
    history={key:np.asarray(value,dtype=np.float64) for key,value in history.items()}
    shape=(len(candidates['event_ids']),2)
    mean=np.full(shape,np.nan);elapsed=np.zeros(shape)
    for row in np.flatnonzero(eligible):
        event=int(candidates['event_ids'][row]);u=int(data['sources'][event]);t=float(data['timestamps'][event])
        for slot,v in enumerate([data['destinations'][event],candidates['negative_destinations'][row]]):
            past=history.get((u,int(v)),np.array([]));n=int(np.searchsorted(past,t,side='left'))
            if n:elapsed[row,slot]=t-past[n-1]
            if n>=2 and past[n-1]>past[0]:mean[row,slot]=(past[n-1]-past[0])/(n-1)
    available=np.isfinite(mean)&(mean>0)
    train=(candidates['split']==0)[:,None]&available
    if not train.any():raise ValueError('No estimable training cadence')
    fallback=float(np.median(mean[train]))
    effective=np.where(available,mean,fallback)
    extra=np.stack([np.log1p(effective/fallback),available.astype(float),
                    np.log1p(elapsed/(effective+tau0))],axis=-1).astype(np.float32)
    extra[~eligible]=0
    return extra,fallback


def replay(attempt,output):
    started=time.monotonic();attempt=attempt.resolve()
    job=json.loads((attempt/'job.json').read_text())
    assert job['status']=='completed' and job['cleanup_verified']
    assert sha256(attempt/'source.zip')==job['source_bundle_sha256']
    output.mkdir(parents=True,exist_ok=False);source=output/'sealed_source'
    with zipfile.ZipFile(attempt/'source.zip') as bundle:
        manifest=json.loads(bundle.read('source_manifest.json'))
        for name,expected in manifest.items():
            if name.startswith('src/') and name.endswith('.py'):
                target=(source/name).resolve();assert target.is_relative_to(source.resolve())
                raw=bundle.read(name)
                import hashlib
                assert hashlib.sha256(raw).hexdigest()==expected
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    assert not any(key=='lifecycle_readout' or key.startswith('lifecycle_readout.') for key in sys.modules)
    sys.path.insert(0,str(source/'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    import lifecycle_readout.hard_controls as archived
    assert Path(archived.__file__).resolve().is_relative_to(source.resolve())
    artifacts=attempt/'artifacts';result=json.loads((artifacts/'results.json').read_text());settings=result['settings']
    assert result['protocol_id']=='LCR-P-CADENCE-DEV-001' and not result['test_scored']
    assert settings['evaluation_split']=='validation' and result['complete_declared_seeds']
    prepared=ROOT/'resources/corpora'/f'{result["dataset"]}_causal_v2.npz'
    assert sha256(prepared)==result['data']['derived_sha256']
    with np.load(prepared,allow_pickle=False) as saved:
        data={key:saved[key] for key in ['sources','destinations','timestamps','features']}
    torch.set_num_threads(1);records=[]
    for run in result['per_seed_runs']:
        seed=run['seed'];folder=artifacts/f'seed{seed}'
        with np.load(folder/'candidate_identity.npz',allow_pickle=False) as saved:candidates={key:saved[key] for key in saved.files}
        wanted=candidates['split']<=1
        # The existing independent history builder selects split 2 by default.
        # Relabel only the selection array; original IDs/exclusions are retained.
        selection=dict(candidates,split=np.where(wanted,2,-1))
        end=data['timestamps'][candidates['event_ids'][wanted]].max()
        stop=int(np.searchsorted(data['timestamps'],end,side='right'))
        prefix={key:value[:stop] for key,value in data.items()}
        ids,edges,phi6,elapsed=rebuild_inputs(prefix,selection,settings['max_neighbors'],settings['history_decay'])
        np.testing.assert_array_equal(ids,candidates['event_ids'][wanted])
        np.testing.assert_allclose(phi6,candidates['pair_features'][wanted,:,:6],rtol=2e-6,atol=1e-7)
        np.testing.assert_array_equal(elapsed,candidates['elapsed_times'][wanted])
        extra,fallback=independent_clock(prefix,candidates,settings['pair_clock_tau0_seconds'])
        meta=json.loads((folder/'pair_clock.json').read_text());assert fallback==meta['fallback_seconds']
        if settings.get('mask_pair_clock_features',False):extra[:]=0
        np.testing.assert_array_equal(extra,candidates['pair_features'][:,:,6:])
        phi=np.concatenate([phi6,extra[wanted]],axis=-1)
        train=candidates['split'][wanted]==0;val=candidates['split'][wanted]==1
        train_phi=phi[train].reshape(-1,9)
        effective=dict(settings,decay_time_unit=float(np.median(elapsed[train][elapsed[train]>0])))
        for kind,info in run['models'].items():
            checkpoint=folder/f'{kind}_seed{seed}.pt';prediction=folder/f'{kind}_seed{seed}_predictions.npz'
            assert sha256(checkpoint)==info['checkpoint_sha256'] and sha256(prediction)==info['predictions_sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['kind']==kind and saved['seed']==seed and saved['settings']==effective
            best=max(info['validation_history'],key=lambda row:row['validation']['ap'])
            assert saved['selected_epoch']==info['selected_epoch']==best['epoch']
            np.testing.assert_allclose(saved['phi_mean'],train_phi.mean(0),rtol=2e-6,atol=1e-7)
            np.testing.assert_allclose(saved['phi_std'],train_phi.std(0).clip(min=1e-5),rtol=2e-6,atol=1e-7)
            model=HistoryControl(kind,data['features'].shape[1],effective,saved['phi_mean'],saved['phi_std'])
            assert sum(p.numel() for p in model.parameters())==info['parameter_count']
            model.load_state_dict(saved['model'],strict=True);model.eval()
            positive=[];negative=[];rows=np.flatnonzero(val)
            with torch.inference_mode():
                for start in range(0,len(rows),settings['batch_size']):
                    batch=rows[start:start+settings['batch_size']];indices=edges[batch];valid=indices>=0;safe=np.maximum(indices,0)
                    ages=np.where(valid,prefix['timestamps'][ids[batch],None,None]-prefix['timestamps'][safe],0).astype(np.float32)
                    features=prefix['features'][safe]*valid[...,None]
                    p,q=model(*(torch.from_numpy(x.copy()) for x in [features,ages,valid,phi[batch],elapsed[batch]]))
                    positive.append(p['prob'].numpy());negative.append(q['prob'].numpy())
            p,q=np.concatenate(positive),np.concatenate(negative)
            with np.load(prediction,allow_pickle=False) as original:
                np.testing.assert_array_equal(original['event_ids'],ids[val])
                np.testing.assert_allclose(p,original['positive'],rtol=2e-4,atol=2e-5)
                np.testing.assert_allclose(q,original['negative'],rtol=2e-4,atol=2e-5)
                error=float(max(np.max(np.abs(p-original['positive'])),np.max(np.abs(q-original['negative']))))
            records.append(dict(seed=seed,model=kind,query_count=len(rows),max_absolute_probability_error=error,
                                checkpoint_sha256=sha256(checkpoint),prediction_sha256=sha256(prediction)))
            print(f'Cadence inference verified dataset={result["dataset"]} seed={seed} model={kind}',flush=True)
    report=dict(status='PASS',dataset=result['dataset'],config_path=settings['config_path'],per_model=records,
                source_bundle_sha256=sha256(attempt/'source.zip'),independent_train_validation_history=True,
                independent_cadence_features=True,full_retained_validation_checkpoint_inference=True,
                full_training_reproduction=False,fresh_dependency_environment=False,new_model_fit=False,
                test_candidates_queried=False,test_scores_generated=False,test_clock_extracted=False,
                rtol=2e-4,atol=2e-5,elapsed_seconds=time.monotonic()-started,auditor_sha256=sha256(Path(__file__)),
                history_helper_sha256=sha256(Path(__file__).with_name('audit_hard_checkpoint_replay.py')),
                local_environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu'),
                original_environment=result['environment'],publication_eligible=False,
                limitations='Reuses prepared corpus and retained checkpoints with current host dependencies. No optimization replay, fresh candidates, causal identification or scientific admission.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();replay(args.attempt,args.output)
