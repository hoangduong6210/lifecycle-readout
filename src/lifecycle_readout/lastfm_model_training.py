"""Fixed recorded-artist comparison; real fitting requires owned CUDA runtime."""
from pathlib import Path
import hashlib,json,platform,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .benchmark import save_json,sha256,tensor_batch
from .causal_stream import CandidateStream
from .event_coupling_study import training_units,CONTRASTS
from .hard_controls import score,report_metrics
from .lastfm_model import RecordedArtistControl,artist_arm_stream


def load_stream(path):
    with np.load(path,allow_pickle=False) as z:
        fields={k:z[k] for k in CandidateStream.__dataclass_fields__}
    for k in ('train_end','validation_end'):fields[k]=float(fields[k])
    return CandidateStream(**fields)


def common_identity(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('base.','history_head.','artist_embedding.')):
            h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_arm(kind,seed,stream,data,settings,device,output):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    train=np.flatnonzero(stream.split==0);val=np.flatnonzero(stream.split==1)
    if not len(train) or not len(val):raise ValueError('Empty fixed candidate split')
    values=stream.pair_features[train].reshape(-1,settings['pair_feature_dim']);mean=values.mean(0);std=values.std(0).clip(min=1e-5);unit,exposure_unit=training_units(stream);vocabulary=int(data['training_artist_vocabulary'][0])
    model=RecordedArtistControl(kind,settings,mean,std,unit,exposure_unit,artist_vocabulary=vocabulary,seed=seed).to(device);common_hash=common_identity(model);optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay']);rng=np.random.RandomState(seed);history=[];best=-float('inf');checkpoint=output/f'{kind}_seed{seed}.pt';start=time.monotonic()
    for epoch in range(1,settings['epochs']+1):
        model.train();losses=[];order=rng.permutation(train)
        for i in range(0,len(order),settings['batch_size']):
            predictions=model(*tensor_batch(stream,order[i:i+settings['batch_size']],data,device));loss=predictions[0]['logits'].new_zeros(())
            for prediction,target in zip(predictions,(1.,0.)):
                truth=torch.full_like(prediction['logits'],target);loss=loss+F.binary_cross_entropy_with_logits(prediction['logits'],truth)+settings['affinity_auxiliary_weight']*F.binary_cross_entropy_with_logits(prediction['base_logits'],truth)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            optimizer.zero_grad();loss.backward();optimizer.step();model.project_parameters_();losses.append(float(loss.detach()))
        pos,neg=score(model,val,stream,data,settings,device);metrics=report_metrics(pos,neg);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics))
        if metrics['ap']>best:
            best=metrics['ap'];torch.save(dict(model=model.state_dict(),kind=kind,seed=seed,selected_epoch=epoch,settings=settings,phi_mean=mean.tolist(),phi_std=std.tolist(),time_unit=unit,exposure_unit=exposure_unit,artist_vocabulary=vocabulary,initial_common_sha256=common_hash),checkpoint)
        print(f"lastfm {kind} seed={seed} epoch={epoch} validation_AP={metrics['ap']:.6f}",flush=True)
    saved=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(saved['model']);pos,neg=score(model,val,stream,data,settings,device);predictions=output/f'{kind}_seed{seed}_predictions.npz';np.savez_compressed(predictions,positive=pos,negative=neg,event_ids=stream.event_ids[val],negative_destinations=stream.negative_destinations[val]);metrics=report_metrics(pos,neg);metrics['pair_accuracy']=float(np.mean((pos>neg)+.5*(pos==neg)));parameters=sum(p.numel() for p in model.parameters());branch=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('base.','history_head.','artist_embedding.')))
    report=dict(kind=kind,seed=seed,selected_epoch=saved['selected_epoch'],validation_history=history,all_validation=metrics,evaluation_split='validation',test_scored=False,parameter_count=parameters,branch_parameter_count=branch,elapsed_seconds=time.monotonic()-start,initial_common_sha256=common_hash,time_unit=unit,exposure_unit=exposure_unit,amplitude=float(model.amplitude.detach()),checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(predictions))
    if model.phase_kind:report['state_parameters']=dict(clock_rates=F.softplus(model.raw_clock).detach().cpu().tolist(),event_rates=(F.softplus(model.raw_event)*F.softplus(model.raw_clock) if kind=='scalar_clock' else F.softplus(model.raw_event)).detach().cpu().tolist(),readout_gaps=F.softplus(model.raw_gap).detach().cpu().tolist())
    save_json(output/f'{kind}_seed{seed}.json',report);return report


def run_lastfm_model(settings,protocol,output,*,device='cuda',smoke_data=None,smoke_streams=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);smoke=smoke_data is not None
    if device!='cuda' and not smoke:raise ValueError('Real optimizer requires owned Colab CUDA')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no training fallback')
    root=Path(__file__).resolve().parents[2]
    if smoke:
        if len(smoke_data['features'])>1000 or smoke_streams is None:raise ValueError('Small synthetic override required')
        data=smoke_data
    else:
        for item in settings['input_files']:
            target=(root/item['path']).resolve();assert target.is_relative_to(root) and sha256(target)==item['sha256']
        prepared=root/settings['prepared_directory'];data=dict(np.load(prepared/'data.npz',allow_pickle=False))
    result=dict(protocol_id=protocol['protocol_id'],dataset='lastfm-original',settings=settings,evaluation_split='validation',test_scored=False,publication_eligible=False,smoke=smoke,per_seed_runs=[],environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device))
    for seed in settings['seeds']:
        if smoke:stream,extra=smoke_streams[seed]
        else:
            stream=load_stream(prepared/f'seed{seed}/candidate_stream.npz');extra=dict(np.load(prepared/f'seed{seed}/artist_features.npz',allow_pickle=False))
        folder=output/f'seed{seed}';folder.mkdir();entry=dict(seed=seed,candidate_digest=stream.digest(),coverage={name:int((stream.split==split).sum()) for split,name in [(0,'train'),(1,'validation')]},models={});common=None
        for kind in settings['models']:
            arm=artist_arm_stream(stream,extra,kind);report=train_arm(kind,seed,arm,data,settings,torch.device(device),folder)
            if common is None:common=report['initial_common_sha256']
            if report['initial_common_sha256']!=common:raise ValueError('Common initial tensors differ across declared arms')
            entry['models'][kind]=report;save_json(output/'partial_results.json',dict(result,current_seed=entry))
        a,b=(entry['models'][k] for k in ['phase_competition','dose_mlp']);assert a['parameter_count']==b['parameter_count'] and a['branch_parameter_count']==b['branch_parameter_count']==13
        result['per_seed_runs'].append(entry);save_json(output/'partial_results.json',result)
    result['paired_validation_ap']={}
    for name,(a,b) in CONTRASTS.items():
        values=[x['models'][a]['all_validation']['ap']-x['models'][b]['all_validation']['ap'] for x in result['per_seed_runs']];result['paired_validation_ap'][name]=dict(treatment=a,control=b,values=values,mean=float(np.mean(values)),std_ddof1=float(np.std(values,ddof=1)) if len(values)>1 else None)
    result['complete_declared_seeds']=settings['seeds']==protocol['seeds'];save_json(output/'results.json',result);return result
