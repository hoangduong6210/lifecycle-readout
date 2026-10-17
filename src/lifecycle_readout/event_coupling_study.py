"""Prospective exact-state competition comparison; all prior studies unchanged."""
from dataclasses import replace
from pathlib import Path
import io
import json
import platform
import random
import time
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .benchmark import load_dataset,save_json,sha256,tensor_batch
from .causal_stream import build_candidate_stream
from .competitive_event_stream import event_frozen_exposure
from .competitive_lifecycle import competitive_step
from .hard_controls import HistoryControl,report_metrics,score


ARMS=('history_only','dose_mlp','scalar_clock','phase_competition','shuffled_phase','activity_mlp')


def event_stream(stream,arrays,kind):
    if kind not in ARMS:raise ValueError('Unknown event-coupling arm')
    raw=arrays['raw'].copy()
    if kind=='activity_mlp':raw[:,:,1]=0
    if kind=='shuffled_phase':raw[:,:,1]=arrays['shuffled_exposure']
    for index in (0,1,3):raw[:,:,index]=np.log1p(raw[:,:,index])
    raw[~arrays['queried']]=0
    return replace(stream,pair_features=np.concatenate((stream.pair_features,raw.astype(np.float32)),axis=-1))


def training_units(stream):
    rows=stream.split==0
    clock=stream.elapsed_times[rows].ravel()
    exposure=np.expm1(stream.pair_features[rows,:,7]).ravel()
    positive=exposure[exposure>0]
    if not (clock>0).all() or not len(clock):raise ValueError('Recurrent training times required')
    return float(np.median(clock)),float(np.median(positive)) if len(positive) else 1.


class EventCouplingControl(HistoryControl):
    """Shared history scorer plus a fixed-reset exact-state residual or control.

    Every own event group resets a recurrent edge to the uniform K-simplex.
    Constant learned phase rates propagate it under physical time and event dose.
    r=a*(w.z-w.z_init), a>=0, gaps>=0. Base/context may depend on time/exposure,
    so this certifies only the residual at fixed base, not the whole predictor.
    The primary tiny-MLP comparator has exactly the same branch parameter count.
    """
    def __init__(self,kind,feat_dim,settings,phi_mean,phi_std,time_unit,exposure_unit):
        if kind not in ARMS:raise ValueError('Unknown event-coupling arm')
        super().__init__('base_stats',feat_dim,settings,phi_mean,phi_std)
        self.kind=kind;self.time_unit=float(time_unit);self.exposure_unit=float(exposure_unit)
        if min(self.time_unit,self.exposure_unit)<=0 or not np.isfinite([self.time_unit,self.exposure_unit]).all():
            raise ValueError('Finite positive training scales required')
        self.phase_kind=kind in ('scalar_clock','phase_competition','shuffled_phase')
        if self.phase_kind:
            rates=torch.tensor(settings['initial_phase_rates'],dtype=torch.float32)
            gaps=torch.tensor(settings['initial_readout_gaps'],dtype=torch.float32)
            if rates.shape!=(4,) or gaps.shape!=(4,) or (rates<=0).any() or (gaps<=0).any():
                raise ValueError('Four positive phase rates and gaps required')
            self.raw_clock=nn.Parameter(torch.log(torch.expm1(rates)))
            self.raw_gap=nn.Parameter(torch.log(torch.expm1(gaps)))
            self.raw_event=nn.Parameter(torch.log(torch.expm1(rates))) if kind!='scalar_clock' else nn.Parameter(torch.log(torch.expm1(torch.ones(()))))
            self.amplitude=nn.Parameter(torch.zeros(()))
        elif kind in ('dose_mlp','activity_mlp'):
            # 9 first-layer + 3 final-layer weights + scalar amplitude = 13.
            self.dose_head=nn.Sequential(nn.Linear(2,3),nn.GELU(),nn.Linear(3,1,bias=False))
            self.amplitude=nn.Parameter(torch.zeros(()))
        else:self.register_buffer('amplitude',torch.zeros(()))

    def dose_response(self,clock,exposure):
        if self.kind=='history_only':return torch.zeros_like(clock),None
        if not self.phase_kind:
            inputs=torch.stack((torch.log1p(clock),torch.log1p(exposure)),-1)
            return self.amplitude*self.dose_head(inputs).squeeze(-1),None
        d=F.softplus(self.raw_clock)
        gamma=F.softplus(self.raw_event)*d if self.kind=='scalar_clock' else F.softplus(self.raw_event)
        initial=clock.new_full((len(clock),5),1/5)
        state=competitive_step(initial,clock,exposure,d,gamma)
        residual=self.amplitude.clamp(min=0)*((state[:,1:]-initial[:,1:])*F.softplus(self.raw_gap)).sum(-1)
        return residual,state

    def forward(self,x,dt,mask,phi,pair_dt):
        outputs=[]
        for dest,slot in ((1,0),(2,1)):
            h,logit=self.base(x[:,0],dt[:,0],mask[:,0],x[:,dest],dt[:,dest],mask[:,dest])
            inputs=torch.cat((h,(phi[:,slot]-self.phi_mean)/self.phi_std),-1)
            base=logit+self.history_head(inputs).squeeze(-1)
            residual,state=self.dose_response(pair_dt[:,slot]/self.time_unit,torch.expm1(phi[:,slot,7])/self.exposure_unit)
            logits=base+residual
            outputs.append(dict(base_logits=base,logits=logits,prob=torch.sigmoid(logits),residual=residual,state=state))
        return outputs

    @torch.no_grad()
    def project_parameters_(self):
        if self.phase_kind:self.amplitude.clamp_(min=0)


def train_event_coupling(kind, seed, stream, data, settings, device, output):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    training, validation = (np.flatnonzero(stream.split == phase) for phase in (0, 1))
    if not len(training) or not len(validation):
        raise ValueError("No supported training or validation rows; no fallback")
    values = stream.pair_features[training].reshape(-1, settings["pair_feature_dim"])
    mean, std = values.mean(0), values.std(0).clip(min=1e-5)
    time_unit,exposure_unit=training_units(stream)
    model = EventCouplingControl(kind, data["features"].shape[1], settings, mean, std,time_unit,exposure_unit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = np.random.RandomState(seed)
    checkpoint = output/f"{kind}_seed{seed}.pt"
    history, best = [], -float("inf")
    started = time.monotonic()
    for epoch in range(1, settings["epochs"]+1):
        model.train(); losses = []
        shuffled_rows = rng.permutation(training)
        for start in range(0, len(training), settings["batch_size"]):
            predictions = model(*tensor_batch(stream, shuffled_rows[start:start+settings["batch_size"]], data, device))
            loss = torch.zeros((), device=device)
            for prediction, target in zip(predictions, (1., 0.)):
                truth = torch.full_like(prediction["logits"], target)
                loss = loss+F.binary_cross_entropy_with_logits(prediction["logits"], truth)
                loss = loss+settings["affinity_auxiliary_weight"]*F.binary_cross_entropy_with_logits(prediction["base_logits"], truth)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite competition objective")
            optimizer.zero_grad(); loss.backward(); optimizer.step(); model.project_parameters_()
            losses.append(float(loss.detach()))
        positive, negative = score(model, validation, stream, data, settings, device)
        metrics = report_metrics(positive, negative)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics))
        if metrics["ap"] > best:
            best = metrics["ap"]
            torch.save(dict(model=model.state_dict(), selected_epoch=epoch, kind=kind, seed=seed,
                            settings=settings, phi_mean=mean.tolist(), phi_std=std.tolist(),time_unit=time_unit,exposure_unit=exposure_unit), checkpoint)
        print(f"event_coupling {kind} seed={seed} epoch={epoch} validation_AP={metrics['ap']:.6f}", flush=True)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    positive, negative = score(model, validation, stream, data, settings, device)
    prediction_file = output/f"{kind}_seed{seed}_predictions.npz"
    np.savez_compressed(prediction_file, positive=positive, negative=negative,
                        event_ids=stream.event_ids[validation], negative_destinations=stream.negative_destinations[validation])
    validation_metrics = report_metrics(positive, negative)
    validation_metrics["pair_accuracy"] = float(np.mean((positive > negative)+.5*(positive == negative)))
    report = dict(kind=kind, seed=seed, selected_epoch=saved["selected_epoch"], validation_history=history,
                  all_validation=validation_metrics, evaluation_split="validation", test_scored=False,
                  parameter_count=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.monotonic()-started,
                  amplitude=float(model.amplitude.detach()), time_unit=time_unit, exposure_unit=exposure_unit, checkpoint_sha256=sha256(checkpoint),
                  predictions_sha256=sha256(prediction_file))
    report['branch_parameter_count']=report['parameter_count']-sum(p.numel() for module in (model.base,model.history_head) for p in module.parameters())
    report['state_parameters']=(dict(clock_rates=F.softplus(model.raw_clock).detach().cpu().tolist(),event_rates=(F.softplus(model.raw_event)*F.softplus(model.raw_clock) if kind=='scalar_clock' else F.softplus(model.raw_event)).detach().cpu().tolist(),readout_gaps=F.softplus(model.raw_gap).detach().cpu().tolist()) if model.phase_kind else None)
    save_json(output/f"{kind}_seed{seed}.json", report)
    return report


CONTRASTS={'primary_shape':('phase_competition','dose_mlp'),
           'phase_specific_rates':('phase_competition','scalar_clock'),
           'profile_alignment':('phase_competition','shuffled_phase'),
           'profile_information':('dose_mlp','activity_mlp'),
           'residual_vs_history':('phase_competition','history_only')}


def run_event_coupling(dataset,settings,protocol,*,output_dir,device='cuda',smoke=False,data_override=None):
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh attempt required')
    output.mkdir(parents=True,exist_ok=True)
    if device!='cuda' and not smoke:raise ValueError('Real fitting requires owned Colab CUDA')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA; no training fallback')
    root=Path(__file__).resolve().parents[2]
    if smoke:
        if data_override is None or len(data_override['sources'])>settings['cpu_smoke_event_limit']:
            raise ValueError('Small synthetic data override required')
        data=data_override;identity={'dataset':'synthetic'}
    else:
        for item in settings['input_files']:
            if sha256(root/item['path'])!=item['sha256']:raise ValueError('Frozen input mismatch')
        with zipfile.ZipFile(root/settings['data_archive']) as archive:
            rel='resources/corpora/wikipedia_causal_v2.npz';raw=archive.read(rel)
            manifest=json.loads(archive.read('source_manifest.json'))
            import hashlib
            assert hashlib.sha256(raw).hexdigest()==manifest[rel]==settings['prepared_data_sha256']
            data=dict(np.load(io.BytesIO(raw),allow_pickle=False))
        identity={'dataset':dataset,'derived_sha256':settings['prepared_data_sha256'],'data_archive':settings['data_archive']}
    result=dict(protocol_id=protocol['protocol_id'],dataset=dataset,settings=settings,data=identity,per_seed_runs=[],evaluation_split='validation',test_scored=False,publication_eligible=False,smoke=smoke,environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device))
    for seed in settings['seeds']:
        stream=build_candidate_stream(**data,seed=seed,split=protocol['split_chronological'],split_seed=settings['split_seed'],unseen_ratio=protocol['inductive_unseen_node_ratio'],max_neighbors=settings['max_neighbors'],history_decay=settings['history_decay'],negative_sampling='matched_historical',hard_time_unit=settings['hard_time_unit'])
        folder=output/f'seed{seed}';folder.mkdir()
        if smoke:arrays=event_frozen_exposure(data,stream,dimensions=settings['profile_dimensions'],profile_seed=settings['profile_seed'])
        else:
            arrays=dict(np.load(root/settings['feature_directory']/f'seed{seed}_features.npz'))
            parent=dict(np.load(root/settings['candidate_directory']/f'seed{seed}/candidate_identity.npz'))
            for key in ('event_ids','negative_destinations','split','pair_features','elapsed_times','inductive','excluded_training_ids','unseen_nodes'):
                np.testing.assert_array_equal(getattr(stream,key),parent[key])
            for key in ('event_ids','negative_destinations','split'):np.testing.assert_array_equal(arrays[key],getattr(stream,key))
            np.testing.assert_allclose(arrays['elapsed'][stream.split<=1],stream.elapsed_times[stream.split<=1],rtol=2e-6,atol=1e-7)
        assert not arrays['raw'][stream.split==2].any()
        np.savez_compressed(folder/'event_features.npz',**arrays)
        np.savez_compressed(folder/'candidate_identity.npz',event_ids=stream.event_ids,negative_destinations=stream.negative_destinations,split=stream.split,pair_features=stream.pair_features,elapsed_times=stream.elapsed_times,inductive=stream.inductive,excluded_training_ids=stream.excluded_training_ids,skipped_no_hard_support_ids=stream.skipped_no_negative_ids,unseen_nodes=stream.unseen_nodes)
        entry=dict(seed=seed,candidate_sha256=stream.digest(),models={},coverage={name:int((stream.split==phase).sum()) for phase,name in enumerate(('train','validation','structural_test'))})
        for arm in settings['models']:
            arm_stream=event_stream(stream,arrays,arm)
            entry['models'][arm]=train_event_coupling(arm,seed,arm_stream,data,settings,torch.device(device),folder)
            save_json(output/'partial_results.json',dict(result,current_seed=entry))
        result['per_seed_runs'].append(entry);save_json(output/'partial_results.json',result)
    result['complete_declared_seeds']=settings['seeds']==protocol['seeds']
    result['paired_validation_ap']={}
    for name,(treatment,control) in CONTRASTS.items():
        values=[x['models'][treatment]['all_validation']['ap']-x['models'][control]['all_validation']['ap'] for x in result['per_seed_runs']]
        result['paired_validation_ap'][name]=dict(treatment=treatment,control=control,values=values,mean=float(np.mean(values)),std_ddof1=float(np.std(values,ddof=1)) if len(values)>1 else None)
    save_json(output/'results.json',result)
    return result
