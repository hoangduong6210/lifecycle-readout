"""State/clock distinction, primary capacity match and validation-only pipeline."""
from pathlib import Path
import sys
import tomllib
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout import event_coupling_study as study
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competitive_event_stream import event_frozen_exposure
ROOT=Path(__file__).resolve().parents[1]


def settings():
    config=tomllib.loads((ROOT/'configs/event_coupling_v1.toml').read_text())
    config.update(hidden_dim=8,time_dim=4,max_neighbors=3)
    return config


def model(kind):
    torch.manual_seed(101)
    return study.EventCouplingControl(kind,2,settings(),np.zeros(10),np.ones(10),2.,3.)


def test_primary_exact_capacity_match_and_shared_base_initialization():
    models={kind:model(kind) for kind in study.ARMS}
    common=models['history_only']
    common_count=sum(p.numel() for p in common.parameters())
    expected=dict(history_only=0,dose_mlp=13,scalar_clock=10,phase_competition=13,shuffled_phase=13,activity_mlp=13)
    x=torch.ones(2,3,3,2);dt=torch.ones(2,3,3);mask=torch.ones(2,3,3,dtype=torch.bool)
    phi=torch.ones(2,2,10);elapsed=torch.ones(2,2)
    predictions=[]
    for kind,m in models.items():
        assert sum(p.numel() for p in m.parameters())-common_count==expected[kind]
        for module in ('base','history_head'):
            for k,v in getattr(m,module).state_dict().items():
                torch.testing.assert_close(v,getattr(common,module).state_dict()[k])
        out=m(x,dt,mask,phi,elapsed);predictions.append(out)
        for pred in out:torch.testing.assert_close(pred['logits'],pred['base_logits'])
    for out in predictions[1:]:
        for a,b in zip(out,predictions[0]):torch.testing.assert_close(a['prob'],b['prob'])


def test_exact_state_residual_derivatives_and_proportional_equivalence():
    phase=model('phase_competition').double();scalar=model('scalar_clock').double()
    phase.amplitude.data.fill_(.7);scalar.amplitude.data.fill_(.7)
    t=torch.tensor([0.,.1,2.],dtype=torch.float64,requires_grad=True)
    e=torch.tensor([0.,.3,3.],dtype=torch.float64,requires_grad=True)
    r,z=phase.dose_response(t,e);rs,zs=scalar.dose_response(t,e)
    torch.testing.assert_close(r,rs,rtol=2e-7,atol=1e-8)
    torch.testing.assert_close(z,zs,rtol=2e-7,atol=1e-8)
    dt,de=torch.autograd.grad(r.sum(),(t,e))
    assert (dt<=0).all() and (de<=0).all()
    torch.testing.assert_close(z.sum(-1),torch.ones_like(t))
    assert r[0]==0
    phase.raw_event.data[0]+=1
    r2,z2=phase.dose_response(t,e)
    assert not torch.allclose(z2[1:],zs[1:])


def test_units_fit_only_training_and_feature_masks_preserve_history():
    n=480
    data=dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,timestamps=np.floor(np.arange(n)*11./30)*30,features=np.zeros((n,2),dtype=np.float32))
    stream=build_candidate_stream(**data,unseen_ratio=0,negative_sampling='matched_historical',max_neighbors=3)
    arrays=event_frozen_exposure(data,stream,dimensions=8,profile_seed=17)
    full=study.event_stream(stream,arrays,'phase_competition')
    baseline=study.training_units(full)
    full.elapsed_times=full.elapsed_times.copy();full.pair_features=full.pair_features.copy()
    full.elapsed_times[full.split!=0]=1e20;full.pair_features[full.split!=0,:,7]=50
    assert study.training_units(full)==baseline
    activity=study.event_stream(stream,arrays,'activity_mlp')
    assert not activity.pair_features[:,:,7].any()
    for kind in study.ARMS:
        x=study.event_stream(stream,arrays,kind)
        np.testing.assert_array_equal(x.pair_features[:,:,:6],stream.pair_features)
        np.testing.assert_array_equal(x.event_ids,stream.event_ids)
        np.testing.assert_array_equal(x.negative_destinations,stream.negative_destinations)


def test_validation_only_synthetic_pipeline_and_cpu_guard(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    n=480
    data=dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,timestamps=np.floor(np.arange(n)*11./30)*30,features=np.zeros((n,2),dtype=np.float32))
    cfg=settings();cfg.update(seeds=[101],epochs=1,profile_dimensions=8)
    protocol=tomllib.loads((ROOT/cfg['protocol']).read_text());protocol['inductive_unseen_node_ratio']=0
    score=study.score
    def only_val(model,rows,stream,*args):
        assert (stream.split[rows]==1).all()
        return score(model,rows,stream,*args)
    monkeypatch.setattr(study,'score',only_val)
    result=study.run_event_coupling('wikipedia',cfg,protocol,output_dir=tmp_path/'smoke',device='cpu',smoke=True,data_override=data)
    assert not result['test_scored'] and not result['complete_declared_seeds']
    assert len(result['per_seed_runs'][0]['models'])==6
    assert set(result['paired_validation_ap'])==set(study.CONTRASTS)
    with pytest.raises(ValueError):study.run_event_coupling('wikipedia',cfg,protocol,output_dir=tmp_path/'guard',device='cpu')
