"""Check clock history, fallback fitting and the actual model integration."""
from pathlib import Path
import sys
import tomllib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.pair_clock import augment_pair_clock
from lifecycle_readout import hard_controls


def fixture():
    n = 600
    return dict(sources=np.arange(n) % 3, destinations=10 + (np.arange(n) // 3) % 8,
                timestamps=np.arange(n, dtype=float) * 11,
                features=np.random.RandomState(4).normal(size=(n, 2)).astype(np.float32))


def test_independent_binary_search_audit_rejects_changed_clock_and_fallback():
    import importlib.util
    from dataclasses import replace
    spec=importlib.util.spec_from_file_location('clock_input_auditor',ROOT/'scripts/audit_pair_clock_inputs.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    data=fixture()
    stream=build_candidate_stream(**data,unseen_ratio=.1,negative_sampling='matched_historical')
    augmented,metadata,raw=augment_pair_clock(data,stream,tau0_seconds=1.)
    assert auditor.independently_check(data,stream,augmented,metadata,raw)['status']=='PASS'
    changed=augmented.pair_features.copy();changed[0,0,8]+=.1
    with pytest.raises(AssertionError):
        auditor.independently_check(data,stream,replace(augmented,pair_features=changed),metadata,raw)
    with pytest.raises(AssertionError):
        auditor.independently_check(data,stream,augmented,dict(metadata,fallback_seconds=metadata['fallback_seconds']*2),raw)


def test_cadence_uses_only_past_and_preserves_candidate_support():
    data = fixture()
    stream = build_candidate_stream(**data, unseen_ratio=.1, negative_sampling='matched_historical')
    new, metadata, raw = augment_pair_clock(data, stream, tau0_seconds=1.)
    np.testing.assert_array_equal(new.event_ids, stream.event_ids)
    np.testing.assert_array_equal(new.negative_destinations, stream.negative_destinations)
    np.testing.assert_array_equal(new.pair_features[:, :, :6], stream.pair_features)
    # Direct raw scans independently check every train/validation query.
    for row, event in enumerate(stream.event_ids):
        if stream.split[row] == 2:
            assert not raw['queried'][row]
            assert np.isnan(raw['mean_interval_seconds'][row]).all()
            assert (new.pair_features[row, :, 6:] == 0).all()
            continue
        for slot, dst in enumerate([data['destinations'][event], stream.negative_destinations[row]]):
            mask = ((data['sources'] == data['sources'][event]) & (data['destinations'] == dst)
                    & (data['timestamps'] < data['timestamps'][event]))
            mask[stream.excluded_training_ids] = False
            ts = data['timestamps'][mask]
            assert raw['prior_count'][row, slot] == len(ts)
            if len(ts) >= 2:
                assert raw['mean_interval_seconds'][row, slot] == (ts[-1] - ts[0])/(len(ts)-1)
    usable = raw['mean_interval_seconds'][stream.split == 0]
    assert metadata['fallback_seconds'] == np.median(usable[np.isfinite(usable)])
    future = {k: v.copy() for k, v in data.items()}
    future['timestamps'][future['timestamps'] > stream.validation_end] += 1e8
    again, other, _ = augment_pair_clock(future, stream, tau0_seconds=1.)
    np.testing.assert_array_equal(again.pair_features, new.pair_features)
    assert other == metadata


def test_development_configs_keep_sampling_and_clock_controls_separate():
    configs=[tomllib.loads((ROOT/f'configs/pair_cadence_{name}_v1.toml').read_text())
             for name in ['original','features','relative']]
    for key in ['seeds','epochs','batch_size','hard_time_unit','decay_time_unit','evaluation_split']:
        assert configs[0][key] == configs[1][key] == configs[2][key]
    assert all(c['pair_feature_dim']==9 for c in configs)
    assert configs[0]['mask_pair_clock_features'] is True
    assert configs[1]['pair_clock_mode']=='absolute' and configs[2]['pair_clock_mode']=='relative'


def test_zero_span_and_singleton_are_missing_not_fast_clocks():
    # Two same-time prior events for pair (0,10); one prior event for (0,11).
    # Pair (1,12) has a valid training cadence of 4 seconds for fallback fitting.
    data = dict(sources=np.array([0,0,0,1,1,1,0]), destinations=np.array([10,10,11,12,12,12,10]),
                timestamps=np.array([1.,1.,1.,2.,6.,10.,11.]))
    from dataclasses import dataclass
    @dataclass
    class Stream:
        event_ids: np.ndarray
        negative_destinations: np.ndarray
        pair_features: np.ndarray
        elapsed_times: np.ndarray
        split: np.ndarray
        excluded_training_ids: np.ndarray
        validation_end: float
    phi=np.zeros((2,2,6),dtype=np.float32)
    phi[0,:,2]=np.log1p(2);phi[1,:,2]=np.log1p([2,1])
    stream=Stream(np.array([5,6]),np.array([12,11]),phi,np.array([[4.,4.],[10.,10.]]),
                  np.array([0,1]),np.array([],dtype=int),11.)
    new, meta, raw=augment_pair_clock(data,stream,tau0_seconds=1.)
    assert meta['fallback_seconds']==4.
    assert not raw['cadence_available'][1].any()
    np.testing.assert_allclose(new.pair_features[1,:,8],np.log1p(10/5))
    # Removing training cadence must fail instead of fitting on validation.
    stream.split=np.array([1,1])
    with pytest.raises(ValueError,match='No estimable training'):
        augment_pair_clock(data,stream,tau0_seconds=1.)


def test_same_features_and_parameters_isolate_clock_and_preserve_neutral_residual():
    settings=tomllib.loads((ROOT/'configs/q1_development_v1.toml').read_text())
    settings.update(hidden_dim=8,time_dim=4,max_neighbors=3,pair_feature_dim=9,decay_time_unit=1000.)
    models=[]
    for clock in ['absolute','relative']:
        torch.manual_seed(9)
        models.append(hard_controls.HistoryControl('residual_scalar',2,dict(settings,pair_clock_mode=clock),np.zeros(9),np.ones(9)).eval())
    for key,value in models[0].state_dict().items():
        torch.testing.assert_close(value,models[1].state_dict()[key])
    x=torch.zeros(2,3,3,2);dt=torch.ones(2,3,3);mask=torch.ones(2,3,3,dtype=torch.bool)
    phi=torch.zeros(2,2,9);phi[:,:,8]=.1
    elapsed=torch.full((2,2),1000.)
    for model in models:
        out=model(x,dt,mask,phi,elapsed)
        assert torch.equal(out[0]['logits'],out[0]['base_logits'])
        model.residual.amplitude.data.fill_(1.)
    absolute=models[0](x,dt,mask,phi,elapsed)[0]
    relative=models[1](x,dt,mask,phi,elapsed)[0]
    assert (relative['residual']>absolute['residual']).all()


@pytest.mark.parametrize('mode,masked',[('absolute',False),('relative',False),('absolute',True)])
def test_tiny_cadence_fit_scores_validation_only(tmp_path,monkeypatch,mode,masked):
    torch.set_num_threads(1)
    monkeypatch.setattr(hard_controls,'load_dataset',lambda *_:(fixture(),{'dataset':'synthetic'}))
    settings=tomllib.loads((ROOT/'configs/q1_development_v1.toml').read_text())
    settings.update(seeds=[101],epochs=1,hidden_dim=8,time_dim=4,max_neighbors=3,batch_size=64,
        models=['base_stats','residual_scalar','residual_mixture','mlp_residual'],
        pair_feature_dim=9,pair_clock_mode=mode,pair_clock_tau0_seconds=1.,mask_pair_clock_features=masked)
    protocol=tomllib.loads((ROOT/settings['protocol']).read_text());protocol['inductive_unseen_node_ratio']=0
    original=hard_controls.score
    def only_validation(model,rows,stream,*args):
        assert (stream.split[rows]==1).all()
        if masked:
            assert (stream.pair_features[:,:,6:]==0).all()
        return original(model,rows,stream,*args)
    monkeypatch.setattr(hard_controls,'score',only_validation)
    result=hard_controls.run_controls('wikipedia',settings,protocol,data_dir=tmp_path,output_dir=tmp_path/'out',device='cpu',smoke=True)
    assert result['test_scored'] is False and not result['publication_eligible']
    import json
    clock=json.loads((tmp_path/'out/seed101/pair_clock.json').read_text())
    assert clock['test_clocks_constructed'] is False
    import importlib.util
    spec=importlib.util.spec_from_file_location('saved_clock_auditor',ROOT/'scripts/audit_pair_clock_inputs.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    with np.load(tmp_path/'out/seed101/candidate_identity.npz') as z:
        candidates={k:z[k] for k in z.files}
    protocol['tau0_seconds']=1.
    assert auditor.verify_saved_clock(fixture(),candidates,tmp_path/'out/seed101',settings,protocol)['status']=='PASS'
