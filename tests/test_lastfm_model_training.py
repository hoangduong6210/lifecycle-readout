from pathlib import Path
import importlib.util,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_model_training import run_lastfm_model


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_complete_synthetic_optimizer_selection_and_independent_checkpoint_audit(tmp_path):
    stream,extra,s,data=load(ROOT/'tests/test_lastfm_model.py','model_fixture').setup();s=dict(s,epochs=2,seeds=[101],batch_size=32,hidden_dim=16,time_dim=4,artist_embedding_dim=4);data['training_artist_vocabulary']=np.array([10]);protocol=dict(protocol_id='synthetic-only',seeds=[101]);output=tmp_path/'fit';result=run_lastfm_model(s,protocol,output,device='cpu',smoke_data=data,smoke_streams={101:(stream,extra)})
    assert len(result['per_seed_runs'][0]['models'])==6 and result['smoke'] and not result['test_scored']
    audit=load(ROOT/'scripts/audit_lastfm_model.py','checkpoint_audit').audit_models(output,s,data,{101:(stream,extra)})
    assert audit['status']=='PASS' and len(audit['records'])==6


def test_real_cpu_optimizer_is_rejected_before_inputs_are_loaded(tmp_path):
    with pytest.raises(ValueError,match='owned Colab CUDA'):run_lastfm_model({}, {},tmp_path/'invalid',device='cpu')


def test_initial_common_restoration_rejects_wrong_digest_and_noncommon_keys():
    import torch
    from lifecycle_readout.lastfm_model import RecordedArtistControl
    from lifecycle_readout.lastfm_model_training import common_identity
    stream,extra,settings,data=load(ROOT/'tests/test_lastfm_model.py','initial_fixture').setup()
    torch.manual_seed(101)
    model=RecordedArtistControl('phase_competition',settings,np.zeros(11),np.ones(11),1.,1.,artist_vocabulary=10,seed=101)
    initial={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base.','history_head.','artist_embedding.'))}
    expected=common_identity(model);restore=load(ROOT/'scripts/audit_lastfm_model.py','initial_audit').restore_initial_common
    with torch.no_grad():model.base.empty_node.add_(1.)
    assert common_identity(model)!=expected
    restore(model,initial,expected)
    assert common_identity(model)==expected
    with pytest.raises(AssertionError,match='digest mismatch'):restore(model,initial,'0'*64)
    with pytest.raises(AssertionError,match='complete common'):restore(model,dict(initial,amplitude=model.amplitude.clone()),expected)
