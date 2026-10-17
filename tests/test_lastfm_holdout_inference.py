from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_holdout_inference import frozen_checkpoint_scores,state_identity
from lifecycle_readout.lastfm_model import RecordedArtistControl,artist_arm_stream
from lifecycle_readout.hard_controls import score


def test_every_arm_uses_saved_normalization_and_unchanged_weights(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('saved_checkpoint_fixture',ROOT/'tests/test_lastfm_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream,extra,s,data=module.setup();s=dict(s,batch_size=4,hidden_dim=16,time_dim=4,artist_embedding_dim=4)
    data['training_artist_vocabulary']=np.array([10]);rows=np.flatnonzero(stream.split==1)
    def reject_optimizer(*args,**kwargs):raise AssertionError('Inference attempted optimization')
    monkeypatch.setattr(torch.optim.Adam,'__init__',reject_optimizer)
    for kind in s['models']:
        model=RecordedArtistControl(kind,s,np.arange(11)*.07,np.arange(11)*.03+.3,9700.,17.,artist_vocabulary=10,seed=101)
        with torch.no_grad():model.amplitude.fill_(.2)
        checkpoint=tmp_path/f'{kind}.pt'
        torch.save(dict(kind=kind,seed=101,selected_epoch=3,settings=s,phi_mean=(np.arange(11)*.07).tolist(),
            phi_std=(np.arange(11)*.03+.3).tolist(),time_unit=9700.,exposure_unit=17.,artist_vocabulary=10,
            model=model.state_dict()),checkpoint)
        expected=score(model,rows,artist_arm_stream(stream,extra,kind),data,s,torch.device('cpu'))
        p,n,report=frozen_checkpoint_scores(checkpoint,kind,101,stream,extra,data,rows,torch.device('cpu'))
        np.testing.assert_array_equal(p,expected[0]);np.testing.assert_array_equal(n,expected[1])
        assert report['frozen_state_sha256']==state_identity(model) and report['selected_epoch']==3
        assert report['unchanged_state'] and not report['model_fit']
    with pytest.raises(ValueError,match='identity mismatch'):
        frozen_checkpoint_scores(checkpoint,kind,999,stream,extra,data,rows,torch.device('cpu'))
