from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_intake import summarize_history


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('bad',[False,True])
def test_full_artist_audit_pipeline_and_input_identity_rejection(tmp_path,bad):
    f=load(ROOT/'tests/test_lastfm_artist_history.py','artist_fixture').fixture()
    npz=tmp_path/'selected_training_events.npz';np.savez_compressed(npz,**{k:f[k].to_numpy() for k in f})
    parent=tmp_path/'results.json';parent.write_text(json.dumps(dict(summary=summarize_history(f))))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'artifact_manifest.json').write_text(json.dumps({npz.name:'0'*64 if bad else digest(npz),parent.name:digest(parent)}))
    worker=load(ROOT/'scripts/audit_lastfm_artist_history.py','artist_worker');out=tmp_path/'out'
    if bad:
        with pytest.raises(AssertionError):worker.run(npz,parent,ROOT/'protocols/lastfm_artist_history_v1.toml',out)
        assert not (out/'artist_history_features.npz').exists() and (out/'failure.json').exists()
    else:
        worker.run(npz,parent,ROOT/'protocols/lastfm_artist_history_v1.toml',out)
        result=json.loads((out/'results.json').read_text());assert result['status']=='PASS' and result['parent_denominators_match'] and result['real_anchor_count']>0
