"""Synthetic archive intake and rejection before extracting unverified bytes."""
from pathlib import Path
import hashlib,importlib.util,json,tarfile,zipfile
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]


def fixture(tmp_path,monkeypatch,bad=False):
    spec=importlib.util.spec_from_file_location('intake_worker',ROOT/'scripts/run_kuairand_intake.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    data=tmp_path/'fixture';data.mkdir()
    pd.DataFrame({'user_id':np.arange(1000)}).to_csv(data/'user_features_1k.csv',index=False)
    n=6000;i=np.arange(n)
    pd.DataFrame(dict(user_id=i%1000,video_id=10+(i//1000)%4,time_ms=(i//1000)*3600000+(i%1000)*100,is_click=np.ones(n,dtype=int))).to_csv(data/'log_standard_4_08_to_4_21_1k.csv',index=False)
    pd.DataFrame(dict(video_id=[10,11,12,13],tag=['1,2','3','-1',None],author_id=[30]*4,upload_dt=['2022-01-01']*4)).to_csv(data/'video_features_basic_1k.csv',index=False)
    archive=tmp_path/'fixture.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in data.iterdir():tar.add(path,arcname='KuaiRand-1K/data/'+path.name)
    raw=archive.read_bytes()
    protocol=(ROOT/'protocols/kuairand_intake_v1.toml').read_text().replace('6b0b9c8222d67fcd4c676218edca3f1f','0'*32 if bad else hashlib.md5(raw).hexdigest())
    files={'intake_recurrence.py':(ROOT/'src/lifecycle_readout/intake_recurrence.py').read_bytes(),'protocol.toml':protocol.encode(),'README.md':b'synthetic fixture','LICENSE':b'synthetic fixture'}
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:
        for name,value in files.items():z.writestr(name,value)
        z.writestr('manifest.json',json.dumps({name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
    class Response:
        def __init__(self,data):self.content=data
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def json(self):return json.loads(self.content)
        def iter_content(self,size):
            for start in range(0,len(self.content),size):yield self.content[start:start+size]
    def get(url,**kwargs):return Response(b'{"metadata":{"license":{"id":"synthetic"}}}' if '/api/' in url else raw)
    monkeypatch.setattr(worker.requests,'get',get)
    return worker,source


def test_synthetic_intake_end_to_end(tmp_path,monkeypatch):
    worker,source=fixture(tmp_path,monkeypatch)
    worker.run(source,tmp_path/'out')
    report=json.loads((tmp_path/'out/results.json').read_text())
    assert report['selected_training_rows']==600 and report['positive_clicks']['recurrent_rows']==200
    assert len(report['sampled_user_ids'])==100
    assert not report['engineering_feasibility_gate_passed']
    assert not report['model_fit'] and not report['test_scored']
    assert (tmp_path/'out/selected_training_events.npz').is_file()
    assert json.loads((tmp_path/'kuairand-intake-complete.json').read_text())['status']=='completed'


def test_checksum_mismatch_is_retained_and_never_extracted(tmp_path,monkeypatch):
    worker,source=fixture(tmp_path,monkeypatch,bad=True)
    with pytest.raises(RuntimeError):worker.run(source,tmp_path/'out')
    assert not (tmp_path/'selected_raw').exists()
    failure=json.loads((tmp_path/'out/failure.json').read_text())
    assert 'MD5 mismatch' in failure['message']
    assert json.loads((tmp_path/'kuairand-intake-complete.json').read_text())['status']=='failed'
