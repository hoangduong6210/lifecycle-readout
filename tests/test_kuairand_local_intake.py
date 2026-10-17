"""Existing-archive path preserves raw bytes and checksum rejection semantics."""
from pathlib import Path
import importlib.util,hashlib,json
import pytest
ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('bad',[False,True])
def test_local_archive_processing_or_rejection_without_source_mutation(tmp_path,monkeypatch,bad):
    fixture=load(ROOT/'tests/test_kuairand_intake_pipeline.py','intake_fixtures')
    _,source=fixture.fixture(tmp_path,monkeypatch,bad=bad)
    worker=load(ROOT/'scripts/run_kuairand_local_intake.py','local_worker')
    archive=tmp_path/'fixture.tar.gz';before=hashlib.sha256(archive.read_bytes()).hexdigest()
    if bad:
        with pytest.raises(RuntimeError):worker.run(source,tmp_path/'out',archive)
        assert not (tmp_path/'selected_raw').exists()
    else:
        worker.run(source,tmp_path/'out',archive)
        result=json.loads((tmp_path/'out/results.json').read_text())
        assert result['selected_training_rows']==600
        assert result['data_archive']['acquisition_route']=='existing-local-archive'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==before
