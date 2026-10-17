"""Pinned asset recovery: preserve unknown bytes and never execute downloads."""
from pathlib import Path
import importlib.util
import json

import pytest

spec=importlib.util.spec_from_file_location('craft_asset_fetch',Path(__file__).parents[1]/'scripts/fetch_craft_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(tmp_path):
    payloads={'LICENSE':b'fixture license','models/CRAFT.py':b'raise RuntimeError("must not execute")\n'}
    prefix='https://raw.githubusercontent.com/example/CRAFT/'+'a'*40+'/'
    identity=tmp_path/'identity.json';identity.write_text(json.dumps(dict(repository='https://github.com/example/CRAFT',commit='a'*40,
        files={name:dict(url=prefix+name,sha256=module.digest(data)) for name,data in payloads.items()})))
    paper=tmp_path/'paper.json';paper.write_text('{}')
    calls=[]
    def fetch(url):
        calls.append(url);return payloads[url.removeprefix(prefix)]
    args=dict(identity_sha256=module.digest(identity.read_bytes()),paper_sha256=module.digest(paper.read_bytes()),fetcher=fetch)
    return identity,paper,tmp_path/'recovered',calls,args


def test_roundtrip_and_idempotent_recheck_without_executing_code(tmp_path):
    identity,paper,output,calls,args=fixture(tmp_path)
    report=module.restore(identity,paper,output,**args)
    assert report['network_download_count']==2 and report['code_executed'] is False
    assert (output/'source_identity.json').read_bytes()==identity.read_bytes()
    second=module.restore(identity,paper,output,**args)
    assert second['network_download_count']==0 and len(calls)==2


def test_unfamiliar_existing_asset_is_preserved_before_network(tmp_path):
    identity,paper,output,calls,args=fixture(tmp_path);output.mkdir();(output/'LICENSE').write_bytes(b'local changes')
    with pytest.raises(FileExistsError):module.restore(identity,paper,output,**args)
    assert not calls and (output/'LICENSE').read_bytes()==b'local changes'


def test_bad_download_is_not_installed(tmp_path):
    identity,paper,output,calls,args=fixture(tmp_path);args['fetcher']=lambda _: b'wrong'
    with pytest.raises(ValueError,match='checksum mismatch'):module.restore(identity,paper,output,**args)
    assert not (output/'LICENSE').exists()


def test_path_escape_and_changed_metadata_rejected(tmp_path):
    identity,paper,output,calls,args=fixture(tmp_path)
    with pytest.raises(ValueError):module.safe_target(output,'../escaped.py')
    with pytest.raises(ValueError):module.safe_target(output,'/absolute.py')
    args['identity_sha256']='0'*64
    with pytest.raises(ValueError,match='Metadata'):module.restore(identity,paper,output,**args)
    assert not calls


def test_changed_download_host_is_rejected_before_network(tmp_path):
    identity,paper,output,calls,args=fixture(tmp_path)
    metadata=json.loads(identity.read_text());metadata['files']['LICENSE']['url']='https://example.invalid/LICENSE'
    identity.write_text(json.dumps(metadata));args['identity_sha256']=module.digest(identity.read_bytes())
    with pytest.raises(ValueError,match='URL/hash'):module.restore(identity,paper,output,**args)
    assert not calls
