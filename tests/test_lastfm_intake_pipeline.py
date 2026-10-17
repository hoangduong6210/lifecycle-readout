from pathlib import Path
import hashlib,importlib.util,io,json,tarfile,tomllib
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('bad',[False,True])
def test_raw_archive_checksum_gate_and_complete_training_pipeline(tmp_path,bad):
    spec=importlib.util.spec_from_file_location('lastfm_worker',ROOT/'scripts/run_lastfm_intake.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    uid='user_000001';track='12345678-1234-1234-1234-123456789abc';artist='12345678-1234-1234-1234-123456789def'
    events=''.join(f'{uid}\t{time}\t{artist}\tartist\t{tid}\ttitle\n' for time,tid in [('2008-01-01T00:00:00Z',track),('2008-01-01T00:30:00Z',''),('2008-01-01T01:00:00Z',track),('2009-01-01T00:00:00Z',track)]).encode()
    profile=b'#userid\tgender\nuser_000001\tm\nuser_000002\tf\n'
    p=tomllib.loads((ROOT/'protocols/lastfm_intake_v1.toml').read_text())
    archive=tmp_path/'fixture.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,raw in [(p['event_member'],events),(p['profile_member'],profile)]:
            info=tarfile.TarInfo('data/'+name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    md5=lambda x:hashlib.md5(x).hexdigest()
    original=archive.read_bytes();text=(ROOT/'protocols/lastfm_intake_v1.toml').read_text().replace(p['archive_md5'],'0'*32 if bad else md5(original)).replace(p['event_md5'],md5(events)).replace(p['profile_md5'],md5(profile)).replace('sample_users = 100','sample_users = 2')
    protocol=tmp_path/'protocol.toml';protocol.write_text(text)
    if bad:
        with pytest.raises(ValueError,match='Archive MD5'):m.run(archive,protocol,tmp_path/'out',tmp_path/'raw')
        assert not list((tmp_path/'raw').iterdir())
        assert (tmp_path/'out/failure.json').exists()
    else:
        m.run(archive,protocol,tmp_path/'out',tmp_path/'raw');result=json.loads((tmp_path/'out/results.json').read_text())
        assert result['summary']['rows']==3 and result['summary']['long_gap']['3600']['with_competitors']==1
        assert not result['engineering_feasibility_gate_passed'] and result['sampled_users']==2
    assert archive.read_bytes()==original
