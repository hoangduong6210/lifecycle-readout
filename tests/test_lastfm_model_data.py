from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_model_data import append_validation


def test_january_only_uses_existing_training_ids_and_unknown_embedding_for_new_artist(tmp_path):
    track='12345678-1234-1234-1234-123456789abc';newtrack='22345678-1234-1234-1234-123456789abc';artist='12345678-1234-1234-1234-123456789def';newartist='22345678-1234-1234-1234-123456789def';user='user_000001'
    parent=dict(source=np.array([0]),destination=np.array([0]),timestamp_ms=np.array([1230000000000]),artist=np.array([0]));maps=dict(sampled_users=[user],track_uuids=[track],artist_uuids=[artist])
    raw=tmp_path/'raw.tsv';raw.write_text(''.join(f'{user}\t{time}\t{a}\tartist\t{t}\ttitle\n' for time,a,t in [('2008-12-31T00:00:00Z',artist,track),('2009-01-01T01:00:00Z',artist,track),('2009-01-02T01:00:00Z',newartist,newtrack),('2009-01-03T01:00:00Z',newartist,''),('2009-02-01T00:00:00Z','bad','bad')]))
    frame,x,ident,report=append_validation(parent,maps,raw,'2009-01-01T00:00:00Z','2009-02-01T00:00:00Z')
    assert len(frame)==4 and report['validation_rows']==3
    assert ident['track_uuids'][0]==track and ident['artist_uuids'][0]==artist
    assert x[:,0].tolist()==[1,1,0,0] and x[:,2].tolist()==[1,1,1,1]
    assert frame.destination.to_list()==[0,0,1,-1]
