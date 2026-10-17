from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_intake import sample_users,parse_selected_line,summarize_history


def test_fixed_user_sampling_invariant():
    users=[f'user_{i:06}' for i in range(120)]
    assert sample_users(users)==sample_users(users[::-1])
    with pytest.raises(ValueError):sample_users(users+[users[0]])


def test_missing_targets_still_supply_competitor_history_and_ties_excluded():
    f=pd.DataFrame(dict(source=[0]*6,destination=[1,-1,-1,1,-1,1],timestamp_ms=[0,0,1000,3600000,3600000,7200000],artist=[1,-1,-1,1,-1,1]))
    s=summarize_history(f)
    assert s['long_gap']['3600']==dict(rows=2,users=1,with_competitors=1,without_competitors=1)
    assert s['valid_track_rows']==3 and s['recurrent_rows']==2
    assert s==summarize_history(f.iloc[::-1])


def test_train_parser_cutoff_missing_id_and_unused_titles():
    uid='user_000001';chosen={uid};cutoff='2009-01-01T00:00:00Z'
    row=uid+'\t2008-01-01T00:00:00Z\t12345678-1234-1234-1234-123456789abc\tartist\t\ttitle\twith tab'
    p=parse_selected_line(row,chosen,cutoff)
    assert p[2]=='' and p[3] and not p[4]
    assert parse_selected_line(row.replace('2008-01','2009-01'),chosen,cutoff) is None
    with pytest.raises(ValueError):parse_selected_line(uid+'\t2008-01-01T00:00:00Z',chosen,cutoff)
    with pytest.raises(ValueError):parse_selected_line(row.replace('2008-01-01','2008-02-30'),chosen,cutoff)


def test_unknown_only_and_multiple_sources():
    f=pd.DataFrame(dict(source=[0,1,0],destination=[1,-1,1],timestamp_ms=[0,1000,3600000],artist=[1,-1,1]))
    assert summarize_history(f)['long_gap']['3600']['without_competitors']==1
    only=f.assign(destination=-1);s=summarize_history(only)
    assert s['recurrent_rows']==0 and s['valid_artist_on_track_fraction'] is None
