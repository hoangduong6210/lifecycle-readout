from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_artist_history import extract_artist_history


def direct(frame,index):
    row=frame.iloc[index];prior=frame[(frame.source==row.source)&(frame.timestamp_ms<row.timestamp_ms)]
    own=prior[prior.destination==row.destination] if row.destination>=0 else prior.iloc[:0]
    if not len(own):return dict(has_own=0,gap_seconds=0,prior_own_count=0,reference_status=0,reference_artist=-1,total_competitors=0,same_artist=0,other_artist=0,unknown_artist=0)
    time=own.timestamp_ms.max();reference=own[own.timestamp_ms==time].artist;valid=set(reference[reference>=0]);status=3 if len(valid)>1 else (1 if len(valid)==1 and (reference>=0).all() else 2);artist=next(iter(valid)) if status==1 else -1
    between=prior[prior.timestamp_ms>time]
    return dict(has_own=1,gap_seconds=(row.timestamp_ms-time)/1000,prior_own_count=len(own),reference_status=status,reference_artist=artist,total_competitors=len(between),same_artist=int((between.artist==artist).sum()) if status==1 else 0,other_artist=int(((between.artist>=0)&(between.artist!=artist)).sum()) if status==1 else 0,unknown_artist=int((between.artist<0).sum()))


def fixture():
    return pd.DataFrame(dict(source=[0]*11+[1,1],destination=[1,2,-1,3,1,1,1,4,1,1,1,1,1],timestamp_ms=[0,0,1000,2000,3600000,3600000,7200000,7200000,10800000,10800000,14400000,0,3600000],artist=[5,5,5,-1,5,-1,9,9,5,9,5,8,8]))


def test_all_synthetic_rows_against_direct_oracle():
    f=fixture();got=extract_artist_history(f)
    for i in range(len(f)):
        expected=direct(f,i)
        assert {k:got[k][i] for k in got}==expected
    assert got['same_artist'][4]==1 and got['unknown_artist'][4]==1
    assert got['reference_status'][6]==2 and got['reference_status'][10]==3


def test_order_and_future_invariance():
    f=fixture();original=extract_artist_history(f);shuffled=f.sample(frac=1,random_state=17)
    got=extract_artist_history(shuffled)
    for name in got:np.testing.assert_array_equal(got[name],original[name][shuffled.index])
    future=pd.DataFrame(dict(source=[0],destination=[1],timestamp_ms=[20000000],artist=[77]))
    extended=extract_artist_history(pd.concat([f,future],ignore_index=True))
    for name in original:np.testing.assert_array_equal(original[name],extended[name][:-1])


def test_current_artist_cannot_supply_reference_or_dose():
    f=fixture();got=extract_artist_history(f);altered=f.copy();altered.loc[4,'artist']=100
    changed=extract_artist_history(altered)
    for name in got:assert changed[name][4]==got[name][4]


def test_invalid_domain_and_empty_history():
    f=fixture();f.loc[0,'destination']=-2
    with pytest.raises(ValueError):extract_artist_history(f)
    f=fixture().astype(float);f.loc[0,'timestamp_ms']=.5
    with pytest.raises(ValueError):extract_artist_history(f)
    assert all(len(x)==0 for x in extract_artist_history(f.iloc[:0]).values())
