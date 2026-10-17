from pathlib import Path
import math,sys,tomllib
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_model_stream import HistoricalCatalog,build_lastfm_stream
from lifecycle_readout.lastfm_artist_history import extract_artist_history


def fixture():
    rows=[(0,t,0,a) for t,a in [(1,5),(2,5),(3,6),(4,7),(6,6),(7,7)]]+[(1,1,0,8),(1,5,0,8),(0,-1,3600000,5)]
    for hour in range(2,25,2):rows.append((0,1+(hour//2-1)%4,hour*3600000,5 if (hour//2-1)%4<2 else 6 if (hour//2-1)%4==2 else 7))
    rows.append((0,6,14*3600000,6))
    rows.extend([(1,1,4*3600000,8),(1,5,12*3600000,8),(1,1,20*3600000,8)])
    return pd.DataFrame(rows,columns=['source','destination','timestamp_ms','artist']).sort_values(['timestamp_ms','source','destination'],kind='stable').reset_index(drop=True)


def settings():
    s=tomllib.loads((ROOT/'configs/lastfm_model_v1.toml').read_text());s['training_query_modulus']=1;return s


def build(f):return build_lastfm_stream(f,settings(),seed=101,train_end_ms=10*3600000,validation_end_ms=30*3600000)


def test_exact_sorted_catalog_windows_and_forbidden_pool():
    c=HistoricalCatalog(1000)
    for track,time in [(1,9000),(2,8000),(3,7000),(4,6999)]:c.update(0,track,time,1)
    pool,lo,hi=c.pool_bounds(0,1,1000,10000)
    assert pool[lo:hi]==[(8000,2),(9000,1)]  # gap 3000 is next bin, not this bin
    rng=np.random.RandomState(17)
    assert {c.choose(0,1,1000,10000,{1},rng) for _ in range(20)}=={2}
    assert c.choose(0,1,1000,10000,{1,2},rng) is None
    c.update(0,2,9500,1);assert c.state[0,2]==(9500,2)


def test_candidates_doses_and_all_graph_snapshots_against_direct_history():
    f=fixture();stream,extra=build(f);oracle=extract_artist_history(f)
    for j,i in enumerate(stream.event_ids):
        row=f.iloc[i];past=f[f.timestamp_ms<row.timestamp_ms];own=past[(past.source==row.source)&(past.destination==row.destination)]
        last=own.timestamp_ms.max();count=len(own);gap=(row.timestamp_ms-last)/1000
        positives=set(f.loc[(f.timestamp_ms==row.timestamp_ms)&(f.source==row.source),'destination']);pool=[]
        for track,g in past[(past.source==row.source)&(past.destination>=0)].groupby('destination'):
            if track not in positives and int(math.log2(len(g)))==int(math.log2(count)) and int(math.log2(1+(row.timestamp_ms-g.timestamp_ms.max())/3600000))==int(math.log2(1+gap/3600)):pool.append(track)
        assert int(stream.negative_destinations[j]) in pool
        np.testing.assert_array_equal(extra['raw'][j,0],np.array([oracle[k][i] for k in ['total_competitors','same_artist','unknown_artist','reference_status','reference_artist','prior_own_count']]+[extra['raw'][j,0,6]]))
        for slot,(kind,node) in enumerate([('u',row.source),('v',row.destination),('v',stream.negative_destinations[j])]):
            history=past[past.source==node] if kind=='u' else past[(past.destination==node)&(past.destination>=0)]
            indices=history.index.to_numpy()[-settings()['max_neighbors']:];observed=stream.neighbor_edges[j,slot];np.testing.assert_array_equal(observed[observed>=0],indices)
    assert (stream.split==0).any() and (stream.split==1).any()
    assert (extra['reason']==1).sum()==1


def test_future_append_preserves_prior_candidates_and_features():
    f=fixture();a,x=build(f);future=pd.DataFrame([(0,1,26*3600000,999)],columns=f.columns);b,y=build(pd.concat([f,future],ignore_index=True));m=len(a.event_ids)
    for name in ['event_ids','negative_destinations','neighbor_edges','pair_features','elapsed_times','split']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name)[:m])
    np.testing.assert_array_equal(x['raw'],y['raw'][:m]);np.testing.assert_array_equal(x['alternate_exposure'],y['alternate_exposure'][:m])


def test_current_metadata_cannot_enter_current_prediction_features():
    f=fixture();a,x=build(f);i=int(a.event_ids[0]);changed=f.copy();changed.loc[i,'artist']=12345;b,y=build(changed);j=int(np.flatnonzero(b.event_ids==i)[0]);np.testing.assert_array_equal(a.pair_features[0],b.pair_features[j]);np.testing.assert_array_equal(x['raw'][0],y['raw'][j]);np.testing.assert_array_equal(x['alternate_exposure'][0],y['alternate_exposure'][j])


def test_query_thinning_never_thins_history():
    f=fixture();s=settings();s['training_query_modulus']=2;stream,extra=build_lastfm_stream(f,s,seed=101,train_end_ms=10*3600000,validation_end_ms=30*3600000)
    assert (extra['reason']==4).any()
    # Every emitted source snapshot still contains all preceding source rows up to budget.
    for j,i in enumerate(stream.event_ids):
        row=f.iloc[i];expected=f.index[(f.source==row.source)&(f.timestamp_ms<row.timestamp_ms)].to_numpy()[-s['max_neighbors']:];got=stream.neighbor_edges[j,0];np.testing.assert_array_equal(got[got>=0],expected)
