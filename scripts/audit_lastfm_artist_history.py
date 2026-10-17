"""Bounded training-input audit; no optimizer or predictor."""
from pathlib import Path
import argparse,hashlib,heapq,json,sys,tomllib,traceback
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_artist_history import extract_artist_history
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def direct(frame,i):
    # Independent full prepared-history scan, no prefix extractor/counter reuse.
    row=frame.iloc[i];prior=frame[(frame.source==row.source)&(frame.timestamp_ms<row.timestamp_ms)]
    own=prior[prior.destination==row.destination] if row.destination>=0 else prior.iloc[:0]
    if not len(own):return dict(has_own=0,gap_seconds=0,prior_own_count=0,reference_status=0,reference_artist=-1,total_competitors=0,same_artist=0,other_artist=0,unknown_artist=0)
    time=own.timestamp_ms.max();artists=own.loc[own.timestamp_ms==time,'artist'];valid=set(artists[artists>=0]);status=3 if len(valid)>1 else (1 if len(valid)==1 and (artists>=0).all() else 2);ref=next(iter(valid)) if status==1 else -1
    between=prior[prior.timestamp_ms>time]
    return dict(has_own=1,gap_seconds=float((row.timestamp_ms-time)/1000),prior_own_count=len(own),reference_status=status,reference_artist=int(ref),total_competitors=len(between),same_artist=int((between.artist==ref).sum()) if status==1 else 0,other_artist=int(((between.artist>=0)&(between.artist!=ref)).sum()) if status==1 else 0,unknown_artist=int((between.artist<0).sum()))


def describe(frame,features,mask):
    available=mask & (features['reference_status']==1);n=features['total_competitors'][available];e=features['same_artist'][available];o=features['other_artist'][available];u=features['unknown_artist'][available]
    result=dict(rows=int(mask.sum()),users=int(frame.loc[mask,'source'].nunique()),reference_available_rows=int(available.sum()),reference_unavailable_rows=int((mask & (features['reference_status']>1)).sum()),dose_population='Available-reference subset only')
    if not len(n):return result
    assert np.array_equal(n,e+o+u)
    corr=float(np.corrcoef(np.log1p(n),np.log1p(e))[0,1]) if np.std(n)>0 and np.std(e)>0 else None
    result.update(nonzero_same_rows=int((e>0).sum()),zero_same_with_activity_rows=int(((e==0)&(n>0)).sum()),mixed_same_other_rows=int(((e>0)&(o>0)).sum()),all_activity_same_rows=int(((e==n)&(n>0)).sum()),total_dose_sum=int(n.sum()),same_dose_sum=int(e.sum()),other_dose_sum=int(o.sum()),unknown_dose_sum=int(u.sum()),log1p_total_same_correlation=corr,quantiles={name:np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist() for name,values in [('total',n),('same',e)]})
    grouped=pd.DataFrame(dict(source=frame.loc[available,'source'].to_numpy(),total=n,same=e)).groupby(['source','total']).same.agg(['min','max','size'])
    variable=grouped[grouped['min']!=grouped['max']]
    result['within_source_fixed_total']=dict(groups=len(grouped),variable_same_groups=len(variable),rows_in_variable_groups=int(variable['size'].sum()))
    if len(variable):
        (source,total),w=next(iter(variable.sort_index().iterrows()));result['within_source_fixed_total']['first_lexicographic_witness']=dict(source=int(source),total=int(total),min_same=int(w['min']),max_same=int(w['max']))
    return result


def run(input_path,parent_results,protocol,output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        p=tomllib.loads(protocol.read_text());sources=[ROOT/'src/lifecycle_readout/lastfm_artist_history.py',Path(__file__),ROOT/'tests/test_lastfm_artist_history.py',protocol]
        pins={str(x.relative_to(ROOT)):sha(x) for x in sources};input_hash=sha(input_path);parent_hash=sha(parent_results)
        parent_manifest=json.loads((input_path.parent/'artifact_manifest.json').read_text());assert parent_manifest[input_path.name]==input_hash and parent_manifest[parent_results.name]==parent_hash
        save(output/'preanalysis.json',dict(protocol=p,source_hashes=pins,input_sha256=input_hash,parent_results_sha256=parent_hash,model_fit=False))
        with np.load(input_path,allow_pickle=False) as data:frame=pd.DataFrame({k:data[k] for k in ('source','destination','timestamp_ms','artist')})
        features=extract_artist_history(frame);np.savez_compressed(output/'artist_history_features.npz',**features)
        recurrent=features['has_own']==1;parent=json.loads(parent_results.read_text())['summary']
        assert int(recurrent.sum())==parent['recurrent_rows']
        assert int((recurrent & (features['total_competitors']>0)).sum())==parent['recurrent_with_competitors']
        for t in (60,3600,86400):
            mask=recurrent & (features['gap_seconds']>=t);q=parent['long_gap'][str(t)]
            assert int(mask.sum())==q['rows'] and frame.loc[mask,'source'].nunique()==q['users']
            assert int((mask & (features['total_competitors']>0)).sum())==q['with_competitors']
        anchors=[]
        strata={'supported_one_hour':recurrent & (features['gap_seconds']>=3600)&(features['reference_status']==1),'first_own':(~recurrent)&(frame.destination.to_numpy()>=0),'unavailable_reference':recurrent&(features['reference_status']>1)}
        source_ids=frame.source.to_numpy();times=frame.timestamp_ms.to_numpy();targets=frame.destination.to_numpy()
        def rank(i):
            return hashlib.sha256(f"lastfm-artist-audit:{int(source_ids[i])}:{int(times[i])}:{int(targets[i])}:{i}".encode()).digest()
        for name,mask in strata.items():
            chosen=heapq.nsmallest(16,np.flatnonzero(mask).tolist(),key=rank)
            for i in chosen:
                expected=direct(frame,i);actual={k:float(v[i]) if k=='gap_seconds' else int(v[i]) for k,v in features.items()};assert actual==expected,(name,i,actual,expected)
                anchors.append(dict(stratum=name,row=i,expected=expected))
        save(output/'direct_history_anchors.json',dict(status='PASS',strata_population={k:int(v.sum()) for k,v in strata.items()},anchors=anchors,scope='Full prepared same-source history per anchor; raw TSV lineage inherits the previous fixed intake anchors'))
        assert all(sha(ROOT/name)==h for name,h in pins.items()) and sha(input_path)==input_hash and sha(parent_results)==parent_hash
        result=dict(status='PASS',protocol_id=p['protocol_id'],input_rows=len(frame),reference_status_counts={str(code):int((features['reference_status']==code).sum()) for code in range(4)},all_recurrent=describe(frame,features,recurrent),one_hour_returns=describe(frame,features,recurrent & (features['gap_seconds']>=3600)),real_anchor_count=len(anchors),parent_denominators_match=True,source_input_guard_passed=True,model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
        save(output/'results.json',result);print('ARTIST_AUDIT',json.dumps(result),flush=True)
    except BaseException as exc:
        save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:save(output/'artifact_manifest.json',{x.name:sha(x) for x in output.iterdir() if x.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['input','parent_results','protocol','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.input,args.parent_results,args.protocol,args.output)
