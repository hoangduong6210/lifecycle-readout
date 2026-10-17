"""Pinned January development inputs and common candidates; no optimizer."""
from pathlib import Path
import argparse,hashlib,json,math,random,sys,tomllib,traceback
from datetime import datetime
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_model_data import append_validation
from lifecycle_readout.lastfm_model_stream import build_lastfm_stream
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def direct_history(frame,user,track,time,seed):
    past=frame[(frame.source==user)&(frame.timestamp_ms<time)];own=past[past.destination==track]
    assert len(own);last=int(own.timestamp_ms.max());refrows=own[own.timestamp_ms==last].artist;valid=set(refrows[refrows>=0]);status=3 if len(valid)>1 else (1 if len(valid)==1 and (refrows>=0).all() else 2);ref=next(iter(valid)) if status==1 else -1
    between=past[past.timestamp_ms>last];same=int((between.artist==ref).sum()) if status==1 else 0;unknown=int((between.artist<0).sum());alternate=-1
    if status==1:
        prefix=past[(past.timestamp_ms<=last)&(past.artist>=0)];first=prefix.groupby('artist').timestamp_ms.min().reset_index().sort_values(['timestamp_ms','artist']);artists=first.artist.to_list()
        if len(artists)>1:
            value=int.from_bytes(hashlib.sha256(f'lastfm-alternate:{seed}:{user}:{track}:{last}'.encode()).digest(),'big');pick=random.Random(value).randrange(len(artists)-1);position=artists.index(ref);alternate=artists[pick+(pick>=position)]
    alt=int((between.artist==alternate).sum()) if alternate>=0 else 0
    return dict(gap_seconds=(time-last)/1000,count=len(own),total=len(between),same=same,unknown=unknown,status=status,ref=int(ref),alternate_available=int(alternate>=0),alternate_exposure=alt)


def anchor_audit(frame,stream,extra,settings):
    records=[];timestamps=frame.timestamp_ms.to_numpy();sources=frame.source.to_numpy();targets=frame.destination.to_numpy()
    for split in (0,1):
        rows=np.flatnonzero(stream.split==split)
        chosen=sorted(rows.tolist(),key=lambda j:hashlib.sha256(f'lastfm-model-anchor:17:{int(stream.event_ids[j])}'.encode()).digest())[:16]
        for j in chosen:
            i=int(stream.event_ids[j]);user=int(sources[i]);positive=int(targets[i]);time=int(timestamps[i]);negative=int(stream.negative_destinations[j]);pair=[]
            assert negative not in set(targets[(sources==user)&(timestamps==time)])
            for slot,track in enumerate((positive,negative)):
                expected=direct_history(frame,user,track,time,settings['query_hash_seed']);row=extra['raw'][j,slot]
                assert row.tolist()==[expected[k] for k in ['total','same','unknown','status','ref','count','alternate_available']]
                assert extra['alternate_exposure'][j,slot]==expected['alternate_exposure']
                assert np.isclose(stream.elapsed_times[j,slot],expected['gap_seconds'],rtol=1e-6,atol=1e-5)
                pair.append(expected)
            assert int(math.log2(pair[0]['count']))==int(math.log2(pair[1]['count']))
            assert int(math.log2(1+pair[0]['gap_seconds']/settings['hard_time_unit_seconds']))==int(math.log2(1+pair[1]['gap_seconds']/settings['hard_time_unit_seconds']))
            for slot,(kind,node) in enumerate((('u',user),('v',positive),('v',negative))):
                mask=(timestamps<time)&((sources==node) if kind=='u' else ((targets==node)&(targets>=0)))
                expected=np.flatnonzero(mask)[-settings['max_neighbors']:];history=stream.neighbor_edges[j,slot];np.testing.assert_array_equal(history[history>=0],expected)
            records.append(dict(split=split,event_id=i,negative_destination=negative,expected=pair))
    return dict(status='PASS',anchors=records,scope='Independent direct full prepared-history scans of fixed train/validation anchors, including alternate references, counts/time bins and graph snapshots; not every-row raw lineage replay')


def run(parent_dir,raw_events,protocol,config,output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        p=tomllib.loads(protocol.read_text());settings=tomllib.loads(config.read_text());parent_manifest=json.loads((parent_dir/'artifact_manifest.json').read_text())
        used=['selected_training_events.npz','identity_maps.json','selected_members.json'];input_pins={name:sha(parent_dir/name) for name in used}
        assert all(parent_manifest[name]==h for name,h in input_pins.items())
        raw_before=raw_events.stat();raw_hash=sha(raw_events);member=json.loads((parent_dir/'selected_members.json').read_text())['files'][raw_events.name];assert raw_hash==member['sha256'] and raw_before.st_size==member['bytes']
        source_names=['src/lifecycle_readout/lastfm_model_data.py','src/lifecycle_readout/lastfm_model_stream.py','src/lifecycle_readout/lastfm_intake.py','src/lifecycle_readout/causal_stream.py','scripts/prepare_lastfm_model.py','tests/test_lastfm_model_stream.py','tests/test_lastfm_model_data.py']
        source_paths=[ROOT/name for name in source_names]+[protocol,config];source_pins={str(path.relative_to(ROOT)):sha(path) for path in source_paths}
        save(output/'preanalysis.json',dict(protocol=p,settings=settings,source_hashes=source_pins,parent_input_hashes=input_pins,raw_event_sha256=raw_hash,before_january_feedback_preparation=True,model_fit=False))
        for path in source_paths:
            snapshot=output/'sources'/str(path.relative_to(ROOT));snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
        maps=json.loads((parent_dir/'identity_maps.json').read_text())
        with np.load(parent_dir/'selected_training_events.npz',allow_pickle=False) as z:parent={k:z[k] for k in z.files}
        frame,features,identities,window=append_validation(parent,maps,raw_events,settings['training_end_exclusive_utc'],settings['validation_end_exclusive_utc'])
        train_end=int(datetime.fromisoformat(settings['training_end_exclusive_utc']).timestamp())*1000;validation_end=int(datetime.fromisoformat(settings['validation_end_exclusive_utc']).timestamp())*1000
        assert (frame.timestamp_ms<validation_end).all();assert int((frame.timestamp_ms<train_end).sum())==len(parent['source'])
        np.savez_compressed(output/'data.npz',sources=frame.source.to_numpy(),destinations=frame.destination.to_numpy(),artist_ids=frame.artist.to_numpy(),timestamp_ms=frame.timestamp_ms.to_numpy(),timestamps=frame.timestamp_ms.to_numpy()/1000,features=features,training_artist_vocabulary=np.array([window['training_artist_vocabulary']]))
        save(output/'identity_maps.json',identities);save(output/'window_report.json',window);per_seed=[]
        print('LASTFM_JANUARY_PREPARED',json.dumps(window),flush=True)
        for seed in settings['seeds']:
            stream,extra=build_lastfm_stream(frame,settings,seed=seed,train_end_ms=train_end,validation_end_ms=validation_end)
            folder=output/f'seed{seed}';folder.mkdir();np.savez_compressed(folder/'candidate_stream.npz',**{name:getattr(stream,name) for name in stream.__dataclass_fields__});np.savez_compressed(folder/'artist_features.npz',**extra)
            audit=anchor_audit(frame,stream,extra,settings);save(folder/'anchor_audit.json',audit)
            report=dict(seed=seed,candidate_digest=stream.digest(),train_queries=int((stream.split==0).sum()),validation_queries=int((stream.split==1).sum()),reason_counts={str(k):int((extra['reason']==k).sum()) for k in range(6)},real_anchor_count=len(audit['anchors']),candidate_sha256=sha(folder/'candidate_stream.npz'),artist_features_sha256=sha(folder/'artist_features.npz'))
            per_seed.append(report);save(output/'partial_results.json',dict(window=window,per_seed=per_seed,model_fit=False));print('LASTFM_CANDIDATES',json.dumps(report),flush=True)
        assert all(sha(ROOT/name)==h for name,h in source_pins.items()) and all(sha(parent_dir/name)==h for name,h in input_pins.items())
        assert (raw_before.st_size,raw_before.st_mtime_ns)==(raw_events.stat().st_size,raw_events.stat().st_mtime_ns)
        result=dict(status='PASS',protocol_id=p['protocol_id'],window=window,per_seed=per_seed,source_input_guards_passed=True,model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
        save(output/'results.json',result);print('LASTFM_MODEL_PREPARATION_PASS',flush=True)
    except BaseException as exc:
        save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:save(output/'artifact_manifest.json',{str(path.relative_to(output)):sha(path) for path in output.rglob('*') if path.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['parent_dir','raw_events','protocol','config','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.parent_dir,args.raw_events,args.protocol,args.config,args.output)
