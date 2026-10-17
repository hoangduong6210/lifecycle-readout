"""Thin local-CPU original-byte intake; no predictor or optimizer."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile,tomllib,traceback
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_intake import sample_users,parse_selected_line,summarize_history


def digest(path,kind='sha256'):
    with path.open('rb') as f:return hashlib.file_digest(f,kind).hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(archive,protocol,output,scratch):
    output.mkdir(parents=True,exist_ok=False);scratch.mkdir(parents=True,exist_ok=False)
    try:
        p=tomllib.loads(protocol.read_text());before=archive.stat()
        if before.st_size>p['max_download_bytes']:raise ValueError('Archive byte bound exceeded')
        identity=dict(bytes=before.st_size,md5=digest(archive,'md5'),sha256=digest(archive))
        save(output/'archive_identity.json',identity)
        if identity['md5']!=p['archive_md5']:raise ValueError('Archive MD5 mismatch')
        wanted={p['event_member']:p['event_md5'],p['profile_member']:p['profile_md5']};members={}
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=Path(member.name).name
                if name not in wanted:continue
                if not member.isfile() or name in members or member.size>p['max_event_bytes']:raise ValueError('Unexpected selected archive member')
                target=scratch/name;written=0
                with tar.extractfile(member) as src,target.open('xb') as dst:
                    for raw in iter(lambda:src.read(1024*1024),b''):
                        written+=len(raw)
                        if written>p['max_event_bytes']:raise ValueError('Extracted byte bound exceeded')
                        dst.write(raw)
                md5=digest(target,'md5');members[name]=dict(member=member.name,bytes=written,md5=md5,sha256=digest(target))
                if md5!=wanted[name]:raise ValueError('Primary TSV MD5 mismatch: '+name)
                if set(members)==set(wanted):break
        save(output/'selected_members.json',dict(files=members,full_listing=False))
        if set(members)!=set(wanted):raise ValueError('Required TSV missing')
        ids=[]
        with (scratch/p['profile_member']).open(encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():continue
                ids.append(line.split('\t',1)[0])
        chosen=sample_users(ids,p['sample_users'],p['sample_seed']);chosen_set=set(chosen);rows=[];all_rows=0
        with (scratch/p['event_member']).open(encoding='utf-8') as f:
            for line in f:
                all_rows+=1
                row=parse_selected_line(line,chosen_set,p['training_cutoff_exclusive_utc'])
                if row is not None:rows.append(row)
        if not rows:raise ValueError('No selected training events')
        users={u:i for i,u in enumerate(chosen)};tracks=sorted({x[2] for x in rows if x[2]});artists=sorted({x[3] for x in rows if x[3]})
        trackmap={x:i for i,x in enumerate(tracks)};artistmap={x:i for i,x in enumerate(artists)}
        frame=pd.DataFrame(dict(source=[users[x[0]] for x in rows],destination=[trackmap.get(x[2],-1) for x in rows],timestamp_ms=[x[1] for x in rows],artist=[artistmap.get(x[3],-1) for x in rows])).sort_values(['source','timestamp_ms','destination'],kind='stable').reset_index(drop=True)
        summary=summarize_history(frame);eligible=summary['long_gap']['3600'];coverage=summary['valid_artist_on_track_fraction']
        passed=eligible['rows']>=500 and eligible['users']>=25 and coverage is not None and coverage>=.8
        np.savez_compressed(output/'selected_training_events.npz',**{k:frame[k].to_numpy() for k in frame.columns})
        save(output/'identity_maps.json',dict(sampled_users=chosen,track_uuids=tracks,artist_uuids=artists))
        result=dict(status='PASS',protocol_id=p['protocol_id'],archive=identity,member_identity=members,profile_users=len(ids),sampled_users=len(chosen),source_file_rows=all_rows,training_cutoff_exclusive_utc=p['training_cutoff_exclusive_utc'],summary=summary,invalid_nonempty_track_rows=sum(x[4] for x in rows),invalid_nonempty_artist_rows=sum(x[5] for x in rows),engineering_feasibility_gate_passed=bool(passed),model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
        if (before.st_size,before.st_mtime_ns)!=(archive.stat().st_size,archive.stat().st_mtime_ns):raise ValueError('Source archive changed')
        save(output/'results.json',result);print('LASTFM_INTAKE',json.dumps(result),flush=True)
    except BaseException as exc:
        save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        save(output/'artifact_manifest.json',{x.name:digest(x) for x in output.iterdir() if x.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['archive','protocol','output','scratch']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.protocol,args.output,args.scratch)
