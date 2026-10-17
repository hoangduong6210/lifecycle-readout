"""Bounded local intake from an existing archive matching the published checksum."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tarfile,time,traceback,zipfile,tomllib
import numpy as np
import pandas as pd
import requests


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(source,output,local_archive):
    if output.exists():raise FileExistsError('Fresh output required')
    output.mkdir(parents=True);error=None
    try:
        root=output.parent/'intake_source';root.mkdir(exist_ok=False)
        with zipfile.ZipFile(source) as z:
            manifest=json.loads(z.read('manifest.json'))
            for name,h in manifest.items():
                raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==h
                if name not in ('intake_recurrence.py','protocol.toml','README.md','LICENSE'):raise ValueError('Unexpected source member')
                (root/name).write_bytes(raw)
        spec=importlib.util.spec_from_file_location('intake_recurrence',root/'intake_recurrence.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        p=tomllib.loads((root/'protocol.toml').read_text())
        try:
            import torch
            if torch.cuda.is_available():raise RuntimeError('CPU intake unexpectedly has CUDA; stop instead')
        except ImportError:pass
        save(output/'source_identity.json',dict(bundle_sha256=sha(source),worker_sha256=sha(Path(__file__)),manifest=manifest,protocol=p,python=sys.version,numpy=np.__version__,pandas=pd.__version__,cuda_available=False))
        # This amendment changes acquisition/execution only; sampling and gates stay fixed.
        archive=Path(local_archive);before=archive.stat();md5=hashlib.md5();s256=hashlib.sha256();count=0
        with archive.open('rb') as f:
            for raw in iter(lambda:f.read(4*1024*1024),b''):
                count+=len(raw)
                if count>p['max_archive_bytes']:raise RuntimeError('Local archive exceeds fixed byte bound')
                md5.update(raw);s256.update(raw)
        after=archive.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Archive changed while hashing')
        license_record=None
        identity=dict(published_url=p['archive_url'],acquisition_route='existing-local-archive',local_archive_name=archive.name,bytes=count,md5=md5.hexdigest(),sha256=s256.hexdigest(),expected_md5=p['expected_archive_md5'],metadata_license=license_record)
        save(output/'archive_identity.json',identity)
        if identity['md5']!=p['expected_archive_md5']:raise ValueError('Primary archive MD5 mismatch')
        if p.get('archive_sha256') and identity['sha256']!=p['archive_sha256']:raise ValueError('Local archive SHA256 mismatch')
        wanted={'user_features_1k.csv','log_standard_4_08_to_4_21_1k.csv','video_features_basic_1k.csv'}
        rawdir=output.parent/'selected_raw';rawdir.mkdir(exist_ok=False);selected={}
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=Path(member.name).name
                if name not in wanted:continue
                if name in selected or not member.isfile() or member.size>1000000000:raise ValueError('Unexpected selected archive member')
                with tar.extractfile(member) as src,(rawdir/name).open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                selected[name]=dict(member=member.name,bytes=member.size,sha256=sha(rawdir/name))
                if set(selected)==wanted:break
        if set(selected)!=wanted:raise ValueError('Missing required primary CSV')
        save(output/'selected_members.json',dict(files=selected,full_archive_listing=False))
        users=pd.read_csv(rawdir/'user_features_1k.csv',usecols=['user_id'])['user_id'].to_numpy()
        chosen=module.select_users(users,n=p['sample_users'],seed=p['sample_seed']);parts=[];all_rows=0
        path=rawdir/'log_standard_4_08_to_4_21_1k.csv';columns=pd.read_csv(path,nrows=0).columns.tolist()
        needed=['user_id','video_id','time_ms','is_click']
        if not set(needed)<=set(columns):raise ValueError('Training log schema missing required columns')
        for chunk in pd.read_csv(path,usecols=needed,chunksize=200000):
            all_rows+=len(chunk);parts.append(chunk[chunk.user_id.isin(chosen)].copy())
        events=pd.concat(parts,ignore_index=True)
        if len(events)==0 or not set(events.is_click.unique())<={0,1}:raise ValueError('Empty/invalid selected history')
        if events.isna().any().any():raise ValueError('Incomplete selected history')
        events=events.sort_values(['time_ms','user_id','video_id'],kind='stable').reset_index(drop=True)
        frame=events.rename(columns={'user_id':'source','video_id':'destination','time_ms':'timestamp_ms'})[['source','destination','timestamp_ms']]
        impression=module.recurrence_summary(frame);click=module.recurrence_summary(frame[events.is_click==1])
        videos=set(map(int,events.video_id.unique()));meta=[]
        basic=rawdir/'video_features_basic_1k.csv';basic_columns=pd.read_csv(basic,nrows=0).columns.tolist()
        required=['video_id','tag','author_id','upload_dt']
        if not set(required)<=set(basic_columns):raise ValueError('Basic metadata schema missing required columns')
        for chunk in pd.read_csv(basic,usecols=required,dtype={'tag':'string'},chunksize=200000):meta.append(chunk[chunk.video_id.isin(videos)].copy())
        metadata=pd.concat(meta,ignore_index=True)
        if metadata.video_id.duplicated().any():raise ValueError('Duplicate item metadata identity')
        def parse_tag(value):
            if pd.isna(value) or str(value).strip() in ('','-1','UNKNOWN','[]'):return False,False
            tokens=[x.strip() for x in str(value).strip('[]').split(',')]
            valid=all(x.isdigit() for x in tokens)
            return bool(tokens) and valid,not valid
        parsed=[parse_tag(x) for x in metadata.tag];metadata['valid_tag']=[x[0] for x in parsed]
        metadata['unrecognized_tag']=[x[1] for x in parsed]
        joined=events[['video_id','is_click']].merge(metadata[['video_id','valid_tag']],on='video_id',how='left',validate='many_to_one')
        coverage=float(joined.valid_tag.fillna(False).mean());clicked=joined[joined.is_click==1]
        click_coverage=float(clicked.valid_tag.fillna(False).mean()) if len(clicked) else None
        eligible=click['long_gap'].get('3600',{'rows':0,'users':0})
        gate=eligible['rows']>=500 and eligible['users']>=25 and click_coverage is not None and click_coverage>=.8 and not metadata.unrecognized_tag.any()
        np.savez_compressed(output/'selected_training_events.npz',user_id=events.user_id.to_numpy(),video_id=events.video_id.to_numpy(),time_ms=events.time_ms.to_numpy(),is_click=events.is_click.to_numpy())
        metadata.to_csv(output/'selected_video_metadata.csv',index=False)
        result=dict(status='PASS',protocol_id=p['protocol_id'],data_archive=identity,sampled_user_ids=chosen,source_users=len(users),training_file_rows=all_rows,selected_training_rows=len(events),impressions=impression,positive_clicks=click,metadata=dict(selected_unique_videos=len(videos),joined_videos=len(metadata),event_valid_tag_fraction=coverage,clicked_event_valid_tag_fraction=click_coverage,unrecognized_tag_rows=int(metadata.unrecognized_tag.sum()),training_columns=columns,basic_columns=basic_columns),engineering_feasibility_gate_passed=bool(gate),model_fit=False,predictor_scored=False,test_scored=False,gpu_allocated=False,publication_eligible=False)
        save(output/'results.json',result);print('INTAKE_RESULT',json.dumps(dict(status=result['status'],rows=len(events),gate=bool(gate))),flush=True)
    except BaseException as exc:
        error=exc;save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));print('INTAKE_FAILURE',type(exc).__name__,flush=True)
    finally:
        manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
        save(output/'artifact_manifest.json',manifest)
        archive_out=output.parent/'kuairand-intake-artifacts.tar.gz'
        with tarfile.open(archive_out,'w:gz') as tar:tar.add(output,arcname='artifacts')
        save(output.parent/'kuairand-intake-complete.json',dict(status='failed' if error else 'completed',source_bundle_sha256=sha(source),archive_sha256=sha(archive_out)))
    if error:raise RuntimeError('Intake failed; retained report') from error


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source_zip',type=Path,required=True);parser.add_argument('--output_dir',type=Path,required=True);parser.add_argument('--local_archive',type=Path,required=True);args=parser.parse_args();run(args.source_zip,args.output_dir,args.local_archive)
