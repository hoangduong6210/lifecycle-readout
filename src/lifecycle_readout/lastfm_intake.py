"""Fixed-history Last.fm intake with unsupported targets retained as events."""
import hashlib,re
from datetime import datetime,timezone
import numpy as np
import pandas as pd

UUID=re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
USER=re.compile(r"user_[0-9]{6}\Z")
TIME=re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


def sample_users(ids,n=100,seed=17):
    ids=list(ids)
    if len(set(ids))!=len(ids) or n<1 or n>len(ids) or not all(USER.fullmatch(u) for u in ids):
        raise ValueError("Unique valid profile IDs and feasible sample required")
    return sorted(ids,key=lambda u:(hashlib.sha256(f"lastfm-intake:{seed}:{u}".encode()).digest(),u))[:n]


def parse_selected_line(line,chosen,cutoff):
    """Do not parse later feedback/identities; all earlier selected rows retained."""
    fields=line.rstrip("\r\n").split("\t",5)
    if fields[0] not in chosen:return None
    if len(fields)<2 or not TIME.fullmatch(fields[1]):raise ValueError("Invalid selected timestamp")
    if fields[1]>=cutoff:return None
    if len(fields)!=6:raise ValueError("Malformed selected training row")
    timestamp=int(datetime.strptime(fields[1],"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())*1000
    artist=fields[2].lower() if UUID.fullmatch(fields[2]) else ""
    track=fields[4].lower() if UUID.fullmatch(fields[4]) else ""
    return fields[0],timestamp,track,artist,bool(fields[4] and not track),bool(fields[2] and not artist)


def summarize_history(frame):
    """destination -1 is unsupported but contributes to intervening source events."""
    required={'source','destination','timestamp_ms','artist'}
    if set(frame.columns)!=required or frame.isna().any().any():raise ValueError("Complete four-column history required")
    if not len(frame):raise ValueError("No sampled training history")
    if not np.isfinite(frame.to_numpy()).all():raise ValueError("Finite history required")
    known=frame[frame.destination>=0]
    source=frame.groupby(['source','timestamp_ms']).size().rename('n').reset_index()
    source['after']=source.groupby('source')['n'].cumsum();source['before']=source['after']-source['n']
    pair=known.groupby(['source','destination','timestamp_ms']).size().rename('multiplicity').reset_index()
    pair=pair.merge(source[['source','timestamp_ms','before','after']],on=['source','timestamp_ms'],validate='many_to_one').sort_values(['source','destination','timestamp_ms'])
    groups=pair.groupby(['source','destination'],sort=False)
    pair['previous']=groups.timestamp_ms.shift();pair['previous_after']=groups.after.shift()
    pair['gap']=(pair.timestamp_ms-pair.previous)/1000;pair['competitors']=pair.before-pair.previous_after
    recurrent=pair.previous.notna()
    if (pair.loc[recurrent,'gap']<=0).any() or (pair.loc[recurrent,'competitors']<0).any():raise ArithmeticError("Invalid history ordering")
    gaps={}
    for t in (60,3600,86400):
        mask=recurrent & (pair.gap>=t)
        gaps[str(t)]=dict(rows=int(pair.loc[mask,'multiplicity'].sum()),users=int(pair.loc[mask,'source'].nunique()),with_competitors=int(pair.loc[mask & (pair.competitors>0),'multiplicity'].sum()),without_competitors=int(pair.loc[mask & (pair.competitors==0),'multiplicity'].sum()))
    return dict(rows=len(frame),active_users=int(frame.source.nunique()),valid_track_rows=len(known),valid_track_fraction=len(known)/len(frame),valid_artist_on_track_fraction=float((known.artist>=0).mean()) if len(known) else None,track_pairs=int(len(groups)),recurrent_rows=int(pair.loc[recurrent,'multiplicity'].sum()),recurrent_with_competitors=int(pair.loc[recurrent & (pair.competitors>0),'multiplicity'].sum()),long_gap=gaps)
