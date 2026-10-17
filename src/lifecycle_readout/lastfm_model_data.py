"""Append the prospective January window while preserving original train IDs."""
import numpy as np
import pandas as pd
from .lastfm_intake import parse_selected_line


def append_validation(parent,identity_maps,raw_events,train_end,validation_end):
    users=identity_maps['sampled_users'];chosen=set(users);parts=[];lines=0
    with raw_events.open(encoding='utf-8') as f:
        for line in f:
            lines+=1;head=line.split('\t',2)
            if head[0] not in chosen:continue
            if len(head)<2:raise ValueError('Missing selected raw timestamp')
            if not train_end<=head[1]<validation_end:continue
            row=parse_selected_line(line,chosen,validation_end)
            if row is not None:parts.append(row)
    if not parts:raise ValueError('No fixed January validation events')
    tracks=list(identity_maps['track_uuids']);artists=list(identity_maps['artist_uuids']);ntrain_artists=len(artists)
    tracks.extend(sorted({x[2] for x in parts if x[2]}-set(tracks)));artists.extend(sorted({x[3] for x in parts if x[3]}-set(artists)))
    usermap={x:i for i,x in enumerate(users)};trackmap={x:i for i,x in enumerate(tracks)};artistmap={x:i for i,x in enumerate(artists)}
    val=pd.DataFrame(dict(source=[usermap[x[0]] for x in parts],destination=[trackmap.get(x[2],-1) for x in parts],timestamp_ms=[x[1] for x in parts],artist=[artistmap.get(x[3],-1) for x in parts]))
    frame=pd.concat([pd.DataFrame(parent),val],ignore_index=True).sort_values(['timestamp_ms','source','destination'],kind='stable').reset_index(drop=True)
    artist=frame.artist.to_numpy();features=np.column_stack((np.where((artist>=0)&(artist<ntrain_artists),artist+1,0),(frame.destination.to_numpy()>=0).astype(int),(artist>=0).astype(int))).astype(np.float32)
    report=dict(source_file_lines=lines,parent_rows=len(parent['source']),validation_rows=len(val),training_artist_vocabulary=ntrain_artists,validation_unseen_artist_rows=int((val.artist>=ntrain_artists).sum()),validation_missing_track_rows=int((val.destination<0).sum()),invalid_nonempty_validation_track_rows=sum(x[4] for x in parts),invalid_nonempty_validation_artist_rows=sum(x[5] for x in parts))
    return frame,features,dict(sampled_users=users,track_uuids=tracks,artist_uuids=artists,training_artist_vocabulary=ntrain_artists),report
