"""Strict-past recorded-artist event doses; exact IDs and fixed own-event reference."""
from collections import Counter
import numpy as np

FEATURES=('has_own','gap_seconds','prior_own_count','reference_status','reference_artist','total_competitors','same_artist','other_artist','unknown_artist')


def extract_artist_history(frame):
    """Status: 0 no prior own, 1 available, 2 missing, 3 conflicting.

    All source rows update activity, including destination=-1. Unavailable
    reference doses are masked zeros with explicit status, not observed zeros.
    Returned arrays preserve input order; current artist is used only on update.
    """
    if set(frame.columns)!={'source','destination','timestamp_ms','artist'}:
        raise ValueError('Expected four-column prepared history')
    raw=frame[['source','destination','timestamp_ms','artist']].to_numpy()
    if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():raise ValueError('Finite integer IDs and milliseconds required')
    if (raw[:,0]<0).any() or (raw[:,1]<-1).any() or (raw[:,3]<-1).any():raise ValueError('Invalid ID domain')
    n=len(frame);out={name:np.zeros(n,dtype=np.float64 if name=='gap_seconds' else np.int64) for name in FEATURES}
    out['reference_artist'].fill(-1)
    if not n:return out
    order=np.lexsort((np.arange(n),raw[:,2],raw[:,0]));values=raw[order];i=0;active_source=None
    while i<n:
        source,t=int(values[i,0]),int(values[i,2]);j=i+1
        while j<n and values[j,0]==source and values[j,2]==t:j+=1
        if source!=active_source:
            active_source=source;last={};artist_counts=Counter();total=0;known=0
        # Query before updating ANY row at this source timestamp.
        own_groups={};artist_delta=Counter()
        for k in range(i,j):
            idx=int(order[k]);track,artist=int(values[k,1]),int(values[k,3])
            if artist>=0:artist_delta[artist]+=1
            if track<0:continue
            own_groups.setdefault(track,[]).append(artist)
            prior=last.get(track)
            if prior is None:continue
            prev_time,prev_total,prev_known,ref_artist,prev_same,own_count,status=prior
            out['has_own'][idx]=1;out['gap_seconds'][idx]=(t-prev_time)/1000;out['prior_own_count'][idx]=own_count
            out['reference_status'][idx]=status;out['reference_artist'][idx]=ref_artist
            count=total-prev_total;unknown=count-(known-prev_known)
            out['total_competitors'][idx]=count;out['unknown_artist'][idx]=unknown
            if status==1:
                same=artist_counts[ref_artist]-prev_same;other=known-prev_known-same
                if min(same,other,unknown)<0 or same+other+unknown!=count:raise ArithmeticError('Invalid dose partition')
                out['same_artist'][idx]=same;out['other_artist'][idx]=other
            if count<0 or unknown<0 or t<=prev_time:raise ArithmeticError('Invalid strict-past interval')
        total+=j-i;known+=sum(artist_delta.values());artist_counts.update(artist_delta)
        # Store AFTER-group counters so the whole old boundary is excluded.
        for track,artists in own_groups.items():
            valid={x for x in artists if x>=0}
            status=3 if len(valid)>1 else (1 if len(valid)==1 and all(x>=0 for x in artists) else 2)
            ref=next(iter(valid)) if status==1 else -1
            own_count=(last[track][5] if track in last else 0)+len(artists)
            last[track]=(t,total,known,ref,artist_counts[ref] if status==1 else 0,own_count,status)
        i=j
    return out
