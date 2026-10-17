"""Strict-past exposure with a target profile frozen at its last own event.

Separate from the completed query-time-profile residual pilot. Profiles are
behavioral hashes, not semantic labels. No test-split features are queried.
"""
import numpy as np
from lifecycle_readout.causal_stream import validate_events
from lifecycle_readout.competing_edges import source_bucket


def event_frozen_exposure(data, stream, *, dimensions: int, profile_seed: int):
    """Extract [count, exposure, reference available, usable competitor count].

    At the target's last event timestamp s, store e_v(s-) and the source's
    prefix AFTER all allowed updates at s. At query t use
        E_uv(t) = e_v(s-) dot sum_{s<r<t, (u,w,r)} e_w(r-).
    Thus the reference is fixed between own-edge events; each competitor
    contribution uses strictly prior information. Ties query before updates.
    Unsupported allowed events count; held-out training events never count.
    This routine supplies history, not a learned state reset or score.
    """
    validate_events(**{k:data[k] for k in ('sources','destinations','timestamps','features')})
    if isinstance(dimensions,bool) or not isinstance(dimensions,int) or dimensions<2:
        raise ValueError('Profile dimensions must be an integer at least two')
    times,sources,destinations=(data[k] for k in ('timestamps','sources','destinations'))
    queried=stream.split<=1
    rows={int(event):i for i,event in enumerate(stream.event_ids) if queried[i]}
    if len(rows)!=int(queried.sum()):
        raise ValueError('Duplicate query event identity')
    excluded=set(map(int,stream.excluded_training_ids))
    if any(event in excluded for event in rows):
        raise ValueError('Excluded event cannot be a query')
    requested={(int(sources[event]),int(v)) for event,row in rows.items()
               for v in (destinations[event],stream.negative_destinations[row])}
    raw=np.zeros((len(stream.event_ids),2,4),dtype=np.float64)
    shuffled=np.zeros(raw.shape[:2],dtype=np.float64)
    reference=np.zeros((*raw.shape[:2],dimensions),dtype=np.float64)
    prior_count=np.zeros(raw.shape[:2],dtype=np.int64)
    elapsed=np.zeros(raw.shape[:2],dtype=np.float64)
    reset_time=np.zeros(raw.shape[:2],dtype=np.float64)
    profiles,totals,resets,history={},{},{},{}
    permutation=np.random.RandomState(profile_seed).permutation(dimensions)
    if np.array_equal(permutation,np.arange(dimensions)):
        permutation=np.roll(permutation,1)
    def unit(v):
        x=profiles.get(int(v),np.zeros(dimensions));norm=np.linalg.norm(x)
        return x/norm if norm else np.zeros(dimensions)
    start=0
    while start<len(times) and times[start]<=stream.validation_end:
        t=float(times[start]);stop=int(np.searchsorted(times,t,side='right'))
        for event in range(start,stop):
            row=rows.get(event)
            if row is None:continue
            u=int(sources[event]);total,n,usable=totals.get(u,(np.zeros(dimensions),0,0))
            for slot,v in enumerate((destinations[event],stream.negative_destinations[row])):
                key=(u,int(v))
                if key not in resets:raise ValueError('Recurrent target lacks a past reset')
                s,count=history[key];prefix,n0,usable0,ref=resets[key]
                if s>=t:raise ValueError('Target history must be strictly earlier')
                recent=total-prefix;c=n-n0
                exposure=float(ref@recent)
                error=32*np.finfo(float).eps*(np.linalg.norm(total)+np.linalg.norm(prefix)+1)
                if exposure < -error or exposure > c+error:
                    raise ArithmeticError('Prefix exposure violates nonnegative kernel bounds')
                exposure=float(np.clip(exposure,0,c))
                raw[row,slot]=(c,exposure,float(ref.any()),usable-usable0)
                shuffled[row,slot]=float(np.clip(ref[permutation]@recent,0,c))
                reference[row,slot]=ref;prior_count[row,slot]=count
                elapsed[row,slot]=t-s;reset_time[row,slot]=s
        allowed=[event for event in range(start,stop) if event not in excluded]
        # All destination vectors come from before ANY event in this group.
        vectors={int(destinations[event]):unit(destinations[event]) for event in allowed}
        touched=set()
        for event in allowed:
            u,v=int(sources[event]),int(destinations[event]);key=(u,v)
            total,n,usable=totals.get(u,(np.zeros(dimensions),0,0))
            vec=vectors[v];totals[u]=(total+vec,n+1,usable+int(vec.any()))
            if key in requested:
                _,count=history.get(key,(t,0));history[key]=(t,count+1);touched.add(key)
        for u,v in touched:
            total,n,usable=totals[u]
            resets[(u,v)]=(total.copy(),n,usable,vectors[v].copy())
        for event in allowed:
            u,v=int(sources[event]),int(destinations[event])
            if v not in profiles:profiles[v]=np.zeros(dimensions)
            profiles[v][source_bucket(u,dimensions,profile_seed)]+=1
        start=stop
    np.testing.assert_allclose(elapsed[queried],stream.elapsed_times[queried],rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(np.log1p(prior_count[queried]),stream.pair_features[queried,:,2],rtol=2e-6,atol=1e-7)
    return dict(raw=raw,shuffled_exposure=shuffled,reference=reference,
                prior_count=prior_count,elapsed=elapsed,reset_time=reset_time,
                queried=queried,coordinate_permutation=permutation)
