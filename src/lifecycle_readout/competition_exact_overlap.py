"""Exact-ID exposure anatomy on all supported validation queries, no fitting."""
import numpy as np
from .causal_stream import validate_events
from .competing_edges import source_bucket


def validation_overlap(data,streams,*,dimensions,profile_seed,max_prefix_entries=2000000):
    """Sparse exact-ID prefixes and hashed prefixes share strict-past event norms.

    Components are own-ID overlap, same-other-ID overlap and cross-ID hash
    collisions, under the original hashed profile normalizers. All streams must
    share history exclusions/boundaries. Only validation rows are queried.
    """
    validate_events(**{k:data[k] for k in ('sources','destinations','timestamps','features')})
    if not streams or isinstance(dimensions,bool) or not isinstance(dimensions,int) or dimensions<2:
        raise ValueError('Streams and profile width required')
    first=streams[0]
    for stream in streams:
        np.testing.assert_array_equal(stream.excluded_training_ids,first.excluded_training_ids)
        if stream.validation_end!=first.validation_end:raise ValueError('History boundaries differ')
    t,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    rows={};requested=set();outputs=[]
    for index,stream in enumerate(streams):
        outputs.append(dict(components=np.zeros((len(stream.event_ids),2,3)),raw_components=np.zeros((len(stream.event_ids),2,3)),queried=stream.split==1))
        for row in np.flatnonzero(stream.split==1):
            event=int(stream.event_ids[row]);rows.setdefault(event,[]).append((index,row))
            requested.update((int(u[event]),int(dest)) for dest in (v[event],stream.negative_destinations[row]))
    needed_sources={source for source,dest in requested};excluded=set(map(int,first.excluded_training_ids))
    if any(event in excluded for event in rows):raise ValueError('Excluded query')
    profiles,hashed,exact_totals,hash_totals,resets={},{},{},{},{}
    entries=0;start=0;max_negative=0.
    while start<len(t) and t[start]<=first.validation_end:
        time=float(t[start]);stop=int(np.searchsorted(t,time,side='right'))
        for event in range(start,stop):
            source=int(u[event])
            for index,row in rows.get(event,[]):
                stream=streams[index];total=exact_totals.get(source,{})
                ht=hash_totals.get(source,np.zeros(dimensions))
                for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
                    key=(source,int(dest))
                    if key not in resets:raise ValueError('Missing recurrent reference')
                    s,ref,href,baseline,ownbaseline,hashbaseline=resets[key]
                    if s>=time:raise ValueError('Non-past reference')
                    same=sum(value*total.get(a,0.) for a,value in ref.items())-baseline
                    own=ref.get(source,0.)*(total.get(source,0.)-ownbaseline)
                    full=float(href@ht-hashbaseline)
                    raw=np.array([own,same-own,full-same]);max_negative=max(max_negative,float(-raw.min()))
                    if raw.min()<-1e-8:raise ArithmeticError('Negative exact component beyond tolerance')
                    outputs[index]['raw_components'][row,slot]=raw
                    outputs[index]['components'][row,slot]=np.maximum(raw,0.)
        allowed=[event for event in range(start,stop) if event not in excluded]
        snapshots={}
        for source,dest in {(int(u[k]),int(v[k])) for k in allowed} & requested:
            h=hashed.get(dest,np.zeros(dimensions));norm=np.linalg.norm(h)
            reference={a:c/norm for a,c in profiles.get(dest,{}).items()} if norm else {}
            snapshots[(source,dest)]=(reference,h/norm if norm else np.zeros(dimensions))
        for event in allowed:
            source,dest=int(u[event]),int(v[event])
            if source not in needed_sources:continue
            h=hashed.get(dest,np.zeros(dimensions));norm=np.linalg.norm(h)
            if not norm:continue
            total=exact_totals.setdefault(source,{})
            for a,count in profiles[dest].items():
                if a not in total:
                    entries+=1
                    if entries>max_prefix_entries:raise RuntimeError('Exact prefix memory budget exceeded')
                total[a]=total.get(a,0.)+count/norm
            hash_totals[source]=hash_totals.get(source,np.zeros(dimensions))+h/norm
        for (source,dest),(ref,href) in snapshots.items():
            total=exact_totals.get(source,{});ht=hash_totals.get(source,np.zeros(dimensions))
            baseline=sum(value*total.get(a,0.) for a,value in ref.items())
            resets[(source,dest)]=(time,ref,href.copy(),baseline,total.get(source,0.),float(href@ht))
        for event in allowed:
            source,dest=int(u[event]),int(v[event])
            counts=profiles.setdefault(dest,{});counts[source]=counts.get(source,0)+1
            if dest not in hashed:hashed[dest]=np.zeros(dimensions)
            hashed[dest][source_bucket(source,dimensions,profile_seed)]+=1
        start=stop
    return outputs,dict(exact_prefix_entries=entries,queried_sources=len(needed_sources),max_negative_roundoff=max_negative)
