"""Posthoc input anatomy, without training or causal performance attribution."""
import numpy as np
from .causal_stream import validate_events
from .competing_edges import source_bucket


def kernel_attribution(source,reference_counts,competitor_counts,*,dimensions,profile_seed):
    """Independent exact-ID kernel numerator anatomy under hashed normalizers.

    Dictionaries contain strictly past event counts by exact source identity.
    Cross-source same-bucket products are collisions, not same-user overlap.
    The original hashed normalizers are retained so components sum to its kernel.
    """
    if not isinstance(dimensions,int) or isinstance(dimensions,bool) or dimensions<2:
        raise ValueError('At least two profile coordinates required')
    for counts in (reference_counts,competitor_counts):
        if any(not np.isfinite(x) or x<0 for x in counts.values()):raise ValueError('Nonnegative finite counts required')
    def hashed(counts):
        h=np.zeros(dimensions)
        for key,value in counts.items():h[source_bucket(key,dimensions,profile_seed)]+=value
        return h
    a,b=hashed(reference_counts),hashed(competitor_counts);den=np.linalg.norm(a)*np.linalg.norm(b)
    if not den:return dict(parts=np.zeros(4),same_other=0.,collision=0.,uniform_background=0.)
    bucket=source_bucket(source,dimensions,profile_seed)
    own_a=reference_counts.get(source,0.);own_b=competitor_counts.get(source,0.)
    other_a=a.copy();other_a[bucket]-=own_a
    other_b=b.copy();other_b[bucket]-=own_b
    parts=np.array([own_a*own_b,own_a*other_b[bucket],other_a[bucket]*own_b,other_a@other_b])/den
    same_other=sum(value*competitor_counts.get(key,0.) for key,value in reference_counts.items() if key!=source)/den
    collision=max(0.,float(parts.sum()-parts[0]-same_other))
    return dict(parts=parts,same_other=float(same_other),collision=collision,uniform_background=float(a.sum()*b.sum()/dimensions/den))


def exposure_attribution(data,stream,*,dimensions,profile_seed):
    """Prefix decomposition: own/own, own/other, other/own, other/other.

    The two cross terms are guaranteed hash-collision contributions. The final
    other/other term includes both exact-ID overlap and additional collisions.
    Exact-ID splitting of that last term is handled by direct anchor replay.
    """
    validate_events(**{k:data[k] for k in ('sources','destinations','timestamps','features')})
    if not isinstance(dimensions,int) or isinstance(dimensions,bool) or dimensions<2:raise ValueError('Invalid width')
    times,u,v=(data[k] for k in ('timestamps','sources','destinations'))
    queried=stream.split<=1;rows={int(event):i for i,event in enumerate(stream.event_ids) if queried[i]}
    if len(rows)!=queried.sum():raise ValueError('Duplicate query identity')
    excluded=set(map(int,stream.excluded_training_ids))
    if any(k in excluded for k in rows):raise ValueError('Excluded query')
    requested={(int(u[event]),int(dest)) for event,row in rows.items() for dest in (v[event],stream.negative_destinations[row])}
    profiles,counts,totals,resets={},{},{},{}
    parts=np.zeros((len(stream.event_ids),2,4));background=np.zeros(parts.shape[:2]);shuffled=np.zeros(parts.shape[:2])
    permutation=np.random.RandomState(profile_seed).permutation(dimensions)
    if np.array_equal(permutation,np.arange(dimensions)):permutation=np.roll(permutation,1)
    start=0
    while start<len(times) and times[start]<=stream.validation_end:
        t=float(times[start]);stop=int(np.searchsorted(times,t,side='right'))
        for event in range(start,stop):
            row=rows.get(event)
            if row is None:continue
            source=int(u[event]);bucket=source_bucket(source,dimensions,profile_seed)
            total_a,total_b=totals.get(source,(0.,np.zeros(dimensions)))
            for slot,dest in enumerate((v[event],stream.negative_destinations[row])):
                key=(source,int(dest))
                if key not in resets:raise ValueError('No prior own-event reference')
                s,prefix_a,prefix_b,alpha,beta=resets[key]
                if s>=t:raise ValueError('Non-past reset')
                da=total_a-prefix_a;db=total_b-prefix_b
                values=np.array([alpha*da,alpha*db[bucket],beta[bucket]*da,beta@db])
                tol=1e-9*(1+abs(total_a)+np.linalg.norm(total_b))
                if values.min()<-tol:raise ArithmeticError('Negative component')
                parts[row,slot]=np.maximum(values,0.)
                background[row,slot]=(alpha+beta.sum())*(da+db.sum())/dimensions
                ref=beta.copy();ref[bucket]+=alpha
                recent=db.copy();recent[bucket]+=da
                shuffled[row,slot]=max(0.,float(ref[permutation]@recent))
        allowed=[event for event in range(start,stop) if event not in excluded]
        # Count/profile snapshots are computed before every update in this group.
        split={}
        for event in allowed:
            source,dest=int(u[event]),int(v[event]);key=(source,dest)
            if key in split:continue
            h=profiles.get(dest,np.zeros(dimensions));norm=np.linalg.norm(h)
            alpha=counts.get(key,0)/norm if norm else 0.
            beta=h/norm if norm else np.zeros(dimensions)
            beta=beta.copy();beta[source_bucket(source,dimensions,profile_seed)]-=alpha
            split[key]=(alpha,beta)
        for event in allowed:
            source,dest=int(u[event]),int(v[event]);alpha,beta=split[(source,dest)]
            ta,tb=totals.get(source,(0.,np.zeros(dimensions)))
            totals[source]=(ta+alpha,tb+beta)
        for key in set(split)&requested:
            source,dest=key;ta,tb=totals[source];alpha,beta=split[key]
            resets[key]=(t,ta,tb.copy(),alpha,beta)
        for event in allowed:
            source,dest=int(u[event]),int(v[event]);key=(source,dest)
            counts[key]=counts.get(key,0)+1
            if dest not in profiles:profiles[dest]=np.zeros(dimensions)
            profiles[dest][source_bucket(source,dimensions,profile_seed)]+=1
        start=stop
    return dict(parts=parts,uniform_background=background,shuffled_exposure=shuffled,queried=queried)
