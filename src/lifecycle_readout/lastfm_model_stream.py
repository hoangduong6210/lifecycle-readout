"""Exact chronological Last.fm candidates, artist doses and graph snapshots."""
from bisect import bisect_left,bisect_right,insort
from collections import Counter,defaultdict,deque
from dataclasses import dataclass
import hashlib,math,random
import numpy as np
from .causal_stream import CandidateStream


class HistoricalCatalog:
    """Uniform draws from exact historical count/time bins; no unmatched fallback."""
    def __init__(self,unit_ms):
        if unit_ms<1 or int(unit_ms)!=unit_ms:raise ValueError('Positive integer time unit required')
        self.unit=int(unit_ms);self.buckets=defaultdict(list);self.state={}

    def update(self,user,track,time,multiplicity):
        key=(user,track);old=self.state.get(key);count=multiplicity
        if old:
            last,prior=old;bucket=self.buckets[(user,prior.bit_length()-1)];i=bisect_left(bucket,(last,track));assert bucket[i]==(last,track);bucket.pop(i);count+=prior
        insort(self.buckets[(user,count.bit_length()-1)],(time,track));self.state[key]=(time,count)

    def pool_bounds(self,user,count,gap,time):
        if min(count,gap)<1:raise ValueError('Historical query required')
        b=((self.unit+int(gap))//self.unit).bit_length()-1
        pool=self.buckets[(user,int(count).bit_length()-1)]
        lo=bisect_right(pool,(time-self.unit*((1<<(b+1))-1),float('inf')))
        hi=bisect_right(pool,(time-self.unit*((1<<b)-1),float('inf')))
        return pool,lo,hi

    def choose(self,user,count,gap,time,forbidden,rng):
        pool,lo,hi=self.pool_bounds(user,count,gap,time)
        if lo>=hi:return None
        for _ in range(32):
            track=pool[int(rng.randint(lo,hi))][1]
            if track not in forbidden:return track
        allowed=[v for _,v in pool[lo:hi] if v not in forbidden]
        return allowed[int(rng.randint(len(allowed)))] if allowed else None


@dataclass
class OwnReference:
    time:int
    total:int
    known:int
    artist:int
    same:int
    count:int
    status:int
    alternate:int
    alternate_count:int


class ArtistPrefix:
    def __init__(self,seed=17):
        self.seed=seed;self.total=Counter();self.known=Counter();self.artists=defaultdict(Counter);self.artist_lists=defaultdict(list);self.artist_positions=defaultdict(dict);self.own={}

    def query(self,user,track,time):
        ref=self.own.get((user,track))
        if ref is None:return None
        if time<=ref.time:raise ValueError('Strictly prior own reference required')
        n=self.total[user]-ref.total;unknown=n-(self.known[user]-ref.known)
        e=self.artists[user][ref.artist]-ref.same if ref.status==1 else 0
        alternate=self.artists[user][ref.alternate]-ref.alternate_count if ref.alternate>=0 else 0
        assert min(n,unknown,e,alternate)>=0 and e+unknown<=n
        return dict(gap_seconds=(time-ref.time)/1000,prior_own_count=ref.count,total_competitors=n,same_artist=e,unknown_artist=unknown,reference_status=ref.status,reference_artist=ref.artist,alternate_exposure=alternate,alternate_available=int(ref.alternate>=0))

    def update_group(self,time,rows):
        # rows are (source,track,artist), including unknown tracks.
        byuser=defaultdict(list)
        for user,track,artist in rows:byuser[user].append((track,artist))
        for user,events in byuser.items():
            additions=Counter(a for _,a in events if a>=0);self.total[user]+=len(events);self.known[user]+=sum(additions.values())
            for artist in sorted(additions):
                if artist not in self.artist_positions[user]:self.artist_positions[user][artist]=len(self.artist_lists[user]);self.artist_lists[user].append(artist)
            self.artists[user].update(additions);tracks=defaultdict(list)
            for track,artist in events:
                if track>=0:tracks[track].append(artist)
            for track,artists in tracks.items():
                valid={a for a in artists if a>=0};status=3 if len(valid)>1 else (1 if len(valid)==1 and min(artists)>=0 else 2);artist=next(iter(valid)) if status==1 else -1
                alternate=-1
                if status==1 and len(self.artist_lists[user])>1:
                    seed=int.from_bytes(hashlib.sha256(f'lastfm-alternate:{self.seed}:{user}:{track}:{time}'.encode()).digest(),'big');pick=random.Random(seed).randrange(len(self.artist_lists[user])-1);position=self.artist_positions[user][artist]
                    alternate=self.artist_lists[user][pick+(pick>=position)]
                old=self.own.get((user,track));count=(old.count if old else 0)+len(artists)
                self.own[user,track]=OwnReference(time,self.total[user],self.known[user],artist,self.artists[user][artist] if status==1 else 0,count,status,alternate,self.artists[user][alternate] if alternate>=0 else 0)


def build_lastfm_stream(frame,settings,*,seed,train_end_ms,validation_end_ms):
    """Input must be global chronological canonical rows; original IDs are typed.

    Numeric source/track IDs may overlap; graph keys retain their namespaces.
    Missing tracks update user history only and never create a fictitious item.
    """
    if set(frame.columns)!={'source','destination','timestamp_ms','artist'}:raise ValueError('Expected prepared four-column history')
    values=frame[['source','destination','timestamp_ms','artist']].to_numpy()
    if not len(values) or not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():raise ValueError('Finite nonempty integer history required')
    if (values[:,0]<0).any() or (values[:,1]<-1).any() or (values[:,3]<-1).any() or np.any(np.diff(values[:,2])<0):raise ValueError('Invalid IDs or nonchronological history')
    if not train_end_ms<validation_end_ms or values[-1,2]>=validation_end_ms:raise ValueError('Invalid fixed chronological window')
    times=np.ascontiguousarray(values[:,2])
    prefix=ArtistPrefix(settings['query_hash_seed']);catalog=HistoricalCatalog(int(settings['hard_time_unit_seconds']*1000));neighbors=defaultdict(lambda:deque(maxlen=settings['max_neighbors']));rng=np.random.RandomState(seed)
    ids=[];negatives=[];snapshots=[];phis=[];elapsed=[];phases=[];alternate=[];raws=[];reason=np.full(len(frame),-1,dtype=np.int8);start=0
    def features(query,time,index):
        gap=query['gap_seconds'];count=query['prior_own_count'];day=(time/1000)%86400
        return [0.1+0.9*math.exp(-gap/3600),math.log1p(gap),math.log1p(count),count/math.sqrt(max(1,index)),math.sin(2*math.pi*day/86400),math.cos(2*math.pi*day/86400),math.log1p(query['total_competitors']),math.log1p(query['same_artist']),int(query['reference_status']==1),math.log1p(query['unknown_artist']),query['alternate_available']]
    while start<len(frame):
        time=int(values[start,2]);stop=int(np.searchsorted(times,time,side='right'));phase=0 if time<train_end_ms else 1
        forbidden=defaultdict(set)
        for user,track,_,_ in values[start:stop]:
            if track>=0:forbidden[int(user)].add(int(track))
        for i in range(start,stop):
            user,track=map(int,values[i,:2])
            if track<0:reason[i]=1;continue
            positive=prefix.query(user,track,time)
            if positive is None:reason[i]=2;continue
            if positive['gap_seconds']<settings['minimum_gap_seconds']:reason[i]=3;continue
            if phase==0 and int.from_bytes(hashlib.sha256(f"lastfm-model-query:{settings['query_hash_seed']}:{i}".encode()).digest(),'big')%settings['training_query_modulus']:
                reason[i]=4;continue
            negative=catalog.choose(user,positive['prior_own_count'],time-prefix.own[user,track].time,time,forbidden[user],rng)
            if negative is None:reason[i]=5;continue
            neg=prefix.query(user,negative,time);assert neg is not None
            snapshot=np.full((3,settings['max_neighbors']),-1,dtype=np.int32)
            for slot,node in enumerate((('u',user),('v',track),('v',negative))):
                history=list(neighbors[node]);snapshot[slot,:len(history)]=history
                if history:assert (values[history,2]<time).all()
            ids.append(i);negatives.append(negative);snapshots.append(snapshot);phis.append([features(positive,time,start),features(neg,time,start)]);elapsed.append([positive['gap_seconds'],neg['gap_seconds']]);phases.append(phase);alternate.append([positive['alternate_exposure'],neg['alternate_exposure']]);raws.append([positive,neg]);reason[i]=0
            if len(ids)>settings['max_prepared_queries_per_seed']:raise ValueError('Fixed query bound exceeded; no truncation')
        # Entire current timestamp is queried before any update.
        rows=[(int(u),int(v),int(a)) for u,v,_,a in values[start:stop]];prefix.update_group(time,rows);own_counts=Counter((u,v) for u,v,_ in rows if v>=0)
        for (user,track),count in sorted(own_counts.items()):catalog.update(user,track,time,count)
        for i in range(start,stop):
            user,track=map(int,values[i,:2]);neighbors['u',user].append(i)
            if track>=0:neighbors['v',track].append(i)
        start=stop
    if not ids:raise ValueError('No supported matched candidates')
    if not set(phases)=={0,1}:raise ValueError('Empty training or validation support; no fallback')
    stream=CandidateStream(np.asarray(ids,dtype=np.int64),np.asarray(negatives,dtype=np.int64),np.stack(snapshots),np.asarray(phis,dtype=np.float32),np.asarray(elapsed,dtype=np.float32),np.asarray(phases,dtype=np.int8),np.zeros(len(ids),bool),np.array([],np.int64),np.flatnonzero(reason==5),np.array([],np.int64),train_end_ms/1000,validation_end_ms/1000)
    raw=np.asarray([[[q[k] for k in ('total_competitors','same_artist','unknown_artist','reference_status','reference_artist','prior_own_count','alternate_available')] for q in pair] for pair in raws],dtype=np.int64)
    return stream,dict(alternate_exposure=np.asarray(alternate,dtype=np.int64),raw=raw,reason=reason)
