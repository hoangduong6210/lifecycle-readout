"""Descriptive strictly-earlier recurrence gates for dataset intake, no predictor."""
import hashlib
import numpy as np
import pandas as pd


def select_users(ids,n=100,seed=17):
    users=list(map(int,ids))
    if len(set(users))!=len(users) or n<1 or n>len(users):raise ValueError('Unique users and feasible sample required')
    return sorted(users,key=lambda u:(hashlib.sha256(f'kuairand-intake:{seed}:{u}'.encode()).digest(),u))[:n]


def recurrence_summary(events):
    """All rows at one timestamp query before the entire timestamp-group update.

    events has source, destination and millisecond timestamp. Multiplicity is
    retained. Competitor counts exclude ALL events at the last own timestamp
    and current query timestamp. This describes observed input events; it is not
    an online availability certificate for delayed click/watch feedback.
    """
    if set(events.columns)!={'source','destination','timestamp_ms'}:raise ValueError('Expected three event columns')
    if len(events)==0:return dict(rows=0,users=0,pairs=0,recurrent_rows=0,recurrent_fraction=0.,long_gap={})
    if events.isna().any().any() or not np.isfinite(events.to_numpy()).all():raise ValueError('Finite complete events required')
    frame=events.sort_values(['source','timestamp_ms','destination'])
    source=(frame.groupby(['source','timestamp_ms']).size().rename('source_count').reset_index())
    source['source_after']=source.groupby('source')['source_count'].cumsum()
    source['source_before']=source['source_after']-source['source_count']
    pair=(frame.groupby(['source','destination','timestamp_ms']).size().rename('multiplicity').reset_index())
    pair=pair.merge(source[['source','timestamp_ms','source_before','source_after']],on=['source','timestamp_ms'],validate='many_to_one').sort_values(['source','destination','timestamp_ms'])
    grouped=pair.groupby(['source','destination'],sort=False)
    pair['previous_time']=grouped['timestamp_ms'].shift()
    pair['prior_count']=grouped['multiplicity'].cumsum()-pair['multiplicity']
    pair['previous_source_after']=grouped['source_after'].shift()
    recurrent=pair['prior_count']>0
    pair['gap_seconds']=(pair['timestamp_ms']-pair['previous_time'])/1000
    pair['competitors']=pair['source_before']-pair['previous_source_after']
    if (pair.loc[recurrent,'gap_seconds']<=0).any() or (pair.loc[recurrent,'competitors']<0).any():raise ArithmeticError('Invalid strict-past recurrence')
    rows=int(pair.loc[recurrent,'multiplicity'].sum())
    result=dict(rows=len(frame),users=int(frame['source'].nunique()),pairs=int(len(grouped)),recurrent_rows=rows,recurrent_fraction=rows/len(frame),recurrent_users=int(pair.loc[recurrent,'source'].nunique()),recurrent_with_competitors_rows=int(pair.loc[recurrent & (pair['competitors']>0),'multiplicity'].sum()),long_gap={})
    for threshold in (60,3600,86400):
        selected=recurrent & (pair['gap_seconds']>=threshold)
        result['long_gap'][str(threshold)]=dict(rows=int(pair.loc[selected,'multiplicity'].sum()),users=int(pair.loc[selected,'source'].nunique()),with_competitors=int(pair.loc[selected & (pair['competitors']>0),'multiplicity'].sum()),without_competitors=int(pair.loc[selected & (pair['competitors']==0),'multiplicity'].sum()))
    return result
