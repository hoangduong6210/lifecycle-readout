"""Timestamp-group and sampling invariants for the prospective intake."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.intake_recurrence import select_users,recurrence_summary


def test_sample_is_id_based_and_order_independent():
    ids=np.arange(1000)
    a=select_users(ids);b=select_users(ids[::-1]);assert a==b and len(a)==len(set(a))==100
    with pytest.raises(ValueError):select_users([1,1],1)


def test_ties_exclude_entire_previous_and_current_groups():
    # At t=0 both a and b occur. At t=1h both a and c occur.
    # Neither b at the old boundary nor c at the query boundary is a competitor.
    events=pd.DataFrame(dict(source=[1]*5,destination=[10,11,10,12,10],timestamp_ms=[0,0,3600000,3600000,7200000]))
    x=recurrence_summary(events)
    assert x['recurrent_rows']==2 and x['recurrent_with_competitors_rows']==0
    assert x['long_gap']['3600']==dict(rows=2,users=1,with_competitors=0,without_competitors=2)


def test_intervening_events_and_tied_multiplicity_are_counted():
    events=pd.DataFrame(dict(source=[1]*6,destination=[10,11,11,10,10,10],timestamp_ms=[0,1000,1000,3600000,3600000,7200000]))
    x=recurrence_summary(events)
    assert x['recurrent_rows']==3 and x['recurrent_with_competitors_rows']==2
    assert x['long_gap']['3600']['rows']==3
    assert x['long_gap']['3600']['with_competitors']==2
    assert x==recurrence_summary(events.iloc[::-1])


def test_sources_are_independent_and_empty_input_is_explicit():
    events=pd.DataFrame(dict(source=[1,2,1],destination=[10,11,10],timestamp_ms=[0,1000,3600000]))
    assert recurrence_summary(events)['recurrent_with_competitors_rows']==0
    assert recurrence_summary(events.iloc[:0])['rows']==0
