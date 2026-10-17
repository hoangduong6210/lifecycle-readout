"""Independent cadence clocks at tied timestamps and held-out train exclusions."""
from pathlib import Path
import importlib.util
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('cadence_checkpoint_audit',Path(__file__).parents[1]/'scripts/audit_cadence_checkpoint_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_clock_uses_strict_past_and_train_fallback_with_ties():
    data=dict(sources=np.zeros(8,dtype=int),destinations=np.array([1,2,1,2,1,2,1,2]),
              timestamps=np.array([1.,1.,3.,3.,5.,5.,9.,9.]))
    candidates=dict(event_ids=np.array([4,5,6,7]),negative_destinations=np.array([2,1,2,1]),
                    split=np.array([0,0,1,2]),excluded_training_ids=np.array([],dtype=int))
    extra,fallback=module.independent_clock(data,candidates,1.)
    assert fallback==2.
    np.testing.assert_array_equal(extra[:3,:,1],np.ones((3,2)))
    np.testing.assert_allclose(extra[:2,:,2],np.log1p(2/3))
    np.testing.assert_allclose(extra[2,:,2],np.log1p(4/3))
    np.testing.assert_array_equal(extra[3],np.zeros((2,3)))


def test_excluded_train_event_cannot_create_cadence():
    data=dict(sources=np.zeros(7,dtype=int),destinations=np.array([1,2,1,2,1,2,1]),
              timestamps=np.array([1.,1.,3.,3.,5.,5.,9.]))
    candidates=dict(event_ids=np.array([4,5,6]),negative_destinations=np.array([2,1,2]),
                    split=np.array([0,0,1]),excluded_training_ids=np.array([2]))
    extra,fallback=module.independent_clock(data,candidates,1.)
    assert fallback==2.
    np.testing.assert_array_equal(extra[:2,:,1],[[0,1],[1,0]])
    assert extra[2,0,1]==1
