from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pytest
from lifecycle_readout.lastfm_diagnostics import cohort_masks,paired_metrics,phase_terms


def test_half_open_query_cohorts_partition_and_keep_pairs():
    gap=np.array([3600,86399,86400,604799,604800]);dose=np.array([0,1,9,10,20]);m=cohort_masks(gap,dose)
    assert len(m)==16 and m['gap_1d_to_7d'].tolist()==[False,False,True,True,False]
    assert m['dose_1_to_9'].tolist()==[False,True,True,False,False]
    joint=[v for k,v in m.items() if '__' in k];np.testing.assert_array_equal(np.sum(joint,axis=0),np.ones(5))
    assert paired_metrics([.9,.1],[.8,.2],np.array([True,False]))==dict(ap=1.,auc=1.,pair_accuracy=1.)
    assert paired_metrics([],[])['ap'] is None
    with pytest.raises(ValueError):cohort_masks([3599],[1])


def test_exact_phase_components_bound_and_permutation_invariance():
    t=np.array([[0.,1.,1e6]]);e=np.array([[0.,7.,1e6]]);d=np.array([.1,.5,1.,3.]);h=np.array([1.,.3,.1,2.]);g=np.array([.2,1.,.5,2.]);a=.7
    terms,bound=phase_terms(t,e,d,h,g,a,5.,3.);assert terms[0,0].sum()==0 and np.all(terms<=0) and np.all(terms.sum(-1)>=-bound-1e-15)
    permutation=[3,1,0,2];other,_=phase_terms(t,e,d[permutation],h[permutation],g[permutation],a,5.,3.);np.testing.assert_allclose(terms.sum(-1),other.sum(-1),rtol=0,atol=1e-15)
    assert abs(terms[0,-1].sum()+bound)<1e-15
    with pytest.raises(ValueError):phase_terms(t,e,d,h,g,-1.,5.,3.)
