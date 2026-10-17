from pathlib import Path
import sys,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from lifecycle_readout.modeling.fixed_rate_tilt import rate_cone_check
from lifecycle_readout.modeling.inverse_tilt_projection import inverse_tilt_constraints,all_rate_inverse_projection,project_inverse_tilts


def enumerate_qp(raw,A,b):
    candidates=[];m=len(raw)
    for k in range(min(m,len(b))+1):
        for ids in itertools.combinations(range(len(b)),k):
            if k:
                rows=A[list(ids)];gram=rows@rows.T
                if np.linalg.matrix_rank(gram)<k:continue
                dual=np.linalg.solve(gram,b[list(ids)]-rows@raw)
                if dual.min()< -1e-8:continue
                value=raw+rows.T@dual
            else:value=raw.copy()
            if len(b)==0 or np.min(A@value-b)>=-1e-8:candidates.append(value)
    assert candidates;return min(candidates,key=lambda x:np.sum((x-raw)**2))


def test_coordered_strict_distance_separation():
    raw=np.array([.5,.5]);g=np.array([1.,2.]);result=project_inverse_tilts(raw,g,[1,2]);np.testing.assert_array_equal(result['inverse_tilt'],raw);assert result['cycles']==0 and len(result['b'])==0
    old=all_rate_inverse_projection(raw,g);np.testing.assert_allclose(np.sum((old-raw)**2),(3-2*np.sqrt(2))/4,atol=1e-15)


def test_known_coupled_projection_and_independent_solution():
    raw=np.array([.5,.5]);g=np.array([1.,2.]);rates=np.array([10.,1.]);result=project_inverse_tilts(raw,g,rates);assert result['squared_distance']>0;expected=enumerate_qp(raw,result['A'],result['b']);np.testing.assert_allclose(result['inverse_tilt'],expected,atol=1e-9)
    c=np.r_[1,1/result['inverse_tilt']**2];assert rate_cone_check(c,g,[rates])['status']!='FAIL'


def test_random_cones_match_exhaustive_active_sets_and_old_distance():
    rng=np.random.default_rng(66103)
    for m in [2,3,4]:
        for _ in range(15):
            raw=np.exp(rng.uniform(-2,1,m));g=rng.uniform(.01,3,m);rates=np.exp(rng.uniform(-2,2,(2,m)));result=project_inverse_tilts(raw,g,rates,tolerance=1e-11);expected=enumerate_qp(raw,result['A'],result['b']);np.testing.assert_allclose(result['inverse_tilt'],expected,rtol=0,atol=1e-8);old=all_rate_inverse_projection(raw,g);assert result['squared_distance']<=np.sum((old-raw)**2)+1e-8;assert np.all(result['inverse_tilt']>=raw)
            c=np.r_[1,1/result['inverse_tilt']**2];assert rate_cone_check(c,g,rates,relative_tolerance=1e-8)['status']!='FAIL'


def test_batched_zero_gap_and_orthant_recovery():
    raw=np.array([[.2,.3],[1.,2.]]);g=np.array([0.,1.]);answer=project_inverse_tilts(raw,g,np.eye(2));expected=all_rate_inverse_projection(raw,g);np.testing.assert_allclose(answer['inverse_tilt'],expected,atol=1e-9)
    np.testing.assert_array_equal(project_inverse_tilts(raw,[0,0],[[2,1]])['inverse_tilt'],raw)


def test_invalid_input_and_budget_fail_instead_of_accepting():
    with pytest.raises(ValueError):project_inverse_tilts([0,.1],[1,2],[2,1])
    with pytest.raises(ValueError):project_inverse_tilts([.1,.1],[1,-2],[2,1])
    with pytest.raises(ValueError):project_inverse_tilts([.1,.1],[1,2],[2,1],max_cycles=0)
