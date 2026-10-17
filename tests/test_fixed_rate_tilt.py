from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import torch
from lifecycle_readout.modeling.fixed_rate_tilt import fixed_rate_check,rate_cone_check,OrderedRateTilt


def direct_derivative(z,c,g,d):
    z=np.asarray(z);c=np.asarray(c);q=c*z/(c@z);v=np.r_[np.dot(d,z[1:]),-np.asarray(d)*z[1:]];dq=c*v/(c@z)-q*(c@v)/(c@z);return float(np.asarray(g)@dq[1:]),dq


def test_strict_separation_and_active_coordinate_increase():
    c=[1,4,4];g=[1,2];d=[1,2];check=fixed_rate_check(c,g,d);assert check['status']=='PASS';assert (max(g)-g[0])*c[1]>max(g)*c[0]
    derivative,dq=direct_derivative([.1,.1,.8],c,g,d);assert dq[1]>0 and derivative<0;np.testing.assert_allclose([dq[1],derivative],[56/1369,-1416/1369],rtol=1e-13)


def test_failure_constructs_real_derivative_witness_and_zero_gaps():
    for g,d in [([1,2],[10,1]),([0,1],[2,1]),([1,0],[1,2])]:
        check=fixed_rate_check([1,4,4],g,d);assert check['status']=='FAIL';w=check['witness'];assert w['readout_derivative']>0;observed,_=direct_derivative(w['initial'],[1,4,4],g,d);assert observed>0;np.testing.assert_allclose(observed,w['readout_derivative'],atol=1e-12)
    assert fixed_rate_check([1,4,4],[0,0],[10,1])['status']=='BOUNDARY'
    assert fixed_rate_check([1,100],[2],[4])['status']=='PASS'


def test_rate_cone_and_orthant_recover_all_rate_condition():
    rng=np.random.default_rng(7361)
    for m in [2,3,5]:
        for _ in range(30):
            c=np.exp(rng.uniform(-2,2,m+1));g=rng.uniform(.1,3,m);old=np.all((g.max()-g)*c[1:]<=g.max()*c[0]);new=rate_cone_check(c,g,np.eye(m));assert (new['status']!='FAIL')==old
    assert rate_cone_check([1,4,4],[1,2],[[1,2],[2,3]])['status']=='PASS'
    assert rate_cone_check([1,4,4],[1,2],[[1,2],[10,1]])['status']=='FAIL'


def test_random_certified_states_and_all_returned_witnesses():
    rng=np.random.default_rng(9027);counts={'PASS':0,'FAIL':0}
    for m in [2,3,5,9]:
        for _ in range(25):
            c=np.exp(rng.uniform(-3,3,m+1));g=rng.uniform(.01,3,m);d=np.exp(rng.uniform(-3,3,m));check=fixed_rate_check(c,g,d);counts[check['status']]+=1
            if check['status']=='FAIL':assert direct_derivative(check['witness']['initial'],c,g,d)[0]>0
            else:
                for z in rng.dirichlet(np.ones(m+1),size=30):assert direct_derivative(z,c,g,d)[0]<=1e-10
    assert counts['PASS']>0 and counts['FAIL']>0


def test_trainable_ordered_constructor_both_directions_boundaries_and_gradients():
    torch.manual_seed(421);model=OrderedRateTilt(5).double()
    with torch.no_grad():model.active_log_tilt.copy_(torch.tensor([3.,-2.,4.,1.]));model.raw_clock.copy_(torch.randn(4));model.raw_event.copy_(torch.randn(4));model.raw_gap.copy_(torch.randn(4))
    initial=torch.softmax(torch.randn(8,5,dtype=torch.float64),-1);t=torch.linspace(.01,4,8,dtype=torch.float64,requires_grad=True);e=torch.linspace(.1,10,8,dtype=torch.float64,requires_grad=True);output=model(initial,t,e);dt,de=torch.autograd.grad(output['logit'].sum(),(t,e),retain_graph=True);assert (dt<=1e-12).all() and (de<=1e-12).all();output['logit'].sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    check=rate_cone_check(torch.exp(output['log_tilt']).detach().numpy(),output['gaps'].detach().numpy(),np.stack([output['clock_rates'].detach().numpy(),output['event_rates'].detach().numpy()]));assert check['status']=='PASS'
    c=torch.exp(output['log_tilt']).detach().numpy();g=output['gaps'].detach().numpy();assert np.any((g.max()-g)*c[1:]>g.max()*c[0])
    boundary=torch.eye(5,dtype=torch.float64);q=model(boundary,torch.zeros(5,dtype=torch.float64),torch.zeros(5,dtype=torch.float64));assert torch.isfinite(q['logit']).all();np.testing.assert_allclose(q['state'].detach().numpy(),np.eye(5),atol=1e-15)


def test_invalid_domain_is_rejected():
    for c,g,d in [([0,1,1],[1,2],[1,2]),([1,1,1],[-1,2],[1,2]),([1,1,1],[1,2],[np.inf,2])]:
        with pytest.raises(ValueError):fixed_rate_check(c,g,d)


def test_boundary_time_gradient_and_exact_representability():
    model=OrderedRateTilt(3).double()
    with torch.no_grad():
        for parameter in (model.raw_gap,model.raw_clock,model.raw_event):parameter.copy_(torch.log(torch.expm1(torch.ones(2,dtype=torch.float64))))
        model.active_log_tilt.fill_(float(np.log(4)))
    t=torch.zeros(1,dtype=torch.float64,requires_grad=True);e=torch.zeros(1,dtype=torch.float64);initial=torch.tensor([[.1,.1,.8]],dtype=torch.float64);out=model(initial,t,e);dq=torch.autograd.grad(out['state'][:,1].sum(),t,retain_graph=True)[0];dg=torch.autograd.grad(out['logit'].sum(),t)[0];torch.testing.assert_close(dq,torch.tensor([56/1369],dtype=torch.float64));torch.testing.assert_close(dg,torch.tensor([-1416/1369],dtype=torch.float64))
    time=torch.zeros(1,dtype=torch.float64,requires_grad=True);boundary=torch.tensor([[0.,1.,0.]],dtype=torch.float64);result=model(boundary,time,e);derivative=torch.autograd.grad(result['logit'].sum(),time)[0];torch.testing.assert_close(derivative,torch.tensor([-.25],dtype=torch.float64))
