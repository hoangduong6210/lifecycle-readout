"""Independent generator reference and expressivity boundaries for event flow."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.linalg import expm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from lifecycle_readout.competitive_lifecycle import competitive_step


def inputs():
    z=torch.tensor([[.2,.3,.5],[0.,.7,.3],[1.,0.,0.]],dtype=torch.float64)
    t=torch.tensor([2.,0.,4.],dtype=z.dtype)
    e=torch.tensor([1.2,8.,0.],dtype=z.dtype)
    d=torch.tensor([.2,.9],dtype=z.dtype)
    g=torch.tensor([.8,.1],dtype=z.dtype)
    return z,t,e,d,g


def test_independent_markov_generator_and_boundary_mass():
    z,t,e,d,g=inputs()
    out=competitive_step(z,t,e,d,g)
    for i in range(len(z)):
        dose=(t[i]*d+e[i]*g).numpy()
        generator=np.zeros((3,3))
        generator[1:,0]=dose
        generator[1:,1:]=-np.diag(dose)
        np.testing.assert_allclose(out[i].numpy(),z[i].numpy()@expm(generator),atol=2e-15)
    assert (out>=0).all()
    torch.testing.assert_close(out.sum(-1),torch.ones_like(t))
    torch.testing.assert_close(out[-1],z[-1])


def test_sequential_events_and_time_commute_and_equal_aggregate():
    z,t,e,d,g=inputs()
    zero=torch.zeros_like(t)
    q=competitive_step(z,t,zero,d,g)
    for part in (.2,.3,.5):
        q=competitive_step(q,zero,part*e,d,g)
    alternate=competitive_step(competitive_step(z,zero,e,d,g),t,zero,d,g)
    torch.testing.assert_close(q,competitive_step(z,t,e,d,g),rtol=1e-14,atol=1e-15)
    torch.testing.assert_close(q,alternate,rtol=1e-14,atol=1e-15)


def test_proportional_rates_are_exactly_one_exposure_clock():
    z,t,e,d,_=inputs()
    a=1.7
    coupled=competitive_step(z,t,e,d,a*d)
    scalar=competitive_step(z,t+a*e,torch.zeros_like(e),d,torch.zeros_like(d))
    torch.testing.assert_close(coupled,scalar,rtol=1e-14,atol=1e-15)


def test_nonproportional_rates_have_no_common_scalar_clock_for_interior_state():
    z,t,e,d,g=inputs()
    q=competitive_step(z[:1],t[:1],e[:1],d,g)
    implied_time=-(q[:,1:]/z[:1,1:]).log()/d
    torch.testing.assert_close(implied_time,t[:1,None]+e[:1,None]*(g/d))
    assert not torch.isclose(implied_time[0,0],implied_time[0,1])


def test_differentiable_and_conditional_readout_decreases_in_exposure():
    z,t,e,d,g=inputs()
    e=e.requires_grad_(); d=d.requires_grad_(); g=g.requires_grad_()
    # An interior subset avoids finite differences outside the nonnegative domain.
    assert torch.autograd.gradcheck(lambda ee,dd,gg:competitive_step(z[:1],t[:1],ee,dd,gg),
                                   (e[:1],d,g))
    q=competitive_step(z,t,e,d,g)
    weights=z.new_tensor([-2.,.1,3.])
    slope=torch.autograd.grad((q*weights).sum(),e)[0]
    assert (slope<=0).all()
    expected=-(z[:,1:]*torch.exp(-t[:,None]*d-e[:,None]*g)*g*(weights[1:]-weights[0])).sum(-1)
    torch.testing.assert_close(slope,expected)


def test_large_and_tiny_doses_without_floor_or_euler_negativity():
    z=torch.tensor([[0.,1.]],dtype=torch.float64)
    zero=z.new_zeros(1); d=z.new_ones(1)
    tiny=competitive_step(z,zero,z.new_tensor([1e-20]),d,d)
    assert tiny[0,0]>0
    huge=competitive_step(z,zero,z.new_tensor([1e6]),d,d)
    torch.testing.assert_close(huge,z.new_tensor([[1.,0.]]))
    # The literal linear counting-process jump differs and can be negative.
    assert 1.-2.<0
    exact=competitive_step(z,zero,z.new_tensor([2.]),d,d)
    torch.testing.assert_close(exact[0,1],torch.exp(z.new_tensor(-2.)))


@pytest.mark.parametrize("kind",["negative","nan","simplex","shape","dtype","overflow"])
def test_invalid_domain_rejected(kind):
    z,t,e,d,g=inputs()
    if kind=="negative": e[0]=-1
    elif kind=="nan": g[0]=float("nan")
    elif kind=="simplex": z[0,0]=.9
    elif kind=="shape": d=d[:1]
    elif kind=="dtype": t=t.float()
    elif kind=="overflow": t[0]=1e308; d[0]=1e308
    with pytest.raises(ValueError): competitive_step(z,t,e,d,g)
