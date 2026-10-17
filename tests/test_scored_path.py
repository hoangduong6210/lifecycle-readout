"""Independent autodiff checks of the directional certificate and its scope."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from lifecycle_readout.scored_path import entropic_direction, linear_readout_direction


@pytest.mark.parametrize("phases", [2, 5, 9])
@pytest.mark.parametrize("clamped", [False, True])
def test_exact_chain_rule_including_transition_drift(phases, clamped):
    gen=torch.Generator().manual_seed(91+phases)
    z0=torch.softmax(torch.randn(7,phases,generator=gen,dtype=torch.double),-1)
    if clamped:
        z0[:,1]=1e-9;z0=z0/z0.sum(-1,keepdim=True)
    d=torch.linspace(.2,2.,phases-1,dtype=torch.double)
    t=torch.linspace(.1,1.,7,dtype=torch.double).requires_grad_()
    active=z0[:,1:]*torch.exp(-t[:,None]*d)
    z=torch.cat((1-active.sum(-1,keepdim=True),active),-1)
    velocity=torch.cat(((active*d).sum(-1,keepdim=True),-active*d),-1)
    constant=torch.randn(phases,phases,generator=gen,dtype=torch.double)
    drift=torch.randn(phases,phases,generator=gen,dtype=torch.double)
    matrix=constant+t[:,None,None].sin()*drift
    matrix_velocity=t[:,None,None].cos()*drift
    q,derivative,h=entropic_direction(z,velocity,matrix,matrix_velocity)
    actual=torch.softmax(z.clamp(min=1e-6).log()+torch.einsum('bi,bij->bj',z,matrix),-1)
    torch.testing.assert_close(q,actual,rtol=1e-12,atol=1e-12)
    for k in range(phases):
        measured,=torch.autograd.grad(actual[:,k].sum(),t,retain_graph=True)
        torch.testing.assert_close(derivative[:,k],measured,rtol=1e-11,atol=1e-12)
    weights=torch.randn(phases,generator=gen,dtype=torch.double)
    measured,=torch.autograd.grad((actual*weights).sum(),t)
    covariance=(q*(weights-(q*weights).sum(-1,keepdim=True))*(h-(q*h).sum(-1,keepdim=True))).sum(-1)
    torch.testing.assert_close(linear_readout_direction(q,derivative,weights),measured)
    torch.testing.assert_close(covariance,measured)


def test_row_common_transition_gauge_and_drift_cancel():
    z=torch.tensor([[.2,.3,.5]],dtype=torch.double)
    v=torch.tensor([[.8,-.3,-.5]],dtype=torch.double)
    matrix=torch.arange(9,dtype=torch.double).reshape(1,3,3)/9
    drift=torch.zeros_like(matrix)
    base=entropic_direction(z,v,matrix,drift)
    a=torch.tensor([[[1000.],[-1000.],[300.]]],dtype=torch.double)
    adot=torch.tensor([[[-500.],[200.],[1700.]]],dtype=torch.double)
    shifted=entropic_direction(z,v,matrix+a,drift+adot)
    torch.testing.assert_close(base[0],shifted[0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(base[1],shifted[1],rtol=1e-11,atol=1e-12)


def test_nonsmooth_clamp_boundary_rejected():
    z=torch.tensor([[1e-6,1-1e-6]],dtype=torch.double)
    with pytest.raises(ValueError,match='nonsmooth'):
        entropic_direction(z,torch.zeros_like(z),torch.zeros(1,2,2,dtype=torch.double),torch.zeros(1,2,2,dtype=torch.double))
