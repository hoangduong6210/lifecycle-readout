"""The exact restricted condition and constructive separation from phasewise decline."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from lifecycle_readout.modeling.readout_aware_tilt import ReadoutAwareTilt


def raw_tilt_flow(initial,rates,t,tilt):
    active=initial[:,1:]*torch.exp(-t[:,None]*rates)
    z=torch.cat((1-active.sum(-1,keepdim=True),active),-1)
    return torch.softmax(z.log()+tilt.log(),-1)


@pytest.mark.parametrize("phases",[2,3,5,9])
def test_parameterized_readout_is_monotone_for_unequal_rates(phases):
    gen=torch.Generator().manual_seed(101+phases)
    model=ReadoutAwareTilt(phases).double()
    with torch.no_grad():
        model.raw_gap.copy_(torch.randn(phases-1,generator=gen,dtype=torch.double)*2)
        model.raw_tilt.copy_(torch.randn(phases-1,generator=gen,dtype=torch.double)*4)
    initial=torch.softmax(torch.randn(31,phases,generator=gen,dtype=torch.double),-1)
    rates=torch.exp(torch.linspace(-5,5,phases-1,dtype=torch.double))
    time=torch.linspace(0,10,31,dtype=torch.double).requires_grad_()
    out=model(initial,time,rates)
    derivative,=torch.autograd.grad(out['prob'].sum(),time)
    assert (derivative<=1e-12).all()
    torch.testing.assert_close(out['state'].sum(-1),torch.ones_like(time))
    gaps,tilt=out['gaps'],out['log_tilt'].exp()
    assert (((gaps[-1]-gaps)*tilt[1:])<=gaps[-1]*tilt[0]+1e-12).all()


def test_phase_mass_can_increase_while_certified_readout_decreases():
    # delta=(1,2), c=(1,1.5,4) satisfies the exact readout condition.
    initial=torch.tensor([[.2,.2,.6]],dtype=torch.double)
    rates=torch.tensor([.1,5.],dtype=torch.double)
    time=torch.zeros(1,dtype=torch.double,requires_grad=True)
    q=raw_tilt_flow(initial,rates,time,torch.tensor([1.,1.5,4.],dtype=torch.double))
    phase_slope,=torch.autograd.grad(q[:,1].sum(),time,retain_graph=True)
    readout_slope,=torch.autograd.grad((q[:,1]+2*q[:,2]).sum(),time)
    assert phase_slope>0 and readout_slope<0
    model=ReadoutAwareTilt(3).double()
    with torch.no_grad():
        model.raw_gap.fill_(torch.log(torch.expm1(torch.tensor(1.,dtype=torch.double))))
        model.raw_tilt.copy_(torch.tensor([1/3,4.],dtype=torch.double).log())
    out=model(initial,time,rates)
    torch.testing.assert_close(out['state'],q,rtol=1e-12,atol=1e-12)
    model_phase_slope,=torch.autograd.grad(out['state'][:,1].sum(),time,retain_graph=True)
    model_score_slope,=torch.autograd.grad(out['prob'].sum(),time)
    assert model_phase_slope>0 and model_score_slope<0


def test_violating_condition_produces_an_increasing_readout():
    initial=torch.tensor([[.19,.01,.8]],dtype=torch.double)
    rates=torch.tensor([1000.,.1],dtype=torch.double)
    time=torch.zeros(1,dtype=torch.double,requires_grad=True)
    tilt=torch.tensor([1.,4.,1.],dtype=torch.double)
    q=raw_tilt_flow(initial,rates,time,tilt)
    derivative,=torch.autograd.grad((q[:,1]+2*q[:,2]).sum(),time)
    assert derivative>0


def test_vertex_condition_matches_all_exact_affine_numerator_minima():
    gen=torch.Generator().manual_seed(39)
    for phases in [2,3,7]:
        gaps=torch.rand(25,phases-1,generator=gen,dtype=torch.double)*4
        tilt=torch.exp(torch.randn(25,phases,generator=gen,dtype=torch.double)*2)
        # Affine partial-derivative numerator at dormant and active vertices.
        vertices=tilt[:,None,1:]*(gaps[:,:,None]*tilt[:,1:,None]-gaps[:,None,:]*(tilt[:,1:,None]-tilt[:,:1,None]))
        explicit=(vertices>=-1e-12).all(dim=-1).all(dim=-1)
        reduced=((gaps.max(-1,keepdim=True).values-gaps)*tilt[:,1:]<=gaps.max(-1,keepdim=True).values*tilt[:,:1]+1e-12).all(-1)
        torch.testing.assert_close(explicit,reduced)


def test_boundary_and_gradients_through_the_constrained_parameters():
    model=ReadoutAwareTilt(3).double()
    initial=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.double)
    out=model(initial,torch.tensor([0.,.3,1e6],dtype=torch.double),torch.tensor([.2,2.],dtype=torch.double))
    assert torch.isfinite(out['prob']).all() and (out['state']>=0).all()
    torch.testing.assert_close(out['state'][[0,2]],initial[[0,0]])
    out['logit'].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
