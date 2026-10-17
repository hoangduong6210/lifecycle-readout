"""Numerical checks of a sufficient bound for the ideal unclamped simplex map.

These support an analytic derivation, not a proof from random trials, a test of
trained checkpoints, or a constraint implemented in the training pipeline.
"""
import math
import pytest
import torch


def ideal_map(initial, rates, times, constant, varying, frequency):
    active = initial[:, 1:] * torch.exp(-times[:, None]*rates)
    state = torch.cat([1-active.sum(1, keepdim=True), active], dim=1)
    matrix = constant[None] + torch.sin(frequency*times[:, None, None])*varying[None]
    logits = state.log() + torch.bmm(state[:, None, :], matrix).squeeze(1)
    return torch.softmax(logits, dim=1), state


@pytest.mark.parametrize('phases', [2, 5, 8])
@pytest.mark.parametrize('varying_transition', [False, True])
def test_sufficient_bound_controls_every_active_coordinate(phases, varying_transition):
    generator = torch.Generator().manual_seed(1701+phases)
    dtype = torch.float64
    samples, magnitude, frequency = 32, .05, .8
    initial = .1 + torch.rand((samples, phases), generator=generator, dtype=dtype)
    initial /= initial.sum(1, keepdim=True)
    # Include an almost empty dormant phase and an almost empty active phase.
    initial[0, 0] = 1e-10
    initial[0, 1:] = (1-1e-10)/(phases-1)
    initial[1, 1] = 1e-10
    initial[1, 0] += 1-initial[1].sum()
    rates = torch.linspace(.8, 1., phases-1, dtype=dtype)
    times = torch.linspace(0., 3., samples, dtype=dtype).requires_grad_()
    constant = magnitude*(torch.rand((phases, phases), generator=generator, dtype=dtype)-.5)
    varying = magnitude*(torch.rand((phases, phases), generator=generator, dtype=dtype)-.5)
    mask = 1-torch.eye(phases, dtype=dtype)
    constant *= mask
    varying *= mask if varying_transition else 0.
    drift = magnitude*.5*frequency if varying_transition else 0.
    coefficient = math.exp(2*magnitude)-math.exp(-2*magnitude)+4*magnitude
    upper = -rates.min().item() + rates.max().item()*coefficient + 2*drift
    assert upper < 0
    q, state = ideal_map(initial, rates, times, constant, varying, frequency)
    assert (state > 0).all()
    for k in range(1, phases):
        derivative, = torch.autograd.grad(q[:, k].sum(), times, retain_graph=True)
        assert (derivative < 0).all()
        assert (derivative/q[:, k] <= upper+2e-9).all()


def test_zero_transition_recovers_the_exact_pure_flow():
    initial = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    rates = torch.tensor([.8, 1.], dtype=torch.float64)
    times = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros((3, 3), dtype=torch.float64)
    q, state = ideal_map(initial, rates, times, zero, zero, 0.)
    torch.testing.assert_close(q, state, rtol=1e-14, atol=1e-14)
    for k in [1, 2]:
        derivative, = torch.autograd.grad(q[:, k].sum(), times, retain_graph=True)
        torch.testing.assert_close(derivative, -rates[k-1]*q[:, k], rtol=1e-14, atol=1e-14)


def test_small_transition_magnitude_without_drift_control_is_insufficient():
    # T_01(t)=.01*sin(1000*t): at t=0, q_1'=.25*(-2+.01*1000/2)=.75.
    initial = torch.tensor([[.5, .5]], dtype=torch.float64)
    rates = torch.ones(1, dtype=torch.float64)
    times = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    constant = torch.zeros((2, 2), dtype=torch.float64)
    varying = torch.tensor([[0., .01], [0., 0.]], dtype=torch.float64)
    q, _ = ideal_map(initial, rates, times, constant, varying, 1000.)
    derivative, = torch.autograd.grad(q[:, 1].sum(), times)
    assert derivative.item() == pytest.approx(.75, abs=1e-12)
    # Ignoring the time derivative would falsely suggest the constant-map bound holds.
    coefficient = math.exp(.02)-math.exp(-.02)+.04
    assert coefficient < 1
