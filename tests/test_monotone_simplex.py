"""Constructive derivative, boundary and optimization checks; no benchmark fit."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifecycle_readout.modeling.monotone_simplex import MonotoneSimplexTransition


@pytest.mark.parametrize("phases", [2, 3, 5, 9])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_all_active_derivatives_and_arbitrary_gap_gate(phases, nonlinear):
    gen = torch.Generator().manual_seed(430 + phases)
    module = MonotoneSimplexTransition(phases, nonlinear).double()
    with torch.no_grad():
        module.transition_logits.copy_(torch.randn(phases-1, phases, generator=gen)*3)
    z = torch.softmax(torch.randn(17, phases, generator=gen, dtype=torch.double), -1)
    rates = torch.exp(torch.linspace(-4, 4, phases-1, dtype=torch.double))
    time = torch.linspace(0, 12, 17, dtype=torch.double).requires_grad_()
    state = module(z, time, rates)["state"]
    assert (state >= -1e-14).all()
    torch.testing.assert_close(state.sum(-1), torch.ones_like(time))
    for k in range(1, phases):
        derivative, = torch.autograd.grad(state[:, k].sum(), time, retain_graph=True)
        assert (derivative <= 1e-13).all()
    # Unordered active weights, all above dormant, suffice.
    gaps = torch.rand(phases-1, generator=gen, dtype=torch.double)*5
    score = torch.sigmoid(-2 + state[:, 1:] @ gaps)
    derivative, = torch.autograd.grad(score.sum(), time)
    assert (derivative <= 1e-13).all()
    epsilon = 1e-5
    interior = time.detach() + .1
    lo = module(z, interior-epsilon, rates)["state"][:, 1:] @ gaps
    hi = module(z, interior+epsilon, rates)["state"][:, 1:] @ gaps
    assert ((hi-lo)/(2*epsilon) <= 1e-10).all()


def test_boundary_and_long_silence_have_no_log_clamps():
    module = MonotoneSimplexTransition(3).double()
    z = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.double)
    output = module(z, torch.tensor([0., 0., 1e6], dtype=torch.double),
                    torch.tensor([.1, 10.], dtype=torch.double))["state"]
    torch.testing.assert_close(output[[0, 2]], z[[0, 0]])
    assert torch.isfinite(output).all() and (output >= 0).all()


def test_trainable_parameters_receive_gradients_and_preserve_constraint_after_steps():
    module = MonotoneSimplexTransition(3).double()
    optimizer = torch.optim.SGD(module.parameters(), lr=2.)
    z = torch.tensor([[.1, .2, .7]], dtype=torch.double)
    rates = torch.tensor([.5, 2.], dtype=torch.double, requires_grad=True)
    for _ in range(4):
        optimizer.zero_grad()
        output = module(z, torch.tensor([.3], dtype=torch.double), rates)["state"]
        loss = (output * torch.tensor([[0., 1., 3.]], dtype=torch.double)).sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in module.parameters())
        assert rates.grad is not None and rates.grad.abs().sum() > 0
        optimizer.step()
        time = torch.tensor([.4], dtype=torch.double, requires_grad=True)
        state = module(z, time, rates)["state"]
        derivative, = torch.autograd.grad((state[:, 1:] @
                                          torch.tensor([1., 3.], dtype=torch.double)).sum(), time)
        assert derivative < 0


def test_skew_symmetry_does_not_certify_entropic_candidate_gate():
    z = torch.tensor([.125, .16, .715], dtype=torch.double)
    rates = torch.tensor([2., .25], dtype=torch.double)
    matrix = torch.tensor([[0., 5., 12.], [-5., 0., 2.], [-12., -2., 0.]], dtype=torch.double)
    assert torch.equal(matrix.T, -matrix)
    time = torch.zeros((), dtype=torch.double, requires_grad=True)
    active = z[1:] * torch.exp(-rates*time)
    flow = torch.cat(((1-active.sum()).reshape(1), active))
    q = torch.softmax(flow.log() + matrix.T @ flow, -1)
    gate_logit = q @ torch.tensor([0., 1., 2.], dtype=torch.double)
    derivative, = torch.autograd.grad(gate_logit, time)
    torch.testing.assert_close(derivative, torch.tensor(.06660243621780697, dtype=torch.double))
    assert derivative > 0


def test_changing_context_is_outside_certificate():
    # Even an absorbing linear map can invert if its retention changes with time.
    time = torch.zeros((), dtype=torch.double, requires_grad=True)
    active = .5 * torch.exp(-time)
    retention = torch.sigmoid(10*time)
    derivative, = torch.autograd.grad(active*retention, time)
    assert derivative > 0


def test_zero_row_and_column_sums_do_not_repair_skew_candidate():
    z = torch.tensor([.125, .16, .715], dtype=torch.double)
    matrix = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]], dtype=torch.double)
    assert torch.equal(matrix.T, -matrix)
    assert torch.equal(matrix.sum(0), torch.zeros(3, dtype=torch.double))
    assert torch.equal(matrix.sum(1), torch.zeros(3, dtype=torch.double))
    time = torch.zeros((), dtype=torch.double, requires_grad=True)
    active = z[1:] * torch.exp(-torch.tensor([2., .25], dtype=torch.double)*time)
    flow = torch.cat(((1-active.sum()).reshape(1), active))
    q = torch.softmax(flow.log() + matrix.T @ flow, -1)
    derivative, = torch.autograd.grad(q[1]+2*q[2], time)
    assert derivative > 0


def test_linear_mixing_can_be_absorbed_into_readout_weights():
    module = MonotoneSimplexTransition(4, nonlinear=False).double()
    z = torch.tensor([[.1, .2, .3, .4]], dtype=torch.double)
    out = module(z, torch.ones(1, dtype=torch.double), torch.ones(3, dtype=torch.double))
    gaps = torch.tensor([2., .5, 1.], dtype=torch.double)
    torch.testing.assert_close(out["state"][:, 1:] @ gaps,
                               out["active_before"] @ (out["mixing"][:, 1:] @ gaps))


@pytest.mark.parametrize("case", ["simplex", "negative_time", "rate", "nan", "shape"])
def test_invalid_inputs_fail(case):
    module = MonotoneSimplexTransition(3).double()
    z = torch.tensor([[.2, .3, .5]], dtype=torch.double)
    t = torch.ones(1, dtype=torch.double)
    rates = torch.ones(2, dtype=torch.double)
    if case == "simplex": z *= 2
    if case == "negative_time": t *= -1
    if case == "rate": rates[0] = 0
    if case == "nan": z[0, 0] = float("nan")
    if case == "shape": rates = rates[:1]
    with pytest.raises(ValueError): module(z, t, rates)
