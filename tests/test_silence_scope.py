"""Constructive checks of the distinction between silence flow and scored jumps."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.modeling.latent_lifecycle import (
    LatentLifecycleConfig, LatentLifecycleModule,
)


def constant_jump_model(phases=2, off_diagonal=3.):
    """An admissible model with constant encoder, transition and affinity."""
    model = LatentLifecycleModule(LatentLifecycleConfig(
        n_phases=phases, hidden_dim=8, decay_init=1.)).double().eval()
    initial = torch.tensor([.375, .625] if phases == 2 else
                           [.375, .595, .01, .01, .01], dtype=torch.float64)
    inverse_softplus_one = math.log(math.expm1(1.))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.encoder.net[-1].bias.copy_(initial.log())
        model.kernel.decay_param.fill_(inverse_softplus_one)
        matrix = torch.zeros((phases, phases), dtype=torch.float64)
        matrix[0, 1] = matrix[1, 0] = off_diagonal
        model.kernel.trans_net[-1].bias.copy_(matrix.flatten())
        model.survival_gate.deltas.fill_(inverse_softplus_one)
    return model


def scored(model, time):
    return model(torch.zeros((1, 8), dtype=torch.float64),
                 torch.zeros((1, 6), dtype=torch.float64), time,
                 torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize('phases', [2, 5])
def test_fixed_transition_can_reverse_the_full_score_with_ordered_weights(phases):
    model = constant_jump_model(phases)
    time = torch.tensor([math.log(1.25)], dtype=torch.float64, requires_grad=True)
    out = scored(model, time)
    pure_state = model.kernel.continuous_decay(out['z_curr'], time)
    pure_score = .5 * model.survival_gate(pure_state)
    pure_derivative, = torch.autograd.grad(pure_score.sum(), time, retain_graph=True)
    scored_derivative, = torch.autograd.grad(out['gated_prob'].sum(), time)
    assert pure_derivative.item() < 0 < scored_derivative.item()
    # This example does not depend on a changing encoder/base/T or a clamp boundary.
    assert (pure_state > 1e-6).all()
    assert (torch.diff(model.survival_gate.survival_weights) > 0).all()
    epsilon = 1e-5
    low = scored(model, time.detach() - epsilon)
    high = scored(model, time.detach() + epsilon)
    torch.testing.assert_close(low['T_matrix'], high['T_matrix'], rtol=0, atol=0)
    torch.testing.assert_close(low['z_curr'], high['z_curr'], rtol=0, atol=0)
    torch.testing.assert_close(scored_derivative,
                               (high['gated_prob'] - low['gated_prob']) / (2*epsilon),
                               rtol=1e-7, atol=1e-10)
    if phases == 2:
        # x=.5; q1'=q1(1-q1)*(-x)*(1/[x(1-x)]-6)=.25.
        expected = .125 * torch.sigmoid(torch.tensor(.5, dtype=torch.float64)) * (
            1-torch.sigmoid(torch.tensor(.5, dtype=torch.float64)))
        torch.testing.assert_close(scored_derivative.squeeze(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('off_diagonal', [1., 2., 3.])
def test_two_phase_fixed_jump_matches_the_closed_form_derivative(off_diagonal):
    model = constant_jump_model(off_diagonal=off_diagonal)
    time = torch.linspace(.01, 1.5, 13, dtype=torch.float64, requires_grad=True)
    result = model(torch.zeros((13, 8), dtype=torch.float64),
                   torch.zeros((13, 6), dtype=torch.float64), time,
                   torch.zeros(13, dtype=torch.float64))
    measured, = torch.autograd.grad(result['gated_prob'].sum(), time)
    x = .625 * torch.exp(-time.detach())
    q = torch.sigmoid(torch.log(x/(1-x)) + off_diagonal*(1-2*x))
    qprime = q*(1-q)*(-x)*(1/(x*(1-x))-2*off_diagonal)
    gate = torch.sigmoid(q)
    expected = .5*gate*(1-gate)*qprime
    torch.testing.assert_close(measured, expected, rtol=1e-12, atol=1e-12)
    if 2*off_diagonal <= 4:
        assert (measured <= 1e-14).all()
    else:
        assert (measured > 0).any()


def test_large_transition_norm_does_not_imply_monotonicity_failure():
    # a=-100, b=100: norm is large, but a+b=0 satisfies the exact two-phase condition.
    # Both signs are allowed by the implemented unconstrained off-diagonal matrix.
    model = constant_jump_model()
    with torch.no_grad():
        model.kernel.trans_net[-1].bias.copy_(
            torch.tensor([0., -100., 100., 0.], dtype=torch.float64))
    time = torch.linspace(.01, 1.5, 13, dtype=torch.float64, requires_grad=True)
    result = model(torch.zeros((13, 8), dtype=torch.float64),
                   torch.zeros((13, 6), dtype=torch.float64), time,
                   torch.zeros(13, dtype=torch.float64))
    derivative, = torch.autograd.grad(result['gated_prob'].sum(), time)
    assert torch.linalg.matrix_norm(result['T_matrix'][0]).item() > 100
    assert (derivative < 0).all()
