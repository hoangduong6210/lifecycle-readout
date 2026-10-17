"""Check mathematical contracts before any real-data optimization."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.modeling.monotone_residual import NestedMonotoneResidual


def test_zero_amplitude_exactly_recovers_base_and_learns_at_boundary():
    torch.manual_seed(9)
    model = NestedMonotoneResidual(3)
    base = torch.tensor([-100., -.4, 0., .7, 100.])
    context = torch.randn(5, 3)
    elapsed = torch.tensor([0., 1e-10, 1., 100., 1e10])
    out = model(base, context, elapsed)
    assert torch.equal(out['logits'], base)
    assert torch.equal(out['prob'], base.sigmoid())
    # Negative labels permit learning attenuation from the exact neutral point.
    torch.nn.functional.binary_cross_entropy_with_logits(out['logits'], torch.zeros(5)).backward()
    assert (model.amplitude.grad < 0).all()
    # Encoder/rate gradients are expected to be zero until amplitudes move.
    assert torch.equal(model.raw_rate.grad, torch.zeros(4))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.step(); model.project_parameters_()
    assert (model.amplitude > 0).all()
    assert (model(base, context, elapsed)['residual'][1:] < 0).all()


def test_fixed_context_silence_derivative_and_extreme_time_limits():
    torch.manual_seed(3)
    model = NestedMonotoneResidual(3).double()
    with torch.no_grad():
        model.amplitude.copy_(torch.tensor([.2, 2., .7, 1.3]))
        model.raw_rate.copy_(torch.tensor([-2., 0., .8, 2.]))
    context = torch.randn(8, 3, dtype=torch.float64)
    base = torch.randn(8, dtype=torch.float64)
    elapsed = torch.linspace(.01, 8., 8, dtype=torch.float64, requires_grad=True)
    out = model(base, context, elapsed)
    derivative, = torch.autograd.grad(out['logits'].sum(), elapsed)
    expected = -(out['weights'] * model.amplitude * out['rates']
                 * torch.exp(-elapsed[:, None] * out['rates'])).sum(-1)
    torch.testing.assert_close(derivative, expected)
    assert (derivative < 0).all()
    zero = model(base, context, torch.zeros(8, dtype=torch.float64))
    assert torch.equal(zero['logits'], base)
    limit = model(base, context, torch.full((8,), 1e20, dtype=torch.float64))
    torch.testing.assert_close(limit['residual'], -(out['weights'] * model.amplitude).sum(-1))
    assert torch.isfinite(limit['logits']).all()


def test_projection_and_forward_enforce_nonnegative_amplitudes():
    model = NestedMonotoneResidual(2)
    with torch.no_grad():
        model.amplitude.copy_(torch.tensor([-2., .4, -1., 3.]))
    base = torch.randn(3); context = torch.randn(3, 2); times = torch.arange(3.)
    before = model(base, context, times)['logits']
    model.project_parameters_()
    assert (model.amplitude >= 0).all()
    torch.testing.assert_close(before, model(base, context, times)['logits'])


@pytest.mark.parametrize('time', [-1., float('nan'), float('inf')])
def test_invalid_elapsed_is_rejected(time):
    model = NestedMonotoneResidual(2)
    with pytest.raises(ValueError, match='finite'):
        model(torch.zeros(1), torch.zeros(1, 2), torch.tensor([time]))


def test_silence_claim_does_not_cover_changing_context():
    # Context weights can switch to a weaker component; the scope matters.
    model = NestedMonotoneResidual(1, n_components=2, hidden_dim=1).double()
    with torch.no_grad():
        model.encoder[0].weight.fill_(1.); model.encoder[0].bias.zero_()
        model.encoder[2].weight.copy_(torch.tensor([[-20.], [20.]]))
        model.encoder[2].bias.copy_(torch.tensor([10., -10.]))
        model.amplitude.copy_(torch.tensor([10., .01]))
    early = model(torch.zeros(1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    late = model(torch.zeros(1, dtype=torch.float64), torch.full((1, 1), 2., dtype=torch.float64), torch.full((1,), 2., dtype=torch.float64))
    assert late['prob'].item() > early['prob'].item()
