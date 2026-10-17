"""Exact absorbing lifecycle steps under clock and competitor-event exposure.

An exponential event jump is a specified update, not the exact solution of
 dz=-gamma*kappa*z*dN (whose jump factor is 1-gamma*kappa).
This operator has no event extractor, training loop, or empirical efficacy claim.
"""
import torch
from torch import Tensor


def competitive_step(state: Tensor, elapsed: Tensor, exposure: Tensor,
                     clock_rates: Tensor, event_rates: Tensor) -> Tensor:
    """Move active mass to dormant under nonnegative additive hazard doses.

    state: [batch, K]; elapsed, exposure: [batch]; rates: [K-1].
    Exposure is a sum of nonnegative event-time kernels for competing events;
    the caller must exclude the target edge and enforce strict-past queries.
    A single event is elapsed=0, exposure=kappa; physical flow is exposure=0.
    Rates and kernels are held fixed for each supplied step. All scalar-kernel
    diagonal jumps commute; this operator does not encode competitor order.
    """
    if state.ndim != 2 or state.shape[-1] < 2 or not state.is_floating_point():
        raise ValueError("Floating state [batch, phases>=2] required")
    batch, phases = state.shape
    if elapsed.shape != (batch,) or exposure.shape != (batch,) or any(
            v.shape != (phases-1,) for v in (clock_rates, event_rates)):
        raise ValueError("Time/exposure [batch] and rates [phases-1] required")
    values = (state, elapsed, exposure, clock_rates, event_rates)
    if any(v.dtype != state.dtype or v.device != state.device for v in values):
        raise ValueError("Inputs must share floating dtype and device")
    if any(not torch.isfinite(v).all() or (v < 0).any() for v in values):
        raise ValueError("Finite nonnegative inputs required")
    if not torch.allclose(state.sum(-1), torch.ones_like(elapsed), rtol=0,
                          atol=16*torch.finfo(state.dtype).eps):
        raise ValueError("State must lie on the probability simplex")
    dose = elapsed[:, None]*clock_rates + exposure[:, None]*event_rates
    # Reject overflow rather than silently saturating derivatives at infinity.
    if not torch.isfinite(dose).all():
        raise ValueError("Accumulated hazard overflow")
    active = state[:, 1:]*torch.exp(-dose)
    lost = (state[:, 1:]*(-torch.expm1(-dose))).sum(-1, keepdim=True)
    return torch.cat((state[:, :1]+lost, active), dim=-1)
