"""Exploratory simplex readout transform with a fixed-context silence guarantee.

This is not the event-update operator and is not installed in existing studies.
Inputs, mixing parameters and gate parameters must stay fixed along a silence
trajectory. Absorbing stochastic mixing is established mathematics, not a
novelty claim. The optional nonlinear throttle trades expressivity for a proof.
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MonotoneSimplexTransition(nn.Module):
    """Mix decaying active mass, then optionally attenuate it monotonically.

    x_k(t)=z_k(0)*exp(-d_k*t); A is row-substochastic, h(S)=sigmoid(a+b*S),
    b>=0, S=sum(x). Output active mass is h(S)*(x@A), with all remaining
    mass assigned to dormant. Set nonlinear=False for h=1. Any nonnegative
    active-to-dormant readout gaps give a nonincreasing gate logit.
    Rates/times are supplied explicitly; no hidden time units or context encoder.
    """

    def __init__(self, n_phases: int, nonlinear: bool = True):
        super().__init__()
        if isinstance(n_phases, bool) or not isinstance(n_phases, int) or n_phases < 2:
            raise ValueError("n_phases must be an integer at least two")
        self.n_phases = n_phases
        self.nonlinear = nonlinear
        # Column zero is absorption; other columns redistribute active mass.
        self.transition_logits = nn.Parameter(torch.zeros(n_phases - 1, n_phases))
        if nonlinear:
            self.throttle_bias = nn.Parameter(torch.zeros(()))
            self.raw_throttle_slope = nn.Parameter(torch.zeros(()))

    def forward(self, initial: Tensor, elapsed: Tensor, rates: Tensor) -> dict[str, Tensor]:
        if initial.ndim != 2 or initial.shape[1] != self.n_phases:
            raise ValueError("initial must have shape [batch, n_phases]")
        if not initial.is_floating_point():
            raise ValueError("initial must be floating point")
        if elapsed.shape != (len(initial),) or rates.shape != (self.n_phases - 1,):
            raise ValueError("elapsed must be [batch] and rates [n_phases-1]")
        if any(x.device != initial.device or x.dtype != initial.dtype for x in
               (elapsed, rates, self.transition_logits)):
            raise ValueError("inputs and module must share floating dtype and device")
        if not all(torch.isfinite(x).all() for x in (initial, elapsed, rates)):
            raise ValueError("inputs must be finite")
        if (initial < 0).any() or (elapsed < 0).any() or (rates <= 0).any():
            raise ValueError("initial/elapsed must be nonnegative and rates positive")
        tolerance = 16 * torch.finfo(initial.dtype).eps
        if not torch.allclose(initial.sum(-1), torch.ones_like(elapsed), rtol=0, atol=tolerance):
            raise ValueError("initial rows must lie on the simplex")
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("module parameters must be finite")
        active = initial[:, 1:] * torch.exp(-elapsed[:, None] * rates)
        mixing = torch.softmax(self.transition_logits, dim=-1)
        total = active.sum(-1, keepdim=True)
        throttle = (torch.sigmoid(self.throttle_bias +
                                 F.softplus(self.raw_throttle_slope) * total)
                    if self.nonlinear else torch.ones_like(total))
        transformed = throttle * (active @ mixing[:, 1:])
        state = torch.cat((1 - transformed.sum(-1, keepdim=True), transformed), dim=-1)
        return {"state": state, "active_before": active,
                "mixing": mixing, "throttle": throttle}
