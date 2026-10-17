"""Development-only nested logit correction; no empirical efficacy claim.

The context must remain fixed when interpreting the silence derivative. This
module is a conditional mixture of decay bases, not an identified lifecycle.
"""
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NestedMonotoneResidual(nn.Module):
    """Subtract a bounded nonnegative correction from a supplied base logit.

    r(c,t) = -sum_k a_k q_k(c) (1-exp(-d_k t)), a_k >= 0, d_k > 0.
    Zero amplitudes exactly recover the base. Amplitudes use projected updates
    rather than squared parameters, whose gradients would vanish at zero.
    Call project_parameters_ immediately after each optimizer step. Forward
    also clamps amplitudes, so monotonicity survives an unprojected update.
    Times must be finite nonnegative dimensionless values. No probability
    clamping or multiplication is used.
    """

    def __init__(self, context_dim: int, n_components: int = 4,
                 hidden_dim: int = 32, rate_init: float = 1.0):
        super().__init__()
        if min(context_dim, n_components, hidden_dim) < 1:
            raise ValueError("Dimensions must be positive")
        if not math.isfinite(rate_init) or rate_init <= 0:
            raise ValueError("Initial rate must be finite and positive")
        self.context_dim = context_dim
        self.encoder = (nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, n_components))
                        if n_components > 1 else None)
        self.amplitude = nn.Parameter(torch.zeros(n_components))
        inverse_softplus = rate_init + math.log(-math.expm1(-rate_init))
        self.raw_rate = nn.Parameter(torch.full((n_components,), inverse_softplus))

    @torch.no_grad()
    def project_parameters_(self):
        self.amplitude.clamp_(min=0)
        return self

    def forward(self, base_logits: Tensor, context: Tensor, elapsed: Tensor):
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("Context must have shape [batch, context_dim]")
        if base_logits.shape != (len(context),) or elapsed.shape != base_logits.shape:
            raise ValueError("Logits and elapsed must both have shape [batch]")
        if not (torch.isfinite(context).all() and torch.isfinite(base_logits).all()
                and torch.isfinite(elapsed).all() and (elapsed >= 0).all()):
            raise ValueError("Inputs must be finite and elapsed nonnegative")
        weights = (F.softmax(self.encoder(context), dim=-1) if self.encoder is not None
                   else context.new_ones((len(context), 1)))
        rates = F.softplus(self.raw_rate)
        removed = -torch.expm1(-elapsed[:, None] * rates[None, :])
        correction = -(weights * self.amplitude.clamp(min=0) * removed).sum(-1)
        logits = base_logits + correction
        return {"logits": logits, "prob": torch.sigmoid(logits),
                "residual": correction, "weights": weights, "rates": rates}
