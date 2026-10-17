"""Pointwise directional certificate for a softmax candidate transform.

This is the ordinary softmax chain rule, not a new optimization theorem.
The supplied velocities must describe the actual path being certified. Freezing
context or omitting its derivative changes the path and therefore the claim.
"""
import torch
from torch import Tensor


def entropic_direction(z: Tensor, velocity: Tensor, transition: Tensor,
                       transition_velocity: Tensor, *, log_floor: float = 1e-6):
    """Return transformed state, directional derivative and logit velocity.

    The clamp derivative is zero below the floor, matching torch.clamp's
    off-boundary derivative. Equality at the floor is excluded as nonsmooth.
    Shapes are [batch, phases] and [batch, phases, phases]. No initial-state,
    base-score or readout drift is implicitly supplied by this routine.
    """
    if z.ndim != 2 or velocity.shape != z.shape:
        raise ValueError("State and velocity must have matching matrix shapes")
    shape = (z.shape[0], z.shape[1], z.shape[1])
    if transition.shape != shape or transition_velocity.shape != shape:
        raise ValueError("Transition and velocity must be [batch, phases, phases]")
    if not log_floor > 0:
        raise ValueError("Positive log floor required")
    if not all(torch.isfinite(x).all() for x in (z, velocity, transition, transition_velocity)) or (z < 0).any():
        raise ValueError("Finite inputs and nonnegative state required")
    if (z == log_floor).any():
        raise ValueError("Derivative at log-clamp boundary is nonsmooth")
    logits = z.clamp(min=log_floor).log()+torch.einsum("bi,bij->bj", z, transition)
    h = torch.where(z > log_floor, velocity/z.clamp(min=log_floor), torch.zeros_like(z))
    h = h+torch.einsum("bi,bij->bj", velocity, transition)+torch.einsum("bi,bij->bj", z, transition_velocity)
    q = torch.softmax(logits, dim=-1)
    derivative = q*(h-(q*h).sum(-1, keepdim=True))
    return q, derivative, h


def linear_readout_direction(q: Tensor, derivative: Tensor, weights: Tensor):
    """Derivative of fixed w^T q; its sign is Cov_q(w, logit_velocity)."""
    if derivative.shape != q.shape or weights.shape != (q.shape[1],):
        raise ValueError("Readout shapes mismatch")
    return (derivative*weights).sum(-1)
