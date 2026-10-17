"""Exploratory readout-specific certificate for a restricted entropic transform.

For q_i=c_i*z_i/sum(c*z), a fixed nonnegative-gap readout is nonincreasing
under every positive absorbing decay rate vector iff
    (delta_max-delta_j)*c_j <= delta_max*c_0, for every active j.
This module parameterizes a sufficient interior of that exact feasible set.
The result is restricted to constant tilts along each silence path. It is not
a certificate for arbitrary context-dependent transition matrices or a claim
of priority over normalized reweighting/linear-fractional analysis.
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ReadoutAwareTilt(nn.Module):
    """Allow active-phase redistribution while keeping a chosen readout monotone.

    Cumulative positive readout gaps make the final active phase maximal.
    Its log tilt is unconstrained; lower-readout phases obey exact upper bounds.
    Model weights, initial state, rates and optional external affinity must be
    fixed along a silence trajectory. No existing fitted model is replaced.
    """
    def __init__(self, n_phases: int):
        super().__init__()
        if not isinstance(n_phases,int) or isinstance(n_phases,bool) or n_phases<2:
            raise ValueError("At least two phases required")
        self.n_phases=n_phases
        self.raw_gap=nn.Parameter(torch.zeros(n_phases-1))
        self.raw_tilt=nn.Parameter(torch.zeros(n_phases-1))
        self.dormant_logit=nn.Parameter(torch.zeros(()))

    def readout_and_tilt(self):
        increments=F.softplus(self.raw_gap)
        gaps=increments.cumsum(-1)
        if self.n_phases>2:
            # Suffix sums avoid cancellation in delta_max-delta_j.
            remaining=increments.flip(0).cumsum(0).flip(0)[1:]
            cap=gaps[-1].log()-remaining.log()
            active_tilt=torch.cat((cap-F.softplus(self.raw_tilt[:-1]),self.raw_tilt[-1:]))
        else:
            active_tilt=self.raw_tilt
        tilt=torch.cat((active_tilt.new_zeros(1),active_tilt))
        if not torch.isfinite(tilt).all() or not (increments>0).all():
            raise ValueError("Parameters exceed the finite positive certificate domain")
        return gaps,tilt

    def forward(self, initial: Tensor, elapsed: Tensor, rates: Tensor):
        if initial.ndim!=2 or initial.shape[-1]!=self.n_phases or elapsed.shape!=(len(initial),) or rates.shape!=(self.n_phases-1,):
            raise ValueError("State [batch, phases], elapsed [batch], rates [phases-1] required")
        if not initial.is_floating_point() or any(x.dtype!=initial.dtype or x.device!=initial.device for x in (elapsed,rates,self.raw_gap)):
            raise ValueError("Inputs and model must share floating dtype/device")
        if not all(torch.isfinite(x).all() for x in (initial,elapsed,rates)) or (initial<0).any() or (elapsed<0).any() or (rates<=0).any():
            raise ValueError("Finite nonnegative state/time and positive rates required")
        if not torch.allclose(initial.sum(-1),torch.ones_like(elapsed),rtol=0,atol=16*torch.finfo(initial.dtype).eps):
            raise ValueError("Initial rows must lie on the simplex")
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("Finite parameters required")
        gaps,tilt=self.readout_and_tilt()
        argument=-elapsed[:,None]*rates
        active=initial[:,1:]*torch.exp(argument)
        dormant=initial[:,:1]+(initial[:,1:]*(-torch.expm1(argument))).sum(-1,keepdim=True)
        flow=torch.cat((dormant,active),-1)
        # Handle exact boundary zeros without a positive artificial log floor.
        safe=torch.where(flow>0,flow,torch.ones_like(flow))
        log_flow=torch.where(flow>0,safe.log(),torch.full_like(flow,-torch.inf))
        state=torch.softmax(log_flow+tilt,-1)
        logit=self.dormant_logit+(state[:,1:]*gaps).sum(-1)
        return dict(state=state,flow=flow,logit=logit,prob=torch.sigmoid(logit),gaps=gaps,log_tilt=tilt)
