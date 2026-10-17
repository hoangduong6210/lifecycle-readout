"""No-fit projection onto a restricted readout-preserving transition family.

Projection uses classical least squares/KKT algebra, not a novelty assertion.
The input transition is evaluated at a fixed reference clock by the caller.
All state/context/affinity assumptions of the constant-tilt theorem still apply.
"""
import torch


def gradient_potential(matrix):
    """Frobenius-nearest T_ij=b_j-b_i; choose the gauge b_0=0."""
    if matrix.ndim!=3 or matrix.shape[-1]!=matrix.shape[-2]:
        raise ValueError('Batched square transition required')
    k=matrix.shape[-1]
    b=(matrix.sum(-2)-matrix.sum(-1))/(2*k)
    return b-b[:,:1]


def project_potential(potential,caps):
    """Nearest gradient matrix with b_0=0 and active b_j<=caps_j.

    In active coordinates the metric is K*I-11^T. KKT gives
    b_j=min(bhat_j+alpha,cap_j), alpha=sum(b-bhat)/K.
    The active set only shrinks from its alpha=0 initialization; at most m+1
    passes recover the exact real-arithmetic solution. Infinite caps are free.
    """
    if potential.ndim!=2 or caps.shape!=(potential.shape[-1]-1,):
        raise ValueError('Potential/cap dimensions mismatch')
    if not torch.isfinite(potential).all() or torch.isnan(caps).any() or torch.isneginf(caps).any():
        raise ValueError('Invalid potential or cap')
    u=caps[None,:]-potential[:,1:]
    active=u<0
    for _ in range(potential.shape[-1]):
        alpha=torch.where(active,u,torch.zeros_like(u)).sum(-1,keepdim=True)/(1+active.sum(-1,keepdim=True))
        active=u<alpha
    repaired=torch.minimum(potential[:,1:]+alpha,caps)
    return torch.cat((torch.zeros_like(potential[:,:1]),repaired),-1)


def readout_caps(weights):
    """Exact fixed-readout constant-tilt bounds; maximal gaps are unbounded."""
    gaps=weights[1:]-weights[0]
    if weights.ndim!=1 or (gaps<0).any() or not torch.isfinite(weights).all():
        raise ValueError('Finite weights with nonnegative active gaps required')
    maximum=gaps.max()
    if maximum==0:return torch.full_like(gaps,torch.inf)
    remainder=maximum-gaps
    safe=torch.where(remainder>0,remainder,torch.ones_like(remainder))
    return torch.where(remainder>0,maximum.log()-safe.log(),torch.full_like(gaps,torch.inf))


def safe_flow(initial,rates,time):
    argument=-time[:,None]*rates
    active=initial[:,1:]*torch.exp(argument)
    dormant=initial[:,:1]+(initial[:,1:]*(-torch.expm1(argument))).sum(-1,keepdim=True)
    return torch.cat((dormant,active),-1)


def tilt_state(flow,potential):
    safe=torch.where(flow>0,flow,torch.ones_like(flow))
    logits=torch.where(flow>0,safe.log(),torch.full_like(flow,-torch.inf))+potential
    return torch.softmax(logits,-1)


def repair_score(lifecycle,initial,context,base,time,variant):
    """Return unclipped score and reference potential; no parameter updates."""
    if variant=='original_dynamic':
        q,_=lifecycle.kernel(initial,context,time)
    else:
        _,reference=lifecycle.kernel(initial,context,torch.zeros_like(time))
        if variant=='legacy_reference':
            flow=lifecycle.kernel.continuous_decay(initial,time)
            q=torch.softmax(flow.clamp(min=1e-6).log()+torch.einsum('bi,bij->bj',flow,reference),-1)
        else:
            flow=safe_flow(initial,lifecycle.kernel.decay_rates,time)
            if variant=='safe_reference':
                safe=torch.where(flow>0,flow,torch.ones_like(flow))
                logits=torch.where(flow>0,safe.log(),torch.full_like(flow,-torch.inf))
                q=torch.softmax(logits+torch.einsum('bi,bij->bj',flow,reference),-1)
            else:
                potential=gradient_potential(reference)
                if variant=='readout_projected':potential=project_potential(potential,readout_caps(lifecycle.survival_gate.survival_weights))
                elif variant=='coordinate_projected':potential=project_potential(potential,potential.new_zeros(potential.shape[-1]-1))
                elif variant=='zero_safe':potential=torch.zeros_like(potential)
                elif variant!='gradient_unconstrained':raise ValueError('Unknown repair variant')
                q=tilt_state(flow,potential)
    return torch.sigmoid(base)*lifecycle.survival_gate(q)
