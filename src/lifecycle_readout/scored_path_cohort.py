"""Batched conditional lifecycle paths; no model fitting or sampling decisions."""
import torch


def path_state(lifecycle, initial, context, fixed, time, path):
    if path == 'ideal_flow':
        active=initial[:,1:]*torch.exp(-time[:,None]*lifecycle.kernel.decay_rates)
        return torch.cat((1-active.sum(-1,keepdim=True),active),-1)
    if path == 'implemented_flow':
        return lifecycle.kernel.continuous_decay(initial,time)
    if path == 'fixed_transition':
        flow=lifecycle.kernel.continuous_decay(initial,time)
        return torch.softmax(flow.clamp(min=1e-6).log()+torch.einsum('bi,bij->bj',flow,fixed),-1)
    if path == 'dynamic_transition':
        return lifecycle.kernel(initial,context,time)[0]
    raise ValueError('Unknown conditional path')


def evaluate_paths(lifecycle, initial, context, base, fixed, grid, *, paths,
                   derivative_support=None):
    """Use native dtype scores, with optional autodiff/certificate cross-check.

    The caller freezes original encoder/base outputs before dtype conversion.
    A microbatch bounds memory regardless of full validation cohort size.
    """
    count=len(initial);steps=len(grid)
    initial=initial.repeat_interleave(steps,0);context=context.repeat_interleave(steps,0)
    base=base.repeat_interleave(steps,0);fixed=fixed.repeat_interleave(steps,0)
    outputs={}
    for path in paths:
        time=grid.repeat(count).requires_grad_(derivative_support is not None)
        state=path_state(lifecycle,initial,context,fixed,time,path)
        gate=lifecycle.survival_gate(state);score=torch.sigmoid(base)*gate
        record={'scores':score.detach().reshape(count,steps).cpu().numpy()}
        if derivative_support is not None:
            derivative,=torch.autograd.grad(score.sum(),time,retain_graph=True)
            record['derivatives']=derivative.detach().reshape(count,steps).cpu().numpy()
            record['certificate_max_error']=None
            if path in ('fixed_transition','dynamic_transition'):
                def state_and_transition(tau):
                    flow=lifecycle.kernel.continuous_decay(initial,tau)
                    matrix=(fixed if path=='fixed_transition' else lifecycle.kernel(initial,context,tau)[1])
                    return flow,matrix
                (flow,matrix),(velocity,drift)=torch.func.jvp(state_and_transition,(time,),(torch.ones_like(time),))
                valid=~(flow==1e-6).any(-1)
                record['certificate_boundary_exclusions']=int((~valid).sum())
                q,qdot,_=derivative_support(flow[valid],velocity[valid],matrix[valid],drift[valid])
                weights=lifecycle.survival_gate.survival_weights
                certificate=torch.sigmoid(base[valid])*gate[valid]*(1-gate[valid])*(qdot*weights).sum(-1)
                torch.testing.assert_close(q,state[valid],rtol=1e-11,atol=1e-12)
                torch.testing.assert_close(certificate,derivative[valid],rtol=1e-8,atol=1e-10)
                record['certificate_max_error']=float((certificate-derivative[valid]).detach().abs().max())
        outputs[path]=record
    return outputs
