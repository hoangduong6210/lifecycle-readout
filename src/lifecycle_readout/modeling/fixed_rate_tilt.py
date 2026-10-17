"""Fixed-rate constant-tilt readout checks and a rate-ordered prototype.

The exact real-arithmetic condition is pairwise; floating margins near zero are
reported as boundary cases. It does not certify a changing context/tilt/readout.
The LP support argument and binary quadratic test are established techniques;
no priority or predictor efficacy claim is made by this implementation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..competitive_lifecycle import competitive_step


def fixed_rate_check(tilt,gaps,rates,*,relative_tolerance=1e-12):
    c,g,d=map(lambda x:np.asarray(x,dtype=np.float64),(tilt,gaps,rates))
    if g.ndim!=1 or d.shape!=g.shape or c.shape!=(len(g)+1,) or not len(g):raise ValueError('Positive tilt [m+1], nonnegative gaps/rates [m] required')
    if not all(np.isfinite(x).all() for x in (c,g,d)) or (c<=0).any() or (g<0).any() or (d<0).any() or not np.isfinite(relative_tolerance) or relative_tolerance<0:raise ValueError('Finite positive tilt, nonnegative gaps/rates/tolerance required')
    ratio=c[0]/c[1:];diagonal=g*d*ratio;pairs=[];worst=None
    for i in range(len(g)):
        for j in range(i+1,len(g)):
            cross=(g[i]-g[j])*(d[i]-d[j])+g[i]*d[j]*ratio[j]+g[j]*d[i]*ratio[i]
            root=2*np.sqrt(diagonal[i])*np.sqrt(diagonal[j]);margin=cross+root;scale=max(1.,abs(cross),abs(root));tol=relative_tolerance*scale
            if not np.isfinite([cross,root,margin]).all():raise ValueError('Certificate arithmetic overflow')
            row=dict(i=i+1,j=j+1,diagonal_i=float(diagonal[i]),diagonal_j=float(diagonal[j]),cross=float(cross),margin=float(margin),tolerance=float(tol),status='FAIL' if margin< -tol else 'BOUNDARY' if abs(margin)<=tol else 'PASS');pairs.append(row)
            if row['status']=='FAIL' and (worst is None or margin<worst['margin']):worst=row
    if not np.isfinite(diagonal).all():raise ValueError('Certificate arithmetic overflow')
    witness=None
    if worst is not None:
        i,j=worst['i']-1,worst['j']-1;ai,aj,b=diagonal[i],diagonal[j],worst['cross']
        if ai>0 and aj>0:u=np.sqrt(aj)/(np.sqrt(ai)+np.sqrt(aj))
        elif ai==0 and aj>0:u=1+b/(2*(aj-b))
        elif aj==0 and ai>0:u=-b/(2*(ai-b))
        else:u=.5
        q=np.zeros(len(c));q[i+1]=u;q[j+1]=1-u;z=q/c;z/=z.sum();e=d*(1-ratio);derivative=-(g*d)@q[1:]+(g@q[1:])*(e@q[1:]);witness=dict(q=q.tolist(),initial=z.tolist(),readout_derivative=float(derivative))
    return dict(status='FAIL' if worst is not None else 'BOUNDARY' if any(x['status']=='BOUNDARY' for x in pairs) else 'PASS',pairs=pairs,diagonal=diagonal.tolist(),witness=witness,scope='Real-arithmetic theorem evaluated with floating margins; boundary is not a machine-arithmetic certificate')


def rate_cone_check(tilt,gaps,generators,*,relative_tolerance=1e-12):
    vectors=np.asarray(generators,dtype=np.float64)
    if vectors.ndim!=2 or not len(vectors):raise ValueError('Nonempty rate generators [channels,m] required')
    checks=[fixed_rate_check(tilt,gaps,row,relative_tolerance=relative_tolerance) for row in vectors]
    return dict(status='FAIL' if any(x['status']=='FAIL' for x in checks) else 'BOUNDARY' if any(x['status']=='BOUNDARY' for x in checks) else 'PASS',generators=checks)


class OrderedRateTilt(nn.Module):
    """Untrained constant-tilt gate with co-ordered gaps and both rate vectors.

    Arbitrary fixed active log tilts are permitted. All inputs/context and model
    weights except elapsed/exposure must stay fixed along the certified path.
    This does not certify an external affinity or identify semantic phase stages.
    """
    def __init__(self,n_phases):
        super().__init__()
        if not isinstance(n_phases,int) or isinstance(n_phases,bool) or n_phases<2:raise ValueError('At least two phases required')
        self.n_phases=n_phases
        self.raw_gap=nn.Parameter(torch.zeros(n_phases-1));self.raw_clock=nn.Parameter(torch.zeros(n_phases-1));self.raw_event=nn.Parameter(torch.zeros(n_phases-1));self.active_log_tilt=nn.Parameter(torch.zeros(n_phases-1));self.dormant_logit=nn.Parameter(torch.zeros(()))

    def forward(self,initial,elapsed,exposure):
        if initial.ndim!=2 or initial.shape[1]!=self.n_phases:raise ValueError('Initial state [batch,phases] required')
        if not initial.is_floating_point() or any(x.dtype!=initial.dtype or x.device!=initial.device for x in (elapsed,exposure,self.raw_gap)):raise ValueError('Inputs and parameters must share floating dtype/device')
        if not all(torch.isfinite(p).all() for p in self.parameters()):raise ValueError('Finite parameters required')
        gaps=F.softplus(self.raw_gap).cumsum(0);clock=F.softplus(self.raw_clock).cumsum(0);event=F.softplus(self.raw_event).cumsum(0)
        if not all((x>0).all() and torch.isfinite(x).all() for x in (gaps,clock,event)):raise ValueError('Positive finite gaps/rates required')
        flow=competitive_step(initial,elapsed,exposure,clock,event);log_tilt=torch.cat((self.active_log_tilt.new_zeros(1),self.active_log_tilt))
        # Direct positive normalization retains derivatives as zero mass enters
        # a boundary coordinate. Masking log(0) would silently remove that path.
        weights=torch.exp(log_tilt-log_tilt.max())
        if not (weights>0).all():raise ValueError('Tilt scale underflows the supported native precision')
        weighted=flow*weights;denominator=weighted.sum(-1,keepdim=True)
        if not torch.isfinite(denominator).all() or not (denominator>0).all():raise ValueError('Positive finite normalizer required')
        state=weighted/denominator;logit=self.dormant_logit+(state[:,1:]*gaps).sum(-1)
        return dict(state=state,flow=flow,logit=logit,prob=torch.sigmoid(logit),gaps=gaps,clock_rates=clock,event_rates=event,log_tilt=log_tilt)
