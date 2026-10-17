"""Posthoc paired-cohort metrics and exact fixed-reset branch decomposition."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score


def cohort_masks(gap,dose,gap_edges=(3600,86400,604800),dose_edges=(0,1,10)):
    gap=np.asarray(gap);dose=np.asarray(dose)
    if gap.ndim!=1 or dose.shape!=gap.shape or not np.isfinite(gap).all() or not np.isfinite(dose).all() or (gap<gap_edges[0]).any() or (dose<0).any():raise ValueError('Finite supported positive-query gaps and doses required')
    g=[(gap>=a)&(gap<b) for a,b in zip(gap_edges,(*gap_edges[1:],np.inf))]
    e=[(dose>=a)&(dose<b) for a,b in zip(dose_edges,(*dose_edges[1:],np.inf))]
    gn=('gap_1h_to_1d','gap_1d_to_7d','gap_ge_7d');en=('dose_zero','dose_1_to_9','dose_ge_10')
    masks={'all':np.ones(len(gap),bool),**dict(zip(gn,g)),**dict(zip(en,e))}
    masks.update({a+'__'+b:x&y for a,x in zip(gn,g) for b,y in zip(en,e)})
    assert np.all(np.sum(g,axis=0)==1) and np.all(np.sum(e,axis=0)==1)
    return masks


def paired_metrics(positive,negative,mask=None):
    p=np.asarray(positive);n=np.asarray(negative)
    if p.ndim!=1 or n.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Paired finite score vectors required')
    if mask is not None:p=p[mask];n=n[mask]
    if not len(p):return dict(ap=None,auc=None,pair_accuracy=None,reason='empty paired cohort')
    labels=np.r_[np.ones(len(p)),np.zeros(len(n))];scores=np.r_[p,n]
    return dict(ap=float(average_precision_score(labels,scores)),auc=float(roc_auc_score(labels,scores)),pair_accuracy=float(np.mean((p>n)+.5*(p==n))))


def phase_terms(elapsed,exposure,clock,event,gaps,amplitude,time_unit,exposure_unit):
    t,e,d,h,g=map(lambda x:np.asarray(x,dtype=np.float64),(elapsed,exposure,clock,event,gaps))
    if t.shape!=e.shape or d.shape!=h.shape or d.shape!=g.shape or d.ndim!=1 or not len(d):raise ValueError('Compatible time/dose and component vectors required')
    if not all(np.isfinite(x).all() and (x>=0).all() for x in (t,e,d,h,g)) or not np.isfinite([amplitude,time_unit,exposure_unit]).all() or amplitude<0 or min(time_unit,exposure_unit)<=0:raise ValueError('Finite nonnegative branch parameters and positive units required')
    coefficients=amplitude*g/(len(d)+1);hazard=t[...,None]*d/time_unit+e[...,None]*h/exposure_unit
    if not np.isfinite(hazard).all():raise ValueError('Hazard overflow')
    return coefficients*np.expm1(-hazard),float(coefficients.sum())
