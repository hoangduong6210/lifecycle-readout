"""Exact fixed-rate feasible geometry in inverse-square-root tilt coordinates.

The halfspace geometry is model-specific. Euclidean projection and cyclic dual
coordinate updates are standard convex-QP tools, not a new generic optimizer.
No Frobenius-distance, prediction-error or ranking-optimality claim is implied.
"""
import numpy as np


def inverse_tilt_constraints(gaps,generators):
    g=np.asarray(gaps,dtype=np.float64);rates=np.asarray(generators,dtype=np.float64)
    if rates.ndim==1:rates=rates[None,:]
    if g.ndim!=1 or not len(g) or rates.ndim!=2 or rates.shape[1]!=len(g) or not len(rates):raise ValueError('Gaps [m] and rate generators [channels,m] required')
    if not np.isfinite(g).all() or not np.isfinite(rates).all() or (g<0).any() or (rates<0).any():raise ValueError('Finite nonnegative gaps/rates required')
    rows=[];bounds=[];identities=[]
    for channel,d in enumerate(rates):
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                cross=(g[i]-g[j])*(d[i]-d[j])
                if not np.isfinite(cross):raise ValueError('Pair arithmetic overflow')
                if cross>=0:continue
                ai,aj=np.sqrt(g[j]*d[i]),np.sqrt(g[i]*d[j]);norm=np.hypot(ai,aj);bound=np.sqrt(-cross)
                if not np.isfinite([ai,aj,norm,bound]).all() or norm<=0:raise ValueError('Unsupported constraint scale')
                row=np.zeros(len(g));row[i]=ai/norm;row[j]=aj/norm;rows.append(row);bounds.append(bound/norm);identities.append(dict(channel=channel,i=i+1,j=j+1,normalization=float(norm)))
    return np.asarray(rows,dtype=np.float64).reshape(-1,len(g)),np.asarray(bounds,dtype=np.float64),identities


def _initial(raw,gaps):
    x=np.asarray(raw,dtype=np.float64);g=np.asarray(gaps,dtype=np.float64);vector=x.ndim==1
    if vector:x=x[None,:]
    if x.ndim!=2 or x.shape[1:]!=g.shape or not len(x) or not np.isfinite(x).all() or (x<=0).any():raise ValueError('Finite strictly positive inverse tilts [...,m] required')
    return x.copy(),vector


def all_rate_inverse_projection(raw,gaps):
    x,vector=_initial(raw,gaps);g=np.asarray(gaps,dtype=np.float64)
    if not np.isfinite(g).all() or (g<0).any():raise ValueError('Finite nonnegative gaps required')
    floor=np.sqrt((g.max()-g)/g.max()) if g.max()>0 else np.zeros_like(g)
    result=np.maximum(x,floor);return result[0] if vector else result


def project_inverse_tilts(raw,gaps,generators,*,tolerance=1e-10,max_cycles=10000):
    x0,vector=_initial(raw,gaps);A,b,identities=inverse_tilt_constraints(gaps,generators)
    if not np.isfinite(tolerance) or tolerance<=0 or not isinstance(max_cycles,int) or isinstance(max_cycles,bool) or max_cycles<1:raise ValueError('Positive tolerance and cycle budget required')
    dual=np.zeros((len(x0),len(b)));x=x0.copy();cycle=0
    if len(b):
        for cycle in range(max_cycles+1):
            slack=x@A.T-b;primal=float(np.maximum(-slack,0).max());complementarity=float(np.abs(dual*slack).max())
            if max(primal,complementarity)<=tolerance:break
            if cycle==max_cycles:raise RuntimeError('Projection KKT tolerance not reached within fixed cycle budget')
            for j,row in enumerate(A):
                previous=dual[:,j].copy();dual[:,j]=np.maximum(0,previous+b[j]-x@row);x+=(dual[:,j]-previous)[:,None]*row
            # Enforce the dual primal relationship without accumulated update drift.
            x=x0+dual@A
        stationarity=float(np.abs(x-x0-dual@A).max())
    else:primal=complementarity=stationarity=0.
    if not np.isfinite(x).all() or (x<x0).any() or min(primal,complementarity,stationarity)<0 or max(primal,complementarity,stationarity)>tolerance*2:raise RuntimeError('Projection certificate failed')
    squared=np.square(x-x0).sum(-1)
    if not np.isfinite(squared).all():raise ValueError('Projection distance overflow')
    return dict(inverse_tilt=x[0] if vector else x,dual=dual[0] if vector else dual,A=A,b=b,constraint_identities=identities,cycles=cycle,squared_distance=float(squared[0]) if vector else squared,kkt=dict(primal_violation=primal,complementarity=complementarity,stationarity=stationarity),scope='Unique Euclidean projection in inverse-square-root tilt coordinates, within declared floating KKT tolerance')
