"""Exact finite witnesses delimiting coefficient-shape sufficient predicates."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.modeling.fixed_rate_tilt import fixed_rate_check


def pair_margin(c,g,d,i,j):
    a=c[0]*g[i]*d[i]/c[i+1];b=c[0]*g[j]*d[j]/c[j+1]
    cross=(g[i]-g[j])*(d[i]-d[j])+c[0]*(g[i]*d[j]/c[j+1]+g[j]*d[i]/c[i+1])
    return sp.simplify(cross+2*sp.sqrt(a*b))


def coefficients(c,g,initial):
    return [sp.Integer(0)]+[gap*tilt*mass for gap,tilt,mass in zip(g,c[1:],initial[1:])], [c[0]]+[(tilt-c[0])*mass for tilt,mass in zip(c[1:],initial[1:])]


def derivative_at_zero(c,g,d,initial):
    numerator,denominator=coefficients(c,g,initial)
    rates=[sp.Integer(0)]+list(d)
    n=sum(numerator);den=sum(denominator)
    dn=-sum(a*rate for a,rate in zip(numerator,rates))
    dd=-sum(b*rate for b,rate in zip(denominator,rates))
    return sp.factor((dn*den-n*dd)/den**2)


def verify():
    t=sp.symbols('t',nonnegative=True)
    symbolic=[]
    for m in (1,2,3,4):
        c=sp.symbols('c0:'+str(m+1),positive=True)
        g=sp.symbols('g0:'+str(m),nonnegative=True)
        d=sp.symbols('d0:'+str(m),positive=True)
        mass=sp.symbols('p0:'+str(m),nonnegative=True)
        active=[p*sp.exp(-rate*t) for p,rate in zip(mass,d)]
        dormant=1-sum(active)
        den=c[0]*dormant+sum(ci*z for ci,z in zip(c[1:],active))
        spectral_den=c[0]+sum((ci-c[0])*p*sp.exp(-rate*t) for ci,p,rate in zip(c[1:],mass,d))
        numerator=sum(delta*ci*z for delta,ci,z in zip(g,c[1:],active))
        assert sp.expand(den-spectral_den)==0
        q=[ci*z/den for ci,z in zip(c[1:],active)]
        margin=sum(delta*rate*qi for delta,rate,qi in zip(g,d,q))-sum(delta*qi for delta,qi in zip(g,q))*sum(rate*(1-c[0]/ci)*qi for rate,ci,qi in zip(d,c[1:],q))
        assert sp.factor(sp.diff(numerator/den,t)+margin)==0
        symbolic.append(m)
    R=sp.Rational
    c=list(map(sp.Integer,[1,2]))+[R(11,10),sp.Integer(2),R(11,10)]
    g=list(map(sp.Integer,[1,2,3,4]));d=g.copy();initial=[R(1,5)]*5
    numerator,denominator=coefficients(c,g,initial)
    quotients=[sp.factor(a/b) for a,b in zip(numerator,denominator)]
    assert quotients==[0,2,22,6,44] and all(b>0 for b in denominator)
    signs=[sp.sign(b-a) for a,b in zip(quotients,quotients[1:])]
    assert signs==[1,1,-1,1]  # two changes; neither unimodal orientation
    margins=[pair_margin(c,g,d,i,j) for i in range(4) for j in range(i+1,4)]
    assert all(value>0 for value in margins)
    assert fixed_rate_check(list(map(float,c)),list(map(float,g)),list(map(float,d)))['status']=='PASS'
    nonunimodal=dict(tilts=list(map(str,c)),gaps=list(map(str,g)),rates=list(map(str,d)),
        initial=list(map(str,initial)),coefficient_quotients=list(map(str,quotients)),
        adjacent_signs=list(map(int,signs)),pair_margins=list(map(str,margins)),all_state_safe=True)
    # Equal quotient shape does not decide monotone decrease without rate/state checks.
    c=list(map(sp.Integer,[1,4,4]));g=list(map(sp.Integer,[2,1]));initial=[sp.Integer(0),R(1,2),R(1,2)]
    d_safe=list(map(sp.Integer,[1,2]));d_unsafe=list(map(sp.Integer,[1,10]))
    numerator,denominator=coefficients(c,g,initial)
    quotients=[sp.factor(a/b) for a,b in zip(numerator,denominator)]
    assert quotients==[0,R(8,3),R(4,3)]
    assert pair_margin(c,g,d_safe,0,1)==R(5,4)
    assert pair_margin(c,g,d_unsafe,0,1)==-R(15,4)+sp.sqrt(5)<0
    assert derivative_at_zero(c,g,d_safe,initial)==-R(5,16)
    assert derivative_at_zero(c,g,d_unsafe,initial)==R(3,16)
    safe_unsafe=dict(tilts=list(map(str,c)),gaps=list(map(str,g)),initial=list(map(str,initial)),
        same_ordered_coefficient_quotients=list(map(str,quotients)),safe_rates=[1,2],unsafe_rates=[1,10],
        safe_all_state_pair_margin='5/4',unsafe_pair_margin=str(-R(15,4)+sp.sqrt(5)),
        safe_initial_derivative='-5/16',unsafe_initial_derivative='3/16',
        scope='Ordering predicate alone is insufficient; existing endpoint refinements can distinguish fixed trajectories. Not a counterexample to cited theorems.')
    c=[sp.Integer(1),R(1,2),sp.Integer(2)];g=[sp.Integer(1),sp.Integer(2)];d=g.copy();initial=[R(1,3)]*3
    numerator,denominator=coefficients(c,g,initial)
    assert denominator==[1,-R(1,6),R(1,3)] and pair_margin(c,g,d,0,1)>0
    mixed=dict(tilts=list(map(str,c)),gaps=[1,2],rates=[1,2],initial=list(map(str,initial)),
        spectral_denominator_coefficients=list(map(str,denominator)),all_state_safe=True,
        positive_normalizer_reason='A positive tilt applied to a simplex state has normalizer at least the minimum tilt, here1/2.',
        scope='The natural finite exponential expansion violates positive denominator-coefficient hypotheses. No universal nonrepresentability claim.')
    return dict(status='PASS',symbolic_identity_dimensions=symbolic,nonunimodal_safe=nonunimodal,
        same_quotient_shape_safe_unsafe=safe_unsafe,mixed_denominator_safe=mixed,
        model_fit=False,gpu_allocated=False,new_mathematical_priority_established=False,publication_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=verify();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
