"""Reproduce exact algebra identities; requires optional SymPy, no model fitting."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json

import sympy as s


def verify():
    records=[]
    for m in [1,2,4]:
        x=s.symbols('x0:'+str(m));c=s.symbols('c0:'+str(m+1),positive=True)
        delta=s.symbols('d0:'+str(m),nonnegative=True)
        denominator=c[0]+sum((c[i+1]-c[0])*x[i] for i in range(m))
        numerator=sum(delta[i]*c[i+1]*x[i] for i in range(m))
        for j in range(m):
            affine=delta[j]*c[j+1]*denominator-numerator*(c[j+1]-c[0])
            assert s.simplify(s.diff(numerator/denominator,x[j])-affine/denominator**2)==0
            assert s.simplify(affine.subs({v:0 for v in x})-delta[j]*c[j+1]*c[0])==0
            for i in range(m):
                vertex=affine.subs({v:int(k==i) for k,v in enumerate(x)})
                expected=c[i+1]*(delta[j]*c[j+1]-delta[i]*(c[j+1]-c[0]))
                assert s.simplify(vertex-expected)==0
        records.append(dict(active_phases=m,quotient_derivatives='exact symbolic identity',
                            dormant_and_active_vertex_reductions='exact symbolic identity'))
    z=[F(1,5),F(1,5),F(3,5)];c=[F(1),F(3,2),F(4)];d=[F(1,10),F(5)]
    v=[d[0]*z[1]+d[1]*z[2],-d[0]*z[1],-d[1]*z[2]]
    denominator=sum(a*b for a,b in zip(z,c));drift=sum(a*b for a,b in zip(v,c))
    qdot=[ci*(vi*denominator-zi*drift)/denominator**2 for zi,vi,ci in zip(z,v,c)]
    assert qdot[1]>0>qdot[1]+2*qdot[2]
    return dict(status='PASS_SYMBOLIC_IDENTITIES_AND_EXACT_RATIONAL_WITNESS',sympy_version=s.__version__,records=records,
        witness=dict(initial=[str(x) for x in z],tilt=[str(x) for x in c],rates=[str(x) for x in d],
                     phase1_derivative=str(qdot[1]),readout_derivative=str(qdot[1]+2*qdot[2])),
        scope='Algebra checks plus rational witness; general theorem supplied by written affine-vertex proof, not symbolic induction or human peer review.',
        publication_eligible=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh verification output required')
    report=verify();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
