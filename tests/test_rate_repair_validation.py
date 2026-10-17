from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.special import expit
from lifecycle_readout.rate_repair_validation import tilt_score,metrics


def test_score_and_derivative_match_exact_tilt_witness():
    initial=np.array([[.1,.1,.8]]);base=np.zeros(1);potential=np.array([[0.,np.log(4),np.log(4)]]);rates=np.array([1.,2.]);gaps=np.array([1.,2.]);time=np.zeros(1);p,d=tilt_score(initial,base,time,potential,rates,gaps,0.,0.);gate=expit(68/37);np.testing.assert_allclose(p,.5*gate,atol=1e-15);np.testing.assert_allclose(d,-.5*gate*(1-gate)*1416/1369,atol=1e-15)
    right,_=tilt_score(initial,base,np.array([1e-6]),potential,rates,gaps,0.,0.);np.testing.assert_allclose((right-p)/1e-6,d,atol=1e-6)


def test_metrics_keep_ties_and_paired_query_order():
    out=metrics(np.array([.6,.6,.6,.3]),2,1e-6);np.testing.assert_allclose([out['ap'],out['auc'],out['pair_accuracy']],[2/3,.75,.75],atol=1e-15)
