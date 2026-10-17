"""Independent convex optimization and derivative checks for no-fit repairs."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.readout_repair import gradient_potential,project_potential,readout_caps,repair_score
from lifecycle_readout.modeling.latent_lifecycle import LatentLifecycleModule,LatentLifecycleConfig


def test_projection_matches_independent_full_matrix_optimization():
    rng=np.random.RandomState(93)
    for k in [2,3,5]:
        for _ in range(5):
            matrix=rng.normal(size=(k,k))*4
            caps=np.r_[rng.uniform(0,.7,k-2),np.inf]
            tensor=torch.tensor(matrix,dtype=torch.double)[None]
            potential=gradient_potential(tensor)
            projected=project_potential(potential,torch.tensor(caps,dtype=torch.double))[0].numpy()
            def objective(active):
                b=np.r_[0.,active];return np.square(matrix-(b[None,:]-b[:,None])).sum()
            fit=minimize(objective,np.minimum(potential[0,1:].numpy(),caps),method='SLSQP',
                bounds=[(None,c if np.isfinite(c) else None) for c in caps],options={'ftol':1e-11,'maxiter':500})
            assert fit.success
            assert abs(objective(projected[1:])-fit.fun)<1e-7
            np.testing.assert_allclose(projected[1:],fit.x,atol=2e-6,rtol=2e-6)


def test_projection_feasible_idempotent_and_not_naive_clipping():
    b=torch.tensor([[0.,np.log(4.),0.]],dtype=torch.double)
    caps=torch.tensor([np.log(2.),np.inf],dtype=torch.double)
    out=project_potential(b,caps)
    torch.testing.assert_close(out,torch.tensor([[0.,np.log(2.),-np.log(2.)/2]],dtype=torch.double))
    torch.testing.assert_close(project_potential(out,caps),out)


def test_projected_variants_have_nonpositive_silence_derivatives():
    torch.manual_seed(73)
    model=LatentLifecycleModule(LatentLifecycleConfig(hidden_dim=8)).double().eval()
    initial=torch.softmax(torch.randn(17,5,dtype=torch.double),-1);context=torch.randn(17,6,dtype=torch.double)
    base=torch.randn(17,dtype=torch.double);time=torch.logspace(-3,3,17,dtype=torch.double).requires_grad_()
    for variant in ['readout_projected','coordinate_projected','zero_safe']:
        value=repair_score(model,initial,context,base,time,variant)
        derivative,=torch.autograd.grad(value.sum(),time)
        assert (derivative<=1e-12).all()
    _,matrix=model.kernel(initial,context,torch.zeros_like(time))
    b=project_potential(gradient_potential(matrix),readout_caps(model.survival_gate.survival_weights))
    gaps=model.survival_gate.survival_weights[1:]-model.survival_gate.survival_weights[0]
    assert (((gaps.max()-gaps)*b[:,1:].exp())<=gaps.max()+1e-12).all()
