"""Batched full-cohort kernel engine versus direct archived-compatible calls."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.scored_path_cohort import evaluate_paths
from lifecycle_readout.scored_path import entropic_direction
from lifecycle_readout.modeling.latent_lifecycle import LatentLifecycleModule,LatentLifecycleConfig


def test_batched_engine_agrees_with_direct_calls_and_both_ad_modes():
    torch.manual_seed(92);torch.set_num_threads(1)
    model=LatentLifecycleModule(LatentLifecycleConfig(hidden_dim=8)).double().eval()
    initial=torch.softmax(torch.randn(7,5,dtype=torch.double),-1)
    context=torch.randn(7,6,dtype=torch.double);base=torch.randn(7,dtype=torch.double)
    grid=torch.logspace(-3,3,25,dtype=torch.double)
    _,matrix=model.kernel(initial,context,torch.ones(7,dtype=torch.double))
    matrix=matrix.detach()
    paths=['ideal_flow','implemented_flow','fixed_transition','dynamic_transition']
    full=evaluate_paths(model,initial,context,base,matrix,grid,paths=paths,derivative_support=entropic_direction)
    for path in paths:
        chunks=[evaluate_paths(model,initial[i:i+2],context[i:i+2],base[i:i+2],matrix[i:i+2],grid,paths=[path],derivative_support=entropic_direction)[path] for i in range(0,7,2)]
        import numpy as np
        np.testing.assert_allclose(full[path]['scores'],np.concatenate([x['scores'] for x in chunks]),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(full[path]['derivatives'],np.concatenate([x['derivatives'] for x in chunks]),rtol=1e-10,atol=1e-12)
    for column,t in enumerate(grid):
        state,_=model.kernel(initial,context,t.expand(7))
        expected=(torch.sigmoid(base)*model.survival_gate(state)).detach().numpy()
        np.testing.assert_allclose(full['dynamic_transition']['scores'][:,column],expected,rtol=1e-12,atol=1e-12)
