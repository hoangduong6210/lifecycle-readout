from pathlib import Path
import sys,importlib.util
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_model import RecordedArtistControl,artist_arm_stream
from lifecycle_readout.benchmark import tensor_batch


def setup():
    spec=importlib.util.spec_from_file_location('model_stream_fixture',ROOT/'tests/test_lastfm_model_stream.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);frame=m.fixture();stream,extra=m.build(frame);s=m.settings();a=frame.artist.to_numpy();data=dict(timestamps=frame.timestamp_ms.to_numpy()/1000,features=np.column_stack((a+1,(frame.destination>=0).astype(int),(frame.artist>=0).astype(int))).astype(np.float32))
    return stream,extra,s,data


def model(kind,s,seed=101):
    torch.manual_seed(seed)
    return RecordedArtistControl(kind,s,np.zeros(11),np.ones(11),3600,1,artist_vocabulary=10,seed=seed)


def test_primary_common_tensors_predictions_and_branch_capacity():
    stream,extra,s,data=setup();a=model('phase_competition',s);b=model('dose_mlp',s)
    for name,param in a.state_dict().items():
        if name.startswith(('base.','history_head.','artist_embedding.')):torch.testing.assert_close(param,b.state_dict()[name],rtol=0,atol=0)
    for m in [a,b]:m.eval()
    rows=np.arange(len(stream.event_ids));x=tensor_batch(stream,rows,data,torch.device('cpu'))
    for p,q in zip(a(*x),b(*x)):torch.testing.assert_close(p['logits'],q['logits'],rtol=0,atol=0)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    def branch(m):return sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('base.','history_head.','artist_embedding.')))
    assert branch(a)==branch(b)==13 and branch(model('scalar_clock',s))==10 and branch(model('history_only',s))==0


def test_arm_mask_and_alternate_availability_are_explicit():
    stream,extra,s,data=setup();original=stream.pair_features.copy();changed=artist_arm_stream(stream,extra,'shuffled_phase');np.testing.assert_allclose(changed.pair_features[:,:,7],np.log1p(extra['alternate_exposure']),rtol=1e-6);np.testing.assert_array_equal(changed.pair_features[:,:,8],changed.pair_features[:,:,10]);assert not artist_arm_stream(stream,extra,'activity_mlp').pair_features[:,:,7].any();np.testing.assert_array_equal(stream.pair_features,original)


def test_conditional_phase_dose_derivatives_and_scalar_equivalence():
    _,_,s,_=setup();m=model('phase_competition',s).double();m.amplitude.data.fill_(.2);clock=torch.tensor([.5,2.],dtype=torch.float64,requires_grad=True);dose=torch.tensor([1.,3.],dtype=torch.float64,requires_grad=True);r,state=m.dose_response(clock,dose);dc,de=torch.autograd.grad(r.sum(),[clock,dose]);assert (dc<=0).all() and (de<=0).all();torch.testing.assert_close(state.sum(-1),torch.ones(2,dtype=torch.float64))
    scalar=model('scalar_clock',s).double();alpha=F.softplus(scalar.raw_event);d=F.softplus(scalar.raw_clock);_,z=scalar.dose_response(clock,dose);expected=.2*torch.exp(-d[None,:]*(clock[:,None]+alpha*dose[:,None]));torch.testing.assert_close(z[:,1:],expected)


def test_small_synthetic_optimizer_and_cpu_checkpoint_replay(tmp_path):
    stream,extra,s,data=setup();m=model('dose_mlp',s);rows=np.arange(len(stream.event_ids));batch=tensor_batch(stream,rows,data,torch.device('cpu'));optimizer=torch.optim.Adam(m.parameters(),lr=.001)
    initial=m.artist_embedding.weight.detach().clone()
    for _ in range(2):
        predictions=m(*batch);loss=sum(F.binary_cross_entropy_with_logits(p['logits'],torch.full_like(p['logits'],target)) for p,target in zip(predictions,[1.,0.]));assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
    assert not torch.equal(initial,m.artist_embedding.weight)
    m.eval();expected=[p['prob'].detach() for p in m(*batch)];checkpoint=tmp_path/'synthetic.pt';torch.save(m.state_dict(),checkpoint);replayed=model('dose_mlp',s);replayed.load_state_dict(torch.load(checkpoint,weights_only=True));replayed.eval()
    for a,p in zip(expected,replayed(*batch)):torch.testing.assert_close(a,p['prob'],rtol=0,atol=0)
