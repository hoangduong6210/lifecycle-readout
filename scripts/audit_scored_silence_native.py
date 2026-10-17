"""Validate lifted lifecycle trajectories with native float32 full-model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(directory,output):
    if output.exists():raise FileExistsError('Fresh native audit required')
    report=json.loads((directory/'report.json').read_text())
    attempt=ROOT/'results/audit'/report['parent_attempt']
    assert sha(attempt/'source.zip')==report['source_archive_sha256']
    sealed=directory/'sealed_source'
    with zipfile.ZipFile(attempt/'source.zip') as bundle:
        manifest=json.loads(bundle.read('source_manifest.json'))
        for rel,digest in manifest.items():
            if rel.startswith('src/') and rel.endswith('.py'):assert sha(sealed/rel)==digest
    assert not any(k=='lifecycle_readout' or k.startswith('lifecycle_readout.') for k in sys.modules)
    sys.path.insert(0,str(sealed/'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    results=json.loads((attempt/'artifacts/results.json').read_text())
    prepared=ROOT/'resources/corpora'/f"{report['dataset']}_causal_v2.npz"
    assert sha(prepared)==results['data']['derived_sha256']
    data=dict(np.load(prepared));torch.set_num_threads(1);records=[]
    for run in results['per_seed_runs']:
        seed=run['seed'];stored=dict(np.load(directory/f'seed{seed}.npz'))
        folder=attempt/'artifacts'/f'seed{seed}';checkpoint=folder/f'ordered_gate_seed{seed}.pt'
        assert sha(checkpoint)==run['models']['ordered_gate']['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=HistoryControl('ordered_gate',data['features'].shape[1],saved['settings'],saved['phi_mean'],saved['phi_std'])
        model.load_state_dict(saved['model']);model.eval()
        edges=stored['neighbor_edges'];valid=edges>=0;safe=np.maximum(edges,0)
        features=data['features'][safe]*valid[...,None]
        age=np.where(valid,data['timestamps'][stored['event_ids'],None,None]-data['timestamps'][safe],0).astype(np.float32)
        steps=len(stored['normalized_times']);queries=len(stored['event_ids'])
        tensors=[torch.from_numpy(value.copy()).repeat_interleave(steps,0) for value in [features,age,valid,stored['pair_features']]]
        times=torch.tensor(np.tile(stored['normalized_times'],queries),dtype=torch.float32,requires_grad=True)
        elapsed=torch.stack((times,times),-1)*model.time_unit
        predictions=model(*tensors,elapsed)
        raw_scores=[];clipped_scores=[];raw_derivatives=[]
        for value in predictions:
            derivative,=torch.autograd.grad(value['gated_prob'].sum(),times,retain_graph=True)
            raw_scores.append(value['gated_prob'].detach().reshape(queries,steps).numpy())
            clipped_scores.append(value['prob'].detach().reshape(queries,steps).numpy())
            raw_derivatives.append(derivative.detach().reshape(queries,steps).numpy())
        raw=np.concatenate(raw_scores);clipped=np.concatenate(clipped_scores);slopes=np.concatenate(raw_derivatives)
        reference=stored['dynamic_transition_scores'];reference_slope=stored['dynamic_transition_derivatives']
        np.testing.assert_allclose(raw,reference,rtol=2e-4,atol=2e-5)
        np.testing.assert_allclose(clipped,np.clip(reference,1e-6,1-1e-6),rtol=2e-4,atol=2e-5)
        np.testing.assert_allclose(slopes,reference_slope,rtol=2e-4,atol=2e-5)
        native_violations=(np.diff(raw,axis=-1)>1e-5).any(-1)
        reference_violations=(np.diff(reference,axis=-1)>1e-5).any(-1)
        np.testing.assert_array_equal(native_violations,reference_violations)
        candidates=dict(np.load(folder/'candidate_identity.npz'))
        clock=candidates['elapsed_times'][candidates['split']==0]/model.time_unit
        lo,hi=float(clock.min()),float(clock.max());grid=stored['normalized_times']
        within=(grid[:-1]>=lo)&(grid[1:]<=hi)
        record=dict(seed=seed,candidate_trajectories=len(raw),native_violations=int(native_violations.sum()),
            native_clipped_violations=int((np.diff(clipped,axis=-1)>1e-5).any(-1).sum()),
            max_raw_probability_error=float(np.max(np.abs(raw-reference))),
            max_derivative_error=float(np.max(np.abs(slopes-reference_slope))),
            posthoc_train_clock_support=dict(minimum=lo,maximum=hi,grid_intervals_inside=int(within.sum()),
                native_violations_within_global_train_clock_range=int((np.diff(raw,axis=-1)[:,within]>1e-5).any(-1).sum()),
                scope='Descriptive global clock-range check; not per-query support or a new primary endpoint'))
        records.append(record)
        np.savez_compressed(directory/f'native_seed{seed}.npz',raw_scores=raw,clipped_scores=clipped,raw_derivatives=slopes)
    result=dict(status='PASS_NATIVE_FULL_MODEL_REPLAY',dataset=report['dataset'],records=records,
        script_sha256=sha(Path(__file__)),source_archive_sha256=report['source_archive_sha256'],
        model_fit=False,gpu_allocated=False,test_scored=False,publication_eligible=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.directory,args.output)
