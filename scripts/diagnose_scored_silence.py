"""No-fit validation-anchor diagnostic through exact archived model source."""
from collections import defaultdict, deque
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tomllib
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_history(data,candidates,ids,budget,decay):
    rows={int(event):index for index,event in enumerate(ids)}
    selected={int(event):row for row,event in enumerate(candidates['event_ids'])}
    excluded=set(map(int,candidates['excluded_training_ids']))
    neighbors=defaultdict(lambda:deque(maxlen=budget));pairs={}
    edges=np.full((len(ids),3,budget),-1,dtype=np.int64)
    phi=np.zeros((len(ids),2,6),dtype=np.float32);elapsed=np.zeros((len(ids),2),dtype=np.float32)
    source,destination,times=(data[k] for k in ['sources','destinations','timestamps'])
    start=0
    while start<=int(ids[-1]):
        time=float(times[start]);stop=int(np.searchsorted(times,time,side='right'))
        for event in range(start,stop):
            if event not in rows:continue
            row=rows[event];u=int(source[event]);v=int(destination[event]);neg=int(candidates['negative_destinations'][selected[event]])
            assert event not in excluded
            for slot,node in enumerate((u,v,neg)):
                history=list(neighbors[node]);edges[row,slot,:len(history)]=history
            for slot,node in enumerate((v,neg)):
                last,count=pairs[(u,node)];gap=time-last;assert gap>0
                elapsed[row,slot]=gap
                phi[row,slot]=[.1+.9*np.exp(-decay*gap),np.log1p(gap),np.log1p(count),count/np.sqrt(max(1,event)),
                               np.sin(2*np.pi*(time%86400)/86400),np.cos(2*np.pi*(time%86400)/86400)]
        for event in range(start,stop):
            if event in excluded:continue
            u,v=int(source[event]),int(destination[event]);neighbors[u].append(event);neighbors[v].append(event)
            pairs[(u,v)]=(time,pairs.get((u,v),(time,0))[1]+1)
        start=stop
    return edges,phi,elapsed


def diagnose(attempt,output,protocol_path):
    attempt=attempt.resolve();output.mkdir(parents=True,exist_ok=False)
    protocol=tomllib.loads(protocol_path.read_text());job=json.loads((attempt/'job.json').read_text())
    assert job['status']=='completed' and job['cleanup_verified']
    assert sha(attempt/'source.zip')==job['source_bundle_sha256']
    sealed=output/'sealed_source'
    with zipfile.ZipFile(attempt/'source.zip') as archive:
        manifest=json.loads(archive.read('source_manifest.json'))
        for rel,digest in manifest.items():
            if not rel.startswith('src/') or not rel.endswith('.py'):continue
            raw=archive.read(rel);assert hashlib.sha256(raw).hexdigest()==digest
            target=(sealed/rel).resolve();assert target.is_relative_to(sealed.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    assert not any(k=='lifecycle_readout' or k.startswith('lifecycle_readout.') for k in sys.modules)
    sys.path.insert(0,str(sealed/'src'))
    from lifecycle_readout.hard_controls import HistoryControl
    import lifecycle_readout.hard_controls as archived
    assert Path(archived.__file__).resolve().is_relative_to(sealed.resolve())
    # Load current analysis support without importing current model package code.
    spec=importlib.util.spec_from_file_location('direction_support',ROOT/'src/lifecycle_readout/scored_path.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    results=json.loads((attempt/'artifacts/results.json').read_text());assert results['protocol_id']=='LCR-P-HARD-001'
    prepared=ROOT/'resources/corpora'/f"{results['dataset']}_causal_v2.npz"
    assert sha(prepared)==results['data']['derived_sha256']
    data=dict(np.load(prepared));settings=results['settings'];torch.set_num_threads(1)
    grid=torch.logspace(protocol['normalized_times_log10_min'],protocol['normalized_times_log10_max'],protocol['normalized_times_count'],dtype=torch.double)
    records=[]
    for run in results['per_seed_runs']:
        seed=run['seed'];folder=attempt/'artifacts'/f'seed{seed}';candidate_path=folder/'candidate_identity.npz'
        candidates=dict(np.load(candidate_path));valid=np.flatnonzero(candidates['split']==1)
        selected=valid[np.linspace(0,len(valid)-1,protocol['anchor_queries_per_seed'],dtype=int)]
        ids=candidates['event_ids'][selected]
        edges,phi,elapsed=selected_history(data,candidates,ids,settings['max_neighbors'],settings['history_decay'])
        np.testing.assert_allclose(phi,candidates['pair_features'][selected],rtol=2e-6,atol=1e-7)
        np.testing.assert_array_equal(elapsed,candidates['elapsed_times'][selected])
        masks=edges>=0;safe=np.maximum(edges,0)
        ages=np.where(masks,data['timestamps'][ids,None,None]-data['timestamps'][safe],0).astype(np.float32)
        assert (ages[masks]>0).all()
        features=data['features'][safe]*masks[...,None]
        checkpoint=folder/f'ordered_gate_seed{seed}.pt';assert sha(checkpoint)==run['models']['ordered_gate']['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=HistoryControl('ordered_gate',data['features'].shape[1],saved['settings'],saved['phi_mean'],saved['phi_std'])
        model.load_state_dict(saved['model']);model.eval()
        inputs=[torch.from_numpy(x.copy()) for x in (features,ages,masks,phi,elapsed)]
        with torch.no_grad():reference=model(*inputs)
        x,dt,mask,phi_tensor,elapsed_tensor=inputs
        states=[];contexts=[];bases=[];anchor_transitions=[];observed_times=[]
        with torch.no_grad():
            for candidate,dest in enumerate((1,2)):
                h,base=model.base(x[:,0],dt[:,0],mask[:,0],x[:,dest],dt[:,dest],mask[:,dest])
                context=(phi_tensor[:,candidate]-model.phi_mean)/model.phi_std
                base=base+model.history_head(torch.cat((h,context),-1)).squeeze(-1)
                initial=model.lifecycle.encoder(h,context)
                q,matrix=model.lifecycle.kernel(initial,context,elapsed_tensor[:,candidate]/model.time_unit)
                raw=torch.sigmoid(base)*model.lifecycle.survival_gate(q)
                np.testing.assert_allclose(raw.clamp(1e-6,1-1e-6).numpy(),reference[candidate]['prob'].numpy(),rtol=2e-4,atol=2e-5)
                states.append(initial.double());contexts.append(context.double());bases.append(base.double());anchor_transitions.append(matrix.double())
                observed_times.append((elapsed_tensor[:,candidate]/model.time_unit).double())
        # Preserve the actual float32 anchor encoder/backbone; lift only the
        # frozen anchor state/context and lifecycle scorer for derivative analysis.
        model.lifecycle.double()
        initial=torch.cat(states);context=torch.cat(contexts);base=torch.cat(bases);fixed=torch.cat(anchor_transitions)
        with torch.no_grad():
            anchor_q,_=model.lifecycle.kernel(initial,context,torch.cat(observed_times))
            lifted_anchor=(torch.sigmoid(base)*model.lifecycle.survival_gate(anchor_q)).clamp(1e-6,1-1e-6)
            reference_anchor=torch.cat([value['prob'] for value in reference]).double()
            np.testing.assert_allclose(lifted_anchor.numpy(),reference_anchor.numpy(),rtol=2e-4,atol=2e-5)
            anchor_max_error=float((lifted_anchor-reference_anchor).abs().max())
        batch=len(initial);steps=len(grid);k=initial.shape[-1]
        initial=initial.repeat_interleave(steps,0);context=context.repeat_interleave(steps,0)
        affinity=torch.sigmoid(base).repeat_interleave(steps,0);fixed=fixed.repeat_interleave(steps,0)
        outputs={};rates=model.lifecycle.kernel.decay_rates.detach();weights=model.lifecycle.survival_gate.survival_weights.detach()
        for path in protocol['paths']:
            time=grid.repeat(batch).requires_grad_()
            active=initial[:,1:]*torch.exp(-time[:,None]*rates)
            ideal=torch.cat((1-active.sum(-1,keepdim=True),active),-1)
            implemented=model.lifecycle.kernel.continuous_decay(initial,time)
            matrix=None
            if path=='ideal_flow':q=ideal
            elif path=='implemented_flow':q=implemented
            elif path=='fixed_transition':
                matrix=fixed
                q=torch.softmax(implemented.clamp(min=1e-6).log()+torch.einsum('bi,bij->bj',implemented,matrix),-1)
            elif path=='dynamic_transition':q,matrix=model.lifecycle.kernel(initial,context,time)
            else:raise ValueError(path)
            gate=model.lifecycle.survival_gate(q);raw=affinity*gate
            derivative,=torch.autograd.grad(raw.sum(),time,retain_graph=True)
            certificate_error=None;boundary_count=0
            if matrix is not None:
                velocity=torch.stack([torch.autograd.grad(implemented[:,phase].sum(),time,retain_graph=True)[0] for phase in range(k)],-1)
                drift=torch.zeros_like(matrix)
                if path=='dynamic_transition':
                    # Direct network derivative includes its log1p-time input.
                    columns=[]
                    for i in range(k):
                        columns.append(torch.stack([torch.autograd.grad(matrix[:,i,j].sum(),time,retain_graph=True)[0] for j in range(k)],-1))
                    drift=torch.stack(columns,1)
                eligible=~(implemented==1e-6).any(-1);boundary_count=int((~eligible).sum())
                cert_q,cert_qdot,_=support.entropic_direction(implemented[eligible],velocity[eligible],matrix[eligible],drift[eligible])
                torch.testing.assert_close(cert_q,q[eligible],rtol=1e-12,atol=1e-12)
                certificate=affinity[eligible]*gate[eligible]*(1-gate[eligible])*support.linear_readout_direction(cert_q,cert_qdot,weights)
                torch.testing.assert_close(certificate,derivative[eligible],rtol=1e-8,atol=1e-10)
                certificate_error=float((certificate-derivative[eligible]).detach().abs().max())
            scores=raw.detach().reshape(batch,steps).numpy();slopes=derivative.detach().reshape(batch,steps).numpy()
            clipped=np.clip(scores,1e-6,1-1e-6);delta=np.diff(scores,axis=-1)
            outputs[path]=dict(scores=scores,derivatives=slopes)
            records.append(dict(dataset=results['dataset'],seed=seed,path=path,candidate_trajectories=batch,grid_points=steps,
                adjacent_increase_trajectories=int((delta>1e-5).any(-1).sum()),adjacent_increase_fraction=float((delta>1e-5).any(-1).mean()),
                positive_derivative_trajectories=int((slopes>1e-8).any(-1).sum()),maximum_adjacent_increase=float(delta.max()),
                maximum_positive_slope=float(max(0.,slopes.max())),anchor_max_probability_error=anchor_max_error,clipped_increase_trajectories=int((np.diff(clipped,axis=-1)>1e-5).any(-1).sum()),
                certificate_max_absolute_error=certificate_error,nonsmooth_boundary_points_excluded=boundary_count))
        arrays=dict(event_ids=ids,negative_destinations=candidates['negative_destinations'][selected],initial_state=torch.cat(states).numpy(),
                    normalized_context=torch.cat(contexts).numpy(),base_logits=base.numpy(),normalized_times=grid.numpy(),neighbor_edges=edges,
                    pair_features=phi,elapsed_times=elapsed)
        for path,values in outputs.items():
            for name,value in values.items():arrays[path+'_'+name]=value
        np.savez_compressed(output/f'seed{seed}.npz',**arrays)
    report=dict(status='PASS_SCOPED_DIAGNOSTIC',protocol_id=protocol['protocol_id'],protocol_sha256=sha(protocol_path),
        dataset=results['dataset'],parent_attempt=attempt.name,source_archive_sha256=sha(attempt/'source.zip'),
        analysis_support_sha256=sha(ROOT/'src/lifecycle_readout/scored_path.py'),script_sha256=sha(Path(__file__)),records=records,
        model_fit=False,gpu_allocated=False,test_scored=False,publication_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SCORED_PATH_DIAGNOSTIC',results['dataset'],'records',len(records))
    for record in records:print(json.dumps({k:record[k] for k in ['seed','path','adjacent_increase_trajectories','candidate_trajectories','maximum_adjacent_increase']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('attempt',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocols/scored_path_diagnostic_v1.toml')
    args=parser.parse_args();diagnose(args.attempt,args.output,args.protocol)
