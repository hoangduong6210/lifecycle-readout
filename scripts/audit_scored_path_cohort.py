"""Reconstruct complete-cohort flags/support identities from retained arrays.

This is an independent artifact/coverage/aggregation audit, not another full
optimizer or whole-predictor replay. Remote native anchor reconstruction and
dual-mode derivative checks have separate, explicitly scoped evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(attempt,output):
    if output.exists():raise FileExistsError('Fresh artifact audit output required')
    job=json.loads((attempt/'job.json').read_text());assert job['status']=='completed' and job['cleanup_verified'] and job['artifact_checksums_verified']
    assert sha(attempt/'source.zip')==job['source_bundle_sha256']
    result=json.loads((attempt/'artifacts/results.json').read_text());settings=result['settings']
    assert result['protocol_id']=='LCR-P-SCORED-FULL-001' and result['complete_declared_seeds'] and not result['model_fit'] and not result['test_scored']
    assert [r['seed'] for r in result['per_seed_runs']]==[1,7,42]
    with zipfile.ZipFile(attempt/'source.zip') as bundle:
        manifest=json.loads(bundle.read('source_manifest.json'))
        for rel,digest in manifest.items():
            if rel.startswith('src/') or rel==settings['config_path'] or rel==settings['protocol']:
                assert sha(ROOT/rel)==digest
    parent=ROOT/settings['parent_attempt'];records=[];anchor_records=[]
    for entry in result['per_seed_runs']:
        seed=entry['seed'];folder=parent/'artifacts'/f'seed{seed}'
        candidates=dict(np.load(folder/'candidate_identity.npz'));valid=candidates['split']==1;ids=candidates['event_ids'][valid]
        arrays=dict(np.load(attempt/'artifacts'/f'seed{seed}.npz'))
        np.testing.assert_array_equal(arrays['event_ids'],ids)
        np.testing.assert_array_equal(arrays['negative_destinations'],candidates['negative_destinations'][valid])
        assert len(ids)==entry['query_count'] and 2*len(ids)==entry['candidate_trajectories']
        assert len(np.unique(ids))==len(ids)
        saved=torch.load(folder/f'ordered_gate_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert sha(folder/f'ordered_gate_seed{seed}.pt')==entry['checkpoint_sha256']
        normalized=(candidates['pair_features'][valid]-np.array(saved['phi_mean'],dtype=np.float32))/np.array(saved['phi_std'],dtype=np.float32)
        np.testing.assert_allclose(arrays['context'],np.concatenate([normalized[:,0],normalized[:,1]]),rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(arrays['initial'].sum(-1),np.ones(2*len(ids)),rtol=1e-6,atol=1e-7)
        assert np.isfinite(arrays['base']).all() and np.isfinite(arrays['context']).all()
        for record in entry['path_results']:
            path=record['path'];score=arrays[path+'_native_scores'];double=arrays[path+'_double_scores'];slopes=arrays[path+'_derivatives']
            assert score.shape==double.shape==slopes.shape==(2*len(ids),25)
            assert np.isfinite(score).all() and np.isfinite(double).all() and np.isfinite(slopes).all()
            assert ((score>=0)&(score<=1)).all()
            flags=(np.diff(score,axis=-1)>1e-5).any(-1);double_flags=(np.diff(double,axis=-1)>1e-5).any(-1)
            assert int(flags.sum())==record['violation_count']
            assert float(flags.mean())==record['violation_fraction']
            assert int((flags!=double_flags).sum())==record['native_double_flag_disagreements']
            assert int((np.diff(score.clip(1e-6,1-1e-6),axis=-1)>1e-5).any(-1).sum())==record['clipped_violation_count']
            assert int((slopes>1e-8).any(-1).sum())==record['positive_derivative_trajectories']
            assert float(np.diff(score,axis=-1).max())==record['maximum_adjacent_increase']
            np.testing.assert_allclose(score,double,rtol=2e-4,atol=2e-5)
            records.append(dict(dataset=result['dataset'],seed=seed,**record))
        old=ROOT/'results/audit/scored_path_diagnostic_v1'/f"{result['dataset']}_corrected_precision"/f'seed{seed}.npz'
        anchors=dict(np.load(old));positions=np.searchsorted(ids,anchors['event_ids'])
        np.testing.assert_array_equal(ids[positions],anchors['event_ids'])
        gather=np.r_[positions,positions+len(ids)]
        maximum=0.
        for path in ['ideal_flow','implemented_flow','fixed_transition','dynamic_transition']:
            a=arrays[path+'_double_scores'][gather];b=anchors[path+'_scores']
            np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5);maximum=max(maximum,float(np.max(np.abs(a-b))))
        anchor_records.append(dict(seed=seed,prior_queries=len(positions),max_anchor_probability_difference=maximum))
        assert entry['certificate_max_error']<1e-10
    report=dict(status='PASS_COMPLETE_COHORT_ARTIFACT_AUDIT',dataset=result['dataset'],records=records,prior_anchor_agreement=anchor_records,
        runtime_release_verified=job['cleanup_verified'],source_bundle_sha256=job['source_bundle_sha256'],
        scope='All supported validation identities, saved-input normalization, complete path arrays/flags and prior anchor agreement. Remote evidence supplies native full-predictor reconstruction at observed inputs and dual-mode derivative/certificate checks. No new full-predictor or optimizer replay is claimed here.',
        model_fit=False,test_scored=False,publication_eligible=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],result['dataset'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.attempt,args.output)
