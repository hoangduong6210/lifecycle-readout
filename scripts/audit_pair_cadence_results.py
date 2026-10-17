"""Audit each completed cadence arm, then compare matched development outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from audit_hard_controls import audit


def candidate_fingerprint(candidates, *, all_features=False):
    h=hashlib.sha256()
    for key in sorted(candidates):
        value=candidates[key]
        if key=='pair_features' and not all_features:
            value=value[:,:,:6]
        value=np.ascontiguousarray(value)
        h.update(key.encode());h.update(str(value.shape).encode());h.update(value.dtype.str.encode());h.update(value.tobytes())
    return h.hexdigest()


def compare_arms(records):
    """Require common candidates, features, training source, budgets and capacity."""
    assert set(records)=={'original','features','relative'}
    original,features,relative=(records[k] for k in ['original','features','relative'])
    excluded={'config_path','models','pair_clock_mode','mask_pair_clock_features'}
    common=lambda r:{k:v for k,v in r['settings'].items() if k not in excluded}
    assert common(original)==common(features)==common(relative)
    assert original['dataset']==features['dataset']==relative['dataset']
    assert original['source_files_sha256']==features['source_files_sha256']==relative['source_files_sha256']
    seeds=original['settings']['seeds']
    assert all(list(r['seeds'])==seeds for r in records.values())
    differences={}
    for seed in seeds:
        a,b,c=(r['seeds'][seed] for r in [original,features,relative])
        assert a['candidate_fingerprint']==b['candidate_fingerprint']==c['candidate_fingerprint']
        assert b['full_feature_fingerprint']==c['full_feature_fingerprint']
        for model in b['models']:
            assert a['models'][model]['parameter_count']==b['models'][model]['parameter_count']
        for model in c['models']:
            assert b['models'][model]['parameter_count']==c['models'][model]['parameter_count']
        deltas={f'feature_{model}':b['models'][model]['validation_ap']-a['models'][model]['validation_ap']
                for model in b['models']}
        deltas.update({f'clock_{model}':c['models'][model]['validation_ap']-b['models'][model]['validation_ap']
                       for model in c['models']})
        for key,value in deltas.items():
            differences.setdefault(key,[]).append(value)
    return {key:dict(values=values,mean=float(np.mean(values)),
                     std_ddof1=float(np.std(values,ddof=1)) if len(values)>1 else None)
            for key,values in differences.items()}


def audit_study(attempts):
    records={};audits={}
    for arm,attempt in attempts.items():
        audits[arm]=audit(attempt.resolve(),study='cadence-v1')
        result=json.loads((attempt/'artifacts/results.json').read_text())
        assert result['settings']['config_path']==f'configs/pair_cadence_{arm}_v1.toml'
        record=dict(dataset=result['dataset'],settings=result['settings'],
                    source_files_sha256=result['source_files_sha256'],seeds={})
        for run in result['per_seed_runs']:
            with np.load(attempt/'artifacts'/f"seed{run['seed']}/candidate_identity.npz") as z:
                candidates={k:z[k] for k in z.files}
            record['seeds'][run['seed']]=dict(candidate_fingerprint=candidate_fingerprint(candidates),
                full_feature_fingerprint=candidate_fingerprint(candidates,all_features=True),
                models={model:dict(parameter_count=r['parameter_count'],validation_ap=r['all_validation']['ap'])
                        for model,r in run['models'].items()})
        records[arm]=record
    differences=compare_arms(records)
    return dict(status='PASS',dataset=records['original']['dataset'],arm_audits=audits,
        paired_validation_ap=differences,primary_comparison='clock_residual_scalar',
        test_scored=False,publication_eligible=False,
        limitations='Development validation selection on previously inspected datasets. No significance, causality, novelty or fresh-confirmation claim. Checkpoint scores are not independently re-inferred by this auditor.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arm in ['original','features','relative']:
        parser.add_argument('--'+arm,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh audit output')
    report=audit_study({arm:getattr(args,arm) for arm in ['original','features','relative']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Cadence development audit PASS; no scientific admission.')
