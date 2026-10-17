"""Synthetic reporting envelopes, not execution or predictive evidence."""
from pathlib import Path
import importlib.util
import json

import pytest

spec=importlib.util.spec_from_file_location('cadence_summary',Path(__file__).parents[1]/'scripts/summarize_pair_cadence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def make_report(root):
    report=dict(status='PASS',dataset='fixture',test_scored=False,
                primary_comparison='clock_residual_scalar',arm_audits={},paired_validation_ap={})
    original=[.6,.62,.59];offsets={'original':0.,'features':.002,'relative':.001}
    for arm,models in module.MODELS.items():
        attempt=root/arm;folder=attempt/'artifacts';folder.mkdir(parents=True)
        runs=[];audited=[]
        for seed,value in zip(module.SEEDS,original):
            metrics={k:value+offsets[arm] for k in ['ap','auc','brier']}
            seed_folder=folder/f'seed{seed}';seed_folder.mkdir()
            (seed_folder/'pair_clock.json').write_text(json.dumps(dict(test_clocks_constructed=False,
                sampling_or_support_changed=False,train_candidate_slots=20,train_available_slots=10,
                validation_candidate_slots=14,validation_available_slots=6)))
            runs.append(dict(seed=seed,coverage={'validation':7,'train':10},models={model:dict(
                all_validation=metrics,selected_epoch=1,parameter_count=100,
                elapsed_seconds=2.) for model in models}))
            audited.append(dict(seed=seed,models={model:dict(metrics) for model in models}))
        result=dict(protocol_id='LCR-P-CADENCE-DEV-001',dataset='fixture',
                    complete_declared_seeds=True,test_scored=False,per_seed_runs=runs,
                    settings=dict(models=models,epochs=10,config_path=f'configs/pair_cadence_{arm}_v1.toml'))
        (folder/'results.json').write_text(json.dumps(result))
        report['arm_audits'][arm]=dict(dataset='fixture',all_checks_pass=True,
            cleanup_verified=True,checksums_verified=True,per_seed=audited,attempt=arm)
    for key in [*(f'feature_{m}' for m in module.MODELS['features']),
                'clock_residual_scalar','clock_residual_mixture']:
        delta=.002 if key.startswith('feature_') else -.001
        report['paired_validation_ap'][key]=dict(values=[delta]*3,mean=delta,std_ddof1=0.)
    return report


def test_complete_dataset_keeps_primary_negative_and_every_seed(tmp_path):
    report=make_report(tmp_path);result=module.summarize_dataset(tmp_path,report)
    assert result['fit_count']==30
    primary=result['paired_validation_ap']['clock_residual_scalar']
    assert primary['negative']==3 and primary['positive']==0
    assert primary['mean']==pytest.approx(-.001)
    table,secondary=module.render_tables({'fixture':result})
    assert '-0.001000' in table and 'clock_residual_mixture' in secondary


@pytest.mark.parametrize('mutation',['missing_seed','changed_primary','changed_mean'])
def test_invalid_audit_envelopes_rejected(tmp_path,mutation):
    report=make_report(tmp_path)
    if mutation=='missing_seed':report['arm_audits']['relative']['per_seed'].pop()
    if mutation=='changed_primary':report['primary_comparison']='clock_residual_mixture'
    if mutation=='changed_mean':report['paired_validation_ap']['clock_residual_scalar']['mean']=.1
    with pytest.raises(AssertionError):module.summarize_dataset(tmp_path,report)


def test_only_existing_audit_rounding_tolerance_is_allowed(tmp_path):
    report=make_report(tmp_path)
    checked=report['arm_audits']['features']['per_seed'][0]['models']['residual_scalar']
    checked['brier']+=1e-8
    result=module.summarize_dataset(tmp_path,report)
    assert result['arms']['features']['models']['residual_scalar']['metrics']['brier']['values'][0]==checked['brier']
    checked['brier']+=.01
    with pytest.raises(AssertionError):module.summarize_dataset(tmp_path,report)


def test_running_study_cannot_generate_final_tables(tmp_path):
    (tmp_path/'progress.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(AssertionError):module.summarize_study(tmp_path,tmp_path)
