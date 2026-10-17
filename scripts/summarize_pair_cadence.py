"""Generate development tables from complete cadence audits; never select fits."""
from pathlib import Path
import argparse
import hashlib
import json
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303]
MODELS = {'original':['base_stats','residual_scalar','residual_mixture','mlp_residual'],
          'features':['base_stats','residual_scalar','residual_mixture','mlp_residual'],
          'relative':['residual_scalar','residual_mixture']}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def paired_summary(values):
    values = np.asarray(values,dtype=np.float64)
    assert values.shape == (3,) and np.isfinite(values).all()
    return dict(values=values.tolist(),mean=float(values.mean()),
                std_ddof1=float(values.std(ddof=1)),positive=int((values>0).sum()),
                zero=int((values==0).sum()),negative=int((values<0).sum()))


def summarize_dataset(root, report):
    assert report['status']=='PASS' and not report['test_scored']
    assert report['primary_comparison']=='clock_residual_scalar'
    assert set(report['arm_audits']) == set(MODELS)
    dataset=report['dataset']; arms={}; hashes={}; comparisons={}
    for arm,models in MODELS.items():
        audit=report['arm_audits'][arm]
        assert audit['dataset']==dataset and audit['all_checks_pass']
        assert audit['cleanup_verified'] and audit['checksums_verified']
        assert [run['seed'] for run in audit['per_seed']]==SEEDS
        attempt=(root/audit['attempt']).resolve();assert attempt.is_relative_to(root.resolve())
        result_path=attempt/'artifacts/results.json';result=json.loads(result_path.read_text())
        hashes[result_path.relative_to(root.resolve()).as_posix()]=sha(result_path)
        assert result['protocol_id']=='LCR-P-CADENCE-DEV-001' and result['dataset']==dataset
        assert result['complete_declared_seeds'] and not result['test_scored']
        assert [run['seed'] for run in result['per_seed_runs']]==SEEDS
        assert result['settings']['models']==models and result['settings']['epochs']==10
        assert result['settings']['config_path']==f'configs/pair_cadence_{arm}_v1.toml'
        clock_metadata=[]
        for run in result['per_seed_runs']:
            clock_path=attempt/f'seed{run["seed"]}/pair_clock.json'
            clock_path=attempt/'artifacts'/clock_path.relative_to(attempt)
            info=json.loads(clock_path.read_text())
            assert not info['test_clocks_constructed'] and not info['sampling_or_support_changed']
            for split in ['train','validation']:
                assert info[f'{split}_candidate_slots']==2*run['coverage'][split]
                assert 0<=info[f'{split}_available_slots']<=info[f'{split}_candidate_slots']
            clock_metadata.append(dict(seed=run['seed'],**info))
            hashes[clock_path.relative_to(root.resolve()).as_posix()]=sha(clock_path)
        summaries={}
        for model in models:
            runs=[run['models'][model] for run in result['per_seed_runs']]
            for run,checked in zip(runs,audit['per_seed']):
                for metric in ['ap','auc','brier']:
                    np.testing.assert_allclose(run['all_validation'][metric],checked['models'][model][metric],rtol=1e-6,atol=1e-8)
            summaries[model]=dict(
                metrics={metric:paired_summary([checked['models'][model][metric] for checked in audit['per_seed']])
                         for metric in ['ap','auc','brier']},
                selected_epochs=[run['selected_epoch'] for run in runs],
                parameter_counts=[run['parameter_count'] for run in runs],
                fit_elapsed_seconds=[run['elapsed_seconds'] for run in runs])
        arms[arm]=dict(models=summaries,coverage=[run['coverage'] for run in result['per_seed_runs']],
                       pair_clock=clock_metadata,attempt=audit['attempt'])
    for comparison,declared in report['paired_validation_ap'].items():
        kind,model=comparison.split('_',1)
        first,second=('original','features') if kind=='feature' else ('features','relative')
        assert kind in {'feature','clock'}
        left=arms[first]['models'][model]['metrics']['ap']['values']
        right=arms[second]['models'][model]['metrics']['ap']['values']
        value=paired_summary(np.asarray(right)-np.asarray(left))
        for key in ['values','mean','std_ddof1']:
            np.testing.assert_allclose(value[key],declared[key],rtol=1e-14,atol=1e-15)
        comparisons[comparison]=value
    assert set(comparisons)=={*(f'feature_{model}' for model in MODELS['features']),
                             'clock_residual_scalar','clock_residual_mixture'}
    return dict(dataset=dataset,seeds=SEEDS,arms=arms,paired_validation_ap=comparisons,
                primary_comparison='clock_residual_scalar',result_sha256=hashes,
                fit_count=sum(len(models)*len(SEEDS) for models in MODELS.values()))


def render_tables(datasets):
    primary=['| Dataset | Scalar relative − absolute AP, mean ± sample SD | Positive / zero / negative seeds |',
             '|---|---:|---:|']
    details=['| Dataset | Contrast | AP difference, mean ± sample SD | Positive / zero / negative seeds |',
             '|---|---|---:|---:|']
    for dataset,result in datasets.items():
        for name,summary in result['paired_validation_ap'].items():
            values=f'`{summary["mean"]:+.6f} ± {summary["std_ddof1"]:.6f}`'
            signs=f'`{summary["positive"]} / {summary["zero"]} / {summary["negative"]}`'
            if name=='clock_residual_scalar':primary.append(f'| {dataset} | {values} | {signs} |')
            else:details.append(f'| {dataset} | {name} | {values} | {signs} |')
    return '\n'.join(primary)+'\n','\n'.join(details)+'\n'


def render_details(datasets):
    metrics=['| Dataset | Arm | Model | AP, mean ± SD | AUC, mean ± SD | Brier, mean ± SD | Parameters | Fit seconds, mean ± SD |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    support=['| Dataset | Seed | Train queries | Validation queries | Excluded train events | Unsupported events, all splits | Train cadence slots available / total | Validation cadence slots available / total | Train fallback, seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,d in datasets.items():
        for arm,a in d['arms'].items():
            for kind,m in a['models'].items():
                assert len(set(m['parameter_counts']))==1
                cells=[f'`{m["metrics"][key]["mean"]:.6f} ± {m["metrics"][key]["std_ddof1"]:.6f}`' for key in ['ap','auc','brier']]
                metrics.append(f'| {dataset} | {arm} | {kind} | '+' | '.join(cells)+f' | `{m["parameter_counts"][0]}` | `{mean(m["fit_elapsed_seconds"]):.3f} ± {stdev(m["fit_elapsed_seconds"]):.3f}` |')
        for seed,c,m in zip(d['seeds'],d['arms']['features']['coverage'],d['arms']['features']['pair_clock']):
            support.append(f'| {dataset} | `{seed}` | `{c["train"]}` | `{c["validation"]}` | `{c["excluded_train"]}` | `{c["skipped_no_hard_support"]}` | `{m["train_available_slots"]} / {m["train_candidate_slots"]}` | `{m["validation_available_slots"]} / {m["validation_candidate_slots"]}` | `{m["fallback_seconds"]:.6f}` |')
    return '\n'.join(metrics)+'\n','\n'.join(support)+'\n'


def summarize_study(root, execution):
    progress=json.loads((execution/'progress.json').read_text())
    assert progress['status']=='completed' and progress['stage']=='all_arms_audited'
    expected={(d,a) for d in ['wikipedia','reddit'] for a in MODELS}
    assert len(progress['completed'])==6
    assert {(x['dataset'],x['arm']) for x in progress['completed']}==expected
    assert all(x['cleanup_verified'] and x['arm_audit_completed'] for x in progress['completed'])
    assert progress['dataset_audits']=={'wikipedia':'PASS','reddit':'PASS'}
    datasets={};hashes={}
    for dataset in ['wikipedia','reddit']:
        path=execution/f'{dataset}_comparison_audit.json'
        report=json.loads(path.read_text());assert report['dataset']==dataset
        datasets[dataset]=summarize_dataset(root,report);hashes[path.name]=sha(path)
    return dict(status='COMPLETE_DEVELOPMENT_SUMMARY',datasets=datasets,
                audit_sha256=hashes,progress_sha256=sha(execution/'progress.json'),
                summarizer_sha256=sha(Path(__file__)),test_scored=False,publication_eligible=False,
                limitations='Validation-selected development on inspected datasets; paired sample SD is descriptive, not a confidence interval or significance test. No novelty, causality, confirmation or lifecycle semantics admission.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Validate completeness before creating any final output directory.
    result=summarize_study(ROOT,args.execution.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    primary,secondary=render_tables(result['datasets'])
    (args.output/'primary.md').write_text(primary)
    (args.output/'secondary.md').write_text(secondary)
    metrics,support=render_details(result['datasets'])
    (args.output/'detailed_metrics.md').write_text(metrics)
    (args.output/'support.md').write_text(support)
    print('Complete audited cadence development tables generated; scientific release remains open.')
