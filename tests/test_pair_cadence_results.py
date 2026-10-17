"""Cross-arm checks must reject confounded comparisons before aggregation."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pair_cadence_results import compare_arms


def records():
    arms={}
    for arm in ['original','features','relative']:
        models=['residual_scalar','residual_mixture'] if arm=='relative' else ['base_stats','residual_scalar','residual_mixture','mlp_residual']
        arms[arm]=dict(dataset='toy',settings=dict(seeds=[101,202,303],epochs=10,
            config_path=f'configs/pair_cadence_{arm}_v1.toml',models=models,
            pair_clock_mode='relative' if arm=='relative' else 'absolute',
            mask_pair_clock_features=arm=='original'),source_files_sha256={'model':'same'},seeds={})
        for seed in [101,202,303]:
            arms[arm]['seeds'][seed]=dict(candidate_fingerprint='same',
                full_feature_fingerprint='masked' if arm=='original' else 'exposed',
                models={model:dict(parameter_count=10 if model=='residual_scalar' else 20,
                    validation_ap={'original':.5,'features':.6,'relative':.65}[arm]) for model in models})
    return arms


def test_paired_differences_separate_information_from_explicit_clock():
    actual=compare_arms(records())
    assert actual['clock_residual_scalar']['values']==pytest.approx([.05]*3)
    assert actual['feature_residual_scalar']['values']==pytest.approx([.1]*3)


@pytest.mark.parametrize('defect',['budget','candidates','feature_access','capacity','source'])
def test_confounded_comparison_rejected(defect):
    arms=deepcopy(records())
    if defect=='budget':arms['relative']['settings']['epochs']=11
    if defect=='candidates':arms['relative']['seeds'][101]['candidate_fingerprint']='different'
    if defect=='feature_access':arms['relative']['seeds'][101]['full_feature_fingerprint']='different'
    if defect=='capacity':arms['relative']['seeds'][101]['models']['residual_scalar']['parameter_count']=11
    if defect=='source':arms['relative']['source_files_sha256']['model']='changed'
    with pytest.raises(AssertionError):compare_arms(arms)
