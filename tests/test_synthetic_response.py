"""Scientific boundaries of the small oracle response diagnostic."""
from pathlib import Path
import sys
import tomllib
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.synthetic_response import oracle_logits, oracle_metrics, probability, run_synthetic


def test_oracle_kl_is_zero_at_truth_and_matches_bernoulli_definition():
    truth = np.array([-2., -.1, 1., 3.])
    np.testing.assert_allclose(oracle_metrics(truth, truth)['excess_expected_bce'], 0., atol=1e-15)
    predicted = np.array([-.5, 0., .3, 1.])
    p, q = probability(truth), probability(predicted)
    expected = np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
    np.testing.assert_allclose(oracle_metrics(predicted, truth)['excess_expected_bce'], expected)


def test_increasing_truth_cannot_be_improved_by_negative_only_correction():
    context = np.array([[-1.], [0.], [1.]])
    base, truth = oracle_logits(context, np.array([.1, 1., 5.]), 'increasing')
    assert (truth > base).all()
    neutral = oracle_metrics(base, truth)['excess_expected_bce']
    for offset in [.01, .1, 1., 10.]:
        assert oracle_metrics(base - offset, truth)['excess_expected_bce'] > neutral


def test_null_and_contextual_generators_do_not_assign_phase_labels():
    context = np.array([[-.9], [.9]])
    times = np.ones(2)
    base, truth = oracle_logits(context, times, 'null')
    np.testing.assert_array_equal(base, truth)
    base, truth = oracle_logits(context, times, 'contextual_decay')
    assert (truth < base).all()
    assert (base - truth)[1] > (base - truth)[0]


def test_small_run_keeps_separate_data_and_validation_selected_checkpoints(tmp_path):
    settings = tomllib.loads((ROOT / 'configs/q1_synthetic_v1.toml').read_text())
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    settings.update(seeds=[11], scenarios=['null'], train_count=16, validation_count=8,
                    evaluation_count=16, steps=3, hidden_dim=4)
    protocol.update(seeds=settings['seeds'], scenarios=settings['scenarios'])
    result = run_synthetic(settings, protocol, tmp_path / 'run')
    assert not result['publication_eligible']
    folder = tmp_path / 'run/null_seed11'
    with np.load(folder / 'train_data.npz') as train, np.load(folder / 'test_data.npz') as test:
        assert not np.array_equal(train['context'], test['context'])
    for kind in settings['models']:
        report = result['runs'][0]['models'][kind]
        best = min(report['validation_history'], key=lambda x: x['excess_expected_bce'])
        assert report['selected_step'] == best['step']
        saved = torch.load(folder / f'{kind}.pt', weights_only=True)
        assert saved['selected_step'] == best['step']
        with np.load(folder / f'{kind}_test.npz') as p, np.load(folder / 'test_data.npz') as data:
            actual = oracle_metrics(p['logits'], data['truth'])
            assert actual == report['endpoints']['test']


def test_large_cpu_run_and_protocol_mismatch_are_rejected(tmp_path):
    settings = tomllib.loads((ROOT / 'configs/q1_synthetic_v1.toml').read_text())
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    settings['train_count'] = 100000
    with pytest.raises(ValueError, match='Colab'):
        run_synthetic(settings, protocol, tmp_path / 'run')
    settings['seeds'] = [123]
    with pytest.raises(ValueError, match='declared'):
        run_synthetic(settings, protocol, tmp_path / 'run')
