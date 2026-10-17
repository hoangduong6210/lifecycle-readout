"""Training correctness on tiny data, not author benchmark reproduction."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.craft_training import fit, training_negatives


def fixture():
    return dict(src=np.array([10, 11, 10, 10, 11, 10]), dst=np.array([1, 2, 3, 2, 1, 3]),
        time=np.array([1., 1., 2., 2., 3., 4.]) + 1_000_000_000.,
        split=np.array([0, 0, 0, 0, 1, 1]), event_ids=np.arange(6))


def test_training_negative_excludes_other_positive_at_tied_timestamp():
    data = fixture()
    rows = np.array([2, 3] * 40)
    negative = training_negatives(data, rows, np.random.RandomState(42), np.array([1, 2, 3]))
    # Both destinations 2 and 3 are positive for source 10 at t=2.
    np.testing.assert_array_equal(negative, np.ones(len(rows)))


def test_complete_toy_epochs_select_validation_checkpoint_and_reconstruct_mrr(tmp_path):
    folder = ROOT / 'results/audit/q1_preflight_v1/baseline_review/craft'
    if not (folder / 'models/modules.py').exists():
        pytest.skip('Optional pinned author cache absent')
    torch.set_num_threads(1)
    data = fixture()
    candidates = dict(event_ids=data['event_ids'][4:], sources=data['src'][4:],
        positive_destinations=data['dst'][4:], timestamps=data['time'][4:],
        negative_destinations=np.array([[2, 3], [1, 2]]))
    settings = dict(protocol_id='technical-smoke-only', seeds=[7], max_epochs=2, patience=1,
        training_budget_seconds=60, train_batch_size=3, validation_batch_size=2, negative_count=2,
        layers=1, heads=2, embedding_size=8, history_length=3, learning_rate=.001,
        hidden_dropout=0., attention_dropout=0., embedding_dropout=0.)
    result = fit(data, candidates, settings, folder, tmp_path / 'out', device='cpu', technical_smoke=True)
    run = result['per_seed_runs'][0]
    assert all(epoch['train_count'] == 4 for epoch in run['validation_history'])
    assert run['selected_epoch'] == 1 + np.argmax([e['validation_mrr'] for e in run['validation_history']])
    with np.load(tmp_path / 'out/CRAFT_seed7_validation_predictions.npz') as saved:
        p, q = saved['positive'], saved['negative']
        expected = 1. / (1 + (q > p[:, None]).sum(axis=1) + .5 * (q == p[:, None]).sum(axis=1))
        np.testing.assert_array_equal(saved['reciprocal_rank'], expected)
        assert float(expected.mean()) == run['models']['CRAFT']['validation_mrr']
        np.testing.assert_array_equal(saved['event_ids'], candidates['event_ids'])
    assert result['smoke'] and not result['test_scored'] and not result['publication_eligible']
    assert json.loads((tmp_path / 'out/results.json').read_text())['complete_declared_seeds']


def test_cpu_full_training_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='Full training'):
        fit(fixture(), {}, {}, tmp_path, tmp_path / 'not-created', device='cpu')
    assert not (tmp_path / 'not-created').exists()
