"""CRAFT boundary checks against raw strict-past history, including UNIX precision."""
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.craft_adapter import StrictPastCraftHistory, make_author_model, load_author_class


def fixture():
    return dict(src=np.array([10, 11, 10, 10, 11, 10]),
                dst=np.array([1, 2, 3, 2, 1, 3]),
                time=np.array([1., 1., 2., 2., 3., 4.]) + 1_000_000_000.,
                split=np.array([0, 0, 0, 0, 1, 1]))


def test_adapter_queries_match_bruteforce_and_exclude_tied_and_future_events():
    data = fixture()
    queries = np.array([10, 10, 11, 99])
    times = np.array([2., 3., 4., 5.]) + 1_000_000_000.
    candidates = np.array([[1, 2, 3, 4]] * 4)
    actual = StrictPastCraftHistory(data).batch(queries, times, candidates, 2)
    seq, count, seq_times, query_times, items, last = [x.numpy() for x in actual]
    for row, (u, t) in enumerate(zip(queries, times)):
        history = np.flatnonzero((data['src'] == u) & (data['time'] < t))[-2:]
        assert count[row] == len(history)
        np.testing.assert_array_equal(seq[row, :len(history)], data['dst'][history])
        np.testing.assert_array_equal(seq_times[row, :len(history)], data['time'][history])
        assert (seq[row, len(history):] == 0).all()
        for col, dst in enumerate(candidates[row]):
            prior = data['time'][(data['dst'] == dst) & (data['time'] < t)]
            assert last[row, col] == (prior[-1] if len(prior) else -100000.)
    assert seq_times.dtype == np.float64 and last.dtype == np.float64
    assert seq_times[0, 0] == times[0] - 1.  # Lost under float32 UNIX timestamps.
    np.testing.assert_array_equal(items, candidates)
    np.testing.assert_array_equal(query_times, times)


def test_adapter_rejects_test_split_and_padding_collisions():
    data = fixture(); data['split'][-1] = 2
    with pytest.raises(ValueError, match='train/validation'):
        StrictPastCraftHistory(data)
    data = fixture(); data['dst'][0] = 0
    with pytest.raises(ValueError, match='padding'):
        StrictPastCraftHistory(data)


def test_pinned_author_forward_backward_and_candidate_permutation():
    folder = Path(__file__).resolve().parents[1] / 'results/audit/q1_preflight_v1/baseline_review/craft'
    if not (folder / 'models/modules.py').exists():
        pytest.skip('Optional hash-pinned author cache absent')
    torch.set_num_threads(1)
    torch.manual_seed(7)
    settings = dict(layers=2, heads=2, embedding_size=8, history_length=3,
                    hidden_dropout=0., attention_dropout=0., embedding_dropout=0.)
    model = make_author_model(folder, settings, item_count=4)
    inputs = StrictPastCraftHistory(fixture()).batch(np.array([10, 99]),
        np.array([3., 3.]) + 1_000_000_000., np.array([[1, 3], [1, 3]]), 3)
    model.eval()
    positive, negative = model.predict(*inputs)
    assert positive.shape == negative.shape == (2,)
    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    shuffled = list(inputs); shuffled[4] = inputs[4][:, [1, 0]]; shuffled[5] = inputs[5][:, [1, 0]]
    p2, n2 = model.predict(*shuffled)
    torch.testing.assert_close(positive, n2)
    torch.testing.assert_close(negative, p2)
    loss, _, _ = model.calculate_loss(*inputs)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
               for p in model.parameters())


def test_author_bytes_must_match_identity(tmp_path):
    folder = Path(__file__).resolve().parents[1] / 'results/audit/q1_preflight_v1/baseline_review/craft'
    if not (folder / 'models/modules.py').exists():
        pytest.skip('Optional hash-pinned author cache absent')
    (tmp_path / 'models').mkdir()
    (tmp_path / 'source_identity.json').write_bytes((folder / 'source_identity.json').read_bytes())
    (tmp_path / 'models/modules.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='checksum'):
        load_author_class(tmp_path)


def test_profile_entry_accepts_launcher_default_seed_without_allocating_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip('This guard probe must run without CUDA')
    root = Path(__file__).resolve().parents[1]
    # The launcher omits --seeds when the full configured seed set is requested.
    # The child must reach the CUDA guard rather than reject its arguments.
    result = subprocess.run([sys.executable, str(root / 'scripts/profile_craft.py'),
        '--dataset', 'ML-20M', '--device', 'cuda', '--config', str(root / 'configs/craft_profile_v2.toml'),
        '--output_dir', str(tmp_path / 'uncreated')], capture_output=True, text=True)
    assert result.returncode == 1 and 'requires Colab CUDA' in result.stderr
    assert not (tmp_path / 'uncreated').exists()
