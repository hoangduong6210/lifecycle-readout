"""Scientific acceptance tests for shared history and hard-negative support."""
from pathlib import Path
import sys
import tomllib
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout import hard_controls
from lifecycle_readout.colab_workflow import bootstrap_script, source_bundle


def data_fixture():
    # Many competing recurrent destinations at each source, including tied events.
    n = 600
    return dict(sources=np.arange(n) % 3, destinations=10 + (np.arange(n) // 3) % 8,
                timestamps=np.arange(n, dtype=float) * 11,
                features=np.random.RandomState(4).normal(size=(n, 2)).astype(np.float32))


def test_hard_negatives_are_matched_and_do_not_include_simultaneous_events():
    data = data_fixture(); data["timestamps"] = np.floor(data["timestamps"] / 30) * 30
    stream = build_candidate_stream(**data, unseen_ratio=0, negative_sampling="matched_historical")
    assert stream.skipped_no_negative_ids.size
    assert len(stream.event_ids) > 10
    for row, event in enumerate(stream.event_ids):
        u, t = data["sources"][event], data["timestamps"][event]
        bins = []
        for dest in (data["destinations"][event], stream.negative_destinations[row]):
            history = (data["sources"] == u) & (data["destinations"] == dest) & (data["timestamps"] < t)
            assert history.sum() > 0
            elapsed = t - data["timestamps"][history][-1]
            bins.append((int(np.floor(np.log2(history.sum()))), int(np.floor(np.log2(1. + elapsed / 1000.)))))
        assert bins[0] == bins[1]
        assert stream.negative_destinations[row] not in data["destinations"][(data["sources"] == u) & (data["timestamps"] == t)]
        edges = stream.neighbor_edges[row]
        assert (data["timestamps"][edges[edges >= 0]] < t).all()


def test_hard_empty_support_never_falls_back_to_easy_negatives():
    data = data_fixture(); data["destinations"][:] = 10
    with pytest.raises(ValueError, match="No eligible negatives"):
        build_candidate_stream(**data, unseen_ratio=0, negative_sampling="matched_historical")


def test_all_controls_retain_identical_test_candidates_and_train_only_normalization(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = data_fixture()
    monkeypatch.setattr(hard_controls, "load_dataset", lambda *_: (data, {"dataset": "synthetic"}))
    settings = tomllib.loads((ROOT / "configs/hard_controls_v1.toml").read_text())
    settings.update(seeds=[1], epochs=1, hidden_dim=8, time_dim=4, max_neighbors=3, batch_size=64)
    protocol = tomllib.loads((ROOT / settings["protocol"]).read_text())
    protocol["inductive_unseen_node_ratio"] = 0
    result = hard_controls.run_controls("wikipedia", settings, protocol, data_dir=tmp_path,
                                        output_dir=tmp_path / "attempt", device="cpu", smoke=True)
    assert not result["publication_eligible"] and not result["complete_declared_seeds"]
    predictions = []
    stream = build_candidate_stream(**data, seed=1, unseen_ratio=0, max_neighbors=3, negative_sampling="matched_historical")
    expected_mean = stream.pair_features[stream.split == 0].reshape(-1, 6).mean(0)
    for kind in settings["models"]:
        predictions.append(np.load(tmp_path / "attempt/seed1" / f"{kind}_seed1_predictions.npz"))
        checkpoint = torch.load(tmp_path / "attempt/seed1" / f"{kind}_seed1.pt", weights_only=True)
        np.testing.assert_allclose(checkpoint["phi_mean"], expected_mean)
        assert result["per_seed_runs"][0]["models"][kind]["selected_epoch"] == 1
    for p in predictions[1:]:
        np.testing.assert_array_equal(p["negative_destinations"], predictions[0]["negative_destinations"])
        np.testing.assert_array_equal(p["event_ids"], predictions[0]["event_ids"])
        np.testing.assert_array_equal(p["pair_features"], predictions[0]["pair_features"])
    assert (predictions[0]["pair_features"][:, :, 2] > 0).all()


def test_same_seed_backbone_and_history_head_are_equal_for_controls():
    settings = tomllib.loads((ROOT / "configs/hard_controls_v1.toml").read_text())
    models = []
    for kind in settings["models"]:
        torch.manual_seed(7)
        models.append(hard_controls.HistoryControl(kind, 2, settings, np.zeros(6), np.ones(6)))
    for model in models[1:]:
        for name, parameter in models[0].base.state_dict().items():
            torch.testing.assert_close(parameter, model.base.state_dict()[name])
    for model in models[2:]:
        for name, parameter in models[1].history_head.state_dict().items():
            torch.testing.assert_close(parameter, model.history_head.state_dict()[name])


def test_colab_bundle_uses_exact_new_protocol_and_entrypoint(tmp_path):
    import zipfile
    settings = tomllib.loads((ROOT / "configs/hard_controls_v1.toml").read_text())
    target = tmp_path / "source.zip"
    source_bundle(ROOT, target, settings=settings)
    with zipfile.ZipFile(target) as bundle:
        assert "scripts/train_hard_controls.py" in bundle.namelist()
        assert settings["protocol"] in bundle.namelist()
        assert settings["config_path"] in bundle.namelist()
    script = bootstrap_script("wikipedia", ["--seeds", "42"], False, settings=settings)
    assert "scripts/train_hard_controls.py" in script
    compile(script, "remote_job.py", "exec")


def test_monotonicity_does_not_require_ordering_between_active_phases():
    # Exact derivative of the theorem, including an unordered active vector.
    weights = np.array([-2., 3., 0., 2., 1.])
    rates = np.array([.1, .3, .2, .4])
    z = np.random.RandomState(1).dirichlet(np.ones(5), 100)
    for t in [0., .1, 2., 100.]:
        derivative = -(rates * (weights[1:] - weights[0]) * z[:, 1:] * np.exp(-rates * t)).sum(1)
        assert (derivative <= 0).all()
    # If one active weight is below dormant, that basis state increases.
    weights[2] = -3.
    assert -rates[1] * (weights[2] - weights[0]) > 0


def test_connected_decay_can_have_exactly_zero_gradient():
    from lifecycle_readout.modeling.latent_lifecycle import LatentLifecycleModule
    module = LatentLifecycleModule().eval()
    dormant = torch.tensor([[1., 0., 0., 0., 0.]])
    probability = module.survival_gate(module.kernel.continuous_decay(dormant, torch.tensor([1.])))
    probability.sum().backward()
    torch.testing.assert_close(module.kernel.decay_param.grad, torch.zeros(4))


def test_proposition_one_includes_opposing_negative_example_gradient():
    w = torch.tensor([-.4, .5], requires_grad=True)
    z = torch.tensor([.7, .3]); base = torch.tensor(.8)
    gate = torch.sigmoid(w @ z)
    positive_loss = -torch.log(base * gate)
    negative_loss = -torch.log1p(-base * gate)
    pos_grad = torch.autograd.grad(positive_loss, w, retain_graph=True)[0][0]
    neg_grad = torch.autograd.grad(negative_loss, w)[0][0]
    torch.testing.assert_close(pos_grad, -(1 - gate.detach()) * z[0])
    torch.testing.assert_close(neg_grad, base * gate.detach() * (1 - gate.detach()) * z[0] / (1 - base * gate.detach()))
    assert pos_grad < 0 and neg_grad > 0


def test_development_controls_do_not_score_test_and_fit_clock_on_train(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = data_fixture()
    monkeypatch.setattr(hard_controls, 'load_dataset', lambda *_: (data, {'dataset': 'synthetic'}))
    settings = tomllib.loads((ROOT / 'configs/q1_development_v1.toml').read_text())
    settings.update(seeds=[101], epochs=1, hidden_dim=8, time_dim=4, max_neighbors=3, batch_size=64)
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    protocol['inductive_unseen_node_ratio'] = 0
    original_score = hard_controls.score

    def score_validation_only(model, rows, stream, *args):
        assert (stream.split[rows] == 1).all()
        return original_score(model, rows, stream, *args)

    monkeypatch.setattr(hard_controls, 'score', score_validation_only)
    result = hard_controls.run_controls('wikipedia', settings, protocol, data_dir=tmp_path,
        output_dir=tmp_path / 'attempt', device='cpu', smoke=True)
    assert result['evaluation_split'] == 'validation' and not result['test_scored']
    stream = build_candidate_stream(**data, seed=101, unseen_ratio=0, max_neighbors=3,
                                    negative_sampling='matched_historical')
    train_times = stream.elapsed_times[stream.split == 0].reshape(-1)
    expected_scale = np.median(train_times[train_times > 0])
    for kind, report in result['per_seed_runs'][0]['models'].items():
        assert 'all_test' not in report
        assert report['effective_decay_time_unit'] == expected_scale
        with np.load(tmp_path / 'attempt/seed101' / f'{kind}_seed101_predictions.npz') as pred:
            np.testing.assert_array_equal(pred['event_ids'], stream.event_ids[stream.split == 1])
    assert 'recurrence_only_test' not in result['per_seed_runs'][0]


def test_development_colab_bootstrap_selects_development_configuration():
    settings = tomllib.loads((ROOT / 'configs/q1_development_v1.toml').read_text())
    script = bootstrap_script('wikipedia', [], False, settings=settings)
    assert '--config' in script and 'configs/q1_development_v1.toml' in script
    compile(script, 'remote_job.py', 'exec')
