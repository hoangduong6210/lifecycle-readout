"""Causal-history and matched-capacity checks for the competition pilot."""
from pathlib import Path
import sys
import tomllib

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.competing_edges import competing_exposure, competition_stream, oracle_exposure
from lifecycle_readout import competition_study as study


def fixture():
    n = 600
    return dict(sources=np.arange(n)%3, destinations=10+(np.arange(n)//3)%8,
                timestamps=np.floor(np.arange(n)*11./30)*30,
                features=np.random.RandomState(4).normal(size=(n, 2)).astype(np.float32))


def prepared(unseen=0.):
    data = fixture()
    stream = build_candidate_stream(**data, unseen_ratio=unseen, negative_sampling="matched_historical", max_neighbors=3)
    arrays = competing_exposure(data, stream, dimensions=8, profile_seed=17)
    return data, stream, arrays


@pytest.mark.parametrize("unseen", [0., .1])
def test_full_synthetic_feature_reconstruction_exclusions_and_ties(unseen):
    data, stream, arrays = prepared(unseen)
    rows = np.flatnonzero(stream.split <= 1)
    raw, shuffle = oracle_exposure(data, stream, rows, dimensions=8, profile_seed=17)
    np.testing.assert_allclose(arrays["raw"][rows], raw, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(arrays["shuffled_exposure"][rows], shuffle, rtol=1e-10, atol=1e-10)
    assert not arrays["queried"][stream.split == 2].any()
    assert not arrays["raw"][stream.split == 2].any()
    assert (arrays["raw"][:, :, 1] <= arrays["raw"][:, :, 0]+1e-10).all()


def test_future_events_and_current_event_features_do_not_change_past_exposure():
    data, stream, arrays = prepared()
    altered = {key: value.copy() for key, value in data.items()}
    altered["destinations"][data["timestamps"] > stream.validation_end] += 1000
    altered["features"][:] = 12345
    again = competing_exposure(altered, stream, dimensions=8, profile_seed=17)
    np.testing.assert_array_equal(arrays["raw"], again["raw"])
    np.testing.assert_array_equal(arrays["shuffled_exposure"], again["shuffled_exposure"])


def test_arm_masks_keep_candidates_and_primary_inputs_identical():
    _, stream, arrays = prepared()
    arms = {arm:competition_stream(stream, arrays, arm) for arm in
            ("masked_mlp","activity_mlp","profile_mlp","shuffled_mlp","activity_displacement","profile_displacement")}
    assert not arms["masked_mlp"].pair_features[:, :, 6:].any()
    assert not arms["activity_mlp"].pair_features[:, :, 7].any()
    for arm in ("profile_displacement", "activity_displacement"):
        np.testing.assert_array_equal(arms[arm].pair_features, arms["profile_mlp"].pair_features)
    for other in arms.values():
        np.testing.assert_array_equal(other.event_ids, stream.event_ids)
        np.testing.assert_array_equal(other.negative_destinations, stream.negative_destinations)
        np.testing.assert_array_equal(other.pair_features[:, :, :6], stream.pair_features)


def test_initialization_capacity_and_exact_zero_amplitude_recovery():
    settings = tomllib.loads((ROOT/"configs/competition_development_v1.toml").read_text())
    settings.update(hidden_dim=8, time_dim=4, max_neighbors=3)
    models = []
    for kind in settings["models"]:
        torch.manual_seed(101)
        models.append(study.CompetitionControl(kind, 2, settings, np.zeros(10), np.ones(10)))
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert len(set(counts)) == 1
    for model in models[1:]:
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, models[0].state_dict()[key])
    x = torch.ones(2, 3, 3, 2); dt = torch.ones(2, 3, 3); mask = torch.ones(2, 3, 3, dtype=torch.bool)
    phi = torch.ones(2, 2, 10); elapsed = torch.ones(2, 2)
    for model in models:
        for prediction in model(x, dt, mask, phi, elapsed):
            torch.testing.assert_close(prediction["logits"], prediction["base_logits"])


def test_inhibitory_partial_derivative_at_fixed_context():
    # Context fixed: softplus(g) is a positive scale; gamma is projected.
    exposure = torch.tensor([0., 1., 100.], requires_grad=True)
    probability = torch.sigmoid(2.-torch.tensor(.7).clamp(min=0)*torch.nn.functional.softplus(torch.tensor(.3))*torch.log1p(exposure))
    derivative, = torch.autograd.grad(probability.sum(), exposure)
    assert (derivative < 0).all()


def test_small_synthetic_pipeline_saves_validation_only(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = fixture()
    monkeypatch.setattr(study, "load_dataset", lambda *_:(data, {"dataset":"synthetic"}))
    settings = tomllib.loads((ROOT/"configs/competition_development_v1.toml").read_text())
    settings.update(seeds=[101], epochs=1, hidden_dim=8, time_dim=4, max_neighbors=3, profile_dimensions=8, oracle_anchors_per_split=2)
    protocol = tomllib.loads((ROOT/settings["protocol"]).read_text())
    protocol["inductive_unseen_node_ratio"] = 0
    original = study.score
    def validation_only(model, rows, stream, *args):
        assert (stream.split[rows] == 1).all()
        return original(model, rows, stream, *args)
    monkeypatch.setattr(study, "score", validation_only)
    result = study.run_competition("wikipedia", settings, protocol, data_dir=tmp_path,
                                  output_dir=tmp_path/"attempt", device="cpu", smoke=True)
    assert not result["test_scored"] and not result["publication_eligible"]
    assert not result["complete_declared_seeds"]
    assert len(result["per_seed_runs"][0]["models"]) == 6
    assert len(set(x["parameter_count"] for x in result["per_seed_runs"][0]["models"].values())) == 1
