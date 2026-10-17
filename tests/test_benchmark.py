import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifecycle_readout import benchmark
from lifecycle_readout.modeling.latent_lifecycle import LatentLifecycleModule

ROOT = Path(__file__).resolve().parents[1]


def test_snap_variable_width_header_preserves_event_columns(tmp_path):
    raw = tmp_path / "wikipedia.csv"
    lines = ["user_id,item_id,timestamp,state_label,comma_separated_list_of_features"]
    lines.extend(f"{i % 3},{i % 4},{i},0,{i / 10},{-i / 10}" for i in range(20))
    raw.write_text("\n".join(lines) + "\n")
    data, identity = benchmark.load_dataset("wikipedia", tmp_path)
    np.testing.assert_array_equal(data["timestamps"], np.arange(20))
    np.testing.assert_array_equal(data["sources"], np.arange(20) % 3)
    assert data["features"].shape == (20, 2)
    assert identity["raw_sha256"] == benchmark.sha256(raw)
    # Prepared cache has the same raw provenance and validates its own bytes.
    cached, provenance = benchmark.load_dataset("wikipedia", tmp_path)
    np.testing.assert_array_equal(cached["features"], data["features"])
    assert provenance == identity


def test_gate_thresholds_are_read_from_protocol_and_missing_gain_fails():
    protocol = tomllib.loads((ROOT / "protocols/lifecycle_tkde_v2.toml").read_text())
    report = dict(transductive={"ap": 0.6}, batch_diversity_entropy=1.1,
                  min_decay_rate=0.02, max_transition_norm=0.1, trajectory_monotonic_rate=1.)
    base = {"transductive": {"ap": 0.5}}
    assert benchmark.gate_verdicts(report, base, protocol)["numerical_gates_pass"]
    protocol["gates"]["shortcut_mitigation_gain"]["min_threshold"] = 0.2
    assert not benchmark.gate_verdicts(report, base, protocol)["numerical_gates_pass"]
    report["transductive"]["ap"] = None
    assert not benchmark.gate_verdicts(report, base, protocol)["numerical_gates_pass"]


def test_untrained_monotonicity_is_an_architectural_property():
    module = LatentLifecycleModule().eval()
    torch.manual_seed(1)
    with torch.no_grad():
        z = torch.softmax(torch.randn(100, 5), dim=-1)
        result = module.counterfactual_trajectory(z, torch.zeros(100))
    assert result["monotonic_rate"].item() == 1.
    assert module.survival_gate(torch.tensor([[1., 0., 0., 0., 0.]])).item() > 0


def test_small_training_retains_predictions_and_all_models_use_identical_candidates(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    n = 100
    data = dict(sources=np.arange(n) % 4, destinations=10 + np.arange(n) % 7,
                timestamps=np.arange(n, dtype=float),
                features=np.random.RandomState(3).randn(n, 2).astype(np.float32))
    monkeypatch.setattr(benchmark, "load_dataset", lambda *_: (data, {"license_review": "PENDING"}))
    settings = tomllib.loads((ROOT / "configs/tkde_causal_v2.toml").read_text())
    settings.update(seeds=[1], epochs=1, hidden_dim=8, time_dim=4, max_neighbors=3,
                    batch_size=16, counterfactual_pairs=10)
    protocol = tomllib.loads((ROOT / "protocols/lifecycle_tkde_v2.toml").read_text())
    output = tmp_path / "attempt"
    result = benchmark.run_benchmark("wikipedia", settings, protocol, data_dir=tmp_path,
                                     output_dir=output, device="cpu", smoke=True)
    assert result["publication_eligible"] is False
    assert result["complete_declared_seeds"] is False
    assert (output / "checksums.sha256").exists()
    predictions = [np.load(output / "seed1" / f"{kind}_seed1_predictions.npz") for kind in settings["models"]]
    for p in predictions[1:]:
        np.testing.assert_array_equal(p["event_ids"], predictions[0]["event_ids"])
        np.testing.assert_array_equal(p["negative_destinations"], predictions[0]["negative_destinations"])
    for kind in settings["models"]:
        report = result["per_seed_runs"][0]["models"][kind]
        assert report["selected_epoch"] == 1
        assert report["validation_history"][0]["validation"]["positive_count"] > 0
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark("wikipedia", settings, protocol, data_dir=tmp_path,
                                 output_dir=output, device="cpu", smoke=True)
