"""Reject tampered selection records without fitting a model."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tomllib
import zipfile
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("craft_fit_auditor", Path(__file__).resolve().parents[1] / "scripts/audit_craft_fit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def records(folder, vectors, patience=2):
    epochs = []
    best = -1
    selected = None
    for i, vector in enumerate(vectors, 1):
        rr = np.array(vector, dtype=float)
        np.savez(folder / f"seed42_epoch{i}_validation_rr.npz", reciprocal_rank=rr)
        improved = float(rr.mean()) > best
        if improved:
            best, selected = float(rr.mean()), i
        epochs.append(dict(epoch=i, train_count=10, candidate_trace_sha256="a"*64,
            training_seconds=1., train_loss=.5, elapsed_seconds=float(i*2),
            validation_mrr=float(rr.mean()), selected_at_this_epoch=improved))
    settings = dict(max_epochs=len(vectors), patience=patience, negative_count=2, training_budget_seconds=100)
    run = dict(seed=42, completed_epochs=len(vectors), selected_epoch=selected,
        stop_reason="max_epochs", validation_history=epochs)
    return run, settings


def test_equal_means_retain_earliest_per_query_vector(tmp_path):
    run, settings = records(tmp_path, [[1., .5], [.5, 1.]])
    _, selected, vector = audit.verify_epoch_selection(tmp_path, run, settings, 10, 2, 10.)
    assert selected == 1
    np.testing.assert_array_equal(vector, [1., .5])


def test_recovery_after_patience_does_not_excuse_continued_training(tmp_path):
    run, settings = records(tmp_path, [[1., .5], [.5, .5], [1., 1.]], patience=1)
    with pytest.raises(AssertionError, match="continued past patience"):
        audit.verify_epoch_selection(tmp_path, run, settings, 10, 2, 10.)


def test_finite_but_impossible_reciprocal_ranks_rejected(tmp_path):
    run, settings = records(tmp_path, [[.7, .5]])
    with pytest.raises(AssertionError):
        audit.verify_epoch_selection(tmp_path, run, settings, 10, 2, 10.)


def test_unknown_runtime_reason_rejected(tmp_path):
    run, settings = records(tmp_path, [[1., .5]])
    run["stop_reason"] = "runtime_cap_unrecorded"
    with pytest.raises(AssertionError, match="Undeclared"):
        audit.verify_epoch_selection(tmp_path, run, settings, 10, 2, 110.)


def test_reported_settings_and_launch_hash_both_bound():
    raw = b'config_path="config.toml"\nprotocol="protocol.toml"\nprotocol_id="P"\ndatasets=["D"]\nseeds=[42]\nmax_epochs=3\npatience=2\ntraining_budget_seconds=100\n'
    protocol = b'protocol_id="P"\nseeds=[42]\nmax_epochs=3\npatience=2\ntraining_budget_seconds=100\n'
    manifest = {"config.toml":hashlib.sha256(raw).hexdigest(), "protocol.toml":hashlib.sha256(protocol).hexdigest()}
    job = dict(config_sha256=manifest["config.toml"], protocol_sha256=manifest["protocol.toml"])
    result = dict(settings=tomllib.loads(raw.decode()), protocol_id="P", dataset="D", convergence_demonstrated=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("config.toml", raw)
        archive.writestr("protocol.toml", protocol)
    with zipfile.ZipFile(buffer) as archive:
        audit.verify_archived_contract(archive, job, manifest, result)
        changed = copy.deepcopy(result)
        changed["settings"]["patience"] = 3
        with pytest.raises(AssertionError, match="sealed configuration"):
            audit.verify_archived_contract(archive, job, manifest, changed)
        job["config_sha256"] = "0"*64
        with pytest.raises(AssertionError):
            audit.verify_archived_contract(archive, job, manifest, result)
