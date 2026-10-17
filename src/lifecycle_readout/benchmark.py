"""Matched causal-input benchmarks with validation-only checkpoint selection."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import tomllib
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from .causal_stream import build_candidate_stream, neighbor_batch, validate_events
from .modeling.temporal_mixer import BaseTemporalPredictor
from .modeling.latent_lifecycle import LatentLifecycleConfig, LatentLifecycleModule

URLS = {name: f"https://snap.stanford.edu/jodie/{name}.csv"
        for name in ("wikipedia", "reddit", "mooc", "lastfm")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_dataset(name, data_dir):
    """Retain raw bytes and record source identity without inferring license rights."""
    if name not in URLS:
        raise ValueError(f"Unknown dataset: {name}")
    root = Path(data_dir); root.mkdir(parents=True, exist_ok=True)
    csv = root / f"{name}.csv"
    cache = root / f"{name}_causal_v2.npz"
    identity = root / f"{name}_causal_v2.identity.json"
    if cache.exists() and identity.exists():
        provenance = json.loads(identity.read_text())
        if provenance.get("derived_sha256") != sha256(cache):
            raise ValueError("Prepared corpus checksum mismatch")
        with np.load(cache, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("sources", "destinations", "timestamps", "features")}
        validate_events(**arrays)
        return arrays, provenance
    if not csv.exists():
        tmp = csv.with_suffix(".download")
        try:
            request = urllib.request.Request(URLS[name], headers={"User-Agent": "lifecycle-readout-research"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            tmp.replace(csv)
        finally:
            tmp.unlink(missing_ok=True)
    # SNAP's header describes all feature columns with one placeholder, while
    # data rows contain many feature fields. Letting pandas infer the header
    # silently turns leading event columns into a MultiIndex.
    df = pd.read_csv(csv, skiprows=1, header=None)
    if df.shape[1] < 4:
        raise ValueError("SNAP CSV requires user, destination, timestamp, label columns")
    sources = df.iloc[:, 0].to_numpy(dtype=np.int64)
    destinations = df.iloc[:, 1].to_numpy(dtype=np.int64) + int(sources.max()) + 1
    timestamps = df.iloc[:, 2].to_numpy(dtype=np.float64)
    features = df.iloc[:, 4:].to_numpy(dtype=np.float32)
    if not features.shape[1]:
        features = np.zeros((len(df), 1), dtype=np.float32)
    validate_events(sources, destinations, timestamps, features)
    arrays = dict(sources=sources, destinations=destinations, timestamps=timestamps, features=features)
    provenance = dict(dataset=name, source_url=URLS[name], raw_sha256=sha256(csv),
                      raw_bytes=csv.stat().st_size, total_events=len(df), feature_dim=features.shape[1],
                      preprocessing="bipartite-disjoint-ID-shift; native-time-units; no-current-event-features",
                      license_review="PENDING")
    np.savez_compressed(cache, **arrays)
    provenance["derived_sha256"] = sha256(cache)
    save_json(identity, provenance)
    return arrays, provenance


class MatchedPredictor(nn.Module):
    def __init__(self, kind, feat_dim, settings):
        super().__init__()
        if kind not in ("base", "ordered_gate", "unordered_gate"):
            raise ValueError(kind)
        self.kind = kind
        self.neutral_new_edges = settings.get("neutral_new_edges", False) if isinstance(settings, dict) else False
        self.base = BaseTemporalPredictor(feat_dim=feat_dim, hidden_dim=settings["hidden_dim"],
                                         time_dim=settings["time_dim"], max_neighbors=settings["max_neighbors"])
        self.lifecycle = None
        if kind != "base":
            cfg = LatentLifecycleConfig(n_phases=settings["latent_phases"], hidden_dim=settings["hidden_dim"])
            self.lifecycle = LatentLifecycleModule(cfg, unconstrained=kind == "unordered_gate")

    def forward(self, x, dt, mask, phi, pair_dt):
        outputs = []
        for destination_row, candidate in ((1, 0), (2, 1)):
            h, logits = self.base(x[:, 0], dt[:, 0], mask[:, 0], x[:, destination_row],
                                 dt[:, destination_row], mask[:, destination_row])
            if self.lifecycle is None:
                outputs.append({"gated_logits": logits, "gated_prob": torch.sigmoid(logits),
                                "base_logits": logits})
            else:
                out = self.lifecycle(h, phi[:, candidate], pair_dt[:, candidate], logits)
                has_history = (phi[:, candidate, 2] > 0.0).unsqueeze(-1)
                e_0 = torch.zeros_like(out["z_next"])
                e_0[:, 0] = 1.0
                z_eff = torch.where(has_history, out["z_next"], e_0)
                psi_computed = self.lifecycle.survival_gate(z_eff)
                if self.neutral_new_edges:
                    psi_eff = torch.where(has_history, psi_computed, torch.ones_like(psi_computed))
                else:
                    psi_eff = psi_computed
                base_prob = torch.sigmoid(logits)
                gated_prob = base_prob * psi_eff
                gated_prob_clamped = gated_prob.clamp(1e-6, 1.0 - 1e-6)
                gated_logits = torch.log(gated_prob_clamped / (1.0 - gated_prob_clamped))
                out["z_next"] = z_eff
                out["survival_gate"] = psi_eff
                out["gated_prob"] = gated_prob
                out["gated_logits"] = gated_logits
                out["base_logits"] = logits
                outputs.append(out)
        return outputs


def tensor_batch(stream, rows, data, device):
    x, dt, mask = neighbor_batch(stream, rows, data["timestamps"], data["features"])
    return (torch.as_tensor(x, device=device), torch.as_tensor(dt, device=device),
            torch.as_tensor(mask, device=device),
            torch.as_tensor(stream.pair_features[rows], device=device),
            torch.as_tensor(stream.elapsed_times[rows], device=device))


def metrics(pos, neg):
    if not len(pos):
        return {"ap": None, "auc": None, "positive_count": 0, "negative_count": 0}
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    return {"ap": float(average_precision_score(y, scores)),
            "auc": float(roc_auc_score(y, scores)),
            "positive_count": len(pos), "negative_count": len(neg)}


def evaluate(model, rows, stream, data, settings, device):
    model.eval(); positive, negative, phases, base_logits, z_next = [], [], [], [], []
    max_norm = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), settings["batch_size"]):
            batch = rows[start:start + settings["batch_size"]]
            p, q = model(*tensor_batch(stream, batch, data, device))
            positive.append(p["gated_prob"].cpu().numpy()); negative.append(q["gated_prob"].cpu().numpy())
            if model.lifecycle is not None:
                phases.append(p["z_curr"].cpu().numpy()); z_next.append(p["z_next"].cpu().numpy())
                base_logits.append(p["base_logits"].cpu().numpy())
                max_norm = max(max_norm, float(torch.linalg.matrix_norm(p["T_matrix"], ord="fro").max()))
    if not len(rows):
        raise ValueError("No eligible candidates in requested evaluation split")
    return dict(positive=np.concatenate(positive), negative=np.concatenate(negative),
                z=np.concatenate(phases) if phases else None,
                z_next=np.concatenate(z_next) if z_next else None,
                base_logits=np.concatenate(base_logits) if base_logits else None,
                max_transition_norm=max_norm)


def train_model(kind, seed, stream, data, settings, device, output):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MatchedPredictor(kind, data["features"].shape[1], settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    train_rows = np.flatnonzero(stream.split == 0)
    val_rows = np.flatnonzero(stream.split == 1)
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Training or validation split has no eligible candidates")
    best_ap, best_epoch, history = -1.0, None, []
    checkpoint = output / f"{kind}_seed{seed}.pt"
    generator = np.random.RandomState(seed)
    for epoch in range(1, settings["epochs"] + 1):
        model.train(); losses = []
        rows = generator.permutation(train_rows)  # pre-event snapshots are frozen, no live mutable history
        for start in range(0, len(rows), settings["batch_size"]):
            batch = rows[start:start + settings["batch_size"]]
            p, q = model(*tensor_batch(stream, batch, data, device))
            if model.lifecycle is None:
                loss = F.binary_cross_entropy_with_logits(p["gated_logits"], torch.ones_like(p["gated_logits"]))
                loss += F.binary_cross_entropy_with_logits(q["gated_logits"], torch.zeros_like(q["gated_logits"]))
            else:
                loss = model.lifecycle.compute_losses(p, torch.ones_like(p["gated_logits"]))["loss"]
                loss += model.lifecycle.compute_losses(q, torch.zeros_like(q["gated_logits"]))["loss"]
                loss += 0.5 * (F.binary_cross_entropy_with_logits(p["base_logits"], torch.ones_like(p["base_logits"])) +
                               F.binary_cross_entropy_with_logits(q["base_logits"], torch.zeros_like(q["base_logits"])))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at {kind} seed {seed} epoch {epoch}")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, val_rows, stream, data, settings, device)
        val_metrics = metrics(validation["positive"], validation["negative"])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=val_metrics))
        if val_metrics["ap"] > best_ap:
            best_ap, best_epoch = val_metrics["ap"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_ap": best_ap,
                        "kind": kind, "seed": seed, "settings": settings,
                        "lifecycle_config": asdict(model.lifecycle.cfg) if model.lifecycle else None}, checkpoint)
        print(f"{kind} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} validation_AP={val_metrics['ap']:.4f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    test_rows = np.flatnonzero(stream.split == 2)
    scored = evaluate(model, test_rows, stream, data, settings, device)
    report = {"kind": kind, "seed": seed, "selected_epoch": best_epoch, "validation_history": history,
              "checkpoint_sha256": sha256(checkpoint), "parameter_count": sum(p.numel() for p in model.parameters()),
              "all_test": metrics(scored["positive"], scored["negative"])}
    for label, mask in (("transductive", ~stream.inductive[test_rows]), ("inductive", stream.inductive[test_rows])):
        report[label] = metrics(scored["positive"][mask], scored["negative"][mask])
    prediction_path = output / f"{kind}_seed{seed}_predictions.npz"
    np.savez_compressed(prediction_path, event_ids=stream.event_ids[test_rows],
                        negative_destinations=stream.negative_destinations[test_rows],
                        positive=scored["positive"], negative=scored["negative"],
                        inductive=stream.inductive[test_rows],
                        **({"z_curr": scored["z"], "z_next": scored["z_next"]} if model.lifecycle else {}))
    report["predictions_sha256"] = sha256(prediction_path)
    # Strata quantify where gains occur; zero/empty cells remain visible.
    recurrence = np.expm1(stream.pair_features[test_rows, 0, 2])
    elapsed = stream.elapsed_times[test_rows, 0]
    report["history_strata"] = {}
    for label, mask in (("never_observed_pair", recurrence == 0), ("recurrent_pair", recurrence > 0),
                        ("silence_ge_1000", (recurrence > 0) & (elapsed >= 1000))):
        report["history_strata"][label] = metrics(scored["positive"][mask], scored["negative"][mask])
    if model.lifecycle:
        z = scored["z"]
        mean_z = z.mean(axis=0)
        report["batch_diversity_entropy"] = float(-(mean_z * np.log(np.maximum(mean_z, 1e-12))).sum())
        report["individual_entropy"] = float(-(z * np.log(np.maximum(z, 1e-12))).sum(axis=1).mean())
        report["min_decay_rate"] = float(model.lifecycle.kernel.decay_rates.min().detach())
        report["max_transition_norm"] = scored["max_transition_norm"]
        report["survival_weights"] = model.lifecycle.survival_gate.survival_weights.detach().cpu().tolist()
        report["phase_semantics"] = []
        for k in range(settings["latent_phases"]):
            mask = z.argmax(axis=1) == k
            report["phase_semantics"].append(dict(phase=k, count=int(mask.sum()),
                mean_dt=float(elapsed[mask].mean()) if mask.any() else None,
                mean_recurrence=float(recurrence[mask].mean()) if mask.any() else None))
        # Fixed seed random sample, with exact denominator and IDs retained.
        subset = np.random.RandomState(seed + 1000).choice(len(z), min(settings["counterfactual_pairs"], len(z)), replace=False)
        with torch.no_grad():
            phase = torch.as_tensor(z[subset], device=device)
            base = torch.as_tensor(scored["base_logits"][subset], device=device)
            cf = model.lifecycle.counterfactual_trajectory(phase, base, settings["silence_grid"])
            report["trajectory_monotonic_rate"] = float(cf["monotonic_rate"])
            report["silence_treatment_effect"] = float(cf["silence_treatment_effect"])
            report["phase_ates"] = [float(model.lifecycle.phase_intervention_ate(base, k)) for k in range(1, settings["latent_phases"])]
            report["counterfactual_count"] = len(subset)
            report["counterfactual_interpretation"] = "model response; ordered_gate has imposed monotonicity"
            np.savez_compressed(output / f"{kind}_seed{seed}_counterfactual.npz",
                                event_ids=stream.event_ids[test_rows][subset],
                                trajectory=cf["prob_trajectory"].cpu().numpy())
    save_json(output / f"{kind}_seed{seed}.json", report)
    return report


def gate_verdicts(report, base, protocol):
    gates = protocol["gates"]
    gain = report["transductive"]["ap"] - base["transductive"]["ap"] if report["transductive"]["ap"] is not None and base["transductive"]["ap"] is not None else None
    verdicts = {
        "latent_phase_diversity": report["batch_diversity_entropy"] >= gates["latent_phase_diversity"]["min_threshold"],
        "continuous_transition_stability": report["min_decay_rate"] >= gates["continuous_transition_stability"]["min_threshold"] and report["max_transition_norm"] <= gates["continuous_transition_stability"]["max_transition_norm"],
        "shortcut_mitigation_gain": gain is not None and gain >= gates["shortcut_mitigation_gain"]["min_threshold"],
        "counterfactual_decay_monotonicity": report["trajectory_monotonic_rate"] >= gates["counterfactual_decay_monotonicity"]["min_threshold"],
    }
    return dict(gain_transductive_ap=gain, verdicts=verdicts, numerical_gates_pass=all(verdicts.values()),
                publication_eligible=False)


def run_benchmark(dataset, settings, protocol, *, data_dir, output_dir, device="cuda", smoke=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh attempt directory; previous attempts must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a heavy CPU fallback")
    dev = torch.device(device)
    data, provenance = load_dataset(dataset, data_dir)
    if smoke:
        data = {k: v[:2000] for k, v in data.items()}
        settings = dict(settings, epochs=min(2, settings["epochs"]))
        provenance["smoke_events"] = len(data["sources"])
    save_json(output / "data_identity.json", provenance)
    result = dict(schema_version=2, protocol_id=protocol["protocol_id"], dataset=dataset, smoke=smoke,
                  settings=settings, data=provenance, source_files_sha256={},
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                   device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
                  per_seed_runs=[], publication_eligible=False)
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "src").rglob("*.py")):
        result["source_files_sha256"][str(path.relative_to(root))] = sha256(path)
    start = time.time()
    for seed in settings["seeds"]:
        stream = build_candidate_stream(**data, seed=seed, split=protocol["split_chronological"],
                    split_seed=settings["split_seed"], unseen_ratio=protocol["inductive_unseen_node_ratio"],
                    max_neighbors=settings["max_neighbors"], history_decay=settings["history_decay"])
        attempt = output / f"seed{seed}"; attempt.mkdir()
        np.savez_compressed(attempt / "candidate_identity.npz", event_ids=stream.event_ids,
                            negative_destinations=stream.negative_destinations, split=stream.split,
                            inductive=stream.inductive, unseen_nodes=stream.unseen_nodes,
                            skipped_no_negative_ids=stream.skipped_no_negative_ids,
                            excluded_training_ids=stream.excluded_training_ids)
        reports = {}
        for kind in settings["models"]:
            reports[kind] = train_model(kind, seed, stream, data, settings, dev, attempt)
        entry = dict(seed=seed, candidate_sha256=stream.digest(), train_end=stream.train_end,
                     validation_end=stream.validation_end,
                     coverage={"train": int((stream.split == 0).sum()), "validation": int((stream.split == 1).sum()),
                               "test": int((stream.split == 2).sum()), "excluded_train": len(stream.excluded_training_ids),
                               "no_negative": len(stream.skipped_no_negative_ids)}, models=reports)
        entry["gates"] = gate_verdicts(reports["ordered_gate"], reports["base"], protocol)
        result["per_seed_runs"].append(entry)
        save_json(output / "partial_results.json", result)
    result["elapsed_seconds"] = time.time() - start
    result["numerical_gates_all_seeds_pass"] = all(r["gates"]["numerical_gates_pass"] for r in result["per_seed_runs"])
    result["complete_declared_seeds"] = settings["seeds"] == protocol["seeds"]
    result["aggregate"] = {}
    for kind in settings["models"]:
        result["aggregate"][kind] = {}
        for split in ("transductive", "inductive"):
            for metric in ("ap", "auc"):
                values = [r["models"][kind][split][metric] for r in result["per_seed_runs"]]
                result["aggregate"][kind][f"{split}_{metric}"] = {
                    "mean": float(np.mean(values)) if all(v is not None for v in values) else None,
                    "std_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None,
                    "seed_count": len(values)}
    save_json(output / "results.json", result)
    checksums = [f"{sha256(p)}  {p.relative_to(output).as_posix()}" for p in sorted(output.rglob("*")) if p.is_file()]
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return result
