"""Prospective feature-matched controls on common recurrent hard negatives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import random
import time
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import load_dataset, metrics, save_json, sha256, tensor_batch
from .causal_stream import build_candidate_stream
from .modeling.latent_lifecycle import LatentLifecycleConfig, LatentLifecycleModule
from .modeling.temporal_mixer import BaseTemporalPredictor
from .modeling.monotone_residual import NestedMonotoneResidual


class HistoryControl(nn.Module):
    """All non-base controls receive the same explicit historical features.

    Every gated model shares the base_stats affinity architecture. The scalar
    control learns a global exponential multiplier; the MLP learns a flexible
    gate from the same pair representation and history. No dormant routing is
    needed here: both candidates must have prior history under this protocol.
    """
    def __init__(self, kind, feat_dim, settings, phi_mean, phi_std):
        super().__init__()
        if kind not in ("base", "base_stats", "scalar_decay", "mlp_gate", "ordered_gate", "residual_scalar", "residual_mixture", "mlp_residual"):
            raise ValueError(kind)
        self.kind = kind
        self.time_unit = float(settings["decay_time_unit"])
        self.base = BaseTemporalPredictor(feat_dim=feat_dim, hidden_dim=settings["hidden_dim"],
                                         time_dim=settings["time_dim"], max_neighbors=settings["max_neighbors"])
        self.register_buffer("phi_mean", torch.as_tensor(phi_mean, dtype=torch.float32))
        self.register_buffer("phi_std", torch.as_tensor(phi_std, dtype=torch.float32))
        hidden = settings["hidden_dim"]
        self.phi_dim = int(settings.get("pair_feature_dim", 6))
        self.pair_clock_mode = settings.get("pair_clock_mode", "absolute")
        if self.pair_clock_mode not in ("absolute", "relative"):
            raise ValueError("Unknown pair clock mode")
        if self.pair_clock_mode == "relative" and self.phi_dim != 9:
            raise ValueError("Relative clock requires the nine-feature cadence stream")
        self.history_head = None
        self.gate = None
        self.lifecycle = None
        self.residual = None
        if kind != "base":
            self.history_head = nn.Sequential(nn.Linear(hidden + self.phi_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        if kind == "scalar_decay":
            rate = float(settings.get("initial_decay_rate", .1))
            self.raw_rate = nn.Parameter(torch.tensor(float(np.log(np.expm1(rate)))))
        if kind in ("mlp_gate", "mlp_residual"):
            self.gate = nn.Sequential(nn.Linear(hidden + self.phi_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        if kind == "mlp_residual":
            nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        if kind in ("residual_scalar", "residual_mixture"):
            self.residual = NestedMonotoneResidual(hidden + self.phi_dim,
                n_components=1 if kind == "residual_scalar" else settings["residual_components"],
                hidden_dim=hidden // 2, rate_init=settings.get("initial_decay_rate", 1.))
        if kind == "ordered_gate":
            self.lifecycle = LatentLifecycleModule(LatentLifecycleConfig(
                n_phases=settings["latent_phases"], hidden_dim=hidden, phi_dim=self.phi_dim,
                decay_init=settings.get("initial_decay_rate", .1)))

    def forward(self, x, dt, mask, phi, pair_dt):
        outputs = []
        for dest, candidate in ((1, 0), (2, 1)):
            h, logits = self.base(x[:, 0], dt[:, 0], mask[:, 0], x[:, dest], dt[:, dest], mask[:, dest])
            normalized = (phi[:, candidate] - self.phi_mean) / self.phi_std
            clock = (phi[:, candidate, 8] if self.pair_clock_mode == "relative"
                     else pair_dt[:, candidate] / self.time_unit)
            inputs = torch.cat((h, normalized), dim=-1)
            if self.history_head is not None:
                logits = logits + self.history_head(inputs).squeeze(-1)
            prob = torch.sigmoid(logits)
            output = {"base_logits": logits}
            if self.residual is not None or self.kind == "mlp_residual":
                if self.residual is not None:
                    correction = self.residual(logits, inputs, clock)
                    output.update(correction)
                else:
                    output["logits"] = logits + self.gate(inputs).squeeze(-1)
                    output["prob"] = torch.sigmoid(output["logits"])
                outputs.append(output)
                continue
            if self.kind == "scalar_decay":
                # Preserve the original absolute-clock operation order.
                attenuation = (-F.softplus(self.raw_rate) * clock if self.pair_clock_mode == "relative"
                               else -F.softplus(self.raw_rate) * pair_dt[:, candidate] / self.time_unit)
                prob = prob * torch.exp(attenuation)
            elif self.kind == "mlp_gate":
                prob = prob * torch.sigmoid(self.gate(inputs).squeeze(-1))
            elif self.lifecycle is not None:
                output.update(self.lifecycle(h, normalized, clock, logits))
                prob = output["gated_prob"]
            prob = prob.clamp(1e-6, 1. - 1e-6)
            output.update(prob=prob, logits=torch.logit(prob))
            outputs.append(output)
        return outputs


def score(model, rows, stream, data, settings, device):
    model.eval()
    pos, neg = [], []
    with torch.no_grad():
        for start in range(0, len(rows), settings["batch_size"]):
            p, q = model(*tensor_batch(stream, rows[start:start + settings["batch_size"]], data, device))
            pos.append(p["prob"].cpu().numpy()); neg.append(q["prob"].cpu().numpy())
    if not pos:
        raise ValueError("No matched candidates in evaluation split")
    return np.concatenate(pos), np.concatenate(neg)


def report_metrics(pos, neg):
    result = metrics(pos, neg)
    result["brier"] = float(np.mean(np.r_[(1. - pos) ** 2, neg ** 2])) if len(pos) else None
    return result


def train_control(kind, seed, stream, data, settings, device, output):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train = np.flatnonzero(stream.split == 0)
    val = np.flatnonzero(stream.split == 1)
    test = np.flatnonzero(stream.split == 2)
    if any(len(a) == 0 for a in (train, val, test)):
        raise ValueError("Hard matching leaves an empty split; no fallback is allowed")
    # Fitted only on training candidates, using positive and negative equally.
    phi_train = stream.pair_features[train].reshape(-1, int(settings.get("pair_feature_dim", 6)))
    mean, std = phi_train.mean(0), phi_train.std(0).clip(min=1e-5)
    settings = dict(settings)
    if settings["decay_time_unit"] == "train_median":
        times = stream.elapsed_times[train].reshape(-1)
        settings["decay_time_unit"] = float(np.median(times[times > 0]))
        if not np.isfinite(settings["decay_time_unit"]) or settings["decay_time_unit"] <= 0:
            raise ValueError("No positive finite training clock scale")
    evaluation_split = settings.get("evaluation_split", "test")
    if evaluation_split not in ("test", "validation"):
        raise ValueError("Unknown evaluation split")
    evaluation_rows = test if evaluation_split == "test" else val
    model = HistoryControl(kind, data["features"].shape[1], settings, mean, std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    order = np.random.RandomState(seed)
    best_ap, best_epoch, history = -1., None, []
    checkpoint = output / f"{kind}_seed{seed}.pt"
    started = time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        model.train(); losses = []
        rows = order.permutation(train)
        for start in range(0, len(rows), settings["batch_size"]):
            p, q = model(*tensor_batch(stream, rows[start:start + settings["batch_size"]], data, device))
            loss = torch.zeros((), device=device)
            for pred, target in ((p, 1.), (q, 0.)):
                truth = torch.full_like(pred["logits"], target)
                # Same task + affinity auxiliary objective for every variant.
                loss = loss + F.binary_cross_entropy_with_logits(pred["logits"], truth)
                loss = loss + .5 * F.binary_cross_entropy_with_logits(pred["base_logits"], truth)
                if model.lifecycle is not None:
                    loss = loss + model.lifecycle.compute_losses(pred, truth)["reg_loss"]
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training objective: {kind} {seed} {epoch}")
            opt.zero_grad(); loss.backward(); opt.step()
            if model.residual is not None:
                model.residual.project_parameters_()
            losses.append(float(loss.detach()))
        p, q = score(model, val, stream, data, settings, device)
        val_report = report_metrics(p, q)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=val_report))
        if val_report["ap"] > best_ap:
            best_ap, best_epoch = val_report["ap"], epoch
            torch.save({"model": model.state_dict(), "settings": settings, "seed": seed, "kind": kind,
                        "selected_epoch": epoch, "phi_mean": mean.tolist(), "phi_std": std.tolist()}, checkpoint)
        print(f"{kind} seed={seed} epoch={epoch} validation_AP={val_report['ap']:.6f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    positive, negative = score(model, evaluation_rows, stream, data, settings, device)
    prediction_file = output / f"{kind}_seed{seed}_predictions.npz"
    np.savez_compressed(prediction_file, positive=positive, negative=negative,
                        event_ids=stream.event_ids[evaluation_rows], negative_destinations=stream.negative_destinations[evaluation_rows],
                        pair_features=stream.pair_features[evaluation_rows], elapsed_times=stream.elapsed_times[evaluation_rows],
                        inductive=stream.inductive[evaluation_rows])
    report = dict(kind=kind, seed=seed, selected_epoch=best_epoch, validation_history=history,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  checkpoint_sha256=sha256(checkpoint), predictions_sha256=sha256(prediction_file),
                  elapsed_seconds=time.monotonic() - started, strata={},
                  evaluation_split=evaluation_split,
                  decay_clock_mode=settings.get("pair_clock_mode", "absolute"),
                  effective_decay_time_unit=(settings["decay_time_unit"]
                      if settings.get("pair_clock_mode", "absolute") == "absolute" else None))
    report["all_test" if evaluation_split == "test" else "all_validation"] = report_metrics(positive, negative)
    both_silent = (stream.elapsed_times[evaluation_rows] >= 1000.).all(axis=1)
    for name, selected in (("both_silence_ge_1000", both_silent), ("transductive", ~stream.inductive[evaluation_rows]),
                           ("inductive", stream.inductive[evaluation_rows])):
        report["strata"][name] = report_metrics(positive[selected], negative[selected])
    save_json(output / f"{kind}_seed{seed}.json", report)
    return report


def run_controls(dataset, settings, protocol, *, data_dir, output_dir, device="cuda", smoke=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device; no heavy local fallback")
    data, identity = load_dataset(dataset, data_dir)
    if smoke:
        data = {k: v[:2000] for k, v in data.items()}
        settings = dict(settings, epochs=min(2, settings["epochs"]))
    evaluation_split = settings.get("evaluation_split", "test")
    endpoint_key = "all_test" if evaluation_split == "test" else "all_validation"
    result = dict(protocol_id=protocol["protocol_id"], dataset=dataset, settings=settings,
                  evaluation_split=evaluation_split, test_scored=evaluation_split == "test",
                  data=identity, smoke=smoke, publication_eligible=False, per_seed_runs=[],
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   numpy=np.__version__, device=device,
                                   gpu=torch.cuda.get_device_name(0) if device == "cuda" else None))
    root = Path(__file__).resolve().parents[2]
    result["source_files_sha256"] = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))}
    for seed in settings["seeds"]:
        stream = build_candidate_stream(**data, seed=seed, split=protocol["split_chronological"],
                  split_seed=settings["split_seed"], unseen_ratio=protocol["inductive_unseen_node_ratio"],
                  max_neighbors=settings["max_neighbors"], history_decay=settings["history_decay"],
                  negative_sampling="matched_historical", hard_time_unit=settings["hard_time_unit"])
        attempt = output / f"seed{seed}"; attempt.mkdir()
        if settings.get("pair_feature_dim", 6) == 9:
            if evaluation_split != "validation":
                raise ValueError("Cadence development must not score test")
            from .pair_clock import augment_pair_clock
            stream, clock_metadata, clock_arrays = augment_pair_clock(
                data, stream, tau0_seconds=settings["pair_clock_tau0_seconds"])
            if settings.get("mask_pair_clock_features", False):
                if settings.get("pair_clock_mode", "absolute") != "absolute":
                    raise ValueError("Masked cadence control must use the absolute clock")
                masked = stream.pair_features.copy()
                masked[:, :, 6:] = 0
                stream = replace(stream, pair_features=masked)
            clock_metadata['features_masked_for_capacity_control'] = settings.get("mask_pair_clock_features", False)
            save_json(attempt / "pair_clock.json", clock_metadata)
            np.savez_compressed(attempt / "pair_clock.npz", **clock_arrays)
        np.savez_compressed(attempt / "candidate_identity.npz", event_ids=stream.event_ids,
            negative_destinations=stream.negative_destinations, pair_features=stream.pair_features,
            elapsed_times=stream.elapsed_times, split=stream.split, inductive=stream.inductive,
            unseen_nodes=stream.unseen_nodes, excluded_training_ids=stream.excluded_training_ids,
            skipped_no_hard_support_ids=stream.skipped_no_negative_ids)
        counts = np.expm1(stream.pair_features[:, :, 2])
        if not (counts > 0).all():
            raise AssertionError("Hard candidates must both have prior history")
        test = stream.split == 2
        entry = dict(seed=seed, candidate_sha256=stream.digest(),
                     coverage=dict(train=int((stream.split == 0).sum()), validation=int((stream.split == 1).sum()),
                        test=int(test.sum()), skipped_no_hard_support=int(len(stream.skipped_no_negative_ids)),
                        excluded_train=int(len(stream.excluded_training_ids)),
                        total_events=len(data["sources"])), models={})
        # Feature-only recurrence diagnostic on the exact same candidate rows.
        diagnostic_rows = test if evaluation_split == "test" else stream.split == 1
        entry["recurrence_only_" + evaluation_split] = metrics(counts[diagnostic_rows, 0], counts[diagnostic_rows, 1])
        for kind in settings["models"]:
            entry["models"][kind] = train_control(kind, seed, stream, data, settings, torch.device(device), attempt)
            save_json(output / "partial_results.json", dict(result, current_seed=entry))
        result["per_seed_runs"].append(entry)
        save_json(output / "partial_results.json", result)
    result["complete_declared_seeds"] = settings["seeds"] == protocol["seeds"]
    result["aggregate"] = {}
    for kind in settings["models"]:
        result["aggregate"][kind] = {}
        for endpoint in ("ap", "auc", "brier"):
            values = [r["models"][kind][endpoint_key][endpoint] for r in result["per_seed_runs"]]
            result["aggregate"][kind][endpoint] = dict(mean=float(np.mean(values)),
                std_ddof1=float(np.std(values, ddof=1)) if len(values) > 1 else None, seed_count=len(values))
    result["paired_ap_differences"] = {}
    for comparator in settings["models"]:
        if "ordered_gate" in settings["models"] and comparator != "ordered_gate":
            delta = [r["models"]["ordered_gate"][endpoint_key]["ap"] - r["models"][comparator][endpoint_key]["ap"]
                     for r in result["per_seed_runs"]]
            result["paired_ap_differences"][comparator] = dict(values=delta, mean=float(np.mean(delta)),
                std_ddof1=float(np.std(delta, ddof=1)) if len(delta) > 1 else None)
    save_json(output / "results.json", result)
    (output / "checksums.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}"
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.sha256") + "\n")
    return result
