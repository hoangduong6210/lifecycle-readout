"""Validation-only exploratory competition study; existing studies are untouched."""
from dataclasses import replace
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import load_dataset, save_json, sha256, tensor_batch
from .causal_stream import build_candidate_stream
from .competing_edges import competing_exposure, competition_stream, oracle_exposure
from .hard_controls import HistoryControl, report_metrics, score


class CompetitionControl(HistoryControl):
    """Same parameter count and initialization for all feature and mechanism arms.

    Flexible residual: a*g(h,phi). Inhibition: -max(a,0)*softplus(g)*exposure.
    Both start at exact baseline recovery (a=0). The inhibition certificate is
    partial in exposure at fixed context, not an end-to-end chronological claim.
    """
    def __init__(self, kind, feat_dim, settings, phi_mean, phi_std):
        super().__init__("base_stats", feat_dim, settings, phi_mean, phi_std)
        self.kind = kind
        hidden = settings["hidden_dim"]
        self.gate = nn.Sequential(nn.Linear(hidden+self.phi_dim, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.amplitude = nn.Parameter(torch.zeros(()))
        self.inhibitory = kind.endswith("displacement")

    def forward(self, x, dt, mask, phi, pair_dt):
        outputs = []
        for dest, candidate in ((1, 0), (2, 1)):
            h, logit = self.base(x[:, 0], dt[:, 0], mask[:, 0], x[:, dest], dt[:, dest], mask[:, dest])
            inputs = torch.cat((h, (phi[:, candidate]-self.phi_mean)/self.phi_std), -1)
            base = logit+self.history_head(inputs).squeeze(-1)
            gate = self.gate(inputs).squeeze(-1)
            if self.inhibitory:
                index = 6 if self.kind == "activity_displacement" else 7
                residual = -self.amplitude.clamp(min=0)*F.softplus(gate)*phi[:, candidate, index]
            else:
                residual = self.amplitude*gate
            logits = base+residual
            outputs.append(dict(base_logits=base, logits=logits, prob=torch.sigmoid(logits), residual=residual))
        return outputs

    @torch.no_grad()
    def project_parameters_(self):
        if self.inhibitory:
            self.amplitude.clamp_(min=0)


def train_competition(kind, seed, stream, data, settings, device, output):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    training, validation = (np.flatnonzero(stream.split == phase) for phase in (0, 1))
    if not len(training) or not len(validation):
        raise ValueError("No supported training or validation rows; no fallback")
    values = stream.pair_features[training].reshape(-1, settings["pair_feature_dim"])
    mean, std = values.mean(0), values.std(0).clip(min=1e-5)
    model = CompetitionControl(kind, data["features"].shape[1], settings, mean, std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = np.random.RandomState(seed)
    checkpoint = output/f"{kind}_seed{seed}.pt"
    history, best = [], -float("inf")
    started = time.monotonic()
    for epoch in range(1, settings["epochs"]+1):
        model.train(); losses = []
        shuffled_rows = rng.permutation(training)
        for start in range(0, len(training), settings["batch_size"]):
            predictions = model(*tensor_batch(stream, shuffled_rows[start:start+settings["batch_size"]], data, device))
            loss = torch.zeros((), device=device)
            for prediction, target in zip(predictions, (1., 0.)):
                truth = torch.full_like(prediction["logits"], target)
                loss = loss+F.binary_cross_entropy_with_logits(prediction["logits"], truth)
                loss = loss+settings["affinity_auxiliary_weight"]*F.binary_cross_entropy_with_logits(prediction["base_logits"], truth)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite competition objective")
            optimizer.zero_grad(); loss.backward(); optimizer.step(); model.project_parameters_()
            losses.append(float(loss.detach()))
        positive, negative = score(model, validation, stream, data, settings, device)
        metrics = report_metrics(positive, negative)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics))
        if metrics["ap"] > best:
            best = metrics["ap"]
            torch.save(dict(model=model.state_dict(), selected_epoch=epoch, kind=kind, seed=seed,
                            settings=settings, phi_mean=mean.tolist(), phi_std=std.tolist()), checkpoint)
        print(f"competition {kind} seed={seed} epoch={epoch} validation_AP={metrics['ap']:.6f}", flush=True)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    positive, negative = score(model, validation, stream, data, settings, device)
    prediction_file = output/f"{kind}_seed{seed}_predictions.npz"
    np.savez_compressed(prediction_file, positive=positive, negative=negative,
                        event_ids=stream.event_ids[validation], negative_destinations=stream.negative_destinations[validation])
    validation_metrics = report_metrics(positive, negative)
    validation_metrics["pair_accuracy"] = float(np.mean((positive > negative)+.5*(positive == negative)))
    report = dict(kind=kind, seed=seed, selected_epoch=saved["selected_epoch"], validation_history=history,
                  all_validation=validation_metrics, evaluation_split="validation", test_scored=False,
                  parameter_count=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.monotonic()-started,
                  amplitude=float(model.amplitude.detach()), checkpoint_sha256=sha256(checkpoint),
                  predictions_sha256=sha256(prediction_file))
    save_json(output/f"{kind}_seed{seed}.json", report)
    return report


def run_competition(dataset, settings, protocol, *, data_dir, output_dir, device="cuda", smoke=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh attempt required")
    output.mkdir(parents=True, exist_ok=True)
    if device != "cuda" and not smoke:
        raise ValueError("Real-data fitting requires Colab CUDA")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA; no local training fallback")
    data, identity = load_dataset(dataset, data_dir)
    if smoke:
        if len(data["sources"]) > settings["cpu_smoke_event_limit"]:
            raise ValueError("CPU smoke exceeds its synthetic event guard")
    result = dict(protocol_id=protocol["protocol_id"], dataset=dataset, settings=settings, data=identity,
                  evaluation_split="validation", test_scored=False, publication_eligible=False, smoke=smoke,
                  per_seed_runs=[], environment=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, device=device))
    for seed in settings["seeds"]:
        stream = build_candidate_stream(**data, seed=seed, split=protocol["split_chronological"],
            split_seed=settings["split_seed"], unseen_ratio=protocol["inductive_unseen_node_ratio"],
            max_neighbors=settings["max_neighbors"], history_decay=settings["history_decay"],
            negative_sampling="matched_historical", hard_time_unit=settings["hard_time_unit"])
        folder = output/f"seed{seed}"; folder.mkdir()
        arrays = competing_exposure(data, stream, dimensions=settings["profile_dimensions"], profile_seed=settings["profile_seed"])
        anchors = []
        for phase in (0, 1):
            rows = np.flatnonzero(stream.split == phase)
            if not len(rows):
                raise ValueError("Empty development split")
            anchors.extend(rows[np.linspace(0, len(rows)-1, min(settings["oracle_anchors_per_split"], len(rows)), dtype=int)].tolist())
        independent, independent_shuffle = oracle_exposure(data, stream, anchors,
            dimensions=settings["profile_dimensions"], profile_seed=settings["profile_seed"])
        np.testing.assert_allclose(arrays["raw"][anchors], independent, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(arrays["shuffled_exposure"][anchors], independent_shuffle, rtol=1e-8, atol=1e-8)
        save_json(folder/"feature_anchor_audit.json", dict(status="PASS", rows=anchors,
            raw_max_absolute_error=float(np.max(np.abs(arrays["raw"][anchors]-independent))),
            shuffle_max_absolute_error=float(np.max(np.abs(arrays["shuffled_exposure"][anchors]-independent_shuffle))),
            scope="Independent direct raw-history anchors; not full-record feature replay"))
        np.savez_compressed(folder/"competition_features.npz", **arrays)
        np.savez_compressed(folder/"candidate_identity.npz", event_ids=stream.event_ids,
            negative_destinations=stream.negative_destinations, split=stream.split, pair_features=stream.pair_features,
            elapsed_times=stream.elapsed_times, inductive=stream.inductive, excluded_training_ids=stream.excluded_training_ids,
            skipped_no_hard_support_ids=stream.skipped_no_negative_ids, unseen_nodes=stream.unseen_nodes)
        entry = dict(seed=seed, candidate_sha256=stream.digest(), models={},
            coverage={name:int((stream.split == phase).sum()) for phase,name in enumerate(("train","validation","structural_test"))},
            feature_anchor_audit="PASS")
        for arm in settings["models"]:
            arm_stream = competition_stream(stream, arrays, arm)
            entry["models"][arm] = train_competition(arm, seed, arm_stream, data, settings, torch.device(device), folder)
            save_json(output/"partial_results.json", dict(result, current_seed=entry))
        result["per_seed_runs"].append(entry)
        save_json(output/"partial_results.json", result)
    result["complete_declared_seeds"] = settings["seeds"] == protocol["seeds"]
    comparisons = {"primary_mechanism":"profile_displacement profile_mlp",
                   "profile_information":"profile_mlp activity_mlp", "profile_alignment":"profile_mlp shuffled_mlp",
                   "activity_information":"activity_mlp masked_mlp", "profile_vs_activity_hazard":"profile_displacement activity_displacement"}
    result["paired_validation_ap"] = {}
    for name, arms in comparisons.items():
        treatment, control = arms.split()
        delta = [entry["models"][treatment]["all_validation"]["ap"]-entry["models"][control]["all_validation"]["ap"]
                 for entry in result["per_seed_runs"]]
        result["paired_validation_ap"][name] = dict(treatment=treatment, control=control, values=delta,
            mean=float(np.mean(delta)), std_ddof1=float(np.std(delta, ddof=1)) if len(delta)>1 else None)
    save_json(output/"results.json", result)
    return result
