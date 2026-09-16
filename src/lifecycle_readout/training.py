"""
Training and evaluation pipeline for SR-GNN and baselines.
Supports: transductive & inductive link prediction on Wikipedia.

Usage:
  python train.py --model srgnn --dataset wikipedia --epochs 50
  python train.py --model all   --dataset wikipedia --epochs 50
"""

import os
import sys
import time
import argparse
import random
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .datasets import load_dataset, get_data_splits, download_dataset
from .metrics import RunningMetrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Device selection: env override SRGNN_DEVICE wins, else auto cuda > mps > cpu.
_dev_env = os.environ.get("SRGNN_DEVICE")
if _dev_env:
    DEVICE = torch.device(_dev_env)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[device] {DEVICE}")


def _dev_sync():
    """Make wall-clock reads trustworthy on CUDA (async kernels) — no-op elsewhere."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────
# Negative sampling
# ─────────────────────────────────────────────────────────────

def sample_negatives(dst: np.ndarray, num_nodes: int, inductive_nodes=None) -> np.ndarray:
    """
    Random negative destination sampling.
    If inductive_nodes given, sample only from unseen nodes.
    """
    if inductive_nodes is not None:
        pool = np.array(inductive_nodes)
    else:
        pool = np.arange(num_nodes)

    neg = np.random.choice(pool, size=len(dst), replace=True)
    # Avoid trivial positives (same as dst)
    mask = neg == dst
    while mask.any():
        neg[mask] = np.random.choice(pool, size=mask.sum(), replace=True)
        mask = neg == dst
    return neg


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def build_model(name: str, num_nodes: int, feat_dim: int, hidden: int):
    from .modeling.sr_gnn import SRGNN
    from .modeling.sr_gnn_v2 import SRGNN_v2
    from .modeling.sr_gnn_v3 import SRGNN_v3

    # Instantiate only the selected model.  The old eager dictionary constructed
    # every variant, so an obsolete variant could break even the baseline path.
    factories = {
        "srgnn": lambda: SRGNN(num_nodes, feat_dim, hidden, device=DEVICE),
        "srgnn_v2": lambda: SRGNN_v2(num_nodes, feat_dim, hidden, device=DEVICE),
        "srgnn_v3": lambda: SRGNN_v3(num_nodes, feat_dim, hidden, device=DEVICE),
    }
    if name not in factories:
        choices = ", ".join(sorted(factories))
        raise ValueError(f"Unknown or unsupported model {name!r}; choose from: {choices}")
    return factories[name]().to(DEVICE)


# ─────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────

def run_epoch(model, split_data, num_nodes, batch_size, optimizer=None,
              inductive_nodes=None, seen_nodes=None, desc="train",
              het_collector=None, score_collector=None,
              neg_strategy="random", hist_neg_ctx=None):
    """
    Run one epoch of training or evaluation.

    Inductive evaluation (TGB-standard fair protocol):
      Filter to edges with at least one inductive endpoint.
      Negatives sampled from SAME pool as pos dst:
        - If pos dst is seen → neg from seen nodes (fair: both have memory)
        - If pos dst is inductive → neg from inductive nodes (fair: both zero)
      This prevents the model from trivially distinguishing by memory magnitude.

    PER-EDGE SCORE DUMP (no semantics change): if `score_collector` is a dict, the
    SAME sigmoid(pos_score)/sigmoid(neg_score) arrays already fed to the AP metric are
    accumulated per batch and, at end-of-epoch, written back as the keys:
        "pos"  (N_eval,) positive scores in chronological eval-row order
        "neg"  (N_eval,) negative scores, row-aligned to pos
        "n"    int, number of evaluated positive rows
    These are EXACTLY the scores used by RunningMetrics (captured at the identical
    point, BEFORE any optimizer/post_step), so re-deriving AP from them reproduces the
    reported value bit-for-bit and introduces NO temporal leak. The caller maps these
    rows to global test indices via test_idx = np.arange(val_end, n) (transductive only;
    for inductive the row set is the filtered ind_mask subset — caller must not assume
    np.arange there).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    src_all = split_data["sources"]
    dst_all = split_data["destinations"]
    t_all   = split_data["timestamps"]
    feat_all = split_data["features"]

    # For inductive: filter to edges with at least one inductive endpoint
    if inductive_nodes is not None:
        ind_set = set(inductive_nodes)
        ind_mask = np.array([
            (int(s) in ind_set) or (int(d) in ind_set)
            for s, d in zip(src_all, dst_all)
        ])
        src_all  = src_all[ind_mask]
        dst_all  = dst_all[ind_mask]
        t_all    = t_all[ind_mask]
        feat_all = feat_all[ind_mask]

    N = len(src_all)
    if N < 10:
        return {"AP": float("nan"), "AUC": float("nan"), "Loss": float("nan")}

    # Build per-dst negative pool for fair inductive eval
    ind_set_for_neg = set(inductive_nodes) if inductive_nodes else None
    seen_set_for_neg = set(seen_nodes) if seen_nodes else None

    metrics = RunningMetrics()
    indices = np.arange(N)   # chronological order (NO shuffle for temporal)
    _pos_dump = [] if score_collector is not None else None
    _neg_dump = [] if score_collector is not None else None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for start in tqdm(range(0, N, batch_size), desc=desc, leave=False, ncols=80):
            idx = indices[start: start + batch_size]
            if len(idx) == 0:
                continue

            src  = torch.tensor(src_all[idx],  dtype=torch.long,  device=DEVICE)
            dst  = torch.tensor(dst_all[idx],  dtype=torch.long,  device=DEVICE)
            t    = torch.tensor(t_all[idx],    dtype=torch.float, device=DEVICE)
            feat = torch.tensor(feat_all[idx], dtype=torch.float, device=DEVICE)

            # ── Negative sampling ────────────────────────────────────────────
            # neg_strategy:
            #   "random"     : fair pool-matched random (seen→seen, ind→ind) — the
            #                  harness default (Table 3 reference numbers).
            #   "historical" : Poursafaei et al. (2022) historical NS — sample neg dst
            #                  from the pool of destinations SEEN in TRAIN edges but NOT
            #                  present at the current eval timestamp (i.e. a node-pair
            #                  that existed historically and is absent now). Hard because
            #                  the negative is a plausible past partner, not a random node.
            #   "inductive"  : Poursafaei inductive NS — neg dst sampled from the pool
            #                  of destinations of TEST-PHASE-ONLY edges, i.e. (src,dst)
            #                  pairs observed during test but NEVER present in train.
            #                  This is about test-phase edges, NOT unseen nodes.
            # hist_neg_ctx (built once per run, passed in) carries:
            #   "hist_dst_pool"     : np.int64[] train destination multiset (historical)
            #   "hist_dst_pool_ind" : np.int64[] destinations of test-phase-only edges
            #   "active_pos_set"    : set of (src,dst) positive pairs at THIS eval split
            #                         (used to reject a sampled neg that is actually a
            #                          current positive — keeps the negative truly absent)
            batch_dst = dst_all[idx]
            batch_src = src_all[idx]
            if neg_strategy in ("historical", "inductive") and hist_neg_ctx is not None:
                if neg_strategy == "inductive":
                    pool = hist_neg_ctx["hist_dst_pool_ind"]
                else:
                    pool = hist_neg_ctx["hist_dst_pool"]
                active_pos = hist_neg_ctx["active_pos_set"]
                neg_dst_np = np.zeros(len(batch_dst), dtype=np.int64)
                for bi in range(len(batch_dst)):
                    s = int(batch_src[bi]); d = int(batch_dst[bi])
                    nd = int(np.random.choice(pool))
                    tries = 0
                    # reject if equal to the true dst OR if (s,nd) is itself a current
                    # positive (would make the "negative" a real present edge).
                    while (nd == d or (s, nd) in active_pos) and tries < 20:
                        nd = int(np.random.choice(pool)); tries += 1
                    neg_dst_np[bi] = nd
            elif ind_set_for_neg is not None and seen_set_for_neg is not None:
                # Fair random neg for inductive eval: same pool as pos dst.
                neg_dst_np = np.zeros(len(batch_dst), dtype=np.int64)
                for bi, d in enumerate(batch_dst):
                    if int(d) in ind_set_for_neg:
                        pool = list(ind_set_for_neg)
                    else:
                        pool = list(seen_set_for_neg)
                    neg_dst_np[bi] = np.random.choice(pool)
                    while neg_dst_np[bi] == d:
                        neg_dst_np[bi] = np.random.choice(pool)
            else:
                neg_dst_np = sample_negatives(dst_all[idx], num_nodes)

            neg_dst = torch.tensor(neg_dst_np, dtype=torch.long, device=DEVICE)

            out = model(src, dst, t, feat, neg_dst)

            if is_train:
                optimizer.zero_grad()
                out["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Invariant I3: Hopfield pass MUST be after optimizer step (anti-leakage)
                if hasattr(model, "post_step"):
                    model.post_step()

            pos_np = torch.sigmoid(out["pos_score"]).detach().cpu().numpy()
            neg_np = torch.sigmoid(out["neg_score"]).detach().cpu().numpy()
            loss_val = out["loss"].item()

            extras = {}
            for k in ("ccs", "salience", "tip_loss", "causal_loss"):
                if k in out:
                    v = out[k]
                    extras[k] = v.item() if hasattr(v, "item") else float(v)

            # ── LOGGING ONLY (evaluation-harness smoke): collect per-batch pair_het_var
            # trajectory for fsm_arch="v3". No model change; default None = no-op.
            if het_collector is not None:
                phv = out.get("pair_het_var", None)
                if phv is not None:
                    het_collector.append(
                        phv.item() if hasattr(phv, "item") else float(phv))

            metrics.update(pos_np, neg_np, loss_val, extras)
            if _pos_dump is not None:
                _pos_dump.append(pos_np)
                _neg_dump.append(neg_np)

    if score_collector is not None and _pos_dump:
        pos_cat = np.concatenate(_pos_dump)
        neg_cat = np.concatenate(_neg_dump)
        score_collector["pos"] = pos_cat
        score_collector["neg"] = neg_cat
        score_collector["n"] = int(len(pos_cat))

    return metrics.compute()


# ─────────────────────────────────────────────────────────────
# Full experiment for one model
# ─────────────────────────────────────────────────────────────

def run_experiment(model_name: str, dataset_name: str,
                   epochs: int, hidden: int, batch_size: int,
                   lr: float, seed: int, dump_dir: str = None):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Load data
    data = download_dataset(dataset_name)
    splits = get_data_splits(data)
    num_nodes = data["num_nodes"]
    feat_dim  = data["feat_dim"]
    # Global test indices for the per-edge dump (mirrors get_data_splits: slice
    # [val_end:n] of the ts-sorted arrays, NO re-sort → chronological test order).
    _n_total = int(data["num_edges"])
    _val_end = int(_n_total * (0.70 + 0.15))
    test_idx_global = np.arange(_val_end, _n_total)

    # Inductive nodes: nodes NOT seen in train OR val (strictly new in test)
    seen_nodes = (set(splits["train"]["sources"]) | set(splits["train"]["destinations"])
                  | set(splits["val"]["sources"]) | set(splits["val"]["destinations"]))
    test_nodes = set(splits["test"]["sources"]) | set(splits["test"]["destinations"])
    inductive_nodes = sorted(test_nodes - seen_nodes)
    if len(inductive_nodes) < 10:
        inductive_nodes = None  # not enough inductive nodes for meaningful eval

    model = build_model(model_name, num_nodes, feat_dim, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ap = 0.0
    best_results = {}
    history = []

    print(f"\n{'─'*60}")
    print(f"  Model: {model_name.upper():<15}  Dataset: {dataset_name}")
    print(f"  Nodes: {num_nodes}  Edges: {data['num_edges']}  Feat: {feat_dim}")
    print(f"  Hidden: {hidden}  Batch: {batch_size}  LR: {lr}  Seed: {seed}")
    print(f"{'─'*60}")

    # ── Untimed warmup: one full train pass to absorb cudnn autotune + lazy
    #    CUDA allocation so the timed region reflects steady-state compute. ──
    if hasattr(model, "reset"):
        model.reset()
    if hasattr(model, "set_epoch"):
        model.set_epoch(0)
    run_epoch(model, splits["train"], num_nodes, batch_size,
              optimizer=optimizer, desc="warmup(untimed)")
    _dev_sync()
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        if hasattr(model, "reset"):
            model.reset()

        train_m = run_epoch(model, splits["train"], num_nodes, batch_size,
                            optimizer=optimizer, desc=f"E{epoch:02d}/train")
        val_m   = run_epoch(model, splits["val"],   num_nodes, batch_size,
                            desc=f"E{epoch:02d}/val")

        scheduler.step()

        log = {
            "epoch": epoch,
            "train_ap": train_m["AP"], "train_auc": train_m["AUC"], "train_loss": train_m["Loss"],
            "val_ap":   val_m["AP"],   "val_auc":   val_m["AUC"],   "val_loss":   val_m["Loss"],
        }
        history.append(log)

        if val_m["AP"] > best_val_ap:
            best_val_ap = val_m["AP"]
            # save best model state for final test
            best_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            _dev_sync()
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:02d}/{epochs}  "
                  f"Train AP={train_m['AP']:.4f} AUC={train_m['AUC']:.4f}  "
                  f"Val AP={val_m['AP']:.4f} AUC={val_m['AUC']:.4f}  "
                  f"[{elapsed:.0f}s]")

    # Final test — transductive (capture per-edge scores for the post-CP dump)
    if hasattr(model, "reset"): model.reset()
    model.load_state_dict(best_state)
    trans_scores = {}
    test_trans = run_epoch(model, splits["test"], num_nodes, batch_size,
                           desc="test_trans", score_collector=trans_scores)

    # Final test — inductive (if applicable)
    test_ind = {"AP": float("nan"), "AUC": float("nan")}
    if inductive_nodes:
        if hasattr(model, "reset"): model.reset()
        model.load_state_dict(best_state)
        # Run train+val first to build memory, then evaluate on inductive test edges
        run_epoch(model, splits["train"], num_nodes, batch_size, desc="ind_warmup_train")
        run_epoch(model, splits["val"],   num_nodes, batch_size, desc="ind_warmup_val")
        test_ind = run_epoch(model, splits["test"], num_nodes, batch_size,
                             inductive_nodes=inductive_nodes,
                             seen_nodes=sorted(seen_nodes),
                             desc="test_ind")

    _dev_sync()
    total_time = time.time() - t_start
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── PER-EDGE DUMP (pack post-CP eval INTO the run; same scores as the metric) ──
    postcp = None
    npz_path = None
    if dump_dir is not None and trans_scores.get("pos") is not None:
        os.makedirs(dump_dir, exist_ok=True)
        pos = np.asarray(trans_scores["pos"], dtype=np.float64)
        neg = np.asarray(trans_scores["neg"], dtype=np.float64)
        n_pos = len(pos)
        if n_pos != len(test_idx_global):
            print(f"  [dump][WARN] n_pos={n_pos} != len(test_idx)={len(test_idx_global)}; "
                  f"saving raw arrays only, NO test_idx map.")
            ti = np.full(n_pos, -1, dtype=np.int64)
        else:
            ti = test_idx_global.astype(np.int64)
        y_true = np.concatenate([np.ones(n_pos), np.zeros(len(neg))]).astype(np.int8)
        y_score = np.concatenate([pos, neg]).astype(np.float64)
        npz_path = os.path.join(
            dump_dir, f"peredge_{model_name}_{dataset_name}_seed{seed}.npz")
        np.savez_compressed(npz_path, y_true=y_true, y_score=y_score,
                            pos_score=pos, neg_score=neg,
                            test_idx=ti, n_pos=np.int64(n_pos))
        print(f"  [dump] per-edge scores -> {npz_path} (n_pos={n_pos}, n_neg={len(neg)})")
        if dataset_name == "synthetic_regime" and ti[0] != -1:
            try:
                from data.regime_postcp_eval import (
                    load_test_anomaly_flag, load_test_phase,
                    load_test_relationship_id, postcp_window_mask,
                    postcp_window_mask_perpair, ap_on_pos_subset)
                flag = load_test_anomaly_flag(ti)
                phase = load_test_phase(ti)
                rid = load_test_relationship_id(ti)
                postcp = {"overall_ap": test_trans["AP"], "n_pos": int(n_pos)}
                postcp["cp_w0_ap"] = ap_on_pos_subset(
                    y_true, y_score, postcp_window_mask(flag, window=0))
                for w in (1, 2, 5):
                    postcp[f"perpair_w{w}_ap"] = ap_on_pos_subset(
                        y_true, y_score, postcp_window_mask_perpair(flag, rid, window=w))
                postcp["high_phase1_ap"] = ap_on_pos_subset(y_true, y_score, (phase == 1))
                print(f"  [postcp] {model_name} s{seed}: cp_w0={postcp['cp_w0_ap']:.4f} "
                      f"pp_w1={postcp['perpair_w1_ap']:.4f} pp_w2={postcp['perpair_w2_ap']:.4f} "
                      f"pp_w5={postcp['perpair_w5_ap']:.4f} hi={postcp['high_phase1_ap']:.4f}")
            except Exception as e:
                print(f"  [postcp][WARN] inline post-CP failed ({e}); .npz saved.")
                import traceback; traceback.print_exc()
                postcp = None

    results = {
        "model":          model_name,
        "dataset":        dataset_name,
        "seed":           seed,
        "trans_ap":       test_trans["AP"],
        "trans_auc":      test_trans["AUC"],
        "ind_ap":         test_ind["AP"],
        "ind_auc":        test_ind["AUC"],
        "best_val_ap":    best_val_ap,
        "train_time_s":   total_time,
        "num_params":     num_params,
        "history":        history,
        "postcp":         postcp,
        "peredge_npz":    npz_path,
    }

    # Include SR-GNN-specific metrics if available
    for k in ("ccs", "salience"):
        if k in test_trans:
            results[k] = test_trans[k]

    print(f"\n  ✓ FINAL  Trans AP={test_trans['AP']:.4f} AUC={test_trans['AUC']:.4f} | "
          f"Ind AP={test_ind['AP']:.4f}  [{total_time:.0f}s  params={num_params:,}]")

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

ALL_MODELS = [
    "srgnn", "srgnn_v2", "srgnn_v3",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="srgnn",
                        help=f"Model name or 'all'. Choices: {ALL_MODELS}")
    parser.add_argument("--dataset",   default="coedit", choices=["coedit"])
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--hidden",    type=int, default=128)
    parser.add_argument("--batch",     type=int, default=500)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "audit" / "training.json")
    )
    args = parser.parse_args()

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    all_results = []
    failures = []
    for model_name in models_to_run:
        try:
            r = run_experiment(
                model_name, args.dataset,
                epochs=args.epochs, hidden=args.hidden,
                batch_size=args.batch, lr=args.lr, seed=args.seed
            )
            all_results.append(r)
        except Exception as e:
            print(f"  [!] {model_name} failed: {e}")
            failures.append({"model": model_name, "error": str(e)})
            import traceback; traceback.print_exc()

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runs": all_results, "failures": failures}, f, indent=2, default=str)
    print(f"\n[saved] {args.out}")

    # Print summary table
    print("\n" + "="*78)
    print(f"{'Model':<18} {'Trans AP':>9} {'Trans AUC':>10} {'Ind AP':>8} {'Ind AUC':>9} {'Params':>10}")
    print("─"*78)
    for r in all_results:
        print(f"{r['model']:<18} "
              f"{r['trans_ap']:>9.4f} "
              f"{r['trans_auc']:>10.4f} "
              f"{r['ind_ap']:>8.4f} "
              f"{r['ind_auc']:>9.4f} "
              f"{r['num_params']:>10,}")
    print("="*78)
    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed; see {args.out}")
