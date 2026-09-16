"""
Full benchmark: RS-GNN v3.3 — 3 seeds × 20 epochs × 3 datasets.

Reports both Transductive and Inductive metrics, comparable with v3.1 lean
benchmark in Master_ML.md.

Output: LAB/v3_3/results/v3_3_benchmark.json
"""
import os, sys, time, json, random, argparse
from pathlib import Path
import numpy as np
import torch

from lifecycle_readout.datasets import download_dataset, get_data_splits
from lifecycle_readout.training import run_epoch, DEVICE, _dev_sync
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = str(PROJECT_ROOT / "experiments")


def measure(model):
    if not model.edge_mem._state_table:
        return {"n_edges": 0}
    states = torch.stack(model.edge_mem._state_table)
    # fsm_arch="v2": the de-collapse REDESIGN measures the SEPARATE soft-masked FSM
    # head s_{t+1} (persisted per pair via edge_mem.update_symbolic), NOT the pinned
    # ECTGv3 hard-masked continuous chain states[:,:5]. The soft head has a FINITE
    # sigmoid(prior+delta) penalty so supervision can move its argmax distribution
    # (CPU-proven movable); the hard-masked chain is structurally pinned at BIRTH.
    het = None
    if (getattr(model, "fsm_arch", "v1") in ("v2", "v3")
            and getattr(model.edge_mem, "_sym_table", None)):
        sym = torch.stack(list(model.edge_mem._sym_table.values()))   # (P,5)
        counts = torch.bincount(sym.argmax(-1), minlength=5).float()
        # ── HETEROGENEITY diagnostic (fsm_arch="v3", Part D) ──────────────────────
        # PM's real goal is NOT just high H but that DIFFERENT pairs flip DIFFERENTLY.
        # Two complementary measures over the per-pair next-state distributions sym:
        #   het_argmax_entropy = entropy of the argmax-state histogram across pairs
        #     (same as the reported H but over pairs, not events).
        #   het_pair_var = mean across the 5 states of the cross-pair VARIANCE of the
        #     per-pair next-state probability. >0 ⇔ pairs genuinely differ; ==0 ⇔ all
        #     pairs share one flip distribution (the v2 over-correction failure mode).
        if getattr(model, "fsm_arch", "v1") == "v3":
            het_pair_var = float(sym.var(dim=0, unbiased=False).mean())
            p = counts / counts.sum().clamp(min=1)
            het_argmax_entropy = float(-(p * (p + 1e-12).log()).sum())
            het = {"pair_state_var": het_pair_var,
                   "argmax_entropy": het_argmax_entropy,
                   "n_pairs": int(sym.size(0))}
    else:
        counts = torch.bincount(states[:, :5].argmax(-1), minlength=5).float()
    dist = (counts / counts.sum()).tolist()
    ev = model.ever_alive
    if bool(ev.registered.any()):
        ever_alive_mean = float(ev.values[ev.registered].mean())
    else:
        ever_alive_mean = 0.0
    out = {
        "n_edges":         int(states.size(0)),
        "edge_state_dist": dist,
        "ever_alive_mean": ever_alive_mean,
        "hawkes_lam_mean": float(states[:, 6].mean()),
    }
    if het is not None:
        out["heterogeneity"] = het
    return out


def run_epoch_v33(model, split_data, num_nodes, batch_size, optimizer=None,
                  inductive_nodes=None, seen_nodes=None, desc="train", epoch=0,
                  het_collector=None, score_collector=None,
                  neg_strategy="random", hist_neg_ctx=None):
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    return run_epoch(model, split_data, num_nodes, batch_size,
                     optimizer=optimizer,
                     inductive_nodes=inductive_nodes,
                     seen_nodes=seen_nodes, desc=desc,
                     het_collector=het_collector,
                     score_collector=score_collector,
                     neg_strategy=neg_strategy, hist_neg_ctx=hist_neg_ctx)


def run_one(dataset: str, seed: int, epochs: int, hidden: int,
            batch_size: int, lr: float, p0_fix: bool = True,
            enable_lfg: bool = True, enable_echo: bool = False,
            fix_existence_init: bool = False, entropy_reg_weight: float = 0.0,
            design: str = "canonical",
            lambda_edge_trans: float = None, edge_state_entropy_w: float = None,
            edge_uniform_kl_w: float = None, fsm_arch: str = "v1",
            fsm_decode: str = "flat", decol_hier_v2: bool = False,
            causal_batch: bool = False, hier_causal_policy: bool = False,
            lfg_mode: str = None, compliance_floor: float = None,
            causal_confidence: bool = False, cc_C: str = "band",
            cc_thr: float = 0.0, cc_self_consist_w: float = 0.0,
            cc_grounded_init: bool = False,
            hardneg_eval: bool = False, hardneg_eval_seed: int = 12345,
            edge_h_detach_scorepath: bool = True,
            main_predictor_detach: bool = False,
            determ_only_backbone: bool = False,
            frozen_probe: bool = False, probe_epochs: int = None,
            dump_dir: str = None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # ── FREEZE-THEN-PROBE (FtP) control (reviewer #1) ─────────────────────────
    # When frozen_probe=True this run executes the classic linear-probing-transfer
    # baseline for the decoupling-by-construction claim:
    #   PHASE 1 (pretrain): force the END-TO-END link-pred head on (p0_fix→True =
    #     "K1"/correct-e2e) so the backbone csn/ectg/drgc IS shaped by link-pred BCE,
    #   PHASE 2 (freeze+probe): freeze that backbone, throw away the co-trained head,
    #     train a FRESH link head on the frozen features, then measure inductive AP.
    # Every OTHER flag (fsm_arch/decode/decol_hier_v2/causal_batch/lambda_edge_trans/
    # hier_causal_policy/lfg/...) is passed through UNCHANGED so the ONLY difference
    # vs config-B is the training protocol (decoupled-by-construction vs FtP), which
    # is exactly the control the panel asked for. probe_epochs defaults to `epochs`.
    if frozen_probe:
        p0_fix = True                      # PHASE-1 must shape the backbone e2e
        if probe_epochs is None:
            probe_epochs = epochs

    data = download_dataset(dataset)
    splits = get_data_splits(data)
    num_nodes, feat_dim = data["num_nodes"], data["feat_dim"]
    # Global test indices for the per-edge dump. get_data_splits does NOT re-sort;
    # it slices the ts-sorted arrays [val_end:n], so the chronological test-edge order
    # the eval emits == global indices np.arange(val_end, n). (transductive only)
    _n_total = int(data["num_edges"])
    _val_end = int(_n_total * (0.70 + 0.15))   # mirrors get_data_splits exactly
    test_idx_global = np.arange(_val_end, _n_total)

    seen_nodes = (set(splits["train"]["sources"]) | set(splits["train"]["destinations"])
                  | set(splits["val"]["sources"]) | set(splits["val"]["destinations"]))
    test_nodes = set(splits["test"]["sources"]) | set(splits["test"]["destinations"])
    inductive_nodes = sorted(test_nodes - seen_nodes)
    if len(inductive_nodes) < 10:
        inductive_nodes = None

    # Prediction-head ablation toggle:
    #   p0_fix=False → CANONICAL detached readout (existence-decoder/lifecycle head on
    #                  detached features; backbone shaped by KL/TIP, not link-pred BCE).
    #   p0_fix=True  → end-to-end ablation arm (enable_main_predictor: non-detached main
    #                  head trains the backbone by link prediction). Empirically worse.
    # design="correct" turns on the full intended TWO-STREAM CAUSAL-GRADIENT-MASK
    # stack inside the model (enable_main_predictor + lfg_mode=hard + compliance_floor=0
    # + revived lambda_trans transition-CE + entropy_reg + fix_existence_init). The
    # individual flags below are still forwarded; the preset only fills flags left at
    # their canonical default, so design="canonical" (the no-flag path) is unchanged.
    # De-collapse weight overrides: only pass when the CLI explicitly set them, so the
    # design preset (correct_decoupled) fills its tuned defaults otherwise. Passing the
    # value (even 0.0) overrides the preset's `== default` guard, enabling a grid sweep.
    decol_kw = {}
    if lambda_edge_trans is not None:
        decol_kw["lambda_edge_trans"] = lambda_edge_trans
    if edge_state_entropy_w is not None:
        decol_kw["edge_state_entropy_w"] = edge_state_entropy_w
    if edge_uniform_kl_w is not None:
        decol_kw["edge_uniform_kl_w"] = edge_uniform_kl_w
    # LFG gradient-mode override (only pass when CLI set ⇒ overrides preset default,
    # which only fills flags left at the __init__ sentinel). Lets the HARD gate run
    # on the DETACHED correct_decoupled arm without touching enable_main_predictor.
    if lfg_mode is not None:
        decol_kw["lfg_mode"] = lfg_mode
    if compliance_floor is not None:
        decol_kw["compliance_floor"] = compliance_floor
    model = SRGNN_v3_3(num_nodes, feat_dim, hidden, device=DEVICE,
                       enable_main_predictor=p0_fix,
                       enable_lfg=enable_lfg,
                       enable_echo=enable_echo,
                       fix_existence_init=fix_existence_init,
                       entropy_reg_weight=entropy_reg_weight,
                       design=design, fsm_arch=fsm_arch,
                       fsm_decode=fsm_decode, decol_hier_v2=decol_hier_v2,
                       causal_batch=causal_batch,
                       hier_causal_policy=hier_causal_policy,
                       causal_confidence=causal_confidence,
                       cc_C=cc_C, cc_thr=cc_thr,
                       cc_self_consist_w=cc_self_consist_w,
                       cc_grounded_init=cc_grounded_init,
                       edge_h_detach_scorepath=edge_h_detach_scorepath,
                       main_predictor_detach=main_predictor_detach,
                       determ_only_backbone=determ_only_backbone,
                       **decol_kw).to(DEVICE)
    # DETERM-ONLY freezes csn/ectg/drgc at init (requires_grad=False); optimize only
    # the params that still require grad so the frozen backbone is provably untrained.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[v3.3] {dataset}  seed={seed}  epochs={epochs}  "
          f"p0_fix={'on' if p0_fix else 'off'}  "
          f"lfg={'on' if enable_lfg else 'off'}  echo={'on' if enable_echo else 'off'}  "
          f"interp={'on' if (fix_existence_init or entropy_reg_weight > 0) else 'off'}"
          f"(fix_init={fix_existence_init},ent_w={entropy_reg_weight})  "
          f"inductive={'Y' if inductive_nodes else 'N'}")

    # Untimed warmup epoch (absorb cudnn autotune + lazy CUDA alloc) before t0.
    if hasattr(model, "reset"): model.reset()
    run_epoch_v33(model, splits["train"], num_nodes, batch_size,
                  optimizer=optimizer, desc=f"{dataset[:3]}/s{seed}/warmup(untimed)", epoch=0)
    _dev_sync()
    t0 = time.time()
    best_val_ap = 0.0
    best_state = None
    het_traj = []   # per-train-epoch (first_batch, last_batch, n_batches) pair_het_var
    for ep in range(1, epochs + 1):
        if hasattr(model, "reset"): model.reset()
        het_collector = [] if fsm_arch == "v3" else None
        tr = run_epoch_v33(model, splits["train"], num_nodes, batch_size,
                           optimizer=optimizer,
                           desc=f"{dataset[:3]}/s{seed}/E{ep}/tr", epoch=ep,
                           het_collector=het_collector)
        if het_collector:
            first, last = het_collector[0], het_collector[-1]
            nb = len(het_collector)
            het_traj.append({"epoch": ep, "first": first, "last": last,
                             "max": max(het_collector), "n_batches": nb})
            print(f"  E{ep:02d} pair_het_var: first={first:.3e} "
                  f"last={last:.3e} max={max(het_collector):.3e} "
                  f"(n_batch={nb})")
        va = run_epoch_v33(model, splits["val"], num_nodes, batch_size,
                           desc=f"{dataset[:3]}/s{seed}/E{ep}/va", epoch=ep)
        scheduler.step()
        if va["AP"] > best_val_ap:
            best_val_ap = va["AP"]
            best_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            _dev_sync()
            print(f"  E{ep:02d}  tr_AP={tr['AP']:.4f}  va_AP={va['AP']:.4f}  [{time.time()-t0:.0f}s]")

    # ── FREEZE-THEN-PROBE PHASE 2 (reviewer #1 control) ───────────────────────
    # Pretraining above ran with enable_main_predictor=True (forced when
    # frozen_probe), so the backbone is now shaped by link-pred BCE. Load the
    # best-val pretrained weights, FREEZE the backbone, RE-INIT the link head,
    # and train ONLY that fresh head on the frozen features. best_val_ap and
    # best_state are then REDEFINED on the probe head so the test eval below
    # reports the freeze-then-probe result (not the co-trained pretrain head).
    if frozen_probe:
        if hasattr(model, "reset"): model.reset()
        model.load_state_dict(best_state)
        n_frozen = model.freeze_backbone()
        model.reinit_main_predictor()
        head_params = [p for p in model.parameters() if p.requires_grad]
        n_head = sum(p.numel() for p in head_params)
        print(f"  [FtP] backbone frozen ({n_frozen} param-tensors); fresh head "
              f"trainable params={n_head}; probe_epochs={probe_epochs}")
        probe_opt = torch.optim.Adam(head_params, lr=lr, weight_decay=1e-5)
        probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            probe_opt, T_max=max(1, probe_epochs))
        best_val_ap = 0.0
        best_state = None
        # Untimed warmup to repopulate the stateful stores from a clean reset.
        if hasattr(model, "reset"): model.reset()
        run_epoch_v33(model, splits["train"], num_nodes, batch_size,
                      optimizer=probe_opt,
                      desc=f"{dataset[:3]}/s{seed}/FtP/warmup", epoch=0)
        for ep in range(1, probe_epochs + 1):
            if hasattr(model, "reset"): model.reset()
            tr = run_epoch_v33(model, splits["train"], num_nodes, batch_size,
                               optimizer=probe_opt,
                               desc=f"{dataset[:3]}/s{seed}/FtP/E{ep}/tr", epoch=ep)
            va = run_epoch_v33(model, splits["val"], num_nodes, batch_size,
                               desc=f"{dataset[:3]}/s{seed}/FtP/E{ep}/va", epoch=ep)
            probe_sched.step()
            if va["AP"] > best_val_ap:
                best_val_ap = va["AP"]
                best_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                              for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == 1 or ep == probe_epochs:
                _dev_sync()
                print(f"  [FtP] E{ep:02d}  tr_AP={tr['AP']:.4f}  "
                      f"va_AP={va['AP']:.4f}  [{time.time()-t0:.0f}s]")

    # Transductive test (capture per-edge scores for the post-CP dump)
    if hasattr(model, "reset"): model.reset()
    if best_state is not None:
        model.load_state_dict(best_state)
    trans_scores = {}
    test_trans = run_epoch_v33(model, splits["test"], num_nodes, batch_size,
                               desc=f"{dataset[:3]}/s{seed}/trans", epoch=epochs,
                               score_collector=trans_scores)

    # Inductive test
    test_ind = {"AP": float("nan"), "AUC": float("nan")}
    # Hard-negative inductive metrics (Poursafaei et al. 2022). Populated only when
    # hardneg_eval=True; evaluated on the SAME trained model + SAME warmup state as the
    # random-NS inductive number, so the ONLY thing that changes across the three
    # strategies is the negative set. PAIRING across config arms (B vs K1) is guaranteed
    # by resetting np.random to a FIXED hardneg_eval_seed immediately before each
    # strategy's eval — so for a given (dataset, data-seed, strategy) both B and K1
    # draw bit-identical negative sets.
    test_ind_hist = {"AP": float("nan"), "AUC": float("nan")}
    test_ind_indneg = {"AP": float("nan"), "AUC": float("nan")}
    hist_ctx = None
    if inductive_nodes:
        def _warmup_and_eval(strategy, hnc):
            if hasattr(model, "reset"): model.reset()
            model.load_state_dict(best_state)
            run_epoch_v33(model, splits["train"], num_nodes, batch_size,
                          desc=f"{dataset[:3]}/s{seed}/warmup_tr", epoch=epochs)
            run_epoch_v33(model, splits["val"], num_nodes, batch_size,
                          desc=f"{dataset[:3]}/s{seed}/warmup_va", epoch=epochs)
            return run_epoch_v33(model, splits["test"], num_nodes, batch_size,
                                 inductive_nodes=inductive_nodes,
                                 seen_nodes=sorted(seen_nodes),
                                 desc=f"{dataset[:3]}/s{seed}/ind_{strategy}",
                                 epoch=epochs, neg_strategy=strategy, hist_neg_ctx=hnc)

        # random NS (the existing Table-3 reference) — fixed eval RNG for pairing
        np.random.seed(hardneg_eval_seed)
        test_ind = _warmup_and_eval("random", None)

        if hardneg_eval:
            # Build the historical + inductive negative pools ONCE.
            # historical pool (Poursafaei 2022) = multiset of train destination nodes
            # (a "plausible past partner" absent at the current eval timestamp).
            ind_set = set(inductive_nodes)
            tr_src = splits["train"]["sources"]
            tr_dst = splits["train"]["destinations"]
            hist_dst_pool = np.asarray(tr_dst, dtype=np.int64)
            # "active" positive pairs at the eval split — a sampled neg that equals one
            # of these would be a real present edge, so it is rejected.
            active_pos_set = set(
                (int(s), int(d)) for s, d in zip(splits["test"]["sources"],
                                                 splits["test"]["destinations"]))
            # inductive pool (Poursafaei 2022 CORRECT def) = destinations of edges that
            # appear ONLY in the TEST phase: (src,dst) pairs observed during test but
            # NEVER present in train. This is about TEST-PHASE edges, NOT unseen nodes —
            # the prior (buggy) def restricted the TRAIN-dst pool to inductive nodes,
            # which is empty by construction (unseen nodes never appear in train) and
            # degenerated to a fallback. The test-only pool is non-empty & non-degenerate.
            train_pair_set = set(
                (int(s), int(d)) for s, d in zip(tr_src, tr_dst))
            test_only_dst = np.asarray(
                [int(d) for s, d in zip(splits["test"]["sources"],
                                        splits["test"]["destinations"])
                 if (int(s), int(d)) not in train_pair_set],
                dtype=np.int64)
            hist_dst_pool_ind = test_only_dst
            # honest flag: pool size, and whether it is still small/degenerate.
            n_indneg_pool = int(hist_dst_pool_ind.size)
            indpool_from_train = bool(n_indneg_pool > 0)   # retained key name; now means
                                                           # "test-only pool non-empty"
            if not indpool_from_train:
                # last-resort fallback (should not trigger on real datasets): inductive
                # node set itself, flagged so the JSON is honest about degeneracy.
                hist_dst_pool_ind = np.asarray(sorted(ind_set), dtype=np.int64)
                n_indneg_pool = int(hist_dst_pool_ind.size)
            hist_ctx = {
                "hist_dst_pool": hist_dst_pool,
                "hist_dst_pool_ind": hist_dst_pool_ind,
                "active_pos_set": active_pos_set,
                "indpool_from_train": indpool_from_train,
                "n_indneg_pool": n_indneg_pool,
            }
            print(f"  [hardneg] s{seed}: test-only inductive-NS pool n={n_indneg_pool} "
                  f"(from_test_only={indpool_from_train})")
            np.random.seed(hardneg_eval_seed)
            test_ind_hist = _warmup_and_eval("historical", hist_ctx)
            np.random.seed(hardneg_eval_seed)
            test_ind_indneg = _warmup_and_eval("inductive", hist_ctx)

    _dev_sync()
    total_time = time.time() - t0
    final_info = measure(model)

    # ── PER-EDGE DUMP (pack post-CP eval INTO the run) ────────────────────────
    # Save y_true/y_score/test_idx for the TRANSDUCTIVE test so evaluation harness (or this
    # runner inline below) can compute post-change-point AP. y_score holds the
    # positive scores (one per test edge, row-aligned to test_idx_global) plus the
    # negative scores; all captured at the SAME pre-update point the AP metric used.
    postcp = None
    if dump_dir is not None and trans_scores.get("pos") is not None:
        os.makedirs(dump_dir, exist_ok=True)
        pos = np.asarray(trans_scores["pos"], dtype=np.float64)
        neg = np.asarray(trans_scores["neg"], dtype=np.float64)
        n_pos = len(pos)
        # Guard: emitted positive rows must align 1:1 with the global test index.
        if n_pos != len(test_idx_global):
            print(f"  [dump][WARN] n_pos={n_pos} != len(test_idx)={len(test_idx_global)} "
                  f"— dataset/eval row mismatch; saving raw arrays only, NO test_idx map.")
            ti = np.full(n_pos, -1, dtype=np.int64)
        else:
            ti = test_idx_global.astype(np.int64)
        y_true = np.concatenate([np.ones(n_pos), np.zeros(len(neg))]).astype(np.int8)
        y_score = np.concatenate([pos, neg]).astype(np.float64)
        npz_path = os.path.join(
            dump_dir,
            f"peredge_{dataset}_seed{seed}_p0{'on' if p0_fix else 'off'}_{fsm_arch}.npz")
        np.savez_compressed(npz_path,
                            y_true=y_true, y_score=y_score,
                            pos_score=pos, neg_score=neg,
                            test_idx=ti, n_pos=np.int64(n_pos))
        print(f"  [dump] per-edge scores -> {npz_path} (n_pos={n_pos}, n_neg={len(neg)})")

        # Inline post-CP AP for synthetic_regime (best-effort; NEVER fabricate — on
        # any error we leave postcp=None and evaluation harness recomputes from the .npz).
        if dataset == "synthetic_regime" and ti[0] != -1:
            try:
                sys.path.insert(0, EXP_DIR)
                from data.regime_postcp_eval import (
                    load_test_anomaly_flag, load_test_phase,
                    load_test_relationship_id, postcp_window_mask,
                    postcp_window_mask_perpair, ap_on_pos_subset)
                flag = load_test_anomaly_flag(ti)
                phase = load_test_phase(ti)
                rid = load_test_relationship_id(ti)
                postcp = {"overall_ap": test_trans["AP"], "n_pos": int(n_pos)}
                # exact CP edges (W=0 global == per-pair W=0)
                postcp["cp_w0_ap"] = ap_on_pos_subset(
                    y_true, y_score, postcp_window_mask(flag, window=0))
                # per-pair windows (the recommended granularity)
                for w in (1, 2, 5):
                    postcp[f"perpair_w{w}_ap"] = ap_on_pos_subset(
                        y_true, y_score, postcp_window_mask_perpair(flag, rid, window=w))
                # high-regime subset
                postcp["high_phase1_ap"] = ap_on_pos_subset(
                    y_true, y_score, (phase == 1))
                print(f"  [postcp] {dataset} s{seed}: cp_w0={postcp['cp_w0_ap']:.4f} "
                      f"pp_w1={postcp['perpair_w1_ap']:.4f} "
                      f"pp_w2={postcp['perpair_w2_ap']:.4f} "
                      f"pp_w5={postcp['perpair_w5_ap']:.4f} "
                      f"hi={postcp['high_phase1_ap']:.4f}")
            except Exception as e:
                print(f"  [postcp][WARN] inline post-CP failed ({e}); .npz saved, "
                      f"evaluation harness can recompute.")
                import traceback; traceback.print_exc()
                postcp = None

    return {
        "dataset":     dataset,
        "seed":        seed,
        "epochs":      epochs,
        "postcp":      postcp,
        "peredge_npz": npz_path if (dump_dir is not None and trans_scores.get("pos") is not None) else None,
        "p0_fix":      "on" if p0_fix else "off",
        "lfg":         "on" if enable_lfg else "off",
        "echo":        "on" if enable_echo else "off",
        "fix_existence_init": fix_existence_init,
        "entropy_reg_weight": entropy_reg_weight,
        "design":      design,
        "fsm_arch":    fsm_arch,
        "fsm_decode":  fsm_decode,
        "decol_hier_v2": decol_hier_v2,
        "causal_batch": causal_batch,
        "hier_causal_policy": hier_causal_policy,
        "edge_h_detach_scorepath": bool(edge_h_detach_scorepath),
        "main_predictor_detach": bool(main_predictor_detach),
        "determ_only_backbone": bool(determ_only_backbone),
        "n_backbone_params_frozen": int(getattr(model, "_n_backbone_params_frozen", 0)),
        "frozen_probe": bool(frozen_probe),
        "causal_confidence": causal_confidence,
        "cc_C":        cc_C if causal_confidence else None,
        "cc_thr":      cc_thr if causal_confidence else None,
        "cc_self_consist_w": cc_self_consist_w if causal_confidence else None,
        "cc_grounded_init": bool(cc_grounded_init) if causal_confidence else None,
        "trans_ap":    test_trans["AP"], "trans_auc": test_trans["AUC"],
        "ind_ap":      test_ind["AP"],   "ind_auc":   test_ind["AUC"],
        "ind_ap_histneg":  test_ind_hist["AP"],   "ind_auc_histneg":  test_ind_hist["AUC"],
        "ind_ap_indneg":   test_ind_indneg["AP"], "ind_auc_indneg":   test_ind_indneg["AUC"],
        "hardneg_eval":    bool(hardneg_eval),
        "hardneg_indpool_from_train": (hist_ctx.get("indpool_from_train") if hist_ctx else None),
        "hardneg_n_indneg_pool": (hist_ctx.get("n_indneg_pool") if hist_ctx else None),
        "best_val_ap": best_val_ap,
        "train_time_s": total_time,
        "final_info":  final_info,
        "het_traj":    het_traj,
    }


def summarize(results):
    """Group by (dataset, p0_fix arm), compute mean±std."""
    by_ds = {}
    for r in results:
        key = (r["dataset"], r.get("p0_fix", "off"))
        by_ds.setdefault(key, []).append(r)
    summary = []
    for (ds, arm), rows in by_ds.items():
        aps_t = [r["trans_ap"] for r in rows]
        aucs_t = [r["trans_auc"] for r in rows]
        aps_i = [r["ind_ap"] for r in rows if not np.isnan(r["ind_ap"])]
        aucs_i = [r["ind_auc"] for r in rows if not np.isnan(r["ind_auc"])]
        times = [r["train_time_s"] for r in rows]
        summary.append({
            "dataset":         ds,
            "p0_fix":          arm,
            "trans_ap_mean":   float(np.mean(aps_t)),
            "trans_ap_std":    float(np.std(aps_t)),
            "trans_auc_mean":  float(np.mean(aucs_t)),
            "trans_auc_std":   float(np.std(aucs_t)),
            "ind_ap_mean":     float(np.mean(aps_i)) if aps_i else float("nan"),
            "ind_ap_std":      float(np.std(aps_i)) if aps_i else float("nan"),
            "ind_auc_mean":    float(np.mean(aucs_i)) if aucs_i else float("nan"),
            "ind_auc_std":     float(np.std(aucs_i)) if aucs_i else float("nan"),
            "time_mean":       float(np.mean(times)),
            "n_seeds":         len(rows),
        })
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="SR-GNN v3.3 benchmark / P0 A/B runner")
    # Accept both --dataset and --datasets (comma-sep or repeated). Default = full 3-dataset set.
    p.add_argument("--datasets", "--dataset", dest="datasets", default="coedit",
                   help="Dataset list owned by this project: coedit")
    p.add_argument("--seeds", default="1,7,42",
                   help="Comma-separated seed list (default: 1,7,42)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch",  type=int, default=500)
    p.add_argument("--lr",     type=float, default=1e-3)
    # Prediction-head ablation: 'off'=CANONICAL detached readout (default, empirically
    # better — wins all 3 datasets in the 2026-05-30 A/B), 'on'=end-to-end main-predictor
    # ablation arm, 'both'=run both arms (the headline A/B).
    p.add_argument("--p0_fix", choices=["on", "off", "both"], default="off",
                   help="prediction head: off=canonical detached readout (default), "
                        "on=end-to-end main predictor ablation, both=A/B")
    # ── Three independently-flagged improvements (each OFF/canonical by default) ──
    # #1 LFG toggle: on (default) = canonical (LFG compliance reweighting on);
    #    off = uniform weight=1 on positives (LFG disabled) → isolates LFG effect.
    p.add_argument("--lfg", choices=["on", "off"], default="on",
                   help="Lifecycle-Filtered Gradient reweighting: on=canonical (default), "
                        "off=uniform weight=1 (ablation)")
    # #3 Echo toggle: off (default) = canonical (no echo); on = port v3.1 EchoMemory.
    p.add_argument("--echo", choices=["on", "off"], default="off",
                   help="EchoMemory injection: off=canonical no-echo (default), "
                        "on=port v3.1 time-decayed echo into the backbone")
    # #2 Interpretability variant: off (default, canonical = fix_existence_init=False,
    #    entropy_reg_weight=0.0). on = fix_existence_init=True + entropy_reg_weight=0.01
    #    (entropy reg pushes the symbolic state distribution off the ~0.95-IDLE collapse;
    #    0.01 chosen as a small term that does not dominate the O(1) pred_loss).
    p.add_argument("--interp", choices=["on", "off"], default="off",
                   help="Interpretability variant: off=canonical (default); "
                        "on=fix_existence_init + entropy_reg_weight=0.01")
    p.add_argument("--interp_entropy_w", type=float, default=0.01,
                   help="entropy_reg_weight used when --interp on (default 0.01)")
    # Composite preset for the INTENDED two-stream causal-gradient-mask model.
    #   canonical (default) = no preset (no-flag run == current canonical detached).
    #   correct = enable_main_predictor + lfg_mode=hard (compliance_floor=0) +
    #             revived lambda_trans transition-CE + entropy_reg + fix_existence_init.
    p.add_argument("--design",
                   choices=["canonical", "correct", "correct_decoupled"],
                   default="canonical",
                   help="model preset: canonical=current detached default; "
                        "correct=intended two-stream HARD causal-gradient-mask stack "
                        "(Stream1 trainable + hard LFG gate + revived transition-CE + "
                        "entropy + fix_init) [Tier-1, FAILED de-collapse 2026-05-30]; "
                        "correct_decoupled=Tier-1 RE-GATE: keep the BETTER-AP DETACHED "
                        "head (enable_main_predictor=False) but add strong de-collapse "
                        "supervision on the CONTINUOUS ECTGv3 edge state (corrected "
                        "n_obs-based target + edge-state transition-CE + per-event "
                        "entropy floor + uniform-KL floor) — isolates FSM health from "
                        "the head choice")
    # ── De-collapse weight overrides (for design=correct_decoupled grid sweep) ──
    # Default None → use the preset's tuned defaults (CE 0.10 / ent 0.02 / ukl 0.01).
    # Pass a NON-default value to override and sweep a balance grid in ONE job. The
    # corrected target now CARRIES the lifecycle spread (BIRTH .60/REINF .31/DECAY .07,
    # H=0.94 CPU-measured), so the intended balance is CE-led with gentle floors; a
    # strong floor pushes toward ARTIFICIAL uniformity (meaningless states).
    p.add_argument("--lambda_edge_trans", type=float, default=None,
                   help="edge-state transition-CE weight (preset default 0.10)")
    p.add_argument("--edge_entropy_w", type=float, default=None,
                   help="per-event entropy floor weight (preset default 0.02)")
    p.add_argument("--edge_uniform_kl_w", type=float, default=None,
                   help="uniform-KL floor weight (preset default 0.01)")
    p.add_argument("--causal_batch", action="store_true",
                   help="CAUSAL intra-batch accumulation (P1 fix): fold repeated same-"
                        "pair events WITHIN a batch in stream order so Welford n/μ/var, "
                        "Hawkes λ and rate fast/slow/peak match an event-by-event "
                        "reference. Legacy (default off) snapshots once/batch ⇒ n caps "
                        "(~6 on coedit b=500), rate pinned at RATE_INIT on recurring "
                        "pairs. A/B knob; AP MAY shift (corrupted stats feed the gate φ).")
    p.add_argument("--fsm_arch", choices=["v1", "v2", "v3"], default="v1",
                   help="symbolic-FSM architecture for the gate+de-collapse CE. "
                        "v1 (default, canonical): ECTGv3 VALID-HARD-MASKED continuous "
                        "chain (structurally PINNED at BIRTH, invariant to supervision). "
                        "v2: SEPARATE soft-masked FSM head s_{t+1} (finite penalty → "
                        "MOVABLE; de-pins but OVER-corrects to one shared state). "
                        "v3: PER-PAIR flip dynamics — v2 soft head + per-pair operator "
                        "g(phi_uv) from EdgeStateStoreV3 history + observed self-"
                        "supervised lifecycle target (no entropy hammer). Self-contained "
                        "(auto-enables decollapse_target + soft-head CE).")
    # ── FSM lifecycle readout (validated config: fsm_arch=v3 + fsm_decode=hier +
    #    decol_hier_v2). fsm_decode default "flat" ⇒ byte-identical to the existing
    #    v3 flat readout; "hier" = HIERARCHICAL/ORDINAL tree decode (BIRTH→alive→
    #    {rising:REINFORCE | falling:DECAY}, dead:DEATH) that lets the intermediate
    #    DECAY state win argmax. Only the STATE readout is rerouted; the existence
    #    AP score (s_t1_pos) is UNTOUCHED (AP-path Δ=0). fsm_arch=v3 only.
    p.add_argument("--fsm_decode", choices=["flat", "hier"], default="flat",
                   help="symbolic-state READOUT: flat (default, canonical) = single "
                        "5-way softmax; hier = ordinal lifecycle tree decode "
                        "(fsm_arch=v3 only; AP-path unchanged).")
    # decol_hier_v2 (default False ⇒ hier-v1 byte-identical): re-anchors the p_alive/
    # p_rising priors on true_occ + sustained-silence/staleness-relative axes so
    # REINFORCE survives the alive branch while DECAY competes — the 3-seed-validated
    # balance (REINFORCE .95 / DECAY .04 / DEATH .01 on true_occ>=2). fsm_decode=hier only.
    p.add_argument("--decol_hier_v2", action="store_true",
                   help="re-anchored hierarchical de-collapse priors (true_occ-based "
                        "p_alive/p_rising; the 3-seed-validated 5-state balance). "
                        "Requires --fsm_decode hier --fsm_arch v3.")
    p.add_argument("--hier_causal_policy", action="store_true",
                   help="apply the causal policy (ever_alive DEATH gate + soft "
                        "expected-admissibility C-mask) to the PUBLISHED state s_t1_cal "
                        "(the de-collapse-CE / faithfulness / edge_state_dist quantity). "
                        "AP path (s_t1_pos→existence_decoder) is UNTOUCHED ⇒ AP Δ=0. "
                        "Default OFF = byte-identical hier behavior. Requires "
                        "--fsm_decode hier --fsm_arch v3.")
    # ── LFG gradient-mode override (decouples lfg_mode from the design preset) ──
    # The design presets pin lfg_mode/compliance_floor (correct_decoupled keeps
    # lfg_mode="soft" + floor=0.05 by design). These flags let evaluation harness run the
    # HARD causal-gradient gate ON THE DETACHED arm (correct_decoupled) WITHOUT
    # flipping enable_main_predictor — the exact LFG-gradient-mode A/B (HARD vs SOFT
    # vs NONE) requested by PM. Defaults None ⇒ the model/preset default is used
    # (byte-identical to before). Forwarded explicitly so they OVERRIDE the preset
    # (which only fills flags left at their __init__ default sentinel).
    #   ARM-HARD : --lfg_mode hard --compliance_floor 0.0
    #   ARM-SOFT : (omit ⇒ preset soft, floor 0.05)  [== current config B]
    #   ARM-NONE : --lfg off  (forces lfg_weight=0 ⇒ uniform weight=1 everywhere)
    p.add_argument("--lfg_mode", choices=["soft", "hard"], default=None,
                   help="LFG gradient mode override. hard = causal-rule HARD gate "
                        "(C[argmax s_t, argmax s_t1]; impossible→weight compliance_floor) "
                        "on pred_loss; soft = compliance-ramp reweight. Default None = "
                        "use preset/model default (correct_decoupled→soft). Runs on the "
                        "DETACHED arm (does NOT flip enable_main_predictor).")
    p.add_argument("--compliance_floor", type=float, default=None,
                   help="per-event gradient weight for causally-IMPOSSIBLE positives "
                        "under lfg_mode=hard (0.0 = full hard gate, zero gradient). "
                        "Default None = model default (0.05). Set 0.0 for ARM-HARD.")
    # ── WC-CONF: walked-chain causal-confidence (PM 2026-06-06) ─────────────────
    # Causality does NOT mask the prediction (AP path stays FREE); it (1) emits a
    # coherence/confidence c_t and (2) SELECTS gradient (scales the FSM-block CE by
    # c_t / zeroes below cc_thr). Default OFF = byte-identical. Requires --fsm_decode
    # hier. Recommended WC-CONF arm vs config B keeps the score path free (do NOT pass
    # --hier_causal_policy) so the AP comparison is fair.
    p.add_argument("--causal_confidence", action="store_true",
                   help="WC-CONF: add walked-chain belief b_t + coherence c_t + "
                        "gradient-selection (FSM-block CE scaled by c_t). Does NOT mask "
                        "the prediction/AP value. Requires --fsm_decode hier.")
    p.add_argument("--cc_C", choices=["band", "rule"], default="band",
                   help="WC-CONF causal admissibility matrix: band=C_BAND_5 (strict "
                        "|i-j|<=1, no nhay coc); rule=CAUSAL_RULE_MATRIX (legacy).")
    p.add_argument("--cc_thr", type=float, default=0.0,
                   help="WC-CONF hard coherence floor: events with c_t<cc_thr get ZERO "
                        "FSM-block gradient. 0.0 = pure soft c_t scaling (no hard cutoff).")
    p.add_argument("--cc_self_consist_w", type=float, default=0.0,
                   help="WC-CONF belief self-consistency aux-loss weight. >0 restores a "
                        "LEARNABLE observation-coupling (w_obs=sigmoid(param)) in the belief "
                        "filter, trained by CE(belief||free-next-state). Gradient-isolated "
                        "from predict (AP Δ=0). 0.0 = FIX-3 closed-loop filter (no aux).")
    p.add_argument("--cc_grounded_init", action="store_true",
                   help="WC-CONF GROUNDED belief init: seed the walked-chain at the MODEL-"
                        "INFERRED phase (softmax s_t_pos) for a pair's FIRST appearance in "
                        "the split, NOT IDLE one-hot. Fixes the structural ceiling where a "
                        "mature pair entering test reset to IDLE and could never climb. Pure "
                        "init-source swap (no new param/state_dict key). Requires "
                        "--causal_confidence. Default OFF = byte-identical IDLE init.")
    # ── SINGLE-VARIABLE DETACH PROBE (PM 2026-06-08, reviewer #1) ───────────────
    # The ONE bit the panel asked to isolate from the B-vs-C confound: whether the
    # link-prediction SCORE path (s_t1_pos→existence_decoder, + symmetric neg) carries
    # gradient back into the backbone (csn/ectg/drgc) or is .detach()ed.
    #   on  (default) = CANONICAL config-B: detach ON ⇒ backbone gets ZERO link-pred
    #                   gradient (regularization-by-decoupling). BYTE-IDENTICAL to all
    #                   prior config-B runs.
    #   off           = remove the detach ONLY on the pos/neg scoring path ⇒ link-pred
    #                   BCE trains the backbone end-to-end. enable_main_predictor STAYS
    #                   off; EVERY other knob (lfg_mode/floor/lambda_edge_trans/entropy/
    #                   kl/fsm_arch/decode/init) is IDENTICAL. The clean A/B that the
    #                   old B-vs-C arm (which flipped MANY knobs at once) conflated.
    p.add_argument("--detach_scorepath", choices=["on", "off"], default="on",
                   help="link-pred score-path→backbone gradient: on=canonical detached "
                        "config-B (default, byte-identical); off=un-detach the score "
                        "path so link-pred BCE trains the backbone (single-variable "
                        "decoupling probe). enable_main_predictor unchanged.")
    # ── FREEZE-THEN-PROBE control (PM 2026-06-09, reviewer #1) ──────────────────
    # The decisive novelty control: separate SR-GNN's "decoupling-by-construction"
    # (backbone NEVER sees link-pred grad) from the classic "freeze-then-probe /
    # linear-probing transfer" (train backbone WITH link-pred, FREEZE it, train a
    # fresh head). --frozen_probe runs the FtP arm: PHASE-1 pretrain forces the e2e
    # head on (backbone shaped by link-pred), PHASE-2 freezes backbone + trains a
    # fresh head + measures inductive AP. Every other flag passes through unchanged
    # so the ONLY diff vs config-B is the training protocol. Default OFF (no-op).
    p.add_argument("--frozen_probe", action="store_true",
                   help="freeze-then-probe control: pretrain e2e (backbone shaped by "
                        "link-pred) → freeze backbone → train fresh link head → measure "
                        "inductive AP. Forces enable_main_predictor on in PHASE-1.")
    p.add_argument("--probe_epochs", type=int, default=None,
                   help="epochs for the frozen-head PHASE-2 probe (default = --epochs).")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "audit" / "v3_3_benchmark.json"),
                   help="Output JSON path (runs + summary)")
    p.add_argument("--dump_dir", default=None,
                   help="If set, write per-edge (y_true,y_score,test_idx) .npz per "
                        "(dataset,seed) for audited downstream evaluation.")
    return p.parse_args()


def main():
    args = parse_args()
    DATASETS = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unsupported = sorted(set(DATASETS) - {"coedit"})
    if unsupported:
        raise ValueError(f"Datasets outside this project: {unsupported}")
    SEEDS    = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    if args.p0_fix == "both":
        ARMS = [("on", True), ("off", False)]
    else:
        ARMS = [(args.p0_fix, args.p0_fix == "on")]

    # Improvement flags (each OFF/canonical by default → no-flag run == canonical).
    enable_lfg = (args.lfg == "on")
    enable_echo = (args.echo == "on")
    if args.interp == "on":
        fix_existence_init = True
        entropy_reg_weight = args.interp_entropy_w
    else:
        fix_existence_init = False
        entropy_reg_weight = 0.0

    results = []
    failures = []
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total = len(ARMS) * len(DATASETS) * len(SEEDS)
    idx = 0
    for arm_name, arm_flag in ARMS:
        for ds in DATASETS:
            for s in SEEDS:
                idx += 1
                print(f"\n{'='*60}\nRUN {idx}/{total}  p0_fix={arm_name}  dataset={ds}  seed={s}\n{'='*60}")
                try:
                    r = run_one(ds, s, args.epochs, args.hidden, args.batch, args.lr,
                                p0_fix=arm_flag,
                                enable_lfg=enable_lfg, enable_echo=enable_echo,
                                fix_existence_init=fix_existence_init,
                                entropy_reg_weight=entropy_reg_weight,
                                design=args.design,
                                lambda_edge_trans=args.lambda_edge_trans,
                                edge_state_entropy_w=args.edge_entropy_w,
                                edge_uniform_kl_w=args.edge_uniform_kl_w,
                                fsm_arch=args.fsm_arch,
                                fsm_decode=args.fsm_decode,
                                decol_hier_v2=args.decol_hier_v2,
                                causal_batch=args.causal_batch,
                                hier_causal_policy=args.hier_causal_policy,
                                lfg_mode=args.lfg_mode,
                                compliance_floor=args.compliance_floor,
                                causal_confidence=args.causal_confidence,
                                cc_C=args.cc_C, cc_thr=args.cc_thr,
                                cc_self_consist_w=args.cc_self_consist_w,
                                cc_grounded_init=args.cc_grounded_init,
                                edge_h_detach_scorepath=(args.detach_scorepath == "on"),
                                frozen_probe=args.frozen_probe,
                                probe_epochs=args.probe_epochs,
                                dump_dir=args.dump_dir)
                    results.append(r)
                    with open(out_path, "w") as f:
                        json.dump({"runs": results, "summary": summarize(results)}, f, indent=2, default=str)
                    print(f"  → [{arm_name}] {ds} s{s}: Trans AP={r['trans_ap']:.4f}  "
                          f"Ind AP={r['ind_ap']:.4f}  edge_state_dist={r['final_info'].get('edge_state_dist')}")
                except Exception as e:
                    print(f"  ✗ FAILED [{arm_name}] {ds} s{s}: {e}")
                    failures.append({"arm": arm_name, "dataset": ds, "seed": s,
                                     "error": str(e)})
                    import traceback; traceback.print_exc()

    summary = summarize(results)
    print("\n" + "="*84)
    print(f"v3.3 BENCHMARK  ({len(SEEDS)} seeds × {args.epochs} epochs × "
          f"{len(DATASETS)} datasets × {len(ARMS)} arm(s))")
    print("="*84)
    print(f"{'Dataset':<12} {'P0':>4} | {'Trans AP':>14} | {'Trans AUC':>14} | "
          f"{'Ind AP':>14} | {'Ind AUC':>14} | {'Time':>6}")
    print("-"*92)
    for s in summary:
        print(f"{s['dataset']:<12} {s['p0_fix']:>4} | "
              f"{s['trans_ap_mean']:.4f}±{s['trans_ap_std']:.4f} | "
              f"{s['trans_auc_mean']:.4f}±{s['trans_auc_std']:.4f} | "
              f"{s['ind_ap_mean']:.4f}±{s['ind_ap_std']:.4f} | "
              f"{s['ind_auc_mean']:.4f}±{s['ind_auc_std']:.4f} | "
              f"{s['time_mean']:>5.0f}s")
    with open(out_path, "w") as f:
        json.dump({"runs": results, "summary": summary, "failures": failures},
                  f, indent=2, default=str)
    print(f"\nSaved → {out_path}")
    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed; see {out_path}")


if __name__ == "__main__":
    main()
