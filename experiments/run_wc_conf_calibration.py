"""
WC-CONF (walked-chain causal-confidence) vs config B — CALIBRATION driver.
evaluation harness, protocol requirement 2026-06-06.

Design lock (PM): causality = a CONFIDENCE score for humans + gradient-selection.
It does NOT bend the prediction / AP score. This driver MEASURES whether the
emitted confidence c_t is CALIBRATED:
  (a) c_t LOW  <->  causal violation (free next-state lands OUTSIDE the walked-chain
                    reachable set: teleport, e.g. never-born -> DEATH)
  (b) c_t LOW  <->  link-pred error (true-positive test edge scored LOW)

Two arms, SAME protocol (coedit, seed 42, 20 ep, batch 500, design=correct_decoupled,
fsm v3 + hier + decol_hier_v2 + causal_batch + lambda_edge_trans 0.5):
  WC-CONF : --causal_confidence --cc_C band --cc_thr 0.0  (FREE score path; no policy mask)
  B       : --hier_causal_policy (value-mask), no causal_confidence

Packs aggregation INTO the run: one invocation -> predictions (AP/AUC) + per-event
npz + calibration summary JSON.  NO post-hoc aggregator pass.

Eval-only instrumentation: the transductive test pass runs in eval()/no_grad and reads
the model's ALREADY-EMITTED out[cc_coherence/cc_weight/cc_reach/cc_belief/s_t1_cal/
pos_score].  AP-path (pos_score) is byte-identical to the standard run_epoch (same
pre-update sigmoid(pos_score)); the model/loss are UNTOUCHED.

Honest limitations reported by the driver:
 * Calibration is measured on the TRANSDUCTIVE TEST POSITIVES only (the events the
   model interprets). Negatives have no lifecycle ground-truth, so the c_t<->error
   axis uses positives (a low score on a true positive == a real miss).
 * "Causal violation" ground-truth is STRUCTURAL: argmax(s_t1_cal) not in the
   walked-chain reachable set cc_reach. This is the model's own admissibility, not an
   external oracle; coedit has no external lifecycle labels. Reported as such.
"""
import os, sys, time, json, random, argparse
import numpy as np
import torch
from project_paths import AUDIT_DIR

V33_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(V33_DIR)
EXP_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, V33_DIR)

from lifecycle_readout.datasets import download_dataset, get_data_splits
from lifecycle_readout.training import run_epoch, sample_negatives, DEVICE, _dev_sync
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3

IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
STATE_NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (no sklearn dependency assumption -> compute AP/AUC/rank-AUC by hand,
# matching RunningMetrics' average_precision semantics).
# ─────────────────────────────────────────────────────────────────────────────
def _ap(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(yt.sum()), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def _auc(y_true, y_score):
    """ROC-AUC via Mann-Whitney U (rank-sum). Returns nan if one class absent."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ties
    s_sorted = y_score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Train one arm with the SHARED protocol, then a calibration test pass.
# ─────────────────────────────────────────────────────────────────────────────
def build_model(dataset, num_nodes, feat_dim, hidden, *, arm,
                cc_self_consist_w=0.0, cc_grounded_init=False):
    common = dict(
        design="correct_decoupled", fsm_arch="v3", fsm_decode="hier",
        decol_hier_v2=True, causal_batch=True, lambda_edge_trans=0.5,
        enable_main_predictor=False,   # p0_fix off (detached AP head)
    )
    if arm == "wc":
        # cc_self_consist_w>0 restores LEARNABLE obs-coupling w_obs trained by the
        # self-consistency aux CE (predict path stays byte-identical; only cc_w_obs_logit
        # gets gradient). Default 0 == FIX-3 zero-hand belief (legacy diagnostic).
        # cc_grounded_init seeds the belief at the model-inferred phase (softmax s_t_pos)
        # for a pair's first appearance in the split instead of IDLE (PM 2026-06-07).
        m = SRGNN_v3_3(num_nodes, feat_dim, hidden, device=DEVICE,
                       causal_confidence=True, cc_C="band", cc_thr=0.0,
                       cc_self_consist_w=cc_self_consist_w,
                       cc_grounded_init=cc_grounded_init, **common)
    elif arm == "B":
        m = SRGNN_v3_3(num_nodes, feat_dim, hidden, device=DEVICE,
                       hier_causal_policy=True, **common)
    else:
        raise ValueError(arm)
    return m.to(DEVICE)


def calib_test_pass(model, split, num_nodes, batch_size, seed):
    """Transductive test pass (eval, no_grad). Collect per-POSITIVE-event:
       c_t, cc_weight, argmax(s_t1_cal), reach-hit(argmax), belief-argmax, pos_sigmoid;
       and the negative scores for AP. AP-path (pos_score) byte-identical to run_epoch.
    """
    model.eval()
    if hasattr(model, "reset"):
        model.reset()
    src_all = split["sources"]; dst_all = split["destinations"]
    t_all = split["timestamps"]; feat_all = split["features"]
    N = len(src_all)
    np.random.seed(seed)   # deterministic neg pool (transductive uses sample_negatives)

    buf = {k: [] for k in ("c_t", "cc_w", "argmax", "reach_hit", "belief_arg",
                            "pos", "neg", "ever_alive_proxy")}
    has_cc = None
    with torch.no_grad():
        for start in range(0, N, batch_size):
            idx = np.arange(start, min(start + batch_size, N))
            if len(idx) == 0:
                continue
            src = torch.tensor(src_all[idx], dtype=torch.long, device=DEVICE)
            dst = torch.tensor(dst_all[idx], dtype=torch.long, device=DEVICE)
            t = torch.tensor(t_all[idx], dtype=torch.float, device=DEVICE)
            feat = torch.tensor(feat_all[idx], dtype=torch.float, device=DEVICE)
            neg_np = sample_negatives(dst_all[idx], num_nodes)
            neg = torch.tensor(neg_np, dtype=torch.long, device=DEVICE)

            out = model(src, dst, t, feat, neg)
            pos = torch.sigmoid(out["pos_score"]).detach().cpu().numpy()
            negs = torch.sigmoid(out["neg_score"]).detach().cpu().numpy()
            buf["pos"].append(pos); buf["neg"].append(negs)

            s_cal = out["s_t1_cal"]                       # (B,5)
            argmax = s_cal.argmax(-1).detach().cpu().numpy()
            buf["argmax"].append(argmax)

            cc = out.get("cc_coherence", None)
            if cc is not None:
                has_cc = True
                buf["c_t"].append(cc.detach().cpu().numpy())
                buf["cc_w"].append(out["cc_weight"].detach().cpu().numpy())
                reach = out["cc_reach"].detach().cpu().numpy()      # (B,5) hard mask
                belief = out["cc_belief"].detach().cpu().numpy()    # (B,5)
                bidx = np.arange(len(argmax))
                buf["reach_hit"].append(reach[bidx, argmax])        # 1 if reachable
                buf["belief_arg"].append(belief.argmax(-1))
            else:
                has_cc = False
    cat = {k: (np.concatenate(v) if v else np.array([])) for k, v in buf.items()}
    cat["_has_cc"] = bool(has_cc)
    return cat


def calibration_metrics(cat):
    """Compute AP/AUC + (if WC) c_t calibration vs violation and vs link-pred error."""
    pos = cat["pos"]; neg = cat["neg"]
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int8)
    y_score = np.concatenate([pos, neg]).astype(np.float64)
    res = {
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "trans_ap": _ap(y_true, y_score), "trans_auc": _auc(y_true, y_score),
    }
    # lifecycle dist over positive test events
    if len(cat["argmax"]):
        cnt = np.bincount(cat["argmax"].astype(int), minlength=5).astype(float)
        res["pos_lifecycle_dist"] = (cnt / cnt.sum()).tolist()
    if not cat.get("_has_cc", False) or not len(cat["c_t"]):
        res["has_cc"] = False
        return res, None
    res["has_cc"] = True

    c_t = cat["c_t"].astype(np.float64)
    cc_w = cat["cc_w"].astype(np.float64)
    reach_hit = cat["reach_hit"].astype(int)            # 1 reachable, 0 violation
    violation = (1 - reach_hit).astype(int)             # 1 = causal violation (teleport)
    belief_arg = cat["belief_arg"].astype(int)
    argmax = cat["argmax"].astype(int)

    # never-born proxy: belief argmax == IDLE (b ~ IDLE) AND free next-state == DEATH
    never_born_death = ((belief_arg == IDLE) & (argmax == DEATH)).astype(int)

    # ── (1) calibration: c_t LOW <-> causal violation. AUC(violation | -c_t) ──
    #    label=violation(1), score=(1 - c_t) so "low c_t" predicts "violation".
    res["c_t_stats"] = {
        "mean": float(c_t.mean()), "std": float(c_t.std()),
        "min": float(c_t.min()), "max": float(c_t.max()),
        "p10": float(np.percentile(c_t, 10)), "p50": float(np.percentile(c_t, 50)),
        "p90": float(np.percentile(c_t, 90)),
    }
    res["cc_weight_stats"] = {
        "mean": float(cc_w.mean()), "std": float(cc_w.std()),
        "min": float(cc_w.min()), "max": float(cc_w.max()),
        "frac_zero": float((cc_w == 0).mean()),
        "frac_lt_0.5": float((cc_w < 0.5).mean()),
    }
    res["violation_rate"] = float(violation.mean())
    res["never_born_death_rate"] = float(never_born_death.mean())

    if violation.sum() > 0 and violation.sum() < len(violation):
        res["AUC_lowc_predicts_violation"] = _auc(violation, 1.0 - c_t)
        res["corr_lowc_violation"] = _pearson(1.0 - c_t, violation)
        res["mean_c_t_violation"] = float(c_t[violation == 1].mean())
        res["mean_c_t_coherent"] = float(c_t[violation == 0].mean())
    else:
        res["AUC_lowc_predicts_violation"] = None
        res["note_violation"] = (f"violation count={int(violation.sum())} of "
                                 f"{len(violation)} -> degenerate, AUC undefined")
    if never_born_death.sum() > 0 and never_born_death.sum() < len(never_born_death):
        res["AUC_lowc_predicts_neverbornDEATH"] = _auc(never_born_death, 1.0 - c_t)
        res["mean_c_t_neverbornDEATH"] = float(c_t[never_born_death == 1].mean())
    else:
        res["AUC_lowc_predicts_neverbornDEATH"] = None

    # ── (2) calibration: c_t LOW <-> link-pred error on TRUE POSITIVES ──
    #    A true positive scored LOW is a real miss. Does low c_t flag it?
    pos_score = cat["pos"].astype(np.float64)           # sigmoid pos score
    # define "error" = positive in the bottom decile of pos score (hard misses)
    thr10 = np.percentile(pos_score, 10)
    miss10 = (pos_score <= thr10).astype(int)
    res["corr_lowc_posScore"] = _pearson(1.0 - c_t, 1.0 - pos_score)  # both "badness"
    res["corr_c_t_posScore"] = _pearson(c_t, pos_score)

    # ── (3) R²(c_t ~ free_argmax) : how much of c_t is explained by the FREE next-state
    #    argmax alone. High R² (old 0.908, zero-hand 0.955) => c_t is a deterministic
    #    function of free decode (degenerate). One-hot OLS R² = 1 - SS_res/SS_tot where
    #    each free-state group prediction = its own mean c_t (best one-hot fit).
    grp_mean = np.zeros(5, dtype=np.float64)
    for s in range(5):
        m = (argmax == s)
        grp_mean[s] = c_t[m].mean() if m.any() else c_t.mean()
    pred = grp_mean[argmax]
    ss_tot = float(((c_t - c_t.mean()) ** 2).sum())
    ss_res = float(((c_t - pred) ** 2).sum())
    res["R2_c_t_free_argmax"] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    # corr(c_t, free_argmax) as integer code, for reference vs old −0.94 (pos was a
    # proxy; here we also report against free-argmax which is the more direct driver).
    res["corr_c_t_free_argmax"] = _pearson(c_t, argmax.astype(np.float64))
    if miss10.sum() > 0 and miss10.sum() < len(miss10):
        res["AUC_lowc_predicts_posMiss10"] = _auc(miss10, 1.0 - c_t)
        res["mean_c_t_posMiss10"] = float(c_t[miss10 == 1].mean())
        res["mean_c_t_posHit"] = float(c_t[miss10 == 0].mean())
    else:
        res["AUC_lowc_predicts_posMiss10"] = None

    # ── interpretability: low-c examples — are they implausible? ──
    order = np.argsort(c_t)
    examples = []
    for i in order[:12]:
        examples.append({
            "c_t": round(float(c_t[i]), 4),
            "cc_weight": round(float(cc_w[i]), 4),
            "belief_argmax": STATE_NAMES[belief_arg[i]],
            "free_next_state": STATE_NAMES[argmax[i]],
            "reachable": bool(reach_hit[i]),
            "pos_sigmoid": round(float(pos_score[i]), 4),
        })
    return res, {"c_t": c_t, "cc_w": cc_w, "argmax": argmax, "reach_hit": reach_hit,
                 "belief_arg": belief_arg, "pos_sigmoid": pos_score,
                 "violation": violation, "never_born_death": never_born_death,
                 "low_c_examples": examples}


def train_arm(arm, dataset, seed, epochs, hidden, batch_size, lr, dump_dir,
              cc_self_consist_w=0.0, cc_grounded_init=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = download_dataset(dataset)
    splits = get_data_splits(data)
    num_nodes, feat_dim = data["num_nodes"], data["feat_dim"]

    seen_nodes = (set(splits["train"]["sources"]) | set(splits["train"]["destinations"])
                  | set(splits["val"]["sources"]) | set(splits["val"]["destinations"]))
    test_nodes = set(splits["test"]["sources"]) | set(splits["test"]["destinations"])
    inductive_nodes = sorted(test_nodes - seen_nodes)
    if len(inductive_nodes) < 10:
        inductive_nodes = None

    model = build_model(dataset, num_nodes, feat_dim, hidden, arm=arm,
                        cc_self_consist_w=cc_self_consist_w,
                        cc_grounded_init=cc_grounded_init)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n{'='*70}\nARM={arm}  dataset={dataset}  seed={seed}  ep={epochs}  "
          f"bs={batch_size}  inductive={'Y' if inductive_nodes else 'N'}\n{'='*70}")
    if hasattr(model, "reset"): model.reset()
    if hasattr(model, "set_epoch"): model.set_epoch(0)
    run_epoch(model, splits["train"], num_nodes, batch_size, optimizer=opt,
              desc=f"{dataset[:3]}/warmup")
    _dev_sync(); t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        if hasattr(model, "set_epoch"): model.set_epoch(ep)
        run_epoch(model, splits["train"], num_nodes, batch_size, optimizer=opt,
                  desc=f"{dataset[:3]}/E{ep}/tr")
        va = run_epoch(model, splits["val"], num_nodes, batch_size, desc=f"E{ep}/va")
        sched.step()
        if va["AP"] > best_val:
            best_val = va["AP"]
            best_state = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  E{ep:02d} va_AP={va['AP']:.4f} best={best_val:.4f} "
                  f"[{time.time()-t0:.0f}s]")
    if best_state is not None:
        model.load_state_dict(best_state)

    # standard transductive + inductive AP via run_epoch (canonical protocol)
    if hasattr(model, "reset"): model.reset()
    if hasattr(model, "set_epoch"): model.set_epoch(epochs)
    test_trans_std = run_epoch(model, splits["test"], num_nodes, batch_size,
                               desc="test_trans_std")
    test_ind = {"AP": float("nan"), "AUC": float("nan")}
    if inductive_nodes:
        if hasattr(model, "reset"): model.reset()
        run_epoch(model, splits["train"], num_nodes, batch_size, desc="warm_tr")
        run_epoch(model, splits["val"], num_nodes, batch_size, desc="warm_va")
        test_ind = run_epoch(model, splits["test"], num_nodes, batch_size,
                             inductive_nodes=inductive_nodes,
                             seen_nodes=sorted(seen_nodes), desc="test_ind")
    train_time = time.time() - t0

    # CALIBRATION test pass (fresh reset -> chronological replay -> collect cc fields)
    cat = calib_test_pass(model, splits["test"], num_nodes, batch_size, seed)
    cmetrics, perevent = calibration_metrics(cat)

    cmetrics["std_trans_ap"] = float(test_trans_std["AP"])
    cmetrics["std_trans_auc"] = float(test_trans_std["AUC"])
    cmetrics["ind_ap"] = float(test_ind["AP"])
    cmetrics["ind_auc"] = float(test_ind["AUC"])
    cmetrics["best_val_ap"] = float(best_val)
    cmetrics["train_time_s"] = round(train_time, 1)
    cmetrics["arm"] = arm
    cmetrics["nan_check"] = bool(np.isnan(test_trans_std["AP"]) or
                                 np.isnan(cmetrics["trans_ap"]))
    cmetrics["leak_check_ap_eq_1"] = bool(cmetrics["trans_ap"] >= 0.9999)

    # ── LEARNED obs-coupling weight w_obs = sigmoid(cc_w_obs_logit). Trained ONLY by
    #    the self-consistency aux CE. Init logit=0 => init w_obs=0.5. If converges ~0
    #    the obs term vanishes -> belief reverts to FIX-3 zero-hand (stuck) regime.
    cmetrics["cc_self_consist_w"] = float(cc_self_consist_w)
    cmetrics["cc_grounded_init"] = bool(getattr(model, "cc_grounded_init", False))
    if getattr(model, "cc_w_obs_logit", None) is not None:
        logit = float(model.cc_w_obs_logit.detach().cpu())
        cmetrics["cc_w_obs_logit"] = logit
        cmetrics["w_obs"] = float(1.0 / (1.0 + np.exp(-logit)))
    else:
        cmetrics["cc_w_obs_logit"] = None
        cmetrics["w_obs"] = None

    if perevent is not None and dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        npz = os.path.join(dump_dir, f"wc_conf_calib_{dataset}_{arm}_s{seed}.npz")
        np.savez_compressed(
            npz,
            c_t=perevent["c_t"], cc_weight=perevent["cc_w"],
            argmax_s_t1_cal=perevent["argmax"], reach_hit=perevent["reach_hit"],
            belief_argmax=perevent["belief_arg"], pos_sigmoid=perevent["pos_sigmoid"],
            violation=perevent["violation"], never_born_death=perevent["never_born_death"])
        cmetrics["npz"] = npz
        cmetrics["low_c_examples"] = perevent["low_c_examples"]
        print(f"  [dump] per-event calibration -> {npz}")
    return cmetrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="coedit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=None)
    p.add_argument("--dump_dir", default=str(AUDIT_DIR))
    # plumbing only: >0 enables LEARNABLE w_obs (self-consistency aux). default 0 = FIX-3.
    p.add_argument("--cc_self_consist_w", type=float, default=0.0)
    # GROUNDED belief init (PM 2026-06-07): seed walked-chain at softmax(s_t_pos) for a
    # pair's first appearance in the split instead of IDLE. wc arm only. Decisive test:
    # does belief argmax dist stop being stuck IDLE and does R2(c_t~free) drop off 0.98.
    p.add_argument("--cc_grounded_init", action="store_true")
    args = p.parse_args()

    out_json = args.out or os.path.join(
        AUDIT_DIR,
        f"wc_conf_calib_{args.dataset}_s{args.seed}_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    results = {}
    for arm in ("wc", "B"):
        # self-consistency learnable w_obs applies to the wc arm only (B has no cc).
        scw = args.cc_self_consist_w if arm == "wc" else 0.0
        cgi = args.cc_grounded_init if arm == "wc" else False
        results[arm] = train_arm(arm, args.dataset, args.seed, args.epochs,
                                 args.hidden, args.batch, args.lr, args.dump_dir,
                                 cc_self_consist_w=scw, cc_grounded_init=cgi)
        _dev_sync()

    summary = {
        "dataset": args.dataset, "seed": args.seed, "epochs": args.epochs,
        "batch": args.batch, "protocol": "design=correct_decoupled,fsm v3+hier+"
        "decol_hier_v2+causal_batch,let0.5,p0_fix=off,bs500",
        "arms": results,
    }
    # headline comparison
    wc, B = results["wc"], results["B"]
    summary["headline"] = {
        "wc_trans_ap": wc["trans_ap"], "B_trans_ap": B["trans_ap"],
        "delta_trans_ap_pp": round((wc["trans_ap"] - B["trans_ap"]) * 100, 3),
        "wc_ind_ap": wc["ind_ap"], "B_ind_ap": B["ind_ap"],
        "delta_ind_ap_pp": round((wc["ind_ap"] - B["ind_ap"]) * 100, 3),
        "wc_AUC_lowc_violation": wc.get("AUC_lowc_predicts_violation"),
        "wc_AUC_lowc_posMiss10": wc.get("AUC_lowc_predicts_posMiss10"),
        "wc_violation_rate": wc.get("violation_rate"),
        "wc_cc_self_consist_w": wc.get("cc_self_consist_w"),
        "wc_w_obs_converged": wc.get("w_obs"),
        "wc_cc_w_obs_logit": wc.get("cc_w_obs_logit"),
        "wc_pos_lifecycle_dist": wc.get("pos_lifecycle_dist"),
        "any_leak_ap_eq_1": bool(wc["leak_check_ap_eq_1"] or B["leak_check_ap_eq_1"]),
        "any_nan": bool(wc["nan_check"] or B["nan_check"]),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] -> {out_json}")
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
