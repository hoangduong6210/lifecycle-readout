"""
FAITHFULNESS eval driver (coedit, fsm_arch=v3, lambda_edge_trans=0.5, seed 42).

WHAT IT DOES (one self-contained job for evaluation harness; GPU):
  1. Train SRGNN_v3_3 on coedit at the sweet-spot config below (same scaffold as
     run_v3_3_benchmark.run_one: warmup + N epochs + best-val checkpoint).
  2. Reload best-val state, ARM the eval-only faithfulness probe, and run ONE
     transductive TEST forward pass → per-event .npz (z_pair, n_prior, true_occ,
     argmax_s_t1_pos, p_idle/birth/reinforce/decay/death, hawkes_lam).
  3. Print the test AP/AUC (so the dump-armed pass is auditable vs a normal run).

The probe is pure logging (CPU-verified byte-identical AP/state_dist with it
on/off; see verification record 2026-06-01). Offline analysis:
    python analyze_faithfulness.py <out.npz>

Run (evaluation harness, GPU node):
    python run_faithfulness_eval.py \
        --dataset coedit --seed 42 --epochs 20 \
        --fsm_arch v3 --lambda_edge_trans 0.5 \
        --out results/faithfulness_coedit_v3_let0.5_s42.npz
"""
import os, sys, time, json, random, argparse
import numpy as np
import torch

V33_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(V33_DIR)
EXP_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, V33_DIR)

from lifecycle_readout.datasets import download_dataset, get_data_splits
from lifecycle_readout.training import run_epoch, DEVICE, _dev_sync
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3
from project_paths import AUDIT_DIR


def run_epoch_v33(model, split_data, num_nodes, batch_size, optimizer=None,
                  inductive_nodes=None, seen_nodes=None, desc="train", epoch=0):
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    return run_epoch(model, split_data, num_nodes, batch_size,
                     optimizer=optimizer, inductive_nodes=inductive_nodes,
                     seen_nodes=seen_nodes, desc=desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coedit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fsm_arch", default="v3")
    ap.add_argument("--fsm_decode", default="flat", choices=["flat", "hier"],
                    help="state readout: 'flat' (5-class softmax) or 'hier' "
                         "(hierarchical tree decode; fsm_arch=v3 only)")
    ap.add_argument("--lambda_edge_trans", type=float, default=0.5)
    ap.add_argument("--design", default="correct")
    ap.add_argument("--decol_hier_v2", action="store_true",
                    help="HIER v2 gate priors (round-7 fix): anchor p_alive/p_rising on "
                         "true_occ (reliable) instead of the intra-batch-corrupted "
                         "rate/Welford signals; REINFORCE is the alive-branch default, "
                         "DEATH only on TRUSTWORTHY sustained silence. fsm_decode=hier only.")
    ap.add_argument("--causal_batch", action="store_true",
                    help="CAUSAL intra-batch accumulation (P1 fix): fold repeated same-"
                         "pair events WITHIN a batch in stream order so Welford n / μ / "
                         "var / Hawkes λ / rate fast-slow-peak match an event-by-event "
                         "reference (legacy snapshots once/batch ⇒ n caps, rate pinned).")
    ap.add_argument("--hier_causal_policy", action="store_true",
                    help="apply the causal policy (ever_alive DEATH gate + soft "
                         "expected-admissibility C-mask) to the PUBLISHED state "
                         "s_t1_cal. Default OFF = byte-identical hier behavior. "
                         "AP path (s_t1_pos) is untouched ⇒ AP Δ=0.")
    ap.add_argument("--strict_ordered_fsm", action="store_true",
                    help="HƯỚNG A: re-decode the PUBLISHED state as the strict-ordered "
                         "6-state lifecycle PRE_BIRTH/BIRTH/REINFORCE/DECAY/DORMANT/"
                         "DEATH (IDLE split so 'ever-alive' is in-state, Markovian) "
                         "constrained by a band-diagonal C' (|i−j|≤1 only — every "
                         "non-adjacent jump hard-masked). Makes the ever_alive gate "
                         "redundant. Requires --hier_causal_policy + --fsm_decode hier. "
                         "AP path untouched ⇒ AP Δ=0. Emits s_t1_cal6 / argmax_s_t1_cal6.")
    ap.add_argument("--strict_ordered_5state", action="store_true",
                    help="OPTION (b): keep the NATIVE 5-state lifecycle "
                         "IDLE/BIRTH/REINFORCE/DECAY/DEATH and HARD-mask the PUBLISHED "
                         "state s_t1_cal with the strict band-diagonal C_BAND_5 (|i−j|≤1 "
                         "only — every non-adjacent jump zeroed+renormed). IDLE(0) and "
                         "DEATH(4) are axis ends so IDLE→DEATH is band-blocked ⇒ NO IDLE "
                         "split, NO ever_alive gate. Requires --hier_causal_policy + "
                         "--fsm_decode hier. Mutually exclusive with --strict_ordered_fsm "
                         "(6-state). AP path untouched ⇒ AP Δ=0. npz tag _so5.")
    # ── LFG gradient-mode override (minimal passthrough, mirrors run_v3_3_benchmark).
    #    Lets evaluation harness dump argmax_s_t1_cal + ever_alive PER LFG-arm (HARD/SOFT/NONE)
    #    so the DEATH-before-alive C-violation rate (item 4) reflects each arm. None ⇒
    #    model/preset default (byte-identical to before). Detached arm; does NOT flip
    #    enable_main_predictor (faith driver never sets it ⇒ stays False).
    ap.add_argument("--lfg_mode", choices=["soft", "hard"], default=None,
                    help="LFG gradient mode: hard=causal HARD gate; soft=compliance "
                         "ramp; None=model default (correct_decoupled→soft).")
    ap.add_argument("--compliance_floor", type=float, default=None,
                    help="per-event gradient weight for causally-impossible positives "
                         "under lfg_mode=hard (0.0=full hard gate). None=model default.")
    ap.add_argument("--lfg", choices=["on", "off"], default="on",
                    help="LFG reweighting: on=canonical; off=uniform weight=1 (ARM-NONE).")
    ap.add_argument("--out", default=None,
                    help="output .npz path for the per-event faithfulness dump; "
                         "None ⇒ auto-named from config (see below).")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(
            AUDIT_DIR,
            f"faithfulness_{args.dataset}_{args.fsm_arch}_{args.fsm_decode}"
            f"{'_hv2' if args.decol_hier_v2 else ''}"
            f"{'_so6' if args.strict_ordered_fsm else ''}"
            f"{'_so5' if args.strict_ordered_5state else ''}"
            f"_let{args.lambda_edge_trans}_s{args.seed}"
            f"{'_cbON' if args.causal_batch else ''}"
            f"{'_hcpON' if args.hier_causal_policy else ''}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = download_dataset(args.dataset)
    splits = get_data_splits(data)
    num_nodes, feat_dim = data["num_nodes"], data["feat_dim"]

    # LFG gradient-mode override (only pass when CLI set ⇒ overrides the model default;
    # correct_decoupled does NOT pin lfg_mode/floor, so these are honored verbatim).
    _lfg_kw = {}
    if args.lfg_mode is not None:
        _lfg_kw["lfg_mode"] = args.lfg_mode
    if args.compliance_floor is not None:
        _lfg_kw["compliance_floor"] = args.compliance_floor
    _lfg_kw["enable_lfg"] = (args.lfg == "on")
    model = SRGNN_v3_3(num_nodes, feat_dim, args.hidden, device=DEVICE,
                       design=args.design, fsm_arch=args.fsm_arch,
                       fsm_decode=args.fsm_decode,
                       decol_hier_v2=args.decol_hier_v2,
                       causal_batch=args.causal_batch,
                       hier_causal_policy=args.hier_causal_policy,
                       strict_ordered_fsm=args.strict_ordered_fsm,
                       strict_ordered_5state=args.strict_ordered_5state,
                       decollapse_target=True,
                       lambda_edge_trans=args.lambda_edge_trans,
                       **_lfg_kw).to(DEVICE)
    print(f"[faith][lfg] mode={getattr(model,'lfg_mode',None)} "
          f"floor={getattr(model,'compliance_floor',None)} "
          f"enable_lfg={getattr(model,'enable_lfg',None)} "
          f"enable_main_predictor={getattr(model,'enable_main_predictor',None)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[faith] {args.dataset} seed={args.seed} epochs={args.epochs} "
          f"fsm_arch={args.fsm_arch} lambda_edge_trans={args.lambda_edge_trans} "
          f"design={args.design}")

    # warmup (untimed)
    if hasattr(model, "reset"): model.reset()
    run_epoch_v33(model, splits["train"], num_nodes, args.batch_size,
                  optimizer=optimizer, desc="warmup", epoch=0)

    best_val_ap, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        if hasattr(model, "reset"): model.reset()
        tr = run_epoch_v33(model, splits["train"], num_nodes, args.batch_size,
                           optimizer=optimizer, desc=f"E{ep}/tr", epoch=ep)
        va = run_epoch_v33(model, splits["val"], num_nodes, args.batch_size,
                           desc=f"E{ep}/va", epoch=ep)
        scheduler.step()
        if va["AP"] > best_val_ap:
            best_val_ap = va["AP"]
            best_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"  E{ep:02d} tr_AP={tr['AP']:.4f} va_AP={va['AP']:.4f}")

    # ── DUMP-ARMED transductive TEST forward (single pass) ────────────────────
    if hasattr(model, "reset"): model.reset()
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── SERIALIZE TRAINED EXISTENCE-DECODER theta (reviewer tautology #4) ──────
    # The do(state) ladder DEATH<IDLE<DECAY<REINF=BIRTH is currently read off the
    # HARDCODED init spec EXISTENCE_W=[0.1,1,1,0.3,0] (fsm_intervene.py). To prove the
    # ordering is a TRAINED property (not just the designer's init order), dump the
    # per-seed TRAINED weights w_s = softplus(existence_decoder.theta) from the
    # best-val checkpoint so the counterfactual battery can re-run do(state) with the
    # REAL theta. Sidecar JSON next to the npz; cheap, eval-only, no model change.
    try:
        import torch.nn.functional as _F
        _theta = best_state["existence_decoder.theta"].detach().float().cpu()
        _w_s = _F.softplus(_theta).numpy().tolist()
        _theta_json = os.path.splitext(args.out)[0] + "_theta.json"
        with open(_theta_json, "w") as _tf:
            json.dump({
                "seed": args.seed, "dataset": args.dataset,
                "theta": _theta.numpy().tolist(),
                "w_s_trained": _w_s,                      # softplus(theta), the REAL weights
                "w_s_init_spec": [0.1, 1.0, 1.0, 0.3, 0.0],  # hardcoded EXISTENCE_W init
                "state_names": ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"],
                "best_val_ap": float(best_val_ap),
            }, _tf, indent=2)
        print(f"[faith] trained existence theta -> {_theta_json}  "
              f"w_s={['%.4f' % x for x in _w_s]}")
    except Exception as _e:
        print(f"[faith][WARN] could not serialize trained theta: {_e}")

    model.enable_faithfulness_dump(args.out)
    test = run_epoch_v33(model, splits["test"], num_nodes, args.batch_size,
                         desc="test/faith", epoch=args.epochs)
    written = model.flush_faithfulness()

    print(f"[faith] test AP={test['AP']:.4f} AUC={test['AUC']:.4f}")
    print(f"[faith] dump -> {written}")
    if written:
        d = np.load(written)
        n = len(d["z_pair"])
        am = d["argmax_s_t1_pos"]
        dist = np.bincount(am, minlength=5) / max(n, 1)
        print(f"[faith] n_events={n}  RAW argmax[I,B,R,D,Dt]="
              + "[" + ", ".join(f"{x:.3f}" for x in dist) + "]")
        # Calibrated argmax (post per-class bias) — the gate's committed symbolic state.
        # Present only when the run carried the argmax-bias calibration (v3 PM-config);
        # == RAW otherwise. Shows whether DECAY/DEATH now win argmax after calibration.
        if "argmax_s_t1_cal" in d.files:
            amc = d["argmax_s_t1_cal"]
            distc = np.bincount(amc, minlength=5) / max(n, 1)
            print(f"[faith]            CAL argmax[I,B,R,D,Dt]="
                  + "[" + ", ".join(f"{x:.3f}" for x in distc) + "]")
            # n_prior>=2 subset (where frequency/dynamics is meaningful — PM item 3)
            mask2 = d["n_prior"] >= 2.0
            if mask2.sum() > 0:
                dc2 = np.bincount(amc[mask2], minlength=5) / int(mask2.sum())
                print(f"[faith]   n_prior>=2 ({int(mask2.sum())}) CAL argmax="
                      + "[" + ", ".join(f"{x:.3f}" for x in dc2) + "]")
    print("[faith] analyze with: python analyze_faithfulness.py", written)


if __name__ == "__main__":
    main()
