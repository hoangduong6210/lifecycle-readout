"""cf_trained_theta_battery.py — ANALYSIS 2 (reviewer tautology #4).

The do(state) ladder DEATH<IDLE<DECAY<REINF=BIRTH is, in fsm_intervene.py, read off
the HARDCODED init spec EXISTENCE_W=[0.1,1,1,0.3,0]. Reviewer: that is the designer's
chosen init order — every counterfactual fraction trivially follows it. To prove the
ordering is a TRAINED property, we re-run the do(state) battery with the per-seed
TRAINED existence-decoder weights w_s = softplus(existence_decoder.theta), serialized
from the best-val config-B checkpoint by run_faithfulness_eval.py (sidecar _theta.json).

For each seed we report:
  * w_s_trained (the 5 trained weights) vs w_s_init (the hardcoded spec)
  * do(state=X) p_edge for every X∈{DEATH,IDLE,DECAY,REINFORCE,BIRTH} using w_s_trained
  * whether the LADDER  DEATH < IDLE < DECAY < {REINFORCE ≈ BIRTH}  HOLDS with trained w_s
3-seed aggregate (mean±std n−1) of each do(state) p_edge + ladder-held count.

NO fabrication: every number derives from the real npz + the real trained theta JSON.
Run (CPU): python cf_trained_theta_battery.py [--seeds 42,1,7] [--out <path>]
"""
import os, sys, json, argparse
import numpy as np
from project_paths import AUDIT_DIR, LEGACY_RESULTS_DIR

V33 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V33)
import fsm_intervene as fi

STATES = ["DEATH", "IDLE", "DECAY", "REINFORCE", "BIRTH"]  # ascending ladder order
IDX = {"IDLE": 0, "BIRTH": 1, "REINFORCE": 2, "DECAY": 3, "DEATH": 4}


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": m, "std": s, "per_seed": vals}


def ladder_holds(pe_by_state, eq_tol=0.02):
    """pe_by_state: dict state->p_edge (population mean). Ladder:
    DEATH < IDLE < DECAY < REINFORCE and DEATH<IDLE<DECAY<BIRTH and
    REINFORCE ≈ BIRTH (|Δ|<=eq_tol). Returns (bool, detail)."""
    d, i, c = pe_by_state["DEATH"], pe_by_state["IDLE"], pe_by_state["DECAY"]
    r, b = pe_by_state["REINFORCE"], pe_by_state["BIRTH"]
    strict = (d < i) and (i < c) and (c < r) and (c < b)
    eq_rb = abs(r - b) <= eq_tol
    return bool(strict and eq_rb), {
        "DEATH<IDLE": bool(d < i), "IDLE<DECAY": bool(i < c),
        "DECAY<REINFORCE": bool(c < r), "DECAY<BIRTH": bool(c < b),
        "REINFORCE≈BIRTH": bool(eq_rb), "|REINF-BIRTH|": abs(r - b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--resdir", default=str(LEGACY_RESULTS_DIR))
    ap.add_argument("--npz_tmpl",
                    default="faithfulness_coedit_v3_hier_hv2_cb_hcp_let0.5_s{seed}.npz",
                    help="npz filename template (config-B faithfulness dump)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = args.out or os.path.join(AUDIT_DIR, "cf_trained_theta_3seed.json")
    out_path = os.path.abspath(out_path)

    result = {"seeds": seeds, "states_ladder_order": STATES,
              "w_s_init_spec": [0.1, 1.0, 1.0, 0.3, 0.0],
              "per_seed": {}, "summary": {}}

    for s in seeds:
        npz = os.path.join(args.resdir, args.npz_tmpl.format(seed=s))
        theta_json = os.path.splitext(npz)[0] + "_theta.json"
        rec = {"npz": os.path.basename(npz), "theta_json": os.path.basename(theta_json)}
        if not os.path.exists(npz) or not os.path.exists(theta_json):
            rec["missing"] = [p for p in (npz, theta_json) if not os.path.exists(p)]
            result["per_seed"][str(s)] = rec
            continue
        tj = json.load(open(theta_json))
        w_trained = np.asarray(tj["w_s_trained"], dtype=np.float64)  # [IDLE,BIRTH,REINF,DECAY,DEATH]
        rec["w_s_trained"] = w_trained.tolist()
        rec["best_val_ap"] = tj.get("best_val_ap")

        scm = fi.HierV2SCM(npz)
        # do(state=X) population-mean p_edge with TRAINED w, and with INIT w (contrast)
        pe_tr, pe_init = {}, {}
        for st in STATES:
            r_tr = scm.do_state(state=st, w=w_trained)
            r_in = scm.do_state(state=st)            # default = init EXISTENCE_W
            pe_tr[st] = float(np.mean(r_tr["p_edge_forced"]))
            pe_init[st] = float(np.mean(r_in["p_edge_forced"]))
        held_tr, detail_tr = ladder_holds(pe_tr)
        held_in, detail_in = ladder_holds(pe_init)
        rec["p_edge_trained"] = pe_tr
        rec["p_edge_init"] = pe_init
        rec["ladder_holds_trained"] = held_tr
        rec["ladder_detail_trained"] = detail_tr
        rec["ladder_holds_init"] = held_in
        result["per_seed"][str(s)] = rec
        print(f"[seed {s}] w_s_trained={['%.3f'%x for x in w_trained]}  "
              f"ladder_trained={'HOLD' if held_tr else 'BREAK'}  "
              f"p_edge(D/I/De/R/B)="
              f"{pe_tr['DEATH']:.3f}/{pe_tr['IDLE']:.3f}/{pe_tr['DECAY']:.3f}/"
              f"{pe_tr['REINFORCE']:.3f}/{pe_tr['BIRTH']:.3f}")

    # 3-seed aggregate
    valid = [s for s in seeds if "p_edge_trained" in result["per_seed"].get(str(s), {})]
    if valid:
        for st in STATES:
            result["summary"][f"p_edge_trained_{st}"] = _mean_std(
                [result["per_seed"][str(s)]["p_edge_trained"][st] for s in valid])
        n_hold = sum(result["per_seed"][str(s)]["ladder_holds_trained"] for s in valid)
        result["summary"]["ladder_holds_trained_count"] = f"{n_hold}/{len(valid)}"
        result["summary"]["w_s_trained_mean"] = _mean_std_vec(
            [result["per_seed"][str(s)]["w_s_trained"] for s in valid])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== cf_trained_theta_3seed SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved -> {out_path}")


def _mean_std_vec(vecs):
    a = np.asarray(vecs, dtype=np.float64)
    return {"mean": a.mean(0).tolist(),
            "std": (a.std(0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])).tolist(),
            "order": ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]}


if __name__ == "__main__":
    main()
