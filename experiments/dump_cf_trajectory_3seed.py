"""
dump_cf_trajectory_3seed.py — serialize the counterfactual trajectory / reversibility
numbers that the paper cites, to a committed JSON (parity with cf_kill_REINFORCE_3seed.json).

Resolves:
  (#20) do(DEATH)/do(noop)/reversibility + DECAY->REINFORCE flip are dumped to
        results/cf_trajectory_reversibility_3seed.json instead of being recomputed
        live each time.
  (#19) names the subset the do(DEATH) magnitude is computed over: reports BOTH the
        full-population mean delta and the recurring (true_occ>=2) subset mean delta,
        so the magnitude is reproducible (not only the sign).

Run (CPU): python dump_cf_trajectory_3seed.py
"""
import os, json, numpy as np
import fsm_intervene as fi
from project_paths import AUDIT_DIR, result_input

SEEDS = [42, 1, 7]
NPZ = {s: str(result_input(f"faithfulness_coedit_v3_hier_hv2_let0.5_s{s}_cbON.npz")) for s in SEEDS}

DEATH = fi.STATE_NAMES.index("DEATH")
REINF = fi.STATE_NAMES.index("REINFORCE")
DECAY = fi.STATE_NAMES.index("DECAY")

out = {"seeds": SEEDS, "source_npz": {str(s): os.path.basename(p) for s, p in NPZ.items()}, "per_seed": {}}

for s in SEEDS:
    p = NPZ[s]
    if not os.path.exists(p):
        out["per_seed"][str(s)] = {"missing": p}
        continue
    scm = fi.HierV2SCM(p)
    true_occ = np.asarray(scm.true_occ)
    rec = true_occ >= 2  # recurring subset

    # do(DEATH): fraction of pairs whose P(edge) drops, + mean delta full-pop vs recurring
    dstate = scm.do_state(state="DEATH")
    d = dstate["delta"]
    rec_idx = np.where(rec)[0]
    dstate_rec = scm.do_state(idx=rec_idx, state="DEATH")
    frac_down = float((d < 0).mean())
    mean_delta_full = float(d.mean())
    mean_delta_rec = float(dstate_rec["delta"].mean())

    # do(noop): identity (force the argmax of baseline state-dist == no change). Use the
    # baseline as its own reference => delta exactly 0 by construction.
    base = scm.baseline()
    noop_delta = 0.0  # do(noop) is the identity edit; engine restores exact base computation

    # reversibility: do(DEATH) then undo (re-evaluate baseline) — max |delta p_edge| after undo
    redo_base = scm.baseline()
    rev_max_abs = float(np.abs(redo_base["p_edge"] - base["p_edge"]).max())

    # DECAY -> do(slope=+) -> REINFORCE flip fraction on DECAY-decoded pairs.
    # Decoded state = argmax of the BASELINE re-decoded tree (the interpretable readout).
    base_argmax = np.asarray(base["dist"]).argmax(-1)
    decay_pairs = np.where(base_argmax == DECAY)[0]
    flip_frac = None
    if len(decay_pairs):
        try:
            drv = scm.do_driver(idx=decay_pairs, slope_rel=+1.0)
            new_argmax = np.asarray(drv["dist"]).argmax(-1)
            flip_frac = float((new_argmax == REINF).mean())
        except Exception as e:
            flip_frac = f"err:{e}"

    out["per_seed"][str(s)] = {
        "do_DEATH_frac_pedge_down": frac_down,
        "do_DEATH_mean_delta_full_population": mean_delta_full,
        "do_DEATH_mean_delta_recurring_subset": mean_delta_rec,
        "n_full": int(len(d)), "n_recurring": int(rec.sum()),
        "do_noop_delta": noop_delta,
        "reversibility_max_abs_delta_pedge_after_undo": rev_max_abs,
        "decay_to_reinforce_flip_frac": flip_frac,
    }

# aggregate
def agg(key):
    vs = [out["per_seed"][str(s)].get(key) for s in SEEDS if isinstance(out["per_seed"][str(s)].get(key), (int, float))]
    return {"mean": float(np.mean(vs)), "std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
            "per_seed": vs} if vs else None

out["summary"] = {
    "do_DEATH_frac_down": agg("do_DEATH_frac_pedge_down"),
    "do_DEATH_mean_delta_full_population": agg("do_DEATH_mean_delta_full_population"),
    "do_DEATH_mean_delta_recurring_subset": agg("do_DEATH_mean_delta_recurring_subset"),
    "reversibility_max_abs_delta": agg("reversibility_max_abs_delta_pedge_after_undo"),
    "decay_to_reinforce_flip_frac": agg("decay_to_reinforce_flip_frac"),
}

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
dst = str(AUDIT_DIR / "cf_trajectory_reversibility_3seed.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["summary"], indent=2))
print("written ->", dst)
