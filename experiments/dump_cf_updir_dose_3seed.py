"""
Serialize the previously-underived counterfactual quantities to a committed JSON,
so every number in Section 4 / Figure 2 / Section 6.4 is independently
reproducible from a dumped artifact (closes paper items 8.2-iv, 8.2-v(partial),
9.2 dose-signs, 9.2 faithfulness-traceability).

All quantities are computed OFFLINE from the config-B cbON faithfulness npz
(no GPU, no re-train, no leak): the HierV2SCM engine reconstructs the existence
readout from EXISTENCE_W (the effective, init-equivalent softplus(theta) target;
see fsm_intervene.py docstring) and re-decodes the per-event SCM.

What this DOES serialize (newly):
  * up-direction do(REINFORCE)/do(BIRTH) per-seed fractions of pairs whose
    P(edge) moves UP  -> closes 8.2-iv (was reported only as a qualitative ordering)
  * the existence-readout one-hot ladder w[state] (the monotone structure that
    carries the do(state) ordering) -> closes the Figure 2 ordering as a measured
    structural constant, not a hand-set schematic
  * dose-response regression SIGNS (rate->REINFORCE +, rate->DEATH -,
    slope->REINFORCE +, true_occ->BIRTH -) with per-seed slopes -> closes 9.2 dose
  * 3-seed hier vs flat DECAY-argmax fractions + Spearman(rho) -> closes 9.2
    faithfulness traceability (was seed-42 only)

What this canNOT serialize (kept honest, listed as remaining work in 8.2-v):
  * trained per-seed existence-decoder theta -- NOT saved to disk by the eval
    driver (no checkpoints). EXISTENCE_W is the init-equivalent constant, so the
    ORDERING and SIGN are exact, but the deployed trained MAGNITUDES are not
    recoverable. Figure 2 therefore stays ordinal for magnitudes; only the
    ordering/sign/reversibility are claimed.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from project_paths import AUDIT_DIR, result_input

sys.path.insert(0, ".")
from fsm_intervene import (BIRTH, DEATH, DECAY, EXISTENCE_W, HierV2SCM,
                           REINFORCE, STATE_NAMES)

SEEDS = [42, 1, 7]
HIER = {s: str(result_input(f"faithfulness_coedit_v3_hier_hv2_let0.5_s{s}_cbON.npz")) for s in SEEDS}
FLAT = {s: str(result_input(f"faithfulness_coedit_v3_flat_let0.5_s{s}.npz")) for s in SEEDS}


def _sstd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def _frac(dist, k):
    return float(dist[:, k].mean())


def _slope(x, y):
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


# ---- 1. existence-readout one-hot ladder (structural, seed-independent) -------
ladder = {}
for nm in STATE_NAMES:
    oh = np.zeros((1, 5))
    oh[0, STATE_NAMES.index(nm)] = 1.0
    ladder[nm] = float((oh * EXISTENCE_W[None, :]).sum())

per_seed = {}
for s in SEEDS:
    m = HierV2SCM(HIER[s])
    base = m.baseline()
    rec = np.where(m.true_occ >= 2.0)[0]

    # up-direction do(state) fractions: P(edge) moves UP
    up_reinf = m.do_state(state="REINFORCE")
    up_birth = m.do_state(state="BIRTH")
    down_death = m.do_state(state="DEATH")

    # dose-response regression slopes (sign is the claim) -- recurring subset
    grid_rr = np.linspace(-4, 4, 9)
    pre = [_frac(m.do_driver(rec, rate_ratio=v)["dist"], REINFORCE) for v in grid_rr]
    pdt = [_frac(m.do_driver(rec, rate_ratio=v)["dist"], DEATH) for v in grid_rr]
    grid_sl = np.linspace(-1.0, 1.0, 9)
    sre = [_frac(m.do_driver(rec, slope_rel=v)["dist"], REINFORCE) for v in grid_sl]
    grid_to = np.array([1, 2, 3, 5, 10, 20, 50], float)
    obi = [_frac(m.do_driver(slice(None), true_occ=v)["dist"], BIRTH) for v in grid_to]

    # faithfulness DECAY-argmax (hier) and flat counterpart + spearman
    d = np.load(HIER[s])
    cal = d["argmax_s_t1_cal"]
    recm = d["true_occ"] >= 2.0
    rho, pval = spearmanr(d["p_decay_cal"][recm], d["slope_rel"][recm])
    df = np.load(FLAT[s])
    calf = df["argmax_s_t1_cal"]

    per_seed[str(s)] = {
        "do_REINFORCE_frac_pedge_up": float((up_reinf["delta"] > 0).mean()),
        "do_BIRTH_frac_pedge_up": float((up_birth["delta"] > 0).mean()),
        "do_DEATH_frac_pedge_down": float((down_death["delta"] < 0).mean()),
        "dose_rate_to_REINFORCE_slope": _slope(grid_rr, pre),
        "dose_rate_to_DEATH_slope": _slope(grid_rr, pdt),
        "dose_slope_to_REINFORCE_slope": _slope(grid_sl, sre),
        "dose_trueocc_to_BIRTH_slope": _slope(grid_to, obi),
        "hier_DECAY_argmax_frac": float((cal == DECAY).mean()),
        "hier_DECAY_argmax_count": int((cal == DECAY).sum()),
        "flat_DECAY_argmax_frac": float((calf == DECAY).mean()),
        "flat_DECAY_argmax_count": int((calf == DECAY).sum()),
        "n": int(len(cal)),
        "n_recurring": int(recm.sum()),
        "spearman_pdecaycal_sloperel_rho": float(rho),
        "spearman_pdecaycal_sloperel_p": float(pval),
    }


def _agg(key):
    v = [per_seed[str(s)][key] for s in SEEDS]
    return {"mean": float(np.mean(v)), "std": _sstd(v), "per_seed": v}


out = {
    "source_npz_hier": HIER,
    "source_npz_flat": FLAT,
    "seeds": SEEDS,
    "method": "offline HierV2SCM over config-B cbON npz; EXISTENCE_W is the "
              "init-equivalent softplus(theta) target (trained theta NOT serialized; "
              "ordering/sign exact, magnitudes ordinal)",
    "existence_readout_onehot_ladder": ladder,
    "ladder_ordering": "DEATH(0.0) < IDLE(0.1) < DECAY(0.3) < BIRTH=REINFORCE(1.0)",
    "per_seed": per_seed,
    "summary": {
        "do_REINFORCE_frac_pedge_up": _agg("do_REINFORCE_frac_pedge_up"),
        "do_BIRTH_frac_pedge_up": _agg("do_BIRTH_frac_pedge_up"),
        "do_DEATH_frac_pedge_down": _agg("do_DEATH_frac_pedge_down"),
        "dose_rate_to_REINFORCE_slope": _agg("dose_rate_to_REINFORCE_slope"),
        "dose_rate_to_DEATH_slope": _agg("dose_rate_to_DEATH_slope"),
        "dose_slope_to_REINFORCE_slope": _agg("dose_slope_to_REINFORCE_slope"),
        "dose_trueocc_to_BIRTH_slope": _agg("dose_trueocc_to_BIRTH_slope"),
        "hier_DECAY_argmax_frac": _agg("hier_DECAY_argmax_frac"),
        "flat_DECAY_argmax_frac": _agg("flat_DECAY_argmax_frac"),
        "spearman_pdecaycal_sloperel_rho": _agg("spearman_pdecaycal_sloperel_rho"),
    },
}

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
with (AUDIT_DIR / "cf_updir_dose_faith_3seed.json").open("w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["summary"], indent=2))
print("ladder:", ladder)
