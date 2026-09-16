"""Counterfactual BATTERY — causal_batch A/B (ON vs OFF), single seed.

Reads the two faithfulness npz dumps produced this session (HV2 decode,
let=0.5, seed 42), causal_batch ON vs OFF, and runs the SAME battery as
_battery_tmp.py (dose-response / sign / trajectory-CF / existence-CF /
reversibility) so we can see whether the P1 fix unlocks the two SHALLOW axes
(slope_rel, staleness) on REAL coedit pairs.

Honest caveat carried from fsm_intervene.py docstring: stale_rel is absorbed
into the frozen r_obs at eval time, so the staleness dose-response is a
SYNTHETIC SCM probe, not a real-pair separable axis. slope_rel & rate_ratio
ARE exactly recoverable from the npz → those CFs are faithful on real pairs.
"""
import os
from pathlib import Path
import numpy as np
from fsm_intervene import (HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESDIR = Path(os.environ.get(
    "LIFECYCLE_RESULTS_DIR",
    PROJECT_ROOT / "results" / "historical" / "legacy_import",
))
ARMS = {
    "OFF": RESDIR / "faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbOFF.npz",
    "ON":  RESDIR / "faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz",
}
np.set_printoptions(precision=4, suppress=True)


def mono(x):
    dx = np.diff(x)
    up = (dx >= -1e-9).all(); dn = (dx <= 1e-9).all()
    return "UP-mono" if up and not dn else "DN-mono" if dn and not up else \
        "FLAT" if np.abs(dx).max() < 1e-4 else "non-mono"


def frac(d, s):
    return (d.argmax(1) == s).mean()


for arm, path in ARMS.items():
    print("=" * 78)
    print(f"ARM causal_batch={arm}  npz={path}")
    try:
        m = HierV2SCM(path)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        continue
    b = m.baseline()
    print(f"  N={m.N}")
    print("  baseline argmax marginal: " +
          " ".join(f"{STATE_NAMES[i]}={np.bincount(b['dist'].argmax(1),minlength=5)[i]/m.N:.3f}"
                   for i in range(1, 5)))

    rec = np.where(m.true_occ >= 2.0)[0]
    print(f"  recurring(true_occ>=2) n={len(rec)}/{m.N}")
    print("  recurring argmax marginal: " +
          " ".join(f"{STATE_NAMES[i]}={frac(b['dist'][rec], i):.3f}" for i in range(1, 5)))

    # ── P1 HEALTH: n_prior (Welford count) vs true_occ, mu_pair/var degeneracy ──
    print("\n  [P1] STORE-STAT HEALTH:")
    npri = m.n_prior
    to = m.true_occ
    print(f"    n_prior  : min={npri.min():.1f} max={npri.max():.1f} "
          f"mean={npri.mean():.3f}  (legacy cap was ~6)")
    print(f"    true_occ : min={to.min():.1f} max={to.max():.1f} mean={to.mean():.3f}")
    rmask = to >= 2.0
    if rmask.any():
        agree = np.mean(np.minimum(npri[rmask], to[rmask]) /
                        np.maximum(npri[rmask], to[rmask] + 1e-9))
        print(f"    n_prior/true_occ ratio (recurring mean min/max) = {agree:.3f} "
              f"(1.0 = perfect match)")
        print(f"    frac recurring with n_prior>=2 = {(npri[rmask] >= 2).mean():.3f}")
    # slope_rel population (the shallow axis the fix should de-pin)
    sr = m.slope_rel
    print(f"    slope_rel: min={sr.min():+.3f} max={sr.max():+.3f} mean={sr.mean():+.3f}")
    print(f"    slope_rel RISING frac (>0) = {(sr > 0).mean():.3f} "
          f"(legacy was ~0 → all <=0)")
    if hasattr(m, "stale_rel0"):
        st = m.stale_rel0
        print(f"    stale_rel: min={st.min():.3f} max={st.max():.3f} mean={st.mean():.3f} "
              f"(absorbed into r_obs → SYNTHETIC axis)")

    # ── 1. DOSE-RESPONSE ──
    print("\n  [1] DOSE-RESPONSE (recurring subset):")
    grid_sl = np.linspace(-1.0, 1.0, 9)
    sre = [frac(m.do_driver(rec, slope_rel=v)['dist'], REINFORCE) for v in grid_sl]
    sde = [frac(m.do_driver(rec, slope_rel=v)['dist'], DECAY) for v in grid_sl]
    print(f"    slope_rel grid {grid_sl}")
    print(f"      P(REINFORCE): {np.array(sre)}  [{mono(sre)}]  EXPECT UP")
    print(f"      P(DECAY)    : {np.array(sde)}  [{mono(sde)}]  EXPECT DOWN")
    grid_st = np.linspace(0.0, 6.0, 9)
    tde = [frac(m.do_driver(rec, stale_rel=v)['dist'], DECAY) for v in grid_st]
    tdt = [frac(m.do_driver(rec, stale_rel=v)['dist'], DEATH) for v in grid_st]
    print(f"    stale_rel grid (SYNTHETIC) {grid_st}")
    print(f"      P(DECAY)    : {np.array(tde)}  [{mono(tde)}]")
    print(f"      P(DEATH)    : {np.array(tdt)}  [{mono(tdt)}]  EXPECT UP")

    # ── 2. SIGN-CORRECTNESS ──
    print("\n  [2] SIGN-CORRECTNESS (Δgate per +0.5 driver step, recurring mean):")
    base = m.baseline(rec)
    d_sl = m.do_driver(rec, slope_rel=m.slope_rel[rec] + 0.5)
    d_st = m.do_driver(rec, stale_rel=m.stale_rel0[rec] + 0.5)
    print(f"    slope_rel↑ : Δp_rising={(d_sl['p_rising']-base['p_rising']).mean():+.4f} (EXPECT +)")
    print(f"    stale_rel↑ : Δp_alive ={(d_st['p_alive']-base['p_alive']).mean():+.4f} (EXPECT -)")

    # ── 3. TRAJECTORY COUNTERFACTUAL (real pairs) ──
    print("\n  [3] TRAJECTORY COUNTERFACTUAL (real recurring pairs):")
    dec = rec[b['dist'][rec].argmax(1) == DECAY]
    if len(dec):
        cf = m.do_driver(dec, rate_ratio=4.0, slope_rel=1.0)
        flip = (cf['dist'].argmax(1) == REINFORCE).mean()
        print(f"    DECAY→do(dense edits): n={len(dec)} flip→REINFORCE = {flip:.3f}")
    else:
        print(f"    DECAY→do(dense edits): n=0 baseline-DECAY pairs")
    rei = rec[b['dist'][rec].argmax(1) == REINFORCE]
    cf2 = m.do_driver(rei, stale_rel=6.0, slope_rel=-1.0, rate_ratio=-4.0)
    print(f"    REINFORCE→do(long silence): n={len(rei)} "
          f"→DECAY={(cf2['dist'].argmax(1)==DECAY).mean():.3f} "
          f"→DEATH={(cf2['dist'].argmax(1)==DEATH).mean():.3f}")

    # ── 4. EXISTENCE COUNTERFACTUAL ──
    print("\n  [4] EXISTENCE COUNTERFACTUAL (force state → REAL existence_decoder):")
    for st in (DEATH, REINFORCE, DECAY, BIRTH):
        r = m.do_state(rec, state=st)
        print(f"    do(state={STATE_NAMES[st]:9s}): P(edge) {r['p_edge_base'].mean():.3f}"
              f" → {r['p_edge_forced'].mean():.3f}  Δ={r['delta'].mean():+.3f}")

    # ── 5. REVERSIBILITY ──
    noop = m.do_driver()
    print(f"\n  [5] do(noop) max|Δ dist| = {np.abs(noop['dist']-b['dist']).max():.1e} "
          f" NaN={np.isnan(b['dist']).any() or np.isnan(b['p_edge']).any()}")

print("=" * 78)
print("BATTERY A/B DONE")
