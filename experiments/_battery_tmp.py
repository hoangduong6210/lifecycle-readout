import numpy as np
from fsm_intervene import (HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH)
from project_paths import result_input

SEEDS = [42, 1, 7]
PATHS = {s: str(result_input(f"faithfulness_coedit_v3_hier_hv2_let0.5_s{s}.npz")) for s in SEEDS}
np.set_printoptions(precision=4, suppress=True)


def mono(x):
    dx = np.diff(x)
    up = (dx >= -1e-9).all(); dn = (dx <= 1e-9).all()
    return "UP-mono" if up and not dn else "DN-mono" if dn and not up else \
           "FLAT" if np.abs(dx).max() < 1e-4 else "non-mono"


def frac(d, s):
    return (d.argmax(1) == s).mean()


for s in SEEDS:
    print("=" * 78)
    print(f"SEED {s}  npz={PATHS[s]}")
    m = HierV2SCM(PATHS[s])
    b = m.baseline()
    print(f"  N={m.N}  baseline argmax: " +
          " ".join(f"{STATE_NAMES[i]}={np.bincount(b['dist'].argmax(1),minlength=5)[i]/m.N:.3f}"
                   for i in range(1, 5)))

    # recurring active subset (the genuine lifecycle pairs)
    rec = np.where(m.true_occ >= 2.0)[0]
    print(f"  recurring(true_occ>=2) n={len(rec)}")

    # ── 1. DOSE-RESPONSE / MONOTONICITY (sweep each clean driver, recurring set) ──
    print("\n  [1] DOSE-RESPONSE (recurring subset, sweep driver → P(target state)):")
    # rate↑ → P(REINFORCE)↑  (intervene via rate_ratio directly: clean, exact)
    grid_rr = np.linspace(-4, 4, 9)
    pre = [frac(m.do_driver(rec, rate_ratio=v)['dist'], REINFORCE) for v in grid_rr]
    pde = [frac(m.do_driver(rec, rate_ratio=v)['dist'], DECAY) for v in grid_rr]
    pdt = [frac(m.do_driver(rec, rate_ratio=v)['dist'], DEATH) for v in grid_rr]
    print(f"    rate_ratio grid {grid_rr}")
    print(f"      P(REINFORCE): {np.array(pre)}  [{mono(pre)}]  EXPECT UP")
    print(f"      P(DEATH)    : {np.array(pdt)}  [{mono(pdt)}]  EXPECT DOWN")

    # slope_rel↑ → P(REINFORCE)↑ / P(DECAY)↓
    grid_sl = np.linspace(-1.0, 1.0, 9)
    sre = [frac(m.do_driver(rec, slope_rel=v)['dist'], REINFORCE) for v in grid_sl]
    sde = [frac(m.do_driver(rec, slope_rel=v)['dist'], DECAY) for v in grid_sl]
    print(f"    slope_rel grid {grid_sl}")
    print(f"      P(REINFORCE): {np.array(sre)}  [{mono(sre)}]  EXPECT UP")
    print(f"      P(DECAY)    : {np.array(sde)}  [{mono(sde)}]  EXPECT DOWN")

    # staleness↑ (SYNTHETIC) → P(DECAY) then P(DEATH)↑
    grid_st = np.linspace(0.0, 6.0, 9)
    tde = [frac(m.do_driver(rec, stale_rel=v)['dist'], DECAY) for v in grid_st]
    tdt = [frac(m.do_driver(rec, stale_rel=v)['dist'], DEATH) for v in grid_st]
    tre = [frac(m.do_driver(rec, stale_rel=v)['dist'], REINFORCE) for v in grid_st]
    print(f"    stale_rel grid (SYNTHETIC) {grid_st}")
    print(f"      P(REINFORCE): {np.array(tre)}  [{mono(tre)}]  EXPECT DOWN")
    print(f"      P(DECAY)    : {np.array(tde)}  [{mono(tde)}]")
    print(f"      P(DEATH)    : {np.array(tdt)}  [{mono(tdt)}]  EXPECT UP")

    # true_occ (recurrence) 1→50
    grid_to = np.array([1, 2, 3, 5, 10, 20, 50], float)
    ore = [frac(m.do_driver(slice(None), true_occ=v)['dist'], REINFORCE) for v in grid_to]
    obi = [frac(m.do_driver(slice(None), true_occ=v)['dist'], BIRTH) for v in grid_to]
    print(f"    true_occ grid (ALL) {grid_to}")
    print(f"      P(BIRTH)    : {np.array(obi)}  [{mono(obi)}]  EXPECT DOWN")
    print(f"      P(REINFORCE): {np.array(ore)}  [{mono(ore)}]  EXPECT UP")

    # ── 2. SIGN-CORRECTNESS (mean gate move per unit driver at observed point) ──
    print("\n  [2] SIGN-CORRECTNESS (Δgate per +driver step, recurring mean):")
    eps = 0.5
    base = m.baseline(rec)
    d_rr = m.do_driver(rec, rate_ratio=m.rate_ratio[rec] + eps)
    d_sl = m.do_driver(rec, slope_rel=m.slope_rel[rec] + eps)
    d_st = m.do_driver(rec, stale_rel=m.stale_rel0[rec] + eps)
    print(f"    rate_ratio↑ : Δp_alive ={ (d_rr['p_alive']-base['p_alive']).mean():+.4f} (EXPECT +)")
    print(f"    slope_rel↑  : Δp_rising={ (d_sl['p_rising']-base['p_rising']).mean():+.4f} (EXPECT +)")
    print(f"    stale_rel↑  : Δp_alive ={ (d_st['p_alive']-base['p_alive']).mean():+.4f} (EXPECT -),"
          f" Δp_rising={ (d_st['p_rising']-base['p_rising']).mean():+.4f} (EXPECT -)")

    # ── 3. TRAJECTORY COUNTERFACTUAL on REAL pairs ──
    print("\n  [3] TRAJECTORY COUNTERFACTUAL (real recurring pairs):")
    # pairs currently DECAY → do(dense edits = high rate + rising slope) → flip to REINFORCE?
    dec = rec[b['dist'][rec].argmax(1) == DECAY]
    if len(dec):
        cf = m.do_driver(dec, rate_ratio=4.0, slope_rel=1.0)
        flip = (cf['dist'].argmax(1) == REINFORCE).mean()
        print(f"    DECAY→do(dense edits): n={len(dec)} flip→REINFORCE = {flip:.3f}")
    else:
        print(f"    DECAY→do(dense edits): n=0 baseline-DECAY pairs (see caveat)")
    # pairs currently REINFORCE → do(long silence = high stale) → go to DECAY/DEATH?
    rei = rec[b['dist'][rec].argmax(1) == REINFORCE]
    cf2 = m.do_driver(rei, stale_rel=6.0, slope_rel=-1.0, rate_ratio=-4.0)
    to_decay = (cf2['dist'].argmax(1) == DECAY).mean()
    to_death = (cf2['dist'].argmax(1) == DEATH).mean()
    print(f"    REINFORCE→do(long silence): n={len(rei)} →DECAY={to_decay:.3f} →DEATH={to_death:.3f}")

    # ── 4. EXISTENCE COUNTERFACTUAL (force state → real existence readout) ──
    print("\n  [4] EXISTENCE COUNTERFACTUAL (force state → REAL existence_decoder):")
    for st in (DEATH, REINFORCE, DECAY, BIRTH):
        r = m.do_state(rec, state=st)
        print(f"    do(state={STATE_NAMES[st]:9s}): P(edge) {r['p_edge_base'].mean():.3f}"
              f" → {r['p_edge_forced'].mean():.3f}  Δ={r['delta'].mean():+.3f}")

    # ── 5. REVERSIBILITY / CONSISTENCY ──
    print("\n  [5] REVERSIBILITY (intervene then restore → back to baseline?):")
    step1 = m.do_driver(rec, rate_ratio=4.0, slope_rel=1.0)
    # restore by re-decoding at OBSERVED values (engine is stateless ⇒ exact)
    step2 = m.do_driver(rec, rate_ratio=m.rate_ratio[rec], slope_rel=m.slope_rel[rec])
    print(f"    after restore max|Δ dist| vs baseline = {np.abs(step2['dist']-base['dist']).max():.2e}"
          f"  (EXPECT 0 → stateless, no path-dependence)")

print("=" * 78)
print("NaN/AP-INVARIANCE SUMMARY (all seeds):")
for s in SEEDS:
    m = HierV2SCM(PATHS[s])
    b = m.baseline(); noop = m.do_driver()
    nan = np.isnan(b['dist']).any() or np.isnan(b['p_edge']).any()
    inv = np.abs(noop['dist'] - b['dist']).max()
    print(f"  s{s}: NaN={nan}  do(noop) max|Δ|={inv:.1e}")
