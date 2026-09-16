"""
COUNTERFACTUAL battery — config-B (cbON) 3-seed, run INLINE in the GPU job.

Reads the three config-B faithfulness npz (causal_batch ON, design=correct_decoupled,
hier+decol_hier_v2, let0.5):
    results/faithfulness_coedit_v3_hier_hv2_let0.5_s{42,1,7}_cbON.npz
and runs the fsm_intervene HierV2SCM battery on each, then aggregates mean+/-std
across seeds for the headline counterfactual metrics. NO fabrication: every number
is computed from the dumped PRE-update drivers via the offline SCM.

Headline metrics aggregated 3-seed:
  - existence-CF do(DEATH): P(edge) drop % of pairs whose P(edge) falls, meanDelta,
    and the forced-state edge ladder DEATH<DECAY<BIRTH~REINFORCE (by construction).
    reversibility max|Delta| after restore (must be ~0).
  - dose-response: SIGN of rate_ratio-> P(REINFORCE) slope and -> P(DEATH) slope,
    slope_rel-> P(REINFORCE) sign, true_occ-> P(BIRTH) sign. Report sign stability.
  - trajectory-CF: DECAY pairs -> do(dense edits) flip->REINFORCE %.
"""
import numpy as np
from fsm_intervene import (HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH)
from project_paths import result_input

SEEDS = [42, 1, 7]
PATHS = {s: str(result_input(f"faithfulness_coedit_v3_hier_hv2_let0.5_s{s}_cbON.npz")) for s in SEEDS}
np.set_printoptions(precision=4, suppress=True)


def frac(d, s):
    return float((d.argmax(1) == s).mean())


def slope_sign(grid, ys):
    """least-squares slope sign over a monotone grid sweep."""
    g = np.asarray(grid, float); y = np.asarray(ys, float)
    a = np.polyfit(g, y, 1)[0]
    return a


per = {}   # per-seed metric dict
for s in SEEDS:
    print("=" * 78)
    print(f"SEED {s}  npz={PATHS[s]}")
    m = HierV2SCM(PATHS[s])
    b = m.baseline()
    bdist = np.bincount(b['dist'].argmax(1), minlength=5) / m.N
    print(f"  N={m.N}  baseline argmax[I,B,R,D,Dt]={np.round(bdist,4)}")
    rec = np.where(m.true_occ >= 2.0)[0]
    print(f"  recurring(true_occ>=2) n={len(rec)}  ({len(rec)/m.N:.3f})")

    md = {}

    # ---- EXISTENCE-CF: do(force state) -> REAL existence_decoder, recurring subset ----
    print("\n  [EXISTENCE-CF] force state -> P(edge) (recurring subset):")
    ladder = {}
    for st in (DEATH, DECAY, BIRTH, REINFORCE):
        r = m.do_state(rec, state=st)
        ladder[STATE_NAMES[st]] = float(r['p_edge_forced'].mean())
        print(f"    do({STATE_NAMES[st]:9s}): base {r['p_edge_base'].mean():.4f} "
              f"-> forced {r['p_edge_forced'].mean():.4f}  Delta={r['delta'].mean():+.4f}")
    # the headline existence-CF: do(DEATH)
    rD = m.do_state(rec, state=DEATH)
    drop_frac = float((rD['delta'] < -1e-9).mean())   # frac pairs P(edge) DROPS
    md['exCF_doDEATH_dropfrac'] = drop_frac
    md['exCF_doDEATH_meanDelta'] = float(rD['delta'].mean())
    md['exCF_doDEATH_pe_forced'] = float(rD['p_edge_forced'].mean())
    md['exCF_doDEATH_pe_base'] = float(rD['p_edge_base'].mean())
    md['ladder_DEATH'] = ladder['DEATH']
    md['ladder_DECAY'] = ladder['DECAY']
    md['ladder_BIRTH'] = ladder['BIRTH']
    md['ladder_REINFORCE'] = ladder['REINFORCE']
    # ladder ordering check DEATH < DECAY < BIRTH ~ REINFORCE
    md['ladder_ok'] = int(ladder['DEATH'] < ladder['DECAY'] < ladder['BIRTH']
                          and ladder['DEATH'] < ladder['REINFORCE'])
    print(f"    do(DEATH): frac-pairs-P(edge)-DROP={drop_frac:.4f}  meanDelta={rD['delta'].mean():+.4f}")
    print(f"    ladder OK (DEATH<DECAY<BIRTH&REINF)? {bool(md['ladder_ok'])}")

    # ---- REVERSIBILITY: intervene then restore at observed -> back to baseline ----
    base_rec = m.baseline(rec)
    _ = m.do_driver(rec, rate_ratio=4.0, slope_rel=1.0)   # perturb
    step2 = m.do_driver(rec, rate_ratio=m.rate_ratio[rec], slope_rel=m.slope_rel[rec])
    rev = float(np.abs(step2['dist'] - base_rec['dist']).max())
    md['reversibility_maxabsdelta'] = rev
    print(f"  [REVERSIBILITY] after restore max|Delta dist|={rev:.2e} (EXPECT ~0)")

    # ---- DOSE-RESPONSE signs (recurring subset) ----
    grid_rr = np.linspace(-4, 4, 9)
    pre = [frac(m.do_driver(rec, rate_ratio=v)['dist'], REINFORCE) for v in grid_rr]
    pdt = [frac(m.do_driver(rec, rate_ratio=v)['dist'], DEATH) for v in grid_rr]
    grid_sl = np.linspace(-1.0, 1.0, 9)
    sre = [frac(m.do_driver(rec, slope_rel=v)['dist'], REINFORCE) for v in grid_sl]
    grid_to = np.array([1, 2, 3, 5, 10, 20, 50], float)
    obi = [frac(m.do_driver(slice(None), true_occ=v)['dist'], BIRTH) for v in grid_to]
    md['dose_rate_REINF_slope'] = float(slope_sign(grid_rr, pre))   # EXPECT +
    md['dose_rate_DEATH_slope'] = float(slope_sign(grid_rr, pdt))   # EXPECT -
    md['dose_slope_REINF_slope'] = float(slope_sign(grid_sl, sre))  # EXPECT +
    md['dose_trueocc_BIRTH_slope'] = float(slope_sign(grid_to, obi))  # EXPECT -
    print("\n  [DOSE-RESPONSE] regression slopes (sign is the claim):")
    print(f"    rate_ratio  -> P(REINFORCE) slope={md['dose_rate_REINF_slope']:+.4f} (EXPECT +)")
    print(f"    rate_ratio  -> P(DEATH)     slope={md['dose_rate_DEATH_slope']:+.4f} (EXPECT -)")
    print(f"    slope_rel   -> P(REINFORCE) slope={md['dose_slope_REINF_slope']:+.4f} (EXPECT +)")
    print(f"    true_occ    -> P(BIRTH)     slope={md['dose_trueocc_BIRTH_slope']:+.4f} (EXPECT -)")

    # ---- TRAJECTORY-CF: DECAY pairs -> do(dense edits) flip -> REINFORCE % ----
    dec = rec[base_rec['dist'].argmax(1) == DECAY]
    if len(dec):
        cf = m.do_driver(dec, rate_ratio=4.0, slope_rel=1.0)
        flip = float((cf['dist'].argmax(1) == REINFORCE).mean())
    else:
        flip = float('nan')
    md['traj_DECAY_to_REINF_flip'] = flip
    md['n_DECAY_baseline'] = int(len(dec))
    print(f"\n  [TRAJECTORY-CF] DECAY pairs n={len(dec)} -> do(dense edits) "
          f"flip->REINFORCE = {flip:.4f}")

    # ---- SANITY ----
    md['NaN'] = bool(np.isnan(b['dist']).any() or np.isnan(b['p_edge']).any())
    noop = m.do_driver()
    md['noop_maxabsdelta'] = float(np.abs(noop['dist'] - b['dist']).max())
    per[s] = md

# ============================ 3-SEED AGGREGATION ============================
print("=" * 78)
print("3-SEED AGGREGATION (config-B cbON, coedit, seeds 42,1,7)")
print("=" * 78)


def agg(key, pct=False, signfmt=False):
    vals = np.array([per[s][key] for s in SEEDS], float)
    mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
    per_s = "  ".join(f"s{s}={per[s][key]:+.4f}" if signfmt else f"s{s}={per[s][key]:.4f}"
                      for s in SEEDS)
    scale = 100.0 if pct else 1.0
    print(f"  {key:30s}: {mu*scale:+.4f} +/- {sd*scale:.4f}   [{per_s}]")
    return mu, sd, vals


print("\n-- EXISTENCE-CF do(force DEATH) --")
agg('exCF_doDEATH_dropfrac', pct=False)
agg('exCF_doDEATH_meanDelta', signfmt=True)
agg('exCF_doDEATH_pe_base')
agg('exCF_doDEATH_pe_forced')
print("\n-- EXISTENCE ladder (forced-state P(edge)) --")
agg('ladder_DEATH'); agg('ladder_DECAY'); agg('ladder_BIRTH'); agg('ladder_REINFORCE')
print("  ladder_ok per seed:", {s: per[s]['ladder_ok'] for s in SEEDS})
print("\n-- REVERSIBILITY --")
agg('reversibility_maxabsdelta')
print("\n-- DOSE-RESPONSE slopes (sign-stability across seeds) --")
agg('dose_rate_REINF_slope', signfmt=True)
agg('dose_rate_DEATH_slope', signfmt=True)
agg('dose_slope_REINF_slope', signfmt=True)
agg('dose_trueocc_BIRTH_slope', signfmt=True)
print("\n-- TRAJECTORY-CF --")
agg('traj_DECAY_to_REINF_flip')
print("  n_DECAY_baseline per seed:", {s: per[s]['n_DECAY_baseline'] for s in SEEDS})
print("\n-- SANITY --")
print("  NaN per seed:", {s: per[s]['NaN'] for s in SEEDS})
agg('noop_maxabsdelta')

# verdict inputs
print("\n-- VERDICT INPUTS --")
dd = np.array([per[s]['exCF_doDEATH_dropfrac'] for s in SEEDS])
print(f"  existence-CF do(DEATH) drop-frac all>=0.99 (by construction)? {(dd>=0.99).all()}")
lad = all(per[s]['ladder_ok'] for s in SEEDS)
print(f"  existence ladder correct all seeds? {lad}")
rrs = np.sign([per[s]['dose_rate_REINF_slope'] for s in SEEDS])
rds = np.sign([per[s]['dose_rate_DEATH_slope'] for s in SEEDS])
sls = np.sign([per[s]['dose_slope_REINF_slope'] for s in SEEDS])
tos = np.sign([per[s]['dose_trueocc_BIRTH_slope'] for s in SEEDS])
print(f"  dose rate->REINF sign all + ? {(rrs>0).all()}  per-seed signs={rrs}")
print(f"  dose rate->DEATH sign all - ? {(rds<0).all()}  per-seed signs={rds}")
print(f"  dose slope->REINF sign all + ? {(sls>0).all()}  per-seed signs={sls}")
print(f"  dose trueocc->BIRTH sign all - ? {(tos<0).all()}  per-seed signs={tos}")
fl = np.array([per[s]['traj_DECAY_to_REINF_flip'] for s in SEEDS])
print(f"  trajectory DECAY->REINF flip mean={np.nanmean(fl):.3f} std={np.nanstd(fl):.3f} "
      f"(seed-variable? std>0.05: {np.nanstd(fl)>0.05})")
print("=" * 78)
print("BATTERY DONE")
