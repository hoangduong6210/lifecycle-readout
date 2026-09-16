"""
LIFECYCLE-SANITY analyzer for the config-B faithfulness npz on a given dataset.

Usage: python _lifecycle_analyze.py <npz_path> <dataset_label>

Reports (all from the dumped PRE-update probe — no re-run, no fabrication):
  1. 5-state lifecycle distribution (BIRTH/REINFORCE/DECAY/DEATH; IDLE) on
     RAW argmax_s_t1_pos and CALIBRATED argmax_s_t1_cal, full set + recurring
     (true_occ>=2) subset + n_prior>=2 subset. Collapse check: is mass spread
     across REINFORCE/DECAY/DEATH or pinned to one state?
  2. Per-pair COHERENCE via the offline SCM: re-decode hier-v2 tree at observed
     drivers, report argmax agreement vs the published cal state, and how often a
     never-recurring pair (true_occ<2) lands on a "recurring" state (REINFORCE/DECAY)
     = structural violation proxy.
  3. FAITHFULNESS rho(state, pair-history): correlation of the lifecycle 'alive-ness'
     score (P(REINFORCE)+P(DECAY) - P(DEATH)) against recurrence drivers true_occ,
     n_prior, and rate_ratio. Meaningful lifecycle => positive rho with recurrence.
"""
import sys
import numpy as np
from fsm_intervene import HierV2SCM, STATE_NAMES, IDLE, BIRTH, REINFORCE, DECAY, DEATH

np.set_printoptions(precision=4, suppress=True)
NPZ = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "?"

d = np.load(NPZ)
N = len(d["true_occ"])
true_occ = d["true_occ"].astype(np.float64)
n_prior = d["n_prior"].astype(np.float64)
am_pos = d["argmax_s_t1_pos"].astype(np.int64)
am_cal = d["argmax_s_t1_cal"].astype(np.int64) if "argmax_s_t1_cal" in d.files else am_pos


def dist(am, idx=None):
    a = am if idx is None else am[idx]
    n = max(len(a), 1)
    return np.bincount(a, minlength=5) / n


print("=" * 78)
print(f"LIFECYCLE SANITY — dataset={LABEL}  npz={NPZ}")
print(f"  N_events={N}  recurring(true_occ>=2)={int((true_occ>=2).sum())} "
      f"({(true_occ>=2).mean():.3f})  n_prior>=2={int((n_prior>=2).sum())} "
      f"({(n_prior>=2).mean():.3f})")
print("=" * 78)

# ---- 1. 5-state distributions ----
rec = np.where(true_occ >= 2.0)[0]
np2 = np.where(n_prior >= 2.0)[0]
print("\n[1] 5-STATE LIFECYCLE DIST [IDLE,BIRTH,REINFORCE,DECAY,DEATH]:")
print(f"  RAW  full      : {dist(am_pos)}")
print(f"  CAL  full      : {dist(am_cal)}")
print(f"  CAL  recurring : {dist(am_cal, rec)}  (n={len(rec)})")
print(f"  CAL  n_prior>=2: {dist(am_cal, np2)}  (n={len(np2)})")

# collapse metric: how concentrated is the CAL dist on its top state (full + recurring)
cf = dist(am_cal); cr = dist(am_cal, rec)
print(f"\n  COLLAPSE CHECK (max single-state share; 1.0=fully collapsed):")
print(f"    full  top={STATE_NAMES[cf.argmax()]} share={cf.max():.3f} | "
      f"alive-states(R+D+Dt) mass={cf[REINFORCE]+cf[DECAY]+cf[DEATH]:.3f}")
print(f"    recur top={STATE_NAMES[cr.argmax()]} share={cr.max():.3f} | "
      f"REINFORCE={cr[REINFORCE]:.3f} DECAY={cr[DECAY]:.3f} DEATH={cr[DEATH]:.3f}")
# entropy of the recurring-subset dist (nats) — degenerate ~0
pr = cr[cr > 0]
ent = float(-(pr * np.log(pr)).sum())
print(f"    recurring-subset entropy={ent:.4f} nats (0=degenerate, ln5={np.log(5):.3f}=uniform)")

# ---- 2. per-pair coherence via SCM ----
print("\n[2] PER-PAIR COHERENCE (offline SCM re-decode of hier-v2 tree):")
try:
    m = HierV2SCM(NPZ)
    b = m.baseline()
    scm_am = b['dist'].argmax(1)
    agree = float((scm_am == am_cal).mean())
    print(f"  SCM-argmax vs published CAL-argmax agreement = {agree:.4f}")
    # structural violation proxy: never-recurring pair (true_occ<2) decoded as a
    # RECURRING state (REINFORCE or DECAY) — should be rare if lifecycle coherent.
    nonrec = np.where(true_occ < 2.0)[0]
    if len(nonrec):
        viol = float(np.isin(scm_am[nonrec], [REINFORCE, DECAY]).mean())
        print(f"  non-recurring(true_occ<2) n={len(nonrec)} decoded as REINFORCE/DECAY "
              f"(violation proxy) = {viol:.4f} (EXPECT low)")
    # coherent: recurring pair decoded alive (not DEATH/IDLE)
    if len(rec):
        alive = float(np.isin(scm_am[rec], [BIRTH, REINFORCE, DECAY]).mean())
        print(f"  recurring(true_occ>=2) decoded ALIVE (BIRTH/REINFORCE/DECAY) = "
              f"{alive:.4f} (EXPECT high)")
    scm_ok = True
except Exception as e:
    print(f"  SCM coherence SKIPPED (npz missing gate fields?): {e}")
    scm_ok = False

# ---- 3. faithfulness correlation state vs pair-history ----
print("\n[3] FAITHFULNESS rho(lifecycle-aliveness, pair-history drivers):")
# aliveness score from the dumped class probs
alive_score = (d["p_reinforce"].astype(np.float64) + d["p_decay"].astype(np.float64)
               - d["p_death"].astype(np.float64))


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


print(f"  rho(aliveness, true_occ) = {corr(alive_score, true_occ):+.4f} (EXPECT + : more recurrence -> more alive)")
print(f"  rho(aliveness, n_prior)  = {corr(alive_score, n_prior):+.4f} (EXPECT +)")
if scm_ok:
    print(f"  rho(aliveness, rate_ratio)= {corr(alive_score, m.rate_ratio):+.4f} (EXPECT +)")
# death score vs staleness-ish (low rate)
death_score = d["p_death"].astype(np.float64)
print(f"  rho(P(DEATH), true_occ)  = {corr(death_score, true_occ):+.4f} (EXPECT - : recurrence -> less death)")

print("\n[VERDICT INPUTS]")
print(f"  alive-state mass (R+D+Dt) full = {cf[REINFORCE]+cf[DECAY]+cf[DEATH]:.3f}")
print(f"  recurring-subset entropy = {ent:.4f} (degenerate if <~0.3)")
print(f"  4 lifecycle states all have >=1% mass (CAL full)? "
      f"{bool((cf[1:] >= 0.01).all())}  per-state={np.round(cf,4)}")
print("=" * 78)
print("LIFECYCLE ANALYZE DONE")
