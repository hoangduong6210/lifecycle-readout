"""
ITEM-1 closure: REINFORCE -> do(long silence / rate=dead) -> DEATH, 3-SEED (offline).

Reproduces EXACTLY the seed-42 "kill a thriving pair" counterfactual
(_battery_tmp.py:93-97) on ALL THREE config-B npz (s42/s1/s7, cbON) and aggregates
mean+/-std. The intervention is the FULL dead-driver combo applied to pairs whose
baseline argmax is REINFORCE:
    do_driver(rei, stale_rel=6.0, slope_rel=-1.0, rate_ratio=-4.0)
i.e. force long silence (high stale) + falling slope + dead rate. We report the
fraction of those REINFORCE pairs that move to DECAY and to DEATH.

Honest caveat carried from fsm_intervene.py: stale_rel is partly absorbed into the
frozen r_obs at eval time, so the staleness arm is a LOWER BOUND on the true effect.
Every number computed from dumped PRE-update drivers; NO fabrication.
"""
import numpy as np
from fsm_intervene import HierV2SCM, REINFORCE, DECAY, DEATH
from project_paths import result_input

SEEDS = [42, 1, 7]
PATHS = {s: str(result_input(f"faithfulness_coedit_v3_hier_hv2_let0.5_s{s}_cbON.npz")) for s in SEEDS}
np.set_printoptions(precision=4, suppress=True)

per = {}
for s in SEEDS:
    m = HierV2SCM(PATHS[s])
    rec = np.where(m.true_occ >= 2.0)[0]
    b = m.baseline()
    rei = rec[b['dist'][rec].argmax(1) == REINFORCE]
    # ISOLATED interventions on REINFORCE-decoded pairs (matching paper §4)
    cf_rate  = m.do_driver(rei, rate_ratio=-4.0)              # do(rate=dead) ALONE
    cf_stale = m.do_driver(rei, stale_rel=6.0)                # do(staleness=high) ALONE
    def brk(cf):
        a = cf['dist'].argmax(1)
        return (float((a == DEATH).mean()), float((a == DECAY).mean()),
                float((a != REINFORCE).mean()))   # ->DEATH, ->DECAY, leaves REINFORCE
    rd, rde, rl = brk(cf_rate)
    sd, sde, sl = brk(cf_stale)
    per[s] = dict(n_rei=len(rei), rate_death=rd, rate_leaves=rl,
                  stale_death=sd, stale_leaves=sl)
    print(f"seed {s}: n_REINFORCE={len(rei)}")
    print(f"   do(rate=dead) : ->DEATH={rd:.4f} ->DECAY={rde:.4f} leaves_REINF={rl:.4f}")
    print(f"   do(stale=high): ->DEATH={sd:.4f} ->DECAY={sde:.4f} leaves_REINF={sl:.4f}")

print("=" * 72)
for k, lbl in (("rate_death", "do(rate=dead)->DEATH"),
               ("rate_leaves", "do(rate=dead) leaves"),
               ("stale_death", "do(stale=high)->DEATH"),
               ("stale_leaves", "do(stale=high) leaves")):
    v = np.array([per[s][k] for s in SEEDS])
    print(f"  {lbl:24s}: {v.mean():.4f} +/- {v.std(ddof=1):.4f}   per-seed={np.round(v,4)}")
