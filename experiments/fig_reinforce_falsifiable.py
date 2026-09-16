"""
PAPER FIGURE: "Faithful & falsifiable lifecycle readout."

Rebuts the reviewer critique that the intervene-able readout is near-tautological
(entailed by a hardcoded EXISTENCE_W). It shows the readout encodes a SPECIFIC,
FALSIFIABLE rule -- "REINFORCE iff the edit-rate is RISING" -- and that the rule is
confirmed from THREE independent angles, all off the LEARNED gates (no hardcoded
ordering):

  (A) population: slope_rel distribution per argmax state -> REINFORCE sits at higher
      momentum than DECAY (the rule holds in aggregate, falsifiable & could have failed).
  (B) counterfactual: do(slope sweep) on a fixed cohort -> P(REINFORCE) rises
      monotonically with injected momentum (the transition is causally wired).
  (C) single-pair counterfactual: a real REINFORCE pair, do(silence) -> DEATH, with a
      driver attribution (why DEATH) and exact reversibility (intervene-able, reversible).

All numbers from the trained model via the offline SCM on the config-B faithfulness npz.
"""
import numpy as np
from fsm_intervene import HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH
from project_paths import GENERATED_FIGURES_DIR, result_input

NPZ = str(result_input("faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"))


def main():
    m = HierV2SCM(NPZ)
    b = m.baseline(); dist = b['dist']; arg = dist.argmax(1)
    rec = np.where(m.true_occ >= 1.0)[0]

    # ---- (A) population: slope per state ----
    by = {s: m.slope_rel[rec[arg[rec] == s]] for s in (BIRTH, REINFORCE, DECAY, DEATH)}
    print("(A) slope_rel by argmax state (median):")
    for s in (BIRTH, REINFORCE, DECAY, DEATH):
        v = by[s]
        print(f"    {STATE_NAMES[s]:9s} n={len(v):5d}  median={np.median(v):+.3f}  mean={v.mean():+.3f}")
    print(f"    => REINFORCE momentum > DECAY momentum : "
          f"{np.median(by[REINFORCE]):+.3f} > {np.median(by[DECAY]):+.3f}  "
          f"({'HOLDS' if np.median(by[REINFORCE]) > np.median(by[DECAY]) else 'FAILS'})")

    # ---- (B) counterfactual: do(slope sweep) on a fixed cohort -> P(REINFORCE) ----
    born = rec[arg[rec] == BIRTH]
    cohort = born                                   # newborns: clean substrate
    grid = np.linspace(-1.0, 2.0, 13)
    preinf = []
    for s in grid:
        d = m.do_driver(cohort, true_occ=m.true_occ[cohort] + 5, n_prior=m.n_prior[cohort] + 5,
                        slope_rel=np.full(len(cohort), s))['dist']
        preinf.append(float((d.argmax(1) == REINFORCE).mean()))
    preinf = np.array(preinf)
    print("\n(B) do(slope sweep) on newborn cohort -> fraction predicted REINFORCE:")
    for s, p in list(zip(grid, preinf))[::3]:
        print(f"    injected slope={s:+.2f} -> P(REINFORCE)={p:.2f}")
    print(f"    => monotone-rising? {bool(np.all(np.diff(preinf) >= -1e-6))}  "
          f"(span {preinf.min():.2f} -> {preinf.max():.2f})")

    # ---- (C) single-pair CF: real REINFORCE pair -> do(silence) -> DEATH + attribution ----
    rei = rec[arg[rec] == REINFORCE]
    i = int(rei[np.argmax(dist[rei][:, REINFORCE])])
    base = dist[i]
    cf = m.do_driver(np.array([i]), stale_rel=6.0, rate_ratio=-4.0, slope_rel=-1.0)['dist'][0]
    onlystale = m.do_driver(np.array([i]), stale_rel=6.0)['dist'][0][DEATH] - base[DEATH]
    onlyrate = m.do_driver(np.array([i]), rate_ratio=-4.0)['dist'][0][DEATH] - base[DEATH]
    onlyslope = m.do_driver(np.array([i]), slope_rel=-1.0)['dist'][0][DEATH] - base[DEATH]
    rev = m.do_driver(np.array([i]), stale_rel=float(m.stale_rel0[i]),
                      rate_ratio=float(m.rate_ratio[i]), slope_rel=float(m.slope_rel[i]))['dist'][0]
    print(f"\n(C) pair {i}: REINFORCE {base[REINFORCE]:.2f} -do(silence)-> DEATH {cf[DEATH]:.2f}")
    print(f"    attribution dP(DEATH): stale {onlystale:+.2f} rate {onlyrate:+.2f} slope {onlyslope:+.2f}")
    print(f"    reversibility max|Delta|={np.abs(rev-base).max():.1e}")

    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cols = {BIRTH: "#e9c46a", REINFORCE: "#2a9d8f", DECAY: "#f4a261", DEATH: "#e76f51"}
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        # A: violin/box of slope by state
        order = [BIRTH, REINFORCE, DECAY, DEATH]
        data = [by[s] for s in order]
        bp = ax[0].boxplot(data, vert=True, patch_artist=True, showfliers=False,
                           labels=[STATE_NAMES[s] for s in order])
        for patch, s in zip(bp['boxes'], order):
            patch.set_facecolor(cols[s])
        ax[0].axhline(0, ls="--", c="gray", lw=0.8)
        ax[0].set_ylabel("slope_rel (edit-rate momentum)")
        ax[0].set_title("(A) Population: REINFORCE sits above DECAY in momentum")
        # B: do(slope) -> P(REINFORCE)
        ax[1].plot(grid, preinf, "-o", color="#2a9d8f")
        ax[1].axvline(0, ls="--", c="gray", lw=0.8)
        ax[1].set_xlabel("injected slope (do)"); ax[1].set_ylabel("P(predicted REINFORCE)")
        ax[1].set_title("(B) Counterfactual: momentum -> REINFORCE (wired)")
        # C: single-pair CF bars
        labs = ["BIRTH", "REINF", "DECAY", "DEATH"]
        x = np.arange(4); idx4 = [BIRTH, REINFORCE, DECAY, DEATH]
        ax[2].bar(x - 0.2, [base[k] for k in idx4], 0.4, label="factual", color="#2a9d8f")
        ax[2].bar(x + 0.2, [cf[k] for k in idx4], 0.4, label="do(silence)", color="#e76f51")
        ax[2].set_xticks(x); ax[2].set_xticklabels(labs)
        ax[2].set_title(f"(C) Pair {i}: do(silence) -> DEATH, reversible")
        ax[2].legend(); ax[2].set_ylabel("probability")
        GENERATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); out = str(GENERATED_FIGURES_DIR / "fig6_faithful_falsifiable.png")
        plt.savefig(out, dpi=140); print(f"\nfigure -> {out}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
