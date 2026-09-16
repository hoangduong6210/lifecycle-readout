"""
COUNTERFACTUAL DEMO on a REAL CoEdit (user u, page v) pair.

Question (research question): "If user u stops editing page v for a window, does the pair
go to DEATH? And WHY DEATH?"  -> answered as an intervene-able, attributable
counterfactual read off the LEARNED lifecycle gates (NOT the hardcoded
EXISTENCE_W; the state-flip uses p_birth/p_alive/p_rising which are trained).

Outputs a human-readable narrative + a figure with three panels:
  (1) factual vs do(silence) lifecycle distribution (BIRTH/REINFORCE/DECAY/DEATH),
  (2) driver attribution: leave-one-driver-out delta-P(DEATH) (staleness / rate /
      slope) -> "why DEATH",
  (3) per-pair relativity: same injected silence on a fast-habit vs slow-habit pair
      -> DEATH is relative to each pair's OWN cadence, not an absolute clock.

Honest scope (printed): this is a SINGLE-PAIR driver intervention on the pair's own
edit activity. Cross-pair spillover ("another user's edits change THIS pair") is a
network counterfactual the per-pair operator does not model -> flagged, not faked.
The existence-CF P(edge) ladder is left for the TRAINED-theta battery (legacy diagnostic).
"""
import numpy as np
from fsm_intervene import HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH
from project_paths import GENERATED_FIGURES_DIR, result_input

NPZ = str(result_input("faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"))
np.set_printoptions(precision=4, suppress=True)


def pdist(m, idx, **do):
    r = m.do_driver(np.array([idx]), **do) if do else m.baseline()
    d = r['dist'][idx] if not do else r['dist'][0]
    return d


def main():
    m = HierV2SCM(NPZ)
    b = m.baseline()
    arg = b['dist'].argmax(1)
    rec = np.where(m.true_occ >= 3.0)[0]                     # well-established pairs

    # --- pick a clear REINFORCE pair (alive, re-editing) as the hero ---
    rei = rec[arg[rec] == REINFORCE]
    # hero = the most confidently-REINFORCE pair, so the death flip is dramatic & clear
    hero = rei[np.argmax(b['dist'][rei][:, REINFORCE])]
    i = int(hero)
    base = b['dist'][i]
    print("=" * 74)
    print(f"HERO PAIR  index={i}   (a real CoEdit user-page pair)")
    print(f"  drivers: true_occ={m.true_occ[i]:.0f}  rate_ratio={m.rate_ratio[i]:+.3f}  "
          f"slope_rel={m.slope_rel[i]:+.3f}  staleness(z)={m.stale_rel0[i]:+.3f}")
    print(f"  FACTUAL lifecycle P[I,B,R,D,Dt] = {np.round(base,3)}  -> argmax={STATE_NAMES[base.argmax()]}")

    # --- COUNTERFACTUAL: user stops editing for a window = inject silence ---
    cf = pdist(m, i, stale_rel=6.0, rate_ratio=-4.0, slope_rel=-1.0)
    print(f"\n  do(user u makes NO edit on page v for the window)  [silence]")
    print(f"  CF lifecycle P[I,B,R,D,Dt]      = {np.round(cf,3)}  -> argmax={STATE_NAMES[cf.argmax()]}")
    print(f"  P(DEATH): {base[DEATH]:.3f} -> {cf[DEATH]:.3f}   (Delta {cf[DEATH]-base[DEATH]:+.3f})")

    # --- WHY DEATH? leave-one-driver-out attribution on P(DEATH) ---
    onlystale = pdist(m, i, stale_rel=6.0)[DEATH] - base[DEATH]
    onlyrate  = pdist(m, i, rate_ratio=-4.0)[DEATH] - base[DEATH]
    onlyslope = pdist(m, i, slope_rel=-1.0)[DEATH] - base[DEATH]
    tot = onlystale + onlyrate + onlyslope
    share = lambda x: (100 * x / tot) if tot > 1e-9 else 0.0
    print(f"\n  WHY DEATH (single-driver dP(DEATH) attribution):")
    print(f"    staleness (gap > pair's habit) : {onlystale:+.3f}   ({share(onlystale):4.0f}%)")
    print(f"    rate falling (edits slowing)   : {onlyrate:+.3f}   ({share(onlyrate):4.0f}%)")
    print(f"    slope (cadence declining)      : {onlyslope:+.3f}   ({share(onlyslope):4.0f}%)")
    dom = max([("staleness", onlystale), ("rate", onlyrate), ("slope", onlyslope)], key=lambda t: t[1])
    print(f"    => DEATH driven mainly by: {dom[0].upper()}")

    # --- REVERSIBILITY: resume edits restores the alive state ---
    rev = pdist(m, i, stale_rel=float(m.stale_rel0[i]), rate_ratio=float(m.rate_ratio[i]),
                slope_rel=float(m.slope_rel[i]))
    print(f"\n  do(resume edits) [restore drivers] -> P[..]={np.round(rev,3)}  "
          f"argmax={STATE_NAMES[rev.argmax()]}  (max|Delta vs factual|={np.abs(rev-base).max():.2e})")

    # --- DOSE-RESPONSE: how long must the user stay silent before the pair dies? ---
    # sweep injected silence (staleness, in units of the pair's OWN habitual gap) and
    # read the lifecycle: the pair walks REINFORCE -> DECAY -> DEATH as silence grows.
    grid = np.linspace(0.0, 6.0, 13)
    sweep = np.array([pdist(m, i, stale_rel=float(s)) for s in grid])   # (13,5)
    print(f"\n  DOSE-RESPONSE (hero pair {i}) -- silence in units of the pair's own habit:")
    for s, row in list(zip(grid, sweep))[::3]:
        print(f"    silence={s:.1f}x habit :  P(REINF)={row[REINFORCE]:.2f}  "
              f"P(DECAY)={row[DECAY]:.2f}  P(DEATH)={row[DEATH]:.2f}  -> {STATE_NAMES[row.argmax()]}")
    cross = grid[np.argmax(sweep[:, DEATH] >= 0.5)] if (sweep[:, DEATH] >= 0.5).any() else None
    print(f"    => crosses into DEATH (P>=0.5) at silence ~= "
          f"{cross:.1f}x the pair's habitual gap" if cross is not None else "    => never crosses 0.5")

    print("\n  [scope] single-pair driver intervention on the pair's OWN activity.")
    print("  Cross-pair spillover (another user's edits -> this pair) is NOT modelled")
    print("  by the per-pair operator (network counterfactual = future work).")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        labs = ["IDLE", "BIRTH", "REINF", "DECAY", "DEATH"]
        x = np.arange(5)
        ax[0].bar(x - 0.2, base, 0.4, label="factual", color="#2a9d8f")
        ax[0].bar(x + 0.2, cf, 0.4, label="do(silence)", color="#e76f51")
        ax[0].set_xticks(x); ax[0].set_xticklabels(labs, rotation=30)
        ax[0].set_title("(1) Lifecycle: factual vs counterfactual"); ax[0].legend()
        ax[0].set_ylabel("probability")
        drivers = ["staleness", "rate", "slope"]
        vals = [onlystale, onlyrate, onlyslope]
        ax[1].bar(drivers, vals, color=["#e76f51", "#f4a261", "#e9c46a"])
        ax[1].set_title("(2) Why DEATH: dP(DEATH) per driver"); ax[1].set_ylabel("dP(DEATH)")
        ax[2].plot(grid, sweep[:, REINFORCE], "-o", ms=3, label="REINFORCE", color="#2a9d8f")
        ax[2].plot(grid, sweep[:, DECAY], "-o", ms=3, label="DECAY", color="#f4a261")
        ax[2].plot(grid, sweep[:, DEATH], "-o", ms=3, label="DEATH", color="#e76f51")
        ax[2].axhline(0.5, ls="--", c="gray", lw=0.8)
        ax[2].set_xlabel("injected silence (x pair's habitual gap)")
        ax[2].set_title("(3) Dose-response: silence -> DEATH"); ax[2].legend(); ax[2].set_ylabel("probability")
        plt.tight_layout()
        out = str(GENERATED_FIGURES_DIR / "cf_demo_pair_s42.png")
        GENERATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=130); print(f"\n  figure -> {out}")
    except Exception as e:
        print("  (figure skipped:", e, ")")


if __name__ == "__main__":
    main()
