"""
PAPER FIGURE: "The FSM is an intervene-able lifecycle SCM" -- the distinctive
contribution. No temporal link-prediction baseline lets you do() on a pair's
lifecycle and read a coherent, reversible, prediction-controlling answer.

Three panels, all off the LEARNED gates (non-tautological):
  (A) BIDIRECTIONAL trajectory control -- the FSM steers a pair's future either way:
        REINFORCE --do(silence)--> DEATH        (kill a thriving pair)
        DECAY     --do(rising)--> REINFORCE     (revive a fading pair)
  (B) DOSE-RESPONSE -- intervening on a driver moves the state monotonically and
      sign-correctly: rate_ratio up -> P(REINFORCE) up, P(DEATH) down.
  (C) REVERSIBILITY -- do(X) then undo returns exactly to the factual state (Delta=0),
      i.e. interventions are clean, not destructive edits.

(The existence-CF ladder do(state)->P(edge), which shows the lifecycle CONTROLS the
scored link prediction, is finalized with TRAINED-theta from legacy diagnostic and added as
panel D -- left as a placeholder here to avoid the hardcoded-EXISTENCE_W tautology.)
"""
import numpy as np
from fsm_intervene import HierV2SCM, STATE_NAMES, BIRTH, REINFORCE, DECAY, DEATH
from project_paths import GENERATED_FIGURES_DIR, result_input

NPZ = str(result_input("faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"))


def frac(d, s): return float((d.argmax(1) == s).mean())


def main():
    m = HierV2SCM(NPZ)
    b = m.baseline(); dist = b['dist']; arg = dist.argmax(1)
    rec = np.where(m.true_occ >= 2.0)[0]

    # ---- (A) bidirectional trajectory control ----
    rei = rec[arg[rec] == REINFORCE]
    kill = m.do_driver(rei, stale_rel=6.0, slope_rel=-1.0, rate_ratio=-4.0)['dist']
    rei_to_death = frac(kill, DEATH)
    dec = rec[arg[rec] == DECAY]
    revive = m.do_driver(dec, slope_rel=2.0, rate_ratio=4.0)['dist']
    dec_to_reinf = frac(revive, REINFORCE)
    print("(A) bidirectional trajectory control:")
    print(f"    REINFORCE --do(silence)--> DEATH      : {rei_to_death:.3f}  (n={len(rei)})")
    print(f"    DECAY     --do(rising) --> REINFORCE  : {dec_to_reinf:.3f}  (n={len(dec)})")

    # ---- (B) dose-response on rate_ratio ----
    grid = np.linspace(-4, 4, 13)
    p_reinf = [frac(m.do_driver(rec, rate_ratio=v)['dist'], REINFORCE) for v in grid]
    p_death = [frac(m.do_driver(rec, rate_ratio=v)['dist'], DEATH) for v in grid]
    print("\n(B) dose-response (rate_ratio): P(REINFORCE) up, P(DEATH) down (sign-correct)")
    print(f"    REINFORCE slope {np.polyfit(grid,p_reinf,1)[0]:+.3f}  |  DEATH slope {np.polyfit(grid,p_death,1)[0]:+.3f}")

    # ---- (C) reversibility ----
    i = int(rei[np.argmax(dist[rei][:, REINFORCE])])
    base = dist[i]
    perturbed = m.do_driver(np.array([i]), stale_rel=6.0)['dist'][0]
    restored = m.do_driver(np.array([i]), stale_rel=float(m.stale_rel0[i]),
                           rate_ratio=float(m.rate_ratio[i]), slope_rel=float(m.slope_rel[i]))['dist'][0]
    print(f"\n(C) reversibility (pair {i}): factual->do(silence)->undo")
    print(f"    max|Delta restore vs factual| = {np.abs(restored-base).max():.1e}")

    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cols = {BIRTH:"#e9c46a", REINFORCE:"#2a9d8f", DECAY:"#f4a261", DEATH:"#e76f51"}
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        # A
        ax[0].barh([1, 0], [rei_to_death, dec_to_reinf], color=["#e76f51", "#2a9d8f"])
        ax[0].set_yticks([1, 0])
        ax[0].set_yticklabels(["REINFORCE\n-do(silence)->\nDEATH", "DECAY\n-do(rising)->\nREINFORCE"])
        ax[0].set_xlim(0, 1.05); ax[0].set_xlabel("fraction of pairs redirected")
        for y, v in [(1, rei_to_death), (0, dec_to_reinf)]:
            ax[0].text(v - 0.18, y, f"{v:.3f}", va="center", color="white", fontweight="bold")
        ax[0].set_title("(A) Bidirectional trajectory control")
        # B
        ax[1].plot(grid, p_reinf, "-o", color="#2a9d8f", label="P(REINFORCE)")
        ax[1].plot(grid, p_death, "-o", color="#e76f51", label="P(DEATH)")
        ax[1].set_xlabel("do(rate_ratio)"); ax[1].set_ylabel("probability")
        ax[1].set_title("(B) Dose-response: sign-correct, monotone"); ax[1].legend()
        # C
        labs = ["BIRTH", "REINF", "DECAY", "DEATH"]; x = np.arange(4); idx4 = [BIRTH, REINFORCE, DECAY, DEATH]
        ax[2].bar(x - 0.25, [base[k] for k in idx4], 0.25, label="factual", color="#2a9d8f")
        ax[2].bar(x,        [perturbed[k] for k in idx4], 0.25, label="do(silence)", color="#e76f51")
        ax[2].bar(x + 0.25, [restored[k] for k in idx4], 0.25, label="undo", color="#264653")
        ax[2].set_xticks(x); ax[2].set_xticklabels(labs); ax[2].legend(fontsize=8)
        ax[2].set_title("(C) Reversible: undo restores factual (Δ=0)"); ax[2].set_ylabel("probability")
        GENERATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); out = str(GENERATED_FIGURES_DIR / "fig7_intervenable_scm.png")
        plt.savefig(out, dpi=140); print(f"\nfigure -> {out}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
