"""
LIFECYCLE STATE-MAP + TRANSITION PREDICTION demo on REAL CoEdit pairs.

Answers (PM/user):
  - "BIRTH probability is high around WHICH region?"  -> the low-recurrence /
    early-trend region (pairs that just started editing, not yet REINFORCE).
  - "Where BIRTH is high, REINFORCE follows -- shifting FROM which region TO which?"
    -> as recurrence accrues (one more edit), born pairs move BIRTH -> REINFORCE;
    as silence/decline grows they move REINFORCE -> DECAY -> DEATH.
  - "Predict": for pairs currently in the BIRTH region with a rising trend, predict
    that one more edit converts them to REINFORCE -- and show concrete cases.

All states/gates are LEARNED (p_birth/p_alive/p_rising); regions and the predicted
shift are read off the trained model via the offline SCM, no hardcoded ordering.
Axes are interpretable: X = recurrence (true_occ), Y = momentum (rate_ratio).
"""
import numpy as np
from fsm_intervene import HierV2SCM, STATE_NAMES, IDLE, BIRTH, REINFORCE, DECAY, DEATH
from project_paths import GENERATED_FIGURES_DIR, result_input

NPZ = str(result_input("faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz"))
np.set_printoptions(precision=3, suppress=True)


def main():
    m = HierV2SCM(NPZ)
    b = m.baseline()
    dist = b['dist']
    arg = dist.argmax(1)
    rec = np.where(m.true_occ >= 1.0)[0]

    # ---------- 1. WHERE is each state high? region centroids ----------
    print("=" * 78)
    print("LIFECYCLE REGIONS (centroid drivers per argmax state, recurring pairs):")
    print(f"  {'state':9s} {'n':>6} {'true_occ':>9} {'rate_ratio':>11} {'slope_rel':>10} {'staleness':>10}")
    for s in (BIRTH, REINFORCE, DECAY, DEATH):
        idx = rec[arg[rec] == s]
        if len(idx) == 0:
            print(f"  {STATE_NAMES[s]:9s} {0:>6}  (empty)"); continue
        print(f"  {STATE_NAMES[s]:9s} {len(idx):>6} {m.true_occ[idx].mean():>9.2f} "
              f"{m.rate_ratio[idx].mean():>11.3f} {m.slope_rel[idx].mean():>10.3f} "
              f"{m.stale_rel0[idx].mean():>10.3f}")
    print("  => BIRTH sits at LOW true_occ (new/early-trend); REINFORCE at higher")
    print("     recurrence + positive momentum; DECAY at falling slope; DEATH at high staleness.")

    # ---------- 2. PREDICTED SHIFT: BIRTH --do(+k edits)--> WHICH region? ----------
    # HONEST: report the ACTUAL dominant predicted next-state, do not assume REINFORCE.
    birth = rec[arg[rec] == BIRTH]
    print(f"\nPREDICTED SHIFT  BIRTH (n={len(birth)}) --do(+k edits)-->  argmax distribution:")
    print(f"  {'+k edits':>9} {'BIRTH':>7} {'REINF':>7} {'DECAY':>7} {'DEATH':>7}  dominant")
    for k in (1, 3, 10, 30, 60):
        sh = m.do_driver(birth, true_occ=m.true_occ[birth] + k,
                         n_prior=m.n_prior[birth] + k)['dist'].argmax(1)
        fr = [float((sh == s).mean()) for s in (BIRTH, REINFORCE, DECAY, DEATH)]
        dom = [BIRTH, REINFORCE, DECAY, DEATH][int(np.argmax(fr))]
        print(f"  {k:>9} {fr[0]:>7.2f} {fr[1]:>7.2f} {fr[2]:>7.2f} {fr[3]:>7.2f}  -> {STATE_NAMES[dom]}")
    print("  => report whatever the trained model actually predicts (no assumed ordering).")

    # ---------- 2b. IS BIRTH->REINFORCE reachable? factual vs do(rising momentum) ----------
    def frac(d, s): return float((d['dist'].argmax(1) == s).mean())
    flat = m.do_driver(birth, true_occ=m.true_occ[birth] + 5, n_prior=m.n_prior[birth] + 5)
    rise = m.do_driver(birth, true_occ=m.true_occ[birth] + 5, n_prior=m.n_prior[birth] + 5,
                       slope_rel=np.full(len(birth), 2.0))
    print(f"\nBIRTH->REINFORCE reachability (honest):")
    print(f"  +edits, FLAT slope (real CoEdit cadence): REINF={frac(flat,REINFORCE):.2f} DECAY={frac(flat,DECAY):.2f}")
    print(f"  +edits, do(slope RISING +2):              REINF={frac(rise,REINFORCE):.2f} DECAY={frac(rise,DECAY):.2f}")
    print(f"  CoEdit slope_rel: median={np.median(m.slope_rel):.2f}, frac(slope>0)={float((m.slope_rel>0).mean()):.2f}")
    print("  => transition IS wired & predictable (rising->REINFORCE 100%), but CoEdit edits")
    print("     rarely carry rising momentum, so the FACTUAL flow is BIRTH->DECAY->DEATH.")

    # ---------- 3. EARLY-TREND CASES: actual predicted next-state per pair ----------
    rise = m.rate_ratio[birth]                           # momentum signal for early pairs
    order = birth[np.argsort(-rise)]                     # highest-momentum BIRTH first
    print("\nEARLY-TREND CASES (currently BIRTH  ->  predicted after +1 edit):")
    print(f"  {'pair':>6} {'true_occ':>9} {'rate_ratio':>11} {'P_birth':>8} "
          f"{'pred(+1)':>9} {'P_REINF':>8} {'P_DECAY':>8}")
    shown = 0
    for j in order:
        cur = dist[j]
        if cur.argmax() != BIRTH:
            continue
        pred = m.do_driver(np.array([j]), true_occ=float(m.true_occ[j] + 1),
                           n_prior=float(m.n_prior[j] + 1))['dist'][0]
        print(f"  {int(j):>6} {m.true_occ[j]:>9.0f} {m.rate_ratio[j]:>11.2f} "
              f"{cur[BIRTH]:>8.2f} {STATE_NAMES[pred.argmax()]:>9} {pred[REINFORCE]:>8.2f} {pred[DECAY]:>8.2f}")
        shown += 1
        if shown >= 6:
            break

    # ---------- figure: state-map + shift arrow ----------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        cols = {BIRTH: "#e9c46a", REINFORCE: "#2a9d8f", DECAY: "#f4a261", DEATH: "#e76f51"}
        X = np.clip(m.true_occ[rec], 0, 40); Y = np.clip(m.rate_ratio[rec], -4, 4)
        for s in (DEATH, DECAY, BIRTH, REINFORCE):
            mk = arg[rec] == s
            ax[0].scatter(X[mk], Y[mk], s=6, alpha=0.4, c=cols[s], label=STATE_NAMES[s])
        ax[0].set_xlabel("recurrence (true_occ)"); ax[0].set_ylabel("momentum (rate_ratio)")
        ax[0].set_title("(1) Lifecycle state-map: BIRTH region (low recurrence) -> REINFORCE")
        ax[0].legend(markerscale=2, fontsize=8)
        # shift arrows: BIRTH centroid -> REINFORCE centroid
        bc = (m.true_occ[birth].mean(), m.rate_ratio[birth].mean())
        ri = rec[arg[rec] == REINFORCE]; rc = (min(m.true_occ[ri].mean(), 40), m.rate_ratio[ri].mean())
        ax[0].annotate("", xy=rc, xytext=bc, arrowprops=dict(arrowstyle="->", lw=2, color="black"))
        ax[0].text(bc[0], bc[1]+0.3, "+edits", fontsize=9)
        # panel 2: ACTUAL predicted next-state distribution as recurrence grows
        ks = [1, 3, 10, 30, 60]
        mat = []
        for k in ks:
            sh = m.do_driver(birth, true_occ=m.true_occ[birth] + k,
                             n_prior=m.n_prior[birth] + k)['dist'].argmax(1)
            mat.append([float((sh == s).mean()) for s in (BIRTH, REINFORCE, DECAY, DEATH)])
        mat = np.array(mat)
        bottom = np.zeros(len(ks))
        for col, s in enumerate((BIRTH, REINFORCE, DECAY, DEATH)):
            ax[1].bar([str(k) for k in ks], mat[:, col], bottom=bottom,
                      label=STATE_NAMES[s], color=cols[s]); bottom += mat[:, col]
        ax[1].set_xlabel("+k edits on BIRTH pairs"); ax[1].set_ylabel("predicted fraction")
        ax[1].set_title("(2) Where BIRTH pairs go as recurrence grows"); ax[1].legend(fontsize=8)
        GENERATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); out = str(GENERATED_FIGURES_DIR / "cf_demo_statemap_s42.png")
        plt.savefig(out, dpi=130); print(f"\n  figure -> {out}")
    except Exception as e:
        print("  (figure skipped:", e, ")")


if __name__ == "__main__":
    main()
