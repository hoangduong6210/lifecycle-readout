"""
Offline FAITHFULNESS analysis for the FSM v3 next-state head (coedit).

Reads the per-event .npz produced by SRGNN_v3_3.flush_faithfulness() (eval-only
probe, see models/sr_gnn_v3_3.py forward FAITHFULNESS DUMP block) and answers ONE
question:

  Does the predicted state (argmax s_{t+1}) flip TRUTHFULLY with each pair's OWN
  history, or is it just the batch-wide marginal spread out cosmetically?

Faithful expectation (state reflects the pair's own lifecycle):
  - n_prior < 2 (new pair)              -> BIRTH should dominate
  - |z_pair| ~ 0 (typical gap FOR THIS PAIR) -> REINFORCE
  - z_pair >> 0 (gap long FOR THIS PAIR)     -> DECAY / DEATH rises

Reports:
  (a) contingency table  z-bucket x argmax_state   (row-normalized)
  (b) Spearman rank-corr  z_pair  vs  (p_decay + p_death)
  (c) accuracy of the z-bucket -> expected-state map

Verdict:
  COSMETIC  if state ~ independent of z (corr ~ 0, flat contingency rows)
  FAITHFUL  if corr clearly positive AND contingency is diagonal/staircase.

LOGIN-NODE SAFE: numpy/scipy on the already-saved arrays, single pass. No torch,
no model, no training loop.

Usage:
  python analyze_faithfulness.py <dump.npz> [--z late_thr dead_thr]
  defaults: late_thr=0.7  dead_thr=1.3  (match decol_late_thr / decol_dead_thr)
"""
import sys
import argparse
import numpy as np

IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
STATE_NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]


def spearman(x, y):
    """Spearman rho via Pearson on ranks. scipy-free so it runs anywhere."""
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        # rank-transform + Pearson; p-value left as NaN.
        def rank(a):
            order = a.argsort()
            r = np.empty(len(a), dtype=np.float64)
            r[order] = np.arange(len(a))
            return r
        rx, ry = rank(x), rank(y)
        rx -= rx.mean(); ry -= ry.mean()
        denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        if denom == 0:
            return float("nan"), float("nan")
        return float((rx * ry).sum() / denom), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--late_thr", type=float, default=0.7)
    ap.add_argument("--dead_thr", type=float, default=1.3)
    ap.add_argument("--new_pair_n", type=float, default=2.0,
                    help="n_prior < this => 'new pair' (matches has_hist guard)")
    ap.add_argument("--subset_field", type=str, default="n_prior",
                    help="npz field used to define the RECURRING (>=new_pair_n) headline "
                         "subset for the Spearman rho + trend/slope corrs. PM 2026-06-02: "
                         "use 'true_occ' (reliable count) because n_prior (Welford count) is "
                         "intra-batch-corrupted (caps ~6). Falls back to n_prior if absent.")
    ap.add_argument("--decline_thr", type=float, default=0.3,
                    help="decline >= this => DECLINING trend bucket (matches "
                         "decol_decline_thr); λ this fraction below its own recent peak")
    ap.add_argument("--mu", type=float, default=0.1,
                    help="Hawkes baseline μ (HAWKES_MU); SILENT when carried λ near μ")
    ap.add_argument("--silence_margin", type=float, default=0.15,
                    help="carried λ <= μ+margin => SILENT bucket (matches "
                         "decol_silence_margin)")
    ap.add_argument("--slope_margin", type=float, default=0.05,
                    help="SLOPE dead-band (matches decol_slope_margin): slope>=+margin "
                         "=> RISING, slope<=-margin => FALLING")
    ap.add_argument("--rate_dead", type=float, default=0.25,
                    help="carried λ <= rate_dead => DEAD (matches decol_rate_dead); "
                         "FALLING-active requires λ above this. ABSOLUTE fallback only.")
    # PER-PAIR-RELATIVE bucketing (PM 2026-06-01 fix). When the dump carries slope_rel /
    # rate_peak (decol_rate_relative on), bucket by % rate change and the per-pair γ·peak
    # DEAD floor instead of the off-scale absolutes above.
    ap.add_argument("--margin_rel", type=float, default=0.15,
                    help="RELATIVE slope dead-band (matches decol_margin_rel): "
                         "slope_rel>=+margin_rel => RISING, <=-margin_rel => FALLING")
    ap.add_argument("--rate_dead_gamma", type=float, default=0.20,
                    help="per-pair DEAD floor fraction (matches decol_rate_dead_gamma): "
                         "rate_fast < gamma*rate_peak_pair => DEAD")
    args = ap.parse_args()

    d = np.load(args.npz)
    z      = d["z_pair"].astype(np.float64)          # per-pair z; == the z the v3 target uses
    n_prior = d["n_prior"].astype(np.float64)
    # RECURRING-subset counter for the headline rho. n_prior (Welford count) is
    # intra-batch-corrupted (caps ~6); PM 2026-06-02 wants true_occ>=2 as the headline.
    sf = args.subset_field
    if sf in d.files:
        recur = d[sf].astype(np.float64)
        recur_name = sf
    else:
        recur = n_prior
        recur_name = "n_prior(fallback)"
    st     = d["argmax_s_t1_pos"].astype(np.int64)
    p_decay = d["p_decay"].astype(np.float64)
    p_death = d["p_death"].astype(np.float64)
    N = len(z)

    # ── data hygiene (report, don't fabricate) ──────────────────────────────
    nan_mask = ~(np.isfinite(z) & np.isfinite(p_decay)
                 & np.isfinite(p_death) & np.isfinite(n_prior))
    n_nan = int(nan_mask.sum())
    print("=" * 72)
    print(f"FAITHFULNESS REPORT  ({args.npz})")
    print(f"N events = {N}   non-finite rows = {n_nan} (excluded)")
    keep = ~nan_mask
    z, n_prior, st = z[keep], n_prior[keep], st[keep]
    recur = recur[keep]
    p_decay, p_death = p_decay[keep], p_death[keep]
    N = len(z)
    # The headline RECURRING subset mask (PM 2026-06-02: true_occ>=2 is the CORRECT
    # subset; n_prior>=2 was the BROKEN subset used in prior rounds).
    recur_hist = recur >= args.new_pair_n
    print(f"recurring subset field = {recur_name}  ( >= {args.new_pair_n:g} : "
          f"n={int(recur_hist.sum())} of {N} = {100*recur_hist.mean():.1f}% )")

    print(f"z_pair : min {z.min():+.3f}  p10 {np.percentile(z,10):+.3f}  "
          f"med {np.median(z):+.3f}  p90 {np.percentile(z,90):+.3f}  max {z.max():+.3f}")
    print(f"n_prior: 0..1 {np.mean(n_prior < args.new_pair_n)*100:.1f}%   "
          f">=2 {np.mean(n_prior >= args.new_pair_n)*100:.1f}%")
    marg = np.bincount(st, minlength=5) / max(N, 1)
    print("marginal argmax dist [I,B,R,D,Dt] = "
          + "[" + ", ".join(f"{m:.3f}" for m in marg) + "]")

    # ── z-buckets (aligned to the de-collapse target thresholds) ────────────
    # NEW    : new pair (history too short for a trustworthy sigma)
    # TYPICAL: |z| small -> gap is normal FOR THIS PAIR        -> expect REINFORCE
    # LATE   : z above late_thr (below dead_thr)               -> expect DECAY
    # DEAD   : z above dead_thr                                -> expect DEATH
    bucket = np.empty(N, dtype="<U8")
    is_new = n_prior < args.new_pair_n
    bucket[is_new] = "NEW"
    rest = ~is_new
    bucket[rest & (z < args.late_thr)] = "TYPICAL"
    bucket[rest & (z >= args.late_thr) & (z < args.dead_thr)] = "LATE"
    bucket[rest & (z >= args.dead_thr)] = "DEAD"

    order = ["NEW", "TYPICAL", "LATE", "DEAD"]
    expected = {"NEW": BIRTH, "TYPICAL": REINFORCE, "LATE": DECAY, "DEAD": DEATH}

    # ── (a) row-normalized contingency table ────────────────────────────────
    print("\n(a) CONTINGENCY  z-bucket x argmax_state  (row-normalized; n in [])")
    print(f"    {'bucket':<8} {'n':>7}  " + "  ".join(f"{s:>9}" for s in STATE_NAMES))
    for b in order:
        m = bucket == b
        nb = int(m.sum())
        if nb == 0:
            print(f"    {b:<8} {nb:>7}  " + "  ".join(f"{'-':>9}" for _ in STATE_NAMES))
            continue
        row = np.bincount(st[m], minlength=5) / nb
        cells = "  ".join(f"{row[k]:9.3f}" for k in range(5))
        exp = STATE_NAMES[expected[b]]
        print(f"    {b:<8} {nb:>7}  {cells}   <- expect {exp}")

    # ── (a') CALIBRATED contingency (post per-class argmax bias) ─────────────
    # Present only when the run carried the v3 argmax-bias calibration; this is the
    # committed symbolic state the gate reads. Shows whether DECAY/DEATH now WIN argmax
    # in the LATE/DEAD buckets after calibration (the PM under-fire fix). AP is proven
    # invariant to this bias (ML CPU verify 2026-06-01) — it only moves the argmax.
    if "argmax_s_t1_cal" in d.files:
        stc = d["argmax_s_t1_cal"].astype(np.int64)[keep]
        margc = np.bincount(stc, minlength=5) / max(N, 1)
        print("\n(a') CALIBRATED contingency  z-bucket x argmax_cal  (row-normalized)")
        print(f"    calibrated marginal [I,B,R,D,Dt] = "
              + "[" + ", ".join(f"{m:.3f}" for m in margc) + "]")
        # PM 2026-06-02: calibrated argmax marginal on the CORRECT recurring subset
        # (true_occ>=2). REINFORCE must be ALIVE here (~95% target), DEATH ~5%.
        rh = recur >= args.new_pair_n
        if rh.sum() > 0:
            mrc = np.bincount(stc[rh], minlength=5) / int(rh.sum())
            print(f"    calibrated marginal [{recur_name}>=2, n={int(rh.sum())}] "
                  + "[I,B,R,D,Dt] = ["
                  + ", ".join(f"{m:.3f}" for m in mrc) + "]  <- HEADLINE recurring subset")
        print(f"    {'bucket':<8} {'n':>7}  " + "  ".join(f"{s:>9}" for s in STATE_NAMES))
        for b in order:
            m = bucket == b
            nb = int(m.sum())
            if nb == 0:
                print(f"    {b:<8} {nb:>7}  " + "  ".join(f"{'-':>9}" for _ in STATE_NAMES))
                continue
            row = np.bincount(stc[m], minlength=5) / nb
            cells = "  ".join(f"{row[k]:9.3f}" for k in range(5))
            exp = STATE_NAMES[expected[b]]
            acc = row[expected[b]]
            print(f"    {b:<8} {nb:>7}  {cells}   <- expect {exp} (hit {acc:.3f})")

    # ── (a'') TREND bucketing  (PM 2026-06-01: DECAY = λ rolling off the pair's own
    #         recent peak, NOT absolute gap length). The z-bucket view above buckets by
    #         gap MAGNITUDE; this view buckets by the per-pair λ TREND `decline`. The PM
    #         redefinition expects: DECLINING events (λ down off peak, still alive) →
    #         DECAY argmax; only SILENT events (extreme z) → DEATH. Present only when the
    #         run carried the v3 λ-trend (decline field). Uses argmax_cal if available.
    if "decline" in d.files:
        dec = d["decline"].astype(np.float64)[keep]
        stt = (d["argmax_s_t1_cal"].astype(np.int64)[keep]
               if "argmax_s_t1_cal" in d.files else st)
        cal_tag = "cal" if "argmax_s_t1_cal" in d.files else "raw"
        # TREND buckets (n_prior>=2 only — decline needs a trustworthy peak):
        #   STEADY    : on rhythm, λ near peak (decline < decline_thr)       -> REINFORCE
        #   DECLINING : λ down off peak BUT carried λ still alive (>silence)  -> DECAY
        #   SILENT    : carried Hawkes λ collapsed to ~μ (absolute silence)   -> DEATH
        # SILENT is keyed on the ABSOLUTE carried λ (lam_carried -> μ), matching the
        # model's is_silent gate — NOT on z. A tight-history pair with a merely-stretched
        # gap (high z, λ still well above μ) is DECLINING, not SILENT. Falls back to the
        # z>=dead_thr rule only if lam_carried was not dumped (older npz).
        hh = n_prior >= args.new_pair_n
        if "lam_carried" in d.files:
            lc = d["lam_carried"].astype(np.float64)[keep]
            silent_mask = lc <= (args.mu + args.silence_margin)
        else:
            silent_mask = z >= args.dead_thr
        tb = np.empty(N, dtype="<U10")
        tb[:] = "NEW"
        tb[hh & silent_mask] = "SILENT"
        tb[hh & (~silent_mask) & (dec >= args.decline_thr)] = "DECLINING"
        tb[hh & (~silent_mask) & (dec < args.decline_thr)] = "STEADY"
        torder = ["NEW", "STEADY", "DECLINING", "SILENT"]
        texp = {"NEW": BIRTH, "STEADY": REINFORCE, "DECLINING": DECAY, "SILENT": DEATH}
        print(f"\n(a'') TREND contingency  trend-bucket x argmax_{cal_tag}  (row-norm)")
        print(f"    decline: med {np.median(dec[hh]) if hh.any() else float('nan'):.3f}  "
              f"p90 {np.percentile(dec[hh],90) if hh.any() else float('nan'):.3f}  "
              f"(thr={args.decline_thr})")
        print(f"    {'bucket':<10} {'n':>7}  " + "  ".join(f"{s:>9}" for s in STATE_NAMES))
        for b in torder:
            m = tb == b
            nb = int(m.sum())
            if nb == 0:
                print(f"    {b:<10} {nb:>7}  " + "  ".join(f"{'-':>9}" for _ in STATE_NAMES)
                      + f"   <- expect {STATE_NAMES[texp[b]]}")
                continue
            row = np.bincount(stt[m], minlength=5) / nb
            cells = "  ".join(f"{row[k]:9.3f}" for k in range(5))
            print(f"    {b:<10} {nb:>7}  {cells}   <- expect {STATE_NAMES[texp[b]]} "
                  f"(hit {row[texp[b]]:.3f})")

    # ── (a''') SLOPE bucketing  (PM 2026-06-01 THIRD re-chốt: all 3 active states on
    #          ONE signed axis = the slope of the edit-RATE = fast(carried λ) − slow
    #          (slow-EWMA λ)). This is now the AUTHORITATIVE view — the discriminator
    #          REINFORCE↔DECAY is the SIGN of the slope, not gap magnitude (a) nor the
    #          level-off-peak `decline` (a''). Expects:
    #            RISING         (slope >= +margin)              -> REINFORCE
    #            FALLING-active (slope <= −margin & λ>rate_dead) -> DECAY
    #            DEAD           (λ <= rate_dead)                 -> DEATH
    #          The key win vs legacy diagnostic (FALLING swallowed by DEATH): FALLING-active
    #          should now show argmax DECAY mass > 0, NOT ~0.2%, and not collapse to DEATH.
    if "lam_slope" in d.files:
        # PER-PAIR-RELATIVE (PM 2026-06-01 fix): prefer slope_rel = (fast−slow)/(slow+ε)
        # (% rate change) and the per-pair γ·rate_peak DEAD floor when the dump carries
        # them. Falls back to the off-scale absolute lam_slope / rate_dead otherwise.
        rel_mode = ("slope_rel" in d.files) and ("rate_peak" in d.files)
        rf = (d["rate_fast"].astype(np.float64)[keep]
              if "rate_fast" in d.files else
              (d["lam_carried"].astype(np.float64)[keep]
               if "lam_carried" in d.files else np.full(N, 1.0)))
        if rel_mode:
            slp = d["slope_rel"].astype(np.float64)[keep]
            rpk = d["rate_peak"].astype(np.float64)[keep]
            dead_floor = args.rate_dead_gamma * np.maximum(rpk, 1e-6)
            dead_mask  = rf < dead_floor
            _margin    = args.margin_rel
            _slope_lbl = "slope_rel(%)"
            _dead_lbl  = f"gamma={args.rate_dead_gamma}*rate_peak"
        else:
            # ABSOLUTE fallback. DEAD keyed on the RATE level (rate_fast), NOT the
            # cumulative Hawkes λ (which rises through accel+decel alike).
            slp = d["lam_slope"].astype(np.float64)[keep]
            dead_mask = rf <= args.rate_dead
            _margin    = args.slope_margin
            _slope_lbl = "slope(abs)"
            _dead_lbl  = f"rate_dead={args.rate_dead}"
        stt = (d["argmax_s_t1_cal"].astype(np.int64)[keep]
               if "argmax_s_t1_cal" in d.files else st)
        cal_tag = "cal" if "argmax_s_t1_cal" in d.files else "raw"
        hh = n_prior >= args.new_pair_n
        sb = np.empty(N, dtype="<U14")
        sb[:] = "NEW"
        sb[hh & dead_mask] = "DEAD"
        sb[hh & (~dead_mask) & (slp >= _margin)] = "RISING"
        sb[hh & (~dead_mask) & (slp <= -_margin)] = "FALLING-active"
        sb[hh & (~dead_mask) & (slp > -_margin)
                            & (slp < _margin)] = "FLAT-active"
        sorder = ["NEW", "RISING", "FLAT-active", "FALLING-active", "DEAD"]
        sexp = {"NEW": BIRTH, "RISING": REINFORCE, "FLAT-active": REINFORCE,
                "FALLING-active": DECAY, "DEAD": DEATH}
        print(f"\n(a''') SLOPE contingency  slope-bucket x argmax_{cal_tag}  (row-norm)  "
              f"[{'PER-PAIR-RELATIVE' if rel_mode else 'ABSOLUTE'}]")
        if hh.any():
            print(f"    {_slope_lbl}: p10 {np.percentile(slp[hh],10):+.3f}  "
                  f"med {np.median(slp[hh]):+.3f}  p90 {np.percentile(slp[hh],90):+.3f}  "
                  f"(margin=±{_margin}, DEAD: {_dead_lbl})")
        print(f"    {'bucket':<14} {'n':>7}  " + "  ".join(f"{s:>9}" for s in STATE_NAMES))
        for b in sorder:
            m = sb == b
            nb = int(m.sum())
            if nb == 0:
                print(f"    {b:<14} {nb:>7}  " + "  ".join(f"{'-':>9}" for _ in STATE_NAMES)
                      + f"   <- expect {STATE_NAMES[sexp[b]]}")
                continue
            row = np.bincount(stt[m], minlength=5) / nb
            cells = "  ".join(f"{row[k]:9.3f}" for k in range(5))
            print(f"    {b:<14} {nb:>7}  {cells}   <- expect {STATE_NAMES[sexp[b]]} "
                  f"(hit {row[sexp[b]]:.3f})")

    # ── (b) Spearman z vs (p_decay + p_death) ───────────────────────────────
    # Spearman is rank-based → invariant to the (huge) z magnitudes that fresh
    # pairs get from a near-zero σ, so the FULL-sample rho is well-defined. We ALSO
    # report it restricted to n_prior>=2, the regime where σ is trustworthy and the
    # per-pair faithfulness claim actually lives (matches the model's has_hist guard).
    p_late = p_decay + p_death
    rho, pval = spearman(z, p_late)
    rho_h, _ = spearman(z, p_death)
    hist = recur_hist                      # PM 2026-06-02: headline subset = true_occ>=2
    sub_tag = f"{recur_name}>=2"
    if hist.sum() >= 3:
        rho_h2, pval_h2 = spearman(z[hist], p_late[hist])
    else:
        rho_h2, pval_h2 = float("nan"), float("nan")
    # also report the OLD (broken) n_prior>=2 subset for cross-round comparison
    np_hist = n_prior >= args.new_pair_n
    if np_hist.sum() >= 3:
        rho_np2, _ = spearman(z[np_hist], p_late[np_hist])
    else:
        rho_np2 = float("nan")
    print("\n(b) SPEARMAN rank-correlation")
    print(f"    z_pair vs (p_decay+p_death) [all]      : rho = {rho:+.4f}"
          + (f"  (p={pval:.2e})" if np.isfinite(pval) else ""))
    print(f"    z_pair vs (p_decay+p_death) [{sub_tag}]: rho = {rho_h2:+.4f}"
          + (f"  (p={pval_h2:.2e})" if np.isfinite(pval_h2) else "")
          + f"   (n={int(hist.sum())})  <- HEADLINE")
    print(f"    z_pair vs (p_decay+p_death) [n_prior>=2 OLD/broken]: rho = {rho_np2:+.4f}"
          + f"   (n={int(np_hist.sum())})")
    print(f"    z_pair vs  p_death          [all]      : rho = {rho_h:+.4f}")
    print("    (faithful => clearly POSITIVE: longer-than-usual gap -> more decay/death)")
    # ── TREND Spearman: decline vs p_decay (PM 2026-06-01 DECAY axis). The DECAY
    #    state should track the λ-TREND specifically, distinct from p_death which
    #    tracks absolute silence (z). If decline->p_decay rho is positive AND
    #    decline->p_death is ~0/weaker, DECAY and DEATH are reading DIFFERENT signals
    #    (= the separation the PM redefinition wants), not collapsing into one band.
    if "decline" in d.files:
        dec = d["decline"].astype(np.float64)[keep]
        if hist.sum() >= 3:
            rho_dd, p_dd = spearman(dec[hist], p_decay[hist])
            rho_dt, _    = spearman(dec[hist], p_death[hist])
            print(f"    decline vs p_decay          [{sub_tag}]: rho = {rho_dd:+.4f}"
                  + (f"  (p={p_dd:.2e})" if np.isfinite(p_dd) else "")
                  + f"   (n={int(hist.sum())})")
            print(f"    decline vs p_death          [{sub_tag}]: rho = {rho_dt:+.4f}"
                  "   (want ~0/weaker than decline->p_decay = DECAY≠DEATH separation)")
    # ── SLOPE Spearman (PM 2026-06-01 third re-chốt, the authoritative axis): a
    #    NEGATIVE slope (rate falling) should track p_decay (DECAY), and a POSITIVE
    #    slope (rate rising) should track p_reinforce. So want corr(slope, p_reinforce)
    #    POSITIVE and corr(slope, p_decay) NEGATIVE — the sign flip that proves
    #    REINFORCE↔DECAY is governed by the slope SIGN, not λ magnitude.
    if "lam_slope" in d.files and "p_reinforce" in d.files:
        slp = d["lam_slope"].astype(np.float64)[keep]
        p_reinf = d["p_reinforce"].astype(np.float64)[keep]
        if hist.sum() >= 3:
            rho_sr, p_sr = spearman(slp[hist], p_reinf[hist])
            rho_sd, _    = spearman(slp[hist], p_decay[hist])
            print(f"    slope vs p_reinforce        [{sub_tag}]: rho = {rho_sr:+.4f}"
                  + (f"  (p={p_sr:.2e})" if np.isfinite(p_sr) else "")
                  + "   (want POSITIVE: rising rate -> REINFORCE)")
            print(f"    slope vs p_decay            [{sub_tag}]: rho = {rho_sd:+.4f}"
                  "   (want NEGATIVE: falling rate -> DECAY = slope-sign discriminator)")
    # the n_prior>=2 corr is the headline faithfulness number
    rho = rho_h2 if np.isfinite(rho_h2) else rho

    # ── (c) accuracy of z-bucket -> expected-state map ──────────────────────
    exp_state = np.array([expected[b] for b in bucket])
    acc = float(np.mean(st == exp_state))
    # chance = sum over buckets of bucket_frac * marginal_prob(expected_state)
    chance = 0.0
    for b in order:
        fb = np.mean(bucket == b)
        chance += fb * marg[expected[b]]
    print("\n(c) z-bucket -> expected-state ACCURACY")
    print(f"    accuracy = {acc:.4f}   (chance ~ {chance:.4f})")
    per = {}
    for b in order:
        m = bucket == b
        per[b] = float(np.mean(st[m] == expected[b])) if m.any() else float("nan")
    print("    per-bucket recall: "
          + "  ".join(f"{b}={per[b]:.3f}" for b in order))

    # ── verdict ─────────────────────────────────────────────────────────────
    diag_strength = acc - chance
    print("\nVERDICT")
    cosmetic = (abs(rho) < 0.05) and (diag_strength < 0.05)
    faithful = (rho > 0.15) and (diag_strength > 0.10)
    if cosmetic:
        v = "COSMETIC (state ~ independent of per-pair z; corr~0, flat contingency)"
    elif faithful:
        v = "FAITHFUL (positive z->decay/death corr AND diagonal contingency)"
    else:
        v = "PARTIAL / INCONCLUSIVE (weak but non-zero signal)"
    print(f"    rho={rho:+.4f}  acc-chance={diag_strength:+.4f}  ->  {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()
