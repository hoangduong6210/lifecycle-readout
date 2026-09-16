"""
RS-GNN v3.3 — Transition-Aware + Lifecycle-Filtered Gradient (LFG).

CANONICAL MODEL = DETACHED READOUT  (enable_main_predictor=False, the default).
  The link-prediction logit is produced by the lifecycle / existence-decoder
  readout operating on DETACHED backbone features. The backbone (CSN/ECTG/DRGC)
  is shaped by the continuous laws + the VAE-KL / TIP parsimony term
  (lambda_echo·KL), NOT by the link-prediction BCE. Empirically this decoupled
  head generalizes BETTER inductively than routing BCE into the backbone — the
  3-dataset / 3-seed A/B (2026-05-30) confirmed the detached arm wins every
  dataset (wiki +0.59%, mooc +0.26%, coedit +9.4% Ind-AP). The "detach" is an
  intentional design choice, not a bug.

Pipeline (canonical, detached):
  Backbone (v3.1 lean continuous laws) → h_uv   [shaped by KL/TIP, not by BCE]
       ↓ (features read with stop-grad by the readout stream)
  ┌─── (stop-grad) ────────────────────────────┐
  │ State Observer → s_t                        │
  │ Transition Predictor + FSM Mask → s_{t+1}  │
  │ Existence Decoder → P(edge | symbolic)      │  ← link-prediction logit
  │ LFG Compliance Score → c                    │
  └─────────────────────────────────────────────┘
       ↓
  L = c · BCE(existence_logit, label)       (readout on detached features)
      + λ_echo · KL  + λ_mask · L_violation

PREDICTION-HEAD ABLATION = END-TO-END  (enable_main_predictor=True).
  Toggled via the benchmark runner's --p0_fix {on,off,both}. This adds a
  NON-detached Main Edge Predictor on the post-DRGC embeddings and routes the
  link-prediction BCE (+LFG) through it, so the backbone IS trained end-to-end
  by link prediction. This is our headline prediction-head ablation (decoupled
  readout vs end-to-end backbone training); empirically it HURTS inductive AP
  (it was the original "P0 fix"). Kept fully runnable for the ablation table.

Key properties:
  1. FSM symbolic stream is fully decoupled (stop-grad) — interpretation +,
     in the canonical arm, the prediction readout.
  2. The end-to-end Main Edge Predictor (ablation arm only) carries the
     link-prediction signal into DRGC/ECTG/CSN.
  3. Gradient gating (LFG) filters rule-violating events.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from .sr_gnn import (
    IDLE, BIRTH, REINFORCE, DECAY, DEATH,
    TimeEncoder, NodeMemoryStore,
    VALID_TRANSITIONS,
)
from .sr_gnn_v2 import ResidualCSN, DRGC_v2
from .sr_gnn_v3 import (
    EdgeStateStoreV3, ECTGv3,
    HAWKES_MU as HAWKES_MU_DECOL,      # baseline rate (0.1)  — for v3 λ-trend
    HAWKES_BETA as HAWKES_BETA_DECOL,  # inter-event decay (0.01) — for v3 λ-trend
    RATE_INIT,                          # rate-EWMA baseline (per-pair rel gating)
)
from .fsm_head import (
    StateObserver, TransitionPredictor, LifecycleFSMMask,
    ExistenceDecoder, compute_compliance,
    CAUSAL_RULE_MATRIX, compute_causal_validity,
)

# ─────────────────────────────────────────────────────────────────────────────
# HƯỚNG A (PM 2026-06-06): STRICT-ORDERED 6-STATE LIFECYCLE FSM
# ─────────────────────────────────────────────────────────────────────────────
# Design-smell fix: the legacy FSM special-cased "đã-từng-sống" with a SEPARATE
# `ever_alive` gate (phi-Markov — the next-state depended on an out-of-state
# accumulator, not on s_t). HƯỚNG A makes the policy STRICTLY MARKOVIAN by
# SPLITTING the single IDLE state into TWO causally-distinct states so that
# "ever-alive" is represented INSIDE the state itself:
#   PRE_BIRTH : NEVER been alive (no BIRTH yet) — the honest pre-birth state.
#   DORMANT   : WAS alive, now silent — "đã sống, đang im" (not yet dead).
# With the two split, DEATH becomes structurally unreachable from PRE_BIRTH
# WITHOUT the separate ever_alive gate (it must traverse the lifecycle axis),
# so the ever_alive gate becomes REDUNDANT (probe-verified below / on CPU).
#
# LIFECYCLE AXIS (index order chosen so adjacency == |i−j|≤1):
#   PRE_BIRTH(0) ↔ BIRTH(1) ↔ REINFORCE(2) ↔ DECAY(3) ↔ DORMANT(4) ↔ DEATH(5)
# DORMANT is inserted between DECAY and DEATH (PM: "giữa DECAY↔DEATH"): a
# decayed pair goes silent (DORMANT) BEFORE it can die, and can revive back
# down the axis one rung at a time (DORMANT→DECAY→REINFORCE…). DEATH is reached
# ONLY from DORMANT; DORMANT only from DECAY; PRE_BIRTH only reaches BIRTH.
SO_PRE_BIRTH, SO_BIRTH, SO_REINFORCE, SO_DECAY, SO_DORMANT, SO_DEATH = 0, 1, 2, 3, 4, 5
SO_STATE_NAMES = ["PRE_BIRTH", "BIRTH", "REINFORCE", "DECAY", "DORMANT", "DEATH"]
SO_N_STATES = 6

# C' ∈ {0,1}^{6×6} — STRICT band-diagonal causal admissibility on the lifecycle
# axis. C'[i,j]=1 iff |i−j|≤1 (self-loop + single adjacent rung), else 0.
# EVERY non-adjacent ("nhảy cóc") transition is FORBIDDEN, e.g.
#   PRE_BIRTH→DEATH, BIRTH→DEATH, REINFORCE→DEATH, PRE_BIRTH→REINFORCE,
#   BIRTH→DECAY, REINFORCE→DORMANT … all = 0. DEATH is reached ONLY from
#   DORMANT (its single lower neighbour). Self-loops are admissible everywhere.
#         PRE  BIR  REI  DEC  DOR  DEA   (to →)
STRICT_ORDERED_C = torch.tensor([
    [ 1,   1,   0,   0,   0,   0 ],  # PRE_BIRTH : stay / be born only
    [ 1,   1,   1,   0,   0,   0 ],  # BIRTH     : regress(pre) / stay / reinforce
    [ 0,   1,   1,   1,   0,   0 ],  # REINFORCE : birth / stay / decay
    [ 0,   0,   1,   1,   1,   0 ],  # DECAY     : reinforce / stay / dormant
    [ 0,   0,   0,   1,   1,   1 ],  # DORMANT   : decay(revive) / stay / DIE
    [ 0,   0,   0,   0,   1,   1 ],  # DEATH     : dormant(revive) / stay dead
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OPTION (b) — STRICT-ORDERED 5-STATE NATIVE (PM 2026-06-06): NO IDLE split.
# ─────────────────────────────────────────────────────────────────────────────
# PM verdict after the 6-state probe: on coedit PRE_BIRTH/DORMANT mass ≈ 0 (the
# IDLE split was dead weight), so we keep the NATIVE 5-state lifecycle and just
# impose the SAME strict band-diagonal "no nhảy cóc" rule directly on it.
#
# LIFECYCLE AXIS (5-state native, index order == the legacy IDLE,BIRTH,…,DEATH):
#   IDLE(0) ↔ BIRTH(1) ↔ REINFORCE(2) ↔ DECAY(3) ↔ DEATH(4)
# C_BAND_5 ∈ {0,1}^{5×5}, C_BAND_5[i,j]=1 iff |i−j|≤1 (self-loop + single
# adjacent rung), else 0. EVERY non-adjacent jump is FORBIDDEN, e.g.
#   IDLE→{REINFORCE,DECAY,DEATH}, BIRTH→{DECAY,DEATH}, REINFORCE→DEATH,
#   DECAY→{IDLE,BIRTH}, DEATH→{IDLE,BIRTH,REINFORCE} … all = 0.
# KEY CONSEQUENCE: IDLE and DEATH sit at the two ENDS of the axis (|0−4|=4) ⇒
#   IDLE→DEATH is band-blocked. "Death-before-alive" is therefore handled by the
#   ORDERING ITSELF — no IDLE split, NO ever_alive gate needed. A pair that is
#   genuinely dying travels DECAY(3)→DEATH(4), an ordinary adjacent rung.
# NOTE — this is STRICTER than the legacy CAUSAL_RULE_MATRIX / VALID_TRANSITIONS,
#   which allowed IDLE→BIRTH plus revival edges like DEATH→BIRTH. Under C_BAND_5
#   DEATH revival is single-rung only: DEATH→DECAY→REINFORCE (2 hops), no direct
#   DEATH→BIRTH. Confirmed change vs the older matrices.
#         IDL  BIR  REI  DEC  DEA   (to →)
C_BAND_5 = torch.tensor([
    [ 1,   1,   0,   0,   0 ],  # IDLE      : stay / be born only
    [ 1,   1,   1,   0,   0 ],  # BIRTH     : regress(idle) / stay / reinforce
    [ 0,   1,   1,   1,   0 ],  # REINFORCE : birth / stay / decay
    [ 0,   0,   1,   1,   1 ],  # DECAY     : reinforce / stay / die
    [ 0,   0,   0,   1,   1 ],  # DEATH     : decay(revive) / stay dead
], dtype=torch.float32)


class EverAliveStore:
    """Per-edge accumulator: max(p(BIRTH) + p(REINFORCE)) over history.

    Vectorized implementation (P1 perf fix, 2026-05-29). Keyed by the dense hash
    ``key = u * N + v`` (same key scheme as the rest of the model). State lives in
    two flat tensors of length ``N*N``:
      - ``values``     : the accumulated max(alive) per key.
      - ``registered`` : bool, whether a key was ever produced by ``get`` (the old
                         dict membership). ``update`` only writes registered keys,
                         exactly like the old loop which skipped unseen keys.
    All ops are scatter/index operations — no per-event Python loop, no per-event
    ``.item()`` / ``.append()`` — so it runs entirely on CUDA with no host sync.

    Numerics are IDENTICAL to the old per-event loop: ``get`` returns the stored
    value (0 for never-seen keys) and registers the keys; ``update`` takes the
    elementwise max over the batch (handling intra-batch duplicate keys via
    ``scatter_reduce(amax)``) but only for keys that were already registered.
    """
    def __init__(self, num_nodes: int, device: torch.device):
        self.N = num_nodes
        self.device = device
        self._size = num_nodes * num_nodes
        self.values     = torch.zeros(self._size, device=device)
        self.registered = torch.zeros(self._size, dtype=torch.bool, device=device)

    def _keys(self, src: Tensor, dst: Tensor) -> Tensor:
        return src.long() * self.N + dst.long()

    def get(self, src: Tensor, dst: Tensor) -> Tensor:
        keys = self._keys(src, dst)
        out = self.values[keys]
        # Register every key seen here (mirrors the dict insert in the old get()).
        self.registered[keys] = True
        return out

    def peek(self, src: Tensor, dst: Tensor) -> Tensor:
        """READ-ONLY: stored ever_alive per key WITHOUT registering (0 for unseen).
        Used to give the NEGATIVE pathway its OWN real ever_alive (anti-leak fix,
        2026-05-31) instead of a hard-coded 0 — so ever_alive stops being a perfect
        'real edge vs random neg' indicator. Does not mutate ``registered``."""
        return self.values[self._keys(src, dst)]

    def update(self, src: Tensor, dst: Tensor, alive_score: Tensor):
        keys = self._keys(src, dst)
        a = alive_score.detach().clamp(0, 1)
        # Only registered keys may be updated (old loop skipped unseen keys). Build a
        # candidate tensor = max(existing value, batch amax per key) and write it back
        # exclusively at registered positions.
        cand = self.values.clone()
        cand.scatter_reduce_(0, keys, a, reduce="amax", include_self=True)
        write = torch.zeros_like(self.registered)
        write[keys] = True
        write &= self.registered
        self.values = torch.where(write, cand, self.values)

    def reset(self):
        self.values.zero_()
        self.registered.zero_()


class EchoMemory(nn.Module):
    """Per-node accumulated resonance echo (∞-hop via Krylov recursion).

    Faithful port of experiments/models/sr_gnn_v3.py:221-293 (the REAL EchoMemory
    used by the published v3.1-lean numbers), inlined here because the LAB v3.3
    package's own models/sr_gnn_v3.py is the docstring-only copy with no EchoMemory.
    Single-scale only (v3.1-lean ran use_hopfield=False / use_router=False, so the
    multi-scale bank + Hopfield pass + adaptive router are intentionally omitted —
    those were the ablation-killed REACT components, not part of the surviving echo).

    Invariants kept identical:
      - decay() applied BEFORE propagation (anti-staleness)
      - update() called AFTER prediction (anti-leakage)
      - echo & last_t are PLAIN ATTRIBUTES (not buffers) → excluded from state_dict
        so train-time accumulated echo never leaks into test via load_state_dict.
    """
    def __init__(self, num_nodes: int, hidden: int,
                 tau: float = 0.9, lambda_echo: float = 0.01,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.tau = tau
        self.lambda_echo = lambda_echo
        self.device = device
        self.echo   = torch.zeros(num_nodes, hidden, device=device)
        self.last_t = torch.zeros(num_nodes, device=device)

    @torch.no_grad()
    def decay_get(self, idx: Tensor, t_now: Tensor) -> Tensor:
        dt = (t_now - self.last_t[idx]).clamp(min=0.0)
        factor = torch.exp(-self.lambda_echo * dt).unsqueeze(-1)
        return self.echo[idx] * factor

    @torch.no_grad()
    def update(self, u: Tensor, v: Tensor, R_uv: Tensor,
               h_u: Tensor, h_v: Tensor, t_now: Tensor,
               bidirectional: bool = True):
        echo_u_decayed = self.decay_get(u, t_now)
        echo_v_decayed = self.decay_get(v, t_now)
        R = R_uv.unsqueeze(-1)

        delta_u = R * (h_v + echo_v_decayed)
        new_echo_u = self.tau * echo_u_decayed + (1 - self.tau) * delta_u
        self.echo[u] = new_echo_u.detach()
        self.last_t[u] = t_now

        if bidirectional:
            delta_v = R * (h_u + echo_u_decayed)
            new_echo_v = self.tau * echo_v_decayed + (1 - self.tau) * delta_v
            self.echo[v] = new_echo_v.detach()
            self.last_t[v] = t_now

    def reset(self):
        self.echo.zero_()
        self.last_t.zero_()


class SRGNN_v3_3(nn.Module):
    """
    RS-GNN v3.3 — Transition-aware with LFG.

    Hyperparams:
      lambda_trans: weight for transition loss (FSM head training)
      lambda_violation: weight for mask violation penalty
      lambda_fsm: weight for FSM-stream existence BCE (trains FSM head only)
      lambda_echo: TIP/KL regularizer
      lfg_warmup_epochs: epochs before LFG kicks in
      compliance_floor: minimum compliance value (avoid full gate)
      enable_lfg: Lifecycle-Filtered Gradient toggle (default True = CANONICAL).
        True  → positives reweighted by the LFG compliance score (warmup/ramp →
                per-positive compliance ∈ [compliance_floor, 1]); negatives always 1.
        False → ABLATION: uniform weight 1 on ALL events (LFG disabled). Lets
                evaluation harness measure LFG's standalone effect. Everything else (the
                compliance computation for logging) is unchanged; only the weight
                applied to pred_loss is forced to 1.
      fix_existence_init: ExistenceDecoder init toggle (default False = CANONICAL).
        False → original init theta=log(x).clamp(-3) (off-spec effective weights).
        True  → softplus-inverse init so softplus(theta)==[0.1,1,1,0.3,0] exactly
                (the spec); DEATH(x=0)→theta=-10. Interpretability variant (#2a).
      entropy_reg_weight: weight on a symbolic-state entropy regularizer (default
        0.0 = CANONICAL, no term). When > 0, ADD  -entropy_reg_weight * H(state_dist)
        to the total loss, where H is the Shannon entropy of the batch-mean symbolic
        next-state distribution s_{t+1}. The NEGATIVE-entropy term means minimizing
        the loss MAXIMIZES entropy → PUSHES the symbolic stream AWAY from the
        ~0.95-IDLE collapse (encourages a more spread-out state distribution).
        Interpretability variant (#2b).
      enable_echo: EchoMemory toggle (default False = CANONICAL = NO echo).
        False → v3.3 backbone exactly as before (byte-identical code path).
        True  → port of the v3.1 EchoMemory (time-decayed per-node resonance echo,
                learnable echo_gate + echo_norm) injected into h_src/h_dst (and the
                negative dst) BEFORE DRGC, faithful to experiments/models/sr_gnn_v3.py
                (single-scale, bidirectional, @no_grad echo content). Improvement #3.
      enable_main_predictor: prediction-head ablation toggle (exposed as the
        benchmark runner flag --p0_fix {on,off,both}).
        False (DEFAULT, CANONICAL = DETACHED READOUT): link-prediction BCE+LFG
          (pred_loss) routes through the all-detached existence_decoder path
          (pos_logit/neg_logit) → backbone (CSN/ECTG/DRGC) receives ZERO gradient
          from link prediction; it is shaped only by lambda_echo*KL (TIP/VAE
          parsimony) + the continuous laws. pos_score/neg_score = existence-decoder
          logits. main_predictor is unused (no grad to it). No separate fsm_loss
          term (the existence BCE *is* pred_loss here; a separate term would
          double-count). This is the empirically-better arm (3-dataset/3-seed A/B
          2026-05-30: wins all three datasets; coedit +9.4% Ind-AP).
        True  (ABLATION = END-TO-END, the original "P0 fix"): link-prediction
          BCE+LFG routes through the NON-detached main_predictor → backbone IS
          trained end-to-end by link prediction. pos_score/neg_score = main head.
          The FSM stream is supervised separately by fsm_loss (stop-grad, FSM head
          only). Empirically HURTS inductive AP relative to the detached default;
          kept runnable as the headline prediction-head ablation.
        FSM stop-grad behavior is IDENTICAL in both arms — the only difference
        between the two arms is whether the backbone receives link-pred gradient.

      lfg_mode: {"soft","hard"} (default "soft" = CANONICAL, current behavior).
        "soft" → LFG applies the soft compliance scalar ∈ [floor,1] reweight on
                 pred_loss (the existing behavior; nothing changes).
        "hard" → the INTENDED LFG = a HARD causal-rule GRADIENT MASK. A per-event
                 causal-validity v_e ∈ {0,1} is computed from the (detached) FSM
                 transition argmax(s_t)→argmax(s_{t+1}) looked up in the fixed rule
                 matrix C (registered buffer, CAUSAL_RULE_MATRIX). The per-positive
                 gradient weight is m_e = 1 if admissible else compliance_floor
                 (a HARD gate when compliance_floor=0). m_e is detached and
                 MULTIPLIES the per-event loss → it can ONLY mask/attenuate the
                 link-prediction gradient of causally-incoherent events; it never
                 contributes its own gradient and never changes the prediction VALUE.
                 Negatives are always weight 1 (not gated). This is meaningful only
                 when the backbone actually receives link-pred gradient, i.e. with
                 enable_main_predictor=True (Stream 1 trainable).
      design: {"canonical","correct"} composite preset (default "canonical").
        "canonical" → no preset; each individual flag takes its own default → the
                      no-arg model reproduces the canonical detached run byte-for-byte.
        "correct"   → the intended TWO-STREAM CAUSAL-GRADIENT-MASK model: turns ON
                      enable_main_predictor (Stream 1 trainable predictor),
                      lfg_mode="hard" with compliance_floor=0.0 (Stream 2 hard causal
                      gate), revives lambda_trans (transition-CE de-collapse of the
                      FSM), entropy_reg_weight, and fix_existence_init. Individual
                      explicit kwargs still override the preset. `causal_mask=True`
                      is an alias for design="correct".
      causal_mask: bool alias — True ≡ design="correct" (intended full stack).
    """
    def __init__(self, num_nodes: int, feat_dim: int, hidden: int = 128,
                 tip_beta: float = 0.001,
                 lambda_echo: float = 0.01,
                 lambda_trans: float = 0.1,
                 lambda_violation: float = 0.05,
                 lambda_fsm: float = 0.1,
                 lfg_warmup_epochs: int = 2,
                 compliance_floor: float = 0.05,
                 enable_lfg: bool = True,
                 fix_existence_init: bool = False,
                 entropy_reg_weight: float = 0.0,
                 enable_echo: bool = False,
                 echo_tau: float = 0.9,
                 enable_main_predictor: bool = False,
                 lfg_mode: str = "soft",
                 design: str = "canonical",
                 causal_mask: bool = False,
                 # ── De-collapse levers (all OFF/0 by default = CANONICAL) ──
                 fsm_decouple: bool = False,
                 decollapse_target: bool = False,
                 lambda_edge_trans: float = 0.0,
                 edge_state_entropy_w: float = 0.0,
                 edge_uniform_kl_w: float = 0.0,
                 # De-collapse target thresholds (coedit-calibrated; only used when
                 # decollapse_target is on — see the target rebuild block).
                 decol_hawkes_thr: float = 1.05,
                 decol_late_thr: float = 0.7,
                 decol_dead_thr: float = 1.3,
                 # ── Dynamics-aware target + argmax calibration (fsm_arch="v3" ONLY;
                 #    all default OFF/0 ⇒ v1/v2 byte-identical, and v3 unchanged unless
                 #    the v3 enablement block flips them on — see below) ──
                 # decol_use_dynamics: fold the per-pair MOMENTUM (Hawkes carried-
                 #   excitation fraction exp(-β·Δt), the "λ đang rớt" signal) into the
                 #   late/dead drive so DECAY/DEATH sharpen when frequency COLLAPSES, not
                 #   only when the z-gap is large. PM axis #2 (ĐỘNG LỰC).
                 decol_use_dynamics: bool = False,
                 decol_mom_thr: float = 0.5,    # decay_factor below this ⇒ momentum gone
                 # decol_decline_thr: fraction λ must drop OFF the pair's recent leaky
                 #   peak before the trend is read as DECAY (the TREND axis, PM 2026-06-01).
                 #   decline=(λ_peak−λ_carried)/λ_peak ∈[0,1]; is_decline=sigmoid(6·(decline
                 #   −thr)). 0.3 = λ ~30% below its own recent peak ⇒ "đang giảm dần".
                 decol_decline_thr: float = 0.3,
                 # decol_silence_*: DEATH = ABSOLUTE silence — carried Hawkes λ collapsed
                 #   to within `margin` of the baseline μ (≈0.1). is_silent=sigmoid(scale·
                 #   ((μ+margin)−λ_carried)). margin small ⇒ only a near-fully-decayed pair
                 #   (λ≈μ) is "dead"; a tight-history pair with a stretched gap but λ still
                 #   well above μ is DECAYING, not dead (the DECAY≠DEATH separation).
                 decol_silence_margin: float = 0.15,
                 decol_silence_scale: float = 8.0,
                 # ── SLOPE-OF-RATE axis (PM 2026-06-01 THIRD re-chốt) — all 3 active
                 #    states on ONE signed axis = slope of the edit-rate (fast−slow EWMA
                 #    of Hawkes λ). REINFORCE=rising, DECAY=falling-but-alive, DEATH=
                 #    rate≈0 sustained (only from DECAY). REINFORCE is NO LONGER absolute-
                 #    high λ; the discriminator REINFORCE↔DECAY is the SIGN of the slope.
                 # decol_slope_margin: |slope| dead-band (in λ units) around 0 separating
                 #   rising/flat from falling. slope ≥ +margin ⇒ rising; slope ≤ −margin ⇒
                 #   falling. Anchored to the carried-λ scale (μ=0.1, peaks ~1).
                 decol_slope_margin: float = 0.05,
                 decol_slope_scale: float = 8.0,   # sigmoid sharpness on the slope band
                 # decol_rate_dead: carried-λ level below which the pair is "barely
                 #   active" — DECAY requires rate STILL ABOVE this (falling-but-alive);
                 #   DEATH requires rate AT/BELOW it AND sustained (after decline). Sits
                 #   just above μ so a pair whose λ has collapsed to baseline is dead, not
                 #   merely decaying. Replaces the old z-gap DEATH (which swallowed DECAY).
                 decol_rate_dead: float = 0.25,
                 decol_rate_dead_scale: float = 10.0,
                 # ── PER-PAIR-RELATIVE rate gating (PM 2026-06-01 fix) ────────────────
                 # The absolute decol_slope_margin / decol_rate_dead are off the coedit
                 # rate scale (rate=1/Δt med≈0.10, max≈0.195) ⇒ FALLING-active never opened
                 # (legacy diagnostic: DECAY=0 mass, 9580/12000 slope≈0). When decol_rate_relative
                 # is True (default for v3 dynamics), the gates normalise by the pair ITSELF:
                 #   slope_rel = (rate_fast − rate_slow)/(rate_slow+ε)  — % rate change, not
                 #     an absolute Δ. RISING: slope_rel ≥ +margin_rel; FALLING: ≤ −margin_rel.
                 #   dead = rate_fast < γ·rate_peak_pair (leaky per-pair peak, PRE-update) —
                 #     the rate has collapsed to a small fraction of THIS pair's own peak.
                 # margin_rel=0.15 ⇒ a ±15% rate change crosses the band (fast/slow EWMA
                 # gap on coedit gives ~±20-40% on accel/decel bursts → opens easily);
                 # γ=0.20 ⇒ "dead" = rate down to ≤20% of the pair's recent peak. Both are
                 # RELATIVE so they transfer across datasets (per-pair representation law).
                 decol_rate_relative: bool = True,
                 decol_margin_rel: float = 0.15,
                 decol_rate_dead_gamma: float = 0.20,
                 decol_rate_eps: float = 1e-3,
                 # ── HIER v2 gate priors (PM round-7 fix, 2026-06-02) ─────────────────
                 # Round-7 (legacy diagnostic, fsm_decode=hier) DIAGNOSIS from
                 # faithfulness_coedit_v3_hier_let0.5_s42.npz: on the GENUINE recurring set
                 # (true_occ>=2, n=9157) the observed TARGET argmax is REINFORCE 95% /
                 # DEATH 5% — i.e. a pair editing for the 2nd..201st time, on/near its own
                 # peak cadence (rate_fast/peak med=1.0), is REINFORCING. But the hier-v1
                 # PRED was DEATH 93.8% / REINFORCE 0%. TWO structural gate bugs cause it:
                 #   (1) p_alive prior too weak: at rate≈floor & stale≈1 the prior≈0 ⇒
                 #       p_alive≈0.5, but in the tree DEATH=(1−p_alive) is an UNDIVIDED leaf
                 #       (med 0.425) while the alive branch SPLITS into REINFORCE+DECAY (0.25
                 #       /0.18) ⇒ DEATH wins even on active pairs. The "dead" definition was
                 #       PEAK-relative-instantaneous (rate_fast < γ·leaky-peak), which flags
                 #       a pair dead the moment it is not at its tightest-ever burst.
                 #   (2) slope_rel=(fast−slow)/(slow+ε) is DEGENERATE on coedit: fast/slow
                 #       EWMAs both converge to the same small steady rate ⇒ slope_rel med
                 #       0.000, 100% ≤0 on recurring ⇒ p_rising defaults ~0.5 and can NEVER
                 #       signal RISING ⇒ REINFORCE→0 in BOTH target and pred.
                 # decol_hier_v2 (default False ⇒ hier-v1 byte-identical) rebuilds the two
                 # weak priors on the SUSTAINED-silence + staleness-relative axes:
                 #   DEAD = SUSTAINED quiet: dt/μ_pair ≫ 1 (gap many× the pair's own rhythm)
                 #     AND rate collapsed — NOT a one-step rate dip. Recently-edited
                 #     (dt ≤ μ_pair) ⇒ strongly ALIVE. p_alive prior pushed up so genuinely
                 #     active pairs clear the undivided-DEATH leaf.
                 #   RISING = editing FASTER than its own rhythm: stale_rel<1 (dt<μ_pair) ⇒
                 #     REINFORCE; stale_rel>1 (cooling) ⇒ DECAY. Primary signal is the
                 #     staleness ratio (robust), with slope_rel as a secondary nudge.
                 decol_hier_v2: bool = False,
                 # dt/μ_pair threshold above which silence is "sustained" ⇒ dead-leaning.
                 decol_dead_stale_mult: float = 3.0,
                 # hier_causal_policy (default False ⇒ hier byte-identical to current):
                 #   when True, the PUBLISHED interpretable state s_t1_cal (hier tree
                 #   distribution) is post-processed to OBEY causal policy — (1) the
                 #   ever_alive gate (no DEATH-before-alive) and (2) a SOFT expected-
                 #   admissibility mask M[j]=Σ_i s_t[i]·C[i,j] (floor-blended; supersedes
                 #   the old brittle hard C[argmax(s_t),:] row) — then renormalized.
                 #   ONLY touches s_t1_cal (the interpretation quantity);
                 #   s_t1_pos → existence_decoder → AP score is NEVER touched ⇒ AP Δ=0
                 #   EXACT. Both inputs are PRE-update (ever_alive read at L935 via .get
                 #   BEFORE update_batch; s_t_pos from StateObserver, history-only) ⇒ no
                 #   leak. Soft / differentiable (multiplicative gate + renorm) so the CE
                 #   gradient still reaches the hier heads.
                 hier_causal_policy: bool = False,
                 # ── HƯỚNG A: STRICT-ORDERED 6-STATE FSM (PM 2026-06-06) ─────────────
                 # strict_ordered_fsm (default False ⇒ byte-identical to current hier):
                 #   when True (requires hier_causal_policy + fsm_decode="hier"), the
                 #   PUBLISHED interpretable state is decoded as the 6-state lifecycle
                 #   PRE_BIRTH/BIRTH/REINFORCE/DECAY/DORMANT/DEATH (IDLE split into
                 #   PRE_BIRTH vs DORMANT, so "ever-alive" is REPRESENTED IN-STATE,
                 #   Markovian) and constrained by the STRICT band-diagonal C'
                 #   (|i−j|≤1 only — every non-adjacent jump HARD-masked). The separate
                 #   ever_alive gate becomes REDUNDANT (PRE_BIRTH cannot reach DEATH
                 #   without traversing BIRTH→…→DORMANT). This produces a NEW 6-vector
                 #   `s_t1_cal6` (returned for evaluation harness / faithfulness / symbolic), and
                 #   FOLDS BACK to the legacy 5-class `s_t1_cal` (PRE_BIRTH+DORMANT→IDLE)
                 #   so the de-collapse CE target (B,5) and update_symbolic are unchanged.
                 #   s_t1_pos → existence_decoder → AP score is NEVER touched ⇒ AP Δ=0.
                 strict_ordered_fsm: bool = False,
                 # ── OPTION (b): STRICT-ORDERED 5-STATE NATIVE (PM 2026-06-06) ───────
                 # strict_ordered_5state (default False ⇒ byte-identical to current hier):
                 #   when True (requires hier_causal_policy + fsm_decode="hier"), the
                 #   PUBLISHED interpretable state s_t1_cal keeps the NATIVE 5 classes
                 #   IDLE/BIRTH/REINFORCE/DECAY/DEATH and is HARD-masked by the strict
                 #   band-diagonal C_BAND_5 (|i−j|≤1 only — every non-adjacent jump
                 #   zeroed, renormed). Because IDLE(0) and DEATH(4) are the two ENDS of
                 #   the axis, IDLE→DEATH is band-blocked, so death-before-alive needs NO
                 #   ever_alive gate and NO IDLE split — the ordering does it. This branch
                 #   is MUTUALLY EXCLUSIVE with strict_ordered_fsm (the 6-state path) and
                 #   does NOT touch s_t1_pos → existence_decoder → AP score ⇒ AP Δ=0.
                 strict_ordered_5state: bool = False,
                 # ── WC-CONF: WALKED-CHAIN CAUSAL-CONFIDENCE (PM 2026-06-06) ─────────
                 # causal_confidence (default False ⇒ byte-identical to current):
                 #   when True, three NEW quantities are added — prediction is NOT masked
                 #   (the AP path stays FREE, fair vs config B):
                 #   (1) WALKED-CHAIN belief b_t (per-pair, carried in the edge store):
                 #         b_step = normalize( b_{t-1} @ C )   (C-admissible mass only)
                 #         b_t    = normalize( b_step ⊙ obs(s_t) )
                 #       "where this pair SHOULD be by the causal chain it actually walked".
                 #       PRE-update read (no leak); update lands POST-scoring.
                 #   (2) COHERENCE c_t ∈[0,1] = mass of the FREE next-state s_t1_cal on the
                 #       states REACHABLE in one C-step from b_{t-1}:
                 #         reach[j] = 1{ (b_{t-1} @ C)[j] > eps };  c_t = Σ_j s_t1_cal[j]·reach[j].
                 #       Low c = the free prediction lands on a causally-unreachable state
                 #       (e.g. never-born pair carrying b≈IDLE while s_t1 demands DEATH).
                 #       c_t is a CONFIDENCE OUTPUT — it does NOT bend the prediction.
                 #   (3) GRADIENT-SELECTION: the per-event FSM-block loss (edge-trans CE) is
                 #       scaled by c_t.detach() (or zeroed when c<cc_thr) ⇒ the model does
                 #       NOT learn from causally-incoherent trajectories. AP path / backbone
                 #       gradients are UNCHANGED (backbone still 0 link-pred grad, detached).
                 #   ONLY touches s_t1_cal-derived confidence + the FSM-head CE weight; the
                 #   existence_decoder AP score (s_t1_pos) is NEVER touched ⇒ AP Δ=0 EXACT.
                 causal_confidence: bool = False,
                 # cc_C: which causal admissibility matrix the walked-chain uses.
                 #   "band"  → C_BAND_5 (strict |i−j|≤1 band-diagonal; matches the strict-
                 #             ordered work — no nhảy cóc; IDLE↔DEATH at axis ends blocked).
                 #   "rule"  → CAUSAL_RULE_MATRIX (legacy tridiagonal + revival edges).
                 cc_C: str = "band",
                 # cc_thr: HARD coherence floor for gradient-selection. Events with c_t<cc_thr
                 #   get ZERO FSM-block gradient (weight 0); c_t≥cc_thr keep the c_t soft scale.
                 #   0.0 ⇒ pure soft scaling (weight=c_t, no hard cutoff).
                 cc_thr: float = 0.0,
                 # cc_self_consist_w: λ for the WC-CONF belief SELF-CONSISTENCY auxiliary
                 #   loss (default 0 ⇒ exactly the FIX-3 closed-loop filter, no aux loss,
                 #   byte-identical). When >0 AND causal_confidence on, the belief regains a
                 #   LEARNABLE observation-coupling step: w_obs = sigmoid(cc_w_obs_logit) is a
                 #   trainable scalar param that blends the learned-forward belief b_step with
                 #   the current observation obs=s_t_pos. w_obs is trained by an auxiliary CE
                 #   (belief ‖ free next-state argmax, detached target) computed OUTSIDE the
                 #   no_grad belief block so the param is NOT dead. The CE depends ONLY on
                 #   w_obs + DETACHED tensors ⇒ NO gradient reaches predict/backbone (predict
                 #   Δ=0 preserved). This restores the filter's MEASUREMENT step that FIX-3
                 #   dropped (legacy diagnostic: belief stuck IDLE 0.87) without a hand constant.
                 cc_self_consist_w: float = 0.0,
                 # cc_grounded_init: GROUND the walked-chain belief INIT at the pair's REAL
                 #   inferred phase, NOT IDLE (PM 2026-06-07 — fix the WC-CONF *structural
                 #   ceiling*: 4 prior rounds stuck IDLE/BIRTH because the belief RESET to
                 #   IDLE one-hot for any pair making its FIRST appearance IN A SPLIT — incl.
                 #   already-MATURE pairs entering test. With IDLE init + a band-step walk a
                 #   mature pair could never climb to its true phase ⇒ belief ≈ IDLE 0.98,
                 #   R²(c_t~free) ≈ 0.98, c_t degenerate. legacy diagnostic: belief IDLE 0.984.)
                 #   Default OFF ⇒ peek_belief keeps IDLE init, byte-identical to FIX-3/R2.
                 #   When ON (AND causal_confidence): the FIRST time a pair is peeked (no entry
                 #   in the belief store), init the belief to the MODEL-INFERRED state at that
                 #   event = softmax(s_t_pos) (the StateObserver readout, history-only, PRE-
                 #   update, detached — the SAME quantity the score path reads ⇒ NO leak, NO
                 #   hand-set phase). From that grounded init the belief then walks causally
                 #   (learned T_uv + C-ray projection + obs-coupling) exactly as before; only
                 #   the SEED of the chain changes. A genuinely-new pair (no history) yields a
                 #   StateObserver readout near BIRTH/IDLE ⇒ grounded init is still honest pre-
                 #   birth for them; a mature pair entering test reads its real phase.
                 cc_grounded_init: bool = False,
                 # decol_class_balance: inverse-frequency per-event weight on the edge-
                 #   trans KL so the head DARES to commit DECAY/DEATH when the target says
                 #   so (counters the intermediate-DECAY commitment failure). PM item 2a.
                 decol_class_balance: bool = False,
                 # decol_argmax_bias: learnable per-class additive log-bias applied ONLY
                 #   to the CE-supervised / argmax-measured distribution — NOT to the
                 #   existence_decoder logits that score AP — so DECAY/DEATH can win argmax
                 #   without perturbing the link-prediction readout. PM item 2b.
                 decol_argmax_bias: bool = False,
                 # ── Symbolic-FSM architecture selector (de-collapse REDESIGN) ──
                 # "v1" (DEFAULT, byte-identical canonical): the gate + de-collapse CE
                 #   operate on the ECTGv3 VALID-HARD-MASKED continuous chain
                 #   (edge_mem._state_table[:,:5]). STRUCTURALLY PINNED at BIRTH≈0.84,
                 #   invariant to supervision weight (DIAGNOSED 2026-05-31): the forward
                 #   chain advances the DETACHED argmax at most one rung/event and the
                 #   -1e9 mask gives ZERO gradient toward any one-step-unreachable state.
                 # "v2": the gate + de-collapse CE operate on the SEPARATE soft-masked
                 #   FSM head s_{t+1} (state_observer→transition_predictor→soft
                 #   LifecycleFSMMask, the sigmoid(prior+delta) FINITE penalty). The
                 #   symbolic state is DECOUPLED from the ECTG continuous accumulator,
                 #   has a finite-penalty soft prior (gradient crosses it → MOVABLE by
                 #   supervision, CPU-proven), is persisted via edge_mem.update_symbolic
                 #   for the gate, and still yields a per-event compliance ∈[0,1] for LFG.
                 fsm_arch: str = "v1",
                 # ── Symbolic-state READOUT decode (de-collapse STRUCTURAL fix) ──
                 # "flat" (DEFAULT, byte-identical): the state distribution that the gate
                 #   measures / the analyzer argmaxes for faithfulness IS the flat 5-class
                 #   softmax s_t1_pos (optionally argmax-bias-calibrated). DIAGNOSED dead-
                 #   end (legacy diagnostic, 5 GPU rounds): DECAY is the MIDDLE class on the
                 #   sequential axis BIRTH→REINFORCE→DECAY→DEATH; in a flat softmax its
                 #   mass is split between its two neighbours (REINFORCE & DEATH) so it can
                 #   NEVER win the argmax even when the prob is faithful (ρ=0.887, job
                 #   5452105 — the model KNOWS the state, the flat readout cannot express
                 #   it). NO amount of target/threshold tuning fixes this — it is the
                 #   OUTPUT STRUCTURE.
                 # "hier": HIERARCHICAL decode (PM 2026-06-01). Factor the 5-state dist as
                 #   a decision TREE matching the user's mental model, from per-pair
                 #   PRE-update signals (no leak):
                 #     gate p_birth   : new pair (n_prior low / true first occurrence).
                 #     gate p_alive   : SUSTAINED activity (rate-stable / recently edited)
                 #                      vs DEAD (rate≈0 KÉO DÀI). NOT peak-relative instant.
                 #     gate p_rising  : rate slope ≥0 (REINFORCE) vs <0 (DECAY), ALIVE branch.
                 #   P(BIRTH)=p_birth; P(REINFORCE)=(1−p_birth)·p_alive·p_rising;
                 #   P(DECAY)=(1−p_birth)·p_alive·(1−p_rising); P(DEATH)=(1−p_birth)·(1−p_alive).
                 #   ⇒ DECAY competes ONLY with REINFORCE inside the alive branch, NEVER
                 #   directly with DEATH, so argmax-DECAY becomes reachable. ONLY the STATE
                 #   readout (s_t1_cal: gate + faithfulness + de-collapse CE) is rerouted;
                 #   the existence_decoder AP score (s_t1_pos) is UNTOUCHED (Δ=0). Each gate
                 #   is an analytic soft prior over the PRE-update signals + a tiny learnable
                 #   residual head so the de-collapse KL trains them (gradient flows).
                 fsm_decode: str = "flat",
                 # ── CAUSAL INTRA-BATCH ACCUMULATION (P1 bug fix, 2026-06-03) ─────────
                 # default False ⇒ byte-identical to the legacy (buggy) batched store.
                 # When True, the EdgeStateStoreV3 continuous channels (Welford μ/var/n,
                 # Hawkes λ, recurrence EWMA) AND the dict helpers (rate fast/slow EWMA,
                 # rate-peak, λ-peak) are accumulated EVENT-BY-EVENT IN STREAM ORDER
                 # within each batch, so a pair firing K× in one batch reads the state
                 # AFTER its (k−1)-th in-batch event — not the same pre-batch snapshot K×.
                 # ROOT CAUSE (DIAGNOSED 2026-05-30, confirmed npz dumps): the legacy
                 # get_batch snapshots once/batch ⇒ Welford n caps at #batches-the-pair-
                 # appears-in (≈6 on coedit batch=500) even for pairs editing 201×, and
                 # rate_fast/peak pin at RATE_INIT ⇒ slope_rel≈0 / μ_pair degenerate ⇒
                 # the staleness & slope counterfactual axes were dead. true_occ was the
                 # only batch-immune signal. With causal_batch=True ALL the continuous
                 # stats match an event-by-event (batch_size=1) reference (CPU-verified).
                 # NB: only the deterministic stat channels are causalised — the learned
                 # state-logit chain [0:5] still uses the model's vectorised forward, but
                 # those are not the corrupted quantities. Scoring stays PRE-update
                 # (no re-leak): the per-pair φ / gate / target read the CAUSAL pre-event
                 # state of THIS event (post all earlier same-pair events, pre this one).
                 causal_batch: bool = False,
                 # ── SINGLE-VARIABLE DETACH PROBE (PM 2026-06-08, reviewer #1) ───────
                 # edge_h_detach_scorepath (default True = CANONICAL / config B
                 # byte-identical): controls the ONE bit the panel asked to isolate —
                 # whether the LINK-PREDICTION score path (s_t1_pos→existence_decoder
                 # →pos_logit, and the symmetric neg path) carries gradient back into
                 # the backbone (edge_h → DRGC/ECTG/CSN) or is .detach()ed.
                 #   True  (OFF the gradient): the canonical "regularization-by-
                 #         decoupling" arm. state_observer detaches h internally AND the
                 #         transition_predictor scoring calls pass edge_h.detach() ⇒
                 #         backbone gets ZERO link-pred gradient (only λ_echo·KL trains
                 #         it). This == every prior config-B run, BYTE-IDENTICAL.
                 #   False (ON the gradient): remove BOTH detaches ONLY on the pos/neg
                 #         SCORING path so the link-pred BCE flows into the backbone.
                 #         enable_main_predictor STAYS False (head capacity unchanged);
                 #         every other knob (lfg_mode, compliance_floor, lambda_edge_
                 #         trans, entropy/kl weights, fsm_arch/decode, init) is IDENTICAL.
                 # This is the clean A/B the B-vs-C arm conflated. NOTE: the FSM-stream
                 # / de-collapse-CE / faithfulness branches still read edge_h.detach()
                 # in BOTH settings (they are interpretation-only by design); only the
                 # AP-scoring s_t1_pos/s_t1_neg transition_predictor calls + the
                 # state_observer used for scoring are un-detached when this is False.
                 edge_h_detach_scorepath: bool = True,
                 # ── IDENTICAL-HEAD K1 DETACH PROBE (PM 2026-06-10, reviewer §8.2.i) ──
                 # main_predictor_detach (default False = canonical K1 / end-to-end):
                 # controls ONLY whether the backbone edge_h fed to the SHARED 2-layer
                 # MLP scoring head (self.main_predictor) is .detach()ed. It is meaningful
                 # ONLY when enable_main_predictor=True, i.e. when main_predictor IS the
                 # AP-scored head. Holding enable_main_predictor=True in BOTH arms means
                 # the SAME nn.Sequential head module (created identically → identical
                 # init per seed) scores both arms; the SOLE difference is this one
                 # .detach() on the backbone→head input:
                 #   False  COUPLED-MLP : main_predictor(edge_h)         → link grad flows
                 #          into DRGC/ECTG/CSN (this == the prior K1 arm, byte-identical).
                 #   True   DETACHED-MLP: main_predictor(edge_h.detach()) → backbone gets
                 #          ZERO link-pred gradient, but the head architecture/capacity/
                 #          init are IDENTICAL to COUPLED-MLP. Isolates gradient-flow from
                 #          head-architecture (eliminates the "a coupled MLP merely
                 #          destroys hand-crafted features" confound of the B-vs-K1 arm,
                 #          where B also swaps the head to the FSM existence decoder).
                 # NB: independent of edge_h_detach_scorepath, which gates the SEPARATE
                 # existence-decoder/transition_predictor FSM-score path; that path is
                 # not the AP-scored head when enable_main_predictor=True.
                 main_predictor_detach: bool = False,
                 # ── DETERMINISTIC-ONLY BACKBONE (PM 2026-06-11, reviewer Q4/caveat-2) ──
                 # determ_only_backbone (default False = canonical config B):
                 # removes the LEARNABLE backbone (the KL/parsimony-shaped ResidualCSN
                 # event encoder AND the DRGC_v2 coupled-GRU node-memory) from the
                 # SCORE path, so the AP-scored detached head (existence_decoder, the
                 # config-B head) is driven ONLY by the DETERMINISTIC point-process
                 # channels (Hawkes λ, Welford μ/var/n, recurrence/rate EWMA, staleness,
                 # ever_alive — i.e. pair_phi from edge_st + the lifecycle mask/gate).
                 # Mechanism when True (single axis vs FULL-B, everything else held):
                 #   (1) CSN bypassed: feat_g = feat (raw), salience sal = 0 → no learned
                 #       event-encoder transform.
                 #   (2) DRGC bypassed for the SCORE: new_h_src/new_h_dst pass through the
                 #       raw node memory unchanged (no coupled-GRU update), AND the edge_h
                 #       fed to the FSM scoring head (state_observer / transition_predictor
                 #       for BOTH pos and neg) is ZEROED → the scored FSM state depends on
                 #       NO learnable-backbone content, only pair_phi (deterministic).
                 #   (3) csn / ectg / drgc parameters are frozen at init (requires_grad
                 #       False) so DETERM-ONLY has ZERO trainable backbone params; the KL
                 #       (lambda_echo*kl) term therefore trains nothing here either.
                 # HONEST CAVEAT (documented): the detached FSM head modules
                 # (state_observer, transition_predictor, lifecycle_mask, existence_
                 # decoder, hier gate heads) remain trainable — they ARE the scoring
                 # head, NOT the "learnable backbone" being removed; this is the SAME
                 # detached head as config B, so the axis toggled is purely
                 # learnable-CSN+DRGC-vs-deterministic-only. The head still requires a
                 # 2H edge_h tensor by signature, so it is fed a constant zero (carries
                 # no backbone info / no gradient) rather than being deleted.
                 determ_only_backbone: bool = False,
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        # ── Composite preset resolution (design / causal_mask) ──
        # `causal_mask=True` is an alias for design="correct". When the "correct"
        # preset is requested, turn ON the full intended TWO-STREAM CAUSAL-GRADIENT-
        # MASK stack — but ONLY for flags the caller left at their canonical default,
        # so explicit kwargs still win and design="canonical" (the no-arg path) is a
        # strict no-op (byte-identical canonical model).
        if causal_mask:
            design = "correct"
        if design == "correct":
            if enable_main_predictor is False:   # Stream 1: trainable predictor
                enable_main_predictor = True
            if lfg_mode == "soft":               # Stream 2: hard causal gradient gate
                lfg_mode = "hard"
            if compliance_floor == 0.05:         # hard gate ⇒ zero floor (full mask)
                compliance_floor = 0.0
            if entropy_reg_weight == 0.0:         # de-collapse: entropy reg
                entropy_reg_weight = 0.01
            if fix_existence_init is False:       # spec-correct existence init
                fix_existence_init = True
            # lambda_trans is revived unconditionally under the preset (it is DEAD by
            # default — never in the loss — so reviving it only adds the intended
            # transition-CE de-collapse term; keeps its passed value as the weight).
        elif design == "correct_decoupled":
            # ── DE-COLLAPSE, DECOUPLED FROM THE HEAD (Tier-1 re-gate) ──
            # Root cause of the Tier-1 FAIL (diagnosed 2026-05-30): (1) the gate metric
            # measures the ECTGv3 CONTINUOUS edge-state argmax, not the FSM s_{t+1};
            # (2) that continuous chain has DEATH absorbing + a dead `is_first` (recur
            # jumps 0→0.3 on event 1 so BIRTH never fires) → two low-entropy attractors
            # (IDLE pin / DEATH sink); (3) the old trans-CE supervised the FSM stream,
            # NOT the quantity the gate measures, AND its heuristic target is itself
            # collapsed. This preset attacks the MEASURED distribution directly while
            # keeping the BETTER-AP DETACHED head (enable_main_predictor stays False):
            #   - fsm_decouple: keep detached head (do NOT flip enable_main_predictor)
            #   - decollapse_target: rebuild the transition target with a working
            #     n_obs-based is_first + class balance (de-degenerated supervision)
            #   - lambda_edge_trans: CE on ECTGv3 student_logits (the gate's own net)
            #   - edge_state_entropy_w: per-event entropy floor on the continuous dist
            #   - edge_uniform_kl_w: KL(new_dist‖uniform) floor to drain the DEATH sink
            #   - keep entropy_reg + fix_existence_init for the FSM stream too
            if not fsm_decouple:
                fsm_decouple = True
            if not decollapse_target:
                decollapse_target = True
            # WEIGHT BALANCE (tuned from CPU micro-train, a temporary diagnostic): with a strong
            # edge-trans CE the corrected target's BIRTH spike creates a NEW pure-BIRTH
            # attractor (H still 0) — the target moved IDLE→BIRTH but over-corrected. So
            # the spread floors (entropy + uniform-KL) must DOMINATE the CE: the CE only
            # needs to break the IDLE/DEATH pin; the floors do the de-collapse work.
            #
            # REBALANCED 2026-05-30 (root cause FIXED at the SOURCE): the BIRTH pile-up
            # was the TARGET, not the weights — the old target was 84.7% BIRTH because
            # is_first used the collision-corrupted Welford n_obs. With the collision-
            # immune true_occ is_first + recalibrated chain-aware lifecycle logits, the
            # target's OWN argmax is now BIRTH .185 / REINFORCE .585 / DECAY .201 /
            # DEATH .029, H=1.051 (CPU-measured, coedit). The target now CARRIES the spread, so
            # we INVERT the old balance: the CE leads (lets the model match a meaningful
            # target) and the entropy/uniform floors drop to gentle anti-degeneracy
            # insurance — a strong floor would now push the model AWAY from the
            # meaningful lifecycle toward ARTIFICIAL uniformity (the failure mode the PM
            # flagged: high H, meaningless states, uninformative causal mask).
            if lambda_edge_trans == 0.0:
                lambda_edge_trans = 0.10      # CE LEADS (was 0.03)
            if edge_state_entropy_w == 0.0:
                edge_state_entropy_w = 0.02   # gentle floor (was 0.20)
            if edge_uniform_kl_w == 0.0:
                edge_uniform_kl_w = 0.01      # gentle floor (was 0.10)
            if entropy_reg_weight == 0.0:
                entropy_reg_weight = 0.01
            if fix_existence_init is False:
                fix_existence_init = True
            # enable_main_predictor INTENTIONALLY left at its default (False = detached,
            # better-AP arm). lfg_mode stays "soft" (no hard gate — the hard gate needs
            # the e2e head to matter). This isolates "can we make the FSM/edge-state
            # healthy" from "which head predicts" — the exact Tier-1 confound.
        elif design != "canonical":
            raise ValueError(
                f"design must be 'canonical', 'correct', or 'correct_decoupled', "
                f"got {design!r}")
        # Under the "correct"/"correct_decoupled" presets the FSM transition-CE term is
        # active; in every other config lambda_trans stays DEAD (back-compat).
        self.use_trans_loss = design in ("correct", "correct_decoupled")
        self.design  = design
        if fsm_arch not in ("v1", "v2", "v3"):
            raise ValueError(f"fsm_arch must be 'v1', 'v2', or 'v3', got {fsm_arch!r}")
        self.fsm_arch = fsm_arch
        if fsm_decode not in ("flat", "hier"):
            raise ValueError(f"fsm_decode must be 'flat' or 'hier', got {fsm_decode!r}")
        # Hierarchical decode is ONLY meaningful for fsm_arch="v3" (it consumes the v3
        # PRE-update rate/slope/per-pair signals). For v1/v2 it is silently a no-op so the
        # flag never perturbs the canonical/v2 byte-identical paths.
        self.fsm_decode = fsm_decode if fsm_arch == "v3" else "flat"
        # fsm_arch="v3" REDESIGN: per-pair flip dynamics. Builds the per-pair operator
        # g(φ_uv) inside transition_predictor (φ from EdgeStateStoreV3 continuous stats)
        # and supervises the soft FSM head with a SELF-SUPERVISED, per-pair, observed
        # lifecycle flip target (no entropy hammer; H is diagnostic only). v3 reuses
        # the v2 soft-head path (finite-penalty mask = soft structural prior, part A)
        # so it is automatically de-pinned; the per-pair operator+target (parts B/C)
        # supply the HETEROGENEITY (each pair flips per its own past).
        # φ_uv dim = 7 for v3: the 6 PRE-update history channels (Hawkes λ, log mean_dt,
        # log var_dt, recurrence EWMA, log staleness, ever_alive) + the PER-PAIR-RELATIVE
        # z = (Δt − μ_pair^pre)/σ_pair^pre — the single most discriminative "is this gap
        # long FOR THIS PAIR" signal. Feeding z into g(φ) lets the per-pair transition
        # operator bias the flip dynamics by each pair's OWN deviation, so heterogeneity
        # is driven directly (not only via the supervised target). 0 for v1/v2 (no pair_g).
        self._pair_phi_dim = 7 if fsm_arch == "v3" else 0
        # ── fsm_arch="v3" SELF-CONTAINED de-collapse enablement (no entropy hammer) ──
        # Make `--fsm_arch v3` a complete, self-sufficient recipe (independent of the
        # design preset): turn ON the OBSERVED per-pair lifecycle target + the soft-head
        # transition-CE, KEEP the detached (better-AP) head, and DELIBERATELY LEAVE the
        # entropy/uniform-KL floors at ZERO — PM mandate: H is a DIAGNOSTIC, not a loss;
        # entropy hammer + raw lambda_edge_trans sweeps are the DEAD levers that produced
        # the v2 over-correction (REINFORCE≈0.97 / H≈0.09). De-collapse here comes from
        # heterogeneous per-pair targets + the per-pair operator g(φ), NOT entropy fiat.
        # Explicit kwargs still win (the `== default` guards), so a sweep can override.
        if fsm_arch == "v3":
            # NOTE: do NOT force use_trans_loss — the FSM-stream trans_loss_term and the
            # edge_trans_term would both KL s_t1_pos→decol_target (double-drive). v3 uses
            # the edge_trans_term channel only (it routes to s_t1_pos for v2/v3 below).
            if not fsm_decouple:
                fsm_decouple = True
            if not decollapse_target:
                decollapse_target = True
            if lambda_edge_trans == 0.0:
                lambda_edge_trans = 0.10   # CE on the soft head (the gate's quantity)
            # entropy/uniform-KL floors STAY 0.0 (PM: no entropy hammer)
            if fix_existence_init is False:
                fix_existence_init = True
            # ── PM-directed (2026-06-01) v3 defaults: FREQUENCY + DYNAMICS target +
            #    argmax calibration. All ON for v3 unless the caller explicitly set them
            #    False (the `is False` guards make explicit kwargs win for a sweep). The
            #    diagnosis (CPU a temporary calibration diagnostic): DECAY soft-mass is healthy but argmax
            #    never commits to the INTERMEDIATE DECAY state (REINFORCE ties+wins), so
            #    the fix is class-balance + a per-class argmax bias, NOT another target
            #    reshape. Dynamics sharpens DECAY/DEATH where frequency collapses.
            if decol_use_dynamics is False:
                decol_use_dynamics = True
            if decol_class_balance is False:
                decol_class_balance = True
            if decol_argmax_bias is False:
                decol_argmax_bias = True
        # De-collapse levers (stored; default 0/False = canonical no-op)
        self.fsm_decouple         = fsm_decouple
        self.decollapse_target    = decollapse_target
        self.lambda_edge_trans    = lambda_edge_trans
        self.edge_state_entropy_w = edge_state_entropy_w
        self.edge_uniform_kl_w    = edge_uniform_kl_w
        self.decol_hawkes_thr     = decol_hawkes_thr
        self.decol_late_thr       = decol_late_thr
        self.decol_dead_thr       = decol_dead_thr
        self.decol_use_dynamics   = decol_use_dynamics
        self.decol_mom_thr        = decol_mom_thr
        self.decol_decline_thr    = decol_decline_thr
        self.decol_silence_margin = decol_silence_margin
        self.decol_silence_scale  = decol_silence_scale
        self.decol_slope_margin   = decol_slope_margin
        self.decol_slope_scale    = decol_slope_scale
        self.decol_rate_dead      = decol_rate_dead
        self.decol_rate_dead_scale= decol_rate_dead_scale
        self.decol_rate_relative  = decol_rate_relative
        self.decol_margin_rel     = decol_margin_rel
        self.decol_rate_dead_gamma= decol_rate_dead_gamma
        self.decol_rate_eps       = decol_rate_eps
        self.decol_class_balance  = decol_class_balance
        self.decol_argmax_bias    = decol_argmax_bias
        self.decol_hier_v2        = decol_hier_v2
        self.decol_dead_stale_mult= decol_dead_stale_mult
        self.hier_causal_policy   = hier_causal_policy
        # Causal-policy on s_t1_cal needs the HARD admissibility matrix C even in the
        # canonical soft path (lfg_mode != "hard" does NOT register `causal_rule`). Keep
        # a private, non-state_dict tensor so enabling the flag never changes state_dict
        # keys (canonical / v1 / v2 / flat stay byte-identical). Lazily moved to device
        # at forward time. Only materialized when the flag is on.
        if hier_causal_policy:
            self._hier_causal_C = CAUSAL_RULE_MATRIX.clone()
        else:
            self._hier_causal_C = None

        # ── HƯỚNG A: strict-ordered 6-state FSM (PM 2026-06-06) ─────────────────
        # strict_ordered_fsm needs hier_causal_policy + hier decode to be meaningful
        # (it re-decodes the hier tree output into the 6-state lifecycle). Guard so the
        # flag is a strict no-op unless its prerequisites are on (keeps every other
        # config — canonical / v1 / v2 / flat / hier-without-policy — byte-identical).
        self.strict_ordered_fsm = bool(
            strict_ordered_fsm and hier_causal_policy and fsm_decode == "hier")
        if self.strict_ordered_fsm:
            # private non-state_dict tensor (lazily moved to device at forward time) so
            # enabling the flag never adds state_dict keys.
            self._strict_C = STRICT_ORDERED_C.clone()
        else:
            self._strict_C = None

        # ── OPTION (b): strict-ordered 5-STATE native (PM 2026-06-06) ──────────
        # Same prerequisites as the 6-state path; mutually exclusive with it (if both
        # flags were set, the 6-state path takes precedence and this is forced off so
        # the band mask is not applied twice). Private non-state_dict tensor ⇒ no new
        # state_dict keys ⇒ canonical / config B / every other config byte-identical.
        self.strict_ordered_5state = bool(
            strict_ordered_5state and hier_causal_policy
            and fsm_decode == "hier" and not self.strict_ordered_fsm)
        if self.strict_ordered_5state:
            self._band5_C = C_BAND_5.clone()
        else:
            self._band5_C = None

        # ── WC-CONF: walked-chain causal-confidence (PM 2026-06-06) ─────────────
        # Default OFF ⇒ no belief store touched, no CE reweight ⇒ byte-identical to
        # config B / canonical. The C matrix is a private non-state_dict tensor (lazily
        # moved to device at forward) so enabling the flag never adds state_dict keys.
        # Requires the hier readout (s_t1_cal) to be meaningful as the FREE next-state;
        # guarded so the flag is a strict no-op without fsm_decode="hier".
        self.causal_confidence = bool(causal_confidence and fsm_decode == "hier")
        self.cc_thr = float(cc_thr)
        self.cc_self_consist_w = float(cc_self_consist_w)
        # GROUNDED belief init (PM 2026-06-07): seed the walked-chain at the model-
        # inferred phase (softmax s_t_pos) instead of IDLE for a pair's FIRST peek in
        # the split. Pure init-source swap (no new param, no state_dict key). Gated to
        # causal_confidence so it is a strict no-op everywhere else.
        self.cc_grounded_init = bool(cc_grounded_init and self.causal_confidence)
        # ── WC-CONF learnable observation-weight (restores the dropped measurement step).
        # w_obs = sigmoid(cc_w_obs_logit) blends learned-forward belief with the current
        # observation. Created ONLY when causal_confidence AND cc_self_consist_w>0 so a
        # plain config-B / canonical / cc-off-or-aux-off state_dict gains ZERO keys (the
        # param is the ONLY new state). Init logit=0 ⇒ w_obs=0.5 at start; the
        # self-consistency CE then moves it. nn.Parameter so it appears in .parameters()
        # and the optimizer trains it; gradient reaches it ONLY through the aux CE term.
        if self.causal_confidence and self.cc_self_consist_w > 0.0:
            self.cc_w_obs_logit = nn.Parameter(torch.zeros(()))
        else:
            self.cc_w_obs_logit = None
        if self.causal_confidence:
            if cc_C == "band":
                self._cc_C = C_BAND_5.clone()
            elif cc_C == "rule":
                self._cc_C = CAUSAL_RULE_MATRIX.clone()
            else:
                raise ValueError(f"cc_C must be 'band' or 'rule', got {cc_C!r}")
            self._cc_C_name = cc_C
        else:
            self._cc_C = None
            self._cc_C_name = None

        if lfg_mode not in ("soft", "hard"):
            raise ValueError(f"lfg_mode must be 'soft' or 'hard', got {lfg_mode!r}")
        self.lfg_mode = lfg_mode

        self.num_nodes  = num_nodes
        self.feat_dim   = feat_dim
        self.hidden     = hidden
        self.device     = device
        self._feat_in   = max(feat_dim, 1)

        # HARD causal-rule admissibility matrix C ∈ {0,1}^{5×5}. Registered as a buffer
        # (so it moves with .to(device) and is excluded from gradient) ONLY when the hard
        # gate is active (lfg_mode="hard"). In the canonical soft path it is NOT
        # registered → the canonical state_dict has the exact same keys/params as before
        # (byte-identical no-arg model). Used ONLY to derive the detached gradient gate;
        # it never enters the prediction value.
        if lfg_mode == "hard":
            self.register_buffer("causal_rule", CAUSAL_RULE_MATRIX.clone())
        else:
            self.causal_rule = None

        # Hyperparams
        self.lambda_echo      = lambda_echo
        self.lambda_trans     = lambda_trans
        self.lambda_violation = lambda_violation
        self.lambda_fsm       = lambda_fsm
        self.lfg_warmup_epochs = lfg_warmup_epochs
        self.compliance_floor  = compliance_floor
        self.enable_lfg          = enable_lfg
        self.fix_existence_init  = fix_existence_init
        self.entropy_reg_weight  = entropy_reg_weight
        self.enable_echo         = enable_echo
        self.echo_tau            = echo_tau
        self.enable_main_predictor = enable_main_predictor
        # Single-variable detach probe (PM 2026-06-08). True = canonical config-B
        # (link-pred score path detached from backbone). False = let pos/neg scoring
        # path train the backbone end-to-end (the ONLY bit that differs in the A/B).
        self.edge_h_detach_scorepath = bool(edge_h_detach_scorepath)
        # IDENTICAL-HEAD K1 detach probe (reviewer §8.2.i): toggles ONLY the .detach()
        # on the backbone→shared-MLP-head input. Head module is unchanged (same
        # self.main_predictor created identically), so DETACHED-MLP vs COUPLED-MLP
        # differ by this single bit. Meaningful only with enable_main_predictor=True.
        self.main_predictor_detach = bool(main_predictor_detach)
        # DETERMINISTIC-ONLY BACKBONE (reviewer Q4): see ctor docstring. Bypasses
        # learnable CSN+DRGC on the score path so the detached head reads only the
        # deterministic point-process channels (pair_phi from edge_st).
        self.determ_only_backbone = bool(determ_only_backbone)

        # Backbone (v3.1 continuous laws)
        self.csn  = ResidualCSN(self._feat_in, hidden)
        self.ectg = ECTGv3(self._feat_in, hidden)  # produces edge_ctx, hawkes_lam etc
        self.drgc = DRGC_v2(self._feat_in, hidden, tip_beta)

        # Main Edge Predictor (NON-detached) — carries link-prediction gradient
        # back into CSN / ECTG / DRGC. This is the branch whose logits are scored
        # for AP/AUC and whose BCE (+ LFG) trains the backbone.
        self.main_predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # FSM stream (all stop-grad) — interpretation/symbolic only, no backbone grad.
        self.state_observer       = StateObserver(hidden * 2, n_states=5)
        self.transition_predictor = TransitionPredictor(
            hidden * 2, n_states=5, rank=3, pair_phi_dim=self._pair_phi_dim)
        self.lifecycle_mask       = LifecycleFSMMask(n_states=5)
        self.existence_decoder    = ExistenceDecoder(
            n_states=5, fix_existence_init=fix_existence_init)

        # ── Per-class ARGMAX-calibration log-bias (decol_argmax_bias, fsm_arch="v3") ──
        # A single learnable 5-vector added (in log-space) to the CE-supervised /
        # argmax-measured next-state distribution ONLY. It lets the head COMMIT to the
        # intermediate DECAY (and DEATH) state when their soft mass is high but ties with
        # REINFORCE at the raw argmax (the diagnosed commitment failure). It is NOT added
        # to the existence_decoder logits that score AP, so the link-prediction readout —
        # hence AP/AUC — is mathematically untouched by this parameter. Created ONLY when
        # the flag is on ⇒ v1/v2/canonical and plain-v3 state_dicts gain zero keys when it
        # is off. Init zero ⇒ a fresh model starts == no-bias; supervision moves it. Only
        # the DECAY/DEATH entries are left trainable-relevant; IDLE/BIRTH/REINFORCE are
        # held at 0 via a registered mask so the bias cannot trivially inflate REINFORCE.
        if self.fsm_arch == "v3" and decol_argmax_bias:
            self.argmax_bias = nn.Parameter(torch.zeros(5))
            self.register_buffer(
                "_argmax_bias_mask",
                torch.tensor([0., 0., 0., 1., 1.]))  # only DECAY,DEATH get a learned bias
        else:
            self.argmax_bias = None

        # ── HIERARCHICAL DECODE gate heads (fsm_decode="hier", fsm_arch="v3" only) ──
        # Three tiny LEARNABLE residual heads, one per binary gate (birth / alive /
        # rising). Each takes a small per-pair PRE-update feature vector φ_g (built in
        # forward, no leak) and emits a scalar logit ADDED to an ANALYTIC prior logit (so
        # zero-trained the gate == the analytic soft function of the signals; the de-
        # collapse KL then refines it — gradient flows into these heads). Param count:
        # 3 × [Linear(GATE_FEAT_DIM=4, 8) + Linear(8,1)] = 3×((4·8+8)+(8·1+1)) = 3×49
        # = 147 params. Created ONLY for hier ⇒ flat / v1 / v2 state_dicts gain ZERO keys.
        # The gate features (per gate): see forward _hier block.
        if self.fsm_decode == "hier":
            GATE_FEAT_DIM = 4
            def _gate_head():
                h = nn.Sequential(nn.Linear(GATE_FEAT_DIM, 8), nn.Tanh(),
                                  nn.Linear(8, 1))
                nn.init.zeros_(h[-1].weight)   # zero-init OUTPUT ⇒ residual starts at 0
                nn.init.zeros_(h[-1].bias)     # ⇒ gate == analytic prior at init
                return h
            self.hier_birth_head  = _gate_head()
            self.hier_alive_head  = _gate_head()
            self.hier_rising_head = _gate_head()
        else:
            self.hier_birth_head  = None
            self.hier_alive_head  = None
            self.hier_rising_head = None

        # Echo Memory (improvement #3) — only instantiated when enabled so the
        # canonical (enable_echo=False) module list / param set is byte-identical
        # to before. Faithful port of experiments/models/sr_gnn_v3.py EchoMemory:
        # single-scale, learnable echo_gate (init 0.1) + echo_norm LayerNorm,
        # @no_grad echo content injected into h before DRGC.
        if enable_echo:
            self.echo      = EchoMemory(num_nodes, hidden, tau=echo_tau,
                                        lambda_echo=lambda_echo, device=device)
            self.echo_gate = nn.Parameter(torch.tensor(0.1))
            self.echo_norm = nn.LayerNorm(hidden)
        else:
            self.echo = None

        # Memories
        self.causal_batch = causal_batch
        self.node_mem     = NodeMemoryStore(num_nodes, hidden, device)
        self.edge_mem     = EdgeStateStoreV3(num_nodes, hidden, device,
                                             causal_batch=causal_batch)
        self.ever_alive   = EverAliveStore(num_nodes, device)

        # Training state
        self.current_epoch = 0

        # ── DETERM-ONLY: freeze the learnable backbone at init (reviewer Q4) ──
        # Set requires_grad=False on every csn / ectg / drgc parameter so the
        # DETERM-ONLY arm has ZERO trainable backbone params. These modules are also
        # bypassed on the score path in forward (feat_g=feat, sal=0, edge_h zeroed to
        # the FSM head), so they contribute neither learned signal nor gradient to the
        # AP-scored logit. Frozen-at-init (never receiving any link or KL gradient)
        # makes "deterministic-only" exact for the parts the experiment targets.
        # The detached FSM head (state_observer/transition_predictor/lifecycle_mask/
        # existence_decoder/hier heads) stays trainable — it is the config-B scoring
        # head, not the backbone being ablated.
        self._n_backbone_params_frozen = 0
        if self.determ_only_backbone:
            for _mod in (self.csn, self.ectg, self.drgc):
                for _p in _mod.parameters():
                    if _p.requires_grad:
                        self._n_backbone_params_frozen += 1
                    _p.requires_grad_(False)

    def set_epoch(self, ep: int):
        self.current_epoch = ep

    def reset(self):
        self.node_mem.reset()
        self.edge_mem.reset()
        self.ever_alive.reset()
        if self.echo is not None:
            self.echo.reset()

    def _build_edge_h(self, h_src: Tensor, h_dst: Tensor) -> Tensor:
        return torch.cat([h_src, h_dst], dim=-1)

    # ── FREEZE-THEN-PROBE control (reviewer #1, decoupling-vs-linear-probe) ────────
    # The classic "freeze-then-probe / linear-probing transfer" control (Alain&Bengio
    # 2017; Kumar et al. 2022) for the decoupling-by-construction claim. Protocol:
    #   (1) pretrain WITH the end-to-end link-pred head (enable_main_predictor=True ⇒
    #       backbone csn/ectg/drgc shaped by link-pred BCE = config "K1"/correct e2e),
    #   (2) freeze_backbone()  → csn/ectg/drgc (+ FSM stream + echo gate, all the shaped
    #       representation modules) requires_grad=False; the stateful stores are not
    #       gradient-trained anyway and keep updating in forward as usual,
    #   (3) reinit_main_predictor() → throw away the co-trained head, train a FRESH
    #       link head on the FROZEN features,
    #   (4) measure inductive AP.
    # If decoupling-by-construction (backbone NEVER saw link-pred grad) > this FtP arm,
    # the mechanism is genuinely different from freeze-then-probe. If they tie, the
    # claim must be rescoped. These two methods are the ONLY model-side surface; the
    # phase orchestration lives in the runner. They are pure parameter-state ops — no
    # forward/loss/state_dict-key change — so canonical paths are byte-identical.
    def freeze_backbone(self) -> int:
        """Set requires_grad=False on every representation module that produced the
        edge features (the 'backbone'): csn/ectg/drgc, the FSM stream, and (if present)
        the echo gate/norm + argmax/hier heads. Leaves ONLY main_predictor trainable.
        Returns the number of parameter tensors frozen. The stateful memory stores
        (node_mem/edge_mem/ever_alive/echo content) are NOT nn.Parameters and continue
        to update in forward — they carry temporal STATE, not learned weights."""
        frozen = 0
        for name, p in self.named_parameters():
            if name.startswith("main_predictor."):
                p.requires_grad_(True)
                continue
            if p.requires_grad:
                frozen += 1
            p.requires_grad_(False)
        return frozen

    def reinit_main_predictor(self) -> None:
        """Re-initialize the main_predictor link head from scratch (fresh probe head on
        the frozen backbone). Uses the same default init as construction so the probe
        starts tabula-rasa, decoupled from whatever head co-trained during pretraining."""
        for m in self.main_predictor.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)

    # ── FAITHFULNESS PROBE (eval-only) ────────────────────────────────────────
    def enable_faithfulness_dump(self, path: str) -> None:
        """Arm the per-event faithfulness probe (see forward). Pure logging; does
        NOT alter model/loss/gradient. Call ONCE before an eval forward pass."""
        self._dump_faithfulness = path
        self._faith_buf = None

    def flush_faithfulness(self, path: Optional[str] = None) -> Optional[str]:
        """Concatenate buffered per-event probe arrays and save to .npz. Returns the
        path written, or None if the probe was never armed / collected nothing."""
        import numpy as np
        buf = getattr(self, "_faith_buf", None)
        if buf is None:
            return None
        out = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        dest = path or self._dump_faithfulness
        np.savez(dest, **out)
        return dest

    def forward(self, src: Tensor, dst: Tensor, t: Tensor,
                feat: Tensor, neg_dst: Tensor,
                rel_type: Optional[Tensor] = None) -> Dict[str, Tensor]:

        device = self.device
        B = src.size(0)

        if feat.shape[-1] == 0:
            feat = torch.zeros(B, 1, device=device)
        elif feat.shape[-1] < self._feat_in:
            feat = F.pad(feat, (0, self._feat_in - feat.shape[-1]))

        # ── L1: Residual CSN ──
        dt_src = self.node_mem.delta_t(src, t)
        if self.determ_only_backbone:
            # DETERM-ONLY: bypass the learnable event encoder. feat_g = raw feat (no
            # learned transform), salience = 0 (no learned salience gate). ECTG still
            # consumes feat_g to advance the DETERMINISTIC point-process state stores
            # (Hawkes/Welford/recurrence), which is what the scored pair_phi reads.
            feat_g = feat
            sal = torch.zeros(B, device=device)
        else:
            feat_g, sal = self.csn(feat, dt_src)

        # ── L2: ECTG v3 (multi-signal continuous, no symbolic constraint) ──
        # causal_batch=True ⇒ pass dt_src so the store folds repeated same-pair events
        # WITHIN this batch in stream order (P1 fix). dt is ignored in the legacy path.
        edge_st = self.edge_mem.get_batch(src, dst, dt=dt_src)
        # ── PER-PAIR λ-TREND (fsm_arch="v3" + decol_use_dynamics): read the pair's
        #    PRE-update leaky-peak Hawkes λ + the carried-forward λ AT this event's time
        #    BEFORE the current jump is folded in. This is the missing DECAY signal:
        #    "đang reinforce thì cường độ khựng lại rồi GIẢM DẦN" = the carried λ has
        #    fallen off the pair's OWN recent peak. lam_carried = μ + (λ_prev−μ)·exp(−β·Δt)
        #    is exactly the Hawkes recursion's pre-jump value (sr_gnn_v3.update_multisignal
        #    L183) — derived from the store's OWN law, no fabricated channel. Both read
        #    PRE-update (peek + edge_st) ⇒ no re-leak (independent of the scored event).
        if self.fsm_arch == "v3" and self.decol_use_dynamics:
            with torch.no_grad():
                _lam_peak_pre = self.edge_mem.peek_lam_peak(src, dst)        # (B,) pre-update peak (legacy/dump)
                # EDIT-RATE fast/slow EWMA (PRE-update ⇒ no re-leak). FAST = recent rate
                # (short memory), SLOW = lagged rate (long memory). Both are EWMAs of
                # r=1/Δt, the events-per-unit-time the user defines ("7-8/5d" vs
                # "7-8/2-3d"). NOT the carried Hawkes λ — that just accumulates (≈ burst
                # count) and rises through BOTH accel AND decel phases, so it cannot sign
                # the trend (CPU-proven 2026-06-01).
                if self.causal_batch:
                    # CAUSAL (P1 fix): replay the rate EWMA + per-pair leaky peak event-
                    # by-event so repeated same-pair events in this batch read distinct
                    # PRE values (legacy peek pinned them at the pre-batch snapshot ⇒
                    # slope_rel≈0 / rate_peak≈RATE_INIT on recurring pairs). This call
                    # ALSO advances _rate_ewma/_rate_peak to post-batch ⇒ the separate
                    # update_rate_ewma/update_rate_peak below are SKIPPED (causal_batch).
                    _rate_fast, _rate_slow, _rate_peak = \
                        self.edge_mem.peek_step_rate_causal(src, dst, dt_src)
                else:
                    _rate_fast, _rate_slow = self.edge_mem.peek_rate_ewma(src, dst)
                    # PER-PAIR leaky-PEAK of the rate (PRE-update ⇒ no re-leak): the ref
                    # the RELATIVE death gate divides by. dead ⇔ rate_fast<γ·rate_peak.
                    _rate_peak = self.edge_mem.peek_rate_peak(src, dst)
                # carried Hawkes λ kept for the DUMP/analyzer continuity + as the legacy
                # "decline" level (NOT used as the slope discriminator anymore).
                _lam_prev     = edge_st[:, 6]
                _lam_carried  = (HAWKES_MU_DECOL
                                 + (_lam_prev - HAWKES_MU_DECOL)
                                 * torch.exp(-HAWKES_BETA_DECOL * dt_src.float().clamp(min=0.0)))
                # SLOPE = fast − slow EWMA of the RATE (both pre-update). The SIGN is the
                # discriminator REINFORCE↔DECAY (PM 2026-06-01 third re-chốt): >0 ⇒ rate
                # rising ⇒ REINFORCE; <0 ⇒ rate falling ⇒ DECAY (if still alive). The
                # rate LEVEL (rate_fast) gates DEATH (rate≈0).
                _lam_slope    = _rate_fast - _rate_slow
                _rate_level   = _rate_fast
                # PER-PAIR-RELATIVE slope: % change of the rate vs the pair's own slow
                # baseline. slope_rel = (fast−slow)/(slow+ε). Scale-free ⇒ the ±margin_rel
                # band transfers across datasets and OPENS on coedit (abs Δ is tiny there).
                _slope_rel    = (_rate_fast - _rate_slow) / (_rate_slow + self.decol_rate_eps)
        else:
            _lam_peak_pre = None
            _lam_carried  = None
            _lam_slope    = None
            _slope_rel    = None
            _rate_fast    = None
            _rate_slow    = None
            _rate_level   = None
            _rate_peak    = None
        new_est, edge_ctx, student_logits, heuristic_target = self.ectg(
            feat_g, sal, dt_src, edge_st
        )
        # heuristic_target (B,5) = ECTGv3's detached multi-signal (Hawkes/recurrence)
        # weak target over next FSM state. Discarded in canonical; used by the revived
        # lambda_trans transition-CE term (design="correct") to de-collapse the FSM.
        self.edge_mem.update_batch(src, dst, new_est)
        # POST-update: fold this event's λ into the per-pair leaky peak (real events
        # only; v3 + dynamics only). Read PRE (above) → update POST keeps the peek
        # strictly causal: the NEXT event sees this event in the peak, the CURRENT one
        # does not.
        if self.fsm_arch == "v3" and self.decol_use_dynamics:
            self.edge_mem.update_lam_peak(src, dst, new_est[:, 6])
            if not self.causal_batch:
                self.edge_mem.update_rate_ewma(src, dst, dt_src)
                # POST-update leaky per-pair RATE peak: read the just-written rate_fast
                # and fold it in. peek_rate_ewma is READ-ONLY (re-peeking POST is fine).
                _rf_post, _ = self.edge_mem.peek_rate_ewma(src, dst)
                self.edge_mem.update_rate_peak(src, dst, _rf_post)
            # causal_batch=True: peek_step_rate_causal (above) already advanced
            # _rate_ewma and _rate_peak to post-batch, event-by-event ⇒ no double-update.
        # Collision-immune TRUE occurrence index per event (1-based, stream order). Used
        # by the de-collapse target's is_first (n_obs from Welford is corrupted by
        # intra-batch read-before-write — DIAGNOSED 2026-05-30). Only consumed when
        # decollapse_target is on; cheap (computed always so reset/state stays coherent).
        true_occ = self.edge_mem.get_true_occ(src, dst)  # (B,)
        hawkes_lam = new_est[:, 6]  # hawkes intensity from ECTGv3

        # ── L3: DRGC v2 ──
        h_src = self.node_mem.get(src)
        h_dst = self.node_mem.get(dst)
        dt_dst = self.node_mem.delta_t(dst, t)
        all_idx = torch.unique(torch.cat([src, dst]))
        all_h = self.node_mem.get(all_idx)
        all_staleness = (t.max().float() - self.node_mem.last_t[all_idx]).clamp(0)

        # ── Echo Memory injection (improvement #3, enable_echo=True only) ──
        # Faithful to experiments/models/sr_gnn_v3.py:690-695: decayed echo (from
        # BEFORE this batch, @no_grad content) is normalized + gated and ADDED to
        # the raw node memory before DRGC. echo_gate/echo_norm are learnable, so
        # echo measurably shifts the score and receives gradient through them.
        if self.echo is not None:
            echo_src = self.echo.decay_get(src, t)
            echo_dst = self.echo.decay_get(dst, t)
            echo_g   = torch.sigmoid(self.echo_gate)
            h_src_in = h_src + echo_g * self.echo_norm(echo_src)
            h_dst_in = h_dst + echo_g * self.echo_norm(echo_dst)
        else:
            h_src_in = h_src
            h_dst_in = h_dst

        if self.determ_only_backbone:
            # DETERM-ONLY: bypass the learnable DRGC coupled-GRU node-memory update.
            # Node memory passes through unchanged (no learned message-passing / GRU /
            # TIP compression), so new_h_* carry NO learnable-backbone signal. parsed_h
            # = all_h unchanged keeps the node_mem.set write coherent; kl = 0 so the
            # lambda_echo*kl term contributes nothing (and there is nothing to train).
            new_h_src, new_h_dst = h_src_in, h_dst_in
            parsed_h = all_h
            kl = torch.zeros((), device=device)
        else:
            new_h_src, new_h_dst, parsed_h, kl = self.drgc(
                h_src_in, h_dst_in, feat_g, edge_ctx, dt_src, dt_dst,
                hawkes_lam, all_h, all_staleness
            )

        # ── L4a: Main Edge Predictor (NON-detached) — trains the backbone ──
        # This is the link-prediction head. Its gradient flows through new_h_src /
        # new_h_dst into DRGC → ECTG → CSN, which the detached FSM stream never did.
        edge_h_pos = self._build_edge_h(new_h_src, new_h_dst)  # (B, 2H)
        neg_emb_main = self.node_mem.get(neg_dst)
        if self.echo is not None:
            # Parity with pos: augment neg dst with its own gated echo (same gate).
            neg_echo = self.echo.decay_get(neg_dst, t)
            neg_emb_main = neg_emb_main + echo_g * self.echo_norm(neg_echo)
        edge_h_neg_main = self._build_edge_h(new_h_src, neg_emb_main)
        # IDENTICAL-HEAD K1 DETACH PROBE (reviewer §8.2.i): the SAME 2-layer MLP head
        # scores both arms; the SOLE difference is whether its backbone input is
        # detached. main_predictor_detach=True ⇒ DETACHED-MLP (zero backbone link
        # grad, head/init identical to COUPLED-MLP). False ⇒ COUPLED-MLP (canonical
        # K1, gradient flows). No-op unless enable_main_predictor=True (else this head
        # is unused). edge_h_neg_main detaches in lockstep so neither pos nor neg leaks
        # link gradient into the backbone in the DETACHED arm.
        _mp_in_pos = edge_h_pos.detach() if self.main_predictor_detach else edge_h_pos
        _mp_in_neg = (edge_h_neg_main.detach()
                      if self.main_predictor_detach else edge_h_neg_main)
        main_pos_logit = self.main_predictor(_mp_in_pos).squeeze(-1)      # (B,)
        main_neg_logit = self.main_predictor(_mp_in_neg).squeeze(-1)  # (B,)

        # ── L4b: FSM Stream (stop-grad to backbone) — interpretation only ──

        # DETERM-ONLY: zero the backbone-derived edge_h fed to the FSM SCORING head so
        # the scored next-state distribution depends on NO learnable CSN/DRGC content —
        # only on the deterministic pair_phi (Hawkes/Welford/recurrence/staleness/
        # ever_alive) and the lifecycle mask/gate. state_observer/transition_predictor
        # still receive a constant zero (their bias terms remain, identical for pos/neg
        # so no leak), but carry zero backbone signal/gradient. edge_h_neg is zeroed
        # symmetrically at its build site below.
        if self.determ_only_backbone:
            edge_h_pos = torch.zeros_like(edge_h_pos)

        # Soft current state from history (read-only).
        # DETACH PROBE: when edge_h_detach_scorepath=False, let gradient flow through
        # edge_h_pos into the backbone (the AP-scoring s_t_pos→s_t1_pos→existence_decoder
        # chain becomes a trainable link-pred head). Default True = canonical detach.
        _detach_score = self.edge_h_detach_scorepath
        s_t_pos = self.state_observer(edge_h_pos, detach_h=_detach_score)  # (B, 5)

        # Get ever_alive for this edge
        ever_alive_pos = self.ever_alive.get(src, dst)  # (B,)

        # ── Per-pair operator features φ_uv (fsm_arch="v3" only) ──────────────────
        # Each interacting pair gets its OWN flip dynamics, derived from that pair's
        # CONTINUOUS history already accumulated in EdgeStateStoreV3 / ECTGv3 (new_est):
        #   [6]=Hawkes λ, [7]=Welford mean inter-event gap, [8]=Welford var,
        #   [5]=recurrence EWMA, dt_src=staleness (Δt since last touch), ever_alive.
        # Detached (history features + the FSM head is decoupled from the backbone by
        # design — TransitionPredictor reads edge_h.detach()). pair_phi feeds g(φ_uv)
        # which biases the per-pair transition operator T_uv. v1/v2 → pair_phi=None.
        if self.fsm_arch == "v3":
            with torch.no_grad():
                # ── ANTI-LEAK FIX (2026-05-31, ml): build φ_uv from the PRE-update
                # edge state ``edge_st`` (the state BEFORE this event), NOT the
                # post-update ``new_est``. new_est has ALREADY folded the current
                # event into the Hawkes jump / recurrence EWMA / Welford stats, so a
                # φ from new_est tells the per-pair operator "this event happened" —
                # information a true POSITIVE has and a counterfactual NEGATIVE does
                # not → a textbook temporal label leak (smoke 5448819: trans_ap=1.0).
                # edge_st[:, k] are the same channels pre-event: [6]=Hawkes λ,
                # [7]=mean_dt, [8]=var_dt, [5]=recurrence EWMA. This restores the
                # canonical "score BEFORE memory update" invariant for the v3 operator.
                # PER-PAIR-RELATIVE z from PRE-update Welford (anti-leak: edge_st = state
                # BEFORE this event; does NOT fold in the current Δt). Count-guarded: a
                # pair with <2 prior obs has a degenerate σ → z forced 0 (matches the
                # target guard) so single-shot pairs feed the operator a neutral z, not
                # noise. clamp keeps the operator input bounded.
                _mean_pre = edge_st[:, 7]
                _std_pre  = edge_st[:, 8].clamp(min=1e-6).sqrt()
                _z_pre    = (dt_src.float() - _mean_pre) / (_std_pre + 1e-6)
                _z_pre    = _z_pre * (edge_st[:, 9] >= 2.0).float()
                pair_phi = torch.stack([
                    edge_st[:, 6],                              # Hawkes λ  (pre-update)
                    torch.log1p(edge_st[:, 7].clamp(min=0)),   # log mean_dt
                    torch.log1p(edge_st[:, 8].clamp(min=0)),   # log var_dt
                    edge_st[:, 5],                             # recurrence EWMA
                    torch.log1p(dt_src.float().clamp(min=0)),  # staleness
                    ever_alive_pos,                            # ever_alive ∈[0,1]
                    _z_pre.clamp(-5.0, 5.0),                   # per-pair-relative z (NEW)
                ], dim=-1)                                # (B, 7)
            # DETACH PROBE: same single bit — un-detach edge_h on the SCORING path only.
            _h_score_pos = edge_h_pos if not _detach_score else edge_h_pos.detach()
            trans_logits_pos, T_uv_pos = self.transition_predictor(
                _h_score_pos, s_t_pos, pair_phi=pair_phi, return_T=True)
        else:
            pair_phi = None
            T_uv_pos = None
            _h_score_pos = edge_h_pos if not _detach_score else edge_h_pos.detach()
            trans_logits_pos = self.transition_predictor(_h_score_pos, s_t_pos)

        # Apply lifecycle FSM mask
        mask_pos = self.lifecycle_mask.get_mask_from_state(s_t_pos)
        s_t1_pos = trans_logits_pos + (mask_pos + 1e-6).log()  # log-space addition
        s_t1_pos = self.lifecycle_mask.apply_ever_alive_gate(s_t1_pos, ever_alive_pos)
        s_t1_pos = torch.softmax(s_t1_pos, dim=-1)  # final next-state distribution

        # ── HIERARCHICAL STATE READOUT (fsm_decode="hier", fsm_arch="v3") ──────────
        # STRUCTURAL fix for the DECAY-argmax-never-wins dead-end (legacy diagnostic). The
        # flat softmax s_t1_pos puts BIRTH/REINFORCE/DECAY/DEATH on ONE flat axis; DECAY,
        # the MIDDLE of the sequential lifecycle BIRTH→REINFORCE→DECAY→DEATH, has its
        # mass split between its two neighbours so it cannot win argmax even when its
        # PROB is faithful (ρ=0.887). We instead DECODE the next-state distribution as a
        # DECISION TREE so DECAY competes ONLY with REINFORCE (its sibling in the alive
        # branch) — never directly with DEATH:
        #   P(BIRTH)     = p_birth
        #   P(REINFORCE) = (1−p_birth)·p_alive·p_rising
        #   P(DECAY)     = (1−p_birth)·p_alive·(1−p_rising)
        #   P(DEATH)     = (1−p_birth)·(1−p_alive)
        #   P(IDLE)      = 0   (no separate IDLE state in this lifecycle readout)
        # Each gate = sigmoid(analytic_prior_logit + learnable_residual_head(φ_g)). The
        # analytic prior makes the zero-trained gate already correct (so a fresh model
        # reads the right state); the de-collapse KL on s_t1_cal then trains the residual
        # heads (gradient flows). ALL inputs are PRE-update (computed at the top of
        # forward: _slope_rel, _rate_fast, _rate_peak, dt_src, true_occ, edge_st[:,9]) so
        # there is NO label leak. This reroutes ONLY s_t1_cal (the gate + faithfulness +
        # de-collapse-CE quantity). s_t1_pos → existence_decoder → AP score is UNTOUCHED.
        _hier_probs = None  # (p_birth, p_alive, p_rising) for the dump (hier only)
        s_t1_cal6   = None  # HƯỚNG A 6-state published dist (strict_ordered_fsm only)
        if (self.fsm_decode == "hier" and self.fsm_arch == "v3"
                and self.decol_use_dynamics and _slope_rel is not None
                and _rate_fast is not None and _rate_peak is not None):
            with torch.no_grad():
                _n_prior_h = edge_st[:, 9].float()
                _has_hist_h = (_n_prior_h >= 2.0).float()
                # per-pair DEAD floor = γ·leaky-peak rate (same as the target gate).
                _floor_h = (self.decol_rate_dead_gamma
                            * _rate_peak.clamp(min=RATE_INIT))
                # rate ABOVE its own floor (log-ratio, per-pair scale-free, clamped).
                _rate_ratio = torch.log((_rate_fast.clamp(min=1e-4))
                                        / (_floor_h + 1e-4)).clamp(-4.0, 4.0)
                # staleness RELATIVE to the pair's own mean gap: recently edited (small
                # dt vs μ_pair) ⇒ STILL ALIVE regardless of rate slope. This is the P1
                # fix — the per-pair leaky-max alone mislabels a "burst-then-slow" pair
                # DEAD; the time-since-last term keeps a recently-active pair alive.
                _mean_pre_h = edge_st[:, 7].clamp(min=1e-3)
                _stale_rel  = (dt_src.float() / _mean_pre_h).clamp(0.0, 10.0)
            # ── gate FEATURES (per gate, PRE-update, no leak) ─────────────────────
            # birth: how-new (true_occ, n_prior). alive: rate-vs-floor, staleness, slope.
            # rising: slope_rel, rate-vs-floor.
            _birth_feat = torch.stack([
                (true_occ.float() <= 1.0).float(),       # is genuine first occurrence
                torch.log1p(_n_prior_h),                 # log prior-event count
                torch.log1p(dt_src.float().clamp(min=0)),# staleness (fresh edges look new)
                _has_hist_h,                             # has trustworthy history
            ], dim=-1)
            _alive_feat = torch.stack([
                _rate_ratio,                             # rate above per-pair dead floor
                -_stale_rel,                             # recently edited ⇒ alive (neg stale)
                _slope_rel.clamp(-5.0, 5.0),             # rising rate ⇒ clearly alive
                _has_hist_h,
            ], dim=-1)
            _rising_feat = torch.stack([
                _slope_rel.clamp(-5.0, 5.0),             # the sign axis REINFORCE↔DECAY
                _rate_ratio,                             # well-above-floor pairs reinforce
                torch.log1p(_rate_fast.clamp(min=0)),    # rate level
                _has_hist_h,
            ], dim=-1)
            # ── ANALYTIC prior logits (zero-trained gate == this soft function) ───
            # p_birth: HIGH on the true first occurrence / low-history pairs, ≈0 once the
            #   pair has accumulated history.
            _birth_prior = 4.0 * ((true_occ.float() <= 1.0).float() - 0.5) \
                           - 1.5 * (_n_prior_h >= 2.0).float()
            if self.decol_hier_v2:
                # ── HIER v2 priors (round-7 fix, 2026-06-02) ──────────────────────
                # ROOT-CAUSE (npz faithfulness_coedit_v3_hier_let0.5_s42, decisive):
                #   • The rate/Welford signals (_rate_fast, _rate_peak, edge_st[:,7/8/9]=
                #     μ/var/n_obs) are CORRUPTED ON RECURRING PAIRS by intra-batch read-
                #     before-write: get_batch snapshots the pair ONCE per 500-event batch,
                #     so a pair firing K× in a batch reads the SAME stale row K× and only
                #     ONE fold survives. Result in the dump: n_prior caps at 6 for pairs
                #     editing 201×; rate_fast/peak pinned at RATE_INIT=0.1 (coedit median
                #     gap≈10 ⇒ 1/dt≈0.1≡init) ⇒ slope_rel med 0.000 (REINFORCE unreachable)
                #     AND μ_pair≈0 ⇒ _stale_rel saturates at the cap (looks DEAD). These
                #     corrupted signals are WHY v1 gave DEATH 93.8% where the observed
                #     TARGET is REINFORCE 95%.
                #   • true_occ (_occ_count) is the ONE reliable persistent signal — it is
                #     incremented PER-EVENT inside the loop, immune to the batch snapshot.
                # DESIGN: anchor ALIVE on true_occ. A POSITIVE event at occurrence ≥2 is BY
                #   CONSTRUCTION an active recurring pair that JUST edited ⇒ ALIVE. DEATH is
                #   reserved for the SUSTAINED-silence case we can actually observe: a
                #   trustworthy long gap (only when _has_hist_h, i.e. μ_pair is real). So
                #   the corrupted stale_rel is gated behind _has_hist_h and never drives a
                #   recurring active pair to DEATH.
                # p_alive: high for any pair with history (true_occ≥2). Subtract only a
                #   TRUSTWORTHY sustained-silence term (gated on _has_hist_h so corrupted
                #   μ_pair≈0 cannot fire it). This lifts p_alive well above 0.5 on the
                #   recurring set ⇒ the alive branch beats the UNDIVIDED-DEATH leaf.
                _recurring = (true_occ.float() >= 2.0).float()
                _stale_center = 0.5 * (1.0 + self.decol_dead_stale_mult)
                _sustained_dead = (_has_hist_h
                                   * (_stale_rel - _stale_center).clamp(min=0.0))
                _alive_prior = (2.0 * _recurring
                                - 2.2 * _sustained_dead
                                + 0.4 * _rate_ratio)
                # p_rising: REINFORCE vs DECAY inside the alive branch. Default a recurring
                #   active pair to RISING (REINFORCE) — on coedit the overwhelming observed
                #   target on recurring is REINFORCE (95%), so REINFORCE must be the alive-
                #   branch DEFAULT, not a knife-edge. DECAY is carved out only when there is
                #   a TRUSTWORTHY cooling signal (slope_rel falling OR trustworthy staleness
                #   rising), gated on _has_hist_h. This is the REINFORCE-survival fix: the
                #   degenerate slope_rel (≈0) no longer collapses p_rising to a coin-flip.
                _cooling = (_has_hist_h
                            * ((_stale_rel - 1.0).clamp(min=0.0)
                               - self.decol_slope_scale
                                 * _slope_rel.clamp(-5.0, 5.0)))
                _rising_prior = (1.2 * _recurring
                                 - 1.5 * _cooling)
            else:
                # p_alive: rate above its own floor OR recently edited ⇒ alive; rate≈0 AND
                #   stale ⇒ dead. Combine rate-ratio with the staleness-rel penalty. Pairs
                #   without enough history default toward alive (cannot be confidently dead).
                _alive_prior = (self.decol_rate_dead_scale * 0.3) * _rate_ratio \
                               - 0.8 * (_stale_rel - 1.0) \
                               + 1.0 * (1.0 - _has_hist_h)
                # p_rising: sign of the per-pair-relative rate slope. ≥+margin_rel ⇒ rising
                #   (REINFORCE); ≤−margin_rel ⇒ falling (DECAY). Centered at 0.
                _rising_prior = self.decol_slope_scale * _slope_rel.clamp(-5.0, 5.0)
            # ── gate = sigmoid(prior + learnable residual). Residual heads zero-init at
            #    OUTPUT ⇒ gate starts EXACTLY at the analytic prior; KL trains the heads.
            p_birth  = torch.sigmoid(
                _birth_prior  + self.hier_birth_head(_birth_feat).squeeze(-1))
            p_alive  = torch.sigmoid(
                _alive_prior  + self.hier_alive_head(_alive_feat).squeeze(-1))
            p_rising = torch.sigmoid(
                _rising_prior + self.hier_rising_head(_rising_feat).squeeze(-1))
            # ── compose the hierarchical next-state distribution (B,5) ────────────
            _not_birth = 1.0 - p_birth
            _hier = torch.zeros_like(s_t1_pos)
            _hier[:, IDLE]      = 0.0
            _hier[:, BIRTH]     = p_birth
            _hier[:, REINFORCE] = _not_birth * p_alive * p_rising
            _hier[:, DECAY]     = _not_birth * p_alive * (1.0 - p_rising)
            _hier[:, DEATH]     = _not_birth * (1.0 - p_alive)
            # numeric safety: renormalize (IDLE=0; the four terms already sum to 1 up to
            # fp error, but renorm guards the clamp(min=1e-8).log() downstream).
            s_t1_cal = _hier / _hier.sum(-1, keepdim=True).clamp(min=1e-8)

            # ── CAUSAL POLICY on the PUBLISHED state s_t1_cal (hier_causal_policy) ────
            # FIX (PM audit 2026-06-03): the hier tree above can emit DEATH=0.9 on a pair
            # that was NEVER alive, and can place mass on causally-IMPOSSIBLE transitions
            # (e.g. REINFORCE→DEATH directly), because the tree distribution BYPASSES the
            # ever_alive gate + CAUSAL_RULE_MATRIX that s_t1_pos passes through (L968-976).
            # Here we make the PUBLISHED interpretable state obey causal policy. This
            # touches ONLY s_t1_cal (the gate/faithfulness/CE quantity); s_t1_pos →
            # existence_decoder → AP score is untouched ⇒ AP Δ=0 EXACT. Both inputs are
            # PRE-update (ever_alive_pos = .get at L935 BEFORE update_batch; s_t_pos from
            # the history-only StateObserver) ⇒ NO leak. Operations are MULTIPLICATIVE +
            # renorm (differentiable) so the de-collapse CE gradient still trains the hier
            # heads. Soft (not hard zeroing of probs) where it matters, so the 5-state
            # distribution is not collapsed to a single state by masking.
            if self.hier_causal_policy and self.strict_ordered_5state:
                # ── OPTION (b): STRICT-ORDERED 5-STATE NATIVE (PM 2026-06-06) ─────
                # NO IDLE split, NO ever_alive gate, NO soft floor-blend. We take the
                # raw hier-tree distribution s_t1_cal (IDLE,BIRTH,REINFORCE,DECAY,DEATH)
                # and HARD-mask every non-adjacent ("nhảy cóc") transition with the
                # strict band-diagonal C_BAND_5 (|i−j|≤1). Because IDLE(0) and DEATH(4)
                # are the two ENDS of the axis, IDLE→DEATH is band-blocked ⇒
                # death-before-alive is enforced by the ORDERING alone (the ever_alive
                # gate is structurally REDUNDANT here). The expected-admissibility uses
                # the FULL current-state distribution s_t_pos (NOT argmax — avoids the
                # near-uniform brittleness) and is binarized to a HARD {0,1} mask so a
                # fully-forbidden next-state is driven to EXACTLY 0 (then renormed).
                # Touches ONLY s_t1_cal ⇒ AP path (s_t1_pos) untouched ⇒ AP Δ=0 EXACT.
                if self._band5_C.device != s_t1_cal.device:
                    self._band5_C = self._band5_C.to(s_t1_cal.device)
                _st_cur5 = s_t_pos.detach()                          # (B,5) current state
                # M5[b,j] = Σ_i s_t[b,i]·C[i,j]: prob-mass of occupied current states
                # that ADMIT a transition into j. j is forbidden iff EVERY occupied
                # current state forbids it (M5[j]≈0).
                _M5 = torch.einsum("bi,ij->bj", _st_cur5, self._band5_C)   # (B,5)
                _M5_hard = (_M5 > 1e-6).to(s_t1_cal.dtype)          # HARD {0,1} band mask
                s_t1_cal = s_t1_cal * _M5_hard
                _den5 = s_t1_cal.sum(-1, keepdim=True)
                # Self-loops are always admissible (C[i,i]=1) so as long as s_t has any
                # mass the row is non-empty; the torch.where guards the degenerate
                # all-zero current-state row (never happens in practice) → fall back to
                # the unmasked hier tree so we never emit an all-zero row / NaN.
                s_t1_cal = torch.where(
                    _den5 > 1e-8,
                    s_t1_cal / _den5.clamp(min=1e-8),
                    _hier / _hier.sum(-1, keepdim=True).clamp(min=1e-8),
                )
                s_t1_cal6 = None        # 5-state native ⇒ no 6-state view emitted
            elif self.hier_causal_policy:
                if self._hier_causal_C.device != s_t1_cal.device:
                    self._hier_causal_C = self._hier_causal_C.to(s_t1_cal.device)
                # (1) ever_alive gate — CANNOT die before ever being alive. ever_alive_pos
                #     ∈[0,1] PRE-update. The DEATH leaf is scaled by ever_alive; the freed
                #     dead-mass (1−ever_alive)·P(DEATH) is REDISTRIBUTED to IDLE (a pair
                #     that was never alive and is "asked to die" is, causally, still IDLE —
                #     it has not been born; IDLE is the honest pre-birth state). BIRTH is
                #     left to the tree's own p_birth so we do not double-count births.
                _ea = ever_alive_pos.clamp(0.0, 1.0)                     # (B,)
                _dead_mass   = s_t1_cal[:, DEATH]                        # (B,)
                _freed_death = (1.0 - _ea) * _dead_mass                 # mass to move out
                s_t1_cal = s_t1_cal.clone()
                s_t1_cal[:, DEATH] = _ea * _dead_mass
                s_t1_cal[:, IDLE]  = s_t1_cal[:, IDLE] + _freed_death
                # renorm (no-op mathematically — mass is conserved — but guards fp drift)
                s_t1_cal = s_t1_cal / s_t1_cal.sum(-1, keepdim=True).clamp(min=1e-8)

                # (2) SOFT EXPECTED-ADMISSIBILITY mask (PM 2026-06-03, supersedes the
                #     hard-argmax C-row mask). The previous version used C[argmax(s_t),:],
                #     which is BRITTLE: when s_t_pos is near-uniform (StateObserver
                #     uncertain), argmax(s_t) jitters / lands on DEATH, and the chosen ROW
                #     C[DEATH,:]={0,0,0,1,1} forbids everything except DECAY/DEATH ⇒
                #     s_t1_cal collapses onto DECAY (the diagnosed 0.92 collapse). Instead
                #     we use the FULL soft distribution s_t to take the EXPECTED admissibility
                #     of each NEXT state, mirroring LifecycleFSMMask.get_mask_from_state
                #     (fsm_head.py L170-180, the einsum "bi,ij->bj" pattern):
                #         M[b,j] = Σ_i s_t[b,i] · C[i,j]
                #     M[b,j] ∈ [0,1] is the probability MASS of current states that admit a
                #     transition INTO j. A near-uniform s_t therefore averages over all five
                #     rows of C and yields a SMOOTH M (no row pins everything onto one state),
                #     so multiplying s_t1_cal by M and renormalizing CANNOT collapse the
                #     5-state distribution onto a single state. Causally-impossible
                #     transitions (M[j]→0 only when EVERY occupied current state forbids j)
                #     are still suppressed, but proportionally to how CERTAIN s_t is.
                # C-FLOOR blend: M ← floor + (1−floor)·M with floor=0.05. This guarantees
                #     M[j] ≥ 0.05 for every state, so when s_t is uncertain we never HARD-zero
                #     any next-state (avoids a confident mask from an unconfident current
                #     state). At floor=0.05 a fully-forbidden state is attenuated ~20× vs an
                #     admissible one (0.05 vs ~1.0) — strong suppression, never a hard kill.
                _C_floor = 0.05
                _M = torch.einsum("bi,ij->bj", s_t_pos.detach(),
                                  self._hier_causal_C)          # (B,5) expected admissibility
                _M = _C_floor + (1.0 - _C_floor) * _M          # floor-blend, never 0
                s_t1_cal = s_t1_cal * _M
                # renorm; denom ≥ floor·(min mass) > 0 always ⇒ no all-zero row, no NaN.
                # The torch.where fallback is kept only as a defensive numeric guard.
                _denom = s_t1_cal.sum(-1, keepdim=True)
                s_t1_cal = torch.where(
                    _denom > 1e-8,
                    s_t1_cal / _denom.clamp(min=1e-8),
                    _hier / _hier.sum(-1, keepdim=True).clamp(min=1e-8),
                )

                # ── HƯỚNG A: STRICT-ORDERED 6-STATE LIFECYCLE (PM 2026-06-06) ─────
                # Re-express the (now ever_alive-gated) 5-state s_t1_cal as the 6-state
                # Markovian lifecycle by SPLITTING IDLE into PRE_BIRTH vs DORMANT using
                # the IN-STATE ever_alive evidence, then HARD-mask every non-adjacent
                # ("nhảy cóc") transition with the strict band-diagonal C' and renorm.
                # The 5-state s_t1_cal IDLE leaf carries: (a) the tree's IDLE (≈0) and
                # (b) the freed dead-mass moved from DEATH for never-alive pairs (the
                # ever_alive redistribution above). We split that IDLE mass:
                #   PRE_BIRTH ← IDLE · (1 − ever_alive)   (never alive → honest pre-birth)
                #   DORMANT   ← IDLE · ever_alive          (was alive, now silent)
                # The remaining 5-state DEATH leaf (already scaled by ever_alive) maps
                # straight to the 6-state DEATH. BIRTH/REINFORCE/DECAY map 1:1.
                if self.strict_ordered_fsm:
                    if self._strict_C.device != s_t1_cal.device:
                        self._strict_C = self._strict_C.to(s_t1_cal.device)
                    _ea6 = ever_alive_pos.clamp(0.0, 1.0)            # (B,)
                    _idle5 = s_t1_cal[:, IDLE]                       # (B,) 5-state IDLE mass
                    s6 = torch.zeros(s_t1_cal.size(0), SO_N_STATES,
                                     device=s_t1_cal.device, dtype=s_t1_cal.dtype)
                    s6[:, SO_PRE_BIRTH] = _idle5 * (1.0 - _ea6)
                    s6[:, SO_DORMANT]   = _idle5 * _ea6
                    s6[:, SO_BIRTH]     = s_t1_cal[:, BIRTH]
                    s6[:, SO_REINFORCE] = s_t1_cal[:, REINFORCE]
                    s6[:, SO_DECAY]     = s_t1_cal[:, DECAY]
                    s6[:, SO_DEATH]     = s_t1_cal[:, DEATH]
                    s6 = s6 / s6.sum(-1, keepdim=True).clamp(min=1e-8)

                    # STRICT C' mask. PM wants "không thể nhảy" ⇒ HARD-mask the
                    # forbidden (non-adjacent) transitions to ~0 via the expected-
                    # admissibility M6[b,j] = Σ_i s6_t[b,i]·C'[i,j], but with NO floor
                    # (unlike the soft 5-state policy) so a fully-forbidden next-state is
                    # driven to exactly 0 (renormalized → no NaN). The current-state
                    # distribution s6_t is the SPLIT of s_t_pos's IDLE the same way (so
                    # the mask "knows" whether the pair is pre-birth or dormant).
                    _st_pos5 = s_t_pos.detach()                     # (B,5) current state
                    _idle_cur = _st_pos5[:, IDLE]
                    s6_t = torch.zeros_like(s6)
                    s6_t[:, SO_PRE_BIRTH] = _idle_cur * (1.0 - _ea6)
                    s6_t[:, SO_DORMANT]   = _idle_cur * _ea6
                    s6_t[:, SO_BIRTH]     = _st_pos5[:, BIRTH]
                    s6_t[:, SO_REINFORCE] = _st_pos5[:, REINFORCE]
                    s6_t[:, SO_DECAY]     = _st_pos5[:, DECAY]
                    s6_t[:, SO_DEATH]     = _st_pos5[:, DEATH]
                    s6_t = s6_t / s6_t.sum(-1, keepdim=True).clamp(min=1e-8)
                    _M6 = torch.einsum("bi,ij->bj", s6_t, self._strict_C)  # (B,6)
                    # HARD mask: forbidden next-states (M6≈0 because EVERY occupied
                    # current state forbids them) are zeroed; admissible ones kept. We
                    # binarize M6 with a tiny eps so a current state that is itself a
                    # rounding-zero does not accidentally admit a forbidden target.
                    _M6_hard = (_M6 > 1e-6).to(s6.dtype)
                    s6 = s6 * _M6_hard
                    _den6 = s6.sum(-1, keepdim=True)
                    # If the hard mask zeroed everything (pathological: all mass on a
                    # next-state forbidden by the current state — should not happen since
                    # self-loops are always admissible), fall back to the masked-soft M6
                    # so we never emit an all-zero row / NaN.
                    s6 = torch.where(
                        _den6 > 1e-8,
                        s6 / _den6.clamp(min=1e-8),
                        s6_t,  # self-loop-admissible fallback (= current dist)
                    )
                    s_t1_cal6 = s6

                    # FOLD BACK to the legacy 5-state s_t1_cal so the de-collapse CE
                    # target (B,5) and update_symbolic are byte-shape-identical:
                    #   IDLE ← PRE_BIRTH + DORMANT ; the other four map 1:1. This makes
                    # the strict C' (which CANNOT place mass on a forbidden 5-state path
                    # either, since folding only merges the two IDLE sub-states) the
                    # supervised/measured distribution too — so the strict ordering is
                    # enforced on the trained quantity, not only on the 6-state view.
                    s_t1_cal = torch.zeros_like(s_t1_cal)
                    s_t1_cal[:, IDLE]      = s6[:, SO_PRE_BIRTH] + s6[:, SO_DORMANT]
                    s_t1_cal[:, BIRTH]     = s6[:, SO_BIRTH]
                    s_t1_cal[:, REINFORCE] = s6[:, SO_REINFORCE]
                    s_t1_cal[:, DECAY]     = s6[:, SO_DECAY]
                    s_t1_cal[:, DEATH]     = s6[:, SO_DEATH]
                    s_t1_cal = s_t1_cal / s_t1_cal.sum(-1, keepdim=True).clamp(min=1e-8)
            _hier_probs = (p_birth, p_alive, p_rising)
        # ── ARGMAX-CALIBRATED next-state distribution (decol_argmax_bias, v3) ──────
        # s_t1_cal = softmax(log s_t1_pos + (argmax_bias ⊙ mask)). This is the
        # distribution we SUPERVISE (edge-trans CE) and MEASURE (update_symbolic gate +
        # the FSM symbolic state). The learnable bias on DECAY/DEATH lets the argmax
        # COMMIT to those intermediate states when their soft mass is competitive but
        # loses the raw argmax to REINFORCE (the diagnosed commitment failure). The raw
        # s_t1_pos still feeds existence_decoder (pos_logit, the AP readout) UNCHANGED,
        # so AP is invariant to this parameter. When the flag is off (v1/v2/plain), the
        # bias is None ⇒ s_t1_cal IS s_t1_pos (object-identical) ⇒ byte-identical.
        # NOTE: skipped entirely when hier decode already produced s_t1_cal (the
        # hierarchical tree IS the calibrated readout in that mode).
        elif getattr(self, "argmax_bias", None) is not None:
            _bias = self.argmax_bias * self._argmax_bias_mask    # only DECAY,DEATH active
            s_t1_cal = torch.softmax(s_t1_pos.clamp(min=1e-8).log() + _bias, dim=-1)
        else:
            s_t1_cal = s_t1_pos

        # ── FAITHFULNESS DUMP (eval-only, default OFF — pure logging) ──────────────
        # When self._dump_faithfulness is set (a path; set externally by the eval
        # driver, NEVER in __init__ / loss), append per-event PRE-update probe arrays
        # for offline faithfulness analysis. This block is wrapped in no_grad, reads
        # ONLY the already-computed PRE-update tensors (edge_st captured at L546 BEFORE
        # update_batch L553; dt_src/true_occ/hawkes_lam) and the finalized prediction
        # s_t1_pos — it does NOT touch any tensor that feeds the loss/gradient, does
        # NOT mutate any store, and is independent of fsm_arch/decollapse_target. So
        # AP, state_dist and every loss term are byte-identical whether it is on/off.
        #   z_pair = (dt − μ_pair^pre)/(σ_pair^pre+ε) where σ = sqrt(edge_st[:,8]);
        #            edge_st[:,8] stores VARIANCE in this LAB store (sr_gnn_v3
        #            update_multisignal L199 writes var_new), so sqrt → std. This is
        #            BYTE-IDENTICAL to the z the v3 de-collapse target computes
        #            (L904-905: var_dt=edge_st[:,8]; std_dt=var_dt.sqrt()). "Is THIS
        #            gap long FOR THIS PAIR" — the per-pair signal the analysis buckets.
        #   n_prior = edge_st[:,9] (PRE-update Welford count = # events of this pair
        #            BEFORE the current one). p_decay/p_death = s_t1_pos[:,DECAY/DEATH].
        # Index literals 7/8/9 match the de-collapse block above (this LAB store does
        # NOT export IDX_* symbols).
        if getattr(self, "_dump_faithfulness", None) is not None:
            with torch.no_grad():
                _mean_dt = edge_st[:, 7].float()
                _var_pre = edge_st[:, 8].float()                   # variance (LAB store)
                _n_prior = edge_st[:, 9].float()
                _std_pre = _var_pre.clamp(min=1e-6).sqrt()
                _z_pair  = (dt_src.float() - _mean_dt) / (_std_pre + 1e-6)
                _rec = {
                    "z_pair":   _z_pair.detach().cpu().numpy(),
                    "n_prior":  _n_prior.detach().cpu().numpy(),
                    "true_occ": true_occ.detach().float().cpu().numpy(),
                    # RAW (pre-calibration) head distribution — preserved for ρ continuity
                    # and a pre/post argmax contrast. p_* are RAW so the faithfulness
                    # Spearman ρ(z, p_decay+p_death) is comparable to the prior probe.
                    "argmax_s_t1_pos": s_t1_pos.argmax(-1).detach().cpu().numpy(),
                    "p_idle":   s_t1_pos[:, IDLE].detach().cpu().numpy(),
                    "p_birth":  s_t1_pos[:, BIRTH].detach().cpu().numpy(),
                    "p_reinforce": s_t1_pos[:, REINFORCE].detach().cpu().numpy(),
                    "p_decay":  s_t1_pos[:, DECAY].detach().cpu().numpy(),
                    "p_death":  s_t1_pos[:, DEATH].detach().cpu().numpy(),
                    # CALIBRATED (post argmax-bias) — the quantity the gate/measure() now
                    # commits on; lets evaluation harness measure post-calibration argmax accuracy.
                    # == RAW when decol_argmax_bias is off (s_t1_cal is s_t1_pos).
                    "argmax_s_t1_cal": s_t1_cal.argmax(-1).detach().cpu().numpy(),
                    "p_decay_cal":  s_t1_cal[:, DECAY].detach().cpu().numpy(),
                    "p_death_cal":  s_t1_cal[:, DEATH].detach().cpu().numpy(),
                    # HIERARCHICAL-DECODE gate probs (fsm_decode="hier"). p_birth/p_alive/
                    # p_rising are the three tree gates; the analyzer can read DECAY mass
                    # directly as (1−p_birth)·p_alive·(1−p_rising). Recorded as all-zeros
                    # when hier decode is off (flat) so the npz schema stays stable.
                    "p_birth_gate": (
                        _hier_probs[0].detach().cpu().numpy()
                        if _hier_probs is not None
                        else torch.zeros(B, device=device).cpu().numpy()),
                    "p_alive_gate": (
                        _hier_probs[1].detach().cpu().numpy()
                        if _hier_probs is not None
                        else torch.zeros(B, device=device).cpu().numpy()),
                    "p_rising_gate": (
                        _hier_probs[2].detach().cpu().numpy()
                        if _hier_probs is not None
                        else torch.zeros(B, device=device).cpu().numpy()),
                    "hawkes_lam": hawkes_lam.detach().cpu().numpy(),
                    # PER-PAIR λ-TREND (PM 2026-06-01): decline = fraction λ has rolled
                    # OFF the pair's PRE-update recent peak = (λ_peak−λ_carried)/λ_peak.
                    # This is the TREND axis the analyzer needs to bucket DECLINING (vs
                    # bucketing only by z-magnitude). PRE-update ⇒ no re-leak. Recorded as
                    # all-zeros when dynamics is off (so the npz schema stays stable).
                    "decline": (
                        ((_lam_peak_pre.clamp(min=HAWKES_MU_DECOL) - _lam_carried)
                         / (_lam_peak_pre.clamp(min=HAWKES_MU_DECOL) + 1e-6))
                        .clamp(0.0, 1.0).detach().cpu().numpy()
                        if (_lam_peak_pre is not None and _lam_carried is not None)
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    "lam_peak": (
                        _lam_peak_pre.detach().cpu().numpy()
                        if _lam_peak_pre is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    # carried Hawkes λ at this event's time BEFORE the current jump
                    # (μ+(λ_prev−μ)·exp(−β·Δt)); the ABSOLUTE-silence signal DEATH reads
                    # (λ→μ ⇒ dead). Lets the analyzer bucket SILENT by carried λ, not z.
                    "lam_carried": (
                        _lam_carried.detach().cpu().numpy()
                        if _lam_carried is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    # SLOPE of the edit-RATE = rate_fast − rate_slow (fast/slow EWMA of
                    # r=1/Δt), the ONE signed axis (PM 2026-06-01 third re-chốt). The
                    # analyzer buckets RISING(slope≥+margin)/FALLING-active(slope≤−margin &
                    # rate>rate_dead)/DEAD(rate≤rate_dead) on THIS, not on z-magnitude.
                    # rate_fast IS the rate-level the DEATH gate reads. PRE-update ⇒ no leak.
                    "rate_fast": (
                        _rate_fast.detach().cpu().numpy()
                        if _rate_fast is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    "rate_slow": (
                        _rate_slow.detach().cpu().numpy()
                        if _rate_slow is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    "lam_slope": (
                        _lam_slope.detach().cpu().numpy()
                        if _lam_slope is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    # PER-PAIR-RELATIVE channels (PM 2026-06-01 fix). slope_rel = (fast−
                    # slow)/(slow+ε) = % rate change; the analyzer buckets RISING/FALLING on
                    # ±margin_rel of THIS. rate_peak = pair's leaky-max rate (PRE-update);
                    # rate_dead_pp = γ·rate_peak = the per-pair DEAD floor (rate_fast below
                    # it ⇒ DEAD). These REPLACE the abs slope/rate_dead off the coedit scale.
                    "slope_rel": (
                        _slope_rel.detach().cpu().numpy()
                        if _slope_rel is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    "rate_peak": (
                        _rate_peak.detach().cpu().numpy()
                        if _rate_peak is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    "rate_dead_pp": (
                        (self.decol_rate_dead_gamma
                         * _rate_peak.clamp(min=RATE_INIT)).detach().cpu().numpy()
                        if _rate_peak is not None
                        else torch.zeros(B, device=device).cpu().numpy()
                    ),
                    # ever_alive ∈[0,1] PRE-update (read at L949 BEFORE update_batch ⇒
                    # no leak). Logging-only; lets evaluation harness compute the DEATH-before-
                    # alive violation rate = frac(never-alive pairs whose argmax_s_t1_cal
                    # == DEATH). never-alive ⇔ ever_alive_pos ≈ 0. The hier_causal_policy
                    # gate scales the DEATH leaf by ever_alive on s_t1_cal, so ON should
                    # drive this violation rate → 0 vs OFF.
                    "ever_alive": ever_alive_pos.detach().cpu().numpy(),
                    # HƯỚNG A: 6-state argmax (PRE_BIRTH/BIRTH/REINFORCE/DECAY/DORMANT/
                    # DEATH) so evaluation harness can audit the strict-ordered lifecycle dist on
                    # the TRAINED model. -1 (all-zeros sentinel) when strict_ordered_fsm
                    # is off so the npz schema stays stable.
                    "argmax_s_t1_cal6": (
                        s_t1_cal6.argmax(-1).detach().cpu().numpy()
                        if s_t1_cal6 is not None
                        else torch.full((B,), -1, dtype=torch.long).cpu().numpy()),
                }
                if not hasattr(self, "_faith_buf") or self._faith_buf is None:
                    self._faith_buf = {k: [] for k in _rec}
                for k, v in _rec.items():
                    self._faith_buf[k].append(v)

        # ── fsm_arch="v2": persist the soft FSM-head next-state per pair so measure()
        # reads THIS (movable, soft-masked) symbolic state instead of the pinned ECTGv3
        # hard-masked chain. update_symbolic is a no-op store keyed by the same pairs
        # get_batch already registered (called above); v1 never reaches this branch so
        # the canonical store/state is untouched. Detached snapshot (gate is read-only).
        if self.fsm_arch in ("v2", "v3"):
            # Gate measures the CALIBRATED distribution (== s_t1_pos when bias off) so the
            # symbolic state the gate/measure() reads reflects the argmax calibration.
            self.edge_mem.update_symbolic(src, dst, s_t1_cal)

        # ── WC-CONF: WALKED-CHAIN belief b_t + COHERENCE c_t (PM 2026-06-06) ───────
        # Computed here so it sees the FINAL free next-state s_t1_cal (after any hier
        # policy) and uses the PRE-update belief b_{t-1}. Belief read is PRE-update / the
        # write lands POST-scoring (before update_symbolic already happened, but the
        # belief is a SEPARATE store not consumed by scoring). All quantities DETACHED ⇒
        # confidence-only; the prediction VALUE is NOT bent (s_t1_cal/s_t1_pos untouched).
        # Default: cc_* sentinels None so nothing changes when the flag is off.
        cc_coherence = None     # (B,) c_t ∈[0,1]
        cc_weight    = None     # (B,) gradient-selection scale (detached)
        cc_belief    = None     # (B,5) b_t for the probe/dump
        cc_reach     = None     # (B,5) reachable mask from b_{t-1}
        cc_self_consist_loss = None  # scalar aux CE for w_obs (computed OUTSIDE no_grad)
        _cc_bstep_det = None    # (B,5) detached learned-forward belief (for the aux CE)
        _cc_obs_det   = None    # (B,5) detached, ray-projected observation (for the aux CE)
        if self.causal_confidence:
            with torch.no_grad():
                if self._cc_C.device != s_t1_cal.device:
                    self._cc_C = self._cc_C.to(s_t1_cal.device)
                C = self._cc_C                                   # (5,5) admissibility
                # GROUNDED INIT (PM 2026-06-07): for a pair's FIRST appearance in the
                # split, seed the belief at the MODEL-INFERRED phase = softmax(s_t_pos)
                # rather than IDLE. s_t_pos is the StateObserver readout (history-only,
                # built from edge_h_pos.detach() at :1146, PRE the symbolic/store update)
                # — the SAME quantity the AP/score path reads ⇒ NO leak, NO hand phase.
                # Carried beliefs (seen pairs) still override this seed (see peek_belief).
                # When cc_grounded_init is OFF, init_belief=None ⇒ legacy IDLE one-hot
                # (byte-identical). This is a SEED swap only; the walk below is unchanged.
                _cc_init = (torch.softmax(s_t_pos.detach(), dim=-1)
                            if self.cc_grounded_init else None)
                b_prev = self.edge_mem.peek_belief(src, dst, init_belief=_cc_init)  # (B,5) PRE-update
                # obs = s_t_pos (current StateObserver readout) is the filter's MEASUREMENT;
                # it re-enters the belief via the LEARNABLE coupling at step (c) below when
                # cc_self_consist_w>0 (PM 2026-06-07 R2 — FIX-3's full drop stuck belief @
                # IDLE init, legacy diagnostic). When the aux is off, obs is unused (pure FIX-3).
                _B = s_t1_cal.size(0)
                _dev = s_t1_cal.device
                _dt5 = s_t1_cal.dtype
                # ── FIX 1 (REFINE 2026-06-06, PM): BELIEF MOVES BY *LEARNED* DYNAMICS ──
                #     PROJECTED ONTO THE CAUSAL RAY. ────────────────────────────────────
                # The PM retracted the hand-tuned phase-ANCHOR (true_occ/rate/slope ngưỡng):
                # belief must have the SAME flip-engine the FSM learned (knows BIRTH→
                # REINFORCE and back, REINFORCE→DECAY and back per the trained per-pair
                # operator), NOT a heuristic threshold. The ONLY thing causality does to
                # the belief is constrain it to causal-valid transitions (the "ray").
                #
                # MECHANISM:
                #  (a) push b_{t-1} through the LEARNED per-pair transition operator
                #      T_uv (B,5,5) — the SAME operator the FSM head uses to predict the
                #      free next-state (sr_gnn fsm_head TransitionPredictor: T = U@V + g(φ);
                #      next = softmax(s @ T)). So b_step is "where the LEARNED dynamics say
                #      this pair goes next" given its OWN continuous history φ — it knows
                #      the pair's flip timing because g(φ) encodes it. Detached: we READ the
                #      operator's value but no gradient flows back from the belief filter.
                #  (b) PROJECT b_step onto the CAUSAL-VALID ray: ⊙ C_BAND_5 (only |i−j|≤1
                #      transitions survive), renorm. The learned dynamics may favour a jump
                #      (noisy g) → causal projection forbids nhảy-cóc, keeping belief on the
                #      monotone lifecycle ray. b_step is thus "learned timing, on the ray".
                #  (c) light observation nudge (small): a confident current obs can tilt the
                #      belief, but learned dynamics DOMINATE (no hand threshold).
                # b_prev is IDLE-init for unseen pairs (honest pre-birth); the operator then
                # walks it forward event-by-event, so mature states are reached by LEARNED
                # timing, not by a ngưỡng. NON-v3 (T_uv None) → fall back to the pure causal
                # C-step ⊙ obs (guarded; preserves the old behaviour where no operator).
                if T_uv_pos is not None:
                    # (a) learned forward step: next-state logits = b_prev @ T_uv (mirrors
                    #     fsm_head.TransitionPredictor convention next = softmax(s @ T)).
                    _Tdet   = T_uv_pos.detach()                          # (B,5,5)
                    _blog   = torch.bmm(b_prev.unsqueeze(1), _Tdet).squeeze(1)  # (B,5)
                    b_learn = torch.softmax(_blog, dim=-1)              # learned next belief
                else:
                    # no learned operator (non-v3): identity-ish — keep prev belief as the
                    # "dynamics" so the causal projection still applies.
                    b_learn = b_prev
                # (b) project the LEARNED step onto the causal-valid ray (⊙ C_BAND_5 row of
                #     the belief's source mass), renorm. A learned jump to a non-adjacent
                #     state is zeroed here — belief can only flip one rung along the ray.
                _reach_from = torch.einsum("bi,ij->bj", b_prev, C)     # (B,5) admissible set
                _reach_from = (_reach_from > 1e-8).to(_dt5)            # {0,1} reachable mask
                b_step = b_learn * _reach_from                         # kill off-ray mass
                _bs_den = b_step.sum(-1, keepdim=True)
                # if the learned step put ALL mass off-ray (den≈0), fall back to the pure
                # causal C-step from b_prev so belief never collapses to all-zero/NaN.
                b_step = torch.where(
                    _bs_den > 1e-8,
                    b_step / _bs_den.clamp(min=1e-8),
                    torch.einsum("bi,ij->bj", b_prev, C)
                          / torch.einsum("bi,ij->bj", b_prev, C).sum(-1, keepdim=True).clamp(min=1e-8),
                )
                # (c) MEASUREMENT STEP — LEARNABLE observation coupling (PM 2026-06-07 R2).
                #     FIX-3 dropped obs entirely → the filter lost its measurement and the
                #     belief got STUCK at the IDLE init (legacy diagnostic: IDLE 0.87, R²(c~free)
                #     0.908→0.955 WORSE). The filter NEEDS a measurement: the protocol requirement
                #     "no hand" means no hand-anchored phase rate/slope→state, NOT "no obs".
                #     We restore obs-coupling as a proper Bayes-filter correction:
                #       b_t = normalize( b_step^(1-w) ⊙ obs^w )   (geometric / log-linear)
                #     where obs = softmax current StateObserver readout s_t_pos, projected
                #     onto the SAME causal-reachable set so obs cannot teleport the belief.
                #     The mix weight w is NOT a hand constant: w_obs = sigmoid(cc_w_obs_logit),
                #     a LEARNED scalar param trained by the self-consistency CE below. Here
                #     (inside no_grad, detached from predict) we use its CURRENT value to
                #     position the carried belief; the param's GRADIENT comes only from the
                #     out-of-no_grad aux CE (see :after this block). When the aux is off
                #     (cc_w_obs_logit is None) we keep the pure FIX-3 learned-forward filter.
                if self.cc_w_obs_logit is not None:
                    _w_obs = torch.sigmoid(self.cc_w_obs_logit.detach())   # scalar in (0,1)
                    _obs = torch.softmax(s_t_pos.detach(), dim=-1)         # (B,5) measurement
                    _obs = _obs * _reach_from                              # project onto ray
                    _obs = _obs / _obs.sum(-1, keepdim=True).clamp(min=1e-8)
                    # geometric blend in log-space: (1-w)·log b_step + w·log obs, renorm.
                    _blend = ((1.0 - _w_obs) * (b_step.clamp(min=1e-8)).log()
                              + _w_obs * (_obs.clamp(min=1e-8)).log())
                    cc_belief = torch.softmax(_blend, dim=-1)
                    # cache the detached ingredients for the aux CE (recomputed WITH grad
                    # below using the live param — these are all leaves / detached).
                    _cc_bstep_det = b_step.detach()
                    _cc_obs_det   = _obs.detach()
                else:
                    cc_belief = b_step
                # (3) reachable SET membership — STRUCTURAL from C (PM 2026-06-07: ZERO hand
                #     const; the old 0.25·max magnitude threshold is GONE).
                #     reach_mask[j]=1 iff state j is C-ADJACENT to where the belief currently
                #     SITS — pure adjacency, no magnitude cut. "Where the belief sits" = the
                #     belief's argmax rung (its single causal position on the lifecycle ray);
                #     reach = that rung's ROW of C (the |i−j|≤1 band neighbours of the rung).
                #     This is structural: reach_mask = C[argmax(b_prev)] ∈ {0,1}, read straight
                #     off the adjacency matrix. It needs NO threshold because we condition on
                #     the belief's discrete POSITION (argmax), not on a soft mass that could
                #     leak onto forbidden states (the leak was the ONLY reason 0.25·max
                #     existed). For an IDLE belief reach={IDLE,BIRTH}; for DECAY reach=
                #     {REINFORCE,DECAY,DEATH} — the exact one-step band, derived from C alone.
                _pos    = b_prev.argmax(dim=-1)                        # (B,) belief rung
                cc_reach = C[_pos].to(_dt5)                            # (B,5) {0,1} C-adjacency
                # ── FIX 2 (2026-06-06, PM): COHERENCE = REACHABLE-SET MEMBERSHIP ───────
                # The old c = Σ s_t1·b_step / max(b_step) PEAK-NORMALIZED → it penalised an
                # ADMISSIBLE but off-peak transition (legacy diagnostic: BIRTH→REINFORCE
                # is admissible |1−2|=1 but scored c=0.12 because b_step peaked on
                # BIRTH-self). FIX: c_t = Σ_j s_t1[j]·reach_mask[j] (mass of the FREE next-
                # state that lands inside the C-reachable SET), NO division by max. Any
                # admissible transition — including mature off-peak (REINFORCE→DECAY) — is
                # reachable ⇒ c HIGH; a real teleport (never-born→DEATH, nhảy-cóc) lands on
                # an unreachable state ⇒ c LOW. c∈[0,1] (s_t1 is a distribution, reach∈{0,1}).
                cc_coherence = (s_t1_cal.detach() * cc_reach).sum(-1).clamp(0.0, 1.0)  # (B,)
                # (5) gradient-selection weight: soft scale by c_t, HARD-zeroed below thr.
                cc_weight = cc_coherence.clone()
                if self.cc_thr > 0.0:
                    cc_weight = torch.where(
                        cc_coherence >= self.cc_thr,
                        cc_coherence,
                        torch.zeros_like(cc_coherence),
                    )
                # carry the walked-chain belief to the pair's NEXT event (POST-scoring).
                self.edge_mem.update_belief(src, dst, cc_belief)

            # ── SELF-CONSISTENCY AUX LOSS for the LEARNABLE w_obs (OUTSIDE no_grad) ─────
            # The belief block above ran under torch.no_grad() and is detached from predict,
            # so w_obs (= sigmoid(cc_w_obs_logit)) would be a DEAD param if its only use were
            # inside that block. Here we RECOMPUTE the same geometric belief blend WITH grad
            # enabled, using the LIVE param and the DETACHED ingredients (_cc_bstep_det,
            # _cc_obs_det) cached above. The CE target is the FREE next-state argmax
            # (s_t1_cal.detach()) — "the filter's belief should agree with the model's own
            # free read of where the pair is heading." Minimising CE moves w_obs to the
            # measurement weight that makes the carried belief track the realized state
            # (un-sticking the IDLE init of legacy diagnostic).
            #
            # GRADIENT-ISOLATION PROOF (predict Δ=0):
            #   * _cc_bstep_det, _cc_obs_det, and the CE TARGET are ALL .detach()ed.
            #   * the ONLY non-detached tensor in this expression is cc_w_obs_logit.
            #   ⇒ d(cc_self_consist_loss)/d(anything) is NON-zero ONLY for cc_w_obs_logit.
            #   ⇒ NO gradient reaches s_t1_cal / s_t1_pos / existence_decoder / backbone.
            #   The predict scores (pos_logit/score_*) never read cc_w_obs_logit, so adding
            #   this term to `total` leaves the prediction VALUE and its gradients unchanged
            #   (AP Δ=0, backbone 0 link-pred grad). cc_w_obs_logit is the SOLE new gradient
            #   recipient of this loss.
            if (self.cc_w_obs_logit is not None and self.cc_self_consist_w > 0.0
                    and _cc_bstep_det is not None):
                w_obs_g = torch.sigmoid(self.cc_w_obs_logit)            # grad-enabled scalar
                blend_g = ((1.0 - w_obs_g) * _cc_bstep_det.clamp(min=1e-8).log()
                           + w_obs_g * _cc_obs_det.clamp(min=1e-8).log())
                log_belief_g = F.log_softmax(blend_g, dim=-1)          # (B,5)
                sc_target = s_t1_cal.detach().argmax(dim=-1)           # (B,) free next-state
                cc_self_consist_loss = F.nll_loss(log_belief_g, sc_target)

        # Existence decoder (deterministic decode)
        pos_logit = self.existence_decoder(s_t1_pos)

        # Negative pathway (reuse the same echo-augmented neg embedding for parity)
        neg_emb = neg_emb_main
        edge_h_neg = self._build_edge_h(new_h_src, neg_emb)
        # DETERM-ONLY: zero the backbone-derived edge_h for the NEGATIVE scoring path,
        # symmetric with the positive — the negative's FSM state also depends only on
        # its deterministic pair_phi_neg, not on learnable CSN/DRGC embeddings.
        if self.determ_only_backbone:
            edge_h_neg = torch.zeros_like(edge_h_neg)
        # DETACH PROBE: keep the negative SCORING path symmetric with the positive.
        s_t_neg = self.state_observer(edge_h_neg, detach_h=_detach_score)
        # ── ANTI-LEAK FIX (2026-05-31, ml): the NEGATIVE candidate (src, neg_dst) now
        # gets its OWN ever_alive AND (for v3) its OWN per-pair operator φ, read
        # READ-ONLY from the stores so the negative is not mutated into them. Before
        # this fix the negative was forced ever_alive=0 AND pair_phi=None, while the
        # positive carried real ever_alive>0 + a trained per-pair operator g(φ). That
        # asymmetry IS a label shortcut: "has history / has an operator" ⇔ "real edge",
        # which the high-capacity v3 g(φ) learned into a PERFECT separator (smoke
        # 5448819: trans_ap=ind_ap=1.0, std 0). A random negative that happens to be a
        # recurring pair now correctly scores high; a first-occurrence positive scores
        # like a fresh edge — the legitimate, non-leaky behavior. φ uses the SAME
        # PRE-update channels as the positive (peek_batch = stored state, init for
        # unseen pairs) so pos and neg are built identically. GATED to fsm_arch="v3"
        # ONLY — v1/v2 keep the original ever_alive_neg=0 / no-pair_phi negative path
        # (byte-identical; the v3 g(φ) operator is what turned the asymmetry into a
        # PERFECT separator, so the symmetry fix is needed only where g(φ) exists).
        if self.fsm_arch == "v3":
            ever_alive_neg = self.ever_alive.peek(src, neg_dst)
            with torch.no_grad():
                neg_st = self.edge_mem.peek_batch(src, neg_dst)   # (B,12) read-only
                # Per-pair-relative z for the NEGATIVE candidate, built IDENTICALLY to the
                # positive (same PRE-update channels via peek, same count guard) so pos/neg
                # operators stay symmetric — no "pos has z, neg has none" shortcut.
                _mean_pre_n = neg_st[:, 7]
                _std_pre_n  = neg_st[:, 8].clamp(min=1e-6).sqrt()
                _z_pre_n    = (dt_src.float() - _mean_pre_n) / (_std_pre_n + 1e-6)
                _z_pre_n    = _z_pre_n * (neg_st[:, 9] >= 2.0).float()
                pair_phi_neg = torch.stack([
                    neg_st[:, 6],                              # Hawkes λ  (pre-update)
                    torch.log1p(neg_st[:, 7].clamp(min=0)),    # log mean_dt
                    torch.log1p(neg_st[:, 8].clamp(min=0)),    # log var_dt
                    neg_st[:, 5],                              # recurrence EWMA
                    torch.log1p(dt_src.float().clamp(min=0)),  # staleness (src-anchored)
                    ever_alive_neg,                            # ever_alive ∈[0,1]
                    _z_pre_n.clamp(-5.0, 5.0),                 # per-pair-relative z (NEW)
                ], dim=-1)
            _h_score_neg = edge_h_neg if not _detach_score else edge_h_neg.detach()
            trans_logits_neg = self.transition_predictor(
                _h_score_neg, s_t_neg, pair_phi=pair_phi_neg)
        else:
            # v1/v2 canonical negative: ever_alive=0 ("fresh" edge), no pair_phi.
            ever_alive_neg = torch.zeros_like(ever_alive_pos)
            _h_score_neg = edge_h_neg if not _detach_score else edge_h_neg.detach()
            trans_logits_neg = self.transition_predictor(_h_score_neg, s_t_neg)
        mask_neg = self.lifecycle_mask.get_mask_from_state(s_t_neg)
        s_t1_neg = trans_logits_neg + (mask_neg + 1e-6).log()
        s_t1_neg = self.lifecycle_mask.apply_ever_alive_gate(s_t1_neg, ever_alive_neg)
        s_t1_neg = torch.softmax(s_t1_neg, dim=-1)
        neg_logit = self.existence_decoder(s_t1_neg)

        # Update ever_alive (after prediction — anti-leakage)
        alive_now = s_t1_pos[:, BIRTH] + s_t1_pos[:, REINFORCE]
        self.ever_alive.update(src, dst, alive_now)

        # ── Echo update (AFTER prediction — anti-leakage; #3 enable_echo only) ──
        # Faithful to sr_gnn_v3.py:741-752 (single-scale, lean config: no router so
        # ALL edges update, bidirectional). v3.3's DRGC_v2 has no resonance head, so
        # the resonance coefficient R is derived from the hawkes intensity via the
        # same normalization used in compute_compliance (lam/(lam+1) ∈ (0,1)). This
        # is the one documented deviation from v3.1 (which used DRGC's R_uv head).
        if self.echo is not None:
            with torch.no_grad():
                lam = hawkes_lam.detach()
                lam_norm = lam / (lam.mean() + 1e-6)
                R_echo = (lam_norm / (lam_norm + 1.0)).clamp(0, 1)  # (B,)
                self.echo.update(src, dst, R_echo,
                                 new_h_src.detach(), new_h_dst.detach(), t,
                                 bidirectional=True)

        # Memory update (after scoring)
        self.node_mem.set(all_idx, parsed_h)
        self.node_mem.set(src, new_h_src)
        self.node_mem.set(dst, new_h_dst)
        self.node_mem.update_time(torch.cat([src, dst]), torch.cat([t, t]))

        # ── Loss ──
        # 1. Edge prediction with LFG. The head ablation toggle selects which head
        #    carries it:
        #      enable_main_predictor=False → existence_decoder logits (all-detached)
        #                                     → backbone gets ZERO link-pred gradient.
        #                                     CANONICAL default (empirically better).
        #      enable_main_predictor=True  → MAIN (non-detached) head → backbone IS
        #                                     trained end-to-end by link prediction.
        #                                     ABLATION arm (original "P0 fix").
        labels = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])
        if self.enable_main_predictor:
            score_pos_logit = main_pos_logit
            score_neg_logit = main_neg_logit
        else:
            score_pos_logit = pos_logit
            score_neg_logit = neg_logit
        all_logits = torch.cat([score_pos_logit, score_neg_logit])
        loss_per_event = F.binary_cross_entropy_with_logits(
            all_logits, labels, reduction='none'
        )

        # Compliance for pos and neg
        compliance_pos = compute_compliance(s_t_pos, s_t1_pos, ever_alive_pos, hawkes_lam)
        # For neg, use uniform compliance = 1 (don't filter negatives)
        compliance_neg = torch.ones(B, device=device)
        compliance = torch.cat([compliance_pos, compliance_neg])

        # LFG warmup: in first warmup epochs, no gating
        if self.current_epoch < self.lfg_warmup_epochs:
            lfg_weight = 0.0
        else:
            ramp = min(1.0, (self.current_epoch - self.lfg_warmup_epochs) / 3.0)
            lfg_weight = ramp

        # #1 LFG toggle: when disabled, force uniform weight=1 on every event so the
        # backbone/readout sees the unweighted BCE (lets evaluation harness isolate LFG's
        # standalone effect). Canonical (enable_lfg=True) keeps the compliance ramp.
        if not self.enable_lfg:
            lfg_weight = 0.0

        compliance_effective = (
            (1 - lfg_weight) * torch.ones_like(compliance)
            + lfg_weight * compliance
        )

        # ── LFG HARD causal-rule GRADIENT MASK (lfg_mode="hard", the intended LFG) ──
        # Build a per-event gradient weight m_e from the FIXED causal rule matrix C:
        #   v_e = C[argmax(s_t), argmax(s_{t+1})] ∈ {0,1}  (admissible / impossible)
        #   m_e = 1.0          if admissible
        #   m_e = compliance_floor (= 0.0 under design="correct") if impossible
        # m_e is DETACHED → it can ONLY mask/attenuate the link-prediction gradient of
        # causally-incoherent positive events; it adds NO gradient of its own and does
        # NOT change the prediction VALUE (pos_score/neg_score are untouched). Negatives
        # are never gated (weight 1). When lfg_weight is in warmup (==0) the gate is
        # also ramped off (no masking before LFG kicks in), consistent with the soft LFG.
        if self.lfg_mode == "hard":
            with torch.no_grad():
                v_pos = compute_causal_validity(s_t_pos, s_t1_pos, self.causal_rule)  # (B,)
                gate_pos = torch.where(
                    v_pos > 0.5,
                    torch.ones_like(v_pos),
                    torch.full_like(v_pos, float(self.compliance_floor)),
                )
                gate_neg = torch.ones(B, device=device)
                hard_gate = torch.cat([gate_pos, gate_neg])
                # Ramp the hard gate in with the same warmup/ramp as the soft LFG so the
                # backbone is not starved during warmup; full hard gate once ramped.
                hard_mask = (1.0 - lfg_weight) * torch.ones_like(hard_gate) + lfg_weight * hard_gate
            # detach() is defensive (hard_mask already built under no_grad).
            pred_weight = (compliance_effective * hard_mask).detach()
        else:
            pred_weight = compliance_effective
        pred_loss = (pred_weight * loss_per_event).mean()

        # 2. Mask violation penalty: P(s_{t+1} = DEATH | never_alive) should be ≈ 0
        violation_pos = s_t1_pos[:, DEATH] * (1 - ever_alive_pos)
        violation_loss = violation_pos.mean()

        # 3. FSM-stream supervision: keep the symbolic existence head meaningful as an
        #    INTERPRETATION of the same link-prediction target. pos_logit/neg_logit are
        #    built from state_observer/transition_predictor, which read h.detach(), so
        #    this BCE trains ONLY the FSM head params — zero gradient to the backbone.
        #    In the CANONICAL detached arm (enable_main_predictor=False, default)
        #    pred_loss IS this same existence BCE, so a separate fsm_loss term would
        #    double-count it; we drop it (set 0) so the canonical arm reproduces the
        #    exact intended objective (BCE on the detached readout).
        fsm_logits = torch.cat([pos_logit, neg_logit])
        if self.enable_main_predictor:
            fsm_loss = F.binary_cross_entropy_with_logits(fsm_logits, labels)
            fsm_loss_term = self.lambda_fsm * fsm_loss
        else:
            # FSM head still gets trained — by pred_loss, which in this arm routes
            # through the same (detached) existence logits. No extra term.
            fsm_loss = F.binary_cross_entropy_with_logits(fsm_logits, labels).detach()
            fsm_loss_term = 0.0

        # #2b Symbolic-state entropy regularizer (entropy_reg_weight>0 only).
        # H = Shannon entropy of the batch-mean symbolic next-state distribution.
        # We SUBTRACT entropy_reg_weight*H from the total loss (i.e. ADD the
        # NEGATIVE-entropy term), so minimizing the loss MAXIMIZES H → pushes the
        # symbolic stream AWAY from the ~0.95-IDLE collapse toward a spread-out
        # state distribution. state_dist is post-softmax (already a valid simplex).
        if self.entropy_reg_weight > 0.0:
            state_mean = s_t1_pos.mean(0)  # (5,), valid distribution
            state_entropy = -(state_mean * (state_mean + 1e-8).log()).sum()
            entropy_term = -self.entropy_reg_weight * state_entropy  # negative → maximize H
        else:
            entropy_term = 0.0

        # 4. Transition-CE term — REVIVE lambda_trans to DE-COLLAPSE the FSM (KEY Tier-1
        #    lever). lambda_trans was declared but DEAD (never in the loss). We now add a
        #    transition supervision = KL( predicted s_{t+1} ‖ weak heuristic target ),
        #    where the target is ECTGv3's detached Hawkes/recurrence multi-signal state
        #    distribution (heuristic_target, B×5). It trains ONLY the FSM head:
        #      - s_t1_pos is built from trans_logits_pos (edge_h.detach()) and s_t_pos
        #        (StateObserver detaches h internally) → ZERO backbone gradient.
        #      - heuristic_target is detached (it is the supervision label).
        #    This pulls the next-state distribution toward the lifecycle phases implied
        #    by the temporal signals, counteracting the ~0.95-IDLE collapse. Template:
        #    experiments/models/sr_gnn_v3.py:763-776 (loss_heur). Gated by use_trans_loss
        #    so it is DEAD in every config except design="correct" (back-compat).
        # ── De-collapse target rebuild (decollapse_target only) ──────────────────
        # ROOT CAUSE (diagnosed 2026-05-30, CPU evidence): the canonical heuristic
        # target uses is_first = (recur < 0.05). But `recur` is the POST-UPDATE EWMA
        # which jumps 0→0.3 on an edge's FIRST event (r_new = 0.3 + 0.7*0), so is_first
        # is ESSENTIALLY NEVER true → BIRTH is never targeted → from IDLE the only
        # legal target (VALID allows {IDLE,BIRTH}) is IDLE → permanent IDLE lock; and
        # the continuous VALID chain has DEATH absorbing, so any edge that does leave
        # IDLE drains to DEATH (H→0). We rebuild the target with a WORKING is_first
        # from n_obs (state dim 9, the post-update observation count: ==1 on the true
        # first event) and add class balancing so DEATH/IDLE do not dominate.
        if self.decollapse_target:
            with torch.no_grad():
                hawk_e  = new_est[:, 6]
                # ── PER-PAIR-RELATIVE z (fsm_arch="v3"): PRE-update Welford + count guard ─
                # v1/v2 (byte-identical): z from POST-update new_est[:,7:8] mean/var. But
                # the POST-update Welford has ALREADY folded the current Δt into its mean,
                # so z self-SHRINKS (CPU probe: a genuine 20x gap relative to a pair's tight
                # history gives z_POST≈+2.45, capped, vs the true per-pair deviation z_PRE≫1)
                # → DECAY/DEATH structurally under-fire → REINFORCE-heavy target on the
                # repeat-heavy coedit stream. v3 instead measures the current gap against the
                # pair's PRE-update mean/var (edge_st[:,7:8], the stats BEFORE this event):
                #   z = (dt_src − μ_pair^pre) / (σ_pair^pre + ε).
                # "Is THIS gap long FOR THIS PAIR" — heterogeneity nucleates from each pair's
                # OWN history, not a global gap threshold. PRE-update also means z does NOT
                # see the current event → no re-leak (pos_logit/target independent of the
                # event being scored; same invariant the anti-leak φ fix restored).
                # MIN-COUNT GUARD (PM item 3): a pair with <2 prior observations has a
                # degenerate σ (Welford var=0 at n≤1) → z is garbage. For those pairs we
                # ZERO the late/dead drive so they fall to BIRTH/REINFORCE only and a single-
                # shot pair cannot inject spurious DECAY/DEATH noise. Only pairs with enough
                # history can be labelled dormant/dead. n_obs = edge_st[:,9] (PRE-update count).
                if self.fsm_arch == "v3":
                    mean_dt = edge_st[:, 7]
                    var_dt  = edge_st[:, 8]
                    n_prior = edge_st[:, 9]
                    std_dt  = var_dt.clamp(min=1e-6).sqrt()
                    z_e     = (dt_src.float() - mean_dt) / (std_dt + 1e-6)
                    has_hist = (n_prior >= 2.0).float()   # σ trustworthy only with ≥2 prior
                    z_e      = z_e * has_hist              # n_prior<2 → z=0 → REINFORCE band
                    # ── DYNAMICS axis (PM 2026-06-01: ĐỘNG LỰC = TREND of λ, NOT gap len) ──
                    # PM redefinition: DECAY is NOT "long absolute gap". It is the
                    # TREND/derivative of the pair's interaction INTENSITY: the pair WAS
                    # reinforcing (λ high near its OWN recent peak), then the intensity
                    # KHỰNG lại rồi GIẢM DẦN — λ rolls OFF that recent peak — while the
                    # pair is STILL alive (not yet silent-long enough to be DEATH).
                    #   decline = (λ_peak^pre − λ_carried) / (λ_peak^pre + ε) ∈ [0,1]
                    # λ_peak^pre = the pair's leaky running peak Hawkes λ (peek_lam_peak,
                    # PRE-update); λ_carried = μ + (λ_prev−μ)·exp(−β·Δt) = the Hawkes
                    # recursion's pre-jump carried value at this event's time (sr_gnn_v3
                    # update_multisignal L183). BOTH derived from the store's OWN Hawkes
                    # law — no fabricated channel — and BOTH PRE-update (no re-leak).
                    # is_decline fires when λ has dropped a meaningful FRACTION off the
                    # pair's recent peak. Count-guarded (needs a peak to fall from).
                    #
                    # DEATH stays absolute: a very long silence (z extreme) = "im lặng
                    # kéo dài, không còn gì" (absorbing). The temporal precedence
                    # REINFORCE→DECAY→DEATH is enforced below: DEATH is gated by decline
                    # (only a pair that has already started declining can die) and DECAY
                    # is suppressed once the pair is fully dead (high z).
                    # ── SLOPE-OF-RATE axis (PM 2026-06-01 THIRD re-chốt) ──────────────
                    # All three active states live on ONE signed axis = the slope of the
                    # edit-RATE r=1/Δt. slope = rate_fast − rate_slow (fast/slow EWMA of
                    # the rate), both PRE-update. The SIGN of the slope is the
                    # discriminator REINFORCE↔DECAY:
                    #   slope ≥ +margin                 → rate RISING  → REINFORCE
                    #   slope ≤ −margin & rate>rate_dead → rate FALLING-but-alive → DECAY
                    #   rate_fast ≤ rate_dead (sustained, after decay) → DEATH
                    # This FIXES the two prior failures: REINFORCE is NO LONGER "absolute-
                    # high λ / active" (which left DECAY no room and let DEATH swallow it,
                    # legacy diagnostic DECLINING→95.7% DEATH); it is now strictly "rate rising".
                    if _lam_slope is not None and _rate_level is not None:
                        _slope   = _lam_slope
                        _rate    = _rate_level
                        # decline (kept for dump/analyzer continuity) = how far the carried
                        # λ has rolled off the pair's own recent peak.
                        if _lam_peak_pre is not None and _lam_carried is not None:
                            _peak = _lam_peak_pre.clamp(min=HAWKES_MU_DECOL)
                            decline = ((_peak - _lam_carried) / (_peak + 1e-6)).clamp(0.0, 1.0)
                        else:
                            decline = torch.zeros_like(z_e)
                    else:  # safety (dynamics on but trend tensors missing) → flat slope
                        _slope   = torch.zeros_like(z_e)
                        _rate    = torch.full_like(z_e, 1.0)  # > rate_dead ⇒ "alive" default
                        decline  = torch.zeros_like(z_e)
                    # ── PER-PAIR-RELATIVE vs ABSOLUTE gating (PM 2026-06-01 fix) ──────────
                    if self.decol_rate_relative and _slope_rel is not None \
                            and _rate_peak is not None:
                        # RELATIVE: slope_rel = (fast−slow)/(slow+ε) is the % rate change;
                        # the ±margin_rel band is scale-free so it OPENS on coedit (abs Δ
                        # is ~0.01-0.05 there, never crossing the old abs 0.05). dead floor
                        # is the pair's OWN peak: γ·rate_peak_pair. is_alive/dead read the
                        # rate_fast level vs that per-pair floor.
                        _srel       = _slope_rel
                        _dead_floor = self.decol_rate_dead_gamma * _rate_peak.clamp(
                            min=RATE_INIT)
                        is_rising  = torch.sigmoid(
                            self.decol_slope_scale * (_srel - self.decol_margin_rel)) * has_hist
                        is_falling = torch.sigmoid(
                            self.decol_slope_scale * (-self.decol_margin_rel - _srel)) * has_hist
                        is_alive_rate = torch.sigmoid(
                            self.decol_rate_dead_scale * (_rate - _dead_floor)) * has_hist
                        is_dead_rate  = torch.sigmoid(
                            self.decol_rate_dead_scale * (_dead_floor - _rate)) * has_hist
                        # alias so the legacy branch's named gates below are not recomputed
                        _rel_gates_done = True
                    else:
                        _srel = None
                        _dead_floor = None
                        _rel_gates_done = False
                    # is_rising  : slope clearly positive  (rate accelerating) → REINFORCE
                    # is_falling : slope clearly negative  (rate decelerating) → DECAY
                    # Soft, complementary bands around 0 with a dead-zone of width 2·margin.
                    # (ABSOLUTE fallback — only when decol_rate_relative is off.)
                    if not _rel_gates_done:
                        is_rising  = torch.sigmoid(
                            self.decol_slope_scale * (_slope - self.decol_slope_margin)) * has_hist
                        is_falling = torch.sigmoid(
                            self.decol_slope_scale * (-self.decol_slope_margin - _slope)) * has_hist
                    # rate-level gates (ABSOLUTE fallback, from the rate_fast level):
                    # is_alive_rate = the pair is still firing above the dead floor;
                    # is_dead_rate = the rate has collapsed to ~0. ONLY when the relative
                    # gates were not already computed (decol_rate_relative off) — otherwise
                    # the per-pair γ·peak floor above stands.
                    if not _rel_gates_done:
                        is_alive_rate = torch.sigmoid(
                            self.decol_rate_dead_scale * (_rate - self.decol_rate_dead)) * has_hist
                        is_dead_rate  = torch.sigmoid(
                            self.decol_rate_dead_scale * (self.decol_rate_dead - _rate)) * has_hist
                    # is_decline / is_silent retained as NAMED aliases for downstream code
                    # and the dump (decline = level-off-peak; silent = rate at/below dead).
                    is_decline = is_falling * is_alive_rate
                    is_silent  = is_dead_rate
                else:
                    mean_dt = new_est[:, 7]
                    var_dt  = new_est[:, 8]
                    std_dt  = var_dt.clamp(min=1e-6).sqrt()
                    z_e     = (dt_src.float() - mean_dt) / (std_dt + 1e-6)
                # ── is_first from COLLISION-IMMUNE true occurrence index ──────────
                # ROOT CAUSE of the relocated pure-BIRTH collapse (Tier-1b, H≈0.49):
                # the previous is_first = (n_obs<=1) used the Welford count, which is
                # corrupted by intra-batch read-before-write — at batch=500 on coedit
                # ~75% of events collide with another occurrence of the same pair in
                # the SAME batch, so they all read pre-batch state (n_obs<=1) and look
                # "first". MEASURED (CPU, a temporary target diagnostic): the (n_obs<=1) target's
                # own argmax distribution is 84.7% BIRTH / H=0.484 — the model faithfully
                # matched a target that was ITSELF 85% BIRTH. The heuristic was the
                # collapse, not the weights. true_occ (edge_mem.get_true_occ) advances
                # once per event in stream order → is_first fires on the genuine first
                # occurrence only (18.5% of coedit events), matching the data.
                is_first  = (true_occ <= 1.0).float()
                # Recurring (non-first) events drive the lifecycle by the temporal
                # signal. Thresholds RECALIBRATED to coedit's measured ranges (CPU
                # probe: hawkes median≈1.10, z p90≈1.0, max≈1.63 — the old 1.0/1.5/3.0
                # cutoffs left REINFORCE/DECAY/DEATH structurally starved → IDLE pin).
                is_active = torch.sigmoid(4.0 * (hawk_e - self.decol_hawkes_thr))
                is_late   = torch.sigmoid(4.0 * (z_e - self.decol_late_thr))
                is_dead   = torch.sigmoid(4.0 * (z_e - self.decol_dead_thr))
                # ── TREND-DRIVEN DECAY/DEATH (v3 + decol_use_dynamics) ────────────────
                # PM 2026-06-01: the OLD design tied BOTH DECAY and DEATH to the SAME
                # absolute z (gap length), so on coedit only ~17/12000 events crossed
                # z≥0.7 and most of those had z high enough that DEATH won → the model
                # never saw a clean DECAY band (legacy diagnostic: LATE bucket argmax 100%
                # DEATH). The fix makes DECAY a TREND state and DEATH an absolute-silence
                # state, with REINFORCE→DECAY→DEATH precedence:
                #   DECAY  := λ rolling off the pair's recent peak (is_decline) AND the
                #             pair is STILL ALIVE (gap not yet in the extreme-silence
                #             band) → declining-but-alive.
                #   DEATH  := extreme absolute silence (high z, is_dead) — only reachable
                #             once the pair has STARTED declining (gated by is_decline)
                #             so we never jump REINFORCE→DEATH.
                # is_late (the DECAY DRIVE) is REPLACED by the trend; the raw-z is_late is
                # kept ONLY as a weak floor so a pair whose gap genuinely stretches but
                # whose λ-peak we under-estimated still registers some DECAY.
                if self.fsm_arch == "v3" and self.decol_use_dynamics:
                    # ── ONE-AXIS SLOPE DRIVES (PM 2026-06-01 third re-chốt) ──────────
                    # REINFORCE = rate RISING (slope ≥ +margin) AND still alive.
                    #   reinf_drive feeds the REINFORCE logit POSITIVELY (it is no longer
                    #   the leftover after suppression — it is its OWN rising signal).
                    reinf_drive = is_rising * is_alive_rate
                    # DECAY = rate FALLING-but-ALIVE (slope ≤ −margin AND carried λ still
                    #   above the dead floor). NOT gap length, NOT |λ|. This is the
                    #   MANDATORY pre-death phase.
                    decay_drive = is_falling * is_alive_rate
                    # DEATH = rate ≈ 0 (carried λ ≤ rate_dead) AND the pair has ALREADY
                    #   started declining (temporal precedence DECAY→DEATH, CAUSAL_RULE_
                    #   MATRIX C — never REINFORCE→DEATH directly). started_decline is the
                    #   falling-or-already-low signal so a pair that goes silent must have
                    #   passed through DECAY first; a still-RISING pair (is_rising high)
                    #   that suddenly reads low rate is NOT yet labeled dead.
                    started_decline = torch.clamp(is_falling + is_dead_rate * (1.0 - is_rising),
                                                  0.0, 1.0)
                    death_drive = is_dead_rate * started_decline
                    is_late = torch.clamp(decay_drive, 0.0, 1.0)   # "in the DECAY band"
                    is_dead = torch.clamp(death_drive, 0.0, 1.0)
                    # REINFORCE is suppressed exactly when the pair is NOT rising — i.e.
                    # it is decaying OR dead. Using (1−reinf_drive) keeps the REINFORCE
                    # logit positive ONLY on rising-and-alive events, so a dying pair
                    # (reinf_drive≈0) loses its REINFORCE mass to DECAY/DEATH.
                    reinf_suppress = torch.clamp(1.0 - reinf_drive, 0.0, 1.0)
                else:
                    reinf_suppress = is_late   # v1/v2/no-dynamics: legacy (DECAY only)
                rec = (1.0 - is_first)
                NEG = -4.0   # low floor so masked-out lifecycle states stay ~0
                tl = torch.full((B, 5), NEG, device=device)
                # First event → BIRTH; IDLE only weakly available on a (rare) first
                # event so an unseen edge can stay dormant. Recurring events advance the
                # lifecycle: REINFORCE while active, DECAY when inter-event gap grows
                # (z high), DEATH when the gap is extreme. IDLE is SUPPRESSED for active
                # recurring edges — the old (1-is_active)*(1-is_first) IDLE logit pinned
                # 75% of mass in IDLE even with is_first fixed (CPU-verified).
                tl[:, IDLE]      = torch.where(is_first > 0.5,
                                               torch.full_like(hawk_e, -1.0),
                                               torch.full_like(hawk_e, NEG))
                tl[:, BIRTH]     = is_first * 3.0 - 2.0
                tl[:, REINFORCE] = torch.where(rec > 0.5,
                                               (1.0 - reinf_suppress) * 2.0,
                                               torch.full_like(hawk_e, NEG))
                tl[:, DECAY]     = torch.where(rec > 0.5,
                                               is_late * (1.0 - is_dead) * 2.5 - 0.5,
                                               torch.full_like(hawk_e, NEG))
                tl[:, DEATH]     = torch.where(rec > 0.5,
                                               is_dead * 2.5 - 1.0,
                                               torch.full_like(hawk_e, NEG))
                decol_target = torch.softmax(tl, dim=-1)  # (B,5), pre-projection
                # ── PER-CUR-STATE VALID-FEASIBLE PROJECTION (Tier-1c fix, 2026-05-31) ──
                # The gate measures new_dist = softmax(masked_logits), where the VALID
                # mask zeroes any next-state NOT reachable in ONE step from the current
                # state (VALID_TRANSITIONS, sr_gnn.py:27). The edge-trans CE is now
                # supervised on that SAME masked distribution (see lambda_edge_trans
                # block below), so the target MUST live on the same per-cur-state valid
                # support — otherwise KL(masked_pred ‖ full_target) chases mass on
                # one-step-UNREACHABLE states (e.g. demanding REINFORCE/DECAY/DEATH from
                # a fresh IDLE/BIRTH edge), which the mask zeroes in the prediction → an
                # unsatisfiable objective that pins the measured argmax at the BIRTH-
                # reachable frontier (the ~0.844 BIRTH the gate saw). We therefore
                # PROJECT decol_target onto the valid support of the CURRENT state and
                # renormalize. cur_idx = argmax of the PRE-update state logits edge_st[:,:5]
                # — IDENTICAL to ECTGv3's cur_idx (softmax(edge_states[:,:5]).argmax,
                # sr_gnn_v3.py:204-205; argmax(softmax(x))==argmax(x)) — so this projects
                # onto EXACTLY the support the prediction's mask permits. The meaningful
                # REINFORCE/DECAY/DEATH spread now EMERGES via state CHAINING as edges
                # mature across events (cur=BIRTH→REINFORCE reachable, etc.), it is NOT
                # demanded one-step from a fresh edge.
                # The per-cur-state VALID projection is a v1-ONLY workaround for the
                # HARD -1e9 mask. Under fsm_arch="v2" the supervised quantity is the soft
                # FSM head (finite sigmoid(prior+delta) penalty), which CAN place mass on
                # any state in one step, so the full meaningful target (BIRTH .185 /
                # REINFORCE .585 / DECAY .201 / DEATH .029, H≈1.05) is directly
                # representable and MOVABLE — projecting onto hard valid support would
                # re-impose the very constraint v2 removes. So skip the projection in v2.
                if self.fsm_arch == "v1":
                    _cur_idx = edge_st[:, :5].argmax(-1)               # (B,) pre-update state
                    _vmask   = VALID_TRANSITIONS[_cur_idx.cpu()].to(device)  # (B,5) bool
                    decol_target = decol_target * _vmask.float()
                    decol_target = decol_target / decol_target.sum(-1, keepdim=True).clamp(min=1e-8)
                decol_target = decol_target.detach()  # (B,5), v1: VALID-feasible; v2: full
                # EVAL-ONLY: record the TARGET argmax alongside the head dump so the
                # analyzer can compare the supervision target (what the de-collapse logic
                # ASKS for) against the head's committed state — the DECLINING-target rate
                # is the ceiling the head can reach. Buffer already exists (created in the
                # faithfulness dump block above, same forward); pure logging, no grad.
                if getattr(self, "_dump_faithfulness", None) is not None \
                        and getattr(self, "_faith_buf", None) is not None:
                    self._faith_buf.setdefault("argmax_target", [])
                    self._faith_buf["argmax_target"].append(
                        decol_target.argmax(-1).detach().cpu().numpy())
                # MEASURED target self-distribution on coedit train (CPU replay of THIS
                # exact logic, batch=500, 2026-05-30): argmax IDLE 0 / BIRTH .185 /
                # REINFORCE .585 / DECAY .201 / DEATH .029, H=1.051 (soft-mean H=1.205).
                # MEANINGFUL: BIRTH = true first-occurrence rate (18.5% of coedit
                # events); the bulk is REINFORCE (81.5% repeat events / 92.5% repeat
                # mass); DECAY/DEATH from growing inter-event z. Driven by the true-rank
                # + z signal, NOT entropy fiat. (Compare the OLD n_obs<=1 target:
                # 84.7% BIRTH / H=0.484 — the collapse the model faithfully matched.)
        else:
            decol_target = heuristic_target

        # ── FSM-stream transition-CE (use_trans_loss): KL(pred s_{t+1} ‖ target) ──
        # FSM-head grad only (s_t1_pos built from detached h; target detached).
        if self.use_trans_loss:
            log_pred_next = (s_t1_pos.clamp(min=1e-8)).log()          # (B,5), FSM-head grad only
            trans_loss = F.kl_div(
                log_pred_next, decol_target.detach(), reduction="batchmean"
            )
            trans_loss_term = self.lambda_trans * trans_loss
        else:
            trans_loss = torch.tensor(0.0, device=device)
            trans_loss_term = 0.0

        # ── Edge-state transition-CE on the MASKED next-state dist (lambda_edge_trans>0) ──
        # KEY LEVER + TIER-1c FIX (2026-05-31): the GATE measures the CONTINUOUS ECTGv3
        # edge-state argmax = edge_mem._state_table[:,:5], which is new_dist =
        # softmax(student_logits.masked_fill(~VALID,-1e9)) stored DETACHED (the VALID-
        # MASKED softmax; sr_gnn_v3.py:206-208, run_v3_3_benchmark.py:24-29). The PREVIOUS
        # CE supervised log_softmax(student_logits) — the UNMASKED distribution — a
        # DIFFERENT tensor object: the optimizer moved the unmasked head while the mask
        # diverted the MEASURED argmax, pinning it at the BIRTH-reachable frontier
        # (~0.844 BIRTH) despite a corrected H=1.05 target. We now supervise the SAME
        # tensor the gate argmaxes: new_est[:, :5] IS that masked new_dist (the live,
        # grad-bearing slice; the store detaches a COPY in update_batch, sr_gnn_v3.py:114),
        # so the CE input is OBJECT-IDENTICAL to the measured quantity. KL(masked_pred ‖
        # valid-feasible target) → the optimizer directly moves the measured argmax.
        # Grad flows: new_dist→masked_logits→student_logits→trans_net (ECTGv3 backbone).
        # In the DETACHED head arm (fsm_decouple, default enable_main_predictor=False)
        # link-pred BCE gives the backbone no gradient, so this is the FSM/edge-state
        # supervision channel into ECTG — by design.
        if self.lambda_edge_trans > 0.0:
            if self.fsm_arch in ("v2", "v3"):
                # REDESIGN: supervise the SEPARATE soft-masked FSM head s_{t+1} (the
                # quantity v2's gate measures via update_symbolic above), NOT the pinned
                # ECTGv3 hard-masked chain. s_t1_pos = softmax(trans_logits + log soft-
                # mask); its finite sigmoid(prior+delta) penalty lets the KL pull mass
                # onto ANY state → the measured argmax distribution is MOVABLE (CPU-
                # proven). Grad flows into transition_predictor + lifecycle_mask.delta
                # (FSM-head params; backbone untouched — the head reads edge_h.detach()),
                # so de-collapse is decoupled from the continuous backbone by design.
                # Supervise the CALIBRATED distribution (== s_t1_pos when bias off). The
                # bias is learnable, so the KL trains BOTH transition_predictor/mask AND
                # the per-class bias to match the target on DECAY/DEATH.
                ce_input = s_t1_cal
            else:
                # v1 (canonical): supervise the VALID-MASKED ECTGv3 next-dist — the EXACT
                # tensor the v1 gate argmaxes (new_est[:,:5]); object-identical to measured.
                ce_input = new_est[:, :5]
            log_masked  = ce_input.clamp(min=1e-8).log()             # log-prob input to KL
            # ── CLASS-BALANCED edge-trans CE (decol_class_balance, v3) ────────────────
            # PM item 2a + diagnosis (CPU a temporary calibration diagnostic): the head will not commit to
            # the rare/intermediate DECAY (and DEATH) classes under an unweighted batchmean
            # KL — REINFORCE dominates the gradient by sheer frequency. We reweight each
            # event's KL by the INVERSE frequency of its TARGET argmax class (computed per
            # batch, detached), normalized to mean 1 so the overall KL scale — hence the
            # lambda_edge_trans balance vs pred_loss — is preserved. This makes the head
            # DARE to place DECAY/DEATH mass where the target says so, WITHOUT changing the
            # target or the prediction value (AP untouched: the existence_decoder readout
            # never sees this weight). Off ⇒ plain batchmean (byte-identical).
            # ── WC-CONF GRADIENT-SELECTION (causal_confidence) ────────────────────
            # The FSM-block CE is the loss whose gradient trains the FSM head. WC-CONF
            # scales EACH event's CE by its coherence weight c_t (detached), so events
            # whose FREE prediction lands on a causally-UNREACHABLE state contribute
            # LESS (or zero, below cc_thr) gradient → the model does NOT learn from
            # causally-incoherent trajectories. This CHOOSES which gradient is learned;
            # it does NOT touch the prediction VALUE (s_t1_cal/s_t1_pos unchanged) and
            # does NOT touch the AP path (existence_decoder reads s_t1_pos). Normalized
            # mean→1 so the lambda_edge_trans balance vs pred_loss is preserved.
            cc_w = None
            if self.causal_confidence and cc_weight is not None:
                cc_w = cc_weight.detach()
                _m = cc_w.mean().clamp(min=1e-8)
                cc_w = cc_w / _m                                   # mean→1 (scale-preserving)
            if self.fsm_arch == "v3" and self.decol_class_balance:
                with torch.no_grad():
                    tgt_cls = decol_target.argmax(-1)                 # (B,) target class
                    cls_cnt = torch.bincount(tgt_cls, minlength=5).float()  # (5,)
                    inv = 1.0 / cls_cnt.clamp(min=1.0)               # inverse freq
                    w_cls = inv[tgt_cls]                            # (B,)
                    w_cls = w_cls * (w_cls.numel() / w_cls.sum().clamp(min=1e-8))  # mean→1
                # per-event KL = Σ_j target*(log target − log pred); weight, then mean.
                kl_pe = (decol_target.detach()
                         * (decol_target.detach().clamp(min=1e-8).log() - log_masked)
                         ).sum(-1)                                  # (B,)
                _w = w_cls if cc_w is None else (w_cls * cc_w)
                edge_trans_loss = (_w * kl_pe).mean()
            elif cc_w is not None:
                # plain CE but per-event so WC-CONF can gate it (== batchmean when cc_w≡1).
                kl_pe = (decol_target.detach()
                         * (decol_target.detach().clamp(min=1e-8).log() - log_masked)
                         ).sum(-1)                                  # (B,)
                edge_trans_loss = (cc_w * kl_pe).mean()
            else:
                edge_trans_loss = F.kl_div(
                    log_masked, decol_target.detach(), reduction="batchmean"
                )
            edge_trans_term = self.lambda_edge_trans * edge_trans_loss
        else:
            edge_trans_loss = torch.tensor(0.0, device=device)
            edge_trans_term = 0.0

        # ── Per-event entropy floor on the CONTINUOUS edge-state distribution ───────
        # (edge_state_entropy_w>0). Maximize the per-event Shannon entropy of
        # softmax(student_logits) → push EACH event's continuous distribution off a
        # pure single-state (directly counters the H→0 pure-state collapse the gate
        # flagged). NEGATIVE-entropy added to loss = maximize H. Trains ECTGv3.
        if self.edge_state_entropy_w > 0.0:
            p_student = F.softmax(student_logits, dim=-1)             # (B,5)
            ent_per_event = -(p_student * (p_student + 1e-8).log()).sum(-1)  # (B,)
            edge_entropy_term = -self.edge_state_entropy_w * ent_per_event.mean()
        else:
            edge_entropy_term = 0.0

        # ── Uniform-prior KL floor on the continuous distribution (edge_uniform_kl_w>0)
        # KL(softmax(student_logits) ‖ Uniform) — a weak floor that drains the DEATH
        # absorbing sink by penalizing any state-distribution that concentrates on one
        # state. Complementary to the entropy floor (this one is mode-symmetric).
        if self.edge_uniform_kl_w > 0.0:
            log_student2 = F.log_softmax(student_logits, dim=-1)      # (B,5)
            uniform = torch.full_like(log_student2, 1.0 / 5.0)
            # KL(p‖u) = sum p (log p - log u); use log_softmax for stability
            p_s = log_student2.exp()
            uniform_kl = (p_s * (log_student2 - uniform.log())).sum(-1).mean()
            edge_uniform_term = self.edge_uniform_kl_w * uniform_kl
        else:
            edge_uniform_term = 0.0

        # ── WC-CONF self-consistency aux term (trains ONLY cc_w_obs_logit). λ small;
        # gradient-isolated from predict (see :belief block proof). Off ⇒ 0.0 ⇒ no-op.
        if cc_self_consist_loss is not None:
            cc_self_consist_term = self.cc_self_consist_w * cc_self_consist_loss
        else:
            cc_self_consist_term = 0.0

        total = (pred_loss
                 + self.lambda_echo * kl
                 + self.lambda_violation * violation_loss
                 + fsm_loss_term
                 + entropy_term
                 + trans_loss_term
                 + edge_trans_term
                 + edge_entropy_term
                 + edge_uniform_term
                 + cc_self_consist_term)

        return {
            # AP/AUC are scored on whichever head pred_loss used in this arm, so the
            # reported metric reflects the head that was actually optimized:
            #   enable_main_predictor=False → existence-decoder/lifecycle readout
            #                                 (CANONICAL detached default)
            #   enable_main_predictor=True  → main head (end-to-end ablation arm)
            "pos_score":      score_pos_logit,
            "neg_score":      score_neg_logit,
            # FSM symbolic existence scores, always exposed for interpretation/monitoring.
            "fsm_pos_score":  pos_logit,
            "fsm_neg_score":  neg_logit,
            "loss":           total,
            "pred_loss":      pred_loss,
            "fsm_loss":       fsm_loss,
            "trans_loss":     trans_loss.detach() if torch.is_tensor(trans_loss) else trans_loss,
            "edge_trans_loss": edge_trans_loss.detach() if torch.is_tensor(edge_trans_loss) else edge_trans_loss,
            # Batch-mean of the STORED continuous ECTGv3 next-state dist new_est[:,:5]
            # (the EXACT quantity the gate's measure() argmaxes), for early-signal
            # entropy monitoring during a run — matches the gate metric, not the raw
            # unmasked student_logits.
            "edge_state_dist_batch": new_est[:, :5].detach().mean(0),
            "tip_loss":       kl,
            "violation_loss": violation_loss,
            "compliance_mean": compliance.mean(),
            "lfg_weight":     torch.tensor(lfg_weight),
            "ccs":            (1.0 - violation_pos).mean(),
            "salience":       sal.mean(),
            "state_dist":     s_t1_pos.mean(0),  # for monitoring
            # fsm_arch="v3" HETEROGENEITY signal (Part D): cross-pair variance of the
            # per-batch next-state dist (>0 ⇔ pairs flip differently). None for v1/v2.
            "pair_het_var":   (s_t1_pos.var(dim=0, unbiased=False).mean().detach()
                               if self.fsm_arch == "v3" else None),
            # PUBLISHED interpretable next-state dist (hier tree, post causal-policy when
            # hier_causal_policy=True). Exposed for the causal-policy probe / evaluation harness
            # lifecycle audit — does NOT feed any loss or score (AP reads pos_score).
            "s_t1_cal":       s_t1_cal.detach(),
            # HƯỚNG A: STRICT-ORDERED 6-state published lifecycle dist (PRE_BIRTH/
            # BIRTH/REINFORCE/DECAY/DORMANT/DEATH), post strict band-diagonal C'.
            # None unless strict_ordered_fsm. Exposed for the evaluation harness lifecycle /
            # transition-block probe; does NOT feed any loss or score.
            "s_t1_cal6":      (s_t1_cal6.detach() if s_t1_cal6 is not None else None),
            # ── WC-CONF outputs (causal_confidence). All None unless the flag is on.
            #   cc_coherence: (B,) c_t ∈[0,1] — CONFIDENCE score per event (high=coherent
            #     with the walked chain, low=causally implausible). Does NOT bend pred.
            #   cc_weight:    (B,) gradient-selection scale applied to the FSM-block CE.
            #   cc_belief:    (B,5) walked-chain belief b_t (carried in the store).
            #   cc_reach:     (B,5) reachable-from-b_{t-1} hard mask.
            "cc_coherence":   (cc_coherence.detach() if cc_coherence is not None else None),
            "cc_weight":      (cc_weight.detach() if cc_weight is not None else None),
            "cc_belief":      (cc_belief.detach() if cc_belief is not None else None),
            "cc_reach":       (cc_reach.detach() if cc_reach is not None else None),
            #   cc_self_consist_loss: scalar aux CE that trains w_obs (None unless aux on).
            #   cc_w_obs: current learned observation-coupling weight sigmoid(logit) ∈(0,1).
            "cc_self_consist_loss": (cc_self_consist_loss.detach()
                                     if cc_self_consist_loss is not None else None),
            "cc_w_obs": (torch.sigmoid(self.cc_w_obs_logit.detach())
                         if self.cc_w_obs_logit is not None else None),
        }
