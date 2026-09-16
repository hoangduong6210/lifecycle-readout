"""
RS-GNN v3 — Multi-signal Edge State + (future) REACT.

This file implements STEP 1 of Master_ML.md roadmap:
  - EdgeStateStoreV3: 12-dim state vector with Welford stats + Hawkes λ + recurrence EWMA
  - ECTGv3: heuristic targets driven by z-score, Hawkes intensity, recurrence
  - SRGNN_v3: drop-in for srgnn_v2 in train.py

Future steps (REACT) will add:
  - EchoMemory with time-decay
  - Adaptive router (state-gated update)
  - Periodic Hopfield pass (post-batch)
  - JointProfile (pair affinity)

State vector layout (12 dims):
  [0:5]  state_logits  (IDLE, BIRTH, REINFORCE, DECAY, DEATH)
  [5]    recur          EWMA recurrence with time decay
  [6]    hawkes_lambda  self-exciting intensity
  [7]    mean_dt        Welford running mean of inter-arrival
  [8]    var_dt         Welford running variance
  [9]    n_obs          observation count (for Welford)
  [10]   vol            volatility (state-flip rate)
  [11]   life           cumulative log-time
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from .sr_gnn import (
    IDLE, BIRTH, REINFORCE, DECAY, DEATH,
    VALID_TRANSITIONS,
    TimeEncoder,
    NodeMemoryStore,
)
from .sr_gnn_v2 import ResidualCSN, DRGC_v2, NSCP_v2

STATE_DIM = 12

# Multi-signal hyperparameters
RECUR_ALPHA   = 0.3      # EWMA weight for new event
RECUR_LAMBDA  = 0.001    # decay rate for forgetting
HAWKES_ALPHA  = 1.0      # jump magnitude on event
HAWKES_BETA   = 0.01     # decay rate of hawkes intensity
HAWKES_MU     = 0.1      # baseline rate

# fsm_arch="v3" edit-RATE EWMA params (PM 2026-06-01 SLOPE axis; v1/v2 never use these)
RATE_INIT     = 0.1      # init EWMA value for rate_fast/rate_slow (low baseline rate;
                         #   fresh pair ⇒ slope≈0 ⇒ not spuriously DECAY)
RATE_DT_FLOOR = 1e-3     # Δt floor so rate=1/Δt stays finite for coincident events


class EdgeStateStoreV3:
    """
    Sparse 12-dim edge state store with Welford online stats + Hawkes recursion.

    On each event (u,v,t,Δt):
      1. Update Welford (n, mean_dt, M2 for var)
      2. Update recurrence EWMA: r ← α·1 + (1-α)·r_prev·exp(-λ·Δt)
      3. Update hawkes_λ: λ ← α + (λ_prev - μ)·exp(-β·Δt) + μ
      4. (state_logits updated outside by ECTGv3)
    """
    def __init__(self, num_nodes: int, hidden: int, device: torch.device,
                 causal_batch: bool = False):
        self.N      = num_nodes
        self.device = device
        # CAUSAL INTRA-BATCH ACCUMULATION (P1 fix, 2026-06-03). When True, get_batch /
        # the rate-EWMA / peak helpers fold repeated same-pair events WITHIN one batch
        # in stream order, so each event reads the state AFTER the previous same-pair
        # in-batch event (not the same pre-batch snapshot K×). Default False = legacy
        # (buggy) batched read-before-write, kept for A/B. See SRGNN_v3_3.causal_batch.
        self.causal_batch = causal_batch
        self._key_to_idx: Dict[int, int] = {}
        self._state_table: list = []   # list of 12-dim tensors
        # Persistent per-pair TRUE occurrence counter. Incremented once per event in
        # stream order (collision-immune): unlike the Welford n_obs in update_batch,
        # which suffers batched read-before-write (multiple occurrences of one pair in
        # ONE batch all read the pre-batch state and only the last write persists, so
        # ~72-75% of repeat events spuriously look like n_obs<=1 at batch=200-500 on
        # coedit — DIAGNOSED 2026-05-30), this counter advances for EVERY event. It is
        # the correct signal for `is_first` (BIRTH) in the de-collapse target.
        self._occ_count: Dict[int, int] = {}

    def _make_init_state(self) -> Tensor:
        s = torch.zeros(STATE_DIM, device=self.device)
        s[IDLE]       = 1.0   # start IDLE
        s[6]          = HAWKES_MU  # hawkes_λ baseline
        return s

    def get_batch(self, src: Tensor, dst: Tensor, dt: Tensor = None) -> Tensor:
        """Return the PRE-event 12-dim state per event.

        Legacy (causal_batch=False): snapshots the store ONCE; every repeated same-pair
        event in this batch reads the IDENTICAL pre-batch row (the P1 read-before-write
        bug — Welford n caps, μ/var/rate corrupted on recurring pairs).

        Causal (causal_batch=True, requires `dt`): for repeated same-pair events in this
        batch, the DETERMINISTIC continuous channels [5:10] (recur EWMA, Hawkes λ,
        mean_dt, var_dt, n_obs) of the k-th in-batch occurrence are replayed event-by-
        event so they already fold in the (k−1) earlier in-batch events of that pair.
        Each event's own current dt is NOT folded here (that happens in ECTGv3.forward's
        update_multisignal) ⇒ scoring stays strictly PRE-update / no re-leak. The learned
        channels [0:5],[10:12] are carried from the pre-batch row (the model rewrites them
        per-event in forward). After this call the store row reflects ALL but the last
        in-batch event's continuous folds, so update_batch (which writes the model's
        new_est for the last occurrence) lands on the correct accumulator."""
        B = src.size(0)
        states = torch.zeros(B, STATE_DIM, device=self.device)
        src_c = src.tolist()
        dst_c = dst.tolist()
        if not self.causal_batch or dt is None:
            for i, (u, v) in enumerate(zip(src_c, dst_c)):
                key = u * self.N + v
                if key not in self._key_to_idx:
                    idx = len(self._state_table)
                    self._key_to_idx[key] = idx
                    self._state_table.append(self._make_init_state())
                states[i] = self._state_table[self._key_to_idx[key]]
            return states

        # ── CAUSAL PATH ──────────────────────────────────────────────────────────
        dt_c = dt.detach().float().clamp(min=0.0).tolist()
        # running causal pre-state per pair WITHIN this batch (channels [5:10] only;
        # learned channels carried from the stored/init row). We fold each event's dt
        # into a SCRATCH copy AFTER reading, so the NEXT same-pair event in this batch
        # reads the post-state of the previous one.
        scratch: Dict[int, Tensor] = {}
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key not in self._key_to_idx:
                idx = len(self._state_table)
                self._key_to_idx[key] = idx
                self._state_table.append(self._make_init_state())
            if key not in scratch:
                # first in-batch occurrence: pre-state = the stored (pre-batch) row.
                scratch[key] = self._state_table[self._key_to_idx[key]].clone()
            pre = scratch[key]
            states[i] = pre                      # PRE-event state for THIS event
            # fold THIS event's dt into a fresh copy → the pre-state for the pair's
            # NEXT in-batch event (deterministic [5:10] update; ECTGv3 will recompute
            # the same fold for the CURRENT event's own new_est, so no double count).
            dt_i = torch.tensor([dt_c[i]], device=self.device)
            folded = update_multisignal(pre.unsqueeze(0), dt_i).squeeze(0)
            # keep learned channels [0:5],[10:12] from `pre` (the model owns them);
            # only the deterministic accumulators [5:10] advance in the scratch.
            nxt = pre.clone()
            nxt[5:10] = folded[5:10]
            scratch[key] = nxt
            # persist the causal accumulator so the NEXT batch starts correct. The last
            # in-batch occurrence's full new_est (incl. learned channels) is written by
            # update_batch; here we keep [5:10] coherent for intermediate occurrences.
            self._state_table[self._key_to_idx[key]][5:10] = nxt[5:10]
        return states

    def peek_batch(self, src: Tensor, dst: Tensor) -> Tensor:
        """READ-ONLY view of the stored 12-dim state per pair, WITHOUT registering
        unseen keys or mutating the store. Unseen pairs return the fresh init state
        (IDLE, hawkes_λ baseline). Used to build the NEGATIVE pathway's per-pair
        operator features symmetrically with the positive (anti-leak fix, 2026-05-31):
        the negative candidate (src,neg_dst) gets φ from its OWN accumulated history
        (= init state for a never-seen pair) instead of NO φ — closing the
        'pos has a per-pair operator, neg has none' label shortcut. Must NOT touch
        _key_to_idx / _state_table / _occ_count (negatives are not real events)."""
        B = src.size(0)
        states = torch.zeros(B, STATE_DIM, device=self.device)
        src_c = src.tolist()
        dst_c = dst.tolist()
        init = self._make_init_state()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            idx = self._key_to_idx.get(key, None)
            states[i] = self._state_table[idx] if idx is not None else init
        return states

    # ── fsm_arch="v3" per-pair Hawkes-λ PEAK store (OPT-IN; v1/v2 never call these) ──
    # The 12-dim state vector carries the CURRENT Hawkes λ ([6]) but NOT the pair's
    # recent PEAK λ. The DECAY state (per PM 2026-06-01) is the *trend* of the
    # interaction intensity ROLLING OFF that pair's own recent peak while the pair is
    # still alive — distinct from DEATH (long absolute silence). To measure "λ rớt khỏi
    # đỉnh gần đây của CHÍNH cặp đó" we keep a leaky running peak per pair:
    #   peak ← max(λ_now, PEAK_LEAK · peak_prev)    (PEAK_LEAK<1 slowly forgets old highs)
    # and read it PRE-update via peek_lam_peak so the trend does not see the current
    # event's own jump (anti-leak, same invariant as edge_st). Lazily created; ONLY
    # touched when peek_lam_peak / update_lam_peak are called (fsm_arch="v3"), so the
    # v1/v2 store / state_dict / forward are byte-identical.
    def peek_lam_peak(self, src: Tensor, dst: Tensor) -> Tensor:
        """READ-ONLY per-pair leaky-peak Hawkes λ (PRE-update). Unseen pairs return the
        baseline HAWKES_MU. Does NOT register keys or mutate the store."""
        if not hasattr(self, "_lam_peak"):
            self._lam_peak: Dict[int, float] = {}
        B = src.size(0)
        out = torch.full((B,), HAWKES_MU, device=self.device)
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            p = self._lam_peak.get(u * self.N + v, None)
            if p is not None:
                out[i] = p
        return out

    def update_lam_peak(self, src: Tensor, dst: Tensor, lam_now: Tensor,
                        leak: float = 0.97):
        """Update the leaky per-pair peak with the POST-update λ of THIS event:
        peak ← max(λ_now, leak·peak_prev). Call AFTER update_batch (so lam_now is the
        post-event Hawkes intensity new_est[:,6]). Real events only — negatives never
        call this (parallels update_batch)."""
        if not hasattr(self, "_lam_peak"):
            self._lam_peak: Dict[int, float] = {}
        src_c = src.tolist(); dst_c = dst.tolist()
        ln = lam_now.detach().tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            prev = self._lam_peak.get(key, HAWKES_MU)
            self._lam_peak[key] = max(float(ln[i]), leak * prev)
        return

    # ── fsm_arch="v3" per-pair EDIT-RATE fast/slow EWMA (OPT-IN; v1/v2 never call) ──
    # PM 2026-06-01 (THIRD re-chốt): all three active states live on ONE axis = the
    # edit-RATE per time-window and the SIGN of its SLOPE. The rate of THIS event =
    # 1/Δt (events per unit time; "7-8 lần/5 ngày" ⇒ rate≈1.5, "7-8 lần/2-3 ngày" ⇒
    # rate≈3). NB: the carried Hawkes λ is NOT a rate — with α=1,β=0.01 it just
    # ACCUMULATES (≈ a burst event-count) and rises monotonically as long as the pair
    # fires, so it cannot tell "firing faster" from "firing slower" (CPU-proven
    # 2026-06-01: λ climbs through BOTH the accelerating AND the decelerating phase).
    # The RATE 1/Δt does separate them. We keep TWO EWMAs of the rate per pair:
    #   rate_fast ← af·r + (1−af)·rate_fast   (af large ⇒ short memory, tracks recent)
    #   rate_slow ← as·r + (1−as)·rate_slow   (as small ⇒ long memory, lags)
    #   slope = rate_fast − rate_slow
    #     slope ≥ +margin            → rate RISING (gaps shrinking)  ⇒ REINFORCE
    #     slope ≤ −margin & rate>dead → rate FALLING-but-alive       ⇒ DECAY
    #     rate ≤ rate_dead (sustained, after decay)                  ⇒ DEATH
    # The DISCRIMINATOR REINFORCE↔DECAY is the SIGN of this slope — NOT |λ| magnitude
    # nor gap length — fixing both prior failures (REINFORCE=absolute-high λ left DECAY
    # no room; DEATH swallowed DECAY). Read PRE-update (peek) → written POST-update
    # (update) ⇒ strictly causal, no re-leak. Both EWMAs init at RATE_INIT (a low
    # baseline rate) so a brand-new pair has slope≈0 (not spuriously DECAY). Lazily
    # created; v1/v2 never call it ⇒ store/state_dict/forward byte-identical.
    def peek_rate_ewma(self, src: Tensor, dst: Tensor) -> Tuple[Tensor, Tensor]:
        """READ-ONLY per-pair (rate_fast, rate_slow) EWMA of the edit rate 1/Δt
        (PRE-update). Unseen pairs return RATE_INIT for both (slope≈0). Does NOT
        register keys or mutate the store."""
        if not hasattr(self, "_rate_ewma"):
            self._rate_ewma: Dict[int, Tuple[float, float]] = {}
        B = src.size(0)
        fast = torch.full((B,), RATE_INIT, device=self.device)
        slow = torch.full((B,), RATE_INIT, device=self.device)
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            p = self._rate_ewma.get(u * self.N + v, None)
            if p is not None:
                fast[i], slow[i] = p[0], p[1]
        return fast, slow

    def update_rate_ewma(self, src: Tensor, dst: Tensor, dt: Tensor,
                         af: float = 0.6, as_: float = 0.2):
        """Update the per-pair (rate_fast, rate_slow) EWMAs with THIS event's rate
        r = 1/Δt (clamped). af>as ⇒ fast tracks recent rate, slow lags ⇒ slope=
        fast−slow encodes the rate TREND. Call AFTER update_batch (real events only;
        negatives never call this, parallels update_batch/update_lam_peak)."""
        if not hasattr(self, "_rate_ewma"):
            self._rate_ewma: Dict[int, Tuple[float, float]] = {}
        src_c = src.tolist(); dst_c = dst.tolist()
        dtl = dt.detach().float().clamp(min=RATE_DT_FLOOR).tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            r = 1.0 / float(dtl[i])
            pf, ps = self._rate_ewma.get(key, (RATE_INIT, RATE_INIT))
            self._rate_ewma[key] = (af * r + (1.0 - af) * pf,
                                    as_ * r + (1.0 - as_) * ps)
        return

    # ── fsm_arch="v3" per-pair leaky-PEAK of the edit-RATE (OPT-IN; v1/v2 never call) ──
    # PER-PAIR-RELATIVE DEATH gate (PM 2026-06-01 fix): the absolute rate_dead=0.25 was
    # off the coedit scale — rate=1/Δt on coedit has median≈0.10, max≈0.195, so a fixed
    # 0.25 floor reads EVERY active pair as DEAD and the FALLING-active (DECAY) gate never
    # opened (legacy diagnostic: 9580/12000 events slope≈0, DECAY=0 mass). The fix: "dead" is
    # per-pair RELATIVE — the current rate has fallen to a small FRACTION γ of the pair's
    # OWN recent peak rate (rate_fast < γ·rate_peak_pair). We keep a leaky running max of
    # rate_fast per pair (mirrors _lam_peak): peak ← max(rate_fast_now, leak·peak_prev).
    # Read PRE-update via peek_rate_peak (no re-leak — the current event's own rate is NOT
    # yet in the peak), updated POST-update. Lazily created; v1/v2 never call ⇒ store /
    # state_dict / forward byte-identical. Cleared in reset().
    def peek_rate_peak(self, src: Tensor, dst: Tensor) -> Tensor:
        """READ-ONLY per-pair leaky-peak of rate_fast (PRE-update). Unseen pairs return
        RATE_INIT (their baseline). Does NOT register keys or mutate the store."""
        if not hasattr(self, "_rate_peak"):
            self._rate_peak: Dict[int, float] = {}
        B = src.size(0)
        out = torch.full((B,), RATE_INIT, device=self.device)
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            p = self._rate_peak.get(u * self.N + v, None)
            if p is not None:
                out[i] = p
        return out

    def update_rate_peak(self, src: Tensor, dst: Tensor, rate_fast_now: Tensor,
                         leak: float = 0.97):
        """Update the leaky per-pair RATE peak with the POST-update rate_fast of THIS
        event: peak ← max(rate_fast_now, leak·peak_prev). Call AFTER update_rate_ewma so
        rate_fast_now is the post-event fast EWMA. Real events only."""
        if not hasattr(self, "_rate_peak"):
            self._rate_peak: Dict[int, float] = {}
        src_c = src.tolist(); dst_c = dst.tolist()
        rn = rate_fast_now.detach().tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            prev = self._rate_peak.get(key, RATE_INIT)
            self._rate_peak[key] = max(float(rn[i]), leak * prev)
        return

    # ── CAUSAL intra-batch peeks for the rate signals (P1 fix, causal_batch=True) ──
    # The legacy model pattern is peek_*(pre, read once/batch) → update_*(post, loops
    # event-by-event). The UPDATE is already causal (writes back each iter); only the
    # PEEK is stale (every repeated same-pair event reads the same pre-batch value).
    # These three methods replay the EXACT pre→post sequence event-by-event so each
    # event's returned PRE value reflects all earlier same-pair events in THIS batch,
    # AND advance the dict to the post-batch state (so the model must NOT also call the
    # separate update_rate_ewma / update_rate_peak when causal_batch is on — the caller
    # gates that). lam_now for the peak update is the post-fast (== update_rate_peak via
    # _rf_post). Returns (rate_fast_pre, rate_slow_pre, rate_peak_pre), each (B,).
    def peek_step_rate_causal(self, src: Tensor, dst: Tensor, dt: Tensor,
                              af: float = 0.6, as_: float = 0.2,
                              leak: float = 0.97) -> Tuple[Tensor, Tensor, Tensor]:
        if not hasattr(self, "_rate_ewma"):
            self._rate_ewma: Dict[int, Tuple[float, float]] = {}
        if not hasattr(self, "_rate_peak"):
            self._rate_peak: Dict[int, float] = {}
        B = src.size(0)
        fast_pre = torch.empty(B, device=self.device)
        slow_pre = torch.empty(B, device=self.device)
        peak_pre = torch.empty(B, device=self.device)
        src_c = src.tolist(); dst_c = dst.tolist()
        dtl = dt.detach().float().clamp(min=RATE_DT_FLOOR).tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            pf, ps = self._rate_ewma.get(key, (RATE_INIT, RATE_INIT))
            pk     = self._rate_peak.get(key, RATE_INIT)
            # PRE values for THIS event (matches the legacy peek semantics):
            fast_pre[i] = pf
            slow_pre[i] = ps
            peak_pre[i] = pk
            # advance EWMA with this event's rate (== update_rate_ewma):
            r = 1.0 / float(dtl[i])
            nf = af * r + (1.0 - af) * pf
            ns = as_ * r + (1.0 - as_) * ps
            self._rate_ewma[key] = (nf, ns)
            # advance peak with the POST fast (== update_rate_peak via _rf_post):
            self._rate_peak[key] = max(nf, leak * pk)
        return fast_pre, slow_pre, peak_pre

    def peek_lam_peak_causal(self, src: Tensor, dst: Tensor) -> Tensor:
        """CAUSAL per-event PRE leaky-peak Hawkes λ. The model updates _lam_peak with
        new_est[:,6] AFTER this peek; that update already loops event-by-event (causal),
        so here we only need the per-event PRE value, which the legacy peek_lam_peak
        already returns correctly (it reads the dict once, but _lam_peak is written by
        update_lam_peak AFTER the whole peek — so the first-in-batch value is the only
        coherent PRE anyway). We return the same read; the update remains the legacy
        per-event loop. Kept as a named method for symmetry / future per-event PRE."""
        return self.peek_lam_peak(src, dst)

    def get_true_occ(self, src: Tensor, dst: Tensor) -> Tensor:
        """Return the 1-based TRUE occurrence index of each event in (src,dst) IN
        STREAM ORDER, then advance the persistent counter. Collision-immune: events
        of the same pair within one batch get distinct, increasing indices
        (1,2,3,...) — fixing the n_obs<=1 BIRTH inflation. Returns float (B,)."""
        B = src.size(0)
        out = torch.empty(B, device=self.device)
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            c = self._occ_count.get(key, 0) + 1     # 1-based index of THIS event
            self._occ_count[key] = c
            out[i] = float(c)
        return out

    def update_batch(self, src: Tensor, dst: Tensor, new_states: Tensor):
        src_c = src.tolist()
        dst_c = dst.tolist()
        ns = new_states.detach()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key in self._key_to_idx:
                self._state_table[self._key_to_idx[key]] = ns[i]

    # ── fsm_arch="v2" symbolic-state persistence (OPT-IN; v1 never calls these) ──
    # The v1 gate measures _state_table[:,:5] = the VALID-HARD-MASKED ECTGv3 continuous
    # chain, which is structurally pinned (one-rung/event forward chain on a DETACHED
    # argmax + gradient cannot cross the -1e9 mask → BIRTH-frontier attractor, invariant
    # to supervision weight — DIAGNOSED 2026-05-31). fsm_arch="v2" instead measures the
    # SEPARATE soft-masked FSM head s_{t+1} (StateObserver→TransitionPredictor→soft
    # LifecycleFSMMask, sr_gnn_v3_3.py:543-555), persisted here so measure() can read a
    # per-pair snapshot. The soft sigmoid(prior+delta) penalty is FINITE, so supervision
    # can move every state's probability (CPU-proven movable). This store is lazily
    # created and ONLY touched when update_symbolic is called.
    def update_symbolic(self, src: Tensor, dst: Tensor, sym_states: Tensor):
        """Persist the soft FSM-head next-state dist (B,5) per pair, DETACHED. Keys are
        registered by get_batch (called first in forward), so reuses _key_to_idx."""
        if not hasattr(self, "_sym_table"):
            self._sym_table: Dict[int, Tensor] = {}
        src_c = src.tolist(); dst_c = dst.tolist()
        ss = sym_states.detach()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            self._sym_table[u * self.N + v] = ss[i]

    # ── WALKED-CHAIN BELIEF b_t (WC-CONF, PM 2026-06-06) ──────────────────────
    # Per-pair causal forward-filter belief over the 5-state lifecycle. b_t is the
    # "where this pair SHOULD be according to the causal chain it actually walked"
    # (carried across the pair's events). Stored DETACHED — it is a confidence /
    # gradient-selection signal, NEVER a learned/prediction quantity. Lazily created;
    # ONLY touched when the SRGNN_v3_3 causal_confidence flag is on (default OFF ⇒ this
    # store is never instantiated ⇒ canonical/config-B byte-identical, no new state).
    def peek_belief(self, src: Tensor, dst: Tensor,
                    init_belief: Tensor = None) -> Tensor:
        """PRE-update read of b_{t-1} per event (B,5). A pair that already has a
        carried belief returns it. A pair making its FIRST appearance in the split
        (no entry in the store) gets the INIT:
          • init_belief is None (default, WC-CONF FIX-3/R2) → honest pre-birth IDLE
            one-hot (legacy, byte-identical when cc_grounded_init is off);
          • init_belief is a (B,5) tensor (cc_grounded_init ON) → the per-row
            MODEL-INFERRED phase for unseen pairs (caller passes softmax(s_t_pos),
            history-only, PRE-update, detached). This GROUNDS the chain at the pair's
            real phase instead of resetting mature pairs to IDLE at test time.
        Read-only ⇒ no leak (scoring reads b_{t-1}; the update lands AFTER scoring via
        update_belief). The grounded init does NOT write the store — it only seeds the
        return for this event; update_belief writes the walked belief POST-scoring."""
        if not hasattr(self, "_belief_table"):
            self._belief_table: Dict[int, Tensor] = {}
        B = src.size(0)
        if init_belief is not None:
            # grounded init: unseen rows fall back to the model-inferred phase, NOT IDLE.
            out = init_belief.detach().to(self.device).clone()
        else:
            out = torch.zeros(B, STATE_DIM_5 := 5, device=self.device)
            out[:, IDLE] = 1.0                   # IDLE-init for unseen pairs (legacy)
        bt = self._belief_table
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            b = bt.get(u * self.N + v)
            if b is not None:
                out[i] = b                       # carried belief overrides init
        return out

    def update_belief(self, src: Tensor, dst: Tensor, b_states: Tensor):
        """POST-scoring write of b_t (B,5) per pair, DETACHED. Keys already
        registered by get_batch. Carries the belief to the pair's NEXT event."""
        if not hasattr(self, "_belief_table"):
            self._belief_table: Dict[int, Tensor] = {}
        bs = b_states.detach()
        src_c = src.tolist(); dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            self._belief_table[u * self.N + v] = bs[i]

    def reset(self):
        self._key_to_idx.clear()
        self._state_table.clear()
        self._occ_count.clear()
        if hasattr(self, "_sym_table"):
            self._sym_table.clear()
        if hasattr(self, "_belief_table"):
            self._belief_table.clear()
        if hasattr(self, "_lam_peak"):
            self._lam_peak.clear()
        if hasattr(self, "_rate_ewma"):
            self._rate_ewma.clear()
        if hasattr(self, "_rate_peak"):
            self._rate_peak.clear()


def update_multisignal(prev_state: Tensor, dt: Tensor) -> Tensor:
    """
    Vectorised update of recur, hawkes_λ, Welford stats given Δt.
    Input: prev_state (B, 12), dt (B,)
    Output: updated state (B, 12) with [5:10] refreshed; [0:5], [10:12] kept as-is.
    """
    s = prev_state.clone()
    dt_f = dt.float().clamp(min=0.0)

    # (a) Recurrence EWMA with forgetting
    r_prev = s[:, 5]
    r_new  = RECUR_ALPHA + (1 - RECUR_ALPHA) * r_prev * torch.exp(-RECUR_LAMBDA * dt_f)
    s[:, 5] = r_new

    # (b) Hawkes self-exciting intensity (recursive form)
    lam_prev = s[:, 6]
    lam_new  = HAWKES_ALPHA + (lam_prev - HAWKES_MU) * torch.exp(-HAWKES_BETA * dt_f) + HAWKES_MU
    s[:, 6] = lam_new

    # (c) Welford online: mean_dt, var_dt, n
    n_prev    = s[:, 9]
    mean_prev = s[:, 7]
    M2_prev   = s[:, 8] * n_prev.clamp(min=1.0)   # store var; reconstruct M2

    n_new    = n_prev + 1.0
    delta    = dt_f - mean_prev
    mean_new = mean_prev + delta / n_new
    delta2   = dt_f - mean_new
    M2_new   = M2_prev + delta * delta2
    var_new  = M2_new / n_new.clamp(min=1.0)

    s[:, 7] = mean_new
    s[:, 8] = var_new
    s[:, 9] = n_new

    return s


class ECTGv3(nn.Module):
    """
    Multi-signal ECTG with z-score / Hawkes / recurrence-driven heuristic targets.
    """
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        in_dim = feat_dim + 1 + 5 + 3   # feat | log_dt | cur_dist | (recur, hawkes_λ, z)
        self.trans_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 5)
        )
        self.intensity_fc = nn.Linear(1, 1)
        self.ctx_encoder  = nn.Linear(STATE_DIM, hidden)

    def forward(self, feat: Tensor, salience: Tensor,
                delta_t: Tensor, edge_states: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        edge_states: (B, 12)  raw state BEFORE this event update
        Returns:
          new_states (B, 12)
          edge_ctx (B, hidden)
          student_logits (B, 5)
          heuristic_target_dist (B, 5)
        """
        B      = feat.size(0)
        cur_dist = torch.softmax(edge_states[:, :5], dim=-1)
        log_dt   = torch.log1p(delta_t.float()).unsqueeze(-1)

        # 1) Update multi-signal stats (recur, hawkes, mean_dt, var_dt, n)
        s_updated = update_multisignal(edge_states, delta_t)
        recur      = s_updated[:, 5:6]
        hawkes_lam = s_updated[:, 6:7]
        mean_dt    = s_updated[:, 7:8]
        var_dt     = s_updated[:, 8:9]

        # 2) z-score (use updated mean/var)
        std_dt = torch.sqrt(var_dt.clamp(min=1e-6))
        z      = (delta_t.float().unsqueeze(-1) - mean_dt) / (std_dt + 1e-6)

        # 3) Transition network input
        trans_in = torch.cat([
            feat, log_dt, cur_dist,
            recur, torch.log1p(hawkes_lam), z.clamp(-5, 5),
        ], dim=-1)
        student_logits = self.trans_net(trans_in)

        # 4) Causal mask
        cur_idx = cur_dist.argmax(-1)
        mask    = VALID_TRANSITIONS[cur_idx.cpu()].to(feat.device)
        masked_logits = student_logits.masked_fill(~mask, -1e9)
        new_dist = torch.softmax(masked_logits, dim=-1)

        # 5) Update intensity (cumulative salience), volatility, life
        intensity_old = edge_states[:, 5:6]   # NOTE: dim 5 is recur in v3
        # We keep "intensity"-like behavior via hawkes_lam → not duplicating
        new_vol_in   = (new_dist.argmax(-1) != cur_idx).float().unsqueeze(-1)
        new_vol      = edge_states[:, 10:11] * 0.9 + new_vol_in * 0.1
        new_life     = edge_states[:, 11:12] + log_dt * 0.01

        new_states = torch.cat([
            new_dist,                  # [0:5]
            recur,                     # [5]
            hawkes_lam,                # [6]
            mean_dt, var_dt,           # [7], [8]
            s_updated[:, 9:10],        # [9] n_obs
            new_vol, new_life,         # [10], [11]
        ], dim=-1)
        edge_ctx = self.ctx_encoder(new_states)

        # 6) Heuristic target — multi-signal
        is_first  = (recur.squeeze(-1) < 0.05).float()
        is_active = torch.sigmoid(hawkes_lam.squeeze(-1) - 1.0)
        is_late   = torch.sigmoid(z.squeeze(-1) - 1.5)
        is_dead   = torch.sigmoid(z.squeeze(-1) - 3.0)

        target_logits = torch.zeros(B, 5, device=feat.device)
        target_logits[:, IDLE]      = (1 - is_active - is_first).clamp(min=-2, max=2)
        target_logits[:, BIRTH]     = is_first * 2.0
        target_logits[:, REINFORCE] = is_active * (1 - is_late) * 1.5
        target_logits[:, DECAY]     = is_late * (1 - is_dead) * 1.5
        target_logits[:, DEATH]     = is_dead * 2.0

        target_masked = target_logits.masked_fill(~mask, -1e9)
        heuristic_target_dist = torch.softmax(target_masked, dim=-1).detach()

        return new_states, edge_ctx, student_logits, heuristic_target_dist


class SRGNN_v3(nn.Module):
    """
    RS-GNN v3 (Step 1): multi-signal edge state.
    Drops in for srgnn_v2: same forward signature, same return dict.
    """
    def __init__(self, num_nodes: int, feat_dim: int, hidden: int = 128,
                 tip_beta: float = 0.001,
                 lambda_tip: float = 0.01, lambda_causal: float = 0.1,
                 lambda_trans: float = 0.05, lambda_distill: float = 0.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_nodes      = num_nodes
        self.feat_dim       = feat_dim
        self.hidden         = hidden
        self.lambda_tip     = lambda_tip
        self.lambda_causal  = lambda_causal
        self.lambda_trans   = lambda_trans
        self.lambda_distill = lambda_distill
        self.device         = device
        self._feat_in       = max(feat_dim, 1)

        self.csn  = ResidualCSN(self._feat_in, hidden)
        self.ectg = ECTGv3(self._feat_in, hidden)
        self.drgc = DRGC_v2(self._feat_in, hidden, tip_beta)
        self.nscp = NSCP_v2(hidden)

        # NSCP_v2 expects 8-dim edge_state; we'll pass [logits(5) | hawkes_lam | recur | vol] = 8 dims
        # state_oracle: predict 5-dim state from [h_src, h_dst]
        self.state_oracle = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        ).to(device)

        self.node_mem = NodeMemoryStore(num_nodes, hidden, device)
        self.edge_mem = EdgeStateStoreV3(num_nodes, hidden, device)

    def reset(self):
        self.node_mem.reset()
        self.edge_mem.reset()

    def _v3_to_v2_state(self, s12: Tensor) -> Tensor:
        """Adapt 12-dim v3 state to 8-dim format expected by NSCP_v2.
        NSCP_v2 uses dims [:5] as state_logits, [5] as intensity, [6] as vol, [7] as life.
        We map: hawkes_lam → intensity, vol → vol, life → life."""
        return torch.cat([
            s12[:, :5],         # state_logits
            s12[:, 6:7],        # hawkes_lam (proxy for intensity)
            s12[:, 10:11],      # vol
            s12[:, 11:12],      # life
        ], dim=-1)

    def forward(self, src: Tensor, dst: Tensor, t: Tensor,
                feat: Tensor, neg_dst: Tensor,
                rel_type: Optional[Tensor] = None) -> Dict[str, Tensor]:

        device = self.device
        B = src.size(0)

        if feat.shape[-1] == 0:
            feat = torch.zeros(B, 1, device=device)
        elif feat.shape[-1] < self._feat_in:
            feat = F.pad(feat, (0, self._feat_in - feat.shape[-1]))

        # L1 — Residual CSN
        dt_src = self.node_mem.delta_t(src, t)
        feat_g, sal = self.csn(feat, dt_src)

        # L2 — ECTG v3 (multi-signal)
        edge_st = self.edge_mem.get_batch(src, dst)
        new_est, edge_ctx, student_logits, heuristic_target_dist = self.ectg(
            feat_g, sal, dt_src, edge_st
        )
        self.edge_mem.update_batch(src, dst, new_est)

        # Adapt for NSCP_v2 (still expects 8-dim)
        edge_state_v2 = self._v3_to_v2_state(new_est)
        intensity = edge_state_v2[:, 5]   # hawkes_lam used as intensity proxy

        # L3 — DRGC v2 + TIP
        h_src = self.node_mem.get(src)
        h_dst = self.node_mem.get(dst)
        dt_dst = self.node_mem.delta_t(dst, t)
        all_idx = torch.unique(torch.cat([src, dst]))
        all_h = self.node_mem.get(all_idx)
        all_staleness = (t.max().float() - self.node_mem.last_t[all_idx]).clamp(0)

        new_h_src, new_h_dst, parsed_h, kl = self.drgc(
            h_src, h_dst, feat_g, edge_ctx, dt_src, dt_dst,
            intensity, all_h, all_staleness
        )

        # L4 — StateOracle + NSCP_v2
        teacher_logits = self.state_oracle(torch.cat([new_h_src, new_h_dst], dim=-1))
        neg_emb = self.node_mem.get(neg_dst)
        pos_sc, neg_sc, c_loss, ccs = self.nscp(
            new_h_src, new_h_dst, neg_emb, edge_state_v2, teacher_logits
        )

        # Memory update (after scoring)
        self.node_mem.set(all_idx, parsed_h)
        self.node_mem.set(src, new_h_src)
        self.node_mem.set(dst, new_h_dst)
        self.node_mem.update_time(torch.cat([src, dst]), torch.cat([t, t]))

        # Loss
        pred_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])
        )

        # Hybrid trans loss (heuristic + optional distill)
        log_s_student = F.log_softmax(student_logits, dim=-1)
        loss_heuristic = F.kl_div(log_s_student, heuristic_target_dist, reduction='batchmean')
        if self.lambda_distill > 0:
            s_teacher = F.softmax(teacher_logits, dim=-1).detach()
            loss_predictive = F.kl_div(log_s_student, s_teacher, reduction='batchmean')
            trans_loss = (1 - self.lambda_distill) * loss_heuristic + self.lambda_distill * loss_predictive
        else:
            trans_loss = loss_heuristic

        total = (pred_loss
                 + self.lambda_tip * kl
                 + self.lambda_causal * c_loss
                 + self.lambda_trans * trans_loss)

        return {
            "pos_score": pos_sc, "neg_score": neg_sc,
            "loss": total, "pred_loss": pred_loss,
            "tip_loss": kl, "causal_loss": c_loss,
            "trans_loss": trans_loss,
            "ccs": ccs.mean(), "salience": sal.mean(),
        }
