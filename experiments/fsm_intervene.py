"""
FSM hier-v2 COUNTERFACTUAL / INTERVENTION engine  (eval-only, 0 new params).

WHY THIS FILE EXISTS
--------------------
hier-v2 (decol_hier_v2, sr_gnn_v3_3.py L1008-1052) is a small structural causal
model (SCM) for the lifecycle next-state:

    p_birth  = sigmoid( birth_prior(true_occ, n_prior)            + r_birth )
    p_alive  = sigmoid( 2·_recurring − 2.2·_sustained_dead
                        + 0.4·rate_ratio                          + r_alive )
    p_rising = sigmoid( 1.2·_recurring − 1.5·_cooling             + r_rising )

    P(BIRTH)     = p_birth
    P(REINFORCE) = (1−p_birth)·p_alive·p_rising
    P(DECAY)     = (1−p_birth)·p_alive·(1−p_rising)
    P(DEATH)     = (1−p_birth)·(1−p_alive)

The gate ARGUMENTS are the causal lifecycle DRIVERS:
    true_occ   (recurrence)      → _recurring, birth
    n_prior    (trustworthy hist)→ _has_hist_h gate on the silence terms
    rate_fast  + rate_dead_pp    → rate_ratio        (recoverable from npz exactly)
    slope_rel  (rate slope)      → _cooling (rising), bounded effect
    stale_rel  = dt/μ_pair       → _sustained_dead (alive), _cooling (rising)

Intervening on a driver and RE-DECODING gives the counterfactual state dist.
do(state=X) forces the next-state distribution and pushes it through the REAL
existence readout to get a counterfactual P(edge).

FIDELITY / HONESTY
------------------
* This engine works OFFLINE on the faithfulness npz (no GPU, no re-run, no leak):
  every quantity it reads is a PRE-update probe already dumped by the model.
* The trained residual heads (hier_birth/alive/rising_head, +147 params) are NOT
  saved to disk by the eval driver. We therefore RECOVER, per event, the residual
  by INVERTING the observed gate against the analytic prior we CAN reconstruct:
        r_obs = logit(p_gate_observed) − analytic_prior(drivers_observed)
  and HOLD r_obs FIXED across an intervention. This is an EXACT local intervention
  on the analytic prior (the part driven by the lifecycle causes) with the learned
  residual frozen at its observed per-event value — the most faithful thing
  possible without re-running the trained model. do(driver) moves only the prior.
* stale_rel = dt/μ_pair is NOT in the npz (μ_pair is intra-batch-corrupted to ≈0 on
  coedit; z_pair uses σ not μ and explodes). On REAL coedit pairs its observed
  contribution is absorbed into r_obs (frozen). We therefore intervene on staleness
  ONLY in SYNTHETIC mode (sweep stale_rel directly in the prior). This is reported
  honestly — staleness dose-response on real coedit pairs is NOT separable.
* rate_ratio and slope_rel ARE exactly recoverable from the npz, so rate / slope /
  true_occ / n_prior interventions on REAL pairs are faithful.

All state indices and existence weights match the model:
    IDLE,BIRTH,REINFORCE,DECAY,DEATH = 0,1,2,3,4
    existence_decoder w = softplus(theta_init) = [0.1, 1.0, 1.0, 0.3, 0.0]
"""
import numpy as np

# ── model constants (mirror sr_gnn_v3_3.py / fsm_head.py — DO NOT diverge) ──────
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
STATE_NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]
EXISTENCE_W = np.array([0.1, 1.0, 1.0, 0.3, 0.0], dtype=np.float64)  # fsm_head L211

RATE_INIT = 0.1
DECOL_RATE_DEAD_GAMMA = 0.20
DECOL_DEAD_STALE_MULT = 3.0          # _stale_center = 0.5*(1+mult) = 2.0
DECOL_SLOPE_SCALE     = 8.0
_STALE_CENTER = 0.5 * (1.0 + DECOL_DEAD_STALE_MULT)

_EPS = 1e-6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


# ── analytic priors (mirror the decol_hier_v2 block, vectorised, float64) ──────
def _birth_prior(true_occ, n_prior):
    return 4.0 * ((true_occ <= 1.0).astype(np.float64) - 0.5) \
        - 1.5 * (n_prior >= 2.0).astype(np.float64)


def _alive_prior(true_occ, n_prior, rate_ratio, stale_rel):
    has_hist = (n_prior >= 2.0).astype(np.float64)
    recurring = (true_occ >= 2.0).astype(np.float64)
    sustained_dead = has_hist * np.clip(stale_rel - _STALE_CENTER, 0.0, None)
    return 2.0 * recurring - 2.2 * sustained_dead + 0.4 * rate_ratio


def _rising_prior(true_occ, n_prior, slope_rel, stale_rel):
    has_hist = (n_prior >= 2.0).astype(np.float64)
    recurring = (true_occ >= 2.0).astype(np.float64)
    cooling = has_hist * (np.clip(stale_rel - 1.0, 0.0, None)
                          - DECOL_SLOPE_SCALE * np.clip(slope_rel, -5.0, 5.0))
    return 1.2 * recurring - 1.5 * cooling


def _rate_ratio(rate_fast, rate_dead_pp):
    return np.clip(np.log(np.clip(rate_fast, 1e-4, None) / (rate_dead_pp + 1e-4)),
                   -4.0, 4.0)


def _decode_tree(p_birth, p_alive, p_rising):
    """hier-v2 decision tree → (N,5) next-state distribution (renormalised)."""
    nb = 1.0 - p_birth
    P = np.zeros((len(p_birth), 5), dtype=np.float64)
    P[:, BIRTH]     = p_birth
    P[:, REINFORCE] = nb * p_alive * p_rising
    P[:, DECAY]     = nb * p_alive * (1.0 - p_rising)
    P[:, DEATH]     = nb * (1.0 - p_alive)
    return P / P.sum(1, keepdims=True).clip(min=1e-8)


def p_edge(state_dist, w=None):
    """Push a next-state distribution through the REAL existence readout.

    Mirrors ExistenceDecoder.forward: p = w·dist, logit = log(p/(1−p)).
    w (default None ⇒ EXISTENCE_W, the hardcoded INIT spec): pass the per-seed
    TRAINED weights softplus(existence_decoder.theta) to test whether the do(state)
    ladder survives training (reviewer tautology #4 — the init order must not be the
    sole reason DEATH<IDLE<DECAY<REINF=BIRTH holds). Returns (p_edge_prob, logit).
    """
    w_eff = EXISTENCE_W if w is None else np.asarray(w, dtype=np.float64)
    p = (state_dist * w_eff[None, :]).sum(1)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return p, np.log(pc / (1.0 - pc))


class HierV2SCM:
    """Offline counterfactual engine over a hier-v2 faithfulness npz.

    Reconstructs the per-event SCM from dumped PRE-update drivers, recovers the
    frozen residual (trained head + unrecoverable stale term), and exposes do().
    """

    # drivers recoverable on REAL coedit pairs (faithful intervention)
    CLEAN_DRIVERS = ("true_occ", "n_prior", "rate_fast", "slope_rel")
    # synthetic-only (observed value not separable on coedit → absorbed in residual)
    SYNTH_DRIVERS = ("stale_rel",)

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.path = npz_path
        # observed clean drivers
        self.true_occ  = d["true_occ"].astype(np.float64)
        self.n_prior   = d["n_prior"].astype(np.float64)
        self.rate_fast = d["rate_fast"].astype(np.float64)
        self.rate_dead_pp = d["rate_dead_pp"].astype(np.float64)
        self.slope_rel = d["slope_rel"].astype(np.float64)
        self.argmax_target = d["argmax_target"].astype(np.int64)
        # observed gate OUTPUTS (include trained residual)
        self._pg_birth  = d["p_birth_gate"].astype(np.float64)
        self._pg_alive  = d["p_alive_gate"].astype(np.float64)
        self._pg_rising = d["p_rising_gate"].astype(np.float64)
        # raw s_t1_pos (the path that ACTUALLY scores AP today) — for contrast
        self.s_t1_pos = np.stack([d["p_idle"], d["p_birth"], d["p_reinforce"],
                                  d["p_decay"], d["p_death"]], 1).astype(np.float64)
        self.N = len(self.true_occ)

        # rate_ratio is EXACT from npz
        self.rate_ratio = _rate_ratio(self.rate_fast, self.rate_dead_pp)
        # stale_rel is NOT in the npz on coedit → treat as 0 baseline for the
        # analytic prior; its true observed effect is folded into the recovered
        # residual below, so the BASELINE reproduces the observed gates exactly.
        self.stale_rel0 = np.zeros(self.N, dtype=np.float64)

        # ── recover frozen residuals by inverting observed gates ──────────────
        bp = _birth_prior(self.true_occ, self.n_prior)
        ap = _alive_prior(self.true_occ, self.n_prior, self.rate_ratio,
                          self.stale_rel0)
        rp = _rising_prior(self.true_occ, self.n_prior, self.slope_rel,
                           self.stale_rel0)
        self.r_birth  = _logit(self._pg_birth)  - bp
        self.r_alive  = _logit(self._pg_alive)  - ap
        self.r_rising = _logit(self._pg_rising) - rp

    # ── baseline reconstruction (must equal the observed gates / argmax) ──────
    def _gates(self, true_occ, n_prior, rate_ratio, slope_rel, stale_rel,
               r_birth, r_alive, r_rising):
        # residuals (r_*) MUST be sliced to the same idx as the drivers — passing
        # them in (instead of reading self.r_* unsliced) is the shape-mismatch fix.
        pb = _sigmoid(_birth_prior(true_occ, n_prior) + r_birth)
        pa = _sigmoid(_alive_prior(true_occ, n_prior, rate_ratio, stale_rel)
                      + r_alive)
        pr = _sigmoid(_rising_prior(true_occ, n_prior, slope_rel, stale_rel)
                      + r_rising)
        return pb, pa, pr

    def baseline(self, idx=None, w=None):
        """Re-decode at the OBSERVED drivers. Returns dict with state dist + gates.
        w (default None ⇒ init EXISTENCE_W) lets the caller push the SAME baseline
        dist through the per-seed TRAINED existence weights for the do(state) test."""
        sl = slice(None) if idx is None else idx
        pb, pa, pr = self._gates(self.true_occ[sl], self.n_prior[sl],
                                 self.rate_ratio[sl], self.slope_rel[sl],
                                 self.stale_rel0[sl],
                                 self.r_birth[sl], self.r_alive[sl],
                                 self.r_rising[sl])
        dist = _decode_tree(pb, pa, pr)
        pe, _ = p_edge(dist, w=w)
        return dict(dist=dist, p_birth=pb, p_alive=pa, p_rising=pr, p_edge=pe)

    def do_driver(self, idx=None, *, true_occ=None, n_prior=None,
                  rate_fast=None, slope_rel=None, stale_rel=None,
                  rate_ratio=None):
        """do(driver=value) → counterfactual state dist + gates + P(edge).

        Pass an array (broadcastable to idx) or scalar for any driver to CLAMP it;
        unspecified drivers keep their OBSERVED value. Residuals are held fixed.
        `rate_ratio` may be set directly (otherwise derived from rate_fast).
        """
        sl = slice(None) if idx is None else idx
        n = self.N if idx is None else len(np.atleast_1d(self.true_occ[sl]))

        def _pick(val, base):
            if val is None:
                return base.copy()
            return np.broadcast_to(np.asarray(val, dtype=np.float64),
                                   base.shape).astype(np.float64).copy()

        to = _pick(true_occ, self.true_occ[sl])
        npri = _pick(n_prior, self.n_prior[sl])
        sr = _pick(slope_rel, self.slope_rel[sl])
        st = _pick(stale_rel, self.stale_rel0[sl])
        if rate_ratio is not None:
            rr = _pick(rate_ratio, self.rate_ratio[sl])
        elif rate_fast is not None:
            rf = _pick(rate_fast, self.rate_fast[sl])
            rr = _rate_ratio(rf, self.rate_dead_pp[sl])
        else:
            rr = self.rate_ratio[sl].copy()

        # residuals frozen at observed (slice them too)
        rb, ra, rri = self.r_birth[sl], self.r_alive[sl], self.r_rising[sl]
        pb = _sigmoid(_birth_prior(to, npri) + rb)
        pa = _sigmoid(_alive_prior(to, npri, rr, st) + ra)
        pr = _sigmoid(_rising_prior(to, npri, sr, st) + rri)
        dist = _decode_tree(pb, pa, pr)
        pe, _ = p_edge(dist)
        return dict(dist=dist, p_birth=pb, p_alive=pa, p_rising=pr, p_edge=pe)

    def do_state(self, idx=None, *, state, w=None):
        """do(state=X) → force one-hot next-state, push through REAL existence
        readout. `state` is an int (BIRTH/.../DEATH) or one of STATE_NAMES.
        w (default None ⇒ init EXISTENCE_W): pass per-seed TRAINED softplus(theta)
        to test whether the do(state) ladder DEATH<IDLE<DECAY<REINF=BIRTH survives
        training (NOT just the hardcoded init order — reviewer tautology #4).
        Returns P(edge) under the forced state vs baseline."""
        if isinstance(state, str):
            state = STATE_NAMES.index(state.upper())
        sl = slice(None) if idx is None else idx
        n = len(np.atleast_1d(self.true_occ[sl]))
        forced = np.zeros((n, 5), dtype=np.float64)
        forced[:, state] = 1.0
        pe_forced, logit_forced = p_edge(forced, w=w)
        base = self.baseline(idx, w=w)
        return dict(state=STATE_NAMES[state],
                    p_edge_forced=pe_forced, p_edge_base=base["p_edge"],
                    delta=pe_forced - base["p_edge"])
