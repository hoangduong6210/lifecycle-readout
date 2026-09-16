"""CPU probe — hier_causal_policy on s_t1_cal (PM audit fix 2026-06-03).

Verifies (real numbers, no fabrication):
  1. AP-path Δ=0 EXACT: pos_score/neg_score identical with policy ON vs OFF.
  2. never-alive pair asking DEATH: P(DEATH) ~0.9 (OFF) → ~0 (ON), argmax != DEATH.
  3. CAUSAL_RULE_MATRIX: REINFORCE->DEATH suppressed; DECAY->DEATH preserved.
  4. lifecycle does NOT collapse under the mask (5-state dist still spread).
  5. gradient flows to hier heads; no NaN.

The crafted-case checks (2,3) exercise the EXACT policy math (copied verbatim from
the inserted block sr_gnn_v3_3.py) AND are cross-validated against a real forward
pass where ever_alive is forced to 0 (so the in-model block actually runs).
CPU-only.
"""
import os, sys, random
import numpy as np
import torch

V33_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(V33_DIR)
EXP_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, V33_DIR)

from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3
from lifecycle_readout.modeling.fsm_head import (IDLE, BIRTH, REINFORCE, DECAY, DEATH,
                             CAUSAL_RULE_MATRIX)

torch.manual_seed(0); np.random.seed(0); random.seed(0)
DEV = torch.device("cpu")

CFG = dict(design="correct_decoupled", fsm_arch="v3", fsm_decode="hier",
           decol_hier_v2=True, causal_batch=True, lambda_edge_trans=0.5)

N, FEAT, H = 60, 4, 32


def build(policy):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    m = SRGNN_v3_3(N, FEAT, H, device=DEV, hier_causal_policy=policy, **CFG).to(DEV)
    return m


def synth_batch(B, t0, rng):
    src = torch.tensor(rng.integers(0, N, B), dtype=torch.long)
    dst = torch.tensor(rng.integers(0, N, B), dtype=torch.long)
    t = torch.tensor(t0 + np.sort(rng.uniform(0, 5, B)), dtype=torch.float32)
    feat = torch.randn(B, FEAT)
    neg = torch.tensor(rng.integers(0, N, B), dtype=torch.long)
    return src, dst, t, feat, neg


# ── policy transform copied VERBATIM from the inserted block ──────────────────
# SOFT expected-admissibility (PM 2026-06-03, supersedes hard C[argmax(s_t),:]).
_C_FLOOR = 0.05
def apply_policy(s_t1_cal, ever_alive_pos, s_t_pos, C, _hier):
    _ea = ever_alive_pos.clamp(0.0, 1.0)
    _dead_mass = s_t1_cal[:, DEATH]
    _freed_death = (1.0 - _ea) * _dead_mass
    s = s_t1_cal.clone()
    s[:, DEATH] = _ea * _dead_mass
    s[:, IDLE] = s[:, IDLE] + _freed_death
    s = s / s.sum(-1, keepdim=True).clamp(min=1e-8)
    # soft expected admissibility: M[b,j] = Σ_i s_t[b,i]·C[i,j], floor-blended
    _M = torch.einsum("bi,ij->bj", s_t_pos.detach(), C)
    _M = _C_FLOOR + (1.0 - _C_FLOOR) * _M
    s = s * _M
    _denom = s.sum(-1, keepdim=True)
    s = torch.where(_denom > 1e-8, s / _denom.clamp(min=1e-8),
                    _hier / _hier.sum(-1, keepdim=True).clamp(min=1e-8))
    return s


def main():
    C = CAUSAL_RULE_MATRIX.clone()
    print("=" * 72)
    print("CRAFTED CASES — exact policy math (verbatim from inserted block)")
    print("=" * 72)

    def onehot(i):
        v = torch.zeros(1, 5); v[0, i] = 1.0; return v
    # near-uniform s_t with argmax accidentally landing on DEATH (the EXACT input
    # that made hard-C collapse s_t1_cal onto DECAY 0.92).
    nearuni_death_argmax = torch.tensor([[0.198, 0.199, 0.200, 0.201, 0.202]])

    cases = [
        # THE CASE THAT FAILED WITH HARD-C: s_t near-uniform, argmax→DEATH ⇒ hard
        # C[DEATH,:]={0,0,0,1,1} killed everything except DECAY/DEATH ⇒ collapse to
        # DECAY 0.92. With soft mask it must NOT collapse (5-state stays spread).
        ("NEAR-UNIFORM s_t (argmax->DEATH) [hard-C FAIL case]",
         torch.tensor([[0.20, 0.20, 0.20, 0.20, 0.20]]),
         torch.tensor([1.0]), nearuni_death_argmax),
        ("never-alive asks DEATH", torch.tensor([[0.05, 0.04, 0.005, 0.005, 0.90]]),
         torch.tensor([0.0]), onehot(DECAY)),          # s_t=DECAY (DEATH admissible)
        ("REINFORCE -> DEATH (C forbids)", torch.tensor([[0.0, 0.0, 0.50, 0.05, 0.45]]),
         torch.tensor([1.0]), onehot(REINFORCE)),
        ("DECAY -> DEATH (C allows)", torch.tensor([[0.0, 0.0, 0.10, 0.40, 0.50]]),
         torch.tensor([1.0]), onehot(DECAY)),
        ("alive + decaying -> DEATH normal", torch.tensor([[0.0, 0.0, 0.05, 0.45, 0.50]]),
         torch.tensor([1.0]), onehot(DECAY)),
    ]
    names = ["IDLE", "BIRTH", "REINF", "DECAY", "DEATH"]

    # OLD hard-C transform (for explicit contrast on the FAIL case) ───────────
    def apply_hard_C(s_t1_cal, ever_alive_pos, s_t_pos, C, _hier):
        _ea = ever_alive_pos.clamp(0.0, 1.0)
        _dm = s_t1_cal[:, DEATH]; _fd = (1.0 - _ea) * _dm
        s = s_t1_cal.clone(); s[:, DEATH] = _ea * _dm; s[:, IDLE] = s[:, IDLE] + _fd
        s = s / s.sum(-1, keepdim=True).clamp(min=1e-8)
        s = s * C[s_t_pos.detach().argmax(-1)]
        d = s.sum(-1, keepdim=True)
        return torch.where(d > 1e-8, s / d.clamp(min=1e-8),
                           _hier / _hier.sum(-1, keepdim=True).clamp(min=1e-8))

    for label, cal, ea, st in cases:
        cal = cal / cal.sum(-1, keepdim=True)
        out = apply_policy(cal.clone(), ea, st, C, cal.clone())
        spread = int((out[0] > 1e-2).sum())
        print(f"\n[{label}]  ever_alive={float(ea):.0f}  s_t=argmax->{names[int(st.argmax())]}")
        print("  before:", {names[k]: round(float(cal[0, k]), 3) for k in range(5)})
        print("  after :", {names[k]: round(float(out[0, k]), 3) for k in range(5)})
        print(f"  argmax: {names[int(cal.argmax())]} -> {names[int(out.argmax())]}"
              f"  | P(DEATH) {float(cal[0,DEATH]):.3f} -> {float(out[0,DEATH]):.3f}"
              f"  | states>1e-2: {spread}/5")
        if "FAIL case" in label:
            hard = apply_hard_C(cal.clone(), ea, st, C, cal.clone())
            print("  [hard-C] :", {names[k]: round(float(hard[0, k]), 3) for k in range(5)},
                  f"-> argmax {names[int(hard.argmax())]}, states>1e-2: "
                  f"{int((hard[0] > 1e-2).sum())}/5  (this is the COLLAPSE soft-mask fixes)")

    # ── REAL forward A/B on a synthetic stream ────────────────────────────────
    print("\n" + "=" * 72)
    print("REAL FORWARD A/B (config B) — policy OFF vs ON")
    print("=" * 72)
    m_off, m_on = build(False), build(True)
    # warm up identical streams
    rng = np.random.default_rng(7)
    t0 = 0.0
    for _ in range(6):
        sb = synth_batch(200, t0, rng); t0 += 6.0
        with torch.no_grad():
            m_off(*sb); m_on(*sb)

    # scored batch
    sb = synth_batch(300, t0, np.random.default_rng(99))
    o_off = m_off(*sb)
    o_on = m_on(*sb)

    dpos = (o_off["pos_score"] - o_on["pos_score"]).abs().max().item()
    dneg = (o_off["neg_score"] - o_on["neg_score"]).abs().max().item()
    print(f"\nAP-path  max|Δ pos_score| = {dpos:.3e}   max|Δ neg_score| = {dneg:.3e}")
    print("  NOTE: this 2-INSTANCE compare carries build/warm-up nondeterminism "
          "(OFF-vs-OFF control ~1e-3 > this). The CLEAN AP Δ=0 proof is the "
          "WITHIN-INSTANCE check below (deepcopy + flip flag + same state).")
    # ── within-instance: identical model state, flip the flag, same batch → exact ──
    import copy as _copy
    from lifecycle_readout.modeling.fsm_head import CAUSAL_RULE_MATRIX as _C
    _m = build(False)
    _rng = np.random.default_rng(7); _t = 0.0
    for _ in range(6):
        _b = synth_batch(200, _t, _rng); _t += 6.0
        with torch.no_grad():
            _m(*_b)
    _x = synth_batch(300, _t, np.random.default_rng(99))
    _a = _copy.deepcopy(_m); _c = _copy.deepcopy(_m)
    _c.hier_causal_policy = True; _c._hier_causal_C = _C.clone()
    with torch.no_grad():
        _oa = _a(*_x); _oc = _c(*_x)
    _dp = (_oa["pos_score"] - _oc["pos_score"]).abs().max().item()
    _dn = (_oa["neg_score"] - _oc["neg_score"]).abs().max().item()
    _ds = (_oa["s_t1_cal"] - _oc["s_t1_cal"]).abs().max().item()
    print(f"  WITHIN-INSTANCE  max|Δ pos|={_dp:.3e}  max|Δ neg|={_dn:.3e}  "
          f"(s_t1_cal Δ={_ds:.3f})  -> AP Δ=0 EXACT: {_dp == 0.0 and _dn == 0.0}")

    cal_off = o_off["s_t1_cal"]; cal_on = o_on["s_t1_cal"]
    print("\nlifecycle dist (mean over batch):")
    print("  OFF:", {names[k]: round(float(cal_off.mean(0)[k]), 3) for k in range(5)})
    print("  ON :", {names[k]: round(float(cal_on.mean(0)[k]), 3) for k in range(5)})
    am_off = torch.bincount(cal_off.argmax(-1), minlength=5).float() / cal_off.size(0)
    am_on = torch.bincount(cal_on.argmax(-1), minlength=5).float() / cal_on.size(0)
    print("  argmax frac OFF:", {names[k]: round(float(am_off[k]), 3) for k in range(5)})
    print("  argmax frac ON :", {names[k]: round(float(am_on[k]), 3) for k in range(5)})
    nz_on = int((cal_on.mean(0) > 1e-3).sum())
    print(f"  states with >1e-3 mean mass (ON): {nz_on}/5  (collapse if <=1)")

    # ── crafted REAL forward: force ALL ever_alive=0, ask the model on a fresh
    #    batch of pairs the model has 'seen' but we zero ever_alive → never-alive.
    print("\n" + "=" * 72)
    print("REAL never-alive suppression (force ever_alive store -> 0)")
    print("=" * 72)
    sb2 = synth_batch(300, t0 + 50, np.random.default_rng(123))
    with torch.no_grad():
        cal_off2 = m_off(*sb2)["s_t1_cal"]
        m_on.ever_alive.values.zero_()           # force never-alive for ALL keys
        cal_on2 = m_on(*sb2)["s_t1_cal"]
    death_off = cal_off2[:, DEATH]
    death_on = cal_on2[:, DEATH]
    # pairs that OFF would call DEATH (mass>0.5) but are forced never-alive
    mask = death_off > 0.5
    print(f"pairs with OFF P(DEATH)>0.5: {int(mask.sum())}/{mask.numel()}")
    if int(mask.sum()) > 0:
        print(f"  mean P(DEATH)  OFF={float(death_off[mask].mean()):.3f}"
              f"  -> ON(never-alive)={float(death_on[mask].mean()):.3f}")
        argmax_death_on = (cal_on2[mask].argmax(-1) == DEATH).float().mean()
        print(f"  frac still argmax=DEATH after policy: {float(argmax_death_on):.3f}")
    print(f"  overall max P(DEATH) ON (never-alive): {float(death_on.max()):.4f}")

    # ── gradient flow to hier heads ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GRADIENT FLOW (policy ON) — loss.backward, hier heads get grad")
    print("=" * 72)
    m_on.zero_grad()
    out = m_on(*synth_batch(200, t0 + 100, np.random.default_rng(5)))
    loss = out["loss"]
    loss.backward()
    has_nan = bool(torch.isnan(loss.detach()))
    g = m_on.hier_alive_head[-1].weight.grad
    gb = m_on.hier_birth_head[-1].weight.grad
    gr = m_on.hier_rising_head[-1].weight.grad
    def gnorm(x): return float(x.norm()) if x is not None else None
    print(f"  loss={float(loss.detach()):.4f}  NaN={has_nan}")
    print(f"  grad-norm hier_birth={gnorm(gb)}  alive={gnorm(g)}  rising={gnorm(gr)}")


if __name__ == "__main__":
    main()
