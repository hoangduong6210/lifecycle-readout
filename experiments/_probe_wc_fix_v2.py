"""
CPU probe: WC-CONF FIX-1 *REFINE* (PM 2026-06-06) — belief moves by the LEARNED
transition operator T_uv, PROJECTED onto the causal-valid ray; the hand-tuned phase
ANCHOR (true_occ/rate/slope ngưỡng) is RETRACTED.

REAL model compute (no fabricated numbers). Verifies:
  [A] Belief flips by LEARNED dynamics: a learned operator that ADVANCES the rung
      (BIRTH->REINFORCE->DECAY->DEATH) walks the belief forward over events; a learned
      operator that STALLS keeps it; a learned operator that REVERSES walks it back —
      belief is NOT pinned BIRTH and is driven by T_uv, not a threshold.
  [B] Always on the causal RAY: a learned operator that tries to JUMP (BIRTH->DEATH,
      |i-j|>1) is projected back — belief never nhảy-cóc (off-ray mass killed).
  [C] Coherence = % free FOLLOWS the learned-causal lifecycle: free next-state that
      lands inside the LEARNED-on-ray reach -> c HIGH; free that jumps off ray / against
      the learned step -> c LOW (FIX-2 reach-set membership, threshold 0.25, no /max).
  [D] Belief now depends on HISTORY (learned operator x b_prev), not on free-argmax:
      conditional-on-free coherence spread > 0.
  [E] e2e invariants on the REAL model: prediction FREE (Delta=0 vs no-causal),
      backbone 0 link-pred grad, no NaN, flag OFF byte-identical, state_dict keys equal.

The belief-update MATH is replicated here EXACTLY as in sr_gnn_v3_3.py forward (lines
~1694-1752) so we can drive controlled T_uv; [E] then runs the ACTUAL model forward.
"""
import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3, C_BAND_5

torch.manual_seed(0)
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]
C = C_BAND_5.clone()


# ── EXACT replica of the model belief-update (learned-operator + causal projection) ──
def belief_update(b_prev, T_uv, obs, w_obs=0.15):
    """b_prev (B,5), T_uv (B,5,5) learned logit operator, obs (B,5) free observed state.
    Returns (cc_belief, b_step) exactly as the model computes them."""
    _blog   = torch.bmm(b_prev.unsqueeze(1), T_uv).squeeze(1)
    b_learn = torch.softmax(_blog, dim=-1)
    _reach_from = torch.einsum("bi,ij->bj", b_prev, C)
    _reach_from = (_reach_from > 1e-8).float()
    b_step = b_learn * _reach_from
    _den = b_step.sum(-1, keepdim=True)
    pure_c = torch.einsum("bi,ij->bj", b_prev, C)
    pure_c = pure_c / pure_c.sum(-1, keepdim=True).clamp(min=1e-8)
    b_step = torch.where(_den > 1e-8, b_step / _den.clamp(min=1e-8), pure_c)
    obs_n = obs / obs.sum(-1, keepdim=True).clamp(min=1e-8)
    _bt = (b_step.clamp(min=1e-8) ** (1 - w_obs)) * (obs_n.clamp(min=1e-8) ** w_obs)
    _d = _bt.sum(-1, keepdim=True)
    cc_belief = torch.where(_d > 1e-8, _bt / _d.clamp(min=1e-8), b_step)
    return cc_belief, b_step


def coherence(b_prev, s_t1):
    # reach = causal-valid ray reachable in one step from the belief (b_prev ⊙ C),
    # leak-robust 0.25·max threshold — EXACTLY as in sr_gnn_v3_3.py FIX-2.
    ray = torch.einsum("bi,ij->bj", b_prev, C)
    ray = ray / ray.sum(-1, keepdim=True).clamp(min=1e-8)
    rmax = ray.max(-1, keepdim=True).values
    reach = (ray >= 0.25 * rmax).float()
    return (s_t1 * reach).sum(-1).clamp(0, 1), reach


def oh(idx, B=1):
    v = torch.zeros(B, 5); v[:, idx] = 1.0; return v


def advance_T(scale=6.0):
    """Learned operator that pushes each state to its NEXT rung (i -> i+1), DEATH self."""
    T = torch.full((5, 5), -scale)
    for i in range(5):
        j = min(i + 1, 4)
        T[i, j] = scale
    return T.unsqueeze(0)


def stall_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        T[i, i] = scale
    return T.unsqueeze(0)


def reverse_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        j = max(i - 1, 0)
        T[i, j] = scale
    return T.unsqueeze(0)


def jump_T(scale=6.0):
    """Operator that tries BIRTH->DEATH (off-ray) and IDLE->DECAY: tests projection."""
    T = torch.full((5, 5), -scale)
    T[BIRTH, DEATH] = scale
    T[IDLE, DECAY]  = scale
    for i in (REINFORCE, DECAY, DEATH):
        T[i, i] = scale
    return T.unsqueeze(0)


print("=" * 80)
print("[A] BELIEF FLIPS BY LEARNED DYNAMICS (walk forward, stall, reverse) — not pinned")
print("=" * 80)
print("  ADVANCE operator, start belief=BIRTH, iterate (each step b_prev<-cc_belief):")
b = oh(BIRTH)
traj = [NAMES[int(b.argmax())]]
for _ in range(5):
    b, bs = belief_update(b, advance_T(), obs=oh(int(b.argmax())))
    traj.append(NAMES[int(b.argmax())])
print("   trajectory:", " -> ".join(traj))
adv_ok = traj[-1] in ("DECAY", "DEATH") and traj != [traj[0]] * len(traj)

print("  REVERSE operator, start belief=DECAY:")
b = oh(DECAY); rtraj = [NAMES[int(b.argmax())]]
for _ in range(4):
    b, bs = belief_update(b, reverse_T(), obs=oh(int(b.argmax())))
    rtraj.append(NAMES[int(b.argmax())])
print("   trajectory:", " -> ".join(rtraj))
rev_ok = NAMES.index(rtraj[-1]) < NAMES.index(rtraj[0])

print("  STALL operator, start belief=REINFORCE (should hold):")
b = oh(REINFORCE); straj = [NAMES[int(b.argmax())]]
for _ in range(3):
    b, bs = belief_update(b, stall_T(), obs=oh(REINFORCE))
    straj.append(NAMES[int(b.argmax())])
print("   trajectory:", " -> ".join(straj))
stall_ok = set(straj) == {"REINFORCE"}
print(f"  RESULT [A]: advance={adv_ok} reverse={rev_ok} stall={stall_ok} -> "
      f"{'PASS' if (adv_ok and rev_ok and stall_ok) else 'CHECK'}")

print()
print("=" * 80)
print("[B] ALWAYS ON THE CAUSAL RAY: a learned JUMP operator is projected back")
print("=" * 80)
# BIRTH belief, JUMP operator wants BIRTH->DEATH (|1-4|=3, off-ray). After projection the
# belief must stay among BIRTH's ray neighbours {IDLE,BIRTH,REINFORCE}, NOT land on DEATH.
b0 = oh(BIRTH)
b1, bs1 = belief_update(b0, jump_T(), obs=oh(BIRTH))
ray_birth = {IDLE, BIRTH, REINFORCE}
land = int(b1.argmax())
print(f"  BIRTH belief x JUMP(BIRTH->DEATH) op -> belief argmax = {NAMES[land]} "
      f"(off-ray DEATH mass on b_step = {bs1[0, DEATH].item():.4f}; must be ~0)")
b_jump_ok = (land in ray_birth) and (bs1[0, DEATH].item() < 1e-6)
# IDLE belief, JUMP wants IDLE->DECAY (|0-3|=3 off-ray); ray of IDLE = {IDLE,BIRTH}
b2, bs2 = belief_update(oh(IDLE), jump_T(), obs=oh(IDLE))
land2 = int(b2.argmax())
print(f"  IDLE  belief x JUMP(IDLE->DECAY) op  -> belief argmax = {NAMES[land2]} "
      f"(off-ray DECAY mass on b_step = {bs2[0, DECAY].item():.4f}; must be ~0)")
i_jump_ok = (land2 in {IDLE, BIRTH}) and (bs2[0, DECAY].item() < 1e-6)
print(f"  RESULT [B]: {'PASS (no nhay-coc; off-ray mass killed)' if (b_jump_ok and i_jump_ok) else 'CHECK'}")

print()
print("=" * 80)
print("[C] COHERENCE = % FREE FOLLOWS LEARNED-CAUSAL LIFECYCLE (FIX-2 membership)")
print("=" * 80)
# belief BIRTH: causal ray = {IDLE,BIRTH,REINFORCE}. free ON the lifecycle ray
# (REINFORCE = advance, BIRTH = hold) -> HIGH; free that JUMPS off ray (DEATH) -> LOW.
b_prev = oh(BIRTH)
print(f"  belief=BIRTH -> causal ray (admissible one-step) = {{IDLE,BIRTH,REINFORCE}}")
ctab = []
for free, exp in [(REINFORCE, "HIGH"), (BIRTH, "HIGH"), (DEATH, "LOW"), (DECAY, "LOW")]:
    c, reach = coherence(b_prev, oh(free))
    ctab.append((NAMES[free], c.item(), exp))
    print(f"    free={NAMES[free]:<10} c={c.item():.3f}  (reach={[int(x) for x in reach[0].tolist()]})  exp={exp}")
follow_high = [c for n, c, e in ctab if e == "HIGH"]
jump_low    = [c for n, c, e in ctab if e == "LOW"]
c_ok = min(follow_high) >= 0.5 and max(jump_low) < 0.5
# teleport from a never-born IDLE belief firing DEATH (the headline nhay-coc)
c_tele, _ = coherence(oh(IDLE), oh(DEATH))
print(f"  TELEPORT: never-born(belief=IDLE)->free DEATH  c={c_tele.item():.3f} (must be LOW)")
c_ok = c_ok and c_tele.item() < 0.5
print(f"  RESULT [C]: {'PASS (follow->HIGH, jump/teleport->LOW)' if c_ok else 'CHECK'}")

print()
print("=" * 80)
print("[D] COHERENCE depends on HISTORY (learned op x belief), NOT free-argmax alone")
print("=" * 80)
# fix FREE next-state, vary the BELIEF (and its learned step). If c were f(free) it would
# be constant; learned-on-ray membership makes it swing with the walked history.
print("  fix FREE=DECAY, vary belief: c must swing (DECAY on-ray only for REINFORCE/DECAY/DEATH)")
spread = []
for bel in range(5):
    c, _ = coherence(oh(bel), oh(DECAY))
    spread.append(c.item())
    print(f"    belief={NAMES[bel]:<10} -> c(free=DECAY)={c.item():.3f}")
sp = max(spread) - min(spread)
print(f"  conditional spread over belief at fixed free = {sp:.3f}  "
      f"(>0 => c NOT a function of free-argmax alone)")
print(f"  RESULT [D]: {'PASS' if sp > 0.3 else 'CHECK'}")

print()
print("=" * 80)
print("[E] e2e on the REAL model: FREE prediction, 0 backbone link-pred grad, byte-id OFF")
print("=" * 80)
BASE = dict(num_nodes=64, feat_dim=4, hidden=16, design="correct_decoupled",
            fsm_arch="v3", fsm_decode="hier", decol_hier_v2=True, causal_batch=True,
            lambda_edge_trans=0.5)


def build_WC(thr=0.0):
    return SRGNN_v3_3(**BASE, causal_confidence=True, cc_C="band", cc_thr=thr)


def build_FREE():
    return SRGNN_v3_3(**BASE, causal_confidence=False)


torch.manual_seed(3)
Bsz = 24
src = torch.randint(0, 64, (Bsz,))
dst = (src + 1 + torch.randint(0, 60, (Bsz,))) % 64
ts = torch.sort(torch.rand(Bsz) * 100.0).values
ft = torch.randn(Bsz, 4)
nd = torch.randint(0, 64, (Bsz,))
ref = build_WC().state_dict()


def run(builder):
    mm = builder(); mm.load_state_dict(ref, strict=False)
    mm.train(); mm.set_epoch(10); mm.reset()
    return mm, mm(src, dst, ts.clone(), ft.clone(), nd.clone())


mWC, oWC = run(build_WC)
mFR, oFR = run(build_FREE)
dpos = (oWC["pos_score"] - oFR["pos_score"]).abs().max().item()
dneg = (oWC["neg_score"] - oFR["neg_score"]).abs().max().item()
print(f"  max|pos_score WC-FREE|={dpos:.3e}  max|neg|={dneg:.3e}  (AP FREE, Delta=0)")
mWC.zero_grad(set_to_none=True); oWC["loss"].backward()
bbWC = sum(float(p.grad.pow(2).sum()) for n, p in mWC.named_parameters()
           if any(h in n for h in ("csn", "ectg", "drgc")) and p.grad is not None) ** 0.5
mFR.zero_grad(set_to_none=True); oFR["loss"].backward()
bbFR = sum(float(p.grad.pow(2).sum()) for n, p in mFR.named_parameters()
           if any(h in n for h in ("csn", "ectg", "drgc")) and p.grad is not None) ** 0.5
print(f"  backbone full-loss grad-norm: WC={bbWC:.6e}  FREE={bbFR:.6e}  "
      f"max|WC-FREE|={abs(bbWC-bbFR):.3e} (WC adds 0 backbone grad; echo-KL only)")
print(f"  loss finite={torch.isfinite(oWC['loss']).item()}  loss={oWC['loss'].item():.4f}")
kWC = set(build_WC().state_dict().keys()); kFR = set(build_FREE().state_dict().keys())
print(f"  state_dict keys WC={len(kWC)} FREE={len(kFR)} symdiff={len(kWC ^ kFR)} (expect 0)")
mDEF = SRGNN_v3_3(**BASE); mDEF.load_state_dict(ref, strict=False)
mDEF.train(); mDEF.set_epoch(10); mDEF.reset()
oDEF = mDEF(src, dst, ts.clone(), ft.clone(), nd.clone())
print(f"  default(no flag) vs explicit-off max|pos|="
      f"{(oDEF['pos_score']-oFR['pos_score']).abs().max().item():.3e}")
cc = oWC["cc_coherence"]
print(f"  cc_coherence this batch: min={cc.min():.3f} max={cc.max():.3f} mean={cc.mean():.3f}")
e_ok = (dpos < 1e-6 and dneg < 1e-6 and abs(bbWC - bbFR) < 1e-6
        and torch.isfinite(oWC['loss']).item() and len(kWC ^ kFR) == 0)
print(f"  RESULT [E]: {'PASS' if e_ok else 'CHECK'}")
print("\nDONE.")
