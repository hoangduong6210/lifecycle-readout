"""
CPU probe: WC-CONF FIX-3 *ZERO HAND-CONST* (PM 2026-06-07).

Removes the LAST two hand constants from the belief block:
  (1) obs-nudge w=0.15  -> DROPPED entirely. Belief = PURE learned-forward filter.
      The measurement enters via the LEARNED operator T_uv(phi) (re-derived per event
      from the pair's live continuous history), NOT via a hand-weighted obs term.
      Rejected a learnable scalar: the belief block is under torch.no_grad() and
      detached from predict (Delta=0, backbone 0 link-pred grad) -> any param mixed in
      here would be a DEAD param (no gradient). Clean filter = no const, no dead param.
  (2) reach-threshold 0.25*max -> REPLACED by STRUCTURAL C-adjacency:
      reach_mask = C[argmax(b_prev)]  (the belief's rung's row of the band matrix).
      Pure adjacency read off C; no magnitude cut.

REAL model compute (no fabricated numbers). Verifies:
  [A] Belief flips by LEARNED dynamics (advance/stall/reverse), not pinned, no obs.
  [B] Always on the causal RAY: a learned JUMP is projected back (no nhay-coc).
  [C] Coherence = % free follows the learned-causal lifecycle; teleport -> LOW.
  [D] Coherence depends on HISTORY (belief), not on free-argmax alone.
  [E] e2e on the REAL model: prediction FREE (Delta=0), backbone 0 link-pred grad,
      no NaN, flag OFF byte-identical, state_dict keys equal.
The belief-update MATH is replicated here EXACTLY as in the patched sr_gnn_v3_3.py.
"""
import os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3, C_BAND_5

torch.manual_seed(0)
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]
C = C_BAND_5.clone()


# ── EXACT replica of the PATCHED model belief-update (NO obs nudge) ──────────────────
def belief_update(b_prev, T_uv):
    """b_prev (B,5), T_uv (B,5,5) learned logit operator. Returns (cc_belief, b_step).
    NO obs argument anymore: belief = pure learned-forward step projected on the ray."""
    _blog   = torch.bmm(b_prev.unsqueeze(1), T_uv).squeeze(1)
    b_learn = torch.softmax(_blog, dim=-1)
    _reach_from = torch.einsum("bi,ij->bj", b_prev, C)
    _reach_from = (_reach_from > 1e-8).float()
    b_step = b_learn * _reach_from
    _den = b_step.sum(-1, keepdim=True)
    pure_c = torch.einsum("bi,ij->bj", b_prev, C)
    pure_c = pure_c / pure_c.sum(-1, keepdim=True).clamp(min=1e-8)
    b_step = torch.where(_den > 1e-8, b_step / _den.clamp(min=1e-8), pure_c)
    cc_belief = b_step                       # (c) NO nudge: belief = b_step exactly
    return cc_belief, b_step


def coherence(b_prev, s_t1):
    # STRUCTURAL reach = C-adjacency of the belief's argmax rung (no magnitude threshold).
    pos = b_prev.argmax(dim=-1)
    reach = C[pos].float()
    return (s_t1 * reach).sum(-1).clamp(0, 1), reach


def oh(idx, B=1):
    v = torch.zeros(B, 5); v[:, idx] = 1.0; return v


def advance_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        T[i, min(i + 1, 4)] = scale
    return T.unsqueeze(0)


def stall_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        T[i, i] = scale
    return T.unsqueeze(0)


def reverse_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        T[i, max(i - 1, 0)] = scale
    return T.unsqueeze(0)


def jump_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    T[BIRTH, DEATH] = scale
    T[IDLE, DECAY]  = scale
    for i in (REINFORCE, DECAY, DEATH):
        T[i, i] = scale
    return T.unsqueeze(0)


print("=" * 80)
print("[A] BELIEF FLIPS BY LEARNED DYNAMICS (advance/stall/reverse) — NO obs nudge")
print("=" * 80)
b = oh(BIRTH); traj = [NAMES[int(b.argmax())]]
for _ in range(5):
    b, _ = belief_update(b, advance_T()); traj.append(NAMES[int(b.argmax())])
print("  ADVANCE  start=BIRTH :", " -> ".join(traj))
adv_ok = traj[-1] in ("DECAY", "DEATH") and traj != [traj[0]] * len(traj)

b = oh(DECAY); rtraj = [NAMES[int(b.argmax())]]
for _ in range(4):
    b, _ = belief_update(b, reverse_T()); rtraj.append(NAMES[int(b.argmax())])
print("  REVERSE  start=DECAY :", " -> ".join(rtraj))
rev_ok = NAMES.index(rtraj[-1]) < NAMES.index(rtraj[0])

b = oh(REINFORCE); straj = [NAMES[int(b.argmax())]]
for _ in range(3):
    b, _ = belief_update(b, stall_T()); straj.append(NAMES[int(b.argmax())])
print("  STALL    start=REINF :", " -> ".join(straj))
stall_ok = set(straj) == {"REINFORCE"}
print(f"  RESULT [A]: advance={adv_ok} reverse={rev_ok} stall={stall_ok} -> "
      f"{'PASS' if (adv_ok and rev_ok and stall_ok) else 'CHECK'}")

print("\n" + "=" * 80)
print("[B] ALWAYS ON THE CAUSAL RAY: a learned JUMP operator is projected back")
print("=" * 80)
b1, bs1 = belief_update(oh(BIRTH), jump_T())
land = int(b1.argmax())
print(f"  BIRTH x JUMP(BIRTH->DEATH) -> argmax={NAMES[land]} "
      f"(off-ray DEATH mass on b_step={bs1[0, DEATH].item():.4f}; must ~0)")
b_jump_ok = (land in {IDLE, BIRTH, REINFORCE}) and (bs1[0, DEATH].item() < 1e-6)
b2, bs2 = belief_update(oh(IDLE), jump_T())
land2 = int(b2.argmax())
print(f"  IDLE  x JUMP(IDLE->DECAY)  -> argmax={NAMES[land2]} "
      f"(off-ray DECAY mass on b_step={bs2[0, DECAY].item():.4f}; must ~0)")
i_jump_ok = (land2 in {IDLE, BIRTH}) and (bs2[0, DECAY].item() < 1e-6)
print(f"  RESULT [B]: {'PASS (no nhay-coc; off-ray killed)' if (b_jump_ok and i_jump_ok) else 'CHECK'}")

print("\n" + "=" * 80)
print("[C] COHERENCE = % FREE FOLLOWS LEARNED-CAUSAL LIFECYCLE (structural reach)")
print("=" * 80)
b_prev = oh(BIRTH)
print("  belief=BIRTH -> structural reach = C[BIRTH] = {IDLE,BIRTH,REINFORCE}")
ctab = []
for free, exp in [(REINFORCE, "HIGH"), (BIRTH, "HIGH"), (DEATH, "LOW"), (DECAY, "LOW")]:
    c, reach = coherence(b_prev, oh(free))
    ctab.append((NAMES[free], c.item(), exp))
    print(f"    free={NAMES[free]:<10} c={c.item():.3f} reach={[int(x) for x in reach[0].tolist()]} exp={exp}")
follow_high = [c for _, c, e in ctab if e == "HIGH"]
jump_low    = [c for _, c, e in ctab if e == "LOW"]
c_ok = min(follow_high) >= 0.5 and max(jump_low) < 0.5
c_tele, _ = coherence(oh(IDLE), oh(DEATH))
print(f"  TELEPORT: never-born(belief=IDLE)->free DEATH  c={c_tele.item():.3f} (must be LOW)")
c_ok = c_ok and c_tele.item() < 0.5
print(f"  RESULT [C]: {'PASS' if c_ok else 'CHECK'}")

print("\n" + "=" * 80)
print("[D] COHERENCE depends on HISTORY (belief), NOT free-argmax alone")
print("=" * 80)
print("  fix FREE=DECAY, vary belief: c must swing (DECAY in C-row of REINFORCE/DECAY/DEATH)")
spread = []
for bel in range(5):
    c, _ = coherence(oh(bel), oh(DECAY))
    spread.append(c.item())
    print(f"    belief={NAMES[bel]:<10} -> c(free=DECAY)={c.item():.3f}")
sp = max(spread) - min(spread)
print(f"  conditional spread = {sp:.3f} (>0 => c NOT f(free-argmax))")
print(f"  RESULT [D]: {'PASS' if sp > 0.3 else 'CHECK'}")

print("\n" + "=" * 80)
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
      f"max|WC-FREE|={abs(bbWC-bbFR):.3e}")
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
nan_ok = torch.isfinite(cc).all().item() and torch.isfinite(oWC["cc_belief"]).all().item()
e_ok = (dpos < 1e-6 and dneg < 1e-6 and abs(bbWC - bbFR) < 1e-6
        and torch.isfinite(oWC['loss']).item() and len(kWC ^ kFR) == 0 and nan_ok)
print(f"  cc finite={nan_ok}")
print(f"  RESULT [E]: {'PASS' if e_ok else 'CHECK'}")
print("\nDONE.")
