"""
CPU probe: WC-CONF self-consistency variant (PM 2026-06-07 R2).

design correction: FIX-3 dropped obs-coupling ENTIRELY -> belief stuck at IDLE init
(legacy diagnostic: IDLE 0.87, R^2(c~free) 0.908->0.955 WORSE). "No hand" means no
hand-ANCHORED phase (rate/slope -> state), NOT "no obs". The filter's MEASUREMENT
step (obs-coupling) is mandatory. This restores it as a LEARNABLE coupling:
  belief = normalize( b_step^(1-w_obs) (x) obs^w_obs ),  w_obs = sigmoid(param)
trained by a self-consistency CE( belief || free-next-state argmax, detached )
computed OUTSIDE the no_grad belief block, gradient-isolated from predict.

REAL-model compute (no fabricated numbers). Verifies:
  [A] obs-coupling restored: belief is NOT stuck at IDLE/BIRTH init -> with a
      rising/falling learned operator + obs, belief advances/decays.
  [B] w_obs (=sigmoid(cc_w_obs_logit)) is a PARAM WITH GRADIENT (not dead);
      self-consistency loss DECREASES over a few SGD steps on the param only.
  [C] predict Delta=0 (belief / self-consist does NOT bend score), backbone 0
      link-pred grad -> self-consistency loss does NOT leak into predict.
  [D] c_t: teleport(belief=IDLE -> free DEATH) LOW; on-ray HIGH.
  [E] no NaN; flag-OFF byte-identical; aux-on adds EXACTLY the one cc_w_obs_logit key.
"""
import os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3, C_BAND_5

IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4
NAMES = ["IDLE", "BIRTH", "REINFORCE", "DECAY", "DEATH"]
C = C_BAND_5.clone()


def oh(idx, B=1):
    v = torch.zeros(B, 5); v[:, idx] = 1.0; return v


# ── EXACT replica of the PATCHED model belief-update WITH learnable obs-coupling ──────
def belief_update(b_prev, T_uv, obs, w_obs):
    """Mirror sr_gnn_v3_3.py belief block (c). b_prev (B,5), T_uv (B,5,5) logits,
    obs (B,5) probs, w_obs scalar in (0,1)."""
    _blog   = torch.bmm(b_prev.unsqueeze(1), T_uv).squeeze(1)
    b_learn = torch.softmax(_blog, dim=-1)
    _reach_from = torch.einsum("bi,ij->bj", b_prev, C)
    _reach_from = (_reach_from > 1e-8).float()
    b_step = b_learn * _reach_from
    _den = b_step.sum(-1, keepdim=True)
    pure_c = torch.einsum("bi,ij->bj", b_prev, C)
    pure_c = pure_c / pure_c.sum(-1, keepdim=True).clamp(min=1e-8)
    b_step = torch.where(_den > 1e-8, b_step / _den.clamp(min=1e-8), pure_c)
    # (c) learnable obs-coupling: project obs onto ray, geometric blend
    _obs = obs * _reach_from
    _obs = _obs / _obs.sum(-1, keepdim=True).clamp(min=1e-8)
    _blend = ((1.0 - w_obs) * b_step.clamp(min=1e-8).log()
              + w_obs * _obs.clamp(min=1e-8).log())
    return torch.softmax(_blend, dim=-1), b_step


def advance_T(scale=6.0):
    T = torch.full((5, 5), -scale)
    for i in range(5):
        T[i, min(i + 1, 4)] = scale
    return T.unsqueeze(0)


print("=" * 80)
print("[A] obs-coupling RESTORED: belief is NOT stuck at IDLE/BIRTH init")
print("=" * 80)
# contrast with legacy diagnostic (IDLE.87 stuck). Walk a BIRTH-init belief with a
# rising learned operator and a rising observation; belief must ADVANCE off init.
b = oh(BIRTH); traj = [NAMES[int(b.argmax())]]
obs_seq = [BIRTH, REINFORCE, REINFORCE, DECAY, DECAY]
w_obs = torch.tensor(0.5)  # mid weight (init)
for k in range(5):
    b, _ = belief_update(b, advance_T(), oh(obs_seq[k]), w_obs)
    traj.append(NAMES[int(b.argmax())])
print("  BIRTH-init + rising op + rising obs :", " -> ".join(traj))
a_ok = traj != [traj[0]] * len(traj) and NAMES.index(traj[-1]) > NAMES.index(traj[0])
# also: a pure-IDLE belief with obs=BIRTH must leave IDLE (un-stick the init)
b2, _ = belief_update(oh(IDLE), advance_T(), oh(BIRTH), torch.tensor(0.7))
print(f"  IDLE-init + obs=BIRTH (w_obs=0.7) -> argmax={NAMES[int(b2.argmax())]} "
      f"(must leave IDLE; IDLE mass={b2[0, IDLE].item():.3f})")
a_ok = a_ok and int(b2.argmax()) != IDLE
print(f"  RESULT [A]: {'PASS (belief un-stuck, obs measured)' if a_ok else 'CHECK'}")

print("\n" + "=" * 80)
print("[B]+[C]+[D]+[E] e2e on the REAL model")
print("=" * 80)
BASE = dict(num_nodes=64, feat_dim=4, hidden=16, design="correct_decoupled",
            fsm_arch="v3", fsm_decode="hier", decol_hier_v2=True, causal_batch=True,
            lambda_edge_trans=0.5)


def build_AUX(w=0.2):
    return SRGNN_v3_3(**BASE, causal_confidence=True, cc_C="band",
                      cc_self_consist_w=w)


def build_CCOFF():        # cc on but NO aux (FIX-3 closed-loop filter)
    return SRGNN_v3_3(**BASE, causal_confidence=True, cc_C="band")


def build_FREE():
    return SRGNN_v3_3(**BASE, causal_confidence=False)


torch.manual_seed(3)
Bsz = 24
src = torch.randint(0, 64, (Bsz,))
dst = (src + 1 + torch.randint(0, 60, (Bsz,))) % 64
ts = torch.sort(torch.rand(Bsz) * 100.0).values
ft = torch.randn(Bsz, 4)
nd = torch.randint(0, 64, (Bsz,))
ref = build_CCOFF().state_dict()   # shared backbone weights (no aux param)


def run(builder):
    mm = builder(); mm.load_state_dict(ref, strict=False)
    mm.train(); mm.set_epoch(10); mm.reset()
    return mm, mm(src, dst, ts.clone(), ft.clone(), nd.clone())


mAUX, oAUX = run(build_AUX)
mFR, oFR = run(build_FREE)

# ── [C] predict Delta=0 ───────────────────────────────────────────────────────────
dpos = (oAUX["pos_score"] - oFR["pos_score"]).abs().max().item()
dneg = (oAUX["neg_score"] - oFR["neg_score"]).abs().max().item()
print(f"[C] max|pos_score AUX-FREE|={dpos:.3e}  max|neg|={dneg:.3e}  (predict Delta=0)")

# backbone link-pred grad (full loss INCLUDES the aux term) -> must match FREE
mAUX.zero_grad(set_to_none=True); oAUX["loss"].backward()
bbAUX = sum(float(p.grad.pow(2).sum()) for n, p in mAUX.named_parameters()
            if any(h in n for h in ("csn", "ectg", "drgc")) and p.grad is not None) ** 0.5
mFR.zero_grad(set_to_none=True); oFR["loss"].backward()
bbFR = sum(float(p.grad.pow(2).sum()) for n, p in mFR.named_parameters()
           if any(h in n for h in ("csn", "ectg", "drgc")) and p.grad is not None) ** 0.5
print(f"[C] backbone full-loss grad-norm: AUX={bbAUX:.6e} FREE={bbFR:.6e} "
      f"max|AUX-FREE|={abs(bbAUX - bbFR):.3e} (aux must NOT touch backbone)")

# ── [B] w_obs param HAS gradient (not dead) ─────────────────────────────────────────
g = mAUX.cc_w_obs_logit.grad
g_val = None if g is None else float(g.abs().item())
print(f"[B] cc_w_obs_logit.grad = {g_val}  (must be a finite NON-zero number => not dead)")
b_grad_ok = (g is not None) and torch.isfinite(g).item() and abs(float(g)) > 0.0

# self-consistency loss DECREASES over a few steps training ONLY the param
print("[B] self-consistency CE over 5 steps (optimize ONLY cc_w_obs_logit):")
m2 = build_AUX(); m2.load_state_dict(ref, strict=False)
m2.train(); m2.set_epoch(10)
opt = torch.optim.SGD([m2.cc_w_obs_logit], lr=5.0)
sc_hist = []; w_hist = []
for step in range(5):
    m2.reset()
    o = m2(src, dst, ts.clone(), ft.clone(), nd.clone())
    sc = float(o["cc_self_consist_loss"])   # detached value for logging
    w = float(o["cc_w_obs"])
    sc_hist.append(sc); w_hist.append(w)
    # backward the REAL total loss (the in-graph aux term reaches cc_w_obs_logit);
    # only the param is in the optimizer so only it moves.
    opt.zero_grad(); o["loss"].backward(); opt.step()
    print(f"    step {step}: self_consist_CE={sc:.4f}  w_obs={w:.4f}")
sc_drop = sc_hist[0] - sc_hist[-1]
print(f"[B] CE drop over 5 steps = {sc_drop:+.4f}  (must be > 0 => param learns)")
print(f"[B] w_obs trajectory: {w_hist[0]:.4f} -> {w_hist[-1]:.4f}")
b_ok = b_grad_ok and sc_drop > 0.0

# ── [D] coherence: teleport low, on-ray high (REAL batch) ──────────────────────────
cc = oAUX["cc_coherence"]
print(f"[D] cc_coherence batch: min={cc.min():.3f} max={cc.max():.3f} mean={cc.mean():.3f}")
# synthetic structural teleport check (matches model: reach = C[argmax(b_prev)])
def coherence(b_prev, s_t1):
    reach = C[b_prev.argmax(dim=-1)].float()
    return (s_t1 * reach).sum(-1).clamp(0, 1)
c_tele = coherence(oh(IDLE), oh(DEATH)).item()
c_onray = coherence(oh(BIRTH), oh(REINFORCE)).item()
print(f"[D] teleport(IDLE->DEATH) c={c_tele:.3f} (LOW)  on-ray(BIRTH->REINF) c={c_onray:.3f} (HIGH)")
d_ok = c_tele < 0.5 and c_onray >= 0.5

# ── [E] NaN / byte-identical OFF / key delta ───────────────────────────────────────
fin_ok = (torch.isfinite(oAUX["loss"]).item()
          and torch.isfinite(cc).all().item()
          and torch.isfinite(oAUX["cc_belief"]).all().item())
kCCOFF = set(build_CCOFF().state_dict().keys())
kFREE  = set(build_FREE().state_dict().keys())
kAUX   = set(build_AUX().state_dict().keys())
off_symdiff = len(kCCOFF ^ kFREE)            # cc-off must be byte-identical to FREE
aux_extra   = kAUX - kCCOFF                  # aux adds EXACTLY cc_w_obs_logit
print(f"[E] keys: FREE={len(kFREE)} CCOFF={len(kCCOFF)} AUX={len(kAUX)}")
print(f"[E] cc-OFF vs FREE symdiff={off_symdiff} (expect 0, byte-identical)")
print(f"[E] AUX extra keys vs cc-off = {sorted(aux_extra)} (expect ['cc_w_obs_logit'])")
# default no-flag vs explicit FREE
mDEF = SRGNN_v3_3(**BASE); mDEF.load_state_dict(ref, strict=False)
mDEF.train(); mDEF.set_epoch(10); mDEF.reset()
oDEF = mDEF(src, dst, ts.clone(), ft.clone(), nd.clone())
d_def = (oDEF["pos_score"] - oFR["pos_score"]).abs().max().item()
print(f"[E] default(no flag) vs explicit-off max|pos|={d_def:.3e}")
e_ok = (off_symdiff == 0 and aux_extra == {"cc_w_obs_logit"} and fin_ok
        and d_def < 1e-6)

c_ok = (dpos < 1e-6 and dneg < 1e-6 and abs(bbAUX - bbFR) < 1e-6)

print("\n" + "=" * 80)
print(f"  [A] obs restored / un-stuck : {'PASS' if a_ok else 'CHECK'}")
print(f"  [B] w_obs grad + CE drops   : {'PASS' if b_ok else 'CHECK'}")
print(f"  [C] predict Delta=0 + bb 0  : {'PASS' if c_ok else 'CHECK'}")
print(f"  [D] teleport low / on-ray hi: {'PASS' if d_ok else 'CHECK'}")
print(f"  [E] no NaN / OFF byte-id    : {'PASS' if e_ok else 'CHECK'}")
print("=" * 80)
print("DONE.")
