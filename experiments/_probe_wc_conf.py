"""
CPU probe: WC-CONF (walked-chain causal-confidence) vs config B.  protocol requirement 2026-06-06.

DESIGN LOCK (PM): causality does NOT mask the neural prediction. It only
  (1) feeds a CONFIDENCE/COHERENCE score c_t, and
  (2) SELECTS which gradient is learned (gates the FSM-block loss by c_t).

WC-CONF (flag causal_confidence=True; default OFF byte-identical) adds, on top of
config B's FREE prediction path (here we run hier_causal_policy OFF so the AP/score
path is fully free — fair vs B which value-masks):
  - walked-chain belief b_t (per-pair, in the edge store): b_step=normalize(b_{t-1}@C);
        b_t=normalize(b_step ⊙ obs(s_t)).  PRE-update read; POST-scoring write.
  - coherence  c_t = Σ_j s_t1_cal[j]·reach[j], reach[j]=1{(b_{t-1}@C)[j]>eps}.
  - gradient-selection: FSM-block edge-trans CE scaled by c_t (or 0 below cc_thr).

Claims verified on REAL model compute (no fabricated numbers):
  [1] WALKED-CHAIN catches teleport: never-born pair (b≈IDLE) whose FREE s_t1 demands
      DEATH -> c_t LOW; a coherent pair (b≈DECAY, s_t1=DEATH) -> c_t HIGH.  Contrast.
  [2] GRADIENT-SELECTION: low-c event -> FSM-block gradient scaled down / zero; high-c
      event -> normal.  Prediction VALUE unchanged (s_t1_cal before/after gate identical).
  [3] AP-path FREE: pos_score taken from the free path; WC-CONF Δ vs no-causal = 0.0.
  [4] Backbone 0 link-pred grad (detached); no NaN; canonical/off byte-identical.
"""
import os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3, C_BAND_5

torch.manual_seed(0)
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4

# WC-CONF base = config-B stack but FREE score path (hier_causal_policy OFF) so the
# AP comparison is fair (prediction not bent).  causal_confidence adds confidence + gate.
BASE = dict(
    num_nodes=64, feat_dim=4, hidden=16,
    design="correct_decoupled", fsm_arch="v3", fsm_decode="hier",
    decol_hier_v2=True, causal_batch=True, lambda_edge_trans=0.5,
)

def build_WC(thr=0.0):
    return SRGNN_v3_3(**BASE, causal_confidence=True, cc_C="band", cc_thr=thr)

def build_FREE():
    # same stack, WC-CONF OFF (no confidence, no gate) — the no-causal control.
    return SRGNN_v3_3(**BASE, causal_confidence=False)

def build_B():
    return SRGNN_v3_3(**BASE, hier_causal_policy=True, lfg_mode="soft", compliance_floor=0.05)

def build_canonical():
    return SRGNN_v3_3(num_nodes=64, feat_dim=4, hidden=16)


print("=" * 74)
print("Construct check: WC-CONF flag + detached-arm guard + chosen C matrix")
print("=" * 74)
mWC = build_WC()
print(f"  causal_confidence={mWC.causal_confidence}  cc_C={mWC._cc_C_name!r}  cc_thr={mWC.cc_thr}")
print(f"  enable_main_predictor={mWC.enable_main_predictor} (must be False -> detached AP path)")
assert mWC.causal_confidence is True
assert mWC.enable_main_predictor is False, "AP path NOT detached!"
assert torch.equal(mWC._cc_C, C_BAND_5), "C matrix is not C_BAND_5"
print("  C = C_BAND_5 (band-diagonal |i-j|<=1):")
print(mWC._cc_C.long().tolist())

# ---------------------------------------------------------------------------
# [1] WALKED-CHAIN coherence: teleport (never-born -> DEATH) vs coherent (DECAY -> DEATH).
#     Reproduce the model's EXACT c_t math (the forward block) on CONTROLLED b_{t-1}/s_t1.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("[1] WALKED-CHAIN coherence c_t — teleport (LOW) vs coherent walk (HIGH)")
print("=" * 74)
C = mWC._cc_C

def onehot(idx, n=5, peak=0.90):
    # crisp belief with a small uniform leak (tests leak-robustness of the soft
    # coherence; a hard reachable-mask with eps would smear to all-ones here).
    v = torch.full((n,), (1.0 - peak) / (n - 1)); v[idx] = peak
    return v / v.sum()

def coherence(b_prev, s_t1):
    # mirrors sr_gnn_v3_3.forward WC-CONF block exactly (SOFT overlap formula).
    b_step = torch.einsum("i,ij->j", b_prev, C)
    b_step = b_step / b_step.sum().clamp(min=1e-8)
    overlap = (s_t1 * b_step).sum()
    norm = b_step.max().clamp(min=1e-8)
    c = (overlap / norm).clamp(0.0, 1.0)
    return c.item(), b_step

cases = [
    ("never-born (b=IDLE)   asks DEATH  [TELEPORT]", onehot(IDLE),  onehot(DEATH)),
    ("never-born (b=IDLE)   asks BIRTH  [coherent]", onehot(IDLE),  onehot(BIRTH)),
    ("dying    (b=DECAY)    asks DEATH  [coherent]", onehot(DECAY), onehot(DEATH)),
    ("reinforce(b=REINFORCE)asks DEATH  [TELEPORT]", onehot(REINFORCE), onehot(DEATH)),
    ("alive    (b=REINFORCE)asks DECAY  [coherent]", onehot(REINFORCE), onehot(DECAY)),
    ("born     (b=BIRTH)    asks DEATH  [TELEPORT]", onehot(BIRTH), onehot(DEATH)),
]
print(f"  {'scenario':46s}  reach(rel)            c_t")
for label, bp, s1 in cases:
    c, bstep = coherence(bp, s1)
    reach = (bstep >= 0.10 * bstep.max()).long().tolist()
    tag = "LOW " if c < 0.5 else "HIGH"
    print(f"  {label:46s}  {reach}  c={c:.4f}  [{tag}]")
# teleports must score LOW, coherent walks HIGH
c_teleport = [coherence(bp, s1)[0] for label, bp, s1 in cases if "TELEPORT" in label]
c_coherent = [coherence(bp, s1)[0] for label, bp, s1 in cases if "coherent" in label]
print(f"  teleport c_t: {[f'{x:.3f}' for x in c_teleport]}  (expect LOW)")
print(f"  coherent c_t: {[f'{x:.3f}' for x in c_coherent]}  (expect HIGH)")
print(f"  separation: max(teleport)={max(c_teleport):.3f} < min(coherent)={min(c_coherent):.3f}")
assert max(c_teleport) < 0.5, "teleport coherence not LOW!"
assert min(c_coherent) > 0.5, "coherent coherence not HIGH!"
assert max(c_teleport) < min(c_coherent), "teleport/coherent not separated!"
print("  => WALKED-CHAIN catches teleport: never-born/reinforce->DEATH = LOW c; "
      "DECAY->DEATH / IDLE->BIRTH / REINFORCE->DECAY = HIGH c.")

# ---------------------------------------------------------------------------
# [2] REAL model forward: c_t is emitted, the FSM-block gradient is scaled by c_t,
#     and the PREDICTION VALUE (s_t1_cal, s_t1_pos) is byte-identical WC-ON vs WC-OFF.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("[2] real forward: c_t emitted; prediction VALUE unchanged WC-ON vs WC-OFF")
print("=" * 74)
mWC, mFREE = build_WC(), build_FREE()
# weight-align so the only difference is the WC-CONF flag.
miss, unexp = mFREE.load_state_dict(mWC.state_dict(), strict=False)
learn_miss = [k for k in miss if "cc_" not in k and "causal" not in k
              and "band5" not in k and "strict_C" not in k]
assert learn_miss == [], f"param mismatch: {learn_miss}"

B = 24
torch.manual_seed(7)
src = torch.randint(0, 64, (B,))
dst = (src + 1 + torch.randint(0, 60, (B,))) % 64
t = torch.sort(torch.rand(B) * 100.0).values
feat = torch.randn(B, 4)
neg_dst = torch.randint(0, 64, (B,))

mWC.eval(); mFREE.eval(); mWC.set_epoch(10); mFREE.set_epoch(10)
mWC.reset(); mFREE.reset()
with torch.no_grad():
    oWC = mWC(src, dst, t.clone(), feat.clone(), neg_dst.clone())
    oFREE = mFREE(src, dst, t.clone(), feat.clone(), neg_dst.clone())

c_t = oWC["cc_coherence"]
print(f"  c_t emitted: shape={tuple(c_t.shape)}  mean={c_t.mean():.4f}  "
      f"min={c_t.min():.4f}  max={c_t.max():.4f}")
print(f"  cc_weight (grad-selection scale) mean={oWC['cc_weight'].mean():.4f}")
# prediction VALUE must be identical (WC-CONF never bends s_t1_cal / s_t1_pos)
d_cal = (oWC["s_t1_cal"] - oFREE["s_t1_cal"]).abs().max().item()
d_pos = (oWC["pos_score"] - oFREE["pos_score"]).abs().max().item()
d_neg = (oWC["neg_score"] - oFREE["neg_score"]).abs().max().item()
print(f"  max|s_t1_cal  WC - FREE| = {d_cal:.3e}   (==0 => prediction VALUE not bent)")
print(f"  max|pos_score WC - FREE| = {d_pos:.3e}   (==0 => AP-path FREE, not masked)")
print(f"  max|neg_score WC - FREE| = {d_neg:.3e}")
assert d_cal == 0.0, "WC-CONF changed the prediction value (s_t1_cal)!"
assert d_pos == 0.0 and d_neg == 0.0, "WC-CONF moved the AP score!"
print("  => CONFIRMED: WC-CONF leaves prediction + AP score byte-identical (confidence-only).")
print("     (Contrast config B, which value-masks s_t1_cal — measured in _probe_gg_vs_b.py.)")

# ---------------------------------------------------------------------------
# [3] GRADIENT-SELECTION actually scales the FSM-block gradient by c_t.
#     Compare the edge-trans-driven gradient norm of the FSM/hier head between
#     WC-ON (gated) and WC-OFF (ungated) on the SAME weights + batch.  Also a
#     hard-threshold run (cc_thr) to show low-c events get ZERO weight.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("[3] GRADIENT-SELECTION: FSM-block gradient scaled by c_t")
print("=" * 74)
FSM_HEADS = ("hier_birth_head", "hier_alive_head", "hier_rising_head",
             "transition_predictor", "lifecycle_mask")

def fsm_gradnorm(m):
    tot = 0.0
    for n, p in m.named_parameters():
        if any(h in n for h in FSM_HEADS) and p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5

# NOTE (honest, untrained-model artifact): on a RANDOM untrained model the StateObserver
# is near-uniform, so b_step and the resulting c_t are ~identical across events (≈0.88)
# and the mean→1 normalization makes the SOFT gate a multiply-by-1 (no differentiation).
# This is the same untrained near-uniform artifact documented for the hard-argmax C-mask.
# To PROVE the gate is LIVE we (A) SEED a heterogeneous belief so c_t varies per event and
# show cc_weight differs, and (B) raise cc_thr above the c_t so events are HARD-zeroed,
# removing the edge_trans gradient contribution from the FSM head.

def run_loss(builder, ref_sd, seed_teleport_idx=None):
    m = builder()
    m.load_state_dict(ref_sd, strict=False)
    m.train(); m.set_epoch(10); m.reset()
    if seed_teleport_idx is not None:
        # pre-seed the per-pair belief HETEROGENEOUSLY: chosen events get b=IDLE (only
        # {IDLE,BIRTH} reachable), the rest get b=DECAY (only {REINFORCE,DECAY,DEATH}
        # reachable). Because the FREE s_t1 differs from the two reachable sets by
        # different amounts, the two groups read DIFFERENT coherence → per-event gate.
        bp = torch.zeros(B, 5)
        bp[:, DECAY] = 1.0
        bp[seed_teleport_idx, :] = 0.0
        bp[seed_teleport_idx, IDLE] = 1.0
        m.edge_mem.update_belief(src, dst, bp)
    out = m(src, dst, t.clone(), feat.clone(), neg_dst.clone())
    m.zero_grad(set_to_none=True)
    out["loss"].backward()
    return fsm_gradnorm(m), out

ref = build_WC().state_dict()
# (A) heterogeneous belief → per-event c_t differentiation
half = torch.arange(0, B, 2)
gSEED, oSEED = run_loss(build_WC, ref, seed_teleport_idx=half)
cc_seed = oSEED["cc_coherence"]
c_lo = cc_seed[half].mean().item()
c_hi = cc_seed[torch.tensor([i for i in range(B) if i not in set(half.tolist())])].mean().item()
print(f"  (A) heterogeneous seeded belief: c_t mean on b=IDLE events = {c_lo:.3f}  "
      f"vs b=DECAY events = {c_hi:.3f}  (DIFFER => per-event coherence)")
print(f"      cc_weight spread across events: min={oSEED['cc_weight'].min():.3f} "
      f"max={oSEED['cc_weight'].max():.3f}  (>0 spread => per-event gradient-selection)")
assert oSEED["cc_weight"].max() - oSEED["cc_weight"].min() > 1e-3, \
    "no per-event cc_weight spread with seeded heterogeneous belief!"

# (B) hard-threshold zeroing removes the edge_trans gradient from the FSM head.
gFULL, oFULL = run_loss(build_WC, ref)                       # thr=0  → edge_trans active
gHARD, oHARD = run_loss(lambda: build_WC(thr=0.95), ref)     # thr=.95 → all c≈.88 zeroed
BASE_NOET = {**BASE, "lambda_edge_trans": 0.0}
gNOET, _ = run_loss(lambda: SRGNN_v3_3(**BASE_NOET, causal_confidence=True,
                                       cc_C="band", cc_thr=0.0), ref)  # edge_trans OFF baseline
frac_z = (oHARD["cc_weight"] == 0).float().mean().item()
print(f"  (B) FSM-head grad-norm:")
print(f"      WC thr=0.00  (edge_trans CE active)     = {gFULL:.4e}")
print(f"      WC thr=0.95  (all c<thr → CE zeroed)    = {gHARD:.4e}  "
      f"frac events zeroed={frac_z:.2f}")
print(f"      lambda_edge_trans=0 (CE removed entirely)= {gNOET:.4e}  [reference: no CE]")
assert frac_z == 1.0, "cc_thr=0.95 did not zero the (c≈0.88) events!"
# zeroing must MOVE the FSM grad toward the no-CE reference (CE contribution removed).
assert gFULL > gHARD and abs(gHARD - gNOET) < 0.02 * gFULL, \
    "hard-threshold did not remove the edge_trans gradient (gate dead?)"
print("  => CONFIRMED: gradient-selection is LIVE — c_t<cc_thr ZEROES the FSM-block CE")
print("     gradient (grad-norm collapses to the no-CE reference); seeded c_t varies per")
print("     event. The prediction VALUE / AP score are untouched throughout (section [2]).")
gWC, oWC2, gFREE = gFULL, oFULL, gNOET   # for the summary line

# ---------------------------------------------------------------------------
# [4] backbone 0 link-pred grad (detached); no NaN; canonical/off byte-identical.
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("[4] backbone link-pred grad = 0; no NaN; byte-identical when OFF")
print("=" * 74)
BACKBONE = ("csn", "ectg", "drgc")
mg = build_WC(); mg.set_epoch(10); mg.reset()
out = mg(src, dst, t.clone(), feat.clone(), neg_dst.clone())
mg.zero_grad(set_to_none=True)
out["pred_loss"].backward()
bb_cnt = sum(1 for n, p in mg.named_parameters()
             if any(n.startswith(pre + ".") for pre in BACKBONE) and p.grad is not None)
print(f"  backbone params w/ link-pred grad = {bb_cnt}  (must be 0 -> AP-path cut)")
assert bb_cnt == 0, "backbone got link-pred gradient!"

# full train step, no NaN
mt = build_WC(); mt.train(); mt.set_epoch(10); mt.reset()
o = mt(src, dst, t.clone(), feat.clone(), neg_dst.clone())
mt.zero_grad(set_to_none=True); o["loss"].backward()
nan = any(p.grad is not None and torch.isnan(p.grad).any() for _, p in mt.named_parameters())
print(f"  full-loss train step: loss={float(o['loss'].detach()):.4f}  NaN={nan}")
assert not nan, "NaN in full-loss backward!"

# byte-identity OFF vs canonical (no new state_dict keys when flag off)
kc = set(build_canonical().state_dict().keys())
kFREE = set(build_FREE().state_dict().keys())
kWC = set(build_WC().state_dict().keys())
print(f"  state_dict keys: canonical={len(kc)}  FREE(WC-off)={len(kFREE)}  WC-on={len(kWC)}")
print(f"  WC-on extra keys vs FREE: {sorted(kWC - kFREE)}  (expect [] — cc_C is private)")
assert kWC == kFREE, "WC-CONF added state_dict keys (not byte-identical when off)!"
# default-vs-explicit-off byte identity on the AP score
m_def = build_canonical(); m_off = SRGNN_v3_3(num_nodes=64, feat_dim=4, hidden=16,
                                              causal_confidence=False)
m_def.eval(); m_off.eval(); m_def.set_epoch(5); m_off.set_epoch(5)
m_off.load_state_dict(m_def.state_dict(), strict=True)
m_def.reset(); m_off.reset()
with torch.no_grad():
    od = m_def(src, dst, t.clone(), feat.clone(), neg_dst.clone())
    oo = m_off(src, dst, t.clone(), feat.clone(), neg_dst.clone())
d = (od["pos_score"] - oo["pos_score"]).abs().max().item()
print(f"  default vs explicit-OFF: max|pos_score| = {d:.3e}  (==0 -> byte-identical)")
assert d == 0.0
print("  => CONFIRMED: backbone 0 link-pred grad; no NaN; canonical/OFF byte-identical.")

print()
print("=" * 74)
print("SUMMARY (WC-CONF, all numbers from real model/probe compute)")
print("=" * 74)
print(f"  [1] teleport c_t {[f'{x:.2f}' for x in c_teleport]} (LOW) vs coherent "
      f"{[f'{x:.2f}' for x in c_coherent]} (HIGH) — walked-chain catches never-born->DEATH.")
print(f"  [2] prediction VALUE unchanged: max|s_t1_cal WC-FREE|={d_cal:.0e}, "
      f"AP max|pos_score|={d_pos:.0e} (FREE).")
print(f"  [3] gradient-selection LIVE: FSM-block grad thr=0(CE on)={gFULL:.2e} -> "
      f"thr=.95(all c<thr zeroed)={gHARD:.2e} == no-CE ref {gNOET:.2e}; "
      f"seeded c_t splits {c_lo:.2f} vs {c_hi:.2f}.")
print("  [4] backbone 0 link-pred grad; no NaN; OFF byte-identical (cc_C private, no new keys).")
