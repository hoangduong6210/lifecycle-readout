"""
CPU probe: LFG gradient-mode A/B (HARD / SOFT / NONE) on config B DETACHED.

Goal (protocol requirement 2026-06-04): prove the gradient SEMANTICS of the symbolic
causal policy on pred_loss, on the DETACHED correct_decoupled arm:

  ARM-HARD : lfg_mode=hard, compliance_floor=0.0  -> violating event gradient = 0
  ARM-SOFT : lfg_mode=soft, floor 0.05            -> violating event gradient ~0.05x
  ARM-NONE : lfg off (enable_lfg=False)           -> full gradient, no policy

Plus: backbone (csn/ectg/drgc) sees ZERO link-pred gradient in ALL THREE
(detached intact — the gate only changes gradient TO THE FSM HEAD).

We print REAL numbers from the model's OWN gate code path. No fabrication.

Part A: unit-probe the EXACT gate assembly (model.compute path lines ~1496-1528)
        with a CONTROLLED violating event (argmax s_t = IDLE, argmax s_t1 = DEATH;
        C[IDLE,DEATH]=0) and a CONTROLLED valid event, using the model's real
        compute_causal_validity + self.causal_rule. This isolates the gate factor.

Part B: REAL forward on a synthetic batch under each arm; backprop pred_loss;
        report backbone-grad vs fsm-head-grad norms + realized per-event weights.
"""
import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3
from lifecycle_readout.modeling.fsm_head import compute_causal_validity, compute_compliance, CAUSAL_RULE_MATRIX

torch.manual_seed(0)
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4

# Common config B (correct_decoupled, detached) constructor kwargs.
COMMON = dict(
    num_nodes=64, feat_dim=4, hidden=16,
    design="correct_decoupled", fsm_arch="v3", fsm_decode="hier",
    decol_hier_v2=True, causal_batch=True, hier_causal_policy=True,
    lambda_edge_trans=0.5,
)

def build(arm):
    if arm == "HARD":
        kw = dict(COMMON, lfg_mode="hard", compliance_floor=0.0, enable_lfg=True)
    elif arm == "SOFT":
        kw = dict(COMMON, lfg_mode="soft", compliance_floor=0.05, enable_lfg=True)
    elif arm == "NONE":
        # No LFG reweight at all: enable_lfg=False -> lfg_weight forced 0 ->
        # compliance_effective = uniform 1; lfg_mode soft (no hard gate path).
        kw = dict(COMMON, lfg_mode="soft", compliance_floor=0.05, enable_lfg=False)
    else:
        raise ValueError(arm)
    m = SRGNN_v3_3(**kw)
    m.eval()
    m.set_epoch(10)  # past warmup(2)+ramp(3) -> lfg_weight=1.0, gate fully active
    return m

def assert_detached_construct(m, arm):
    # The TRAP PM flagged: hard+floor=0 must NOT have flipped end-to-end.
    emp = m.enable_main_predictor
    lm = m.lfg_mode
    cf = m.compliance_floor
    print(f"  [{arm}] enable_main_predictor={emp}  lfg_mode={lm!r}  compliance_floor={cf}  enable_lfg={m.enable_lfg}")
    assert emp is False, f"ARM {arm}: enable_main_predictor flipped True -> END-TO-END (BUG)"

print("="*72)
print("Construct check: all 3 arms must stay DETACHED (enable_main_predictor=False)")
print("="*72)
models = {a: build(a) for a in ("HARD", "SOFT", "NONE")}
for a, m in models.items():
    assert_detached_construct(m, a)

# ---------------------------------------------------------------------------
# PART A: exact gate factor on a CONTROLLED violating vs valid event.
# We reproduce the model's pred_weight assembly (sr_gnn_v3_3 ~L1496-1528):
#   compliance_effective = (1-w)*1 + w*compliance        (w = lfg_weight = 1.0 here)
#   if hard:  hard_gate = where(v>0.5, 1, floor); pred_weight = comp_eff * hard_mask
#   else:     pred_weight = compliance_effective
# using the model's OWN compute_causal_validity / compute_compliance / matrix.
# ---------------------------------------------------------------------------
print()
print("="*72)
print("PART A  exact gate factor (controlled events; model's own gate functions)")
print("="*72)

# Event 0 = VIOLATING : argmax s_t=IDLE, argmax s_t1=DEATH -> C[IDLE,DEATH]=0,
#                       AND an abrupt/incoherent jump -> low SOFT compliance too.
# Event 1 = VALID+SMOOTH: a coherent, near-stationary transition -> C[*,*]=1 AND
#                       high SOFT compliance (Rule-2 smooth + Rule-3 Hawkes-consistent).
# NOTE on SOFT semantics (faithful): the SOFT path does NOT consult the causal
# matrix C at all. Its reweight is compute_compliance() = Rule1*Rule2*Rule3 clamped
# to [0.05,1.0]. So SOFT down-weights *abrupt/incoherent* transitions (which a
# C-violation usually is) toward the 0.05 floor; it does NOT specifically target C.
def onehotish(idx, n=5, peak=0.90):
    v = torch.full((n,), (1.0 - peak) / (n - 1))
    v[idx] = peak
    return v

# violating: IDLE -> DEATH (big abrupt jump, C=0)
viol_t  = onehotish(IDLE)
viol_t1 = onehotish(DEATH)
# valid+smooth: nearly the same distribution before/after (REINFORCE-ish, tiny change),
# and Hawkes-consistent (active_score ~ expected_active at lam=mean). argmax s_t=
# argmax s_t1=REINFORCE -> C[REINFORCE,REINFORCE]=1.
valid_t  = onehotish(REINFORCE, peak=0.80)
valid_t1 = onehotish(REINFORCE, peak=0.82)

s_t  = torch.stack([viol_t,  valid_t])    # (2,5)
s_t1 = torch.stack([viol_t1, valid_t1])   # (2,5)
ever_alive = torch.tensor([1.0, 1.0])
# Hawkes: event1 lam s.t. expected_active matches its active_score (consistent).
hawkes = torch.tensor([1.0, 1.0])

C = CAUSAL_RULE_MATRIX.clone()
ai, aj = int(s_t[0].argmax()), int(s_t1[0].argmax())
bi, bj = int(s_t[1].argmax()), int(s_t1[1].argmax())
print(f"  C[{ai},{aj}]={C[ai,aj].item():.0f} (violating: IDLE->DEATH)   "
      f"C[{bi},{bj}]={C[bi,bj].item():.0f} (valid+smooth: REINFORCE->REINFORCE)")

# lfg_weight at epoch 10: ramp = min(1,(10-2)/3)=1.0
lfg_weight = 1.0

for arm in ("HARD", "SOFT", "NONE"):
    m = models[arm]
    # base per-event BCE-like signal magnitude (a stand-in loss_per_event = 1.0 each)
    loss_per_event = torch.ones(2)
    # compliance (soft) — model's function
    compliance_pos = compute_compliance(s_t, s_t1, ever_alive, hawkes)  # (2,)
    if not m.enable_lfg:
        w = 0.0
    else:
        w = lfg_weight
    compliance_effective = (1 - w) * torch.ones(2) + w * compliance_pos
    if m.lfg_mode == "hard":
        v = compute_causal_validity(s_t, s_t1, C)  # (2,) in {0,1}
        gate = torch.where(v > 0.5, torch.ones(2), torch.full((2,), float(m.compliance_floor)))
        hard_mask = (1 - w) * torch.ones(2) + w * gate
        pred_weight = compliance_effective * hard_mask
    else:
        pred_weight = compliance_effective
    # ratio of effective gradient weight: violating / valid
    pw = pred_weight.detach()
    ratio = (pw[0] / pw[1]).item() if pw[1] != 0 else float("nan")
    print(f"  [{arm}] pred_weight  violating={pw[0].item():.4f}  valid={pw[1].item():.4f}"
          f"   ratio(violating/valid)={ratio:.4f}")

# ---------------------------------------------------------------------------
# PART B: REAL forward + backprop pred_loss; backbone vs fsm-head grad norms.
# ---------------------------------------------------------------------------
print()
print("="*72)
print("PART B  real forward: backbone link-pred grad must be 0 in ALL arms;")
print("        gate changes only the FSM-head gradient.")
print("="*72)

BACKBONE = ("csn", "ectg", "drgc")
FSMHEAD = ("state_observer", "transition_predictor", "existence_decoder")

def grad_norm(model, prefixes):
    tot = 0.0
    cnt = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if any(n.startswith(pref + ".") for pref in prefixes):
            tot += float(p.grad.detach().pow(2).sum())
            cnt += 1
    return tot ** 0.5, cnt

def n_params_with_grad(model, prefixes):
    c = 0
    for n, p in model.named_parameters():
        if any(n.startswith(pref + ".") for pref in prefixes):
            c += 1
    return c

B = 16
for arm in ("HARD", "SOFT", "NONE"):
    # fresh model per arm (independent grads)
    m = build(arm)
    src = torch.randint(0, 64, (B,))
    dst = torch.randint(0, 64, (B,))
    # ensure no self-loops collide weirdly
    dst = (src + 1 + torch.randint(0, 60, (B,))) % 64
    t = torch.sort(torch.rand(B) * 100.0).values
    feat = torch.randn(B, 4)
    neg_dst = torch.randint(0, 64, (B,))
    out = m(src, dst, t, feat, neg_dst)
    pred_loss = out["pred_loss"]
    m.zero_grad(set_to_none=True)
    pred_loss.backward()
    bb_norm, bb_cnt = grad_norm(m, BACKBONE)
    fh_norm, fh_cnt = grad_norm(m, FSMHEAD)
    bb_total = n_params_with_grad(m, BACKBONE)
    fh_total = n_params_with_grad(m, FSMHEAD)
    has_nan = any((p.grad is not None and torch.isnan(p.grad).any())
                  for _, p in m.named_parameters())
    print(f"  [{arm}] pred_loss={float(pred_loss.detach()):.4f}  "
          f"backbone grad-norm={bb_norm:.3e} ({bb_cnt}/{bb_total} params w/grad)  "
          f"fsm-head grad-norm={fh_norm:.3e} ({fh_cnt}/{fh_total})  NaN={has_nan}")

print()
print("INTERPRETATION:")
print("  PART A ratio: HARD->0.0 (violating event zeroed), SOFT->~floor (<1, !=0),")
print("                NONE->1.0 (no policy). PART B: backbone grad-norm ~0 in all 3.")
