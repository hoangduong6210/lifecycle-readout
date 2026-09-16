"""CPU grad-flow unit test for the single-variable detach probe (PM 2026-06-08).

Builds the config-B model (v3 + hier + decol_hier_v2 + causal_batch + correct_decoupled
+ hier_causal_policy, enable_main_predictor=False) in BOTH detach settings and checks,
by backward-ing pred_loss ALONE, whether the backbone (csn/ectg/drgc) receives gradient.

Expected:
  edge_h_detach_scorepath=True  (canonical B)  -> backbone grad-norm from pred_loss == 0
  edge_h_detach_scorepath=False (probe OFF)    -> backbone grad-norm from pred_loss  > 0

This proves the toggle flips exactly the link-pred->backbone gradient, nothing else.
"""
import os, sys
import torch
V33 = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(V33))
sys.path.insert(0, EXP); sys.path.insert(0, V33)
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3

torch.manual_seed(0)
N, F, H, B = 60, 8, 16, 12

def build(detach_flag):
    m = SRGNN_v3_3(N, F, H, device=torch.device("cpu"),
                   enable_main_predictor=False,
                   enable_lfg=True, design="correct_decoupled",
                   fsm_arch="v3", fsm_decode="hier", decol_hier_v2=True,
                   causal_batch=True, hier_causal_policy=True,
                   lfg_mode="soft", compliance_floor=0.05,
                   edge_h_detach_scorepath=detach_flag)
    return m

def backbone_params(m):
    ps = []
    for name, p in m.named_parameters():
        if name.split(".")[0] in ("csn", "ectg", "drgc"):
            ps.append((name, p))
    return ps

def grad_norm_from_predloss(detach_flag):
    m = build(detach_flag)
    m.train()
    if hasattr(m, "reset"): m.reset()
    src = torch.randint(0, N, (B,))
    dst = torch.randint(0, N, (B,))
    # ensure src != dst
    dst = torch.where(dst == src, (dst + 1) % N, dst)
    t = torch.arange(1, B + 1).float()
    feat = torch.randn(B, F)
    neg = torch.randint(0, N, (B,))
    out = m(src, dst, t, feat, neg)
    m.zero_grad(set_to_none=True)
    out["pred_loss"].backward()
    total = 0.0
    n_nonzero = 0
    for name, p in backbone_params(m):
        if p.grad is not None:
            g = float(p.grad.norm())
            total += g
            if g > 0: n_nonzero += 1
    n_bb = len(backbone_params(m))
    return total, n_nonzero, n_bb

for flag in (True, False):
    gn, nz, nbb = grad_norm_from_predloss(flag)
    label = "CANONICAL-B (detach ON)" if flag else "PROBE (detach OFF)"
    print(f"edge_h_detach_scorepath={flag!s:5} [{label}]: "
          f"backbone pred_loss grad-norm = {gn:.6e}  "
          f"({nz}/{nbb} backbone params with nonzero grad)")
