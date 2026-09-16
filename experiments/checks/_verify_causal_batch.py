"""CPU verification of the P1 causal_batch fix on EdgeStateStoreV3.

Builds a small synthetic stream where ONE pair fires MANY times within a single
batch, then checks:
  (A) causal_batch=ON Welford (n, μ_dt, var_dt), Hawkes λ, recur EWMA, and rate
      fast/slow/peak MATCH an event-by-event (batch_size=1) reference  (max|Δ|~0).
  (B) causal_batch=OFF REPRODUCES the bug: Welford n caps at #batches (not #events).
  (C) no NaN/Inf anywhere; true_occ matches the genuine per-event count.

Pure store-level test (no model weights) → deterministic, isolates the fix.
"""
import torch
from lifecycle_readout.modeling.sr_gnn_v3 import (
    EdgeStateStoreV3,
    RATE_DT_FLOOR,
    RATE_INIT,
    update_multisignal,
)

torch.manual_seed(0)
N = 100
DEV = torch.device("cpu")

# ── synthetic stream ──────────────────────────────────────────────────────────
# Pair A=(1,2) fires 30 times; pair B=(3,4) fires 12 times; plus singletons.
# All events of A and B fall inside ONE batch (batch_size=64) to trigger the bug.
events = []   # (u, v, dt)
import random; random.seed(1)
for k in range(30):
    events.append((1, 2, 2.0 + 0.1 * k))         # A: drifting inter-arrival
for k in range(12):
    events.append((3, 4, 5.0 - 0.2 * k))         # B: shrinking gaps
for k in range(10):
    events.append((10 + k, 20 + k, 3.0))         # singletons
random.shuffle(events)                            # interleave in stream order
src = torch.tensor([e[0] for e in events])
dst = torch.tensor([e[1] for e in events])
dt  = torch.tensor([e[2] for e in events], dtype=torch.float)
E = len(events)
print(f"stream: {E} events, batch covers all (A fires 30x, B 12x in ONE batch)")

af, as_, leak = 0.6, 0.2, 0.97   # match SRGNN defaults (update_rate_ewma/peak)


def run_store(causal, batch_size):
    """Replay the EXACT model store-call sequence per batch and collect, per event:
    the PRE-event edge_st (Welford/Hawkes/recur), and the PRE rate fast/slow/peak."""
    store = EdgeStateStoreV3(N, 8, DEV, causal_batch=causal)
    pre_st, pre_fast, pre_slow, pre_peak, occ = [], [], [], [], []
    for b0 in range(0, E, batch_size):
        s = src[b0:b0+batch_size]; d = dst[b0:b0+batch_size]; dd = dt[b0:b0+batch_size]
        # 1) get_batch (PRE state) — causal path folds in-batch same-pair events.
        edge_st = store.get_batch(s, d, dt=dd)
        # 2) rate peeks (PRE) — causal vs legacy peek+update.
        if causal:
            rf, rs, rp = store.peek_step_rate_causal(s, d, dd, af, as_, leak)
        else:
            rf, rs = store.peek_rate_ewma(s, d)
            rp = store.peek_rate_peak(s, d)
        # 3) model folds THIS event's dt → new_est (deterministic [5:10]).
        new_est = update_multisignal(edge_st, dd)
        store.update_batch(s, d, new_est)
        # 4) legacy post-updates for rate (causal path already advanced them).
        if not causal:
            store.update_rate_ewma(s, d, dd, af, as_)
            rf_post, _ = store.peek_rate_ewma(s, d)
            store.update_rate_peak(s, d, rf_post, leak)
        # 5) true_occ
        to = store.get_true_occ(s, d)
        for i in range(s.size(0)):
            pre_st.append(edge_st[i].clone()); pre_fast.append(float(rf[i]))
            pre_slow.append(float(rs[i])); pre_peak.append(float(rp[i]))
            occ.append(float(to[i]))
    return (torch.stack(pre_st), torch.tensor(pre_fast), torch.tensor(pre_slow),
            torch.tensor(pre_peak), torch.tensor(occ))


# REFERENCE = event-by-event (batch_size=1) with causal_batch=ON (== legacy semantics
# when batch_size=1, since no two same-pair events ever share a batch).
ref_st, ref_fast, ref_slow, ref_peak, ref_occ = run_store(causal=True, batch_size=1)
# CAUSAL at the REAL batch size (64 here ⇒ A/B repeat inside one batch).
c_st, c_fast, c_slow, c_peak, c_occ = run_store(causal=True, batch_size=64)
# LEGACY (buggy) at the same batch size.
b_st, b_fast, b_slow, b_peak, b_occ = run_store(causal=False, batch_size=64)


def maxabs(a, b):
    return float((a - b).abs().max())


print("\n=== (A) CAUSAL (batch=64) vs REFERENCE (batch=1) — should be ~0 ===")
for name, col in [("recur[5]", 5), ("hawkes_lam[6]", 6), ("mean_dt[7]", 7),
                  ("var_dt[8]", 8), ("n_obs[9]", 9)]:
    print(f"  max|Δ| {name:14s} = {maxabs(c_st[:, col], ref_st[:, col]):.3e}")
print(f"  max|Δ| rate_fast      = {maxabs(c_fast, ref_fast):.3e}")
print(f"  max|Δ| rate_slow      = {maxabs(c_slow, ref_slow):.3e}")
print(f"  max|Δ| rate_peak      = {maxabs(c_peak, ref_peak):.3e}")
print(f"  max|Δ| true_occ       = {maxabs(c_occ, ref_occ):.3e}")

print("\n=== (B) LEGACY (batch=64) — should REPRODUCE the bug ===")
A_mask = (src == 1) & (dst == 2)
print(f"  pair A fires {int(A_mask.sum())}x in stream")
print(f"  REFERENCE max n_obs (pre, all events) = {ref_st[:,9].max():.0f}  "
      f"(== fires-1 the pair has seen before)")
print(f"  CAUSAL    max n_obs                   = {c_st[:,9].max():.0f}")
print(f"  LEGACY    max n_obs                   = {b_st[:,9].max():.0f}   <- caps (bug)")
print(f"  LEGACY    max rate_peak               = {b_peak.max():.4f}  "
      f"(RATE_INIT={RATE_INIT}; pinned ⇒ slope/rate degenerate)")
print(f"  CAUSAL    max rate_peak               = {c_peak.max():.4f}")
print(f"  CAUSAL    rate_slope_rel spread (fast-slow)/(slow+1e-3): "
      f"min={((c_fast-c_slow)/(c_slow+1e-3)).min():.3f} "
      f"max={((c_fast-c_slow)/(c_slow+1e-3)).max():.3f}")
print(f"  LEGACY    rate_slope_rel spread: "
      f"min={((b_fast-b_slow)/(b_slow+1e-3)).min():.3f} "
      f"max={((b_fast-b_slow)/(b_slow+1e-3)).max():.3f}")

print("\n=== (C) sanity ===")
allc = torch.cat([c_st.flatten(), c_fast, c_slow, c_peak])
print(f"  CAUSAL any NaN={bool(torch.isnan(allc).any())}  "
      f"any Inf={bool(torch.isinf(allc).any())}")
print(f"  true_occ matches genuine per-pair count (A last occ == 30): "
      f"{int(c_occ[A_mask][-1]) if A_mask.any() else 'NA'}")

# Hard asserts (fail loud if the fix regresses)
tol = 1e-5
ok = (maxabs(c_st[:, 5:10], ref_st[:, 5:10]) < tol
      and maxabs(c_fast, ref_fast) < tol and maxabs(c_slow, ref_slow) < tol
      and maxabs(c_peak, ref_peak) < tol and maxabs(c_occ, ref_occ) < tol)
bug = (b_st[:, 9].max() < ref_st[:, 9].max() - 1)   # legacy caps below reference
print(f"\nRESULT: causal==reference (all stats): {ok}   |   legacy reproduces cap-bug: {bool(bug)}")
assert ok, "CAUSAL stats do NOT match event-by-event reference — fix is wrong"
assert bug, "LEGACY did not reproduce the cap bug — test stream too weak"
print("PASS")
