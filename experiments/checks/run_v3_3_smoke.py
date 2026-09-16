"""
Smoke test for RS-GNN v3.3 (transition-aware + LFG).

Goals (acceptance):
  - Forward pass works, no NaN
  - 5 states all have meaningful mass (entropy > 1.0)
  - "Chưa sinh đã chết" violation rate ≈ 0
  - LFG compliance distribution varied (not all 1.0 or 0.05)
  - Test AP ≥ 0.93 (sanity, not necessarily beat v3.1)
"""
import os, sys, time, json, random
import numpy as np
import torch

V33_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(V33_DIR))
LAB_DIR = os.path.dirname(V33_DIR)
EXP_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, EXP_DIR)        # data, utils, train
sys.path.insert(0, V33_DIR)        # v3_3/models takes precedence

from lifecycle_readout.datasets import download_dataset, get_data_splits
from lifecycle_readout.training import run_epoch, DEVICE
from lifecycle_readout.modeling.sr_gnn_v3_3 import SRGNN_v3_3


def measure(model):
    """Inspect model state for diagnostics."""
    # ever_alive distribution
    if not model.ever_alive._values:
        ever_alive_mean = 0.0
    else:
        ev = torch.stack(model.ever_alive._values)
        ever_alive_mean = float(ev.mean())

    # edge state from edge_mem (multi-signal)
    if not model.edge_mem._state_table:
        return {"ever_alive_mean": ever_alive_mean, "n_edges": 0}
    states = torch.stack(model.edge_mem._state_table)
    state_idx = states[:, :5].argmax(dim=-1)
    counts = torch.bincount(state_idx, minlength=5).float()
    dist = (counts / counts.sum()).tolist()
    return {
        "ever_alive_mean": ever_alive_mean,
        "n_edges":         int(states.size(0)),
        "hawkes_lam_mean": float(states[:, 6].mean()),
        "edge_state_dist": dist,
    }


def run_epoch_v33(model, split_data, num_nodes, batch_size, optimizer=None,
                  inductive_nodes=None, seen_nodes=None, desc="train", epoch=0):
    """Wrapper to set epoch on model before run_epoch."""
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    return run_epoch(model, split_data, num_nodes, batch_size,
                     optimizer=optimizer,
                     inductive_nodes=inductive_nodes,
                     seen_nodes=seen_nodes, desc=desc)


def main():
    DATASET = "wikipedia"
    EPOCHS  = 5
    HIDDEN  = 128
    BATCH   = 500
    LR      = 1e-3
    SEED    = 42

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = download_dataset(DATASET)
    splits = get_data_splits(data)
    num_nodes, feat_dim = data["num_nodes"], data["feat_dim"]

    model = SRGNN_v3_3(num_nodes, feat_dim, HIDDEN, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    print(f"\n=== SMOKE TEST: SRGNN_v3_3 (Transition-Aware + LFG) ===")
    print(f"Dataset: {DATASET}  nodes: {num_nodes}  edges: {data['num_edges']}")
    print(f"Epochs: {EPOCHS}  batch: {BATCH}  lr: {LR}  seed: {SEED}\n")

    t0 = time.time()
    history = []
    for ep in range(1, EPOCHS + 1):
        if hasattr(model, "reset"): model.reset()
        tr = run_epoch_v33(model, splits["train"], num_nodes, BATCH,
                           optimizer=optimizer,
                           desc=f"v33/E{ep}/tr", epoch=ep)
        va = run_epoch_v33(model, splits["val"], num_nodes, BATCH,
                           desc=f"v33/E{ep}/va", epoch=ep)
        info = measure(model)
        elapsed = time.time() - t0
        log = {"epoch": ep,
               "tr_AP": tr["AP"], "tr_loss": tr["Loss"],
               "va_AP": va["AP"], "va_loss": va["Loss"],
               "info":  info,
               "elapsed_s": elapsed}
        history.append(log)
        print(f"E{ep:02d}  tr_AP={tr['AP']:.4f} tr_loss={tr['Loss']:.4f}  "
              f"va_AP={va['AP']:.4f}  [{elapsed:.0f}s]")
        if info["n_edges"] > 0:
            d = info["edge_state_dist"]
            print(f"     ever_alive={info['ever_alive_mean']:.3f}  hawkes_λ={info['hawkes_lam_mean']:.3f}")
            print(f"     edge_state[I,B,R,D,Dt]=[{d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}, {d[3]:.2f}, {d[4]:.2f}]")

    # Test
    if hasattr(model, "reset"): model.reset()
    te = run_epoch_v33(model, splits["test"], num_nodes, BATCH, desc="v33/test", epoch=EPOCHS)
    final_info = measure(model)

    out = {
        "model":     "srgnn_v3_3_smoke",
        "dataset":   DATASET,
        "seed":      SEED,
        "epochs":    EPOCHS,
        "test_ap":   te["AP"],
        "test_auc":  te["AUC"],
        "final_info": final_info,
        "history":   history,
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(PROJECT_ROOT, "results", "audit", "v3_3_smoke.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "="*60)
    print(f"FINAL  test AP={te['AP']:.4f}  AUC={te['AUC']:.4f}")
    if final_info["n_edges"] > 0:
        d = final_info["edge_state_dist"]
        print(f"edge_state_dist=[{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, {d[3]:.3f}, {d[4]:.3f}]")
        print(f"ever_alive_mean={final_info['ever_alive_mean']:.3f}")

    print("\n=== Acceptance Criteria ===")
    print(f"  [{'✓' if te['AP'] >= 0.93 else '✗'}] Test AP ≥ 0.93 (sanity): {te['AP']:.4f}")
    print(f"  [{'✓' if te['AP'] >= 0.94 else '✗'}] Test AP ≥ 0.94 (beat v2-ish): {te['AP']:.4f}")
    print(f"  [{'✓' if te['AP'] >= 0.95 else '✗'}] Test AP ≥ 0.95 (compete v3.1): {te['AP']:.4f}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
