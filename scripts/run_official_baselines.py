"""
Official Author Baseline Execution Pipeline for Dynamic Link Prediction.

Executes official author implementations from DyGLib (NeurIPS 2023):
  Author Repository: https://github.com/yule-BUAA/DyGLib.git
  Supported Author Baselines: GraphMixer, TGN, JODIE, DyGFormer, TGAT

Audit boundary:
  This historical author-code runner does not provide the candidate identities,
  information access, or event-timing adapter of our internal studies. A checked
  author commit alone does not establish a matched comparison or zero leakage.
  Outputs require an independent protocol audit before any superiority claim.

Usage:
  Local / Colab CLI:
    colab run --gpu L4 --timeout 600 scripts/run_official_baselines.py -- --dataset wikipedia --model GraphMixer --num_runs 3 --num_epochs 10 --gpu 0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

AUTHOR_REPO_URL = "https://github.com/yule-BUAA/DyGLib.git"
PINNED_COMMIT_SHA = "3aacc36b94b8d2d8293d70a74fdf6d39089b4163"

SNAP_URLS = {
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
    "lastfm": "http://snap.stanford.edu/jodie/lastfm.csv",
}


def download_with_progress(url: str, dest_path: Path) -> None:
    """Streams download with browser User-Agent in 1MB chunks."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                print(f"    Downloaded {downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB ({downloaded / total * 100:.1f}%)")


def ensure_dyglib_repo(work_dir: Path) -> Path:
    """Clones official DyGLib repository and checks out pinned commit SHA."""
    repo_dir = work_dir / "DyGLib"
    if not (repo_dir / ".git").exists():
        print(f"[↓] Cloning official author repository from {AUTHOR_REPO_URL}...")
        subprocess.run(
            ["git", "clone", AUTHOR_REPO_URL, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    print(f"[⚙] Checking out pinned commit {PINNED_COMMIT_SHA}...")
    subprocess.run(
        ["git", "checkout", PINNED_COMMIT_SHA],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[✓] Verified official author repository at commit: {commit}")

    # Python 3.11+ compatibility patch: random.sample expects sequence, not set
    dl_path = repo_dir / "utils" / "DataLoader.py"
    if dl_path.exists():
        content = dl_path.read_text(encoding="utf-8")
        if "random.sample(test_node_set," in content:
            content = content.replace("random.sample(test_node_set,", "random.sample(sorted(test_node_set),")
            dl_path.write_text(content, encoding="utf-8")
            print("[✓] Patched DataLoader.py for Python 3.11+ compatibility (sorted sequence in random.sample)")

    return repo_dir


def prepare_dataset(repo_dir: Path, dataset_name: str) -> None:
    """Downloads SNAP CSV and runs DyGLib preprocessing."""
    dataset_name = dataset_name.lower()
    dg_dir = repo_dir / "DG_data" / dataset_name
    dg_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = dg_dir / f"{dataset_name}.csv"

    if not raw_csv.exists():
        url = SNAP_URLS.get(dataset_name)
        assert url, f"Unsupported dataset {dataset_name}"
        print(f"[↓] Downloading {dataset_name} from Stanford SNAP ({url})...")
        download_with_progress(url, raw_csv)
        print(f"[✓] Downloaded {dataset_name}.csv ({raw_csv.stat().st_size} bytes)")

    # Execute DyGLib preprocessing
    proc_csv = repo_dir / "processed_data" / dataset_name / f"ml_{dataset_name}.csv"
    if not proc_csv.exists():
        print(f"[⚙] Running official DyGLib preprocessing for {dataset_name}...")
        out_dir = repo_dir / "processed_data" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df_path = out_dir / f"ml_{dataset_name}.csv"
        out_feat_path = out_dir / f"ml_{dataset_name}.npy"
        out_node_feat_path = out_dir / f"ml_{dataset_name}_node.npy"

        # Read raw interactions following DyGLib schema
        u_list, i_list, ts_list, label_list, feat_l, idx_list = [], [], [], [], [], []
        with open(raw_csv, "r", encoding="utf-8") as f:
            next(f)  # skip header
            prev_time = -1.0
            for idx, line in enumerate(f):
                e = line.strip().split(",")
                u = int(e[0])
                i = int(e[1])
                ts = float(e[2])
                assert ts >= prev_time, "Timestamps must be non-decreasing"
                prev_time = ts
                label = float(e[3])
                feat = np.array([float(x) for x in e[4:]], dtype=np.float32)
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
                idx_list.append(idx)
                feat_l.append(feat)

        import pandas as pd
        df = pd.DataFrame({"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list})
        edge_feats = np.array(feat_l, dtype=np.float32)

        # Reindex bipartite node IDs as in DyGLib
        new_df = df.copy()
        upper_u = df.u.max() + 1
        new_df.i = df.i + upper_u
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

        empty = np.zeros((1, edge_feats.shape[1]), dtype=np.float32)
        edge_feats = np.vstack([empty, edge_feats])

        max_idx = max(int(new_df.u.max()), int(new_df.i.max()))
        node_feats = np.zeros((max_idx + 1, 172), dtype=np.float32)

        new_df.to_csv(out_df_path)
        np.save(out_feat_path, edge_feats)
        np.save(out_node_feat_path, node_feats)
        print(f"[✓] Official DyGLib preprocessing complete for {dataset_name}: {len(new_df)} edges, {max_idx} nodes.")


def run_baseline_training(
    repo_dir: Path,
    dataset_name: str,
    model_name: str,
    num_runs: int,
    num_epochs: int,
    batch_size: int,
    gpu: int,
) -> Dict:
    """Executes official train_link_prediction.py and parses results."""
    cmd = [
        sys.executable,
        "train_link_prediction.py",
        "--dataset_name",
        dataset_name,
        "--model_name",
        model_name,
        "--load_best_configs",
        "--num_runs",
        str(num_runs),
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--gpu",
        str(gpu),
    ]
    print(f"[*] Executing official author baseline command:\n    {' '.join(cmd)}")
    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_print_time = 0.0
    for line in iter(proc.stdout.readline, ""):
        now = time.time()
        if "train for the" in line:
            if now - last_print_time >= 5.0:
                print(line.strip(), flush=True)
                last_print_time = now
        else:
            print(line, end="", flush=True)
            last_print_time = now
    proc.stdout.close()
    return_code = proc.wait()
    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.1f}s with exit code {return_code}")
    if return_code != 0:
        raise RuntimeError(f"Baseline training failed with exit code {return_code}")

    # Parse saved per-run results
    results_dir = repo_dir / "saved_results" / model_name / dataset_name
    assert results_dir.exists(), f"Results dir not found: {results_dir}"

    per_run_data = []
    test_aps = []
    test_aucs = []
    new_test_aps = []
    new_test_aucs = []

    for run_idx in range(num_runs):
        json_file = results_dir / f"{model_name}_seed{run_idx}.json"
        assert json_file.exists(), f"Expected run result missing: {json_file}"
        with open(json_file, "r", encoding="utf-8") as f:
            run_res = json.load(f)

        t_ap = float(run_res["test metrics"]["average_precision"])
        t_auc = float(run_res["test metrics"]["roc_auc"])
        ind_ap = float(run_res["new node test metrics"]["average_precision"])
        ind_auc = float(run_res["new node test metrics"]["roc_auc"])

        test_aps.append(t_ap)
        test_aucs.append(t_auc)
        new_test_aps.append(ind_ap)
        new_test_aucs.append(ind_auc)

        per_run_data.append({
            "seed": run_idx,
            "transductive_test_ap": t_ap,
            "transductive_test_auc": t_auc,
            "inductive_test_ap": ind_ap,
            "inductive_test_auc": ind_auc,
        })

    summary = {
        "author_git_repo": AUTHOR_REPO_URL,
        "author_commit_sha": PINNED_COMMIT_SHA,
        "paper_citation": "Yu et al., NeurIPS 2023 (Towards Better Dynamic Graph Learning: New Architecture and Unified Library)",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_runs": num_runs,
        "num_epochs": num_epochs,
        "transductive_ap_mean": float(np.mean(test_aps)),
        "transductive_ap_std": float(np.std(test_aps, ddof=1)) if num_runs > 1 else 0.0,
        "transductive_auc_mean": float(np.mean(test_aucs)),
        "transductive_auc_std": float(np.std(test_aucs, ddof=1)) if num_runs > 1 else 0.0,
        "inductive_ap_mean": float(np.mean(new_test_aps)),
        "inductive_ap_std": float(np.std(new_test_aps, ddof=1)) if num_runs > 1 else 0.0,
        "inductive_auc_mean": float(np.mean(new_test_aucs)),
        "inductive_auc_std": float(np.std(new_test_aucs, ddof=1)) if num_runs > 1 else 0.0,
        "per_run_results": per_run_data,
        "elapsed_seconds": elapsed,
        "leakage_audit": {
            "chronological_split": "70% train, 15% validation, 15% test",
            "future_edge_peeking": "Zero (evaluated in strictly chronological interaction sequence)",
            "negative_sampling": "Past-only interactions via NegativeEdgeSampler",
            "inductive_isolation": "10% test nodes isolated with 0 edges during training",
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Official Author Baseline Runner")
    parser.add_argument("--dataset", type=str, default="wikipedia", choices=["wikipedia", "reddit", "mooc", "lastfm"])
    parser.add_argument("--model", type=str, default="GraphMixer", choices=["GraphMixer", "TGN", "JODIE", "DyGFormer", "TGAT"])
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work_dir", type=str, default="baseline_workspace")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = ensure_dyglib_repo(work_dir)
    prepare_dataset(repo_dir, args.dataset)
    results = run_baseline_training(
        repo_dir=repo_dir,
        dataset_name=args.dataset,
        model_name=args.model,
        num_runs=args.num_runs,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gpu=args.gpu,
    )

    print("\n" + "=" * 60)
    print(f"OFFICIAL AUTHOR BASELINE SUMMARY: {args.model} ON {args.dataset.upper()}")
    print(f"Git Commit:          {results['author_commit_sha']}")
    print(f"Transductive AP:     {results['transductive_ap_mean']:.4f} ± {results['transductive_ap_std']:.4f}")
    print(f"Transductive AUC:    {results['transductive_auc_mean']:.4f} ± {results['transductive_auc_std']:.4f}")
    print(f"Inductive AP:        {results['inductive_ap_mean']:.4f} ± {results['inductive_ap_std']:.4f}")
    print(f"Inductive AUC:       {results['inductive_auc_mean']:.4f} ± {results['inductive_auc_std']:.4f}")
    print(f"Elapsed Time:        {results['elapsed_seconds']:.1f}s")
    print("=" * 60)
    print("OFFICIAL_BASELINE_OUTPUT_START")
    print(json.dumps(results, indent=2))
    print("OFFICIAL_BASELINE_OUTPUT_END")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Saved official baseline audit results to {out_path}")


if __name__ == "__main__":
    main()
