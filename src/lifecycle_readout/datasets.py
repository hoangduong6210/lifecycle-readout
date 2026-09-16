"""
Download and preprocess temporal graph datasets.
Datasets: Wikipedia, Reddit, MOOC (from Jodie / TGB benchmarks)
Format: each row = (source, destination, timestamp, edge_idx, [features])
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    os.environ.get("LIFECYCLE_DATA_DIR", PROJECT_ROOT / "resources" / "corpora")
).resolve()

URLS = {
    "wikipedia": "https://snap.stanford.edu/jodie/wikipedia.csv",
    "reddit": "https://snap.stanford.edu/jodie/reddit.csv",
    "mooc": "https://snap.stanford.edu/jodie/mooc.csv",
    "lastfm": "https://snap.stanford.edu/jodie/lastfm.csv",
}


def download_dataset(name: str):
    """Download raw CSV from SNAP/Jodie, or load local dataset."""
    name = name.lower()

    # Check for local datasets (e.g., coedit) that don't need download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(local_path) and name not in URLS:
        return load_dataset(name)

    assert name in URLS, f"Unknown dataset: {name}. Choose from {list(URLS.keys()) + ['coedit']}"

    raw_path = os.path.join(DATA_DIR, f"{name}.csv")
    processed_path = os.path.join(DATA_DIR, f"{name}.npz")

    if os.path.exists(processed_path):
        print(f"[✓] {name}.npz already exists, loading...")
        return load_dataset(name)

    if not os.path.exists(raw_path):
        print(f"[↓] Downloading {name} from {URLS[name]}...")
        urllib.request.urlretrieve(URLS[name], raw_path)
        print(f"[✓] Downloaded to {raw_path}")

    return preprocess_dataset(name, raw_path)


def preprocess_dataset(name: str, raw_path: str):
    """Parse Jodie CSV format into structured arrays."""
    print(f"[⚙] Preprocessing {name}...")

    df = pd.read_csv(raw_path, skiprows=1, header=None)

    # Jodie format: user_id, item_id, timestamp, state_label, comma-separated features
    sources = df.iloc[:, 0].values.astype(np.int64)
    destinations = df.iloc[:, 1].values.astype(np.int64)
    timestamps = df.iloc[:, 2].values.astype(np.float64)
    labels = df.iloc[:, 3].values.astype(np.int64)

    # Features: remaining columns
    features = df.iloc[:, 4:].values.astype(np.float32)

    # Re-index nodes into a DISJOINT bipartite id space.
    #
    # The Jodie CSV format is BIPARTITE: `user_id` and `item_id` are two
    # INDEPENDENT id namespaces, each numbered from 0. Concatenating them and
    # taking np.unique() merges the two namespaces, so user k and item k are
    # assigned the SAME node id -- they then share a row in every node-indexed
    # store (node memory, echo memory, ...). Instead, give users the low block
    # and offset the items above them:
    #     users -> [0, num_users)
    #     items -> [num_users, num_users + num_items)
    unique_users = np.unique(sources)
    unique_items = np.unique(destinations)
    num_users = len(unique_users)
    num_items = len(unique_items)

    user_map = {old: new for new, old in enumerate(unique_users)}
    item_map = {old: num_users + new for new, old in enumerate(unique_items)}

    sources = np.array([user_map[s] for s in sources], dtype=np.int64)
    destinations = np.array([item_map[d] for d in destinations], dtype=np.int64)

    # Normalize timestamps to start from 0
    timestamps = timestamps - timestamps.min()

    # Sort by time. kind="stable" so that events sharing a timestamp keep their
    # original CSV order -- the default (quicksort) permutes ties differently
    # between runs, which makes the resulting .npz non-reproducible.
    sort_idx = np.argsort(timestamps, kind="stable")
    sources = sources[sort_idx]
    destinations = destinations[sort_idx]
    timestamps = timestamps[sort_idx]
    labels = labels[sort_idx]
    features = features[sort_idx]

    # TRUE total node count: users and items are disjoint, so they add.
    num_nodes = num_users + num_items
    num_edges = len(sources)
    feat_dim = features.shape[1]

    # Integrity: the two namespaces must not overlap after the remap.
    assert len(np.intersect1d(np.unique(sources), np.unique(destinations))) == 0, \
        "user/item id namespaces overlap after remap"

    processed_path = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez(processed_path,
             sources=sources,
             destinations=destinations,
             timestamps=timestamps,
             labels=labels,
             features=features,
             num_nodes=np.array([num_nodes]),
             num_edges=np.array([num_edges]),
             feat_dim=np.array([feat_dim]))

    print(f"[✓] {name}: {num_nodes} nodes ({num_users} users + {num_items} items), "
          f"{num_edges} edges, feat_dim={feat_dim}")
    return load_dataset(name)


def load_dataset(name: str):
    """Load preprocessed dataset."""
    if name == "coedit" and not os.path.exists(os.path.join(DATA_DIR, "coedit.npz")):
        raise FileNotFoundError(
            f"Missing {DATA_DIR / 'coedit.npz'}. See resources/README.md for the "
            "checksum-verified corpus contract. Dataset building is never run "
            "implicitly."
        )
    path = os.path.join(DATA_DIR, f"{name}.npz")
    data = np.load(path)
    # Sanitize features: the Jodie wikipedia.csv has a truncated final row whose
    # trailing feature fields are NaN (edge 157468, cols 149-171). Identity on
    # mooc/coedit (0 NaN/inf), so this does not perturb their numerics.
    features = np.nan_to_num(data["features"].astype(np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "sources": data["sources"],
        "destinations": data["destinations"],
        "timestamps": data["timestamps"],
        "labels": data["labels"],
        "features": features,
        "num_nodes": int(data["num_nodes"][0]),
        "num_edges": int(data["num_edges"][0]),
        "feat_dim": int(data["feat_dim"][0]),
    }


def get_data_splits(data, train_ratio=0.70, val_ratio=0.15):
    """Chronological train/val/test split."""
    n = data["num_edges"]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {}
    for split_name, start, end in [("train", 0, train_end),
                                    ("val", train_end, val_end),
                                    ("test", val_end, n)]:
        splits[split_name] = {
            "sources": data["sources"][start:end],
            "destinations": data["destinations"][start:end],
            "timestamps": data["timestamps"][start:end],
            "labels": data["labels"][start:end],
            "features": data["features"][start:end],
        }

    return splits


if __name__ == "__main__":
    data = load_dataset("coedit")
    splits = get_data_splits(data)
    print(f"  Train: {len(splits['train']['sources'])} edges")
    print(f"  Val:   {len(splits['val']['sources'])} edges")
    print(f"  Test:  {len(splits['test']['sources'])} edges")
