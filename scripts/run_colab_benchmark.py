"""Schedule one bounded accelerator attempt and verify its teardown."""
import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifecycle_readout.colab_workflow import run_colab_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikipedia", "reddit", "mooc", "lastfm"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = tomllib.loads((ROOT / "configs/tkde_causal_v2.toml").read_text())
    attempt = run_colab_attempt(ROOT, settings, args.dataset, seeds=args.seeds, smoke=args.smoke)
    print("Artifacts retrieved and owned runtime release verified:", attempt)


if __name__ == "__main__":
    main()
