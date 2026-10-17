"""Run one bounded hard-negative study and verify release of its Colab GPU."""
from pathlib import Path
import argparse
import sys
import tomllib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifecycle_readout.colab_workflow import run_colab_attempt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikipedia", "reddit"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hard_controls_v1.toml")
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    attempt = run_colab_attempt(ROOT, settings, args.dataset, seeds=args.seeds, smoke=args.smoke)
    print("Hard-control artifacts verified; runtime release verified:", attempt)
