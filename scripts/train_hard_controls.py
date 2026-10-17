"""Train the prospective shared-hard-negative control study on Colab CUDA."""
from pathlib import Path
import argparse
import sys
import tomllib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifecycle_readout.hard_controls import run_controls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["wikipedia", "reddit"], required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/hard_controls_v1.toml")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, default=ROOT / "resources/corpora")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.device != "cuda" and not args.smoke:
        p.error("Heavy training requires Colab CUDA")
    settings = tomllib.loads(args.config.read_text())
    if settings.get("evaluation_split", "test") == "validation":
        print("Development run: validation outcomes only; no test scoring.", flush=True)
    if args.seeds:
        settings["seeds"] = args.seeds
    protocol = tomllib.loads((ROOT / settings["protocol"]).read_text())
    run_controls(args.dataset, settings, protocol, data_dir=args.data_dir, output_dir=args.output_dir,
                 device=args.device, smoke=args.smoke)

if __name__ == "__main__":
    main()
