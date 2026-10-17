"""Thin Colab entry point for the independently versioned competition pilot."""
from pathlib import Path
import argparse
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lifecycle_readout.event_coupling_study import run_event_coupling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikipedia"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/event_coupling_v1.toml")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=ROOT/"resources/corpora")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    if args.seeds:
        settings["seeds"] = args.seeds
    protocol = tomllib.loads((ROOT/settings["protocol"]).read_text())
    run_event_coupling(args.dataset, settings, protocol,
                    output_dir=args.output_dir, device=args.device, smoke=args.smoke)
