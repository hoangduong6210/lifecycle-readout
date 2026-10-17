"""Run the versioned causal-input benchmark; heavy runs belong on Colab."""
from pathlib import Path
import argparse
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifecycle_readout.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tkde_causal_v2.toml")
    parser.add_argument("--dataset", default="wikipedia", choices=["wikipedia", "reddit", "mooc", "lastfm"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "resources/corpora")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    protocol = tomllib.loads((ROOT / settings["protocol"]).read_text())
    for arg, key in ((args.epochs, "epochs"), (args.batch_size, "batch_size"), (args.seeds, "seeds")):
        if arg is not None:
            settings[key] = arg
    if not args.smoke and args.device != "cuda":
        parser.error("Full benchmark runs require Colab CUDA; use --smoke for local checks")
    result = run_benchmark(args.dataset, settings, protocol, data_dir=args.data_dir,
                           output_dir=args.output_dir, device=args.device, smoke=args.smoke)
    print("Completed attempt; publication eligibility:", result["publication_eligible"])


if __name__ == "__main__":
    main()
