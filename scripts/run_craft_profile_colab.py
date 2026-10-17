"""Profile the pinned author interface on one owned, bounded Colab runtime."""
from pathlib import Path
import argparse
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.colab_workflow import run_colab_attempt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/craft_profile_v1.toml')
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    attempt = run_colab_attempt(ROOT, settings, 'ML-20M')
    print('Author feasibility profile retrieved; owned runtime release verified:', attempt)
