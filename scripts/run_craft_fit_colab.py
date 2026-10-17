"""Run one sealed author-baseline fit with owned-runtime cleanup."""
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.colab_workflow import run_colab_attempt

if __name__ == '__main__':
    settings = tomllib.loads((ROOT / 'configs/craft_fit_v1.toml').read_text())
    attempt = run_colab_attempt(ROOT, settings, 'ML-20M')
    print('CRAFT validation fit retrieved; owned runtime release verified:', attempt)
