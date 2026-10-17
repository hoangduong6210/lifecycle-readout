"""One owned bounded Colab screen; official test file is never uploaded."""
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.colab_workflow import run_colab_attempt

if __name__ == '__main__':
    settings = tomllib.loads((ROOT / 'configs/tgbseq_baseline_screen_v1.toml').read_text())
    attempt = run_colab_attempt(ROOT, settings, settings['datasets'][0])
    print('Validation screen retrieved; owned runtime release verified:', attempt)
