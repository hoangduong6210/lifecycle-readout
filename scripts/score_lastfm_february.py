from pathlib import Path
import argparse
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_holdout_inference import score_holdout

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['lastfm-february'],required=True)
    parser.add_argument('--device',choices=['cuda'],required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    a=parser.parse_args();settings=tomllib.loads(a.config.read_text())
    protocol=tomllib.loads((ROOT/settings['protocol']).read_text())
    score_holdout(ROOT,settings,protocol,a.output_dir,a.device)
