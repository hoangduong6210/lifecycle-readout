from pathlib import Path
import argparse,sys,tomllib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.lastfm_model_training import run_lastfm_model
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['lastfm-original'],required=True);p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);args=p.parse_args();settings=tomllib.loads(args.config.read_text());protocol=tomllib.loads((ROOT/settings['protocol']).read_text());run_lastfm_model(settings,protocol,args.output_dir,device=args.device)
