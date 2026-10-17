from pathlib import Path
import argparse,sys,tomllib,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lifecycle_readout.rate_repair_validation import run_study,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.config=a.config.resolve();a.output=a.output.resolve();settings=tomllib.loads(a.config.read_text());protocol=tomllib.loads((ROOT/settings['protocol']).read_text())
    paths=[a.config,Path(__file__),ROOT/'src/lifecycle_readout/rate_repair_validation.py',ROOT/'src/lifecycle_readout/modeling/inverse_tilt_projection.py']
    pins={str(x.relative_to(ROOT)):sha(x) for x in paths};(a.output/'execution_source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    for x in paths:shutil.copy2(x,a.output/str(x.relative_to(ROOT)).replace('/','__'))
    run_study(ROOT,settings,protocol,a.output)
    assert all(sha(ROOT/name)==h for name,h in pins.items()),'Diagnostic source/config changed during execution'
