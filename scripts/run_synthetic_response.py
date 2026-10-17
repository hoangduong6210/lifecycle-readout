"""Run a small synthetic oracle diagnostic with archived protocol and source."""
from pathlib import Path
import argparse
import json
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.synthetic_response import run_synthetic
from lifecycle_readout.benchmark import sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/q1_synthetic_v1.toml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    sources = [args.config.resolve(), ROOT / settings['protocol'], Path(__file__).resolve(),
               ROOT / 'pyproject.toml', ROOT / 'configs/dependencies.lock',
               *sorted((ROOT / 'src').rglob('*.py'))]
    # Read all exact source bytes before running, then retain those same bytes.
    snapshot = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    run_synthetic(settings, protocol, args.output)
    for rel, content in snapshot.items():
        dest = args.output / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    manifest = {str(p.relative_to(args.output / 'source')): sha256(p)
                for p in (args.output / 'source').rglob('*') if p.is_file()}
    (args.output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'checksums.sha256').write_text(''.join(
        f'{sha256(p)}  {p.relative_to(args.output)}\n' for p in sorted(args.output.rglob('*')) if p.is_file()))
