"""Verify pinned file bytes and official schema without scoring held-out data."""
from pathlib import Path
import argparse
import json
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import sha256
from lifecycle_readout.official_data import audit_stream, inspect_npy_header

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'resources/tgbseq_ml20m_v1.toml')
    parser.add_argument('--prepared-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.prepared_output = args.prepared_output.resolve()
    args.output = args.output.resolve()
    if args.output.exists(): parser.error('Use a fresh audit path')
    if args.prepared_output.exists(): parser.error('Use a fresh prepared-data path')
    if not args.prepared_output.is_relative_to(ROOT):
        parser.error('Prepared data must be inside this repository')
    manifest = tomllib.loads(args.manifest.read_text())
    files = {}
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert path.stat().st_size == entry['size_bytes'] and sha256(path) == entry['sha256']
        files[path.suffix] = path
    result = audit_stream(files['.csv'], prepared_output=args.prepared_output)
    negatives = inspect_npy_header(files['.npy'])
    assert negatives['shape'] == [result['official_split_counts']['2'], 100]
    result.update(dataset_id=manifest['dataset_id'], source_manifest_sha256=sha256(args.manifest),
                  negative_header=negatives, prepared_sha256=sha256(args.prepared_output),
                  prepared_path=str(args.prepared_output.relative_to(ROOT)),
                  all_structural_checks_pass=True, scientific_admission=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
