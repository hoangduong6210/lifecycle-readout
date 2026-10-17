"""Restore pinned upstream CRAFT assets without executing or relicensing them."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import ssl
import tomllib
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_target(output, relative):
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or '..' in name.parts or '\\' in relative or str(name) != relative:
        raise ValueError('Unsafe upstream relative path')
    target = output.joinpath(*name.parts)
    if not target.resolve().is_relative_to(output.resolve()):
        raise ValueError('Asset path escapes output directory')
    return target


def restore(identity_path, paper_path, output, *, identity_sha256, paper_sha256, fetcher):
    identity_bytes, paper_bytes = identity_path.read_bytes(), paper_path.read_bytes()
    if digest(identity_bytes) != identity_sha256 or digest(paper_bytes) != paper_sha256:
        raise ValueError('Metadata differs from the pinned training configuration')
    identity = json.loads(identity_bytes)
    repository = urlsplit(identity['repository'])
    parts = repository.path.strip('/').split('/')
    commit = identity['commit']
    if (repository.scheme != 'https' or repository.netloc != 'github.com'
            or repository.query or repository.fragment or len(parts) != 2
            or not re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise ValueError('Expected a pinned GitHub repository and commit')
    prefix = f'https://raw.githubusercontent.com/{parts[0]}/{parts[1]}/{commit}/'
    plan = []
    for relative, entry in identity['files'].items():
        target = safe_target(output, relative)
        if entry['url'] != prefix + relative or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Asset URL/hash differs from the pinned repository path')
        plan.append((relative, target, entry['sha256'], entry['url'], None))
    if 'LICENSE' not in identity['files']:
        raise ValueError('Upstream license must be included')
    for name, data in [('source_identity.json', identity_bytes), ('paper_hyperparameters.json', paper_bytes)]:
        if name in identity['files']:
            raise ValueError('Upstream asset conflicts with provenance metadata')
        plan.append((name, safe_target(output, name), digest(data), None, data))
    # Reject unfamiliar existing bytes before retrieving or writing anything.
    for relative, target, expected, _, _ in plan:
        if target.exists() and digest(target.read_bytes()) != expected:
            raise FileExistsError(f'Existing asset differs; preserved: {relative}')
    records = []
    for relative, target, expected, url, metadata in plan:
        existed = target.exists()
        if not existed:
            raw = metadata if metadata is not None else fetcher(url)
            if digest(raw) != expected:
                raise ValueError(f'Upstream checksum mismatch: {relative}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        assert digest(target.read_bytes()) == expected
        records.append(dict(path=relative, sha256=expected, already_present=existed))
    return dict(status='PINNED_ASSETS_RESTORED', repository=identity['repository'], commit=commit,
                files=records, network_download_count=sum(not row['already_present'] for row in records[:-2]),
                code_executed=False, model_fit=False, gpu_allocated=False,
                project_license_granted=False,
                scope='Byte recovery including the original license and historical metadata; not training reproduction or project ownership clearance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/craft_fit_v1.toml')
    parser.add_argument('--identity', type=Path, default=ROOT/'resources/craft_author_source_identity_v1.json')
    parser.add_argument('--paper-settings', type=Path, default=ROOT/'resources/craft_paper_hyperparameters_v1.json')
    parser.add_argument('--ca-file', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Use a fresh report path')
    settings = tomllib.loads(args.config.read_text())
    expected = {Path(row['path']).name: row['sha256'] for row in settings['input_files']}
    context = ssl.create_default_context(cafile=str(args.ca_file) if args.ca_file else None)

    def fetch(url):
        with urlopen(url, context=context, timeout=30) as response:
            raw = response.read(8*1024*1024+1)
        if len(raw) > 8*1024*1024:
            raise ValueError('Unexpected upstream asset size')
        return raw

    result = restore(args.identity, args.paper_settings, args.output.resolve(),
                     identity_sha256=expected['source_identity.json'],
                     paper_sha256=expected['paper_hyperparameters.json'], fetcher=fetch)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['status', 'network_download_count', 'code_executed']}))
