import json
from pathlib import Path
import shutil
import sys
import tarfile
import tomllib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifecycle_readout import colab_workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fail_at", [None, "upload", "exec", "download"])
@pytest.mark.parametrize('chunked', [False, True])
def test_attempt_always_stops_only_owned_runtime_and_retains_failure(tmp_path, monkeypatch, fail_at, chunked):
    settings = tomllib.loads((ROOT / "configs/tkde_causal_v2.toml").read_text())
    if chunked:
        settings['compute']['source_upload_chunk_bytes'] = 4096
    for rel in ("src", "configs", "protocols", "scripts"):
        shutil.copytree(ROOT / rel, tmp_path / rel)
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    monkeypatch.setattr(shutil, "which", lambda _: "colab")
    calls, active, holder = [], [], {}
    def inventory(_):
        return [{"name": "unrelated-session", "endpoint": "unrelated-endpoint"}, *active]
    def executor(args, **kwargs):
        calls.append(args)
        op = args[1]
        if op == "new":
            name = args[args.index("-s") + 1]; holder["name"] = name
            active.append({"name": name, "endpoint": "owned-endpoint"})
        elif op == "stop":
            assert args[args.index("-s") + 1] == holder["name"]
            active.clear()
        elif op == "download" and fail_at != op:
            source = tmp_path / "result-source"; source.mkdir()
            (source / "results.json").write_text('{"per_seed_runs": [{"seed": 42}]}')
            (source / "checksums.sha256").write_text(colab_workflow.sha256(source / "results.json") + "  results.json\n")
            with tarfile.open(args[-1], "w:gz") as archive:
                archive.add(source, arcname="artifacts")
        return 1 if op == fail_at else 0
    if fail_at:
        with pytest.raises(RuntimeError):
            colab_workflow.run_colab_attempt(tmp_path, settings, "wikipedia", executor=executor, inventory=inventory)
    else:
        attempt = colab_workflow.run_colab_attempt(tmp_path, settings, "wikipedia", executor=executor, inventory=inventory)
        assert (attempt / "artifacts/results.json").exists()
    assert any(c[1] == "stop" for c in calls)
    job = next((tmp_path / "results/audit").glob("*/job.json"))
    record = json.loads(job.read_text())
    assert record["cleanup_verified"]
    assert record["status"] == ("failed" if fail_at else "completed")
    if chunked and fail_at != 'upload':
        assert sum(c[1] == 'upload' for c in calls) > 1


def test_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        member = tarfile.TarInfo("../outside")
        out.addfile(member)
    with pytest.raises(ValueError, match="Unsafe"):
        colab_workflow.extract_artifacts(archive, tmp_path / "target")


def test_cli_zero_exit_does_not_override_remote_failure_artifact(tmp_path):
    (tmp_path / "failure.json").write_text('{"type":"RuntimeError"}')
    with pytest.raises(RuntimeError, match="Remote failure artifact"):
        colab_workflow.verify_attempt_outcome(tmp_path)


@pytest.mark.parametrize('status, expected', [('completed', 0), ('failed', 1)])
def test_stalled_client_is_terminated_on_verified_remote_completion(tmp_path, status, expected):
    transcript = tmp_path / 'execution.txt'
    marker = {'status': status, 'archive_sha256': 'a' * 64}
    code = colab_workflow.run_command([sys.executable, '-c', 'import time; time.sleep(60)'],
        timeout=2, transcript=transcript, completion_probe=lambda: marker, poll_seconds=.02)
    assert code == expected
    events = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert events[1]['remote_completion_verified'] == marker
    assert events[1]['transport_exit_code'] != 0
    assert events[1]['reason'] == 'completion_marker_overrides_stalled_client'


def test_missing_completion_marker_does_not_turn_timeout_into_success(tmp_path):
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        colab_workflow.run_command([sys.executable, '-c', 'import time; time.sleep(60)'],
            timeout=.1, transcript=tmp_path / 'execution.txt',
            completion_probe=lambda: None, poll_seconds=.02)


def test_declared_prepared_input_bundle_excludes_raw_test_candidates(tmp_path):
    import zipfile
    settings = tomllib.loads((ROOT / 'configs/hard_controls_v1.toml').read_text())
    settings['input_files'] = []
    for rel in ('src', 'configs', 'protocols', 'scripts'):
        shutil.copytree(ROOT / rel, tmp_path / rel)
    shutil.copy(ROOT / 'pyproject.toml', tmp_path / 'pyproject.toml')
    prepared = tmp_path / 'resources/corpora/development.npz'
    prepared.parent.mkdir(parents=True); prepared.write_bytes(b'train-validation-fixture')
    settings['input_files'] = [{'path': str(prepared.relative_to(tmp_path)),
                                'sha256': colab_workflow.sha256(prepared)}]
    target = tmp_path / 'source.zip'
    colab_workflow.source_bundle(tmp_path, target, 'ML-20M', settings=settings)
    with zipfile.ZipFile(target) as archive:
        assert 'resources/corpora/development.npz' in archive.namelist()
        assert not any('_test_ns' in name or name.startswith('resources/staging/') for name in archive.namelist())


def test_bad_prepared_input_fails_before_runtime_allocation(tmp_path, monkeypatch):
    settings = {'input_files': [{'path': 'bad.npz', 'sha256': '0' * 64}]}
    (tmp_path / 'bad.npz').write_bytes(b'modified input')
    monkeypatch.setattr(shutil, 'which', lambda _: 'colab')
    with pytest.raises(ValueError, match='before allocation'):
        colab_workflow.run_colab_attempt(tmp_path, settings, 'ML-20M', inventory=lambda _: [])


@pytest.mark.parametrize('corrupt', [False, True])
def test_chunked_source_reassembly_requires_all_hashes_before_install(tmp_path, corrupt):
    import subprocess
    source = tmp_path / 'source.zip'
    source.write_bytes(b'full source with exact binary bytes\x00\xff')
    parts = colab_workflow.prepare_source_upload(source, chunk_bytes=7)
    assert len(parts) > 1 and max(p['bytes'] for p in parts) <= 7
    for part in parts:
        part['remote_path'] = part['local_path']
    if corrupt:
        Path(parts[1]['local_path']).write_bytes(b'changed')
    target = tmp_path / 'installed.zip'
    script = colab_workflow.source_assembly_script(parts, target, colab_workflow.sha256(source))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    if corrupt:
        assert result.returncode != 0 and not target.exists()
        assert 'identity mismatch' in result.stderr
    else:
        assert result.returncode == 0 and target.read_bytes() == source.read_bytes()
