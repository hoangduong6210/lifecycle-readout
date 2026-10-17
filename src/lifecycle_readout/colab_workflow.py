"""Run a bounded Colab attempt, retrieve artifacts, and verify resource release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile

from .benchmark import load_dataset, save_json, sha256


def run_command(args, *, timeout, transcript, completion_probe=None, poll_seconds=30):
    """No shell interpolation; CLI stdout is retained as execution evidence."""
    with Path(transcript).open("a") as log:
        log.write(json.dumps({"argv": args, "started_unix": time.time()}) + "\n"); log.flush()
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    code = process.wait(timeout=min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    marker = completion_probe() if completion_probe else None
                    if marker is None:
                        continue
                    # Remote archive is closed before its completion marker is
                    # written. A stalled stdout client must not retain a GPU.
                    process.terminate()
                    try:
                        transport_code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); transport_code = process.wait()
                    code = 0 if marker['status'] == 'completed' else 1
                    log.write(json.dumps({'remote_completion_verified': marker,
                        'transport_exit_code': transport_code,
                        'reason': 'completion_marker_overrides_stalled_client'}) + '\n')
                    log.flush()
                    break
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        log.write(json.dumps({"exit_code": code, "ended_unix": time.time()}) + "\n")
    return code


def cli_python(cli):
    cli_path = Path(cli).resolve()
    if cli_path.suffix.lower() == ".exe":
        python_exe = cli_path.parent / "python.exe"
        if not python_exe.is_file():
            python_exe = cli_path.parent.parent / "python.exe"
        if python_exe.is_file():
            return str(python_exe)
        return sys.executable
    try:
        first = cli_path.read_text(encoding="utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return sys.executable
    if not first.startswith("#!"):
        raise RuntimeError("Cannot locate isolated Colab CLI interpreter")
    interpreter = first[2:].strip()
    if not Path(interpreter).is_file():
        raise RuntimeError("Unsupported Colab CLI interpreter layout")
    return interpreter


def assignments(cli):
    code = ("import json; from colab_cli.common import state; "
            "local,remote=state.sync_sessions(); "
            "print(json.dumps([{'endpoint':a.endpoint,'name':next((s.name for s in local.values() if s.endpoint==a.endpoint),None)} for a in remote]))")
    output = subprocess.check_output([cli_python(cli), "-c", code], timeout=60, stdin=subprocess.DEVNULL, text=True)
    return json.loads(output.strip().splitlines()[-1])


def source_bundle(root, target, dataset=None, settings=None):
    settings = settings or {}
    files = [*sorted((root / "src").rglob("*.py")), root / settings.get("entry_script", "scripts/train_tkde.py"),
             root / settings.get("config_path", "configs/tkde_causal_v2.toml"), root / settings.get("protocol", "protocols/lifecycle_tkde_v2.toml"),
             root / "pyproject.toml", root / "configs/dependencies.lock"]
    if settings.get('input_files'):
        for item in settings['input_files']:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != item['sha256']:
                raise ValueError('Prepared input identity mismatch')
            files.append(path)
    elif dataset is not None:
        files.extend([root / "resources/corpora" / f"{dataset}_causal_v2.npz",
                      root / "resources/corpora" / f"{dataset}_causal_v2.identity.json"])
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("source_manifest.json", json.dumps({p.relative_to(root).as_posix(): sha256(p) for p in files}, indent=2))


def prepare_source_upload(bundle, chunk_bytes=64 * 1024 * 1024):
    """Bound notebook uploads; source identity remains that of the complete ZIP."""
    if not 0 < chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError('Source upload chunks must be between 1 byte and 64 MiB')
    bundle = Path(bundle)
    if bundle.stat().st_size <= chunk_bytes:
        return [{'local_path': str(bundle), 'remote_path': '/content/lifecycle-source.zip',
                 'sha256': sha256(bundle), 'bytes': bundle.stat().st_size}]
    pieces = []
    with bundle.open('rb') as source:
        while raw := source.read(chunk_bytes):
            part = bundle.with_name(f'source.part-{len(pieces):04d}')
            with part.open('xb') as out:
                out.write(raw)
            pieces.append({'local_path': str(part), 'remote_path': '/content/' + part.name,
                           'sha256': sha256(part), 'bytes': len(raw)})
    return pieces


def source_assembly_script(parts, target, expected):
    return f'''import pathlib, hashlib, shutil, os
target = pathlib.Path({str(target)!r})
temporary = target.with_name(target.name + '.assembling')
with temporary.open('xb') as assembled:
    for part in {parts!r}:
        path = pathlib.Path(part['remote_path'])
        with path.open('rb') as source:
            if path.stat().st_size != part['bytes'] or hashlib.file_digest(source, 'sha256').hexdigest() != part['sha256']:
                raise RuntimeError('Source upload part identity mismatch')
            source.seek(0)
            shutil.copyfileobj(source, assembled)
with temporary.open('rb') as source:
    if hashlib.file_digest(source, 'sha256').hexdigest() != {expected!r}:
        raise RuntimeError('Assembled source identity mismatch')
os.replace(temporary, target)
'''


def bootstrap_script(dataset, seed_args, smoke, settings=None, upload_parts=None, source_sha256=None):
    settings = settings or {}
    # Run remotely from the uploaded source snapshot; the working tree is never cloned mid-job.
    args = [settings.get("entry_script", "scripts/train_tkde.py"), "--dataset", dataset, "--device", "cuda",
            "--output_dir", "/content/lifecycle-attempt", *seed_args]
    if settings.get("config_path"):
        args.extend(["--config", settings["config_path"]])
    if smoke:
        args.append("--smoke")
    assembly = source_assembly_script(upload_parts, '/content/lifecycle-source.zip', source_sha256) if upload_parts else ''
    return assembly + f'''import hashlib, json, os, pathlib, subprocess, sys, tarfile, traceback, zipfile
root = pathlib.Path("/content/lifecycle-source")
root.mkdir(exist_ok=True)
with zipfile.ZipFile("/content/lifecycle-source.zip") as bundle:
    bundle.extractall(root)
for rel, expected in json.loads((root / "source_manifest.json").read_text()).items():
    if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Uploaded source checksum mismatch: " + rel)
output = pathlib.Path("/content/lifecycle-attempt")
error = None
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "numpy>=2,<3", "pandas>=2.3,<3", "scikit-learn>=1.6,<2",
                    "scipy>=1.13,<2", "tqdm>=4.67,<5"], check=True)
    import torch, numpy, pandas, sklearn
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated runtime has no CUDA GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Environment:", {{"python":sys.version,"torch":torch.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__,"sklearn":sklearn.__version__}}, flush=True)
    training_log = root / "training.txt"
    with training_log.open("w") as log:
        child = subprocess.Popen([sys.executable, "-u", *{args!r}], cwd=root,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = child.wait()
    if code:
        raise RuntimeError("Training process failed with exit code " + str(code))
except BaseException as exc:
    error = exc
    output.mkdir(exist_ok=True)
    (output / "failure.json").write_text(json.dumps({{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}}, indent=2))
finally:
    output.mkdir(exist_ok=True)
    (output / "source_manifest.json").write_bytes((root / "source_manifest.json").read_bytes())
    if (root / "training.txt").exists():
        (output / "training.txt").write_bytes((root / "training.txt").read_bytes())
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"], capture_output=True,text=True)
    (output / "environment.txt").write_text(freeze.stdout)
    manifest = []
    for artifact in sorted(output.rglob("*")):
        if artifact.is_file() and artifact.name != "checksums.sha256":
            manifest.append(hashlib.sha256(artifact.read_bytes()).hexdigest() + "  " + artifact.relative_to(output).as_posix())
    (output / "checksums.sha256").write_text("\\n".join(manifest) + "\\n")
    with tarfile.open("/content/lifecycle-artifacts.tar.gz", "w:gz") as archive:
        archive.add(output, arcname="artifacts")
    # This is outside the archive and written only after the complete archive
    # has closed. It allows independent observation if CLI stdout stalls.
    (pathlib.Path("/content") / "lifecycle-complete.json").write_text(json.dumps({{
        "status": "completed" if error is None else "failed",
        "source_manifest_sha256": hashlib.sha256((root / "source_manifest.json").read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(pathlib.Path("/content/lifecycle-artifacts.tar.gz").read_bytes()).hexdigest()
    }}))
    print("ARTIFACT_ARCHIVE_READY", flush=True)
if error is not None:
    raise RuntimeError("Remote benchmark failed; failure artifact preserved") from error
'''


def extract_artifacts(archive, destination):
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError("Unsafe artifact archive member")
        bundle.extractall(destination, filter="data")


def verify_artifacts(directory):
    manifest = directory / "checksums.sha256"
    if not manifest.exists():
        raise ValueError("Retrieved artifacts have no checksum manifest")
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Retrieved artifact checksum mismatch")


def verify_attempt_outcome(directory):
    if (directory / "failure.json").exists():
        raise RuntimeError("Remote failure artifact present despite CLI exit status")
    results = directory / "results.json"
    if not results.exists():
        raise RuntimeError("Remote results missing despite CLI exit status")
    result = json.loads(results.read_text())
    if not result.get("per_seed_runs"):
        raise RuntimeError("Remote results contain no completed seed records")


def run_colab_attempt(root, settings, dataset, *, seeds=None, smoke=False,
                      executor=run_command, inventory=assignments):
    root = Path(root)
    cli = shutil.which("colab")
    if cli is None:
        raise RuntimeError("Install the pinned Colab CLI before scheduling")
    before = inventory(cli)  # Authentication and inventory before spending compute.
    # Download/parse/compress data locally before allocating a billable GPU.
    # This is data preparation, not local training.
    if executor is run_command:
        print("Preparing and hashing dataset before GPU allocation:", dataset, flush=True)
        if settings.get('input_files'):
            for item in settings['input_files']:
                path = (root / item['path']).resolve()
                if not path.is_relative_to(root.resolve()) or sha256(path) != item['sha256']:
                    raise ValueError('Prepared input identity mismatch before allocation')
        else:
            load_dataset(dataset, root / "resources/corpora")
    name = "lcr-" + dataset + "-" + uuid.uuid4().hex[:10]
    attempt = root / "results/audit" / name
    attempt.mkdir(parents=True, exist_ok=False)
    transcript = attempt / "execution.txt"
    bundle = attempt / "source.zip"
    source_bundle(root, bundle, dataset if executor is run_command else None, settings=settings)
    upload_parts = prepare_source_upload(bundle, settings['compute'].get('source_upload_chunk_bytes', 64 * 1024 * 1024))
    seed_args = ["--seeds", *map(str, seeds)] if seeds else []
    bootstrap = attempt / "remote_job.py"
    bootstrap.write_text(bootstrap_script(dataset, seed_args, smoke, settings=settings,
        upload_parts=upload_parts if len(upload_parts) > 1 else None, source_sha256=sha256(bundle)))
    record = {"schema_version": 2, "session_name": name, "dataset": dataset, "smoke": smoke,
              "requested_seeds": seeds or settings["seeds"], "accelerator": settings["compute"]["accelerator"],
              "source_bundle_sha256": sha256(bundle), "config_sha256": sha256(root / settings.get("config_path", "configs/tkde_causal_v2.toml")),
              "protocol_sha256": sha256(root / settings["protocol"]),
              "status": "allocating", "cleanup_verified": False, "commands": [],
              "baseline_assignments": [a["endpoint"] for a in before]}
    record['source_upload_parts'] = upload_parts
    save_json(attempt / "job.json", record)
    allocated = False
    endpoint = None
    error = None

    with zipfile.ZipFile(bundle) as snapshot:
        expected_manifest = hashlib.sha256(snapshot.read('source_manifest.json')).hexdigest()

    def completion_probe():
        target = attempt / 'remote_completion.json'
        try:
            response = subprocess.run([cli, 'download', '-s', name,
                '/content/lifecycle-complete.json', str(target)], timeout=15,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if response.returncode or not target.is_file():
                return None
            marker = json.loads(target.read_text())
            if (marker.get('status') not in ('completed', 'failed')
                    or marker.get('source_manifest_sha256') != expected_manifest
                    or len(marker.get('archive_sha256', '')) != 64):
                return None
            record['completion_marker'] = marker
            return marker
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    def invoke(args, timeout):
        extra = {'completion_probe': completion_probe} if executor is run_command and args[1] == 'exec' else {}
        code = executor(args, timeout=timeout, transcript=transcript, **extra)
        record["commands"].append({"argv": args, "exit_code": code})
        save_json(attempt / "job.json", record)
        return code

    # SIGTERM must run the same finally path as ordinary failure or interruption.
    previous_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("Colab workflow interrupted")
    signal.signal(signal.SIGTERM, terminate)
    try:
        allocated = True  # A failed allocation can still leave a server-side assignment.
        if invoke([cli, "new", "-s", name, "--gpu", settings["compute"]["accelerator"]], 240):
            raise RuntimeError("Colab allocation failed")
        for item in inventory(cli):
            if item["name"] == name:
                endpoint = item["endpoint"]
        if endpoint is None:
            raise RuntimeError("Cannot verify ownership of allocated runtime")
        record["endpoint"] = endpoint
        for part in upload_parts:
            if invoke([cli, "upload", "-s", name, part['local_path'], part['remote_path']], 180):
                raise RuntimeError("Source upload failed")
        record["status"] = "executing"; save_json(attempt / "job.json", record)
        code = invoke([cli, "exec", "-s", name, "--timeout", str(settings["compute"]["timeout_seconds"]),
                       "-f", str(bootstrap)], settings["compute"]["wall_timeout_seconds"])
        record["execution_exit_code"] = code
        record["status"] = "retrieving"
        if code:
            error = RuntimeError(f"Remote execution failed with exit code {code}")
    except BaseException as exc:
        error = exc
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if allocated:
            try:
                archive = attempt / "artifacts.tar.gz"
                if invoke([cli, "download", "-s", name, "/content/lifecycle-artifacts.tar.gz", str(archive)], 180) == 0:
                    record["archive_sha256"] = sha256(archive)
                    if record.get('completion_marker') and record['completion_marker']['archive_sha256'] != record['archive_sha256']:
                        raise ValueError('Retrieved archive differs from remote completion marker')
                    extract_artifacts(archive, attempt)
                    verify_artifacts(attempt / "artifacts")
                    record["artifact_checksums_verified"] = True
                    verify_attempt_outcome(attempt / "artifacts")
                    record["remote_outcome_verified"] = True
                else:
                    record["retrieval_failed"] = True
            except BaseException as exc:
                record["retrieval_error"] = str(exc)
                if error is None:
                    error = exc
            # Stop only this attempt's session. Verify the endpoint against backend state.
            for retry in range(3):
                try:
                    invoke([cli, "stop", "-s", name], 60)
                    remaining = inventory(cli)
                    owned = [a for a in remaining if a["name"] == name or (endpoint and a["endpoint"] == endpoint)]
                    new_unknown = [a for a in remaining if a["endpoint"] not in record["baseline_assignments"]]
                    if not owned and (endpoint is not None or not new_unknown):
                        record["cleanup_verified"] = True
                        break
                except BaseException as exc:
                    record["cleanup_error"] = str(exc)
            record["status"] = "completed" if error is None and record.get("remote_outcome_verified") else "failed"
            if not record["cleanup_verified"]:
                record["status"] = "cleanup_required"
        record["ended_unix"] = time.time()
        save_json(attempt / "job.json", record)
        signal.signal(signal.SIGTERM, previous_handler)
    if not record["cleanup_verified"]:
        raise RuntimeError(f"Runtime release unverified; inspect {attempt / 'job.json'}")
    if error is not None:
        raise RuntimeError(f"Attempt failed; preserved evidence in {attempt}") from error
    if not record.get("artifact_checksums_verified"):
        raise RuntimeError(f"Artifacts not retrieved; preserved failure in {attempt}")
    return attempt
