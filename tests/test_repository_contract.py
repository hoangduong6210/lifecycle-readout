from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
PACKAGE = "lifecycle_readout"
SIBLING_PACKAGE = "temporal_link_decoupling"
PREFIX = "LCR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"missing front matter: {path}"
    end = text.find("\n---\n", 4)
    assert end > 0, f"unterminated front matter: {path}"
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def _manifest_entries() -> list[dict[str, str]]:
    text = (ROOT / "resources/manifest.toml").read_text(encoding="utf-8")
    entries = []
    for block in text.split("[[dataset]]")[1:]:
        entries.append(dict(re.findall(r'^(\w+)\s*=\s*"([^"]+)"', block, re.MULTILINE)))
    return entries


def test_required_repository_contract() -> None:
    required = [
        "README.md", "AGENTS.md", "PROJECT.toml", "pyproject.toml",
        "results/CURRENT", "paper/CURRENT", "resources/manifest.toml",
        "wiki/README.md", "wiki/INDEX.md", "wiki/START-HERE.md",
        "wiki/LIMITATIONS.md", "wiki/REPRODUCIBILITY.md",
        "wiki/status/Project-Status.md",
        "wiki/claims/Current-Claim-Language.md",
        "wiki/evidence/Evidence-Ledger.md",
        "wiki/governance/License-and-Assets.md",
        "wiki/governance/Numeric-Evidence-and-Publication-Hygiene.md",
        "evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml",
        "evidence/jobs/checksums.sha256",
    ]
    assert not [path for path in required if not (ROOT / path).is_file()]
    project = (ROOT / "PROJECT.toml").read_text(encoding="utf-8")
    assert 'evidence_release = "UNRELEASED"' in project
    assert 'paper_snapshot = "UNRELEASED"' in project
    assert (ROOT / "results/CURRENT").read_text().strip() == "UNRELEASED"
    assert (ROOT / "paper/CURRENT").read_text().strip() == "UNRELEASED"


def test_wiki_front_matter_index_and_links() -> None:
    pages = sorted(WIKI.rglob("*.md"))
    index = (WIKI / "INDEX.md").read_text(encoding="utf-8")
    for page in pages:
        fields = _front_matter(page)
        assert fields.get("title")
        assert fields.get("status")
        assert fields.get("paper_source") in {"true", "false"}
        date_fields = [key for key in ("date", "last_updated") if key in fields]
        assert len(date_fields) == 1
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields[date_fields[0]])
        rel = page.relative_to(WIKI).as_posix()
        assert f"({rel})" in index, f"wiki page absent from INDEX: {rel}"
        text = page.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (page.parent / target.split("#", 1)[0]).resolve()
            assert resolved.is_relative_to(ROOT)
            assert resolved.exists(), f"broken link {target} in {page}"


def test_wiki_identifier_and_governance_contracts() -> None:
    index = (WIKI / "INDEX.md").read_text(encoding="utf-8")
    identity_sources = [
        *WIKI.rglob("*.md"),
        ROOT / "protocols/lifecycle_readout_v1.toml",
        ROOT / "resources/manifest.toml",
    ]
    identity_text = "\n".join(path.read_text(encoding="utf-8") for path in identity_sources)
    ids = set(re.findall(r"\b(?:LCR-(?:RQ|D|P|E|C|H)-[A-Z0-9-]+|DEC-\d{4})\b", identity_text))
    assert ids
    assert not [identifier for identifier in sorted(ids) if f"`{identifier}`" not in index]
    assert "None. `paper/CURRENT`" in index

    claims = (WIKI / "claims/Current-Claim-Language.md").read_text(encoding="utf-8")
    for field in (
        "Exact permitted statement", "Lifecycle status", "Scope and population",
        "Dataset and fidelity", "Metric and uncertainty unit", "Evidence IDs",
        "Execution job",
        "Required qualifiers", "Known limitations", "Paper eligibility",
        "Last review date",
    ):
        assert claims.count(f"**{field}:**") == 4
    assert "## Validated artifacts that are not admitted claims" in claims
    assert "## Blocked or proposed claims" in claims
    assert "## Rejected positive claims" in claims

    evidence = (WIKI / "evidence/Evidence-Ledger.md").read_text(encoding="utf-8")
    for field in (
        "Scientific purpose", "Lifecycle", "Source commit",
        "Protocol, configuration, and data hashes", "Execution identity",
        "Artifact path or release URI", "Artifact checksum", "Coverage and failures",
        "Acceptance-gate outcome", "Supported claim IDs", "Rejected claim IDs",
        "Scientific-use boundary",
    ):
        assert evidence.count(f"**{field}:**") == 4

    decision = (WIKI / "decisions/0001-separate-link-prediction-and-lifecycle-readout.md").read_text()
    for heading in (
        "# DEC-0001", "## Context", "## Options considered", "## Decision",
        "## Scientific consequences", "## Evidence and affected IDs",
        "## Supersedes / superseded by",
    ):
        assert heading in decision


def test_legacy_paper_quarantine_banners() -> None:
    required = [ROOT / "paper/README.md"]
    optional_local = [
        ROOT / "paper/working/Lifecycle_Readout_IEEE.md",
        ROOT / "paper/working/overleaf/body.tex",
    ]
    for path in [*required, *(path for path in optional_local if path.exists())]:
        assert "QUARANTINED LEGACY WORKING" in path.read_text(encoding="utf-8")


def test_claim_evidence_namespace_resolves() -> None:
    claims = (WIKI / "claims/Current-Claim-Language.md").read_text()
    evidence = (WIKI / "evidence/Evidence-Ledger.md").read_text()
    assert not re.search(r"\bLP-[A-Z]", claims + evidence)
    for evidence_id in set(re.findall(rf"{PREFIX}-E-[A-Z0-9-]+", claims)):
        assert evidence_id in evidence


def test_source_boundary_syntax_and_imports() -> None:
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    src_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
    assert SIBLING_PACKAGE not in src_text
    assert "sys.path.insert" not in src_text
    assert "sys.path.append" not in src_text
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(ROOT)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = (
        f"import {PACKAGE}, {PACKAGE}.datasets, {PACKAGE}.training; "
        f"from {PACKAGE}.modeling import sr_gnn_v3_3, fsm_head; "
        f"print({PACKAGE}.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd="/tmp", env=env,
        text=True, capture_output=True, check=True,
    )
    assert Path(result.stdout.strip()).resolve().is_relative_to(ROOT)


def test_json_artifact_hygiene_and_anonymity() -> None:
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    forbidden = {"__pycache__", ".claude", ".pytest_cache"}
    bad_suffixes = {".pyc", ".aux", ".out", ".log"}
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in ignore and "*.py[cod]" in ignore and ".pytest_cache/" in ignore
    for path in ROOT.rglob("*"):
        # Python/pytest create caches while this suite is running. Their exclusion
        # is a VCS contract; filesystem presence during a test is not a violation.
        if forbidden.intersection(path.parts) or path.suffix in {".pyc"}:
            continue
        assert path.suffix not in bad_suffixes

    blind_files = [
        ROOT / "paper/working/overleaf/main_anonymous.tex",
        ROOT / "paper/working/overleaf/body.tex",
        ROOT / "paper/working/overleaf/preamble_common.tex",
    ]
    if any(path.exists() for path in blind_files):
        assert all(path.is_file() for path in blind_files), "incomplete local blind manuscript"
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in blind_files)
        for token in ("Duong Viet Hoang", "Duong Viet Huy", "Lun-Min Shih", "icloud.com", "gmail.com", "mail.dyu"):
            assert token.lower() not in text.lower()


def test_public_files_contain_no_private_paths() -> None:
    excluded = [ROOT / "results/historical", ROOT / "paper/working"]
    suffixes = {".py", ".sh", ".sbatch", ".md", ".toml", ".yaml", ".yml", ".json"}
    pattern = re.compile(r"/(?:users|home|private|scratch)/")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(path.is_relative_to(base) for base in excluded):
            continue
        assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path


def test_runtime_and_historical_checksum_manifests() -> None:
    required_manifests = [
        ROOT / "configs/shared-runtime.sha256",
        ROOT / "resources/checksums.sha256",
        ROOT / "evidence/jobs/checksums.sha256",
    ]
    optional_local_manifests = [
        ROOT / "results/historical/legacy_import/checksums.sha256",
        ROOT / "paper/working/checksums.sha256",
    ]
    local_only_roots = [
        ROOT / "resources/corpora",
        ROOT / "results/historical",
        ROOT / "paper/working",
    ]
    for manifest_path in [*required_manifests, *(path for path in optional_local_manifests if path.exists())]:
        for line in manifest_path.read_text().splitlines():
            expected, rel = line.split(maxsplit=1)
            artifact = ROOT / rel
            if not artifact.exists() and any(artifact.is_relative_to(root) for root in local_only_roots):
                continue
            assert artifact.exists()
            assert _sha256(artifact) == expected


def test_numeric_evidence_jobs_and_publication_boundary() -> None:
    claims = (WIKI / "claims/Current-Claim-Language.md").read_text(encoding="utf-8")
    assert claims.count("**Execution job:**") == 4

    wiki_text = "\n".join(path.read_text(encoding="utf-8") for path in WIKI.rglob("*.md"))
    job_ids = set(re.findall(r"\bLCR-JOB-[A-Z0-9-]+\b", wiki_text))
    assert job_ids
    checksums = (ROOT / "evidence/jobs/checksums.sha256").read_text(encoding="utf-8")
    for job_id in job_ids:
        record = ROOT / "evidence" / "jobs" / f"{job_id}.toml"
        assert record.is_file(), job_id
        text = record.read_text(encoding="utf-8")
        assert f'job_id = "{job_id}"' in text
        assert "source_state" in text and "command" in text and "exit_code" in text
        assert record.relative_to(ROOT).as_posix() in checksums

    quantitative = re.compile(
        r"(?<![A-Za-z0-9_-])(?:\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|pp)|±\s*\d)"
    )
    violations = []
    for page in WIKI.rglob("*.md"):
        for lineno, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            prose = re.sub(r"`[^`]+`", "", line)
            if quantitative.search(prose):
                violations.append(f"{page.relative_to(ROOT)}:{lineno}:{line}")
    assert not violations

    gate = (ROOT / "publication/PUBLICATION_GATE.toml").read_text(encoding="utf-8")
    assert 'status = "BLOCKED"' in gate
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for banned in ("paper/working/", "paper/figs/", "figures/generated/",
                   "results/historical/", "resources/corpora/",
                   "docs/PERPAIR_GLOBALIZATION_DESIGN.md"):
        assert banned in gate
        assert f"/{banned}" in ignore


def test_active_surface_has_no_ai_or_internal_orphan_markers() -> None:
    patterns = re.compile(
        r"(?i)claude|grok|chatgpt|codex|openai|anthropic|gemini|"
        r"PM directive|TESTBENCH|team report|NHIỆM VỤ|flagged to PM|"
        r"reported to PM|human directive|AI[- ]tell|humanization|de-AI|"
        r"job\s*[#:=_-]?\s*\d{5,}"
    )
    violations = []
    for base in (ROOT / "src", ROOT / "experiments", ROOT / "wiki"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".json"}:
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if patterns.search(line):
                        violations.append(f"{path.relative_to(ROOT)}:{lineno}:{line}")
    assert not violations


def test_dataset_manifest_schema_and_coedit_identity() -> None:
    entries = _manifest_entries()
    assert len({entry["id"] for entry in entries}) == len(entries)
    by_id = {entry["id"]: entry for entry in entries}
    data_root = Path(os.environ.get(
        "LIFECYCLE_DATA_DIR", ROOT / "resources" / "corpora"
    ))
    paths = [data_root / Path(entry["path"]).relative_to("corpora") for entry in entries]
    checksum_entries = {}
    for line in (ROOT / "resources/checksums.sha256").read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        checksum_entries[rel] = expected
    for entry in entries:
        assert checksum_entries[f'resources/{entry["path"]}'] == entry["sha256"]
    if not any(path.exists() for path in paths) and "LIFECYCLE_DATA_DIR" not in os.environ:
        import pytest
        pytest.skip("optional local corpora are absent; identities remain in manifest.toml")
    assert all(path.exists() for path in paths), "configured corpus set is incomplete"
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        path = data_root / Path(entry["path"]).relative_to("corpora")
        assert path.is_file()
        assert _sha256(path) == entry["sha256"]
        with np.load(path, allow_pickle=False) as data:
            required = {"sources", "destinations", "timestamps", "labels", "features",
                        "num_nodes", "num_edges", "feat_dim"}
            assert required.issubset(data.files)
            n = int(data["num_edges"][0]); nodes = int(data["num_nodes"][0])
            feat_dim = int(data["feat_dim"][0])
            assert all(len(data[key]) == n for key in ("sources", "destinations", "timestamps", "labels"))
            assert data["features"].shape == (n, feat_dim)
            assert data["sources"].min() >= 0 and data["destinations"].min() >= 0
            assert data["sources"].max() < nodes and data["destinations"].max() < nodes
            assert np.all(np.diff(data["timestamps"]) >= 0)
    assert by_id["LCR-D-COEDIT-001"]["sha256"] == by_id["LCR-D-COEDIT-PREIDFIX-001"]["sha256"]
