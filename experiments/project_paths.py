"""Canonical project-local paths for directly executable experiment scripts."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
LEGACY_RESULTS_DIR = PROJECT_ROOT / "results" / "historical" / "legacy_import"
GENERATED_FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"


def result_input(name: str) -> Path:
    """Resolve an active input without silently mixing in quarantined evidence."""
    root = Path(os.environ.get("LIFECYCLE_RESULT_INPUT_DIR", AUDIT_DIR))
    return root / name
