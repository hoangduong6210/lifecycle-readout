"""Verify public report hashes, overview values, figure data and repository links.

This checks distributed aggregate records. It does not replay undistributed
checkpoints or establish scientific admission.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(name):
    with (ROOT / "paper/working/article" / name).open() as stream:
        return list(csv.DictReader(stream))


def same(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=2e-12), (actual, expected)


def verify():
    manifest = read("evidence/publication-manifest.json")
    for entry in manifest["artifacts"]:
        p = ROOT / entry["path"]
        assert p.is_file(), entry["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["public_sha256"], entry["path"]
        if not entry["host_paths_redacted"]:
            assert entry["source_sha256"] == entry["public_sha256"], entry["path"]
    paths = read("results/reports/scored-paths.json")
    repair = read("results/reports/rate-repair.json")
    feb = read("results/reports/february-holdout.json")
    jan = read("results/reports/january-cohorts.json")
    total = sum(p["candidate_trajectories"] for d in paths["datasets"].values() for p in d["per_seed"])
    assert total == repair["candidate_trajectories"]
    changes = sum(p["all_rate_changed_candidates"] for d in repair["datasets"].values() for p in d["per_seed"])
    assert changes == repair["all_rate_changed_candidates"]
    assert all(p["fixed_changed_candidates"] == 0 for d in repair["datasets"].values() for p in d["per_seed"])
    assert all(path["violation_count"] == 0 for d in paths["datasets"].values() for p in d["per_seed"] for path in p["path_results"] if path["path"] == "implemented_flow")
    assert feb["local_confirmation_screen"] == "NOT_ESTABLISHED"
    assert feb["conditional_bootstrap_interval"][0] < 0 < feb["conditional_bootstrap_interval"][1]
    assert sum(x > 0 for x in feb["primary"]["values"]) < len(feb["primary"]["values"])

    # Independently connect article CSV counts and endpoints to the JSON owners.
    figure6 = rows("Figure6_data.csv")
    for row in figure6:
        dataset = row.get("Dataset", row.get("dataset", "")).lower()
        seed = int(row.get("Seed", row.get("seed", "0")))
        owner = next(p for p in paths["datasets"][dataset]["per_seed"] if p["seed"] == seed)
        lookup = {p["path"]:p for p in owner["path_results"]}
        label = row["Path"]
        mapping = {"Flow only":"implemented_flow", "Frozen map":"fixed_transition", "Dynamic map":"dynamic_transition"}
        assert label in mapping, label
        path = lookup[mapping[label]]
        same(row["Violating_trajectories"], path["violation_count"])
        same(row["Candidate_trajectories"], owner["candidate_trajectories"])
    assert len(figure6) == sum(len(d["per_seed"]) for d in paths["datasets"].values()) * len(mapping)
    for row in rows("Figure7_data.csv"):
        dataset, seed = row["Dataset"], int(row["Seed"])
        owner=next(p for p in repair["datasets"][dataset]["per_seed"] if p["seed"]==seed)
        same(row["Candidates"],owner["candidate_count"])
        same(row["All_rate_changes"],owner["all_rate_changed_candidates"])
        same(row["Specified_rate_changes"],owner["fixed_changed_candidates"])
        same(row["All_rate_squared_distance"],owner["mean_squared_distance_all_rate"])
        same(row["Specified_rate_squared_distance"],owner["mean_squared_distance_fixed"])
    months={"January":jan["aggregate"]["gap_1d_to_7d__dose_1_to_9"]["ap_contrasts"]["primary_shape"]["values"],"February":feb["primary"]["values"]}
    month_rows=rows("Figure8_data.csv")
    for month,values in months.items():
        actual=[row for row in month_rows if row["Month"]==month]
        assert len(actual)==len(values)
        for row,expected in zip(actual,values): same(row["AP_difference"],expected)
    for row in rows("Figure8_interval.csv"):
        if row["Month"] == "February":
            same(row["Mean_AP_difference"], feb["primary"]["mean"])
            same(row["Lower_95"], feb["conditional_bootstrap_interval"][0])
            same(row["Upper_95"], feb["conditional_bootstrap_interval"][1])
        else:
            same(row["Mean_AP_difference"], jan["aggregate"]["gap_1d_to_7d__dose_1_to_9"]["ap_contrasts"]["primary_shape"]["mean"])

    plot = read("figures/generated/overview/plot-data.json")
    for name, expected in plot["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    same(plot["february_subgroup_mean_delta_ap"], feb["primary"]["mean"])
    assert plot["february_conditional_interval"] == feb["conditional_bootstrap_interval"]

    # The README's text and linked values must match the evidence-driven generator.
    spec = importlib.util.spec_from_file_location("overview_builder", ROOT/"scripts/build_readme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT/"README.md").read_text() == module.render(), "Regenerate the README from its evidence."

    broken=[]; count=0
    active=[ROOT/"README.md", *(ROOT/"wiki").rglob("*.md"), ROOT/"paper/README.md", ROOT/"figures/README.md"]
    for p in active:
        for link in re.findall(r'!?\[[^]]*\]\(([^)]+)\)', p.read_text()):
            if '://' in link or link.startswith('#'): continue
            destination=(p.parent/link.split('#')[0]).resolve()
            count+=1
            if not destination.is_relative_to(ROOT) or not destination.exists():broken.append((str(p.relative_to(ROOT)),link))
    assert not broken, broken
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in (ROOT/'README.md').read_text())
    return {"distributed_evidence_records":len(manifest["artifacts"]), "markdown_links":count, "wiki_pages":len(list((ROOT/'wiki').rglob('*.md'))), "reported_candidate_trajectories":total, "status":"PASS"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
