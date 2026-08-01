#!/usr/bin/env python3
"""Validate the public release structure and delegate to the scientific validator."""
from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
REQUIRED_ROOT = {".gitignore", "LICENSE", "README.md", "index.html", "main.tex", "main.pdf", "code"}
REQUIRED_CODE = {"README.md", "pyproject.toml", "src", "tests", "scripts", "configs", "data", "results", "manuscript_assets"}
DELEGATE = []
EXPECTED_TITLE = r"\title{Effective-Toggle Persistence for Rolling Forecasts\\"
EXPECTED_TABLES = {"hardware_data.tex", "rolling_folds.tex", "scale_comparison.tex"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root_names = {p.name for p in REPOSITORY.iterdir() if p.name != ".git"}
    missing_root = sorted(REQUIRED_ROOT - root_names)
    extra_root = sorted(root_names - REQUIRED_ROOT)
    code_names = {p.name for p in CODE.iterdir()}
    missing_code = sorted(REQUIRED_CODE - code_names)
    if missing_root or extra_root or missing_code:
        raise SystemExit(
            json.dumps(
                {"missing_root": missing_root, "extra_root": extra_root, "missing_code": missing_code},
                indent=2,
            )
        )
    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    if not manuscript.startswith(r"\documentclass[10pt,letterpaper,twoside]{article}"):
        raise SystemExit("one-column preprint class contract changed")
    if EXPECTED_TITLE not in manuscript:
        raise SystemExit("corrected saturation-audit title is missing")
    normalized = " ".join(manuscript.split())
    required_claims = [
        "reduces mean-squared error relative to the strongest of a stationary toggle fit",
        "not a stationary noise-model validation",
        "stationary fit remains systematically low",
    ]
    missing_claims = [claim for claim in required_claims if claim not in normalized]
    if missing_claims:
        raise SystemExit(f"missing audit conclusions: {missing_claims}")
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", manuscript))
    table_count = len(re.findall(r"\\begin\{table\*?\}", manuscript))
    if (figure_count, table_count) != (4, 2):
        raise SystemExit(
            f"expected 4 evidence figures and 2 tables, got {figure_count} and {table_count}"
        )
    table_dir = CODE / "manuscript_assets" / "tables"
    actual_tables = {path.name for path in table_dir.glob("*.tex")}
    if actual_tables != EXPECTED_TABLES:
        raise SystemExit(
            f"generated table contract mismatch: {sorted(actual_tables)}"
        )
    results_path = CODE / "results" / "results.json"
    locked_path = CODE / "results" / "locked_results.json"
    results = json.loads(results_path.read_text())
    if results != json.loads(locked_path.read_text()):
        raise SystemExit("locked_results.json differs from results.json")
    for toggle, legacy in zip(
        results["hardware_fold_toggle_mses"],
        results["hardware_fold_legacy_mses"],
    ):
        if not toggle < legacy:
            raise SystemExit("saturation does not beat the legacy method on every fold")
    if results["hardware_fold_audited_alphas"] != [0.0] * 4:
        raise SystemExit("audited matrix-free correction weight is not zero in every fold")
    if results["hardware_toggle_compatible_groups"] != 0:
        raise SystemExit("stationary-toggle rejection count changed")
    if not math.isclose(results["hardware_toggle_mse"], 0.000140319292, abs_tol=5e-13):
        raise SystemExit("locked final saturation MSE changed")
    evidence = json.loads((CODE / "evidence_manifest.json").read_text())
    recorded = evidence["release_validation"]
    hash_targets = {
        "manuscript_source_sha256": REPOSITORY / "main.tex",
        "bibliography_sha256": CODE / "manuscript_assets" / "references.bib",
        "manuscript_pdf_sha256": REPOSITORY / "main.pdf",
    }
    for field, path in hash_targets.items():
        if recorded[field] != sha256(path):
            raise SystemExit(f"stale evidence hash for {path}")
    if evidence["locked_results"]["sha256"] != sha256(results_path):
        raise SystemExit("stale results hash in evidence_manifest.json")
    result_manifest = json.loads((CODE / "results" / "manifest.json").read_text())
    if result_manifest["locked_results"]["sha256"] != sha256(locked_path):
        raise SystemExit("stale results hash in code/results/manifest.json")
    if DELEGATE:
        command = list(DELEGATE)
        if command[0] == "python":
            command[0] = sys.executable
        elif shutil.which(command[0]) is None:
            raise SystemExit(
                f"{command[0]!r} is not installed. Run: python -m pip install -e code"
            )
        subprocess.run(command, cwd=REPOSITORY, check=True)
    print("release contract: PASS")


if __name__ == "__main__":
    main()
