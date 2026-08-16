#!/usr/bin/env python
"""Audit the private/open-knowledge reproduction package before upload."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REQUIRED = (
    "README.md",
    "requirements.txt",
    "LICENSE_DECISION_REQUIRED.md",
    "docs/OPEN_KNOWLEDGE_DISCLOSURE.md",
    "docs/DATA_USE_NOTICE.md",
    "docs/THIRD_PARTY_SOFTWARE.md",
    "docs/MODEL_CARD.md",
    "scripts/run_inference.py",
    "baseline/baseline/final_submission_v2.py",
    "project_v2/data_contract/feature_contract.json",
    "external_data/chemistry/chemical_features.npz",
    "external_data/chemistry/source_manifest.json",
    "external_data/genome/strain_features.npz",
    "external_data/genome/source_manifest.json",
    "reports/model_v2_stage_s2c/checkpoint_manifest.json",
    "reports/model_v2_stage_s2d/checkpoint_manifest.json",
)

FORBIDDEN_NAME_PARTS = (
    "proteome_raw",
    "metadata_train_val",
    "metadata_test",
    "prediction.csv",
    "routing_manifest.csv",
    "experimental_parity_fc_pairing.csv",
    "holdout_ids.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_has_sample_id(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return "sample_ID" in next(csv.reader(stream), [])
    except UnicodeDecodeError:
        return False


def audit(root: Path) -> dict:
    root = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    forbidden = []
    sample_level_csv = []
    oversized = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if relative != "WAYB_WAYC/README.md" and any(token in lowered for token in FORBIDDEN_NAME_PARTS):
            forbidden.append(relative)
        if path.suffix.lower() == ".csv" and csv_has_sample_id(path):
            sample_level_csv.append(relative)
        if path.stat().st_size >= 100 * 1024 * 1024:
            oversized.append(relative)
    checks = {
        "required_files_present": not missing,
        "no_raw_or_generated_competition_files": not forbidden,
        "no_sample_level_csv": not sample_level_csv,
        "no_file_at_or_above_github_100mb": not oversized,
        "genome_raw_cache_not_distributed": not (root / "external_data/genome/raw_cache").exists(),
    }
    return {
        "schema_version": "1.0",
        "technical_status": "PASS" if all(checks.values()) else "FAIL",
        "public_release_status": "BLOCKED_PENDING_TEAM_LICENSE_AND_ENTITY_INDEX_PERMISSION",
        "checks": checks,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "missing": missing,
        "forbidden_files": forbidden,
        "sample_level_csv": sample_level_csv,
        "oversized_files": oversized,
    }


def write_manifest(root: Path) -> Path:
    root = root.resolve()
    target = root / "PACKAGE_MANIFEST.json"
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != target):
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    target.write_text(json.dumps({"schema_version": "1.0", "files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest(args.root)
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["technical_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
