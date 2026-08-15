"""Verify the portable V2 package, frozen checkpoints and upload safety."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KNOWN_HASHES = {
    "reports/model_v2_stage_s2b/formal_d0_hierarchical_none_huber_seed_20260814/stage_b_best.pt": "a216cb68a9dc345a6b1a1a9cb03c033d2e1570bb3d4417e36655d798b36fb56b",
    "reports/model_v2_stage_s2b/formal_d2_hierarchical_morgan_fc_seed_20260814/stage_b_best.pt": "e2db629ae549769b996d86b55c101d9dd2cd68e7dfb96887a9ad91e737f2b209",
    "reports/model_v2_stage_s1/formal_c_anchor_morgan_fc_seed_20260814/stage_b_best.pt": "8af52cd79dade3464dae8c46710f104b7d7f0731a2a737bb29f306c9fd92ebc7",
    "reports/model_v2_stage_s2d/formal_d0_hierarchical_none_huber_seed_20260815/stage_b_best.pt": "1b82d4e0c52683b1bef271e22b66035eda17ea64db0843dd8f913216e7d8df14",
    "reports/model_v2_stage_s2d/formal_d2_hierarchical_morgan_fc_seed_20260815/stage_b_best.pt": "6d089faa863ef1615d4760cfee4f715a3addec46b317fd5040fddd0019475891",
    "reports/model_v2_stage_s2d/formal_s1c_flat_morgan_fc_seed_20260815/stage_b_best.pt": "1dd79ca80091c5dd3ca20bb78d3e47445e67f710e2a2cb096e27c6f5049f4aa4",
    "project_v2/data_contract/feature_contract.json": "9ef3685f16796c949a7ec4b7c2b3f4b1c157ce374970ebcbc77f832dcf26f4f8",
    "external_data/chemistry/source_manifest.json": "128b63ff38283ecf848ba3875b5f7f2ee19caeb82d260c347cebc21342d768a2",
    "external_data/genome/source_manifest.json": "703f8289daf0069a6fafba41851bfac4c22be424be2926b9d2536676c692e815",
    "reports/model_v2_stage2/formal_huber_no_batch_seed_20260814/stage_a_best.pt": "0a2141098c1a6476673ab83f1790da3bccecaade034bd15d1e7bc87fd9035155",
}

REQUIRED = (
    "baseline/baseline/model_v2.py",
    "baseline/baseline/final_submission_v2.py",
    "baseline/baseline/novelty_gated_ensemble_v2.py",
    "baseline/baseline/second_seed_ensemble_v2.py",
    "external_data/chemistry/chemical_features.npz",
    "external_data/genome/strain_features.npz",
    "reports/model_v2_stage_s2a/final/hierarchical_batch/stage_a_final.pt",
    "reports/model_v2_stage_s2d/stage_a_hierarchical_seed_20260815/stage_a_final.pt",
    "reports/model_v2_stage2/formal_huber_seed_20260814/stage_a_best.pt",
    "reports/model_v2_stage_s2d/stage_a_flat_seed_20260815/stage_a_best.pt",
    *KNOWN_HASHES.keys(),
)

FORBIDDEN_NAME_TOKENS = (
    "proteome_raw_train_val",
    "proteome_raw_test",
    "metadata_train_val(1).csv",
    "metadata_test(1).csv",
)
FORBIDDEN_PATH_TOKENS = (
    "yeast_functional_pretraining",
    "networx_phase_b",
)
MAX_GITHUB_FILE_BYTES = 100_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.json":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def audit(root: Path) -> dict:
    rows = inventory(root)
    by_path = {row["path"]: row for row in rows}
    missing = sorted(path for path in REQUIRED if path not in by_path)
    hash_failures = {
        path: {"expected": expected, "actual": by_path.get(path, {}).get("sha256")}
        for path, expected in KNOWN_HASHES.items()
        if by_path.get(path, {}).get("sha256") != expected
    }
    forbidden = []
    oversized = []
    for row in rows:
        lower = row["path"].lower()
        if any(token in Path(lower).name for token in FORBIDDEN_NAME_TOKENS):
            forbidden.append(row["path"])
        if any(token in lower for token in FORBIDDEN_PATH_TOKENS):
            forbidden.append(row["path"])
        if row["bytes"] >= MAX_GITHUB_FILE_BYTES:
            oversized.append(row["path"])
    checks = {
        "required_files_present": not missing,
        "frozen_hashes_match": not hash_failures,
        "no_competition_or_restricted_files": not forbidden,
        "no_file_at_or_above_github_100mb_limit": not oversized,
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "missing": missing,
        "hash_failures": hash_failures,
        "forbidden_files": sorted(set(forbidden)),
        "oversized_files": oversized,
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    if args.write_manifest:
        (root / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
