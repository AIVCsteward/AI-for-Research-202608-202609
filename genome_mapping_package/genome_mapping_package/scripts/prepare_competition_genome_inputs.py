#!/usr/bin/env python
"""Map official GOAI metadata rows to frozen yeast genome features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENOME_DIR = ROOT / "external_data" / "genome"
CONTRACT = ROOT / "competition_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"metadata has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_inputs(train_val_path: Path, test_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    required = set(contract["required_columns"])

    index_fields, index_rows = read_csv(GENOME_DIR / "strain_feature_index.csv")
    del index_fields
    lookup = {row["strain_id"]: row for row in index_rows}
    arrays = np.load(GENOME_DIR / "strain_features.npz", allow_pickle=False)

    all_rows: list[dict[str, str]] = []
    source_file_index: list[int] = []
    source_row_index: list[int] = []
    input_manifest: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()

    for file_index, (role, path) in enumerate(
        (("train_val", train_val_path), ("test", test_path))
    ):
        fields, rows = read_csv(path)
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{role} metadata is missing columns: {missing}")
        for row_index, row in enumerate(rows):
            sample_id = row["sample_ID"]
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_ID across metadata files: {sample_id}")
            seen_sample_ids.add(sample_id)
            if row["Strains"] not in lookup:
                raise ValueError(f"strain absent from frozen index: {row['Strains']}")
            all_rows.append(row)
            source_file_index.append(file_index)
            source_row_index.append(row_index)
        input_manifest.append(
            {
                "role": role,
                "path_name": path.name,
                "row_count": len(rows),
                "sha256": sha256_file(path),
            }
        )

    strain_rows = np.asarray(
        [int(lookup[row["Strains"]]["row_index"]) for row in all_rows], dtype=np.int64
    )
    features = arrays["genome_features"][strain_rows].astype(np.float32, copy=False)
    valid_mask = arrays["feature_valid_mask"][strain_rows].astype(bool, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("non-finite genome feature detected")
    if np.any(features[~valid_mask] != 0):
        raise ValueError("invalid genome feature cells must be numeric zero")

    output_dir.mkdir(parents=True)
    npz_path = output_dir / "competition_genome_features.npz"
    np.savez_compressed(
        npz_path,
        sample_ids=np.asarray([row["sample_ID"] for row in all_rows], dtype="U128"),
        strain_ids=np.asarray([row["Strains"] for row in all_rows], dtype="U32"),
        genome_features=features,
        feature_valid_mask=valid_mask,
        source_file_index=np.asarray(source_file_index, dtype=np.int8),
        source_row_index=np.asarray(source_row_index, dtype=np.int64),
    )

    mapping_rows: list[dict[str, object]] = []
    for row, frozen_row in zip(all_rows, strain_rows, strict=True):
        strain = lookup[row["Strains"]]
        mapping_rows.append(
            {
                "sample_ID": row["sample_ID"],
                "split_final": row["split_final"],
                "strain_id": row["Strains"],
                "genome_row_index": int(frozen_row),
                "external_isolate_id": strain["external_isolate_id"],
                "mapping_type": strain["mapping_type"],
                "mapping_confidence": strain["mapping_confidence"],
                "proxy_flag": strain["proxy_flag"],
            }
        )
    mapping_path = output_dir / "competition_sample_strain_mapping.csv"
    mapping_columns = [
        "sample_ID", "split_final", "strain_id", "genome_row_index",
        "external_isolate_id", "mapping_type", "mapping_confidence", "proxy_flag",
    ]
    write_csv(mapping_path, mapping_rows, mapping_columns)

    split_counts = dict(sorted(Counter(row["split_final"] for row in all_rows).items()))
    manifest = {
        "schema_version": "1.0",
        "competition": contract["competition"],
        "metadata_only": True,
        "proteome_opened": False,
        "inputs": input_manifest,
        "sample_count": len(all_rows),
        "strain_count": len(set(row["Strains"] for row in all_rows)),
        "split_counts": split_counts,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_schema_sha256": sha256_file(GENOME_DIR / "genome_feature_schema.json"),
        "strain_index_sha256": sha256_file(GENOME_DIR / "strain_feature_index.csv"),
        "outputs": {
            "competition_genome_features.npz": sha256_file(npz_path),
            "competition_sample_strain_mapping.csv": sha256_file(mapping_path),
        },
    }
    manifest_path = output_dir / "competition_genome_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-train-val", type=Path, required=True)
    parser.add_argument("--metadata-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_inputs(
        args.metadata_train_val.resolve(), args.metadata_test.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

