#!/usr/bin/env python
"""Load one frozen GOAI strain feature row without competition labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENOME_DIR = ROOT / "external_data" / "genome"


def _read_index() -> list[dict[str, str]]:
    with (GENOME_DIR / "strain_feature_index.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def load_strain_feature(strain_id: str) -> dict[str, object]:
    rows = _read_index()
    lookup = {row["strain_id"]: row for row in rows}
    if strain_id not in lookup:
        supported = ", ".join(lookup)
        raise ValueError(
            f"strain {strain_id!r} is absent from the frozen index; "
            f"supported strains: {supported}"
        )

    row = lookup[strain_id]
    row_index = int(row["row_index"])
    arrays = np.load(GENOME_DIR / "strain_features.npz", allow_pickle=False)
    schema = json.loads(
        (GENOME_DIR / "genome_feature_schema.json").read_text(encoding="utf-8")
    )
    if schema["strain_id_order"][row_index] != strain_id:
        raise ValueError("strain index and schema order disagree")

    return {
        "strain_id": strain_id,
        "external_isolate_id": row["external_isolate_id"],
        "mapping_type": row["mapping_type"],
        "mapping_confidence": row["mapping_confidence"],
        "proxy_flag": row["proxy_flag"].lower() == "true",
        "features": arrays["genome_features"][row_index].copy(),
        "valid_mask": arrays["feature_valid_mask"][row_index].copy(),
        "feature_names": tuple(schema["genome_features"]["feature_names"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strain", required=True)
    args = parser.parse_args()
    record = load_strain_feature(args.strain)
    summary = {
        "strain_id": record["strain_id"],
        "external_isolate_id": record["external_isolate_id"],
        "mapping_type": record["mapping_type"],
        "mapping_confidence": record["mapping_confidence"],
        "proxy_flag": record["proxy_flag"],
        "feature_dim": int(record["features"].shape[0]),
        "valid_feature_count": int(record["valid_mask"].sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

