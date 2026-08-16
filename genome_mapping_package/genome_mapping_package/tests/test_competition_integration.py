from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_competition_genome_inputs.py"
spec = importlib.util.spec_from_file_location("prepare_competition", SCRIPT)
adapter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


def write_metadata(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_ID", "Strains", "split_final"])
        writer.writerows(rows)


def test_official_metadata_adapter_preserves_sample_order(tmp_path: Path):
    train_val = tmp_path / "train_val.csv"
    test = tmp_path / "test.csv"
    write_metadata(train_val, [("s1", "BAH", "train"), ("s2", "BAI", "val_both")])
    write_metadata(test, [("s3", "CRD", "test_both"), ("s4", "DHY210", "test_time")])
    output = tmp_path / "output"
    manifest = adapter.build_inputs(train_val, test, output)
    arrays = np.load(output / "competition_genome_features.npz", allow_pickle=False)
    assert arrays["sample_ids"].tolist() == ["s1", "s2", "s3", "s4"]
    assert arrays["genome_features"].shape == (4, 24)
    assert arrays["feature_valid_mask"].shape == (4, 24)
    assert manifest["metadata_only"] is True
    assert manifest["proteome_opened"] is False


def test_unknown_competition_strain_fails_closed(tmp_path: Path):
    train_val = tmp_path / "train_val.csv"
    test = tmp_path / "test.csv"
    write_metadata(train_val, [("s1", "UNKNOWN", "train")])
    write_metadata(test, [("s2", "BAH", "test_chem_only")])
    with pytest.raises(ValueError, match="strain absent from frozen index"):
        adapter.build_inputs(train_val, test, tmp_path / "output")


def test_duplicate_sample_id_fails_closed(tmp_path: Path):
    train_val = tmp_path / "train_val.csv"
    test = tmp_path / "test.csv"
    write_metadata(train_val, [("same", "BAH", "train")])
    write_metadata(test, [("same", "BAI", "test_both")])
    with pytest.raises(ValueError, match="duplicate sample_ID"):
        adapter.build_inputs(train_val, test, tmp_path / "output")

