from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "external_data" / "chemistry"
SCRIPT = ROOT / "scripts" / "build_chemical_features.py"
TRAIN_VAL = ROOT / "WAYB_WAYC" / "WAYB_WAYC_metadata_train_val(1).csv"
TEST_META = ROOT / "WAYB_WAYC" / "WAYB_WAYC_metadata_test(1).csv"
OVERRIDES = OUTPUT / "manual_overrides.csv"

spec = importlib.util.spec_from_file_location("build_chemical_features", SCRIPT)
chem = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = chem
spec.loader.exec_module(chem)


@pytest.fixture(scope="module")
def artifacts():
    mapping = pd.read_csv(OUTPUT / "compound_mapping.csv", keep_default_na=False)
    index = pd.read_csv(OUTPUT / "chemical_feature_index.csv", keep_default_na=False)
    review = pd.read_csv(OUTPUT / "manual_review.csv", keep_default_na=False)
    schema = json.loads((OUTPUT / "feature_schema.json").read_text(encoding="utf-8"))
    features = np.load(OUTPUT / "chemical_features.npz")
    similarity = np.load(OUTPUT / "tanimoto_similarity.npz")["tanimoto"]
    rows = pd.read_csv(OUTPUT / "tanimoto_row_index.csv", keep_default_na=False)
    cols = pd.read_csv(OUTPUT / "tanimoto_column_index.csv", keep_default_na=False)
    return mapping, index, review, schema, features, similarity, rows, cols


def test_all_metadata_entities_are_extracted_once(artifacts):
    mapping = artifacts[0]
    expected = set(pd.read_csv(TRAIN_VAL, usecols=["perturbation_no_concentration"])["perturbation_no_concentration"])
    expected |= set(pd.read_csv(TEST_META, usecols=["perturbation_no_concentration"])["perturbation_no_concentration"])
    assert mapping["raw_name"].is_unique
    assert set(mapping["raw_name"]) == expected
    assert len(mapping) == 57


def test_mapping_status_and_special_controls(artifacts):
    mapping = artifacts[0]
    assert set(mapping["mapping_status"]) <= set(chem.STATUSES)
    specials = mapping[mapping["raw_name"].isin(["Water", "DMSO", "Quality Control"])]
    assert len(specials) == 3
    assert specials["mapping_status"].eq("special_control").all()
    assert specials["structure_valid"].eq(False).all()


def test_confirmed_and_proxy_structures_parse_and_are_traceable(artifacts):
    mapping = artifacts[0]
    selected = mapping[mapping["mapping_status"].isin(["confirmed", "proxy"])]
    for row in selected.itertuples():
        assert row.raw_structure_smiles and Chem.MolFromSmiles(row.raw_structure_smiles) is not None
        assert row.raw_structure_inchikey
        assert row.pubchem_query_endpoint.startswith("https://pubchem.ncbi.nlm.nih.gov/")
        assert row.raw_response_sha256
        if row.structure_valid:
            assert row.parent_smiles and Chem.MolFromSmiles(row.parent_smiles) is not None
            assert row.parent_inchikey
        else:
            assert not row.parent_smiles and not row.parent_inchikey


def test_feature_shape_finiteness_binary_fp_and_fallback(artifacts):
    mapping, index, _, schema, data = artifacts[:5]
    assert data["features"].shape == (len(mapping), schema["chemical_feature_dim"])
    assert data["feature_valid_mask"].shape == data["features"].shape
    assert np.isfinite(data["features"]).all()
    assert np.isfinite(data["rdkit_descriptors_raw"]).all()
    assert set(np.unique(data["morgan_fp"])) <= {0.0, 1.0}
    assert index["chemical_id"].tolist() == mapping["chemical_id"].tolist()
    fallback = ~mapping["structure_valid"].astype(bool).to_numpy()
    assert np.all(data["features"][fallback] == 0)
    assert not data["feature_valid_mask"][fallback].any()
    assert data["feature_valid_mask"][~fallback].all()


def test_descriptor_order_entity_order_and_scaler_are_frozen(artifacts):
    mapping, _, _, schema, data = artifacts[:5]
    descriptor_hash = hashlib.sha256("\n".join(schema["descriptors"]["names"]).encode()).hexdigest()
    entity_hash = hashlib.sha256("\n".join(mapping["chemical_id"]).encode()).hexdigest()
    assert descriptor_hash == schema["descriptor_order_sha256"]
    assert entity_hash == schema["entity_order_sha256"]
    expected_fit = (
        mapping["seen_in_train"].astype(bool)
        & mapping["mapping_status"].eq("confirmed")
        & mapping["structure_valid"].astype(bool)
        & mapping["special_control_type"].eq("")
    ).to_numpy()
    assert np.array_equal(data["scaler_fit_mask"], expected_fit)
    assert mapping.loc[expected_fit, "raw_name"].tolist() == schema["scaler_fit_chemical_names"]
    assert not mapping.loc[expected_fit, "raw_name"].isin(["Water", "DMSO", "Quality Control"]).any()


def test_same_full_inchikey_never_maps_to_conflicting_parent(artifacts):
    mapping = artifacts[0]
    selected = mapping[mapping["raw_structure_inchikey"].ne("")]
    counts = selected.groupby("raw_structure_inchikey")["parent_inchikey"].nunique()
    assert (counts <= 1).all()


def test_manual_review_is_complete_and_final(artifacts):
    mapping, _, review = artifacts[:3]
    assert set(review["raw_name"]) == set(mapping.loc[~mapping["mapping_status"].eq("special_control"), "raw_name"])
    required = {"raw_name", "normalized_name", "candidate_cids", "selected_cid", "review_status", "mapping_status", "decision_reason", "structure_issue", "evidence_url", "review_date"}
    assert required <= set(review.columns)
    assert review["review_status"].eq("reviewed").all()
    assert review["decision_reason"].ne("").all()
    expected = mapping.set_index("raw_name")["mapping_status"]
    assert all(expected[row.raw_name] == row.mapping_status for row in review.itertuples())


def test_tanimoto_range_self_similarity_and_order(artifacts):
    mapping, _, _, _, _, similarity, rows, cols = artifacts
    assert similarity.shape == (len(rows), len(cols))
    assert np.isfinite(similarity).all()
    assert ((similarity >= 0) & (similarity <= 1)).all()
    normal = mapping[
        mapping["mapping_status"].isin(["confirmed", "proxy"])
        & mapping["structure_valid"].astype(bool)
        & mapping["special_control_type"].eq("")
    ]
    assert rows["chemical_id"].tolist() == normal[normal["seen_in_train"].astype(bool)]["chemical_id"].tolist()
    assert cols["chemical_id"].tolist() == normal["chemical_id"].tolist()
    col_pos = {name: i for i, name in enumerate(cols["chemical_id"])}
    for r, name in enumerate(rows["chemical_id"]):
        assert similarity[r, col_pos[name]] == pytest.approx(1.0)
    forbidden = set(mapping.loc[~mapping["structure_valid"].astype(bool), "chemical_id"])
    assert forbidden.isdisjoint(rows["chemical_id"])
    assert forbidden.isdisjoint(cols["chemical_id"])


def test_correct_shuffle_zero_interface_is_deterministic(artifacts):
    mapping, _, _, _, feature_data = artifacts[:5]
    x = feature_data["features"]
    correct, identity = chem.apply_feature_variant(x, "correct")
    zero, _ = chem.apply_feature_variant(x, "zero")
    kwargs = {
        "mapping_status": mapping["mapping_status"],
        "structure_valid": mapping["structure_valid"],
        "special_control_type": mapping["special_control_type"],
    }
    shuffled1, order1 = chem.apply_feature_variant(x, "shuffle", seed=17, **kwargs)
    shuffled2, order2 = chem.apply_feature_variant(x, "shuffle", seed=17, **kwargs)
    assert np.array_equal(correct, x) and np.array_equal(identity, np.arange(len(x)))
    assert np.all(zero == 0)
    assert np.array_equal(shuffled1, shuffled2) and np.array_equal(order1, order2)
    assert sorted(order1.tolist()) == list(range(len(x)))
    eligible = (
        mapping["mapping_status"].isin(["confirmed", "proxy"])
        & mapping["structure_valid"].astype(bool)
        & mapping["special_control_type"].eq("")
    ).to_numpy()
    ineligible = ~eligible
    eligible_idx = np.flatnonzero(eligible)
    assert np.array_equal(order1[ineligible], np.flatnonzero(ineligible))
    assert np.array_equal(shuffled1[ineligible], x[ineligible])
    assert set(order1[eligible].tolist()) == set(eligible_idx.tolist())
    assert np.array_equal(shuffled1[eligible], x[order1[eligible]])
    assert not np.any(np.all(shuffled1[eligible] == 0, axis=1))
    assert np.all(shuffled1[ineligible] == 0)
    assert sorted(row.tobytes() for row in shuffled1[eligible]) == sorted(row.tobytes() for row in x[eligible])


def test_high_risk_identity_and_feature_applicability(artifacts):
    mapping = artifacts[0].set_index("raw_name")
    assert mapping.loc["U-73122", "mapping_status"] == "confirmed"
    assert mapping.loc["U-73122", "pubchem_cid"] == "104794"
    assert mapping.loc["U-73122", "cas_number"] == "112648-68-7"
    assert mapping.loc["U-73122", "inchikey"] == "LUFAORPFSVMJIW-ZRJUGLEFSA-N"
    assert bool(mapping.loc["U-73122", "structure_valid"])
    for name, cid, smiles in [("Cisplatin", "5460033", "N.N.Cl[Pt]Cl"), ("NaCl", "5234", "[Na+].[Cl-]")]:
        assert mapping.loc[name, "mapping_status"] == "confirmed"
        assert mapping.loc[name, "pubchem_cid"] == cid
        assert mapping.loc[name, "raw_structure_smiles"] == smiles
        assert not bool(mapping.loc[name, "structure_valid"])
        assert "身份已确认，但当前 Morgan/RDKit 特征体系不适用" in mapping.loc[name, "structure_issue"]
    assert mapping.loc["Oligomycin", "mapping_status"] == "proxy"
    assert mapping.loc["Oligomycin", "mapping_confidence"] == "medium"
    assert mapping.loc["Tunicamycin", "mapping_status"] == "unresolved"


def test_descriptor_extreme_diagnostics_are_report_only(artifacts):
    schema = artifacts[3]
    diagnostics = schema["descriptor_extreme_value_diagnostics"]
    quantiles = diagnostics["absolute_value_quantiles"]
    assert float(quantiles["0.5"]) <= float(quantiles["0.9"]) <= float(quantiles["0.99"]) <= float(quantiles["1.0"])
    assert diagnostics["count_abs_gt_5"] >= diagnostics["count_abs_gt_10"] >= 0
    assert diagnostics["top_extreme_cells"]
    assert "recommendation_only" in diagnostics
    assert schema["descriptors"]["standardization"] == "z-score; ddof=0; train confirmed ordinary chemicals only"


def test_manifest_file_and_cache_hashes_are_current():
    manifest = json.loads((OUTPUT / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"]
    for source in manifest["sources"]:
        assert source.get("url") and source.get("version") and source.get("license")
    for record in manifest["input_files"]:
        path = Path(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    cache_records = manifest["requests"]
    cache_names = {path.name for path in (OUTPUT / "pubchem_cache").glob("*.json")}
    assert {record["cache_file"] for record in cache_records} == cache_names
    for record in cache_records:
        path = OUTPUT / "pubchem_cache" / record["cache_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["cache_file_sha256"]
        envelope = json.loads(path.read_text(encoding="utf-8"))
        assert envelope["response_body_sha256"] == record["response_body_sha256"]


def test_cli_has_no_proteome_interface_or_search():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, capture_output=True, text=True)
    assert "proteome" not in result.stdout.casefold()
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "rglob(" not in source and "glob(" not in source
    assert "proteome" not in source


def test_offline_rebuild_matches_checked_outputs(tmp_path, artifacts):
    target = tmp_path / "chemistry"
    target.mkdir()
    shutil.copytree(OUTPUT / "pubchem_cache", target / "pubchem_cache")
    override_copy = target / "manual_overrides.csv"
    shutil.copy2(OVERRIDES, override_copy)
    command = [
        sys.executable, str(SCRIPT), "--metadata-train-val", str(TRAIN_VAL),
        "--metadata-test", str(TEST_META), "--output-dir", str(target),
        "--overrides", str(override_copy), "--offline",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    for name in ["compound_mapping.csv", "manual_review.csv", "feature_schema.json", "chemical_feature_index.csv", "tanimoto_row_index.csv", "tanimoto_column_index.csv"]:
        assert (target / name).read_bytes() == (OUTPUT / name).read_bytes()
    rebuilt_features = np.load(target / "chemical_features.npz")
    original_features = artifacts[4]
    assert set(rebuilt_features.files) == set(original_features.files)
    for name in rebuilt_features.files:
        assert np.array_equal(rebuilt_features[name], original_features[name])
    assert np.array_equal(np.load(target / "tanimoto_similarity.npz")["tanimoto"], artifacts[5])
