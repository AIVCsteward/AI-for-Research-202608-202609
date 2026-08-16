from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "external_data" / "genome"
RAW = OUTPUT / "raw_cache"
SCRIPT = ROOT / "scripts" / "build_genome_features.py"
TRAIN_VAL = ROOT / "WAYB_WAYC" / "WAYB_WAYC_metadata_train_val(1).csv"
TEST_META = ROOT / "WAYB_WAYC" / "WAYB_WAYC_metadata_test(1).csv"
XL_DIR = ROOT / "tmp" / "python_pkgs"

spec = importlib.util.spec_from_file_location("build_genome_features", SCRIPT)
genome = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = genome
spec.loader.exec_module(genome)


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_artifacts():
    mapping = read_csv(OUTPUT / "strain_mapping.csv")
    index = read_csv(OUTPUT / "strain_feature_index.csv")
    schema = json.loads((OUTPUT / "genome_feature_schema.json").read_text(encoding="utf-8"))
    features = np.load(OUTPUT / "strain_features.npz")
    return mapping, index, schema, features


def test_01_metadata_inventory_matches_real_files():
    rows = genome.extract_strains([TRAIN_VAL, TEST_META])
    expected = {
        "BAH": (2235, True, True, True),
        "BAI": (2248, False, True, True),
        "CEK": (2254, True, True, True),
        "CGD": (2207, True, True, True),
        "CRD": (2231, False, False, True),
        "DHY210": (2237, True, True, True),
    }
    assert {row["strain_id"] for row in rows} == set(expected)
    for row in rows:
        assert (row["metadata_row_count"], row["seen_in_train"], row["seen_in_validation"], row["seen_in_test"]) == expected[row["strain_id"]]


def test_02_cli_has_no_forbidden_data_interface_or_discovery():
    help_text = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, capture_output=True, text=True).stdout.casefold()
    assert "protein" not in help_text
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    forbidden = "proto" + "me"
    assert forbidden not in source
    assert "rglob(" not in source and ".glob(" not in source and "os.walk(" not in source


def test_03_six_strains_have_one_valid_frozen_status():
    mapping, _, _, _ = load_artifacts()
    assert [row["strain_id"] for row in mapping] == list(genome.STRAIN_ORDER)
    assert len({row["strain_id"] for row in mapping}) == 6
    assert all(row["mapping_type"] in genome.MAPPING_TYPES for row in mapping)
    assert all(row["mapping_confidence"] in genome.CONFIDENCES for row in mapping)


def test_04_supported_mappings_have_multiple_evidence_sources():
    mapping, _, _, _ = load_artifacts()
    selected = [row for row in mapping if row["mapping_type"] in {"exact", "supported"}]
    assert len(selected) == 5
    for row in selected:
        assert row["evidence_summary"] and row["source_files"] and row["source_sha256"]
        assert len(row["evidence_urls"].split("|")) >= 2
        assert row["candidates"] and row["rejected_candidates"]


def test_05_bai_and_crd_have_dedicated_audit_sections():
    evidence = (OUTPUT / "mapping_evidence.md").read_text(encoding="utf-8")
    assert "## BAI 单独审计" in evidence
    assert "## CRD 单独审计" in evidence
    mapping = {row["strain_id"]: row for row in load_artifacts()[0]}
    assert mapping["BAI"]["external_isolate_id"] == "BJ6"
    assert mapping["CRD"]["external_isolate_id"] == "FIMA_3"


def test_06_dhy210_is_always_low_confidence_s288c_proxy():
    mapping = {row["strain_id"]: row for row in load_artifacts()[0]}
    row = mapping["DHY210"]
    assert row["external_isolate_id"] == "S288C"
    assert row["external_assembly_id"] == "GCF_000146045.2"
    assert row["external_matrix_id"] == ""
    assert row["mapping_type"] == "proxy"
    assert row["mapping_confidence"] in {"low", "medium"}
    assert row["proxy_flag"] == "True"
    assert "lost" in row["evidence_summary"].casefold()


def test_07_unresolved_fallback_is_explicit_and_nonfabricated():
    _, _, schema, _ = load_artifacts()
    assert schema["fallback_contract"]["unresolved"].startswith("all-zero")
    synthetic = np.asarray([[1.0, 2.0], [0.0, 0.0]])
    zero, order = genome.apply_feature_variant(synthetic, "zero")
    assert np.array_equal(zero, np.zeros_like(synthetic))
    assert np.array_equal(order, np.arange(2))


def test_08_all_cached_external_resource_hashes_verify():
    manifest = json.loads((OUTPUT / "raw_resource_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resources"]
    for record in manifest["resources"]:
        path = Path(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        if record["upstream_md5"]:
            assert hashlib.md5(path.read_bytes()).hexdigest() == record["upstream_md5"]  # nosec - upstream verification
        assert record["url"] and record["version"] and record["download_date"] and record["license"]
    for excluded in manifest["excluded_or_unavailable"]:
        assert excluded["status"] in {"not_downloaded", "not_cached"}
        assert excluded["reason"]


def test_09_coordinate_and_feature_orders_are_frozen():
    _, _, schema, data = load_artifacts()
    assert genome.order_hash(schema["strain_id_order"]) == schema["strain_id_order_sha256"]
    names = schema["genome_features"]["feature_names"]
    assert genome.order_hash(names) == schema["genome_features"]["feature_order_sha256"]
    assert data["genome_feature_names"].tolist() == names
    assert data["raw_feature_names"].tolist() == schema["raw_features"]["feature_names"]
    assert schema["supplemental_orders"]["pangenome_orf_count"] == 7796
    contract = schema["coordinate_contract"]
    for key in ["reference_assembly", "chromosome_naming", "vcf_reference_allele_direction", "multiallelic_handling", "missing_genotype", "heterozygous_encoding", "callable_region", "gene_region_definition", "multiple_transcripts"]:
        assert contract[key]


def test_10_feature_shapes_masks_and_values_are_finite():
    mapping, index, schema, data = load_artifacts()
    expected = (6, schema["genome_dim"])
    assert data["genome_features"].shape == expected
    assert data["feature_valid_mask"].shape == expected
    assert data["raw_feature_valid_mask"].shape == (6, schema["raw_feature_dim"])
    assert data["qc_features"].shape == (6, schema["qc_feature_dim"])
    assert np.isfinite(data["genome_features"]).all()
    assert np.isfinite(data["raw_features"]).all()
    assert [row["strain_id"] for row in index] == [row["strain_id"] for row in mapping] == data["strain_ids"].tolist()


def test_11_fit_mask_is_exactly_eligible_training_strains():
    mapping, _, schema, data = load_artifacts()
    expected = np.asarray([
        row["seen_in_train"] == "True" and row["mapping_type"] in {"exact", "supported"} and row["feature_valid"] == "True"
        for row in mapping
    ])
    assert np.array_equal(data["eligible_train_fit_mask"], expected)
    assert data["strain_ids"][expected].tolist() == ["BAH", "CEK", "CGD"]
    assert schema["fit_contract"]["fit_strains"] == ["BAH", "CEK", "CGD"]
    assert not data["eligible_train_fit_mask"][data["proxy_flag"]].any()


def test_12_unseen_strains_receive_only_frozen_transforms():
    mapping, _, _, data = load_artifacts()
    fit = data["eligible_train_fit_mask"]
    names = data["raw_feature_names"].tolist()
    positions = [names.index(name) for name in data["genome_feature_names"].tolist()]
    raw = data["raw_features"][:, positions].astype(np.float64)
    valid = data["feature_valid_mask"]
    median = data["imputation_median"]
    mean = data["scaler_mean"]
    scale = data["scaler_scale"]
    variance = data["variance_mask"]
    rebuilt = (np.where(valid, raw, median) - mean) / scale
    rebuilt[:, ~variance] = 0
    rebuilt[~valid] = 0
    assert np.allclose(rebuilt, data["genome_features"], atol=5e-5)
    assert [mapping[i]["strain_id"] for i in np.flatnonzero(~fit)] == ["BAI", "CRD", "DHY210"]


def test_13_pca_dimensions_do_not_exceed_centered_fit_rank():
    _, _, schema, data = load_artifacts()
    fit_matrix = data["genome_features"][data["eligible_train_fit_mask"]][:, data["variance_mask"]]
    centered = fit_matrix - fit_matrix.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)
    assert schema["fit_contract"]["matrix_rank"] == rank
    assert data["pca_features"].shape[1] == schema["fit_contract"]["pca_n_components"] <= rank
    assert np.isclose(data["pca_explained_variance_ratio"].sum(), 1.0)


def test_14_correct_shuffle_zero_are_reproducible_and_distribution_safe():
    mapping, _, _, data = load_artifacts()
    values = data["genome_features"]
    types = [row["mapping_type"] for row in mapping]
    validity = [row["feature_valid"] == "True" for row in mapping]
    correct, identity = genome.apply_feature_variant(values, "correct")
    zero, zero_order = genome.apply_feature_variant(values, "zero")
    shuffled_a, order_a = genome.apply_feature_variant(values, "shuffle", mapping_type=types, feature_valid=validity, seed=17)
    shuffled_b, order_b = genome.apply_feature_variant(values, "shuffle", mapping_type=types, feature_valid=validity, seed=17)
    eligible = np.asarray([value in {"exact", "supported"} and valid for value, valid in zip(types, validity)])
    assert np.array_equal(correct, values) and np.array_equal(identity, np.arange(6))
    assert np.array_equal(zero, np.zeros_like(values)) and np.array_equal(zero_order, identity)
    assert np.array_equal(shuffled_a, shuffled_b) and np.array_equal(order_a, order_b)
    assert sorted(order_a[eligible].tolist()) == sorted(np.flatnonzero(eligible).tolist())
    assert np.array_equal(order_a[~eligible], np.flatnonzero(~eligible))
    assert sorted(row.tobytes() for row in shuffled_a[eligible]) == sorted(row.tobytes() for row in values[eligible])


def test_15_shuffle_never_changes_proxy_or_invalid_entities():
    mapping, _, _, data = load_artifacts()
    types = [row["mapping_type"] for row in mapping]
    validity = [row["feature_valid"] == "True" for row in mapping]
    shuffled, order = genome.apply_feature_variant(data["genome_features"], "shuffle", mapping_type=types, feature_valid=validity, seed=29)
    frozen = np.asarray([value in {"proxy", "unresolved"} or not valid for value, valid in zip(types, validity)])
    assert frozen.any()
    assert np.array_equal(order[frozen], np.flatnonzero(frozen))
    assert np.array_equal(shuffled[frozen], data["genome_features"][frozen])


def test_16_offline_rebuild_is_byte_and_numeric_identical(tmp_path):
    target = tmp_path / "genome"
    target.mkdir()
    env = os.environ.copy()
    if XL_DIR.is_dir():
        env["PYTHONPATH"] = str(XL_DIR)
    command = [
        sys.executable, str(SCRIPT), "--metadata-train-val", str(TRAIN_VAL),
        "--metadata-test", str(TEST_META), "--output-dir", str(target),
        "--raw-cache-dir", str(RAW), "--offline",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    for name in ["strain_mapping.csv", "strain_feature_index.csv", "genome_feature_schema.json", "raw_resource_manifest.json", "source_manifest.json", "GO_OR_PATHWAY_MAPPING.csv"]:
        assert (target / name).read_bytes() == (OUTPUT / name).read_bytes()
    rebuilt = np.load(target / "strain_features.npz")
    original = np.load(OUTPUT / "strain_features.npz")
    assert set(rebuilt.files) == set(original.files)
    for name in rebuilt.files:
        assert np.array_equal(rebuilt[name], original[name])


def test_17_invalid_denominators_invalidate_all_dependent_dhy210_features():
    _, _, schema, data = load_artifacts()
    row = data["strain_ids"].tolist().index("DHY210")
    names = schema["genome_features"]["feature_names"]
    valid = data["feature_valid_mask"][row]
    dependency_groups = {
        "pangenome": [name for name in names if name.startswith("pangenome_")],
        "frameshift": [name for name in names if name.startswith("frameshift_")],
        "gwas": [name for name in names if name.startswith("gwas_")],
    }
    for dependent_names in dependency_groups.values():
        assert dependent_names
        assert not valid[[names.index(name) for name in dependent_names]].any()
    # The ploidy source was not frozen, so it too must be invalid.
    assert not valid[names.index("ploidy_n")]
    support = schema["supplemental_orders"]
    assert support["s288c_invalid_dependency_groups"]
    raw_names = data["raw_feature_names"].tolist()
    raw_valid = data["raw_feature_valid_mask"][row]
    valid_raw_names = {name for name, is_valid in zip(raw_names, raw_valid) if is_valid}
    assert valid_raw_names == set(support["s288c_valid_feature_sources"])
    assert all(support["s288c_valid_feature_sources"].values())


def test_18_every_invalid_final_feature_is_strict_zero_and_dhy_has_no_imputation_outlier():
    _, _, _, data = load_artifacts()
    invalid = ~data["feature_valid_mask"]
    assert invalid.any()
    assert np.count_nonzero(data["genome_features"][invalid]) == 0
    dhy = data["strain_ids"].tolist().index("DHY210")
    valid_values = np.abs(data["genome_features"][data["feature_valid_mask"]])
    train_valid_values = np.abs(data["genome_features"][data["eligible_train_fit_mask"]])
    assert np.max(np.abs(data["genome_features"][dhy])) <= max(1.0, float(train_valid_values.max()))
    assert float(valid_values.max()) < 10.0


def test_19_external_isolate_matrix_and_assembly_ids_are_not_confused():
    mapping, index, _, data = load_artifacts()
    expected = {
        "BAH": ("SX3", "BAH"), "BAI": ("BJ6", "BAI"),
        "CEK": ("JCM_2985-4B", "CEK"), "CGD": ("UCD_09-448", "CGD"),
        "CRD": ("FIMA_3", "CRD"),
    }
    for row in mapping:
        strain = row["strain_id"]
        if strain in expected:
            assert (row["external_isolate_id"], row["external_matrix_id"]) == expected[strain]
            assert row["external_isolate_id"] != row["external_matrix_id"]
            assert row["external_assembly_id"] == ""
    assert [row["external_isolate_id"] for row in index] == data["external_isolate_ids"].tolist()
    assert [row["external_matrix_id"] for row in index] == data["external_matrix_ids"].tolist()
    assert [row["external_assembly_id"] for row in index] == data["external_assembly_ids"].tolist()


def test_20_technical_qc_is_excluded_from_main_features_and_fit_stays_train_only():
    mapping, _, schema, data = load_artifacts()
    main_names = set(schema["genome_features"]["feature_names"])
    qc_names = schema["technical_qc"]["feature_names"]
    assert qc_names == genome.TECHNICAL_QC_FEATURE_NAMES
    assert main_names.isdisjoint(qc_names)
    assert data["qc_feature_names"].tolist() == qc_names
    expected = np.asarray([
        row["seen_in_train"] == "True" and row["mapping_type"] in {"exact", "supported"} and row["feature_valid"] == "True"
        for row in mapping
    ])
    assert data["strain_ids"][expected].tolist() == ["BAH", "CEK", "CGD"]
    assert np.array_equal(data["eligible_train_fit_mask"], expected)
