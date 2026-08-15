import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.baseline.second_seed_ensemble_v2 import (
    SEED1, SEED2, S2C_MEAN, arithmetic_seed_mean, route_identity_records,
)
from scripts.audit_competition_score_alignment import stable_json_sha256


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/model_v2_stage_s2d"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_two_seed_mean_is_exact_unweighted_arithmetic_mean():
    first = np.array([[1.0, -2.0], [3.0, 5.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0], [-1.0, 7.0]], dtype=np.float32)
    expected = np.array([[2.0, 1.0], [1.0, 6.0]], dtype=np.float32)
    assert np.array_equal(arithmetic_seed_mean(first, second), expected)
    assert "weight" not in inspect.signature(arithmetic_seed_mean).parameters


def test_seed2_stage_a_was_retrained_from_controls_with_declared_semantics():
    hierarchical = _json(OUT / "stage_a_hierarchical_seed_20260815/stage_a_summary.json")
    flat = _json(OUT / "stage_a_flat_seed_20260815/stage_a_summary.json")
    assert hierarchical["seed"] == SEED2
    assert hierarchical["train_control_count"] == 751
    assert hierarchical["fixed_epochs"] == 30
    assert hierarchical["validation_control_count_for_selection"] == 0
    assert hierarchical["checkpoint_selection"] == "fixed_epochs_without_validation_labels"
    assert hierarchical["test_proteome_opened"] is False
    assert flat["seed"] == SEED2
    assert flat["train_control_count"] == 751
    assert flat["validation_control_count"] == 205
    assert flat["test_proteome_opened"] is False


def test_seed2_experts_share_holdout_and_hierarchical_initialization_and_freeze():
    directories = {
        "d0": OUT / "formal_d0_hierarchical_none_huber_seed_20260815",
        "d2": OUT / "formal_d2_hierarchical_morgan_fc_seed_20260815",
        "s1c": OUT / "formal_s1c_flat_morgan_fc_seed_20260815",
    }
    summaries = {key: _json(path / "training_summary.json") for key, path in directories.items()}
    expected_holdout = "7cfd2fc2c8877425c98f845ffc9275cd023cee7da31aa677437aaa65ffd11734"
    assert {value["train_internal_batch_holdout"]["holdout_sample_ids_sha256"] for value in summaries.values()} == {expected_holdout}
    assert {value["stage_b_training_ids_sha256"] for value in summaries.values()} == {"b7dc844ac46b110eef05b56b5be8002d3a8bd566edf4677d9f2409db6303ee89"}
    assert summaries["d0"]["stage_b_initial_chemical_response_sha256"] == summaries["d2"]["stage_b_initial_chemical_response_sha256"]
    assert summaries["d0"]["stage_a"]["fixed_checkpoint"]["sha256"] == summaries["d2"]["stage_a"]["fixed_checkpoint"]["sha256"]
    for key in ("d0", "d2"):
        audit = summaries[key]["stage_b_freeze_audit"]
        assert audit["status"] == "PASS"
        assert all(audit["parameter_hashes_exactly_equal"].values())
        assert all(audit["probe_outputs_bitwise_equal"].values())
        assert audit["optimizer"]["only_chemical_encoder_and_response_branch"] is True
    assert summaries["s1c"]["stage_a"]["stop_reason"] == "loaded_fixed_stage_a_checkpoint_no_retraining"
    assert all(value["test_proteome_opened"] is False for value in summaries.values())


def test_seed2_configs_only_change_authorized_seed_owned_fields():
    pairs = [
        ("model_v2_stage_s2b_d0_hierarchical_none_huber.yaml", "model_v2_stage_s2d_d0_hierarchical_none_huber_seed_20260815.yaml"),
        ("model_v2_stage_s2b_d2_hierarchical_morgan_fc.yaml", "model_v2_stage_s2d_d2_hierarchical_morgan_fc_seed_20260815.yaml"),
        ("model_v2_stage_s1_c_anchor_morgan_fc.yaml", "model_v2_stage_s2d_s1c_flat_morgan_fc_seed_20260815.yaml"),
    ]
    allowed = {
        "task_stage", "experiment_name", "training.seed", "training.fixed_stage_a_checkpoint.path",
        "training.fixed_stage_a_checkpoint.sha256", "training.stage_a.name",
        "training.train_internal_batch_holdout.selection_seed",
    }
    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, item in value.items():
                result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        else:
            result[prefix] = value
        return result
    for old_name, new_name in pairs:
        old = flatten(_json(ROOT / "baseline/configs" / old_name))
        new = flatten(_json(ROOT / "baseline/configs" / new_name))
        changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
        assert changed <= allowed
        assert new["training.seed"] == SEED2
        assert new["training.train_internal_batch_holdout.selection_seed"] == SEED1


def test_routing_is_rowwise_identical_and_test_counts_are_unchanged():
    audit = _json(OUT / "routing_identity_audit.json")
    assert audit["rowwise_identical"] is True
    assert audit["seed1_manifest_sha256"] == audit["seed2_manifest_sha256"]
    assert audit["seed2_manifest_sha256"] == "f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4"
    assert audit["test_metadata_counts_identical"] is True
    assert audit["test_metadata"]["expert_counts"] == {
        "S2B D0 hierarchical none Huber": 2997,
        "S1 C anchor Morgan Huber plus experimental FC": 1322,
        "S2B D2 hierarchical Morgan experimental FC": 135,
    }
    first = pd.read_csv(ROOT / "reports/model_v2_stage_s2c/validation_routing.csv")
    second = pd.read_csv(OUT / "validation_routing_seed2.csv")
    assert route_identity_records(first) == route_identity_records(second)
    assert stable_json_sha256(route_identity_records(second)) == audit["seed2_manifest_sha256"]


def test_delivered_metrics_are_finite_and_gate_outcome_is_consistent():
    comparison = _json(OUT / "second_seed_comparison.json")
    ensemble = _json(OUT / "two_seed_ensemble_metrics.json")
    assert comparison["test_proteome_opened"] is False
    assert comparison["test_prediction_generated"] is False
    assert ensemble["model"] == S2C_MEAN and ensemble["weights"] == [0.5, 0.5]
    assert ensemble["eligible_for_final_submission_audit"] == all(ensemble["gates"].values())
    assert ensemble["gates"]["gate_1_seed2_s2c_first_in_at_least_two_planning_proxies"] is False
    assert ensemble["gates"]["gate_6_two_seed_mean_first_in_all_three_planning_proxies"] is True
    for filename in (
        "second_seed_absolute_metrics.csv", "second_seed_fc_metrics.csv",
        "second_seed_high_metrics.csv", "seed_prediction_stability.csv",
        "seed_stability_per_entity.csv",
    ):
        frame = pd.read_csv(OUT / filename)
        if "model" in frame:
            frame = frame.loc[frame.model.ne("Matched Control")]
        numeric = frame.select_dtypes(include=["number"]).dropna(axis=1, how="all")
        assert np.isfinite(numeric.to_numpy()).all()
    source = (ROOT / "baseline/baseline/second_seed_ensemble_v2.py").read_text(encoding="utf-8").lower()
    assert "proteome_raw_test" not in source
    assert "tanimoto" not in source
    assert "chemical_name ==" not in source
    assert (ROOT / "reports/MODEL_V2_STAGE_S2D_SECOND_SEED_REPORT.md").is_file()
