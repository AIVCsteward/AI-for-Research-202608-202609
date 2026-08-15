import inspect
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from baseline.baseline.stage_s2d_gate_review_v2 import (
    D0_2, D2_2, EXPECTED_ORIGINAL_SHA256, EXCLUDED_FROM_REPLICATION_POOL,
    METRIC_FILES, S1C_2, S2C2, S2C_MEAN, SEED2_POOL,
    original_hash_audit, recompute_seed2_replication_ranks, sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/model_v2_stage_s2d"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_legacy_conclusion_is_retained_separately():
    review = _json(OUT / "gate_semantics_review.json")
    legacy = review["legacy_global_rank"]
    assert legacy["legacy_gate_1_global_rank_first"] is False
    assert review["literal_legacy_conclusion"]["legacy_gate_1"] is False
    assert review["literal_legacy_conclusion"]["eligible_under_legacy_literal_all-gates_rule"] is False
    assert "seed1 + seed2 + two-seed mean" in legacy["pool_semantics"]
    assert {row["rank"] for row in legacy["s2c_seed2_rows"]} == {3}


def test_seed2_replication_pool_has_only_seed_matched_candidates():
    review = _json(OUT / "gate_semantics_review.json")
    pool = review["seed2_replication_pool"]
    assert tuple(pool["included_models"]) == SEED2_POOL
    assert set(pool["included_models"]) == {S2C2, D0_2, D2_2, S1C_2}
    assert set(pool["excluded_models"]) == set(EXCLUDED_FROM_REPLICATION_POOL)
    assert S2C_MEAN not in pool["included_models"]
    assert not any("seed1" in model for model in pool["included_models"])
    assert pool["underlying_metrics_recomputed"] is False


def test_corrected_ranks_are_recomputed_from_unchanged_metric_values():
    review = _json(OUT / "gate_semantics_review.json")
    tables = {key: pd.read_csv(OUT / filename) for key, filename in METRIC_FILES.items()}
    _, actual = recompute_seed2_replication_ranks(tables)
    delivered = pd.DataFrame(review["seed2_replication_pool"]["ranking"])
    delivered = delivered.loc[:, actual.columns]
    assert_frame_equal(actual.reset_index(drop=True), delivered.reset_index(drop=True), check_exact=True)
    s2c = actual.loc[actual.model.eq(S2C2)]
    assert len(s2c) == 3 and s2c["rank"].eq(1).all()
    assert review["corrected_gate_1"]["passed"] is True
    assert review["corrected_gate_1"]["number_of_first_place_proxies"] == 3
    for row in review["seed2_replication_pool"]["proxy_differences"]:
        assert row["absolute_difference_vs_d2_seed2"] > 0
        assert row["absolute_difference_vs_s1_c_seed2"] > 0
        assert row["absolute_difference_vs_d0_seed2"] > 0


def test_original_gates_two_to_ten_and_final_eligibility_are_preserved():
    review = _json(OUT / "gate_semantics_review.json")
    original = _json(OUT / "two_seed_ensemble_metrics.json")
    expected = {key: value for key, value in original["gates"].items() if not key.startswith("gate_1_")}
    assert review["original_gates_2_to_10"] == expected
    assert len(expected) == 9 and all(expected.values())
    assert review["two_seed_mean_global_first_in_all_three_proxies"] is True
    corrected = review["corrected_replication_conclusion"]
    assert corrected["eligible_for_final_presubmission_audit"] is True
    assert corrected["eligibility_basis"] == "corrected_seed_matched_replication_gate"
    assert corrected["protocol_clarification"] is True
    assert corrected["threshold_changed"] is False
    assert corrected["mixture_weight_searched"] is False
    assert corrected["model_or_prediction_changed"] is False


def test_original_s2d_files_and_route_and_metric_hashes_are_unchanged():
    review = _json(OUT / "gate_semantics_review.json")
    audit = original_hash_audit(ROOT)
    assert audit["all_original_s2d_files_unchanged"] is True
    assert audit == review["original_s2d_hash_audit"]
    for relative, expected in EXPECTED_ORIGINAL_SHA256.items():
        assert sha256_file(ROOT / relative) == expected
    checks = review["consistency_checks"]
    assert checks["routing_hash_unchanged"] is True
    assert checks["two_seed_mean_metrics_hash_unchanged"] is True
    assert checks["s2c_seed2_rmse_better_than_d2_seed2"] is True
    assert checks["raw_fc_pcc_within_original_tolerance"] is True
    assert checks["raw_prediction_tensor_hash_status"] == "NOT_MATERIALIZED_BY_ORIGINAL_S2D"
    assert checks["raw_prediction_tensor_hash"] is None


def test_review_is_read_only_and_never_accesses_held_out_protein_or_predictions():
    review = _json(OUT / "gate_semantics_review.json")
    assert review["review_only"] is True
    assert review["training_performed"] is False
    assert review["inference_performed"] is False
    assert review["predictions_modified"] is False
    assert review["checkpoints_modified"] is False
    assert review["routing_modified"] is False
    assert review["test_proteome_opened"] is False
    assert review["test_prediction_generated"] is False
    source = inspect.getsource(__import__(
        "baseline.baseline.stage_s2d_gate_review_v2", fromlist=["*"]
    )).lower()
    assert "proteome_raw_test" not in source
    assert "torch" not in source
    assert "load_checkpoint" not in source
    assert "load_label_frames" not in source
    assert (ROOT / "reports/MODEL_V2_STAGE_S2D_GATE_REVIEW.md").is_file()
