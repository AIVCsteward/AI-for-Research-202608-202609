import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline.baseline.final_submission_v2 import (
    D0, D2, S1C, EXPECTED_TEST_COUNTS, TestProteomeFileAccessGuard,
    _test_route_records, audit_csv_contract, read_reference_header_only,
    verify_frozen_candidate,
)
from baseline.baseline.novelty_gated_ensemble_v2 import (
    derive_metadata_routing, fit_seen_sets_from_train_metadata,
)
from baseline.baseline.training_v2 import load_artifact_bundle, load_train_val_metadata
from scripts.audit_competition_score_alignment import stable_json_sha256


ROOT = Path(__file__).resolve().parents[2]


def test_explicit_test_proteome_guard_rejects_all_open_paths(tmp_path):
    forbidden = tmp_path / ("WAYB_WAYC_" + "proteome" + "_raw_test.csv")
    with TestProteomeFileAccessGuard(ROOT) as guard:
        with pytest.raises(PermissionError):
            open(forbidden, "rb")
        with pytest.raises(PermissionError):
            forbidden.open("rb")
    assert len(guard.denied_attempts) == 2


def test_frozen_candidate_hashes_match_s2d_and_s2dr():
    audit = verify_frozen_candidate(ROOT)
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["experts"]) == 6
    assert {item["seed"] for item in audit["experts"].values()} == {20260814, 20260815}


def test_train_only_seen_sets_reproduce_expected_test_routes_without_labels():
    meta = load_train_val_metadata(ROOT)
    seen = fit_seen_sets_from_train_metadata(meta)
    test = pd.read_csv(ROOT / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv").set_index("sample_ID", drop=False)
    route = derive_metadata_routing(test, seen)
    assert route.selected_expert.value_counts().to_dict() == EXPECTED_TEST_COUNTS
    assert len(_test_route_records(route)) == 4454
    assert stable_json_sha256(_test_route_records(route)) == stable_json_sha256(_test_route_records(route.copy()))
    source = (ROOT / "baseline/baseline/novelty_gated_ensemble_v2.py").read_text(encoding="utf-8")
    assert "chemical_name ==" not in source


def test_csv_writer_parser_contract_handles_comma_protein_names(tmp_path):
    proteins = ["ARG5,6", "DUR1,2"]
    ids = pd.DataFrame({"sample_ID": ["x", "y"]})
    values = np.array([[1.25, -2.5], [3.0, 4.125]], dtype=np.float32)
    frame = pd.DataFrame(values, columns=proteins)
    frame.insert(0, "sample_ID", ids.sample_ID)
    path = tmp_path / "candidate.csv"
    frame.to_csv(path, index=False)
    parsed = pd.read_csv(path)
    assert parsed.columns.tolist() == ["sample_ID", *proteins]
    assert np.array_equal(parsed.iloc[:, 1:].to_numpy(np.float32), values)


def test_legacy_prediction_is_header_only_format_reference():
    artifacts = load_artifact_bundle(ROOT)
    header = read_reference_header_only(ROOT / "baseline_models/prediction.csv")
    assert header == ["sample_ID", *artifacts.feature_contract.proteins]


def test_delivered_candidate_passes_all_hard_audits_if_materialized():
    submission = ROOT / "submissions/model_v2_s2c_two_seed"
    report = ROOT / "reports/model_v2_stage_s3"
    if not (submission / "prediction.csv").is_file():
        pytest.skip("Stage S3 candidate has not yet been materialized")
    audit = json.loads((submission / "submission_audit.json").read_text(encoding="utf-8"))
    contract = json.loads((report / "submission_contract_audit.json").read_text(encoding="utf-8"))
    determinism = json.loads((report / "determinism_audit.json").read_text(encoding="utf-8"))
    access = json.loads((report / "data_access_audit.json").read_text(encoding="utf-8"))
    assert audit["eligible_for_manual_submission"] is True
    assert contract["passed"] is True and all(contract["checks"].values())
    assert determinism["passed"] is True
    assert access["test_proteome_opened"] is False
    assert audit["platform_upload_performed"] is False
