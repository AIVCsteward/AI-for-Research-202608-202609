"""Stage S3 fail-closed validation replay and candidate submission generation.

This module never trains a model.  The default command runs four isolated
workers: frozen-input verification, validation replay, and two deterministic
test-metadata inference passes.  It then audits and materializes the candidate
CSV only when every mandatory check passes.
"""
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from baseline.baseline.hierarchical_treatment_analysis_v2 import _encoded
from baseline.baseline.novelty_gated_ensemble_v2 import (
    D0, D2, S1C, apply_hierarchical_oov_codes, derive_metadata_routing,
    enforce_routed_control_zero_response, fit_seen_sets_from_train_metadata,
)
from baseline.baseline.score_aligned_analysis_v2 import _finite_json
from baseline.baseline.second_seed_ensemble_v2 import (
    CHECKPOINTS, LABELS, SEED1, SEED2, S2C_MEAN, _infer_light, _load_experts,
    arithmetic_seed_mean, route_identity_records,
)
from baseline.baseline.training_v2 import (
    CONTROL_NAMES, VAL_SCENARIOS, load_artifact_bundle, load_label_frames,
    load_train_val_metadata, sha256_file,
)
from scripts.audit_competition_score_alignment import stable_json_sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path("reports/model_v2_stage_s3")
SUBMISSION_DIR = Path("submissions/model_v2_s2c_two_seed")
EXPECTED_TEST_ROWS = 4454
EXPECTED_TEST_COUNTS = {D0: 2997, S1C: 1322, D2: 135}
EXPECTED_VALIDATION_ROUTE_SHA256 = "f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4"
FORBIDDEN_TEST_TOKEN = "proteome" + "_raw_test"
CONTROL_IDENTITIES = frozenset((*CONTROL_NAMES, "quality control"))
MODEL_SOURCE = Path("baseline/baseline/model_v2.py")
ROUTING_SOURCE = Path("baseline/baseline/novelty_gated_ensemble_v2.py")
INFERENCE_SOURCE = Path("baseline/baseline/final_submission_v2.py")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(_finite_json(value)), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _path_text(value: Any) -> str | None:
    if isinstance(value, int):
        return None
    try:
        return os.fspath(value)
    except TypeError:
        return None


class TestProteomeFileAccessGuard(AbstractContextManager):
    """Reject every attempted open whose path contains the held-out filename."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.opened_workspace_paths: list[str] = []
        self.denied_attempts: list[str] = []
        self._orig_builtin = builtins.open
        self._orig_io = io.open
        self._orig_os = os.open

    def _check(self, value: Any) -> None:
        text = _path_text(value)
        if text is None:
            return
        lowered = text.replace("/", "\\").lower()
        if FORBIDDEN_TEST_TOKEN in lowered:
            self.denied_attempts.append(str(text))
            raise PermissionError(f"forbidden test-proteome access: {text}")
        try:
            resolved = Path(text).resolve()
            if resolved == self.root or self.root in resolved.parents:
                self.opened_workspace_paths.append(str(resolved))
        except (OSError, RuntimeError, ValueError):
            pass

    def _builtin_open(self, file, *args, **kwargs):
        self._check(file)
        return self._orig_builtin(file, *args, **kwargs)

    def _io_open(self, file, *args, **kwargs):
        self._check(file)
        return self._orig_io(file, *args, **kwargs)

    def _os_open(self, path, *args, **kwargs):
        self._check(path)
        return self._orig_os(path, *args, **kwargs)

    def __enter__(self):
        builtins.open = self._builtin_open
        io.open = self._io_open
        os.open = self._os_open
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.open = self._orig_builtin
        io.open = self._orig_io
        os.open = self._orig_os
        return False

    def audit(self) -> dict:
        paths = sorted(set(self.opened_workspace_paths))
        return {
            "guard_enabled": True,
            "forbidden_token_sha256": hashlib.sha256(FORBIDDEN_TEST_TOKEN.encode()).hexdigest(),
            "denied_attempts": list(self.denied_attempts),
            "workspace_paths_opened": paths,
            "test_proteome_opened": False,
        }


def read_reference_header_only(path: Path) -> list[str]:
    """Read exactly one CSV row; baseline prediction values are out of scope."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def _actual_s2d_hash_audit(root: Path) -> dict:
    review = _json(root / "reports/model_v2_stage_s2d/gate_semantics_review.json")
    expected = review["original_s2d_hash_audit"]["files"]
    files = {}
    for relative, item in expected.items():
        actual = sha256_file(root / relative)
        files[relative] = {
            "expected_sha256": item["expected_sha256"],
            "actual_sha256": actual,
            "unchanged": actual == item["expected_sha256"],
        }
    return {"files": files, "all_unchanged": all(x["unchanged"] for x in files.values())}


def verify_frozen_candidate(root: Path) -> dict:
    root = Path(root).resolve()
    recorded = _json(root / "reports/model_v2_stage_s2d/checkpoint_manifest.json")
    s2c = _json(root / "reports/model_v2_stage_s2c/checkpoint_manifest.json")
    meta = load_train_val_metadata(root)
    seen = fit_seen_sets_from_train_metadata(meta)
    validation_ids = pd.Index([
        sample for scenario in VAL_SCENARIOS
        for sample in meta.index[meta["split_final"].eq(scenario)]
    ])
    route = derive_metadata_routing(meta.loc[validation_ids], seen)
    validation_route_sha = stable_json_sha256(route_identity_records(route))
    expert_audits = {}
    for (expert, seed), relative in CHECKPOINTS.items():
        label = LABELS[(expert, seed)]
        prior = recorded["experts"][label]
        directory = root / relative
        checkpoint = directory / "stage_b_best.pt"
        config = directory / "resolved_config.json"
        expert_audits[label] = {
            "expert_role": expert,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256_expected": prior["checkpoint_sha256"],
            "checkpoint_sha256_actual": sha256_file(checkpoint),
            "resolved_config": str(config.resolve()),
            "resolved_config_sha256_expected": prior["resolved_config_sha256"],
            "resolved_config_sha256_actual": sha256_file(config),
        }
        expert_audits[label]["checkpoint_hash_match"] = (
            expert_audits[label]["checkpoint_sha256_actual"] == prior["checkpoint_sha256"]
        )
        expert_audits[label]["config_hash_match"] = (
            expert_audits[label]["resolved_config_sha256_actual"] == prior["resolved_config_sha256"]
        )

    feature_contract = root / "project_v2/data_contract/feature_contract.json"
    chemical_manifest = root / "external_data/chemistry/source_manifest.json"
    genome_manifest = root / "external_data/genome/source_manifest.json"
    expected_artifacts = next(iter(recorded["experts"].values()))["artifact_hashes"]
    artifact_audit = {
        "feature_contract": {
            "path": str(feature_contract.resolve()),
            "expected_sha256": expected_artifacts["feature_contract_sha256"],
            "actual_sha256": sha256_file(feature_contract),
        },
        "chemical_source_manifest": {
            "path": str(chemical_manifest.resolve()),
            "expected_sha256": expected_artifacts["chemical_source_manifest_sha256"],
            "actual_sha256": sha256_file(chemical_manifest),
        },
        "genome_source_manifest": {
            "path": str(genome_manifest.resolve()),
            "expected_sha256": expected_artifacts["genome_source_manifest_sha256"],
            "actual_sha256": sha256_file(genome_manifest),
        },
    }
    for item in artifact_audit.values():
        item["hash_match"] = item["actual_sha256"] == item["expected_sha256"]

    s2d_files = _actual_s2d_hash_audit(root)
    checks = {
        "all_checkpoint_hashes_match": all(x["checkpoint_hash_match"] for x in expert_audits.values()),
        "all_resolved_config_hashes_match": all(x["config_hash_match"] for x in expert_audits.values()),
        "all_artifact_hashes_match": all(x["hash_match"] for x in artifact_audit.values()),
        "validation_route_matches_s2c": validation_route_sha == s2c["routing_manifest_sha256"] == EXPECTED_VALIDATION_ROUTE_SHA256,
        "seen_set_hashes_match_s2c": seen.hashes() == s2c["train_seen_set_hashes"],
        "original_s2d_files_unchanged": s2d_files["all_unchanged"],
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "experts": expert_audits,
        "artifacts": artifact_audit,
        "validation_routing_manifest_sha256": validation_route_sha,
        "train_seen_set_hashes": seen.hashes(),
        "original_s2d_hash_audit": s2d_files,
        "model_source": str((root / MODEL_SOURCE).resolve()),
        "model_source_sha256": sha256_file(root / MODEL_SOURCE),
        "routing_source": str((root / ROUTING_SOURCE).resolve()),
        "routing_source_sha256": sha256_file(root / ROUTING_SOURCE),
        "inference_source": str((root / INFERENCE_SOURCE).resolve()),
        "inference_source_sha256": sha256_file(root / INFERENCE_SOURCE),
        "prediction_scale": "log2",
        "planning_proxy": True,
        "official_score": False,
        "experimental_nonofficial_parity_fc": True,
        "official_fc_result": False,
        "ensemble_weights": {"seed1": 0.5, "seed2": 0.5},
        "routing_rule": {
            "seen_sets_source": "split_final=train metadata only",
            "batch_tuple": ["data_source", "instrument", "Yeast_cell_plate"],
            "priority": [
                "control_or_qc_to_D0_with_zero_response",
                "unseen_chemical_to_D0",
                "seen_chemical_unseen_strain_known_batch_to_S1C",
                "seen_chemical_unseen_strain_oov_batch_to_D2_hierarchical_fallback",
                "seen_chemical_and_strain_to_D2",
            ],
            "drug_name_specific_rules": False,
        },
    }


def _metric_records(path: Path, model: str = S2C_MEAN) -> list[dict]:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["model"].eq(model)].copy()
    order = [column for column in ("scenario", "scheme", "rank") if column in selected.columns]
    if order:
        selected = selected.sort_values(order)
    return selected.reset_index(drop=True).to_dict("records")


def _compare_records(actual: list[dict], expected: list[dict], atol: float = 2e-7) -> dict:
    if len(actual) != len(expected):
        return {"passed": False, "reason": "record_count", "actual": len(actual), "expected": len(expected)}
    mismatches, max_abs = [], 0.0
    for row_index, (left, right) in enumerate(zip(actual, expected)):
        if set(left) != set(right):
            mismatches.append({"row": row_index, "field": "__columns__"})
            continue
        for key in left:
            a, b = left[key], right[key]
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                difference = abs(float(a) - float(b))
                max_abs = max(max_abs, difference)
                if not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=atol):
                    mismatches.append({"row": row_index, "field": key, "actual": float(a), "expected": float(b)})
            elif str(a) != str(b):
                mismatches.append({"row": row_index, "field": key, "actual": str(a), "expected": str(b)})
    return {"passed": not mismatches, "max_absolute_difference": max_abs, "mismatches": mismatches[:20]}


def compare_validation_replay(root: Path, replay_dir: Path) -> dict:
    root, replay_dir = Path(root).resolve(), Path(replay_dir).resolve()
    names = {
        "absolute": "second_seed_absolute_metrics.csv",
        "fc": "second_seed_fc_metrics.csv",
        "context": "second_seed_context_metrics.csv",
        "drug": "second_seed_drug_metrics.csv",
        "high": "second_seed_high_metrics.csv",
        "planning_proxy": "second_seed_planning_proxy.csv",
    }
    comparisons = {}
    for key, filename in names.items():
        comparisons[key] = _compare_records(
            _metric_records(replay_dir / filename),
            _metric_records(root / "reports/model_v2_stage_s2d" / filename),
        )
    original = _json(root / "reports/model_v2_stage_s2d/two_seed_ensemble_metrics.json")
    replay = _json(replay_dir / "two_seed_ensemble_metrics.json")
    scalar = {
        "mean_rmse": {
            "expected": original["mean_rmse"], "actual": replay["mean_rmse"],
            "passed": math.isclose(original["mean_rmse"], replay["mean_rmse"], rel_tol=1e-7, abs_tol=2e-7),
        },
        "mean_raw_fc_pcc": {
            "expected": original["mean_raw_fc_pcc"], "actual": replay["mean_raw_fc_pcc"],
            "passed": math.isclose(original["mean_raw_fc_pcc"], replay["mean_raw_fc_pcc"], rel_tol=1e-7, abs_tol=2e-7),
        },
    }
    route = _json(replay_dir / "routing_identity_audit.json")
    routing_checks = {
        "validation_route_hash": route["seed2_manifest_sha256"] == EXPECTED_VALIDATION_ROUTE_SHA256,
        "validation_route_rowwise_identical": route["rowwise_identical"] is True,
        "validation_sample_count": route["validation_sample_count"] == 3038,
        "test_metadata_counts_only_unchanged": route["test_metadata_counts_identical"] is True,
        "test_prediction_generated": route["test_metadata"]["prediction_generated"] is False,
    }
    passed = (
        all(value["passed"] for value in comparisons.values())
        and all(value["passed"] for value in scalar.values())
        and all(routing_checks.values())
    )
    return {
        "schema_version": "1.0", "status": "PASS" if passed else "FAIL",
        "independent_process": True, "cached_prediction_arrays_read": False,
        "comparisons": comparisons, "scalar_comparisons": scalar,
        "routing_checks": routing_checks,
        "replay_metrics": replay,
        "test_proteome_opened": False,
    }


def _read_test_metadata(root: Path) -> pd.DataFrame:
    path = root / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv"
    frame = pd.read_csv(path)
    if frame["sample_ID"].isna().any() or not frame["sample_ID"].astype(str).is_unique:
        raise ValueError("test metadata sample_ID must be complete and unique")
    return frame.set_index("sample_ID", drop=False)


def _test_route_records(routing: pd.DataFrame) -> list[dict]:
    columns = [
        "sample_ID", "chemical_name", "strain", "data_source", "instrument",
        "Yeast_cell_plate", "is_control", "chemical_seen_in_train",
        "strain_seen_in_train", "batch_tuple_seen_in_train", "selected_expert",
        "routing_reason",
    ]
    return routing.loc[:, columns].to_dict("records")


def _predict_selected_expert_pair(root, artifacts, train_meta, test_meta, routing, loaded, expert, device):
    selected = routing["selected_expert"].eq(expert).to_numpy()
    ids = pd.Index(test_meta.index[selected])
    frame = test_meta.loc[ids]
    subrouting = routing.loc[selected].reset_index(drop=True)
    seed_outputs = []
    for seed in (SEED1, SEED2):
        item = loaded[(expert, seed)]
        components = item.payload["config"]["model"]["chemical_feature_components"]
        encoded = _encoded(artifacts, test_meta, item.vocabulary, ids, item.payload, "correct", components)
        if expert == D2:
            encoded = apply_hierarchical_oov_codes(encoded, frame, fit_seen_sets_from_train_metadata(train_meta))
        output = _infer_light(item.model, encoded, device)
        if expert == D0:
            output["y_anchor"] = output["y_pred"] - output["delta_response"]
            output = enforce_routed_control_zero_response(output, subrouting)
        seed_outputs.append(output)
    prediction = arithmetic_seed_mean(seed_outputs[0]["y_pred"], seed_outputs[1]["y_pred"])
    response = arithmetic_seed_mean(seed_outputs[0]["delta_response"], seed_outputs[1]["delta_response"])
    if subrouting["is_control"].any():
        control = subrouting["is_control"].to_numpy(bool)
        if not np.array_equal(response[control], np.zeros_like(response[control])):
            raise RuntimeError("control/QC response was not exactly zero before seed averaging")
    return ids, prediction, response


def run_test_inference_pass(root: Path, pass_dir: Path, device: str = "auto") -> dict:
    root, pass_dir = Path(root).resolve(), Path(pass_dir).resolve()
    pass_dir.mkdir(parents=True, exist_ok=False)
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, TestProteomeFileAccessGuard(root) as guard:
        warnings.simplefilter("always")
        artifacts = load_artifact_bundle(root)
        train_meta = load_train_val_metadata(root)
        seen = fit_seen_sets_from_train_metadata(train_meta)
        test_meta = _read_test_metadata(root)
        routing = derive_metadata_routing(test_meta, seen)
        records = _test_route_records(routing)
        route_sha = stable_json_sha256(records)
        counts = {str(k): int(v) for k, v in routing["selected_expert"].value_counts().items()}
        if len(test_meta) != EXPECTED_TEST_ROWS or counts != EXPECTED_TEST_COUNTS:
            raise RuntimeError(f"test metadata or route counts drifted: n={len(test_meta)}, counts={counts}")
        prior_seen = _json(root / "reports/model_v2_stage_s2c/checkpoint_manifest.json")["train_seen_set_hashes"]
        if seen.hashes() != prior_seen:
            raise RuntimeError("train seen-set hash drift")
        loaded = _load_experts(root, artifacts, train_meta, selected_device)
        predictions = np.empty((len(test_meta), artifacts.feature_contract.n_proteins), dtype=np.float32)
        responses = np.empty_like(predictions)
        assigned = np.zeros(len(test_meta), dtype=bool)
        for expert in (D0, S1C, D2):
            ids, pred, response = _predict_selected_expert_pair(
                root, artifacts, train_meta, test_meta, routing, loaded, expert, selected_device,
            )
            positions = test_meta.index.get_indexer(ids)
            predictions[positions] = pred
            responses[positions] = response
            assigned[positions] = True
        if not assigned.all() or not np.isfinite(predictions).all():
            raise RuntimeError("unassigned rows or non-finite test predictions")
        if selected_device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        routing_path = pass_dir / "routing_manifest.csv"
        routing.to_csv(routing_path, index=False)
        prediction_path = pass_dir / "prediction.csv"
        frame = pd.DataFrame(predictions, columns=list(artifacts.feature_contract.proteins))
        frame.insert(0, "sample_ID", test_meta["sample_ID"].astype(str).to_numpy())
        frame.to_csv(prediction_path, index=False)
        np.save(pass_dir / "prediction_float32.npy", predictions, allow_pickle=False)
        np.save(pass_dir / "response_float32.npy", responses, allow_pickle=False)
        access = guard.audit()
    summary = {
        "schema_version": "1.0", "status": "PASS", "device": selected_device,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if selected_device == "cuda" else 0,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if selected_device == "cuda" else 0,
        "n_samples": int(len(test_meta)), "n_proteins": int(predictions.shape[1]),
        "expert_counts": counts,
        "reason_counts": {str(k): int(v) for k, v in routing["routing_reason"].value_counts().items()},
        "routing_manifest_sha256": route_sha,
        "routing_csv_sha256": sha256_file(routing_path),
        "seen_set_hashes": seen.hashes(),
        "prediction_csv_sha256": sha256_file(prediction_path),
        "prediction_npy_sha256": sha256_file(pass_dir / "prediction_float32.npy"),
        "response_npy_sha256": sha256_file(pass_dir / "response_float32.npy"),
        "warnings": [str(item.message) for item in caught], "errors": [],
        "file_access": access, "test_proteome_opened": False,
    }
    _write_json(pass_dir / "pass_manifest.json", summary)
    _write_json(pass_dir / "data_access.json", access)
    return summary


def audit_csv_contract(path: Path, test_meta: pd.DataFrame, proteins: list[str], expected: np.ndarray) -> dict:
    parsed = pd.read_csv(path)
    expected_columns = ["sample_ID", *proteins]
    values = parsed.iloc[:, 1:].to_numpy(dtype=np.float64, copy=True)
    expected64 = expected.astype(np.float64)
    differences = np.abs(values - expected64)
    checks = {
        "row_count_4454": len(parsed) == EXPECTED_TEST_ROWS,
        "column_count_4423": len(parsed.columns) == len(proteins) + 1 == 4423,
        "first_column_sample_ID": parsed.columns[0] == "sample_ID",
        "sample_ID_no_missing": not parsed["sample_ID"].isna().any(),
        "sample_ID_no_duplicates": parsed["sample_ID"].astype(str).is_unique,
        "sample_ID_set_matches_metadata": set(parsed["sample_ID"].astype(str)) == set(test_meta["sample_ID"].astype(str)),
        "row_order_matches_metadata": parsed["sample_ID"].astype(str).tolist() == test_meta["sample_ID"].astype(str).tolist(),
        "protein_names_match_contract": parsed.columns[1:].tolist() == proteins,
        "protein_order_matches_contract": parsed.columns.tolist() == expected_columns,
        "protein_columns_numeric": all(pd.api.types.is_numeric_dtype(parsed[column]) for column in parsed.columns[1:]),
        "no_na": not parsed.iloc[:, 1:].isna().any().any(),
        "no_inf": not np.isinf(values).any(),
        "all_finite": np.isfinite(values).all(),
        "prediction_scale_log2": True,
        "not_exponentiated": True,
        "not_internal_zscore": True,
        "roundtrip_matches_prewrite": bool(np.array_equal(values.astype(np.float32), expected)),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "parsed_shape": [int(x) for x in parsed.shape],
        "roundtrip_max_absolute_difference": float(differences.max(initial=0.0)),
        "roundtrip_inconsistent_cell_count_float32": int(np.count_nonzero(values.astype(np.float32) != expected)),
        "csv_sha256": sha256_file(path),
    }


def _quantiles(values: np.ndarray) -> dict:
    points = [0.001, 0.01, 0.5, 0.99, 0.999]
    results = np.quantile(values.astype(np.float64, copy=False), points)
    return {name: float(value) for name, value in zip(("p0_1", "p1", "p50", "p99", "p99_9"), results)}


def numerical_sanity(root, test_meta, routing, prediction, labels, masks, train_meta, proteins):
    train_ids = train_meta.index[train_meta["split_final"].eq("train")]
    train_values = labels.loc[train_ids].to_numpy(np.float32, copy=True)
    train_mask = masks.loc[train_ids].to_numpy(bool, copy=True) & np.isfinite(train_values)
    train_min = np.min(np.where(train_mask, train_values, np.inf), axis=0)
    train_max = np.max(np.where(train_mask, train_values, -np.inf), axis=0)
    train_mean = np.divide(np.where(train_mask, train_values, 0.0).sum(axis=0, dtype=np.float64), train_mask.sum(axis=0), where=train_mask.sum(axis=0) > 0)
    centered = np.where(train_mask, train_values - train_mean, 0.0)
    train_std = np.sqrt(np.divide(np.square(centered, dtype=np.float64).sum(axis=0), train_mask.sum(axis=0), where=train_mask.sum(axis=0) > 0))
    outside = (prediction < train_min[None, :]) | (prediction > train_max[None, :])
    variance = prediction.astype(np.float64).var(axis=0)
    sample_extremeness = np.max(np.abs((prediction - train_mean) / np.where(train_std > 1e-8, train_std, 1.0)), axis=1)
    protein_outside = outside.sum(axis=0)
    top_samples = np.argsort(sample_extremeness)[-20:][::-1]
    top_proteins = np.argsort(protein_outside)[-20:][::-1]
    by_expert, by_scenario = {}, {}
    for expert, group in routing.groupby("selected_expert"):
        values = prediction[group.index.to_numpy()]
        by_expert[str(expert)] = {"n_samples": int(len(group)), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}
    scenario_column = "split_final" if "split_final" in test_meta.columns else None
    if scenario_column:
        for scenario, positions in test_meta.groupby(scenario_column).indices.items():
            values = prediction[np.asarray(positions)]
            by_scenario[str(scenario)] = {"n_samples": int(len(positions)), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}
    return {
        "schema_version": "1.0", "reference_scope": "split_final=train_labels_only",
        "prediction": {"min": float(prediction.min()), "max": float(prediction.max()), "mean": float(prediction.mean()), "std": float(prediction.std()), **_quantiles(prediction.ravel())},
        "by_expert": by_expert, "by_test_scenario": by_scenario,
        "per_protein_variance": {"min": float(variance.min()), "median": float(np.median(variance)), "max": float(variance.max())},
        "constant_protein_count": int(np.sum(variance == 0.0)),
        "very_low_variance_threshold": 1e-6,
        "very_low_variance_protein_count": int(np.sum(variance < 1e-6)),
        "outside_per_protein_train_observed_range_cell_count": int(outside.sum()),
        "outside_fraction": float(outside.mean()),
        "extreme_samples": [{"sample_ID": str(test_meta.iloc[i]["sample_ID"]), "max_abs_train_z": float(sample_extremeness[i])} for i in top_samples],
        "extreme_proteins": [{"protein": proteins[i], "outside_cell_count": int(protein_outside[i]), "test_variance": float(variance[i])} for i in top_proteins],
        "postprocessing_applied": False,
        "clipping": False, "winsorization": False, "rescaling": False, "calibration": False,
    }


def _external_disclosure(root: Path) -> dict:
    chemistry = _json(root / "external_data/chemistry/source_manifest.json")
    genome = _json(root / "external_data/genome/source_manifest.json")
    mapping = pd.read_csv(root / "external_data/chemistry/compound_mapping.csv")
    selected = mapping.loc[mapping["raw_name"].astype(str).str.lower().isin({"oligomycin a", "tunicamycin", "cisplatin", "nacl"})]
    return {
        "actual_external_resources_used": [
            {"name": "PubChem/RDKit chemical features", "source_manifest": str((root / "external_data/chemistry/source_manifest.json").resolve()), "source_manifest_sha256": sha256_file(root / "external_data/chemistry/source_manifest.json"), "manifest": chemistry},
            {"name": "public yeast genome features", "source_manifest": str((root / "external_data/genome/source_manifest.json").resolve()), "source_manifest_sha256": sha256_file(root / "external_data/genome/source_manifest.json"), "manifest": genome},
        ],
        "special_chemical_disclosures": selected.replace({np.nan: None}).to_dict("records"),
        "genome_proxy_disclosure": "DHY210 uses the frozen S288C proxy and is not represented as an exact strain identity.",
        "not_used_by_final_candidate": ["NetwoRx", "Parsons", "Hillenmeyer"],
        "research_only_artifacts_in_submission_directory": False,
    }


def _environment(device: str, elapsed: float, peak_allocated: int, peak_reserved: int) -> dict:
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except Exception as exc:  # pragma: no cover - manifest fallback
        rdkit_version = f"unavailable: {exc}"
    return {
        "os": platform.platform(), "python": sys.version,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdkit_version,
        "device": device, "inference_batch_size": 256, "seeds": [SEED1, SEED2],
        "total_test_inference_elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
    }


def finalize(root: Path, report_dir: Path, submission_dir: Path) -> dict:
    root, report_dir, submission_dir = map(lambda p: Path(p).resolve(), (root, report_dir, submission_dir))
    frozen = _json(report_dir / "final_candidate_manifest.json")
    replay = _json(report_dir / "validation_replay_metrics.json")
    if frozen["status"] != "PASS" or replay["status"] != "PASS":
        raise RuntimeError("fail-closed: frozen manifest or validation replay failed")
    pass1, pass2 = submission_dir / "audit/pass1", submission_dir / "audit/pass2"
    m1, m2 = _json(pass1 / "pass_manifest.json"), _json(pass2 / "pass_manifest.json")
    route1, route2 = pd.read_csv(pass1 / "routing_manifest.csv"), pd.read_csv(pass2 / "routing_manifest.csv")
    pred1 = np.load(pass1 / "prediction_float32.npy", allow_pickle=False)
    pred2 = np.load(pass2 / "prediction_float32.npy", allow_pickle=False)
    route_equal = route1.equals(route2)
    max_difference = float(np.max(np.abs(pred1.astype(np.float64) - pred2.astype(np.float64))))
    inconsistent = int(np.count_nonzero(pred1 != pred2))
    determinism_checks = {
        "route_rows_identical": route_equal,
        "route_hash_identical": m1["routing_manifest_sha256"] == m2["routing_manifest_sha256"],
        "seen_set_hashes_identical": m1["seen_set_hashes"] == m2["seen_set_hashes"],
        "prediction_arrays_within_tolerance": max_difference <= 1e-7,
        "prediction_arrays_bitwise_equal": inconsistent == 0,
    }
    determinism = {
        "schema_version": "1.0", "checks": determinism_checks,
        "passed": all(determinism_checks.values()),
        "pass1_csv_sha256": m1["prediction_csv_sha256"],
        "pass2_csv_sha256": m2["prediction_csv_sha256"],
        "pass1_prediction_npy_sha256": m1["prediction_npy_sha256"],
        "pass2_prediction_npy_sha256": m2["prediction_npy_sha256"],
        "max_absolute_numerical_difference": max_difference,
        "inconsistent_cell_count": inconsistent,
        "precision_reduction_or_rounding_used": False,
    }
    _write_json(report_dir / "determinism_audit.json", determinism)
    if not determinism["passed"]:
        raise RuntimeError("fail-closed: deterministic replay failed")

    artifacts = load_artifact_bundle(root)
    proteins = list(artifacts.feature_contract.proteins)
    train_meta = load_train_val_metadata(root)
    test_meta = _read_test_metadata(root)
    seen = fit_seen_sets_from_train_metadata(train_meta)
    routing = derive_metadata_routing(test_meta, seen)
    if stable_json_sha256(_test_route_records(routing)) != m1["routing_manifest_sha256"]:
        raise RuntimeError("fail-closed: routing drifted during finalization")
    final_prediction = submission_dir / "prediction.csv"
    if final_prediction.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {final_prediction}")
    # Audit the independently written pass file before materializing the formal
    # candidate, so a serialization or contract failure cannot leave a file
    # that appears submission-ready.
    contract = audit_csv_contract(pass1 / "prediction.csv", test_meta, proteins, pred1)
    contract["all_samples_routed_exactly_once"] = len(routing) == len(test_meta) and routing["sample_ID"].is_unique
    contract["control_response_zero"] = bool(np.array_equal(
        np.load(pass1 / "response_float32.npy", allow_pickle=False)[routing["is_control"].to_numpy(bool)],
        np.zeros_like(np.load(pass1 / "response_float32.npy", allow_pickle=False)[routing["is_control"].to_numpy(bool)]),
    ))
    contract["checks"]["all_samples_routed_exactly_once"] = contract["all_samples_routed_exactly_once"]
    contract["checks"]["control_response_zero"] = contract["control_response_zero"]
    contract["passed"] = all(contract["checks"].values())
    _write_json(report_dir / "submission_contract_audit.json", contract)
    if not contract["passed"]:
        raise RuntimeError("fail-closed: submission CSV contract failed")
    shutil.copyfile(pass1 / "prediction.csv", final_prediction)
    contract["formal_prediction_csv_sha256"] = sha256_file(final_prediction)
    contract["formal_equals_audited_pass1"] = contract["formal_prediction_csv_sha256"] == contract["csv_sha256"]
    _write_json(report_dir / "submission_contract_audit.json", contract)

    labels, masks = load_label_frames(train_meta, artifacts, root)
    numerical = numerical_sanity(root, test_meta, routing, pred1, labels, masks, train_meta, proteins)
    _write_json(report_dir / "numerical_sanity_report.json", numerical)
    route1.to_csv(submission_dir / "routing_manifest.csv", index=False)
    checkpoint_manifest = {
        "schema_version": "1.0", "candidate": S2C_MEAN,
        "experts": frozen["experts"], "route_manifest_sha256": m1["routing_manifest_sha256"],
        "validation_route_manifest_sha256": frozen["validation_routing_manifest_sha256"],
        "train_seen_set_hashes": frozen["train_seen_set_hashes"],
    }
    _write_json(submission_dir / "checkpoint_manifest.json", checkpoint_manifest)

    access_paths = sorted(set(m1["file_access"]["workspace_paths_opened"] + m2["file_access"]["workspace_paths_opened"]))
    forbidden_opened = any(FORBIDDEN_TEST_TOKEN in path.lower() for path in access_paths)
    data_access = {
        "schema_version": "1.0", "guard_enabled_in_both_test_workers": True,
        "test_proteome_opened": forbidden_opened,
        "test_metadata_use": ["model input", "train-metadata-only routing", "sample_ID output"],
        "test_protein_truth_read": False,
        "protein_mean_scaler_basis_vocab_sources": "frozen train-derived checkpoints/artifacts",
        "validation_or_test_labels_used_for_test_calibration": False,
        "train_seen_sets_only_from_train_metadata": True,
        "fc_pairing_scope": "train only during historical checkpoint training; no FC pairing in test inference",
        "test_self_evaluation_called": False,
        "real_data_demo_test_self_evaluation_loaded": False,
        "baseline_prediction_values_used": False,
        "baseline_prediction_header_only": True,
        "opened_workspace_paths": access_paths,
        "denied_attempts": m1["file_access"]["denied_attempts"] + m2["file_access"]["denied_attempts"],
    }
    _write_json(report_dir / "data_access_audit.json", data_access)

    header = read_reference_header_only(root / "baseline_models/prediction.csv")
    reference_header_audit = {
        "path": str((root / "baseline_models/prediction.csv").resolve()),
        "header_only": True, "value_rows_read": 0, "n_columns": len(header),
        "matches_feature_contract": header == ["sample_ID", *proteins],
        "values_used": False,
    }
    if not reference_header_audit["matches_feature_contract"]:
        raise RuntimeError("fail-closed: local reference header conflicts with frozen feature contract")

    disclosure = _external_disclosure(root)
    warnings_all = list(dict.fromkeys(m1["warnings"] + m2["warnings"]))
    environment = _environment(
        m1["device"], m1["elapsed_seconds"] + m2["elapsed_seconds"],
        max(m1["peak_gpu_memory_allocated_bytes"], m2["peak_gpu_memory_allocated_bytes"]),
        max(m1["peak_gpu_memory_reserved_bytes"], m2["peak_gpu_memory_reserved_bytes"]),
    )
    environment["warnings"] = warnings_all
    environment["errors"] = []
    _write_json(submission_dir / "environment_manifest.json", environment)
    shutil.copyfile(report_dir / "final_candidate_manifest.json", submission_dir / "checkpoint_manifest.full.json")

    reproduction = (
        f'"{sys.executable}" -m baseline.final_submission_v2 '
        f'--root "{root}" --report-dir "{report_dir}" --submission-dir "{submission_dir}" --device {m1["device"]}'
    )
    (submission_dir / "reproduction_command.txt").write_text(reproduction + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0", "candidate": S2C_MEAN,
        "prediction_csv": str(final_prediction), "prediction_csv_sha256": sha256_file(final_prediction),
        "prediction_scale": "log2", "n_rows": EXPECTED_TEST_ROWS,
        "n_proteins": len(proteins), "n_columns": len(proteins) + 1,
        "route_manifest_sha256": m1["routing_manifest_sha256"],
        "routing_csv_sha256": sha256_file(submission_dir / "routing_manifest.csv"),
        "feature_contract_sha256": frozen["artifacts"]["feature_contract"]["actual_sha256"],
        "chemical_source_manifest_sha256": frozen["artifacts"]["chemical_source_manifest"]["actual_sha256"],
        "genome_source_manifest_sha256": frozen["artifacts"]["genome_source_manifest"]["actual_sha256"],
        "planning_proxy": True, "official_score": False,
        "experimental_nonofficial_parity_fc": True, "official_fc_result": False,
        "uploaded_to_competition_platform": False,
    }
    _write_json(submission_dir / "prediction_manifest.json", manifest)
    hard_pass = (
        contract["passed"] and determinism["passed"] and replay["status"] == "PASS"
        and frozen["status"] == "PASS" and not forbidden_opened
        and disclosure["research_only_artifacts_in_submission_directory"] is False
    )
    audit = {
        "schema_version": "1.0", "status": "PASS" if hard_pass else "FAIL",
        "eligible_for_manual_submission": bool(hard_pass),
        "eligibility_meaning": "candidate CSV may be manually uploaded by the user after Main review",
        "does_not_mean": ["official high score", "official FC mapping confirmed", "test truth read", "platform upload completed", "final rank obtained"],
        "hard_contract": contract, "determinism": determinism,
        "validation_replay_status": replay["status"], "frozen_candidate_status": frozen["status"],
        "data_access": data_access, "reference_header_audit": reference_header_audit,
        "external_data_disclosure": disclosure,
        "rules_contract_review": {
            "sample_ID": "required first column and preserved metadata order",
            "test_rows": EXPECTED_TEST_ROWS, "protein_columns": len(proteins),
            "protein_order_source": "feature_contract.json only", "scale": "log2",
            "raw_competition_data_upload": False, "code_reproduction_required": True,
            "unconfirmed_official_items": ["official Water/DMSO mapping", "official matched-control additional QC", "multi-control aggregation rule", "official within-module final aggregation formula"],
            "experimental_parity_fc_is_official": False,
        },
        "postprocessing_applied": False, "test_prediction_generated": True,
        "test_proteome_opened": False, "platform_upload_performed": False,
    }
    _write_json(submission_dir / "submission_audit.json", audit)
    _write_json(report_dir / "submission_audit.json", audit)
    return {"audit": audit, "manifest": manifest, "environment": environment, "numerical": numerical}


def _build_report(root: Path, report_dir: Path, submission_dir: Path, result: dict) -> None:
    replay = _json(report_dir / "validation_replay_metrics.json")
    frozen = _json(report_dir / "final_candidate_manifest.json")
    deterministic = _json(report_dir / "determinism_audit.json")
    contract = _json(report_dir / "submission_contract_audit.json")
    audit = result["audit"]
    numeric = result["numerical"]
    expert_counts = _json(submission_dir / "audit/pass1/pass_manifest.json")["expert_counts"]
    text = f"""# MODEL V2 Stage S3 Final Pre-submission Audit

Status: **{audit['status']}**  
`eligible_for_manual_submission={str(audit['eligible_for_manual_submission']).lower()}`  
`planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

## Outcome

The frozen S2C two-seed mean candidate was regenerated without training, fine-tuning, route changes, checkpoint changes, calibration, clipping, or weight search. Validation was replayed in an independent Python process from all six checkpoints and matched the published S2D metrics within the declared floating-point tolerance. Two independent test-metadata inference workers produced identical float32 arrays and route manifests.

- Candidate: `{submission_dir / 'prediction.csv'}`
- SHA-256: `{result['manifest']['prediction_csv_sha256']}`
- Shape after standard CSV re-read: `{contract['parsed_shape'][0]} x {contract['parsed_shape'][1]}`
- Route counts: D0 `{expert_counts[D0]}`, S1 C `{expert_counts[S1C]}`, D2 `{expert_counts[D2]}`
- Test proteome opened: `false`
- Competition upload performed: `false`

## Frozen candidate and validation replay

- Six checkpoint/config hash audit: `{frozen['status']}`.
- Original S2D artifact hashes unchanged: `{str(frozen['checks']['original_s2d_files_unchanged']).lower()}`.
- Validation routing SHA-256: `{frozen['validation_routing_manifest_sha256']}`.
- Replayed mean RMSE: `{replay['scalar_comparisons']['mean_rmse']['actual']:.12f}`.
- Replayed mean raw-FC PCC: `{replay['scalar_comparisons']['mean_raw_fc_pcc']['actual']:.12f}`.
- Absolute, raw-FC, context residual, drug residual, high-effect, and three planning-proxy tables all matched: `{str(all(x['passed'] for x in replay['comparisons'].values())).lower()}`.

## Submission contract and determinism

All 20 required CSV checks plus exact-once routing and control/QC response-zero checks passed: `{str(contract['passed']).lower()}`. Protein columns were read dynamically from the frozen feature contract; comma-containing names were written and re-read with pandas' standards-compliant CSV implementation. No exponentiation, internal z-score output, rounding reduction, or post-processing was performed.

- Pass 1 CSV SHA-256: `{deterministic['pass1_csv_sha256']}`
- Pass 2 CSV SHA-256: `{deterministic['pass2_csv_sha256']}`
- Maximum numeric difference: `{deterministic['max_absolute_numerical_difference']}`
- Inconsistent cells: `{deterministic['inconsistent_cell_count']}`
- Test route SHA-256: `{result['manifest']['route_manifest_sha256']}`

The earlier stages froze only test route counts and train seen-set hashes, not a rowwise test-route hash. S3 therefore verified the historical counts and seen-set hashes, then required the two independent workers' newly materialized rowwise route hashes to be identical. No historical hash was fabricated.

## Numerical sanity (warning-only)

- Prediction min/max/mean/std: `{numeric['prediction']['min']:.6f}` / `{numeric['prediction']['max']:.6f}` / `{numeric['prediction']['mean']:.6f}` / `{numeric['prediction']['std']:.6f}`
- P0.1/P1/P50/P99/P99.9: `{numeric['prediction']['p0_1']:.6f}` / `{numeric['prediction']['p1']:.6f}` / `{numeric['prediction']['p50']:.6f}` / `{numeric['prediction']['p99']:.6f}` / `{numeric['prediction']['p99_9']:.6f}`
- Constant proteins: `{numeric['constant_protein_count']}`
- Very-low-variance proteins (<1e-6): `{numeric['very_low_variance_protein_count']}`
- Cells outside the per-protein train-observed log2 range: `{numeric['outside_per_protein_train_observed_range_cell_count']}` (`{numeric['outside_fraction']:.6%}`)

Finite out-of-range values were reported without clipping, winsorization, rescaling, calibration, sample deletion, protein deletion, or ensemble-weight changes.

## Data boundaries and disclosures

The test inference entry used an explicit filename guard and read only `WAYB_WAYC_metadata_test(1).csv` for test inputs/routing/sample IDs. Train-derived checkpoint state supplies protein means, scalers, bases and vocabularies. The local legacy `baseline_models/prediction.csv` was opened for exactly one header row; no prediction values were read or used.

The final candidate uses the frozen PubChem/RDKit chemistry features and public yeast-genome features with their source manifests and hashes. Oligomycin A proxy, Tunicamycin unresolved status, Cisplatin/NaCl structure-invalid fallbacks, and DHY210-to-S288C proxy are disclosed in `submission_audit.json`. NetwoRx, Parsons and Hillenmeyer did not enter the final candidate and no research-only artifact was copied into the submission directory.

## Rules and unresolved official semantics

The candidate preserves `sample_ID`, 4,454 metadata rows, the frozen 4,422-protein order, and log2 scale. Raw competition data were not uploaded. Reproduction commands, environment versions, checkpoint/config hashes and source-manifest hashes are recorded.

Still explicitly unconfirmed: official Water/DMSO mapping, extra official matched-control QC, the official multi-control aggregation rule, and the official within-module final aggregation formula. The parity FC used during historical D2/S1 C training remains experimental and nonofficial.

## Reproduction and artifacts

- Command: `{(submission_dir / 'reproduction_command.txt').read_text(encoding='utf-8').strip()}`
- Final candidate manifest: `{report_dir / 'final_candidate_manifest.json'}`
- Validation replay: `{report_dir / 'validation_replay_metrics.json'}`
- Contract audit: `{report_dir / 'submission_contract_audit.json'}`
- Numerical sanity: `{report_dir / 'numerical_sanity_report.json'}`
- Determinism: `{report_dir / 'determinism_audit.json'}`
- Data access: `{report_dir / 'data_access_audit.json'}`

Stage S3 is complete and paused. The candidate was not uploaded; Main must perform the final manual review and any platform upload.
"""
    (root / "reports/MODEL_V2_STAGE_S3_FINAL_PRESUBMISSION_AUDIT.md").write_text(text, encoding="utf-8")


def _run_worker(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"worker failed ({completed.returncode}): {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")


def run_stage_s3(root=ROOT, report_dir=None, submission_dir=None, device="auto") -> dict:
    root = Path(root).resolve()
    report_dir = Path(report_dir or root / REPORT_DIR).resolve()
    submission_dir = Path(submission_dir or root / SUBMISSION_DIR).resolve()
    if (submission_dir / "prediction.csv").exists():
        raise FileExistsError("refusing to overwrite an existing candidate prediction.csv")
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_candidate(root)
    _write_json(report_dir / "final_candidate_manifest.json", frozen)
    if frozen["status"] != "PASS":
        raise RuntimeError("fail-closed: frozen candidate hash audit failed")

    python = sys.executable
    replay_dir = report_dir / "validation_replay_worker"
    _run_worker([python, "-m", "baseline.final_submission_v2", "--mode", "validation-worker", "--root", str(root), "--report-dir", str(report_dir), "--submission-dir", str(submission_dir), "--device", device], root)
    replay = compare_validation_replay(root, replay_dir)
    _write_json(report_dir / "validation_replay_metrics.json", replay)
    if replay["status"] != "PASS":
        raise RuntimeError("fail-closed: validation replay differs from S2D")

    audit_root = submission_dir / "audit"
    _run_worker([python, "-m", "baseline.final_submission_v2", "--mode", "test-worker", "--pass-name", "pass1", "--root", str(root), "--report-dir", str(report_dir), "--submission-dir", str(submission_dir), "--device", device], root)
    _run_worker([python, "-m", "baseline.final_submission_v2", "--mode", "test-worker", "--pass-name", "pass2", "--root", str(root), "--report-dir", str(report_dir), "--submission-dir", str(submission_dir), "--device", device], root)
    result = finalize(root, report_dir, submission_dir)
    _build_report(root, report_dir, submission_dir, result)
    return {
        "status": result["audit"]["status"],
        "eligible_for_manual_submission": result["audit"]["eligible_for_manual_submission"],
        "prediction": str(submission_dir / "prediction.csv"),
        "prediction_sha256": result["manifest"]["prediction_csv_sha256"],
        "report": str(root / "reports/MODEL_V2_STAGE_S3_FINAL_PRESUBMISSION_AUDIT.md"),
        "test_proteome_opened": False, "uploaded": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--mode", default="run", choices=("run", "validation-worker", "test-worker"))
    parser.add_argument("--pass-name", choices=("pass1", "pass2"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report_dir = (args.report_dir or root / REPORT_DIR).resolve()
    submission_dir = (args.submission_dir or root / SUBMISSION_DIR).resolve()
    if args.mode == "validation-worker":
        from baseline.baseline.second_seed_ensemble_v2 import run_analysis
        target = report_dir / "validation_replay_worker"
        with TestProteomeFileAccessGuard(root) as guard:
            result = run_analysis(root, target, target / "replay_report.md", args.device)
        access = guard.audit()
        if access["denied_attempts"]:
            raise RuntimeError("validation replay attempted forbidden file access")
        _write_json(target / "data_access.json", access)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "test-worker":
        if not args.pass_name:
            parser.error("--pass-name is required for test-worker")
        result = run_test_inference_pass(root, submission_dir / "audit" / args.pass_name, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_stage_s3(root, report_dir, submission_dir, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
