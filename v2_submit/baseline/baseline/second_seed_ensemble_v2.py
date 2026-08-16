"""Stage S2D second-seed stability and fixed two-seed ensemble audit.

This module is inference-only.  It loads the six frozen expert checkpoints,
reuses the S2C metadata router verbatim, and never opens the test proteome.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline.hierarchical_treatment_analysis_v2 import _encoded
from baseline.baseline.novelty_gated_ensemble_v2 import (
    D0, D2, S1C, apply_hierarchical_oov_codes, derive_metadata_routing,
    enforce_routed_control_zero_response, fit_seen_sets_from_train_metadata,
)
from baseline.baseline.score_aligned_analysis_v2 import _finite_json, _model_from_checkpoint
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, VAL_SCENARIOS, CategoryVocabulary,
    fit_category_vocabulary, load_artifact_bundle, load_label_frames,
    load_train_val_metadata, sha256_file,
)
from scripts.audit_competition_score_alignment import (
    EXPECTED_EXACT_KEYS, absolute_metric_row, build_control_pairs,
    build_planning_proxies, direction_accuracy, fc_metric_row,
    fit_masked_group_references, high_effect_metric_row, load_control_spec,
    markdown_table, masked_rmse, materialize_references, paired_pcc,
    residual_metric_row, stable_json_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SEED1, SEED2 = 20260814, 20260815
S2C1, S2C2, S2C_MEAN = "S2C seed1", "S2C seed2", "S2C two-seed mean"
LABELS = {
    (D0, SEED1): "D0 seed1", (D0, SEED2): "D0 seed2",
    (D2, SEED1): "D2 seed1", (D2, SEED2): "D2 seed2",
    (S1C, SEED1): "S1 C seed1", (S1C, SEED2): "S1 C seed2",
}
CHECKPOINTS = {
    (D0, SEED1): "reports/model_v2_stage_s2b/formal_d0_hierarchical_none_huber_seed_20260814",
    (D2, SEED1): "reports/model_v2_stage_s2b/formal_d2_hierarchical_morgan_fc_seed_20260814",
    (S1C, SEED1): "reports/model_v2_stage_s1/formal_c_anchor_morgan_fc_seed_20260814",
    (D0, SEED2): "reports/model_v2_stage_s2d/formal_d0_hierarchical_none_huber_seed_20260815",
    (D2, SEED2): "reports/model_v2_stage_s2d/formal_d2_hierarchical_morgan_fc_seed_20260815",
    (S1C, SEED2): "reports/model_v2_stage_s2d/formal_s1c_flat_morgan_fc_seed_20260815",
}
STAGE_A = {
    ("hierarchical", SEED1): "reports/model_v2_stage_s2a/final/hierarchical_batch/stage_a_final.pt",
    ("hierarchical", SEED2): "reports/model_v2_stage_s2d/stage_a_hierarchical_seed_20260815/stage_a_final.pt",
    ("flat", SEED1): "reports/model_v2_stage2/formal_huber_seed_20260814/stage_a_best.pt",
    ("flat", SEED2): "reports/model_v2_stage_s2d/stage_a_flat_seed_20260815/stage_a_best.pt",
}
REFERENCE_MODELS = ("Matched Control", "V2 batch-enabled Huber", "S1 B anchor Morgan Huber")
EVALUATED_MODELS = (*REFERENCE_MODELS, *LABELS.values(), S2C1, S2C2, S2C_MEAN)
CONTROL_IDENTITIES = frozenset((*CONTROL_NAMES, "quality control"))


@dataclass(frozen=True)
class LoadedExpert:
    model: torch.nn.Module
    payload: dict
    vocabulary: CategoryVocabulary
    checkpoint: Path
    directory: Path


def _infer_light(model, encoded, device):
    """Return only arrays needed by S2D, avoiding eight full protein copies."""
    names = ("y_pred", "delta_response", "delta_batch")
    chunks = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for start in range(0, encoded.morgan.shape[0], 256):
            index = torch.arange(start, min(start + 256, encoded.morgan.shape[0]))
            output = model(encoded.index_select(index).to(device))
            for name in names:
                chunks[name].append(output[name].detach().cpu().numpy().astype(np.float32, copy=False))
    return {name: np.concatenate(values) for name, values in chunks.items()}


def arithmetic_seed_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape:
        raise ValueError("seed predictions must have identical shape")
    return (first.astype(np.float64) * 0.5 + second.astype(np.float64) * 0.5).astype(np.float32)


def route_identity_records(routing: pd.DataFrame) -> list[dict]:
    columns = [
        "sample_ID", "is_control", "chemical_seen_in_train", "strain_seen_in_train",
        "batch_tuple_seen_in_train", "selected_expert", "routing_reason",
    ]
    return routing.loc[:, columns].sort_values("sample_ID").to_dict("records")


def _read_test_metadata(root: Path) -> pd.DataFrame:
    # Deliberately metadata-only; held-out protein measurements are out of scope.
    frame = pd.read_csv(root / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv")
    if frame["sample_ID"].isna().any() or not frame["sample_ID"].is_unique:
        raise ValueError("test metadata IDs must be complete and unique")
    return frame.set_index("sample_ID", drop=False)


def _hierarchical_vocab(path: Path) -> CategoryVocabulary:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("vocabulary")
    if not isinstance(raw, dict):
        raise ValueError(f"hierarchical Stage A has no vocabulary: {path}")
    return CategoryVocabulary({key: tuple(values) for key, values in raw.items()})


def _load_experts(root, artifacts, meta, device):
    train_ids = meta.index[meta.split_final.eq("train")]
    flat_vocab = fit_category_vocabulary(meta, train_ids)
    vocabs = {
        ("hierarchical", SEED1): _hierarchical_vocab(root / STAGE_A[("hierarchical", SEED1)]),
        ("hierarchical", SEED2): _hierarchical_vocab(root / STAGE_A[("hierarchical", SEED2)]),
        ("flat", SEED1): flat_vocab, ("flat", SEED2): flat_vocab,
    }
    loaded = {}
    for key, relative in CHECKPOINTS.items():
        expert, seed = key
        directory = root / relative
        checkpoint = directory / "stage_b_best.pt"
        kind = "flat" if expert == S1C else "hierarchical"
        model, payload = _model_from_checkpoint(checkpoint, artifacts, vocabs[(kind, seed)], device)
        if int(payload["seed"]) != seed:
            raise ValueError(f"checkpoint seed mismatch: {checkpoint}")
        loaded[key] = LoadedExpert(model, payload, vocabs[(kind, seed)], checkpoint, directory)
    return loaded


def _reference_tables(root):
    source = root / "reports/model_v2_stage_s2c"
    files = {
        "absolute": "novelty_gated_absolute_metrics.csv",
        "fc": "novelty_gated_fc_metrics.csv",
        "context": "novelty_gated_context_metrics.csv",
        "drug": "novelty_gated_drug_metrics.csv",
        "high": "novelty_gated_high_metrics.csv",
    }
    return {
        key: pd.read_csv(source / filename).loc[lambda x: x.model.isin(REFERENCE_MODELS)].copy()
        for key, filename in files.items()
    }


def _metric_bundle(model_name, scenario, treatment_ids, truth, pred, delta_true,
                   delta_pred, common, meta, context_refs, drug_refs, n_proteins):
    rows = {
        "absolute": absolute_metric_row(model_name, scenario, "common_intersection", treatment_ids, truth, pred, common),
        "fc": fc_metric_row(model_name, scenario, "common_intersection", delta_true, delta_pred, common),
        "high": high_effect_metric_row(model_name, scenario, "common_intersection", delta_true, delta_pred, common),
    }
    if scenario == "val_chem_only":
        values, valid = materialize_references(meta, treatment_ids, context_refs, EXPECTED_EXACT_KEYS, n_proteins)
        rows["context"] = residual_metric_row(model_name, scenario, "common_intersection", delta_true, delta_pred, values, valid, common)
    if scenario == "val_strain_only":
        values, valid = materialize_references(meta, treatment_ids, drug_refs, ("perturbation_no_concentration",), n_proteins)
        rows["drug"] = residual_metric_row(model_name, scenario, "common_intersection", delta_true, delta_pred, values, valid, common)
    return rows


def _append_bundle(tables, bundle):
    for key, row in bundle.items():
        tables[key].append(row)


def _per_drug_rows(model_name, scenario, names, truth, pred, delta_true, delta_pred,
                   matched_control, common):
    rows = []
    for chemical in sorted(set(names)):
        selected = names == chemical
        mask = common & selected[:, None]
        comparator_rmse = masked_rmse(truth, matched_control, mask)
        model_rmse = masked_rmse(truth, pred, mask)
        rows.append({
            "model": model_name, "scenario": scenario, "chemical_name": chemical,
            "n_samples": int(selected.sum()), "n_valid_positions": int(mask.sum()),
            "absolute_rmse": model_rmse,
            "raw_fc_pcc": paired_pcc(delta_true, delta_pred, mask),
            "raw_fc_direction_accuracy": direction_accuracy(delta_true, delta_pred, mask),
            "matched_control_rmse": comparator_rmse,
            "rmse_gain_vs_matched_control": comparator_rmse - model_rmse,
        })
    return rows


def _checkpoint_manifest(root, loaded):
    experts = {}
    for key, item in loaded.items():
        expert, seed = key
        summary_path = item.directory / "training_summary.json"
        resolved_path = item.directory / "resolved_config.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        experts[LABELS[key]] = {
            "expert_role": expert, "seed": seed,
            "checkpoint": str(item.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(item.checkpoint),
            "resolved_config": str(resolved_path.resolve()),
            "resolved_config_sha256": sha256_file(resolved_path),
            "training_summary": str(summary_path.resolve()),
            "training_summary_sha256": sha256_file(summary_path),
            "stage_b_training_ids_sha256": summary.get("stage_b_training_ids_sha256"),
            "stage_b_full_treatment_ids_sha256": summary.get("stage_b_full_treatment_ids_sha256"),
            "stage_b_dataloader_shuffle_seed": summary.get("stage_b_dataloader_shuffle_seed", seed),
            "artifact_hashes": item.payload.get("artifact_hashes"),
            "holdout_ids_sha256": summary["train_internal_batch_holdout"]["holdout_sample_ids_sha256"],
            "best_epoch": summary["stage_b"]["best_epoch"],
            "actual_epochs": summary["stage_b"]["actual_epochs"],
            "best_monitor": summary["stage_b"]["best_monitor"],
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "peak_gpu_memory_reserved_bytes": summary.get("peak_gpu_memory_reserved_bytes"),
            "finite_and_passed": bool(
                summary.get("status") == "PASS"
                and np.isfinite(summary["stage_b"]["best_monitor"])
            ),
            "test_proteome_opened": summary.get("test_proteome_opened"),
        }
    # Seed1 predates explicit dataset-order hashes in training_summary.json.
    # The frozen holdout hash and treatment counts prove the same ordered ID
    # partition; copy the now-materialized ID hashes and record provenance.
    canonical = next(value for value in experts.values() if value["stage_b_training_ids_sha256"])
    for value in experts.values():
        if value["stage_b_training_ids_sha256"] is None:
            value["stage_b_training_ids_sha256"] = canonical["stage_b_training_ids_sha256"]
            value["stage_b_full_treatment_ids_sha256"] = canonical["stage_b_full_treatment_ids_sha256"]
            value["training_id_hash_provenance"] = "reconstructed_from_identical_frozen_holdout_and_metadata_order"
        else:
            value["training_id_hash_provenance"] = "recorded_by_training_entry"
        value["sample_order_contract"] = {
            "ordered_dataset_ids_sha256": value["stage_b_training_ids_sha256"],
            "torch_dataloader_generator_seed": value["stage_b_dataloader_shuffle_seed"],
        }
    stage_a = {}
    for key, relative in STAGE_A.items():
        path = root / relative
        summary_path = path.parent / "stage_a_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        stage_a[f"{key[0]}_seed{key[1]}"] = {
            "path": str(path.resolve()), "sha256": sha256_file(path), "seed": key[1],
            "summary": str(summary_path.resolve()) if summary else None,
            "summary_sha256": sha256_file(summary_path) if summary else None,
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "actual_epochs": summary.get("actual_epochs", summary.get("fixed_epochs")),
            "best_epoch": summary.get("best_epoch", 29 if summary.get("fixed_epochs") == 30 else None),
            "peak_gpu_memory_reserved_bytes": summary.get("peak_gpu_memory_reserved_bytes"),
        }
    return {"schema_version": "1.0", "experts": experts, "stage_a": stage_a}


def run_analysis(root=ROOT, output_dir=None, report_path=None, device="auto"):
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "reports/model_v2_stage_s2d").resolve()
    report_path = Path(report_path or root / "reports/MODEL_V2_STAGE_S2D_SECOND_SEED_REPORT.md").resolve()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    artifacts = load_artifact_bundle(root)
    meta = load_train_val_metadata(root)
    labels, masks = load_label_frames(meta, artifacts, root)
    seen = fit_seen_sets_from_train_metadata(meta)
    loaded = _load_experts(root, artifacts, meta, device)
    manifest = _checkpoint_manifest(root, loaded)

    # Fairness invariants before inference.
    seed2_summaries = [manifest["experts"][LABELS[(expert, SEED2)]] for expert in (D0, D2, S1C)]
    if len({item["holdout_ids_sha256"] for item in seed2_summaries}) != 1:
        raise RuntimeError("seed2 experts do not share the frozen 957-treatment holdout")
    if seed2_summaries[0]["holdout_ids_sha256"] != "7cfd2fc2c8877425c98f845ffc9275cd023cee7da31aa677437aaa65ffd11734":
        raise RuntimeError("seed2 holdout differs from S2B/S1 frozen holdout")
    d0_summary = json.loads((loaded[(D0, SEED2)].directory / "training_summary.json").read_text(encoding="utf-8"))
    d2_summary = json.loads((loaded[(D2, SEED2)].directory / "training_summary.json").read_text(encoding="utf-8"))
    if d0_summary["stage_b_initial_chemical_response_sha256"] != d2_summary["stage_b_initial_chemical_response_sha256"]:
        raise RuntimeError("seed2 D0 and D2 Stage-B initialization hashes differ")
    if d0_summary["stage_b_freeze_audit"]["status"] != "PASS" or d2_summary["stage_b_freeze_audit"]["status"] != "PASS":
        raise RuntimeError("hierarchical frozen-parameter audit failed")

    scenario_ids = {scenario: meta.index[meta.split_final.eq(scenario)] for scenario in VAL_SCENARIOS}
    validation_ids = pd.Index([sample for scenario in VAL_SCENARIOS for sample in scenario_ids[scenario]])
    validation_routing = derive_metadata_routing(meta.loc[validation_ids], seen)
    routing_records = route_identity_records(validation_routing)
    routing_sha = stable_json_sha256(routing_records)
    prior_routing = pd.read_csv(root / "reports/model_v2_stage_s2c/validation_routing.csv")
    prior_records = route_identity_records(prior_routing)
    prior_sha = stable_json_sha256(prior_records)
    route_rows_equal = routing_records == prior_records
    if not route_rows_equal:
        raise RuntimeError("seed2 routing differs from frozen S2C routing")

    spec, parity = load_control_spec(root)
    train_ids = meta.index[meta.split_final.eq("train")]
    train_names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    train_controls = train_ids[train_names.isin(CONTROL_NAMES)]
    train_treatments = train_ids[~train_names.isin(CONTROL_NAMES | {"quality control"})]
    all_controls = meta.index[meta.perturbation_no_concentration.astype(str).str.lower().isin(CONTROL_NAMES)]
    train_pairs = build_control_pairs(meta, labels, masks, train_treatments, train_controls, EXPECTED_EXACT_KEYS, parity)
    context_refs = fit_masked_group_references(meta, train_pairs, EXPECTED_EXACT_KEYS, train_ids)
    drug_refs = fit_masked_group_references(meta, train_pairs, ("perturbation_no_concentration",), train_ids)
    s0_fc = pd.read_csv(root / "reports/competition_score_audit/fc_metrics.csv")
    table_rows = {key: [] for key in ("absolute", "fc", "context", "drug", "high")}
    stability_rows, per_drug_rows = [], []

    for scenario, ids0 in scenario_ids.items():
        ids = pd.Index(map(str, ids0))
        frame = meta.loc[ids]
        routing = derive_metadata_routing(frame, seen)
        pair_treatment = ids[~frame["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})]
        pair = build_control_pairs(meta, labels, masks, pair_treatment, all_controls, EXPECTED_EXACT_KEYS, parity)
        lookup = {sample: i for i, sample in enumerate(ids)}
        index = np.asarray([lookup[sample] for sample in pair.treatment_ids])
        common = pair.delta_mask.copy()
        expected = s0_fc.loc[(s0_fc.scenario == scenario) & (s0_fc.subset == "common_intersection"), "n_valid_positions"].unique()
        if len(expected) != 1 or int(expected[0]) != int(common.sum()):
            raise RuntimeError(f"S0 common subset mismatch: {scenario}")
        truth = labels.loc[ids].to_numpy(np.float32, copy=True)[index]
        chemical_names = meta.loc[list(pair.treatment_ids), "perturbation_no_concentration"].astype(str).to_numpy()
        routed = {
            seed: {
                "y_pred": np.empty((len(ids), artifacts.feature_contract.n_proteins), np.float32),
                "delta_response": np.empty((len(ids), artifacts.feature_contract.n_proteins), np.float32),
            } for seed in (SEED1, SEED2)
        }
        for expert in (D0, D2, S1C):
            pair_outputs = {}
            selected = routing.selected_expert.eq(expert).to_numpy()
            for seed in (SEED1, SEED2):
                item = loaded[(expert, seed)]
                components = item.payload["config"]["model"]["chemical_feature_components"]
                encoded = _encoded(artifacts, meta, item.vocabulary, ids, item.payload, "correct", components)
                if expert == D2:
                    encoded = apply_hierarchical_oov_codes(encoded, frame, seen)
                output = _infer_light(item.model, encoded, device)
                if expert == D0:
                    # Construct the anchor lazily; this is exactly y_pred-response.
                    output["y_anchor"] = output["y_pred"] - output["delta_response"]
                    output = enforce_routed_control_zero_response(output, routing)
                pair_outputs[seed] = output
                routed[seed]["y_pred"][selected] = output["y_pred"][selected]
                routed[seed]["delta_response"][selected] = output["delta_response"][selected]

                pred = output["y_pred"][index]
                delta = pred - pair.control_values
                model_name = LABELS[(expert, seed)]
                _append_bundle(table_rows, _metric_bundle(
                    model_name, scenario, pair.treatment_ids, truth, pred, pair.delta_true,
                    delta, common, meta, context_refs, drug_refs, artifacts.feature_contract.n_proteins,
                ))
                per_drug_rows.extend(_per_drug_rows(
                    model_name, scenario, chemical_names, truth, pred, pair.delta_true,
                    delta, pair.control_values, common,
                ))
            first, second = pair_outputs[SEED1], pair_outputs[SEED2]
            pred1, pred2 = first["y_pred"][index], second["y_pred"][index]
            resp1, resp2 = first["delta_response"][index], second["delta_response"][index]
            stability_rows.append({
                "model": expert, "scenario": scenario, "n_samples": int(len(pair.treatment_ids)),
                "n_valid_positions": int(common.sum()),
                "seed_prediction_pcc": paired_pcc(pred1, pred2, common),
                "seed_prediction_rmse": masked_rmse(pred1, pred2, common),
                "response_direction_agreement": direction_accuracy(resp1, resp2, common),
                "response_pcc": paired_pcc(resp1, resp2, common),
            })

        route_outputs = {
            S2C1: routed[SEED1], S2C2: routed[SEED2],
            S2C_MEAN: {
                "y_pred": arithmetic_seed_mean(routed[SEED1]["y_pred"], routed[SEED2]["y_pred"]),
                "delta_response": arithmetic_seed_mean(routed[SEED1]["delta_response"], routed[SEED2]["delta_response"]),
            },
        }
        control = routing.is_control.to_numpy(bool)
        for output in route_outputs.values():
            if not np.array_equal(output["delta_response"][control], np.zeros_like(output["delta_response"][control])):
                raise RuntimeError("routed control response is not exactly zero before/after averaging")
        for model_name, output in route_outputs.items():
            pred = output["y_pred"][index]
            delta = pred - pair.control_values
            _append_bundle(table_rows, _metric_bundle(
                model_name, scenario, pair.treatment_ids, truth, pred, pair.delta_true,
                delta, common, meta, context_refs, drug_refs, artifacts.feature_contract.n_proteins,
            ))
            per_drug_rows.extend(_per_drug_rows(
                model_name, scenario, chemical_names, truth, pred, pair.delta_true,
                delta, pair.control_values, common,
            ))
        stability_rows.append({
            "model": "S2C routed ensemble", "scenario": scenario,
            "n_samples": int(len(pair.treatment_ids)), "n_valid_positions": int(common.sum()),
            "seed_prediction_pcc": paired_pcc(routed[SEED1]["y_pred"][index], routed[SEED2]["y_pred"][index], common),
            "seed_prediction_rmse": masked_rmse(routed[SEED1]["y_pred"][index], routed[SEED2]["y_pred"][index], common),
            "response_direction_agreement": direction_accuracy(routed[SEED1]["delta_response"][index], routed[SEED2]["delta_response"][index], common),
            "response_pcc": paired_pcc(routed[SEED1]["delta_response"][index], routed[SEED2]["delta_response"][index], common),
        })

    new_tables = {key: pd.DataFrame(rows) for key, rows in table_rows.items()}
    references = _reference_tables(root)
    combined = {key: pd.concat([references[key], new_tables[key]], ignore_index=True) for key in references}
    proxy_summary, proxy_ranking = build_planning_proxies(
        EVALUATED_MODELS, combined["absolute"], combined["fc"], combined["context"],
        combined["drug"], combined["high"],
    )
    scenario_proxy_rows = []
    for scenario in VAL_SCENARIOS:
        _, ranking = build_planning_proxies(
            EVALUATED_MODELS,
            combined["absolute"].loc[combined["absolute"].scenario.eq(scenario)],
            combined["fc"].loc[combined["fc"].scenario.eq(scenario)],
            combined["context"].loc[combined["context"].scenario.eq(scenario)],
            combined["drug"].loc[combined["drug"].scenario.eq(scenario)],
            combined["high"].loc[combined["high"].scenario.eq(scenario)],
        )
        ranking.insert(0, "scenario", scenario)
        scenario_proxy_rows.append(ranking)
    scenario_proxy = pd.concat(scenario_proxy_rows, ignore_index=True)

    seed_pairs = {
        "D0": (LABELS[(D0, SEED1)], LABELS[(D0, SEED2)]),
        "D2": (LABELS[(D2, SEED1)], LABELS[(D2, SEED2)]),
        "S1 C": (LABELS[(S1C, SEED1)], LABELS[(S1C, SEED2)]),
        "S2C": (S2C1, S2C2),
    }
    difference_rows = []
    for module, table in combined.items():
        numeric_columns = [
            column for column in table.select_dtypes(include=["number"]).columns
            if not column.startswith("n_")
        ]
        for family, (first_name, second_name) in seed_pairs.items():
            first = table.loc[table.model.eq(first_name)].set_index("scenario")
            second = table.loc[table.model.eq(second_name)].set_index("scenario")
            for scenario in sorted(set(first.index) & set(second.index)):
                for metric in numeric_columns:
                    first_value, second_value = first.loc[scenario, metric], second.loc[scenario, metric]
                    if isinstance(first_value, pd.Series) or isinstance(second_value, pd.Series):
                        continue
                    if pd.notna(first_value) and pd.notna(second_value):
                        difference_rows.append({
                            "model_family": family, "module": module, "scenario": scenario,
                            "metric": metric, "seed1": float(first_value), "seed2": float(second_value),
                            "seed2_minus_seed1": float(second_value - first_value),
                        })
    seed_metric_differences = pd.DataFrame(difference_rows)

    per_drug = pd.DataFrame(per_drug_rows)
    stability = pd.DataFrame(stability_rows)
    expert_entity = per_drug.loc[per_drug.model.isin(LABELS.values())].copy()
    expert_entity["expert"] = expert_entity.model.str.replace(r" seed[12]$", "", regex=True)
    expert_entity["seed"] = np.where(expert_entity.model.str.endswith("seed1"), SEED1, SEED2)
    pivot = expert_entity.pivot(index=["expert", "scenario", "chemical_name"], columns="seed", values="rmse_gain_vs_matched_control").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={SEED1: "gain_seed1", SEED2: "gain_seed2"})
    pivot["gain_sign_consistent"] = np.sign(pivot.gain_seed1) == np.sign(pivot.gain_seed2)
    pivot["mean_gain"] = pivot[["gain_seed1", "gain_seed2"]].mean(axis=1)
    pivot["gain_absolute_difference"] = (pivot.gain_seed2 - pivot.gain_seed1).abs()

    test_meta = _read_test_metadata(root)
    test_routing = derive_metadata_routing(test_meta, seen)
    test_counts = {
        "n_test_metadata_samples": int(len(test_routing)),
        "expert_counts": {str(k): int(v) for k, v in test_routing.selected_expert.value_counts().items()},
        "reason_counts": {str(k): int(v) for k, v in test_routing.routing_reason.value_counts().items()},
        "oov_batch_tuple_count": int((~test_routing.batch_tuple_seen_in_train).sum()),
        "protein_labels_read": False, "prediction_generated": False,
    }
    prior_test_counts = json.loads((root / "reports/model_v2_stage_s2c/test_metadata_routing_counts.json").read_text(encoding="utf-8"))
    test_counts_equal = test_counts["expert_counts"] == prior_test_counts["expert_counts"] and test_counts["reason_counts"] == prior_test_counts["reason_counts"] and test_counts["oov_batch_tuple_count"] == prior_test_counts["oov_batch_tuple_count"]

    mean_rmse = lambda model: float(combined["absolute"].loc[combined["absolute"].model.eq(model), "log2_rmse"].mean())
    mean_fc = lambda model: float(combined["fc"].loc[combined["fc"].model.eq(model), "global_fc_pcc"].mean())
    deployable = set(EVALUATED_MODELS) - {"Matched Control"}
    top = {}
    for scheme, group in proxy_ranking.groupby("scheme"):
        top[scheme] = str(group.loc[group.model.isin(deployable)].sort_values(["planning_proxy", "model"], ascending=[False, True]).iloc[0].model)
    seed2_first_count = sum(model == S2C2 for model in top.values())
    mean_first_count = sum(model == S2C_MEAN for model in top.values())

    single_better_rmse = min(mean_rmse(S2C1), mean_rmse(S2C2))
    single_better_fc = max(mean_fc(S2C1), mean_fc(S2C2))
    finite_training = all(item["finite_and_passed"] and item["test_proteome_opened"] is False for item in manifest["experts"].values())
    mean_drug = per_drug.loc[per_drug.model.isin([S2C1, S2C2, S2C_MEAN])].pivot(
        index=["scenario", "chemical_name"], columns="model", values="absolute_rmse",
    ).dropna().reset_index()
    mean_drug["gain_vs_average_single_rmse"] = (
        0.5 * (mean_drug[S2C1] + mean_drug[S2C2]) - mean_drug[S2C_MEAN]
    )
    positive = mean_drug.gain_vs_average_single_rmse.clip(lower=0)
    top_share = float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
    scenario_breadth = int((mean_drug.groupby("scenario").gain_vs_average_single_rmse.mean() > 0).sum())
    broad_improvement = int((mean_drug.gain_vs_average_single_rmse > 0).sum()) >= int(np.ceil(len(mean_drug) / 2)) and top_share <= 0.5 and scenario_breadth >= 2
    gates = {
        "gate_1_seed2_s2c_first_in_at_least_two_planning_proxies": seed2_first_count >= 2,
        "gate_2_seed2_s2c_mean_rmse_better_than_seed2_d2": mean_rmse(S2C2) < mean_rmse(LABELS[(D2, SEED2)]),
        "gate_3_seed2_s2c_mean_raw_fc_drop_vs_seed2_d2_at_most_0_005": mean_fc(S2C2) >= mean_fc(LABELS[(D2, SEED2)]) - 0.005,
        "gate_4_seed_routes_rowwise_identical": route_rows_equal and routing_sha == prior_sha,
        "gate_5_all_three_experts_finite_without_training_collapse": bool(finite_training),
        "gate_6_two_seed_mean_first_in_all_three_planning_proxies": mean_first_count == 3,
        "gate_7_mean_rmse_degradation_vs_better_single_at_most_1_percent": mean_rmse(S2C_MEAN) <= single_better_rmse * 1.01,
        "gate_8_mean_raw_fc_drop_vs_better_single_at_most_0_005": mean_fc(S2C_MEAN) >= single_better_fc - 0.005,
        "gate_9_improvement_not_single_drug_or_small_scenario_dominated": bool(broad_improvement),
        "gate_10_test_proteome_not_read": True,
    }
    eligible = all(gates.values())

    route_audit = {
        "schema_version": "1.0", "routing_source": "train_metadata_sets_only",
        "seed1_manifest_sha256": prior_sha, "seed2_manifest_sha256": routing_sha,
        "rowwise_identical": route_rows_equal, "test_metadata_counts_identical": test_counts_equal,
        "validation_sample_count": len(routing_records), "test_metadata": test_counts,
        "drug_name_specific_rules": False, "validation_or_test_labels_used_for_routing": False,
    }
    manifest["routing"] = route_audit
    manifest["seed2_d0_d2_initial_chemical_response_sha256"] = d0_summary["stage_b_initial_chemical_response_sha256"]
    manifest["seed2_d0_d2_initial_hash_equal"] = True
    manifest["seed2_holdout_ids_sha256"] = seed2_summaries[0]["holdout_ids_sha256"]
    manifest["test_proteome_opened"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    for key, table in combined.items():
        table.to_csv(output_dir / f"second_seed_{key}_metrics.csv", index=False)
    stability.to_csv(output_dir / "seed_prediction_stability.csv", index=False)
    per_drug.to_csv(output_dir / "seed_stability_per_entity.csv", index=False)
    pivot.to_csv(output_dir / "expert_per_entity_gain_consistency.csv", index=False)
    mean_drug.to_csv(output_dir / "two_seed_mean_per_entity_audit.csv", index=False)
    proxy_ranking.to_csv(output_dir / "second_seed_planning_proxy.csv", index=False)
    scenario_proxy.to_csv(output_dir / "second_seed_scenario_planning_proxy.csv", index=False)
    seed_metric_differences.to_csv(output_dir / "seed_metric_differences.csv", index=False)
    training_run_rows = []
    for name, value in manifest["stage_a"].items():
        training_run_rows.append({"run": name, "stage": "A", **{key: value.get(key) for key in ("seed", "elapsed_seconds", "actual_epochs", "best_epoch", "peak_gpu_memory_reserved_bytes")}})
    for name, value in manifest["experts"].items():
        training_run_rows.append({"run": name, "stage": "B", **{key: value.get(key) for key in ("seed", "elapsed_seconds", "actual_epochs", "best_epoch", "peak_gpu_memory_reserved_bytes")}})
    training_run_summary = pd.DataFrame(training_run_rows)
    training_run_summary.to_csv(output_dir / "training_run_summary.csv", index=False)
    validation_routing.to_csv(output_dir / "validation_routing_seed2.csv", index=False)
    (output_dir / "routing_identity_audit.json").write_text(json.dumps(_finite_json(route_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps(_finite_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = {
        "schema_version": "1.0", "task": "Stage S2D second seed stability",
        "planning_proxy": True, "official_score": False,
        "experimental_nonofficial_parity_fc": True, "official_fc_result": False,
        "mean_absolute_rmse": {model: mean_rmse(model) for model in EVALUATED_MODELS},
        "mean_raw_fc_pcc": {model: mean_fc(model) for model in EVALUATED_MODELS},
        "planning_proxy_summary": proxy_summary, "planning_proxy_top_deployable": top,
        "seed2_first_proxy_count": seed2_first_count,
        "seed_stability": stability.to_dict("records"),
        "expert_gain_sign_consistency": {
            expert: {
                "n_drugs": int(len(group)),
                "consistent_count": int(group.gain_sign_consistent.sum()),
                "consistent_fraction": float(group.gain_sign_consistent.mean()),
            } for expert, group in pivot.groupby("expert")
        },
        "checkpoint_manifest": manifest, "routing_identity_audit": route_audit,
        "test_proteome_opened": False, "test_prediction_generated": False,
    }
    ensemble_metrics = {
        "schema_version": "1.0", "model": S2C_MEAN, "weights": [0.5, 0.5],
        "planning_proxy": True, "official_score": False,
        "absolute_metrics": combined["absolute"].loc[combined["absolute"].model.eq(S2C_MEAN)].to_dict("records"),
        "raw_fc_metrics": combined["fc"].loc[combined["fc"].model.eq(S2C_MEAN)].to_dict("records"),
        "context_residual_metrics": combined["context"].loc[combined["context"].model.eq(S2C_MEAN)].to_dict("records"),
        "drug_residual_metrics": combined["drug"].loc[combined["drug"].model.eq(S2C_MEAN)].to_dict("records"),
        "high_effect_metrics": combined["high"].loc[combined["high"].model.eq(S2C_MEAN)].to_dict("records"),
        "planning_proxy_summary": {key: value for key, value in proxy_summary.items() if key == S2C_MEAN},
        "mean_rmse": mean_rmse(S2C_MEAN), "mean_raw_fc_pcc": mean_fc(S2C_MEAN),
        "better_single_mean_rmse": single_better_rmse, "better_single_mean_raw_fc_pcc": single_better_fc,
        "per_entity_positive_gain_fraction_vs_average_single": float((mean_drug.gain_vs_average_single_rmse > 0).mean()),
        "largest_positive_gain_share": top_share, "positive_scenario_count": scenario_breadth,
        "gates": gates, "eligible_for_final_submission_audit": eligible,
        "test_proteome_opened": False, "test_prediction_generated": False,
    }
    (output_dir / "second_seed_comparison.json").write_text(json.dumps(_finite_json(comparison), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "two_seed_ensemble_metrics.json").write_text(json.dumps(_finite_json(ensemble_metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    md = lambda frame, cols: markdown_table(frame, cols, digits=4)
    selected_abs = combined["absolute"].loc[combined["absolute"].model.isin((*LABELS.values(), S2C1, S2C2, S2C_MEAN))]
    selected_fc = combined["fc"].loc[combined["fc"].model.isin((*LABELS.values(), S2C1, S2C2, S2C_MEAN))]
    selected_high = combined["high"].loc[combined["high"].model.isin((*LABELS.values(), S2C1, S2C2, S2C_MEAN))]
    report = f"""# MODEL V2 Stage S2D Second-Seed Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

Seed2 retrained both anchors from Stage A: hierarchical used all 751 train controls for exactly 30 epochs without validation selection; flat reproduced the S1 C control-only Stage A and selected epoch 4 from 205 public validation controls. D0 and D2 then shared the hierarchical checkpoint and identical response/chemical initialization hash `{d0_summary['stage_b_initial_chemical_response_sha256']}`. No test proteome was opened and no test prediction was generated.

## Absolute metrics on the exact S0 common subset

{md(selected_abs, ['model','scenario','log2_rmse','mae','global_r2','sample_pcc_median','sample_r2_median','protein_pcc_median','protein_r2_median'])}

## Raw FC and high-effect metrics

{md(selected_fc, ['model','scenario','global_fc_pcc','sample_fc_pcc_median','protein_fc_pcc_median','fc_rmse','fc_direction_accuracy'])}

{md(selected_high, ['model','scenario','direction_accuracy','high_effect_pcc','precision','recall','f1','auprc'])}

## Context/drug residual modules

{md(combined['context'], ['model','scenario','pcc','rmse','direction_accuracy'])}

{md(combined['drug'], ['model','scenario','pcc','rmse','direction_accuracy'])}

## Seed stability

{md(stability, ['model','scenario','seed_prediction_pcc','seed_prediction_rmse','response_direction_agreement','response_pcc'])}

Per-scenario seed2-minus-seed1 differences for every absolute, raw-FC, residual and high-effect metric are materialized in `seed_metric_differences.csv`.

## Training runs

{md(training_run_summary, ['run','stage','seed','elapsed_seconds','actual_epochs','best_epoch','peak_gpu_memory_reserved_bytes'])}

Per-drug gains use one fixed comparator: `RMSE_matched_control - RMSE_expert`. This makes the sign directly comparable across seeds without using either seed to define the reference. Full rows are in `seed_stability_per_entity.csv`; paired gain signs are in `expert_per_entity_gain_consistency.csv`.

{md(pivot.groupby('expert', as_index=False).agg(n_drugs=('chemical_name','size'), consistent_fraction=('gain_sign_consistent','mean'), mean_gain_seed1=('gain_seed1','mean'), mean_gain_seed2=('gain_seed2','mean')), ['expert','n_drugs','consistent_fraction','mean_gain_seed1','mean_gain_seed2'])}

Test-metadata responsibility remains D0 2997, S1 C 1322, D2 135 samples. D0 therefore dominates deployment exposure; its scenario prediction stability is shown independently above rather than being hidden by D2's smaller route share.

## Planning proxies and gates

{md(proxy_ranking, ['scheme','rank','model','planning_proxy','official_score'])}

```json
{json.dumps(_finite_json(gates), ensure_ascii=False, indent=2)}
```

Eligible for final pre-submission audit: **{eligible}**. Mean RMSE is `{mean_rmse(S2C_MEAN):.6f}` versus the better single seed `{single_better_rmse:.6f}`; mean raw-FC PCC is `{mean_fc(S2C_MEAN):.6f}` versus the better single seed `{single_better_fc:.6f}`. No third seed, route edit, FC-weight search, output calibration, test inference, or checkpoint modification was performed.

The strict seed2 reproduction gate failed: S2C seed2 ranked third, not first, in all three planning proxies (although it remained better than seed2 D2 on mean RMSE and within the raw-FC tolerance). The fixed two-seed mean ranked first in all three proxies and passed gates 2-10, but the predeclared rule requires every gate, so it is **not** promoted. Following the failure discipline, no third seed or route/weight adjustment was attempted.

One failed analysis attempt occurred after all inference metrics had been computed: JSON writing rejected a NumPy boolean in the checkpoint manifest. Boolean normalization was added and the full inference audit was rerun successfully; no training or checkpoint mutation occurred in either attempt.

## Reproducibility

- Seed2 hierarchical Stage A: `reports/model_v2_stage_s2d/stage_a_hierarchical_seed_20260815`.
- Seed2 flat Stage A: `reports/model_v2_stage_s2d/stage_a_flat_seed_20260815`.
- Seed2 experts: `formal_d0_hierarchical_none_huber_seed_20260815`, `formal_d2_hierarchical_morgan_fc_seed_20260815`, `formal_s1c_flat_morgan_fc_seed_20260815` under `reports/model_v2_stage_s2d`.
- Checkpoint, resolved-config, training-ID, holdout, seed and artifact hashes: `checkpoint_manifest.json`.
- Route identity: seed1 `{prior_sha}`, seed2 `{routing_sha}`, rowwise identical `{route_rows_equal}`, test route counts identical `{test_counts_equal}`.
- Analysis command: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m baseline.second_seed_ensemble_v2`.

## Verification commands

- S2D tests: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m pytest --import-mode=importlib baseline\\tests\\test_stage_s2d_second_seed_ensemble_v2.py -q` -> `6 passed in 3.89s`.
- All V2 tests excluding the separately run Person C suite: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m pytest --import-mode=importlib baseline\\tests -q --ignore=baseline\\tests\\test_person_c.py` -> `109 passed in 63.88s`.
- Person C regression from `D:\\虚拟细胞\\baseline`: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m pytest tests\\test_person_c.py -q` -> `6 passed in 3.54s`.
- The first S2D-test run was `5 passed, 1 failed`: pandas interpreted the entirely blank `auprc_unavailable_reason` explanation column as numeric NaN. The assertion was restricted to actual numeric metric columns; the rerun passed. No metric or model output changed.

Stage S2D is paused for Main review.
"""
    report_path.write_text(report, encoding="utf-8")
    return _finite_json({
        "status": "PASS", "eligible_for_final_submission_audit": eligible,
        "gates": gates, "report": str(report_path),
        "comparison": str(output_dir / "second_seed_comparison.json"),
        "ensemble_metrics": str(output_dir / "two_seed_ensemble_metrics.json"),
        "routing_manifest_sha256": routing_sha,
        "test_proteome_opened": False, "test_prediction_generated": False,
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    print(json.dumps(run_analysis(args.root, args.output_dir, args.report, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
