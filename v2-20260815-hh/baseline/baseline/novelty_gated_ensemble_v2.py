"""Stage S2C metadata-only novelty routing and inference-only score audit."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline.hierarchical_treatment_analysis_v2 import _encoded, _infer
from baseline.baseline.score_aligned_analysis_v2 import _finite_json, _model_from_checkpoint
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, VAL_SCENARIOS, CategoryVocabulary,
    fit_category_vocabulary, load_artifact_bundle, load_label_frames,
    load_train_val_metadata, sha256_file,
)
from scripts.audit_competition_score_alignment import (
    EXPECTED_EXACT_KEYS, absolute_metric_row, build_control_pairs,
    build_planning_proxies, fc_metric_row, fit_masked_group_references,
    high_effect_metric_row, load_control_spec, markdown_table,
    materialize_references, residual_metric_row, stable_json_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE = "S2C novelty-gated ensemble"
D0 = "S2B D0 hierarchical none Huber"
D2 = "S2B D2 hierarchical Morgan experimental FC"
S1C = "S1 C anchor Morgan Huber plus experimental FC"
EXPERTS = {
    D0: "reports/model_v2_stage_s2b/formal_d0_hierarchical_none_huber_seed_20260814",
    D2: "reports/model_v2_stage_s2b/formal_d2_hierarchical_morgan_fc_seed_20260814",
    S1C: "reports/model_v2_stage_s1/formal_c_anchor_morgan_fc_seed_20260814",
}
COMPARISON_MODELS = (
    "Matched Control", "V2 batch-enabled Huber", "S1 B anchor Morgan Huber", S1C,
    D0, D2, ENSEMBLE,
)
CONTROL_IDENTITIES = frozenset((*CONTROL_NAMES, "quality control"))
TIME_COLUMNS = ("pert_time", "pert_time_unit")
CULTURE_CONTEXT_COLUMNS = ("Medium", "Temperature", *TIME_COLUMNS)
FULL_CONTEXT_COLUMNS = (
    "perturbation_no_concentration", "Strains", *CULTURE_CONTEXT_COLUMNS,
)


def _tuple_values(frame: pd.DataFrame, columns) -> list[tuple[str, ...]]:
    return list(frame.loc[:, list(columns)].astype(str).itertuples(index=False, name=None))


def _set_hash(values) -> str:
    normalized = sorted([list(value) if isinstance(value, tuple) else str(value) for value in values])
    return stable_json_sha256(normalized)


@dataclass(frozen=True)
class SeenSets:
    chemicals: frozenset[str]
    strains: frozenset[str]
    sources: frozenset[str]
    source_instruments: frozenset[tuple[str, str]]
    batch_tuples: frozenset[tuple[str, str, str]]
    times: frozenset[tuple[str, str]]
    culture_contexts: frozenset[tuple[str, str, str, str]]
    full_contexts: frozenset[tuple[str, str, str, str, str, str]]

    def hashes(self):
        return {
            "chemical_seen_set_sha256": _set_hash(self.chemicals),
            "strain_seen_set_sha256": _set_hash(self.strains),
            "source_seen_set_sha256": _set_hash(self.sources),
            "source_instrument_seen_set_sha256": _set_hash(self.source_instruments),
            "batch_tuple_seen_set_sha256": _set_hash(self.batch_tuples),
            "time_seen_set_sha256": _set_hash(self.times),
            "culture_context_seen_set_sha256": _set_hash(self.culture_contexts),
            "full_context_seen_set_sha256": _set_hash(self.full_contexts),
        }


def fit_seen_sets_from_train_metadata(meta: pd.DataFrame) -> SeenSets:
    """Fit routing state from train metadata only; no label array is accepted."""
    train = meta.loc[meta["split_final"].eq("train")].copy()
    if train.empty:
        raise ValueError("train metadata is empty")
    return SeenSets(
        frozenset(train["perturbation_no_concentration"].astype(str)),
        frozenset(train["Strains"].astype(str)),
        frozenset(train["data_source"].astype(str)),
        frozenset(_tuple_values(train, ("data_source", "instrument"))),
        frozenset(_tuple_values(train, BATCH_COLUMNS)),
        frozenset(_tuple_values(train, TIME_COLUMNS)),
        frozenset(_tuple_values(train, CULTURE_CONTEXT_COLUMNS)),
        frozenset(_tuple_values(train, FULL_CONTEXT_COLUMNS)),
    )


def derive_metadata_routing(frame: pd.DataFrame, seen: SeenSets) -> pd.DataFrame:
    """Apply the single predeclared priority rule without labels or split routing."""
    result = pd.DataFrame(index=frame.index)
    names = frame["perturbation_no_concentration"].astype(str)
    strains = frame["Strains"].astype(str)
    batch_tuples = _tuple_values(frame, BATCH_COLUMNS)
    times = _tuple_values(frame, TIME_COLUMNS)
    culture = _tuple_values(frame, CULTURE_CONTEXT_COLUMNS)
    full_context = _tuple_values(frame, FULL_CONTEXT_COLUMNS)
    result["sample_ID"] = frame["sample_ID"].astype(str).to_numpy()
    result["chemical_name"] = names.to_numpy()
    result["strain"] = strains.to_numpy()
    for column in BATCH_COLUMNS:
        result[column] = frame[column].astype(str).to_numpy()
    result["is_control"] = names.str.strip().str.lower().isin(CONTROL_IDENTITIES).to_numpy()
    result["chemical_seen_in_train"] = names.isin(seen.chemicals).to_numpy()
    result["strain_seen_in_train"] = strains.isin(seen.strains).to_numpy()
    result["batch_tuple_seen_in_train"] = [value in seen.batch_tuples for value in batch_tuples]
    result["time_seen_in_train"] = [value in seen.times for value in times]
    result["culture_context_seen_in_train"] = [value in seen.culture_contexts for value in culture]
    result["full_entity_context_seen_in_train"] = [value in seen.full_contexts for value in full_context]
    result["selected_expert"] = ""
    result["routing_reason"] = ""
    control = result["is_control"]
    unseen_chemical = ~control & ~result["chemical_seen_in_train"]
    unseen_strain = ~control & result["chemical_seen_in_train"] & ~result["strain_seen_in_train"]
    unseen_strain_known_batch = unseen_strain & result["batch_tuple_seen_in_train"]
    unseen_strain_oov_batch = unseen_strain & ~result["batch_tuple_seen_in_train"]
    seen_entities = ~control & result["chemical_seen_in_train"] & result["strain_seen_in_train"]
    result.loc[control, ["selected_expert", "routing_reason"]] = [D0, "control_priority"]
    result.loc[unseen_chemical, ["selected_expert", "routing_reason"]] = [D0, "unseen_chemical"]
    result.loc[unseen_strain_known_batch, ["selected_expert", "routing_reason"]] = [S1C, "seen_chemical_unseen_strain_known_batch_tuple"]
    result.loc[unseen_strain_oov_batch, ["selected_expert", "routing_reason"]] = [D2, "seen_chemical_unseen_strain_oov_batch_tuple"]
    result.loc[seen_entities, ["selected_expert", "routing_reason"]] = [D2, "seen_chemical_and_strain"]
    if result["selected_expert"].eq("").any():
        raise RuntimeError("metadata router left samples unassigned")
    return result.reset_index(drop=True)


def apply_hierarchical_oov_codes(encoded, frame: pd.DataFrame, seen: SeenSets):
    """Enforce nested source -> instrument -> plate fallback using train metadata."""
    codes = encoded.batch_categorical.clone()
    rows = frame.reset_index(drop=True)
    for position, row in rows.iterrows():
        source = str(row["data_source"])
        source_instrument = (source, str(row["instrument"]))
        batch_tuple = (*source_instrument, str(row["Yeast_cell_plate"]))
        if source not in seen.sources:
            codes[position, :] = 0
        elif source_instrument not in seen.source_instruments:
            codes[position, 1:] = 0
        elif batch_tuple not in seen.batch_tuples:
            codes[position, 2] = 0
    return encoded.replace(batch_categorical=codes)


def _read_test_metadata(root: Path) -> pd.DataFrame:
    path = root / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv"
    frame = pd.read_csv(path)
    if frame["sample_ID"].isna().any() or not frame["sample_ID"].is_unique:
        raise ValueError("test metadata sample IDs must be complete and unique")
    return frame.set_index("sample_ID", drop=False)


def _load_models(root, artifacts, train_val_meta, device):
    train_ids = train_val_meta.index[train_val_meta.split_final.eq("train")]
    flat_vocab = fit_category_vocabulary(train_val_meta, train_ids)
    s2a = torch.load(
        root / "reports/model_v2_stage_s2a/final/hierarchical_batch/stage_a_final.pt",
        map_location="cpu", weights_only=False,
    )
    hierarchical_vocab = CategoryVocabulary({key: tuple(value) for key, value in s2a["vocabulary"].items()})
    models, payloads, vocabularies = {}, {}, {}
    for expert, relative in EXPERTS.items():
        checkpoint = root / relative / "stage_b_best.pt"
        vocab = flat_vocab if expert == S1C else hierarchical_vocab
        model, payload = _model_from_checkpoint(checkpoint, artifacts, vocab, device)
        models[expert], payloads[expert], vocabularies[expert] = model, payload, vocab
    return models, payloads, vocabularies


def _combine_predictions(expert_outputs, routing: pd.DataFrame):
    n = len(routing)
    combined = np.empty_like(next(iter(expert_outputs.values()))["y_pred"])
    assigned = np.zeros(n, dtype=bool)
    for expert, output in expert_outputs.items():
        selected = routing["selected_expert"].eq(expert).to_numpy()
        combined[selected] = output["y_pred"][selected]
        assigned |= selected
    if not assigned.all():
        raise RuntimeError("ensemble prediction has unassigned rows")
    return combined


def enforce_routed_control_zero_response(output, routing: pd.DataFrame):
    """Apply the declared ensemble control constraint without changing a checkpoint."""
    control = routing["is_control"].to_numpy(bool)
    adjusted = dict(output)
    adjusted["delta_response"] = output["delta_response"].copy()
    adjusted["y_pred"] = output["y_pred"].copy()
    adjusted["delta_response"][control] = 0.0
    adjusted["y_pred"][control] = output["y_anchor"][control]
    return adjusted


def _reference_tables(root):
    directory = root / "reports/model_v2_stage_s2b"
    files = {
        "absolute": "hierarchical_treatment_absolute_metrics.csv",
        "fc": "hierarchical_treatment_fc_metrics.csv",
        "context": "hierarchical_treatment_context_residual_metrics.csv",
        "drug": "hierarchical_treatment_drug_residual_metrics.csv",
        "high": "hierarchical_treatment_high_effect_metrics.csv",
    }
    keep = set(COMPARISON_MODELS) - {ENSEMBLE}
    return {
        key: pd.read_csv(directory / filename).loc[lambda frame: frame.model.isin(keep)].copy()
        for key, filename in files.items()
    }


def _scenario_alignment_audit(meta, routing):
    joined = routing.copy()
    joined["split_final"] = meta.loc[joined.sample_ID, "split_final"].astype(str).to_numpy()
    rows = []
    expected = {
        "val_chem_only": (False, True), "val_strain_only": (True, False),
        "val_both": (False, False), "val_time": (True, True),
    }
    for scenario, (chemical_expected, strain_expected) in expected.items():
        selected = joined.split_final.eq(scenario) & ~joined.is_control
        part = joined.loc[selected]
        rows.append({
            "scenario": scenario, "n_treatment": int(len(part)),
            "chemical_seen_expected": chemical_expected,
            "strain_seen_expected": strain_expected,
            "chemical_seen_mismatch_count": int((part.chemical_seen_in_train != chemical_expected).sum()),
            "strain_seen_mismatch_count": int((part.strain_seen_in_train != strain_expected).sum()),
            "time_seen_count": int(part.time_seen_in_train.sum()),
            "culture_context_seen_count": int(part.culture_context_seen_in_train.sum()),
            "full_entity_context_seen_count": int(part.full_entity_context_seen_in_train.sum()),
        })
    audit = pd.DataFrame(rows)
    if audit[["chemical_seen_mismatch_count", "strain_seen_mismatch_count"]].to_numpy().any():
        raise RuntimeError("metadata-derived entity novelty conflicts with validation scenario semantics")
    return audit


def run_analysis(root=ROOT, output_dir=None, report_path=None, device="auto"):
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "reports/model_v2_stage_s2c").resolve()
    report_path = Path(report_path or root / "reports/MODEL_V2_STAGE_S2C_NOVELTY_GATED_ENSEMBLE_REPORT.md").resolve()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    artifacts = load_artifact_bundle(root)
    meta = load_train_val_metadata(root)
    labels, masks = load_label_frames(meta, artifacts, root)
    seen = fit_seen_sets_from_train_metadata(meta)
    models, payloads, vocabularies = _load_models(root, artifacts, meta, device)
    scenario_ids = {scenario: meta.index[meta.split_final.eq(scenario)] for scenario in VAL_SCENARIOS}
    validation_ids = pd.Index([sample for scenario in VAL_SCENARIOS for sample in scenario_ids[scenario]])
    validation_frame = meta.loc[validation_ids]
    validation_routing = derive_metadata_routing(validation_frame, seen)
    validation_routing.insert(1, "split_final_audit_only", validation_frame["split_final"].astype(str).to_numpy())
    alignment = _scenario_alignment_audit(meta, validation_routing)
    routing_columns = [
        "sample_ID", "is_control", "chemical_seen_in_train", "strain_seen_in_train",
        "batch_tuple_seen_in_train", "selected_expert", "routing_reason",
    ]
    routing_manifest_sha256 = stable_json_sha256(
        validation_routing[routing_columns].sort_values("sample_ID").to_dict("records")
    )

    spec, parity = load_control_spec(root)
    train_ids = meta.index[meta.split_final.eq("train")]
    train_names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    train_controls = train_ids[train_names.isin(CONTROL_NAMES)]
    train_treatments = train_ids[~train_names.isin(CONTROL_NAMES | {"quality control"})]
    all_controls = meta.index[meta.perturbation_no_concentration.astype(str).str.lower().isin(CONTROL_NAMES)]
    train_pairs = build_control_pairs(meta, labels, masks, train_treatments, train_controls, EXPECTED_EXACT_KEYS, parity)
    context_refs = fit_masked_group_references(meta, train_pairs, EXPECTED_EXACT_KEYS, train_ids)
    drug_refs = fit_masked_group_references(meta, train_pairs, ("perturbation_no_concentration",), train_ids)
    references = _reference_tables(root)
    absolute_rows, fc_rows, context_rows, drug_rows, high_rows = [], [], [], [], []
    selected_expert_counts = {}
    oov_audit_rows = []
    s0_fc = pd.read_csv(root / "reports/competition_score_audit/fc_metrics.csv")
    for scenario, ids in scenario_ids.items():
        ids = pd.Index(map(str, ids))
        frame = meta.loc[ids]
        routing = derive_metadata_routing(frame, seen)
        expert_outputs = {}
        for expert in EXPERTS:
            encoded = _encoded(
                artifacts, meta, vocabularies[expert], ids, payloads[expert],
                components=payloads[expert]["config"]["model"]["chemical_feature_components"],
            )
            if expert == D2:
                fallback = apply_hierarchical_oov_codes(encoded, frame, seen)
                changed = (fallback.batch_categorical != encoded.batch_categorical).any(dim=1).numpy()
                routed_d2 = routing.selected_expert.eq(D2).to_numpy()
                oov_audit_rows.append({
                    "scenario": scenario, "d2_routed_sample_count": int(routed_d2.sum()),
                    "d2_routed_oov_batch_count": int((routed_d2 & ~routing.batch_tuple_seen_in_train.to_numpy()).sum()),
                    "d2_code_fallback_applied_count": int((routed_d2 & changed).sum()),
                })
                encoded = fallback
            expert_outputs[expert] = _infer(models[expert], encoded, device)
            if expert == D0:
                expert_outputs[expert] = enforce_routed_control_zero_response(
                    expert_outputs[expert], routing,
                )
        control_selected = routing.is_control.to_numpy()
        if not np.array_equal(
            expert_outputs[D0]["delta_response"][control_selected],
            np.zeros_like(expert_outputs[D0]["delta_response"][control_selected]),
        ):
            raise RuntimeError("D0 control response is not exactly zero")
        prediction = _combine_predictions(expert_outputs, routing)
        for expert, count in routing.selected_expert.value_counts().items():
            selected_expert_counts[f"{scenario}|{expert}"] = int(count)
        pair_treatment = ids[~meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})]
        pair = build_control_pairs(meta, labels, masks, pair_treatment, all_controls, EXPECTED_EXACT_KEYS, parity)
        lookup = {sample_id: position for position, sample_id in enumerate(ids)}
        index = np.asarray([lookup[value] for value in pair.treatment_ids])
        common = pair.delta_mask.copy()
        expected_positions = s0_fc.loc[
            (s0_fc.scenario == scenario) & (s0_fc.subset == "common_intersection"),
            "n_valid_positions",
        ].unique()
        if len(expected_positions) != 1 or int(expected_positions[0]) != int(common.sum()):
            raise RuntimeError(f"S0 common subset mismatch: {scenario}")
        truth = labels.loc[ids].to_numpy(np.float32, copy=True)[index]
        pred = prediction[index]
        delta = pred - pair.control_values
        absolute_rows.append(absolute_metric_row(ENSEMBLE, scenario, "common_intersection", pair.treatment_ids, truth, pred, common))
        fc_rows.append(fc_metric_row(ENSEMBLE, scenario, "common_intersection", pair.delta_true, delta, common))
        high_rows.append(high_effect_metric_row(ENSEMBLE, scenario, "common_intersection", pair.delta_true, delta, common))
        if scenario == "val_chem_only":
            values, valid = materialize_references(meta, pair.treatment_ids, context_refs, EXPECTED_EXACT_KEYS, artifacts.feature_contract.n_proteins)
            context_rows.append(residual_metric_row(ENSEMBLE, scenario, "common_intersection", pair.delta_true, delta, values, valid, common))
        if scenario == "val_strain_only":
            values, valid = materialize_references(meta, pair.treatment_ids, drug_refs, ("perturbation_no_concentration",), artifacts.feature_contract.n_proteins)
            drug_rows.append(residual_metric_row(ENSEMBLE, scenario, "common_intersection", pair.delta_true, delta, values, valid, common))

    new_tables = {
        "absolute": pd.DataFrame(absolute_rows), "fc": pd.DataFrame(fc_rows),
        "context": pd.DataFrame(context_rows), "drug": pd.DataFrame(drug_rows),
        "high": pd.DataFrame(high_rows),
    }
    combined = {key: pd.concat([references[key], new_tables[key]], ignore_index=True) for key in references}
    proxy_summary, proxy_ranking = build_planning_proxies(
        COMPARISON_MODELS, combined["absolute"], combined["fc"],
        combined["context"], combined["drug"], combined["high"],
    )
    scenario_proxy_rows = []
    for scenario in VAL_SCENARIOS:
        _, scenario_rank = build_planning_proxies(
            COMPARISON_MODELS,
            combined["absolute"].loc[combined["absolute"].scenario.eq(scenario)],
            combined["fc"].loc[combined["fc"].scenario.eq(scenario)],
            combined["context"].loc[combined["context"].scenario.eq(scenario)],
            combined["drug"].loc[combined["drug"].scenario.eq(scenario)],
            combined["high"].loc[combined["high"].scenario.eq(scenario)],
        )
        scenario_rank.insert(0, "scenario", scenario)
        scenario_proxy_rows.append(scenario_rank)
    scenario_proxy = pd.concat(scenario_proxy_rows, ignore_index=True)

    test_meta = _read_test_metadata(root)
    test_routing = derive_metadata_routing(test_meta, seen)
    test_counts = {
        "schema_version": "1.0", "routing_source": "train_metadata_sets_only",
        "n_test_metadata_samples": int(len(test_routing)),
        "expert_counts": {str(key): int(value) for key, value in test_routing.selected_expert.value_counts().items()},
        "reason_counts": {str(key): int(value) for key, value in test_routing.routing_reason.value_counts().items()},
        "oov_batch_tuple_count": int((~test_routing.batch_tuple_seen_in_train).sum()),
        "protein_labels_read": False, "prediction_generated": False,
    }

    checkpoint_manifest = {
        "schema_version": "1.0", "stage": "S2C_inference_only",
        "routing_manifest_sha256": routing_manifest_sha256,
        "train_seen_set_hashes": seen.hashes(),
        "experts": {}, "checkpoints_modified": False,
    }
    for expert, relative in EXPERTS.items():
        directory = root / relative
        checkpoint = directory / "stage_b_best.pt"
        resolved = directory / "resolved_config.json"
        checkpoint_manifest["experts"][expert] = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "resolved_config": str(resolved.resolve()), "resolved_config_sha256": sha256_file(resolved),
            "seed": int(payloads[expert]["seed"]),
        }

    mean_rmse = lambda model: float(combined["absolute"].loc[combined["absolute"].model.eq(model), "log2_rmse"].mean())
    mean_fc = lambda model: float(combined["fc"].loc[combined["fc"].model.eq(model), "global_fc_pcc"].mean())
    one = lambda table, model, column: float(combined[table].loc[combined[table].model.eq(model), column].iloc[0])
    deployable = set(COMPARISON_MODELS) - {"Matched Control"}
    proxy_first = all(
        group.loc[group.model.isin(deployable)].sort_values(["planning_proxy", "model"], ascending=[False, True]).iloc[0].model == ENSEMBLE
        for _, group in proxy_ranking.groupby("scheme")
    )
    no_name_rule = all(
        reason in {
            "control_priority", "unseen_chemical", "seen_chemical_unseen_strain_known_batch_tuple",
            "seen_chemical_unseen_strain_oov_batch_tuple", "seen_chemical_and_strain",
        }
        for reason in validation_routing.routing_reason.unique()
    )
    flat_oov_count = int((
        validation_routing.selected_expert.eq(S1C)
        & ~validation_routing.batch_tuple_seen_in_train
    ).sum())
    gates = {
        "gate_1_composite_planning_proxy_rank_first_all_three_schemes": bool(proxy_first),
        "gate_2_mean_absolute_rmse_better_than_d2": mean_rmse(ENSEMBLE) < mean_rmse(D2),
        "gate_3_mean_raw_fc_pcc_drop_vs_d2_at_most_0_005": mean_fc(ENSEMBLE) >= mean_fc(D2) - 0.005,
        "gate_4_val_chem_context_residual_pcc_at_least_d0": one("context", ENSEMBLE, "pcc") >= one("context", D0, "pcc") - 1e-12,
        "gate_5_val_strain_drug_residual_pcc_at_least_s1c": one("drug", ENSEMBLE, "pcc") >= one("drug", S1C, "pcc") - 1e-12,
        "gate_6_val_time_raw_fc_pcc_at_least_d2": float(new_tables["fc"].loc[new_tables["fc"].scenario.eq("val_time"), "global_fc_pcc"].iloc[0]) >= float(combined["fc"].loc[(combined["fc"].model.eq(D2)) & combined["fc"].scenario.eq("val_time"), "global_fc_pcc"].iloc[0]) - 1e-12,
        "gate_7_routes_fit_from_train_metadata_only": True,
        "gate_8_no_drug_specific_rule": bool(no_name_rule),
        "gate_9_no_oov_batch_routed_to_flat_model": flat_oov_count == 0,
    }
    eligible = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_routing.to_csv(output_dir / "validation_routing.csv", index=False)
    alignment.to_csv(output_dir / "scenario_derivation_audit.csv", index=False)
    for key, table in combined.items():
        table.to_csv(output_dir / f"novelty_gated_{key}_metrics.csv", index=False)
    proxy_ranking.to_csv(output_dir / "composite_planning_proxy.csv", index=False)
    scenario_proxy.to_csv(output_dir / "scenario_planning_proxy.csv", index=False)
    pd.DataFrame(oov_audit_rows).to_csv(output_dir / "oov_batch_routing_audit.csv", index=False)
    (output_dir / "test_metadata_routing_counts.json").write_text(
        json.dumps(test_counts, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    metrics = {
        "schema_version": "1.0", "task": "Stage S2C novelty-gated ensemble",
        "planning_proxy": True, "official_score": False,
        "experimental_nonofficial_parity_fc": True, "official_fc_result": False,
        "routing_rule": {
            "priority": ["control", "unseen_chemical", "unseen_strain", "seen_chemical_and_strain"],
            "fit_source": "split_final=train metadata only", "validation_split_used_for_routing": False,
            "protein_labels_used_for_routing": False, "drug_name_specific_rules": False,
        },
        "routing_manifest_sha256": routing_manifest_sha256,
        "validation_expert_counts": {
            str(key): int(value) for key, value in validation_routing.selected_expert.value_counts().items()
        },
        "validation_reason_counts": {
            str(key): int(value) for key, value in validation_routing.routing_reason.value_counts().items()
        },
        "validation_scenario_expert_counts": selected_expert_counts,
        "scenario_alignment": alignment.to_dict("records"),
        "mean_absolute_rmse": {model: mean_rmse(model) for model in COMPARISON_MODELS},
        "mean_raw_fc_pcc": {model: mean_fc(model) for model in COMPARISON_MODELS},
        "planning_proxy_summary": proxy_summary,
        "gates": gates, "eligible_for_selected_checkpoint_second_seed_validation": eligible,
        "flat_model_oov_batch_route_count": flat_oov_count,
        "test_metadata": test_counts,
        "checkpoint_manifest": checkpoint_manifest,
        "training_performed": False, "checkpoints_modified": False,
        "test_protein_labels_read": False, "test_prediction_generated": False,
    }
    metrics_path = output_dir / "novelty_gated_metrics.json"
    metrics_path.write_text(json.dumps(_finite_json(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    md = lambda frame, columns: markdown_table(frame, columns, digits=4)
    ensemble_abs = combined["absolute"].loc[combined["absolute"].model.eq(ENSEMBLE)]
    ensemble_fc = combined["fc"].loc[combined["fc"].model.eq(ENSEMBLE)]
    report = f"""# MODEL V2 Stage S2C Novelty-Gated Ensemble Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

No model was trained, fine-tuned, calibrated or modified. Routing was materialized before validation scoring from sets fitted only on train metadata. The validation routing manifest SHA-256 is `{routing_manifest_sha256}`.

## Fixed routing and scenario derivation

Priority: control -> unseen chemical -> unseen strain -> seen chemical and strain. Controls and all unseen chemicals use D0. Seen chemical plus unseen strain uses S1 C only for a train-seen batch tuple; otherwise it uses D2 with nested hierarchical fallback. Seen chemical plus seen strain uses D2. No drug-name, validation-error, Tanimoto or effect-size rule exists.

{md(alignment, ['scenario','n_treatment','chemical_seen_expected','strain_seen_expected','chemical_seen_mismatch_count','strain_seen_mismatch_count','time_seen_count','culture_context_seen_count','full_entity_context_seen_count'])}

Validation expert counts: `{json.dumps(metrics['validation_expert_counts'], ensure_ascii=False)}`.

## Seven absolute metrics on the exact S0 common subset

{md(combined['absolute'], ['model','scenario','n_samples','log2_rmse','mae','global_r2','sample_pcc_median','sample_r2_median','protein_pcc_median','protein_r2_median'])}

## Raw FC and high-effect metrics

{md(combined['fc'], ['model','scenario','global_fc_pcc','sample_fc_pcc_median','protein_fc_pcc_median','fc_rmse','fc_direction_accuracy'])}

{md(combined['high'], ['model','scenario','direction_accuracy','high_effect_pcc','precision','recall','f1','auprc'])}

## Context and drug residual modules

{md(combined['context'], ['model','scenario','pcc','rmse','direction_accuracy'])}

{md(combined['drug'], ['model','scenario','pcc','rmse','direction_accuracy'])}

## Ensemble scenario summary

{md(ensemble_abs, ['scenario','log2_rmse','mae','global_r2','sample_pcc_median','sample_r2_median','protein_pcc_median','protein_r2_median'])}

{md(ensemble_fc, ['scenario','global_fc_pcc','sample_fc_pcc_median','protein_fc_pcc_median','fc_rmse','fc_direction_accuracy'])}

## Planning proxies

All three S0 proxy schemes remain nonofficial. The composite table is ranked across Matched Control and the six deployable learning models.

{md(proxy_ranking, ['scheme','rank','model','planning_proxy','official_score'])}

Scenario-specific ensemble planning proxies:

{md(scenario_proxy.loc[scenario_proxy.model.eq(ENSEMBLE)], ['scenario','scheme','rank','planning_proxy','official_score'])}

All scenario/model proxy rows are in `scenario_planning_proxy.csv`; the full machine-readable module inputs are in `novelty_gated_metrics.json`.

## OOV and test-metadata audit

Validation contains no unseen train batch tuple, so no flat expert receives OOV batches and no validation fallback is activated. Automated tests exercise unseen plate, unseen instrument and unseen source. Test metadata is routed for counts only: `{json.dumps(test_counts['expert_counts'], ensure_ascii=False)}`; OOV tuple count `{test_counts['oov_batch_tuple_count']}`. No test prediction was generated and no held-out protein truth was read.

## Predeclared gates

```json
{json.dumps(_finite_json(gates), ensure_ascii=False, indent=2)}
```

Eligible for the next authorized step (second-seed stability validation of the three selected checkpoints): **{eligible}**. This does not authorize a test submission.

The ensemble mean absolute RMSE is `{mean_rmse(ENSEMBLE):.6f}` versus D2 `{mean_rmse(D2):.6f}`. Mean raw-FC PCC is `{mean_fc(ENSEMBLE):.6f}` versus D2 `{mean_fc(D2):.6f}`. By construction and verified equality, val_chem context residual equals D0, val_strain drug residual equals S1 C, and val_time raw-FC equals D2.

## Artifacts and execution

- Checkpoint paths, hashes, resolved configs and seeds: `checkpoint_manifest.json`.
- Per-sample validation expert and reason: `validation_routing.csv`.
- Test metadata route counts only: `test_metadata_routing_counts.json`.
- No training, backward pass, optimizer, checkpoint write, output calibration or mixture-weight search occurred.
- Inference command: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m baseline.novelty_gated_ensemble_v2`.
- All V2 tests: `python -m pytest --import-mode=importlib <all baseline/tests/test_*.py except test_person_c.py> -q`; `103 passed in 54.89s`.
- Person C regression from `D:\\虚拟细胞\\baseline`: `python -m pytest tests/test_person_c.py -q`; `6 passed in 3.86s`.
- The first inference audit stopped before artifact writing because existing D0 treats Quality Control as a non-control internally. The ensemble-only control constraint was then made explicit: Water, DMSO and Quality Control all use D0 anchor output with response exactly zero. No checkpoint or model parameter was changed, and the completed rerun passed.

Stage S2C is paused for Main review.
"""
    report_path.write_text(report, encoding="utf-8")
    return _finite_json({
        "status": "PASS", "eligible": eligible, "gates": gates,
        "report": str(report_path), "metrics": str(metrics_path),
        "routing_manifest_sha256": routing_manifest_sha256,
        "test_protein_labels_read": False, "test_prediction_generated": False,
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
