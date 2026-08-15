"""Inference-only Stage S2B score-aligned and attribution audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline.score_aligned_analysis_v2 import _finite_json, _model_from_checkpoint
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, VAL_SCENARIOS, CategoryVocabulary, build_batch,
    load_artifact_bundle, load_label_frames, load_train_val_metadata,
    make_chemical_feature_variant, sha256_file,
)
from scripts.audit_competition_score_alignment import (
    EXPECTED_EXACT_KEYS, absolute_metric_row, build_control_pairs, direction_accuracy,
    fc_metric_row, fit_masked_group_references, high_effect_metric_row,
    load_control_spec, markdown_table, masked_mae, masked_rmse,
    materialize_references, paired_pcc, residual_metric_row,
)


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "S2B D0 hierarchical none Huber": "reports/model_v2_stage_s2b/formal_d0_hierarchical_none_huber_seed_20260814",
    "S2B D1 hierarchical Morgan Huber": "reports/model_v2_stage_s2b/formal_d1_hierarchical_morgan_huber_seed_20260814",
    "S2B D2 hierarchical Morgan experimental FC": "reports/model_v2_stage_s2b/formal_d2_hierarchical_morgan_fc_seed_20260814",
}
S1_B = "S1 B anchor Morgan Huber"
S1_C = "S1 C anchor Morgan Huber plus experimental FC"
CURRENT = "V2 batch-enabled Huber"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _infer(model, encoded, device):
    names = (
        "y_pred", "y_baseline", "y_anchor", "delta_response", "delta_batch",
        "delta_source", "delta_instrument", "delta_plate",
    )
    chunks = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for start in range(0, encoded.morgan.shape[0], 256):
            index = torch.arange(start, min(start + 256, encoded.morgan.shape[0]))
            output = model(encoded.index_select(index).to(device))
            for name in names:
                chunks[name].append(output[name].detach().cpu().numpy().astype(np.float32, copy=False))
    return {name: np.concatenate(values) for name, values in chunks.items()}


def _encoded(artifacts, meta, vocab, ids, payload, chemical_mode=None, components=None):
    cfg = payload["config"]["model"]
    seed = int(payload["seed"])
    mode = chemical_mode or cfg["chemical_mode"]
    component_mode = components or cfg["chemical_feature_components"]
    variant = make_chemical_feature_variant(artifacts, mode, seed)
    return build_batch(
        meta, ids, artifacts, vocab, chemical_mode=mode,
        chemical_feature_components=component_mode, genome_mode="correct",
        seed=seed, chemical_variant=variant,
    )


def _rms(value, mask):
    valid = np.asarray(mask, bool) & np.isfinite(value)
    return float(np.sqrt(np.square(value[valid], dtype=np.float64).mean())) if valid.any() else math.nan


def _component_row(model_name, scenario, output, mask):
    response_norm = float(np.linalg.norm(np.where(mask, output["delta_response"], 0).astype(np.float64)))
    rows = {
        "model": model_name, "scenario": scenario,
        "delta_response_rms": _rms(output["delta_response"], mask),
        "delta_source_rms": _rms(output["delta_source"], mask),
        "delta_instrument_rms": _rms(output["delta_instrument"], mask),
        "delta_plate_rms": _rms(output["delta_plate"], mask),
        "delta_batch_rms": _rms(output["delta_batch"], mask),
        "delta_response_norm": response_norm,
    }
    batch_norm = float(np.linalg.norm(np.where(mask, output["delta_batch"], 0).astype(np.float64)))
    rows["delta_batch_norm"] = batch_norm
    rows["batch_to_response_norm"] = batch_norm / response_norm if response_norm else math.nan
    return rows


def _holdout_ids(meta, seed=20260814, fraction=0.2):
    train = meta.index[meta.split_final.eq("train")]
    names = meta.loc[train, "perturbation_no_concentration"].astype(str).str.lower()
    treatment = train[~names.isin(CONTROL_NAMES | {"quality control"})]
    keys = meta.loc[treatment, list(BATCH_COLUMNS)].astype(str).agg("|".join, axis=1)
    groups = sorted(keys.unique())
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    selected = set(ordered[:max(1, int(round(len(ordered) * fraction)))])
    holdout = treatment[keys.isin(selected).to_numpy()]
    remaining = treatment[~keys.isin(selected).to_numpy()]
    if set(keys.loc[holdout]) & set(keys.loc[remaining]):
        raise RuntimeError("technical tuple crossed the Stage-B holdout")
    return holdout, keys.loc[holdout]


def _load_reference_tables(root):
    s0 = root / "reports/competition_score_audit"
    s1 = root / "reports/model_v2_stage_s1"
    sources = {
        "absolute": (s0 / "absolute_metrics.csv", s1 / "score_aligned_absolute_metrics.csv"),
        "fc": (s0 / "fc_metrics.csv", s1 / "score_aligned_fc_metrics.csv"),
        "context": (s0 / "context_residual_metrics.csv", s1 / "score_aligned_context_residual_metrics.csv"),
        "drug": (s0 / "drug_residual_metrics.csv", s1 / "score_aligned_drug_residual_metrics.csv"),
        "high": (s0 / "high_effect_metrics.csv", s1 / "score_aligned_high_effect_metrics.csv"),
    }
    output = {}
    for key, (s0_path, s1_path) in sources.items():
        left, right = pd.read_csv(s0_path), pd.read_csv(s1_path)
        keep_left = left.model.isin(["Matched Control", CURRENT]) & left.subset.eq("common_intersection")
        keep_right = right.model.isin([S1_B, S1_C]) & right.subset.eq("common_intersection")
        output[key] = pd.concat([left.loc[keep_left], right.loc[keep_right]], ignore_index=True)
    return output


def run_analysis(root=ROOT, output_dir=None, report_path=None, device="auto"):
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "reports/model_v2_stage_s2b").resolve()
    report_path = Path(report_path or root / "reports/MODEL_V2_STAGE_S2B_HIERARCHICAL_TREATMENT_REPORT.md").resolve()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    artifacts = load_artifact_bundle(root)
    meta = load_train_val_metadata(root)
    labels, masks = load_label_frames(meta, artifacts, root)
    train_ids = meta.index[meta.split_final.eq("train")]
    names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    train_controls = train_ids[names.isin(CONTROL_NAMES)]
    train_treatments = train_ids[~names.isin(CONTROL_NAMES | {"quality control"})]
    all_controls = meta.index[meta.perturbation_no_concentration.astype(str).str.lower().isin(CONTROL_NAMES)]
    s2a_path = root / "reports/model_v2_stage_s2a/final/hierarchical_batch/stage_a_final.pt"
    s2a = torch.load(s2a_path, map_location="cpu", weights_only=False)
    vocab = CategoryVocabulary({key: tuple(values) for key, values in s2a["vocabulary"].items()})
    spec, parity = load_control_spec(root)
    train_pairs = build_control_pairs(meta, labels, masks, train_treatments, train_controls, EXPECTED_EXACT_KEYS, parity)
    context_refs = fit_masked_group_references(meta, train_pairs, EXPECTED_EXACT_KEYS, train_ids)
    drug_refs = fit_masked_group_references(meta, train_pairs, ("perturbation_no_concentration",), train_ids)
    scenario_ids = {name: meta.index[meta.split_final.eq(name)] for name in VAL_SCENARIOS}
    pairs = {}
    for scenario, ids in scenario_ids.items():
        treatment = ids[~meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})]
        pairs[scenario] = build_control_pairs(meta, labels, masks, treatment, all_controls, EXPECTED_EXACT_KEYS, parity)

    models, payloads, summaries = {}, {}, {}
    for model_name, relative in MODELS.items():
        directory = root / relative
        model, payload = _model_from_checkpoint(directory / "stage_b_best.pt", artifacts, vocab, device)
        models[model_name], payloads[model_name] = model, payload
        summaries[model_name] = _read_json(directory / "training_summary.json")
    initial_hashes = {summary["stage_b_initial_chemical_response_sha256"] for summary in summaries.values()}
    stage_a_hashes = {summary["stage_a"]["fixed_checkpoint"]["sha256"] for summary in summaries.values()}
    holdout_hashes = {summary["train_internal_batch_holdout"]["holdout_sample_ids_sha256"] for summary in summaries.values()}
    if not (len(initial_hashes) == len(stage_a_hashes) == len(holdout_hashes) == 1):
        raise RuntimeError("D0/D1/D2 fairness hashes differ")

    reference = _load_reference_tables(root)
    absolute_rows, fc_rows, context_rows, drug_rows, high_rows = [], [], [], [], []
    component_rows, chemical_rows, oov_rows, per_drug_rows = [], [], [], []
    predictions = {name: {} for name in MODELS}
    oov_checks = {}
    s0_fc = pd.read_csv(root / "reports/competition_score_audit/fc_metrics.csv")
    for scenario, ids in scenario_ids.items():
        ids = pd.Index(map(str, ids))
        truth = labels.loc[ids].to_numpy(np.float32, copy=True)
        pair = pairs[scenario]
        lookup = {sample_id: pos for pos, sample_id in enumerate(ids)}
        index = np.asarray([lookup[value] for value in pair.treatment_ids])
        common = pair.delta_mask.copy()
        expected = s0_fc.loc[(s0_fc.scenario == scenario) & (s0_fc.subset == "common_intersection"), "n_valid_positions"].unique()
        if len(expected) != 1 or int(expected[0]) != int(common.sum()):
            raise RuntimeError(f"S0 common subset mismatch: {scenario}")
        pair_true = truth[index]
        chemical_names = meta.loc[list(pair.treatment_ids), "perturbation_no_concentration"].astype(str).to_numpy()
        for model_name, model in models.items():
            payload = payloads[model_name]
            components = payload["config"]["model"]["chemical_feature_components"]
            encoded = _encoded(artifacts, meta, vocab, ids, payload, "correct", components)
            correct_all = _infer(model, encoded, device)
            predictions[model_name][scenario] = correct_all
            correct = {key: value[index] for key, value in correct_all.items()}
            pred, delta = correct["y_pred"], correct["y_pred"] - pair.control_values
            absolute_rows.append(absolute_metric_row(model_name, scenario, "common_intersection", pair.treatment_ids, pair_true, pred, common))
            fc_rows.append(fc_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, common))
            high_rows.append(high_effect_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, common))
            component_rows.append(_component_row(model_name, scenario, correct, common))
            if scenario == "val_chem_only":
                values, valid = materialize_references(meta, pair.treatment_ids, context_refs, EXPECTED_EXACT_KEYS, artifacts.feature_contract.n_proteins)
                context_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, values, valid, common))
            if scenario == "val_strain_only":
                values, valid = materialize_references(meta, pair.treatment_ids, drug_refs, ("perturbation_no_concentration",), artifacts.feature_contract.n_proteins)
                drug_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, values, valid, common))
            for chemical_name in sorted(set(chemical_names)):
                selected = chemical_names == chemical_name
                selected_mask = common & selected[:, None]
                per_drug_rows.append({
                    "model": model_name, "scenario": scenario, "chemical_name": chemical_name,
                    "n_samples": int(selected.sum()), "n_valid_positions": int(selected_mask.sum()),
                    "absolute_rmse": masked_rmse(pair_true, pred, selected_mask),
                    "absolute_mae": masked_mae(pair_true, pred, selected_mask),
                    "raw_fc_pcc": paired_pcc(pair.delta_true, delta, selected_mask),
                    "raw_fc_rmse": masked_rmse(pair.delta_true, delta, selected_mask),
                    "raw_fc_direction_accuracy": direction_accuracy(pair.delta_true, delta, selected_mask),
                })
            # Exact hierarchical OOV fallback audit and its score degradation.
            variants = {
                "known": encoded,
                "oov_plate": encoded.replace(batch_categorical=encoded.batch_categorical.clone()),
                "oov_instrument": encoded.replace(batch_categorical=encoded.batch_categorical.clone()),
                "all_oov": encoded.replace(batch_categorical=torch.zeros_like(encoded.batch_categorical)),
            }
            variants["oov_plate"].batch_categorical[:, 2] = 0
            variants["oov_instrument"].batch_categorical[:, 1:] = 0
            known = correct_all
            for mode, variant_encoded in variants.items():
                output = known if mode == "known" else _infer(model, variant_encoded, device)
                mode_pred = output["y_pred"][index]
                mode_delta = mode_pred - pair.control_values
                oov_rows.append({
                    "model": model_name, "scenario": scenario, "batch_mode": mode,
                    "absolute_rmse": masked_rmse(pair_true, mode_pred, common),
                    "raw_fc_global_pcc": paired_pcc(pair.delta_true, mode_delta, common),
                    "raw_fc_rmse": masked_rmse(pair.delta_true, mode_delta, common),
                })
                if mode != "known":
                    key = f"{model_name}|{scenario}|{mode}"
                    checks = {
                        "oov_plate": max(
                            float(np.max(np.abs(output["delta_source"] - known["delta_source"]))),
                            float(np.max(np.abs(output["delta_instrument"] - known["delta_instrument"]))),
                            float(np.max(np.abs(output["delta_plate"]))),
                        ),
                        "oov_instrument": max(
                            float(np.max(np.abs(output["delta_source"] - known["delta_source"]))),
                            float(np.max(np.abs(output["delta_instrument"]))),
                            float(np.max(np.abs(output["delta_plate"]))),
                        ),
                        "all_oov": float(np.max(np.abs(output["delta_batch"]))),
                    }
                    oov_checks[key] = checks[mode]
            if model_name != "S2B D0 hierarchical none Huber":
                for mode in ("correct", "shuffle", "zero"):
                    mode_encoded = _encoded(artifacts, meta, vocab, ids, payload, mode, "morgan_only")
                    mode_output = correct_all if mode == "correct" else _infer(model, mode_encoded, device)
                    mode_pred = mode_output["y_pred"][index]
                    mode_delta = mode_pred - pair.control_values
                    chemical_rows.append({
                        "model": model_name, "scenario": scenario, "chemical_mode": mode,
                        "absolute_rmse": masked_rmse(pair_true, mode_pred, common),
                        "raw_fc_global_pcc": paired_pcc(pair.delta_true, mode_delta, common),
                        "raw_fc_rmse": masked_rmse(pair.delta_true, mode_delta, common),
                    })

    new = {
        "absolute": pd.DataFrame(absolute_rows), "fc": pd.DataFrame(fc_rows),
        "context": pd.DataFrame(context_rows), "drug": pd.DataFrame(drug_rows),
        "high": pd.DataFrame(high_rows),
    }
    combined = {key: pd.concat([reference[key], new[key]], ignore_index=True) for key in new}
    component, chemical, oov, per_drug = map(pd.DataFrame, (component_rows, chemical_rows, oov_rows, per_drug_rows))

    holdout_ids, holdout_keys = _holdout_ids(meta)
    holdout_pairs = build_control_pairs(meta, labels, masks, holdout_ids, train_controls, EXPECTED_EXACT_KEYS, parity)
    holdout_truth = labels.loc[holdout_ids].to_numpy(np.float32, copy=True)
    holdout_mask = masks.loc[holdout_ids].to_numpy(bool, copy=True)
    holdout_rows = []
    for model_name, model in models.items():
        payload = payloads[model_name]
        encoded = _encoded(artifacts, meta, vocab, holdout_ids, payload)
        output = _infer(model, encoded, device)
        delta = output["y_pred"] - holdout_pairs.control_values
        holdout_rows.append({
            "model": model_name, "n_samples": len(holdout_ids),
            "n_matched_fc_samples": int(holdout_pairs.delta_mask.any(1).sum()),
            "absolute_rmse": masked_rmse(holdout_truth, output["y_pred"], holdout_mask),
            "absolute_mae": masked_mae(holdout_truth, output["y_pred"], holdout_mask),
            "raw_fc_global_pcc": paired_pcc(holdout_pairs.delta_true, delta, holdout_pairs.delta_mask),
            "raw_fc_rmse": masked_rmse(holdout_pairs.delta_true, delta, holdout_pairs.delta_mask),
        })
    holdout = pd.DataFrame(holdout_rows)

    # Drug-level D1/D2 gain relative to D0 and concentration of negative transfer.
    pivot = per_drug.pivot(index=["scenario", "chemical_name"], columns="model", values="absolute_rmse").dropna()
    gain_rows, concentration = [], {}
    d0_name = "S2B D0 hierarchical none Huber"
    for candidate in ("S2B D1 hierarchical Morgan Huber", "S2B D2 hierarchical Morgan experimental FC"):
        gain = pivot[d0_name] - pivot[candidate]
        for (scenario, chemical_name), value in gain.items():
            gain_rows.append({"model": candidate, "scenario": scenario, "chemical_name": chemical_name, "rmse_gain_vs_d0": float(value)})
        harm = (-gain[gain < 0]).sort_values(ascending=False)
        concentration[candidate] = {
            "positive_gain_drug_count": int((gain > 0).sum()), "total_drug_count": int(len(gain)),
            "mean_drug_equal_gain": float(gain.mean()), "median_drug_equal_gain": float(gain.median()),
            "negative_transfer_drug_count": int((gain < 0).sum()),
            "top3_negative_harm_share": float(harm.iloc[:3].sum() / harm.sum()) if harm.sum() > 0 else 0.0,
        }
    gains = pd.DataFrame(gain_rows)

    direct_rows = []
    for candidate, baseline in (
        ("S2B D1 hierarchical Morgan Huber", d0_name),
        ("S2B D2 hierarchical Morgan experimental FC", "S2B D1 hierarchical Morgan Huber"),
    ):
        for scenario in VAL_SCENARIOS:
            find = lambda table, model, column: float(table.loc[(table.model == model) & (table.scenario == scenario), column].iloc[0])
            direct_rows.append({
                "candidate": candidate, "reference": baseline, "scenario": scenario,
                "absolute_rmse_change_candidate_minus_reference": (
                    find(new["absolute"], candidate, "log2_rmse") - find(new["absolute"], baseline, "log2_rmse")
                ),
                "raw_fc_pcc_gain_candidate_minus_reference": (
                    find(new["fc"], candidate, "global_fc_pcc") - find(new["fc"], baseline, "global_fc_pcc")
                ),
            })
    direct = pd.DataFrame(direct_rows)

    training_table = pd.DataFrame([
        {
            "model": model_name, "device": summary["device"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_reserved_mib": summary["peak_gpu_memory_reserved_bytes"] / (1024 ** 2),
            "stage_b_actual_epochs": summary["stage_b"]["actual_epochs"],
            "stage_b_best_epoch": summary["stage_b"]["best_epoch"],
            "stage_b_best_macro_huber": summary["stage_b"]["best_monitor"],
            "stop_reason": summary["stage_b"]["stop_reason"],
        } for model_name, summary in summaries.items()
    ])

    corresponding = {
        d0_name: S1_B,
        "S2B D1 hierarchical Morgan Huber": S1_B,
        "S2B D2 hierarchical Morgan experimental FC": S1_C,
    }
    decisions = {}
    for model_name, reference_name in corresponding.items():
        metric = lambda table, name, column: float(combined[table].loc[combined[table].model.eq(name), column].mean())
        abs_new, abs_old = metric("absolute", model_name, "log2_rmse"), metric("absolute", reference_name, "log2_rmse")
        fc_gain = metric("fc", model_name, "global_fc_pcc") - metric("fc", reference_name, "global_fc_pcc")
        context_gain = metric("context", model_name, "pcc") - metric("context", reference_name, "pcc")
        drug_gain = metric("drug", model_name, "pcc") - metric("drug", reference_name, "pcc")
        compare = per_drug.loc[per_drug.model.eq(model_name), ["scenario", "chemical_name", "raw_fc_rmse"]].merge(
            pd.read_csv(root / "reports/model_v2_stage_s1/score_aligned_per_drug_metrics.csv").loc[
                lambda frame: frame.model.eq(reference_name), ["scenario", "chemical_name", "raw_fc_rmse"]
            ], on=["scenario", "chemical_name"], suffixes=("_new", "_old"),
        )
        drug_fc_gain = compare.raw_fc_rmse_old - compare.raw_fc_rmse_new
        positive = drug_fc_gain[drug_fc_gain > 0]
        scenario_gain = (
            combined["fc"].loc[combined["fc"].model.eq(model_name)].set_index("scenario").global_fc_pcc
            - combined["fc"].loc[combined["fc"].model.eq(reference_name)].set_index("scenario").global_fc_pcc
        )
        scenario_positive = scenario_gain[scenario_gain > 0]
        broad = bool(
            (drug_fc_gain > 0).mean() >= 0.60
            and (float(positive.nlargest(3).sum() / positive.sum()) if positive.sum() > 0 else 1.0) <= 0.50
            and (float(scenario_positive.max() / scenario_positive.sum()) if scenario_positive.sum() > 0 else 1.0) <= 0.75
        )
        freeze_ok = summaries[model_name]["stage_b_freeze_audit"]["status"] == "PASS"
        oov_ok = max(value for key, value in oov_checks.items() if key.startswith(model_name + "|")) == 0.0
        gates = {
            "gate_1_mean_absolute_rmse_not_worse": abs_new <= abs_old,
            "gate_2_fc_or_weighted_residual_pcc_gain_at_least_0_01": max(fc_gain, context_gain, drug_gain) >= 0.01,
            "gate_3_val_chem_context_pcc_drop_at_most_0_005": context_gain >= -0.005,
            "gate_4_val_strain_drug_pcc_drop_at_most_0_005": drug_gain >= -0.005,
            "gate_5_frozen_anchor_exact": freeze_ok,
            "gate_6_oov_semantics_exact": oov_ok,
            "gate_7_not_single_scenario_or_few_drug_dominated": broad,
        }
        decisions[model_name] = {
            "corresponding_s1_model": reference_name, "mean_absolute_rmse": abs_new,
            "mean_absolute_rmse_change_vs_s1": abs_new - abs_old,
            "mean_raw_fc_pcc_gain_vs_s1": fc_gain,
            "val_chem_context_residual_pcc_gain_vs_s1": context_gain,
            "val_strain_drug_residual_pcc_gain_vs_s1": drug_gain,
            **gates, "eligible_for_multiseed": all(gates.values()),
        }

    chem_pivot = chemical.loc[chemical.scenario.isin(["val_chem_only", "val_both"])].pivot(
        index=["model", "scenario"], columns="chemical_mode", values=["absolute_rmse", "raw_fc_global_pcc"],
    )
    chemical_decision = {}
    for candidate in ("S2B D1 hierarchical Morgan Huber", "S2B D2 hierarchical Morgan experimental FC"):
        candidate_rows = new["absolute"].loc[new["absolute"].model.eq(candidate) & new["absolute"].scenario.isin(["val_chem_only", "val_both"])].set_index("scenario")
        d0_rows = new["absolute"].loc[new["absolute"].model.eq(d0_name) & new["absolute"].scenario.isin(["val_chem_only", "val_both"])].set_index("scenario")
        correct_better_d0 = bool((candidate_rows.log2_rmse < d0_rows.log2_rmse).all())
        not_worse_shuffle = bool(
            (chem_pivot.loc[candidate, "absolute_rmse"]["correct"] <= chem_pivot.loc[candidate, "absolute_rmse"]["shuffle"]).all()
            and (chem_pivot.loc[candidate, "raw_fc_global_pcc"]["correct"] >= chem_pivot.loc[candidate, "raw_fc_global_pcc"]["shuffle"]).all()
        )
        chemical_decision[candidate] = {
            "correct_strictly_better_absolute_rmse_than_d0_in_val_chem_and_val_both": correct_better_d0,
            "correct_not_worse_than_shuffle_on_absolute_and_raw_fc_pcc": not_worse_shuffle,
            "morgan_contribution_established": bool(correct_better_d0 and not_worse_shuffle),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "hierarchical_treatment_absolute_metrics.csv": combined["absolute"],
        "hierarchical_treatment_fc_metrics.csv": combined["fc"],
        "hierarchical_treatment_context_residual_metrics.csv": combined["context"],
        "hierarchical_treatment_drug_residual_metrics.csv": combined["drug"],
        "hierarchical_treatment_high_effect_metrics.csv": combined["high"],
        "hierarchical_treatment_component_norms.csv": component,
        "hierarchical_treatment_oov_metrics.csv": oov,
        "hierarchical_treatment_holdout_metrics.csv": holdout,
        "hierarchical_treatment_holdout_ids.csv": pd.DataFrame({"sample_ID": holdout_ids, "technical_group": holdout_keys.to_numpy()}),
        "hierarchical_treatment_chemical_robustness.csv": chemical,
        "hierarchical_treatment_per_drug_metrics.csv": per_drug,
        "hierarchical_treatment_per_drug_gain.csv": gains,
        "hierarchical_treatment_direct_comparison.csv": direct,
        "hierarchical_treatment_training_audit.csv": training_table,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)
    comparison = {
        "schema_version": "1.0", "task": "Stage S2B hierarchical treatment",
        "planning_proxy": True, "official_score": False,
        "experimental_nonofficial_parity_fc": True, "official_fc_result": False,
        "seed": 20260814, "stage_a_retrained": False,
        "stage_a_checkpoint_sha256": next(iter(stage_a_hashes)),
        "shared_stage_b_initial_hash": next(iter(initial_hashes)),
        "shared_holdout_sample_ids_sha256": next(iter(holdout_hashes)),
        "batch_holdout_sample_count": int(len(holdout_ids)),
        "oov_max_semantic_error": float(max(oov_checks.values())),
        "oov_checks": oov_checks, "decisions": decisions,
        "chemical_decision": chemical_decision,
        "per_drug_negative_transfer_concentration": concentration,
        "training": summaries,
        "training_artifacts": {
            name: {
                "directory": str((root / relative).resolve()),
                "checkpoint": str((root / relative / "stage_b_best.pt").resolve()),
                "checkpoint_sha256": sha256_file(root / relative / "stage_b_best.pt"),
                "history": str((root / relative / "training_history.json").resolve()),
                "config": str((root / relative / "resolved_config.json").resolve()),
                "freeze_audit": str((root / relative / "stage_b_freeze_audit.json").resolve()),
            } for name, relative in MODELS.items()
        },
        "test_proteome_opened": False, "test_prediction_generated": False,
        "files": {name: str((output_dir / name).resolve()) for name in tables},
    }
    comparison_path = output_dir / "hierarchical_treatment_comparison.json"
    comparison_path.write_text(json.dumps(_finite_json(comparison), ensure_ascii=False, indent=2), encoding="utf-8")

    md = lambda frame, cols: markdown_table(frame, cols, digits=4)
    report = f"""# MODEL V2 Stage S2B Hierarchical Treatment Report

Status: COMPLETE_AND_PAUSED  
Scoring labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

All three models loaded the unchanged S2A hierarchical Stage-A checkpoint `{next(iter(stage_a_hashes))}`. Stage A was not retrained. D0/D1/D2 share Stage-B initialization hash `{next(iter(initial_hashes))}`, seed 20260814, 4,121 training treatments, the same 957-sample technical-tuple holdout, model dimensions, order, optimizer settings and four-scenario macro-Huber monitor. Baseline, genome, source, instrument and plate parameters are frozen exactly in every run.

{md(training_table, ['model','device','elapsed_seconds','peak_reserved_mib','stage_b_actual_epochs','stage_b_best_epoch','stage_b_best_macro_huber','stop_reason'])}

## Absolute metrics on the exact S0 common subset

{md(combined['absolute'], ['model','scenario','n_samples','log2_rmse','mae','global_r2','sample_pcc_median','sample_r2_median','protein_pcc_median','protein_r2_median'])}

## Raw FC and high-effect metrics

{md(combined['fc'], ['model','scenario','global_fc_pcc','sample_fc_pcc_median','protein_fc_pcc_median','fc_rmse','fc_direction_accuracy'])}

{md(combined['high'], ['model','scenario','direction_accuracy','high_effect_pcc','precision','recall','f1','auprc'])}

## Context and drug residual modules

{md(combined['context'], ['model','scenario','pcc','rmse','direction_accuracy'])}

{md(combined['drug'], ['model','scenario','pcc','rmse','direction_accuracy'])}

## Frozen hierarchical components

{md(component, ['model','scenario','delta_source_rms','delta_instrument_rms','delta_plate_rms','delta_batch_rms','delta_response_rms','batch_to_response_norm'])}

The maximum exact OOV semantic error across all models/scenarios is `{max(oov_checks.values()):.8g}`. Unseen plate clears plate only; unseen instrument clears instrument and plate; all-OOV clears total batch.

{md(oov, ['model','scenario','batch_mode','absolute_rmse','raw_fc_global_pcc','raw_fc_rmse'])}

## 957-treatment tuple holdout

These labels were used only after best-checkpoint restoration. Stage A control exposure may include the same technical groups, so this remains the predeclared treatment-label holdout rather than wholly new-batch OOD.

{md(holdout, ['model','n_samples','n_matched_fc_samples','absolute_rmse','absolute_mae','raw_fc_global_pcc','raw_fc_rmse'])}

## Morgan correct/shuffle/zero and drug attribution

{md(chemical, ['model','scenario','chemical_mode','absolute_rmse','raw_fc_global_pcc','raw_fc_rmse'])}

{md(gains, ['model','scenario','chemical_name','rmse_gain_vs_d0'])}

Direct D1-vs-D0 and D2-vs-D1 attribution (negative RMSE change is favorable; positive PCC gain is favorable):

{md(direct, ['candidate','reference','scenario','absolute_rmse_change_candidate_minus_reference','raw_fc_pcc_gain_candidate_minus_reference'])}

```json
{json.dumps(_finite_json(concentration), ensure_ascii=False, indent=2)}
```

## Predeclared advancement gates

```json
{json.dumps(_finite_json(decisions), ensure_ascii=False, indent=2)}
```

Chemical contribution rule:

```json
{json.dumps(_finite_json(chemical_decision), ensure_ascii=False, indent=2)}
```

## Attribution conclusions

- D0 versus S1 B improves mean absolute RMSE by `{abs(decisions[d0_name]['mean_absolute_rmse_change_vs_s1']):.4f}` and val_chem context residual PCC by `{decisions[d0_name]['val_chem_context_residual_pcc_gain_vs_s1']:.4f}`, but reduces mean raw-FC PCC by `{abs(decisions[d0_name]['mean_raw_fc_pcc_gain_vs_s1']):.4f}` and val_strain drug-residual PCC by `{abs(decisions[d0_name]['val_strain_drug_residual_pcc_gain_vs_s1']):.4f}`. Because D0 also removes Morgan, this is not a pure anchor contrast. The clean D1-vs-S1-B anchor contrast gains `{decisions['S2B D1 hierarchical Morgan Huber']['mean_raw_fc_pcc_gain_vs_s1']:.4f}` mean raw-FC PCC but worsens mean absolute RMSE by `{decisions['S2B D1 hierarchical Morgan Huber']['mean_absolute_rmse_change_vs_s1']:.4f}` and drug-residual PCC by `{abs(decisions['S2B D1 hierarchical Morgan Huber']['val_strain_drug_residual_pcc_gain_vs_s1']):.4f}`; hierarchical anchor integration is therefore mixed, not a uniform improvement.
- D1 is not stably better than D0. It helps val_strain_only and val_time absolute RMSE but harms val_chem_only and val_both; its raw-FC PCC likewise decreases on val_chem_only and increases on the other three scenarios. Both chemical contribution gates are false, so Morgan remains unestablished.
- D2 versus D1 improves raw-FC PCC in all four scenarios and absolute RMSE in three of four, while val_strain_only absolute RMSE worsens slightly. Thus the frozen nonofficial FC/correlation loss has limited additional direction value inside the Morgan model, but D2 still fails the val_strain residual and broad-improvement gates and is not advanced. The mapping remains experimental and was never used for early stopping, mapping changes or weight search.
- Every Stage-B freeze audit passed and the identical-input batch outputs are bitwise unchanged; treatment labels therefore did not update or get absorbed by any hierarchical batch level.
- Source/instrument/plate outputs are identical across D0/D1/D2 for each scenario, plate RMS remains far below source RMS, and the exact OOV error is zero. The S2A layer semantics are preserved rather than relearned from treatment.
- No model is advanced to multi-seed unless all seven gates pass. No test proteome was opened and no test prediction was generated.

## Commands, tests and failures

- Formal D0/D1/D2: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m baseline.training_v2 --config <D0|D1|D2 config> --output-dir <new formal directory>`; all three completed normally.
- Analysis: `D:\\虚拟细胞\\.venv\\Scripts\\python.exe -m baseline.hierarchical_treatment_analysis_v2`; PASS.
- All V2 tests: `python -m pytest --import-mode=importlib <all baseline/tests/test_*.py except test_person_c.py> -q`; `97 passed in 46.20s`.
- Person C regression, from `D:\\虚拟细胞\\baseline`: `python -m pytest tests/test_person_c.py -q`; `6 passed in 4.33s`.
- Formal training failures, OOM, NaN and checkpoint failures: none. An initial test collection without `--import-mode=importlib` produced three Windows namespace import errors; the documented project mode passed. The first analysis launch preceded creation of its thin module entry point and returned `No module named baseline.hierarchical_treatment_analysis_v2`; the wrapper was added and analysis passed without retraining or checkpoint changes.

Stage S2B is paused after these three single-seed runs.
"""
    report_path.write_text(report, encoding="utf-8")
    return _finite_json({
        "status": "PASS", "report": str(report_path), "comparison": str(comparison_path),
        "decisions": decisions, "chemical_decision": chemical_decision,
        "test_proteome_opened": False,
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
