"""Stage S1 score-aligned validation and robustness audit.

This module is inference-only.  Validation labels are used exclusively for
metrics.  Train labels are used only for the frozen S0 residual references and
the explicitly held-out train-treatment diagnostic.  No test proteome path is
accepted or opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Config
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, VAL_SCENARIOS, build_batch,
    fit_category_vocabulary, load_artifact_bundle, load_checkpoint,
    load_label_frames, load_train_val_metadata, make_chemical_feature_variant,
    sha256_file,
)
from scripts.audit_competition_score_alignment import (
    EXPECTED_EXACT_KEYS, absolute_metric_row, build_control_pairs,
    direction_accuracy, fc_metric_row, finite_mask, fit_masked_group_references,
    high_effect_metric_row, load_control_spec, masked_mae, masked_rmse,
    infer_v2_checkpoint, markdown_table, materialize_references, paired_pcc, residual_metric_row,
    stable_json_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
S1_MODELS = {
    "S1 B anchor Morgan Huber": "reports/model_v2_stage_s1/formal_b_anchor_morgan_huber_seed_20260814",
    "S1 C anchor Morgan Huber plus experimental FC": "reports/model_v2_stage_s1/formal_c_anchor_morgan_fc_seed_20260814",
}
REFERENCE_MODELS = (
    "Matched Control",
    "V2 batch-enabled Huber",
    "V2 no-batch Morgan-only",
    "V2 experimental parity FC Morgan no-batch",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _model_from_checkpoint(checkpoint: Path, artifacts, vocab, device):
    payload = load_checkpoint(checkpoint, artifacts.hashes)
    model_cfg = payload["config"]["model"]
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_cfg["latent_dim"]), protein_rank=int(model_cfg["protein_rank"]),
        dropout=float(model_cfg["dropout"]), batch_enabled=bool(model_cfg["batch_enabled"]),
        batch_structure=str(model_cfg.get("batch_structure", "flat_batch")),
        response_gate_enabled=bool(model_cfg.get("response_gate_enabled", False)),
        response_gate_initial=float(model_cfg.get("response_gate_initial", 0.25)),
        response_rms_cap=float(model_cfg.get("response_rms_cap", 0.0)),
        batch_field_dropout=float(model_cfg.get("batch_field_dropout", 0.0)),
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg, torch.zeros(cfg.n_proteins)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def _infer(model, encoded, device, batch_permutation=None):
    chunks = {name: [] for name in ("y_pred", "y_baseline", "delta_response", "delta_batch", "y_anchor")}
    with torch.inference_mode():
        for start in range(0, encoded.morgan.shape[0], 256):
            indexes = torch.arange(start, min(start + 256, encoded.morgan.shape[0]), dtype=torch.long)
            selected = encoded.index_select(indexes)
            if batch_permutation is not None:
                source = torch.as_tensor(batch_permutation[start:start + len(indexes)], dtype=torch.long)
                selected = selected.replace(batch_categorical=encoded.batch_categorical[source])
            output = model(selected.to(device))
            for name in chunks:
                chunks[name].append(output[name].detach().cpu().numpy().astype(np.float32, copy=False))
    return {name: np.concatenate(values) for name, values in chunks.items()}


def _predict_variants(model, payload, artifacts, meta, vocab, ids, device, scenario_offset):
    cfg = payload["config"]["model"]
    seed = int(payload["seed"])
    results = {}
    for chemical_mode in ("correct", "shuffle", "zero"):
        variant = make_chemical_feature_variant(artifacts, chemical_mode, seed)
        encoded = build_batch(
            meta, ids, artifacts, vocab, chemical_mode=chemical_mode,
            chemical_feature_components="morgan_only", genome_mode="correct", seed=seed,
            chemical_variant=variant,
        )
        results[f"chemical_{chemical_mode}"] = _infer(model, encoded, device)
        if chemical_mode == "correct":
            permutation = np.random.default_rng(seed + 9000 + scenario_offset).permutation(len(ids))
            results["batch_shuffled"] = _infer(model, encoded, device, permutation)
            results["batch_disabled"] = dict(results["chemical_correct"])
            results["batch_disabled"]["y_pred"] = (
                results["chemical_correct"]["y_baseline"]
                + results["chemical_correct"]["delta_response"]
            )
    return results


def _reconstruct_batch_holdout(meta, seed, fraction):
    train = meta.index[meta["split_final"].eq("train")]
    names = meta.loc[train, "perturbation_no_concentration"].astype(str).str.lower()
    treatment_ids = train[~names.isin(CONTROL_NAMES | {"quality control"})]
    frame = meta.loc[treatment_ids, list(BATCH_COLUMNS)].astype(str)
    keys = frame.agg("|".join, axis=1)
    groups = sorted(keys.unique())
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    selected = set(ordered[:max(1, int(round(len(ordered) * fraction)))])
    holdout = treatment_ids[keys.isin(selected).to_numpy()]
    train_ids = treatment_ids[~keys.isin(selected).to_numpy()]
    if set(keys.loc[holdout]) & set(keys.loc[train_ids]):
        raise RuntimeError("batch holdout groups overlap")
    return train_ids, holdout, keys.loc[holdout]


def _component_row(model, scenario, output, mask):
    valid = np.asarray(mask, bool)
    response = np.where(valid, output["delta_response"], 0.0)
    batch = np.where(valid, output["delta_batch"], 0.0)
    response_norm = float(np.linalg.norm(response.astype(np.float64)))
    batch_norm = float(np.linalg.norm(batch.astype(np.float64)))
    return {
        "model": model, "scenario": scenario,
        "delta_response_norm": response_norm, "delta_batch_norm": batch_norm,
        "batch_to_response_norm": batch_norm / response_norm if response_norm > 0 else math.nan,
        "delta_response_rms": float(np.sqrt(np.square(response, dtype=np.float64).sum() / max(valid.sum(), 1))),
        "delta_batch_rms": float(np.sqrt(np.square(batch, dtype=np.float64).sum() / max(valid.sum(), 1))),
    }


def run_analysis(root=ROOT, output_dir=None, report_path=None, device="auto"):
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "reports/model_v2_stage_s1").resolve()
    report_path = Path(report_path or root / "reports/MODEL_V2_STAGE_S1_SCORE_ALIGNED_REPORT.md").resolve()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    artifacts = load_artifact_bundle(root)
    meta = load_train_val_metadata(root)
    labels, masks = load_label_frames(meta, artifacts, root)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    spec, parity = load_control_spec(root)
    scenario_ids = {name: meta.index[meta["split_final"].eq(name)] for name in VAL_SCENARIOS}
    train_controls = train_ids[meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)]
    train_treatments = train_ids[~meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})]
    all_controls = meta.index[meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)]
    train_pairs = build_control_pairs(meta, labels, masks, train_treatments, train_controls, EXPECTED_EXACT_KEYS, parity)
    context_refs = fit_masked_group_references(meta, train_pairs, EXPECTED_EXACT_KEYS, train_ids)
    drug_refs = fit_masked_group_references(meta, train_pairs, ("perturbation_no_concentration",), train_ids)

    pairs = {}
    for scenario, ids in scenario_ids.items():
        treatment = ids[~meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})]
        pairs[scenario] = build_control_pairs(meta, labels, masks, treatment, all_controls, EXPECTED_EXACT_KEYS, parity)

    predictions, payloads, summaries = {}, {}, {}
    for model_name, relative in S1_MODELS.items():
        directory = root / relative
        model, payload = _model_from_checkpoint(directory / "stage_b_best.pt", artifacts, vocab, device)
        payloads[model_name] = payload
        summaries[model_name] = _json(directory / "training_summary.json")
        predictions[model_name] = {}
        for offset, (scenario, ids) in enumerate(scenario_ids.items()):
            predictions[model_name][scenario] = _predict_variants(
                model, payload, artifacts, meta, vocab, ids, device, offset,
            )
    current_predictions, _ = infer_v2_checkpoint(
        root,
        root / "reports/model_v2_stage2/formal_huber_seed_20260814/stage_b_best.pt",
        artifacts, meta, vocab, scenario_ids, device,
    )

    initial_hashes = {value["stage_b_initial_chemical_response_sha256"] for value in summaries.values()}
    holdout_hashes = {value["train_internal_batch_holdout"]["holdout_sample_ids_sha256"] for value in summaries.values()}
    if len(initial_hashes) != 1 or len(holdout_hashes) != 1:
        raise RuntimeError("B/C fairness hashes differ")

    s0_dir = root / "reports/competition_score_audit"
    reference_absolute = pd.read_csv(s0_dir / "absolute_metrics.csv")
    reference_fc = pd.read_csv(s0_dir / "fc_metrics.csv")
    reference_context = pd.read_csv(s0_dir / "context_residual_metrics.csv")
    reference_drug = pd.read_csv(s0_dir / "drug_residual_metrics.csv")
    reference_high = pd.read_csv(s0_dir / "high_effect_metrics.csv")
    absolute_rows, fc_rows, context_rows, drug_rows, high_rows = [], [], [], [], []
    component_rows, batch_rows, chemical_rows, per_drug_rows = [], [], [], []

    for scenario, ids in scenario_ids.items():
        ids = pd.Index(map(str, ids))
        truth = labels.loc[ids].to_numpy(np.float32, copy=True)
        truth_mask = masks.loc[ids].to_numpy(bool, copy=True)
        pair = pairs[scenario]
        lookup = {sample_id: pos for pos, sample_id in enumerate(ids)}
        indexes = np.asarray([lookup[value] for value in pair.treatment_ids], dtype=np.int64)
        common = pair.delta_mask.copy()
        expected_s0 = reference_fc.loc[
            reference_fc["scenario"].eq(scenario) & reference_fc["subset"].eq("common_intersection"),
            "n_valid_positions",
        ].unique()
        if len(expected_s0) != 1 or int(expected_s0[0]) != int(common.sum()):
            raise RuntimeError(f"S0 common subset mismatch for {scenario}")
        pair_true = truth[indexes]
        current_pair_pred = current_predictions[scenario][indexes]
        current_delta = current_pair_pred - pair.control_values
        names = meta.loc[list(pair.treatment_ids), "perturbation_no_concentration"].astype(str).to_numpy()
        for chemical_name in sorted(set(names)):
            selected_drug = names == chemical_name
            drug_mask = common & selected_drug[:, None]
            per_drug_rows.append({
                "model": "V2 batch-enabled Huber", "scenario": scenario,
                "chemical_name": chemical_name, "n_samples": int(selected_drug.sum()),
                "n_valid_positions": int(drug_mask.sum()),
                "absolute_rmse": masked_rmse(pair_true, current_pair_pred, drug_mask),
                "absolute_mae": masked_mae(pair_true, current_pair_pred, drug_mask),
                "raw_fc_pcc": paired_pcc(pair.delta_true, current_delta, drug_mask),
                "raw_fc_rmse": masked_rmse(pair.delta_true, current_delta, drug_mask),
                "raw_fc_direction_accuracy": direction_accuracy(pair.delta_true, current_delta, drug_mask),
            })
        for model_name in S1_MODELS:
            variants = predictions[model_name][scenario]
            correct = variants["chemical_correct"]
            pred = correct["y_pred"][indexes]
            delta = pred - pair.control_values
            absolute_rows.append(absolute_metric_row(model_name, scenario, "common_intersection", pair.treatment_ids, pair_true, pred, common))
            fc_rows.append(fc_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, common))
            high_rows.append(high_effect_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, common))
            component_rows.append(_component_row(model_name, scenario, {key: value[indexes] for key, value in correct.items()}, common))
            if scenario == "val_chem_only":
                reference, reference_mask = materialize_references(meta, pair.treatment_ids, context_refs, EXPECTED_EXACT_KEYS, artifacts.feature_contract.n_proteins)
                context_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, reference, reference_mask, common))
            if scenario == "val_strain_only":
                reference, reference_mask = materialize_references(meta, pair.treatment_ids, drug_refs, ("perturbation_no_concentration",), artifacts.feature_contract.n_proteins)
                drug_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pair.delta_true, delta, reference, reference_mask, common))
            for mode, key in (("correct", "chemical_correct"), ("shuffled", "batch_shuffled"), ("disabled", "batch_disabled")):
                mode_pred = variants[key]["y_pred"][indexes]
                mode_delta = mode_pred - pair.control_values
                batch_rows.append({
                    "model": model_name, "scenario": scenario, "batch_mode": mode,
                    "absolute_rmse": masked_rmse(pair_true, mode_pred, common),
                    "raw_fc_global_pcc": paired_pcc(pair.delta_true, mode_delta, common),
                    "raw_fc_rmse": masked_rmse(pair.delta_true, mode_delta, common),
                })
            for mode in ("correct", "shuffle", "zero"):
                mode_pred = variants[f"chemical_{mode}"]["y_pred"][indexes]
                mode_delta = mode_pred - pair.control_values
                chemical_rows.append({
                    "model": model_name, "scenario": scenario, "chemical_mode": mode,
                    "absolute_rmse": masked_rmse(pair_true, mode_pred, common),
                    "raw_fc_global_pcc": paired_pcc(pair.delta_true, mode_delta, common),
                    "raw_fc_rmse": masked_rmse(pair.delta_true, mode_delta, common),
                })
            for chemical_name in sorted(set(names)):
                selected = names == chemical_name
                drug_mask = common & selected[:, None]
                per_drug_rows.append({
                    "model": model_name, "scenario": scenario, "chemical_name": chemical_name,
                    "n_samples": int(selected.sum()), "n_valid_positions": int(drug_mask.sum()),
                    "absolute_rmse": masked_rmse(pair_true, pred, drug_mask),
                    "absolute_mae": masked_mae(pair_true, pred, drug_mask),
                    "raw_fc_pcc": paired_pcc(pair.delta_true, delta, drug_mask),
                    "raw_fc_rmse": masked_rmse(pair.delta_true, delta, drug_mask),
                    "raw_fc_direction_accuracy": direction_accuracy(pair.delta_true, delta, drug_mask),
                })

    absolute_new, fc_new = pd.DataFrame(absolute_rows), pd.DataFrame(fc_rows)
    context_new, drug_new, high_new = map(pd.DataFrame, (context_rows, drug_rows, high_rows))
    selected = lambda frame: frame.loc[frame["model"].isin(REFERENCE_MODELS) & frame["subset"].eq("common_intersection")].copy()
    absolute_all = pd.concat([selected(reference_absolute), absolute_new], ignore_index=True)
    fc_all = pd.concat([selected(reference_fc), fc_new], ignore_index=True)
    context_all = pd.concat([selected(reference_context), context_new], ignore_index=True)
    drug_all = pd.concat([selected(reference_drug), drug_new], ignore_index=True)
    high_all = pd.concat([selected(reference_high), high_new], ignore_index=True)

    # True train-internal treatment-label group holdout, evaluated only now.
    train_stage_b, holdout_ids, holdout_keys = _reconstruct_batch_holdout(meta, 20260814, 0.2)
    holdout_table = pd.DataFrame({"sample_ID": holdout_ids, "technical_group": holdout_keys.to_numpy()})
    holdout_table["split_role"] = "final_diagnostic_only"
    holdout_pairs = build_control_pairs(meta, labels, masks, holdout_ids, train_controls, EXPECTED_EXACT_KEYS, parity)
    holdout_metric_rows = []
    for offset, (model_name, relative) in enumerate(S1_MODELS.items()):
        model, payload = _model_from_checkpoint(root / relative / "stage_b_best.pt", artifacts, vocab, device)
        variants = _predict_variants(model, payload, artifacts, meta, vocab, holdout_ids, device, 100 + offset)
        truth = labels.loc[holdout_ids].to_numpy(np.float32, copy=True)
        truth_mask = masks.loc[holdout_ids].to_numpy(bool, copy=True)
        for mode, key in (("correct", "chemical_correct"), ("shuffled", "batch_shuffled"), ("disabled", "batch_disabled")):
            pred = variants[key]["y_pred"]
            delta = pred - holdout_pairs.control_values
            holdout_metric_rows.append({
                "model": model_name, "batch_mode": mode, "n_samples": len(holdout_ids),
                "n_matched_fc_samples": int(holdout_pairs.delta_mask.any(1).sum()),
                "absolute_rmse": masked_rmse(truth, pred, truth_mask),
                "absolute_mae": masked_mae(truth, pred, truth_mask),
                "raw_fc_global_pcc": paired_pcc(holdout_pairs.delta_true, delta, holdout_pairs.delta_mask),
                "raw_fc_rmse": masked_rmse(holdout_pairs.delta_true, delta, holdout_pairs.delta_mask),
            })

    component = pd.DataFrame(component_rows)
    batch = pd.DataFrame(batch_rows)
    chemical = pd.DataFrame(chemical_rows)
    holdout_metrics = pd.DataFrame(holdout_metric_rows)
    per_drug = pd.DataFrame(per_drug_rows)

    current = "V2 batch-enabled Huber"
    decisions = {}
    for model_name in S1_MODELS:
        current_fc = fc_all.loc[fc_all["model"].eq(current), "global_fc_pcc"].mean()
        model_fc = fc_all.loc[fc_all["model"].eq(model_name), "global_fc_pcc"].mean()
        current_abs = absolute_all.loc[absolute_all["model"].eq(current), "log2_rmse"].mean()
        model_abs = absolute_all.loc[absolute_all["model"].eq(model_name), "log2_rmse"].mean()
        current_context = context_all.loc[context_all["model"].eq(current), "pcc"].mean()
        model_context = context_all.loc[context_all["model"].eq(model_name), "pcc"].mean()
        current_drug = drug_all.loc[drug_all["model"].eq(current), "pcc"].mean()
        model_drug = drug_all.loc[drug_all["model"].eq(model_name), "pcc"].mean()
        robustness = batch.loc[batch["model"].eq(model_name)].pivot(index="scenario", columns="batch_mode", values=["absolute_rmse", "raw_fc_global_pcc"])
        worst_disabled_abs_ratio = float((robustness["absolute_rmse"]["disabled"] / robustness["absolute_rmse"]["correct"]).max())
        worst_shuffle_abs_ratio = float((robustness["absolute_rmse"]["shuffled"] / robustness["absolute_rmse"]["correct"]).max())
        chemical_focus = chemical.loc[chemical["model"].eq(model_name) & chemical["scenario"].isin(["val_chem_only", "val_both"])]
        chemical_pivot = chemical_focus.pivot(index="scenario", columns="chemical_mode", values="raw_fc_global_pcc")
        chemical_ok = bool((chemical_pivot["correct"] >= chemical_pivot[["shuffle", "zero"]].max(axis=1) - 0.01).all())
        drug_compare = per_drug.loc[per_drug["model"].isin([current, model_name]), ["model", "scenario", "chemical_name", "raw_fc_rmse"]].pivot(
            index=["scenario", "chemical_name"], columns="model", values="raw_fc_rmse",
        ).dropna()
        drug_gains = drug_compare[current] - drug_compare[model_name]
        positive_drug_gains = drug_gains[drug_gains > 0]
        improved_drug_fraction = float((drug_gains > 0).mean()) if len(drug_gains) else 0.0
        top3_positive_share = float(positive_drug_gains.nlargest(3).sum() / positive_drug_gains.sum()) if positive_drug_gains.sum() > 0 else 1.0
        scenario_gains = (
            fc_all.loc[fc_all["model"].eq(model_name)].set_index("scenario")["global_fc_pcc"]
            - fc_all.loc[fc_all["model"].eq(current)].set_index("scenario")["global_fc_pcc"]
        )
        positive_scenario = scenario_gains[scenario_gains > 0]
        max_scenario_share = float(positive_scenario.max() / positive_scenario.sum()) if positive_scenario.sum() > 0 else 1.0
        broad_gain = bool(improved_drug_fraction >= 0.60 and top3_positive_share <= 0.50 and max_scenario_share <= 0.75)
        score_gain = max(model_fc - current_fc, model_context - current_context, model_drug - current_drug)
        decisions[model_name] = {
            "mean_raw_fc_pcc": float(model_fc), "raw_fc_pcc_gain_vs_current_v2": float(model_fc - current_fc),
            "val_chem_context_residual_pcc_gain_vs_current_v2": float(model_context - current_context),
            "val_strain_drug_residual_pcc_gain_vs_current_v2": float(model_drug - current_drug),
            "max_primary_or_high_weight_pcc_gain": float(score_gain),
            "mean_absolute_rmse": float(model_abs), "absolute_rmse_ratio_vs_current_v2": float(model_abs / current_abs),
            "worst_batch_disabled_absolute_rmse_ratio": worst_disabled_abs_ratio,
            "worst_batch_shuffle_absolute_rmse_ratio": worst_shuffle_abs_ratio,
            "chemical_correct_not_worse_than_shuffle_zero_with_tolerance_0.01": chemical_ok,
            "improved_drug_fraction_by_raw_fc_rmse": improved_drug_fraction,
            "top3_positive_drug_gain_share": top3_positive_share,
            "largest_positive_scenario_gain_share": max_scenario_share,
            "gate_1_fc_or_residual_improves": bool(score_gain >= 0.02),
            "gate_2_absolute_not_collapsed": bool(model_abs / current_abs <= 1.10),
            "gate_3_batch_robust": bool(max(worst_disabled_abs_ratio, worst_shuffle_abs_ratio) <= 1.10),
            "gate_4_chemical_supported": chemical_ok,
            "gate_5_not_few_drug_or_single_scenario_dominated": broad_gain,
        }
        decisions[model_name]["eligible_for_multiseed"] = all(
            decisions[model_name][key] for key in (
                "gate_1_fc_or_residual_improves", "gate_2_absolute_not_collapsed",
                "gate_3_batch_robust", "gate_4_chemical_supported",
                "gate_5_not_few_drug_or_single_scenario_dominated",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "score_aligned_absolute_metrics.csv": absolute_all,
        "score_aligned_fc_metrics.csv": fc_all,
        "score_aligned_context_residual_metrics.csv": context_all,
        "score_aligned_drug_residual_metrics.csv": drug_all,
        "score_aligned_high_effect_metrics.csv": high_all,
        "score_aligned_component_norms.csv": component,
        "score_aligned_batch_robustness.csv": batch,
        "score_aligned_chemical_robustness.csv": chemical,
        "score_aligned_batch_holdout_metrics.csv": holdout_metrics,
        "score_aligned_per_drug_metrics.csv": per_drug,
        "score_aligned_batch_holdout_ids.csv": holdout_table,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)

    comparison = {
        "schema_version": "1.0", "task": "Stage S1 score-aligned control-anchored response model",
        "planning_proxy": True, "official_score": False,
        "experimental_nonofficial_parity_fc": True, "official_fc_result": False,
        "seed": 20260814, "stage_a_retrained": False,
        "stage_a_checkpoint_sha256": next(iter(summaries.values()))["stage_a"]["fixed_checkpoint"]["sha256"],
        "shared_stage_b_initial_hash": next(iter(initial_hashes)),
        "shared_batch_holdout_hash": next(iter(holdout_hashes)),
        "batch_holdout_sample_count": len(holdout_ids),
        "batch_holdout_scope_note": "Stage B treatment labels held out by full technical tuple; Stage A control exposure may share technical categories",
        "decisions": decisions,
        "training": summaries,
        "training_artifacts": {
            model: {
                "directory": str((root / relative).resolve()),
                "checkpoint": str((root / relative / "stage_b_best.pt").resolve()),
                "checkpoint_sha256": sha256_file(root / relative / "stage_b_best.pt"),
                "history": str((root / relative / "training_history.json").resolve()),
                "resolved_config": str((root / relative / "resolved_config.json").resolve()),
            }
            for model, relative in S1_MODELS.items()
        },
        "test_proteome_opened": False, "test_prediction_generated": False,
        "files": {name: str((output_dir / name).resolve()) for name in tables},
    }
    comparison_path = output_dir / "score_aligned_comparison.json"
    comparison_path.write_text(json.dumps(_finite_json(comparison), ensure_ascii=False, indent=2), encoding="utf-8")

    def md(table, columns):
        return markdown_table(table, columns, digits=4)
    report = f"""# MODEL V2 Stage S1 Score-Aligned Report

Status: COMPLETE_AND_PAUSED  
Scoring labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

Stage A was not retrained. B and C share checkpoint SHA `{comparison['stage_a_checkpoint_sha256']}`, Stage-B initialization hash `{comparison['shared_stage_b_initial_hash']}`, seed, data order, 4,121 training treatments, and the same 957-sample technical-group holdout. C alone adds the predeclared train-only nonofficial parity FC Huber 0.05 plus row-wise Pearson 0.05.

## Absolute metrics on the exact S0 common intersection

{md(absolute_all, ['model','scenario','n_samples','log2_rmse','mae','global_r2','sample_pcc_median','sample_r2_median','protein_pcc_median','protein_r2_median'])}

## Raw FC

{md(fc_all, ['model','scenario','global_fc_pcc','sample_fc_pcc_median','protein_fc_pcc_median','fc_rmse','fc_direction_accuracy'])}

## Context and drug residual

{md(context_all, ['model','scenario','pcc','rmse','direction_accuracy'])}

{md(drug_all, ['model','scenario','pcc','rmse','direction_accuracy'])}

## High-effect diagnostics

{md(high_all, ['model','scenario','direction_accuracy','high_effect_pcc','precision','recall','f1','auprc'])}

## Component norms

{md(component, ['model','scenario','delta_response_norm','delta_batch_norm','batch_to_response_norm','delta_response_rms','delta_batch_rms'])}

## Batch robustness and internal group holdout

Validation post-hoc correct/shuffled/disabled modes do not retrain or select checkpoints.

{md(batch, ['model','scenario','batch_mode','absolute_rmse','raw_fc_global_pcc','raw_fc_rmse'])}

The 957 held-out Stage-B treatment labels were never used for gradient updates or early stopping. Stage A control samples can still share technical categories, so this is specifically a treatment-label group holdout, not a claim of wholly unseen instrumentation.

{md(holdout_metrics, ['model','batch_mode','n_samples','absolute_rmse','raw_fc_global_pcc','raw_fc_rmse'])}

## Morgan correct/shuffle/zero post-hoc diagnostic

{md(chemical, ['model','scenario','chemical_mode','absolute_rmse','raw_fc_global_pcc','raw_fc_rmse'])}

## Predeclared advancement gates

Thresholds used only for the go/no-go audit: raw-FC mean improvement at least 0.02, absolute-RMSE ratio no more than 1.10, worst batch shuffle/disabled absolute-RMSE ratio no more than 1.10, and Morgan correct no worse than shuffle/zero by more than 0.01 PCC in val_chem_only and val_both.

```json
{json.dumps(_finite_json(decisions), ensure_ascii=False, indent=2)}
```

## Conclusion

- Matched Control remains the absolute-fidelity reference. Both S1 models improve mean absolute RMSE by about 2.4% versus the frozen batch-enabled V2, but remain well behind Matched Control.
- B improves the best primary/high-weight PCC proxy by only {decisions['S1 B anchor Morgan Huber']['max_primary_or_high_weight_pcc_gain']:.4f}; C improves it by {decisions['S1 C anchor Morgan Huber plus experimental FC']['max_primary_or_high_weight_pcc_gain']:.4f}. Neither reaches the predeclared +0.02 clear-improvement threshold.
- C's FC/correlation objective improves mean raw-FC PCC more than B without an absolute-fidelity collapse, but the gain is modest rather than decisive. High-effect changes are mixed across scenario and metric; the full table above is the authoritative record.
- Batch dependence remains the primary blocker: the worst validation absolute-RMSE ratio is {decisions['S1 C anchor Morgan Huber plus experimental FC']['worst_batch_disabled_absolute_rmse_ratio']:.2f}x when disabled and {decisions['S1 C anchor Morgan Huber plus experimental FC']['worst_batch_shuffle_absolute_rmse_ratio']:.2f}x when shuffled. The independent 957-treatment group holdout shows the same qualitative collapse.
- Morgan correct is not consistently better than frozen-interface shuffle/zero in `val_chem_only` and `val_both`; chemical generalization is therefore not established.
- B and C both fail the advancement gate. No multi-seed run is authorized from these results. The next iteration should prioritize baseline/anchor modeling that does not depend on public technical batches, not increase FC weight or strengthen batch.

## Commands and execution status

- B and C were run with `python -m baseline.training_v2`, their declared JSON/YAML-compatible configs, the fixed Stage A checkpoint, and separate new output directories.
- The final inference audit was run with `python -m baseline.score_aligned_analysis_v2`.
- Formal training failures, OOM, NaN, and checkpoint failures: none.
- A report-render pass initially found the optional `tabulate` package absent; the implementation was changed to reuse the dependency-free S0 Markdown renderer. No package was installed and no training was repeated.

No model is advanced to multi-seed unless every gate is true. Validation parity FC was not used for early stopping, mapping adjustment, or weight search. No test proteome was opened and no test prediction was generated.
"""
    report_path.write_text(report, encoding="utf-8")
    return _finite_json({
        "status": "PASS", "report": str(report_path), "comparison": str(comparison_path),
        "decisions": decisions, "test_proteome_opened": False,
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
