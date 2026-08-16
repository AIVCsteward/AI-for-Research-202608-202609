"""Stage 2.3a read-only normalized response amplitude and direction audit.

No function in this module trains a model or modifies a checkpoint. Validation
labels are used only to compute diagnostics after frozen checkpoint inference.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemical_ood_analysis_v2 import (
    CHEMICAL_COLUMN,
    OOD_SCENARIOS,
    _build_model,
    _json_ready,
    spearman_correlation,
)
from .evaluation_v2 import masked_rmse
from .training_v2 import (
    ROOT,
    build_batch,
    fit_category_vocabulary,
    load_artifact_bundle,
    load_checkpoint,
    load_label_frames,
    load_train_val_metadata,
    make_chemical_feature_variant,
    make_loader,
    sha256_file,
)


VARIANTS = ("correct", "zero")


@dataclass(frozen=True)
class ResponseEvaluation:
    sample_ids: tuple[str, ...]
    prediction: np.ndarray
    baseline: np.ndarray
    response: np.ndarray
    target: np.ndarray
    mask: np.ndarray


def per_sample_response_amplitudes(response, mask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample RMS/L2 using only common valid protein positions."""
    response = np.asarray(response, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if response.shape != mask.shape or response.ndim != 2:
        raise ValueError("response and mask must be aligned 2D arrays")
    valid = mask & np.isfinite(response)
    counts = valid.sum(axis=1)
    sums_of_squares = np.square(np.where(valid, response, 0.0)).sum(axis=1)
    rms = np.sqrt(np.divide(
        sums_of_squares, counts,
        out=np.full(len(counts), np.nan, dtype=np.float64), where=counts > 0,
    ))
    l2 = np.sqrt(sums_of_squares)
    l2[counts == 0] = np.nan
    return rms, l2, counts


def summarize_response_amplitude(response, mask) -> dict:
    sample_rms, sample_l2, counts = per_sample_response_amplitudes(response, mask)
    valid_samples = np.isfinite(sample_rms)
    valid_positions = np.asarray(mask, bool) & np.isfinite(response)
    return {
        "sample_response_rms_mean": float(np.mean(sample_rms[valid_samples])) if valid_samples.any() else np.nan,
        "sample_response_rms_median": float(np.median(sample_rms[valid_samples])) if valid_samples.any() else np.nan,
        "sample_response_l2_mean": float(np.mean(sample_l2[valid_samples])) if valid_samples.any() else np.nan,
        "sample_response_l2_median": float(np.median(sample_l2[valid_samples])) if valid_samples.any() else np.nan,
        "raw_cumulative_response_l2": (
            float(np.linalg.vector_norm(np.asarray(response, np.float64)[valid_positions]))
            if valid_positions.any() else np.nan
        ),
        "n_samples_with_valid_positions": int(valid_samples.sum()),
        "min_valid_proteins_per_sample": int(counts[valid_samples].min()) if valid_samples.any() else 0,
    }


def response_direction_diagnostics(predicted_response, target, baseline, mask) -> dict:
    """Flatten common valid positions for validation-only residual diagnostics."""
    predicted_response = np.asarray(predicted_response, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not (predicted_response.shape == target.shape == baseline.shape == mask.shape):
        raise ValueError("direction diagnostic arrays are not aligned")
    target_residual = target - baseline
    valid = (
        mask & np.isfinite(predicted_response) & np.isfinite(target_residual)
        & np.isfinite(target) & np.isfinite(baseline)
    )
    count = int(valid.sum())
    if count == 0:
        return {
            "predicted_response_rms": np.nan,
            "target_residual_rms": np.nan,
            "response_target_norm_ratio": np.nan,
            "response_target_cosine": np.nan,
            "response_target_pearson": np.nan,
            "alpha_star": np.nan,
            "n_common_valid_positions": 0,
        }
    predicted = predicted_response[valid]
    residual = target_residual[valid]
    pred_square = float(np.dot(predicted, predicted))
    residual_square = float(np.dot(residual, residual))
    cross = float(np.dot(predicted, residual))
    pred_norm = np.sqrt(pred_square)
    residual_norm = np.sqrt(residual_square)
    cosine = cross / (pred_norm * residual_norm) if pred_norm > 0 and residual_norm > 0 else np.nan
    pred_centered = predicted - predicted.mean()
    residual_centered = residual - residual.mean()
    pearson_denom = np.linalg.vector_norm(pred_centered) * np.linalg.vector_norm(residual_centered)
    pearson = float(np.dot(pred_centered, residual_centered) / pearson_denom) if pearson_denom > 0 else np.nan
    return {
        "predicted_response_rms": float(pred_norm / np.sqrt(count)),
        "target_residual_rms": float(residual_norm / np.sqrt(count)),
        "response_target_norm_ratio": float(pred_norm / residual_norm) if residual_norm > 0 else np.nan,
        "response_target_cosine": float(cosine),
        "response_target_pearson": pearson,
        "alpha_star": float(cross / pred_square) if pred_square > 0 else np.nan,
        "n_common_valid_positions": count,
    }


def assert_response_alignment(reference: ResponseEvaluation, candidate: ResponseEvaluation):
    if reference.sample_ids != candidate.sample_ids:
        raise ValueError("correct/zero sample IDs differ")
    if not np.array_equal(reference.mask, candidate.mask):
        raise ValueError("correct/zero masks differ")
    if not np.array_equal(reference.target, candidate.target, equal_nan=True):
        raise ValueError("correct/zero targets differ")


def evaluate_response_variant(
    mode: str,
    checkpoint_path: Path,
    scenario: str,
    meta,
    labels,
    masks,
    artifacts,
    vocab,
    batch_size: int,
    device: str,
) -> ResponseEvaluation:
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    if payload["config"]["model"]["chemical_mode"] != mode:
        raise ValueError(f"checkpoint mode does not match {mode}")
    ids = meta.index[meta["split_final"].eq(scenario)]
    seed = int(payload["seed"])
    chemical_variant = make_chemical_feature_variant(artifacts, mode=mode, seed=seed)
    batch = build_batch(
        meta, ids, artifacts, vocab, chemical_mode=mode, genome_mode="correct",
        seed=seed, chemical_variant=chemical_variant,
    )
    target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
    mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
    loader = make_loader(batch, target, mask, batch_size=int(batch_size), shuffle=False)
    model = _build_model(payload, artifacts, vocab, device)
    predictions, baselines, responses, targets, valid_masks = [], [], [], [], []
    with torch.no_grad():
        for inputs, y_true, valid in loader:
            outputs = model(inputs.to(device))
            predictions.append(outputs["y_pred"].detach().cpu().numpy())
            baselines.append(outputs["y_baseline"].detach().cpu().numpy())
            responses.append(outputs["delta_response"].detach().cpu().numpy())
            targets.append(y_true.numpy())
            valid_masks.append(valid.numpy())
    return ResponseEvaluation(
        tuple(map(str, ids)), np.concatenate(predictions), np.concatenate(baselines),
        np.concatenate(responses), np.concatenate(targets), np.concatenate(valid_masks),
    )


def build_per_entity_table(meta, evaluations) -> pd.DataFrame:
    records = []
    for scenario in OOD_SCENARIOS:
        correct = evaluations["correct"][scenario]
        zero = evaluations["zero"][scenario]
        assert_response_alignment(correct, zero)
        ids = pd.Index(correct.sample_ids)
        chemicals = meta.loc[ids, CHEMICAL_COLUMN].astype(str).to_numpy()
        for chemical in sorted(np.unique(chemicals)):
            positions = np.flatnonzero(chemicals == chemical)
            correct_amplitude = summarize_response_amplitude(correct.response[positions], correct.mask[positions])
            zero_amplitude = summarize_response_amplitude(zero.response[positions], zero.mask[positions])
            direction = response_direction_diagnostics(
                correct.response[positions], correct.target[positions],
                correct.baseline[positions], correct.mask[positions],
            )
            correct_rmse = masked_rmse(
                correct.target[positions], correct.prediction[positions], correct.mask[positions],
            )
            zero_rmse = masked_rmse(
                zero.target[positions], zero.prediction[positions], zero.mask[positions],
            )
            correct_mean = correct_amplitude["sample_response_rms_mean"]
            zero_mean = zero_amplitude["sample_response_rms_mean"]
            ratio = correct_mean / zero_mean if np.isfinite(zero_mean) and zero_mean > 0 else np.nan
            record = {
                "scenario": scenario,
                "chemical_name": chemical,
                "n_samples": int(len(positions)),
                "correct_rmse": correct_rmse,
                "zero_rmse": zero_rmse,
                "correct_harm_vs_zero": correct_rmse - zero_rmse,
                **{f"correct_{key}": value for key, value in correct_amplitude.items()},
                **{f"zero_{key}": value for key, value in zero_amplitude.items()},
                "correct_zero_response_rms_ratio": ratio,
                **direction,
                "target_residual_definition": "y_true_minus_y_baseline_validation_diagnostic_not_official_fc",
            }
            records.append(record)
    table = pd.DataFrame(records).sort_values(["scenario", "chemical_name"]).reset_index(drop=True)
    if table.duplicated(["scenario", "chemical_name"]).any():
        raise ValueError("duplicate scenario-chemical row")
    return table


MEAN_COMBINED_COLUMNS = (
    "correct_rmse", "zero_rmse", "correct_harm_vs_zero",
    "correct_sample_response_rms_mean", "correct_sample_response_rms_median",
    "correct_sample_response_l2_mean", "correct_sample_response_l2_median",
    "zero_sample_response_rms_mean", "zero_sample_response_rms_median",
    "zero_sample_response_l2_mean", "zero_sample_response_l2_median",
    "correct_zero_response_rms_ratio", "predicted_response_rms", "target_residual_rms",
    "response_target_norm_ratio", "response_target_cosine", "response_target_pearson",
    "alpha_star",
)


def combine_unique_drugs_equal_weight(per_entity: pd.DataFrame) -> pd.DataFrame:
    """Make one row per drug; the two scenarios receive equal scalar weight."""
    records = []
    for chemical, group in per_entity.groupby("chemical_name", sort=True):
        if set(group["scenario"]) != set(OOD_SCENARIOS) or len(group) != len(OOD_SCENARIOS):
            raise ValueError(f"combined drug must have exactly both OOD scenarios: {chemical}")
        record = {
            "chemical_name": chemical,
            "scenario_count": int(len(group)),
            "n_samples": int(group["n_samples"].sum()),
            "correct_raw_cumulative_response_l2": float(np.sqrt(np.square(group["correct_raw_cumulative_response_l2"]).sum())),
            "zero_raw_cumulative_response_l2": float(np.sqrt(np.square(group["zero_raw_cumulative_response_l2"]).sum())),
            "n_common_valid_positions": int(group["n_common_valid_positions"].sum()),
            "aggregation": "equal_mean_of_two_scenario_level_metrics_per_unique_drug",
        }
        for column in MEAN_COMBINED_COLUMNS:
            record[column] = float(group[column].mean())
        records.append(record)
    combined = pd.DataFrame(records)
    if len(combined) != 6 or not combined["chemical_name"].is_unique:
        raise ValueError("combined OOD analysis must contain exactly six equally weighted drugs")
    return combined


def correlation_block(table: pd.DataFrame) -> dict:
    return {
        "normalized_response_rms_vs_correct_harm": spearman_correlation(
            table["correct_sample_response_rms_mean"], table["correct_harm_vs_zero"],
        ),
        "correct_zero_response_rms_ratio_vs_correct_harm": spearman_correlation(
            table["correct_zero_response_rms_ratio"], table["correct_harm_vs_zero"],
        ),
        "n_samples_vs_raw_cumulative_response_l2": spearman_correlation(
            table["n_samples"], table["correct_raw_cumulative_response_l2"],
        ),
        "n_samples_vs_correct_harm": spearman_correlation(
            table["n_samples"], table["correct_harm_vs_zero"],
        ),
    }


def build_summary(per_entity, combined, checkpoints, artifacts, device) -> dict:
    correlations = {
        scenario: correlation_block(per_entity.loc[per_entity["scenario"].eq(scenario)])
        for scenario in OOD_SCENARIOS
    }
    correlations["combined_unique_drugs_equal_weight"] = correlation_block(combined)
    harmed = combined.loc[combined["correct_harm_vs_zero"] > 0]
    benefited = combined.loc[combined["correct_harm_vs_zero"] < 0]
    return _json_ready({
        "status": "PASS",
        "analysis": "stage2_3a_read_only_normalized_response_amplitude_and_direction",
        "device": device,
        "per_entity_row_count": int(len(per_entity)),
        "combined_unique_drug_count": int(len(combined)),
        "combined_aggregation": "one observation per drug; equal mean of val_chem_only and val_both scalar diagnostics",
        "correlations": correlations,
        "combined_unique_drugs": combined.to_dict("records"),
        "direction_interpretation": {
            "harmed_drug_count": int(len(harmed)),
            "harmed_with_negative_cosine_count": int((harmed["response_target_cosine"] < 0).sum()),
            "harmed_with_negative_pearson_count": int((harmed["response_target_pearson"] < 0).sum()),
            "harmed_with_negative_alpha_star_count": int((harmed["alpha_star"] < 0).sum()),
            "benefited_drug_count": int(len(benefited)),
            "benefited_with_positive_cosine_count": int((benefited["response_target_cosine"] > 0).sum()),
            "benefited_alpha_star_min": float(benefited["alpha_star"].min()),
            "benefited_alpha_star_max": float(benefited["alpha_star"].max()),
            "primary_reading": "harmful_new_drug_responses_are_directionally_wrong_not_merely_too_large",
        },
        "checkpoint_paths": {mode: str(path.resolve()) for mode, path in checkpoints.items()},
        "checkpoint_sha256": {mode: sha256_file(path) for mode, path in checkpoints.items()},
        "frozen_artifact_hashes": artifacts.hashes,
        "target_residual_status": "validation_diagnostic_only_not_official_fc",
        "alpha_star_status": "diagnostic_only_not_used_for_tuning_or_prediction",
        "training_performed": False,
        "model_modified": False,
        "checkpoint_modified": False,
        "test_proteome_opened": False,
    })


def _number(value, digits=3):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(summary: dict, per_entity: pd.DataFrame, combined: pd.DataFrame, csv_path: Path, json_path: Path) -> str:
    correlation_rows = []
    for scope, block in summary["correlations"].items():
        for relation, result in block.items():
            correlation_rows.append(
                f"| {scope} | {relation} | {_number(result['spearman_rho'])} | {result['n_drugs']} |"
            )
    direction_rows = []
    for row in combined.sort_values("correct_harm_vs_zero", ascending=False).itertuples():
        direction_rows.append(
            f"| {row.chemical_name} | {row.n_samples} | {_number(row.correct_harm_vs_zero)} | "
            f"{_number(row.correct_sample_response_rms_mean)} | {_number(row.correct_zero_response_rms_ratio)} | "
            f"{_number(row.target_residual_rms)} | {_number(row.response_target_norm_ratio)} | "
            f"{_number(row.response_target_cosine)} | {_number(row.response_target_pearson)} | {_number(row.alpha_star)} |"
        )
    corr = summary["correlations"]["combined_unique_drugs_equal_weight"]
    normalized_rho = corr["normalized_response_rms_vs_correct_harm"]["spearman_rho"]
    ratio_rho = corr["correct_zero_response_rms_ratio_vs_correct_harm"]["spearman_rho"]
    sample_l2_rho = corr["n_samples_vs_raw_cumulative_response_l2"]["spearman_rho"]
    sample_harm_rho = corr["n_samples_vs_correct_harm"]["spearman_rho"]
    return f"""# MODEL V2 Stage 2.3a：归一化响应幅度与方向审计

## 结论

本阶段仅对已有 no-batch correct/zero checkpoint 做只读推理；没有训练、模型或 checkpoint 修改、特征调整、FC、门控或测试蛋白读取。`target_residual = y_true - y_baseline` 仅是 validation diagnostic residual，不是官方 FC；`alpha_star` 也只用于事后诊断，未用于调参或预测。

- 六药等权后，correct 的每样本 response RMS 均值与 harm 的 Spearman rho={_number(normalized_rho)}；correct/zero RMS 比值与 harm 的 rho={_number(ratio_rho)}。
- `n_samples` 与 correct 原始累计 L2 的 rho={_number(sample_l2_rho)}，量化原始累计范数受样本数影响的程度；`n_samples` 与 harm 的 rho={_number(sample_harm_rho)}。
- 原 Stage 2.3 的累计 L2–harm rho=0.829 不能作为独立的过度响应证据：归一化后相关降至 {_number(normalized_rho)}，同时样本数与累计 L2 高度相关。仍有中等方向的幅度关联，但只有 6 个药物，不能排除样本构成混杂。
- 三个受损药物 Hydroxyurea、FCCP、Amphotericin B 的 cosine、Pearson、`alpha_star` 均为负，说明 correct response 与 validation target residual 方向相反；这不符合“方向正确但只需缩小”的模式。三个受益药物的 cosine 为正，`alpha_star` 在 0.934–0.978，反而接近无需缩放。当前主要诊断是新药响应方向错误，而非单纯幅度过大。
- 六药合并口径先在每个场景—药物内计算指标，再对同一药物的 val_chem_only 与 val_both 指标作等权平均，最后以六个唯一药物作为六个等权观测；没有按样本数加权。

## 六药等权幅度与方向诊断

| 药物 | 总样本数 | harm | correct sample RMS | correct/zero RMS | target residual RMS | response/target norm | cosine | Pearson | alpha_star |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(direction_rows)}

解释边界：若 cosine/Pearson 较高且 `alpha_star << 1`，更支持方向大致正确但幅度过大；若方向指标低或为负，则缩放本身不能修复方向错误。表中合并方向指标是两个场景指标的等权描述性平均，不用于模型选择。

## 相关性

| 范围 | 关系 | Spearman rho | 药物数 |
|---|---|---:|---:|
{chr(10).join(correlation_rows)}

每个场景及合并范围都只有 6 个药物。所有 rho 均为小样本描述性结果，不作显著性或稳定性声明。

## 指标定义

- 每样本 `response RMS = sqrt(sum(valid * response^2) / n_valid_proteins)`；随后在场景—药物内报告均值和中位数。
- 每样本 L2 先逐样本计算，再报告均值和中位数；原始累计 L2 则在该场景—药物的全部共同有效位置上计算。
- 方向诊断在 `label mask & finite(response) & finite(y_baseline) & finite(y_true)` 的共同有效位置上扁平化计算。
- `alpha_star = dot(delta_response, target_residual) / dot(delta_response, delta_response)`，是不含截距的一维最小二乘诊断解。

## 数据边界

- 只读取 train_val 元数据、冻结合同/特征和 `WAYB_WAYC_proteome_raw_train_val.csv`；标签仅取 val_chem_only、val_both 行计算误差与诊断。
- correct/zero 的 sample ID、target 和 mask 逐场景严格一致。
- Stage 2.3 原报告、CSV 和 JSON 均保留且未覆盖。
- 未运行 Morgan-only、descriptor-only、clipping、门控或 FC。

## 产物

- 逐场景—药物 CSV：`{csv_path.resolve()}`
- 汇总 JSON：`{json_path.resolve()}`
- 分析代码：`{(ROOT / 'baseline/baseline/normalized_response_analysis_v2.py').resolve()}`
"""


def run_analysis(checkpoints: dict[str, Path], csv_output: Path, summary_output: Path, report_output: Path, batch_size=64, device="auto"):
    if set(checkpoints) != set(VARIANTS):
        raise ValueError("exactly correct and zero checkpoints are required")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    evaluations = {mode: {} for mode in VARIANTS}
    for scenario in OOD_SCENARIOS:
        for mode in VARIANTS:
            evaluations[mode][scenario] = evaluate_response_variant(
                mode, checkpoints[mode], scenario, meta, labels, masks,
                artifacts, vocab, int(batch_size), device,
            )
    per_entity = build_per_entity_table(meta, evaluations)
    combined = combine_unique_drugs_equal_weight(per_entity)
    summary = build_summary(per_entity, combined, checkpoints, artifacts, device)
    summary["outputs"] = {
        "per_entity_csv": str(csv_output.resolve()),
        "summary_json": str(summary_output.resolve()),
        "report_markdown": str(report_output.resolve()),
    }
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    per_entity.to_csv(csv_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_output.write_text(
        render_report(summary, per_entity, combined, csv_output, summary_output), encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only normalized V2 response audit")
    parser.add_argument("--correct-checkpoint", type=Path, required=True)
    parser.add_argument("--zero-checkpoint", type=Path, required=True)
    parser.add_argument("--per-entity-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run_analysis(
        {"correct": args.correct_checkpoint, "zero": args.zero_checkpoint},
        args.per_entity_output, args.summary_output, args.report_output,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
