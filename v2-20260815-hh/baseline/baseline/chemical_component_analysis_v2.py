"""Stage 2.4 read-only comparison of frozen chemical component checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemical_ood_analysis_v2 import CHEMICAL_COLUMN, OOD_SCENARIOS, _build_model, _json_ready
from .evaluation_v2 import VAL_SCENARIOS, masked_rmse
from .normalized_response_analysis_v2 import (
    ResponseEvaluation,
    response_direction_diagnostics,
    summarize_response_amplitude,
)
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
    read_json,
    sha256_file,
)


COMPONENT_MODES = ("full", "morgan_only", "descriptor_only", "none")
METRIC_NAMES = (
    "rmse", "mae", "global_r2", "median_per_sample_pcc", "median_per_sample_r2",
    "median_per_protein_pcc", "median_per_protein_r2",
)


def resolve_checkpoint_components(payload: dict) -> str:
    model = payload["config"]["model"]
    if "chemical_feature_components" in model:
        components = model["chemical_feature_components"]
    else:
        components = "none" if model["chemical_mode"] == "zero" else "full"
    if components not in COMPONENT_MODES:
        raise ValueError(f"invalid checkpoint chemical components: {components}")
    return components


def evaluate_component_checkpoint(
    expected_components: str,
    checkpoint_path: Path,
    scenario: str,
    meta,
    labels,
    masks,
    artifacts,
    vocab,
    batch_size: int,
    device: str,
    *,
    allow_experimental_nonofficial_parity_fc: bool = False,
) -> ResponseEvaluation:
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    components = resolve_checkpoint_components(payload)
    if components != expected_components:
        raise ValueError(f"checkpoint components {components} != {expected_components}")
    model_config = payload["config"]["model"]
    entity_mode = model_config["chemical_mode"]
    if entity_mode not in {"correct", "zero"}:
        raise ValueError("Stage 2.4 accepts correct or legacy-zero entity modes only")
    ids = meta.index[meta["split_final"].eq(scenario)]
    seed = int(payload["seed"])
    variant = make_chemical_feature_variant(artifacts, entity_mode, seed)
    batch = build_batch(
        meta, ids, artifacts, vocab, chemical_mode=entity_mode,
        chemical_feature_components=components, genome_mode="correct", seed=seed,
        chemical_variant=variant,
    )
    target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
    mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
    loader = make_loader(batch, target, mask, batch_size=int(batch_size), shuffle=False)
    model = _build_model(
        payload,
        artifacts,
        vocab,
        device,
        allow_experimental_nonofficial_parity_fc=allow_experimental_nonofficial_parity_fc,
    )
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


def assert_all_mode_alignment(evaluations: dict[str, ResponseEvaluation]):
    reference = evaluations["full"]
    for mode in COMPONENT_MODES[1:]:
        candidate = evaluations[mode]
        if reference.sample_ids != candidate.sample_ids:
            raise ValueError(f"{mode} sample IDs differ from full")
        if not np.array_equal(reference.mask, candidate.mask):
            raise ValueError(f"{mode} masks differ from full")
        if not np.array_equal(reference.target, candidate.target, equal_nan=True):
            raise ValueError(f"{mode} targets differ from full")
        if not np.array_equal(reference.baseline, candidate.baseline):
            raise ValueError(f"{mode} baseline predictions differ from full")


def build_amplitude_table(evaluations) -> pd.DataFrame:
    records = []
    for scenario in VAL_SCENARIOS:
        assert_all_mode_alignment({mode: evaluations[mode][scenario] for mode in COMPONENT_MODES})
        for mode in COMPONENT_MODES:
            amplitude = summarize_response_amplitude(
                evaluations[mode][scenario].response, evaluations[mode][scenario].mask,
            )
            records.append({"mode": mode, "scenario": scenario, **amplitude})
    return pd.DataFrame(records)


def build_per_entity_table(meta, evaluations) -> pd.DataFrame:
    records = []
    for scenario in OOD_SCENARIOS:
        per_mode = {mode: evaluations[mode][scenario] for mode in COMPONENT_MODES}
        assert_all_mode_alignment(per_mode)
        ids = pd.Index(per_mode["full"].sample_ids)
        chemicals = meta.loc[ids, CHEMICAL_COLUMN].astype(str).to_numpy()
        for chemical in sorted(np.unique(chemicals)):
            positions = np.flatnonzero(chemicals == chemical)
            for mode in COMPONENT_MODES:
                evaluation = per_mode[mode]
                amplitude = summarize_response_amplitude(
                    evaluation.response[positions], evaluation.mask[positions],
                )
                direction = response_direction_diagnostics(
                    evaluation.response[positions], evaluation.target[positions],
                    evaluation.baseline[positions], evaluation.mask[positions],
                )
                records.append({
                    "mode": mode,
                    "scenario": scenario,
                    "chemical_name": chemical,
                    "n_samples": int(len(positions)),
                    "rmse": masked_rmse(
                        evaluation.target[positions], evaluation.prediction[positions],
                        evaluation.mask[positions],
                    ),
                    **amplitude,
                    **direction,
                    "target_residual_status": "validation_diagnostic_only_not_official_fc",
                })
    table = pd.DataFrame(records).sort_values(["scenario", "chemical_name", "mode"]).reset_index(drop=True)
    if len(table) != 48 or table.duplicated(["mode", "scenario", "chemical_name"]).any():
        raise ValueError("Stage 2.4 per-entity table must have 48 unique rows")
    return table


def combine_entity_directions_equal_scenarios(per_entity: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "rmse", "sample_response_rms_mean", "sample_response_rms_median",
        "predicted_response_rms", "target_residual_rms", "response_target_norm_ratio",
        "response_target_cosine", "response_target_pearson", "alpha_star",
    )
    records = []
    for (mode, chemical), group in per_entity.groupby(["mode", "chemical_name"], sort=True):
        if set(group["scenario"]) != set(OOD_SCENARIOS) or len(group) != 2:
            raise ValueError("each mode-drug combination must contain both OOD scenarios")
        record = {
            "mode": mode, "chemical_name": chemical,
            "n_samples_total": int(group["n_samples"].sum()),
            "aggregation": "equal_mean_of_val_chem_only_and_val_both",
        }
        for column in columns:
            record[column] = float(group[column].mean())
        records.append(record)
    combined = pd.DataFrame(records)
    if len(combined) != 24 or combined.duplicated(["mode", "chemical_name"]).any():
        raise ValueError("combined table must contain four modes x six drugs")
    return combined


def load_training_results(run_dirs: dict[str, Path]) -> tuple[dict, dict]:
    summaries, histories = {}, {}
    stage2_2_comparison_path = ROOT / "reports/model_v2_stage2/stage2_2_chemical_comparison.json"
    stage2_2_comparison = read_json(stage2_2_comparison_path)
    for mode, directory in run_dirs.items():
        summaries[mode] = read_json(directory / "training_summary.json")
        histories[mode] = read_json(directory / "training_history.json")
        if summaries[mode]["test_proteome_opened"] is not False:
            raise ValueError(f"{mode} summary does not attest test-proteome boundary")
        if "stage_b_initial_chemical_response_sha256" not in summaries[mode]:
            stage_a_path = directory / "stage_a_best.pt"
            if not stage_a_path.exists():
                fixed = summaries[mode]["stage_a"].get("fixed_checkpoint")
                if not fixed:
                    raise ValueError(f"{mode} has no auditable Stage A checkpoint")
                stage_a_path = Path(fixed["path"])
            summaries[mode]["stage_b_initial_chemical_response_sha256"] = chemical_response_state_sha256(
                stage_a_path,
            )
        if "chemical_encoder_parameter_change_l2" not in summaries[mode]:
            if mode != "full":
                raise ValueError(f"{mode} lacks chemical encoder parameter-change audit")
            summaries[mode]["chemical_encoder_parameter_change_l2"] = stage2_2_comparison["variants"]["correct"][
                "chemical_encoder_parameter_change_l2"
            ]
            summaries[mode]["stage2_4_compatibility_backfill"] = {
                "field": "chemical_encoder_parameter_change_l2",
                "source": str(stage2_2_comparison_path.resolve()),
                "source_analysis": stage2_2_comparison["analysis"],
            }
    hashes = {summary["stage_b_initial_chemical_response_sha256"] for summary in summaries.values()}
    if len(hashes) != 1:
        raise ValueError("Stage B initial chemical+response hashes are not identical")
    return summaries, histories


def chemical_response_state_sha256(checkpoint_path: Path) -> str:
    """Reproduce the training-time module hash from a frozen Stage A state."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    digest = hashlib.sha256()
    for module in ("chemical_encoder", "response_branch"):
        prefix = f"{module}."
        items = [(key[len(prefix):], value) for key, value in state.items() if key.startswith(prefix)]
        if not items:
            raise ValueError(f"checkpoint lacks {module} parameters")
        for name, value in sorted(items):
            value = value.detach().cpu().contiguous()
            digest.update(f"{module}.{name}|{tuple(value.shape)}|{value.dtype}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _mode_record(mode, summary, amplitude_table):
    component = summary.get("chemical_feature_components", mode)
    if component != mode:
        raise ValueError(f"summary component mismatch for {mode}")
    amplitude = amplitude_table.loc[amplitude_table["mode"].eq(mode)].set_index("scenario")
    return {
        "stage_b_best_epoch": int(summary["stage_b"]["best_epoch"]),
        "best_macro_huber": float(summary["stage_b"]["best_monitor"]),
        "chemical_encoder_parameter_change_l2": float(summary["chemical_encoder_parameter_change_l2"]),
        "stage_b_initial_chemical_response_sha256": summary["stage_b_initial_chemical_response_sha256"],
        "stage_a_fixed_checkpoint": summary["stage_a"].get("fixed_checkpoint"),
        "scenario_metrics": {
            scenario: {name: summary["validation_scenarios"][scenario][name] for name in METRIC_NAMES}
            for scenario in VAL_SCENARIOS
        },
        "scenario_normalized_response": {
            scenario: {
                "sample_response_rms_mean": float(amplitude.loc[scenario, "sample_response_rms_mean"]),
                "sample_response_rms_median": float(amplitude.loc[scenario, "sample_response_rms_median"]),
                "raw_cumulative_response_l2": float(amplitude.loc[scenario, "raw_cumulative_response_l2"]),
            }
            for scenario in VAL_SCENARIOS
        },
    }


def build_summary(training_summaries, amplitude, per_entity, combined, checkpoints, run_dirs, artifacts, device):
    modes = {
        mode: _mode_record(mode, training_summaries[mode], amplitude)
        for mode in COMPONENT_MODES
    }
    macro = {mode: modes[mode]["best_macro_huber"] for mode in COMPONENT_MODES}
    ood_rmse = {
        mode: float(np.mean([
            modes[mode]["scenario_metrics"][scenario]["rmse"] for scenario in OOD_SCENARIOS
        ]))
        for mode in COMPONENT_MODES
    }
    failed = {"Hydroxyurea", "FCCP", "Amphotericin B"}
    benefited = {"Pentamidine isethionate", "Raloxifene hydrochloride", "Sulfometuron methyl"}
    direction = {}
    for mode in COMPONENT_MODES:
        rows = combined.loc[combined["mode"].eq(mode)].set_index("chemical_name")
        direction[mode] = {
            "failed_drug_cosines": {drug: float(rows.loc[drug, "response_target_cosine"]) for drug in sorted(failed)},
            "failed_drug_alpha_star": {drug: float(rows.loc[drug, "alpha_star"]) for drug in sorted(failed)},
            "benefited_drug_cosines": {drug: float(rows.loc[drug, "response_target_cosine"]) for drug in sorted(benefited)},
            "benefited_positive_direction_count": int((rows.loc[sorted(benefited), "response_target_cosine"] > 0).sum()),
        }
    judgement = {
        "macro_huber_order_best_to_worst": sorted(macro, key=macro.get),
        "ood_mean_rmse_order_best_to_worst": sorted(ood_rmse, key=ood_rmse.get),
        "morgan_only_better_than_full_macro": macro["morgan_only"] < macro["full"],
        "descriptor_only_better_than_full_macro": macro["descriptor_only"] < macro["full"],
        "morgan_only_better_than_none_macro": macro["morgan_only"] < macro["none"],
        "descriptor_only_better_than_none_macro": macro["descriptor_only"] < macro["none"],
        "ood_mean_rmse": ood_rmse,
        "rule_1_morgan_repairs_all_three_failed_directions": all(
            direction["morgan_only"]["failed_drug_cosines"][drug] > 0 for drug in failed
        ),
        "morgan_failed_direction_repair_count": int(sum(
            direction["morgan_only"]["failed_drug_cosines"][drug] > 0 for drug in failed
        )),
        "descriptor_failed_direction_repair_count": int(sum(
            direction["descriptor_only"]["failed_drug_cosines"][drug] > 0 for drug in failed
        )),
        "rule_3_both_single_components_better_than_full_macro": (
            macro["morgan_only"] < macro["full"] and macro["descriptor_only"] < macro["full"]
        ),
        "rule_4_both_not_better_than_none_macro": (
            macro["morgan_only"] >= macro["none"] and macro["descriptor_only"] >= macro["none"]
        ),
        "primary_reading": (
            "aggregate_fusion_conflict_with_morgan_signal_for_hydroxyurea_and_fccp; "
            "amphotericin_and_ood_generalization_remain_unresolved"
        ),
        "note": "direction repair is assessed per drug; no validation-derived change was applied",
    }
    resolved_configs = {mode: read_json(run_dirs[mode] / "resolved_config.json") for mode in COMPONENT_MODES}
    signatures = {}
    for mode, config in resolved_configs.items():
        signatures[mode] = {
            "seed": config["training"]["seed"],
            "batch_size": config["training"]["batch_size"],
            "weight_decay": config["training"]["weight_decay"],
            "num_workers": config["training"]["num_workers"],
            "stage_b": config["training"]["stage_b"],
            "model_dimensions": {
                key: config["model"][key] for key in ("latent_dim", "protein_rank", "dropout")
            },
            "genome_mode": config["model"]["genome_mode"],
            "similarity_enabled": config["model"]["similarity_enabled"],
            "batch_enabled": config["model"]["batch_enabled"],
            "loss": config["loss"],
        }
    signature_hashes = {
        mode: hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        for mode, signature in signatures.items()
    }
    stage_a_paths = {}
    for mode, summary in training_summaries.items():
        fixed = summary["stage_a"].get("fixed_checkpoint")
        stage_a_paths[mode] = Path(fixed["path"]) if fixed else run_dirs[mode] / "stage_a_best.pt"
    stage_a_hashes = {mode: sha256_file(path) for mode, path in stage_a_paths.items()}
    fairness_audit = {
        "all_stage_b_training_signatures_identical": len(set(signature_hashes.values())) == 1,
        "stage_b_training_signature_sha256": signature_hashes,
        "all_stage_a_checkpoint_hashes_identical": len(set(stage_a_hashes.values())) == 1,
        "stage_a_checkpoint_sha256": stage_a_hashes,
        "all_stage_b_initial_chemical_response_hashes_identical": len({
            result["stage_b_initial_chemical_response_sha256"] for result in modes.values()
        }) == 1,
        "data_order_basis": "same split_final=train treatment IDs, seed and DataLoader implementation",
    }
    if not all((
        fairness_audit["all_stage_b_training_signatures_identical"],
        fairness_audit["all_stage_a_checkpoint_hashes_identical"],
        fairness_audit["all_stage_b_initial_chemical_response_hashes_identical"],
    )):
        raise ValueError("Stage 2.4 fairness audit failed")
    return _json_ready({
        "status": "PASS",
        "analysis": "stage2_4_chemical_feature_component_ablation",
        "device": device,
        "modes": modes,
        "per_entity_records": per_entity.to_dict("records"),
        "combined_equal_scenario_direction_records": combined.to_dict("records"),
        "focus_direction": direction,
        "judgement_inputs": judgement,
        "fairness_audit": fairness_audit,
        "checkpoint_paths": {mode: str(path.resolve()) for mode, path in checkpoints.items()},
        "checkpoint_sha256": {mode: sha256_file(path) for mode, path in checkpoints.items()},
        "run_directories": {mode: str(path.resolve()) for mode, path in run_dirs.items()},
        "shared_stage_b_initial_hash": next(iter({s["stage_b_initial_chemical_response_sha256"] for s in training_summaries.values()})),
        "frozen_artifact_hashes": artifacts.hashes,
        "training_scope": "only_morgan_only_and_descriptor_only_stage_b_were_newly_trained",
        "stage_a_retrained": False,
        "test_proteome_opened": False,
        "fc_enabled": False,
        "similarity_enabled": False,
        "batch_enabled": False,
    })


def _number(value, digits=3):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(summary, per_entity, combined, csv_path, json_path):
    training_rows = []
    scenario_rows = []
    for mode in COMPONENT_MODES:
        result = summary["modes"][mode]
        training_rows.append(
            f"| {mode} | {result['stage_b_best_epoch']} | {_number(result['best_macro_huber'], 6)} | "
            f"{_number(result['chemical_encoder_parameter_change_l2'], 3)} |"
        )
        for scenario in VAL_SCENARIOS:
            metrics = result["scenario_metrics"][scenario]
            amplitude = result["scenario_normalized_response"][scenario]
            scenario_rows.append(
                f"| {mode} | {scenario} | {_number(metrics['rmse'])} | {_number(metrics['mae'])} | "
                f"{_number(metrics['global_r2'])} | {_number(metrics['median_per_sample_pcc'])} | "
                f"{_number(metrics['median_per_sample_r2'])} | {_number(metrics['median_per_protein_pcc'])} | "
                f"{_number(metrics['median_per_protein_r2'])} | {_number(amplitude['sample_response_rms_mean'])} |"
            )
    direction_rows = []
    for row in combined.sort_values(["chemical_name", "mode"]).itertuples():
        direction_rows.append(
            f"| {row.chemical_name} | {row.mode} | {_number(row.rmse)} | {_number(row.sample_response_rms_mean)} | "
            f"{_number(row.response_target_cosine)} | {_number(row.response_target_pearson)} | {_number(row.alpha_star)} |"
        )
    judgement = summary["judgement_inputs"]
    return f"""# MODEL V2 Stage 2.4：化学特征组分消融

## 结论

本阶段只新增 Morgan-only 与 descriptor-only 两次 Stage B 训练；full、none 复用既有 checkpoint。两次新训练均加载同一个 Stage A checkpoint，Stage A 实际训练 epoch=0，四模式 Stage B 初始 chemical encoder+response hash 均为 `{summary['shared_stage_b_initial_hash']}`。

- Macro Huber 从优到劣：{', '.join(judgement['macro_huber_order_best_to_worst'])}。
- val_chem_only/val_both 平均 RMSE 从优到劣：{', '.join(judgement['ood_mean_rmse_order_best_to_worst'])}。
- Morgan-only 相对 full 的 macro 更优：{judgement['morgan_only_better_than_full_macro']}；相对 none 更优：{judgement['morgan_only_better_than_none_macro']}。
- Descriptor-only 相对 full 的 macro 更优：{judgement['descriptor_only_better_than_full_macro']}；相对 none 更优：{judgement['descriptor_only_better_than_none_macro']}。
- 判定规则 1 未完整满足：Morgan-only 只修复三个失败药物中的 {judgement['morgan_failed_direction_repair_count']} 个（Hydroxyurea、FCCP），Amphotericin B 仍为负方向且 RMSE 恶化。
- 判定规则 3 在总体 macro 上满足：两种单组分都优于 full，说明当前 Morgan–descriptor 融合存在冲突；但两种单组分的 OOD 两场景平均 RMSE 都没有优于 none，因此不能宣称单一组分已经解决新药迁移。
- Descriptor-only 修复失败方向数为 {judgement['descriptor_failed_direction_repair_count']}，但保留三个原受益药物的正方向；Morgan-only 则明显损害 Pentamidine。综合结论是：Morgan 对 Hydroxyurea/FCCP 含有可迁移方向信号，descriptor/融合对这两药有负作用；Amphotericin 与整体 OOD 泛化仍指向表征到 response 的映射问题。

## 训练与模型选择

| 模式 | Stage B 最佳 epoch | 最佳 macro Huber | chemical encoder 参数变化 L2 |
|---|---:|---:|---:|
{chr(10).join(training_rows)}

## 四场景完整指标与归一化 response RMS

| 模式 | 场景 | RMSE | MAE | Global R2 | sample PCC med | sample R2 med | protein PCC med | protein R2 med | response RMS mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

## 六个新药方向诊断（两个 OOD 场景等权平均）

| 药物 | 模式 | RMSE | response RMS | cosine | Pearson | alpha_star |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(direction_rows)}

CSV 保留 val_chem_only 与 val_both 的逐场景结果；上表仅为两个场景的等权描述性汇总。`target residual` 不是官方 FC，方向指标和 alpha_star 只用于归因，未用于模型选择或调参。

## 数据与训练边界

- `chemical_mode` 与 `chemical_feature_components` 是两个独立字段；本阶段新训练均为 entity mode=correct。
- Morgan-only 严格清零 descriptors 与其 mask；descriptor-only 严格清零 Morgan 与其 mask；identity quality flags 保留。
- seed=20260814、no-batch、genome=correct、Huber-only、similarity=false；未修改 scaler、未 clipping。
- 未训练 full/none，未重训 Stage A，未读取测试蛋白真值，未运行 FC、门控、新架构或其他消融。

## 产物

- 逐药物 CSV：`{csv_path.resolve()}`
- 汇总 JSON：`{json_path.resolve()}`
- 分析代码：`{(ROOT / 'baseline/baseline/chemical_component_analysis_v2.py').resolve()}`
"""


def run_analysis(checkpoints, run_dirs, csv_output, summary_output, report_output, batch_size=64, device="auto"):
    if set(checkpoints) != set(COMPONENT_MODES) or set(run_dirs) != set(COMPONENT_MODES):
        raise ValueError("all four component modes are required")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    evaluations = {mode: {} for mode in COMPONENT_MODES}
    for scenario in VAL_SCENARIOS:
        for mode in COMPONENT_MODES:
            evaluations[mode][scenario] = evaluate_component_checkpoint(
                mode, checkpoints[mode], scenario, meta, labels, masks,
                artifacts, vocab, int(batch_size), device,
            )
    amplitude = build_amplitude_table(evaluations)
    per_entity = build_per_entity_table(meta, evaluations)
    combined = combine_entity_directions_equal_scenarios(per_entity)
    training_summaries, _ = load_training_results(run_dirs)
    summary = build_summary(
        training_summaries, amplitude, per_entity, combined,
        checkpoints, run_dirs, artifacts, device,
    )
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
    report_output.write_text(render_report(summary, per_entity, combined, csv_output, summary_output), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only Stage 2.4 chemical component comparison")
    for mode in COMPONENT_MODES:
        parser.add_argument(f"--{mode.replace('_', '-')}-run-dir", type=Path, required=True)
    parser.add_argument("--per-entity-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_dirs = {
        mode: getattr(args, f"{mode}_run_dir") for mode in COMPONENT_MODES
    }
    checkpoints = {mode: directory / "stage_b_best.pt" for mode, directory in run_dirs.items()}
    summary = run_analysis(
        checkpoints, run_dirs, args.per_entity_output, args.summary_output,
        args.report_output, batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
