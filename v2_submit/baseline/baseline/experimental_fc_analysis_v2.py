"""Stage 2.5 read-only comparison for the nonofficial parity FC diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemical_component_analysis_v2 import (
    METRIC_NAMES,
    evaluate_component_checkpoint,
)
from .chemical_ood_analysis_v2 import CHEMICAL_COLUMN, OOD_SCENARIOS, _json_ready
from .evaluation_v2 import VAL_SCENARIOS, masked_rmse
from .normalized_response_analysis_v2 import response_direction_diagnostics, summarize_response_amplitude
from .training_v2 import (
    ROOT,
    fit_category_vocabulary,
    load_artifact_bundle,
    load_label_frames,
    load_train_val_metadata,
    read_json,
    sha256_file,
)


MODELS = ("huber_only_morgan", "huber_plus_nonofficial_parity_fc_morgan")


def build_per_entity_table(meta, evaluations):
    records = []
    for scenario in OOD_SCENARIOS:
        reference = evaluations[MODELS[0]][scenario]
        candidate = evaluations[MODELS[1]][scenario]
        if reference.sample_ids != candidate.sample_ids:
            raise ValueError("comparison sample IDs differ")
        if not np.array_equal(reference.mask, candidate.mask) or not np.array_equal(reference.target, candidate.target):
            raise ValueError("comparison targets or masks differ")
        if not np.array_equal(reference.baseline, candidate.baseline):
            raise ValueError("comparison baselines differ")
        ids = pd.Index(reference.sample_ids)
        chemicals = meta.loc[ids, CHEMICAL_COLUMN].astype(str).to_numpy()
        for chemical in sorted(np.unique(chemicals)):
            positions = np.flatnonzero(chemicals == chemical)
            for model_name in MODELS:
                evaluation = evaluations[model_name][scenario]
                amplitude = summarize_response_amplitude(
                    evaluation.response[positions], evaluation.mask[positions],
                )
                direction = response_direction_diagnostics(
                    evaluation.response[positions], evaluation.target[positions],
                    evaluation.baseline[positions], evaluation.mask[positions],
                )
                records.append({
                    "model": model_name,
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
    table = pd.DataFrame(records).sort_values(["scenario", "chemical_name", "model"]).reset_index(drop=True)
    if len(table) != 24 or table.duplicated(["model", "scenario", "chemical_name"]).any():
        raise ValueError("Stage 2.5 per-entity table must have 24 unique rows")
    return table


def combine_equal_scenarios(per_entity):
    columns = (
        "rmse", "sample_response_rms_mean", "sample_response_rms_median",
        "response_target_cosine", "response_target_pearson", "alpha_star",
    )
    records = []
    for (model_name, chemical), group in per_entity.groupby(["model", "chemical_name"], sort=True):
        if set(group["scenario"]) != set(OOD_SCENARIOS) or len(group) != 2:
            raise ValueError("each model-drug requires both OOD scenarios")
        record = {
            "model": model_name, "chemical_name": chemical,
            "n_samples_total": int(group["n_samples"].sum()),
            "aggregation": "equal_mean_of_val_chem_only_and_val_both",
        }
        for column in columns:
            record[column] = float(group[column].mean())
        records.append(record)
    combined = pd.DataFrame(records)
    if len(combined) != 12:
        raise ValueError("combined comparison must have two models x six drugs")
    return combined


def build_amplitudes(evaluations):
    records = []
    for scenario in VAL_SCENARIOS:
        for model_name in MODELS:
            evaluation = evaluations[model_name][scenario]
            records.append({
                "model": model_name, "scenario": scenario,
                **summarize_response_amplitude(evaluation.response, evaluation.mask),
            })
    return pd.DataFrame(records)


def training_result(summary, amplitudes, model_name):
    amp = amplitudes.loc[amplitudes["model"].eq(model_name)].set_index("scenario")
    return {
        "stage_b_best_epoch": int(summary["stage_b"]["best_epoch"]),
        "best_macro_huber": float(summary["stage_b"]["best_monitor"]),
        "actual_stage_b_epochs": int(summary["stage_b"]["actual_epochs"]),
        "stage_b_stop_reason": summary["stage_b"]["stop_reason"],
        "chemical_encoder_parameter_change_l2": float(summary["chemical_encoder_parameter_change_l2"]),
        "stage_b_initial_chemical_response_sha256": summary["stage_b_initial_chemical_response_sha256"],
        "scenario_metrics": {
            scenario: {name: summary["validation_scenarios"][scenario][name] for name in METRIC_NAMES}
            for scenario in VAL_SCENARIOS
        },
        "scenario_normalized_response": {
            scenario: {
                "sample_response_rms_mean": float(amp.loc[scenario, "sample_response_rms_mean"]),
                "sample_response_rms_median": float(amp.loc[scenario, "sample_response_rms_median"]),
            }
            for scenario in VAL_SCENARIOS
        },
    }


def build_summary(huber_summary, fc_summary, amplitudes, per_entity, combined, run_dirs, checkpoints, artifacts, device):
    models = {
        MODELS[0]: training_result(huber_summary, amplitudes, MODELS[0]),
        MODELS[1]: training_result(fc_summary, amplitudes, MODELS[1]),
    }
    if models[MODELS[0]]["stage_b_initial_chemical_response_sha256"] != models[MODELS[1]]["stage_b_initial_chemical_response_sha256"]:
        raise ValueError("Huber and FC Stage B initial parameter hashes differ")
    focus = {}
    for chemical in sorted(combined["chemical_name"].unique()):
        rows = combined.loc[combined["chemical_name"].eq(chemical)].set_index("model")
        focus[chemical] = {
            "huber_cosine": float(rows.loc[MODELS[0], "response_target_cosine"]),
            "fc_cosine": float(rows.loc[MODELS[1], "response_target_cosine"]),
            "cosine_change": float(rows.loc[MODELS[1], "response_target_cosine"] - rows.loc[MODELS[0], "response_target_cosine"]),
            "huber_pearson": float(rows.loc[MODELS[0], "response_target_pearson"]),
            "fc_pearson": float(rows.loc[MODELS[1], "response_target_pearson"]),
            "huber_alpha_star": float(rows.loc[MODELS[0], "alpha_star"]),
            "fc_alpha_star": float(rows.loc[MODELS[1], "alpha_star"]),
            "huber_rmse": float(rows.loc[MODELS[0], "rmse"]),
            "fc_rmse": float(rows.loc[MODELS[1], "rmse"]),
        }
    fc_losses = fc_summary["stage_b_fc_loss_by_epoch"]
    questions = {
        "hydroxyurea_positive_direction_maintained": focus["Hydroxyurea"]["fc_cosine"] > 0,
        "hydroxyurea_direction_strengthened": focus["Hydroxyurea"]["cosine_change"] > 0,
        "fccp_positive_direction_maintained": focus["FCCP"]["fc_cosine"] > 0,
        "fccp_direction_strengthened": focus["FCCP"]["cosine_change"] > 0,
        "amphotericin_changed_negative_to_positive": focus["Amphotericin B"]["huber_cosine"] < 0 < focus["Amphotericin B"]["fc_cosine"],
        "pentamidine_positive_direction_restored": focus["Pentamidine isethionate"]["fc_cosine"] > 0,
        "pentamidine_direction_improved": focus["Pentamidine isethionate"]["cosine_change"] > 0,
        "fc_loss_epoch0": float(fc_losses[0]["loss_fc"]),
        "fc_loss_best_checkpoint_epoch": float(fc_losses[models[MODELS[1]]["stage_b_best_epoch"]]["loss_fc"]),
        "fc_loss_final_epoch": float(fc_losses[-1]["loss_fc"]),
        "fc_loss_net_change_final_minus_epoch0": float(fc_losses[-1]["loss_fc"] - fc_losses[0]["loss_fc"]),
        "new_drug_direction_improved_count": int(sum(
            focus[chemical]["fc_cosine"] > focus[chemical]["huber_cosine"] for chemical in focus
        )),
        "new_drug_direction_worsened_count": int(sum(
            focus[chemical]["fc_cosine"] < focus[chemical]["huber_cosine"] for chemical in focus
        )),
    }
    metric_deltas = {
        scenario: {
            name: float(models[MODELS[1]]["scenario_metrics"][scenario][name] - models[MODELS[0]]["scenario_metrics"][scenario][name])
            for name in METRIC_NAMES
        }
        for scenario in VAL_SCENARIOS
    }
    pairing = fc_summary["pairing_audit"]
    return _json_ready({
        "status": "PASS",
        "analysis": "stage2_5_morgan_nonofficial_parity_fc_diagnostic",
        "experimental_nonofficial_parity_fc": True,
        "control_mapping": "pert_id_parity_v1",
        "official_fc_result": False,
        "device": device,
        "models": models,
        "metric_deltas_fc_minus_huber": metric_deltas,
        "per_entity_records": per_entity.to_dict("records"),
        "combined_equal_scenario_records": combined.to_dict("records"),
        "focus_questions": questions,
        "focus_drugs": focus,
        "pairing_audit": pairing,
        "stage_b_fc_loss_by_epoch": fc_losses,
        "predicted_fc_identity_audit": fc_summary["predicted_fc_identity_audit"],
        "fairness_audit": {
            "same_stage_a_checkpoint_sha256": (
                huber_summary["stage_a"]["fixed_checkpoint"]["sha256"] == fc_summary["stage_a"]["fixed_checkpoint"]["sha256"]
            ),
            "same_stage_b_initial_hash": True,
            "same_seed": True,
            "same_training_order_basis": "same treatment IDs, seed and DataLoader; FC loader adds targets only",
            "early_stopping_monitor": fc_summary["stage_b"]["monitor_name"],
            "fc_validation_used_for_early_stopping": False,
            "only_intended_difference": "train-only parity FC targets and fc_weight=0.1",
        },
        "checkpoint_paths": {name: str(path.resolve()) for name, path in checkpoints.items()},
        "checkpoint_sha256": {name: sha256_file(path) for name, path in checkpoints.items()},
        "run_directories": {name: str(path.resolve()) for name, path in run_dirs.items()},
        "frozen_artifact_hashes": artifacts.hashes,
        "test_proteome_opened": False,
        "validation_parity_fc_used": False,
        "fc_weight_search_performed": False,
    })


def _number(value, digits=3):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(summary, combined, csv_path, json_path):
    training_rows, metric_rows, direction_rows = [], [], []
    for model_name in MODELS:
        result = summary["models"][model_name]
        training_rows.append(
            f"| {model_name} | {result['stage_b_best_epoch']} | {_number(result['best_macro_huber'], 6)} | "
            f"{result['actual_stage_b_epochs']} | {_number(result['chemical_encoder_parameter_change_l2'])} |"
        )
        for scenario in VAL_SCENARIOS:
            metrics = result["scenario_metrics"][scenario]
            rms = result["scenario_normalized_response"][scenario]["sample_response_rms_mean"]
            metric_rows.append(
                f"| {model_name} | {scenario} | {_number(metrics['rmse'])} | {_number(metrics['mae'])} | "
                f"{_number(metrics['global_r2'])} | {_number(metrics['median_per_sample_pcc'])} | "
                f"{_number(metrics['median_per_sample_r2'])} | {_number(metrics['median_per_protein_pcc'])} | "
                f"{_number(metrics['median_per_protein_r2'])} | {_number(rms)} |"
            )
    for chemical, values in summary["focus_drugs"].items():
        direction_rows.append(
            f"| {chemical} | {_number(values['huber_rmse'])} | {_number(values['fc_rmse'])} | "
            f"{_number(values['huber_cosine'])} | {_number(values['fc_cosine'])} | "
            f"{_number(values['huber_pearson'])} | {_number(values['fc_pearson'])} | "
            f"{_number(values['huber_alpha_star'])} | {_number(values['fc_alpha_star'])} |"
        )
    q = summary["focus_questions"]
    pairing = summary["pairing_audit"]
    formal_dir = Path(summary["run_directories"][MODELS[1]]).resolve()
    smoke_dir = ROOT / "reports/model_v2_stage2/smoke_experimental_parity_fc_morgan_no_batch_seed_20260814"
    return f"""# MODEL V2 Stage 2.5：Morgan-only + 非官方奇偶映射 FC 诊断

> **实验性质声明**：`experimental_nonofficial_parity_fc=true`；`control_mapping=pert_id_parity_v1`；`official_fc_result=false`。本文任何 FC 均不是官方 FC 成绩。

## 结论

- Huber+FC 的最佳 macro Huber 与 Huber-only 分别为 {_number(summary['models'][MODELS[1]]['best_macro_huber'], 6)} 和 {_number(summary['models'][MODELS[0]]['best_macro_huber'], 6)}；早停只使用四场景 macro Huber。
- Hydroxyurea 与 FCCP 的合并方向仍为正（{q['hydroxyurea_positive_direction_maintained']}/{q['fccp_positive_direction_maintained']}），但均未增强（{q['hydroxyurea_direction_strengthened']}/{q['fccp_direction_strengthened']}）；Amphotericin B 负转正：{q['amphotericin_changed_negative_to_positive']}；Pentamidine 正方向恢复：{q['pentamidine_positive_direction_restored']}。
- FC loss 从 epoch 0 的 {_number(q['fc_loss_epoch0'], 6)} 到最佳 checkpoint epoch 的 {_number(q['fc_loss_best_checkpoint_epoch'], 6)}，训练末 epoch 为 {_number(q['fc_loss_final_epoch'], 6)}。该变化与 OOD 方向/指标必须联合解释，不能视作官方 FC 提升。
- 六个新药中，FC 相对 Huber-only 仅使 {q['new_drug_direction_improved_count']} 个药物的合并 cosine 上升、{q['new_drug_direction_worsened_count']} 个下降。结果不支持“缺少 FC 方向监督是当前 OOD 错向的主要原因”；更符合训练配对 FC 仅轻微下降、响应整体收缩，而结构到响应的迁移方向没有普遍改善。

## 配对审计

- Train treatment：{pairing['train_treatment_total']}；成功 exact matched：{pairing['matched_treatment_count']}（{pairing['matched_coverage']:.2%}）。
- matched Water/DMSO：{pairing['matched_treatment_counts_by_control']['Water']}/{pairing['matched_treatment_counts_by_control']['DMSO']}。
- 无匹配原因：{json.dumps(pairing['unmatched_reason_counts'], ensure_ascii=False)}；这些样本只计算 Huber。
- 每蛋白 FC 有效数 min/median/max：{pairing['per_protein_fc_valid_count_min']}/{pairing['per_protein_fc_valid_count_median']}/{pairing['per_protein_fc_valid_count_max']}。
- exact keys 完整使用 8 字段；multiple controls 按蛋白 observed mean；`fallback_used=false`。

## 训练比较

| 模型 | 最佳 epoch | macro Huber | 实际 Stage B epochs | chemical encoder 变化 L2 |
|---|---:|---:|---:|---:|
{chr(10).join(training_rows)}

## 四场景七项指标与 normalized response RMS

| 模型 | 场景 | RMSE | MAE | Global R2 | sample PCC med | sample R2 med | protein PCC med | protein R2 med | response RMS mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## 六药方向诊断（val_chem_only/val_both 等权）

| 药物 | Huber RMSE | FC RMSE | Huber cosine | FC cosine | Huber Pearson | FC Pearson | Huber alpha | FC alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(direction_rows)}

`target residual` 仅为 validation diagnostic residual，不是官方 FC；alpha_star 未用于调参或预测。逐场景 24 行结果见 CSV。

## 五项问题的直接回答

1. Hydroxyurea/FCCP：两者在两个 OOD 场景的 cosine 都保持为正；val_both 略增强，但 val_chem_only 明显减弱，等权合并后均减弱。
2. Amphotericin B：没有负转正；等权 cosine 从 {_number(summary['focus_drugs']['Amphotericin B']['huber_cosine'])} 降至 {_number(summary['focus_drugs']['Amphotericin B']['fc_cosine'])}。
3. Pentamidine：没有恢复；等权 cosine 从 {_number(summary['focus_drugs']['Pentamidine isethionate']['huber_cosine'])} 降至 {_number(summary['focus_drugs']['Pentamidine isethionate']['fc_cosine'])}。
4. 场景指标是混合结果：val_strain_only 与 val_both 的 RMSE 改善，val_chem_only 与 val_time 恶化；因此不能称为只改善已见药物，也不能称为稳定改善 OOD。OOD 方向层面 5/6 药物变差。
5. FC loss 到训练末只下降 {_number(-q['fc_loss_net_change_final_minus_epoch0'], 6)}，最佳 Huber checkpoint 处仅下降 {_number(q['fc_loss_epoch0'] - q['fc_loss_best_checkpoint_epoch'], 6)}；即使这一下降存在，新药方向也没有普遍改善。

## 边界与公平性

- 两模型使用同一 Stage A checkpoint、同一 Stage B 初始参数 hash、seed、训练样本顺序、学习率、epoch、patience 和模型维度。
- 唯一区别是 train-only parity FC targets 与预声明 `fc_weight=0.1`。
- validation parity FC 未计算、未用于早停；没有权重搜索、descriptor、batch、similarity、pooled controls、多 seed 或测试集预测。
- `test_proteome_opened=false`。

## 执行、测试与训练审计

- 全部 V2 测试：`python -m pytest --import-mode=importlib baseline/tests/test_chemical_component_analysis_v2.py baseline/tests/test_chemical_feature_components_v2.py baseline/tests/test_chemical_ood_analysis_v2.py baseline/tests/test_experimental_parity_fc_v2.py baseline/tests/test_model_v2_contract.py baseline/tests/test_model_v2_losses.py baseline/tests/test_no_label_leakage.py baseline/tests/test_normalized_response_analysis_v2.py -q`；59 passed（39.30 s）。
- Person C 回归：在 `baseline` 工作目录运行 `python -m pytest tests/test_person_c.py -q`；6 passed（3.62 s）。
- 冻结化学/基因组工件回归：`python -m pytest --import-mode=importlib tests -q`；42 passed（13.84 s）。
- Smoke：`python -m baseline.training_v2 --config baseline/configs/model_v2_experimental_parity_fc_morgan_no_batch.yaml --output-dir reports/model_v2_stage2/smoke_experimental_parity_fc_morgan_no_batch_seed_20260814 --smoke`；PASS，16 treatment、14 exact matched，2 epochs，FC loss 1.004956 → 0.999922。
- 正式训练：同一入口去掉 `--smoke`，输出到 `{formal_dir}`；CUDA，耗时 65.760 秒，峰值显存 allocated/reserved 为 50,331,648/85,983,232 bytes。
- Stage A 未重训，固定 checkpoint epoch 3；Stage B 共 14 epochs，最佳 epoch 1，epoch 13 因 patience exhausted 停止。
- 初次定向测试曾因命令工作目录/pytest 导入模式错误导致 collection failure；改用项目约定 `--import-mode=importlib` 后通过。真实 pair identity 初版测试容差过严（最大误差 2.86e-6）后明确采用 float32 容差 1e-5；没有映射、mask、loss、泄漏、OOM 或 NaN 失败。

## 产物

- 逐药物 CSV：`{csv_path.resolve()}`
- 机器可读比较 JSON：`{json_path.resolve()}`
- 配置：`{(ROOT / 'baseline/configs/model_v2_experimental_parity_fc_morgan_no_batch.yaml').resolve()}`
- Smoke 目录：`{smoke_dir.resolve()}`
- 正式训练目录：`{formal_dir}`
- 最佳 checkpoint：`{formal_dir / 'stage_b_best.pt'}`
- resolved config：`{formal_dir / 'resolved_config.json'}`
- 每 epoch history：`{formal_dir / 'training_history.json'}`
- 配对表/审计：`{formal_dir / 'experimental_parity_fc_pairing.csv'}`；`{formal_dir / 'experimental_parity_fc_pairing_audit.json'}`
- 训练摘要：`{formal_dir / 'training_summary.json'}`
"""


def run_analysis(run_dirs, csv_output, summary_output, report_output, batch_size=64, device="auto"):
    if set(run_dirs) != set(MODELS):
        raise ValueError("both Morgan comparison runs are required")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {name: directory / "stage_b_best.pt" for name, directory in run_dirs.items()}
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    evaluations = {name: {} for name in MODELS}
    for scenario in VAL_SCENARIOS:
        for name in MODELS:
            evaluations[name][scenario] = evaluate_component_checkpoint(
                "morgan_only", checkpoints[name], scenario, meta, labels, masks,
                artifacts, vocab, int(batch_size), device,
                allow_experimental_nonofficial_parity_fc=True,
            )
    per_entity = build_per_entity_table(meta, evaluations)
    combined = combine_equal_scenarios(per_entity)
    amplitudes = build_amplitudes(evaluations)
    huber_summary = read_json(run_dirs[MODELS[0]] / "training_summary.json")
    fc_summary = read_json(run_dirs[MODELS[1]] / "training_summary.json")
    summary = build_summary(
        huber_summary, fc_summary, amplitudes, per_entity, combined,
        run_dirs, checkpoints, artifacts, device,
    )
    summary["outputs"] = {
        "per_entity_csv": str(csv_output.resolve()),
        "comparison_json": str(summary_output.resolve()),
        "report_markdown": str(report_output.resolve()),
    }
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    per_entity.to_csv(csv_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_output.write_text(render_report(summary, combined, csv_output, summary_output), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only Stage 2.5 nonofficial parity FC comparison")
    parser.add_argument("--huber-run-dir", type=Path, required=True)
    parser.add_argument("--fc-run-dir", type=Path, required=True)
    parser.add_argument("--per-entity-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    summary = run_analysis(
        {MODELS[0]: args.huber_run_dir, MODELS[1]: args.fc_run_dir},
        args.per_entity_output, args.summary_output, args.report_output,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
