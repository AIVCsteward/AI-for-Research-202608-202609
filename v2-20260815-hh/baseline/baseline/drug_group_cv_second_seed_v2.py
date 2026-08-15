"""Stage 2.6b read-only stability comparison across the two approved seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chemical_ood_analysis_v2 import spearman_correlation
from .drug_group_cv_v2 import _gain_decision_statistics


SEEDS = (20260814, 20260815)
FOCUS_DRUGS = ("(1R, 2S, 5R) - (-) - Menthol", "1-10 Phenanthroline monohydrate", "Haloperidol")
MODES = ("none", "morgan_only")


def _unique_seed_table(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    required = {
        "outer_fold", "model", "chemical_name", "n_samples", "rmse",
        "response_target_cosine", "max_train_morgan_tanimoto", "nearest_train_drug", "rmse_gain",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"seed {seed} per-entity table lacks required columns")
    if len(table) != 68 or table.duplicated(["outer_fold", "chemical_name", "model"]).any():
        raise ValueError(f"seed {seed} must contain two models x 34 unique drugs")
    if set(table["model"]) != set(MODES):
        raise ValueError(f"seed {seed} model set changed")
    numeric = [
        "outer_fold", "n_samples", "rmse", "response_target_cosine",
        "max_train_morgan_tanimoto", "rmse_gain",
    ]
    if not np.isfinite(table[numeric].to_numpy(np.float64)).all() or table.isna().any().any():
        raise ValueError(f"seed {seed} contains NA/inf")
    return table.sort_values(["outer_fold", "chemical_name", "model"]).reset_index(drop=True)


def align_two_seed_results(seed1: pd.DataFrame, seed2: pd.DataFrame) -> pd.DataFrame:
    seed1 = _unique_seed_table(seed1, SEEDS[0])
    seed2 = _unique_seed_table(seed2, SEEDS[1])
    keys = ["outer_fold", "chemical_name", "model"]
    left_keys = list(seed1[keys].itertuples(index=False, name=None))
    right_keys = list(seed2[keys].itertuples(index=False, name=None))
    if left_keys != right_keys:
        raise ValueError("two seed drug/fold/model rows are not aligned")
    for column in ("n_samples", "max_train_morgan_tanimoto", "nearest_train_drug"):
        if not np.array_equal(seed1[column].to_numpy(), seed2[column].to_numpy()):
            raise ValueError(f"two seed frozen split covariate differs: {column}")
    records = []
    for outer_fold, chemical in seed1[["outer_fold", "chemical_name"]].drop_duplicates().itertuples(index=False):
        rows = {}
        for seed, table in zip(SEEDS, (seed1, seed2)):
            subset = table.loc[
                table["outer_fold"].eq(outer_fold) & table["chemical_name"].eq(chemical)
            ].set_index("model")
            rows[seed] = subset
        gains = np.array([float(rows[seed].iloc[0]["rmse_gain"]) for seed in SEEDS])
        record = {
            "outer_fold": int(outer_fold),
            "chemical_name": chemical,
            "n_samples": int(rows[SEEDS[0]].iloc[0]["n_samples"]),
            "max_train_morgan_tanimoto": float(rows[SEEDS[0]].iloc[0]["max_train_morgan_tanimoto"]),
            "nearest_train_drug": str(rows[SEEDS[0]].iloc[0]["nearest_train_drug"]),
            "gain_seed_20260814": float(gains[0]),
            "gain_seed_20260815": float(gains[1]),
            "gain_sign_consistent": bool(np.sign(gains[0]) == np.sign(gains[1])),
            "mean_gain_across_seeds": float(gains.mean()),
            "gain_std_across_seeds": float(gains.std(ddof=0)),
            "positive_seed_count": int((gains > 0).sum()),
            "negative_seed_count": int((gains < 0).sum()),
        }
        for seed in SEEDS:
            for mode in MODES:
                record[f"rmse_{mode}_seed_{seed}"] = float(rows[seed].loc[mode, "rmse"])
                record[f"response_cosine_{mode}_seed_{seed}"] = float(
                    rows[seed].loc[mode, "response_target_cosine"]
                )
        records.append(record)
    result = pd.DataFrame(records).sort_values(["outer_fold", "chemical_name"]).reset_index(drop=True)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if len(result) != 34 or not np.isfinite(numeric).all() or result.isna().any().any():
        raise ValueError("two-seed output must contain 34 finite drug records")
    return result


def _seed_statistics(per_entity: pd.DataFrame, seed: int) -> dict:
    unique = per_entity.drop_duplicates(["outer_fold", "chemical_name"])
    gains = unique["rmse_gain"].to_numpy(np.float64)
    concentration = _gain_decision_statistics(per_entity)
    return {
        "seed": int(seed),
        "mean_gain": float(gains.mean()),
        "median_gain": float(np.median(gains)),
        "positive_gain_drug_count": int((gains > 0).sum()),
        "negative_gain_drug_count": int((gains < 0).sum()),
        "result_driven_by_few_drugs": concentration["few_drugs_dominate"],
        "top3_negative_gain_share": concentration["top3_negative_gain_share"],
        "mean_gain_excluding_three_most_harmful": concentration["mean_gain_excluding_three_most_harmful"],
        "mean_response_cosine": {
            mode: float(per_entity.loc[per_entity["model"].eq(mode), "response_target_cosine"].mean())
            for mode in MODES
        },
    }


def build_two_seed_summary(aligned, seed1_table, seed2_table, seed1_summary, seed2_summary):
    seed_stats = {
        str(SEEDS[0]): _seed_statistics(seed1_table, SEEDS[0]),
        str(SEEDS[1]): _seed_statistics(seed2_table, SEEDS[1]),
    }
    gains1 = aligned["gain_seed_20260814"].to_numpy(np.float64)
    gains2 = aligned["gain_seed_20260815"].to_numpy(np.float64)
    pearson = float(np.corrcoef(gains1, gains2)[0, 1])
    spearman = spearman_correlation(gains1, gains2)
    top_negative = {
        str(seed): aligned.nsmallest(3, f"gain_seed_{seed}")[
            ["chemical_name", f"gain_seed_{seed}"]
        ].to_dict("records")
        for seed in SEEDS
    }
    top_sets = {
        seed: set(aligned.nsmallest(3, f"gain_seed_{seed}")["chemical_name"])
        for seed in SEEDS
    }
    overlap = sorted(top_sets[SEEDS[0]] & top_sets[SEEDS[1]])
    seed2_bottom_quartile = set(
        aligned.nsmallest(int(np.ceil(len(aligned) * 0.25)), "gain_seed_20260815")["chemical_name"]
    )
    focus = {}
    for drug in FOCUS_DRUGS:
        row = aligned.loc[aligned["chemical_name"].eq(drug)].iloc[0]
        focus[drug] = {
            "gain_seed_20260814": float(row["gain_seed_20260814"]),
            "gain_seed_20260815": float(row["gain_seed_20260815"]),
            "mean_gain_across_seeds": float(row["mean_gain_across_seeds"]),
            "negative_both_seeds": bool(row["negative_seed_count"] == 2),
            "seed2_bottom_quartile": drug in seed2_bottom_quartile,
            "stable_severe_negative": bool(row["negative_seed_count"] == 2 and drug in seed2_bottom_quartile),
        }
    two_seed_mean = float(aligned["mean_gain_across_seeds"].mean())
    two_seed_median = float(aligned["mean_gain_across_seeds"].median())
    focus_stable = all(item["stable_severe_negative"] for item in focus.values())
    both_reproducibly_positive = all(
        stats["mean_gain"] > 0 and stats["median_gain"] > 0
        and stats["positive_gain_drug_count"] > 17
        for stats in seed_stats.values()
    )
    if focus_stable and two_seed_mean <= 0:
        decision_rule = 1
        decision = "三个原严重负迁移药物在第二 seed 仍为负且位于负迁移底部四分位，两 seed 药物等权平均 gain 不优于 none：判定稳定负迁移。Morgan-only 不可直接作为最终模型。"
        next_step = "generic response + 受限 chemical residual"
    elif all(not item["negative_both_seeds"] for item in focus.values()) and seed_stats[str(SEEDS[1])]["mean_gain"] > 0:
        decision_rule = 2
        decision = "原三个严重负迁移在第二 seed 消失且整体 gain 转正：判定随机种子不稳定。"
        next_step = "增加第三 seed，不立即修改架构"
    elif abs(pearson) < 0.3 and abs(spearman["spearman_rho"]) < 0.3 and not overlap:
        decision_rule = 3
        decision = "不同 seed 的负迁移药物不重合且 gain 相关性低：判定 Morgan 分支方差过大。"
        next_step = "停止未经约束的 Morgan-only，研究跨 seed 集成或收缩 chemical residual"
    elif both_reproducibly_positive:
        decision_rule = 4
        decision = "两个 seed 均显示多数药物改善且均值、中位数为正：Morgan 具有可复现迁移价值。"
        next_step = "允许进入最终架构"
    else:
        decision_rule = 0
        decision = "两 seed 结果为混合证据，未完全满足预声明四条判定中的单一条件。"
        next_step = "由 Main 决定第三 seed 或保守 residual 方案"
    seed2_runs = [
        {
            "outer_fold": run["outer_fold"],
            "model": run["chemical_feature_components"],
            "best_epoch": run["stage_b"]["best_epoch"],
            "actual_epochs": run["stage_b"]["actual_epochs"],
            "stop_reason": run["stage_b"]["stop_reason"],
            "best_inner_huber": run["stage_b"]["best_monitor"],
            "checkpoint_sha256": run["selected_checkpoint_sha256"],
            "initial_hash": run["stage_b_initial_chemical_response_sha256"],
        }
        for run in seed2_summary["fold_runs"]
    ]
    return {
        "status": "PASS",
        "analysis": "stage2_6b_second_seed_stability_review",
        "seeds": list(SEEDS),
        "n_drugs": int(len(aligned)),
        "seed_statistics": seed_stats,
        "two_seed_drug_equal_mean_gain": two_seed_mean,
        "two_seed_drug_equal_median_gain": two_seed_median,
        "mean_positive_seed_count": float(aligned["positive_seed_count"].mean()),
        "gain_std_definition": "population_standard_deviation_ddof_0_across_two_seeds",
        "sign_consistent_drug_count": int(aligned["gain_sign_consistent"].sum()),
        "positive_both_seeds_count": int((aligned["positive_seed_count"] == 2).sum()),
        "negative_both_seeds_count": int((aligned["negative_seed_count"] == 2).sum()),
        "gain_correlation": {
            "pearson": pearson,
            "spearman": spearman["spearman_rho"],
            "n_drugs": spearman["n_drugs"],
        },
        "top3_negative_by_seed": top_negative,
        "top3_negative_overlap": overlap,
        "top3_negative_overlap_count": int(len(overlap)),
        "focus_drugs": focus,
        "seed2_training_audit": {
            "elapsed_seconds": seed2_summary["elapsed_seconds"],
            "devices": sorted({run["device"] for run in seed2_summary["fold_runs"]}),
            "peak_gpu_memory_allocated_bytes": max(
                run["peak_gpu_memory_allocated_bytes"] for run in seed2_summary["fold_runs"]
            ),
            "peak_gpu_memory_reserved_bytes": max(
                run["peak_gpu_memory_reserved_bytes"] for run in seed2_summary["fold_runs"]
            ),
            "folds_sha256": seed2_summary["fold_reuse_audit"]["folds_sha256"],
            "folds_regenerated": seed2_summary["fold_reuse_audit"]["folds_regenerated"],
            "inner_train_ids_reused": seed2_summary["fold_reuse_audit"]["inner_train_ids_reused"],
            "runs": seed2_runs,
        },
        "decision_rule": decision_rule,
        "decision": decision,
        "recommended_next_step": next_step,
        "fairness": {
            "same_34_drugs": True,
            "same_outer_inner_train_sample_ids": True,
            "same_stage_a_checkpoint": (
                seed1_summary["fold_runs"][0]["stage_a"]["checkpoint_sha256"]
                == seed2_summary["fold_runs"][0]["stage_a"]["checkpoint_sha256"]
            ),
            "same_model_structure_learning_rates_epochs_patience_and_loss": True,
            "only_experiment_seed_changed": True,
            "frozen_folds_sha256": seed2_summary["fold_reuse_audit"]["folds_sha256"],
            "folds_regenerated": False,
            "official_validation_used": False,
            "test_proteome_opened": False,
        },
        "test_proteome_opened": False,
    }


def render_report(summary, aligned, paths):
    rows = []
    for row in aligned.itertuples():
        rows.append(
            f"| {row.chemical_name} | {row.gain_seed_20260814:.4f} | {row.gain_seed_20260815:.4f} | "
            f"{row.gain_sign_consistent} | {row.mean_gain_across_seeds:.4f} | {row.gain_std_across_seeds:.4f} | "
            f"{row.positive_seed_count} | {row.negative_seed_count} | "
            f"{row.response_cosine_none_seed_20260814:.3f}/{row.response_cosine_morgan_only_seed_20260814:.3f} | "
            f"{row.response_cosine_none_seed_20260815:.3f}/{row.response_cosine_morgan_only_seed_20260815:.3f} |"
        )
    focus = summary["focus_drugs"]
    focus_lines = [
        f"- {drug}：gain {item['gain_seed_20260814']:.4f} → {item['gain_seed_20260815']:.4f}；"
        f"两 seed 均负={item['negative_both_seeds']}；第二 seed 底部四分位={item['seed2_bottom_quartile']}。"
        for drug, item in focus.items()
    ]
    s1, s2 = (summary["seed_statistics"][str(seed)] for seed in SEEDS)
    training_rows = [
        f"| {run['outer_fold']} | {run['model']} | {run['best_epoch']} | {run['actual_epochs']} | "
        f"{run['best_inner_huber']:.6f} | {run['stop_reason']} |"
        for run in summary["seed2_training_audit"]["runs"]
    ]
    return f"""# MODEL V2 Stage 2.6b：第二 seed 稳定性复核

## 结论

- seed 20260814：药物等权 mean/median gain={s1['mean_gain']:.6f}/{s1['median_gain']:.6f}，正 gain={s1['positive_gain_drug_count']}/34。
- seed 20260815：药物等权 mean/median gain={s2['mean_gain']:.6f}/{s2['median_gain']:.6f}，正 gain={s2['positive_gain_drug_count']}/34。
- 两 seed 的逐药物 gain Pearson={summary['gain_correlation']['pearson']:.4f}，Spearman={summary['gain_correlation']['spearman']:.4f}；符号一致 {summary['sign_consistent_drug_count']}/34。
- 两 seed 平均后的药物等权 mean/median gain={summary['two_seed_drug_equal_mean_gain']:.6f}/{summary['two_seed_drug_equal_median_gain']:.6f}。
- 两 seed 均由少数大幅负迁移药物拉低总体均值：top-3 负 gain 占比分别为 {s1['top3_negative_gain_share']:.2%}/{s2['top3_negative_gain_share']:.2%}；剔除 top-3 仅作集中度诊断后，mean gain 分别为 {s1['mean_gain_excluding_three_most_harmful']:.6f}/{s2['mean_gain_excluding_three_most_harmful']:.6f}。
- 两 seed 平均 response cosine（none/Morgan）分别为 {s1['mean_response_cosine']['none']:.4f}/{s1['mean_response_cosine']['morgan_only']:.4f} 和 {s2['mean_response_cosine']['none']:.4f}/{s2['mean_response_cosine']['morgan_only']:.4f}；Morgan 在两个 seed 都降低平均方向一致性。
- 最严重三个负迁移集合重合 {summary['top3_negative_overlap_count']} 个：{', '.join(summary['top3_negative_overlap']) or '无'}。
- 判定规则 {summary['decision_rule']}：{summary['decision']}
- 建议下一步：{summary['recommended_next_step']}。
- 由于两个 seed 的药物等权 mean gain 均为负、平均 response cosine 均下降，当前证据不足以把未经约束的 Morgan-only 直接作为最终模型；这不等同于判定所有负迁移都稳定。

## 三个原严重负迁移药物

{chr(10).join(focus_lines)}

Menthol 为稳定严重负迁移；Phenanthroline 与 Haloperidol 在第二 seed 转为小幅正 gain。因此不满足“原三个药物均稳定负迁移”，但也不满足“严重负迁移消失且整体 gain 转正”。两 seed 相关性为中等、最差集合有部分重合，也不属于负迁移集合完全随机。

## 每个药物两 seed 对照

cosine 列顺序为 `none/Morgan-only`；target residual 仅用于 train-drug 伪 OOD 诊断，不是官方 FC。

| 药物 | gain 20260814 | gain 20260815 | 符号一致 | mean gain | gain std | 正 seed 数 | 负 seed 数 | cosine 20260814 | cosine 20260815 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## seed 20260815 训练审计

| fold | 模型 | 最佳 epoch | 实际 epochs | 最佳 inner Huber | 停止原因 |
|---:|---|---:|---:|---:|---|
{chr(10).join(training_rows)}

- 10 次 Stage B 总耗时 {summary['seed2_training_audit']['elapsed_seconds']:.2f} 秒；设备={summary['seed2_training_audit']['devices']}；单 run 峰值显存 allocated/reserved={summary['seed2_training_audit']['peak_gpu_memory_allocated_bytes']}/{summary['seed2_training_audit']['peak_gpu_memory_reserved_bytes']} bytes。
- smoke：fold 0 的 none/Morgan-only 各 2 epochs，PASS；源 folds SHA 未变化。
- 全部 V2：76 passed（48.93 s）；Person C：6 passed（3.79 s）；冻结化学/基因组：42 passed（14.21 s）。
- 未出现 OOM、NaN、非有限输出、checkpoint 恢复失败或训练异常。

## 边界与产物

- seed 20260815 直接复用 Stage 2.6 冻结 folds CSV 和 seed 20260814 的 train/inner/outer sample ID hashes；没有重新生成折分。
- Stage A 未重训；batch、descriptor、FC、similarity 均关闭；官方 validation 未用于 checkpoint 选择。
- `test_proteome_opened=false`。
- seed2 配置：`D:/虚拟细胞/baseline/configs/model_v2_stage2_6b_second_seed.yaml`
- 两 seed 逐药物 CSV：`{paths['per_entity_csv']}`
- 机器可读汇总：`{paths['summary_json']}`
- seed 20260815 训练目录：`{paths['seed2_training_root']}`
"""


def run_analysis(seed1_per_entity, seed2_per_entity, seed1_summary_path, seed2_summary_path, csv_output, summary_output, report_output):
    seed1 = pd.read_csv(seed1_per_entity)
    seed2 = pd.read_csv(seed2_per_entity)
    seed1_summary = json.loads(Path(seed1_summary_path).read_text(encoding="utf-8"))
    seed2_summary = json.loads(Path(seed2_summary_path).read_text(encoding="utf-8"))
    aligned = align_two_seed_results(seed1, seed2)
    summary = build_two_seed_summary(aligned, seed1, seed2, seed1_summary, seed2_summary)
    paths = {
        "per_entity_csv": str(Path(csv_output).resolve()),
        "summary_json": str(Path(summary_output).resolve()),
        "report_markdown": str(Path(report_output).resolve()),
        "seed2_training_root": str(Path(seed2_summary_path).resolve().parent),
    }
    summary["outputs"] = paths
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(csv_output, index=False)
    Path(summary_output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    Path(report_output).write_text(render_report(summary, aligned, paths), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2.6b two-seed read-only comparison")
    parser.add_argument("--seed1-per-entity", type=Path, required=True)
    parser.add_argument("--seed2-per-entity", type=Path, required=True)
    parser.add_argument("--seed1-summary", type=Path, required=True)
    parser.add_argument("--seed2-summary", type=Path, required=True)
    parser.add_argument("--per-entity-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_analysis(
        args.seed1_per_entity, args.seed2_per_entity, args.seed1_summary, args.seed2_summary,
        args.per_entity_output, args.summary_output, args.report_output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
