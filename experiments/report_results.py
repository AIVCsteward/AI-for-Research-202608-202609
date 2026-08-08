"""Render structured C5 results into the Person C6 experiment log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.evaluation import TEST_SPLITS, VAL_SPLITS


def _display(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_experiment_log(payload):
    """Render one reproducible Markdown log from the C5 result schema."""
    records = payload.get("records", [])
    discipline = payload.get("data_discipline", {})
    lines = [
        "# EXPERIMENT LOG",
        "",
        "## 1. 数据与复现纪律",
        "",
        f"- 统计量拟合范围：{discipline.get('fit_statistics', '未声明')}",
        f"- 早停场景：{discipline.get('early_stopping_split', '未声明')}",
        f"- 预测尺度：{discipline.get('prediction_scale', '未声明')}",
        f"- 验证集参与训练：{discipline.get('validation_used_for_training', '未声明')}",
        f"- 测试真值参与训练：{discipline.get('test_truth_used_for_training', '未声明')}",
        "",
        "## 2. Loss 消融执行状态",
        "",
        "| 实验 | 组合 | 状态 | B4/C3 依赖 |",
        "|---|---|---|---|",
    ]
    for record in records:
        weights = record.get("loss_weights", {})
        combination = ", ".join(f"{key}={value:g}" for key, value in weights.items())
        dependency = "需要 edge_index + target_edge_corr" if record.get(
            "requires_correlation_bundle"
        ) else "无"
        lines.append(
            f"| {record.get('name')} | {combination} | {record.get('status')} | {dependency} |"
        )

    lines.extend([
        "",
        "## 3. 训练与早停记录",
        "",
        "| 实验 | 实际 epoch | 最佳 epoch | 最佳 val_both monitor | 最终 total | MSE | FC | L2 | Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        result = record.get("result") or {}
        history = result.get("history") or {}
        monitor = history.get("monitor") or []
        best_position = max(range(len(monitor)), key=monitor.__getitem__) if monitor else None
        lines.append(
            "| {name} | {epochs} | {best_epoch} | {best_monitor} | {total} | {mse} | {fc} | {l2} | {corr} |".format(
                name=record.get("name"),
                epochs=len(history.get("loss_total") or []),
                best_epoch="—" if best_position is None else best_position + 1,
                best_monitor="—" if best_position is None else _display(monitor[best_position]),
                total=_display((history.get("loss_total") or [None])[-1]),
                mse=_display((history.get("loss_mse") or [None])[-1]),
                fc=_display((history.get("loss_fc") or [None])[-1]),
                l2=_display((history.get("loss_l2") or [None])[-1]),
                corr=_display((history.get("loss_corr") or [None])[-1]),
            )
        )

    lines.extend([
        "",
        "## 4. 分场景指标",
        "",
        "| 实验 | Split | n | log2 RMSE | Global R² | Per-Protein R² 中位数 | FC Pearson |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        result = record.get("result") or {}
        metrics = result.get("metrics", {})
        for split_name in VAL_SPLITS + TEST_SPLITS:
            split_metrics = metrics.get(split_name)
            if split_metrics is None:
                continue
            lines.append(
                "| {name} | {split} | {n} | {rmse} | {global_r2} | {protein_r2} | {fc} |".format(
                    name=record.get("name"),
                    split=split_name,
                    n=_display(split_metrics.get("n_samples"), 0),
                    rmse=_display(split_metrics.get("log2_rmse")),
                    global_r2=_display(split_metrics.get("global_r2")),
                    protein_r2=_display(split_metrics.get("per_protein_r2_median")),
                    fc=_display(split_metrics.get("fc_pearson")),
                )
            )

    lines.extend([
        "",
        "## 5. 基线对标与归因",
        "",
        "| 实验 | 对标结果 | 归因记录 |",
        "|---|---|---|",
    ])
    for record in records:
        result = record.get("result") or {}
        comparison = result.get("baseline_comparison") or {}
        lines.append(
            f"| {record.get('name')} | {_display(comparison.get('summary'))} | "
            f"{_display(comparison.get('attribution'))} |"
        )

    lines.extend([
        "",
        "## 6. 提交检查",
        "",
        "| 实验 | 样本数 | 蛋白列数 | 无 NA | 无 Inf | prediction_scale |",
        "|---|---:|---:|---|---|---|",
    ])
    for record in records:
        result = record.get("result") or {}
        checks = result.get("submission_checks") or {}
        lines.append(
            "| {name} | {rows} | {cols} | {no_na} | {no_inf} | {scale} |".format(
                name=record.get("name"),
                rows=_display(checks.get("n_samples"), 0),
                cols=_display(checks.get("n_proteins"), 0),
                no_na=_display(checks.get("no_na")),
                no_inf=_display(checks.get("no_inf")),
                scale=_display(checks.get("prediction_scale")),
            )
        )

    lines.extend([
        "",
        "## 7. 产物索引",
        "",
        "| 实验 | checkpoint | config | prediction |",
        "|---|---|---|---|",
    ])
    for record in records:
        result = record.get("result") or {}
        artifacts = result.get("artifacts") or {}
        lines.append(
            f"| {record.get('name')} | {_display(artifacts.get('checkpoint'))} | "
            f"{_display(artifacts.get('config'))} | {_display(artifacts.get('prediction'))} |"
        )

    lines.extend([
        "",
        "## 8. 科学解释与失败回退",
        "",
        "每次正式实验完成后，从结构化结果中的归因记录提取：哪些蛋白或响应模式改善、",
        "改善集中在哪个 OOD 场景、是否超过 Matched Control，以及失败时回退到哪个 loss 组合。",
        "",
    ])
    return "\n".join(lines)


def write_experiment_log(results_path, output_path):
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    markdown = render_experiment_log(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render C5 results into EXPERIMENT_LOG.md")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("EXPERIMENT_LOG.md"))
    args = parser.parse_args(argv)
    path = write_experiment_log(args.results, args.output)
    print(f"C6 experiment log written to: {path}")


if __name__ == "__main__":
    main()
