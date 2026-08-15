"""Train-only Water/DMSO control-mapping sensitivity audit.

This module is deliberately separate from the official data-contract audit.
It compares an inferred odd/even pert_id mapping with a pooled Water+DMSO
exact-context diagnostic.  Neither rule is promoted to the official mapping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import audit_data_contract as contract


SCHEMA_VERSION = "1.0.0"
INFERRED_MAPPING_NAME = "pert_id_parity_v1"
POOLED_RULE_NAME = "pooled_water_dmso_exact_context_v1"
HIGH_EFFECT_ABS_FC_THRESHOLD = 1.0


def inferred_parity_mapping() -> dict[str, str]:
    """Frozen inferred mapping for #1..#47; #48 is QC, not a solvent."""
    return {
        f"#{number}": "Water" if number % 2 else "DMSO"
        for number in range(1, 48)
    }


def classify_pert_id(pert_id: object) -> str:
    value = str(pert_id)
    if value == "#48":
        return "Quality Control"
    mapping = inferred_parity_mapping()
    if value not in mapping:
        raise contract.ContractError(f"pert_id outside frozen #1..#48 rule: {value}")
    return mapping[value]


def _train_treatment_ids(metadata_train: pd.DataFrame) -> pd.Index:
    names = metadata_train["perturbation_no_concentration"]
    eligible = ~names.isin((*contract.CONTROL_NAMES, contract.QUALITY_CONTROL_NAME))
    return pd.Index(metadata_train.loc[eligible, "sample_ID"].astype(str))


def read_train_proteome_without_parsing_validation_labels(
    proteome_path: Path,
    metadata_train_val: pd.DataFrame,
    proteins: Sequence[str],
) -> pd.DataFrame:
    """Load only train protein cells from a mixed train/validation CSV.

    The first pass reads only sample_ID.  The second pass uses ``skiprows`` to
    reject every non-train data line before pandas tokenizes its protein cells.
    No validation protein column is selected into a dataframe or statistic.
    """
    required = {"sample_ID", "split_final"}
    missing = required.difference(metadata_train_val.columns)
    if missing:
        raise contract.ContractError(f"metadata missing columns: {sorted(missing)}")
    metadata = metadata_train_val.copy()
    metadata["sample_ID"] = metadata["sample_ID"].astype(str)
    if metadata["sample_ID"].duplicated().any():
        raise contract.ContractError("metadata sample_ID must be unique")

    train_ids = pd.Index(
        metadata.loc[metadata["split_final"].eq(contract.TRAIN_SPLIT), "sample_ID"]
    )
    contract.require_train_fit_ids(metadata, train_ids)
    source_ids = pd.read_csv(proteome_path, usecols=["sample_ID"])["sample_ID"].astype(str)
    if source_ids.duplicated().any():
        raise contract.ContractError("proteome sample_ID must be unique")
    missing_train = train_ids.difference(source_ids)
    if len(missing_train):
        raise contract.ContractError(
            f"proteome missing train sample_ID values: {missing_train[:10].tolist()}"
        )

    header = pd.read_csv(proteome_path, nrows=0)
    missing_proteins = sorted(set(proteins).difference(header.columns))
    if missing_proteins:
        raise contract.ContractError(
            f"feature contract proteins absent from source: {missing_proteins[:10]}"
        )
    train_set = set(train_ids)
    # CSV line 0 is the header; data line i is skiprows line i + 1.
    skipped_non_train_lines = frozenset(
        index + 1 for index, sample_id in enumerate(source_ids) if sample_id not in train_set
    )
    usecols = ["sample_ID", *proteins]
    dtype = {protein: "float32" for protein in proteins}
    train_proteome = pd.read_csv(
        proteome_path,
        usecols=usecols,
        dtype=dtype,
        skiprows=lambda line_number: line_number in skipped_non_train_lines,
    )
    train_proteome["sample_ID"] = train_proteome["sample_ID"].astype(str)
    loaded_ids = pd.Index(train_proteome["sample_ID"])
    if set(loaded_ids) != train_set or len(loaded_ids) != len(train_ids):
        raise contract.ContractError("train-only proteome load returned an unexpected ID set")
    return train_proteome.set_index("sample_ID", drop=False).loc[train_ids]


def build_pooled_exact_context_pairs(
    metadata_train: pd.DataFrame,
    treatment_sample_ids: Iterable[str],
    match_keys: Sequence[str] = contract.MATCH_KEYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match all Water and DMSO controls in the same exact context."""
    required = {"sample_ID", "perturbation_no_concentration", *match_keys}
    missing = sorted(required.difference(metadata_train.columns))
    if missing:
        raise contract.ContractError(f"metadata missing control columns: {missing}")
    indexed = metadata_train.assign(
        sample_ID=metadata_train["sample_ID"].astype(str)
    ).set_index("sample_ID", drop=False)
    if not indexed.index.is_unique:
        raise contract.ContractError("metadata sample_ID must be unique")
    treatment_ids = pd.Index([str(value) for value in treatment_sample_ids])
    unknown = treatment_ids.difference(indexed.index)
    if len(unknown):
        raise contract.ContractError(f"unknown treatment IDs: {unknown[:10].tolist()}")

    controls = indexed[
        indexed["perturbation_no_concentration"].isin(contract.CONTROL_NAMES)
    ]
    lookup: dict[tuple[object, ...], list[str]] = {}
    solvent_counts: dict[tuple[object, ...], Counter[str]] = {}
    for sample_id, row in controls.iterrows():
        key = tuple(row[column] for column in match_keys)
        lookup.setdefault(key, []).append(sample_id)
        solvent_counts.setdefault(key, Counter())[row["perturbation_no_concentration"]] += 1

    pairs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for sample_id, row in indexed.loc[treatment_ids].iterrows():
        key = tuple(row[column] for column in match_keys)
        matched = tuple(lookup.get(key, ()))
        if not matched:
            failures.append(
                {"treatment_sample_ID": sample_id, "reason": "no_exact_water_or_dmso_control"}
            )
            continue
        counts = solvent_counts[key]
        pairs.append(
            {
                "treatment_sample_ID": sample_id,
                "control_sample_ids": matched,
                "expected_control": "Water+DMSO pooled",
                "n_controls": len(matched),
                "n_water": counts.get("Water", 0),
                "n_dmso": counts.get("DMSO", 0),
            }
        )
    return pd.DataFrame(pairs), pd.DataFrame(failures)


def _log2_train_values(
    train_proteome: pd.DataFrame, proteins: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    raw = train_proteome.loc[:, proteins].to_numpy(dtype=np.float32)
    observed = np.isfinite(raw) & (raw > 0)
    transformed = np.full(raw.shape, np.nan, dtype=np.float32)
    np.log2(raw, out=transformed, where=observed)
    return transformed, observed


def compute_fc_for_pairs(
    train_proteome: pd.DataFrame,
    proteins: Sequence[str],
    pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute observed treatment-control FC with a joint valid mask."""
    log_values, observed = _log2_train_values(train_proteome, proteins)
    row_number = {sample_id: i for i, sample_id in enumerate(train_proteome.index)}
    treatment_ids = pairs["treatment_sample_ID"].astype(str).tolist()
    fc = np.full((len(pairs), len(proteins)), np.nan, dtype=np.float32)
    fc_mask = np.zeros(fc.shape, dtype=bool)
    control_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    for output_row, pair in enumerate(pairs.itertuples(index=False)):
        control_ids = tuple(str(value) for value in pair.control_sample_ids)
        if control_ids not in control_cache:
            control_rows = [row_number[value] for value in control_ids]
            control_cache[control_ids] = contract.aggregate_observed_controls(
                log_values[control_rows], observed[control_rows]
            )
        control_mean, control_mask = control_cache[control_ids]
        treatment_row = row_number[str(pair.treatment_sample_ID)]
        valid = observed[treatment_row] & control_mask & np.isfinite(control_mean)
        np.subtract(log_values[treatment_row], control_mean, out=fc[output_row], where=valid)
        fc_mask[output_row] = valid
    return fc, fc_mask, treatment_ids


def high_effect_summary(
    fc: np.ndarray,
    mask: np.ndarray,
    threshold: float = HIGH_EFFECT_ABS_FC_THRESHOLD,
) -> dict[str, object]:
    valid_high = mask & np.isfinite(fc) & (np.abs(fc) > threshold)
    per_treatment = valid_high.sum(axis=1)
    return {
        "threshold": f"abs(FC) > {threshold:g}",
        "sample_protein_entries": int(valid_high.sum()),
        "unique_proteins": int(valid_high.any(axis=0).sum()),
        "per_treatment_median": float(np.median(per_treatment)) if len(per_treatment) else None,
        "per_treatment_mean": float(per_treatment.mean()) if len(per_treatment) else None,
    }


def compare_fc_definitions(
    parity_fc: np.ndarray,
    parity_mask: np.ndarray,
    parity_ids: Sequence[str],
    pooled_fc: np.ndarray,
    pooled_mask: np.ndarray,
    pooled_ids: Sequence[str],
) -> dict[str, object]:
    parity_pos = {sample_id: i for i, sample_id in enumerate(parity_ids)}
    pooled_pos = {sample_id: i for i, sample_id in enumerate(pooled_ids)}
    common_ids = [sample_id for sample_id in parity_ids if sample_id in pooled_pos]
    if not common_ids:
        raise contract.ContractError("control rules have no common matched treatments")
    p_rows = [parity_pos[value] for value in common_ids]
    q_rows = [pooled_pos[value] for value in common_ids]
    p_fc = parity_fc[p_rows]
    q_fc = pooled_fc[q_rows]
    common_mask = (
        parity_mask[p_rows]
        & pooled_mask[q_rows]
        & np.isfinite(p_fc)
        & np.isfinite(q_fc)
    )
    if not common_mask.any():
        raise contract.ContractError("control rules have no common valid FC values")
    p_values = p_fc[common_mask]
    q_values = q_fc[common_mask]
    direction_equal = np.sign(p_values) == np.sign(q_values)
    return {
        "matched_treatments": len(common_ids),
        "joint_valid_sample_protein_entries": int(common_mask.sum()),
        "fc_pearson_global_flattened": contract.masked_pearson(
            p_fc, q_fc, common_mask
        ),
        "direction_consistency_rate": float(direction_equal.mean()),
        "direction_definition": "sign(FC_parity) == sign(FC_pooled), including exact zero",
        "both_exact_zero_entries": int(((p_values == 0) & (q_values == 0)).sum()),
        "mean_absolute_fc_difference": float(np.abs(p_values - q_values).mean()),
        "high_effect_on_common_subset": {
            "inferred_parity": high_effect_summary(p_fc, common_mask),
            "pooled_context": high_effect_summary(q_fc, common_mask),
        },
    }


def _reason_counts(failures: pd.DataFrame) -> dict[str, int]:
    if failures.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in failures["reason"].value_counts().sort_index().items()
    }


def _pair_composition(pairs: pd.DataFrame) -> dict[str, int]:
    if pairs.empty or "n_water" not in pairs or "n_dmso" not in pairs:
        return {}
    water = pairs["n_water"].to_numpy(dtype=int)
    dmso = pairs["n_dmso"].to_numpy(dtype=int)
    return {
        "both_water_and_dmso": int(((water > 0) & (dmso > 0)).sum()),
        "water_only": int(((water > 0) & (dmso == 0)).sum()),
        "dmso_only": int(((water == 0) & (dmso > 0)).sum()),
    }


def _coverage(total: int, pairs: pd.DataFrame, failures: pd.DataFrame) -> dict[str, object]:
    matched = len(pairs)
    return {
        "eligible_train_treatments": total,
        "matched": matched,
        "unmatched": total - matched,
        "coverage_rate": matched / total if total else None,
        "failure_reasons": _reason_counts(failures),
        "matched_control_composition": _pair_composition(pairs),
    }


def run_sensitivity(
    metadata_train_val: pd.DataFrame,
    train_proteome: pd.DataFrame,
    feature_contract: Mapping[str, object],
) -> dict[str, object]:
    """Run the sensitivity audit from already train-restricted protein data."""
    metadata = metadata_train_val.copy()
    metadata["sample_ID"] = metadata["sample_ID"].astype(str)
    train_meta = metadata.loc[metadata["split_final"].eq(contract.TRAIN_SPLIT)].copy()
    train_ids = pd.Index(train_meta["sample_ID"])
    contract.require_train_fit_ids(metadata, train_ids)
    if set(train_proteome.index.astype(str)) != set(train_ids):
        raise contract.ContractError(
            "sensitivity proteome must contain exactly split_final=train IDs"
        )
    train_proteome = train_proteome.copy()
    train_proteome.index = train_proteome.index.astype(str)
    train_proteome = train_proteome.loc[train_ids]
    train_meta = train_meta.set_index("sample_ID", drop=False).loc[train_ids]

    proteins = [str(value) for value in feature_contract["proteins"]]
    treatment_ids = _train_treatment_ids(train_meta)
    control_ids = pd.Index(
        train_meta.loc[
            train_meta["perturbation_no_concentration"].isin(contract.CONTROL_NAMES),
            "sample_ID",
        ]
    )
    parity_pairs, parity_failures = contract.build_matched_control_pairs(
        train_meta,
        treatment_ids,
        control_ids,
        inferred_parity_mapping(),
    )
    pooled_pairs, pooled_failures = build_pooled_exact_context_pairs(
        train_meta, treatment_ids
    )
    parity_fc, parity_mask, parity_ids = compute_fc_for_pairs(
        train_proteome, proteins, parity_pairs
    )
    pooled_fc, pooled_mask, pooled_ids = compute_fc_for_pairs(
        train_proteome, proteins, pooled_pairs
    )

    pert_names = train_meta.groupby("pert_id")["perturbation_no_concentration"].agg(
        lambda values: sorted(set(map(str, values)))
    )
    qc_names = pert_names.get("#48", [])
    qc_rows = train_meta["pert_id"].astype(str).eq("#48")
    qc_consistent = bool(
        qc_rows.any()
        and train_meta.loc[qc_rows, "perturbation_no_concentration"]
        .eq(contract.QUALITY_CONTROL_NAME)
        .all()
    )
    total = len(treatment_ids)
    comparison = compare_fc_definitions(
        parity_fc,
        parity_mask,
        parity_ids,
        pooled_fc,
        pooled_mask,
        pooled_ids,
    )
    parity_failure_by_solvent = {}
    if not parity_failures.empty and "expected_control" in parity_failures:
        parity_failure_by_solvent = {
            str(key): int(value)
            for key, value in parity_failures["expected_control"]
            .dropna()
            .value_counts()
            .sort_index()
            .items()
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_train_only_sensitivity",
        "generated_on": date.today().isoformat(),
        "official_mapping_status": "blocked_pending_organizer_confirmation",
        "decision_policy": {
            "validation_or_test_score_used_to_select_rule": False,
            "rule_selection_permitted_from_this_result": False,
            "official_mapping_changed": False,
        },
        "data_scope": {
            "metadata_scope": "metadata_train_val read for split membership; calculations subset to split_final=train",
            "protein_label_scope": "split_final=train rows only",
            "mixed_train_validation_csv_physical_scan_required": True,
            "non_train_protein_cell_handling": "non-train lines skipped before protein-cell tokenization; never materialized or used",
            "fit_split": "train",
            "fit_sample_count": len(train_ids),
            "validation_protein_labels_read_or_used": False,
            "test_proteome_opened": False,
            "feature_contract_generation_sha256": feature_contract.get("generation_sha256"),
            "n_frozen_proteins": len(proteins),
            "source_files": feature_contract.get("source_files", {}),
        },
        "exact_context_keys": list(contract.MATCH_KEYS),
        "rules": {
            "inferred_parity": {
                "name": INFERRED_MAPPING_NAME,
                "status": "inferred_mapping",
                "official": False,
                "allowed_use": "experimental FC loss on split_final=train only",
                "mapping": inferred_parity_mapping(),
                "quality_control": {"pert_id": "#48", "name": "Quality Control"},
                "frozen_without_validation_or_test_scores": True,
            },
            "pooled_context": {
                "name": POOLED_RULE_NAME,
                "status": "sensitivity_comparator_only",
                "official": False,
                "allowed_use": "diagnostic comparison only",
                "rule": "protein-wise mean of all observed Water and DMSO controls in the same exact context",
            },
        },
        "train_metadata_checks": {
            "quality_control_pert_id_48_consistent": qc_consistent,
            "pert_id_48_observed_names": qc_names,
            "quality_control_rows": int(qc_rows.sum()),
        },
        "coverage": {
            "inferred_parity": _coverage(total, parity_pairs, parity_failures),
            "pooled_context": _coverage(total, pooled_pairs, pooled_failures),
        },
        "coverage_diagnostics": {
            "inferred_parity_unmatched_by_expected_solvent": parity_failure_by_solvent,
            "prior_audit_coverage_is_not_directly_comparable": (
                "the prior broad metadata diagnostic allowed controls from all splits; "
                "this sensitivity audit restricts both treatment and observed control labels to train"
            ),
        },
        "comparison_on_common_matched_and_valid_mask": comparison,
        "native_matched_high_effect": {
            "inferred_parity": high_effect_summary(parity_fc, parity_mask),
            "pooled_context": high_effect_summary(pooled_fc, pooled_mask),
        },
        "metric_contract": {
            "fc": "log2(treatment raw intensity) - protein-wise mean log2(control raw intensity)",
            "control_aggregation": "protein-wise mean over observed controls",
            "valid_raw_intensity": "finite and strictly greater than zero",
            "comparison_mask": "parity treatment/control valid AND pooled treatment/control valid AND both FC finite",
            "fc_pearson": "Pearson over the globally flattened common-valid sample-protein entries",
            "direction_consistency": "mean(sign(FC_parity) == sign(FC_pooled)) on the comparison mask",
            "high_effect": "strict abs(FC) > 1",
        },
    }


def render_report(result: Mapping[str, object]) -> str:
    coverage = result["coverage"]
    parity = coverage["inferred_parity"]
    pooled = coverage["pooled_context"]
    comparison = result["comparison_on_common_matched_and_valid_mask"]
    native = result["native_matched_high_effect"]
    common_high = comparison["high_effect_on_common_subset"]
    return f"""# Water/DMSO 映射敏感性测试报告

生成日期：{result['generated_on']}

## 结论

本报告是现有数据合同之上的独立、仅 train 的敏感性分析，不推翻原审计。官方 Water/DMSO 映射继续为 `BLOCKED_PENDING_ORGANIZER_CONFIRMATION`。奇偶规则只标记为 `inferred_mapping`，仅可用于实验性 train FC loss；合并均值规则只作诊断对照。不得依据验证或测试分数在两者间选择。

## 已确认事实

- 所有蛋白计算仅使用 `split_final=train` 的 {result['data_scope']['fit_sample_count']} 个样本和已冻结的 {result['data_scope']['n_frozen_proteins']} 个蛋白。
- 未打开 test proteome；没有将 validation protein cells 载入计算数据框或统计量。
- exact context 键：`{'`, `'.join(result['exact_context_keys'])}`。
- `#48` 在 train 中有 {result['train_metadata_checks']['quality_control_rows']} 行，全部为 Quality Control：{result['train_metadata_checks']['quality_control_pert_id_48_consistent']}。

## 冻结的实验规则

- `pert_id #1–#47`：奇数 → Water，偶数 → DMSO。
- `#48` → Quality Control，排除于 treatment/control FC 配对。
- 该映射是 `inferred_mapping`，不是主办方确认规则，不可用于官方 FC 口径。
- 对照规则：同一 exact context 下所有 Water/DMSO control 按蛋白、仅在有效观测上合并均值。

## 匹配覆盖率

| 规则 | 可配 treatment | 成功匹配 | 失败 | 覆盖率 | 失败原因 |
|---|---:|---:|---:|---:|---|
| inferred parity | {parity['eligible_train_treatments']} | {parity['matched']} | {parity['unmatched']} | {parity['coverage_rate']:.6%} | `{json.dumps(parity['failure_reasons'], ensure_ascii=False)}` |
| pooled context | {pooled['eligible_train_treatments']} | {pooled['matched']} | {pooled['unmatched']} | {pooled['coverage_rate']:.6%} | `{json.dumps(pooled['failure_reasons'], ensure_ascii=False)}` |

pooled 成功匹配样本的 control 构成：`{json.dumps(pooled['matched_control_composition'], ensure_ascii=False)}`。奇偶规则失败按预期溶剂分解：`{json.dumps(result['coverage_diagnostics']['inferred_parity_unmatched_by_expected_solvent'], ensure_ascii=False)}`。

原审计中的候选覆盖率诊断允许从全部 split 的元数据中寻找 control；本次按新增要求将 treatment 与 observed control label 池都严格限制为 train。因此两组覆盖数字不可直接互换，这一差异不修改原审计事实或官方 BLOCKED 状态。

## FC 敏感性结果

为避免覆盖差异造成混杂，以下核心比较仅使用两条规则都成功匹配的 {comparison['matched_treatments']} 个 treatment，并使用共同有效 mask（{comparison['joint_valid_sample_protein_entries']} 个样本×蛋白位置）。

| 指标 | 结果 |
|---|---:|
| FC Pearson（全局展平、共同 mask） | {comparison['fc_pearson_global_flattened']:.9f} |
| 方向一致率 | {comparison['direction_consistency_rate']:.9%} |
| FC 平均绝对差 | {comparison['mean_absolute_fc_difference']:.9f} |
| 两规则 FC 同为精确 0 的位置 | {comparison['both_exact_zero_entries']} |

方向一致定义为共同 mask 上 `sign(FC_parity) == sign(FC_pooled)`；精确零仅在符号相等时计为一致。

### 高效应蛋白（严格 `abs(FC) > 1`）

| 统计范围 | 规则 | 样本×蛋白位置数 | 去重蛋白数 | 每 treatment 中位数 |
|---|---|---:|---:|---:|
| 各自全部成功匹配样本 | inferred parity | {native['inferred_parity']['sample_protein_entries']} | {native['inferred_parity']['unique_proteins']} | {native['inferred_parity']['per_treatment_median']:.3f} |
| 各自全部成功匹配样本 | pooled context | {native['pooled_context']['sample_protein_entries']} | {native['pooled_context']['unique_proteins']} | {native['pooled_context']['per_treatment_median']:.3f} |
| 共同匹配且共同有效 | inferred parity | {common_high['inferred_parity']['sample_protein_entries']} | {common_high['inferred_parity']['unique_proteins']} | {common_high['inferred_parity']['per_treatment_median']:.3f} |
| 共同匹配且共同有效 | pooled context | {common_high['pooled_context']['sample_protein_entries']} | {common_high['pooled_context']['unique_proteins']} | {common_high['pooled_context']['per_treatment_median']:.3f} |

“去重蛋白数”表示至少在一个纳入样本中达到阈值的 frozen protein 数；它不等同于高效应事件数，所以同时给出样本×蛋白位置数。

## 防泄漏与使用限制

- 敏感性规则在 train 元数据上冻结，未使用 validation/test 分数选择。
- 混合 train/validation proteome CSV 先只读取 `sample_ID`；蛋白读取阶段用 `skiprows` 在解析蛋白单元格之前排除全部非 train 行。
- 因源文件物理上混合 train/validation，顺序扫描文件字节不可避免；但 validation 蛋白单元格不被 tokenization、物化或用于任何统计。
- 脚本不接受或打开 test proteome 路径。
- `inferred_parity` 仅供实验性 FC loss；`pooled_context` 仅为敏感性比较；两者都不得替代官方 BLOCKED 映射。
- 本次未下载或引入任何外部资源，因此无新增 URL、许可或外部资源校验记录。

## 尚待主办方确认

- 权威 `pert_id → Water/DMSO` 映射。
- 是否存在 exact context、溶剂匹配之外的样本级额外 QC。
- 多个匹配 control 的官方聚合规则。
"""


def default_paths(data_dir: Path, contract_dir: Path) -> dict[str, Path]:
    return {
        "metadata": data_dir / "WAYB_WAYC_metadata_train_val(1).csv",
        "proteome": data_dir / "WAYB_WAYC_proteome_raw_train_val.csv",
        "feature_contract": contract_dir / "feature_contract.json",
        "result": contract_dir / "control_mapping_sensitivity_results.json",
        "report": contract_dir / "CONTROL_MAPPING_SENSITIVITY_REPORT.md",
    }


def audit_from_files(paths: Mapping[str, Path]) -> dict[str, object]:
    feature_contract = json.loads(paths["feature_contract"].read_text(encoding="utf-8"))
    metadata = pd.read_csv(paths["metadata"])
    train_proteome = read_train_proteome_without_parsing_validation_labels(
        paths["proteome"], metadata, feature_contract["proteins"]
    )
    return run_sensitivity(metadata, train_proteome, feature_contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "WAYB_WAYC",
    )
    parser.add_argument("--contract-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paths = default_paths(args.data_dir, args.contract_dir)
    result = audit_from_files(paths)
    paths["result"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"result": str(paths["result"]), "report": str(paths["report"])}))


if __name__ == "__main__":
    main()
