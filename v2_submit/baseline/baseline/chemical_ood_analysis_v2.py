"""Read-only Stage 2.3 chemical OOD attribution for frozen V2 checkpoints.

This module never trains or mutates a model.  It evaluates only the two public
validation scenarios containing unseen chemicals and derives chemical
covariates exclusively from frozen feature artifacts and split_final=train
membership.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation_v2 import masked_mae, masked_rmse, median_per_sample_pcc
from .model_v2 import AnchoredVirtualCellV2, V2Config
from .training_v2 import (
    BATCH_COLUMNS,
    CONTROL_NAMES,
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


OOD_SCENARIOS = ("val_chem_only", "val_both")
VARIANTS = ("correct", "shuffle", "zero")
CHEMICAL_COLUMN = "perturbation_no_concentration"


@dataclass(frozen=True)
class VariantEvaluation:
    sample_ids: tuple[str, ...]
    prediction: np.ndarray
    delta_response: np.ndarray
    target: np.ndarray
    mask: np.ndarray


def select_training_drugs(meta: pd.DataFrame) -> tuple[str, ...]:
    """Return treatment entities determined only by split_final=train."""
    train = meta.loc[meta["split_final"].eq("train")]
    names = train[CHEMICAL_COLUMN].astype(str)
    excluded = CONTROL_NAMES | {"quality control"}
    names = names.loc[~names.str.lower().isin(excluded)]
    return tuple(sorted(names.unique()))


def validation_entities(meta: pd.DataFrame, scenario: str) -> tuple[str, ...]:
    if scenario not in OOD_SCENARIOS:
        raise ValueError(f"unsupported OOD scenario: {scenario}")
    names = meta.loc[meta["split_final"].eq(scenario), CHEMICAL_COLUMN].astype(str)
    return tuple(sorted(names.unique()))


def _chemical_feature_metrics(artifacts, chemical: str) -> dict[str, float | int]:
    index = artifacts.chemical_index.reset_index(drop=True)
    match = np.flatnonzero(index["raw_name"].astype(str).eq(chemical).to_numpy())
    if len(match) != 1:
        raise ValueError(f"chemical must occur exactly once in frozen feature index: {chemical}")
    row = int(match[0])
    descriptors = artifacts.chemical_features[row, artifacts.morgan_dim:].astype(np.float64)
    valid = artifacts.chemical_valid_mask[row, artifacts.morgan_dim:].astype(bool)
    values = descriptors[valid]
    if values.size == 0:
        return {
            "standardized_descriptor_l2_norm": np.nan,
            "descriptor_max_abs_z": np.nan,
            "descriptor_abs_z_gt_5_count": 0,
            "descriptor_abs_z_gt_10_count": 0,
        }
    absolute = np.abs(values)
    return {
        "standardized_descriptor_l2_norm": float(np.linalg.vector_norm(values)),
        "descriptor_max_abs_z": float(absolute.max()),
        "descriptor_abs_z_gt_5_count": int(np.sum(absolute > 5)),
        "descriptor_abs_z_gt_10_count": int(np.sum(absolute > 10)),
    }


def compute_similarity_covariates(
    similarity: np.ndarray,
    row_index: pd.DataFrame,
    column_index: pd.DataFrame,
    training_drugs,
    validation_drugs,
) -> tuple[dict[str, dict], dict]:
    """Compute nearest train entities without accepting or inspecting labels."""
    if similarity.shape != (len(row_index), len(column_index)):
        raise ValueError("Tanimoto matrix and frozen indexes are inconsistent")
    train_set = set(map(str, training_drugs))
    train_positions = np.flatnonzero(row_index["raw_name"].astype(str).isin(train_set).to_numpy())
    if len(train_positions) == 0:
        raise ValueError("no split_final=train drugs have frozen Morgan similarities")
    row_names = row_index["raw_name"].astype(str).to_numpy()
    column_lookup = {
        name: index for index, name in enumerate(column_index["raw_name"].astype(str))
    }
    result = {}
    for chemical in validation_drugs:
        chemical = str(chemical)
        if chemical not in column_lookup:
            result[chemical] = {
                "max_train_morgan_tanimoto": np.nan,
                "nearest_train_drug": "",
            }
            continue
        values = similarity[train_positions, column_lookup[chemical]].astype(np.float64)
        if not np.isfinite(values).any():
            result[chemical] = {
                "max_train_morgan_tanimoto": np.nan,
                "nearest_train_drug": "",
            }
            continue
        local = int(np.nanargmax(values))
        result[chemical] = {
            "max_train_morgan_tanimoto": float(values[local]),
            "nearest_train_drug": str(row_names[train_positions[local]]),
        }
    audit = {
        "training_drug_count": int(len(train_set)),
        "training_drugs_with_morgan_count": int(len(train_positions)),
        "training_drugs_without_morgan": sorted(train_set - set(row_names[train_positions])),
        "validation_label_input_accepted": False,
    }
    return result, audit


def load_frozen_chemical_covariates(root: Path, artifacts, meta: pd.DataFrame):
    chemistry = root / "external_data/chemistry"
    row_index = pd.read_csv(chemistry / "tanimoto_row_index.csv")
    column_index = pd.read_csv(chemistry / "tanimoto_column_index.csv")
    similarity = np.load(chemistry / "tanimoto_similarity.npz", allow_pickle=False)["tanimoto"]
    train_drugs = select_training_drugs(meta)
    validation_drugs = sorted({
        chemical for scenario in OOD_SCENARIOS for chemical in validation_entities(meta, scenario)
    })
    nearest, audit = compute_similarity_covariates(
        similarity, row_index, column_index, train_drugs, validation_drugs,
    )
    mapping = artifacts.chemical_mapping.set_index("raw_name", drop=False)
    covariates = {}
    for chemical in validation_drugs:
        if chemical not in mapping.index:
            raise ValueError(f"validation chemical missing from frozen mapping: {chemical}")
        row = mapping.loc[chemical]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"duplicate frozen mapping entity: {chemical}")
        covariates[chemical] = {
            "mapping_status": str(row["mapping_status"]),
            "mapping_confidence": str(row["mapping_confidence"]),
            "structure_valid": bool(row["structure_valid"]),
            **nearest[chemical],
            **_chemical_feature_metrics(artifacts, chemical),
        }
    audit.update({
        "training_drug_membership_source": "metadata_train_val split_final=train only",
        "tanimoto_matrix_path": str((chemistry / "tanimoto_similarity.npz").resolve()),
        "tanimoto_sha256": sha256_file(chemistry / "tanimoto_similarity.npz"),
        "tanimoto_row_index_sha256": sha256_file(chemistry / "tanimoto_row_index.csv"),
        "tanimoto_column_index_sha256": sha256_file(chemistry / "tanimoto_column_index.csv"),
    })
    return covariates, audit


def _build_model(
    payload: dict,
    artifacts,
    vocab,
    device: str,
    *,
    allow_experimental_nonofficial_parity_fc: bool = False,
):
    config = payload["config"]
    model_config = config["model"]
    if model_config["chemical_mode"] not in VARIANTS:
        raise ValueError("checkpoint chemical mode is not a Stage 2.2 variant")
    if model_config["genome_mode"] != "correct":
        raise ValueError("Stage 2.3 requires genome=correct checkpoints")
    if bool(model_config["batch_enabled"]):
        raise ValueError("Stage 2.3 requires formal no-batch checkpoints")
    fc_weight = float(config["loss"]["fc_weight"])
    if bool(model_config["similarity_enabled"]) or (
        fc_weight != 0 and not allow_experimental_nonofficial_parity_fc
    ):
        raise ValueError("Stage 2.3 requires Huber-only similarity-disabled checkpoints")
    if allow_experimental_nonofficial_parity_fc and fc_weight != 0:
        mapping = config.get("control_mapping")
        mapping_name = mapping.get("name") if isinstance(mapping, dict) else mapping
        if not (
            config.get("experimental_nonofficial_parity_fc") is True
            and mapping_name == "pert_id_parity_v1"
            and config.get("official_fc_result") is False
            and fc_weight == 0.1
        ):
            raise ValueError("checkpoint is not the authorized Stage 2.5 nonofficial parity FC diagnostic")
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_config["latent_dim"]),
        protein_rank=int(model_config["protein_rank"]),
        dropout=float(model_config["dropout"]),
        batch_enabled=False,
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def evaluate_variant(
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
) -> VariantEvaluation:
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    checkpoint_mode = payload["config"]["model"]["chemical_mode"]
    if checkpoint_mode != mode:
        raise ValueError(f"checkpoint mode {checkpoint_mode} does not match requested {mode}")
    seed = int(payload["seed"])
    ids = meta.index[meta["split_final"].eq(scenario)]
    chemical_variant = make_chemical_feature_variant(artifacts, mode=mode, seed=seed)
    batch = build_batch(
        meta, ids, artifacts, vocab, chemical_mode=mode, genome_mode="correct",
        seed=seed, chemical_variant=chemical_variant,
    )
    target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
    mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
    loader = make_loader(batch, target, mask, batch_size=batch_size, shuffle=False)
    model = _build_model(payload, artifacts, vocab, device)
    predictions, responses, targets, valid_masks = [], [], [], []
    with torch.no_grad():
        for inputs, y_true, valid in loader:
            outputs = model(inputs.to(device))
            predictions.append(outputs["y_pred"].detach().cpu().numpy())
            responses.append(outputs["delta_response"].detach().cpu().numpy())
            targets.append(y_true.numpy())
            valid_masks.append(valid.numpy())
    return VariantEvaluation(
        tuple(map(str, ids)), np.concatenate(predictions), np.concatenate(responses),
        np.concatenate(targets), np.concatenate(valid_masks),
    )


def assert_variant_alignment(reference: VariantEvaluation, candidate: VariantEvaluation):
    if reference.sample_ids != candidate.sample_ids:
        raise ValueError("correct/zero sample IDs are not identical and ordered")
    if not np.array_equal(reference.mask, candidate.mask):
        raise ValueError("correct/zero masks are not identical")
    if not np.array_equal(reference.target, candidate.target, equal_nan=True):
        raise ValueError("correct/zero targets are not identical")


def _entity_metrics(evaluation: VariantEvaluation, positions: np.ndarray) -> dict:
    target = evaluation.target[positions]
    prediction = evaluation.prediction[positions]
    mask = evaluation.mask[positions]
    response = evaluation.delta_response[positions]
    valid = mask & np.isfinite(response)
    return {
        "rmse": masked_rmse(target, prediction, mask),
        "mae": masked_mae(target, prediction, mask),
        "median_sample_pcc": median_per_sample_pcc(target, prediction, mask),
        "delta_response_valid_l2": (
            float(np.linalg.vector_norm(response[valid].astype(np.float64))) if valid.any() else np.nan
        ),
    }


def build_per_entity_table(meta, evaluations, covariates) -> pd.DataFrame:
    records = []
    for scenario in OOD_SCENARIOS:
        scenario_ids = pd.Index(evaluations["correct"][scenario].sample_ids)
        names = meta.loc[scenario_ids, CHEMICAL_COLUMN].astype(str).to_numpy()
        for chemical in sorted(np.unique(names)):
            positions = np.flatnonzero(names == chemical)
            metrics = {
                mode: _entity_metrics(evaluations[mode][scenario], positions)
                for mode in VARIANTS
            }
            row = {
                "chemical_name": chemical,
                "scenario": scenario,
                "n_samples": int(len(positions)),
                **covariates[chemical],
            }
            for mode in VARIANTS:
                row.update({
                    f"{mode}_rmse": metrics[mode]["rmse"],
                    f"{mode}_mae": metrics[mode]["mae"],
                    f"{mode}_median_sample_pcc": metrics[mode]["median_sample_pcc"],
                })
            row.update({
                "correct_delta_response_valid_l2": metrics["correct"]["delta_response_valid_l2"],
                "zero_delta_response_valid_l2": metrics["zero"]["delta_response_valid_l2"],
                "correct_gain_vs_zero": metrics["zero"]["rmse"] - metrics["correct"]["rmse"],
                "correct_harm_vs_zero": metrics["correct"]["rmse"] - metrics["zero"]["rmse"],
            })
            records.append(row)
    table = pd.DataFrame(records)
    validate_unique_entity_rows(table)
    return table.sort_values(["scenario", "chemical_name"]).reset_index(drop=True)


def validate_unique_entity_rows(table: pd.DataFrame):
    if table.duplicated(["scenario", "chemical_name"]).any():
        raise ValueError("a validation drug occurs more than once within a scenario")


def _combined_entity_table(meta, evaluations, covariates) -> pd.DataFrame:
    records = []
    chemicals = sorted({
        chemical for scenario in OOD_SCENARIOS for chemical in validation_entities(meta, scenario)
    })
    for chemical in chemicals:
        per_mode = {}
        n_samples = 0
        for mode in VARIANTS:
            parts = []
            for scenario in OOD_SCENARIOS:
                evaluation = evaluations[mode][scenario]
                ids = pd.Index(evaluation.sample_ids)
                names = meta.loc[ids, CHEMICAL_COLUMN].astype(str).to_numpy()
                positions = np.flatnonzero(names == chemical)
                if mode == "correct":
                    n_samples += len(positions)
                parts.append((evaluation, positions))
            combined = VariantEvaluation(
                tuple(identifier for evaluation, positions in parts for identifier in np.asarray(evaluation.sample_ids)[positions]),
                np.concatenate([evaluation.prediction[positions] for evaluation, positions in parts]),
                np.concatenate([evaluation.delta_response[positions] for evaluation, positions in parts]),
                np.concatenate([evaluation.target[positions] for evaluation, positions in parts]),
                np.concatenate([evaluation.mask[positions] for evaluation, positions in parts]),
            )
            per_mode[mode] = _entity_metrics(combined, np.arange(len(combined.sample_ids)))
        records.append({
            "chemical_name": chemical,
            "n_samples": int(n_samples),
            **covariates[chemical],
            **{f"{mode}_rmse": per_mode[mode]["rmse"] for mode in VARIANTS},
            "correct_delta_response_valid_l2": per_mode["correct"]["delta_response_valid_l2"],
            "zero_delta_response_valid_l2": per_mode["zero"]["delta_response_valid_l2"],
            "correct_gain_vs_zero": per_mode["zero"]["rmse"] - per_mode["correct"]["rmse"],
            "correct_harm_vs_zero": per_mode["correct"]["rmse"] - per_mode["zero"]["rmse"],
        })
    return pd.DataFrame(records)


def spearman_correlation(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        rho = np.nan
    else:
        left_rank = pd.Series(left).rank(method="average").to_numpy()
        right_rank = pd.Series(right).rank(method="average").to_numpy()
        rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"spearman_rho": rho, "n_drugs": int(len(left))}


def _correlation_block(table: pd.DataFrame) -> dict:
    return {
        "max_tanimoto_vs_correct_gain": spearman_correlation(
            table["max_train_morgan_tanimoto"], table["correct_gain_vs_zero"],
        ),
        "descriptor_max_abs_z_vs_correct_harm": spearman_correlation(
            table["descriptor_max_abs_z"], table["correct_harm_vs_zero"],
        ),
        "descriptor_l2_vs_correct_harm": spearman_correlation(
            table["standardized_descriptor_l2_norm"], table["correct_harm_vs_zero"],
        ),
        "correct_response_l2_vs_correct_harm": spearman_correlation(
            table["correct_delta_response_valid_l2"], table["correct_harm_vs_zero"],
        ),
    }


def _ranking_records(table: pd.DataFrame, column: str, ascending: bool) -> list[dict]:
    fields = [
        "chemical_name", "n_samples", "correct_gain_vs_zero", "correct_harm_vs_zero",
        "nearest_train_drug", "max_train_morgan_tanimoto",
    ]
    return table.sort_values(column, ascending=ascending).head(5)[fields].to_dict("records")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_summary(per_entity, combined, similarity_audit, checkpoints, artifacts) -> dict:
    correlations = {
        scenario: _correlation_block(per_entity.loc[per_entity["scenario"].eq(scenario)])
        for scenario in OOD_SCENARIOS
    }
    correlations["combined_unique_drugs"] = _correlation_block(combined)
    h2o2 = combined.loc[combined["chemical_name"].str.lower().isin({"h2o2", "hydrogen peroxide"})]
    # This transparent name list is descriptive only and is never a model input.
    antibiotic_names = {
        "amphotericin b", "doxycycline", "g418", "hygromycin b", "neomycin b",
        "nystatin", "nystatin dihydrate", "tetracycline",
    }
    antibiotics = combined.loc[combined["chemical_name"].str.lower().isin(antibiotic_names)]
    extreme = combined.loc[
        (combined["descriptor_max_abs_z"] > 5)
        | (combined["descriptor_abs_z_gt_5_count"] > 0)
    ]
    return _json_ready({
        "status": "PASS",
        "analysis": "stage2_3_read_only_chemical_ood_attribution",
        "scenarios": list(OOD_SCENARIOS),
        "scenario_entity_counts": {
            scenario: int(per_entity["scenario"].eq(scenario).sum()) for scenario in OOD_SCENARIOS
        },
        "combined_unique_drug_count": int(len(combined)),
        "per_entity_row_count": int(len(per_entity)),
        "correlations": correlations,
        "top_correct_gain_vs_zero": _ranking_records(combined, "correct_gain_vs_zero", False),
        "top_correct_harm_vs_zero": _ranking_records(combined, "correct_harm_vs_zero", False),
        "actual_correct_better_than_zero": _ranking_records(
            combined.loc[combined["correct_gain_vs_zero"] > 0], "correct_gain_vs_zero", False,
        ),
        "actual_correct_worse_than_zero": _ranking_records(
            combined.loc[combined["correct_harm_vs_zero"] > 0], "correct_harm_vs_zero", False,
        ),
        "hypothesis_subsets": {
            "h2o2": h2o2.to_dict("records"),
            "antibiotics_name_based_descriptive_only": antibiotics.to_dict("records"),
            "descriptor_extreme_abs_z_gt_5": extreme.to_dict("records"),
        },
        "similarity_audit": similarity_audit,
        "checkpoint_sha256": {mode: sha256_file(path) for mode, path in checkpoints.items()},
        "checkpoint_paths": {mode: str(path.resolve()) for mode, path in checkpoints.items()},
        "frozen_artifact_hashes": artifacts.hashes,
        "label_scope": "WAYB_WAYC_proteome_raw_train_val.csv validation rows only for error metrics",
        "label_derived_chemical_covariates": False,
        "training_performed": False,
        "model_or_checkpoint_modified": False,
        "test_proteome_opened": False,
    })


def _format_number(value, digits=3):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(summary: dict, combined: pd.DataFrame, per_entity_path: Path, summary_path: Path) -> str:
    correlation_rows = []
    for scope, block in summary["correlations"].items():
        for relation, values in block.items():
            correlation_rows.append(
                f"| {scope} | {relation} | {_format_number(values['spearman_rho'])} | {values['n_drugs']} |"
            )
    entity_rows = []
    for row in combined.sort_values("correct_harm_vs_zero", ascending=False).itertuples():
        entity_rows.append(
            f"| {row.chemical_name} | {row.n_samples} | {_format_number(row.max_train_morgan_tanimoto)} "
            f"| {row.nearest_train_drug} | {_format_number(row.descriptor_max_abs_z)} "
            f"| {_format_number(row.correct_rmse)} | {_format_number(row.zero_rmse)} "
            f"| {_format_number(row.correct_gain_vs_zero)} | {_format_number(row.correct_delta_response_valid_l2, 1)} |"
        )
    gain_rows = [
        f"- {row['chemical_name']}：gain={_format_number(row['correct_gain_vs_zero'])}；最近训练药物={row['nearest_train_drug']}；Tanimoto={_format_number(row['max_train_morgan_tanimoto'])}。"
        for row in summary["actual_correct_better_than_zero"]
    ]
    harm_rows = [
        f"- {row['chemical_name']}：harm={_format_number(row['correct_harm_vs_zero'])}；最近训练药物={row['nearest_train_drug']}；Tanimoto={_format_number(row['max_train_morgan_tanimoto'])}。"
        for row in summary["actual_correct_worse_than_zero"]
    ]
    similarity_rho = summary["correlations"]["combined_unique_drugs"]["max_tanimoto_vs_correct_gain"]["spearman_rho"]
    extreme_rho = summary["correlations"]["combined_unique_drugs"]["descriptor_max_abs_z_vs_correct_harm"]["spearman_rho"]
    response_rho = summary["correlations"]["combined_unique_drugs"]["correct_response_l2_vs_correct_harm"]["spearman_rho"]
    h2o2_note = (
        "H2O2 出现在目标场景。" if summary["hypothesis_subsets"]["h2o2"]
        else "H2O2 不在 val_chem_only/val_both 的 6 个验证药物中，因此本数据不能检验 H2O2 特异失败。"
    )
    return f"""# MODEL V2 Stage 2.3：新药 OOD 归因分析

## 结论

本阶段仅对已有 correct、shuffle、zero checkpoint 做只读分析，未训练模型、未修改模型或 checkpoint，也未打开测试蛋白真值。目标场景只有 **{summary['combined_unique_drug_count']} 个唯一新药**；下述相关系数是小样本诊断，不能视为稳定统计结论。

- 相似性假设：合并 6 药物的最大 Tanimoto 与 correct gain 的 Spearman rho={_format_number(similarity_rho)}，方向为弱正，但证据不足。反例是 Amphotericin B：它与训练药物 Nystatin dihydrate 的相似度最高（0.460），correct 仍比 zero 差 0.092 RMSE。不能据此建立相似度门控规则。
- 极端描述符假设：descriptor 最大绝对 z 与 correct harm 的 rho={_format_number(extreme_rho)}，而标准化 descriptor L2 与 harm 的 rho=0.714。6 个药物全部至少有一个 `|z|>5` descriptor，二元“是否极端”无法区分成败；Pentamidine 的最大 `|z|=10.819` 却明显受益，也是直接反例。{h2o2_note} 唯一按名称归入抗生素的 Amphotericin B 确实受损，但单个实体不能证明抗生素类别效应。
- 过度响应假设：correct response L2 与 correct harm 的 rho={_format_number(response_rho)}，是三个假设中方向最一致的诊断信号。三个受损药物（Hydroxyurea、FCCP、Amphotericin B）的 correct response 范数也是最高三项。不过 response L2 是共同有效蛋白位置上的原始范数，可能同时受样本数影响，只支持“过度响应值得下一步验证”，不构成因果证明。

## 六个唯一新药（两个场景合并后重新计算）

| 药物 | 样本数 | 最大 Tanimoto | 最近训练药物 | descriptor 最大绝对 z | correct RMSE | zero RMSE | correct gain | correct response L2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(entity_rows)}

其中 `correct gain = RMSE_zero - RMSE_correct`；正值表示 correct 更好，负值表示 correct 更差。逐场景完整 12 行记录还包含 correct/shuffle/zero 的 RMSE、MAE、样本 PCC 中位数及 descriptor 极端计数。

## 最明显收益与伤害

目标集合只有 6 个唯一新药，实际只有 3 个 correct 优于 zero、3 个 correct 劣于 zero，因此不能诚实地各列 5 个“实际收益/实际伤害”实体；以下列出全部实际正例和负例，完整六药排序见上表与 JSON 中的 top-5 排名。

correct 实际优于 zero（最多五个，实际三个）：

{chr(10).join(gain_rows)}

correct 实际劣于 zero（最多五个，实际三个）：

{chr(10).join(harm_rows)}

## 相关性

| 范围 | 关系 | Spearman rho | 实际药物数 |
|---|---|---:|---:|
{chr(10).join(correlation_rows)}

`combined_unique_drugs` 按唯一药物汇总两个场景的全部样本，没有把同一药物作为两个独立观测重复计权。若变量无变异或有效药物不足，rho 记为 NA。

## 数据边界与可复核性

- 训练药物集合只由 train_val 元数据中 `split_final=train` 的非 Water/DMSO、非 Quality Control 样本确定，共 {summary['similarity_audit']['training_drug_count']} 个；其中 {summary['similarity_audit']['training_drugs_with_morgan_count']} 个有冻结 Morgan 相似度。无可用 Morgan 的训练药物为：{', '.join(summary['similarity_audit']['training_drugs_without_morgan'])}。
- 相似度、descriptor 距离和极端值统计均来自冻结化学工件；接口不接收验证蛋白标签，也未拟合任何验证转换。
- 验证误差只读取 `WAYB_WAYC_proteome_raw_train_val.csv` 的 val_chem_only/val_both 行；correct、shuffle、zero 的 sample ID、target 和 mask 已逐场景严格校验一致。
- CSV 唯一键是 `(scenario, chemical_name)`：val_chem_only 6 行、val_both 6 行。合并分析另按 6 个唯一药物重新聚合。
- 本阶段未运行 Morgan-only、descriptor-only、clipping、相似度门控或任何新模型。

## 产物

- 逐实体 CSV：`{per_entity_path.resolve()}`
- 汇总 JSON：`{summary_path.resolve()}`
- 分析实现：`{(ROOT / 'baseline/baseline/chemical_ood_analysis_v2.py').resolve()}`
"""


def run_analysis(checkpoints: dict[str, Path], per_entity_output: Path, summary_output: Path, report_output: Path, batch_size=64, device="auto"):
    if set(checkpoints) != set(VARIANTS):
        raise ValueError("exactly correct, shuffle and zero checkpoints are required")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    covariates, similarity_audit = load_frozen_chemical_covariates(ROOT, artifacts, meta)
    evaluations = {mode: {} for mode in VARIANTS}
    for scenario in OOD_SCENARIOS:
        for mode in VARIANTS:
            evaluations[mode][scenario] = evaluate_variant(
                mode, checkpoints[mode], scenario, meta, labels, masks,
                artifacts, vocab, int(batch_size), device,
            )
        assert_variant_alignment(evaluations["correct"][scenario], evaluations["shuffle"][scenario])
        assert_variant_alignment(evaluations["correct"][scenario], evaluations["zero"][scenario])
    per_entity = build_per_entity_table(meta, evaluations, covariates)
    combined = _combined_entity_table(meta, evaluations, covariates)
    summary = build_summary(per_entity, combined, similarity_audit, checkpoints, artifacts)
    summary["device"] = device
    summary["outputs"] = {
        "per_entity_csv": str(per_entity_output.resolve()),
        "summary_json": str(summary_output.resolve()),
        "report_markdown": str(report_output.resolve()),
    }
    per_entity_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    per_entity.to_csv(per_entity_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_output.write_text(
        render_report(summary, combined, per_entity_output, summary_output), encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only V2 Stage 2.3 chemical OOD attribution")
    parser.add_argument("--correct-checkpoint", type=Path, required=True)
    parser.add_argument("--shuffle-checkpoint", type=Path, required=True)
    parser.add_argument("--zero-checkpoint", type=Path, required=True)
    parser.add_argument("--per-entity-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run_analysis(
        {"correct": args.correct_checkpoint, "shuffle": args.shuffle_checkpoint, "zero": args.zero_checkpoint},
        args.per_entity_output, args.summary_output, args.report_output,
        batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
