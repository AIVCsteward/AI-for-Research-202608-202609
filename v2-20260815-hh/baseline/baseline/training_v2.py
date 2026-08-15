"""Canonical V2 data, leakage guards, staged training and checkpointing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.build_chemical_features import apply_feature_variant

from .evaluation_v2 import (
    VAL_SCENARIOS, evaluate_batch_diagnostics, evaluate_component_norms,
    evaluate_four_scenarios,
)
from .losses_v2 import LossWeights, compute_losses, masked_huber_sum_count
from .model_v2 import AnchoredVirtualCellV2, V2Batch, V2Config


ROOT = Path(__file__).resolve().parents[2]
CONTROL_NAMES = frozenset({"water", "dmso"})
BATCH_COLUMNS = ("data_source", "instrument", "Yeast_cell_plate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_parameters_sha256(model, module_names):
    digest = hashlib.sha256()
    for module_name in module_names:
        module = getattr(model, module_name)
        for name, parameter in sorted(module.named_parameters()):
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{module_name}.{name}|{tuple(value.shape)}|{value.dtype}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def named_parameter_prefixes_sha256(model, prefixes):
    """Hash an exact, sorted parameter subset selected by fully-qualified prefixes."""
    prefixes = tuple(str(prefix) for prefix in prefixes)
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(model.named_parameters()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(value.shape)}|{value.dtype}".encode())
        digest.update(value.numpy().tobytes())
        matched += 1
    if matched == 0:
        raise ValueError(f"no parameters matched prefixes: {prefixes}")
    return digest.hexdigest()


def snapshot_module_parameters(model, module_name):
    return {name: value.detach().cpu().clone() for name, value in getattr(model, module_name).named_parameters()}


def parameter_change_l2(model, module_name, initial):
    total = 0.0
    current = dict(getattr(model, module_name).named_parameters())
    if set(current) != set(initial):
        raise ValueError("module parameter set changed during training")
    for name, parameter in current.items():
        difference = parameter.detach().cpu().double() - initial[name].double()
        total += float(difference.square().sum())
    return float(np.sqrt(total))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_config(path: Path) -> dict:
    """Configs use JSON syntax, which is a strict YAML 1.2 subset."""
    return read_json(path)


@dataclass(frozen=True)
class FeatureContract:
    proteins: tuple[str, ...]
    n_proteins: int
    protein_order_sha256: str
    generation_sha256: str


@dataclass(frozen=True)
class ArtifactBundle:
    feature_contract: FeatureContract
    chemical_features: np.ndarray
    chemical_valid_mask: np.ndarray
    chemical_index: pd.DataFrame
    chemical_mapping: pd.DataFrame
    genome_features: np.ndarray
    genome_valid_mask: np.ndarray
    genome_index: pd.DataFrame
    morgan_dim: int
    descriptor_dim: int
    genome_dim: int
    hashes: dict[str, str]


def load_feature_contract(path: Path | None = None) -> FeatureContract:
    path = path or ROOT / "project_v2/data_contract/feature_contract.json"
    raw = read_json(path)
    proteins = tuple(raw["proteins"])
    n_proteins = int(raw["n_proteins"])
    if len(proteins) != n_proteins or len(set(proteins)) != n_proteins:
        raise ValueError("invalid frozen protein contract")
    return FeatureContract(proteins, n_proteins, raw["protein_order_sha256"], raw["generation_sha256"])


def load_artifact_bundle(root: Path = ROOT) -> ArtifactBundle:
    paths = {
        "feature_contract": root / "project_v2/data_contract/feature_contract.json",
        "chemical_features": root / "external_data/chemistry/chemical_features.npz",
        "chemical_index": root / "external_data/chemistry/chemical_feature_index.csv",
        "chemical_schema": root / "external_data/chemistry/feature_schema.json",
        "chemical_source_manifest": root / "external_data/chemistry/source_manifest.json",
        "genome_features": root / "external_data/genome/strain_features.npz",
        "genome_index": root / "external_data/genome/strain_feature_index.csv",
        "genome_schema": root / "external_data/genome/genome_feature_schema.json",
        "genome_source_manifest": root / "external_data/genome/source_manifest.json",
    }
    contract = load_feature_contract(paths["feature_contract"])
    chemical_schema = read_json(paths["chemical_schema"])
    genome_schema = read_json(paths["genome_schema"])
    chemical = np.load(paths["chemical_features"], allow_pickle=False)
    genome = np.load(paths["genome_features"], allow_pickle=False)
    chemical_index = pd.read_csv(paths["chemical_index"])
    chemical_mapping = pd.read_csv(root / "external_data/chemistry/compound_mapping.csv")
    genome_index = pd.read_csv(paths["genome_index"])
    morgan_dim = int(chemical_schema["morgan"]["n_bits"])
    descriptor_dim = int(chemical_schema["descriptors"]["count"])
    genome_dim = int(genome_schema["genome_dim"])
    if (morgan_dim, descriptor_dim, int(chemical_schema["chemical_feature_dim"])) != (2048, 217, 2265):
        raise ValueError("unexpected frozen chemical dimensions")
    if chemical["features"].shape != chemical["feature_valid_mask"].shape:
        raise ValueError("chemical mask mismatch")
    if not chemical_index["raw_name"].astype(str).equals(chemical_mapping["raw_name"].astype(str)):
        raise ValueError("compound mapping order does not match frozen chemical feature index")
    if genome["genome_features"].shape != genome["feature_valid_mask"].shape:
        raise ValueError("genome mask mismatch")
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    hashes.update({
        "protein_order_sha256": contract.protein_order_sha256,
        "generation_sha256": contract.generation_sha256,
    })
    return ArtifactBundle(
        contract,
        chemical["features"].astype(np.float32, copy=False),
        chemical["feature_valid_mask"].astype(bool, copy=False),
        chemical_index,
        chemical_mapping,
        genome["genome_features"].astype(np.float32, copy=False),
        genome["feature_valid_mask"].astype(bool, copy=False),
        genome_index,
        morgan_dim, descriptor_dim, genome_dim, hashes,
    )


@dataclass(frozen=True)
class CategoryVocabulary:
    values: dict[str, tuple[str, ...]]

    def size(self, column: str) -> int:
        return len(self.values[column]) + 1


def load_train_val_metadata(root: Path = ROOT) -> pd.DataFrame:
    # Deliberately names the only metadata file the V2 training path may open.
    path = root / "WAYB_WAYC/WAYB_WAYC_metadata_train_val(1).csv"
    meta = pd.read_csv(path)
    if meta["sample_ID"].isna().any() or not meta["sample_ID"].is_unique:
        raise ValueError("sample_ID must be complete and unique")
    return meta.set_index("sample_ID", drop=False)


def _require_fit_ids(meta: pd.DataFrame, fit_sample_ids, allowed_splits=("train",)) -> pd.Index:
    if fit_sample_ids is None:
        raise ValueError("fit_sample_ids must be explicit")
    ids = pd.Index(fit_sample_ids)
    if ids.has_duplicates or not ids.isin(meta.index).all():
        raise ValueError("fit_sample_ids must be unique known IDs")
    if not meta.loc[ids, "split_final"].isin(allowed_splits).all():
        raise ValueError("label statistics may only use the declared fold-train IDs")
    return ids


def fit_category_vocabulary(meta, fit_sample_ids) -> CategoryVocabulary:
    ids = _require_fit_ids(meta, fit_sample_ids)
    columns = ("Medium", *BATCH_COLUMNS)
    return CategoryVocabulary({column: tuple(sorted(meta.loc[ids, column].astype(str).unique())) for column in columns})


def fit_protein_mean(meta, labels, masks, fit_sample_ids):
    ids = _require_fit_ids(meta, fit_sample_ids)
    values = labels.loc[ids].to_numpy(np.float32)
    valid = masks.loc[ids].to_numpy(bool) & np.isfinite(values)
    return np.divide(
        np.where(valid, values, 0).sum(0), valid.sum(0),
        out=np.zeros(values.shape[1], np.float32), where=valid.sum(0) > 0,
    )


def fit_control_protein_anchor(meta, labels, masks, control_ids):
    """Fit the baseline anchor exclusively from explicit train Water/DMSO IDs."""
    ids = _require_fit_ids(meta, control_ids)
    names = meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower()
    if not names.isin(CONTROL_NAMES).all():
        raise ValueError("baseline anchor IDs must all be train Water/DMSO controls")
    values = labels.loc[ids].to_numpy(np.float32)
    valid = masks.loc[ids].to_numpy(bool) & np.isfinite(values)
    counts = valid.sum(0)
    means = np.divide(
        np.where(valid, values, 0).sum(0), counts,
        out=np.zeros(values.shape[1], np.float32), where=counts > 0,
    )
    return means, counts


def stage_a_validation_control_ids(meta):
    validation = meta["split_final"].astype(str).str.startswith("val_")
    controls = meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    return meta.index[validation & controls]


def fit_low_rank_basis(meta, labels, masks, fit_sample_ids, rank: int):
    """Optional train-only SVD basis; default V2 learns its basis by gradient."""
    ids = _require_fit_ids(meta, fit_sample_ids)
    values = labels.loc[ids].to_numpy(np.float32)
    valid = masks.loc[ids].to_numpy(bool) & np.isfinite(values)
    means = np.divide(np.where(valid, values, 0).sum(0), valid.sum(0), out=np.zeros(values.shape[1]), where=valid.sum(0) > 0)
    filled = np.where(valid, values, means)
    _, _, vh = np.linalg.svd(filled - means, full_matrices=False)
    return vh[:rank].astype(np.float32)


def build_fc_anchor(meta, fc_labels, fc_masks, fit_sample_ids, chemical_column="perturbation_no_concentration"):
    ids = _require_fit_ids(meta, fit_sample_ids)
    anchors = {}
    for chemical, group in meta.loc[ids].groupby(chemical_column):
        group_ids = group.index
        values = fc_labels.loc[group_ids].to_numpy(np.float32)
        valid = fc_masks.loc[group_ids].to_numpy(bool) & np.isfinite(values)
        anchors[str(chemical)] = np.divide(
            np.where(valid, values, 0).sum(0), valid.sum(0),
            out=np.zeros(values.shape[1], np.float32), where=valid.sum(0) > 0,
        )
    return anchors


def estimate_high_effect_weights(meta, fc_labels, fc_masks, fit_sample_ids, threshold=1.0):
    ids = _require_fit_ids(meta, fit_sample_ids)
    values = fc_labels.loc[ids].to_numpy(np.float32)
    valid = fc_masks.loc[ids].to_numpy(bool) & np.isfinite(values)
    high = valid & (np.abs(values) > float(threshold))
    return 1.0 + np.divide(high.sum(0), valid.sum(0), out=np.zeros(values.shape[1]), where=valid.sum(0) > 0)


def fit_cv_fold_statistics(meta, labels, masks, fold_train_ids, rank: int):
    """Every CV fold must pass its train IDs; there is no implicit global fit."""
    return {
        "protein_mean": fit_protein_mean(meta, labels, masks, fold_train_ids),
        "protein_basis": fit_low_rank_basis(meta, labels, masks, fold_train_ids, rank),
    }


def load_label_frames(meta, artifacts, root: Path = ROOT):
    # There is intentionally no path discovery/glob and no test-proteome path.
    path = root / "WAYB_WAYC/WAYB_WAYC_proteome_raw_train_val.csv"
    raw = pd.read_csv(path, usecols=["sample_ID", *artifacts.feature_contract.proteins]).set_index("sample_ID")
    raw = raw.loc[meta.index]
    values = raw.to_numpy(np.float32)
    valid = np.isfinite(values) & (values > 0)
    log2_values = np.zeros_like(values)
    log2_values[valid] = np.log2(values[valid])
    labels = pd.DataFrame(log2_values, index=meta.index, columns=artifacts.feature_contract.proteins)
    masks = pd.DataFrame(valid, index=meta.index, columns=artifacts.feature_contract.proteins)
    return labels, masks


@dataclass(frozen=True)
class ExperimentalFCTargets:
    treatment_ids: tuple[str, ...]
    fc_true: np.ndarray
    fc_mask: np.ndarray
    pairing_table: pd.DataFrame
    audit: dict


def build_experimental_parity_fc_targets(
    meta,
    labels,
    masks,
    treatment_ids,
    spec_path: Path | None = None,
) -> ExperimentalFCTargets:
    """Build nonofficial parity FC targets from train-only exact control matches."""
    spec_path = spec_path or ROOT / "project_v2/data_contract/control_matching_spec.json"
    spec = read_json(spec_path)
    inferred = spec["inferred_mapping_for_experimental_fc_loss"]
    if inferred["name"] != "pert_id_parity_v1" or inferred["official"] is not False:
        raise ValueError("unexpected experimental control mapping")
    exact_keys = tuple(spec["exact_match_keys"])
    required_exact_keys = (
        "data_source", "Strains", "Medium", "Temperature", "pert_time",
        "pert_time_unit", "instrument", "Yeast_cell_plate",
    )
    if exact_keys != required_exact_keys:
        raise ValueError("frozen exact control match keys changed")
    ids = _require_fit_ids(meta, treatment_ids)
    treatment_names = meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower()
    if treatment_names.isin(CONTROL_NAMES | {"quality control"}).any():
        raise ValueError("experimental FC treatment IDs must exclude controls and Quality Control")
    if meta.loc[ids, list(exact_keys)].isna().any().any():
        raise ValueError("experimental FC treatment exact-match keys must be complete")

    train = meta.loc[meta["split_final"].eq("train")]
    controls = train.loc[
        train["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    ]
    if controls[list(exact_keys)].isna().any().any():
        raise ValueError("train control exact-match keys must be complete")
    lookup = {}
    for control_id, row in controls.iterrows():
        key = tuple(row[column] for column in exact_keys) + (
            str(row["perturbation_no_concentration"]).lower(),
        )
        lookup.setdefault(key, []).append(str(control_id))

    mapping = {str(key): str(value) for key, value in inferred["mapping"].items()}
    expected_mapping = {f"#{number}": ("Water" if number % 2 else "DMSO") for number in range(1, 48)}
    if mapping != expected_mapping or inferred["quality_control"] != {"#48": "Quality Control"}:
        raise ValueError("frozen pert_id parity mapping is not the declared v1 rule")

    n_treatments, n_proteins = len(ids), labels.shape[1]
    fc_true = np.zeros((n_treatments, n_proteins), dtype=np.float32)
    fc_mask = np.zeros((n_treatments, n_proteins), dtype=bool)
    pairing_records = []
    matched_by_solvent = {"Water": 0, "DMSO": 0}
    requested_by_solvent = {"Water": 0, "DMSO": 0}
    unmatched_reasons = {}
    replicate_counts = []
    for position, treatment_id in enumerate(ids):
        row = meta.loc[treatment_id]
        pert_id = str(row["pert_id"])
        expected_control = mapping.get(pert_id)
        matched_control_ids = []
        reason = ""
        if expected_control is None:
            reason = "pert_id_not_in_frozen_parity_mapping"
        else:
            requested_by_solvent[expected_control] += 1
            key = tuple(row[column] for column in exact_keys) + (expected_control.lower(),)
            matched_control_ids = lookup.get(key, [])
            if not matched_control_ids:
                reason = "no_exact_requested_solvent_control"
        if matched_control_ids:
            matched_by_solvent[expected_control] += 1
            replicate_counts.append(len(matched_control_ids))
            control_values = labels.loc[matched_control_ids].to_numpy(np.float32, copy=True)
            control_valid = masks.loc[matched_control_ids].to_numpy(bool, copy=True) & np.isfinite(control_values)
            control_counts = control_valid.sum(axis=0)
            control_mean = np.divide(
                np.where(control_valid, control_values, 0.0).sum(axis=0), control_counts,
                out=np.zeros(n_proteins, dtype=np.float32), where=control_counts > 0,
            )
            treatment_values = labels.loc[treatment_id].to_numpy(np.float32, copy=True)
            treatment_valid = masks.loc[treatment_id].to_numpy(bool, copy=True) & np.isfinite(treatment_values)
            joint = treatment_valid & (control_counts > 0) & np.isfinite(control_mean)
            fc_true[position, joint] = treatment_values[joint] - control_mean[joint]
            fc_mask[position] = joint
            status = "matched"
        else:
            unmatched_reasons[reason] = unmatched_reasons.get(reason, 0) + 1
            status = "unmatched_huber_only"
        pairing_records.append({
            "treatment_sample_ID": str(treatment_id),
            "pert_id": pert_id,
            "expected_control": expected_control or "",
            "status": status,
            "unmatched_reason": reason,
            "matched_control_count": int(len(matched_control_ids)),
            "matched_control_sample_IDs": "|".join(matched_control_ids),
            **{f"match_{column}": row[column] for column in exact_keys},
        })
    protein_valid_counts = fc_mask.sum(axis=0)
    matched_count = int(sum(record["status"] == "matched" for record in pairing_records))
    audit = {
        "experimental_nonofficial_parity_fc": True,
        "control_mapping": "pert_id_parity_v1",
        "official_fc_result": False,
        "fit_scope": "split_final=train only",
        "train_treatment_total": int(n_treatments),
        "matched_treatment_count": matched_count,
        "matched_coverage": matched_count / max(n_treatments, 1),
        "requested_control_counts": requested_by_solvent,
        "matched_treatment_counts_by_control": matched_by_solvent,
        "unmatched_reason_counts": unmatched_reasons,
        "matched_control_replicate_count_min": int(min(replicate_counts)) if replicate_counts else 0,
        "matched_control_replicate_count_median": float(np.median(replicate_counts)) if replicate_counts else 0.0,
        "matched_control_replicate_count_max": int(max(replicate_counts)) if replicate_counts else 0,
        "per_protein_fc_valid_count_min": int(protein_valid_counts.min()),
        "per_protein_fc_valid_count_median": float(np.median(protein_valid_counts)),
        "per_protein_fc_valid_count_max": int(protein_valid_counts.max()),
        "exact_match_keys": list(exact_keys),
        "multiple_controls": "protein_wise_mean_over_observed_controls",
        "fallback_used": False,
        "control_matching_spec_path": str(spec_path.resolve()),
        "control_matching_spec_sha256": sha256_file(spec_path),
        "test_proteome_opened": False,
    }
    return ExperimentalFCTargets(
        tuple(map(str, ids)), fc_true, fc_mask, pd.DataFrame(pairing_records), audit,
    )


def _codes(series, values):
    lookup = {value: index + 1 for index, value in enumerate(values)}
    return series.astype(str).map(lookup).fillna(0).to_numpy(np.int64, copy=True)


def _mapping_codes(series, domain):
    mapping = ({"confirmed": 0, "proxy": 1, "special_control": 2, "unresolved": 3}
               if domain == "chemical" else {"exact": 0, "supported": 1, "proxy": 2, "unresolved": 3})
    return series.astype(str).map(mapping).fillna(3).to_numpy(np.int64, copy=True)


def _confidence_codes(series):
    return series.astype(str).map({"none": 0, "low": 1, "medium": 2, "high": 3}).fillna(0).to_numpy(np.int64, copy=True)


def _safe_genome_shuffle(values, masks, eligible, seed):
    output, output_mask = values.copy(), masks.copy()
    positions = np.flatnonzero(eligible)
    permutation = np.random.default_rng(seed).permutation(positions)
    output[positions], output_mask[positions] = values[permutation], masks[permutation]
    return output, output_mask


def make_chemical_feature_variant(artifacts, mode="correct", seed=20260814):
    """Call the frozen interface once and apply its entity permutation to masks."""
    mapping = artifacts.chemical_mapping
    values, order = apply_feature_variant(
        artifacts.chemical_features,
        mode=mode,
        seed=int(seed),
        mapping_status=mapping["mapping_status"],
        structure_valid=mapping["structure_valid"],
        special_control_type=mapping["special_control_type"].fillna(""),
    )
    order = np.asarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("frozen chemical variant did not return a permutation")
    masks = (
        np.zeros_like(artifacts.chemical_valid_mask, dtype=bool)
        if mode == "zero"
        else artifacts.chemical_valid_mask[order].copy()
    )
    if mode == "zero" and np.any(values):
        raise ValueError("zero chemical variant retained numeric features")
    permutation_hash = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    index = artifacts.chemical_index.reset_index(drop=True)
    table = pd.DataFrame({
        "target_row": np.arange(len(order), dtype=np.int64),
        "target_chemical_id": index["chemical_id"].astype(str),
        "target_raw_name": index["raw_name"].astype(str),
        "source_row": order,
        "source_chemical_id": index.iloc[order]["chemical_id"].astype(str).to_numpy(),
        "source_raw_name": index.iloc[order]["raw_name"].astype(str).to_numpy(),
        "is_fixed_point": order == np.arange(len(order)),
    })
    audit = {
        "mode": mode,
        "seed": int(seed),
        "n_entities": int(len(order)),
        "fixed_point_count": int(np.sum(order == np.arange(len(order)))),
        "permutation_sha256": permutation_hash,
        "interface": "scripts/build_chemical_features.py::apply_feature_variant",
    }
    return values, masks, order, table, audit


CHEMICAL_FEATURE_COMPONENTS = ("full", "morgan_only", "descriptor_only", "none")


def apply_chemical_feature_components(values, masks, morgan_dim, components="full"):
    """Apply an orthogonal numeric-component ablation without changing identity metadata."""
    values = np.asarray(values)
    masks = np.asarray(masks, dtype=bool)
    if values.ndim != 2 or values.shape != masks.shape:
        raise ValueError("chemical component values and masks must be aligned 2D arrays")
    morgan_dim = int(morgan_dim)
    if not 0 < morgan_dim < values.shape[1]:
        raise ValueError("morgan_dim must split Morgan and descriptor columns")
    if components not in CHEMICAL_FEATURE_COMPONENTS:
        raise ValueError(f"invalid chemical_feature_components: {components}")
    output = values.copy()
    output_mask = masks.copy()
    if components == "morgan_only":
        output[:, morgan_dim:] = 0
        output_mask[:, morgan_dim:] = False
    elif components == "descriptor_only":
        output[:, :morgan_dim] = 0
        output_mask[:, :morgan_dim] = False
    elif components == "none":
        output.fill(0)
        output_mask.fill(False)
    audit = {
        "chemical_feature_components": components,
        "morgan_enabled": components in {"full", "morgan_only"},
        "descriptor_enabled": components in {"full", "descriptor_only"},
        "identity_quality_flags_preserved": True,
        "protein_label_input_accepted": False,
    }
    return output, output_mask, audit


def build_batch(meta, sample_ids, artifacts, vocab, chemical_mode="correct", genome_mode="correct", seed=20260814, device="cpu", chemical_variant=None, chemical_feature_components="full"):
    rows = meta.loc[pd.Index(sample_ids)]
    chemical_index = artifacts.chemical_index
    chemical_lookup = {name: i for i, name in enumerate(chemical_index["raw_name"].astype(str))}
    chemical_rows = rows["perturbation_no_concentration"].astype(str).map(chemical_lookup)
    if chemical_rows.isna().any():
        raise ValueError("chemical absent from frozen index")
    chemical_rows = chemical_rows.to_numpy(np.int64)
    if chemical_variant is None:
        chemical_variant = make_chemical_feature_variant(artifacts, chemical_mode, seed)
    chemical_values, chemical_masks, _, _, chemical_audit = chemical_variant
    if chemical_audit["mode"] != chemical_mode or chemical_audit["seed"] != int(seed):
        raise ValueError("chemical variant does not match requested mode/seed")
    chemical_values, chemical_masks, _ = apply_chemical_feature_components(
        chemical_values, chemical_masks, artifacts.morgan_dim,
        components=chemical_feature_components,
    )

    genome_index = artifacts.genome_index
    genome_lookup = {name: i for i, name in enumerate(genome_index["strain_id"].astype(str))}
    genome_rows = rows["Strains"].astype(str).map(genome_lookup)
    if genome_rows.isna().any():
        raise ValueError("strain absent from frozen index")
    genome_rows = genome_rows.to_numpy(np.int64)
    genome_values, genome_masks = artifacts.genome_features.copy(), artifacts.genome_valid_mask.copy()
    eligible_genome = genome_index["mapping_type"].isin(["exact", "supported"]).to_numpy()
    if genome_mode == "shuffle":
        genome_values, genome_masks = _safe_genome_shuffle(genome_values, genome_masks, eligible_genome, seed + 1)
    elif genome_mode == "zero":
        genome_values, genome_masks = np.zeros_like(genome_values), np.zeros_like(genome_masks)
    elif genome_mode != "correct":
        raise ValueError("invalid genome mode")

    time_minutes = rows["pert_time"].astype(float).to_numpy(np.float32)
    if not rows["pert_time_unit"].astype(str).str.lower().eq("min").all():
        raise ValueError("only frozen minute units are accepted")
    condition = np.column_stack([
        (rows["Temperature"].astype(float).to_numpy() - 30.0) / 7.0,
        np.log1p(time_minutes) / np.log1p(1440.0),
        np.sin(2 * np.pi * time_minutes / 1440.0),
        np.cos(2 * np.pi * time_minutes / 1440.0),
    ]).astype(np.float32)
    batch_categorical = np.column_stack([_codes(rows[column], vocab.values[column]) for column in BATCH_COLUMNS])
    chemical_metadata = chemical_index.iloc[chemical_rows]
    genome_metadata = genome_index.iloc[genome_rows]
    selected_chemical = chemical_values[chemical_rows]
    return V2Batch(
        torch.as_tensor(selected_chemical[:, :artifacts.morgan_dim], device=device),
        torch.as_tensor(selected_chemical[:, artifacts.morgan_dim:], device=device),
        torch.as_tensor(chemical_masks[chemical_rows], device=device),
        torch.as_tensor(_mapping_codes(chemical_metadata["mapping_status"], "chemical"), device=device),
        torch.as_tensor(_confidence_codes(chemical_metadata["mapping_confidence"]), device=device),
        torch.as_tensor(chemical_metadata["structure_valid"].astype(bool).to_numpy(copy=True)[:, None].astype(np.float32), device=device),
        torch.as_tensor(genome_values[genome_rows], device=device),
        torch.as_tensor(genome_masks[genome_rows], device=device),
        torch.as_tensor(_mapping_codes(genome_metadata["mapping_type"], "genome"), device=device),
        torch.as_tensor(_confidence_codes(genome_metadata["mapping_confidence"]), device=device),
        torch.as_tensor(genome_metadata["proxy_flag"].to_numpy(np.float32, copy=True)[:, None], device=device),
        torch.as_tensor(_codes(rows["Medium"], vocab.values["Medium"]), device=device),
        torch.as_tensor(condition, device=device),
        torch.as_tensor(batch_categorical, device=device),
        torch.as_tensor(rows["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES).to_numpy(copy=True)[:, None], device=device),
    )


class V2Dataset(Dataset):
    def __init__(self, batch: V2Batch, labels: torch.Tensor, masks: torch.Tensor):
        if len(labels) != len(masks) or len(labels) != batch.morgan.shape[0]:
            raise ValueError("dataset arrays are not aligned")
        self.batch, self.labels, self.masks = batch, labels, masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index

    def collate(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        return self.batch.index_select(index), self.labels[index], self.masks[index]


class ExperimentalFCV2Dataset(V2Dataset):
    def __init__(self, batch, labels, masks, fc_true, fc_mask):
        super().__init__(batch, labels, masks)
        if len(fc_true) != len(labels) or fc_true.shape != fc_mask.shape:
            raise ValueError("experimental FC arrays are not aligned")
        self.fc_true, self.fc_mask = fc_true, fc_mask

    def collate(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        return (
            self.batch.index_select(index), self.labels[index], self.masks[index],
            self.fc_true[index], self.fc_mask[index],
        )


def make_loader(batch, labels, masks, batch_size, shuffle=False, seed=0, num_workers=0):
    dataset = V2Dataset(batch, labels, masks)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        collate_fn=dataset.collate, num_workers=int(num_workers),
    )


def make_experimental_fc_loader(batch, labels, masks, fc_true, fc_mask, batch_size, shuffle=False, seed=0, num_workers=0):
    dataset = ExperimentalFCV2Dataset(batch, labels, masks, fc_true, fc_mask)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        collate_fn=dataset.collate, num_workers=int(num_workers),
    )


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_macro_huber(model, validation_loaders, weights, device="cpu"):
    if set(validation_loaders) != set(VAL_SCENARIOS):
        raise ValueError("early stopping requires exactly four validation scenarios")
    scores = {}
    model.eval()
    with torch.no_grad():
        for scenario in VAL_SCENARIOS:
            numerator, count = 0.0, 0
            for batch, target, mask in validation_loaders[scenario]:
                batch = batch.to(device)
                target, mask = target.to(device), mask.to(device)
                huber_sum, valid_count = masked_huber_sum_count(
                    model(batch)["y_pred"], target, mask,
                )
                numerator += float(huber_sum)
                count += valid_count
            if count == 0:
                raise ValueError(f"validation scenario {scenario} has no valid protein positions")
            scores[scenario] = numerator / count
    return float(np.mean(list(scores.values()))), scores


def control_validation_huber(model, loader, device="cpu"):
    numerator, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            if not bool(batch.is_control.bool().all()):
                raise ValueError("stage A monitor loader contains a non-control sample")
            batch = batch.to(device)
            target, mask = target.to(device), mask.to(device)
            outputs = model(batch)
            if not torch.equal(outputs["delta_response"], torch.zeros_like(outputs["delta_response"])):
                raise ValueError("control delta_response must be exactly zero")
            huber_sum, valid_count = masked_huber_sum_count(outputs["y_pred"], target, mask)
            numerator += float(huber_sum)
            count += valid_count
    if count == 0:
        raise ValueError("stage A control validation has no valid protein positions")
    return numerator / count, {"control_only_validation_huber": numerator / count, "n_valid_positions": count}


def single_validation_huber(model, loader, device="cpu"):
    """Exact valid-position Huber for one explicitly supplied inner-CV loader."""
    numerator, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            batch = batch.to(device)
            target, mask = target.to(device), mask.to(device)
            huber_sum, valid_count = masked_huber_sum_count(
                model(batch)["y_pred"], target, mask,
            )
            numerator += float(huber_sum)
            count += valid_count
    if count == 0:
        raise ValueError("inner drug validation has no valid protein positions")
    score = numerator / count
    return score, {"inner_drug_validation_huber": score, "n_valid_positions": count}


PARAMETER_MODULES = {
    "baseline_branch": "baseline_branch",
    "genome_encoder": "genome_encoder",
    "chemical_encoder": "chemical_encoder",
    "response_branch": "response_branch",
    "batch_branch": "batch_branch",
}


def configure_stage_optimizer(model, stage_config, weight_decay):
    """Apply explicit freeze states and construct named, non-overlapping groups."""
    specifications = stage_config["parameter_groups"]
    if set(specifications) != set(PARAMETER_MODULES):
        raise ValueError("stage parameter_groups must explicitly name all five branches/encoders")
    groups = []
    for group_name, module_name in PARAMETER_MODULES.items():
        spec = specifications[group_name]
        trainable = bool(spec["trainable"])
        learning_rate = float(spec["lr"])
        if group_name == "batch_branch" and not model.cfg.batch_enabled:
            trainable = False
        if not trainable and learning_rate != 0:
            if group_name != "batch_branch" or model.cfg.batch_enabled:
                raise ValueError(f"frozen group {group_name} must have lr=0")
        parameters = list(getattr(model, module_name).parameters())
        for parameter in parameters:
            parameter.requires_grad_(trainable)
        if trainable:
            if learning_rate <= 0:
                raise ValueError(f"trainable group {group_name} must have positive lr")
            groups.append({"params": parameters, "lr": learning_rate, "group_name": group_name})
    if not groups:
        raise ValueError("stage has no trainable parameter group")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def checkpoint_payload(model, optimizer, epoch, monitor, early_stopping, config, hashes, seed, stage, run_metadata=None):
    payload = {
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "epoch": int(epoch),
        "monitor": float(monitor),
        "early_stopping": copy.deepcopy(early_stopping),
        "config": copy.deepcopy(config),
        "artifact_hashes": dict(hashes),
        "seed": int(seed),
        "stage": str(stage),
        "output_scale": "log2",
    }
    if run_metadata is not None:
        payload["run_metadata"] = copy.deepcopy(run_metadata)
    return payload


def save_checkpoint(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, expected_hashes):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["artifact_hashes"] != expected_hashes:
        raise ValueError("checkpoint artifact hashes do not match frozen inputs")
    if payload.get("output_scale") != "log2":
        raise ValueError("checkpoint is not on log2 scale")
    return payload


def restore_checkpoint(path, model, optimizer, expected_hashes):
    payload = load_checkpoint(path, expected_hashes)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


@dataclass(frozen=True)
class StageResult:
    best_epoch: int
    best_monitor: float
    stopped_epoch: int
    checkpoint_path: str
    history: tuple[dict, ...]
    stop_reason: str


@dataclass(frozen=True)
class TrainResult:
    """Backward-compatible small-test result, implemented on canonical losses."""
    best_epoch: int
    best_monitor: float
    stopped_epoch: int
    history: tuple[dict, ...]


def train_small(model, train_batch, train_y, train_mask, val_batch, val_y, val_mask, epochs=4, lr=1e-3, patience=2, weights=LossWeights()):
    """In-memory convenience adapter used only by unit tests, not the stage-2 entry."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_epoch, best_monitor, best_model, bad_epochs = -1, float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_loss = compute_losses(model(train_batch), train_y, train_mask, weights)["loss_total"]
        train_loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            monitor = float(compute_losses(model(val_batch), val_y, val_mask, weights)["loss_absolute"])
        history.append({"epoch": epoch, "train_total": float(train_loss.detach()), "val_absolute": monitor})
        if monitor < best_monitor:
            best_epoch, best_monitor = epoch, monitor
            best_model, bad_epochs = copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_model is None:
        raise RuntimeError("no best model was captured")
    model.load_state_dict(best_model)
    return TrainResult(best_epoch, best_monitor, epoch, tuple(history))


def run_training_stage(model, optimizer, train_loader, validation_loaders, weights, epochs, patience, checkpoint_path, config, hashes, seed, stage, start_epoch=0, early_state=None, device="cpu", monitor_kind="stage_b_macro", run_metadata=None):
    state = copy.deepcopy(early_state) if early_state else {"best_epoch": -1, "best_monitor": float("inf"), "bad_epochs": 0, "patience": int(patience)}
    history = []
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_components = {name: [] for name in (
            "loss_total", "loss_absolute", "loss_fc_absolute", "loss_fc",
            "loss_response_magnitude", "loss_batch_reg", "loss_batch_source_reg",
            "loss_batch_instrument_reg", "loss_batch_plate_reg",
        )}
        for training_items in train_loader:
            if len(training_items) == 3:
                batch, target, mask = training_items
                fc_true = fc_mask = None
            elif len(training_items) == 5:
                batch, target, mask, fc_true, fc_mask = training_items
                fc_true, fc_mask = fc_true.to(device), fc_mask.to(device)
            else:
                raise ValueError("unexpected training loader item structure")
            batch = batch.to(device)
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = compute_losses(
                outputs, target, mask, weights,
                fc_pred=(outputs["delta_response"] if fc_true is not None else None),
                fc_true=fc_true, fc_mask=fc_mask,
            )
            if not torch.isfinite(losses["loss_total"]):
                raise FloatingPointError(f"non-finite training loss in {stage} epoch {epoch}")
            losses["loss_total"].backward()
            optimizer.step()
            for name in train_components:
                train_components[name].append(float(losses[name].detach()))
        if monitor_kind == "stage_a_control_only":
            monitor, monitor_details = control_validation_huber(model, validation_loaders, device=device)
            monitor_name = "control_only_validation_huber"
        elif monitor_kind == "stage_b_macro":
            monitor, monitor_details = validation_macro_huber(model, validation_loaders, weights, device=device)
            monitor_name = "macro_huber_equal_four_scenarios"
        elif monitor_kind == "inner_drug_validation":
            monitor, monitor_details = single_validation_huber(
                model, validation_loaders, device=device,
            )
            monitor_name = "inner_drug_validation_huber"
        else:
            raise ValueError(f"unknown monitor kind: {monitor_kind}")
        if not np.isfinite(monitor):
            raise FloatingPointError(f"non-finite monitor in {stage} epoch {epoch}")
        if monitor < state["best_monitor"]:
            state.update({"best_epoch": epoch, "best_monitor": monitor, "bad_epochs": 0})
            payload = checkpoint_payload(model, optimizer, epoch, monitor, state, config, hashes, seed, stage, run_metadata)
            save_checkpoint(checkpoint_path, payload)
        else:
            state["bad_epochs"] += 1
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_components["loss_total"])),
            "train_loss_components": {
                name: float(np.mean(values)) for name, values in train_components.items()
            },
            "monitor_name": monitor_name,
            "monitor": monitor,
            "monitor_details": monitor_details,
            "bad_epochs": int(state["bad_epochs"]),
            "parameter_group_learning_rates": {
                group["group_name"]: float(group["lr"]) for group in optimizer.param_groups
            },
        })
        if state["bad_epochs"] >= state["patience"]:
            break
    restore_checkpoint(checkpoint_path, model, optimizer, hashes)
    stop_reason = (
        "early_stopping_patience_exhausted"
        if state["bad_epochs"] >= state["patience"]
        else "configured_max_epochs_completed"
    )
    return StageResult(
        state["best_epoch"], state["best_monitor"], epoch,
        str(checkpoint_path), tuple(history), stop_reason,
    )


def _resolve_device(configured):
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    return configured


def _loss_weights(config):
    loss = config["loss"]
    return LossWeights(
        absolute=float(loss["absolute_weight"]),
        fc_absolute=float(loss.get("fc_absolute_weight", 0.0)),
        fc=float(loss["fc_weight"]), dep=float(loss["dep_weight"]),
        pathway=float(loss["pathway_weight"]),
        batch_reg=(float(loss["batch_reg_weight"]) if config["model"]["batch_enabled"] else 0.0),
        batch_center_strength=float(loss["batch_center_strength"]),
        response_magnitude=float(loss.get("response_magnitude_weight", 0.0)),
        batch_source_reg=(float(loss.get("batch_source_reg_weight", 0.0))
                          if config["model"]["batch_enabled"] else 0.0),
        batch_instrument_reg=(float(loss.get("batch_instrument_reg_weight", 0.0))
                              if config["model"]["batch_enabled"] else 0.0),
        batch_plate_reg=(float(loss.get("batch_plate_reg_weight", 0.0))
                         if config["model"]["batch_enabled"] else 0.0),
    )


def _validate_experimental_fc_config(config):
    fc_weight = float(config["loss"]["fc_weight"])
    if fc_weight == 0:
        return False
    model = config["model"]
    mapping = config["control_mapping"]
    if config.get("task_stage") in {
        "S1_score_aligned", "S2B_hierarchical_treatment",
        "S2D_second_seed_hierarchical_treatment",
    }:
        required = (
            config.get("experimental_nonofficial_parity_fc") is True,
            config.get("official_fc_result") is False,
            config.get("planning_proxy") is True,
            config.get("official_score") is False,
            mapping.get("name") == "pert_id_parity_v1",
            mapping.get("official") is False,
            mapping.get("fit_scope") == "split_final=train_only",
            float(config["loss"].get("fc_absolute_weight", 0.0)) > 0,
            model.get("chemical_mode") == "correct",
            model.get("chemical_feature_components") == "morgan_only",
            model.get("genome_mode") == "correct",
            model.get("batch_enabled") is True,
            (config.get("task_stage") not in {"S2B_hierarchical_treatment", "S2D_second_seed_hierarchical_treatment"}
             or model.get("batch_structure") == "hierarchical_batch"),
            model.get("response_gate_enabled") is True,
            model.get("similarity_enabled") is False,
        )
        if not all(required):
            raise RuntimeError("score-aligned FC configuration violates the frozen contract")
        return True
    required = (
        config.get("experimental_nonofficial_parity_fc") is True,
        config.get("official_fc_result") is False,
        mapping.get("name") == "pert_id_parity_v1",
        mapping.get("official") is False,
        mapping.get("fit_scope") == "split_final=train_only",
        fc_weight == 0.1,
        model.get("chemical_mode") == "correct",
        model.get("chemical_feature_components") == "morgan_only",
        model.get("genome_mode") == "correct",
        model.get("batch_enabled") is False,
        model.get("similarity_enabled") is False,
    )
    if not all(required):
        raise RuntimeError("nonzero FC is allowed only for the frozen Stage 2.5 nonofficial parity diagnostic")
    if any(float(config["loss"][name]) != 0 for name in ("dep_weight", "pathway_weight")):
        raise RuntimeError("experimental FC diagnostic requires all other auxiliary losses disabled")
    return True


def _validate_s2b_config(config):
    """Enforce the frozen single-seed S2B treatment-training contract."""
    if config.get("task_stage") not in {
        "S2B_hierarchical_treatment", "S2D_second_seed_hierarchical_treatment",
    }:
        return
    model = config["model"]
    training = config["training"]
    stage_b_groups = training["stage_b"]["parameter_groups"]
    required = (
        int(training["seed"]) == (
            20260815 if config.get("task_stage") == "S2D_second_seed_hierarchical_treatment" else 20260814
        ),
        model.get("batch_enabled") is True,
        model.get("batch_structure") == "hierarchical_batch",
        model.get("chemical_mode") == "correct",
        model.get("chemical_feature_components") in {"none", "morgan_only"},
        model.get("genome_mode") == "correct",
        model.get("similarity_enabled") is False,
        bool(training.get("train_internal_batch_holdout", {}).get("enabled")) is True,
        float(training["train_internal_batch_holdout"]["fraction"]) == 0.2,
    )
    if not all(required):
        raise RuntimeError("Stage S2B configuration violates the frozen experiment scope")
    expected_trainable = {"chemical_encoder", "response_branch"}
    actual_trainable = {
        name for name, spec in stage_b_groups.items() if bool(spec["trainable"])
    }
    if actual_trainable != expected_trainable:
        raise RuntimeError("Stage S2B must freeze baseline, genome and hierarchical batch branches")
    for name in {"baseline_branch", "genome_encoder", "batch_branch"}:
        if float(stage_b_groups[name]["lr"]) != 0.0:
            raise RuntimeError(f"Stage S2B frozen group {name} must have lr=0")


def validate_real_matched_pair_prediction_identity(
    model,
    meta,
    pairing_table,
    artifacts,
    vocab,
    chemical_variant,
    chemical_mode,
    chemical_feature_components,
    genome_mode,
    seed,
    device="cpu",
):
    """Verify y_treatment-y_control equals treatment response on a real exact pair."""
    matched = pairing_table.loc[pairing_table["status"].eq("matched")]
    if matched.empty:
        raise ValueError("no real matched treatment/control pair is available")
    row = matched.iloc[0]
    control_id = str(row["matched_control_sample_IDs"]).split("|")[0]
    ids = [str(row["treatment_sample_ID"]), control_id]
    batch = build_batch(
        meta, ids, artifacts, vocab, chemical_mode=chemical_mode,
        chemical_feature_components=chemical_feature_components,
        genome_mode=genome_mode, seed=seed, chemical_variant=chemical_variant,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(batch.to(device))
    difference = outputs["y_pred"][0] - outputs["y_pred"][1]
    expected = outputs["delta_response"][0]
    max_abs_error = float((difference - expected).abs().max())
    baseline_max_abs_difference = float(
        (outputs["y_baseline"][0] - outputs["y_baseline"][1]).abs().max()
    )
    control_response_max_abs = float(outputs["delta_response"][1].abs().max())
    if not torch.allclose(difference, expected, atol=1e-5, rtol=1e-5):
        raise ValueError("predicted treatment-control difference is not delta_response")
    if control_response_max_abs != 0.0:
        raise ValueError("matched control delta_response is not exactly zero")
    return {
        "treatment_sample_ID": ids[0],
        "control_sample_ID": ids[1],
        "max_abs_identity_error": max_abs_error,
        "baseline_max_abs_difference": baseline_max_abs_difference,
        "float_tolerance_atol_rtol": 1e-5,
        "control_delta_response_max_abs": control_response_max_abs,
        "identity": "predicted_treatment_minus_predicted_control_equals_delta_response_treatment",
    }


def run_configured_training(output_dir: Path, config_path: Path, smoke=False, stage_a_checkpoint: Path | None = None):
    """Configuration-driven two-stage entry; callers decide smoke versus full data."""
    started = time.perf_counter()
    config = load_config(config_path)
    experimental_fc_enabled = _validate_experimental_fc_config(config)
    _validate_s2b_config(config)
    training = config["training"]
    s2b_hierarchical = config.get("task_stage") in {
        "S2B_hierarchical_treatment", "S2D_second_seed_hierarchical_treatment",
    }
    # A declared fixed Stage-A anchor is configuration-owned for every Stage-B
    # experiment.  Historically the flat path required the CLI flag as well,
    # which made an otherwise resolved config silently retrain Stage A.
    if stage_a_checkpoint is None:
        declared_checkpoint = training.get("fixed_stage_a_checkpoint", {}).get("path")
        if s2b_hierarchical and not declared_checkpoint:
            raise ValueError("Stage S2B requires a declared S2A Stage A checkpoint")
        if declared_checkpoint:
            stage_a_checkpoint = Path(declared_checkpoint)
    seed = int(training["seed"])
    set_seed(seed)
    device = _resolve_device(training["device"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    artifacts = load_artifact_bundle()
    meta = load_train_val_metadata()
    labels, masks = load_label_frames(meta, artifacts)
    train_ids = meta.index[meta["split_final"].eq("train")]
    fixed_s2a_payload = None
    if s2b_hierarchical:
        specification = training.get("fixed_stage_a_checkpoint")
        if not specification or stage_a_checkpoint is None:
            raise ValueError("Stage S2B requires fixed_stage_a_checkpoint metadata")
        actual_hash = sha256_file(stage_a_checkpoint)
        if actual_hash != specification["sha256"]:
            raise ValueError("fixed S2A Stage A checkpoint SHA-256 mismatch")
        fixed_s2a_payload = torch.load(
            stage_a_checkpoint, map_location="cpu", weights_only=False,
        )
        required_s2a = (
            fixed_s2a_payload.get("artifact_hashes") == artifacts.hashes,
            fixed_s2a_payload.get("stage") == "S2A_stage_a_only_final",
            fixed_s2a_payload.get("structure") == "hierarchical_batch",
            int(fixed_s2a_payload.get("seed", -1)) == seed,
            fixed_s2a_payload.get("validation_labels_used_for_training_or_selection") is False,
            fixed_s2a_payload.get("test_proteome_opened") is False,
        )
        if not all(required_s2a):
            raise ValueError("fixed checkpoint is not the frozen leakage-safe S2A hierarchical Stage A")
        raw_vocab = fixed_s2a_payload.get("vocabulary")
        if not isinstance(raw_vocab, dict):
            raise ValueError("S2A checkpoint does not contain its train-control vocabulary")
        vocab = CategoryVocabulary({
            key: tuple(values) for key, values in raw_vocab.items()
        })
    else:
        vocab = fit_category_vocabulary(meta, train_ids)
    perturbations = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    control_ids = train_ids[perturbations.isin(CONTROL_NAMES)]
    treatment_ids = train_ids[~perturbations.isin(CONTROL_NAMES | {"quality control"})]
    validation_ids = {scenario: meta.index[meta["split_final"].eq(scenario)] for scenario in VAL_SCENARIOS}
    stage_a_control_validation_ids = stage_a_validation_control_ids(meta)
    stage_a_train_ids = control_ids
    if smoke:
        smoke_config = training["smoke"]
        treatment_ids = treatment_ids[:int(smoke_config["n_treatment"])]
    protein_mean_values, protein_control_counts = fit_control_protein_anchor(
        meta, labels, masks, control_ids,
    )
    if int(protein_control_counts.min()) <= 0:
        raise ValueError("every contracted protein must be observed in train controls")
    protein_mean = torch.from_numpy(protein_mean_values)
    anchor_metadata = {
        "anchor_source": "train_controls_only",
        "control_sample_count": int(len(control_ids)),
        "min_observations_per_protein": int(protein_control_counts.min()),
    }
    model_config = config["model"]
    chemical_components = model_config.get(
        "chemical_feature_components",
        "none" if model_config["chemical_mode"] == "zero" else "full",
    )
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_config["latent_dim"]),
        protein_rank=int(model_config["protein_rank"]),
        dropout=float(model_config["dropout"]),
        batch_enabled=bool(model_config["batch_enabled"]),
        batch_structure=str(model_config.get("batch_structure", "flat_batch")),
        response_gate_enabled=bool(model_config.get("response_gate_enabled", False)),
        response_gate_initial=float(model_config.get("response_gate_initial", 0.25)),
        response_rms_cap=float(model_config.get("response_rms_cap", 0.0)),
        batch_field_dropout=float(model_config.get("batch_field_dropout", 0.0)),
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg, protein_mean).to(device)
    fixed_stage_a = None
    if stage_a_checkpoint is not None:
        specification = training.get("fixed_stage_a_checkpoint")
        if not specification:
            raise ValueError("config must declare fixed_stage_a_checkpoint metadata")
        actual_hash = sha256_file(stage_a_checkpoint)
        if actual_hash != specification["sha256"]:
            raise ValueError("fixed Stage A checkpoint SHA-256 mismatch")
        if s2b_hierarchical:
            fixed_payload = fixed_s2a_payload
            mapped_state = {
                (("batch_branch." + name[len("hierarchical_batch."):])
                 if name.startswith("hierarchical_batch.") else name): value
                for name, value in fixed_payload["model_state"].items()
            }
            incompat = model.load_state_dict(mapped_state, strict=False)
            allowed_missing = {
                name for name in model.state_dict()
                if name.startswith("chemical_encoder.") or name.startswith("response_branch.")
            }
            if set(incompat.missing_keys) != allowed_missing or incompat.unexpected_keys:
                raise ValueError(f"unexpected S2A-to-S2B compatibility keys: {incompat}")
            if not torch.equal(
                model.train_protein_mean.detach().cpu(),
                torch.as_tensor(fixed_payload["model_state"]["train_protein_mean"]),
            ):
                raise ValueError("S2A train-control protein anchor differs from the current control-only fit")
            fixed_stage_a = {
                "path": str(stage_a_checkpoint.resolve()),
                "sha256": actual_hash,
                "checkpoint_epoch": int(fixed_payload["epoch"]),
                "checkpoint_monitor": fixed_payload.get("monitor"),
                "stage": fixed_payload["stage"],
                "structure": fixed_payload["structure"],
                "vocabulary_source": "S2A_fold_train_controls_only",
            }
        else:
            fixed_payload = load_checkpoint(stage_a_checkpoint, artifacts.hashes)
            if fixed_payload.get("stage") != "control_first_A":
                raise ValueError("fixed checkpoint is not a Stage A checkpoint")
            if int(fixed_payload["seed"]) != seed:
                raise ValueError("fixed Stage A checkpoint seed mismatch")
            if bool(fixed_payload["config"]["model"]["batch_enabled"]) != bool(model_config["batch_enabled"]):
                raise ValueError("fixed Stage A checkpoint batch mode mismatch")
            incompat = model.load_state_dict(
                fixed_payload["model_state"],
                strict=not bool(model_config.get("response_gate_enabled", False)),
            )
            if model_config.get("response_gate_enabled", False):
                allowed_missing = {"response_branch.gate.weight", "response_branch.gate.bias"}
                if set(incompat.missing_keys) != allowed_missing or incompat.unexpected_keys:
                    raise ValueError(f"unexpected Stage A compatibility keys: {incompat}")
            fixed_stage_a = {
                "path": str(stage_a_checkpoint.resolve()),
                "sha256": actual_hash,
                "checkpoint_epoch": int(fixed_payload["epoch"]),
                "checkpoint_monitor": float(fixed_payload["monitor"]),
            }
    stage_b_initial_component_hash = module_parameters_sha256(
        model, ("chemical_encoder", "response_branch"),
    )
    initial_chemical_parameters = snapshot_module_parameters(model, "chemical_encoder")
    frozen_hashes_before = None
    if s2b_hierarchical:
        frozen_hashes_before = {
            "baseline_branch": named_parameter_prefixes_sha256(model, ("baseline_branch",)),
            "genome_encoder": named_parameter_prefixes_sha256(model, ("genome_encoder",)),
            "batch_source": named_parameter_prefixes_sha256(model, ("batch_branch.source",)),
            "batch_instrument": named_parameter_prefixes_sha256(model, ("batch_branch.instrument",)),
            "batch_plate": named_parameter_prefixes_sha256(model, ("batch_branch.plate",)),
            "batch_all": named_parameter_prefixes_sha256(model, ("batch_branch",)),
        }
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    chemical_variant = make_chemical_feature_variant(
        artifacts, model_config["chemical_mode"], seed,
    )
    _, _, chemical_component_audit = apply_chemical_feature_components(
        chemical_variant[0], chemical_variant[1], artifacts.morgan_dim,
        components=chemical_components,
    )
    def loader_for(ids, shuffle=False):
        batch = build_batch(
            meta, ids, artifacts, vocab,
            chemical_mode=model_config["chemical_mode"],
            genome_mode=model_config["genome_mode"], seed=seed,
            chemical_variant=chemical_variant,
            chemical_feature_components=chemical_components,
        )
        target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
        mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
        return make_loader(
            batch, target, mask, batch_size=batch_size, shuffle=shuffle,
            seed=seed, num_workers=num_workers,
        )
    full_treatment_ids = treatment_ids.copy()
    holdout_ids = pd.Index([], dtype=object)
    holdout_spec = training.get("train_internal_batch_holdout")
    if holdout_spec and bool(holdout_spec.get("enabled", False)) and not smoke:
        fraction = float(holdout_spec["fraction"])
        if not 0.0 < fraction < 0.5:
            raise ValueError("train internal batch holdout fraction must be in (0, 0.5)")
        treatment_frame = meta.loc[treatment_ids, list(BATCH_COLUMNS)].astype(str)
        group_keys = treatment_frame.agg("|".join, axis=1)
        unique_groups = sorted(group_keys.unique())
        holdout_selection_seed = int(holdout_spec.get("selection_seed", seed))
        scored = sorted(
            unique_groups,
            key=lambda value: hashlib.sha256(f"{holdout_selection_seed}|{value}".encode("utf-8")).hexdigest(),
        )
        n_holdout = max(1, int(round(len(scored) * fraction)))
        selected = set(scored[:n_holdout])
        holdout_ids = treatment_ids[group_keys.isin(selected).to_numpy()]
        treatment_ids = treatment_ids[~group_keys.isin(selected).to_numpy()]
        if len(treatment_ids) == 0 or len(holdout_ids) == 0:
            raise ValueError("batch-group holdout produced an empty partition")
        train_groups = set(group_keys.loc[treatment_ids])
        holdout_groups = set(group_keys.loc[holdout_ids])
        if train_groups & holdout_groups:
            raise ValueError("technical batch group crossed train/holdout boundary")
    experimental_fc_targets = None
    if experimental_fc_enabled:
        experimental_fc_targets = build_experimental_parity_fc_targets(
            meta, labels, masks, treatment_ids,
            ROOT / "project_v2/data_contract/control_matching_spec.json",
        )
    def experimental_fc_loader_for(ids, shuffle=False):
        if experimental_fc_targets is None or tuple(map(str, ids)) != experimental_fc_targets.treatment_ids:
            raise ValueError("experimental FC loader must use the exact audited treatment order")
        batch = build_batch(
            meta, ids, artifacts, vocab,
            chemical_mode=model_config["chemical_mode"],
            genome_mode=model_config["genome_mode"], seed=seed,
            chemical_variant=chemical_variant,
            chemical_feature_components=chemical_components,
        )
        target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
        mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
        return make_experimental_fc_loader(
            batch, target, mask,
            torch.from_numpy(experimental_fc_targets.fc_true),
            torch.from_numpy(experimental_fc_targets.fc_mask),
            batch_size=batch_size, shuffle=shuffle, seed=seed, num_workers=num_workers,
        )
    val_loaders = {name: loader_for(ids) for name, ids in validation_ids.items()}
    stage_a_control_loader = loader_for(stage_a_control_validation_ids)
    weights = _loss_weights(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    chemical_variant[3].to_csv(output_dir / "chemical_entity_permutation.csv", index=False)
    (output_dir / "chemical_variant_manifest.json").write_text(
        json.dumps(chemical_variant[4], indent=2), encoding="utf-8",
    )
    (output_dir / "chemical_component_manifest.json").write_text(
        json.dumps(chemical_component_audit, indent=2), encoding="utf-8",
    )
    if experimental_fc_targets is not None:
        experimental_fc_targets.pairing_table.to_csv(
            output_dir / "experimental_parity_fc_pairing.csv", index=False,
        )
        (output_dir / "experimental_parity_fc_pairing_audit.json").write_text(
            json.dumps(experimental_fc_targets.audit, indent=2), encoding="utf-8",
        )
    resolved_config_path = output_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    stage_a_config = training["stage_a"]
    stage_b_config = training["stage_b"]
    stage_a_epochs = int(training["smoke"]["stage_a_epochs"] if smoke else stage_a_config["epochs"])
    stage_b_epochs = int(training["smoke"]["stage_b_epochs"] if smoke else stage_b_config["epochs"])
    run_metadata = dict(anchor_metadata)
    run_metadata["stage_b_initial_chemical_response_sha256"] = stage_b_initial_component_hash
    freeze_probe_batch = None
    frozen_batch_outputs_before = None
    if s2b_hierarchical:
        probe_ids = treatment_ids[:min(16, len(treatment_ids))]
        if len(probe_ids) == 0:
            raise ValueError("Stage S2B freeze probe has no treatment samples")
        freeze_probe_batch = build_batch(
            meta, probe_ids, artifacts, vocab,
            chemical_mode=model_config["chemical_mode"],
            genome_mode=model_config["genome_mode"], seed=seed,
            chemical_variant=chemical_variant,
            chemical_feature_components=chemical_components,
        ).to(device)
        model.eval()
        with torch.no_grad():
            probe_outputs = model(freeze_probe_batch)
        frozen_batch_outputs_before = {
            name: probe_outputs[name].detach().cpu().clone()
            for name in ("delta_batch", "delta_source", "delta_instrument", "delta_plate")
        }
        run_metadata["stage_b_frozen_parameter_hashes_before"] = frozen_hashes_before
    pair_identity_audit = None
    if experimental_fc_targets is not None:
        run_metadata["experimental_parity_fc_pairing_audit"] = experimental_fc_targets.audit
        pair_identity_audit = validate_real_matched_pair_prediction_identity(
            model, meta, experimental_fc_targets.pairing_table, artifacts, vocab,
            chemical_variant, model_config["chemical_mode"], chemical_components,
            model_config["genome_mode"], seed, device=device,
        )
        run_metadata["predicted_fc_identity_audit"] = pair_identity_audit
    if fixed_stage_a is None:
        optimizer_a = configure_stage_optimizer(model, stage_a_config, training["weight_decay"])
        stage_a = run_training_stage(
            model, optimizer_a, loader_for(stage_a_train_ids, True), stage_a_control_loader, weights,
            stage_a_epochs, int(stage_a_config["patience"]), output_dir / "stage_a_best.pt",
            config, artifacts.hashes, seed, "control_first_A", device=device,
            monitor_kind="stage_a_control_only", run_metadata=run_metadata,
        )
    else:
        run_metadata["fixed_stage_a_checkpoint"] = fixed_stage_a
        stage_a = StageResult(
            fixed_stage_a["checkpoint_epoch"], fixed_stage_a["checkpoint_monitor"],
            fixed_stage_a["checkpoint_epoch"], fixed_stage_a["path"], tuple(),
            "loaded_fixed_stage_a_checkpoint_no_retraining",
        )
    optimizer_b = configure_stage_optimizer(model, stage_b_config, training["weight_decay"])
    optimizer_parameter_audit = None
    if s2b_hierarchical:
        optimizer_group_names = {str(group["group_name"]) for group in optimizer_b.param_groups}
        if optimizer_group_names != {"chemical_encoder", "response_branch"}:
            raise ValueError("Stage S2B optimizer must contain only chemical_encoder and response_branch")
        allowed_parameter_ids = {
            id(parameter)
            for module_name in ("chemical_encoder", "response_branch")
            for parameter in getattr(model, module_name).parameters()
        }
        optimizer_parameter_ids = {
            id(parameter) for group in optimizer_b.param_groups for parameter in group["params"]
        }
        frozen_parameter_ids = {
            id(parameter)
            for module_name in ("baseline_branch", "genome_encoder", "batch_branch")
            for parameter in getattr(model, module_name).parameters()
        }
        if optimizer_parameter_ids != allowed_parameter_ids or optimizer_parameter_ids & frozen_parameter_ids:
            raise ValueError("Stage S2B optimizer contains a frozen or missing trainable parameter")
        optimizer_parameter_audit = {
            "group_names": sorted(optimizer_group_names),
            "optimizer_parameter_count": int(len(optimizer_parameter_ids)),
            "frozen_parameter_overlap_count": 0,
            "only_chemical_encoder_and_response_branch": True,
        }
    stage_b_train_loader = (
        experimental_fc_loader_for(treatment_ids, True)
        if experimental_fc_enabled else loader_for(treatment_ids, True)
    )
    stage_b = run_training_stage(
        model, optimizer_b, stage_b_train_loader, val_loaders, weights,
        stage_b_epochs, int(stage_b_config["patience"]), output_dir / "stage_b_best.pt",
        config, artifacts.hashes, seed, "treatment_huber_B", device=device,
        monitor_kind="stage_b_macro", run_metadata=run_metadata,
    )
    stage_b_freeze_audit = None
    if s2b_hierarchical:
        frozen_hashes_after = {
            "baseline_branch": named_parameter_prefixes_sha256(model, ("baseline_branch",)),
            "genome_encoder": named_parameter_prefixes_sha256(model, ("genome_encoder",)),
            "batch_source": named_parameter_prefixes_sha256(model, ("batch_branch.source",)),
            "batch_instrument": named_parameter_prefixes_sha256(model, ("batch_branch.instrument",)),
            "batch_plate": named_parameter_prefixes_sha256(model, ("batch_branch.plate",)),
            "batch_all": named_parameter_prefixes_sha256(model, ("batch_branch",)),
        }
        model.eval()
        with torch.no_grad():
            probe_outputs_after = model(freeze_probe_batch)
        exact_output_equal = {
            name: bool(torch.equal(
                frozen_batch_outputs_before[name], probe_outputs_after[name].detach().cpu(),
            ))
            for name in frozen_batch_outputs_before
        }
        hash_equal = {
            name: frozen_hashes_before[name] == frozen_hashes_after[name]
            for name in frozen_hashes_before
        }
        if not all(hash_equal.values()) or not all(exact_output_equal.values()):
            raise RuntimeError("Stage B modified a frozen baseline or hierarchical batch component")
        stage_b_freeze_audit = {
            "parameter_hashes_before": frozen_hashes_before,
            "parameter_hashes_after": frozen_hashes_after,
            "parameter_hashes_exactly_equal": hash_equal,
            "probe_outputs_bitwise_equal": exact_output_equal,
            "optimizer": optimizer_parameter_audit,
            "status": "PASS",
        }
        freeze_audit_path = output_dir / "stage_b_freeze_audit.json"
        freeze_audit_path.write_text(
            json.dumps(stage_b_freeze_audit, indent=2), encoding="utf-8",
        )
        best_payload = torch.load(stage_b.checkpoint_path, map_location="cpu", weights_only=False)
        best_payload["stage_b_freeze_audit"] = stage_b_freeze_audit
        best_payload["batch_structure"] = "hierarchical_batch"
        torch.save(best_payload, stage_b.checkpoint_path)
    final_macro_huber, final_scenario_huber = validation_macro_huber(
        model, val_loaders, weights, device=device,
    )
    scenario_metrics = evaluate_four_scenarios(model, val_loaders, device=device)
    component_norms = evaluate_component_norms(model, val_loaders, device=device)
    chemical_parameter_change = parameter_change_l2(
        model, "chemical_encoder", initial_chemical_parameters,
    )
    diagnostics = evaluate_batch_diagnostics(
        model, loader_for(full_treatment_ids), device=device,
    )
    histories = {"stage_a": list(stage_a.history), "stage_b": list(stage_b.history)}
    (output_dir / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8",
    )
    elapsed_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0
    summary = {
        "status": "PASS",
        "mode": "smoke" if smoke else "formal",
        "config": str(config_path),
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "resolved_config_path": str(resolved_config_path.resolve()),
        "stage_a": {
            "best_epoch": stage_a.best_epoch, "best_monitor": stage_a.best_monitor,
            "actual_epochs": len(stage_a.history), "stopped_epoch": stage_a.stopped_epoch,
            "stop_reason": stage_a.stop_reason,
            "monitor_name": (
                "fixed_epochs_all_train_controls_no_validation_selection"
                if s2b_hierarchical else "control_only_validation_huber"
            ),
            "train_control_sample_count": int(len(stage_a_train_ids)),
            "validation_control_sample_count": int(len(stage_a_control_validation_ids)),
            "validation_control_scenarios": {
                name: int(meta.loc[stage_a_control_validation_ids, "split_final"].eq(name).sum())
                for name in VAL_SCENARIOS
                if int(meta.loc[stage_a_control_validation_ids, "split_final"].eq(name).sum()) > 0
            },
            "fixed_checkpoint": fixed_stage_a,
        },
        "stage_b": {
            "best_epoch": stage_b.best_epoch, "best_monitor": stage_b.best_monitor,
            "actual_epochs": len(stage_b.history), "stopped_epoch": stage_b.stopped_epoch,
            "stop_reason": stage_b.stop_reason,
            "monitor_name": "macro_huber_equal_four_scenarios",
        },
        "validation_macro": "equal_weight_mean_of_exact_valid_position_scenario_huber",
        "final_validation_macro_huber": final_macro_huber,
        "validation_scenario_huber": final_scenario_huber,
        "validation_scenarios": scenario_metrics,
        "validation_component_norms": component_norms,
        "stage_b_initial_chemical_response_sha256": stage_b_initial_component_hash,
        "chemical_encoder_parameter_change_l2": chemical_parameter_change,
        "chemical_variant": chemical_variant[4],
        "chemical_permutation_file": "chemical_entity_permutation.csv",
        "chemical_feature_components": chemical_components,
        "chemical_component_manifest_file": "chemical_component_manifest.json",
        **anchor_metadata,
        "n_control": len(control_ids), "n_treatment": len(treatment_ids),
        "n_treatment_full": len(full_treatment_ids),
        "train_internal_batch_holdout": {
            "enabled": bool(len(holdout_ids)),
            "train_sample_count": int(len(treatment_ids)),
            "holdout_sample_count": int(len(holdout_ids)),
            "holdout_sample_ids_sha256": hashlib.sha256(
                json.dumps(list(map(str, holdout_ids)), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "group_columns": list(BATCH_COLUMNS),
            "labels_used_for_training_or_early_stopping": False,
            "selection_seed": int(
                training.get("train_internal_batch_holdout", {}).get("selection_seed", seed)
            ),
        },
        "stage_b_training_ids_sha256": hashlib.sha256(
            json.dumps(list(map(str, treatment_ids)), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "stage_b_full_treatment_ids_sha256": hashlib.sha256(
            json.dumps(list(map(str, full_treatment_ids)), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "stage_b_dataloader_shuffle_seed": seed,
        "n_validation": {name: len(ids) for name, ids in validation_ids.items()},
        "n_proteins": artifacts.feature_contract.n_proteins,
        "batch_enabled": bool(model_config["batch_enabled"]),
        "batch_structure": str(model_config.get("batch_structure", "flat_batch")),
        "effective_batch_reg_weight": float(weights.batch_reg),
        "batch_source_reg_weight": float(weights.batch_source_reg),
        "batch_instrument_reg_weight": float(weights.batch_instrument_reg),
        "batch_plate_reg_weight": float(weights.batch_plate_reg),
        "fc_loss_enabled": bool(experimental_fc_enabled), "similarity_enabled": False,
        "experimental_nonofficial_parity_fc": bool(experimental_fc_enabled),
        "control_mapping": "pert_id_parity_v1" if experimental_fc_enabled else "disabled_for_fc",
        "official_fc_result": False,
        "experimental_parity": "RUN_NONOFFICIAL_DIAGNOSTIC" if experimental_fc_enabled else "NOT RUN",
        "fc_weight": float(weights.fc),
        "fc_absolute_weight": float(weights.fc_absolute),
        "response_magnitude_weight": float(weights.response_magnitude),
        "response_gate_enabled": bool(model_config.get("response_gate_enabled", False)),
        "response_rms_cap": float(model_config.get("response_rms_cap", 0.0)),
        "batch_field_dropout": float(model_config.get("batch_field_dropout", 0.0)),
        "planning_proxy": bool(config.get("planning_proxy", False)),
        "official_score": False,
        "fc_early_stopping_used": False,
        "pairing_audit": experimental_fc_targets.audit if experimental_fc_targets is not None else None,
        "pairing_file": "experimental_parity_fc_pairing.csv" if experimental_fc_targets is not None else None,
        "pairing_audit_file": "experimental_parity_fc_pairing_audit.json" if experimental_fc_targets is not None else None,
        "predicted_fc_identity_audit": pair_identity_audit,
        "stage_b_fc_loss_by_epoch": [
            {"epoch": item["epoch"], "loss_fc": item["train_loss_components"]["loss_fc"]}
            for item in stage_b.history
        ],
        "batch_diagnostic_scope": "all_selected_train_treatments",
        "batch_diagnostic_sample_count": int(len(full_treatment_ids)),
        "training_history_file": "training_history.json",
        "stage_b_checkpoint": str(Path(stage_b.checkpoint_path).resolve()),
        "stage_b_checkpoint_sha256": sha256_file(Path(stage_b.checkpoint_path)),
        "stage_b_freeze_audit_file": "stage_b_freeze_audit.json" if stage_b_freeze_audit is not None else None,
        "stage_b_freeze_audit": stage_b_freeze_audit,
        "test_proteome_opened": False, **diagnostics,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_two_stage_smoke(output_dir: Path, config_path: Path, seed=None):
    """Compatibility adapter for the prior smoke command."""
    if seed is not None and seed != int(load_config(config_path)["training"]["seed"]):
        raise ValueError("seed is configuration-owned in the formal entry")
    return run_configured_training(output_dir, config_path, smoke=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the contract-anchored V2 model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-a-checkpoint", type=Path)
    args = parser.parse_args(argv)
    summary = run_configured_training(
        args.output_dir, args.config, smoke=args.smoke,
        stage_a_checkpoint=args.stage_a_checkpoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
