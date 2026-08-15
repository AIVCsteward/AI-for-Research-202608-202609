"""Independent, fail-closed audit helpers for the GOAI WAYB/WAYC data contract.

This module deliberately contains no model code.  Label-derived statistics require
an explicit ``fit_sample_ids`` argument and reject every non-train ID.  Test
proteome labels are never loaded; only their ``sample_ID`` column may be inspected
for alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRAIN_SPLIT = "train"
MISSING_THRESHOLD = 0.80
CONTROL_NAMES = ("Water", "DMSO")
QUALITY_CONTROL_NAME = "Quality Control"
MATCH_KEYS = (
    "data_source",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
    "instrument",
    "Yeast_cell_plate",
)
EXPECTED_SPLITS = (
    "train",
    "val_chem_only",
    "val_strain_only",
    "val_both",
    "val_time",
    "test_chem_only",
    "test_strain_only",
    "test_both",
    "test_time",
)


class ContractError(ValueError):
    """Raised when an input violates the frozen data contract."""


class UnconfirmedControlMappingError(ContractError):
    """Raised when official FC is requested without a confirmed solvent map."""


@dataclass(frozen=True)
class DataPaths:
    metadata_train_val: Path
    proteome_train_val: Path
    metadata_test: Path
    proteome_test: Path


def default_paths(data_dir: Path) -> DataPaths:
    return DataPaths(
        metadata_train_val=data_dir / "WAYB_WAYC_metadata_train_val(1).csv",
        proteome_train_val=data_dir / "WAYB_WAYC_proteome_raw_train_val.csv",
        metadata_test=data_dir / "WAYB_WAYC_metadata_test(1).csv",
        proteome_test=data_dir / "WAYB_WAYC_proteome_raw_test.csv",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_unique_non_null_ids(frame: pd.DataFrame, label: str) -> pd.Index:
    if "sample_ID" not in frame.columns:
        raise ContractError(f"{label} has no sample_ID column")
    if frame["sample_ID"].isna().any():
        raise ContractError(f"{label} contains null sample_ID")
    ids = pd.Index(frame["sample_ID"].astype(str))
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].unique().tolist()[:10]
        raise ContractError(f"{label} contains duplicate sample_ID: {duplicates}")
    return ids


def align_by_sample_id(
    metadata: pd.DataFrame, proteome: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate ID sets and reorder proteome rows to metadata order."""
    metadata_ids = _require_unique_non_null_ids(metadata, "metadata")
    proteome_ids = _require_unique_non_null_ids(proteome, "proteome")
    metadata_only = metadata_ids.difference(proteome_ids)
    proteome_only = proteome_ids.difference(metadata_ids)
    if len(metadata_only) or len(proteome_only):
        raise ContractError(
            "sample_ID sets differ: "
            f"metadata_only={metadata_only[:10].tolist()}, "
            f"proteome_only={proteome_only[:10].tolist()}"
        )
    aligned_metadata = metadata.copy()
    aligned_metadata["sample_ID"] = metadata_ids
    aligned_metadata = aligned_metadata.set_index("sample_ID", drop=False)
    aligned_proteome = proteome.copy()
    aligned_proteome["sample_ID"] = proteome_ids
    aligned_proteome = aligned_proteome.set_index("sample_ID", drop=False).loc[metadata_ids]
    return aligned_metadata, aligned_proteome


def read_proteome_sample_ids(path: Path) -> pd.DataFrame:
    """Read no labels: this is the only permitted test-proteome access."""
    return pd.read_csv(path, usecols=["sample_ID"])


def read_fit_proteome_rows(
    path: Path,
    metadata: pd.DataFrame,
    fit_sample_ids: Iterable[str],
) -> pd.DataFrame:
    """Parse protein cells only for explicit train IDs from a mixed CSV.

    A sample_ID-only pass determines physical CSV row numbers.  Non-fit lines
    are then rejected by ``skiprows`` before their protein cells are tokenized.
    """
    fit_ids = require_train_fit_ids(metadata, fit_sample_ids)
    source_ids = read_proteome_sample_ids(path)["sample_ID"].astype(str)
    if source_ids.duplicated().any():
        raise ContractError("proteome sample_ID must be unique")
    missing = fit_ids.difference(source_ids)
    if len(missing):
        raise ContractError(f"proteome missing fit IDs: {missing[:10].tolist()}")
    fit_set = set(fit_ids)
    skipped_non_fit_lines = frozenset(
        index + 1 for index, sample_id in enumerate(source_ids) if sample_id not in fit_set
    )
    proteome = pd.read_csv(
        path,
        low_memory=False,
        skiprows=lambda line_number: line_number in skipped_non_fit_lines,
    )
    proteome["sample_ID"] = proteome["sample_ID"].astype(str)
    loaded_ids = pd.Index(proteome["sample_ID"])
    if set(loaded_ids) != fit_set or len(loaded_ids) != len(fit_ids):
        raise ContractError("train-only proteome load returned an unexpected ID set")
    return proteome.set_index("sample_ID", drop=False).loc[fit_ids].reset_index(drop=True)


def require_train_fit_ids(
    metadata: pd.DataFrame, fit_sample_ids: Iterable[str]
) -> pd.Index:
    """Return unique fit IDs, rejecting missing or non-train samples."""
    if "sample_ID" not in metadata.columns or "split_final" not in metadata.columns:
        raise ContractError("metadata must contain sample_ID and split_final")
    metadata_by_id = metadata.assign(sample_ID=metadata["sample_ID"].astype(str)).set_index(
        "sample_ID", drop=False
    )
    if not metadata_by_id.index.is_unique:
        raise ContractError("metadata sample_ID must be unique")
    fit_ids = pd.Index([str(value) for value in fit_sample_ids])
    if fit_ids.empty:
        raise ContractError("fit_sample_ids must not be empty")
    if fit_ids.duplicated().any():
        raise ContractError("fit_sample_ids must be unique")
    missing = fit_ids.difference(metadata_by_id.index)
    if len(missing):
        raise ContractError(f"unknown fit_sample_ids: {missing[:10].tolist()}")
    non_train = metadata_by_id.loc[fit_ids, "split_final"].ne(TRAIN_SPLIT)
    if non_train.any():
        bad = metadata_by_id.loc[fit_ids[non_train], ["sample_ID", "split_final"]]
        raise ContractError(
            "label-derived statistics may use split_final=train only: "
            f"{bad.to_dict(orient='records')[:10]}"
        )
    return fit_ids


def raw_valid_mask(values: np.ndarray) -> np.ndarray:
    """A raw intensity is observed iff it is finite and strictly positive."""
    array = np.asarray(values, dtype=np.float64)
    return np.isfinite(array) & (array > 0)


def log2_with_mask(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform only legal intensities; invalid entries remain NaN and masked."""
    array = np.asarray(values, dtype=np.float64)
    mask = raw_valid_mask(array)
    transformed = np.full(array.shape, np.nan, dtype=np.float64)
    np.log2(array, out=transformed, where=mask)
    return transformed, mask


def select_proteins(
    metadata: pd.DataFrame,
    proteome: pd.DataFrame,
    fit_sample_ids: Iterable[str],
    missing_threshold: float = MISSING_THRESHOLD,
) -> tuple[list[str], pd.Series]:
    """Select columns with invalid-rate strictly below threshold on train IDs."""
    if not 0 <= missing_threshold <= 1:
        raise ContractError("missing_threshold must be in [0, 1]")
    metadata_aligned, proteome_aligned = align_by_sample_id(metadata, proteome)
    fit_ids = require_train_fit_ids(metadata_aligned, fit_sample_ids)
    proteins = [column for column in proteome.columns if column != "sample_ID"]
    if not proteins:
        raise ContractError("proteome has no protein columns")
    values = proteome_aligned.loc[fit_ids, proteins].to_numpy(dtype=np.float64)
    missing_rate = pd.Series(
        (~raw_valid_mask(values)).mean(axis=0), index=proteins, name="missing_rate"
    )
    kept = missing_rate.index[missing_rate < missing_threshold].tolist()
    return kept, missing_rate


def fit_masked_mean(
    metadata: pd.DataFrame,
    values: pd.DataFrame,
    fit_sample_ids: Iterable[str],
) -> pd.Series:
    """Example reusable statistic fitter with the same train-only guard."""
    fit_ids = require_train_fit_ids(metadata, fit_sample_ids)
    if not values.index.is_unique:
        raise ContractError("values index must be unique sample_ID")
    missing = fit_ids.difference(values.index.astype(str))
    if len(missing):
        raise ContractError(f"values missing fit IDs: {missing[:10].tolist()}")
    subset = values.loc[fit_ids].to_numpy(dtype=np.float64)
    valid = np.isfinite(subset)
    counts = valid.sum(axis=0)
    sums = np.where(valid, subset, 0.0).sum(axis=0)
    means = np.divide(
        sums, counts, out=np.full(values.shape[1], np.nan), where=counts > 0
    )
    return pd.Series(means, index=values.columns)


def _normalise_control_mapping(
    control_mapping: Mapping[str, str] | None,
) -> dict[str, str]:
    mapping = {} if control_mapping is None else dict(control_mapping)
    invalid = {key: value for key, value in mapping.items() if value not in CONTROL_NAMES}
    if invalid:
        raise ContractError(f"control mapping values must be Water or DMSO: {invalid}")
    return {str(key): value for key, value in mapping.items()}


def build_matched_control_pairs(
    metadata: pd.DataFrame,
    treatment_sample_ids: Iterable[str],
    allowed_control_sample_ids: Iterable[str],
    control_mapping: Mapping[str, str] | None,
    mapping_key: str = "pert_id",
    match_keys: Sequence[str] = MATCH_KEYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match exact, solvent-correct controls or return structured failures.

    The function is fail-closed: no mapping means no official pair.  It never
    falls back to the other solvent, a global mean, a looser context, or a
    validation/test label outside ``allowed_control_sample_ids``.
    """
    mapping = _normalise_control_mapping(control_mapping)
    required = {"sample_ID", "perturbation_no_concentration", mapping_key, *match_keys}
    missing_columns = sorted(required.difference(metadata.columns))
    if missing_columns:
        raise ContractError(f"metadata missing control columns: {missing_columns}")
    indexed = metadata.assign(sample_ID=metadata["sample_ID"].astype(str)).set_index(
        "sample_ID", drop=False
    )
    if not indexed.index.is_unique:
        raise ContractError("metadata sample_ID must be unique")
    treatment_ids = pd.Index([str(value) for value in treatment_sample_ids])
    control_ids = pd.Index([str(value) for value in allowed_control_sample_ids])
    for label, ids in (("treatment", treatment_ids), ("control", control_ids)):
        unknown = ids.difference(indexed.index)
        if len(unknown):
            raise ContractError(f"unknown {label} IDs: {unknown[:10].tolist()}")

    allowed_controls = indexed.loc[control_ids]
    allowed_controls = allowed_controls[
        allowed_controls["perturbation_no_concentration"].isin(CONTROL_NAMES)
    ]
    lookup: dict[tuple[object, ...], list[str]] = {}
    for sample_id, row in allowed_controls.iterrows():
        key = (row["perturbation_no_concentration"],) + tuple(
            row[column] for column in match_keys
        )
        lookup.setdefault(key, []).append(sample_id)

    pairs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for treatment_id, row in indexed.loc[treatment_ids].iterrows():
        treatment_name = row["perturbation_no_concentration"]
        if treatment_name in CONTROL_NAMES:
            failures.append(
                {"treatment_sample_ID": treatment_id, "reason": "sample_is_control"}
            )
            continue
        if treatment_name == QUALITY_CONTROL_NAME:
            failures.append(
                {"treatment_sample_ID": treatment_id, "reason": "quality_control_excluded"}
            )
            continue
        mapping_value = str(row[mapping_key])
        expected_control = mapping.get(mapping_value)
        if expected_control is None:
            failures.append(
                {
                    "treatment_sample_ID": treatment_id,
                    "reason": "unconfirmed_control_mapping",
                    "mapping_key": mapping_value,
                }
            )
            continue
        key = (expected_control,) + tuple(row[column] for column in match_keys)
        matched_ids = tuple(lookup.get(key, ()))
        if not matched_ids:
            failures.append(
                {
                    "treatment_sample_ID": treatment_id,
                    "reason": "no_exact_expected_control",
                    "expected_control": expected_control,
                }
            )
            continue
        pairs.append(
            {
                "treatment_sample_ID": treatment_id,
                "control_sample_ids": matched_ids,
                "expected_control": expected_control,
                "n_controls": len(matched_ids),
            }
        )
    return pd.DataFrame(pairs), pd.DataFrame(failures)


def compute_fc_arrays(
    y_treatment_true: np.ndarray,
    y_treatment_pred: np.ndarray,
    y_control_observed: np.ndarray,
    treatment_obs_mask: np.ndarray,
    control_obs_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return official FC tensors using the strict common-valid mask.

    ``fc_true = y_treatment_true - y_control_observed``
    ``fc_pred_official = y_treatment_pred - y_control_observed``
    """
    arrays = [
        np.asarray(value)
        for value in (
            y_treatment_true,
            y_treatment_pred,
            y_control_observed,
            treatment_obs_mask,
            control_obs_mask,
        )
    ]
    if len({value.shape for value in arrays}) != 1:
        raise ContractError("FC arrays and masks must have identical shapes")
    y_true, y_pred, y_control = [value.astype(np.float64) for value in arrays[:3]]
    treatment_mask, control_mask = [value.astype(bool) for value in arrays[3:]]
    fc_mask = (
        treatment_mask
        & control_mask
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(y_control)
    )
    fc_true = np.full(y_true.shape, np.nan, dtype=np.float64)
    fc_pred = np.full(y_true.shape, np.nan, dtype=np.float64)
    np.subtract(y_true, y_control, out=fc_true, where=fc_mask)
    np.subtract(y_pred, y_control, out=fc_pred, where=fc_mask)
    return {"fc_true": fc_true, "fc_pred_official": fc_pred, "fc_mask": fc_mask}


def aggregate_observed_controls(
    control_values: np.ndarray, control_obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average control replicates protein-wise over jointly finite observations."""
    values = np.asarray(control_values, dtype=np.float64)
    mask = np.asarray(control_obs_mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ContractError("control values and mask must share shape [controls, proteins]")
    valid = mask & np.isfinite(values)
    counts = valid.sum(axis=0)
    sums = np.where(valid, values, 0.0).sum(axis=0)
    means = np.divide(
        sums, counts, out=np.full(values.shape[1], np.nan), where=counts > 0
    )
    return means, counts > 0


def _metric_vectors(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ContractError("metric arrays and mask must have identical shapes")
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise ContractError("metric mask contains no jointly finite values")
    return y_true[valid], y_pred[valid]


def masked_pearson(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    truth, pred = _metric_vectors(y_true, y_pred, mask)
    truth_centered = truth - truth.mean()
    pred_centered = pred - pred.mean()
    denominator = np.sqrt(np.square(truth_centered).sum() * np.square(pred_centered).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.dot(truth_centered, pred_centered) / denominator)


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    truth, pred = _metric_vectors(y_true, y_pred, mask)
    denominator = np.square(truth - truth.mean()).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(truth - pred).sum() / denominator)


def _parity_candidate_mapping(metadata: pd.DataFrame) -> dict[str, str]:
    """Observed plate-layout hypothesis; not an official mapping."""
    values = metadata["pert_id"].dropna().astype(str).unique()
    mapping: dict[str, str] = {}
    for value in values:
        try:
            number = int(value.removeprefix("#"))
        except ValueError:
            continue
        if 1 <= number <= 47:
            mapping[value] = "Water" if number % 2 else "DMSO"
    return mapping


def _control_coverage(
    metadata: pd.DataFrame, control_mapping: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SPLITS:
        split_ids = metadata.loc[metadata["split_final"].eq(split), "sample_ID"].astype(str)
        pairs, failures = build_matched_control_pairs(
            metadata,
            treatment_sample_ids=split_ids,
            allowed_control_sample_ids=metadata["sample_ID"].astype(str),
            control_mapping=control_mapping,
        )
        reasons = failures["reason"].value_counts().to_dict() if len(failures) else {}
        output[split] = {
            "all_samples": int(len(split_ids)),
            "matched_treatments": int(len(pairs)),
            **{str(key): int(value) for key, value in reasons.items()},
        }
    return output


def audit(paths: DataPaths) -> tuple[dict[str, object], dict[str, object]]:
    """Audit real files without loading test protein labels."""
    metadata_train_val = pd.read_csv(paths.metadata_train_val)
    metadata_test = pd.read_csv(paths.metadata_test)
    proteome_train_val_ids = read_proteome_sample_ids(paths.proteome_train_val)
    test_ids = read_proteome_sample_ids(paths.proteome_test)
    aligned_metadata, _ = align_by_sample_id(
        metadata_train_val, proteome_train_val_ids
    )
    align_by_sample_id(metadata_test, test_ids)

    fit_ids = aligned_metadata.loc[
        aligned_metadata["split_final"].eq(TRAIN_SPLIT), "sample_ID"
    ].tolist()
    proteome_train = read_fit_proteome_rows(
        paths.proteome_train_val, metadata_train_val, fit_ids
    )
    metadata_train = metadata_train_val.loc[
        metadata_train_val["split_final"].eq(TRAIN_SPLIT)
    ]
    proteins, missing_rate = select_proteins(
        metadata_train, proteome_train, fit_ids, MISSING_THRESHOLD
    )
    _, aligned_train_proteome = align_by_sample_id(metadata_train, proteome_train)
    raw_train = aligned_train_proteome.loc[
        fit_ids, aligned_train_proteome.columns != "sample_ID"
    ].to_numpy(
        dtype=np.float64
    )
    metadata_all = pd.concat([metadata_train_val, metadata_test], ignore_index=True)
    parity_mapping = _parity_candidate_mapping(metadata_all)
    files = {}
    for field, path in paths.__dict__.items():
        files[field] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
        }
        if field == "proteome_test":
            files[field]["sha256"] = None
            files[field]["sha256_status"] = (
                "deliberately_not_computed_to_avoid_reading_test_truth_bytes"
            )
        else:
            files[field]["sha256"] = sha256_file(path)

    result: dict[str, object] = {
        "files": files,
        "split_counts": {
            str(key): int(value)
            for key, value in metadata_all["split_final"].value_counts().sort_index().items()
        },
        "alignment": {
            "train_val_rows": int(len(metadata_train_val)),
            "test_rows": int(len(metadata_test)),
            "sample_id_sets_equal": True,
            "source_row_order_equal": bool(
                metadata_train_val["sample_ID"].astype(str).reset_index(drop=True).equals(
                    proteome_train_val_ids["sample_ID"].astype(str).reset_index(drop=True)
                )
            ),
            "validation_protein_cells_materialized": False,
        },
        "feature_audit": {
            "raw_proteins": int(len(missing_rate)),
            "fit_split": TRAIN_SPLIT,
            "fit_samples": int(len(fit_ids)),
            "missing_threshold": MISSING_THRESHOLD,
            "keep_condition": "invalid_rate < 0.80",
            "kept_proteins": int(len(proteins)),
            "removed_proteins": int(len(missing_rate) - len(proteins)),
            "nan_values_train": int(np.isnan(raw_train).sum()),
            "zero_values_train": int((raw_train == 0).sum()),
            "negative_values_train": int((raw_train < 0).sum()),
            "positive_inf_values_train": int(np.isposinf(raw_train).sum()),
            "negative_inf_values_train": int(np.isneginf(raw_train).sum()),
            "exactly_80_percent_invalid_columns": int(
                np.isclose((~raw_valid_mask(raw_train)).mean(axis=0), MISSING_THRESHOLD).sum()
            ),
        },
        "control_audit": {
            "official_mapping_status": "blocked_pending_organizer_confirmation",
            "official_mapping": {},
            "parity_candidate_mapping": parity_mapping,
            "parity_candidate_coverage": _control_coverage(metadata_all, parity_mapping),
            "no_mapping_fail_closed_coverage": _control_coverage(metadata_all, {}),
        },
        "test_label_access": "sample_ID column only; no test protein values loaded",
    }
    feature_contract: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "frozen_for_current_real_files",
        "generated_by": "audit_data_contract.py",
        "fit_split": TRAIN_SPLIT,
        "fit_sample_count": len(fit_ids),
        "raw_intensity_validity": "finite and strictly greater than zero",
        "invalid_value_policy": "masked; current files contain no finite non-positive or infinite train values",
        "transform": "log2 on valid raw intensities only",
        "missing_threshold": MISSING_THRESHOLD,
        "keep_condition": "train_invalid_rate < 0.80",
        "n_source_proteins": len(missing_rate),
        "n_proteins": len(proteins),
        "proteins": proteins,
        "protein_order_sha256": sha256_lines(proteins),
        "source_files": {
            key: value
            for key, value in files.items()
            if key in {"metadata_train_val", "proteome_train_val"}
        },
        "leakage_guard": {
            "label_derived_statistics": "explicit fit_sample_ids required; every ID must have split_final=train",
            "validation_labels_used": False,
            "test_labels_used": False,
        },
    }
    canonical_inputs = json.dumps(feature_contract, ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    feature_contract["generation_sha256"] = hashlib.sha256(canonical_inputs).hexdigest()
    return result, feature_contract


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result, feature_contract = audit(default_paths(args.data_dir))
    if args.output_dir is not None:
        write_json(args.output_dir / "feature_contract.json", feature_contract)
        # This evidence file is intentionally optional and not part of model inputs.
        write_json(args.output_dir / "audit_results.json", result)
    if args.print_json or args.output_dir is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
