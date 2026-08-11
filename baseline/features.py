"""Condition feature engineering for AIVC stage 2-3.

The public workflow is:

    encoders = fit_feature_encoders(train_meta, train_y_log2, train_mask)
    X_all = build_condition_features(all_meta, encoders=encoders)

All statistical representations and the final projection are fitted on the
training rows only.  Unseen categorical values use an all-zero one-hot vector,
train-derived statistical fallbacks, or deterministic hash features rather
than being silently mapped to the first known category.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from aivc.config import EXPERIMENT_CONFIG, get_experiment_config
from aivc.entity_representations import (
    ChemicalAnchorEncoder,
    CrossFeatureEncoder,
    HashEncoder,
    StrainPriorEncoder,
)


CAT_COLS = {
    "strains": "Strains",
    "chemicals": "perturbation_no_concentration",
    "media": "Medium",
    "instruments": "instrument",
}


class FixedDimProjector:
    """Train-only standardized linear projection to a fixed embedding width."""

    def __init__(self, output_dim: int = 256):
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(output_dim)
        self.input_dim_ = None
        self.mean_ = None
        self.scale_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        self.input_dim_ = X.shape[1]
        self.mean_ = np.nan_to_num(X.mean(axis=0), nan=0.0)
        self.scale_ = np.nan_to_num(X.std(axis=0), nan=1.0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Z = np.nan_to_num((X - self.mean_) / self.scale_, nan=0.0)

        if self.input_dim_ <= self.output_dim:
            self.components_ = np.eye(self.input_dim_, dtype=np.float64)
        else:
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
            self.components_ = vt[: self.output_dim]
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("FixedDimProjector has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Z = np.nan_to_num((X - self.mean_) / self.scale_, nan=0.0)
        projected = Z @ self.components_.T
        output = np.zeros((X.shape[0], self.output_dim), dtype=np.float32)
        width = min(projected.shape[1], self.output_dim)
        if width:
            output[:, :width] = projected[:, :width].astype(np.float32)
        return output


def _require_columns(meta_df, columns):
    missing = [column for column in columns if column not in meta_df.columns]
    if missing:
        raise KeyError(f"Missing metadata columns: {missing}")


def _one_hot(values: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    """One-hot encode known values; unseen values remain all-zero rows."""
    output = np.zeros((len(values), len(mapping)), dtype=np.float32)
    for row, value in enumerate(values.astype(str)):
        idx = mapping.get(value)
        if idx is not None:
            output[row, idx] = 1.0
    return output


def _categorical_features(meta_df, encoders, slices):
    chunks = []
    for name, column in CAT_COLS.items():
        if not encoders["config"]["encoder"].get("use_categorical_features", True):
            continue
        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = _one_hot(meta_df[column], encoders[name])
        chunks.append(chunk)
        slices[name] = (start, start + chunk.shape[1])
    return chunks


def build_raw_condition_features(meta_df, encoders) -> np.ndarray:
    """Build the unprojected feature matrix using fitted train-only encoders."""
    if "config" not in encoders:
        raise ValueError("encoders is not a fitted feature encoder dictionary")

    cfg = encoders["config"]["encoder"]
    _require_columns(meta_df, list(CAT_COLS.values()) + ["Temperature", "pert_time"])
    if cfg.get("use_hash_features", True):
        _require_columns(meta_df, ["Yeast_cell_plate"])

    chunks = []
    slices: Dict[str, tuple] = {}

    for name, column in CAT_COLS.items():
        if cfg.get("use_categorical_features", True):
            start = sum(chunk.shape[1] for chunk in chunks)
            chunk = _one_hot(meta_df[column], encoders[name])
            chunks.append(chunk)
            slices[name] = (start, start + chunk.shape[1])

    if cfg.get("use_strain_prior", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = encoders["strain_prior"].transform(meta_df)
        chunks.append(chunk)
        slices["strain_prior"] = (start, start + chunk.shape[1])

    if cfg.get("use_chem_anchor", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = encoders["chem_anchor"].transform(meta_df)
        chunks.append(chunk)
        slices["chem_anchor"] = (start, start + chunk.shape[1])

    if cfg.get("use_hash_features", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = encoders["hash_chemical"].transform(
            meta_df["perturbation_no_concentration"].astype(str)
        )
        chunks.append(chunk)
        slices["hash_chemical"] = (start, start + chunk.shape[1])

        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = encoders["hash_plate"].transform(meta_df["Yeast_cell_plate"].astype(str))
        chunks.append(chunk)
        slices["hash_plate"] = (start, start + chunk.shape[1])

    if cfg.get("use_cross_features", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        chunk = encoders["cross_features"].transform(meta_df)
        chunks.append(chunk)
        slices["cross_features"] = (start, start + chunk.shape[1])

    if cfg.get("use_temperature", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        temperature = pd.to_numeric(meta_df["Temperature"], errors="coerce").fillna(0)
        chunk = (temperature.to_numpy() == 37).astype(np.float32).reshape(-1, 1)
        chunks.append(chunk)
        slices["temperature"] = (start, start + chunk.shape[1])

    if cfg.get("use_time_features", True):
        start = sum(chunk.shape[1] for chunk in chunks)
        period = float(cfg.get("time_period_minutes", 240.0))
        if period <= 0:
            raise ValueError("time_period_minutes must be positive")
        time = pd.to_numeric(meta_df["pert_time"], errors="coerce").fillna(0).to_numpy()
        angle = 2.0 * np.pi * time / period
        chunk = np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)
        chunks.append(chunk)
        slices["time"] = (start, start + chunk.shape[1])

    if not chunks:
        raise ValueError("At least one feature group must be enabled")
    raw = np.concatenate(chunks, axis=1).astype(np.float32)
    encoders["feature_slices"] = slices
    encoders["raw_dim"] = int(raw.shape[1])
    return raw


def fit_feature_encoders(
    train_meta: pd.DataFrame,
    y_log2: pd.DataFrame,
    mask_matrix: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
):
    """Fit every input-side representation using training rows only."""
    if y_log2 is None:
        raise ValueError("y_log2 is required to fit statistical entity representations")
    config = get_experiment_config(config)
    _require_columns(train_meta, list(CAT_COLS.values()) + ["Temperature", "pert_time", "Yeast_cell_plate"])
    if "split_final" in train_meta and not train_meta["split_final"].astype(str).eq("train").all():
        raise ValueError("fit_feature_encoders must receive train-only metadata")

    y_train = y_log2.reindex(train_meta.index)
    if y_train.isna().all(axis=None):
        raise ValueError("No target rows overlap train_meta")

    encoders = {
        "config": config,
        "cat_encoders": {},
    }
    for name, column in CAT_COLS.items():
        values = train_meta[column].dropna().astype(str).unique()
        mapping = {value: idx for idx, value in enumerate(sorted(values))}
        encoders[name] = mapping
        encoders["cat_encoders"][column] = mapping

    encoder_cfg = config["encoder"]
    if encoder_cfg.get("use_strain_prior", True):
        encoders["strain_prior"] = StrainPriorEncoder(
            encoder_cfg.get("strain_prior_pca_dim", 32)
        ).fit(train_meta, y_train)
    if encoder_cfg.get("use_chem_anchor", True):
        encoders["chem_anchor"] = ChemicalAnchorEncoder(
            encoder_cfg.get("chem_anchor_pca_dim", 64)
        ).fit(train_meta, y_train)
    if encoder_cfg.get("use_hash_features", True):
        encoders["hash_chemical"] = HashEncoder(
            encoder_cfg.get("hash_dim_chemical", 32), seed=11
        )
        encoders["hash_plate"] = HashEncoder(
            encoder_cfg.get("hash_dim_plate", 16), seed=17
        )
    if encoder_cfg.get("use_cross_features", True):
        encoders["cross_features"] = CrossFeatureEncoder(
            encoder_cfg.get("cross_dim_strain_medium", 10),
            encoder_cfg.get("cross_dim_chemical_temperature", 92),
        ).fit(train_meta)

    raw_train = build_raw_condition_features(train_meta, encoders)
    projector = FixedDimProjector(encoder_cfg.get("d_emb", 256)).fit(raw_train)
    encoders["projector"] = projector
    encoders["d_emb"] = projector.output_dim
    encoders["raw_dim"] = raw_train.shape[1]
    return encoders


def build_condition_features(
    meta_df: pd.DataFrame,
    fit_encoders: bool = False,
    encoders: Optional[dict] = None,
    y_log2: Optional[pd.DataFrame] = None,
    mask_matrix: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
):
    """Transform metadata to fixed ``(N, d_emb)`` condition features.

    For a new experiment, call with ``fit_encoders=True`` and provide
    ``y_log2``; the returned encoder dictionary is then reused for every
    validation/test row.  Existing callers can pass a fitted ``encoders``
    dictionary directly.
    """
    if fit_encoders or encoders is None:
        encoders = fit_feature_encoders(meta_df, y_log2, mask_matrix, config)
        return build_condition_features(meta_df, encoders=encoders), encoders
    raw = build_raw_condition_features(meta_df, encoders)
    return encoders["projector"].transform(raw)


# Backward-compatible name used by downstream code.
DEFAULT_FEATURE_CONFIG = EXPERIMENT_CONFIG
