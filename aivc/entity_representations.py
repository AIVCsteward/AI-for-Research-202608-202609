"""Train-only entity representations used by the AIVC condition encoder.

This module intentionally depends only on NumPy and pandas.  It implements the
closed-data-benchmark version of the planned representations:

* ``StrainPriorEncoder``: train-set strain protein means -> padded PCA vectors.
* ``ChemicalAnchorEncoder``: train-set treatment-minus-control means -> PCA.
* ``HashEncoder``: deterministic hashed one-hot features for unseen entities.
* ``CrossFeatureEncoder``: deterministic hashed interaction features.

All fit methods require training metadata and training labels.  The caller is
responsible for passing only ``split_final == 'train'`` rows.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


MATCH_KEYS = [
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
]
CONTROL_NAMES = {"water", "dmso"}


def _as_bool_mask(mask, index: pd.Index, name: str = "mask") -> pd.Series:
    if mask is None:
        return pd.Series(True, index=index)
    if isinstance(mask, pd.Series):
        result = mask.reindex(index)
    else:
        result = pd.Series(mask, index=index)
    if result.isna().any():
        raise ValueError(f"{name} is not aligned with metadata index")
    return result.astype(bool)


def _select_train(meta_df, y_log2, train_mask=None):
    """Align metadata/targets and apply an optional train mask."""
    if not isinstance(meta_df, pd.DataFrame) or not isinstance(y_log2, pd.DataFrame):
        raise TypeError("meta_df and y_log2 must be pandas DataFrame objects")
    common = meta_df.index.intersection(y_log2.index)
    if len(common) == 0:
        raise ValueError("meta_df and y_log2 have no common sample_ID index")
    meta = meta_df.loc[common]
    y = y_log2.loc[common]
    keep = _as_bool_mask(train_mask, common, "train_mask")
    return meta.loc[keep], y.loc[keep]


def _fill_matrix(values, fallback):
    values = np.asarray(values, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a 2D matrix")
    if fallback.shape != (values.shape[1],):
        raise ValueError("fallback dimension does not match values")
    return np.where(np.isfinite(values), values, fallback[None, :])


class _PaddedPCA:
    """Small NumPy-only PCA implementation with a fixed output width."""

    def __init__(self, n_components: int):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None

    def fit(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("PCA input must be a non-empty 2D matrix")
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        if matrix.shape[0] == 1 or np.allclose(centered, 0.0):
            components = np.zeros((0, matrix.shape[1]), dtype=np.float64)
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            rank = min(self.n_components, vt.shape[0])
            components = vt[:rank]
        self.components_ = components
        return self

    def transform(self, matrix):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA encoder has not been fitted")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        centered = matrix - self.mean_
        if self.components_.shape[0]:
            projected = centered @ self.components_.T
        else:
            projected = np.zeros((matrix.shape[0], 0), dtype=np.float64)
        output = np.zeros((matrix.shape[0], self.n_components), dtype=np.float32)
        width = min(projected.shape[1], self.n_components)
        if width:
            output[:, :width] = projected[:, :width].astype(np.float32)
        return output


class StrainPriorEncoder:
    """Encode each strain by its train-only mean protein profile."""

    def __init__(self, n_components: int = 32, column: str = "Strains"):
        self.n_components = int(n_components)
        self.column = column
        self.lookup_: Dict[str, np.ndarray] = {}
        self.fallback_: Optional[np.ndarray] = None
        self.pca_ = _PaddedPCA(self.n_components)
        self.n_proteins_ = None

    def fit(self, train_meta, y_log2, train_mask=None):
        meta, y = _select_train(train_meta, y_log2, train_mask)
        if self.column not in meta:
            raise KeyError(f"Missing metadata column: {self.column}")
        self.n_proteins_ = y.shape[1]
        fallback = y.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        fallback = np.nan_to_num(fallback, nan=0.0)
        self.fallback_ = fallback

        groups = sorted(meta[self.column].dropna().astype(str).unique())
        vectors = []
        for value in groups:
            group_y = y.loc[meta[self.column].astype(str) == value]
            vector = group_y.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
            vector = np.where(np.isfinite(vector), vector, fallback)
            vectors.append(vector)
        if not vectors:
            vectors = [fallback]
            groups = ["__fallback__"]
        matrix = np.vstack(vectors)
        self.pca_.fit(matrix)
        for value, vector in zip(groups, matrix):
            self.lookup_[value] = vector
        return self

    def transform(self, meta_df) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("StrainPriorEncoder has not been fitted")
        values = meta_df[self.column].astype(str)
        matrix = np.vstack([self.lookup_.get(value, self.fallback_) for value in values])
        return self.pca_.transform(matrix)


class ChemicalAnchorEncoder:
    """Encode chemicals by train-only treatment-minus-matched-control profiles."""

    def __init__(
        self,
        n_components: int = 64,
        chemical_column: str = "perturbation_no_concentration",
        match_keys: Sequence[str] = MATCH_KEYS,
    ):
        self.n_components = int(n_components)
        self.chemical_column = chemical_column
        self.match_keys = list(match_keys)
        self.lookup_: Dict[str, np.ndarray] = {}
        self.fallback_: Optional[np.ndarray] = None
        self.pca_ = _PaddedPCA(self.n_components)
        self.n_proteins_ = None
        self.matched_samples_ = 0

    @staticmethod
    def _is_control(values: pd.Series) -> pd.Series:
        return values.astype(str).str.strip().str.lower().isin(CONTROL_NAMES)

    @staticmethod
    def _key(row, keys: Iterable[str]):
        return tuple(row[key] for key in keys)

    def fit(self, train_meta, y_log2, train_mask=None):
        meta, y = _select_train(train_meta, y_log2, train_mask)
        if self.chemical_column not in meta:
            raise KeyError(f"Missing metadata column: {self.chemical_column}")
        missing_keys = [key for key in self.match_keys if key not in meta]
        if missing_keys:
            raise KeyError(f"Missing matched-control columns: {missing_keys}")
        self.n_proteins_ = y.shape[1]
        self.lookup_ = {}
        self.matched_samples_ = 0

        # Factorize the full matched-control key once.  The previous row-wise
        # implementation repeatedly asked pandas to average thousands of
        # protein columns for every treatment sample, which made a single fit
        # take about ten minutes on the competition data.  Computing each
        # control profile once preserves the same means while reducing the fit
        # to a few vectorized passes over the matrix.
        key_index = pd.MultiIndex.from_frame(meta[self.match_keys])
        group_ids, _ = pd.factorize(key_index, sort=False)
        is_control = self._is_control(meta[self.chemical_column]).to_numpy(dtype=bool)
        y_values = y.to_numpy(dtype=np.float64, copy=False)

        control_means: Dict[int, np.ndarray] = {}
        for group_id in np.unique(group_ids[is_control]):
            if group_id < 0:
                continue
            rows = is_control & (group_ids == group_id)
            block = y_values[rows]
            valid_count = np.isfinite(block).sum(axis=0)
            control_means[int(group_id)] = np.divide(
                np.nansum(block, axis=0),
                valid_count,
                out=np.full(self.n_proteins_, np.nan, dtype=np.float64),
                where=valid_count > 0,
            )

        global_sum = np.zeros(self.n_proteins_, dtype=np.float64)
        global_count = np.zeros(self.n_proteins_, dtype=np.int64)
        chemical_sums: Dict[str, np.ndarray] = {}
        chemical_counts: Dict[str, np.ndarray] = {}
        chemical_values = meta[self.chemical_column].astype(str).to_numpy()

        for row_pos in np.flatnonzero(~is_control):
            control = control_means.get(int(group_ids[row_pos]))
            if control is None:
                continue
            delta = y_values[row_pos] - control
            valid = np.isfinite(delta)
            if not valid.any():
                continue

            chemical = chemical_values[row_pos]
            if chemical not in chemical_sums:
                chemical_sums[chemical] = np.zeros(self.n_proteins_, dtype=np.float64)
                chemical_counts[chemical] = np.zeros(self.n_proteins_, dtype=np.int64)
            global_sum[valid] += delta[valid]
            global_count[valid] += 1
            chemical_sums[chemical][valid] += delta[valid]
            chemical_counts[chemical][valid] += 1
            self.matched_samples_ += 1

        global_delta = np.divide(
            global_sum,
            global_count,
            out=np.zeros(self.n_proteins_, dtype=np.float64),
            where=global_count > 0,
        )
        global_delta = np.nan_to_num(global_delta, nan=0.0)
        self.fallback_ = global_delta

        chemicals = sorted(meta[self.chemical_column].dropna().astype(str).unique())
        vectors = []
        for chemical in chemicals:
            if chemical.lower() in CONTROL_NAMES:
                vector = np.zeros(self.n_proteins_, dtype=np.float64)
            elif chemical in chemical_sums:
                vector = np.divide(
                    chemical_sums[chemical],
                    chemical_counts[chemical],
                    out=global_delta.copy(),
                    where=chemical_counts[chemical] > 0,
                )
            else:
                vector = global_delta
            vector = np.nan_to_num(vector, nan=0.0)
            vectors.append(vector)
        if not vectors:
            chemicals = ["__fallback__"]
            vectors = [global_delta]
        matrix = np.vstack(vectors)
        self.pca_.fit(matrix)
        for chemical, vector in zip(chemicals, matrix):
            self.lookup_[chemical] = vector
        return self

    def transform(self, meta_df) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("ChemicalAnchorEncoder has not been fitted")
        values = meta_df[self.chemical_column].astype(str)
        matrix = np.vstack([self.lookup_.get(value, self.fallback_) for value in values])
        return self.pca_.transform(matrix)


class HashEncoder:
    """Deterministic hashed one-hot encoder that works for unseen values."""

    def __init__(self, hash_dim: int, seed: int = 0):
        if hash_dim <= 0:
            raise ValueError("hash_dim must be positive")
        self.hash_dim = int(hash_dim)
        self.seed = int(seed)

    def _bucket(self, value) -> int:
        token = f"{self.seed}|{value}".encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.hash_dim

    def transform(self, values: Iterable) -> np.ndarray:
        values = list(values)
        output = np.zeros((len(values), self.hash_dim), dtype=np.float32)
        for row, value in enumerate(values):
            output[row, self._bucket(value)] = 1.0
        return output


class CrossFeatureEncoder:
    """Hashed interaction features for strain/medium and chemical/temperature."""

    def __init__(self, strain_medium_dim: int = 10, chemical_temperature_dim: int = 92):
        self.strain_medium = HashEncoder(strain_medium_dim, seed=101)
        self.chemical_temperature = HashEncoder(chemical_temperature_dim, seed=202)

    def fit(self, train_meta):
        # Hash features do not need a vocabulary.  Keeping fit makes the object
        # compatible with the other train-only encoders and future extensions.
        required = ["Strains", "Medium", "perturbation_no_concentration", "Temperature"]
        missing = [column for column in required if column not in train_meta]
        if missing:
            raise KeyError(f"Missing cross-feature columns: {missing}")
        return self

    def transform(self, meta_df) -> np.ndarray:
        strain_medium = (
            meta_df["Strains"].astype(str) + "|" + meta_df["Medium"].astype(str)
        )
        chemical_temperature = (
            meta_df["perturbation_no_concentration"].astype(str)
            + "|"
            + meta_df["Temperature"].astype(str)
        )
        return np.concatenate(
            [
                self.strain_medium.transform(strain_medium),
                self.chemical_temperature.transform(chemical_temperature),
            ],
            axis=1,
        )

