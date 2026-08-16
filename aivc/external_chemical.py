"""External chemical structure features (SMILES -> fingerprints/descriptors).

This module turns the PubChem SMILES recorded in
``data/external/chemical_mapping.csv`` into fixed-length numeric features with
RDKit:

* Morgan fingerprint (circular, radius 2, 2048 bits) -> PCA to ``morgan_pca_dim``.
* MACCS structural keys (166 bits, kept as-is).
* Molecular descriptors (12 scalars, kept raw; the downstream
  ``FixedDimProjector`` standardizes the whole feature matrix).

Only treatment compounds receive their own structure features.  Solvent
controls (Water / DMSO) and the QC samples (pert_id=48) are encoded as an
all-zero vector, so the structural block represents "which drug is applied".
This aligns with the residual decomposition: the drug structure is the
treatment-specific signal, while controls carry none.

Leakage discipline: the Morgan PCA is fitted on *training compounds only*.
Unseen test compounds are projected using their own SMILES (never mapped to a
training fallback), which is exactly the open-data extrapolation this feature
exists to enable.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, rdFingerprintGenerator

MAPPING_PATH = "data/external/chemical_mapping.csv"
CHEMICAL_COLUMN = "perturbation_no_concentration"

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumHeteroatoms",
    "FractionCsp3",
    "RingCount",
    "NumSaturatedRings",
    "HeavyAtomCount",
]


def _bitvec_to_array(bv) -> np.ndarray:
    """Convert an RDKit bit vector to a dense float32 0/1 array."""
    arr = np.zeros((bv.GetNumBits(),), dtype=np.float32)
    for idx in bv.GetOnBits():
        arr[idx] = 1.0
    return arr


def _descriptors(mol) -> np.ndarray:
    """Return the descriptor vector for a molecule, NaN-safe."""
    values = []
    for name in DESCRIPTOR_NAMES:
        try:
            value = getattr(Descriptors, name)(mol)
        except Exception:
            value = np.nan
        values.append(float(value) if value is not None else np.nan)
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)


class _PaddedPCA:
    """Minimal NumPy PCA that pads/truncates to a fixed output width."""

    def __init__(self, n_components: int):
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None

    def fit(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        if matrix.shape[0] <= 1 or np.allclose(centered, 0.0):
            self.components_ = np.zeros((0, matrix.shape[1]), dtype=np.float64)
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            rank = min(self.n_components, vt.shape[0])
            self.components_ = vt[:rank]
        return self

    def transform(self, matrix):
        if self.mean_ is None:
            raise RuntimeError("PCA has not been fitted")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        centered = matrix - self.mean_
        projected = centered @ self.components_.T if self.components_.shape[0] else \
            np.zeros((matrix.shape[0], 0), dtype=np.float64)
        output = np.zeros((matrix.shape[0], self.n_components), dtype=np.float32)
        width = min(projected.shape[1], self.n_components)
        if width:
            output[:, :width] = projected[:, :width].astype(np.float32)
        return output


class ChemicalStructureEncoder:
    """Encode each sample's chemical by its PubChem structure (or zero for controls)."""

    def __init__(
        self,
        mapping_path: str = MAPPING_PATH,
        chemical_column: str = CHEMICAL_COLUMN,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        morgan_pca_dim: int = 64,
    ):
        self.mapping_path = mapping_path
        self.chemical_column = chemical_column
        self.morgan_bits = int(morgan_bits)
        self.morgan_radius = int(morgan_radius)
        self.morgan_pca_dim = int(morgan_pca_dim)
        self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius, fpSize=self.morgan_bits
        )
        self.mapping_ = self._load_mapping()
        self._cache: Dict[str, np.ndarray] = {}
        self._morgan_pca = _PaddedPCA(self.morgan_pca_dim)
        self.n_features_ = None

    def _load_mapping(self) -> Dict[str, dict]:
        df = pd.read_csv(self.mapping_path, dtype=str)
        return {
            row["raw_name"]: row
            for _, row in df.iterrows()
        }

    def _structure_features(self, raw_name: str) -> Optional[np.ndarray]:
        """Return the concatenated structural feature vector for one chemical.

        ``None`` (not zero) for controls/QC, which the caller turns into the
        all-zero vector; ``None`` also signals an unmapped name.
        """
        entry = self.mapping_.get(raw_name)
        if entry is None:
            return None
        if entry.get("entity_type") != "compound":
            return None
        if raw_name in self._cache:
            return self._cache[raw_name]

        smiles = entry.get("isomeric_smiles") or entry.get("canonical_smiles") or ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # A SMILES that RDKit cannot parse yields a zero block rather than
            # a crash; all current entries parse, so this is defensive only.
            self._cache[raw_name] = np.zeros(
                (self.morgan_bits + 167 + len(DESCRIPTOR_NAMES),), dtype=np.float32
            )
            return self._cache[raw_name]

        morgan = _bitvec_to_array(self._morgan_gen.GetFingerprint(mol))
        macs = _bitvec_to_array(MACCSkeys.GenMACCSKeys(mol))
        descr = _descriptors(mol)
        features = np.concatenate([morgan, macs, descr]).astype(np.float32)
        self._cache[raw_name] = features
        return features

    def fit(self, train_meta: pd.DataFrame) -> "ChemicalStructureEncoder":
        """Fit the Morgan PCA on the *unique training compounds* only."""
        names = train_meta[self.chemical_column].astype(str).unique()
        morgan_vectors = []
        for name in names:
            feats = self._structure_features(name)
            if feats is None:
                continue
            # The Morgan block is the leading ``morgan_bits`` columns.
            morgan_vectors.append(feats[: self.morgan_bits])
        if morgan_vectors:
            self._morgan_pca.fit(np.vstack(morgan_vectors))
        else:
            # Degenerate fallback: identity-ish PCA (all-zero projection).
            self._morgan_pca.mean_ = np.zeros(self.morgan_bits, dtype=np.float64)
            self._morgan_pca.components_ = np.zeros((0, self.morgan_bits), dtype=np.float64)
        self.n_features_ = self.morgan_pca_dim + 167 + len(DESCRIPTOR_NAMES)
        return self

    def transform(self, meta_df: pd.DataFrame) -> np.ndarray:
        """Return the ``(N, n_features_)`` chemical-structure matrix."""
        if self.n_features_ is None:
            raise RuntimeError("ChemicalStructureEncoder has not been fitted")
        rows = []
        for name in meta_df[self.chemical_column].astype(str):
            feats = self._structure_features(name)
            if feats is None:
                rows.append(np.zeros(self.n_features_, dtype=np.float32))
                continue
            morgan_pca = self._morgan_pca.transform(feats[: self.morgan_bits])[0]
            rest = feats[self.morgan_bits:]
            rows.append(np.concatenate([morgan_pca, rest]).astype(np.float32))
        return np.vstack(rows)
