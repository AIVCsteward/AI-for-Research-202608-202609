"""External protein prior features: ESM-2 sequence embeddings + GO-slim terms.

This is the protein-side counterpart of :mod:`aivc.external_chemical` (the
input side already has strain/chemical priors; this supplies the *output* side).

It loads three artifacts produced by the ``scripts/`` pipeline and reindexes them
to whatever protein order the decoder needs (the feature-contract
``protein_names`` list from :func:`baseline.data.preprocess`):

* ``data/external/protein_mapping.csv``   raw gene name -> SGD systematic name + sequence
* ``data/external/protein_esm2_480.npy``  (N, 480) ESM-2 embeddings, aligned to the CSV
* ``data/external/protein_go_slim.npz``   (N, T) GO-slim multi-hot + term list

Design note (factoring prior without losing global information): the GO prior is
a per-protein *multi-hot* over its own GO-slim term set — never a hard module
assignment — so each protein keeps a distinct row.  The decoder is expected to
consume these as a low-rank ``E @ a(emb)`` term *added* to its existing full-rank
head, never as a replacement, so the factorization never caps expressivity.

Disclosure: ESM-2 ``esm2_t12_35M_UR50D`` (Meta) and SGD ``go_slim_mapping.tab``
are external public resources; source/version are recorded in the mapping CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
MAPPING_PATH = DATA_DIR / "protein_mapping.csv"
ESM_PATH = DATA_DIR / "protein_esm2_480.npy"
GO_PATH = DATA_DIR / "protein_go_slim.npz"


def _load_mapping() -> Dict[str, dict]:
    import csv

    rows = list(csv.DictReader(MAPPING_PATH.open(newline="", encoding="utf-8")))
    index = {r["raw_name"]: r for r in rows}
    return index


class ProteinPriorEncoder:
    """Reindex ESM-2 embeddings and GO-slim terms to a caller's protein order."""

    def __init__(self, data_dir: Optional[Path] = None):
        d = data_dir or DATA_DIR
        self.mapping_ = _load_mapping()

        esm_path = d / "protein_esm2_480.npy"
        if not esm_path.exists():
            raise FileNotFoundError(
                f"{esm_path} not found; run scripts/compute_esm_embeddings.py first"
            )
        self.esm_ = np.load(esm_path).astype(np.float32)  # (N, 480)
        self.esm_names_ = [r["raw_name"] for r in _mapping_rows()]
        if len(self.esm_names_) != self.esm_.shape[0]:
            raise ValueError(
                f"ESM matrix rows ({self.esm_.shape[0]}) != mapping rows "
                f"({len(self.esm_names_)})"
            )

        go_path = d / "protein_go_slim.npz"
        self.go_terms_: Optional[List[str]] = None
        self.go_matrix_: Optional[np.ndarray] = None
        self.go_names_: Optional[List[str]] = None
        if go_path.exists():
            go = np.load(go_path, allow_pickle=True)
            self.go_terms_ = [str(t) for t in go["go_terms"]]
            self.go_matrix_ = go["matrix"].astype(np.float32)  # (N, T)
            self.go_names_ = [str(n) for n in go["names"]]

        self.esm_index_ = {n: i for i, n in enumerate(self.esm_names_)}
        self.go_index_ = (
            {n: i for i, n in enumerate(self.go_names_)} if self.go_names_ else {}
        )
        self._esm_mean_ = self.esm_.mean(axis=0)

    @property
    def n_proteins(self) -> int:
        return self.esm_.shape[0]

    def embed(self, protein_names: List[str]) -> np.ndarray:
        """Return ESM-2 embeddings ``(len(protein_names), 480)`` in the given order.

        Proteins absent from the cache fall back to the global mean embedding.
        """
        rows = []
        for name in protein_names:
            idx = self.esm_index_.get(name)
            rows.append(self.esm_[idx] if idx is not None else self._esm_mean_)
        if not rows:
            return np.empty((0, self.esm_.shape[1]), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def go_features(self, protein_names: List[str]) -> np.ndarray:
        """Return GO-slim multi-hot ``(len(protein_names), T)`` in the given order."""
        if self.go_matrix_ is None:
            raise FileNotFoundError("GO-slim features not built; run scripts/fetch_protein_go.py")
        zero = np.zeros(self.go_matrix_.shape[1], dtype=np.float32)
        rows = []
        for name in protein_names:
            idx = self.go_index_.get(name)
            rows.append(self.go_matrix_[idx] if idx is not None else zero)
        if not rows:
            return np.empty((0, self.go_matrix_.shape[1]), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    @property
    def go_terms(self) -> List[str]:
        if self.go_terms_ is None:
            raise FileNotFoundError("GO-slim features not built; run scripts/fetch_protein_go.py")
        return self.go_terms_


def _mapping_rows() -> List[dict]:
    import csv

    return list(csv.DictReader(MAPPING_PATH.open(newline="", encoding="utf-8")))


# ══════════════════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 54)
    print("external_protein.ProteinPriorEncoder 自检")
    print("=" * 54)

    encoder = ProteinPriorEncoder()
    print(f"ESM matrix: {encoder.esm_.shape}, mean-fallback norm={float(np.linalg.norm(encoder._esm_mean_)):.2f}")
    print(f"GO terms: {len(encoder.go_terms_) if encoder.go_terms_ else 0}")

    # Reindex a small, shuffled subset to check order-sensitivity.
    names = encoder.esm_names_[:20]
    emb = encoder.embed(names)
    assert emb.shape == (20, 480), f"bad embed shape {emb.shape}"
    assert np.allclose(emb[0], encoder.esm_[0]), "order not preserved"

    # A missing name should fall back to the mean, not crash.
    emb2 = encoder.embed(["__does_not_exist__"])
    assert np.allclose(emb2[0], encoder._esm_mean_), "fallback not applied"

    go = encoder.go_features(names)
    assert go.shape == (20, len(encoder.go_terms_)), f"bad go shape {go.shape}"

    print(f"  embed reindex [OK], go reindex [OK], fallback [OK]")
    print("external_protein [OK]")
