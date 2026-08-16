"""Build protein GO-slim term features (L4) for the competition proteins.

SGD publishes a GO-slim mapping (``go_slim_mapping.tab``) that assigns each ORF a
small curated set of GO terms across the three aspects (P = biological process,
F = molecular function, C = cellular component).  This script turns that file
into a per-protein multi-hot matrix aligned to ``protein_mapping.csv`` order.

A protein is represented by the *set* of GO-slim terms it is annotated with, not
by a single hard module assignment — each protein gets its own multi-hot row, so
two proteins in the same term set still differ if their term sets differ, and no
per-protein identity is collapsed.

Output:
  data/external/protein_go_slim.npz  with keys
    names     (P,)  protein raw names, in protein_mapping.csv order
    go_terms  (T,)  GO-slim term IDs (root terms excluded)
    aspects   (T,)  GO aspect letter per term ('P'/'F'/'C')
    matrix    (P, T) uint8 multi-hot

Disclosure: source = SGD go_slim_mapping.tab, source_version = fetch date.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import requests

# Allow running as ``python scripts/fetch_protein_go.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
MAPPING_PATH = DATA_DIR / "protein_mapping.csv"
OUTPUT_PATH = DATA_DIR / "protein_go_slim.npz"
CACHE_DIR = DATA_DIR / "cache"
GO_SLIM_PATH = CACHE_DIR / "go_slim_mapping.tab"

GO_SLIM_URL = "https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab"

# The three GO root terms are present on every annotated protein and therefore
# carry no discriminative signal; drop them.
ROOT_TERMS = {"GO:0005575", "GO:0003674", "GO:0008150"}

_SYSTEMATIC_RE = re.compile(r"^(Y[A-P][LR]\d{3}[CW](-[A-Z])?|Q\d{4})$")


def _download(url: str, out_path: Path, timeout: int = 120, max_retries: int = 5) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {url}") from last_exc


def _parse_go_slim(path: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Return ``{systematic_name: {aspect: {term_id, ...}}}`` for ORF rows only."""
    go: Dict[str, Dict[str, Set[str]]] = {}
    for line in path.open(encoding="utf-8"):
        if line.startswith("!"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 7:
            continue
        feature, aspect, term_id = fields[0], fields[3], fields[5]
        if not _SYSTEMATIC_RE.match(feature):
            continue
        if aspect not in ("P", "F", "C"):
            continue
        go.setdefault(feature, {}).setdefault(aspect, set()).add(term_id)
    return go


def build_matrix(names: List[str], systematic_names: List[str], go) -> Dict[str, np.ndarray]:
    # Order of GO terms: sorted by ID for reproducibility.  A GO term belongs to
    # exactly one aspect, so the per-term aspect is read straight off the
    # mapping's (systematic -> aspect -> terms) structure.
    term_aspect: Dict[str, str] = {}
    for aspects in go.values():
        for aspect, terms in aspects.items():
            for t in terms:
                term_aspect[t] = aspect

    all_terms = sorted(t for t in term_aspect if t not in ROOT_TERMS)
    term_index = {t: i for i, t in enumerate(all_terms)}

    aspects = np.array([term_aspect[t] for t in all_terms], dtype="<U1")
    matrix = np.zeros((len(names), len(all_terms)), dtype=np.uint8)
    for row, systematic in enumerate(systematic_names):
        aspects_for_protein = go.get(systematic)
        if not aspects_for_protein:
            continue
        for terms in aspects_for_protein.values():
            for t in terms:
                if t in term_index:
                    matrix[row, term_index[t]] = 1
    return {
        "names": np.array(names, dtype=object),
        "go_terms": np.array(all_terms, dtype=object),
        "aspects": aspects,
        "matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protein GO-slim features")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="Output .npz path"
    )
    args = parser.parse_args()

    _download(GO_SLIM_URL, GO_SLIM_PATH)
    go = _parse_go_slim(GO_SLIM_PATH)
    print(f"parsed GO-slim: {len(go)} systematic names with annotations")

    rows = list(csv.DictReader(MAPPING_PATH.open(newline="", encoding="utf-8")))
    names = [r["raw_name"] for r in rows]
    systematic_names = [r["systematic_name"] for r in rows]

    data = build_matrix(names, systematic_names, go)
    np.savez(args.output, **data)
    n_terms = len(data["go_terms"])
    n_annotated = int((data["matrix"].sum(axis=1) > 0).sum())
    print(f"saved {data['matrix'].shape} multi-hot -> {args.output}")
    print(f"  {n_annotated}/{len(names)} proteins have >=1 GO-slim term")
    print(f"  {n_terms} GO-slim terms (root terms excluded)")


if __name__ == "__main__":
    main()
