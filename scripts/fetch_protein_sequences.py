"""Fetch SGD systematic names + protein sequences for the competition proteins.

The proteome CSV columns are standard yeast *gene names* (``AAC1``, ``ZWF1``,
``ZTA1`` ...).  ESM-2 embeddings and GO annotations both key on the SGD
*systematic name* (ORF, e.g. ``YAL001C``), so the first step is a
``gene_name -> systematic_name -> amino-acid-sequence`` mapping.

Sources (single-file downloads from SGD, one HTTP GET each):

* ``SGD_features.tab``     -> gene_name <-> systematic_name <-> feature_type
* ``orf_trans_all.fasta``  -> systematic_name -> protein sequence

Two modes:

* ``python scripts/fetch_protein_sequences.py --skeleton``
  writes ``data/external/protein_mapping.csv`` with only the column names and
  anomaly flags filled in (no network).  Use this to review the name list
  (e.g. the ``1-Oct`` column) before fetching.

* ``python scripts/fetch_protein_sequences.py``  (default)
  downloads the two SGD files, resolves systematic names + sequences, and
  fills the CSV.  Progress is written incrementally so an interrupted request
  does not lose earlier results.

Disclosure: ``source`` / ``source_version`` carry the SGD filename and fetch
date so the mapping satisfies the competition's "external resources must
disclose source and version" rule.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Allow running as ``python scripts/fetch_protein_sequences.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROTEOME_TRAIN = DATA_DIR / "WAYB_WAYC_proteome_raw_train_val.csv"

SGD_FEATURES_URL = "https://downloads.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab"
SGD_FASTA_URL = "https://downloads.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz"

# A systematic (ORF) name looks like YAL001C / YAL001W, optionally with an
# overlapping-ORF suffix (YBL029C-A).  Uncharacterized ORFs in the proteome are
# listed by their systematic name, which is why a meaningful fraction of the
# columns already match this pattern.
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[CW](-[A-Z])?$")

# Mitochondrial ORFs use a Q-number systematic name (Q0080, Q0255 ...) and are
# present in the same orf_trans_all.fasta, just under a different namespace.
_MITO_RE = re.compile(r"^Q\d{4}$")


def _is_systematic_name(name: str) -> bool:
    return bool(_SYSTEMATIC_RE.match(name) or _MITO_RE.match(name))


# Manually pinned resolutions for names the SGD indexes cannot handle.  ``IMP2'``
# is the deprecated prime spelling of IMP21 (the second IMP2); it is reported by
# mass-spec as ``IMP2'`` to distinguish it from the IMP2 gene itself.
MANUAL_OVERRIDES: Dict[str, str] = {
    "IMP2'": "YIL154C",
}


def _categorize_not_found(name: str) -> str:
    """Return a review note for a name the SGD indexes could not resolve."""
    if re.match(r"^\d", name):
        return "not a gene name (likely compound/contaminant)"
    if name in {"REP1", "REP2", "FLP1"}:
        return "2-micron plasmid gene (not in nuclear/mito ORF fasta)"
    if re.match(r"^TY[12][AB]-", name):
        return "Ty retrotransposon gene (SGD transposable-element locus naming)"
    if name.endswith("'"):
        return "prime-suffixed deprecated name"
    return "not found in SGD_features.tab"

CSV_COLUMNS = [
    "raw_name",          # gene name as it appears in the proteome column (e.g. 'AAC1')
    "systematic_name",   # SGD ORF (e.g. 'YAL001C'); empty if unresolved
    "gene_name",         # SGD standard gene name (may differ from raw_name)
    "aliases",           # pipe-separated SGD aliases, for manual review
    "sequence",          # amino-acid sequence (empty in skeleton)
    "seq_length",        # length of the sequence
    "source",            # SGD file the mapping came from
    "source_version",    # fetch date
    "review_status",     # pending_fetch / auto_resolved / already_systematic /
                         # not_found / ambiguous / anomaly_candidate
    "notes",             # manual-review notes
]


def _csv_header(file_path: Path) -> List[str]:
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def read_protein_columns(proteome_path: Path = PROTEOME_TRAIN) -> List[str]:
    """Return the protein column names (everything except ``sample_ID``)."""
    header = _csv_header(proteome_path)
    columns = [c for c in header if c != "sample_ID"]
    return columns


def detect_anomaly(name: str) -> Optional[str]:
    """Return a review note only for names that are clearly not yeast genes.

    Yeast gene names legitimately contain punctuation (``ARG5,6``, ``CUP1-1``,
    ``MF(ALPHA)1``, ``TY1A-BL``), so a punctuation-based heuristic would flood
    the review list with false positives.  The one reliable red flag in this
    dataset is a column that is not a gene at all (``1-Oct``, a compound), which
    the leading-digit check surfaces.  Everything else is left to the SGD
    fetch, where ``not_found`` is the correct signal for manual review.
    """
    if re.match(r"^\d", name):
        return "starts with a digit (likely not a gene)"
    return None


def _parse_sgd_features(text: str) -> Dict[str, dict]:
    """Parse SGD_features.tab into ``{systematic_name: {gene_name, aliases}}``.

    Only ``ORF`` features are kept.  The standard gene name (column 5) and the
    pipe-separated aliases (column 6) are collected so a proteome column can be
    resolved whether it is a standard name or an alias.
    """
    records: Dict[str, dict] = {}
    for line in text.splitlines():
        if not line or line.startswith("!"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        feature_type = fields[1]
        # ORFs are the real genes; pseudogenes and blocked reading frames are
        # also translated in orf_trans_all.fasta, and the mass-spec data does
        # report signal under some of their names (e.g. AAD6), so keep them.
        if feature_type not in {"ORF", "pseudogene", "blocked reading frame"}:
            continue
        systematic = fields[3].strip()
        gene_name = fields[4].strip()
        aliases = [a.strip() for a in fields[5].split("|") if a.strip()]
        if not systematic:
            continue
        records[systematic] = {"gene_name": gene_name, "aliases": aliases}
    return records


def _parse_orf_fasta(text: str) -> Dict[str, str]:
    """Parse orf_trans_all.fasta into ``{systematic_name: sequence}``.

    SGD headers look like ``>YAL001C TFC3 SGDID:S000000001, chrI:..., Verified``
    where the first whitespace token is the systematic name.  Uncharacterized
    ORFs have no gene name, e.g. ``>YAL002W SGDID:S000000003,...``.
    """
    sequences: Dict[str, str] = {}
    current = None
    for line in text.splitlines():
        if line.startswith(">"):
            header = line[1:].strip()
            parts = header.split()
            current = parts[0] if parts else None
            sequences[current] = ""
        elif current is not None:
            sequences[current] += line.strip()
    return sequences


def _build_gene_indexes(
    sgd_records: Dict[str, dict]
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return ``(standard, alias)`` indexes mapping gene name -> systematic names.

    A single name is often the *standard* name of one ORF and an *alias* of
    another (e.g. ``ADH5`` is the standard name of YBR145W but an alias of
    YDL168W).  Keeping the two tiers separate lets the resolver prefer the
    standard-name match, which resolves most of the apparent ambiguities.
    """
    standard: Dict[str, List[str]] = {}
    alias: Dict[str, List[str]] = {}
    for systematic, info in sgd_records.items():
        gene_name = info["gene_name"]
        if gene_name:
            for name in {gene_name, gene_name.upper()}:
                standard.setdefault(name, []).append(systematic)
        for a in info["aliases"]:
            if not a:
                continue
            for name in {a, a.upper()}:
                alias.setdefault(name, []).append(systematic)
    return standard, alias


def _download(url: str, timeout: int = 120, max_retries: int = 5) -> str:
    """Download ``url`` with retries and disk caching, returning UTF-8 text.

    SGD serves some files gzip-compressed; the gzip magic bytes are detected and
    decompressed transparently.  Results are cached under
    ``data/external/cache/`` so re-runs (and later steps) do not re-download,
    and partial transfers are never cached — the bytes are only written after a
    complete response is received.
    """
    import gzip
    import time

    cache_dir = DATA_DIR / "external" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / url.rstrip("/").split("/")[-1]

    if cache_path.exists() and cache_path.stat().st_size > 0:
        content = cache_path.read_bytes()
    else:
        last_exc: Optional[Exception] = None
        content = b""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                content = response.content
                cache_path.write_bytes(content)
                break
            except requests.RequestException as exc:  # includes ChunkedEncodingError
                last_exc = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"download failed after {max_retries} attempts: {url}") from last_exc

    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return content.decode("utf-8")


def build_rows(fetch: bool, proteome_path: Path = PROTEOME_TRAIN) -> List[Dict[str, str]]:
    columns = read_protein_columns(proteome_path)
    source_version = _dt.date.today().isoformat()

    if not fetch:
        rows: List[Dict[str, str]] = []
        for name in columns:
            note = detect_anomaly(name)
            status = "already_systematic" if _is_systematic_name(name) else "pending_fetch"
            if note:
                status = "anomaly_candidate"
            rows.append({
                "raw_name": name,
                "systematic_name": name if _is_systematic_name(name) else "",
                "gene_name": "",
                "aliases": "",
                "sequence": "",
                "seq_length": "",
                "source": "",
                "source_version": source_version,
                "review_status": status,
                "notes": note or "",
            })
        return rows

    print("Downloading SGD_features.tab ...")
    sgd_records = _parse_sgd_features(_download(SGD_FEATURES_URL))
    print(f"  parsed {len(sgd_records)} ORF records")

    print("Downloading orf_trans_all.fasta ...")
    sequences = _parse_orf_fasta(_download(SGD_FASTA_URL))
    print(f"  parsed {len(sequences)} sequences")

    standard_index, alias_index = _build_gene_indexes(sgd_records)

    def _resolve(name: str) -> List[str]:
        """Prefer the standard-name match, then the alias match, then case-fold."""
        for idx in (standard_index, alias_index):
            hits = idx.get(name) or idx.get(name.upper())
            if hits:
                return hits
        return []

    rows = []
    for name in columns:
        note = detect_anomaly(name)
        row = {
            "raw_name": name,
            "systematic_name": "",
            "gene_name": "",
            "aliases": "",
            "sequence": "",
            "seq_length": "",
            "source": "",
            "source_version": source_version,
            "review_status": "not_found",
            "notes": note or "",
        }

        if _is_systematic_name(name):
            # The column is already a systematic name (nuclear Y-number or
            # mitochondrial Q-number).
            if name in sequences:
                row["systematic_name"] = name
                row["gene_name"] = sgd_records.get(name, {}).get("gene_name", "")
                row["aliases"] = "|".join(sgd_records.get(name, {}).get("aliases", []))
                row["sequence"] = sequences[name]
                row["seq_length"] = str(len(sequences[name]))
                row["source"] = "SGD orf_trans_all.fasta"
                row["review_status"] = "auto_resolved"
            else:
                row["notes"] = (note or "").strip() or "systematic name not in fasta"
            rows.append(row)
            continue

        hits = _resolve(name)
        if not hits and name in MANUAL_OVERRIDES:
            hits = [MANUAL_OVERRIDES[name]]
        if not hits:
            row["notes"] = _categorize_not_found(name)
            rows.append(row)  # review_status stays 'not_found'
            continue

        if len(hits) > 1:
            row["systematic_name"] = ""
            row["aliases"] = "|".join(hits)
            row["review_status"] = "ambiguous"
            row["notes"] = (note or "").strip() + (f" maps to {len(hits)} ORFs: {', '.join(hits)}")
            rows.append(row)
            continue

        systematic = hits[0]
        info = sgd_records.get(systematic, {})
        seq = sequences.get(systematic, "")
        row["systematic_name"] = systematic
        row["gene_name"] = info.get("gene_name", "")
        row["aliases"] = "|".join(info.get("aliases", []))
        row["sequence"] = seq
        row["seq_length"] = str(len(seq)) if seq else ""
        row["source"] = "SGD SGD_features.tab + orf_trans_all.fasta"
        row["review_status"] = "auto_resolved" if seq else "resolved_no_sequence"
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    print(f"Wrote {len(rows)} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protein mapping from SGD")
    parser.add_argument(
        "--skeleton", action="store_true",
        help="Write only the name list with anomaly flags (no network).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/external/protein_mapping.csv"),
        help="Output CSV path (default: data/external/protein_mapping.csv)",
    )
    args = parser.parse_args()

    rows = build_rows(fetch=not args.skeleton)
    write_csv(rows, args.output)

    if not args.skeleton:
        status_counts: Dict[str, int] = {}
        for row in rows:
            status_counts[row["review_status"]] = status_counts.get(row["review_status"], 0) + 1
        print("review_status counts:", status_counts)


if __name__ == "__main__":
    main()
