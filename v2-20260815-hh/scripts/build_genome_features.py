#!/usr/bin/env python
"""Build auditable strain mappings and simple public-genome features.

This module reads only the two explicitly supplied competition metadata CSVs and
an allow-listed set of public yeast-genome resources. It never discovers inputs
by walking directories. Network access is disabled unless ``--download`` is
explicitly requested; ``--offline`` is accepted as a documented no-network mode.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPLIT_ORDER = [
    "train", "val_chem_only", "val_strain_only", "val_both", "val_time",
    "test_chem_only", "test_strain_only", "test_both", "test_time",
]
MAPPING_TYPES = ("exact", "supported", "proxy", "unresolved")
CONFIDENCES = ("high", "medium", "low", "none")
STRAIN_ORDER = ("BAH", "BAI", "CEK", "CGD", "CRD", "DHY210")
PUBLIC_ISOLATES = ("BAH", "BAI", "CEK", "CGD", "CRD")
DEFAULT_SEED = 20260814
DOWNLOAD_DATE = "2026-08-14"

# Paths are explicit by design; input discovery is forbidden for this module.
METADATA_BASENAMES = {
    "WAYB_WAYC_metadata_train_val(1).csv",
    "WAYB_WAYC_metadata_test(1).csv",
}
RESOURCE_SPECS: dict[str, dict[str, Any]] = {
    "Peter2018_Supplementary_Tables.xls": {
        "url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-018-0030-5/MediaObjects/41586_2018_30_MOESM3_ESM.xls",
        "version": "Peter et al. 2018 Supplementary Tables 1-21",
        "release_date": "2018-04-11",
        "license": "CC BY 4.0 for article and included supplementary material",
        "sha256": "b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273",
    },
    "1011GWASMatrix.tar.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/1011GWASMatrix.tar.gz",
        "version": "Peter et al. 2018 public server snapshot; index observed 2025-03-31",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "4fb9ae6c2d24f169c522b582e79b81e76e436f34367df1f749ced715fbdc746a",
        "md5": "c68c2ee5d9f4b5c85b8436a4a2f31ba5",
    },
    "genesMatrix_CopyNumber.tab.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/genesMatrix_CopyNumber.tab.gz",
        "version": "Peter et al. 2018 1,011-isolate pangenome matrix",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "d22fe0a2024921163739553afb612e97e229c99b51a4301021c2433e11b2ee4d",
        "md5": "c36cdf023c554580fe5b67101742fc09",
    },
    "genesMatrix_PresenceAbsence.tab.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/genesMatrix_PresenceAbsence.tab.gz",
        "version": "Peter et al. 2018 1,011-isolate pangenome matrix",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "63b62d6761c378f71360a259add9ad3317befb27cfa45944d0b11433766bf7fc",
        "md5": "b3b96719de226eaa277589d8d01f9202",
    },
    "genesMatrix_Frameshift.tab.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/genesMatrix_Frameshift.tab.gz",
        "version": "Peter et al. 2018 1,011-isolate gene frameshift matrix",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "110debdf846e48ff71d21c69ec66b1dfde0239be3e199ba432544d218634bee5",
        "md5": "6f8ec446283817be303e20f5661550b0",
    },
    "1011DistanceMatrixBasedOnSNPs.tab.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnSNPs.tab.gz",
        "version": "Peter et al. 2018 1,011-isolate SNP distance matrix",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928",
        "md5": "817f7456393dee33f6b41d3d59ddc985",
    },
    "1011DistanceMatrixBasedOnORFs.tab.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnORFs.tab.gz",
        "version": "Peter et al. 2018 1,011-isolate ORF distance matrix",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "d443b51a9cbff5b7f223b088902bdf1db711ed07aa692eac0834fd1309420568",
        "md5": "7fd9048da9b828041a4333c0e991e6d8",
    },
    "allORFs_pangenome.fasta.gz": {
        "url": "http://1002genomes.u-strasbg.fr/files/allORFs_pangenome.fasta.gz",
        "version": "Peter et al. 2018 7,796 pangenome ORFs",
        "release_date": "2018-04-11",
        "license": "Paper is CC BY 4.0; raw-server reuse terms are not separately stated",
        "sha256": "22417aa8c9f82d9e495e1cc663e646a06b69b9a542a3974d1b65f3c8ed224478",
        "md5": "4f1ebba402ae64e52f2af72fa6942f38",
    },
}

MAPPINGS: dict[str, dict[str, str]] = {
    "BAH": {"external_isolate_id": "SX3", "external_matrix_id": "BAH", "external_assembly_id": "", "external_full_name": "Saccharomyces cerevisiae isolate SX3", "mapping_type": "supported", "mapping_confidence": "high", "evidence_summary": "Peter Table S1 maps standardized name BAH to isolate SX3; NCBI BioProject PRJNA396809 independently lists SX3, and later peer-reviewed work reports SX3 as BAH.", "evidence_urls": "https://doi.org/10.1038/s41586-018-0030-5|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396809|https://doi.org/10.1111/1751-7915.70337", "candidates": "SX3", "rejected_candidates": "No alternate isolate accepted; acronym similarity alone is disallowed."},
    "BAI": {"external_isolate_id": "BJ6", "external_matrix_id": "BAI", "external_assembly_id": "", "external_full_name": "Saccharomyces cerevisiae isolate BJ6", "mapping_type": "supported", "mapping_confidence": "high", "evidence_summary": "Peter Table S1 maps standardized name BAI to isolate BJ6; NCBI BioProject PRJNA396809 independently lists BJ6, and ScRAPdb lists BJ6 with BAI aliases.", "evidence_urls": "https://doi.org/10.1038/s41586-018-0030-5|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396809|https://evomicslab.org/db/ScRAPdb/strains/", "candidates": "BJ6", "rejected_candidates": "No alternate isolate accepted; acronym similarity alone is disallowed."},
    "CEK": {"external_isolate_id": "JCM_2985-4B", "external_matrix_id": "CEK", "external_assembly_id": "", "external_full_name": "Saccharomyces cerevisiae isolate JCM_2985-4B", "mapping_type": "supported", "mapping_confidence": "medium", "evidence_summary": "Peter Table S1 maps standardized name CEK to JCM_2985-4B; an independent University of Washington re-analysis table repeats CEK, JCM_2985-4B and SRA ERR1309167.", "evidence_urls": "https://doi.org/10.1038/s41586-018-0030-5|https://digital.lib.washington.edu/researchworks/bitstreams/1fe8f5c6-1b14-4bd1-abb6-14f74c0ea9be/download", "candidates": "JCM_2985-4B;ERR1309167", "rejected_candidates": "No alternate isolate accepted; no direct competition-provider identity statement was found."},
    "CGD": {"external_isolate_id": "UCD_09-448", "external_matrix_id": "CGD", "external_assembly_id": "", "external_full_name": "Saccharomyces cerevisiae isolate UCD_09-448", "mapping_type": "supported", "mapping_confidence": "medium", "evidence_summary": "Peter Table S1 maps standardized name CGD to UCD_09-448; independent academic strain tables repeat the CGD/UCD_09-448 pair.", "evidence_urls": "https://doi.org/10.1038/s41586-018-0030-5|https://repository.lib.ncsu.edu/server/api/core/bitstreams/5b4e10ff-fd0e-4e8e-9261-e2ca87dadb03/content|https://riunet.upv.es/bitstreams/f9c19cb9-cf10-4018-a43e-6cbab176301d/download", "candidates": "UCD_09-448", "rejected_candidates": "No alternate isolate accepted; no direct competition-provider identity statement was found."},
    "CRD": {"external_isolate_id": "FIMA_3", "external_matrix_id": "CRD", "external_assembly_id": "", "external_full_name": "Saccharomyces cerevisiae isolate FIMA_3", "mapping_type": "supported", "mapping_confidence": "medium", "evidence_summary": "Peter Table S1 maps standardized name CRD to FIMA_3; independent peer-reviewed work identifies FIMA_3 as the Italian winery isolate used from Peter et al.", "evidence_urls": "https://doi.org/10.1038/s41586-018-0030-5|https://doi.org/10.15252/msb.202110160", "candidates": "FIMA_3", "rejected_candidates": "No alternate isolate accepted; no direct competition-provider identity statement was found."},
    "DHY210": {"external_isolate_id": "S288C", "external_matrix_id": "", "external_assembly_id": "GCF_000146045.2", "external_full_name": "Saccharomyces cerevisiae S288C reference strain", "mapping_type": "proxy", "mapping_confidence": "low", "evidence_summary": "The task approach permits S288C only as an explicit DHY210 proxy. It is not an identity claim; DHY210-specific variants are lost.", "evidence_urls": "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2|local:references/20260812Approach.pdf", "candidates": "S288C proxy", "rejected_candidates": "Rejected exact/equivalent status: no public one-to-one DHY210 to S288C identity evidence."},
}

RAW_FEATURE_NAMES = [
    "snp_count", "snp_per_kb", "homozygous_snp_count", "heterozygous_snp_count",
    "heterozygous_snp_fraction", "singleton_snp_count", "mean_read_coverage_x",
    "assembly_total_bp", "assembly_n50_bp", "assembly_contig_count", "assembly_max_contig_bp",
    "assembly_gc_percent", "ploidy_n", "aneuploid_segment_count",
    "pangenome_orf_called_count", "pangenome_orf_absent_count", "pangenome_orf_absent_fraction",
    "pangenome_orf_multi_copy_count", "pangenome_orf_multi_copy_fraction",
    "pangenome_copy_number_absolute_burden", "frameshift_called_gene_count",
    "frameshift_gene_count", "frameshift_gene_fraction", "gwas_autosomal_called_marker_count",
    "gwas_autosomal_allele1_homozygous_count", "gwas_autosomal_heterozygous_count",
    "gwas_autosomal_missing_marker_count", "gwas_autosomal_allele1_dosage_fraction",
]
TECHNICAL_QC_FEATURE_NAMES = [
    "mean_read_coverage_x", "assembly_n50_bp", "assembly_contig_count", "assembly_max_contig_bp",
]
MAIN_FEATURE_NAMES = [name for name in RAW_FEATURE_NAMES if name not in TECHNICAL_QC_FEATURE_NAMES]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # nosec - compatibility check against upstream manifest
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def order_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_metadata_path(path: Path) -> None:
    if path.name not in METADATA_BASENAMES:
        raise ValueError(f"metadata basename is not allow-listed: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(path)


def extract_strains(metadata_paths: list[Path]) -> list[dict[str, Any]]:
    required = {"Strains", "split_final"}
    grouped: dict[str, dict[str, Any]] = {}
    for path in metadata_paths:
        validate_metadata_path(path)
        fields, rows = read_csv_rows(path)
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in rows:
            split = row["split_final"]
            if split not in SPLIT_ORDER:
                raise ValueError(f"unexpected split_final={split!r}")
            strain = row["Strains"].strip()
            record = grouped.setdefault(strain, {"splits": set(), "count": 0})
            record["splits"].add(split)
            record["count"] += 1
    rows = []
    for strain in STRAIN_ORDER:
        if strain not in grouped:
            raise ValueError(f"expected strain absent from real metadata: {strain}")
        splits = [value for value in SPLIT_ORDER if value in grouped[strain]["splits"]]
        rows.append({
            "strain_id": strain,
            "seen_splits": "|".join(splits),
            "first_seen_split": splits[0],
            "seen_in_train": "train" in splits,
            "seen_in_validation": any(value.startswith("val_") for value in splits),
            "seen_in_test": any(value.startswith("test_") for value in splits),
            "metadata_row_count": grouped[strain]["count"],
        })
    unexpected = set(grouped) - set(STRAIN_ORDER)
    if unexpected:
        raise ValueError(f"unreviewed strain(s) found in real metadata: {sorted(unexpected)}")
    return rows


def ensure_resources(raw_dir: Path, download: bool) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, spec in RESOURCE_SPECS.items():
        path = raw_dir / name
        if not path.exists() and download:
            with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                with urllib.request.urlopen(spec["url"], timeout=120) as response, temporary_path.open("wb") as target:
                    shutil.copyfileobj(response, target)
                if sha256_file(temporary_path) != spec["sha256"]:
                    raise ValueError(f"downloaded SHA-256 mismatch for {name}")
                os.replace(temporary_path, path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        if not path.is_file():
            raise FileNotFoundError(f"offline resource missing: {path}; use --download to fetch allow-listed resources")
        sha = sha256_file(path)
        if sha != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}: {sha}")
        upstream_md5 = spec.get("md5", "")
        actual_md5 = md5_file(path) if upstream_md5 else ""
        if upstream_md5 and actual_md5 != upstream_md5:
            raise ValueError(f"upstream MD5 mismatch for {name}: {actual_md5}")
        records.append({
            "name": name, "path": str(path.resolve()), "url": spec["url"],
            "version": spec["version"], "release_date": spec["release_date"],
            "download_date": DOWNLOAD_DATE, "license": spec["license"],
            "bytes": path.stat().st_size, "sha256": sha, "upstream_md5": upstream_md5,
            "upstream_md5_verified": bool(upstream_md5),
        })
    return records


def xls_rows(xls_path: Path, sheet_name: str, header_row: int) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("xlrd>=2.0 is required only to read the frozen Peter 2018 .xls supplement") from exc
    book = xlrd.open_workbook(str(xls_path), on_demand=True)
    sheet = book.sheet_by_name(sheet_name)
    base_headers = [str(sheet.cell_value(header_row, col)).strip() for col in range(sheet.ncols)]
    seen: dict[str, int] = {}
    headers = []
    for base in base_headers:
        occurrence = seen.get(base, 0)
        headers.append(base if occurrence == 0 else f"{base}.{occurrence}")
        seen[base] = occurrence + 1
    rows = []
    for row_index in range(header_row + 1, sheet.nrows):
        values = [sheet.cell_value(row_index, col) for col in range(sheet.ncols)]
        if any(value not in ("", None) for value in values):
            rows.append(dict(zip(headers, values)))
    book.release_resources()
    return headers, rows


def keyed_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]).strip(): row for row in rows if str(row.get(key, "")).strip()}


def parse_row_matrix(path: Path, strains: tuple[str, ...]) -> tuple[list[str], dict[str, np.ndarray]]:
    selected: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        names = header[1:]
        for row in reader:
            if row and row[0] in strains:
                selected[row[0]] = np.asarray([np.nan if value == "NA" else float(value) for value in row[1:]], dtype=np.float64)
    if set(selected) != set(strains):
        raise ValueError(f"row matrix missing strains: {sorted(set(strains) - set(selected))}")
    return names, selected


def parse_frameshift_matrix(path: Path, strains: tuple[str, ...]) -> tuple[list[str], dict[str, np.ndarray]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        columns = {strain: header.index(strain) for strain in strains}
        genes, values = [], {strain: [] for strain in strains}
        for row in reader:
            genes.append(row[1] or row[0])
            for strain, position in columns.items():
                values[strain].append(np.nan if row[position] == "NA" else float(row[position]))
    return genes, {strain: np.asarray(vector, dtype=np.float64) for strain, vector in values.items()}


def parse_gwas(path: Path, strains: tuple[str, ...]) -> dict[str, dict[str, float]]:
    with tarfile.open(path, "r:gz") as archive:
        family = [line.decode("ascii").split()[0] for line in archive.extractfile("1011GWAS_matrix.fam")]  # type: ignore[arg-type]
        chromosome = np.asarray([int(line.decode("ascii").split("\t", 1)[0]) for line in archive.extractfile("1011GWAS_matrix.bim")], dtype=np.int16)  # type: ignore[arg-type]
        bed = archive.extractfile("1011GWAS_matrix.bed").read()  # type: ignore[union-attr]
    if bed[:3] != b"\x6c\x1b\x01":
        raise ValueError("GWAS PLINK bed magic/mode bytes are invalid")
    bytes_per_marker = (len(family) + 3) // 4
    payload = np.frombuffer(bed[3:], dtype=np.uint8)
    if payload.size != chromosome.size * bytes_per_marker:
        raise ValueError("GWAS bed dimensions disagree with fam/bim")
    packed = payload.reshape(chromosome.size, bytes_per_marker)
    autosomal = chromosome <= 16
    result = {}
    for strain in strains:
        index = family.index(strain)
        codes = ((packed[:, index // 4] >> (2 * (index % 4))) & 3)[autosomal]
        called = int(np.count_nonzero(codes != 1))
        hom_nonref = int(np.count_nonzero(codes == 0))
        hetero = int(np.count_nonzero(codes == 2))
        missing = int(np.count_nonzero(codes == 1))
        # PLINK SNP-major two-bit encoding: 00 homozygous first BIM allele,
        # 10 heterozygous, 11 homozygous second BIM allele, 01 missing. The
        # matrix server uses BIM numeric alleles and does not expose REF/ALT,
        # so this is explicitly called allele1 dosage, not an independently
        # reconstructed VCF REF/ALT burden.
        dosage_fraction = (2 * hom_nonref + hetero) / (2 * called) if called else math.nan
        result[strain] = {"called": called, "hom_nonref": hom_nonref, "hetero": hetero, "missing": missing, "dosage_fraction": dosage_fraction}
    return result


def build_raw_features(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _, s1_rows = xls_rows(raw_dir / "Peter2018_Supplementary_Tables.xls", "Table S1", 3)
    _, s8_rows = xls_rows(raw_dir / "Peter2018_Supplementary_Tables.xls", "Table S8", 2)
    _, s16_rows = xls_rows(raw_dir / "Peter2018_Supplementary_Tables.xls", "Table S16", 2)
    _, s17_rows = xls_rows(raw_dir / "Peter2018_Supplementary_Tables.xls", "Table S17", 2)
    s1 = keyed_table(s1_rows, "Standardized name")
    s8 = keyed_table(s8_rows, "Standardized_name")
    s16 = keyed_table(s16_rows, "Isolate")
    s17 = keyed_table(s17_rows, "Isolate")

    copy_names, copy_values = parse_row_matrix(raw_dir / "genesMatrix_CopyNumber.tab.gz", PUBLIC_ISOLATES)
    presence_names, presence_values = parse_row_matrix(raw_dir / "genesMatrix_PresenceAbsence.tab.gz", PUBLIC_ISOLATES)
    if copy_names != presence_names:
        raise ValueError("copy-number and presence/absence ORF orders differ")
    frame_genes, frame_values = parse_frameshift_matrix(raw_dir / "genesMatrix_Frameshift.tab.gz", PUBLIC_ISOLATES)
    gwas = parse_gwas(raw_dir / "1011GWASMatrix.tar.gz", PUBLIC_ISOLATES)

    raw = np.zeros((len(STRAIN_ORDER), len(RAW_FEATURE_NAMES)), dtype=np.float64)
    valid = np.ones_like(raw, dtype=bool)
    for row_index, strain in enumerate(PUBLIC_ISOLATES):
        a, b, c, d = s1[strain], s8[strain], s16[strain], s17[strain]
        present = presence_values[strain]
        copy = copy_values[strain]
        frame = frame_values[strain]
        called_orf = np.isfinite(present) & np.isfinite(copy)
        called_frame = np.isfinite(frame)
        ploidy = float(str(a["Ploidy"]).strip())
        vector = [
            b["# SNPs"], b["SNPs.kb-1"], b["# homozygous SNPs"], b["# heterozygous SNPs"],
            b["proportion of heterozygous SNPs"], a["Number of singletons"], a["Mean coverage"],
            d["total_bases"], d["n50"], d["total_contigs"], d["contig_max_size"], d["CG content"],
            ploidy, c["total"], int(called_orf.sum()), int(np.count_nonzero((present == 0) & called_orf)),
            float(np.mean(present[called_orf] == 0)), int(np.count_nonzero((copy > 1) & called_orf)),
            float(np.mean(copy[called_orf] > 1)), float(np.sum(np.abs(copy[called_orf] - 1))),
            int(called_frame.sum()), int(np.count_nonzero((frame > 0) & called_frame)),
            float(np.mean(frame[called_frame] > 0)), gwas[strain]["called"], gwas[strain]["hom_nonref"],
            gwas[strain]["hetero"], gwas[strain]["missing"], gwas[strain]["dosage_fraction"],
        ]
        raw[row_index] = np.asarray(vector, dtype=np.float64)

    # Explicit S288C proxy: only assembly facts directly supported by the fixed
    # NCBI R64 record (17 sequences, 12,157,105 bp, max chromosome 1,531,933 bp,
    # GC 38.15%) are valid. Peter isolate-matrix fields, their denominators, and
    # every dependent count/fraction/burden remain invalid. We do not infer
    # biological zero merely because S288C is the reference.
    proxy_index = STRAIN_ORDER.index("DHY210")
    proxy = {
        "assembly_total_bp": 12157105.0,
        "assembly_contig_count": 17.0,
        "assembly_max_contig_bp": 1531933.0,
        "assembly_gc_percent": 38.15,
    }
    valid[proxy_index] = False
    for name, value in proxy.items():
        position = RAW_FEATURE_NAMES.index(name)
        raw[proxy_index, position] = value
        valid[proxy_index, position] = True

    support = {
        "pangenome_orf_order_sha256": order_hash(copy_names),
        "pangenome_orf_count": len(copy_names),
        "frameshift_gene_order_sha256": order_hash(frame_genes),
        "frameshift_gene_count": len(frame_genes),
        "source_isolate_names": {strain: str(s1[strain]["Isolate name"]) for strain in PUBLIC_ISOLATES},
        "s288c_valid_feature_sources": {
            "assembly_total_bp": "NCBI GCF_000146045.2 assembly R64; 12,157,105 bp",
            "assembly_contig_count": "NCBI GCF_000146045.2 assembly R64; 16 nuclear chromosomes plus mitochondrion",
            "assembly_max_contig_bp": "NCBI GCF_000146045.2 chromosome IV NC_001136.10; 1,531,933 bp",
            "assembly_gc_percent": "NCBI GCF_000146045.2 assembly R64; 38.15% GC",
        },
        "s288c_invalid_dependency_groups": {
            "pangenome": "S288C is not a row in the Peter pangenome matrices; called denominator unavailable",
            "copy_number": "S288C is not a row in the Peter copy-number matrix; called denominator unavailable",
            "frameshift": "S288C is not a column in the Peter frameshift matrix; called denominator unavailable",
            "gwas": "S288C is not a sample in the Peter PLINK fam; called denominator unavailable",
            "ploidy": "No frozen authority was acquired for the specific S288C proxy culture used by this task",
        },
    }
    return raw, valid, support


def fit_transform(raw: np.ndarray, valid: np.ndarray, fit_mask: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray | int]:
    fit_rows = raw[fit_mask]
    fit_valid = valid[fit_mask]
    if fit_rows.shape[0] < 2:
        raise ValueError("at least two eligible train strains are required")
    impute = np.empty(raw.shape[1], dtype=np.float64)
    for column in range(raw.shape[1]):
        values = fit_rows[fit_valid[:, column], column]
        if values.size == 0:
            raise ValueError(f"feature has no valid eligible-train values: {feature_names[column]}")
        impute[column] = float(np.median(values))
    filled = np.where(valid, raw, impute)
    mean = filled[fit_mask].mean(axis=0)
    scale = filled[fit_mask].std(axis=0, ddof=0)
    variance_mask = scale > 0
    safe_scale = np.where(variance_mask, scale, 1.0)
    standardized = (filled - mean) / safe_scale
    standardized[:, ~variance_mask] = 0.0
    # Imputation exists only to apply frozen train transforms. Invalid entity-
    # feature cells are never exposed as imputed z-scores to the model.
    standardized[~valid] = 0.0
    centered_fit = standardized[fit_mask][:, variance_mask]
    pca_center = centered_fit.mean(axis=0)
    centered_fit = centered_fit - pca_center
    rank = int(np.linalg.matrix_rank(centered_fit))
    if rank:
        _, singular, vt = np.linalg.svd(centered_fit, full_matrices=False)
        n_components = min(rank, fit_rows.shape[0] - 1)
        components = vt[:n_components]
        scores = (standardized[:, variance_mask] - pca_center) @ components.T
        component_variance = (singular[:n_components] ** 2) / (fit_rows.shape[0] - 1)
        total_variance = float(np.sum((singular ** 2) / (fit_rows.shape[0] - 1)))
        explained = component_variance / total_variance if total_variance else np.zeros(n_components)
    else:
        n_components = 0
        components = np.zeros((0, int(variance_mask.sum())), dtype=np.float64)
        scores = np.zeros((raw.shape[0], 0), dtype=np.float64)
        explained = np.zeros(0, dtype=np.float64)
    return {
        "imputation_median": impute, "scaler_mean": mean, "scaler_scale": safe_scale,
        "variance_mask": variance_mask, "standardized": standardized,
        "pca_center": pca_center, "pca_components": components, "pca_scores": scores,
        "pca_explained_variance_ratio": explained, "rank": rank,
        "n_components": n_components,
    }


def apply_feature_variant(
    features: np.ndarray,
    mode: str,
    *,
    mapping_type: Iterable[str] | None = None,
    feature_valid: Iterable[bool] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a feature variant and target-row-to-source-row permutation."""
    values = np.asarray(features)
    identity = np.arange(len(values), dtype=np.int64)
    if mode == "correct":
        return values.copy(), identity
    if mode == "zero":
        return np.zeros_like(values), identity
    if mode != "shuffle":
        raise ValueError("mode must be correct, shuffle, or zero")
    if mapping_type is None or feature_valid is None:
        raise ValueError("shuffle requires mapping_type and feature_valid")
    mapping = np.asarray(list(mapping_type), dtype=str)
    validity = np.asarray(list(feature_valid), dtype=bool)
    if len(mapping) != len(values) or len(validity) != len(values):
        raise ValueError("shuffle metadata length mismatch")
    eligible = np.isin(mapping, ["exact", "supported"]) & validity
    positions = np.flatnonzero(eligible)
    shuffled = np.random.default_rng(seed).permutation(positions)
    permutation = identity.copy()
    permutation[positions] = shuffled
    return values[permutation].copy(), permutation


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_go_status(output_dir: Path) -> None:
    rows = [
        {
            "mapping_status": "not_available",
            "database": "SGD Yeast GO Slim",
            "frozen_version": "download attempted 2026-08-14; complete snapshot not acquired",
            "gene_id": "",
            "go_slim_id": "",
            "go_slim_term": "",
            "reason": "Official download was incomplete and was deleted. The 7,796 pangenome ORF identifiers cannot be reliably joined to SGD systematic genes without a validated crosswalk; no zero-valued GO features were fabricated.",
            "source_url": "https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab",
        }
    ]
    write_csv(output_dir / "GO_OR_PATHWAY_MAPPING.csv", rows, list(rows[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build V1 public-genome strain features without label access.")
    parser.add_argument("--metadata-train-val", type=Path, required=True)
    parser.add_argument("--metadata-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-cache-dir", type=Path, required=True)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="require a complete verified local raw cache (default)")
    network.add_argument("--download", action="store_true", help="download missing allow-listed public resources")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    raw_dir = args.raw_cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_paths = [args.metadata_train_val.resolve(), args.metadata_test.resolve()]
    entities = extract_strains(metadata_paths)
    resource_records = ensure_resources(raw_dir, args.download)
    raw, raw_valid, support = build_raw_features(raw_dir)

    mapping_rows = []
    for entity in entities:
        row = dict(entity)
        row.update(MAPPINGS[row["strain_id"]])
        row.update({
            "source_files": "Peter2018_Supplementary_Tables.xls|1011GWASMatrix.tar.gz|genesMatrix_CopyNumber.tab.gz|genesMatrix_PresenceAbsence.tab.gz|genesMatrix_Frameshift.tab.gz" if row["strain_id"] in PUBLIC_ISOLATES else "SGD/NCBI S288C R64 reference facts",
            "reference_genome": "S288C R64-1-1 / GCF_000146045.2",
            "coordinate_system": "Peter 2018 R64-1-1; chromosomes 1-16; 1-based source coordinates",
            "download_date": DOWNLOAD_DATE,
            "source_sha256": RESOURCE_SPECS["Peter2018_Supplementary_Tables.xls"]["sha256"] if row["strain_id"] in PUBLIC_ISOLATES else "not_downloaded_reference_fasta",
            "proxy_flag": row["mapping_type"] == "proxy",
            "feature_valid": bool(raw_valid[STRAIN_ORDER.index(row["strain_id"])].any()),
        })
        mapping_rows.append(row)
    mapping_columns = [
        "strain_id", "seen_splits", "first_seen_split", "seen_in_train", "seen_in_validation", "seen_in_test", "metadata_row_count",
        "external_isolate_id", "external_matrix_id", "external_assembly_id", "external_full_name", "mapping_type", "mapping_confidence", "proxy_flag",
        "feature_valid", "source_files", "reference_genome", "coordinate_system", "evidence_summary", "evidence_urls", "download_date",
        "source_sha256", "candidates", "rejected_candidates",
    ]
    write_csv(output_dir / "strain_mapping.csv", mapping_rows, mapping_columns)

    fit_mask = np.asarray([
        bool(row["seen_in_train"]) and row["mapping_type"] in {"exact", "supported"} and bool(row["feature_valid"])
        for row in mapping_rows
    ])
    main_positions = np.asarray([RAW_FEATURE_NAMES.index(name) for name in MAIN_FEATURE_NAMES], dtype=np.int64)
    qc_positions = np.asarray([RAW_FEATURE_NAMES.index(name) for name in TECHNICAL_QC_FEATURE_NAMES], dtype=np.int64)
    main_raw = raw[:, main_positions]
    main_valid = raw_valid[:, main_positions]
    qc_raw = raw[:, qc_positions]
    qc_valid = raw_valid[:, qc_positions]
    transformed = fit_transform(main_raw, main_valid, fit_mask, MAIN_FEATURE_NAMES)
    standardized = transformed["standardized"]
    assert isinstance(standardized, np.ndarray)
    # The public isolates have all main dimensions. The S288C proxy retains a
    # dimension-level validity mask because read-derived fields are unavailable.
    model_valid_mask = main_valid.copy()
    mapping_types = np.asarray([row["mapping_type"] for row in mapping_rows], dtype="U16")
    confidences = np.asarray([row["mapping_confidence"] for row in mapping_rows], dtype="U8")
    proxy_flags = mapping_types == "proxy"
    np.savez_compressed(
        output_dir / "strain_features.npz",
        strain_ids=np.asarray(STRAIN_ORDER, dtype="U16"),
        external_isolate_ids=np.asarray([row["external_isolate_id"] for row in mapping_rows], dtype="U32"),
        external_matrix_ids=np.asarray([row["external_matrix_id"] for row in mapping_rows], dtype="U16"),
        external_assembly_ids=np.asarray([row["external_assembly_id"] for row in mapping_rows], dtype="U32"),
        mapping_type=mapping_types,
        mapping_confidence=confidences,
        proxy_flag=proxy_flags,
        raw_feature_names=np.asarray(RAW_FEATURE_NAMES, dtype="U64"),
        raw_features=raw.astype(np.float32),
        raw_feature_valid_mask=raw_valid,
        qc_feature_names=np.asarray(TECHNICAL_QC_FEATURE_NAMES, dtype="U64"),
        qc_features=qc_raw.astype(np.float32),
        qc_feature_valid_mask=qc_valid,
        genome_feature_names=np.asarray(MAIN_FEATURE_NAMES, dtype="U64"),
        genome_features=standardized.astype(np.float32),
        feature_valid_mask=model_valid_mask,
        eligible_train_fit_mask=fit_mask,
        imputation_median=np.asarray(transformed["imputation_median"], dtype=np.float32),
        scaler_mean=np.asarray(transformed["scaler_mean"], dtype=np.float32),
        scaler_scale=np.asarray(transformed["scaler_scale"], dtype=np.float32),
        variance_mask=np.asarray(transformed["variance_mask"], dtype=bool),
        pca_features=np.asarray(transformed["pca_scores"], dtype=np.float32),
        pca_center=np.asarray(transformed["pca_center"], dtype=np.float32),
        pca_components=np.asarray(transformed["pca_components"], dtype=np.float32),
        pca_explained_variance_ratio=np.asarray(transformed["pca_explained_variance_ratio"], dtype=np.float32),
    )
    index_rows = []
    for index, row in enumerate(mapping_rows):
        index_rows.append({
            "row_index": index, "strain_id": row["strain_id"], "external_isolate_id": row["external_isolate_id"],
            "external_matrix_id": row["external_matrix_id"], "external_assembly_id": row["external_assembly_id"],
            "mapping_type": row["mapping_type"], "mapping_confidence": row["mapping_confidence"],
            "proxy_flag": row["proxy_flag"], "feature_valid": row["feature_valid"],
            "seen_splits": row["seen_splits"], "first_seen_split": row["first_seen_split"],
        })
    write_csv(output_dir / "strain_feature_index.csv", index_rows, list(index_rows[0]))
    build_go_status(output_dir)

    schema = {
        "schema_version": "1.0.0",
        "generated_date": DOWNLOAD_DATE,
        "strain_id_order": list(STRAIN_ORDER),
        "strain_id_order_sha256": order_hash(STRAIN_ORDER),
        "identity_fields": {
            "strain_id": "competition metadata Strains value",
            "external_isolate_id": "public biological isolate name (SX3, BJ6, JCM_2985-4B, UCD_09-448, FIMA_3, or S288C)",
            "external_matrix_id": "Peter matrix row/column ID for the five public isolates; empty for DHY210 proxy",
            "external_assembly_id": "reference assembly accession for the S288C proxy; empty for Peter isolates in this V1",
        },
        "genome_dim": len(MAIN_FEATURE_NAMES),
        "raw_feature_dim": len(RAW_FEATURE_NAMES),
        "qc_feature_dim": len(TECHNICAL_QC_FEATURE_NAMES),
        "genome_features": {
            "description": "train-only median-imputed, z-scored biological V1 feature matrix; zero-variance columns and every invalid entity-feature cell are forced to zero",
            "dtype": "float32", "shape": [len(STRAIN_ORDER), len(MAIN_FEATURE_NAMES)],
            "feature_names": MAIN_FEATURE_NAMES, "feature_order_sha256": order_hash(MAIN_FEATURE_NAMES),
        },
        "feature_valid_mask": {
            "shape": [len(STRAIN_ORDER), len(MAIN_FEATURE_NAMES)],
            "semantics": "dimension-level availability; public isolates have all dimensions, while the S288C proxy masks unavailable read/matrix fields",
        },
        "raw_features": {
            "description": "all biological and technical/QC source summaries before main-feature selection",
            "feature_names": RAW_FEATURE_NAMES, "feature_order_sha256": order_hash(RAW_FEATURE_NAMES),
        },
        "technical_qc": {
            "description": "retained for audit only and excluded from genome_features, scaler and PCA",
            "feature_names": TECHNICAL_QC_FEATURE_NAMES,
            "feature_order_sha256": order_hash(TECHNICAL_QC_FEATURE_NAMES),
        },
        "coordinate_contract": {
            "reference_assembly": "Saccharomyces cerevisiae S288C R64-1-1",
            "refseq_accession": "GCF_000146045.2",
            "sequence_release_date": "2011-02-03",
            "chromosome_naming": "Peter matrices use chromosomes 1-16; mitochondrial chromosome 17 is excluded from GWAS autosomal summaries",
            "annotation_version": "SGD R64-5-1 (2026-01-23) documented but not used for coordinates or V1 aggregation",
            "vcf_reference_allele_direction": "Peter population calls are aligned to S288C R64-1-1; raw VCF was not downloaded. PLINK numeric-allele summaries retain source coding and must not be interpreted as independently verified REF/ALT.",
            "multiallelic_handling": "not_available in the high-MAF biallelic GWAS matrix; no reconstruction attempted",
            "variant_definitions": {"snp": "Peter Table S8 project SNP definition", "insertion": "not_available without full gVCF", "deletion": "not_available without full gVCF"},
            "missing_genotype": "PLINK two-bit code 01; excluded from called-marker denominator",
            "heterozygous_encoding": "PLINK two-bit code 10; one unit in a 0/1/2 dosage numerator",
            "homozygous_encoding": "PLINK 00/11 according to source BIM allele order; Peter Table S8 supplies total homozygous SNP counts",
            "callable_region": "not_available; coverage and assembly length are reported separately and never called callable_bp",
            "low_quality_or_repeat_filter": "inherited from Peter et al. pipeline; no additional filtering possible from the derived matrix",
            "gene_region_definition": "not_used in V1 because full VCF and a validated coordinate annotation were not both cached",
            "multiple_transcripts": "not_applicable in V1; future rule proposed: union exons per SGD systematic gene",
        },
        "fit_contract": {
            "expression": "seen_in_train AND mapping_type in {exact,supported} AND feature_valid",
            "fit_strains": [STRAIN_ORDER[i] for i in np.flatnonzero(fit_mask)],
            "fit_mask": fit_mask.tolist(),
            "missing_value_fit": "feature-wise median on eligible fit strains only",
            "standardization": "feature-wise z-score, ddof=0, eligible fit strains only",
            "variance_filter": "strictly positive eligible-fit standard deviation",
            "matrix_rank": int(transformed["rank"]),
            "pca_n_components": int(transformed["n_components"]),
            "pca_role": "auxiliary only; standardized interpretable features remain the primary artifact",
            "pca_explained_variance_ratio": np.asarray(transformed["pca_explained_variance_ratio"]).tolist(),
        },
        "supplemental_orders": support,
        "unavailable_feature_groups": [
            "indel count and callable-bp burden", "chromosome-specific variant burden", "coding/promoter burden",
            "per-gene SNP/indel burden normalized by callable gene length", "GO/pathway aggregated burden",
        ],
        "fallback_contract": {"unresolved": "all-zero genome_features and feature_valid_mask=false", "proxy": "proxy features with proxy_flag=true; excluded from all fit statistics"},
        "variant_interface": {"modes": ["correct", "shuffle", "zero"], "default_seed": DEFAULT_SEED, "permutation_semantics": "target row index to source row index"},
    }
    write_json(output_dir / "genome_feature_schema.json", schema)

    metadata_records = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in metadata_paths]
    raw_manifest = {
        "manifest_version": "1.0.0", "generated_date": DOWNLOAD_DATE,
        "raw_cache_policy": "Large/raw public resources remain only in raw_cache and are not copied into reports.",
        "resources": resource_records,
        "excluded_or_unavailable": [
            {"name": "1011Matrix.gvcf.gz", "status": "not_downloaded", "reason": "5.4 GB source not required for this conservative V1; SNP/indel subfeatures remain not_available", "url": "http://1002genomes.u-strasbg.fr/files/1011Matrix.gvcf.gz", "upstream_md5": "42478e3e9dff4bd46993d82d8eab40d4"},
            {"name": "S288C_reference_genome_R64-1-1_20110203.tgz", "status": "not_cached", "reason": "bounded download was truncated and deleted; proxy uses cited assembly facts only", "url": "https://downloads.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-1-1_20110203.tgz"},
            {"name": "GCF_000146045.2_R64_genomic.fna.gz", "status": "not_cached", "reason": "bounded NCBI download timed out incomplete and was deleted; proxy uses cited assembly facts only", "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz"},
            {"name": "SGD go_slim_mapping.tab", "status": "not_cached", "reason": "bounded download was incomplete and deleted; GO features fail closed", "url": "https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab"},
        ],
        "rebuild_command": "python scripts/build_genome_features.py --metadata-train-val WAYB_WAYC/WAYB_WAYC_metadata_train_val(1).csv --metadata-test WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv --output-dir external_data/genome --raw-cache-dir external_data/genome/raw_cache --offline",
    }
    write_json(output_dir / "raw_resource_manifest.json", raw_manifest)
    source_manifest = {
        "manifest_version": "1.0.0", "generated_date": DOWNLOAD_DATE,
        "input_metadata": metadata_records,
        "sources": [
            {"name": "Peter et al. 2018", "url": "https://doi.org/10.1038/s41586-018-0030-5", "version": "Nature 556:339-344", "release_date": "2018-04-11", "license": "CC BY 4.0"},
            {"name": "1002 Yeast Genomes public files", "url": "http://1002genomes.u-strasbg.fr/files/", "version": "server index observed 2026-08-14", "release_date": "2018 dataset; server mtime varies", "license": "Raw-server terms not separately stated; paper is CC BY 4.0"},
            {"name": "NCBI S288C R64", "url": "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2", "version": "GCF_000146045.2 / SGD R64", "release_date": "2011-05-27", "license": "NCBI public data; see NCBI data policies"},
            {"name": "SGD", "url": "https://www.yeastgenome.org/", "version": "R64-5-1 annotation noted 2026-01-23; not used in V1", "release_date": "2026-01-23", "license": "SGD site/data terms must be reviewed by Main before redistribution"},
        ],
        "reference_and_processing": schema["coordinate_contract"],
        "processing_steps": [
            "extract unique Strains and frozen split coverage from metadata", "verify every cached public resource hash",
            "read Peter Tables S1/S8/S16/S17", "read selected rows/columns from public pangenome matrices",
            "decode selected strain rows from the PLINK bed matrix", "derive interpretable aggregate features",
            "separate four technical QC fields from the main biological feature matrix",
            "propagate missing denominators to dependent validity masks and force invalid final cells to zero",
            "fit medians/scaler/variance mask/PCA only on eligible training strains", "write deterministic CSV/JSON/NPZ artifacts",
        ],
        "raw_resource_manifest_sha256": "set_after_write",
        "genome_feature_schema_sha256": "set_after_write",
        "processing_script": str(Path(__file__).resolve()),
    }
    source_manifest["raw_resource_manifest_sha256"] = sha256_file(output_dir / "raw_resource_manifest.json")
    source_manifest["genome_feature_schema_sha256"] = sha256_file(output_dir / "genome_feature_schema.json")
    write_json(output_dir / "source_manifest.json", source_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
