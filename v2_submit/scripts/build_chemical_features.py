#!/usr/bin/env python
"""Build auditable PubChem/RDKit features from competition metadata only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

SPLIT_ORDER = [
    "train", "val_chem_only", "val_strain_only", "val_both", "val_time",
    "test_chem_only", "test_strain_only", "test_both", "test_time",
]
STATUSES = ["confirmed", "proxy", "special_control", "unresolved"]
CONFIDENCES = ["high", "medium", "low", "none"]
SPECIAL = {"water": "water", "dmso": "dmso", "quality control": "quality_control"}
ALIASES = {
    "(1R, 2S, 5R) - (-) - Menthol": ["(-)-menthol", "L-menthol"],
    "CHX": ["cycloheximide"],
    "FCCP": ["carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone"],
    "G418": ["geneticin"],
    "H2O2": ["hydrogen peroxide"],
    "MMS": ["methyl methanesulfonate"],
    "SDS": ["sodium dodecyl sulfate"],
    "1-10 Phenanthroline monohydrate": ["1,10-phenanthroline monohydrate", "1,10-phenanthroline"],
    "U-73122": ["U73122"],
    "LY 294002 hydrochloride": ["LY294002 hydrochloride"],
    "Oligomycin": ["oligomycin A"],
    "Tunicamycin": ["tunicamycin V"],
}
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_DOC = "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_CHIRALITY = True


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def normalize_name(value: str) -> str:
    value = value.strip().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*-\s*", "-", value)
    return value


def cache_key(kind: str, key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")[:55] or "query"
    return f"{kind}_{safe}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}.json"


@dataclass
class CachedResponse:
    payload: dict[str, Any]
    sha256: str
    url: str
    retrieved_date: str
    cache_path: Path
    cache_sha256: str


class PubChemClient:
    def __init__(self, cache_dir: Path, offline: bool, timeout: float, retries: int, min_interval: float):
        self.cache_dir = cache_dir
        self.offline = offline
        self.timeout = timeout
        self.retries = retries
        self.min_interval = max(min_interval, 0.2)  # PubChem asks for <=5 requests/s.
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GOAI-virtual-cell-chemistry/1.0 (auditable academic workflow)"})
        self.records: list[dict[str, Any]] = []

    def get(self, kind: str, key: str, url: str) -> CachedResponse:
        path = self.cache_dir / cache_key(kind, key)
        if path.exists():
            raw = path.read_bytes()
            envelope = json.loads(raw)
            response = CachedResponse(envelope["payload"], envelope["response_body_sha256"], envelope["url"], envelope["retrieved_date"], path, sha256_bytes(raw))
            self._record(response, True)
            return response
        if self.offline:
            raise FileNotFoundError(f"offline cache miss: {path.name}")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                self.last_request = time.monotonic()
                result = self.session.get(url, timeout=self.timeout)
                if result.status_code == 404:
                    payload: dict[str, Any] = {"Fault": result.text}
                else:
                    result.raise_for_status()
                    payload = result.json()
                envelope = {"url": url, "retrieved_date": date.today().isoformat(), "http_status": result.status_code, "response_body_sha256": sha256_bytes(result.content), "payload": payload}
                raw = canonical_json(envelope)
                path.write_bytes(raw)
                response = CachedResponse(payload, envelope["response_body_sha256"], url, envelope["retrieved_date"], path, sha256_bytes(raw))
                self._record(response, False)
                return response
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"PubChem request failed after retries: {url}: {last_error}")

    def _record(self, response: CachedResponse, cache_hit: bool) -> None:
        self.records.append({
            "url": response.url,
            "retrieved_date": response.retrieved_date,
            "cache_file": response.cache_path.name,
            "response_body_sha256": response.sha256,
            "cache_file_sha256": response.cache_sha256,
            "cache_hit": cache_hit,
        })

    def cids_for_name(self, alias: str) -> CachedResponse:
        url = f"{PUBCHEM_BASE}/compound/name/{quote(alias, safe='')}/cids/JSON"
        return self.get("name", alias, url)

    def properties(self, cid: str) -> CachedResponse:
        fields = "ConnectivitySMILES,SMILES,InChIKey,MolecularFormula,IUPACName,Title"
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{fields}/JSON"
        return self.get("properties", cid, url)

    def synonyms(self, cid: str) -> CachedResponse:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        return self.get("synonyms", cid, url)


def extract_entities(paths: list[Path]) -> pd.DataFrame:
    required = {"perturbation_no_concentration", "pert_id", "chemical_role", "split_final"}
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        frames.append(frame[list(required)].copy())
    data = pd.concat(frames, ignore_index=True)
    unexpected = sorted(set(data["split_final"]) - set(SPLIT_ORDER))
    if unexpected:
        raise ValueError(f"unexpected split_final values: {unexpected}")
    rows = []
    for raw_name, group in data.groupby("perturbation_no_concentration", sort=True):
        splits = [s for s in SPLIT_ORDER if s in set(group["split_final"])]
        rows.append({
            "chemical_id": raw_name,
            "raw_name": raw_name,
            "normalized_name": normalize_name(raw_name),
            "pert_ids": "|".join(sorted(set(group["pert_id"]))),
            "chemical_roles": "|".join(sorted(set(group["chemical_role"]))),
            "seen_splits": "|".join(splits),
            "first_seen_split": splits[0],
            "seen_in_train": "train" in splits,
            "seen_in_validation": any(x.startswith("val_") for x in splits),
            "seen_in_test": any(x.startswith("test_") for x in splits),
            "n_metadata_rows": len(group),
        })
    return pd.DataFrame(rows).sort_values("raw_name", kind="stable").reset_index(drop=True)


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"raw_name", "selected_cid", "mapping_status", "mapping_confidence", "review_status", "decision_reason", "parent_action", "evidence_url", "review_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"override file missing columns: {sorted(missing)}")
    if frame["raw_name"].duplicated().any():
        raise ValueError("override file contains duplicate raw_name")
    bad = sorted(set(frame["mapping_status"]) - set(STATUSES))
    if bad:
        raise ValueError(f"invalid override mapping_status: {bad}")
    return {row["raw_name"]: row for row in frame.to_dict("records")}


def candidate_ids(payload: dict[str, Any]) -> list[str]:
    values = payload.get("IdentifierList", {}).get("CID", [])
    return [str(x) for x in values]


def first_property(payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("PropertyTable", {}).get("Properties", [])
    return values[0] if values else {}


def synonym_values(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("InformationList", {}).get("Information", [])
    return list(rows[0].get("Synonym", [])) if rows else []


def discover(entities: pd.DataFrame, client: PubChemClient, overrides: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_rows, review_rows = [], []
    for entity in entities.to_dict("records"):
        raw = entity["raw_name"]
        special_type = SPECIAL.get(raw.casefold(), "")
        if special_type:
            row = dict(entity)
            row.update({
                "query_name": "", "used_alias": "", "match_source": "special_control_rule",
                "candidate_cids": "", "pubchem_cid": "", "pubchem_query_endpoint": "",
                "query_date": "", "raw_response_sha256": "", "pubchem_smiles_field": "",
                "canonical_smiles": "", "isomeric_smiles": "", "inchikey": "",
                "molecular_formula": "", "pubchem_title": "", "mapping_status": "special_control",
                "mapping_confidence": "none", "special_control_type": special_type,
                "cas_number": "",
                "structure_issue": "special control; model chemistry feature intentionally disabled",
                "decision_reason": "Frozen task rule; not encoded as an ordinary perturbation molecule.",
            })
            mapping_rows.append(row)
            continue

        aliases = [entity["normalized_name"], *ALIASES.get(raw, [])]
        aliases = list(dict.fromkeys(x for x in aliases if x))
        candidates: dict[str, set[str]] = {}
        name_responses: list[CachedResponse] = []
        for alias in aliases:
            response = client.cids_for_name(alias)
            name_responses.append(response)
            for cid in candidate_ids(response.payload):
                candidates.setdefault(cid, set()).add(alias)

        override = overrides.get(raw, {})
        selected = override.get("selected_cid", "")
        if selected and selected not in candidates:
            # A reviewed authoritative CID may correct an ambiguous name endpoint.
            candidates[selected] = {"manual_authoritative_cid"}
        status = override.get("mapping_status", "unresolved")
        confidence = override.get("mapping_confidence", "none")
        reason = override.get("decision_reason", "No approved manual selection; candidate retained for review.")
        if status in {"confirmed", "proxy"} and not selected:
            raise ValueError(f"{raw!r}: {status} override requires selected_cid")

        prop: dict[str, Any] = {}
        prop_response: CachedResponse | None = None
        syn_response: CachedResponse | None = None
        used_alias = ""
        synonyms: list[str] = []
        candidate_summaries: list[dict[str, str]] = []
        candidate_property_responses: dict[str, CachedResponse] = {}
        candidate_synonym_responses: dict[str, CachedResponse] = {}
        for cid in sorted(candidates, key=lambda x: int(x)):
            candidate_property_responses[cid] = client.properties(cid)
            candidate_synonym_responses[cid] = client.synonyms(cid)
            candidate_prop = first_property(candidate_property_responses[cid].payload)
            candidate_summaries.append({
                "cid": cid,
                "title": str(candidate_prop.get("Title", candidate_prop.get("IUPACName", ""))),
                "formula": str(candidate_prop.get("MolecularFormula", "")),
                "connectivity_smiles": str(candidate_prop.get("ConnectivitySMILES", candidate_prop.get("CanonicalSMILES", ""))),
                "isomeric_smiles": str(candidate_prop.get("SMILES", candidate_prop.get("IsomericSMILES", ""))),
                "matched_aliases": sorted(candidates[cid]),
            })
        if selected:
            selected_aliases = candidates[selected]
            query_aliases = [alias for alias in aliases if alias in selected_aliases]
            used_alias = query_aliases[0] if query_aliases else "manual_authoritative_cid"
            prop_response = candidate_property_responses[selected]
            syn_response = candidate_synonym_responses[selected]
            prop = first_property(prop_response.payload)
            synonyms = synonym_values(syn_response.payload)
        smiles_field = "ConnectivitySMILES" if "ConnectivitySMILES" in prop else ("CanonicalSMILES" if "CanonicalSMILES" in prop else "")
        canonical_smiles = str(prop.get(smiles_field, ""))
        isomeric_field = "SMILES" if "SMILES" in prop else ("IsomericSMILES" if "IsomericSMILES" in prop else "")
        isomeric_smiles = str(prop.get(isomeric_field, ""))
        issue_bits = []
        if len(candidates) > 1: issue_bits.append("multiple_candidate_cids")
        if "." in (isomeric_smiles or canonical_smiles): issue_bits.append("multicomponent_or_salt")
        if "@" in isomeric_smiles or re.search(r"(^|\W)[RSEZ][,)-]", raw): issue_bits.append("stereochemistry")
        if selected and raw.casefold() not in {s.casefold() for s in synonyms} and normalize_name(raw).casefold() not in {normalize_name(s).casefold() for s in synonyms}:
            issue_bits.append("name_not_exact_synonym")
        if not candidates: issue_bits.append("no_pubchem_result")
        if status == "proxy": issue_bits.append("proxy_structure")
        if override.get("parent_action", "") == "identity_only":
            issue_bits.append("身份已确认，但当前 Morgan/RDKit 特征体系不适用")
        hashes = [x.sha256 for x in name_responses]
        hashes.extend(candidate_property_responses[cid].sha256 for cid in sorted(candidate_property_responses, key=lambda x: int(x)))
        hashes.extend(candidate_synonym_responses[cid].sha256 for cid in sorted(candidate_synonym_responses, key=lambda x: int(x)))
        row = dict(entity)
        row.update({
            "query_name": entity["normalized_name"], "used_alias": used_alias,
            "match_source": ("manual_authoritative_cid+pubchem_properties_synonyms" if used_alias == "manual_authoritative_cid" else "manual_override+pubchem_name_synonym") if selected else "pubchem_name_candidates",
            "candidate_cids": "|".join(sorted(candidates, key=lambda x: int(x))), "pubchem_cid": selected,
            "pubchem_query_endpoint": prop_response.url if prop_response else "|".join(x.url for x in name_responses),
            "query_date": max((x.retrieved_date for x in name_responses + ([prop_response] if prop_response else []) + ([syn_response] if syn_response else [])), default=""),
            "raw_response_sha256": "|".join(hashes), "pubchem_smiles_field": f"{smiles_field}|{isomeric_field}" if selected else "",
            "canonical_smiles": canonical_smiles, "isomeric_smiles": isomeric_smiles,
            "inchikey": str(prop.get("InChIKey", "")), "molecular_formula": str(prop.get("MolecularFormula", "")),
            "pubchem_title": str(prop.get("Title", prop.get("IUPACName", ""))), "mapping_status": status,
            "mapping_confidence": confidence, "special_control_type": "",
            "cas_number": override.get("cas_number", ""),
            "structure_issue": "|".join(issue_bits), "decision_reason": reason,
        })
        mapping_rows.append(row)
        review_rows.append({
            "raw_name": raw, "normalized_name": entity["normalized_name"],
            "candidate_cids": row["candidate_cids"], "selected_cid": selected,
            "review_status": override.get("review_status", "pending"), "mapping_status": status,
            "decision_reason": reason, "structure_issue": row["structure_issue"],
            "evidence_url": override.get("evidence_url", prop_response.url if prop_response else row["pubchem_query_endpoint"]),
            "review_date": override.get("review_date", ""),
            "cas_number": override.get("cas_number", ""),
            "candidate_summaries_json": json.dumps(candidate_summaries, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        })
    return pd.DataFrame(mapping_rows), pd.DataFrame(review_rows)


def standardize_structures(mapping: pd.DataFrame, overrides: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, list[Chem.Mol | None]]:
    output = mapping.copy()
    parents: list[Chem.Mol | None] = []
    for idx, row in output.iterrows():
        raw_smiles = row["isomeric_smiles"] or row["canonical_smiles"]
        output.loc[idx, "raw_structure_smiles"] = raw_smiles
        output.loc[idx, "raw_structure_inchikey"] = row["inchikey"]
        output.loc[idx, "parent_smiles"] = ""
        output.loc[idx, "parent_inchikey"] = ""
        output.loc[idx, "standardization_operations"] = ""
        output.loc[idx, "structure_valid"] = False
        if row["mapping_status"] not in {"confirmed", "proxy"}:
            parents.append(None); continue
        mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
        if mol is None:
            output.loc[idx, "mapping_status"] = "unresolved"
            output.loc[idx, "mapping_confidence"] = "none"
            output.loc[idx, "decision_reason"] = str(row["decision_reason"]) + " RDKit could not parse selected structure."
            parents.append(None); continue
        action = overrides.get(row["raw_name"], {}).get("parent_action", "cleanup")
        if action == "identity_only":
            output.loc[idx, "standardization_operations"] = "identity confirmed; full PubChem structure retained; RDKit/Morgan feature generation disabled"
            output.loc[idx, "structure_valid"] = False
            parents.append(None); continue
        operations = ["Chem.MolFromSmiles(sanitize=True)"]
        parent = rdMolStandardize.Cleanup(mol)
        operations.append("rdMolStandardize.Cleanup(default parameters)")
        if len(Chem.GetMolFrags(parent)) > 1:
            if action != "fragment_parent":
                output.loc[idx, "mapping_status"] = "unresolved"
                output.loc[idx, "mapping_confidence"] = "none"
                output.loc[idx, "decision_reason"] = str(row["decision_reason"]) + " Multicomponent structure lacks approved fragment_parent action."
                parents.append(None); continue
            parent = rdMolStandardize.FragmentParent(parent)
            operations.append("rdMolStandardize.FragmentParent(default parameters; manually approved)")
        elif action not in {"cleanup", "fragment_parent"}:
            raise ValueError(f"unsupported parent_action {action!r} for {row['raw_name']!r}")
        try:
            Chem.SanitizeMol(parent)
            Chem.GetSymmSSSR(parent)
        except Exception as exc:
            output.loc[idx, "mapping_status"] = "unresolved"
            output.loc[idx, "mapping_confidence"] = "none"
            output.loc[idx, "decision_reason"] = str(row["decision_reason"]) + f" RDKit parent sanitization failed: {exc}"
            parents.append(None); continue
        operations.append("Chem.SanitizeMol; Chem.GetSymmSSSR")
        parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        output.loc[idx, "parent_smiles"] = parent_smiles
        output.loc[idx, "parent_inchikey"] = Chem.MolToInchiKey(parent)
        output.loc[idx, "standardization_operations"] = "; ".join(operations)
        output.loc[idx, "structure_valid"] = True
        parents.append(parent)
    return output, parents


def generate_features(mapping: pd.DataFrame, parents: list[Chem.Mol | None]) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    descriptor_names = [name for name, _ in Descriptors._descList]
    descriptor_functions = [func for _, func in Descriptors._descList]
    n, d = len(mapping), len(descriptor_names)
    morgan = np.zeros((n, MORGAN_BITS), dtype=np.float32)
    descriptors = np.zeros((n, d), dtype=np.float64)
    valid = np.zeros((n, MORGAN_BITS + d), dtype=bool)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=MORGAN_CHIRALITY)
    for i, mol in enumerate(parents):
        if mol is None: continue
        try:
            fp = generator.GetFingerprint(mol)
        except Exception as exc:
            raise RuntimeError(f"Morgan generation failed for {mapping.iloc[i]['raw_name']}: {exc}") from exc
        DataStructs.ConvertToNumpyArray(fp, morgan[i])
        vals = np.asarray([func(mol) for func in descriptor_functions], dtype=np.float64)
        if not np.isfinite(vals).all():
            bad = [descriptor_names[j] for j in np.where(~np.isfinite(vals))[0]]
            raise ValueError(f"non-finite descriptors for {mapping.iloc[i]['raw_name']}: {bad}")
        descriptors[i] = vals
        valid[i] = True
    fit_mask = (
        mapping["seen_in_train"].astype(bool).to_numpy()
        & mapping["mapping_status"].eq("confirmed").to_numpy()
        & mapping["structure_valid"].astype(bool).to_numpy()
        & mapping["special_control_type"].eq("").to_numpy()
    )
    if not fit_mask.any():
        raise ValueError("no confirmed ordinary train chemicals available to fit descriptor scaler")
    mean = descriptors[fit_mask].mean(axis=0)
    scale_raw = descriptors[fit_mask].std(axis=0, ddof=0)
    zero_variance = scale_raw == 0
    scale = scale_raw.copy(); scale[zero_variance] = 1.0
    descriptor_z = np.zeros_like(descriptors, dtype=np.float32)
    structure_valid = mapping["structure_valid"].astype(bool).to_numpy()
    descriptor_z[structure_valid] = ((descriptors[structure_valid] - mean) / scale).astype(np.float32)
    features = np.concatenate([morgan, descriptor_z], axis=1).astype(np.float32)
    if not np.isfinite(features).all(): raise ValueError("chemical feature matrix contains NA/inf")
    status_codes = np.asarray([STATUSES.index(x) for x in mapping["mapping_status"]], dtype=np.int8)
    confidence_codes = np.asarray([CONFIDENCES.index(x) for x in mapping["mapping_confidence"]], dtype=np.int8)
    arrays = {
        "features": features, "morgan_fp": morgan, "rdkit_descriptors_raw": descriptors.astype(np.float32),
        "rdkit_descriptors_standardized": descriptor_z, "feature_valid_mask": valid,
        "structure_valid": structure_valid, "mapping_status_code": status_codes,
        "mapping_confidence_code": confidence_codes, "descriptor_mean": mean.astype(np.float64),
        "descriptor_scale": scale.astype(np.float64), "scaler_fit_mask": fit_mask,
    }
    schema = {
        "schema_version": "1.0", "rdkit_version": rdBase.rdkitVersion,
        "morgan": {"radius": MORGAN_RADIUS, "n_bits": MORGAN_BITS, "use_chirality": MORGAN_CHIRALITY, "dtype": "float32", "binary": True},
        "descriptors": {"names": descriptor_names, "count": d, "order_source": "rdkit.Chem.Descriptors._descList", "standardization": "z-score; ddof=0; train confirmed ordinary chemicals only", "zero_variance_descriptors": [descriptor_names[i] for i in np.where(zero_variance)[0]]},
        "feature_concatenation_order": ["morgan_fp[0:2048]", "rdkit_descriptors_standardized[in listed order]"],
        "chemical_feature_dim": MORGAN_BITS + d, "mapping_status_values": STATUSES,
        "mapping_confidence_values": CONFIDENCES,
        "scaler_fit_chemical_names": mapping.loc[fit_mask, "raw_name"].tolist(),
        "fallback": "special_control, unresolved, and identity-confirmed but structure-invalid entities use all-zero numeric features and feature_valid_mask=false",
    }
    abs_z = np.abs(descriptor_z[structure_valid].astype(np.float64))
    extreme_cells = []
    if abs_z.size:
        flat_order = np.argsort(abs_z, axis=None)[::-1][:20]
        valid_rows = np.where(structure_valid)[0]
        for flat in flat_order:
            local_row, descriptor_idx = np.unravel_index(flat, abs_z.shape)
            entity_idx = valid_rows[local_row]
            extreme_cells.append({
                "raw_name": mapping.iloc[entity_idx]["raw_name"],
                "descriptor": descriptor_names[descriptor_idx],
                "standardized_value": float(descriptor_z[entity_idx, descriptor_idx]),
                "absolute_standardized_value": float(abs_z[local_row, descriptor_idx]),
            })
    schema["descriptor_extreme_value_diagnostics"] = {
        "scope": "all structure-valid entities transformed by the frozen train-only scaler",
        "absolute_value_quantiles": {str(q): float(np.quantile(abs_z, q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]} if abs_z.size else {},
        "count_abs_gt_5": int((abs_z > 5).sum()),
        "count_abs_gt_10": int((abs_z > 10).sum()),
        "top_extreme_cells": extreme_cells,
        "recommendation_only": "Main may evaluate train-only clipping bounds or a train-only robust scaler; frozen features are unchanged in this revision.",
    }
    schema["descriptor_order_sha256"] = sha256_bytes("\n".join(descriptor_names).encode("utf-8"))
    schema["entity_order_sha256"] = sha256_bytes("\n".join(mapping["chemical_id"]).encode("utf-8"))
    scaler_payload = {"mean": mean.tolist(), "scale": scale.tolist(), "fit_names": schema["scaler_fit_chemical_names"]}
    schema["scaler_sha256"] = sha256_bytes(canonical_json(scaler_payload))
    return arrays, schema, descriptor_names


def generate_similarity(mapping: pd.DataFrame, parents: list[Chem.Mol | None]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    ordinary = mapping["mapping_status"].isin(["confirmed", "proxy"]) & mapping["structure_valid"].astype(bool) & mapping["special_control_type"].eq("")
    rows = ordinary & mapping["seen_in_train"].astype(bool)
    row_idx = np.where(rows.to_numpy())[0]
    col_idx = np.where(ordinary.to_numpy())[0]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=MORGAN_CHIRALITY)
    fps = {i: generator.GetFingerprint(parents[i]) for i in set(row_idx) | set(col_idx)}
    sim = np.zeros((len(row_idx), len(col_idx)), dtype=np.float32)
    for r, i in enumerate(row_idx):
        sim[r] = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in col_idx]), dtype=np.float32)
    if sim.size and ((sim < 0).any() or (sim > 1).any()): raise ValueError("Tanimoto out of range")
    row_frame = mapping.iloc[row_idx][["chemical_id", "raw_name", "parent_inchikey"]].reset_index(drop=True)
    col_frame = mapping.iloc[col_idx][["chemical_id", "raw_name", "parent_inchikey"]].reset_index(drop=True)
    for r, name in enumerate(row_frame["chemical_id"]):
        matches = np.where(col_frame["chemical_id"].to_numpy() == name)[0]
        if len(matches) != 1 or not math.isclose(float(sim[r, matches[0]]), 1.0, abs_tol=1e-7):
            raise ValueError(f"missing Tanimoto self-similarity for {name}")
    return sim, row_frame, col_frame


def apply_feature_variant(
    features: np.ndarray,
    mode: str = "correct",
    seed: int = 20260813,
    *,
    mapping_status: np.ndarray | pd.Series | list[str] | None = None,
    structure_valid: np.ndarray | pd.Series | list[bool] | None = None,
    special_control_type: np.ndarray | pd.Series | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a feature variant and auditable target-row -> source-row permutation.

    Shuffle is restricted to confirmed/proxy, structure-valid, ordinary entities.
    Every ineligible row remains at its original index with its original features.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if mode == "correct":
        order = np.arange(len(features))
        return features.copy(), order
    if mode == "zero":
        order = np.arange(len(features))
        return np.zeros_like(features), order
    if mode == "shuffle":
        if mapping_status is None or structure_valid is None or special_control_type is None:
            raise ValueError("shuffle requires mapping_status, structure_valid, and special_control_type")
        status = np.asarray(mapping_status, dtype=str)
        valid = np.asarray(structure_valid, dtype=bool)
        special = np.asarray(special_control_type, dtype=str)
        if not (len(status) == len(valid) == len(special) == len(features)):
            raise ValueError("shuffle eligibility metadata must match feature rows")
        eligible = np.isin(status, ["confirmed", "proxy"]) & valid & (special == "")
        eligible_idx = np.flatnonzero(eligible)
        order = np.arange(len(features))
        order[eligible_idx] = np.random.default_rng(seed).permutation(eligible_idx)
        return features[order].copy(), order
    raise ValueError("mode must be one of: correct, shuffle, zero")


def save_outputs(output: Path, mapping: pd.DataFrame, reviews: pd.DataFrame, arrays: dict[str, np.ndarray], schema: dict[str, Any], similarity: np.ndarray, row_idx: pd.DataFrame, col_idx: pd.DataFrame, client: PubChemClient, metadata_paths: list[Path], override_path: Path) -> None:
    final_status = mapping.set_index("raw_name")[["mapping_status", "decision_reason", "structure_issue"]]
    for idx, review in reviews.iterrows():
        if review["raw_name"] in final_status.index:
            reviews.loc[idx, ["mapping_status", "decision_reason", "structure_issue"]] = final_status.loc[review["raw_name"]].to_list()
    mapping.to_csv(output / "compound_mapping.csv", index=False, lineterminator="\n")
    reviews.to_csv(output / "manual_review.csv", index=False, lineterminator="\n")
    np.savez_compressed(output / "chemical_features.npz", **arrays)
    write_json(output / "feature_schema.json", schema)
    mapping[["chemical_id", "raw_name", "mapping_status", "mapping_confidence", "structure_valid", "parent_inchikey"]].to_csv(output / "chemical_feature_index.csv", index=False, lineterminator="\n")
    np.savez_compressed(output / "tanimoto_similarity.npz", tanimoto=similarity)
    row_idx.to_csv(output / "tanimoto_row_index.csv", index=False, lineterminator="\n")
    col_idx.to_csv(output / "tanimoto_column_index.csv", index=False, lineterminator="\n")
    files = []
    reference_paths = [
        Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tests" / "test_chemical_features.py",
        Path(__file__).resolve().parents[1] / "references" / "20260812Approach.pdf",
        Path(__file__).resolve().parents[1] / "references" / "OfficialRules.pdf",
        Path(__file__).resolve().parents[1] / "reports" / "CHEMICAL_FEATURE_REPORT.md",
        Path(__file__).resolve().parents[1] / "external_data" / "chemistry" / "identity_evidence.json",
    ]
    for path in [*metadata_paths, override_path, *reference_paths]:
        if path.exists():
            files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "role": "input_or_reference"})
    for name in ["compound_mapping.csv", "manual_review.csv", "feature_schema.json", "chemical_features.npz", "chemical_feature_index.csv", "tanimoto_similarity.npz", "tanimoto_row_index.csv", "tanimoto_column_index.csv"]:
        path = output / name
        files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "role": "derived_artifact"})
    manifest = {
        "manifest_version": "1.0", "sources": [
            {"name": "PubChem PUG-REST", "url": PUBCHEM_DOC, "accessed_dates": sorted({x["retrieved_date"] for x in client.records}), "version": "dynamic service; per-response cache and hash recorded", "license": "PubChem data are provided by NCBI; see https://pubchem.ncbi.nlm.nih.gov/docs/copyright"},
            {"name": "RDKit", "version": rdBase.rdkitVersion, "url": "https://www.rdkit.org/", "license": "BSD-3-Clause"},
            {"name": "Datawhale optimization tutorial", "url": "https://ailc.datawhale.cn/hall/group/100001094/task/100001302", "accessed_dates": [date.today().isoformat()], "version": "dynamic web page; automated retrieval unavailable during this run", "license": "not stated/blocked; used as a contextual reference only, not as data"},
            {"name": "GOAI virtual-cell approach PDF", "url": "local competition reference: references/20260812Approach.pdf", "accessed_dates": [date.today().isoformat()], "version": "local snapshot", "license": "not stated; competition reference only; not redistributed"},
            {"name": "GOAI OfficialRules planning PDF", "url": "local competition reference: references/OfficialRules.pdf", "accessed_dates": [date.today().isoformat()], "version": "2026-07 planning version; final organizer release controls", "license": "not stated; competition reference only; not redistributed"},
            {"name": "GOAI competition metadata", "url": "local competition input under WAYB_WAYC", "accessed_dates": [date.today().isoformat()], "version": "local provided snapshot", "license": "competition-restricted; not uploaded or redistributed"},
            {"name": "NLM MeSH U-73122 record", "url": "https://meshb.nlm.nih.gov/record/ui?name=U+73122", "accessed_dates": [date.today().isoformat()], "version": "MeSH Supplementary Concept Data 2026", "license": "NLM web policies: https://www.nlm.nih.gov/web_policies.html; evidence citation only"},
            {"name": "Tocris U 73122 product 1268", "url": "https://www.tocris.com/products/u-73122_1268", "accessed_dates": [date.today().isoformat()], "version": "dynamic product page", "license": "Copyright R&D Systems/Bio-Techne; evidence citation only; site Terms & Conditions apply"},
        ],
        "input_files": files, "requests": sorted(client.records, key=lambda x: (x["url"], x["cache_file"])),
        "processing_steps": ["metadata-only entity extraction", "cached PubChem name/CID/property/synonym lookup", "manual override selection", "RDKit cleanup and explicitly approved fragment-parent standardization", "Morgan and descriptor generation", "train-only descriptor scaler", "Tanimoto matrix"],
        "split_priority": SPLIT_ORDER,
    }
    write_json(output / "source_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-train-val", type=Path, required=True)
    parser.add_argument("--metadata-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Use existing small response caches only.")
    parser.add_argument("--discovery-only", action="store_true", help="Write mapping/review candidates without requiring a feature-ready override set.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-request-interval", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "pubchem_cache"; cache_dir.mkdir(exist_ok=True)
    metadata_paths = [args.metadata_train_val.resolve(), args.metadata_test.resolve()]
    entities = extract_entities(metadata_paths)
    overrides = load_overrides(args.overrides.resolve())
    client = PubChemClient(cache_dir, args.offline, args.timeout, args.retries, args.min_request_interval)
    mapping, reviews = discover(entities, client, overrides)
    if args.discovery_only:
        mapping.to_csv(args.output_dir / "compound_mapping.csv", index=False, lineterminator="\n")
        reviews.to_csv(args.output_dir / "manual_review.csv", index=False, lineterminator="\n")
        print(json.dumps({"entities": len(mapping), "special_controls": int(mapping.mapping_status.eq('special_control').sum()), "unresolved": int(mapping.mapping_status.eq('unresolved').sum()), "mode": "discovery-only"}))
        return 0
    mapping, parents = standardize_structures(mapping, overrides)
    arrays, schema, _ = generate_features(mapping, parents)
    similarity, row_idx, col_idx = generate_similarity(mapping, parents)
    save_outputs(args.output_dir, mapping, reviews, arrays, schema, similarity, row_idx, col_idx, client, metadata_paths, args.overrides.resolve())
    print(json.dumps({"entities": len(mapping), "confirmed": int(mapping.mapping_status.eq('confirmed').sum()), "proxy": int(mapping.mapping_status.eq('proxy').sum()), "special_controls": int(mapping.mapping_status.eq('special_control').sum()), "unresolved": int(mapping.mapping_status.eq('unresolved').sum()), "feature_dim": schema["chemical_feature_dim"], "tanimoto_shape": list(similarity.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
