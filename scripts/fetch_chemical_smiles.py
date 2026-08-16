"""Fetch CID + canonical SMILES for every competition compound from PubChem.

The competition metadata only gives compound *names*.  This script turns those
names into authoritative structures by querying PubChem's PUG REST API, using
the de-salted / de-abbreviated ``query_name`` from
``aivc.chemical_standardization``.

Two modes:

* ``python scripts/fetch_chemical_smiles.py --skeleton``
  writes ``data/external/chemical_mapping.csv`` with only the name
  standardization filled in and empty CID/SMILES columns (no network).  Use
  this to inspect the standardization first.

* ``python scripts/fetch_chemical_smiles.py``  (default)
  queries PubChem and fills CID, canonical/isomeric SMILES, formula and
  molecular weight.  Progress is written incrementally so a failed request does
  not lose earlier results.

Requirements: ``requests`` (already a project dependency).

Disclosure: ``source`` is set to ``PubChem PUG REST`` and ``source_version``
carries the fetch date so the mapping satisfies the competition's
"external resources must disclose source and version" rule.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Allow running as ``python scripts/fetch_chemical_smiles.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aivc.chemical_standardization import STANDARDIZATION, ENTITY_COMPOUND

PUBMED_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

PROPERTY_NAMES = ["CanonicalSMILES", "IsomericSMILES", "MolecularFormula", "MolecularWeight"]

CSV_COLUMNS = [
    "raw_name", "std_name", "entity_type", "parent_name", "query_name", "notes",
    "pubchem_cid", "canonical_smiles", "isomeric_smiles",
    "molecular_formula", "molecular_weight",
    "source", "source_version", "review_status",
]


def _encode(name: str) -> str:
    return name.replace(" ", "%20")


def resolve_cid(session: requests.Session, query_name: str) -> Optional[str]:
    """Return the first CID for a name, or ``None`` if lookup fails."""
    url = f"{PUBMED_BASE}/compound/name/{_encode(query_name)}/cids/JSON"
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        return None
    data = response.json()
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None
    return str(cids[0])


def resolve_properties(session: requests.Session, cid: str) -> Dict[str, str]:
    """Return the requested properties for a CID, empty on failure.

    PubChem currently aliases the requested ``CanonicalSMILES`` property to the
    ``ConnectivitySMILES`` key and ``IsomericSMILES`` to the ``SMILES`` key in
    the JSON response, so both spellings are read defensively.
    """
    url = (
        f"{PUBMED_BASE}/compound/cid/{cid}/property/"
        f"{','.join(PROPERTY_NAMES)}/JSON"
    )
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        return {}
    props = response.json().get("PropertyTable", {}).get("Properties", [{}])[0]
    return {
        "canonical_smiles": props.get("ConnectivitySMILES") or props.get("CanonicalSMILES", ""),
        "isomeric_smiles": props.get("SMILES") or props.get("IsomericSMILES", ""),
        "molecular_formula": props.get("MolecularFormula", ""),
        "molecular_weight": str(props.get("MolecularWeight", "")),
    }


def build_rows(fetch: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    source_version = _dt.date.today().isoformat()
    session = requests.Session()
    session.headers["User-Agent"] = "AIVC-chemical-mapping/1.0 (competition data prep)"

    for entry in STANDARDIZATION:
        row = {
            "raw_name": entry["raw_name"],
            "std_name": entry["std_name"],
            "entity_type": entry["entity_type"],
            "parent_name": entry.get("parent_name", ""),
            "query_name": entry.get("query_name", ""),
            "notes": entry.get("notes", ""),
            "pubchem_cid": "",
            "canonical_smiles": "",
            "isomeric_smiles": "",
            "molecular_formula": "",
            "molecular_weight": "",
            "source": "",
            "source_version": "",
            "review_status": "pending_fetch",
        }

        if not fetch or entry["entity_type"] != ENTITY_COMPOUND:
            if entry["entity_type"] != ENTITY_COMPOUND:
                row["review_status"] = "not_a_compound"
            rows.append(row)
            continue

        query = entry["query_name"]
        try:
            # A manually pinned CID (e.g. for mixtures whose name search fails)
            # takes precedence over name resolution.
            cid = entry.get("known_cid") or resolve_cid(session, query)
            if cid is None:
                row["review_status"] = "not_found"
            else:
                props = resolve_properties(session, cid)
                row["pubchem_cid"] = cid
                row["canonical_smiles"] = props.get("canonical_smiles", "")
                row["isomeric_smiles"] = props.get("isomeric_smiles", "")
                row["molecular_formula"] = props.get("molecular_formula", "")
                row["molecular_weight"] = props.get("molecular_weight", "")
                row["source"] = "PubChem PUG REST"
                row["source_version"] = source_version
                row["review_status"] = (
                    "auto_fetched" if row["canonical_smiles"] else "resolved_no_smiles"
                )
        except requests.RequestException as exc:
            row["review_status"] = f"error: {type(exc).__name__}"
        rows.append(row)

        # Be polite to the API; also lets the loop be interrupted safely.
        time.sleep(0.25)
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
    parser = argparse.ArgumentParser(description="Build chemical mapping from PubChem")
    parser.add_argument(
        "--skeleton", action="store_true",
        help="Write only the name-standardization skeleton (no network).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/external/chemical_mapping.csv"),
        help="Output CSV path (default: data/external/chemical_mapping.csv)",
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
