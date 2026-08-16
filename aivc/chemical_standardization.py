"""Compound name standardization for the AIVC external chemical features.

The competition only provides compound *names* (plus salt forms, hydrates,
abbreviations and solvent controls).  Before any SMILES-based feature can be
built, each raw name must be mapped to a single, unambiguous structure.  This
module holds that mapping as pure data plus a small validation helper, so it
can be imported by both the offline fetch script and the feature builder.

Each entry records:

* ``raw_name``   - the exact string seen in the metadata column
  ``perturbation_no_concentration``.
* ``std_name``   - a cleaned, conventional name.
* ``entity_type``- one of ``"compound"``, ``"control"``, ``"qc"``.
* ``parent_name``- for salts/hydrates, the de-counterion parent molecule.
* ``query_name`` - the preferred PubChem name lookup key.
* ``notes``      - the standardization action taken (salt stripped, abbreviation
  expanded, stereochemistry, mixture, etc.).

Nothing here depends on RDKit or network access; the SMILES/CID columns are
filled by ``scripts/fetch_chemical_smiles.py`` from PubChem and recorded in
``data/external/chemical_mapping.csv``.
"""
from __future__ import annotations

from typing import List, Dict

ENTITY_COMPOUND = "compound"
ENTITY_CONTROL = "control"
ENTITY_QC = "qc"


# Ordered exactly as the union of train/val + test metadata (57 unique names).
STANDARDIZATION: List[Dict[str, str]] = [
    # --- solvent / treatment controls ---
    {"raw_name": "Water", "std_name": "Water", "entity_type": ENTITY_CONTROL,
     "parent_name": "", "query_name": "", "notes": "溶剂对照"},
    {"raw_name": "DMSO", "std_name": "Dimethyl sulfoxide", "entity_type": ENTITY_CONTROL,
     "parent_name": "", "query_name": "Dimethyl sulfoxide", "notes": "溶剂对照"},
    {"raw_name": "Quality Control", "std_name": "Quality Control", "entity_type": ENTITY_QC,
     "parent_name": "", "query_name": "", "notes": "pert_id=48，非化合物，不计入化学扰动预测"},

    # --- salts / hydrates (de-counterion) ---
    {"raw_name": "Amiodarone hydrochloride", "std_name": "Amiodarone",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Amiodarone",
     "query_name": "Amiodarone", "notes": "盐酸盐，去盐"},
    {"raw_name": "Clomiphene citrate", "std_name": "Clomiphene",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Clomiphene",
     "query_name": "Clomiphene", "notes": "枸橼酸盐，去盐"},
    {"raw_name": "Desipramine hydrochloride", "std_name": "Desipramine",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Desipramine",
     "query_name": "Desipramine", "notes": "盐酸盐，去盐"},
    {"raw_name": "Doxycycline hyclate", "std_name": "Doxycycline",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Doxycycline",
     "query_name": "Doxycycline", "notes": "盐酸半乙醇半水合物(hyclate)，去盐"},
    {"raw_name": "Dyclonine hydrochloride", "std_name": "Dyclonine",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Dyclonine",
     "query_name": "Dyclonine", "notes": "盐酸盐，去盐"},
    {"raw_name": "Harmine hydrochloride", "std_name": "Harmine",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Harmine",
     "query_name": "Harmine", "notes": "盐酸盐，去盐"},
    {"raw_name": "LY 294002 hydrochloride", "std_name": "LY294002",
     "entity_type": ENTITY_COMPOUND, "parent_name": "LY294002",
     "query_name": "LY294002", "notes": "盐酸盐，去盐"},
    {"raw_name": "Nystatin dihydrate", "std_name": "Nystatin",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Nystatin",
     "query_name": "Nystatin", "notes": "二水合物，去水"},
    {"raw_name": "Pentamidine isethionate", "std_name": "Pentamidine",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Pentamidine",
     "query_name": "Pentamidine", "notes": "羟乙基磺酸盐，去盐"},
    {"raw_name": "Raloxifene hydrochloride", "std_name": "Raloxifene",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Raloxifene",
     "query_name": "Raloxifene", "notes": "盐酸盐，去盐"},
    {"raw_name": "Trifluoperazine dihydrochloride", "std_name": "Trifluoperazine",
     "entity_type": ENTITY_COMPOUND, "parent_name": "Trifluoperazine",
     "query_name": "Trifluoperazine", "notes": "二盐酸盐，去盐"},
    {"raw_name": "1-10 Phenanthroline monohydrate", "std_name": "1,10-Phenanthroline",
     "entity_type": ENTITY_COMPOUND, "parent_name": "1,10-Phenanthroline",
     "query_name": "1,10-Phenanthroline", "notes": "一水合物，去水"},

    # --- abbreviations (expanded to full name) ---
    {"raw_name": "CHX", "std_name": "Cycloheximide", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Cycloheximide", "notes": "缩写展开"},
    {"raw_name": "FCCP", "std_name": "Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone",
     "notes": "缩写展开"},
    {"raw_name": "SDS", "std_name": "Sodium dodecyl sulfate", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Sodium dodecyl sulfate", "notes": "缩写展开，去垢剂"},
    {"raw_name": "MMS", "std_name": "Methyl methanesulfonate", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Methyl methanesulfonate", "notes": "缩写展开"},
    {"raw_name": "G418", "std_name": "Geneticin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Geneticin", "notes": "缩写展开，氨基糖苷"},

    # --- simple / ionic molecules ---
    {"raw_name": "NaCl", "std_name": "Sodium chloride", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Sodium chloride", "notes": "离子化合物"},
    {"raw_name": "H2O2", "std_name": "Hydrogen peroxide", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Hydrogen peroxide", "notes": "简单分子"},
    {"raw_name": "EDTA", "std_name": "Edetic acid", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Edetic acid", "notes": "缩写展开(乙二胺四乙酸)"},
    {"raw_name": "Sorbitol", "std_name": "Sorbitol", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Sorbitol", "notes": "糖醇"},

    # --- stereochemistry-sensitive ---
    {"raw_name": "(1R, 2S, 5R) - (-) - Menthol", "std_name": "(-)-Menthol",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "(-)-Menthol", "notes": "立体异构，保留手性"},
    {"raw_name": "(S)-(+)-Camptothecin", "std_name": "(S)-(+)-Camptothecin",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "(S)-(+)-Camptothecin", "notes": "立体异构，保留手性"},
    {"raw_name": "4-Hydroxytamoxifen", "std_name": "4-Hydroxytamoxifen",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "4-Hydroxytamoxifen", "notes": ""},

    # --- standard small molecules / natural products (keep as-is) ---
    {"raw_name": "Abietic acid", "std_name": "Abietic acid", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Abietic acid", "notes": ""},
    {"raw_name": "Amphotericin B", "std_name": "Amphotericin B", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Amphotericin B", "notes": "复杂天然产物"},
    {"raw_name": "Anisomycin", "std_name": "Anisomycin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Anisomycin", "notes": ""},
    {"raw_name": "Artemisinin", "std_name": "Artemisinin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Artemisinin", "notes": ""},
    {"raw_name": "Brefeldin A", "std_name": "Brefeldin A", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Brefeldin A", "notes": ""},
    {"raw_name": "Cisplatin", "std_name": "Cisplatin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Cisplatin", "notes": "含铂"},
    {"raw_name": "Clotrimazole", "std_name": "Clotrimazole", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Clotrimazole", "notes": ""},
    {"raw_name": "Cyclopiazonic acid", "std_name": "Cyclopiazonic acid",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "Cyclopiazonic acid", "notes": ""},
    {"raw_name": "Emodin", "std_name": "Emodin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Emodin", "notes": ""},
    {"raw_name": "Fluconazole", "std_name": "Fluconazole", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Fluconazole", "notes": ""},
    {"raw_name": "Geldanamycin", "std_name": "Geldanamycin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Geldanamycin", "notes": "复杂天然产物"},
    {"raw_name": "Haloperidol", "std_name": "Haloperidol", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Haloperidol", "notes": ""},
    {"raw_name": "Hoechst 33258", "std_name": "Hoechst 33258", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Hoechst 33258", "notes": ""},
    {"raw_name": "Hydroxyurea", "std_name": "Hydroxyurea", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Hydroxyurea", "notes": ""},
    {"raw_name": "Hygromycin B", "std_name": "Hygromycin B", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Hygromycin B", "notes": "氨基糖苷"},
    {"raw_name": "Neomycin B", "std_name": "Neomycin B", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Neomycin B", "notes": "氨基糖苷，混合物"},
    {"raw_name": "Nigericin", "std_name": "Nigericin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Nigericin", "notes": "离子载体"},
    {"raw_name": "Nocodazole", "std_name": "Nocodazole", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Nocodazole", "notes": ""},
    {"raw_name": "Oligomycin", "std_name": "Oligomycin A", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Oligomycin A", "notes": "混合物，取 A 组分"},
    {"raw_name": "Parthenolide", "std_name": "Parthenolide", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Parthenolide", "notes": ""},
    {"raw_name": "Plumbagin", "std_name": "Plumbagin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Plumbagin", "notes": ""},
    {"raw_name": "Rapamycin", "std_name": "Sirolimus", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Sirolimus", "notes": "Rapamycin=Sirolimus"},
    {"raw_name": "Staurosporine", "std_name": "Staurosporine", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Staurosporine", "notes": ""},
    {"raw_name": "Sulfometuron methyl", "std_name": "Sulfometuron methyl",
     "entity_type": ENTITY_COMPOUND, "parent_name": "",
     "query_name": "Sulfometuron methyl", "notes": ""},
    {"raw_name": "Tamoxifen", "std_name": "Tamoxifen", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Tamoxifen", "notes": ""},
    {"raw_name": "Trichostatin A", "std_name": "Trichostatin A", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Trichostatin A", "notes": ""},
    {"raw_name": "Tunicamycin", "std_name": "Tunicamycin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Tunicamycin", "known_cid": "6433557",
     "notes": "复杂天然产物，混合物；以 A1 主组分(CID 6433557)代表"},
    {"raw_name": "U-73122", "std_name": "U-73122", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "U-73122", "notes": ""},
    {"raw_name": "Valinomycin", "std_name": "Valinomycin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Valinomycin", "notes": "环肽"},
    {"raw_name": "Wortmannin", "std_name": "Wortmannin", "entity_type": ENTITY_COMPOUND,
     "parent_name": "", "query_name": "Wortmannin", "notes": ""},
]


def as_lookup() -> Dict[str, Dict[str, str]]:
    """Return ``{raw_name: entry}`` for fast lookup by metadata value."""
    lookup: Dict[str, Dict[str, str]] = {}
    for entry in STANDARDIZATION:
        if entry["raw_name"] in lookup:
            raise ValueError(f"Duplicate raw_name in STANDARDIZATION: {entry['raw_name']}")
        lookup[entry["raw_name"]] = entry
    return lookup


def validate_coverage(raw_names) -> List[str]:
    """Return the subset of ``raw_names`` missing from STANDARDIZATION."""
    known = {entry["raw_name"] for entry in STANDARDIZATION}
    return [name for name in raw_names if name not in known]


if __name__ == "__main__":
    from collections import Counter

    types = Counter(entry["entity_type"] for entry in STANDARDIZATION)
    print(f"STANDARDIZATION entries: {len(STANDARDIZATION)}")
    print(f"entity_type counts: {dict(types)}")
    duplicates = [n for n, c in Counter(e['raw_name'] for e in STANDARDIZATION).items() if c > 1]
    print(f"duplicate raw_names: {duplicates or 'none'}")
    empty_query = [e["raw_name"] for e in STANDARDIZATION
                   if e["entity_type"] == ENTITY_COMPOUND and not e["query_name"]]
    print(f"compounds missing query_name: {empty_query or 'none'}")
