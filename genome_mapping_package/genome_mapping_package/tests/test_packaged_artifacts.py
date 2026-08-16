from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "load_strain_features.py"
spec = importlib.util.spec_from_file_location("load_strain_features", SCRIPT)
loader = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = loader
spec.loader.exec_module(loader)


def test_all_frozen_strains_are_loadable():
    schema = json.loads(
        (ROOT / "external_data/genome/genome_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    for strain_id in schema["strain_id_order"]:
        record = loader.load_strain_feature(strain_id)
        assert record["features"].shape == (24,)
        assert record["valid_mask"].shape == (24,)
        assert np.isfinite(record["features"]).all()
        assert np.all(record["features"][~record["valid_mask"]] == 0)


def test_dhy210_is_explicit_proxy():
    record = loader.load_strain_feature("DHY210")
    assert record["external_isolate_id"] == "S288C"
    assert record["mapping_type"] == "proxy"
    assert record["proxy_flag"] is True
    assert int(record["valid_mask"].sum()) == 2


def test_unknown_strain_fails_closed():
    try:
        loader.load_strain_feature("UNKNOWN_STRAIN")
    except ValueError as exc:
        assert "absent from the frozen index" in str(exc)
    else:
        raise AssertionError("unknown strain must not receive a silent fallback")

