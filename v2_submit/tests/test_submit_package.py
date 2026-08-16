from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline.training_v2 import load_artifact_bundle, load_checkpoint  # noqa: E402
from scripts.verify_submit_package import audit  # noqa: E402


CHECKPOINTS = (
    "reports/model_v2_stage_s1/formal_c_anchor_morgan_fc_seed_20260814/stage_b_best.pt",
    "reports/model_v2_stage_s2b/formal_d0_hierarchical_none_huber_seed_20260814/stage_b_best.pt",
    "reports/model_v2_stage_s2b/formal_d2_hierarchical_morgan_fc_seed_20260814/stage_b_best.pt",
    "reports/model_v2_stage_s2d/formal_d0_hierarchical_none_huber_seed_20260815/stage_b_best.pt",
    "reports/model_v2_stage_s2d/formal_d2_hierarchical_morgan_fc_seed_20260815/stage_b_best.pt",
    "reports/model_v2_stage_s2d/formal_s1c_flat_morgan_fc_seed_20260815/stage_b_best.pt",
)

STAGE_A = (
    "reports/model_v2_stage_s2a/final/hierarchical_batch/stage_a_final.pt",
    "reports/model_v2_stage_s2d/stage_a_hierarchical_seed_20260815/stage_a_final.pt",
    "reports/model_v2_stage2/formal_huber_seed_20260814/stage_a_best.pt",
    "reports/model_v2_stage_s2d/stage_a_flat_seed_20260815/stage_a_best.pt",
)


def test_upload_layout_has_no_raw_or_sample_level_competition_files():
    result = audit(ROOT)
    assert result["technical_status"] == "PASS", result


def test_final_modules_import_from_submit_root():
    for name in (
        "baseline.baseline.model_v2",
        "baseline.baseline.novelty_gated_ensemble_v2",
        "baseline.baseline.second_seed_ensemble_v2",
        "baseline.baseline.final_submission_v2",
    ):
        importlib.import_module(name)


def test_frozen_artifacts_and_six_experts_match_checkpoint_contract():
    artifacts = load_artifact_bundle(ROOT)
    assert artifacts.feature_contract.n_proteins == 4422
    assert artifacts.morgan_dim == 2048
    assert artifacts.descriptor_dim == 217
    assert artifacts.genome_dim == 24
    for relative in CHECKPOINTS:
        payload = load_checkpoint(ROOT / relative, artifacts.hashes)
        assert int(payload["seed"]) in {20260814, 20260815}


def test_four_stage_a_checkpoints_are_present_and_readable():
    for relative in STAGE_A:
        payload = torch.load(ROOT / relative, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict)
        assert "model_state" in payload or "model_state_dict" in payload


def test_open_knowledge_disclosure_and_license_boundary_are_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    disclosure = (ROOT / "docs/OPEN_KNOWLEDGE_DISCLOSURE.md").read_text(encoding="utf-8")
    assert "开放知识榜" in readme
    assert "PubChem" in disclosure and "Peter et al. 2018" in disclosure
    assert (ROOT / "LICENSE_DECISION_REQUIRED.md").is_file()


def test_package_manifest_covers_every_non_manifest_file():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    listed = {item["path"] for item in manifest["files"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    }
    assert listed == actual
