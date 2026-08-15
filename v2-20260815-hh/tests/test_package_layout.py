from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_package import audit  # noqa: E402


def test_upload_safety_and_frozen_hashes() -> None:
    result = audit(ROOT)
    assert result["status"] == "PASS", result


def test_final_inference_modules_import() -> None:
    for name in (
        "baseline.baseline.model_v2",
        "baseline.baseline.novelty_gated_ensemble_v2",
        "baseline.baseline.second_seed_ensemble_v2",
        "baseline.baseline.final_submission_v2",
    ):
        importlib.import_module(name)
