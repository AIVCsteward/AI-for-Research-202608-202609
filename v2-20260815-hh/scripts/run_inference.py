"""Portable entry point for the frozen GOAI V2 two-seed ensemble."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from baseline.baseline.final_submission_v2 import run_test_inference_pass  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    required_metadata = (
        root / "WAYB_WAYC/WAYB_WAYC_metadata_train_val(1).csv",
        root / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv",
    )
    missing = [str(path) for path in required_metadata if not path.is_file()]
    if missing:
        raise SystemExit("Missing authorized competition metadata:\n- " + "\n- ".join(missing))
    if output.exists():
        raise SystemExit(f"Output directory must not already exist: {output}")

    summary = run_test_inference_pass(root, output, args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
