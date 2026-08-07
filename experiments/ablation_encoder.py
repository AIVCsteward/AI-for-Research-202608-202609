"""Feature-only ablation utilities for Person A.

This module prepares comparable fixed-256-dimensional inputs.  Model training and
metric evaluation are intentionally injected by the caller so Person C can use
the same ablation definitions with the shared training loop.

Example:
    python -m experiments.ablation_encoder --output-dir experiments/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from baseline.config import get_experiment_config
from baseline.data import load_raw_data, preprocess
from baseline.features import build_condition_features, fit_feature_encoders


ABLATION_OVERRIDES = {
    "full": {},
    "no_strain_prior": {"encoder": {"use_strain_prior": False}},
    "no_chem_anchor": {"encoder": {"use_chem_anchor": False}},
    "no_hash": {"encoder": {"use_hash_features": False}},
    "no_cross_features": {"encoder": {"use_cross_features": False}},
}


def get_ablation_configs() -> Dict[str, dict]:
    """Return isolated configs for the full encoder and four single ablations."""
    return {
        name: get_experiment_config(overrides)
        for name, overrides in ABLATION_OVERRIDES.items()
    }


def prepare_ablation_features(meta, y_log2, mask_matrix=None):
    """Fit and transform every encoder ablation using train rows only.

    Returns ``{name: {"X": ndarray, "encoders": dict, "config": dict}}``.
    The returned arrays are in the same sample order as ``meta``.
    """
    if "split_final" in meta:
        train_mask = meta["split_final"].astype(str).eq("train")
    else:
        train_mask = np.ones(len(meta), dtype=bool)
    train_meta = meta.loc[train_mask]
    train_y = y_log2.loc[train_meta.index]
    train_mask_matrix = None if mask_matrix is None else mask_matrix.loc[train_meta.index]

    results = {}
    for name, config in get_ablation_configs().items():
        encoders = fit_feature_encoders(
            train_meta,
            train_y,
            train_mask_matrix,
            config=config,
        )
        results[name] = {
            "X": build_condition_features(meta, encoders=encoders),
            "encoders": encoders,
            "config": config,
        }
    return results


def save_ablation_manifest(results, output_dir: Path):
    """Save small metadata/feature arrays without serializing raw data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, result in results.items():
        x_path = output_dir / f"X_{name}.npy"
        np.save(x_path, result["X"])
        manifest[name] = {
            "feature_file": x_path.name,
            "shape": list(result["X"].shape),
            "d_emb": int(result["X"].shape[1]),
            "raw_dim": int(result["encoders"]["raw_dim"]),
            "feature_slices": {
                key: list(value)
                for key, value in result["encoders"].get("feature_slices", {}).items()
            },
        }
    manifest_path = output_dir / "encoder_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare AIVC encoder ablation features")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Ignored output directory for generated feature arrays and manifest",
    )
    args = parser.parse_args(argv)

    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, _, _ = preprocess(meta, prot)
    results = prepare_ablation_features(meta, y_log2, mask_matrix)
    path = save_ablation_manifest(results, args.output_dir)
    print(f"已生成 {len(results)} 组 encoder 特征: {path}")


if __name__ == "__main__":
    main()
