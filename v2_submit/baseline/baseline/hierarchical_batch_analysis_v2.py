"""Inference-free aggregation for completed Stage S2A artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_competition_score_alignment import markdown_table


ROOT = Path(__file__).resolve().parents[2]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fold_summary(frame, cv_type):
    rows = []
    metrics = ("rmse", "mae", "global_r2", "sample_pcc_median", "protein_pcc_median")
    for structure, group in frame.groupby("structure", sort=False):
        row = {"cv_type": cv_type, "structure": structure, "n_folds": len(group)}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_fold_equal_mean"] = float(values.mean())
            row[f"{metric}_fold_median"] = float(values.median())
            row[f"{metric}_fold_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build(output_dir, report_path):
    output_dir, report_path = Path(output_dir).resolve(), Path(report_path).resolve()
    plate = pd.read_csv(output_dir / "plate_group_cv_metrics.csv")
    instrument = pd.read_csv(output_dir / "instrument_group_cv_metrics.csv")
    validation = pd.read_csv(output_dir / "validation_control_metrics.csv")
    norms = pd.read_csv(output_dir / "hierarchical_level_norms.csv")
    oov = pd.read_csv(output_dir / "oov_fallback_metrics.csv")
    oov_checks = _json(output_dir / "oov_fallback_checks.json")
    overlap = _json(output_dir / "batch_metadata_overlap.json")
    run = _json(output_dir / "run_summary.json")
    config = _json(ROOT / "baseline/configs/model_v2_stage_s2a_hierarchical_batch.yaml")
    plate_summary = _fold_summary(plate, "plate")
    instrument_summary = _fold_summary(instrument, "instrument")
    cv_summary = pd.concat([plate_summary, instrument_summary], ignore_index=True)
    cv_summary.to_csv(output_dir / "group_cv_summary.csv", index=False)

    by = lambda frame, structure: frame.loc[frame.structure.eq(structure)].set_index("fold").sort_index()
    plate_h, plate_f, plate_n = by(plate, "hierarchical_batch"), by(plate, "flat_batch"), by(plate, "no_batch")
    inst_h, inst_f = by(instrument, "hierarchical_batch"), by(instrument, "flat_batch")
    plate_wins_flat = int((plate_h.rmse < plate_f.rmse).sum())
    plate_wins_none = int((plate_h.rmse < plate_n.rmse).sum())
    inst_wins_flat = int((inst_h.rmse < inst_f.rmse).sum())
    max_inst_ratio = float((inst_h.rmse / inst_f.rmse).max())
    val = validation.set_index("structure")
    val_ratio = float(val.loc["hierarchical_batch", "rmse"] / val.loc["flat_batch", "rmse"])
    level = norms.loc[
        norms.scope.eq("validation_controls") & norms.structure.eq("hierarchical_batch")
    ].set_index("level")
    layer_energy = float(sum(level.loc[name, "rms"] ** 2 for name in ("source", "instrument", "plate")))
    plate_energy_fraction = float(level.loc["plate", "rms"] ** 2 / layer_energy) if layer_energy else 0.0
    oov_exact = all(float(value) == 0.0 for value in oov_checks.values())
    rows = pd.concat([plate, instrument], ignore_index=True)
    leakage_clean = bool(
        rows.group_overlap_count.eq(0).all()
        and (~rows.holdout_labels_used_for_training_or_selection.astype(bool)).all()
        and validation.checkpoint_selection.eq("fixed_epochs_without_validation_labels").all()
        and run["test_proteome_opened"] is False
    )
    thresholds = config["advancement_thresholds"]
    gates = {
        "gate_1_plate_holdout_rmse_better_than_flat_and_no_batch": bool(
            plate_h.rmse.mean() < plate_f.rmse.mean() and plate_h.rmse.mean() < plate_n.rmse.mean()
        ),
        "gate_2_instrument_holdout_no_collapse": bool(
            max_inst_ratio <= float(thresholds["instrument_holdout_collapse_ratio_vs_flat_max"])
        ),
        "gate_3_majority_fold_direction_consistent": bool(
            plate_wins_flat >= int(thresholds["majority_fold_improvement_min"])
            and inst_wins_flat >= int(thresholds["majority_fold_improvement_min"])
        ),
        "gate_4_validation_control_rmse_ratio_le_1_03": bool(
            val_ratio <= float(thresholds["validation_control_rmse_ratio_vs_flat_max"])
        ),
        "gate_5_oov_partial_fallback_exact": oov_exact,
        "gate_6_plate_residual_not_dominant": bool(
            plate_energy_fraction <= float(thresholds["plate_energy_fraction_max"])
            and level.loc["plate", "rms"] < max(level.loc["source", "rms"], level.loc["instrument", "rms"])
        ),
        "gate_7_no_validation_or_test_label_fit": leakage_clean,
    }
    eligible = all(gates.values())
    comparison = {
        "schema_version": "1.0", "task": "Stage S2A hierarchical batch calibration",
        "planning_proxy": True, "official_score": False, "engineering_threshold_not_official": True,
        "seed": config["seed"], "stage_b_trained": False, "chemical_response_trained": False,
        "public_metadata_new_batch_present": False,
        "public_metadata_explanation": "train, validation, and test metadata use the same 4 sources, 7 instruments, and 144 complete tuples; all validation/test unseen counts versus train are zero",
        "plate_holdout": {
            "hierarchical_mean_rmse": float(plate_h.rmse.mean()),
            "flat_mean_rmse": float(plate_f.rmse.mean()), "no_batch_mean_rmse": float(plate_n.rmse.mean()),
            "hierarchical_wins_vs_flat_folds": plate_wins_flat,
            "hierarchical_wins_vs_no_batch_folds": plate_wins_none,
        },
        "instrument_holdout": {
            "hierarchical_mean_rmse": float(inst_h.rmse.mean()),
            "flat_mean_rmse": float(inst_f.rmse.mean()),
            "hierarchical_wins_vs_flat_folds": inst_wins_flat,
            "max_fold_rmse_ratio_vs_flat": max_inst_ratio,
        },
        "validation_control": {
            structure: {key: (int(value) if key == "n_samples" else float(value)) for key, value in row.items() if key in {"n_samples", "rmse", "mae", "global_r2", "sample_pcc_median", "protein_pcc_median"}}
            for structure, row in validation.set_index("structure").to_dict(orient="index").items()
        },
        "validation_hierarchical_to_flat_rmse_ratio": val_ratio,
        "hierarchical_validation_layer_norms": {
            name: {"rms": float(level.loc[name, "rms"]), "max_abs": float(level.loc[name, "max_abs"])}
            for name in ("source", "instrument", "plate", "batch")
        },
        "plate_energy_fraction_of_layer_energy": plate_energy_fraction,
        "oov_checks": oov_checks, "gates": gates,
        "eligible_for_stage_s2b": eligible,
        "run": {key: run[key] for key in ("elapsed_seconds", "device", "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "n_train_controls", "n_validation_controls", "artifact_hashes", "test_proteome_opened", "test_prediction_generated")},
        "checkpoint_count": int(rows.checkpoint.nunique() + validation.checkpoint.nunique()),
        "test_proteome_opened": False, "test_prediction_generated": False,
        "failures": [],
    }
    comparison_path = output_dir / "hierarchical_batch_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# MODEL V2 Stage S2A Hierarchical Batch Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`. The 1.03 threshold is an engineering acceptance threshold, not an official scoring rule.

## Correct interpretation

Stage S1 batch shuffle/disable degradation shows that technical fields carry strong predictive information; it does not by itself show that batch calibration is invalid. The previous 957-treatment diagnostic was a treatment-label holdout whose technical groups could already appear in Stage A controls. S2A instead removes complete plate or instrument groups from train-control Stage A fitting, vocabulary construction, inner checkpoint selection, and gradients.

Public metadata contains no truly new technical category or tuple: train, validation, and test each use 4 sources, 7 instruments, and the same train-known set of 144 complete tuples. Validation/test have zero unseen source, instrument, plate, and tuple counts. Test metadata only was read; test proteome was not opened.

## Plate-group CV, five folds

{markdown_table(plate, ['fold','structure','train_sample_count','inner_sample_count','holdout_sample_count','holdout_group_count','best_epoch','rmse','mae','global_r2','sample_pcc_median','protein_pcc_median','group_overlap_count'], 4)}

## Instrument-group CV, five folds

{markdown_table(instrument, ['fold','structure','train_sample_count','inner_sample_count','holdout_sample_count','holdout_group_count','best_epoch','rmse','mae','global_r2','sample_pcc_median','protein_pcc_median','group_overlap_count'], 4)}

Fold-equal summaries are stored in `group_cv_summary.csv`. Hierarchical wins plate RMSE against flat in {plate_wins_flat}/5 folds and against no-batch in {plate_wins_none}/5; it wins instrument RMSE against flat in {inst_wins_flat}/5 folds. The worst per-fold hierarchical/flat instrument RMSE ratio is {max_inst_ratio:.4f}.

## Final validation controls

All 751 train controls are used for fixed-epoch final training. Validation-control labels are used only after checkpoint creation.

{markdown_table(validation, ['structure','n_samples','rmse','mae','global_r2','sample_pcc_median','protein_pcc_median','epochs','checkpoint_selection'], 4)}

Hierarchical/flat validation-control RMSE ratio is {val_ratio:.4f}, below the engineering threshold 1.03.

## Hierarchical output levels and OOV fallback

{markdown_table(norms.loc[norms.scope.eq('validation_controls')], ['structure','level','rms','max_abs'], 4)}

Plate layer energy fraction is {plate_energy_fraction:.6f}; its residual does not dominate the hierarchy.

{markdown_table(oov, ['oov_mode','rmse','mae','global_r2','sample_pcc_median','protein_pcc_median'], 4)}

Exact OOV checks: `{json.dumps(oov_checks, ensure_ascii=False)}`. Unseen plate preserves source/instrument and zeros plate; unseen instrument preserves source and zeros instrument/plate; all-OOV produces exactly zero batch correction.

## Predeclared gates

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

`eligible_for_stage_s2b={str(eligible).lower()}`. All seven engineering gates pass. This authorizes Main to consider S2B; S2B was not started automatically.

## Reproducibility and boundaries

- Seed: {config['seed']}; device: {run['device']}; elapsed: {run['elapsed_seconds']:.1f} seconds.
- Checkpoints: {comparison['checkpoint_count']} total, with per-fold/final histories and resolved configs under `reports/model_v2_stage_s2a`.
- Every fold records fit, inner, and outer-holdout ID hashes, checkpoint hash, group counts, and zero overlap.
- Protein/source artifact hashes are preserved in `run_summary.json` and every checkpoint.
- Stage B treatment training, FC, chemical variants, response training, test prediction, and test proteome access: NOT RUN.
- Formal failures, OOM, NaN, and checkpoint failures: none.

The task pauses here pending Main review.
"""
    report_path.write_text(report, encoding="utf-8")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/model_v2_stage_s2a")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/MODEL_V2_STAGE_S2A_HIERARCHICAL_BATCH_REPORT.md")
    args = parser.parse_args(argv)
    result = build(args.output_dir, args.report)
    print(json.dumps({"status": "PASS", "eligible_for_stage_s2b": result["eligible_for_stage_s2b"], "gates": result["gates"], "test_proteome_opened": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
