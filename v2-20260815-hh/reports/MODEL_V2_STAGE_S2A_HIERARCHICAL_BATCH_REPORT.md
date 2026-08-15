# MODEL V2 Stage S2A Hierarchical Batch Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`. The 1.03 threshold is an engineering acceptance threshold, not an official scoring rule.

## Correct interpretation

Stage S1 batch shuffle/disable degradation shows that technical fields carry strong predictive information; it does not by itself show that batch calibration is invalid. The previous 957-treatment diagnostic was a treatment-label holdout whose technical groups could already appear in Stage A controls. S2A instead removes complete plate or instrument groups from train-control Stage A fitting, vocabulary construction, inner checkpoint selection, and gradients.

Public metadata contains no truly new technical category or tuple: train, validation, and test each use 4 sources, 7 instruments, and the same train-known set of 144 complete tuples. Validation/test have zero unseen source, instrument, plate, and tuple counts. Test metadata only was read; test proteome was not opened.

## Plate-group CV, five folds

| fold | structure | train_sample_count | inner_sample_count | holdout_sample_count | holdout_group_count | best_epoch | rmse | mae | global_r2 | sample_pcc_median | protein_pcc_median | group_overlap_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | no_batch | 541 | 60 | 150 | 28 | 28 | 0.9315 | 0.7076 | 0.8876 | 0.9593 | 0.2125 | 0 |
| 0 | flat_batch | 541 | 60 | 150 | 28 | 29 | 0.7308 | 0.5160 | 0.9308 | 0.9738 | 0.7320 | 0 |
| 0 | hierarchical_batch | 541 | 60 | 150 | 28 | 29 | 0.6755 | 0.4731 | 0.9409 | 0.9773 | 0.7674 | 0 |
| 1 | no_batch | 540 | 60 | 151 | 29 | 28 | 0.9007 | 0.6807 | 0.8964 | 0.9618 | 0.2560 | 0 |
| 1 | flat_batch | 540 | 60 | 151 | 29 | 29 | 0.7026 | 0.4945 | 0.9370 | 0.9748 | 0.7608 | 0 |
| 1 | hierarchical_batch | 540 | 60 | 151 | 29 | 29 | 0.6333 | 0.4384 | 0.9488 | 0.9786 | 0.7966 | 0 |
| 2 | no_batch | 541 | 60 | 150 | 29 | 29 | 0.9179 | 0.6998 | 0.8912 | 0.9613 | 0.2333 | 0 |
| 2 | flat_batch | 541 | 60 | 150 | 29 | 29 | 0.7271 | 0.5198 | 0.9317 | 0.9738 | 0.7313 | 0 |
| 2 | hierarchical_batch | 541 | 60 | 150 | 29 | 29 | 0.6501 | 0.4598 | 0.9454 | 0.9792 | 0.7747 | 0 |
| 3 | no_batch | 541 | 60 | 150 | 29 | 29 | 0.9234 | 0.7130 | 0.8895 | 0.9630 | 0.2537 | 0 |
| 3 | flat_batch | 541 | 60 | 150 | 29 | 29 | 0.7278 | 0.5227 | 0.9313 | 0.9744 | 0.7580 | 0 |
| 3 | hierarchical_batch | 541 | 60 | 150 | 29 | 29 | 0.6656 | 0.4702 | 0.9426 | 0.9790 | 0.7856 | 0 |
| 4 | no_batch | 541 | 60 | 150 | 29 | 29 | 0.9258 | 0.7002 | 0.8885 | 0.9600 | 0.2587 | 0 |
| 4 | flat_batch | 541 | 60 | 150 | 29 | 29 | 0.7499 | 0.5320 | 0.9268 | 0.9716 | 0.7219 | 0 |
| 4 | hierarchical_batch | 541 | 60 | 150 | 29 | 29 | 0.6801 | 0.4742 | 0.9398 | 0.9772 | 0.7666 | 0 |

## Instrument-group CV, five folds

| fold | structure | train_sample_count | inner_sample_count | holdout_sample_count | holdout_group_count | best_epoch | rmse | mae | global_r2 | sample_pcc_median | protein_pcc_median | group_overlap_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | no_batch | 516 | 57 | 178 | 1 | 29 | 1.0701 | 0.8311 | 0.8444 | 0.9458 | 0.3411 | 0 |
| 0 | flat_batch | 516 | 57 | 178 | 1 | 29 | 0.9443 | 0.7075 | 0.8789 | 0.9559 | 0.2975 | 0 |
| 0 | hierarchical_batch | 516 | 57 | 178 | 1 | 29 | 0.8437 | 0.6192 | 0.9033 | 0.9618 | 0.4117 | 0 |
| 1 | no_batch | 546 | 61 | 144 | 1 | 29 | 1.1346 | 0.8763 | 0.8398 | 0.9448 | 0.4992 | 0 |
| 1 | flat_batch | 546 | 61 | 144 | 1 | 29 | 0.8351 | 0.5755 | 0.9132 | 0.9651 | 0.5586 | 0 |
| 1 | hierarchical_batch | 546 | 61 | 144 | 1 | 29 | 0.7934 | 0.5579 | 0.9217 | 0.9683 | 0.5887 | 0 |
| 2 | no_batch | 579 | 64 | 108 | 1 | 29 | 1.0289 | 0.7838 | 0.8680 | 0.9509 | 0.4382 | 0 |
| 2 | flat_batch | 579 | 64 | 108 | 1 | 29 | 0.7712 | 0.5338 | 0.9259 | 0.9689 | 0.5682 | 0 |
| 2 | hierarchical_batch | 579 | 64 | 108 | 1 | 29 | 0.7264 | 0.5079 | 0.9342 | 0.9721 | 0.5677 | 0 |
| 3 | no_batch | 510 | 57 | 184 | 2 | 29 | 0.9417 | 0.7119 | 0.8854 | 0.9565 | 0.2738 | 0 |
| 3 | flat_batch | 510 | 57 | 184 | 2 | 29 | 0.8132 | 0.5886 | 0.9145 | 0.9661 | 0.7365 | 0 |
| 3 | hierarchical_batch | 510 | 57 | 184 | 2 | 29 | 0.7722 | 0.5530 | 0.9229 | 0.9690 | 0.7688 | 0 |
| 4 | no_batch | 553 | 61 | 137 | 2 | 29 | 1.0977 | 0.8591 | 0.8358 | 0.9440 | 0.3061 | 0 |
| 4 | flat_batch | 553 | 61 | 137 | 2 | 29 | 0.9908 | 0.7581 | 0.8662 | 0.9535 | 0.2860 | 0 |
| 4 | hierarchical_batch | 553 | 61 | 137 | 2 | 29 | 0.8700 | 0.6501 | 0.8969 | 0.9607 | 0.3914 | 0 |

Fold-equal summaries are stored in `group_cv_summary.csv`. Hierarchical wins plate RMSE against flat in 5/5 folds and against no-batch in 5/5; it wins instrument RMSE against flat in 5/5 folds. The worst per-fold hierarchical/flat instrument RMSE ratio is 0.9501.

## Final validation controls

All 751 train controls are used for fixed-epoch final training. Validation-control labels are used only after checkpoint creation.

| structure | n_samples | rmse | mae | global_r2 | sample_pcc_median | protein_pcc_median | epochs | checkpoint_selection |
|---|---|---|---|---|---|---|---|---|
| no_batch | 205 | 1.0295 | 0.7763 | 0.8635 | 0.9445 | 0.1818 | 30 | fixed_epochs_without_validation_labels |
| flat_batch | 205 | 0.7948 | 0.5532 | 0.9186 | 0.9608 | 0.7832 | 30 | fixed_epochs_without_validation_labels |
| hierarchical_batch | 205 | 0.7666 | 0.5281 | 0.9243 | 0.9635 | 0.8088 | 30 | fixed_epochs_without_validation_labels |

Hierarchical/flat validation-control RMSE ratio is 0.9646, below the engineering threshold 1.03.

## Hierarchical output levels and OOV fallback

| structure | level | rms | max_abs |
|---|---|---|---|
| no_batch | source | 0.0000 | 0.0000 |
| no_batch | instrument | 0.0000 | 0.0000 |
| no_batch | plate | 0.0000 | 0.0000 |
| no_batch | batch | 0.0000 | 0.0000 |
| flat_batch | source | 0.0000 | 0.0000 |
| flat_batch | instrument | 0.0000 | 0.0000 |
| flat_batch | plate | 0.0000 | 0.0000 |
| flat_batch | batch | 0.6084 | 1.3648 |
| hierarchical_batch | source | 0.5206 | 1.0513 |
| hierarchical_batch | instrument | 0.1863 | 0.5883 |
| hierarchical_batch | plate | 0.0446 | 0.2862 |
| hierarchical_batch | batch | 0.6507 | 1.7858 |

Plate layer energy fraction is 0.006461; its residual does not dominate the hierarchy.

| oov_mode | rmse | mae | global_r2 | sample_pcc_median | protein_pcc_median |
|---|---|---|---|---|---|
| known_full | 0.7666 | 0.5281 | 0.9243 | 0.9635 | 0.8088 |
| oov_plate | 0.7669 | 0.5293 | 0.9242 | 0.9633 | 0.8077 |
| oov_instrument | 0.7943 | 0.5534 | 0.9187 | 0.9602 | 0.8007 |
| all_oov | 1.0264 | 0.7833 | 0.8643 | 0.9458 | 0.1812 |

Exact OOV checks: `{"oov_plate_source_unchanged_max_abs": 0.0, "oov_plate_instrument_unchanged_max_abs": 0.0, "oov_plate_delta_plate_max_abs": 0.0, "oov_instrument_source_unchanged_max_abs": 0.0, "oov_instrument_delta_instrument_max_abs": 0.0, "oov_instrument_delta_plate_max_abs": 0.0, "all_oov_delta_batch_max_abs": 0.0}`. Unseen plate preserves source/instrument and zeros plate; unseen instrument preserves source and zeros instrument/plate; all-OOV produces exactly zero batch correction.

## Predeclared gates

```json
{
  "gate_1_plate_holdout_rmse_better_than_flat_and_no_batch": true,
  "gate_2_instrument_holdout_no_collapse": true,
  "gate_3_majority_fold_direction_consistent": true,
  "gate_4_validation_control_rmse_ratio_le_1_03": true,
  "gate_5_oov_partial_fallback_exact": true,
  "gate_6_plate_residual_not_dominant": true,
  "gate_7_no_validation_or_test_label_fit": true
}
```

`eligible_for_stage_s2b=true`. All seven engineering gates pass. This authorizes Main to consider S2B; S2B was not started automatically.

## Reproducibility and boundaries

- Seed: 20260814; device: cuda; elapsed: 273.0 seconds.
- Checkpoints: 33 total, with per-fold/final histories and resolved configs under `reports/model_v2_stage_s2a`.
- Every fold records fit, inner, and outer-holdout ID hashes, checkpoint hash, group counts, and zero overlap.
- Protein/source artifact hashes are preserved in `run_summary.json` and every checkpoint.
- Stage B treatment training, FC, chemical variants, response training, test prediction, and test proteome access: NOT RUN.
- Formal failures, OOM, NaN, and checkpoint failures: none.

The task pauses here pending Main review.
