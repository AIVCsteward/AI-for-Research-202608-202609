# Competition Score Alignment Audit (Stage S0)

Generated: 2026-08-15T03:10:08+08:00  
Mode: inference-only audit; no training, architecture change, checkpoint mutation, test prediction, or Git operation.  
Scoring label: `control_mapping=pert_id_parity_v1`, `experimental_nonofficial_parity_fc=true`, `official_score=false`.

## Executive conclusion

- Absolute-fidelity winner on the strict common intersection (mean rank across all seven required absolute metrics and four scenarios): **Matched Control**.
- Raw-FC winner on the same intersection (mean rank across PCC, RMSE, and direction metrics): **V2 batch-enabled Huber**.
- `val_chem_only` context-residual winner: **V2 batch-enabled Huber**.
- `val_strain_only` drug-residual winner: **V2 batch-enabled Huber**.
- V1 ConditionMLP is NOT_EVALUABLE: the workspace preserves only aggregate reproduction metrics, not a checkpoint or per-sample predictions. Retraining was forbidden.
- The planning proxy is explicitly nonofficial. Scheme winners: bounded_quality=V2 batch-enabled Huber, correlation_priority=V2 batch-enabled Huber, weighted_module_rank=V2 batch-enabled Huber. Stable across schemes: **True**.
- No result in this report is an official FC score or leaderboard score because the organizer Water/DMSO map and the module-internal aggregation formula remain unpublished.

## Rule extraction and interface status

`OfficialRules.pdf` pages 16-17 specify module weights of 20% absolute fidelity, 25% matched-control raw FC, 20% context residual, 20% drug residual, 10% double-unknown/time, and 5% high-effect/DEP. The PDF does not publish the exact within-module normalization or aggregation into one score. Accordingly, this audit reports raw metrics and three labeled planning proxies only.

The frozen exact-control key is the full eight-field tuple: `data_source, Strains, Medium, Temperature, pert_time, pert_time_unit, instrument, Yeast_cell_plate`. Multiple exact controls are aggregated protein-wise over observed positions. No other-solvent, global-mean, or looser-context fallback is used.

Interface conflicts / unresolved items:

1. The official `pert_id -> Water/DMSO` mapping is still unconfirmed. This audit uses only the permitted parity mapping and is nonofficial.
2. The published diagnostic counts cannot all be reproduced by the parity rule. The parity rule is frozen from train metadata/plate evidence and was not selected using validation scores.
3. `OfficialRules.pdf` calls the fourth module time extrapolation, while `20260812Approach.pdf` describes `val_time` as interpolation between observed time points. This audit preserves the frozen split name and does not invent a new residual.
4. The official PDFs give module descriptions but no unique internal aggregation formula; a unique official total is therefore blocked.

## Artifact recovery

| model | status | seed | checkpoint_sha256 | output_scale | reason |
|---|---|---|---|---|---|
| Matched Control | EVALUABLE | NA |  | log2 | exact matched observed control; no fallback |
| V1 ConditionMLP | NOT_EVALUABLE | 42.0000 |  | log2 | aggregate metrics only; no checkpoint or per-sample prediction artifact; retraining forbidden |
| V2 batch-enabled Huber | EVALUABLE | 20260814.0000 | d7661f57a7e1aff15f34a5d7d54156d3fc064be4effe3780eaa864f9a18e72c7 | log2 |  |
| V2 no-batch correct | EVALUABLE | 20260814.0000 | 7c33ab3ee18ad1d48f169f5bfdb513a4703625cb7a745c664966b464b587b79a | log2 |  |
| V2 no-batch chemical zero | EVALUABLE | 20260814.0000 | 98c31ecb4321109c75df07519b9c23459ec0bcccf818c46f25681e32eb0e7f97 | log2 |  |
| V2 no-batch chemical shuffle | EVALUABLE | 20260814.0000 | 1c8fc7f67cab6be1e2cb4d9312bd98c4d341d876096de44c6cad2b2da97b9cd1 | log2 |  |
| V2 no-batch Morgan-only | EVALUABLE | 20260814.0000 | 8cae218f98fb66d447cdbf763709ddbd1d8298b3bcec99b767b8d82084071b8a | log2 |  |
| V2 experimental parity FC Morgan no-batch | EVALUABLE | 20260814.0000 | faa3c132724d022fd973db12e5bb7cdb20f2dda6b49fd059db02ff9afef588b4 | log2 |  |

Every recovered V2 checkpoint matched the current frozen artifact hashes and declared `output_scale=log2`. V1 was not reconstructed by training.

## Common-subset coverage

| scenario | scenario_sample_count | treatment_count | matched_control_samples | unmatched_treatment_samples | common_valid_positions | common_mask_coverage_of_scenario_truth |
|---|---|---|---|---|---|---|
| val_chem_only | 1065 | 1065 | 981 | 84 | 3577143 | 0.889604 |
| val_strain_only | 1547 | 1333 | 1313 | 20 | 4824234 | 0.817286 |
| val_both | 269 | 269 | 266 | 3 | 960434 | 0.952113 |
| val_time | 157 | 139 | 134 | 5 | 486393 | 0.825517 |

The common intersection includes only validation treatment samples with an exact parity-selected control and protein positions where treatment truth, observed control, and every evaluable model prediction are finite. NOT_EVALUABLE models do not silently shrink the intersection; they remain excluded and are reported separately.

## Matched Control regression against the published diagnostic table

| scenario | parity_n | published_n | delta_n | parity_rmse | published_rmse | parity_global_r2 | published_global_r2 | parity_protein_r2 | published_protein_r2 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only | 981 | 1015 | -34 | 0.3817 | 0.3790 | 0.9804 | 0.9800 | 0.8279 | 0.8360 |
| val_strain_only | 1313 | 1293 | 20 | 0.4101 | 0.3990 | 0.9769 | 0.9780 | 0.7037 | 0.7260 |
| val_both | 266 | 266 | 0 | 0.3853 | 0.3820 | 0.9800 | 0.9800 | 0.8018 | 0.8090 |
| val_time | 134 | 128 | 6 | 0.4336 | 0.4260 | 0.9745 | 0.9750 | 0.7066 | 0.7190 |

Count differences are expected under the permitted parity mapping: the published table states that it used an organizer Water/DMSO map that is absent from the released materials. The earlier metadata-only contract diagnostic reported 1,007 / 1,313 / 266 / 135 because its candidate lookup allowed control IDs from all metadata splits, including test metadata. Restricting the pool to train/validation controls with legally observable labels gives 981 / 1,313 / 266 / 134; the 26 chem-only and one time sample difference would otherwise require test-control protein truth, which this audit never opens. Remaining differences from 1,015 / 1,293 / 266 / 128 are attributed to the unpublished organizer mapping and possibly an undisclosed sample-level QC/control-replicate rule.

## Absolute fidelity on the common intersection

| model | scenario | n_samples | n_valid_positions | log2_rmse | mae | global_r2 | sample_pcc_median | sample_r2_median | protein_pcc_median | protein_r2_median |
|---|---|---|---|---|---|---|---|---|---|---|
| V2 batch-enabled Huber | val_chem_only | 981 | 3577143 | 0.6196 | 0.4496 | 0.9483 | 0.9804 | 0.9561 | 0.8169 | 0.5494 |
| V2 no-batch correct | val_chem_only | 981 | 3577143 | 1.0718 | 0.8002 | 0.8452 | 0.9264 | 0.8205 | 0.1932 | -0.1282 |
| V2 no-batch chemical zero | val_chem_only | 981 | 3577143 | 0.9721 | 0.7490 | 0.8727 | 0.9486 | 0.8605 | 0.1587 | -0.0411 |
| V2 no-batch chemical shuffle | val_chem_only | 981 | 3577143 | 1.1035 | 0.8440 | 0.8360 | 0.9155 | 0.8073 | -0.0500 | -0.2303 |
| V2 no-batch Morgan-only | val_chem_only | 981 | 3577143 | 0.9536 | 0.7044 | 0.8775 | 0.9683 | 0.9075 | 0.2238 | -0.0166 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 981 | 3577143 | 0.9899 | 0.7610 | 0.8680 | 0.9569 | 0.8876 | 0.0363 | -0.0734 |
| Matched Control | val_chem_only | 981 | 3577143 | 0.3817 | 0.2543 | 0.9804 | 0.9924 | 0.9834 | 0.9163 | 0.8279 |
| V2 batch-enabled Huber | val_strain_only | 1313 | 4824234 | 0.7368 | 0.5126 | 0.9253 | 0.9653 | 0.9278 | 0.7714 | 0.3476 |
| V2 no-batch correct | val_strain_only | 1313 | 4824234 | 0.8136 | 0.5852 | 0.9089 | 0.9694 | 0.9328 | 0.4481 | 0.0083 |
| V2 no-batch chemical zero | val_strain_only | 1313 | 4824234 | 0.8450 | 0.6284 | 0.9018 | 0.9626 | 0.9154 | 0.1648 | -0.0410 |
| V2 no-batch chemical shuffle | val_strain_only | 1313 | 4824234 | 0.7958 | 0.5760 | 0.9129 | 0.9690 | 0.9304 | 0.3912 | 0.0192 |
| V2 no-batch Morgan-only | val_strain_only | 1313 | 4824234 | 0.9194 | 0.6404 | 0.8837 | 0.9584 | 0.9107 | 0.4554 | -0.0218 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 1313 | 4824234 | 0.7940 | 0.5782 | 0.9133 | 0.9691 | 0.9291 | 0.3830 | 0.0199 |
| Matched Control | val_strain_only | 1313 | 4824234 | 0.4101 | 0.2735 | 0.9769 | 0.9911 | 0.9811 | 0.8573 | 0.7037 |
| V2 batch-enabled Huber | val_both | 266 | 960434 | 0.8315 | 0.5945 | 0.9068 | 0.9621 | 0.9098 | 0.8115 | 0.3373 |
| V2 no-batch correct | val_both | 266 | 960434 | 1.1011 | 0.8331 | 0.8365 | 0.9244 | 0.8119 | 0.1860 | -0.2311 |
| V2 no-batch chemical zero | val_both | 266 | 960434 | 0.9612 | 0.7455 | 0.8754 | 0.9520 | 0.8633 | 0.1814 | -0.0643 |
| V2 no-batch chemical shuffle | val_both | 266 | 960434 | 1.0702 | 0.8257 | 0.8456 | 0.9231 | 0.8191 | -0.0530 | -0.2478 |
| V2 no-batch Morgan-only | val_both | 266 | 960434 | 1.1233 | 0.8297 | 0.8299 | 0.9394 | 0.8494 | 0.1448 | -0.2150 |
| V2 experimental parity FC Morgan no-batch | val_both | 266 | 960434 | 0.9917 | 0.7664 | 0.8674 | 0.9526 | 0.8858 | 0.0319 | -0.1323 |
| Matched Control | val_both | 266 | 960434 | 0.3853 | 0.2570 | 0.9800 | 0.9918 | 0.9825 | 0.9061 | 0.8018 |
| V2 batch-enabled Huber | val_time | 134 | 486393 | 0.5563 | 0.3819 | 0.9580 | 0.9820 | 0.9620 | 0.7581 | 0.5483 |
| V2 no-batch correct | val_time | 134 | 486393 | 0.7438 | 0.5257 | 0.9250 | 0.9759 | 0.9423 | 0.4292 | 0.1468 |
| V2 no-batch chemical zero | val_time | 134 | 486393 | 0.8403 | 0.6160 | 0.9043 | 0.9620 | 0.9140 | 0.1431 | -0.0041 |
| V2 no-batch chemical shuffle | val_time | 134 | 486393 | 0.7907 | 0.5706 | 0.9152 | 0.9669 | 0.9258 | 0.3447 | 0.0738 |
| V2 no-batch Morgan-only | val_time | 134 | 486393 | 0.6949 | 0.4765 | 0.9345 | 0.9817 | 0.9520 | 0.5171 | 0.2362 |
| V2 experimental parity FC Morgan no-batch | val_time | 134 | 486393 | 0.7885 | 0.5718 | 0.9157 | 0.9662 | 0.9210 | 0.3391 | 0.0750 |
| Matched Control | val_time | 134 | 486393 | 0.4336 | 0.2843 | 0.9745 | 0.9917 | 0.9816 | 0.8549 | 0.7066 |

Winner selection above uses all seven metrics, not RMSE or Global R2 alone. Mean-rank details: [{'model': 'Matched Control', 'rank': 1.0}, {'model': 'V2 batch-enabled Huber', 'rank': 2.2142857142857144}, {'model': 'V2 no-batch Morgan-only', 'rank': 4.5}, {'model': 'V2 experimental parity FC Morgan no-batch', 'rank': 4.535714285714286}, {'model': 'V2 no-batch correct', 'rank': 4.892857142857143}, {'model': 'V2 no-batch chemical zero', 'rank': 5.285714285714286}, {'model': 'V2 no-batch chemical shuffle', 'rank': 5.571428571428571}].

## Raw FC on the common intersection

Definition: `delta_pred = y_pred_treatment - y_control_observed`; `delta_true = y_true_treatment - y_control_observed`. The exact same pairwise mask is applied to both.

| model | scenario | n_samples | n_proteins | n_valid_positions | global_fc_pcc | sample_fc_pcc_median | protein_fc_pcc_median | fc_rmse | fc_direction_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | 981 | 4422 | 3577143 | NA | NA | NA | 0.3817 | 0.0000 |
| V2 batch-enabled Huber | val_chem_only | 981 | 4422 | 3577143 | 0.3072 | 0.3000 | 0.3460 | 0.6196 | 0.5976 |
| V2 no-batch correct | val_chem_only | 981 | 4422 | 3577143 | 0.1764 | 0.1989 | 0.2167 | 1.0718 | 0.5549 |
| V2 no-batch chemical zero | val_chem_only | 981 | 4422 | 3577143 | 0.1962 | 0.2102 | 0.2157 | 0.9721 | 0.5541 |
| V2 no-batch chemical shuffle | val_chem_only | 981 | 4422 | 3577143 | 0.1767 | 0.1902 | 0.2078 | 1.1035 | 0.5510 |
| V2 no-batch Morgan-only | val_chem_only | 981 | 4422 | 3577143 | 0.1937 | 0.2459 | 0.2208 | 0.9536 | 0.5597 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 981 | 4422 | 3577143 | 0.1908 | 0.2084 | 0.2133 | 0.9899 | 0.5516 |
| Matched Control | val_strain_only | 1313 | 4413 | 4824234 | NA | NA | NA | 0.4101 | 0.0000 |
| V2 batch-enabled Huber | val_strain_only | 1313 | 4413 | 4824234 | 0.2443 | 0.2344 | 0.4034 | 0.7368 | 0.5873 |
| V2 no-batch correct | val_strain_only | 1313 | 4413 | 4824234 | 0.2220 | 0.2517 | 0.2853 | 0.8136 | 0.5759 |
| V2 no-batch chemical zero | val_strain_only | 1313 | 4413 | 4824234 | 0.2310 | 0.2426 | 0.2614 | 0.8450 | 0.5739 |
| V2 no-batch chemical shuffle | val_strain_only | 1313 | 4413 | 4824234 | 0.2429 | 0.2663 | 0.2762 | 0.7958 | 0.5831 |
| V2 no-batch Morgan-only | val_strain_only | 1313 | 4413 | 4824234 | 0.1946 | 0.2160 | 0.2905 | 0.9194 | 0.5735 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 1313 | 4413 | 4824234 | 0.2485 | 0.2719 | 0.2777 | 0.7940 | 0.5836 |
| Matched Control | val_both | 266 | 4408 | 960434 | NA | NA | NA | 0.3853 | 0.0000 |
| V2 batch-enabled Huber | val_both | 266 | 4408 | 960434 | 0.2257 | 0.2086 | 0.3726 | 0.8315 | 0.5720 |
| V2 no-batch correct | val_both | 266 | 4408 | 960434 | 0.1488 | 0.1680 | 0.2239 | 1.1011 | 0.5431 |
| V2 no-batch chemical zero | val_both | 266 | 4408 | 960434 | 0.1850 | 0.2090 | 0.2242 | 0.9612 | 0.5494 |
| V2 no-batch chemical shuffle | val_both | 266 | 4408 | 960434 | 0.1693 | 0.1955 | 0.2175 | 1.0702 | 0.5459 |
| V2 no-batch Morgan-only | val_both | 266 | 4408 | 960434 | 0.1461 | 0.1690 | 0.2233 | 1.1233 | 0.5436 |
| V2 experimental parity FC Morgan no-batch | val_both | 266 | 4408 | 960434 | 0.1783 | 0.2047 | 0.2210 | 0.9917 | 0.5462 |
| Matched Control | val_time | 134 | 4422 | 486393 | NA | NA | NA | 0.4336 | 0.0000 |
| V2 batch-enabled Huber | val_time | 134 | 4422 | 486393 | 0.3139 | 0.3182 | 0.3363 | 0.5563 | 0.6096 |
| V2 no-batch correct | val_time | 134 | 4422 | 486393 | 0.2459 | 0.2723 | 0.2508 | 0.7438 | 0.5850 |
| V2 no-batch chemical zero | val_time | 134 | 4422 | 486393 | 0.2091 | 0.2198 | 0.2182 | 0.8403 | 0.5660 |
| V2 no-batch chemical shuffle | val_time | 134 | 4422 | 486393 | 0.2248 | 0.2331 | 0.2341 | 0.7907 | 0.5750 |
| V2 no-batch Morgan-only | val_time | 134 | 4422 | 486393 | 0.2711 | 0.3357 | 0.2725 | 0.6949 | 0.6032 |
| V2 experimental parity FC Morgan no-batch | val_time | 134 | 4422 | 486393 | 0.2348 | 0.2364 | 0.2369 | 0.7885 | 0.5770 |

Mean-rank details: [{'model': 'V2 batch-enabled Huber', 'rank': 1.55}, {'model': 'V2 experimental parity FC Morgan no-batch', 'rank': 3.65}, {'model': 'V2 no-batch Morgan-only', 'rank': 3.8}, {'model': 'V2 no-batch chemical zero', 'rank': 4.15}, {'model': 'V2 no-batch correct', 'rank': 4.25}, {'model': 'V2 no-batch chemical shuffle', 'rank': 4.8}, {'model': 'Matched Control', 'rank': 5.8}].

## Train-only residual modules

`mu_ctx` is a protein-wise mean of train treatment `delta_true` grouped by the complete frozen eight-field context. `mu_drug` is a protein-wise mean grouped by the train drug name. Both use train treatments with train exact controls only; validation labels never enter either reference.

### New-compound context residual (`val_chem_only`)

| model | n_samples | n_valid_positions | pcc | rmse | direction_accuracy | coverage |
|---|---|---|---|---|---|---|
| Matched Control | 979 | 3564377 | 0.1880 | 0.3808 | 0.5636 | 0.9964 |
| V2 batch-enabled Huber | 979 | 3564377 | 0.1978 | 0.6188 | 0.5566 | 0.9964 |
| V2 no-batch correct | 979 | 3564377 | 0.1021 | 1.0695 | 0.5300 | 0.9964 |
| V2 no-batch chemical zero | 979 | 3564377 | 0.1127 | 0.9702 | 0.5281 | 0.9964 |
| V2 no-batch chemical shuffle | 979 | 3564377 | 0.1006 | 1.1014 | 0.5258 | 0.9964 |
| V2 no-batch Morgan-only | 979 | 3564377 | 0.1155 | 0.9526 | 0.5346 | 0.9964 |
| V2 experimental parity FC Morgan no-batch | 979 | 3564377 | 0.1143 | 0.9885 | 0.5291 | 0.9964 |

### New-strain drug residual (`val_strain_only`)

| model | n_samples | n_valid_positions | pcc | rmse | direction_accuracy | coverage |
|---|---|---|---|---|---|---|
| Matched Control | 1313 | 4824118 | 0.1372 | 0.4101 | 0.5370 | 1.0000 |
| V2 batch-enabled Huber | 1313 | 4824118 | 0.2839 | 0.7368 | 0.5979 | 1.0000 |
| V2 no-batch correct | 1313 | 4824118 | 0.2579 | 0.8136 | 0.5880 | 1.0000 |
| V2 no-batch chemical zero | 1313 | 4824118 | 0.2548 | 0.8450 | 0.5825 | 1.0000 |
| V2 no-batch chemical shuffle | 1313 | 4824118 | 0.2697 | 0.7958 | 0.5924 | 1.0000 |
| V2 no-batch Morgan-only | 1313 | 4824118 | 0.2292 | 0.9194 | 0.5828 | 1.0000 |
| V2 experimental parity FC Morgan no-batch | 1313 | 4824118 | 0.2688 | 0.7940 | 0.5914 | 1.0000 |

## High-effect proteins and DEP

The event threshold is strictly `abs(delta_true) > 1`; equality is not positive. Predicted positives use the same strict magnitude threshold. AUPRC uses `abs(delta_pred)` as the continuous score.

| model | scenario | n_true_high_effect | direction_accuracy | high_effect_pcc | precision | recall | f1 | auprc |
|---|---|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | 85691 | 0.0000 | NA | NA | 0.0000 | NA | 0.0210 |
| V2 batch-enabled Huber | val_chem_only | 85691 | 0.8110 | 0.6081 | 0.0857 | 0.3198 | 0.1352 | 0.0808 |
| V2 no-batch correct | val_chem_only | 85691 | 0.6934 | 0.4662 | 0.0394 | 0.4808 | 0.0728 | 0.0521 |
| V2 no-batch chemical zero | val_chem_only | 85691 | 0.7027 | 0.5159 | 0.0416 | 0.4635 | 0.0764 | 0.0617 |
| V2 no-batch chemical shuffle | val_chem_only | 85691 | 0.6852 | 0.4748 | 0.0374 | 0.5062 | 0.0697 | 0.0564 |
| V2 no-batch Morgan-only | val_chem_only | 85691 | 0.6970 | 0.4869 | 0.0461 | 0.4498 | 0.0837 | 0.0568 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 85691 | 0.6943 | 0.5052 | 0.0412 | 0.4713 | 0.0758 | 0.0605 |
| Matched Control | val_strain_only | 144037 | 0.0000 | NA | NA | 0.0000 | NA | 0.0262 |
| V2 batch-enabled Huber | val_strain_only | 144037 | 0.7671 | 0.5105 | 0.0859 | 0.3418 | 0.1373 | 0.0752 |
| V2 no-batch correct | val_strain_only | 144037 | 0.7565 | 0.5309 | 0.0640 | 0.3526 | 0.1084 | 0.0620 |
| V2 no-batch chemical zero | val_strain_only | 144037 | 0.7521 | 0.5202 | 0.0607 | 0.3884 | 0.1050 | 0.0726 |
| V2 no-batch chemical shuffle | val_strain_only | 144037 | 0.7683 | 0.5446 | 0.0666 | 0.3695 | 0.1129 | 0.0713 |
| V2 no-batch Morgan-only | val_strain_only | 144037 | 0.7518 | 0.4726 | 0.0599 | 0.3755 | 0.1034 | 0.0551 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 144037 | 0.7743 | 0.5542 | 0.0665 | 0.3683 | 0.1126 | 0.0722 |
| Matched Control | val_both | 23953 | 0.0000 | NA | NA | 0.0000 | NA | 0.0212 |
| V2 batch-enabled Huber | val_both | 23953 | 0.7737 | 0.5391 | 0.0554 | 0.3735 | 0.0965 | 0.0535 |
| V2 no-batch correct | val_both | 23953 | 0.6781 | 0.4359 | 0.0380 | 0.4717 | 0.0703 | 0.0488 |
| V2 no-batch chemical zero | val_both | 23953 | 0.7011 | 0.5126 | 0.0419 | 0.4414 | 0.0765 | 0.0601 |
| V2 no-batch chemical shuffle | val_both | 23953 | 0.6833 | 0.4716 | 0.0389 | 0.4864 | 0.0720 | 0.0572 |
| V2 no-batch Morgan-only | val_both | 23953 | 0.6828 | 0.4317 | 0.0390 | 0.4657 | 0.0720 | 0.0448 |
| V2 experimental parity FC Morgan no-batch | val_both | 23953 | 0.6942 | 0.5015 | 0.0412 | 0.4554 | 0.0755 | 0.0577 |
| Matched Control | val_time | 16926 | 0.0000 | NA | NA | 0.0000 | NA | 0.0318 |
| V2 batch-enabled Huber | val_time | 16926 | 0.7555 | 0.5134 | 0.1675 | 0.2418 | 0.1979 | 0.1162 |
| V2 no-batch correct | val_time | 16926 | 0.7438 | 0.4970 | 0.0798 | 0.2992 | 0.1260 | 0.0747 |
| V2 no-batch chemical zero | val_time | 16926 | 0.7109 | 0.4330 | 0.0682 | 0.3388 | 0.1135 | 0.0723 |
| V2 no-batch chemical shuffle | val_time | 16926 | 0.7277 | 0.4673 | 0.0710 | 0.3136 | 0.1158 | 0.0729 |
| V2 no-batch Morgan-only | val_time | 16926 | 0.7704 | 0.5328 | 0.0883 | 0.2715 | 0.1333 | 0.0735 |
| V2 experimental parity FC Morgan no-batch | val_time | 16926 | 0.7405 | 0.4895 | 0.0717 | 0.3186 | 0.1171 | 0.0749 |

## Required attribution answers

1. **Absolute fidelity:** Matched Control by the declared seven-metric common-subset rank.
2. **Raw FC:** V2 batch-enabled Huber by the declared five-metric common-subset rank.
3. **New-compound context residual:** V2 batch-enabled Huber by the official-rule primary PCC; Matched Control has lower RMSE, so the win is not metric-unanimous.
4. **New-strain drug residual:** V2 batch-enabled Huber by PCC.
5. **Batch attribution:** `V2 batch-enabled Huber` beats `V2 no-batch correct` not only on absolute error but also on raw-FC PCC/RMSE/direction and both train-referenced residual modules. The batch advantage is therefore not merely an RMSE-only effect in this audit, although the batch component's biological legitimacy remains a separate structural concern.
6. **Morgan attribution:** `V2 no-batch Morgan-only` modestly improves `val_chem_only` residual PCC/RMSE/direction and most chem-only FC diagnostics versus `V2 no-batch chemical zero`, but loses badly on `val_both` FC/absolute metrics. Morgan supplies some new-compound signal but is not a robust overall chemical solution.
7. **Experimental FC attribution:** `V2 experimental parity FC Morgan no-batch` worsens chem-only FC/context metrics versus `V2 no-batch Morgan-only`, improves some strain/both metrics, and worsens time. It is mixed rather than a stable main-module improvement and remains a nonofficial loss-weight diagnostic.
8. **Matched Control:** no learned model stably exceeds it. Matched Control wins every absolute-fidelity scenario, while learned models only exceed its zero-delta baseline on perturbation modules.
9. **V1 vs V2:** no valid competition-module V2-over-V1 claim is possible because V1 lacks a recoverable checkpoint/predictions. Historical absolute-only aggregates show mixed scenario behavior and cannot substitute for the paired common-subset audit.
10. **Next training target:** prioritize the 25% matched-control raw-FC module, with the 20% new-compound context-residual module as the coupled guardrail. The concrete target is a control-anchored delta predictor that raises FC/context PCC without losing Matched Control's absolute fidelity.

## Planning proxy and ranking sensitivity

`official_score=false`. The three schemes use the official module weights but different reasonable internal aggregation choices because the organizer formula is unavailable.

| scheme | rank | model | planning_proxy | official_score |
|---|---|---|---|---|
| bounded_quality | 1 | V2 batch-enabled Huber | 0.651959 | False |
| bounded_quality | 2 | V2 experimental parity FC Morgan no-batch | 0.601582 | False |
| bounded_quality | 3 | V2 no-batch Morgan-only | 0.601220 | False |
| bounded_quality | 4 | V2 no-batch correct | 0.599007 | False |
| bounded_quality | 5 | V2 no-batch chemical zero | 0.596654 | False |
| bounded_quality | 6 | V2 no-batch chemical shuffle | 0.594576 | False |
| bounded_quality | 7 | Matched Control | 0.499980 | False |
| correlation_priority | 1 | V2 batch-enabled Huber | 0.686002 | False |
| correlation_priority | 2 | V2 no-batch Morgan-only | 0.636923 | False |
| correlation_priority | 3 | V2 no-batch correct | 0.633401 | False |
| correlation_priority | 4 | V2 experimental parity FC Morgan no-batch | 0.633137 | False |
| correlation_priority | 5 | V2 no-batch chemical zero | 0.628119 | False |
| correlation_priority | 6 | V2 no-batch chemical shuffle | 0.626671 | False |
| correlation_priority | 7 | Matched Control | 0.483181 | False |
| weighted_module_rank | 1 | V2 batch-enabled Huber | 0.933333 | False |
| weighted_module_rank | 2 | V2 no-batch Morgan-only | 0.583333 | False |
| weighted_module_rank | 3 | Matched Control | 0.566667 | False |
| weighted_module_rank | 4 | V2 experimental parity FC Morgan no-batch | 0.508333 | False |
| weighted_module_rank | 5 | V2 no-batch correct | 0.366667 | False |
| weighted_module_rank | 6 | V2 no-batch chemical zero | 0.333333 | False |
| weighted_module_rank | 7 | V2 no-batch chemical shuffle | 0.208333 | False |

Winner stable across proxy schemes: **True**. No proxy value is a leaderboard score.

## Compliance verification

- Test proteome opened: **false**. The source contains no executable path to that file; only train/validation metadata and proteome are loaded.
- Training/backward/optimizer step: **none**. Checkpoints are restored under `torch.inference_mode()`.
- Validation labels in fitted statistics: **none**. Residual references enforce `split_final=train` IDs.
- Pairwise mask: **verified** for treatment truth, observed control, prediction finiteness, and control finiteness.
- Scale: **log2**, verified from every checkpoint and from the raw-to-log2 loader.
- `sample_ID` and protein order: metadata is indexed by `sample_ID`; protein order comes from the frozen feature contract and checkpoint artifact-hash equality is required.
- Parity labeling: every machine-readable output carries `official_score=false` and/or the nonofficial parity flags.

## Failures and Main decisions

- NOT_EVALUABLE: V1 ConditionMLP.
- Official Water/DMSO mapping, sample-level QC, and duplicate-control aggregation remain organizer/Main decisions.
- The official within-module aggregation and normalization formula remains unavailable.
- This audit stops here. It did not train or generate test predictions.
