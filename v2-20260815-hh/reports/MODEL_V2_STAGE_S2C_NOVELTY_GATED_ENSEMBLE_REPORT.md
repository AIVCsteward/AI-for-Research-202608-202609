# MODEL V2 Stage S2C Novelty-Gated Ensemble Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

No model was trained, fine-tuned, calibrated or modified. Routing was materialized before validation scoring from sets fitted only on train metadata. The validation routing manifest SHA-256 is `f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4`.

## Fixed routing and scenario derivation

Priority: control -> unseen chemical -> unseen strain -> seen chemical and strain. Controls and all unseen chemicals use D0. Seen chemical plus unseen strain uses S1 C only for a train-seen batch tuple; otherwise it uses D2 with nested hierarchical fallback. Seen chemical plus seen strain uses D2. No drug-name, validation-error, Tanimoto or effect-size rule exists.

| scenario | n_treatment | chemical_seen_expected | strain_seen_expected | chemical_seen_mismatch_count | strain_seen_mismatch_count | time_seen_count | culture_context_seen_count | full_entity_context_seen_count |
|---|---|---|---|---|---|---|---|---|
| val_chem_only | 1065 | False | True | 0 | 0 | 1065 | 1065 | 0 |
| val_strain_only | 1333 | True | False | 0 | 0 | 1333 | 1333 | 0 |
| val_both | 269 | False | False | 0 | 0 | 269 | 269 | 0 |
| val_time | 139 | True | True | 0 | 0 | 139 | 139 | 52 |

Validation expert counts: `{"S2B D0 hierarchical none Huber": 1566, "S1 C anchor Morgan Huber plus experimental FC": 1333, "S2B D2 hierarchical Morgan experimental FC": 139}`.

## Seven absolute metrics on the exact S0 common subset

| model | scenario | n_samples | log2_rmse | mae | global_r2 | sample_pcc_median | sample_r2_median | protein_pcc_median | protein_r2_median |
|---|---|---|---|---|---|---|---|---|---|
| V2 batch-enabled Huber | val_chem_only | 981 | 0.6196 | 0.4496 | 0.9483 | 0.9804 | 0.9561 | 0.8169 | 0.5494 |
| Matched Control | val_chem_only | 981 | 0.3817 | 0.2543 | 0.9804 | 0.9924 | 0.9834 | 0.9163 | 0.8279 |
| V2 batch-enabled Huber | val_strain_only | 1313 | 0.7368 | 0.5126 | 0.9253 | 0.9653 | 0.9278 | 0.7714 | 0.3476 |
| Matched Control | val_strain_only | 1313 | 0.4101 | 0.2735 | 0.9769 | 0.9911 | 0.9811 | 0.8573 | 0.7037 |
| V2 batch-enabled Huber | val_both | 266 | 0.8315 | 0.5945 | 0.9068 | 0.9621 | 0.9098 | 0.8115 | 0.3373 |
| Matched Control | val_both | 266 | 0.3853 | 0.2570 | 0.9800 | 0.9918 | 0.9825 | 0.9061 | 0.8018 |
| V2 batch-enabled Huber | val_time | 134 | 0.5563 | 0.3819 | 0.9580 | 0.9820 | 0.9620 | 0.7581 | 0.5483 |
| Matched Control | val_time | 134 | 0.4336 | 0.2843 | 0.9745 | 0.9917 | 0.9816 | 0.8549 | 0.7066 |
| S1 B anchor Morgan Huber | val_chem_only | 981 | 0.5965 | 0.4164 | 0.9521 | 0.9801 | 0.9535 | 0.7991 | 0.6114 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 981 | 0.6141 | 0.4321 | 0.9492 | 0.9778 | 0.9510 | 0.8090 | 0.5903 |
| S1 B anchor Morgan Huber | val_strain_only | 1313 | 0.7275 | 0.5072 | 0.9272 | 0.9667 | 0.9310 | 0.7682 | 0.3637 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 1313 | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| S1 B anchor Morgan Huber | val_both | 266 | 0.7985 | 0.5608 | 0.9140 | 0.9604 | 0.9132 | 0.7865 | 0.3895 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 266 | 0.8054 | 0.5661 | 0.9125 | 0.9608 | 0.9116 | 0.8052 | 0.3986 |
| S1 B anchor Morgan Huber | val_time | 134 | 0.5546 | 0.3853 | 0.9583 | 0.9824 | 0.9616 | 0.7561 | 0.5445 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 134 | 0.5402 | 0.3740 | 0.9604 | 0.9839 | 0.9648 | 0.7652 | 0.5648 |
| S2B D0 hierarchical none Huber | val_chem_only | 981 | 0.5478 | 0.3862 | 0.9596 | 0.9834 | 0.9636 | 0.8362 | 0.6641 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 981 | 0.5794 | 0.4075 | 0.9548 | 0.9817 | 0.9585 | 0.8146 | 0.6282 |
| S2B D0 hierarchical none Huber | val_strain_only | 1313 | 0.7653 | 0.5283 | 0.9194 | 0.9626 | 0.9209 | 0.7559 | 0.3247 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 1313 | 0.7529 | 0.5190 | 0.9220 | 0.9638 | 0.9240 | 0.7795 | 0.3732 |
| S2B D0 hierarchical none Huber | val_both | 266 | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 266 | 0.8114 | 0.5650 | 0.9112 | 0.9614 | 0.9135 | 0.8039 | 0.4159 |
| S2B D0 hierarchical none Huber | val_time | 134 | 0.5496 | 0.3812 | 0.9590 | 0.9834 | 0.9619 | 0.7599 | 0.5551 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 134 | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |
| S2C novelty-gated ensemble | val_chem_only | 981 | 0.5478 | 0.3862 | 0.9596 | 0.9834 | 0.9636 | 0.8362 | 0.6641 |
| S2C novelty-gated ensemble | val_strain_only | 1313 | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| S2C novelty-gated ensemble | val_both | 266 | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| S2C novelty-gated ensemble | val_time | 134 | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |

## Raw FC and high-effect metrics

| model | scenario | global_fc_pcc | sample_fc_pcc_median | protein_fc_pcc_median | fc_rmse | fc_direction_accuracy |
|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | NA | NA | NA | 0.3817 | 0.0000 |
| V2 batch-enabled Huber | val_chem_only | 0.3072 | 0.3000 | 0.3460 | 0.6196 | 0.5976 |
| Matched Control | val_strain_only | NA | NA | NA | 0.4101 | 0.0000 |
| V2 batch-enabled Huber | val_strain_only | 0.2443 | 0.2344 | 0.4034 | 0.7368 | 0.5873 |
| Matched Control | val_both | NA | NA | NA | 0.3853 | 0.0000 |
| V2 batch-enabled Huber | val_both | 0.2257 | 0.2086 | 0.3726 | 0.8315 | 0.5720 |
| Matched Control | val_time | NA | NA | NA | 0.4336 | 0.0000 |
| V2 batch-enabled Huber | val_time | 0.3139 | 0.3182 | 0.3363 | 0.5563 | 0.6096 |
| S1 B anchor Morgan Huber | val_chem_only | 0.3042 | 0.3062 | 0.3333 | 0.5965 | 0.6038 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.3106 | 0.3095 | 0.3494 | 0.6141 | 0.6076 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2492 | 0.2404 | 0.3941 | 0.7275 | 0.5883 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| S1 B anchor Morgan Huber | val_both | 0.2224 | 0.2133 | 0.3512 | 0.7985 | 0.5738 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 0.2306 | 0.2181 | 0.3712 | 0.8054 | 0.5784 |
| S1 B anchor Morgan Huber | val_time | 0.3214 | 0.3245 | 0.3323 | 0.5546 | 0.6100 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 0.3493 | 0.3507 | 0.3600 | 0.5402 | 0.6197 |
| S2B D0 hierarchical none Huber | val_chem_only | 0.3274 | 0.3281 | 0.3529 | 0.5478 | 0.6105 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.3235 | 0.3278 | 0.3461 | 0.5794 | 0.6099 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.2240 | 0.2232 | 0.3795 | 0.7653 | 0.5788 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.2495 | 0.2393 | 0.4203 | 0.7529 | 0.5888 |
| S2B D0 hierarchical none Huber | val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.2189 | 0.2089 | 0.3615 | 0.8114 | 0.5731 |
| S2B D0 hierarchical none Huber | val_time | 0.3149 | 0.3327 | 0.3258 | 0.5496 | 0.6055 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |
| S2C novelty-gated ensemble | val_chem_only | 0.3274 | 0.3281 | 0.3529 | 0.5478 | 0.6105 |
| S2C novelty-gated ensemble | val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| S2C novelty-gated ensemble | val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| S2C novelty-gated ensemble | val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |

| model | scenario | direction_accuracy | high_effect_pcc | precision | recall | f1 | auprc |
|---|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | 0.0000 | NA | NA | 0.0000 | NA | 0.0210 |
| V2 batch-enabled Huber | val_chem_only | 0.8110 | 0.6081 | 0.0857 | 0.3198 | 0.1352 | 0.0808 |
| Matched Control | val_strain_only | 0.0000 | NA | NA | 0.0000 | NA | 0.0262 |
| V2 batch-enabled Huber | val_strain_only | 0.7671 | 0.5105 | 0.0859 | 0.3418 | 0.1373 | 0.0752 |
| Matched Control | val_both | 0.0000 | NA | NA | 0.0000 | NA | 0.0212 |
| V2 batch-enabled Huber | val_both | 0.7737 | 0.5391 | 0.0554 | 0.3735 | 0.0965 | 0.0535 |
| Matched Control | val_time | 0.0000 | NA | NA | 0.0000 | NA | 0.0318 |
| V2 batch-enabled Huber | val_time | 0.7555 | 0.5134 | 0.1675 | 0.2418 | 0.1979 | 0.1162 |
| S1 B anchor Morgan Huber | val_chem_only | 0.7875 | 0.6038 | 0.1030 | 0.3290 | 0.1569 | 0.0909 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.8053 | 0.6143 | 0.0933 | 0.3391 | 0.1463 | 0.0842 |
| S1 B anchor Morgan Huber | val_strain_only | 0.7723 | 0.5232 | 0.0874 | 0.3426 | 0.1393 | 0.0755 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.7853 | 0.5433 | 0.0891 | 0.3398 | 0.1412 | 0.0764 |
| S1 B anchor Morgan Huber | val_both | 0.7692 | 0.5403 | 0.0631 | 0.3730 | 0.1080 | 0.0589 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 0.7788 | 0.5441 | 0.0610 | 0.3748 | 0.1049 | 0.0568 |
| S1 B anchor Morgan Huber | val_time | 0.7587 | 0.5350 | 0.1662 | 0.2480 | 0.1990 | 0.1173 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 0.7883 | 0.5727 | 0.1775 | 0.2592 | 0.2107 | 0.1282 |
| S2B D0 hierarchical none Huber | val_chem_only | 0.8080 | 0.6190 | 0.1166 | 0.2923 | 0.1667 | 0.0964 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.8121 | 0.6251 | 0.1061 | 0.3244 | 0.1599 | 0.0927 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.7479 | 0.4821 | 0.0841 | 0.3435 | 0.1351 | 0.0715 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.7836 | 0.5341 | 0.0882 | 0.3651 | 0.1421 | 0.0758 |
| S2B D0 hierarchical none Huber | val_both | 0.7607 | 0.5142 | 0.0627 | 0.3584 | 0.1068 | 0.0552 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.7728 | 0.5347 | 0.0624 | 0.3711 | 0.1068 | 0.0567 |
| S2B D0 hierarchical none Huber | val_time | 0.7591 | 0.5234 | 0.1704 | 0.2327 | 0.1968 | 0.1147 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.8316 | 0.6615 | 0.2233 | 0.2914 | 0.2529 | 0.1651 |
| S2C novelty-gated ensemble | val_chem_only | 0.8080 | 0.6190 | 0.1166 | 0.2923 | 0.1667 | 0.0964 |
| S2C novelty-gated ensemble | val_strain_only | 0.7853 | 0.5433 | 0.0891 | 0.3398 | 0.1412 | 0.0764 |
| S2C novelty-gated ensemble | val_both | 0.7607 | 0.5142 | 0.0627 | 0.3584 | 0.1068 | 0.0552 |
| S2C novelty-gated ensemble | val_time | 0.8316 | 0.6615 | 0.2233 | 0.2914 | 0.2529 | 0.1651 |

## Context and drug residual modules

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_chem_only | 0.1880 | 0.3808 | 0.5636 |
| V2 batch-enabled Huber | val_chem_only | 0.1978 | 0.6188 | 0.5566 |
| S1 B anchor Morgan Huber | val_chem_only | 0.2024 | 0.5954 | 0.5672 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.1973 | 0.6132 | 0.5639 |
| S2B D0 hierarchical none Huber | val_chem_only | 0.2146 | 0.5468 | 0.5678 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.2081 | 0.5786 | 0.5657 |
| S2C novelty-gated ensemble | val_chem_only | 0.2146 | 0.5468 | 0.5678 |

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_strain_only | 0.1372 | 0.4101 | 0.5370 |
| V2 batch-enabled Huber | val_strain_only | 0.2839 | 0.7368 | 0.5979 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2864 | 0.7275 | 0.5974 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.2914 | 0.7180 | 0.6005 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.2654 | 0.7653 | 0.5893 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.2554 | 0.7529 | 0.5890 |
| S2C novelty-gated ensemble | val_strain_only | 0.2914 | 0.7180 | 0.6005 |

## Ensemble scenario summary

| scenario | log2_rmse | mae | global_r2 | sample_pcc_median | sample_r2_median | protein_pcc_median | protein_r2_median |
|---|---|---|---|---|---|---|---|
| val_chem_only | 0.5478 | 0.3862 | 0.9596 | 0.9834 | 0.9636 | 0.8362 | 0.6641 |
| val_strain_only | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| val_both | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| val_time | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |

| scenario | global_fc_pcc | sample_fc_pcc_median | protein_fc_pcc_median | fc_rmse | fc_direction_accuracy |
|---|---|---|---|---|---|
| val_chem_only | 0.3274 | 0.3281 | 0.3529 | 0.5478 | 0.6105 |
| val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |

## Planning proxies

All three S0 proxy schemes remain nonofficial. The composite table is ranked across Matched Control and the six deployable learning models.

| scheme | rank | model | planning_proxy | official_score |
|---|---|---|---|---|
| bounded_quality | 1 | S2C novelty-gated ensemble | 0.6631 | False |
| bounded_quality | 2 | S2B D2 hierarchical Morgan experimental FC | 0.6581 | False |
| bounded_quality | 3 | S1 C anchor Morgan Huber plus experimental FC | 0.6568 | False |
| bounded_quality | 4 | S1 B anchor Morgan Huber | 0.6545 | False |
| bounded_quality | 5 | S2B D0 hierarchical none Huber | 0.6542 | False |
| bounded_quality | 6 | V2 batch-enabled Huber | 0.6520 | False |
| bounded_quality | 7 | Matched Control | 0.5000 | False |
| correlation_priority | 1 | S2C novelty-gated ensemble | 0.6958 | False |
| correlation_priority | 2 | S2B D2 hierarchical Morgan experimental FC | 0.6914 | False |
| correlation_priority | 3 | S1 C anchor Morgan Huber plus experimental FC | 0.6908 | False |
| correlation_priority | 4 | S1 B anchor Morgan Huber | 0.6878 | False |
| correlation_priority | 5 | S2B D0 hierarchical none Huber | 0.6872 | False |
| correlation_priority | 6 | V2 batch-enabled Huber | 0.6860 | False |
| correlation_priority | 7 | Matched Control | 0.4832 | False |
| weighted_module_rank | 1 | S2C novelty-gated ensemble | 0.8417 | False |
| weighted_module_rank | 2 | S2B D2 hierarchical Morgan experimental FC | 0.5917 | False |
| weighted_module_rank | 3 | S1 C anchor Morgan Huber plus experimental FC | 0.5667 | False |
| weighted_module_rank | 4 | Matched Control | 0.4667 | False |
| weighted_module_rank | 5 | S1 B anchor Morgan Huber | 0.4333 | False |
| weighted_module_rank | 6 | S2B D0 hierarchical none Huber | 0.4333 | False |
| weighted_module_rank | 7 | V2 batch-enabled Huber | 0.1667 | False |

Scenario-specific ensemble planning proxies:

| scenario | scheme | rank | planning_proxy | official_score |
|---|---|---|---|---|
| val_chem_only | bounded_quality | 1 | 0.4848 | False |
| val_chem_only | correlation_priority | 1 | 0.4926 | False |
| val_chem_only | weighted_module_rank | 3 | 0.5583 | False |
| val_strain_only | bounded_quality | 1 | 0.4718 | False |
| val_strain_only | correlation_priority | 1 | 0.4861 | False |
| val_strain_only | weighted_module_rank | 4 | 0.5583 | False |
| val_both | bounded_quality | 4 | 0.3996 | False |
| val_both | correlation_priority | 2 | 0.4390 | False |
| val_both | weighted_module_rank | 7 | 0.2083 | False |
| val_time | bounded_quality | 1 | 0.4402 | False |
| val_time | correlation_priority | 1 | 0.4694 | False |
| val_time | weighted_module_rank | 5 | 0.4750 | False |

All scenario/model proxy rows are in `scenario_planning_proxy.csv`; the full machine-readable module inputs are in `novelty_gated_metrics.json`.

## OOV and test-metadata audit

Validation contains no unseen train batch tuple, so no flat expert receives OOV batches and no validation fallback is activated. Automated tests exercise unseen plate, unseen instrument and unseen source. Test metadata is routed for counts only: `{"S2B D0 hierarchical none Huber": 2997, "S1 C anchor Morgan Huber plus experimental FC": 1322, "S2B D2 hierarchical Morgan experimental FC": 135}`; OOV tuple count `0`. No test prediction was generated and no held-out protein truth was read.

## Predeclared gates

```json
{
  "gate_1_composite_planning_proxy_rank_first_all_three_schemes": true,
  "gate_2_mean_absolute_rmse_better_than_d2": true,
  "gate_3_mean_raw_fc_pcc_drop_vs_d2_at_most_0_005": true,
  "gate_4_val_chem_context_residual_pcc_at_least_d0": true,
  "gate_5_val_strain_drug_residual_pcc_at_least_s1c": true,
  "gate_6_val_time_raw_fc_pcc_at_least_d2": true,
  "gate_7_routes_fit_from_train_metadata_only": true,
  "gate_8_no_drug_specific_rule": true,
  "gate_9_no_oov_batch_routed_to_flat_model": true
}
```

Eligible for the next authorized step (second-seed stability validation of the three selected checkpoints): **True**. This does not authorize a test submission.

The ensemble mean absolute RMSE is `0.645840` versus D2 `0.663276`. Mean raw-FC PCC is `0.298976` versus D2 `0.297493`. By construction and verified equality, val_chem context residual equals D0, val_strain drug residual equals S1 C, and val_time raw-FC equals D2.

## Artifacts and execution

- Checkpoint paths, hashes, resolved configs and seeds: `checkpoint_manifest.json`.
- Per-sample validation expert and reason: `validation_routing.csv`.
- Test metadata route counts only: `test_metadata_routing_counts.json`.
- No training, backward pass, optimizer, checkpoint write, output calibration or mixture-weight search occurred.
- Inference command: `D:\虚拟细胞\.venv\Scripts\python.exe -m baseline.novelty_gated_ensemble_v2`.
- All V2 tests: `python -m pytest --import-mode=importlib <all baseline/tests/test_*.py except test_person_c.py> -q`; `103 passed in 54.89s`.
- Person C regression from `D:\虚拟细胞\baseline`: `python -m pytest tests/test_person_c.py -q`; `6 passed in 3.86s`.
- The first inference audit stopped before artifact writing because existing D0 treats Quality Control as a non-control internally. The ensemble-only control constraint was then made explicit: Water, DMSO and Quality Control all use D0 anchor output with response exactly zero. No checkpoint or model parameter was changed, and the completed rerun passed.

Stage S2C is paused for Main review.
