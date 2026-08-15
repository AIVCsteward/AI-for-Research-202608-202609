# MODEL V2 Stage S2B Hierarchical Treatment Report

Status: COMPLETE_AND_PAUSED  
Scoring labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

All three models loaded the unchanged S2A hierarchical Stage-A checkpoint `99f997f815eec95bde70f78afcc533f649b7519435f04b8ef207da90c60f7e73`. Stage A was not retrained. D0/D1/D2 share Stage-B initialization hash `b1d6d40fc41679e442b887f31e2467591fc80bd17020a68bcb3c0edd16fe00ff`, seed 20260814, 4,121 training treatments, the same 957-sample technical-tuple holdout, model dimensions, order, optimizer settings and four-scenario macro-Huber monitor. Baseline, genome, source, instrument and plate parameters are frozen exactly in every run.

| model | device | elapsed_seconds | peak_reserved_mib | stage_b_actual_epochs | stage_b_best_epoch | stage_b_best_macro_huber | stop_reason |
|---|---|---|---|---|---|---|---|
| S2B D0 hierarchical none Huber | cuda | 103.6736 | 90.0000 | 33 | 20 | 0.1947 | early_stopping_patience_exhausted |
| S2B D1 hierarchical Morgan Huber | cuda | 75.3076 | 90.0000 | 24 | 11 | 0.2012 | early_stopping_patience_exhausted |
| S2B D2 hierarchical Morgan experimental FC | cuda | 99.0270 | 90.0000 | 25 | 12 | 0.1941 | early_stopping_patience_exhausted |

## Absolute metrics on the exact S0 common subset

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
| S2B D1 hierarchical Morgan Huber | val_chem_only | 981 | 0.6079 | 0.4270 | 0.9502 | 0.9797 | 0.9532 | 0.8018 | 0.6020 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 981 | 0.5794 | 0.4075 | 0.9548 | 0.9817 | 0.9585 | 0.8146 | 0.6282 |
| S2B D0 hierarchical none Huber | val_strain_only | 1313 | 0.7653 | 0.5283 | 0.9194 | 0.9626 | 0.9209 | 0.7559 | 0.3247 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 1313 | 0.7499 | 0.5161 | 0.9226 | 0.9640 | 0.9243 | 0.7834 | 0.3866 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 1313 | 0.7529 | 0.5190 | 0.9220 | 0.9638 | 0.9240 | 0.7795 | 0.3732 |
| S2B D0 hierarchical none Huber | val_both | 266 | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| S2B D1 hierarchical Morgan Huber | val_both | 266 | 0.8402 | 0.5873 | 0.9048 | 0.9599 | 0.9076 | 0.7868 | 0.3642 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 266 | 0.8114 | 0.5650 | 0.9112 | 0.9614 | 0.9135 | 0.8039 | 0.4159 |
| S2B D0 hierarchical none Huber | val_time | 134 | 0.5496 | 0.3812 | 0.9590 | 0.9834 | 0.9619 | 0.7599 | 0.5551 |
| S2B D1 hierarchical Morgan Huber | val_time | 134 | 0.5105 | 0.3546 | 0.9647 | 0.9858 | 0.9693 | 0.7975 | 0.6181 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 134 | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |

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
| S2B D1 hierarchical Morgan Huber | val_chem_only | 0.3060 | 0.3151 | 0.3365 | 0.6079 | 0.6042 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.3235 | 0.3278 | 0.3461 | 0.5794 | 0.6099 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.2240 | 0.2232 | 0.3795 | 0.7653 | 0.5788 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 0.2427 | 0.2320 | 0.4142 | 0.7499 | 0.5873 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.2495 | 0.2393 | 0.4203 | 0.7529 | 0.5888 |
| S2B D0 hierarchical none Huber | val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| S2B D1 hierarchical Morgan Huber | val_both | 0.2108 | 0.2025 | 0.3487 | 0.8402 | 0.5685 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.2189 | 0.2089 | 0.3615 | 0.8114 | 0.5731 |
| S2B D0 hierarchical none Huber | val_time | 0.3149 | 0.3327 | 0.3258 | 0.5496 | 0.6055 |
| S2B D1 hierarchical Morgan Huber | val_time | 0.3789 | 0.3726 | 0.3878 | 0.5105 | 0.6209 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |

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
| S2B D1 hierarchical Morgan Huber | val_chem_only | 0.8031 | 0.6085 | 0.0965 | 0.3297 | 0.1493 | 0.0846 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.8121 | 0.6251 | 0.1061 | 0.3244 | 0.1599 | 0.0927 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.7479 | 0.4821 | 0.0841 | 0.3435 | 0.1351 | 0.0715 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 0.7732 | 0.5179 | 0.0876 | 0.3532 | 0.1404 | 0.0745 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.7836 | 0.5341 | 0.0882 | 0.3651 | 0.1421 | 0.0758 |
| S2B D0 hierarchical none Huber | val_both | 0.7607 | 0.5142 | 0.0627 | 0.3584 | 0.1068 | 0.0552 |
| S2B D1 hierarchical Morgan Huber | val_both | 0.7648 | 0.5254 | 0.0590 | 0.3798 | 0.1022 | 0.0550 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.7728 | 0.5347 | 0.0624 | 0.3711 | 0.1068 | 0.0567 |
| S2B D0 hierarchical none Huber | val_time | 0.7591 | 0.5234 | 0.1704 | 0.2327 | 0.1968 | 0.1147 |
| S2B D1 hierarchical Morgan Huber | val_time | 0.8222 | 0.6395 | 0.2143 | 0.2623 | 0.2359 | 0.1515 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.8316 | 0.6615 | 0.2233 | 0.2914 | 0.2529 | 0.1651 |

## Context and drug residual modules

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_chem_only | 0.1880 | 0.3808 | 0.5636 |
| V2 batch-enabled Huber | val_chem_only | 0.1978 | 0.6188 | 0.5566 |
| S1 B anchor Morgan Huber | val_chem_only | 0.2024 | 0.5954 | 0.5672 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.1973 | 0.6132 | 0.5639 |
| S2B D0 hierarchical none Huber | val_chem_only | 0.2146 | 0.5468 | 0.5678 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | 0.1995 | 0.6072 | 0.5634 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.2081 | 0.5786 | 0.5657 |

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_strain_only | 0.1372 | 0.4101 | 0.5370 |
| V2 batch-enabled Huber | val_strain_only | 0.2839 | 0.7368 | 0.5979 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2864 | 0.7275 | 0.5974 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.2914 | 0.7180 | 0.6005 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.2654 | 0.7653 | 0.5893 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 0.2620 | 0.7499 | 0.5922 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.2554 | 0.7529 | 0.5890 |

## Frozen hierarchical components

| model | scenario | delta_source_rms | delta_instrument_rms | delta_plate_rms | delta_batch_rms | delta_response_rms | batch_to_response_norm |
|---|---|---|---|---|---|---|---|
| S2B D0 hierarchical none Huber | val_chem_only | 0.5288 | 0.1739 | 0.0421 | 0.6471 | 0.1960 | 3.3009 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | 0.5288 | 0.1739 | 0.0421 | 0.6471 | 0.3012 | 2.1483 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.5288 | 0.1739 | 0.0421 | 0.6471 | 0.2540 | 2.5472 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.4445 | 0.2060 | 0.0446 | 0.5884 | 0.1829 | 3.2168 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 0.4445 | 0.2060 | 0.0446 | 0.5884 | 0.2988 | 1.9694 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.4445 | 0.2060 | 0.0446 | 0.5884 | 0.2745 | 2.1440 |
| S2B D0 hierarchical none Huber | val_both | 0.5313 | 0.1821 | 0.0428 | 0.6568 | 0.1727 | 3.8026 |
| S2B D1 hierarchical Morgan Huber | val_both | 0.5313 | 0.1821 | 0.0428 | 0.6568 | 0.2739 | 2.3984 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.5313 | 0.1821 | 0.0428 | 0.6568 | 0.2181 | 3.0111 |
| S2B D0 hierarchical none Huber | val_time | 0.4290 | 0.1736 | 0.0458 | 0.5297 | 0.1961 | 2.7013 |
| S2B D1 hierarchical Morgan Huber | val_time | 0.4290 | 0.1736 | 0.0458 | 0.5297 | 0.3180 | 1.6656 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.4290 | 0.1736 | 0.0458 | 0.5297 | 0.2876 | 1.8422 |

The maximum exact OOV semantic error across all models/scenarios is `0`. Unseen plate clears plate only; unseen instrument clears instrument and plate; all-OOV clears total batch.

| model | scenario | batch_mode | absolute_rmse | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|
| S2B D0 hierarchical none Huber | val_chem_only | known | 0.5478 | 0.3274 | 0.5478 |
| S2B D0 hierarchical none Huber | val_chem_only | oov_plate | 0.5489 | 0.3258 | 0.5489 |
| S2B D0 hierarchical none Huber | val_chem_only | oov_instrument | 0.5857 | 0.3049 | 0.5857 |
| S2B D0 hierarchical none Huber | val_chem_only | all_oov | 0.8944 | 0.1963 | 0.8944 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | known | 0.6079 | 0.3060 | 0.6079 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | oov_plate | 0.6086 | 0.3048 | 0.6086 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | oov_instrument | 0.6424 | 0.2888 | 0.6424 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | all_oov | 0.9336 | 0.1962 | 0.9336 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | known | 0.5794 | 0.3235 | 0.5794 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | oov_plate | 0.5801 | 0.3223 | 0.5801 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | oov_instrument | 0.6145 | 0.3044 | 0.6145 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | all_oov | 0.9134 | 0.2028 | 0.9134 |
| S2B D0 hierarchical none Huber | val_strain_only | known | 0.7653 | 0.2240 | 0.7653 |
| S2B D0 hierarchical none Huber | val_strain_only | oov_plate | 0.7645 | 0.2233 | 0.7645 |
| S2B D0 hierarchical none Huber | val_strain_only | oov_instrument | 0.7984 | 0.2151 | 0.7984 |
| S2B D0 hierarchical none Huber | val_strain_only | all_oov | 0.9817 | 0.1788 | 0.9817 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | known | 0.7499 | 0.2427 | 0.7499 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | oov_plate | 0.7493 | 0.2419 | 0.7493 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | oov_instrument | 0.7850 | 0.2327 | 0.7850 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | all_oov | 0.9745 | 0.1918 | 0.9745 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | known | 0.7529 | 0.2495 | 0.7529 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | oov_plate | 0.7522 | 0.2488 | 0.7522 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | oov_instrument | 0.7890 | 0.2391 | 0.7890 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | all_oov | 0.9792 | 0.1973 | 0.9792 |
| S2B D0 hierarchical none Huber | val_both | known | 0.8083 | 0.2090 | 0.8083 |
| S2B D0 hierarchical none Huber | val_both | oov_plate | 0.8066 | 0.2088 | 0.8066 |
| S2B D0 hierarchical none Huber | val_both | oov_instrument | 0.8300 | 0.2031 | 0.8300 |
| S2B D0 hierarchical none Huber | val_both | all_oov | 1.0603 | 0.1510 | 1.0603 |
| S2B D1 hierarchical Morgan Huber | val_both | known | 0.8402 | 0.2108 | 0.8402 |
| S2B D1 hierarchical Morgan Huber | val_both | oov_plate | 0.8387 | 0.2105 | 0.8387 |
| S2B D1 hierarchical Morgan Huber | val_both | oov_instrument | 0.8617 | 0.2051 | 0.8617 |
| S2B D1 hierarchical Morgan Huber | val_both | all_oov | 1.0862 | 0.1549 | 1.0862 |
| S2B D2 hierarchical Morgan experimental FC | val_both | known | 0.8114 | 0.2189 | 0.8114 |
| S2B D2 hierarchical Morgan experimental FC | val_both | oov_plate | 0.8099 | 0.2186 | 0.8099 |
| S2B D2 hierarchical Morgan experimental FC | val_both | oov_instrument | 0.8330 | 0.2128 | 0.8330 |
| S2B D2 hierarchical Morgan experimental FC | val_both | all_oov | 1.0628 | 0.1590 | 1.0628 |
| S2B D0 hierarchical none Huber | val_time | known | 0.5496 | 0.3149 | 0.5496 |
| S2B D0 hierarchical none Huber | val_time | oov_plate | 0.5519 | 0.3140 | 0.5519 |
| S2B D0 hierarchical none Huber | val_time | oov_instrument | 0.5740 | 0.3026 | 0.5740 |
| S2B D0 hierarchical none Huber | val_time | all_oov | 0.7724 | 0.2223 | 0.7724 |
| S2B D1 hierarchical Morgan Huber | val_time | known | 0.5105 | 0.3789 | 0.5105 |
| S2B D1 hierarchical Morgan Huber | val_time | oov_plate | 0.5121 | 0.3778 | 0.5121 |
| S2B D1 hierarchical Morgan Huber | val_time | oov_instrument | 0.5367 | 0.3619 | 0.5367 |
| S2B D1 hierarchical Morgan Huber | val_time | all_oov | 0.7510 | 0.2605 | 0.7510 |
| S2B D2 hierarchical Morgan experimental FC | val_time | known | 0.5093 | 0.3980 | 0.5093 |
| S2B D2 hierarchical Morgan experimental FC | val_time | oov_plate | 0.5111 | 0.3968 | 0.5111 |
| S2B D2 hierarchical Morgan experimental FC | val_time | oov_instrument | 0.5359 | 0.3804 | 0.5359 |
| S2B D2 hierarchical Morgan experimental FC | val_time | all_oov | 0.7532 | 0.2750 | 0.7532 |

## 957-treatment tuple holdout

These labels were used only after best-checkpoint restoration. Stage A control exposure may include the same technical groups, so this remains the predeclared treatment-label holdout rather than wholly new-batch OOD.

| model | n_samples | n_matched_fc_samples | absolute_rmse | absolute_mae | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|---|
| S2B D0 hierarchical none Huber | 957 | 873 | 0.5785 | 0.4025 | 0.3246 | 0.5567 |
| S2B D1 hierarchical Morgan Huber | 957 | 873 | 0.5295 | 0.3636 | 0.3857 | 0.5077 |
| S2B D2 hierarchical Morgan experimental FC | 957 | 873 | 0.5274 | 0.3619 | 0.4019 | 0.5033 |

## Morgan correct/shuffle/zero and drug attribution

| model | scenario | chemical_mode | absolute_rmse | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|
| S2B D1 hierarchical Morgan Huber | val_chem_only | correct | 0.6079 | 0.3060 | 0.6079 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | shuffle | 0.5732 | 0.3150 | 0.5732 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | zero | 0.5487 | 0.3241 | 0.5487 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | correct | 0.5794 | 0.3235 | 0.5794 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | shuffle | 0.5723 | 0.3236 | 0.5723 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | zero | 0.5436 | 0.3326 | 0.5436 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | correct | 0.7499 | 0.2427 | 0.7499 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | shuffle | 0.8182 | 0.2148 | 0.8182 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | zero | 0.7912 | 0.2117 | 0.7912 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | correct | 0.7529 | 0.2495 | 0.7529 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | shuffle | 0.8136 | 0.2139 | 0.8136 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | zero | 0.7668 | 0.2245 | 0.7668 |
| S2B D1 hierarchical Morgan Huber | val_both | correct | 0.8402 | 0.2108 | 0.8402 |
| S2B D1 hierarchical Morgan Huber | val_both | shuffle | 0.8179 | 0.2070 | 0.8179 |
| S2B D1 hierarchical Morgan Huber | val_both | zero | 0.7751 | 0.2110 | 0.7751 |
| S2B D2 hierarchical Morgan experimental FC | val_both | correct | 0.8114 | 0.2189 | 0.8114 |
| S2B D2 hierarchical Morgan experimental FC | val_both | shuffle | 0.8194 | 0.2119 | 0.8194 |
| S2B D2 hierarchical Morgan experimental FC | val_both | zero | 0.7627 | 0.2204 | 0.7627 |
| S2B D1 hierarchical Morgan Huber | val_time | correct | 0.5105 | 0.3789 | 0.5105 |
| S2B D1 hierarchical Morgan Huber | val_time | shuffle | 0.6111 | 0.2888 | 0.6111 |
| S2B D1 hierarchical Morgan Huber | val_time | zero | 0.6440 | 0.2621 | 0.6440 |
| S2B D2 hierarchical Morgan experimental FC | val_time | correct | 0.5093 | 0.3980 | 0.5093 |
| S2B D2 hierarchical Morgan experimental FC | val_time | shuffle | 0.6019 | 0.2938 | 0.6019 |
| S2B D2 hierarchical Morgan experimental FC | val_time | zero | 0.6189 | 0.2773 | 0.6189 |

| model | scenario | chemical_name | rmse_gain_vs_d0 |
|---|---|---|---|
| S2B D1 hierarchical Morgan Huber | val_both | Amphotericin B | -0.1115 |
| S2B D1 hierarchical Morgan Huber | val_both | FCCP | -0.0479 |
| S2B D1 hierarchical Morgan Huber | val_both | Hydroxyurea | 0.0634 |
| S2B D1 hierarchical Morgan Huber | val_both | Pentamidine isethionate | -0.0395 |
| S2B D1 hierarchical Morgan Huber | val_both | Raloxifene hydrochloride | -0.0197 |
| S2B D1 hierarchical Morgan Huber | val_both | Sulfometuron methyl | -0.0091 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | Amphotericin B | -0.1741 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | FCCP | -0.0673 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | Hydroxyurea | 0.0560 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | Pentamidine isethionate | -0.1227 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | Raloxifene hydrochloride | -0.0151 |
| S2B D1 hierarchical Morgan Huber | val_chem_only | Sulfometuron methyl | 0.0239 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | (1R, 2S, 5R) - (-) - Menthol | -0.0037 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 1-10 Phenanthroline monohydrate | -0.0101 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | 4-Hydroxytamoxifen | -0.0965 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Amiodarone hydrochloride | -0.0545 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Anisomycin | -0.0057 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Artemisinin | -0.0065 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Brefeldin A | -0.0105 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | CHX | 0.0607 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Cisplatin | -0.0075 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Clomiphene citrate | -0.0103 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Clotrimazole | -0.0121 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Cyclopiazonic acid | -0.0032 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Desipramine hydrochloride | -0.0107 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Dyclonine hydrochloride | -0.0112 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | EDTA | 0.0637 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Emodin | 0.0050 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Geldanamycin | -0.0101 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Haloperidol | -0.0087 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Harmine hydrochloride | -0.0095 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Hoechst 33258 | 0.0083 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | LY 294002 hydrochloride | -0.0033 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | NaCl | -0.0093 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Nigericin | 0.0062 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Nocodazole | -0.0079 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Nystatin dihydrate | 0.0131 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Oligomycin | 0.0108 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Parthenolide | 0.0024 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Rapamycin | 0.0613 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | SDS | 0.0851 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Sorbitol | 0.0587 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Staurosporine | -0.0098 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Trichostatin A | -0.0052 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Trifluoperazine dihydrochloride | -0.0127 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Tunicamycin | -0.0005 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | U-73122 | -0.0048 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Valinomycin | -0.0088 |
| S2B D1 hierarchical Morgan Huber | val_strain_only | Wortmannin | 0.0145 |
| S2B D1 hierarchical Morgan Huber | val_time | (1R, 2S, 5R) - (-) - Menthol | 0.0325 |
| S2B D1 hierarchical Morgan Huber | val_time | 4-Hydroxytamoxifen | 0.2846 |
| S2B D1 hierarchical Morgan Huber | val_time | Amiodarone hydrochloride | 0.1702 |
| S2B D1 hierarchical Morgan Huber | val_time | Anisomycin | 0.0476 |
| S2B D1 hierarchical Morgan Huber | val_time | Artemisinin | 0.0302 |
| S2B D1 hierarchical Morgan Huber | val_time | Brefeldin A | 0.0419 |
| S2B D1 hierarchical Morgan Huber | val_time | CHX | 0.0432 |
| S2B D1 hierarchical Morgan Huber | val_time | Cisplatin | -0.0062 |
| S2B D1 hierarchical Morgan Huber | val_time | Clomiphene citrate | -0.0008 |
| S2B D1 hierarchical Morgan Huber | val_time | Clotrimazole | 0.0684 |
| S2B D1 hierarchical Morgan Huber | val_time | Cyclopiazonic acid | 0.0793 |
| S2B D1 hierarchical Morgan Huber | val_time | Desipramine hydrochloride | -0.0013 |
| S2B D1 hierarchical Morgan Huber | val_time | Dyclonine hydrochloride | 0.0115 |
| S2B D1 hierarchical Morgan Huber | val_time | EDTA | 0.0709 |
| S2B D1 hierarchical Morgan Huber | val_time | Emodin | 0.0414 |
| S2B D1 hierarchical Morgan Huber | val_time | Geldanamycin | -0.0228 |
| S2B D1 hierarchical Morgan Huber | val_time | Haloperidol | -0.1067 |
| S2B D1 hierarchical Morgan Huber | val_time | Harmine hydrochloride | 0.0047 |
| S2B D1 hierarchical Morgan Huber | val_time | Hoechst 33258 | 0.0428 |
| S2B D1 hierarchical Morgan Huber | val_time | LY 294002 hydrochloride | -0.0352 |
| S2B D1 hierarchical Morgan Huber | val_time | NaCl | 0.0006 |
| S2B D1 hierarchical Morgan Huber | val_time | Nigericin | 0.0238 |
| S2B D1 hierarchical Morgan Huber | val_time | Nocodazole | 0.0594 |
| S2B D1 hierarchical Morgan Huber | val_time | Nystatin dihydrate | 0.0347 |
| S2B D1 hierarchical Morgan Huber | val_time | Oligomycin | -0.0539 |
| S2B D1 hierarchical Morgan Huber | val_time | Parthenolide | 0.0473 |
| S2B D1 hierarchical Morgan Huber | val_time | Rapamycin | 0.0624 |
| S2B D1 hierarchical Morgan Huber | val_time | SDS | 0.0891 |
| S2B D1 hierarchical Morgan Huber | val_time | Sorbitol | 0.0686 |
| S2B D1 hierarchical Morgan Huber | val_time | Staurosporine | 0.0463 |
| S2B D1 hierarchical Morgan Huber | val_time | Trichostatin A | 0.0193 |
| S2B D1 hierarchical Morgan Huber | val_time | Trifluoperazine dihydrochloride | 0.0412 |
| S2B D1 hierarchical Morgan Huber | val_time | Tunicamycin | 0.0694 |
| S2B D1 hierarchical Morgan Huber | val_time | U-73122 | 0.0448 |
| S2B D1 hierarchical Morgan Huber | val_time | Valinomycin | 0.0270 |
| S2B D1 hierarchical Morgan Huber | val_time | Wortmannin | 0.0566 |
| S2B D2 hierarchical Morgan experimental FC | val_both | Amphotericin B | -0.0853 |
| S2B D2 hierarchical Morgan experimental FC | val_both | FCCP | 0.0171 |
| S2B D2 hierarchical Morgan experimental FC | val_both | Hydroxyurea | 0.0723 |
| S2B D2 hierarchical Morgan experimental FC | val_both | Pentamidine isethionate | -0.0089 |
| S2B D2 hierarchical Morgan experimental FC | val_both | Raloxifene hydrochloride | -0.0369 |
| S2B D2 hierarchical Morgan experimental FC | val_both | Sulfometuron methyl | -0.0182 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | Amphotericin B | -0.1339 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | FCCP | -0.0097 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | Hydroxyurea | 0.0502 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | Pentamidine isethionate | -0.0827 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | Raloxifene hydrochloride | -0.0057 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | Sulfometuron methyl | 0.0200 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | (1R, 2S, 5R) - (-) - Menthol | -0.0217 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 1-10 Phenanthroline monohydrate | -0.0214 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 4-Hydroxytamoxifen | -0.1252 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Amiodarone hydrochloride | -0.0922 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Anisomycin | -0.0167 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Artemisinin | -0.0211 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Brefeldin A | -0.0222 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | CHX | 0.0642 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Cisplatin | -0.0050 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Clomiphene citrate | -0.0291 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Clotrimazole | -0.0244 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Cyclopiazonic acid | -0.0408 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Desipramine hydrochloride | -0.0214 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Dyclonine hydrochloride | -0.0233 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | EDTA | 0.0656 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Emodin | -0.0112 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Geldanamycin | -0.0211 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Haloperidol | -0.0211 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Harmine hydrochloride | -0.0212 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Hoechst 33258 | 0.0473 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | LY 294002 hydrochloride | -0.0034 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | NaCl | -0.0010 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Nigericin | 0.0464 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Nocodazole | -0.0176 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Nystatin dihydrate | 0.0539 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Oligomycin | -0.0131 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Parthenolide | -0.0236 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Rapamycin | 0.0616 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | SDS | 0.0823 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Sorbitol | 0.0681 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Staurosporine | -0.0217 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Trichostatin A | -0.0146 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Trifluoperazine dihydrochloride | -0.0255 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Tunicamycin | -0.0036 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | U-73122 | -0.0134 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Valinomycin | -0.0227 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | Wortmannin | 0.0676 |
| S2B D2 hierarchical Morgan experimental FC | val_time | (1R, 2S, 5R) - (-) - Menthol | 0.0298 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 4-Hydroxytamoxifen | 0.3423 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Amiodarone hydrochloride | 0.2493 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Anisomycin | 0.0477 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Artemisinin | 0.0256 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Brefeldin A | 0.0302 |
| S2B D2 hierarchical Morgan experimental FC | val_time | CHX | 0.0514 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Cisplatin | -0.0065 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Clomiphene citrate | 0.0046 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Clotrimazole | 0.0799 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Cyclopiazonic acid | 0.1327 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Desipramine hydrochloride | -0.0115 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Dyclonine hydrochloride | -0.0061 |
| S2B D2 hierarchical Morgan experimental FC | val_time | EDTA | 0.0617 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Emodin | 0.0352 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Geldanamycin | -0.0066 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Haloperidol | -0.0922 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Harmine hydrochloride | 0.0100 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Hoechst 33258 | 0.0388 |
| S2B D2 hierarchical Morgan experimental FC | val_time | LY 294002 hydrochloride | -0.0198 |
| S2B D2 hierarchical Morgan experimental FC | val_time | NaCl | -0.0058 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Nigericin | 0.0099 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Nocodazole | 0.0512 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Nystatin dihydrate | 0.0350 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Oligomycin | -0.1103 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Parthenolide | 0.0553 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Rapamycin | 0.0568 |
| S2B D2 hierarchical Morgan experimental FC | val_time | SDS | 0.0853 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Sorbitol | 0.0705 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Staurosporine | 0.0524 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Trichostatin A | 0.0237 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Trifluoperazine dihydrochloride | 0.0414 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Tunicamycin | 0.0543 |
| S2B D2 hierarchical Morgan experimental FC | val_time | U-73122 | 0.0328 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Valinomycin | 0.0173 |
| S2B D2 hierarchical Morgan experimental FC | val_time | Wortmannin | 0.0736 |

Direct D1-vs-D0 and D2-vs-D1 attribution (negative RMSE change is favorable; positive PCC gain is favorable):

| candidate | reference | scenario | absolute_rmse_change_candidate_minus_reference | raw_fc_pcc_gain_candidate_minus_reference |
|---|---|---|---|---|
| S2B D1 hierarchical Morgan Huber | S2B D0 hierarchical none Huber | val_chem_only | 0.0601 | -0.0214 |
| S2B D1 hierarchical Morgan Huber | S2B D0 hierarchical none Huber | val_strain_only | -0.0154 | 0.0187 |
| S2B D1 hierarchical Morgan Huber | S2B D0 hierarchical none Huber | val_both | 0.0320 | 0.0018 |
| S2B D1 hierarchical Morgan Huber | S2B D0 hierarchical none Huber | val_time | -0.0391 | 0.0641 |
| S2B D2 hierarchical Morgan experimental FC | S2B D1 hierarchical Morgan Huber | val_chem_only | -0.0285 | 0.0175 |
| S2B D2 hierarchical Morgan experimental FC | S2B D1 hierarchical Morgan Huber | val_strain_only | 0.0030 | 0.0068 |
| S2B D2 hierarchical Morgan experimental FC | S2B D1 hierarchical Morgan Huber | val_both | -0.0288 | 0.0081 |
| S2B D2 hierarchical Morgan experimental FC | S2B D1 hierarchical Morgan Huber | val_time | -0.0012 | 0.0191 |

```json
{
  "S2B D1 hierarchical Morgan Huber": {
    "positive_gain_drug_count": 44,
    "total_drug_count": 85,
    "mean_drug_equal_gain": 0.012073050877627204,
    "median_drug_equal_gain": 0.0023707151412963867,
    "negative_transfer_drug_count": 41,
    "top3_negative_harm_share": 0.3499982485488314
  },
  "S2B D2 hierarchical Morgan experimental FC": {
    "positive_gain_drug_count": 41,
    "total_drug_count": 85,
    "mean_drug_equal_gain": 0.013833887436810662,
    "median_drug_equal_gain": -0.00337904691696167,
    "negative_transfer_drug_count": 44,
    "top3_negative_harm_share": 0.2758133032001308
  }
}
```

## Predeclared advancement gates

```json
{
  "S2B D0 hierarchical none Huber": {
    "corresponding_s1_model": "S1 B anchor Morgan Huber",
    "mean_absolute_rmse": 0.6677444130182266,
    "mean_absolute_rmse_change_vs_s1": -0.0015369951725006104,
    "mean_raw_fc_pcc_gain_vs_s1": -0.005480951339041529,
    "val_chem_context_residual_pcc_gain_vs_s1": 0.012135501003310678,
    "val_strain_drug_residual_pcc_gain_vs_s1": -0.021067187448981506,
    "gate_1_mean_absolute_rmse_not_worse": true,
    "gate_2_fc_or_weighted_residual_pcc_gain_at_least_0_01": true,
    "gate_3_val_chem_context_pcc_drop_at_most_0_005": true,
    "gate_4_val_strain_drug_pcc_drop_at_most_0_005": false,
    "gate_5_frozen_anchor_exact": true,
    "gate_6_oov_semantics_exact": true,
    "gate_7_not_single_scenario_or_few_drug_dominated": false,
    "eligible_for_multiseed": false
  },
  "S2B D1 hierarchical Morgan Huber": {
    "corresponding_s1_model": "S1 B anchor Morgan Huber",
    "mean_absolute_rmse": 0.6771520227193832,
    "mean_absolute_rmse_change_vs_s1": 0.007870614528656006,
    "mean_raw_fc_pcc_gain_vs_s1": 0.010311827303699117,
    "val_chem_context_residual_pcc_gain_vs_s1": -0.0028763218600418305,
    "val_strain_drug_residual_pcc_gain_vs_s1": -0.024406059326888052,
    "gate_1_mean_absolute_rmse_not_worse": false,
    "gate_2_fc_or_weighted_residual_pcc_gain_at_least_0_01": true,
    "gate_3_val_chem_context_pcc_drop_at_most_0_005": true,
    "gate_4_val_strain_drug_pcc_drop_at_most_0_005": false,
    "gate_5_frozen_anchor_exact": true,
    "gate_6_oov_semantics_exact": true,
    "gate_7_not_single_scenario_or_few_drug_dominated": false,
    "eligible_for_multiseed": false
  },
  "S2B D2 hierarchical Morgan experimental FC": {
    "corresponding_s1_model": "S1 C anchor Morgan Huber plus experimental FC",
    "mean_absolute_rmse": 0.6632760316133499,
    "mean_absolute_rmse_change_vs_s1": -0.006138056516647339,
    "mean_raw_fc_pcc_gain_vs_s1": 0.009482409102265554,
    "val_chem_context_residual_pcc_gain_vs_s1": 0.010820977872110837,
    "val_strain_drug_residual_pcc_gain_vs_s1": -0.03602862511380717,
    "gate_1_mean_absolute_rmse_not_worse": true,
    "gate_2_fc_or_weighted_residual_pcc_gain_at_least_0_01": true,
    "gate_3_val_chem_context_pcc_drop_at_most_0_005": true,
    "gate_4_val_strain_drug_pcc_drop_at_most_0_005": false,
    "gate_5_frozen_anchor_exact": true,
    "gate_6_oov_semantics_exact": true,
    "gate_7_not_single_scenario_or_few_drug_dominated": false,
    "eligible_for_multiseed": false
  }
}
```

Chemical contribution rule:

```json
{
  "S2B D1 hierarchical Morgan Huber": {
    "correct_strictly_better_absolute_rmse_than_d0_in_val_chem_and_val_both": false,
    "correct_not_worse_than_shuffle_on_absolute_and_raw_fc_pcc": false,
    "morgan_contribution_established": false
  },
  "S2B D2 hierarchical Morgan experimental FC": {
    "correct_strictly_better_absolute_rmse_than_d0_in_val_chem_and_val_both": false,
    "correct_not_worse_than_shuffle_on_absolute_and_raw_fc_pcc": false,
    "morgan_contribution_established": false
  }
}
```

## Attribution conclusions

- D0 versus S1 B improves mean absolute RMSE by `0.0015` and val_chem context residual PCC by `0.0121`, but reduces mean raw-FC PCC by `0.0055` and val_strain drug-residual PCC by `0.0211`. Because D0 also removes Morgan, this is not a pure anchor contrast. The clean D1-vs-S1-B anchor contrast gains `0.0103` mean raw-FC PCC but worsens mean absolute RMSE by `0.0079` and drug-residual PCC by `0.0244`; hierarchical anchor integration is therefore mixed, not a uniform improvement.
- D1 is not stably better than D0. It helps val_strain_only and val_time absolute RMSE but harms val_chem_only and val_both; its raw-FC PCC likewise decreases on val_chem_only and increases on the other three scenarios. Both chemical contribution gates are false, so Morgan remains unestablished.
- D2 versus D1 improves raw-FC PCC in all four scenarios and absolute RMSE in three of four, while val_strain_only absolute RMSE worsens slightly. Thus the frozen nonofficial FC/correlation loss has limited additional direction value inside the Morgan model, but D2 still fails the val_strain residual and broad-improvement gates and is not advanced. The mapping remains experimental and was never used for early stopping, mapping changes or weight search.
- Every Stage-B freeze audit passed and the identical-input batch outputs are bitwise unchanged; treatment labels therefore did not update or get absorbed by any hierarchical batch level.
- Source/instrument/plate outputs are identical across D0/D1/D2 for each scenario, plate RMS remains far below source RMS, and the exact OOV error is zero. The S2A layer semantics are preserved rather than relearned from treatment.
- No model is advanced to multi-seed unless all seven gates pass. No test proteome was opened and no test prediction was generated.

## Commands, tests and failures

- Formal D0/D1/D2: `D:\虚拟细胞\.venv\Scripts\python.exe -m baseline.training_v2 --config <D0|D1|D2 config> --output-dir <new formal directory>`; all three completed normally.
- Analysis: `D:\虚拟细胞\.venv\Scripts\python.exe -m baseline.hierarchical_treatment_analysis_v2`; PASS.
- All V2 tests: `python -m pytest --import-mode=importlib <all baseline/tests/test_*.py except test_person_c.py> -q`; `97 passed in 46.20s`.
- Person C regression, from `D:\虚拟细胞\baseline`: `python -m pytest tests/test_person_c.py -q`; `6 passed in 4.33s`.
- Formal training failures, OOM, NaN and checkpoint failures: none. An initial test collection without `--import-mode=importlib` produced three Windows namespace import errors; the documented project mode passed. The first analysis launch preceded creation of its thin module entry point and returned `No module named baseline.hierarchical_treatment_analysis_v2`; the wrapper was added and analysis passed without retraining or checkpoint changes.

Stage S2B is paused after these three single-seed runs.
