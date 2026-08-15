# MODEL V2 Stage S1 Score-Aligned Report

Status: COMPLETE_AND_PAUSED  
Scoring labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

Stage A was not retrained. B and C share checkpoint SHA `d84aa821a6d39dea876c91dc66314d6eb93cb577890c179687e036761d1a167e`, Stage-B initialization hash `b1d6d40fc41679e442b887f31e2467591fc80bd17020a68bcb3c0edd16fe00ff`, seed, data order, 4,121 training treatments, and the same 957-sample technical-group holdout. C alone adds the predeclared train-only nonofficial parity FC Huber 0.05 plus row-wise Pearson 0.05.

## Absolute metrics on the exact S0 common intersection

| model | scenario | n_samples | log2_rmse | mae | global_r2 | sample_pcc_median | sample_r2_median | protein_pcc_median | protein_r2_median |
|---|---|---|---|---|---|---|---|---|---|
| V2 batch-enabled Huber | val_chem_only | 981 | 0.6196 | 0.4496 | 0.9483 | 0.9804 | 0.9561 | 0.8169 | 0.5494 |
| V2 no-batch Morgan-only | val_chem_only | 981 | 0.9536 | 0.7044 | 0.8775 | 0.9683 | 0.9075 | 0.2238 | -0.0166 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 981 | 0.9899 | 0.7610 | 0.8680 | 0.9569 | 0.8876 | 0.0363 | -0.0734 |
| Matched Control | val_chem_only | 981 | 0.3817 | 0.2543 | 0.9804 | 0.9924 | 0.9834 | 0.9163 | 0.8279 |
| V2 batch-enabled Huber | val_strain_only | 1313 | 0.7368 | 0.5126 | 0.9253 | 0.9653 | 0.9278 | 0.7714 | 0.3476 |
| V2 no-batch Morgan-only | val_strain_only | 1313 | 0.9194 | 0.6404 | 0.8837 | 0.9584 | 0.9107 | 0.4554 | -0.0218 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 1313 | 0.7940 | 0.5782 | 0.9133 | 0.9691 | 0.9291 | 0.3830 | 0.0199 |
| Matched Control | val_strain_only | 1313 | 0.4101 | 0.2735 | 0.9769 | 0.9911 | 0.9811 | 0.8573 | 0.7037 |
| V2 batch-enabled Huber | val_both | 266 | 0.8315 | 0.5945 | 0.9068 | 0.9621 | 0.9098 | 0.8115 | 0.3373 |
| V2 no-batch Morgan-only | val_both | 266 | 1.1233 | 0.8297 | 0.8299 | 0.9394 | 0.8494 | 0.1448 | -0.2150 |
| V2 experimental parity FC Morgan no-batch | val_both | 266 | 0.9917 | 0.7664 | 0.8674 | 0.9526 | 0.8858 | 0.0319 | -0.1323 |
| Matched Control | val_both | 266 | 0.3853 | 0.2570 | 0.9800 | 0.9918 | 0.9825 | 0.9061 | 0.8018 |
| V2 batch-enabled Huber | val_time | 134 | 0.5563 | 0.3819 | 0.9580 | 0.9820 | 0.9620 | 0.7581 | 0.5483 |
| V2 no-batch Morgan-only | val_time | 134 | 0.6949 | 0.4765 | 0.9345 | 0.9817 | 0.9520 | 0.5171 | 0.2362 |
| V2 experimental parity FC Morgan no-batch | val_time | 134 | 0.7885 | 0.5718 | 0.9157 | 0.9662 | 0.9210 | 0.3391 | 0.0750 |
| Matched Control | val_time | 134 | 0.4336 | 0.2843 | 0.9745 | 0.9917 | 0.9816 | 0.8549 | 0.7066 |
| S1 B anchor Morgan Huber | val_chem_only | 981 | 0.5965 | 0.4164 | 0.9521 | 0.9801 | 0.9535 | 0.7991 | 0.6114 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 981 | 0.6141 | 0.4321 | 0.9492 | 0.9778 | 0.9510 | 0.8090 | 0.5903 |
| S1 B anchor Morgan Huber | val_strain_only | 1313 | 0.7275 | 0.5072 | 0.9272 | 0.9667 | 0.9310 | 0.7682 | 0.3637 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 1313 | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| S1 B anchor Morgan Huber | val_both | 266 | 0.7985 | 0.5608 | 0.9140 | 0.9604 | 0.9132 | 0.7865 | 0.3895 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 266 | 0.8054 | 0.5661 | 0.9125 | 0.9608 | 0.9116 | 0.8052 | 0.3986 |
| S1 B anchor Morgan Huber | val_time | 134 | 0.5546 | 0.3853 | 0.9583 | 0.9824 | 0.9616 | 0.7561 | 0.5445 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 134 | 0.5402 | 0.3740 | 0.9604 | 0.9839 | 0.9648 | 0.7652 | 0.5648 |

## Raw FC

| model | scenario | global_fc_pcc | sample_fc_pcc_median | protein_fc_pcc_median | fc_rmse | fc_direction_accuracy |
|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | NA | NA | NA | 0.3817 | 0.0000 |
| V2 batch-enabled Huber | val_chem_only | 0.3072 | 0.3000 | 0.3460 | 0.6196 | 0.5976 |
| V2 no-batch Morgan-only | val_chem_only | 0.1937 | 0.2459 | 0.2208 | 0.9536 | 0.5597 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 0.1908 | 0.2084 | 0.2133 | 0.9899 | 0.5516 |
| Matched Control | val_strain_only | NA | NA | NA | 0.4101 | 0.0000 |
| V2 batch-enabled Huber | val_strain_only | 0.2443 | 0.2344 | 0.4034 | 0.7368 | 0.5873 |
| V2 no-batch Morgan-only | val_strain_only | 0.1946 | 0.2160 | 0.2905 | 0.9194 | 0.5735 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 0.2485 | 0.2719 | 0.2777 | 0.7940 | 0.5836 |
| Matched Control | val_both | NA | NA | NA | 0.3853 | 0.0000 |
| V2 batch-enabled Huber | val_both | 0.2257 | 0.2086 | 0.3726 | 0.8315 | 0.5720 |
| V2 no-batch Morgan-only | val_both | 0.1461 | 0.1690 | 0.2233 | 1.1233 | 0.5436 |
| V2 experimental parity FC Morgan no-batch | val_both | 0.1783 | 0.2047 | 0.2210 | 0.9917 | 0.5462 |
| Matched Control | val_time | NA | NA | NA | 0.4336 | 0.0000 |
| V2 batch-enabled Huber | val_time | 0.3139 | 0.3182 | 0.3363 | 0.5563 | 0.6096 |
| V2 no-batch Morgan-only | val_time | 0.2711 | 0.3357 | 0.2725 | 0.6949 | 0.6032 |
| V2 experimental parity FC Morgan no-batch | val_time | 0.2348 | 0.2364 | 0.2369 | 0.7885 | 0.5770 |
| S1 B anchor Morgan Huber | val_chem_only | 0.3042 | 0.3062 | 0.3333 | 0.5965 | 0.6038 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.3106 | 0.3095 | 0.3494 | 0.6141 | 0.6076 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2492 | 0.2404 | 0.3941 | 0.7275 | 0.5883 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| S1 B anchor Morgan Huber | val_both | 0.2224 | 0.2133 | 0.3512 | 0.7985 | 0.5738 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 0.2306 | 0.2181 | 0.3712 | 0.8054 | 0.5784 |
| S1 B anchor Morgan Huber | val_time | 0.3214 | 0.3245 | 0.3323 | 0.5546 | 0.6100 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 0.3493 | 0.3507 | 0.3600 | 0.5402 | 0.6197 |

## Context and drug residual

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_chem_only | 0.1880 | 0.3808 | 0.5636 |
| V2 batch-enabled Huber | val_chem_only | 0.1978 | 0.6188 | 0.5566 |
| V2 no-batch Morgan-only | val_chem_only | 0.1155 | 0.9526 | 0.5346 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 0.1143 | 0.9885 | 0.5291 |
| S1 B anchor Morgan Huber | val_chem_only | 0.2024 | 0.5954 | 0.5672 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.1973 | 0.6132 | 0.5639 |

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_strain_only | 0.1372 | 0.4101 | 0.5370 |
| V2 batch-enabled Huber | val_strain_only | 0.2839 | 0.7368 | 0.5979 |
| V2 no-batch Morgan-only | val_strain_only | 0.2292 | 0.9194 | 0.5828 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 0.2688 | 0.7940 | 0.5914 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2864 | 0.7275 | 0.5974 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.2914 | 0.7180 | 0.6005 |

## High-effect diagnostics

| model | scenario | direction_accuracy | high_effect_pcc | precision | recall | f1 | auprc |
|---|---|---|---|---|---|---|---|
| Matched Control | val_chem_only | 0.0000 | NA | NA | 0.0000 | NA | 0.0210 |
| V2 batch-enabled Huber | val_chem_only | 0.8110 | 0.6081 | 0.0857 | 0.3198 | 0.1352 | 0.0808 |
| V2 no-batch Morgan-only | val_chem_only | 0.6970 | 0.4869 | 0.0461 | 0.4498 | 0.0837 | 0.0568 |
| V2 experimental parity FC Morgan no-batch | val_chem_only | 0.6943 | 0.5052 | 0.0412 | 0.4713 | 0.0758 | 0.0605 |
| Matched Control | val_strain_only | 0.0000 | NA | NA | 0.0000 | NA | 0.0262 |
| V2 batch-enabled Huber | val_strain_only | 0.7671 | 0.5105 | 0.0859 | 0.3418 | 0.1373 | 0.0752 |
| V2 no-batch Morgan-only | val_strain_only | 0.7518 | 0.4726 | 0.0599 | 0.3755 | 0.1034 | 0.0551 |
| V2 experimental parity FC Morgan no-batch | val_strain_only | 0.7743 | 0.5542 | 0.0665 | 0.3683 | 0.1126 | 0.0722 |
| Matched Control | val_both | 0.0000 | NA | NA | 0.0000 | NA | 0.0212 |
| V2 batch-enabled Huber | val_both | 0.7737 | 0.5391 | 0.0554 | 0.3735 | 0.0965 | 0.0535 |
| V2 no-batch Morgan-only | val_both | 0.6828 | 0.4317 | 0.0390 | 0.4657 | 0.0720 | 0.0448 |
| V2 experimental parity FC Morgan no-batch | val_both | 0.6942 | 0.5015 | 0.0412 | 0.4554 | 0.0755 | 0.0577 |
| Matched Control | val_time | 0.0000 | NA | NA | 0.0000 | NA | 0.0318 |
| V2 batch-enabled Huber | val_time | 0.7555 | 0.5134 | 0.1675 | 0.2418 | 0.1979 | 0.1162 |
| V2 no-batch Morgan-only | val_time | 0.7704 | 0.5328 | 0.0883 | 0.2715 | 0.1333 | 0.0735 |
| V2 experimental parity FC Morgan no-batch | val_time | 0.7405 | 0.4895 | 0.0717 | 0.3186 | 0.1171 | 0.0749 |
| S1 B anchor Morgan Huber | val_chem_only | 0.7875 | 0.6038 | 0.1030 | 0.3290 | 0.1569 | 0.0909 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.8053 | 0.6143 | 0.0933 | 0.3391 | 0.1463 | 0.0842 |
| S1 B anchor Morgan Huber | val_strain_only | 0.7723 | 0.5232 | 0.0874 | 0.3426 | 0.1393 | 0.0755 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.7853 | 0.5433 | 0.0891 | 0.3398 | 0.1412 | 0.0764 |
| S1 B anchor Morgan Huber | val_both | 0.7692 | 0.5403 | 0.0631 | 0.3730 | 0.1080 | 0.0589 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 0.7788 | 0.5441 | 0.0610 | 0.3748 | 0.1049 | 0.0568 |
| S1 B anchor Morgan Huber | val_time | 0.7587 | 0.5350 | 0.1662 | 0.2480 | 0.1990 | 0.1173 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 0.7883 | 0.5727 | 0.1775 | 0.2592 | 0.2107 | 0.1282 |

## Component norms

| model | scenario | delta_response_norm | delta_batch_norm | batch_to_response_norm | delta_response_rms | delta_batch_rms |
|---|---|---|---|---|---|---|
| S1 B anchor Morgan Huber | val_chem_only | 428.4729 | 1081.0208 | 2.5230 | 0.2265 | 0.5716 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 573.3249 | 1153.1916 | 2.0114 | 0.3031 | 0.6097 |
| S1 B anchor Morgan Huber | val_strain_only | 608.5562 | 1139.4545 | 1.8724 | 0.2771 | 0.5188 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 566.4008 | 1173.2432 | 2.0714 | 0.2579 | 0.5342 |
| S1 B anchor Morgan Huber | val_both | 222.6857 | 563.9312 | 2.5324 | 0.2272 | 0.5754 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 267.4890 | 601.9088 | 2.2502 | 0.2729 | 0.6142 |
| S1 B anchor Morgan Huber | val_time | 196.3803 | 351.0643 | 1.7877 | 0.2816 | 0.5034 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 201.9401 | 359.3080 | 1.7793 | 0.2896 | 0.5152 |

## Batch robustness and internal group holdout

Validation post-hoc correct/shuffled/disabled modes do not retrain or select checkpoints.

| model | scenario | batch_mode | absolute_rmse | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|
| S1 B anchor Morgan Huber | val_chem_only | correct | 0.5965 | 0.3042 | 0.5965 |
| S1 B anchor Morgan Huber | val_chem_only | shuffled | 1.0811 | 0.1651 | 1.0811 |
| S1 B anchor Morgan Huber | val_chem_only | disabled | 0.8998 | 0.1987 | 0.8998 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | correct | 0.6141 | 0.3106 | 0.6141 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | shuffled | 1.1230 | 0.1673 | 1.1230 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | disabled | 0.9257 | 0.2029 | 0.9257 |
| S1 B anchor Morgan Huber | val_strain_only | correct | 0.7275 | 0.2492 | 0.7275 |
| S1 B anchor Morgan Huber | val_strain_only | shuffled | 1.0286 | 0.1731 | 1.0286 |
| S1 B anchor Morgan Huber | val_strain_only | disabled | 0.9248 | 0.2025 | 0.9248 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | correct | 0.7180 | 0.2615 | 0.7180 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | shuffled | 1.0454 | 0.1765 | 1.0454 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | disabled | 0.9262 | 0.2101 | 0.9262 |
| S1 B anchor Morgan Huber | val_both | correct | 0.7985 | 0.2224 | 0.7985 |
| S1 B anchor Morgan Huber | val_both | shuffled | 1.1959 | 0.1465 | 1.1959 |
| S1 B anchor Morgan Huber | val_both | disabled | 1.0424 | 0.1630 | 1.0424 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | correct | 0.8054 | 0.2306 | 0.8054 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | shuffled | 1.2314 | 0.1474 | 1.2314 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | disabled | 1.0580 | 0.1660 | 1.0580 |
| S1 B anchor Morgan Huber | val_time | correct | 0.5546 | 0.3214 | 0.5546 |
| S1 B anchor Morgan Huber | val_time | shuffled | 0.8259 | 0.2171 | 0.8259 |
| S1 B anchor Morgan Huber | val_time | disabled | 0.7518 | 0.2370 | 0.7518 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | correct | 0.5402 | 0.3493 | 0.5402 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | shuffled | 0.8341 | 0.2313 | 0.8341 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | disabled | 0.7524 | 0.2559 | 0.7524 |

The 957 held-out Stage-B treatment labels were never used for gradient updates or early stopping. Stage A control samples can still share technical categories, so this is specifically a treatment-label group holdout, not a claim of wholly unseen instrumentation.

| model | batch_mode | n_samples | absolute_rmse | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|
| S1 B anchor Morgan Huber | correct | 957 | 0.5655 | 0.3331 | 0.5430 |
| S1 B anchor Morgan Huber | shuffled | 957 | 0.9068 | 0.1976 | 0.8838 |
| S1 B anchor Morgan Huber | disabled | 957 | 0.7823 | 0.2379 | 0.7556 |
| S1 C anchor Morgan Huber plus experimental FC | correct | 957 | 0.5574 | 0.3544 | 0.5352 |
| S1 C anchor Morgan Huber plus experimental FC | shuffled | 957 | 0.9415 | 0.2041 | 0.9231 |
| S1 C anchor Morgan Huber plus experimental FC | disabled | 957 | 0.7869 | 0.2528 | 0.7601 |

## Morgan correct/shuffle/zero post-hoc diagnostic

| model | scenario | chemical_mode | absolute_rmse | raw_fc_global_pcc | raw_fc_rmse |
|---|---|---|---|---|---|
| S1 B anchor Morgan Huber | val_chem_only | correct | 0.5965 | 0.3042 | 0.5965 |
| S1 B anchor Morgan Huber | val_chem_only | shuffle | 0.5650 | 0.3149 | 0.5650 |
| S1 B anchor Morgan Huber | val_chem_only | zero | 0.5756 | 0.3047 | 0.5756 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | correct | 0.6141 | 0.3106 | 0.6141 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | shuffle | 0.5742 | 0.3246 | 0.5742 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | zero | 0.5592 | 0.3223 | 0.5592 |
| S1 B anchor Morgan Huber | val_strain_only | correct | 0.7275 | 0.2492 | 0.7275 |
| S1 B anchor Morgan Huber | val_strain_only | shuffle | 0.7827 | 0.2299 | 0.7827 |
| S1 B anchor Morgan Huber | val_strain_only | zero | 0.7963 | 0.2125 | 0.7963 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | correct | 0.7180 | 0.2615 | 0.7180 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | shuffle | 0.7701 | 0.2349 | 0.7701 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | zero | 0.7694 | 0.2283 | 0.7694 |
| S1 B anchor Morgan Huber | val_both | correct | 0.7985 | 0.2224 | 0.7985 |
| S1 B anchor Morgan Huber | val_both | shuffle | 0.7782 | 0.2220 | 0.7782 |
| S1 B anchor Morgan Huber | val_both | zero | 0.7826 | 0.2147 | 0.7826 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | correct | 0.8054 | 0.2306 | 0.8054 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | shuffle | 0.7770 | 0.2335 | 0.7770 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | zero | 0.7627 | 0.2309 | 0.7627 |
| S1 B anchor Morgan Huber | val_time | correct | 0.5546 | 0.3214 | 0.5546 |
| S1 B anchor Morgan Huber | val_time | shuffle | 0.6263 | 0.2802 | 0.6263 |
| S1 B anchor Morgan Huber | val_time | zero | 0.6551 | 0.2543 | 0.6551 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | correct | 0.5402 | 0.3493 | 0.5402 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | shuffle | 0.6088 | 0.2945 | 0.6088 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | zero | 0.6288 | 0.2683 | 0.6288 |

## Predeclared advancement gates

Thresholds used only for the go/no-go audit: raw-FC mean improvement at least 0.02, absolute-RMSE ratio no more than 1.10, worst batch shuffle/disabled absolute-RMSE ratio no more than 1.10, and Morgan correct no worse than shuffle/zero by more than 0.01 PCC in val_chem_only and val_both.

```json
{
  "S1 B anchor Morgan Huber": {
    "mean_raw_fc_pcc": 0.2743052569560935,
    "raw_fc_pcc_gain_vs_current_v2": 0.0015211120316738214,
    "val_chem_context_residual_pcc_gain_vs_current_v2": 0.0046601867537528485,
    "val_strain_drug_residual_pcc_gain_vs_current_v2": 0.0025817321006925686,
    "max_primary_or_high_weight_pcc_gain": 0.0046601867537528485,
    "mean_absolute_rmse": 0.6692814081907272,
    "absolute_rmse_ratio_vs_current_v2": 0.9755861889656081,
    "worst_batch_disabled_absolute_rmse_ratio": 1.5085949633096292,
    "worst_batch_shuffle_absolute_rmse_ratio": 1.8125909564539247,
    "chemical_correct_not_worse_than_shuffle_zero_with_tolerance_0.01": false,
    "improved_drug_fraction_by_raw_fc_rmse": 0.7176470588235294,
    "top3_positive_drug_gain_share": 0.20233097584721912,
    "largest_positive_scenario_gain_share": 0.6044085015297584,
    "gate_1_fc_or_residual_improves": false,
    "gate_2_absolute_not_collapsed": true,
    "gate_3_batch_robust": false,
    "gate_4_chemical_supported": false,
    "gate_5_not_few_drug_or_single_scenario_dominated": true,
    "eligible_for_multiseed": false
  },
  "S1 C anchor Morgan Huber plus experimental FC": {
    "mean_raw_fc_pcc": 0.28801040087514895,
    "raw_fc_pcc_gain_vs_current_v2": 0.015226255950729295,
    "val_chem_context_residual_pcc_gain_vs_current_v2": -0.00043829683656174123,
    "val_strain_drug_residual_pcc_gain_vs_current_v2": 0.007530569383342511,
    "max_primary_or_high_weight_pcc_gain": 0.015226255950729295,
    "mean_absolute_rmse": 0.6694140881299973,
    "absolute_rmse_ratio_vs_current_v2": 0.9757795914936338,
    "worst_batch_disabled_absolute_rmse_ratio": 1.5075671303998723,
    "worst_batch_shuffle_absolute_rmse_ratio": 1.8287569592012882,
    "chemical_correct_not_worse_than_shuffle_zero_with_tolerance_0.01": false,
    "improved_drug_fraction_by_raw_fc_rmse": 0.8705882352941177,
    "top3_positive_drug_gain_share": 0.1454512083693011,
    "largest_positive_scenario_gain_share": 0.5810407042915876,
    "gate_1_fc_or_residual_improves": false,
    "gate_2_absolute_not_collapsed": true,
    "gate_3_batch_robust": false,
    "gate_4_chemical_supported": false,
    "gate_5_not_few_drug_or_single_scenario_dominated": true,
    "eligible_for_multiseed": false
  }
}
```

## Conclusion

- Matched Control remains the absolute-fidelity reference. Both S1 models improve mean absolute RMSE by about 2.4% versus the frozen batch-enabled V2, but remain well behind Matched Control.
- B improves the best primary/high-weight PCC proxy by only 0.0047; C improves it by 0.0152. Neither reaches the predeclared +0.02 clear-improvement threshold.
- C's FC/correlation objective improves mean raw-FC PCC more than B without an absolute-fidelity collapse, but the gain is modest rather than decisive. High-effect changes are mixed across scenario and metric; the full table above is the authoritative record.
- Batch dependence remains the primary blocker: the worst validation absolute-RMSE ratio is 1.51x when disabled and 1.83x when shuffled. The independent 957-treatment group holdout shows the same qualitative collapse.
- Morgan correct is not consistently better than frozen-interface shuffle/zero in `val_chem_only` and `val_both`; chemical generalization is therefore not established.
- B and C both fail the advancement gate. No multi-seed run is authorized from these results. The next iteration should prioritize baseline/anchor modeling that does not depend on public technical batches, not increase FC weight or strengthen batch.

## Commands and execution status

- B and C were run with `python -m baseline.training_v2`, their declared JSON/YAML-compatible configs, the fixed Stage A checkpoint, and separate new output directories.
- The final inference audit was run with `python -m baseline.score_aligned_analysis_v2`.
- Formal training failures, OOM, NaN, and checkpoint failures: none.
- A report-render pass initially found the optional `tabulate` package absent; the implementation was changed to reuse the dependency-free S0 Markdown renderer. No package was installed and no training was repeated.

No model is advanced to multi-seed unless every gate is true. Validation parity FC was not used for early stopping, mapping adjustment, or weight search. No test proteome was opened and no test prediction was generated.
