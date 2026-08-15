# MODEL V2 Stage S2D Second-Seed Report

Status: COMPLETE_AND_PAUSED  
Labels: `planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

Seed2 retrained both anchors from Stage A: hierarchical used all 751 train controls for exactly 30 epochs without validation selection; flat reproduced the S1 C control-only Stage A and selected epoch 4 from 205 public validation controls. D0 and D2 then shared the hierarchical checkpoint and identical response/chemical initialization hash `ae6cabcfcb83a41d50438d9f928a9e31caac18737d885620f8136a839c63b625`. No test proteome was opened and no test prediction was generated.

## Absolute metrics on the exact S0 common subset

| model | scenario | log2_rmse | mae | global_r2 | sample_pcc_median | sample_r2_median | protein_pcc_median | protein_r2_median |
|---|---|---|---|---|---|---|---|---|
| D0 seed1 | val_chem_only | 0.5478 | 0.3862 | 0.9596 | 0.9834 | 0.9636 | 0.8362 | 0.6641 |
| D0 seed2 | val_chem_only | 0.5678 | 0.4009 | 0.9566 | 0.9812 | 0.9589 | 0.8252 | 0.6466 |
| D2 seed1 | val_chem_only | 0.5794 | 0.4075 | 0.9548 | 0.9817 | 0.9585 | 0.8146 | 0.6282 |
| D2 seed2 | val_chem_only | 0.5882 | 0.4125 | 0.9534 | 0.9805 | 0.9563 | 0.8156 | 0.6261 |
| S1 C seed1 | val_chem_only | 0.6141 | 0.4321 | 0.9492 | 0.9778 | 0.9510 | 0.8090 | 0.5903 |
| S1 C seed2 | val_chem_only | 0.6302 | 0.4348 | 0.9465 | 0.9769 | 0.9519 | 0.8097 | 0.6130 |
| S2C seed1 | val_chem_only | 0.5478 | 0.3862 | 0.9596 | 0.9834 | 0.9636 | 0.8362 | 0.6641 |
| S2C seed2 | val_chem_only | 0.5678 | 0.4009 | 0.9566 | 0.9812 | 0.9589 | 0.8252 | 0.6466 |
| S2C two-seed mean | val_chem_only | 0.5492 | 0.3861 | 0.9594 | 0.9829 | 0.9622 | 0.8382 | 0.6671 |
| D0 seed1 | val_strain_only | 0.7653 | 0.5283 | 0.9194 | 0.9626 | 0.9209 | 0.7559 | 0.3247 |
| D0 seed2 | val_strain_only | 0.7497 | 0.5199 | 0.9227 | 0.9643 | 0.9247 | 0.7535 | 0.3272 |
| D2 seed1 | val_strain_only | 0.7529 | 0.5190 | 0.9220 | 0.9638 | 0.9240 | 0.7795 | 0.3732 |
| D2 seed2 | val_strain_only | 0.7356 | 0.5125 | 0.9256 | 0.9652 | 0.9259 | 0.7675 | 0.3591 |
| S1 C seed1 | val_strain_only | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| S1 C seed2 | val_strain_only | 0.7488 | 0.5106 | 0.9229 | 0.9620 | 0.9201 | 0.7800 | 0.3766 |
| S2C seed1 | val_strain_only | 0.7180 | 0.4983 | 0.9291 | 0.9674 | 0.9321 | 0.7730 | 0.3758 |
| S2C seed2 | val_strain_only | 0.7488 | 0.5106 | 0.9229 | 0.9620 | 0.9201 | 0.7800 | 0.3766 |
| S2C two-seed mean | val_strain_only | 0.7181 | 0.4937 | 0.9291 | 0.9657 | 0.9277 | 0.7947 | 0.4077 |
| D0 seed1 | val_both | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| D0 seed2 | val_both | 0.7832 | 0.5468 | 0.9173 | 0.9622 | 0.9180 | 0.8236 | 0.4570 |
| D2 seed1 | val_both | 0.8114 | 0.5650 | 0.9112 | 0.9614 | 0.9135 | 0.8039 | 0.4159 |
| D2 seed2 | val_both | 0.7930 | 0.5537 | 0.9152 | 0.9623 | 0.9166 | 0.8073 | 0.4296 |
| S1 C seed1 | val_both | 0.8054 | 0.5661 | 0.9125 | 0.9608 | 0.9116 | 0.8052 | 0.3986 |
| S1 C seed2 | val_both | 0.8637 | 0.5966 | 0.8994 | 0.9499 | 0.8959 | 0.7768 | 0.3622 |
| S2C seed1 | val_both | 0.8083 | 0.5605 | 0.9119 | 0.9609 | 0.9122 | 0.8257 | 0.4432 |
| S2C seed2 | val_both | 0.7832 | 0.5468 | 0.9173 | 0.9622 | 0.9180 | 0.8236 | 0.4570 |
| S2C two-seed mean | val_both | 0.7894 | 0.5483 | 0.9160 | 0.9623 | 0.9170 | 0.8317 | 0.4603 |
| D0 seed1 | val_time | 0.5496 | 0.3812 | 0.9590 | 0.9834 | 0.9619 | 0.7599 | 0.5551 |
| D0 seed2 | val_time | 0.5627 | 0.3901 | 0.9571 | 0.9822 | 0.9594 | 0.7458 | 0.5326 |
| D2 seed1 | val_time | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |
| D2 seed2 | val_time | 0.5057 | 0.3525 | 0.9653 | 0.9850 | 0.9679 | 0.7996 | 0.6236 |
| S1 C seed1 | val_time | 0.5402 | 0.3740 | 0.9604 | 0.9839 | 0.9648 | 0.7652 | 0.5648 |
| S1 C seed2 | val_time | 0.4749 | 0.3292 | 0.9694 | 0.9872 | 0.9724 | 0.8204 | 0.6611 |
| S2C seed1 | val_time | 0.5093 | 0.3526 | 0.9648 | 0.9851 | 0.9680 | 0.8009 | 0.6272 |
| S2C seed2 | val_time | 0.5057 | 0.3525 | 0.9653 | 0.9850 | 0.9679 | 0.7996 | 0.6236 |
| S2C two-seed mean | val_time | 0.4993 | 0.3456 | 0.9662 | 0.9859 | 0.9693 | 0.8071 | 0.6355 |

## Raw FC and high-effect metrics

| model | scenario | global_fc_pcc | sample_fc_pcc_median | protein_fc_pcc_median | fc_rmse | fc_direction_accuracy |
|---|---|---|---|---|---|---|
| D0 seed1 | val_chem_only | 0.3274 | 0.3281 | 0.3529 | 0.5478 | 0.6105 |
| D0 seed2 | val_chem_only | 0.3158 | 0.3165 | 0.3390 | 0.5678 | 0.6043 |
| D2 seed1 | val_chem_only | 0.3235 | 0.3278 | 0.3461 | 0.5794 | 0.6099 |
| D2 seed2 | val_chem_only | 0.3196 | 0.3270 | 0.3432 | 0.5882 | 0.6075 |
| S1 C seed1 | val_chem_only | 0.3106 | 0.3095 | 0.3494 | 0.6141 | 0.6076 |
| S1 C seed2 | val_chem_only | 0.2992 | 0.3154 | 0.3462 | 0.6302 | 0.6048 |
| S2C seed1 | val_chem_only | 0.3274 | 0.3281 | 0.3529 | 0.5478 | 0.6105 |
| S2C seed2 | val_chem_only | 0.3158 | 0.3165 | 0.3390 | 0.5678 | 0.6043 |
| S2C two-seed mean | val_chem_only | 0.3268 | 0.3270 | 0.3530 | 0.5492 | 0.6098 |
| D0 seed1 | val_strain_only | 0.2240 | 0.2232 | 0.3795 | 0.7653 | 0.5788 |
| D0 seed2 | val_strain_only | 0.2332 | 0.2303 | 0.3854 | 0.7497 | 0.5824 |
| D2 seed1 | val_strain_only | 0.2495 | 0.2393 | 0.4203 | 0.7529 | 0.5888 |
| D2 seed2 | val_strain_only | 0.2625 | 0.2476 | 0.4163 | 0.7356 | 0.5904 |
| S1 C seed1 | val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| S1 C seed2 | val_strain_only | 0.2595 | 0.2416 | 0.4370 | 0.7488 | 0.5943 |
| S2C seed1 | val_strain_only | 0.2615 | 0.2545 | 0.4064 | 0.7180 | 0.5943 |
| S2C seed2 | val_strain_only | 0.2595 | 0.2416 | 0.4370 | 0.7488 | 0.5943 |
| S2C two-seed mean | val_strain_only | 0.2664 | 0.2537 | 0.4383 | 0.7181 | 0.5968 |
| D0 seed1 | val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| D0 seed2 | val_both | 0.2203 | 0.2071 | 0.3700 | 0.7832 | 0.5745 |
| D2 seed1 | val_both | 0.2189 | 0.2089 | 0.3615 | 0.8114 | 0.5731 |
| D2 seed2 | val_both | 0.2296 | 0.2184 | 0.3664 | 0.7930 | 0.5771 |
| S1 C seed1 | val_both | 0.2306 | 0.2181 | 0.3712 | 0.8054 | 0.5784 |
| S1 C seed2 | val_both | 0.2063 | 0.1907 | 0.3518 | 0.8637 | 0.5720 |
| S2C seed1 | val_both | 0.2090 | 0.1994 | 0.3683 | 0.8083 | 0.5708 |
| S2C seed2 | val_both | 0.2203 | 0.2071 | 0.3700 | 0.7832 | 0.5745 |
| S2C two-seed mean | val_both | 0.2165 | 0.2054 | 0.3781 | 0.7894 | 0.5736 |
| D0 seed1 | val_time | 0.3149 | 0.3327 | 0.3258 | 0.5496 | 0.6055 |
| D0 seed2 | val_time | 0.3099 | 0.3143 | 0.3250 | 0.5627 | 0.6044 |
| D2 seed1 | val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |
| D2 seed2 | val_time | 0.4042 | 0.3691 | 0.4118 | 0.5057 | 0.6267 |
| S1 C seed1 | val_time | 0.3493 | 0.3507 | 0.3600 | 0.5402 | 0.6197 |
| S1 C seed2 | val_time | 0.4344 | 0.4128 | 0.4425 | 0.4749 | 0.6388 |
| S2C seed1 | val_time | 0.3980 | 0.3834 | 0.4076 | 0.5093 | 0.6268 |
| S2C seed2 | val_time | 0.4042 | 0.3691 | 0.4118 | 0.5057 | 0.6267 |
| S2C two-seed mean | val_time | 0.4082 | 0.3820 | 0.4179 | 0.4993 | 0.6300 |

| model | scenario | direction_accuracy | high_effect_pcc | precision | recall | f1 | auprc |
|---|---|---|---|---|---|---|---|
| D0 seed1 | val_chem_only | 0.8080 | 0.6190 | 0.1166 | 0.2923 | 0.1667 | 0.0964 |
| D0 seed2 | val_chem_only | 0.8069 | 0.6152 | 0.1106 | 0.3066 | 0.1625 | 0.0932 |
| D2 seed1 | val_chem_only | 0.8121 | 0.6251 | 0.1061 | 0.3244 | 0.1599 | 0.0927 |
| D2 seed2 | val_chem_only | 0.8134 | 0.6229 | 0.1046 | 0.3304 | 0.1590 | 0.0900 |
| S1 C seed1 | val_chem_only | 0.8053 | 0.6143 | 0.0933 | 0.3391 | 0.1463 | 0.0842 |
| S1 C seed2 | val_chem_only | 0.7920 | 0.6063 | 0.0927 | 0.3504 | 0.1466 | 0.0835 |
| S2C seed1 | val_chem_only | 0.8080 | 0.6190 | 0.1166 | 0.2923 | 0.1667 | 0.0964 |
| S2C seed2 | val_chem_only | 0.8069 | 0.6152 | 0.1106 | 0.3066 | 0.1625 | 0.0932 |
| S2C two-seed mean | val_chem_only | 0.8112 | 0.6211 | 0.1168 | 0.2951 | 0.1673 | 0.0965 |
| D0 seed1 | val_strain_only | 0.7479 | 0.4821 | 0.0841 | 0.3435 | 0.1351 | 0.0715 |
| D0 seed2 | val_strain_only | 0.7571 | 0.4914 | 0.0859 | 0.3443 | 0.1375 | 0.0742 |
| D2 seed1 | val_strain_only | 0.7836 | 0.5341 | 0.0882 | 0.3651 | 0.1421 | 0.0758 |
| D2 seed2 | val_strain_only | 0.7873 | 0.5536 | 0.0909 | 0.3691 | 0.1459 | 0.0795 |
| S1 C seed1 | val_strain_only | 0.7853 | 0.5433 | 0.0891 | 0.3398 | 0.1412 | 0.0764 |
| S1 C seed2 | val_strain_only | 0.7970 | 0.5621 | 0.0888 | 0.3567 | 0.1423 | 0.0739 |
| S2C seed1 | val_strain_only | 0.7853 | 0.5433 | 0.0891 | 0.3398 | 0.1412 | 0.0764 |
| S2C seed2 | val_strain_only | 0.7970 | 0.5621 | 0.0888 | 0.3567 | 0.1423 | 0.0739 |
| S2C two-seed mean | val_strain_only | 0.7979 | 0.5639 | 0.0910 | 0.3393 | 0.1435 | 0.0756 |
| D0 seed1 | val_both | 0.7607 | 0.5142 | 0.0627 | 0.3584 | 0.1068 | 0.0552 |
| D0 seed2 | val_both | 0.7684 | 0.5284 | 0.0654 | 0.3599 | 0.1106 | 0.0586 |
| D2 seed1 | val_both | 0.7728 | 0.5347 | 0.0624 | 0.3711 | 0.1068 | 0.0567 |
| D2 seed2 | val_both | 0.7750 | 0.5437 | 0.0651 | 0.3749 | 0.1110 | 0.0595 |
| S1 C seed1 | val_both | 0.7788 | 0.5441 | 0.0610 | 0.3748 | 0.1049 | 0.0568 |
| S1 C seed2 | val_both | 0.7585 | 0.5156 | 0.0586 | 0.3860 | 0.1017 | 0.0546 |
| S2C seed1 | val_both | 0.7607 | 0.5142 | 0.0627 | 0.3584 | 0.1068 | 0.0552 |
| S2C seed2 | val_both | 0.7684 | 0.5284 | 0.0654 | 0.3599 | 0.1106 | 0.0586 |
| S2C two-seed mean | val_both | 0.7681 | 0.5241 | 0.0644 | 0.3540 | 0.1090 | 0.0572 |
| D0 seed1 | val_time | 0.7591 | 0.5234 | 0.1704 | 0.2327 | 0.1968 | 0.1147 |
| D0 seed2 | val_time | 0.7589 | 0.5215 | 0.1586 | 0.2390 | 0.1907 | 0.1113 |
| D2 seed1 | val_time | 0.8316 | 0.6615 | 0.2233 | 0.2914 | 0.2529 | 0.1651 |
| D2 seed2 | val_time | 0.8332 | 0.6700 | 0.2264 | 0.2933 | 0.2556 | 0.1690 |
| S1 C seed1 | val_time | 0.7883 | 0.5727 | 0.1775 | 0.2592 | 0.2107 | 0.1282 |
| S1 C seed2 | val_time | 0.8444 | 0.6939 | 0.2552 | 0.2789 | 0.2665 | 0.1871 |
| S2C seed1 | val_time | 0.8316 | 0.6615 | 0.2233 | 0.2914 | 0.2529 | 0.1651 |
| S2C seed2 | val_time | 0.8332 | 0.6700 | 0.2264 | 0.2933 | 0.2556 | 0.1690 |
| S2C two-seed mean | val_time | 0.8358 | 0.6708 | 0.2323 | 0.2874 | 0.2570 | 0.1708 |

## Context/drug residual modules

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_chem_only | 0.1880 | 0.3808 | 0.5636 |
| V2 batch-enabled Huber | val_chem_only | 0.1978 | 0.6188 | 0.5566 |
| S1 B anchor Morgan Huber | val_chem_only | 0.2024 | 0.5954 | 0.5672 |
| D0 seed1 | val_chem_only | 0.2146 | 0.5468 | 0.5678 |
| D0 seed2 | val_chem_only | 0.2081 | 0.5669 | 0.5655 |
| D2 seed1 | val_chem_only | 0.2081 | 0.5786 | 0.5657 |
| D2 seed2 | val_chem_only | 0.2067 | 0.5874 | 0.5661 |
| S1 C seed1 | val_chem_only | 0.1973 | 0.6132 | 0.5639 |
| S1 C seed2 | val_chem_only | 0.1945 | 0.6295 | 0.5677 |
| S2C seed1 | val_chem_only | 0.2146 | 0.5468 | 0.5678 |
| S2C seed2 | val_chem_only | 0.2081 | 0.5669 | 0.5655 |
| S2C two-seed mean | val_chem_only | 0.2152 | 0.5482 | 0.5684 |

| model | scenario | pcc | rmse | direction_accuracy |
|---|---|---|---|---|
| Matched Control | val_strain_only | 0.1372 | 0.4101 | 0.5370 |
| V2 batch-enabled Huber | val_strain_only | 0.2839 | 0.7368 | 0.5979 |
| S1 B anchor Morgan Huber | val_strain_only | 0.2864 | 0.7275 | 0.5974 |
| D0 seed1 | val_strain_only | 0.2654 | 0.7653 | 0.5893 |
| D0 seed2 | val_strain_only | 0.2727 | 0.7497 | 0.5929 |
| D2 seed1 | val_strain_only | 0.2554 | 0.7529 | 0.5890 |
| D2 seed2 | val_strain_only | 0.2671 | 0.7356 | 0.5914 |
| S1 C seed1 | val_strain_only | 0.2914 | 0.7180 | 0.6005 |
| S1 C seed2 | val_strain_only | 0.2681 | 0.7488 | 0.5966 |
| S2C seed1 | val_strain_only | 0.2914 | 0.7180 | 0.6005 |
| S2C seed2 | val_strain_only | 0.2681 | 0.7488 | 0.5966 |
| S2C two-seed mean | val_strain_only | 0.2857 | 0.7181 | 0.6012 |

## Seed stability

| model | scenario | seed_prediction_pcc | seed_prediction_rmse | response_direction_agreement | response_pcc |
|---|---|---|---|---|---|
| S2B D0 hierarchical none Huber | val_chem_only | 0.9973 | 0.1963 | 0.7875 | 0.7987 |
| S2B D2 hierarchical Morgan experimental FC | val_chem_only | 0.9968 | 0.2156 | 0.7919 | 0.8636 |
| S1 C anchor Morgan Huber plus experimental FC | val_chem_only | 0.9921 | 0.3541 | 0.6067 | 0.5654 |
| S2C routed ensemble | val_chem_only | 0.9973 | 0.1963 | 0.7875 | 0.7987 |
| S2B D0 hierarchical none Huber | val_strain_only | 0.9974 | 0.1926 | 0.7859 | 0.7836 |
| S2B D2 hierarchical Morgan experimental FC | val_strain_only | 0.9972 | 0.2014 | 0.8029 | 0.8648 |
| S1 C anchor Morgan Huber plus experimental FC | val_strain_only | 0.9938 | 0.3000 | 0.6531 | 0.5458 |
| S2C routed ensemble | val_strain_only | 0.9938 | 0.3000 | 0.6531 | 0.5458 |
| S2B D0 hierarchical none Huber | val_both | 0.9973 | 0.2015 | 0.7870 | 0.7790 |
| S2B D2 hierarchical Morgan experimental FC | val_both | 0.9963 | 0.2338 | 0.7526 | 0.7992 |
| S1 C anchor Morgan Huber plus experimental FC | val_both | 0.9904 | 0.3889 | 0.6050 | 0.4813 |
| S2C routed ensemble | val_both | 0.9973 | 0.2015 | 0.7870 | 0.7790 |
| S2B D0 hierarchical none Huber | val_time | 0.9975 | 0.1876 | 0.7807 | 0.7996 |
| S2B D2 hierarchical Morgan experimental FC | val_time | 0.9977 | 0.1822 | 0.8294 | 0.8979 |
| S1 C anchor Morgan Huber plus experimental FC | val_time | 0.9954 | 0.2561 | 0.6871 | 0.6524 |
| S2C routed ensemble | val_time | 0.9977 | 0.1822 | 0.8294 | 0.8979 |

Per-scenario seed2-minus-seed1 differences for every absolute, raw-FC, residual and high-effect metric are materialized in `seed_metric_differences.csv`.

## Training runs

| run | stage | seed | elapsed_seconds | actual_epochs | best_epoch | peak_gpu_memory_reserved_bytes |
|---|---|---|---|---|---|---|
| hierarchical_seed20260814 | A | 20260814 | NA | NA | NA | NA |
| hierarchical_seed20260815 | A | 20260815 | 26.0769 | 30.0000 | 29.0000 | 62914560.0000 |
| flat_seed20260814 | A | 20260814 | NA | NA | NA | NA |
| flat_seed20260815 | A | 20260815 | 27.2853 | 11.0000 | 4.0000 | 90177536.0000 |
| D0 seed1 | B | 20260814 | 103.6736 | 33.0000 | 20.0000 | 94371840.0000 |
| D2 seed1 | B | 20260814 | 99.0270 | 25.0000 | 12.0000 | 94371840.0000 |
| S1 C seed1 | B | 20260814 | 87.4927 | 20.0000 | 7.0000 | 88080384.0000 |
| D0 seed2 | B | 20260815 | 72.2686 | 23.0000 | 10.0000 | 94371840.0000 |
| D2 seed2 | B | 20260815 | 100.0404 | 29.0000 | 16.0000 | 94371840.0000 |
| S1 C seed2 | B | 20260815 | 177.5822 | 60.0000 | 47.0000 | 90177536.0000 |

Per-drug gains use one fixed comparator: `RMSE_matched_control - RMSE_expert`. This makes the sign directly comparable across seeds without using either seed to define the reference. Full rows are in `seed_stability_per_entity.csv`; paired gain signs are in `expert_per_entity_gain_consistency.csv`.

| expert | n_drugs | consistent_fraction | mean_gain_seed1 | mean_gain_seed2 |
|---|---|---|---|---|
| D0 | 85 | 0.9882 | -0.2246 | -0.2238 |
| D2 | 85 | 0.9882 | -0.2108 | -0.1970 |
| S1 C | 85 | 0.9176 | -0.2048 | -0.1959 |

Test-metadata responsibility remains D0 2997, S1 C 1322, D2 135 samples. D0 therefore dominates deployment exposure; its scenario prediction stability is shown independently above rather than being hidden by D2's smaller route share.

## Planning proxies and gates

| scheme | rank | model | planning_proxy | official_score |
|---|---|---|---|---|
| bounded_quality | 1 | S2C two-seed mean | 0.6647 | False |
| bounded_quality | 2 | S2C seed1 | 0.6631 | False |
| bounded_quality | 3 | S2C seed2 | 0.6606 | False |
| bounded_quality | 4 | D2 seed2 | 0.6598 | False |
| bounded_quality | 5 | D2 seed1 | 0.6581 | False |
| bounded_quality | 6 | S1 C seed2 | 0.6576 | False |
| bounded_quality | 7 | S1 C seed1 | 0.6568 | False |
| bounded_quality | 8 | S1 B anchor Morgan Huber | 0.6545 | False |
| bounded_quality | 9 | D0 seed1 | 0.6542 | False |
| bounded_quality | 10 | D0 seed2 | 0.6540 | False |
| bounded_quality | 11 | V2 batch-enabled Huber | 0.6520 | False |
| bounded_quality | 12 | Matched Control | 0.5000 | False |
| correlation_priority | 1 | S2C two-seed mean | 0.6974 | False |
| correlation_priority | 2 | S2C seed1 | 0.6958 | False |
| correlation_priority | 3 | S2C seed2 | 0.6937 | False |
| correlation_priority | 4 | D2 seed2 | 0.6928 | False |
| correlation_priority | 5 | S1 C seed2 | 0.6925 | False |
| correlation_priority | 6 | D2 seed1 | 0.6914 | False |
| correlation_priority | 7 | S1 C seed1 | 0.6908 | False |
| correlation_priority | 8 | S1 B anchor Morgan Huber | 0.6878 | False |
| correlation_priority | 9 | D0 seed1 | 0.6872 | False |
| correlation_priority | 10 | D0 seed2 | 0.6870 | False |
| correlation_priority | 11 | V2 batch-enabled Huber | 0.6860 | False |
| correlation_priority | 12 | Matched Control | 0.4832 | False |
| weighted_module_rank | 1 | S2C two-seed mean | 0.9182 | False |
| weighted_module_rank | 2 | S2C seed1 | 0.8000 | False |
| weighted_module_rank | 3 | S2C seed2 | 0.6318 | False |
| weighted_module_rank | 4 | D2 seed2 | 0.6045 | False |
| weighted_module_rank | 5 | Matched Control | 0.5091 | False |
| weighted_module_rank | 6 | S1 C seed1 | 0.4727 | False |
| weighted_module_rank | 7 | D2 seed1 | 0.4318 | False |
| weighted_module_rank | 8 | S1 C seed2 | 0.4136 | False |
| weighted_module_rank | 9 | D0 seed1 | 0.3682 | False |
| weighted_module_rank | 10 | S1 B anchor Morgan Huber | 0.3636 | False |
| weighted_module_rank | 11 | D0 seed2 | 0.3227 | False |
| weighted_module_rank | 12 | V2 batch-enabled Huber | 0.1636 | False |

```json
{
  "gate_1_seed2_s2c_first_in_at_least_two_planning_proxies": false,
  "gate_2_seed2_s2c_mean_rmse_better_than_seed2_d2": true,
  "gate_3_seed2_s2c_mean_raw_fc_drop_vs_seed2_d2_at_most_0_005": true,
  "gate_4_seed_routes_rowwise_identical": true,
  "gate_5_all_three_experts_finite_without_training_collapse": true,
  "gate_6_two_seed_mean_first_in_all_three_planning_proxies": true,
  "gate_7_mean_rmse_degradation_vs_better_single_at_most_1_percent": true,
  "gate_8_mean_raw_fc_drop_vs_better_single_at_most_0_005": true,
  "gate_9_improvement_not_single_drug_or_small_scenario_dominated": true,
  "gate_10_test_proteome_not_read": true
}
```

Eligible for final pre-submission audit: **False**. Mean RMSE is `0.638990` versus the better single seed `0.645840`; mean raw-FC PCC is `0.304467` versus the better single seed `0.299944`. No third seed, route edit, FC-weight search, output calibration, test inference, or checkpoint modification was performed.

The strict seed2 reproduction gate failed: S2C seed2 ranked third, not first, in all three planning proxies (although it remained better than seed2 D2 on mean RMSE and within the raw-FC tolerance). The fixed two-seed mean ranked first in all three proxies and passed gates 2-10, but the predeclared rule requires every gate, so it is **not** promoted. Following the failure discipline, no third seed or route/weight adjustment was attempted.

One failed analysis attempt occurred after all inference metrics had been computed: JSON writing rejected a NumPy boolean in the checkpoint manifest. Boolean normalization was added and the full inference audit was rerun successfully; no training or checkpoint mutation occurred in either attempt.

## Reproducibility

- Seed2 hierarchical Stage A: `reports/model_v2_stage_s2d/stage_a_hierarchical_seed_20260815`.
- Seed2 flat Stage A: `reports/model_v2_stage_s2d/stage_a_flat_seed_20260815`.
- Seed2 experts: `formal_d0_hierarchical_none_huber_seed_20260815`, `formal_d2_hierarchical_morgan_fc_seed_20260815`, `formal_s1c_flat_morgan_fc_seed_20260815` under `reports/model_v2_stage_s2d`.
- Checkpoint, resolved-config, training-ID, holdout, seed and artifact hashes: `checkpoint_manifest.json`.
- Route identity: seed1 `f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4`, seed2 `f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4`, rowwise identical `True`, test route counts identical `True`.
- Analysis command: `D:\虚拟细胞\.venv\Scripts\python.exe -m baseline.second_seed_ensemble_v2`.

## Verification commands

- S2D tests: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest --import-mode=importlib baseline\tests\test_stage_s2d_second_seed_ensemble_v2.py -q` -> `6 passed in 3.89s`.
- All V2 tests excluding the separately run Person C suite: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest --import-mode=importlib baseline\tests -q --ignore=baseline\tests\test_person_c.py` -> `109 passed in 63.88s`.
- Person C regression from `D:\虚拟细胞\baseline`: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest tests\test_person_c.py -q` -> `6 passed in 3.54s`.
- The first S2D-test run was `5 passed, 1 failed`: pandas interpreted the entirely blank `auprc_unavailable_reason` explanation column as numeric NaN. The assertion was restricted to actual numeric metric columns; the rerun passed. No metric or model output changed.

Stage S2D is paused for Main review.
