# MODEL V2 Stage S2D-R Gate Semantics Review

Status: COMPLETE_AND_PAUSED  
Nature: `protocol_clarification=true`; no training, inference, prediction change, checkpoint change, route change, threshold change or weight search.

## Literal legacy conclusion

The original mixed global pool is retained as `legacy_global_rank`. It contains seed1, seed2, the two-seed mean and historical baselines. Under that literal pool, S2C seed2 ranked third in all three proxies and `legacy_gate_1_global_rank_first=false`. This is a valid historical global ranking, but it cannot by itself answer whether the fixed route replicated against the corresponding seed2 experts, because seed1 and the two-seed mean are not seed2 single-expert controls.

| scheme | rank | planning_proxy |
|---|---|---|
| bounded_quality | 3 | 0.660615 |
| correlation_priority | 3 | 0.693692 |
| weighted_module_rank | 3 | 0.631818 |

## Corrected seed-matched replication pool

Included: S2C seed2, D0 seed2, D2 seed2 and S1 C seed2. Excluded: all seed1 experts, S2C seed1 and the two-seed mean. Historical non-seed baselines remain reference-only and are not ranked in this replication pool. The five underlying metric CSVs are byte-identical to the frozen S2D delivery; only pool membership and ranks were recomputed.

| scheme | rank | model | planning_proxy |
|---|---|---|---|
| bounded_quality | 1 | S2C seed2 | 0.660615 |
| bounded_quality | 2 | D2 seed2 | 0.659809 |
| bounded_quality | 3 | S1 C seed2 | 0.657587 |
| bounded_quality | 4 | D0 seed2 | 0.653996 |
| correlation_priority | 1 | S2C seed2 | 0.693692 |
| correlation_priority | 2 | D2 seed2 | 0.692797 |
| correlation_priority | 3 | S1 C seed2 | 0.692527 |
| correlation_priority | 4 | D0 seed2 | 0.686963 |
| weighted_module_rank | 1 | S2C seed2 | 0.866667 |
| weighted_module_rank | 2 | D2 seed2 | 0.666667 |
| weighted_module_rank | 3 | S1 C seed2 | 0.266667 |
| weighted_module_rank | 4 | D0 seed2 | 0.200000 |

S2C seed2 ranks first in `3/3` proxies. Corrected gate 1 (`gate_1_seed2_s2c_first_among_seed2_candidates_in_at_least_two_proxies`) is **True**.

| scheme | s2c_seed2_rank | s2c_seed2_planning_proxy | absolute_difference_vs_d2_seed2 | absolute_difference_vs_s1_c_seed2 | absolute_difference_vs_d0_seed2 |
|---|---|---|---|---|---|
| bounded_quality | 1 | 0.660615 | 0.000805 | 0.003027 | 0.006619 |
| correlation_priority | 1 | 0.693692 | 0.000895 | 0.001165 | 0.006729 |
| weighted_module_rank | 1 | 0.866667 | 0.200000 | 0.600000 | 0.666667 |

## Preserved gates and final determination

Original gates 2-10 remain byte-derived and all true. The two-seed mean remains globally first in all three proxies. S2C seed2 mean RMSE is `0.651385` versus D2 seed2 `0.655616`. Mean raw-FC PCC is `0.299944` versus `0.303958` (difference `-0.004015`, within the original -0.005 tolerance).

- Literal legacy conclusion: gate 1 false; legacy all-gates eligibility false.
- Corrected replication conclusion: gate 1 true; `eligible_for_final_presubmission_audit=True`.
- `eligibility_basis=corrected_seed_matched_replication_gate`.

This is a comparison-pool semantic correction, not a threshold adjustment, output-weight search or post-hoc model change.

## Immutability and data-boundary audit

All ten frozen S2D report/JSON/metric/routing files match the SHA-256 values captured before this review. Routing remains `f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4`. The two-seed metrics artifact remains `06fd9a77369e6df0e94c4f8ae547af09aa85de994f15aca04da0e2e37724d8d5`. Original S2D did not materialize raw prediction tensors, so an independent raw-tensor hash is unavailable without prohibited re-inference; this review therefore records that limitation rather than fabricating a hash. It did not read or write predictions. `test_proteome_opened=false`; `test_prediction_generated=false`.

Machine-readable detail: `reports/model_v2_stage_s2d/gate_semantics_review.json`.

## Verification

- `python -m pytest --import-mode=importlib baseline/tests/test_stage_s2d_gate_semantics.py -q` -> `6 passed in 4.42s`.
- `python -m pytest --import-mode=importlib baseline/tests -q --ignore=baseline/tests/test_person_c.py` -> `115 passed in 68.83s`.
- Person C regression -> `6 passed in 3.95s`.

Stage S2D-R is paused for Main review.
