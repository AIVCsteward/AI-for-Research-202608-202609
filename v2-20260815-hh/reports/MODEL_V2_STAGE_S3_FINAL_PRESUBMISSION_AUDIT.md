# MODEL V2 Stage S3 Final Pre-submission Audit

Status: **PASS**  
`eligible_for_manual_submission=true`  
`planning_proxy=true`, `official_score=false`, `experimental_nonofficial_parity_fc=true`, `official_fc_result=false`.

## Outcome

The frozen S2C two-seed mean candidate was regenerated without training, fine-tuning, route changes, checkpoint changes, calibration, clipping, or weight search. Validation was replayed in an independent Python process from all six checkpoints and matched the published S2D metrics within the declared floating-point tolerance. Two independent test-metadata inference workers produced identical float32 arrays and route manifests.

- Candidate: `D:\虚拟细胞\submissions\model_v2_s2c_two_seed\prediction.csv`
- SHA-256: `9739f087788bfd37f57750564c9d351bbf37bd867ab7e87820294eec52c8dbe9`
- Shape after standard CSV re-read: `4454 x 4423`
- Route counts: D0 `2997`, S1 C `1322`, D2 `135`
- Test proteome opened: `false`
- Competition upload performed: `false`

## Frozen candidate and validation replay

- Six checkpoint/config hash audit: `PASS`.
- Original S2D artifact hashes unchanged: `true`.
- Validation routing SHA-256: `f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4`.
- Replayed mean RMSE: `0.638989552855`.
- Replayed mean raw-FC PCC: `0.304467385473`.
- Absolute, raw-FC, context residual, drug residual, high-effect, and three planning-proxy tables all matched: `true`.

## Submission contract and determinism

All 20 required CSV checks plus exact-once routing and control/QC response-zero checks passed: `true`. Protein columns were read dynamically from the frozen feature contract; comma-containing names were written and re-read with pandas' standards-compliant CSV implementation. No exponentiation, internal z-score output, rounding reduction, or post-processing was performed.

- Pass 1 CSV SHA-256: `9739f087788bfd37f57750564c9d351bbf37bd867ab7e87820294eec52c8dbe9`
- Pass 2 CSV SHA-256: `9739f087788bfd37f57750564c9d351bbf37bd867ab7e87820294eec52c8dbe9`
- Maximum numeric difference: `0.0`
- Inconsistent cells: `0`
- Test route SHA-256: `7b1d399f43d745ea892d55bf1c98f02c540c6c67de45bbe1f8643d4cb180992d`

The earlier stages froze only test route counts and train seen-set hashes, not a rowwise test-route hash. S3 therefore verified the historical counts and seen-set hashes, then required the two independent workers' newly materialized rowwise route hashes to be identical. No historical hash was fabricated.

## Numerical sanity (warning-only)

- Prediction min/max/mean/std: `11.084402` / `33.474312` / `21.060694` / `2.853480`
- P0.1/P1/P50/P99/P99.9: `14.080093` / `15.386377` / `20.896969` / `28.569482` / `30.382498`
- Constant proteins: `0`
- Very-low-variance proteins (<1e-6): `0`
- Cells outside the per-protein train-observed log2 range: `11` (`0.000056%`)

Finite out-of-range values were reported without clipping, winsorization, rescaling, calibration, sample deletion, protein deletion, or ensemble-weight changes.

## Data boundaries and disclosures

The test inference entry used an explicit filename guard and read only `WAYB_WAYC_metadata_test(1).csv` for test inputs/routing/sample IDs. Train-derived checkpoint state supplies protein means, scalers, bases and vocabularies. The local legacy `baseline_models/prediction.csv` was opened for exactly one header row; no prediction values were read or used.

The final candidate uses the frozen PubChem/RDKit chemistry features and public yeast-genome features with their source manifests and hashes. Oligomycin A proxy, Tunicamycin unresolved status, Cisplatin/NaCl structure-invalid fallbacks, and DHY210-to-S288C proxy are disclosed in `submission_audit.json`. NetwoRx, Parsons and Hillenmeyer did not enter the final candidate and no research-only artifact was copied into the submission directory.

## Rules and unresolved official semantics

The candidate preserves `sample_ID`, 4,454 metadata rows, the frozen 4,422-protein order, and log2 scale. Raw competition data were not uploaded. Reproduction commands, environment versions, checkpoint/config hashes and source-manifest hashes are recorded.

Still explicitly unconfirmed: official Water/DMSO mapping, extra official matched-control QC, the official multi-control aggregation rule, and the official within-module final aggregation formula. The parity FC used during historical D2/S1 C training remains experimental and nonofficial.

## Execution notes and verification

Two fail-closed implementation attempts occurred before the final successful run. The first replay completed but the comparison reader tried to sort the planning-proxy table by a nonexistent `scenario` column; no test inference or candidate CSV was generated. The second completed validation replay and both test inferences, then failed while serializing a NumPy boolean in the contract audit. Its unaudited candidate and temporary workers were moved intact to `submissions/model_v2_s2c_two_seed/audit_failed_serialization_attempt_1`; they were not overwritten or promoted. The serializer and materialization order were corrected, and the complete S3 pipeline was rerun from frozen hash verification.

- S3 tests: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest --import-mode=importlib baseline\tests\test_stage_s3_final_submission_v2.py -q` -> `6 passed in 5.12s`.
- All V2 tests excluding the separately run Person C suite: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest --import-mode=importlib baseline\tests -q --ignore=baseline\tests\test_person_c.py` -> `121 passed in 55.70s`.
- Person C regression from `D:\虚拟细胞\baseline`: `D:\虚拟细胞\.venv\Scripts\python.exe -m pytest tests\test_person_c.py -q` -> `6 passed in 3.88s`.
- Pytest emitted one collection warning because the runtime guard class name begins with `Test`; it has a constructor and is intentionally not a pytest test class. The actual guard tests passed, and inference workers recorded no warnings or errors.

## Reproduction and artifacts

- Command (writes to new, non-overwriting reproduction directories): `"D:\虚拟细胞\.venv\Scripts\python.exe" -m baseline.final_submission_v2 --root "D:\虚拟细胞" --report-dir "D:\虚拟细胞\reports\model_v2_stage_s3_reproduction" --submission-dir "D:\虚拟细胞\submissions\model_v2_s2c_two_seed_reproduction" --device cuda`
- Final candidate manifest: `D:\虚拟细胞\reports\model_v2_stage_s3\final_candidate_manifest.json`
- Validation replay: `D:\虚拟细胞\reports\model_v2_stage_s3\validation_replay_metrics.json`
- Contract audit: `D:\虚拟细胞\reports\model_v2_stage_s3\submission_contract_audit.json`
- Numerical sanity: `D:\虚拟细胞\reports\model_v2_stage_s3\numerical_sanity_report.json`
- Determinism: `D:\虚拟细胞\reports\model_v2_stage_s3\determinism_audit.json`
- Data access: `D:\虚拟细胞\reports\model_v2_stage_s3\data_access_audit.json`

Stage S3 is complete and paused. The candidate was not uploaded; Main must perform the final manual review and any platform upload.
