# Model V2 stage 0/1 report

Task: Task 4 - V1 baseline freeze and control-anchored chemical-genome virtual cell V2  
Task ID: model-v2  
Working branch: not applicable, local mode  
Commit: none  
Date: 2026-08-14  
Stage gate: stage 0 PASS; stage 1 final targeted correction complete; paused for Main acceptance

## 1. Outcome

Stage 0 froze a reproducible minimal ConditionMLP and documented why the old
full-model results are not reproducible. Main accepted stage 0. The first two
stage-1 reviews closed the model contract, checkpoint, isolation and leakage
gaps. This final targeted correction fixes stage-specific parameter updates,
valid-position early-stopping aggregation, the formal configuration-driven CLI,
and seven non-FC validation metrics. No stage-2 long training, ablation, seed
sweep, experimental FC run, or test prediction was started.

## 1.1 Targeted completion: before and after

| Main finding | Before | After |
|---|---|---|
| Required delivery paths | Logic lived primarily in root `model_v2`. | Canonical code is exclusively in `baseline/baseline/model_v2.py`, `training_v2.py`, `losses_v2.py`, `evaluation_v2.py`; root modules are compatibility re-exports/CLI wrappers. |
| Similarity return | Key absent. | `"similarity_gate": None`; no fabricated value. |
| Batch initialization/identifiability | Output random at initialization; L2 only. | Final projection zero-initialized; L2 plus across-sample per-protein centering penalty; diagnostics recorded. |
| Checkpoint | Could save a post-training optimizer unrelated to best epoch; no early-stop state or source-manifest hashes. | Best-improvement moment deep-copies real model and optimizer states, early-stop state, config, seed, monitor and stage; both source manifests are hashed; restore-and-next-step equality is tested. |
| Isolation tests | Mostly signatures. | Actual input-family perturbations verify all four required invariants under `dropout=0`, `eval()`. |
| Leakage tests | Train vocabulary guard only. | Runtime file-open guard plus validation-backward, basis, FC anchor, high-effect and CV fold-train-ID tests. |
| Training entry | Single in-memory batch smoke. | Mini-batch DataLoader; control-first A; treatment Huber-only B; equal four-scenario macro early stop; best restore; independent four-scenario evaluator. |
| Stage B updates | One shared optimizer could continue updating baseline/genome at the same learning rate. | Baseline and shared genome encoder are frozen by default; chemical and response groups train at `1e-3`; batch has an independent `1e-4` rate. A/B freeze states and rates are explicit in YAML. |
| Early-stop aggregation | Averaged per-batch mean Huber, making the result batch-size dependent. | Each scenario is `sum Huber over all valid positions / valid-position count`; monitor is the equal mean of exactly four scenarios. Batch-size invariance is tested. |
| Batch centering semantics | `batch_center_weight` was ambiguous. | Renamed `batch_center_strength`; it is internal to batch regularization, so the effective center coefficient is `batch_reg_weight * batch_center_strength = 1e-4`. Total-loss equality is tested. |
| Non-FC metrics | RMSE, global R2 and per-protein R2 only. | Each scenario reports mask-aware RMSE/MAE, global R2, per-sample PCC/R2 medians, and per-protein PCC/R2 medians, with joint masks and minimum-count/variance guards. |
| Baseline anchor | Smoke used selected controls plus treatments; formal mode used all train samples. | Both modes fit `train_protein_mean` from all 751 train Water/DMSO controls only. Treatment, QC and validation labels cannot affect it. |
| Stage A monitor | Used the same four full treatment/control validation scenarios as stage B. | Stage A uses only the 205 public validation Water/DMSO controls: 190 `val_strain_only` plus 15 `val_time`; `delta_response` is asserted exactly zero. Stage B retains the four complete scenarios. |
| Training history | Epoch records existed only in returned Python objects. | `training_history.json` persists every epoch's train loss, named monitor and details, bad-epoch count, and parameter-group learning rates. |

## 2. Canonical implementation

The model is split into interfaces that prevent prohibited cross-talk:

- baseline branch: genome latent, medium, temperature/time only;
- response branch: separately encoded Morgan and descriptors, genome latent,
  medium and temperature/time; never batch fields;
- batch branch: data source, instrument and plate only, with zero-initialized
  final projection and explicit centering/L2 constraints;
- output: `y_pred = y_baseline + delta_response + delta_batch`;
- Water/DMSO: `delta_response` is multiplied by an exact control mask and is
  therefore zero;
- protein decoder: low-rank basis sized dynamically from the feature contract;
- genome input: only `genome_features[24]`, its validity mask, mapping type,
  confidence and proxy flag; raw/QC/PCA arrays are not model inputs;
- chemistry input: frozen 2,048-bit Morgan and 217 descriptors are encoded
  separately, with dimension-level validity plus mapping/structure state;
- eligible-only shuffle preserves special controls, invalid/unresolved
  chemistry and the DHY210 proxy row.

The public formal entry is:

```powershell
.venv\Scripts\python.exe -m baseline.training_v2 --config baseline\configs\model_v2_huber.yaml --output-dir reports\model_v2_stage2\huber_seed_20260814
```

It selects all train controls for stage A and all non-control, non-QC train
treatments for stage B, uses mini-batches, independent A/B optimizers, early
stopping and best-checkpoint restoration. This command without `--smoke` was
deliberately **NOT RUN** in this round.

The baseline anchor and both checkpoints record:

- `anchor_source=train_controls_only`;
- 751 train control samples;
- 4,422/4,422 proteins observed in those controls;
- minimum 94 valid control observations for any contracted protein.

Formal similarity-FC transfer is not enabled. `model_v2_huber.yaml` fixes FC
weight to zero and similarity to false. `model_v2_experimental_fc.yaml` records
Main's permission to use `pert_id_parity_v1` only as
`experimental_inferred_mapping`: it is not official, cannot be changed from
validation scores, and the pooled Water/DMSO comparator cannot select a model.
The experimental configuration itself was not trained in this round.

## 3. Frozen contracts consumed

| Contract | Value verified at runtime |
|---|---|
| n_proteins | 4,422 (read dynamically) |
| protein_order_sha256 | `f112e7fffae4d3158ec9985a6c9cfa25e1a7dd40f847092561ea023bc1d8a45f` |
| generation_sha256 | `bb4bf51f8b82d6c19ba0ab12b6ce31b809237f08d8655e6e508fe567b1a0d145` |
| chemical shape | 57 x 2,265; Morgan 2,048 + descriptors 217 |
| genome shape | 6 x 24 |

Checkpoints also store SHA-256 values for the contract, feature arrays, schemas
and row-index files and reject mismatches on load. Output scale is recorded as
`log2`.

## 4. Test evidence

Environment: Windows; Python 3.12.13; PyTorch 2.11.0+cu128; NumPy 2.3.5;
pandas 3.0.1; CUDA available.

Commands and actual results:

```powershell
.venv\Scripts\python.exe -m unittest -v baseline.tests.test_model_v2_contract baseline.tests.test_model_v2_losses baseline.tests.test_no_label_leakage tests.test_model_v2
cd baseline
..\.venv\Scripts\python.exe -m unittest -v tests.test_person_c
cd ..
.venv\Scripts\python.exe -m baseline.training_v2 --config baseline\configs\model_v2_huber.yaml --output-dir reports\model_v2_stage2\huber_seed_20260814 --smoke
```

Specified V2 tests plus compatibility tests: 30 tests, all PASS. Existing Person
C tests: 6 tests, all PASS when run from their legacy package root. Coverage
includes output identity and similarity key; zero batch initialization and
diagnostics; real branch perturbation isolation; strict Huber/FC masks; zero
weights; runtime label-file guard; validation no-backward; train-only basis,
anchor, high-effect and CV statistics; source-manifest hashes; checkpoint exact
resume; configuration decisions; stage-B parameter freezing/update and batch
learning rate; validation batch-size invariance; all seven requested
four-scenario metrics; treatment-invariant/control-sensitive anchor fitting;
real 4,422-protein control coverage; and stage-A control-only monitor isolation.

Smoke test:

| Item | Result |
|---|---:|
| Stage A / stage B smoke samples | 751 controls / 16 treatments |
| Stage A monitor samples | 205 controls: strain-only 190, time 15 |
| Stage B monitor samples | Full scenarios: chem 1,065; strain 1,547; both 269; time 157 |
| Output proteins | 4,422 |
| Anchor source / minimum protein coverage | train controls only / 94 |
| Stage A best epoch / control-only Huber | 0 / 0.388501 |
| Stage B best epoch / four-scenario macro Huber | 1 / 0.376784 |
| Scenario Huber: chem / strain / both / time | 0.402405 / 0.354536 / 0.402108 / 0.348088 |
| delta_batch mean / std | 0.000752 / 0.001256 |
| delta_batch max absolute | 0.003261 |
| delta_batch / delta_response norm | 0.022870 |
| Checkpoint restore | PASS; best model + matching AdamW state + early-stop state |
| Test proteome opened | false |

This smoke test establishes plumbing and invariants only. Its validation loss is
not a scientific comparison and must not be reported as model performance.

## 5. Files added

- `baseline/baseline/model_v2.py`
- `baseline/baseline/training_v2.py`
- `baseline/baseline/losses_v2.py`
- `baseline/baseline/evaluation_v2.py`
- `baseline/training_v2.py` (thin public module/CLI wrapper)
- `baseline/configs/model_v2_huber.yaml`
- `baseline/configs/model_v2_experimental_fc.yaml`
- `baseline/configs/model_v2_ablation.yaml`
- `baseline/tests/test_model_v2_contract.py`
- `baseline/tests/test_model_v2_losses.py`
- `baseline/tests/test_no_label_leakage.py`
- root `model_v2/*`: compatibility re-exports, V1 audit and smoke CLI only
- `reports/V1_BASELINE_AUDIT.md`
- `reports/v1_condition_mlp_reproduction.json`
- `reports/model_v2_stage2/huber_seed_20260814/training_summary.json`
- `reports/model_v2_stage2/huber_seed_20260814/training_history.json`
- `reports/model_v2_stage2/huber_seed_20260814/stage_a_best.pt`
- `reports/model_v2_stage2/huber_seed_20260814/stage_b_best.pt`
- `reports/MODEL_V2_STAGE_0_1_REPORT.md`

No frozen artifact, generator script, test truth or V1 source file was modified.

## 6. External resources and disclosure

This stage consumes, but does not regenerate, the accepted PubChem/RDKit
chemical artifacts and Peter et al. 2018 / SGD-NCBI genome artifacts. Their
URLs, versions, dates, licenses/evidence and SHA-256 records remain in the
frozen upstream `source_manifest.json`, mapping tables and existing chemical /
genome reports. V2 checkpoints pin the derived-artifact hashes used by a run.

## 7. Failures, not-run items and Main decisions

| Item | Status | Reason / decision needed |
|---|---|---|
| Old full AIVCModel reproduction | NOT REPRODUCIBLE | Assigned legacy trees do not contain its runnable model/decoder/GNN code or matching checkpoint. |
| Old claimed ConditionMLP numbers | NOT REPRODUCIBLE | Current exact V1 architecture run produces different validation numbers; environment/checkpoint provenance is absent. |
| Official FC loss/anchor | BLOCKED | Official mapping remains unconfirmed. Experimental parity config exists but is not official and was NOT RUN. |
| Similar-drug transfer | NOT RUN | Depends on a compliant train-only FC anchor. |
| Experimental parity FC | NOT RUN | Configuration remains labeled `experimental_inferred_mapping`; CLI refuses to present it as an implemented official FC run. |
| Real V2 training | NOT RUN | Formal CLI is implemented, but explicit stage gate still requires Main approval. |
| Correct/shuffle/zero performance ablation | NOT RUN | Interface tested; scientific experiments belong to stages 2/3. |
| Multiple seeds and scenario reports | NOT RUN | Stages 2/3. |
| Public interface changes | none | V2 reads frozen interfaces without changing them. |

## 8. Recommended next step

Pause again for Main stage-1 acceptance. If stage 2 is later approved, use
`model_v2_huber.yaml` as the formal mainline. Treat the parity configuration as
a separately labeled experiment only; do not alter its mapping using validation
scores and do not use the pooled comparator for selection.
