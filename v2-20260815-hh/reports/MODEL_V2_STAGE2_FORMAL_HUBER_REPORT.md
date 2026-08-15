# Model V2 Stage 2 — Formal Huber Mainline, Seed 20260814

Date: 2026-08-14  
Task ID: model-v2  
Working branch: not applicable, local mode  
Commit: none  
Status: single authorized formal run completed; paused before any ablation

## 1. Authorized run and safeguards

Executed exactly one formal configuration:

```powershell
D:\虚拟细胞\.venv\Scripts\python.exe -m baseline.training_v2 --config D:\虚拟细胞\baseline\configs\model_v2_huber.yaml --output-dir D:\虚拟细胞\reports\model_v2_stage2\formal_huber_seed_20260814
```

Resolved semantics: Huber-only, chemical=`correct`, genome=`correct`,
batch=`enabled`, FC weight=0, similarity=false, seed=20260814. Experimental
parity FC, similarity-FC, shuffle/zero inputs, batch ablations, other seeds and
test predictions were not run. The training path reports
`test_proteome_opened=false`.

## 2. Runtime and data coverage

| Item | Actual value |
|---|---:|
| Device | NVIDIA GeForce RTX 5060 Laptop GPU / CUDA |
| Measured pipeline time | 56.93 seconds |
| Peak PyTorch GPU memory allocated | 44.55 MiB |
| Peak PyTorch GPU memory reserved | 60.00 MiB |
| Stage A train controls | 751 |
| Anchor source | train controls only |
| Minimum control observations per protein | 94 |
| Stage A validation controls | 205: strain-only 190, time 15 |
| Stage B train treatments | 5,078, excluding controls and QC |
| Stage B validation | Full chem-only 1,065; strain-only 1,547; both 269; time 157 |

The time measurement covers input loading, both stages, best-checkpoint
restoration, validation evaluation and artifact writing. GPU peaks are PyTorch
allocator measurements, not whole-system GPU telemetry.

## 3. Training and early stopping

| Stage | Monitor | Actual epochs | Best epoch | Best monitor | Stop reason |
|---|---|---:|---:|---:|---|
| A | control-only validation Huber | 30 (epochs 0–29) | 28 | 0.236440 | Configured 30 epochs completed; patience not exhausted |
| B | equal-weight four-scenario macro Huber | 13 (epochs 0–12) | 0 | 0.207567 | 12 consecutive non-improvements exhausted patience |

Stage B training loss continued from 0.160396 at epoch 0 to 0.103732 at epoch
12, while its validation macro never improved on epoch 0. The restored final
model is therefore the epoch-0 Stage B checkpoint, not the last epoch.

Every epoch's train loss, monitor, monitor details, `bad_epochs` and named
parameter-group learning rates are persisted in `training_history.json`.

## 4. Validation metrics from restored best Stage B checkpoint

All metrics use common valid positions. Per-sample and per-protein medians skip
items that fail minimum-count or variance requirements.

| Scenario | RMSE | MAE | Global R² | Sample PCC median | Sample R² median | Protein PCC median | Protein R² median |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_chem_only | 0.639219 | 0.462048 | 0.947382 | 0.979519 | 0.955269 | 0.819389 | 0.554388 |
| val_strain_only | 0.750377 | 0.521934 | 0.926326 | 0.965290 | 0.928382 | 0.774717 | 0.366407 |
| val_both | 0.836965 | 0.598189 | 0.909884 | 0.962309 | 0.912704 | 0.813124 | 0.361272 |
| val_time | 0.578361 | 0.393263 | 0.956639 | 0.981483 | 0.960035 | 0.767145 | 0.563493 |

Exact valid-position scenario Huber values were 0.179061, 0.227816, 0.281376
and 0.142016 respectively; their equal-weight macro is 0.207567.

## 5. Comparison with frozen V1 ConditionMLP

Only metrics available in the frozen V1 artifact are compared. Negative RMSE
delta and positive R² delta favor V2.

| Scenario | V1 RMSE | V2 RMSE | Δ RMSE | V1 Global R² | V2 Global R² | Δ R² | V1 protein R² median | V2 protein R² median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val_chem_only | 0.689792 | 0.639219 | -0.050573 | 0.938727 | 0.947382 | +0.008655 | 0.462774 | 0.554388 |
| val_strain_only | 0.871096 | 0.750377 | -0.120719 | 0.900714 | 0.926326 | +0.025612 | 0.182915 | 0.366407 |
| val_both | 0.955083 | 0.836965 | -0.118118 | 0.882654 | 0.909884 | +0.027230 | 0.203320 | 0.361272 |
| val_time | 0.518549 | 0.578361 | +0.059813 | 0.965144 | 0.956639 | -0.008505 | 0.567851 | 0.563493 |

V2 improves all three shared measures in chem-only, strain-only and both, but
regresses on all three in the time scenario. This is a single-seed result and
does not establish stable improvement.

## 6. Batch branch diagnostics

Diagnostics were recomputed after training from the restored best Stage B
checkpoint over all 5,078 train treatments; this recomputation did not update
the model.

| Diagnostic | Value |
|---|---:|
| delta_batch mean | 0.149180 |
| delta_batch standard deviation | 0.596358 |
| delta_batch maximum absolute value | 1.769542 |
| delta_batch / delta_response norm | 2.711795 |

The batch-to-response norm is abnormally high and indicates that the batch
branch may be absorbing substantial signal. Per instruction, no batch weight,
architecture or training setting was changed and the run was not repeated.
The likely contributors to investigate are the broad batch correction learned
during control-first Stage A and the fact that Stage B retained the batch branch
while its best checkpoint occurred after only one treatment epoch. This is a
hypothesis, not a confirmed causal attribution.

## 7. Artifacts

- Checkpoints: `stage_a_best.pt`, `stage_b_best.pt`
- Summary: `training_summary.json`
- Per-epoch history: `training_history.json`
- Complete resolved configuration: `resolved_config.json`
- Source configuration: `D:\虚拟细胞\baseline\configs\model_v2_huber.yaml`
- Output directory: `D:\虚拟细胞\reports\model_v2_stage2\formal_huber_seed_20260814`

Both checkpoints contain artifact hashes and the control-only anchor metadata.

## 8. Exceptions and gate status

No OOM, NaN, non-finite monitor, checkpoint restore failure or abnormal early
stopping implementation event occurred. The scientific warning is the time
scenario regression and high batch/response norm. No test protein truth was
opened. No ablation or additional seed was started. Work is paused for Main to
decide whether to correct the mainline, run another baseline, or authorize a
subsequent ablation.
