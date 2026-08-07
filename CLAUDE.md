# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Virtual Cell (AIVC)** — predict proteome-wide protein expression changes under experimental perturbations (different yeast strains, chemical treatments, media, temperatures, and time points). This is a regression problem: given condition metadata, output log2 protein intensities for ~4,232 proteins.

The design document is at [rules/VirtualCellProblemSolvingApproach.pdf](rules/VirtualCellProblemSolvingApproach.pdf). It describes the problem formulation, baseline, and improvement roadmap (in Chinese).

## Key Project Details (from the design doc)

- **Output**: 4,232 proteins (log2 intensity scale), submitted as `prediction.csv` with `prediction_scale=log2`
- **Training samples**: ~5,243 (after filtering proteins with >80% missing rate on log2 scale)
- **Data splits**: `split_final` column — `train`, `val_chem_only`, `val_strain_only`, `val_both`, `test`
- **Condition features**: strain, chemical (drug/DMSO/Water), medium, temperature, time
- **Primary metric**: Global R² (mean baseline achieves ~0.87; matched control baseline achieves ~0.98)
- **Critical nuance**: Per-protein R² on treatment-vs-control fold change is the real signal — naive mean prediction scores R² ≈ -0.06 per-protein vs 0.72–0.84 for control samples. A model that predicts the mean everywhere scores well on Global R² but fails on fold-change evaluation.
- **Missing data**: proteomics has substantial NA values. Use a binary mask (`~proteome.isna()`) and mask-aware MSE loss — fill NA with 0, multiply squared error by mask, divide by mask sum.
- **Matched control**: DMSO/Water solvent controls serve as matched-control baselines. Treatment effect = `y_treat - y_control`.

## Baseline Architecture

```
Condition (strain/chem/medium/temp/time) → one-hot encoding → MLP (2-3 layers, 256-512 hidden, ReLU, Dropout 0.1) → 4,232 protein log2 predictions
```

Key baseline techniques from the doc:
- Hash encoding as an alternative to one-hot for high-cardinality categoricals
- Cyclic (sin/cos) time encoding for time points
- Target encoding: prior mean per strain, prior delta per chemical — concatenated as features
- Mask-aware MSE loss
- K-fold by `(strain, chemical)` group for robust validation

## Improvement Roadmap (ordered by priority from doc)

1. **Feature engineering** (high priority):
   - Protein embeddings via ESM (ESM-2) from SGD sequences → `(4232, 480)` decoder input
   - Chemical features via RDKit: Morgan fingerprints (2048-bit), MACCS keys, molecular descriptors (MolWt, MolLogP, TPSA, etc.)
   - Strain embeddings via phylogenetic distance MDS from the 1,011 yeast genomes project

2. **Architecture upgrades**:
   - Encoder + Decoder with protein embeddings as decoder queries
   - Transformer with self-attention over conditions
   - Cross-attention between chemical/strain embeddings and protein embeddings

3. **Loss function**:
   - Primary: mask-aware MSE
   - Add fold change Pearson correlation loss: `1 - corr(fc_pred, fc_true)`
   - Add protein-protein correlation consistency loss
   - L2 regularization on embeddings

4. **Calibration**: Subtract control mean bias, add global mean (`pred_calibrated = pred - control_mean + global_mean`)

5. **Advanced** (stretch): Conditional VAE, Flow Matching, Diffusion for generative protein expression prediction
## Person A Feature Pipeline (Stage 2-3)

The current input-side implementation is split across:

- `baseline/entity_representations.py`: train-only strain prior, chemical anchor, deterministic hash, and cross-feature encoders.
- `baseline/features.py`: fits all encoders and projects the concatenated raw features to a fixed 256-dimensional embedding.
- `baseline/config.py`: shared encoder/decoder/GNN/loss/training configuration contract.
- `experiments/ablation_encoder.py`: prepares full/no-strain-prior/no-chem-anchor/no-hash/no-cross-feature inputs for ablation experiments.

Fit representations only on `meta["split_final"] == "train"`:

```python
from baseline.features import fit_feature_encoders, build_condition_features

train_mask = meta["split_final"].eq("train")
encoders = fit_feature_encoders(
    meta.loc[train_mask],
    y_log2.loc[train_mask],
    mask_matrix.loc[train_mask],
)
X_all = build_condition_features(meta, encoders=encoders)  # (N, 256)
```

Unseen categorical values are encoded as all-zero one-hot rows, while strain and chemical representations use train-derived fallback vectors; hash features remain available for unseen entities.
