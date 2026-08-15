import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline.hierarchical_batch_v2 import (
    StageABatchCalibrationModel, deterministic_group_folds, metadata_overlap,
)
from baseline.baseline.model_v2 import V2Batch, V2Config
from baseline.baseline.training_v2 import CONTROL_NAMES, load_train_val_metadata


ROOT = Path(__file__).resolve().parents[2]


def _batch(codes):
    n = len(codes)
    return V2Batch(
        morgan=torch.zeros(n, 2), descriptors=torch.zeros(n, 1),
        chemical_valid_mask=torch.zeros(n, 3, dtype=torch.bool),
        chemical_mapping=torch.zeros(n, dtype=torch.long), chemical_confidence=torch.zeros(n, dtype=torch.long),
        chemical_structure_valid=torch.zeros(n, 1), genome=torch.randn(n, 3),
        genome_valid_mask=torch.ones(n, 3, dtype=torch.bool), genome_mapping=torch.zeros(n, dtype=torch.long),
        genome_confidence=torch.zeros(n, dtype=torch.long), genome_proxy=torch.zeros(n, 1),
        medium=torch.ones(n, dtype=torch.long), condition_numeric=torch.zeros(n, 4),
        batch_categorical=torch.tensor(codes, dtype=torch.long), is_control=torch.ones(n, 1, dtype=torch.bool),
    )


def _hierarchical_model():
    cfg = V2Config(
        n_proteins=5, morgan_dim=2, descriptor_dim=1, genome_dim=3,
        latent_dim=4, protein_rank=2, medium_vocab_size=3,
        batch_vocab_sizes=(4, 4, 5), dropout=0.0,
    )
    model = StageABatchCalibrationModel(cfg, "hierarchical_batch", torch.zeros(5)).eval()
    with torch.no_grad():
        for level in (model.hierarchical_batch.source, model.hierarchical_batch.instrument, model.hierarchical_batch.plate):
            level.decode.basis.weight.normal_(0, 0.1)
    return model


def test_hierarchical_oov_fallback_is_exact_and_partial():
    model = _hierarchical_model()
    full = model(_batch([[1, 1, 1]]))
    plate = model(_batch([[1, 1, 0]]))
    instrument = model(_batch([[1, 0, 0]]))
    all_oov = model(_batch([[0, 0, 0]]))
    assert torch.allclose(full["delta_source"], plate["delta_source"])
    assert torch.allclose(full["delta_instrument"], plate["delta_instrument"])
    assert torch.equal(plate["delta_plate"], torch.zeros_like(plate["delta_plate"]))
    assert torch.allclose(full["delta_source"], instrument["delta_source"])
    assert torch.equal(instrument["delta_instrument"], torch.zeros_like(instrument["delta_instrument"]))
    assert torch.equal(instrument["delta_plate"], torch.zeros_like(instrument["delta_plate"]))
    assert torch.equal(all_oov["delta_batch"], torch.zeros_like(all_oov["delta_batch"]))


def test_hierarchical_sum_and_no_batch_identity():
    hierarchical = _hierarchical_model()
    output = hierarchical(_batch([[1, 2, 3], [2, 1, 4]]))
    assert torch.allclose(output["delta_batch"], output["delta_source"] + output["delta_instrument"] + output["delta_plate"])
    no_batch = StageABatchCalibrationModel(hierarchical.cfg, "no_batch", torch.zeros(5)).eval()
    base = _batch([[1, 1, 1]])
    left = no_batch(base)
    right = no_batch(base.replace(batch_categorical=torch.tensor([[3, 3, 4]])))
    assert torch.equal(left["delta_batch"], torch.zeros_like(left["delta_batch"]))
    assert torch.allclose(left["y_pred"], right["y_pred"])
    assert not any("chemical" in name or "response" in name for name, _ in hierarchical.named_parameters())


def test_real_plate_and_instrument_folds_are_group_disjoint_and_reproducible():
    meta = load_train_val_metadata(ROOT)
    train = meta.index[meta.split_final.eq("train")]
    names = meta.loc[train, "perturbation_no_concentration"].astype(str).str.lower()
    controls = train[names.isin(CONTROL_NAMES)]
    for column in ("Yeast_cell_plate", "instrument"):
        first = deterministic_group_folds(meta, controls, column, 5, 20260814)
        second = deterministic_group_folds(meta, controls, column, 5, 20260814)
        assert [[*fold["holdout_groups"]] for fold in first] == [[*fold["holdout_groups"]] for fold in second]
        assert sum(len(fold["holdout_ids"]) for fold in first) == 751
        for fold in first:
            assert not (set(meta.loc[fold["train_ids"], column].astype(str)) & set(meta.loc[fold["holdout_ids"], column].astype(str)))
            assert meta.loc[fold["holdout_ids"], "split_final"].eq("train").all()


def test_public_metadata_has_no_new_technical_category_or_tuple(tmp_path):
    output = tmp_path / "overlap.json"
    result = metadata_overlap(ROOT, output)
    assert result["test_proteome_opened"] is False
    assert result["splits"]["train"]["unique_tuple_count"] == 144
    for split in ("validation", "test"):
        assert result["splits"][split]["unseen_tuple_count_vs_train"] == 0
        assert all(not values for values in result["splits"][split]["unseen_vs_train"].values())


def test_config_declares_ordered_regularization_and_stage_a_only():
    config = json.loads((ROOT / "baseline/configs/model_v2_stage_s2a_hierarchical_batch.yaml").read_text(encoding="utf-8"))
    loss = config["loss"]
    assert loss["hierarchical_source_reg_weight"] < loss["hierarchical_instrument_reg_weight"] < loss["hierarchical_plate_reg_weight"]
    assert config["test_proteome_forbidden"] is True
    source = (ROOT / "baseline/baseline/hierarchical_batch_v2.py").read_text(encoding="utf-8").lower()
    assert "proteome_raw_test" not in source


def test_formal_cv_artifacts_are_complete_finite_and_leakage_free():
    output = ROOT / "reports/model_v2_stage_s2a"
    for name in ("plate", "instrument"):
        frame = pd.read_csv(output / f"{name}_group_cv_metrics.csv")
        assert len(frame) == 15
        assert set(frame.structure) == {"no_batch", "flat_batch", "hierarchical_batch"}
        assert set(frame.fold) == set(range(5))
        assert frame.group_overlap_count.eq(0).all()
        assert (~frame.holdout_labels_used_for_training_or_selection.astype(bool)).all()
        numeric = frame[["rmse", "mae", "global_r2", "sample_pcc_median", "protein_pcc_median"]].to_numpy(float)
        assert np.isfinite(numeric).all()
        for fold in range(5):
            rows = frame.loc[frame.fold.eq(fold)].set_index("structure")
            assert rows.loc["hierarchical_batch", "rmse"] < rows.loc["flat_batch", "rmse"]
    assert len(list(output.glob("*_cv/fold_*/*/stage_a_best.pt"))) == 30
    assert len(list(output.glob("*_cv/fold_*/*/history.json"))) == 30


def test_formal_comparison_passes_declared_gates_but_does_not_start_s2b():
    output = ROOT / "reports/model_v2_stage_s2a"
    comparison = json.loads((output / "hierarchical_batch_comparison.json").read_text(encoding="utf-8"))
    assert comparison["eligible_for_stage_s2b"] is True
    assert all(comparison["gates"].values())
    assert comparison["stage_b_trained"] is False
    assert comparison["chemical_response_trained"] is False
    assert comparison["test_proteome_opened"] is False
    assert comparison["test_prediction_generated"] is False
    assert comparison["checkpoint_count"] == 33
    assert comparison["validation_hierarchical_to_flat_rmse_ratio"] <= 1.03
    assert comparison["plate_energy_fraction_of_layer_energy"] <= 0.5
    assert all(float(value) == 0.0 for value in comparison["oov_checks"].values())
    assert (ROOT / "reports/MODEL_V2_STAGE_S2A_HIERARCHICAL_BATCH_REPORT.md").is_file()
