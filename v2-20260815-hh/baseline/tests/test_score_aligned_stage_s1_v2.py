import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from baseline.baseline.losses_v2 import LossWeights, compute_losses
from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Batch, V2Config


ROOT = Path(__file__).resolve().parents[2]


def _batch(n=4):
    return V2Batch(
        morgan=torch.randn(n, 8), descriptors=torch.zeros(n, 3),
        chemical_valid_mask=torch.cat([torch.ones(n, 8), torch.zeros(n, 3)], 1).bool(),
        chemical_mapping=torch.zeros(n, dtype=torch.long),
        chemical_confidence=torch.full((n,), 3, dtype=torch.long),
        chemical_structure_valid=torch.ones(n, 1),
        genome=torch.randn(n, 5), genome_valid_mask=torch.ones(n, 5).bool(),
        genome_mapping=torch.zeros(n, dtype=torch.long),
        genome_confidence=torch.full((n,), 3, dtype=torch.long), genome_proxy=torch.zeros(n, 1),
        medium=torch.zeros(n, dtype=torch.long), condition_numeric=torch.zeros(n, 4),
        batch_categorical=torch.tensor([[1, 1, 1], [2, 2, 2], [1, 2, 1], [2, 1, 2]]),
        is_control=torch.tensor([[True], [False], [False], [False]]),
    )


def _model():
    cfg = V2Config(
        n_proteins=7, morgan_dim=8, descriptor_dim=3, genome_dim=5,
        latent_dim=6, protein_rank=3, medium_vocab_size=2,
        batch_vocab_sizes=(3, 3, 3), dropout=0.0, batch_enabled=True,
        response_gate_enabled=True, response_gate_initial=0.25,
        response_rms_cap=0.75, batch_field_dropout=0.25,
    )
    return AnchoredVirtualCellV2(cfg)


def test_score_aligned_identity_control_zero_and_response_bound():
    model = _model().eval()
    output = model(_batch())
    assert torch.equal(output["delta_response"][0], torch.zeros(7))
    assert torch.allclose(output["y_anchor"], output["y_baseline"] + output["delta_batch"])
    assert torch.allclose(output["y_pred"], output["y_anchor"] + output["delta_response"])
    assert float(output["delta_response"].square().mean(1).sqrt().max().detach()) <= 0.75 + 1e-6
    assert output["response_gate"] is not None
    assert bool(((output["response_gate"] > 0) & (output["response_gate"] < 1)).all())
    assert output["similarity_gate"] is None


def test_batch_field_dropout_drops_whole_group_to_unknown_code():
    model = _model().train()
    observed = []
    hooks = []
    for embedding in model.batch_branch.embeddings:
        hooks.append(embedding.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone())))
    with mock.patch("torch.rand", return_value=torch.zeros(4, 1)):
        model(_batch())
    for hook in hooks:
        hook.remove()
    assert len(observed) == 3
    assert all(torch.equal(value, torch.zeros_like(value)) for value in observed)


def test_score_aligned_total_loss_exact_weighted_sum_and_empty_fc_safe():
    model = _model().eval()
    output = model(_batch())
    target = torch.randn_like(output["y_pred"])
    mask = torch.ones_like(target, dtype=torch.bool)
    empty = torch.zeros_like(mask)
    weights = LossWeights(
        absolute=1.0, fc_absolute=0.05, fc=0.05, batch_reg=0.001,
        batch_center_strength=2.0, response_magnitude=0.01,
    )
    parts = compute_losses(
        output, target, mask, weights, fc_pred=output["delta_response"],
        fc_true=torch.zeros_like(target), fc_mask=empty,
    )
    expected = (
        parts["loss_absolute"] + 0.05 * parts["loss_fc_absolute"]
        + 0.05 * parts["loss_fc"] + 0.001 * parts["loss_batch_reg"]
        + 0.01 * parts["loss_response_magnitude"]
    )
    assert torch.allclose(parts["loss_total"], expected)
    assert parts["loss_fc"].requires_grad
    assert torch.isfinite(parts["loss_total"])


def test_b_c_configs_are_fair_and_explicitly_nonofficial():
    paths = [
        ROOT / "baseline/configs/model_v2_stage_s1_b_anchor_morgan_huber.yaml",
        ROOT / "baseline/configs/model_v2_stage_s1_c_anchor_morgan_fc.yaml",
    ]
    b, c = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for config in (b, c):
        assert config["planning_proxy"] is True
        assert config["official_score"] is False
        assert config["official_fc_result"] is False
        assert config["model"]["chemical_feature_components"] == "morgan_only"
        assert config["model"]["similarity_enabled"] is False
        assert config["model"]["batch_enabled"] is True
        assert config["training"]["fixed_stage_a_checkpoint"] == b["training"]["fixed_stage_a_checkpoint"]
        assert config["training"]["stage_b"]["parameter_groups"] == b["training"]["stage_b"]["parameter_groups"]
    assert b["loss"]["fc_weight"] == b["loss"]["fc_absolute_weight"] == 0
    assert c["loss"]["fc_weight"] == c["loss"]["fc_absolute_weight"] == 0.05


def test_smoke_shared_initialization_and_no_test_truth():
    roots = [
        ROOT / "reports/model_v2_stage_s1/smoke_b_anchor_morgan_huber_seed_20260814",
        ROOT / "reports/model_v2_stage_s1/smoke_c_anchor_morgan_fc_seed_20260814",
    ]
    summaries = [json.loads((path / "training_summary.json").read_text(encoding="utf-8")) for path in roots]
    assert summaries[0]["stage_b_initial_chemical_response_sha256"] == summaries[1]["stage_b_initial_chemical_response_sha256"]
    assert summaries[0]["stage_a"]["fixed_checkpoint"]["sha256"] == summaries[1]["stage_a"]["fixed_checkpoint"]["sha256"]
    assert all(summary["test_proteome_opened"] is False for summary in summaries)
    source = (ROOT / "baseline/baseline/training_v2.py").read_text(encoding="utf-8").lower()
    assert "proteome_raw_test" not in source


def test_formal_outputs_and_score_proxy_boundaries():
    output = ROOT / "reports/model_v2_stage_s1"
    comparison = json.loads((output / "score_aligned_comparison.json").read_text(encoding="utf-8"))
    assert comparison["planning_proxy"] is True
    assert comparison["official_score"] is False
    assert comparison["official_fc_result"] is False
    assert comparison["test_proteome_opened"] is False
    assert comparison["test_prediction_generated"] is False
    assert comparison["batch_holdout_sample_count"] == 957
    assert len({value["stage_b_initial_chemical_response_sha256"] for value in comparison["training"].values()}) == 1
    assert len({value["train_internal_batch_holdout"]["holdout_sample_ids_sha256"] for value in comparison["training"].values()}) == 1
    assert all(not value["eligible_for_multiseed"] for value in comparison["decisions"].values())


def test_s0_common_subset_and_robustness_modes_are_complete():
    import pandas as pd

    output = ROOT / "reports/model_v2_stage_s1"
    s0 = pd.read_csv(ROOT / "reports/competition_score_audit/fc_metrics.csv")
    current = pd.read_csv(output / "score_aligned_fc_metrics.csv")
    for scenario in ("val_chem_only", "val_strain_only", "val_both", "val_time"):
        expected = s0.loc[(s0.scenario == scenario) & (s0.subset == "common_intersection"), "n_valid_positions"].unique()
        observed = current.loc[(current.scenario == scenario) & current.model.str.startswith("S1 "), "n_valid_positions"].unique()
        assert expected.tolist() == observed.tolist()
    batch = pd.read_csv(output / "score_aligned_batch_robustness.csv")
    chemical = pd.read_csv(output / "score_aligned_chemical_robustness.csv")
    assert set(batch.batch_mode) == {"correct", "shuffled", "disabled"}
    assert set(chemical.chemical_mode) == {"correct", "shuffle", "zero"}
    assert len(batch) == 2 * 4 * 3
    assert len(chemical) == 2 * 4 * 3
    numeric = pd.concat([
        batch[["absolute_rmse", "raw_fc_global_pcc", "raw_fc_rmse"]],
        chemical[["absolute_rmse", "raw_fc_global_pcc", "raw_fc_rmse"]],
    ])
    assert bool(np.isfinite(numeric.to_numpy(float)).all())


def test_batch_holdout_groups_do_not_cross_training_partition():
    import pandas as pd
    from baseline.baseline.training_v2 import BATCH_COLUMNS, CONTROL_NAMES, load_train_val_metadata

    meta = load_train_val_metadata(ROOT)
    holdout = pd.read_csv(ROOT / "reports/model_v2_stage_s1/score_aligned_batch_holdout_ids.csv")
    holdout_ids = pd.Index(holdout.sample_ID.astype(str))
    train = meta.index[meta.split_final.eq("train")]
    names = meta.loc[train, "perturbation_no_concentration"].astype(str).str.lower()
    treatment = train[~names.isin(CONTROL_NAMES | {"quality control"})]
    remaining = treatment[~treatment.isin(holdout_ids)]
    key = lambda ids: set(meta.loc[ids, list(BATCH_COLUMNS)].astype(str).agg("|".join, axis=1))
    assert not (key(holdout_ids) & key(remaining))
    assert len(holdout_ids) == 957
