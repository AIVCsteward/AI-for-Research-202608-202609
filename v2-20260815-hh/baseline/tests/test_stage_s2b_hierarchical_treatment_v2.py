import json
from pathlib import Path

import torch

from baseline.baseline.losses_v2 import LossWeights, compute_losses
from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Batch, V2Config
from baseline.baseline.training_v2 import configure_stage_optimizer, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    ROOT / "baseline/configs/model_v2_stage_s2b_d0_hierarchical_none_huber.yaml",
    ROOT / "baseline/configs/model_v2_stage_s2b_d1_hierarchical_morgan_huber.yaml",
    ROOT / "baseline/configs/model_v2_stage_s2b_d2_hierarchical_morgan_fc.yaml",
]


def _batch(codes=((1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0))):
    n = len(codes)
    return V2Batch(
        morgan=torch.randn(n, 8), descriptors=torch.zeros(n, 3),
        chemical_valid_mask=torch.cat([torch.ones(n, 8), torch.zeros(n, 3)], 1).bool(),
        chemical_mapping=torch.zeros(n, dtype=torch.long),
        chemical_confidence=torch.ones(n, dtype=torch.long),
        chemical_structure_valid=torch.ones(n, 1),
        genome=torch.randn(n, 5), genome_valid_mask=torch.ones(n, 5).bool(),
        genome_mapping=torch.zeros(n, dtype=torch.long),
        genome_confidence=torch.ones(n, dtype=torch.long), genome_proxy=torch.zeros(n, 1),
        medium=torch.ones(n, dtype=torch.long), condition_numeric=torch.zeros(n, 4),
        batch_categorical=torch.tensor(codes, dtype=torch.long),
        is_control=torch.zeros(n, 1, dtype=torch.bool),
    )


def _model():
    config = V2Config(
        n_proteins=7, morgan_dim=8, descriptor_dim=3, genome_dim=5,
        latent_dim=6, protein_rank=3, medium_vocab_size=3,
        batch_vocab_sizes=(4, 4, 4), dropout=0.0, batch_enabled=True,
        batch_structure="hierarchical_batch", response_gate_enabled=True,
        response_gate_initial=0.25, response_rms_cap=0.75,
    )
    model = AnchoredVirtualCellV2(config)
    with torch.no_grad():
        for level in (model.batch_branch.source, model.batch_branch.instrument, model.batch_branch.plate):
            level.decode.basis.weight.normal_(0, 0.1)
    return model


def test_canonical_hierarchical_sum_and_nested_oov_fallback():
    output = _model().eval()(_batch())
    assert torch.allclose(output["delta_batch"], output["delta_source"] + output["delta_instrument"] + output["delta_plate"])
    assert torch.equal(output["delta_plate"][1], torch.zeros(7))
    assert torch.equal(output["delta_instrument"][2], torch.zeros(7))
    assert torch.equal(output["delta_plate"][2], torch.zeros(7))
    assert torch.equal(output["delta_batch"][3], torch.zeros(7))
    assert torch.allclose(output["delta_source"][0], output["delta_source"][1])
    assert torch.allclose(output["delta_instrument"][0], output["delta_instrument"][1])


def test_stage_b_optimizer_excludes_every_frozen_anchor_parameter():
    model = _model()
    config = json.loads(CONFIGS[1].read_text(encoding="utf-8"))
    optimizer = configure_stage_optimizer(model, config["training"]["stage_b"], 1e-4)
    assert {group["group_name"] for group in optimizer.param_groups} == {"chemical_encoder", "response_branch"}
    frozen_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
        if name.startswith(("baseline_branch.", "genome_encoder.", "batch_branch."))
    }
    response_before = {name: parameter.detach().clone() for name, parameter in model.response_branch.named_parameters()}
    output = model(_batch())
    output["y_pred"].square().mean().backward()
    optimizer.step()
    assert all(torch.equal(parameter.detach(), frozen_before[name]) for name, parameter in model.named_parameters() if name in frozen_before)
    assert any(not torch.equal(parameter.detach(), response_before[name]) for name, parameter in model.response_branch.named_parameters())


def test_hierarchical_regularized_total_loss_is_exact_weighted_sum():
    output = _model().eval()(_batch())
    target = torch.randn_like(output["y_pred"])
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = LossWeights(
        absolute=1.0, response_magnitude=0.01,
        batch_source_reg=0.0001, batch_instrument_reg=0.0005, batch_plate_reg=0.002,
    )
    losses = compute_losses(output, target, mask, weights)
    expected = (
        losses["loss_absolute"] + 0.01 * losses["loss_response_magnitude"]
        + 0.0001 * losses["loss_batch_source_reg"]
        + 0.0005 * losses["loss_batch_instrument_reg"]
        + 0.002 * losses["loss_batch_plate_reg"]
    )
    assert torch.allclose(losses["loss_total"], expected)


def test_d0_d1_d2_configs_are_fair_and_use_exact_s2a_checkpoint():
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in CONFIGS]
    checkpoint_specs = [config["training"]["fixed_stage_a_checkpoint"] for config in configs]
    assert len({json.dumps(spec, sort_keys=True) for spec in checkpoint_specs}) == 1
    checkpoint = Path(checkpoint_specs[0]["path"])
    assert sha256_file(checkpoint) == checkpoint_specs[0]["sha256"]
    common_keys = ("seed", "batch_size", "weight_decay", "device", "num_workers", "train_internal_batch_holdout", "stage_a")
    for key in common_keys:
        assert configs[0]["training"][key] == configs[1]["training"][key] == configs[2]["training"][key]
    for config in configs:
        assert config["planning_proxy"] is True and config["official_score"] is False
        assert config["official_fc_result"] is False
        assert config["model"]["batch_structure"] == "hierarchical_batch"
        assert config["model"]["genome_mode"] == "correct"
        assert config["model"]["similarity_enabled"] is False
        groups = config["training"]["stage_b"]["parameter_groups"]
        assert {name for name, spec in groups.items() if spec["trainable"]} == {"chemical_encoder", "response_branch"}
    assert [config["model"]["chemical_feature_components"] for config in configs] == ["none", "morgan_only", "morgan_only"]
    assert configs[0]["loss"]["fc_weight"] == configs[1]["loss"]["fc_weight"] == 0
    assert configs[2]["loss"]["fc_weight"] == configs[2]["loss"]["fc_absolute_weight"] == 0.05


def test_s2b_smokes_share_initialization_and_pass_freeze_audit():
    roots = [ROOT / f"reports/model_v2_stage_s2b/smoke_d{index}" for index in range(3)]
    summaries = [json.loads((root / "training_summary.json").read_text(encoding="utf-8")) for root in roots]
    assert len({summary["stage_b_initial_chemical_response_sha256"] for summary in summaries}) == 1
    assert len({summary["stage_a"]["fixed_checkpoint"]["sha256"] for summary in summaries}) == 1
    for summary in summaries:
        assert summary["stage_a"]["actual_epochs"] == 0
        assert summary["stage_b_freeze_audit"]["status"] == "PASS"
        assert all(summary["stage_b_freeze_audit"]["parameter_hashes_exactly_equal"].values())
        assert all(summary["stage_b_freeze_audit"]["probe_outputs_bitwise_equal"].values())
        assert summary["test_proteome_opened"] is False
    source = (ROOT / "baseline/baseline/training_v2.py").read_text(encoding="utf-8").lower()
    assert "proteome_raw_test" not in source


def test_formal_s2b_outputs_are_complete_fair_finite_and_paused():
    output = ROOT / "reports/model_v2_stage_s2b"
    comparison = json.loads((output / "hierarchical_treatment_comparison.json").read_text(encoding="utf-8"))
    assert comparison["planning_proxy"] is True
    assert comparison["official_score"] is False
    assert comparison["official_fc_result"] is False
    assert comparison["test_proteome_opened"] is False
    assert comparison["test_prediction_generated"] is False
    assert comparison["batch_holdout_sample_count"] == 957
    assert comparison["oov_max_semantic_error"] == 0.0
    assert all(not decision["eligible_for_multiseed"] for decision in comparison["decisions"].values())
    assert all(not decision["morgan_contribution_established"] for decision in comparison["chemical_decision"].values())
    assert len({summary["stage_b_initial_chemical_response_sha256"] for summary in comparison["training"].values()}) == 1
    assert len({summary["train_internal_batch_holdout"]["holdout_sample_ids_sha256"] for summary in comparison["training"].values()}) == 1
    for model_name, summary in comparison["training"].items():
        assert summary["n_treatment"] == 4121 and summary["n_treatment_full"] == 5078
        assert summary["train_internal_batch_holdout"]["holdout_sample_count"] == 957
        assert summary["stage_a"]["actual_epochs"] == 0
        assert summary["stage_b_freeze_audit"]["status"] == "PASS"
        checkpoint = torch.load(comparison["training_artifacts"][model_name]["checkpoint"], map_location="cpu", weights_only=False)
        assert checkpoint["batch_structure"] == "hierarchical_batch"
        assert checkpoint["stage_b_freeze_audit"]["status"] == "PASS"
    import pandas as pd
    absolute = pd.read_csv(output / "hierarchical_treatment_absolute_metrics.csv")
    rows = absolute.loc[absolute.model.str.startswith("S2B D")]
    required = ["log2_rmse", "mae", "global_r2", "sample_pcc_median", "sample_r2_median", "protein_pcc_median", "protein_r2_median"]
    assert len(rows) == 12
    assert bool(torch.isfinite(torch.as_tensor(rows[required].to_numpy(float))).all())
    assert (ROOT / "reports/MODEL_V2_STAGE_S2B_HIERARCHICAL_TREATMENT_REPORT.md").is_file()
