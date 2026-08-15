from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from baseline.baseline.evaluation_v2 import VAL_SCENARIOS, batch_diagnostics, evaluate_four_scenarios
from baseline.baseline.losses_v2 import LossWeights, compute_losses
from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Batch, V2Config
from baseline.baseline.training_v2 import (
    checkpoint_payload, configure_stage_optimizer, load_artifact_bundle,
    control_validation_huber, fit_control_protein_anchor, load_config,
    load_label_frames, load_train_val_metadata, make_loader, restore_checkpoint,
    make_chemical_feature_variant, save_checkpoint, sha256_file,
    stage_a_validation_control_ids, validation_macro_huber,
    _loss_weights,
)


def synthetic_batch(n=4):
    return V2Batch(
        morgan=torch.randn(n, 4), descriptors=torch.randn(n, 3),
        chemical_valid_mask=torch.ones(n, 7, dtype=torch.bool),
        chemical_mapping=torch.zeros(n, dtype=torch.long),
        chemical_confidence=torch.full((n,), 3, dtype=torch.long),
        chemical_structure_valid=torch.ones(n, 1),
        genome=torch.randn(n, 2), genome_valid_mask=torch.ones(n, 2, dtype=torch.bool),
        genome_mapping=torch.ones(n, dtype=torch.long),
        genome_confidence=torch.full((n,), 3, dtype=torch.long),
        genome_proxy=torch.zeros(n, 1), medium=torch.ones(n, dtype=torch.long),
        condition_numeric=torch.randn(n, 4),
        batch_categorical=torch.ones(n, 3, dtype=torch.long),
        is_control=torch.zeros(n, 1, dtype=torch.bool),
    )


def make_model(batch_enabled=True):
    cfg = V2Config(
        n_proteins=6, morgan_dim=4, descriptor_dim=3, genome_dim=2,
        latent_dim=8, protein_rank=3, medium_vocab_size=4,
        batch_vocab_sizes=(4, 4, 4), dropout=0.0,
        batch_enabled=batch_enabled,
    )
    model = AnchoredVirtualCellV2(cfg)
    model.eval()
    return model


class ModelV2ContractTests(unittest.TestCase):
    def test_required_configs_encode_main_decision(self):
        root = Path(__file__).resolve().parents[2]
        huber = json.loads((root / "baseline/configs/model_v2_huber.yaml").read_text(encoding="utf-8"))
        experimental = json.loads((root / "baseline/configs/model_v2_experimental_fc.yaml").read_text(encoding="utf-8"))
        ablation = json.loads((root / "baseline/configs/model_v2_ablation.yaml").read_text(encoding="utf-8"))
        self.assertEqual(huber["loss"]["fc_weight"], 0.0)
        self.assertFalse(huber["model"]["similarity_enabled"])
        self.assertEqual(experimental["control_mapping"]["name"], "pert_id_parity_v1")
        self.assertEqual(experimental["control_mapping"]["status"], "experimental_inferred_mapping")
        self.assertFalse(experimental["control_mapping"]["official"])
        self.assertTrue(experimental["control_mapping"]["mapping_mutation_from_validation_forbidden"])
        self.assertFalse(experimental["control_mapping"]["pooled_water_dmso_comparator_for_model_selection"])
        self.assertFalse(ablation["training"]["full_training_authorized"])
        required_training = {
            "seed", "batch_size", "weight_decay", "device", "num_workers",
            "stage_a", "stage_b",
        }
        self.assertTrue(required_training.issubset(huber["training"]))
        for stage in ("stage_a", "stage_b"):
            self.assertEqual(
                set(huber["training"][stage]["parameter_groups"]),
                {"baseline_branch", "genome_encoder", "chemical_encoder", "response_branch", "batch_branch"},
            )
        self.assertEqual(huber["training"]["stage_a"]["early_stopping_monitor"], "control_only_validation_huber")
        self.assertEqual(huber["training"]["stage_b"]["early_stopping_monitor"], "macro_huber_equal_four_scenarios")
        self.assertEqual(
            huber["loss"]["batch_center_effective_coefficient"],
            huber["loss"]["batch_reg_weight"] * huber["loss"]["batch_center_strength"],
        )
    def test_forward_contract_similarity_none_and_batch_zero_init(self):
        model, batch = make_model(), synthetic_batch()
        with torch.no_grad():
            output = model(batch)
        self.assertIsNone(output["similarity_gate"])
        self.assertTrue(torch.allclose(output["y_pred"], output["y_baseline"] + output["delta_response"] + output["delta_batch"], atol=1e-6))
        self.assertEqual(float(output["delta_batch"].abs().max()), 0.0)
        diagnostics = batch_diagnostics(output)
        self.assertEqual(diagnostics["delta_batch_mean"], 0.0)
        self.assertEqual(diagnostics["delta_batch_std"], 0.0)
        self.assertEqual(diagnostics["delta_batch_max_abs"], 0.0)

    def test_branch_isolation_by_perturbing_each_input_family(self):
        model, batch = make_model(), synthetic_batch()
        # Make batch output nonzero so invariance is tested beyond zero-init equality.
        torch.nn.init.normal_(model.batch_branch.decode.basis.weight, std=0.02)
        model.eval()
        with torch.no_grad():
            reference = model(batch)
            chemical = model(batch.replace(morgan=batch.morgan + 3, descriptors=batch.descriptors - 2))
            genome = model(batch.replace(genome=batch.genome + 4))
            culture = model(batch.replace(
                medium=(batch.medium + 1) % 4,
                condition_numeric=batch.condition_numeric + 1.5,
            ))
            batch_changed = model(batch.replace(batch_categorical=(batch.batch_categorical + 1) % 4))
        self.assertTrue(torch.equal(reference["y_baseline"], chemical["y_baseline"]))
        self.assertTrue(torch.equal(reference["delta_batch"], chemical["delta_batch"]))
        self.assertTrue(torch.equal(reference["delta_batch"], genome["delta_batch"]))
        self.assertTrue(torch.equal(reference["delta_batch"], culture["delta_batch"]))
        self.assertTrue(torch.equal(reference["y_baseline"], batch_changed["y_baseline"]))
        self.assertTrue(torch.equal(reference["delta_response"], batch_changed["delta_response"]))

    def test_control_response_is_exactly_zero(self):
        model, batch = make_model(), synthetic_batch()
        batch = batch.replace(is_control=torch.ones_like(batch.is_control))
        with torch.no_grad():
            self.assertEqual(float(model(batch)["delta_response"].abs().max()), 0.0)

    def test_four_scenario_evaluator_reports_each_split(self):
        model, batch = make_model(), synthetic_batch(2)
        target = torch.zeros(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        loaders = {name: [(batch, target, mask)] for name in VAL_SCENARIOS}
        result = evaluate_four_scenarios(model, loaders)
        self.assertEqual(set(result), set(VAL_SCENARIOS))
        expected = {
            "rmse", "mae", "global_r2", "median_per_sample_pcc",
            "median_per_sample_r2", "median_per_protein_pcc",
            "median_per_protein_r2", "n_valid_positions",
        }
        self.assertTrue(all(set(metrics) == expected for metrics in result.values()))

    def test_stage_b_freezes_baseline_updates_response_and_uses_batch_lr(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "baseline/configs/model_v2_huber.yaml")
        stage_b = config["training"]["stage_b"]
        model, batch = make_model(), synthetic_batch(4)
        model.train()
        optimizer = configure_stage_optimizer(model, stage_b, config["training"]["weight_decay"])
        baseline_before = [parameter.detach().clone() for parameter in model.baseline_branch.parameters()]
        response_before = [parameter.detach().clone() for parameter in model.response_branch.parameters()]
        target = torch.randn(4, 6)
        loss = compute_losses(model(batch), target, torch.ones_like(target, dtype=torch.bool), LossWeights(batch_reg=0.0))["loss_total"]
        loss.backward(); optimizer.step()
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(baseline_before, model.baseline_branch.parameters())))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(response_before, model.response_branch.parameters())))
        batch_group = next(group for group in optimizer.param_groups if group["group_name"] == "batch_branch")
        self.assertEqual(batch_group["lr"], stage_b["parameter_groups"]["batch_branch"]["lr"])

    def test_no_batch_mode_is_structural_not_zero_learning_rate(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "baseline/configs/model_v2_huber_no_batch.yaml")
        self.assertFalse(config["model"]["batch_enabled"])
        # The source LR remains unchanged; structural disablement must override it.
        self.assertEqual(config["training"]["stage_b"]["parameter_groups"]["batch_branch"]["lr"], 0.0001)
        model, batch = make_model(batch_enabled=False), synthetic_batch(4)
        torch.nn.init.normal_(model.batch_branch.decode.basis.weight, std=1.0)
        model.eval()
        with torch.no_grad():
            original = model(batch)
            changed = model(batch.replace(batch_categorical=(batch.batch_categorical + 2) % 4))
        self.assertTrue(torch.equal(original["delta_batch"], torch.zeros_like(original["delta_batch"])))
        self.assertTrue(torch.equal(original["y_pred"], original["y_baseline"] + original["delta_response"]))
        self.assertTrue(torch.equal(original["y_pred"], changed["y_pred"]))

        optimizer = configure_stage_optimizer(
            model, config["training"]["stage_b"], config["training"]["weight_decay"],
        )
        self.assertNotIn("batch_branch", {group["group_name"] for group in optimizer.param_groups})
        self.assertTrue(all(not parameter.requires_grad for parameter in model.batch_branch.parameters()))

        weights = _loss_weights(config)
        self.assertEqual(weights.batch_reg, 0.0)
        target = torch.randn_like(original["y_pred"])
        parts = compute_losses(original, target, torch.ones_like(target, dtype=torch.bool), weights)
        self.assertTrue(torch.equal(parts["loss_total"], parts["loss_absolute"]))

        payload = checkpoint_payload(
            model, optimizer, 0, 1.0, {"best_epoch": 0}, config,
            {"x": "hash"}, 20260814, "B",
        )
        self.assertFalse(payload["config"]["model"]["batch_enabled"])

    def test_frozen_chemical_variant_interface_entity_permutation_and_zero(self):
        artifacts = load_artifact_bundle()
        frozen_interface = __import__(
            "scripts.build_chemical_features", fromlist=["apply_feature_variant"]
        ).apply_feature_variant
        with patch("baseline.baseline.training_v2.apply_feature_variant", wraps=frozen_interface) as called:
            shuffled_a = make_chemical_feature_variant(artifacts, "shuffle", 20260814)
        self.assertEqual(called.call_count, 1)
        shuffled_b = make_chemical_feature_variant(artifacts, "shuffle", 20260814)
        self.assertTrue(torch.equal(torch.from_numpy(shuffled_a[2]), torch.from_numpy(shuffled_b[2])))
        values, masks, order, table, audit = shuffled_a
        self.assertTrue((values == artifacts.chemical_features[order]).all())
        self.assertTrue((masks == artifacts.chemical_valid_mask[order]).all())
        self.assertEqual(len(table), len(artifacts.chemical_index))
        self.assertEqual(audit["n_entities"], len(order))
        mapping = artifacts.chemical_mapping
        ineligible = ~(
            mapping["mapping_status"].isin(["confirmed", "proxy"])
            & mapping["structure_valid"].astype(bool)
            & mapping["special_control_type"].fillna("").eq("")
        ).to_numpy()
        self.assertTrue((order[ineligible] == torch.arange(len(order)).numpy()[ineligible]).all())

        zero_values, zero_masks, zero_order, _, _ = make_chemical_feature_variant(artifacts, "zero", 20260814)
        self.assertFalse(zero_values.any())
        self.assertFalse(zero_masks.any())
        self.assertTrue((zero_order == torch.arange(len(zero_order)).numpy()).all())

    def test_shuffle_is_entity_level_and_zero_preserves_identity_quality_flags(self):
        artifacts = load_artifact_bundle()
        meta = load_train_val_metadata()
        train_ids = meta.index[meta["split_final"].eq("train")]
        vocab = __import__("baseline.baseline.training_v2", fromlist=["fit_category_vocabulary"]).fit_category_vocabulary(meta, train_ids)
        names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str)
        repeated_name = names.value_counts()[lambda x: x >= 2].index[0]
        repeated_ids = names.index[names.eq(repeated_name)][:2]
        shuffled = __import__("baseline.baseline.training_v2", fromlist=["build_batch"]).build_batch(
            meta, repeated_ids, artifacts, vocab, chemical_mode="shuffle", seed=20260814,
        )
        self.assertTrue(torch.equal(shuffled.morgan[0], shuffled.morgan[1]))
        self.assertTrue(torch.equal(shuffled.descriptors[0], shuffled.descriptors[1]))
        self.assertTrue(torch.equal(shuffled.chemical_valid_mask[0], shuffled.chemical_valid_mask[1]))
        correct = __import__("baseline.baseline.training_v2", fromlist=["build_batch"]).build_batch(
            meta, repeated_ids, artifacts, vocab, chemical_mode="correct", seed=20260814,
        )
        zero = __import__("baseline.baseline.training_v2", fromlist=["build_batch"]).build_batch(
            meta, repeated_ids, artifacts, vocab, chemical_mode="zero", seed=20260814,
        )
        self.assertEqual(float(zero.morgan.abs().sum() + zero.descriptors.abs().sum()), 0.0)
        self.assertFalse(zero.chemical_valid_mask.any())
        self.assertTrue(torch.equal(zero.chemical_mapping, correct.chemical_mapping))
        self.assertTrue(torch.equal(zero.chemical_confidence, correct.chemical_confidence))
        self.assertTrue(torch.equal(zero.chemical_structure_valid, correct.chemical_structure_valid))

    def test_chemical_ablation_configs_share_fixed_stage_a_and_hyperparameters(self):
        root = Path(__file__).resolve().parents[2]
        base = load_config(root / "baseline/configs/model_v2_huber_no_batch.yaml")
        checkpoint = root / "reports/model_v2_stage2/formal_huber_no_batch_seed_20260814/stage_a_best.pt"
        for mode in ("shuffle", "zero"):
            config = load_config(root / f"baseline/configs/model_v2_huber_no_batch_chemical_{mode}.yaml")
            self.assertEqual(config["model"]["chemical_mode"], mode)
            self.assertFalse(config["model"]["batch_enabled"])
            self.assertEqual(config["model"]["genome_mode"], "correct")
            self.assertEqual(config["loss"], base["loss"])
            fixed = config["training"]["fixed_stage_a_checkpoint"]
            self.assertEqual(fixed["sha256"], sha256_file(checkpoint))
            training = dict(config["training"]); training.pop("fixed_stage_a_checkpoint")
            self.assertEqual(training, base["training"])

    def test_validation_macro_huber_is_invariant_to_batch_size(self):
        model, batch = make_model(), synthetic_batch(7)
        model.eval()
        target = torch.randn(7, 6)
        mask = torch.rand(7, 6) > 0.25
        loaders_small = {name: make_loader(batch, target, mask, 2) for name in VAL_SCENARIOS}
        loaders_large = {name: make_loader(batch, target, mask, 7) for name in VAL_SCENARIOS}
        small, small_scenarios = validation_macro_huber(model, loaders_small, LossWeights())
        large, large_scenarios = validation_macro_huber(model, loaders_large, LossWeights())
        self.assertAlmostEqual(small, large, places=7)
        for scenario in VAL_SCENARIOS:
            self.assertAlmostEqual(small_scenarios[scenario], large_scenarios[scenario], places=7)

    def test_stage_a_control_monitor_is_response_invariant_and_rejects_treatment(self):
        model, batch = make_model(), synthetic_batch(4)
        control_batch = batch.replace(is_control=torch.ones_like(batch.is_control))
        target = torch.randn(4, 6)
        mask = torch.ones(4, 6, dtype=torch.bool)
        before, _ = control_validation_huber(model, [(control_batch, target, mask)])
        with torch.no_grad():
            for parameter in model.response_branch.parameters():
                parameter.add_(torch.randn_like(parameter) * 100)
        after, _ = control_validation_huber(model, [(control_batch, target, mask)])
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            control_validation_huber(model, [(batch, target, mask)])

    def test_real_control_anchor_has_full_protein_coverage_and_expected_validation_controls(self):
        artifacts = load_artifact_bundle()
        meta = load_train_val_metadata()
        labels, masks = load_label_frames(meta, artifacts)
        train_control_ids = meta.index[
            meta["split_final"].eq("train")
            & meta["perturbation_no_concentration"].astype(str).str.lower().isin({"water", "dmso"})
        ]
        _, counts = fit_control_protein_anchor(meta, labels, masks, train_control_ids)
        self.assertEqual(len(counts), artifacts.feature_contract.n_proteins)
        self.assertGreater(int(counts.min()), 0)
        validation_control_ids = stage_a_validation_control_ids(meta)
        splits = meta.loc[validation_control_ids, "split_final"].value_counts().to_dict()
        self.assertEqual(splits, {"val_strain_only": 190, "val_time": 15})
        self.assertTrue(meta.loc[validation_control_ids, "perturbation_no_concentration"].str.lower().isin({"water", "dmso"}).all())

    def test_checkpoint_contains_manifests_and_resumes_optimizer(self):
        artifacts = load_artifact_bundle()
        self.assertIn("chemical_source_manifest_sha256", artifacts.hashes)
        self.assertIn("genome_source_manifest_sha256", artifacts.hashes)
        model, batch = make_model(), synthetic_batch(2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        target = torch.ones(2, 6)
        loss = (model(batch)["y_pred"] - target).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        early = {"best_epoch": 3, "best_monitor": 0.4, "bad_epochs": 0, "patience": 2}
        anchor_metadata = {"anchor_source": "train_controls_only", "control_sample_count": 751, "min_observations_per_protein": 94}
        payload = checkpoint_payload(model, optimizer, 3, 0.4, early, {"x": 1}, artifacts.hashes, 19, "B", anchor_metadata)
        saved_state = copy.deepcopy(payload["model_state"])
        expected_model = make_model(); expected_model.load_state_dict(saved_state)
        expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=0.01)
        expected_optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
        expected_loss = (expected_model(batch)["y_pred"] - target).square().mean()
        expected_loss.backward(); expected_optimizer.step()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            save_checkpoint(path, payload)
            for parameter in model.parameters():
                parameter.data.add_(10)
            restored = restore_checkpoint(path, model, optimizer, artifacts.hashes)
            resumed_loss = (model(batch)["y_pred"] - target).square().mean()
            resumed_loss.backward(); optimizer.step()
        self.assertEqual(restored["early_stopping"], early)
        self.assertEqual(restored["epoch"], 3)
        self.assertEqual(restored["seed"], 19)
        self.assertEqual(restored["config"], {"x": 1})
        self.assertEqual(restored["run_metadata"], anchor_metadata)
        for actual, expected in zip(model.parameters(), expected_model.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
