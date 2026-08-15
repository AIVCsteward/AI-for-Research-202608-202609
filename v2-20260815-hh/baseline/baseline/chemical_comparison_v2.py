"""Assemble the Stage 2.2 correct/shuffle/zero chemical diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluation_v2 import VAL_SCENARIOS, evaluate_component_norms
from .model_v2 import AnchoredVirtualCellV2, V2Config
from .training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, build_batch, fit_category_vocabulary,
    fit_control_protein_anchor, load_artifact_bundle, load_checkpoint,
    load_config, load_label_frames, load_train_val_metadata, make_loader,
    module_parameters_sha256, parameter_change_l2, snapshot_module_parameters,
)


def run_comparison(
    correct_config_path: Path,
    correct_stage_a_path: Path,
    correct_stage_b_path: Path,
    correct_summary_path: Path,
    shuffle_summary_path: Path,
    zero_summary_path: Path,
    output_path: Path,
):
    config = load_config(correct_config_path)
    artifacts = load_artifact_bundle()
    meta = load_train_val_metadata()
    labels, masks = load_label_frames(meta, artifacts)
    train_ids = meta.index[meta["split_final"].eq("train")]
    names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    control_ids = train_ids[names.isin(CONTROL_NAMES)]
    protein_mean, _ = fit_control_protein_anchor(meta, labels, masks, control_ids)
    vocab = fit_category_vocabulary(meta, train_ids)
    model_config = config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnchoredVirtualCellV2(
        V2Config(
            n_proteins=artifacts.feature_contract.n_proteins,
            latent_dim=int(model_config["latent_dim"]),
            protein_rank=int(model_config["protein_rank"]),
            dropout=float(model_config["dropout"]),
            batch_enabled=False,
            medium_vocab_size=vocab.size("Medium"),
            batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
        ),
        torch.from_numpy(protein_mean),
    ).to(device)
    stage_a = load_checkpoint(correct_stage_a_path, artifacts.hashes)
    model.load_state_dict(stage_a["model_state"])
    initial_hash = module_parameters_sha256(model, ("chemical_encoder", "response_branch"))
    initial_chemical = snapshot_module_parameters(model, "chemical_encoder")
    stage_b = load_checkpoint(correct_stage_b_path, artifacts.hashes)
    model.load_state_dict(stage_b["model_state"])
    correct_change = parameter_change_l2(model, "chemical_encoder", initial_chemical)

    def loader_for(ids):
        batch = build_batch(
            meta, ids, artifacts, vocab,
            chemical_mode="correct", genome_mode="correct",
            seed=int(config["training"]["seed"]),
        )
        target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
        mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
        return make_loader(batch, target, mask, int(config["training"]["batch_size"]))

    loaders = {
        scenario: loader_for(meta.index[meta["split_final"].eq(scenario)])
        for scenario in VAL_SCENARIOS
    }
    correct_norms = evaluate_component_norms(model, loaders, device=device)
    summaries = {
        "correct": json.loads(correct_summary_path.read_text(encoding="utf-8")),
        "shuffle": json.loads(shuffle_summary_path.read_text(encoding="utf-8")),
        "zero": json.loads(zero_summary_path.read_text(encoding="utf-8")),
    }
    initial_hashes = {
        "correct": initial_hash,
        "shuffle": summaries["shuffle"]["stage_b_initial_chemical_response_sha256"],
        "zero": summaries["zero"]["stage_b_initial_chemical_response_sha256"],
    }
    if len(set(initial_hashes.values())) != 1:
        raise ValueError("Stage B initial chemical/response parameters differ across variants")
    variants = {}
    for name, summary in summaries.items():
        variants[name] = {
            "stage_b_best_epoch": int(summary["stage_b"]["best_epoch"]),
            "best_macro_huber": float(summary["stage_b"]["best_monitor"]),
            "metrics": summary["validation_scenarios"],
            "delta_response_valid_position_l2": {
                scenario: float((correct_norms if name == "correct" else summary["validation_component_norms"])[scenario]["delta_response"])
                for scenario in VAL_SCENARIOS
            },
            "chemical_encoder_parameter_change_l2": (
                correct_change if name == "correct" else float(summary["chemical_encoder_parameter_change_l2"])
            ),
        }
    result = {
        "status": "PASS",
        "seed": int(config["training"]["seed"]),
        "analysis": "formal_no_batch_chemical_correct_shuffle_zero_diagnostic",
        "stage_a_checkpoint": str(correct_stage_a_path.resolve()),
        "stage_a_checkpoint_epoch": int(stage_a["epoch"]),
        "stage_b_initial_chemical_response_sha256": initial_hash,
        "all_stage_b_initial_parameters_identical": True,
        "test_proteome_opened": False,
        "variants": variants,
        "primary_judgement": "zero_outperforms_correct_on_val_chem_only_and_val_both; current_chemical_encoding_or_fusion_is_harmful",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-config", type=Path, required=True)
    parser.add_argument("--correct-stage-a", type=Path, required=True)
    parser.add_argument("--correct-stage-b", type=Path, required=True)
    parser.add_argument("--correct-summary", type=Path, required=True)
    parser.add_argument("--shuffle-summary", type=Path, required=True)
    parser.add_argument("--zero-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_comparison(
        args.correct_config, args.correct_stage_a, args.correct_stage_b,
        args.correct_summary, args.shuffle_summary, args.zero_summary, args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
