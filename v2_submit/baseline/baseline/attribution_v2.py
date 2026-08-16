"""Post-hoc component attribution for an existing batch-enabled V2 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluation_v2 import VAL_SCENARIOS, evaluate_component_attribution
from .model_v2 import AnchoredVirtualCellV2, V2Config
from .training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, ROOT, _resolve_device, build_batch,
    fit_category_vocabulary, fit_control_protein_anchor, load_artifact_bundle,
    load_checkpoint, load_config, load_label_frames, load_train_val_metadata,
    make_loader,
)


def run_attribution(config_path: Path, checkpoint_path: Path, output_path: Path):
    config = load_config(config_path)
    if not config["model"]["batch_enabled"]:
        raise ValueError("post-hoc batch attribution requires a batch-enabled checkpoint")
    if config["loss"]["fc_weight"] != 0 or config["model"]["similarity_enabled"]:
        raise ValueError("only the approved Huber-only mainline may be attributed")
    device = _resolve_device(config["training"]["device"])
    artifacts = load_artifact_bundle()
    meta = load_train_val_metadata()
    labels, masks = load_label_frames(meta, artifacts)
    train_ids = meta.index[meta["split_final"].eq("train")]
    names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    control_ids = train_ids[names.isin(CONTROL_NAMES)]
    protein_mean, _ = fit_control_protein_anchor(meta, labels, masks, control_ids)
    vocab = fit_category_vocabulary(meta, train_ids)
    model_config = config["model"]
    model = AnchoredVirtualCellV2(
        V2Config(
            n_proteins=artifacts.feature_contract.n_proteins,
            latent_dim=int(model_config["latent_dim"]),
            protein_rank=int(model_config["protein_rank"]),
            dropout=float(model_config["dropout"]),
            batch_enabled=True,
            medium_vocab_size=vocab.size("Medium"),
            batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
        ),
        torch.from_numpy(protein_mean),
    ).to(device)
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    model.load_state_dict(payload["model_state"])

    def loader_for(ids):
        batch = build_batch(
            meta, ids, artifacts, vocab,
            chemical_mode=model_config["chemical_mode"],
            genome_mode=model_config["genome_mode"],
            seed=int(config["training"]["seed"]),
        )
        target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
        mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
        return make_loader(batch, target, mask, int(config["training"]["batch_size"]))

    loaders = {
        scenario: loader_for(meta.index[meta["split_final"].eq(scenario)])
        for scenario in VAL_SCENARIOS
    }
    result = {
        "status": "PASS",
        "analysis_type": "posthoc_component_attribution_not_a_formal_model_result",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "config": str(config_path.resolve()),
        "device": device,
        "test_proteome_opened": False,
        "scenarios": evaluate_component_attribution(model, loaders, device=device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-hoc V2 component attribution")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_attribution(args.config, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
