"""Concrete C5 real-data runner for the loss-ablation framework.

The default command only prints the plan.  ``--run-ready`` is required to
start training.  ``--include-correlation`` must not be used until B4's final
``build_protein_graph`` implementation is merged; C then computes aligned edge
targets without changing the graph.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.config import get_experiment_config
from baseline.data import get_split_masks, preprocess
from baseline.evaluation import (
    TEST_SPLITS,
    VAL_SPLITS,
    build_control_lookup,
    build_matched_control_pairs,
    compute_protein_mean,
    matched_control_predict,
)
from baseline.features import build_condition_features, fit_feature_encoders
from baseline.losses import compute_target_edge_corr
from baseline.model import AIVCModel
from baseline.training import prepare_fold_change_index, prepare_training_data, train
from experiments.ablation_loss import (
    build_experiment_plan,
    evaluate_prediction_metrics,
    run_experiment_plan,
    validate_correlation_bundle,
)
from experiments.report_results import write_experiment_log
from run_stage2 import load_train_val


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_metadata(data_dir):
    path = Path(data_dir) / "WAYB_WAYC_metadata_test(1).csv"
    return pd.read_csv(path).set_index("sample_ID")


def load_test_truth(data_dir, protein_names):
    """Load held-out truth only for post-training self-evaluation."""
    data_dir = Path(data_dir)
    meta = load_test_metadata(data_dir)
    header = pd.read_csv(data_dir / "WAYB_WAYC_proteome_raw_test.csv", nrows=0).columns
    missing = [name for name in protein_names if name not in header]
    if missing:
        raise ValueError(f"Test truth is missing retained proteins: {missing[:5]}")
    usecols = ["sample_ID", *protein_names]
    dtypes = {name: np.float32 for name in protein_names}
    raw = pd.read_csv(
        data_dir / "WAYB_WAYC_proteome_raw_test.csv",
        usecols=usecols,
        dtype=dtypes,
    ).set_index("sample_ID")
    common = meta.index.intersection(raw.index, sort=False)
    meta = meta.loc[common]
    raw = raw.loc[common, protein_names]
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log2 = np.log2(raw.astype(np.float32))
    mask = pd.DataFrame(
        np.isfinite(y_log2.to_numpy()), index=y_log2.index, columns=y_log2.columns
    )
    y_log2 = y_log2.where(mask)
    return meta, y_log2, mask


@torch.no_grad()
def predict_in_batches(model, features, device, batch_size=256):
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(
            features[start : start + batch_size], dtype=torch.float32, device=device
        )
        output = model(x)
        pred = output["y_pred"] if isinstance(output, dict) else output
        predictions.append(pred.detach().cpu().numpy())
    if not predictions:
        return np.empty((0, model.n_proteins), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def evaluate_splits(model, X_all, meta, y_log2, mask_matrix, split_masks, splits, device):
    metrics = {}
    for split_name in splits:
        split_mask = split_masks.get(split_name)
        if split_mask is None or int(split_mask.sum()) == 0:
            continue
        rows = split_mask.to_numpy(dtype=bool)
        prediction = predict_in_batches(model, X_all[rows], device)
        split_metrics = evaluate_prediction_metrics(
            y_log2.loc[split_mask].to_numpy(),
            prediction,
            mask_matrix.loc[split_mask].to_numpy(),
        )
        split_metrics["n_samples"] = int(rows.sum())
        split_metrics["fc_pearson"] = None
        metrics[split_name] = split_metrics
    return metrics


def build_matched_control_reference(meta, y_log2, mask_matrix, train_mask, split_masks):
    protein_mean = compute_protein_mean(y_log2, train_mask)
    lookup, control_mean, _ = build_control_lookup(meta, y_log2, train_mask)
    metrics = {}
    for split_name in VAL_SPLITS:
        split_mask = split_masks.get(split_name)
        if split_mask is None or int(split_mask.sum()) == 0:
            continue
        prediction = matched_control_predict(
            meta.loc[split_mask], y_log2, lookup, protein_mean, control_mean
        )
        values = evaluate_prediction_metrics(
            y_log2.loc[split_mask].to_numpy(),
            prediction,
            mask_matrix.loc[split_mask].to_numpy(),
        )
        values["n_samples"] = int(split_mask.sum())
        metrics[split_name] = values
    return metrics


def prepare_c5_context(data_dir, device, config):
    raw_meta, raw_proteome = load_train_val(Path(data_dir))
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(
        raw_meta, raw_proteome
    )
    split_masks = get_split_masks(meta)
    train_meta = meta.loc[train_mask]
    encoders = fit_feature_encoders(
        train_meta,
        y_log2.loc[train_mask],
        mask_matrix.loc[train_mask],
        config=config,
    )
    X_all = build_condition_features(meta, encoders=encoders)
    X_train, y_train, mask_train, val_data = prepare_training_data(
        X_all,
        y_log2,
        mask_matrix,
        train_mask,
        split_masks,
        VAL_SPLITS,
        device,
    )
    pairs = build_matched_control_pairs(meta, train_mask)
    fc_control_index = prepare_fold_change_index(
        meta.index[train_mask].tolist(), pairs, device=device
    )
    baseline_metrics = build_matched_control_reference(
        meta, y_log2, mask_matrix, train_mask, split_masks
    )
    return {
        "meta": meta,
        "y_log2": y_log2,
        "mask_matrix": mask_matrix,
        "protein_names": protein_names,
        "train_mask": train_mask,
        "split_masks": split_masks,
        "encoders": encoders,
        "X_all": X_all,
        "X_train": X_train,
        "y_train": y_train,
        "mask_train": mask_train,
        "val_data": val_data,
        "fc_control_index": fc_control_index,
        "baseline_metrics": baseline_metrics,
    }


def prepare_correlation_bundle(context, config):
    """Consume B4's final graph and produce C3 targets in identical edge order."""
    from baseline.protein_graph import build_protein_graph

    edge_index = build_protein_graph(
        context["y_log2"].loc[context["train_mask"]],
        context["mask_matrix"].loc[context["train_mask"]],
        k=config["gnn"]["k_neighbors"],
        threshold=config["gnn"]["pearson_threshold"],
    )
    target = compute_target_edge_corr(
        context["y_train"].detach().cpu(),
        edge_index.detach().cpu(),
        mask=context["mask_train"].detach().cpu().bool(),
    )
    bundle = {"edge_index": edge_index.detach().cpu(), "target_edge_corr": target}
    validate_correlation_bundle(bundle, len(context["protein_names"]))
    return bundle


def compare_with_matched_control(model_metrics, baseline_metrics):
    by_split = {}
    wins = 0
    compared = 0
    for split_name in VAL_SPLITS:
        model_value = model_metrics.get(split_name, {}).get("per_protein_r2_median")
        baseline_value = baseline_metrics.get(split_name, {}).get("per_protein_r2_median")
        if model_value is None or baseline_value is None:
            continue
        delta = float(model_value - baseline_value)
        by_split[split_name] = {
            "model_per_protein_r2": float(model_value),
            "matched_control_per_protein_r2": float(baseline_value),
            "delta": delta,
        }
        compared += 1
        wins += int(delta > 0)
    return {
        "summary": f"{wins}/{compared} validation splits exceed Matched Control",
        "attribution": "Machine comparison only; biological attribution is recorded in C6 after review.",
        "by_split": by_split,
    }


def make_real_experiment_runner(
    context,
    config,
    data_dir,
    output_dir,
    device,
    seed,
    evaluate_test_truth=False,
    create_submission=True,
    use_gnn=False,
):
    def runner(plan_item, correlation_bundle):
        set_seed(seed)
        experiment_dir = Path(output_dir) / plan_item["name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if use_gnn and correlation_bundle is None:
            raise ValueError("use_gnn=True requires the final B4/C3 correlation bundle")

        model = AIVCModel(
            dim_in=context["X_train"].shape[1],
            n_proteins=len(context["protein_names"]),
            dim_emb=config["encoder"]["d_emb"],
            use_gnn=use_gnn,
        ).to(device)
        edge_index = None
        target_edge_corr = None
        if correlation_bundle is not None:
            edge_index = correlation_bundle["edge_index"].to(device)
            target_edge_corr = correlation_bundle["target_edge_corr"].to(device)
            if use_gnn:
                model.set_edge_index(edge_index)

        model, history = train(
            model,
            context["X_train"],
            context["y_train"],
            context["mask_train"],
            context["val_data"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            device=device,
            verbose=True,
            fc_control_index=context["fc_control_index"],
            edge_index=edge_index,
            target_edge_corr=target_edge_corr,
            loss_weights=plan_item["loss_weights"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            early_stopping_split="val_both",
        )

        checkpoint_path = experiment_dir / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)
        metrics = evaluate_splits(
            model,
            context["X_all"],
            context["meta"],
            context["y_log2"],
            context["mask_matrix"],
            context["split_masks"],
            VAL_SPLITS,
            device,
        )

        if evaluate_test_truth:
            test_meta, test_y, test_mask = load_test_truth(data_dir, context["protein_names"])
            X_test = build_condition_features(test_meta, encoders=context["encoders"])
            test_split_masks = get_split_masks(test_meta)
            metrics.update(
                evaluate_splits(
                    model,
                    X_test,
                    test_meta,
                    test_y,
                    test_mask,
                    test_split_masks,
                    TEST_SPLITS,
                    device,
                )
            )

        submission_checks = {
            "n_samples": None,
            "n_proteins": len(context["protein_names"]),
            "no_na": None,
            "no_inf": None,
            "prediction_scale": "log2",
        }
        prediction_path = None
        if create_submission:
            test_meta = load_test_metadata(data_dir)
            X_test = build_condition_features(test_meta, encoders=context["encoders"])
            prediction = predict_in_batches(model, X_test, device)
            submission = pd.DataFrame(
                prediction, index=test_meta.index, columns=context["protein_names"]
            )
            submission.index.name = "sample_ID"
            submission_checks.update(
                {
                    "n_samples": int(submission.shape[0]),
                    "no_na": bool(not submission.isna().any().any()),
                    "no_inf": bool(np.isfinite(submission.to_numpy()).all()),
                }
            )
            if not submission_checks["no_na"] or not submission_checks["no_inf"]:
                raise RuntimeError("Submission contains NA or Inf")
            prediction_path = experiment_dir / "prediction.csv"
            submission.to_csv(prediction_path)

        config_path = experiment_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "loss_weights": plan_item["loss_weights"],
                    "shared_config": config,
                    "use_gnn": use_gnn,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "history": history,
            "metrics": metrics,
            "baseline_comparison": compare_with_matched_control(
                metrics, context["baseline_metrics"]
            ),
            "submission_checks": submission_checks,
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "prediction": None if prediction_path is None else str(prediction_path),
            },
        }

    return runner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Person C5 loss ablations")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/c5"))
    parser.add_argument("--run-ready", action="store_true", help="Start ready experiments")
    parser.add_argument("--include-correlation", action="store_true")
    parser.add_argument("--use-gnn", action="store_true")
    parser.add_argument("--evaluate-test-truth", action="store_true")
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.use_gnn and not args.include_correlation:
        parser.error("--use-gnn requires --include-correlation")
    overrides = {"training": {}}
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    config = get_experiment_config(overrides)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not args.run_ready:
        plan = build_experiment_plan(4422)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Plan only. Use --run-ready to start training after confirming dependencies.")
        return

    context = prepare_c5_context(args.data_dir, device, config)
    correlation_bundle = None
    if args.include_correlation:
        correlation_bundle = prepare_correlation_bundle(context, config)
    runner = make_real_experiment_runner(
        context,
        config,
        args.data_dir,
        args.output_dir,
        device,
        args.seed,
        evaluate_test_truth=args.evaluate_test_truth,
        create_submission=not args.no_submission,
        use_gnn=args.use_gnn,
    )
    payload, results_path = run_experiment_plan(
        runner,
        len(context["protein_names"]),
        args.output_dir,
        correlation_bundle=correlation_bundle,
        shared_config=config,
    )
    del payload
    log_path = write_experiment_log(results_path, Path(args.output_dir) / "EXPERIMENT_LOG.md")
    print(f"C5 results: {results_path}")
    print(f"C6 log: {log_path}")


if __name__ == "__main__":
    main()
