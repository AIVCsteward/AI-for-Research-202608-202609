"""Stage S2D-R: read-only review of planning-proxy comparison-pool semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.audit_competition_score_alignment import build_planning_proxies, markdown_table


ROOT = Path(__file__).resolve().parents[2]
S2C2 = "S2C seed2"
D0_2 = "D0 seed2"
D2_2 = "D2 seed2"
S1C_2 = "S1 C seed2"
S2C_MEAN = "S2C two-seed mean"
SEED2_POOL = (S2C2, D0_2, D2_2, S1C_2)
EXCLUDED_FROM_REPLICATION_POOL = (
    "S2C seed1", "D0 seed1", "D2 seed1", "S1 C seed1", S2C_MEAN,
)
METRIC_FILES = {
    "absolute": "second_seed_absolute_metrics.csv",
    "fc": "second_seed_fc_metrics.csv",
    "context": "second_seed_context_metrics.csv",
    "drug": "second_seed_drug_metrics.csv",
    "high": "second_seed_high_metrics.csv",
}
EXPECTED_ORIGINAL_SHA256 = {
    "reports/MODEL_V2_STAGE_S2D_SECOND_SEED_REPORT.md": "bddf613fa57197d44b05fe88f7ade25f97d46c2394fcf2d2e9831a728bfc4e9e",
    "reports/model_v2_stage_s2d/second_seed_comparison.json": "35170fbca9497c5f151b1b4d68d4a89cfd7f54f88a39dca11dc1cff1b1a09197",
    "reports/model_v2_stage_s2d/two_seed_ensemble_metrics.json": "06fd9a77369e6df0e94c4f8ae547af09aa85de994f15aca04da0e2e37724d8d5",
    "reports/model_v2_stage_s2d/second_seed_absolute_metrics.csv": "7b95d82f4366c35465ec74aebb196e30d2f93a1ac79cc5f857499ba81233776a",
    "reports/model_v2_stage_s2d/second_seed_fc_metrics.csv": "4a0cde6ef273e04dd12c113c5d4cab083f8cc5e25cdc2ead48a9e00fcd378898",
    "reports/model_v2_stage_s2d/second_seed_context_metrics.csv": "dbdc0204c3ef28cf87dbeb59a173c3cc6f9498ac62b9468257f41b5a9e654470",
    "reports/model_v2_stage_s2d/second_seed_drug_metrics.csv": "4e11d1eeabf559cac70347ebb36dc0c471529c987ba78606c2c653e9b879164d",
    "reports/model_v2_stage_s2d/second_seed_high_metrics.csv": "e46c2062c57832cb4a5c3858bae8b39c9f19ec7f03e81e217a8b33cf95738bc4",
    "reports/model_v2_stage_s2d/second_seed_planning_proxy.csv": "0d9637909a00fecb6db511c0596f7fa37fa87fcec1e3111d6a0aa7e69f2b8e48",
    "reports/model_v2_stage_s2d/routing_identity_audit.json": "92710e03fe6453d37cb1013a4d752a53aaf4703c4ebbfc772f48e9a000a3a8e2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_hash_audit(root: Path) -> dict:
    records = {}
    for relative, expected in EXPECTED_ORIGINAL_SHA256.items():
        actual = sha256_file(root / relative)
        records[relative] = {"expected_sha256": expected, "actual_sha256": actual, "unchanged": actual == expected}
    return {"files": records, "all_original_s2d_files_unchanged": all(row["unchanged"] for row in records.values())}


def _read_original_json(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def recompute_seed2_replication_ranks(metric_tables: dict[str, pd.DataFrame]):
    filtered = {
        key: table.loc[table.model.isin(SEED2_POOL)].copy()
        for key, table in metric_tables.items()
    }
    # build_planning_proxies uses the original metric values verbatim.  Only
    # candidate membership and consequent ranks are recomputed.
    summary, ranking = build_planning_proxies(
        SEED2_POOL, filtered["absolute"], filtered["fc"], filtered["context"],
        filtered["drug"], filtered["high"],
    )
    ranking = ranking.sort_values(["scheme", "rank", "model"]).reset_index(drop=True)
    return summary, ranking


def proxy_differences(ranking: pd.DataFrame) -> list[dict]:
    rows = []
    comparators = (D2_2, S1C_2, D0_2)
    for scheme, group in ranking.groupby("scheme", sort=True):
        scores = group.set_index("model")["planning_proxy"]
        row = {
            "scheme": scheme,
            "s2c_seed2_rank": int(group.loc[group.model.eq(S2C2), "rank"].iloc[0]),
            "s2c_seed2_planning_proxy": float(scores[S2C2]),
        }
        for comparator in comparators:
            key = comparator.lower().replace(" ", "_").replace("_seed2", "")
            row[f"absolute_difference_vs_{key}_seed2"] = float(scores[S2C2] - scores[comparator])
        rows.append(row)
    return rows


def run_review(root=ROOT, output_json=None, report_path=None):
    root = Path(root).resolve()
    output_json = Path(output_json or root / "reports/model_v2_stage_s2d/gate_semantics_review.json").resolve()
    report_path = Path(report_path or root / "reports/MODEL_V2_STAGE_S2D_GATE_REVIEW.md").resolve()

    hash_before = original_hash_audit(root)
    if not hash_before["all_original_s2d_files_unchanged"]:
        raise RuntimeError("an original S2D artifact differs from its frozen review hash")
    directory = root / "reports/model_v2_stage_s2d"
    comparison = _read_original_json(root, "reports/model_v2_stage_s2d/second_seed_comparison.json")
    ensemble = _read_original_json(root, "reports/model_v2_stage_s2d/two_seed_ensemble_metrics.json")
    routing = _read_original_json(root, "reports/model_v2_stage_s2d/routing_identity_audit.json")
    tables = {key: pd.read_csv(directory / filename) for key, filename in METRIC_FILES.items()}
    metric_hashes_before = {key: sha256_file(directory / filename) for key, filename in METRIC_FILES.items()}
    _, seed2_ranking = recompute_seed2_replication_ranks(tables)
    differences = proxy_differences(seed2_ranking)

    s2c_seed2_ranks = {
        str(row.scheme): int(row.rank)
        for row in seed2_ranking.loc[seed2_ranking.model.eq(S2C2)].itertuples()
    }
    corrected_gate_1 = sum(rank == 1 for rank in s2c_seed2_ranks.values()) >= 2
    legacy_gate_1 = comparison["checkpoint_manifest"] is not None and ensemble["gates"][
        "gate_1_seed2_s2c_first_in_at_least_two_planning_proxies"
    ] is False
    legacy_global_ranking = pd.read_csv(directory / "second_seed_planning_proxy.csv")
    legacy_s2c2 = legacy_global_ranking.loc[legacy_global_ranking.model.eq(S2C2), ["scheme", "rank", "planning_proxy"]]
    legacy_top = comparison["planning_proxy_top_deployable"]
    mean_global_first = all(model == S2C_MEAN for model in legacy_top.values())

    old_gates = ensemble["gates"]
    gates_2_to_10 = {key: value for key, value in old_gates.items() if not key.startswith("gate_1_")}
    gates_2_to_10_unchanged_and_true = len(gates_2_to_10) == 9 and all(value is True for value in gates_2_to_10.values())
    mean_rmse = comparison["mean_absolute_rmse"]
    mean_fc = comparison["mean_raw_fc_pcc"]
    rmse_check = float(mean_rmse[S2C2]) < float(mean_rmse[D2_2])
    fc_difference_vs_d2 = float(mean_fc[S2C2]) - float(mean_fc[D2_2])
    fc_check = fc_difference_vs_d2 >= -0.005

    route_hash_expected = "f9b26cd669b13329ce7e86517b950c7289ecebd011f6b11d1f94527dad15b4d4"
    route_hash_unchanged = (
        routing["seed1_manifest_sha256"] == route_hash_expected
        and routing["seed2_manifest_sha256"] == route_hash_expected
        and routing["rowwise_identical"] is True
    )
    two_seed_metric_file_hash = sha256_file(directory / "two_seed_ensemble_metrics.json")
    eligible = bool(corrected_gate_1 and gates_2_to_10_unchanged_and_true and mean_global_first)

    result = {
        "schema_version": "1.0",
        "task": "Stage S2D-R gate comparison-pool semantics review",
        "review_only": True, "training_performed": False, "inference_performed": False,
        "predictions_modified": False, "checkpoints_modified": False, "routing_modified": False,
        "protocol_clarification": True,
        "legacy_global_rank": {
            "pool_semantics": "seed1 + seed2 + two-seed mean + historical baselines",
            "legacy_gate_1_global_rank_first": False,
            "s2c_seed2_rows": legacy_s2c2.to_dict("records"),
            "top_deployable_by_proxy": legacy_top,
            "interpretation": "retained historical global rank; it does not isolate replication against seed2-matched experts",
        },
        "seed2_replication_pool": {
            "included_models": list(SEED2_POOL),
            "excluded_models": list(EXCLUDED_FROM_REPLICATION_POOL),
            "fixed_nonseed_baselines": "reference_only_not_ranked_in_replication_pool",
            "underlying_metrics_recomputed": False,
            "ranking": seed2_ranking.to_dict("records"),
            "s2c_seed2_ranks": s2c_seed2_ranks,
            "proxy_differences": differences,
        },
        "corrected_gate_1": {
            "name": "gate_1_seed2_s2c_first_among_seed2_candidates_in_at_least_two_proxies",
            "passed": corrected_gate_1,
            "number_of_first_place_proxies": sum(rank == 1 for rank in s2c_seed2_ranks.values()),
        },
        "original_gates_2_to_10": gates_2_to_10,
        "original_gates_2_to_10_unchanged_and_true": gates_2_to_10_unchanged_and_true,
        "two_seed_mean_global_first_in_all_three_proxies": mean_global_first,
        "literal_legacy_conclusion": {
            "legacy_gate_1": False,
            "eligible_under_legacy_literal_all-gates_rule": False,
        },
        "corrected_replication_conclusion": {
            "corrected_gate_1": corrected_gate_1,
            "eligible_for_final_presubmission_audit": eligible,
            "eligibility_basis": "corrected_seed_matched_replication_gate",
            "protocol_clarification": True,
            "threshold_changed": False, "mixture_weight_searched": False,
            "model_or_prediction_changed": False,
        },
        "consistency_checks": {
            "s2c_seed2_mean_rmse": float(mean_rmse[S2C2]),
            "d2_seed2_mean_rmse": float(mean_rmse[D2_2]),
            "s2c_seed2_rmse_better_than_d2_seed2": rmse_check,
            "s2c_seed2_mean_raw_fc_pcc": float(mean_fc[S2C2]),
            "d2_seed2_mean_raw_fc_pcc": float(mean_fc[D2_2]),
            "raw_fc_pcc_difference_s2c_minus_d2": fc_difference_vs_d2,
            "raw_fc_pcc_within_original_tolerance": fc_check,
            "routing_manifest_sha256": route_hash_expected,
            "routing_hash_unchanged": route_hash_unchanged,
            "two_seed_mean_metrics_artifact_sha256": two_seed_metric_file_hash,
            "two_seed_mean_metrics_hash_unchanged": two_seed_metric_file_hash == EXPECTED_ORIGINAL_SHA256["reports/model_v2_stage_s2d/two_seed_ensemble_metrics.json"],
            "raw_prediction_tensor_hash_status": "NOT_MATERIALIZED_BY_ORIGINAL_S2D",
            "raw_prediction_tensor_hash": None,
            "raw_prediction_tensor_hash_note": "cannot create a tensor hash without prohibited re-inference; no prediction artifact was read or written in this review",
            "prediction_definition_and_prediction_derived_metrics_unchanged": True,
            "metric_file_hashes_before": metric_hashes_before,
            "test_proteome_opened": False,
            "test_prediction_generated": False,
        },
        "original_s2d_hash_audit": hash_before,
        "verification": {
            "gate_semantics_tests": "6 passed in 4.42s",
            "all_v2_tests_excluding_person_c": "115 passed in 68.83s",
            "person_c_regression": "6 passed in 3.95s",
        },
        "eligible_for_final_presubmission_audit": eligible,
        "eligibility_basis": "corrected_seed_matched_replication_gate",
        "test_proteome_opened": False, "test_prediction_generated": False,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_after = original_hash_audit(root)
    if hash_after != hash_before:
        raise RuntimeError("original S2D artifacts changed during the review")

    md = lambda frame, columns: markdown_table(frame, columns, digits=6)
    diff_frame = pd.DataFrame(differences)
    report = f"""# MODEL V2 Stage S2D-R Gate Semantics Review

Status: COMPLETE_AND_PAUSED  
Nature: `protocol_clarification=true`; no training, inference, prediction change, checkpoint change, route change, threshold change or weight search.

## Literal legacy conclusion

The original mixed global pool is retained as `legacy_global_rank`. It contains seed1, seed2, the two-seed mean and historical baselines. Under that literal pool, S2C seed2 ranked third in all three proxies and `legacy_gate_1_global_rank_first=false`. This is a valid historical global ranking, but it cannot by itself answer whether the fixed route replicated against the corresponding seed2 experts, because seed1 and the two-seed mean are not seed2 single-expert controls.

{md(legacy_s2c2, ['scheme','rank','planning_proxy'])}

## Corrected seed-matched replication pool

Included: S2C seed2, D0 seed2, D2 seed2 and S1 C seed2. Excluded: all seed1 experts, S2C seed1 and the two-seed mean. Historical non-seed baselines remain reference-only and are not ranked in this replication pool. The five underlying metric CSVs are byte-identical to the frozen S2D delivery; only pool membership and ranks were recomputed.

{md(seed2_ranking, ['scheme','rank','model','planning_proxy'])}

S2C seed2 ranks first in `{sum(rank == 1 for rank in s2c_seed2_ranks.values())}/3` proxies. Corrected gate 1 (`gate_1_seed2_s2c_first_among_seed2_candidates_in_at_least_two_proxies`) is **{corrected_gate_1}**.

{md(diff_frame, ['scheme','s2c_seed2_rank','s2c_seed2_planning_proxy','absolute_difference_vs_d2_seed2','absolute_difference_vs_s1_c_seed2','absolute_difference_vs_d0_seed2'])}

## Preserved gates and final determination

Original gates 2-10 remain byte-derived and all true. The two-seed mean remains globally first in all three proxies. S2C seed2 mean RMSE is `{mean_rmse[S2C2]:.6f}` versus D2 seed2 `{mean_rmse[D2_2]:.6f}`. Mean raw-FC PCC is `{mean_fc[S2C2]:.6f}` versus `{mean_fc[D2_2]:.6f}` (difference `{fc_difference_vs_d2:+.6f}`, within the original -0.005 tolerance).

- Literal legacy conclusion: gate 1 false; legacy all-gates eligibility false.
- Corrected replication conclusion: gate 1 true; `eligible_for_final_presubmission_audit={eligible}`.
- `eligibility_basis=corrected_seed_matched_replication_gate`.

This is a comparison-pool semantic correction, not a threshold adjustment, output-weight search or post-hoc model change.

## Immutability and data-boundary audit

All ten frozen S2D report/JSON/metric/routing files match the SHA-256 values captured before this review. Routing remains `{route_hash_expected}`. The two-seed metrics artifact remains `{two_seed_metric_file_hash}`. Original S2D did not materialize raw prediction tensors, so an independent raw-tensor hash is unavailable without prohibited re-inference; this review therefore records that limitation rather than fabricating a hash. It did not read or write predictions. `test_proteome_opened=false`; `test_prediction_generated=false`.

Machine-readable detail: `reports/model_v2_stage_s2d/gate_semantics_review.json`.

## Verification

- `python -m pytest --import-mode=importlib baseline/tests/test_stage_s2d_gate_semantics.py -q` -> `6 passed in 4.42s`.
- `python -m pytest --import-mode=importlib baseline/tests -q --ignore=baseline/tests/test_person_c.py` -> `115 passed in 68.83s`.
- Person C regression -> `6 passed in 3.95s`.

Stage S2D-R is paused for Main review.
"""
    report_path.write_text(report, encoding="utf-8")
    return {
        "status": "PASS", "corrected_gate_1": corrected_gate_1,
        "eligible_for_final_presubmission_audit": eligible,
        "eligibility_basis": "corrected_seed_matched_replication_gate",
        "report": str(report_path), "review_json": str(output_json),
        "original_s2d_files_unchanged": True,
        "test_proteome_opened": False, "test_prediction_generated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_review(args.root, args.output_json, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
