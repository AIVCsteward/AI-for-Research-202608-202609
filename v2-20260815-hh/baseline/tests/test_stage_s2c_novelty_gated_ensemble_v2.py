import inspect
import json
from pathlib import Path

import pandas as pd
import torch

from baseline.baseline.model_v2 import V2Batch
from baseline.baseline.novelty_gated_ensemble_v2 import (
    D0, D2, S1C, apply_hierarchical_oov_codes, derive_metadata_routing,
    enforce_routed_control_zero_response, fit_seen_sets_from_train_metadata,
)
from baseline.baseline.training_v2 import load_train_val_metadata
from scripts.audit_competition_score_alignment import stable_json_sha256


ROOT = Path(__file__).resolve().parents[2]


def _row(sample, split, chemical, strain, source="S", instrument="I", plate="P"):
    return {
        "sample_ID": sample, "split_final": split,
        "perturbation_no_concentration": chemical, "Strains": strain,
        "data_source": source, "instrument": instrument, "Yeast_cell_plate": plate,
        "Medium": "M", "Temperature": "30", "pert_time": "24", "pert_time_unit": "hour",
    }


def _batch(n=4):
    return V2Batch(
        morgan=torch.zeros(n, 2), descriptors=torch.zeros(n, 1),
        chemical_valid_mask=torch.zeros(n, 3, dtype=torch.bool),
        chemical_mapping=torch.zeros(n, dtype=torch.long), chemical_confidence=torch.zeros(n, dtype=torch.long),
        chemical_structure_valid=torch.zeros(n, 1), genome=torch.zeros(n, 3),
        genome_valid_mask=torch.zeros(n, 3, dtype=torch.bool), genome_mapping=torch.zeros(n, dtype=torch.long),
        genome_confidence=torch.zeros(n, dtype=torch.long), genome_proxy=torch.zeros(n, 1),
        medium=torch.zeros(n, dtype=torch.long), condition_numeric=torch.zeros(n, 4),
        batch_categorical=torch.ones(n, 3, dtype=torch.long), is_control=torch.zeros(n, 1, dtype=torch.bool),
    )


def test_router_obeys_fixed_priority_without_split_or_labels():
    frame = pd.DataFrame([
        _row("T0", "train", "ChemA", "StrainA"),
        _row("V0", "arbitrary", "Water", "Unknown", "X", "Y", "Z"),
        _row("V1", "arbitrary", "ChemB", "StrainA"),
        _row("V2", "arbitrary", "ChemA", "StrainB"),
        _row("V3", "arbitrary", "ChemA", "StrainB", "S", "I", "P2"),
        _row("V4", "arbitrary", "ChemA", "StrainA", "S", "I", "P2"),
    ]).set_index("sample_ID", drop=False)
    seen = fit_seen_sets_from_train_metadata(frame)
    routed = derive_metadata_routing(frame.loc[["V0", "V1", "V2", "V3", "V4"]], seen).set_index("sample_ID")
    assert routed.loc["V0", "selected_expert"] == D0
    assert routed.loc["V1", "selected_expert"] == D0
    assert routed.loc["V2", "selected_expert"] == S1C
    assert routed.loc["V3", "selected_expert"] == D2
    assert routed.loc["V4", "selected_expert"] == D2
    assert routed.loc["V0", "routing_reason"] == "control_priority"
    assert routed.loc["V3", "routing_reason"] == "seen_chemical_unseen_strain_oov_batch_tuple"
    assert "labels" not in inspect.signature(derive_metadata_routing).parameters


def test_seen_sets_are_unchanged_when_validation_metadata_changes():
    frame = pd.DataFrame([
        _row("T0", "train", "ChemA", "StrainA"),
        _row("V0", "val_chem_only", "ChemB", "StrainA"),
    ]).set_index("sample_ID", drop=False)
    first = fit_seen_sets_from_train_metadata(frame)
    changed = frame.copy()
    changed.loc["V0", ["perturbation_no_concentration", "Strains", "data_source"]] = ["Anything", "Anything", "Anything"]
    second = fit_seen_sets_from_train_metadata(changed)
    assert first == second
    assert first.hashes() == second.hashes()


def test_nested_oov_code_fallback_is_exact():
    train = pd.DataFrame([_row("T0", "train", "ChemA", "StrainA")]).set_index("sample_ID", drop=False)
    seen = fit_seen_sets_from_train_metadata(train)
    rows = pd.DataFrame([
        _row("K", "test", "ChemA", "StrainA"),
        _row("P", "test", "ChemA", "StrainA", "S", "I", "P2"),
        _row("I", "test", "ChemA", "StrainA", "S", "I2", "P2"),
        _row("S", "test", "ChemA", "StrainA", "S2", "I2", "P2"),
    ])
    result = apply_hierarchical_oov_codes(_batch(), rows, seen).batch_categorical
    assert torch.equal(result[0], torch.tensor([1, 1, 1]))
    assert torch.equal(result[1], torch.tensor([1, 1, 0]))
    assert torch.equal(result[2], torch.tensor([1, 0, 0]))
    assert torch.equal(result[3], torch.tensor([0, 0, 0]))


def test_control_constraint_clears_quality_control_response_without_checkpoint_change():
    routing = pd.DataFrame({"is_control": [True, False]})
    output = {
        "y_pred": torch.tensor([[3.0, 4.0], [5.0, 6.0]]).numpy(),
        "y_anchor": torch.tensor([[1.0, 2.0], [2.0, 3.0]]).numpy(),
        "delta_response": torch.tensor([[2.0, 2.0], [3.0, 3.0]]).numpy(),
    }
    original = output["y_pred"].copy()
    adjusted = enforce_routed_control_zero_response(output, routing)
    assert (adjusted["delta_response"][0] == 0).all()
    assert (adjusted["y_pred"][0] == output["y_anchor"][0]).all()
    assert (adjusted["y_pred"][1] == output["y_pred"][1]).all()
    assert (output["y_pred"] == original).all()


def test_real_validation_scenario_semantics_and_routing_counts():
    meta = load_train_val_metadata(ROOT)
    seen = fit_seen_sets_from_train_metadata(meta)
    validation = meta.loc[meta.split_final.astype(str).str.startswith("val_")]
    routing = derive_metadata_routing(validation, seen)
    routing["split_final"] = meta.loc[routing.sample_ID, "split_final"].to_numpy()
    treatment = ~routing.is_control
    expected = {
        "val_chem_only": (False, True, D0, 1065),
        "val_strain_only": (True, False, S1C, 1333),
        "val_both": (False, False, D0, 269),
        "val_time": (True, True, D2, 139),
    }
    for scenario, (chemical, strain, expert, count) in expected.items():
        part = routing.loc[treatment & routing.split_final.eq(scenario)]
        assert len(part) == count
        assert part.chemical_seen_in_train.eq(chemical).all()
        assert part.strain_seen_in_train.eq(strain).all()
        assert part.selected_expert.eq(expert).all()
    flat = routing.selected_expert.eq(S1C)
    assert routing.loc[flat, "batch_tuple_seen_in_train"].all()


def test_delivered_artifacts_are_inference_only_and_pass_all_gates():
    output = ROOT / "reports/model_v2_stage_s2c"
    metrics = json.loads((output / "novelty_gated_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    test_counts = json.loads((output / "test_metadata_routing_counts.json").read_text(encoding="utf-8"))
    routing = pd.read_csv(output / "validation_routing.csv")
    assert all(metrics["gates"].values())
    assert metrics["eligible_for_selected_checkpoint_second_seed_validation"] is True
    assert metrics["training_performed"] is False and metrics["checkpoints_modified"] is False
    assert metrics["test_protein_labels_read"] is False and metrics["test_prediction_generated"] is False
    assert manifest["checkpoints_modified"] is False and len(manifest["experts"]) == 3
    assert test_counts["protein_labels_read"] is False and test_counts["prediction_generated"] is False
    assert len(routing) == 3038
    assert not (routing.selected_expert.eq(S1C) & ~routing.batch_tuple_seen_in_train.astype(bool)).any()
    route_columns = [
        "sample_ID", "is_control", "chemical_seen_in_train", "strain_seen_in_train",
        "batch_tuple_seen_in_train", "selected_expert", "routing_reason",
    ]
    assert stable_json_sha256(routing[route_columns].sort_values("sample_ID").to_dict("records")) == metrics["routing_manifest_sha256"]
    absolute = pd.read_csv(output / "novelty_gated_absolute_metrics.csv")
    fc = pd.read_csv(output / "novelty_gated_fc_metrics.csv")
    selected = {"val_chem_only": D0, "val_strain_only": S1C, "val_both": D0, "val_time": D2}
    for scenario, expert in selected.items():
        for table, columns in (
            (absolute, ["log2_rmse", "mae", "global_r2", "sample_pcc_median", "sample_r2_median", "protein_pcc_median", "protein_r2_median"]),
            (fc, ["global_fc_pcc", "sample_fc_pcc_median", "protein_fc_pcc_median", "fc_rmse", "fc_direction_accuracy"]),
        ):
            ensemble_row = table.loc[(table.model == "S2C novelty-gated ensemble") & (table.scenario == scenario), columns].iloc[0]
            expert_row = table.loc[(table.model == expert) & (table.scenario == scenario), columns].iloc[0]
            assert ensemble_row.equals(expert_row)
    source = (ROOT / "baseline/baseline/novelty_gated_ensemble_v2.py").read_text(encoding="utf-8").lower()
    assert "proteome_raw_test" not in source
    assert "tanimoto_similarity" not in source
    assert "chemical_name ==" not in source
    assert (ROOT / "reports/MODEL_V2_STAGE_S2C_NOVELTY_GATED_ENSEMBLE_REPORT.md").is_file()
