import unittest

import numpy as np
import pandas as pd

from aivc.entity_representations import ChemicalAnchorEncoder, HashEncoder
from baseline.features import (
    build_condition_features,
    build_raw_condition_features,
    fit_feature_encoders,
)
from experiments.ablation_encoder import prepare_ablation_features


def make_fixture():
    rows = []
    values = []
    idx = []
    sample_id = 0
    strains = ["A", "B"]
    chemicals = ["Water", "DrugX"]
    for strain in strains:
        for chemical in chemicals:
            for replicate in range(2):
                sid = f"s{sample_id}"
                sample_id += 1
                idx.append(sid)
                rows.append(
                    {
                        "sample_ID": sid,
                        "split_final": "train",
                        "data_source": "WAYB",
                        "Strains": strain,
                        "Medium": "M1",
                        "Temperature": 30,
                        "pert_time": 60,
                        "perturbation_no_concentration": chemical,
                        "instrument": "O",
                        "Yeast_cell_plate": f"P{replicate}",
                    }
                )
                base = 10 if strain == "A" else 20
                effect = 0 if chemical == "Water" else 2
                values.append([base + effect + replicate, base + 1 + effect, base + 2 + effect])

    test_rows = [
        {
            "sample_ID": "unseen",
            "split_final": "val_both",
            "data_source": "WAYB",
            "Strains": "UNSEEN_STRAIN",
            "Medium": "M1",
            "Temperature": 37,
            "pert_time": 120,
            "perturbation_no_concentration": "UNSEEN_CHEM",
            "instrument": "UNSEEN_INSTRUMENT",
            "Yeast_cell_plate": "UNSEEN_PLATE",
        }
    ]
    meta = pd.DataFrame(rows + test_rows).set_index("sample_ID")
    y = pd.DataFrame(
        values + [[np.nan, np.nan, np.nan]],
        index=meta.index,
        columns=["p1", "p2", "p3"],
    )
    return meta, y


class FeaturePipelineTest(unittest.TestCase):
    def test_fixed_dimension_and_unseen_fallbacks(self):
        meta, y = make_fixture()
        train_meta = meta.loc[meta["split_final"] == "train"]
        encoders = fit_feature_encoders(train_meta, y.loc[train_meta.index])
        X = build_condition_features(meta, encoders=encoders)
        raw = build_raw_condition_features(meta, encoders)

        self.assertEqual(X.shape, (len(meta), 256))
        self.assertEqual(raw.shape[0], len(meta))
        self.assertEqual(encoders["d_emb"], 256)
        self.assertEqual(encoders["raw_dim"], raw.shape[1])

        unknown_row = raw[-1]
        start, end = encoders["feature_slices"]["strains"]
        self.assertTrue(np.allclose(unknown_row[start:end], 0.0))
        self.assertFalse(np.isnan(X).any())
        self.assertTrue(np.isfinite(X).all())

    def test_hash_is_deterministic_for_unseen_values(self):
        encoder = HashEncoder(16, seed=123)
        first = encoder.transform(["new-chemical", "new-plate"])
        second = encoder.transform(["new-chemical", "new-plate"])
        np.testing.assert_array_equal(first, second)

    def test_chemical_anchor_matches_manual_control_delta(self):
        meta, y = make_fixture()
        train_meta = meta.loc[meta["split_final"] == "train"]
        encoder = ChemicalAnchorEncoder(n_components=3).fit(
            train_meta, y.loc[train_meta.index]
        )
        np.testing.assert_allclose(encoder.lookup_["Water"], np.zeros(3))
        np.testing.assert_allclose(encoder.lookup_["DrugX"], np.full(3, 2.0))
        np.testing.assert_allclose(encoder.fallback_, np.full(3, 2.0))
        self.assertEqual(encoder.matched_samples_, 4)

    def test_statistical_encoders_are_train_only(self):
        meta, y = make_fixture()
        train_meta = meta.loc[meta["split_final"] == "train"]
        encoders = fit_feature_encoders(train_meta, y.loc[train_meta.index])
        self.assertEqual(encoders["strain_prior"].n_proteins_, 3)
        self.assertEqual(encoders["chem_anchor"].n_proteins_, 3)
        self.assertGreaterEqual(encoders["chem_anchor"].matched_samples_, 1)

    def test_all_encoder_ablations_keep_fixed_width(self):
        meta, y = make_fixture()
        results = prepare_ablation_features(meta, y)
        self.assertEqual(
            set(results),
            {"full", "no_strain_prior", "no_chem_anchor", "no_hash",
             "no_cross_features", "no_chemical_structure"},
        )
        for bundle in results.values():
            self.assertEqual(bundle["X"].shape, (len(meta), 256))
            self.assertTrue(np.isfinite(bundle["X"]).all())

    def test_fit_rejects_mixed_split_metadata(self):
        meta, y = make_fixture()
        with self.assertRaises(ValueError):
            fit_feature_encoders(meta, y)


if __name__ == "__main__":
    unittest.main()
