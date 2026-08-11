import importlib.util
import unittest

import numpy as np
import pandas as pd

from baseline.features import build_condition_features, fit_feature_encoders


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the AIVC model contract test")
class PersonAtoBContractTest(unittest.TestCase):
    def test_aivc_model_accepts_person_a_256d_features(self):
        import torch
        from aivc.model import AIVCModel

        rows = []
        values = []
        for i, chemical in enumerate(["Water", "DrugX", "Water", "DrugX"]):
            rows.append(
                {
                    "sample_ID": f"s{i}",
                    "split_final": "train",
                    "data_source": "WAYB",
                    "Strains": "A" if i < 2 else "B",
                    "Medium": "M1",
                    "Temperature": 30,
                    "pert_time": 60,
                    "perturbation_no_concentration": chemical,
                    "instrument": "O",
                    "Yeast_cell_plate": f"P{i % 2}",
                }
            )
            values.append([10.0 + i, 11.0 + i, 12.0 + i])
        meta = pd.DataFrame(rows).set_index("sample_ID")
        y = pd.DataFrame(values, index=meta.index, columns=["p1", "p2", "p3"])

        encoders = fit_feature_encoders(meta, y)
        features = build_condition_features(meta, encoders=encoders)
        self.assertEqual(features.shape, (4, 256))

        model = AIVCModel(dim_in=features.shape[1], n_proteins=3, use_gnn=False)
        output = model(torch.from_numpy(features))
        self.assertEqual(output["y_pred"].shape, (4, 3))
        self.assertEqual(output["delta_context"].shape, (4, 3))
        self.assertTrue(torch.isfinite(output["y_pred"]).all())


if __name__ == "__main__":
    unittest.main()
