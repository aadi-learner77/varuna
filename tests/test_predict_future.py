"""
tests/test_predict_future.py
============================
Automated verification tests for 2026-2027 future rainfall prediction engine.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.predict_future import (
    load_prediction_artifacts,
    predict_future_for_station,
    FUTURE_PROJECTIONS_PATH,
    CLEAN_DATA_PATH
)


class TestFuturePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_df = pd.read_csv(CLEAN_DATA_PATH)
        cls.classifier, cls.regressor, cls.feature_cols = load_prediction_artifacts()

    def test_artifacts_loaded(self):
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.regressor)
        self.assertGreater(len(self.feature_cols), 5)

    def test_station_prediction_shape_and_values(self):
        # Pick first available station
        first_row = self.clean_df.iloc[0]
        d, t, h = first_row["district"], first_row["taluk"], first_row["hobli"]

        preds = predict_future_for_station(self.clean_df, d, t, h, target_years=[2026, 2027])
        self.assertEqual(len(preds), 2)

        p2026, p2027 = preds[0], preds[1]
        self.assertEqual(p2026["year"], 2026)
        self.assertEqual(p2027["year"], 2027)

        # Check physical bounds
        self.assertGreaterEqual(p2026["predicted_actual_mm"], 0.0)
        self.assertGreaterEqual(p2027["predicted_actual_mm"], 0.0)
        self.assertLessEqual(p2026["lower_actual_mm"], p2026["upper_actual_mm"])
        self.assertLessEqual(p2027["lower_actual_mm"], p2027["upper_actual_mm"])

        # Check probability distributions sum to ~1.0
        prob_sum_26 = (
            p2026["prob_drought_prone"] + p2026["prob_stable"] + p2026["prob_flood_prone"]
        )
        self.assertAlmostEqual(prob_sum_26, 1.0, places=2)

    def test_precomputed_csv_exists_and_populated(self):
        self.assertTrue(FUTURE_PROJECTIONS_PATH.exists())
        df = pd.read_csv(FUTURE_PROJECTIONS_PATH)
        self.assertGreater(len(df), 1500)
        self.assertIn("predicted_actual_mm", df.columns)
        self.assertIn("risk_zone", df.columns)
        self.assertTrue(set(df["year"].unique()) == {2026, 2027})


if __name__ == "__main__":
    unittest.main()
