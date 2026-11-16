"""
src/predict_risk.py
===================
Inference helper for predicting hobli-level drought and flood risk zones.

Provides `predict_risk(row)` which:
1. Validates that ALL feature columns defined in `models/feature_columns.json` are present.
2. Raises a clear, explicit ValueError naming any missing feature columns.
3. Loads the serialized top-performing model (`models/best_classifier.joblib`).
4. Returns the predicted risk zone ('drought_prone', 'stable', 'flood_prone')
   along with calibrated class probabilities.
"""

import os
import json
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import joblib

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"

# In-memory singleton caches
_MODEL = None
_FEATURE_COLS = None


def _load_artifacts():
    """Load model and feature column specification into singleton memory."""
    global _MODEL, _FEATURE_COLS
    if _FEATURE_COLS is None:
        if not FEATURE_COLS_PATH.exists():
            raise FileNotFoundError(f"Feature columns file not found at {FEATURE_COLS_PATH}. Run src/train_models.py first.")
        with open(FEATURE_COLS_PATH, "r") as f:
            _FEATURE_COLS = json.load(f)

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run src/train_models.py first.")
        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL, _FEATURE_COLS


def predict_risk(row: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Predict rainfall risk zone for a single hobli observation.

    Parameters:
    -----------
    row : dict, pd.Series, or 1-row pd.DataFrame
        Must contain ALL columns specified in models/feature_columns.json:
        ['normal_mm', 'gap_years', 'n_prior_years', 'deficit_count', 'excess_count',
         'mean_actual_mm', 'std_actual_mm', 'mean_departure_pct', 'std_departure_pct',
         'previous_year_departure', 'taluk_mean_departure', 'district_mean_departure',
         'hobli_minus_taluk']

    Returns:
    --------
    dict
        {
            "risk_zone": str,  # 'drought_prone', 'stable', or 'flood_prone'
            "probabilities": {
                "drought_prone": float,
                "stable": float,
                "flood_prone": float
            },
            "model_used": str
        }

    Raises:
    -------
    ValueError:
        If any required column from models/feature_columns.json is missing.
    """
    model, feature_cols = _load_artifacts()

    # Convert input to DataFrame row
    if isinstance(row, dict):
        df_row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df_row = pd.DataFrame([row.to_dict()])
    elif isinstance(row, pd.DataFrame):
        if len(row) == 0:
            raise ValueError("Input DataFrame is empty.")
        df_row = row.head(1).copy()
    else:
        raise ValueError(f"Unsupported row type: {type(row)}. Expected dict, pd.Series, or pd.DataFrame.")

    # Strict Validation: verify ALL required columns are present
    missing_cols = [col for col in feature_cols if col not in df_row.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns for prediction: {missing_cols}. "
            f"All {len(feature_cols)} features defined in {FEATURE_COLS_PATH.name} must be provided."
        )

    # Reorder features strictly matching training specification
    X = df_row[feature_cols].copy()

    # Predict risk zone
    pred_class = model.predict(X)[0]

    # Predict class probabilities
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        for cls_name, p in zip(classes, probs):
            probabilities[str(cls_name)] = round(float(p), 4)

    return {
        "risk_zone": str(pred_class),
        "probabilities": probabilities,
        "model_used": "RandomForestClassifier (best_classifier.joblib)"
    }


def _run_tests():
    """Verify predict_risk against a real row from features.csv and test error handling."""
    print("=" * 60)
    print("TESTING INFERENCE HELPER: src/predict_risk.py")
    print("=" * 60)

    features_path = Path("data/features/features.csv")
    if not features_path.exists():
        print(f"[SKIP] {features_path} not found. Skipping live test.")
        return

    df = pd.read_csv(features_path)
    sample_row = df[df["year"] == 2025].iloc[0]

    print(f"Testing with real 2025 observation: {sample_row['district']} - {sample_row['taluk']} - {sample_row['hobli']}")
    result = predict_risk(sample_row)

    print("\n[OK] Prediction Result:")
    print(f"  Predicted Risk Zone: {result['risk_zone']}")
    print(f"  Class Probabilities: {result['probabilities']}")
    print(f"  Model Used:         {result['model_used']}")

    assert result["risk_zone"] in ["drought_prone", "stable", "flood_prone"], "Invalid risk zone returned!"
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0: {prob_sum}"

    # Test error handling when required feature is missing
    print("\nTesting missing column validation...")
    incomplete_row = sample_row.to_dict()
    del incomplete_row["previous_year_departure"]
    try:
        predict_risk(incomplete_row)
        raise AssertionError("Validation failed: did not raise ValueError on missing column!")
    except ValueError as e:
        print(f"[OK] Correctly caught expected error: {e}")

    print("\nAll src/predict_risk.py tests passed successfully!")


if __name__ == "__main__":
    _run_tests()
