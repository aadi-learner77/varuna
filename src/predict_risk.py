"""
src/predict_risk.py
===================
Inference helper for predicting hobli-level drought and flood risk zones.

Provides `predict_risk(row)` which loads the top-performing model
(Gradient Boosting Classifier trained on Karnataka rainfall data)
and returns the predicted risk zone and associated class probabilities.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, Any

# Paths to serialized model and metadata
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Lazy-loaded singleton references
_MODEL = None
_FEATURE_COLS = None


def _load_model_and_features():
    """Load model and feature list once into memory."""
    global _MODEL, _FEATURE_COLS
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train_models.py first.")
        _MODEL = joblib.load(MODEL_PATH)
    if _FEATURE_COLS is None:
        if not os.path.exists(FEATURE_COLS_PATH):
            raise FileNotFoundError(f"Feature columns file not found at {FEATURE_COLS_PATH}.")
        with open(FEATURE_COLS_PATH, "r") as f:
            _FEATURE_COLS = json.load(f)
    return _MODEL, _FEATURE_COLS


def predict_risk(row: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Predict risk zone for a single hobli observation.

    Parameters:
    -----------
    row : dict, pd.Series, or 1-row pd.DataFrame
        Input data containing the required features:
        - normal_mm (float)
        - deficit_count (int / float)
        - excess_count (int / float)
        - mean_actual_mm (float, optional / NaN if first year)
        - std_actual_mm (float, optional / NaN if <2 years)
        - std_departure_pct (float, optional / NaN if <2 years)
        - previous_year_departure (float, optional / NaN if first year)

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
    """
    model, feature_cols = _load_model_and_features()

    # Convert input to DataFrame
    if isinstance(row, dict):
        df_row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df_row = pd.DataFrame([row.to_dict()])
    elif isinstance(row, pd.DataFrame):
        df_row = row.copy()
    else:
        raise ValueError(f"Unsupported row type: {type(row)}. Expected dict, pd.Series, or pd.DataFrame.")

    # Ensure all required features are present; missing ones will be NaN and handled by pipeline imputer
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = np.nan

    # Extract feature matrix in exact column order
    X = df_row[feature_cols].copy()

    # Model inference
    prediction = model.predict(X)[0]
    
    # Class probabilities if supported
    probs = {}
    if hasattr(model, "predict_proba"):
        prob_values = model.predict_proba(X)[0]
        classes = model.classes_
        for cls_name, prob in zip(classes, prob_values):
            probs[cls_name] = round(float(prob), 4)

    return {
        "risk_zone": str(prediction),
        "probabilities": probs,
        "model_used": "GradientBoostingClassifier"
    }


if __name__ == "__main__":
    # Test inference on a sample observation
    sample_row = {
        "normal_mm": 800.0,
        "deficit_count": 0,
        "excess_count": 2,
        "mean_actual_mm": 950.0,
        "std_actual_mm": 120.0,
        "std_departure_pct": 15.0,
        "previous_year_departure": 25.0
    }
    result = predict_risk(sample_row)
    print("Test inference successful:")
    print(json.dumps(result, indent=2))

