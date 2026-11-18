"""
src/predict_future.py
=====================
Forecasting & Predictive Analysis for Karnataka Hoblis (2026 and 2027).

Provides functions to:
1. Roll forward historical data (2020-2025) to construct leak-free ex-ante features for 2026.
2. Run trained regression & classification models to forecast rainfall (mm), percentage departure,
   and drought/flood risk zones with calibrated class probabilities.
3. Recursively roll forward the 2026 prediction to forecast 2027, applying multi-step uncertainty bounds.
4. Batch-generate projections for all 850+ hoblis across Karnataka.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CLEAN_DATA_PATH = DATA_DIR / "clean" / "clean_rainfall.csv"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
FUTURE_PROJECTIONS_PATH = DATA_DIR / "features" / "future_projections_2026_2027.csv"

# Empirical error margins derived from 2025 out-of-time evaluation
BASE_MAE_PCT = 24.4  # Model test MAE for departure percentage
UNCERTAINTY_MULTIPLIERS = {
    2026: 1.0,  # 1-step ahead: ~24.4% margin
    2027: 1.35  # 2-step ahead compounded error: ~33.0% margin
}

# Singleton model cache
_MODELS_CACHE = {}


def load_prediction_artifacts():
    """Load serialized models and feature columns list into cache."""
    if "loaded" not in _MODELS_CACHE:
        if not FEATURE_COLS_PATH.exists():
            raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLS_PATH}")
        with open(FEATURE_COLS_PATH, "r") as f:
            _MODELS_CACHE["feature_cols"] = json.load(f)

        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")
        _MODELS_CACHE["classifier"] = joblib.load(CLASSIFIER_PATH)

        if not REGRESSOR_PATH.exists():
            raise FileNotFoundError(f"Regressor not found: {REGRESSOR_PATH}")
        _MODELS_CACHE["regressor"] = joblib.load(REGRESSOR_PATH)

        _MODELS_CACHE["loaded"] = True

    return (
        _MODELS_CACHE["classifier"],
        _MODELS_CACHE["regressor"],
        _MODELS_CACHE["feature_cols"]
    )


def project_station_features(
    history_df: pd.DataFrame,
    target_year: int,
    taluk_mean_dep: float,
    district_mean_dep: float
) -> Dict[str, Any]:
    """
    Construct the ex-ante feature dictionary for a single station for target_year,
    using all rows in history_df (which must contain observations up to target_year - 1).
    """
    sorted_hist = history_df.sort_values("year").reset_index(drop=True)
    last_row = sorted_hist.iloc[-1]
    last_year = int(last_row["year"])

    normal_mm = float(last_row["normal_mm"])
    gap_years = target_year - last_year
    n_prior_years = len(sorted_hist)

    # Historical departure counts
    deps = sorted_hist["departure_pct"].values
    acts = sorted_hist["actual_mm"].values

    deficit_count = int(np.sum(deps <= -20.0))
    excess_count = int(np.sum(deps >= 20.0))

    mean_actual_mm = round(float(np.mean(acts)), 2)
    std_actual_mm = round(float(np.std(acts, ddof=1)), 2) if len(acts) > 1 else np.nan

    mean_departure_pct = round(float(np.mean(deps)), 2)
    std_departure_pct = round(float(np.std(deps, ddof=1)), 2) if len(deps) > 1 else np.nan

    previous_year_departure = round(float(deps[-1]), 2)
    hobli_minus_taluk = round(mean_departure_pct - taluk_mean_dep, 2)

    return {
        "normal_mm": normal_mm,
        "gap_years": gap_years,
        "n_prior_years": n_prior_years,
        "deficit_count": deficit_count,
        "excess_count": excess_count,
        "mean_actual_mm": mean_actual_mm,
        "std_actual_mm": std_actual_mm,
        "mean_departure_pct": mean_departure_pct,
        "std_departure_pct": std_departure_pct,
        "previous_year_departure": previous_year_departure,
        "taluk_mean_departure": taluk_mean_dep,
        "district_mean_departure": district_mean_dep,
        "hobli_minus_taluk": hobli_minus_taluk,
    }


def predict_future_for_station(
    clean_df: pd.DataFrame,
    district: str,
    taluk: str,
    hobli: str,
    target_years: List[int] = [2026, 2027]
) -> List[Dict[str, Any]]:
    """
    Generate sequential multi-year predictions for a specific hobli across target_years.
    Handles recursive feature updating from year T to year T+1.
    """
    classifier, regressor, feature_cols = load_prediction_artifacts()

    # Pre-calculate overall taluk and district historical departure means
    hist_all = clean_df.copy()
    taluk_mean_dep = float(
        hist_all[hist_all["taluk"] == taluk]["departure_pct"].mean()
        if len(hist_all[hist_all["taluk"] == taluk]) > 0
        else hist_all["departure_pct"].mean()
    )
    dist_mean_dep = float(
        hist_all[hist_all["district"] == district]["departure_pct"].mean()
        if len(hist_all[hist_all["district"] == district]) > 0
        else hist_all["departure_pct"].mean()
    )

    # Station history
    st_hist = clean_df[
        (clean_df["district"] == district) &
        (clean_df["taluk"] == taluk) &
        (clean_df["hobli"] == hobli)
    ].copy()

    if len(st_hist) == 0:
        return []

    results = []
    current_hist = st_hist.copy()

    for yr in sorted(target_years):
        feat_dict = project_station_features(current_hist, yr, taluk_mean_dep, dist_mean_dep)
        row_df = pd.DataFrame([feat_dict])[feature_cols]

        # 1. Regress expected departure %
        pred_dep_pct = float(regressor.predict(row_df)[0])
        normal_mm = feat_dict["normal_mm"]
        pred_actual_mm = max(0.0, normal_mm * (1.0 + pred_dep_pct / 100.0))

        # 2. Uncertainty bounds
        multiplier = UNCERTAINTY_MULTIPLIERS.get(yr, 1.5)
        margin_pct = BASE_MAE_PCT * multiplier
        lower_dep = pred_dep_pct - margin_pct
        upper_dep = pred_dep_pct + margin_pct
        lower_mm = max(0.0, normal_mm * (1.0 + lower_dep / 100.0))
        upper_mm = max(0.0, normal_mm * (1.0 + upper_dep / 100.0))

        # 3. Classify Risk Zone & Calibrated Probabilities
        probs = classifier.predict_proba(row_df)[0]
        classes = classifier.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        pred_zone = str(classifier.predict(row_df)[0])

        rec = {
            "district": district,
            "taluk": taluk,
            "hobli": hobli,
            "year": yr,
            "normal_mm": round(normal_mm, 2),
            "predicted_actual_mm": round(pred_actual_mm, 2),
            "lower_actual_mm": round(lower_mm, 2),
            "upper_actual_mm": round(upper_mm, 2),
            "predicted_departure_pct": round(pred_dep_pct, 2),
            "lower_departure_pct": round(lower_dep, 2),
            "upper_departure_pct": round(upper_dep, 2),
            "risk_zone": pred_zone,
            "prob_drought_prone": round(prob_dict.get("drought_prone", 0.0), 3),
            "prob_stable": round(prob_dict.get("stable", 0.0), 3),
            "prob_flood_prone": round(prob_dict.get("flood_prone", 0.0), 3),
            "uncertainty_margin_pct": round(margin_pct, 1),
        }
        results.append(rec)

        # Append recursive row to current_hist for next year
        next_row = {
            "district": district,
            "taluk": taluk,
            "hobli": hobli,
            "year": yr,
            "normal_mm": normal_mm,
            "actual_mm": pred_actual_mm,
            "departure_pct": pred_dep_pct,
        }
        current_hist = pd.concat([current_hist, pd.DataFrame([next_row])], ignore_index=True)

    return results


def generate_all_future_projections(
    clean_csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate and persist 2026 & 2027 forecasts for all unique stations in Karnataka.
    """
    clean_csv = clean_csv_path or CLEAN_DATA_PATH
    out_file = output_path or FUTURE_PROJECTIONS_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df_clean = pd.read_csv(clean_csv)
    stations = df_clean[["district", "taluk", "hobli"]].drop_duplicates().reset_index(drop=True)

    print(f"Generating 2026-2027 forecasts for {len(stations)} Karnataka stations...")
    all_records = []

    for idx, st_row in stations.iterrows():
        d, t, h = st_row["district"], st_row["taluk"], st_row["hobli"]
        st_preds = predict_future_for_station(df_clean, d, t, h, target_years=[2026, 2027])
        all_records.extend(st_preds)

    df_future = pd.DataFrame(all_records)
    df_future.to_csv(out_file, index=False)
    print(f"[OK] Generated {len(df_future)} projections and saved to {out_file}")
    return df_future


if __name__ == "__main__":
    generate_all_future_projections()
