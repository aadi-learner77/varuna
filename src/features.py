"""
src/features.py
===============
Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka.

This module engineers historical, contextual, and geographic deviation features
from clean rainfall data and constructs the 3-class target variable `risk_zone`.

Inputs:
    data/clean/clean_rainfall.csv

Outputs:
    data/features/features.csv
    data/features/README.md

Data Leakage Prevention:
    All historical features (deficit_count, excess_count, mean_actual_mm,
    std_actual_mm, std_departure_pct, previous_year_departure) are calculated
    strictly using earlier observed years (years < Y). In the initial year (2020),
    prior historical features evaluate to NaN or 0 count.
    Within-year spatial context features (taluk_mean_departure, district_mean_departure,
    hobli_minus_taluk) capture contemporaneous regional micro-climatic patterns.
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# THRESHOLDS & CONSTANTS
# ==============================================================================
# Target Classification Thresholds (KSNDMC / IMD monsoon departure standards):
#   - drought_prone : departure_pct <= DROUGHT_THRESHOLD (-20.0% or lower)
#   - flood_prone   : departure_pct >= FLOOD_THRESHOLD   (+40.0% or higher)
#   - stable        : -20.0% < departure_pct < +40.0%
DROUGHT_THRESHOLD = -20.0
FLOOD_THRESHOLD = 40.0

# Threshold for historical mild excess rainfall (+20% departure):
EXCESS_THRESHOLD = 20.0

INPUT_PATH = "data/clean/clean_rainfall.csv"
OUTPUT_DIR = "data/features"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "features.csv")


def load_clean_data(path: str = INPUT_PATH) -> pd.DataFrame:
    """Load cleaned rainfall dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} records across {df['year'].nunique()} years "
          f"({sorted(df['year'].unique())}) from {path}")
    return df


def create_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a unique, persistent tracking key for each geographic reporting station.
    
    In Karnataka KSNDMC records:
    1. Vijayanagar district was carved out of Ballari in late 2021. In 2022 raw data,
       these taluks were temporarily reported under Ballari. We harmonize them to 'VIJAYANAGAR'.
    2. Certain taluks contain multiple stations with the same hobli name (e.g. Sarjapura_1 in Anekal).
       We track cumulative station occurrence to maintain exact historical station continuity.
    """
    df = df.copy()
    
    # 1. Harmonize Ballari / Vijayanagar taluks in 2022
    vijayanagar_taluks = [
        "Hagaribommanahalli", "Hadagali", "Kotturu",
        "Harappanahalli", "Kudligi", "Hosapete"
    ]
    df["district_canon"] = df["district"]
    mask_vj = (df["year"] == 2022) & (df["district"] == "BALLARI") & (df["taluk"].isin(vijayanagar_taluks))
    df.loc[mask_vj, "district_canon"] = "VIJAYANAGAR"

    # 2. Assign station sub-index for multi-station hoblis within a single year
    df["station_sub_id"] = df.groupby(["district_canon", "hobli", "year"]).cumcount()

    # 3. Unique persistent station ID across years
    df["station_id"] = (
        df["district_canon"] + "_" + df["hobli"] + "_" + df["station_sub_id"].astype(str)
    )
    return df


def assign_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the 3-class target variable `risk_zone` based on annual rainfall departure.
    
    Rule:
      - drought_prone : departure_pct <= -20.0%
      - flood_prone   : departure_pct >= +40.0%
      - stable        : -20.0% < departure_pct < +40.0%
    """
    conditions = [
        df["departure_pct"] <= DROUGHT_THRESHOLD,
        df["departure_pct"] >= FLOOD_THRESHOLD,
    ]
    choices = ["drought_prone", "flood_prone"]
    df["risk_zone"] = np.select(conditions, choices, default="stable")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer leak-free historical features and cross-sectional contextual features.
    
    Historical Features (Strictly earlier years: year < Y):
      - deficit_count: count of earlier years where departure_pct < -20.0%
      - excess_count: count of earlier years where departure_pct > +20.0%
      - mean_actual_mm: expanding mean of actual_mm over earlier years
      - std_actual_mm: expanding sample std of actual_mm over earlier years (ddof=1)
      - std_departure_pct: expanding sample std of departure_pct over earlier years (ddof=1)
      - previous_year_departure: lag-1 departure_pct from previous observed year
      
    Contextual Spatial Features (Contemporaneous cross-sectional):
      - taluk_mean_departure: mean departure_pct across all hoblis in that taluk and year
      - district_mean_departure: mean departure_pct across all hoblis in that district and year
      - hobli_minus_taluk: departure_pct - taluk_mean_departure
    """
    df = df.copy()

    # Ensure chronological order per persistent station
    df = df.sort_values(["station_id", "year"]).reset_index(drop=True)
    grp = df.groupby("station_id", sort=False)

    # 1. Deficit count (earlier years with departure < -20%)
    df["deficit_count"] = grp["departure_pct"].transform(
        lambda s: (s < DROUGHT_THRESHOLD).astype(int).expanding().sum().shift(1).fillna(0).astype(int)
    )

    # 2. Excess count (earlier years with departure > +20%)
    df["excess_count"] = grp["departure_pct"].transform(
        lambda s: (s > EXCESS_THRESHOLD).astype(int).expanding().sum().shift(1).fillna(0).astype(int)
    )

    # 3. Expanding mean of actual_mm (earlier years only)
    df["mean_actual_mm"] = grp["actual_mm"].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    # 4. Expanding sample standard deviation of actual_mm (earlier years only, min_periods=2)
    df["std_actual_mm"] = grp["actual_mm"].transform(
        lambda s: s.expanding().std().shift(1)
    )

    # 5. Expanding sample standard deviation of departure_pct (earlier years only, min_periods=2)
    df["std_departure_pct"] = grp["departure_pct"].transform(
        lambda s: s.expanding().std().shift(1)
    )

    # 6. Previous observed year departure (lag-1)
    df["previous_year_departure"] = grp["departure_pct"].shift(1)

    # 7. Taluk-level mean departure for the current year
    taluk_mean = df.groupby(["taluk", "year"])["departure_pct"].transform("mean")
    df["taluk_mean_departure"] = taluk_mean.round(2)

    # 8. District-level mean departure for the current year
    dist_mean = df.groupby(["district", "year"])["departure_pct"].transform("mean")
    df["district_mean_departure"] = dist_mean.round(2)

    # 9. Hobli departure minus taluk average
    df["hobli_minus_taluk"] = (df["departure_pct"] - taluk_mean).round(2)

    return df


def verify_no_leakage(df: pd.DataFrame) -> bool:
    """Verify that no current or future year data leaked into historical lag features."""
    min_year = df.groupby("station_id")["year"].transform("min")
    first_year_rows = df["year"] == min_year
    
    # In the very first observation of each station, historical values must be NaN / zero
    no_lag_leak = df.loc[first_year_rows, "previous_year_departure"].isna().all()
    no_mean_leak = df.loc[first_year_rows, "mean_actual_mm"].isna().all()
    no_std_leak = df.loc[first_year_rows, "std_actual_mm"].isna().all()
    no_deficit_leak = (df.loc[first_year_rows, "deficit_count"] == 0).all()
    no_excess_leak = (df.loc[first_year_rows, "excess_count"] == 0).all()
    
    is_valid = all([no_lag_leak, no_mean_leak, no_std_leak, no_deficit_leak, no_excess_leak])
    return is_valid


def print_report(df: pd.DataFrame) -> None:
    """Print feature statistics, class balance, and verification numbers."""
    total = len(df)
    counts = df["risk_zone"].value_counts()
    
    print("\n" + "=" * 60)
    print("TARGET CLASS BALANCE (`risk_zone`)")
    print("=" * 60)
    print(f"Rule: drought_prone (<= {DROUGHT_THRESHOLD}%), flood_prone (>= +{FLOOD_THRESHOLD}%), stable (otherwise)")
    print(f"{'Class':<16} {'Count':>8} {'Share':>10}")
    print("-" * 38)
    for cls in ["stable", "flood_prone", "drought_prone"]:
        cnt = counts.get(cls, 0)
        pct = (cnt / total) * 100
        print(f"{cls:<16} {cnt:>8,} {pct:>9.2f}%")
    print("-" * 38)
    print(f"{'Total':<16} {total:>8,} {'100.00%':>10}")

    print("\n" + "=" * 60)
    print("ENGINEERED FEATURES SUMMARY")
    print("=" * 60)
    feature_cols = [
        "deficit_count", "excess_count", "mean_actual_mm", "std_actual_mm",
        "std_departure_pct", "previous_year_departure",
        "taluk_mean_departure", "district_mean_departure", "hobli_minus_taluk"
    ]
    for col in feature_cols:
        null_cnt = df[col].isna().sum()
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col:<25} | NaNs: {null_cnt:>5} | Mean: {mean_val:>8.2f} | Std: {std_val:>8.2f} | Range: [{min_val:>7.2f}, {max_val:>7.2f}]")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load clean data
    raw_clean = load_clean_data(INPUT_PATH)
    
    # 2. Persistent station identification
    data_with_keys = create_entity_key(raw_clean)
    
    # 3. Add target risk_zone
    data_with_target = assign_target(data_with_keys)
    
    # 4. Feature engineering
    featured = engineer_features(data_with_target)
    
    # 5. Leakage audit
    leak_check = verify_no_leakage(featured)
    print(f"\nAudit: Data Leakage Verification Passed? {leak_check}")
    assert leak_check, "Data leakage detected: first-year records contain non-null lag values!"

    # 6. Reorder final columns strictly according to the contract
    final_columns = [
        "district", "taluk", "hobli", "year",
        "normal_mm", "actual_mm", "departure_pct",
        "deficit_count", "excess_count",
        "mean_actual_mm", "std_actual_mm", "std_departure_pct",
        "previous_year_departure",
        "taluk_mean_departure", "district_mean_departure",
        "hobli_minus_taluk",
        "risk_zone"
    ]
    final_df = featured[final_columns].copy()

    # 7. Save output
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved final features to {OUTPUT_PATH} ({len(final_df):,} rows x {len(final_df.columns)} columns)")

    # 8. Print audit & diagnostic reports
    print_report(final_df)


if __name__ == "__main__":
    main()
