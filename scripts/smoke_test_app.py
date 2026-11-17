"""
scripts/smoke_test_app.py
=========================
Smoke test script for verifying app/app.py functionality across 3 distinct hoblis,
including at least one suspect-normal station.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.predict_risk import predict_risk


def run_smoke_test():
    print("=" * 70)
    print("RUNNING HEADLESS SMOKE TEST FOR APP LOGIC")
    print("=" * 70)

    # 1. Test data files loading
    clean_path = ROOT_DIR / "data" / "clean" / "clean_rainfall.csv"
    features_path = ROOT_DIR / "data" / "features" / "features.csv"
    clusters_path = ROOT_DIR / "data" / "features" / "hobli_clusters.csv"
    suspect_path = ROOT_DIR / "reports" / "suspect_normals.csv"
    results_path = ROOT_DIR / "models" / "results.csv"

    for p in [clean_path, features_path, clusters_path, suspect_path, results_path]:
        assert p.exists(), f"File does not exist: {p}"
        print(f"[OK] File verified: {p.name}")

    df_clean = pd.read_csv(clean_path)
    df_feat = pd.read_csv(features_path)
    df_clusters = pd.read_csv(clusters_path)
    df_suspect = pd.read_csv(suspect_path)
    df_results = pd.read_csv(results_path)

    suspect_keys = set(zip(df_suspect["district"], df_suspect["taluk"], df_suspect["hobli"]))

    # Test stations to audit:
    # 1. Agumbe (Tirthahalli, Shivamogga) - Classic suspect normal outlier
    # 2. Kulgeri (Badami, Bagalkote) - Suspect normal in northern dry plains
    # 3. Kasaba (Mandya, Mandya) - Regular agricultural station
    test_stations = [
        {"district": "SHIVAMOGGA", "taluk": "Tirthahalli", "hobli": "Agumbe"},
        {"district": "BAGALKOTE", "taluk": "Badami", "hobli": "Kulgeri"},
        {"district": "MANDYA", "taluk": "Krishnarajapet", "hobli": "Krishnarajapet"}
    ]

    for i, stn in enumerate(test_stations, 1):
        d = stn["district"]
        t = stn["taluk"]
        h = stn["hobli"]

        print(f"\n--- Smoke Test Station #{i}: {h} ({t}, {d}) ---")

        # Clean rainfall records
        clean_sub = df_clean[(df_clean["district"] == d) & (df_clean["taluk"] == t) & (df_clean["hobli"] == h)]
        print(f"  Historical Clean Records: {len(clean_sub)} years found ({sorted(clean_sub['year'].tolist())})")
        assert len(clean_sub) >= 4, f"Insufficient records for {h}"

        # Suspect normal check
        is_sus = (d, t, h) in suspect_keys
        print(f"  Suspect Normal Flag: {is_sus}")
        if is_sus:
            sus_row = df_suspect[(df_suspect["district"] == d) & (df_suspect["taluk"] == t) & (df_suspect["hobli"] == h)].iloc[0]
            print(f"  -> Suspect details: Normal={sus_row['normal_mm']:.1f} mm, Direction={sus_row['direction']}, Mean Dep={sus_row['mean_departure_pct']:+.2f}%")

        # Cluster assignment
        cl_sub = df_clusters[(df_clusters["district"] == d) & (df_clusters["taluk"] == t) & (df_clusters["hobli"] == h)]
        assert len(cl_sub) > 0, f"No cluster record for {h}"
        cl_row = cl_sub.iloc[0]
        print(f"  Cluster Assignment: Cluster {cl_row['cluster_id']} - {cl_row['cluster_name']}")
        print(f"  -> 5-Yr Mean Dep: {cl_row['mean_departure']:+.2f}%, Deficit Yrs: {cl_row['deficit_count']}, Excess Yrs: {cl_row['excess_count']}")

        # 2025 Model prediction
        feat_2025 = df_feat[(df_feat["district"] == d) & (df_feat["taluk"] == t) & (df_feat["hobli"] == h) & (df_feat["year"] == 2025)]
        assert len(feat_2025) > 0, f"No 2025 feature record for {h}"
        row_2025 = feat_2025.iloc[0]

        pred_res = predict_risk(row_2025)
        print(f"  2025 Prediction: {pred_res['risk_zone']}")
        print(f"  Class Probabilities: {pred_res['probabilities']}")
        assert pred_res["risk_zone"] in ["drought_prone", "stable", "flood_prone"]
        prob_sum = sum(pred_res["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0: {prob_sum}"

        # Spatial comparison (hobli vs taluk vs district for 2025)
        h_dep = clean_sub[clean_sub["year"] == 2025].iloc[0]["departure_pct"]
        t_dep = df_clean[(df_clean["district"] == d) & (df_clean["taluk"] == t) & (df_clean["year"] == 2025)]["departure_pct"].mean()
        d_dep = df_clean[(df_clean["district"] == d) & (df_clean["year"] == 2025)]["departure_pct"].mean()
        print(f"  2025 Departure: Hobli = {h_dep:+.2f}% | Taluk Mean = {t_dep:+.2f}% | District Mean = {d_dep:+.2f}%")

    print("\n" + "=" * 70)
    print("ALL 3 SMOKE TEST STATIONS PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
