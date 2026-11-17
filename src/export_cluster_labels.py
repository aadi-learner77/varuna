"""
src/export_cluster_labels.py
============================
Generates per-hobli climatological cluster labels using the trained K-Means
model (k=4) and scaler saved during Stage 3 modelling.

Saves: data/features/hobli_clusters.csv
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE_DIR / "data" / "clean" / "clean_rainfall.csv"
SUSPECT_PATH = BASE_DIR / "reports" / "suspect_normals.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "data" / "features" / "hobli_clusters.csv"

CLUSTER_NAMES = {
    0: "Western Ghats & Coastal High-Rainfall Belt",
    1: "High-Surplus / Flood-Prone Interior Plains",
    2: "Stable Semi-Arid Agricultural Plains",
    3: "Deficit-Vulnerable / Chronic-Drought Pocket"
}

CLUSTER_DESCRIPTIONS = {
    0: "Heavy monsoon corridor (Western Ghats & Coastal). High volumetric rainfall with low relative percentage variance.",
    1: "Semi-arid northern and eastern plains with chronic flood surges and high percentage volatility.",
    2: "Central and southern agricultural belt. Moderately wet, predictable rainfall with low drought risk.",
    3: "Vulnerable pockets exhibiting repeated negative departures and chronic rainfall deficits."
}


def export_clusters():
    print("=" * 60)
    print("EXPORTING HOBLI CLUSTER LABELS")
    print("=" * 60)

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Clean rainfall data not found at {CLEAN_PATH}")
    if not (MODELS_DIR / "cluster_scaler.joblib").exists():
        raise FileNotFoundError("cluster_scaler.joblib not found. Run src/train_models.py first.")
    if not (MODELS_DIR / "kmeans_cluster.joblib").exists():
        raise FileNotFoundError("kmeans_cluster.joblib not found. Run src/train_models.py first.")

    clean_df = pd.read_csv(CLEAN_PATH)
    
    sus_keys = set()
    if SUSPECT_PATH.exists():
        sus_df = pd.read_csv(SUSPECT_PATH)
        sus_keys = set(zip(sus_df["district"], sus_df["taluk"], sus_df["hobli"]))

    df = clean_df.copy()
    df["sub_id"] = df.groupby(["district", "taluk", "hobli", "year"]).cumcount()
    df["station_id"] = df["district"] + "_" + df["taluk"] + "_" + df["hobli"] + "_" + df["sub_id"].astype(str)

    hobli_agg = df.groupby("station_id").agg(
        district=("district", "first"),
        taluk=("taluk", "first"),
        hobli=("hobli", "first"),
        mean_departure=("departure_pct", "mean"),
        std_departure=("departure_pct", "std"),
        deficit_count=("departure_pct", lambda s: (s < -20.0).sum()),
        excess_count=("departure_pct", lambda s: (s > 20.0).sum()),
        mean_normal_mm=("normal_mm", "mean")
    ).reset_index()

    hobli_agg["std_departure"] = hobli_agg["std_departure"].fillna(0.0)
    hobli_agg["is_suspect_normal"] = hobli_agg.apply(
        lambda r: 1 if (r["district"], r["taluk"], r["hobli"]) in sus_keys else 0, axis=1
    )

    cluster_features = ["mean_departure", "std_departure", "deficit_count", "excess_count", "mean_normal_mm"]
    scaler = joblib.load(MODELS_DIR / "cluster_scaler.joblib")
    km4 = joblib.load(MODELS_DIR / "kmeans_cluster.joblib")

    X_cluster = scaler.transform(hobli_agg[cluster_features])
    hobli_agg["cluster_id"] = km4.predict(X_cluster)
    hobli_agg["cluster_name"] = hobli_agg["cluster_id"].map(CLUSTER_NAMES)
    hobli_agg["cluster_desc"] = hobli_agg["cluster_id"].map(CLUSTER_DESCRIPTIONS)

    # Save to data/features/hobli_clusters.csv
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hobli_agg.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Exported {len(hobli_agg)} hobli cluster records to {OUTPUT_PATH}")

    print("\nCluster Distribution:")
    for cid in sorted(CLUSTER_NAMES.keys()):
        cnt = (hobli_agg["cluster_id"] == cid).sum()
        pct = (cnt / len(hobli_agg)) * 100
        print(f"  Cluster {cid}: {CLUSTER_NAMES[cid]:<45} | Count: {cnt:3d} ({pct:5.1f}%)")

    return hobli_agg


if __name__ == "__main__":
    export_clusters()

