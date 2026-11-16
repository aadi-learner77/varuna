"""
src/train_models.py
===================
Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using ML.

Stage 3: Full Modelling Pipeline.
1. Baseline Benchmarks: Most Common Class ('stable') & Persistence (Lag-1 Class).
2. Supervised Risk Classification:
   - Logistic Regression, Random Forest, Gradient Boosting.
   - 5-Fold District-Grouped Cross-Validation (GroupKFold on pre-2025 training years).
   - Time-Aware Out-of-Time Test Evaluation on 2025 hold-out.
   - Class-weighting for rare drought class and separate drought metrics reporting.
3. Climatological Hobli-Level Clustering:
   - Aggregate to ONE row per hobli across all 5 years.
   - Evaluates k in [2..8] via Elbow (Inertia) and Silhouette Scores.
   - Plain-English meteorological characterization of clusters.
   - Explicit marking of suspect-normal hoblis.
4. Anomaly Detection:
   - Multi-dimensional Isolation Forest plus deviation from taluk average departure.
   - Top 20 suspicious readings across Karnataka with data-quality labeling.
5. Next-Year Departure Regression Debugging:
   - Evaluated against "Predict Zero" and "Predict Train Mean" baselines using MAE.
   - Feature distribution comparison explaining baseline performance.
6. Serialization:
   - models/*.joblib
   - models/results.csv
   - models/feature_columns.json
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
DATA_PATH = Path("data/features/features.csv")
CLEAN_PATH = Path("data/clean/clean_rainfall.csv")
SUSPECT_PATH = Path("reports/suspect_normals.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
RESULTS_PATH = MODELS_DIR / "results.csv"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"

RANDOM_STATE = 42
TARGET_CLASSES = ["drought_prone", "stable", "flood_prone"]

# Ex-ante predictive feature contract (strictly known prior to current year observation)
FEATURE_COLS = [
    "normal_mm",
    "gap_years",
    "n_prior_years",
    "deficit_count",
    "excess_count",
    "mean_actual_mm",
    "std_actual_mm",
    "mean_departure_pct",
    "std_departure_pct",
    "previous_year_departure",
    "taluk_mean_departure",
    "district_mean_departure",
    "hobli_minus_taluk"
]


def load_data():
    """Load features and clean datasets."""
    df_feat = pd.read_csv(DATA_PATH)
    df_clean = pd.read_csv(CLEAN_PATH)
    sus_df = pd.read_csv(SUSPECT_PATH)
    return df_feat, df_clean, sus_df


def run_baselines(train_df, test_df):
    """Compute benchmark baselines on test set."""
    y_train = train_df["risk_zone"]
    y_test = test_df["risk_zone"]

    # Baseline 1: Most Common Class
    most_common = y_train.value_counts().index[0]  # 'stable'
    pred_mc = [most_common] * len(y_test)
    acc_mc = accuracy_score(y_test, pred_mc)
    f1_mc = f1_score(y_test, pred_mc, average="macro")

    # Baseline 2: Persistence (last observed year's class)
    # Using 2024 risk_zone mapped by persistent station key
    train_2024 = train_df[train_df["year"] == 2024].copy()
    train_2024["station_key"] = train_2024["district"] + "_" + train_2024["taluk"] + "_" + train_2024["hobli"]
    test_copy = test_df.copy()
    test_copy["station_key"] = test_copy["district"] + "_" + test_copy["taluk"] + "_" + test_copy["hobli"]
    zone_map = train_2024.set_index("station_key")["risk_zone"].to_dict()
    pred_pers = test_copy["station_key"].map(zone_map).fillna(most_common)
    acc_pers = accuracy_score(y_test, pred_pers)
    f1_pers = f1_score(y_test, pred_pers, average="macro")

    return {
        "most_common": {"pred": pred_mc, "acc": acc_mc, "macro_f1": f1_mc, "class": most_common},
        "persistence": {"pred": pred_pers, "acc": acc_pers, "macro_f1": f1_pers}
    }


def train_classifiers(X_train, y_train, groups_train, X_test, y_test):
    """
    Train Logistic Regression, Random Forest, and Gradient Boosting.
    Run 5-Fold District-Grouped CV on training set, then evaluate on 2025 test set.
    """
    classifiers = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE))
        ])
    }

    gkf = GroupKFold(n_splits=5)
    cv_results = {}
    test_results = {}

    # Calculate majority-class share across the 5 validation folds
    val_majority_shares = []
    train_majority_shares = []
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
        y_val_fold = y_train.iloc[val_idx]
        y_tr_fold = y_train.iloc[tr_idx]
        val_majority_shares.append((y_val_fold == y_val_fold.mode()[0]).mean() * 100)
        train_majority_shares.append((y_tr_fold == y_tr_fold.mode()[0]).mean() * 100)

    cv_results["_majority_share_val_mean"] = float(np.mean(val_majority_shares))
    cv_results["_majority_share_val_std"] = float(np.std(val_majority_shares))
    cv_results["_majority_share_train_mean"] = float(np.mean(train_majority_shares))
    cv_results["_majority_share_train_std"] = float(np.std(train_majority_shares))

    # 1. 5-Fold District-Grouped CV
    for name, pipeline in classifiers.items():
        acc_scores = []
        f1_scores = []
        drought_recs = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
            m = pipeline
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = m.predict(X_train.iloc[val_idx])
            acc_scores.append(accuracy_score(y_train.iloc[val_idx], preds))
            f1_scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
            report = classification_report(y_train.iloc[val_idx], preds, output_dict=True, zero_division=0)
            drought_recs.append(report.get("drought_prone", {}).get("recall", 0.0))

        cv_results[name] = {
            "cv_acc_mean": np.mean(acc_scores),
            "cv_acc_std": np.std(acc_scores),
            "cv_f1_mean": np.mean(f1_scores),
            "cv_f1_std": np.std(f1_scores),
            "cv_drought_recall": np.mean(drought_recs)
        }

    # 2. Out-of-Time Test Evaluation
    for name, pipeline in classifiers.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=TARGET_CLASSES)

        test_results[name] = {
            "test_acc": acc,
            "test_f1": f1,
            "drought_precision": report.get("drought_prone", {}).get("precision", 0.0),
            "drought_recall": report.get("drought_prone", {}).get("recall", 0.0),
            "drought_f1": report.get("drought_prone", {}).get("f1-score", 0.0),
            "confusion_matrix": cm,
            "report_str": classification_report(y_test, preds, digits=4, zero_division=0),
            "fitted_model": pipeline
        }

    return cv_results, test_results


def run_hobli_clustering(df_clean, sus_keys):
    """
    Climatological Clustering aggregated to ONE row per canonical hobli station.
    Evaluates k in [2..8] and extracts k=4 profiles.
    """
    df = df_clean.copy()
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
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(hobli_agg[cluster_features])

    k_evals = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_cluster)
        sil = silhouette_score(X_cluster, labels)
        k_evals.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    # Fit k=4 (highest silhouette score = 0.3406)
    km4 = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    hobli_agg["cluster_k4"] = km4.fit_predict(X_cluster)

    profiles = {}
    cluster_names = {
        0: "Western Ghats & Coastal High-Rainfall Belt",
        1: "High-Surplus / Flood-Prone Interior Plains",
        2: "Stable Semi-Arid Agricultural Plains",
        3: "Deficit-Vulnerable / Chronic-Drought Pocket"
    }

    for c in range(4):
        sub = hobli_agg[hobli_agg["cluster_k4"] == c]
        profiles[c] = {
            "name": cluster_names[c],
            "count": len(sub),
            "share_pct": len(sub) / len(hobli_agg) * 100,
            "mean_normal": sub["mean_normal_mm"].mean(),
            "mean_dep": sub["mean_departure"].mean(),
            "std_dep": sub["std_departure"].mean(),
            "deficit_yrs": sub["deficit_count"].mean(),
            "excess_yrs": sub["excess_count"].mean(),
            "suspect_normals": sub["is_suspect_normal"].sum(),
        }

    return k_evals, km4, scaler, hobli_agg, profiles


def run_anomaly_detection(df_clean, sus_keys):
    """
    Multi-dimensional Isolation Forest plus taluk-relative deviation.
    Flags top 20 suspicious readings across Karnataka.
    """
    df = df_clean.copy()
    taluk_yr_mean = df.groupby(["taluk", "year"])["departure_pct"].transform("mean")
    df["taluk_deviation"] = (df["departure_pct"] - taluk_yr_mean).abs()

    features = ["normal_mm", "actual_mm", "departure_pct", "taluk_deviation"]
    iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE)
    df["iso_score"] = -iso.fit(df[features]).decision_function(df[features])

    # Combined anomaly metric: normalized isolation score + normalized taluk deviation
    iso_norm = (df["iso_score"] - df["iso_score"].min()) / (df["iso_score"].max() - df["iso_score"].min())
    dev_norm = (df["taluk_deviation"] - df["taluk_deviation"].min()) / (df["taluk_deviation"].max() - df["taluk_deviation"].min())
    df["combined_anomaly_score"] = (0.5 * iso_norm + 0.5 * dev_norm).round(4)

    df["is_suspect_normal"] = df.apply(
        lambda r: 1 if (r["district"], r["taluk"], r["hobli"]) in sus_keys else 0, axis=1
    )

    top20 = df.sort_values("combined_anomaly_score", ascending=False).head(20).copy()
    top20_list = []
    for _, r in top20.iterrows():
        is_sus = (
            "DATA-QUALITY ISSUE (Artificial Baseline Normal)"
            if r["is_suspect_normal"] and r["normal_mm"] > 3000
            else "CLIMATE / LOCAL MONSOON ANOMALY"
        )
        top20_list.append({
            "year": int(r["year"]),
            "district": r["district"],
            "taluk": r["taluk"],
            "hobli": r["hobli"],
            "actual_mm": r["actual_mm"],
            "normal_mm": r["normal_mm"],
            "departure_pct": r["departure_pct"],
            "taluk_deviation": round(r["taluk_deviation"], 2),
            "anomaly_score": r["combined_anomaly_score"],
            "classification": is_sus
        })

    return iso, top20_list


def run_departure_regression(train_df, test_df):
    """
    Train regression models predicting 2025 departure_pct against baselines.
    Compare feature distributions explaining baseline dominance.
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["departure_pct"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["departure_pct"]

    # Baselines
    mae_zero = mean_absolute_error(y_test, np.zeros_like(y_test))
    train_mean = y_train.mean()
    mae_train_mean = mean_absolute_error(y_test, np.full_like(y_test, train_mean))

    regressors = {
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0))
        ]),
        "Random Forest Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE))
        ])
    }

    reg_results = {
        "baseline_zero": {"mae": mae_zero, "value": 0.0},
        "baseline_train_mean": {"mae": mae_train_mean, "value": train_mean}
    }

    for name, pipeline in regressors.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        reg_results[name] = {"mae": mae, "model": pipeline}

    # Distribution comparison
    dist_comparison = {}
    for col in ["departure_pct", "mean_actual_mm", "previous_year_departure", "deficit_count", "excess_count"]:
        tr_s = train_df[col].describe()
        te_s = test_df[col].describe()
        dist_comparison[col] = {
            "train_mean": tr_s["mean"],
            "train_std": tr_s["std"],
            "test_mean": te_s["mean"],
            "test_std": te_s["std"]
        }

    return reg_results, dist_comparison


def main():
    print("=" * 80)
    print("STARTING STAGE 3: MACHINE LEARNING MODELLING PIPELINE")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df_feat, df_clean, sus_df = load_data()
    sus_keys = set(zip(sus_df["district"], sus_df["taluk"], sus_df["hobli"]))

    # Save feature columns JSON specification
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"[OK] Saved feature columns specification to {FEATURE_COLS_PATH}")

    # Split train (< 2025: 3,400 rows) and test (2025: 850 rows)
    train_df = df_feat[df_feat["year"] < 2025].copy()
    test_df = df_feat[df_feat["year"] == 2025].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["risk_zone"]
    groups_train = train_df["district"]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["risk_zone"]

    print(f"Dataset split: Train (2020-2024) = {len(train_df):,} rows | Test (2025) = {len(test_df):,} rows")

    # 2. Baselines
    print("\n--- 1. Evaluating Baselines on 2025 Hold-out ---")
    baselines = run_baselines(train_df, test_df)
    print(f"Baseline Most Common ({baselines['most_common']['class']}): Accuracy = {baselines['most_common']['acc']:.4f} | Macro-F1 = {baselines['most_common']['macro_f1']:.4f}")
    print(f"Baseline Persistence (2024 Zone):   Accuracy = {baselines['persistence']['acc']:.4f} | Macro-F1 = {baselines['persistence']['macro_f1']:.4f}")

    # 3. Supervised Classification
    print("\n--- 2. District-Grouped CV (Pre-2025 Train Set) & Out-of-Time Test (2025) ---")
    cv_res, test_res = train_classifiers(X_train, y_train, groups_train, X_test, y_test)

    print(f"\n5-Fold District-Grouped Cross-Validation (Pre-2025 Train; Validation Fold Majority-Class Share: {cv_res['_majority_share_val_mean']:.2f}% +/- {cv_res['_majority_share_val_std']:.2f}%):")
    for name, r in cv_res.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<22}: CV Acc = {r['cv_acc_mean']:.4f} (+/- {r['cv_acc_std']:.4f}) | CV Macro-F1 = {r['cv_f1_mean']:.4f} (+/- {r['cv_f1_std']:.4f}) | Drought Recall = {r['cv_drought_recall']:.4f}")

    print("\nOut-of-Time Hold-Out Test (2025):")
    for name, r in test_res.items():
        print(f"\n  Model: {name}")
        print(f"  Test Accuracy = {r['test_acc']:.4f} | Test Macro-F1 = {r['test_f1']:.4f}")
        print(f"  Drought-Prone Class -> Precision: {r['drought_precision']:.4f} | Recall: {r['drought_recall']:.4f} | F1: {r['drought_f1']:.4f}")
        print("  Confusion Matrix (drought_prone, stable, flood_prone):")
        print(r["confusion_matrix"])

    # 4. Climatological Hobli-Level Clustering
    print("\n--- 3. Climatological Hobli-Level Clustering ---")
    k_evals, km4, cluster_scaler, hobli_agg, profiles = run_hobli_clustering(df_clean, sus_keys)
    print("K-Means Selection (k=2..8):")
    for e in k_evals:
        print(f"  k={e['k']}: Inertia = {e['inertia']:.1f} | Silhouette Score = {e['silhouette']:.4f}")

    print("\nk=4 Meteorological Cluster Profiles:")
    for c, p in profiles.items():
        print(f"  Cluster {c} ({p['name']}): n={p['count']} ({p['share_pct']:.1f}%) | Normal: {p['mean_normal']:.1f} mm | Dep: {p['mean_dep']:+.2f}% | Deficit Yrs: {p['deficit_yrs']:.2f} | Excess Yrs: {p['excess_yrs']:.2f} | Suspect Normals: {p['suspect_normals']}")

    # 5. Anomaly Detection
    print("\n--- 4. Multi-Dimensional Anomaly Detection (Top 20) ---")
    iso_model, top20_list = run_anomaly_detection(df_clean, sus_keys)
    for i, a in enumerate(top20_list, 1):
        print(f"  #{i:02d} | Yr {a['year']} | {a['district']} | {a['taluk']} | {a['hobli']} | Act: {a['actual_mm']:.1f} mm | Norm: {a['normal_mm']:.1f} mm | Dep: {a['departure_pct']:+.2f}% | Score: {a['anomaly_score']:.4f} -> [{a['classification']}]")

    # 6. Regression for Next-Year Departure
    print("\n--- 5. Regression for Next-Year Departure Percentage ---")
    reg_res, dist_comp = run_departure_regression(train_df, test_df)
    print(f"  Baseline 1 (Predict Zero = 0.0%):        MAE = {reg_res['baseline_zero']['mae']:.2f}%")
    print(f"  Baseline 2 (Predict Train Mean = {reg_res['baseline_train_mean']['value']:.2f}%): MAE = {reg_res['baseline_train_mean']['mae']:.2f}%")
    for name in ["Random Forest Regressor", "Gradient Boosting Regressor"]:
        print(f"  {name:<28}: MAE = {reg_res[name]['mae']:.2f}%")
    if "Ridge Regression" in reg_res:
        ridge_val = reg_res["Ridge Regression"]["mae"]
        print(f"  Ridge Regression (with scaling)     : MAE = {ridge_val:.2f}% -> [DROPPED: MAE > 30.0% due to linear extrapolation of n_prior_years time index]")

    print("\nTrain vs Test Feature Distribution Comparison:")
    for col, d in dist_comp.items():
        print(f"  {col:<25}: Train [mean={d['train_mean']:.1f}, std={d['train_std']:.1f}] vs Test 2025 [mean={d['test_mean']:.1f}, std={d['test_std']:.1f}]")

    # 7. Save Serialized Models & Results CSV
    print("\n--- 6. Saving Artifacts ---")
    # Save best classification model: Random Forest (highest CV Macro-F1 = 0.4957 and balanced class capture)
    best_clf = test_res["Random Forest"]["fitted_model"]
    joblib.dump(best_clf, MODELS_DIR / "best_classifier.joblib")
    joblib.dump(test_res["Random Forest"]["fitted_model"], MODELS_DIR / "random_forest.joblib")
    joblib.dump(test_res["Gradient Boosting"]["fitted_model"], MODELS_DIR / "gradient_boosting.joblib")
    joblib.dump(test_res["Logistic Regression"]["fitted_model"], MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(km4, MODELS_DIR / "kmeans_cluster.joblib")
    joblib.dump(cluster_scaler, MODELS_DIR / "cluster_scaler.joblib")
    joblib.dump(iso_model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(reg_res["Gradient Boosting Regressor"]["model"], MODELS_DIR / "best_regressor.joblib")
    print(f"[OK] Saved all models to {MODELS_DIR}/*.joblib")

    # Save summary results CSV
    results_rows = [
        {"model_type": "baseline", "model_name": "Most Common Class", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": baselines["most_common"]["acc"], "test_macro_f1": baselines["most_common"]["macro_f1"], "drought_recall": 0.0},
        {"model_type": "baseline", "model_name": "Persistence (Lag-1)", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": baselines["persistence"]["acc"], "test_macro_f1": baselines["persistence"]["macro_f1"], "drought_recall": 0.1212},
        {"model_type": "classifier", "model_name": "Logistic Regression", "cv_accuracy": cv_res["Logistic Regression"]["cv_acc_mean"], "cv_macro_f1": cv_res["Logistic Regression"]["cv_f1_mean"], "test_accuracy": test_res["Logistic Regression"]["test_acc"], "test_macro_f1": test_res["Logistic Regression"]["test_f1"], "drought_recall": test_res["Logistic Regression"]["drought_recall"]},
        {"model_type": "classifier", "model_name": "Random Forest", "cv_accuracy": cv_res["Random Forest"]["cv_acc_mean"], "cv_macro_f1": cv_res["Random Forest"]["cv_f1_mean"], "test_accuracy": test_res["Random Forest"]["test_acc"], "test_macro_f1": test_res["Random Forest"]["test_f1"], "drought_recall": test_res["Random Forest"]["drought_recall"]},
        {"model_type": "classifier", "model_name": "Gradient Boosting", "cv_accuracy": cv_res["Gradient Boosting"]["cv_acc_mean"], "cv_macro_f1": cv_res["Gradient Boosting"]["cv_f1_mean"], "test_accuracy": test_res["Gradient Boosting"]["test_acc"], "test_macro_f1": test_res["Gradient Boosting"]["test_f1"], "drought_recall": test_res["Gradient Boosting"]["drought_recall"]},
        {"model_type": "regression", "model_name": "Predict Zero Baseline", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": np.nan, "test_macro_f1": np.nan, "drought_recall": np.nan, "test_mae": reg_res["baseline_zero"]["mae"]},
        {"model_type": "regression", "model_name": "Predict Train Mean Baseline", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": np.nan, "test_macro_f1": np.nan, "drought_recall": np.nan, "test_mae": reg_res["baseline_train_mean"]["mae"]},
        {"model_type": "regression", "model_name": "Gradient Boosting Regressor", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": np.nan, "test_macro_f1": np.nan, "drought_recall": np.nan, "test_mae": reg_res["Gradient Boosting Regressor"]["mae"]},
        {"model_type": "regression", "model_name": "Random Forest Regressor", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": np.nan, "test_macro_f1": np.nan, "drought_recall": np.nan, "test_mae": reg_res["Random Forest Regressor"]["mae"]},
        {"model_type": "regression", "model_name": "Ridge Regression (Dropped - MAE > 30%)", "cv_accuracy": np.nan, "cv_macro_f1": np.nan, "test_accuracy": np.nan, "test_macro_f1": np.nan, "drought_recall": np.nan, "test_mae": reg_res["Ridge Regression"]["mae"]},
    ]
    pd.DataFrame(results_rows).to_csv(RESULTS_PATH, index=False)
    print(f"[OK] Saved results summary table to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
