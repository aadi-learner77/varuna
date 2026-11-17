import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate():
    df = pd.read_csv("data/features/features.csv")
    with open("models/feature_columns.json") as f:
        feat_cols = json.load(f)

    rf = joblib.load("models/random_forest.joblib")
    gb = joblib.load("models/gradient_boosting.joblib")
    lr = joblib.load("models/logistic_regression.joblib")

    df_sorted = df.sort_values(["district", "taluk", "hobli", "year"])
    df["prev_zone"] = df_sorted.groupby(["district", "taluk", "hobli"])["risk_zone"].shift(1)

    for yr in [2020, 2021, 2022, 2024, 2025]:
        sub = df[df["year"] == yr]
        X = sub[feat_cols]
        y = sub["risk_zone"]
        maj = y.value_counts().index[0]
        pers = sub["prev_zone"].fillna(maj)

        rf_preds = rf.predict(X)
        gb_preds = gb.predict(X)
        lr_preds = lr.predict(X)

        rf_acc = accuracy_score(y, rf_preds)
        gb_acc = accuracy_score(y, gb_preds)
        lr_acc = accuracy_score(y, lr_preds)
        pers_acc = accuracy_score(y, pers)
        maj_acc = (y == maj).mean()

        rf_f1 = f1_score(y, rf_preds, average="macro")
        gb_f1 = f1_score(y, gb_preds, average="macro")
        lr_f1 = f1_score(y, lr_preds, average="macro")
        pers_f1 = f1_score(y, pers, average="macro")
        maj_f1 = f1_score(y, [maj] * len(y), average="macro")

        print(f"=== Year {yr} (N={len(sub)}) ===")
        print(f"  Dominant Regime: {maj} ({maj_acc*100:.2f}%)")
        print(f"  Random Forest:       Accuracy = {rf_acc:.4f} ({rf_acc*100:.2f}%) | Macro-F1 = {rf_f1:.4f}")
        print(f"  Gradient Boosting:   Accuracy = {gb_acc:.4f} ({gb_acc*100:.2f}%) | Macro-F1 = {gb_f1:.4f}")
        print(f"  Logistic Regression: Accuracy = {lr_acc:.4f} ({lr_acc*100:.2f}%) | Macro-F1 = {lr_f1:.4f}")
        print(f"  Persistence (Lag-1): Accuracy = {pers_acc:.4f} ({pers_acc*100:.2f}%) | Macro-F1 = {pers_f1:.4f}")
        print(f"  Majority Baseline:   Accuracy = {maj_acc:.4f} ({maj_acc*100:.2f}%) | Macro-F1 = {maj_f1:.4f}\n")

if __name__ == "__main__":
    evaluate()
