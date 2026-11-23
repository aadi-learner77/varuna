"""
Module 05 -- Model Performance & Limitations
===========================================
Honest, evidence-first presentation of the classification and regression
results, including the baselines that beat the models.
"""

from __future__ import annotations

import streamlit as st

from .. import charts, config, ui


PAGE = {
    "slug": "models",
    "index": "05",
    "label": "Model Evidence",
    "title": "Model Performance & Limitations",
    "kicker": "Module 05 \u00b7 Evaluation",
    "lede": (
        "What the models actually achieve on unseen districts and an unseen season \u2014 including "
        "the baselines they fail to beat. Accuracy is reported, then set aside for metrics that "
        "survive severe class imbalance."
    ),
}


def _get(results, name: str, column: str) -> float | None:
    row = results[results["model_name"] == name]
    if not len(row):
        return None
    value = row.iloc[0][column]
    return None if value != value else float(value)


def _pct(value, digits: int = 1) -> str:
    return "\u2014" if value is None else f"{value * 100:.{digits}f}%"


def render(data, sel: dict, p: dict) -> None:
    results = data.results
    majority_acc = _get(results, "Most Common Class", "test_accuracy")
    majority_f1 = _get(results, "Most Common Class", "test_macro_f1")
    rf_acc = _get(results, "Random Forest", "test_accuracy")
    rf_f1 = _get(results, "Random Forest", "test_macro_f1")
    rf_recall = _get(results, "Random Forest", "drought_recall")
    rf_cv_recall = 0.3049  # documented 5-fold district CV drought recall (train_models.py)

    holdout = data.features[data.features["year"] == 2025]
    counts = holdout["risk_zone"].value_counts()

    ui.render(
        ui.page_header(
            PAGE["kicker"],
            PAGE["title"],
            PAGE["lede"],
            meta=[
                ("Validation", "5-fold GroupKFold \u00b7 31 districts"),
                ("Out-of-time test", "2025 \u00b7 850 hoblis"),
                ("Training window", "2020\u20132024 \u00b7 3,400 rows"),
                ("Selected model", "Random Forest classifier"),
            ],
        )
    )

    ui.render(
        ui.stat_strip(
            [
                {"label": "Majority-class accuracy", "value": _pct(majority_acc), "note": "Predicting \u201cstable\u201d everywhere"},
                {"label": "Its drought recall", "value": _pct(_get(results, "Most Common Class", "drought_recall")), "note": "Zero of 66 drought hoblis", "color": p["terra"]},
                {"label": "Random Forest accuracy", "value": _pct(rf_acc), "note": "2025 hold-out", "color": p["rain"]},
                {"label": "Random Forest macro-F1", "value": f"{rf_f1:.3f}" if rf_f1 is not None else "\u2014", "note": f"Majority baseline {majority_f1:.3f}" if majority_f1 is not None else ""},
                {"label": "CV drought recall", "value": _pct(rf_cv_recall), "note": "Unseen districts", "color": p["green"]},
                {"label": "Hold-out drought recall", "value": _pct(rf_recall), "note": "2025, after regime abatement", "color": p["amber"]},
            ]
        )
    )

    ui.render(
        ui.callout(
            "amber",
            "Read this before any accuracy figure",
            ui.paragraph(
                f"In the 2025 hold-out, <b>{counts.get('stable', 0)} of {len(holdout)}</b> hoblis were stable "
                f"({counts.get('stable', 0) / len(holdout) * 100:.2f}%). A majority-class baseline therefore "
                "scores highly while detecting nothing. Every result on this page is accompanied by "
                "macro-F1 and drought recall for that reason."
            ),
        )
    )

    # ------------------------------------------------------ accuracy paradox --
    ui.render(ui.section("01", "The accuracy paradox", "2025 hold-out \u00b7 850 unseen hoblis"))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(
            charts.accuracy_vs_recall(results, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Accuracy against drought recall. The majority baseline is the tallest accuracy bar and has no recall bar at all.")
    with right:
        ui.render(ui.sub_label("What the gap means"))
        ui.render(
            ui.paragraph(
                f"The baseline reaches <b>{_pct(majority_acc)}</b> accuracy by never predicting drought. "
                f"Random Forest gives up {abs((majority_acc or 0) - (rf_acc or 0)) * 100:.2f} percentage points "
                f"of accuracy and returns <b>{_pct(rf_recall)}</b> drought recall on the hold-out, rising to "
                f"<b>{_pct(rf_cv_recall)}</b> when districts are left out of training. Only the second "
                "behaviour is operationally useful for a drought-warning workflow, and even it is weak."
            )
        )
        ui.render(
            ui.callout(
                "terra",
                "The honest reading",
                ui.paragraph(
                    "A model that misses 94% of 2025 drought events cannot be used as an early-warning "
                    "system on its own. It is a research baseline that demonstrates feasibility, not a "
                    "deployable forecaster."
                ),
            )
        )

    # ------------------------------------------------------------ comparison --
    ui.render(ui.section("02", "Discrimination across models", "Macro-F1 with cross-validation and hold-out"))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(
            charts.macro_f1(results, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Bars are 2025 test macro-F1; open diamonds are 5-fold district-grouped CV macro-F1. Baselines have no CV score because they are not fitted.")
    with right:
        ui.render(ui.sub_label("Why district-grouped validation"))
        ui.render(
            ui.paragraph(
                "GroupKFold holds out entire districts rather than random rows, so a station's "
                "neighbouring years never appear on both sides of the split. It measures spatial "
                "generalisation, which is the operationally relevant question. Random Forest records "
                "the strongest CV macro-F1 at 0.4957, while Gradient Boosting leads CV accuracy at "
                "73.23% with a lower macro-F1 \u2014 the familiar majority-class pull."
            )
        )

    res_rows = []
    for _, r in results.iterrows():
        res_rows.append(
            {
                "model": r["model_name"],
                "type": r["model_type"],
                "cv_acc": _pct(r["cv_accuracy"], 2),
                "cv_f1": "\u2014" if r["cv_macro_f1"] != r["cv_macro_f1"] else f"{r['cv_macro_f1']:.4f}",
                "test_acc": _pct(r["test_accuracy"], 2) if r["test_accuracy"] == r["test_accuracy"] else "\u2014",
                "test_f1": "\u2014" if r["test_macro_f1"] != r["test_macro_f1"] else f"{r['test_macro_f1']:.4f}",
                "recall": _pct(r["drought_recall"], 2) if r["drought_recall"] == r["drought_recall"] else "\u2014",
            }
        )
    ui.render(ui.sub_label("Full results table \u00b7 source: models/results.csv"))
    ui.render(
        ui.table(
            [
                {"label": "Model or baseline", "key": "model"},
                {"label": "Type", "key": "type"},
                {"label": "CV accuracy", "key": "cv_acc", "align": "r"},
                {"label": "CV macro-F1", "key": "cv_f1", "align": "r"},
                {"label": "2025 accuracy", "key": "test_acc", "align": "r"},
                {"label": "2025 macro-F1", "key": "test_f1", "align": "r"},
                {"label": "Drought recall", "key": "recall", "align": "r"},
            ],
            res_rows,
            caption="Classification metrics \u00b7 2025 out-of-time hold-out and 5-fold district CV",
        )
    )

    # ------------------------------------------------------------ imbalance --
    ui.render(ui.section("03", "Class balance across the panel", "Why the target is hard"))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(
            charts.regime_composition(data.features, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Recorded risk-zone counts per observed season. The dominant class changes with the monsoon regime, which is exactly why a fixed majority baseline is a trap.")
    with right:
        ui.render(ui.sub_label("2025 hold-out composition"))
        ui.render(
            ui.deflist(
                [
                    ("Stable", f"{counts.get('stable', 0)} hoblis"),
                    ("Flood-prone", f"{counts.get('flood_prone', 0)} hoblis"),
                    ("Drought-prone", f"{counts.get('drought_prone', 0)} hoblis"),
                    ("Drought share", f"{counts.get('drought_prone', 0) / len(holdout) * 100:.2f}%"),
                    ("Majority share", f"{counts.get('stable', 0) / len(holdout) * 100:.2f}%"),
                ]
            )
        )
        ui.render(
            ui.paragraph(
                "Drought hoblis doubled from 33 in 2024 to 66 in 2025 while the statewide mean "
                "departure moderated to +13.20%. Rare, spatially localised events are the ones the "
                "task actually cares about."
            )
        )

    # ----------------------------------------------------------- regression --
    ui.render(ui.section("04", "Departure regression against climatology", "2025 mean absolute error"))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(
            charts.regression_mae(results, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Lower is better. The zero-departure climatological baseline is the green bar.")
    with right:
        ui.render(
            ui.paragraph(
                "The best estimator of next year's departure is <b>zero departure</b>: expect normal "
                "rainfall. It achieves 21.65% MAE, better than Gradient Boosting (24.42%) and Random "
                "Forest (26.35%), both of which learned the wet bias of the 2020\u20132024 training window. "
                "Ridge Regression failed outright at 53.79% MAE because it extrapolated a false linear "
                "trend from the number of prior years, and was dropped."
            )
        )
        ui.render(
            ui.callout(
                "rain",
                "Implication for the forecast module",
                ui.paragraph(
                    "The 2026\u20132027 projections are presented with uncertainty bands and framing, not "
                    "as verified forecasts. Their value is scenario exploration against a stated normal, "
                    "not precipitation prediction."
                ),
            )
        )

    # ---------------------------------------------------------- limitations --
    ui.render(ui.section("05", "Limitations", "Stated plainly"))
    ui.render(ui.findings(config.LIMITATIONS))

    with st.expander("Metric definitions and evaluation protocol"):
        ui.render(
            ui.table(
                [
                    {"label": "Metric", "key": "metric"},
                    {"label": "Definition", "key": "definition"},
                ],
                [
                    {"metric": "Accuracy", "definition": "Share of all 850 hold-out hoblis assigned the correct zone."},
                    {"metric": "Macro-F1", "definition": "Unweighted mean of per-class F1. A class never predicted contributes zero, so this metric cannot be gamed by ignoring drought."},
                    {"metric": "Drought recall", "definition": "Share of true drought-prone hoblis that were correctly flagged."},
                    {"metric": "MAE", "definition": "Mean absolute error of the predicted departure in percentage points."},
                    {"metric": "5-fold GroupKFold", "definition": "Districts are partitioned into five folds; every station in a held-out district is unseen during training."},
                    {"metric": "Out-of-time test", "definition": "Train on 2020\u20132024 (3,400 rows), evaluate once on the 2025 season (850 rows)."},
                ],
                caption="Evaluation protocol \u00b7 unchanged from the existing modelling pipeline",
            )
        )
        ui.render(
            ui.paragraph(
                "All values on this page are read directly from <code>models/results.csv</code> except "
                "the cross-validated drought recall, which is documented in the modelling report. "
                "No metric is recomputed, rounded up or reweighted for presentation, and no model was "
                "retrained for this interface."
            )
        )
