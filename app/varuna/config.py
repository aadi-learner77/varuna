"""
app/varuna/config.py
====================
Static project metadata, scientific thresholds and copy used across the
Varuna interface. Nothing here recomputes or alters modelling results: the
values are the documented contracts of the existing pipeline (README,
reports/cleaning_report.md, reports/modelling_report.md) plus display labels.
"""

PRODUCT = "VARUNA"
TAGLINE = "Karnataka Rainfall Intelligence"
SOURCE = "KSNDMC"
SOURCE_LONG = "Karnataka State Natural Disaster Monitoring Centre"
LICENSE = "ODC-BY"

OBSERVED_YEARS = [2020, 2021, 2022, 2024, 2025]
MISSING_YEAR = 2023
FORECAST_YEARS = [2026, 2027]

# KSNDMC risk-zone thresholds used by the target definition and the app.
DROUGHT_THRESHOLD = -20.0
EXCESS_THRESHOLD = 40.0
HISTORIC_EXCESS_THRESHOLD = 20.0  # used for the cluster "excess years" count

ZONE_ORDER = ["drought_prone", "stable", "flood_prone"]
ZONE_LABELS = {
    "drought_prone": "Drought-prone",
    "stable": "Stable",
    "flood_prone": "Flood-prone",
}
ZONE_TONES = {
    "drought_prone": "terra",
    "stable": "green",
    "flood_prone": "rain",
}
ZONE_SHORT = {
    "drought_prone": "Deficit",
    "stable": "Normal",
    "flood_prone": "Excess",
}

CLUSTERS = {
    0: {
        "name": "Western Ghats & Coastal High-Rainfall Belt",
        "short": "Ghats high-rainfall belt",
        "glyph": "circle",
        "tone": "teal",
        "desc": (
            "Heavy monsoon corridor of the Western Ghats and coastal districts. "
            "Very high volumetric rainfall with comparatively low percentage variance."
        ),
    },
    1: {
        "name": "High-Surplus / Flood-Prone Interior Plains",
        "short": "High-surplus interior",
        "glyph": "diamond",
        "tone": "rain",
        "desc": (
            "Semi-arid northern and eastern plains that swing into large positive "
            "departures during active monsoon surges."
        ),
    },
    2: {
        "name": "Stable Semi-Arid Agricultural Plains",
        "short": "Stable agricultural plains",
        "glyph": "square",
        "tone": "green",
        "desc": (
            "Central and southern agricultural belt. Moderate rainfall with the "
            "lowest year-to-year percentage volatility in the panel."
        ),
    },
    3: {
        "name": "Deficit-Vulnerable / Chronic-Drought Pocket",
        "short": "Chronic-deficit pockets",
        "glyph": "triangle",
        "tone": "terra",
        "desc": (
            "Pockets with repeated negative departures: the highest deficit-year "
            "frequency and widest downside spread in the observed panel."
        ),
    },
}

# Raw-to-clean provenance, transcribed from reports/cleaning_report.md.
CLEANING_ROWS = [
    {"year": 2020, "file": "rainfall_2020.csv", "raw": 1134, "duplicates": 27, "summary": 257, "clean": 850, "mean_dep": 20.97},
    {"year": 2021, "file": "rainfall_2021.csv", "raw": 1113, "duplicates": 0, "summary": 263, "clean": 850, "mean_dep": 27.65},
    {"year": 2022, "file": "rainfall_2022.csv", "raw": 1150, "duplicates": 0, "summary": 300, "clean": 850, "mean_dep": 50.66},
    {"year": 2024, "file": "rainfall_2024.csv", "raw": 1122, "duplicates": 0, "summary": 272, "clean": 850, "mean_dep": 18.74},
    {"year": 2025, "file": "rainfall_2025.csv", "raw": 1117, "duplicates": 0, "summary": 267, "clean": 850, "mean_dep": 13.20},
]

CLEANING_TOTALS = {"raw": 5636, "duplicates": 27, "summary": 1359, "clean": 4250}

NORMALIZATION_FACTS = {
    "name_variants": 898,
    "canonical": 877,
    "persistent": 824,
    "persistent_pct": 94.0,
    "min_match_score": 85,
    "formula_deviations": 0,
    "vijayanagar_taluks": 6,
}

# The 13 features the serialized classifier expects, in contract order.
FEATURE_DICTIONARY = [
    ("normal_mm", "Long-term normal rainfall (mm) reported for the station."),
    ("gap_years", "Calendar years since the previous observation (2 across the missing 2023)."),
    ("n_prior_years", "Number of observed years available before the target year."),
    ("deficit_count", "Prior years with departure \u2264 \u221220%."),
    ("excess_count", "Prior years with departure \u2265 +20%."),
    ("mean_actual_mm", "Expanding mean of prior observed rainfall (mm)."),
    ("std_actual_mm", "Expanding volatility of prior observed rainfall (mm)."),
    ("mean_departure_pct", "Expanding mean of prior departures (%)."),
    ("std_departure_pct", "Expanding volatility of prior departures (%)."),
    ("previous_year_departure", "Lag-1 departure from the most recent observed year (%)."),
    ("taluk_mean_departure", "Mean departure of the station's taluk in the target year."),
    ("district_mean_departure", "Mean departure of the station's district in the target year."),
    ("hobli_minus_taluk", "Station departure minus taluk mean: a micro-climatic offset."),
]

PIPELINE_STAGES = [
    ("Raw source tables", "Five KSNDMC annual CSVs as published", "5,636", "raw records"),
    ("Administrative rows removed", "District (D) and taluk (T) summary lines interleaved with stations", "\u22121,359", "rows"),
    ("Printed duplicate block dropped", "Raw 2020 indices 263\u2013289, a reprinted Chitradurga block", "\u221227", "rows"),
    ("Canonical station-years", "877 harmonized stations across five observed years", "4,250", "clean records"),
]

ISSUE_NOTES = [
    {
        "title": "Swapped year labels in the source publications",
        "body": (
            "Two raw tables carried incorrect calendar years. OpenCity metadata and the "
            "KSNDMC published tables were used to restore the true temporal order before "
            "any feature was engineered."
        ),
    },
    {
        "title": "No hobli tables for calendar year 2023",
        "body": (
            "KSNDMC published no hobli-level annual table for 2023. The gap is modelled "
            "explicitly as gap_years = 2 between 2022 and 2024 rather than hidden or "
            "interpolated."
        ),
    },
    {
        "title": "Administrative reorganisation mid-panel",
        "body": (
            "Vijayanagar was carved out of Ballari in late 2021, moving six taluks across "
            "a district boundary. Station keys were harmonized so the panel stays "
            "longitudinal rather than splitting a station's history."
        ),
    },
    {
        "title": "898 spelling variants for 877 stations",
        "body": (
            "Kannada-to-English transliteration produced 898 raw station strings. RapidFuzz "
            "matching at \u2265 85% similarity resolved them to 877 canonical stations, with "
            "borderline pairs written to an audit file for review."
        ),
    },
    {
        "title": "338 baseline normals that never move",
        "body": (
            "338 hoblis carry an unchanged normal alongside a persistent one-sided "
            "departure. Their departures measure the benchmark as much as the weather, so "
            "the interface flags them wherever they appear."
        ),
    },
]

MODELS = [
    {
        "key": "random_forest",
        "name": "Random Forest",
        "role": "Selected classifier",
        "note": "Best cross-validated macro-F1 of the supervised models; the serialized inference artifact.",
    },
    {
        "key": "gradient_boosting",
        "name": "Gradient Boosting",
        "role": "Challenger",
        "note": "Highest cross-validated accuracy, but defaults toward the majority class.",
    },
    {
        "key": "logistic_regression",
        "name": "Logistic Regression",
        "role": "Linear reference",
        "note": "Balanced class weights distort the decision surface under this imbalance.",
    },
    {
        "key": "majority",
        "name": "Majority class",
        "role": "Naive baseline",
        "note": "Predicts \u201cstable\u201d everywhere: high accuracy, zero drought detections.",
    },
    {
        "key": "persistence",
        "name": "Persistence (lag-1)",
        "role": "Weather baseline",
        "note": "Repeats the previous observed year's class.",
    },
]

LIMITATIONS = [
    (
        "Accuracy is not skill under imbalance",
        "In 2025, 79.65% of hoblis were classified stable. Guessing \u201cstable\u201d for every "
        "station reproduces that accuracy and detects none of the 66 drought hoblis. "
        "Macro-F1 and drought recall are reported alongside accuracy for this reason.",
    ),
    (
        "No predictive edge over climatology in regression",
        "Predicting a zero departure \u2014 that is, simply expecting normal rainfall \u2014 beats "
        "every trained regressor on 2025 MAE. Monsoon departures are not linearly "
        "persistent, and the models learned the wet bias of the 2020\u20132024 window.",
    ),
    (
        "Rainfall is the only input",
        "The models see surface rainfall only. Agronomic drought also depends on "
        "evapotranspiration, soil moisture, groundwater and temperature, none of which "
        "are in the panel.",
    ),
    (
        "Baseline normals are not equally trustworthy",
        "338 stations carry static normals with persistent one-sided departures. Extreme "
        "percentage values for these stations can be artifacts of the benchmark rather "
        "than genuine meteorological anomalies.",
    ),
    (
        "No verified geography",
        "The KSNDMC tables contain no latitude or longitude. Coordinates and district "
        "polygons were not synthesized, so spatial analysis is delivered as district "
        "departure matrices rather than a map.",
    ),
]

PROVENANCE = (
    "KSNDMC via OpenCity Karnataka \u00b7 2020\u20132025 panel \u00b7 850 stations per observed year \u00b7 "
    "Licensed under ODC-BY / NDSAP"
)
