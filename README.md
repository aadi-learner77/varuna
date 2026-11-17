# Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: ODC-BY](https://img.shields.io/badge/License-ODC--BY-green.svg)](https://opendatacommons.org/licenses/by/)

---

## 📌 Problem Statement

In Karnataka, the **hobli** (a sub-taluk revenue circle comprising 10 to 30 agricultural villages) serves as the fundamental administrative unit for ground-level governance, official drought declarations, state disaster relief funding, and Pradhan Mantri Fasal Bima Yojana (PMFBY) crop insurance payouts. While state and district-level monsoon forecasts mask severe micro-climatic rainfall deficits, predicting hyper-local annual rainfall departure and drought risk across 850+ hoblis is severely impeded by unstandardized, noisy open-government records and extreme class imbalance. This project develops an end-to-end, reproducible machine learning pipeline that harmonizes multi-year rainfall observations, engineers strict temporal-leakage-free features, evaluates spatial and out-of-time generalizability against honest baseline benchmarks, and delivers an interactive Streamlit decision-support application for rural disaster monitoring.

---

## ⚠️ Why the Data Is Hard (Data Quality Challenges)

Real-world meteorological data from state monitoring centres is rife with administrative inconsistencies and artifacts:
1. **Mislabeled Raw Tables**: Raw government CSVs contained swapped calendar years (e.g., `rainfall_2022.csv` was mislabeled as 2023, and `rainfall_2024.csv` was mislabeled as 2020). Correct temporal mappings were established via OpenCity metadata and KSNDMC official table verification.
2. **Missing Year 2023**: KSNDMC published no hobli-level annual tables for calendar year 2023. This non-consecutive temporal jump is explicitly parameterized via a `gap_years = 2` feature (between 2022 and 2024).
3. **Hierarchical Formatting & Duplicates**: Raw files totaled 5,636 rows with embedded District (`D`) and Taluk (`T`) summary rows interleaved with station records. Raw 2020 contained a duplicate printed block of 27 Chitradurga rows (indices 263–289) that had to be dropped.
4. **Nomenclature & Boundary Shifts**: 898 raw station name variants existed across files due to spelling mutations and Kannada-to-English phonetic transliterations. Furthermore, the late-2021 bifurcation of Vijayanagar from Ballari shifted 6 taluks across district boundaries.
5. **Artificial Baseline Normals (Suspect Normals)**: **338 hoblis** exhibit unchanged static normals alongside persistent one-sided departures. Most critically, **Agumbe** in Shivamogga is assigned a normal of **7,565.0 mm**; even when receiving a massive **5,383.1 mm** of rainfall in 2024, it is mechanically flagged as "drought-prone" (-28.84% departure) due to an uncalibrated historical benchmark.
6. **No Geospatial Coordinates**: The raw KSNDMC publications contain no latitude or longitude coordinates. In strict observance of data integrity, coordinates were not synthesized; spatial analysis is provided via district departure heatmaps.

---

## 🌐 Data Source & Licensing

- **Data Source**: Karnataka State Natural Disaster Monitoring Centre (KSNDMC), Government of Karnataka, accessed through the [OpenCity Karnataka Urban Data Portal](https://opencity.in/).
- **Temporal Horizon**: 5 observed years — **2020, 2021, 2022, 2024, and 2025** (850 canonical stations per year; 4,250 total cleaned records).
- **Licensing Note**: Open Data Commons Attribution License (ODC-BY) and National Data Sharing and Accessibility Policy (NDSAP), Government of India.

---

## 🏗️ Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
|                           KSNDMC OpenCity Raw Tables                              |
|          (5 files, 5,636 raw rows, mislabeled years, missing 2023)                 |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|               STAGE 1: Data Cleaning & Harmonization (clean_data.py)              |
|  - Drop 27 duplicate block rows (2020) & 1,359 summary rows (D/T)                 |
|  - Harmonize 898 name variants -> 877 canonical stations (RapidFuzz >= 85%)       |
|  - Resolve Vijayanagar/Ballari bifurcation (6 taluks)                             |
|  - Verify 0 departure recomputation discrepancies across 4,250 rows               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|          STAGE 2: Feature Engineering & Leakage Prevention (features.py)          |
|  - Ex-ante features strictly computed on prior years (year < Y)                   |
|  - Expanding means, volatilities (std), lag-1 departure, regional aggregations    |
|  - Temporal distance tracking (gap_years: 0 for 2020, 1 for 2021/22/25, 2 for 24) |
|  - Flag 338 static baseline stations (suspect_normals.csv)                        |
|  - Target: KSNDMC Risk Zone (drought <= -20%, flood >= +40%, stable -20% to +40%) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|            STAGE 3: Machine Learning & Scientific Evaluation (train_models.py)    |
|  - 5-Fold District-Grouped Cross-Validation (GroupKFold across 31 districts)      |
|  - Out-of-Time Stress Test (Train 2020-2024: 3,400 rows | Test 2025: 850 rows)    |
|  - Benchmarked against Majority-Class & Persistence Baselines                     |
|  - Climatological K-Means Clustering (k=4, silhouette = 0.3406)                   |
|  - Anomaly Detection (Isolation Forest + Taluk Micro-climatic Deviation)          |
|  - Departure Regression Debugging (Drop Ridge MAE 53.79%; evaluate zero baseline) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|              STAGE 4: Interactive Application & Reporting (app/app.py)            |
|  - Streamlit multi-tab interface with dependent dropdowns (District->Taluk->Hobli)|
|  - Live inference via src/predict_risk.py with class probability distribution     |
|  - Climatological cluster explorer & filterable chronic-deficit registry          |
|  - District departure heatmap replacing spatial map without coordinates           |
|  - Full transparency on data quality and machine learning model limits            |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
ml-project/
├── .gitignore
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Production Python dependencies
├── app/
│   └── app.py                         # Streamlit multi-tab decision-support web app
├── data/
│   ├── raw/                           # Standardized raw CSV files (5 years)
│   ├── clean/
│   │   ├── clean_rainfall.csv         # Cleaned longitudinal records (4,250 rows x 9 cols)
│   │   └── name_mapping.csv           # RapidFuzz cross-year name harmonization map
│   └── features/
│       ├── features.csv               # Engineered ML features (4,250 rows x 22 cols)
│       ├── hobli_clusters.csv         # Exported k=4 climatological cluster profiles
│       └── README.md                  # Comprehensive feature dictionary and target contracts
├── docs/
│   ├── presentation_outline.md        # 10-slide presentation deck with speaker notes
│   └── viva_qa.md                     # 20 rigorous examiner viva Q&A responses
├── models/
│   ├── best_classifier.joblib         # Serialized Random Forest classifier
│   ├── best_regressor.joblib          # Serialized Gradient Boosting regressor
│   ├── cluster_scaler.joblib          # Fitted StandardScaler for K-Means clustering
│   ├── feature_columns.json           # 13 verified predictive feature columns contract
│   ├── gradient_boosting.joblib       # Serialized Gradient Boosting classifier
│   ├── isolation_forest.joblib        # Fitted Isolation Forest anomaly detector
│   ├── kmeans_cluster.joblib          # Fitted 4-cluster K-Means model
│   ├── logistic_regression.joblib     # Serialized Logistic Regression classifier
│   ├── random_forest.joblib           # Serialized Random Forest model
│   └── results.csv                    # Empirical evaluation metrics and baseline comparisons
├── notebooks/
│   └── eda.ipynb                      # Comprehensive Exploratory Data Analysis notebook
├── reports/
│   ├── cleaning_report.md             # Stage 1 data cleaning and raw audit report
│   ├── final_report.md                # Comprehensive final academic research report
│   ├── modelling_report.md            # Stage 3 machine learning experimentation report
│   ├── review_names.csv               # Audit trail for borderline fuzzy station matches
│   └── suspect_normals.csv            # 338 flagged stations with static baseline normals
├── scripts/
│   ├── audit_name_merges.py           # Cross-station suffix and boundary audit script
│   ├── refresh_eda.py                 # Automated EDA notebook recreation runner
│   └── smoke_test_app.py              # Automated 3-station headless application smoke test
└── src/
    ├── clean_data.py                  # Stage 1 extraction, parsing, and harmonization
    ├── export_cluster_labels.py       # Cluster assignment exporter generating hobli_clusters.csv
    ├── features.py                    # Stage 2 temporal feature engineering pipeline
    ├── predict_risk.py                # Production inference module with strict schema validation
    └── train_models.py                # Stage 3 cross-validation and model training pipeline
```

---

## 🚀 How to Run the Project (Step-by-Step)

### 1. Environment Setup
```bash
git clone https://github.com/pranavhm25/varuna.git
cd varuna
git checkout app

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Stage-by-Stage Execution in Order

```bash
# Stage 1: Clean raw KSNDMC files and harmonize station nomenclature
python src/clean_data.py

# Stage 2: Engineer leakage-free temporal features and audit suspect normals
python src/features.py

# Stage 3: Train machine learning models, run GroupKFold CV, and evaluate baselines
python src/train_models.py

# Export Hobli Climatological Cluster Labels
python src/export_cluster_labels.py

# Verify Inference Engine Contract
python src/predict_risk.py

# Run Headless Application Smoke Test
python scripts/smoke_test_app.py
```

### 3. Launching the Streamlit Web Application
```bash
streamlit run app/app.py
```
*The app will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 📊 Empirical Results Summary

All metrics are pulled directly from `models/results.csv` and reflect rigorous evaluation on the **2025 out-of-time hold-out test set (850 unseen hoblis)**:

### 1. Classification Performance vs. Baselines

| Model Architecture / Baseline | 5-Fold District CV Acc | 5-Fold District Macro-F1 | 2025 Test Accuracy | 2025 Test Macro-F1 | 2025 Drought Recall | Status / Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline: Most Common Class (`stable`)** | — | — | **0.7965** | 0.2956 | 0.0000 | Naive benchmark; catches 0 drought events |
| **Baseline: Persistence (Lag-1 2024)** | — | — | 0.6812 | **0.4095** | **0.1212** | Naive weather persistence benchmark |
| **Random Forest Classifier** | 0.6581 | **0.4957** | **0.7082** | **0.4077** | 0.0606 | **Selected Best Classifier** (30.49% CV drought recall) |
| **Gradient Boosting Classifier** | **0.7323** | 0.4674 | 0.6788 | 0.3840 | 0.0303 | Defaults toward majority class; high precision |
| **Logistic Regression (Balanced)** | 0.4464 | 0.3590 | 0.1376 | 0.0948 | 0.0152 | Severe class-weight penalty distortion |

### 2. Next-Year Departure Regression vs. Baselines

| Model Architecture / Baseline | Test MAE (%) | Operational Outcome |
| :--- | :---: | :--- |
| **Baseline: Predict Zero Departure ($0.0\%$)** | **21.65%** | **Best Estimator**: Represents long-term climatological normal |
| **Baseline: Predict Train Mean ($+29.50\%$)** | 24.53% | Suffers from wet training window bias |
| **Gradient Boosting Regressor** | **24.42%** | Best ML model; lags zero baseline by 2.77 pp |
| **Random Forest Regressor** | 26.35% | Lags zero baseline by 4.70 pp |
| **Ridge Regression (StandardScaler, $\alpha=10$)** | **53.79%** | **DROPPED**: Linearly extrapolated $n\_prior\_years$ time index |

---

## 💡 Key Scientific Findings & Limitations

1. **Accuracy Is Misleading Under Severe Imbalance**: In 2025, **79.65%** of Karnataka hoblis experienced `stable` rainfall. A trivial dummy model predicting `stable` for all stations scores ~80% accuracy but has **0% drought recall**. Random Forest achieves 70.82% accuracy but successfully detects localized moisture stress (achieving 30.49% CV drought recall across unseen districts).
2. **Post-Deluge Regime Abatement**: The 2020–2024 training window was dominated by heavy monsoon deluges culminating in the historic 2022 flood (+50.67%). In 2025, rainfall cooled toward normal (+13.20%). The simple climatological benchmark of **Predict Zero Departure ($0.0\%$)** beats all ML regressors (21.65% MAE vs. GBR's 24.42%) because chaotic synoptic atmospheric circulation does not exhibit linear persistence.
3. **Data Quality Artifacts**: 338 hoblis exhibit static baseline normals. Stations like **Agumbe** receive over 5,000 mm of rainfall yet are labeled "drought-prone" because their baseline normal is fixed at 7,565 mm.
4. **Precipitation-Only Limitation**: Predictions rely exclusively on surface rainfall. Real-world agronomic drought depends on evapotranspiration, soil moisture index, groundwater levels, and temperature.
5. **No Coordinates**: Lack of GIS polygon or point coordinates limits spatial mapping; district heatmaps serve as an interim visualization pending coordinate ingestion.

---

## 👥 Team Roles and Contributions

This project was developed collaboratively as a 4-person academic machine learning project:

- **Person 1 (Data Cleaning & Harmonization)**:
  - Raw table extraction, year correction, summary row filtering (`D`/`T` rows), and duplicate block elimination.
  - Multi-year station name standardization using RapidFuzz phonetic matching and Ballari/Vijayanagar bifurcation resolution.
  - Delivery: `src/clean_data.py`, `data/clean/clean_rainfall.csv`, `reports/cleaning_report.md`.

- **Person 2 (Feature Engineering & Exploratory Data Analysis)**:
  - Temporal feature engineering strictly preventing lookahead leakage (`year < Y`).
  - Expanding statistical aggregations, lag indicators, regional offsets, and `gap_years` handling for missing 2023.
  - Exploratory data analysis, departure distribution analysis, and suspect normal identification.
  - Delivery: `src/features.py`, `notebooks/eda.ipynb`, `data/features/README.md`, `reports/suspect_normals.csv`.

- **Person 3 (Machine Learning Modelling & Diagnostics)**:
  - Baseline establishment (Majority-Class, Persistence, Predict Zero, Predict Train Mean).
  - Supervised classification with 5-Fold District-Grouped CV (`GroupKFold`) and out-of-time test evaluation on 2025.
  - K-Means climatological clustering ($k=4$), Isolation Forest anomaly detection, and regression debugging (Ridge drop).
  - Delivery: `src/train_models.py`, `src/predict_risk.py`, `models/*`, `reports/modelling_report.md`.

- **Person 4 (Application Engineering, System Integration, Documentation & Defense Preparation)**:
  - Streamlit web application development (`app/app.py`), cluster labels export (`src/export_cluster_labels.py`), and headless smoke testing (`scripts/smoke_test_app.py`).
  - System integration, dependencies specification (`requirements.txt`), and comprehensive documentation (`README.md`).
  - Final academic report (`reports/final_report.md`), 10-slide presentation deck (`docs/presentation_outline.md`), and 20 viva examination defense responses (`docs/viva_qa.md`).
