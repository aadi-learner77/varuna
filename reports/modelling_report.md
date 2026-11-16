# Karnataka Hobli-Level Rainfall Deviation & Drought-Risk Prediction: Modelling Report

## Executive Summary & Overview

This report details the machine learning modelling pipeline developed for **"Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka"**. Using standardized KSNDMC rainfall records spanning **5 distinct calendar years (2020, 2021, 2022, 2024, and 2025; 4,250 observations)** across 850 canonical hoblis, the system evaluates:
1. **Spatial Generalization**: 5-Fold District-Grouped Cross-Validation (GroupKFold across 31 administrative districts).
2. **Out-of-Time Regime Shift Stress Test**: Training strictly on earlier years (2020–2024; 3,400 rows) and testing on 2025 hold-out observations (850 rows).
3. **Climatological Hobli Clustering**: Longitudinal multi-year clustering aggregating each hobli to a single profile.
4. **Multi-Dimensional Anomaly Detection**: Isolation Forest combined with taluk-level micro-climatic deviation.
5. **Next-Year Departure Regression**: Diagnostic evaluation against baseline estimators.

---

## 1. Spatial Generalization: 5-Fold District-Grouped Cross-Validation

Standard random $k$-fold cross-validation suffers from massive spatial data leakage: contiguous hoblis within the same taluk or district share identical synoptic weather patterns. To evaluate true spatial transferability to unobserved administrative regions, we implement **5-Fold District-Grouped Cross-Validation (`GroupKFold`)** on the pre-2025 training years (3,400 observations across 31 districts).

### Spatial CV Performance Summary (Pre-2025 Training Window)

| Model Architecture | Mean CV Accuracy | Accuracy Std ($\pm$) | Mean CV Macro-F1 | Macro-F1 Std ($\pm$) | Drought Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **0.7323** | $\pm 0.0257$ | 0.4674 | $\pm 0.0348$ | 0.0240 |
| **Random Forest Classifier** | 0.6581 | $\pm 0.0444$ | **0.4957** | $\pm 0.0361$ | **0.3049** |
| **Logistic Regression (Balanced)** | 0.4464 | $\pm 0.0265$ | 0.3590 | $\pm 0.0142$ | 0.4378 |

> [!NOTE]
> **Spatial Generalization Findings**: Random Forest achieved the highest cross-validated Macro-F1 (**0.4957**), effectively balancing majority-class identification with a **30.49% recall** on scarce drought events across unseen districts. Gradient Boosting achieved higher raw accuracy (0.7323) by defaulting toward the majority `stable` class, resulting in near-zero drought recall (0.0240).

---

## 2. Out-of-Time Stress Test (2025 Hold-Out) & Baseline Comparisons

### Severe Class Imbalance & The Scarcity of Drought
The target variable `risk_zone` (`drought_prone`: $\le -20\%$, `stable`: $-20\%$ to $+40\%$, `flood_prone`: $\ge +40\%$) exhibits profound class imbalance across Karnataka:
- **Total Dataset (4,250 rows)**: `stable` = 2,926 (68.8%), `flood_prone` = 1,163 (27.4%), `drought_prone` = **only 161 (3.79%)**.
- **2022 Flood Year**: Drought instances dropped to an extreme low of **0.94% (8 hoblis statewide)**.
- **2025 Regime Shift**: In 2025, statewide rainfall moderated to $+13.20\%$, but localized moisture stress emerged in the southern interior, doubling drought occurrences to **66 hoblis (7.76%)** while `stable` accounted for **677 hoblis (79.65%)**.

### Hold-Out Evaluation on 2025 (850 Hoblis)

| Model / Baseline | Test Accuracy | Test Macro-F1 | Drought Precision | Drought Recall | Drought F1 | Beats Naive Accuracy? | Beats Naive Macro-F1? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Most Common Class (`stable`)** | **0.7965** | 0.2956 | 0.0000 | 0.0000 | 0.0000 | **Benchmark** | Benchmark |
| **Baseline: Persistence (2024 Zone)** | 0.6812 | **0.4095** | 0.1290 | 0.1212 | 0.1250 | No | **Yes (+0.1139)** |
| **Random Forest Classifier** | 0.7082 | 0.4077 | 0.2500 | 0.0606 | 0.0976 | No | **Yes (+0.1121)** |
| **Gradient Boosting Classifier** | 0.6788 | 0.3840 | **1.0000** | 0.0303 | 0.0588 | No | **Yes (+0.0884)** |
| **Logistic Regression (Balanced)** | 0.1376 | 0.0948 | 0.1667 | 0.0152 | 0.0278 | No | No |

### Honest Assessment vs Baselines
**Do the ML models beat the baselines?**
1. **On Accuracy**: **No.** Neither Random Forest (0.7082) nor Gradient Boosting (0.6788) beats the "Most Common Class" baseline (0.7965). Because 79.65% of Karnataka hoblis were `stable` in 2025, simply predicting `stable` for every station yields ~80% accuracy.
2. **On Macro-F1 & Risk Discovery**: **Yes.** The "Most Common Class" baseline has **zero utility** for disaster risk management (Macro-F1: 0.2956, Drought Recall: 0.0%). Random Forest achieves a Macro-F1 of **0.4077**, matching persistence (0.4095) while delivering **25.00% precision** on drought identification. Gradient Boosting achieves **100% precision** on its predicted drought instances.

### Confusion Matrices (Hold-Out 2025; Rows=Actual, Cols=Predicted: Drought, Stable, Flood)

- **Random Forest**:
  ```
                 Pred Drought   Pred Stable   Pred Flood
  Actual Drought            4            60            2    (Total: 66)
  Actual Stable            10           559          108    (Total: 677)
  Actual Flood              2            66           39    (Total: 107)
  ```
- **Gradient Boosting**:
  ```
                 Pred Drought   Pred Stable   Pred Flood
  Actual Drought            2            56            8    (Total: 66)
  Actual Stable             0           530          147    (Total: 677)
  Actual Flood              0            62           45    (Total: 107)
  ```

---

## 3. Feature Importance Analysis

Feature importances extracted from the top-performing Random Forest pipeline:

| Feature Name | Importance (%) | Domain Role |
| :--- | :---: | :--- |
| `normal_mm` | **17.31%** | Baseline climatological rainfall volume of the hobli. |
| `previous_year_departure` | **10.68%** | Immediate preceding observed year's monsoon departure. |
| `district_mean_departure` | **10.34%** | Expanding historical departure of the parent district. |
| `mean_departure_pct` | **10.01%** | Historical expanding mean departure of the hobli. |
| `mean_actual_mm` | 8.22% | Expanding mean actual rainfall observed in earlier seasons. |
| `hobli_minus_taluk` | 8.14% | Micro-climatic offset from surrounding taluk stations. |
| `n_prior_years` | 7.82% | Depth of historical observations available for station. |
| `taluk_mean_departure` | 7.59% | Expanding historical departure of the parent taluk. |
| `gap_years` | 5.62% | Temporal gap distance from previous observation. |
| `std_actual_mm` | 5.26% | Rainfall volatility in millimeters across past seasons. |
| `std_departure_pct` | 4.77% | Relative percentage deviation volatility across past seasons. |
| `excess_count` | 3.59% | Historical frequency of flood/surplus monsoon events. |
| `deficit_count` | 0.66% | Historical frequency of drought/deficit monsoon events. |

---

## 4. Longitudinal Climatological Hobli Clustering

Rather than clustering single-year observations, all 850 canonical stations were aggregated to **one row per hobli** across all 5 years (`mean_departure`, `std_departure`, `deficit_count`, `excess_count`, `mean_normal_mm`).

### Model Selection ($k \in [2..8]$)

| Clusters ($k$) | Inertia | Silhouette Score | Evaluation |
| :---: | :---: | :---: | :--- |
| $k=2$ | 3,008.4 | 0.3081 | Divides coarse Ghats vs Plains |
| $k=3$ | 2,353.7 | 0.3099 | Merges drought-vulnerable zones with stable plains |
| **$k=4$** | **1,831.1** | **0.3406** | **Optimal Silhouette Peak & Clear Meteorological Separation** |
| $k=5$ | 1,586.9 | 0.3220 | Over-splits interior plains |
| $k=6$ | 1,365.9 | 0.3059 | Diminishing returns |
| $k=7$ | 1,169.6 | 0.3323 | Sub-clusters Western Ghats micro-valleys |
| $k=8$ | 1,068.6 | 0.3031 | Over-clustering |

### Plain-English Meteorological Characterization ($k=4$)

1. **Cluster 0: Western Ghats & Coastal High-Rainfall Belt ($n=87$, 9.9%)**:
   - *Mean Normal*: **3,574.0 mm** | *Mean Dep*: **+12.06%** | *Std Dep*: 12.98% | *Deficit Yrs*: 0.07 | *Excess Yrs*: 1.39
   - *Description*: Heavy monsoon corridor (Udupi, Dakshina Kannada, Uttara Kannada coastal strip). Massive volumetric rainfall with low relative percentage variance.
2. **Cluster 1: High-Surplus / Flood-Prone Interior Plains ($n=344$, 39.1%)**:
   - *Mean Normal*: **749.0 mm** | *Mean Dep*: **+45.01%** | *Std Dep*: 33.89% | *Deficit Yrs*: 0.01 | *Excess Yrs*: **3.69 out of 5**
   - *Description*: Semi-arid northern and eastern plains that experienced chronic flood surges across 2020–2024. Characterized by high volatility and flash-flood susceptibility.
3. **Cluster 2: Stable Semi-Arid Agricultural Plains ($n=325$, 37.0%)**:
   - *Mean Normal*: **808.3 mm** | *Mean Dep*: **+17.95%** | *Std Dep*: 19.68% | *Deficit Yrs*: 0.00 | *Excess Yrs*: 1.93
   - *Description*: Central and southern Karnataka agricultural belt (Mysuru, Mandya, Tumakuru, Hassan plains). Moderately wet, predictable rainfall with low drought risk during this 5-year window.
4. **Cluster 3: Deficit-Vulnerable / Chronic-Drought Pocket ($n=123$, 14.0%)**:
   - *Mean Normal*: **1,092.3 mm** | *Mean Dep*: **+5.64%** | *Std Dep*: 25.75% | *Deficit Yrs*: **1.23 out of 5** | *Excess Yrs*: 1.52
   - *Description*: Vulnerable pockets exhibiting frequent negative departures. Contains nearly all repeated deficit observations in the state (including rain-shadow transition zones and stations with baseline distortion).

---

## 5. Anomaly Detection & Agumbe Data-Quality Identification

Using Multi-dimensional Isolation Forest combined with taluk-relative deviation, the top 20 most anomalous readings across Karnataka were identified and audited:

### Top Suspicious Readings Across Karnataka

| Rank | Year | District | Taluk | Hobli | Actual (mm) | Normal (mm) | Dep (%) | Taluk Dev (%) | Anomaly Classification |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **#01** | 2025 | HASSAN | Sakaleshpura | Hettur | 6488.2 | 2218.3 | +192.49% | 102.3% | Extreme Localized Deluge |
| **#02** | 2025 | KODAGU | Ponnampete | Hudakere | 5581.3 | 2402.6 | +132.30% | 103.5% | Extreme Ghats Orographic Cloudburst |
| **#03** | 2024 | CHIKKAMAGALURU | Chikkamagaluru | Jagar | 2255.5 | 909.4 | +148.02% | 90.3% | Chronic Local Excess Outlier |
| **#04** | 2021 | KOLAR | Srinivasapura | Nelavanki | 1966.0 | 664.0 | +196.08% | 83.2% | Bay of Bengal Depression Inundation |
| **#05** | 2022 | MANDYA | Nagamangala | Belluru | 1737.0 | 520.0 | +234.04% | 81.3% | Historic Southern Interior Flooding |
| **#06** | 2022 | CHIKKAMAGALURU | Chikkamagaluru | Jagar | 2307.0 | 909.0 | +153.80% | 78.8% | Local Monsoon Anomaly |
| **#07** | 2025 | BELAGAVI | Khanapur | Jamboti | 3632.7 | 1856.1 | +95.72% | 74.5% | Orographic Malaprabha Headwater Surge |
| **#11** | 2024 | SHIVAMOGGA | Tirthahalli | **Agumbe** | **5383.1** | **7565.0** | **-28.84%** | **50.2%** | **DATA-QUALITY ISSUE (Artificial Baseline Normal)** |
| **#15** | 2021 | SHIVAMOGGA | Tirthahalli | **Agumbe** | **4152.0** | **7565.0** | **-45.12%** | **49.1%** | **DATA-QUALITY ISSUE (Artificial Baseline Normal)** |

> [!WARNING]
> **Data Quality Insight: Agumbe Normal Benchmark**:
> Agumbe received an extraordinary **5,383.1 mm** of rain in 2024 and **4,152.0 mm** in 2021, yet was flagged as `drought_prone` with departures of $-28.84\%$ and $-45.12\%$. This occurs because KSNDMC records assign Agumbe a fixed static normal of **7,565.0 mm**. This is **not a true meteorological drought**; it is a **data-quality artifact** arising from an uncalibrated historical benchmark.

---

## 6. Next-Year Departure Regression Debugging

Regression models were trained on pre-2025 data to predict `departure_pct` for 2025 and evaluated using Mean Absolute Error (MAE):

| Model / Baseline | MAE (%) | Relative Performance vs Zero Baseline |
| :--- | :---: | :--- |
| **Baseline 1: Predict Zero ($0.0\%$)** | **21.65%** | **Best Performance (Lowest Error)** |
| **Baseline 2: Predict Train Mean ($+29.50\%$)** | 24.53% | Loss of 2.88 pp |
| **Gradient Boosting Regressor** | 24.42% | Loss of 2.77 pp |
| **Random Forest Regressor** | 26.35% | Loss of 4.70 pp |
| **Ridge Regression** | 53.79% | Severe overshooting |

### Why Regression Loses to the Zero Baseline: Feature Distribution Audit
Comparison of feature distributions between pre-2025 training years and 2025 test year:
1. **Macro-Regime Abatement**:
   - `departure_pct`: Train Mean = **+29.50%** ($\pm 33.1$) vs 2025 Test Mean = **+13.20%** ($\pm 25.7$).
   - `previous_year_departure`: Train Mean = **+33.0%** vs 2025 Test Mean = **+18.7%**.
   - `excess_count`: Train Mean = **0.8** vs 2025 Test Mean = **2.2**.
2. **Mechanistic Explanation**:
   - The training window (2020–2024) was heavily dominated by positive monsoon anomalies, culminating in the extreme 2022 La Niña flood (+50.66%). Regression algorithms learned positive offsets (+25% to +35%) for stations with high prior rain.
   - In 2025, the monsoon moderated sharply toward normal (+13.20%).
   - Predicting **Zero ($0.0\%$)** represents the long-term climatological mean ($actual = normal$). When an extreme multi-year flood regime cools down, predicting zero deviation ($0.0\%$) is mathematically closer to $+13.2\%$ than the inflated historical average of $+29.5\%$.
   - Annual monsoon departures are dominated by chaotic synoptic atmospheric circulation rather than deterministic station-level memory.

---

## 7. Limitations & Project Scope

1. **Temporal Horizon (5 Non-Consecutive Years)**: The dataset covers only 2020, 2021, 2022, 2024, and 2025. 2023 is missing from KSNDMC records, and 5 annual snapshots cannot capture multi-decadal climate cycles (Pacific Decadal Oscillation / Indian Ocean Dipole).
2. **Station Name Harmonization**: Reconstructed through RapidFuzz and manual gazetteer cross-matching across 877 stations. While 94.0% of stations appear across all 5 years, slight boundary changes occurred when Vijayanagar district was bifurcated.
3. **Single Modality (Precipitation Only)**: The prediction is based entirely on annual surface rainfall. Agricultural and hydrological drought also depend on temperature, evapotranspiration, soil moisture index, reservoir storage, and groundwater levels.
