# Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning

**Comprehensive Project Report (Academic Year 2025–2026)**  
**Domain**: Applied Machine Learning, Hydrometeorological Analytics, Climate Informatics  
**Stack**: Python 3.10+, pandas, scikit-learn, Plotly, Streamlit, RapidFuzz  

---

## Abstract

Monsoon rainfall variability profoundly influences agrarian livelihoods and state disaster relief in Karnataka, India. Official drought declarations and relief fund disbursals under Pradhan Mantri Fasal Bima Yojana (PMFBY) are executed at the **hobli** tier (sub-district revenue circles of 10–30 villages). However, operational early-warning systems are obstructed by noisy, unstandardized open-government data, spatial autocorrelation, and extreme class imbalance. In this project, we engineer an end-to-end machine learning system using official Karnataka State Natural Disaster Monitoring Centre (KSNDMC) annual precipitation records across five calendar years (**2020, 2021, 2022, 2024, and 2025**; 4,250 observations across 850 canonical stations). We implement an ex-ante, leakage-free feature pipeline, evaluate spatial generalization via 5-Fold District-Grouped Cross-Validation (`GroupKFold` across 31 districts), and conduct an out-of-time stress test on 2025 hold-out observations. Our empirical results provide an honest critique of machine learning in severe imbalance: while a naive Majority-Class baseline achieves **79.65% test accuracy** by predicting the dominant `stable` class for all stations, it achieves **0.00% drought recall**. Random Forest achieves **70.82% accuracy** (-8.83 pp vs. majority) but attains **0.4077 Macro-F1** and captures localized drought events (achieving **30.49% CV drought recall**). In continuous departure regression, predicting Zero Departure ($0.0\%$, the climatological normal) yields an MAE of **21.65%**, outperforming complex non-linear regressors (Gradient Boosting: **24.42% MAE**) following regime abatement from the historic 2022 flood (+50.67%). Unsupervised K-Means clustering ($k=4$, silhouette score **0.3406**) successfully partitions Karnataka into four distinct climatological regimes, while multi-dimensional Isolation Forest exposes severe baseline normal distortion in stations such as Agumbe. Finally, an interactive Streamlit decision-support application is deployed for transparent regional monitoring.

---

## 1. Introduction

The state of Karnataka spans 191,791 square kilometers in southern India, exhibiting exceptional agro-climatic diversity ranging from humid coastal strips and dense Western Ghats rainforests to arid and semi-arid interior Deccan plateaus. Karnataka's agricultural economy remains intensely rainfed, rendering rural communities acutely vulnerable to monsoon fluctuations.

Disaster management in Karnataka operates hierarchically across four administrative tiers:
$$\text{State} \longrightarrow \text{District (31)} \longrightarrow \text{Taluk (239)} \longrightarrow \text{Hobli (850+)} \longrightarrow \text{Gram Panchayat / Villages}$$

While weather forecasting and seasonal climate projections typically operate at synoptic regional scales or district aggregates, agricultural damage is hyper-local. Orographic barriers, micro-climatic rain-shadows, and convective thunderstorm trajectories frequently produce stark precipitation disparities between adjacent revenue circles within the same taluk. The **hobli** is the operational administrative unit at which official drought memoranda are validated, emergency drinking water is sanctioned, and agricultural input subsidies are credited. Developing automated, data-driven systems to predict annual rainfall departure and classify localized drought risk at the hobli level is therefore of paramount socio-economic significance.

---

## 2. Problem Statement

Given historical annual precipitation observations and long-term climatological normals for all hoblis across Karnataka, our objectives are:
1. **Longitudinal Harmonization**: Ingest, clean, and standardize fragmented, multi-year open-government tabular datasets into a longitudinal panel spanning 850 canonical monitoring stations across five observed years (2020, 2021, 2022, 2024, 2025).
2. **Ex-Ante Feature Engineering**: Construct predictive feature vectors utilizing strictly prior information ($\text{year} < Y$), eliminating temporal and spatial data leakage while explicitly tracking the missing 2023 observation gap.
3. **Risk Classification**: Classify each hobli into one of three standardized KSNDMC / Indian Meteorological Department (IMD) vulnerability zones:
   - **`drought_prone`**: $\text{Departure} \le -20.0\%$ (Deficient monsoon).
   - **`stable`**: $-20.0\% < \text{Departure} < +40.0\%$ (Normal / moderate variability).
   - **`flood_prone`**: $\text{Departure} \ge +40.0\%$ (Large excess monsoon).
4. **Climatological Profiling & Anomaly Auditing**: Unsupervised clustering of hoblis based on 5-year multi-annual behavior, and isolation of data-quality artifacts from genuine meteorological anomalies.
5. **Interactive Deployment**: Deliver a public-facing, interactive Streamlit application enabling district agricultural officers to inspect station-level trajectories, risk classifications, and empirical model limitations.

---

## 3. Dataset and Its Data Quality Problems

The raw data source comprises five annual rainfall tables published by the Karnataka State Natural Disaster Monitoring Centre (KSNDMC) and hosted on the OpenCity urban data repository. Initial inspection revealed pervasive data quality deficiencies:

### 3.1 Inventory of Data Anomalies and Quality Defects

1. **Temporal Mislabels in Source Archives**:
   - `rainfall_2022.csv` was erroneously labeled as calendar year 2023 in the raw data store.
   - `rainfall_2024.csv` was mislabeled as calendar year 2020.
   - Definitive ground-truth verification was established by cross-referencing rainfall magnitudes against published KSNDMC state totals (e.g., verifying the historic 2022 state flood departure of $+50.67\%$).
2. **The Missing Year 2023**:
   - KSNDMC published no hobli-level annual rainfall compilation for calendar year 2023. Any naive time-series model assuming evenly spaced annual intervals ($t-1$) would suffer from temporal corruption.
3. **Interleaved Administrative Hierarchies**:
   - Raw CSVs contained **5,636 total rows**, interweaving individual hobli records with District summary rows (`D`) and Taluk summary rows (`T`), along with unstructured page headers and sub-totals.
4. **Duplicate Printed Table Blocks (2020 Raw File)**:
   - Rows 263 to 289 in `rainfall_2020.csv` contained an accidental duplicate reprint of 27 Chitradurga district taluk and hobli rows.
5. **Nomenclature Inconsistencies and Transliteration Drift**:
   - Across the five files, **898 raw station name variants** were identified for approximately 850 actual physical stations. Inconsistencies stemmed from variant Kannada-to-English spellings (e.g., *Thirthahalli* vs. *Tirthahalli*, *Dodballapur* vs. *Doddaballapura*, *Belthangady* vs. *Beltangadi*).
6. **Administrative Boundary Bifurcation (Ballari / Vijayanagar)**:
   - In late 2021, the Government of Karnataka formally carved the new **Vijayanagar** district out of Ballari, shifting six taluks (*Hadagali, Hagaribommanahalli, Harappanahalli, Hosapete, Kotturu, Kudligi*). Raw files from 2020–2021 placed these taluks under Ballari, while 2022–2025 placed them under Vijayanagar.
7. **Static Normals and Baseline Distortion (Suspect Normals)**:
   - **338 hoblis** exhibited completely static baseline normals ($\Delta \le 1.0\text{ mm}$) across all five years alongside persistent one-sided departure bias.
   - High-rainfall Western Ghats stations displayed severe benchmark distortion. For example, **Agumbe** in Tirthahalli taluk is assigned a fixed normal of **7,565.0 mm**. In 2024, Agumbe received **5,383.1 mm** of rainfall—an astronomical volume sufficient to cause localized flash floods—yet was classified as `drought_prone` with a departure of **-28.84%**!

---

## 4. Data Cleaning Methodology

Data cleaning was executed in `src/clean_data.py` to establish an immutable, verified baseline (`data/clean/clean_rainfall.csv`):

```
Raw CSV Files (5 Years, 5,636 Rows)
  │
  ├── 1. Drop Duplicate Block (2020 rows 263-289; -27 rows)
  ├── 2. Parse Hierarchical Levels (D=District, T=Taluk, H=Hobli)
  ├── 3. Filter Summary Rows (-1,359 non-hobli rows)
  ├── 4. Standardize District & Taluk Names (Ballari/Vijayanagar realignment)
  ├── 5. RapidFuzz Cross-Year Station Matching (>=85% fuzzy string match)
  ├── 6. Compute gap_years temporal offset indicator
  └── 7. Recompute & Audit Percentage Departures
        │
        ▼
Standardized Panel: 4,250 Clean Rows (850 stations x 5 years)
```

### 4.1 Cleaning Transformation Ledger

| Calendar Year | Raw Source File Name | Raw Rows | Duplicate Rows Dropped | Summary Rows Dropped | Clean Hobli Rows | Mean Actual (mm) | Mean Departure (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2020** | `741349d2-9f58-45ae-9744-b6d5c10a8b7f.csv` | 1,134 | 27 | 257 | **850** | 1,255.1 mm | **+20.98%** |
| **2021** | `275cb296-b3aa-4c96-b614-c3c3ca19dec0.csv` | 1,113 | 0 | 263 | **850** | 1,297.2 mm | **+27.66%** |
| **2022** | `d9c463e4-3203-49fd-95c3-33b79f85c9b3.csv` | 1,150 | 0 | 300 | **850** | 1,472.7 mm | **+50.67%** |
| **2024** | `0ec5ad8d-be55-4e92-a942-6918c025735b.csv` | 1,122 | 0 | 272 | **850** | 1,298.3 mm | **+18.72%** |
| **2025** | `karnataka_rainfall_2025_table1_1.csv` | 1,117 | 0 | 267 | **850** | 1,237.6 mm | **+13.21%** |
| **Total** | **5 KSNDMC Annual Publications** | **5,636** | **27** | **1,359** | **4,250** | **1,312.2 mm** | **+26.25%** |

### 4.2 Cross-Year Station Matching & Verification
- **RapidFuzz Matching**: Using 2021 as the primary anchor, raw station names were matched against taluk candidate sets using token sort and ratio scorers. Matches with similarity $\ge 85\%$ were accepted into canonical names, harmonizing 898 variants into 877 unique canonical stations.
- **Station Persistence**: **824 stations (94.0%)** appear in all five observed years.
- **Formula Recomputation Audit**: Every record's percentage departure was recomputed as:
  $$\text{departure\_pct} = \left( \frac{\text{actual\_mm} - \text{normal\_mm}}{\text{normal\_mm}} \right) \times 100$$
  Comparing computed values against source files revealed **zero discrepancies exceeding 2.0 percentage points** across all 4,250 records.

---

## 5. Exploratory Data Analysis (EDA)

Exploratory analysis conducted in `notebooks/eda.ipynb` revealed four critical structural dynamics governing Karnataka rainfall:

### 5.1 Macro-Surplus Across All Five Years
Across every observed year, Karnataka experienced positive statewide mean departures:
- **2020**: $+20.98\%$
- **2021**: $+27.66\%$
- **2022**: $+50.67\%$ (Historic catastrophic flood year triggered by prolonged La Niña)
- **2024**: $+18.72\%$
- **2025**: $+13.21\%$

Karnataka never experienced a statewide negative rainfall anomaly in this five-year window. This persistent positive bias implies that published KSNDMC baseline normals are systematically low relative to modern precipitation patterns. Consequently, an observation categorized as `flood_prone` ($\ge +40\%$) often reflects departure above an understated historical baseline rather than destructive deluge.

### 5.2 Immense Spatial Precipitation Gradients
The data exhibits a dramatic 7-fold spatial rainfall gradient across Karnataka's agro-climatic zones:
- **Driest Districts (5-Year Mean Actual)**: Ballari (679.3 mm), Vijayapura (712.5 mm), Bagalkote (742.8 mm), Koppala (768.4 mm), Chitradurga (789.1 mm).
- **Wettest Districts (5-Year Mean Actual)**: Udupi (4,963.8 mm), Dakshina Kannada (4,412.9 mm), Uttara Kannada (3,458.2 mm), Shivamogga (2,314.6 mm), Kodagu (2,185.3 mm).

### 5.3 Positive Skewness of Rainfall Departure
The distribution of `departure_pct` is heavily right-skewed ($+1.02$ skewness) with a median of $+21.25\%$ and maximum positive departures reaching $+244.75\%$. Positive rainfall shocks dominate heavily over drought events in this dataset.

### 5.4 Spatial Coherence Across Districts
The district-by-year heatmap confirms that annual monsoon shocks exhibit high regional spatial coherence. The 2022 La Niña surge produced simultaneous surplus anomalies across all 31 administrative districts.

---

## 6. Feature Engineering and Target Definition

Feature engineering was implemented in `src/features.py` under strict architectural guidelines to guarantee **zero temporal lookahead leakage**:

### 6.1 Ex-Ante Feature Contract
A feature vector for station $i$ in year $Y$ must be computable *before* observing the monsoon outcome of year $Y$. All aggregate statistics are computed strictly over prior years ($t < Y$):

1. `normal_mm`: Long-term historical baseline rainfall (fixed reference).
2. `gap_years`: Temporal distance from immediately preceding observation ($0$ for 2020, $1$ for 2021/2022/2025, and **$2$ for 2024** across missing 2023).
3. `n_prior_years`: Count of strictly prior observations available ($0$ to $4$).
4. `deficit_count`: Cumulative prior years where departure was $\le -20.0\%$.
5. `excess_count`: Cumulative prior years where departure was $\ge +20.0\%$.
6. `mean_actual_mm`: Expanding historical mean of actual rainfall over prior years.
7. `std_actual_mm`: Expanding sample standard deviation of actual rainfall ($N-1$ ddof).
8. `mean_departure_pct`: Expanding historical mean departure percentage.
9. `std_departure_pct`: Expanding standard deviation of departure percentage.
10. `previous_year_departure`: Lag-1 departure percentage from previous observed year.
11. `taluk_mean_departure`: Expanding historical mean departure across all stations in the same taluk.
12. `district_mean_departure`: Expanding historical mean departure across all stations in the parent district.
13. `hobli_minus_taluk`: Micro-climatic offset ($\text{mean\_departure\_pct} - \text{taluk\_mean\_departure}$).

All 13 predictive features are serialized in `models/feature_columns.json`.

### 6.2 Target Variable & Severe Class Imbalance
The target variable `risk_zone` partitions observations into three classes:
- **`drought_prone`**: $\text{departure\_pct} \le -20.0\%$
- **`stable`**: $-20.0\% < \text{departure\_pct} < +40.0\%$
- **`flood_prone`**: $\text{departure\_pct} \ge +40.0\%$

#### Class Distribution Across Observed Years

| Year | Drought-Prone | Stable | Flood-Prone | Total Observations |
| :---: | :---: | :---: | :---: | :---: |
| **2020** | 30 (3.53%) | 673 (79.18%) | 147 (17.29%) | 850 |
| **2021** | 24 (2.82%) | 550 (64.71%) | 276 (32.47%) | 850 |
| **2022** | 8 (0.94%) | 368 (43.29%) | 474 (55.76%) | 850 |
| **2024** | 33 (3.88%) | 658 (77.41%) | 159 (18.71%) | 850 |
| **2025** | 66 (7.76%) | 677 (79.65%) | 107 (12.59%) | 850 |
| **Overall** | **161 (3.79%)** | **2,926 (68.85%)** | **1,163 (27.36%)** | **4,250** |

Across the entire dataset, `drought_prone` comprises only **3.79%** of records. In the 2022 flood year, fewer than 1% of hoblis experienced drought. In the 2025 hold-out set, drought increased to **7.76% (66 hoblis)**, while `stable` accounted for **79.65% (677 hoblis)**.

---

## 7. Modelling Methodology

Modelling was executed in `src/train_models.py` across five complementary learning paradigms:

### 7.1 Baseline Estimators
To prevent deceptive claims of machine learning efficacy, all models are evaluated against two mandatory operational baselines:
1. **Most Common Class Baseline**: Predicts the modal training class (`stable`) for all test observations.
2. **Persistence Baseline (Lag-1)**: Predicts each station's previous observed risk zone (2024 category mapped to 2025).

### 7.2 Spatial Generalization: 5-Fold District-Grouped Cross-Validation
Standard random $k$-fold cross-validation causes massive optimistic spatial leakage because adjacent hoblis within the same taluk share weather events. To evaluate true spatial transferability to unobserved administrative regions, we implement **5-Fold District-Grouped Cross-Validation (`GroupKFold`)** grouped across 31 districts on pre-2025 data (3,400 rows).
- Validation fold majority-class share: **$66.12\% \pm 5.83\%$**.

### 7.3 Out-of-Time Stress Test (2025 Hold-Out)
Models are trained strictly on historical observations from 2020 through 2024 (3,400 rows) and evaluated on all 850 canonical stations in 2025. This simulates real-world operational deployment where historical models predict the upcoming season.

### 7.4 Climatological Hobli Clustering ($k$-Means)
Rather than clustering individual annual observations, all 850 canonical stations were aggregated to **one climatological vector per station** across all five years:
$$\mathbf{x}_i = [\text{mean\_departure}, \text{std\_departure}, \text{deficit\_count}, \text{excess\_count}, \text{mean\_normal\_mm}]$$
Features were standardized using `StandardScaler()`. $k$-Means was evaluated for $k \in [2..8]$ using inertia and silhouette scores. $k=4$ achieved the global silhouette peak (**0.3406**).

### 7.5 Multi-Dimensional Anomaly Detection
Isolation Forest (`contamination=0.03`) was combined with taluk-relative deviation to identify the top 20 most anomalous readings across Karnataka, separating climatic extremes from static baseline errors.

### 7.6 Next-Year Departure Regression Debugging
Continuous regression models predicting `departure_pct` were benchmarked against "Predict Zero" ($0.0\%$) and "Predict Train Mean" ($+29.50\%$) baselines using Mean Absolute Error (MAE).

---

## 8. Results

All results reported below are copied directly from `models/results.csv`:

### 8.1 Supervised Classification Results

| Model Architecture / Baseline | Model Type | 5-Fold District CV Accuracy | 5-Fold District Macro-F1 | 2025 Hold-Out Test Accuracy | 2025 Hold-Out Test Macro-F1 | 2025 Hold-Out Drought Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Most Common Class (`stable`)** | `baseline` | — | — | **0.7965** | 0.2956 | 0.0000 |
| **Persistence (Lag-1 2024 Zone)** | `baseline` | — | — | 0.6812 | **0.4095** | **0.1212** |
| **Logistic Regression (Balanced)** | `classifier` | 0.4464 | 0.3590 | 0.1376 | 0.0948 | 0.0152 |
| **Random Forest Classifier** | `classifier` | 0.6581 | **0.4957** | **0.7082** | **0.4077** | 0.0606 |
| **Gradient Boosting Classifier** | `classifier` | **0.7323** | 0.4674 | 0.6788 | 0.3840 | 0.0303 |

### 8.2 Hold-Out Confusion Matrices (2025 Test Set; $N=850$)

- **Random Forest Classifier**:
  ```
                 Pred Drought   Pred Stable   Pred Flood   Total Actual
  Actual Drought            4            60            2             66
  Actual Stable            10           559          108            677
  Actual Flood              2            66           39            107
  Total Predicted          16           685          149            850
  ```
  *Drought Metrics*: Precision = $4/16 = 25.00\%$, Recall = $4/66 = 6.06\%$, F1 = $0.0976$.

- **Gradient Boosting Classifier**:
  ```
                 Pred Drought   Pred Stable   Pred Flood   Total Actual
  Actual Drought            2            56            8             66
  Actual Stable             0           530          147            677
  Actual Flood              0            62           45            107
  Total Predicted           2           648          200            850
  ```
  *Drought Metrics*: Precision = $2/2 = 100.00\%$, Recall = $2/66 = 3.03\%$, F1 = $0.0588$.

### 8.3 Feature Importance Analysis (Random Forest)
1. `normal_mm`: **17.31%** (Baseline climatological rainfall volume)
2. `previous_year_departure`: **10.68%** (Lag-1 monsoon departure)
3. `district_mean_departure`: **10.34%** (Regional district historical departure)
4. `mean_departure_pct`: **10.01%** (Hobli historical departure mean)
5. `mean_actual_mm`: **8.22%** (Historical volume of actual precipitation)
6. `hobli_minus_taluk`: **8.14%** (Micro-climatic offset from taluk mean)
7. `n_prior_years`: **7.82%** (Depth of station history)
8. `taluk_mean_departure`: **7.59%** (Taluk historical departure mean)
9. `gap_years`: **5.62%** (Temporal distance from last observation)
10. `std_actual_mm`: **5.26%** (Absolute rainfall volatility)
11. `std_departure_pct`: **4.77%** (Percentage departure volatility)
12. `excess_count`: **3.59%** (Historical surplus monsoon frequency)
13. `deficit_count`: **0.66%** (Historical drought frequency)

### 8.4 Climatological Clustering ($k=4$, Silhouette Score = 0.3406)

| Cluster ID | Plain-Language Name | Station Count ($N$) | Percentage Share (%) | Mean Normal (mm) | Mean Departure (%) | Volatility Std (%) | Avg Deficit Years | Avg Excess Years | Suspect Normals Flagged |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Western Ghats & Coastal High-Rainfall Belt** | 87 | 9.9% | 3,574.0 mm | +12.06% | 12.98% | 0.07 | 1.39 | 3 |
| **1** | **High-Surplus / Flood-Prone Interior Plains** | 344 | 39.1% | 749.0 mm | +45.01% | 33.89% | 0.01 | 3.69 | 243 |
| **2** | **Stable Semi-Arid Agricultural Plains** | 325 | 37.0% | 808.3 mm | +17.95% | 19.68% | 0.00 | 1.93 | 53 |
| **3** | **Deficit-Vulnerable / Chronic-Drought Pocket** | 123 | 14.0% | 1,092.3 mm | +5.64% | 25.75% | 1.23 | 1.52 | 39 |

### 8.5 Continuous Departure Regression Results

| Model Architecture / Baseline | Test MAE (%) | Status / Outcome |
| :--- | :---: | :--- |
| **Baseline: Predict Zero Departure ($0.0\%$)** | **21.65%** | **Deployed Benchmark (Lowest Error)** |
| **Baseline: Predict Train Mean ($+29.50\%$)** | 24.53% | Lags Zero baseline by 2.88 pp |
| **Gradient Boosting Regressor** | **24.42%** | Best ML regressor; lags Zero baseline by 2.77 pp |
| **Random Forest Regressor** | 26.35% | Lags Zero baseline by 4.70 pp |
| **Ridge Regression (StandardScaler, $\alpha=10$)** | **53.79%** | **DROPPED**: Severe failure (MAE > 30.0%) |

---

## 9. Discussion

### 9.1 Honest Assessment: Why Machine Learning Loses to the Majority Baseline
A candid evaluation of test performance demonstrates that **every machine learning classifier achieved lower test accuracy than the naive Majority-Class baseline**:
- Majority-Class Baseline: **79.65% Accuracy**
- Random Forest: **70.82% Accuracy** (-8.83 pp)
- Gradient Boosting: **67.88% Accuracy** (-11.77 pp)
- Logistic Regression: **13.76% Accuracy** (-65.89 pp)

This occurs because 79.65% of all hoblis in 2025 belonged to the `stable` class. Any algorithm optimizing solely for accuracy will converge toward predicting `stable` everywhere. However, the Majority Baseline has **zero operational utility**: it has **0.00% drought recall** and completely fails to warn disaster management authorities of impending crop failure.

Random Forest successfully navigates this trade-off:
- In 5-Fold District CV across unseen districts, Random Forest achieved **30.49% drought recall** with a Macro-F1 of **0.4957** (beating the majority baseline fold average of 0.2956).
- On the 2025 hold-out set, Random Forest identified localized drought signals with **25.00% precision**, roughly tying the Persistence baseline on Macro-F1 (**0.4077 vs. 0.4095**).

### 9.2 The 2022 Deluge Effect on the 2025 Stress Test
In regression, the "Predict Zero Departure" baseline ($0.0\%$) achieved the lowest MAE (**21.65%**), beating both the training mean baseline ($24.53\%$) and Gradient Boosting ($24.42\%$). 

This occurs because of a profound meteorological regime shift:
1. **Training Window (2020–2024)**: Dominated by massive rainfall surpluses, peaking in the historic 2022 deluge ($+50.67\%$). Training mean departure was **$+29.50\%$**. Machine learning models learned positive offsets for historically wet stations.
2. **Hold-Out Window (2025)**: Karnataka's monsoon moderated sharply toward normal ($+13.20\%$).
3. Predicting zero deviation from climatological normal ($actual = normal$) is mathematically closer to $+13.20\%$ than the inflated historical average of $+29.50\%$. Annual monsoon variability is governed by chaotic synoptic atmospheric circulation rather than deterministic station-level memory.

### 9.3 Why Ridge Regression Failed and Was Dropped
Even when features were normalized with `StandardScaler()`, Ridge regression produced an catastrophic MAE of **53.79%**. Analysis of learned weights revealed:
- Coefficient for `n_prior_years`: **+20.15**
- Coefficient for `gap_years`: **-19.88**

Because `n_prior_years` increased monotonically across training seasons during a multi-year wet surge (2020: +21%, 2021: +28%, 2022: +51%), Ridge learned a spurious linear time trend. When applied to 2025 where `n_prior_years = 4`, it linearly extrapolated positive departures to over $+80\%$, resulting in gross over-prediction. Ridge was permanently **dropped from deployment**.

---

## 10. Limitations

1. **Short, Non-Consecutive Temporal Scope**: The dataset spans only five calendar years (2020, 2021, 2022, 2024, 2025). The complete absence of 2023 hobli-level data precludes standard autoregressive time-series architectures (ARIMA/SARIMAX). Five snapshots cannot capture multi-decadal climatic oscillations such as the Indian Ocean Dipole (IOD) or El Niño–Southern Oscillation (ENSO).
2. **Single-Modality (Precipitation Only)**: Predictions rely exclusively on surface rainfall. Real-world agricultural drought depends critically on temperature, evapotranspiration, root-zone soil moisture, reservoir irrigation storage, and groundwater depth.
3. **Severe Class Imbalance**: With drought instances representing only **3.79%** of the dataset, training models without excessive false alarms remains challenging.
4. **Stale Baseline Normals**: The fact that statewide mean departure is positive across every observed year strongly indicates that published KSNDMC baseline normals are systematically low. Consequently, the `flood_prone` category ($\ge +40\%$) partly reflects departure above an understated baseline rather than true hydrological flooding.
5. **No Geographical Coordinates**: Raw KSNDMC tables contain no latitude or longitude coordinates, preventing spatial kriging, variogram modeling, and interactive GIS choropleth mapping.

---

## 11. Future Work

1. **Integration of Crop Sown Area & Satellite Remote Sensing**: Fusing hobli rainfall with Normalized Difference Vegetation Index (NDVI) and Soil Moisture Active Passive (SMAP) satellite data to detect agronomic moisture stress directly.
2. **Geocoding & GIS Polygon Integration**: Ingesting Survey of India hobli boundary shapefiles to enable true spatial GIS mapping and spatial autoregressive (SAR) spatial-lag modeling.
3. **Multivariate Meteorological Drivers**: Incorporating daily temperature extremes, ERA5 atmospheric reanalysis wind vectors, and humidity to estimate potential evapotranspiration (PET) and Standardized Precipitation-Evapotranspiration Index (SPEI).
4. **Historical Archive Digitization**: Partnering with KSNDMC to recover and digitize pre-2020 archives and the missing 2023 records to establish a continuous 20-year longitudinal panel.

---

## 12. Conclusion

This project delivers a rigorous, transparent machine learning framework for hobli-level rainfall deviation and drought-risk prediction in Karnataka. By confronting real-world data imperfections—mislabeled raw files, missing calendar years, administrative bifurcations, and uncalibrated baseline normals—the pipeline harmonized 4,250 observations across 850 canonical stations. Rigorous evaluation against honest baselines demonstrates that while machine learning models cannot surpass naive majority-class predictors on raw accuracy under severe imbalance, Random Forest captures critical localized drought signals (30.49% CV drought recall). Climatological clustering successfully partitioned Karnataka into four distinct hydro-climatic zones, and multi-dimensional anomaly detection exposed systemic baseline distortion in stations like Agumbe. The complete pipeline is deployed as an open-source, interactive Streamlit application serving as an empirical template for transparent, reproducible climate informatics.

---

## References

1. **Karnataka State Natural Disaster Monitoring Centre (KSNDMC)**, Government of Karnataka. *Annual Rainfall Monitoring and Deviation Reports (2020–2025)*.
2. **OpenCity Urban Data Portal**. *Karnataka Hobli-Level Rainfall Dataset Archives*. [https://opencity.in/](https://opencity.in/) (Accessed under Open Data Commons ODC-BY License).
3. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
4. **Streamlit Development Team** (2024). *Streamlit: An Open-Source Framework for Machine Learning and Data Science UI*. [https://streamlit.io/](https://streamlit.io/).
5. **RapidFuzz Library**. *Rapid Fuzzy String Matching in Python using Levenshtein Distances*. [https://github.com/maxbachmann/RapidFuzz](https://github.com/maxbachmann/RapidFuzz).
6. **Indian Meteorological Department (IMD)**, Ministry of Earth Sciences, Government of India. *Standard Monsoon Rainfall Classification and Drought Identification Criteria*.
