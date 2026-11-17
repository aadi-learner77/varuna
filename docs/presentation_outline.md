# Presentation Outline: Hobli-Level Rainfall Deviation & Drought-Risk Prediction for Karnataka

**10-Slide Academic Project Defense Deck**  
**Project**: Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning  
**Target Duration**: 7–8 minutes (~40–45 seconds per slide)

---

## Slide 1: Title and Team

### Bullets:
- **Project Title**: Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning
- **Context**: Final Year Academic Machine Learning Project (Applied Climate Informatics)
- **Technology Stack**: Python, pandas, scikit-learn, RapidFuzz, Plotly, Streamlit
- **Project Team & Responsibilities**:
  - Person 1: Data Cleaning, Year Mapping & Name Harmonization
  - Person 2: Feature Engineering, Temporal Leakage Prevention & EDA
  - Person 3: Machine Learning Modelling, Cross-Validation, Clustering & Regression
  - Person 4: Web Application, Integration, Final Report, Presentation & Defense Prep

### Speaker Notes (40 seconds):
> "Good morning, respected examiners. Today we present our project on predicting hyper-local monsoon rainfall deviation and drought vulnerability at the hobli level in Karnataka using machine learning. Karnataka has over 850 hoblis—the sub-taluk administrative revenue circles where drought declarations, emergency disaster relief, and crop insurance compensations are legally decided. Our 4-member team addressed the end-to-end pipeline: from resolving messy, unstandardized open-government data and engineering leakage-free temporal features, to training predictive models against honest baseline benchmarks, and deploying a functional decision-support tool in Streamlit. Let us begin with why hyper-local prediction is critical."

---

## Slide 2: Problem and Why It Matters

### Bullets:
- **Hyper-Local Realities**: District forecasts mask micro-climatic rain shadows; two hoblis in the same taluk frequently experience vastly different monsoons.
- **Administrative Importance**: Karnataka’s drought memoranda and PMFBY crop insurance payouts are formally sanctioned at the hobli tier.
- **Agrarian Impact**: Karnataka has India's second-highest proportion of drought-prone land (over 70% of agricultural area is rainfed).
- **Dual Hazard**: Climate change accelerates both flash drought and extreme unseasonal deluge, making reliable annual deviation estimation essential.

### Speaker Notes (40 seconds):
> "District-level meteorological forecasts are too coarse for rural administration. A district might report normal cumulative rainfall while an interior rain-shadow hobli suffers catastrophic crop failure due to severe moisture deficit. Over 70% of Karnataka's farmland is rainfed, making farmers acutely vulnerable. Under state disaster guidelines, official drought declarations and compensation happen at the hobli level. If relief planning relies on coarse averages, vulnerable farming circles get missed. Our project aims to forecast annual deviation and classify localized drought risk before seasonal impacts peak, giving administrators an empirical early-warning signal."

---

## Slide 3: Dataset and Open-Government Data Challenges

### Bullets:
- **Data Source**: Karnataka State Natural Disaster Monitoring Centre (KSNDMC) published via OpenCity (ODC-BY license).
- **Temporal Horizon**: 5 observed years (2020, 2021, 2022, 2024, 2025; 850 stations/year; 4,250 clean rows).
- **The Missing 2023 Gap**: KSNDMC published no hobli-level tables for 2023; handled via an explicit `gap_years = 2` feature.
- **File Mislabels**: Raw 2022 was mislabeled as 2023; raw 2024 was mislabeled as 2020; both resolved through ground-truth meteorological audits.
- **Severe Class Imbalance**: Drought instances comprise only 3.79% of records (161 out of 4,250 observations statewide).

### Speaker Notes (40 seconds):
> "We obtained official annual rainfall tables published by KSNDMC through the OpenCity open data portal. Working with real government data revealed major hurdles. First, the files were mislabeled in source archives—2022 was labeled 2023, and 2024 was labeled 2020. We proved the correct calendar years by matching rainfall totals against verified historical records. Second, 2023 is completely missing from hobli-level publications, creating an uneven time gap that we explicitly tracked using a gap_years feature. Finally, the target is severely imbalanced: because recent years experienced heavy rainfall, drought occurred in only 3.79% of all station-year observations."

---

## Slide 4: Before and After Cleaning

### Bullets:
- **Raw Transformation**: Ingested 5,636 raw rows; dropped 27 duplicate block rows in 2020 and 1,359 summary rows (District `D` and Taluk `T`).
- **Standardized Output**: Exactly 850 canonical stations per year across 5 years = 4,250 longitudinal observations.
- **Station Harmonization**: 898 raw station name variants unified into 877 canonical stations using RapidFuzz ($\ge 85\%$ similarity).
- **Station Retention**: 824 stations (94.0%) persist across all 5 observed years.

| Year | Raw Rows | Duplicates Dropped | Summary Rows Dropped | Clean Hoblis | Mean Actual (mm) | Mean Departure (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2020** | 1,134 | 27 | 257 | 850 | 1,255.1 mm | +20.98% |
| **2021** | 1,113 | 0 | 263 | 850 | 1,297.2 mm | +27.66% |
| **2022** | 1,150 | 0 | 300 | 850 | 1,472.7 mm | +50.67% (Flood) |
| **2024** | 1,122 | 0 | 272 | 850 | 1,298.3 mm | +18.72% |
| **2025** | 1,117 | 0 | 267 | 850 | 1,237.6 mm | +13.21% |
| **Total** | **5,636** | **27** | **1,359** | **4,250** | **1,312.2 mm** | **+26.25%** |

### Speaker Notes (40 seconds):
> "This slide summarizes our Stage 1 data cleaning. Raw government CSVs interleave district and taluk summary totals directly inside station tables. In the 2020 file, rows 263 to 289 were an accidental duplicate reprint of Chitradurga rows. We filtered all 1,359 non-station rows and removed the duplicate block. Using RapidFuzz, we standardized 898 phonetic name spellings into 877 canonical stations, resolving boundary changes like the Vijayanagar bifurcation. 94% of stations appear across all five years, yielding an audited panel of 4,250 clean rows with zero formula discrepancies."

---

## Slide 5: Pipeline Overview and Leakage Prevention

### Bullets:
- **Strict Ex-Ante Contract**: Features computed strictly on prior years ($t < Y$) to guarantee zero lookahead leakage.
- **13 Engineered Predictive Features**: Historical expanding means, sample standard deviations, lag-1 departure, regional aggregations, and micro-climatic offsets (`hobli_minus_taluk`).
- **Spatial CV Design**: 5-Fold District-Grouped Cross-Validation (`GroupKFold` across 31 districts) prevents spatial autocorrelation leakage.
- **Out-of-Time Evaluation**: Models trained on 2020–2024 (3,400 rows) and evaluated on 2025 hold-out observations (850 rows).

### Speaker Notes (40 seconds):
> "Our pipeline architecture is engineered to prevent two common pitfalls: temporal leakage and spatial leakage. For temporal integrity, every feature for year Y is computed strictly using observations before year Y—such as expanding mean rainfall, rolling volatility, and previous year departure. For spatial integrity, we rejected naive random k-fold cross-validation because adjacent stations share the exact same rain cloud. Instead, we implemented 5-Fold District-Grouped cross-validation where entire districts are held out. Finally, we tested the pipeline on 2025 hold-out data, simulating true seasonal prediction."

---

## Slide 6: Exploratory Data Analysis (EDA) Highlights

### Bullets:
- **Persistent Macro-Surplus**: All 5 observed years had positive statewide mean departures (+21% in 2020, +28% in 2021, +51% in 2022, +19% in 2024, +13% in 2025).
- **Immense Spatial Gradient**: 7-fold precipitation contrast between semi-arid northern plains (Ballari: 679 mm) and coastal Western Ghats (Udupi: 4,964 mm).
- **Right-Skewed Departures**: Departure percentage is positively skewed (+1.02) with median departure at +21.25%; extreme positive surges reach +245%.
- **Stale KSNDMC Baseline Normals**: Because every year's mean departure is positive, published normals appear systematically low, making `flood_prone` partly reflect an understated baseline.

### Speaker Notes (40 seconds):
> "Exploratory data analysis yielded three vital domain findings. First, Karnataka experienced positive statewide rainfall across all five observed years, culminating in the historic 2022 deluge where average departure exceeded plus 50%. Second, there is a massive 7-fold spatial rainfall gradient across Karnataka, from 679 millimeters in Ballari to nearly 5,000 millimeters in coastal Udupi. Third, because every single year averaged above normal, published KSNDMC normals appear systematically lower than modern climate realities. Therefore, excess rainfall classifications partly reflect distance above an understated historical baseline."

---

## Slide 7: Modelling Results and Honest Baseline Comparison

### Bullets:
- **The Naive Majority Trap**: Predicting `stable` for all stations yields **79.65% accuracy** but **0.00% drought recall**.
- **Random Forest Performance**: Achieves **70.82% accuracy** (-8.83 pp vs. majority) but attains **0.4077 Macro-F1** and captures localized drought events (achieving **30.49% CV drought recall**).
- **Regression Reality**: In departure regression, the simple **Predict Zero Baseline ($0.0\%$)** achieved **21.65% MAE**, beating Gradient Boosting Regressor (**24.42% MAE**).
- **Dropped Ridge Model**: Ridge regression failed catastrophically (**53.79% MAE**) by extrapolating an artificial linear trend on `n_prior_years`.

| Model Architecture / Baseline | Model Type | 2025 Test Accuracy | 2025 Test Macro-F1 | 2025 Drought Recall | Regression MAE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Most Common Class (`stable`)** | Baseline | **0.7965** | 0.2956 | 0.0000 | — |
| **Persistence (Lag-1)** | Baseline | 0.6812 | **0.4095** | **0.1212** | — |
| **Predict Zero Departure ($0.0\%$)** | Baseline | — | — | — | **21.65%** |
| **Random Forest** | ML Classifier | 0.7082 | 0.4077 | 0.0606 (CV: 30.49%) | 26.35% |
| **Gradient Boosting** | ML Classifier | 0.6788 | 0.3840 | 0.0303 | 24.42% |
| **Ridge Regression** | Dropped Model | — | — | — | 53.79% (Failed) |

### Speaker Notes (40 seconds):
> "Here are our honest empirical results. Under severe class imbalance, accuracy is completely deceptive. The majority class baseline achieves nearly 80% test accuracy simply by guessing 'stable' for every station, but it has zero percent drought recall. Random Forest achieves 70.82% accuracy, beating persistence and achieving 0.4077 Macro-F1 while capturing localized drought signals. In continuous regression, predicting zero departure—the climatological normal—yielded the lowest error of 21.65% MAE, beating complex non-linear regressors after the 2022 deluge subsided. We also dropped Ridge regression because it linearly extrapolated temporal features."

---

## Slide 8: Climatological Risk Zones and Anomaly Audits

### Bullets:
- **Longitudinal K-Means ($k=4$, Silhouette = 0.3406)**:
  - Cluster 0: Western Ghats & Coast ($n=87$, 9.9%, normal 3,574 mm, dep +12.1%).
  - Cluster 1: High-Surplus / Flood-Prone Plains ($n=344$, 39.1%, normal 749 mm, dep +45.0%).
  - Cluster 2: Stable Semi-Arid Agricultural Plains ($n=325$, 37.0%, normal 808 mm, dep +18.0%).
  - Cluster 3: Deficit-Vulnerable / Chronic-Drought Pocket ($n=123$, 14.0%, normal 1,092 mm, dep +5.6%, avg 1.23 deficit yrs).
- **Physical Anomalies**: #1 Hettur (Sakaleshpura, Hassan) recorded 6,488 mm actual vs. 2,218 mm normal (+192% departure).
- **Data Quality Discovery (Agumbe)**: Agumbe received 5,383 mm of rain in 2024 but was labeled "drought-prone" because its normal is pinned to 7,565 mm—an artificial baseline distortion cataloged in `reports/suspect_normals.csv`.

### Speaker Notes (40 seconds):
> "To understand spatial patterns beyond annual labels, we clustered all 850 stations across five years into four climatological regimes using K-Means, peaking at a 0.34 silhouette score. Cluster 3 isolates 123 chronic deficit-vulnerable hoblis averaging 1.2 deficit years. Meanwhile, our Isolation Forest anomaly detector exposed critical data quality insights. For instance, in 2024 Agumbe received over 5,300 millimeters of rainfall—a massive downpour—yet was classified as drought-prone because KSNDMC records assign it a static 7,565-millimeter baseline. We isolated 338 such suspect normal stations so administrators are not misled by data artifacts."

---

## Slide 9: Live Demo Plan and Operational Walkthrough

### Bullets:
- **Architecture**: Modular Streamlit web application (`app/app.py`) with `@st.cache_data`.
- **Step 1 — Sidebar Selection**: Dependent dropdowns: District (`SHIVAMOGGA`) $\rightarrow$ Taluk (`Tirthahalli`) $\rightarrow$ Hobli (`Agumbe`).
- **Step 2 — Hobli Report Tab**: Observe the suspect normal warning banner, actual vs. normal bar chart, and model inference with probability distribution.
- **Step 3 — Risk Zones Tab**: Inspect Agumbe's assignment to Cluster 3 (Deficit-Vulnerable) and explore the filterable deficit registry.
- **Step 4 — Spatial Heatmap & Model Limits Tabs**: View the 31-district departure heatmap and review the honest baseline comparison table.
- **Offline Backup Plan**: If network or browser displays fail, execute `python scripts/smoke_test_app.py` in terminal showing verified output across 3 stations with exit code 0.

### Speaker Notes (40 seconds):
> "Our application is structured around four user journeys. In the sidebar, selecting district, taluk, and hobli triggers dynamic updates. In the Hobli Report tab, selecting Agumbe immediately displays a prominent warning banner explaining that its normal is suspect, alongside its rainfall trajectory and live model probabilities. The Risk Zones tab allows filtering chronic deficit stations, while the District Trends tab presents a 31-district departure heatmap that replaces spatial maps without fabricating coordinates. If live execution faces display hurdles, our automated smoke test script demonstrates clean headless execution across three test stations."

---

## Slide 10: Limitations and Future Work

### Bullets:
- **Key Limitations**:
  - Only 5 non-consecutive years (2020–2022, 2024–2025; 2023 missing from KSNDMC archives).
  - Single-modality (surface rainfall only; lacks temperature, evapotranspiration, and soil moisture).
  - Machine learning models do not outperform the naive majority baseline on raw accuracy.
  - Absence of geographic coordinates prevents spatial polygon mapping.
- **High-Priority Future Extensions**:
  - Ingestion of Survey of India GIS shapefiles for true hobli-level choropleth mapping.
  - Integration of satellite remote sensing (Sentinel NDVI and SMAP soil moisture data).
  - Calibration of historical baseline normals in partnership with KSNDMC.

### Speaker Notes (40 seconds):
> "In conclusion, our project demonstrates both the power and the boundaries of applied machine learning in rural meteorology. Our key limitations are having only five non-consecutive years without 2023, relying solely on surface rainfall without soil moisture, and the absence of geographic coordinates. In future work, we plan to ingest GIS boundary shapefiles for true spatial mapping, incorporate satellite vegetation indices like NDVI, and collaborate with KSNDMC to recalibrate historical normals. We believe transparently communicating baseline comparisons and data limitations is essential for responsible AI deployment. Thank you, and we welcome your questions."

