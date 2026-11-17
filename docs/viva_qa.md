# Comprehensive Viva Examination Q&A Guide

**Project**: Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning  
**Target Audience**: Academic Project Viva Examiners, Technical Evaluators, and Domain Reviewers  
**Role**: Person 4 (Integration, Web Application, Documentation & Defense)

---

### Q1: Why did you use a time-aware train/test split instead of a random shuffle train/test split?
**Answer**:  
Monsoon rainfall is an autoregressive temporal process. A random train/test split would cause massive temporal data leakage: models would be trained on future observations (e.g., 2024 or 2025) and tested on past observations (e.g., 2021), fabricating unrealistically high performance. In real-world deployment, disaster authorities must forecast the upcoming year using strictly historical data. Therefore, we used a strict chronological time-aware split: training on earlier years (**2020–2024; 3,400 rows**) and evaluating on the **2025 hold-out season (850 rows)**.

---

### Q2: Why is raw accuracy a misleading metric for evaluating your classification models?
**Answer**:  
Because of severe class imbalance. In the 2025 hold-out test set, **79.65% (677 of 850 hoblis)** were in the `stable` class, while `drought_prone` accounted for only **7.76% (66 hoblis)**. A trivial dummy model that always predicts `stable` for every single station scores **79.65% accuracy**, yet its operational utility is zero because its drought recall is **0.00%**. Relying on accuracy would reward a model that never warns administrators about droughts. We evaluate using **Macro-F1** and **Drought-Class Recall**.

---

### Q3: Why is drought so difficult to predict in this specific dataset?
**Answer**:  
Drought is extremely scarce in this historical window. Across the entire 5-year dataset of 4,250 observations, only **161 instances (3.79%)** were drought-prone ($\le -20\%$ departure). In the catastrophic 2022 flood year, statewide drought dropped below 1% (only 8 hoblis). Because recent years were dominated by surplus monsoon rainfall, machine learning models had very few historical examples of drought to learn the atmospheric and historical signals preceding moisture deficits.

---

### Q4: What is `gap_years` and why was it necessary in your feature engineering?
**Answer**:  
`gap_years` is a temporal distance feature indicating how many years elapsed since the station's immediately preceding observation. In standard annual data, observations are consecutive ($t - 1$), so `gap_years = 1`. However, KSNDMC published no hobli-level data for 2023. Between 2022 and 2024, two calendar years elapsed. Setting `gap_years = 2` explicitly informs tree models that the historical lag features span a two-year jump rather than an immediate prior season, preventing artificial temporal continuity assumptions.

---

### Q5: Why do your machine learning models fail to beat the naive majority class baseline on accuracy?
**Answer**:  
The naive majority baseline achieved **79.65% accuracy** on the 2025 test set by exploiting the 79.65% class dominance of the `stable` category. Our best classifier (Random Forest) achieved **70.82% accuracy** (-8.83 percentage points). It loses on accuracy because it was trained with balanced class weights to actively search for minority drought and flood events, leading to a small number of false alarms on stable hoblis. However, while the majority baseline achieved **0.2956 Macro-F1 and 0.0% drought recall**, Random Forest achieved **0.4077 Macro-F1** and captured localized drought signals with 25.0% precision.

---

### Q6: What is the "Agumbe issue" and how does it demonstrate a data-quality defect?
**Answer**:  
Agumbe (in Tirthahalli taluk, Shivamogga) is one of India's wettest rainfall stations. In 2024, Agumbe received an enormous **5,383.1 mm** of rainfall, and in 2021 it received **4,152.0 mm**. Despite these astronomical volumes, KSNDMC tables flagged Agumbe as `drought_prone` with departures of **-28.84%** and **-45.12%**. This happened because KSNDMC records assign Agumbe a fixed static normal of **7,565.0 mm**. This is not a true meteorological drought; it is a **data-quality artifact** arising from an uncalibrated, static historical normal benchmark. We flagged 338 such stations in `reports/suspect_normals.csv`.

---

### Q7: How were hobli names matched across different years in the raw data?
**Answer**:  
Raw government tables contained **898 variant spellings** for ~850 physical stations due to transliteration variations (e.g., *Thirthahalli* vs. *Tirthahalli*, *Sakaleshpur* vs. *Sakaleshpura*). We used **RapidFuzz** to execute cross-year fuzzy matching against an anchored 2021 reference gazetteer. Matching candidates were restricted hierarchically to the same taluk and district. Matches with similarity score $\ge 85\%$ were harmonized into **877 canonical stations**, resulting in **824 stations (94.0%)** persisting across all five observed years with zero manual uncertainty flags.

---

### Q8: What is data leakage and how did your pipeline prevent both temporal and spatial leakage?
**Answer**:  
Data leakage occurs when information from outside the training split or from the prediction target contaminates model training. We prevented leakage on two fronts:
1. **Temporal Leakage**: Every feature in `features.csv` is strictly *ex-ante*—computed only over prior years ($t < Y$). Current year actual rainfall and current departure are never used as inputs to predict current risk.
2. **Spatial Leakage**: Random $k$-fold cross-validation causes spatial leakage because adjacent hoblis share identical synoptic weather patterns. We implemented **5-Fold District-Grouped Cross-Validation (`GroupKFold` across 31 districts)**, ensuring all hoblis from a given district reside exclusively in either the train or test fold.

---

### Q9: Why is calendar year 2023 missing from the dataset? Did you drop it or lose it?
**Answer**:  
We did not drop or lose 2023. Calendar year 2023 was never published at the hobli level by KSNDMC in OpenCity archives. In fact, an earlier version of this repository mistakenly mislabeled `rainfall_2022.csv` as 2023. Person 1 conducted a meteorological audit, verifying that the file with $+50.67\%$ departure was the historic 2022 flood year. Because hobli-level data for 2023 does not exist in open-government archives, we explicitly handled the gap using `gap_years = 2` between 2022 and 2024.

---

### Q10: Why did you drop Ridge Regression from deployment in Stage 3?
**Answer**:  
Ridge Regression achieved an unacceptable Mean Absolute Error of **53.79%** on the 2025 hold-out test set, failing our pre-established validation threshold (MAE $\le 30.0\%$). Inspection of learned coefficients revealed that Ridge assigned a huge positive weight (**+20.15**) to `n_prior_years` and a large negative weight (**-19.88**) to `gap_years`. Because `n_prior_years` increased monotonically across training years during a multi-year wet cycle (2020: +21%, 2021: +28%, 2022: +51%), Ridge learned an artificial linear time trend. In 2025 where `n_prior_years = 4`, it linearly extrapolated departure to over $+80\%$. It was therefore permanently dropped.

---

### Q11: In continuous departure regression, why does the simple "Predict Zero Departure" baseline beat machine learning regressors?
**Answer**:  
Predicting Zero Departure ($0.0\%$) represents predicting the long-term climatological normal ($actual = normal$). It achieved an MAE of **21.65%**, outperforming both Gradient Boosting Regressor (**24.42%**) and Random Forest Regressor (**26.35%**). This happened because of macro-regime abatement: the 2020–2024 training window was heavily skewed by extreme excess rainfall (+50.67% in 2022), leading ML regressors to learn positive departure offsets. In 2025, Karnataka's weather moderated sharply toward normal (+13.20%). Predicting zero deviation was mathematically closer to the true outcome than models carrying historical flood bias.

---

### Q12: How did you select $k=4$ for K-Means clustering, and what do the four clusters represent?
**Answer**:  
We evaluated $k \in [2..8]$ on standardized 5-year aggregated station profiles (`mean_departure`, `std_departure`, `deficit_count`, `excess_count`, `mean_normal_mm`). $k=4$ achieved the highest silhouette score (**0.3406**) with clear meteorological separation:
- **Cluster 0 (Ghats & Coast, $n=87$)**: High-rainfall belt (normal 3,574 mm, mean departure +12.1%).
- **Cluster 1 (Flood-Prone Plains, $n=344$)**: Semi-arid northern/eastern plains with chronic flood surges (mean departure +45.0%, avg 3.69 excess years).
- **Cluster 2 (Stable Plains, $n=325$)**: Central/southern agricultural plains with low variability (mean departure +18.0%, 0.0 deficit years).
- **Cluster 3 (Deficit-Vulnerable, $n=123$)**: Chronic deficit pockets averaging 1.23 deficit years out of 5.

---

### Q13: How did you perform anomaly detection and what is the difference between physical anomalies and data-quality anomalies?
**Answer**:  
We fitted an **Isolation Forest** on multi-dimensional variables (`normal_mm`, `actual_mm`, `departure_pct`) combined with taluk-relative micro-climatic deviation, ranking the top 20 most anomalous observations.
- **Physical Anomalies**: True extreme convective events, such as **Hettur** in Hassan recording 6,488.2 mm vs. 2,218.3 mm normal (+192.5% departure) during a localized cloudburst.
- **Data-Quality Anomalies**: Stations like **Agumbe**, where an astronomical actual rainfall of 5,383 mm produced a false -28.8% drought label because of an inflated 7,565 mm static normal.

---

### Q14: Why is there no geographical map in your Streamlit application? Did you consider generating mock coordinates?
**Answer**:  
The raw KSNDMC tables contain only administrative string identifiers (`district`, `taluk`, `hobli`) and completely lack latitude and longitude coordinates. Under strict academic and scientific fidelity guidelines, **we refused to synthesize or fabricate mock coordinates**. Fabricated coordinates in rural disaster management could mislead relief operations. Instead, we replaced the map with a comprehensive **31-District Departure Heatmap** and flagged GIS shapefile ingestion as high-priority future work.

---

### Q15: What were the most important features in your Random Forest classifier?
**Answer**:  
The top five predictive features were:
1. `normal_mm` (**17.31%**): The baseline climatological volume of the station.
2. `previous_year_departure` (**10.68%**): Short-term persistence from the immediately preceding monsoon season.
3. `district_mean_departure` (**10.34%**): Regional macro-climatic signal of the parent district.
4. `mean_departure_pct` (**10.01%**): Long-term historical departure average of the hobli.
5. `mean_actual_mm` (**8.22%**): Historical absolute rainfall volume.  
Micro-climatic offset (`hobli_minus_taluk`) also contributed 8.14%, proving that deviation from taluk neighbors is an effective local predictor.

---

### Q16: What was the purpose of `src/export_cluster_labels.py` and why was it separated from `train_models.py`?
**Answer**:  
`train_models.py` executes model fitting, evaluation, cross-validation, and saves serialized `.joblib` artifacts. However, downstream consumer applications like the Streamlit UI need access to a dedicated per-hobli lookup table mapping each station to its cluster assignment and plain-language meteorological description. `src/export_cluster_labels.py` loads the saved scaler and fitted K-Means model, transforms the aggregated station profiles, and exports `data/features/hobli_clusters.csv` (879 records) without re-running time-consuming training loops.

---

### Q17: How did you handle the administrative bifurcation of Vijayanagar from Ballari?
**Answer**:  
In late 2021, Karnataka created Vijayanagar district by carving out six taluks from Ballari (*Hadagali, Hagaribommanahalli, Harappanahalli, Hosapete, Kotturu, Kudligi*). In 2020–2021 raw files, these taluks appeared under Ballari; in 2022–2025, they appeared under Vijayanagar. Person 1 implemented a canonical district normalizer in `clean_data.py` that checks the taluk name and maps all six taluks consistently to Vijayanagar across all five years, eliminating artificial district fragmentation.

---

### Q18: What is the Persistence Baseline and how did your models perform against it?
**Answer**:  
The Persistence Baseline (Lag-1) predicts that each station's risk zone in 2025 will be identical to its observed risk zone in 2024. In weather forecasting, "tomorrow's weather will be like today's" is a surprisingly strong benchmark. On the 2025 test set, Persistence achieved **68.12% accuracy and 0.4095 Macro-F1**. Our Random Forest classifier beat Persistence on accuracy (**70.82% vs. 68.12%**) and roughly tied it on Macro-F1 (**0.4077 vs. 0.4095**), demonstrating that it learned meaningful predictive structure beyond mere persistence.

---

### Q19: Why did Balanced Logistic Regression perform so poorly on the 2025 test set (13.76% accuracy)?
**Answer**:  
Balanced Logistic Regression suffered from severe class-weight distortion. In the 2020–2024 training window, wet years heavily predominated. The inverse-frequency `class_weight='balanced'` heuristic assigned extreme mathematical penalty multipliers to minority classes. When evaluated on 2025, the linear decision boundary over-compensated, misclassifying 661 out of 677 `stable` hoblis as `flood_prone` false positives. Non-linear tree ensembles handled these class weights much more gracefully.

---

### Q20: If you had another 6 months on this project, what would be your top three priorities?
**Answer**:  
1. **Multivariate Meteorological Ingestion**: Incorporate temperature, potential evapotranspiration (PET), and ERA5 soil moisture indices, moving from rainfall-only metrics to Standardized Precipitation-Evapotranspiration Index (SPEI).
2. **GIS Boundary Polygon Integration**: Ingest Survey of India hobli shapefiles to enable true interactive spatial choropleth mapping and spatial-lag econometric modeling.
3. **Collaboration with KSNDMC**: Work with state disaster authorities to recalibrate static baseline normals (correcting anomalies like Agumbe) and digitize pre-2020 and missing 2023 records.

