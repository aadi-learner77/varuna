# Feature Engineering Documentation: `features.csv`

## Overview

The `data/features/features.csv` dataset contains tabular features engineered for machine learning models predicting annual rainfall deviations and drought/flood risk zones across hoblis (revenue circles) in Karnataka, India.

- **Source**: `data/clean/clean_rainfall.csv` (originating from official KSNDMC annual rainfall tables).
- **Temporal Scope**: **2020, 2021, 2022, 2024, and 2025** (5 observed years; 2023 is completely absent from KSNDMC hobli records).
- **Dimensions**: **4,250 rows × 22 columns** (exactly 850 canonical hobli stations per year).
- **Grain**: One row per hobli recording station per observed year.

---

## Column Descriptions

| Column | Type | Description | Leakage Free? |
| :--- | :--- | :--- | :--- |
| `district` | string | Name of the administrative district (31 districts in Karnataka). | Identity |
| `taluk` | string | Name of the administrative sub-district (taluk). | Identity |
| `hobli` | string | Name of the hobli (revenue circle) / KSNDMC recording station. | Identity |
| `year` | integer | Calendar year of rainfall observation (2020, 2021, 2022, 2024, 2025). | Temporal |
| `normal_mm` | float | Long-term normal baseline annual rainfall in millimeters (KSNDMC reference). | Pre-observation Baseline |
| `actual_mm` | float | Total observed annual rainfall in millimeters for that hobli. | Current Year Observation |
| `departure_pct` | float | Percentage rainfall departure: `((actual_mm - normal_mm) / normal_mm) * 100`. | Current Year Observation |
| `gap_years` | integer | Temporal distance (years) from the previous observed year (0 for 2020, 1 for 2021/2022/2025, 2 for 2024). | Temporal Distance |
| `n_prior_years` | integer | Number of strictly prior observed annual records available for this station (0 to 4). | Historical Depth |
| `has_prior_history` | integer | Binary flag (1 if `n_prior_years >= 1`, 0 if 2020/first observation). | History Availability Flag |
| `deficit_count` | integer | Cumulative number of prior observed years where departure was `< -20.0%`. | Yes (Prior years only) |
| `excess_count` | integer | Cumulative number of prior observed years where departure was `> +20.0%`. | Yes (Prior years only) |
| `mean_actual_mm` | float | Expanding mean of `actual_mm` over strictly prior observed years (`year < Y`). `NaN` in 2020. | Yes (Prior years only) |
| `std_actual_mm` | float | Expanding sample standard deviation of `actual_mm` over strictly prior years ($N-1$ ddof). `NaN` for $<2$ prior observations. | Yes (Prior years only) |
| `mean_departure_pct`| float | Expanding mean of `departure_pct` over strictly prior observed years (`year < Y`). `NaN` in 2020. | Yes (Prior years only) |
| `std_departure_pct` | float | Expanding sample standard deviation of `departure_pct` over strictly prior years. `NaN` for $<2$ prior observations. | Yes (Prior years only) |
| `previous_year_departure` | float | Departure percentage from the immediately preceding observed year (lag-1). `NaN` in 2020. | Yes (Prior years only) |
| `taluk_mean_departure` | float | Expanding mean departure across all prior observed records in the same taluk (`year < Y`). `NaN` in 2020. | Yes (Prior years only) |
| `district_mean_departure`| float | Expanding mean departure across all prior observed records in the same district (`year < Y`). `NaN` in 2020. | Yes (Prior years only) |
| `hobli_minus_taluk` | float | Micro-climatic offset: hobli's past mean departure minus taluk's past mean departure (`mean_departure_pct - taluk_mean_departure`). `NaN` in 2020. | Yes (Prior years only) |
| `is_suspect_normal` | integer | Flag (1 if station has unchanged baseline normal and persistent one-sided departure; see `reports/suspect_normals.csv`, else 0). | Quality Flag |
| `risk_zone` | string | **Target Variable**: 3-class drought/flood vulnerability label (`drought_prone`, `stable`, `flood_prone`). | Target |

---

## Target Definition & Class Balance

`risk_zone` is determined by standard KSNDMC / IMD monsoon deviation thresholds:
- **`drought_prone`**: `departure_pct <= -20.0%` (Deficient rainfall $\ge 20\%$ below normal).
- **`flood_prone`**: `departure_pct >= +40.0%` (Large excess rainfall $\ge 40\%$ above normal).
- **`stable`**: `-20.0% < departure_pct < +40.0%` (Normal and moderate rainfall variability).

### Class Distribution per Year and Overall

| Year | Drought-Prone | Stable | Flood-Prone | Total | Meteorological Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2020** | 30 (3.5%) | 673 (79.2%) | 147 (17.3%) | 850 | Above-normal monsoon (+20.98% mean dep) |
| **2021** | 24 (2.8%) | 550 (64.7%) | 276 (32.5%) | 850 | Heavy monsoon & depression activity (+27.66%) |
| **2022** | 8 (0.9%) | 368 (43.3%) | 474 (55.8%) | 850 | Historic catastrophic flood year (+50.67%) |
| **2024** | 33 (3.9%) | 658 (77.4%) | 159 (18.7%) | 850 | Normal-to-above monsoon year (+18.72%) |
| **2025** | 66 (7.8%) | 677 (79.7%) | 107 (12.6%) | 850 | Moderate year with emerging localized deficits (+13.21%) |
| **All** | **161 (3.8%)** | **2,926 (68.8%)** | **1,163 (27.4%)** | **4,250** | **Statewide Multi-Year Total** |

> [!WARNING]
> **Severe Class Imbalance**: `drought_prone` instances account for only **3.79%** of the entire dataset. In 2022, fewer than 1% of hoblis experienced drought due to state-wide extreme rainfall. Downstream models must use class weighting (`class_weight='balanced'`) and evaluate drought precision/recall separately from majority-class accuracy.

---

## Missing History Handling (No Silent Imputation)

In time-series tabular datasets, early observations have no prior history. Rather than silently imputing synthetic mean values or zeros that fabricate false stability:
1. **Explicit Tracking**: We maintain `n_prior_years` (0 to 4) and a binary flag `has_prior_history` (1 for rows with $\ge 1$ prior year; 0 for initial year 2020).
2. **Transparent Missingness**: Lag and rolling statistics for initial year records evaluate to `NaN` (or `0` for cumulative event counts).
3. **Training Filter**: Supervised classification models that require historical lag features filter for `has_prior_history == 1` (years 2021–2025; 3,400 observations), ensuring all model training is conducted on verified historical signals.

---

## Suspect Baseline Normals Audit (`reports/suspect_normals.csv`)

A major data-quality insight discovered during analysis:
- **338 hoblis** exhibit unchanged baseline normal values ($\Delta \le 1.0\text{ mm}$) across all 5 years alongside persistent one-sided departures.
- **Critical Baseline Distortion**: In high-rainfall Western Ghats stations (e.g., **Agumbe** in Tirthahalli, Shivamogga), `normal_mm` is pinned to an immense **7,565.0 mm**. Even when Agumbe received an astounding **5,383.1 mm** of rainfall in 2024, its departure was calculated as **-28.84%**, erroneously classifying it as `drought_prone`!
- Stations exhibiting this artificial baseline distortion are flagged via `is_suspect_normal = 1` and logged in `reports/suspect_normals.csv` so clustering and anomaly detection algorithms can isolate data-quality artifacts from true climatic vulnerability.
