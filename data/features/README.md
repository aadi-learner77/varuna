# Feature Engineering Documentation: `features.csv`

## Overview

The `data/features/features.csv` dataset contains tabular features engineered for machine learning models predicting rainfall deviations and drought/flood risk zones across hoblis (revenue circles) in Karnataka, India.

- **Source**: `data/clean/clean_rainfall.csv` (originating from KSNDMC annual rainfall reports).
- **Temporal Scope**: 2020, 2021, 2022, 2023, and 2025 (5 observation years; 2024 is missing from KSNDMC records).
- **Dimensions**: 3,088 rows × 17 columns.
- **Grain**: One row per hobli recording station per observed year.

---

## Column Descriptions

| Column | Type | Description | Leakage Free? |
| :--- | :--- | :--- | :--- |
| `district` | string | Name of the administrative district (31 districts in Karnataka). | N/A (Identity) |
| `taluk` | string | Name of the administrative sub-district (taluk). | N/A (Identity) |
| `hobli` | string | Name of the hobli (revenue circle) / KSNDMC station. | N/A (Identity) |
| `year` | integer | Calendar year of rainfall observation (2020, 2021, 2022, 2023, 2025). | N/A (Temporal) |
| `normal_mm` | float | Long-term normal baseline annual rainfall in millimeters (KSNDMC reference). | Pre-observation Baseline |
| `actual_mm` | float | Total observed annual rainfall in millimeters for that hobli. | Current Year Observation |
| `departure_pct` | float | Percentage rainfall departure: `((actual_mm - normal_mm) / normal_mm) * 100`. | Current Year Observation |
| `deficit_count` | integer | Cumulative number of strictly prior observed years where departure was `< -20.0%`. | Yes (Prior years only) |
| `excess_count` | integer | Cumulative number of strictly prior observed years where departure was `> +20.0%`. | Yes (Prior years only) |
| `mean_actual_mm` | float | Expanding mean of `actual_mm` over strictly prior observed years (`year < Y`). `NaN` in year 2020. | Yes (Prior years only) |
| `std_actual_mm` | float | Expanding sample standard deviation ($N-1$ degrees of freedom) of `actual_mm` over strictly prior years. `NaN` for years with $<2$ prior observations (2020, 2021). | Yes (Prior years only) |
| `std_departure_pct` | float | Expanding sample standard deviation ($N-1$ degrees of freedom) of `departure_pct` over strictly prior years. `NaN` for years with $<2$ prior observations. | Yes (Prior years only) |
| `previous_year_departure` | float | Departure percentage from the immediately preceding observed year (lag-1). `NaN` in year 2020. | Yes (Prior years only) |
| `taluk_mean_departure` | float | Mean `departure_pct` across all hoblis in the same taluk during the same year. | Spatial Context |
| `district_mean_departure` | float | Mean `departure_pct` across all hoblis in the same district during the same year. | Spatial Context |
| `hobli_minus_taluk` | float | Difference between hobli's `departure_pct` and `taluk_mean_departure`, capturing local micro-climatic anomaly. | Spatial Context |
| `risk_zone` | string | **Target Variable**: 3-class drought/flood vulnerability label (`drought_prone`, `stable`, `flood_prone`). | Target |

---

## Target Definition & Class Balance

`risk_zone` is determined by KSNDMC / IMD monsoon deviation standards using thresholds defined at the module level:

- **`drought_prone`**: `departure_pct <= -20.0%` (Deficient rainfall $\ge 20\%$ below normal).
- **`flood_prone`**: `departure_pct >= +40.0%` (Large excess rainfall $\ge 40\%$ above normal).
- **`stable`**: `-20.0% < departure_pct < +40.0%` (Normal and mild excess rainfall).

### Class Distribution (3,088 total records)

| Class | Count | Percentage | Meteorological Context |
| :--- | :---: | :---: | :--- |
| `stable` | 2,067 | 66.94% | Rainfall within expected variability bounds (-20% to +40%). |
| `flood_prone` | 904 | 29.27% | Severe surplus rainfall triggering waterlogging, river spillage, and landslides in Western Ghats/Coastal belts. |
| `drought_prone` | 117 | 3.79% | Severe deficit rainfall posing agricultural drought risk in vulnerable semi-arid hoblis. |

---

## Leakage Prevention Architecture

In time-series machine learning, aggregating features across the entire dataset or including the current year's observation introduces lookahead data leakage. To guarantee zero leakage:

1. **Expanding Window Shift**: All history-based features (`deficit_count`, `excess_count`, `mean_actual_mm`, `std_actual_mm`, `std_departure_pct`, `previous_year_departure`) for year $Y$ evaluate strictly over records with $\text{year} < Y$.
2. **First-Year Integrity**: In 2020 (the initial year), all lag and rolling statistical features evaluate to `NaN` (or `0` for cumulative counts).
3. **Station Persistence**: Stations are tracked via a composite key that accounts for multi-station taluks (e.g. numbered hoblis) and district reorganizations (e.g. Ballari / Vijayanagar bifurcation in 2021-2022).
