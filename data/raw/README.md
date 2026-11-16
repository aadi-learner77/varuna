# Raw Rainfall Data

Source CSV files for Karnataka district-wise rainfall, 2020–2025.

## Files

| File | Year | Notes |
|------|------|-------|
| `0ec5ad8d-…csv` | 2020 | 5 columns, D/T/H level column, hobli rows mixed with district/taluk headers |
| `275cb296-…csv` | 2021 | 16 columns, includes season-wise columns; annual data in cols 12–14 |
| `741349d2-…csv` | 2022 | 15 columns, season-wise with annual summary in cols 12–14; 21 duplicate rows |
| `d9c463e4-…csv` | 2023 | 8 columns, 37 blank rows, 36 duplicate rows |
| `karnataka_rainfall_2025_table1_1.csv` | 2025 | 5 columns, no D/T/H level column — geo inferred |

## Layout quirks

- Column indices and names differ across years.
- Some files carry district/taluk header rows interleaved with hobli rows (marked D/T/H).
- 2022 and 2023 include seasonal breakdowns; cleaning uses annual figures where available.
- 2025 file has only Sl.No, District/Taluk/Hobli, Normal_mm, Actual_mm, Departure (%).

## Cleaning

See `src/clean_data.py` and `reports/cleaning_report.md`.
