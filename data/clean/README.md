# Clean Rainfall Data

Cleaned, standardised Karnataka district-wise rainfall data for 2020–2025.

## File

- `clean_rainfall.csv` — one row per hobli per year.

## Columns

| Column | Description |
|--------|-------------|
| `district` | District name |
| `taluk` | Taluk name |
| `hobli` | Hobli name |
| `year` | Year (2020–2025) |
| `normal_mm` | Normal rainfall (mm) |
| `actual_mm` | Actual rainfall (mm) |
| `departure_pct` | Departure from normal (%) |
| `rain_class` | Rainfall classification (where available) |

## Notes

- 2020–2023 geo (district/taluk) taken from the D/T/H level column in the raw files.
- 2025 has no level column; district and taluk inferred by name matching against 2020–2023 reference lists. 2025 mismatches vs other years are listed in the report (genuine disagreements only — different district/taluk, not case variants).
- See `reports/cleaning_report.md` for full cleaning details.
