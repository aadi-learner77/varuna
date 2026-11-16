# Data Cleaning Report

## Overview

- Source: 5 CSV files in `data/raw/`
- Years: [2020, 2021, 2022, 2023, 2025]
- Output: `data/clean/clean_rainfall.csv` (3088 hobli rows)

## 2020  (`0ec5ad8d-be55-4e92-a942-6918c025735b.csv`)

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1122 | 618 |
| Columns (raw) | 5 | 8 (standardised) |
| Null cells | 0 | 0 (after coercion) |
| Duplicate rows dropped | 0 | — |
| Blank rows dropped | 0 | — |
| Summary/region rows dropped | — | — |
| Districts (unique) | — | 31 |
| Taluks (unique) | — | 215 |
| Hoblis (unique) | — | 587 |

## 2021  (`275cb296-b3aa-4c96-b614-c3c3ca19dec0.csv`)

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1113 | 617 |
| Columns (raw) | 16 | 8 (standardised) |
| Null cells | 0 | 0 (after coercion) |
| Duplicate rows dropped | 0 | — |
| Blank rows dropped | 0 | — |
| Summary/region rows dropped | — | — |
| Districts (unique) | — | 31 |
| Taluks (unique) | — | 209 |
| Hoblis (unique) | — | 586 |

## 2022  (`741349d2-9f58-45ae-9744-b6d5c10a8b7f.csv`)

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1134 | 617 |
| Columns (raw) | 15 | 8 (standardised) |
| Null cells | 0 | 0 (after coercion) |
| Duplicate rows dropped | 21 | — |
| Blank rows dropped | 0 | — |
| Summary/region rows dropped | — | — |
| Districts (unique) | — | 30 |
| Taluks (unique) | — | 209 |
| Hoblis (unique) | — | 586 |

## 2023  (`d9c463e4-3203-49fd-95c3-33b79f85c9b3.csv`)

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1150 | 617 |
| Columns (raw) | 8 | 8 (standardised) |
| Null cells | 0 | 0 (after coercion) |
| Duplicate rows dropped | 36 | — |
| Blank rows dropped | 37 | — |
| Summary/region rows dropped | — | — |
| Districts (unique) | — | 31 |
| Taluks (unique) | — | 209 |
| Hoblis (unique) | — | 586 |

## 2025  (`karnataka_rainfall_2025_table1_1.csv`)

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1117 | 619 |
| Columns (raw) | 5 | 8 (standardised) |
| Null cells | 0 | 0 (after coercion) |
| Duplicate rows dropped | 0 | — |
| Blank rows dropped | 0 | — |
| Summary/region rows dropped | — | — |
| Districts (unique) | — | 31 |
| Taluks (unique) | — | 219 |
| Hoblis (unique) | — | 587 |

## Departure_pct recomputation

- Rows where recomputed departure differs from file by > 2 pp: **0**

## Cross-year name matching (rapidfuzz)

- Unique base hobli names: **825**

- Names in < 5 years: **9**

- Uncertain matches (< 85%, ≥ 60%): **0**

## Summary/region rows dropped

- **15** rows dropped across all years (SIK, NIK, MALNAD, COASTAL, State).


## 2025 geographic inference

- The 2025 file has no D/T/H level column.  District & taluk were inferred by walking 
  the file in order: when a name matches a known district (from 2020-2023 D rows) it starts 
  a new district block; when a name matches a known taluk it starts a new taluk block; 
  otherwise the row inherits the current district/taluk.

- **137** hobli rows where the 2025 inference differs from the majority 
  vote across 2020-2023 (only shown when ≥ 2 other years agree):

| Hobli | 2025 district | 2025 taluk | Majority district | Majority taluk | Votes |
|-------|---------------|------------|------------------|----------------|-------|
| channarayapatna | BENGALURU RURAL | Channarayapatna | HASSAN | Channarayapatna | 8/12 |
| kundana | BENGALURU RURAL | Channarayapatna | BENGALURU RURAL | Devanahalli | 4/4 |
| vijaypura | BENGALURU RURAL | Vijaypura | BENGALURU RURAL | Devanahalli | 4/4 |
| doddaballapura | BENGALURU RURAL | Doddaballapura | BENGALURU RURAL | DODDABALLAPURA | 3/4 |
| dodballapur | BENGALURU RURAL | Dodballapur | BENGALURU RURAL | DODDABALLAPURA | 3/4 |
| hosakote | BENGALURU RURAL | Hosakote | BENGALURU RURAL | HOSAKOTE | 6/8 |
| maluru | RAMANAGARA | Maluru | RAMANAGARA | Channapatna | 4/4 |
| harohalli | RAMANAGARA | Harohalli | RAMANAGARA | Kanakapura | 3/5 |
| mulabagilu | KOLAR | Mulabagilu | KOLAR | MULABAGILU | 3/4 |
| robertsonpet | KOLAR | K.G.F | KOLAR | Bangarapet | 4/8 |
| chikballapura | CHIKKABALLAPURA | Chikballapura | CHIKKABALLAPURA | CHIKKABALLAPURA | 3/4 |
| chikkanayakanahalli | TUMAKURU | Chikkanayakanahalli | TUMAKURU | CHIKKANAYAKANAHALLI | 3/4 |
| chiknayakanahalli | TUMAKURU | Chiknayakanahalli | TUMAKURU | CHIKKANAYAKANAHALLI | 3/4 |
| chelur | TUMAKURU | Gubbi | CHIKKABALLAPURA | Bagepalli | 4/8 |
| kadaba | TUMAKURU | Kadaba | DAKSHINA KANNADA | Kadaba | 8/12 |
| nittur | TUMAKURU | Kadaba | TUMAKURU | Gubbi | 4/4 |
| guluru | TUMAKURU | Tumakuru | CHIKKABALLAPURA | Bagepalli | 4/8 |
| davangere | DAVANAGERE | Davangere | DAVANAGERE | Davanagere | 4/4 |
| harihara | DAVANAGERE | Harihara | DAVANAGERE | HARIHARA | 6/8 |
| hangala | CHAMARAJANAGARA | Hangala | CHAMARAJANAGARA | Gundlupet | 4/4 |
| kollegala | CHAMARAJANAGARA | Kollegala | CHAMARAJANAGARA | Kollegal | 4/4 |
| agara | CHAMARAJANAGARA | Agara | CHAMARAJANAGARA | Yelandur | 4/4 |
| heggadadevanakote | MYSURU | Heggadadevanakote | MYSURU | HEGGADADEVANAKOTE | 6/8 |
| jayapura | MYSURU | Jayapura | MYSURU | Mysuru | 4/4 |
| nanjangud | MYSURU | Nanjangud | MYSURU | Nanjanagud | 4/4 |
... and 112 more

## Yearly counts

|   year |   districts |   taluks |   hoblis |   rows |
|-------:|------------:|---------:|---------:|-------:|
|   2020 |          31 |      236 |      929 |   1110 |
|   2021 |          31 |      228 |      926 |   1107 |
|   2022 |          30 |      228 |      914 |   1101 |
|   2023 |          31 |      228 |      926 |   1107 |
|   2025 |          31 |      328 |      919 |   1117 |


Expected approx: 31 districts, 227 taluks, 850 hoblis.
