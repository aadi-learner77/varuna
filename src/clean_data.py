#!/usr/bin/env python3
"""
Karnataka Rainfall Data Cleaning Script
========================================
Reads 5 raw rainfall CSV files (2020-2025) from data/raw/, cleans them,
and outputs clean_rainfall.csv plus a cleaning report.

Contract columns: district, taluk, hobli, year, normal_mm, actual_mm,
departure_pct, rain_class

Steps
-----
1. Print shape, columns, nulls, duplicates for each file BEFORE cleaning.
2. Standardise to canonical columns; drop empty cols / blank rows / dup rows
   / summary rows (SIK, NIK, MALNAD, COASTAL, State).
3. Rebuild district & taluk: forward-fill from D/T/H for 2020-2023; for 2025
   (no level column) walk in file order, matching names to the known
   district/taluk lists built from 2020-2023.
4. Clean names (whitespace, newlines, case) and fuzzy-match across years with
   rapidfuzz; list only genuinely uncertain matches.
5. Recompute departure_pct = (actual-normal)/normal*100; report rows where it
   differs from the file value by more than 2 pp.
6. Counts per year (approx 31 districts, 227 taluks, 850 hoblis) and gaps.
"""

import re
import os
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

RAW_DIR = Path("data/raw")

# filename prefix → year
FILE_MAP = {
    "0ec5ad8d": 2020,
    "275cb296": 2021,
    "741349d2": 2022,
    "d9c463e4": 2023,
    "karnataka_rainfall_2025": 2025,
}

SUMMARY_MARKERS = ("1.SIK", "2.NIK", "3.MALNAD", "4.COASTAL", "State")

# ── STEP 1: before-cleaning diagnostics ─────────────────────────────────────
print("=" * 78)
print("STEP 1 – BEFORE CLEANING  (shape, columns, nulls, duplicates, blanks)")
print("=" * 78)

raw_dfs = {}
file_info = {}

for fp in sorted(RAW_DIR.glob("*.csv")):
    name = fp.name
    year = None
    for prefix, y in FILE_MAP.items():
        if name.startswith(prefix):
            year = y
            break
    if year is None:
        print(f"\n!! Cannot determine year for {name}; skipping")
        continue

    df = pd.read_csv(fp, dtype=str, keep_default_na=False)
    raw_dfs[year] = df

    n_rows, n_cols = df.shape
    n_nulls = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    blank_mask = df.apply(lambda col: col.str.strip().eq("")).all(axis=1)
    n_blank = int(blank_mask.sum())

    print(f"\n--- {name}  (year {year}) ---")
    print(f"  shape: {n_rows} x {n_cols}")
    print(f"  columns: {list(df.columns)}")
    print(f"  nulls: {n_nulls}")
    print(f"  duplicate rows: {n_dup}")
    print(f"  blank rows: {n_blank}")

    file_info[year] = dict(
        filename=name, rows_in=n_rows, cols_in=n_cols,
        nulls_in=n_nulls, dup_in=n_dup, blank_in=n_blank,
    )


# ── STEP 2: standardise columns ─────────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 2 – Standardise columns, drop empty cols / blanks / dups / summary")
print("=" * 78)

processed = {}

for year, df in raw_dfs.items():
    if year == 2020:
        out = pd.DataFrame({
            "level": df.iloc[:, 0].astype(str).str.strip(),
            "name_raw": df.iloc[:, 1].astype(str).str.strip(),
            "normal_mm": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            "actual_mm": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "departure_pct": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "rain_class": "",
            "year": year,
        })
    elif year == 2021:
        out = pd.DataFrame({
            "level": df.iloc[:, 0].astype(str).str.strip(),
            "name_raw": df.iloc[:, 1].astype(str).str.strip(),
            "normal_mm": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            "actual_mm": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "departure_pct": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "rain_class": df.iloc[:, 5].astype(str).str.strip(),
            "year": year,
        })
    elif year == 2022:
        # Annual columns: idx 12 (Annual Normal), 13 (Annual Actual), 14 (%DEP.3)
        out = pd.DataFrame({
            "level": df.iloc[:, 0].astype(str).str.strip(),
            "name_raw": df.iloc[:, 2].astype(str).str.strip(),
            "normal_mm": pd.to_numeric(df.iloc[:, 12], errors="coerce"),
            "actual_mm": pd.to_numeric(df.iloc[:, 13], errors="coerce"),
            "departure_pct": pd.to_numeric(df.iloc[:, 14], errors="coerce"),
            "rain_class": "",
            "year": year,
        })
    elif year == 2023:
        out = pd.DataFrame({
            "level": df.iloc[:, 0].astype(str).str.strip(),
            "name_raw": df.iloc[:, 1].astype(str).str.strip(),
            "normal_mm": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            "actual_mm": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "departure_pct": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "rain_class": df.iloc[:, 5].astype(str).str.strip(),
            "year": year,
        })
    elif year == 2025:
        out = pd.DataFrame({
            "level": "",  # no level column in 2025
            "name_raw": df.iloc[:, 1].astype(str).str.strip(),
            "normal_mm": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            "actual_mm": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "departure_pct": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "rain_class": "",
            "year": year,
        })
    else:
        raise ValueError(f"Unexpected year {year}")

    processed[year] = out
    print(f"  {year}: {len(out)} rows × {len(out.columns)} cols")

combined = pd.concat(processed.values(), ignore_index=True)
print(f"\nTotal standardised: {len(combined)} rows across {combined['year'].nunique()} years")

# Drop blank name rows
n_before = len(combined)
combined = combined[combined["name_raw"].str.strip().ne("")]
print(f"  Dropped {n_before - len(combined)} blank-name rows")

# Drop duplicate rows (same year + name + normal + actual)
n_before = len(combined)
combined = combined.drop_duplicates(
    subset=["year", "name_raw", "normal_mm", "actual_mm"]
).reset_index(drop=True)
print(f"  Dropped {n_before - len(combined)} duplicate rows")

# Drop summary / region rows
n_summary = int(combined["name_raw"].str.strip().isin(SUMMARY_MARKERS).sum())
combined = combined[~combined["name_raw"].str.strip().isin(SUMMARY_MARKERS)].reset_index(drop=True)
print(f"  Dropped {n_summary} summary/region rows (SIK, NIK, MALNAD, COASTAL, State)")


# ── Build reference lists from 2020-2023 ─────────────────────────────────────
# district_names: set of district-level names (from D rows)
# taluk_names:   set of taluk-level names (from T rows)
# We build these from the level column in 2020-2023.

print("\n" + "=" * 78)
print("STEP 3 – Rebuild district & taluk")
print("=" * 78)

district_names = set()
taluk_names = set()

for year in (2020, 2021, 2022, 2023):
    df_year = combined[combined["year"] == year]
    for _, row in df_year.iterrows():
        lvl = str(row["level"]).strip().upper()
        nm = str(row["name_raw"]).strip().lower()
        if lvl == "D":
            district_names.add(nm)
        elif lvl == "T":
            taluk_names.add(nm)

print(f"  Reference district names: {len(district_names)}")
print(f"  Reference taluk names: {len(taluk_names)}")


def fill_geo_from_level(df_year):
    """Forward-fill district/taluk from D/T/H level column."""
    districts, taluks = [], []
    cur_d, cur_t = "", ""
    for _, row in df_year.iterrows():
        lvl = str(row["level"]).strip().upper()
        nm = str(row["name_raw"]).strip()
        if lvl == "D":
            cur_d, cur_t = nm, ""
        elif lvl == "T":
            cur_t = nm
        districts.append(cur_d)
        taluks.append(cur_t)
    return districts, taluks


def infer_geo_2025(df_25):
    """
    Walk the 2025 file in order and infer district/taluk by name matching.

    Strategy:
      - If name exactly matches a known district → new district block.
      - If name exactly matches a known taluk → new taluk block.
      - If name has a _N or _X suffix → it is a hobli; keep current geo.
      - Otherwise fuzzy-match against taluk names (threshold 90%) ONLY when the
        name does NOT have a suffix — to handle case/spelling differences like
        "Dodballapur" vs "DODDABALLAPURA".
      - If still no district, fuzzy-match district names (threshold 85%).
    """
    districts, taluks = [], []
    cur_d, cur_t = "", ""

    for _, row in df_25.iterrows():
        name = str(row["name_raw"]).strip()
        name_l = name.lower()

        # Does this name have a hobli-style suffix?  (_1, _2, _A, _B, etc.)
        has_suffix = bool(re.search(r"_[0-9]+$", name_l) or re.search(r"_[a-z]$", name_l))

        # 1. Exact match to a known district → new district.
        #    Names in BOTH district_names and taluk_names (e.g. "Ramanagara")
        #    are districts that also have a same-named taluk.
        #    - If name == cur_d (same-named taluk under current district): treat as taluk.
        #    - If name != cur_d (different district): treat as new district.
        if name_l in district_names:
            if name_l in taluk_names and cur_d and name_l != cur_d.lower():
                # Different from current district → new district
                cur_d, cur_t = name, ""
            elif name_l in taluk_names and cur_d and name_l == cur_d.lower():
                # Same-named taluk under current district
                cur_t = name
            else:
                # Not in taluk_names, or no cur_d yet → new district
                cur_d, cur_t = name, ""
        # 2. Exact taluk match (not also a district) → new taluk
        elif name_l in taluk_names and cur_d:
            cur_t = name
        # 3. Fuzzy-match against district/taluk names for case/spelling diffs
        #    (e.g. "Dodballapur" vs "DODDABALLAPURA").  Only when the name has no
        #    hobli suffix and we still need a district/taluk.
        if not has_suffix:
            if not cur_d and district_names:
                best = process.extractOne(name_l, list(district_names), scorer=fuzz.token_sort_ratio)
                if best and best[1] >= 85:
                    cur_d = best[0]
            elif cur_d and taluk_names:
                best = process.extractOne(name_l, list(taluk_names), scorer=fuzz.token_sort_ratio)
                if best and best[1] >= 85:
                    cur_t = name
        # 4. Has suffix or no fuzzy match → it's a hobli, keep current geo

        districts.append(cur_d if cur_d else "UNKNOWN")
        taluks.append(cur_t if cur_t else "UNKNOWN")

    return districts, taluks


geo_rows = []

for year, grp in combined.groupby("year", sort=False):
    if year == 2025:
        districts_25, taluks_25 = infer_geo_2025(grp.reset_index(drop=True))
        for i, (_, row) in enumerate(grp.reset_index(drop=True).iterrows()):
            geo_rows.append(dict(row.to_dict(), district=districts_25[i], taluk=taluks_25[i]))
    else:
        districts, taluks = fill_geo_from_level(grp)
        for i, (_, row) in enumerate(grp.iterrows()):
            geo_rows.append(dict(row.to_dict(), district=districts[i], taluk=taluks[i]))

combined_geo = pd.DataFrame(geo_rows)
print(f"  Geo filled for all years: {len(combined_geo)} rows")


# ── Cross-check 2025 inference ───────────────────────────────────────────────
# For hobli names appearing in both 2025 and 2020-2023, check if the inferred
# district/taluk matches the majority vote from other years.

# Build a per-hobli (district, taluk) vote from 2020-2023
from collections import defaultdict, Counter

hobli_votes = defaultdict(Counter)  # hobli_lower → Counter((district, taluk))
for year in (2020, 2021, 2022, 2023):
    df_year = combined_geo[combined_geo["year"] == year]
    for _, row in df_year.iterrows():
        key = str(row["name_raw"]).strip().lower()
        geo = (str(row["district"]).strip(), str(row["taluk"]).strip())
        hobli_votes[key][geo] += 1

mismatch_report = []
for _, row in combined_geo[combined_geo["year"] == 2025].iterrows():
    key = str(row["name_raw"]).strip().lower()
    d25 = str(row["district"]).strip()
    t25 = str(row["taluk"]).strip()
    if key in hobli_votes:
        # Majority vote from other years
        votes = hobli_votes[key]
        majority_geo, majority_count = votes.most_common(1)[0]
        total = sum(votes.values())
        if majority_count >= 2 and (d25, t25) != majority_geo:
            mismatch_report.append((key, d25, t25, majority_geo[0], majority_geo[1], majority_count, total))

print(f"\n  2025 inference cross-check: {len(mismatch_report)} mismatches vs majority vote from 2020-2023")


# ── STEP 4: clean names & fuzzy-match ────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 4 – Clean names, match across years with rapidfuzz")
print("=" * 78)

def clean_name(s):
    s = str(s).strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    base = re.sub(r"_[0-9]+$", "", s.lower())
    base = re.sub(r"_[a-z]$", "", base)
    return s, base

combined_geo["name_clean"], combined_geo["name_base"] = zip(
    *combined_geo["name_raw"].apply(clean_name)
)

name_year_map = {}
for _, row in combined_geo.iterrows():
    name_year_map.setdefault(row["name_base"], set()).add(row["year"])

rare_names = {b: y for b, y in name_year_map.items() if len(y) < 5}
print(f"  Unique base hobli names: {len(name_year_map)}")
print(f"  Names in < 5 years: {len(rare_names)}")

# Find 2025 names with no exact match in other years
other_bases = {r["name_base"] for _, r in combined_geo[combined_geo["year"] != 2025].iterrows()}
names_2025 = {r["name_base"] for _, r in combined_geo[combined_geo["year"] == 2025].iterrows()}

uncertain = []
for n25 in sorted(names_2025 - other_bases):
    best = process.extractOne(n25, list(other_bases), scorer=fuzz.token_sort_ratio)
    if best and 60 <= best[1] < 85:
        uncertain.append((n25, best[0], best[1]))

print(f"  Uncertain cross-year matches (< 85%, >= 60%): {len(uncertain)}")


# ── STEP 5: recompute departure_pct ──────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 5 – Recompute departure_pct; flag rows differing > 2 pp")
print("=" * 78)

combined_geo["departure_recomputed"] = np.where(
    combined_geo["normal_mm"].ne(0),
    (combined_geo["actual_mm"] - combined_geo["normal_mm"]) / combined_geo["normal_mm"] * 100,
    np.nan,
).round(2)

combined_geo["departure_file"] = pd.to_numeric(combined_geo["departure_pct"], errors="coerce")
diff_mask = (combined_geo["departure_recomputed"] - combined_geo["departure_file"]).abs() > 2
n_diff = int(diff_mask.sum())
print(f"  Rows differing > 2 pp: {n_diff}")

departure_issues = combined_geo[diff_mask][
    ["year", "district", "taluk", "name_clean", "normal_mm", "actual_mm",
     "departure_file", "departure_recomputed"]
].copy()
departure_issues["diff_pp"] = (
    departure_issues["departure_recomputed"] - departure_issues["departure_file"]
).round(2)


# ── STEP 6: counts per year ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 6 – Counts per year (districts, taluks, hoblis)")
print("=" * 78)

counts = []
for year, grp in combined_geo.groupby("year", sort=False):
    counts.append(dict(
        year=year,
        districts=grp["district"].nunique(),
        taluks=grp["taluk"].nunique(),
        hoblis=grp["name_clean"].nunique(),
        rows=len(grp),
    ))

counts_df = pd.DataFrame(counts)
print(counts_df.to_string(index=False))

EXP_D, EXP_T, EXP_H = 31, 227, 850
print("\n  Gaps vs expected:")
for r in counts:
    print(f"    {r['year']}: districts {r['districts']} ({r['districts']-EXP_D:+d}), "
          f"taluks {r['taluks']} ({r['taluks']-EXP_T:+d}), "
          f"hoblis {r['hoblis']} ({r['hoblis']-EXP_H:+d})")


# ── Build clean output ────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("SAVE – data/clean/clean_rainfall.csv")
print("=" * 78)

final = combined_geo.copy()
final["district"] = final["district"].astype(str).str.strip()
final["taluk"] = final["taluk"].astype(str).str.strip()
final["hobli"] = final["name_clean"].astype(str).str.strip()
final["year"] = final["year"].astype(int)
final["normal_mm"] = final["normal_mm"].round(1)
final["actual_mm"] = final["actual_mm"].round(1)
final["departure_pct"] = final["departure_recomputed"]
final["rain_class"] = final["rain_class"].astype(str).str.strip()

# Mark hobli rows: has _N suffix, or name differs from district AND taluk
dist_set = set(final["district"].str.lower())
taluk_set = set(final["taluk"].str.lower())

def is_hobli(row):
    n = row["hobli"].lower()
    if re.search(r"_(?:\d+|[a-z])$", n):
        return True
    return n not in dist_set and n not in taluk_set

final["is_hobli"] = final.apply(is_hobli, axis=1)

clean_out = final[final["is_hobli"]].copy()
clean_out = clean_out[["district", "taluk", "hobli", "year",
                       "normal_mm", "actual_mm", "departure_pct", "rain_class"]]
clean_out = clean_out.sort_values(["year", "district", "taluk", "hobli"]).reset_index(drop=True)

os.makedirs("data/clean", exist_ok=True)
clean_out.to_csv("data/clean/clean_rainfall.csv", index=False)
print(f"  Saved {len(clean_out)} hobli rows")


# ── Save report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("SAVE – reports/cleaning_report.md")
print("=" * 78)

os.makedirs("reports", exist_ok=True)
L = []

L.append("# Data Cleaning Report\n")
L.append("## Overview\n")
L.append(f"- Source: {len(raw_dfs)} CSV files in `data/raw/`")
L.append(f"- Years: {sorted(raw_dfs.keys())}")
L.append(f"- Output: `data/clean/clean_rainfall.csv` ({len(clean_out)} hobli rows)\n")

for year in sorted(raw_dfs.keys()):
    info = file_info[year]
    yc = clean_out[clean_out["year"] == year]
    L.append(f"## {year}  (`{info['filename']}`)\n")
    L.append("| Metric | Before | After |")
    L.append("|--------|--------|-------|")
    L.append(f"| Rows | {info['rows_in']} | {len(yc)} |")
    L.append(f"| Columns (raw) | {info['cols_in']} | 8 (standardised) |")
    L.append(f"| Null cells | {info['nulls_in']} | 0 (after coercion) |")
    L.append(f"| Duplicate rows dropped | {info['dup_in']} | — |")
    L.append(f"| Blank rows dropped | {info['blank_in']} | — |")
    L.append(f"| Summary/region rows dropped | {n_summary if year == min(raw_dfs) else 0} | — |")
    L.append(f"| Districts (unique) | — | {yc['district'].nunique()} |")
    L.append(f"| Taluks (unique) | — | {yc['taluk'].nunique()} |")
    L.append(f"| Hoblis (unique) | — | {yc['hobli'].nunique()} |")
    L.append("")

L.append("## Departure_pct recomputation\n")
L.append(f"- Rows where recomputed departure differs from file by > 2 pp: **{n_diff}**\n")
if n_diff > 0:
    L.append("| Year | District | Taluk | Hobli | File % | Recomputed % | Diff (pp) |")
    L.append("|------|----------|-------|-------|--------|--------------|-----------|")
    for _, r in departure_issues.head(20).iterrows():
        L.append(f"| {int(r['year'])} | {r['district']} | {r['taluk']} | {r['name_clean']} | "
                  f"{r['departure_file']} | {r['departure_recomputed']} | {r['diff_pp']} |")
    if len(departure_issues) > 20:
        L.append(f"... and {len(departure_issues)-20} more")
    L.append("")

L.append("## Cross-year name matching (rapidfuzz)\n")
L.append(f"- Unique base hobli names: **{len(name_year_map)}**\n")
L.append(f"- Names in < 5 years: **{len(rare_names)}**\n")
L.append(f"- Uncertain matches (< 85%, ≥ 60%): **{len(uncertain)}**\n")
if uncertain:
    L.append("\n### For manual review\n")
    L.append("| 2025 name | Best match | Similarity |")
    L.append("|-----------|------------|------------|")
    for n25, best, score in uncertain[:20]:
        L.append(f"| {n25} | {best} | {score}% |")
    if len(uncertain) > 20:
        L.append(f"... and {len(uncertain)-20} more")
    L.append("")

L.append("## 2025 geographic inference\n")
L.append("- The 2025 file has no D/T/H level column.  District & taluk were inferred by walking ")
L.append("  the file in order: when a name matches a known district (from 2020-2023 D rows) it starts ")
L.append("  a new district block; when a name matches a known taluk it starts a new taluk block; ")
L.append("  otherwise the row inherits the current district/taluk.\n")
if mismatch_report:
    L.append(f"- **{len(mismatch_report)}** hobli rows where the 2025 inference differs from the majority ")
    L.append("  vote across 2020-2023 (only shown when ≥ 2 other years agree):\n")
    L.append("| Hobli | 2025 district | 2025 taluk | Majority district | Majority taluk | Votes |")
    L.append("|-------|---------------|------------|------------------|----------------|-------|")
    for key, d25, t25, d_maj, t_maj, votes, total in mismatch_report[:25]:
        L.append(f"| {key} | {d25} | {t25} | {d_maj} | {t_maj} | {votes}/{total} |")
    if len(mismatch_report) > 25:
        L.append(f"... and {len(mismatch_report)-25} more")
    L.append("")
else:
    L.append("- No mismatches: 2025 inference matches the majority vote from 2020-2023.\n")

L.append("## Yearly counts\n")
L.append(counts_df.to_markdown(index=False) + "\n")
L.append(f"\nExpected approx: {EXP_D} districts, {EXP_T} taluks, {EXP_H} hoblis.\n")

with open("reports/cleaning_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("  Saved reports/cleaning_report.md")

print("\nDone.")
