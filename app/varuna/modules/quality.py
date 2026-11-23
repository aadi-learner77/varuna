"""
Module 04 -- Data Quality & Normalization Audit
===============================================
The cleaning pipeline presented as a sequence of scientific decisions rather
than a table of technical footnotes.
"""

from __future__ import annotations

import streamlit as st

from .. import config, ui


PAGE = {
    "slug": "quality",
    "index": "04",
    "label": "Data Audit",
    "title": "Data Quality & Normalization Audit",
    "kicker": "Module 04 \u00b7 Provenance",
    "lede": (
        "Open government rainfall tables are not analysis-ready. This is the full record of what "
        "was removed, what was harmonized, and which findings should temper every number in the "
        "other four modules."
    ),
}


def render(data, sel: dict, p: dict) -> None:
    suspect = data.suspect
    facts = config.NORMALIZATION_FACTS

    ui.render(
        ui.page_header(
            PAGE["kicker"],
            PAGE["title"],
            PAGE["lede"],
            meta=[
                ("Raw records", f"{config.CLEANING_TOTALS['raw']:,}"),
                ("Clean station-years", f"{len(data.clean):,}"),
                ("Canonical stations", f"{facts['canonical']}"),
                ("Flagged normals", f"{len(suspect)}"),
            ],
        )
    )

    ui.render(
        ui.stat_strip(
            [
                {"label": "Raw rows ingested", "value": f"{config.CLEANING_TOTALS['raw']:,}", "note": "Five source tables"},
                {"label": "Summary rows removed", "value": f"\u2212{config.CLEANING_TOTALS['summary']:,}", "note": "District and taluk lines"},
                {"label": "Duplicates removed", "value": f"\u2212{config.CLEANING_TOTALS['duplicates']}", "note": "2020 printed block"},
                {"label": "Clean station-years", "value": f"{config.CLEANING_TOTALS['clean']:,}", "note": "850 per season", "color": p["rain"]},
                {"label": "Name variants resolved", "value": f"{facts['name_variants']}", "note": f"\u2192 {facts['canonical']} canonical"},
                {"label": "Flagged baseline normals", "value": f"{len(suspect)}", "note": "Static normal, one-sided bias", "color": p["amber"]},
            ]
        )
    )

    # ---------------------------------------------------------------- flow --
    ui.render(ui.section("01", "Raw-to-clean transformation", "Every reduction accounted for"))
    ui.render(ui.flow(config.PIPELINE_STAGES))
    ui.render(
        ui.paragraph(
            "The pipeline retains 75.4% of raw rows as analysis records. Nothing was imputed: the "
            "missing 2023 season stays missing, and the temporal gap is carried as an explicit "
            "feature rather than filled with an estimate."
        )
    )

    # --------------------------------------------------------- ledger table --
    ui.render(ui.section("02", "Transformation ledger by season", "File-level audit trail"))
    ledger_rows = []
    for row in config.CLEANING_ROWS:
        ledger_rows.append(
            {
                "year": str(row["year"]),
                "file": row["file"],
                "raw": f"{row['raw']:,}",
                "dup": f"\u2212{row['duplicates']}" if row["duplicates"] else "0",
                "summary": f"\u2212{row['summary']:,}",
                "clean": f"{row['clean']:,}",
                "mean": ui.signed(row["mean_dep"], 1, "%"),
            }
        )
    ledger_rows.append(
        {
            "year": "Total",
            "file": "5 source tables",
            "raw": f"{config.CLEANING_TOTALS['raw']:,}",
            "dup": f"\u2212{config.CLEANING_TOTALS['duplicates']}",
            "summary": f"\u2212{config.CLEANING_TOTALS['summary']:,}",
            "clean": f"{config.CLEANING_TOTALS['clean']:,}",
            "mean": ui.signed(data.clean["departure_pct"].mean(), 1, "%"),
        }
    )
    ui.render(
        ui.table(
            [
                {"label": "Season", "key": "year", "cls": "v-num"},
                {"label": "Source table", "key": "file", "cls": "v-mono"},
                {"label": "Raw rows", "key": "raw", "align": "r"},
                {"label": "Duplicates", "key": "dup", "align": "r"},
                {"label": "Summary rows", "key": "summary", "align": "r"},
                {"label": "Clean records", "key": "clean", "align": "r"},
                {"label": "Mean departure", "key": "mean", "align": "r"},
            ],
            ledger_rows,
            caption="Raw input, removals and retained records per source table",
        )
    )

    # ------------------------------------------------------------ findings --
    ui.render(ui.section("03", "Normalization findings", "Five issues that materially shape the analysis"))
    ui.render(ui.findings(config.ISSUE_NOTES))
    ui.render(
        ui.callout(
            "green",
            "Formula verification",
            ui.paragraph(
                "Reported departures were independently recomputed as (actual \u2212 normal) / normal "
                f"\u00d7 100 across all {config.CLEANING_TOTALS['clean']:,} clean rows. No recomputed value "
                "deviated from the published figure by more than 2.0 percentage points. Station matching "
                f"used RapidFuzz at \u2265 {facts['min_match_score']}% similarity, with zero automated matches "
                "below that floor and borderline pairs written to an audit file for manual review."
            ),
        )
    )

    # ------------------------------------------------------ suspect normals --
    ui.render(ui.section("04", "Suspect baseline normals", f"{len(suspect)} stations with static normals"))

    agumbe = suspect[suspect["hobli"].str.casefold() == "agumbe"]
    case = agumbe.iloc[0] if len(agumbe) else suspect.sort_values("normal_mm", ascending=False).iloc[0]
    station_hist = data.clean[
        (data.clean["district"] == case["district"])
        & (data.clean["taluk"] == case["taluk"])
        & (data.clean["hobli"] == case["hobli"])
    ]
    peak = station_hist.sort_values("actual_mm", ascending=False).iloc[0] if len(station_hist) else None
    peak_text = (
        f"it still recorded <b>{peak['actual_mm']:,.1f} mm</b> in {int(peak['year'])}, a departure of "
        f"<b>{peak['departure_pct']:+.2f}%</b>."
        if peak is not None
        else "its departures are mechanically negative."
    )
    ui.render(
        ui.callout(
            "amber",
            f"Case study \u00b7 {case['hobli']}, {case['taluk']} ({case['district']})",
            ui.paragraph(
                f"This station's normal is fixed at <b>{case['normal_mm']:,.1f} mm</b>. {peak_text} "
                "The zone label is therefore an artifact of an uncalibrated extreme benchmark, not "
                "evidence of drought. 330 of the 338 flagged stations run persistently to excess; "
                "only 8 run persistently to deficit."
            ),
        )
    )

    f1, f2, f3 = st.columns([1.3, 1, 1], gap="medium")
    with f1:
        query = st.text_input("Search flagged stations", placeholder="Station, taluk or district")
    with f2:
        direction = st.selectbox(
            "Departure direction",
            ["All directions", "always_excess", "always_deficit"],
            format_func=lambda v: "All directions" if v == "All directions" else v.replace("_", " "),
        )
    with f3:
        rows_limit = st.selectbox("Rows shown", [50, 100, 250, "All"], index=1)

    view = suspect.copy()
    if query.strip():
        q = query.strip().lower()
        view = view[
            view["hobli"].str.lower().str.contains(q, regex=False)
            | view["taluk"].str.lower().str.contains(q, regex=False)
            | view["district"].str.lower().str.contains(q, regex=False)
        ]
    if direction != "All directions":
        view = view[view["direction"] == direction]
    view = view.sort_values("mean_departure_pct", ascending=True)

    total = len(view)
    shown = view if rows_limit == "All" else view.head(int(rows_limit))
    ui.render(
        ui.table(
            [
                {"label": "Station", "key": "station"},
                {"label": "Taluk", "key": "taluk"},
                {"label": "District", "key": "district"},
                {"label": "Static normal mm", "key": "normal", "align": "r"},
                {"label": "Mean observed mm", "key": "actual", "align": "r"},
                {"label": "Mean departure", "key": "dep", "align": "r"},
                {"label": "Range", "key": "range", "align": "r"},
                {"label": "Bias", "key": "bias"},
            ],
            [
                {
                    "station": r["hobli"],
                    "taluk": r["taluk"],
                    "district": r["district"],
                    "normal": ui.num(r["normal_mm"], 1),
                    "actual": ui.num(r["mean_actual_mm"], 1),
                    "dep": ui.signed(r["mean_departure_pct"], 2, "%"),
                    "range": f"{r['min_departure_pct']:+.0f} \u2026 {r['max_departure_pct']:+.0f}%",
                    "bias": {
                        "raw": ui.chip(
                            "excess" if r["direction"] == "always_excess" else "deficit",
                            "rain" if r["direction"] == "always_excess" else "terra",
                        )
                    },
                }
                for _, r in shown.iterrows()
            ],
            caption="Sorted from most negative mean departure. Range is the minimum and maximum observed departure across the panel.",
            empty="No flagged station matches this search.",
            foot=f"{len(shown):,} of {total:,} flagged stations shown" if rows_limit != "All" else f"{total:,} flagged stations",
        )
    )

    with st.expander("Methodology: how a normal was flagged as suspect"):
        ui.render(
            ui.paragraph(
                "A station is catalogued when its reported normal is unchanged across all observed "
                "years (difference \u2264 1.0 mm) and its departures remain one-sided. That combination "
                "indicates the baseline is a static administrative value rather than a rolling "
                "climatology. The full list is <code>reports/suspect_normals.csv</code>; these stations "
                "are flagged wherever they appear in the application and are never silently corrected."
            )
        )
