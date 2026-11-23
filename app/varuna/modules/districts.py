"""
Module 03 -- District Rainfall Intelligence
==========================================
District-by-year departure matrices, anomaly distributions and district
comparisons. No map is drawn because the source tables carry no coordinates.
"""

from __future__ import annotations

import streamlit as st

from .. import charts, config, ui
from ..data import zone_of

PAGE = {
    "slug": "districts",
    "index": "03",
    "label": "District Intelligence",
    "title": "District Rainfall Intelligence",
    "kicker": "Module 03 \u00b7 Regional analysis",
    "lede": (
        "Departure behaviour across all 31 districts and five observed seasons, read as a "
        "spatial matrix. Unweighted station means, threshold classifications and year-by-year "
        "comparisons \u2014 with no invented geography."
    ),
}


def render(data, sel: dict, p: dict) -> None:
    clean = data.clean
    dist_year = data.district_year
    pivot = data.district_pivot
    years = data.years
    districts = pivot.index.tolist()

    ui.render(
        ui.page_header(
            PAGE["kicker"],
            PAGE["title"],
            PAGE["lede"],
            meta=[
                ("Districts", f"{len(districts)}"),
                ("Panel", "5 observed seasons"),
                ("Hold-out year", "2025"),
                ("Geography", "No coordinates in source data"),
            ],
        )
    )

    ui.render(
        ui.callout(
            "rain",
            "Why this is a matrix and not a map",
            ui.paragraph(
                "The KSNDMC annual tables contain no latitude, longitude or survey geometry. "
                "Coordinates and district polygons were left unfabricated. District means are "
                "unweighted averages of station departures, so they describe the station network "
                "rather than district area."
            ),
        )
    )

    # -------------------------------------------------------------- heatmap --
    ui.render(ui.section("01", "District departure matrix", "31 districts \u00d7 5 observed seasons"))
    st.plotly_chart(
        charts.district_heatmap(pivot, p),
        theme=None,
        config={"displayModeBar": False, "responsive": True},
        width="stretch",
    )
    st.caption("Cell value is the mean departure of all stations in that district for the season. The colour scale is symmetric around zero so equal surpluses and deficits read equally; 2023 is absent by source, not by omission.")

    # ---------------------------------------------------------- distribution --
    ui.render(ui.section("02", "Anomaly distribution and regime composition", "4,250 station-years \u00b7 850 per season"))
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        ui.render(ui.sub_label("Station-year departure distribution"))
        st.plotly_chart(
            charts.departure_distribution(clean, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        counts = {
            "deficit": int((clean["departure_pct"] <= config.DROUGHT_THRESHOLD).sum()),
            "stable": int(
                (
                    (clean["departure_pct"] > config.DROUGHT_THRESHOLD)
                    & (clean["departure_pct"] < config.EXCESS_THRESHOLD)
                ).sum()
            ),
            "excess": int((clean["departure_pct"] >= config.EXCESS_THRESHOLD).sum()),
        }
        st.caption(
            f"{counts['deficit']:,} station-years fall in the deficit band, {counts['stable']:,} are stable "
            f"and {counts['excess']:,} exceed +40%. The distribution is right-skewed: wet extremes are "
            "larger in magnitude than dry ones."
        )
    with right:
        ui.render(ui.sub_label("Regime composition by season"))
        st.plotly_chart(
            charts.regime_composition(data.features, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Stacked counts of the recorded risk-zone label per season. 2022 is the La Ni\u00f1a flood year; 2025 returns toward normal with localised deficit pockets.")

    # ------------------------------------------------------------ comparison --
    ui.render(ui.section("03", "District comparison", "Rank every district for one season"))
    controls = st.columns([1, 1.4], gap="medium")
    with controls[0]:
        year = st.selectbox("Season", years, index=len(years) - 1, format_func=lambda y: f"{y}")
    with controls[1]:
        focus = st.selectbox(
            "Focus district",
            districts,
            index=districts.index(sel["district"]) if sel["district"] in districts else 0,
        )

    focus_rows = dist_year[(dist_year["district"] == focus) & (dist_year["year"] == int(year))]
    focus_dep = float(focus_rows["departure_pct"].iloc[0]) if len(focus_rows) else 0.0
    focus_zone = zone_of(focus_dep)
    rank = int(
        (dist_year[dist_year["year"] == int(year)]["departure_pct"] > focus_dep).sum() + 1
    )

    chart_col, meta_col = st.columns([1.35, 1], gap="large")
    with chart_col:
        st.plotly_chart(
            charts.district_ranking(dist_year, int(year), focus, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption(f"{year} district means, driest at the bottom. The outlined bar is {focus}.")
    with meta_col:
        ui.render(
            f"""
            <div class="v-verdict" style="border-left:2px solid {p[config.ZONE_TONES[focus_zone]]}">
              <div class="v-verdict__label">{focus} \u00b7 {year}</div>
              <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin-top:.35rem">
                <div class="v-verdict__big" style="color:{p[config.ZONE_TONES[focus_zone]]}">{focus_dep:+.1f}%</div>
                {ui.zone_chip(focus_zone)}
              </div>
              <div class="v-verdict__meta" style="margin-top:.6rem">
                <span>Rank {rank} of {len(districts)}</span>
                <span>{int(focus_rows['stations'].iloc[0]) if len(focus_rows) else 0} stations</span>
              </div>
            </div>
            """
        )
        ui.render(
            ui.deflist(
                [
                    ("Driest district", f"{dist_year[dist_year['year'] == int(year)].sort_values('departure_pct').iloc[0]['district']}"),
                    ("Wettest district", f"{dist_year[dist_year['year'] == int(year)].sort_values('departure_pct').iloc[-1]['district']}"),
                    ("Districts in deficit", str(int((dist_year[dist_year['year'] == int(year)]['departure_pct'] <= config.DROUGHT_THRESHOLD).sum()))),
                    ("Districts in excess", str(int((dist_year[dist_year['year'] == int(year)]['departure_pct'] >= config.EXCESS_THRESHOLD).sum()))),
                ]
            )
        )
        ui.render(ui.sub_label(f"{focus} trajectory"))
        st.plotly_chart(
            charts.district_history(dist_year, focus, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )

    with st.expander("Methodology: district aggregation and thresholds"):
        ui.render(
            ui.paragraph(
                "District values are arithmetic means of every station departure available in that "
                "district and season, with no weighting by station area, record length or data quality. "
                "They are therefore network averages. Annual risk labels use the project thresholds: "
                f"deficit \u2264 {config.DROUGHT_THRESHOLD:.0f}%, stable between "
                f"{config.DROUGHT_THRESHOLD:.0f}% and +{config.EXCESS_THRESHOLD:.0f}%, excess "
                f"\u2265 +{config.EXCESS_THRESHOLD:.0f}%. Reading a district mean as an area-wide "
                "drought declaration would overstate what the station network can support."
            )
        )
