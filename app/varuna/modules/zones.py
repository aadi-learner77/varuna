"""
Module 02 -- Climatological Risk Zones
=====================================
Longitudinal K-Means (k=4) profiles over the five-year panel, plus the
filterable chronic-deficit registry.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, config, ui
from ..data import zone_of

PAGE = {
    "slug": "zones",
    "index": "02",
    "label": "Risk Zones",
    "title": "Climatological Risk Zones",
    "kicker": "Module 02 \u00b7 Climatology",
    "lede": (
        "Every station is reduced to one climatological profile across the observed panel, then "
        "clustered into four regimes. These are descriptions of historical behaviour \u2014 not "
        "predictions of the next drought."
    ),
}


def _cluster_stats(clusters: pd.DataFrame) -> dict:
    stats = {}
    total = len(clusters)
    for cid in sorted(config.CLUSTERS):
        sub = clusters[clusters["cluster_id"] == cid]
        stats[cid] = {
            "n": int(len(sub)),
            "share": (len(sub) / total * 100) if total else 0.0,
            "normal": float(sub["mean_normal_mm"].mean()),
            "departure": float(sub["mean_departure"].mean()),
            "std": float(sub["std_departure"].mean()),
            "deficit": float(sub["deficit_count"].mean()),
            "excess": float(sub["excess_count"].mean()),
            "suspect": int(sub["is_suspect_normal"].sum()),
        }
    return stats


def render(data, sel: dict, p: dict) -> None:
    clusters = data.clusters
    stats = _cluster_stats(clusters)
    station = data.station_cluster(sel["district"], sel["taluk"], sel["hobli"])

    ui.render(
        ui.page_header(
            PAGE["kicker"],
            PAGE["title"],
            PAGE["lede"],
            meta=[
                ("Panel", "5 observed seasons \u00b7 2020\u20132025"),
                ("Stations profiled", f"{clusters['station_id'].nunique():,}"),
                ("Clusters", "k = 4 \u00b7 silhouette 0.3406"),
                ("Features", "Departure mean, volatility, deficit & excess years, normal"),
            ],
        )
    )

    if station is not None:
        cid = int(station["cluster_id"])
        meta = config.CLUSTERS[cid]
        ui.render(
            ui.callout(
                "rain",
                f"Selected station \u00b7 Cluster {cid}",
                ui.paragraph(
                    f"<b>{ui.esc(sel['hobli'])}</b> ({ui.esc(sel['taluk'])}, {ui.esc(sel['district'])}) "
                    f"belongs to <b>{ui.esc(meta['name'])}</b>. Its five-year mean departure is "
                    f"<b>{station['mean_departure']:+.2f}%</b> with a standard deviation of "
                    f"{station['std_departure']:.2f} pp, {int(station['deficit_count'])} deficit "
                    f"{'year' if int(station['deficit_count']) == 1 else 'years'} and "
                    f"{int(station['excess_count'])} excess "
                    f"{'year' if int(station['excess_count']) == 1 else 'years'}. Mean normal: "
                    f"{station['mean_normal_mm']:,.1f} mm."
                    + (
                        " <b>Flagged baseline normal</b>: interpret the departure carefully."
                        if int(station["is_suspect_normal"])
                        else ""
                    )
                ),
            )
        )

    # ---------------------------------------------------------------- atlas --
    ui.render(ui.section("01", "Cluster atlas", "One profile per station \u00b7 shape and label, not colour alone"))
    ui.render(ui.cluster_tiles(stats, p))

    # ------------------------------------------------------- characteristics --
    ui.render(ui.section("02", "What separates the four regimes", "Departure bias, volatility and rainfall volume"))
    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.render(ui.sub_label("Mean departure and station-level spread"))
        st.plotly_chart(
            charts.cluster_profiles(stats, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Bars are cluster means of station five-year mean departures. Whiskers show the mean station standard deviation inside the cluster.")
    with right:
        ui.render(ui.sub_label("Rainfall volume against volatility"))
        st.plotly_chart(
            charts.cluster_map(clusters, stats, p),
            theme=None,
            config={"displayModeBar": False, "responsive": True},
            width="stretch",
        )
        st.caption("Position: mean station normal (mm) and mean departure volatility (pp). Marker area scales with the number of hoblis in the cluster.")

    ui.render(
        ui.callout(
            "amber",
            "How to read the clusters",
            ui.paragraph(
                "Cluster 3 concentrates the chronic-deficit behaviour: it holds the highest average "
                "number of deficit years (1.23 of 5) and the widest downside spread, while its mean "
                "normal is moderate. Cluster 1 is the large surplus-prone interior group, where "
                "positive departures dominate. Cluster 0 is the high-rainfall Western Ghats and "
                "coastal belt, where a large volumetric surplus can still be a small percentage. "
                "Assignment describes the observed record only; it does not project future drought."
            ),
        )
    )

    # -------------------------------------------------------------- registry --
    ui.render(ui.section("03", "Chronic-deficit registry", f"{len(clusters):,} station profiles \u00b7 filter, search, sort"))

    f1, f2, f3 = st.columns([1.25, 1, 1.25], gap="medium")
    with f1:
        cluster_labels = ["All clusters"] + [
            f"Cluster {cid} \u00b7 {config.CLUSTERS[cid]['short']}" for cid in sorted(config.CLUSTERS)
        ]
        cluster_choice = st.selectbox("Cluster", cluster_labels, index=4)
    with f2:
        min_deficit = st.selectbox("Minimum deficit years", ["Any", "1+", "2+", "3+", "4+", "5"], index=1)
    with f3:
        query = st.text_input("Search station, taluk or district", placeholder="e.g. Agumbe, Badami, MANDYA")

    g1, g2, g3 = st.columns([1.4, 1, 1], gap="medium")
    with g1:
        sort_key = st.selectbox(
            "Sort by",
            [
                "Mean departure (driest first)",
                "Deficit years",
                "Departure volatility",
                "Mean normal rainfall",
                "Station name",
            ],
            index=0,
        )
    with g2:
        order = st.selectbox("Order", ["Descending", "Ascending"], index=0)
    with g3:
        rows_limit = st.selectbox("Rows shown", [50, 100, 250, "All"], index=1)

    only_suspect = st.checkbox("Only stations with a flagged baseline normal", value=False)

    view = clusters.copy()
    if cluster_choice != "All clusters":
        cid = int(cluster_choice.split(" ")[1])
        view = view[view["cluster_id"] == cid]
    if min_deficit != "Any":
        view = view[view["deficit_count"] >= int(min_deficit.replace("+", ""))]
    if query.strip():
        q = query.strip().lower()
        view = view[
            view["hobli"].str.lower().str.contains(q, regex=False)
            | view["taluk"].str.lower().str.contains(q, regex=False)
            | view["district"].str.lower().str.contains(q, regex=False)
        ]
    if only_suspect:
        view = view[view["is_suspect_normal"] == 1]

    if sort_key == "Mean departure (driest first)":
        view = view.sort_values("mean_departure", ascending=(order == "Ascending"))
    elif sort_key == "Deficit years":
        view = view.sort_values(
            ["deficit_count", "mean_departure"],
            ascending=[order == "Ascending", order == "Ascending"],
        )
    elif sort_key == "Departure volatility":
        view = view.sort_values("std_departure", ascending=(order == "Ascending"))
    elif sort_key == "Mean normal rainfall":
        view = view.sort_values("mean_normal_mm", ascending=(order == "Ascending"))
    else:
        view = view.sort_values("hobli", ascending=(order == "Ascending"))

    total = len(view)
    shown = view if rows_limit == "All" else view.head(int(rows_limit))

    rows = []
    for _, r in shown.iterrows():
        cid = int(r["cluster_id"])
        rows.append(
            {
                "station": {
                    "raw": f'<span class="v-rowkey">{ui.esc(r["hobli"])}</span>'
                    + (
                        f' <span class="v-chip v-chip--amber" style="font-size:.52rem;padding:0 .25rem">flagged</span>'
                        if int(r["is_suspect_normal"])
                        else ""
                    )
                },
                "district": ui.esc(r["district"]),
                "taluk": ui.esc(r["taluk"]),
                "regime": {
                    "raw": f'<span style="display:inline-flex;align-items:center;gap:.4rem">'
                    f'{ui.glyph(cid, p, size=12)}<span class="v-dim" style="font-family:var(--v-mono);font-size:.65rem">C{cid}</span></span>'
                },
                "normal": ui.num(r["mean_normal_mm"], 0),
                "dep": ui.signed(r["mean_departure"], 1, "%"),
                "vol": ui.num(r["std_departure"], 1),
                "def": str(int(r["deficit_count"])),
                "exc": str(int(r["excess_count"])),
                "zone": {"raw": ui.zone_chip(zone_of(r["mean_departure"]))},
            }
        )

    if total:
        foot = f"{len(shown):,} of {total:,} matching hoblis shown" if rows_limit != "All" else f"{total:,} matching hoblis"
    else:
        foot = None

    ui.render(
        ui.table(
            [
                {"label": "Hobli", "key": "station"},
                {"label": "District", "key": "district"},
                {"label": "Taluk", "key": "taluk"},
                {"label": "Cluster", "key": "regime", "align": "c"},
                {"label": "Mean normal", "key": "normal", "align": "r"},
                {"label": "Mean dep.", "key": "dep", "align": "r"},
                {"label": "Volatility", "key": "vol", "align": "r"},
                {"label": "Deficit yrs", "key": "def", "align": "r"},
                {"label": "Excess yrs", "key": "exc", "align": "r"},
                {"label": "5-yr regime", "key": "zone"},
            ],
            rows,
            caption="Station climatological profiles \u00b7 deficit threshold \u2264 \u221220%, excess year threshold \u2265 +20%",
            empty="No station matches this combination of cluster, deficit minimum and search text.",
            foot=foot,
        )
    )

    with st.expander("Methodology: clustering, thresholds and what this is not"):
        ui.render(
            ui.paragraph(
                "Each station is represented by five aggregated statistics across its observed years: "
                "mean departure, departure standard deviation, deficit-year count (\u2264 \u221220%), "
                "excess-year count (\u2265 +20%) and mean normal rainfall. These are standardized and "
                "clustered with K-Means. k = 4 was selected as the highest-silhouette configuration "
                "(0.3406), and the fitted scaler and model are the stored pipeline artifacts. "
                "Deficit counts here use a +20% excess definition for historical frequency, which is "
                "deliberately lower than the +40% flood threshold used for the annual risk label. "
                "Cluster names are descriptive interpretations of the fitted group statistics. "
                "No cluster is a forecast, a probability, or a drought declaration."
            )
        )
