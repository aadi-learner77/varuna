"""
app/app.py
==========
Hobli-Level Rainfall Deviation and Drought-Risk Prediction for Karnataka Using Machine Learning.
Streamlit Web Application (Person 4 Delivery).

Features:
- Dependent dropdowns: District -> Taluk -> Hobli (sorted).
- Tab 1: Hobli Report (climatological comparison, actual vs normal rainfall, risk zones, model inference).
- Tab 2: Risk Zones (longitudinal 4-cluster K-Means categorization & deficit-prone audit).
- Tab 3: District Trends (district-level departure heatmap replacing spatial map without coordinates).
- Tab 4: Data Quality Audit (raw-to-clean reduction, duplicate blocks, spelling harmonization, suspect normals).
- Tab 5: Model Limits & Baselines (honest evaluation comparing ML against majority & persistence benchmarks).
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on sys.path for local module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.predict_risk import predict_risk
except ImportError as e:
    predict_risk = None
    _import_err = e

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Karnataka Rainfall & Drought-Risk Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# DATA LOADING WITH CACHING & ERROR HANDLING
# ------------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"


@st.cache_data
def load_datasets():
    clean_path = DATA_DIR / "clean" / "clean_rainfall.csv"
    features_path = DATA_DIR / "features" / "features.csv"
    clusters_path = DATA_DIR / "features" / "hobli_clusters.csv"
    suspect_path = REPORTS_DIR / "suspect_normals.csv"
    results_path = MODELS_DIR / "results.csv"

    errors = []
    for p in [clean_path, features_path, clusters_path, suspect_path, results_path]:
        if not p.exists():
            errors.append(str(p.relative_to(ROOT_DIR)))

    if errors:
        st.error(f"⚠️ Missing required repository files: {', '.join(errors)}. Please run data preparation and modelling pipelines first.")
        st.stop()

    df_clean = pd.read_csv(clean_path)
    df_feat = pd.read_csv(features_path)
    df_clusters = pd.read_csv(clusters_path)
    df_suspect = pd.read_csv(suspect_path)
    df_results = pd.read_csv(results_path)

    return df_clean, df_feat, df_clusters, df_suspect, df_results


df_clean, df_feat, df_clusters, df_suspect, df_results = load_datasets()

# Pre-calculate suspect hoblis set for fast O(1) lookup
suspect_keys = set(zip(df_suspect["district"], df_suspect["taluk"], df_suspect["hobli"]))

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DEPENDENT DROPDOWNS
# ------------------------------------------------------------------------------
st.sidebar.title("🌧️ Karnataka Rainfall")
st.sidebar.caption("Hobli-Level Drought-Risk & Deviation Predictor")
st.sidebar.markdown("---")

st.sidebar.subheader("Select Station")

# 1. District selection
district_list = sorted(df_clean["district"].unique())
selected_district = st.sidebar.selectbox("District", district_list, index=0)

# 2. Taluk selection (filtered by district)
taluk_list = sorted(df_clean[df_clean["district"] == selected_district]["taluk"].unique())
selected_taluk = st.sidebar.selectbox("Taluk", taluk_list, index=0)

# 3. Hobli selection (filtered by taluk and district)
hobli_sub = df_clean[
    (df_clean["district"] == selected_district) &
    (df_clean["taluk"] == selected_taluk)
]
hobli_list = sorted(hobli_sub["hobli"].unique())
selected_hobli = st.sidebar.selectbox("Hobli (Station)", hobli_list, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Scope**:
    - **5 Observed Years**: 2020, 2021, 2022, 2024, 2025.
    - **Missing Year**: 2023 has no hobli-level publications.
    - **Stations**: 850 canonical stations per year.
    - **Models**: Random Forest, Gradient Boosting, K-Means Clustering, Isolation Forest.
    """
)
st.sidebar.info("💡 **Tip**: Switch tabs to inspect individual hoblis, regional cluster patterns, data quality audits, and baseline comparisons.")

# ------------------------------------------------------------------------------
# APPLICATION HEADER
# ------------------------------------------------------------------------------
st.title("🌧️ Karnataka Hobli-Level Rainfall Deviation & Drought-Risk Prediction")
st.markdown(
    f"**Selected Station:** `{selected_hobli}` | **Taluk:** `{selected_taluk}` | **District:** `{selected_district}`"
)

# ------------------------------------------------------------------------------
# TABS SETUP
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Hobli Report",
    "🌐 Climatological Risk Zones",
    "🗺️ District Trends (Spatial Heatmap)",
    "🔍 Data Quality Audit",
    "⚖️ Model Limits & Baselines"
])

# ==============================================================================
# TAB 1: HOBLI REPORT
# ==============================================================================
with tab1:
    st.header(f"Station Report: {selected_hobli} ({selected_taluk}, {selected_district})")

    # Suspect normal warning banner
    is_suspect = (selected_district, selected_taluk, selected_hobli) in suspect_keys
    if is_suspect:
        sus_info = df_suspect[
            (df_suspect["district"] == selected_district) &
            (df_suspect["taluk"] == selected_taluk) &
            (df_suspect["hobli"] == selected_hobli)
        ].iloc[0]
        st.warning(
            f"⚠️ **Data Quality Warning: Unreliable Baseline Normal**  \n"
            f"**{selected_hobli}** is cataloged in `reports/suspect_normals.csv` with a static normal of "
            f"**{sus_info['normal_mm']:.1f} mm** and persistent **{sus_info['direction']}** departures "
            f"(mean departure: **{sus_info['mean_departure_pct']:+.2f}%**).  \n"
            f"*Important Caveat*: Like Agumbe in Shivamogga, extreme or one-sided departure values for this station "
            f"often represent a **data-quality artifact** arising from an uncalibrated historical benchmark rather than "
            f"genuine meteorological drought or excess."
        )

    # Filter station records across all observed years
    station_clean = df_clean[
        (df_clean["district"] == selected_district) &
        (df_clean["taluk"] == selected_taluk) &
        (df_clean["hobli"] == selected_hobli)
    ].sort_values("year").copy()

    # If duplicate sub-stations exist in same hobli name (e.g. Sarjapura_1), take first occurrence per year
    station_clean = station_clean.drop_duplicates(subset=["year"]).reset_index(drop=True)

    col_hist1, col_hist2 = st.columns([3, 2])

    with col_hist1:
        st.subheader("📈 Annual Rainfall vs. Long-Term Normal")
        st.caption("Observed annual records (2020–2025). *Note: KSNDMC published no hobli records for 2023.*")

        if len(station_clean) > 0:
            # Format years as string categories so missing 2023 is clearly non-consecutive
            chart_df = station_clean.copy()
            chart_df["year_str"] = chart_df["year"].astype(str)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=chart_df["year_str"],
                y=chart_df["actual_mm"],
                name="Actual Rainfall (mm)",
                marker_color="#1f77b4"
            ))
            fig.add_trace(go.Scatter(
                x=chart_df["year_str"],
                y=chart_df["normal_mm"],
                name="Normal Baseline (mm)",
                mode="lines+markers",
                line=dict(color="#d62728", width=3, dash="dash"),
                marker=dict(size=8, color="#d62728")
            ))
            fig.update_layout(
                xaxis_title="Observed Calendar Year",
                yaxis_title="Rainfall (mm)",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rainfall records found for this station.")

    with col_hist2:
        st.subheader("🏷️ Annual Observed Risk Status")
        st.caption(r"KSNDMC thresholds: Deficit $\le -20\%$, Excess $\ge +40\%$, Stable $-20\%$ to $+40\%$.")

        if len(station_clean) > 0:
            for _, r in station_clean.iterrows():
                yr = int(r["year"])
                act = r["actual_mm"]
                norm = r["normal_mm"]
                dep = r["departure_pct"]

                if dep <= -20.0:
                    badge_color = "#dc3545"  # Red
                    zone_label = "Drought-Prone"
                elif dep >= 40.0:
                    badge_color = "#0d6efd"  # Blue
                    zone_label = "Flood-Prone"
                else:
                    badge_color = "#198754"  # Green
                    zone_label = "Stable"

                st.markdown(
                    f"<div style='background-color: #f8f9fa; border-left: 5px solid {badge_color}; "
                    f"padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; display: flex; "
                    f"justify-content: space-between; align-items: center;'>"
                    f"<span><strong>{yr}</strong>: {act:.1f} mm (Normal: {norm:.1f} mm)</span>"
                    f"<span><span style='color: {'#d9534f' if dep < 0 else '#0275d8'}; font-weight: bold;'>{dep:+.1f}%</span> "
                    f"<span style='background-color: {badge_color}; color: white; padding: 2px 8px; "
                    f"border-radius: 12px; font-size: 0.8em; font-weight: 600;'>{zone_label}</span></span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No status records found.")

    st.markdown("---")

    # SECTION 2: MODEL PREDICTION (LATEST YEAR: 2025)
    st.subheader("🤖 Machine Learning Risk Assessment (Latest Year: 2025)")

    station_feat_2025 = df_feat[
        (df_feat["district"] == selected_district) &
        (df_feat["taluk"] == selected_taluk) &
        (df_feat["hobli"] == selected_hobli) &
        (df_feat["year"] == 2025)
    ]

    col_pred1, col_pred2 = st.columns([1, 1])

    if len(station_feat_2025) > 0 and predict_risk is not None:
        row_2025 = station_feat_2025.iloc[0]
        try:
            pred_res = predict_risk(row_2025)
            pred_zone = pred_res["risk_zone"]
            probs = pred_res["probabilities"]

            with col_pred1:
                zone_titles = {
                    "drought_prone": "Drought-Prone (Deficient Rainfall)",
                    "stable": "Stable (Normal Rainfall Regime)",
                    "flood_prone": "Flood-Prone (Large Excess Rainfall)"
                }
                zone_colors = {
                    "drought_prone": "#dc3545",
                    "stable": "#198754",
                    "flood_prone": "#0d6efd"
                }

                st.markdown(
                    f"<div style='padding: 16px; border-radius: 8px; border: 2px solid {zone_colors.get(pred_zone, '#333')}; "
                    f"background-color: rgba(240, 242, 246, 0.5); text-align: center; margin-top: 10px;'>"
                    f"<div style='font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; color: #666;'>Predicted Risk Zone (2025)</div>"
                    f"<div style='font-size: 1.6em; font-weight: 700; color: {zone_colors.get(pred_zone, '#333')}; margin: 8px 0;'>"
                    f"{zone_titles.get(pred_zone, pred_zone)}</div>"
                    f"<div style='font-size: 0.9em; color: #555;'>Model: <code>{pred_res.get('model_used', 'RandomForestClassifier')}</code></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.warning("⚠️ **Caution**: This is a model output with limited accuracy, see the Model Limits tab.")

            with col_pred2:
                prob_df = pd.DataFrame([
                    {"Class": "Drought-Prone", "Probability": probs.get("drought_prone", 0.0), "Color": "#dc3545"},
                    {"Class": "Stable", "Probability": probs.get("stable", 0.0), "Color": "#198754"},
                    {"Class": "Flood-Prone", "Probability": probs.get("flood_prone", 0.0), "Color": "#0d6efd"}
                ])
                fig_prob = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    color="Class",
                    color_discrete_map={
                        "Drought-Prone": "#dc3545",
                        "Stable": "#198754",
                        "Flood-Prone": "#0d6efd"
                    },
                    text_auto=".2%",
                    range_y=[0, 1.0],
                    title="Class Probability Distribution"
                )
                fig_prob.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=35, b=20),
                    height=240
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        except Exception as ex:
            st.error(f"Inference error calling src/predict_risk.py: {ex}")
    else:
        st.info("2025 feature record not available for this station to run inference.")

    st.markdown("---")

    # SECTION 3: DEPARTURE COMPARISON (HOBLI VS TALUK VS DISTRICT)
    st.subheader("📊 Spatial Context: Hobli vs. Taluk & District Departure")
    st.caption("Evaluates micro-climatic departures against surrounding administrative hierarchies for 2025.")

    hobli_2025 = station_clean[station_clean["year"] == 2025]
    if len(hobli_2025) > 0:
        h_dep = hobli_2025.iloc[0]["departure_pct"]

        # Calculate taluk average for 2025
        taluk_clean_2025 = df_clean[
            (df_clean["district"] == selected_district) &
            (df_clean["taluk"] == selected_taluk) &
            (df_clean["year"] == 2025)
        ]
        t_dep = taluk_clean_2025["departure_pct"].mean()

        # Calculate district average for 2025
        dist_clean_2025 = df_clean[
            (df_clean["district"] == selected_district) &
            (df_clean["year"] == 2025)
        ]
        d_dep = dist_clean_2025["departure_pct"].mean()

        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Station Departure", f"{h_dep:+.2f}%")
        c_m2.metric(f"Taluk Mean ({selected_taluk})", f"{t_dep:+.2f}%", f"{h_dep - t_dep:+.2f}% offset")
        c_m3.metric(f"District Mean ({selected_district})", f"{d_dep:+.2f}%", f"{h_dep - d_dep:+.2f}% offset")

        comp_df = pd.DataFrame([
            {"Entity": f"Hobli: {selected_hobli}", "Departure (%)": h_dep, "Type": "Hobli"},
            {"Entity": f"Taluk: {selected_taluk}", "Departure (%)": t_dep, "Type": "Taluk"},
            {"Entity": f"District: {selected_district}", "Departure (%)": d_dep, "Type": "District"},
        ])
        fig_comp = px.bar(
            comp_df,
            x="Entity",
            y="Departure (%)",
            color="Type",
            color_discrete_map={"Hobli": "#2ca02c", "Taluk": "#ff7f0e", "District": "#1f77b4"},
            text_auto="+.2f"
        )
        fig_comp.update_layout(
            yaxis_title="2025 Monsoon Departure (%)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=260
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("2025 observation not found for comparison.")

# ==============================================================================
# TAB 2: RISK ZONES & CLUSTERING
# ==============================================================================
with tab2:
    st.header("🌐 Climatological Hobli Clustering (K-Means k=4)")
    st.markdown(
        "Hoblis are aggregated across all 5 observed years into **one climatological profile per station** "
        "using mean departure, departure volatility (std), deficit frequency, excess frequency, and baseline rainfall normal. "
        "The optimal 4-cluster model achieved the highest silhouette score (**0.3406**)."
    )

    # Station's cluster profile
    station_cluster = df_clusters[
        (df_clusters["district"] == selected_district) &
        (df_clusters["taluk"] == selected_taluk) &
        (df_clusters["hobli"] == selected_hobli)
    ]

    if len(station_cluster) > 0:
        st_row = station_cluster.iloc[0]
        cid = int(st_row["cluster_id"])
        cname = st_row["cluster_name"]
        cdesc = st_row["cluster_desc"]

        st.info(
            f"📍 **Station Cluster Assignment**: `{selected_hobli}` belongs to **Cluster {cid}: {cname}**  \n"
            f"*{cdesc}*  \n"
            f"- **Mean Normal Rainfall**: {st_row['mean_normal_mm']:.1f} mm  \n"
            f"- **5-Year Mean Departure**: {st_row['mean_departure']:+.2f}% (Volatility $\\pm$ {st_row['std_departure']:.2f}%)  \n"
            f"- **Deficit Years (< -20%)**: {int(st_row['deficit_count'])} of 5 observed years | "
            f"**Excess Years (> +20%)**: {int(st_row['excess_count'])} of 5 observed years"
        )

    # Cluster summary table
    st.subheader("📊 Karnataka Climatological Cluster Profiles")

    summary_rows = []
    cluster_names = {
        0: "Western Ghats & Coastal High-Rainfall Belt",
        1: "High-Surplus / Flood-Prone Interior Plains",
        2: "Stable Semi-Arid Agricultural Plains",
        3: "Deficit-Vulnerable / Chronic-Drought Pocket"
    }

    for c in sorted(cluster_names.keys()):
        sub = df_clusters[df_clusters["cluster_id"] == c]
        summary_rows.append({
            "Cluster ID": c,
            "Cluster Name": cluster_names[c],
            "Hobli Count": len(sub),
            "Share (%)": f"{(len(sub) / len(df_clusters)) * 100:.1f}%",
            "Mean Normal (mm)": f"{sub['mean_normal_mm'].mean():.1f}",
            "Mean Departure (%)": f"{sub['mean_departure'].mean():+.2f}%",
            "Departure Std (%)": f"{sub['std_departure'].mean():.2f}%",
            "Avg Deficit Yrs": f"{sub['deficit_count'].mean():.2f}",
            "Avg Excess Yrs": f"{sub['excess_count'].mean():.2f}",
            "Suspect Normals": int(sub["is_suspect_normal"].sum())
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Filterable table of deficit-prone hoblis
    st.subheader("🔍 Filterable Registry: Deficit-Vulnerable Hoblis")
    st.caption("Inspect stations with chronic negative departures across Karnataka.")

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        cluster_filter = st.selectbox(
            "Filter by Cluster",
            ["All Clusters"] + [f"Cluster {c}: {cluster_names[c]}" for c in sorted(cluster_names.keys())],
            index=4  # Defaults to Cluster 3 (Deficit-Vulnerable)
        )
    with col_f2:
        min_deficits = st.slider("Minimum Deficit Years (out of 5)", 0, 5, 1)
    with col_f3:
        search_query = st.text_input("Search Station or District", "")

    filtered_df = df_clusters.copy()
    if cluster_filter != "All Clusters":
        sel_cid = int(cluster_filter.split(":")[0].replace("Cluster ", ""))
        filtered_df = filtered_df[filtered_df["cluster_id"] == sel_cid]

    filtered_df = filtered_df[filtered_df["deficit_count"] >= min_deficits]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["hobli"].str.lower().str.contains(q) |
            filtered_df["district"].str.lower().str.contains(q) |
            filtered_df["taluk"].str.lower().str.contains(q)
        ]

    filtered_df = filtered_df.sort_values(by=["deficit_count", "mean_departure"], ascending=[False, True])

    display_cols = [
        "district", "taluk", "hobli", "cluster_name",
        "mean_normal_mm", "mean_departure", "deficit_count", "excess_count", "is_suspect_normal"
    ]
    renamed_cols = {
        "district": "District",
        "taluk": "Taluk",
        "hobli": "Hobli",
        "cluster_name": "Cluster",
        "mean_normal_mm": "Mean Normal (mm)",
        "mean_departure": "Mean Departure (%)",
        "deficit_count": "Deficit Yrs",
        "excess_count": "Excess Yrs",
        "is_suspect_normal": "Suspect Normal?"
    }

    st.write(f"Showing **{len(filtered_df)}** matching stations:")
    st.dataframe(
        filtered_df[display_cols].rename(columns=renamed_cols).style.format({
            "Mean Normal (mm)": "{:.1f}",
            "Mean Departure (%)": "{:+.2f}%",
            "Deficit Yrs": "{:.0f}",
            "Excess Yrs": "{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# TAB 3: DISTRICT TRENDS (HEATMAP REPLACING SPATIAL MAP)
# ==============================================================================
with tab3:
    st.header("🗺️ District Departure Trends (Macro-Climatic Spatial Heatmap)")

    st.info(
        "📌 **Geographical Note**: The KSNDMC annual rainfall dataset does not contain geographical latitude "
        "and longitude coordinates. In strict accordance with data fidelity rules, coordinates were **not fabricated**. "
        "Instead, a complete **District-by-Year Departure Heatmap** and interactive district trajectory analysis "
        "provide the regional spatial context. Geospatial GIS polygon mapping is identified as high-priority future work."
    )

    # Compute district-year departure matrix
    dist_yr = df_clean.groupby(["district", "year"])["departure_pct"].mean().reset_index()
    dist_pivot = dist_yr.pivot(index="district", columns="year", values="departure_pct")
    dist_pivot = dist_pivot.sort_index()

    st.subheader("🔥 Annual Rainfall Departure Heatmap across 31 Districts")
    st.caption("Values indicate mean % departure from long-term normal. Red indicates deficit; Blue indicates excess.")

    fig_heat = px.imshow(
        dist_pivot,
        labels=dict(x="Observed Year", y="District", color="Departure (%)"),
        x=[str(y) for y in dist_pivot.columns],
        y=dist_pivot.index,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="auto"
    )
    fig_heat.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Bar chart for selected district over time
    st.subheader(f"📊 Historical Departure Trajectory: {selected_district}")
    dist_hist = dist_yr[dist_yr["district"] == selected_district].sort_values("year")
    dist_hist["year_str"] = dist_hist["year"].astype(str)

    fig_dbar = px.bar(
        dist_hist,
        x="year_str",
        y="departure_pct",
        color="departure_pct",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        text_auto="+.1f",
        labels={"year_str": "Observed Year", "departure_pct": "Mean Departure (%)"}
    )
    fig_dbar.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    fig_dbar.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Deficit Threshold (-20%)")
    fig_dbar.add_hline(y=40, line_dash="dash", line_color="blue", annotation_text="Excess Threshold (+40%)")
    fig_dbar.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=320
    )
    st.plotly_chart(fig_dbar, use_container_width=True)

# ==============================================================================
# TAB 4: DATA QUALITY AUDIT
# ==============================================================================
with tab4:
    st.header("🔍 Data Quality & Multi-Year Harmonization Audit")
    st.markdown(
        "OpenCity source data files were heavily unstandardized across formatting, header hierarchies, "
        "and geographical nomenclature. This tab documents the exact transformation metrics from "
        "`reports/cleaning_report.md`."
    )

    st.subheader("📋 Raw-to-Clean Pipeline Transformation Metrics")

    cleaning_table = [
        {"Year": 2020, "Original Source File": "741349d2-9f58-45ae-9744-b6d5c10a8b7f.csv", "Raw Rows": 1134, "Duplicates Dropped": 27, "Summary Rows Dropped": 257, "Clean Hoblis": 850, "Mean Dep (%)": "+20.98%"},
        {"Year": 2021, "Original Source File": "275cb296-b3aa-4c96-b614-c3c3ca19dec0.csv", "Raw Rows": 1113, "Duplicates Dropped": 0, "Summary Rows Dropped": 263, "Clean Hoblis": 850, "Mean Dep (%)": "+27.66%"},
        {"Year": 2022, "Original Source File": "d9c463e4-3203-49fd-95c3-33b79f85c9b3.csv", "Raw Rows": 1150, "Duplicates Dropped": 0, "Summary Rows Dropped": 300, "Clean Hoblis": 850, "Mean Dep (%)": "+50.67%"},
        {"Year": 2024, "Original Source File": "0ec5ad8d-be55-4e92-a942-6918c025735b.csv", "Raw Rows": 1122, "Duplicates Dropped": 0, "Summary Rows Dropped": 272, "Clean Hoblis": 850, "Mean Dep (%)": "+18.72%"},
        {"Year": 2025, "Original Source File": "karnataka_rainfall_2025_table1_1.csv", "Raw Rows": 1117, "Duplicates Dropped": 0, "Summary Rows Dropped": 267, "Clean Hoblis": 850, "Mean Dep (%)": "+13.21%"},
        {"Year": "Total", "Original Source File": "5 Source Tables", "Raw Rows": 5636, "Duplicates Dropped": 27, "Summary Rows Dropped": 1359, "Clean Hoblis": 4250, "Mean Dep (%)": "+26.25%"}
    ]
    st.dataframe(pd.DataFrame(cleaning_table), use_container_width=True, hide_index=True)

    st.markdown("---")

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.subheader("📑 Longitudinal Matching & Station Retention")
        st.markdown(
            """
            - **Raw Station String Variants**: **898** unique station identifiers found across raw tables.
            - **Harmonized Canonical Stations**: **877** canonical stations established via RapidFuzz.
            - **5-Year Persistent Stations**: **824 stations (94.0%)** appear in all 5 observed years.
            - **Uncertain Matches (< 85% match score)**: **0** (all automated matches exceed 85% similarity).
            - **Mathematical Formula Audit**: Recomputed `((actual - normal) / normal) * 100` matched reported numbers with **0 deviations > 2.0 pp** across all 4,250 rows.
            """
        )

    with col_q2:
        st.subheader("⚠️ Catalog of Key Data Anomalies & Edge Cases")
        st.markdown(
            """
            1. **2020 Printed Duplicate Block**: Indices 263–289 in raw 2020 contained a duplicate printed block of Chitradurga rows that had to be dropped.
            2. **Missing Year 2023**: KSNDMC published no hobli-level tables for 2023. Explicitly tracked using `gap_years = 2` between 2022 and 2024.
            3. **Vijayanagar District Bifurcation**: Late 2021 bifurcation carved 6 taluks out of Ballari into Vijayanagar. Standardized across all 5 years.
            4. **Suspect Baseline Normals**: **338 stations** exhibit static baseline normals alongside persistent one-sided departures (e.g., Agumbe normal = 7,565 mm).
            """
        )

    st.markdown("---")
    st.subheader(f"🚩 Suspect Baseline Normals Directory ({len(df_suspect)} Flagged Stations)")
    st.caption(r"Stations with static normals ($\Delta \le 1.0\text{ mm}$) and persistent one-sided departure bias.")
    st.dataframe(
        df_suspect[["district", "taluk", "hobli", "normal_mm", "direction", "mean_actual_mm", "mean_departure_pct", "notes"]].rename(columns={
            "district": "District",
            "taluk": "Taluk",
            "hobli": "Hobli",
            "normal_mm": "Static Normal (mm)",
            "direction": "Direction",
            "mean_actual_mm": "Mean Actual (mm)",
            "mean_departure_pct": "Mean Dep (%)",
            "notes": "Audit Notes"
        }).style.format({
            "Static Normal (mm)": "{:.1f}",
            "Mean Actual (mm)": "{:.1f}",
            "Mean Dep (%)": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# TAB 5: MODEL LIMITS & BASELINES
# ==============================================================================
with tab5:
    st.header("⚖️ Model Limits, Baselines & Honest Performance Evaluation")
    st.markdown(
        "> [!IMPORTANT]  \n"
        "> **Core Scientific Finding**: In severe class-imbalanced meteorological regimes, **raw accuracy is a misleading metric**. "
        "> All machine learning classifiers lose to the naive Majority-Class baseline on raw accuracy, but detect localized drought signals "
        "> that naive baselines completely miss."
    )

    st.subheader("📊 Empirical Results: ML Models vs. Heuristic Baselines")

    # Format results.csv into a clean display table
    res_display = df_results.copy()
    st.dataframe(
        res_display.style.format({
            "cv_accuracy": "{:.4f}",
            "cv_macro_f1": "{:.4f}",
            "test_accuracy": "{:.4f}",
            "test_macro_f1": "{:.4f}",
            "drought_recall": "{:.4f}",
            "test_mae": "{:.2f}%"
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.subheader("💡 Why Accuracy Alone Is Misleading Here")
        st.markdown(
            """
            1. **Class Dominance**: In the 2025 hold-out test set, **79.65% (677 of 850 hoblis)** belong to the `stable` class.
            2. **The Naive Majority Trap**: A trivial dummy baseline that blindly predicts `stable` for every single station achieves **79.65% test accuracy**.
            3. **Zero Utility**: Despite its high accuracy, the majority baseline has:
               - **0.00% Drought Recall** (identifies 0 of 66 drought hoblis).
               - **0.2956 Macro-F1**.
            4. **The Trade-Off**:
               - **Random Forest** achieves **70.82% accuracy** (-8.83 pp vs. majority), but achieves **0.4077 Macro-F1** (+0.1121 vs. majority) and captures localized drought events.
               - In 5-fold district cross-validation, Random Forest achieved **30.49% drought recall** across unobserved districts.
            """
        )

    with col_l2:
        st.subheader("📉 Why Regression Models Lose to the Zero Baseline")
        st.markdown(
            r"""
            1. **Macro-Regime Abatement**:
               - Training Window (2020–2024): Dominated by wet surplus anomalies culminating in the historic **2022 deluge (+50.67%)**.
               - Test Window (2025): State rainfall moderated sharply toward normal (**+13.20%**).
            2. **The Zero Baseline Benchmark**:
               - **Predict Zero Departure ($0.0\%$)**: Represents the long-term climatological normal ($actual = normal$). It achieved the lowest Mean Absolute Error (**21.65% MAE**).
               - **Machine Learning Regressors**: Gradient Boosting (**24.42% MAE**) and Random Forest (**26.35% MAE**) learned positive historical biases from earlier wet seasons and over-predicted departures.
            3. **Linear Extrapolation Failure (Ridge)**:
               - Ridge Regression was **dropped** because it assigned a massive positive weight to `n_prior_years`, extrapolating a false temporal trend and failing with an unacceptable **53.79% MAE**.
            """
        )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "Karnataka Hobli Rainfall & Drought-Risk Prediction System | Academic ML Project (Python, scikit-learn, Streamlit) | "
    "Data Source: KSNDMC via OpenCity under Open Data Commons (ODC-BY) license."
    "</div>",
    unsafe_allow_html=True
)
