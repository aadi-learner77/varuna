"""
app/varuna/charts.py
====================
Chart builders. Every figure is derived from existing project data; no value
is synthesized for display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import (
    CLUSTERS,
    DROUGHT_THRESHOLD,
    EXCESS_THRESHOLD,
    MISSING_YEAR,
    ZONE_LABELS,
    ZONE_TONES,
)
from .theme import MONO, style_figure

def _c(p: dict, tone: str) -> str:
    return p.get(tone, p["ink"])


def _annotation(p: dict, **kwargs) -> dict:
    base = dict(
        font=dict(family=MONO, size=9.5, color=p["muted"]),
        showarrow=False,
        bgcolor="rgba(0,0,0,0)",
    )
    base.update(kwargs)
    return base


def _threshold_lines(fig, p: dict, *, vertical=True, annotate=True, labels=True):
    for value, tone, name in [
        (DROUGHT_THRESHOLD, "terra", "\u221220%"),
        (EXCESS_THRESHOLD, "rain", "+40%"),
    ]:
        kwargs = dict(
            line=dict(color=_c(p, tone), width=1, dash="dot"),
            opacity=0.9,
        )
        if vertical:
            fig.add_vline(x=value, **kwargs)
            if annotate and labels:
                fig.add_annotation(
                    x=value,
                    y=1.0,
                    yref="paper",
                    text=name,
                    **_annotation(p, xanchor="center", yanchor="bottom", font=dict(family=MONO, size=9, color=_c(p, tone))),
                )
        else:
            fig.add_hline(y=value, **kwargs)
            if annotate and labels:
                fig.add_annotation(
                    x=1.0,
                    xref="paper",
                    y=value,
                    text=name,
                    **_annotation(p, xanchor="right", yanchor="bottom", font=dict(family=MONO, size=9, color=_c(p, tone))),
                )


def _gap_marker(fig, p: dict, *, y=1.0):
    fig.add_vrect(
        x0=MISSING_YEAR - 0.46,
        x1=MISSING_YEAR + 0.46,
        fillcolor=p["tint_ink"],
        line_width=0,
        layer="below",
    )
    fig.add_annotation(
        x=MISSING_YEAR,
        y=0.985,
        yref="paper",
        text="2023 \u00b7 not published",
        **_annotation(p, yanchor="top", font=dict(family=MONO, size=8.5, color=p["muted"])),
    )


# --------------------------------------------------------------- module 01 --
def rainfall_history(hist: pd.DataFrame, p: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=hist["year"],
        y=hist["actual_mm"],
        name="Observed rainfall",
        marker=dict(color=_c(p, "rain"), line=dict(width=0)),
        width=0.5,
        hovertemplate="%{x}<br>%{y:,.1f} mm observed<extra></extra>",
    )
    fig.add_scatter(
        x=hist["year"],
        y=hist["normal_mm"],
        mode="lines+markers",
        name="Station normal",
        line=dict(color=_c(p, "muted"), width=1.6, dash="dash"),
        marker=dict(size=5, color=_c(p, "muted")),
        hovertemplate="%{x}<br>%{y:,.1f} mm normal<extra></extra>",
    )
    _gap_marker(fig, p)
    fig.update_xaxes(tickmode="array", tickvals=sorted(hist["year"].tolist()), tickformat="d")
    fig.update_yaxes(title_text="mm", ticksuffix="")
    style_figure(fig, p, height=310)
    return fig


def departure_context(hobli: float, taluk: float, district: float, p: dict) -> go.Figure:
    labels = ["Station", "Taluk mean", "District mean"]
    values = [hobli, taluk, district]
    colors = [_c(p, "ink"), _c(p, "teal"), _c(p, "rain")]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.1f}%" for v in values],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=p["ink_soft"]),
            hovertemplate="%{y}<br>%{x:+.2f}% departure<extra></extra>",
            width=0.52,
        )
    )
    fig.add_vline(x=0, line=dict(color=p["line_strong"], width=1))
    _threshold_lines(fig, p, annotate=True)
    fig.update_xaxes(title_text="2025 departure (%)", ticksuffix="%", range=[min(-30, min(values) - 15), max(60, max(values) + 15)])
    fig.update_yaxes(showgrid=False)
    style_figure(fig, p, height=230, legend=False)
    return fig


def outlook_trajectory(hist: pd.DataFrame, future: pd.DataFrame, p: dict) -> go.Figure:
    fig = go.Figure()
    if future is not None and len(future):
        years = [hist["year"].iloc[-1]] + future["year"].tolist()
        upper = [hist["actual_mm"].iloc[-1]] + future["upper_actual_mm"].tolist()
        lower = [hist["actual_mm"].iloc[-1]] + future["lower_actual_mm"].tolist()
        fig.add_scatter(
            x=years + years[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=p["tint_rain"],
            line=dict(width=0),
            hoverinfo="skip",
            name="Uncertainty band",
            showlegend=True,
        )
        fig.add_scatter(
            x=years,
            y=[hist["actual_mm"].iloc[-1]] + future["predicted_actual_mm"].tolist(),
            mode="lines+markers",
            name="Projected",
            line=dict(color=_c(p, "rain"), width=2, dash="dot"),
            marker=dict(size=7, symbol="diamond", color=_c(p, "rain")),
            hovertemplate="%{x}<br>%{y:,.1f} mm projected<extra></extra>",
        )
    fig.add_scatter(
        x=hist["year"],
        y=hist["actual_mm"],
        mode="lines+markers",
        name="Observed",
        line=dict(color=_c(p, "ink"), width=2),
        marker=dict(size=6, color=_c(p, "ink")),
        hovertemplate="%{x}<br>%{y:,.1f} mm observed<extra></extra>",
    )
    normal = float(hist["normal_mm"].iloc[-1])
    all_years = sorted(set(hist["year"].tolist() + ([int(y) for y in future["year"]] if future is not None and len(future) else [])))
    fig.add_scatter(
        x=all_years,
        y=[normal] * len(all_years),
        mode="lines",
        name="Normal",
        line=dict(color=_c(p, "muted"), width=1.2, dash="dash"),
        hovertemplate="Normal \u00b7 %{y:,.1f} mm<extra></extra>",
    )
    _gap_marker(fig, p)
    fig.update_xaxes(tickmode="array", tickvals=all_years, tickformat="d")
    fig.update_yaxes(title_text="mm")
    style_figure(fig, p, height=330)
    return fig


# --------------------------------------------------------------- module 02 --
def cluster_profiles(stats: dict, p: dict) -> go.Figure:
    order = sorted(stats)
    short_names = {0: "Ghats", 1: "Surplus", 2: "Stable", 3: "Deficit"}
    names = [f"C{c} \u00b7 {short_names.get(c, CLUSTERS[c]['short'])}" for c in order]
    means = [stats[c]["departure"] for c in order]
    stds = [stats[c]["std"] for c in order]
    colors = [_c(p, CLUSTERS[c]["tone"]) for c in order]
    fig = go.Figure(
        go.Bar(
            x=means,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            error_x=dict(type="data", array=stds, color=p["muted"], thickness=1, width=4),
            customdata=stds,
            hovertemplate="%{y}<br>mean departure %{x:+.2f}%<br>station spread \u00b1%{customdata:.1f} pp<extra></extra>",
            cliponaxis=False,
            width=0.55,
        )
    )
    fig.add_vline(x=0, line=dict(color=p["line_strong"], width=1))
    fig.add_vline(
        x=DROUGHT_THRESHOLD,
        line=dict(color=_c(p, "terra"), width=1, dash="dot"),
    )
    low = min(means) - max(stds) - 30
    high = max(means) + max(stds) + 15
    fig.update_xaxes(
        title_text="five-year mean departure (%)",
        ticksuffix="%",
        range=[low, high],
    )
    fig.update_yaxes(showgrid=False, autorange="reversed")
    style_figure(fig, p, height=290, legend=False)
    return fig


def cluster_map(clusters: pd.DataFrame, stats: dict, p: dict) -> go.Figure:
    fig = go.Figure()
    symbols = {"circle": "circle", "diamond": "diamond", "square": "square", "triangle": "triangle-up"}
    for cid in sorted(stats):
        sub = clusters[clusters["cluster_id"] == cid]
        meta = CLUSTERS[cid]
        fig.add_scatter(
            x=[sub["mean_normal_mm"].mean()],
            y=[sub["std_departure"].mean()],
            mode="markers+text",
            name=f"C{cid} \u00b7 {meta['short']}",
            text=[f"C{cid}"],
            textposition="top center",
            textfont=dict(family=MONO, size=10, color=_c(p, meta["tone"])),
            marker=dict(
                size=max(15, min(46, stats[cid]["share"] * 1.15 + 12)),
                symbol=symbols[meta["glyph"]],
                color=_c(p, meta["tone"]),
                opacity=0.85,
                line=dict(width=1, color=p["bg"]),
            ),
            customdata=[[stats[cid]["n"], stats[cid]["departure"], stats[cid]["share"]]],
            hovertemplate=(
                f"C{cid} \u00b7 {meta['short']}<br>"
                "mean normal %{x:,.0f} mm<br>mean volatility %{y:,.1f} pp<br>"
                "%{customdata[0]} hoblis \u00b7 %{customdata[2]:.1f}% of panel<extra></extra>"
            ),
        )
    fig.update_xaxes(title_text="mean station normal (mm)")
    fig.update_yaxes(title_text="departure volatility (pp)")
    style_figure(fig, p, height=330, legend=False)
    return fig


# --------------------------------------------------------------- module 03 --
def departure_distribution(clean: pd.DataFrame, p: dict) -> go.Figure:
    values = clean["departure_pct"].dropna()
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=56,
            marker=dict(color=_c(p, "rain"), line=dict(width=0)),
            opacity=0.8,
            hovertemplate="%{x}%<br>%{y} station-years<extra></extra>",
            name="Station-years",
        )
    )
    _threshold_lines(fig, p, annotate=True)
    fig.add_vline(x=0, line=dict(color=p["line_strong"], width=1))
    fig.update_xaxes(title_text="annual departure (%)", ticksuffix="%", range=[-65, 120])
    fig.update_yaxes(title_text="station-years")
    style_figure(fig, p, height=300, legend=False)
    return fig


def district_heatmap(pivot: pd.DataFrame, p: dict) -> go.Figure:
    z = pivot.values
    limit = max(50.0, float(np.ceil(np.nanmax(np.abs(z)) / 10.0) * 10))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(y) for y in pivot.columns],
            y=pivot.index.tolist(),
            zmid=0,
            zmin=-limit,
            zmax=limit,
            colorscale=[
                [0.0, _c(p, "terra")],
                [0.5, p["heat_zero"]],
                [1.0, _c(p, "rain")],
            ],
            xgap=1.5,
            ygap=1.5,
            colorbar=dict(
                title=dict(text="departure %", font=dict(family=MONO, size=9, color=p["muted"])),
                thickness=7,
                len=0.55,
                outlinewidth=0,
                tickfont=dict(family=MONO, size=8.5, color=p["muted"]),
                ticksuffix="%",
            ),
            hovertemplate="<b>%{y}</b> \u00b7 %{x}<br>%{z:+.1f}% mean departure<extra></extra>",
        )
    )
    fig.update_xaxes(side="top", showgrid=False, ticks="")
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont=dict(family=MONO, size=9, color=p["ink_soft"]))
    style_figure(fig, p, height=760, legend=False, margin=dict(l=6, r=10, t=26, b=6))
    return fig


def district_ranking(dist_year: pd.DataFrame, year: int, focus: str, p: dict) -> go.Figure:
    sub = dist_year[dist_year["year"] == year].sort_values("departure_pct", ascending=True)
    colors = [_c(p, ZONE_TONES[_zone(d)]) for d in sub["departure_pct"]]
    line_colors = [p["ink"] if d == focus else "rgba(0,0,0,0)" for d in sub["district"]]
    fig = go.Figure(
        go.Bar(
            x=sub["departure_pct"],
            y=sub["district"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=line_colors, width=[2 if d == focus else 0 for d in sub["district"]])),
            text=[f"{v:+.0f}%" for v in sub["departure_pct"]],
            textposition="outside",
            textfont=dict(family=MONO, size=8.5, color=p["ink_soft"]),
            hovertemplate="%{y}<br>%{x:+.2f}% mean departure \u00b7 %{customdata} stations<extra></extra>",
            customdata=sub["stations"].tolist(),
            width=0.62,
        )
    )
    fig.add_vline(x=0, line=dict(color=p["line_strong"], width=1))
    _threshold_lines(fig, p, annotate=True)
    fig.update_xaxes(title_text=f"{year} district mean departure (%)", ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(family=MONO, size=8.5, color=p["ink_soft"]))
    style_figure(fig, p, height=720, legend=False, margin=dict(l=6, r=16, t=22, b=6))
    return fig


def district_history(dist_year: pd.DataFrame, focus: str, p: dict) -> go.Figure:
    sub = dist_year[dist_year["district"] == focus].sort_values("year")
    colors = [_c(p, ZONE_TONES[_zone(v)]) for v in sub["departure_pct"]]
    fig = go.Figure(
        go.Bar(
            x=sub["year"],
            y=sub["departure_pct"],
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.0f}%" for v in sub["departure_pct"]],
            textposition="outside",
            textfont=dict(family=MONO, size=9.5, color=p["ink_soft"]),
            hovertemplate="%{x}<br>%{y:+.2f}% mean departure<extra></extra>",
            width=0.5,
        )
    )
    _threshold_lines(fig, p, annotate=True)
    fig.add_hline(y=0, line=dict(color=p["line_strong"], width=1))
    fig.update_xaxes(tickmode="array", tickvals=sorted(sub["year"].tolist()), tickformat="d")
    fig.update_yaxes(title_text="departure (%)", ticksuffix="%")
    style_figure(fig, p, height=280, legend=False)
    return fig


def regime_composition(features: pd.DataFrame, p: dict) -> go.Figure:
    counts = (
        features.groupby(["year", "risk_zone"]).size().unstack(fill_value=0).reindex(columns=["stable", "flood_prone", "drought_prone"], fill_value=0)
    )
    fig = go.Figure()
    for zone in ["stable", "flood_prone", "drought_prone"]:
        if zone not in counts:
            continue
        fig.add_bar(
            x=counts.index,
            y=counts[zone],
            name=ZONE_LABELS[zone],
            marker=dict(color=_c(p, ZONE_TONES[zone]), line=dict(width=0)),
            hovertemplate="%{x}<br>" + ZONE_LABELS[zone] + ": %{y} hoblis<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickmode="array", tickvals=sorted(counts.index.tolist()), tickformat="d")
    fig.update_yaxes(title_text="hoblis")
    style_figure(fig, p, height=280)
    return fig


# --------------------------------------------------------------- module 05 --
def accuracy_vs_recall(results: pd.DataFrame, p: dict) -> go.Figure:
    order = [
        "Most Common Class",
        "Persistence (Lag-1)",
        "Logistic Regression",
        "Gradient Boosting",
        "Random Forest",
    ]
    display = {
        "Most Common Class": "Majority class",
        "Persistence (Lag-1)": "Persistence",
        "Logistic Regression": "Logistic regression",
        "Gradient Boosting": "Gradient boosting",
        "Random Forest": "Random forest",
    }
    df = results[results["model_name"].isin(order)].set_index("model_name").loc[order].reset_index()
    fig = go.Figure()
    fig.add_bar(
        x=df["test_accuracy"] * 100,
        y=[display[m] for m in df["model_name"]],
        orientation="h",
        name="2025 accuracy",
        marker=dict(color=_c(p, "rain"), line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in df["test_accuracy"]],
        textposition="outside",
        textfont=dict(family=MONO, size=9, color=p["ink_soft"]),
        hovertemplate="%{y}<br>accuracy %{x:.2f}%<extra></extra>",
    )
    fig.add_bar(
        x=df["drought_recall"] * 100,
        y=[display[m] for m in df["model_name"]],
        orientation="h",
        name="2025 drought recall",
        marker=dict(color=_c(p, "terra"), line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in df["drought_recall"]],
        textposition="outside",
        textfont=dict(family=MONO, size=9, color=p["ink_soft"]),
        hovertemplate="%{y}<br>drought recall %{x:.2f}%<extra></extra>",
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="percent of 850 hold-out hoblis", ticksuffix="%", range=[0, 108])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    style_figure(fig, p, height=340)
    return fig


def macro_f1(results: pd.DataFrame, p: dict) -> go.Figure:
    order = [
        "Most Common Class",
        "Persistence (Lag-1)",
        "Logistic Regression",
        "Gradient Boosting",
        "Random Forest",
    ]
    display = {
        "Most Common Class": "Majority class",
        "Persistence (Lag-1)": "Persistence",
        "Logistic Regression": "Logistic regression",
        "Gradient Boosting": "Gradient boosting",
        "Random Forest": "Random forest",
    }
    df = results[results["model_name"].isin(order)].set_index("model_name").loc[order].reset_index()
    fig = go.Figure()
    fig.add_bar(
        x=df["test_macro_f1"],
        y=[display[m] for m in df["model_name"]],
        orientation="h",
        name="2025 test macro-F1",
        marker=dict(color=_c(p, "teal"), line=dict(width=0)),
        text=[f"{v:.3f}" for v in df["test_macro_f1"]],
        textposition="outside",
        textfont=dict(family=MONO, size=9, color=p["ink_soft"]),
        hovertemplate="%{y}<br>test macro-F1 %{x:.4f}<extra></extra>",
    )
    cv = df[["model_name", "cv_macro_f1"]].dropna()
    fig.add_scatter(
        x=cv["cv_macro_f1"],
        y=[display[m] for m in cv["model_name"]],
        mode="markers",
        name="5-fold district CV macro-F1",
        marker=dict(symbol="diamond-open", size=9, color=_c(p, "ink"), line=dict(width=1.4)),
        hovertemplate="%{y}<br>CV macro-F1 %{x:.4f}<extra></extra>",
    )
    majority_f1 = float(df.loc[df["model_name"] == "Most Common Class", "test_macro_f1"].iloc[0])
    fig.add_vline(x=majority_f1, line=dict(color=_c(p, "muted"), width=1, dash="dot"))
    fig.add_annotation(
        x=majority_f1,
        y=1.0,
        yref="paper",
        text="majority baseline",
        showarrow=False,
        font=dict(family=MONO, size=8.5, color=p["muted"]),
        xanchor="left",
        yanchor="bottom",
    )
    fig.update_xaxes(title_text="macro-F1 (0\u20131)", range=[0, 0.62])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    style_figure(fig, p, height=330)
    return fig


def regression_mae(results: pd.DataFrame, p: dict) -> go.Figure:
    reg = results[results["model_type"] == "regression"].copy()
    reg = reg.sort_values("test_mae", ascending=False)
    colors = []
    for name in reg["model_name"]:
        if name.startswith("Predict Zero"):
            colors.append(_c(p, "green"))
        elif "Dropped" in name:
            colors.append(_c(p, "terra"))
        else:
            colors.append(_c(p, "rain"))
    fig = go.Figure(
        go.Bar(
            x=reg["test_mae"],
            y=reg["model_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in reg["test_mae"]],
            textposition="outside",
            textfont=dict(family=MONO, size=9, color=p["ink_soft"]),
            hovertemplate="%{y}<br>MAE %{x:.2f}%<extra></extra>",
            width=0.55,
        )
    )
    fig.update_xaxes(title_text="2025 mean absolute error (percentage points)", ticksuffix="%", range=[0, 68])
    fig.update_yaxes(showgrid=False, tickfont=dict(family=MONO, size=9, color=p["ink_soft"]))
    style_figure(fig, p, height=270, legend=False)
    return fig


# ----------------------------------------------------------------- helpers --
def _zone(departure: float) -> str:
    if departure <= DROUGHT_THRESHOLD:
        return "drought_prone"
    if departure >= EXCESS_THRESHOLD:
        return "flood_prone"
    return "stable"
