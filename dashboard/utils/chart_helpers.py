"""
Plotly chart builders for the RIT Clinical Landscape dashboard.
Each function returns a plotly.graph_objects.Figure.

Chart index:
  Trial-level (Tab 1):
    1. build_trials_by_year_category()  — B1_C4_combined (stacked bar + novelty line)
    2. build_trials_by_year_phase()     — B2
    3. build_top_targets()              — C1
    4. build_isotope_distribution()     — D1
    5. build_format_pct_stacked()       — E2
    6. build_isotope_evolution()        — D2 (stacked bar, not heatmap)
    16. build_alpha_emitter_adoption()  — T9 (alpha emitter adoption + % of therapeutic line)
    17. build_isotope_industry_pct_line()    — T10 line (% industry-sponsored per isotope × period)
    18. build_isotope_industry_pct_heatmap() — T10 heatmap (same data as heatmap)
    19. build_isotope_indication_heatmap()        — T11 heatmap (trial count/row% per isotope × indication)
    20. build_isotope_indication_heatmap_assets() — T11 heatmap, asset-level (each drug counted once)
    21. build_novelty_combinations()              — 5D scatter/matrix: novel isotope × novel target
    22. build_novelty_scaffold_target()           — 5D extension: novel scaffold (format) × novel target

  Asset-level (Tab 2):
    7.  build_top_targets_assets()          — C1 assets
    8.  build_phase1_assets_by_period()     — C2 assets
    9.  build_new_assets_by_year()          — C4 assets
    10. build_isotope_distribution_assets() — D1 assets
    11. build_isotope_evolution_assets()    — D2 assets (NEW)
    12. build_format_evolution_assets()     — Format evolution (NEW, toggle)
    13. build_drug_landscape_scatter()      — Bubble: X=Year, Y=Target, Color=toggle, Symbol=Format, Size=N_Trials
    14. build_drug_phase_timeline()         — Bubble: X=Year, Y=Phase, Color=Isotope, Symbol=Format, Size=N_Trials
    15. build_company_portfolio_scatter()   — Bubble: Y=Company, X=Phase, Color=Isotope, Symbol=Format, Size=N_Trials
    23. build_pk_heatmap()                  — 5A heatmap: isotope (t½ sorted) × format (PK sorted), drug count
    24. build_pk_stacked_bar()              — 5A stacked bar: isotopes (t½ sorted), stacked by format (PK color)

  Explorer (Tab 3):
    build_explorer_bar()           — horizontal bar (high- or low-card primary)
    build_explorer_heatmap()       — count heatmap (low-card × low-card)
    build_explorer_time_bar()      — stacked bar over time (individual years or period bins)
    build_explorer_box()           — box plot (continuous primary × categorical group)
    build_explorer_histogram()     — histogram (continuous primary, no grouping)

  Helpers (for selectbox population in tabs):
    get_top_isotopes()   — returns top-N isotope list for a trials/assets DataFrame
    get_top_formats()    — returns format family list present in an assets DataFrame
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.utils.constants import (
    ALL_CATEGORIES, CAT_COLORS, PHASE_COLORS, PHASE_ORDER, QUAL_COLORS,
    REGULATORY_EVENTS, SPONSOR_COLORS,
)
from dashboard.utils.data_loader import (
    build_asset_hover_text, build_hover_text, diag_only, explode_multival,
    get_all_indication_groups, rit_only, top_values, year_bins,
)


# ── Event line styles ──────────────────────────────────────────────────
# Keyed by 'type' field in regulatory_events.yaml

_EVENT_STYLES: dict[str, dict] = {
    "approval":   {"color": "#2CA02C", "dash": "dash", "width": 1.5},
    "withdrawal": {"color": "#D62728", "dash": "dash", "width": 1.5},
    "lapse":      {"color": "#FF7F0E", "dash": "dot",  "width": 1.5},
    "threshold":  {"color": "#7F7F7F", "dash": "dot",  "width": 1.0},
}

# ── Bubble chart style maps (Charts 13+14) ─────────────────────────────

_ISOTOPE_FAMILY_COLORS: dict[str, str] = {
    "Alpha emitter": "#D62728",
    "Beta emitter":  "#1F77B4",
    "PET":           "#2CA02C",
    "SPECT":         "#FF7F0E",
}
_ISOTOPE_FAMILY_ORDER = ["Alpha emitter", "Beta emitter", "PET", "SPECT"]

_FORMAT_SYMBOLS: dict[str, str] = {
    "Full-length IgG":           "circle",
    "Bispecific":                "diamond",
    "mAb Fragment":              "square",
    "Small Ab-derived scaffold": "triangle-up",
    "Non-Ab protein scaffold":   "star",
}

_ASSET_OWNER_TYPES_YAML = Path(__file__).resolve().parent.parent / "data" / "asset_owner_types.yaml"
_DRUG_NAMES_YAML = Path(__file__).resolve().parent.parent / "data" / "drug_names.yaml"


def _load_drug_names() -> dict:
    """Load {Antibody_Name_Norm → {brand, inn, ...}} from drug_names.yaml."""
    if not _DRUG_NAMES_YAML.exists():
        return {}
    with open(_DRUG_NAMES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("drugs", {})


def _load_asset_owner_types() -> dict:
    """Load {Asset_Owner_Norm → {type, company, ...}} from asset_owner_types.yaml."""
    if not _ASSET_OWNER_TYPES_YAML.exists():
        return {}
    with open(_ASSET_OWNER_TYPES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("owners", {})


def _add_size_legend(
    fig: go.Figure,
    sizeref: float,
    max_n: float,
    sizemin: int = 5,
    group_title: str = "N (trials per drug)",
) -> None:
    """Append size-legend dummy traces (grey circles scaled to screen pixels).

    Picks 3 visually distinct N values: smallest (1), mid (~max/3 rounded), max.
    Pixel size is derived from the same sizemode='area' formula used in the data traces,
    so the legend icons match the chart bubbles.
    """
    def _px(n: float) -> float:
        """Screen pixel diameter for data value n under sizemode='area'."""
        return max(2.0 * np.sqrt(max(n, 0) / (sizeref * np.pi)), float(sizemin))

    def _nice(n: float, upper: float) -> int:
        """Round n to the nearest visually clean integer ≤ upper."""
        for d in (100, 50, 20, 10, 5, 2, 1):
            r = round(n / d) * d
            if 1 <= r <= upper:
                return int(r)
        return max(1, min(int(round(n)), int(upper)))

    max_n_int = max(1, int(max_n))
    if max_n_int <= 2:
        ns = [max_n_int]
    else:
        mid = _nice(max_n_int / 3, max_n_int - 1)
        ns = sorted({1, mid, max_n_int})
        # Drop items that are too close visually (< 4 px apart from the previous)
        kept = [ns[0]]
        for n in ns[1:]:
            if _px(n) - _px(kept[-1]) >= 4:
                kept.append(n)
        if kept[-1] != max_n_int:
            kept.append(max_n_int)
        ns = kept

    # Use the actual screen-pixel diameter so legend icons match chart bubbles.
    # Clamp to [5, 60] to stay within a readable legend area.
    for k, n_val in enumerate(ns):
        legend_size = max(5.0, min(60.0, _px(n_val)))
        fig.add_trace(go.Scatter(
            name=f"N = {n_val:,}",
            legendgroup="size",
            legendgrouptitle=dict(text=group_title) if k == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color="#888888", symbol="circle", size=legend_size),
            hoverinfo="skip",
        ))


def add_event_lines(
    fig: go.Figure,
    events: list[dict] | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
) -> go.Figure:
    """Add vertical regulatory event lines to a year-axis Plotly figure.

    Draws a dashed/dotted vertical line for each event with a small rotated
    annotation at the top.  Works with plain figures and make_subplots figures
    (including secondary-y layouts).

    Args:
        fig: Plotly figure whose primary x-axis carries integer years.
        events: List of event dicts from REGULATORY_EVENTS.  Defaults to the
                full catalog loaded from regulatory_events.yaml.
        x_min: Skip events before this year (inclusive lower bound).
        x_max: Skip events after this year (inclusive upper bound).

    Returns:
        The same figure with shapes and annotations added in-place.
    """
    if events is None:
        events = REGULATORY_EVENTS

    for ev in events:
        yr = int(ev["year"])
        if x_min is not None and yr < x_min:
            continue
        if x_max is not None and yr > x_max:
            continue

        style = _EVENT_STYLES.get(
            ev.get("type", "threshold"),
            {"color": "#9E9E9E", "dash": "dot", "width": 1.0},
        )

        # Vertical line — drawn below data so bars / markers stay on top
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=yr, x1=yr,
            y0=0, y1=1,
            line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
            layer="below",
        )

        # Rotated annotation at top of line (reads bottom → top)
        label = ev.get("label", str(yr)).replace("\n", " ")
        fig.add_annotation(
            xref="x",
            yref="paper",
            x=yr,
            y=0.99,
            xanchor="left",
            yanchor="top",
            text=label,
            showarrow=False,
            font=dict(size=8, color=style["color"]),
            textangle=-90,
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=1,
        )

    return fig


# ── Public helpers (used by tab widgets) ───────────────────────────────

def get_top_isotopes(df: pd.DataFrame, top_n: int, year_col: str = "Start_Year") -> list[str]:
    """Return the top-N isotopes by frequency, for populating a 'pin to bottom' selectbox.

    Handles compound isotope entries (e.g. 'Y-90; In-111') via explode.
    Works for both trial-level DataFrames (year_col='Start_Year') and
    asset-level DataFrames (year_col='First_Year').
    """
    df_iso = df.dropna(subset=[year_col, "Radioisotope_Norm"]).copy()
    df_exp = df_iso.assign(
        Isotope=df_iso["Radioisotope_Norm"].str.split("; ")
    ).explode("Isotope").dropna(subset=["Isotope"])
    df_exp["Isotope"] = df_exp["Isotope"].str.strip()
    return df_exp["Isotope"].value_counts().head(top_n).index.tolist()


def get_top_formats(df: pd.DataFrame, year_col: str = "First_Year") -> list[str]:
    """Return format families present in the DataFrame, ordered by frequency.

    Used to populate the 'pin to bottom' selectbox for format evolution charts.
    """
    df_f = df.dropna(subset=[year_col, "Format_Family"]).copy()
    return df_f["Format_Family"].value_counts().index.tolist()


# ── Shared helpers ─────────────────────────────────────────────────────

def _year_range_from_df(df: pd.DataFrame, col: str = "Start_Year") -> tuple[int, int]:
    years = df[col].dropna()
    if years.empty:
        cur = datetime.now().year
        return cur, cur
    return int(years.min()), min(int(years.max()), datetime.now().year)


def _fill_year_gaps(ct: pd.DataFrame) -> pd.DataFrame:
    """Reindex crosstab to fill missing years with 0."""
    all_years = range(int(ct.index.min()), int(ct.index.max()) + 1)
    return ct.reindex(all_years, fill_value=0)


# ── Chart 1: B1_C4_combined ────────────────────────────────────────────

def build_trials_by_year_category(df: pd.DataFrame) -> go.Figure:
    """Stacked bar (trials by year × category) + % new target antigen line.

    Args:
        df: Pre-filtered relevant trials (already filtered by sidebar controls).

    Returns:
        Plotly figure with secondary y-axis for novelty %.
    """
    df_comb = df.dropna(subset=["Start_Year"]).copy()
    df_comb["Start_Year"] = df_comb["Start_Year"].astype(int)
    y_min, y_max = _year_range_from_df(df_comb)
    df_comb = df_comb[(df_comb["Start_Year"] >= y_min) & (df_comb["Start_Year"] <= y_max)]

    ct_cat = pd.crosstab(df_comb["Start_Year"], df_comb["Category"])
    ct_cat = _fill_year_gaps(ct_cat)
    years = list(ct_cat.index)
    cat_cols = [c for c in ALL_CATEGORIES if c in ct_cat.columns]

    # Novelty line: % trials introducing a new target antigen
    # Compound bispecific targets (e.g. "EGFR; c-MET") are split into components;
    # a trial counts as "new" only if at least one component antigen is appearing
    # for the first time across all trials in the current filter selection.
    df_nov = df_comb.dropna(subset=["Target_Antigen_Norm"]).copy()
    if not df_nov.empty:
        def split_targets(s: str) -> list[str]:
            return [t.strip() for t in re.split(r"; | and ", s) if t.strip()]

        df_nov["_Components"] = df_nov["Target_Antigen_Norm"].apply(split_targets)

        # First appearance year for each individual component target
        df_exp = df_nov.explode("_Components").dropna(subset=["_Components"])
        df_exp = df_exp[df_exp["_Components"].str.strip() != ""]
        first_app_component = df_exp.groupby("_Components")["Start_Year"].min()

        # A trial is "new" if ANY of its component targets appears for the first time this year
        def _is_new(row):
            return any(
                row["Start_Year"] == first_app_component.get(comp, float("inf"))
                for comp in row["_Components"]
            )

        df_nov["Is_New"] = df_nov.apply(_is_new, axis=1)
        novelty = df_nov.groupby("Start_Year").agg(
            total=("NCT_ID", "count"), new=("Is_New", "sum")
        )
        novelty["pct_new"] = novelty["new"] / novelty["total"] * 100
        novelty = novelty.reindex(years, fill_value=0)

        # Hover: show the individual component targets that are new this year
        novelty_hover = []
        for yr in years:
            pct = novelty.loc[yr, "pct_new"] if yr in novelty.index else 0
            new_comps = sorted(
                comp for comp, first_yr in first_app_component.items() if first_yr == yr
            )
            tgt_lines = [f"· {t}" for t in new_comps[:8]]
            if len(new_comps) > 8:
                tgt_lines.append(f"<i>... and {len(new_comps) - 8} more</i>")
            tgt_str = "<br>".join(tgt_lines) if tgt_lines else "(none)"
            novelty_hover.append(
                f"% new target: <b>{pct:.1f}%</b><br>"
                f"New target antigens this year:<br>{tgt_str}"
            )
    else:
        novelty = pd.DataFrame({"pct_new": [0] * len(years)}, index=years)
        novelty_hover = ["(no data)"] * len(years)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stacked bars — one trace per category
    for cat in cat_cols:
        counts = ct_cat[cat].tolist()

        # Build hover text for each (year, category) bar segment
        hover_texts = []
        for yr in years:
            subset = df_comb[
                (df_comb["Start_Year"] == yr) & (df_comb["Category"] == cat)
            ]
            hover_texts.append(build_hover_text(subset))

        fig.add_trace(
            go.Bar(
                name=cat,
                x=years,
                y=counts,
                marker_color=CAT_COLORS.get(cat, "#CCCCCC"),
                customdata=hover_texts,
                hovertemplate=(
                    "<b>%{x} · " + cat + "</b><br>"
                    "Trials: <b>%{y}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    _NOVELTY_COLOR = "black"

    # Novelty line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            name="% of trials with new target",
            x=years,
            y=novelty["pct_new"].round(1).tolist(),
            mode="lines+markers",
            line=dict(color=_NOVELTY_COLOR, width=2),
            marker=dict(size=5),
            customdata=novelty_hover,
            hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(t=60, b=40, l=60, r=60),
    )
    fig.update_yaxes(title_text="Number of trials", secondary_y=False, gridcolor="#EEEEEE")
    fig.update_yaxes(
        title_text="% of trials with new target",
        secondary_y=True,
        range=[0, 110],
        ticksuffix="%",
        showgrid=False,
        tickfont=dict(color=_NOVELTY_COLOR),
        title_font_color=_NOVELTY_COLOR,
    )

    return fig


# ── Chart 2: B2 — Trials by year × phase ──────────────────────────────

def build_trials_by_year_phase(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: trials by year, colored by phase."""
    df_ph = df.dropna(subset=["Start_Year", "Phase_Norm"]).copy()
    df_ph["Start_Year"] = df_ph["Start_Year"].astype(int)

    ct = pd.crosstab(df_ph["Start_Year"], df_ph["Phase_Norm"])
    ct = _fill_year_gaps(ct)
    years = list(ct.index)
    phase_cols = [p for p in PHASE_ORDER if p in ct.columns]

    fig = go.Figure()

    for phase in phase_cols:
        counts = ct[phase].tolist()

        hover_texts = []
        for yr in years:
            subset = df_ph[
                (df_ph["Start_Year"] == yr) & (df_ph["Phase_Norm"] == phase)
            ]
            hover_texts.append(build_hover_text(subset))

        fig.add_trace(
            go.Bar(
                name=phase,
                x=years,
                y=counts,
                marker_color=PHASE_COLORS.get(phase, "#CCCCCC"),
                customdata=hover_texts,
                hovertemplate=(
                    "<b>%{x} · " + phase + "</b><br>"
                    "Trials: <b>%{y}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Year",
        yaxis_title="Number of trials",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            traceorder="normal",
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(t=60, b=40, l=60, r=60),
        yaxis=dict(gridcolor="#EEEEEE"),
    )

    return fig


# ── Chart 3: C1 — Top targets (side-by-side, RIT vs Diagnostic) ────────

def build_top_targets(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Side-by-side horizontal bar: top targets for RIT vs Diagnostic."""
    rit = rit_only(df)
    diag = diag_only(df)

    rit_counts = top_values(explode_multival(rit["Target_Antigen_Norm"]), n=top_n)
    diag_counts = top_values(explode_multival(diag["Target_Antigen_Norm"]), n=top_n)
    max_x = max(
        rit_counts.max() if not rit_counts.empty else 0,
        diag_counts.max() if not diag_counts.empty else 0,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Therapeutic RIT", "Diagnostic (ImmunoPET/SPECT)"],
        horizontal_spacing=0.12,
    )

    def _hover_for_target(df_sub, target_col, target_val):
        mask = df_sub[target_col].str.contains(
            target_val.replace("(", r"\(").replace(")", r"\)"), na=False
        )
        return build_hover_text(df_sub[mask])

    # RIT bars
    if not rit_counts.empty:
        fig.add_trace(
            go.Bar(
                x=rit_counts.values[::-1],
                y=rit_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Therapeutic"],
                customdata=[
                    _hover_for_target(rit, "Target_Antigen_Norm", t)
                    for t in rit_counts.index[::-1]
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Trials: <b>%{x}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Diagnostic bars
    if not diag_counts.empty:
        fig.add_trace(
            go.Bar(
                x=diag_counts.values[::-1],
                y=diag_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Diagnostic"],
                customdata=[
                    _hover_for_target(diag, "Target_Antigen_Norm", t)
                    for t in diag_counts.index[::-1]
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Trials: <b>%{x}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )

    n_rows = max(len(rit_counts), len(diag_counts), 5)
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(350, n_rows * 28 + 120),
        margin=dict(t=60, b=40, l=160, r=40),
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    if max_x > 0:
        fig.update_layout(xaxis_range=[0, max_x * 1.1], xaxis2_range=[0, max_x * 1.1])

    return fig


# ── Chart 4: D1 — Isotope distribution (side-by-side) ─────────────────

def build_isotope_distribution(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Side-by-side horizontal bar: top isotopes for RIT vs Diagnostic."""
    rit = rit_only(df)
    diag = diag_only(df)

    rit_counts = top_values(explode_multival(rit["Radioisotope_Norm"]), n=top_n)
    diag_counts = top_values(explode_multival(diag["Radioisotope_Norm"]), n=top_n)
    max_x = max(
        rit_counts.max() if not rit_counts.empty else 0,
        diag_counts.max() if not diag_counts.empty else 0,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Therapeutic RIT", "Diagnostic (ImmunoPET/SPECT)"],
        horizontal_spacing=0.12,
    )

    def _hover_for_isotope(df_sub, iso_val):
        mask = df_sub["Radioisotope_Norm"].str.contains(
            iso_val.replace("-", r"\-"), na=False
        )
        return build_hover_text(df_sub[mask])

    if not rit_counts.empty:
        fig.add_trace(
            go.Bar(
                x=rit_counts.values[::-1],
                y=rit_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Therapeutic"],
                customdata=[_hover_for_isotope(rit, t) for t in rit_counts.index[::-1]],
                hovertemplate="<b>%{y}</b><br>Trials: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    if not diag_counts.empty:
        fig.add_trace(
            go.Bar(
                x=diag_counts.values[::-1],
                y=diag_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Diagnostic"],
                customdata=[_hover_for_isotope(diag, t) for t in diag_counts.index[::-1]],
                hovertemplate="<b>%{y}</b><br>Trials: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    n_rows = max(len(rit_counts), len(diag_counts), 5)
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(350, n_rows * 28 + 120),
        margin=dict(t=60, b=40, l=100, r=40),
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    if max_x > 0:
        fig.update_layout(xaxis_range=[0, max_x * 1.1], xaxis2_range=[0, max_x * 1.1])

    return fig


# ── Chart 5: E2 — Format family % stacked bar ──────────────────────────

def build_format_pct_stacked(df: pd.DataFrame, mode: str = "pct", bin_size: int = 5) -> go.Figure:
    """Stacked bar: antibody format family share (or count) by period.

    Args:
        df: Pre-filtered trials DataFrame.
        mode: 'pct' = 100% stacked (share per period); 'count' = absolute trial counts.
        bin_size: Width of time bins in years (e.g. 3, 5, 10).
    """
    df_fmt = df.dropna(subset=["Start_Year", "Format_Family"]).copy()
    df_fmt["Period"] = year_bins(df_fmt, bin_size=bin_size).astype(str)
    df_fmt = df_fmt[df_fmt["Period"] != "nan"]

    ct = pd.crosstab(df_fmt["Period"], df_fmt["Format_Family"])
    periods = ct.index.tolist()
    totals = ct.sum(axis=1)
    pct = ct.div(totals, axis=0) * 100
    formats = ct.columns.tolist()
    color_map = {fmt: QUAL_COLORS[i] for i, fmt in enumerate(formats)}

    fig = go.Figure()

    for fmt in formats:
        color = color_map[fmt]
        if mode == "pct":
            y_vals = pct[fmt].round(1).tolist()
            customdata = [
                f"Count: {ct.loc[p, fmt]}<br>Total: {totals[p]}"
                if p in ct.index else ""
                for p in periods
            ]
            hover = (
                "<b>%{x} · " + fmt + "</b><br>"
                "Share: <b>%{y:.1f}%</b><br>"
                "%{customdata}<extra></extra>"
            )
        else:
            y_vals = ct[fmt].tolist()
            customdata = [
                f"Total: {totals[p]}"
                if p in ct.index else ""
                for p in periods
            ]
            hover = (
                "<b>%{x} · " + fmt + "</b><br>"
                "Trials: <b>%{y}</b><br>"
                "%{customdata}<extra></extra>"
            )
        fig.add_trace(go.Bar(
            name=fmt,
            x=periods,
            y=y_vals,
            marker_color=color,
            customdata=customdata,
            hovertemplate=hover,
        ))

    if mode == "pct":
        yaxis = dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE")
        ylab = "Share of trials (%)"
    else:
        yaxis = dict(gridcolor="#EEEEEE")
        ylab = "Number of trials"

    fig.update_layout(
        barmode="stack",
        xaxis_title=f"{bin_size}-year period",
        yaxis_title=ylab,
        yaxis=yaxis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
    )

    return fig


# ── Chart 6: D2 — Isotope evolution (stacked bar) ─────────────────────

def build_isotope_evolution(
    df: pd.DataFrame, top_n: int = 8, pin_to_bottom: str | None = None,
    mode: str = "count", bin_size: int = 5,
) -> go.Figure:
    """Stacked bar: trial counts (or share %) by isotope × period.

    Args:
        df: Pre-filtered trials DataFrame.
        top_n: Number of isotopes to show.
        pin_to_bottom: Isotope to place at the bottom of the stack (first trace).
                       None = frequency order (most common on bottom).
        mode: 'count' = absolute trial counts; 'pct' = 100% stacked (share per period).
        bin_size: Width of time bins in years (e.g. 3, 5, 10).
    """
    df_iso = df.dropna(subset=["Start_Year"]).copy()
    df_iso["Period"] = year_bins(df_iso, bin_size=bin_size).astype(str)
    df_iso = df_iso[df_iso["Period"] != "nan"]

    # Explode compound isotope entries
    df_exp = df_iso.copy()
    df_exp = df_exp.assign(
        Isotope=df_exp["Radioisotope_Norm"].str.split("; ")
    ).explode("Isotope").dropna(subset=["Isotope"]).reset_index(drop=True)
    df_exp["Isotope"] = df_exp["Isotope"].str.strip()

    # Top N isotopes overall; pin_to_bottom moves selected isotope to position 0
    top_isos = df_exp["Isotope"].value_counts().head(top_n).index.tolist()
    # Stable color map keyed by name so pinning doesn't shift colors
    color_map = {iso: QUAL_COLORS[i % len(QUAL_COLORS)] for i, iso in enumerate(top_isos)}
    if pin_to_bottom and pin_to_bottom in top_isos:
        top_isos = [pin_to_bottom] + [iso for iso in top_isos if iso != pin_to_bottom]
    df_exp = df_exp[df_exp["Isotope"].isin(top_isos)]

    ct = pd.crosstab(df_exp["Period"], df_exp["Isotope"])
    periods = ct.index.tolist()
    totals = ct.sum(axis=1)
    pct = ct.div(totals, axis=0) * 100

    fig = go.Figure()

    for iso in top_isos:
        color = color_map[iso]
        if iso not in ct.columns:
            continue
        if mode == "pct":
            y_vals = pct[iso].round(1).tolist()
            customdata = [
                f"Count: {ct.loc[p, iso]}<br>Total: {totals[p]}"
                if p in ct.index else ""
                for p in periods
            ]
            hover = (
                "<b>%{x} · " + iso + "</b><br>"
                "Share: <b>%{y:.1f}%</b><br>"
                "%{customdata}<extra></extra>"
            )
        else:
            y_vals = ct[iso].tolist()
            customdata = None
            hover = (
                "<b>%{x} · " + iso + "</b><br>"
                "Trials: <b>%{y}</b><extra></extra>"
            )
        fig.add_trace(go.Bar(
            name=iso,
            x=periods,
            y=y_vals,
            marker_color=color,
            customdata=customdata,
            hovertemplate=hover,
        ))

    if mode == "pct":
        yaxis = dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE")
        ylab = "Share of trials (%)"
    else:
        yaxis = dict(gridcolor="#EEEEEE")
        ylab = "Number of trials"

    fig.update_layout(
        barmode="stack",
        xaxis_title=f"{bin_size}-year period",
        yaxis_title=ylab,
        yaxis=yaxis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
    )

    return fig


# ── Chart 7: C1 assets — Top targets by unique asset count ─────────────

def build_top_targets_assets(assets: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Side-by-side horizontal bar: top targets by unique asset count."""
    thera = assets[assets["Category"] == "Therapeutic"]
    diag = assets[assets["Category"] == "Diagnostic"]

    def _counts_and_hover(df_sub, top_n):
        counts = top_values(explode_multival(df_sub["Target_Antigen_Norm"]), n=top_n)
        hover = []
        for t in counts.index[::-1]:
            mask = df_sub["Target_Antigen_Norm"].str.contains(
                t.replace("(", r"\(").replace(")", r"\)"), na=False
            )
            # build_asset_hover_text shows Antibody · (Isotope) per matched asset
            hover.append(build_asset_hover_text(df_sub[mask]))
        return counts, hover

    thera_counts, thera_hover = _counts_and_hover(thera, top_n)
    diag_counts, diag_hover = _counts_and_hover(diag, top_n)
    max_x = max(
        thera_counts.max() if not thera_counts.empty else 0,
        diag_counts.max() if not diag_counts.empty else 0,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Therapeutic drugs", "Diagnostic drugs"],
        horizontal_spacing=0.12,
    )

    if not thera_counts.empty:
        fig.add_trace(
            go.Bar(
                x=thera_counts.values[::-1],
                y=thera_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Therapeutic"],
                customdata=thera_hover,
                hovertemplate="<b>%{y}</b><br>Drugs: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    if not diag_counts.empty:
        fig.add_trace(
            go.Bar(
                x=diag_counts.values[::-1],
                y=diag_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Diagnostic"],
                customdata=diag_hover,
                hovertemplate="<b>%{y}</b><br>Drugs: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    n_rows = max(len(thera_counts), len(diag_counts), 5)
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(350, n_rows * 28 + 120),
        margin=dict(t=60, b=40, l=160, r=40),
    )
    fig.update_xaxes(gridcolor="#EEEEEE", title_text="Unique drugs")
    if max_x > 0:
        fig.update_layout(xaxis_range=[0, max_x * 1.1], xaxis2_range=[0, max_x * 1.1])

    return fig


# ── Charts 8+9: C2/C4 assets — New drugs per year or 5-year period ─────

def build_new_assets_combined(assets: pd.DataFrame, mode: str = "all", bin_size: int = 5) -> go.Figure:
    """Grouped bar: new drugs per period, Therapeutic vs Diagnostic side by side.

    Args:
        assets: Asset-level DataFrame.
        mode: 'all' = all drugs by year (annual x-axis);
              'phase1' = Phase 1/Early-phase drugs only, binned to N-year periods.
        bin_size: Width of time bins in years; only used in 'phase1' mode.
    """
    if mode == "phase1":
        df_a = assets[
            assets["Highest_Phase_Norm"].str.contains("Phase 1", na=False)
            | assets["Highest_Phase_Norm"].str.startswith("Early", na=False)
        ].dropna(subset=["First_Year"]).copy()
        df_a["First_Year"] = df_a["First_Year"].astype(int)
        df_a["_x"] = year_bins(df_a, col="First_Year", bin_size=bin_size).astype(str)
        df_a = df_a[df_a["_x"] != "nan"]
        x_vals = sorted(df_a["_x"].unique())
        xlab = f"{bin_size}-year period"
    else:
        df_a = assets.dropna(subset=["First_Year"]).copy()
        df_a["First_Year"] = df_a["First_Year"].astype(int)
        x_vals = list(range(int(df_a["First_Year"].min()), int(df_a["First_Year"].max()) + 1))
        df_a["_x"] = df_a["First_Year"]
        xlab = "Year"

    cats = [c for c in ALL_CATEGORIES if c in df_a["Category"].unique()]
    fig = go.Figure()

    for cat in cats:
        df_cat = df_a[df_a["Category"] == cat]
        counts = df_cat.groupby("_x").size().reindex(x_vals, fill_value=0)
        hover_texts = [
            build_asset_hover_text(df_cat[df_cat["_x"] == x]) for x in x_vals
        ]
        fig.add_trace(go.Bar(
            name=cat,
            x=x_vals,
            y=counts.tolist(),
            marker_color=CAT_COLORS.get(cat, "#CCC"),
            customdata=hover_texts,
            hovertemplate=(
                "<b>%{x} · " + cat + "</b><br>"
                "New drugs: <b>%{y}</b><br>"
                "<br>%{customdata}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title=xlab,
        yaxis_title="Number of new drugs",
        yaxis=dict(gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(t=60, b=40, l=60, r=40),
    )

    return fig


# ── Chart 10: D1 assets — Isotope distribution by asset count ──────────

def build_isotope_distribution_assets(assets: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Side-by-side horizontal bar: top isotopes by unique asset count."""
    thera = assets[assets["Category"] == "Therapeutic"]
    diag = assets[assets["Category"] == "Diagnostic"]

    thera_counts = top_values(explode_multival(thera["Radioisotope_Norm"]), n=top_n)
    diag_counts = top_values(explode_multival(diag["Radioisotope_Norm"]), n=top_n)
    max_x = max(
        thera_counts.max() if not thera_counts.empty else 0,
        diag_counts.max() if not diag_counts.empty else 0,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Therapeutic drugs", "Diagnostic drugs"],
        horizontal_spacing=0.12,
    )

    def _iso_hover(df_sub, iso_val):
        mask = df_sub["Radioisotope_Norm"].str.contains(
            iso_val.replace("-", r"\-"), na=False
        )
        return build_asset_hover_text(df_sub[mask])

    if not thera_counts.empty:
        fig.add_trace(
            go.Bar(
                x=thera_counts.values[::-1],
                y=thera_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Therapeutic"],
                customdata=[_iso_hover(thera, iso) for iso in thera_counts.index[::-1]],
                hovertemplate="<b>%{y}</b><br>Assets: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    if not diag_counts.empty:
        fig.add_trace(
            go.Bar(
                x=diag_counts.values[::-1],
                y=diag_counts.index[::-1],
                orientation="h",
                marker_color=CAT_COLORS["Diagnostic"],
                customdata=[_iso_hover(diag, iso) for iso in diag_counts.index[::-1]],
                hovertemplate="<b>%{y}</b><br>Assets: <b>%{x}</b><br><br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    n_rows = max(len(thera_counts), len(diag_counts), 5)
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(350, n_rows * 28 + 120),
        margin=dict(t=60, b=40, l=100, r=40),
    )
    fig.update_xaxes(gridcolor="#EEEEEE", title_text="Unique drugs")
    if max_x > 0:
        fig.update_layout(xaxis_range=[0, max_x * 1.1], xaxis2_range=[0, max_x * 1.1])

    return fig


# ── Chart 11: D2 assets — Isotope evolution (assets, stacked bar) ──────

def build_isotope_evolution_assets(
    assets: pd.DataFrame, top_n: int = 8, mode: str = "stack",
    pin_to_bottom: str | None = None,
) -> go.Figure:
    """Isotope evolution: isotopes of newly registered assets by year of first trial.

    Args:
        assets: Asset-level DataFrame.
        top_n: Number of isotopes to include.
        mode: 'stack' = stacked bar; 'line' = line chart with markers.
        pin_to_bottom: Isotope to place at the bottom of the stack (first trace).
                       Only applies to mode='stack'. None = frequency order.
    """
    df_a = assets.dropna(subset=["First_Year", "Radioisotope_Norm"]).copy()
    df_a["First_Year"] = df_a["First_Year"].astype(int)

    year_min = int(df_a["First_Year"].min())
    year_max = int(df_a["First_Year"].max())
    years = list(range(year_min, year_max + 1))

    # Explode compound isotope entries
    df_exp = df_a.assign(
        Isotope=df_a["Radioisotope_Norm"].str.split("; ")
    ).explode("Isotope").dropna(subset=["Isotope"]).reset_index(drop=True)
    df_exp["Isotope"] = df_exp["Isotope"].str.strip()

    top_isos = df_exp["Isotope"].value_counts().head(top_n).index.tolist()
    # Stable color map keyed by name so pinning doesn't shift colors
    color_map = {iso: QUAL_COLORS[i % len(QUAL_COLORS)] for i, iso in enumerate(top_isos)}
    if mode == "stack" and pin_to_bottom and pin_to_bottom in top_isos:
        top_isos = [pin_to_bottom] + [iso for iso in top_isos if iso != pin_to_bottom]
    df_filtered = df_exp[df_exp["Isotope"].isin(top_isos)]

    ct = pd.crosstab(df_filtered["First_Year"], df_filtered["Isotope"])
    ct = ct.reindex(years, fill_value=0)

    fig = go.Figure()

    for iso in top_isos:
        color = color_map[iso]
        if iso not in ct.columns:
            continue

        hover_texts = []
        for yr in years:
            subset = df_a[
                (df_a["First_Year"] == yr)
                & df_a["Radioisotope_Norm"].str.contains(iso, na=False)
            ]
            hover_texts.append(build_asset_hover_text(subset))

        if mode == "line":
            fig.add_trace(go.Scatter(
                name=iso,
                x=years,
                y=ct[iso].tolist(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                customdata=hover_texts,
                hovertemplate=(
                    "<b>%{x} · " + iso + "</b><br>"
                    "New drugs: <b>%{y}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Bar(
                name=iso,
                x=years,
                y=ct[iso].tolist(),
                marker_color=color,
                customdata=hover_texts,
                hovertemplate=(
                    "<b>%{x} · " + iso + "</b><br>"
                    "New drugs: <b>%{y}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                ),
            ))

    layout_args = dict(
        xaxis_title="Year of first trial",
        yaxis_title="Number of new drugs",
        yaxis=dict(gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
    )
    if mode == "stack":
        layout_args["barmode"] = "stack"
    fig.update_layout(**layout_args)

    return fig


# ── Chart 12: Format evolution (assets) — count or cumulative ──────────

def build_format_evolution_assets(
    assets: pd.DataFrame, mode: str = "new", pin_to_bottom: str | None = None,
) -> go.Figure:
    """Format family evolution over time (asset-level).

    Args:
        assets: Asset-level DataFrame.
        mode: 'new' = new assets per year (stacked); 'pct' = 100% stacked (share per year);
              'cumulative' = rolling total (line chart).
        pin_to_bottom: Format family to place at the bottom of the stack (first trace).
                       Only applies to mode='new' and 'pct'. None = alphabetical order.
    """
    df_a = assets.dropna(subset=["First_Year", "Format_Family"]).copy()
    df_a["First_Year"] = df_a["First_Year"].astype(int)

    ct = pd.crosstab(df_a["First_Year"], df_a["Format_Family"])
    years = list(range(int(ct.index.min()), int(ct.index.max()) + 1))
    ct = ct.reindex(years, fill_value=0)
    formats = ct.columns.tolist()
    # Stable color map keyed by name so pinning doesn't shift colors
    color_map = {fmt: QUAL_COLORS[i] for i, fmt in enumerate(formats)}
    if mode in ("new", "pct") and pin_to_bottom and pin_to_bottom in formats:
        formats = [pin_to_bottom] + [f for f in formats if f != pin_to_bottom]

    totals = ct.sum(axis=1)
    pct_ct = ct.div(totals, axis=0) * 100

    if mode == "cumulative":
        ct = ct.cumsum()

    fig = go.Figure()

    # Pre-compute raw (non-cumulative) crosstab for hover (always shows new assets for that year)
    ct_raw = pd.crosstab(df_a["First_Year"], df_a["Format_Family"]).reindex(years, fill_value=0)

    if mode in ("new", "pct"):
        for fmt in formats:
            color = color_map[fmt]
            hover_texts = []
            for yr in years:
                subset = df_a[(df_a["First_Year"] == yr) & (df_a["Format_Family"] == fmt)]
                hover_texts.append(build_asset_hover_text(subset))
            if mode == "pct":
                y_vals = pct_ct[fmt].round(1).tolist()
                customdata = [
                    f"Count: {ct.loc[yr, fmt]}<br>Total: {totals[yr]}<br><br>{hover_texts[i]}"
                    if yr in ct.index else hover_texts[i]
                    for i, yr in enumerate(years)
                ]
                hover = (
                    "<b>%{x} · " + fmt + "</b><br>"
                    "Share: <b>%{y:.1f}%</b><br>"
                    "%{customdata}<extra></extra>"
                )
            else:
                y_vals = ct[fmt].tolist()
                customdata = hover_texts
                hover = (
                    "<b>%{x} · " + fmt + "</b><br>"
                    "New drugs: <b>%{y}</b><br>"
                    "<br>%{customdata}<extra></extra>"
                )
            fig.add_trace(go.Bar(
                name=fmt,
                x=years,
                y=y_vals,
                marker_color=color,
                customdata=customdata,
                hovertemplate=hover,
            ))
        fig.update_layout(barmode="stack")
        ylab = "Share of new drugs (%)" if mode == "pct" else "Number of new drugs"
    else:
        for fmt in formats:
            color = color_map[fmt]
            hover_texts = []
            for yr in years:
                subset = df_a[(df_a["First_Year"] == yr) & (df_a["Format_Family"] == fmt)]
                hover_texts.append(build_asset_hover_text(subset))
            fig.add_trace(
                go.Scatter(
                    name=fmt,
                    x=years,
                    y=ct[fmt].tolist(),
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    customdata=hover_texts,
                    hovertemplate=(
                        "<b>%{x} · " + fmt + "</b><br>"
                        "Total drugs: <b>%{y}</b><br>"
                        "New this year: <b>" + fmt + "</b><br>"
                        "<br>%{customdata}<extra></extra>"
                    ),
                )
            )
        ylab = "Total unique assets"

    fig.update_layout(
        xaxis_title="Year of first trial",
        yaxis_title=ylab,
        yaxis=dict(gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
    )

    return fig


# ── Explorer Tab (Tab 3) chart builders ────────────────────────────────

def build_explorer_bar(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    color_col: str | None = None,
    color_label: str | None = None,
    color_map: dict | None = None,
    x_order: list[str] | None = None,
    color_order: list[str] | None = None,
) -> go.Figure:
    """Horizontal stacked bar for a categorical primary dimension.

    df must already be preprocessed (top-N filtered, exploded if needed).
    x_order: canonical sort order for low-card x (first item ends up at top of chart).
    """
    df = df.dropna(subset=[x_col]).copy()
    if df.empty:
        return go.Figure()

    has_color = bool(color_col and color_col in df.columns)

    if has_color:
        df = df.dropna(subset=[color_col])
        ct = pd.crosstab(df[x_col], df[color_col])

        if x_order:
            # Use specified order; last item in x_order ends up at top of horizontal chart
            rows = [r for r in reversed(x_order) if r in ct.index]
            rows += ct.sum(axis=1).sort_values(ascending=True).loc[
                [r for r in ct.index if r not in x_order]
            ].index.tolist()
        else:
            rows = ct.sum(axis=1).sort_values(ascending=True).index.tolist()
        ct = ct.loc[[r for r in rows if r in ct.index]]

        if color_order:
            cols_ord = [c for c in color_order if c in ct.columns]
            cols_ord += [c for c in ct.columns if c not in color_order]
        else:
            cols_ord = ct.sum().sort_values(ascending=False).index.tolist()

        fig = go.Figure()
        for val in cols_ord:
            if val not in ct.columns:
                continue
            clr = color_map.get(val, "#CCCCCC") if color_map else "#4C72B0"
            fig.add_trace(go.Bar(
                name=val,
                x=ct[val].tolist(),
                y=ct.index.tolist(),
                orientation="h",
                marker_color=clr,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{color_label}: <b>{val}</b><br>"
                    "Count: <b>%{x}</b><extra></extra>"
                ),
            ))
        fig.update_layout(
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        vc = df[x_col].value_counts()
        if x_order:
            ordered = [c for c in reversed(x_order) if c in vc.index]
            rest = vc.loc[[c for c in vc.sort_values(ascending=True).index
                           if c not in x_order]].index.tolist()
            y_vals = rest + ordered
            vc = vc.loc[y_vals]
        else:
            vc = vc.sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=vc.values.tolist(),
            y=vc.index.tolist(),
            orientation="h",
            marker_color=QUAL_COLORS[0],
            hovertemplate="<b>%{y}</b><br>Count: <b>%{x}</b><extra></extra>",
        ))

    bar_height = max(350, len(df[x_col].unique()) * 24 + 120)
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title=x_label,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=bar_height,
        margin=dict(t=80, b=40, l=200, r=40),
    )
    return fig


def build_explorer_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    x_order: list[str] | None = None,
    y_order: list[str] | None = None,
) -> go.Figure:
    """Count heatmap for two low-cardinality categorical dimensions."""
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return go.Figure()

    ct = pd.crosstab(df[y_col], df[x_col])

    if x_order:
        x_cols = [c for c in x_order if c in ct.columns]
        x_cols += [c for c in ct.columns if c not in x_order]
        ct = ct[x_cols]
    if y_order:
        y_rows = [r for r in y_order if r in ct.index]
        y_rows += [r for r in ct.index if r not in y_order]
        ct = ct.loc[y_rows]

    # Reverse rows so first item in y_order appears at top of chart
    ct = ct.iloc[::-1]

    fig = go.Figure(go.Heatmap(
        z=ct.values.tolist(),
        x=ct.columns.tolist(),
        y=ct.index.tolist(),
        colorscale="Blues",
        text=ct.values.tolist(),
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>Count: <b>%{z}</b><extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(350, len(ct.index) * 45 + 150),
        margin=dict(t=60, b=100, l=220, r=40),
    )
    return fig


def build_explorer_time_bar(
    df: pd.DataFrame,
    time_col: str,
    color_col: str | None,
    color_label: str | None,
    color_map: dict | None,
    color_order: list[str] | None = None,
    time_mode: str = "individual",
    bin_size: int = 5,
) -> go.Figure:
    """Stacked bar over time (individual years or period bins)."""
    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = df[time_col].astype(int)
    if df.empty:
        return go.Figure()

    if time_mode == "bins":
        df["_x"] = year_bins(df, col=time_col, bin_size=bin_size).astype(str)
        df = df[df["_x"] != "nan"]
        x_title = f"{bin_size}-year period"
    else:
        df["_x"] = df[time_col]
        x_title = "Year"

    has_color = bool(color_col and color_col in df.columns)

    if has_color:
        df = df.dropna(subset=[color_col])
        ct = pd.crosstab(df["_x"], df[color_col])

        if time_mode == "individual":
            try:
                all_x = range(int(ct.index.min()), int(ct.index.max()) + 1)
                ct = ct.reindex(all_x, fill_value=0)
            except (ValueError, TypeError):
                pass

        if color_order:
            cols_ord = [c for c in color_order if c in ct.columns]
            cols_ord += [c for c in ct.columns if c not in color_order]
        else:
            cols_ord = ct.sum().sort_values(ascending=False).index.tolist()

        fig = go.Figure()
        for val in cols_ord:
            if val not in ct.columns:
                continue
            clr = color_map.get(val, "#CCCCCC") if color_map else "#4C72B0"
            fig.add_trace(go.Bar(
                name=val,
                x=ct.index.tolist(),
                y=ct[val].tolist(),
                marker_color=clr,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{color_label}: <b>{val}</b><br>"
                    "Count: <b>%{y}</b><extra></extra>"
                ),
            ))
        fig.update_layout(
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        vc = df.groupby("_x").size()
        if time_mode == "individual":
            try:
                all_x = range(int(vc.index.min()), int(vc.index.max()) + 1)
                vc = vc.reindex(all_x, fill_value=0)
            except (ValueError, TypeError):
                pass
        fig = go.Figure(go.Bar(
            x=vc.index.tolist(),
            y=vc.values.tolist(),
            marker_color=QUAL_COLORS[0],
            hovertemplate="<b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
        ))

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Count",
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
        yaxis=dict(gridcolor="#EEEEEE"),
    )
    return fig


def build_explorer_box(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_label: str,
    value_label: str,
    color_map: dict | None = None,
    cat_order: list[str] | None = None,
    log_scale: bool = True,
) -> go.Figure:
    """Box plot: distribution of a continuous dimension grouped by a categorical one."""
    df = df.dropna(subset=[group_col, value_col]).copy()
    if df.empty:
        return go.Figure()

    if cat_order:
        groups = [g for g in cat_order if g in df[group_col].unique()]
        groups += [g for g in df[group_col].unique() if g not in cat_order]
    else:
        groups = df[group_col].value_counts().index.tolist()

    fig = go.Figure()
    for group in groups:
        vals = df[df[group_col] == group][value_col].dropna()
        if vals.empty:
            continue
        clr = color_map.get(group, "#4C72B0") if color_map else "#4C72B0"
        fig.add_trace(go.Box(
            name=group,
            y=vals.tolist(),
            marker_color=clr,
            boxpoints="outliers",
            hovertemplate=f"<b>{group}</b><br>%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title=group_label,
        yaxis_title=value_label + (" (log scale)" if log_scale else ""),
        yaxis=dict(type="log" if log_scale else "linear", gridcolor="#EEEEEE"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(t=60, b=80, l=70, r=40),
    )
    return fig


def build_explorer_histogram(
    df: pd.DataFrame,
    col: str,
    label: str,
    log_scale: bool = True,
) -> go.Figure:
    """Histogram for a single continuous dimension (no categorical grouping)."""
    vals = df[col].dropna()
    if vals.empty:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=vals.tolist(),
        nbinsx=30,
        marker_color=QUAL_COLORS[0],
        hovertemplate="Value: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=label,
        yaxis_title="Count",
        yaxis=dict(type="log" if log_scale else "linear", gridcolor="#EEEEEE"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(t=60, b=40, l=60, r=40),
    )
    return fig


# ── Charts 13+14: Drug landscape bubble charts ─────────────────────────

def _safe_val(raw, default: str = "?") -> str:
    """Return string value or default when the value is NaN/None/empty."""
    if raw is None:
        return default
    s = str(raw).strip()
    return s if s and s.lower() != "nan" else default


def _build_bubble_hover(row: "pd.Series", drug_names: dict | None = None) -> str:
    """Single-row hover string for drug bubble charts."""
    name = _safe_val(row.get("Antibody_Name_Norm"), "Unknown mAb")
    iso  = _safe_val(row.get("Radioisotope_Norm"))
    target = _safe_val(row.get("Target_Antigen_Norm"))
    fmt  = _safe_val(row.get("Antibody_Format_Norm"))
    phase = _safe_val(row.get("Highest_Phase_Norm"))
    first_yr = int(row["First_Year"]) if pd.notna(row.get("First_Year")) else "?"
    latest_yr = int(row["Latest_Year"]) if pd.notna(row.get("Latest_Year")) else "?"
    n_trials = int(row.get("N_Trials", 1)) if pd.notna(row.get("N_Trials")) else 1
    owner = _safe_val(row.get("Asset_Owner_Norm"), "")
    note = _safe_val(row.get("Dashboard_Note"), "")
    asset_names = _safe_val(row.get("Asset_Names"), "")

    # Official name lookup: show brand + INN when available; program name by isotope
    drug_entry = (drug_names or {}).get(name, {})
    brand = drug_entry.get("brand", "")
    inn = drug_entry.get("inn", "")
    if brand:
        display_name = f"{brand} ({inn})" if inn else brand
        title = f"{display_name} · {iso}" if iso != "?" else display_name
    else:
        title = f"{name} ({iso})" if iso != "?" else name

    # Program name: use YAML lookup first, fall back to Asset_Names
    program_map = drug_entry.get("program", {})
    iso_raw = str(row.get("Radioisotope_Norm") or "").split(";")[0].strip()
    program = program_map.get(iso_raw, "") if isinstance(program_map, dict) else str(program_map or "")

    lines = [
        f"<b>{title}</b>",
    ]
    if program:
        lines.append(f"Program: {program}")
    elif asset_names and asset_names != "?":
        first_name = str(asset_names).split(";")[0].strip()
        lines.append(f"Also known as: {first_name}")
    lines += [
        f"Target: {target}",
        f"Format: {fmt}",
        f"Phase: {phase}",
        f"First trial: {first_yr}  |  Latest: {latest_yr}",
        f"N trials: {n_trials}",
    ]
    if owner:
        lines.append(f"Sponsor: {owner}")
    if note:
        lines.append(f"<br><i>⚠ {note}</i>")
    return "<br>".join(lines)


def build_drug_landscape_scatter(
    df: pd.DataFrame,
    color_by: str = "Isotope_Family",
    top_n: int = 25,
) -> go.Figure:
    """Bubble scatter — one dot per drug: X=First_Year, Y=Target_Antigen (top N),
    Color=Isotope_Family or Highest_Phase_Norm, Symbol=Format_Family, Size=N_Trials.

    Args:
        df: Asset-level DataFrame (pre-filtered).
        color_by: Column to use for color encoding — 'Isotope_Family' or 'Highest_Phase_Norm'.
        top_n: Number of target antigens to show (by drug count).
    """
    needed = ["First_Year", "Target_Antigen_Norm", color_by]
    df_plot = df.dropna(subset=needed).copy()
    df_plot["First_Year"] = df_plot["First_Year"].astype(int)
    if df_plot.empty:
        return go.Figure()

    # Top-N targets, sorted by drug count (most frequent first)
    target_counts = df_plot["Target_Antigen_Norm"].value_counts()
    top_targets = target_counts.head(top_n).index.tolist()
    df_plot = df_plot[df_plot["Target_Antigen_Norm"].isin(top_targets)]

    # Y axis: most frequent at top → highest numeric index
    n_targets = len(top_targets)
    target_to_idx = {t: (n_targets - 1 - i) for i, t in enumerate(top_targets)}
    df_plot["_y_idx"] = df_plot["Target_Antigen_Norm"].map(target_to_idx)
    rng = np.random.default_rng(42)
    df_plot["_y_jit"] = df_plot["_y_idx"] + rng.uniform(-0.32, 0.32, size=len(df_plot))

    # Color groups and map
    if color_by == "Isotope_Family":
        color_map = _ISOTOPE_FAMILY_COLORS
        color_groups = [g for g in _ISOTOPE_FAMILY_ORDER if g in df_plot[color_by].values]
        color_label = "Isotope Family"
    else:  # Highest_Phase_Norm
        color_map = PHASE_COLORS
        color_groups = [p for p in PHASE_ORDER if p in df_plot[color_by].values]
        color_label = "Highest Phase"

    # Size scaling: area mode, max bubble ≈ 35px diameter
    max_n = max(df_plot["N_Trials"].fillna(1).max(), 1)
    sizeref = 2.0 * max_n / (35.0 ** 2)

    drug_names = _load_drug_names()
    fig = go.Figure()

    # ── Color traces ───────────────────────────────────────────────────
    for i, group in enumerate(color_groups):
        mask = df_plot[color_by] == group
        sub = df_plot[mask]
        if sub.empty:
            continue
        color = color_map.get(group, "#CCCCCC")
        symbols = sub["Format_Family"].map(
            lambda f: _FORMAT_SYMBOLS.get(str(f), "circle")
        ).tolist()
        hover = [_build_bubble_hover(row, drug_names) for _, row in sub.iterrows()]

        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            showlegend=False,
            x=sub["First_Year"].tolist(),
            y=sub["_y_jit"].tolist(),
            mode="markers",
            marker=dict(
                color=color,
                symbol=symbols,
                size=sub["N_Trials"].fillna(1).tolist(),
                sizemode="area",
                sizeref=sizeref,
                sizemin=5,
                line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
            ),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            legendgrouptitle=dict(text=color_label) if i == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, symbol="circle", size=16),
            hoverinfo="skip",
        ))

    # ── Dummy symbol legend traces ─────────────────────────────────────
    fmt_present = [f for f in _FORMAT_SYMBOLS if f in df_plot["Format_Family"].values]
    for j, fmt in enumerate(fmt_present):
        fig.add_trace(go.Scatter(
            name=fmt,
            legendgroup="format",
            legendgrouptitle=dict(text="Format Family") if j == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color="#555555", symbol=_FORMAT_SYMBOLS[fmt], size=16),
            hoverinfo="skip",
        ))

    # ── Size legend traces ─────────────────────────────────────────────
    _add_size_legend(fig, sizeref, max_n)

    # Y axis: numeric ticks mapped to target names
    tickvals = list(range(n_targets))
    ticktext = list(reversed(top_targets))  # index 0 = bottom = least frequent

    fig.update_layout(
        xaxis_title="Year of first trial",
        yaxis=dict(
            title="Target antigen",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            gridcolor="#EEEEEE",
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(520, n_targets * 30 + 140),
        margin=dict(t=80, b=40, l=200, r=180),
        legend=dict(
            orientation="v",
            xanchor="left", x=1.02,
            yanchor="top", y=1,
            tracegroupgap=12,
        ),
    )
    return fig


def build_drug_phase_timeline(df: pd.DataFrame) -> go.Figure:
    """Bubble scatter — one dot per drug: X=First_Year, Y=Highest_Phase,
    Color=Isotope_Family, Symbol=Format_Family, Size=N_Trials.

    No color toggle — fixed 5-dimension layout (no toggle needed).
    """
    needed = ["First_Year", "Highest_Phase_Norm", "Isotope_Family"]
    df_plot = df.dropna(subset=needed).copy()
    df_plot["First_Year"] = df_plot["First_Year"].astype(int)
    if df_plot.empty:
        return go.Figure()

    # Y axis: phases in clinical order (lowest at bottom)
    phases_present = [p for p in PHASE_ORDER if p in df_plot["Highest_Phase_Norm"].values]
    phase_to_idx = {p: i for i, p in enumerate(phases_present)}
    df_plot["_y_idx"] = df_plot["Highest_Phase_Norm"].map(phase_to_idx)
    rng = np.random.default_rng(42)
    df_plot["_y_jit"] = df_plot["_y_idx"] + rng.uniform(-0.35, 0.35, size=len(df_plot))

    color_groups = [g for g in _ISOTOPE_FAMILY_ORDER if g in df_plot["Isotope_Family"].values]
    max_n = max(df_plot["N_Trials"].fillna(1).max(), 1)
    sizeref = 2.0 * max_n / (35.0 ** 2)

    drug_names = _load_drug_names()
    fig = go.Figure()

    # ── Color traces ───────────────────────────────────────────────────
    for i, group in enumerate(color_groups):
        mask = df_plot["Isotope_Family"] == group
        sub = df_plot[mask]
        if sub.empty:
            continue
        color = _ISOTOPE_FAMILY_COLORS.get(group, "#CCCCCC")
        symbols = sub["Format_Family"].map(
            lambda f: _FORMAT_SYMBOLS.get(str(f), "circle")
        ).tolist()
        hover = [_build_bubble_hover(row, drug_names) for _, row in sub.iterrows()]

        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            showlegend=False,
            x=sub["First_Year"].tolist(),
            y=sub["_y_jit"].tolist(),
            mode="markers",
            marker=dict(
                color=color,
                symbol=symbols,
                size=sub["N_Trials"].fillna(1).tolist(),
                sizemode="area",
                sizeref=sizeref,
                sizemin=5,
                line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
            ),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            legendgrouptitle=dict(text="Isotope Family") if i == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, symbol="circle", size=16),
            hoverinfo="skip",
        ))

    # ── Dummy symbol legend traces ─────────────────────────────────────
    fmt_present = [f for f in _FORMAT_SYMBOLS if f in df_plot["Format_Family"].values]
    for j, fmt in enumerate(fmt_present):
        fig.add_trace(go.Scatter(
            name=fmt,
            legendgroup="format",
            legendgrouptitle=dict(text="Format Family") if j == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color="#555555", symbol=_FORMAT_SYMBOLS[fmt], size=16),
            hoverinfo="skip",
        ))

    # ── Size legend traces ─────────────────────────────────────────────
    _add_size_legend(fig, sizeref, max_n)

    fig.update_layout(
        xaxis_title="Year of first trial",
        yaxis=dict(
            title="Highest clinical phase",
            tickmode="array",
            tickvals=list(range(len(phases_present))),
            ticktext=phases_present,
            gridcolor="#EEEEEE",
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=80, b=40, l=130, r=180),
        legend=dict(
            orientation="v",
            xanchor="left", x=1.02,
            yanchor="top", y=1,
            tracegroupgap=12,
        ),
    )
    return fig


def build_company_portfolio_scatter(
    df_assets: "pd.DataFrame",
    df_trials: "pd.DataFrame",
    top_n: int = 15,
) -> "go.Figure":
    """Chart 15: Company asset portfolio bubble chart.

    Y = company (top N, sorted by drug count desc then alpha).
    X = Highest_Phase_Norm (Early Phase 1 → Phase 4).
    Color = Isotope_Family. Symbol = Format_Family. Size = N_Trials.

    Company attribution uses Asset_Owner_Norm via asset_owner_types.yaml.
    Assets with type='company' or 'mixed' are shown; academic/unknown are excluded.
    """
    # ── Step 1: Look up each asset's company name via YAML ──────────────
    owner_types = _load_asset_owner_types()

    def _company_for(owner_norm) -> str | None:
        if pd.isna(owner_norm) or str(owner_norm).strip() == "":
            return None
        entry = owner_types.get(str(owner_norm).strip())
        if entry is None or entry.get("type") not in ("company", "mixed"):
            return None
        company = entry.get("company")
        return str(company).strip() if company and str(company).strip() else None

    df_plot = df_assets.copy()
    df_plot["_company"] = df_plot["Asset_Owner_Norm"].map(_company_for)
    df_plot = df_plot[df_plot["_company"].notna()]
    df_plot = df_plot.dropna(subset=["Highest_Phase_Norm", "Isotope_Family"])

    if df_plot.empty:
        return go.Figure().update_layout(
            title="No company assets match the current filters."
        )

    # ── Step 2: Top N companies (most drugs, then alphabetical) ─────────
    company_counts = df_plot["_company"].value_counts()
    company_df = pd.DataFrame({
        "company": company_counts.index,
        "n_drugs": company_counts.values,
    })
    company_df = company_df.sort_values(["n_drugs", "company"], ascending=[False, True])
    top_companies = company_df["company"].head(top_n).tolist()
    df_plot = df_plot[df_plot["_company"].isin(top_companies)].copy()
    n_top = len(top_companies)

    # ── Step 3: Axis mappings ────────────────────────────────────────────
    # Y: most drugs at top → highest index
    company_to_idx = {c: (n_top - 1 - i) for i, c in enumerate(top_companies)}
    # X: always show all phases so the axis is stable across filters
    phases_present = PHASE_ORDER
    phase_to_idx = {p: i for i, p in enumerate(phases_present)}

    # ── Step 4: Jitter ───────────────────────────────────────────────────
    # Only jitter within cells where multiple assets share the same (company, phase).
    # Lone assets sit precisely on the grid intersection.
    cell_sizes = df_plot.groupby(["_company", "Highest_Phase_Norm"])["_company"].transform("count")
    needs_jitter = (cell_sizes > 1).values
    rng = np.random.default_rng(42)
    jitter_y = rng.uniform(-0.30, 0.30, size=len(df_plot))
    jitter_x = rng.uniform(-0.30, 0.30, size=len(df_plot))
    df_plot["_y_jit"] = (
        df_plot["_company"].map(company_to_idx)
        + np.where(needs_jitter, jitter_y, 0.0)
    )
    df_plot["_x_jit"] = (
        df_plot["Highest_Phase_Norm"].map(phase_to_idx)
        + np.where(needs_jitter, jitter_x, 0.0)
    )

    # ── Step 5: Size scaling ─────────────────────────────────────────────
    max_n = max(df_plot["N_Trials"].fillna(1).max(), 1)
    sizeref = 2.0 * max_n / (35.0 ** 2)

    # ── Step 6: Build traces ─────────────────────────────────────────────
    color_groups = [g for g in _ISOTOPE_FAMILY_ORDER if g in df_plot["Isotope_Family"].values]
    drug_names = _load_drug_names()
    fig = go.Figure()

    for i, group in enumerate(color_groups):
        mask = df_plot["Isotope_Family"] == group
        sub = df_plot[mask]
        if sub.empty:
            continue
        color = _ISOTOPE_FAMILY_COLORS.get(group, "#CCCCCC")
        symbols = sub["Format_Family"].map(
            lambda f: _FORMAT_SYMBOLS.get(str(f), "circle")
        ).tolist()
        hover = [_build_bubble_hover(row, drug_names) for _, row in sub.iterrows()]

        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            showlegend=False,
            x=sub["_x_jit"].tolist(),
            y=sub["_y_jit"].tolist(),
            mode="markers",
            marker=dict(
                color=color,
                symbol=symbols,
                size=sub["N_Trials"].fillna(1).tolist(),
                sizemode="area",
                sizeref=sizeref,
                sizemin=5,
                line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
            ),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            name=group,
            legendgroup=f"iso_{group}",
            legendgrouptitle=dict(text="Isotope Family") if i == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, symbol="circle", size=16),
            hoverinfo="skip",
        ))

    # ── Dummy symbol legend traces ───────────────────────────────────────
    fmt_present = [f for f in _FORMAT_SYMBOLS if f in df_plot["Format_Family"].values]
    for j, fmt in enumerate(fmt_present):
        fig.add_trace(go.Scatter(
            name=fmt,
            legendgroup="format",
            legendgrouptitle=dict(text="Format Family") if j == 0 else {},
            x=[None], y=[None],
            mode="markers",
            marker=dict(color="#555555", symbol=_FORMAT_SYMBOLS[fmt], size=16),
            hoverinfo="skip",
        ))

    # ── Size legend traces ───────────────────────────────────────────────
    _add_size_legend(fig, sizeref, max_n)

    # ── Layout ───────────────────────────────────────────────────────────
    n_phases = len(phases_present)
    height = max(500, n_top * 34 + 180)
    fig.update_layout(
        xaxis=dict(
            title="Highest clinical phase",
            tickmode="array",
            tickvals=list(range(n_phases)),
            ticktext=phases_present,
            range=[-0.6, n_phases - 0.4],
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(
            title=None,
            tickmode="array",
            tickvals=list(range(n_top)),
            ticktext=list(reversed(top_companies)),
            range=[-0.6, n_top - 0.4],
            gridcolor="#EEEEEE",
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=60, b=60, l=230, r=180),
        legend=dict(
            orientation="v",
            xanchor="left", x=1.02,
            yanchor="top", y=1,
            tracegroupgap=12,
        ),
    )
    return fig


# ── Chart: Sponsor type trend over time ────────────────────────────────

_SPONSOR_GROUP_COLORS: dict[str, str] = {
    "Industry":          SPONSOR_COLORS["Industry"],
    "Academic/Hospital": SPONSOR_COLORS["Academic/Hospital"],
    "NIH / Gov / Other": "#7570B3",
}
_SPONSOR_GROUP_ORDER = ["Industry", "Academic/Hospital", "NIH / Gov / Other"]


def build_industry_sponsorship_trend(
    df: pd.DataFrame,
    category_view: str = "All",
    mode: str = "count",
    rolling_window: int = 5,
) -> go.Figure:
    """Stacked bar (trial counts by sponsor type × year) + rolling % industry line.

    Args:
        df: Pre-filtered relevant trials.
        category_view: "All" / "Therapeutic" / "Diagnostic".
            Theranostic Pair trials are counted in both Therapeutic and Diagnostic views.
        mode: "count" = absolute trial counts; "pct" = 100% stacked (share per year).
        rolling_window: Window size (years) for the rolling % industry trend line.
    """
    if category_view == "Therapeutic":
        df_sub = df[df["Category"].isin(["Therapeutic", "Theranostic Pair"])].copy()
    elif category_view == "Diagnostic":
        df_sub = df[df["Category"].isin(["Diagnostic", "Theranostic Pair"])].copy()
    else:
        df_sub = df.copy()

    df_sub = df_sub.dropna(subset=["Start_Year"]).copy()
    if df_sub.empty:
        return go.Figure()
    df_sub["Start_Year"] = df_sub["Start_Year"].astype(int)
    y_min, y_max = _year_range_from_df(df_sub)
    df_sub = df_sub[(df_sub["Start_Year"] >= y_min) & (df_sub["Start_Year"] <= y_max)]
    years = list(range(y_min, y_max + 1))

    def _sponsor_group(st: str) -> str:
        if st == "Industry":
            return "Industry"
        if st == "Academic/Hospital":
            return "Academic/Hospital"
        return "NIH / Gov / Other"

    df_sub["_Group"] = df_sub["Sponsor_Type"].fillna("NIH / Gov / Other").map(_sponsor_group)

    ct = pd.crosstab(df_sub["Start_Year"], df_sub["_Group"])
    ct = ct.reindex(years, fill_value=0)
    for g in _SPONSOR_GROUP_ORDER:
        if g not in ct.columns:
            ct[g] = 0

    totals = ct.sum(axis=1)
    pct_industry = (ct["Industry"] / totals.replace(0, np.nan) * 100).fillna(0)
    pct_rolling = pct_industry.rolling(rolling_window, center=True, min_periods=2).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if mode == "pct":
        pct_table = ct.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
        for g in _SPONSOR_GROUP_ORDER:
            vals = pct_table[g].round(1).tolist()
            customdata = [
                f"Count: {ct.loc[yr, g]}<br>Total: {totals[yr]}" for yr in years
            ]
            fig.add_trace(
                go.Bar(
                    name=g,
                    x=years,
                    y=vals,
                    marker_color=_SPONSOR_GROUP_COLORS[g],
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{x} · " + g + "</b><br>"
                        "Share: <b>%{y:.1f}%</b><br>"
                        "%{customdata}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
        yaxis_title = "Share of trials (%)"
        yaxis_cfg = dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE")
    else:
        for g in _SPONSOR_GROUP_ORDER:
            vals = ct[g].tolist()
            fig.add_trace(
                go.Bar(
                    name=g,
                    x=years,
                    y=vals,
                    marker_color=_SPONSOR_GROUP_COLORS[g],
                    hovertemplate=(
                        "<b>%{x} · " + g + "</b><br>"
                        "Trials: <b>%{y}</b><extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
        yaxis_title = "Number of trials"
        yaxis_cfg = dict(gridcolor="#EEEEEE")

    # Rolling % industry trend line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            name=f"% Industry ({rolling_window}y rolling avg)",
            x=years,
            y=pct_rolling.round(1).tolist(),
            mode="lines",
            line=dict(color="black", width=2.5, dash="dash"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"% Industry ({rolling_window}y avg): <b>%{{y:.1f}}%</b><extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        yaxis=yaxis_cfg,
        yaxis2=dict(
            range=[0, 100],
            ticksuffix="%",
            showgrid=False,
            title="% Industry-sponsored",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=60),
    )

    return fig


# ── Chart T8: Sponsor type — Therapeutic vs Diagnostic side by side ────

def build_sponsorship_rit_vs_diag(
    df: pd.DataFrame,
    mode: str = "count",
) -> go.Figure:
    """Side-by-side stacked bars: sponsor type evolution for Therapeutic (left)
    and Diagnostic (right) trials on a shared y-axis for direct comparison.

    Theranostic Pair trials are counted in both panels.

    Args:
        df: Pre-filtered relevant trials.
        mode: "count" = absolute trial counts; "pct" = 100% stacked (share per year).
    """
    panels = [
        ("Therapeutic", df[df["Category"].isin(["Therapeutic", "Theranostic Pair"])].copy()),
        ("Diagnostic",  df[df["Category"].isin(["Diagnostic",  "Theranostic Pair"])].copy()),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=["Therapeutic", "Diagnostic"],
        horizontal_spacing=0.04,
    )

    # Shared year range across both panels
    all_years_data = df.dropna(subset=["Start_Year"])["Start_Year"].astype(int)
    y_min = int(all_years_data.min())
    y_max = min(int(all_years_data.max()), datetime.now().year)
    years = list(range(y_min, y_max + 1))

    show_legend = True
    for col_idx, (panel_title, df_sub) in enumerate(panels, start=1):
        df_sub = df_sub.dropna(subset=["Start_Year"]).copy()
        df_sub["Start_Year"] = df_sub["Start_Year"].astype(int)
        df_sub = df_sub[(df_sub["Start_Year"] >= y_min) & (df_sub["Start_Year"] <= y_max)]

        def _sponsor_group(st: str) -> str:
            if st == "Industry":
                return "Industry"
            if st == "Academic/Hospital":
                return "Academic/Hospital"
            return "NIH / Gov / Other"

        df_sub["_Group"] = df_sub["Sponsor_Type"].fillna("NIH / Gov / Other").map(_sponsor_group)
        ct = pd.crosstab(df_sub["Start_Year"], df_sub["_Group"])
        ct = ct.reindex(years, fill_value=0)
        for g in _SPONSOR_GROUP_ORDER:
            if g not in ct.columns:
                ct[g] = 0
        totals = ct.sum(axis=1)

        if mode == "pct":
            pct_table = ct.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100

        for g in _SPONSOR_GROUP_ORDER:
            if mode == "pct":
                vals = pct_table[g].round(1).tolist()
                customdata = [
                    f"Count: {ct.loc[yr, g]}<br>Total: {totals[yr]}" for yr in years
                ]
                hover = (
                    "<b>%{x} · " + g + "</b><br>"
                    "Share: <b>%{y:.1f}%</b><br>"
                    "%{customdata}<extra></extra>"
                )
            else:
                vals = ct[g].tolist()
                customdata = None
                hover = (
                    "<b>%{x} · " + g + "</b><br>"
                    "Trials: <b>%{y}</b><extra></extra>"
                )

            trace = go.Bar(
                name=g,
                x=years,
                y=vals,
                marker_color=_SPONSOR_GROUP_COLORS[g],
                customdata=customdata,
                hovertemplate=hover,
                legendgroup=g,
                showlegend=show_legend,
            )
            fig.add_trace(trace, row=1, col=col_idx)

        show_legend = False  # only show legend entries once

    if mode == "pct":
        yaxis_title = "Share of trials (%)"
        yaxis_cfg = dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE")
    else:
        yaxis_title = "Number of trials"
        yaxis_cfg = dict(gridcolor="#EEEEEE")

    fig.update_layout(
        barmode="stack",
        yaxis_title=yaxis_title,
        yaxis=yaxis_cfg,
        yaxis2=dict(gridcolor="#EEEEEE") if mode == "count" else dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=40),
    )
    fig.update_xaxes(title_text="Year")

    return fig


# ── Chart T9: Alpha emitter adoption ──────────────────────────────────

_ALPHA_EMITTERS_ORDER = ["Ac-225", "At-211", "Th-227", "Pb-212", "Bi-213", "Ra-223"]
_ALPHA_EMITTER_COLORS = {iso: QUAL_COLORS[i] for i, iso in enumerate(_ALPHA_EMITTERS_ORDER)}


def build_alpha_emitter_adoption(
    df: pd.DataFrame,
    mode: str = "count",
    bin_size: int = 5,
) -> go.Figure:
    """Stacked bar: alpha emitter trial counts per period by isotope + % of all therapeutic line.

    Args:
        df: Pre-filtered trials DataFrame.
        mode: "count" = absolute counts; "pct" = 100% stacked (share per period within alpha emitters).
        bin_size: Width of time bins in years (e.g. 3, 5, 10).
    """
    # Therapeutic + Theranostic Pair trials only (alpha emitters are treatment modalities)
    df_ther = df[df["Category"].isin(["Therapeutic", "Theranostic Pair"])].copy()
    df_ther = df_ther.dropna(subset=["Start_Year"]).copy()
    if df_ther.empty:
        return go.Figure()
    df_ther["Period"] = year_bins(df_ther, bin_size=bin_size).astype(str)
    df_ther = df_ther[df_ther["Period"] != "nan"]
    periods = sorted(df_ther["Period"].unique())

    if not periods:
        return go.Figure()

    # Explode Radioisotope_Norm to one row per isotope, keep only alpha emitters
    df_exp = df_ther.copy()
    df_exp["_Iso"] = df_exp["Radioisotope_Norm"].str.split("; ")
    df_exp = df_exp.explode("_Iso").dropna(subset=["_Iso"]).reset_index(drop=True)
    df_exp["_Iso"] = df_exp["_Iso"].str.strip()
    df_exp = df_exp[df_exp["_Iso"].isin(_ALPHA_EMITTERS_ORDER)]

    # Crosstab: period × isotope (each trial×isotope counted once)
    ct = pd.DataFrame(0, index=periods, columns=_ALPHA_EMITTERS_ORDER)
    if not df_exp.empty:
        raw_ct = pd.crosstab(df_exp["Period"], df_exp["_Iso"])
        for iso in raw_ct.columns:
            if iso in ct.columns:
                ct.loc[raw_ct.index, iso] = raw_ct[iso]

    isos_present = [iso for iso in _ALPHA_EMITTERS_ORDER if ct[iso].sum() > 0]

    # Total therapeutic trials per period (denominator for % line)
    ther_total = df_ther.groupby("Period").size().reindex(periods, fill_value=0)
    # Unique alpha emitter trials per period (a trial with two alpha emitters counts once)
    alpha_unique = (
        df_exp.drop_duplicates(subset=["NCT_ID", "Period"])
        .groupby("Period").size()
        .reindex(periods, fill_value=0)
        if "NCT_ID" in df_exp.columns and not df_exp.empty
        else pd.Series(0, index=periods)
    )
    pct_of_ther = (alpha_unique / ther_total.replace(0, np.nan) * 100).fillna(0)

    totals_alpha = ct[isos_present].sum(axis=1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if mode == "pct":
        pct_table = ct[isos_present].div(totals_alpha.replace(0, np.nan), axis=0).fillna(0) * 100
        for iso in isos_present:
            customdata = [
                f"Count: {int(ct.loc[p, iso])}<br>Alpha total: {int(totals_alpha.get(p, 0))}"
                for p in periods
            ]
            fig.add_trace(go.Bar(
                name=iso,
                x=periods,
                y=pct_table[iso].round(1).tolist(),
                marker_color=_ALPHA_EMITTER_COLORS.get(iso, "#888888"),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x} · " + iso + "</b><br>"
                    "Share: <b>%{y:.1f}%</b><br>"
                    "%{customdata}<extra></extra>"
                ),
            ), secondary_y=False)
        yaxis_title = "Share among alpha emitters (%)"
        yaxis_cfg = dict(range=[0, 100], ticksuffix="%", gridcolor="#EEEEEE")
    else:
        for iso in isos_present:
            fig.add_trace(go.Bar(
                name=iso,
                x=periods,
                y=ct[iso].tolist(),
                marker_color=_ALPHA_EMITTER_COLORS.get(iso, "#888888"),
                hovertemplate=(
                    "<b>%{x} · " + iso + "</b><br>"
                    "Trials: <b>%{y}</b><extra></extra>"
                ),
            ), secondary_y=False)
        yaxis_title = "Number of trials"
        yaxis_cfg = dict(gridcolor="#EEEEEE")

    # % of all therapeutic trials on secondary y (line + markers)
    ther_max_pct = max(float(pct_of_ther.max()) * 1.4, 5.0)
    fig.add_trace(
        go.Scatter(
            name="% of therapeutic trials",
            x=periods,
            y=pct_of_ther.round(1).tolist(),
            mode="lines+markers",
            line=dict(color="black", width=2.5, dash="dash"),
            marker=dict(color="black", size=7),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "% of therapeutic trials: <b>%{y:.1f}%</b><extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title=f"{bin_size}-year period",
        yaxis_title=yaxis_title,
        yaxis=yaxis_cfg,
        yaxis2=dict(
            range=[0, ther_max_pct],
            ticksuffix="%",
            showgrid=False,
            title="% of all therapeutic trials",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=60, r=80),
    )

    return fig


# ── Chart T10: % industry-sponsored per isotope over time ─────────────

def _isotope_industry_pct_table(
    df: pd.DataFrame, bin_size: int = 5, min_trials: int = 5,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Shared computation for T10 line + heatmap.

    Returns:
        piv: DataFrame [period × isotope] of % industry-sponsored (NaN where < min_trials).
        periods: sorted period labels.
        isos: isotopes ordered by overall trial count (most common first).
    """
    df_w = df.dropna(subset=["Start_Year", "Radioisotope_Norm"]).copy()
    df_w["Period"] = year_bins(df_w, bin_size=bin_size).astype(str)
    df_w = df_w[df_w["Period"] != "nan"]

    # Explode compound isotope entries
    df_exp = df_w.assign(_Iso=df_w["Radioisotope_Norm"].str.split("; ")).explode("_Iso")
    df_exp = df_exp.dropna(subset=["_Iso"]).reset_index(drop=True)
    df_exp["_Iso"] = df_exp["_Iso"].str.strip()
    df_exp["_Ind"] = (df_exp["Sponsor_Type"] == "Industry").astype(int)

    # Filter to isotopes with enough total trials
    iso_counts = df_exp["_Iso"].value_counts()
    valid_isos = iso_counts[iso_counts >= min_trials].index.tolist()
    df_exp = df_exp[df_exp["_Iso"].isin(valid_isos)]

    periods = sorted(df_exp["Period"].unique())
    isos = iso_counts[iso_counts >= min_trials].index.tolist()  # frequency order

    # Count and % industry per (period, isotope)
    ct_n = pd.crosstab(df_exp["Period"], df_exp["_Iso"]).reindex(periods, fill_value=0)
    ct_ind = pd.crosstab(df_exp["Period"], df_exp["_Iso"],
                         values=df_exp["_Ind"], aggfunc="sum").reindex(periods, fill_value=0)
    for iso in isos:
        if iso not in ct_n.columns:
            ct_n[iso] = 0
            ct_ind[iso] = 0

    piv = ct_ind[isos].div(ct_n[isos].replace(0, np.nan)) * 100
    # Mask cells with fewer than 3 trials — too noisy to show as a number/colour
    mask = ct_n[isos] < 3
    piv = piv.where(~mask)
    piv = piv.reindex(periods)

    return piv, periods, isos


def build_isotope_industry_pct_line(
    df: pd.DataFrame, bin_size: int = 5, min_trials: int = 5,
) -> go.Figure:
    """Line chart: % industry-sponsored per isotope per period (one line per isotope).

    Args:
        df: Pre-filtered trials DataFrame.
        bin_size: Width of time bins in years.
        min_trials: Minimum total trials for an isotope to be included.
    """
    piv, periods, isos = _isotope_industry_pct_table(df, bin_size, min_trials)

    fig = go.Figure()
    for i, iso in enumerate(isos):
        y = piv[iso].tolist() if iso in piv.columns else [None] * len(periods)
        fig.add_trace(go.Scatter(
            name=iso,
            x=periods,
            y=y,
            mode="lines+markers",
            line=dict(color=QUAL_COLORS[i % len(QUAL_COLORS)], width=2),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate=(
                "<b>%{x} · " + iso + "</b><br>"
                "% Industry: <b>%{y:.0f}%</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title=f"{bin_size}-year period",
        yaxis=dict(range=[0, 105], ticksuffix="%", gridcolor="#EEEEEE", title="% industry-sponsored"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=80, b=40, l=70, r=40),
    )
    return fig


def build_isotope_industry_pct_heatmap(
    df: pd.DataFrame, bin_size: int = 5, min_trials: int = 5,
) -> go.Figure:
    """Heatmap: % industry-sponsored per isotope per period.

    Args:
        df: Pre-filtered trials DataFrame.
        bin_size: Width of time bins in years.
        min_trials: Minimum total trials for an isotope to be included.
    """
    piv, periods, isos = _isotope_industry_pct_table(df, bin_size, min_trials)

    # Heatmap: rows = isotopes (reversed so most common at top), cols = periods
    z = [piv[iso].tolist() if iso in piv.columns else [None] * len(periods) for iso in isos]
    z_text = [
        [f"{v:.0f}%" if v is not None and not np.isnan(v) else "" for v in row]
        for row in z
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=periods,
        y=isos,
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        text=z_text,
        texttemplate="%{text}",
        hovertemplate=(
            "<b>%{y} · %{x}</b><br>"
            "% Industry: <b>%{z:.0f}%</b><extra></extra>"
        ),
        colorbar=dict(title="% Industry", ticksuffix="%"),
    ))

    fig.update_layout(
        xaxis_title=f"{bin_size}-year period",
        yaxis_title="Radioisotope",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(300, len(isos) * 38 + 120),
        margin=dict(t=60, b=60, l=90, r=100),
    )
    return fig


# ── Chart T11: Isotope × indication heatmap ───────────────────────────

def build_isotope_indication_heatmap(
    df: pd.DataFrame,
    top_n_iso: int = 12,
    mode: str = "count",
    category: str = "All",
    iso_family_map: dict | None = None,
) -> go.Figure:
    """Heatmap: trial count (or row %) per radioisotope × indication group (T11).

    Args:
        df: Pre-filtered trials DataFrame (must contain Category and Indication_Group columns).
        top_n_iso: Number of isotopes to show, ranked by total trial count.
        mode: 'count' — raw cell counts; 'pct_iso' — % of each isotope's total.
        category: 'All', 'Therapeutic', or 'Diagnostic'.
        iso_family_map: Optional dict mapping isotope name → Isotope_Family
            (e.g. {'Ac-225': 'Alpha emitter', 'Lu-177': 'Beta emitter'}).
            Used for two purposes:
            1. Splitting Theranostic Pair trials — each trial's compound isotopes
               are filtered to the relevant component (Alpha/Beta → Therapeutic;
               PET/SPECT → Diagnostic), so a trial with "Ac-225; Ga-68" contributes
               Ac-225 to Therapeutic and Ga-68 to Diagnostic.
            2. When category='Therapeutic', rows are grouped into alpha / beta
               sections separated by a dashed line.
    """
    # ── Category filter ────────────────────────────────────────────────
    # Theranostic Pair trials carry compound isotopes (e.g. "Ac-225; Ga-68").
    # When iso_family_map is available, include them in both Therapeutic and
    # Diagnostic views, but filter to the relevant isotope component after
    # exploding (Alpha/Beta → Therapeutic; PET/SPECT → Diagnostic).
    # Without iso_family_map, fall back to strict equality (excludes Theranostic Pair).
    if category == "Therapeutic":
        if iso_family_map:
            df_sub = df[df["Category"].isin(["Therapeutic", "Theranostic Pair"])]
        else:
            df_sub = df[df["Category"] == "Therapeutic"]
    elif category == "Diagnostic":
        if iso_family_map:
            df_sub = df[df["Category"].isin(["Diagnostic", "Theranostic Pair"])]
        else:
            df_sub = df[df["Category"] == "Diagnostic"]
    else:
        df_sub = df

    needed = ["NCT_ID", "Category", "Radioisotope_Norm", "Indication_Group"]
    df_w = df_sub.dropna(subset=["Radioisotope_Norm", "Indication_Group"])[needed].copy()

    def _empty(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
        return fig

    if df_w.empty:
        return _empty("No data for selected filters")

    # ── Explode isotopes then indication groups ─────────────────────────
    df_iso = df_w.assign(_Iso=df_w["Radioisotope_Norm"].str.split("; ")).explode("_Iso").reset_index(drop=True)
    df_iso["_Iso"] = df_iso["_Iso"].str.strip()

    # Post-explode: for Theranostic Pair trials, keep only the isotope component
    # that belongs to the requested category (based on isotope family).
    if iso_family_map and category != "All":
        _ther_fam = {"Alpha emitter", "Beta emitter"}
        _diag_fam = {"PET", "SPECT"}
        _target_fam = _ther_fam if category == "Therapeutic" else _diag_fam
        _keep = (df_iso["Category"] != "Theranostic Pair") | (
            df_iso["_Iso"].map(iso_family_map).isin(_target_fam)
        )
        df_iso = df_iso[_keep]

    df_exp = df_iso.assign(_Grp=df_iso["Indication_Group"].str.split("; ")).explode("_Grp").reset_index(drop=True)
    df_exp["_Grp"] = df_exp["_Grp"].str.strip()
    df_exp = df_exp.dropna(subset=["_Iso", "_Grp"])
    df_exp = df_exp[(df_exp["_Iso"] != "") & (df_exp["_Grp"] != "")]
    # Each (trial, isotope, indication) pair counted once
    df_exp = df_exp.drop_duplicates(subset=["NCT_ID", "_Iso", "_Grp"])

    if df_exp.empty:
        return _empty("No indication data for selected filters")

    # ── Select top N isotopes by trial count ───────────────────────────
    iso_counts = df_exp["_Iso"].value_counts()
    top_isos = iso_counts.head(top_n_iso).index.tolist()
    df_exp = df_exp[df_exp["_Iso"].isin(top_isos)]

    # ── Column order: YAML-defined indication groups (only present ones) ─
    all_indications = get_all_indication_groups()
    present_inds = set(df_exp["_Grp"].unique())
    cols = [g for g in all_indications if g in present_inds]

    # ── Cross-tabulation ───────────────────────────────────────────────
    ct = pd.crosstab(df_exp["_Iso"], df_exp["_Grp"]).reindex(
        index=top_isos, columns=cols, fill_value=0
    )

    # ── Row ordering: group alpha / beta when in Therapeutic mode ─────
    _group_separator_y: float | None = None
    _n_alpha = _n_beta = 0

    if category == "Therapeutic" and iso_family_map:
        alpha_isos = sorted(
            [iso for iso in top_isos if iso_family_map.get(iso) == "Alpha emitter"],
            key=lambda x: -iso_counts.get(x, 0),
        )
        beta_isos = sorted(
            [iso for iso in top_isos if iso_family_map.get(iso) != "Alpha emitter"],
            key=lambda x: -iso_counts.get(x, 0),
        )
        # y_labels bottom→top: least-common alpha … most-common alpha | least-common beta … most-common beta
        y_labels = list(reversed(alpha_isos)) + list(reversed(beta_isos))
        _n_alpha, _n_beta = len(alpha_isos), len(beta_isos)
        if _n_alpha > 0 and _n_beta > 0:
            _group_separator_y = _n_alpha - 0.5
    else:
        # Default: most common at top (Plotly: index 0 = bottom)
        y_labels = list(reversed(top_isos))

    z_count = ct.reindex(y_labels).values.astype(float)

    # ── Build z matrix and annotations ────────────────────────────────
    if mode == "pct_iso":
        row_totals = z_count.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = np.nan
        z_vals = z_count / row_totals * 100
        colorscale = "YlOrRd"
        colorbar = dict(title="% of<br>isotope trials", ticksuffix="%")
        z_text = [
            [f"{v:.0f}%" if not np.isnan(v) and v > 0 else "" for v in row]
            for row in z_vals
        ]
        hovertemplate = (
            "<b>%{y} · %{x}</b><br>"
            "% of isotope trials: <b>%{z:.0f}%</b><br>"
            "Raw count: <b>%{customdata}</b><extra></extra>"
        )
        customdata = [[int(v) for v in row] for row in z_count]
    else:
        z_text = [
            [str(int(v)) if v > 0 else "" for v in row]
            for row in z_count
        ]
        z_vals = z_count.copy()
        z_vals[z_vals == 0] = np.nan
        colorscale = "Blues"
        colorbar = dict(title="Trial count")
        hovertemplate = (
            "<b>%{y} · %{x}</b><br>"
            "Trials: <b>%{z:.0f}</b><extra></extra>"
        )
        customdata = None

    heatmap_kwargs = dict(
        z=z_vals,
        x=cols,
        y=y_labels,
        colorscale=colorscale,
        text=z_text,
        texttemplate="%{text}",
        hovertemplate=hovertemplate,
        colorbar=colorbar,
        xgap=1,
        ygap=1,
    )
    if customdata is not None:
        heatmap_kwargs["customdata"] = customdata

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    # ── Alpha / beta group separator and labels ────────────────────────
    left_margin = 90
    if _group_separator_y is not None:
        left_margin = 160  # extra space for group label annotations
        n_cols = len(cols)
        # Dashed separator line across the plot area
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n_cols - 0.5, xref="x",
            y0=_group_separator_y, y1=_group_separator_y, yref="y",
            line=dict(color="#444444", width=1.5, dash="dash"),
        )
        # Group label annotations in the left margin (paper x = 0 is figure left edge)
        for y_pos, yanchor, label, color in [
            (_group_separator_y + 0.1, "bottom", "α emitters", "#b03030"),
            (_group_separator_y - 0.1, "top",    "β emitters", "#1a5fa8"),
        ]:
            fig.add_annotation(
                x=0.0, xref="paper", xanchor="left",
                y=y_pos, yref="y", yanchor=yanchor,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color=color),
            )

    fig.update_layout(
        xaxis=dict(title="Indication group", tickangle=-35, side="bottom"),
        yaxis_title="Radioisotope",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(320, len(top_isos) * 38 + 180),
        margin=dict(t=60, b=140, l=left_margin, r=100),
    )
    return fig


def build_isotope_indication_heatmap_assets(
    df_assets: pd.DataFrame,
    top_n_iso: int = 12,
    mode: str = "count",
    category: str = "All",
    iso_family_map: dict | None = None,
) -> go.Figure:
    """Drug-level isotope × indication heatmap (T11, drug view).

    Each unique drug/asset is counted once per indication it has been studied in,
    regardless of how many trials it accumulated.  Removes inflation from
    heavily-tested drugs (e.g. Y-90-ibritumomab with dozens of lymphoma trials).

    Note: drug Category is based on isotope type (Alpha/Beta = Therapeutic;
    PET/SPECT = Diagnostic). A Therapeutic drug may have been studied in
    Theranostic Pair trials — those trials' indications are included here
    (the therapeutic isotope was still tested therapeutically in those trials).

    Args:
        df_assets: Assets DataFrame (one row per unique drug/asset).
        top_n_iso: Number of isotopes to show, ranked by drug count.
        mode: 'count' — raw drug counts; 'pct_iso' — % of each isotope's total.
        category: 'All', 'Therapeutic', or 'Diagnostic'.
        iso_family_map: Optional dict mapping isotope name → Isotope_Family.
            When provided and category='Therapeutic', rows are grouped into
            alpha / beta sections separated by a dashed line.
    """
    # ── Category filter ────────────────────────────────────────────────
    if category == "Therapeutic":
        df_sub = df_assets[df_assets["Category"] == "Therapeutic"]
    elif category == "Diagnostic":
        df_sub = df_assets[df_assets["Category"] == "Diagnostic"]
    else:
        df_sub = df_assets

    needed = ["Radioisotope_Norm", "Indication_Group"]
    df_w = df_sub.dropna(subset=needed)[needed].copy()
    df_w["_AssetID"] = df_w.index  # preserve original row identity

    def _empty(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
        return fig

    if df_w.empty:
        return _empty("No data for selected filters")

    # ── Explode isotopes (some assets record compound isotopes) ─────────
    df_iso = df_w.assign(_Iso=df_w["Radioisotope_Norm"].str.split("; ")).explode("_Iso").reset_index(drop=True)
    df_iso["_Iso"] = df_iso["_Iso"].str.strip()

    # ── Explode indication groups ──────────────────────────────────────
    df_exp = df_iso.assign(_Grp=df_iso["Indication_Group"].str.split("; ")).explode("_Grp").reset_index(drop=True)
    df_exp["_Grp"] = df_exp["_Grp"].str.strip()
    df_exp = df_exp.dropna(subset=["_Iso", "_Grp"])
    df_exp = df_exp[(df_exp["_Iso"] != "") & (df_exp["_Grp"] != "")]
    # Deduplicate: each (drug, isotope, indication) counted once
    df_exp = df_exp.drop_duplicates(subset=["_AssetID", "_Iso", "_Grp"])

    if df_exp.empty:
        return _empty("No indication data for selected filters")

    # ── Select top N isotopes by drug count ───────────────────────────
    iso_counts = df_exp["_Iso"].value_counts()
    top_isos = iso_counts.head(top_n_iso).index.tolist()
    df_exp = df_exp[df_exp["_Iso"].isin(top_isos)]

    # ── Column order: YAML-defined indication groups ───────────────────
    all_indications = get_all_indication_groups()
    present_inds = set(df_exp["_Grp"].unique())
    cols = [g for g in all_indications if g in present_inds]

    # ── Cross-tabulation ───────────────────────────────────────────────
    ct = pd.crosstab(df_exp["_Iso"], df_exp["_Grp"]).reindex(
        index=top_isos, columns=cols, fill_value=0
    )

    # ── Row ordering: group alpha / beta when in Therapeutic mode ─────
    _group_separator_y: float | None = None
    _n_alpha = _n_beta = 0

    if category == "Therapeutic" and iso_family_map:
        alpha_isos = sorted(
            [iso for iso in top_isos if iso_family_map.get(iso) == "Alpha emitter"],
            key=lambda x: -iso_counts.get(x, 0),
        )
        beta_isos = sorted(
            [iso for iso in top_isos if iso_family_map.get(iso) != "Alpha emitter"],
            key=lambda x: -iso_counts.get(x, 0),
        )
        y_labels = list(reversed(alpha_isos)) + list(reversed(beta_isos))
        _n_alpha, _n_beta = len(alpha_isos), len(beta_isos)
        if _n_alpha > 0 and _n_beta > 0:
            _group_separator_y = _n_alpha - 0.5
    else:
        y_labels = list(reversed(top_isos))

    z_count = ct.reindex(y_labels).values.astype(float)

    # ── Build z matrix and annotations ────────────────────────────────
    if mode == "pct_iso":
        row_totals = z_count.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = np.nan
        z_vals = z_count / row_totals * 100
        colorscale = "YlOrRd"
        colorbar = dict(title="% of<br>isotope drugs", ticksuffix="%")
        z_text = [
            [f"{v:.0f}%" if not np.isnan(v) and v > 0 else "" for v in row]
            for row in z_vals
        ]
        hovertemplate = (
            "<b>%{y} · %{x}</b><br>"
            "% of isotope drugs: <b>%{z:.0f}%</b><br>"
            "Raw count: <b>%{customdata}</b><extra></extra>"
        )
        customdata = [[int(v) for v in row] for row in z_count]
    else:
        z_text = [
            [str(int(v)) if v > 0 else "" for v in row]
            for row in z_count
        ]
        z_vals = z_count.copy()
        z_vals[z_vals == 0] = np.nan
        colorscale = "Blues"
        colorbar = dict(title="Drug count")
        hovertemplate = (
            "<b>%{y} · %{x}</b><br>"
            "Drugs: <b>%{z:.0f}</b><extra></extra>"
        )
        customdata = None

    heatmap_kwargs = dict(
        z=z_vals,
        x=cols,
        y=y_labels,
        colorscale=colorscale,
        text=z_text,
        texttemplate="%{text}",
        hovertemplate=hovertemplate,
        colorbar=colorbar,
        xgap=1,
        ygap=1,
    )
    if customdata is not None:
        heatmap_kwargs["customdata"] = customdata

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    # ── Alpha / beta group separator and labels ────────────────────────
    left_margin = 90
    if _group_separator_y is not None:
        left_margin = 160
        n_cols = len(cols)
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n_cols - 0.5, xref="x",
            y0=_group_separator_y, y1=_group_separator_y, yref="y",
            line=dict(color="#444444", width=1.5, dash="dash"),
        )
        for y_pos, yanchor, label, color in [
            (_group_separator_y + 0.1, "bottom", "α emitters", "#b03030"),
            (_group_separator_y - 0.1, "top",    "β emitters", "#1a5fa8"),
        ]:
            fig.add_annotation(
                x=0.0, xref="paper", xanchor="left",
                y=y_pos, yref="y", yanchor=yanchor,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color=color),
            )

    fig.update_layout(
        xaxis=dict(title="Indication group", tickangle=-35, side="bottom"),
        yaxis_title="Radioisotope",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(320, len(top_isos) * 38 + 180),
        margin=dict(t=60, b=140, l=left_margin, r=100),
    )
    return fig



# ── 5D: Novelty combinations ───────────────────────────────────────────────────

def _compute_novelty_ages(df_display, df_ref,
                          x_col="Radioisotope_Norm", x_age_col="Isotope_Age"):
    """Compute target age and x-dimension age for each asset in df_display.

    Age = years since the target/x-value first appeared in df_ref.
    Multi-value fields (bispecific targets, compound isotopes) are exploded;
    the minimum age across all values is assigned to each asset.

    Args:
        x_col:     Column to use as the x-dimension (default: Radioisotope_Norm).
        x_age_col: Name of the output age column (default: Isotope_Age).

    Returns a copy of df_display (same index) with added Target_Age and
    <x_age_col> columns. Assets missing either column are dropped.
    """
    import pandas as _pd

    def _intro_years(df, col):
        rows = []
        for _, row in df.dropna(subset=[col, "First_Year"]).iterrows():
            for v in str(row[col]).split("; "):
                v = v.strip()
                if v:
                    rows.append({"val": v, "First_Year": row["First_Year"]})
        if not rows:
            return {}
        tmp = _pd.DataFrame(rows)
        return tmp.groupby("val")["First_Year"].min().to_dict()

    t_intro = _intro_years(df_ref, "Target_Antigen_Norm")
    x_intro = _intro_years(df_ref, x_col)

    records = []
    for idx, row in df_display.dropna(
        subset=["Target_Antigen_Norm", x_col, "First_Year"]
    ).iterrows():
        yr = row["First_Year"]
        t_ages = [
            yr - t_intro[tv]
            for t in str(row["Target_Antigen_Norm"]).split("; ")
            if (tv := t.strip()) and tv in t_intro
        ]
        x_ages = [
            yr - x_intro[xv]
            for xstr in str(row[x_col]).split("; ")
            if (xv := xstr.strip()) and xv in x_intro
        ]
        if not t_ages or not x_ages:
            continue
        records.append({
            "asset_idx": idx,
            "Target_Age": min(t_ages),
            x_age_col: min(x_ages),
        })

    if not records:
        return _pd.DataFrame()

    ages = _pd.DataFrame(records).set_index("asset_idx")
    return df_display.loc[ages.index].copy().assign(
        Target_Age=ages["Target_Age"],
        **{x_age_col: ages[x_age_col]},
    )


def build_novelty_combinations(df, df_ref=None, min_year=2000):
    """5D: First-ever novel isotope × first-ever novel target (post-2000 drugs).

    'First-ever' = Age == 0: this asset is the first in the entire dataset to
    use this target or isotope. Debut years are computed from df_ref (full
    dataset) regardless of which subset is passed as df.

    Restricted to First_Year >= min_year to exclude the pre-CT.gov founding era
    where 'first-ever' largely reflects missing registration data rather than
    true clinical novelty.

    Args:
        df: Filtered assets DataFrame.
        df_ref: Full reference assets for debut year computation. If None, uses df.
        min_year: Minimum First_Year to display (default 2000).
    """
    import plotly.graph_objects as _go

    if df_ref is None:
        df_ref = df

    def _empty(msg):
        fig = _go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
        return fig

    df_ages = _compute_novelty_ages(df, df_ref)
    if df_ages.empty:
        return _empty("No data with target, isotope, and year information")

    if "First_Year" in df_ages.columns:
        df_ages = df_ages[df_ages["First_Year"] >= min_year].copy()
    if df_ages.empty:
        return _empty(f"No assets with First_Year \u2265 {min_year}")

    novel_t  = df_ages["Target_Age"]  == 0
    novel_i  = df_ages["Isotope_Age"] == 0
    n_both    = int((novel_t & novel_i).sum())
    n_t_only  = int((novel_t & ~novel_i).sum())
    n_i_only  = int((~novel_t & novel_i).sum())
    n_neither = int((~novel_t & ~novel_i).sum())
    n_total   = len(df_ages)

    labels_col = ["First-ever isotope", "Established isotope"]
    labels_row = ["First-ever target", "Established target"]
    z_vals = [[n_both, n_t_only], [n_i_only, n_neither]]
    text = [
        [f"<b>{z_vals[r][c]}</b><br>{100*z_vals[r][c]/n_total:.0f}%" for c in range(2)]
        for r in range(2)
    ]

    fig = _go.Figure(_go.Heatmap(
        z=z_vals, x=labels_col, y=labels_row,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
        hovertemplate="Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Radioisotope", side="bottom"),
        yaxis=dict(title="Target antigen", autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=340,
        margin=dict(t=40, b=80, l=160, r=40),
        font=dict(size=13),
    )
    return fig


def build_novelty_scaffold_target(df, df_ref=None, threshold=3, view="matrix"):
    """Novel scaffold (format) x novel target — one variable at a time hypothesis.

    Same logic as build_novelty_combinations() but uses Antibody_Format_Norm
    instead of Radioisotope_Norm on the x-axis. Tests whether a novel scaffold
    format is preferentially paired with an established target and vice versa.

    Note: only ~15 distinct formats exist vs many more targets, so the
    'established format' column will dominate (Full-length IgG debuted 1985).
    The signal of interest is whether novel formats cluster with established targets.

    Args:
        df: Filtered assets DataFrame.
        df_ref: Full reference assets for computing debut years. If None, uses df.
        threshold: Years after debut to count as 'novel' (default 3).
        view: 'matrix' — 2x2 annotated heatmap (default);
              'scatter' — scatter plot with quadrant lines.
    """
    import plotly.graph_objects as _go

    if df_ref is None:
        df_ref = df

    def _empty(msg):
        fig = _go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
        return fig

    df_ages = _compute_novelty_ages(
        df, df_ref,
        x_col="Antibody_Format_Norm", x_age_col="Format_Age",
    )
    if df_ages.empty:
        return _empty("No data with format, target, and year information")

    novel_t = df_ages["Target_Age"] <= threshold
    novel_f = df_ages["Format_Age"] <= threshold
    n_both    = int((novel_t & novel_f).sum())
    n_t_only  = int((novel_t & ~novel_f).sum())
    n_f_only  = int((~novel_t & novel_f).sum())
    n_neither = int((~novel_t & ~novel_f).sum())
    n_total   = len(df_ages)

    # ── 2x2 matrix view (default) ──────────────────────────────────────
    if view != "scatter":
        labels_col = [f"Novel format (\u2264{threshold}y)", f"Established format (>{threshold}y)"]
        labels_row = [f"Novel target (\u2264{threshold}y)", f"Established target (>{threshold}y)"]
        z_vals = [[n_both, n_t_only], [n_f_only, n_neither]]
        text = [
            [f"<b>{z_vals[r][c]}</b><br>{100*z_vals[r][c]/n_total:.0f}%" for c in range(2)]
            for r in range(2)
        ]
        fig = _go.Figure(_go.Heatmap(
            z=z_vals, x=labels_col, y=labels_row,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            hovertemplate="Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="Antibody format novelty", side="bottom"),
            yaxis=dict(title="Target novelty", autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=340,
            margin=dict(t=40, b=80, l=160, r=40),
            font=dict(size=13),
        )
        return fig

    # ── Scatter view ───────────────────────────────────────────────────
    cat_color_map = {
        "Therapeutic":      CAT_COLORS.get("Therapeutic",      "#66C2A5"),
        "Diagnostic":       CAT_COLORS.get("Diagnostic",       "#FC8D62"),
        "Theranostic Pair": CAT_COLORS.get("Theranostic Pair", "#8DA0CB"),
    }
    fig = _go.Figure()
    for cat, grp in df_ages.groupby("Category"):
        color = cat_color_map.get(cat, "#888888")
        fig.add_trace(_go.Scatter(
            x=grp["Format_Age"],
            y=grp["Target_Age"],
            mode="markers",
            name=cat,
            marker=dict(color=color, size=9, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            customdata=grp[["Antibody_Name_Norm", "Target_Antigen_Norm",
                             "Antibody_Format_Norm", "First_Year"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Target: %{customdata[1]}<br>"
                "Format: %{customdata[2]}<br>"
                "First year: %{customdata[3]:.0f}<br>"
                "Format age: %{x:.0f}y  |  Target age: %{y:.0f}y"
                "<extra></extra>"
            ),
        ))

    ax_max = max(df_ages["Format_Age"].max(), df_ages["Target_Age"].max()) + 3
    for axis, val in [("x", threshold), ("y", threshold)]:
        fig.add_shape(
            type="line",
            x0=val if axis == "x" else 0,  x1=val if axis == "x" else ax_max,
            y0=val if axis == "y" else 0,   y1=val if axis == "y" else ax_max,
            line=dict(color="grey", width=1.5, dash="dash"),
        )

    q_x_novel = threshold / 2
    q_x_estab = threshold + (ax_max - threshold) / 2
    q_y_novel = threshold / 2
    q_y_estab = threshold + (ax_max - threshold) / 2
    for qx, qy, count, label in [
        (q_x_novel, q_y_novel, n_both,    f"<b>Both novel</b><br>n={n_both}"),
        (q_x_estab, q_y_novel, n_t_only,  f"Novel target<br>Established format<br>n={n_t_only}"),
        (q_x_novel, q_y_estab, n_f_only,  f"Established target<br>Novel format<br>n={n_f_only}"),
        (q_x_estab, q_y_estab, n_neither, f"Neither novel<br>n={n_neither}"),
    ]:
        pct = 100 * count / n_total if n_total else 0
        fig.add_annotation(
            x=qx, y=qy,
            text=f"{label}<br><i>({pct:.0f}%)</i>",
            showarrow=False, font=dict(size=11), align="center",
            bgcolor="rgba(255,255,255,0.65)", borderpad=3,
        )

    fig.update_layout(
        xaxis=dict(
            title="Format established years at drug debut  (0 = first appearance in the field)",
            range=[-1, ax_max], zeroline=False,
        ),
        yaxis=dict(
            title="Target established years at drug debut  (0 = first appearance in the field)",
            range=[-1, ax_max], zeroline=False,
        ),
        legend=dict(title="Category", orientation="v", x=1.01, y=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=520,
        margin=dict(t=40, b=80, l=90, r=160),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ── PK Compatibility charts (Tab 2, Step 5A) ──────────────────────────

# Physical half-lives in hours (source: NNDC/Brookhaven nuclear data)
_ISOTOPE_HL_H: dict[str, float] = {
    "Ga-68": 1.13,
    "F-18": 1.83,
    "Tc-99m": 6.0,
    "At-211": 7.2,
    "Pb-212": 10.6,
    "Cu-64": 12.7,
    "I-123": 13.2,
    "Re-188": 17.0,
    "Y-90": 64.0,
    "In-111": 67.3,
    "Zr-89": 78.4,
    "I-124": 100.3,
    "Lu-177": 159.5,
    "Tb-161": 166.9,
    "I-131": 193.0,
    "Ac-225": 238.1,
    "Th-227": 448.8,
    "I-125": 1426.0,
}

# Format-family order: short assumed PK → long assumed PK
# Non-Ab: Adnectin ~2h, Affibody ~28h (ABY-271, engineered for HSA binding)
# Small Ab: nanobody ~2-6h, minibody/diabody/SIP ~5-20h
# mAb Fragment: Fab ~12-24h, F(ab')2 ~24-48h; Full IgG ~504h (FcRn-mediated)
_FORMAT_PK_ORDER = [
    "Non-Ab protein scaffold",
    "Small Ab-derived scaffold",
    "mAb Fragment",
    "Full-length IgG",
]

# Individual scaffold order: short assumed PK → long assumed PK
# Sources: published PK data and size-based estimates; see CLAUDE.md notes.
# Affibody placed at ~28h based on ABY-271 (PMC8226825); unengineered affibodies are shorter.
_SCAFFOLD_PK_ORDER = [
    "Adnectin",           # Non-Ab, ~2h
    "scFv",               # mAb Fragment, ~2–4h (28 kDa, renal clearance)
    "Nanobody",           # Small Ab, ~2–6h (15 kDa, renal clearance)
    "Diabody",            # Small Ab, ~5h (~50–60 kDa)
    "Minibody",           # Small Ab, ~5–12h (~80 kDa)
    "SIP",                # Small Ab, ~5–20h (small immunoprotein, homodimer)
    "Antibody fragment",  # mAb Fragment, variable
    "Fab fragment",       # mAb Fragment, ~12–24h (~50 kDa)
    "F(ab')2 fragment",   # mAb Fragment, ~24–48h (~100 kDa)
    "Affibody",           # Non-Ab, ~28h for ABY-271 (engineered; typical unmodified: shorter)
    "One-armed antibody", # mAb Fragment, Fc-retained, ~7–21 days
    "Full-length IgG",    # ~21 days (FcRn-mediated recycling)
]

# Isotope category (Therapeutic = Alpha/Beta emitters; Diagnostic = PET/SPECT)
_ISOTOPE_CATEGORY: dict[str, str] = {
    "Ac-225": "Therapeutic", "At-211": "Therapeutic",
    "Th-227": "Therapeutic", "Pb-212": "Therapeutic",
    "Tb-161": "Therapeutic", "Lu-177": "Therapeutic",
    "Y-90":   "Therapeutic", "I-131":  "Therapeutic",
    "Re-188": "Therapeutic", "I-125":  "Therapeutic",
    "Ga-68":  "Diagnostic",  "F-18":   "Diagnostic",
    "Tc-99m": "Diagnostic",  "Cu-64":  "Diagnostic",
    "I-123":  "Diagnostic",  "Zr-89":  "Diagnostic",
    "In-111": "Diagnostic",  "I-124":  "Diagnostic",
}

# Format-family colors by isotope category: greens (therapeutic) and oranges (diagnostic)
# Shade encodes assumed PK: light = short, dark = long
_FORMAT_THERA_COLORS: dict[str, str] = {
    "Non-Ab protein scaffold":   "#A1D99B",
    "Small Ab-derived scaffold":  "#41AB5D",
    "mAb Fragment":              "#238B45",
    "Full-length IgG":           "#00441B",
}
_FORMAT_DIAG_COLORS: dict[str, str] = {
    "Non-Ab protein scaffold":   "#FDAE6B",
    "Small Ab-derived scaffold":  "#FD8D3C",
    "mAb Fragment":              "#D94801",
    "Full-length IgG":           "#7F2704",
}

# Individual scaffold colors by isotope category
_SCAFFOLD_THERA_COLORS: dict[str, str] = {
    "Adnectin":           "#C7E9C0",
    "scFv":               "#A1D99B",
    "Nanobody":           "#74C476",
    "Diabody":            "#41AB5D",
    "Minibody":           "#2CA25F",
    "SIP":                "#238B45",
    "Antibody fragment":  "#006D2C",
    "Fab fragment":       "#005924",
    "F(ab')2 fragment":   "#00441B",
    "Affibody":           "#003915",
    "One-armed antibody": "#002D10",
    "Full-length IgG":    "#00220C",
}
_SCAFFOLD_DIAG_COLORS: dict[str, str] = {
    "Adnectin":           "#FDD0A2",
    "scFv":               "#FDAE6B",
    "Nanobody":           "#FD8D3C",
    "Diabody":            "#F16913",
    "Minibody":           "#E05B0A",
    "SIP":                "#D94801",
    "Antibody fragment":  "#A63603",
    "Fab fragment":       "#8B2E02",
    "F(ab')2 fragment":   "#7F2704",
    "Affibody":           "#6B2103",
    "One-armed antibody": "#561A02",
    "Full-length IgG":    "#430F00",
}


def _hl_label(iso: str) -> str:
    """Return isotope name with half-life annotation, e.g. 'Lu-177 (6.6d)'."""
    hl = _ISOTOPE_HL_H.get(iso)
    if hl is None:
        return iso
    if hl < 24:
        return f"{iso}  ({hl:.1f}h)"
    elif hl < 240:
        return f"{iso}  ({hl / 24:.1f}d)"
    else:
        return f"{iso}  ({hl / 24:.0f}d)"


def _prepare_pk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared preprocessing for PK compatibility charts.
    Excludes: bispecifics, pretargeting assets (Linker_Technology_Norm),
              unknown scaffold (Antibody_Format_Norm == 'Other protein scaffold (specify)').
    Expands compound isotopes ('Y-90; In-111'): each drug is counted once per isotope
    it contains (weight=1), so integer counts appear in charts.
    Keeps only isotopes present in _ISOTOPE_HL_H.
    Returns DataFrame with columns: Radioisotope_Norm, Format_Family,
                                    Antibody_Format_Norm, _weight.
    """
    out = df.copy()
    out = out[out["Format_Family"] != "Bispecific"]
    if "Linker_Technology_Norm" in out.columns:
        out = out[~out["Linker_Technology_Norm"].str.contains(
            "pretargeting", case=False, na=False
        )]
    if "Antibody_Format_Norm" in out.columns:
        out = out[out["Antibody_Format_Norm"] != "Other protein scaffold (specify)"]
    out = out.dropna(subset=["Radioisotope_Norm", "Format_Family"])

    rows: list[dict] = []
    for _, row in out.iterrows():
        isotopes = [i.strip() for i in str(row["Radioisotope_Norm"]).split("; ")]
        fmt_norm = str(row.get("Antibody_Format_Norm", ""))
        for iso in isotopes:
            if iso in _ISOTOPE_HL_H:
                rows.append({
                    "Radioisotope_Norm": iso,
                    "Format_Family": str(row["Format_Family"]),
                    "Antibody_Format_Norm": fmt_norm,
                    "_weight": 1,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "Radioisotope_Norm", "Format_Family", "Antibody_Format_Norm", "_weight"
        ])
    return pd.DataFrame(rows)


def build_pk_heatmap(df: pd.DataFrame, grouping: str = "family") -> go.Figure:
    """
    PK compatibility heatmap.
    Y = radioisotopes sorted by physical t½ ascending (shortest at bottom).
    X = antibody format sorted by assumed biological PK ascending (shortest at left).
    Cell value = drug count (each drug counted once per isotope it contains).
    grouping: 'family' → Format_Family (4 groups); 'scaffold' → Antibody_Format_Norm (individual types).
    Bispecifics, pretargeting assets, and unknown scaffolds excluded.
    """
    pk_df = _prepare_pk_df(df)
    if pk_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    iso_present = pk_df["Radioisotope_Norm"].unique().tolist()
    iso_sorted = sorted(iso_present, key=lambda x: _ISOTOPE_HL_H.get(x, 9999))
    iso_labels = [_hl_label(i) for i in iso_sorted]

    if grouping == "scaffold":
        grp_col = "Antibody_Format_Norm"
        full_order = _SCAFFOLD_PK_ORDER
    else:
        grp_col = "Format_Family"
        full_order = _FORMAT_PK_ORDER

    grp_present = pk_df[grp_col].unique().tolist()
    grp_ordered = [g for g in full_order if g in grp_present]

    pivot = (
        pk_df.groupby(["Radioisotope_Norm", grp_col])["_weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=iso_sorted, columns=grp_ordered, fill_value=0)
    )

    z_vals = pivot.values.tolist()
    annot_text = [
        [("" if v == 0 else str(int(v))) for v in row]
        for row in z_vals
    ]

    height = max(420, 60 + len(iso_sorted) * 26)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=grp_ordered,
        y=iso_labels,
        text=annot_text,
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Drugs (n)", thickness=12, len=0.6),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Drugs: %{z:d}<extra></extra>",
    ))

    x_label = (
        "Antibody scaffold  →  short to long assumed PK"
        if grouping == "scaffold"
        else "Antibody format  →  short to long assumed PK"
    )
    fig.update_layout(
        xaxis=dict(title=x_label, side="bottom", tickangle=30 if grouping == "scaffold" else 0),
        yaxis=dict(title="Radioisotope  →  short to long physical t½", autorange=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=60, b=140 if grouping == "scaffold" else 100, l=170, r=80),
    )
    return fig


def _pk_color_lists(
    grp_ordered: list[str],
    iso_sorted: list[str],
    thera_colors: dict,
    diag_colors: dict,
) -> dict[str, list[str]]:
    """Return {group: [color_per_isotope]} using green for therapeutic, orange for diagnostic."""
    result = {}
    for grp in grp_ordered:
        result[grp] = [
            thera_colors.get(grp, "#66C2A5")
            if _ISOTOPE_CATEGORY.get(iso) == "Therapeutic"
            else diag_colors.get(grp, "#FC8D62")
            for iso in iso_sorted
        ]
    return result


def build_pk_stacked_bar(df: pd.DataFrame, grouping: str = "family") -> go.Figure:
    """
    PK compatibility stacked bar (combined, therapeutic + diagnostic isotopes).
    X = radioisotopes sorted by physical t½ (short→long, annotated).
    Bar shade encodes assumed PK (light = short, dark = long).
    Bar hue encodes isotope category: green = therapeutic, orange = diagnostic.
    grouping: 'family' → Format_Family (4 groups); 'scaffold' → Antibody_Format_Norm.
    Bispecifics, pretargeting assets, and unknown scaffolds excluded.
    """
    pk_df = _prepare_pk_df(df)
    if pk_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    iso_present = pk_df["Radioisotope_Norm"].unique().tolist()
    iso_sorted = sorted(iso_present, key=lambda x: _ISOTOPE_HL_H.get(x, 9999))
    iso_labels = [_hl_label(i) for i in iso_sorted]

    if grouping == "scaffold":
        grp_col = "Antibody_Format_Norm"
        full_order = _SCAFFOLD_PK_ORDER
        thera_colors, diag_colors = _SCAFFOLD_THERA_COLORS, _SCAFFOLD_DIAG_COLORS
    else:
        grp_col = "Format_Family"
        full_order = _FORMAT_PK_ORDER
        thera_colors, diag_colors = _FORMAT_THERA_COLORS, _FORMAT_DIAG_COLORS

    grp_present = pk_df[grp_col].unique().tolist()
    grp_ordered = [g for g in full_order if g in grp_present]

    pivot = (
        pk_df.groupby(["Radioisotope_Norm", grp_col])["_weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=iso_sorted, columns=grp_ordered, fill_value=0)
    )

    color_lists = _pk_color_lists(grp_ordered, iso_sorted, thera_colors, diag_colors)

    fig = go.Figure()
    for grp in grp_ordered:
        fig.add_trace(go.Bar(
            name=grp,
            x=iso_labels,
            y=pivot[grp].tolist(),
            marker_color=color_lists[grp],
            showlegend=False,
            hovertemplate=f"<b>%{{x}}</b><br>{grp}: %{{y:.0f}}<extra></extra>",
        ))

    # Legend: two sets of boxes — one for therapeutic (green), one for diagnostic (orange)
    for grp in grp_ordered:
        fig.add_trace(go.Scatter(
            name=grp, x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=thera_colors.get(grp, "#66C2A5")),
            legendgroup="thera",
            legendgrouptitle_text="Therapeutic isotopes",
            showlegend=True,
        ))
    for grp in grp_ordered:
        fig.add_trace(go.Scatter(
            name=grp, x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=diag_colors.get(grp, "#FC8D62")),
            legendgroup="diag",
            legendgrouptitle_text="Diagnostic isotopes",
            showlegend=True,
        ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Radioisotope  →  physical t½ (short to long)", tickangle=30),
        yaxis=dict(title="Number of drugs", gridcolor="#EEEEEE"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=40, b=180, l=60, r=40),
    )
    return fig


def build_pk_stacked_bar_split(df: pd.DataFrame, grouping: str = "family") -> go.Figure:
    """
    PK compatibility stacked bar, split into Therapeutic (left) and Diagnostic (right) panels.
    Therapeutic panel uses green shades; Diagnostic uses orange shades.
    Shade encodes assumed PK: light = short, dark = long.
    grouping: 'family' → Format_Family; 'scaffold' → Antibody_Format_Norm.
    """
    pk_df = _prepare_pk_df(df)
    if pk_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    iso_present = pk_df["Radioisotope_Norm"].unique().tolist()
    iso_sorted = sorted(iso_present, key=lambda x: _ISOTOPE_HL_H.get(x, 9999))

    thera_isos = [i for i in iso_sorted if _ISOTOPE_CATEGORY.get(i) == "Therapeutic"]
    diag_isos  = [i for i in iso_sorted if _ISOTOPE_CATEGORY.get(i) == "Diagnostic"]

    if grouping == "scaffold":
        grp_col = "Antibody_Format_Norm"
        full_order = _SCAFFOLD_PK_ORDER
        thera_colors, diag_colors = _SCAFFOLD_THERA_COLORS, _SCAFFOLD_DIAG_COLORS
    else:
        grp_col = "Format_Family"
        full_order = _FORMAT_PK_ORDER
        thera_colors, diag_colors = _FORMAT_THERA_COLORS, _FORMAT_DIAG_COLORS

    grp_present = pk_df[grp_col].unique().tolist()
    grp_ordered = [g for g in full_order if g in grp_present]

    pivot = (
        pk_df.groupby(["Radioisotope_Norm", grp_col])["_weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=iso_sorted, columns=grp_ordered, fill_value=0)
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Therapeutic isotopes", "Diagnostic isotopes"],
        horizontal_spacing=0.08,
    )

    for col_idx, (isos, colors) in enumerate(
        [(thera_isos, thera_colors), (diag_isos, diag_colors)], start=1
    ):
        iso_labels_sub = [_hl_label(i) for i in isos]
        pivot_sub = pivot.reindex(index=isos, fill_value=0)
        show_legend = col_idx == 1

        for grp in grp_ordered:
            fig.add_trace(
                go.Bar(
                    name=grp,
                    x=iso_labels_sub,
                    y=pivot_sub[grp].tolist(),
                    marker_color=colors.get(grp, "#888888"),
                    showlegend=show_legend,
                    legendgroup=grp,
                    hovertemplate=f"<b>%{{x}}</b><br>{grp}: %{{y:.0f}}<extra></extra>",
                ),
                row=1, col=col_idx,
            )

    legend_title = (
        "<b>Antibody scaffold</b><br><i>short → long PK ↓</i>"
        if grouping == "scaffold"
        else "<b>Antibody format</b><br><i>short → long PK ↓</i>"
    )
    fig.update_layout(
        barmode="stack",
        legend=dict(title=legend_title, orientation="v", x=1.01, y=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(t=60, b=130, l=60, r=220),
    )
    fig.update_xaxes(tickangle=30, title_text="Radioisotope (t½)")
    fig.update_yaxes(title_text="Number of drugs", gridcolor="#EEEEEE", col=1)
    fig.update_yaxes(gridcolor="#EEEEEE", col=2)
    return fig
