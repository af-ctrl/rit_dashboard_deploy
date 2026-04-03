"""
Tab 3: Explorer
Free-parameter chart: select any primary dimension and optional color-by split.
Chart type is auto-selected based on the dimension types.
"""

import pandas as pd
import streamlit as st

from dashboard.utils.chart_helpers import (
    build_explorer_bar,
    build_explorer_box,
    build_explorer_histogram,
    build_explorer_heatmap,
    build_explorer_time_bar,
)
from dashboard.utils.constants import (
    CAT_COLORS, PHASE_COLORS, PHASE_ORDER, QUAL_COLORS, SPONSOR_COLORS,
)
from dashboard.utils.data_loader import apply_sidebar_filters, load_antigen_family_lookup


# ── Status grouping ────────────────────────────────────────────────────

_STATUS_GROUPS: dict[str, str] = {
    "RECRUITING":               "Active",
    "ACTIVE_NOT_RECRUITING":    "Active",
    "NOT_YET_RECRUITING":       "Active",
    "ENROLLING_BY_INVITATION":  "Active",
    "COMPLETED":                "Completed",
    "TERMINATED":               "Terminated",
    "UNKNOWN":                  "Unknown",
    "WITHDRAWN":                "Withdrawn/Suspended",
    "SUSPENDED":                "Withdrawn/Suspended",
    "AVAILABLE":                "Withdrawn/Suspended",
    "NO_LONGER_AVAILABLE":      "Withdrawn/Suspended",
}
_STATUS_GROUP_ORDER = ["Active", "Completed", "Terminated", "Unknown", "Withdrawn/Suspended"]
_STATUS_GROUP_COLORS = {
    "Active":               "#2196F3",
    "Completed":            "#4CAF50",
    "Terminated":           "#D62728",
    "Unknown":              "#9E9E9E",
    "Withdrawn/Suspended":  "#FF9800",
}

# ── Parameter definitions ──────────────────────────────────────────────
# type: "time" | "high_cat" | "low_cat" | "continuous"
# col:  actual column name (after any preprocessing — e.g. Status_Group)
# order: canonical sort order for low-card dimensions (first = leftmost / topmost)
# explode: True when compound values need splitting on "; "

_TRIAL_PARAMS: dict[str, dict] = {
    "Start Year":    {"col": "Start_Year",         "type": "time"},
    "Target Antigen":{"col": "Target_Antigen_Norm", "type": "high_cat", "explode": True},
    "Radioisotope":  {"col": "Radioisotope_Norm",   "type": "high_cat"},
    "Category":      {"col": "Category",            "type": "low_cat"},
    "Phase":         {"col": "Phase_Norm",          "type": "low_cat",  "order": PHASE_ORDER},
    "Isotope Family":{"col": "Isotope_Family",      "type": "low_cat"},
    "Format Family": {"col": "Format_Family",       "type": "low_cat"},
    "Sponsor Type":  {"col": "Sponsor_Type",        "type": "low_cat"},
    "Target Family": {"col": "Target_Family",       "type": "low_cat"},
    "Status":        {"col": "Status_Group",        "type": "low_cat",  "order": _STATUS_GROUP_ORDER},
    "Enrollment":    {"col": "Enrollment",          "type": "continuous"},
}

_ASSET_PARAMS: dict[str, dict] = {
    "First Year":    {"col": "First_Year",          "type": "time"},
    "Target Antigen":{"col": "Target_Antigen_Norm", "type": "high_cat", "explode": True},
    "Radioisotope":  {"col": "Radioisotope_Norm",   "type": "high_cat"},
    "Category":      {"col": "Category",            "type": "low_cat"},
    "Highest Phase": {"col": "Highest_Phase_Norm",  "type": "low_cat",  "order": PHASE_ORDER},
    "Isotope Family":{"col": "Isotope_Family",      "type": "low_cat"},
    "Format Family": {"col": "Format_Family",       "type": "low_cat"},
    "Target Family": {"col": "Target_Family",       "type": "low_cat"},
    "Trials per drug":{"col": "N_Trials",            "type": "continuous"},
}

# Low-card parameters eligible as "Split / color by" (exclude high-card and time)
_COLOR_OPTIONS_TRIAL = [
    "Category", "Phase", "Isotope Family", "Format Family",
    "Sponsor Type", "Target Family", "Status",
]
_COLOR_OPTIONS_ASSET = [
    "Category", "Highest Phase", "Isotope Family", "Format Family", "Target Family",
]

# Parameters with potentially high null rates — show caveat when selected
_NULL_CAVEAT_PARAMS = {"Target Family"}

# Human-readable chart type labels
_CHART_TYPE_LABELS = {
    "time":      "Stacked bar over time",
    "bar":       "Horizontal bar chart",
    "heatmap":   "Count heatmap",
    "box":       "Box plot",
    "histogram": "Histogram",
}


def _build_color_map(col: str, values: list[str]) -> dict[str, str]:
    """Return a stable name → hex color mapping for the given column and values."""
    if col == "Category":
        return {v: CAT_COLORS.get(v, "#CCCCCC") for v in values}
    if col in ("Phase_Norm", "Highest_Phase_Norm"):
        return {v: PHASE_COLORS.get(v, "#CCCCCC") for v in values}
    if col == "Sponsor_Type":
        return {v: SPONSOR_COLORS.get(v, "#CCCCCC") for v in values}
    if col == "Status_Group":
        return {v: _STATUS_GROUP_COLORS.get(v, "#CCCCCC") for v in values}
    return {v: QUAL_COLORS[i % len(QUAL_COLORS)] for i, v in enumerate(values)}


def render(
    df_relevant: "pd.DataFrame",
    df_assets: "pd.DataFrame",
    year_range: tuple,
    categories: list,
    format_families: list | None = None,
    isotope_families: list | None = None,
    phases: list | None = None,
    antigens: list | None = None,
    indication_groups: list | None = None,
    bin_size: int = 5,
) -> None:
    """Render the Explorer tab."""

    st.caption(
        "Free-form exploration of trials and drugs. "
        "Select a primary dimension and an optional split variable — "
        "the chart type is chosen automatically. "
        "Use this tab to cross-tabulate any two variables not covered by the pre-built charts."
    )

    # ── Data level toggle ──────────────────────────────────────────────
    data_level = st.radio(
        "Data level",
        ["Trials", "Drugs (unique)"],
        horizontal=True,
        key="explorer_data_level",
    )
    is_trials = data_level == "Trials"
    params = _TRIAL_PARAMS if is_trials else _ASSET_PARAMS
    color_options = _COLOR_OPTIONS_TRIAL if is_trials else _COLOR_OPTIONS_ASSET
    entity = "trials" if is_trials else "drugs"

    # ── Apply sidebar filters ──────────────────────────────────────────
    if is_trials:
        df = apply_sidebar_filters(
            df_relevant, year_range, categories,
            format_families=format_families, isotope_families=isotope_families,
            phases=phases, antigens=antigens, indication_groups=indication_groups,
        )
        df = df.copy()
        df["Status_Group"] = df["Status"].map(_STATUS_GROUPS).fillna("Unknown")
    else:
        df = df_assets.copy()
        if categories:
            df = df[df["Category"].isin(categories)]
        if "First_Year" in df.columns:
            yr = df["First_Year"]
            df = df[yr.isna() | ((yr >= year_range[0]) & (yr <= year_range[1]))]
        if format_families is not None and "Format_Family" in df.columns:
            df = df[df["Format_Family"].isin(format_families)]
        if isotope_families is not None and "Isotope_Family" in df.columns:
            df = df[df["Isotope_Family"].isin(isotope_families)]
        if phases is not None and "Highest_Phase_Norm" in df.columns:
            df = df[df["Highest_Phase_Norm"].isin(phases)]
        if antigens is not None and "Target_Antigen_Norm" in df.columns:
            _antigen_set = set(antigens)
            df = df[df["Target_Antigen_Norm"].apply(
                lambda v: not pd.isna(v) and bool({s.strip() for s in str(v).split("; ")} & _antigen_set)
            )]
        if indication_groups is not None and "Indication_Group" in df.columns:
            _grp_set = set(indication_groups)
            df = df[df["Indication_Group"].apply(
                lambda v: not pd.isna(v) and bool({s.strip() for s in str(v).split("; ")} & _grp_set)
            )]

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    # ── Parameter selectors ────────────────────────────────────────────
    primary_options = list(params.keys())
    default_primary = "Target Antigen"

    col1, col2 = st.columns(2)
    with col1:
        primary = st.selectbox(
            "Primary dimension",
            primary_options,
            index=primary_options.index(default_primary) if default_primary in primary_options else 0,
            key=f"explorer_primary_{data_level}",
        )

    avail_color = [o for o in color_options if o != primary]
    # Default color-by: prefer Isotope Family, then first available
    default_color = next(
        (o for o in ["Isotope Family", "Category", "Format Family"] if o in avail_color),
        avail_color[0] if avail_color else None,
    )
    color_opts = ["(none)"] + avail_color
    default_color_idx = color_opts.index(default_color) if default_color in color_opts else 0

    with col2:
        color_by = st.selectbox(
            "Split / color by",
            color_opts,
            index=default_color_idx,
            key=f"explorer_color_{data_level}",
        )
    color_by_resolved = None if color_by == "(none)" else color_by

    # ── Look up parameter metadata ─────────────────────────────────────
    p_info   = params[primary]
    p_col    = p_info["col"]
    p_type   = p_info["type"]
    p_order  = p_info.get("order")
    p_explode = p_info.get("explode", False)

    c_info   = params.get(color_by_resolved) if color_by_resolved else None
    c_col    = c_info["col"]   if c_info else None
    c_order  = c_info.get("order") if c_info else None
    c_type   = c_info["type"]  if c_info else None

    # ── Auto-select chart type ─────────────────────────────────────────
    if p_type == "time":
        chart_type = "time"
    elif p_type == "high_cat":
        chart_type = "bar"
    elif p_type == "low_cat":
        chart_type = "heatmap" if c_type == "low_cat" else "bar"
    else:  # continuous
        chart_type = "box" if c_col else "histogram"

    # ── Additional controls (context-sensitive) ────────────────────────
    ctrl_cols = st.columns([2, 2, 3])
    top_n      = 15
    time_mode  = "individual"
    log_scale  = True

    if p_type == "high_cat":
        with ctrl_cols[0]:
            top_n = st.slider(
                "Show top N", 5, 25, 15,
                key=f"explorer_topn_{data_level}_{primary}",
            )

    if p_type == "time":
        with ctrl_cols[0]:
            time_label = st.radio(
                "Time resolution",
                ["Individual years", f"Period bins ({bin_size}y)"],
                horizontal=True,
                key=f"explorer_timemode_{data_level}",
            )
            time_mode = "bins" if "Period" in time_label else "individual"

    if p_type == "continuous":
        with ctrl_cols[0]:
            log_scale = st.checkbox(
                "Log scale", value=True,
                key=f"explorer_log_{data_level}_{primary}",
            )

    with ctrl_cols[2]:
        st.markdown(f"**Chart type:** {_CHART_TYPE_LABELS[chart_type]}")

    # ── Null caveat tracking (before any data manipulation) ────────────
    n_total     = len(df)
    null_notes  = []

    if p_col in df.columns:
        n_null_p = int(df[p_col].isna().sum())
        if n_null_p > 0 and primary in _NULL_CAVEAT_PARAMS:
            null_notes.append(
                f"**{primary}**: {n_null_p:,} of {n_total:,} {entity} "
                "have no value for this field and are excluded from the chart."
            )

    if c_col and c_col in df.columns:
        n_null_c = int(df[c_col].isna().sum())
        if n_null_c > 0 and color_by_resolved in _NULL_CAVEAT_PARAMS:
            null_notes.append(
                f"**{color_by_resolved}**: {n_null_c:,} of {n_total:,} {entity} "
                "have no value for this field and are excluded from the chart."
            )

    # ── Preprocess data ────────────────────────────────────────────────
    df_chart = df.copy()

    # Explode compound target values ("EGFR; c-MET" → two rows)
    if p_explode and p_type == "high_cat":
        df_chart = df_chart.dropna(subset=[p_col]).copy()
        df_chart = df_chart.assign(
            **{p_col: df_chart[p_col].str.split("; ")}
        ).explode(p_col).reset_index(drop=True)
        df_chart[p_col] = df_chart[p_col].str.strip()
        # After exploding, re-derive Target_Family per individual antigen.
        # Without this, compound rows (e.g. "VEGF-A; CAIX") inherit the compound
        # entry's family ("Vascular / angiogenesis") for both split antigens,
        # causing CAIX to appear in the wrong family.
        if "Target_Family" in df_chart.columns:
            _tf_lookup = load_antigen_family_lookup()
            if _tf_lookup:
                df_chart["Target_Family"] = df_chart[p_col].map(_tf_lookup)

    # Restrict to top-N values for high-cardinality primary
    if p_type == "high_cat":
        top_vals = df_chart[p_col].value_counts().head(top_n).index.tolist()
        df_chart = df_chart[df_chart[p_col].isin(top_vals)]
        p_order = None  # frequency order used instead

    # Build color map
    color_map = None
    if c_col and c_col in df_chart.columns:
        c_vals = df_chart[c_col].dropna().unique().tolist()
        if c_order:
            c_vals_ordered = [v for v in c_order if v in c_vals]
            c_vals_ordered += [v for v in c_vals if v not in c_order]
        else:
            c_vals_ordered = sorted(c_vals)
        color_map = _build_color_map(c_col, c_vals_ordered)

    # ── Count and show caption ─────────────────────────────────────────
    n_chart = int(df_chart[p_col].notna().sum()) if p_col in df_chart.columns else 0
    st.caption(f"Showing **{n_chart:,}** {entity} in this chart.")

    # ── Build chart ────────────────────────────────────────────────────
    if chart_type == "time":
        fig = build_explorer_time_bar(
            df_chart, time_col=p_col,
            color_col=c_col, color_label=color_by_resolved,
            color_map=color_map, color_order=c_order,
            time_mode=time_mode, bin_size=bin_size,
        )

    elif chart_type == "heatmap":
        fig = build_explorer_heatmap(
            df_chart,
            x_col=p_col, y_col=c_col,
            x_label=primary, y_label=color_by_resolved,
            x_order=p_order, y_order=c_order,
        )

    elif chart_type == "bar":
        fig = build_explorer_bar(
            df_chart,
            x_col=p_col, x_label=primary,
            color_col=c_col, color_label=color_by_resolved,
            color_map=color_map,
            x_order=p_order, color_order=c_order,
        )

    elif chart_type == "box":
        fig = build_explorer_box(
            df_chart,
            group_col=c_col, value_col=p_col,
            group_label=color_by_resolved, value_label=primary,
            color_map=color_map, cat_order=c_order,
            log_scale=log_scale,
        )

    else:  # histogram
        fig = build_explorer_histogram(df_chart, col=p_col, label=primary, log_scale=log_scale)

    st.plotly_chart(fig, use_container_width=True)

    # ── Null caveats ───────────────────────────────────────────────────
    for note in null_notes:
        st.caption(f"Note: {note}")

    # ── Dimension-specific explanatory notes ───────────────────────────
    if primary == "Trials per drug" or color_by_resolved == "Trials per drug":
        st.caption(
            "**Trials per drug**: number of registered clinical trials in the CT.gov dataset "
            "that include this drug (unique antibody + isotope + chelator combination). "
            "Reflects how extensively each drug has been studied, not the total trial count "
            "for an isotope family."
        )

    # ── Status grouping explanation ────────────────────────────────────
    if is_trials and (primary == "Status" or color_by_resolved == "Status"):
        with st.expander("Status grouping"):
            st.markdown(
                "Trial statuses are grouped into 5 categories for readability:  \n"
                "**Active** — Recruiting / Active (not recruiting) / Not yet recruiting / "
                "Enrolling by invitation  \n"
                "**Completed** — Completed  \n"
                "**Terminated** — Terminated  \n"
                "**Unknown** — Status unknown  \n"
                "**Withdrawn/Suspended** — Withdrawn / Suspended / No longer available"
            )
