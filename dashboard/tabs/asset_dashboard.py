"""
Tab 2: Drug-Level Analysis
One row per unique (Antibody + Isotope + Chelator) combination.
"""

import pandas as pd
import streamlit as st

from dashboard.utils.chart_helpers import (
    add_event_lines,
    build_company_portfolio_scatter,
    build_drug_landscape_scatter,
    build_drug_phase_timeline,
    build_format_evolution_assets,
    build_isotope_distribution_assets,
    build_isotope_evolution_assets,
    build_new_assets_combined,
    build_pk_heatmap,
    build_pk_stacked_bar,
    build_pk_stacked_bar_split,
    build_top_targets_assets,
    get_top_formats,
    get_top_isotopes,
)


def render(
    assets: "pd.DataFrame",
    df_trials: "pd.DataFrame",
    year_range: tuple,
    categories: list,
    format_families: list | None = None,
    isotope_families: list | None = None,
    phases: list | None = None,
    antigens: list | None = None,
    indication_groups: list | None = None,
    show_events: bool = False,
    bin_size: int = 5,
) -> None:
    """Render all drug-level charts."""
    # Apply filters to assets
    df = assets.copy()
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
        st.warning("No assets match the current filters.")
        return

    n_assets = len(df)
    st.caption(
        f"Showing **{n_assets:,}** unique drugs (Antibody + Isotope + Chelator combinations) "
        "matching current filters."
    )

    # ── D8→order 1: Company clinical portfolios ──────────────────────
    st.markdown("### Overview of current company clinical portfolios")
    top_n_company = st.slider(
        "Show top N companies",
        min_value=5, max_value=30, value=15,
        key="company_portfolio_top_n",
    )
    st.plotly_chart(
        build_company_portfolio_scatter(df, df_trials, top_n=top_n_company),
        use_container_width=True,
    )
    st.caption(
        "Industry sponsor portfolios ranked by number of distinct drugs (highest at top). "
        "Each dot represents a distinct drug. Dot size reflects trial count; "
        "color encodes isotope family; shape indicates drug format. "
        "Only company-owned assets are shown (academic and unknown ownership excluded)."
    )

    st.divider()

    # ── D2→order 2: New unique drugs over time ───────────────────────
    st.markdown("### New unique drugs over time")
    new_drugs_mode = st.radio(
        "Show",
        ["All drugs (by year)", "Phase 1 entrants (by period)"],
        horizontal=True,
        key="new_drugs_mode",
    )
    fig_new = build_new_assets_combined(
        df, mode="phase1" if new_drugs_mode.startswith("Phase 1") else "all",
        bin_size=bin_size,
    )
    if show_events and new_drugs_mode.startswith("All"):
        add_event_lines(fig_new, x_min=year_range[0], x_max=year_range[1])
    st.plotly_chart(fig_new, use_container_width=True)
    st.caption(
        "**All drugs**: number of distinct drugs entering clinical development each year "
        "(year of first registered trial, any phase). "
        "**Phase 1 entrants**: drugs reaching Phase 1 or Phase 1/2 for the first time, "
        "aggregated by period to compensate for data sparsity."
    )

    st.divider()

    # ── D1→order 3: Target distribution ──────────────────────────────
    st.markdown("### Target distribution")
    top_n = st.slider("Show top N targets", min_value=5, max_value=25, value=15, key="asset_top_n")
    st.plotly_chart(build_top_targets_assets(df, top_n=top_n), use_container_width=True)
    st.caption(
        "Distribution of target antigens across therapeutic (left) and diagnostic (right) drugs. "
        "Bispecific antibody targets are split so that each antigen is counted individually. "
        "Each distinct antibody-isotope-chelator combination is counted once, "
        "removing the bias from drugs with many registered trials."
    )

    st.divider()

    # ── D6→order 4: Drug landscape by target antigen ─────────────────
    st.markdown("### Drug landscape by target antigen")

    col_filter, col_n, col_color = st.columns([3, 2, 3])

    with col_filter:
        filter_recent = st.checkbox(
            "Limit to drugs with recent trials (Latest Year >= 2020)",
            value=True,
            key="landscape_filter_recent",
            help=(
                "When enabled, only drugs that had at least one trial registered in 2020 "
                "or later are shown. Phase 3 and Phase 4 drugs are always included as "
                "historical reference regardless of recency."
            ),
        )

    with col_n:
        top_n_landscape = st.slider(
            "Top N targets",
            min_value=5, max_value=40, value=10,
            key="landscape_top_n",
        )

    with col_color:
        color_choice = st.radio(
            "Color by",
            ["Isotope Family", "Highest Phase"],
            horizontal=True,
            key="landscape_color_by",
        )
    color_col = "Isotope_Family" if color_choice == "Isotope Family" else "Highest_Phase_Norm"

    # Apply recency filter (Phase 3+ always included)
    df_bubble = df.copy()
    if filter_recent:
        df_bubble = df_bubble[
            (df_bubble["Latest_Year"] >= 2020)
            | df_bubble["Highest_Phase_Norm"].isin(["Phase 3", "Phase 4"])
        ]

    st.plotly_chart(
        build_drug_landscape_scatter(df_bubble, color_by=color_col, top_n=top_n_landscape),
        use_container_width=True,
    )
    st.caption(
        "Drug landscape by target antigen (top N by drug count). "
        "Y-axis ranked by number of distinct drugs. Dot size reflects trial count; "
        "dot shape indicates antibody format. "
        "Toggle color between isotope family and highest clinical phase."
    )

    st.divider()

    # ── D3→order 5: Radioisotope distribution ────────────────────────
    st.markdown("### Radioisotope distribution")
    def _exploded_nunique(sub):
        return sub["Radioisotope_Norm"].dropna().str.split("; ").explode().str.strip().nunique()
    _n_iso_avail = max(5, max(
        _exploded_nunique(df[df["Category"] == "Therapeutic"]),
        _exploded_nunique(df[df["Category"] == "Diagnostic"]),
    ))
    if _n_iso_avail <= 5:
        top_n_iso = _n_iso_avail
    else:
        top_n_iso = st.slider("Show top N isotopes", min_value=5, max_value=_n_iso_avail, value=min(12, _n_iso_avail), key="asset_top_n_iso")
    st.plotly_chart(build_isotope_distribution_assets(df, top_n=top_n_iso), use_container_width=True)
    st.caption(
        "Distribution of radioisotopes across therapeutic (left) and diagnostic (right) drugs. "
        "Unlike trial-level counts, this view corrects for the overrepresentation of "
        "extensively studied drugs."
    )

    st.divider()

    # ── D4→order 6: Radioisotope evolution ────────────────────────────
    st.markdown("### Evolution of radioisotope use")
    col_slider, col_pin = st.columns([3, 2])
    with col_slider:
        top_n_evo = st.slider("Show top N isotopes", min_value=4, max_value=12, value=8, key="asset_iso_evo")
    with col_pin:
        iso_opts_a = get_top_isotopes(df, top_n_evo, year_col="First_Year")
        pin_iso_a = st.selectbox(
            "Pin isotope to bottom of stack",
            ["(none)"] + iso_opts_a,
            key=f"pin_iso_asset_{top_n_evo}",
        )
    pin_to_bottom_a = None if pin_iso_a == "(none)" else pin_iso_a
    fig11a = build_isotope_evolution_assets(df, top_n=top_n_evo, mode="stack", pin_to_bottom=pin_to_bottom_a)
    if show_events:
        add_event_lines(fig11a, x_min=year_range[0], x_max=year_range[1])
    st.plotly_chart(fig11a, use_container_width=True)
    st.caption(
        "Evolution of radioisotope use among newly registered drugs by year of first trial. "
        "Illustrates the transition from legacy isotopes (Y-90, I-131) toward newer "
        "candidates (Lu-177, Ac-225). "
        "\"Pin isotope to bottom\" anchors a selected isotope at the baseline."
    )

    st.divider()

    # ── D5→order 7: Antibody format evolution ─────────────────────────
    st.markdown("### Evolution of antibody\* format use")
    col_pin_fmt, col_fmt_mode = st.columns([3, 2])
    with col_pin_fmt:
        fmt_opts = get_top_formats(df, year_col="First_Year")
        pin_fmt = st.selectbox(
            "Pin format to bottom of stack",
            ["(none)"] + fmt_opts,
            key="pin_fmt_evo",
        )
    with col_fmt_mode:
        fmt_evo_mode = st.radio(
            "View",
            ["Count", "Share (%)"],
            horizontal=True,
            key="fmt_evo_mode",
        )
    pin_to_bottom_fmt = None if pin_fmt == "(none)" else pin_fmt
    st.plotly_chart(
        build_format_evolution_assets(
            df,
            mode="pct" if fmt_evo_mode == "Share (%)" else "new",
            pin_to_bottom=pin_to_bottom_fmt,
        ),
        use_container_width=True,
    )
    with st.expander("Cumulative totals (rolling count of unique drugs per format)"):
        st.plotly_chart(build_format_evolution_assets(df, mode="cumulative"), use_container_width=True)
    st.caption(
        "Evolution of antibody format use at the drug level. "
        "Count shows new drugs per year by format; Share (%) normalizes annually for "
        "proportional comparison. \"Pin format to bottom\" anchors a selected format at "
        "the baseline. The cumulative view (expandable above) shows rolling totals over time. "
        "\* 'Antibody formats' includes non-Ab protein scaffolds "
        "(DARPins, Affibodies, Adnectins) alongside full-length IgG and antibody-derived formats."
    )

    st.divider()

    # ── D7→order 8: Clinical phase landscape ─────────────────────────
    st.markdown("### Clinical phase landscape by isotope family and format")
    st.plotly_chart(
        build_drug_phase_timeline(df_bubble),
        use_container_width=True,
    )
    st.caption(
        "Clinical progression timeline. Each dot represents one drug positioned by year of "
        "first trial (x-axis) and highest phase reached (y-axis). Color encodes isotope "
        "family; shape encodes antibody format. Vertical jitter applied to separate "
        "overlapping entries."
    )

    st.divider()

    # ── 5A: PK Compatibility ───────────────────────────────────────────
    st.markdown("### Radioisotope and scaffold format half-life compatibility")
    st.plotly_chart(build_pk_stacked_bar(df), use_container_width=True)
    st.caption(
        "How well do clinical drugs match their radioisotope half-life with "
        "the drug's expected clearance? "
        "Radioisotopes are sorted by physical half-life. "
        "Formats are colored by assumed biological half-life. "
        "No potential half-life extensions or other modifications were taken into account. "
        "Bispecific formats and pre-targeting constructs are excluded."
    )

    st.divider()

    # ── Full asset table ───────────────────────────────────────────────
    with st.expander("View full drug list", expanded=True):
        display_cols = [
            c for c in ["Antibody_Name_Norm", "Radioisotope_Norm", "Chelator_Norm",
                         "Target_Antigen_Norm", "Antibody_Format_Norm", "Category",
                         "Highest_Phase_Norm", "First_Year", "Latest_Year",
                         "N_Trials", "Asset_Owner_Norm"]
            if c in df.columns
        ]
        rename_map = {c: c.replace("_Norm", "") for c in display_cols}
        table_df = df[display_cols].sort_values("First_Year", ascending=False).rename(columns=rename_map)
        st.download_button(
            "Download table as CSV",
            data=table_df.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="rit_drugs_filtered.csv",
            mime="text/csv",
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
