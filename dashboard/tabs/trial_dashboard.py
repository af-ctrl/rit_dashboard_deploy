"""
Tab 1: Trial-Level Analysis
Contains 6 interactive charts, all driven by the shared sidebar filters.
"""

import streamlit as st

from dashboard.utils.chart_helpers import (
    add_event_lines,
    build_alpha_emitter_adoption,
    build_format_pct_stacked,
    build_industry_sponsorship_trend,
    build_isotope_distribution,
    build_isotope_evolution,
    build_isotope_indication_heatmap,
    build_isotope_indication_heatmap_assets,
    build_top_targets,
    build_trials_by_year_category,
    build_trials_by_year_phase,
    get_top_isotopes,
)
from dashboard.utils.constants import PRE2000_NOTE
from dashboard.utils.data_loader import apply_sidebar_filters, enrich_sponsor_type_from_assets


def render(
    df_relevant: "pd.DataFrame",
    year_range: tuple,
    categories: list,
    format_families: list | None = None,
    isotope_families: list | None = None,
    phases: list | None = None,
    antigens: list | None = None,
    indication_groups: list | None = None,
    show_events: bool = False,
    bin_size: int = 5,
    df_assets: "pd.DataFrame | None" = None,
) -> None:
    """Render all trial-level charts."""
    df = apply_sidebar_filters(
        df_relevant, year_range, categories,
        format_families=format_families, isotope_families=isotope_families,
        phases=phases, antigens=antigens, indication_groups=indication_groups,
    )

    if df.empty:
        st.warning("No trials match the current filters. Try widening the year range or selecting more categories.")
        return

    n_trials = len(df)
    st.caption(f"Showing **{n_trials:,}** trials matching current filters.")

    # ── T1: Trials by year x category (order 1) ──────────────────────
    st.markdown("### Trials per year by category")
    fig1 = build_trials_by_year_category(df)
    if show_events:
        add_event_lines(fig1, x_min=year_range[0], x_max=year_range[1])
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Annual trial registrations by category (Therapeutic, Diagnostic, Theranostic Pair). "
        "Blue line indicates the proportion of trials introducing a novel target antigen. "
        "For bispecific antibodies, novelty is assessed at the level of each individual antigen. "
        "Novelty is computed relative to the active filter selection: "
        "an antigen is counted as novel if it has not appeared in any previous trial within the selection. "
        "Dashed lines represent major regulatory events."
    )
    st.caption(f"_{PRE2000_NOTE}_")

    st.divider()

    # ── T2: Trials by year x phase (order 2) ─────────────────────────
    st.markdown("### Trials per year by phase")
    fig2 = build_trials_by_year_phase(df)
    if show_events:
        add_event_lines(fig2, x_min=year_range[0], x_max=year_range[1])
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Annual trial registrations by clinical development phase. "
        "Use the phase filter to select which phases are displayed. "
        "Dashed lines represent major regulatory events."
    )

    st.divider()

    # ── T3: Target distribution (order 3) ─────────────────────────────
    st.markdown("### Target distribution")
    top_n_targets = st.slider("Show top N targets", min_value=5, max_value=25, value=15, key="top_n_targets")
    st.plotly_chart(build_top_targets(df, top_n=top_n_targets), use_container_width=True)
    st.caption(
        "Distribution of target antigens across therapeutic (left) and diagnostic (right) trials. "
        "Bispecific antibody targets are split so that each antigen is counted individually."
    )

    st.divider()

    # ── T4: Radioisotope distribution (order 4) ──────────────────────
    st.markdown("### Radioisotope distribution")
    def _iso_nunique(sub):
        return sub["Radioisotope_Norm"].dropna().str.split("; ").explode().str.strip().nunique()
    _n_iso_avail = max(5, max(
        _iso_nunique(df[df["Category"].isin(["Therapeutic", "Theranostic Pair"])]),
        _iso_nunique(df[df["Category"].isin(["Diagnostic", "Theranostic Pair"])]),
    ))
    if _n_iso_avail <= 5:
        top_n_iso = _n_iso_avail
    else:
        top_n_iso = st.slider("Show top N isotopes", min_value=5, max_value=_n_iso_avail, value=min(12, _n_iso_avail), key="top_n_iso")
    st.plotly_chart(build_isotope_distribution(df, top_n=top_n_iso), use_container_width=True)
    st.caption(
        "Distribution of radioisotopes across therapeutic (left) and diagnostic (right) trials. "
        "Trials employing multiple isotopes (e.g. theranostic pairs) contribute one count per isotope."
    )

    st.divider()

    # ── T6→order 5: Radioisotope evolution ────────────────────────────
    st.markdown("### Evolution of radioisotope use")
    col_slider, col_pin, col_mode = st.columns([3, 2, 2])
    with col_slider:
        top_n_iso_evo = st.slider("Show top N isotopes", min_value=4, max_value=12, value=8, key="top_n_iso_evo")
    with col_pin:
        iso_opts = get_top_isotopes(df, top_n_iso_evo, year_col="Start_Year")
        pin_iso_evo = st.selectbox(
            "Pin isotope to bottom of stack",
            ["(none)"] + iso_opts,
            key=f"pin_iso_evo_{top_n_iso_evo}",
        )
    with col_mode:
        iso_evo_mode = st.radio(
            "View",
            ["Count", "Share (%)"],
            horizontal=True,
            key="iso_evo_mode",
        )
    pin_to_bottom_evo = None if pin_iso_evo == "(none)" else pin_iso_evo
    st.plotly_chart(
        build_isotope_evolution(
            df, top_n=top_n_iso_evo, pin_to_bottom=pin_to_bottom_evo,
            mode="pct" if iso_evo_mode == "Share (%)" else "count",
            bin_size=bin_size,
        ),
        use_container_width=True,
    )
    st.caption(
        f"Evolution of radioisotope use across {bin_size}-year periods. "
        "Newer isotopes (e.g. Lu-177, Ac-225) have expanded relative to earlier standards "
        "(Y-90, I-131). Count shows absolute number of trials using a specific isotope; "
        "Share (%) normalizes per period. "
        "\"Pin isotope to bottom\" anchors a selected isotope at the baseline."
    )

    st.divider()

    # ── T5→order 6: Antibody format evolution ─────────────────────────
    st.markdown("### Evolution of antibody\* format use")
    fmt_pct_mode = st.radio(
        "View",
        ["Share (%)", "Count"],
        horizontal=True,
        key="fmt_pct_mode",
    )
    st.plotly_chart(
        build_format_pct_stacked(df, mode="count" if fmt_pct_mode == "Count" else "pct", bin_size=bin_size),
        use_container_width=True,
    )
    st.caption(
        f"Evolution of antibody format use across {bin_size}-year periods. "
        "Full-length IgG remains dominant, but the chart reveals the gradual emergence of "
        "alternative formats (nanobodies, bispecifics, fragments). "
        "Count shows absolute number of trials featuring a specific antibody format; "
        "Share (%) normalizes per period. "
        "\* 'Antibody formats' includes non-Ab protein scaffolds "
        "(DARPins, Affibodies, Adnectins) alongside full-length IgG and antibody-derived formats."
    )

    st.divider()

    # ── T7: Industry vs academic sponsorship trend ────────────────────
    st.markdown("### Industry vs academic sponsorship over time")
    col_view, col_basis = st.columns([3, 3])
    with col_view:
        sponsorship_view = st.radio(
            "Category",
            ["All trials", "Therapeutic", "Diagnostic"],
            horizontal=True,
            key="sponsorship_view",
        )
    with col_basis:
        sponsorship_basis = st.radio(
            "Sponsorship basis",
            ["Registry (Lead Sponsor)", "Asset ownership"],
            horizontal=True,
            key="sponsorship_basis",
            help=(
                "**Registry**: uses the Lead Sponsor field from the trial registry. "
                "**Asset ownership**: upgrades a trial to 'Industry' if the drug/asset "
                "is owned by a company, even when a hospital or academic center is listed "
                "as lead sponsor (common for investigator-initiated trials)."
            ),
        )
    _cat_view_map = {"All trials": "All", "Therapeutic": "Therapeutic", "Diagnostic": "Diagnostic"}
    _df_t7 = (
        enrich_sponsor_type_from_assets(df, df_assets)
        if sponsorship_basis == "Asset ownership" and df_assets is not None
        else df
    )
    fig_sponsor = build_industry_sponsorship_trend(
        _df_t7,
        category_view=_cat_view_map[sponsorship_view],
        mode="count",
    )
    if show_events:
        add_event_lines(fig_sponsor, x_min=year_range[0], x_max=year_range[1])
    st.plotly_chart(fig_sponsor, use_container_width=True)
    st.caption(
        "Annual trial registrations by sponsor type. "
        "Industry = commercial sponsor; Academic/Hospital = university/hospital-led IIT; "
        "NIH/Gov/Other = NIH, government agencies, cooperative networks. "
        "Dashed line shows the 5-year centred rolling average of the industry-sponsored fraction. "
        "In the Therapeutic and Diagnostic views, Theranostic Pair trials are counted toward both. "
        "Dashed vertical lines represent major regulatory events. "
        "_Registry (Lead Sponsor): based on the Lead Sponsor field in the trial registry — "
        "may undercount industry involvement for investigator-initiated trials. "
        "Asset ownership: a trial is counted as industry-sponsored if the drug/asset is "
        "commercially owned, regardless of who ran the trial._"
    )
    st.caption(f"_{PRE2000_NOTE}_")

    st.divider()

    # ── T9: Alpha emitter (exotic isotope) adoption ───────────────────
    st.markdown("### Adoption of alpha emitters over time")
    st.plotly_chart(
        build_alpha_emitter_adoption(df, mode="count", bin_size=bin_size),
        use_container_width=True,
    )
    st.caption(
        f"Adoption of alpha-emitting radioisotopes in therapeutic and theranostic trials across {bin_size}-year periods. "
        "Ac-225 dominates the alpha emitter landscape, with At-211 and Th-227 making smaller contributions. "
        "Dashed red line shows alpha emitters as a fraction of all therapeutic/theranostic trials in the same period (right axis). "
        "Nearly all alpha emitter trials are Phase 1 or Phase 1/2, reflecting their early stage of clinical development. "
        "The earliest trials (pre-2010) were predominantly academic; industry and academic sponsorship have grown in parallel since ~2015."
    )
    st.caption(f"_{PRE2000_NOTE}_")

    st.divider()

    # ── T11: Isotope × indication heatmap ────────────────────────────
    st.markdown("### Isotope × indication heatmap")
    col_t11_level, col_t11_cat, col_t11_mode = st.columns([3, 3, 3])
    with col_t11_level:
        t11_level = st.radio(
            "Level",
            ["Drug", "Trial"],
            horizontal=True,
            key="t11_level",
            help=(
                "**Trial**: each trial counts once per (isotope, indication) pair. "
                "Drugs tested in many trials dominate. "
                "**Drug**: each unique drug counts once per indication it has been studied in, "
                "regardless of how many trials it accumulated — removes inflation from "
                "heavily-tested drugs."
            ),
        )
    with col_t11_cat:
        t11_cat = st.radio(
            "Category",
            ["Therapeutic", "All", "Diagnostic"],
            horizontal=True,
            key="t11_category",
        )
    with col_t11_mode:
        t11_mode = st.radio(
            "View",
            ["Count", "% by isotope"],
            horizontal=True,
            key="t11_mode",
        )
    top_n_t11 = st.slider(
        "Show top N isotopes", min_value=5, max_value=20, value=12, key="top_n_t11",
    )
    _t11_mode_arg = "pct_iso" if t11_mode == "% by isotope" else "count"

    # Build isotope→family lookup from assets (used for alpha/beta grouping in Therapeutic view)
    _iso_family_map: dict = {}
    if df_assets is not None and "Radioisotope_Norm" in df_assets.columns and "Isotope_Family" in df_assets.columns:
        _iso_family_map = (
            df_assets[~df_assets["Radioisotope_Norm"].str.contains("; ", na=False)]
            .dropna(subset=["Radioisotope_Norm", "Isotope_Family"])
            .set_index("Radioisotope_Norm")["Isotope_Family"]
            .to_dict()
        )

    if t11_level == "Drug" and df_assets is not None:
        st.plotly_chart(
            build_isotope_indication_heatmap_assets(
                df_assets,
                top_n_iso=top_n_t11,
                mode=_t11_mode_arg,
                category=t11_cat,
                iso_family_map=_iso_family_map,
            ),
            use_container_width=True,
        )
        _drug_thera_note = (
            " Isotopes are grouped into α (alpha) and β (beta/other) emitters separated by a dashed line. "
            "I-131 is classified as β despite a minor Auger component. "
            "In-111 appears only as a Y-90 dosimetry agent, not a therapeutic isotope."
            if t11_cat == "Therapeutic" else ""
        )
        st.caption(
            "**Drug-level view**: each unique drug is counted once per indication it has been "
            "studied in across all its trials — removes inflation from heavily-tested drugs "
            "(e.g. Y-90-ibritumomab with dozens of lymphoma trials contributes 1, not 80+). "
            "Sidebar year/phase filters do not apply at the drug level. "
            "% by isotope: fraction of each isotope's indication assignments in that disease group."
            + _drug_thera_note + " "
            "Indication columns follow the order in `config/normalization/indication_group.yaml`."
        )
    else:
        st.plotly_chart(
            build_isotope_indication_heatmap(
                df,
                top_n_iso=top_n_t11,
                mode=_t11_mode_arg,
                category=t11_cat,
                iso_family_map=_iso_family_map,
            ),
            use_container_width=True,
        )
        _thera_note = (
            " In Therapeutic mode, isotopes are grouped into α (alpha) and β (beta/other) emitters "
            "separated by a dashed line. I-131 is classified as β despite a minor Auger component. "
            "In-111 appears in 4 Therapeutic trials as a dosimetry agent paired with Y-90, not as a therapeutic isotope."
            if t11_cat == "Therapeutic" else ""
        )
        st.caption(
            "**Trial-level view**: each trial contributes one count per unique (isotope, indication) "
            "combination it covers. Drugs with many trials are weighted proportionally. "
            "% by isotope: what fraction of each isotope's indication assignments fall in that disease group. "
            "Isotopes ranked by total trial count; most common at top." + _thera_note + " "
            "Indication columns follow the order in `config/normalization/indication_group.yaml`. "
            "_Theranostic Pair trials (e.g. Ac-225 + Ga-68) are split by isotope type: "
            "the therapeutic isotope counts in the Therapeutic view, the diagnostic isotope in the Diagnostic view._"
        )

    # ── Full trial table ───────────────────────────────────────────────
    st.divider()
    with st.expander("View full trial list", expanded=True):
        display_cols = [
            c for c in ["NCT_ID", "Antibody_Name_Norm", "Radioisotope_Norm",
                         "Target_Antigen_Norm", "Phase_Norm", "Category",
                         "Start_Year", "Lead_Sponsor", "Status"]
            if c in df.columns
        ]
        rename_map = {c: c.replace("_Norm", "") for c in display_cols}
        table_df = df[display_cols].sort_values("Start_Year", ascending=False).rename(columns=rename_map)
        st.download_button(
            "Download table as CSV",
            data=table_df.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="rit_trials_filtered.csv",
            mime="text/csv",
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
