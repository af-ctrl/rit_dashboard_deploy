"""
RIT Clinical Landscape — Interactive Dashboard
Entry point: streamlit run dashboard/app.py

Two tabs:
  Tab 1 — Trial-Level Analysis (6 charts)
  Tab 2 — Asset-Level Analysis (6 charts)

Run from project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="RIT Clinical Landscape",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports after page config
from dashboard.tabs import about_dashboard, asset_dashboard, explorer_dashboard, trial_dashboard
from dashboard.utils.constants import (
    ALL_CATEGORIES, ALL_FORMAT_FAMILIES, ALL_ISOTOPE_FAMILIES,
    PHASE_ORDER, YEAR_DEFAULT, YEAR_MAX, YEAR_MIN,
)
from dashboard.utils.data_loader import (
    DISC_MODE_ALL, DISC_MODE_CONFIRMED, DISC_MODE_SUSPECTED,
    apply_discontinued_filter,
    ctgov_only, filter_assets_ctgov, get_all_indication_groups, load_data, relevant,
)


# ── Data loading ───────────────────────────────────────────────────────
@st.cache_data
def get_data():
    master, assets = load_data()
    rel = relevant(master)
    if "Source_Registry" in master.columns:
        src = master["Source_Registry"]
        n_ctgov_total = int((src == "ClinicalTrials.gov").sum())
        n_gap_total = int((src != "ClinicalTrials.gov").sum())
        n_anzctr = int((src == "ANZCTR").sum())
        n_non_ctgov_registries = int(src[src != "ClinicalTrials.gov"].nunique())
    else:
        n_ctgov_total = len(master)
        n_gap_total = 0
        n_anzctr = 0
        n_non_ctgov_registries = 0
    n_fp = len(master) - len(rel)
    return rel, assets, n_ctgov_total, n_gap_total, n_fp, n_anzctr, n_non_ctgov_registries


df_relevant, df_assets, _n_ctgov_total, _n_gap_total, _n_fp, _n_anzctr, _n_non_ctgov_registries = get_data()

# ── Sidebar CSS ─────────────────────────────────────────────────────────
st.markdown(
    """<style>
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
}
</style>""",
    unsafe_allow_html=True,
)

# Antigen list built once from all relevant trials (stable across registry-filter changes)
_all_antigens = sorted({
    a.strip()
    for val in df_relevant["Target_Antigen_Norm"].dropna()
    for a in str(val).split("; ")
    if a.strip()
})

# Indication groups: ordered list from YAML (stable across registry-filter changes)
_all_indication_groups = get_all_indication_groups()

# ── Sidebar filters ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    year_range = st.slider(
        "Year range",
        min_value=YEAR_MIN,
        max_value=YEAR_MAX,
        value=YEAR_DEFAULT,
        step=1,
        help="Filter trials by start year (CT.gov data is sparse before 2000).",
    )

    show_events = st.checkbox(
        "Show regulatory milestones",
        value=True,
        help=(
            "Draw vertical lines on year-axis charts for key events: "
            "CT.gov registration mandate (2000), "
            "Zevalin FDA approval (2002), Bexxar FDA approval (2003), "
            "Zevalin EMA approval (2004), Bexxar withdrawal (2014), "
            "Zevalin EMA lapse (2024). "
            "Edit dashboard/data/regulatory_events.yaml to add or remove events."
        ),
    )

    categories = st.multiselect(
        "Category",
        options=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
        help="Therapeutic = treatment trials; Diagnostic = immunoPET/SPECT; Theranostic Pair = trial using the same antibody scaffold labeled with both a therapeutic and a diagnostic isotope.",
    )

    phases = st.multiselect(
        "Phase",
        options=PHASE_ORDER,
        default=PHASE_ORDER,
        key="phases",
        help="Filter by clinical development phase. Select a subset to focus on specific phases.",
    )

    antigens = st.multiselect(
        "Target antigen",
        options=_all_antigens,
        default=[],
        key="antigen_filter",
        help="Filter to specific target antigens. Leave empty to include all.",
    )

    indications = st.multiselect(
        "Indication",
        options=_all_indication_groups,
        default=[],
        key="indication_filter",
        help=(
            "Filter to trials/drugs with at least one trial in the selected cancer indication(s). "
            "Leave empty to include all. "
            "Groups are keyword-matched against CT.gov Conditions — see About & Methods for details."
        ),
    )

    format_families = st.multiselect(
        "Antibody format",
        options=ALL_FORMAT_FAMILIES,
        default=ALL_FORMAT_FAMILIES,
        help=(
            "Full-length IgG: standard mAbs · "
            "Bispecific: bispecific antibodies · "
            "mAb Fragment: Fab, F(ab')2, scFv, one-armed · "
            "Small Ab-derived scaffold: Nanobody, Minibody, SIP, Diabody · "
            "Non-Ab protein scaffold: Affibody, Adnectin, etc."
        ),
    )

    isotope_families = st.multiselect(
        "Isotope type",
        options=ALL_ISOTOPE_FAMILIES,
        default=ALL_ISOTOPE_FAMILIES,
        help=(
            "Alpha emitter: Ac-225, At-211, Th-227, Pb-212 · "
            "Beta emitter: Y-90, I-131, Lu-177 · "
            "PET: Zr-89, Cu-64, I-124, Ga-68 · "
            "SPECT: In-111, Tc-99m"
        ),
    )

    # Treat full selection as no filter (avoids excluding rows with null Family/Phase columns)
    _format_filter = format_families if set(format_families) != set(ALL_FORMAT_FAMILIES) else None
    _isotope_filter = isotope_families if set(isotope_families) != set(ALL_ISOTOPE_FAMILIES) else None
    _phase_filter = phases if set(phases) != set(PHASE_ORDER) else None
    _antigen_filter = antigens if antigens else None
    _indication_filter = indications if indications else None

    bin_size = st.radio(
        "Period bin size",
        options=[3, 5, 10],
        index=1,
        horizontal=True,
        format_func=lambda x: f"{x}y",
        help="Width of time bins used in period charts (format evolution, isotope evolution, Phase 1 drugs).",
    )

    disc_filter = st.radio(
        "Drug program status",
        options=[DISC_MODE_ALL, DISC_MODE_CONFIRMED, DISC_MODE_SUSPECTED],
        index=0,
        help=(
            "**All programs**: show every drug regardless of development status. "
            "**Hide confirmed discontinued**: removes 23 drugs whose programs were verified as "
            "discontinued in the ownership review (Module 3). "
            "**Hide confirmed & suspected discontinued**: also removes drugs with no verified "
            f"active status and no clinical trial activity since {2020} "
            "(heuristic — see About & Methods)."
        ),
    )

    st.divider()

    registry_filter = st.radio(
        "Data source",
        options=["All registries", "ClinicalTrials.gov only"],
        index=0,
        help=(
            f"**All registries**: includes ClinicalTrials.gov ({_n_ctgov_total:,} trials) plus "
            f"{_n_gap_total} gap trials from EUCTR, ANZCTR, NL-OMON, and other ICTRP registries. "
            "**ClinicalTrials.gov only**: restricts to the main registry for a "
            "cleaner, directly comparable view."
        ),
    )



# ── Apply registry filter ───────────────────────────────────────────────
_ctgov_mode = registry_filter == "ClinicalTrials.gov only"
if _ctgov_mode:
    df_trials = ctgov_only(df_relevant)
    df_drugs = filter_assets_ctgov(df_assets, set(df_trials["NCT_ID"]))
else:
    df_trials = df_relevant
    df_drugs = df_assets

# ── Apply discontinued filter ────────────────────────────────────────────
# Filter assets, then propagate to trials: keep a trial if at least one
# non-discontinued asset links to it (or if it has no asset link at all).
if disc_filter != DISC_MODE_ALL:
    df_drugs = apply_discontinued_filter(df_drugs, disc_filter)
    _kept_ncts: set[str] = set()
    for ids_str in df_drugs["Trial_NCT_IDs"].dropna():
        for nid in str(ids_str).split("; "):
            _kept_ncts.add(nid.strip())
    df_trials = df_trials[df_trials["NCT_ID"].isin(_kept_ncts)].copy()

# ── Header (after filters so counts reflect the registry selection) ─────
st.title("The Clinical Development Landscape of Antibody-Based Radiopharmaceuticals")
_source_label = (
    "ClinicalTrials.gov" if _ctgov_mode
    else f"ClinicalTrials.gov + {_n_non_ctgov_registries} additional registries"
)
st.markdown(
    f"Data source: {_source_label} "
    f"· {len(df_trials):,} relevant trials · {len(df_drugs):,} unique drugs"
)
st.markdown("*All charts are interactive — hover for details, click legend items to toggle.*")

# ── Tab navigation ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Trial-Level Analysis", "Drug-Level Analysis", "Explorer", "About & Methods"])

with tab1:
    trial_dashboard.render(
        df_trials, year_range=year_range, categories=categories,
        format_families=_format_filter, isotope_families=_isotope_filter,
        phases=_phase_filter, antigens=_antigen_filter,
        indication_groups=_indication_filter,
        show_events=show_events, bin_size=bin_size,
        df_assets=df_drugs,
    )

with tab2:
    asset_dashboard.render(
        df_drugs, df_trials=df_trials,
        year_range=year_range, categories=categories,
        format_families=_format_filter, isotope_families=_isotope_filter,
        phases=_phase_filter, antigens=_antigen_filter,
        indication_groups=_indication_filter,
        show_events=show_events, bin_size=bin_size,
    )

with tab3:
    explorer_dashboard.render(
        df_trials, df_drugs, year_range=year_range, categories=categories,
        format_families=_format_filter, isotope_families=_isotope_filter,
        phases=_phase_filter, antigens=_antigen_filter,
        indication_groups=_indication_filter,
        bin_size=bin_size,
    )

with tab4:
    about_dashboard.render(
        n_trials=len(df_relevant),
        n_drugs=len(df_assets),
        n_ctgov_total=_n_ctgov_total,
        n_gap_total=_n_gap_total,
        n_fp=_n_fp,
        n_anzctr=_n_anzctr,
        n_non_ctgov_registries=_n_non_ctgov_registries,
    )
