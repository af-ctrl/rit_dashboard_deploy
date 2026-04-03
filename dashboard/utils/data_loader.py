"""
Data loading and filtering utilities for the RIT dashboard.
Logic ported from scripts/eda_utils.py (no matplotlib dependency).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_CSV = PROJECT_ROOT / "data" / "analysis" / "rit_master.csv"
ASSETS_CSV = PROJECT_ROOT / "data" / "analysis" / "assets.csv"
TRIAL_NOTES_YAML = PROJECT_ROOT / "dashboard" / "data" / "trial_notes.yaml"
INDICATION_GROUP_YAML = PROJECT_ROOT / "config" / "normalization" / "indication_group.yaml"
TARGET_FAMILY_YAML = PROJECT_ROOT / "config" / "normalization" / "target_family.yaml"

# Year threshold for "suspected discontinued" heuristic
SUSPECTED_DISC_YEAR_CUTOFF = 2020


@st.cache_data
def load_antigen_family_lookup() -> dict[str, str]:
    """Return {antigen_norm: family} for single-antigen entries in target_family.yaml.

    Compound entries (containing '; ') are excluded so that exploded antigen rows
    get the correct per-antigen family instead of inheriting the compound entry's family.
    """
    if not TARGET_FAMILY_YAML.exists():
        return {}
    with open(TARGET_FAMILY_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    lookup: dict[str, str] = {}
    for family, members in raw.items():
        if members:
            for member in members:
                if "; " not in member:
                    lookup[member] = family
    return lookup


def _load_trial_notes() -> dict:
    """Load manual trial annotations from YAML. Returns {nct_id: text} dict."""
    if not TRIAL_NOTES_YAML.exists():
        return {}
    with open(TRIAL_NOTES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = (data or {}).get("notes", {})
    return {nct: entry["text"].strip() for nct, entry in raw.items() if "text" in entry}


@st.cache_data
def _load_indication_keywords() -> list[tuple[str, list[str]]]:
    """Load indication group → keyword list from YAML. Returns ordered list of (group, keywords)."""
    if not INDICATION_GROUP_YAML.exists():
        return []
    with open(INDICATION_GROUP_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return [(group, [kw.lower() for kw in keywords]) for group, keywords in raw.items()]


def _classify_indication(conditions_val: str | None, keyword_groups: list[tuple[str, list[str]]]) -> str | None:
    """Map a raw Conditions string to semicolon-joined indication group(s)."""
    if pd.isna(conditions_val) or not str(conditions_val).strip():
        return None
    val_lower = str(conditions_val).lower()
    matched = []
    for group, keywords in keyword_groups:
        if any(kw in val_lower for kw in keywords):
            matched.append(group)
    return "; ".join(matched) if matched else None


def get_all_indication_groups() -> list[str]:
    """Return the ordered list of all indication group names from YAML."""
    kw_groups = _load_indication_keywords()
    return [g for g, _ in kw_groups]


# ── Data loading ───────────────────────────────────────────────────────

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache both datasets. Returns (master_df, assets_df)."""
    master = pd.read_csv(MASTER_CSV, sep=";", encoding="utf-8-sig")
    assets = pd.read_csv(ASSETS_CSV, sep=";", encoding="utf-8-sig")

    # Ensure numeric dtypes — same fix as eda_utils.load_master()
    if "Start_Year" in master.columns:
        master["Start_Year"] = pd.to_numeric(master["Start_Year"], errors="coerce")

    if "Enrollment" in master.columns:
        master["Enrollment"] = pd.to_numeric(master["Enrollment"], errors="coerce")

    for col in ("First_Year", "Latest_Year", "N_Trials"):
        if col in assets.columns:
            assets[col] = pd.to_numeric(assets[col], errors="coerce")

    # Display rename: keep raw CSV value "Theranostic" but show "Theranostic Pair" in UI
    # (only trials with two distinct antibody-based isotopes qualify; raw value unchanged in CSV)
    master["Category"] = master["Category"].replace("Theranostic", "Theranostic Pair")
    assets["Category"] = assets["Category"].replace("Theranostic", "Theranostic Pair")

    # Apply manual trial annotations as Dashboard_Note column
    _notes = _load_trial_notes()
    if _notes:
        if "NCT_ID" in master.columns:
            master["Dashboard_Note"] = master["NCT_ID"].map(_notes)
        if "Trial_NCT_IDs" in assets.columns:
            def _asset_note(ids_str):
                if pd.isna(ids_str):
                    return None
                notes = [_notes[nid] for nid in str(ids_str).split("; ") if nid.strip() in _notes]
                return " | ".join(notes) if notes else None
            assets["Dashboard_Note"] = assets["Trial_NCT_IDs"].map(_asset_note)

    # ── Indication group classification ────────────────────────────
    # Compute trial-level Indication_Group from raw Conditions field
    _kw_groups = _load_indication_keywords()
    if _kw_groups and "Conditions" in master.columns:
        master["Indication_Group"] = master["Conditions"].apply(
            lambda v: _classify_indication(v, _kw_groups)
        )
        # Asset-level: union of indication groups across all linked trials
        _nct_to_groups = master.set_index("NCT_ID")["Indication_Group"].to_dict()
        def _asset_indication(ids_str):
            if pd.isna(ids_str):
                return None
            all_groups: list[str] = []
            seen: set[str] = set()
            for nid in str(ids_str).split("; "):
                grp_str = _nct_to_groups.get(nid.strip())
                if grp_str and pd.notna(grp_str):
                    for g in str(grp_str).split("; "):
                        g = g.strip()
                        if g and g not in seen:
                            all_groups.append(g)
                            seen.add(g)
            return "; ".join(all_groups) if all_groups else None
        assets["Indication_Group"] = assets["Trial_NCT_IDs"].apply(_asset_indication)

    return master, assets


# ── Filters ────────────────────────────────────────────────────────────

def relevant(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant trials: RIT + Diagnostic + Theranostic Pair."""
    return df[
        (df["Is_RIT"] == "Yes") | (df["Category"].isin(["Diagnostic", "Theranostic Pair"]))
    ].copy()


def ctgov_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ClinicalTrials.gov-only trials (excludes EUCTR, ANZCTR, etc.)."""
    if "Source_Registry" not in df.columns:
        return df
    return df[df["Source_Registry"] == "ClinicalTrials.gov"].copy()


def filter_assets_ctgov(assets: pd.DataFrame, ctgov_ids: set) -> pd.DataFrame:
    """Keep assets that have at least one ClinicalTrials.gov trial (NCT ID)."""
    def _has_ctgov(ids_str: str) -> bool:
        if pd.isna(ids_str):
            return True
        return any(i.strip() in ctgov_ids for i in str(ids_str).split("; "))
    return assets[assets["Trial_NCT_IDs"].apply(_has_ctgov)].copy()


DISC_MODE_ALL = "All programs"
DISC_MODE_CONFIRMED = "Hide confirmed discontinued"
DISC_MODE_SUSPECTED = "Hide confirmed & suspected discontinued"


def apply_discontinued_filter(assets: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter assets by program status.

    Args:
        assets: assets DataFrame (must have Program_Status, Latest_Year columns).
        mode: one of DISC_MODE_* constants.
    """
    if mode == DISC_MODE_ALL:
        return assets
    # Always remove confirmed discontinued
    mask = ~(assets["Program_Status"].fillna("") == "Discontinued")
    if mode == DISC_MODE_SUSPECTED:
        # Also remove suspected inactive: no verified status + no trial activity since cutoff
        suspected = (
            assets["Program_Status"].isna()
            & (assets["Latest_Year"].fillna(9999) <= SUSPECTED_DISC_YEAR_CUTOFF)
        )
        mask = mask & ~suspected
    return assets[mask].copy()


_ASSET_OWNER_TYPES_YAML = PROJECT_ROOT / "dashboard" / "data" / "asset_owner_types.yaml"


def enrich_sponsor_type_from_assets(
    df_trials: pd.DataFrame,
    df_assets: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of df_trials where Sponsor_Type is overridden to 'Industry'
    for any trial linked to a company-owned (or mixed-ownership) asset.

    Uses asset_owner_types.yaml to classify owners. Trials already marked
    'Industry' via Lead Sponsor are unchanged. A new column '_SponsorBasis'
    is added: 'registry' = Lead Sponsor field; 'asset' = upgraded via asset ownership.

    Args:
        df_trials: Relevant trials DataFrame.
        df_assets: Assets DataFrame with Asset_Owner_Norm and Trial_NCT_IDs columns.
    """
    # Load owner type classifications
    try:
        with open(_ASSET_OWNER_TYPES_YAML, encoding="utf-8") as f:
            owner_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return df_trials.copy()
    owner_types = owner_data.get("owners", {})
    company_owners = {k for k, v in owner_types.items() if v.get("type") in ("company", "mixed")}

    # Build set of NCT IDs linked to at least one company-owned asset
    nct_company: set[str] = set()
    for _, row in df_assets.iterrows():
        if row.get("Asset_Owner_Norm") in company_owners and pd.notna(row.get("Trial_NCT_IDs")):
            for nid in str(row["Trial_NCT_IDs"]).split("; "):
                nct_company.add(nid.strip())

    df_out = df_trials.copy()
    df_out["_SponsorBasis"] = "registry"
    upgraded = (
        df_out["NCT_ID"].isin(nct_company)
        & (df_out["Sponsor_Type"] != "Industry")
    )
    df_out.loc[upgraded, "Sponsor_Type"] = "Industry"
    df_out.loc[upgraded, "_SponsorBasis"] = "asset"
    return df_out


def rit_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to confirmed RIT trials (Is_RIT=Yes, includes Theranostic)."""
    return df[df["Is_RIT"] == "Yes"].copy()


def diag_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnostic trials."""
    return df[df["Category"] == "Diagnostic"].copy()


def apply_sidebar_filters(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    categories: list[str],
    format_families: list[str] | None = None,
    isotope_families: list[str] | None = None,
    phases: list[str] | None = None,
    antigens: list[str] | None = None,
    indication_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Apply year range, category, format, isotope, phase, antigen, and indication filters."""
    out = df.copy()
    if "Start_Year" in out.columns:
        yr = out["Start_Year"]
        out = out[yr.isna() | ((yr >= year_range[0]) & (yr <= year_range[1]))]
    if categories and "Category" in out.columns:
        out = out[out["Category"].isin(categories)]
    if format_families is not None and "Format_Family" in out.columns:
        out = out[out["Format_Family"].isin(format_families)]
    if isotope_families is not None and "Isotope_Family" in out.columns:
        out = out[out["Isotope_Family"].isin(isotope_families)]
    if phases is not None and "Phase_Norm" in out.columns:
        out = out[out["Phase_Norm"].isin(phases)]
    if antigens is not None and "Target_Antigen_Norm" in out.columns:
        antigen_set = set(antigens)
        out = out[out["Target_Antigen_Norm"].apply(
            lambda v: not pd.isna(v) and bool({s.strip() for s in str(v).split("; ")} & antigen_set)
        )]
    if indication_groups is not None and "Indication_Group" in out.columns:
        grp_set = set(indication_groups)
        out = out[out["Indication_Group"].apply(
            lambda v: not pd.isna(v) and bool({s.strip() for s in str(v).split("; ")} & grp_set)
        )]
    return out


# ── Utility helpers ────────────────────────────────────────────────────

def explode_multival(series: pd.Series, sep: str = "; ") -> pd.Series:
    """Split multi-value entries (e.g. 'Y-90; In-111') and explode into rows."""
    return series.dropna().str.split(sep).explode().str.strip()


def top_values(series: pd.Series, n: int = None, min_count: int = None) -> pd.Series:
    """Value counts filtered to top N or minimum threshold."""
    vc = series.dropna().value_counts()
    if min_count is not None:
        vc = vc[vc >= min_count]
    if n is not None:
        vc = vc.head(n)
    return vc


def year_bins(df: pd.DataFrame, col: str = "Start_Year", bin_size: int = 5) -> pd.Series:
    """Bin years into periods using backward alignment (last bin ends at current year).

    Backward alignment ensures the most recent bin contains only years that can have data,
    avoiding an artificially sparse last bucket. The oldest bin may be slightly incomplete.
    E.g. with bin_size=5 and current year 2026: ..., 2017-2021, 2022-2026.
    """
    years = df[col].dropna()
    if years.empty:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    data_min = int(years.min())
    # upper is exclusive for pd.cut (right=False), so bins end at current_year
    upper = datetime.now().year + 1
    # Generate edges backward from upper, far enough to capture data_min
    edges = sorted(range(upper, data_min - bin_size, -bin_size))
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)]
    return pd.cut(df[col], bins=edges, labels=labels, right=False)


def build_hover_text(subset: pd.DataFrame, max_items: int = 6) -> str:
    """Build a compact hover string listing trial IDs, drug names, and radioisotopes.

    Shows up to max_items rows, then '... and N more'.
    """
    if subset.empty:
        return "(no trials)"

    cols = [c for c in ["NCT_ID", "Antibody_Name_Norm", "Radioisotope_Norm"] if c in subset.columns]
    all_rows = subset[cols].drop_duplicates()
    n_total = len(all_rows)
    rows = all_rows.head(max_items)

    lines = ["<b>NCT · Antibody* Name · Isotope</b>"]
    for _, row in rows.iterrows():
        nct = row.get("NCT_ID", "")
        drug = row.get("Antibody_Name_Norm", "")
        iso = row.get("Radioisotope_Norm", "")
        drug_str = str(drug) if pd.notna(drug) and str(drug).strip() else "—"
        iso_str = str(iso) if pd.notna(iso) and str(iso).strip() else "—"
        lines.append(f"{nct} · {drug_str} · {iso_str}")

    remaining = n_total - max_items
    if remaining > 0:
        lines.append(f"<i>... and {remaining} more</i>")

    # Append manual notes for any trial in the full subset (not just the displayed rows)
    if "Dashboard_Note" in subset.columns and "NCT_ID" in subset.columns:
        noted = (
            subset[subset["Dashboard_Note"].notna()][["NCT_ID", "Dashboard_Note"]]
            .drop_duplicates()
        )
        for _, nr in noted.iterrows():
            lines.append(f"<br><i>⚠ {nr['NCT_ID']}: {nr['Dashboard_Note']}</i>")

    return "<br>".join(lines)


def build_asset_hover_text(subset: pd.DataFrame, max_items: int = 8) -> str:
    """Hover text for asset charts: Target · Antibody (Isotope).

    Args:
        subset: Slice of assets DataFrame.
        max_items: Maximum assets to list before '... and N more'.
    """
    if subset.empty:
        return "(no assets)"

    key_cols = [c for c in ["Target_Antigen_Norm", "Antibody_Name_Norm", "Radioisotope_Norm"]
                if c in subset.columns]
    all_rows = subset[key_cols].drop_duplicates()
    n_total = len(all_rows)
    rows = all_rows.head(max_items)

    lines = ["<b>Target · Drug · Isotope</b>"]
    for _, row in rows.iterrows():
        target = row.get("Target_Antigen_Norm", "")
        drug = row.get("Antibody_Name_Norm", "")
        iso = row.get("Radioisotope_Norm", "")

        target_str = str(target) if pd.notna(target) and str(target).strip() else ""
        drug_str = str(drug) if pd.notna(drug) and str(drug).strip() else ""
        iso_str = f"({iso})" if pd.notna(iso) and str(iso).strip() else ""

        parts = [p for p in [target_str, drug_str, iso_str] if p]
        lines.append(" · ".join(parts) if parts else "(unknown)")

    remaining = n_total - max_items
    if remaining > 0:
        lines.append(f"<i>... and {remaining} more</i>")

    return "<br>".join(lines)
