"""
Constants for the RIT Clinical Landscape dashboard.
Color palettes ported from scripts/eda_utils.py (matplotlib-independent).
"""

from pathlib import Path

import yaml

# ── Regulatory event catalog ───────────────────────────────────────────────
# Loaded from dashboard/data/regulatory_events.yaml.
# Add/remove events there — no Python changes needed.
_EVENTS_FILE = Path(__file__).parent.parent / "data" / "regulatory_events.yaml"


def _load_regulatory_events() -> list[dict]:
    try:
        with open(_EVENTS_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f).get("events", [])
    except FileNotFoundError:
        return []


REGULATORY_EVENTS: list[dict] = _load_regulatory_events()

# Category colors — ColorBrewer Set2 (colorblind-safe)
CAT_COLORS = {
    "Therapeutic": "#66C2A5",
    "Diagnostic": "#FC8D62",
    "Theranostic Pair": "#8DA0CB",
}

# Phase colors — Viridis 7-stop
PHASE_ORDER = [
    "Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2",
    "Phase 2/3", "Phase 3", "Phase 4",
]
PHASE_COLORS = {
    "Early Phase 1": "#B5DE2B",
    "Phase 1":       "#6ECE58",
    "Phase 1/2":     "#35B779",
    "Phase 2":       "#1F9E89",
    "Phase 2/3":     "#26828E",
    "Phase 3":       "#31688E",
    "Phase 4":       "#440154",
}

# Sponsor colors — ColorBrewer Dark2
SPONSOR_COLORS = {
    "Academic/Hospital": "#1B9E77",
    "Industry":          "#D95F02",
    "NIH":               "#7570B3",
    "Network":           "#E7298A",
    "Government":        "#66A61E",
}

# Qualitative palette for isotopes / free-form series (20 colors to avoid repetition at max slider)
QUAL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    # Extended palette for > 10 series:
    "#E41A1C", "#4DAF4A", "#FF7F00", "#984EA3", "#00AEA8",
    "#A65628", "#F781BF", "#66A61E", "#E6AB02", "#1B7837",
]

# Year range for slider defaults
YEAR_MIN = 1985
YEAR_MAX = 2026
YEAR_DEFAULT = (1985, 2026)

# Category options
ALL_CATEGORIES = ["Therapeutic", "Diagnostic", "Theranostic Pair"]

# Antibody format family options (from Format_Family column)
ALL_FORMAT_FAMILIES = [
    "Full-length IgG", "Bispecific", "mAb Fragment",
    "Small Ab-derived scaffold", "Non-Ab protein scaffold",
]

# Isotope family options (from Isotope_Family column)
ALL_ISOTOPE_FAMILIES = ["Alpha emitter", "Beta emitter", "PET", "SPECT"]

# Pre-2000 data note shown on temporal charts
PRE2000_NOTE = (
    "Note: CT.gov registration became mandatory in 2000. "
    "Pre-2000 data reflects selective retrospective registration and is sparse."
)
