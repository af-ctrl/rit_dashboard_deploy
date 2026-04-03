"""
Tab 4: About & Methods
Describes data sources, processing pipeline, key definitions, limitations,
and a public-data disclaimer.
"""

import streamlit as st


def render(
    n_trials: int,
    n_drugs: int,
    n_ctgov_total: int,
    n_gap_total: int,
    n_fp: int,
    n_anzctr: int,
    n_non_ctgov_registries: int,
) -> None:
    """Render the About & Methods tab."""

    st.markdown("## Scope")
    st.markdown(
        "This dashboard maps the clinical development landscape of **antibody-based "
        "radiopharmaceuticals** (including non-antibody protein scaffolds), both therapeutic "
        "radioimmunotherapy (RIT) and diagnostic immunoimaging (immunoPET, immunoSPECT) drugs. "
        "**Small-molecule radioligands** (e.g. PSMA-617/Pluvicto, DOTATATE/Lutathera) and "
        "peptide-based radiopharmaceuticals are excluded."
    )

    st.markdown("## Data sources")
    st.markdown(
        "All data in this dashboard is derived exclusively from **publicly available sources**:"
    )
    st.markdown(
        f"- **ClinicalTrials.gov** ({n_ctgov_total:,} trials): the primary data source, "
        "queried via the CT.gov API v2.\n"
        f"- **WHO ICTRP and regional registries** ({n_gap_total:,} additional trials from "
        f"{n_non_ctgov_registries} registries): gap trials not cross-listed on CT.gov, "
        "identified through a systematic cross-registry search of the WHO International "
        "Clinical Trials Registry Platform.\n"
        "- **Open-access publications** (PubMed, PubMed Central): used to enrich trial "
        "records with missing drug details.\n"
        "- **Public web pages**: company pipelines, press releases, and regulatory filings "
        "used to verify asset ownership and fill remaining data gaps.\n\n"
        "No proprietary databases, paywalled publications, or non-public clinical data were used."
    )

    st.markdown("## Data processing pipeline")
    st.markdown(
        f"1. **Acquisition**: {n_ctgov_total + n_gap_total:,} unique trial records were retrieved "
        f"from ClinicalTrials.gov and {n_non_ctgov_registries} WHO ICTRP-linked registries.\n"
        "2. **Classification**: Each trial was classified as Therapeutic, Diagnostic, "
        "Theranostic Pair, or false positive using a combination of rule-based "
        "pre-classification (known drugs, known false-positive patterns) and LLM-assisted "
        "extraction (Claude, Anthropic). "
        f"Of {n_ctgov_total + n_gap_total:,} candidate trials, **{n_trials:,} were identified "
        f"as relevant** and {n_fp:,} were excluded as false positives.\n"
        "3. **Extraction**: Several structured fields were extracted per trial (antibody name, "
        "target antigen, radioisotope, chelator, antibody format, category, phase, sponsor, "
        "asset owner) via LLM, each with a confidence rating. Literature and web enrichment "
        "was used to complement missing information.\n"
        "4. **Normalization**: Raw field values were collapsed to standardized names (e.g. "
        "\"Ibritumomab tiuxetan\" and \"Zevalin\" to a single entry), then grouped into "
        "biological families (e.g. isotope families: Alpha emitter, Beta emitter, PET, SPECT). "
        f"Result: {n_drugs:,} unique drugs (antibody + isotope + chelator combinations).\n"
        "5. **Owner verification**: Industry asset ownership, inferred from trial text during "
        "the extraction step, was systematically verified against current company pipelines, "
        "press releases, and regulatory filings.\n"
        "6. **Validation**: Multi-stage validation protocol including internal cross-validation, "
        "cross-model verification, review article comparison, WHO INN stem checks, "
        "and cross-registry gap analysis against WHO ICTRP."
    )

    st.markdown("## Key definitions")
    st.markdown(
        "- **Trial**: One registered clinical study record (identified by registry ID, "
        "e.g. NCT number).\n"
        "- **Drug**: A unique combination of antibody + radioisotope + chelator. One drug may "
        "appear in multiple trials. Radiolabeled versions of common antibodies are collapsed "
        "into a single drug (e.g. multiple Zr-89-labelled versions of trastuzumab), even if "
        "different trials are conducted by different sponsors.\n"
        "- **Category**: *Therapeutic* = treatment intent (RIT); *Diagnostic* = imaging intent "
        "(immunoPET/SPECT); *Theranostic Pair* = the same antibody scaffold labelled with a "
        "therapeutic isotope for treatment and a diagnostic isotope for imaging. Trials where "
        "the imaging component is a conventional small-molecule or peptide-based radiotracer "
        "are classified as *Therapeutic* only, as the diagnostic component falls outside the "
        "scope of this dataset.\n"
        "- **Antibody format families**:\n"
        "  - *Full-length IgG*: Standard full-length monoclonal antibodies.\n"
        "  - *Bispecific*: Antibodies engineered to bind two different antigens simultaneously.\n"
        "  - *mAb Fragment*: Antibody-derived fragments retaining the binding domain but lacking "
        "the Fc region; includes Fab, F(ab')₂, scFv, and one-armed constructs.\n"
        "  - *Small Ab-derived scaffold*: Smaller antibody-derived formats engineered for faster "
        "clearance or improved tissue penetration; includes Nanobody (VHH), Minibody, SIP "
        "(small immunoprotein), and Diabody.\n"
        "  - *Non-Ab protein scaffold*: Protein binders derived from non-immunoglobulin "
        "scaffolds; includes DARPin, Affibody, Adnectin, and Cystatin."
    )

    st.markdown("## Sidebar filters")
    st.markdown(
        "**Indication filter**: filters trials and drugs to those with at least one trial "
        "in the selected cancer indication(s). Approximately 14% of trials carry only generic "
        "condition labels such as \"Solid Tumor\" or \"Cancer\" that do not match any specific "
        "group. At the drug level, a drug is shown if *any* of its trials matches the "
        "selected indication.\n\n"
        "**Drug program status**: three options control which drug programs are shown:\n"
        "- *All programs* (default): no filtering by status.\n"
        "- *Hide confirmed discontinued*: removes **23 drugs** whose development programs "
        "were verified as discontinued during the ownership review. Verification used company "
        "pipelines, press releases, and regulatory filings as of March 2026.\n"
        "- *Hide confirmed & suspected discontinued*: additionally removes drugs that "
        "**have not received a verified Active status** and show **no clinical trial activity "
        "since 2020** (heuristic threshold). This removes an additional ~106 drugs. "
        "Note that no recent trial activity does not prove discontinuation. "
        "This option is provided for exploratory use only; results should be interpreted "
        "with caution."
    )

    st.markdown("## Limitations")
    st.markdown(
        "- Pre-2000 trial data is sparse. ClinicalTrials.gov launched in February 2000 and "
        "retrospective registration of earlier trials was incomplete.\n"
        "- Trial start dates for non-CT.gov registries are approximated from registration dates.\n"
        "- Asset ownership is LLM-inferred from trial text and systematically web-verified for "
        "industry-owned assets; it may not reflect current licensing arrangements."
    )

    st.divider()

    st.markdown("## Disclaimer")
    st.markdown(
        "This analysis relies exclusively on publicly available information. It is provided "
        "for informational and research purposes. No proprietary or confidential data was used. "
        "The author makes no claim of completeness. While systematic multi-registry search "
        "and validation were applied, individual trials or drugs may be missing."
    )
    st.markdown(
        "You are welcome to share, cite, or build upon this work. "
        "If you do, a mention of the source is appreciated. "
        "LinkedIn: [Andreas Franz](https://www.linkedin.com/in/andreas-franz-berlin/)"
    )
    st.caption(
        "ClinicalTrials.gov data accessed: February 2026. "
        "WHO ICTRP data accessed: March 2026. "
        "Dashboard built with Streamlit and Plotly."
    )
