from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Compound Profile")
page_title("Compound Profile")

compounds = data.search_compounds("", limit=200)
if not compounds:
    st.info("No compounds are indexed.")
    st.stop()

selected_id = st.session_state.get("selected_compound_id")
compound_ids = [compound["compound_id"] for compound in compounds]
if selected_id not in compound_ids:
    selected_id = compound_ids[0]

selected_label = {
    compound["compound_id"]: f"{compound['preferred_name']} ({compound.get('pubchem_cid') or 'no PubChem CID'})"
    for compound in compounds
}
selected_id = st.selectbox(
    "Compound",
    compound_ids,
    index=compound_ids.index(selected_id),
    format_func=lambda value: selected_label[value],
)
st.session_state["selected_compound_id"] = selected_id
compound = data.get_compound(selected_id)

if not compound:
    st.error("Selected compound was not found.")
    st.stop()

st.header(compound["preferred_name"])

metric_columns = st.columns(4)
metric_columns[0].metric("Evidence records", compound["evidence_count"])
metric_columns[1].metric("Clinical PK", compound["clinical_pk_count"])
metric_columns[2].metric("Labels", compound["label_count"])
metric_columns[3].metric("Public models", compound["public_model_count"])

identifier_rows = [
    {"identifier": "Preferred name", "value": compound["preferred_name"]},
    {"identifier": "Synonyms", "value": data.format_list(compound["synonyms"])},
    {"identifier": "PubChem CID", "value": compound["pubchem_cid"]},
    {"identifier": "ChEMBL ID", "value": compound["chembl_id"]},
    {"identifier": "DrugCentral ID", "value": compound["drugcentral_id"]},
    {"identifier": "RxNorm RxCUI", "value": compound["rxnorm_rxcui"]},
    {"identifier": "InChIKey", "value": compound["inchi_key"]},
    {"identifier": "Source", "value": f"{compound['source_id']} / {compound['source_location']}"},
    {"identifier": "Curation status", "value": compound["curation_status"]},
]

tabs = st.tabs(
    [
        "Identifiers",
        "Physicochemical",
        "ADME",
        "Clinical PK",
        "DDI",
        "Special Populations",
        "Public Models",
        "Source Documents",
        "Evidence Gaps",
    ]
)

with tabs[0]:
    display_table(identifier_rows, height=320)

def evidence_rows(evidence_type: str) -> list[dict]:
    rows = data.compound_evidence(selected_id, evidence_type=evidence_type)
    return [
        {
            "parameter": row.get("parameter_name"),
            "value": data.evidence_value(row),
            "source": row["source_id"],
            "location": row["source_location"],
            "role": row.get("compound_role"),
            "method": row["extraction_method"],
            "confidence": row["confidence_score"],
            "attribution_confidence": row.get("attribution_confidence"),
            "status": row["curation_status"],
            "text": data.snippet(row["evidence_text"]),
        }
        for row in rows
    ]


with tabs[1]:
    display_table(evidence_rows("physicochemical"), height=420)

with tabs[2]:
    display_table(evidence_rows("adme"), height=420)

with tabs[3]:
    clinical_rows = evidence_rows("clinical_pk")
    display_table(clinical_rows, height=460)
    if clinical_rows:
        if st.button("Search related evidence", width="stretch"):
            st.session_state["evidence_compound_filter"] = compound["preferred_name"]
            st.session_state["evidence_type_filter"] = "clinical_pk"
            st.switch_page("pages/05_evidence_search.py")

with tabs[4]:
    ddi_rows = evidence_rows("ddi")
    display_table(ddi_rows, height=320)

with tabs[5]:
    display_table(evidence_rows("special_population"), height=420)

with tabs[6]:
    model_rows = data.model_registry(compound=compound["preferred_name"], limit=50)
    display_table(
        [
            {
                "model_name": row["model_name"],
                "compound": row["compound_name"],
                "software": row["software"],
                "source": row["source_id"],
                "model_file_path": row["model_file_path"],
                "validation_status": row["validation_status"],
                "curation_status": row["curation_status"],
                "notes": row["notes"],
            }
            for row in model_rows
        ],
        height=360,
    )

with tabs[7]:
    display_table(data.compound_source_documents(selected_id), height=420)

with tabs[8]:
    all_evidence = data.compound_evidence(selected_id, limit=500)
    present_types = {row["evidence_type"] for row in all_evidence}
    gap_rows = [
        {
            "area": "DDI evidence",
            "status": "gap" if "ddi" not in present_types else "indexed",
            "next_step": "Add public DDI or bioactivity source connectors before Q&A.",
        },
        {
            "area": "Auto-validated evidence",
            "status": "indexed" if all_evidence else "gap",
            "next_step": (
                "Only evidence passing deterministic and independent quality "
                "validation is served."
            ),
        },
        {
            "area": "Linked public model",
            "status": "gap" if not compound["public_model_count"] else "indexed",
            "next_step": "Curate OSP model compound linkage when model metadata supports it.",
        },
    ]
    display_table(gap_rows, height=240)
    status_note(
        "This profile serves source-backed evidence from the final "
        "auto-validated quality layer."
    )
