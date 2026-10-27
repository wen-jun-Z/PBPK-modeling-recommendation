from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import data
from pbpk_kb.ui.layout import configure_page, display_table, page_title


configure_page("Compound Search")
page_title("Compound Search")

query = st.text_input("Compound, synonym, or public identifier", st.session_state.get("compound_query", "midazolam"))
st.session_state["compound_query"] = query

limit = st.slider("Result limit", 5, 50, 25, 5)
rows = data.search_compounds(query, limit=limit) if query.strip() else data.search_compounds("", limit=limit)

display_rows = []
for row in rows:
    display_rows.append(
        {
            "preferred_name": row["preferred_name"],
            "pubchem_cid": row["pubchem_cid"],
            "chembl_id": row["chembl_id"],
            "rxnorm_rxcui": row["rxnorm_rxcui"],
            "inchi_key": row["inchi_key"],
            "evidence": row["evidence_count"],
            "clinical_pk": row["clinical_pk_count"],
            "labels": row["label_count"],
            "public_models": row["public_model_count"],
            "confidence": row["confidence_score"],
            "curation_status": row["curation_status"],
        }
    )

display_table(display_rows, height=350)

if rows:
    st.subheader("Open Compound Profile")
    for row in rows:
        columns = st.columns([3, 2, 2, 2])
        columns[0].write(row["preferred_name"])
        columns[1].write(f"PubChem CID: {row['pubchem_cid'] or ''}")
        columns[2].write(f"Evidence: {row['evidence_count']}")
        if columns[3].button("Open profile", key=f"profile-{row['compound_id']}", width="stretch"):
            st.session_state["selected_compound_id"] = row["compound_id"]
            st.switch_page("pages/04_compound_profile.py")
