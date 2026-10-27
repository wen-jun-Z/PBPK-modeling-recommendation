"""Native Streamlit views over the verified, immutable research snapshot."""
from collections import Counter
from html import escape
import json
import ipaddress
import math
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LABELS = {
    "physicochemical": "Physicochemical", "adme": "ADME", "clinical_pk": "Clinical pharmacokinetics",
    "ddi": "Drug interactions", "special_population": "Special populations", "tissue_expression": "Tissue expression",
    "compound_identity": "Compound identity", "absorption": "Absorption", "distribution": "Distribution",
    "metabolism": "Metabolism", "excretion": "Excretion and clearance", "transport": "Transport",
    "public_model_template": "Public model template", "base_pk": "Basic pharmacokinetics",
    "public_model_reuse": "Public model reuse", "ddi_victim": "DDI victim",
    "ddi_perpetrator": "DDI perpetrator", "renal_impairment": "Renal impairment",
    "hepatic_impairment": "Hepatic impairment", "pediatric": "Pediatric", "tissue_exposure": "Tissue exposure",
    "oral": "Oral", "intravenous": "Intravenous", "iv": "Intravenous", "unspecified": "Unspecified",
    "human": "Human", "rat": "Rat", "mouse": "Mouse", "monkey": "Monkey",
    "evidence_supported_candidate": "Evidence supported", "recommended": "Recommended candidate",
    "conditional": "Conditional candidate", "insufficient_evidence": "Insufficient evidence",
    "auto_validated": "Automatically validated", "needs_review": "Review pending", "manually_curated": "Manually curated",
    "passed": "Passed", "success": "Passed", "not_run": "Not executed", "not_tested": "Not tested",
    "failed": "Not passed", "unknown": "Not recorded", "single": "Single dose", "multiple": "Multiple doses",
    "whole_body": "Whole-body PBPK", "parent_only": "Parent drug", "whole_body_pbpk": "Whole-body PBPK",
    "reduced_pbpk": "Reduced PBPK", "parent_metabolite": "Parent drug and metabolites",
    "concentration_time_profile": "Concentration profile",
    "parent": "Parent drug",
    "CRITICAL_EVIDENCE_GAP": "Critical evidence gaps", "ENDPOINT_UNSUPPORTED": "Requested endpoint not supported",
    "LICENSE_UNRESOLVED": "License unresolved", "VALIDATION_CRITERIA_MISSING": "Validation criteria missing",
    "VALIDATION_NOT_TRANSFERABLE": "Existing validation is not transferable", "EXECUTION_PROBE_FAILED": "Execution check failed",
    "PARSER_FAILED": "Model parsing failed", "REQUIRED_MECHANISM_MISSING": "Required mechanism missing",
    "SPECIES_MISMATCH": "Species mismatch",
    "public_pbpk_model": "Public PBPK model",
    "same_compound_qualification_candidate": "Same compound; further validation needed",
    "direct_reuse_candidate": "Candidate for direct reuse",
    "cross_compound_reference": "Cross-compound reference",
    "blocked": "Not currently applicable",
}
GREEN = "#628f7a"
BLUE = "#7296ae"
GOLD = "#b6a17c"
PURPLE = "#9b8caf"


def label(value):
    return LABELS.get(str(value), str(value)) if value is not None else "—"


def text(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(text(v) for v in value) or "—"
    if isinstance(value, dict):
        return str(value.get("name") or value.get("preferred_name") or value.get("message") or "—")
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def url(value):
    """Only emit genuine external HTTP(S) links, never local paths or code URLs."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").rstrip(".").casefold()
        if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        if parts.scheme in {"https", "http"} and host and not parts.username and not parts.password:
            return value
    except ValueError:
        pass
    return None


def title(name, subtitle):
    st.title(name)
    st.caption(subtitle)
    st.write("")


def empty(message):
    st.markdown(f'<div class="pbpk-empty">{escape(message)}</div>', unsafe_allow_html=True)


def quote(message):
    st.markdown(f'<div class="pbpk-quote">{escape(str(message))}</div>', unsafe_allow_html=True)


def details(fields):
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(text(v))}</dd>" for k, v in fields.items() if v not in (None, "", [], {}))
    st.markdown(f'<dl class="pbpk-detail">{body}</dl>', unsafe_allow_html=True)


def badges(values):
    st.markdown("".join(f'<span class="pbpk-badge">{escape(label(v))}</span>' for v in values if v), unsafe_allow_html=True)


def download(data, filename, key, caption="Download this record"):
    st.download_button(caption, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), file_name=filename, mime="application/json", key=key)


def compound_name(row):
    return str(row.get("compound_name") or row.get("preferred_name") or row.get("name") or "Unnamed")


def covered_compounds(data):
    counts = Counter(e.get("compound_id") for e in data.get("evidence", []))
    return sorted([c for c in data.get("compounds", []) if counts.get(c.get("compound_id"))], key=lambda c: compound_name(c).casefold())


def model_name(row):
    return str(row.get("model_name") or row.get("name") or "Unnamed model")


def model_compounds(row):
    raw = row.get("compounds") or []
    return [str(v.get("name") or v.get("compound_name") or "") if isinstance(v, dict) else str(v) for v in (raw if isinstance(raw, list) else [raw])]


def source_link(row, caption="View original source ↗"):
    link = url(row.get("source_url"))
    if link:
        st.link_button(caption, link)
    else:
        st.caption("A verified direct source link is not yet available for this record.")


def selected_row(rows, columns, key, height=360, column_config=None):
    """Stable single-row selection with a useful first-row detail on initial load."""
    if not rows:
        return None
    frame = pd.DataFrame([{name: getter(row) for name, getter in columns.items()} for row in rows])
    event = st.dataframe(frame, hide_index=True, width="stretch", height=height,
                         on_select="rerun", selection_mode="single-row", key=key,
                         column_config=column_config or {})
    indices = event.selection.rows
    return rows[indices[0]] if indices and indices[0] < len(rows) else rows[0]


def horizontal_bars(counts, height=300, color=GREEN):
    rows = list(counts)
    figure = go.Figure(go.Bar(x=[v for _, v in rows], y=[k for k, _ in rows], orientation="h",
                             marker_color=color, marker_line_width=0, text=[f"{v:,}" for _, v in rows],
                             textposition="outside", cliponaxis=False, hovertemplate="%{y} · %{x:,} records<extra></extra>"))
    figure.update_layout(height=height, margin=dict(l=0, r=35, t=10, b=15), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#718579", size=12), showlegend=False,
                         xaxis=dict(showgrid=True, gridcolor="#edf1ed", zeroline=False, title=None),
                         yaxis=dict(autorange="reversed", showgrid=False, title=None), bargap=.42)
    return figure


def render_overview(data):
    evidence = data.get("evidence", [])
    models = data.get("models", [])
    compounds = covered_compounds(data)
    families = {m.get("model_id") or m.get("model_name") for m in models}
    sources = {e.get("source_id") for e in evidence if e.get("source_id")}
    title("From evidence to models to predictions.", "Explore compound evidence, public models, recommendation rationale, and PBPK blueprints.")
    for col, name, value in zip(st.columns(4), ["Compounds with evidence", "Available evidence", "Public model versions", "Evidence sources"],
                                 [len(compounds), len(evidence), len(models), len(sources)]):
        col.metric(name, f"{value:,}")
    st.markdown('<div class="pbpk-hero"><div class="pbpk-eyebrow">EVIDENCE TO MODEL</div><h2>Shape your next PBPK study</h2><p>Connect traceable drug knowledge, public models, and physiology. Start with individual evidence, examine recommendation rationale, and explore predictions for a study case.</p></div>', unsafe_allow_html=True)
    a, b = st.columns([1.15, 1], gap="large")
    with a, st.container(border=True):
        st.subheader("Evidence coverage")
        counts = Counter(label(e.get("evidence_type")) for e in evidence)
        st.plotly_chart(horizontal_bars(counts.most_common()), width="stretch", config={"displayModeBar": False})
        st.caption("Imported evidence; automated validation does not replace scientific review.")
    with b, st.container(border=True):
        st.subheader("Research workflow")
        for number, heading, description in [
            ("01", "Explore evidence", "Browse drug parameters, study conditions, and the original source of each evidence record."),
            ("02", "Understand recommendations", "Review saved model suggestions, specific components, parameter candidates, and citations."),
            ("03", "Design and predict", "Keep the case physiology fixed, adjust dosing, and calculate concentration profiles."),
        ]:
            st.markdown(f"**{number}　{heading}**")
            st.caption(description)
            if number != "03":
                st.divider()
    with st.container(border=True):
        st.subheader("Evidence by compound")
        counts = Counter(e.get("compound_name") or "Not recorded" for e in evidence)
        st.plotly_chart(horizontal_bars(counts.most_common(8), 330, "#91ac9a"), width="stretch", config={"displayModeBar": False})
        st.caption(f"Top 8 compounds by evidence count · Complete catalog: {len(families):,} model families")


def render_evidence(data):
    title("Compounds & evidence", "Explore study conditions and parameters, with a direct source link for each traceable record.")
    mode = st.radio("Browse", ["Evidence search", "Compound directory"], horizontal=True, label_visibility="collapsed", key="evidence_mode")
    compounds = covered_compounds(data)
    evidence = data.get("evidence", [])
    if mode == "Compound directory":
        query = st.text_input("Find a compound", placeholder="Name or synonym, e.g. midazolam", key="compound_query").strip().casefold()
        rows = [r for r in compounds if query in " ".join([compound_name(r), *r.get("synonyms", [])]).casefold()]
        counts = Counter(e.get("compound_id") for e in evidence)
        st.caption(f"{len(rows)} compounds with available evidence")
        selected = selected_row(rows, {"Compound": compound_name, "Evidence records": lambda r: counts[r["compound_id"]],
                                "PubChem CID": lambda r: text(r.get("pubchem_cid")), "ChEMBL": lambda r: text(r.get("chembl_id"))}, "compound_table")
        if selected:
            with st.container(border=True):
                st.subheader(compound_name(selected))
                details({"Synonyms": selected.get("synonyms", [])[:6], "PubChem CID": selected.get("pubchem_cid"),
                         "ChEMBL": selected.get("chembl_id"), "InChIKey": selected.get("inchi_key")})
                st.caption("Counts below refer to evidence imported for this compound.")
                types = Counter(label(e.get("evidence_type")) for e in evidence if e.get("compound_id") == selected.get("compound_id"))
                badges([f"{k} · {v}" for k, v in types.most_common()])
        else:
            empty("No matching compound. Try another search term.")
        return
    a, b, c = st.columns([1, 1, 1.45])
    names = [compound_name(r) for r in compounds]
    with a:
        compound = st.selectbox("Compound", ["All compounds", *names], key="evidence_compound")
    with b:
        evidence_type = st.selectbox("Evidence type", ["all", *sorted({e.get("evidence_type", "unknown") for e in evidence})],
                                     format_func=lambda v: "All types" if v == "all" else label(v))
    with c:
        query = st.text_input("Search evidence", placeholder="Parameter, mechanism, or source-text keyword").strip().casefold()
    rows = [r for r in evidence if (compound == "All compounds" or r.get("compound_name") == compound)
            and (evidence_type == "all" or r.get("evidence_type") == evidence_type)
            and query in " ".join(str(r.get(k) or "") for k in ["evidence_text", "parameter_name", "context", "source_title"]).casefold()]
    st.caption(f"Found {len(rows):,} records · Select a row to inspect the excerpt and study conditions")
    if not rows:
        empty("No evidence matches these filters. Try another keyword or compound.")
        return
    def value(r):
        return " ".join(str(v) for v in [r.get("numeric_value") if r.get("numeric_value") is not None else r.get("value_text"), r.get("unit")] if v is not None) or "Qualitative evidence"
    selected = selected_row(rows, {"Compound": lambda r: r.get("compound_name"), "Type": lambda r: label(r.get("evidence_type")),
                            "Parameter": lambda r: r.get("parameter_name"), "Value": value,
                            "Study conditions": lambda r: r.get("context"), "Source": lambda r: r.get("source_name") or r.get("source_id")}, "evidence_table", 350)
    with st.container(border=True):
        st.subheader(f"{selected.get('compound_name')} · {selected.get('parameter_name') or label(selected.get('evidence_type'))}")
        badges([selected.get("evidence_type"), selected.get("source_name") or selected.get("source_id"), selected.get("curation_status")])
        quote(selected.get("evidence_text") or "No source excerpt was saved for this record.")
        details({"Value and unit": value(selected), "Study conditions": selected.get("context"), "Population": selected.get("population"),
                 "Route": label(selected["route"]) if selected.get("route") else None, "Dose": selected.get("dose")})
        source_link(selected)
    download(rows, "pbpk-evidence-selection.json", "evidence_download", "Download matching evidence")


def render_model_detail(row, prefix):
    st.subheader(model_name(row))
    badges([row.get("software"), row.get("version_label")])
    details({"Compound": model_compounds(row), "Species": [label(v) for v in row.get("species", [])],
             "Route": [label(v) for v in row.get("routes", [])], "Population": row.get("populations"),
             "Enzymes": row.get("enzymes"), "Transporters": row.get("transporters"),
             "Compartments": row.get("compartments"), "Outputs": row.get("output_names"), "License": row.get("license_spdx_id")})
    if row.get("scientific_accepted") is True:
        st.caption("This version has a recorded scientific validation. Check the original intended use and study conditions before reuse.")
    else:
        st.caption("Catalog registration or successful execution does not establish suitability for a new prediction scenario.")
    link = url(row.get("repository_url"))
    if link:
        st.link_button("Open model repository ↗", link)
    download(row, "pbpk-model-record.json", prefix + "_download")


def render_models(data):
    title("Public models", "Explore the complete public model catalog and inspect compounds, model structures, and conditions for reuse.")
    models = data.get("models", [])
    a, b, c = st.columns([1.5, 1, 1])
    with a:
        query = st.text_input("Search models", placeholder="Model name, compound, or enzyme").strip().casefold()
    with b:
        software = st.selectbox("Software", ["All software", *sorted({str(r["software"]) for r in models if r.get("software")})])
    with c:
        route = st.selectbox("Route", ["all", *sorted({v for r in models for v in r.get("routes", [])})],
                             format_func=lambda v: "All routes" if v == "all" else label(v), key="model_route")
    rows = [r for r in models if (software == "All software" or r.get("software") == software)
            and (route == "all" or route in r.get("routes", []))
            and query in " ".join([model_name(r), *model_compounds(r), *r.get("enzymes", []), *r.get("transporters", [])]).casefold()]
    st.caption(f"{len(rows):,} model versions · Select a row to inspect its details")
    selected = selected_row(rows, {"Model": model_name, "Version": lambda r: text(r.get("version_label")),
                            "Compound": lambda r: text(model_compounds(r)), "Software": lambda r: text(r.get("software")),
                            "Route": lambda r: text([label(v) for v in r.get("routes", [])])}, "model_table", 390)
    if selected:
        with st.container(border=True):
            render_model_detail(selected, "model")
    else:
        empty("No models match these filters.")


def request_summary(request):
    fields = [label(request.get("objective")), label(request.get("species")), label(request.get("route")), request.get("population")]
    if request.get("dose") is not None:
        fields.append(f"{request['dose']} {request.get('dose_unit', '')}")
    return " · ".join(str(v) for v in fields if v and v != "—")


def render_candidate(row, index, models):
    with st.container(border=True):
        left, right = st.columns([5, 1])
        left.markdown(f"**{row.get('rank') or index}　{row.get('name') or model_name(row)}**")
        left.caption(text(row.get("compounds")))
        score = row.get("score")
        if isinstance(score, (float, int)) and math.isfinite(score):
            right.metric("Relevance score", f"{score:.3f}")
        badges([row.get("display_layer")])
        reasons = row.get("reason_codes") or []
        if reasons:
            st.caption("Rationale: " + "; ".join(label(v) for v in reasons))
        version_id = row.get("model_version_id") or row.get("candidate_id")
        match = next((m for m in models if m.get("model_version_id") == version_id), None)
        if match:
            with st.expander("Structure and applicability"):
                details({"Software": match.get("software"), "Species": match.get("species"), "Route": match.get("routes"),
                         "Population": match.get("populations"), "Enzymes": match.get("enzymes"), "Compartments": match.get("compartments")})
                link = url(match.get("repository_url"))
                if link:
                    st.link_button("Open model repository ↗", link)


def render_saved_recommendations(data):
    rows = sorted(data.get("recommendations", []), key=lambda r: (
        r.get("request", {}).get("species") != "human",
        not r.get("request", {}).get("route"),
        not r.get("request", {}).get("frequency"),
    ))
    if not rows:
        empty("This snapshot contains no completed model recommendation records.")
        return
    names = sorted({r.get("request", {}).get("compound") or r.get("compound_name") or "Not recorded" for r in rows})
    name = st.selectbox("Compound", names, key="saved_recommendation_compound")
    matches = [r for r in rows if (r.get("request", {}).get("compound") or r.get("compound_name") or "Not recorded") == name]
    index = st.selectbox("Completed study request", range(len(matches)),
                         format_func=lambda i: f"{request_summary(matches[i].get('request', {}))} · {str(matches[i].get('created_at') or '')[:16]} · Record {i + 1}")
    row = matches[index]
    with st.container(border=True):
        st.subheader("Modeling request")
        request = row.get("request", {})
        details({"Compound": name, "Objective": label(request.get("objective")), "Species": label(request.get("species")), "Route": label(request.get("route")),
                 "Target population": request.get("population"), "Endpoints": [label(v) for v in request.get("prediction_endpoints", [])],
                 "Dose": f"{request.get('dose')} {request.get('dose_unit', '')}" if request.get("dose") is not None else None})
        st.caption(f"Saved research result · {row.get('method') or 'Original research method'} · Compared {row.get('candidate_count', '—')} candidate versions")
    candidates = row.get("top_candidates") or []
    st.subheader("Candidate suggestions")
    if row.get("display_notice"):
        st.caption("Candidates are grouped by applicability, then ranked within each group. Pending validation does not permit direct reuse. The download preserves the full original ranking.")
    if row.get("request_gaps"):
        with st.expander("Request details needed before direct reuse"):
            for gap in row["request_gaps"]:
                st.write(text(gap))
    st.caption("Saved results are shown for their original requests. Relevance scores are not prediction accuracy. This showcase does not run new GPU ranking jobs.")
    for i, candidate in enumerate(candidates, 1):
        render_candidate(candidate, i, data.get("models", []))
    if not candidates:
        empty("No candidate passed the applicability checks for this request. Reference models and exclusion reasons are shown below.")
    for field, heading in [("reference_candidates", "Cross-compound mechanism references"), ("blocked_candidates", "Excluded candidates")]:
        others = row.get(field) or []
        if others:
            with st.expander(f"{heading} · {len(others)}"):
                st.dataframe(pd.DataFrame([{"Model": r.get("name"), "Applicability": label(r.get("display_layer")),
                                            "Reason": text([label(v) for v in r.get("reason_codes", [])])} for r in others]), hide_index=True, width="stretch")
    download(row, "pbpk-saved-model-recommendation.json", "saved_model_download", "Download recommendation record")


def render_components(data):
    rows = data.get("components", [])
    if not rows:
        empty("This snapshot contains no saved component recommendations.")
        return
    names = sorted({r.get("compound_name") or r.get("request", {}).get("compound") or "Not recorded" for r in rows}, key=str.casefold)
    name = st.selectbox("Compound", names, index=names.index("midazolam") if "midazolam" in names else 0, key="component_compound")
    matches = [r for r in rows if (r.get("compound_name") or r.get("request", {}).get("compound") or "Not recorded") == name]
    index = st.selectbox("Component study record", range(len(matches)),
                         format_func=lambda i: f"{request_summary(matches[i].get('request', {})) or label(matches[i].get('objective'))} · Record {i + 1}")
    row = matches[index]
    components = row.get("component_recommendations") or []
    citations = row.get("citations") or []
    parameters = row.get("parameter_candidates") or []
    st.caption(f"Saved recommendation · {len(components)} components · {len(parameters)} parameter candidates · {len(citations)} citations")
    structure = row.get("recommended_model_structure") or {}
    with st.container(border=True):
        st.subheader("Model structure and mechanisms")
        details({"Model scope": label(structure.get("whole_body_or_reduced_pbpk")),
                 "Drug representation": label(structure.get("parent_or_parent_metabolite")),
                 "Compartments": structure.get("required_compartments"), "Metabolic pathways": row.get("metabolism_pathways"),
                 "Transport pathways": row.get("transport_pathways")})
        st.caption("These evidence-supported component and parameter candidates have not necessarily been calibrated or validated in simulation.")
    for i, component in enumerate(components):
        with st.container(border=True):
            st.subheader(label(component.get("component") or component.get("name")))
            badges([component.get("status")])
            ids = set(component.get("evidence_ids") or [])
            refs = [r for r in citations if r.get("evidence_id") in ids]
            bound = [r for r in parameters if r.get("evidence_id") in ids]
            rationale = component.get("rationale") or component.get("reason") or component.get("description")
            if rationale:
                st.write(rationale)
            mentioned = list(dict.fromkeys(r.get("parameter") or r.get("parameter_name") for r in refs if r.get("parameter") or r.get("parameter_name")))
            if mentioned:
                st.write("Evidence covers: " + ", ".join(mentioned))
            if bound:
                st.dataframe(pd.DataFrame([{"Parameter or endpoint": p.get("parameter") or p.get("parameter_name"),
                                            "Value": text(p.get("value") if p.get("value") is not None else p.get("value_text")),
                                            "Unit": text(p.get("unit") or p.get("canonical_unit") or p.get("reported_unit")),
                                            "Study conditions": text(p.get("condition") or (p.get("study_context") or {}).get("condition"))} for p in bound]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No parameter candidate has passed the numerical checks for this component.")
            with st.expander(f"Linked evidence · {len(refs)}"):
                for j, ref in enumerate(refs):
                    if j:
                        st.divider()
                    st.markdown(f"**{ref.get('parameter') or label(ref.get('evidence_type'))}**")
                    st.write(ref.get("supporting_text") or ref.get("source_text") or "No source excerpt was saved.")
                    source_link(ref)
                if not refs:
                    st.caption("No linked source citation was saved for this component.")
    if row.get("evidence_gaps"):
        with st.expander("Evidence gaps"):
            for gap in row["evidence_gaps"]:
                msg = str(gap)
                if msg.startswith("missing_component:"):
                    msg = "Missing evidence: " + label(msg.split(":", 1)[1])
                if msg == "no_reusable_public_model":
                    msg = "No reusable public model is currently available"
                st.write("• " + msg)
    download(row, "pbpk-saved-components.json", "component_download", "Download components and citations")


def render_recommendations(data):
    title("Recommendations", "Inspect model suggestions and component evidence from real study requests.")
    mode = st.radio("Recommendation type", ["Model recommendations", "Component recommendations"], horizontal=True, label_visibility="collapsed", key="recommendation_mode")
    if mode == "Model recommendations":
        render_saved_recommendations(data)
    else:
        render_components(data)


def graph_data(data, compound, limit=12):
    """Only explicit identity links; no inferred or decorative biological edges."""
    evidence = [e for e in data.get("evidence", []) if e.get("compound_id") == compound.get("compound_id")]
    evidence = sorted(evidence, key=lambda e: (str(e.get("evidence_type")), str(e.get("parameter_name")), str(e.get("evidence_id"))))
    # Interleave evidence types so a dense physicochemical block does not hide PK.
    grouped = {}
    for row in evidence:
        grouped.setdefault(row.get("evidence_type"), []).append(row)
    shown = []
    depth = 0
    while len(shown) < min(limit, len(evidence)):
        for group in grouped.values():
            if depth < len(group) and len(shown) < limit:
                shown.append(group[depth])
        depth += 1
    names = {compound_name(compound).casefold(), *(str(v).casefold() for v in compound.get("synonyms", []))}
    models = [m for m in data.get("models", []) if names.intersection(v.casefold() for v in model_compounds(m))]
    models = sorted(models, key=lambda m: (model_name(m), str(m.get("version_label"))))
    sources = list(dict.fromkeys(e.get("source_id") for e in shown if e.get("source_id")))
    return evidence, shown, models, sources


def render_graph(data):
    title("Knowledge graph", "Explore explicit links between compounds, evidence, sources, and public models.")
    compounds = covered_compounds(data)
    if not compounds:
        empty("This snapshot contains no compound graph to display.")
        return
    names = [compound_name(c) for c in compounds]
    default = names.index("midazolam") if "midazolam" in names else 0
    index = st.selectbox("Focus compound", range(len(compounds)), format_func=lambda i: names[i], index=default)
    compound = compounds[index]
    all_evidence, evidence, all_models, sources = graph_data(data, compound)
    models = all_models[:10]
    nodes = [("compound", compound_name(compound), compound)]
    nodes.extend(("evidence", str(e.get("parameter_name") or label(e.get("evidence_type"))), e) for e in evidence)
    nodes.extend(("source", next((e.get("source_name") for e in evidence if e.get("source_id") == s and e.get("source_name")), s), {"source_id": s}) for s in sources)
    nodes.extend(("model", model_name(m), m) for m in models)
    positions = {}
    groups = [("compound", 0, GREEN), ("evidence", 1, BLUE), ("source", 2, GOLD), ("model", 3, PURPLE)]
    for kind, x, color in groups:
        indices = [i for i, n in enumerate(nodes) if n[0] == kind]
        for j, i in enumerate(indices):
            positions[i] = (x, (len(indices) - 1) / 2 - j)
    edges = []
    source_index = {n[2]["source_id"]: i for i, n in enumerate(nodes) if n[0] == "source"}
    for i, (kind, _, row) in enumerate(nodes):
        if kind in {"evidence", "model"}:
            edges.append((0, i))
        if kind == "evidence" and row.get("source_id") in source_index:
            edges.append((i, source_index[row["source_id"]]))
    figure = go.Figure()
    xs, ys = [], []
    for a, b in edges:
        xs.extend([positions[a][0], positions[b][0], None])
        ys.extend([positions[a][1], positions[b][1], None])
    figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1, color="#d3dfd7"), hoverinfo="skip", showlegend=False))
    kind_labels = {"compound": "Compound", "evidence": "Evidence", "source": "Source", "model": "Public models"}
    for kind, x, color in groups:
        indices = [i for i, n in enumerate(nodes) if n[0] == kind]
        if not indices:
            continue
        labels = [nodes[i][1] for i in indices]
        figure.add_trace(go.Scatter(x=[positions[i][0] for i in indices], y=[positions[i][1] for i in indices],
                                   mode="markers+text", marker=dict(size=20 if kind == "compound" else 12, color=color, line=dict(width=2, color="white")),
                                   text=[v if len(v) < 23 else v[:21] + "…" for v in labels],
                                   textposition="top center", textfont=dict(size=11, color="#486455"),
                                   name=kind_labels[kind], customdata=labels,
                                   hovertemplate="%{customdata}<extra>" + kind_labels[kind] + "</extra>"))
    max_rows = max(len(evidence), len(models), len(sources), 3)
    figure.update_layout(height=max(510, max_rows * 45), margin=dict(l=80, r=110, t=55, b=20),
                         paper_bgcolor="#ffffff", plot_bgcolor="#fafcfb", font=dict(color="#72867b"),
                         xaxis=dict(range=[-.3, 3.45], showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(range=[-max_rows / 2 - .5, max_rows / 2 + .7], showgrid=False, zeroline=False, showticklabels=False),
                         showlegend=False, hovermode="closest")
    with st.container(key="knowledge_graph_canvas"):
        st.markdown('<div class="pbpk-graph-legend">' + ''.join(
            f'<span><i style="background:{color}"></i>{kind_labels[kind]}</span>'
            for kind, _, color in groups) + '</div>', unsafe_allow_html=True)
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": False, "toImageButtonOptions": {"format": "svg", "filename": "pbpk-knowledge-graph"}})
    st.caption(f"Complete links: {len(all_evidence)} evidence records · {len(all_models)} model versions. Showing {len(evidence)} evidence records, {len(models)} models, and their sources. Hover to read full names.")
    a, b = st.columns([1.3, 1], gap="large")
    with a, st.container(border=True):
        st.subheader("Evidence in this graph")
        if evidence:
            selected = st.selectbox("Inspect evidence", range(len(evidence)), format_func=lambda i: f"{label(evidence[i].get('evidence_type'))} · {evidence[i].get('parameter_name') or 'Study record'} · {i + 1}")
            row = evidence[selected]
            quote(row.get("evidence_text") or "No source excerpt was saved.")
            source_link(row)
    with b, st.container(border=True):
        st.subheader("Linked models")
        if models:
            for m in models[:5]:
                st.markdown(f"**{model_name(m)}**")
                st.caption(f"{m.get('software') or 'Software not recorded'} · {m.get('version_label') or 'Version not recorded'}")
        else:
            st.caption("No public model is explicitly linked to this compound in the catalog.")
    download({"compound": compound, "evidence": all_evidence, "models": all_models}, "pbpk-compound-graph.json", "graph_download", "Download complete linked records")
