from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Knowledge Graph")
page_title("Knowledge Graph Explorer", "Compound-centered PBPK relationships")


def _safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        st.error(f"{label}: {exc}")
        return None


def _dot_id(node_id: str) -> str:
    return "n_" + re.sub(r"[^a-zA-Z0-9_]", "_", node_id)


def _dot_label(value: Any, limit: int = 54) -> str:
    label = " ".join(str(value or "").split())
    if len(label) > limit:
        label = f"{label[: limit - 3]}..."
    return label.replace("\\", "\\\\").replace('"', '\\"')


def _node_style(node_type: str) -> tuple[str, str]:
    styles = {
        "compound": ("box", "#d8ecff"),
        "evidence": ("ellipse", "#fff4cc"),
        "source": ("folder", "#e6f4df"),
        "model": ("component", "#eadcff"),
        "model_parameter": ("note", "#f0e9ff"),
        "protein_target": ("hexagon", "#ffdede"),
    }
    return styles.get(node_type, ("ellipse", "#eeeeee"))


def _graphviz(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    lines = [
        "digraph PBPKKG {",
        '  graph [rankdir="LR", bgcolor="transparent"];',
        '  node [style="filled", fontname="Helvetica", fontsize="10"];',
        '  edge [fontname="Helvetica", fontsize="9", color="#666666"];',
    ]
    for node in nodes:
        shape, color = _node_style(node.get("type", ""))
        lines.append(
            f'  {_dot_id(node["id"])} [label="{_dot_label(node.get("label") or node["id"])}", shape="{shape}", fillcolor="{color}"];'
        )
    visible_ids = {node["id"] for node in nodes}
    for edge in edges:
        if edge["source"] not in visible_ids or edge["target"] not in visible_ids:
            continue
        lines.append(f'  {_dot_id(edge["source"])} -> {_dot_id(edge["target"])} [label="{_dot_label(edge.get("type"), 32)}"];')
    lines.append("}")
    return "\n".join(lines)


summary = _safe_call("Load graph summary", api_client.graph_summary) or {}
summary_columns = st.columns(6)
summary_columns[0].metric("Compounds", summary.get("compounds", 0))
summary_columns[1].metric("Evidence", summary.get("evidence_records", 0))
summary_columns[2].metric("Sources", summary.get("sources", 0))
summary_columns[3].metric("Models", summary.get("public_models", 0))
summary_columns[4].metric("Targets", summary.get("protein_targets", 0))
summary_columns[5].metric("Target links", summary.get("enzyme_transporter_relationships", 0))

center_mode = st.radio(
    "Graph center",
    ["Compound", "Enzyme / transporter", "Source", "Public model"],
    horizontal=True,
)

if center_mode == "Compound":
    compounds = data.search_compounds("", limit=250)
    if not compounds:
        st.info("No compounds are indexed.")
        st.stop()
    center_ids = [compound["compound_id"] for compound in compounds]
    selected_id = st.selectbox(
        "Compound",
        center_ids,
        format_func=lambda value: next(
            f"{compound['preferred_name']} ({compound.get('pubchem_cid') or 'no PubChem CID'})"
            for compound in compounds
            if compound["compound_id"] == value
        ),
    )
    st.session_state["selected_compound_id"] = selected_id
    graph = _safe_call("Load compound graph", api_client.compound_graph, selected_id)
else:
    center_type = {
        "Enzyme / transporter": "protein_target",
        "Source": "source",
        "Public model": "model",
    }[center_mode]
    centers = _safe_call("Load graph centers", api_client.graph_centers, center_type) or []
    if not centers:
        st.info(f"No {center_mode.lower()} graph centers are indexed.")
        st.stop()
    selected_id = st.selectbox(
        center_mode,
        [center["id"] for center in centers],
        format_func=lambda value: next(center["label"] for center in centers if center["id"] == value),
    )
    graph_loader = {
        "protein_target": api_client.target_graph,
        "source": api_client.source_graph,
        "model": api_client.model_graph,
    }[center_type]
    graph = _safe_call(f"Load {center_mode.lower()} graph", graph_loader, selected_id)
if graph is None:
    st.stop()

counts = graph.get("counts", {})
metric_columns = st.columns(5)
metric_columns[0].metric("Nodes", counts.get("nodes", 0))
metric_columns[1].metric("Edges", counts.get("edges", 0))
metric_columns[2].metric("Evidence nodes", counts.get("evidence", 0))
metric_columns[3].metric("Model nodes", counts.get("models", 0))
metric_columns[4].metric("Target links", counts.get("relationships", 0))

nodes = graph.get("nodes", [])
edges = graph.get("edges", [])
node_types = sorted({node.get("type") for node in nodes if node.get("type")})
selected_types = st.multiselect("Node types", node_types, default=node_types)
visible_nodes = [node for node in nodes if node.get("type") in selected_types]
visible_ids = {node["id"] for node in visible_nodes}
visible_edges = [edge for edge in edges if edge["source"] in visible_ids and edge["target"] in visible_ids]

tabs = st.tabs(["Graph", "Nodes", "Edges", "Open Evidence", "Export JSON"])

with tabs[0]:
    if visible_nodes:
        st.graphviz_chart(_graphviz(visible_nodes, visible_edges), width="stretch")
    else:
        status_note("Select at least one node type to render the graph.")

with tabs[1]:
    display_table(
        [
            {
                "id": node["id"],
                "type": node.get("type"),
                "label": node.get("label"),
                "properties": json.dumps(node.get("properties") or {}, ensure_ascii=False),
            }
            for node in visible_nodes
        ],
        height=420,
    )

with tabs[2]:
    display_table(
        [
            {
                "source": edge["source"],
                "type": edge.get("type"),
                "target": edge["target"],
                "properties": json.dumps(edge.get("properties") or {}, ensure_ascii=False),
            }
            for edge in visible_edges
        ],
        height=420,
    )

with tabs[3]:
    evidence_nodes = [node for node in nodes if node.get("type") == "evidence"]
    if not evidence_nodes:
        status_note("No evidence nodes are available for this compound graph.")
    else:
        selected_evidence = st.selectbox(
            "Evidence node",
            [node["id"] for node in evidence_nodes],
            format_func=lambda value: next(node.get("label") or value for node in evidence_nodes if node["id"] == value),
        )
        if st.button("Open evidence detail", width="stretch"):
            _, evidence_type, evidence_id = selected_evidence.split(":", 2)
            st.session_state["selected_evidence_type"] = evidence_type
            st.session_state["selected_evidence_id"] = evidence_id
            st.switch_page("pages/08_evidence_detail.py")

with tabs[4]:
    st.download_button(
        "Download graph JSON",
        data=json.dumps(graph, ensure_ascii=False, indent=2),
        file_name=f"compound_graph_{selected_id}.json",
        mime="application/json",
        width="stretch",
    )
    st.json({"counts": counts, "compound": graph.get("compound")})

status_note(
    "Every relationship is derived from final auto-validated evidence and retains "
    "its source location for provenance."
)
