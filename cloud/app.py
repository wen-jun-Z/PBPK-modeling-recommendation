"""Private, self-contained Streamlit presentation of the PBPK research workspace."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from cloud.data_store import load_blueprints, load_catalog
from cloud.access import enforce_preview_access
from cloud.blueprints import render_blueprints
from cloud.views import render_overview, render_evidence, render_models, render_recommendations, render_graph

st.set_page_config(page_title="PBPK · Model discovery", page_icon="◈", layout="wide")
enforce_preview_access()
st.markdown(f"<style>{(ROOT / 'cloud/theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

PAGES = {
    "Overview": render_overview,
    "Compounds & evidence": render_evidence,
    "Public models": render_models,
    "Recommendations": render_recommendations,
    "Blueprint design": render_blueprints,
    "Knowledge graph": render_graph,
}

try:
    catalog = load_catalog()
except (OSError, ValueError, KeyError) as exc:
    st.error("The research snapshot could not be loaded. Please contact the app owner to check the release files.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="pbpk-brand"><span class="pbpk-mark">P<span>k</span></span><div><strong>PBPK Workspace</strong><small>MODELING RESEARCH</small></div></div>', unsafe_allow_html=True)
    st.caption("RESEARCH WORKSPACE")
    page = st.radio("Navigation", list(PAGES), key="workspace_page", label_visibility="collapsed")
    st.markdown('<div class="sidebar-note">Research showcase<br><span>Evidence · Models · Physiology</span></div>', unsafe_allow_html=True)

left, right = st.columns([3, 1])
left.caption(f"Workspace / {page}")
right.markdown('<div class="showcase-badge">PBPK RESEARCH</div>', unsafe_allow_html=True)

if page == "Blueprint design":
    try:
        PAGES[page](load_blueprints())
    except (OSError, ValueError, KeyError) as exc:
        st.error("This blueprint snapshot is incomplete. Please select another case or contact the app owner.")
else:
    PAGES[page](catalog)

meta = catalog.get("metadata", {})
snapshot = meta.get("snapshot_at") or meta.get("exported_at") or meta.get("created_at") or ""
st.divider()
st.caption(f"PBPK MODELING WORKSPACE  ·  Research snapshot{(' · ' + str(snapshot)[:10]) if snapshot else ''}  ·  For research use")
