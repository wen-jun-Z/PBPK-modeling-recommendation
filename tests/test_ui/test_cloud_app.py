"""Deployment checks against the real portable research snapshot."""
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

import pytest

from cloud import data_store


def test_snapshot_integrity_counts_links_and_no_private_runtime_data():
    catalog = data_store.load_catalog()
    blueprints = data_store.load_blueprints()
    manifest = json.loads((data_store.DATA / "manifest.json").read_text())
    counts = catalog["metadata"]["counts"]
    assert len(catalog["evidence"]) == counts["evidence"]
    assert len(catalog["models"]) == counts["models"] == 280
    assert len(blueprints["cases"]) == counts["blueprints"] == 30
    assert len({e["compound_id"] for e in catalog["evidence"]}) == counts["covered_compounds"]
    assert sum(manifest["files"][name]["bytes"] for name in manifest["files"]) <= 25 * 1024**2
    for name, entry in manifest["files"].items():
        raw = (data_store.DATA / name).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert not re.search(rb"/(?:home|Users|Volumes)/|(?i:password|access_token|access_code|api_key)\"\s*:", raw)
    for row in catalog["evidence"]:
        link = row.get("source_url")
        if link is None:
            continue
        parsed = urlsplit(link)
        assert parsed.scheme in {"https", "http"} and parsed.hostname
        assert not parsed.username and not parsed.password
        assert parsed.hostname not in {"localhost", "localhost.localdomain"}
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            address = None
        assert address is None or address.is_global
        if parsed.hostname == "eutils.ncbi.nlm.nih.gov":
            ids = parse_qs(parsed.query).get("id", [])
            assert len(ids) == 1 and "," not in ids[0], "Batch URL cannot identify an evidence source"
    assert all(c["provenance"]["source_assets_verified_at_export"] for c in blueprints["cases"])
    assert all(not c["provenance"]["source_assets_present_in_cloud"] for c in blueprints["cases"])
    assert all(r["saved_result"] for r in catalog["recommendations"])


def test_loader_rejects_changed_data_before_serving_it(tmp_path, monkeypatch):
    content = b'{"metadata":{"schema_version":1}}'
    (tmp_path / "catalog.json").write_bytes(content)
    monkeypatch.setattr(data_store, "DATA", tmp_path)
    data_store._verified.cache_clear()
    try:
        with pytest.raises(ValueError, match="checksum failed"):
            data_store._verified("catalog.json", "0" * 64, len(content))
        with pytest.raises(ValueError, match="size"):
            data_store._verified("catalog.json", hashlib.sha256(content).hexdigest(), len(content) + 1)
    finally:
        data_store._verified.cache_clear()


def test_six_native_pages_and_real_search_results(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("PBPK_PREVIEW_ACCESS_HASH", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[2] / "cloud/app.py", default_timeout=40).run()
    expected = {
        "Overview": "From evidence to models to predictions.", "Compounds & evidence": "Compounds & evidence",
        "Public models": "Public models", "Recommendations": "Recommendations",
        "Blueprint design": "Blueprint design", "Knowledge graph": "Knowledge graph",
    }
    assert list(app.radio(key="workspace_page").options) == list(expected)
    for page, heading in expected.items():
        app.radio(key="workspace_page").set_value(page).run()
        assert not app.exception and not app.error, page
        assert app.title[0].value == heading
        if page == "Public models":
            assert len(app.dataframe[0].value) == 280
            app.text_input[0].set_value("midazolam").run()
            assert not app.exception and not app.error
            assert 0 < len(app.dataframe[0].value) < 280
        if page == "Compounds & evidence":
            app.selectbox(key="evidence_compound").set_value("caffeine").run()
            rows = app.dataframe[0].value
            assert len(rows) > 0 and set(rows["Compound"]) == {"caffeine"}
        if page == "Blueprint design":
            assert len(app.selectbox(key="cloud_blueprint_case").options) == 30
            assert len(app.number_input) == 3
        if page == "Knowledge graph":
            assert len(app.get("plotly_chart")) == 1
    app.radio(key="workspace_page").set_value("Recommendations").run()
    next(r for r in app.radio if "Component recommendations" in r.options).set_value("Component recommendations").run()
    app.selectbox(key="component_compound").set_value("midazolam").run()
    assert not app.exception and not app.error
    assert len(app.subheader) > 3, "Must render specific components, not only a saved request label"
    assert len(app.dataframe) > 0, "Must render actual bound parameter candidates"


def test_local_preview_auth_precedes_data_loading(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PBPK_PREVIEW_ACCESS_HASH", hashlib.sha256(b"test-only-value").hexdigest())
    app = AppTest.from_file(Path(__file__).resolve().parents[2] / "cloud/app.py", default_timeout=40).run()
    assert not app.exception and not app.error
    assert app.text_input[0].label == "Preview access code"
    assert not app.metric and not app.radio and not app.dataframe
