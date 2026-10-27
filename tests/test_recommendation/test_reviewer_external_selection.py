import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reviewer_external_selection", ROOT / "scripts/reviewer_external_selection.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def test_methods_parser_excludes_nested_outcome_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "ROOT", tmp_path)
    p = tmp_path / "article.xml"
    p.write_text('<article><body><sec><title>Methods</title><sec id="design"><title>Study design</title><p>Dose80mg</p>'
                 '<table-wrap id="T1"><caption>PK parameters</caption><table><thead><tr><th>Alone</th></tr></thead>'
                 '<tbody><tr><td>SECRET_OUTCOME_123</td></tr></tbody></table></table-wrap>'
                 '<fig><caption>SECRET_FIGURE</caption></fig></sec></sec></body></article>')
    got = selection.methods_and_inventory(p)
    assert "Dose80mg" in got["sections"][0]["text"]
    assert "SECRET" not in str(got)
    assert got["table_inventory_headers_only"][0]["column_headers"] == "Alone"
