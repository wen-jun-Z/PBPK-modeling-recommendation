"""Actual source content must determine acquisition coverage."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pmc_audit", ROOT / "scripts/deployment/audit_expansion_pmc.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_abstract_and_requested_id_are_not_fulltext():
    xml = b'<pmc-articleset><article><front><article-meta><article-id pub-id-type="pmcid">PMC12</article-id><abstract>PK abstract</abstract></article-meta></front></article></pmc-articleset>'
    result = audit.article_content(xml)
    assert result == {"PMC12": {"body_chars": 0, "body_paragraphs": 0, "body_tables": 0, "abstract_chars": 11}}


def test_nested_reference_ids_cannot_change_article_identity():
    xml = b'<article><front><article-meta><article-id pub-id-type="pmc">12</article-id></article-meta></front><body><p>clearance measured</p></body><back><article-id pub-id-type="pmcid">PMC99</article-id></back></article>'
    result = audit.article_content(xml)
    assert set(result) == {"PMC12"}
    assert result["PMC12"]["body_paragraphs"] == 1


def test_namespaced_fulltext_and_heading_only_distinguished():
    xml = b'<article xmlns="urn:jats"><front><article-meta><article-id pub-id-type="pmcid">PMC12</article-id></article-meta></front><body><sec><title>Withdrawn</title></sec></body></article>'
    result = audit.article_content(xml)["PMC12"]
    assert result["body_chars"] > 0 and result["body_paragraphs"] == 0


def test_metadata_or_ambiguous_identity_cannot_be_counted_as_article():
    assert audit.article_content(b'<error>Missing fulltext</error>') == {}
    xml = b'<article><front><article-meta><article-id pub-id-type="pmcid">PMC12</article-id><article-id pub-id-type="pmc">99</article-id></article-meta></front><body><p>text</p></body></article>'
    assert audit.article_content(xml) == {}


def test_table_caption_is_not_missing_narrative_body():
    xml = b'<article><front><article-meta><article-id pub-id-type="pmcid">PMC12</article-id></article-meta></front><body><table-wrap><caption><p>PK parameters</p></caption><table><tr><td>1.2</td></tr></table></table-wrap></body></article>'
    result = audit.article_content(xml)["PMC12"]
    assert result["body_paragraphs"] == 0 and result["body_tables"] == 1
