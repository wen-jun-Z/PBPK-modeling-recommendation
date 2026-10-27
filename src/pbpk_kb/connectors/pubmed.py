from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ParsedSourceRecord


class PubMedConnector(BaseConnector):
    source_id = "pubmed"
    parser_name = "pubmed_parser.py"
    default_query = '("physiologically based pharmacokinetic" OR PBPK) AND midazolam'

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        esearch_url = (
            f"{base_url}/esearch.fcgi?"
            + urlencode({"db": "pubmed", "term": query, "retmode": "json", "retmax": str(retmax)})
        )
        esearch_payload, content_type, status_code, esearch_bytes = self.fetch_json(esearch_url)
        ids = esearch_payload.get("esearchresult", {}).get("idlist", [])
        esearch_raw = self.save_raw_json(
            source_url=esearch_url,
            payload=esearch_payload,
            content_type=content_type,
            status_code=status_code,
            raw_record_count=len(ids),
            retrieval_method="ncbi_eutilities_esearch",
            extra_metadata={"query": query, "retmax": retmax},
            raw_payload_bytes=esearch_bytes,
        )
        if not ids:
            return []

        esummary_url = f"{base_url}/esummary.fcgi?" + urlencode(
            {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        )
        esummary_payload, summary_content_type, summary_status_code, esummary_bytes = self.fetch_json(esummary_url)
        esummary_raw = self.save_raw_json(
            source_url=esummary_url,
            payload=esummary_payload,
            content_type=summary_content_type,
            status_code=summary_status_code,
            raw_record_count=len(ids),
            retrieval_method="ncbi_eutilities_esummary",
            extra_metadata={"pmids": ids},
            raw_payload_bytes=esummary_bytes,
        )
        return self._parse_summary_records(esummary_payload, esummary_raw)

    def _parse_summary_records(self, payload: dict[str, Any], raw) -> list[ParsedSourceRecord]:
        result = payload.get("result", {})
        records: list[ParsedSourceRecord] = []
        for pmid in result.get("uids", []):
            summary = result.get(pmid, {})
            article_ids = summary.get("articleids") or []
            doi = next(
                (article_id.get("value") for article_id in article_ids if article_id.get("idtype") == "doi"),
                None,
            )
            record = {
                "pmid": pmid,
                "title": summary.get("title"),
                "journal": summary.get("fulljournalname") or summary.get("source"),
                "publication_date": summary.get("pubdate"),
                "doi": doi,
                "authors": [author.get("name") for author in summary.get("authors", []) if author.get("name")],
                "article_ids": article_ids,
                "extraction_level": "metadata_summary",
            }
            records.append(
                self.make_record(
                    raw=raw,
                    source_record_id=pmid,
                    source_url=f"https://pubmed.ncbi.nlm.nih.gov/{quote(str(pmid))}/",
                    source_location=f"PMID:{pmid}",
                    record_type="publication_metadata",
                    record=record,
                )
            )
        return records
