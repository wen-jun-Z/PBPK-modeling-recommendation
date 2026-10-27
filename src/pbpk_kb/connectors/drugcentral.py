from __future__ import annotations

import csv
from io import StringIO
from typing import Any
from urllib.parse import urljoin

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ParsedSourceRecord, RawDocumentMetadata


class DrugCentralConnector(BaseConnector):
    source_id = "drugcentral"
    parser_name = "csv_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        download_page_url = str(self.source.access_url)
        page_bytes, page_content_type, page_status_code = self.fetch_bytes(download_page_url)
        self.save_raw_bytes(
            source_url=download_page_url,
            payload=page_bytes,
            extension="html",
            content_type=page_content_type,
            status_code=page_status_code,
            raw_record_count=1,
            retrieval_method="drugcentral_download_page",
        )

        approved_url = urljoin(download_page_url, "/static/FDA_Approved.csv")
        approved_bytes, approved_content_type, approved_status_code = self.fetch_bytes(approved_url)
        rows = self._parse_approved_csv(approved_bytes.decode("utf-8", errors="replace"))
        matches = [row for row in rows if query.lower() in row.get("drug_name", "").lower()]
        approved_raw = self.save_raw_bytes(
            source_url=approved_url,
            payload=approved_bytes,
            extension="csv",
            content_type=approved_content_type,
            status_code=approved_status_code,
            raw_record_count=len(rows),
            retrieval_method="drugcentral_fda_approved_csv",
            extra_metadata={"query": query, "match_count": len(matches)},
        )

        records = [self._parse_drug(row, approved_raw) for row in matches[:retmax]]
        if not records:
            records.append(
                self.make_record(
                    raw=approved_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=approved_url,
                    source_location=f"DRUGCENTRAL_QUERY:{query}",
                    record_type="drugcentral_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            )
        return records

    @staticmethod
    def _parse_approved_csv(csv_text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        reader = csv.reader(StringIO(csv_text))
        for row in reader:
            if len(row) < 2:
                continue
            rows.append({"drugcentral_id": row[0].strip(), "drug_name": row[1].strip(), "raw_fields": row})
        return rows

    def _parse_drug(self, row: dict[str, Any], raw: RawDocumentMetadata) -> ParsedSourceRecord:
        drugcentral_id = str(row.get("drugcentral_id") or "unknown_drugcentral_id")
        record = {
            "drugcentral_id": drugcentral_id,
            "drug_name": row.get("drug_name"),
            "approval_source": "FDA_Approved.csv",
            "raw_fields": row.get("raw_fields", []),
            "extraction_level": "approved_drug_identifier",
        }
        return self.make_record(
            raw=raw,
            source_record_id=drugcentral_id,
            source_url=f"https://drugcentral.org/drugcard/{drugcentral_id}",
            source_location=f"DRUGCENTRAL_ID:{drugcentral_id}",
            record_type="drugcentral_approved_drug",
            record=record,
        )
