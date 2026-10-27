from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ParsedSourceRecord, RawDocumentMetadata
from pbpk_kb.parsers.dailymed_spl import DailyMedSPLParser


class DailyMedConnector(BaseConnector):
    source_id = "dailymed"
    parser_name = "label_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?" + urlencode(
            {"drug_name": query, "page": "1", "pageSize": str(retmax)}
        )
        payload, content_type, status_code, raw_bytes = self.fetch_json(url)
        labels = payload.get("data", [])
        raw = self.save_raw_json(
            source_url=url,
            payload=payload,
            content_type=content_type,
            status_code=status_code,
            raw_record_count=len(labels),
            retrieval_method="dailymed_spls_search",
            extra_metadata={"query": query, "retmax": retmax},
            raw_payload_bytes=raw_bytes,
        )
        records: list[ParsedSourceRecord] = []
        for label in labels[:retmax]:
            set_id = self._label_set_id(label)
            xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
            xml_payload, xml_content_type, xml_status_code = self.fetch_bytes(xml_url)
            xml_raw = self.save_raw_bytes(
                source_url=xml_url,
                payload=xml_payload,
                extension="xml",
                content_type=xml_content_type,
                status_code=xml_status_code,
                raw_record_count=1,
                retrieval_method="dailymed_spl_xml_detail",
                extra_metadata={"set_id": set_id, "listing_raw_checksum": raw.raw_checksum},
            )
            records.append(self._parse_label_detail(label, xml_payload, xml_raw))
        return records

    @staticmethod
    def _label_set_id(label: dict[str, Any]) -> str:
        return str(label.get("setid") or label.get("set_id") or label.get("spl_set_id") or "unknown_setid")

    def _parse_label_detail(
        self,
        label: dict[str, Any],
        xml_payload: bytes,
        raw: RawDocumentMetadata,
    ) -> ParsedSourceRecord:
        set_id = self._label_set_id(label)
        source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
        detail = self.parse_spl_xml(xml_payload)
        record = {
            "spl_set_id": set_id,
            "title": label.get("title"),
            "published_date": label.get("published_date"),
            "spl_version": label.get("spl_version"),
            "labeler": label.get("labeler"),
            "active_ingredients": detail["active_ingredients"],
            "active_moieties": detail["active_moieties"],
            "clinical_pharmacology_section": detail["clinical_pharmacology_section"],
            "pk_section_candidates": detail["pk_section_candidates"],
            "ddi_section_candidates": detail["ddi_section_candidates"],
            "section_count": detail["section_count"],
            "raw_listing_fields": label,
            "extraction_level": "spl_xml_detail",
        }
        return self.make_record(
            raw=raw,
            source_record_id=str(set_id),
            source_url=source_url,
            source_location=f"SPL_SET_ID:{set_id}",
            record_type="label_detail",
            record=record,
        )

    def parse_spl_xml(self, xml_payload: bytes) -> dict[str, Any]:
        return DailyMedSPLParser().parse(xml_payload)
