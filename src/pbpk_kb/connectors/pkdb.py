from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ParsedSourceRecord, RawDocumentMetadata


class PKDBConnector(BaseConnector):
    source_id = "pkdb"
    parser_name = "json_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        swagger_url = "https://pk-db.com/api/v1/swagger.json"
        swagger_payload, swagger_content_type, swagger_status_code, swagger_bytes = self.fetch_json(swagger_url)
        self.source_version = swagger_payload.get("info", {}).get("version") or swagger_payload.get("swagger")
        self.save_raw_json(
            source_url=swagger_url,
            payload=swagger_payload,
            content_type=swagger_content_type,
            status_code=swagger_status_code,
            raw_record_count=len(swagger_payload.get("paths", {})),
            source_version=self.source_version,
            retrieval_method="pkdb_swagger_metadata",
            raw_payload_bytes=swagger_bytes,
        )

        filter_url = "https://pk-db.com/api/v1/filter/?" + urlencode(
            {"studies__substance": query, "concise": "true"}
        )
        filter_payload, filter_content_type, filter_status_code, filter_bytes = self.fetch_json(filter_url)
        self.save_raw_json(
            source_url=filter_url,
            payload=filter_payload,
            content_type=filter_content_type,
            status_code=filter_status_code,
            raw_record_count=1,
            source_version=self.source_version,
            retrieval_method="pkdb_filter_by_substance",
            extra_metadata={"query": query},
            raw_payload_bytes=filter_bytes,
        )

        studies_url = "https://pk-db.com/api/v1/studies/?" + urlencode({"substance": query})
        studies_payload, studies_content_type, studies_status_code, studies_bytes = self.fetch_json(studies_url)
        study_items = self._study_items(studies_payload)
        studies_raw = self.save_raw_json(
            source_url=studies_url,
            payload=studies_payload,
            content_type=studies_content_type,
            status_code=studies_status_code,
            raw_record_count=len(study_items),
            source_version=self.source_version,
            retrieval_method="pkdb_studies_by_substance",
            extra_metadata={"query": query},
            raw_payload_bytes=studies_bytes,
        )
        return [self._parse_study(item, studies_raw, filter_payload) for item in study_items[:retmax]]

    @staticmethod
    def _study_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data", {})
        if isinstance(data, dict):
            items = data.get("data", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        return [item for item in items if isinstance(item, dict)]

    def _parse_study(
        self,
        item: dict[str, Any],
        raw: RawDocumentMetadata,
        filter_payload: dict[str, Any],
    ) -> ParsedSourceRecord:
        sid = str(item.get("sid") or item.get("pk") or item.get("name") or "unknown_study")
        record = {
            "study_sid": item.get("sid"),
            "study_pk": item.get("pk"),
            "study_name": item.get("name"),
            "licence": item.get("licence"),
            "access": item.get("access"),
            "date": item.get("date"),
            "group_count": item.get("group_count"),
            "individual_count": item.get("individual_count"),
            "intervention_count": item.get("intervention_count"),
            "output_count": item.get("output_count"),
            "filter_counts": {
                "studies": filter_payload.get("studies"),
                "groups": filter_payload.get("groups"),
                "individuals": filter_payload.get("individuals"),
                "interventions": filter_payload.get("interventions"),
                "outputs": filter_payload.get("outputs"),
                "timecourses": filter_payload.get("timecourses"),
            },
            "raw_fields": item,
            "extraction_level": "pkdb_study_metadata",
        }
        return self.make_record(
            raw=raw,
            source_record_id=sid,
            source_url=f"https://pk-db.com/data/{sid}" if item.get("sid") else "https://pk-db.com/data",
            source_location=f"PKDB_STUDY:{sid}",
            record_type="pk_study_metadata",
            record=record,
        )
