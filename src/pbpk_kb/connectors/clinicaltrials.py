from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class ClinicalTrialsConnector(BaseConnector):
    source_id = "clinicaltrials"
    parser_name = "json_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        base_url = str(self.source.api_url or self.source.access_url).rstrip("/")
        version_url = f"{base_url}/version"
        try:
            version_payload, version_content_type, version_status_code, version_bytes = self.fetch_json(version_url)
            self.source_version = version_payload.get("dataTimestamp") or version_payload.get("apiVersion")
            self.save_raw_json(
                source_url=version_url,
                payload=version_payload,
                content_type=version_content_type,
                status_code=version_status_code,
                raw_record_count=1,
                source_version=self.source_version,
                retrieval_method="clinicaltrials_api_version",
                raw_payload_bytes=version_bytes,
            )
        except Exception as exc:
            self.errors.append(
                ConnectorError(
                    stage="clinicaltrials_api_version",
                    message=str(exc),
                    retryable=True,
                )
            )

        studies_url = f"{base_url}/studies?" + urlencode(
            {"query.term": query, "pageSize": str(retmax), "format": "json"}
        )
        studies_payload, studies_content_type, studies_status_code, studies_bytes = self.fetch_json(studies_url)
        studies = studies_payload.get("studies", [])
        studies_raw = self.save_raw_json(
            source_url=studies_url,
            payload=studies_payload,
            content_type=studies_content_type,
            status_code=studies_status_code,
            raw_record_count=len(studies),
            source_version=self.source_version,
            retrieval_method="clinicaltrials_study_search",
            extra_metadata={"query": query, "retmax": retmax},
            raw_payload_bytes=studies_bytes,
        )
        records = [self._parse_study(study, studies_raw) for study in studies[:retmax]]
        if not records:
            records.append(
                self.make_record(
                    raw=studies_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=studies_url,
                    source_location=f"CLINICALTRIALS_QUERY:{query}",
                    record_type="clinicaltrials_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            )
        return records

    def _parse_study(self, study: dict[str, Any], raw: RawDocumentMetadata) -> ParsedSourceRecord:
        protocol = study.get("protocolSection", {}) if isinstance(study.get("protocolSection"), dict) else {}
        identification = protocol.get("identificationModule", {}) if isinstance(protocol.get("identificationModule"), dict) else {}
        status = protocol.get("statusModule", {}) if isinstance(protocol.get("statusModule"), dict) else {}
        design = protocol.get("designModule", {}) if isinstance(protocol.get("designModule"), dict) else {}
        arms = protocol.get("armsInterventionsModule", {}) if isinstance(protocol.get("armsInterventionsModule"), dict) else {}
        eligibility = protocol.get("eligibilityModule", {}) if isinstance(protocol.get("eligibilityModule"), dict) else {}
        nct_id = str(identification.get("nctId") or "unknown_nct")
        interventions = arms.get("interventions", []) if isinstance(arms.get("interventions"), list) else []
        arms_groups = arms.get("armGroups", []) if isinstance(arms.get("armGroups"), list) else []
        record = {
            "nct_id": nct_id,
            "brief_title": identification.get("briefTitle"),
            "official_title": identification.get("officialTitle"),
            "organization": identification.get("organization"),
            "overall_status": status.get("overallStatus"),
            "start_date": status.get("startDateStruct"),
            "completion_date": status.get("completionDateStruct"),
            "study_type": design.get("studyType"),
            "phases": design.get("phases", []),
            "enrollment": design.get("enrollmentInfo"),
            "interventions": interventions,
            "arm_groups": arms_groups,
            "eligibility": {
                "sex": eligibility.get("sex"),
                "minimum_age": eligibility.get("minimumAge"),
                "maximum_age": eligibility.get("maximumAge"),
                "healthy_volunteers": eligibility.get("healthyVolunteers"),
            },
            "dose_regimen_text_candidates": self._dose_text_candidates(interventions, arms_groups),
            "raw_fields": study,
            "extraction_level": "trial_intervention_population_metadata",
        }
        return self.make_record(
            raw=raw,
            source_record_id=nct_id,
            source_url=f"https://clinicaltrials.gov/study/{nct_id}",
            source_location=f"NCT_ID:{nct_id}",
            record_type="clinical_trial_metadata",
            record=record,
        )

    @staticmethod
    def _dose_text_candidates(interventions: list[dict[str, Any]], arm_groups: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        for intervention in interventions:
            for field_name in ("name", "description"):
                value = intervention.get(field_name)
                if value:
                    candidates.append(str(value))
        for arm in arm_groups:
            for field_name in ("label", "description"):
                value = arm.get(field_name)
                if value:
                    candidates.append(str(value))
        return candidates[:20]
