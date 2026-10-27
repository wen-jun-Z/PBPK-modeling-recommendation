from __future__ import annotations

import csv
import re
from io import BytesIO, StringIO
from typing import Any
from urllib.parse import quote, urlencode, urljoin
from zipfile import ZipFile

from pbpk_kb.connectors.base import BaseConnector, LiveFetchUnavailable
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class DrugsFDAConnector(BaseConnector):
    source_id = "drugs_fda"
    parser_name = "csv_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        try:
            records = self._fetch_data_file_records(query=query, retmax=retmax)
            if records:
                return records
        except Exception as exc:
            self.errors.append(
                ConnectorError(
                    stage="drugsfda_data_file_download",
                    message=f"FDA data file retrieval failed; using openFDA Drugs@FDA fallback: {exc}",
                    retryable=True,
                )
            )
        return self._fetch_openfda_records(query=query, retmax=retmax)

    def _fetch_openfda_records(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        manifest_url = "https://api.fda.gov/download.json"
        manifest_payload, manifest_content_type, manifest_status_code, manifest_bytes = self.fetch_json(manifest_url)
        drugsfda_manifest = manifest_payload.get("results", {}).get("drug", {}).get("drugsfda", {})
        self.source_version = drugsfda_manifest.get("export_date") or manifest_payload.get("meta", {}).get("last_updated")
        self.save_raw_json(
            source_url=manifest_url,
            payload=manifest_payload,
            content_type=manifest_content_type,
            status_code=manifest_status_code,
            raw_record_count=len(drugsfda_manifest.get("partitions", [])),
            source_version=self.source_version,
            retrieval_method="openfda_download_manifest",
            raw_payload_bytes=manifest_bytes,
        )

        query_payload, query_raw, query_url = self._fetch_drugsfda_query(query=query, retmax=retmax)
        results = query_payload.get("results", [])
        if not results:
            return [
                self.make_record(
                    raw=query_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=query_url,
                    source_location=f"OPENFDA_DRUGSFDA_QUERY:{query}",
                    record_type="drugsfda_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            ]
        return [self._parse_application(item, query_raw, query_url) for item in results[:retmax]]

    def _fetch_data_file_records(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        page_url = str(self.source.access_url)
        page_bytes, page_content_type, page_status_code = self.fetch_bytes(page_url)
        page_text = page_bytes.decode("utf-8", errors="replace")
        data_last_updated = self._data_last_updated(page_text)
        self.source_version = data_last_updated
        self.save_raw_bytes(
            source_url=page_url,
            payload=page_bytes,
            extension="html",
            content_type=page_content_type,
            status_code=page_status_code,
            raw_record_count=1,
            source_version=self.source_version,
            retrieval_method="drugsfda_data_files_page",
            extra_metadata={"data_last_updated": data_last_updated},
        )

        zip_url = self._download_zip_url(page_text, page_url)
        if not zip_url:
            raise LiveFetchUnavailable("No Drugs@FDA ZIP download link found on FDA data files page")
        zip_bytes, zip_content_type, zip_status_code = self.fetch_bytes(zip_url)
        tables = self._read_zip_tables(zip_bytes)
        zip_raw = self.save_raw_bytes(
            source_url=zip_url,
            payload=zip_bytes,
            extension="zip",
            content_type=zip_content_type,
            status_code=zip_status_code,
            raw_record_count=sum(len(rows) for rows in tables.values()),
            source_version=self.source_version,
            retrieval_method="drugsfda_data_file_zip",
            extra_metadata={"query": query, "table_names": sorted(tables)},
        )

        matches = self._matching_product_rows(tables, query)[:retmax]
        if not matches:
            return [
                self.make_record(
                    raw=zip_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=zip_url,
                    source_location=f"DRUGSFDA_DATA_FILE_QUERY:{query}",
                    record_type="drugsfda_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            ]
        return [self._parse_data_file_product(match, tables, zip_raw, zip_url) for match in matches]

    @staticmethod
    def _data_last_updated(page_text: str) -> str | None:
        match = re.search(r"Data Last Updated:\s*([^<\n]+)", page_text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _download_zip_url(page_text: str, page_url: str) -> str | None:
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', page_text, flags=re.IGNORECASE)
        for href in hrefs:
            lowered = href.lower()
            if "drugsatfda" in lowered or ("/media/" in lowered and "download" in lowered):
                return urljoin(page_url, href)
        return None

    @staticmethod
    def _read_zip_tables(zip_bytes: bytes) -> dict[str, list[dict[str, str]]]:
        tables: dict[str, list[dict[str, str]]] = {}
        with ZipFile(BytesIO(zip_bytes)) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                with archive.open(name) as handle:
                    table_text = handle.read().decode("latin-1", errors="replace")
                rows = []
                for row in csv.DictReader(StringIO(table_text), delimiter="\t"):
                    rows.append({key.strip(): (value or "").strip() for key, value in row.items() if key})
                tables[name.rsplit("/", 1)[-1].lower()] = rows
        return tables

    @staticmethod
    def _matching_product_rows(tables: dict[str, list[dict[str, str]]], query: str) -> list[dict[str, str]]:
        query_lower = query.lower()
        return [
            row
            for row in tables.get("products.txt", [])
            if query_lower in row.get("DrugName", "").lower()
            or query_lower in row.get("ActiveIngredient", "").lower()
        ]

    def _parse_data_file_product(
        self,
        product: dict[str, str],
        tables: dict[str, list[dict[str, str]]],
        raw: RawDocumentMetadata,
        zip_url: str,
    ) -> ParsedSourceRecord:
        application_number = product.get("ApplNo") or "unknown_application"
        applications = [row for row in tables.get("applications.txt", []) if row.get("ApplNo") == application_number]
        submissions = [row for row in tables.get("submissions.txt", []) if row.get("ApplNo") == application_number]
        record = {
            "application_number": application_number,
            "application_type": applications[0].get("ApplType") if applications else None,
            "sponsor_name": applications[0].get("SponsorName") if applications else None,
            "product_number": product.get("ProductNo"),
            "drug_name": product.get("DrugName"),
            "active_ingredient": product.get("ActiveIngredient"),
            "form": product.get("Form"),
            "strength": product.get("Strength"),
            "reference_drug": product.get("ReferenceDrug"),
            "reference_standard": product.get("ReferenceStandard"),
            "submission_count": len(submissions),
            "submissions": submissions[:5],
            "raw_fields": product,
            "extraction_level": "approval_product_metadata_data_file",
        }
        product_number = product.get("ProductNo") or "unknown_product"
        return self.make_record(
            raw=raw,
            source_record_id=f"{application_number}:{product_number}",
            source_url=zip_url,
            source_location=f"DRUGSFDA_PRODUCTS_TXT:ApplNo={application_number};ProductNo={product_number}",
            record_type="drugsfda_application",
            record=record,
        )

    def _fetch_drugsfda_query(self, *, query: str, retmax: int) -> tuple[dict[str, Any], RawDocumentMetadata, str]:
        attempts = [
            f'openfda.brand_name:"{query}"',
            f'openfda.generic_name:"{query}"',
            f'products.active_ingredients.name:"{query}"',
        ]
        last_error: Exception | None = None
        for search in attempts:
            url = "https://api.fda.gov/drug/drugsfda.json?" + urlencode({"search": search, "limit": str(retmax)})
            try:
                payload, content_type, status_code, raw_bytes = self.fetch_json(url)
            except LiveFetchUnavailable as exc:
                last_error = exc
                continue
            results = payload.get("results", [])
            raw = self.save_raw_json(
                source_url=url,
                payload=payload,
                content_type=content_type,
                status_code=status_code,
                raw_record_count=len(results),
                source_version=self.source_version or payload.get("meta", {}).get("last_updated"),
                retrieval_method="openfda_drugsfda_query",
                extra_metadata={"query": query, "search": search},
                raw_payload_bytes=raw_bytes,
            )
            return payload, raw, url
        raise LiveFetchUnavailable(str(last_error) if last_error else f"No openFDA Drugs@FDA query succeeded for {query}")

    def _parse_application(self, item: dict[str, Any], raw: RawDocumentMetadata, query_url: str) -> ParsedSourceRecord:
        application_number = str(item.get("application_number") or item.get("applicationNumber") or "unknown_application")
        products = item.get("products", []) if isinstance(item.get("products"), list) else []
        submissions = item.get("submissions", []) if isinstance(item.get("submissions"), list) else []
        openfda = item.get("openfda", {}) if isinstance(item.get("openfda"), dict) else {}
        record = {
            "application_number": application_number,
            "sponsor_name": item.get("sponsor_name"),
            "application_type": item.get("application_type"),
            "brand_names": openfda.get("brand_name", []),
            "generic_names": openfda.get("generic_name", []),
            "rxcuis": openfda.get("rxcui", []),
            "product_count": len(products),
            "submission_count": len(submissions),
            "products": products[:5],
            "submissions": submissions[:5],
            "raw_fields": item,
            "extraction_level": "approval_product_metadata",
        }
        return self.make_record(
            raw=raw,
            source_record_id=application_number,
            source_url=f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={quote(application_number)}",
            source_location=f"APPLICATION_NUMBER:{application_number}",
            record_type="drugsfda_application",
            record=record,
        )
