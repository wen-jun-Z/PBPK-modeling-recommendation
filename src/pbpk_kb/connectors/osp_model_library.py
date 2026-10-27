from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import unquote

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class OSPModelLibraryConnector(BaseConnector):
    source_id = "osp_pbpk_model_library"
    parser_name = "osp_model_parser.py"
    default_query = "model_folder_inventory"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        try:
            return self._fetch_and_parse_github_api(retmax=retmax)
        except Exception as exc:
            self.errors.append(
                ConnectorError(
                    stage="github_api_fetch",
                    message=str(exc),
                    retryable=True,
                    source_record_id=None,
                )
            )
            return self._fetch_and_parse_github_html(retmax=retmax)

    def _fetch_and_parse_github_api(self, *, retmax: int) -> list[ParsedSourceRecord]:
        repo_url = "https://api.github.com/repos/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
        repo_payload, repo_content_type, repo_status_code, repo_bytes = self.fetch_json(repo_url)
        self.source_version = f"{repo_payload.get('default_branch')}@{repo_payload.get('pushed_at')}"
        self.save_raw_json(
            source_url=repo_url,
            payload=repo_payload,
            content_type=repo_content_type,
            status_code=repo_status_code,
            raw_record_count=1,
            source_version=self.source_version,
            retrieval_method="github_repo_metadata",
            raw_payload_bytes=repo_bytes,
        )

        contents_url = f"{repo_url}/contents"
        contents_payload, contents_content_type, contents_status_code, contents_bytes = self.fetch_json(contents_url)
        contents = contents_payload if isinstance(contents_payload, list) else []
        contents_raw = self.save_raw_json(
            source_url=contents_url,
            payload=contents_payload,
            content_type=contents_content_type,
            status_code=contents_status_code,
            raw_record_count=len(contents),
            source_version=self.source_version,
            retrieval_method="github_contents_root",
            raw_payload_bytes=contents_bytes,
        )
        folders = [entry for entry in contents if entry.get("type") == "dir" and not entry.get("name", "").startswith(".")]
        records: list[ParsedSourceRecord] = []
        for folder in folders[:retmax]:
            records.append(self._parse_folder(folder, contents_raw))
        return records

    def _fetch_and_parse_github_html(self, *, retmax: int) -> list[ParsedSourceRecord]:
        html_url = "https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
        payload, content_type, status_code = self.fetch_bytes(html_url)
        html = payload.decode("utf-8", "replace")
        folders = self._parse_html_folders(html)
        self.source_version = "github_html_tree:master"
        raw = self.save_raw_bytes(
            source_url=html_url,
            payload=payload,
            extension="html",
            content_type=content_type,
            status_code=status_code,
            raw_record_count=len(folders),
            source_version=self.source_version,
            retrieval_method="github_html_tree",
            extra_metadata={"fallback_reason": "github_api_unavailable"},
        )
        return [self._parse_html_folder(folder, raw) for folder in folders[:retmax]]

    @staticmethod
    def _parse_html_folders(html: str) -> list[dict[str, str]]:
        pattern = r'href="(/Open-Systems-Pharmacology/OSP-PBPK-Model-Library/tree/master/[^"#?]+)"'
        seen: set[str] = set()
        folders: list[dict[str, str]] = []
        for href in re.findall(pattern, html):
            decoded_href = unquote(unescape(href))
            folder_name = decoded_href.rsplit("/", 1)[-1]
            if not folder_name or folder_name.startswith(".") or folder_name in seen:
                continue
            seen.add(folder_name)
            folders.append(
                {
                    "name": folder_name,
                    "path": folder_name,
                    "html_url": f"https://github.com{decoded_href}",
                }
            )
        return folders

    def _parse_html_folder(self, folder: dict[str, str], raw: RawDocumentMetadata) -> ParsedSourceRecord:
        record = {
            "model_folder_name": folder["name"],
            "model_folder_path": folder["path"],
            "github_html_url": folder["html_url"],
            "github_api_url": None,
            "file_count": None,
            "files": [],
            "has_evaluation_report_candidate": None,
            "source_version": self.source_version,
            "extraction_level": "model_folder_inventory_from_html",
        }
        return self.make_record(
            raw=raw,
            source_record_id=folder["path"],
            source_url=folder["html_url"],
            source_location=f"GITHUB_PATH:{folder['path']}",
            record_type="public_pbpk_model_folder",
            record=record,
        )

    def _parse_folder(self, folder: dict[str, Any], root_raw: RawDocumentMetadata) -> ParsedSourceRecord:
        folder_contents: list[dict[str, Any]] = []
        folder_raw = root_raw
        folder_url = folder.get("url")
        if folder_url:
            try:
                payload, content_type, status_code, raw_bytes = self.fetch_json(folder_url)
                if isinstance(payload, list):
                    folder_contents = payload
                folder_raw = self.save_raw_json(
                    source_url=folder_url,
                    payload=payload,
                    content_type=content_type,
                    status_code=status_code,
                    raw_record_count=len(folder_contents),
                    source_version=self.source_version,
                    retrieval_method="github_contents_model_folder",
                    extra_metadata={"folder": folder.get("path")},
                    raw_payload_bytes=raw_bytes,
                )
            except Exception as exc:
                self.errors.append(
                    ConnectorError(
                        stage="fetch_model_folder_contents",
                        message=str(exc),
                        retryable=True,
                        source_record_id=folder.get("path"),
                    )
                )

        file_names = [entry.get("name") for entry in folder_contents if entry.get("type") == "file"]
        record = {
            "model_folder_name": folder.get("name"),
            "model_folder_path": folder.get("path"),
            "github_html_url": folder.get("html_url"),
            "github_api_url": folder.get("url"),
            "file_count": len(file_names),
            "files": file_names[:50],
            "has_evaluation_report_candidate": any(
                "evaluation" in str(name).lower() or "report" in str(name).lower() for name in file_names
            ),
            "source_version": self.source_version,
            "extraction_level": "model_folder_inventory",
        }
        return self.make_record(
            raw=folder_raw,
            source_record_id=str(folder.get("path") or folder.get("name")),
            source_url=str(folder.get("html_url") or "https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"),
            source_location=f"GITHUB_PATH:{folder.get('path') or folder.get('name')}",
            record_type="public_pbpk_model_folder",
            record=record,
        )
