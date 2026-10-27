from __future__ import annotations

from typing import Any
from urllib.parse import quote

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ParsedSourceRecord


class HumanProteinAtlasConnector(BaseConnector):
    source_id = "hpa"
    parser_name = "hpa_json_parser.py"
    default_query = "pbpk_core_targets"

    PBPK_TARGET_GENES = [
        "CYP3A4",
        "CYP2C9",
        "CYP2C19",
        "CYP2D6",
        "CYP1A2",
        "ABCB1",
        "SLCO1B1",
        "SLCO1B3",
        "UGT1A1",
        "CES1",
    ]

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        genes = self._genes_for_query(query, retmax)
        records: list[ParsedSourceRecord] = []
        for gene_symbol in genes:
            url = f"https://www.proteinatlas.org/search/{quote(gene_symbol)}?format=json"
            payload, content_type, status_code, raw_bytes = self.fetch_json(url)
            hits = payload if isinstance(payload, list) else [payload]
            best = self._best_hit(hits, gene_symbol)
            raw = self.save_raw_json(
                source_url=url,
                payload=payload,
                content_type=content_type,
                status_code=status_code,
                raw_record_count=len(hits),
                retrieval_method="hpa_gene_search_json",
                extra_metadata={"query_gene": gene_symbol, "retmax": retmax},
                raw_payload_bytes=raw_bytes,
            )
            if best is None:
                continue
            record_payload = self._record_payload(best)
            records.append(
                self.make_record(
                    raw=raw,
                    source_record_id=str(record_payload.get("ensembl") or gene_symbol),
                    source_url=url,
                    source_location=f"HPA:GENE:{record_payload.get('gene') or gene_symbol}",
                    record_type="hpa_tissue_expression",
                    record=record_payload,
                )
            )
        return records

    @classmethod
    def _genes_for_query(cls, query: str, retmax: int) -> list[str]:
        cleaned = (query or "").strip()
        if cleaned.upper() in cls.PBPK_TARGET_GENES:
            return [cleaned.upper()]
        return cls.PBPK_TARGET_GENES[: max(1, retmax)]

    @staticmethod
    def _best_hit(hits: list[Any], gene_symbol: str) -> dict[str, Any] | None:
        candidates = [hit for hit in hits if isinstance(hit, dict)]
        for hit in candidates:
            if str(hit.get("Gene") or "").upper() == gene_symbol.upper():
                return hit
        return candidates[0] if candidates else None

    @staticmethod
    def _record_payload(hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "Gene": hit.get("Gene"),
            "Gene synonym": hit.get("Gene synonym") or [],
            "Gene description": hit.get("Gene description"),
            "Ensembl": hit.get("Ensembl"),
            "Uniprot": hit.get("Uniprot") or [],
            "RNA tissue specificity": hit.get("RNA tissue specificity"),
            "RNA tissue distribution": hit.get("RNA tissue distribution"),
            "RNA tissue specific nTPM": hit.get("RNA tissue specific nTPM"),
            "Protein tissue specificity": hit.get("Protein tissue specificity"),
            "Protein tissue distribution": hit.get("Protein tissue distribution"),
            "Protein tissue specific Intensity": hit.get("Protein tissue specific Intensity"),
            "Tissue expression cluster": hit.get("Tissue expression cluster"),
        }
