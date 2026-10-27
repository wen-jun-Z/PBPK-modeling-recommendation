from __future__ import annotations

from typing import Any
from urllib.parse import quote

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class PubChemConnector(BaseConnector):
    source_id = "pubchem"
    parser_name = "json_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        encoded_query = quote(query)
        property_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded_query}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey/JSON"
        )
        property_payload, content_type, status_code, property_bytes = self.fetch_json(property_url)
        properties = property_payload.get("PropertyTable", {}).get("Properties", [])
        property_raw = self.save_raw_json(
            source_url=property_url,
            payload=property_payload,
            content_type=content_type,
            status_code=status_code,
            raw_record_count=len(properties),
            retrieval_method="pubchem_pug_rest_property",
            extra_metadata={"query": query},
            raw_payload_bytes=property_bytes,
        )

        synonyms_by_cid: dict[str, list[str]] = {}
        try:
            synonyms_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/synonyms/JSON"
            synonyms_payload, synonyms_content_type, synonyms_status_code, synonyms_bytes = self.fetch_json(synonyms_url)
            synonym_infos = synonyms_payload.get("InformationList", {}).get("Information", [])
            self.save_raw_json(
                source_url=synonyms_url,
                payload=synonyms_payload,
                content_type=synonyms_content_type,
                status_code=synonyms_status_code,
                raw_record_count=len(synonym_infos),
                retrieval_method="pubchem_pug_rest_synonyms",
                extra_metadata={"query": query},
                raw_payload_bytes=synonyms_bytes,
            )
            for info in synonym_infos:
                cid = str(info.get("CID"))
                synonyms_by_cid[cid] = list(info.get("Synonym", []))[:25]
        except Exception as exc:
            self.errors.append(
                ConnectorError(
                    stage="fetch_synonyms",
                    message=str(exc),
                    retryable=True,
                    source_record_id=None,
                )
            )

        return [self._parse_property_record(item, property_raw, synonyms_by_cid) for item in properties[:retmax]]

    def _parse_property_record(
        self,
        item: dict[str, Any],
        raw: RawDocumentMetadata,
        synonyms_by_cid: dict[str, list[str]],
    ) -> ParsedSourceRecord:
        cid = str(item.get("CID"))
        record = {
            "pubchem_cid": cid,
            "molecular_formula": item.get("MolecularFormula"),
            "molecular_weight": item.get("MolecularWeight"),
            "canonical_smiles": item.get("CanonicalSMILES"),
            "inchi_key": item.get("InChIKey"),
            "synonyms": synonyms_by_cid.get(cid, []),
            "extraction_level": "compound_identity_and_properties",
        }
        return self.make_record(
            raw=raw,
            source_record_id=cid,
            source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            source_location=f"CID:{cid}",
            record_type="compound_metadata",
            record=record,
        )
