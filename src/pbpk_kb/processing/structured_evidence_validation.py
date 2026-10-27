"""Verify typed API facts against local source bytes, not sentence heuristics.

Only the explicitly mapped PubChem/ChEMBL numeric properties are handled here.
No network, database lookups, inferred identity, or trusted cached verdicts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERSION = "structured-api-property-source-v1"
FIELDS = {
    "pubchem": {
        "MolecularWeight": ("g/mol", "molar_mass", {"g/mol"}),
        "TPSA": ("angstrom^2", "area", {"angstrom^2", "angstrom2", "Å²", "Å^2", "A^2"}),
        "XLogP": ("dimensionless", "dimensionless", {"dimensionless", "1", "unitless"}),
    },
    "chembl": {
        "full_mwt": ("g/mol", "molar_mass", {"g/mol"}),
        "psa": ("angstrom^2", "area", {"angstrom^2", "angstrom2", "Å²", "Å^2", "A^2"}),
        "alogp": ("dimensionless", "dimensionless", {"dimensionless", "1", "unitless"}),
    },
}


class SourceValidationError(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SourceValidationError(reason)


def _identity(value: Any) -> str:
    # Do not strip salts, (+/-), stereochemistry, or active moieties to
    # manufacture a molecular match from a related name.
    return " ".join(str(value or "").casefold().split())


def _inside(value: Any) -> Path:
    path = Path(str(value)).resolve()
    _require(path.is_relative_to(PROJECT_ROOT.resolve()), "source_outside_project")
    return path


def _read_json(path: Path, limit: int = 32 * 1024 * 1024) -> tuple[Any, str, int]:
    path = _inside(path)
    _require(path.is_file() and path.stat().st_size <= limit, "source_missing_or_oversized")
    payload = path.read_bytes()
    _require(len(payload) <= limit, "source_missing_or_oversized")
    return json.loads(payload), hashlib.sha256(payload).hexdigest(), len(payload)


def _number(value: Any) -> float:
    _require(not isinstance(value, bool) and value is not None, "non_numeric_property")
    number = float(value)
    _require(math.isfinite(number), "non_finite_property")
    return number


def _registered_source(path: Path, source: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, digest, size = _read_json(path)
    sidecar_path = _inside(str(path) + ".metadata.json")
    sidecar, sidecar_sha, _ = _read_json(sidecar_path, 512 * 1024)
    _require(isinstance(payload, dict) and isinstance(sidecar, dict), "invalid_source_record")
    _require(sidecar.get("source_id") == source and sidecar.get("status_code") == 200,
             "unverified_source_response")
    _require(sidecar.get("checksum") == digest and sidecar.get("byte_count") == size,
             "source_checksum_or_size_mismatch")
    _require(_inside(sidecar.get("storage_path")) == path, "source_registration_path_mismatch")
    _require(bool(sidecar.get("asset_id")) and bool(_identity(sidecar.get("compound"))),
             "source_registration_identity_missing")
    return payload, sidecar, {"storage_path": str(path), "sha256": digest,
                              "metadata_sha256": sidecar_sha, "byte_count": size}


def _pubchem_request(sidecar: dict[str, Any], operation: str, field: str | None = None) -> str:
    url = urlsplit(str(sidecar.get("url") or ""))
    parts = unquote(url.path).split("/")
    _require(url.scheme == "https" and url.netloc == "pubchem.ncbi.nlm.nih.gov"
             and not url.query and not url.fragment and parts[:5] == ["", "rest", "pug", "compound", "name"]
             and len(parts) == (9 if operation == "property" else 8)
             and parts[6] == operation and parts[-1] == "JSON", "unsupported_source_request")
    name = parts[5]
    _require(_identity(name) == _identity(sidecar["compound"]), "request_compound_mismatch")
    if field is not None:
        _require(field in parts[7].split(","), "property_not_requested")
    return name


def _pubchem_identity(path: Path, cid: Any, target: str) -> tuple[set[str], dict[str, Any]]:
    # A property response contains no name. Bind its CID to a separately saved,
    # checksum-verified synonym response from the same compound acquisition.
    matches = []
    for sibling in sorted(path.parent.glob("synonyms_by_name_*")):
        if sibling.name.endswith(".metadata.json"):
            continue
        try:
            payload, metadata, audit = _registered_source(sibling, "pubchem")
            queried = _pubchem_request(metadata, "synonyms")
            _require(_identity(queried) == _identity(target), "synonym_query_mismatch")
            records = payload.get("InformationList", {}).get("Information", [])
            for record in records:
                if str(record.get("CID")) != str(cid):
                    continue
                names = {_identity(name) for name in record.get("Synonym", []) if name}
                if _identity(target) in names:
                    matches.append((names, audit))
        except (SourceValidationError, OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    _require(bool(matches), "pubchem_cid_name_not_proven")
    # Every matching proof has the requested CID. Duplicate identical downloads
    # do not introduce an alternative molecular identity.
    return matches[0]


def _chembl_names(molecule: dict[str, Any]) -> set[str]:
    names = {_identity(molecule.get("pref_name")), _identity(molecule.get("molecule_chembl_id"))}
    for synonym in molecule.get("molecule_synonyms") or []:
        if isinstance(synonym, dict):
            names.update(_identity(synonym.get(key)) for key in ("molecule_synonym", "synonyms"))
        elif isinstance(synonym, str):
            names.add(_identity(synonym))
    return names - {""}


def _chembl_identity_disambiguation(path: Path, target: str, molecule: dict[str, Any], molecules: list) -> dict[str, Any]:
    """Resolve shared aliases using an independently name-bound exact structure.

    A salt or stereoisomer can legitimately have the same loose ChEMBL synonym.
    A matching *full* InChIKey, including stereochemistry, is required here.
    """
    directory = _inside(path.parent.parent / "pubchem")
    proofs = []
    for sibling in sorted(directory.glob("properties_by_name_*")):
        if sibling.name.endswith(".metadata.json"):
            continue
        try:
            payload, metadata, audit = _registered_source(sibling, "pubchem")
            _require(_identity(_pubchem_request(metadata, "property", "InChIKey")) == _identity(target),
                     "cross_source_target_mismatch")
            props = payload["PropertyTable"]["Properties"]
            _require(isinstance(props, list) and len(props) == 1, "ambiguous_cross_source_identity")
            prop = props[0]
            key, cid = prop.get("InChIKey"), prop.get("CID")
            _require(isinstance(key, str) and bool(re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key))
                     and isinstance(cid, int) and not isinstance(cid, bool) and cid > 0, "invalid_cross_source_identity")
            _, synonyms = _pubchem_identity(sibling, cid, target)
            proofs.append({"pubchem_cid": cid, "inchi_key": key, "source": audit, "identity_source": synonyms})
        except (SourceValidationError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            continue
    _require(bool(proofs) and len({(p["pubchem_cid"], p["inchi_key"]) for p in proofs}) == 1,
             "ambiguous_chembl_target_identity")
    proof = proofs[0]
    expected_key = proof["inchi_key"]
    _require((molecule.get("molecule_structures") or {}).get("standard_inchi_key") == expected_key,
             "chembl_cross_source_structure_mismatch")
    matching_ids = {item.get("molecule_chembl_id") for item in molecules if isinstance(item, dict)
                    and _identity(target) in _chembl_names(item)
                    and (item.get("molecule_structures") or {}).get("standard_inchi_key") == expected_key}
    _require(matching_ids == {molecule["molecule_chembl_id"]}, "ambiguous_chembl_structure_identity")
    return proof


def validate_structured_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    source, field = row.get("source_id"), row.get("parameter")
    if row.get("extraction_method") != "source_structured_field" or field not in FIELDS.get(source, {}):
        return None
    audit: dict[str, Any] = {"version": VERSION, "source_id": source, "field": field}
    try:
        _require(row.get("evidence_type") == "physicochemical" and row.get("compound_role") == "target_compound",
                 "property_evidence_type_or_role_mismatch")
        expected_mapper = "pubchem_property_mapper" if source == "pubchem" else "chembl_exact_molecule_mapper"
        _require(row.get("extractor_name") == expected_mapper, "unrecognized_property_mapper")
        metadata = row.get("metadata_json") or {}
        _require(isinstance(metadata, dict) and metadata.get("json_field") == field, "property_mapping_mismatch")
        path = _inside(row.get("storage_path"))
        payload, sidecar, source_audit = _registered_source(path, source)
        audit["source"] = source_audit
        _require(row.get("checksum") == source_audit["sha256"]
                 and row.get("raw_asset_id") == sidecar["asset_id"]
                 and row.get("source_url") == sidecar["url"], "candidate_source_identity_mismatch")
        target = str(sidecar["compound"])
        prefix = f"{source}:{path.name}#json_path="
        location = str(row.get("source_location") or "")
        _require(location.startswith(prefix), "property_source_location_mismatch")
        pointer = location[len(prefix):]
        if source == "pubchem":
            _pubchem_request(sidecar, "property", field)
            match = re.fullmatch(r"\$\.PropertyTable\.Properties\[(\d+)\]\." + re.escape(field), pointer)
            _require(match is not None, "unsupported_property_json_path")
            props = payload["PropertyTable"]["Properties"]
            _require(isinstance(props, list) and len(props) == 1, "ambiguous_pubchem_property_identity")
            prop = props[int(match[1])]
            cid = prop.get("CID")
            _require(isinstance(cid, int) and not isinstance(cid, bool) and cid > 0
                     and str(cid) == str(metadata.get("pubchem_cid")), "pubchem_cid_mismatch")
            names, identity_audit = _pubchem_identity(path, cid, target)
            audit.update({"identity": {"pubchem_cid": cid, "matched_query_name": target},
                          "identity_source": identity_audit})
            raw_value = prop[field]
        else:
            url = urlsplit(str(sidecar["url"]))
            _require(url.scheme == "https" and url.netloc == "www.ebi.ac.uk"
                     and url.path == "/chembl/api/data/molecule/search.json" and not url.fragment
                     and parse_qs(url.query).get("q") == [target], "unsupported_source_request")
            match = re.fullmatch(r"\$\.molecules\[(\d+)\]\.molecule_properties\." + re.escape(field), pointer)
            _require(match is not None, "unsupported_property_json_path")
            molecule = payload["molecules"][int(match[1])]
            molecule_id = molecule.get("molecule_chembl_id")
            _require(isinstance(molecule_id, str) and bool(re.fullmatch(r"CHEMBL\d+", molecule_id))
                     and molecule_id == metadata.get("molecule_chembl_id")
                     and metadata.get("chembl_identity_gate") == "exact", "chembl_molecule_identity_mismatch")
            names = _chembl_names(molecule)
            _require(_identity(target) in names, "chembl_target_name_not_proven")
            matching_ids = {item.get("molecule_chembl_id") for item in payload["molecules"]
                            if isinstance(item, dict) and _identity(target) in _chembl_names(item)}
            if matching_ids != {molecule_id}:
                audit["cross_source_identity"] = _chembl_identity_disambiguation(path, target, molecule, payload["molecules"])
            audit["identity"] = {"molecule_chembl_id": molecule_id, "matched_query_name": target}
            raw_value = molecule["molecule_properties"][field]
        row_names = {str(row.get(key)) for key in ("compound_name", "attributed_compound_name", "compound", "retrieval_compound")
                     if row.get(key)}
        _require(bool(row_names) and all(_identity(name) in names for name in row_names), "candidate_compound_name_mismatch")
        number = _number(raw_value)
        _require(all(row.get(key) is None for key in ("value_low", "value_high", "canonical_value_low", "canonical_value_high")),
                 "scalar_property_has_range")
        for key in ("value", "canonical_value"):
            if key == "value" or row.get(key) is not None:
                _require(math.isclose(_number(row.get(key)), number, rel_tol=1e-12, abs_tol=1e-12), "property_numeric_value_mismatch")
        source_text = str(row.get("source_text") or "")
        _require(source_text == str(raw_value) and hashlib.sha256(source_text.encode()).hexdigest() == row.get("source_text_hash"),
                 "property_verbatim_value_mismatch")
        if row.get("value_text") is not None:
            _require(math.isclose(_number(row["value_text"]), number, rel_tol=1e-12, abs_tol=1e-12), "property_value_text_mismatch")
        unit, dimension, aliases = FIELDS[source][field]
        for key in ("unit", "canonical_unit"):
            _require(row.get(key) in (None, "") or str(row[key]).strip() in aliases, "property_unit_conflicts_with_schema")
        audit.update({"accepted": True, "json_path": pointer, "source_value": raw_value,
                      "unit": unit, "dimension": dimension, "unit_basis": "named_provider_property_schema"})
        return {"accepted": True, "unit": unit, "dimension": dimension, "audit": audit}
    except (SourceValidationError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        reason = str(exc) if isinstance(exc, SourceValidationError) else "source_record_unreadable_or_invalid"
        audit.update({"accepted": False, "reason": reason})
        return {"accepted": False, "error": "structured_property_" + reason, "audit": audit}
