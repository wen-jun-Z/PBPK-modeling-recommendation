"""Bind explicitly planned spelling aliases at import without rewriting evidence.

The frozen expansion target is necessary but not sufficient: the downloaded
PubChem synonym record must identify both spellings under the same exact CID.
No generic fuzzy matching, salt merging, or modification of source artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
STATE = Path("runtime/deployment/evidence_expansion_20261009/status.json")
TARGETS = Path("data_process/knowledge_expansion_20261009/targets.json")
ID_FIELDS = {"compound_id", "attributed_compound_id", "victim_compound_id", "perpetrator_compound_id"}
JSON_FIELDS = {"source_record", "metadata_json", "llm_payload"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _name(value) -> str:
    return str(value or "").strip().casefold()


class ExpansionIdentityMapping:
    def __init__(self, receipt: dict | None = None):
        self.receipt = receipt or {"version": 1, "mappings": []}
        self.ids = {row["original_compound_id"]: row["registered_compound_id"]
                    for row in self.receipt["mappings"]}

    def record(self, value: Any):
        """Transform identity fields only, including structured SQL JSON columns."""
        if not self.ids:
            return value
        if isinstance(value, list):
            return [self.record(item) for item in value]
        if not isinstance(value, dict):
            return value
        mapped = {}
        for key, item in value.items():
            if key in ID_FIELDS and item is not None:
                mapped[key] = self.ids.get(str(item), item)
            elif key in JSON_FIELDS and isinstance(item, str):
                structured = json.loads(item)
                mapped[key] = json.dumps(self.record(structured), ensure_ascii=False)
            else:
                mapped[key] = self.record(item)
        return mapped


def _source_proof(identity, target, inventory, root: Path, *, registered_alias: str | None = None) -> dict:
    canonical, search = _name(target["compound_name"]), _name(target["search_name"])
    explicit = {_name(n) for n in target.get("aliases", [])}
    if registered_alias is None:
        if canonical not in explicit or search not in explicit:
            raise RuntimeError("Identity remap requires both spellings in the frozen explicit aliases")
        required_synonyms = {canonical, search}
    else:
        # A frozen target can use the same canonical/search spelling while its
        # existing database ID uses a separately listed alias. Prove that exact
        # pair, rather than folding punctuation across arbitrary chemical names.
        if canonical != search or registered_alias not in explicit or registered_alias == search:
            raise RuntimeError("Registered spelling is not an explicit frozen target alias")
        required_synonyms = {search, registered_alias}
    cid = str(identity.get("pubchem_cid") or "")
    if not cid or len(set(map(str, identity.get("pubchem_cids") or [cid]))) != 1:
        raise RuntimeError("Identity remap requires one exact PubChem CID")
    assertions = identity.get("identity_provenance") or []
    for assertion in assertions:
        if assertion.get("source_id") != "pubchem" or assertion.get("assertion_type") != "pubchem_synonym":
            continue
        if _name(assertion.get("assertion_value")) != canonical:
            continue
        asset = inventory.get(assertion.get("raw_asset_id"))
        if not asset or asset.get("source_id") != "pubchem":
            continue
        source = Path(str(asset.get("storage_path") or "")).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise RuntimeError("Alias identity proof is outside the project or missing")
        checksum = _sha(source)
        if checksum != asset.get("checksum") or checksum != assertion.get("checksum"):
            raise RuntimeError("Alias identity source checksum mismatch")
        entries = (_read(source).get("InformationList") or {}).get("Information") or []
        proven = [row for row in entries if str(row.get("CID")) == cid
                  and required_synonyms <= {_name(n) for n in row.get("Synonym", [])}]
        if len(proven) != 1:
            continue
        return {"source_id": "pubchem", "pubchem_cid": cid,
                "raw_asset_id": asset["raw_asset_id"], "source_path": str(source.relative_to(root)),
                "sha256": checksum, "canonical_alias": canonical, "search_alias": search,
                **({"registered_alias": registered_alias} if registered_alias is not None else {})}
    raise RuntimeError("Frozen alias lacks source-verified same-CID synonym proof")


def load_expansion_identity_mapping(
    final_dir: Path | None, raw_dir: Path,
    registry_lookup: Callable[[list[str]], list[dict]], *, root: Path = ROOT,
) -> ExpansionIdentityMapping:
    """Return an empty mapping outside this explicitly frozen expansion queue.

    registry_lookup must read only the project's registered compound identities.
    The returned receipt records every binding and the untouched input hashes.
    """
    root = root.resolve()
    state_path, targets_path = root / STATE, root / TARGETS
    if final_dir is None or not state_path.is_file():
        return ExpansionIdentityMapping()
    final_dir, raw_dir = final_dir.resolve(), raw_dir.resolve()
    state = _read(state_path)
    batches = [row for row in state.get("planned_batches", []) if row.get("run_id") == raw_dir.name]
    if not batches:
        return ExpansionIdentityMapping()
    if (len(batches) != 1 or final_dir != root / "data_process/final_evidence" / raw_dir.name
            or raw_dir != root / "data_process/raw_processing" / raw_dir.name):
        raise RuntimeError("Expansion identity mapping run paths are inconsistent")
    expected = (state.get("frozen_inputs") or {}).get(str(TARGETS))
    if not expected or _sha(targets_path) != expected:
        raise RuntimeError("Frozen expansion targets changed before identity binding")
    plan = _read(targets_path)
    batch = batches[0]
    targets = [row for row in plan["targets"] if row["batch_id"] == batch["batch_id"]]
    if [row["search_name"] for row in targets] != batch["compounds"]:
        raise RuntimeError("Expansion identity target order differs from the registered batch")
    identity_path = final_dir / "01_compound_attribution/compound_identity.jsonl"
    identities = _rows(identity_path)
    prepared = []
    destinations = set()
    for target in targets:
        expected_id = target.get("existing_compound_id")
        if not expected_id:
            continue
        matches = [row for row in identities if _name(row.get("preferred_name")) == _name(target["search_name"])]
        if len(matches) != 1:
            raise RuntimeError("Expansion target has no unique source identity")
        identity = matches[0]
        original_id = str(identity["compound_id"])
        if original_id == str(expected_id):
            continue
        if str(expected_id) in destinations:
            raise RuntimeError("Multiple source identities target one registered compound")
        destinations.add(str(expected_id))
        prepared.append((identity, target))
    if not prepared:
        return ExpansionIdentityMapping()
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    inventory = {row["raw_asset_id"]: row for row in _rows(raw_processing_layout(raw_dir).path("raw_inventory.jsonl"))}
    lookup_ids = sorted({str(identity["compound_id"]) for identity, _ in prepared}
                        | {str(target["existing_compound_id"]) for _, target in prepared})
    registered = registry_lookup(lookup_ids)
    registry = {str(row["compound_id"]): row for row in registered}
    if len(registry) != len(registered):
        raise RuntimeError("Duplicate registered compound identity")
    mappings = []
    for identity, target in prepared:
        original, destination = str(identity["compound_id"]), str(target["existing_compound_id"])
        if original in registry:
            raise RuntimeError("Alias source identity already exists; automatic merging is forbidden")
        current = registry.get(destination)
        if not current:
            raise RuntimeError("Frozen registered target identity is missing or changed")
        registered_alias = None
        current_names = {_name(current.get("preferred_name")), _name(current.get("normalized_name"))} - {""}
        canonical, search = _name(target["compound_name"]), _name(target["search_name"])
        if canonical not in current_names:
            aliases = current_names & {_name(n) for n in target.get("aliases", [])}
            if canonical != search or len(aliases) != 1:
                raise RuntimeError("Frozen registered target identity is missing or changed")
            registered_alias = next(iter(aliases))
        for old, new in (("pubchem_cid", "pubchem_cid"), ("chembl_id", "chembl_id"), ("inchi_key", "inchikey")):
            if current.get(old) and identity.get(new) and str(current[old]) != str(identity[new]):
                raise RuntimeError("Registered alias stable identifier conflict")
        proof = _source_proof(identity, target, inventory, root, registered_alias=registered_alias)
        mappings.append({"original_compound_id": original, "registered_compound_id": destination,
                         "original_preferred_name": identity["preferred_name"],
                         "registered_preferred_name": current["preferred_name"],
                         "frozen_target_name": target["compound_name"], "proof": proof})
    return ExpansionIdentityMapping({"version": 1, "run_id": raw_dir.name,
        "method": "frozen_explicit_alias_and_source_verified_pubchem_synonym",
        "targets_path": str(TARGETS), "targets_sha256": expected,
        "source_identity_sha256": _sha(identity_path), "mappings": mappings})


def lookup_registered_identities(database_url: str, ids: list[str]) -> list[dict]:
    """No writes; this URL is supplied by the existing project DB verifier."""
    from sqlalchemy import create_engine, text
    engine = create_engine(database_url, connect_args={"connect_timeout": 5,
        "options": "-c default_transaction_read_only=on -c statement_timeout=30000"})
    try:
        with engine.connect() as connection:
            rows = connection.execute(text("SELECT compound_id::text,preferred_name,normalized_name,"
                "pubchem_cid,chembl_id,inchi_key FROM compound_identity WHERE compound_id::text=ANY(:ids)"),
                {"ids": ids}).mappings().all()
            connection.rollback()
            return [dict(row) for row in rows]
    finally:
        engine.dispose()
