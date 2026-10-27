"""Recover verified project-owned model sources without rewriting raw provenance."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from pbpk_kb.paths import PROJECT_ROOT
from pbpk_kb.acquisition.comprehensive_raw_retriever import file_checksum, iso_now, write_json

MODEL_REPOSITORIES = {
    "caffeine": "Example_Caffeine", "dextromethorphan": "Dextromethorphan-Model",
    "metformin": "Metformin-Model", "repaglinide": "Repaglinide-Model",
    "theophylline": "Theophylline-Model", "warfarin": "Warfarin-Model",
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def atomic_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    temporary.replace(path)


def checked_path(value, root=PROJECT_ROOT):
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source recovery path escaped the project.")
    return path


def asset_record(path, *, compound, url, run_id, metadata, source="osp_pbpk_model_library"):
    digest = file_checksum(path)
    identity = sha256((str(path) + digest).encode()).hexdigest()
    asset = {
        "asset_id": "asset:verified:" + identity[:32], "request_id": source + ":verified:" + identity[:24],
        "request_key": "verified_model_source", "source_id": source, "compound": compound,
        "group": "shared/osp_pbpk_model_library/verified_target_models", "url": url,
        "storage_path": str(path), "metadata_path": str(path.with_suffix(path.suffix + ".metadata.json")),
        "checksum": digest, "byte_count": path.stat().st_size, "downloaded_at": iso_now(),
        "content_type": None, "storage_scope": "shared_source_cache", "snapshot_id": run_id,
        "raw_only": True, "metadata": metadata,
    }
    sidecar = Path(asset["metadata_path"])
    if sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous["checksum"] != digest:
            raise RuntimeError("Immutable model asset changed.")
        return previous
    write_json(sidecar, asset)
    return asset


def repair_model_snapshot(run_root: Path, output_root: Path, compounds: list[str]):
    """Register the pinned official model sources already downloaded by this project.

    No scientific validation/execution status is promoted by acquisition. A
    missing or changed cached asset blocks recovery rather than being inferred.
    """
    run_root, output_root = checked_path(run_root), checked_path(output_root)
    snapshot = output_root / "source_cache/osp_pbpk_model_library" / run_root.name / "verified"
    status_path = snapshot / "RUN_STATUS.json"
    existing_assets = rows(run_root / "assets.jsonl")
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        return json.loads((snapshot / "snapshot_manifest.json").read_text())
    manifests = []
    for compound in compounds:
        name = MODEL_REPOSITORIES.get(compound)
        if name is None:
            raise RuntimeError(f"No verified repository mapping exists for {compound}.")
        path = PROJECT_ROOT / "artifacts/pbpk-model-assets/coverage_expansion_20260907" / name / "asset_manifest.json"
        m = json.loads(path.read_text())
        if m["compound"] != compound or m["repository"] != "Open-Systems-Pharmacology/" + name:
            raise RuntimeError("Cached model repository identity mismatch.")
        if len(m["commit"]) != 40 or not m["files"]:
            raise RuntimeError("Model source lacks a pinned revision or files.")
        archive = path.parent / "source.tar.gz"
        if file_checksum(archive) != m["source_archive_sha256"]:
            raise RuntimeError("Model archive checksum mismatch.")
        for item in m["files"]:
            source = checked_path(PROJECT_ROOT / m["source_directory"] / item["path"])
            if file_checksum(source) != item["sha256"] or source.stat().st_size != item["bytes"]:
                raise RuntimeError("Cached model file failed checksum verification.")
        manifests.append(m)
    snapshot.mkdir(parents=True, exist_ok=True)
    model_assets = []
    for m in manifests:
        for item in m["files"]:
            source = checked_path(PROJECT_ROOT / m["source_directory"] / item["path"])
            destination = checked_path(snapshot / m["compound"] / m["commit"] / item["path"], snapshot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if file_checksum(destination) != item["sha256"]:
                    raise RuntimeError("Recovery destination contains different bytes.")
            else:
                shutil.copyfile(source, destination)
            model_assets.append(asset_record(destination, compound=m["compound"], run_id=run_root.name,
                url=f"https://raw.githubusercontent.com/{m['repository']}/{m['commit']}/{quote(item['path'], safe='/')}",
                metadata={"repository": m["repository"], "commit": m["commit"],
                          "repository_path": item["path"], "origin_manifest": m["source_url"],
                          "reused_verified_project_download": True,
                          "scientific_execution_status": "not_executed", "evidence_status": "raw_unvalidated"}))
    manifest = {
        "source_id": "osp_pbpk_model_library", "snapshot_id": run_root.name,
        "source_version": "pinned_official_target_repositories", "scope": "requested_compounds",
        "compounds": compounds, "repositories": [{k: m[k] for k in ("compound", "repository", "commit", "source_url")} for m in manifests],
        "asset_count": len(model_assets), "assets": model_assets,
        "asset_checksum_aggregate": sha256("\n".join(sorted(a["checksum"] for a in model_assets)).encode()).hexdigest(),
        "raw_only": True, "created_at": iso_now(),
    }
    manifest_path = snapshot / "snapshot_manifest.json"
    write_json(manifest_path, manifest)
    snapshot_asset = asset_record(manifest_path, compound=None, url="https://github.com/Open-Systems-Pharmacology",
                                  run_id=run_root.name, metadata={"scope": "requested_compounds"})
    # Preserve the full original index. Only duplicated metadata entries and the
    # overwritten generated OSP snapshot are excluded from the canonical index.
    repair_dir = run_root / "finalization/source_recovery"
    repair_dir.mkdir(parents=True, exist_ok=True)
    backup = repair_dir / "assets_before_repair.jsonl"
    if not backup.exists():
        shutil.copyfile(run_root / "assets.jsonl", backup)
    clean, excluded, seen = [], [], set()
    for asset in existing_assets:
        path = checked_path(asset["storage_path"])
        if asset["source_id"] == "osp_pbpk_model_library":
            reason = "duplicate_index_entry" if str(path) in seen else None
            if file_checksum(path) != asset["checksum"]:
                if asset.get("group") != "shared/snapshot_manifest":
                    raise RuntimeError("Unexpected source corruption outside the generated OSP snapshot.")
                reason = "overwritten_generated_snapshot_original_checksum_preserved"
            if reason:
                excluded.append({"reason": reason, "original_asset": asset})
                continue
        seen.add(str(path))
        clean.append(asset)
    additions = [a for a in model_assets + [snapshot_asset] if a["storage_path"] not in seen]
    atomic_rows(repair_dir / "excluded_asset_index_rows.jsonl", excluded)
    atomic_rows(run_root / "assets.jsonl", clean + additions)
    atomic_rows(snapshot / "assets.jsonl", model_assets + [snapshot_asset])
    write_json(repair_dir / "model_recovery_summary.json", {
        "original_asset_count": len(existing_assets), "excluded_duplicate_or_generated_rows": len(excluded),
        "model_files_registered": len(model_assets), "compounds": compounds,
        "snapshot_path": str(snapshot), "original_index_preserved": str(backup), "raw_only": True,
    })
    write_json(status_path, {"status": "completed", "accepted_as_final_result": True,
                            "scope": "requested_compounds", "completed_at": iso_now()})
    return manifest


def verified_pkdb_observation_gap(failure, assets, *, root=PROJECT_ROOT):
    """Recognize only an audited empty official export, never transport or integrity failures."""
    from urllib.parse import urlparse, parse_qs
    if failure.get("source_id") != "pkdb" or failure.get("request_key") != "open_bulk_export_incomplete":
        return None
    compound = failure.get("compound")
    url = urlparse(str(failure.get("url") or ""))
    query = parse_qs(url.query)
    if (url.scheme != "https" or url.hostname != "pk-db.com" or url.path != "/api/v1/filter/"
            or query.get("studies__substance") != [compound]
            or query.get("studies__licence__in") != ["open"] or query.get("download") != ["true"]):
        return None
    by_id = {a["asset_id"]: a for a in assets}
    for item in assets:
        if (item.get("compound") != compound or item.get("source_id") != "pkdb"
                or item.get("request_key") != "open_bulk_export_audit" or item.get("url") != failure.get("url")):
            continue
        path = checked_path(item["storage_path"], root)
        if file_checksum(path) != item["checksum"]:
            raise RuntimeError("PK-DB audit checksum mismatch.")
        audit = json.loads(path.read_text())
        archive = by_id.get(item.get("metadata", {}).get("export_asset_id"))
        if not archive or audit.get("accepted_for_observation_processing") is not False:
            continue
        if (archive.get("compound") != compound or archive.get("url") != failure.get("url")
                or audit.get("compound") != compound or audit.get("source_url") != failure.get("url")
                or archive.get("status_code") != 200):
            continue
        archive_path = checked_path(archive["storage_path"], root)
        if (file_checksum(archive_path) != archive.get("checksum")
                or archive["checksum"] != audit.get("archive_sha256")
                or archive_path.stat().st_size != audit.get("archive_byte_count")):
            raise RuntimeError("PK-DB archive differs from its audited bytes.")
        gaps = set(audit.get("gap_codes") or [])
        allowed = {"filter_export_count_mismatch", "outputs_observation_table_missing", "timecourses_observation_table_missing"}
        mismatches = audit.get("count_mismatches") or []
        if not gaps or gaps - allowed or not mismatches or audit.get("missing_required_members"):
            continue
        if any(r.get("resource") not in {"outputs", "timecourses"} or r.get("actual") != 0
               or not isinstance(r.get("expected"), int) or r["expected"] <= 0 for r in mismatches):
            continue
        tables = audit.get("table_audits") or {}
        if any(tables.get(r["resource"], {}).get("row_count") != 0 for r in mismatches):
            continue
        return {"request_id": failure["request_id"], "compound": compound,
                "status": "source_gap_metadata_only", "audit_asset_id": item["asset_id"],
                "audit_sha256": item["checksum"], "archive_asset_id": archive["asset_id"],
                "archive_sha256": archive["checksum"], "gap_codes": sorted(gaps),
                "count_mismatches": mismatches, "observations_recovered": False,
                "accepted_for_observation_processing": False}
    return None


def fetch_bounded_public_response(url, *, max_bytes=65536):
    """One bounded public lookup; no credentials, redirects or retry loop."""
    import requests
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, timeout=(5, 15), allow_redirects=False, stream=True) as response:
            blocks, size = [], 0
            for block in response.iter_content(16384):
                size += len(block)
                if size > max_bytes:
                    return response.status_code, b""
                blocks.append(block)
            return response.status_code, b"".join(blocks)


def fetch_openfda_lookup_response(url):
    return fetch_bounded_public_response(url)


def record_chembl_identity_recovery(raw, failure, assets, content, expected_inchikey, downloaded_at):
    """Register recovered raw bytes only when the exact molecule and stereochemistry match."""
    import re
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("molecules"), list):
        return None
    exact = [m for m in payload["molecules"] if isinstance(m, dict)
             and str(m.get("pref_name") or "").casefold() == failure["compound"].casefold()]
    if (len(exact) != 1 or not re.fullmatch(r"CHEMBL\d+", str(exact[0].get("molecule_chembl_id") or ""))
            or (exact[0].get("molecule_structures") or {}).get("standard_inchi_key") != expected_inchikey):
        return None
    key = sha256(failure["request_id"].encode()).hexdigest()[:24]
    directory = raw / "finalization/source_recovery/chembl_recovered" / key
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256(content).hexdigest()
    path = directory / (digest + ".json")
    if path.exists() and file_checksum(path) != digest:
        raise RuntimeError("Recovered ChEMBL bytes changed.")
    if not path.exists():
        path.write_bytes(content)
    asset = {"asset_id": "asset:chembl-recovered:" + key, "request_id": "chembl:recovered:" + key,
             "source_id": "chembl", "compound": failure["compound"], "request_key": "molecule_search_recovered",
             "group": "compound_api", "url": failure["url"], "storage_path": str(path),
             "metadata_path": str(path.with_suffix(".json.metadata.json")), "checksum": digest,
             "byte_count": len(content), "status_code": 200, "content_type": "application/json",
             "downloaded_at": downloaded_at, "raw_only": True, "storage_scope": "compound_run",
             "metadata": {"original_failure_request_id": failure["request_id"], "exact_identity_verified": True}}
    if not any(a.get("asset_id") == asset["asset_id"] for a in assets):
        write_json(Path(asset["metadata_path"]), asset)
        with (raw / "assets.jsonl").open("ab", buffering=0) as output:
            output.write((json.dumps(asset, ensure_ascii=False) + "\n").encode())
        assets.append(asset)
    write_json(directory / "receipt.json", {"original_failure_request_id": failure["request_id"],
               "url": failure["url"], "status_code": 200, "downloaded_at": downloaded_at,
               "checksum": digest, "storage_path": str(path), "byte_count": len(content)})
    proof = {"request_id": failure["request_id"], "compound": failure["compound"], "status": "resolved",
             "reason": "verified_recovered_chembl_identity", "response_asset_id": asset["asset_id"],
             "response_sha256": digest, "molecule_chembl_id": exact[0]["molecule_chembl_id"],
             "raw_lookup_recovered": True, "scientific_evidence_validated": False}
    resolutions = raw / "finalization/failure_resolutions.jsonl"
    if not any(r.get("original_failure_request_id") == failure["request_id"] for r in rows(resolutions)):
        with resolutions.open("ab", buffering=0) as output:
            output.write((json.dumps({"resolution_id": "resolution:chembl-recovered:" + key,
                "original_failure_request_id": failure["request_id"], "source_id": "chembl",
                "compound": failure["compound"], "original_request_key": failure["request_key"],
                "original_url": failure["url"], "status": "resolved", "resolution_method": proof["reason"],
                "detail": proof, "resolved_at": iso_now(), "raw_only": True}, ensure_ascii=False) + "\n").encode())
    return proof


def verified_chembl_service_gap(raw, failure, assets, *, root=PROJECT_ROOT):
    """Exclude a proven provider outage, only with intact alternative identity inputs.

    This is an unavailable source, never an empty molecule search or scientific
    evidence. Each failure receives at most one bounded recheck; saved receipts
    make monitor restarts idempotent.
    """
    from urllib.parse import urlparse, parse_qs, unquote
    if failure.get("source_id") != "chembl" or failure.get("request_key") != "molecule_search":
        return None
    compound = failure.get("compound")
    url = str(failure.get("url") or "")
    parsed = urlparse(url)
    if (not isinstance(compound, str) or not compound or parsed.scheme != "https"
            or parsed.netloc != "www.ebi.ac.uk" or parsed.path != "/chembl/api/data/molecule/search.json"
            or parse_qs(parsed.query) != {"q": [compound], "limit": ["25"], "offset": ["0"]}
            or failure.get("error") not in {f"HTTP Error {code}: {label}" for code, label in (
                (500, "Internal Server Error"), (502, "Bad Gateway"),
                (503, "Service Unavailable"), (504, "Gateway Timeout"))}):
        return None
    support = {}
    for source, key in (("pubchem", "synonyms_by_name"), ("pubchem", "properties_by_name"),
                        ("rxnorm", "rxcui_by_name")):
        item = next((a for a in assets if a.get("source_id") == source and a.get("compound") == compound
                     and a.get("request_key") == key and a.get("status_code") == 200), None)
        if not item:
            return None
        u = urlparse(item.get("url") or "")
        if source == "pubchem":
            suffix = "/synonyms/JSON" if key == "synonyms_by_name" else "/property/"
            valid_url = (u.scheme == "https" and u.netloc == "pubchem.ncbi.nlm.nih.gov"
                         and unquote(u.path).startswith(f"/rest/pug/compound/name/{compound}{suffix}"))
        else:
            valid_url = (u.scheme == "https" and u.netloc == "rxnav.nlm.nih.gov" and u.path == "/REST/rxcui.json"
                         and parse_qs(u.query) == {"name": [compound], "search": ["2"]})
        if not valid_url:
            return None
        path = checked_path(item["storage_path"], root)
        if file_checksum(path) != item["checksum"]:
            raise RuntimeError("Alternative identity input checksum mismatch.")
        support[key] = (item, json.loads(path.read_text()))
    identities = support["synonyms_by_name"][1].get("InformationList", {}).get("Information", [])
    properties = support["properties_by_name"][1].get("PropertyTable", {}).get("Properties", [])
    rxids = support["rxcui_by_name"][1].get("idGroup", {}).get("rxnormId", [])
    exact = [i for i in identities if compound.casefold() in {str(n).casefold() for n in i.get("Synonym", [])}]
    if (len(exact) != 1 or len(properties) != 1 or not exact[0].get("CID")
            or properties[0].get("CID") != exact[0]["CID"] or not properties[0].get("InChIKey") or not rxids):
        return None
    raw = checked_path(raw, root)
    key = sha256(failure["request_id"].encode()).hexdigest()[:24]
    recovered = raw / "finalization/source_recovery/chembl_recovered" / key / "receipt.json"
    if recovered.exists():
        receipt = json.loads(recovered.read_text())
        path = checked_path(receipt["storage_path"], raw)
        if (receipt.get("original_failure_request_id") != failure["request_id"] or receipt.get("url") != url
                or receipt.get("status_code") != 200 or file_checksum(path) != receipt.get("checksum")):
            raise RuntimeError("Recovered ChEMBL receipt failed integrity validation.")
        return record_chembl_identity_recovery(raw, failure, assets, path.read_bytes(),
                                               properties[0]["InChIKey"], receipt["downloaded_at"])
    directory = checked_path(raw / "finalization/source_recovery/chembl_unavailable" / key, root)
    response_path, metadata_path = directory / "response.bin", directory / "response.bin.metadata.json"
    if metadata_path.exists():
        asset = json.loads(metadata_path.read_text())
        content = response_path.read_bytes()
        if (asset.get("url") != url or asset.get("compound") != compound
                or asset.get("checksum") != sha256(content).hexdigest()
                or asset.get("processing_policy") != "metadata_only"):
            raise RuntimeError("ChEMBL outage receipt failed integrity validation.")
        status = asset.get("status_code")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        attempt = directory / "recheck.json"
        if attempt.exists():
            return None  # A failed/interrupted recheck cannot create an unbounded retry loop.
        write_json(attempt, {"request_id": failure["request_id"], "started_at": iso_now(), "max_attempts": 1})
        try:
            status, content = fetch_bounded_public_response(url, max_bytes=2 * 1024 * 1024)
        except Exception:
            return None
        if status == 200:
            return record_chembl_identity_recovery(raw, failure, assets, content, properties[0]["InChIKey"], iso_now())
        if status not in {500, 502, 503, 504} or not content:
            return None
        response_path.write_bytes(content)
        asset = {"asset_id": "asset:chembl-outage:" + key, "request_id": "chembl:outage:" + key,
                 "source_id": "chembl", "compound": compound, "request_key": "verified_service_unavailable",
                 "group": "identity/source_availability", "url": url, "storage_path": str(response_path),
                 "metadata_path": str(metadata_path), "checksum": sha256(content).hexdigest(),
                 "byte_count": len(content), "status_code": status, "downloaded_at": iso_now(),
                 "raw_only": True, "storage_scope": "compound_run", "processing_policy": "metadata_only",
                 "processing_policy_reason": "official_chembl_service_unavailable",
                 "metadata": {"original_failure_request_id": failure["request_id"]}}
        write_json(metadata_path, asset)
    if status not in {500, 502, 503, 504}:
        return None
    if not any(a.get("asset_id") == asset["asset_id"] for a in assets):
        with (raw / "assets.jsonl").open("ab", buffering=0) as output:
            output.write((json.dumps(asset, ensure_ascii=False) + "\n").encode())
        assets.append(asset)
    proof = {"request_id": failure["request_id"], "compound": compound, "status": "source_gap_metadata_only",
             "reason": "official_chembl_service_unavailable", "response_asset_id": asset["asset_id"],
             "response_sha256": asset["checksum"], "status_code": status, "retryable_in_future_run": True,
             "identity_support": [{"asset_id": a["asset_id"], "checksum": a["checksum"]} for a, _ in support.values()],
             "observations_recovered": False, "accepted_for_observation_processing": False,
             "scope": "ChEMBL unavailable; PubChem/RxNorm remain inputs to the unchanged downstream identity gates."}
    resolutions = raw / "finalization/failure_resolutions.jsonl"
    if not any(r.get("original_failure_request_id") == failure["request_id"] for r in rows(resolutions)):
        resolution = {"resolution_id": "resolution:chembl-outage:" + key,
                      "original_failure_request_id": failure["request_id"], "source_id": "chembl", "compound": compound,
                      "original_request_key": failure["request_key"], "original_url": url,
                      "status": "source_gap_metadata_only", "resolution_method": "verified_outage_with_alternative_identity_inputs",
                      "detail": proof, "resolved_at": iso_now(), "raw_only": True}
        with resolutions.open("ab", buffering=0) as output:
            output.write((json.dumps(resolution, ensure_ascii=False) + "\n").encode())
    return proof


def verified_openfda_download_recovery(raw, failure, assets, *, root=PROJECT_ROOT):
    """Prove complete label pagination and selection after a failed raw request.

    No network request is made here. Merely finding an HTTP 200 first page or a
    manually written resolution never resolves an incomplete acquisition.
    """
    import re
    from urllib.parse import parse_qs, urlparse
    from pbpk_kb.acquisition.comprehensive_raw_retriever import slugify, url_with_params
    from pbpk_kb.acquisition.precision import classify_openfda_label

    source = "openfda_drug_label"
    compound, failed_key = failure.get("compound"), failure.get("request_key")
    if (failure.get("source_id") != source or not isinstance(compound, str) or not compound
            or not isinstance(failed_key, str) or not re.fullmatch(r"label_skip_\d+", failed_key)):
        return None
    failed_skip = int(failed_key.removeprefix("label_skip_"))
    url = str(failure.get("url") or "")
    parsed = urlparse(url)
    if (failed_skip % 100 or parsed.scheme != "https" or parsed.netloc != "api.fda.gov"
            or parsed.path != "/drug/label.json" or parsed.fragment
            or parse_qs(parsed.query) != {"search": [f'openfda.generic_name:"{compound}"'],
                                          "limit": ["100"], "skip": [str(failed_skip)]}):
        return None
    expected_id = source + ":" + failed_key + ":" + sha256(
        f"{source}|{compound}|{failed_key}|{url}".encode()).hexdigest()[:16]
    if failure.get("request_id") != expected_id:
        return None
    raw = checked_path(raw, root)
    page_assets = {}
    for asset in assets:
        if (asset.get("source_id") == source and asset.get("compound") == compound
                and asset.get("status_code") == 200 and asset.get("asset_id")):
            page_assets[asset.get("request_key")] = asset
    failed_asset = page_assets.get(failed_key)
    if (not failed_asset or failed_asset.get("request_id") != expected_id
            or failed_asset.get("url") != url):
        return None
    selection_path = checked_path(raw / "compounds" / slugify(compound) / "selection" / source / "candidate_records.jsonl", raw)
    if not selection_path.is_file():
        return None
    selection_bytes = selection_path.read_bytes()
    selection_rows = [json.loads(line) for line in selection_bytes.splitlines() if line.strip()]
    selection_by_id = {}
    for item in selection_rows:
        selection_by_id.setdefault(item.get("source_record_id"), []).append(item)
    pages, seen_ids, skip, total = [], set(), 0, None
    while total is None or skip < total:
        key = f"label_skip_{skip}"
        asset = page_assets.get(key)
        page_url = url_with_params("https://api.fda.gov/drug/label.json", {
            "search": f'openfda.generic_name:"{compound}"', "limit": 100, "skip": skip})
        request_id = source + ":" + key + ":" + sha256(
            f"{source}|{compound}|{key}|{page_url}".encode()).hexdigest()[:16]
        if (not asset or asset.get("url") != page_url or asset.get("request_id") != request_id
                or asset.get("processing_policy") == "metadata_only"):
            return None
        path = checked_path(asset["storage_path"], raw)
        content = path.read_bytes()
        if (type(asset.get("byte_count")) is not int or len(content) != asset["byte_count"]
                or sha256(content).hexdigest() != asset.get("checksum")):
            raise RuntimeError("Recovered openFDA asset failed integrity validation.")
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        counts = (payload.get("meta") or {}).get("results") or {}
        current_total = counts.get("total")
        if (type(current_total) is not int or not 0 < current_total <= 100000
                or counts.get("skip") != skip or counts.get("limit") != 100
                or (total is not None and total != current_total)):
            return None
        total = current_total
        labels = payload.get("results")
        if not isinstance(labels, list) or len(labels) != min(100, total - skip):
            return None
        for index, label in enumerate(labels):
            if not isinstance(label, dict):
                return None
            label_id = str(label.get("id") or label.get("set_id") or f"skip_{skip}_row_{index}")
            if label_id in seen_ids:
                return None  # A moving/duplicated page cannot prove complete coverage.
            seen_ids.add(label_id)
            expected_record = {"label_id": label_id, "set_id": label.get("set_id"),
                               "openfda": label.get("openfda") or {}, "effective_time": label.get("effective_time")}
            decision = classify_openfda_label(compound, label)
            if not any(item.get("compound") == compound and item.get("source_id") == source
                       and item.get("request_id") == request_id and item.get("source_url") == page_url
                       and item.get("source_location") == f"label_id:{label_id}"
                       and item.get("record") == expected_record
                       and all(item.get(field) == decision.get(field) for field in (
                           "decision", "selection_reason", "identity_match_type"))
                       for item in selection_by_id.get(label_id, [])):
                return None
        pages.append({"request_id": request_id, "asset_id": asset["asset_id"],
                      "checksum": asset["checksum"], "byte_count": asset["byte_count"], "skip": skip})
        skip += 100
    if failed_skip >= total:
        return None
    proof = {"request_id": expected_id, "compound": compound, "status": "resolved",
             "reason": "verified_complete_openfda_pagination_and_selection", "raw_lookup_recovered": True,
             "scientific_evidence_validated": False, "page_count": len(pages), "label_count": len(seen_ids),
             "reported_total": total, "pages": pages,
             "selection_sha256": sha256(selection_bytes).hexdigest(), "selection_path": str(selection_path)}
    resolutions = raw / "finalization/failure_resolutions.jsonl"
    prior = [item for item in rows(resolutions) if item.get("original_failure_request_id") == expected_id]
    if prior:
        if len(prior) != 1 or prior[0].get("status") != "resolved" or prior[0].get("detail") != proof:
            raise RuntimeError("Recovered openFDA resolution no longer matches verified inputs.")
    else:
        resolutions.parent.mkdir(parents=True, exist_ok=True)
        resolution = {"resolution_id": "resolution:openfda-recovered:" + sha256(expected_id.encode()).hexdigest()[:24],
                      "original_failure_request_id": expected_id, "source_id": source, "compound": compound,
                      "original_request_key": failed_key, "original_url": url, "status": "resolved",
                      "resolution_method": proof["reason"], "detail": proof, "resolved_at": iso_now(), "raw_only": True}
        with resolutions.open("ab", buffering=0) as output:
            output.write((json.dumps(resolution, ensure_ascii=False) + "\n").encode())
    return proof


def verified_openfda_empty_lookup(raw, failure, assets, *, root=PROJECT_ROOT):
    """Retain a proven zero-match first-page query; a generic HTTP 404 is not proof."""
    from urllib.parse import urlparse, parse_qs
    if failure.get("source_id") != "openfda_drug_label" or failure.get("request_key") != "label_skip_0":
        return None
    compound = failure.get("compound")
    url = str(failure.get("url") or "")
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if (parsed.scheme != "https" or parsed.netloc != "api.fda.gov" or parsed.path != "/drug/label.json"
            or set(query) != {"search", "limit", "skip"}
            or query["search"] != [f'openfda.generic_name:"{compound}"']
            or query["skip"] != ["0"] or query["limit"] != ["100"]):
        return None
    raw = checked_path(raw, root)
    key = sha256(failure["request_id"].encode()).hexdigest()[:24]
    directory = checked_path(raw / "finalization/source_recovery/openfda_empty_search" / key, root)
    response_path = directory / "response.json"
    metadata_path = directory / "response.json.metadata.json"
    if metadata_path.exists():
        asset = json.loads(metadata_path.read_text())
        content = response_path.read_bytes()
        if (asset.get("url") != url or asset.get("compound") != compound
                or asset.get("checksum") != sha256(content).hexdigest()):
            raise RuntimeError("Saved openFDA empty lookup failed integrity validation.")
        status = asset.get("status_code")
    else:
        try:
            status, content = fetch_openfda_lookup_response(url)
        except Exception:
            return None
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError):
        return None
    if (status != 404 or not isinstance(payload, dict) or payload.get("results")
            or payload.get("error") != {"code": "NOT_FOUND", "message": "No matches found!"}):
        return None
    if not metadata_path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(content)
        asset = {"asset_id": "asset:empty-lookup:" + key, "request_id": "openfda:empty-lookup:" + key,
                 "source_id": "openfda_drug_label", "compound": compound,
                 "group": "openfda_drug_label/verified_empty_search", "request_key": "verified_empty_search",
                 "url": url, "storage_path": str(response_path), "metadata_path": str(metadata_path),
                 "checksum": sha256(content).hexdigest(), "byte_count": len(content),
                 "content_type": "application/json", "status_code": 404, "downloaded_at": iso_now(),
                 "raw_only": True, "storage_scope": "compound_run", "processing_policy": "metadata_only",
                 "processing_policy_reason": "exact_openfda_generic_name_query_has_no_matches",
                 "metadata": {"original_failure_request_id": failure["request_id"], "no_matches_for_exact_query": True}}
        write_json(metadata_path, asset)
    if not any(a.get("asset_id") == asset["asset_id"] for a in assets):
        # One O_APPEND write can coexist with the acquisition worker's ledger appends.
        with (raw / "assets.jsonl").open("ab", buffering=0) as output:
            output.write((json.dumps(asset, ensure_ascii=False) + "\n").encode())
        assets.append(asset)
    proof = {"request_id": failure["request_id"], "compound": compound,
             "status": "source_gap_metadata_only", "response_asset_id": asset["asset_id"],
             "response_sha256": asset["checksum"], "reason": "verified_exact_query_no_matches",
             "scope": "This exact generic-name query only; other drug names and sources remain available.",
             "observations_recovered": False, "accepted_for_observation_processing": False}
    resolutions = raw / "finalization/failure_resolutions.jsonl"
    if not any(r.get("original_failure_request_id") == failure["request_id"] for r in rows(resolutions)):
        resolution = {"resolution_id": "resolution:empty-lookup:" + key,
                      "original_failure_request_id": failure["request_id"], "source_id": failure["source_id"],
                      "compound": compound, "original_request_key": failure["request_key"], "original_url": url,
                      "status": "source_gap_metadata_only", "resolution_method": "verified_official_empty_search_response",
                      "detail": proof, "resolved_at": iso_now(), "raw_only": True}
        resolutions.parent.mkdir(parents=True, exist_ok=True)
        with resolutions.open("ab", buffering=0) as output:
            output.write((json.dumps(resolution, ensure_ascii=False) + "\n").encode())
    return proof


def record_unavailable_pkdb_observations(finalizer):
    """Keep audited incomplete source exports as metadata, never as observations."""
    assets = rows(finalizer.assets_path)
    by_id = {a["asset_id"]: a for a in assets}
    changed = False
    for failure in finalizer.historical_failures:
        if failure.get("source_id") != "pkdb" or failure.get("request_key") != "open_bulk_export_incomplete":
            continue
        compound = failure["compound"]
        audit_asset = next((a for a in assets if a.get("compound") == compound and a.get("request_key") == "open_bulk_export_audit"), None)
        if audit_asset is None:
            continue
        audit_path = checked_path(audit_asset["storage_path"])
        if file_checksum(audit_path) != audit_asset["checksum"]:
            raise RuntimeError("PK-DB audit checksum mismatch.")
        audit = json.loads(audit_path.read_text())
        archive_asset = by_id.get(audit_asset.get("metadata", {}).get("export_asset_id"))
        gaps = set(audit.get("gap_codes") or [])
        if not archive_asset or audit.get("accepted_for_observation_processing") is not False:
            continue
        if not gaps.intersection({"outputs_observation_table_missing", "timecourses_observation_table_missing"}):
            continue
        if file_checksum(checked_path(archive_asset["storage_path"])) != audit["archive_sha256"]:
            raise RuntimeError("PK-DB archive differs from its audited bytes.")
        for a in (archive_asset, audit_asset):
            a["processing_policy"] = "metadata_only"
            a["processing_policy_reason"] = "official_export_observation_tables_missing"
            a["accepted_for_observation_processing"] = False
        changed = True
        finalizer.record_resolution(failure, status="source_gap_metadata_only",
            resolution_method="verified_official_export_missing_observation_tables",
            detail={"audit_asset_id": audit_asset["asset_id"], "audit_path": str(audit_path),
                    "archive_sha256": audit["archive_sha256"], "gap_codes": sorted(gaps),
                    "observations_recovered": False, "accepted_for_observation_processing": False,
                    "alternative": "Extract only from independently retrieved PMC original articles; never impute missing curated values."})
    if changed:
        atomic_rows(finalizer.assets_path, assets)
