"""Keep multiple frozen producer generations without relabelling old results.

Each unchanged record selects one immutable profile. That profile preserves its
own original producer bytes and registers only the explicitly reviewed resource
or readback changes leading to the current code. Native numerical worker drift,
unregistered records and cross-profile substitutions remain forbidden.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = "experiments/model_recommendation/execution_validation_20261001"
REGISTRY = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_bridge_followup_20261002/dual_dose_followup/installation/retained_registry.json'
REGISTRY_SHA = '9b3509e1a3aef2108354780800a9e7e13d6836b5b61fdd011a60b293e8461a65'
# Controller compatibility: these names now designate the immutable registry.
MANIFEST = REGISTRY
MANIFEST_SHA = REGISTRY_SHA
REGISTRY_SCHEMA = "pbpk-retained-execution-registry-v2"
SCHEMA = "pbpk-retained-execution-profile-v2"
REVISION = "engineering_repair_04"
PRODUCER_PATHS = frozenset({
    'scripts/recommendation_execution_repairs/json_racemate_contract.py',
    "scripts/recommendation_execution_repairs/coadministration_isolation.py",
    "scripts/recommendation_candidate_executor.py",
    "scripts/recommendation_execution_repairs/complete_candidate.py",
    "scripts/recommendation_execution_repairs/snapshot_candidate.py",
    "scripts/recommendation_execution_repairs/native_pkml.py",
    "src/pbpk_kb/blueprints/design_runtime.py",
    "src/pbpk_kb/blueprints/source_adaptation_audit.py",
    "src/pbpk_kb/blueprints/source_adaptation.py",
    # Only the registry-bound failure-receipt integrity revision is admitted.
    "scripts/recommendation_execution_repairs/historical_source_proofs.py",
})
ADAPTER_PATHS = ("scripts/recommendation_execution_repairs/json_native_candidate.py",
                 "scripts/recommendation_execution_repairs/snapshot_candidate.py",
                 "scripts/recommendation_execution_repairs/complete_candidate.py")
RECORD_NAMES = {"plan": "execution_plan.json", "result": "result.json",
                "prediction_lock": "prediction_lock.json", "prediction": "prediction.json"}

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError("Retained implementation: " + message)


def _sha_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value, root):
    _require(isinstance(value, str) and bool(value), "invalid resource path")
    item = Path(value)
    _require(not item.is_absolute() and item.as_posix() == value and ".." not in item.parts,
             "resource path is not canonical")
    _require(not any(part.casefold().startswith("sealed") for part in item.parts),
             "sealed resources are forbidden")
    path = (root / item).resolve(strict=True)
    _require(path.is_relative_to(root) and path != root, "resource outside project")
    _require(not any(part.casefold().startswith("sealed") for part in path.relative_to(root).parts),
             "sealed resources are forbidden")
    return path


def _checked(ref, root):
    _require(isinstance(ref, dict) and _sha_value(ref.get("sha256")), "invalid resource checksum")
    path = _path(ref.get("path"), root)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    _require(actual == ref["sha256"], "resource checksum differs: " + ref["path"])
    return path


def _adapter_path(implementation):
    # Snapshot imports the complete adapter as a helper.
    return next((path for path in ADAPTER_PATHS if path in implementation), None)


def _dependency_sets(record, preserved):
    declarations = record.get("adapter_dependencies")
    if declarations is None:
        return {}
    _require(isinstance(declarations, dict) and bool(declarations)
             and set(declarations) <= set(ADAPTER_PATHS), "invalid adapter dependency declarations")
    for adapter, declaration in declarations.items():
        _require(isinstance(declaration, dict) and set(declaration) ==
                 {"producer_sha256", "implementation_sha256"}, "invalid adapter dependency fields")
        implementation = declaration.get("implementation_sha256")
        _require(isinstance(implementation, dict) and bool(implementation)
                 and _adapter_path(implementation) == adapter, "historical adapter identity differs")
        _require(_sha_value(declaration.get("producer_sha256"))
                 and implementation.get(adapter) == declaration["producer_sha256"],
                 "historical adapter producer hash differs")
        for name, checksum in implementation.items():
            _require(isinstance(name, str) and _sha_value(checksum)
                     and name in preserved and preserved[name]["original"]["sha256"] == checksum,
                     "historical dependency differs from producer archive")
    return declarations


def _profile(root, ref, profile_id):
    record = json.loads(_checked(ref, root).read_text())
    _require(record.get("schema_version") == SCHEMA and record.get("execution_revision") == REVISION,
             "manifest schema/revision differs")
    _require(record.get("profile_id") == profile_id, "profile identity differs from registry")
    _require(record.get("sealed_observations_accessed") is False,
             "manifest outcome barrier missing")
    changes = record.get("producer_changes")
    _require(isinstance(changes, list) and 1 <= len(changes) <= len(PRODUCER_PATHS),
             "bounded registered producer changes required")
    by_change = {r.get("path"): r for r in changes if isinstance(r, dict)}
    _require(set(by_change) <= PRODUCER_PATHS and len(by_change) == len(changes),
             "unregistered or duplicate producer change")
    for change in changes:
        _require(_sha_value(change.get("before_sha256")) and _sha_value(change.get("after_sha256"))
                 and change["before_sha256"] != change["after_sha256"], "invalid producer change hashes")
    archive = record.get("code_archive")
    _require(isinstance(archive, list) and bool(archive), "producer archive missing")
    preserved = {}; archive_paths = set()
    for entry in archive:
        _require(isinstance(entry, dict), "invalid archive entry")
        old, saved = entry.get("original", {}), entry.get("preserved", {})
        name = old.get("path")
        _require(isinstance(name, str) and name not in preserved and _sha_value(old.get("sha256")),
                 "duplicate or invalid original producer")
        _require(not Path(name).is_absolute() and Path(name).as_posix() == name
                 and ".." not in Path(name).parts, "invalid original producer path")
        _require(saved.get("path") != name and saved.get("sha256") == old["sha256"],
                 "archive does not preserve original bytes")
        saved_path = _checked(saved, root)
        _require(saved_path not in archive_paths, "duplicate preserved producer path")
        archive_paths.add(saved_path); preserved[name] = entry
    preparation_ref = record.get("preparation")
    preparation = json.loads(_checked(preparation_ref, root).read_text())
    _require(preparation.get("execution_revision") == REVISION
             and preparation.get("sealed_outcomes_accessed") is False,
             "historical preparation revision/outcome barrier differs")
    old_code = preparation.get("code", [])
    _require(isinstance(old_code, list) and all(isinstance(r, dict) for r in old_code),
             "historical preparation code is invalid")
    code_map = {r.get("path"): r.get("sha256") for r in old_code}
    _require(len(code_map) == len(old_code), "historical preparation has duplicate code")
    for name, entry in preserved.items():
        _require(code_map.get(name) == entry["original"]["sha256"],
                 "archive differs from historical preparation: " + name)
    for name, change in by_change.items():
        _require(name in preserved and preserved[name]["original"]["sha256"] == change["before_sha256"],
                 "producer before-hash differs from archive: " + name)
    dependencies = _dependency_sets(record, preserved)
    jobs = record.get("retained_jobs")
    _require(isinstance(jobs, list), "retained job registrations missing")
    seen_jobs = set(); seen_pairs = set(); seen_files = set()
    for job in jobs:
        _require(isinstance(job, dict), "invalid retained job")
        jid, sid, cid = (job.get(k) for k in ("job_id", "scenario_id", "candidate_id"))
        _require(all(isinstance(x, str) and bool(x) for x in (jid, sid, cid))
                 and re.fullmatch(r"[A-Za-z0-9_-]+", jid) is not None,
                 "invalid retained job identity")
        _require(jid not in seen_jobs and (sid, cid) not in seen_pairs, "duplicate retained job identity")
        seen_jobs.add(jid); seen_pairs.add((sid, cid))
        records = job.get("records")
        _require(isinstance(records, dict) and set(records) in
                 ({"plan", "result"}, set(RECORD_NAMES)), "retained record set is incomplete")
        for role, item in records.items():
            expected = OUT + "/native/" + jid + "/" + RECORD_NAMES[role]
            _require(isinstance(item, dict) and item.get("path") == expected
                     and _sha_value(item.get("sha256")) and _sha_value(item.get("canonical_sha256")),
                     "retained record path/checksum differs")
            _require(expected not in seen_files, "duplicate retained record file")
            seen_files.add(expected)
        saved_result = json.loads(_checked(records["result"], root).read_text())
        _require(canonical_sha256(saved_result) == records["result"]["canonical_sha256"],
                 "actual retained result canonical checksum differs")
        _require(saved_result.get("scenario_id") == sid and saved_result.get("candidate_id") == cid,
                 "retained record identity differs")
        if dependencies:
            adapter = _adapter_path(saved_result.get("implementation_sha256", {}))
            _require(adapter in dependencies and saved_result.get("implementation_sha256") ==
                     dependencies[adapter]["implementation_sha256"],
                     "actual retained result dependency set differs")
        status = saved_result.get("status")
        _require(status in {"blocked", "prediction_locked"}
                 and saved_result.get("prediction_locked") is (status == "prediction_locked"),
                 "failed or inconsistent native results cannot be retained")
        _require(set(records) == (set(RECORD_NAMES) if status == "prediction_locked" else {"plan", "result"}),
                 "retained record set differs from terminal status")
    return record, preserved, by_change


def _registry(project_root):
    root = Path(project_root).resolve(strict=True)
    ref = {"path": REGISTRY, "sha256": REGISTRY_SHA}
    registry = json.loads(_checked(ref, root).read_text())
    _require(registry.get("schema_version") == REGISTRY_SCHEMA
             and registry.get("execution_revision") == REVISION
             and registry.get("sealed_observations_accessed") is False,
             "registry schema/revision/outcome barrier differs")
    entries = registry.get("profiles")
    _require(isinstance(entries, list) and 1 <= len(entries) <= 16,
             "bounded nonempty historical profile registration required")
    ids = set(); profile_paths = set(); jobs = set(); pairs = set(); files = set()
    final_hashes = {}; profiles = []
    for entry in entries:
        _require(isinstance(entry, dict), "invalid historical profile registration")
        pid, manifest = entry.get("profile_id"), entry.get("manifest")
        _require(isinstance(pid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", pid) is not None
                 and pid not in ids and isinstance(manifest, dict),
                 "duplicate or invalid historical profile identity")
        _require(manifest.get("path") not in profile_paths, "duplicate historical profile manifest")
        ids.add(pid); profile_paths.add(manifest.get("path"))
        record, archive, changes = _profile(root, manifest, pid)
        for path, change in changes.items():
            _require(path not in final_hashes or final_hashes[path] == change["after_sha256"],
                     "historical profiles disagree on current producer hash")
            final_hashes[path] = change["after_sha256"]
        for job in record["retained_jobs"]:
            pair = (job["scenario_id"], job["candidate_id"])
            _require(job["job_id"] not in jobs and pair not in pairs,
                     "retained job identity is duplicated across profiles")
            jobs.add(job["job_id"]); pairs.add(pair)
            for item in job["records"].values():
                _require(item["path"] not in files, "retained record is duplicated across profiles")
                files.add(item["path"])
        profiles.append({"profile_id": pid, "manifest": dict(manifest), "record": record,
                         "archive": archive, "changes": changes})
    return root, registry, profiles


def profile_refs(*, project_root=ROOT):
    """Resources to freeze before launch; no completed protocol is required."""
    _, _, profiles = _registry(project_root)
    refs = [{"path": REGISTRY, "sha256": REGISTRY_SHA}]
    for profile in profiles:
        refs.extend([profile["manifest"], profile["record"]["preparation"]])
        refs.extend(row["preserved"] for row in profile["record"]["code_archive"])
    unique = {}
    for ref in refs:
        _require(ref["path"] not in unique or unique[ref["path"]] == ref,
                 "conflicting profile resource references")
        unique[ref["path"]] = dict(ref)
    return list(unique.values())


def _bound_record(record, frozen_code_sha256, *, project_root=ROOT):
    """Return the exact producer map for one registered, unchanged record."""
    _require(isinstance(record, dict) and isinstance(frozen_code_sha256, dict), "invalid record/code map")
    implementation = record.get("implementation_sha256")
    _require(isinstance(implementation, dict) and bool(implementation), "implementation map missing")
    _require(all(isinstance(p, str) and _sha_value(h) for p, h in implementation.items()),
             "invalid implementation entry")
    if all(frozen_code_sha256.get(p) == h for p, h in implementation.items()):
        return frozen_code_sha256, None
    root, _, profiles = _registry(project_root)
    target_digest = canonical_sha256(record)
    matches = [(profile, job, role, ref) for profile in profiles
               for job in profile["record"]["retained_jobs"] for role, ref in job["records"].items()
               if ref["canonical_sha256"] == target_digest]
    _require(len(matches) == 1, "record is not uniquely registered for retention")
    profile, job, role, ref = matches[0]
    protocol = json.loads(_path(OUT + "/protocol.json", root).read_text())
    for required in ({"path": REGISTRY, "sha256": REGISTRY_SHA}, profile["manifest"]):
        refs = [r for r in protocol.get("inputs", [])
                if isinstance(r, dict) and r.get("path") == required["path"]]
        _require(refs == [required], "registry/profile is not uniquely bound by the current protocol")
    _require(protocol.get("execution_revision") == REVISION and record.get("execution_revision") == REVISION,
             "current protocol or retained record revision differs")
    archive, changes = profile["archive"], profile["changes"]
    for name, change in changes.items():
        _require(frozen_code_sha256.get(name) == change["after_sha256"],
                 "current producer after-hash differs: " + name)
    for name, checksum in implementation.items():
        _require(name in archive and archive[name]["original"]["sha256"] == checksum,
                 "record producer is absent from original archive: " + name)
        expected = changes[name]["before_sha256"] if name in changes else frozen_code_sha256.get(name)
        _require(checksum == expected, "unregistered producer drift: " + name)
    _require(record.get("scenario_id") == job["scenario_id"] and record.get("candidate_id") == job["candidate_id"],
             "retained record identity differs")
    actual = json.loads(_checked(ref, root).read_text())
    _require(canonical_sha256(actual) == target_digest and actual == record,
             "actual retained record content differs")
    code = dict(frozen_code_sha256)
    for name, change in changes.items():
        code[name] = change["before_sha256"]
    return code, profile


def binding_code(record, frozen_code_sha256, *, project_root=ROOT):
    """Backward-compatible producer map; historical records remain unchanged."""
    return _bound_record(record, frozen_code_sha256, project_root=project_root)[0]


def binding_profile(record, frozen_code_sha256, *, adapter_path, current_required,
                    project_root=ROOT):
    """Bind exact old dependencies only after full immutable-record authentication."""
    code, profile = _bound_record(record, frozen_code_sha256, project_root=project_root)
    implementation = record["implementation_sha256"]
    _require(adapter_path in ADAPTER_PATHS and _adapter_path(implementation) == adapter_path,
             "scoring adapter differs from original producer")
    required = set(current_required)
    if profile is not None and "adapter_dependencies" in profile["record"]:
        declaration = profile["record"]["adapter_dependencies"].get(adapter_path)
        _require(declaration is not None and implementation == declaration["implementation_sha256"],
                 "retained record dependency set differs")
        required = set(declaration["implementation_sha256"])
    _require(required.issubset(implementation), "Repaired executor dependencies are incomplete")
    return code, required
