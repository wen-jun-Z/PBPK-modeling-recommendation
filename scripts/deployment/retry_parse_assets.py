"""Retry only failed, non-archive assets that emitted no partial parse output.

prepare writes an isolated audit view; apply commits only after identical-input
checks under the project's three writer locks. No acquisition or LLM calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.parsers.emitter import PARSER_VERSION, ParseEmitter, hash_text, stable_id
from pbpk_kb.parsers.raw_formats import RawFormatParsers
from pbpk_kb.parsers.registry import ParserRegistry
from pbpk_kb.processing.parse_stage import (
    ArchiveParser, RawParseStage, audit_parse_provenance, prepare_asset_for_dispatch,
    resolve_asset_links, update_pipeline_state,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout

DEPLOY = Path("runtime/deployment/evidence_expansion_20261009")
LEDGERS = {"asset_parse_status.jsonl", "processing_failures.jsonl"}
CODE = (
    "scripts/deployment/retry_parse_assets.py", "src/pbpk_kb/parsers/source_integrity.py",
    "src/pbpk_kb/parsers/raw_formats.py", "src/pbpk_kb/parsers/emitter.py",
    "src/pbpk_kb/processing/parse_stage.py",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path, value):
    temporary = path.with_name(path.name + ".retry-tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path, limit=None, *, offset=0):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = limit
        while remaining is None or remaining > 0:
            chunk = stream.read(min(1024 * 1024, remaining) if remaining is not None else 1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    require(remaining in (None, 0), "File shortened during fingerprint")
    return value.hexdigest()


def inside(root, path):
    path = Path(path)
    candidate = path if path.is_absolute() else root / path
    require(candidate.resolve().is_relative_to(root.resolve()), "Path outside project")
    require(not candidate.is_symlink(), "Symlink cannot be mutated by parse recovery")
    return candidate.resolve()


def relative(root, path):
    return str(inside(root, path).relative_to(root))


def rows(path):
    with path.open("rb") as stream:
        for line in stream:
            require(line.strip() and line.endswith(b"\n"), "Noncanonical JSONL boundary; refusing byte-changing recovery")
            yield line, json.loads(line)


def owned_by(row, asset_ids):
    return any(row.get(key) in asset_ids for key in
               ("raw_asset_id", "archive_raw_asset_id", "alias_raw_asset_id", "canonical_raw_asset_id"))


def ensure_exited(identity):
    require(isinstance(identity, dict) and type(identity.get("pid")) is int, "Missing registered process")
    proc = Path("/proc") / str(identity["pid"])
    if not proc.exists():
        return
    require(proc.stat().st_uid == os.getuid(), "Registered PID owner changed")
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    require(identity.get("start_ticks") == fields[19], "Registered PID identity changed")
    require(fields[0] == "Z", "Registered project process has not exited")


def validate_registration(root, run_id):
    state = read(root / DEPLOY / "status.json")
    batch = state.get("current_batch") or {}
    require(state.get("status") == "stopped_for_review", "Controller must be stopped for review")
    require(batch.get("run_id") == run_id and batch.get("status") == "running_pilot", "Wrong registered pilot batch")
    ensure_exited(state.get("controller"))
    require(batch.get("pipeline_pid") == (batch.get("pipeline_identity") or {}).get("pid"), "Writer registration differs")
    ensure_exited(batch.get("pipeline_identity"))
    monitor = read(root / "runtime/deployment/evidence_monitor" / run_id / "status.json")
    require(monitor.get("monitor_pid") == batch.get("watchdog_pid"), "Wrong registered monitor")
    require(all((monitor.get("target") or {}).get(k) == batch["pipeline_identity"].get(k)
                for k in ("pid", "start_ticks")), "Monitor target differs")
    require(not (Path("/proc") / str(batch["watchdog_pid"])).exists(), "Registered monitor has not exited")
    for name, sha in state.get("frozen_inputs", {}).items():
        require(digest(inside(root, name)) == sha, "Frozen expansion input changed")
    return state, batch


@contextmanager
def locked(root, run_id):
    require(Path(run_id).name == run_id and run_id.startswith("run_"), "Invalid run identifier")
    handles = []
    try:
        for path in (root / DEPLOY / "controller.lock", root / "runtime/deployment/evidence_pipeline.lock",
                     root / "runtime/deployment/evidence_monitor" / run_id / "monitor.lock"):
            handle = inside(root, path).open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_registration(root, run_id)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def fingerprint(root, paths):
    return {relative(root, path): {"sha256": digest(path), "size": path.stat().st_size}
            for path in paths}


def verify_fingerprints(root, expected):
    for name, item in expected.items():
        path = inside(root, name)
        require(path.stat().st_size == item["size"] and digest(path) == item["sha256"],
                "Concurrent or changed input: " + name)


def registry_for(assets, staging):
    registry = ParserRegistry()
    RawFormatParsers().register(registry)
    checksums = {}
    for asset in assets:
        checksums.setdefault(asset.checksum, asset.raw_asset_id)
    registry.register("archive", "archive_parser", ArchiveParser(registry, checksums, staging).parse)
    return registry


def audit_outputs(layout, assets, run_id, selected=(), require_empty=False):
    by_id = {asset.raw_asset_id: asset for asset in assets}
    require(len(by_id) == len(assets), "Duplicate inventory asset identity")
    emitter = ParseEmitter(layout, run_id)
    statuses, canonical = {}, set()
    for name in ParseEmitter.OUTPUT_NAMES:
        for _, row in rows(layout.path(name)):
            require(row.get("run_id") == run_id, "Output belongs to another run")
            if require_empty and name not in LEDGERS:
                require(not owned_by(row, selected), "Partial output exists; full parse recovery required")
            owner = row.get("raw_asset_id") or row.get("archive_raw_asset_id") or row.get("alias_raw_asset_id")
            require(owner in by_id, "Output asset identity missing from inventory")
            asset = by_id[owner]
            checksum = row.get("checksum") or row.get("archive_checksum")
            require(checksum == asset.checksum, "Output checksum differs from inventory")
            if row.get("storage_path"):
                require(row["storage_path"] == asset.storage_path, "Output storage identity differs")
            emitter.counts[Path(name).stem] += 1
            if name == "asset_parse_status.jsonl":
                require(owner not in statuses, "Duplicate asset status")
                statuses[owner] = row
                status = row["terminal_status"]
                emitter.status_counts[status] += 1
                emitter.parser_counts[row["parser_name"]] += 1
                if status not in {"failed", "duplicate_linked"} and row["parser_name"] != "source_availability_gate":
                    canonical.add((asset.checksum, row["parser_name"], row["parser_version"]))
            if name == "processing_failures.jsonl":
                emitter.failure_counts[row["failure_type"]] += 1
    require(set(statuses) == set(by_id), "Status coverage is not exactly the inventory")
    return emitter, statuses, len(canonical)


def validate_publisher_metadata_retry(asset, registered, outcome, emitter):
    """A narrow metadata outcome, never a generic escape from a parse failure."""
    require(asset.source_id == "pmc_oa" and asset.file_family == "xml"
            and registered.name == "pmc_jats_parser" and not asset.inner_path and asset.file_size <= 16384,
            "Publisher metadata retry has an unexpected source or parser")
    from pbpk_kb.parsers.raw_formats import recognize_publisher_data_availability

    payload = asset.physical_path.read_bytes()
    require(len(payload) == asset.file_size and hashlib.sha256(payload).hexdigest() == asset.checksum,
            "Publisher metadata source bytes changed")
    record = recognize_publisher_data_availability(payload)
    require(record is not None, "Publisher metadata is not recognized from the original source")
    expected_metadata = {"publisher_metadata_kind": "data_availability", "scientific_evidence": False,
                         "llm_eligible": False, "source_bytes_preserved": True,
                         "recognition_version": "publisher-data-availability-v1"}
    # JSON comparison also distinguishes false from numeric zero.
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    require(canonical(outcome.metadata) == canonical(expected_metadata),
            "Publisher metadata outcome proof differs from the recognized contract")
    require(dict(emitter.asset_counts[asset.raw_asset_id]) == {"parsed_records": 1},
            "Publisher metadata must emit exactly one record and no scientific content")
    for handle in emitter.handles.values():
        handle.flush()
    actual_records = []
    for name in ParseEmitter.OUTPUT_NAMES:
        for _, row in rows(emitter.layout.path(name)):
            if not owned_by(row, {asset.raw_asset_id}):
                continue
            require(name == "parsed_records.jsonl",
                    "Publisher metadata emitted non-record output")
            actual_records.append(row)
    require(len(actual_records) == 1, "Publisher metadata must contain exactly one stored record")
    row = actual_records[0]
    require(all(row.get(name) == value for name, value in {
        "run_id": asset.run_id, "raw_asset_id": asset.raw_asset_id, "source_id": asset.source_id,
        "compound": asset.compound, "storage_path": asset.storage_path, "source_url": asset.source_url,
        "checksum": asset.checksum, "parser_name": registered.name, "parser_version": PARSER_VERSION,
        "record_type": "publisher_data_availability",
    }.items()), "Publisher metadata record provenance differs from the source")
    require(canonical(row.get("record_json")) == canonical(record)
            and canonical(row.get("metadata_json")) == canonical(expected_metadata),
            "Publisher metadata record content differs from the original source")
    location = row.get("source_location")
    require(location == emitter.location(asset, "xml_path=/data:data/datasets/dataset"),
            "Publisher metadata source location is missing or unrelated")
    record_hash = hash_text(canonical(record))
    require(row.get("record_hash") == record_hash and row.get("parsed_record_id") == stable_id(
        "parsed_record", asset.raw_asset_id, location, "publisher_data_availability", record_hash),
        "Publisher metadata record checksum or stable identity differs")


def prepare(root, run_id, asset_ids, recovery_dir):
    root = root.resolve()
    with locked(root, run_id):
        return _prepare(root, run_id, set(asset_ids), recovery_dir)


def _prepare(root, run_id, selected, recovery_dir):
    recovery = inside(root, recovery_dir)
    require(recovery.parent == root / DEPLOY and not recovery.exists(), "Recovery must be a new direct deployment subdirectory")
    require(selected, "No asset selected")
    output = root / "data_process/raw_processing" / run_id
    raw = root / "artifacts/pbpk_raw_runs/compound_runs" / run_id
    manifest_path = root / "data_process/pipeline_runs" / (run_id + ".json")
    manifest = read(manifest_path)
    _, batch = validate_registration(root, run_id)
    require(manifest.get("run_id") == run_id and manifest.get("compounds") == batch["compounds"], "Wrong pipeline identity")
    require(manifest.get("status") == "failed" and manifest.get("current_stage") == "parse", "Pipeline is not failed at parse")
    require(inside(root, manifest["raw_run_root"]) == raw and inside(root, manifest["raw_processing_dir"]) == output,
            "Pipeline paths differ from registered run")
    stages = manifest.get("stages", {})
    require(set(stages) == {"raw_acquisition", "raw_finalization", "pmc_static_transfer", "raw_reconciliation", "raw_inventory", "parse"},
            "Unexpected upstream or downstream pipeline stages")
    require(all(v.get("accepted") is True and v.get("status") == "completed" for k, v in stages.items() if k != "parse"),
            "An upstream stage is not accepted")
    stage = RawParseStage(run_root=raw, output_dir=output)
    inventory_summary, state = stage._validate_prerequisite()
    require(state["stages"].get("parse", {}).get("status") == "failed"
            and stages["parse"].get("accepted") is False, "Parse failure states disagree")
    require(all(v.get("status") == "not_started" for k, v in state["stages"].items() if k not in {"inventory", "parse"}),
            "Downstream processing has started")
    assets = stage._load_inventory()
    require(len(assets) == inventory_summary["raw_asset_count"], "Inventory count changed")
    by_id = {a.raw_asset_id: a for a in assets}
    require(selected <= by_id.keys(), "Selected asset is outside this run")
    for asset in assets:
        require(asset.run_id == run_id, "Inventory run identity differs")
        require(inside(root, asset.storage_path).is_relative_to(raw), "Raw asset outside this run")
    layout = stage.layout
    files = [layout.path(name) for name in ParseEmitter.OUTPUT_NAMES]
    metadata_names = ["parse_summary.json", "parse_progress.json", "pipeline_state.json", "inventory_summary.json", "raw_inventory.jsonl"]
    files += [layout.path(name) for name in metadata_names]
    files += [manifest_path, root / DEPLOY / "status.json"]
    files += [inside(root, by_id[aid].storage_path) for aid in selected]
    files += [root / name for name in CODE]
    original = fingerprint(root, files)
    _, statuses, _ = audit_outputs(layout, assets, run_id, selected, require_empty=True)
    for aid in selected:
        asset, status = by_id[aid], statuses[aid]
        require(asset.file_family != "archive" and not asset.inner_path, "Archive retry is not supported")
        require(status.get("terminal_status") == "failed" and status.get("output_counts") == {}, "Asset is not an empty failed parse")
        require(digest(asset.physical_path) == asset.checksum and asset.physical_path.stat().st_size == asset.file_size,
                "Raw checksum or size changed")
    require(selected <= {row["raw_asset_id"] for _, row in rows(layout.path("processing_failures.jsonl"))}, "Failure ledger lacks selected asset")
    recovery.mkdir()
    retry = raw_processing_layout(recovery / "retry")
    retry.ensure_dirs()
    registry = registry_for(assets, recovery / "archive_staging")
    with ParseEmitter(retry, run_id) as emitter:
        for aid in sorted(selected):
            asset = by_id[aid]
            dispatch, warnings, metadata = prepare_asset_for_dispatch(asset)
            require(dispatch.file_family != "archive", "Archive dispatch retry is not supported")
            registered = registry.resolve(dispatch)
            outcome = registered.callback(dispatch, emitter)
            require(outcome.status in {"parsed", "unsupported", "metadata_only"}, "Retry did not reach an allowed terminal result")
            if outcome.status == "metadata_only":
                validate_publisher_metadata_retry(dispatch, registered, outcome, emitter)
            if outcome.status == "unsupported":
                proof = outcome.metadata
                require(proof.get("source_content_unavailable") is True and proof.get("checksum_verified") is True
                        and proof.get("official_etag_verified") is True
                        and proof.get("reason") == "official_ole_directory_out_of_bounds",
                        "Unsupported retry lacks verified official corruption metadata")
                require(not emitter.asset_counts[aid], "Unavailable source emitted partial content")
            emitter.status(asset, parser_name=registered.name, status=outcome.status,
                           warnings=[*warnings, *outcome.warnings], metadata={**metadata, **outcome.metadata})
    require(retry.path("processing_failures.jsonl").stat().st_size == 0, "Retry emitted a failure")
    require(retry.path("archive_members.jsonl").stat().st_size == 0 and retry.path("raw_asset_aliases.jsonl").stat().st_size == 0,
            "Retry unexpectedly emitted archive or alias rows")
    view = raw_processing_layout(recovery / "view")
    view.ensure_dirs()
    backups = recovery / "original"
    backups.mkdir()
    for name in (*LEDGERS, "parse_summary.json", "parse_progress.json", "pipeline_state.json"):
        shutil.copyfile(layout.path(name), backups / name)
    shutil.copyfile(manifest_path, backups / "pipeline_manifest.json")
    # Untouched large artifacts are read-only links in the audit view. The link
    # resolver always receives a private copy because it rewrites its input.
    for name in ParseEmitter.OUTPUT_NAMES:
        source, target, additions = layout.path(name), view.path(name, for_write=True), retry.path(name)
        if name in LEDGERS:
            with target.open("wb") as stream:
                for line, row in rows(source):
                    if row.get("raw_asset_id") not in selected:
                        stream.write(line)
                stream.write(additions.read_bytes())
        elif additions.stat().st_size or name == "asset_links.jsonl":
            shutil.copyfile(source, target)
            with target.open("ab") as stream, additions.open("rb") as add:
                shutil.copyfileobj(add, stream)
        else:
            os.link(source, target)
    link_counts = resolve_asset_links(view, assets)
    resolved = list(rows(view.path("asset_links.jsonl")))
    old_links = list(rows(layout.path("asset_links.jsonl")))
    require(len(resolved) >= len(old_links), "Existing asset links disappeared")
    require(all(old == new for (_, old), (_, new) in zip(old_links, resolved)), "Resolver changed another asset's links")
    # Restore exact original bytes; retain only newly resolved rows for append.
    append_links = retry.path("asset_links.jsonl")
    append_links.write_bytes(b"".join(line for line, _ in resolved[len(old_links):]))
    view.path("asset_links.jsonl").write_bytes(b"".join(line for line, _ in old_links + resolved[len(old_links):]))
    counts, recovered_statuses, canonical_count = audit_outputs(view, assets, run_id)
    summary = stage._build_summary(counts, expected_asset_count=len(assets), processed_asset_count=len(assets),
                                   canonical_asset_count=canonical_count, registry=registry)
    summary["asset_link_resolution"] = link_counts
    provenance = audit_parse_provenance(view)
    summary["provenance_audit"] = provenance
    unavailable_ids = sorted(aid for aid, row in recovered_statuses.items()
                             if (row.get("metadata_json") or {}).get("source_content_unavailable") is True)
    summary["source_content_unavailable_count"] = len(unavailable_ids)
    summary["source_content_unavailable_asset_ids"] = unavailable_ids
    # These are precisely RawParseStage.run's acceptance predicates. The audit
    # above additionally requires unique, checksum-bound inventory coverage.
    summary["accepted"] = bool(summary["asset_parse_status_count"] == len(assets)
        and summary["unhandled_raw_asset_count"] == 0 and summary["failed_asset_count"] == 0
        and summary["processing_failure_count"] == 0 and provenance["missing_required_provenance"] == 0)
    require(summary["accepted"], "Full parse acceptance still fails after isolated retry")
    summary.update(status="completed", generated_at=datetime.now(timezone.utc).isoformat(),
                   recovery_receipt=relative(root, recovery / "receipt.json"))
    atomic(view.path("parse_summary.json", for_write=True), summary)
    update_pipeline_state(view.path("pipeline_state.json", for_write=True), state, summary, layout.path("parse_summary.json"))
    atomic(view.path("parse_progress.json", for_write=True), {"stage": "parse", "status": "completed", "run_id": run_id,
        "expected_raw_asset_count": len(assets), "processed_raw_asset_count": len(assets),
        "terminal_status_counts": summary["terminal_status_counts"], "updated_at": summary["generated_at"]})
    old_failure = dict(manifest["stages"]["parse"])
    manifest["stages"]["parse"].update(status="completed", accepted=True, completed_at=summary["generated_at"],
        recovery_receipt=summary["recovery_receipt"], previous_failed_attempt=old_failure,
        result={"accepted": True, "summary": summary})
    manifest["stages"]["parse"].pop("error", None)
    manifest["stages"]["parse"].pop("failed_at", None)
    # No writer is running yet. The controller will set running on its next stage.
    manifest.update(status="ready_to_resume", updated_at=summary["generated_at"])
    atomic(recovery / "pipeline_manifest.json", manifest)
    replace = {relative(root, layout.path(name)): relative(root, view.path(name))
               for name in (*LEDGERS, "parse_summary.json", "parse_progress.json", "pipeline_state.json")}
    replace[relative(root, manifest_path)] = relative(root, recovery / "pipeline_manifest.json")
    append = {relative(root, layout.path(name)): relative(root, retry.path(name))
              for name in ParseEmitter.OUTPUT_NAMES if name not in LEDGERS and retry.path(name).stat().st_size}
    backup = {relative(root, layout.path(name)): relative(root, backups / name)
              for name in (*LEDGERS, "parse_summary.json", "parse_progress.json", "pipeline_state.json")}
    backup[relative(root, manifest_path)] = relative(root, backups / "pipeline_manifest.json")
    verify_fingerprints(root, original)
    receipt = {"status": "prepared", "run_id": run_id, "asset_ids": sorted(selected), "original": original,
               "replace": replace, "append": append, "backup": backup, "summary": summary,
               "prepared": fingerprint(root, [inside(root, p) for p in [*replace.values(), *append.values(), *backup.values()]]),
               "created_at": summary["generated_at"]}
    atomic(recovery / "receipt.json", receipt)
    return receipt


def restore(root, recovery, receipt):
    # Both automatic rollback and an explicit post-crash rollback reject any
    # unexpected content instead of truncating an unrelated/new writer's data.
    for name, addition in receipt["append"].items():
        path = inside(root, name)
        old = receipt["original"][name]
        require(digest(path, old["size"]) == old["sha256"], "Rollback prefix changed")
        suffix = inside(root, addition).read_bytes()
        require(old["size"] <= path.stat().st_size <= old["size"] + len(suffix), "Rollback append length changed")
        with path.open("rb") as stream:
            stream.seek(old["size"])
            actual = stream.read()
        require(actual == suffix[:len(actual)], "Rollback append content changed")
    for name, saved in receipt["backup"].items():
        path, saved_path = inside(root, name), inside(root, saved)
        require(digest(saved_path) == receipt["original"][name]["sha256"], "Rollback backup changed")
        require(digest(path) in {receipt["original"][name]["sha256"], receipt["prepared"][receipt["replace"][name]]["sha256"]},
                "Rollback replacement changed")
    for name in receipt["append"]:
        with inside(root, name).open("r+b") as stream:
            stream.truncate(receipt["original"][name]["size"])
            stream.flush()
            os.fsync(stream.fileno())
    for name, saved in receipt["backup"].items():
        target = inside(root, name)
        temporary = target.with_name(target.name + ".retry-rollback")
        shutil.copyfile(inside(root, saved), temporary)
        temporary.replace(target)
    verify_fingerprints(root, receipt["original"])
    receipt["status"] = "rolled_back"
    atomic(recovery / "receipt.json", receipt)


def apply(root, recovery_dir, *, rollback=False):
    root = root.resolve()
    recovery = inside(root, recovery_dir)
    require(recovery.parent == root / DEPLOY, "Recovery outside deployment directory")
    receipt = read(recovery / "receipt.json")
    with locked(root, receipt["run_id"]):
        verify_fingerprints(root, receipt["prepared"])
        if rollback:
            require(receipt["status"] in {"applying", "applied"}, "No transaction to roll back")
            restore(root, recovery, receipt)
            return receipt
        require(receipt["status"] == "prepared", "Recovery is not prepared or was already applied")
        verify_fingerprints(root, receipt["original"])
        receipt["status"] = "applying"
        atomic(recovery / "receipt.json", receipt)
        try:
            for name, addition in receipt["append"].items():
                with inside(root, name).open("ab") as target, inside(root, addition).open("rb") as source:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            for name, source in receipt["replace"].items():
                target = inside(root, name)
                temporary = target.with_name(target.name + ".retry-apply")
                shutil.copyfile(inside(root, source), temporary)
                temporary.replace(target)
            for name, source in receipt["replace"].items():
                require(digest(inside(root, name)) == receipt["prepared"][source]["sha256"], "Committed replacement differs")
            for name, source in receipt["append"].items():
                path, old, addition = inside(root, name), receipt["original"][name], receipt["prepared"][source]
                require(path.stat().st_size == old["size"] + addition["size"]
                        and digest(path, old["size"]) == old["sha256"]
                        and digest(path, offset=old["size"]) == addition["sha256"], "Committed append differs")
            receipt["status"] = "applied"
            receipt["applied_at"] = datetime.now(timezone.utc).isoformat()
            atomic(recovery / "receipt.json", receipt)
        except BaseException:
            restore(root, recovery, receipt)
            raise
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "rollback"))
    parser.add_argument("--run-id")
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--recovery-dir", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.run_id is not None, "prepare requires --run-id")
        result = prepare(ROOT, args.run_id, args.asset_id, args.recovery_dir)
    else:
        result = apply(ROOT, args.recovery_dir, rollback=args.action == "rollback")
    print(json.dumps({key: result[key] for key in ("status", "run_id", "asset_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
