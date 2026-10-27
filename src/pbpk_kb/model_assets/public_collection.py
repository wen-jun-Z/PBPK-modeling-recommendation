"""Bounded, resumable collection of commit-pinned official OSP model sources.

Collection is not execution or scientific validation. No upstream code is run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import Event, Lock

from pbpk_kb.paths import PROJECT_ROOT
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.repository_census import discover_repositories

MIB = 1024 ** 2
GIB = 1024 ** 3
ORG = "Open-Systems-Pharmacology"


class CollectionSafetyError(RuntimeError):
    """Stop the complete batch on a disk, checksum or archive safety failure."""


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class CollectionBudget:
    def __init__(self, root: Path, *, max_bytes=15 * GIB, min_free=10 * GIB):
        self.root, self.max_bytes, self.min_free = root, max_bytes, min_free
        self.written = 0
        self.lock = Lock()
        self.stop = Event()

    def reserve(self, size: int) -> None:
        with self.lock:
            if self.stop.is_set():
                raise CollectionSafetyError("Collection stopped after a safety failure")
            if self.written + size > self.max_bytes or shutil.disk_usage(self.root).free - size < self.min_free:
                self.stop.set()
                raise CollectionSafetyError("Batch write budget or minimum free disk reached")
            self.written += size


def _open(url: str, *, accept=None):
    headers = {"User-Agent": "pbpk-project-public-models/2.0"}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=45)


def _inventory_json(url, *, token=None):
    if token is not None:
        raise ValueError("This public collector does not use credentials")
    with _open(url) as response:
        raw = response.read(8 * MIB + 1)
        if len(raw) > 8 * MIB:
            raise CollectionSafetyError("Inventory response exceeded 8 MiB")
        return json.loads(raw), {k.lower(): v for k, v in response.headers.items()}, raw


def safe_extract(archive: Path, destination: Path, budget: CollectionBudget, *, max_bytes=GIB, max_files=20000):
    """Extract regular files only, with bounded expansion and no path/link tricks."""
    total = 0
    files = []
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = PurePosixPath(member.name)
            if parts.is_absolute() or ".." in parts.parts or not parts.parts:
                raise CollectionSafetyError("Unsafe archive path")
            if member.isdir():
                continue
            if not member.isfile() or len(parts.parts) < 2:
                raise CollectionSafetyError("Non-regular archive member")
            relative = Path(*parts.parts[1:])
            if str(relative) in seen:
                raise CollectionSafetyError("Duplicate archive path")
            seen.add(str(relative))
            total += member.size
            if total > max_bytes or len(files) >= max_files:
                raise CollectionSafetyError("Archive expansion exceeded its bound")
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()) or target.exists():
                raise CollectionSafetyError("Archive destination is unsafe or already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with source.extractfile(member) as reader, target.open("xb") as writer:
                while block := reader.read(MIB):
                    budget.reserve(len(block))
                    count += len(block)
                    writer.write(block)
                    digest.update(block)
            if count != member.size:
                raise CollectionSafetyError("Truncated archive member")
            files.append({"path": str(relative), "bytes": count, "sha256": digest.hexdigest()})
    return files


def verify_manifest(row: dict) -> None:
    source = (PROJECT_ROOT / row["source_directory"]).resolve()
    if not source.is_relative_to(PROJECT_ROOT.resolve()):
        raise CollectionSafetyError("Source outside this project")
    archive = source.parent / "source.tar.gz"
    if sha256_file(archive) != row["source_archive_sha256"]:
        raise CollectionSafetyError("Source archive checksum mismatch")
    for item in row["files"]:
        path = source / item["path"]
        if not path.resolve().is_relative_to(PROJECT_ROOT.resolve()):
            raise CollectionSafetyError("Source file outside this project")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise CollectionSafetyError("Source file checksum mismatch")


def _pin_head(row: dict) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["name"]) or row["full_name"] != f"{ORG}/{row['name']}":
        raise CollectionSafetyError("Unexpected repository identity")
    # No inherited credentials, SSH configuration or global Git configuration.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(PROJECT_ROOT / "runtime/deployment/home"),
           "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", "-c", "credential.helper=", "-c", "http.version=HTTP/1.1", "ls-remote", "--heads",
             f"https://github.com/{row['full_name']}.git", f"refs/heads/{row['default_branch']}"],
            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=45, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        # One bounded alternate transport, never an automatic retry loop.
        payload, _, _ = _inventory_json(f"https://api.github.com/repos/{row['full_name']}/commits/" + urllib.parse.quote(row["default_branch"], safe=""))
        commit = payload.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise CollectionSafetyError("GitHub did not return a pinned commit")
        return commit
    matches = [line.split()[0] for line in result.stdout.splitlines() if line.endswith("\trefs/heads/" + row["default_branch"])]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0]):
        raise CollectionSafetyError("Could not uniquely pin repository head")
    return matches[0]


def _collect_one(row: dict, assets: Path, budget: CollectionBudget, reusable: dict):
    budget.reserve(0)
    if row["full_name"] in reusable:
        result = reusable[row["full_name"]]
        verify_manifest(result)
        return result
    commit = _pin_head(row)
    key = (row["full_name"], commit)
    if key in reusable:
        result = reusable[key]
        verify_manifest(result)
        return {**result, "reused_existing_snapshot": True}
    destination = assets / f"{row['name']}__{commit[:12]}"
    manifest = destination / "asset_manifest.json"
    if manifest.exists():
        result = json.loads(manifest.read_text())
        verify_manifest(result)
        return result
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise CollectionSafetyError("Incomplete prior extraction requires inspection before resuming")
    staging.mkdir(parents=True)
    archive = staging / "source.tar.gz"
    deadline = time.monotonic() + 240
    total = 0
    try:
        with _open(f"https://codeload.github.com/{row['full_name']}/tar.gz/{commit}") as reader, archive.open("xb") as writer:
            while block := reader.read(MIB):
                total += len(block)
                if total > 200 * MIB:
                    raise CollectionSafetyError("Archive download exceeded 200 MiB")
                if time.monotonic() > deadline:
                    raise CollectionSafetyError("Archive download exceeded 240 seconds")
                budget.reserve(len(block))
                writer.write(block)
        files = safe_extract(archive, staging / "source", budget)
        result = {
            "repository": row["full_name"], "commit": commit,
            "source_url": f"https://github.com/{row['full_name']}/tree/{commit}",
            "source_directory": str((destination / "source").relative_to(PROJECT_ROOT)),
            "source_archive_sha256": sha256_file(archive), "files": files,
            "collection_status": "downloaded_not_scientifically_qualified",
            "execution_status": "not_executed", "independent_validation_status": "not_run",
        }
        write_json(staging / "asset_manifest.json", result)
        staging.rename(destination)
        return result
    except Exception:
        # Only our uncommitted staging files are removed; completed assets are immutable.
        shutil.rmtree(staging)
        raise


def collect_source_files(repository: str, commit: str, tree: dict, paths: list[str], assets: Path) -> dict:
    """Recover selected source files after an archive failure; verify Git blob IDs.

    The manifest explicitly records the subset. This is not a complete mirror.
    """
    if not repository.startswith(ORG + "/") or not re.fullmatch(r"[A-Za-z0-9_.-]+", repository.split("/")[-1]):
        raise ValueError("Only official OSP repositories are supported")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or tree.get("truncated"):
        raise ValueError("An exact commit and complete file tree are required")
    assets = assets.resolve()
    if not assets.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Assets must belong to this project")
    assets.mkdir(parents=True, exist_ok=True)
    budget = CollectionBudget(assets)
    budget.reserve(sum(p.stat().st_size for p in assets.rglob("*") if p.is_file()))
    name = repository.split("/")[-1]
    destination = assets / f"{name}__{commit[:12]}__selected_sources"
    manifest = destination / "asset_manifest.json"
    if manifest.exists():
        result = json.loads(manifest.read_text())
        verify_manifest(result)
        return result
    entries = {item["path"]: item for item in tree["tree"]}
    selected = [entries[path] for path in paths]
    if sum(x.get("size", 0) for x in selected) > 200 * MIB:
        raise CollectionSafetyError("Selected sources exceed 200 MiB")
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir()
    files = []
    try:
        for item in selected:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or item.get("type") != "blob" or item.get("mode") not in {"100644", "100755"}:
                raise CollectionSafetyError("Invalid selected source path or file type")
            target = staging / "source" / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            git_hash = hashlib.sha1(f"blob {item['size']}\0".encode())
            digest = hashlib.sha256()
            total = 0
            deadline = time.monotonic() + 240
            with _open(f"https://api.github.com/repos/{repository}/git/blobs/{item['sha']}", accept="application/vnd.github.raw+json") as reader, target.open("xb") as writer:
                while block := reader.read(MIB):
                    total += len(block)
                    if total > item["size"] or time.monotonic() > deadline:
                        raise CollectionSafetyError("Selected source size or time bound reached")
                    budget.reserve(len(block))
                    writer.write(block)
                    digest.update(block)
                    git_hash.update(block)
            if total != item["size"] or git_hash.hexdigest() != item["sha"]:
                raise CollectionSafetyError("Selected source failed Git blob verification")
            files.append({"path": item["path"], "bytes": total, "sha256": digest.hexdigest(), "git_blob_sha1": item["sha"]})
        archive = staging / "source.tar.gz"
        budget.reserve(sum(x["bytes"] for x in files) + MIB)
        with tarfile.open(archive, "w:gz") as writer:
            for item in files:
                writer.add(staging / "source" / item["path"], arcname=name + "/" + item["path"], recursive=False)
        result = {"repository": repository, "commit": commit, "source_url": f"https://github.com/{repository}/tree/{commit}",
                  "source_directory": str((destination / "source").relative_to(PROJECT_ROOT)),
                  "source_archive_sha256": sha256_file(archive), "files": files,
                  "collection_status": "downloaded_not_scientifically_qualified",
                  "execution_status": "not_executed", "independent_validation_status": "not_run",
                  "collection_scope": "selected_sources_with_git_blob_verification",
                  "omitted_paths": sorted(set(entries) - set(paths))}
        write_json(staging / "asset_manifest.json", result)
        staging.rename(destination)
        return result
    except Exception:
        shutil.rmtree(staging)
        raise


def collect_public_models(output: Path, assets: Path, *, excluded=()) -> dict:
    for path in (output, assets):
        if not path.resolve().is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Collection paths must be inside this project")
        path.mkdir(parents=True, exist_ok=True)
    if (output / "repository_inventory.json").exists():
        inventory = json.loads((output / "repository_inventory.json").read_text())
    else:
        inventory, discovery = discover_repositories(output_directory=output, fetch_json=_inventory_json)
        write_json(output / "repository_inventory.json", inventory)
        write_json(output / "discovery.json", discovery)
    reusable = {}
    if (output / "collected_models.json").exists():
        for row in json.loads((output / "collected_models.json").read_text()):
            if row["collection_status"] != "failed":
                reusable[row["repository"]] = row
    old_assets = PROJECT_ROOT / "artifacts/pbpk-model-assets/coverage_expansion_20260907"
    for path in old_assets.glob("*/asset_manifest.json"):
        row = json.loads(path.read_text())
        reusable[(row["repository"], row["commit"])] = row
    budget = CollectionBudget(assets)
    budget.reserve(sum(p.stat().st_size for p in assets.rglob("*") if p.is_file()))
    started = datetime.now(UTC).isoformat()
    rows = []
    previous = json.loads((output / "collection_summary.json").read_text()) if (output / "collection_summary.json").exists() else {}
    attempts = previous.get("previous_attempts", []) + ([{k: previous.get(k) for k in ("started_at", "status", "completed", "downloaded", "failed", "completed_at")}] if previous else [])
    state = {"started_at": started, "repositories_discovered": len(inventory), "status": "running",
             "excluded_repositories": list(excluded), "previous_attempts": attempts,
             "bounds": {"workers": 3, "max_archive_mib": 200, "max_extracted_gib_per_repository": 1,
                        "max_batch_write_gib": 15, "minimum_free_gib": 10, "retries": 0,
                        "disk_check": "before every write block (at most 1 MiB)"}}
    write_json(output / "collection_summary.json", state)
    # Fixed finite work queue; no retry loop and no unbounded logs.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_collect_one, row, assets, budget, reusable): row for row in inventory if row["name"] not in excluded}
        for future in as_completed(futures):
            source = futures[future]
            if future.cancelled():
                continue
            try:
                row = future.result()
            except Exception as exc:
                row = {"repository": source["full_name"], "collection_status": "failed", "error_type": type(exc).__name__}
                if isinstance(exc, CollectionSafetyError):
                    budget.stop.set()
                    for pending in futures:
                        pending.cancel()
                    state["status"] = "stopped_safety_guard"
                    row["error"] = str(exc)
            rows.append(row)
            state.update(completed=len(rows), downloaded=sum(r["collection_status"] != "failed" for r in rows),
                         failed=sum(r["collection_status"] == "failed" for r in rows),
                         bytes_written=budget.written, updated_at=datetime.now(UTC).isoformat())
            write_json(output / "collected_models.json", sorted(rows, key=lambda r: r["repository"]))
            write_json(output / "collection_summary.json", state)
            print(json.dumps({k: row.get(k) for k in ("repository", "collection_status", "error_type", "error")}), flush=True)
    if not budget.stop.is_set():
        state["status"] = "completed" if state.get("failed", 0) == 0 else "completed_with_download_failures"
    state["completed_at"] = datetime.now(UTC).isoformat()
    write_json(output / "collection_summary.json", state)
    return state
