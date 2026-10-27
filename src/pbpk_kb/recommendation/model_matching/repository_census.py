from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


CENSUS_VERSION = "osp-public-model-repository-census-1.0.0"
DEFAULT_ORGANIZATION = "Open-Systems-Pharmacology"
DEFAULT_TOPIC = "osp-model"

MODEL_SUFFIXES = {".pksim5", ".pkml", ".mobi"}
SNAPSHOT_NAME_PATTERN = re.compile(
    r"(?:model|snapshot|simulation).*(?:\.json)$", re.IGNORECASE
)
DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
REPORT_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".qmd", ".rmd"}
LICENSE_NAMES = {"license", "license.md", "license.txt", "copying", "copying.txt"}


class RepositoryCensusError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256_bytes(payload)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _http_json(
    url: str,
    *,
    token: str | None,
    retries: int = 4,
) -> tuple[dict[str, Any], dict[str, str], bytes]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pbpk-public-model-census/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                return (
                    json.loads(raw),
                    {key.casefold(): value for key, value in response.headers.items()},
                    raw,
                )
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 429, 500, 502, 503, 504} or attempt == retries:
                raise RepositoryCensusError(
                    f"GitHub request failed ({exc.code}): {url}"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 30)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RepositoryCensusError(f"GitHub request failed: {url}") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def discover_repositories(
    *,
    output_directory: Path,
    organization: str = DEFAULT_ORGANIZATION,
    topic: str = DEFAULT_TOPIC,
    token: str | None = None,
    fetch_json: Callable[..., tuple[dict[str, Any], dict[str, str], bytes]] = _http_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    query = f"org:{organization} topic:{topic}"
    repositories: dict[str, dict[str, Any]] = {}
    page = 1
    declared_total: int | None = None
    raw_pages: list[dict[str, Any]] = []
    while True:
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
            {"q": query, "sort": "full_name", "order": "asc", "per_page": 100, "page": page}
        )
        payload, headers, raw = fetch_json(url, token=token)
        raw_path = raw_directory / f"github_search_page_{page:03d}.json"
        raw_path.write_bytes(raw)
        items = payload.get("items") or []
        if declared_total is None:
            declared_total = int(payload.get("total_count") or 0)
        for item in items:
            full_name = str(item.get("full_name") or "")
            if not full_name:
                continue
            license_value = item.get("license") or {}
            repositories[full_name.casefold()] = {
                "repository_id": item.get("id"),
                "name": item.get("name"),
                "full_name": full_name,
                "html_url": item.get("html_url"),
                "clone_url": item.get("clone_url"),
                "default_branch": item.get("default_branch"),
                "description": item.get("description"),
                "archived": bool(item.get("archived")),
                "fork": bool(item.get("fork")),
                "topics": sorted(item.get("topics") or []),
                "license_spdx_from_api": license_value.get("spdx_id"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "pushed_at": item.get("pushed_at"),
            }
        raw_pages.append(
            {
                "page": page,
                "url": url,
                "item_count": len(items),
                "sha256": _sha256_file(raw_path),
                "etag": headers.get("etag"),
                "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
            }
        )
        if not items or len(repositories) >= (declared_total or 0):
            break
        page += 1
    rows = sorted(repositories.values(), key=lambda row: str(row["full_name"]).casefold())
    if declared_total is not None and len(rows) != declared_total:
        raise RepositoryCensusError(
            f"GitHub declared {declared_total} repositories but {len(rows)} were collected."
        )
    manifest = {
        "census_version": CENSUS_VERSION,
        "organization": organization,
        "topic": topic,
        "query": query,
        "declared_repository_count": declared_total,
        "discovered_repository_count": len(rows),
        "raw_pages": raw_pages,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    return rows, manifest


def _git(repository: Path, *args: str, timeout: int = 900) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RepositoryCensusError(
            result.stderr.strip() or result.stdout.strip() or "git command failed"
        )
    return result.stdout


def _git_retry(
    repository: Path,
    *args: str,
    attempts: int = 4,
    timeout: int = 900,
) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _git(repository, *args, timeout=timeout)
        except (RepositoryCensusError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
    assert last_error is not None
    raise last_error


def _safe_repository_name(full_name: str) -> str:
    parts = PurePosixPath(full_name).parts
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise RepositoryCensusError(f"Unsafe repository name: {full_name!r}")
    return "__".join(parts)


def mirror_metadata(repository: Mapping[str, Any], mirror_root: Path) -> tuple[Path, str]:
    destination = mirror_root / _safe_repository_name(str(repository["full_name"]))
    clone_url = str(repository.get("clone_url") or "")
    branch = str(repository.get("default_branch") or "main")
    if not clone_url:
        raise RepositoryCensusError("Repository clone URL is missing.")
    existing_mirror = destination.exists()
    if existing_mirror:
        if not (destination / ".git").is_dir():
            raise RepositoryCensusError(f"Existing mirror is not a Git repository: {destination}")
        _git_retry(
            destination,
            "fetch",
            "--filter=blob:none",
            "--depth=1",
            "origin",
            branch,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth=1",
                "--branch",
                branch,
                clone_url,
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode:
            raise RepositoryCensusError(
                result.stderr.strip() or result.stdout.strip() or "git clone failed"
            )
    commit_ref = "FETCH_HEAD" if existing_mirror else "HEAD"
    commit = _git(destination, "rev-parse", commit_ref).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RepositoryCensusError(
            f"Unable to resolve a 40-character commit for {repository['full_name']}."
        )
    return destination, commit


def classify_repository_tree(
    repository: Mapping[str, Any],
    *,
    commit: str,
    paths: Iterable[str],
    license_texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized = sorted({PurePosixPath(path).as_posix() for path in paths if path.strip()})
    root_license_paths = [
        path
        for path in normalized
        if len(PurePosixPath(path).parts) == 1 and PurePosixPath(path).name.casefold() in LICENSE_NAMES
    ]
    native_executable_model_assets = [
        path for path in normalized if PurePosixPath(path).suffix.casefold() in MODEL_SUFFIXES
    ]
    snapshot_assets = [
        path
        for path in normalized
        if SNAPSHOT_NAME_PATTERN.search(PurePosixPath(path).name)
    ]
    data_assets = [
        path
        for path in normalized
        if PurePosixPath(path).suffix.casefold() in DATA_SUFFIXES
        and path not in snapshot_assets
    ]
    report_assets = [
        path
        for path in normalized
        if PurePosixPath(path).suffix.casefold() in REPORT_SUFFIXES
        and any(token in path.casefold() for token in ("evaluation", "report", "qualification", "validation"))
    ]
    workflow_assets = [
        path
        for path in normalized
        if PurePosixPath(path).suffix.casefold() in {".r", ".py", ".yaml", ".yml"}
    ]
    model_assets = sorted(set(native_executable_model_assets + snapshot_assets))
    api_license = str(repository.get("license_spdx_from_api") or "").strip()
    detected_license = None
    combined_license_text = "\n".join((license_texts or {}).values()).casefold()
    if "gplv2" in combined_license_text or "gpl v2" in combined_license_text:
        detected_license = "GPL-2.0"
    elif "gplv3" in combined_license_text or "gpl v3" in combined_license_text:
        detected_license = "GPL-3.0"
    elif "gnu general public license" in combined_license_text:
        if "version 3" in combined_license_text:
            detected_license = "GPL-3.0"
        elif "version 2" in combined_license_text:
            detected_license = "GPL-2.0"
        else:
            detected_license = "GPL"
    elif "apache license" in combined_license_text and "version 2.0" in combined_license_text:
        detected_license = "Apache-2.0"
    elif "mit license" in combined_license_text:
        detected_license = "MIT"
    elif "mozilla public license" in combined_license_text:
        detected_license = "MPL"
    elif "bsd" in combined_license_text and "redistribution and use" in combined_license_text:
        detected_license = "BSD"
    resolved_spdx = detected_license or api_license or None
    license_evidence_paths = sorted((license_texts or {}))
    license_resolved = bool(license_evidence_paths) and str(resolved_spdx or "").casefold() not in {
        "",
        "noassertion",
        "other",
    }
    static_processing_eligible = bool(model_assets) and license_resolved
    execution_probe_eligible = bool(model_assets) and license_resolved
    scientific_validation_candidate = (
        execution_probe_eligible
        and bool(data_assets or snapshot_assets)
        and bool(report_assets)
    )
    blockers = []
    if not license_evidence_paths:
        blockers.append("LICENSE_EVIDENCE_MISSING")
    if license_evidence_paths and not license_resolved:
        blockers.append("LICENSE_SPDX_UNRESOLVED")
    if not model_assets:
        blockers.append("MODEL_ASSET_MISSING")
    if not model_assets:
        blockers.append("EXECUTION_SOURCE_ASSET_MISSING")
    if not data_assets and not snapshot_assets:
        blockers.append("OBSERVED_OR_VALIDATION_DATA_NOT_DISCOVERED")
    if not report_assets:
        blockers.append("EVALUATION_REPORT_NOT_DISCOVERED")
    return {
        **dict(repository),
        "repository_commit": commit,
        "tracked_file_count": len(normalized),
        "root_license_paths": root_license_paths,
        "license_evidence_paths": license_evidence_paths,
        "root_license_checksums": {
            path: _sha256_bytes(text.encode("utf-8"))
            for path, text in sorted((license_texts or {}).items())
        },
        "license_spdx_detected": detected_license,
        "license_spdx_resolved": resolved_spdx,
        "model_assets": model_assets,
        "native_executable_model_assets": native_executable_model_assets,
        "reconstructable_snapshot_assets": snapshot_assets,
        "executable_model_assets": native_executable_model_assets,
        "snapshot_assets": snapshot_assets,
        "data_assets": data_assets,
        "report_assets": report_assets,
        "workflow_assets": workflow_assets,
        "static_processing_eligible": static_processing_eligible,
        "execution_probe_eligible": execution_probe_eligible,
        "scientific_validation_candidate": scientific_validation_candidate,
        "blockers": blockers,
    }


def inspect_repository(
    repository: Mapping[str, Any],
    *,
    mirror_root: Path,
) -> dict[str, Any]:
    mirror, commit = mirror_metadata(repository, mirror_root)
    paths = _git(mirror, "ls-tree", "-r", "--name-only", commit).splitlines()
    root_license_paths = [
        path
        for path in paths
        if len(PurePosixPath(path).parts) == 1
        and PurePosixPath(path).name.casefold() in LICENSE_NAMES
    ]
    license_texts = {
        path: _git(mirror, "show", f"{commit}:{path}")
        for path in root_license_paths
    }
    root_readmes = [
        path
        for path in paths
        if len(PurePosixPath(path).parts) == 1
        and PurePosixPath(path).name.casefold() in {"readme", "readme.md", "readme.txt", "readme.rst"}
    ]
    for path in root_readmes:
        text = _git(mirror, "show", f"{commit}:{path}")
        lowered = text.casefold()
        if any(
            token in lowered
            for token in (
                "gnu general public license",
                "licensed under",
                "license: gpl",
                "distributed under",
            )
        ):
            license_texts[path] = text
    result = classify_repository_tree(
        repository,
        commit=commit,
        paths=paths,
        license_texts=license_texts,
    )
    result["metadata_mirror_path"] = str(mirror.resolve())
    result["tree_listing_sha256"] = _sha256_bytes(
        ("\n".join(sorted(paths)) + "\n").encode("utf-8")
    )
    return result


def run_census(
    *,
    output_directory: Path,
    mirror_root: Path,
    organization: str = DEFAULT_ORGANIZATION,
    topic: str = DEFAULT_TOPIC,
    workers: int = 8,
    token: str | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    repositories, discovery = discover_repositories(
        output_directory=output_directory,
        organization=organization,
        topic=topic,
        token=token,
    )
    _atomic_jsonl(output_directory / "repositories.jsonl", repositories)
    inspected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reusable_rows: dict[str, dict[str, Any]] = {}
    if resume_from is not None:
        previous_assets = resume_from / "repository_assets.jsonl"
        if not previous_assets.is_file():
            raise FileNotFoundError(previous_assets)
        reusable_rows = {
            str(row["full_name"]).casefold(): row
            for row in (
                json.loads(line)
                for line in previous_assets.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    pending = []
    for row in repositories:
        previous = reusable_rows.get(str(row["full_name"]).casefold())
        if (
            previous is not None
            and previous.get("updated_at") == row.get("updated_at")
            and re.fullmatch(r"[0-9a-f]{40}", str(previous.get("repository_commit") or ""))
        ):
            inspected.append(
                {
                    **previous,
                    "census_reused_from": str(resume_from.resolve()),
                }
            )
        else:
            pending.append(row)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(inspect_repository, row, mirror_root=mirror_root): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                inspected.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "full_name": row.get("full_name"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    inspected.sort(key=lambda row: str(row["full_name"]).casefold())
    failures.sort(key=lambda row: str(row["full_name"]).casefold())
    outputs = {
        "repositories.jsonl": _sha256_file(output_directory / "repositories.jsonl"),
        "repository_assets.jsonl": _atomic_jsonl(
            output_directory / "repository_assets.jsonl", inspected
        ),
        "failures.jsonl": _atomic_jsonl(output_directory / "failures.jsonl", failures),
    }
    summary = {
        **discovery,
        "inspected_repository_count": len(inspected),
        "failed_repository_count": len(failures),
        "static_processing_eligible_count": sum(
            bool(row["static_processing_eligible"]) for row in inspected
        ),
        "execution_probe_eligible_count": sum(
            bool(row["execution_probe_eligible"]) for row in inspected
        ),
        "scientific_validation_candidate_count": sum(
            bool(row["scientific_validation_candidate"]) for row in inspected
        ),
        "complete": len(inspected) == len(repositories) and not failures,
        "mirror_root": str(mirror_root.resolve()),
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "reused_repository_count": len(repositories) - len(pending),
        "retried_repository_count": len(pending),
        "output_checksums": outputs,
    }
    _atomic_json(output_directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and inspect all public OSP repositories tagged osp-model."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--organization", default=DEFAULT_ORGANIZATION)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    result = run_census(
        output_directory=args.output,
        mirror_root=args.mirror_root,
        organization=args.organization,
        topic=args.topic,
        workers=args.workers,
        token=os.getenv(args.github_token_env),
        resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
