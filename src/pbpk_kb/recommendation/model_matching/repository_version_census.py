from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.recommendation.model_matching.repository_census import (
    LICENSE_NAMES,
    RepositoryCensusError,
    _git,
    _safe_repository_name,
    _sha256_bytes,
    classify_repository_tree,
)


VERSION_CENSUS_VERSION = "osp-public-model-version-census-1.0.0"


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
    return hashlib.sha256(payload).hexdigest()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_remote_tags(output: str) -> dict[str, str]:
    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        commit, ref = parts
        prefix = "refs/tags/"
        if not ref.startswith(prefix):
            continue
        tag = ref[len(prefix) :]
        if tag.endswith("^{}"):
            peeled[tag[:-3]] = commit
        else:
            direct[tag] = commit
    return {tag: peeled.get(tag, commit) for tag, commit in sorted(direct.items())}


def _run_with_retry(command: list[str], *, attempts: int = 4) -> str:
    last_error = ""
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
                env={
                    **os.environ,
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_HTTP_LOW_SPEED_LIMIT": "1",
                    "GIT_HTTP_LOW_SPEED_TIME": "60",
                },
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"git command timed out after {exc.timeout} seconds"
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
            continue
        if result.returncode == 0:
            return result.stdout
        last_error = result.stderr.strip() or result.stdout.strip()
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 20))
    raise RepositoryCensusError(last_error or "git command failed")


def discover_remote_tags(clone_url: str) -> dict[str, str]:
    return parse_remote_tags(
        _run_with_retry(["git", "ls-remote", "--tags", clone_url])
    )


def _ensure_tag_commit(mirror: Path, tag: str) -> str:
    _run_with_retry(
        [
            "git",
            "-C",
            str(mirror),
            "fetch",
            "--filter=blob:none",
            "--depth=1",
            "origin",
            f"refs/tags/{tag}:refs/tags/{tag}",
        ]
    )
    commit = _git(mirror, "rev-list", "-n", "1", f"refs/tags/{tag}").strip()
    if len(commit) != 40:
        raise RepositoryCensusError(f"Tag {tag!r} did not resolve to a commit.")
    return commit


def _git_show_text(mirror: Path, object_path: str) -> tuple[str, str]:
    result = subprocess.run(
        ["git", "-C", str(mirror), "show", object_path],
        check=False,
        capture_output=True,
        timeout=900,
    )
    if result.returncode:
        raise RepositoryCensusError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "git show failed"
        )
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return result.stdout.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise RepositoryCensusError(
        f"Text asset cannot be decoded losslessly: {object_path}"
    )


def _inspect_commit(
    repository: Mapping[str, Any],
    *,
    mirror: Path,
    commit: str,
) -> dict[str, Any]:
    paths = _git(mirror, "ls-tree", "-r", "--name-only", commit).splitlines()
    root_license_paths = [
        path
        for path in paths
        if len(Path(path).parts) == 1 and Path(path).name.casefold() in LICENSE_NAMES
    ]
    license_texts: dict[str, str] = {}
    license_encodings: dict[str, str] = {}
    for path in root_license_paths:
        text, encoding = _git_show_text(mirror, f"{commit}:{path}")
        license_texts[path] = text
        license_encodings[path] = encoding
    root_readmes = [
        path
        for path in paths
        if len(Path(path).parts) == 1
        and Path(path).name.casefold()
        in {"readme", "readme.md", "readme.txt", "readme.rst"}
    ]
    for path in root_readmes:
        text, encoding = _git_show_text(mirror, f"{commit}:{path}")
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
            license_encodings[path] = encoding
    result = classify_repository_tree(
        repository,
        commit=commit,
        paths=paths,
        license_texts=license_texts,
    )
    result["metadata_mirror_path"] = str(mirror.resolve())
    result["license_evidence_encodings"] = dict(sorted(license_encodings.items()))
    result["tree_listing_sha256"] = _sha256_bytes(
        ("\n".join(sorted(paths)) + "\n").encode("utf-8")
    )
    return result


def inspect_repository_versions(
    repository: Mapping[str, Any],
    *,
    mirror_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_name = str(repository["full_name"])
    mirror = mirror_root / _safe_repository_name(full_name)
    if not (mirror / ".git").is_dir():
        raise RepositoryCensusError(f"Metadata mirror is missing: {mirror}")
    tags = discover_remote_tags(str(repository["clone_url"]))
    commits: dict[str, list[str]] = {
        str(repository["repository_commit"]): ["HEAD"]
    }
    failures: list[dict[str, Any]] = []
    for tag in tags:
        try:
            commit = _ensure_tag_commit(mirror, tag)
            commits.setdefault(commit, []).append(tag)
        except Exception as exc:
            failures.append(
                {
                    "full_name": full_name,
                    "version_ref": tag,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    versions: list[dict[str, Any]] = []
    for commit, refs in sorted(commits.items()):
        try:
            if commit == repository["repository_commit"]:
                inspected = dict(repository)
            else:
                inspected = _inspect_commit(repository, mirror=mirror, commit=commit)
            versions.append(
                {
                    **inspected,
                    "version_id": f"{full_name}@{commit}",
                    "version_refs": sorted(set(refs), key=lambda value: (value != "HEAD", value)),
                    "is_default_head": "HEAD" in refs,
                    "distinct_commit": commit,
                    "version_census_source": "git_tag_and_default_head",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "full_name": full_name,
                    "version_ref": ",".join(sorted(refs)),
                    "repository_commit": commit,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return versions, failures


def run_version_census(
    *,
    repository_assets_path: Path,
    mirror_root: Path,
    output_directory: Path,
    workers: int = 4,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    repositories = [
        row
        for row in _read_jsonl(repository_assets_path)
        if row.get("execution_probe_eligible")
    ]
    versions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    resumed_names: set[str] = set()
    if resume_from is not None:
        previous_versions = resume_from / "model_versions.jsonl"
        if not previous_versions.is_file():
            previous_versions = resume_from / "partial_model_versions.jsonl"
        if previous_versions.is_file():
            previous_rows = _read_jsonl(previous_versions)
            previous_failures_path = resume_from / "failures.jsonl"
            if not previous_failures_path.is_file():
                previous_failures_path = resume_from / "partial_failures.jsonl"
            failed_names = (
                {
                    str(row.get("full_name") or "")
                    for row in _read_jsonl(previous_failures_path)
                }
                if previous_failures_path.is_file()
                else set()
            )
            current_commits = {
                str(row["full_name"]): str(row["repository_commit"])
                for row in repositories
            }
            grouped_previous: dict[str, list[dict[str, Any]]] = {}
            for row in previous_rows:
                grouped_previous.setdefault(str(row["full_name"]), []).append(row)
            for full_name, rows in grouped_previous.items():
                head_rows = [row for row in rows if row.get("is_default_head")]
                if (
                    full_name not in failed_names
                    and
                    len(head_rows) == 1
                    and current_commits.get(full_name)
                    == str(head_rows[0].get("repository_commit"))
                ):
                    versions.extend(rows)
                    resumed_names.add(full_name)
    pending_repositories = [
        row for row in repositories if str(row["full_name"]) not in resumed_names
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {
            executor.submit(
                inspect_repository_versions,
                row,
                mirror_root=mirror_root,
            ): row
            for row in pending_repositories
        }
        completed = len(resumed_names)
        for future in as_completed(pending):
            row = pending[future]
            try:
                version_rows, failure_rows = future.result()
                versions.extend(version_rows)
                failures.extend(failure_rows)
            except Exception as exc:
                failures.append(
                    {
                        "full_name": row.get("full_name"),
                        "version_ref": "repository",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            completed += 1
            versions.sort(
                key=lambda value: (
                    str(value["full_name"]).casefold(),
                    str(value["repository_commit"]),
                )
            )
            _atomic_jsonl(
                output_directory / "partial_model_versions.jsonl", versions
            )
            _atomic_jsonl(output_directory / "partial_failures.jsonl", failures)
            print(
                json.dumps(
                    {
                        "completed_repositories": completed,
                        "eligible_repositories": len(repositories),
                        "distinct_versions": len(versions),
                        "failures": len(failures),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    versions.sort(
        key=lambda row: (
            str(row["full_name"]).casefold(),
            str(row["repository_commit"]),
        )
    )
    failures.sort(
        key=lambda row: (
            str(row.get("full_name") or "").casefold(),
            str(row.get("version_ref") or ""),
        )
    )
    repository_versions: dict[str, int] = {}
    for row in versions:
        name = str(row["full_name"])
        repository_versions[name] = repository_versions.get(name, 0) + 1
    outputs = {
        "model_versions.jsonl": _atomic_jsonl(
            output_directory / "model_versions.jsonl", versions
        ),
        "failures.jsonl": _atomic_jsonl(
            output_directory / "failures.jsonl", failures
        ),
    }
    summary = {
        "version_census_version": VERSION_CENSUS_VERSION,
        "repository_assets_path": str(repository_assets_path.resolve()),
        "eligible_repository_count": len(repositories),
        "inspected_repository_count": len(repository_versions),
        "distinct_model_version_count": len(versions),
        "repositories_with_multiple_distinct_versions": sum(
            count >= 2 for count in repository_versions.values()
        ),
        "scientific_validation_version_count": sum(
            bool(row.get("scientific_validation_candidate")) for row in versions
        ),
        "failed_version_count": len(failures),
        "complete": len(repository_versions) == len(repositories) and not failures,
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "resumed_repository_count": len(resumed_names),
        "versions_per_repository": dict(sorted(repository_versions.items())),
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect default HEAD and all public Git tags of eligible PBPK model repositories."
    )
    parser.add_argument("--repository-assets", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    result = run_version_census(
        repository_assets_path=args.repository_assets,
        mirror_root=args.mirror_root,
        output_directory=args.output,
        workers=args.workers,
        resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
