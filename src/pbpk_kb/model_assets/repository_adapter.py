from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pbpk_kb.model_assets.models import RepositorySnapshot


class RepositoryError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RepositoryError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _local_default_branch(repository: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "symbolic-ref",
            "--short",
            "refs/remotes/origin/HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().removeprefix("origin/")
    refs = _git(
        repository,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes/origin",
    ).splitlines()
    branches = {
        ref.removeprefix("origin/")
        for ref in refs
        if ref and ref != "origin/HEAD"
    }
    for candidate in ("main", "master"):
        if candidate in branches:
            return candidate
    if branches:
        return sorted(branches)[0]
    current = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return current.stdout.strip() if current.returncode == 0 else "master"


def mirror_repository(
    *,
    repository_url: str,
    destination: Path,
    source_id: str,
    commit: str | None = None,
) -> RepositorySnapshot:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", repository_url, str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RepositoryError(result.stderr.strip() or result.stdout.strip())
    _git(destination, "fetch", "--prune", "--tags", "origin")
    branch = _local_default_branch(destination)
    target = commit or f"origin/{branch}"
    _git(destination, "checkout", "--force", target)
    resolved_commit = _git(destination, "rev-parse", "HEAD")
    porcelain = _git(destination, "status", "--porcelain")
    return RepositorySnapshot(
        source_id=source_id,
        repository_url=repository_url,
        repository_path=str(destination.resolve()),
        commit_sha=resolved_commit,
        default_branch=branch,
        is_clean=not bool(porcelain),
        captured_at=datetime.now(UTC),
    )


def inspect_repository(
    repository: Path,
    *,
    source_id: str,
    repository_url: str,
) -> RepositorySnapshot:
    if not (repository / ".git").is_dir():
        raise RepositoryError(f"Not a Git repository: {repository}")
    remote_url = _git(repository, "remote", "get-url", "origin") or repository_url
    return RepositorySnapshot(
        source_id=source_id,
        repository_url=remote_url,
        repository_path=str(repository.resolve()),
        commit_sha=_git(repository, "rev-parse", "HEAD"),
        default_branch=_local_default_branch(repository),
        is_clean=not bool(_git(repository, "status", "--porcelain")),
        captured_at=datetime.now(UTC),
    )
