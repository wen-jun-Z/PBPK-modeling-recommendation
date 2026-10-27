#!/usr/bin/env python3
"""Recover explicit model licences at the frozen source commit, without relabelling catalogues.

Only a grant made in the source repository itself is accepted. A link to Suite's
licence on its own, public availability, and a different repository's licence are
not grants for a model. No credentials, model execution or observed data are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = Path("experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl")
DEFAULT_DIRECTORY = Path("experiments/model_recommendation/execution_validation_20261001/engineering_repairs/license_audit")
DEFAULT_RECEIPT = DEFAULT_DIRECTORY / "manifest.json"
VERSION = "same_commit_model_license_v1"
MAX_DOCUMENT_BYTES = 512 * 1024


def _path(root, value):
    root = Path(root).resolve()
    p = Path(value)
    p = (p if p.is_absolute() else root / p).resolve()
    if not p.is_relative_to(root) or "sealed" in {x.casefold() for x in p.parts}:
        raise ValueError("PATH_OUTSIDE_PROJECT_OR_SEALED")
    return p


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root, path):
    return str(_path(root, path).relative_to(Path(root).resolve()))


def _provenance(candidate):
    return candidate.get("catalog_provenance", candidate.get("provenance", {}))


def _grant(text, document_name):
    """Deliberately narrow parser: an explicit model GPLv2 grant, not a mention."""
    if Path(document_name).name.casefold().startswith("readme"):
        sections = re.split(r"(?m)^#{1,6}\s+", text)
        for section in sections:
            if not re.match(r"licen[cs]e\s*\n", section, re.I):
                continue
            normalized = " ".join(section.split())
            if re.search(r"\b(?:the|this) model(?:s)? (?:is|are) distributed under (?:the )?\[?GPL\s*v?2\b", normalized, re.I):
                return {"license_id": "GPL-2.0", "scope": "explicit_source_repository_model_grant", "grant_text": section.strip()}
    # A root GPL text is repository-scoped; require the complete identifying
    # preamble and operative copying/distribution section, not a README keyword.
    elif Path(document_name).parent == Path(".") and Path(document_name).name.casefold().startswith(("license", "copying")):
        if ("GNU GENERAL PUBLIC LICENSE" in text and
                re.search(r"Version 2,?\s+June 1991", text) and
                "TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION" in text):
            return {"license_id": "GPL-2.0", "scope": "source_repository_root_license", "grant_text": "GNU GENERAL PUBLIC LICENSE, Version 2, June 1991"}
    return None


def _repository_binding(candidate, root):
    prov = _provenance(candidate)
    source = _path(root, prov["source_path"])
    if _sha(source) != prov["source_sha256"]:
        raise ValueError("CANDIDATE_SOURCE_SHA_MISMATCH")
    commit = prov.get("repository_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("FULL_FROZEN_COMMIT_REQUIRED")
    manifest = source.parent.parent / "asset_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        if data.get("commit") != commit:
            raise ValueError("ASSET_MANIFEST_COMMIT_MISMATCH")
        directory = _path(root, data["source_directory"])
        relative = str(source.relative_to(directory))
        matches = [x for x in data.get("files", []) if x.get("path") == relative]
        if len(matches) != 1 or matches[0].get("sha256") != prov["source_sha256"]:
            raise ValueError("ASSET_MANIFEST_SOURCE_BINDING_MISMATCH")
        repository = data["repository"]
        proof = {"path": _relative(root, manifest), "sha256": _sha(manifest)}
    else:
        library = _path(root, "artifacts/pbpk-model-assets/OSP-PBPK-Model-Library")
        if not source.is_relative_to(library):
            raise ValueError("SOURCE_REPOSITORY_BINDING_UNAVAILABLE")
        repository, relative, proof = "Open-Systems-Pharmacology/OSP-PBPK-Model-Library", str(source.relative_to(library)), None
    if not re.fullmatch(r"Open-Systems-Pharmacology/[A-Za-z0-9_.-]+", repository):
        raise ValueError("NON_ORIGINAL_REPOSITORY_NOT_REGISTERED")
    return {"repository": repository, "commit": commit, "source_relative_path": relative,
            "asset_manifest": proof}


def _url(repository, commit, path):
    return f"https://raw.githubusercontent.com/{repository}/{commit}/{quote(path, safe='/')}"


def _fetch(repository, commit, name):
    url = _url(repository, commit, name)
    try:
        request = Request(url, headers={"User-Agent": "PBPK-project-same-commit-license-audit"})
        with urlopen(request, timeout=20) as response:
            if response.geturl() != url:
                raise ValueError("CROSS_REPOSITORY_REDIRECT_NOT_ACCEPTED")
            body = response.read(MAX_DOCUMENT_BYTES + 1)
        if len(body) > MAX_DOCUMENT_BYTES:
            raise ValueError("LICENSE_DOCUMENT_TOO_LARGE")
        text = body.decode("utf-8-sig")
        return {"name": name, "url": url, "status": "fetched", "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body), "grant": _grant(text, name)}, body
    except (HTTPError, URLError, TimeoutError, ValueError, UnicodeError) as exc:
        return {"name": name, "url": url, "status": "unavailable", "error": type(exc).__name__ + ": " + str(exc)}, None


def collect(*, project_root=ROOT, candidate_audit=DEFAULT_AUDIT, directory=DEFAULT_DIRECTORY):
    root = Path(project_root).resolve()
    audit_path, output = _path(root, candidate_audit), _path(root, directory)
    receipt_path = output / "manifest.json"
    if receipt_path.exists():
        raise ValueError("RECEIPT_EXISTS_USE_A_NEW_DIRECTORY")
    rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("identity_match") and _provenance(r).get("license_status") != "allowed"]
    candidates = {r["candidate_id"]: r for r in rows}
    entries, groups = {}, {}
    for cid, candidate in sorted(candidates.items()):
        entry = {"candidate_id": cid, "catalog_provenance": _provenance(candidate),
                 "status": "unknown", "reason": "NO_EXPLICIT_ORIGINAL_SOURCE_GRANT_AT_FROZEN_COMMIT"}
        try:
            binding = _repository_binding(candidate, root)
            entry["repository_binding"] = binding
            groups.setdefault((binding["repository"], binding["commit"]), []).append(cid)
        except (KeyError, ValueError, OSError) as exc:
            entry["reason"] = str(exc)
        entries[cid] = entry
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    # Fetch only copyright metadata. Do not recursively crawl model/data files.
    tasks = [(repo, commit, "README.md") for repo, commit in sorted(groups)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for phase in range(2):
            futures = [(task, pool.submit(_fetch, *task)) for task in tasks]
            for (repo, commit, name), future in futures:
                doc, body = future.result()
                doc.update(repository=repo, commit=commit)
                if body is not None:
                    local = output / "documents" / (repo.split("/")[1] + "__" + commit) / name
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.write_bytes(body)
                    doc["path"] = _relative(root, local)
                documents.append(doc)
            granted = {(d["repository"], d["commit"]) for d in documents if d.get("grant")}
            tasks = [(repo, commit, name) for repo, commit in sorted(groups) if (repo, commit) not in granted
                     for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")]
    for (repo, commit), ids in groups.items():
        grants = [d for d in documents if d["repository"] == repo and d["commit"] == commit and d.get("grant")]
        if grants:
            for cid in ids:
                entries[cid].update(status="license_verified", reason="EXPLICIT_GRANT_IN_FROZEN_SOURCE_REPOSITORY",
                                    license_id=grants[0]["grant"]["license_id"], evidence_paths=[d["path"] for d in grants])
    allowed = {cid for cid, entry in entries.items() if entry["status"] == "license_verified"}
    receipt = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "candidate_audit": {"path": _relative(root, audit_path), "sha256": _sha(audit_path)},
               "collector": {"path": _relative(ROOT, Path(__file__)), "sha256": _sha(Path(__file__))},
               "scope": "licence evidence only; unchanged frozen candidate catalogue; no scientific qualification",
               "limitations": ["No rights inferred from another repository or public accessibility.",
                               "GPL redistribution obligations and original attribution remain applicable.",
                               "No determination of rights in third-party clinical observations."],
               "counts": {"unresolved_pairs": len(rows), "unresolved_candidates": len(entries),
                          "verified_candidates": len(allowed), "verified_pairs": sum(r["candidate_id"] in allowed for r in rows),
                          "remaining_unknown_pairs": sum(r["candidate_id"] not in allowed for r in rows)},
               "documents": documents, "candidates": list(entries.values())}
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def audit_candidate_license(candidate, *, project_root=ROOT, receipt_path=DEFAULT_RECEIPT):
    """Offline receipt recheck; pass an audit row or frozen catalogue candidate.

    An executor may use ``status == 'license_verified'`` to resolve its licence
    gate while retaining the original catalogue provenance and this receipt hash.
    """
    root = Path(project_root).resolve()
    result = {"status": "unknown", "candidate_id": candidate.get("candidate_id"), "scientific_qualification": False}
    try:
        path = _path(root, receipt_path)
        receipt = json.loads(path.read_text())
        if receipt.get("version") != VERSION:
            raise ValueError("LICENSE_RECEIPT_VERSION_MISMATCH")
        audit = receipt["candidate_audit"]
        if _sha(_path(root, audit["path"])) != audit["sha256"]:
            raise ValueError("FROZEN_CANDIDATE_AUDIT_CHANGED")
        matches = [r for r in receipt["candidates"] if r.get("candidate_id") == candidate.get("candidate_id")]
        if len(matches) != 1:
            raise ValueError("CANDIDATE_LICENSE_RECEIPT_NOT_UNIQUE_OR_MISSING")
        entry = matches[0]
        if entry["catalog_provenance"] != _provenance(candidate):
            raise ValueError("CANDIDATE_FROZEN_PROVENANCE_MISMATCH")
        binding = _repository_binding(candidate, root)
        if binding != entry["repository_binding"]:
            raise ValueError("CURRENT_REPOSITORY_BINDING_MISMATCH")
        if entry.get("status") != "license_verified":
            raise ValueError(entry.get("reason", "LICENSE_UNKNOWN"))
        accepted = []
        for evidence_path in entry.get("evidence_paths", []):
            docs = [d for d in receipt["documents"] if d.get("path") == evidence_path]
            if len(docs) != 1:
                raise ValueError("LICENSE_DOCUMENT_NOT_UNIQUE_OR_MISSING")
            doc = docs[0]
            if (doc["repository"] != binding["repository"] or doc["commit"] != binding["commit"] or
                    doc["url"] != _url(binding["repository"], binding["commit"], doc["name"])):
                raise ValueError("LICENSE_DOCUMENT_SOURCE_COMMIT_MISMATCH")
            document = _path(root, evidence_path)
            if _sha(document) != doc["sha256"]:
                raise ValueError("LICENSE_DOCUMENT_SHA_MISMATCH")
            grant = _grant(document.read_text(encoding="utf-8-sig"), doc["name"])
            if not grant or grant != doc.get("grant") or grant["license_id"] != entry.get("license_id"):
                raise ValueError("EXPLICIT_MODEL_LICENSE_GRANT_NOT_REPRODUCIBLE")
            accepted.append({"path": evidence_path, "sha256": doc["sha256"], "url": doc["url"]})
        if not accepted:
            raise ValueError("LICENSE_GRANT_EVIDENCE_EMPTY")
        result.update(status="license_verified", license_id=entry["license_id"], evidence=accepted,
                      receipt={"path": _relative(root, path), "sha256": _sha(path)}, limits=receipt["limitations"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_DIRECTORY))
    args = parser.parse_args()
    receipt = collect(candidate_audit=args.candidate_audit, directory=args.output)
    print(json.dumps(receipt["counts"], indent=2))


if __name__ == "__main__":
    main()
