"""Collect pinned OSP model assets for the coverage gaps or the public library."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/data_expansion_20260907"
ASSETS = ROOT / "artifacts/pbpk-model-assets/coverage_expansion_20260907"
TARGETS = {
    "caffeine": "Example_Caffeine",
    "dextromethorphan": "Dextromethorphan-Model",
    "metformin": "Metformin-Model",
    "repaglinide": "Repaglinide-Model",
    "theophylline": "Theophylline-Model",
    "warfarin": "Warfarin-Model",
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "pbpk-project-model-expansion/1.0"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
        data = response.read(200 * 1024 * 1024 + 1)
    if len(data) > 200 * 1024 * 1024:
        raise ValueError("Repository response exceeds the 200 MiB collection bound")
    return data


def collect(item, *, revision=None):
    compound, name = item
    destination = ASSETS / (name if revision is None else name + "__" + revision[:12])
    manifest = destination / "asset_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    repository = "Open-Systems-Pharmacology/" + name
    inventory = json.loads((OUT / "repository_inventory.json").read_text())
    row = next(r for r in inventory if r["full_name"] == repository)
    commit_data = fetch(f"https://api.github.com/repos/{repository}/commits/{revision or row['default_branch']}")
    commit = json.loads(commit_data)["sha"]
    url = f"https://codeload.github.com/{repository}/tar.gz/{commit}"
    archive_bytes = fetch(url)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "source.tar.gz"
    archive.write_bytes(archive_bytes)
    (destination / "commit.json").write_bytes(commit_data)
    files = []
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isdir():
                continue
            relative = PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsupported repository archive member")
            relative = Path(*relative.parts[1:])
            target = destination / "source" / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Repository archive escaped destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.extractfile(member).read()
            target.write_bytes(data)
            files.append({"path": str(relative), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    result = {
        "compound": compound, "repository": repository,
        "source_url": f"https://github.com/{repository}/tree/{commit}",
        "commit": commit, "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "source_directory": str((destination / "source").relative_to(ROOT)),
        "files": files,
        "collection_status": "downloaded_not_scientifically_qualified",
        "license_status": "pending_readme_and_asset_review",
        "execution_status": "not_executed", "independent_validation_status": "not_run",
        "candidate_kind": "teaching_example" if name.startswith("Example_") else "public_model",
    }
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-public", action="store_true", help="Collect all official osp-model repositories with bounded writes")
    parser.add_argument("--batch", default="20260909")
    parser.add_argument("--exclude", action="append", default=[], help="Explicit repository name excluded after review")
    args = parser.parse_args()
    if args.all_public:
        import re
        if not re.fullmatch(r"[0-9]{8}", args.batch):
            parser.error("batch must be YYYYMMDD")
        from pbpk_kb.model_assets.public_collection import collect_public_models
        result = collect_public_models(
            ROOT / f"experiments/model_recommendation/model_expansion_{args.batch}",
            ROOT / f"artifacts/pbpk-model-assets/public_collection_{args.batch}",
            excluded=args.exclude,
        )
        raise SystemExit(0 if result["status"] == "completed" else 1)
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {compound: pool.submit(collect, (compound, name)) for compound, name in TARGETS.items()}
        for compound, future in futures.items():
            try:
                row = future.result()
            except Exception as exc:
                row = {"compound": compound, "collection_status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ["compound", "repository", "collection_status", "error_type"]}), flush=True)
    (OUT / "collected_models.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    if any(r["collection_status"] == "failed" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
