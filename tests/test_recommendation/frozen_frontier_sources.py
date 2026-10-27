"""Read exact historical experiment sources from local Git, never the network.

The English presentation revision is intentionally not interchangeable with a
historical source fingerprint. These helpers only build isolated test fixtures.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REVISION = "5a17a155a997831f8ce78da0337a6a981e534856"
SOURCE_SHA256 = {
    "run_frontier_assessment.py": "af598badfe061d545f09356cf2e1921a3818c8b98c05b24b46ad73da3c6a6a03",
    "run_frontier_scoring.py": "90974000a9d31d23deb100155c27763839070793e6ec18b183eb768a160d112e",
    "run_frontier_scoring_v3.py": "f389c9093640325320f5ed7306fd0151f87f88506878bbb5dda1e45d909cc2f2",
    "run_frontier_scoring_reviewed.py": "6d3f42c3ab7d7411e2df6880aa7a8175f4334307f3bbc3c505b08e7b52c6f57b",
    "run_frontier_scoring_reasoned.py": "b5643e31c8bf1b556d6e84ca801ba886df817b0e9ecd0140d3e0a06a7179c76c",
    "run_frontier_scoring_fast.py": "3499524dca7c236a4e9902d3e5be3a23e0c1c8a236afa37c6b030d5ed396b6ee",
    "run_frontier_scoring_citation_complete.py": "5e07bc9750b2c9d62602f793b843d096fd8539629edfe4a9af9eec335003fd1b",
    "run_frontier_scoring_selected_v3.py": "b05a4c7f0c13281db70ff7261e6bc049dc82736bbcf45ad065fdee8b16ec58d0",
    "run_frontier_scoring_conditional_v4.py": "a175b0637302c55964208f87ba489d5d6d9c93858d4cec363ad4b051c13ac86b",
}


def git_read(*arguments):
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(
            f"Historical source revision {REVISION} is unavailable locally. "
            "Use a checkout with complete project Git history; no download or fallback is performed."
        )
    return result.stdout


@lru_cache(maxsize=None)
def source_blob(name):
    if not re.fullmatch(r"[A-Za-z0-9_]+\.py", name):
        raise ValueError("Historical source must be a top-level project Python script")
    content = git_read("show", f"{REVISION}:scripts/{name}")
    expected = SOURCE_SHA256.get(name)
    if expected and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Pinned historical source does not match its original SHA256")
    return content


def historical_sha(name):
    if name not in SOURCE_SHA256:
        raise ValueError("Source is not in the historical fingerprint allowlist")
    return hashlib.sha256(source_blob(name)).hexdigest()


def materialize_scripts(destination):
    """Write temporary exact Git blobs; never modify live sources or fingerprints."""
    directory = Path(destination) / "scripts"
    directory.mkdir(parents=True)
    names = git_read("ls-tree", "-r", "--name-only", REVISION, "scripts").decode().splitlines()
    for relative in names:
        if re.fullmatch(r"scripts/[A-Za-z0-9_]+\.py", relative):
            name = Path(relative).name
            (directory / name).write_bytes(source_blob(name))


def materialize_profile_inputs(destination):
    """Copy only required metadata into the temporary scope; no model weights."""
    prefix = Path("experiments/model_recommendation/frontier_comparison_20260914")
    required = {
        "inputs/queries.jsonl", "inputs/candidates.jsonl", "assessment/judge_protocol.json",
        "assessment/sources.jsonl", "assessment/source_supplements/sources.jsonl",
        "assessment/measured_outcomes.json", "assessment/primary_weights.json",
        "assessment_qwen36/model_weights/qwen.json", "assessment_evidence_first_v3/judge_protocol.json",
        "assessment_evidence_first_v3/source_review_plan.json",
        "assessment_precedence_v5/prompt_freeze.json",
        "assessment_precedence_v5/validation/fresh/cases.jsonl",
        "assessment_precedence_v5/validation/fresh/manifest.json",
        "assessment_conditional_v4/judge_protocol.json",
        "assessment_conditional_v4/validation/phi_records.jsonl",
    }
    for assessment in ("assessment_qwen36", "assessment_qwen36_v2", "assessment_evidence_first_v3"):
        required.update(f"{assessment}/validation/{name}" for name in ("cases.jsonl", "manifest.json"))
    manifest = json.loads((ROOT / prefix / "assessment_evidence_first_v3/validation/manifest.json").read_text())
    required.update(manifest["historical_input_artifact_sha256"])
    paths = [prefix / name for name in sorted(required)] + [
        Path("runtime/deployment/judge_profiles/v5.json"),
        Path("runtime/deployment/judge_profiles/v5.sha256"),
    ]
    total = 0
    for relative in paths:
        source = ROOT / relative
        target = Path(destination) / relative
        if not source.resolve().is_relative_to(ROOT) or not target.resolve().is_relative_to(Path(destination).resolve()):
            raise ValueError("Historical fixture inputs must remain inside their project scope")
        total += source.stat().st_size
        if total > 128 * 1024**2:
            raise ValueError("Historical fixture metadata exceeds the 128 MiB test budget")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
