"""Isolated optimized-kernel environment around the frozen reasoned scheduler."""
from __future__ import annotations

from pathlib import Path
import os
import re
import sys

import run_frontier_scoring_reasoned as reasoned

ROOT = reasoned.ROOT
FAST = ROOT / "runtime/judge-fast-env/bin/python"
LOAD, CONFIGURE, PREPARE = reasoned.load_profile, reasoned.configure, reasoned.prepare
EXECUTE, PUBLISH, FREEZE = reasoned.execute, reasoned.publish_summary, reasoned.ORIGINAL_FREEZE
FROZEN_RUNNER_SHA = "b5643e31c8bf1b556d6e84ca801ba886df817b0e9ecd0140d3e0a06a7179c76c"


def scoped_file(relative):
    if not isinstance(relative, str): raise ValueError("Missing project artifact path")
    path = ROOT / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts or
            not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file()):
        raise ValueError("Fast profile artifact must be an existing project file")
    return path


def bound_artifact(field):
    path = scoped_file(reasoned.CONFIG[field])
    if reasoned.base.file_sha(path) != reasoned.CONFIG[field + "_sha256"]:
        raise ValueError("Frozen fast profile artifact changed: " + field)
    return reasoned.base.read(path)


def verify_bindings():
    config = reasoned.CONFIG
    if (config.get("fast_runner_file") != "scripts/run_frontier_scoring_fast.py" or
            reasoned.base.file_sha(Path(__file__)) != config.get("fast_runner_sha256") or
            reasoned.base.file_sha(ROOT / "scripts/run_frontier_scoring_reasoned.py") != FROZEN_RUNNER_SHA):
        raise ValueError("Frozen fast wrapper or reasoned scheduler changed")
    environment = bound_artifact("fast_environment_manifest")
    if environment.get("interpreter") != "runtime/judge-fast-env/bin/python" or not FAST.is_file():
        raise ValueError("Unexpected isolated Qwen interpreter")
    files = environment.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Fast environment requires hashed file artifacts")
    for relative, digest in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or reasoned.base.file_sha(scoped_file(relative)) != digest:
            raise ValueError("Frozen fast environment file changed")
    comparison = bound_artifact("v6_control_comparison_file")
    return environment, comparison


def load_profile(slug):
    if not re.fullmatch(r"v[0-9]{1,3}", slug) or int(slug[1:]) < 7:
        raise ValueError("Optimized runtime requires an isolated profile v7 or later")
    value = LOAD(slug)
    verify_bindings()
    return value


def configure():
    # The original configure is also called by nested generic preparation.  Its
    # function globals now name these wrappers, so reinjection stays idempotent.
    reasoned.MODERN = FAST
    reasoned.utilities.MODERN = FAST
    # initialize() deliberately clears inherited environment variables; restore
    # the local-only dispatch setting before each real backend import/load.
    os.environ.update(USE_HUB_KERNELS="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    CONFIGURE()


def prepare():
    configure()
    environment, comparison = verify_bindings()
    def freeze(path, value):
        if path == reasoned.ASSESS / "judge_protocol.json":
            value.update(version="pbpk-reasoned-fast-primary-scoring-v1",
                fast_environment_manifest=environment,
                fast_environment_manifest_sha256=reasoned.CONFIG["fast_environment_manifest_sha256"],
                v6_control_comparison=comparison,
                v6_control_comparison_sha256=reasoned.CONFIG["v6_control_comparison_file_sha256"],
                optimized_runtime_scope="Qwen primary only; official causal-conv1d kernels; identical weights, prompt, reasoning budget, decoder and gates; all predictions regenerated")
            value["implementation_sha256"][Path(__file__).name] = reasoned.base.file_sha(Path(__file__))
            qwen = value["execution_environments"]["qwen"]
            if Path(qwen["executable"]).absolute() != FAST.absolute():
                raise ValueError("Prepared Qwen environment is not the isolated fast interpreter")
            qwen["fast_environment_manifest_sha256"] = reasoned.CONFIG["fast_environment_manifest_sha256"]
            # The wrapped prepare returns this same object; mutate before it is
            # first frozen so return value, disk and assessment manifest agree.
        return FREEZE(path, value)
    previous = reasoned.ORIGINAL_FREEZE
    reasoned.ORIGINAL_FREEZE = freeze
    try:
        result = PREPARE()
    finally:
        reasoned.ORIGINAL_FREEZE = previous
    if result[3] != reasoned.base.read(reasoned.ASSESS / "judge_protocol.json"):
        raise ValueError("Returned optimized protocol differs from frozen disk protocol")
    return result


def command(python, *arguments):
    return [str(python), str(Path(__file__).resolve()), "--profile", reasoned.PROFILE, *arguments]


def execute(action, model="qwen", judge="primary", job="all"):
    configure()
    if ((action == "pilot" and model == "qwen") or (action == "run" and judge == "primary")):
        if Path(sys.prefix).resolve() != FAST.parent.parent.resolve():
            raise RuntimeError("Qwen GPU stages require runtime/judge-fast-env/bin/python")
    return EXECUTE(action, model, judge, job)


def publish_summary():
    PUBLISH()
    path = ROOT / "experiments/model_recommendation/results_summary/README.md"
    text = path.read_text()
    first, last = f"<!-- reasoned-{reasoned.PROFILE}:start -->", f"<!-- reasoned-{reasoned.PROFILE}:end -->"
    start, end = text.index(first), text.index(last) + len(last)
    block = text[start:end].replace("scripts/run_frontier_scoring_reasoned.py", "scripts/run_frontier_scoring_fast.py")
    block = block.replace("Prompts remain v5", "Qwen uses an isolated causal-conv1d acceleration environment; Phi is unchanged. Prompts remain v5")
    path.write_text(text[:start] + block + text[end:])


def install():
    reasoned.load_profile, reasoned.configure, reasoned.prepare = load_profile, configure, prepare
    reasoned.command, reasoned.execute, reasoned.publish_summary = command, execute, publish_summary


if __name__ == "__main__":
    install()
    reasoned.main()
