"""Citation-complete development regression using the frozen fast scheduler."""
from __future__ import annotations

from pathlib import Path
import re

import run_frontier_scoring_fast as fast

r, ROOT = fast.reasoned, fast.ROOT
LOAD, CONFIGURE, PREPARE, PUBLISH, FREEZE = fast.load_profile, fast.configure, fast.prepare, fast.publish_summary, fast.FREEZE
ORIGINAL_SCOPE_GUARD = r.base.validate_explicit_scope
DISCLOSURE = ("All 192 synthetic controls are development regression after observed prior outputs; "
    "the 573 real-source cases are source coverage without scientific gold. "
    "No blind or independent admission accuracy is claimed.")
GUARD = None
GUARD_PROTOCOL = None


def verify_bindings():
    global GUARD, GUARD_PROTOCOL
    c = r.CONFIG
    if (c.get("citation_runner_file") != "scripts/run_frontier_scoring_citation_complete.py" or
            r.base.file_sha(Path(__file__)) != c.get("citation_runner_sha256") or
            c.get("regression_disclosure") != DISCLOSURE):
        raise ValueError("Citation-complete runner or development disclosure changed")
    module = r.shared.import_frozen(c["request_scope_guard_file"], c["request_scope_guard_sha256"],
                                   f"request_scope_guard_{r.PROFILE}")
    protocol = module.guard_protocol()
    if protocol != c["request_scope_guard_protocol"]:
        raise ValueError("Explicit-request scope guard protocol changed")
    GUARD, GUARD_PROTOCOL = module.guard_factory(ORIGINAL_SCOPE_GUARD), protocol
    snapshot = fast.bound_artifact("v7_failure_snapshot_file")
    if not isinstance(snapshot.get("files"), dict) or not snapshot["files"]:
        raise ValueError("Preceding failed run requires bound artifact hashes")
    for relative, digest in snapshot["files"].items():
        if r.base.file_sha(fast.scoped_file(relative)) != digest:
            raise ValueError("Preceding failed run artifact changed")
    return snapshot


def load_profile(slug):
    if not re.fullmatch(r"v[0-9]{1,3}", slug) or int(slug[1:]) < 8:
        raise ValueError("Citation development requires isolated profile v8 or later")
    r.shared.DISCLOSURE = DISCLOSURE
    value = LOAD(slug)
    verify_bindings()
    return value


def configure():
    r.shared.DISCLOSURE = DISCLOSURE
    CONFIGURE()
    if GUARD is None: raise ValueError("Load the frozen explicit-request guard before execution")
    # Pilot validation, production validation and v3 replay all dereference this
    # same assessment module at call time. Reassert after every nested configure.
    r.base.validate_explicit_scope = GUARD


def prepare():
    snapshot = verify_bindings()
    configure()
    def freeze(path, value):
        if path == r.ASSESS / "judge_protocol.json":
            value.update(version="pbpk-citation-complete-development-scoring-v1",
                scope_guard=GUARD_PROTOCOL,
                v7_failure_snapshot=snapshot,
                v7_failure_snapshot_sha256=r.CONFIG["v7_failure_snapshot_file_sha256"],
                fresh_origin="Historical fresh32 definitions and IDs reused as development regression; prior model outputs observed; all new predictions regenerated",
                optimized_runtime_scope="Qwen uses the previously validated isolated fast environment; this profile changes the citation-completeness prompt and explicit-request scope guard, while model weights, decoding, thinking budget and numerical gates stay unchanged")
            value["implementation_sha256"].update({Path(__file__).name: r.base.file_sha(Path(__file__)),
                Path(r.CONFIG["request_scope_guard_file"]).name: r.CONFIG["request_scope_guard_sha256"]})
            value["validation_policy"].update(interpretation=DISCLOSURE,
                synthetic_development_regression_count=192, real_source_coverage_count=573,
                independent_control_count=0,
                historical_role_names="fresh32, independent_admission and prospective_source_coverage are unchanged input IDs only; they do not establish blind evaluation in this post-development profile")
            value["judge_development"]["v3_validation"] = "All inherited synthetic control roles now form development regression; real-source cases have no gold labels"
            reviewed = value["judge_development"]["reviewed_profile"]
            reviewed.update(new_control_count=0, synthetic_development_regression_count=192,
                fresh_definition="Historical 32-case file retained unchanged after prior model exposure; development regression, not newly authored or unseen controls")
        return FREEZE(path, value)
    previous = fast.FREEZE
    fast.FREEZE = freeze
    try:
        result = PREPARE()
    finally:
        fast.FREEZE = previous
    if result[3] != r.base.read(r.ASSESS / "judge_protocol.json"):
        raise ValueError("Returned development protocol differs from frozen disk protocol")
    return result


def command(python, *arguments):
    return [str(python), str(Path(__file__).resolve()), "--profile", r.PROFILE, *arguments]


def publish_summary():
    PUBLISH()
    path = ROOT / "experiments/model_recommendation/results_summary/README.md"
    text = path.read_text()
    first, last = f"<!-- reasoned-{r.PROFILE}:start -->", f"<!-- reasoned-{r.PROFILE}:end -->"
    start, end = text.index(first), text.index(last) + len(last)
    block = text[start:end].replace("scripts/run_frontier_scoring_fast.py", "scripts/run_frontier_scoring_citation_complete.py")
    old = "提示词保持 v5，733条回归和此前未被模型调用的32条控制全部重新生成；四组门槛、来源覆盖与39例来源复核不变。"
    new = ("本版完善完整引用链提示，并按实际请求检查用途约束。192条合成控制均为开发回归，"
        "573条原文案例用于来源覆盖；全部重新生成，不宣称盲测或独立准确率。四组数值门槛、494条代表来源覆盖与39例来源复核不变。")
    if old not in block: raise ValueError("Inherited summary disclosure changed")
    block = block.replace(old, new)
    path.write_text(text[:start] + block + text[end:])


def install():
    fast.install()
    fast.configure, fast.prepare, fast.command, fast.publish_summary = configure, prepare, command, publish_summary
    r.load_profile, r.configure, r.prepare, r.command, r.publish_summary = load_profile, configure, prepare, command, publish_summary


if __name__ == "__main__":
    install()
    r.main()
