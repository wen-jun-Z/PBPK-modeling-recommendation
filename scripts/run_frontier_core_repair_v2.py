"""Isolated v2 view correction; reuse only identical actual v1 repair calls.

Navigation citations do not supply scope evidence but do not invalidate other
supporting excerpts. All v1 outputs, including unused attempts, remain frozen.
"""
from __future__ import annotations

import copy
from pathlib import Path

import run_frontier_core_repair as v1
import frontier_scope_repair_v2 as correction

ROOT, OUT, base = v1.ROOT, v1.OUT, v1.base
SCRIPT = Path(__file__).resolve()
PROFILE = "core145-repair-v2"
PARENT_QUAL, PARENT_ASSESS, PARENT_RUN = v1.QUAL, v1.ASSESS, v1.RUN
QUAL = OUT / "qualification_repaired_v2"
ASSESS = OUT / "assessment_repaired_v2"
RUN = ROOT / "runtime/deployment/frontier_core_145_repair_v2"
QUAL_RUN = RUN / "qualification"
_prepare = v1.prepare_plan
_prepare_core = v1.prepare_core
_replay = v1.replay
_adopted = v1.adopted_attempts
_scope = v1.configure_scope
_installed = False


def configure_scope(folder=None, runtime=None):
    # v1's default arguments captured old paths when its function was defined.
    return _scope(QUAL if folder is None else folder, QUAL_RUN if runtime is None else runtime)


def partial_snapshot():
    for directory in (PARENT_RUN, PARENT_RUN / "qualification"):
        for name in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(directory / name, {})):
                raise RuntimeError("The v1 correction must be fully stopped before v2 snapshots it")
    files = [Path(v1.__file__), ROOT / "scripts/frontier_scope_repair.py"]
    for directory in (PARENT_QUAL, PARENT_ASSESS):
        if not directory.is_dir():
            raise ValueError("Complete v1 correction directory is required")
        files.extend(p for p in directory.rglob("*") if p.is_file())
    for path in files:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("v1 parent snapshot escaped the project")
    return {str(p.relative_to(ROOT)): base.file_sha(p) for p in sorted(set(files))}


def prepare_plan():
    snapshot = partial_snapshot()
    # Link neither writable records nor predictions: only immutable input
    # definitions are shared, and reused actual calls get independent envelopes.
    parent_protocol = base.read(PARENT_QUAL / "judge_protocol.json")
    if not parent_protocol or parent_protocol.get("version") != "core145-repair-v1":
        raise ValueError("Unexpected predecessor correction protocol")
    parent_plan = base.read(PARENT_ASSESS / "core_plan.json")
    v1.verify_files(parent_plan["files"])
    previous_freeze = v1.core.freeze
    def freeze(path, value):
        if path == ASSESS / "core_plan.json":
            value = copy.deepcopy(value)
            value["files"].update(snapshot)
            value.update(parent_partial_snapshot=snapshot,
                parent_partial_policy="All v1 scientific artifacts and unused attempts are bound by hashes and preserved. v2 replays v8 initial outputs. Actual v1 repair generations are reused only with identical case inputs, model/decoding, prior output and complete feedback.",
                actual_call_reuse="Derived envelopes explicitly name reused_actual_v1_generation and bind the complete original journal file and row; no reused output is claimed as new GPU inference.")
        elif path == QUAL / "judge_protocol.json":
            value = copy.deepcopy(value)
            value.update(parent_partial_snapshot_sha256=base.sha(base.canonical(snapshot).encode()),
                parent_partial_protocol_sha256=base.file_sha(PARENT_QUAL / "judge_protocol.json"),
                navigation_policy="Pure navigation excerpts contribute no scope evidence; their real nonempty source/citation existence is preserved. Genuine supporting cited body prose is still checked normally.",
                actual_v1_generation_reuse="Only identical complete model messages and decoding; old generation metrics/reasoning references retained and source-bound.")
            value["implementation_sha256"].update({Path(v1.__file__).name: base.file_sha(v1.__file__),
                "frontier_scope_repair.py": base.file_sha(ROOT / "scripts/frontier_scope_repair.py")})
        return previous_freeze(path, value)
    v1.core.freeze = freeze
    try:
        result = _prepare()
    finally:
        v1.core.freeze = previous_freeze
    # _prepare's returned object predates wrapper freeze additions.
    return (*result[:3], base.read(QUAL / "judge_protocol.json"))


def equal_generation_contract():
    parent, current = (base.read(p / "judge_protocol.json") for p in (PARENT_QUAL, QUAL))
    for key in ("system_prompt", "system_prompt_sha256", "primary_repo", "primary_revision",
                "secondary_repo", "secondary_revision", "primary_weights_manifest_sha256", "secondary_weights_manifest_sha256"):
        if parent.get(key) != current.get(key):
            raise ValueError("Actual-call reuse requires identical models and system prompt")
    if parent["generation"] != current["generation"]:
        raise ValueError("Actual-call reuse requires identical generation budgets, decoding and reasoning")
    if parent.get("execution_environments") != current.get("execution_environments"):
        raise ValueError("Actual-call reuse requires identical execution environments")
    changed_wrappers = {"run_frontier_core_repair.py", "run_frontier_core_repair_v2.py",
                        "frontier_scope_repair.py", "frontier_scope_repair_v2.py"}
    implementations = [{k: v for k,v in p.get("implementation_sha256", {}).items() if k not in changed_wrappers}
                       for p in (parent, current)]
    if not implementations[0] or implementations[0] != implementations[1]:
        raise ValueError("Actual-call reuse requires identical actual backend implementations")
    for cohort in ("legacy", "fresh"):
        for name in ("cases.jsonl", "manifest.json"):
            if base.file_sha(parent_folder(cohort) / name) != base.file_sha(v1.folder(cohort) / name):
                raise ValueError("Actual-call reuse requires identical supplied case packets and requests")


def parent_folder(cohort):
    return PARENT_QUAL / ("validation/fresh" if cohort == "fresh" else "validation")


def reusable_row(old, request):
    """The feedback and assistant prefix are part of the actual model input."""
    required = ("case_id", "model", "cohort", "repair_kind", "prior_output_sha256", "feedback")
    return (all(old.get(key) == request.get(key) for key in required) and
            isinstance(old.get("output"), str) and isinstance(old.get("metrics"), dict))


def import_actual_generation(cohort, case, request):
    model, kind = request["model"], request["repair_kind"]
    source = parent_folder(cohort) / f"{model}_{kind}_repair_journal.jsonl"
    if not source.is_file():
        return False
    records = v1.rows_index(source)
    old = records.get(case["case_id"])
    if old is None or not reusable_row(old, request):
        return False
    parent_pilot = parent_folder(cohort) / f"{model}_protocol.json"
    if old.get("pilot_protocol_sha256") != base.file_sha(parent_pilot):
        raise ValueError("v1 actual repair record has invalid pilot provenance")
    if base.sha(request["prior_output"].encode()) != old["prior_output_sha256"]:
        raise ValueError("v1 actual repair used a different assistant prefix")
    value = {k: v for k, v in request.items() if k != "prior_output"}
    value.update(output=old["output"], metrics=copy.deepcopy(old["metrics"]), at=old["at"],
        generation_origin="reused_actual_v1_generation", reused_at=base.now(),
        reused_from={"journal_path": str(source.relative_to(ROOT)), "journal_sha256": base.file_sha(source),
            "record_sha256": base.sha(base.canonical(old).encode()),
            "pilot_protocol_sha256": old["pilot_protocol_sha256"],
            "qualification_protocol_sha256": base.file_sha(PARENT_QUAL / "judge_protocol.json")})
    v1.base.append_rows(v1.folder(cohort) / source.name, [value])
    return True


def verify_reused_record(record):
    origin = record.get("reused_from", {})
    relative = origin.get("journal_path", "")
    source = ROOT / relative
    if (not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or
            not source.resolve().is_relative_to(PARENT_QUAL.resolve()) or
            base.file_sha(source) != origin.get("journal_sha256")):
        raise ValueError("Reused actual v1 generation journal changed")
    old = v1.rows_index(source).get(record["case_id"])
    if (not old or base.sha(base.canonical(old).encode()) != origin.get("record_sha256") or
            not reusable_row(old, record) or old["output"] != record["output"] or old["metrics"] != record["metrics"]):
        raise ValueError("Reused actual v1 generation was altered")
    if (origin.get("pilot_protocol_sha256") != old.get("pilot_protocol_sha256") or
            origin.get("qualification_protocol_sha256") != base.file_sha(PARENT_QUAL / "judge_protocol.json")):
        raise ValueError("Reused actual v1 generation protocol changed")


def adopted_attempts(model, cohort, case, old, journals):
    attempts, request = _adopted(model, cohort, case, old, journals)
    for attempt in attempts:
        origin = attempt.get("generation_origin", "")
        if not origin.startswith("new_"):
            continue
        kind = origin.removeprefix("new_")
        generated = journals[kind][case["case_id"]]
        if generated.get("generation_origin") == "reused_actual_v1_generation":
            verify_reused_record(generated)
            attempt.update(generation_origin="reused_actual_v1_generation",
                           reused_from=copy.deepcopy(generated["reused_from"]))
    return attempts, request


def replay(model):
    equal_generation_contract()
    pending = _replay(model)
    while pending:
        reused = sum(import_actual_generation(cohort, case, request)
                     for cohort, case, old, request in pending)
        if not reused:
            break
        pending = _replay(model)
    return pending


def prepare_core():
    previous_freeze = v1.core.freeze
    def freeze(path, value):
        if path == ASSESS / "judge_protocol.json":
            # The inherited core code has a historic logical directory name.
            # Bind the real v2 directory as well as the evaluator's logical key.
            value["version"] = "pbpk-core145-repaired-production-v2"
            hashes = value["input_sha256"]
            for name in ("sources.jsonl", "source_supplements/sources.jsonl"):
                hashes.pop("assessment_repaired/" + name, None)
                hashes[ASSESS.name + "/" + name] = base.file_sha(ASSESS / name)
        return previous_freeze(path, value)
    v1.core.freeze = freeze
    try:
        result = _prepare_core()
    finally:
        v1.core.freeze = previous_freeze
    return (*result[:3], base.read(ASSESS / "judge_protocol.json"))


def install():
    global _installed
    if _installed:
        return
    if v1._installed:
        raise RuntimeError("Start v2 in its own interpreter, before initializing v1")
    v1.correction, v1.SCRIPT, v1.PROFILE = correction, SCRIPT, PROFILE
    v1.QUAL, v1.ASSESS, v1.RUN, v1.QUAL_RUN = QUAL, ASSESS, RUN, QUAL_RUN
    v1.configure_scope, v1.prepare_plan = configure_scope, prepare_plan
    v1.prepare_core = prepare_core
    v1.replay, v1.adopted_attempts = replay, adopted_attempts
    _installed = True


if __name__ == "__main__":
    install()
    v1.main()
