#!/usr/bin/env python3
"""Immutable 100-compound construction runs with separate held-out reporting.

The nominal fixed-auto and purpose-policy prediction are the same object. Only
the declared-use assessment differs. One hundred engineering cases are not one
hundred clinical curves. Evaluation documents are hashed during preparation,
but decoded only by the separate scorer after every case, including failures,
has a locked outcome. Existing single-case and paper-model code stays frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_key, "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprint_policy.purpose_design import plan_design, assess_design

VERSION = "purpose-hundred-runner-1"
MANIFEST_VERSION = "purpose-hundred-cohort-1"
DEFAULT_NAME = "purpose_hundred_20260930"
RANDOM_SEED = 20260930
MAX_WORKERS = 14
MONITOR_SECONDS = 1200
LIMIT_BYTES = 10 * 1024**3
SCOPES = {"aggregated_human_IV_PK_parameters", "study_curve"}
CODE = ("scripts/run_purpose_hundred.py", "scripts/design_evidence_pbpk.py",
        "src/pbpk_kb/blueprint_policy/purpose_design.py",
        "src/pbpk_kb/blueprint_policy/evidence_decision.py",
        "src/pbpk_kb/blueprint_policy/hundred_evaluation.py",
        "src/pbpk_kb/blueprints/pk_parameter_learning.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class Paused(RuntimeError):
    """Stop submissions while retaining completed and interrupted artifacts."""


class Monitor:
    def __init__(self, runner, run, protocol, state):
        self.runner, self.run, self.protocol, self.state = runner, run, protocol, state
        self.lock = threading.RLock()
        self.halted, self.closed = threading.Event(), threading.Event()
        self.last_hash = self.last_size = 0.

    def persist(self, **updates):
        with self.lock:
            self.state.update(updates, updated_at=now())
            save(self.run / "state.json", self.state)

    def halt(self, reason):
        with self.lock:
            self.halted.set()
            self.persist(status="paused", pause_reason=str(reason))

    def record_result(self, infrastructure_failure):
        with self.lock:
            count = self.state.get("consecutive_infrastructure_errors", 0)
            count = count + 1 if infrastructure_failure else 0
            self.persist(consecutive_infrastructure_errors=count)
            if count >= 3:
                self.halt("Three consecutive infrastructure errors; no further submissions")

    def check(self, force=False):
        with self.lock:
            if self.halted.is_set() or (self.run / "PAUSE").exists():
                raise Paused(self.state.get("pause_reason", "Pause requested"))
            stamp = time.monotonic()
            if force or stamp - self.last_size >= 15:
                size = self.runner.directory_size(self.run)
                self.state["monitor"].update(bytes=size, size_checked_at=now())
                self.last_size = stamp
                if size > self.protocol["directory_limit_bytes"]:
                    self.halt("Experiment exceeds 10 GiB; no further submissions")
                    raise Paused(self.state["pause_reason"])
            if force or stamp - self.last_hash >= self.protocol["monitor_seconds"]:
                try:
                    self.runner.verify(self.protocol["source_pins"])
                    self.runner.verify(self.protocol["code_pins"])
                    self.runner.checked(self.state["protocol"])
                    if self.runner.paper_pins() != self.protocol["paper_guard"]:
                        raise ValueError("Publication files changed")
                except (ValueError, OSError) as error:
                    self.halt("Source/code/publication integrity changed: " + str(error))
                    raise Paused(self.state["pause_reason"]) from error
                self.state["monitor"]["hashes_checked_at"] = now()
                self.last_hash = stamp
            self.persist()

    def watch(self):
        while not self.closed.wait(15):
            try:
                self.check()
            except Exception as error:
                self.halt(error)
                return


class Runner:
    def __init__(self, project_root=ROOT, scenario_simulator=None, scorer=None,
                 expected_compounds=100, adapter=None, identity_normalizer=None):
        self.root = Path(project_root).resolve(strict=True)
        if isinstance(expected_compounds, bool) or not isinstance(expected_compounds, int) or expected_compounds < 1:
            raise ValueError("Expected compound count must be a positive integer")
        self.expected_compounds = expected_compounds
        self.default = self.root / "experiments/blueprint_design" / DEFAULT_NAME
        self.scorer = scorer
        if adapter is None:
            path = self.root / "scripts/design_evidence_pbpk.py"
            spec = importlib.util.spec_from_file_location("_purpose_hundred_single_case_" + digest(str(self.root))[:12], path)
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
        self.adapter = adapter
        self.simulate = scenario_simulator or adapter.simulate_scenario
        if identity_normalizer is None:
            from pbpk_kb.blueprints.pk_parameter_learning import normalize_structure
            identity_normalizer = lambda value: normalize_structure(value)["identity_smiles"]
        self.normalize_identity = identity_normalizer

    def own_run(self, path):
        path = inside(path, self.root)
        if path.parent != self.root / "experiments/blueprint_design" or not path.name.startswith("purpose_hundred_"):
            raise ValueError("Use an immediate project experiments/blueprint_design/purpose_hundred_ directory")
        return path

    def read(self, path):
        path = inside(path, self.root)
        if not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise ValueError("Expected a bounded project JSON file")
        return json.loads(path.read_text())

    def ref(self, path):
        path = inside(path, self.root)
        return {"path": str(path.relative_to(self.root)), "sha256": sha256(path)}

    def verify(self, pins):
        if not isinstance(pins, dict):
            raise ValueError("Source pins must be a path-to-SHA256 mapping")
        for path, expected in pins.items():
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError("Invalid source SHA256")
            if self.ref(path)["sha256"] != expected:
                raise ValueError("Source changed: " + str(path))

    def checked(self, reference):
        if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
            raise ValueError("A located SHA256 reference is required")
        self.verify({reference["path"]: reference["sha256"]})
        return self.read(reference["path"])

    def pin_reference(self, reference, pins):
        if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
            raise ValueError("A located SHA256 reference is required")
        path = str(inside(reference["path"], self.root).relative_to(self.root))
        expected = reference["sha256"]
        self.verify({path: expected})
        if path in pins and pins[path] != expected:
            raise ValueError("Conflicting source hashes")
        pins[path] = expected

    def paper_pins(self):
        folder = self.root / "results/manuscript"
        if not folder.is_dir():
            raise ValueError("Publication directory must exist for the freeze guard")
        return self.artifacts(folder)

    def artifacts(self, folder, *, excluded=()):
        pins = {}
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Experiment/publication artifacts cannot be symlinks")
            if path.is_file() and path.name not in excluded and not path.name.startswith("."):
                pins[str(inside(path, self.root, exists=True).relative_to(self.root))] = sha256(path)
        return pins

    def directory_size(self, folder):
        size = 0
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise Paused("Unexpected symlink in experiment")
            try:
                if path.is_file():
                    size += path.stat().st_size
            except FileNotFoundError:
                pass  # An atomic JSON writer may just have renamed its temporary file.
        return size

    def _manifest_cases(self, manifest):
        if manifest.get("version") != MANIFEST_VERSION or not isinstance(manifest.get("cases"), list):
            raise ValueError("Expected a purpose-hundred cohort manifest")
        cases = manifest["cases"]
        if len(cases) != self.expected_compounds:
            raise ValueError(f"Expected exactly {self.expected_compounds} distinct compounds")
        allowed = {"case_id", "compound_id", "identity_smiles", "drug_name", "cohort_role",
                   "evaluation_scope", "input", "evaluation", "evaluation_sources"}
        ids, compounds, identities = set(), set(), set()
        normalized = []
        for original in cases:
            if not isinstance(original, dict) or set(original) - allowed:
                raise ValueError("Unexpected case fields; held-out values cannot be inline case inputs")
            row = deepcopy(original)
            cid, compound, identity = row.get("case_id"), row.get("compound_id"), row.get("identity_smiles")
            if not isinstance(cid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", cid) is None:
                raise ValueError("Invalid project-local case identifier")
            if not isinstance(compound, str) or not compound.strip():
                raise ValueError("Missing compound identity")
            identity = self.normalize_identity(identity)
            if cid in ids or compound in compounds or identity in identities:
                raise ValueError("Expected distinct case, compound and normalized chemical identities")
            ids.add(cid); compounds.add(compound); identities.add(identity)
            row["identity_smiles"] = identity
            if row.get("evaluation_scope") not in SCOPES:
                raise ValueError("Declare aggregated parameter or study-curve evaluation scope")
            if row.get("cohort_role") not in {None, "development_extension", "previously_viewed"}:
                raise ValueError("Unknown cohort role")
            if not isinstance(row.get("evaluation_sources", []), list):
                raise ValueError("evaluation_sources must be a list of references")
            normalized.append(row)
        return normalized

    def prepare(self, manifest_path, run=None, workers=MAX_WORKERS):
        run = self.own_run(run or self.default)
        if run.exists():
            reserved = {"protocol.json", "state.json", "work", "results.json", "results.csv",
                        "pid.json", "run.log", ".run.lock", ".launch.lock", "PAUSE"}
            if not run.is_dir() or any((run / name).exists() for name in reserved):
                raise FileExistsError("Existing batch is immutable; use run/resume/status/report")
            # Cohort curation may already live here. Its files are only read/pinned;
            # preparation never replaces source, input, evaluation or prior data.
            self.directory_size(run)
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
            raise ValueError("Use one to fourteen case workers")
        manifest = self.read(manifest_path)
        cases = self._manifest_cases(manifest)
        source_pins = deepcopy(manifest.get("source_pins", {}))
        self.verify(source_pins)
        self.pin_reference(self.ref(manifest_path), source_pins)
        evaluation_refs = []
        for case in cases:
            evaluation_refs.extend(([case["evaluation"]] if case.get("evaluation") else []) + case.get("evaluation_sources", []))
        for source in evaluation_refs:
            self.pin_reference(source, source_pins)
        evaluation_paths = {str(inside(source["path"], self.root)) for source in evaluation_refs}
        evaluation_hashes = {source["sha256"] for source in evaluation_refs}
        plans = {}
        for case in cases:
            self.pin_reference(case["input"], source_pins)
            # Hash evaluation files without decoding any labels or clinical curves.
            if (str(inside(case["input"]["path"], self.root)) in evaluation_paths
                    or case["input"]["sha256"] in evaluation_hashes):
                raise ValueError("Construction input is a forbidden evaluation source")
            document = self.checked(case["input"])
            if document.get("calibration_mapping"):
                raise ValueError("This hundred-compound batch requires the same zero-curve uncalibrated regime for every compound")
            if document["request"]["compound_id"] != case["compound_id"]:
                raise ValueError("Case and construction input identify different compounds")
            # The evaluation registry is also enforced against all visible input dependencies.
            forbidden = deepcopy(document["request"])
            forbidden["forbidden_sources"] = list(forbidden.get("forbidden_sources", [])) + evaluation_refs
            self.adapter.reject_forbidden({key: value for key, value in document.items() if key != "request"} |
                {"request": {key: value for key, value in document["request"].items() if key != "forbidden_sources"}}, forbidden)
            source_pins.update(self.adapter.checked_input(document))
            plan = plan_design(document["request"], document["fallback_evidence"], document["ranges"],
                               endpoint_limits=document["endpoint_limits"], purpose=document.get("purpose"))
            selected = deepcopy(plan["selected_request"])
            selected["forbidden_sources"] = forbidden["forbidden_sources"]
            calibration = self.adapter.mapping_parameters(document, selected) if plan["scenarios"] else None
            plans[case["case_id"]] = {"input": document, "decision": plan, "calibration": calibration}
        code_pins = self.adapter.engine._code_pins(self.root)
        for name in CODE:
            code_pins[name] = self.ref(name)["sha256"]
        scorer_module = importlib.import_module("pbpk_kb.blueprint_policy.hundred_evaluation") if self.scorer is None else None
        for name in getattr(scorer_module, "CODE_FILES", ()):
            source = self.ref(name)
            code_pins[source["path"]] = source["sha256"]
        paper = self.paper_pins()
        protocol = {"version": VERSION, "manifest": self.ref(manifest_path), "created_at": now(),
            "cases": cases, "case_protocols": plans, "expected_compounds": self.expected_compounds,
            "workers": workers, "native_threads": 2, "native_timeout_seconds": 180,
            "monitor_seconds": MONITOR_SECONDS, "directory_limit_bytes": LIMIT_BYTES,
            "max_consecutive_infrastructure_errors": 3, "source_pins": source_pins,
            "code_pins": code_pins, "paper_guard": paper, "random_seed": RANDOM_SEED,
            "comparators": ["fixed_auto_all_constructible", "matched_coverage_random", "input_only_range_width"],
            "baseline_semantics": "One shared nominal prediction; purpose policy changes support status only",
            "calibration_regime": "zero_curve_uncalibrated_for_all_compounds",
            "input_only_score": "max(upper/lower) of declared CL/fu ranges; missing predicted ranges unassessed",
            "scope": "100 distinct engineering constructions; aggregated scalar and study-curve evaluations reported separately",
            "scenario_retry_policy": "Completed hash-verified scenarios reused; interrupted native scenarios recorded failed, never silently retried",
            "target_observations_used_for_decisions": False, "clinical_validation": False,
            "paper_updated": False}
        self.verify(source_pins); self.verify(code_pins)
        if self.paper_pins() != paper:
            raise ValueError("Publication changed during preparation")
        run.mkdir(parents=True, exist_ok=True)
        save(run / "protocol.json", protocol)
        save(run / "state.json", {"version": VERSION, "status": "prepared", "protocol": self.ref(run / "protocol.json"),
            "outcomes": {}, "progress": {}, "monitor": {}, "consecutive_infrastructure_errors": 0,
            "all_predictions_locked_before_scoring": False, "monitor_stopped_before_scoring": False})
        return {"status": "prepared", "compounds": len(cases),
                "planned_scenarios": sum(len(value["decision"]["scenarios"]) for value in plans.values()),
                "workers": workers, "clinical_curves_claimed": False}

    def _validate_outcome(self, protocol, case, outcome, *, artifacts=True):
        cid = case["case_id"]
        plan = protocol["case_protocols"][cid]["decision"]
        if (outcome.get("status") != "locked" or outcome.get("case_id") != cid
                or outcome.get("compound_id") != case["compound_id"]
                or outcome.get("case_protocol_sha256") != digest(protocol["case_protocols"][cid])):
            raise ValueError("Locked case identity/protocol differs")
        predictions = outcome.get("predictions", [])
        if (len(predictions) != len(plan["scenarios"])
                or {row["scenario_id"] for row in predictions} != {row["scenario_id"] for row in plan["scenarios"]}
                or any(row.get("status") not in {"predicted", "failed"} for row in predictions)):
            raise ValueError("Locked case does not retain every planned scenario")
        if outcome.get("inline_prediction_sha256") != {row["scenario_id"]: digest(row) for row in predictions}:
            raise ValueError("Inline prediction checksum changed")
        if outcome.get("assessment") != assess_design(plan, predictions):
            raise ValueError("Locked decision differs from the complete scenario predictions")
        baseline = next((row for row in predictions if row["scenario_id"] == "baseline"), None)
        expected = {"scenario_id": "baseline", "prediction_sha256": digest(baseline) if baseline else None,
                    "same_nominal_prediction": True}
        if outcome.get("fixedauto_baseline") != expected:
            raise ValueError("Fixed-auto and purpose nominal predictions differ")
        if artifacts:
            self.verify(outcome["artifact_pins"])

    def load(self, run, *, locked=False, artifacts=True):
        run = self.own_run(run)
        state = self.read(run / "state.json")
        protocol = self.checked(state["protocol"])
        if state.get("version") != VERSION or protocol.get("version") != VERSION:
            raise ValueError("Batch version changed")
        if protocol["expected_compounds"] != self.expected_compounds:
            raise ValueError("Batch compound-count contract changed")
        cases = self._manifest_cases({"version": MANIFEST_VERSION, "cases": protocol["cases"]})
        expected = {case["case_id"] for case in cases}
        if set(protocol["case_protocols"]) != expected or not set(state["outcomes"]) <= expected:
            raise ValueError("Batch case matrix changed")
        self.verify(protocol["source_pins"]); self.verify(protocol["code_pins"])
        if self.paper_pins() != protocol["paper_guard"]:
            raise ValueError("Publication files changed")
        for case in cases:
            if case["case_id"] in state["outcomes"]:
                outcome = self.checked(state["outcomes"][case["case_id"]])
                self._validate_outcome(protocol, case, outcome, artifacts=artifacts)
                if outcome["protocol_ref"] != state["protocol"]:
                    raise ValueError("Case outcome refers to a different frozen protocol")
        if locked and (not state.get("all_predictions_locked_before_scoring")
                or not state.get("monitor_stopped_before_scoring") or set(state["outcomes"]) != expected
                or state["status"] not in {"predictions_locked", "complete"}):
            raise ValueError("All cases, including failures, must be locked before report")
        return protocol, state

    def _scenario(self, run, protocol, case, scenario, monitor):
        cid, sid = case["case_id"], scenario["scenario_id"]
        folder = run / "work" / cid / sid
        receipt = folder / "receipt.json"
        case_digest = digest(protocol["case_protocols"][cid])
        if receipt.exists():
            locked = self.read(receipt)
            if locked.get("case_protocol_sha256") != case_digest or locked.get("scenario") != scenario:
                raise ValueError("Cached scenario contract changed")
            self.verify(locked["artifact_pins"])
            prediction = self.checked(locked["prediction"])
            if digest(prediction) != locked["inline_prediction_sha256"]:
                raise ValueError("Cached prediction differs from its inline checksum")
            return prediction
        monitor.check()
        interrupted = folder.exists() and any(folder.iterdir())
        folder.mkdir(parents=True, exist_ok=True)
        if interrupted:
            prediction = {"scenario_id": sid, "status": "failed", "failure_kind": "interrupted_native_scenario",
                          "reason": "Incomplete native artifacts retained; scenario was not automatically retried"}
            prediction_path = folder / "interrupted_prediction.json"
        else:
            save(folder / "started.json", {"started_at": now(), "scenario": scenario, "case_protocol_sha256": case_digest})
            try:
                prediction = self.simulate(protocol["case_protocols"][cid], scenario, folder, monitor.check)
                if prediction.get("scenario_id") != sid or prediction.get("status") not in {"predicted", "failed"}:
                    raise ValueError("Simulator returned a different scenario or invalid status")
                prediction = deepcopy(prediction)
                monitor.record_result(False)
            except Paused:
                raise
            except (ValueError, ArithmeticError) as error:
                prediction = {"scenario_id": sid, "status": "failed", "failure_kind": "model_or_input_failure", "reason": str(error)}
                monitor.record_result(False)
            except Exception as error:
                prediction = {"scenario_id": sid, "status": "failed", "failure_kind": "infrastructure_failure",
                              "reason": f"{type(error).__name__}: {error}"}
                monitor.record_result(True)
            prediction_path = folder / "prediction.json"
        if prediction_path.exists():
            raise ValueError("An interrupted prediction output cannot be overwritten")
        save(prediction_path, prediction)
        save(receipt, {"version": VERSION, "scenario": scenario, "case_protocol_sha256": case_digest,
             "prediction": self.ref(prediction_path), "inline_prediction_sha256": digest(prediction),
             "artifact_pins": self.artifacts(folder), "locked_at": now()})
        with monitor.lock:
            progress = monitor.state["progress"].setdefault(cid, {"completed": 0, "failed": 0})
            progress["completed"] += 1
            progress["failed"] += prediction["status"] == "failed"
            progress["latest"] = sid
            monitor.persist()
        return prediction

    def _case(self, run, protocol, case, monitor):
        cid = case["case_id"]
        folder = run / "work" / cid
        outcome_path, lock_path = folder / "outcome.json", folder / "case.lock.json"
        if lock_path.exists():
            locked = self.read(lock_path)
            outcome = self.checked(locked["outcome"])
            self._validate_outcome(protocol, case, outcome)
            if outcome["protocol_ref"] != monitor.state["protocol"]:
                raise ValueError("Cached case belongs to a different batch")
        else:
            predictions = []
            for scenario in protocol["case_protocols"][cid]["decision"]["scenarios"]:
                monitor.check()
                predictions.append(self._scenario(run, protocol, case, scenario, monitor))
            plan = protocol["case_protocols"][cid]["decision"]
            baseline = next((row for row in predictions if row["scenario_id"] == "baseline"), None)
            construction = ("blocked" if plan["status"] == "blocked" else
                            "predicted" if baseline and baseline["status"] == "predicted" else "failed")
            input_score = (max((row["upper"] / row["lower"] for row in plan["ranges"]), default=1.)
                           if not plan["missing_ranges"] else None)
            outcome = {key: deepcopy(case[key]) for key in ("case_id", "compound_id", "identity_smiles", "drug_name", "cohort_role", "evaluation_scope") if key in case}
            outcome.update(status="locked", construction_status=construction, predictions=predictions,
                assessment=assess_design(plan, predictions), baseline_scenario_id="baseline",
                fixedauto_baseline={"scenario_id": "baseline", "prediction_sha256": digest(baseline) if baseline else None,
                                    "same_nominal_prediction": True},
                input_only_score=input_score, range_parameters=list(plan["ranged_parameters"]),
                failures=[{"scenario_id": row["scenario_id"], "reason": row.get("reason"), "failure_kind": row.get("failure_kind")}
                          for row in predictions if row["status"] == "failed"],
                inline_prediction_sha256={row["scenario_id"]: digest(row) for row in predictions},
                case_protocol_sha256=digest(protocol["case_protocols"][cid]), protocol_ref=deepcopy(monitor.state["protocol"]),
                locked_at=now(), clinical_validation=False)
            folder.mkdir(parents=True, exist_ok=True)
            outcome["artifact_pins"] = self.artifacts(folder, excluded=("outcome.json", "case.lock.json"))
            # An outcome without its case receipt is an interrupted atomic commit;
            # reconstruct it from independently locked scenario receipts.
            save(outcome_path, outcome)
            save(lock_path, {"outcome": self.ref(outcome_path), "protocol_ref": monitor.state["protocol"]})
        with monitor.lock:
            monitor.state["outcomes"][cid] = self.ref(outcome_path)
            monitor.persist()
        return cid

    def execute(self, run=None):
        run = self.own_run(run or self.default)
        with (run / ".run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("This batch already has a running worker") from error
            protocol, state = self.load(run)
            if state["status"] in {"predictions_locked", "complete"}:
                return {"status": state["status"], "reused": True, "cases_locked": len(state["outcomes"])}
            if (run / "PAUSE").exists():
                return {"status": "paused", "reason": "Use resume to acknowledge the pause"}
            monitor = Monitor(self, run, protocol, state)
            monitor.persist(status="running", pause_reason=None, consecutive_infrastructure_errors=0,
                            all_predictions_locked_before_scoring=False, monitor_stopped_before_scoring=False)
            monitor.check(force=True)
            watcher = threading.Thread(target=monitor.watch, name="purpose-hundred-integrity", daemon=True)
            watcher.start()
            pending = iter(case for case in protocol["cases"] if case["case_id"] not in state["outcomes"])
            futures = {}
            try:
                with ThreadPoolExecutor(max_workers=protocol["workers"], thread_name_prefix="purpose-case") as pool:
                    def submit_next():
                        if monitor.halted.is_set() or (run / "PAUSE").exists():
                            return False
                        try:
                            case = next(pending)
                        except StopIteration:
                            return False
                        monitor.check()
                        futures[pool.submit(self._case, run, protocol, case, monitor)] = case["case_id"]
                        return True
                    for _ in range(protocol["workers"]):
                        if not submit_next():
                            break
                    while futures:
                        completed, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                        for future in completed:
                            cid = futures.pop(future)
                            try:
                                future.result()
                            except Paused as error:
                                monitor.halt(error)
                            except Exception as error:
                                monitor.halt(f"Case {cid} checkpoint/integrity failure: {type(error).__name__}: {error}")
                            if not monitor.halted.is_set():
                                submit_next()
                        if not completed:
                            try:
                                monitor.check()
                            except Paused as error:
                                monitor.halt(error)
            except Paused as error:
                monitor.halt(error)
            finally:
                monitor.closed.set()
                watcher.join(timeout=30)
            if len(state["outcomes"]) == len(protocol["cases"]) and not monitor.halted.is_set():
                monitor.check(force=True)
                monitor.persist(status="predictions_locked", all_predictions_locked_before_scoring=True,
                                monitor_stopped_before_scoring=True, predictions_locked_at=now())
            else:
                monitor.persist(status="paused", monitor_stopped_before_scoring=True)
            return {"status": state["status"], "cases_locked": len(state["outcomes"]),
                    "cases_total": len(protocol["cases"]), "pause_reason": state.get("pause_reason")}

    def report(self, run=None):
        run = self.own_run(run or self.default)
        with (run / ".run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Predictions are still running") from error
            protocol, state = self.load(run, locked=True)
            if state["status"] == "complete":
                self.verify(state["report_artifact_pins"])
                return self.checked(state["results"])
            outcomes = [self.checked(state["outcomes"][case["case_id"]]) for case in protocol["cases"]]
            score = self.scorer or importlib.import_module("pbpk_kb.blueprint_policy.hundred_evaluation").score_hundred
            # This is the only call that may decode held-out endpoint/curve data.
            result = score(deepcopy(protocol), deepcopy(outcomes), project_root=self.root)
            if not isinstance(result, dict):
                raise ValueError("The held-out scorer must return a JSON dictionary")
            result.update(runner_version=VERSION, protocol=deepcopy(state["protocol"]),
                          outcomes={case["case_id"]: state["outcomes"][case["case_id"]] for case in protocol["cases"]},
                          all_predictions_locked_before_scoring=True, total_registered_compounds=len(protocol["cases"]),
                          one_hundred_clinical_curves_claimed=False, paper_updated=False)
            self.verify(protocol["source_pins"]); self.verify(protocol["code_pins"])
            if self.paper_pins() != protocol["paper_guard"]:
                raise ValueError("Publication changed during reporting")
            save(run / "results.json", result)
            rows = result.get("records", result.get("rows", []))
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Scorer records/rows must be a list of dictionaries")
            if not rows:
                rows = [{"case_id": row["case_id"], "compound_id": row["compound_id"],
                         "construction_status": row["construction_status"], "assessment_status": row["assessment"]["status"],
                         "evaluation_scope": row["evaluation_scope"]} for row in outcomes]
            fields = sorted(set().union(*(row.keys() for row in rows)))
            with (run / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                                     if isinstance(value, (dict, list)) else value for key, value in row.items()})
            state.update(status="complete", results=self.ref(run / "results.json"),
                         report_artifact_pins={str((run / name).relative_to(self.root)): sha256(run / name)
                                               for name in ("results.json", "results.csv")}, reported_at=now())
            save(run / "state.json", state)
            return result

    def recorded_process(self, run):
        """Inspect only this batch's recorded PID; never enumerate processes."""
        pidfile = run / "pid.json"
        if not pidfile.exists():
            return None
        record = self.read(pidfile)
        expected_script = self.root / "scripts/run_purpose_hundred.py"
        if record.get("script") != str(expected_script) or record.get("run") != str(run):
            raise ValueError("PID receipt belongs to a different script or project batch")
        import psutil
        try:
            process = psutil.Process(record["pid"])
            if process.create_time() != record["create_time"] or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            if process.uids().real != os.getuid():
                raise ValueError("Recorded PID belongs to a different user")
            argv = process.cmdline()
            if (str(expected_script) not in argv or "--run" not in argv
                    or argv[argv.index("--run") + 1] != str(run)
                    or Path(process.cwd()).resolve() != self.root):
                raise ValueError("Recorded PID command no longer identifies this project batch")
            return {"pid": record["pid"], "create_time": record["create_time"], "running": True}
        except psutil.NoSuchProcess:
            return None

    def background(self, run=None):
        run = self.own_run(run or self.default)
        protocol, state = self.load(run)
        if state["status"] == "complete":
            self.verify(state["report_artifact_pins"])
            return {"status": state["status"], "reused": True}
        if (run / "PAUSE").exists():
            return {"status": "paused", "reason": "Use resume to acknowledge the pause"}
        with (run / ".launch.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            live = self.recorded_process(run)
            if live:
                return {"status": "already_running", **live}
            script = self.root / "scripts/run_purpose_hundred.py"
            environment = os.environ.copy()
            cache = self.root / "runtime/cache/purpose_hundred"
            cache.mkdir(parents=True, exist_ok=True)
            environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", TMPDIR=str(cache))
            with (run / "run.log").open("ab") as log:
                child = subprocess.Popen([sys.executable, "-B", str(script), "run", "--run", str(run), "--report-after-run"],
                    cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, env=environment)
            import psutil
            save(run / "pid.json", {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time(),
                                    "script": str(script), "run": str(run), "started_at": now()})
            return {"status": "started", "pid": child.pid, "log": str(run / "run.log"), "workers": protocol["workers"]}

    def pause(self, run=None):
        run = self.own_run(run or self.default)
        self.load(run, artifacts=False)
        save(run / "PAUSE", {"reason": "User requested pause", "requested_at": now()})
        return {"status": "pause_requested", "policy": "Stop new native submissions; retain in-flight outputs"}

    def resume(self, run=None, *, background=False):
        run = self.own_run(run or self.default)
        self.load(run)
        (run / "PAUSE").unlink(missing_ok=True)
        return self.background(run) if background else self.execute(run)

    def status(self, run=None):
        run = self.own_run(run or self.default)
        protocol, state = self.load(run, artifacts=False)
        return {"status": state["status"], "cases_locked": len(state["outcomes"]), "cases_total": len(protocol["cases"]),
                "planned_scenarios": sum(len(row["decision"]["scenarios"]) for row in protocol["case_protocols"].values()),
                "progress": state["progress"], "monitor": state["monitor"], "pause_requested": (run / "PAUSE").exists(),
                "pause_reason": state.get("pause_reason"), "process": self.recorded_process(run),
                "all_predictions_locked_before_scoring": state["all_predictions_locked_before_scoring"],
                "results": state.get("results"), "clinical_curves_claimed": False, "paper_updated": False}

    def main(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=["prepare", "run", "resume", "status", "pause", "report"])
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--run", type=Path, default=self.default)
        parser.add_argument("--workers", type=int, default=MAX_WORKERS)
        parser.add_argument("--background", action="store_true")
        parser.add_argument("--report-after-run", action="store_true",
                            help="Score only after all predictions lock; background workers always enable this")
        args = parser.parse_args(argv)
        if args.command == "prepare":
            if args.manifest is None:
                parser.error("prepare requires --manifest")
            result = self.prepare(args.manifest, args.run, args.workers)
        elif args.command == "run":
            result = self.background(args.run) if args.background else self.execute(args.run)
        elif args.command == "resume":
            result = self.resume(args.run, background=args.background)
        else:
            result = getattr(self, args.command)(args.run)
        if (args.command in {"run", "resume"} and not args.background and args.report_after_run
                and result.get("status") in {"predictions_locked", "complete"}):
            self.report(args.run)
            result = self.status(args.run)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


def create_runner(**kwargs):
    return Runner(**kwargs)


if __name__ == "__main__":
    create_runner().main()
