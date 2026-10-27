#!/usr/bin/env python3
"""Apply one frozen visible-mass repair policy to every retrospective action.

This is an offline label steward, not a selector. Each counterfactual action may
see its four existing observations plus that action's one acquired observation.
No target concentrations are loaded until every resulting prediction is locked.
Unchanged and already repaired predictions are reused with checked provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import run_sampling_expansion as expansion
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair

BATCH = ROOT / "experiments/blueprint_design/sampling_policy_20261009"
RUN = BATCH / "candidate_repairs"
STABILITY = ROOT / "experiments/blueprint_design/sampling_stability_20261009"
VERSION = "all-actions-visible-mass-repair-1"
read, checked, ref, verify = expansion.read, expansion.checked, expansion.ref, expansion.verify
base, gap, prior, old, data = expansion.base, expansion.gap, expansion.prior, expansion.old, expansion.data


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN):
        raise ValueError("Candidate-repair output boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Guard:
    def __init__(self, protocol):
        self.protocol, self.last = protocol, 0.

    def __call__(self, force=False):
        if (RUN / "PAUSE").exists() or (BATCH / "PAUSE").exists():
            raise RuntimeError("Batch paused; inspect preserved diagnosis before resuming")
        if force or time.monotonic() - self.last >= 1200:
            verify(self.protocol["source_pins"])
            size = sum(p.stat().st_size for p in BATCH.rglob("*") if p.is_file() and not p.is_symlink())
            if size > 10 * 1024 ** 3:
                raise RuntimeError("Sampling-policy batch exceeds 10 GiB")
            self.last = time.monotonic()


def reusable_repair(unit, original_source, training_reference, audits, fits):
    """Reject superficially similar caches with different observations or rules."""
    if (unit.get("original_source") != original_source
            or unit.get("training") != training_reference
            or unit.get("feasibility") != audits
            or unit.get("status") != "predicted"):
        return False
    if {row["formula"] for row in fits} != set(base.FORMULAS):
        return False
    for row in fits:
        fit = row["fit"]
        if (len(fit["starts"]) != 16 or fit["call_limit"] != 16384
                or fit["target_observations_used"] is not False
                or fit["no_warm_start"] is not True):
            return False
        if audits[row["formula"]]["repair_required"]:
            if (fit.get("version") != repair.VERSION
                    or fit.get("partition_feasibility") != audits[row["formula"]]):
                return False
    return True


def prepare():
    if (RUN / "protocol.json").exists():
        return status()
    current = read(expansion.RUN / "state.json")
    original = read(expansion.OLD / "state.json")
    stable = read(STABILITY / "state.json")
    if any(s["status"] != "complete" for s in (current, original, stable)):
        raise ValueError("Frozen source batches must be complete")
    protocol = checked(current["protocol"])
    verify(protocol["source_pins"])
    original_ids = set(protocol["original_case_ids"])
    pins = deepcopy(protocol["source_pins"])
    for path in (Path(__file__), ROOT / "src/pbpk_kb/blueprint_policy/visible_partition_feasibility.py",
                 expansion.RUN / "state.json", expansion.OLD / "state.json", STABILITY / "state.json"):
        pins[str(path.relative_to(ROOT))] = base.sha256(path)
    # Inspect only prediction/construction receipts here, never scored outputs.
    repair_cache = {}
    for entry in stable["units"].values():
        if entry["reused"]:
            continue
        unit = checked(entry["result"])
        verify(unit["output_pins"])
        repair_cache[unit["original_source"]["path"]] = (entry["result"], unit)
    jobs, menus = {}, {}
    for case in protocol["cases"]:
        cid = case["case_id"]
        owner = original if cid in original_ids else current
        development = {}
        for formula in base.FORMULAS:
            request = gap.request(case, "development", formula)
            development[formula] = (request, prior.get_phys(protocol, case, "development", formula, request))
        for regime in gap.REGIMES:
            design_key = cid + "__" + regime
            decision_ref = owner["designs"][design_key]["result"]
            decision = checked(decision_ref)
            pins[decision_ref["path"]] = decision_ref["sha256"]
            candidates = [("baseline", decision["baseline"], None)]
            candidates += [(design_key + f"__point_{i:02d}", owner["units"][design_key + f"__point_{i:02d}"]["result"], t)
                           for i, t in zip(decision["candidate_ids"], decision["candidate_times_h"])]
            menus[design_key] = dict(case_id=cid, regime=regime, decision_source=decision_ref,
                candidate_ids=decision["candidate_ids"], candidate_times_h=decision["candidate_times_h"])
            for candidate, source, acquired_time in candidates:
                key = design_key + "__baseline" if candidate == "baseline" else candidate
                unit = checked(source)
                if unit["status"] != "predicted":
                    raise ValueError("Frozen failed predictions require explicit failure handling: " + key)
                training = checked(unit["training"])
                expected = 4 if candidate == "baseline" else 5
                if len(training["observations"]) != expected:
                    raise ValueError("Incorrect counterfactual observation count")
                audits = {formula: repair.feasibility(phys, request, training)
                          for formula, (request, phys) in development.items()}
                trigger = any(a["repair_required"] for a in audits.values())
                job = dict(case_id=cid, drug=case["drug"], cluster_id=case["cluster_id"],
                    cohort=case["expansion_cohort"], design_key=design_key, regime=regime,
                    candidate=candidate, acquired_time_h=acquired_time, source=source,
                    training=unit["training"], expected_visible_points=expected,
                    feasibility=audits, repair_required=trigger, disposition="new_repair" if trigger else "unchanged",
                    output=None if trigger else source)
                base.pins_in(unit, pins)
                pins[source["path"]] = source["sha256"]
                if trigger and source["path"] in repair_cache:
                    replacement, repaired = repair_cache[source["path"]]
                    fits = checked(repaired["fits"])
                    if not reusable_repair(repaired, source, unit["training"], audits, fits):
                        raise ValueError("Existing repair has incompatible provenance: " + key)
                    base.pins_in(repaired, pins)
                    pins[replacement["path"]] = replacement["sha256"]
                    job.update(disposition="existing_repair", output=replacement)
                jobs[key] = job
    if len(menus) != 60 or len({j["case_id"] for j in jobs.values()}) != 30:
        raise ValueError("Require every menu from all thirty drugs")
    frozen = dict(version=VERSION, created_at=time.time(), source_protocol=current["protocol"],
        source_state=ref(expansion.RUN / "state.json"), source_pins=pins, jobs=jobs, menus=menus,
        fit=dict(formulas=list(base.FORMULAS), starts=16, calls_per_formula=16384,
            equations="frozen four-axis blood_partition", repair_version=repair.VERSION, no_warm_start=True),
        isolation="Offline counterfactual actions see only their own 4+1 observations; selectors must use separate four-point feature files. All target predictions lock before score_case reads target observations.",
        limits=dict(workers=6, blas_threads=1, monitor_seconds=1200, batch_gib=10),
        limitations=["All thirty clinical cases have been viewed previously; retrospective development labels, not new blinded outcomes.",
            "Complete repaired action labels do not make those labels available to held-out selector training.",
            "Existing successful fits are reused only when observations, rule version, budgets and source hashes agree."])
    verify(pins)
    save(RUN / "protocol.json", frozen)
    completed = {key: dict(result=j["output"], disposition=j["disposition"], new_calls=0)
                 for key, j in jobs.items() if j["output"] is not None}
    save(RUN / "state.json", dict(status="prepared", protocol=ref(RUN / "protocol.json"), units=completed))
    return status()


def fit_one(key, job):
    protocol = read(RUN / "protocol.json")
    guard = Guard(protocol)
    guard(force=True)
    folder = RUN / "fits" / key
    path = folder / "result.json"
    if path.exists():
        unit = read(path)
        if unit["protocol"] != ref(RUN / "protocol.json") or unit["original_source"] != job["source"]:
            raise ValueError("Changed resumed repair identity")
        verify(unit["output_pins"])
        return dict(result=ref(path), disposition="new_repair", new_calls=unit["resources"]["calls"])
    start = time.monotonic()
    source_protocol = checked(protocol["source_protocol"])
    case = gap.get_case(source_protocol, job["case_id"])
    original = checked(job["source"])
    training = checked(job["training"])
    fits, calls, failures = [], 0, 0
    for formula in base.FORMULAS:
        request = gap.request(case, "development", formula)
        phys = prior.get_phys(source_protocol, case, "development", formula, request)
        audit = repair.feasibility(phys, request, training)
        if audit != job["feasibility"][formula]:
            raise ValueError("Changed visible-data feasibility")
        fit_path = folder / (formula + ".json")
        if fit_path.exists():
            fit = read(fit_path)
        elif audit["repair_required"]:
            fit = repair.fit(lambda x, t: old.simulate("blood_partition", phys, request, t, x),
                request, training, phys, budget=16384, starts=16, before_trial=guard)
            save(fit_path, fit)
        else:
            fit = checked(next(f["fit"] for f in original["fits"] if f["formula"] == formula))
            save(fit_path, fit)
        if audit["repair_required"]:
            if fit.get("partition_feasibility") != audit:
                raise ValueError("Wrong resumed fit repair rule")
            calls += fit["calls"]
            failures += fit["failures"]
        fits.append(dict(formula=formula, fit=ref(fit_path), best=fit["best"]))
    result = dict(version=VERSION, protocol=ref(RUN / "protocol.json"), original_source=job["source"],
        training=job["training"], status="failed", fits=fits, feasibility=job["feasibility"],
        target_observations_used=False, resources=dict(calls=calls, failures=failures, cpu_threads=1), output_pins={})
    eligible = [f for f in fits if f["best"] is not None]
    if eligible:
        chosen = min(eligible, key=lambda row: row["best"]["objective"])
        formula, best = chosen["formula"], chosen["best"]
        request = gap.request(case, "target", formula)
        phys = prior.get_phys(source_protocol, case, "target", formula, request)
        grid = base.engine.prediction_grid(request)
        save(folder / "selection.json", dict(formula=formula, parameters=best, locked_at=time.time(), target_values_used=False))
        predicted = old.simulate("blood_partition", phys, request, grid, best["log_parameters"])
        save(folder / "prediction.json", prior.curve_document(predicted, grid, request))
        save(folder / "request.json", request)
        result.update(status="predicted", formula=formula, parameters=best,
            prediction=ref(folder / "prediction.json"), request=ref(folder / "request.json"),
            selection=ref(folder / "selection.json"), mass_balance=predicted["mass_balance"])
    else:
        result["reason"] = "No feasible repaired formula; all start diagnostics retained"
    result["resources"]["wall_seconds"] = time.monotonic() - start
    base.pins_in({k: v for k, v in result.items() if k != "output_pins"}, result["output_pins"])
    guard(force=True)
    save(path, result)
    return dict(result=ref(path), disposition="new_repair", new_calls=calls)


def run(workers):
    if not 1 <= workers <= 6:
        raise ValueError("Use one to six single-thread workers")
    state = read(RUN / "state.json")
    protocol = checked(state["protocol"])
    Guard(protocol)(force=True)
    if state["status"] == "complete":
        checked(state["records"])
        return status()
    state.update(status="fitting", pid=os.getpid(), started_at=state.get("started_at", time.time()))
    save(RUN / "state.json", state)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_one, key, job): key for key, job in protocol["jobs"].items()
                       if key not in state["units"]}
            for future in as_completed(futures):
                key = futures[future]
                state["units"][key] = future.result()
                save(RUN / "state.json", state)
                print(json.dumps(dict(done=len(state["units"]), total=len(protocol["jobs"]), key=key)), flush=True)
        lock = dict(protocol=state["protocol"], predictions=state["units"], locked_at=time.time())
        save(RUN / "predictions_locked.json", lock)
        state.update(status="predictions_locked", prediction_lock=ref(RUN / "predictions_locked.json"))
        save(RUN / "state.json", state)
        return report()
    except BaseException as exc:
        state.update(status="paused", reason=str(exc))
        save(RUN / "state.json", state)
        save(RUN / "PAUSE", dict(reason=str(exc), at=time.time()))
        raise


def report():
    state = read(RUN / "state.json")
    protocol = checked(state["protocol"])
    Guard(protocol)(force=True)
    if state["status"] == "complete":
        checked(state["records"])
        return status()
    if state["status"] != "predictions_locked" or set(state["units"]) != set(protocol["jobs"]):
        raise ValueError("All counterfactual predictions must lock before clinical scoring")
    lock = checked(state["prediction_lock"])
    if lock["predictions"] != state["units"] or lock["protocol"] != state["protocol"]:
        raise ValueError("Prediction lock differs from completed units")
    source_protocol = checked(protocol["source_protocol"])
    records, failures = [], []
    for key, job in protocol["jobs"].items():
        entry = state["units"][key]
        unit = checked(entry["result"])
        case = gap.get_case(source_protocol, job["case_id"])
        row = {k: job[k] for k in ("case_id", "drug", "design_key", "regime", "candidate", "cohort")}
        row.update(status=unit["status"], source=entry["result"], original_source=job["source"],
            disposition=entry["disposition"], repair_required=job["repair_required"], cluster_id=job["cluster_id"])
        if unit["status"] == "predicted":
            row.update(data.score_case(case, unit))
        else:
            row.update(reason=unit.get("reason", "Prediction failed"),
                endpoints={ep: dict(value=None, reference=None, error=None, unit=u, reason=unit.get("reason", "Prediction failed"), reference_kind="unavailable")
                           for ep, u in data.UNITS.items()}, curve_metrics=dict(gmfe=None, n=0))
            failures.append(dict(key=key, source=entry["result"], reason=row["reason"]))
        records.append(row)
    counts = {kind: sum(j["disposition"] == kind for j in protocol["jobs"].values())
              for kind in ("unchanged", "existing_repair", "new_repair")}
    document = dict(version=VERSION, protocol=state["protocol"], prediction_lock=state["prediction_lock"],
        records=records, counts=dict(**counts, drugs=30, menus=60, records=len(records),
            failures=len(failures), new_calls=sum(u["new_calls"] for u in state["units"].values())),
        failures=failures, scored_at=time.time(), schema="expansion-compatible records keyed by (design_key,candidate)",
        scope="All counterfactual baseline/acquisition labels. Training consumers must apply whole-study fold allowlists before numerical label access.")
    save(RUN / "records.json", document)
    state.update(status="complete", records=ref(RUN / "records.json"), completed_at=time.time())
    save(RUN / "state.json", state)
    return status()


def status():
    protocol, state = read(RUN / "protocol.json"), read(RUN / "state.json")
    return dict(status=state["status"], drugs=len({j["case_id"] for j in protocol["jobs"].values()}),
        menus=len(protocol["menus"]), completed=len(state["units"]), total=len(protocol["jobs"]),
        dispositions={kind: sum(j["disposition"] == kind for j in protocol["jobs"].values())
                      for kind in ("unchanged", "existing_repair", "new_repair")},
        records=state.get("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "status":
        value = status()
    else:
        RUN.mkdir(parents=True, exist_ok=True)
        with (RUN / "runner.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = prepare() if args.command == "prepare" else run(args.workers) if args.command == "run" else report()
    print(json.dumps(value, ensure_ascii=False))
