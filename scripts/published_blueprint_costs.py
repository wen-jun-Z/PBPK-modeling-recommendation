"""Read saved baseline costs for Table 13; never rerun inference or predictions.

All reads go through the caller's hash-verifying ``checked`` callback. The
comparator protocol and state must already be bound by the published protocol.
"""
from __future__ import annotations

import math
import statistics


COHORT_SIZE = 30
FORMULAS = {"PKSimStandard", "RodgersRowland"}
COMPARATOR_DIRECTORY = "experiments/blueprint_design/reviewer_revision_20261009/comparators"
PBPK_SCOPE = (
    "Sum of saved optimizer wall seconds for both formulae and all starts, "
    "including saved pre-repair fits; reused fit artifacts counted once. "
    "Excludes native-physiology preparation, target prediction and outer I/O."
)
TWO_COMPARTMENT_SCOPE = (
    "Saved job wall seconds: 16-start calibration, checkpoint I/O and two "
    "target weight-transfer predictions; not separately timed calibration "
    "or locked-forward prediction."
)


def _ref(value):
    if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
        raise ValueError("Missing hash-bound cost source")
    return {"path": value["path"], "sha256": value["sha256"]}


def _cost(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing or nonnumeric saved computation cost")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("Invalid saved computation cost")
    return int(value) if integer else float(value)


def _summary(rows, scope):
    if len(rows) != COHORT_SIZE or len({row["case_id"] for row in rows}) != COHORT_SIZE:
        raise ValueError("Baseline cost summary requires exactly 30 unique drugs")
    return {
        "rows": rows, "n": len(rows),
        "median_calls": statistics.median(row["calls"] for row in rows),
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "timing_scope": scope,
        "notes": "Historical saved computation; not a matched-hardware latency benchmark. "
                 "Single locked-forward prediction time was not separately saved.",
    }


def load_baseline_costs(protocol, *, checked):
    """Return hash-audited PBPK and two-compartment cost rows for all 30 drugs."""
    cases = protocol["cases"]
    if len(cases) != COHORT_SIZE:
        raise ValueError("Baseline cost extraction requires all 30 drugs")
    training_refs = {}
    for case_id in cases:
        sources = [_ref(job["adapter"]["training_source"])
                   for job in protocol["jobs"].values() if job["case_id"] == case_id]
        if not sources or any(source != sources[0] for source in sources):
            raise ValueError("Published inference methods have different training sources")
        training = checked(sources[0])
        if len(training.get("observations", [])) != 5:
            raise ValueError("Cost comparison requires the same five development points")
        training_refs[case_id] = sources[0]

    pbpk_rows = []
    for case_id, case in cases.items():
        refs, seen_documents, seen_fits = {}, set(), set()
        calls, seconds, pre_repair_documents = 0, 0.0, 0

        def remember(ref):
            normalized = _ref(ref)
            refs[(normalized["path"], normalized["sha256"])] = normalized
            return normalized

        def read_document(ref, *, pre_repair=False):
            nonlocal calls, seconds, pre_repair_documents
            source = remember(ref)
            if source["sha256"] in seen_documents:
                raise ValueError("Duplicate or cyclic pre-repair result source")
            seen_documents.add(source["sha256"])
            document = checked(source)
            if document.get("status") != "predicted":
                raise ValueError("Baseline cost source is not a locked prediction")
            if _ref(document["training"]) != training_refs[case_id]:
                raise ValueError("PBPK cost source training differs from published comparison")
            remember(document["training"])
            if pre_repair:
                pre_repair_documents += 1
            if isinstance(document["fits"], list):
                entries = document["fits"]
                fit_records = []
                for entry in entries:
                    fit_ref = remember(entry["fit"])
                    fit_records.append((entry["formula"], fit_ref["sha256"], checked(fit_ref)))
            else:
                container = remember(document["fits"])
                entries = checked(container)
                fit_records = [(entry["formula"], (container["sha256"], entry["formula"]), entry["fit"])
                               for entry in entries]
            if len(fit_records) != 2 or {record[0] for record in fit_records} != FORMULAS:
                raise ValueError("PBPK cost source must include both unique formula fits")
            for _, identity, fit in fit_records:
                if fit.get("visible_points") != 5 or fit.get("target_observations_used") is not False:
                    raise ValueError("PBPK fit lacks five-point target-isolation evidence")
                if fit.get("training_source") and _ref(fit["training_source"]) != training_refs[case_id]:
                    raise ValueError("Saved formula training differs from published comparison")
                fit_calls = _cost(fit.get("calls"), integer=True)
                fit_seconds = _cost(fit.get("wall_seconds"))
                if identity not in seen_fits:
                    calls += fit_calls
                    seconds += fit_seconds
                    seen_fits.add(identity)
            if document.get("original_source"):
                read_document(document["original_source"], pre_repair=True)

        read_document(case["frozen_pbpk_prediction"])
        pbpk_rows.append({"case_id": case_id, "calls": calls, "seconds": seconds,
                          "timing_scope": PBPK_SCOPE, "source_refs": list(refs.values()),
                          "pre_repair_documents": pre_repair_documents,
                          "unique_formula_fit_artifacts": len(seen_fits)})

    def pinned(name):
        path = f"{COMPARATOR_DIRECTORY}/{name}.json"
        digest = protocol.get("source_pins", {}).get(path)
        if not digest:
            raise ValueError("Comparator cost source is not bound by the published protocol")
        reference = {"path": path, "sha256": digest}
        return reference, checked(reference)

    comparator_protocol_ref, comparator = pinned("protocol")
    comparator_state_ref, state = pinned("state")
    if state.get("status") != "complete" or _ref(state["protocol"]) != comparator_protocol_ref:
        raise ValueError("Comparator cost state is incomplete or refers to another protocol")
    cells = [cell for cell in comparator["cells"]
             if cell["info"] == "time_five" and cell["method"] == "two_compartment"]
    if len(cells) != COHORT_SIZE or {cell["case_id"] for cell in cells} != set(cases):
        raise ValueError("Comparator costs require exactly one five-point cell per drug")
    two_rows = []
    for cell in cells:
        case_id = cell["case_id"]
        if cell.get("visible_points") != 5 or _ref(cell["training_source"]) != training_refs[case_id]:
            raise ValueError("Comparator cell training differs from published comparison")
        source = _ref(state["completed"][cell["job"]])
        document = checked(source)
        if (_ref(document["protocol"]) != comparator_protocol_ref
                or document.get("status") != "predicted"
                or document.get("target_observations_used") is not False
                or document["job"]["case_id"] != case_id
                or document["job"]["model"] != "two_compartment"
                or _ref(document["job"]["training_source"]) != training_refs[case_id]
                or "linear_weight" not in document.get("predictions", {})):
            raise ValueError("Comparator cost source differs from the frozen five-point prediction")
        two_rows.append({"case_id": case_id,
                         "calls": _cost(document["fit"].get("residual_calls"), integer=True),
                         "seconds": _cost(document.get("wall_seconds")),
                         "timing_scope": TWO_COMPARTMENT_SCOPE,
                         "source_refs": [comparator_protocol_ref, comparator_state_ref,
                                         source, training_refs[case_id]]})
    return {"pbpk": _summary(pbpk_rows, PBPK_SCOPE),
            "two_compartment": _summary(two_rows, TWO_COMPARTMENT_SCOPE)}
