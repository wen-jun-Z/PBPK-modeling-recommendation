"""Clinical runner isolation and statistical contracts; no training/native runs."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import threading

import numpy as np
import pytest

from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, Source
from tests.test_blueprints.test_evidence_decision import request as synthetic_request

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_clinical_pbpk_comparison.py"
_SPEC = importlib.util.spec_from_file_location("clinical_pbpk_runner_test_module", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_EVALUATION_SPEC = importlib.util.spec_from_file_location("clinical_reevaluation_test_module",
    _SCRIPT.with_name("evaluate_existing_clinical_pbpk.py"))
evaluation = importlib.util.module_from_spec(_EVALUATION_SPEC)
_EVALUATION_SPEC.loader.exec_module(evaluation)


def test_reevaluation_thresholds_are_strict_and_denominators_keep_failures():
    result = evaluation.fold_statistics([1., 2., 3., 5.], attempted=6, reference_available=5)
    assert result["threshold_comparison"] == "<"
    assert [result["within_fold"][str(t)]["count"] for t in (2, 3, 5)] == [1, 2, 3]
    assert result["missing_reference"] == result["reference_available_but_unscored"] == 1
    assert result["within_fold"]["2"] == {"count": 1, "scored_denominator": 4,
        "reference_denominator": 5, "attempted_denominator": 6,
        "fraction_of_scored": .25, "fraction_of_attempted": 1/6}
    assert result["gmfe"] == pytest.approx(30 ** .25)
    with pytest.raises(ValueError, match="Invalid fold-error"):
        evaluation.fold_statistics([.9], attempted=1, reference_available=1)


def test_reevaluation_drug_equal_and_common_endpoint_families(monkeypatch):
    monkeypatch.setattr(evaluation, "RESAMPLES", 20)
    cases = cases_twenty()[:3]
    cases[2]["cohort"] = "extension"
    rows = []
    for i, case in enumerate(cases):
        for method in evaluation.METHODS:
            failed = method == "ours" and i == 1
            family = "observation_derived" if i == 2 else "original_reported"
            rows.append({"case_id": case["case_id"], "method": method,
                "reference_family": family, "attempt_status": "failed" if failed else "predicted",
                "curve_metrics": {} if failed else {"gmfe": [1., 4., 9.][i], "n": [1, 10, 100][i]},
                "endpoints": {name: {"reference": 1., "value": None if failed else 2.,
                    "error": None if failed else (0.5 if name == "tmax" else 2.), "weak_tail": False}
                    for name in evaluation.UNITS}})
    summaries, common, pairs = evaluation.summarise(rows, cases)
    ours = next(r for r in summaries if r["dataset"] == "all_20" and r["method"] == "ours")
    assert ours["curve"]["gmfe"] == pytest.approx(3.)
    assert ours["curve"]["gmfe"] != pytest.approx(math.exp(100 * math.log(9.) / 101))
    assert ours["prediction_failed"] == 1
    assert ours["endpoints"]["original_reported"]["auc"]["reference_available"] == 2
    assert ours["endpoints"]["original_reported"]["auc"]["scored"] == 1
    assert ours["endpoints"]["observation_derived"]["auc"]["scored"] == 1
    assert ours["endpoints"]["original_reported"]["tmax"]["mae_h"] == .5
    assert ours["endpoints"]["original_reported"]["tmax"]["gmfe"] is None
    for row in common:
        if row["dataset"] == "all_20":
            assert row["curve"]["gmfe"] == pytest.approx(3.)
            assert row["endpoints"]["original_reported"]["auc"]["shared_scored_case_ids"] == ["case_00"]
    assert pairs["all_20"]["curve"]["rf_standard"]["complete_pairs"] == 2


def test_reevaluation_failed_attempt_retains_truth_and_all_eleven_rows(monkeypatch, tmp_path):
    case = cases_twenty()[0]
    case.update(reference_kind="original_reported", reference_source={"path": "clinical.json", "sha256": "f"*64},
                nca_policy={"sampling_time_h": [0., 1.]})
    truth = {name: {"value": None if name == "vss" else 1., "unit": unit}
             for name, unit in evaluation.UNITS.items()}
    monkeypatch.setattr(evaluation.legacy, "_case_reference", lambda *args: ({}, {}, truth))
    store = type("FrozenStore", (), {"unchanged": lambda self: None})()
    bundle = {"entries": {method: {"status": "failed", "failure": "structural_rejection"}
                           for method in evaluation.METHODS}}
    rows = evaluation.recompute([(case, {"path": "prediction.json", "sha256": "a"*64}, bundle)], store)
    assert all(len(row["endpoints"]) == 10 for row in rows)
    assert all(row["endpoints"]["auc"]["reference"] == 1. for row in rows)
    assert all(row["endpoints"]["auc"]["value"] is None for row in rows)
    assert all(row["endpoints"]["vss"]["reference"] is None for row in rows)
    output = tmp_path / "evaluation.csv"
    evaluation.write_csv(output, rows)
    import csv
    records = list(csv.DictReader(output.open()))
    assert len(records) == 44
    assert all(record["prediction_reused"] == "True" for record in records)
    assert [r for r in records if r["endpoint"] == "auc"][0]["observation"] == "1.0"
    assert [r for r in records if r["endpoint"] == "vss"][0]["observation"] == ""


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    run = root / "experiments/blueprint_design/clinical_pbpk_comparison_synthetic"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "DEFAULT", run)
    monkeypatch.setattr(runner, "CODE", ())
    monkeypatch.setattr(runner, "paper_pins", lambda: {})
    return root, run


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return runner.ref(path)


def test_cleanup_validates_csv_without_decoding_as_json(project):
    root, run = project
    results = put(run / 'results.json', {'units': []})
    table = run / 'results.csv'
    table.write_text('drug,prediction\nExample,1.2\n')
    put(run / 'state.json', {'status': 'complete', 'results': results,
                            'result_table': runner.ref(table)})
    (run / 'unused.tmp').write_text('temporary')
    assert runner.cleanup(run)['removed_temporary_files'] == 1
    assert table.exists()
    table.write_text('changed\n')
    with pytest.raises(ValueError, match='result table changed'):
        runner.cleanup(run)


def test_post_completion_patch_is_explicit_and_cannot_override_data(project):
    root, run = project
    code = root / 'runner.py'; code.write_text('original')
    original = runner.sha256(code)
    protocol = {'source_pins': {}, 'paper_guard': {}, 'code_pins': {'runner.py': original}}
    code.write_text('cleanup correction')
    patch = {'runner.py': {'old_sha256': original, 'new_sha256': runner.sha256(code),
                           'reason': 'CSV cleanup correction'}}
    with pytest.raises(ValueError, match='Frozen source changed'):
        runner.verify(protocol)
    runner.verify(protocol, patch)
    protocol = {'source_pins': {'runner.py': original}, 'paper_guard': {}, 'code_pins': {}}
    with pytest.raises(ValueError, match='Frozen source changed'):
        runner.verify(protocol, patch)


def cases_twenty():
    return [{"case_id": f"case_{i:02d}", "compound_id": f"synthetic:{i}", "drug": f"Drug{i}",
             "study_id": "shared_study" if i < 2 else f"study{i}",
             "cluster_id": "shared_study" if i < 2 else f"study{i}",
             "cohort": "original" if i < 10 else "extension",
             "dataset": "reported_clinical_10" if i < 10 else "osp_observed_10",
             "reference_kind": "observed_curve"} for i in range(20)]


def request_case(root):
    request = synthetic_request()
    source = put(root / "physchem.json", {"input": "synthetic physicochemical data only"})
    clinical = put(root / "clinical.json", {"must_not_be_used_as_construction_input": True})
    for record in request["evidence"]:
        record["source"] = {**source, "location": record["parameter"]}
    for parameter in ("total_clearance", "fraction_unbound"):
        record = next(e for e in request["evidence"] if e["parameter"] == parameter)
        record["source"] = {**clinical, "location": parameter}
    renal = deepcopy(next(e for e in request["evidence"] if e["parameter"] == "total_clearance"))
    renal.update(parameter="renal_clearance", value=1.)
    request["evidence"].append(renal)
    ledger = put(root / "fixture_evidence.json", {"evidence": request["evidence"]})
    request["evidence_ledger"] = {**ledger, "location": "synthetic evidence"}
    return {"compound_id": request["compound_id"], "request_template": request,
            "forbidden_clinical_sources": [{**clinical, "json_pointer": "/observations"}]}


def test_build_request_replaces_cl_renal_fu_and_validates_strict_sources(project):
    root, run = project
    case = request_case(root)
    before = deepcopy(case)
    source = put(root / "predicted_parameters.json", {"training_target_overlap": False})
    parameters = {"total_clearance": 2.5, "fraction_unbound": .4}
    request = runner.build_request(case, parameters, source, run / "inputs")
    assert case == before
    names = [e["parameter"] for e in request["evidence"]]
    assert names.count("total_clearance") == names.count("fraction_unbound") == 1
    assert "renal_clearance" not in names and "hepatic_clearance" not in names
    scope = next(e for e in request["evidence"] if e["parameter"] == "elimination_scope")
    assert scope["value"] == "hepatic_only" and scope["basis"] == "assumption"
    assert scope["source"]["path"] != "clinical.json"
    for record in request["evidence"]:
        Source.model_validate(record["source"])
        if record["parameter"] in parameters:
            assert record["value"] == parameters[record["parameter"]]
            assert record["basis"] == "predicted" and record["prediction_method"]["training_overlap"] is False
    for reference in request["forbidden_sources"]:
        Source.model_validate(reference)
        assert "json_pointer" not in reference
    EvidenceBlueprintRequest.model_validate(request)
    ledger = runner.checked(request["evidence_ledger"])
    assert ledger["evidence"] == request["evidence"]


@pytest.mark.parametrize("bad", [{"total_clearance": 0., "fraction_unbound": .2},
                                  {"total_clearance": 2., "fraction_unbound": 1.1},
                                  {"total_clearance": float("nan"), "fraction_unbound": .2}])
def test_invalid_parameter_prediction_never_writes_request(project, bad):
    root, run = project
    case = request_case(root)
    source = put(root / "prior.json", {})
    with pytest.raises(ValueError, match="Invalid predicted"):
        runner.build_request(case, bad, source, run / "inputs")
    assert not (run / "inputs/request.json").exists()


def test_prepare_covers_same_twenty_cases_four_methods_without_decoding_outcomes(project, monkeypatch):
    root, run = project
    original_cases = []
    forbidden_paths = set()
    for base in cases_twenty():
        cid = base["compound_id"]
        case = request_case(root)
        req = case["request_template"]
        req["compound_id"] = cid
        for e in req["evidence"]:
            e["compound_id"] = cid
        request_ref = put(root / "sources" / (base["case_id"] + "_request.json"), {"target": req})
        chemical = root / "sources" / (base["case_id"] + "_physchem.json")
        put(chemical, {"identity": {"isomeric_smiles": "CCO"}})
        clinical = root / "sources" / (base["case_id"] + "_observations.json")
        clinical.write_text("not decoded by prepare; synthetic hidden outcomes")
        clinical_ref = runner.ref(clinical)
        forbidden_paths.add(clinical)
        original_cases.append({**base, "cross_drug_requests": request_ref,
            "reference_source": clinical_ref, "full_training_source": clinical_ref,
            "five_training_source": clinical_ref, "physchem": str(chemical.relative_to(root)),
            "nca_policy": {"sampling_time_h": [0., 1., 2.]}})
    source = root / "source_protocol.json"
    put(source, {"cases": original_cases})
    monkeypatch.setattr(runner, "SOURCE", source)
    read = runner.read
    def guarded(path):
        assert Path(path) not in forbidden_paths, "prepare decoded clinical outcomes"
        return read(path)
    monkeypatch.setattr(runner, "read", guarded)
    result = runner.prepare(run, workers=14)
    protocol = read(run / "protocol.json")
    assert result["drugs"] == 20 and result["comparisons"] == 80
    assert protocol["methods"] == list(runner.METHODS)
    assert len(protocol["cases"]) == len({c["case_id"] for c in protocol["cases"]}) == 20
    assert sum(c["cohort"] == "original" for c in protocol["cases"]) == 10
    for case in protocol["cases"]:
        assert not {"total_clearance", "hepatic_clearance", "renal_clearance", "fraction_unbound"} & {
            e["parameter"] for e in case["request_template"]["evidence"]}
    assert not read(run / "state.json")["all_predictions_locked_before_scoring"]


def locked_fixture(root, run):
    cases = cases_twenty()
    protocol = {"cases": cases, "source_pins": {}, "code_pins": {}, "paper_guard": {},
                "methods": list(runner.METHODS)}
    protocol_ref = put(run / "protocol.json", protocol)
    outcomes = {}
    for case in cases:
        bundle = {"case_id": case["case_id"], "compound_id": case["compound_id"],
                  "protocol": protocol_ref, "locked_at": "synthetic-lock",
                  "entries": {m: {"status": "predicted", "method": m, "request": {"compound_id": case["compound_id"]},
                                  "curve": {}, "endpoints": {}, "target_observations_used": False}
                              for m in runner.METHODS}}
        outcomes[case["case_id"]] = put(run / "models" / case["case_id"] / "predictions.json", bundle)
    state = {"protocol": protocol_ref, "outcomes": outcomes,
             "all_predictions_locked_before_scoring": True, "status": "predictions_locked"}
    put(run / "state.json", state)
    return protocol, state


@pytest.mark.parametrize("problem", ["unlocked", "missing_case", "missing_method", "unfinished_method",
                                      "wrong_case", "wrong_drug", "wrong_method"])
def test_report_rejects_incomplete_matrix_before_opening_clinical_sources(project, monkeypatch, problem):
    root, run = project
    _, state = locked_fixture(root, run)
    if problem == "unlocked":
        state["all_predictions_locked_before_scoring"] = False
    elif problem == "missing_case":
        state["outcomes"].pop("case_19")
    else:
        reference = state["outcomes"]["case_19"]
        bundle = runner.checked(reference)
        if problem == "missing_method":
            bundle["entries"].pop("ours")
        elif problem == "unfinished_method":
            bundle["entries"]["ours"]["status"] = "running"
        elif problem == "wrong_case":
            bundle["case_id"] = "case_00"
        elif problem == "wrong_drug":
            bundle["entries"]["ours"]["request"]["compound_id"] = "synthetic:0"
        elif problem == "wrong_method":
            bundle["entries"]["ours"]["method"] = "rf_standard"
        state["outcomes"]["case_19"] = put(root / reference["path"], bundle)
    put(run / "state.json", state)
    import report_blueprint_validation as legacy
    import score_cross_drug_blueprint as scoring
    monkeypatch.setattr(scoring, "_native_fields", lambda entry: None)
    monkeypatch.setattr(legacy, "Store", lambda *a: pytest.fail("clinical source access before full lock"))
    with pytest.raises(ValueError):
        runner.report(run)


def test_report_validates_all_eighty_predictions_before_first_clinical_read(project, monkeypatch):
    root, run = project
    protocol, _ = locked_fixture(root, run)
    import report_blueprint_validation as legacy
    import score_cross_drug_blueprint as scoring
    validated, scored = [], []
    monkeypatch.setattr(scoring, "_native_fields", lambda entry: validated.append(entry["method"]))
    class Store:
        def __init__(self, root):
            assert len(validated) == 80
        def unchanged(self):
            assert len(scored) == 80
    monkeypatch.setattr(legacy, "Store", Store)
    def score(case, entry, store):
        assert len(validated) == 80
        scored.append((case["case_id"], entry["method"]))
        return {"curve_metrics": {"gmfe": 2., "ln_rmse": .7, "n": 3},
                "endpoints": {}, "scoring_status": "scored"}
    monkeypatch.setattr(scoring, "score_curve_reference", score)
    result = runner.report(run)
    assert result["status"] == "complete"
    assert set(scored) == {(c["case_id"], m) for c in protocol["cases"] for m in runner.METHODS}
    assert len(runner.read(run / "results.json")["units"]) == 80


@pytest.mark.parametrize("recovery", ["fresh", "completed_uncommitted", "interrupted_without_receipt"])
def test_run_case_shared_inputs_formula_coverage_and_canonical_native_reuse(project, monkeypatch, recovery):
    root, run = project
    case = {**request_case(root), "case_id": "case_00"}
    source = put(root / "prior.json", {"role": "synthetic independently trained model"})
    rich = {"total_clearance": 2., "fraction_unbound": .2, "vss": 1.2, "terminal_half_life": 4.}
    rf = {"total_clearance": 3., "fraction_unbound": .3, "vss": 1., "terminal_half_life": 5.}
    prior = {"parameters": {case["compound_id"]: rich}, "rf_parameters": {case["compound_id"]: rf}}
    import pbpk_kb.blueprint_policy.clinical_pbpk_comparison as methods
    import pbpk_kb.blueprints.extended_prediction as summaries
    curve = {"time_min": [0., 60.], "concentration": [0., 1.], "time_unit": "min", "unit": "umol/l"}
    endpoint = {"observations_used": False, "endpoints": {}}
    calls = []
    def native(request, folder, project_root, formula):
        if recovery == "completed_uncommitted" and folder.name == "native_PKSimStandard":
            pytest.fail("A complete checksum-valid native receipt must not be recomputed")
        if recovery == "interrupted_without_receipt" and folder.name == "native_PKSimStandard":
            assert not folder.exists(), "Interrupted original directory must be preserved before a new solve"
        calls.append((deepcopy(request), formula))
        data = {"version": methods.VERSION, "request_sha256": methods._digest(request),
                "status": "predicted", "curve": deepcopy(curve), "endpoints": deepcopy(endpoint)}
        reference = put(folder / "result.json", data)
        return {**data, "result_ref": reference}
    def ours(request, volume, half, folder, project_root, *, native_result):
        persisted = runner.checked(native_result["result_ref"])
        assert persisted == {k: v for k, v in native_result.items() if k != "result_ref"}
        assert "elapsed_seconds" not in native_result
        assert volume[0]["value"] == rich["vss"] and half[0]["value"] == rich["terminal_half_life"]
        return {"status": "predicted", "curve": deepcopy(curve), "endpoints": deepcopy(endpoint)}
    monkeypatch.setattr(methods, "run_native_partition", native)
    monkeypatch.setattr(methods, "run_ours", ours)
    monkeypatch.setattr(summaries, "summarize_extended_prediction", lambda *a: deepcopy(endpoint))
    class Monitor:
        def __init__(self):
            self.lock, self.halted = threading.RLock(), threading.Event()
            self.state = {"protocol": {"path": "protocol.json", "sha256": "a" * 64}, "outcomes": {}, "progress": {}}
        def check(self, *a, **k):
            pass
        def update(self, **values):
            self.state.update(values)
    monitor = Monitor()
    working = run / "models/case_00/native_PKSimStandard"
    completed_hash = None
    if recovery == "completed_uncommitted":
        # The child finished and saved its result; outer trial/state commit did
        # not finish. Its input hash and subordinate execution receipt are real.
        req = runner.build_request(case, rich, source, working.parent / "rich_inputs")
        execution_ref = put(working / "execution.json", {"status": "synthetic_native_completed"})
        completed = {"version": methods.VERSION, "request_sha256": methods._digest(req),
            "status": "predicted", "curve": deepcopy(curve), "endpoints": deepcopy(endpoint),
            "execution_ref": execution_ref}
        completed_hash = put(working / "result.json", completed)["sha256"]
        # Also retain an outer file not yet recorded in monitor.state.
        put(working.with_suffix(".json"), {"uncommitted_outer_file": True})
        assert monitor.state["progress"] == {} and monitor.state["outcomes"] == {}
    elif recovery == "interrupted_without_receipt":
        working.mkdir(parents=True)
        (working / "partial.pkml").write_text("synthetic interrupted native bytes")
        previous = working.with_name(working.name + ".interrupted_1")
        previous.mkdir()
        (previous / "older.txt").write_text("older diagnosis must survive")
    assert runner.run_case(case, prior, source, run, monitor) == "case_00"
    bundle = runner.checked(monitor.state["outcomes"]["case_00"])
    assert set(bundle["entries"]) == set(runner.METHODS)
    assert all(e["status"] == "predicted" for e in bundle["entries"].values()), bundle
    expected_formulas = list(runner.FORMULAS[1:] if recovery == "completed_uncommitted" else runner.FORMULAS)
    assert [formula for _, formula in calls[:-1]] == expected_formulas
    assert len(calls) == (5 if recovery == "completed_uncommitted" else 6)
    assert all(req == calls[0][0] for req, _ in calls[:-1])
    if recovery == "completed_uncommitted":
        assert runner.sha256(working / "result.json") == completed_hash
        assert runner.read(working.with_suffix(".json.interrupted_1")) == {"uncommitted_outer_file": True}
        resumed = runner.checked(monitor.state["progress"]["case_00"]["trials"]["native_PKSimStandard"])
        assert resumed["status"] == "predicted" and resumed["result_ref"]["sha256"] == completed_hash
    elif recovery == "interrupted_without_receipt":
        preserved = working.with_name(working.name + ".interrupted_2")
        assert (preserved / "partial.pkml").read_text() == "synthetic interrupted native bytes"
        assert (working.with_name(working.name + ".interrupted_1") / "older.txt").read_text() == "older diagnosis must survive"
        assert (working / "result.json").is_file() and not (working / "partial.pkml").exists()
        assert monitor.state.get("consecutive_infrastructure_errors", 0) == 0
    for method, entry in bundle["entries"].items():
        cl = next(e["value"] for e in entry["request"]["evidence"] if e["parameter"] == "total_clearance")
        assert cl == (rf["total_clearance"] if method == "rf_standard" else rich["total_clearance"])
        assert entry["target_observations_used"] is False
        assert entry["endpoints"]["comparison_observations_used"] is False
        assert entry["endpoints"]["endpoint_estimator_uses_observations"] is False


def test_bootstrap_resamples_whole_studies_but_estimate_is_drug_equal():
    cases = cases_twenty()
    rows = []
    for i, case in enumerate(cases):
        for method in runner.METHODS:
            row = {"case_id": case["case_id"], "method": method, "endpoints": {}}
            if not (method == "rf_standard" and i == 19):
                row["curve_metrics"] = {"gmfe": math.exp((i + 1) / 20.) * (1. if method == "ours" else 2.)}
            rows.append(row)
    summary, comparisons = runner.summarize(rows, cases)
    ours = next(r for r in summary if r["dataset"] == "all_20" and r["method"] == "ours")
    score = ours["curve_gmfe"]
    assert score["n_cases"] == 20 and score["n_clusters"] == 19
    assert score["estimate"] == pytest.approx(math.exp(np.mean(np.arange(1, 21) / 20.)))
    # Independently reconstruct whole-cluster resampling, preserving both drugs
    # in the shared study, rather than bootstrapping individual measurements.
    groups = sorted({c["cluster_id"] for c in cases})
    member_logs = [[(i + 1) / 20. for i, c in enumerate(cases) if c["cluster_id"] == g] for g in groups]
    draws = np.random.default_rng(20261004).integers(0, len(groups), size=(2000, len(groups)))
    expected = [math.exp(np.mean([x for index in draw for x in member_logs[index]])) for draw in draws]
    np.testing.assert_allclose(score["ci95"], np.quantile(expected, [.025, .975]), rtol=1e-12)
    paired = comparisons["all_20"]["rf_standard"]
    assert paired["attempted_pairs"] == 20 and paired["complete_pairs"] == 19
    assert paired["incomplete_case_ids"] == ["case_19"]
    assert paired["estimate"] == pytest.approx(.5)
    np.testing.assert_allclose(paired["ci95"], [.5, .5])
    rf = next(r for r in summary if r["dataset"] == "all_20" and r["method"] == "rf_standard")
    assert rf["attempted"] == 20 and rf["failed_or_unscorable"] == 1


def test_endpoint_summary_keeps_tmax_absolute_error_and_other_fold_errors_separate():
    rows = [{"endpoints": {"tmax": {"error": 0., "reference_kind": "original_reported"},
                           "clearance": {"error": 1., "reference_kind": "original_reported"}}},
            {"endpoints": {"tmax": {"error": 2., "reference_kind": "original_reported"},
                           "clearance": {"error": 4., "weak_tail": True, "reference_kind": "original_reported"}}},
            {"endpoints": {}},
            {"endpoints": {"clearance": {"error": 10., "reference_kind": "observation_derived"}}}]
    result = runner.endpoint_summary(rows)
    tmax = result["tmax"]["original_reported"]
    clearance = result["clearance"]["original_reported"]
    assert tmax["gmfe"] is None and tmax["mae_h"] == 1.
    assert clearance["gmfe"] == pytest.approx(2.)
    assert clearance["scored"] == 2 and clearance["attempted"] == 4
    assert clearance["weak_tail"] == 1
    assert result["clearance"]["observation_derived"]["gmfe"] == pytest.approx(10.)
