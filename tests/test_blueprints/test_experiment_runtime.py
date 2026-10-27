"""Toy accounting/optimizer tests; no data assets, native solver or GPU calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import json

import pytest

from pbpk_kb.blueprints import experiment_runtime as runtime
from pbpk_kb.blueprints.design_diagnostics import BudgetExceeded, BudgetLedger


def test_atomic_progress_recovers_past_stale_temp_and_preserves_frozen_results(tmp_path):
    path=tmp_path/'progress.json'
    stale=tmp_path/'progress.json.tmp';stale.write_text('interrupted previous write')
    runtime.save(path,{'stage':1},replace=True)
    runtime.save(path,{'stage':2},replace=True)
    assert json.loads(path.read_text())=={'stage':2}
    assert stale.read_text()=='interrupted previous write'
    with pytest.raises(FileExistsError):runtime.save(path,{'stage':3})
    assert json.loads(path.read_text())=={'stage':2}
    assert list(tmp_path.glob('.progress.json.*.tmp'))==[]


@pytest.fixture(autouse=True)
def forbid_real_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This test must not invoke a native solver or GPU scorer")
    monkeypatch.setattr(runtime, "execute_snapshot", forbidden)
    monkeypatch.setattr(runtime, "run_parameter_batch", forbidden)
    monkeypatch.setattr(runtime, "score_calibration_curves", forbidden)


@pytest.fixture
def case():
    return {
        "purpose": "historical_development_reconstruction",
        "scenarios": [
            {"scenario_id": "study_a_iv", "study_group_id": "study_a", "role": "calibration"},
            {"scenario_id": "study_a_po", "study_group_id": "study_a", "role": "calibration"},
            {"scenario_id": "study_b_iv", "study_group_id": "study_b", "role": "selection"},
        ],
        "forbidden_study_group_ids": ["unopened_test_study"],
        "parameter_search": {"parameters": [
            {"parameter_id": "free", "initial_value": 0.2, "lower_bound": 0.0, "upper_bound": 1.0},
            {"parameter_id": "fixed", "initial_value": 0.10754414957114701,
             "lower_bound": 1e-12, "upper_bound": 1.0},
        ]},
    }


def test_native_parallelism_expands_scenarios_within_shared_cpu_budget(monkeypatch):
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: set(range(56)))
    assert runtime.validate_native_parallelism(cpu_threads=8, scenario_workers=3, drug_workers=2) == {
        'native_cpu_concurrency_limit':48, 'available_cpu_affinity':56}
    with pytest.raises(ValueError, match='exceeds CPU affinity'):
        runtime.validate_native_parallelism(cpu_threads=8, scenario_workers=3, drug_workers=3)
    for values in ({'cpu_threads':True, 'scenario_workers':3},
                   {'cpu_threads':4, 'scenario_workers':0},
                   {'cpu_threads':33, 'scenario_workers':1}):
        with pytest.raises(ValueError):runtime.validate_native_parallelism(**values)


def test_invalid_parallelism_is_rejected_before_creating_output(case, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: {0,1})
    out=tmp_path/'excessive'
    with pytest.raises(ValueError, match='exceeds CPU affinity'):
        runtime.NativeExperiment(case=case, output=out, project_root=tmp_path,
                                 cpu_threads=2, scenario_workers=3, device='cpu')
    assert not out.exists()


def test_three_scenarios_run_concurrently_and_keep_study_weighting(case,tmp_path,monkeypatch):
    from threading import Barrier
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: set(range(56)))
    case['scenarios'].insert(2,{'scenario_id':'study_c_iv','study_group_id':'study_c','role':'calibration'})
    experiment=runtime.NativeExperiment(case=case, output=tmp_path/'parallel', project_root=tmp_path,
                                       cpu_threads=4, scenario_workers=3, device='cpu')
    barrier=Barrier(3)
    def predict(candidate,scenario,points,context,phase):
        barrier.wait(timeout=3)
        return [{'scenario':scenario['scenario_id']}]
    def score(*,runs,observations,device):
        error={'study_a_iv':1.,'study_a_po':9.,'study_c_iv':25.}[runs[0]['scenario']]
        return {'rows':[{'valid':True,'log_mse':error,'gmfe':1.1}]}
    for scene in case['scenarios']:scene['observations']=[]
    monkeypatch.setattr(experiment,'_predict',predict)
    monkeypatch.setattr(runtime,'score_calibration_curves',score)
    try:
        result=experiment.score({},[[.2,.1]],{},role='calibration')
        assert result[0]['log_rmse']==pytest.approx(math.sqrt(((1+9)/2+25)/2))
    finally:experiment.close()


def test_same_study_arms_stay_together_and_cross_role_overlap_is_rejected(case):
    assert runtime.validate_case(case) == {"calibration": {"study_a"}, "selection": {"study_b"}}
    case["scenarios"][2]["study_group_id"] = "study_a"
    with pytest.raises(ValueError, match="disjoint study groups"):
        runtime.validate_case(case)


@pytest.mark.parametrize("change,match", [
    ("forbidden", "forbidden study"),
    ("duplicate", "identifiers must be unique"),
    ("test_role", "only calibration/selection"),
    ("no_selection", "nonempty disjoint"),
])
def test_invalid_study_membership_is_rejected(case, change, match):
    if change == "forbidden":
        case["scenarios"][2]["study_group_id"] = "unopened_test_study"
    elif change == "duplicate":
        case["scenarios"][2]["scenario_id"] = case["scenarios"][0]["scenario_id"]
    elif change == "test_role":
        case["scenarios"][2]["role"] = "test"
    else:
        case["scenarios"] = case["scenarios"][:2]
    with pytest.raises(ValueError, match=match):
        runtime.validate_case(case)


def test_objective_weights_studies_then_curves_not_point_counts():
    # Study A has two unequal-length curves; study B has one. Point counts
    # deliberately differ but each curve score already summarizes its points.
    scenarios = [
        {"scenario_id": "a1", "study_group_id": "a", "observation_count": 2},
        {"scenario_id": "a2", "study_group_id": "a", "observation_count": 20},
        {"scenario_id": "b1", "study_group_id": "b", "observation_count": 200},
    ]
    scores = [{"rows": [{"valid": True, "log_mse": mse, "gmfe": math.exp(absolute)}]}
              for mse, absolute in [(1, .2), (9, .6), (25, 1.2)]]
    row = runtime.aggregate_scores(scenarios, scores)[0]
    assert row["log_rmse"] == pytest.approx(math.sqrt(((1 + 9) / 2 + 25) / 2))
    assert row["gmfe"] == pytest.approx(math.exp(((.2 + .6) / 2 + 1.2) / 2))
    assert set(row["scenarios"]) == {"a1", "a2", "b1"}
    assert row["log_rmse"] != pytest.approx(math.sqrt((1 + 9 + 25) / 3))


def test_invalid_curve_invalidates_candidate_without_dropping_its_row():
    scenarios = [{"scenario_id": "a", "study_group_id": "a"},
                 {"scenario_id": "b", "study_group_id": "b"}]
    good = {"valid": True, "log_mse": .25, "gmfe": 1.1}
    bad = {"valid": False, "log_mse": None, "gmfe": None, "error": "nonpositive_prediction"}
    result = runtime.aggregate_scores(scenarios, [{"rows": [good, good]}, {"rows": [bad, good]}])
    assert len(result) == 2
    assert result[0]["valid"] is False
    assert result[0]["log_rmse"] is result[0]["gmfe"] is None
    assert result[0]["scenarios"]["b"]["error"] == "nonpositive_prediction"
    assert result[1]["valid"] is True
    assert runtime.loss(result[0]) == math.inf


def test_scoped_limit_charges_cache_hits_and_keeps_diagnostics_separate():
    ledger = BudgetLedger(max_solver_calls=10, diagnostic_fraction=.2)
    scope = runtime.ScopedLedger(ledger, allowance=2)
    scope.consume(2, cache_hit=True)
    assert scope.used == 2
    with pytest.raises(BudgetExceeded):
        scope.run(lambda: pytest.fail("Callback must not run after scope exhaustion"))
    scope.run(lambda: None, scenario_calls=2, phase="diagnostics")
    state = scope.snapshot()
    assert scope.used == 2
    assert state["phases"]["search"]["logical_calls"] == 2
    assert state["phases"]["search"]["cache_hits"] == 2
    assert state["phases"]["diagnostics"]["physical_calls"] == 2


def test_failed_attempt_is_charged_once_without_refund():
    ledger = BudgetLedger(max_solver_calls=10, diagnostic_fraction=0)
    scope = runtime.ScopedLedger(ledger, allowance=3)

    def fail():
        raise RuntimeError("toy native failure")

    with pytest.raises(RuntimeError, match="toy native failure"):
        scope.run(fail, scenario_calls=2)
    assert scope.used == 2
    assert scope.snapshot()["totals"] == {
        "logical_calls": 2, "physical_calls": 2, "cache_hits": 0, "failed_calls": 2, "remaining": 8}
    scope.run(lambda: "ok")
    with pytest.raises(BudgetExceeded):
        scope.run(lambda: pytest.fail("Exhausted callback must not execute"))
    assert scope.snapshot()["totals"]["failed_calls"] == 2
    assert scope.snapshot()["totals"]["logical_calls"] == 3


@pytest.mark.parametrize("calls", [-1, 0, True, 1.5])
def test_invalid_scope_charge_cannot_change_available_allowance(calls):
    scope = runtime.ScopedLedger(BudgetLedger(10, 0), allowance=4)
    before = deepcopy(scope.snapshot())
    with pytest.raises(ValueError):
        scope.consume(calls)
    assert scope.used == 0
    assert scope.snapshot() == before


def test_global_budget_rejection_does_not_charge_scope_or_execute():
    scope = runtime.ScopedLedger(BudgetLedger(2, 0), allowance=8)
    with pytest.raises(BudgetExceeded):
        scope.run(lambda: pytest.fail("Global budget must reject before callback"), scenario_calls=3)
    assert scope.used == 0
    assert scope.snapshot()["totals"]["logical_calls"] == 0


def test_parallel_reservations_cannot_exceed_scope():
    scope = runtime.ScopedLedger(BudgetLedger(50, 0), allowance=7)

    def attempt(_):
        try:
            scope.run(lambda: None)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(attempt, range(20)))
    assert sum(outcomes) == 7
    assert scope.used == scope.snapshot()["totals"]["logical_calls"] == 7


def _toy_fit(case, allowance=400, *, fit_start_strategy='legacy', target=.65):
    # Avoid NativeExperiment.__init__: no output directory or worker/solver is
    # needed to exercise the actual fitter and its exact fixed-value handling.
    runner = runtime.NativeExperiment.__new__(runtime.NativeExperiment)
    runner.case = deepcopy(case)
    runner.specs = deepcopy(case["parameter_search"]["parameters"])
    runner.ids = [p["parameter_id"] for p in runner.specs]
    runner.initial = [p["initial_value"] for p in runner.specs]
    runner.progress_callback = lambda **items: None
    runner.fit_start_strategy = fit_start_strategy
    scope = runtime.ScopedLedger(BudgetLedger(1000, 0), allowance=allowance)
    seen = []

    def fake_score(candidate, points, context, *, role="calibration", phase="search"):
        assert role == "calibration"
        assert phase == "search"

        def objective():
            seen.extend(deepcopy(points))
            return [{"valid": True, "log_rmse": abs(point[0] - target), "gmfe": math.exp(abs(point[0] - target))}
                    for point in points]

        # There are two calibration scenario curves, so every vector costs two
        # logical calls even though this test has no physical solver.
        return context["ledger"].run(objective, scenario_calls=2 * len(points))

    runner.score = fake_score
    return runner, {"ledger": scope}, seen


def test_fit_preserves_exact_fixed_value_and_obeys_per_start_limit(case):
    runner, context, seen = _toy_fit(case)
    best, report = runner.fit({"candidate_id": "toy"}, ["free"], context)
    assert seen
    assert all(point[1].hex() == runner.initial[1].hex() for point in seen)
    assert best["point"][1].hex() == runner.initial[1].hex()
    assert best["score"]["log_rmse"] < report["initial_score"]["log_rmse"]
    assert len(report["starts"]) == 3
    assert all(start["evaluations"] <= 50 for start in report["starts"])
    assert len(seen) == 1 + sum(start["evaluations"] for start in report["starts"])
    assert len(seen) <= 151
    assert context["ledger"].used == 2 * len(seen)
    assert all(0 <= point[0] <= 1 for point in seen)
    assert report["optimizer_convergence_proven"] is False


def test_source_anchor_refines_nearby_low_budget_incumbent_that_legacy_leaves_unchanged(case):
    case['parameter_search']['parameters'][0]['initial_value'] = .72
    results = {}
    for strategy in ('legacy', 'source_anchor'):
        runner, context, seen = _toy_fit(case, allowance=68, fit_start_strategy=strategy, target=.73)
        best, report = runner.fit({'candidate_id': 'nearby'}, ['free'], context)
        results[strategy] = best['score']['log_rmse']
        assert report['budget_truncated'] is True
        assert len(report['starts']) == 3
        assert context['ledger'].used == 2 * len(seen) <= 68
    assert results['legacy'] == pytest.approx(.01)
    assert results['source_anchor'] < results['legacy']


def test_source_anchor_reuses_initial_score_once_and_preserves_exact_fixed_dimensions(case):
    runner, context, seen = _toy_fit(case, fit_start_strategy='source_anchor')
    best, report = runner.fit({'candidate_id': 'anchor'}, ['free'], context)
    assert report['fit_start_strategy'] == 'source_anchor'
    assert report['initial_score_reused_as_start'] is True
    assert report['starts'][0]['initial_point'] == runner.initial
    assert [p[0] for p in seen[:3]] == pytest.approx([.2, .1, .9])
    assert all(p[1].hex() == runner.initial[1].hex() for p in seen)
    assert best['point'][1].hex() == runner.initial[1].hex()
    assert len(seen) == sum(s['evaluations'] for s in report['starts'])
    assert context['ledger'].used == 2 * len(seen)
    assert all(s['evaluations'] <= 50 for s in report['starts'])


def test_anchor_equal_to_other_start_still_charges_other_requested_evaluation(case):
    case['parameter_search']['parameters'][0]['initial_value'] = .1
    runner, context, seen = _toy_fit(case, allowance=8, fit_start_strategy='source_anchor')
    best, report = runner.fit({'candidate_id': 'duplicate_start'}, ['free'], context)
    assert seen == [runner.initial, runner.initial, [.9, runner.initial[1]]]
    assert context['ledger'].used == 6
    assert len(report['starts']) == 3
    assert report['initial_score_reused_as_start'] is True
    assert report['budget_stop_stage'] == 'before_coordinate_batch'


def test_legacy_default_keeps_original_start_points_and_accounting(case, tmp_path):
    actual = runtime.NativeExperiment(case=case, output=tmp_path/'default', project_root=tmp_path,
                                      cpu_threads=1, scenario_workers=1, device='cpu')
    try:
        assert actual.fit_start_strategy == 'legacy'
    finally:
        actual.close()
    runner, context, seen = _toy_fit(case, allowance=10)
    best, report = runner.fit({'candidate_id': 'legacy'}, ['free'], context)
    assert [p[0] for p in seen] == pytest.approx([.2, .1, .5, .9])
    assert report['fit_start_strategy'] == 'legacy'
    assert report['initial_score_reused_as_start'] is False
    assert len(seen) == 1 + sum(s['evaluations'] for s in report['starts'])
    assert context['ledger'].used == 8


def test_unknown_start_strategy_is_rejected_before_creating_output(case, tmp_path):
    output = tmp_path/'unknown_strategy'
    with pytest.raises(ValueError, match='fit_start_strategy'):
        runtime.NativeExperiment(case=case, output=output, project_root=tmp_path, fit_start_strategy='random')
    assert not output.exists()


def test_no_free_parameters_only_scores_common_initial_once(case):
    runner, context, seen = _toy_fit(case)
    best, report = runner.fit({"candidate_id": "fixed"}, [], context)
    assert seen == [runner.initial]
    assert best["point"] == runner.initial
    assert best["source"] == "initial"
    assert report["starts"] == []
    assert context["ledger"].used == 2


def test_fit_stops_before_next_batch_and_preserves_initial_incumbent(case):
    runner, context, seen = _toy_fit(case, allowance=5)
    best, report = runner.fit({"candidate_id": "too_small"}, ["free"], context)
    assert seen == [runner.initial]
    assert context["ledger"].used == 2
    assert context["ledger"].snapshot()["totals"]["logical_calls"] == 2
    assert best['point'] == runner.initial
    assert report['budget_truncated'] is True
    assert report['budget_stop_stage'] == 'before_multistart_batch'
    assert report['starts'] == []


def test_nested_scope_remaining_respects_both_phase_and_method_budget():
    ledger = BudgetLedger(500, .2)
    phase = runtime.ScopedLedger(ledger, 200)
    fitting = runtime.ScopedLedger(phase, 196)
    fitting.consume(190)
    assert fitting.remaining() == 6
    assert phase.remaining() == 10
    assert fitting.remaining('diagnostics') == 100
    phase.consume(8)
    assert fitting.remaining() == 2
    with pytest.raises(BudgetExceeded):
        fitting.consume(3)
    assert fitting.remaining() == 2


def _budgeted_candidate_fixture(case, tmp_path):
    # Three Dcal curves reproduce Alprazolam's real scenario multiplicity.
    # Objective values below are synthetic; no drug observations are accessed.
    case = deepcopy(case)
    case['scenarios'].insert(2, {'scenario_id': 'study_c_iv', 'study_group_id': 'study_c', 'role': 'calibration'})
    runner = runtime.NativeExperiment.__new__(runtime.NativeExperiment)
    runner.case = case
    runner.specs = deepcopy(case['parameter_search']['parameters'])
    runner.ids = [p['parameter_id'] for p in runner.specs]
    runner.initial = [p['initial_value'] for p in runner.specs]
    runner.progress_callback = lambda **kwargs: None
    candidate = {'candidate_id': 'native__11', 'structure_id': 'native',
                 'structure_semantic_hash': 'toy-only', 'parameterization_hash': 'toy-only',
                 'free_parameter_ids': runner.ids[:], 'parameter_ids': runner.ids[:]}
    ledger = BudgetLedger(500, .2)
    context = {'ledger': ledger, 'model_requests': set(), 'output': tmp_path / 'method'}
    calls = []
    def score(candidate, points, current, *, role='calibration', phase='search'):
        scenarios = [s for s in case['scenarios'] if s['role'] == role]
        for scenario in scenarios:
            key = (candidate['structure_id'], scenario['scenario_id'])
            if key not in current['model_requests']:
                current['ledger'].consume(phase=phase)
                current['model_requests'].add(key)
        current['ledger'].consume(len(scenarios) * len(points), phase=phase)
        calls.append({'role': role, 'vectors': len(points), 'phase': phase,
                      'parameter_points': deepcopy(points)})
        return [{'valid': True, 'log_rmse': .1 + sum((x-.7)**2 for x in point), 'gmfe': 1.1,
                 'scenarios': {s['scenario_id']: {'predictions_umol_per_l': [math.exp(point[0]), math.exp(point[1])]}
                               for s in scenarios}} for point in points]
    runner.score = score
    return runner, candidate, context, calls


@pytest.mark.parametrize('phase,allowance,adaptation', [('warm_start', 400, False),
                                                      ('warm_start', 400, True),
                                                      ('structure', 200, False)])
def test_500_budget_three_calibration_scenarios_retains_prediction_and_reserved_dsel(
    case, tmp_path, phase, allowance, adaptation,
):
    runner, candidate, context, calls = _budgeted_candidate_fixture(case, tmp_path)
    # Actual finite-difference logic consumes the same cost as production, with
    # synthetic predicted curves supplied above.
    for scenario in runner.case['scenarios']:
        scenario['observations'] = [{'time_min': 1., 'concentration_umol_per_l': 1.},
                                    {'time_min': 2., 'concentration_umol_per_l': 1.}]
    result = runner.evaluator(context)(candidate, method_config={
        'phase': phase, 'remaining_solver_calls': allowance,
        'identifiability_prescreen': adaptation, 'identifiability_feedback': adaptation,
        'fixed_parameterization': not adaptation})
    assert result['status'] == 'budget_exhausted'
    assert result['prediction_available'] is True
    assert result['selection_log_rmse'] is not None
    assert result['fit']['budget_truncated'] is True
    assert result['budget_scope'] == ('phase' if phase == 'structure' else 'method')
    assert result['solver_calls'] <= allowance
    assert result['selection_solver_calls_reserved'] == 2
    assert [c['role'] for c in calls].count('selection') == 1
    assert calls[-1]['role'] == 'selection'
    assert calls[-1]['parameter_points'] == [result['fitted_values']]
    assert context['ledger'].snapshot()['phases']['diagnostics']['remaining'] == 100
    assert result['identifiability']['status'] in {'evaluated', 'unavailable', 'no_free_parameters'}
    assert not context.get('infrastructure_failure')


def test_budget_unavailable_diagnostic_spends_no_partial_steps_or_final_reserve(case, tmp_path):
    runner, candidate, context, calls = _budgeted_candidate_fixture(case, tmp_path)
    context['ledger'].consume(399)
    diagnostic = runner.bounded_diagnose(candidate, runner.initial, runner.ids, context)
    assert diagnostic['status'] == 'unavailable'
    assert diagnostic['rank'] is None
    assert diagnostic['selected_parameter_ids'] == runner.ids
    assert diagnostic['step_stability_checked'] is False
    assert not calls
    assert context['ledger'].snapshot()['phases']['search']['remaining'] == 1
    assert context['ledger'].snapshot()['phases']['diagnostics']['remaining'] == 100


def test_truncated_valid_prediction_is_selection_eligible_but_not_completed_search(case, tmp_path):
    from pbpk_kb.blueprints.search_methods import run_method_search
    runner, candidate, context, calls = _budgeted_candidate_fixture(case, tmp_path)
    candidate.update(features=[], requirement_score=1., evidence_score=1., structure_edit_count=0)
    for scenario in runner.case['scenarios']:
        scenario['observations'] = [{'time_min': 1., 'concentration_umol_per_l': 1.}]
    result = run_method_search(method='M3', candidates=[candidate], evaluate=runner.evaluator(context),
                              base_candidate_id=candidate['candidate_id'], max_solver_calls=400)
    assert result['status'] == 'budget_exhausted'
    assert result['selected_candidate']['candidate_id'] == candidate['candidate_id']
    assert result['selected_result']['prediction_available'] is True
    assert result['evaluations'][0]['selection_eligible'] is True
    assert result['solver_calls'] <= 400


def test_cannot_afford_initial_and_dsel_retains_explicit_missing_prediction(case, tmp_path):
    runner, candidate, context, calls = _budgeted_candidate_fixture(case, tmp_path)
    result = runner.evaluator(context)(candidate, method_config={
        'phase': 'warm_start', 'remaining_solver_calls': 5,
        'identifiability_prescreen': False, 'identifiability_feedback': False})
    assert result['status'] == 'budget_exhausted'
    assert not result.get('prediction_available')
    assert not calls
    assert result['solver_calls'] == 0


def test_m5_truncated_structure_fit_keeps_prediction_and_enters_parameter_phase(case, tmp_path):
    from pbpk_kb.blueprints.search_methods import run_method_search
    runner, full, context, calls = _budgeted_candidate_fixture(case, tmp_path)
    full.update(features=[], requirement_score=1., evidence_score=1., structure_edit_count=0)
    fixed = dict(full, candidate_id='native__00', free_parameter_ids=[])
    for scenario in runner.case['scenarios']:
        scenario['observations'] = [{'time_min': 1., 'concentration_umol_per_l': 1.},
                                    {'time_min': 2., 'concentration_umol_per_l': 1.}]
    result = run_method_search(method='M5', candidates=[fixed, full], evaluate=runner.evaluator(context),
                              base_candidate_id=full['candidate_id'], max_solver_calls=400)
    first = result['evaluations'][0]
    assert first['phase'] == 'structure'
    assert first['result']['status'] == 'budget_exhausted'
    assert first['result']['budget_scope'] == 'phase'
    assert first['result']['prediction_available'] is True
    assert first['result']['solver_calls'] <= 200
    assert [r['phase'] for r in result['evaluations']] == ['structure', 'parameter_subset']
    assert result['locked_structure_id'] == 'native'
    assert result['selected_result']['prediction_available'] is True
    assert result['solver_calls'] <= 400
    assert context['ledger'].snapshot()['phases']['diagnostics']['remaining'] == 100


def test_declared_linear_logp_and_log_clearance_coordinates_are_distinct(case):
    case['parameter_search']['parameters']=[
        {'parameter_id':'logp','initial_value':1.75,'lower_bound':1.3,'upper_bound':2.,'search_scale':'linear'},
        {'parameter_id':'cl','initial_value':.504,'lower_bound':.1,'upper_bound':2.,'search_scale':'log'}]
    runner,_,_=_toy_fit(case)
    middle=runner._decode([.5,.5])
    assert middle==pytest.approx([1.65,math.sqrt(.2)])
    assert runner._encode(middle)==pytest.approx([.5,.5])


def test_logp_sensitivity_uses_bounded_linear_coordinate(case):
    case['parameter_search']['parameters'][0].update(initial_value=1.75,lower_bound=1.3,
         upper_bound=2.,search_scale='linear',sensitivity_scale='bounded_linear')
    for s in case['scenarios']:
        s['observations']=[{'time_min':1,'concentration_umol_per_l':1}]
    runner,context,_=_toy_fit(case)
    runner.case=case
    def score(candidate,points,context,**kwargs):
        return [{'valid':True,'scenarios':{
                    s['scenario_id']:{'predictions_umol_per_l':[math.exp(point[0])]}
                    for s in case['scenarios'] if s['role']=='calibration'}} for point in points]
    runner.score=score
    diagnostic=runner.diagnose({'candidate_id':'toy'},runner.initial,['free'],context)
    assert diagnostic['step_stability_passed'] is True
    assert diagnostic['column_norms']['free']==pytest.approx(.7)
    assert diagnostic['parameter_coordinates']['free']=='bounded_linear'


def _residual_toy(case, *, allowance=400, matrix=((1., 1.), (1., -1.)), target=(.7, .8),
                  log_response=None):
    """Coupled exact log responses with real aggregate weighting and accounting."""
    runner = runtime.NativeExperiment.__new__(runtime.NativeExperiment)
    runner.case = deepcopy(case)
    first = dict(case['parameter_search']['parameters'][0], sensitivity_scale='bounded_linear')
    second = dict(first, parameter_id='volume', initial_value=.2)
    fixed = deepcopy(case['parameter_search']['parameters'][1])
    runner.specs = [first, second, fixed]
    runner.case['parameter_search']['parameters'] = runner.specs
    runner.ids = [p['parameter_id'] for p in runner.specs]
    runner.initial = [p['initial_value'] for p in runner.specs]
    runner.fit_strategy = 'residual'
    runner.require_identifiability = False
    runner.progress_callback = lambda **kwargs: None
    for scene in runner.case['scenarios']:
        scene['observations'] = [{'time_min': i + 1., 'concentration_umol_per_l': 1.}
                                 for i in range(len(matrix))]
    candidate = {'candidate_id': 'native__110', 'structure_id': 'native',
                 'structure_semantic_hash': 'toy', 'free_parameter_ids': ['free', 'volume']}
    ledger = runtime.ScopedLedger(BudgetLedger(1000, .2), allowance)
    context = {'ledger': ledger, 'model_requests': set()}
    seen = []

    def score(candidate, points, current, *, role='calibration', phase='search'):
        scenes = [s for s in runner.case['scenarios'] if s['role'] == role]
        cost = runner._score_call_cost(candidate, len(points), current, role=role)
        assert cost <= runner._remaining_calls(current, phase)
        current['ledger'].consume(cost, phase=phase)
        current['model_requests'].update((candidate['structure_id'], s['scenario_id']) for s in scenes)
        seen.extend({'role': role, 'point': list(point)} for point in points)
        scores = []
        for scene in scenes:
            rows = []
            for point in points:
                log_errors = (log_response(point) if log_response is not None else
                    [a * (point[0] - target[0]) + b * (point[1] - target[1]) for a, b in matrix])
                rows.append({'valid': True, 'log_mse': sum(x*x for x in log_errors)/len(log_errors),
                             'gmfe': math.exp(sum(abs(x) for x in log_errors)/len(log_errors)),
                             'predictions_umol_per_l': [math.exp(x) for x in log_errors]})
            scores.append({'rows': rows})
        return runtime.aggregate_scores(scenes, scores)
    runner.score = score
    return runner, candidate, context, seen


def test_residual_fit_jointly_recovers_coupled_parameters_and_preserves_fixed_bits(case):
    runner, candidate, context, seen = _residual_toy(case)
    best, report = runner.fit(candidate, candidate['free_parameter_ids'], context)
    assert best['point'][:2] == pytest.approx([.7, .8], abs=2e-4)
    assert best['score']['log_rmse'] < 2e-4
    assert all(row['point'][2].hex() == runner.initial[2].hex() for row in seen)
    assert all(row['role'] == 'calibration' for row in seen)
    assert report['fit_strategy'] == 'residual'
    assert report['optimizer_convergence_proven'] is False
    assert len(report['trace']) <= 20
    assert all(item['local_rank'] == 2 for item in report['trace'])
    assert report['starts'][0]['evaluations'] == len(seen)
    assert context['ledger'].used == 2 + 2 * len(seen)


def test_clearance_only_cannot_correct_opposite_early_late_residuals_but_joint_fit_can(case):
    clear, candidate, context, _ = _residual_toy(case, target=(.2, .8))
    single, _ = clear.fit(candidate, ['free'], context)
    joint, candidate, context, _ = _residual_toy(case, target=(.2, .8))
    combined, _ = joint.fit(candidate, ['free', 'volume'], context)
    assert single['score']['log_rmse'] == pytest.approx(.6)
    assert combined['score']['log_rmse'] < 2e-4
    assert single['point'][1].hex() == clear.initial[1].hex()


def test_residual_fitter_respects_source_bounds_when_optimum_is_outside(case):
    runner, candidate, context, seen = _residual_toy(case, target=(1.8, -.6))
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    assert best['point'][:2] == pytest.approx([1., 0.], abs=2e-4)
    assert best['score']['log_rmse'] > 0
    assert all(0 <= row['point'][0] <= 1 and 0 <= row['point'][1] <= 1 for row in seen)
    assert report['optimizer_convergence_proven'] is False


def test_residual_fitter_reports_local_rank_deficiency_without_silently_fixing_parameter(case):
    runner, candidate, context, _ = _residual_toy(case, matrix=((1., 1.), (2., 2.)), target=(.7, .7))
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    assert best['score']['log_rmse'] < 2e-4
    assert best['point'][0] > runner.initial[0] and best['point'][1] > runner.initial[1]
    assert all(item['local_rank'] == 1 and item['local_not_global'] for item in report['trace'])
    assert report['trace'][0]['correlated_parameter_pairs']
    assert 'not global identifiability' in report['rank_interpretation']


@pytest.mark.parametrize('allowance,stage,count', [(5, 'before_residual_jacobian_batch', 1),
                                                (12, 'before_residual_line_search_batch', 5)])
def test_residual_budget_stops_before_whole_batch_and_retains_scored_incumbent(case, allowance, stage, count):
    runner, candidate, context, seen = _residual_toy(case, allowance=allowance)
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    assert len(seen) == count
    assert best['point'] in [row['point'] for row in seen]
    assert best['score']['log_rmse'] <= report['initial_score']['log_rmse']
    assert report['budget_truncated'] is True
    assert report['budget_stop_stage'] == stage
    assert context['ledger'].used <= allowance
    if count > 1:
        # A real finite-difference improvement is retained even when the next
        # optimizer proposal cannot be afforded.
        assert best['score']['log_rmse'] < report['initial_score']['log_rmse']


def test_residual_weights_match_study_curve_point_objective_with_unequal_curves(case):
    runner, candidate, context, _ = _residual_toy(case)
    scenes = runner.case['scenarios']
    scenes.insert(2, {'scenario_id': 'other', 'study_group_id': 'other', 'role': 'calibration',
                     'observations': [{'concentration_umol_per_l': 1.}] * 5})
    calibration = [s for s in scenes if s['role'] == 'calibration']
    calibration[0]['observations'] = [{'concentration_umol_per_l': 1.}] * 2
    calibration[1]['observations'] = [{'concentration_umol_per_l': 1.}] * 20
    scores = [{'rows': [{'valid': True, 'log_mse': v*v, 'gmfe': math.exp(v),
                         'predictions_umol_per_l': [math.exp(v)] * len(s['observations'])}]}
              for s, v in zip(calibration, [1., 3., 5.], strict=True)]
    result = runtime.aggregate_scores(calibration, scores)[0]
    residual = runner._calibration_residuals(result)
    assert len(residual) == 27
    assert sum(x*x for x in residual) == pytest.approx(((1+9)/2+25)/2)
    result['log_rmse'] += .1
    with pytest.raises(ValueError, match='disagree'):
        runner._calibration_residuals(result)


def test_required_posterior_rank_check_reserves_search_calls_and_dsel_without_automatic_shrink(case, tmp_path):
    runner, candidate, context, seen = _residual_toy(case, allowance=40,
                                                 matrix=((1., 1.), (2., 2.)), target=(.7, .7))
    runner.require_identifiability = True
    context['output'] = tmp_path/'required_diagnostic'
    before = context['ledger'].snapshot()['phases']['diagnostics']
    result = runner.evaluator(context)(candidate, method_config={
        'remaining_solver_calls': 40, 'identifiability_prescreen': True,
        'identifiability_feedback': True, 'fixed_parameterization': False})
    assert result['status'] == 'budget_exhausted'
    assert result['prediction_available'] is True
    assert result['fit']['budget_truncated'] is True
    assert result['identifiability']['status'] == 'evaluated'
    assert result['identifiability']['rank'] == 1
    assert result['identifiability']['step_stability_passed'] is True
    assert result['effective_free_parameter_ids'] == ['free', 'volume']
    assert result['identifiability_refit_performed'] is False
    assert result['prescreen'] is None
    assert result['identifiability_solver_calls_reserved'] == 16
    assert result['selection_solver_calls_reserved'] == 2
    assert result['solver_calls'] <= 40
    assert [row['role'] for row in seen].count('selection') == 1
    assert context['ledger'].snapshot()['phases']['diagnostics'] == before


@pytest.mark.parametrize('options,match', [({'fit_strategy': 'invented'}, 'fit_strategy'),
                                        ({'require_identifiability': 1}, 'require_identifiability')])
def test_invalid_residual_options_do_not_create_output(case, tmp_path, options, match):
    output = tmp_path/'invalid_residual_options'
    with pytest.raises(ValueError, match=match):
        runtime.NativeExperiment(case=case, output=output, project_root=tmp_path, **options)
    assert not output.exists()


@pytest.mark.parametrize('free_ids', [['free'], ['free', 'volume']])
def test_screened_residual_escapes_source_local_basin_using_only_real_calibration_scores(case, free_ids):
    def response(point):
        return [(point[axis] - .9) * ((point[axis] - .2) ** 2 + .02)
                for axis in range(len(free_ids))]
    ordinary, candidate, context, _ = _residual_toy(case,
        matrix=[(1., 0.)] * len(free_ids), log_response=response)
    local, old_report = ordinary.fit(candidate, free_ids, context)
    runner, candidate, context, seen = _residual_toy(case,
        matrix=[(1., 0.)] * len(free_ids), log_response=response)
    runner.fit_strategy = 'screened_residual'
    best, report = runner.fit(candidate, free_ids, context)
    assert local['score']['log_rmse'] > .01
    assert best['score']['log_rmse'] < 1e-10
    assert old_report['fit_strategy'] == 'residual' and 'start_screening' not in old_report
    screening = report['start_screening']
    assert screening['complete'] is True and screening['status'] == 'completed'
    assert screening['planned_grid_points'] == screening['evaluated_grid_points'] == 3 ** len(free_ids)
    assert screening['selected_origin'] == 'grid'
    assert screening['selected_point'][:len(free_ids)] == pytest.approx([.9] * len(free_ids))
    assert report['starts'][0]['initial_point'] == screening['selected_point']
    assert report['initial_score_reused_as_start'] is False
    assert report['selected_score_reused_as_start'] is True
    assert sum(row['point'] == screening['selected_point'] for row in seen) == 1
    assert all(row['role'] == 'calibration' for row in seen)
    fixed = set(range(len(runner.ids))) - {runner.ids.index(name) for name in free_ids}
    assert all(row['point'][axis].hex() == runner.initial[axis].hex() for row in seen for axis in fixed)
    assert screening['solver_calls'] == 2 * 3 ** len(free_ids)
    assert report['initial_solver_calls'] == 4
    assert report['fit_solver_calls'] == context['ledger'].used == 2 + 2 * len(seen)
    assert report['local_solver_calls'] + screening['solver_calls'] + 4 == report['fit_solver_calls']
    assert screening['global_optimum_claimed'] is False and report['optimizer_convergence_proven'] is False


@pytest.mark.parametrize('allowance,expected_grid', [(5, 0), (20, 0), (22, 9), (30, 9)])
def test_screened_residual_grid_and_local_fit_share_same_budget_without_partial_grid(case, allowance, expected_grid):
    runner, candidate, context, seen = _residual_toy(case, allowance=allowance)
    runner.fit_strategy = 'screened_residual'
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    screening = report['start_screening']
    assert screening['evaluated_grid_points'] == expected_grid
    assert screening['complete'] is bool(expected_grid)
    assert screening['solver_calls'] == 2 * expected_grid
    assert context['ledger'].used <= allowance
    assert report['fit_solver_calls'] == context['ledger'].used
    assert report['budget_truncated'] is True
    assert best['point'] in [row['point'] for row in seen]
    assert best['score']['log_rmse'] <= report['initial_score']['log_rmse']
    assert all(row['role'] == 'calibration' for row in seen)
    if not expected_grid:
        assert screening['status'] == 'source_only_insufficient_budget'
        assert screening['selected_point'] == runner.initial
        assert screening['required_solver_calls'] == 18
        assert screening['available_solver_calls'] == allowance - 4
        assert report['initial_score_reused_as_start'] is True
    else:
        assert report['budget_stop_stage'] in {
            'before_residual_jacobian_batch', 'before_residual_line_search_batch'}


@pytest.mark.parametrize('free_ids', [['free'], ['free', 'volume']])
def test_screened_residual_reuses_source_when_it_is_a_grid_point(case, free_ids):
    runner, candidate, context, seen = _residual_toy(case, target=(.5, .5))
    runner.fit_strategy = 'screened_residual'
    for axis in range(len(free_ids)):
        runner.initial[axis] = runner.specs[axis]['initial_value'] = .5
    best, report = runner.fit(candidate, free_ids, context)
    screening = report['start_screening']
    assert screening['reused_source_grid_points'] == 1
    assert screening['evaluated_grid_points'] == 3 ** len(free_ids)
    assert screening['solver_calls'] == 2 * (3 ** len(free_ids) - 1)
    assert screening['selected_origin'] == 'source_anchor'
    assert sum(row['point'] == runner.initial for row in seen) == 1
    assert best['point'] == runner.initial
    assert report['initial_score_reused_as_start'] is True


def test_screened_residual_fixed_candidate_uses_only_registered_prediction(case):
    runner, candidate, context, seen = _residual_toy(case)
    runner.fit_strategy = 'screened_residual'
    best, report = runner.fit(candidate, [], context)
    assert best['point'] == runner.initial and len(seen) == 1
    assert report['start_screening']['status'] == 'not_applicable'
    assert report['start_screening']['planned_grid_points'] == 0
    assert report['fit_solver_calls'] == 4 and report['local_solver_calls'] == 0
    assert report['budget_truncated'] is False


def test_screened_residual_uses_normalized_log_coordinates_and_exact_fixed_values(case):
    runner, candidate, context, seen = _residual_toy(case, allowance=22)
    runner.fit_strategy = 'screened_residual'
    runner.specs[0].update(lower_bound=.01, upper_bound=100., search_scale='log')
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    records = report['start_screening']['grid_records']
    assert len(records) == 9
    for record in records:
        coordinate = record['normalized_free_coordinates']
        assert record['point'][0] == pytest.approx(.01 * 10000 ** coordinate[0])
        assert record['point'][1] == pytest.approx(coordinate[1])
        assert record['point'][2].hex() == runner.initial[2].hex()
    assert best['point'] in [row['point'] for row in seen]


def test_screened_residual_keeps_posterior_diagnostic_and_selection_reserves(case, tmp_path):
    runner, candidate, context, seen = _residual_toy(case, allowance=50)
    runner.fit_strategy = 'screened_residual'
    runner.require_identifiability = True
    context['output'] = tmp_path / 'screened_posterior'
    result = runner.evaluator(context)(candidate, method_config={
        'remaining_solver_calls': 50, 'identifiability_prescreen': True,
        'identifiability_feedback': True, 'fixed_parameterization': False})
    assert result['prediction_available'] is True
    assert result['identifiability']['status'] == 'evaluated'
    assert result['identifiability']['rank'] == 2
    assert result['identifiability_solver_calls_reserved'] == 16
    assert result['selection_solver_calls_reserved'] == 2
    assert result['prescreen'] is None and result['identifiability_refit_performed'] is False
    assert result['fit']['fit_strategy'] == 'screened_residual'
    assert result['fit']['start_screening']['complete'] is True
    assert result['solver_calls'] <= 50
    assert [row['role'] for row in seen].count('selection') == 1


def test_screened_residual_option_is_accepted_by_constructor(case, tmp_path):
    experiment = runtime.NativeExperiment(case=case, output=tmp_path / 'screened_option',
        project_root=tmp_path, cpu_threads=1, scenario_workers=1, device='cpu', fit_strategy='screened_residual')
    try:
        assert experiment.fit_strategy == 'screened_residual'
    finally:
        experiment.close()


@pytest.mark.parametrize('all_invalid', [False, True])
def test_screened_residual_requires_a_valid_actually_scored_start(case, all_invalid):
    runner, candidate, context, seen = _residual_toy(case)
    runner.fit_strategy = 'screened_residual'
    original_score = runner.score
    def score(candidate, points, current, **kwargs):
        scores = original_score(candidate, points, current, **kwargs)
        return [dict(result, valid=False, log_rmse=None) if all_invalid or point == runner.initial else result
                for point, result in zip(points, scores, strict=True)]
    runner.score = score
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    assert report['initial_score']['valid'] is False
    assert report['start_screening']['complete'] is True
    assert report['start_screening']['evaluated_grid_points'] == 9
    assert best['point'] in [row['point'] for row in seen]
    if all_invalid:
        assert best['score']['valid'] is False and best['point'] == runner.initial
        assert report['stop_reason'] == 'no_valid_screened_calibration_prediction'
        assert report['starts'] == [] and len(seen) == 10
    else:
        assert best['score']['valid'] is True
        assert best['score']['log_rmse'] < 2e-4
        assert report['start_screening']['selected_origin'] == 'grid'


def test_screened_residual_partial_batch_failure_keeps_only_scored_source_and_real_cost(case):
    runner, candidate, context, seen = _residual_toy(case)
    runner.fit_strategy = 'screened_residual'
    original_score = runner.score
    def score(candidate, points, current, **kwargs):
        if len(points) > 1:
            current['ledger'].consume(2, phase='search')
            raise BudgetExceeded('search', 18, 0)
        return original_score(candidate, points, current, **kwargs)
    runner.score = score
    best, report = runner.fit(candidate, ['free', 'volume'], context)
    screening = report['start_screening']
    assert screening['status'] == 'screening_batch_budget_exhausted'
    assert screening['complete'] is False and screening['evaluated_grid_points'] == 0
    assert screening['solver_calls'] == 2
    assert best['point'] == runner.initial and len(seen) == 1
    assert report['fit_solver_calls'] == 6 and report['local_solver_calls'] == 0
    assert report['budget_truncated'] is True
    assert report['budget_stop_stage'] == 'start_screening_batch_budget_exhausted'
