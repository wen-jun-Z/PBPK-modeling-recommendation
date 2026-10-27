"""Synthetic-only scoring/barrier tests: never open project clinical outcomes."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import score_recommendation_execution as m


def scenario():
    return {'scenario_id': 'clinical:case', 'case_id': 'case', 'compound': 'testdrug',
            'publication_cluster_id': 'one-paper', 'sealed_references': [],
            'normalized_design': {'matrix': 'plasma', 'analyte': 'Total parent plasma drug; not metabolite or unbound concentration',
                                  'endpoint_contracts': [{'endpoint': 'Cmax'}]},
            'curve_views': [{'protocol_fields': {'case_id': 'case_curve'}}]}


def prediction():
    return {'scenario_id': 'clinical:case', 'candidate_id': 'a', 'matrix': 'plasma', 'analyte': 'parent',
            'binding': 'total', 'molecular_weight_g_per_mol': 200.,
            'time_h': [0., 1., 2., 24.], 'concentration_ng_ml': [0., 20., 10., 1.],
            'endpoints': {'sampled_Cmax': {'value': 20., 'unit': 'ng/mL', 'window_h': [0, 24]},
                          'AUC_fixed_sampling_window': {'value': 146., 'unit': 'ng*h/mL', 'window_h': [0, 24]}}}


@pytest.mark.parametrize('value,unit,mw,exposure,expected', [
    (2, 'ug/mL', None, False, 2000), (2, 'ug/L', None, False, 2),
    (100, 'nmol/L', 200, False, 20), (100, 'nM', 200, False, 20),
    (100, 'h*nmol/L', 200, True, 20), (100, 'nM*h', 200, True, 20),
    (2, 'ug*h/mL', None, True, 2000), (120, 'ng*min/mL', None, True, 2)])
def test_explicit_units(value, unit, mw, exposure, expected):
    assert m.convert_unit(value, unit, mw, exposure=exposure) == pytest.approx(expected)


def test_units_and_unqualified_sources_are_not_guessed():
    with pytest.raises(ValueError, match='dose_normalized'):
        m.convert_unit(1, 'kg*ng/(mL*mg)', 200)
    with pytest.raises(ValueError, match='molecular_weight'):
        m.convert_unit(1, 'nmol/L', None)
    row = m.scalar_comparison({'endpoint': 'Cmax', 'value': 100, 'unit': 'ng/mL', 'evaluable_from_source': False}, scenario(), prediction())
    assert row['reason'] == 'source_endpoint_not_qualified'


def test_all_scalar_schemas_and_no_double_conversion():
    for row in [{'endpoint': 'Cmax', 'mean': .02, 'unit': 'ug/mL'},
                {'endpoint': 'Cmax', 'value': 999, 'source_unit': 'ug/mL', 'comparable_value': 20, 'comparable_unit': 'ng/mL'},
                {'endpoint': 'Cmax', 'value': 20, 'unit': 'ng/mL', 'source_value': .02, 'conversion_multiplier': 1000}]:
        result = m.scalar_comparison(row, scenario(), prediction())
        assert result['observed'] == 20 and result['fold_error'] == 1
        assert result['scientific_status'] == 'unknown' and 'estimand' in result['definition_warning']


def test_fixed_auc_requires_exact_definition_and_window():
    p = prediction()
    for name in ['AUClast', 'AUC_last', 'AUCinf', 'AUC_infinity', 'AUC_tau_ss']:
        row = m.scalar_comparison({'endpoint': name, 'value': 146, 'unit': 'ng*h/mL'}, scenario(), p)
        assert row['state'] == 'unavailable' and 'not_last_infinity_or_tau' in row['reason']
    assert m.scalar_comparison({'endpoint': 'AUC_0_24h', 'value': 146, 'unit': 'ng*h/mL'}, scenario(), p)['fold_error'] == 1
    assert m.scalar_comparison({'endpoint': 'AUC0_48', 'value': 146, 'unit': 'ng*h/mL'}, scenario(), p)['state'] == 'unavailable'


def test_caffeine_canonical_rows_filtered_by_case_not_double_counted():
    doc = {'endpoint_rows': [{'case_id': case, 'endpoint': endpoint, 'value': 100, 'unit': unit,
                             'canonical_unit_representation': canonical, 'counts_as_unique_endpoint': canonical}
                            for case in ['case', 'other'] for endpoint in ['cmax', 'auc_last']
                            for unit, canonical in [('ng/mL', False), ('nmol/L', True)]]}
    rows = m.outcome_rows(doc, scenario())
    assert len(rows) == 2 and all(r['unit'] == 'nmol/L' for r in rows)
    assert m.scalar_comparison(rows[0], scenario(), prediction())['observed'] == 20


def curve_doc(points, aggregate=False):
    if aggregate:
        return {'schema_version': 'pbpk-independent-aggregate-curves-sealed-v1', 'case_id': 'case_curve',
                'compound': 'testdrug', 'concentration_unit': 'ng/mL', 'time_unit': 'h',
                'records': [{'time_h': t, 'arithmetic_mean_ng_ml': v, 'published_arithmetic_mean_is_zero': v == 0}
                            for t, v in points]}
    obs = [{'nominal_time_h': t, 'concentration_ng_ml': v,
            'observation_status': 'quantified' if v is not None else 'left_censored_below_LLOQ'} for t, v in points]
    return {'schema_version': 'pbpk-independent-concentration-curves-sealed-v1', 'case_id': 'case_curve',
            'compound': 'testdrug', 'unit': 'ng/ml',
            'records': [{'subject_id': '1', 'treatment': 'R', 'observations': obs},
                        {'subject_id': '1', 'treatment': 'T', 'observations': obs}]}


def test_curve_R_only_and_censoring_does_not_become_zero():
    rows = m.curve_comparison(curve_doc([(0, None), (1, 20), (2, 10)]), prediction(), scenario=scenario())
    assert len(rows) == 1 and rows[0]['treatment'] == 'R'
    assert rows[0]['qualified_point_count'] == 2 and rows[0]['source_point_count'] == 3
    assert rows[0]['window_metrics']['window']['start'] == 1
    assert rows[0]['curve']['gmfe'] == 1
    gap = m.curve_comparison(curve_doc([(0, 1), (1, None), (2, 10)]), prediction(), scenario=scenario())[0]
    # Zero native prediction at t=0 is also explicitly excluded, never pseudocounted.
    assert gap['window_metrics'] is None and len(gap['excluded_points']) == 1
    assert len(gap['prediction_failures']) == 1 and gap['qualified_point_count'] == 2


def test_internal_missing_prevents_auc_but_retains_point_errors():
    p = prediction()
    p['concentration_ng_ml'][0] = 1
    row = m.curve_comparison(curve_doc([(0, 1), (1, None), (2, 10)]), p, scenario=scenario())[0]
    assert row['curve']['gmfe'] == 1 and row['window_metrics'] is None
    assert row['window_reason'] == 'internal_missing_or_censored_points_no_integral_across_gaps'


def test_positive_group_mean_not_filtered_by_individual_lloq_and_no_extrapolation():
    doc = curve_doc([(0, 0), (1, .01), (2, .02)], aggregate=True)
    doc['censoring'] = {'individual_assay_lloq_ng_ml': .1}
    row = m.curve_comparison(doc, prediction(), scenario=scenario())[0]
    assert row['qualified_point_count'] == 2 and row['window_metrics']['state'] == 'computed'
    with pytest.raises(ValueError, match='extrapolation'):
        m.interpolate([0, 1], [0, 2], 2)


def test_bad_prediction_cannot_shorten_observation_window_to_improve_score():
    doc = curve_doc([(1, 20), (2, 10), (24, 1)], aggregate=True)
    good = m.curve_comparison(doc, prediction(), scenario=scenario())[0]
    p = prediction()
    p['concentration_ng_ml'][-1] = 0
    bad = m.curve_comparison(doc, p, scenario=scenario())[0]
    assert good['window_metrics']['state'] == 'computed'
    assert good['qualified_observed_time_h'] == bad['qualified_observed_time_h'] == [1, 2, 24]
    assert bad['window_metrics'] is None and bad['curve']['state'] == 'unavailable'
    assert bad['partial_prediction_diagnostic']['gmfe'] == 1
    assert bad['prediction_coverage_on_qualified_observations'] == pytest.approx(2/3)


def test_curve_source_units_and_complete_subject_denominator():
    doc = curve_doc([(1, 20), (2, 10)])
    rows = m.curve_comparison(doc, prediction(), scenario=scenario())
    assert m.summarize_candidate_curves(rows)['gmfe'] == 1
    assert m.summarize_candidate_curves(rows + [{'state': 'unavailable'}])['state'] == 'unavailable'
    doc['unit'] = 'ug/ml'
    assert m.curve_comparison(doc, prediction(), scenario=scenario())[0]['state'] == 'unavailable'


@pytest.mark.parametrize('times', [[1, 1], [2, 1]])
def test_invalid_observation_order_cannot_produce_complete_curve_score(times):
    row = m.curve_comparison(curve_doc(list(zip(times, [20, 10]))), prediction(), scenario=scenario())[0]
    assert row['state'] == 'unavailable' and row['curve']['state'] == 'unavailable'
    assert row['reason'] == 'qualified_observation_times_duplicate_or_not_increasing'


def test_diagnostic_success_does_not_become_direct_label_or_cross_drug_negative():
    s = scenario()
    a = {'candidate_id': 'a', 'identity_match': True, 'family_id': 'f', 'independence_status': 'pending'}
    n = {'result': {'status': 'prediction_locked'}, 'prediction': prediction()}
    doc = {'case_id': 'case', 'endpoints': [{'endpoint': 'Cmax', 'value': 20, 'unit': 'ng/mL'}]}
    r = m.candidate_comparison(s, a, n, [doc])
    assert r['strict']['scientific_status'] == 'unknown' and not r['strict']['endpoints']
    assert r['reference_diagnostic']['scientific_status'] == 'unknown'
    assert r['reference_diagnostic']['endpoints']['Cmax_reference_vs_group']['observed'] == 20
    excluded = m.candidate_comparison(s, {**a, 'identity_match': False}, None, [])
    assert excluded['strict']['scientific_status'] == 'incompatible'
    assert excluded['reference_diagnostic']['scientific_status'] == 'unknown'
    scoped = m.candidate_comparison(s, a, n, [doc], source_assessment={'status': 'scoped_source_audit_pass'})
    assert scoped['scoped_source_audit_passed'] is True
    assert scoped['strict']['scientific_status'] == 'unknown'
    assert 'independence_unconfirmed' not in scoped['strict']['reason']


def put(root, name, value, *, rows=False):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r) + '\n' for r in value) if rows else json.dumps(value))
    return ref(root, p)


def ref(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root):
    run = root / 'run'
    src = put(root, 'models/a.json', {'model': 'synthetic'})
    code = []
    names = {'scripts/' + name for name in ['score_recommendation_execution.py', 'recommendation_execution_metrics.py',
             'rank_recommendation_execution.py', 'run_frontier_comparison.py', 'frontier_external_models.py']}
    for name in sorted(names | m.EXECUTION_IMPLEMENTATION_REQUIRED):
        code.append(put(root, name, {'code': name}))
    binding = {'execution_revision': m.EXECUTION_REVISION,
               'implementation_sha256': {r['path']: r['sha256'] for r in code if r['path'] in m.EXECUTION_IMPLEMENTATION_REQUIRED}}
    sealed = put(root, 'clinical/sealed/case.json', {'case_id': 'case', 'endpoints': [{'endpoint': 'Cmax', 'value': 20, 'unit': 'ng/mL'}]})
    s = scenario()
    s['sealed_references'] = [sealed]
    audits = [{'scenario_id': s['scenario_id'], 'candidate_id': cid, 'identity_match': True, 'family_id': cid,
               'source_asset': {'path': src['path'], 'actual_sha256': src['sha256'], 'readable': True}} for cid in ['a', 'b']]
    inputs = [put(root, 'run/inputs/cohort.json', {'source_files': {}}),
              put(root, 'run/inputs/scenarios.jsonl', [s], rows=True),
              put(root, 'run/inputs/candidate_audit.jsonl', audits, rows=True)]
    inputs += [put(root, 'run/rankings/inputs/queries.jsonl', [{'query_id': s['scenario_id']}], rows=True),
               put(root, 'run/rankings/inputs/candidates.jsonl', [{'candidate_id': c} for c in ['a', 'b']], rows=True)]
    manifest = {'source_files': {}, 'outputs': {Path(r['path']).name: {'sha256': r['sha256']} for r in inputs[-2:]}}
    inputs.append(put(root, 'run/rankings/inputs/manifest.json', manifest))
    jobs, lock_jobs = [], []
    for i, a in enumerate(audits):
        cid, jid = a['candidate_id'], 'job' + str(i)
        base = 'run/native/' + jid + '/'
        compatibility = {'strict_target_match': False, 'stratum': 'reference_subject_approximation'}
        plan = {**binding, 'scenario_sha256': m.digest(s), 'candidate_sha256': m.digest(a), 'source_path': src['path'],
                'source_sha256': src['sha256'], 'molecular_weight_g_per_mol': 200., 'compatibility': compatibility}
        plan_ref = put(root, base + 'execution_plan.json', plan)
        r = {**binding, 'scenario_id': s['scenario_id'], 'candidate_id': cid, 'status': 'blocked',
             'prediction_locked': False, 'execution_plan_sha256': plan_ref['sha256'],
             'sealed_observations_accessed': False, 'fitting_performed': False, 'reasons': ['synthetic_blocker']}
        if i == 0:
            p = prediction()
            p.update(binding)
            p['compatibility'] = compatibility
            pr = put(root, base + 'prediction.json', p)
            pl = put(root, base + 'prediction_lock.json', {
                **binding,
                **{k: plan[k] for k in ['scenario_sha256', 'candidate_sha256', 'source_sha256']},
                'files': {pr['path']: pr['sha256'], plan_ref['path']: plan_ref['sha256']},
                'sealed_observations_accessed': False, 'execution_verified': True})
            r.update(status='prediction_locked', prediction_locked=True, prediction_file=pr['path'], prediction_sha256=pr['sha256'],
                     prediction_lock=pl['path'], prediction_lock_sha256=pl['sha256'], source=src,
                     compatibility=compatibility, strict_target_match=False, predicted_endpoints=p['endpoints'])
        put(root, base + 'result.json', r)
        jobs.append({'job_id': jid, 'scenario_id': s['scenario_id'], 'candidate_id': cid, 'plan': plan})
        lock_jobs.append({'job_id': jid, 'scenario_id': s['scenario_id'], 'candidate_id': cid,
                          'artifacts': [ref(root, f) for f in (root / base).iterdir()]})
    inputs.append(put(root, 'run/inputs/execution_plan.json', {'jobs': jobs}))
    protocol = {'schema_version': 'pbpk-recommendation-execution-protocol-v1', 'execution_revision': m.EXECUTION_REVISION,
                'inputs': inputs, 'code': code,
                'methods': ['method'], 'cases': 1, 'candidates': 2, 'complete_scope_audit_pairs': 2,
                'planned_candidate_executions': 2, 'outcome_access': {'sealed_values_read_by_preparation': False,
                        'all_predictions_and_rankings_locked_before_scoring': True}}
    protocol_ref = put(root, 'run/protocol.json', protocol)
    put(root, 'run/native/predictions_lock.json', {'protocol': protocol_ref, 'target_observations_read': False,
                                                  'planned_jobs': 2, 'jobs': lock_jobs})
    index = put(root, 'run/rankings/results/method/index.json', {'query_ids': [s['scenario_id']], 'candidate_ids': ['a', 'b'], 'variants': ['original']})
    status = put(root, 'run/rankings/results/method/status.json', {'state': 'complete',
        'input_sha256': {r['path']: r['sha256'] for r in inputs if '/rankings/inputs/' in r['path']},
        'clinical_wrapper_sha256': ref(root, root / 'scripts/rank_recommendation_execution.py')['sha256'],
        'implementation_sha256': {n: ref(root, root / 'scripts' / n)['sha256'] for n in ['run_frontier_comparison.py', 'frontier_external_models.py']}})
    np.save(run / 'rankings/results/method/scores.npy', np.array([[[1., 1.]]]))
    scores = ref(root, run / 'rankings/results/method/scores.npy')
    put(root, 'run/rankings/ranking_lock.json', {'state': 'complete', 'scientific_outcomes_read': False,
         'query_ids': [s['scenario_id']], 'candidate_ids': ['a', 'b'], 'manifest': ref(root, run / 'rankings/inputs/manifest.json'),
         'methods': {'method': {**scores, 'original_scores': scores, 'index': index, 'status': status}}})
    return run


def test_full_synthetic_barrier_and_expected_tie_denominators(tmp_path):
    run = fixture(tmp_path)
    gate = m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)
    assert gate['files'].opened_outcomes == {}  # Hashing outcomes was also forbidden.
    result = m.score_run(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)
    assert result['summary']['all_planned_native_jobs_accounted_for'] == 2
    assert result['summary']['direct_validated_pairs'] == 0
    diag = next(s for s in result['selection'] if s['stratum'] == 'reference_diagnostic')
    assert diag['by_k']['1']['evaluable_any_endpoint_coverage'] == .5
    assert diag['by_k']['1']['exact_recall'] is None
    assert diag['endpoints']['Cmax_reference_vs_group']['known_pool_conditional_regret'] is None
    strict = next(s for s in result['selection'] if s['stratum'] == 'strict')
    assert strict['endpoints']['Cmax_reference_vs_group']['top1_evaluable_probability'] == 0
    assert (run / 'results.json').exists() and (run / 'results.csv').exists()


@pytest.mark.parametrize('tamper', ['missing_native', 'missing_ranking', 'changed_plan', 'changed_index', 'changed_prediction', 'changed_source', 'changed_code'])
def test_any_incomplete_or_changed_unit_blocks_before_sealed_read(tmp_path, monkeypatch, tamper):
    run = fixture(tmp_path)
    target = {'missing_native': run / 'native/job1/result.json',
              'missing_ranking': run / 'rankings/results/method/status.json',
              'changed_plan': run / 'native/job0/execution_plan.json',
              'changed_index': run / 'rankings/results/method/index.json',
              'changed_prediction': run / 'native/job0/prediction.json',
              'changed_source': tmp_path / 'models/a.json',
              'changed_code': tmp_path / 'scripts/score_recommendation_execution.py'}[tamper]
    if tamper.startswith('missing'):
        target.unlink()
    else:
        target.write_text('{}')
    def forbidden(*args, **kwargs):
        pytest.fail('A sealed read occurred before full validation')
    monkeypatch.setattr(m.Files, 'outcome', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.score_run(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)
    assert not (run / 'results.json').exists()


def reseal_synthetic_native(root, run):
    """Refresh outer hashes so revision tests exercise semantics, not stale SHA."""
    plans = json.loads((run / 'inputs/execution_plan.json').read_text())
    native_lock = json.loads((run / 'native/predictions_lock.json').read_text())
    for job in plans['jobs']:
        base = run / 'native' / job['job_id']
        job['plan'] = json.loads((base / 'execution_plan.json').read_text())
        result = json.loads((base / 'result.json').read_text())
        result['execution_plan_sha256'] = ref(root, base / 'execution_plan.json')['sha256']
        if result.get('prediction_locked'):
            lock = json.loads((base / 'prediction_lock.json').read_text())
            lock['files'] = {str(p.relative_to(root)): ref(root, p)['sha256']
                             for p in [base / 'execution_plan.json', base / 'prediction.json']}
            result['prediction_sha256'] = ref(root, base / 'prediction.json')['sha256']
            result['prediction_lock_sha256'] = put(root, str((base / 'prediction_lock.json').relative_to(root)), lock)['sha256']
        put(root, str((base / 'result.json').relative_to(root)), result)
        locked = next(row for row in native_lock['jobs'] if row['job_id'] == job['job_id'])
        locked['artifacts'] = [ref(root, p) for p in sorted(base.iterdir()) if p.is_file()]
    plan_ref = put(root, 'run/inputs/execution_plan.json', plans)
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol['inputs'] = [plan_ref if r['path'] == plan_ref['path'] else r for r in protocol['inputs']]
    native_lock['protocol'] = put(root, 'run/protocol.json', protocol)
    put(root, 'run/native/predictions_lock.json', native_lock)


@pytest.mark.parametrize('artifact', ['execution_plan.json', 'result.json', 'prediction_lock.json', 'prediction.json'])
@pytest.mark.parametrize('field,value', [('execution_revision', None), ('execution_revision', 'initial_attempt'),
                                        ('implementation_sha256', {})])
def test_old_native_artifact_cannot_be_resigned_as_new_revision(tmp_path, monkeypatch, artifact, field, value):
    run = fixture(tmp_path)
    target = run / 'native/job0' / artifact
    document = json.loads(target.read_text())
    if value is None:
        document.pop(field)
    else:
        document[field] = value
    put(tmp_path, str(target.relative_to(tmp_path)), document)
    reseal_synthetic_native(tmp_path, run)
    monkeypatch.setattr(m.Files, 'outcome', lambda *a, **kw: pytest.fail('Outcome opened before revision validation'))
    with pytest.raises(ValueError, match='execution revision|implementation binding'):
        m.score_run(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)
    assert not (run / 'results.json').exists()


def test_blocked_results_also_need_frozen_execution_revision(tmp_path, monkeypatch):
    run = fixture(tmp_path)
    document = json.loads((run / 'native/job1/result.json').read_text())
    document.pop('implementation_sha256')
    put(tmp_path, 'run/native/job1/result.json', document)
    reseal_synthetic_native(tmp_path, run)
    monkeypatch.setattr(m.Files, 'outcome', lambda *a, **kw: pytest.fail('Outcome opened before blocked result validation'))
    with pytest.raises(ValueError, match='Native result execution implementation binding'):
        m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)


def test_all_matching_old_implementation_maps_fail_against_new_protocol_code(tmp_path, monkeypatch):
    run = fixture(tmp_path)
    changed = put(tmp_path, 'scripts/recommendation_candidate_executor.py', {'code': 'new-engineering-implementation'})
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol['code'] = [changed if r['path'] == changed['path'] else r for r in protocol['code']]
    put(tmp_path, 'run/protocol.json', protocol)
    reseal_synthetic_native(tmp_path, run)
    monkeypatch.setattr(m.Files, 'outcome', lambda *a, **kw: pytest.fail('Outcome opened for stale implementation'))
    with pytest.raises(ValueError, match='not bound to current protocol code'):
        m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)


def test_matching_artifacts_cannot_omit_required_dependency(tmp_path):
    run = fixture(tmp_path)
    for name in ['execution_plan.json', 'result.json', 'prediction_lock.json', 'prediction.json']:
        document = json.loads((run / 'native/job0' / name).read_text())
        document['implementation_sha256'].pop('src/pbpk_kb/blueprints/source_output_grid.py')
        put(tmp_path, 'run/native/job0/' + name, document)
    reseal_synthetic_native(tmp_path, run)
    with pytest.raises(ValueError, match='binding is incomplete'):
        m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)


def test_individually_frozen_but_different_artifact_implementation_maps_fail(tmp_path):
    run = fixture(tmp_path)
    path = 'scripts/score_recommendation_execution.py'
    result = json.loads((run / 'native/job0/result.json').read_text())
    result['implementation_sha256'][path] = ref(tmp_path, tmp_path / path)['sha256']
    put(tmp_path, 'run/native/job0/result.json', result)
    reseal_synthetic_native(tmp_path, run)
    with pytest.raises(ValueError, match='differs from frozen native plan'):
        m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)


def test_protocol_must_declare_the_current_execution_revision(tmp_path):
    run = fixture(tmp_path)
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol.pop('execution_revision')
    put(tmp_path, 'run/protocol.json', protocol)
    reseal_synthetic_native(tmp_path, run)
    with pytest.raises(ValueError, match='Protocol execution revision'):
        m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'], allow_missing_declared_tiers=True)


def test_renamed_sealed_path_is_still_protected_before_barrier(tmp_path):
    put(tmp_path, 'renamed.json', {'value': 123})
    f = m.Files(tmp_path)
    f.outcome_paths.add(tmp_path / 'renamed.json')
    with pytest.raises(ValueError, match='Sealed'):
        f.sha('renamed.json')


def test_missing_policy_freeze_is_not_silently_skipped(tmp_path, monkeypatch):
    run = fixture(tmp_path)
    monkeypatch.setattr(m.Files, 'outcome', lambda *a, **k: pytest.fail('Sealed outcome opened'))
    with pytest.raises(ValueError, match='every required input'):
        m.score_run(run, project_root=tmp_path, expected_methods=['method'])
    with pytest.raises(ValueError, match='Production cannot omit'):
        m.verify_barrier(m.ROOT, allow_missing_declared_tiers=True)


def test_existing_policy_actions_preserved_and_threshold_signals_are_not_truth():
    def row(cid, same, action, native, fold=None):
        return {'scenario_id': 's', 'case_id': 'case', 'candidate_id': cid, 'compound': 'drug',
                'identity_match': same, 'native_status': native,
                'existing_policy_decision': {'declared_use': action, 'reasons': ['original_reason'],
                                             'original_policy_class': 3 if action.startswith('not_') else None},
                'reference_diagnostic': {'endpoints': {'Cmax': {'predicted': fold, 'observed': 1}} if fold else {}}}
    data = [row('a', True, 'not_cleared_by_existing_rules', 'prediction_locked', 1.2),
            row('b', True, 'reference_only', 'prediction_locked', 1.8),
            row('c', True, 'unknown', 'blocked'),
            row('d', False, 'reference_only', 'not_planned'),
            row('e', True, 'direct_reuse', 'prediction_locked', 1.1)]
    report = m.audit_declared_policy(data, {'known_representation_limits': ['original limitation']})
    assert report['action_counts_by_parent_scope']['same_parent']['unknown'] == 1
    assert report['same_parent_action_by_native_status']['unknown']['blocked'] == 1
    assert report['action_counts_by_parent_scope']['other_parent']['reference_only'] == 1
    assert report['signal_candidate_pairs'] == 2
    signals = report['not_direct_cleared_but_numerically_within_threshold']
    assert signals[0]['within_1_5_fold'] is True and signals[1]['within_1_5_fold'] is False
    assert all(r['scientific_false_negative_established'] is False for r in signals)
    assert signals[0]['policy_reasons'] == ['original_reason']
    assert report['scientific_F1'] is None


@pytest.mark.parametrize('bad_binding', [None, 'candidate_record_sha256', 'query_graph_sha256', 'source_sha256'])
def test_frozen_declared_tier_full_index_and_bindings_before_reveal(tmp_path, monkeypatch, bad_binding):
    run = fixture(tmp_path)
    s = json.loads((run / 'inputs/scenarios.jsonl').read_text())
    source = ref(tmp_path, tmp_path / 'models/a.json')
    candidates = [{'candidate_id': cid, 'provenance': {'source_path': source['path'], 'source_sha256': source['sha256']}}
                  for cid in ['a', 'b']]
    catalog = tmp_path / 'catalog.jsonl'
    put(tmp_path, 'catalog.jsonl', candidates, rows=True)
    declared = [{'scenario_id': s['scenario_id'], 'candidate_id': c['candidate_id'],
                 'candidate_record_sha256': m.digest(c), 'source_sha256': source['sha256'],
                 'query_graph_sha256': 'fixed-graph', 'declared_use': 'unknown', 'reasons': []}
                for c in candidates]
    if bad_binding:
        declared[0][bad_binding] = 'changed'
    output = put(tmp_path, 'run/inputs/declared_tiers.jsonl', declared, rows=True)
    code = put(tmp_path, 'scripts/recommendation_execution_tiers.py', {'code': 'synthetic'})
    manifest = {'code': [code], 'output': output,
                'mapped_queries': [{'scenario_id': s['scenario_id'], 'scenario_sha256': m.digest(s), 'graph_sha256': 'fixed-graph'}]}
    mf = put(tmp_path, 'run/inputs/declared_tiers_manifest.json', manifest)
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol['inputs'] += [output, mf]
    protocol['code'].append(code)
    pr = put(tmp_path, 'run/protocol.json', protocol)
    native_lock = json.loads((run / 'native/predictions_lock.json').read_text())
    native_lock['protocol'] = pr
    put(tmp_path, 'run/native/predictions_lock.json', native_lock)
    calls = []
    def verify(*, check_original_sources):
        calls.append(check_original_sources)
        return {'manifest': manifest, 'rows': declared}
    monkeypatch.setitem(sys.modules, 'recommendation_execution_tiers', SimpleNamespace(
        ROOT=tmp_path, OUT=run, CATALOG=catalog, rows=lambda p: candidates, verify_declared_tiers=verify))
    monkeypatch.setattr(m.Files, 'outcome', lambda *a, **k: pytest.fail('Sealed outcome opened'))
    if bad_binding:
        with pytest.raises(ValueError, match='binding changed'):
            m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'])
    else:
        bundle = m.verify_barrier(run, project_root=tmp_path, expected_methods=['method'])
        assert len(bundle['declared_tiers']) == 2 and not bundle['files'].opened_outcomes
    assert calls == [True]
