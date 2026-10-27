"""Source/budget/selection regressions with tiny fixtures; never invoke a solver."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_blueprint_development.py'
SPEC = importlib.util.spec_from_file_location('development_runner_under_test', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
MIB = 1024**2


def fixture_case():
    observations = [{'time_min': 1, 'concentration_umol_per_l': 1},
                    {'time_min': 2, 'concentration_umol_per_l': 2}]
    return {
        'purpose': 'historical_development_reconstruction',
        'calibration': {'role': 'calibration', 'observations': observations,
                        'checksum': hashlib.sha256(runner._canonical(observations).encode()).hexdigest()},
        'design_request': {'instance': {'output_selections': ['Plasma|Drug']}},
        'parameter_search': {
            'parameters': [
                {'parameter_id': 'total_cyp1a2_specific_clearance', 'unit': '1/min',
                 'initial_value': .4, 'lower_bound': 0., 'upper_bound': 1.,
                 'search_scale': 'linear_nonnegative', 'source': {'checksum': 'fixture'},
                 'runtime_mappings': [{'runtime_path': 'Drug-CYP1A2-test|Specific clearance', 'multiplier': 1.}]},
                {'parameter_id': 'specific_intestinal_permeability_transcellular', 'unit': 'dm/min',
                 'initial_value': .01, 'lower_bound': .001, 'upper_bound': .1,
                 'search_scale': 'log10', 'source': {'checksum': 'fixture'},
                 'runtime_mappings': [{'runtime_path': 'Drug|Specific intestinal permeability (transcellular)', 'multiplier': 1.}]},
            ],
            'clearance_starts': [{'total_cyp1a2_specific_clearance': x} for x in (.1, .5, .9)],
        },
        'source_audit': [{'path': 'fixture.json', 'sha256': 'fixture'}],
        'limitations': ['Synthetic unit-test input; no real observations or simulation.'],
    }


def test_only_missing_derivable_scale_annotations_are_compatible(monkeypatch, tmp_path):
    expected = fixture_case()
    saved = deepcopy(expected)
    for item in saved['parameter_search']['parameters']:
        item.pop('search_scale')
    original = deepcopy(saved)
    monkeypatch.setattr(runner, 'make_caffeine_development_case', lambda root: expected)
    checked, added = runner._validated_case(saved, tmp_path)
    assert checked == expected
    assert len(added) == 2
    assert saved == original


@pytest.mark.parametrize('change', [
    lambda c: c['parameter_search']['parameters'][0].update(upper_bound=2),
    lambda c: c['parameter_search']['parameters'][0].update(unit='dm/min'),
    lambda c: c['parameter_search']['parameters'][0].update(search_scale='log10'),
    lambda c: c['parameter_search']['parameters'][0]['runtime_mappings'][0].update(multiplier=.5),
    lambda c: c['parameter_search']['parameters'].reverse(),
    lambda c: c['source_audit'][0].update(sha256='changed'),
    lambda c: c['calibration']['observations'][0].update(concentration_umol_per_l=9),
    lambda c: c['calibration'].update(role='selection'),
    lambda c: c['design_request']['instance'].update(dose=99),
    lambda c: c.update(extra_unreviewed_input=True),
])
def test_case_changes_are_rejected_before_native_execution(change, monkeypatch, tmp_path):
    expected = fixture_case()
    saved = deepcopy(expected)
    change(saved)
    monkeypatch.setattr(runner, 'make_caffeine_development_case', lambda root: expected)
    with pytest.raises(ValueError, match='freshly source-verified'):
        runner._validated_case(saved, tmp_path)


def test_source_reaudit_failure_is_not_replaced_by_saved_pass_labels(monkeypatch, tmp_path):
    def source_changed(root):
        raise ValueError('Source checksum mismatch')
    monkeypatch.setattr(runner, 'make_caffeine_development_case', source_changed)
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        runner._validated_case(fixture_case(), tmp_path)


@pytest.mark.parametrize('drift', ['checksum', 'request'])
def test_baseline_snapshot_must_match_its_hash_and_current_request(drift, monkeypatch, tmp_path):
    baseline_dir = tmp_path / 'case/baseline'
    baseline_dir.mkdir(parents=True)
    snapshot_path = baseline_dir / 'snapshot.json'
    snapshot_path.write_text('{"dose": 10}')
    baseline = {'source_sha256': runner.sha(snapshot_path)}
    monkeypatch.setattr(runner, 'BlueprintDesignRequest', SimpleNamespace(model_validate=lambda x: x))
    monkeypatch.setattr(runner, 'design_blueprint', lambda *a, **kw: {
        'candidates': [{'status': 'draft_assembled'}],
        'human_model_binding': {'instance': {'snapshot': {'dose': 20 if drift == 'request' else 10}}},
    })
    if drift == 'checksum':
        snapshot_path.write_text('{"dose": 11}')
    with pytest.raises(ValueError, match='Baseline snapshot'):
        runner._validated_baseline_snapshot(tmp_path / 'case', {'design_request': {}}, baseline, tmp_path)


def test_artifact_cannot_escape_its_declared_directory(tmp_path):
    folder = tmp_path / 'baseline'
    folder.mkdir()
    outside = tmp_path / 'unrelated.pkml'
    outside.write_text('fixture')
    for invalid in ('../unrelated.pkml', str(outside)):
        with pytest.raises(ValueError, match='Artifact path'):
            runner._artifact_file(folder, invalid, tmp_path)
    (folder / 'linked.pkml').symlink_to(outside)
    with pytest.raises(ValueError, match='leaves its output'):
        runner._artifact_file(folder, 'linked.pkml', tmp_path)


def test_disk_budget_counts_actual_output_outside_case_and_not_inputs(tmp_path):
    inputs, output = tmp_path / 'case', tmp_path / 'separate_feedback'
    inputs.mkdir()
    output.mkdir()
    (inputs / 'baseline').write_bytes(b'old input')
    (output / 'new_data').write_bytes(b'123456')
    disk = runner._OutputBudget(output, 4 * MIB)
    assert disk.size() == 6
    assert disk.call_limit(8 * MIB) == 3 * MIB - 6


def test_remaining_budget_caps_batch_and_rebuild_and_stops_before_minimum(monkeypatch, tmp_path):
    disk = runner._OutputBudget(tmp_path, 10 * MIB)
    monkeypatch.setattr(disk, 'size', lambda: 7 * MIB)
    assert disk.call_limit(128 * MIB) == 2 * MIB
    assert disk.call_limit(256 * MIB) == 2 * MIB
    monkeypatch.setattr(disk, 'size', lambda: 8 * MIB + 1)
    with pytest.raises(RuntimeError, match='Insufficient remaining'):
        disk.call_limit(128 * MIB)


def test_metadata_write_is_checked_before_creating_file(tmp_path):
    disk = runner._OutputBudget(tmp_path, 64)
    with pytest.raises(RuntimeError, match='prevents artifact write'):
        disk.write(tmp_path / 'large.json', {'value': 'x' * 100})
    assert not (tmp_path / 'large.json').exists()
    disk.write(tmp_path / 'small.json', {'ok': True})
    assert disk.check() <= 64


def test_output_symlink_is_rejected(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'link').symlink_to(tmp_path / 'unopened')
    with pytest.raises(ValueError, match='symlink'):
        runner._OutputBudget(output, MIB).check()


def test_baseline_is_retained_when_search_is_worse_or_invalid():
    baseline = {'source': 'baseline', 'point': [.4, .01], 'score': {'valid': True, 'log_rmse': .2}}
    worse = {'source': 'search', 'point': [.1, .01], 'score': {'valid': True, 'log_rmse': .5}}
    invalid = {'source': 'search', 'point': [.9, .01], 'score': {'valid': False, 'log_rmse': None}}
    assert runner._best_candidate([worse, baseline, invalid]) is baseline
    worse['score']['log_rmse'] = .1
    assert runner._best_candidate([baseline, worse]) is worse


def test_no_valid_score_serializes_as_null_never_infinity():
    candidates = [{'point': [1], 'score': {'valid': True, 'log_rmse': float('inf')}},
                  {'point': [2], 'score': {'valid': False, 'log_rmse': None}}]
    assert runner._best_candidate(candidates) is None
    encoded = json.dumps({'best': runner._best_log_rmse(candidates)}, allow_nan=False)
    assert encoded == '{"best": null}'


@pytest.mark.parametrize('point', [[-1, .01], [.4, 1], [float('nan'), .01], [True, .01], [.4]])
def test_points_rejected_before_native_batch(point):
    with pytest.raises(ValueError, match='Parameter'):
        runner._validate_points([point], fixture_case()['parameter_search']['parameters'])


def test_mocked_runner_keeps_valid_baseline_after_all_invalid_searches(monkeypatch, tmp_path):
    """Exercise the whole runner control flow with synthetic curves and no native I/O."""
    case = fixture_case()
    snapshot = {'Simulations': [{'Name': 'unit_fixture'}]}
    case_dir = tmp_path / 'case'
    baseline_dir, probe_dir = case_dir / 'baseline', case_dir / 'mapping_probe'
    baseline_dir.mkdir(parents=True)
    probe_dir.mkdir()
    (case_dir / 'case.json').write_text(json.dumps(case))
    (baseline_dir / 'snapshot.json').write_text(json.dumps(snapshot))
    (baseline_dir / 'model.pkml').write_text('mock model, not executable')
    model_sha = runner.sha(baseline_dir / 'model.pkml')
    (baseline_dir / 'execution_manifest.json').write_text(json.dumps({
        'engineering_accepted': True, 'source_sha256': runner.sha(baseline_dir / 'snapshot.json'),
        'pkml_outputs': [{'path': 'model.pkml', 'sha256': model_sha}],
    }))
    native_specs = [{'path': m['runtime_path'], 'unit': p['unit']}
                    for p in case['parameter_search']['parameters'] for m in p['runtime_mappings']]
    (probe_dir / 'native_results.json').write_text(json.dumps({'runs': [{'values': [.4, .01], 'kind': 'baseline'}]}))
    (probe_dir / 'parameter_batch_manifest.json').write_text(json.dumps({
        'status': 'completed', 'native_results_sha256': runner.sha(probe_dir / 'native_results.json'),
        'pkml_sha256': model_sha, 'parameter_specs': native_specs, 'requested_vector_count': 1,
        'target_output': 'Plasma|Drug',
    }))
    for name in ('design_scoring.py', 'parameter_batch.py'):
        path = tmp_path / 'src/pbpk_kb/blueprints' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# checksum-only fixture')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'make_caffeine_development_case', lambda root: deepcopy(case))
    monkeypatch.setattr(runner, 'BlueprintDesignRequest', SimpleNamespace(model_validate=lambda x: x))
    monkeypatch.setattr(runner, 'design_blueprint', lambda *a, **kw: {
        'candidates': [{'status': 'draft_assembled'}], 'human_model_binding': {'instance': {'snapshot': snapshot}},
    })
    monkeypatch.setattr(runner, 'apply_parameters', lambda request, *a: deepcopy(request))
    calls = []

    def batch(**kwargs):
        calls.append(('batch', kwargs['max_output_bytes']))
        kwargs['output_dir'].mkdir()
        kind = 'invalid' if len(calls) <= 2 else 'diagnostic'
        return {'status': 'completed', 'runs': [{'kind': kind} for point in kwargs['values']]}

    def score(*, runs, **kwargs):
        return {'gpu_used': False, 'device': 'cpu', 'rows': [
            {'valid': False, 'log_rmse': None, 'predictions_umol_per_l': None}
            if row.get('kind') == 'invalid' else
            {'valid': True, 'log_rmse': .2, 'predictions_umol_per_l': [1., 2.]}
            for row in runs]}

    def rebuild(**kwargs):
        calls.append(('rebuild', kwargs['max_output_bytes']))
        kwargs['output_dir'].mkdir()
        (kwargs['output_dir'] / 'curve.csv').write_text('Time [min],Plasma|Drug [µmol/l]\n1,1\n2,2\n')
        return {'engineering_accepted': True, 'curve_outputs': [{'path': 'curve.csv'}]}

    monkeypatch.setattr(runner, 'run_parameter_batch', batch)
    monkeypatch.setattr(runner, 'score_calibration_curves', score)
    monkeypatch.setattr(runner, 'execute_snapshot', rebuild)
    output = tmp_path / 'outside_case_feedback'
    assert runner.main(['--case-directory', str(case_dir), '--output', str(output),
                        '--device', 'cpu', '--rounds', '1']) == 0
    fit = json.loads((output / 'fitted_parameters.json').read_text())
    progress = json.loads((output / 'progress.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    assert fit['values'] == [.4, .01]
    assert fit['origin'] == 'retained_development_initial_values'
    assert result['selected_candidate_source'] == 'baseline'
    assert result['fitted_rebuild_verified'] is True
    assert progress['status'] == 'completed'
    assert progress['search_has_valid_candidate'] is False
    assert progress['best_calibration_log_rmse'] == .2
    assert calls[-1][0] == 'rebuild'
    assert all(MIB <= limit <= (256 if kind == 'rebuild' else 128) * MIB for kind, limit in calls)
