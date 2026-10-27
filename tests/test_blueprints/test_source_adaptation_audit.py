"""Audit-contract tests; the native runtime is mocked and no ODE is executed."""
from copy import deepcopy
import hashlib
import json

import pytest

from pbpk_kb.blueprints import source_adaptation_audit as audit


PREFIX = 'Events|Study protocol|Application_1|ProtocolSchemaItem|'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readback(request):
    checks, events = [], []
    for row in request['expectations']:
        factor = {'mg': 1e-6, 'min': 1, '1/min': 1}[row['unit']]
        unit = 'kg' if row['unit'] == 'mg' else row['unit']
        checks.append({'check_id': row['check_id'], 'path': row['path'],
                       'expected_value': row['value'], 'expected_unit': row['unit'],
                       'status': 'evaluated', 'native_value': row['value'] * factor,
                       'expected_native_value': row['value'] * factor,
                       'native_unit': unit, 'native_dimension': 'Mass' if unit == 'kg' else 'Time'})
        if row['path'].startswith(PREFIX):
            events.append({'path': row['path'], 'event_path': PREFIX[:-1],
                           'field': row['path'].split('|')[-1],
                           'status': 'evaluated', 'value': row['value'] * factor, 'unit': unit})
    return {'adapter_version': audit.AUDIT_VERSION, 'pkml_sha256': request['pkml_sha256'],
            'solver_calls': 0, 'model_modified': False, 'parameter_checks': checks,
            'events': events, 'inventory': {'parameter_count': 50, 'container_count': 10,
                'organism_parameter_count': 40, 'organism_container_count': 8}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    native = root / 'execution'
    native.mkdir(parents=True)
    snapshot = {'Version': 80, 'Compounds': [{'Name': 'Fixture'}],
                'Individuals': [{'Name': 'Human', 'Parameters': []}],
                'Simulations': [{'Name': 'Study', 'Individual': 'Human', 'Parameters': [
                    {'Path': 'Fixture-CYP-source|kcat', 'Value': 2., 'Unit': '1/min'}]}]}
    snapshot_path = native / 'snapshot.json'
    snapshot_path.write_text(json.dumps(snapshot))
    pkml = native / 'fixture.pkml'
    pkml.write_text('<synthetic fixture, never loaded>')
    manifest = {'engineering_accepted': True, 'source_unchanged': True,
                'source_sha256': _sha(snapshot_path),
                'manifest_path': str(native / 'execution_manifest.json'),
                'pkml_outputs': [{'path': pkml.name, 'sha256': _sha(pkml)}]}
    (native / 'execution_manifest.json').write_text(json.dumps(manifest))
    settings = {'snapshot': snapshot, 'execution_manifest': manifest,
                'output_dir': root / 'audit', 'project_root': root,
                'expectations': [{'path': PREFIX + 'Dose', 'value': 10., 'unit': 'mg'},
                    {'path': PREFIX + 'Start time', 'value': 0., 'unit': 'min'},
                    {'path': PREFIX + 'Infusion time', 'value': 5., 'unit': 'min'}]}
    state = {'calls': [], 'change': None, 'mutation': None, 'failure': False}

    def worker(**kwargs):
        state['calls'].append(kwargs)
        path = settings['output_dir'] / 'request.json'
        request = json.loads(path.read_text())
        value = _readback(request)
        value['request_sha256'] = _sha(path)
        if state['change']:
            state['change'](value)
        if state['mutation']:
            state['mutation']()
        if state['failure']:
            return {'status': 'failed', 'stop_reason': 'fixture timeout'}
        (settings['output_dir'] / 'native_readback.json').write_text(json.dumps(value))
        return {'status': 'completed'}
    monkeypatch.setattr(audit, '_run_bounded_r', worker)
    return settings, state


def test_success_checks_real_converted_values_source_override_and_complete_events(setup):
    settings, state = setup
    before = deepcopy(settings['snapshot'])
    result = audit.audit_execution(**settings)
    assert result['status'] == 'passed' and result['accepted']
    assert len(result['parameter_checks']) == 4
    dose = result['parameter_checks'][0]
    assert dose['value'] == 10 and dose['native_value'] == pytest.approx(1e-5)
    assert result['parameter_checks'][-1]['origin'] == 'simulation_override'
    assert result['event_count'] == 1
    assert result['scientific_accepted'] is False and result['solver_calls'] == 0
    assert state['calls'][0]['timeout_seconds'] == 180
    assert state['calls'][0]['cpu_threads'] == 2
    assert state['calls'][0]['max_output_bytes'] == 64 * 1024**2
    assert settings['snapshot'] == before
    assert json.loads((settings['output_dir'] / 'audit.json').read_text()) == result


def test_unused_drug_definitions_are_preserved_and_do_not_invent_dosing(setup):
    settings, _ = setup
    snapshot = settings['snapshot']
    snapshot['Compounds'].append({'Name': 'Unused drug', 'Parameters': [{'Name': 'Retain', 'Value': 7}]})
    snapshot['Simulations'][0]['Compounds'] = [{'Name': 'Fixture'}]
    before = deepcopy(snapshot)
    rows = audit.build_expectations(snapshot, settings['expectations'])
    assert snapshot == before
    assert len(rows) == 4
    snapshot['Compounds'].append({'Name': 'Fixture'})
    with pytest.raises(ValueError, match='uniquely named'):
        audit.build_expectations(snapshot, settings['expectations'])


@pytest.mark.parametrize('change', [
    lambda r: r['parameter_checks'].pop(),
    lambda r: r['parameter_checks'][0].update(status='failed', error='Unit unavailable'),
    lambda r: r['parameter_checks'][0].update(native_value=1.),
    lambda r: r['parameter_checks'][0].update(native_value=None),
    lambda r: r['parameter_checks'][0].update(native_value=True),
    lambda r: r['parameter_checks'][0].update(native_value=float('inf')),
    lambda r: r['parameter_checks'][0].update(path=PREFIX + 'Wrong parameter'),
    lambda r: r['parameter_checks'][0].update(expected_unit='g'),
    lambda r: r['parameter_checks'][0].update(expected_value=100.),
    lambda r: r['parameter_checks'][0].update(check_id=1),
    lambda r: r.update(solver_calls=1),
    lambda r: r.update(solver_calls=False),
    lambda r: r.update(model_modified=True),
    lambda r: r.update(pkml_sha256='b' * 64),
    lambda r: r.update(request_sha256='b' * 64),
    lambda r: r.update(events=[]),
    lambda r: r['events'].append(deepcopy(r['events'][0])),
    lambda r: r['events'].pop(0),
    lambda r: r['events'][0].update(status='failed'),
    lambda r: r['inventory'].update(organism_parameter_count=0),
])
def test_corrupt_missing_or_partial_native_readback_fails_closed(setup, change):
    settings, state = setup
    state['change'] = change
    result = audit.audit_execution(**settings)
    assert not result['accepted'] and result['status'] == 'failed'
    assert result['reasons']


def test_additional_unrequested_native_event_does_not_pass(setup):
    settings, state = setup
    def extra(result):
        for field in ['Dose', 'Start time']:
            path = PREFIX.replace('Application_1', 'Application_2') + field
            result['events'].append({'path': path, 'event_path': path.rsplit('|', 1)[0],
                                    'field': field, 'status': 'evaluated', 'value': 1., 'unit': 'min'})
    state['change'] = extra
    result = audit.audit_execution(**settings)
    assert not result['accepted']
    assert any('count differs' in reason for reason in result['reasons'])
    assert any('lacks an explicit expectation' in reason for reason in result['reasons'])


def test_missing_event_expectation_does_not_pass_even_when_count_matches(setup):
    settings, state = setup
    settings['expectations'].pop()
    def extra(result):
        path = PREFIX + 'Infusion time'
        result['events'].append({'path': path, 'event_path': PREFIX[:-1], 'field': 'Infusion time',
                                'status': 'evaluated', 'value': 5., 'unit': 'min'})
    state['change'] = extra
    assert not audit.audit_execution(**settings)['accepted']


def test_changed_pkml_after_native_readback_is_detected(setup):
    settings, state = setup
    state['mutation'] = lambda: (settings['project_root'] / 'execution/fixture.pkml').write_text('changed')
    result = audit.audit_execution(**settings)
    assert not result['accepted'] and 'changed' in str(result['reasons'])


def test_snapshot_and_manifest_binding_are_checked_before_worker(setup):
    settings, state = setup
    settings['snapshot']['Simulations'][0]['Parameters'][0]['Value'] = 5.
    with pytest.raises(ValueError, match='not the snapshot'):
        audit.audit_execution(**settings)
    assert not state['calls']


def test_native_failure_is_reported_without_passing(setup):
    settings, state = setup
    state['failure'] = True
    result = audit.audit_execution(**settings)
    assert not result['accepted'] and 'fixture timeout' in str(result['reasons'])


@pytest.mark.parametrize('bad', [None, True, float('nan'), '2'])
def test_invalid_expectations_never_start_native_worker(setup, bad):
    settings, state = setup
    settings['expectations'][0]['value'] = bad
    with pytest.raises(ValueError):
        audit.audit_execution(**settings)
    assert not state['calls']


def test_nonscalar_source_override_cannot_be_silently_skipped(setup):
    settings, state = setup
    settings['snapshot']['Simulations'][0]['Parameters'][0]['TableFormula'] = {}
    with pytest.raises(ValueError, match='nonscalar'):
        audit.audit_execution(**settings)
    assert not state['calls']


def test_reused_audit_output_is_never_overwritten(setup):
    settings, _ = setup
    audit.audit_execution(**settings)
    with pytest.raises(FileExistsError):
        audit.audit_execution(**settings)


def test_selected_individual_override_is_required_but_simulation_value_takes_precedence(setup):
    settings, _ = setup
    snapshot = settings['snapshot']
    snapshot['Individuals'][0]['Parameters'] = [
        {'Path': 'Organism|Liver|EHC continuous fraction', 'Value': 1.},
        {'Path': 'Fixture-CYP-source|kcat', 'Value': 9., 'Unit': '1/min'}]
    rows = audit.build_expectations(snapshot, settings['expectations'])
    assert len(rows) == 5
    assert next(row for row in rows if row['path'].startswith('Organism|'))['origin'] == 'selected_individual_override'
    assert [row['value'] for row in rows if row['path'] == 'Fixture-CYP-source|kcat'] == [2.]


def _administration_snapshot(*, repeated=False, oral=False):
    item = {'ApplicationType': 'Oral' if oral else 'Intravenous',
            'Parameters': [{'Name': 'InputDose', 'Value': 450., 'Unit': 'mg'},
                           {'Name': 'Start time', 'Value': 0., 'Unit': 'h'}]}
    if oral:
        item['FormulationKey'] = 'Formulation'
    else:
        item['Parameters'].append({'Name': 'Infusion time', 'Value': .5, 'Unit': 'h'})
    if repeated:
        protocol = {'Name': 'Study protocol', 'Schemas': [{'SchemaItems': [item], 'Parameters': [
            {'Name': 'Start time', 'Value': 0., 'Unit': 'h'},
            {'Name': 'NumberOfRepetitions', 'Value': 7.},
            {'Name': 'TimeBetweenRepetitions', 'Value': 24., 'Unit': 'h'}]}]}
    else:
        protocol = dict(item, Name='Study protocol', DosingInterval='Single')
    return {'Protocols': [protocol], 'Simulations': [{'Compounds': [{'Name': 'Parent', 'Protocol': {
        'Name': 'Study protocol', 'Formulations': [{'Name': 'Oral solution', 'Key': 'Formulation'}]}}]}]}


def test_default_administration_expectations_use_source_timing_and_exact_event_names():
    snapshot = _administration_snapshot()
    result = audit.derive_administration_expectations(snapshot)
    assert result['expected_event_count'] == 1
    assert result['expectations'] == [
        {'path': PREFIX + 'Dose', 'value': 450., 'unit': 'mg'},
        {'path': PREFIX + 'Start time', 'value': 0., 'unit': 'min'},
        {'path': PREFIX + 'Infusion time', 'value': 30., 'unit': 'min'}]


def test_default_administration_repetition_and_formulation_paths_are_source_based():
    result = audit.derive_administration_expectations(_administration_snapshot(repeated=True, oral=True))
    assert result['expected_event_count'] == 7
    assert [row['value'] for row in result['expectations'] if row['path'].endswith('|Start time')] == [1440.*i for i in range(7)]
    assert all('|Oral solution|Application_' in row['path'] for row in result['expectations'])


def test_default_simple_oral_formulation_key_can_be_implicit():
    snapshot = _administration_snapshot(oral=True)
    snapshot['Protocols'][0].pop('FormulationKey')
    result = audit.derive_administration_expectations(snapshot)
    assert result['expected_event_count'] == 1
    assert len(result['expectations']) == 2
    assert all('|Oral solution|Application_1|' in row['path'] for row in result['expectations'])


def test_default_administration_applies_final_simulation_infusion_override():
    snapshot = _administration_snapshot()
    snapshot['Simulations'][0]['Parameters'] = [{'Path': PREFIX + 'Infusion time', 'Value': 2., 'Unit': 'min'}]
    result = audit.derive_administration_expectations(snapshot)
    assert result['expectations'][-1]['value'] == 2.


def test_weight_normalized_dose_requires_explicit_binding():
    snapshot = _administration_snapshot()
    snapshot['Protocols'][0]['Parameters'][0]['Unit'] = 'mg/kg'
    with pytest.raises(ValueError, match='absolute mass'):
        audit.derive_administration_expectations(snapshot)
