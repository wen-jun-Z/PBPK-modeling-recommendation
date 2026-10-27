from copy import deepcopy
import math
from pathlib import Path

import pytest

from pbpk_kb.blueprints.source_output_grid import ensure_output_times


ROOT = Path(__file__).resolve().parents[2]


def interval(start, end, resolution=20.):
    return {'Parameters': [
        {'Name': 'Start time', 'Value': start, 'Unit': 'min'},
        {'Name': 'End time', 'Value': end, 'Unit': 'min'},
        {'Name': 'Resolution', 'Value': resolution, 'Unit': 'pts/h'}]}


@pytest.fixture
def snapshot():
    return {'Version': 80, 'Compounds': [{'Name': 'Fixture drug'}],
            'Protocols': [{'Name': 'IV fixture', 'ApplicationType': 'Intravenous',
                           'DosingInterval': 'Single', 'Parameters': [
                               {'Name': 'Start time', 'Value': 0., 'Unit': 'min'},
                               {'Name': 'InputDose', 'Value': 1., 'Unit': 'mg'},
                               {'Name': 'Infusion time', 'Value': 5., 'Unit': 'min'}]}],
            'Individuals': [{'Name': 'Source human', 'Parameters': [{'Path': 'EHC', 'Value': 1.}]}],
            'Simulations': [{'Name': 'Source', 'Individual': 'Source human',
                             'Compounds': [{'Name': 'Fixture drug', 'Protocol': {'Name': 'IV fixture'}}],
                             'Parameters': [], 'Solver': {'Name': 'unchanged'},
                             'OutputSchema': [interval(0., 120.), interval(120., 1440., 4.)]}]}


def test_infusion_and_sampling_times_preserve_original_points_and_all_model_content(snapshot):
    before = deepcopy(snapshot)
    result = ensure_output_times(snapshot, [4.8, 10.02, 19.98, 15.], ROOT)
    manifest = result['manifest']
    assert snapshot == before
    after = deepcopy(result['snapshot'])
    after['Simulations'][0]['OutputSchema'] = before['Simulations'][0]['OutputSchema']
    assert after == before
    assert manifest['original_point_count'] == 129
    assert manifest['estimated_point_count'] == 133
    assert manifest['required_time_points_min'] == [0., 4.8, 5., 10.02, 15., 19.98]
    expected = sorted(set(list(range(0, 121, 3)) + list(range(120, 1441, 15)) + [4.8, 5., 10.02, 19.98]))
    assert manifest['estimated_output_times_min'] == pytest.approx(expected, abs=1e-9)
    assert manifest['window_before_min'] == manifest['window_after_min'] == [0., 1440.]
    assert result['snapshot']['Simulations'][0]['OutputSchema'][-1] == before['Simulations'][0]['OutputSchema'][-1]
    assert manifest['native_time_coverage_verified'] is False
    assert manifest['solver_calls'] == 0
    assert len(manifest['source_format_evidence']) == 7


def test_only_real_snapshot_interval_parameters_are_generated(snapshot):
    result = ensure_output_times(snapshot, [4.8], ROOT)
    for item in result['snapshot']['Simulations'][0]['OutputSchema']:
        assert set(item) == {'Parameters'}
        assert {row['Name'] for row in item['Parameters']} == {'Start time', 'End time', 'Resolution'}
        assert all(row['Value'] >= 0 for row in item['Parameters'])


def test_no_change_when_source_already_covers_all_times(snapshot):
    snapshot['Protocols'][0]['Parameters'][-1]['Value'] = 6.
    result = ensure_output_times(snapshot, [0., 6., 15., 1440.], ROOT)
    assert result['snapshot'] == snapshot
    assert result['snapshot'] is not snapshot
    assert result['manifest']['source_schema_sha256'] == result['manifest']['output_schema_sha256']


def test_decimal_day_roundoff_has_bounded_explicit_integer_grid_registration(snapshot):
    row = snapshot['Simulations'][0]['OutputSchema'][0]['Parameters'][1]
    row.update(Value=.0833333333, Unit='day(s)')
    result = ensure_output_times(snapshot, [120.], ROOT)
    rounding = result['manifest']['near_integer_step_roundoff_registration']
    assert rounding[0]['registered_integer_steps'] == 40
    assert 1e-8 < rounding[0]['absolute_step_roundoff'] < 1e-7
    assert result['manifest']['original_sampling_points_preserved'] is True
    assert result['snapshot']['Compounds'] == snapshot['Compounds']


def test_final_simulation_timing_overrides_take_precedence(snapshot):
    prefix = 'Events|IV fixture|Application_1|ProtocolSchemaItem|'
    snapshot['Simulations'][0]['Parameters'] = [
        {'Path': prefix + 'Start time', 'Value': .5, 'Unit': 'min'},
        {'Path': prefix + 'Infusion time', 'Value': 120., 'Unit': 's'}]
    result = ensure_output_times(snapshot, [], ROOT)
    assert result['manifest']['administration_event_times_min'] == [.5, 2.5]
    assert result['manifest']['administration_events'] == [{
        'event_path': prefix[:-1], 'start_min': .5, 'infusion_end_min': 2.5}]


def test_repeated_oral_all_seven_starts_are_included_without_inspecting_dose(snapshot):
    item = {'ApplicationType': 'Oral', 'FormulationKey': 'Formulation', 'Parameters': [
        {'Name': 'Start time', 'Value': .17, 'Unit': 'h'},
        {'Name': 'InputDose', 'Value': 3., 'Unit': 'mg/kg'}]}
    snapshot['Protocols'] = [{'Name': 'Repeat', 'Schemas': [{'SchemaItems': [item], 'Parameters': [
        {'Name': 'Start time', 'Value': 0., 'Unit': 'h'},
        {'Name': 'NumberOfRepetitions', 'Value': 7.},
        {'Name': 'TimeBetweenRepetitions', 'Value': 24., 'Unit': 'h'}]}]}]
    sim = snapshot['Simulations'][0]
    sim['Compounds'][0]['Protocol'] = {'Name': 'Repeat', 'Formulations': [{'Name': 'Oral solution', 'Key': 'Formulation'}]}
    sim['OutputSchema'] = [interval(0., 10560., 4.)]
    result = ensure_output_times(snapshot, [8760., 8880., 9120., 9360.], ROOT)
    assert result['manifest']['administration_event_times_min'] == pytest.approx([10.2 + 1440.*n for n in range(7)])
    assert result['manifest']['estimated_point_count'] == 712
    assert len(result['manifest']['administration_events']) == 7


def test_simple_oral_implicit_formulation_key(snapshot):
    protocol = snapshot['Protocols'][0]
    protocol['ApplicationType'] = 'Oral'
    protocol['Parameters'] = protocol['Parameters'][:2]
    snapshot['Simulations'][0]['Compounds'][0]['Protocol']['Formulations'] = [{'Name': 'Tablet', 'Key': 'Formulation'}]
    result = ensure_output_times(snapshot, [], ROOT)
    assert result['manifest']['administration_events'] == [{
        'event_path': 'Events|IV fixture|Tablet|Application_1|ProtocolSchemaItem', 'start_min': 0.}]


def test_multischema_native_event_order_is_chronological_without_changing_source(snapshot):
    def schema(start, repetitions, period):
        return {'SchemaItems': [{'ApplicationType': 'Oral', 'FormulationKey': 'Formulation',
                                'Parameters': [{'Name': 'Start time', 'Value': 0., 'Unit': 'h'}]}],
                'Parameters': [{'Name': 'Start time', 'Value': start, 'Unit': 'h'},
                               {'Name': 'NumberOfRepetitions', 'Value': repetitions},
                               {'Name': 'TimeBetweenRepetitions', 'Value': period, 'Unit': 'h'}]}
    snapshot['Protocols'] = [{'Name': 'Repeat', 'Schemas': [schema(0., 7, 24.), schema(192., 3, 24.), schema(180., 1, 0.)]}]
    sim = snapshot['Simulations'][0]
    sim['Compounds'][0]['Protocol'] = {'Name': 'Repeat', 'Formulations': [{'Name': 'Oral solution', 'Key': 'Formulation'}]}
    sim['OutputSchema'] = [interval(0., 264. * 60., 20.)]
    before = deepcopy(snapshot)
    result = ensure_output_times(snapshot, [], ROOT, max_points=8192)
    assert snapshot == before == result['snapshot']
    events = result['manifest']['administration_events']
    assert [event['start_min'] for event in events] == [0., 1440., 2880., 4320., 5760., 7200., 8640., 10800., 11520., 12960., 14400.]
    assert events[7]['event_path'].endswith('|Application_8|ProtocolSchemaItem')
    assert result['manifest']['estimated_point_count'] == 5281


def test_window_extends_without_shifting_existing_times(snapshot):
    sim = snapshot['Simulations'][0]
    sim['OutputSchema'] = [interval(3., 120.)]
    result = ensure_output_times(snapshot, [180., 240.], ROOT)
    assert result['manifest']['window_before_min'] == [3., 120.]
    assert result['manifest']['window_after_min'] == [0., 240.]
    assert all(value in result['manifest']['estimated_output_times_min'] for value in [0., 3., 5., 120., 180., 240.])


def test_requested_gap_times_do_not_remove_original_intervals(snapshot):
    sim = snapshot['Simulations'][0]
    sim['OutputSchema'] = [interval(0., 30.), interval(90., 120.)]
    result = ensure_output_times(snapshot, [60.], ROOT)
    assert all(value in result['manifest']['estimated_output_times_min'] for value in [30., 60., 90.])
    assert result['snapshot']['Simulations'][0]['OutputSchema'][-1] == sim['OutputSchema'][-1]


def test_concentration_values_and_observation_metadata_cannot_choose_grid(snapshot):
    first = deepcopy(snapshot)
    first['ObservedData'] = [{'concentration': [1., 1000.], 'time_min': [7., 8.]}]
    other = deepcopy(snapshot)
    other['ObservedData'] = [{'concentration': [math.nan, math.inf], 'time_min': [45., 60.]}]
    left = ensure_output_times(first, [4.8], ROOT)
    right = ensure_output_times(other, [4.8], ROOT)
    assert left['manifest'] == right['manifest']
    assert left['snapshot']['Simulations'][0]['OutputSchema'] == right['snapshot']['Simulations'][0]['OutputSchema']
    assert left['manifest']['concentration_values_used'] is False


@pytest.mark.parametrize('times', [[True], [-1.], [math.nan], [math.inf], [None], [{'time_min': 4.8, 'concentration': 10.}], (4.8,)])
def test_invalid_or_observation_objects_as_times_fail(snapshot, times):
    with pytest.raises(ValueError):
        ensure_output_times(snapshot, times, ROOT)


def test_duplicate_requested_times_do_not_inflate_point_count(snapshot):
    result = ensure_output_times(snapshot, [4.8, 4.8, 4.800000000000001], ROOT)
    assert result['manifest']['required_times_min'] == [4.8]
    assert result['manifest']['estimated_point_count'] == 131


@pytest.mark.parametrize('mutation', ['fractional_count', 'zero_resolution', 'missing_end', 'unknown_unit', 'negative_time', 'formula'])
def test_unsupported_source_schema_fails_explicitly(snapshot, mutation):
    rows = snapshot['Simulations'][0]['OutputSchema'][0]['Parameters']
    if mutation == 'fractional_count':
        rows[1]['Value'] = 121.
    elif mutation == 'zero_resolution':
        rows[2]['Value'] = 0.
    elif mutation == 'missing_end':
        rows.pop(1)
    elif mutation == 'unknown_unit':
        rows[2]['Unit'] = 'banana'
    elif mutation == 'negative_time':
        rows[0]['Value'] = -1.
    elif mutation == 'formula':
        rows[2]['TableFormula'] = {}
    with pytest.raises(ValueError):
        ensure_output_times(snapshot, [], ROOT)


def test_4096_points_are_rejected_before_producing_a_snapshot(snapshot):
    snapshot['Simulations'][0]['OutputSchema'] = [interval(0., 4095., 60.)]
    with pytest.raises(ValueError, match='4096-point'):
        ensure_output_times(snapshot, [], ROOT)


def test_added_times_must_also_fit_the_limit(snapshot):
    snapshot['Protocols'][0]['Parameters'][-1]['Value'] = 6.
    snapshot['Simulations'][0]['OutputSchema'] = [interval(0., 4094., 60.)]
    assert ensure_output_times(snapshot, [], ROOT)['manifest']['estimated_point_count'] == 4095
    with pytest.raises(ValueError, match='4096-point'):
        ensure_output_times(snapshot, [.5], ROOT)


def test_explicit_bounded_limit_preserves_a_long_source_grid(snapshot):
    snapshot['Protocols'][0]['Parameters'][-1]['Value'] = 6.
    snapshot['Simulations'][0]['OutputSchema'] = [interval(0., 264. * 60., 20.)]
    with pytest.raises(ValueError, match='4096-point'):
        ensure_output_times(snapshot, [], ROOT)
    result = ensure_output_times(snapshot, [], ROOT, max_points=32768)
    assert result['snapshot'] == snapshot
    manifest = result['manifest']
    assert manifest['original_point_count'] == manifest['estimated_point_count'] == 5281
    assert manifest['point_limit_exclusive'] == manifest['absolute_point_limit_exclusive'] == 32768
    assert manifest['default_point_limit_exclusive'] == 4096
    assert manifest['larger_point_limit_explicitly_requested'] is True
    assert manifest['window_after_min'] == [0., 15840.]


@pytest.mark.parametrize('limit', [True, 32768.0, 32769, math.inf, None, 1, -1])
def test_point_limit_is_always_a_bounded_explicit_integer(snapshot, limit):
    with pytest.raises(ValueError, match='max_points'):
        ensure_output_times(snapshot, [], ROOT, max_points=limit)


def test_larger_limit_still_rejects_too_many_points(snapshot):
    snapshot['Simulations'][0]['OutputSchema'] = [interval(0., 32767., 60.)]
    with pytest.raises(ValueError, match='32768-point'):
        ensure_output_times(snapshot, [], ROOT, max_points=32768)


def test_unrelated_interval_metadata_is_preserved_on_split(snapshot):
    row = snapshot['Simulations'][0]['OutputSchema'][0]
    row['Description'] = 'Source interval'
    row['Parameters'][0]['ValueOrigin'] = {'Source': 'Publication'}
    result = ensure_output_times(snapshot, [4.8], ROOT)
    pieces = result['snapshot']['Simulations'][0]['OutputSchema'][:-1]
    assert all(piece['Description'] == 'Source interval' for piece in pieces)
    assert all('ValueOrigin' not in parameter for piece in pieces for parameter in piece['Parameters'])


def test_changed_pinned_source_is_not_silently_accepted(snapshot, tmp_path, monkeypatch):
    from pbpk_kb.blueprints import source_output_grid as grid
    folder = tmp_path / 'source'
    folder.mkdir()
    (folder / 'mapper.cs').write_text('changed source')
    monkeypatch.setattr(grid, '_SOURCE_BASE', 'source/')
    monkeypatch.setattr(grid, '_SOURCE_HASHES', {'mapper.cs': '0' * 64})
    with pytest.raises(ValueError, match='source changed'):
        ensure_output_times(snapshot, [], tmp_path)
