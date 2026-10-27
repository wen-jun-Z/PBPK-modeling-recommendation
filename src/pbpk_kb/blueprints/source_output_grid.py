"""Add administration and requested times to PK-Sim's snapshot output schema.

PK-Sim 12.3.173 snapshots serialize output intervals, not the native R API's
individual time points. This module therefore splits intervals using only the
supported Start time, End time and Resolution parameters. It never inspects
observed concentrations or changes physiological, drug, or solver parameters.
The caller must still verify time coverage in the executed native output.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from .source_adaptation import _named, _parameters, _time


VERSION = 'source-output-grid-v1.2'
POINT_LIMIT_EXCLUSIVE = 4096
MAX_POINT_LIMIT_EXCLUSIVE = 32768
TIME_TOLERANCE_MIN = 1e-8
# Decimal snapshot serialization of 2 h as 0.0833333333 day produces
# 39.999999984 steps at 20 pts/h. Only bounded near-integer output-grid
# roundoff is accepted; arbitrary fractional point counts remain unsupported.
NEAR_INTEGER_STEP_TOLERANCE = 1e-7
_SOURCE_BASE = 'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/'
_SOURCE_HASHES = {
    'OutputSchema.cs': '5bf11d7aa9a8bc97a783b72b0460bb2db1a3b0e46ded25b025c534fe83ff7966',
    'OutputInterval.cs': '8e172d6498575174808397353bd6ec8400bedba18f09a3326656bb2ee7ff966e',
    'Mappers/OutputSchemaMapper.cs': '9c3ad63608d0219ee9e500ba82f4c3f50a6d9ea9b69b1639aa2a24f31795d0d7',
    'Mappers/OutputIntervalMapper.cs': '441b42c4982e4e290a77693681b015657d7cc4128cd388f044e7fdf8f16f337f',
    '../Mappers/ProtocolToSchemaItemsMapper.cs': 'c458506ac5f7525a7a609280cb61091216d64b5edc357a03a9f182468db4e2d3',
    '../Model/Schema.cs': '40f33ea83bb614cfce0d00c2e5795bd1e7fdf65deb7a95f2163235cdb7ef1856',
    '../Services/EventBuildingBlockCreator.cs': 'fec536abd9152e1e18ed97c0cbfa0e8bbfbe3089a4dde51471921ff424767aa9',
}
_RESOLUTION_TO_PER_MIN = {'pts/min': 1., 'pts/h': 1. / 60., 'pts/s': 60.}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('Output times and scalar parameters must be finite, nonnegative numbers')
    return float(value)


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _source_receipt(project_root):
    root = Path(project_root).resolve(strict=True)
    sources = []
    for name, expected in _SOURCE_HASHES.items():
        path = (root / (_SOURCE_BASE + name)).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Pinned output schema source must remain inside the project')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned PK-Sim output schema source changed: ' + name)
        sources.append({'path': str(path.relative_to(root)), 'sha256': actual})
    return sources


def _unique_times(values):
    result = []
    for value in sorted(values):
        if not result or value - result[-1] > TIME_TOLERANCE_MIN:
            result.append(value)
    return result


def _contains(times, value):
    index = bisect_left(times, value)
    return any(abs(times[item] - value) <= TIME_TOLERANCE_MIN
               for item in (index - 1, index) if 0 <= item < len(times))


def _interval(interval, point_limit=POINT_LIMIT_EXCLUSIVE):
    if not isinstance(interval, dict):
        raise ValueError('OutputSchema intervals must be objects')
    parameters = _parameters(interval)
    required = {'Start time', 'End time', 'Resolution'}
    if not required <= parameters.keys():
        raise ValueError('OutputSchema requires explicit Start time, End time and Resolution')
    if any('TableFormula' in parameters[name] for name in required):
        raise ValueError('Formula-defined output intervals cannot be silently approximated')
    start, end = _time(parameters['Start time']), _time(parameters['End time'])
    resolution = parameters['Resolution']
    unit = resolution.get('Unit')
    if unit not in _RESOLUTION_TO_PER_MIN:
        raise ValueError('Unsupported explicit output Resolution unit: ' + str(unit))
    rate = _number(resolution.get('Value')) * _RESOLUTION_TO_PER_MIN[unit]
    if end <= start or rate <= 0:
        raise ValueError('Output intervals require positive duration and resolution')
    steps = (end - start) * rate
    if not math.isfinite(steps) or steps >= point_limit - 1:
        raise ValueError(f'Output grid would reach or exceed the {point_limit}-point limit')
    integer_steps = round(steps)
    if integer_steps < 1 or abs(steps - integer_steps) > NEAR_INTEGER_STEP_TOLERANCE:
        raise ValueError('Nonintegral source interval point counts require native grid registration')
    points = [start + (end - start) * index / integer_steps for index in range(integer_steps + 1)]
    return start, end, points


def _scalar_time(parameter):
    if not isinstance(parameter, dict) or 'TableFormula' in parameter:
        raise ValueError('Source administration timing must be scalar')
    return _time(parameter)


def _source_applications(protocol):
    """Read existing schemas in native order without editing the regimen."""
    if protocol.get('ApplicationType'):
        if protocol.get('Schemas') or protocol.get('DosingInterval') != 'Single':
            raise ValueError('Simple repeated protocols require registered native timing')
        return [(protocol, _scalar_time(_parameters(protocol).get('Start time')))]
    schemas = protocol.get('Schemas')
    if not isinstance(schemas, list) or not 1 <= len(schemas) <= 32:
        raise ValueError('Source timing requires 1..32 explicit schemas')
    expanded = []
    for schema in schemas:
        parameters = _parameters(schema)
        repeat_parameter = parameters.get('NumberOfRepetitions')
        if not isinstance(repeat_parameter, dict) or 'TableFormula' in repeat_parameter:
            raise ValueError('Source repetition count must be scalar')
        repetitions = _number(repeat_parameter.get('Value'))
        if not repetitions.is_integer() or not 1 <= repetitions <= 1024:
            raise ValueError('Source repetition count must be an integer from 1 through 1024')
        start = _scalar_time(parameters.get('Start time'))
        interval = _scalar_time(parameters.get('TimeBetweenRepetitions'))
        if repetitions > 1 and interval <= 0:
            raise ValueError('Repeated administrations require a positive interval')
        items = schema.get('SchemaItems')
        if not isinstance(items, list) or not items or len(expanded) + int(repetitions) * len(items) > 1024:
            raise ValueError('Source timing expansion must contain 1..1024 administrations')
        for repetition in range(int(repetitions)):
            for item in items:
                item_start = _scalar_time(_parameters(item).get('Start time'))
                expanded.append((item, _number(start + repetition * interval + item_start)))
    # ProtocolToSchemaItemsMapper orders the expanded items by StartTime;
    # EventBuildingBlockCreator assigns Application_1..N in that order.
    return sorted(expanded, key=lambda pair: pair[1])


def _event_times(snapshot, simulation):
    from .source_adaptation import _one_administered_selection
    selected = _one_administered_selection(simulation).get('Protocol')
    if not isinstance(selected, dict):
        raise ValueError('A selected source protocol is required')
    protocol = _named(snapshot.get('Protocols'), selected.get('Name'), 'Protocols')
    applications = _source_applications(protocol)
    overrides = simulation.get('Parameters', [])
    if not isinstance(overrides, list) or any(not isinstance(row, dict) or not isinstance(row.get('Path'), str) for row in overrides):
        raise ValueError('Simulation overrides must declare exact paths')
    by_path = {row['Path']: row for row in overrides}
    if len(by_path) != len(overrides):
        raise ValueError('Duplicate simulation override paths are ambiguous')
    events = []
    for index, (item, start) in enumerate(applications, 1):
        parameters = _parameters(item)
        route = item.get('ApplicationType')
        if route not in {'Oral', 'Intravenous', 'IntravenousBolus'}:
            raise ValueError('Unsupported administration route for output time registration')
        parts = ['Events', protocol['Name']]
        if route == 'Oral':
            key = item.get('FormulationKey', 'Formulation' if 'Schemas' not in protocol else None)
            formulations = selected.get('Formulations', [])
            matching = [row for row in formulations if isinstance(row, dict) and row.get('Key') == key]
            if len(matching) != 1 or not isinstance(matching[0].get('Name'), str):
                raise ValueError('Oral administration formulation is missing or ambiguous')
            parts.append(matching[0]['Name'])
        path = '|'.join(parts + [f'Application_{index}', 'ProtocolSchemaItem'])
        if path + '|Start time' in by_path:
            row = by_path[path + '|Start time']
            if 'TableFormula' in row:
                raise ValueError('Formula-defined administration times are unsupported')
            start = _time(row)
        event = {'event_path': path, 'start_min': start}
        if route == 'Intravenous':
            duration = by_path.get(path + '|Infusion time', parameters.get('Infusion time'))
            if not isinstance(duration, dict) or 'TableFormula' in duration:
                raise ValueError('IV infusion requires an explicit scalar duration')
            length = _time(duration)
            if length <= 0:
                raise ValueError('IV infusion duration must be positive')
            event['infusion_end_min'] = start + length
        events.append(event)
    return events


def _piece(template, start, end, steps):
    result = deepcopy(template) if template is not None else {'Parameters': [
        {'Name': 'Start time'}, {'Name': 'End time'}, {'Name': 'Resolution'}]}
    values = {'Start time': (start, 'min'), 'End time': (end, 'min'),
              'Resolution': (60. * steps / (end - start), 'pts/h')}
    for row in result['Parameters']:
        if row['Name'] in values:
            row['Value'], row['Unit'] = values[row['Name']]
            row.pop('ValueOrigin', None)
    return result


def _pieces(template, points):
    """Combine consecutive equal steps without dropping a requested boundary."""
    if len(points) < 2:
        raise ValueError('Output interval requires two distinct boundaries')
    result, begin, previous = [], 0, points[1] - points[0]
    for index in range(2, len(points)):
        step = points[index] - points[index - 1]
        if not math.isclose(step, previous, rel_tol=1e-10, abs_tol=1e-10):
            result.append(_piece(template, points[begin], points[index - 1], index - 1 - begin))
            begin = index - 1
        previous = step
    result.append(_piece(template, points[begin], points[-1], len(points) - 1 - begin))
    return result


def ensure_output_times(snapshot, required_times_min: list[float], project_root,
                        *, max_points: int = POINT_LIMIT_EXCLUSIVE):
    """Return a copy and an audit manifest; execute no native model or solver.

    ``required_times_min`` must contain only sampling times, never observation
    objects or concentrations. Original output points and intervals are kept
    wherever possible. Times outside the old window extend it without moving
    any existing time or administration. Unsupported timing fails explicitly.
    ``max_points`` is an exclusive limit, default 4096, capped at 32768. A
    larger limit must be chosen before execution; it never reduces source
    resolution or silently retries under a looser limit.
    """
    if type(max_points) is not int or not 2 <= max_points <= MAX_POINT_LIMIT_EXCLUSIVE:
        raise ValueError('max_points must be an integer exclusive limit from 2 through 32768')
    sources = _source_receipt(project_root)
    if not isinstance(snapshot, dict) or type(snapshot.get('Version')) is not int or snapshot['Version'] != 80:
        raise ValueError('A Version=80 source snapshot is required')
    simulations = snapshot.get('Simulations')
    if not isinstance(simulations, list) or len(simulations) != 1 or not isinstance(simulations[0], dict):
        raise ValueError('Exactly one selected simulation is required')
    if not isinstance(required_times_min, list) or len(required_times_min) >= max_points:
        raise ValueError(f'Supply fewer than {max_points} explicit sampling times')
    requested = _unique_times([_number(value) for value in required_times_min])
    simulation = simulations[0]
    schema = simulation.get('OutputSchema')
    if not isinstance(schema, list) or not schema or len(schema) >= max_points:
        raise ValueError('An explicit, bounded source OutputSchema is required')
    parsed, enumerated_count, roundoff = [], 0, []
    # Bound enumeration even when the source redundantly overlaps intervals.
    for index, interval in enumerate(schema):
        part = _interval(interval, max_points)
        parameters = _parameters(interval)
        original_steps = (part[1] - part[0]) * parameters['Resolution']['Value'] * _RESOLUTION_TO_PER_MIN[parameters['Resolution']['Unit']]
        if abs(original_steps - round(original_steps)) > 1e-10:
            roundoff.append({'interval_index': index, 'source_step_count': original_steps,
                             'registered_integer_steps': round(original_steps),
                             'absolute_step_roundoff': abs(original_steps - round(original_steps)),
                             'maximum_allowed_step_roundoff': NEAR_INTEGER_STEP_TOLERANCE})
        enumerated_count += len(part[2])
        if enumerated_count > 4 * max_points:
            raise ValueError('Source output interval enumeration exceeds its safety bound')
        parsed.append(part)
    original = _unique_times([point for _, _, points in parsed for point in points])
    events = _event_times(snapshot, simulation)
    administration = _unique_times([event[key] for event in events
                                     for key in ('start_min', 'infusion_end_min') if key in event])
    required = _unique_times(requested + administration)
    desired = _unique_times(original + required)
    if len(desired) >= max_points:
        raise ValueError(f'Output grid would reach or exceed the {max_points}-point limit')
    new_schema = []
    corrected_indices = {row['interval_index'] for row in roundoff}
    for index, (interval, (start, end, points)) in enumerate(zip(schema, parsed, strict=True)):
        additions = [value for value in required if start <= value <= end and not _contains(points, value)]
        new_schema.extend(_pieces(interval, _unique_times(points + additions))
                          if additions or index in corrected_indices else [deepcopy(interval)])
    # Add intervals only in formerly unsampled gaps / window extensions. Their
    # endpoints are requested times or existing boundaries, without new values.
    spans = sorted((start, end) for start, end, _ in parsed)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + TIME_TOLERANCE_MIN:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(desired[0], merged[0][0])]
    gaps.extend((left[1], right[0]) for left, right in zip(merged, merged[1:]))
    gaps.append((merged[-1][1], desired[-1]))
    for start, end in gaps:
        points = _unique_times([start, end] + [value for value in required if start < value < end])
        if end - start > TIME_TOLERANCE_MIN and any(start < value < end for value in required):
            new_schema.extend(_pieces(None, points))
        elif end - start > TIME_TOLERANCE_MIN and (start < original[0] or end > original[-1]):
            new_schema.extend(_pieces(None, points))
    new_schema.sort(key=lambda row: _time(_parameters(row)['Start time']))
    emitted = _unique_times([point for interval in new_schema for point in _interval(interval, max_points)[2]])
    if len(emitted) >= max_points or any(not _contains(emitted, value) for value in original + required):
        raise ValueError('Generated output schema does not preserve the bounded required time grid')
    result = deepcopy(snapshot)
    result['Simulations'][0]['OutputSchema'] = new_schema
    manifest = {
        'version': VERSION, 'concentration_values_used': False, 'solver_calls': 0,
        'native_time_coverage_verified': False, 'time_tolerance_min': TIME_TOLERANCE_MIN,
        'source_schema_sha256': _digest(schema), 'output_schema_sha256': _digest(new_schema),
        'source_format': 'PK-Sim 12.3.173 snapshot intervals', 'source_format_evidence': sources,
        'required_times_min': requested, 'administration_events': events,
        'administration_event_times_min': administration, 'required_time_points_min': required,
        'estimated_output_times_min': emitted, 'estimated_point_count': len(emitted),
        'original_point_count': len(original), 'point_limit_exclusive': max_points,
        'default_point_limit_exclusive': POINT_LIMIT_EXCLUSIVE,
        'larger_point_limit_explicitly_requested': max_points > POINT_LIMIT_EXCLUSIVE,
        'absolute_point_limit_exclusive': MAX_POINT_LIMIT_EXCLUSIVE,
        'window_before_min': [original[0], original[-1]], 'window_after_min': [emitted[0], emitted[-1]],
        'interval_count_before': len(schema), 'interval_count_after': len(new_schema),
        'original_sampling_points_preserved': True,
        'near_integer_step_roundoff_registration': roundoff,
        'limitations': ['Exact time coverage must be checked in native output after execution.',
                        'Output sampling does not prove ODE convergence or improve model physiology.',
                        'Source intervals with nonintegral point counts require native grid registration.'],
    }
    return {'snapshot': result, 'manifest': manifest}
