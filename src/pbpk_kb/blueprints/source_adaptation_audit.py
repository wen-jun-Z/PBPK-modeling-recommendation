"""Verify complete-source adaptations against real exported native PKML values.

No ODE or fitting is performed. Source Simulation.Parameters are checked along
with the caller's explicit event/parameter expectations. Counts describe the
exported model; they do not establish mechanistic equivalence or accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re


AUDIT_VERSION = 'source-adaptation-audit-1.0'
_R_SCRIPT = 'src/pbpk_kb/model_runtime/R/source_adaptation_audit.R'
_EVENT = re.compile(r'^Events\|.+\|ProtocolSchemaItem\|(Dose|Start time|Infusion time)$')
_SHA = re.compile(r'^[0-9a-f]{64}$')


def _run_bounded_r(**kwargs):
    from .design_runtime import run_bounded_r
    return run_bounded_r(**kwargs)


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inside(path, root, *, exists=False):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=exists)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError('Native audit paths must remain inside the project')
    return resolved


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Audit expectation values must be finite real numbers')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Audit expectation values must be finite real numbers')
    return value


def _json_safe(value):
    # Invalid native numbers remain failures, but must not prevent persisting
    # the failure report itself. Preserve the diagnostic reason and emit null.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _expectation(row, *, origin):
    if not isinstance(row, dict):
        raise ValueError('Every audit expectation must be an object')
    path, unit = row.get('path'), row.get('unit')
    if (not isinstance(path, str) or not path or len(path) > 2048
            or any(char in path for char in '*\x00\n\r')):
        raise ValueError('Audit expectations require an exact native parameter path')
    if not isinstance(unit, str) or len(unit) > 128 or any(char in unit for char in '\x00\n\r'):
        raise ValueError('Audit expectations require an explicit unit string')
    label = row.get('label', path)
    if not isinstance(label, str) or len(label) > 4096:
        raise ValueError('Audit expectation label must be a bounded string')
    return {'path': path, 'value': _number(row.get('value')), 'unit': unit,
            'origin': origin, 'label': label}



def _simulation_override_expectation(row, requested_rows, source_paths):
    """Resolve the documented legacy Applications root within a bound event.

    PK-Sim ParameterMapper.adjustedPath maps the pre-V12 Applications root to
    Events. Preserve the original source address and all values/units; the R
    reader must still resolve one exact native path and compare its value.
    """
    check = _expectation({'path': row.get('Path'), 'value': row['Value'],
                          'unit': row.get('Unit', '')}, origin='simulation_override')
    source_path = check['path']
    if not source_path.startswith('Applications|'):
        return check
    parts = source_path.split('|')
    if len(parts) < 5 or parts[-2] != 'ProtocolSchemaItem' or any(not part for part in parts):
        raise ValueError('Legacy application override requires an exact ProtocolSchemaItem parameter')
    native_path = 'Events|' + source_path[len('Applications|'):]
    prefix = native_path.rsplit('|', 1)[0]
    fields = {spec['path'].rsplit('|', 1)[-1] for spec in requested_rows
              if spec.get('origin') == 'requested' and _EVENT.fullmatch(spec['path'])
              and spec['path'].rsplit('|', 1)[0] == prefix}
    if not {'Dose', 'Start time'} <= fields:
        raise ValueError('Legacy application override is not bound to a registered administration event')
    if native_path in source_paths:
        raise ValueError('Legacy and native application override paths are both present')
    check.update(path=native_path, source_path=source_path,
                 path_mapping='legacy_applications_root_to_events')
    return check


def build_expectations(snapshot, expectations):
    """Audit selected-individual and final simulation overrides without gaps."""
    if (not isinstance(snapshot, dict) or type(snapshot.get('Version')) is not int
            or snapshot['Version'] != 80):
        raise ValueError('Native source audit requires a Version=80 snapshot')
    simulations, compounds = snapshot.get('Simulations'), snapshot.get('Compounds')
    if not isinstance(simulations, list) or len(simulations) != 1 or not isinstance(simulations[0], dict):
        raise ValueError('Native source audit requires exactly one simulation')
    if (not isinstance(compounds, list) or not compounds
            or any(not isinstance(c, dict) or not isinstance(c.get('Name'), str) or not c['Name'] for c in compounds)
            or len({c['Name'] for c in compounds}) != len(compounds)):
        raise ValueError('Native source audit requires nonempty uniquely named compound definitions')
    # Complete repository snapshots can retain unused drug definitions and
    # metabolite dependencies. Their presence is not an additional drug dose.
    # The native event inventory below independently checks every actual
    # administration against explicit requested dose/time expectations.
    selections = simulations[0].get('Compounds', [])
    names = {c['Name'] for c in compounds}
    if (not isinstance(selections, list) or any(not isinstance(c, dict) or c.get('Name') not in names for c in selections)
            or len({c['Name'] for c in selections}) != len(selections)):
        raise ValueError('Selected compound definitions are missing or ambiguous')
    if not isinstance(expectations, list):
        raise ValueError('Explicit native expectations must be a list')
    rows = [_expectation(row, origin='requested') for row in expectations]
    source = simulations[0].get('Parameters', [])
    if not isinstance(source, list):
        raise ValueError('Simulation.Parameters must be a list')
    individual = _selected_individual(snapshot)
    individual_parameters = individual.get('Parameters', [])
    if not isinstance(individual_parameters, list):
        raise ValueError('Individual.Parameters must be a list')
    simulation_paths = {row.get('Path') for row in source if isinstance(row, dict)}
    for row in individual_parameters:
        if not isinstance(row, dict) or 'TableFormula' in row or 'Value' not in row:
            raise ValueError('Native audit cannot silently skip nonscalar Individual.Parameters')
        check = _expectation({'path': row.get('Path'), 'value': row['Value'],
                              'unit': row.get('Unit', '')}, origin='selected_individual_override')
        # Simulation-local values are applied after the selected individual.
        # A shadowed individual value cannot also be the final native value.
        if check['path'] not in simulation_paths:
            rows.append(check)
    for row in source:
        if not isinstance(row, dict) or 'TableFormula' in row or 'Value' not in row:
            raise ValueError('Native audit cannot silently skip nonscalar Simulation.Parameters')
        rows.append(_simulation_override_expectation(row, rows, simulation_paths))
    if not 1 <= len(rows) <= 2048:
        raise ValueError('Native audit requires 1..2048 explicit and source parameter checks')
    return [dict(row, check_id=index) for index, row in enumerate(rows)]


def _selected_individual(snapshot):
    simulations = snapshot.get('Simulations', [])
    if len(simulations) != 1 or not isinstance(simulations[0], dict):
        raise ValueError('Exactly one selected simulation is required')
    selected = simulations[0].get('Individual')
    people = snapshot.get('Individuals')
    if not isinstance(selected, str) or not isinstance(people, list):
        raise ValueError('A named, source-defined selected individual is required')
    matching = [row for row in people if isinstance(row, dict) and row.get('Name') == selected]
    if len(matching) != 1:
        raise ValueError('Selected source individual is missing or ambiguous')
    return matching[0]


def derive_administration_expectations(snapshot):
    """Derive CLI readback checks for supported source administration layouts.

    This checks that native compilation implements its input snapshot. Evaluation
    against an independent target must supply separately sourced expectations.
    Weight-normalized doses are deliberately unsupported; they need a separate
    native body-weight binding before converting into event mass.
    """
    from .source_adaptation import _administrations, _named, _parameters, _time, _one_administered_selection

    simulations = snapshot.get('Simulations') if isinstance(snapshot, dict) else None
    if not isinstance(simulations, list) or len(simulations) != 1:
        raise ValueError('Exactly one simulation is required for administration audit')
    simulation = simulations[0]
    selection = _one_administered_selection(simulation)['Protocol']
    protocol = _named(snapshot.get('Protocols'), selection.get('Name'), 'Protocols')
    items, times = _administrations(protocol)
    if len(items) != 1:
        raise ValueError('Automatic audit supports one repeated administration item only')
    item, expectations = items[0], []
    parameters = _parameters(item)
    dose = parameters.get('InputDose')
    if not isinstance(dose, dict) or dose.get('Unit') not in {'g', 'mg', 'µg', 'ug', 'ng'}:
        raise ValueError('Automatic native dose audit requires absolute mass; mg/kg needs explicit expectations')
    route = item.get('ApplicationType')
    if route not in {'Oral', 'Intravenous', 'IntravenousBolus'}:
        raise ValueError('Unsupported route for automatic administration readback')
    path_parts = ['Events', protocol['Name']]
    if route == 'Oral':
        formulations = selection.get('Formulations', [])
        # A simple PK-Sim protocol uses the canonical "Formulation" key and
        # omits FormulationKey in its snapshot; advanced schema items name it.
        formulation_key = item.get('FormulationKey', 'Formulation' if 'Schemas' not in protocol else None)
        matching = [row for row in formulations if row.get('Key') == formulation_key]
        if len(matching) != 1:
            raise ValueError('Exactly one source formulation must match the oral application key')
        path_parts.append(matching[0]['Name'])
    overrides = {row['Path']: row for row in simulation.get('Parameters', [])}
    for index, start in enumerate(times, 1):
        prefix = '|'.join(path_parts + [f'Application_{index}', 'ProtocolSchemaItem'])
        fields = [('Dose', dose['Value'], dose['Unit']), ('Start time', start, 'min')]
        if route == 'Intravenous':
            if 'Infusion time' not in parameters:
                raise ValueError('An IV infusion requires an explicit duration')
            fields.append(('Infusion time', _time(parameters['Infusion time']), 'min'))
        for field, value, unit in fields:
            path = prefix + '|' + field
            if path in overrides:
                override = overrides[path]
                value, unit = override['Value'], override.get('Unit', '')
            expectations.append({'path': path, 'value': _number(value), 'unit': unit})
    return {'expectations': expectations, 'expected_event_count': len(times)}


def assess_native_readback(native, request):
    """Fail closed on missing, altered, nonfinite, or partly covered readback."""
    reasons, parameter_checks = [], []
    if (not isinstance(native, dict) or native.get('adapter_version') != AUDIT_VERSION
            or native.get('pkml_sha256') != request['pkml_sha256']
            or type(native.get('solver_calls')) is not int or native['solver_calls'] != 0
            or native.get('model_modified') is not False):
        return {'accepted': False, 'reasons': ['Native audit identity or read-only contract mismatch'],
                'parameter_checks': [], 'events': [], 'inventory': {}}
    rows = native.get('parameter_checks')
    expected = request['expectations']
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)):
        reasons.append('Native parameter checks must cover every declared expectation')
        rows = []
    by_id = {}
    for row in rows:
        index = row.get('check_id')
        if type(index) is not int or index in by_id or not 0 <= index < len(expected):
            reasons.append('Native parameter check identities are missing or ambiguous')
            continue
        by_id[index] = row
    for spec in expected:
        actual = by_id.get(spec['check_id'], {})
        check = {**spec, 'passed': False, 'native_value': actual.get('native_value'),
                 'expected_native_value': actual.get('expected_native_value'),
                 'native_unit': actual.get('native_unit'),
                 'native_dimension': actual.get('native_dimension'), 'reason': None}
        try:
            if actual.get('path') != spec['path']:
                raise ValueError('Native parameter path differs from the request')
            if (actual.get('expected_unit') != spec['unit']
                    or _number(actual.get('expected_value')) != spec['value']):
                raise ValueError('Native parameter expectation differs from the request')
            if actual.get('status') != 'evaluated':
                raise ValueError(actual.get('error') or 'Native parameter could not be evaluated or converted')
            if not isinstance(actual.get('native_unit'), str) or not isinstance(actual.get('native_dimension'), str):
                raise ValueError('Native parameter dimension and unit must be explicit')
            value, target = _number(actual.get('native_value')), _number(actual.get('expected_native_value'))
            if not math.isclose(value, target, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError('Actual native parameter value differs from the converted expectation')
            check['passed'] = True
        except (ValueError, TypeError, OverflowError) as exc:
            check['reason'] = str(exc)
            reasons.append(spec['path'] + ': ' + str(exc))
        parameter_checks.append(check)
    events = native.get('events')
    if not isinstance(events, list) or not events:
        reasons.append('Native audit found no administration events')
        events = []
    groups, event_paths = {}, set()
    expected_paths = {row['path'] for row in expected}
    for row in events:
        try:
            if not isinstance(row, dict):
                raise ValueError('Native event row must be an object')
            path = row.get('path')
            match = _EVENT.fullmatch(path) if isinstance(path, str) else None
            if not match or path in event_paths or row.get('status') != 'evaluated':
                raise ValueError('Native event path is missing, duplicate, or not evaluable')
            field, prefix = match[1], path.rsplit('|', 1)[0]
            if row.get('field') != field or row.get('event_path') != prefix:
                raise ValueError('Native event grouping differs from its path')
            value = _number(row.get('value'))
            if value < 0 or not isinstance(row.get('unit'), str):
                raise ValueError('Native event values must be nonnegative with explicit units')
            event_paths.add(path)
            groups.setdefault(prefix, set()).add(field)
            if path not in expected_paths:
                reasons.append('Native event parameter lacks an explicit expectation: ' + path)
        except (ValueError, TypeError, OverflowError) as exc:
            reasons.append(str(exc))
    if any(not {'Dose', 'Start time'} <= fields for fields in groups.values()):
        reasons.append('Every native administration event must expose both Dose and Start time')
    if len(groups) != request['expected_event_count']:
        reasons.append('Native administration event count differs from the requested schedule')
    inventory = native.get('inventory')
    if not isinstance(inventory, dict):
        inventory = {}
    for field in ('parameter_count', 'container_count', 'organism_parameter_count', 'organism_container_count'):
        if type(inventory.get(field)) is not int or inventory[field] < 1:
            reasons.append('Native physiological inventory is missing: ' + field)
    return {'accepted': not reasons, 'reasons': reasons, 'parameter_checks': parameter_checks,
            'events': events, 'event_count': len(groups), 'expected_event_count': request['expected_event_count'],
            'inventory': inventory}


def audit_execution(*, snapshot, execution_manifest, output_dir, project_root,
                    expectations, expected_event_count=None):
    """Load one exported model, verify all declared values, and save audit.json.

    All paths must belong to this project. The audit uses at most 180 seconds,
    64 MiB of output and two native CPU threads, with no ODE execution.
    ``execution_manifest`` may be execute_snapshot's returned dictionary or its
    execution_manifest.json path. A fresh output directory is required.
    """
    root = Path(project_root).resolve(strict=True)
    checks = build_expectations(snapshot, expectations)
    if isinstance(execution_manifest, dict):
        manifest_path = _inside(execution_manifest.get('manifest_path', ''), root, exists=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest != execution_manifest:
            raise ValueError('Provided execution manifest differs from the saved native execution')
    else:
        manifest_path = _inside(execution_manifest, root, exists=True)
        manifest = json.loads(manifest_path.read_text())
    if manifest.get('engineering_accepted') is not True or manifest.get('source_unchanged') is not True:
        raise ValueError('A successful, unchanged native execution is required before audit')
    original = _inside(manifest_path.parent / 'snapshot.json', root, exists=True)
    original_sha, manifest_sha = _sha(original), _sha(manifest_path)
    if original_sha != manifest.get('source_sha256') or json.loads(original.read_text()) != snapshot:
        raise ValueError('Audited snapshot is not the snapshot used to produce the native model')
    outputs = manifest.get('pkml_outputs')
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise ValueError('Exactly one exported PKML is required for single-model audit')
    item = outputs[0]
    if not isinstance(item.get('sha256'), str) or not _SHA.fullmatch(item['sha256']):
        raise ValueError('Exported native model requires its verified SHA256')
    pkml = _inside(manifest_path.parent / item['path'], root, exists=True)
    if (not pkml.is_relative_to(manifest_path.parent) or pkml.suffix != '.pkml'
            or _sha(pkml) != item['sha256']):
        raise ValueError('Native model path or SHA256 differs from its execution manifest')
    if expected_event_count is None:
        expected_event_count = len({row['path'].rsplit('|', 1)[0] for row in checks
                                    if _EVENT.fullmatch(row['path']) and row['path'].endswith('|Dose')})
    if type(expected_event_count) is not int or not 1 <= expected_event_count <= 1024:
        raise ValueError('Declare 1..1024 expected administration events with explicit Dose paths')
    directory = _inside(output_dir, root)
    directory.mkdir(parents=True, exist_ok=False)
    result_path, request_path = directory / 'native_readback.json', directory / 'request.json'
    request = {'adapter_version': AUDIT_VERSION, 'pkml_path': str(pkml), 'pkml_sha256': item['sha256'],
               'source_sha256': original_sha, 'expectations': checks,
               'expected_event_count': expected_event_count, 'result_path': str(result_path)}
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    request_sha = _sha(request_path)
    individual = _selected_individual(snapshot)
    simulation_paths = {row['Path'] for row in snapshot['Simulations'][0].get('Parameters', [])}
    report = {'adapter_version': AUDIT_VERSION, 'accepted': False, 'status': 'failed',
              'source_sha256': original_sha, 'pkml_sha256': item['sha256'],
              'execution_manifest': str(manifest_path.relative_to(root)),
              'execution_manifest_sha256': manifest_sha, 'request_sha256': request_sha,
              'scientific_accepted': False, 'solver_calls': 0, 'model_modified': False,
              'selected_individual': individual['Name'],
              'individual_overrides_superseded_by_simulation': [row['Path'] for row in individual.get('Parameters', [])
                                                               if row['Path'] in simulation_paths],
              'claim_boundary': 'Native parameter and administration readback, not predictive validation or complete mechanism-equivalence proof'}
    try:
        process = _run_bounded_r(script=root / _R_SCRIPT, args=[str(request_path), request_sha],
                                output_dir=directory, project_root=root, timeout_seconds=180,
                                max_output_bytes=64 * 1024**2, cpu_threads=2)
        report['process'] = process
        if process.get('status') != 'completed':
            raise ValueError('Native audit worker failed: ' + str(process.get('stop_reason')))
        if not result_path.is_file() or result_path.stat().st_size > 32 * 1024**2:
            raise ValueError('Native audit did not produce a bounded readback file')
        if (request_sha != _sha(request_path) or item['sha256'] != _sha(pkml)
                or original_sha != _sha(original) or manifest_sha != _sha(manifest_path)):
            raise ValueError('Bound native audit inputs changed during inspection')
        native = json.loads(result_path.read_text())
        if native.get('request_sha256') != request_sha:
            raise ValueError('Native readback is not bound to the requested audit')
        report.update(assess_native_readback(native, request))
        report.update(status='passed' if report['accepted'] else 'failed',
                      native_readback_sha256=_sha(result_path),
                      native_readback_file=str(result_path.relative_to(root)))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        report['reasons'] = [str(exc)]
    report = _json_safe(report)
    with (directory / 'audit.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return report
