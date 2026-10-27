"""Bounded execution of complete source models without snapshot reconstruction.

Clinical conditions must be registered before invoking this adapter. It does not
select models, infer dosing histories, fit parameters or read clinical outcomes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.design_runtime import run_bounded_r
from .model_semantics import compare_saved_model_xml

VERSION = 'recommendation-complete-pkml-v1'
WORKER = 'src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R'
ADMINISTRATION = re.compile(r'^(Events|Applications)\|.+\|ProtocolSchemaItem\|(Dose|Start time|Infusion time)$')


def inside(path, root=ROOT, *, exists=True):
    root = Path(root).resolve(strict=True)
    path = (root / path).resolve(strict=exists)
    if path == root or not path.is_relative_to(root) or any(p.lower().startswith('sealed') for p in path.relative_to(root).parts):
        raise ValueError('Only unsealed resources inside this project are allowed')
    return path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def validate_request(request):
    if request.get('version') != VERSION or request.get('action') not in {'inspect', 'run'}:
        raise ValueError('Unknown complete-model request')
    monitors = request.get('zero_monitor_paths', [])
    dimensions = request.get('zero_monitor_dimensions', {})
    if (not isinstance(monitors, list) or len(monitors) > 4096 or
            any(not isinstance(p, str) or not p or len(p) > 2048 for p in monitors) or
            len(set(monitors)) != len(monitors) or not isinstance(dimensions, dict) or
            set(dimensions) != set(monitors) or any(d not in {'Amount', 'Mass', 'Amount per time','Dimensionless'} for d in dimensions.values()) or
            any(d=='Dimensionless' and not p.endswith('|TabletIsActive') for p,d in dimensions.items()) or
            request.get('output_path') in monitors):
        raise ValueError('Exact unique zero-mass monitors and dimensions are required')
    domain = request.get('positive_domain_monitor_dimensions', {})
    if (not isinstance(domain, dict) or len(domain) > 128 or any(not isinstance(p, str) or not p or d != 'Volume' for p,d in domain.items())
            or set(domain).intersection(monitors) or request.get('output_path') in domain):
        raise ValueError('Explicit disjoint dynamic positive-volume domains required')
    requirements = request.get('proof_parameter_requirements', [])
    if (not isinstance(requirements, list) or len(requirements) > 4096 or
            any(not isinstance(r, dict) or not isinstance(r.get('path'), str) or not r['path'] or
                not isinstance(r.get('id'), str) or not r['id'] or not isinstance(r.get('dimension'), str) or
                r.get('finite_required') is not True or type(r.get('strictly_positive_required')) is not bool or
                ('nonnegative_required' in r and type(r['nonnegative_required']) is not bool)
                for r in requirements) or len({r['path'] for r in requirements}) != len(requirements)):
        raise ValueError('Exact finite static proof parameter requirements are required')
    if request['action'] == 'inspect':
        if request.get('patches'):
            raise ValueError('Inspection cannot change parameters')
        return
    if not isinstance(request.get('output_path'), str) or not request['output_path']:
        raise ValueError('An exact output path is required')
    times = request.get('output_times_min')
    if (not isinstance(times, list) or not 2 <= len(times) <= 50000 or
            not all(_finite(t) for t in times) or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Output times must be finite, increasing and start at zero')
    patches = request.get('patches', [])
    if not isinstance(patches, list) or len(patches) > 256:
        raise ValueError('Administration change limit exceeded')
    paths = []
    for patch in patches:
        path = patch.get('path', '')
        if not ADMINISTRATION.fullmatch(path):
            raise ValueError('Only administration dose, start and infusion may change')
        if (not _finite(patch.get('value')) or patch['value'] < 0 or
                not _finite(patch.get('expected_before_value')) or
                not isinstance(patch.get('expected_before_unit'), str) or
                not isinstance(patch.get('unit'), str) or not patch['unit']):
            raise ValueError('Explicit finite values and units are required')
        if path.endswith('|Dose') and patch['unit'] not in {'mg', 'g', 'kg', 'µg', 'ug'}:
            raise ValueError('Absolute parent mass must be resolved before native execution')
        if not path.endswith('|Dose') and patch['unit'] not in {'min', 'h', 's', 'day(s)'}:
            raise ValueError('Explicit administration time units are required')
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate administration paths are forbidden')


def _model_tree(path):
    tree = ET.parse(path)
    models = tree.findall('.//Model')
    if len(models) != 1:
        raise ValueError('Exactly one complete model is required')
    return deepcopy(models[0])


def _canonical_xml(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(), tuple(_canonical_xml(c) for c in node))


def audit_configuration(before, after, *, allow_output_changes, allow_import_encoding=False):
    nodes = []
    for path in (before, after):
        configs = ET.parse(path).findall('.//SimulationConfiguration')
        if len(configs) != 1:
            raise ValueError('Exactly one complete simulation configuration is required')
        config = configs[0]
        if allow_import_encoding:
            # Percentiles are dimensionless. Legacy PK-Sim files attached the
            # parent parameter's display unit; OSPSuite drops that UI label.
            # Values, percentiles, dimensions, origins and all other fields stay
            # protected. Generated calculation-method IDs are unreferenced
            # serializer identities, while their full definitions remain bound.
            referenced = {v for n in config.iter() for k, v in n.attrib.items() if k != 'id'}
            for node in config.iter():
                if node.tag == 'IndividualParameter' and node.get('name') == 'Percentile' and not node.get('dim'):
                    node.attrib.pop('displayUnit', None)
                if node.tag == 'CoreCalculationMethod' and node.get('id') not in referenced:
                    node.attrib.pop('id', None)
        if allow_output_changes:
            output_formula_ids = set()
            for parent in config.iter():
                for child in list(parent):
                    if child.tag in {'OutputSchema', 'OutputSelections'}:
                        output_formula_ids.update(n.get(k) for n in child.iter() for k in ('formula', 'rHSFormula', 'rhs') if n.get(k))
                        parent.remove(child)
            remaining_formula_ids = {n.get(k) for n in config.iter() for k in ('formula', 'rHSFormula', 'rhs') if n.get(k)}
            for parent in config.iter():
                for cache in list(parent):
                    if cache.tag != 'FormulaCache':
                        continue
                    formulas = cache.find('Formulas')
                    if formulas is None:
                        raise ValueError('Configuration formula cache has no definitions')
                    for formula in list(formulas):
                        if (formula.get('id') in output_formula_ids - remaining_formula_ids and
                                formula.tag == 'ConstantFormula' and not len(formula)):
                            formulas.remove(formula)
                    # Only a now-empty output constant cache may disappear.
                    if not len(formulas) and all(n.tag in {'Formulas', 'StringMap'} and not len(n) for n in cache):
                        parent.remove(cache)
        nodes.append(_canonical_xml(config))
    return nodes[0] == nodes[1]


def audit_preserved_model(before, after, changes, *, allow_output_changes=True):
    """Compare every model node except registered administration/output edits.

    StringMap indices are expanded locally within each formula cache. Unknown
    changes are never discarded; formulas and mechanism nodes remain compared.
    """
    trees = [_model_tree(before), _model_tree(after)]
    original_formula_nodes = {n.get('id'): deepcopy(n) for group in trees[0].iter('Formulas') for n in group}
    original_parameter_formulas = {n.get('id'): n.get('formula') for n in trees[0].iter('Parameter')}
    original_parameters = {n.get('id'): dict(n.attrib) for n in trees[0].iter('Parameter')}
    ids = {c['id'] for c in changes}
    if len(ids) != len(changes) or any(not ADMINISTRATION.fullmatch(c['path']) for c in changes):
        raise ValueError('Ambiguous or unauthorized parameter changes')
    removed = []
    def normalize(model, side):
        # Path aliases are serialization compression, not model parameters.
        for cache in model.iter('FormulaCache'):
            mapping_node = cache.find('StringMap')
            if mapping_node is None:
                continue
            mapping = {row.attrib['id']: row.attrib['s'] for row in mapping_node}
            if len(mapping) != len(mapping_node):
                raise ValueError('Ambiguous serialized string map')
            for row in cache.iter('Path'):
                for key, value in list(row.attrib.items()):
                    if key in {'path', 'as', 'dim'} and value in mapping:
                        row.attrib[key] = mapping[value]
            cache.remove(mapping_node)
        found = set()
        formulas = {n.get('id'): (parent, n) for parent in model.iter('Formulas') for n in parent}
        if len(formulas) != sum(len(n) for n in model.iter('Formulas')):
            raise ValueError('Duplicate formula identifiers')
        by_id = {c['id']: c for c in changes}
        removable = set()
        for parent in model.iter():
            for child in list(parent):
                if allow_output_changes and child.tag in {'OutputSchema', 'OutputSelections'}:
                    parent.remove(child)
            if parent.tag == 'Parameter' and parent.get('id') in ids:
                if parent.get('id') in found:
                    raise ValueError('Duplicate changed parameter ID')
                found.add(parent.get('id'))
                change = by_id[parent.get('id')]
                if parent.get('name') != change['path'].split('|')[-1]:
                    raise ValueError('Changed parameter name and path differ')
                expected = change.get('expected_native_value') if side == 'after' else change.get('before', {}).get('value')
                if not _finite(expected):
                    raise ValueError('Independently bound administration values are required')
                if 'value' in parent.attrib:
                    if not math.isclose(float(parent.attrib['value']), expected, rel_tol=1e-12, abs_tol=1e-15):
                        raise ValueError('Inline administration value differs from native readback')
                formula_id = parent.get('formula')
                if formula_id:
                    if formula_id not in formulas:
                        raise ValueError('Administration formula is missing')
                    _, formula = formulas[formula_id]
                    if side == 'after':
                        expected = change.get('expected_native_value')
                        new_constant = (formula.tag == 'ConstantFormula' and not len(formula) and
                                        _finite(expected) and math.isclose(float(formula.get('value')), expected, rel_tol=1e-12, abs_tol=1e-15))
                        original = original_formula_nodes.get(original_parameter_formulas.get(parent.get('id')))
                        old_parameter = original_parameters[parent.get('id')]
                        constant_from_original_literal = (
                            not old_parameter.get('formula') and 'value' in old_parameter and
                            formula.tag == 'ConstantFormula' and not len(formula) and
                            set(formula.attrib).issubset({'id','dim','value'}) and
                            formula.get('dim') == old_parameter.get('dim') and
                            math.isclose(float(formula.get('value')),float(old_parameter['value']),rel_tol=1e-12,abs_tol=1e-15))
                        preserved_fixed_default = (parent.get('isFixedValue') == '1' and 'value' in parent.attrib and
                                                   ((original is not None and _canonical_xml(original) == _canonical_xml(formula)) or
                                                    constant_from_original_literal))
                        if not new_constant and not preserved_fixed_default:
                            raise ValueError('Serialized administration must be the requested scalar or a fixed scalar override of its unchanged original formula')
                    users = [n for n in model.iter() if n.get('formula') == formula_id]
                    # Preserve shared formulas: another mechanism may still use
                    # one. Only an exclusively replaced administration formula
                    # may disappear as part of the registered dose/time edit.
                    if len(users) == 1 and users[0] is parent:
                        removable.add(formula_id)
                    parent.attrib.pop('formula')
                elif 'value' not in parent.attrib:
                    raise ValueError('Administration has neither scalar value nor formula')
                for key in ('value', 'isFixedValue', 'isDefault'):
                    parent.attrib.pop(key, None)
        if found != ids:
            raise ValueError('Changed parameter ID missing in complete model')
        for formula_id in removable:
            parent, formula = formulas[formula_id]
            parent.remove(formula)
        removed.append(len(found))
        return _canonical_xml(model)
    a, b = [normalize(t, side) for t, side in zip(trees, ('before', 'after'))]
    passed = a == b
    return {'accepted': passed, 'all_other_model_nodes_unchanged': passed,
            'registered_administration_parameters': len(ids),
            'full_model_sha_before': sha(before), 'full_model_sha_after': sha(after),
            'allowed_differences': ['explicit dose/start/infusion scalar value flags', 'output selections and schema',
                                    'exclusively replaced administration formula; new value independently read back',
                                    'fixed scalar override may retain its unchanged original default formula',
                                    'formula string-table index encoding'],
            'fitting_performed': False}


def validate_native_output(native, request):
    changes = native.get('parameter_changes', [])
    patches = request.get('patches', [])
    if len(changes) != len(patches) or len({c.get('path') for c in changes}) != len(changes):
        raise ValueError('Native changes do not match registered administration edits')
    by_path = {c['path']: c for c in changes}
    mass = {'kg':1., 'g':.001, 'mg':1e-6, 'ug':1e-9, 'µg':1e-9}
    time = {'min':1., 'h':60., 's':1/60., 'day(s)':1440.}
    for patch in patches:
        change = by_path.get(patch['path'], {})
        before, after = change.get('before', {}), change.get('after', {})
        if (change.get('requested_value') != patch['value'] or change.get('requested_unit') != patch['unit'] or
                before.get('unit') != patch['expected_before_unit'] or before.get('value') != patch['expected_before_value'] or
                after.get('path') != patch['path'] or before.get('path') != patch['path'] or
                change.get('id') != before.get('id') or change.get('id') != after.get('id')):
            raise ValueError('Native administration binding differs from request')
        units = mass if patch['path'].endswith('|Dose') else time
        expected = patch['value'] * units[patch['unit']] / units[patch['expected_before_unit']]
        if (after.get('unit') != before['unit'] or not _finite(after.get('value')) or
                not _finite(change.get('expected_native_value')) or
                not math.isclose(after['value'], expected, rel_tol=1e-12, abs_tol=1e-15) or
                not math.isclose(change['expected_native_value'], expected, rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError('Native unit conversion or administration value differs')
    curve = native.get('curve', {})
    actual, expected = curve.get('time_min', []), request['output_times_min']
    raw_times = native.get('native_output_time_min', actual)
    values = curve.get('concentration', [])
    if (curve.get('time_unit') != 'min' or len(actual) != len(expected) or len(values) != len(actual) or
            not all(_finite(v) for v in actual + values) or any(v < 0 for v in values) or
            actual != expected or len(raw_times) != len(expected) or
            any(not _finite(a) or min(abs(a-b), abs(a-struct.unpack('f', struct.pack('f',b))[0])) > 1e-10
                for a,b in zip(raw_times, expected)) or native.get('output_grid_verified') is not True):
        raise ValueError('Native grid/curve differs from the frozen registration')
    validate_zero_monitors(native, request)
    validate_positive_domains(native, request)


def validate_proof_parameters(native, request):
    expected = request.get('proof_parameter_requirements', [])
    rows = native.get('proof_parameter_readback', [])
    if (not isinstance(rows, list) or len(rows) != len(expected) or
            any(not isinstance(r, dict) for r in rows) or len({r.get('path') for r in rows}) != len(rows)):
        raise ValueError('Static proof parameter readback coverage differs')
    by_path = {r.get('path'): r for r in rows}
    for requirement in expected:
        row = by_path.get(requirement['path'], {})
        if (row.get('id') != requirement['id'] or row.get('dimension') != requirement['dimension'] or
                not isinstance(row.get('unit'), str) or not _finite(row.get('value')) or
                (requirement.get('nonnegative_required') is True and row['value'] < 0) or
                (requirement['strictly_positive_required'] and row['value'] <= 0)):
            raise ValueError('Static source coefficient domain is not independently verified')


def validate_zero_monitors(native, request):
    """Replay every frozen monitor and coefficient check before scoring."""
    validate_proof_parameters(native, request)
    paths = request.get('zero_monitor_paths', [])
    rows = native.get('zero_monitors', [])
    if (not isinstance(rows, list) or len(rows) != len(paths) or
            any(not isinstance(r, dict) for r in rows) or len({r.get('path') for r in rows}) != len(rows) or
            {r.get('path') for r in rows} != set(paths)):
        raise ValueError('Zero-exposure monitor path coverage differs')
    if not paths:
        return
    expected = request['output_times_min']
    actual = native.get('zero_monitor_time_min')
    raw = native.get('zero_monitor_native_time_min')
    if (actual != expected or not isinstance(raw, list) or len(raw) != len(expected) or
            any(not _finite(a) or min(abs(a-b), abs(a-struct.unpack('f', struct.pack('f', b))[0])) > 1e-10
                for a, b in zip(raw, expected)) or native.get('zero_monitor_grid_verified') is not True):
        raise ValueError('Zero-exposure monitor time grid differs')
    units = {'Amount': {'µmol', 'μmol', 'umol', 'mol', 'mmol', 'nmol'},
             'Mass': {'kg', 'g', 'mg', 'µg', 'μg', 'ug', 'ng'},
             'Amount per time': {'µmol/min', 'μmol/min', 'umol/min', 'mol/min', 'mmol/min'},'Dimensionless':{''}}
    dimensions = request['zero_monitor_dimensions']
    for row in rows:
        values = row.get('values')
        if (row.get('dimension') != dimensions[row['path']] or row.get('unit') not in units[row['dimension']] or
                not isinstance(values, list) or len(values) != len(expected) or
                any(not _finite(v) or v != 0 for v in values)):
            raise ValueError('Removed mass has nonzero, incomplete or unqualified native output')



def validate_positive_domains(native, request):
    dimensions = request.get('positive_domain_monitor_dimensions', {})
    rows = native.get('positive_domain_monitors', [])
    if (not isinstance(rows,list) or len(rows) != len(dimensions) or any(not isinstance(r,dict) for r in rows)
            or len({r.get('path') for r in rows}) != len(rows) or {r.get('path') for r in rows} != set(dimensions)):
        raise ValueError('Dynamic domain monitor coverage differs')
    if not dimensions: return
    expected = request['output_times_min']
    if (native.get('domain_monitor_time_min') != expected or native.get('domain_monitor_grid_verified') is not True
            or native.get('domain_monitor_native_time_min') != native.get('native_output_time_min')):
        raise ValueError('Dynamic domain monitor grid differs')
    for row in rows:
        values = row.get('values')
        if (row.get('dimension') != dimensions[row['path']] or row.get('unit') not in {'l','L','ml','mL','µl','ul'}
                or not isinstance(values,list) or len(values) != len(expected)
                or any(not _finite(v) or v <= 0 for v in values)):
            raise ValueError('Dynamic volume denominator domain is not positive on full output grid')

def execute_complete_pkml(pkml, output_dir, *, action='inspect', patches=None, output_path=None,
                          output_times_min=None, timeout_seconds=180, cpu_threads=1,
                          project_root=ROOT, stop_file=None, zero_monitor_paths=None,
                          zero_monitor_dimensions=None, proof_parameter_requirements=None, positive_domain_monitor_dimensions=None):
    root = Path(project_root).resolve(strict=True)
    source = inside(pkml, root)
    if source.suffix != '.pkml' or not 0 < source.stat().st_size < 128 * 1024**2:
        raise ValueError('A bounded complete PKML is required')
    output = inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError('Preserve each attempt in a new directory')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800:
        raise ValueError('Execution timeout must be 1..1800 seconds')
    if cpu_threads not in {1, 2}:
        raise ValueError('One or two native CPU threads are allowed')
    request = dict(version=VERSION, action=action, pkml_path=str(source), pkml_sha256=sha(source),
                   patches=patches or [], output_path=output_path, output_times_min=output_times_min,
                   zero_monitor_paths=zero_monitor_paths or [], zero_monitor_dimensions=zero_monitor_dimensions or {},
                   proof_parameter_requirements=proof_parameter_requirements or [],
                   positive_domain_monitor_dimensions=positive_domain_monitor_dimensions or {},
                   cpu_threads=cpu_threads, result_path=str(output / 'native_result.json'),
                   before_pkml_path=str(output / 'before.pkml'), after_pkml_path=str(output / 'after.pkml'))
    validate_request(request)
    output.mkdir(parents=True)
    config = output / 'request.json'
    save(config, request)
    process = run_bounded_r(script=root / WORKER, args=[str(config), sha(config)], output_dir=output,
                            project_root=root, timeout_seconds=timeout_seconds, cpu_threads=cpu_threads,
                            max_output_bytes=384 * 1024**2, stop_file=stop_file)
    result = {'version': VERSION, 'action': action, 'source_sha256': request['pkml_sha256'],
              'status': 'failed', 'sealed_observations_accessed': False, 'fitting_performed': False,
              'process_manifest': str((output / 'process_manifest.json').relative_to(root)),
              'worker_sha256': sha(root / WORKER), 'python_sha256': sha(Path(__file__)),
              'reasons': [], 'native': None}
    if process.get('status') != 'completed' or process.get('returncode') != 0:
        result['reasons'].append(process.get('stop_reason') or 'native_worker_failed')
    elif sha(source) != request['pkml_sha256']:
        result['reasons'].append('original_model_changed')
    else:
      try:
        native = json.loads((output / 'native_result.json').read_text())
        if native['request_sha256'] != sha(config) or native['pkml_sha256'] != sha(source):
            raise ValueError('Native result binding differs')
        result['native'] = native
        if action == 'inspect':
            validate_proof_parameters(native, request)
            result['status'] = 'inspected'
        else:
            validate_native_output(native, request)
            if (native.get('before_pkml_sha256') != sha(output / 'before.pkml') or
                    native.get('after_pkml_sha256') != sha(output / 'after.pkml')):
                raise ValueError('Native serialized model hash differs')
            imported = compare_saved_model_xml(_model_tree(source), _model_tree(output / 'before.pkml'))
            imported['configuration_unchanged'] = audit_configuration(source, output / 'before.pkml', allow_output_changes=False, allow_import_encoding=True)
            imported['accepted'] &= imported['configuration_unchanged']
            result['source_import_audit'] = imported
            audit = audit_preserved_model(output / 'before.pkml', output / 'after.pkml', native['parameter_changes'])
            audit['configuration_unchanged'] = audit_configuration(output / 'before.pkml', output / 'after.pkml', allow_output_changes=True)
            audit['accepted'] &= audit['configuration_unchanged'] and imported['accepted']
            result['model_preservation_audit'] = audit
            result['status'] = 'prediction_locked' if audit['accepted'] else 'model_preservation_failed'
            if not audit['accepted']:
                result['reasons'].append('unregistered_model_change')
      except (ValueError, KeyError, OSError, TypeError) as exc:
        result['status'] = 'readback_failed'
        result['reasons'].append(type(exc).__name__ + ':' + str(exc)[:500])
    save(output / 'result.json', result)
    files = [p for p in sorted(output.iterdir()) if p.is_file()]
    save(output / 'artifact_lock.json', {'version': VERSION, 'source': {'path': str(source.relative_to(root)), 'sha256': sha(source)},
         'files': {str(p.relative_to(root)): sha(p) for p in files}, 'prediction_accepted': result['status']=='prediction_locked'})
    return result
