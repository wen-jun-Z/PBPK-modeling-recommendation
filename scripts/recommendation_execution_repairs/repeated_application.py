"""Expand an author's complete single tablet application without rebuilding PBPK.

This deliberately bounded adapter handles an existing oral tablet application,
including its drug state, release transports and one-time events. It does not
replace drug models, infer regimens, inspect outcomes or run an ODE. A separately
registered native execution/readback is required after this structural repair.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid
import xml.etree.ElementTree as ET

from . import model_semantics as semantics
from .native_pkml import inside, sha, save
from .source_pkml import (decompress_model_blob, wrap_preserved_model,
                          describe_saved_simulation, element_sha)

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'preserved-tablet-application-expansion-1'
IMPLEMENTATION_FILES = (
    'scripts/recommendation_execution_repairs/repeated_application.py',
    'scripts/recommendation_execution_repairs/model_semantics.py',
    'scripts/recommendation_execution_repairs/source_pkml.py',
    'scripts/recommendation_execution_repairs/native_pkml.py',
)
LINKS = semantics.LINKS
FORMULA_FIELDS = {'formula', 'rHSFormula', 'rhs'}
MAX_BYTES = 128 * 1024**2


def _finite(value):
    return type(value) in {float, int} and math.isfinite(value)


def _canonical(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
            tuple(_canonical(child) for child in node))


def _models(document):
    models = document.findall('.//Model')
    if len(models) != 1 or len(document.findall('.//SimulationConfiguration')) != 1:
        raise ValueError('One complete Model and SimulationConfiguration are required')
    model = models[0]
    if len(model.findall('Root')) != 1 or len(model.findall('FormulaCache/Formulas')) != 1:
        raise ValueError('Incomplete model structure')
    return model


def _inventory(model):
    paths, ids, parents = {}, {}, {}
    root = model.find('Root')
    if not root.get('name'):
        raise ValueError('Named simulation root is required')
    def walk(node, path):
        if node.get('name'):
            path = path + [node.get('name')]
            pathstr = '|'.join(path)
            if pathstr in paths:
                raise ValueError('Duplicate named entity path')
            paths[pathstr] = node
        if node.get('id') and node.get('name'):
            if node.get('id') in ids:
                raise ValueError('Duplicate model entity ID')
            ids[node.get('id')] = node
        for child in node:
            parents[child] = node
            walk(child, path)
    walk(root, [])
    return paths, ids, parents


def _formula_inventory(model):
    cache = model.find('FormulaCache')
    groups = cache.findall('StringMap')
    if len(groups) > 1:
        raise ValueError('Ambiguous formula string table')
    strings = {x.get('id'): x.get('s') for x in groups[0]} if groups else {}
    if groups and len(strings) != len(groups[0]):
        raise ValueError('Duplicate string-table ID')
    formulas = {x.get('id'): x for x in cache.find('Formulas')}
    if None in formulas or len(formulas) != len(cache.find('Formulas')):
        raise ValueError('Duplicate or missing formula ID')
    return strings, formulas


def _resolve_paths(formula, strings):
    return [strings.get(x.get('path'), x.get('path')) for x in formula.iter('Path')]


def _scalar(parameter, formulas):
    if parameter.tag not in {'Parameter', 'Para'} or parameter.get('rHSFormula') or parameter.get('rhs'):
        raise ValueError('Administration scalar cannot be a dynamic state')
    value = parameter.get('value')
    if parameter.get('formula'):
        f = formulas.get(parameter.get('formula'))
        if f is None or f.tag != 'ConstantFormula' or len(f):
            raise ValueError('Administration must have an explicit original scalar')
        if f.get('dim', parameter.get('dim')) != parameter.get('dim'):
            raise ValueError('Scalar formula and quantity dimensions differ')
        if value is not None and float(value) != float(f.get('value')):
            raise ValueError('Conflicting scalar formula and value')
        value = f.get('value')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Explicit finite administration scalar required') from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError('Invalid administration scalar')
    return number


def _validate_spec(times, dose):
    if (not isinstance(times, list) or not 1 <= len(times) <= 64 or
            not all(_finite(t) and t >= 0 for t in times) or
            any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Explicit increasing finite administration times are required')
    if not _finite(dose) or not 0 < dose <= 1e6:
        raise ValueError('Explicit positive absolute parent dose in mg is required')


def _source_contract(document, compound, prefix):
    model = _models(document)
    # Also rejects broken typed IDs/formula definitions before transformation.
    semantics.model_semantics(model)
    paths, ids, parents = _inventory(model)
    strings, formulas = _formula_inventory(model)
    rootname = model.find('Root').get('name')
    if not isinstance(prefix, str) or not prefix.startswith(('Applications|', 'Events|')):
        raise ValueError('An exact complete application prefix is required')
    fullprefix = rootname + '|' + prefix
    branch = paths.get(fullprefix)
    if (branch is None or branch.tag != 'EventGroup' or branch.get('eventGroupType') != 'Oral' or
            branch.get('containerType') != 'Application' or branch.get('name') != 'Application_1'):
        raise ValueError('Only an original first oral Application is supported')
    applications = [n for n in model.find('Root').iter('EventGroup')
                    if any(c.get('name') == 'ProtocolSchemaItem' for c in n.findall('Children/Container'))]
    if applications != [branch]:
        raise ValueError('Source must have exactly one complete dosing application; no coadministration')
    nodes = set(branch.iter())
    # ValueOrigin.id is an integer bibliographical reference, not an entity ID.
    # Native NullableIntAttributeMapper must receive that original integer.
    entities = [n for n in nodes if n.get('id') and n.get('name')]
    branchids = {n.get('id') for n in entities}
    if len(branchids) != len(entities):
        raise ValueError('Duplicate branch object IDs')
    dose = paths.get(fullprefix + '|ProtocolSchemaItem|Dose')
    start = paths.get(fullprefix + '|ProtocolSchemaItem|Start time')
    if dose is None or start is None or dose.get('dim') != 'Mass' or start.get('dim') != 'Time':
        raise ValueError('Original absolute mass/time dimensions required')
    original_dose = _scalar(dose, formulas)
    original_start = _scalar(start, formulas)
    molecules = list(branch.iter('MoleculeAmount'))
    transports = list(branch.iter('Transport'))
    events = list(branch.iter('Event'))
    if (len(molecules) != 1 or molecules[0].get('name') != compound or
            len(transports) != 12 or len(events) != 2 or
            {e.get('name') for e in events} != {'Application_StartEvent', 'Application_StopEvent'} or
            any(e.get('oneTime') != '1' for e in events) or
            not any(n.get('name') == 'TabletIsActive' for n in nodes)):
        raise ValueError('Only the source-verified complete tablet release topology is supported')
    start_event = next(e for e in events if e.get('name') == 'Application_StartEvent')
    condition = formulas.get(start_event.get('formula'))
    condition_paths = list(condition.iter('Path')) if condition is not None else []
    if (condition is None or condition.get('formula', '').replace(' ', '') != 'Time=StartTime' or
            len(condition_paths) != 1 or len(list(condition.iter('TimePath'))) != 1 or
            _resolve_paths(condition, strings) != ['..|ProtocolSchemaItem|Start time'] or
            strings.get(condition_paths[0].get('as'), condition_paths[0].get('as')) != 'StartTime'):
        raise ValueError('Start event must use the original single exact StartTime trigger')
    for transport in transports:
        if (transport.get('sourceAmount') != molecules[0].get('id') or
                transport.get('targetAmount') not in ids or transport.get('targetAmount') in branchids):
            raise ValueError('Tablet transports must target original external body states')
    # Absolute paths into this particular branch would require separately cloned
    # formula definitions. The five identified sources use relative local paths.
    local_formula_ids = {n.get(k) for n in nodes for k in FORMULA_FIELDS if n.get(k)}
    for fid in local_formula_ids:
        f = formulas.get(fid)
        if f is None or any(branch.get('name') in (p or '').split('|') for p in _resolve_paths(f, strings)):
            raise ValueError('Application formulas require unsupported absolute self-path remapping')
    for n in nodes:
        if n.tag == 'ObjectPath':
            p = strings.get(n.get('path'), n.get('path', ''))
            if branch.get('name') in p.split('|'):
                raise ValueError('Application event target has an absolute self path')
    aggregate = paths.get(rootname + '|' + compound + '|Total drug mass')
    if aggregate is None or aggregate.get('dim') != 'Amount' or not aggregate.get('formula'):
        raise ValueError('One explicit source dose aggregate is required')
    aggregate_formula = formulas.get(aggregate.get('formula'))
    expected_path = fullprefix + '|ProtocolSchemaItem|DrugMass'
    if (aggregate_formula is None or aggregate_formula.tag != 'Formula' or
            aggregate_formula.get('formula', '').strip() != 'P_1' or
            _resolve_paths(aggregate_formula, strings) != [expected_path]):
        raise ValueError('Source total drug mass must be the sole original DrugMass')
    groups = aggregate_formula.findall('Paths')
    if len(groups) != 1 or len(groups[0]) != 1 or groups[0][0].tag != 'Path':
        raise ValueError('Unreviewed aggregate formula structure')
    old_alias = strings.get(groups[0][0].get('as'), groups[0][0].get('as'))
    if old_alias != 'P_1':
        raise ValueError('Unreviewed aggregate alias')
    for n in model.find('Root').iter():
        if n in nodes:
            continue
        for k in LINKS:
            if n.get(k) in branchids:
                raise ValueError('An external model entity points into the application')
        for k in FORMULA_FIELDS:
            fid = n.get(k)
            if not fid:
                continue
            if fid == aggregate_formula.get('id') and n is not aggregate:
                raise ValueError('Dose aggregate formula is shared with another quantity')
            f = formulas.get(fid)
            if f is None:
                raise ValueError('Missing external formula')
            if any(branch.get('name') in (p or '').split('|') for p in _resolve_paths(f, strings)) and n is not aggregate:
                raise ValueError('Unregistered external formula depends on the application')
    return {'model': model, 'paths': paths, 'parents': parents, 'branch': branch,
            'formula': aggregate_formula, 'aggregate': aggregate, 'dose': dose, 'start': start,
            'original_dose_kg': original_dose, 'original_start_min': original_start,
            'fullprefix': fullprefix, 'rootname': rootname, 'branchids': branchids,
            'formula_path': groups[0][0], 'compound': compound, 'prefix': prefix}


def _set_scalar(node, value):
    node.attrib.pop('formula', None)
    node.set('value', format(value, '.17g'))


def _intern_string(model, value, added):
    """OSP ObjectPathXmlSerializer requires integer string-table references."""
    cache = model.find('FormulaCache')
    mapping = cache.find('StringMap')
    if mapping is None:
        raise ValueError('Native expansion requires an explicit formula string table')
    for row in mapping:
        if row.get('s') == value:
            return row.get('id')
    if any(not row.get('id', '').isdigit() for row in mapping):
        raise ValueError('String-table IDs must be native nonnegative integers')
    key = str(max([int(row.get('id')) for row in mapping], default=-1) + 1)
    row = ET.SubElement(mapping, 'Map', id=key, s=value)
    added.append(dict(row.attrib))
    return key


def _rename_clone(clone, idmap, oldname, newname):
    for n in clone.iter():
        if n.get('name') and n.get('id') in idmap:
            n.set('id', idmap[n.get('id')])
        for k in LINKS:
            if n.get(k) in idmap:
                n.set(k, idmap[n.get(k)])
        if n.tag == 'Tag' and n.get('value') == oldname:
            n.set('value', newname)
    clone.set('name', newname)


def expand_document(document, *, compound_name, event_prefix, administration_times_min, dose_mg):
    """Pure, bounded transformation; returns the full XML and exact edit plan."""
    _validate_spec(administration_times_min, dose_mg)
    src = _source_contract(document, compound_name, event_prefix)
    result = deepcopy(document)
    dst = _source_contract(result, compound_name, event_prefix)
    seed = hashlib.sha256(ET.tostring(document, encoding='utf-8')).hexdigest()
    parent = dst['parents'][dst['branch']]
    position = list(parent).index(dst['branch'])
    parent.remove(dst['branch'])
    records = []
    for index, time_min in enumerate(administration_times_min, 1):
        clone = deepcopy(src['branch'])
        name = f'Application_{index}'
        idmap = {old: old if index == 1 else str(uuid.uuid5(uuid.NAMESPACE_URL,
                         VERSION + ':' + seed + ':' + str(index) + ':' + old))
                 for old in src['branchids']}
        _rename_clone(clone, idmap, src['branch'].get('name'), name)
        local = {n.get('id'): n for n in clone.iter() if n.get('id')}
        for original, value in [(src['dose'], dose_mg / 1e6), (src['start'], float(time_min))]:
            # Preserve an unchanged scalar representation exactly for count-one
            # identity; changed scalars retain every non-value attribute.
            current = src['original_dose_kg'] if original is src['dose'] else src['original_start_min']
            if current != value:
                _set_scalar(local[idmap[original.get('id')]], value)
        parent.insert(position + index - 1, clone)
        records.append({'name': name, 'id_mapping': idmap,
                        'dose_kg': dose_mg / 1e6, 'start_min': float(time_min),
                        'prefix': event_prefix.rsplit('|', 1)[0] + '|' + name})
    added_strings = []
    if len(records) > 1:
        formula = dst['formula']
        formula.set('formula', ' + '.join(f'P_{i}' for i in range(1, len(records) + 1)))
        paths = formula.find('Paths')
        paths.clear()
        for index, record in enumerate(records, 1):
            path = deepcopy(src['formula_path'])
            path.set('path', _intern_string(dst['model'],
                src['rootname'] + '|' + record['prefix'] + '|ProtocolSchemaItem|DrugMass', added_strings))
            path.set('as', _intern_string(dst['model'], f'P_{index}', added_strings))
            paths.append(path)
    plan = {'version': VERSION, 'compound_name': compound_name, 'event_prefix': event_prefix,
            'administration_times_min': list(administration_times_min), 'dose_mg': dose_mg,
            'applications': records, 'added_string_map_entries': added_strings,
            'dose_aggregate_parameter_id': src['aggregate'].get('id'),
            'dose_aggregate_formula_id': src['formula'].get('id'),
            'original_dose_kg': src['original_dose_kg'], 'original_start_min': src['original_start_min'],
            'allowed_changes': ['complete application clones with internal IDs remapped',
                                'registered absolute dose and start scalars',
                                'dose-derived total drug mass sum'],
            'drug_parameters_fitted': False, 'physiology_rebuilt': False,
            'source_formulation_replaced': False, 'sealed_observations_accessed': False}
    audit = audit_expansion(document, result, plan)
    if not audit['accepted']:
        raise ValueError('Independent expansion audit failed: ' + '; '.join(audit['reasons']))
    return result, plan


def audit_expansion(original, expanded, plan):
    """Check each clone and reverse only registered edits before a full XML diff.

    Original formula definitions, external transport targets, all physiology,
    drug parameters and the entire configuration remain protected byte-semantically.
    The audit is deliberately stricter than ordinary serializer normalization.
    """
    failures = []
    try:
        if plan.get('version') != VERSION:
            raise ValueError('Unrecognized expansion plan')
        _validate_spec(plan['administration_times_min'], plan['dose_mg'])
        src = _source_contract(original, plan['compound_name'], plan['event_prefix'])
        reverse = deepcopy(expanded)
        model = _models(reverse)
        semantic = semantics.model_semantics(model)
        paths, ids, parents = _inventory(model)
        strings, formulas = _formula_inventory(model)
        n = len(plan['administration_times_min'])
        if len(plan['applications']) != n:
            raise ValueError('Application count differs from registered regimen')
        actual_branches = [x for x in model.find('Root').iter('EventGroup')
                           if any(c.get('name') == 'ProtocolSchemaItem' for c in x.findall('Children/Container'))]
        if len(actual_branches) != n:
            raise ValueError('Unregistered or missing dosing application')
        expected_paths = []
        repaired_parent, first_position = None, None
        for index, (record, registered_time) in enumerate(zip(plan['applications'], plan['administration_times_min']), 1):
            expected_name = f'Application_{index}'
            expected_prefix = plan['event_prefix'].rsplit('|', 1)[0] + '|' + expected_name
            if record['name'] != expected_name or record['prefix'] != expected_prefix:
                raise ValueError('Registered application names differ')
            prefix = src['rootname'] + '|' + expected_prefix
            branch = paths.get(prefix)
            if branch is None or branch not in actual_branches:
                raise ValueError('Registered branch missing')
            mapping = record['id_mapping']
            if set(mapping) != src['branchids'] or len(set(mapping.values())) != len(mapping):
                raise ValueError('ID remapping is not a bijection')
            if index == 1 and any(k != v for k, v in mapping.items()):
                raise ValueError('Original application identities changed')
            if index > 1 and set(mapping.values()) & set(_inventory(src['model'])[1]):
                raise ValueError('Cloned IDs collide with original model entities')
            dose = paths[prefix + '|ProtocolSchemaItem|Dose']
            start = paths[prefix + '|ProtocolSchemaItem|Start time']
            if (_scalar(dose, formulas) != plan['dose_mg'] / 1e6 or
                    _scalar(start, formulas) != float(registered_time) or
                    record['dose_kg'] != plan['dose_mg'] / 1e6 or record['start_min'] != float(registered_time)):
                raise ValueError('Dose or time differs from registered regimen')
            # Restore ONLY dose/time attributes, reverse internal IDs and names;
            # every remaining node and attribute must equal the source branch.
            restored = deepcopy(branch)
            reverse_ids = {v: k for k, v in mapping.items()}
            _rename_clone(restored, reverse_ids, expected_name, 'Application_1')
            byid = {x.get('id'): x for x in restored.iter() if x.get('id')}
            for source_param in (src['dose'], src['start']):
                node = byid.get(source_param.get('id'))
                if node is None:
                    raise ValueError('Clone scalar identity missing')
                a, b = dict(node.attrib), dict(source_param.attrib)
                for key in ('formula', 'value'):
                    a.pop(key, None); b.pop(key, None)
                if a != b:
                    raise ValueError('Unregistered scalar attributes changed')
                node.attrib.clear(); node.attrib.update(source_param.attrib)
            if _canonical(restored) != _canonical(src['branch']):
                raise ValueError('A cloned application mechanism or external target changed')
            parent = parents[branch]
            if repaired_parent is None:
                repaired_parent, first_position = parent, list(parent).index(branch)
            elif parent is not repaired_parent:
                raise ValueError('Applications are in different containers')
            expected_paths.append(prefix + '|ProtocolSchemaItem|DrugMass')
            parent.remove(branch)
        repaired_parent.insert(first_position, deepcopy(src['branch']))
        aggregate = paths.get(src['rootname'] + '|' + plan['compound_name'] + '|Total drug mass')
        if aggregate is None or _canonical(aggregate) != _canonical(src['aggregate']):
            raise ValueError('Dose aggregate parameter itself changed')
        formula = formulas.get(plan['dose_aggregate_formula_id'])
        if (plan['dose_aggregate_formula_id'] != src['formula'].get('id') or
                plan['dose_aggregate_parameter_id'] != src['aggregate'].get('id') or formula is None or
                formula.get('formula') != ' + '.join(f'P_{i}' for i in range(1, n + 1)) or
                _resolve_paths(formula, strings) != expected_paths):
            raise ValueError('Dose aggregate expression or paths do not cover exactly the registered doses')
        expected_formula = deepcopy(src['formula'])
        if n > 1:
            expected_formula.set('formula', ' + '.join(f'P_{i}' for i in range(1, n + 1)))
            group = expected_formula.find('Paths'); group.clear()
            for i, path in enumerate(expected_paths, 1):
                node = deepcopy(src['formula_path']); node.set('path', path); node.set('as', f'P_{i}'); group.append(node)
        # Compare path semantics independently of newly allocated string IDs.
        normalized_formula = deepcopy(formula)
        normalized_expected = deepcopy(expected_formula)
        source_strings, _ = _formula_inventory(src['model'])
        for f, mapping in ((normalized_formula, strings), (normalized_expected, source_strings)):
            for path in f.iter('Path'):
                for key in ('path', 'as', 'dim'):
                    if path.get(key) in mapping:
                        path.set(key, mapping[path.get(key)])
        if _canonical(normalized_expected) != _canonical(normalized_formula):
            raise ValueError('Unregistered aggregate formula attributes or aliases changed')
        old_map = src['model'].find('FormulaCache/StringMap')
        new_map = model.find('FormulaCache/StringMap')
        old_rows = list(old_map) if old_map is not None else []
        new_rows = list(new_map) if new_map is not None else []
        if [_canonical(x) for x in new_rows[:len(old_rows)]] != [_canonical(x) for x in old_rows]:
            raise ValueError('Original formula string table changed')
        additions = new_rows[len(old_rows):]
        expected_added = plan.get('added_string_map_entries', [])
        if [dict(x.attrib) for x in additions] != expected_added:
            raise ValueError('Unregistered string-table additions')
        permitted_strings = set(expected_paths) | {f'P_{i}' for i in range(1, n + 1)}
        used = {v for x in formula.iter('Path') for k, v in x.attrib.items() if k in {'path', 'as'}}
        for row in additions:
            if (row.tag != 'Map' or set(row.attrib) != {'id', 's'} or len(row) or
                    row.get('s') not in permitted_strings or row.get('id') not in used):
                raise ValueError('An added formula string is not a registered dose aggregate reference')
            new_map.remove(row)
        group = model.find('FormulaCache/Formulas')
        position = list(group).index(formula); group.remove(formula); group.insert(position, deepcopy(src['formula']))
        if _canonical(reverse) != _canonical(original):
            raise ValueError('A nonadministration Model node, formula, parameter or configuration changed')
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        failures.append(type(exc).__name__ + ':' + str(exc))
        semantic = None
    return {'version': VERSION, 'accepted': not failures, 'reasons': failures,
            'full_nonadministration_xml_and_configuration_unchanged': not failures,
            'independent_cloned_application_audit': not failures,
            'expected_application_count': len(plan.get('administration_times_min', [])),
            'expanded_model_semantics_sha256': semantic['sha256'] if semantic else None,
            'ode_executed': False, 'scientific_applicability_claimed': False}


def _read_xml(path):
    if path.suffix != '.pkml' or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('A bounded native PKML is required')
    data = path.read_bytes()
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('XML entities are forbidden')
    doc = ET.fromstring(data)
    if any(n.tag in {'ObservedData', 'DataRepository', 'IndividualResults', 'SimulationResults'} for n in doc.iter()):
        raise ValueError('Outcome-bearing source documents are forbidden')
    return doc


def _original_document(reference, root):
    source = inside(reference['path'], root)
    if source.suffix != '.pksim5' or sha(source) != reference['sha256']:
        raise ValueError('Original project SHA differs')
    with sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                          'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?',
                          (reference['simulation_id'],)).fetchall()
    if len(rows) != 1:
        raise ValueError('Original simulation identity is ambiguous')
    blob = rows[0][0]
    blobsha = hashlib.sha256(blob).hexdigest()
    if blobsha != reference['source_blob_sha256']:
        raise ValueError('Original saved Model blob SHA differs')
    saved = ET.fromstring(decompress_model_blob(blob))
    if saved.get('id') != reference['simulation_id']:
        raise ValueError('Saved document and original simulation identity differ')
    if sha(source) != reference['sha256']:
        raise ValueError('Original project changed during audit')
    return saved


def _verify_original_source(document, reference, root):
    saved = _original_document(reference, root)
    model_audit = semantics.compare_saved_model_xml(saved.find('Model'), _models(document))
    from .source_pkml import saved_settings
    settings_audit = semantics.compare_saved_settings_xml(saved_settings(saved),
                    saved_settings(document.find('.//SimulationConfiguration')))
    if not model_audit['accepted'] or not settings_audit['accepted']:
        raise ValueError('Input PKML does not preserve the original saved Model and Settings')
    return {'source': dict(reference), 'model_accepted': True, 'settings_accepted': True,
            'source_model_semantics_sha256': semantics.model_semantics(saved.find('Model'))['sha256']}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_source_expansion(source_reference, source_metadata, *, compound_name,
                             event_prefix, administration_times_min, dose_mg, project_root=ROOT):
    """Read-only eligibility from the exact original saved Model, with no export.

    This freezes a structural recipe, not identities from a later native export.
    The original metadata remains a one-application source record.
    """
    root = Path(project_root).resolve(strict=True)
    _validate_spec(administration_times_min, dose_mg)
    if len(administration_times_min) <= 1:
        raise ValueError('An expansion requires more than one target administration')
    saved = _original_document(source_reference, root)
    actual_metadata = describe_saved_simulation(saved)
    if actual_metadata != source_metadata:
        raise ValueError('Expansion eligibility metadata differs from the exact saved source')
    document, _ = wrap_preserved_model(saved)
    source = _source_contract(document, compound_name, event_prefix)
    semantic = semantics.model_semantics(source['model'])
    return {'version': VERSION, 'verified': True, 'source_reference': dict(source_reference),
            'source_metadata_sha256': _digest(source_metadata),
            'compound_name': compound_name, 'event_prefix': event_prefix,
            'original_source_event_count': 1, 'executable_event_count': len(administration_times_min),
            'administration_times_min': list(administration_times_min), 'dose_mg': dose_mg,
            'source_model_semantics_sha256': semantic['sha256'],
            'original_application_xml_sha256': element_sha(source['branch']),
            'source_dose_aggregate_formula_xml_sha256': element_sha(source['formula']),
            'mechanism': 'replicate the complete original tablet application, local state and all12 release transports; preserve original body targets and shared relative formulas',
            'allowed_nonbranch_change': 'the sole total-drug-mass sum of all registered application DrugMass terms',
            'eligibility_basis': 'exact full original Model structure, IDs, formula paths and once-only StartTime trigger; not a metadata count override',
            'native_execution_required': True, 'fitting_performed': False,
            'sealed_observations_accessed': False, 'scientific_applicability_claimed': False}


def verify_expansion_receipt(receipt_path, *, expected_policy=None, project_root=ROOT):
    """Recheck immutable source, current implementation and every expansion node.

    Intended for the scoring barrier as well as the execution adapter; it never
    reads clinical outcomes or invokes a native tool.
    """
    root = Path(project_root).resolve(strict=True)
    receipt_path = inside(receipt_path, root)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('version') != VERSION or receipt.get('status') != 'expanded_pending_native_readback_and_ode':
        raise ValueError('Unrecognized or unverified expansion receipt')
    current = {name: sha(inside(name, ROOT)) for name in IMPLEMENTATION_FILES}
    if receipt.get('implementation_sha256') != current:
        raise ValueError('Expansion implementation has changed')
    for ref in (receipt['source_pkml'], receipt['pkml']):
        if sha(inside(ref['path'], root)) != ref['sha256']:
            raise ValueError('Expansion source or repaired PKML SHA differs')
    for path, expected in receipt['files'].items():
        if sha(inside(path, root)) != expected:
            raise ValueError('An expansion evidence file differs')
    required = {str((receipt_path.parent / name).relative_to(root)) for name in
                ('expansion_plan.json', 'semantic_audit.json', 'source_audit.json')}
    if set(receipt['files']) != required:
        raise ValueError('Expansion evidence inventory differs')
    plan = json.loads((receipt_path.parent / 'expansion_plan.json').read_text())
    original = _read_xml(inside(receipt['source_pkml']['path'], root))
    expanded = _read_xml(inside(receipt['pkml']['path'], root))
    source_audit = _verify_original_source(original, receipt['source_reference'], root)
    if source_audit != json.loads((receipt_path.parent / 'source_audit.json').read_text()):
        raise ValueError('Recorded source audit differs from the current full-source proof')
    audit = audit_expansion(original, expanded, plan)
    if not audit['accepted'] or audit != json.loads((receipt_path.parent / 'semantic_audit.json').read_text()):
        raise ValueError('Recorded expansion does not pass independent full-XML audit')
    if expected_policy is not None:
        if (expected_policy.get('verified') is not True or expected_policy.get('version') != VERSION or
                expected_policy.get('source_reference') != receipt['source_reference'] or
                expected_policy.get('original_source_event_count') != 1 or
                expected_policy.get('executable_event_count') != len(plan['applications'])):
            raise ValueError('Frozen expansion policy identity or counts differ')
        for key in ('compound_name', 'event_prefix', 'administration_times_min', 'dose_mg'):
            if expected_policy.get(key) != plan[key]:
                raise ValueError('Frozen expansion policy differs: ' + key)
        saved = _original_document(receipt['source_reference'], root)
        fresh = prepare_source_expansion(receipt['source_reference'], describe_saved_simulation(saved),
            compound_name=plan['compound_name'], event_prefix=plan['event_prefix'],
            administration_times_min=plan['administration_times_min'], dose_mg=plan['dose_mg'], project_root=root)
        if fresh != expected_policy:
            raise ValueError('Frozen original-source structural qualification differs')
    return {'accepted': True, 'receipt': {'path': str(receipt_path.relative_to(root)), 'sha256': sha(receipt_path)},
            'source_model_and_settings_verified': True, 'full_expansion_xml_verified': True,
            'source_event_count': 1, 'executable_event_count': len(plan['applications']),
            'expanded_model_semantics_sha256': audit['expanded_model_semantics_sha256'],
            'scientific_applicability_claimed': False}


def executable_metadata(receipt_path, original_source_metadata, *, expected_policy, project_root=ROOT):
    """Extract actual expanded Model entities; never relabel original metadata."""
    root = Path(project_root).resolve(strict=True)
    verification = verify_expansion_receipt(receipt_path, expected_policy=expected_policy, project_root=root)
    receipt = json.loads(inside(receipt_path, root).read_text())
    original = _original_document(receipt['source_reference'], root)
    if describe_saved_simulation(original) != original_source_metadata:
        raise ValueError('Original source metadata changed before executable extraction')
    expanded = _read_xml(inside(receipt['pkml']['path'], root))
    original.remove(original.find('Model'))
    original.append(deepcopy(_models(expanded)))
    actual = describe_saved_simulation(original)
    keys = ('event_parameters', 'administrations', 'observer_paths', 'parent_molecular_weight_parameters',
            'parameter_count', 'numeric_convention')
    metadata = {key: actual[key] for key in keys}
    metadata.update(version=VERSION, metadata_scope='actual_expanded_executable_model_only',
                    original_source_metadata_sha256=_digest(original_source_metadata),
                    original_source_event_count=1, executable_event_count=verification['executable_event_count'],
                    expanded_pkml=receipt['pkml'], source_reference=receipt['source_reference'],
                    expansion_receipt=verification['receipt'])
    if len(metadata['administrations']) != metadata['executable_event_count']:
        raise ValueError('Expanded executable administration count differs')
    return metadata, verification


def expand_applications(source_pkml, output_dir, *, source_reference, compound_name,
                        event_prefix, administration_times_min, dose_mg, project_root=ROOT):
    """Write a new repaired PKML, full audit, source/implementation-bound receipt.

    source_reference requires path, sha256, simulation_id, source_blob_sha256.
    This performs no native call. Native import/readback and ODE are later gates.
    """
    root = Path(project_root).resolve(strict=True)
    source = inside(source_pkml, root)
    output = inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError('Preserve every expansion attempt in a new directory')
    document = _read_xml(source)
    source_sha = sha(source)
    implementation = {name: sha(inside(name, ROOT)) for name in IMPLEMENTATION_FILES}
    provenance = _verify_original_source(document, source_reference, root)
    expanded, plan = expand_document(document, compound_name=compound_name, event_prefix=event_prefix,
                          administration_times_min=administration_times_min, dose_mg=dose_mg)
    audit = audit_expansion(document, expanded, plan)
    if not audit['accepted']:
        raise ValueError('Expansion audit failed')
    output.mkdir(parents=True)
    pkml = output / 'repeated_application.pkml'
    ET.ElementTree(expanded).write(pkml, encoding='utf-8', xml_declaration=True)
    # Audit the serialized artifact, not only the in-memory tree.
    serialized = audit_expansion(document, _read_xml(pkml), plan)
    if not serialized['accepted']:
        raise ValueError('Serialized expansion audit failed')
    if sha(source) != source_sha or implementation != {name: sha(inside(name, ROOT)) for name in IMPLEMENTATION_FILES}:
        raise ValueError('Source or implementation changed during expansion')
    save(output / 'expansion_plan.json', plan)
    save(output / 'semantic_audit.json', serialized)
    save(output / 'source_audit.json', provenance)
    receipt = {'version': VERSION, 'status': 'expanded_pending_native_readback_and_ode',
        'source_pkml': {'path': str(source.relative_to(root)), 'sha256': source_sha},
        'source_reference': dict(source_reference), 'implementation_sha256': implementation,
        'pkml': {'path': str(pkml.relative_to(root)), 'sha256': sha(pkml)},
        'files': {str((output/name).relative_to(root)): sha(output/name) for name in
                  ['expansion_plan.json', 'semantic_audit.json', 'source_audit.json']},
        'fitting_performed': False, 'ode_executed': False, 'sealed_observations_accessed': False,
        'scientific_applicability_claimed': False}
    save(output / 'receipt.json', receipt)
    return receipt
