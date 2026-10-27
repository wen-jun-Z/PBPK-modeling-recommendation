"""Prove surplus saved-model administrations cannot trigger in the output window.

No inference is made from event names. A supported application has an explicit
Time = StartTime trigger bound to its actual protocol parameter. Other events
must be gated by a private, initially zero state whose only activating writers
are those time-bound events. Unknown event semantics fail closed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET

from . import source_pkml

VERSION = 'source-bound-surplus-event-time-barrier-v1'


def _normalize(path):
    return re.sub(r'^Applications\|', 'Events|', path)


def _index(model):
    root = model.find('Root')
    if root is None:
        raise ValueError('SURPLUS_COMPLETE_MODEL_ROOT_REQUIRED')
    by_path, by_id, owner = {}, {}, {}

    def walk(node, path, event=None):
        if node is not root and node.get('name') and node.get('id'):
            path = path + [node.get('name')]
            key = '|'.join(path)
            if key in by_path or node.get('id') in by_id:
                raise ValueError('SURPLUS_AMBIGUOUS_MODEL_ENTITY')
            by_path[key] = node
            by_id[node.get('id')] = key
        if node.tag == 'Event':
            event = '|'.join(path)
        if node.tag == 'EventAssignment':
            owner[node.get('id')] = event
        for child in node:
            walk(child, path, event)

    walk(root, [])
    cache = model.find('FormulaCache')
    if cache is None or cache.find('Formulas') is None:
        raise ValueError('SURPLUS_FORMULA_CACHE_REQUIRED')
    definitions = list(cache.find('Formulas'))
    formulas = {node.get('id'): node for node in definitions}
    if None in formulas or len(formulas) != len(definitions):
        raise ValueError('SURPLUS_UNIQUE_FORMULA_IDS_REQUIRED')
    mapping = cache.find('StringMap')
    strings = {node.get('id'): node.get('s') for node in mapping} if mapping is not None else {}
    if mapping is not None and len(strings) != len(mapping):
        raise ValueError('SURPLUS_UNIQUE_STRING_IDS_REQUIRED')
    return by_path, by_id, owner, formulas, strings


def _scalar(node, formulas):
    if node.get('rhs') or node.get('rHSFormula'):
        raise ValueError('SURPLUS_DYNAMIC_PARAMETER_NOT_SUPPORTED')
    values = [float(node.get('value'))] if node.get('value') is not None else []
    if node.get('formula'):
        formula = formulas.get(node.get('formula'))
        if formula is None or formula.tag != 'ConstantFormula' or len(formula):
            raise ValueError('SURPLUS_EXPLICIT_CONSTANT_PARAMETER_REQUIRED')
        values.append(float(formula.get('value')))
    if not values or not all(math.isfinite(v) and v == values[0] for v in values):
        raise ValueError('SURPLUS_UNAMBIGUOUS_FINITE_CONSTANT_REQUIRED')
    return values[0]


def _resolve(raw, context, paths, strings):
    raw = strings.get(raw, raw)
    if not isinstance(raw, str):
        raise ValueError('SURPLUS_EXPLICIT_FORMULA_PATH_REQUIRED')
    if raw in paths:
        return raw
    parts = context.split('|')
    tokens = raw.split('|')
    if not tokens or tokens[0] not in {'.', '..'}:
        # Saved absolute paths may include the simulation root's name.
        matches = [path for path in paths if raw.endswith('|' + path)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError('SURPLUS_FORMULA_PATH_NOT_RESOLVED:' + raw)
    for token in tokens:
        if token == '..':
            if not parts:
                raise ValueError('SURPLUS_FORMULA_PATH_ESCAPES_MODEL')
            parts.pop()
        elif token != '.':
            parts.append(token)
    result = '|'.join(parts)
    if result not in paths:
        raise ValueError('SURPLUS_FORMULA_PATH_NOT_IN_MODEL:' + result)
    return result


def _formula_paths(formula, event_path, paths, strings):
    resolved, times = {}, []
    container = formula.find('Paths')
    for node in container if container is not None else []:
        if node.tag == 'TimePath':
            alias = strings.get(node.get('as'), node.get('as')) or 'Time'
            times.append(alias)
        elif node.tag == 'Path':
            alias = strings.get(node.get('as'), node.get('as'))
            if not alias or alias in resolved:
                raise ValueError('SURPLUS_UNIQUE_FORMULA_ALIASES_REQUIRED')
            resolved[alias] = _resolve(node.get('path'), event_path, paths, strings)
        else:
            raise ValueError('SURPLUS_UNSUPPORTED_FORMULA_PATH_TYPE')
    return resolved, times


def _and_guard(expression):
    """Accept a top-level '(state = 1) AND (...)' with a fully enclosed RHS."""
    match = re.fullmatch(r'\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*1\s*\)\s+AND\s+(\(.*\))', expression)
    if not match:
        return None
    right = match[2]
    depth = 0
    for index, char in enumerate(right):
        depth += (char == '(') - (char == ')')
        if depth < 0 or (depth == 0 and index != len(right) - 1):
            return None
    return match[1] if depth == 0 else None


def prove_application_barrier(model, application_path, start_parameter_id):
    paths, ids, owners, formulas, strings = _index(model)
    original = [path for path in paths if _normalize(path) == _normalize(application_path)]
    if len(original) != 1:
        raise ValueError('SURPLUS_EXACT_ORIGINAL_APPLICATION_REQUIRED')
    prefix = original[0]
    group = paths[prefix]
    if group.tag != 'EventGroup':
        raise ValueError('SURPLUS_ORIGINAL_EVENT_GROUP_REQUIRED')
    start = prefix + '|ProtocolSchemaItem|Start time'
    if start not in paths or paths[start].get('id') != start_parameter_id or paths[start].get('dim') != 'Time':
        raise ValueError('SURPLUS_START_PARAMETER_ID_OR_DIMENSION_DIFFERS')
    if paths[start].tag not in {'Para', 'Parameter'}:
        raise ValueError('SURPLUS_STATIC_START_PARAMETER_REQUIRED')
    _scalar(paths[start], formulas)
    if any(node.get('changedEntity') == start_parameter_id for node in model.iter('EventAssignment')):
        raise ValueError('SURPLUS_START_PARAMETER_HAS_EVENT_WRITER')
    events = {path: node for path, node in paths.items()
              if path.startswith(prefix + '|') and node.tag == 'Event'}
    if not events:
        raise ValueError('SURPLUS_ORIGINAL_APPLICATION_EVENTS_REQUIRED')
    timed, gated = {}, {}
    for path, event in events.items():
        formula = formulas.get(event.get('formula'))
        if formula is None or formula.tag != 'Formula' or not formula.get('formula'):
            raise ValueError('SURPLUS_EXPLICIT_EVENT_CONDITION_REQUIRED')
        refs, times = _formula_paths(formula, path, paths, strings)
        compact = re.sub(r'\s+', '', formula.get('formula'))
        infusion = prefix + '|ProtocolSchemaItem|Infusion time'
        delayed_stop = (compact == 'Time=StartTime+InfusionTime' and
                        refs == {'StartTime': start, 'InfusionTime': infusion} and times == ['Time'])
        if delayed_stop:
            duration = paths.get(infusion)
            if (duration is None or duration.tag not in {'Para', 'Parameter'} or duration.get('dim') != 'Time'
                    or _scalar(duration, formulas) < 0 or any(n.get('changedEntity') == duration.get('id')
                                                            for n in model.iter('EventAssignment'))):
                raise ValueError('SURPLUS_INFUSION_STOP_DURATION_NOT_STATIC_NONNEGATIVE')
        if ((compact == 'Time=StartTime' and refs == {'StartTime': start} and times == ['Time']) or delayed_stop):
            timed[path] = {'event_id': event.get('id'), 'event_path': path,
                           'condition': formula.get('formula'), 'condition_sha256': source_pkml.element_sha(formula),
                           'resolved_start_parameter': start, 'resolved_start_parameter_id': start_parameter_id,
                           'start_time_can_trigger_only_at_registered_time': not delayed_stop,
                           'event_cannot_precede_registered_start': True,
                           'nonnegative_infusion_offset_min': _scalar(paths[infusion], formulas) if delayed_stop else 0.0}
        else:
            alias = _and_guard(formula.get('formula'))
            guard = refs.get(alias) if alias else None
            if (not guard or not guard.startswith(prefix + '|')
                    or paths[guard].tag not in {'Para', 'Parameter'}
                    or _scalar(paths[guard], formulas) != 0.0):
                raise ValueError('SURPLUS_EVENT_NOT_PROVABLY_INACTIVE_BEFORE_START:' + path)
            gated[path] = {'event_id': event.get('id'), 'event_path': path,
                           'condition_sha256': source_pkml.element_sha(formula),
                           'guard_parameter': guard, 'guard_parameter_id': paths[guard].get('id'),
                           'guard_entity_type': paths[guard].tag,
                           'guard_initial_value': 0.0}
    if not timed:
        raise ValueError('SURPLUS_NO_TIME_BOUND_APPLICATION_TRIGGER')
    # Prove each gate stays zero: no ODE RHS, and every nonzero writer belongs
    # to one of the proven future time triggers. Zero resets cannot activate it.
    for row in gated.values():
        writers = []
        for assignment in model.iter('EventAssignment'):
            if assignment.get('changedEntity') != row['guard_parameter_id']:
                continue
            event_path = owners.get(assignment.get('id'))
            formula = formulas.get(assignment.get('formula'))
            zero_reset = (formula is not None and formula.tag == 'ConstantFormula'
                          and not len(formula) and float(formula.get('value')) == 0.0)
            if event_path not in timed and not zero_reset:
                raise ValueError('SURPLUS_PRIVATE_GATE_HAS_UNSAFE_EXTERNAL_WRITER')
            writers.append({'assignment_id': assignment.get('id'), 'event_path': event_path,
                            'kind': 'future_time_trigger' if event_path in timed else 'constant_zero_reset'})
        if not writers:
            raise ValueError('SURPLUS_PRIVATE_GATE_WRITERS_UNRESOLVED')
        row['all_model_writers'] = writers
    return {'application_path': _normalize(prefix), 'application_node_sha256': source_pkml.element_sha(group),
            'time_bound_events': list(timed.values()), 'inactive_guarded_events': list(gated.values()),
            'all_application_events_accounted': len(timed) + len(gated) == len(events),
            'verified': True}


def source_suppression_policy(source, source_sha, simulation_id, source_blob_sha,
                              metadata, contract, *, project_root):
    root = Path(project_root).resolve(strict=True)
    source = source_pkml.inside(source, root)
    if source_pkml.sha(source) != source_sha:
        raise ValueError('SURPLUS_ORIGINAL_SOURCE_SHA_MISMATCH')
    with sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                          'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?', (simulation_id,)).fetchall()
    if len(rows) != 1 or hashlib.sha256(rows[0][0]).hexdigest() != source_blob_sha:
        raise ValueError('SURPLUS_SELECTED_MODEL_BLOB_SHA_MISMATCH')
    document = ET.fromstring(source_pkml.decompress_model_blob(rows[0][0]))
    if document.get('id') != simulation_id:
        raise ValueError('SURPLUS_SOURCE_SIMULATION_ID_DIFFERS')
    model = document.find('Model')
    paths, _, _, formulas, _ = _index(model)
    starts = []
    for row in metadata['event_parameters']:
        if not row['path'].endswith('|ProtocolSchemaItem|Start time'):
            continue
        path = row['path']
        if path not in paths or paths[path].get('id') != row['id']:
            raise ValueError('SURPLUS_SOURCE_METADATA_PARAMETER_DIFFERS')
        starts.append((_scalar(paths[path], formulas), _normalize(path.rsplit('|ProtocolSchemaItem|', 1)[0]), row))
    starts.sort(key=lambda row: (row[0], row[1]))
    target_count = len(contract['administration_times_min'])
    end = contract['absolute_sampling_times_min'][-1]
    if not math.isfinite(end) or end < 0 or target_count > len(starts):
        raise ValueError('SURPLUS_FINITE_FROZEN_WINDOW_AND_DOSE_COUNT_REQUIRED')
    events = []
    for index, (_, prefix, row) in enumerate(starts[target_count:]):
        future = end + 60.0 + index
        if not math.isfinite(future) or future <= end:
            raise ValueError('SURPLUS_STRICTLY_FUTURE_EVENT_TIME_REQUIRED')
        events.append({'event_prefix': prefix, 'start_parameter_id': row['id'],
                       'start_parameter_path': _normalize(row['path']), 'new_start_time_min': future,
                       'dose_mg': 0.0, 'barrier': prove_application_barrier(model, prefix, row['id'])})
    if source_pkml.sha(source) != source_sha:
        raise ValueError('SURPLUS_SOURCE_CHANGED_DURING_PROOF')
    return {'version': VERSION, 'verified': True, 'source_sha256': source_sha,
            'source_model_blob_sha256': source_blob_sha, 'simulation_id': simulation_id,
            'source_model_sha256': source_pkml.element_sha(model),
            'output_end_min': end, 'timing_rule': 'output_end_min + 60 min + zero-based surplus index',
            'ordered_source_event_prefixes': [row[1] for row in starts],
            'target_event_count': target_count, 'events': events,
            'original_source_unchanged': True, 'observations_read': False,
            'ode_run': False, 'scientific_validity_established': False}
