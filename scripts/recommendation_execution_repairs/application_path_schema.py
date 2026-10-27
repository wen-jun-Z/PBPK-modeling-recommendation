"""Reversible administration-path rename in a conversion copy, never source values.

This only supplies a proposed v80 input. Acceptance still requires native
parameter readback and complete original/restored Model and Settings equality.
"""
from copy import deepcopy


def normalize_application_paths(snapshot):
    if snapshot.get('Version') != 80:
        raise ValueError('APPLICATION_PATH_TARGET_VERSION_80_REQUIRED')
    result = deepcopy(snapshot)
    changes = []
    names = set()
    for simulation in result.get('Simulations', []):
        name = simulation['Name']
        if name in names:
            raise ValueError('APPLICATION_PATH_DUPLICATE_SIMULATION')
        names.add(name)
        paths = set()
        for parameter in simulation.get('Parameters', []):
            before = deepcopy(parameter)
            path = parameter['Path']
            if path.startswith('Applications|'):
                parameter['Path'] = 'Events|' + path[len('Applications|'):]
                changes.append({'simulation': name, 'before': before, 'after': deepcopy(parameter)})
            if parameter['Path'] in paths:
                raise ValueError('APPLICATION_PATH_COLLISION')
            paths.add(parameter['Path'])
    inverse = deepcopy(result)
    by_name = {s['Name']: s for s in inverse.get('Simulations', [])}
    for change in changes:
        rows = by_name[change['simulation']]['Parameters']
        matches = [i for i, r in enumerate(rows) if r['Path'] == change['after']['Path']]
        if len(matches) != 1 or rows[matches[0]] != change['after']:
            raise ValueError('APPLICATION_PATH_INVERSE_BINDING_DIFFERS')
        rows[matches[0]] = deepcopy(change['before'])
    if inverse != snapshot:
        raise ValueError('APPLICATION_PATH_NONPATH_CHANGE')
    return result, changes
