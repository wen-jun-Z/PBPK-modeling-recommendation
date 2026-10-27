"""Evidence-bound corrections of source context, without rebuilding the drug.

Changing a documented sex or selecting an existing solubility alternative is
not a guarantee of predictive improvement. Unknown clinical demographics stay
explicitly unknown; source assumptions are recorded rather than invented.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from .source_adaptation import _check_dependencies, _diff, _digest, _named


def _evidence(reference, root):
    if not isinstance(reference, dict) or not all(reference.get(k) for k in ('path', 'sha256', 'locator')):
        raise ValueError('Context change requires a source path, SHA256 and locator')
    path = (root / reference['path']).resolve(strict=True)
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError('Context evidence must belong to the project')
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Context evidence checksum mismatch')
    return deepcopy(reference)


def apply_source_context(blueprint, context, project_root):
    """Correct documented sex or test a source-supported solubility hypothesis.

    context={human_sex:{value:'FEMALE',evidence:{path,sha256,locator}},
             solubility_alternative:{value:'FeSSIF',evidence:{...}}}
    Only explicit inputs are changed. The snapshot's original source and its
    fitted drug processes remain intact. No calibration is performed here.
    """
    root = Path(project_root).resolve(strict=True)
    if not isinstance(blueprint, dict) or blueprint.get('status') != 'adapted':
        raise ValueError('A successful full source adaptation is required')
    original = blueprint['snapshot']
    if _digest(original) != blueprint['manifest']['adapted_snapshot_sha256']:
        raise ValueError('Adapted source snapshot changed before context correction')
    result = {'status': 'blocked', 'reasons': [], 'scientific_validated': False}
    try:
        if not isinstance(context, dict) or set(context) - {'human_sex', 'solubility_alternative'}:
            raise ValueError('Unsupported context field; do not invent age, body weight or disease mechanisms')
        snapshot = deepcopy(original)
        sim = snapshot['Simulations'][0]
        compound = sim['Compounds'][0]['Name']
        evidence = {key: _evidence(value['evidence'], root) for key, value in context.items()}
        assumptions = []
        if 'human_sex' in context:
            sex = context['human_sex']['value']
            if sex not in {'MALE', 'FEMALE'}:
                raise ValueError('Documented PK-Sim sex must be MALE or FEMALE')
            individual = _named(snapshot['Individuals'], sim['Individual'], 'Individuals')
            if individual['OriginData']['Gender'] != sex:
                # Preserved source-specific physiology values could prevent the
                # requested physiology from taking effect. Fail rather than
                # guessing which arbitrary overrides remain appropriate.
                if individual.get('Parameters'):
                    raise ValueError('Demographic correction conflicts with explicit source individual physiology overrides')
                for override in sim.get('Parameters', []):
                    parts = override['Path'].split('|')
                    if parts[0] not in {'Events', 'Applications'} and compound not in parts:
                        raise ValueError('Demographic correction conflicts with source simulation physiology overrides')
                clone = deepcopy(individual)
                clone['Name'] += '_requested_' + sex
                if any(row['Name'] == clone['Name'] for row in snapshot['Individuals']):
                    raise ValueError('Requested individual name already exists')
                clone['OriginData']['Gender'] = sex
                snapshot['Individuals'].append(clone)
                sim['Individual'] = clone['Name']
                assumptions.append({'field': 'other_demographics', 'status': 'source_assumptions_retained',
                                    'origin_data': deepcopy(clone['OriginData']),
                                    'weight_height': 'source_explicit_values_or_PKSim_defaults; not measured patient data'})
        if 'solubility_alternative' in context:
            selection = sim['Compounds'][0]
            drug = _named(snapshot['Compounds'], compound, 'Compounds')
            alternative = context['solubility_alternative']['value']
            _named(drug.get('Solubility'), alternative, 'Solubility')
            choices = [p for p in selection.get('Alternatives', []) if p['GroupName'] == 'COMPOUND_SOLUBILITY']
            if len(choices) != 1:
                raise ValueError('An explicit source solubility binding is required')
            if any(p['Path'].startswith(compound + '|') and p['Path'].split('|')[-1] in
                   {'Solubility at reference pH', 'Reference pH'} for p in sim.get('Parameters', [])):
                raise ValueError('Source solubility override conflicts with alternative selection')
            choices[0]['AlternativeName'] = alternative
            assumptions.append({'field': 'solubility_alternative', 'status': 'mechanistic_hypothesis_not_validated',
                                'note': 'In-vitro medium selection alone does not establish in-vivo food effects.'})
        checks = _check_dependencies(snapshot)
        if snapshot['Compounds'] != original['Compounds']:
            raise ValueError('Context correction must preserve the complete drug building blocks')
        result.update(status='adapted', snapshot=snapshot, diff=_diff(original, snapshot),
                      manifest={'adapter_version': 'source-context-v1', 'source_blueprint': deepcopy(blueprint['manifest']),
                                'context': deepcopy(context), 'evidence': evidence,
                                'adapted_snapshot_sha256': _digest(snapshot), 'assumptions': assumptions,
                                'parameters_fitted': False, 'clinical_qualified': False}, checks=checks)
    except (ValueError, KeyError, TypeError) as error:
        result['reasons'] = [str(error)]
    return result


def install_calibrated_parameters(snapshot, specs, values, calibration_reference):
    """Persist selected native parameter values as explicit simulation overrides.

    The complete source model is retained and a new native run must verify these
    values. Clinical data are not part of this function's inputs.
    """
    if len(specs) != len(values) or not specs or not calibration_reference:
        raise ValueError('Selected parameters and their calibration record are required')
    result = deepcopy(snapshot)
    overrides = result['Simulations'][0].setdefault('Parameters', [])
    for spec, value in zip(specs, values):
        if not spec['lower_bound'] <= value <= spec['upper_bound']:
            raise ValueError('Selected parameter leaves its registered bounds')
        matching = [p for p in overrides if p['Path'] == spec['path']]
        if len(matching) > 1:
            raise ValueError('Ambiguous simulation override')
        row = matching[0] if matching else {'Path': spec['path']}
        row.update(Value=value, Unit=spec['unit'], ValueOrigin={
            'Source': 'ParameterIdentification', 'Method': 'ParameterIdentification',
            'Description': str(calibration_reference)})
        if not matching:
            overrides.append(row)
    return result
