"""Source-bound clinical dose normalization for preserved reference models.

No clinical observations, individual outcomes or model fitting are accessed.
Mean-weight dosing is an explicitly registered representative-dose approximation;
it does not change the original model's physiology or simulate a population.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))

from recommendation_candidate_executor import Blocked, _digest, _inside, _name, scenario_contract as original_contract

VERSION = 'source-bound-clinical-dose-contract-v1'
WEIGHT_RULE = 'Dose actual representative/population weight, not fixed 200 mg vial amount.'


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Blocked('FINITE_POSITIVE_REGISTERED_' + label + '_REQUIRED')
    return Decimal(str(value))


def _sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def _registered_weight(population):
    """Only a clinical mean explicitly encoded in kg, never a midpoint/SD."""
    if not isinstance(population, dict): raise Blocked('FROZEN_MEAN_BODY_WEIGHT_KG_REQUIRED')
    values = []
    if population.get('weight_mean_kg') is not None:
        values.append(('weight_mean_kg', _positive(population['weight_mean_kg'], 'MEAN_WEIGHT_KG')))
    structured = population.get('body_weight_kg')
    if isinstance(structured, dict) and structured.get('mean') is not None:
        if structured.get('unit', 'kg') != 'kg': raise Blocked('MEAN_WEIGHT_UNIT_CONFLICT')
        values.append(('body_weight_kg.mean', _positive(structured['mean'], 'MEAN_WEIGHT_KG')))
    if not values: raise Blocked('FROZEN_MEAN_BODY_WEIGHT_KG_REQUIRED')
    if len({value for _, value in values}) != 1: raise Blocked('CONFLICTING_FROZEN_MEAN_BODY_WEIGHTS')
    return values[0]


def _weight_protocol(scenario, root):
    ref = scenario.get('source_protocol', {})
    if not ref.get('path') or not ref.get('sha256'):
        raise Blocked('WEIGHT_DOSE_PROTOCOL_SOURCE_AND_SHA_REQUIRED')
    path = _inside(ref['path'], root)
    if path.stat().st_size > 2*1024**2: raise Blocked('WEIGHT_DOSE_PROTOCOL_SIZE_LIMIT')
    if _sha(path) != ref['sha256']: raise Blocked('WEIGHT_DOSE_PROTOCOL_CHECKSUM_MISMATCH')
    document = json.loads(path.read_text())
    if any(k in document for k in ('observations', 'observed_values', 'ground_truth', 'concentrations')):
        raise Blocked('OUTCOMES_FORBIDDEN_IN_WEIGHT_DOSE_PROTOCOL')
    if document.get('case_id') != scenario.get('case_id') or _name(document.get('compound')) != _name(scenario.get('compound')):
        raise Blocked('WEIGHT_DOSE_PROTOCOL_CASE_OR_COMPOUND_MISMATCH')
    if _sha(path) != ref['sha256']: raise Blocked('WEIGHT_DOSE_PROTOCOL_CHANGED_DURING_READ')
    return document, {'path': str(path.relative_to(root)), 'sha256': ref['sha256']}


_DENSE_REFERENCE_CASES = {
    'furosemide_pmc8416862_trial2_baseline':
        ('f42b77f358034d7c4bcedf643cfdd40e55f1ea8b634aae68c7d263387bed96b6', 720,
         '552c8e2e3ed305e67090b3eb7e37e44ae66b4c2c81315bc0821675521dabcac1'),
    'posaconazole_pmc10986696_fasted_noxafil100mg_be_subset':
        ('24edd777280d91d929f7c836ae1bd8d8ffc1457bce8b0d3380393e1be0676e7e', 8640,
         '4e7a823664fa9e9548dee4370c0fa23aa8cf154ccc02be916dfea2d981be9fcc'),
}


def dense_reference_contract(scenario, *, project_root=ROOT):
    """Register a numerical mesh; it is never a fabricated clinical sample grid."""
    root = Path(project_root).resolve(strict=True)
    case = scenario.get('case_id')
    if case not in _DENSE_REFERENCE_CASES:
        raise Blocked('DENSE_REFERENCE_CASE_NOT_REGISTERED')
    expected_sha, horizon, scenario_sha = _DENSE_REFERENCE_CASES[case]
    if _digest(scenario) != scenario_sha:
        raise Blocked('DENSE_REFERENCE_FROZEN_SCENARIO_DEFINITIONS_DIFFER')
    if scenario.get('source_protocol', {}).get('sha256') != expected_sha:
        raise Blocked('DENSE_REFERENCE_PROTOCOL_SHA_DIFFERS')
    protocol, source = _weight_protocol(scenario, root)
    if protocol != scenario.get('protocol_fields'):
        raise Blocked('DENSE_REFERENCE_FROZEN_PROTOCOL_FIELDS_DIFFER')
    n = scenario.get('normalized_design', {})
    if n.get('sampling', {}).get('times_h') is not None:
        raise Blocked('DENSE_REFERENCE_CANNOT_REPLACE_REPORTED_SAMPLING_TIMES')
    permission = n.get('prediction_contract', protocol.get('prediction_contract', {})).get(
        'source_reference_subject_prediction_allowed')
    if permission is False or not (permission is True or
            scenario.get('execution_policy', {}).get('allow_reference_subject_approximation') is True):
        raise Blocked('DENSE_REFERENCE_APPROXIMATION_PERMISSION_REQUIRED')
    if case.startswith('furosemide_'):
        declared = protocol['protocol']
        if (declared.get('observation_duration_min') != horizon or
                declared.get('sampling_times_min') is not None or
                protocol['prediction_contract'].get('missing_food_or_sampling_may_use_explicit_sensitivity_scenarios') is not True):
            raise Blocked('DENSE_REFERENCE_SOURCE_HORIZON_OR_PERMISSION_DIFFERS')
    else:
        declared = protocol['sampling']
        if (declared.get('reported_sampling_duration_h') * 60 != horizon or
                declared.get('times_h_from_observation_origin') is not None or
                declared.get('missing_nominal_grid_not_fabricated') is not True or
                not any('Dense-Cmax' in s for s in protocol.get('limitations', []))):
            raise Blocked('DENSE_REFERENCE_SOURCE_HORIZON_OR_PERMISSION_DIFFERS')
    copied = deepcopy(scenario)
    copied['normalized_design']['sampling']['times_h'] = [i / 10 for i in range(horizon // 6 + 1)]
    contract = original_contract(copied)
    if contract['administration_times_min'] != [0] or contract['observation_origin_min'] != 0:
        raise Blocked('DENSE_REFERENCE_REQUIRES_EXACT_SINGLE_DOSE_ORIGIN')
    # Integer minute positions avoid binary hour-conversion noise. This is
    # computation-grid registration, not evidence about actual sampling times.
    contract['sampling_times_min'] = list(range(0, horizon + 1, 6))
    contract['absolute_sampling_times_min'] = list(contract['sampling_times_min'])
    contract['sampling_grid_registration'] = {
        'version': 'frozen-horizon-dense-reference-grid-v1', 'source_protocol': source,
        'reported_clinical_sampling_grid_known': False,
        'grid_kind': 'registered_numerical_mesh_not_clinical_sampling',
        'interval_minutes': 6, 'source_reported_horizon_minutes': horizon,
        'numerical_points': len(contract['sampling_times_min']),
        'Cmax_definition': 'dense_reference_individual_maximum; approximation to reported sampled group Cmax',
        'AUC_definition': 'fixed-window integral only; not AUC_last or AUC_infinity',
        'selection_depends_on_clinical_outcomes': False,
        'approximations': ['clinical_sampling_grid_unreported; registered_dense_grid_Cmax_is_reference_only'],
    }
    return contract


def scenario_contract(scenario, *, project_root=ROOT, candidate=None):
    """Wrap the frozen normalizer; expand only documented mean-weight dosing."""
    if candidate is not None and scenario.get('case_id') == 'bupropion_teaima2020_wellbutrin_sr_150mg':
        from .source_bound_dose import normalized_salt_contract
        return normalized_salt_contract(scenario,candidate,project_root=project_root)
    if scenario.get('case_id') in _DENSE_REFERENCE_CASES:
        return dense_reference_contract(scenario, project_root=project_root)
    n = scenario.get('normalized_design', {}) if isinstance(scenario, dict) else {}
    dose = n.get('dose', {}) if isinstance(n, dict) else {}
    if not isinstance(dose, dict) or dose.get('unit') != 'mg/kg':
        try:
            return original_contract(scenario)
        except Blocked as exc:
            if str(exc) == 'SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN':
                raise Blocked(str(exc) + ':source-bound labelled-mass convention, salt/hydration stoichiometry, '
                              'native InputDose species convention and exact conversion factor are not supplied; '
                              'parent molecular weight alone does not establish a conversion') from exc
            raise
    if any(k in scenario or k in n for k in ('observations', 'observed_values', 'ground_truth', 'concentrations')):
        raise Blocked('OUTCOMES_FORBIDDEN_IN_PREDICTION_INPUT')
    if dose.get('conversion_rule') != WEIGHT_RULE:
        raise Blocked('EXPLICIT_REGISTERED_REPRESENTATIVE_WEIGHT_RULE_REQUIRED')
    raw = scenario.get('protocol_fields', {})
    permission = n.get('prediction_contract', raw.get('prediction_contract', {})).get('source_reference_subject_prediction_allowed')
    allowed = permission is True or (permission is not False and
        scenario.get('execution_policy', {}).get('allow_reference_subject_approximation') is True)
    if not allowed: raise Blocked('MEAN_WEIGHT_DOSE_REQUIRES_EXPLICIT_REFERENCE_PERMISSION')
    if dose.get('active_moiety_dose_mg') is not None:
        raise Blocked('COMBINED_WEIGHT_AND_SALT_DOSE_REQUIRES_SEPARATE_BOUND_CONVERSION')
    root = Path(project_root).resolve(strict=True)
    protocol, source_ref = _weight_protocol(scenario, root)
    declared = protocol.get('dose', {})
    for key in ('unit', 'value', 'conversion_rule', 'basis', 'salt'):
        if declared.get(key) != dose.get(key): raise Blocked('FROZEN_WEIGHT_DOSE_DIFFERS_FROM_SOURCE_PROTOCOL:' + key)
    field, weight = _registered_weight(n.get('population'))
    source_field, source_weight = _registered_weight(protocol.get('population'))
    if weight != source_weight or field != source_field: raise Blocked('FROZEN_WEIGHT_DIFFERS_FROM_SOURCE_PROTOCOL')
    per_kg = _positive(dose.get('value'), 'DOSE_MG_PER_KG')
    mass = float(per_kg*weight)
    if not math.isfinite(mass) or mass <= 0: raise Blocked('FINITE_POSITIVE_DERIVED_MASS_REQUIRED')
    normalized = deepcopy(scenario)
    normalized['normalized_design']['dose'].update(value=mass, unit='mg')
    # Preserve explicit schedules; equal mg/kg entries can be normalized only
    # with the very same frozen weight. Other dose schedules remain blocked.
    administrations = normalized['normalized_design'].get('regimen', {}).get('administrations')
    if administrations is not None:
        for row in administrations:
            if row.get('dose') is None: continue
            if row.get('dose_unit', 'mg/kg') != 'mg/kg' or _positive(row['dose'], 'ADMINISTRATION_DOSE_MG_PER_KG') != per_kg:
                raise Blocked('CONFLICTING_WEIGHT_BASED_ADMINISTRATION_DOSES')
            row.update(dose=mass, dose_unit='mg')
    result = original_contract(normalized)
    result['dose_normalization'] = {
        'version': VERSION, 'operation': 'labelled_mg_per_kg_times_frozen_clinical_mean_weight_kg',
        'original_reported_dose': {'value': dose['value'], 'unit': 'mg/kg'},
        'reference_weight': {'value': float(weight), 'unit': 'kg', 'statistic': 'clinical_cohort_mean', 'field': field},
        'registered_dose_mg': mass, 'dimension_check': '(mg/kg) * kg = mg',
        'source_protocol': source_ref, 'original_scenario_sha256': _digest(scenario),
        'original_dose_record_sha256': _digest(dose), 'original_population_record_sha256': _digest(n['population']),
        'original_model_physiology_changed': False, 'individual_doses_available': False,
        'population_simulation_performed': False, 'strict_target_match': False,
        'approximations': ['clinical_mean_weight_defines_a_representative_absolute_dose; not_individual_weight_based_doses',
                           'source_reference_physiology_is_retained; clinical_group_mean_PK_is_not_reconstructed'],
    }
    return result
