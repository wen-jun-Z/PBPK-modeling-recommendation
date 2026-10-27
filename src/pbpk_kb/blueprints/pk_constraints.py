"""Evidence-bound PK checks for complete source models.

No model parameter is changed here. Each endpoint keeps its own definition,
units, context and uncertainty; a diagnostic difference is not a qualification
failure or permission to replace a mechanistic clearance process.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

VERSION = 'source-pk-constraints-v1'
DEFINITIONS = {'vdss': {'vdss_iv', 'vdss_physchem', 'vdss_over_f'},
               'cl': {'cl_systemic_iv', 'cl_over_f'}, 'fu': {'fu_plasma'}}
KINDS = {'measured', 'derived', 'predicted', 'source_assumption'}
DEVELOPMENT_ROLES = {'development', 'calibration'}
UNKNOWN = {None, '', 'unknown', 'not_reported'}


def _number(value, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('PK quantities must be finite numeric values')
    if value < 0 or (positive and value == 0):
        raise ValueError('PK quantities must be positive')
    return float(value)


def normalise_quantity(endpoint, value, unit):
    """Preserve absolute/per-kg distinction; never invent a subject's weight."""
    value = _number(value, positive=endpoint != 'fu')
    unit = str(unit).strip().lower().replace(' ', '')
    conversions = {
        'vdss': {'l': (1., 'L'), 'ml': (.001, 'L'),
                 'l/kg': (1., 'L/kg'), 'ml/kg': (.001, 'L/kg')},
        'cl': {'ml/min': (1., 'mL/min'), 'l/min': (1000., 'mL/min'),
               'l/h': (1000./60, 'mL/min'), 'ml/h': (1./60, 'mL/min'),
               'ml/min/kg': (1., 'mL/min/kg'), 'l/min/kg': (1000., 'mL/min/kg'),
               'l/h/kg': (1000./60, 'mL/min/kg'), 'ml/h/kg': (1./60, 'mL/min/kg')},
        'fu': {'1': (1., '1'), 'fraction': (1., '1'), '%': (.01, '1'),
               'percent': (.01, '1')},
    }
    try:
        factor, canonical = conversions[endpoint][unit]
    except KeyError as error:
        raise ValueError(f'Unsupported {endpoint} unit: {unit}') from error
    value *= factor
    if not math.isfinite(value) or (endpoint == 'fu' and not 0 <= value <= 1):
        raise ValueError('Invalid normalised PK value')
    return value, canonical


def _source(reference, root):
    if not isinstance(reference, dict) or not all(reference.get(k) for k in ('path', 'sha256', 'locator')):
        raise ValueError('Every PK record requires project evidence, SHA256 and locator')
    path = (root / reference['path']).resolve(strict=True)
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError('PK evidence must remain inside the project')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference['sha256']:
        raise ValueError('PK evidence checksum mismatch')


def _uncertainty(record):
    uncertainty = deepcopy(record.get('uncertainty'))
    if not uncertainty:
        return None
    if not isinstance(uncertainty, dict) or not uncertainty.get('kind'):
        raise ValueError('Uncertainty needs an explicit interpretation')
    for key in ('lower', 'upper', 'sd'):
        if uncertainty.get(key) is not None:
            uncertainty[key] = normalise_quantity(record['endpoint'], uncertainty[key], record['unit'])[0]
    lower, upper = uncertainty.get('lower'), uncertainty.get('upper')
    if (lower is None) != (upper is None):
        raise ValueError('An uncertainty interval needs both limits')
    if lower is not None and not lower <= record['normalised_value'] <= upper:
        raise ValueError('Uncertainty limits must bracket the estimate')
    return uncertainty


def _context_reasons(record, target):
    source, endpoint = record['context'], record['endpoint']
    reasons = []
    for field in ('species', 'population', 'matrix'):
        value = source.get(field)
        if value in UNKNOWN or target.get(field) in UNKNOWN:
            reasons.append(field + '_not_verified')
        elif value != target[field]:
            reasons.append(field + '_mismatch')
    if source.get('species') != 'human':
        reasons.append('not_verified_human_evidence')
    if source.get('matrix') != 'plasma':
        reasons.append('not_plasma_evidence')
    if endpoint != 'fu':
        if source.get('route') != 'iv' or source.get('dosing') != 'single':
            reasons.append('not_single_iv_reference')
        if target.get('route') != 'iv' or target.get('dosing') != 'single':
            reasons.append('target_not_single_iv')
        if record['source_kind'] in {'measured', 'derived'}:
            for field in ('dose_mg', 'infusion_minutes'):
                if source.get(field) is None or target.get(field) is None:
                    reasons.append(field + '_not_verified')
                elif not math.isclose(_number(source[field], positive=False),
                                      _number(target[field], positive=False), rel_tol=1e-6, abs_tol=1e-9):
                    reasons.append(field + '_mismatch')
    for field in ('sex', 'age_years', 'weight_kg', 'food_state'):
        if source.get(field) not in UNKNOWN:
            if target.get(field) in UNKNOWN:
                reasons.append(field + '_not_matched')
            elif source[field] != target[field]:
                reasons.append(field + '_mismatch')
    return list(dict.fromkeys(reasons))


def prepare_pk_constraints(records, target_context, project_root, *, forbidden_study_ids=()):
    """Select matched development evidence without reading held-out values.

    Clinical mismatches and source assumptions remain diagnostic references.
    Final-evaluation/selection evidence is excluded before numeric decoding.
    Predictions of unknown applicability/uncertainty are never fit targets.
    """
    root = Path(project_root).resolve(strict=True)
    if not isinstance(target_context, dict) or not target_context.get('compound_id'):
        raise ValueError('Explicit target drug and clinical context are required')
    if not isinstance(records, list) or len(records) > 1000:
        raise ValueError('Provide at most 1000 explicit PK records')
    if (not isinstance(forbidden_study_ids, (list, tuple, set))
            or any(not isinstance(s, str) or not s for s in forbidden_study_ids)):
        raise ValueError('Excluded evaluation/selection study identifiers must be a list of strings')
    rows, identifiers = [], set()
    for record in records:
        identity = record.get('record_id')
        if not isinstance(identity, str) or not identity or identity in identifiers:
            raise ValueError('PK record identifiers must be nonempty and unique')
        identifiers.add(identity)
        if record.get('study_role') not in DEVELOPMENT_ROLES or record.get('study_id') in forbidden_study_ids:
            rows.append({'record_id': identity, 'status': 'excluded', 'reasons': ['evaluation_or_selection_evidence']})
            continue
        if record.get('compound_id') != target_context['compound_id']:
            rows.append({'record_id': identity, 'status': 'excluded', 'reasons': ['different_compound']})
            continue
        item = deepcopy(record)
        endpoint = item.get('endpoint')
        if endpoint not in DEFINITIONS or item.get('definition') not in DEFINITIONS[endpoint]:
            raise ValueError('PK endpoint and definition must be explicit and compatible')
        if item.get('source_kind') not in KINDS or not isinstance(item.get('context'), dict):
            raise ValueError('PK source kind and clinical context are required')
        _source(item.get('source'), root)
        item['normalised_value'], item['normalised_unit'] = normalise_quantity(endpoint, item['value'], item['unit'])
        item['normalised_uncertainty'] = _uncertainty(item)
        reasons = _context_reasons(item, target_context)
        if item['definition'] in {'vdss_over_f', 'cl_over_f', 'vdss_physchem'}:
            reasons.append('definition_not_observed_systemic_pk')
        if item['source_kind'] == 'source_assumption':
            reasons.append('source_assumption_not_measurement')
        if item['source_kind'] == 'derived' and not item.get('derivation'):
            reasons.append('derivation_not_documented')
        if item['source_kind'] == 'predicted':
            if item.get('preprocessing_compatibility') not in (None, 'original', 'official') or item.get('exploratory') is True:
                reasons.append('predictor_compatibility_variant_not_validated')
            if item.get('applicability') != 'in_domain':
                reasons.append('predictor_applicability_not_confirmed')
            if item.get('training_overlap') is not False:
                reasons.append('predictor_training_overlap_or_unknown')
            u = item.get('normalised_uncertainty') or {}
            if u.get('kind') not in {'prediction_interval', 'empirical_error_range'} or u.get('lower') is None:
                reasons.append('predictive_uncertainty_not_quantified')
        item['reasons'] = reasons
        item['status'] = 'diagnostic_only' if reasons else 'constraint_candidate'
        rows.append(item)
    # Prefer matched measurements; do not average different populations/studies
    # or silently discard contradictory references at the preferred level.
    for endpoint in DEFINITIONS:
        eligible = [r for r in rows if r.get('endpoint') == endpoint and r['status'] == 'constraint_candidate']
        if any(r['source_kind'] in {'measured', 'derived'} for r in eligible):
            for row in eligible:
                if row['source_kind'] == 'predicted':
                    row['status'] = 'diagnostic_only'
                    row['reasons'].append('matched_measurement_available')
    return {'version': VERSION, 'target_context': deepcopy(target_context), 'records': rows,
            'constraint_count': sum(r['status'] == 'constraint_candidate' for r in rows),
            'parameters_modified': False, 'independent_validation': False,
            'unknown_measurement_weights_not_imputed': True}


def compare_pk_constraints(profile, prepared):
    """Report every compatible endpoint separately, without a combined score."""
    comparisons = []
    target = prepared['target_context']
    model_context = profile.get('context', {})
    if model_context.get('compound_id') not in (None, target['compound_id']):
        raise ValueError('Model and evidence refer to different compounds')
    weight = model_context.get('weight_kg')
    for record in prepared['records']:
        if record['status'] == 'excluded':
            continue
        metrics = [m for m in profile.get('metrics', []) if m.get('endpoint') == record['endpoint']]
        if not metrics:
            comparisons.append({'record_id': record['record_id'], 'status': 'model_metric_missing',
                                'endpoint': record['endpoint'], 'eligible_for_constraint': False})
        for metric in metrics:
            row = {'record_id': record['record_id'], 'endpoint': record['endpoint'],
                   'evidence_kind': record['source_kind'], 'evidence_status': record['status'],
                   'evidence_definition': record['definition'], 'model_definition': metric.get('definition'),
                   'eligible_for_constraint': False, 'status': 'unavailable', 'log_ratio': None, 'fold_error': None}
            if metric.get('value') is None:
                row['reason'] = metric.get('reason', 'model_metric_unavailable')
                comparisons.append(row)
                continue
            if metric.get('definition') != record['definition']:
                row['status'], row['reason'] = 'definition_mismatch', 'PK_definitions_are_not_interchangeable'
                comparisons.append(row)
                continue
            try:
                value, unit = normalise_quantity(record['endpoint'], metric['value'], metric['unit'])
                desired = record['normalised_unit']
                conversion = 'none'
                if unit != desired:
                    model_weight = _number(weight)
                    if unit == desired + '/kg':
                        value *= model_weight
                    elif desired == unit + '/kg':
                        value /= model_weight
                    else:
                        raise ValueError('Incompatible model and evidence units')
                    conversion = 'model_own_weight_only'
                evidence = record['normalised_value']
                if value <= 0 or evidence <= 0:
                    raise ValueError('Log-ratio requires positive values; no epsilon substitution')
                ratio = math.log(value) - math.log(evidence)
                fold = math.exp(abs(ratio))
                if not math.isfinite(fold):
                    raise ValueError('PK fold error overflow')
            except ValueError as error:
                row['reason'] = str(error)
                comparisons.append(row)
                continue
            model_context_reasons = []
            fields = ['species', 'matrix', 'population']
            fields += [field for field in ('sex', 'age_years', 'weight_kg', 'food_state')
                       if target.get(field) not in UNKNOWN]
            if record['endpoint'] != 'fu':
                fields += ['route', 'dosing', 'dose_mg', 'infusion_minutes']
            for field in fields:
                expected, actual = target.get(field), model_context.get(field)
                if expected in UNKNOWN or actual in UNKNOWN:
                    model_context_reasons.append(field + '_not_verified_in_model')
                elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
                    if not math.isclose(float(expected), _number(actual, positive=False), rel_tol=1e-6, abs_tol=1e-9):
                        model_context_reasons.append(field + '_model_mismatch')
                elif expected != actual:
                    model_context_reasons.append(field + '_model_mismatch')
            eligible = (record['status'] == 'constraint_candidate' and metric.get('eligible_for_constraint') is True
                        and not model_context_reasons)
            row.update(status='compared', model_value=value, evidence_value=evidence,
                       unit=desired, log_ratio=ratio, fold_error=fold,
                       eligible_for_constraint=eligible, model_unit_conversion=conversion,
                       evidence_reasons=record['reasons'], model_reason=metric.get('reason'),
                       model_context_reasons=model_context_reasons)
            uncertainty = record.get('normalised_uncertainty')
            row['uncertainty'] = uncertainty
            if uncertainty and uncertainty.get('lower') is not None:
                lower, upper = uncertainty['lower'], uncertainty['upper']
                row['reported_interval_relation'] = 'below' if value < lower else 'above' if value > upper else 'within'
            else:
                row['reported_interval_relation'] = 'not_available'
            # No hard success threshold, no sum over different PK endpoints.
            row['scientific_pass_inferred'] = False
            comparisons.append(row)
    return {'version': VERSION, 'target_context': deepcopy(target), 'comparisons': comparisons,
            'constraint_comparison_count': sum(r['eligible_for_constraint'] for r in comparisons),
            'parameters_modified': False, 'composite_score': None, 'scientific_qualified': False,
            'next_action': 'Review comparable distribution/binding/clearance discrepancies before choosing mechanism edits.'}


def check_source_pk(*, snapshot, execution_manifest, evidence, target_context,
                    output_dir, project_root, predictor_input=None, target_output=None):
    """Execute the evidence/predictor/native-profile workflow without new ODEs."""
    from .source_pk_profile import extract_source_pk_profile
    from .source_workflow import inside, save, sha256
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError('PK check outputs must use a new directory')
    compound = snapshot['Simulations'][0]['Compounds'][0]['Name']
    if compound != target_context.get('compound_name'):
        raise ValueError('Target compound name must match the complete source simulation')
    target_output = target_output or f'Organism|PeripheralVenousBlood|{compound}|Plasma (Peripheral Venous Blood)'
    if target_output not in snapshot['Simulations'][0].get('OutputSelections', []):
        raise ValueError('Requested PK output must already exist in the full source model')
    records = deepcopy(evidence['records'] if isinstance(evidence, dict) else evidence)
    # Validate measured records before launching any helper process.
    forbidden = []
    for key in ('evaluation_study_ids', 'selection_study_ids'):
        group = target_context.get(key, [])
        if not isinstance(group, list) or any(not isinstance(s, str) or not s for s in group):
            raise ValueError(key + ' must be an explicit list of study identifiers')
        forbidden.extend(group)
    prepare_pk_constraints(records, target_context, root, forbidden_study_ids=forbidden)
    output.mkdir(parents=True)
    if predictor_input is not None:
        from .pk_predictor import predict_pk
        if predictor_input.get('compound_id') != target_context['compound_id']:
            raise ValueError('Predictor structure must refer to the target compound')
        if not predictor_input.get('input_reference'):
            raise ValueError('Predictor structure requires a checksummed identity source')
        predicted = predict_pk(predictor_input['smiles'], predictor_input['compound_id'], root,
                               input_reference=predictor_input['input_reference'],
                               **({'preprocessing_compatibility': predictor_input['preprocessing_compatibility']}
                                  if predictor_input.get('preprocessing_compatibility') else {}))
        save(output / 'prediction.json', predicted)
        for entry in predicted['records']:
            row = deepcopy(entry)
            row.update(record_id='pksmart_' + target_context['compound_id'] + '_' + row['endpoint'],
                       study_id='PKSmart_published_training_dataset', study_role='development',
                       predictor_provenance=row['source'], source={
                           'path': str((output / 'prediction.json').relative_to(root)),
                           'sha256': sha256(output / 'prediction.json'),
                           'locator': 'records[endpoint=' + row['endpoint'] + ']',
                           'citation': 'https://doi.org/10.1186/s13321-025-01066-5'})
            records.append(row)
    prepared = prepare_pk_constraints(records, target_context, root, forbidden_study_ids=forbidden)
    save(output / 'constraints.json', prepared)
    profile = extract_source_pk_profile(snapshot, execution_manifest, target_output, output / 'native_profile', root)
    comparison = compare_pk_constraints(profile, prepared)
    comparison.update(status='completed' if profile.get('status') == 'completed' else 'native_profile_failed',
                      predictor_used=predictor_input is not None, new_ode_executions=0,
                      implementation_sha256=sha256(Path(__file__)),
                      native_profile_directory=str((output / 'native_profile').relative_to(root)),
                      constraints_path=str((output / 'constraints.json').relative_to(root)),
                      report_path=str((output / 'pk_comparison.json').relative_to(root)))
    save(output / 'pk_comparison.json', comparison)
    return comparison


def main(argv=None):
    import argparse
    from .source_workflow import inside
    parser = argparse.ArgumentParser(description='Check source PBPK model against PK evidence and optional local predictions; no new ODE.')
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--execution-manifest', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--predictor-input', type=Path)
    parser.add_argument('--target-output')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    def read(path):
        return json.loads(inside(path, root, exists=True).read_text())

    result = check_source_pk(snapshot=read(args.snapshot),
                            execution_manifest=inside(args.execution_manifest, root, exists=True),
                            evidence=read(args.evidence), target_context=read(args.context),
                            predictor_input=read(args.predictor_input) if args.predictor_input else None,
                            target_output=args.target_output, output_dir=args.output, project_root=root)
    print(json.dumps({k: result[k] for k in ('status', 'constraint_comparison_count', 'predictor_used',
                                          'new_ode_executions', 'parameters_modified', 'report_path')}, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
