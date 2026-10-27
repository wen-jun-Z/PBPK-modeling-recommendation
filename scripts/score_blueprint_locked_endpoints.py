#!/usr/bin/env python3
"""Score sealed clinical endpoints only after immutable prediction selection.

Reference-subject and dense-grid predictions remain explicitly approximate.
No fitting, model selection, cohort selection or curve editing occurs here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'pbpk-external-predictions-lock-v1'


def inside(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Path outside project')
    return resolved


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified(reference):
    path = inside(ROOT / reference['path'])
    content = path.read_bytes()
    if sha(content) != reference['sha256']:
        raise ValueError('Locked file checksum mismatch: ' + str(path.relative_to(ROOT)))
    return content


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value')
    return float(value)


def curve_endpoints(model, mw, tail_policy):
    if model['unit'] not in {'µmol/l', 'μmol/l', 'umol/l', 'µmol/L', 'μmol/L', 'umol/L'}:
        raise ValueError('Only explicitly labelled micromolar concentration supported')
    time = [finite(v) for v in model['time_min']]
    concentration = [finite(v) for v in model['concentration']]
    if len(time) != len(concentration) or len(time) < 4 or time[0] != 0:
        raise ValueError('Full time-zero curve with at least four samples required')
    if any(b <= a for a, b in zip(time, time[1:])) or any(c < 0 for c in concentration):
        raise ValueError('Invalid time sequence or negative concentration')
    # umol/L * g/mol = ug/L = ng/mL; integrate hours for ng*h/mL.
    ng_ml = [c * mw for c in concentration]
    if any(not math.isfinite(c) for c in ng_ml):
        raise ValueError('Concentration unit conversion overflow')
    auc = sum((time[i] - time[i-1]) / 60 * (ng_ml[i] + ng_ml[i-1]) / 2 for i in range(1, len(time)))
    if not math.isfinite(auc) or auc <= 0:
        raise ValueError('A completely empty concentration curve is not a valid prediction')
    start_h = finite(tail_policy['fit_window_start_h'])
    end_h = finite(tail_policy['fit_window_end_h'])
    minimum_points = tail_policy['minimum_points']
    minimum_r_squared = finite(tail_policy['minimum_r_squared_for_primary'])
    maximum_tail_fraction = finite(tail_policy['maximum_tail_fraction_for_primary'])
    if (not isinstance(minimum_points, int) or isinstance(minimum_points, bool) or minimum_points < 3
        or not 0 <= start_h < end_h or abs(time[-1] / 60 - end_h) > 1e-9
        or not 0 < minimum_r_squared <= 1 or not 0 < maximum_tail_fraction < 1
        or tail_policy.get('require_negative_log_slope') is not True
        or tail_policy.get('do_not_clip_or_replace_failed_tail') is not True):
        raise ValueError('Invalid or inconsistent locked tail policy')
    tail_points = [(t / 60, math.log(c)) for t, c in zip(time, ng_ml) if start_h - 1e-9 <= t / 60 <= end_h + 1e-9 and c > 0]
    tail = None
    reason = None
    slope = r_squared = None
    if ng_ml[-1] == 0:
        # Do not claim an exactly zero tail merely from numerical underflow.
        reason = 'terminal concentration zero; infinity tail cannot be estimated'
    elif len(tail_points) < minimum_points:
        reason = 'fewer positive terminal points than required by locked policy'
    else:
        xs, ys = zip(*tail_points)
        xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
        denominator = sum((x - xmean) ** 2 for x in xs)
        slope = sum((x - xmean) * (y - ymean) for x, y in tail_points) / denominator
        residual = sum((y - (ymean + slope * (x - xmean))) ** 2 for x, y in tail_points)
        variance = sum((y - ymean) ** 2 for y in ys)
        r_squared = 1 - residual / variance if variance > 0 else 0.0
        if slope >= 0 or r_squared < minimum_r_squared:
            reason = 'terminal window is not a descending log-linear phase at the preregistered R2 threshold'
        else:
            tail = ng_ml[-1] / (-slope)
            if tail / (auc + tail) > maximum_tail_fraction:
                reason = 'terminal tail fraction exceeds locked primary endpoint threshold'
    return {'Cmax_dense': max(ng_ml), 'Cmax_unit': 'ng/mL',
        'AUC_observed_simulation_window': auc, 'AUC_unit': 'ng*h/mL',
        'AUC_infinity': None if tail is None or reason else auc + tail,
        'terminal_tail': {'method': 'OLS log concentration in the locked fixed terminal window',
            'locked_policy': tail_policy, 'positive_points': len(tail_points), 'slope_per_h': slope,
            'r_squared': r_squared, 'tail_auc': tail,
            'tail_fraction': None if tail is None else tail / (auc + tail),
            'valid': tail is not None and reason is None, 'failure': reason},
        'time_start_min': time[0], 'time_end_min': time[-1], 'grid_points': len(time)}


def validate_locked_inputs(lock_path, design_path):
    lock_path, design_path = inside(lock_path), inside(design_path)
    lock_bytes, design_bytes = lock_path.read_bytes(), design_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get('schema_version') != SCHEMA or lock.get('model_selection_frozen') is not True:
        raise ValueError('A completed model-selection freeze is required before opening outcomes')
    if lock.get('design_sha256') != sha(design_bytes):
        raise ValueError('Design checksum does not match the locked design')
    design = json.loads(design_bytes)
    protocol = json.loads(verified(lock['protocol']))
    if protocol.get('case_id') != design['case_id'] or protocol.get('design_sha256') != lock['design_sha256']:
        raise ValueError('Locked protocol case or design identity mismatch')
    weights = {sex.lower(): finite(value) for sex, value in protocol['sex_weights'].items()}
    if len(weights) != len(protocol['sex_weights']):
        raise ValueError('Duplicate sex labels after normalization')
    if not weights or any(weight <= 0 for weight in weights.values()) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError('Sex weights must be positive and sum to one')
    mw = finite(protocol.get('molecular_weight_g_mol', protocol.get('MW')))
    if mw <= 0 or not protocol.get('approximation'):
        raise ValueError('Molecular weight and explicit approximation contract required')
    entries = lock['predictions']
    if len(entries) != 1 or entries[0]['case_id'] != design['case_id']:
        raise ValueError('This scorer requires exactly one locked clinical case')
    predictions = json.loads(verified(entries[0]))
    if predictions.get('case_id') != design['case_id']:
        raise ValueError('Prediction case identity does not match design')
    if predictions.get('protocol_sha256') != lock['protocol']['sha256']:
        raise ValueError('Predictions do not bind the locked protocol')
    models = {}
    for model in predictions['models']:
        key, sex = model['model_key'], model['sex'].lower()
        if sex not in weights or sex in models.setdefault(key, {}):
            raise ValueError('Unexpected or duplicated model-sex curve')
        models[key][sex] = curve_endpoints(model, mw, protocol['tail_policy'])
    selections = predictions['selections']
    if selections != protocol['selections']:
        raise ValueError('Prediction selections differ from the prespecified protocol')
    for selection in selections:
        fallback = (str(selection.get('block_id')) + '__' if selection.get('block_id') else '') + str(selection['method']) + '_seed_' + str(selection['seed'])
        selection['selection_id'] = selection.get('selection_id', fallback)
    if not selections or len({s['selection_id'] for s in selections}) != len(selections):
        raise ValueError('Missing or duplicate frozen selections')
    selected_keys = {s['model_key'] for s in selections if s['model_key'] is not None}
    if set(models) != selected_keys:
        raise ValueError('Prediction models must exactly match nonfailed frozen model selections')
    for selected in selections:
        if selected['model_key'] is None:
            continue
        if selected['model_key'] not in models or set(models[selected['model_key']]) != set(weights):
            raise ValueError('A selected model lacks exactly the registered sex curves')
    # All designer-visible objects are authenticated and predictions validated
    # before the caller is allowed to decode the sealed endpoint reference.
    return {'lock': lock, 'lock_sha256': sha(lock_bytes), 'design': design,
        'protocol': protocol, 'weights': weights, 'models': models, 'selections': selections}


def score_locked(lock_path, design_path):
    checked = validate_locked_inputs(lock_path, design_path)
    design = checked['design']
    sealed = json.loads(verified(design['sealed_observation_reference']))
    if sealed['case_id'] != design['case_id'] or sealed['study_id'] != design['study_id']:
        raise ValueError('Sealed source identity does not match the design')
    truth = {}
    for endpoint in sealed['observed_endpoints']:
        name = endpoint['endpoint']
        expected_unit = {'Cmax': 'ng/mL', 'AUC_infinity': 'ng*h/mL'}.get(name)
        if expected_unit is None:
            continue
        if endpoint['unit'] != expected_unit or endpoint['aggregation'] != 'arithmetic_mean':
            raise ValueError('Unsupported endpoint unit or aggregation')
        value = finite(endpoint['value'])
        if value <= 0 or name in truth:
            raise ValueError('Invalid or duplicated sealed endpoint')
        truth[name] = {**endpoint, 'value': value}
    if set(truth) != {'Cmax', 'AUC_infinity'}:
        raise ValueError('Required Furosemide endpoint contract absent')
    results = []
    for selected in checked['selections']:
        selection_failed = selected['model_key'] is None
        model = {} if selection_failed else checked['models'][selected['model_key']]
        row = {**selected, 'endpoints': [], 'population_approximation': checked['protocol']['approximation'],
            'per_sex_prediction_diagnostics': model, 'strict_population_validation': False,
            'prediction_available': not selection_failed, 'retained_in_method_job_denominator': True}
        for name, observed in truth.items():
            metric = 'Cmax_dense' if name == 'Cmax' else 'AUC_infinity'
            values = {sex: curve[metric] for sex, curve in model.items()}
            predicted = None if selection_failed or any(v is None for v in values.values()) else sum(checked['weights'][sex] * value for sex, value in values.items())
            result = {'endpoint': name, 'predicted': predicted, 'observed': observed['value'], 'unit': observed['unit'],
                'prediction_aggregation': 'weighted mean of separately computed sex-reference-subject endpoints',
                'source_aggregation': 'arithmetic mean of individual observed NCA endpoints',
                'sampled_vs_dense_maximum_approximation': name == 'Cmax',
                'reference_subject_population_approximation': True,
                'source_table_id': observed['source_table_id'], 'source_row': observed['source_row'],
                'source_column': observed['source_column'], 'evaluable': predicted is not None and predicted > 0}
            if result['evaluable']:
                ratio = predicted / observed['value']
                result.update(fold_error=max(ratio, 1 / ratio), predicted_over_observed=ratio,
                    absolute_log_error=abs(math.log(ratio)), absolute_relative_error=abs(ratio - 1))
            else:
                result['failure'] = ('No frozen selected model; upstream failure is retained in the denominator'
                    if selection_failed else 'No valid positive infinity prediction; observed endpoint was not replaced by finite-window AUC')
            row['endpoints'].append(result)
        results.append(row)
    return {'schema_version': 'pbpk-external-endpoint-score-v1', 'case_id': design['case_id'],
        'prediction_lock_sha256': checked['lock_sha256'], 'model_selection_frozen': True,
        'opened_outcomes_only_after_lock_validation': True, 'cohort_protocol_approximation': True,
        'strict_main_promotion': False, 'no_post_outcome_model_selection': True, 'results': results,
        'method_job_denominator': len(results),
        'jobs_with_frozen_model_prediction': sum(r['prediction_available'] for r in results),
        'jobs_without_frozen_model_prediction': sum(not r['prediction_available'] for r in results),
        'scorer_sha256': sha(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', required=True)
    parser.add_argument('--design', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    result = score_locked(inside(ROOT / args.lock), inside(ROOT / args.design))
    output = inside(ROOT / args.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    if len(payload) > 8 * 1024 ** 2:
        raise ValueError('Score output limit exceeded')
    target = output / 'endpoint_scores.json'
    with target.open('xb') as handle:
        handle.write(payload)
    print(json.dumps({'case_id': result['case_id'], 'scored_frozen_selections': len(result['results']),
        'output': str(target.relative_to(ROOT)), 'strict_main_promotion': False}))


if __name__ == '__main__':
    main()
