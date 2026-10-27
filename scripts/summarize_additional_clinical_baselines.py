#!/usr/bin/env python3
"""Score already-locked additional predictions with the unchanged clinical NCA.

Write one reusable JSON for Figure 6 and the manuscript. Original Ours/RF
CSV and all clinical references remain unchanged. No model fitting occurs.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from run_complete_reference_pilot import ROOT, read, verify, ref
from run_clinical_method_comparison import predicted_nca, clinical_unit_factor
from render_pbpk_method_comparison import checked_curve, curve_metrics, Ineligible
from pbpk_kb.blueprints.source_workflow import save, sha256

BASE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug'
ADD = BASE / 'additional_baselines'
CSV = ROOT / 'results/tables/pbpk_ours_vs_rf/comparison.csv'
OUT = CSV.with_name('additional_comparison.json')
METHODS = [
    {'id': 'ours', 'label': 'Ours', 'information': 'RF molecular priors and a separate same-drug development clinical curve; two PBPK partition candidates'},
    {'id': 'rf', 'label': 'RF + PBPK', 'information': 'Molecular features and fixed PK-Sim direct PBPK construction; no development clinical curve'},
    {'id': 'svr', 'label': 'SVR + PBPK', 'information': 'Molecular features and the same fixed PBPK construction; training-only scaffold CV; no development clinical curve'},
    {'id': 'xgboost', 'label': 'XGBoost + PBPK', 'information': 'Molecular features and the same fixed PBPK construction; training-only scaffold CV; no development clinical curve'},
    {'id': 'two_compartment', 'label': 'Two-compartment PK', 'information': 'Same development clinical curve as Ours; four PK parameters with fixed weight scaling; no PBPK physiology'},
]


def weak_endpoints(diagnostics):
    keys = ('auc_extrapolated_fraction', 'aumc_extrapolated_fraction', 'terminal_span_in_half_lives')
    if not all(isinstance(diagnostics.get(k), (int, float)) and math.isfinite(diagnostics[k]) for k in keys):
        return []
    weak_slope = diagnostics['terminal_span_in_half_lives'] < 1
    weak_auc = weak_slope or diagnostics['auc_extrapolated_fraction'] > .2
    weak_moment = weak_auc or diagnostics['aumc_extrapolated_fraction'] > .2
    return ((['terminal_half_life'] if weak_slope else []) +
            (['auc_inf', 'clearance', 'vz'] if weak_auc else []) +
            (['mrt', 'vss'] if weak_moment else []))


def prediction_in_hours(native, molecular_weight):
    """Coalesce only exactly equal float times caused by min-to-hour conversion.

    A union of a minute grid and protocol times can differ by one minute ULP
    but become identical after division by 60. Concentrations must agree to
    numerical precision; no observations, smoothing or fitted values are used.
    """
    times, values, merged = [], [], []
    for index, (minute, concentration) in enumerate(zip(native['time_min'], native['concentration'])):
        t, c = minute/60, concentration*molecular_weight
        if times and t == times[-1]:
            if not math.isclose(c, values[-1], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Conflicting predictions at a duplicate converted time')
            merged.append({'native_index': index, 'time_h': t})
            continue
        if times and t < times[-1]: raise ValueError('Native prediction times are unsorted')
        times.append(t); values.append(c)
    return times, values, merged


def main():
    old = read(BASE / 'optimized_run/summary.json')
    verify(old['protocol']['pins']); verify(old['locks']['inputs'])
    ml = read(ADD / 'ml/summary.json')
    if ml['status'] != 'predictions_locked':
        raise ValueError('All molecular-baseline attempts must be locked before scoring')
    verify(ml['protocol']['pins']); verify(ml['locks']['inputs'])
    classical = read(ADD / 'two_compartment/summary.json')
    if classical.get('status') not in {'completed', 'predictions_locked'}:
        raise ValueError('All two-compartment predictions must be locked before scoring')
    verify(classical['protocol']['source_pins'])
    for entry in classical['results'].values():
        verify({entry[key]['path']: entry[key]['sha256'] for key in ('fit', 'prediction')})
        verify(read(ROOT / entry['fit']['path'])['source_pins'])
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in [CSV, BASE / 'optimized_run/comparison_input.json',
        ADD / 'ml/summary.json', ADD / 'two_compartment/summary.json', Path(__file__)]}
    # Lock verification precedes any decoding of target clinical references.
    for cid, methods in old['locks']['predictions'].items():
        verify({r['path']: r['sha256'] for r in methods.values()})
    for cid, methods in ml['results'].items():
        for result in methods.values():
            if result['status'] == 'predicted':
                verify({result['prediction']['path']: result['prediction']['sha256']})
    for case in old['protocol']['cases']:
        path = ADD / 'two_compartment' / case['case_id'] / 'prediction_locked.json'
        fit = read(path.with_name('fit.json'))
        digest = sha256(path)
        # Independently verify against the per-case fit's prediction lock.
        expected = fit.get('prediction_sha256') or (fit.get('prediction') or {}).get('sha256')
        if expected is None:
            entry = classical.get('results', {}).get(case['case_id'], {})
            expected = entry.get('prediction_sha256') or (entry.get('prediction') or {}).get('sha256')
        if expected is None or digest != expected:
            raise ValueError('Two-compartment prediction requires an intact recorded lock: '+case['case_id'])
        pins[str(path.relative_to(ROOT))] = digest
    raw_cases = {c['case_id']: c for c in read(BASE / 'optimized_run/comparison_input.json')['cases']}
    rows = list(csv.DictReader(CSV.open()))
    by_case = {}
    for row in rows: by_case.setdefault(row['case_id'], {})[row['endpoint']] = row
    records = []
    for config in old['protocol']['cases']:
        cid = config['case_id']; case = raw_cases[cid]; csv_rows = by_case[cid]
        policy = config['nca_policy']; weight = policy['weight_kg']
        original_nca = read(BASE / 'optimized_run/predictions' / cid / 'ours_reported.json')['nca_diagnostics']
        mw = read(ROOT / config['physchem'])['identity']['molecular_weight_g_mol']
        reference = checked_curve(case['clinical_reference']['curve'], 'clinical', clinical=True)
        record = {'case_id': cid, 'drug': config['drug'], 'study_id': config['study_id'],
                  'observed_points': int(csv_rows['auc']['curve_scored_points']),
                  'conditions': case['conditions'], 'approximations': case.get('approximations', []),
                  'clinical_source': case['clinical_reference']['primary_source'],
                  'gmfe': {}, 'curve_metrics': {}, 'endpoints': {}, 'prediction_sources': {}, 'nca_quality': {}}
        scored = {}
        for method in [x['id'] for x in METHODS]:
            if method in ('ours', 'rf'):
                lock = old['locks']['predictions'][cid][method]
            elif method in ('svr', 'xgboost'):
                result = ml['results'][cid][method]
                if result['status'] != 'predicted':
                    scored[method] = {'failure': result.get('reason', result['status'])}
                    record['gmfe'][method] = None
                    record['curve_metrics'][method] = {'reason': scored[method]['failure']}
                    continue
                lock = result['prediction']
            else:
                lock = ref(ADD / 'two_compartment' / cid / 'prediction_locked.json', 'Classical PK prediction locked before target scoring')
            verify({lock['path']: lock['sha256']}); pins[lock['path']] = lock['sha256']
            native = read(ROOT / lock['path'])
            assert native['unit'] == 'umol/l' and native['time_unit'] == 'min'
            times, values, merged = prediction_in_hours(native, mw)
            predicted = predicted_nca(policy=policy, time_h=times, concentration_ng_ml=values,
                dose_ng=policy['dose_ng'], infusion_h=policy['infusion_h'],
                nominal_times_h=policy['sampling_time_h'], lloq_ng_ml=original_nca['lloq_ng_ml'], weight_kg=weight)
            factor = clinical_unit_factor('curve', reference['unit'], weight)
            curve = {'time_h': times, 'concentration': [v*factor for v in values], 'unit': reference['unit'],
                     'matrix': 'plasma', 'statistic': 'representative_adult'}
            try:
                metrics = curve_metrics(curve, reference, set(case.get('approximations', [])))
            except (Ineligible, OverflowError) as exc:
                metrics = {'gmfe': None, 'reason': str(exc)}
            if method in ('ours', 'rf'):
                assert math.isclose(metrics['gmfe'], float(csv_rows['auc'][method+'_curve_gmfe']), rel_tol=1e-10)
            record['gmfe'][method] = metrics['gmfe']; record['curve_metrics'][method] = metrics
            record['prediction_sources'][method] = lock
            weak = weak_endpoints(predicted['diagnostics'])
            record['nca_quality'][method] = {k: predicted['diagnostics'].get(k) for k in
                ['auc_extrapolated_fraction', 'aumc_extrapolated_fraction', 'terminal_span_in_half_lives', 'failures']}
            record['nca_quality'][method]['weak_endpoints'] = weak
            record['nca_quality'][method]['coalesced_float_time_duplicates'] = merged
            scored[method] = {'endpoints': predicted['endpoints'], 'weak': weak}
        for ep, row in csv_rows.items():
            factor = float(row['display_value_factor'])
            observed = float(row['clinical_reference'])*factor if row['clinical_reference'] else None
            cell = {'observed': observed, 'unit': row['display_unit'], 'source_unit': row['unit'], 'display_factor': factor,
                    'reference_location': row['clinical_location'], 'predictions': {}}
            for item in METHODS:
                method = item['id']; data = scored[method]
                if 'failure' in data:
                    p = {'value': None, 'weak_tail': False, 'reason': data['failure']}
                else:
                    endpoint = data['endpoints'][ep]
                    value = endpoint['value']
                    if value is not None:
                        value *= clinical_unit_factor(ep, row['unit'], weight) * factor
                    p = {'value': value, 'weak_tail': value is not None and ep in data['weak'],
                         'reason': endpoint.get('reason', '') if value is None else ''}
                if method in ('ours', 'rf'):
                    original_value = float(row[method+'_prediction'])*factor
                    assert math.isclose(p['value'], original_value, rel_tol=1e-10), (cid, ep, method)
                    assert p['weak_tail'] == (row[method+'_weak_tail_estimate'] == 'True')
                    # Preserve precisely the original stored display values.
                    p['value'] = original_value
                p['error'] = None if observed is None or p['value'] is None else (
                    abs(p['value']-observed) if ep == 'tmax' else max(p['value']/observed, observed/p['value']))
                p['error_metric'] = 'absolute_error_h' if ep == 'tmax' else 'fold_error'
                cell['predictions'][method] = p
            record['endpoints'][ep] = cell
        records.append(record)
    totals = {}
    for item in METHODS:
        m = item['id']; scores = [c['gmfe'][m] for c in records if c['gmfe'][m] is not None]
        endpoint_metrics = {}
        for ep in rows[:10]:
            key = ep['endpoint']
            entries = [c['endpoints'][key]['predictions'][m] for c in records]
            es = [p['error'] for p in entries if p['error'] is not None]
            endpoint_metrics[key] = {'n': len(es), 'weak_scored_n': sum(p['error'] is not None and p['weak_tail'] for p in entries)}
            if es:
                endpoint_metrics[key].update({'mean_absolute_error_h': sum(es)/len(es)} if key == 'tmax' else
                    {'gmfe': math.exp(sum(math.log(e) for e in es)/len(es)), 'within_2fold': sum(e<=2 for e in es)})
        totals[m] = {'curve_drugs': len(scores), 'drug_equal_curve_gmfe': math.exp(sum(map(math.log, scores))/len(scores)) if scores else None,
                     'curve_gmfe_within_2fold': sum(x<=2 for x in scores), 'endpoint_metrics': endpoint_metrics}
    verify(pins)
    document = {'schema_version': 'additional_clinical_comparison_v1', 'methods': METHODS, 'cases': records, 'summary': totals,
        'source_pins': pins, 'training': ml['training'],
        'interpretation': 'Exploratory comparison on already-viewed drugs, with two information sets. RF/SVR/XGB receive no development curve; Ours and 2-compartment receive the same development curves. No cross-group algorithm-superiority claim.',
        'tuning_budget': 'RF frozen defaults; SVR three configurations and XGBoost two configurations by training-only three-fold scaffold CV. Tuning budgets differ.',
        'tmax_interpretation': 'Sampled Tmax is strongly constrained by infusion duration and sampling; not a primary discriminator of model quality.',
        'original_values_preserved': True, 'evaluation_outcomes_used_for_new_fit_selection': False}
    save(OUT, document)
    print(json.dumps({m: {k:v for k,v in t.items() if k != 'endpoint_metrics'} for m,t in totals.items()}, indent=2))


if __name__ == '__main__': main()
