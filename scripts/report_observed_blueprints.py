#!/usr/bin/env python3
"""Audit completed observed-data blueprint runs without modifying or rerunning them.

Repeat --run LABEL=project/relative/run; the same label may combine disjoint
cohorts. Output contains compact JSON and one CSV row per case and method.
Numeric curve metrics are independently recomputed with NumPy, not score_case.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-9, 1e-10
UNKNOWN_REFERENCES = ('auc_inf', 'clearance', 'vz', 'mrt', 'vss')


def inside(value, *, exists=True):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve(strict=exists)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f'Path must stay inside the project: {value}')
    return path


def relative(path):
    return str(path.relative_to(ROOT))


class Sources:
    def __init__(self):
        self.hashes = {}

    def read(self, value, expected=None):
        path = inside(value)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        previous = self.hashes.get(relative(path))
        if (expected is not None and digest != expected) or (previous is not None and digest != previous):
            raise ValueError(f'Checksum mismatch: {relative(path)}')
        self.hashes[relative(path)] = digest
        return raw

    def json(self, value, expected=None):
        return json.loads(self.read(value, expected))

    def verify_unchanged(self):
        for path, digest in list(self.hashes.items()):
            self.read(path, digest)


def close(actual, expected, name):
    if isinstance(actual, bool) or isinstance(expected, bool):
        raise ValueError(f'Boolean numeric metric: {name}')
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Non-numeric metric: {name}') from exc
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        raise ValueError(f'Recomputed metric disagrees with source summary: {name}')


def series(times, concentrations, name, *, positive):
    if any(isinstance(x, bool) for x in [*times, *concentrations]):
        raise ValueError(f'Boolean curve coordinate: {name}')
    t, c = np.asarray(times, dtype=float), np.asarray(concentrations, dtype=float)
    if (t.ndim != 1 or c.ndim != 1 or len(t) < 2 or len(t) != len(c)
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c))
            or np.any(t < 0) or np.any(np.diff(t) <= 0)
            or np.any(c <= 0 if positive else c < 0)):
        raise ValueError(f'Invalid curve: {name}')
    return t, c


def observed_series(document):
    if document.get('matrix') != 'plasma' or document.get('time_unit') != 'min' or document.get('unit') != 'umol/l':
        raise ValueError('Observations must declare plasma, min and umol/l')
    rows = document['observations']
    if any(r.get('time_unit', 'min') != 'min' or r.get('unit', 'umol/l') != 'umol/l' for r in rows):
        raise ValueError('Per-row observation unit mismatch')
    return series([r['time_min'] for r in rows], [r['concentration_umol_per_l'] for r in rows], 'observations', positive=True)


def recompute(curve, times, observed):
    if curve.get('time_unit') != 'min' or curve.get('unit') != 'umol/l':
        raise ValueError('Prediction must declare min and umol/l')
    pt, pc = series(curve['time_min'], curve['concentration'], 'prediction', positive=False)
    if times[0] < pt[0] or times[-1] > pt[-1]:
        raise ValueError('Prediction does not cover observations; extrapolation is forbidden')
    predicted = np.interp(times, pt, pc)
    if np.any(predicted <= 0):
        raise ValueError('Nonpositive prediction at an observation time')
    error = np.log(predicted / observed)
    auc = lambda c: float(np.sum(np.diff(times)/60 * (c[:-1]+c[1:])/2))
    fold = lambda a, b: max(a/b, b/a)
    ao, ap = auc(observed), auc(predicted)
    co, cp = float(np.max(observed)), float(np.max(predicted))
    lo, lp = float(observed[-1]), float(predicted[-1])
    to, tp = float(times[np.argmax(observed)]/60), float(times[np.argmax(predicted)]/60)
    metrics = {
        'point_count': len(times), 'gmfe': float(np.exp(np.mean(np.abs(error)))),
        'log_rmse': float(np.sqrt(np.mean(error**2))),
        'observed_window_h': [float(times[0]/60), float(times[-1]/60)],
        'auc': {'observed_umol_h_l': ao, 'predicted_umol_h_l': ap, 'fold_error': fold(ao, ap)},
        'sampled_cmax': {'observed_umol_l': co, 'predicted_umol_l': cp, 'fold_error': fold(co, cp)},
        'c_last': {'observed_umol_l': lo, 'predicted_umol_l': lp, 'fold_error': fold(lo, lp)},
        'sampled_tmax': {'observed_h': to, 'predicted_h': tp, 'absolute_error_h': abs(tp-to)},
    }
    return metrics, predicted


def verify_metrics(metrics, predicted, times, observed, original):
    for name, value in metrics.items():
        if isinstance(value, dict):
            for key, number in value.items():
                close(number, original[name][key], name+'.'+key)
        else:
            close(value, original[name], name)
    for name, values in (('time_min', times), ('observed_umol_l', observed), ('predicted_umol_l', predicted)):
        close(values, original[name], name)


def compact_endpoint(value):
    """Retain scalar endpoint predictions without duplicating native curves."""
    return {k: v for k, v in value.items()
            if k in ('status', 'reason', 'definition', 'units', 'model_prefix_end_h') or k.startswith('value')}


def verify_terminal(times, values, case, original, name):
    """Independently check the existing fixed-window half-life quality gates."""
    hours = times/60
    spec = case['evaluation_target'].get('terminal_fit')
    dose = case['request']['dosing']; start = dose.get('start_h', 0.)
    infusion = dose.get('infusion_minutes', 0.)/60
    qualified = isinstance(spec, dict)
    half = None
    if qualified:
        lower, upper = spec['start_h'], spec['end_h']
        qualified = (dose['route'] == 'intravenous' and lower >= hours[0]
            and lower < upper == hours[-1] and lower >= start
            and (infusion == 0 or lower > start+infusion))
        mask = (hours >= lower) & (hours <= upper)
        x, c = hours[mask], values[mask]
        qualified = qualified and len(x) >= spec.get('min_points', 4)
        qualified = qualified and np.all(c > 0) and np.all(np.diff(c) < 0)
        postdose = (hours >= start) & (hours <= upper)
        if qualified and np.any(postdose):
            peak_time = hours[postdose][np.argmax(values[postdose])]
            qualified = not (lower <= peak_time and peak_time > start)
        else:
            qualified = False
        if qualified:
            y = np.log(c); xc, yc = x-np.mean(x), y-np.mean(y)
            ssy = float(np.sum(yc**2)); ssx = float(np.sum(xc**2))
            slope = float(np.sum(xc*yc)/ssx) if ssx > 0 else 0.
            qualified = slope < 0 and ssy > 0
            if qualified:
                half = math.log(2)/-slope
                r2 = float(np.clip(1-np.sum((yc-slope*xc)**2)/ssy, 0, 1))
                adjusted = 1-(1-r2)*(len(x)-1)/(len(x)-2)
                span = float((x[-1]-x[0])/half)
                qualified = (math.isfinite(half) and adjusted >= spec.get('min_adjusted_r2', .98)
                    and span >= spec.get('min_span_half_lives', 1.))
                diagnostics = original.get('diagnostics', {})
                for key, number in (('lambda_z_per_h', -slope), ('adjusted_r_squared', adjusted), ('span_half_lives', span)):
                    close(number, diagnostics.get(key), name+'.'+key)
    if bool(qualified) != (original.get('status', 'completed') == 'completed'):
        raise ValueError(f'Terminal quality status differs from frozen rules: {name}')
    if qualified:
        close(half, original['value_h'], name+'.value_h')


def half_life(original, policy):
    observed, predicted = original['observed'], original['predicted']
    oq = observed.get('status', 'completed') == 'completed'
    pq = predicted.get('status', 'completed') == 'completed'
    if (original.get('status') == 'completed') != (oq and pq):
        raise ValueError('Terminal summary status disagrees with its observation/prediction qualifications')
    if oq and pq:
        a, b = observed['value_h'], predicted['value_h']
        close(max(a/b, b/a), original.get('fold_error'), 'terminal_half_life.fold_error')
    elif original.get('fold_error') is not None:
        raise ValueError('Unqualified terminal fit has a fold error')
    excluded = policy is not None and policy['reference_qualified'] is False
    return {
        'reference_h': observed.get('value_h') if oq and not excluded else None,
        'predicted_same_window_h': predicted.get('value_h') if pq else None,
        'fold_error': original.get('fold_error') if oq and pq and not excluded else None,
        'reference_qualified': bool(oq and not excluded),
        'reference_reason': policy['reason'] if excluded else observed.get('reason', 'Numeric gates passed; derived from the digitized clinical curve'),
        'observed_numeric_status': observed.get('status', 'completed'),
        'predicted_numeric_status': predicted.get('status', 'completed'),
        'predicted_reason': predicted.get('reason'),
        'numeric_diagnostics': {
            name: {k: v for k, v in endpoint.get('diagnostics', {}).items() if k != 'fit_times_h'}
            for name, endpoint in (('observed', observed), ('predicted', predicted))},
        'definition': 'Same predeclared terminal window and actual observation times; no independent individual endpoint measurement claimed',
    }


def load_policy(path, sources):
    policy = sources.json(path)
    if policy.get('version') != 'clinical-endpoint-reference-policy-1.0':
        raise ValueError('Unsupported reference policy version')
    for case, endpoints in policy['cases'].items():
        for name, rule in endpoints.items():
            if name != 'terminal_half_life' or rule['reference_qualified'] is not False:
                raise ValueError('Policy overrides may only disqualify a terminal reference')
            source = rule['source']; value = sources.json(source['path'], source['sha256'])
            for key in source['json_pointer'].lstrip('/').split('/'):
                value = value[key.replace('~1', '/').replace('~0', '~')]
            if value is not False:
                raise ValueError(f'Reference exclusion not supported by its source: {case}')
    return policy


def audit_run(label, directory, sources, policy):
    summary = sources.json(directory/'summary.json')
    protocol = sources.json(directory/'protocol.json')
    locks = sources.json(directory/'predictions_locked.json')['hashes']
    if summary['status'] not in ('completed', 'completed_with_failures'):
        raise ValueError('Only finished runs can be audited')
    # Algorithms can be upgraded after a run. Check frozen code against its
    # original archive, and data against the actual immutable source files.
    archive = zipfile.ZipFile(io.BytesIO(sources.read(directory/'code_snapshot.zip')))
    archived = set(archive.namelist())
    for name, digest in protocol['pins'].items():
        inside(name, exists=False)
        if name in archived:
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'Frozen code archive checksum mismatch: {name}')
        else:
            sources.read(name, digest)
    archive.close()
    raw_rows = {r['case_id']: r for r in summary['results']}
    cases = {c['case_id']: c for c in protocol['cases']}
    if len(raw_rows) != len(summary['results']) or len(cases) != len(protocol['cases']) or set(raw_rows) != set(cases):
        raise ValueError('Duplicate or inconsistent case inventory')
    if summary['case_count'] != len(cases) or summary['successful_predictions'] != len(locks) or not set(locks) <= set(cases):
        raise ValueError('Prediction coverage does not match frozen cases')
    records = []
    for cid, case in cases.items():
        original = raw_rows[cid]
        for key in ('case_id', 'drug', 'compound_id', 'study_id', 'role'):
            if original[key] != case[key]:
                raise ValueError(f'Case identity mismatch: {cid}/{key}')
        opath = case['observations_path']
        observed_doc = sources.json(opath, protocol['pins'][opath])
        if observed_doc['compound_id'] != case['compound_id'] or observed_doc['role'] != case['role']:
            raise ValueError('Observation identity or role mismatch')
        for source_key in ('source', 'protocol_source'):
            ref = observed_doc.get(source_key)
            if ref is not None:
                sources.read(ref['path'], ref['sha256'])
        times, observed = observed_series(observed_doc)
        record = {k: case[k] for k in ('case_id', 'drug', 'compound_id', 'study_id', 'role')}
        record.update(method=label, run=relative(directory), point_count=len(times),
            dose_mg=case['request']['dosing']['dose_mg'], route=case['request']['dosing']['route'],
            infusion_minutes=case['request']['dosing'].get('infusion_minutes'),
            prediction_status=original['result']['status'], scoring_status=original['scoring_status'],
            failure_reason=original.get('scoring_error') or original['result'].get('reason'),
            observation_sha256=protocol['pins'][opath], observations_path=opath,
            request_sha256=protocol['pins'][case['request_path']],
            evaluation_target=case['evaluation_target'], metrics=None, native_endpoints={})
        if (cid in locks) != (original['result']['status'] == 'predicted'):
            raise ValueError('Prediction lock inventory disagrees with status')
        if cid in locks:
            folder = inside(directory/'cases'/cid)
            if original['result']['prediction_sha256'] != locks[cid]:
                raise ValueError('Result prediction hash disagrees with lock')
            curve = sources.json(folder/'prediction_locked.json', locks[cid])
            actual_result = sources.json(folder/'result.json')
            if actual_result != original['result']:
                raise ValueError('Per-case result disagrees with summary')
            endpoints = sources.json(folder/'endpoints.json')['endpoints']
            record['native_endpoints'] = {k: compact_endpoint(v) for k, v in endpoints.items() if k != 'plasma_concentration_time'}
            record['prediction_sha256'] = locks[cid]
            if original['scoring_status'] == 'completed':
                metrics, predicted = recompute(curve, times, observed)
                verify_metrics(metrics, predicted, times, observed, original['metrics'])
                terminal = original['metrics']['terminal_half_life']
                verify_terminal(times, observed, case, terminal['observed'], cid+'/observed half-life')
                verify_terminal(times, predicted, case, terminal['predicted'], cid+'/predicted half-life')
                rule = policy['cases'].get(cid, {}).get('terminal_half_life')
                metrics['terminal_half_life'] = half_life(terminal, rule)
                record['metrics'] = metrics
        elif original['scoring_status'] == 'completed':
            raise ValueError('Scored case has no locked prediction')
        record['unknown_endpoint_references'] = {name: {'reference': None, 'error': None} for name in UNKNOWN_REFERENCES}
        records.append(record)
    return records, {'method': label, 'path': relative(directory), 'status': summary['status'],
        'recover_failed_trials': protocol.get('recover_failed_trials', False),
        'native_fit_calls': summary['native_fit_calls'], 'elapsed_seconds': summary['elapsed_seconds'],
        'case_count': len(cases), 'successful_predictions': len(locks),
        'frozen_input_and_code_pins_verified': len(protocol['pins'])}


def aggregate(records):
    output = {}
    metrics = {'gmfe': ('gmfe',), 'log_rmse': ('log_rmse',), 'auc_fold': ('auc', 'fold_error'),
        'sampled_cmax_fold': ('sampled_cmax', 'fold_error'), 'c_last_fold': ('c_last', 'fold_error'),
        'tmax_absolute_error_h': ('sampled_tmax', 'absolute_error_h'),
        'terminal_half_life_fold': ('terminal_half_life', 'fold_error')}
    for label in sorted({r['method'] for r in records}):
        output[label] = {}
        for role in sorted({r['role'] for r in records if r['method'] == label}):
            rows = [r for r in records if r['method'] == label and r['role'] == role]
            group = {'attempted_cases': len(rows), 'predicted_cases': sum(r['prediction_status'] == 'predicted' for r in rows),
                'audited_scored_cases': sum(r['metrics'] is not None for r in rows),
                'compound_count': len({r['compound_id'] for r in rows}), 'metrics': {}}
            for metric, path in metrics.items():
                by_study = defaultdict(list); eligible = []
                geometric = metric == 'gmfe' or metric.endswith('_fold')
                for row in rows:
                    value = row['metrics']
                    for key in path:
                        value = value.get(key) if isinstance(value, dict) else None
                    if value is not None:
                        if not math.isfinite(value) or value < 0 or (geometric and value <= 0):
                            raise ValueError('Invalid eligible aggregate metric')
                        eligible.append(value)
                        by_study[(row['compound_id'], row['study_id'])].append(math.log(value) if geometric else value)
                by_drug = defaultdict(list)
                for (drug, _), values in by_study.items():
                    by_drug[drug].append(float(np.mean(values)))
                means = {drug: float(np.mean(values)) for drug, values in by_drug.items()}
                transform = math.exp if geometric else float
                group['metrics'][metric] = {
                    'aggregation': 'geometric' if geometric else 'arithmetic',
                    'eligible_cases': len(eligible), 'attempted_cases': len(rows),
                    'eligible_studies': len(by_study), 'eligible_compounds': len(means),
                    'coverage': len(eligible)/len(rows),
                    'by_compound': {drug: transform(v) for drug, v in means.items()},
                    'macro_mean': transform(float(np.mean(list(means.values())))) if means else None,
                    'within_twofold_cases': sum(v <= 2 or math.isclose(v, 2, rel_tol=1e-12, abs_tol=1e-12) for v in eligible) if geometric else None,
                }
            output[label][role] = group
    return output


def csv_row(row):
    value = {k: row[k] for k in ('method', 'case_id', 'drug', 'compound_id', 'study_id', 'role',
        'dose_mg', 'route', 'infusion_minutes', 'point_count', 'prediction_status', 'scoring_status', 'failure_reason')}
    m = row['metrics'] or {}
    value.update(gmfe=m.get('gmfe'), ln_rmse=m.get('log_rmse'))
    for endpoint, keys in {
        'auc': ('observed_umol_h_l', 'predicted_umol_h_l', 'fold_error'),
        'sampled_cmax': ('observed_umol_l', 'predicted_umol_l', 'fold_error'),
        'c_last': ('observed_umol_l', 'predicted_umol_l', 'fold_error'),
        'sampled_tmax': ('observed_h', 'predicted_h', 'absolute_error_h'),
        'terminal_half_life': ('reference_h', 'predicted_same_window_h', 'fold_error', 'reference_qualified', 'reference_reason'),
    }.items():
        for key in keys:
            value[endpoint+'_'+key] = m.get(endpoint, {}).get(key)
    for name, field in {'terminal_half_life': 'value_h', 'auc_inf': 'value_umol_h_per_l',
        'clearance': 'value_l_per_h', 'vz': 'value_l', 'mrt': 'value_h', 'vss': 'value_l'}.items():
        endpoint = row['native_endpoints'].get(name, {})
        value['native_'+name+'_'+field] = endpoint.get(field)
        value['native_'+name+'_status'] = endpoint.get('status')
        value['native_'+name+'_reason'] = endpoint.get('reason')
        if name != 'terminal_half_life':
            value[name+'_reference'] = None
            value[name+'_fold_error'] = None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=PROJECT_RELATIVE_DIR')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-policy', required=True, type=Path)
    args = parser.parse_args(argv)
    output = inside(args.output, exists=False)
    destinations = [output/'audited_results.json', output/'clinical_comparison.csv']
    if any(p.exists() for p in destinations):
        raise FileExistsError('Audited outputs already exist; use a new directory or explicitly remove only those outputs')
    sources = Sources(); policy = load_policy(args.reference_policy, sources)
    records, runs, seen_runs = [], [], set()
    for item in args.run:
        label, separator, path = item.partition('=')
        if not separator or not label.strip() or Path(path).is_absolute():
            raise ValueError('--run requires LABEL=project_relative_directory')
        directory = inside(path)
        if directory in seen_runs:
            raise ValueError('A run directory was supplied more than once')
        seen_runs.add(directory)
        own, run = audit_run(label.strip(), directory, sources, policy)
        records.extend(own); runs.append(run)
    seen = set(); signatures = {}
    for row in records:
        key = (row['method'], row['case_id'])
        if key in seen:
            raise ValueError('Duplicate method/case; use distinct method labels')
        seen.add(key)
        signature = {k: row[k] for k in ('drug', 'compound_id', 'study_id', 'role', 'observation_sha256', 'request_sha256', 'evaluation_target')}
        if row['case_id'] in signatures and signature != signatures[row['case_id']]:
            raise ValueError('Compared methods have different frozen inputs or scoring windows')
        signatures[row['case_id']] = signature
    sources.verify_unchanged()
    labels = sorted({r['method'] for r in records}); all_cases = sorted(signatures)
    report = {'version': 'observed-blueprint-audit-1.0', 'audited_at': datetime.now(timezone.utc).isoformat(),
        'independent_validation': False, 'reference_policy': {'path': relative(inside(args.reference_policy)),
            'sha256': sources.hashes[relative(inside(args.reference_policy))]},
        'verification': {'independent_numpy_recomputation': True, 'relative_tolerance': RTOL, 'absolute_tolerance': ATOL,
            'checked': ['frozen code archive and source pins', 'prediction locks and observation hashes',
                'GMFE', 'natural-log RMSE', 'same-window AUC', 'sampled Cmax', 'Clast', 'Tmax absolute error',
                'fixed-window terminal regression and quality status'],
            'half_life': 'Existing fixed-window quality gates, with source-qualification exclusions applied equally to all methods'},
        'definitions': {'auc': 'Linear trapezoids over first-to-last real observation times; no extrapolation',
            'sampled_cmax': 'Maximum at actual observation times, not the unobserved continuous peak',
            'tmax': 'Earliest sampled maximum; report absolute time error, not a ratio',
            'aggregation': 'Within each role and method: average cases within study, studies within drug, then give drugs equal weight; geometric for fold metrics, arithmetic for log-RMSE and time error',
            'native_endpoints': 'Separate full model-window outputs; never substitute them for endpoints scored at observation times',
            'missing_references': 'Null; calibration CL inputs are not independent references'},
        'runs': runs, 'cases': records, 'aggregates': aggregate(records),
        'method_case_coverage': {label: {'present': [c for c in all_cases if (label, c) in seen],
            'missing': [c for c in all_cases if (label, c) not in seen]} for label in labels},
        'source_hashes': sources.hashes}
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    csv_text = io.StringIO(newline=''); csv_rows = [csv_row(r) for r in records]
    writer = csv.DictWriter(csv_text, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    output.mkdir(parents=True, exist_ok=True)
    for path, data in zip(destinations, (json_text, csv_text.getvalue())):
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(data)
    print(json.dumps({'status': 'audited', 'run_count': len(runs), 'case_method_rows': len(records),
        'audited_scored_rows': sum(r['metrics'] is not None for r in records), 'output': relative(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
