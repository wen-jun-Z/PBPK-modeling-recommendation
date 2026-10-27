#!/usr/bin/env python3
"""Independently audit frozen direct CL predictions and publish one comparison.

No fitting, prediction, or model selection is performed. ``--check-only`` never
writes files. The older parameter and recalibration audits remain in use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

import report_cl_gap_comparison as legacy

ROOT, RUN, TABLE = legacy.ROOT, legacy.RUN, legacy.TABLE
DIRECT = RUN / 'direct_prediction'
METHODS, ENDPOINTS, COMMON = legacy.METHODS, legacy.ENDPOINTS, legacy.COMMON
SEED, DRAWS = legacy.SEED, legacy.DRAWS
require, near, read, path, sha = legacy.require, legacy.near, legacy.read, legacy.path, legacy.sha
FORMULAS = ('PKSimStandard', 'RodgersRowland')
NAMES = dict(rf='RF', tabpfn='TabPFNv2', complementary='Complementary CL')
SCOPES = ('changed_twenty_four', 'all_thirty')


def arithmetic_clustered(values, clusters):
    """Drug-equal arithmetic mean with whole-study-cluster resampling."""
    require(set(values) == set(clusters), 'Misaligned arithmetic cluster values')
    require(all(math.isfinite(value) for value in values.values()), 'Nonfinite arithmetic error')
    if not values:
        return dict(n_cases=0, n_clusters=0, estimate=None, ci95=None)
    keys, groups = sorted(values), sorted(set(clusters.values()))
    sums = np.asarray([math.fsum(values[k] for k in keys if clusters[k] == group) for group in groups])
    counts = np.asarray([sum(clusters[k] == group for k in keys) for group in groups])
    result = dict(n_cases=len(keys), n_clusters=len(groups),
                  estimate=math.fsum(values.values()) / len(keys))
    if len(groups) < 2:
        return dict(**result, ci95=None)
    draws = np.random.default_rng(SEED).integers(len(groups), size=(DRAWS, len(groups)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(**result, ci95=np.quantile(estimates, [.025, .975]).tolist(),
                resamples=DRAWS, seed=SEED)


def summarize(records):
    """Rebuild every metric and paired comparison from audited case errors."""
    indexed = {(r['method'], r['case_id']): r for r in records}
    require(len(indexed) == len(records) == 90, 'Expected 90 unique stage records')
    summary = {}
    for scope in SCOPES:
        ids = sorted(r['case_id'] for r in records if r['method'] == 'rf'
                     and (scope == 'all_thirty' or r['replaceable']))
        require(len(ids) == (30 if scope == 'all_thirty' else 24), 'Stage cohort changed')
        clusters = {cid: indexed[('rf', cid)]['cluster_id'] for cid in ids}
        summary[scope] = {}
        for method in METHODS:
            rows = {cid: indexed[(method, cid)] for cid in ids}
            metrics, pairs = {}, {}
            for endpoint in ENDPOINTS:
                values = {cid: row['errors'][endpoint] for cid, row in rows.items()
                          if row['errors'][endpoint] is not None}
                stats = (arithmetic_clustered if endpoint == 'tmax' else legacy.clustered)(
                    values, {cid: clusters[cid] for cid in values})
                stats.update(attempted=len(ids), unavailable_case_ids=[cid for cid in ids if cid not in values])
                stats.update(unavailable=len(ids)-len(values),
                    weak_prediction_n=sum(endpoint in r['weak_endpoints'] for r in rows.values()),
                    weak_reference_n=sum(endpoint in r['reference_weak_endpoints'] for r in rows.values()))
                if endpoint == 'tmax':
                    stats.update(n=len(values), mae_h=stats['estimate'])
                else:
                    stats['within_2fold'] = sum(value <= 2 for value in values.values())
                metrics[endpoint] = stats
                if method == 'rf':
                    continue
                paired = {}
                for cid, value in values.items():
                    baseline = indexed[('rf', cid)]['errors'][endpoint]
                    if baseline is not None:
                        paired[cid] = value - baseline if endpoint == 'tmax' else value / baseline
                stats = (arithmetic_clustered if endpoint == 'tmax' else legacy.clustered)(
                    paired, {cid: clusters[cid] for cid in paired})
                threshold = 0 if endpoint == 'tmax' else 1
                stats.update(attempted_pairs=len(ids),
                             complete_pairs=len(paired),
                             incomplete_case_ids=[cid for cid in ids if cid not in paired],
                             per_case_difference_h=paired if endpoint == 'tmax' else None,
                             per_case_ratio=paired if endpoint != 'tmax' else None,
                             metric='candidate_minus_rf_absolute_error_h' if endpoint == 'tmax' else 'candidate_over_rf_fold_error',
                             improved=sum(value < threshold - 1e-6 for value in paired.values()),
                             worsened=sum(value > threshold + 1e-6 for value in paired.values()),
                             tied=sum(abs(value - threshold) <= 1e-6 for value in paired.values()))
                pairs[endpoint] = stats
            summary[scope][method] = dict(enrolled=len(ids),
                predicted=sum(row['status'] == 'predicted' for row in rows.values()), metrics=metrics)
            if method != 'rf':
                summary[scope][method]['paired_to_rf'] = pairs
    return summary


def compare_summary(actual, stored, context):
    for scope in SCOPES:
        for method in METHODS:
            a, b = actual[scope][method], stored[scope][method]
            for field in ('enrolled', 'predicted'):
                require(a[field] == b[field], context + ': ' + field)
            for endpoint in ENDPOINTS:
                aa, bb = a['metrics'][endpoint], b['metrics'][endpoint]
                if endpoint == 'tmax':
                    require(aa['n'] == bb['n'], context + ': Tmax denominator')
                    near(aa['mae_h'], bb['mae_h'], context + ': Tmax MAE')
                else:
                    legacy.verify_statistics(aa, bb, context + '/' + scope + '/' + method + '/' + endpoint)
                    require(aa['within_2fold'] == bb['within_2fold'], context + ': twofold coverage')
                for field in ('unavailable', 'weak_prediction_n', 'weak_reference_n'):
                    if field in bb:
                        require(aa[field] == bb[field], context + ': ' + field)
                if method != 'rf' and endpoint in b.get('paired_to_rf', {}):
                    aa, bb = a['paired_to_rf'][endpoint], b['paired_to_rf'][endpoint]
                    legacy.verify_statistics(aa, bb, context + ': paired ' + endpoint)
                    for field in ('improved', 'worsened', 'tied'):
                        require(aa[field] == bb[field], context + ': paired ' + field)
                    for field in ('attempted_pairs', 'complete_pairs', 'incomplete_case_ids'):
                        if field in bb:
                            require(aa[field] == bb[field], context + ': paired ' + field)
                    if 'per_case_ratio' in bb:
                        require(set(aa['per_case_ratio']) == set(bb['per_case_ratio']), context + ': paired case coverage')
                        for cid, value in aa['per_case_ratio'].items():
                            near(value, bb['per_case_ratio'][cid], context + ': paired case ratio')


def verify_score(stored, recalculated, context):
    """Check scalar outputs, coverage, diagnostics and errors, not only GMFE."""
    require(set(stored['endpoints']) == set(ENDPOINTS[1:]), context + ': endpoint set')
    for endpoint in ENDPOINTS:
        near(legacy.endpoint_error(stored, endpoint), legacy.endpoint_error(recalculated, endpoint),
             context + '/' + endpoint + '/error')
        if endpoint != 'curve':
            a, b = stored['endpoints'][endpoint], recalculated['endpoints'][endpoint]
            for field in ('value', 'reference'):
                near(a[field], b[field], context + '/' + endpoint + '/' + field)
            for field in ('reason', 'reference_reason', 'unit', 'error_metric', 'weak_tail',
                          'reference_weak_tail', 'reference_kind', 'reference_curve_statistic'):
                require(a[field] == b[field], context + '/' + endpoint + '/' + field)
    for field in ('n', 'observed_points', 'zero_observations_not_log_scored', 'reason'):
        require(stored['curve_metrics'].get(field) == recalculated['curve_metrics'].get(field),
                context + ': curve ' + field)
    near(stored['curve_metrics']['ln_rmse'], recalculated['curve_metrics']['ln_rmse'], context + ': curve RMSE')
    for field in ('weak_endpoints', 'reference_weak_endpoints', 'nca_status'):
        require(stored[field] == recalculated[field], context + ': ' + field)


def score_inline(case, unit):
    """The frozen scorer accepts the hash-locked inline trajectory and request."""
    import blueprint_sampling_expansion_data as scorer
    return scorer.score_case(case, unit)


def csv_audit(records):
    with (DIRECT / 'endpoints.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(r['formula'], r['method'], r['case_id'], r['endpoint']): r for r in rows}
    expected = {(r['formula'], r['method'], r['case_id'], endpoint)
                for r in records for endpoint in ENDPOINTS}
    require(len(rows) == len(indexed) == 1980 and set(indexed) == expected,
            'Direct CSV must retain all 1,980 attempted endpoint outputs')
    for record in records:
        for endpoint in ENDPOINTS:
            row = indexed[(record['formula'], record['method'], record['case_id'], endpoint)]
            require(row['status'] == record['status'] and row['replaceable'] == str(record['replaceable']),
                    'Direct CSV status/cohort mismatch')
            near(float(row['error']) if row['error'] else None,
                 legacy.endpoint_error(record['score'], endpoint), 'Direct CSV error')
            if endpoint == 'curve' or record['score'] is None:
                continue
            scalar = record['score']['endpoints'][endpoint]
            for field, source_field in [('prediction', 'value'), ('reference', 'reference')]:
                near(float(row[field]) if row[field] else None, scalar[source_field], 'Direct CSV scalar')
            for field in ('unit', 'error_metric', 'reason', 'reference_reason'):
                require(row[field] == scalar[field], 'Direct CSV ' + field)
            for field in ('weak_tail', 'reference_weak_tail'):
                require(row[field] == str(scalar[field]), 'Direct CSV ' + field)
    return len(rows)


def coverage(records):
    values = {}
    for method in METHODS:
        selected = [r for r in records if r['method'] == method]
        values[method] = {}
        for endpoint in ENDPOINTS:
            reasons = Counter()
            model_available = 0
            for row in selected:
                score = row.get('score')
                if score is None:
                    reasons[row.get('reason') or row['status']] += 1
                    continue
                if endpoint == 'curve':
                    available = score['curve_metrics']['gmfe'] is not None
                    reason = score['curve_metrics'].get('reason')
                else:
                    scalar = score['endpoints'][endpoint]
                    available = scalar['value'] is not None
                    reason = scalar['reason'] or scalar['reference_reason']
                model_available += available
                if reason:
                    reasons[reason] += 1
            values[method][endpoint] = dict(attempted=len(selected), model_available=model_available,
                error_available=sum(legacy.endpoint_error(r.get('score'), endpoint) is not None for r in selected),
                unavailable_reasons=dict(reasons))
    return values


def cross_stage_pairs(numerator, denominator):
    """Calibrated/direct error comparison; it is not a CL-only causal effect."""
    a = {(row['method'], row['case_id']): row for row in numerator}
    b = {(row['method'], row['case_id']): row for row in denominator}
    require(set(a) == set(b), 'Cross-stage paired cases differ')
    summary = {}
    for scope in SCOPES:
        ids = sorted(cid for method, cid in a if method == 'rf'
                     and (scope == 'all_thirty' or a[(method, cid)]['replaceable']))
        summary[scope] = {}
        for method in METHODS:
            paired = {}
            for endpoint in ENDPOINTS:
                values = {}
                for cid in ids:
                    av, bv = a[(method, cid)]['errors'][endpoint], b[(method, cid)]['errors'][endpoint]
                    if av is not None and bv is not None:
                        values[cid] = av - bv if endpoint == 'tmax' else av / bv
                clusters = {cid: a[(method, cid)]['cluster_id'] for cid in values}
                stats = (arithmetic_clustered if endpoint == 'tmax' else legacy.clustered)(values, clusters)
                threshold = 0 if endpoint == 'tmax' else 1
                stats.update(attempted_pairs=len(ids), complete_pairs=len(values),
                    incomplete_case_ids=[cid for cid in ids if cid not in values],
                    per_case_difference_h=values if endpoint == 'tmax' else None,
                    per_case_ratio=values if endpoint != 'tmax' else None,
                    improved=sum(value < threshold-1e-6 for value in values.values()),
                    worsened=sum(value > threshold+1e-6 for value in values.values()),
                    tied=sum(abs(value-threshold) <= 1e-6 for value in values.values()),
                    interpretation='Different information regimes; not an isolated causal effect of CL recalibration')
                paired[endpoint] = stats
            summary[scope][method] = paired
    return summary


def direct_audit(sources, parameter_lock, recalibrated_records):
    protocol = read(DIRECT / 'protocol.json')
    sources.pins(protocol['source_pins'])
    sources.references(protocol)
    require(protocol['formulas'] == list(FORMULAS) and protocol['methods'] == list(METHODS),
            'Direct fixed formulas/methods changed')
    require(protocol['priors'] == parameter_lock['priors'], 'Direct priors differ from frozen parameter priors')
    require(protocol['analysis']['seed'] == SEED and protocol['analysis']['cluster_resamples'] == DRAWS,
            'Direct bootstrap specification changed')
    require(protocol['primary_formula'] == 'PKSimStandard' and protocol['sensitivity_formula'] == 'RodgersRowland'
            and protocol['log_parameters'] == [0.0] * 4 and not protocol['calibration']
            and not protocol['observation_driven_repair'] and protocol['direct_development_points'] == 0
            and protocol['calibrated_development_points'] == 5, 'Direct information regime changed')
    require(protocol['analysis']['ties_relative_tolerance'] == 1e-6
            and protocol['analysis']['primary'] == 'curve', 'Direct primary outcome/tie definition changed')
    bundle = read(DIRECT / 'predictions.json')
    result = read(DIRECT / 'results.json')
    state = read(DIRECT / 'state.json')
    require(state['status'] == 'complete' and state['completed'] == state['total'] == 180,
            'Direct batch is not complete')
    sources.references(state)
    sources.references(bundle)
    sources.references(result)
    protocol_ref = dict(path=str((DIRECT / 'protocol.json').relative_to(ROOT)), sha256=sha(DIRECT / 'protocol.json'))
    require(bundle['protocol'] == protocol_ref and result['protocol'] == protocol_ref,
            'Direct protocol not locked in predictions/results')
    require(result['predictions']['sha256'] == sha(DIRECT / 'predictions.json'), 'Prediction bundle differs after scoring')
    locked_at = bundle['all_predictions_locked_at']
    require(result['predictions_locked_at'] == locked_at and locked_at < result['scored_at'],
            'Direct scoring does not follow the complete prediction lock')
    cases = {case['case_id']: case for case in protocol['cases']}
    require(len(cases) == len(protocol['cases']) == 30, 'Direct clinical cohort changed')
    jobs = protocol['jobs']
    require(set(cases) == set(jobs) and sum(job['replaceable'] for job in jobs.values()) == 24,
            'Direct changed/protected cohort changed')
    expected = {(formula, method, cid) for formula in FORMULAS for method in METHODS for cid in cases}
    records = {(r['formula'], r['method'], r['case_id']): r for r in result['records']}
    require(set(records) == expected and len(records) == len(result['records']) == 180,
            'All 180 direct attempts must remain represented')
    require(set(bundle['units']) == {'__'.join(key) for key in expected},
            'Direct predictions must all lock before scoring')
    priors = {method: sources.document(parameter_lock['priors'][method]) for method in METHODS}
    previous_protocol = read(RUN / 'protocol.json')
    previous_cases = {case['case_id']: case for case in previous_protocol['cases']}
    require(set(cases) == set(previous_cases), 'Direct versus recalibrated cohort differs')
    for cid, case in cases.items():
        for field in ('compound_id', 'cluster_id', 'reference_source', 'target_contract'):
            require(case[field] == previous_cases[cid][field], 'Direct clinical contract differs: ' + field)
        require(jobs[cid]['replaceable'] == previous_protocol['jobs'][cid]['replaceable'], 'Changed CL cohort differs')
    physiology_cache = sources.document(protocol['source_protocol'])['physiology_cache']
    for formula in FORMULAS:
        for cid in cases:
            reference = protocol['physiology_refs'][formula + '__' + cid]
            require(reference == physiology_cache[cid + '__target__' + formula], 'Direct physiology differs from pristine cache')
            physiology = sources.document(reference)
            require(physiology['observations_used'] is False and physiology['solver_calls'] == 0,
                    'Direct physiology is calibrated')
    unchanged, protected = {}, {}
    audited = {formula: [] for formula in FORMULAS}
    failures = {formula: [] for formula in FORMULAS}
    for formula, method, cid in sorted(expected):
        record = records[(formula, method, cid)]
        key = '__'.join((formula, method, cid))
        require(record['prediction_key'] == key, 'Incorrect direct prediction key')
        unit = bundle['units'][key]
        require(unit['locked_at'] <= locked_at and unit['formula'] == formula
                and unit['case_id'] == cid and unit['method'] == method, 'Individual direct lock/identity changed')
        require(unit['status'] == record['status'] and record['replaceable'] == jobs[cid]['replaceable'],
                'Direct attempt status/cohort mismatch')
        require(record['cluster_id'] == cases[cid]['cluster_id'], 'Direct cluster membership changed')
        require(not unit['calibration_performed'] and not unit['repair_performed'], 'Direct calibration/repair flag')
        require(unit['prior'] == parameter_lock['priors'][method], 'Direct unit prior differs')
        require(unit['physiology'] == protocol['physiology_refs'][formula + '__' + cid], 'Direct unit physiology differs')
        if unit['status'] == 'predicted':
            require(unit['log_parameters'] == [0.0] * 4, 'Direct prediction used a calibrated multiplier')
            require(not unit['development_observations_used'] and not unit['target_observations_used'],
                    'Direct prediction used clinical observations')
            request = unit['request']
            original = protocol['originals'][formula + '__' + cid]
            require(request['partition_method'] == formula, 'Direct partition formula differs from locked choice')
            cl = [item for item in request['evidence'] if item['parameter'] == 'total_clearance']
            require(len(cl) == 1 and cl[0]['unit'].lower() == 'ml/min/kg', 'Direct CL evidence/units')
            if jobs[cid]['replaceable']:
                near(cl[0]['value'], priors[method]['predictions'][jobs[cid]['compound_id']], 'Direct CL prior replacement')
                require(len(unit['changes']) == 1 and unit['changes'][0]['parameter'] == 'total_clearance',
                        'Direct changes are not CL-only')
            else:
                require(unit['changes'] == [] and request == original, 'Protected direct input changed')
                if (formula, cid) in protected:
                    require(protected[(formula, cid)] == unit['prediction'], 'Protected direct prediction differs across predictors')
                protected[(formula, cid)] = unit['prediction']
            comparable, base_request = deepcopy(request), deepcopy(original)
            for item in (comparable, base_request):
                changed_cl = next(row for row in item['evidence'] if row['parameter'] == 'total_clearance')
                for field in ('value', 'source', 'prediction_method', 'notes'):
                    changed_cl.pop(field, None)
            require(comparable == base_request, 'Direct request changes extend beyond CL: ' + cid)
            comparable.pop('partition_method')
            require(cid not in unchanged or unchanged[cid] == comparable,
                    'Direct non-CL request differs across predictors/formulas: ' + cid)
            unchanged[cid] = comparable
            clearance = cl[0]['value'] * request['human']['weight_kg'] * 60 / 1000
            near(unit['effective_total_clearance_l_h'], clearance, 'Direct effective CL')
            near(unit['cl_input']['total_clearance_l_h'], clearance, 'Direct converted CL input')
            require(unit['prediction'] is not None, 'Successful direct attempt has no trajectory')
            recalculated = score_inline(cases[cid], unit)
            verify_score(record['score'], recalculated, key)
        else:
            require(record['score'] is None, 'Failed direct attempt has a score')
            failures[formula].append(dict(method=method, case_id=cid, status=unit['status'],
                                          reason=unit.get('reason', 'Unavailable')))
        audited[formula].append(dict(method=method, case_id=cid, cluster_id=record['cluster_id'],
            replaceable=record['replaceable'], status=record['status'],
            errors={endpoint: legacy.endpoint_error(record['score'], endpoint) for endpoint in ENDPOINTS},
            weak_endpoints=(record['score'] or {}).get('weak_endpoints', []),
            reference_weak_endpoints=(record['score'] or {}).get('reference_weak_endpoints', []),
            effective_total_clearance_l_h=unit.get('effective_total_clearance_l_h'),
            cl_input=unit.get('cl_input'), reason=unit.get('reason')))
    stages = {}
    stage_pairs = {}
    for formula in FORMULAS:
        summary = summarize(audited[formula])
        compare_summary(summary, result['summary'][formula], formula)
        stages['direct_' + formula] = dict(summary=summary, records=audited[formula],
            failed_attempts=failures[formula], endpoint_coverage=coverage(
                [row for row in result['records'] if row['formula'] == formula]),
            fixed_partition_method=formula, development_concentrations_used=False,
            calibration_performed=False, multipliers=[1.0] * 4)
        stage_pairs[formula] = cross_stage_pairs(recalibrated_records, audited[formula])
        for scope in SCOPES:
            for method in METHODS:
                for endpoint, stored in result['calibration_comparisons'][formula][scope][method].items():
                    actual = stage_pairs[formula][scope][method][endpoint]
                    legacy.verify_statistics(actual, stored, 'Cross-stage ' + formula + '/' + endpoint)
                    for field in ('attempted_pairs', 'complete_pairs', 'incomplete_case_ids', 'improved', 'worsened', 'tied'):
                        require(actual[field] == stored[field], 'Cross-stage ' + field)
                    require(set(actual['per_case_ratio']) == set(stored['per_case_ratio']), 'Cross-stage ratio coverage')
                    for cid, value in actual['per_case_ratio'].items():
                        near(value, stored['per_case_ratio'][cid], 'Cross-stage per-drug ratio')
    require(bundle['resources']['unique_prediction_attempts'] == 156 and bundle['resources']['logical_reuses'] == 24
            and bundle['resources']['calibrations'] == 0, 'Direct solve/reuse counts differ')
    successful_unique = sum(unit['status'] == 'predicted' and not unit.get('reused_from')
                            for unit in bundle['units'].values())
    require(bundle['resources']['successful_unique_predictions'] == successful_unique,
            'Direct unique-success count differs')
    require(csv_audit(result['records']) == 1980, 'Direct endpoint coverage')
    return stages, dict(prediction_attempts=180, endpoint_csv_rows=1980,
        predictions_locked_at=locked_at, scored_at=result['scored_at'],
        changed_cases=24, protected_cases=6,
        calibrated_over_direct=stage_pairs,
        all_case_scores_recomputed_from_locked_trajectories=True,
        all_scalar_values_and_errors_independently_recomputed=True,
        all_summaries_and_paired_cluster_bootstraps_independently_recomputed=True)


def fnum(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def interval(stat):
    if stat['estimate'] is None:
        return '--'
    ci = stat['ci95']
    return fnum(stat['estimate']) + (' [--]' if ci is None else f' [{ci[0]:.3f}, {ci[1]:.3f}]')


def table_text(audit):
    parameter, stages = audit['parameter'], audit['stages']
    display = [('direct_PKSimStandard', r'Direct prediction: fixed PK-Sim Standard (primary)'),
               ('five_point_recalibrated', r'Five-point recalibration: frozen earlier results'),
               ('direct_RodgersRowland', r'Direct prediction: fixed Rodgers--Rowland (sensitivity)')]
    lines = [r'\begingroup', r'\captionsetup[table]{name=Table}', r'\begin{table}[htbp]',
             r'\centering',
             r'\caption{CL gap-filler accuracy and downstream PBPK accuracy before and after five-point recalibration.}',
             r'\label{tab:blueprint_cl_gap_comparison}', r'\fontsize{8}{9.5}\selectfont',
             r'\setlength{\tabcolsep}{2.6pt}\renewcommand{\arraystretch}{1.04}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
             r'\multicolumn{6}{@{}l}{\textbf{A. Common 100-drug CL holdout (' + str(parameter['n_test_scaffolds']) + r' scaffolds)}}\\',
             r'Predictor & $n$ & GMFE & $\leq2$-fold & $\log_{10}$ RMSE & Ratio to RF [95\% CI]\\', r'\midrule']
    for method in METHODS:
        m = parameter['metrics'][method]
        ratio = '--' if method == 'rf' else interval(parameter['paired_to_rf'][method])
        lines.append(f"{NAMES[method]} & {m['n']} & {m['gmfe']:.3f} & {m['within_2fold']*100:.0f}\\% & {m['log10_rmse']:.3f} & {ratio}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\vspace{4pt}',
              r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}', r'\toprule',
              r'\multicolumn{8}{@{}l}{\textbf{B. Primary analysis: 24 drugs with replaced CL; six reference-supported metrics}}\\',
              r'Predictor & OK/$N$ & Curve & AUC$_{\rm window}$ & $C_{\max}$ & $T_{\max}$ (h) & $C_{\rm last}$ & $t_{1/2}$\\', r'\midrule']
    for stage, label in display:
        lines.append(r'\multicolumn{8}{@{}l}{\emph{' + label + r'}}\\')
        for method in METHODS:
            r = stages[stage]['summary']['changed_twenty_four'][method]
            cells = [NAMES[method], f"{r['predicted']}/{r['enrolled']}"]
            for endpoint in ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life'):
                m = r['metrics'][endpoint]
                cell = fnum(m['estimate'])
                if m['n_cases'] != r['enrolled']:
                    cell += f" ({m['n_cases']})"
                cells.append(cell)
            lines.append(' & '.join(cells) + r'\\')
        if stage != display[-1][0]:
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\vspace{4pt}',
              r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}', r'\toprule',
              r'\multicolumn{5}{@{}l}{\textbf{C. Paired curve-error comparison and 30-drug context}}\\',
              r'Stage & Candidate & 24: ratio [95\% CI] & I/W/T & 30: curve / ratio to RF\\', r'\midrule']
    stage_labels = dict(direct_PKSimStandard='Direct (primary)', five_point_recalibrated='Recalibrated',
                        direct_RodgersRowland='Direct (sensitivity)')
    for stage, _ in display:
        if stage == 'five_point_recalibrated':
            continue
        s = stages[stage]['summary']
        for method in METHODS[1:]:
            pair = s['changed_twenty_four'][method]['paired_to_rf']['curve']
            all_m = s['all_thirty'][method]
            all_pair = all_m['paired_to_rf']['curve']
            cells = [stage_labels[stage], NAMES[method], interval(pair),
                     f"{pair['improved']}/{pair['worsened']}/{pair['tied']}",
                     f"{fnum(all_m['metrics']['curve']['estimate'])} / {fnum(all_pair['estimate'])}"]
            lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\vspace{4pt}',
              r'\begin{minipage}{\linewidth}', r'\fontsize{7.3}{8.7}\selectfont\raggedright']
    lines.append(r'\textit{Design.} The same frozen RF, TabPFNv2 and complementary CL priors use 1,006 human CL labels from 1,008 training structures; the 100 holdout and all 30 clinical identities/scaffolds are excluded from training. RF is the common refit. TabPFNv2 uses 494 training-variance-selected Morgan bits plus six descriptors; complementary CL also uses animal labels. These are practical pipeline comparisons.\par')
    lines.append(r'\textit{Direct versus recalibrated.} Direct predictions use fixed partition formulas and all four multipliers equal to one; no development concentrations, calibration, clinical formula selection or repair are used. All 180 direct attempts lock before target scoring. Only the 24 RF-derived CL inputs are replaced; six independently supported CL inputs are protected. The earlier 90 five-point attempts are reused separately: formulas are selected using visible data, CL/tissue partitioning/exchange/blood partitioning are calibrated, and visible-data repair is allowed. All these protocol differences contribute to cross-stage differences.\par')
    failures = ', '.join(f"{stage_labels[stage]} {len(stages[stage]['failed_attempts'])}/90" for stage, _ in display)
    lines.append(r'\textit{Metrics and coverage.} GMFE averages per-drug fold errors geometrically; curve error first averages absolute log ratios within each drug. $T_{\max}$ is MAE in hours. OK/$N$ is successful predictions/attempts; parenthetical counts show smaller endpoint denominators. All 11 outputs are retained for 270 attempts (2,970 CSV rows across the two stages). AUC$_\infty$, CL, $V_z$, MRT and $V_{ss}$ have no common clinical references and are not error-scored. Failed predictions: ' + failures + r'. Each primary direct predictor has 8/24 weak-tail half-life estimates; these remain included with descriptive flags. Missing endpoints and reasons remain in the audit.\par')
    rf_context = '; '.join(stage_labels[stage] + ' ' + fnum(stages[stage]['summary']['all_thirty']['rf']['metrics']['curve']['estimate']) for stage, _ in display)
    n_clusters = stages['direct_PKSimStandard']['summary']['changed_twenty_four']['rf']['metrics']['curve']['n_clusters']
    lines.append(r'\textit{Paired analysis.} Ratios below one favor the candidate. I/W/T counts improved/worsened/tied drugs ($10^{-6}$ tolerance); incomplete pairs remain listed. The 24-drug analysis has ' + str(n_clusters) + r' study clusters. The 30-drug RF curve GMFEs are ' + rf_context + r'. Descriptive 95\% CIs use 2,000 paired scaffold/study-cluster bootstrap draws (parameter/clinical; seed 20261012), without multiplicity adjustment. The parameter holdout was previously viewed and the clinical cohort is retrospective.\par')
    lines.append(r'\textit{Interpretation.} Lower parameter error does not establish a primary direct curve-error gain: both candidate/RF CIs cross one. Near-equality after recalibration reflects absorption of starting CL differences and cannot rank the CL predictors. Direct accuracy depends on the fixed partition formula; cross-stage gains cannot be attributed to CL calibration alone.')
    lines += [r'\end{minipage}', r'\end{table}', r'\endgroup', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Audit all locked inputs without writing any file')
    args = parser.parse_args()
    sources = legacy.Sources()
    parameter, parameter_lock = legacy.parameter_audit(sources)
    recalibrated = legacy.clinical_audit(sources, parameter_lock)
    recalibrated['summary'] = summarize(recalibrated['records'])
    stages, direct = direct_audit(sources, parameter_lock, recalibrated['records'])
    stages['five_point_recalibrated'] = recalibrated
    for directory, names in [(RUN, ('parameter_protocol.json', 'parameter_metrics.json',
            'parameter_validation.json', 'parameter_predictions_locked.json', 'clinical_inputs.json',
            'protocol.json', 'state.json', 'results.json', 'endpoints.csv')),
            (DIRECT, ('protocol.json', 'predictions.json', 'results.json', 'endpoints.csv', 'state.json'))]:
        for name in names:
            file = directory / name
            sources.check(dict(path=str(file.relative_to(ROOT)), sha256=sha(file)))
    for file in (Path(__file__).resolve(), ROOT / 'scripts/report_cl_gap_comparison.py'):
        sources.check(dict(path=str(file.relative_to(ROOT)), sha256=sha(file)))
    audit = dict(version='direct-cl-gap-publication-1', status='validated', parameter=parameter,
        stages=stages, direct=direct, source_pins=sources.verified, methods=list(METHODS),
        common_reference_classes=list(COMMON), bootstrap_draws=DRAWS, bootstrap_seed=SEED,
        intervals_are_descriptive=True, multiplicity_adjusted=False,
        no_training_refitting_or_simulation_performed=True,
        validation='Source pins, all prediction locks, fixed formulas and multipliers, protected CL inputs, '
                   'saved-trajectory rescoring, complete endpoint CSVs and independent paired cluster statistics agree.')
    latex = table_text(audit)
    audit['table'] = dict(path=str(TABLE.relative_to(ROOT)), sha256=hashlib.sha256(latex.encode()).hexdigest())
    publication = DIRECT / 'publication.json'
    if args.check_only:
        require(TABLE.exists() and TABLE.read_text() == latex, 'Combined table differs from validated results')
        require(publication.exists() and read(publication) == audit, 'Direct publication differs from current audit')
    else:
        TABLE.write_text(latex)
        publication.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='validated', check_only=args.check_only, verified_sources=len(sources.verified),
        direct_rows=180, recalibrated_rows=90, combined_endpoint_rows=2970,
        failed_direct_attempts=sum(len(stages['direct_' + formula]['failed_attempts']) for formula in FORMULAS),
        table=str(TABLE.relative_to(ROOT))), sort_keys=True))


if __name__ == '__main__':
    main()
