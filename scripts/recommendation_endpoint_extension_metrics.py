#!/usr/bin/env python3
"""Score matched PK endpoint extensions without changing any frozen ranking.

Published observations, native predictions and protocol-matched predictions have
separate identities. Original 171 scalar comparisons and six curves remain an
explicit immutable layer; extension summaries never silently pool those values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'experiments/model_recommendation/execution_validation_20261001'
PRIMARY = BASE / 'followup_20261003/primary/results.json'
PRIMARY_SHA = '8798a53c9a452817ba820cdadf948c9edcf5c5beff939c7813f2cd0cd46292e7'
REFS = ROOT / 'results/data/model_recommendation_extended_references.json'
PREDICTIONS = ROOT / 'results/data/model_recommendation_extended_predictions.json'
DEST = ROOT / 'results/data/model_recommendation_extended_metrics.json'
FAMILIES = ('curve', 'finite_AUC', 'Cmax', 'Tmax', 'C_last', 'half_life', 'AUC_infinity', 'CL', 'Vz', 'MRT', 'Vss')
ENDPOINTS = ('Cmax', 'Tmax', 'AUC_window', 'AUC_last', 'AUC_infinity', 'half_life', 'CL', 'Vz', 'MRT', 'Vss', 'C_last')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Artifact outside project')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def endpoint(row):
    value = row.get('endpoint', row.get('endpoint_key'))
    require(value in ENDPOINTS, 'Unknown physical endpoint: ' + str(value))
    return value


def family(value):
    return 'finite_AUC' if value in ('AUC_window', 'AUC_last') or value.startswith('AUC_0_') or value.startswith('AUC_tau') else value


def number(value):
    return isinstance(value, (float, int, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def normalized_unit(value, unit, molecular_weight=None):
    """Only exact dimensional conversions: never infer weight, dose or F."""
    if not number(value):
        raise ValueError('missing_or_nonfinite_value')
    text = str(unit or '').replace('μ', 'u').replace('µ', 'u').replace('·', '*').replace(' ', '').lower()
    aliases = {'hr': 'h', 'hours': 'h', 'hour': 'h', 'minutes': 'min', 'minute': 'min',
               'ug.h/ml': 'ug*h/ml', 'ng.h/ml': 'ng*h/ml', 'ng*h/ml': 'ng*h/ml',
               'ng.h.ml-1': 'ng*h/ml', 'l.h-1': 'l/h', 'ml.min-1': 'ml/min'}
    text = aliases.get(text, text)
    conversions = {
        'h': ('h', 1.), 'min': ('h', 1 / 60), 's': ('h', 1 / 3600), 'day': ('h', 24.), 'd': ('h', 24.),
        'ng/ml': ('ng/mL', 1.), 'ug/l': ('ng/mL', 1.), 'ug/ml': ('ng/mL', 1000.), 'mg/l': ('ng/mL', 1000.),
        'ng*h/ml': ('ng*h/mL', 1.), 'ng*min/ml': ('ng*h/mL', 1 / 60),
        'ug*h/ml': ('ng*h/mL', 1000.), 'mg*h/l': ('ng*h/mL', 1000.),
        'ug*h/l': ('ng*h/mL', 1.),
        'l/h': ('L/h', 1.), 'ml/min': ('L/h', .06), 'ml/h': ('L/h', .001),
        'l/h/kg': ('L/h/kg', 1.), 'ml/min/kg': ('L/h/kg', .06), 'ml/h/kg': ('L/h/kg', .001),
        'l': ('L', 1.), 'ml': ('L', .001), 'l/kg': ('L/kg', 1.), 'ml/kg': ('L/kg', .001),
        '1': ('fraction', 1.), 'fraction': ('fraction', 1.), '%': ('fraction', .01),
        'kg*ng/(ml*mg)': ('kg*ng/(mL*mg)', 1.),
        'kg*ng*h/(ml*mg)': ('kg*ng*h/(mL*mg)', 1.)}
    molar = {'nm': 'ng/mL', 'nmol/l': 'ng/mL', 'nm*h': 'ng*h/mL', 'h*nmol/l': 'ng*h/mL'}
    if text in molar:
        if not number(molecular_weight) or molecular_weight <= 0:
            raise ValueError('molecular_weight_required_for_molar_conversion')
        return float(value) * molecular_weight / 1000., molar[text]
    if text not in conversions:
        raise ValueError('unsupported_unit:' + str(unit))
    canonical, factor = conversions[text]
    return float(value) * factor, canonical


def window(row):
    value = row.get('window_h')
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value.get('start', value.get('start_h')), value.get('end', value.get('end_h'))]
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1] is None:
        require(endpoint(row) in ('AUC_infinity', 'half_life', 'CL', 'Vz', 'MRT', 'Vss'),
                'Infinite window is invalid for a finite endpoint')
        return None
    require(isinstance(value, (list, tuple)) and len(value) == 2 and all(number(x) for x in value)
            and value[1] > value[0], 'Invalid endpoint window')
    return tuple(float(x) for x in value)


def descriptor(row, scenario):
    ep = endpoint(row)
    result = {'endpoint': ep, 'family': family(ep), 'quantity': row.get('quantity') or ep,
              'route': row.get('route') or scenario.get('route'),
              'steady_state': row.get('steady_state'),
              'body_weight_normalized': row.get('body_weight_normalized'),
              'dose_normalization': row.get('dose_normalization'),
              'window_h': list(window(row)) if ep in ('AUC_window', 'C_last') and window(row) is not None else None}
    if ep == 'AUC_last':
        result['comparison_qualification'] = 'Clinical nominal-sampling NCA approximation using predicted last quantifiable time; individual observed tlast is unavailable.'
    # Only fixed-window AUC is a different estimand at every integration window.
    # AUClast retains its published last-quantifiable definition, not fixed AUC.
    parts = [ep, str(result['quantity']), 'route=' + str(result['route']),
             'steady=' + str(result['steady_state']), 'weight_norm=' + str(result['body_weight_normalized']),
             'dose_norm=' + json.dumps(result['dose_normalization'], sort_keys=True, separators=(',', ':'))]
    if ep in ('AUC_window', 'C_last'):
        parts.append('window=' + json.dumps(result['window_h'], separators=(',', ':')))
    result['endpoint_key'] = '|'.join(parts)
    return result


def match_pair(observed, predicted):
    """Return an auditable rejection or correctly scaled, definition-matched error."""
    output = {'scorable': False, 'state': 'unavailable', 'reason': None,
              'predicted': None, 'observed': None, 'unit': None,
              'absolute_log_error': None, 'signed_log_error': None, 'absolute_error_h': None,
              'within_twofold': None}

    def reject(reason):
        return {**output, 'reason': reason}

    if observed is None:
        return reject('no_reference_observation')
    if observed.get('eligibility') is not True:
        return reject('reference_not_eligible:' + str(observed.get('reason', 'definition_not_qualified')))
    if predicted is None:
        return reject('no_prediction')
    if predicted.get('eligibility') is not True:
        return reject('prediction_not_eligible:' + str(predicted.get('reason', 'definition_not_qualified')))
    if observed.get('scenario_id') != predicted.get('scenario_id'):
        return reject('different_scenario')
    if (observed.get('publication_cluster_id') and predicted.get('publication_cluster_id')
            and observed['publication_cluster_id'] != predicted['publication_cluster_id']):
        return reject('different_publication')
    ep = endpoint(observed)
    if ep != endpoint(predicted):
        return reject('different_endpoint')
    for key in ('quantity', 'steady_state', 'body_weight_normalized'):
        a = observed.get(key, ep if key == 'quantity' else None)
        b = predicted.get(key, ep if key == 'quantity' else None)
        if a != b:
            return reject('different_' + key)
        if key != 'quantity' and a is None:
            return reject('unspecified_' + key)
    if observed.get('dose_normalization') != predicted.get('dose_normalization'):
        return reject('different_dose_normalization')
    if observed.get('route') and predicted.get('route') and observed['route'] != predicted['route']:
        return reject('different_route')
    for key in ('matrix', 'binding'):
        a = str(observed.get(key) or '').lower().replace(' ', '_').replace('-', '_')
        b = str(predicted.get(key) or '').lower().replace(' ', '_').replace('-', '_')
        if a and b and a != b:
            return reject('different_' + key)
    ow, pw = window(observed), window(predicted)
    if ep == 'AUC_window' and (ow is None or pw is None):
        return reject('fixed_auc_window_not_specified')
    if ep in ('AUC_window', 'AUC_last', 'C_last', 'Tmax', 'Cmax') and ow is not None:
        if pw is None or not np.allclose(ow, pw, rtol=0, atol=1e-9):
            return reject('different_endpoint_window')
    try:
        molecular_weight = predicted.get('molecular_weight_g_per_mol')
        obs, ou = normalized_unit(observed.get('value'), observed.get('unit'), molecular_weight)
        pred, pu = normalized_unit(predicted.get('value'), predicted.get('unit'), molecular_weight)
    except ValueError as error:
        return reject(str(error))
    if ou != pu:
        return reject('incompatible_units:' + ou + ':' + pu)
    output.update(predicted=pred, observed=obs, unit=ou)
    if ep == 'Tmax':
        if ou != 'h' or min(obs, pred) < 0:
            return reject('invalid_tmax_unit_or_negative_time')
        output['absolute_error_h'] = abs(pred - obs)
    else:
        if min(obs, pred) <= 0:
            return reject('nonpositive_value_for_log_error')
        signed = math.log(pred) - math.log(obs)
        output.update(absolute_log_error=abs(signed), signed_log_error=signed,
                      within_twofold=float(abs(signed) <= math.log(2)))
    return {**output, 'scorable': True, 'state': 'scored', 'reason': None}


def hierarchical(rows, field):
    """Equal scenarios within publication, publications within compound, compounds."""
    papers = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            papers[row['compound'], row['publication_cluster_id']].append(row[field])
    drugs = defaultdict(list)
    for (drug, _), values in papers.items():
        drugs[drug].append(math.fsum(values) / len(values))
    return math.fsum(math.fsum(v) / len(v) for v in drugs.values()) / len(drugs) if drugs else None


def exact_position(scores, candidate_ids, pair_results, rank):
    scores = np.asarray(scores, dtype=float)
    require(scores.shape == (len(candidate_ids),) and np.isfinite(scores).all(), 'Incomplete frozen ranking')
    cutoff = np.sort(scores)[-rank]
    tied = np.flatnonzero(scores == cutoff)
    members = [{'candidate_id': candidate_ids[i], 'weight': 1 / len(tied),
                'scorable': pair_results.get(candidate_ids[i], {}).get('scorable', False),
                'reason': pair_results.get(candidate_ids[i], {}).get('reason', 'no_prediction')}
               for i in tied]
    known = [pair_results[m['candidate_id']] for m in members if m['scorable']]
    output = {'rank': rank, 'score': float(cutoff), 'tie_size': len(tied), 'members': members,
              'coverage': len(known) / len(tied), 'fully_evaluable': len(known) == len(tied)}
    for field in ('absolute_log_error', 'signed_log_error', 'absolute_error_h', 'within_twofold'):
        values = [r[field] for r in known if r.get(field) is not None]
        output[field] = math.fsum(values) / len(values) if values else None
    return output


def summarize_positions(rows, methods, definitions, total_scenarios=41):
    result = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method_id'], row['endpoint_key'], row['analysis_stratum'], row['prediction_kind'], row['rank']].append(row)
    method_order = {m['method_id']: i for i, m in enumerate(methods)}
    for key, chosen in sorted(grouped.items(), key=lambda x: (method_order[x[0][0]], *x[0][1:])):
        method, metric, stratum, kind, rank = key
        eligible = [r for r in chosen if r['reference_eligible']]
        covered = [r for r in eligible if r['coverage'] > 0]
        coverage_sum = math.fsum(r['coverage'] for r in eligible)
        error = hierarchical(covered, 'absolute_log_error')
        result.append({**definitions[metric], 'method_id': method, 'analysis_stratum': stratum,
            'prediction_kind': kind, 'rank': rank,
            'eligible_reference_scenarios': len(eligible), 'requested_reference_scenarios': len(chosen),
            'total_scenarios': total_scenarios, 'covered_scenarios': len(covered),
            'fully_covered_scenarios': sum(r['fully_evaluable'] for r in eligible),
            'coverage_eligible_reference': coverage_sum / len(eligible) if eligible else 0.,
            'coverage_total41': coverage_sum / total_scenarios,
            'mean_absolute_log_error': error, 'gmfe': math.exp(error) if error is not None else None,
            'metric_kind': 'absolute_hours' if definitions[metric]['endpoint'] == 'Tmax' else 'geometric_fold_error',
            'endpoint_label': definitions[metric]['quantity'], 'source_stratum': stratum + ':' + kind,
            'mae_h': hierarchical(covered, 'absolute_error_h'),
            'within_twofold': hierarchical(covered, 'within_twofold'),
            'compounds': len({r['compound'] for r in covered}),
            'publications': len({r['publication_cluster_id'] for r in covered}),
            'scored_scenario_ids': sorted(r['scenario_id'] for r in covered)})
    return result


def load_rankings():
    from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
    from summarize_pbpk_match_light import load_inputs
    require(sha(PRIMARY) == PRIMARY_SHA, 'Original execution results changed')
    legacy = json.loads(PRIMARY.read_text())
    ranking_path = BASE / 'rankings/ranking_lock.json'
    require(sha(ranking_path) == legacy['locks']['ranking_lock_sha256'], 'Ranking lock changed')
    lock = json.loads(ranking_path.read_text())
    require(lock['state'] == 'complete' and lock['scientific_outcomes_read'] is False, 'Baseline ranks were not frozen before outcomes')
    sources = [ref(PRIMARY), ref(ranking_path)]
    original_path = ROOT / 'results/data/model_recommendation_execution.json'
    original = json.loads(original_path.read_text())
    methods = [{'method_id': r['method_id'], 'method': r['method']} for r in original['main_cmax_table']]
    scores = {}
    for method in methods:
        registration = lock['methods'][method['method_id']]
        path = ROOT / registration['path']
        require(sha(path) == registration['sha256'], 'Frozen baseline score changed')
        values = np.load(path, allow_pickle=False)
        require(values.shape == (1, 41, 280) and np.isfinite(values).all(), 'Incomplete baseline')
        scores[method['method_id']] = values[0]
        sources.append(ref(path))
    data = load_dataset()
    protocol, _, own, own_refs, _, _ = load_inputs(data)
    require(protocol['candidate_ids'] == lock['candidate_ids'], 'Own/baseline candidate order differs')
    query_indices = [protocol['query_ids'].index(sid) for sid in lock['query_ids']]
    for name, label, key in [('pbpk_match_light_document', 'PBPK-Match (document)', 'base_scores'),
                             ('pbpk_match_light', 'PBPK-Match', 'scores')]:
        methods.append({'method_id': name, 'method': label})
        scores[name] = own[key][0, query_indices]
    sources.extend(own_refs)
    scenarios = {}
    for record in data['execution_scenarios']:
        # Data export may use a dictionary keyed by scenario rather than a list.
        if isinstance(record, str):
            record = data['execution_scenarios'][record]
        scenarios[record['scenario_id']] = record
        scenarios[record['scenario_id']]['route'] = record.get('normalized_design', {}).get('route')
    return methods, scores, lock['query_ids'], lock['candidate_ids'], legacy, scenarios, sources


def legacy_layer(legacy):
    scalars, curves = [], []
    for candidate in legacy['candidate_results']:
        identity = {k: candidate[k] for k in ('scenario_id', 'candidate_id', 'compound', 'publication_cluster_id')}
        for row in candidate['scalars']:
            if row.get('state') == 'valid':
                scalars.append({**identity, **row})
        for row in candidate['curves']:
            if row.get('state') == 'computed' and row.get('curve', {}).get('state') == 'computed':
                curves.append({**identity, **row})
    require(len(scalars) == 171 and len(curves) == 6, 'Original scalar/curve scope changed')
    return {'scalar_comparisons': scalars, 'curve_comparisons': curves,
            'scalar_count': len(scalars), 'curve_count': len(curves),
            'source': ref(PRIMARY), 'analysis_role': 'Unchanged original reference-individual diagnostics; never pooled with protocol-matched extensions.'}


def legacy_publication_rows(preserved, sources):
    """Copy completed scalar/curve summaries without changing their denominators."""
    scalar_counts = Counter(r['endpoint_key'] for r in preserved['scalar_comparisons'])
    original_path = ROOT / 'results/data/model_recommendation_execution.json'
    original = json.loads(original_path.read_text())
    own_path = ROOT / 'experiments/model_recommendation/pbpk_match_light/report/summary.json'
    own = json.loads(own_path.read_text())
    curve_path = ROOT / 'results/data/pbpk_match_light_curves.json'
    curve = json.loads(curve_path.read_text())
    sources += [ref(original_path), ref(own_path), ref(curve_path)]
    rows = []

    def convert(method, name, rank, scenarios, coverage, covered, gmfe, within):
        physical = 'Cmax' if name.startswith('Cmax') else 'finite_AUC'
        return {'method_id': method, 'endpoint_key': 'legacy:' + name,
            'endpoint': name, 'family': physical, 'quantity': name, 'rank': rank,
            'gmfe': gmfe, 'mae_h': None, 'within_twofold': within,
            'coverage': coverage, 'denominator': scenarios, 'eligible_reference_scenarios': scenarios,
            'requested_reference_scenarios': scenarios, 'covered_scenarios': covered,
            'coverage_eligible_reference': coverage, 'coverage_total41': coverage * scenarios / 41,
            'source_comparisons': scalar_counts[name], 'source_stratum': 'legacy:reference_individual',
            'prediction_kind': 'legacy_native', 'analysis_stratum': 'reference_diagnostic',
            'metric_kind': 'geometric_fold_error', 'endpoint_label': name,
            'route': 'original_frozen_scope', 'steady_state': 'ss' in name or 'tau' in name,
            'body_weight_normalized': 'normalized' in name, 'window_h': None,
            'source_window_and_regimen_definition': name, 'total_scenarios': 41}

    for row in original['rank_positions']['summaries']:
        if row['stratum'] == 'reference_diagnostic' and row['endpoint'] in scalar_counts:
            rows.append(convert(row['method_id'], row['endpoint'], row['position'], row['requested_scenarios'],
                row['evaluable_probability'], row['scenarios_with_nonzero_coverage'], row['conditional_gmfe'], row['conditional_within2fold']))
    for method in ('pbpk_match_light_document', 'pbpk_match_light'):
        for row in own['reports'][method]['execution']['position_summary']:
            if row['endpoint'] in scalar_counts:
                rows.append(convert(method, row['endpoint'], row['rank'], row['scenarios'], row['coverage'],
                    row['covered_scenarios'], row['gmfe'], row['within_twofold']))
    require(len(rows) == 14 * len(scalar_counts) * 3, 'Original scalar summaries are incomplete')
    for row in curve['position_summary']:
        rows.append({'method_id': row['method_id'], 'endpoint_key': 'legacy:curve_gmfe',
            'endpoint': 'curve', 'family': 'curve', 'quantity': 'concentration_time_curve',
            'rank': row['position'], 'gmfe': row['curve_gmfe'], 'mae_h': None,
            'within_twofold': row['within_twofold_fraction'], 'coverage': row['curve_coverage'],
            'denominator': row['scenario_count'], 'eligible_reference_scenarios': row['scenario_count'],
            'requested_reference_scenarios': row['scenario_count'], 'covered_scenarios': row['scenarios_with_any_curve'],
            'coverage_eligible_reference': row['curve_coverage'], 'coverage_total41': row['curve_coverage'] * row['scenario_count'] / 41,
            'source_comparisons': 6, 'source_stratum': 'legacy:reference_individual',
            'prediction_kind': 'legacy_native', 'analysis_stratum': 'reference_diagnostic',
            'metric_kind': 'geometric_fold_error', 'endpoint_label': 'Concentration-time curve',
            'route': 'oral', 'steady_state': False, 'body_weight_normalized': False, 'window_h': None,
            'total_scenarios': 41, 'independent_publications': 1, 'interpretation': 'Two descriptive cases; not general curve validation.'})
    return rows


def prediction_versions(row):
    """Keep P and P* separate even where one is the preferred display value."""
    current = dict(row)
    current.setdefault('endpoint', current.get('endpoint_key'))
    current.setdefault('prediction_kind', 'native_only')
    current['prediction_kind'] = 'native' if current['prediction_kind'] in ('native_only', 'native', 'native_window') else current['prediction_kind']
    yield current
    native = row.get('native_prediction')
    if isinstance(native, dict) and current['prediction_kind'] != 'native':
        nested = {**row, **native, 'endpoint': current['endpoint'], 'prediction_kind': 'native'}
        # Native eligibility must be declared independently of matched eligibility.
        nested['eligibility'] = native.get('eligibility', False)
        nested.setdefault('reason', 'native_eligibility_not_declared')
        yield nested


def verify_source_refs(value, sources):
    """Verify recursively registered source file hashes without reading secrets."""
    seen = {(r['path'], r['sha256']) for r in sources}
    def visit(node):
        if isinstance(node, dict):
            if 'path' in node and 'sha256' in node and isinstance(node['path'], str):
                path = (ROOT / node['path']).resolve()
                require(path.is_relative_to(ROOT), 'Extension source escapes project')
                pair = str(path.relative_to(ROOT)), node['sha256']
                if pair not in seen:
                    require(path.is_file() and sha(path) == node['sha256'], 'Extension source changed: ' + pair[0])
                    sources.append({'path': pair[0], 'sha256': pair[1]}); seen.add(pair)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)


def build(reference_path=REFS, prediction_path=PREDICTIONS):
    methods, ranks, query_ids, candidate_ids, legacy, scenarios, sources = load_rankings()
    references = json.loads(Path(reference_path).read_text())
    predictions = json.loads(Path(prediction_path).read_text())
    sources += [ref(reference_path), ref(prediction_path), ref(__file__)]
    verify_source_refs(references, sources)
    verify_source_refs(predictions, sources)
    by_query = {sid: i for i, sid in enumerate(query_ids)}
    publication = {r['scenario_id']: (r['compound'], r['publication_cluster_id']) for r in legacy['candidate_results']}
    reference_rows = [dict(r) for r in references['references']]
    for row in reference_rows:
        require(row['scenario_id'] in by_query, 'Reference scenario outside frozen 41')
        require((row['compound'], row['publication_cluster_id']) == publication[row['scenario_id']], 'Reference identity/study differs')
        row.setdefault('endpoint', row.get('endpoint_key'))
        definition = descriptor(row, scenarios[row['scenario_id']])
        row['_metric_key'] = definition['endpoint_key']
        row['_definition'] = definition
    groups = defaultdict(list)
    for row in reference_rows:
        groups[row['scenario_id'], row['_metric_key']].append(row)
    unique = []
    for key, values in groups.items():
        eligible = [r for r in values if r.get('eligibility') is True]
        if len(eligible) > 1:
            signatures = {(r.get('value'), r.get('unit'), r.get('statistic')) for r in eligible}
            require(len(signatures) == 1, 'Multiple incompatible observations for scenario and endpoint: ' + str(key))
        unique.append(eligible[0] if eligible else values[0])
    definitions = {r['_metric_key']: r['_definition'] for r in unique}
    versioned = [v for p in predictions['predictions'] for v in prediction_versions(p)]
    indexed = defaultdict(list)
    tracks = set()
    for row in versioned:
        execution_scenario = row['scenario_id']
        row['scenario_id'] = row.get('parent_scenario_id', execution_scenario)
        row['execution_scenario_id'] = execution_scenario
        require(row['scenario_id'] in by_query and row['candidate_id'] in candidate_ids, 'Prediction identity outside frozen catalogue')
        kind, stratum = row['prediction_kind'], row.get('analysis_stratum', 'main')
        if stratum != 'main' and row.get('variant'):
            stratum += ':' + row['variant']
            row['analysis_stratum'] = stratum
        require(kind in ('native', 'protocol_matched'), 'Unknown prediction kind')
        trajectory_index = row.get('trajectory_index')
        if trajectory_index is not None:
            trajectory = predictions['trajectories'][trajectory_index]
            require((trajectory['scenario_id'], trajectory['candidate_id']) == (execution_scenario, row['candidate_id']),
                    'Prediction trajectory identity differs')
            row.setdefault('molecular_weight_g_per_mol', trajectory.get('molecular_weight_g_per_mol'))
            row.setdefault('route', trajectory.get('route'))
        row.setdefault('route', scenarios[row['scenario_id']].get('route'))
        tracks.add((stratum, kind))
        indexed[row['scenario_id'], endpoint(row), stratum, kind].append(row)
    tracks.add(('main', 'protocol_matched'))
    candidate_errors, position_rows = [], []
    for observed in unique:
        sid, metric = observed['scenario_id'], observed['_metric_key']
        for stratum, kind in sorted(tracks):
            available = indexed.get((sid, endpoint(observed), stratum, kind), [])
            # A supplementary protocol is scoped to its registered scenarios.
            # Do not materialize thousands of invented missing runs for other
            # drugs. Main tracks still retain every reference scenario.
            if stratum != 'main' and not available:
                continue
            pairs = {}
            for predicted in available:
                result = match_pair(observed, predicted)
                cid = predicted['candidate_id']
                if cid in pairs:
                    # Several window-specific predictions are legal; use only the
                    # uniquely compatible definition, never the smallest error.
                    if pairs[cid]['scorable'] and result['scorable']:
                        require(pairs[cid]['predicted'] == result['predicted'], 'Ambiguous compatible native predictions')
                    if not result['scorable']:
                        continue
                pairs[cid] = {**result, 'candidate_id': cid,
                    'execution_scenario_id': predicted.get('execution_scenario_id', sid),
                    'source_endpoint_key': predicted.get('endpoint_key'),
                    'prediction_definition': predicted.get('definition'),
                    'prediction_window_h': predicted.get('window_h'),
                    'comparison_qualification': predicted.get('comparison_qualification'),
                    'prediction_source_refs': predicted.get('source_refs', [])}
            for cid, pair in pairs.items():
                candidate_errors.append({**definitions[metric], 'scenario_id': sid, 'candidate_id': cid,
                    'compound': observed['compound'], 'publication_cluster_id': observed['publication_cluster_id'],
                    'analysis_stratum': stratum, 'prediction_kind': kind,
                    'reference_source': observed.get('source_ref'), 'reference_endpoint_key': observed.get('endpoint_key'),
                    'reference_statistic': observed.get('statistic'), 'reference_window_h': observed.get('window_h'),
                    'reference_quality_flags': observed.get('source_quality_flags', []), **pair})
            for method in methods:
                for rank in (1, 2, 3):
                    value = exact_position(ranks[method['method_id']][by_query[sid]], candidate_ids, pairs, rank)
                    if observed.get('eligibility') is not True:
                        require(value['coverage'] == 0, 'Ineligible reference received coverage')
                    position_rows.append({'method_id': method['method_id'], 'scenario_id': sid,
                        'compound': observed['compound'], 'publication_cluster_id': observed['publication_cluster_id'],
                        'endpoint_key': metric, 'family': family(endpoint(observed)),
                        'analysis_stratum': stratum, 'prediction_kind': kind,
                        'reference_eligible': observed.get('eligibility') is True, **value})
    preserved = legacy_layer(legacy)
    inventory = []
    for physical in FAMILIES:
        family_references = [r for r in unique if family(endpoint(r)) == physical]
        refset = {r['scenario_id'] for r in family_references if number(r.get('value'))}
        eligible_refset = {r['scenario_id'] for r in family_references if r.get('eligibility') is True}
        considered = [r for r in candidate_errors if r['family'] == physical]
        matched = [r for r in considered if r['scorable']]
        if physical == 'curve':
            refset = {r['scenario_id'] for r in preserved['curve_comparisons']}
            eligible_refset = refset
        counts = {}
        for stratum, kind in sorted(tracks):
            pool = [r for r in matched if r['analysis_stratum'] == stratum and r['prediction_kind'] == kind]
            counts[stratum + ':' + kind] = {'comparisons': len(pool), 'scenarios': len({r['scenario_id'] for r in pool})}
        chosen_kind = 'protocol_matched' if any(r['analysis_stratum'] == 'main' and r['prediction_kind'] == 'protocol_matched' for r in matched) else 'native'
        selected = [r for r in matched if r['analysis_stratum'] == 'main' and r['prediction_kind'] == chosen_kind]
        missing = [r for r in references.get('missing', []) if family(endpoint(r)) == physical]
        reasons = Counter(r['reason'] for r in considered if not r['scorable'])
        reasons.update('no_reference:' + str(r.get('reason', 'unavailable')) for r in missing)
        legacy_count = (len(preserved['curve_comparisons']) if physical == 'curve' else
                        sum((r['endpoint_key'].startswith('Cmax') and physical == 'Cmax') or
                            (r['endpoint_key'].startswith('AUC') and physical == 'finite_AUC')
                            for r in preserved['scalar_comparisons']))
        legacy_family = ([r for r in preserved['curve_comparisons']] if physical == 'curve' else
                         [r for r in preserved['scalar_comparisons']
                          if (physical == 'Cmax' and r['endpoint_key'].startswith('Cmax'))
                          or (physical == 'finite_AUC' and r['endpoint_key'].startswith('AUC'))])
        all_main = [r for r in matched if r['analysis_stratum'] == 'main']
        comparison_pairs = {(r['scenario_id'], r['candidate_id']) for r in [*all_main, *legacy_family]}
        comparable_scenarios = {sid for sid, _ in comparison_pairs}
        inventory.append({'family': physical, 'reference_scenarios': len(refset), 'reference_scenario_ids': sorted(refset),
            'eligible_reference_scenarios': len(eligible_refset),
            'reference_ineligible_reasons': dict(Counter(str(r.get('reason', 'unspecified')) for r in family_references
                                                       if r.get('eligibility') is not True)),
            'total_scenarios': len(query_ids), 'matched_comparisons': len(comparison_pairs),
            'matched_comparisons_definition': 'Distinct candidate-scenario pairs with at least one qualified main or legacy comparison; per-definition counts are in comparison_counts_by_track.',
            'matched_scenarios': len(comparable_scenarios), 'matched_scenario_ids': sorted(comparable_scenarios),
            'preferred_track_comparisons': len(selected) if physical != 'curve' else 6,
            'primary_prediction_kind': chosen_kind if physical != 'curve' else 'legacy_native_curve',
            'comparison_counts_by_track': counts, 'preserved_legacy_comparisons': legacy_count,
            'unavailable_reasons': dict(reasons), 'metric_keys': sorted(k for k, d in definitions.items() if d['family'] == physical)})
    summary = summarize_positions(position_rows, methods, definitions, len(query_ids))
    comparison_counts = Counter((r['endpoint_key'], r['analysis_stratum'], r['prediction_kind'])
                                for r in candidate_errors if r['scorable'])
    for row in summary:
        row.update(coverage=row['coverage_eligible_reference'], denominator=row['eligible_reference_scenarios'],
                   source_comparisons=comparison_counts[row['endpoint_key'], row['analysis_stratum'], row['prediction_kind']])
    # Predefined reporting scope avoids duplicating the original Cmax/fixed-AUC
    # comparisons. All native and supplementary tracks remain in full summaries.
    original_fixed_windows = {(0., 24., False), (0., 72., False), (0., 12., True), (0., 24., True)}
    primary_additions = []
    for row in summary:
        if row['analysis_stratum'] != 'main':
            continue
        if row['prediction_kind'] == 'native':
            if (row['route'] == 'intravenous' and row['endpoint'] in ('half_life', 'AUC_infinity', 'CL', 'Vz', 'MRT', 'Vss')
                    and row['source_comparisons'] > 0):
                primary_additions.append(row)
            continue
        if row['prediction_kind'] != 'protocol_matched' or row['endpoint'] == 'Cmax':
            continue
        if row['endpoint'] == 'AUC_window' and row.get('window_h') is not None:
            if (*row['window_h'], row['steady_state']) in original_fixed_windows:
                continue
        primary_additions.append(row)
    publication_rows = legacy_publication_rows(preserved, sources) + primary_additions
    panels = {}
    for row in publication_rows:
        key = row['endpoint_key'], row['source_stratum']
        fields = ('endpoint_key', 'endpoint', 'family', 'quantity', 'route', 'steady_state', 'body_weight_normalized',
                  'dose_normalization', 'window_h', 'source_stratum', 'metric_kind', 'endpoint_label',
                  'source_comparisons', 'eligible_reference_scenarios', 'requested_reference_scenarios', 'denominator')
        panel = {k: row.get(k) for k in fields}
        if key in panels:
            require(panel == panels[key], 'Method-dependent endpoint panel or reference denominator')
        panels[key] = panel
    for item in sources:
        require(sha(ROOT / item['path']) == item['sha256'], 'Source changed during endpoint scoring: ' + item['path'])
    return {'schema': 'pbpk-recommendation-extended-endpoint-metrics-v1', 'state': 'complete',
        'methods': methods, 'query_ids': query_ids, 'candidate_ids': candidate_ids,
        'family_inventory': inventory, 'endpoint_definitions': list(definitions.values()),
        'candidate_errors': candidate_errors, 'position_rows': position_rows, 'position_summary': summary,
        'publication_rows': publication_rows, 'endpoint_panels': list(panels.values()),
        'legacy_preserved': preserved,
        'missing_references': references.get('missing', []),
        'policy': {'primary_prediction': 'Protocol-matched P* when a qualified comparison is available; native P remains separately reported.',
            'ranking': 'Frozen original twelve methods and grouped out-of-fold own document/final scores; all 280 candidates retained.',
            'rank_positions': 'Exact unfiltered positions 1/2/3; uniform exact-score ties; no lower-rank backfill.',
            'errors': 'Tmax uses absolute hours; other positive dimensionally matched scalar quantities use absolute log ratios and GMFE.',
            'weighting': 'Equal scenarios per publication, equal publications per compound, equal compounds.',
            'coverage': 'Both eligible-reference scenario denominator and all 41 requested scenarios; unavailable errors are null, never zero.',
            'scope': 'Internal reference-individual diagnostics; source and prediction eligibility are required, not clinical direct-reuse validation.',
            'auc_last': 'Protocol-matched AUClast is a clinical-sampling NCA approximation at predicted last quantifiable time; individual observed tlast is unknown, and this is not an exact matched fixed-window AUC.',
            'publication_scope': 'Preserved original scalar/curve panels plus new main protocol-matched endpoint definitions; duplicated original Cmax/fixed-AUC panels omitted from the main display. Qualified intravenous native terminal endpoints have separate native panels. Every omitted display track remains in position_summary.',
            'legacy': 'The 171 original scalar comparisons and six curves remain unchanged and are not pooled with the new protocol-matched comparisons.'},
        'validation': {'methods': len(methods), 'scenarios': len(query_ids), 'candidates': len(candidate_ids),
            'legacy_scalar_count': 171, 'legacy_curve_count': 6, 'new_ranking_inference': False,
            'parameter_refitting': False, 'all_input_hashes_verified': True}, 'sources': sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=REFS)
    parser.add_argument('--predictions', type=Path, default=PREDICTIONS)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build(args.references, args.predictions)
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if args.check:
        require(DEST.read_bytes() == encoded, 'Extended metrics differ from a complete verified recomputation')
    else:
        DEST.write_bytes(encoded)
    print(json.dumps({'output': str(DEST.relative_to(ROOT)), 'checked': args.check,
                      'candidate_errors': len(result['candidate_errors']), 'rank_rows': len(result['position_rows']),
                      'summary_rows': len(result['position_summary']),
                      'families': [{k: row[k] for k in ('family', 'reference_scenarios', 'eligible_reference_scenarios',
                                                     'matched_scenarios', 'matched_comparisons')} for row in result['family_inventory']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
