#!/usr/bin/env python3
"""Export the two already-scored clinical curves, without fitting or simulation.

Only the two observation files explicitly recorded as opened in the frozen main
report are read. Native curves, identities, and rankings retain their original
locks. Labels identify candidates, never ranks or accuracy. --check verifies the
export byte for byte after replaying all checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('experiments/model_recommendation/execution_validation_20261001')
FUP = OUT / 'followup_20261003'
DEST = Path('results/data/model_recommendation_curve_cases.json')
PINS = {
    str(FUP / 'completion.json'): 'c75d5bc15d1a84ec9cbfc174ffab02c6b9fff3ba1a1f988d680a086ed696c914',
    str(FUP / 'audit.json'): '7b3e7ae134173a0f9521eab23f3aa23ec910b2630d60c598d40ee423bd89640c',
    str(FUP / 'primary/results.json'): '8798a53c9a452817ba820cdadf948c9edcf5c5beff939c7813f2cd0cd46292e7',
    str(FUP / 'final_accounting.json'): 'db710b160a5eec3fd54564c1686b597797f24fa21328beb9ef540266b281e1b7',
}
METHODS = {
    'bm25_text': 'BM25',
    'bm25_top20_rank_r1_3b': 'Rank-R1',
    'bm25_top20_rearank_7b_nf4': 'REA-Rank',
    'bm25_top20_reasonrank_7b_nf4': 'ReasonRank',
    'f2llm_v2_06b': 'F2LLM-v2',
    'g_reasoner_34m': 'G-Reasoner',
    'gfm_rag_8m': 'GFM-RAG',
    'gfm_rag_8m_idf_continuous': 'GFM-RAG + IDF',
    'jina_reranker_v35': 'Jina-v3.5',
    'qwen3_embedding_06b': 'Qwen3-Embedding-0.6B',
    'qwen3_embedding_4b': 'Qwen3-Embedding-4B',
    'qwen3_reranker_06b': 'Qwen3-Reranker-0.6B',
}
CASE_IDS = [
    'clinical:atorvastatin_pmc12378308_monotherapy_day_minus4',
    'clinical:rosuvastatin_pmc12378308_monotherapy_day_minus4',
]
OBSERVATIONS = {
    CASE_IDS[0]: ('atorvastatin_pmc12378308_monotherapy_day_minus4_mean_curve.json',
                  'a2fd74ac5e3190894680680c61726280eadc5f2930b3e331bac1e6f96b73acf6'),
    CASE_IDS[1]: ('rosuvastatin_pmc12378308_monotherapy_day_minus4_mean_curve.json',
                  '95b8f6dbd4354029c304f42848d86114c1c478d5631267016091198864f667d5'),
}
OBS_BASE = Path('experiments/blueprint_design/external_validation/concentration_curve_audit_20260920/aggregate_statin_curves/sealed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.refs = {}

    def check(self, ref):
        require(set(ref) == {'path', 'sha256'}, 'Noncanonical input reference')
        relative = Path(ref['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Input outside project')
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), 'Resolved input outside project')
        require(sha(path) == ref['sha256'], f'Input SHA changed: {relative}')
        require(ref['path'] not in self.refs or self.refs[ref['path']] == ref['sha256'],
                'Conflicting reference')
        self.refs[ref['path']] = ref['sha256']
        return path

    def read(self, ref):
        return json.loads(self.check(ref).read_text())

    def closing_check(self):
        for path, digest in list(self.refs.items()):
            self.check({'path': path, 'sha256': digest})


def one_ref(refs, path):
    matches = [dict(path=x['path'], sha256=x['sha256']) for x in refs if x['path'] == str(path)]
    require(len(matches) == 1, f'Expected one locked reference: {path}')
    return matches[0]


def close(actual, expected, description):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), description)


def export():
    inputs = Inputs()
    docs = {path: inputs.read({'path': path, 'sha256': digest}) for path, digest in PINS.items()}
    completion = docs[str(FUP / 'completion.json')]
    require(completion['completed'] is True and completion['original_artifacts_rechecked_unchanged'] is True,
            'Follow-up was not completed')
    require(docs[str(FUP / 'audit.json')]['accepted'] is True, 'Main numerical audit not accepted')
    report = docs[str(FUP / 'primary/results.json')]
    primary = [x for x in completion['outputs'] if x['analysis'] == 'primary']
    require(len(primary) == 1 and one_ref(primary[0]['files'], FUP / 'primary/results.json')['sha256']
            == PINS[str(FUP / 'primary/results.json')], 'Wrong main result')
    locks = report['locks']
    protocol = inputs.read({'path': str(OUT / 'protocol.json'), 'sha256': locks['protocol_sha256']})
    native_lock = inputs.read({'path': str(OUT / 'native/predictions_lock.json'),
                              'sha256': locks['predictions_lock_sha256']})
    ranking = inputs.read({'path': str(OUT / 'rankings/ranking_lock.json'),
                          'sha256': locks['ranking_lock_sha256']})
    require(native_lock['protocol']['sha256'] == locks['protocol_sha256'], 'Native protocol mismatch')
    require(ranking['state'] == 'complete' and ranking['scientific_outcomes_read'] is False,
            'Ranking is not frozen before outcomes')
    require(set(ranking['methods']) == set(METHODS), 'Method set changed')
    scenario_path = inputs.check(one_ref(protocol['inputs'], OUT / 'inputs/scenarios.jsonl'))
    candidate_path = inputs.check(one_ref(protocol['inputs'], OUT / 'rankings/inputs/candidates.jsonl'))
    scenario_rows = [json.loads(line) for line in scenario_path.read_text().splitlines()]
    candidate_rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    scenarios = {r['scenario_id']: r for r in scenario_rows}
    candidates = {r['candidate_id']: r for r in candidate_rows}
    require(len(scenarios) == len(scenario_rows) == 41 and len(candidates) == len(candidate_rows) == 280,
            'Scenario/candidate duplication or scope change')
    require(set(ranking['query_ids']) == set(scenarios) and set(ranking['candidate_ids']) == set(candidates),
            'Ranking scope differs')
    results = {(r['scenario_id'], r['candidate_id']): r for r in report['candidate_results']}
    require(len(results) == len(report['candidate_results']) == 41 * 280, 'Result scope differs')
    available = {key: row for key, row in results.items()
                 if any(c['state'] == 'computed' for c in row.get('curves', []))}
    require({key[0] for key in available} == set(CASE_IDS) and len(available) == 6,
            'Available curves changed; revisit complete-case inclusion')
    ledger = docs[str(FUP / 'final_accounting.json')]['all_original_pairs']
    require(len({(r['scenario_id'], r['candidate_id']) for r in ledger}) == len(ledger) == 177,
            'Ledger contains duplicated or missing pairs')
    ledger = {(r['scenario_id'], r['candidate_id']): r for r in ledger}
    arrays = {}
    for method, registration in ranking['methods'].items():
        path = inputs.check({'path': registration['path'], 'sha256': registration['sha256']})
        values = np.load(path, allow_pickle=False)
        require(values.shape == (1, 41, 280) and np.isfinite(values).all(), 'Invalid ranking scores')
        arrays[method] = values[0]

    exported_cases = []
    numeric_checks = []
    for scenario_id, prefix in zip(CASE_IDS, ['A', 'R']):
        scenario = scenarios[scenario_id]
        filename, digest = OBSERVATIONS[scenario_id]
        obs_ref = {'path': str(OBS_BASE / filename), 'sha256': digest}
        # Authorization is the actual completed run's recorded observation access.
        require(locks['sealed_sources_read_after_all_locks'].get(obs_ref['path']) == digest,
                'Observation was not opened by the completed primary analysis')
        observation = inputs.read(obs_ref)
        require(observation['case_id'] == scenario['case_id'] + '_mean_curve' and observation['compound'] == scenario['compound'],
                'Observation identity mismatch')
        require(observation['time_unit'] == 'h' and observation['concentration_unit'] == 'ng/mL',
                'Observation units changed')
        require(observation['primary_aggregation'] == 'arithmetic_mean_of_source_BLQ_zeroed_individual_concentrations',
                'Aggregation differs from source')
        design = scenario['curve_views'][0]['normalized_design']
        require(design['analyte'] == 'parent' and design['matrix'] == 'plasma'
                and design['evaluation_target']['binding'] == 'total'
                and design['route'] == 'oral' and design['regimen']['dose_times_h'] == [0.0],
                'Clinical matrix, route or regimen differs')
        records = observation['records']
        obs_times = [x['time_h'] for x in records]
        require(obs_times == sorted(set(obs_times)) and obs_times == design['sampling']['times_h'],
                'Observation sampling differs')
        qualified = [x for x in records if isinstance(x.get('arithmetic_mean_ng_ml'), (int, float))
                     and math.isfinite(x['arithmetic_mean_ng_ml']) and x['arithmetic_mean_ng_ml'] > 0]
        qt = [x['time_h'] for x in qualified]
        qy = [x['arithmetic_mean_ng_ml'] for x in qualified]
        curves = []
        exclusions = None
        for ordinal, key in enumerate(sorted(k for k in available if k[0] == scenario_id), 1):
            candidate_id = key[1]
            row = available[key]
            require(row['native_status'] == 'prediction_locked' and row['identity_match'] is True,
                    'Nonmatching/unlocked curve')
            curvescores = [c for c in row['curves'] if c['state'] == 'computed']
            require(len(curvescores) == 1, 'Ambiguous curve aggregation')
            scored = curvescores[0]
            require(scored['curve_kind'] == 'arithmetic_mean'
                    and scored['qualified_observed_time_h'] == qt
                    and scored['qualified_point_count'] == len(qt)
                    and scored['source_point_count'] == len(records)
                    and scored['comparison_stratum'] == 'reference_diagnostic', 'Qualification changed')
            excluded = [{'index': i, 'time_h': r['time_h'], 'reason': 'unquantified_or_source_missing'}
                        for i, r in enumerate(records) if r not in qualified]
            require(excluded == scored['excluded_points'], 'Excluded source points differ')
            require(exclusions is None or exclusions == excluded, 'Different target qualification between models')
            exclusions = excluded
            ledger_row = ledger[key]
            require(ledger_row['executed_in_main'] is True and not ledger_row['successful_supplements'],
                    'Curve includes a supplementary execution')
            job_id = ledger_row['original_job_id']
            registered = [j for j in native_lock['jobs'] if j['job_id'] == job_id]
            require(len(registered) == 1 and (registered[0]['scenario_id'], registered[0]['candidate_id']) == key,
                    'Native job identity mismatch')
            refs = registered[0]['artifacts']
            directory = OUT / 'native' / job_id
            plock_ref = one_ref(refs, directory / 'prediction_lock.json')
            plock = inputs.read(plock_ref)
            require(plock['scenario_id'] == scenario_id and plock['candidate_id'] == candidate_id
                    and plock['execution_verified'] is True and plock['sealed_observations_accessed'] is False,
                    'Prediction lock invalid')
            read = {}
            for name in ['prediction.json', 'execution_plan.json']:
                ref = one_ref(refs, directory / name)
                require(plock['files'].get(ref['path']) == ref['sha256'], 'Local/global lock disagree')
                read[name] = inputs.read(ref)
            prediction, plan = read['prediction.json'], read['execution_plan.json']
            require(prediction['scenario_id'] == scenario_id and prediction['candidate_id'] == candidate_id,
                    'Prediction identity mismatch')
            require((prediction['analyte'], prediction['binding'], prediction['matrix']) == ('parent', 'total', 'plasma'),
                    'Native output matrix mismatch')
            require(prediction['compatibility']['drug_identity_match'] is True
                    and prediction['compatibility']['route_match'] is True
                    and prediction['compatibility']['strict_target_match'] is False,
                    'Source-reference qualification differs')
            require(prediction['time_origin_h_after_first_dose'] == 0 and plan['contract']['dose_mg'] == design['dose']['value'],
                    'Time origin or dose mismatch')
            cat = candidates[candidate_id]
            source_ref = {'path': plan['source_path'], 'sha256': plan['source_sha256']}
            require(source_ref['sha256'] == plock['source_sha256']
                    and source_ref['sha256'] == cat['provenance']['source_sha256'], 'Original model identity differs')
            inputs.check(source_ref)
            t, y = prediction['time_h'], prediction['concentration_ng_ml']
            native = prediction['native_curve']
            require(native['unit'] == 'umol/l' and native['time_unit'] == 'min', 'Unexpected native units')
            require(len(t) == len(y) == len(native['time_min']) == len(native['concentration'])
                    and t == sorted(set(t)) and np.isfinite(t).all() and np.isfinite(y).all()
                    and min(y) >= 0 and min(qt) >= t[0] and max(qt) <= t[-1], 'Native curve grid invalid')
            for ti, yi, nt, ny in zip(t, y, native['time_min'], native['concentration']):
                close(ti, nt / 60, 'Native time conversion differs')
                close(yi, ny * prediction['molecular_weight_g_per_mol'], 'Native mass conversion differs')
            matched = np.interp(qt, t, y).tolist()
            require(min(matched) > 0, 'Nonpositive prediction at positive observation')
            errors = [math.log(p / o) for p, o in zip(matched, qy)]
            gmfe = math.exp(math.fsum(abs(e) for e in errors) / len(errors))
            rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
            close(gmfe, scored['curve']['gmfe'], 'Curve GMFE differs from frozen report')
            close(rmse, scored['curve']['log_rmse'], 'Curve log-RMSE differs from frozen report')
            label = f'{prefix}{ordinal}'
            numeric_checks.append({'label': label, 'candidate_id': candidate_id, 'qualified_point_count': len(qt),
                                   'recomputed_gmfe': gmfe, 'frozen_gmfe': scored['curve']['gmfe'],
                                   'recomputed_log_rmse': rmse, 'frozen_log_rmse': scored['curve']['log_rmse']})
            curves.append({'label': label, 'candidate_id': candidate_id, 'name': cat['family_name'],
                           'version': cat['provenance'].get('repository_commit'), 'family_id': cat['family_id'],
                           'time_h': t, 'concentration_ng_ml': y, 'gmfe': scored['curve']['gmfe'],
                           'log_rmse': scored['curve']['log_rmse'], 'matched_time_h': qt,
                           'matched_concentration_ng_ml': matched, 'native_output_path': prediction['native_output_path'],
                           'molecular_weight_g_per_mol': prediction['molecular_weight_g_per_mol'],
                           'compatibility': prediction['compatibility'],
                           'source_formulation_classes': plan.get('source_formulation_classes'),
                           'source_simulation': plan.get('source_simulation'),
                           'split_absorption_events': plan.get('split_absorption_contract', {}).get('events'),
                           'provenance': {'catalogue': cat['provenance'], 'original_source': source_ref,
                                          'job_id': job_id, 'prediction': one_ref(refs, directory / 'prediction.json'),
                                          'execution_plan': one_ref(refs, directory / 'execution_plan.json'),
                                          'prediction_lock': plock_ref}})
        labels = {c['candidate_id']: c['label'] for c in curves}
        rank_rows = []
        for method, display in METHODS.items():
            scores = arrays[method][ranking['query_ids'].index(scenario_id)]
            ordered_scores = np.sort(scores)[::-1]
            positions = []
            for position in [1, 2, 3]:
                threshold = ordered_scores[position - 1]
                member_indices = np.flatnonzero(scores == threshold).tolist()
                require(member_indices, 'Empty rank-position block')
                members = []
                for index in sorted(member_indices, key=lambda i: ranking['candidate_ids'][i]):
                    cid = ranking['candidate_ids'][index]
                    cat = candidates[cid]
                    members.append({'candidate_id': cid, 'curve_label': labels.get(cid),
                                    'weight': 1.0 / len(member_indices), 'name': cat['family_name'],
                                    'family_id': cat['family_id'], 'compounds': cat['compounds'],
                                    'identity_match': results[(scenario_id, cid)]['identity_match']})
                positions.append({'position': position, 'score': float(threshold),
                                  'tie_group_size': len(members), 'tie_start_rank': int(np.count_nonzero(scores > threshold)) + 1,
                                  'tie_end_rank': int(np.count_nonzero(scores >= threshold)), 'members': members,
                                  'curve_coverage': math.fsum(m['weight'] for m in members if m['curve_label'] is not None)})
            rank_rows.append({'method_id': method, 'method': display, 'positions': positions})
        exported_cases.append({'compound': scenario['compound'], 'scenario_id': scenario_id,
                               'dose_mg': design['dose']['value'], 'route': design['route'], 'food': design['food'],
                               'formulation': design['formulation'], 'matrix': 'plasma', 'analyte': 'parent', 'binding': 'total',
                               'publication_cluster_id': observation['publication_cluster_id'],
                               'cohort_id': observation['cohort_id'], 'clinical_design': design,
                               'observation': {'time_h': qt, 'concentration_ng_ml': qy,
                                               'sd_ng_ml': [x['sd_ng_ml'] for x in qualified],
                                               'population_n': observation['population_n'],
                                               'source_point_count': len(records), 'qualified_point_count': len(qt),
                                               'excluded_points': exclusions, 'all_source_records': records,
                                               'primary_aggregation': observation['primary_aggregation'],
                                               'censoring': observation['censoring'], 'source': obs_ref,
                                               'source_pdf': observation['source_pdf'], 'source_article': observation['source_article'],
                                               'source_table': observation['source_table']},
                               'curves': curves, 'rankings': rank_rows})
    inputs.check({'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha(Path(__file__))})
    inputs.closing_check()
    return {'schema_version': 'recommendation-native-curve-illustration-v1',
            'cases': exported_cases,
            'selection': {'rule': 'All scenarios and candidates with computed qualified curves in the completed primary report.',
                          'scenario_count': 2, 'candidate_curves': 6, 'publication_count': 1, 'cohort_count': 2,
                          'candidate_labels': 'A1 and R1–R5 are ordered by candidate UUID, not rank or accuracy.',
                          'ranking_rule': 'Unfiltered exact positions 1/2/3; all members of the exact-score tie block have equal weight; no replacement by a lower rank.'},
            'validation': {'accepted': True, 'curve_metric_checks': numeric_checks,
                           'all_rank_position_groups': 72, 'source_input_count': len(inputs.refs),
                           'closing_hash_check': True, 'native_models_reexecuted': False, 'parameters_refitted': False,
                           'scope': 'Figure export validates selected locked artifacts and original report; prior full execution audit remains authoritative.'},
            'limitations': ['Observed points are published arithmetic group means; predictions are original reference individuals, not fitted cohort means.',
                            'Both drug cohorts come from one publication; they are not independent studies.',
                            'Dose, time, route, analyte and matrix are aligned; food, formulation and population equivalence remain unverified.',
                            'Rosuvastatin R4 retains source solution formulation and all five retain source-bound split absorption; no clinical matching is claimed.',
                            'Published zero means retain their BLQ explanation and are excluded from logarithmic metrics; positive means below the individual LLOQ remain included.',
                            'A missing curve at a rank remains unavailable; tied candidates are not chosen by their observed performance.',
                            'Native prediction curves belong to model recommendation; the historical observation file directory name does not designate blueprint predictions.'],
            'provenance': [{'path': p, 'sha256': s} for p, s in sorted(inputs.refs.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and compare the published export without modifying files.')
    args = parser.parse_args()
    payload = export()
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    path = ROOT / DEST
    if args.check:
        require(path.read_bytes() == encoded, 'Export differs; regenerate after investigating')
    else:
        path.write_bytes(encoded)
    print(json.dumps({'checked': args.check, 'output': str(DEST), 'sha256': hashlib.sha256(encoded).hexdigest(),
                      'cases': 2, 'curves': 6, 'rank_groups': 72, 'verified_inputs': payload['validation']['source_input_count']}))


if __name__ == '__main__':
    main()
