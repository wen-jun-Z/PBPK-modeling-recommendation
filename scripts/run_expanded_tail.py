#!/usr/bin/env python3
"""Frozen three-method comparison on twenty drugs; never updates publication assets."""
from __future__ import annotations

import csv
from collections import Counter
from copy import deepcopy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from report_blueprint_validation import Store, positive, clustered_geomean, paired_ratio
from run_tail_blueprint import METHODS, ENGINE_NAMES, VARIANTS

VERSION = 'expanded-tail-blueprint-1'
DEFAULT = ROOT / 'experiments/blueprint_design/expanded_tail_20260930'
SOURCE_SENSITIVITY_EXCLUSIONS = frozenset({'Dexmedetomidine', 'Midazolam', 'Nemonoxacin',
    'Sufentanil', 'Sildenafil', 'Isoniazid', 'Digoxin'})
CODE = ('scripts/run_expanded_tail.py', 'scripts/prepare_expanded_tail.py',
        'scripts/score_expanded_tail.py', 'scripts/run_tail_blueprint.py',
        'scripts/run_robust_blueprint.py', 'scripts/run_blueprint_capability.py',
        'scripts/run_blueprint_evidence_policy.py', 'scripts/run_blueprint_learning.py',
        'scripts/run_blueprint_transport.py', 'scripts/report_blueprint_validation.py')


def select_formula(mappings):
    if (len(mappings) != 2 or {m['construction_partition_method'] for m in mappings}
            != {'PKSimStandard', 'RodgersRowland'} or len({m['variant'] for m in mappings}) != 1):
        raise ValueError('Two locked formulas for one unchanged method are required')
    eligible = [m for m in mappings if m['eligible']]
    def score(m):
        guarded = m['variant'] == 'development_guarded'
        value = m.get('formula_selection_score') if guarded else m['decision']['objective']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Finite development-only selection score required')
        weight = m['clearance_prior_policy']['penalty_weight'] if guarded else 0.
        return value, -weight, m['construction_partition_method']
    return min(eligible, key=score) if eligible else None


def summarize_records(records, cases, scenarios, methods):
    """Separate original-reported and observation-derived scalar references."""
    summary, pairs = [], {}
    groups = {'all_20': cases, 'original_10': [c for c in cases if c['cohort'] == 'original'],
              'extension_10': [c for c in cases if c['cohort'] == 'extension'],
              'source_sensitivity': [c for c in cases if c['drug'] not in SOURCE_SENSITIVITY_EXCLUSIONS]}
    for group, cohort in groups.items():
        ids = {c['case_id'] for c in cohort}; clusters = {c['case_id']: c['cluster_id'] for c in cohort}
        for scenario in scenarios:
            selected = [r for r in records if r['case_id'] in ids and r['scenario'] == scenario]
            scores_by_method = {}
            for method in methods:
                rows = [r for r in selected if r['method'] == method]
                scores = {r['case_id']: r.get('curve_metrics', {}).get('gmfe') for r in rows
                          if positive(r.get('curve_metrics', {}).get('gmfe'))}
                scores_by_method[method] = scores
                endpoints = {}
                for row in rows:
                    for name, e in row.get('endpoints', {}).items():
                        kind = e.get('reference_kind', 'original_reported')
                        val = e.get('error')
                        if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                            continue
                        key = name + '__' + kind
                        dest = endpoints.setdefault(key, {'endpoint': name, 'reference_kind': kind,
                            'error_metric': e['error_metric'], 'values': {}})
                        if dest['error_metric'] != e['error_metric']:
                            raise ValueError('Incompatible endpoint error definitions')
                        dest['values'][row['case_id']] = val
                for item in endpoints.values():
                    vals = item.pop('values'); item['n'] = len(vals)
                    if item['error_metric'] == 'absolute_error_h':
                        item['mean_absolute_error_h'] = float(np.mean(list(vals.values())))
                    else:
                        item['geomean_fold_error'] = math.exp(float(np.mean(np.log(list(vals.values())))))
                        item['within_twofold'] = sum(x <= 2 for x in vals.values())
                summary.append({'cohort': group, 'scenario': scenario, 'method': method,
                    'requested': len(ids), 'scored': len(scores), 'failed': len(ids)-len(scores),
                    'curve_gmfe': clustered_geomean(scores, {k: clusters[k] for k in scores}, seed=20260930),
                    'within_twofold': sum(x <= 2 for x in scores.values()), 'endpoint_errors': endpoints})
            pairs.setdefault(group, {})[scenario] = {m: paired_ratio(scores_by_method[m],
                scores_by_method['current'], clusters, expected_ids=sorted(ids), seed=20260930) for m in methods[1:]}
    return summary, pairs


def prediction_stability(records, cases, store):
    indexed = {(r['case_id'], r['scenario'], r['method']): r for r in records}
    configs = {c['case_id']: c for c in cases}; output = []
    for row in records:
        if row['scenario'] == 'available':
            continue
        full = indexed[row['case_id'], 'available', row['method']]
        item = {k: row[k] for k in ('case_id', 'drug', 'cohort', 'cluster_id', 'method', 'scenario', 'scientific_inputs_changed')}
        item.update(status='failed', endpoint_prediction_drift={}, endpoints_not_compared=[])
        if row['attempt_status'] == full['attempt_status'] == 'predicted':
            a, b = (store.ref(r['prediction_source']) for r in (row, full))
            times = np.asarray(configs[row['case_id']]['nca_policy']['sampling_time_h']) * 60
            valid = (times > 0) & (times >= max(a['time_min'][0], b['time_min'][0])) & (times <= min(a['time_min'][-1], b['time_min'][-1]))
            pa = np.interp(times[valid], a['time_min'], a['concentration'])
            pb = np.interp(times[valid], b['time_min'], b['concentration'])
            if len(pa) and np.isfinite(pa).all() and np.isfinite(pb).all() and (pa > 0).all() and (pb > 0).all():
                d = np.maximum(pa/pb, pb/pa)
                item.update(status='compared', points=len(d), curve_prediction_drift=float(np.exp(np.mean(np.log(d)))),
                            max_point_drift=float(max(d)))
            for name, e in row.get('endpoints', {}).items():
                other = full.get('endpoints', {}).get(name)
                if other is None:
                    item['endpoints_not_compared'].append(name)
                    continue
                p, q = e.get('value'), other.get('value')
                if e.get('unit') != other.get('unit'):
                    raise ValueError('Stability endpoints have incompatible units')
                if name == 'tmax' and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p, q)):
                    if e['unit'] not in ('h', 'min'): raise ValueError('Tmax requires hours/minutes')
                    item['tmax_prediction_shift_h'] = abs(p-q)/(60 if e['unit'] == 'min' else 1)
                elif name != 'tmax' and positive(p) and positive(q):
                    item['endpoint_prediction_drift'][name] = max(p/q, q/p)
        output.append(item)
    return output


def nonregression_checks(records, summary, cases):
    """Separate pooled nonregression from individual regressions; never promote."""
    indexed = {(r['case_id'], r['scenario'], r['method']): r for r in records}
    scores = {(r['cohort'], r['scenario'], r['method']): r for r in summary}
    output = {}
    for method in METHODS[1:]:
        checks, regressions, incomplete = [], [], []
        for scenario in ('available', 'missing_fu', 'missing_cl', 'missing_both'):
            a, b = (scores['all_20', scenario, m] for m in (method, 'current'))
            checks.append(a['scored'] == b['scored'] == len(cases) and
                a['curve_gmfe']['estimate'] <= b['curve_gmfe']['estimate'] + 1e-12)
            for case in cases:
                ra, rb = (indexed[case['case_id'], scenario, m] for m in (method, 'current'))
                va, vb = (r.get('curve_metrics', {}).get('gmfe') for r in (ra, rb))
                if not (positive(va) and positive(vb)):
                    incomplete.append({'case_id': case['case_id'], 'scenario': scenario})
                elif va > vb * (1 + 1e-10):
                    regressions.append({'case_id': case['case_id'], 'drug': case['drug'],
                        'scenario': scenario, 'ratio': va/vb})
        output[method] = {'all_four_aggregate_errors_nonincreasing': all(checks),
            'individual_curve_regressions': regressions, 'incomplete_pairs': incomplete,
            'automatic_promotion': False}
    return output


def create_runner(*, project_root=ROOT, engine_provider=None, controller_class=None):
    root = Path(project_root).resolve()
    default = root / 'experiments/blueprint_design/expanded_tail_20260930'
    spec = importlib.util.spec_from_file_location('_expanded_frozen_runner_' + uuid.uuid4().hex,
                                                 ROOT / 'scripts/run_robust_blueprint.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.__doc__ = __doc__
    original_signature, original_load, original_status = runner.signature, runner.load, runner.status
    runner.ROOT, runner.VERSION, runner.DEFAULT, runner.METHODS = root, VERSION, default, METHODS
    runner.BASE = root / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
    runner.CAPABILITY = root / 'experiments/blueprint_design/blueprint_capability_20260930'
    runner.__file__ = str(root / 'scripts/run_expanded_tail.py')
    runner.api = runner.previous.transport_runner.create_runner(project_root=root)
    for name in ('read', 'ref', 'checked', 'verify'):
        setattr(runner, name, getattr(runner.api, name))
    if controller_class is not None:
        runner.Controller = controller_class
    runner.api.choose_formula = select_formula
    runner.engine = engine_provider or (lambda method: importlib.import_module(ENGINE_NAMES[method]))
    runner.variant = lambda method: VARIANTS[method]
    runner.signature = lambda req, tar, tr, method: original_signature(req, tar, tr, VARIANTS[method])
    def own_run(path):
        raw = Path(path) if Path(path).is_absolute() else root / path
        run = inside(raw, root)
        if raw.is_symlink() or run.parent != default.parent or not run.name.startswith('expanded_tail_'):
            raise ValueError('Use a dedicated immediate expanded_tail_ experiment directory')
        return run
    runner.own_run = own_run

    def prepare(run, workers=14):
        run = own_run(run)
        if (run/'protocol.json').exists() or (run/'state.json').exists():
            raise FileExistsError('Keep frozen batch; use resume')
        if type(workers) is not int or not 1 <= workers <= 14: raise ValueError('Use 1–14 workers')
        manifest_path = run/'inputs/unified_manifest.json'
        manifest = runner.read(manifest_path); runner.verify(manifest['source_pins'])
        cases = deepcopy(manifest['cases'])
        if (len(cases) != 20 or len({c['compound_id'] for c in cases}) != 20 or len({c['case_id'] for c in cases}) != 20
                or Counter(c['cohort'] for c in cases) != {'original': 10, 'extension': 10}):
            raise ValueError('Twenty distinct drugs, ten original and ten extensions, are required')
        pins = deepcopy(manifest['source_pins']); pins[str(manifest_path.relative_to(root))] = sha256(manifest_path)
        publication = runner.paper_pins()
        if publication != manifest['paper_guard']:
            raise ValueError('Publication files changed since input preparation')
        contracts = {m: {'module': ENGINE_NAMES[m], 'version': runner.engine(m).VERSION,
            'config': deepcopy(runner.engine(m).CONFIG), 'code_pins': runner.engine(m)._code_pins(root)} for m in METHODS}
        for contract in contracts.values(): pins.update(contract['code_pins'])
        computations, units, audit = {}, [], []
        changed = {s: [] for s in runner.SCENARIOS}
        for case in cases:
            base, target = (runner.checked(case['requests']['evidence'][k]) for k in ('development', 'target'))
            rf = runner.checked(case['requests']['rf']['development'])
            training = runner.inputs.clean_training(runner.checked(case['sampling_sources']['full']))
            training_ref = {**runner.inputs.store_input(run, training), 'role': 'calibration_data_reconstruction'}
            original_values = {r['parameter']: r['value'] for r in base['evidence']}
            for scenario in runner.SCENARIOS:
                removed = {'fraction_unbound'} if scenario == 'missing_fu' else {'total_clearance'} if scenario == 'missing_cl' else {'fraction_unbound', 'total_clearance'} if scenario == 'missing_both' else set()
                rows = [runner.inputs.visible_record(r) for r in base['evidence'] if r['parameter'] not in removed]
                fallback = [runner.inputs.visible_record(r) for r in rf['evidence'] if r['parameter'] in ('fraction_unbound', 'total_clearance')]
                rows, fallback, view = runner.inputs.bind_view(run, rows, fallback)
                selected, actions = runner.fill_visible(rows, fallback)
                request = runner.inputs.selected_request(run, base, selected); predict = deepcopy(request)
                for field in ('human', 'dosing', 'target'): predict[field] = deepcopy(target[field])
                changed_parameters = [r['parameter'] for r in selected if r['value'] != original_values.get(r['parameter'])]
                if changed_parameters: changed[scenario].append(case['case_id'])
                audit.append({'case_id': case['case_id'], 'scenario': scenario, 'removed_parameters': sorted(removed),
                    'changed_parameters': changed_parameters, 'actions': actions, 'visible_evidence': view})
                for method in METHODS:
                    cid = runner.signature(request, predict, training, method)
                    units.append({'id': '__'.join((case['case_id'], scenario, method)), 'case_id': case['case_id'],
                        'scenario': scenario, 'sampling': 'full', 'method': method, 'computation_id': cid,
                        'scientific_inputs_changed': bool(changed_parameters)})
                    if cid not in computations:
                        computations[cid] = {'case_id': case['case_id'], 'sampling': 'full', 'method': method,
                            'request': runner.inputs.store_input(run, request), 'target_request': runner.inputs.store_input(run, predict),
                            'training_source': training_ref}
        for value in (cases, computations, audit): runner.api.pin_refs(value, pins)
        from score_expanded_tail import CODE_FILES as SCORE_CODE
        for name in set(CODE) | set(SCORE_CODE): pins[name] = sha256(root/name)
        cv_stages = 0
        for c in computations.values():
            if c['method'] == 'development_guarded':
                e = runner.engine(c['method']); policy = e.clearance_policy(runner.checked(c['request']))
                count = len(e.CONFIG['cv']['rf_penalty_weights'] if policy['rule'] == 'qualified_rf' else e.CONFIG['cv']['observed_penalty_weights'])
                cv_stages += 2 * len(e.CONFIG['cv']['folds']) * count
        budget = {'full_fit_stages': 2*len(computations), 'cv_stages': cv_stages,
            'full_optimizer_starts': 32*len(computations), 'cv_optimizer_starts': 8*cv_stages}
        budget['optimizer_starts'] = budget['full_optimizer_starts'] + budget['cv_optimizer_starts']
        protocol = {'version': VERSION, 'created_at': runner.now(), 'cases': cases, 'methods': list(METHODS),
            'sampling': ['full'], 'scenarios': list(runner.SCENARIOS), 'formulas': list(runner.FORMULAS),
            'units': units, 'computations': computations, 'cached_ids': [], 'contracts': contracts,
            'input_audit': audit, 'changed_case_ids': changed, 'source_pins': pins, 'paper_guard': publication,
            'source_protocol': runner.ref(manifest_path), 'workers': workers, 'native_threads': 2,
            'native_timeout_seconds': 180, 'monitor_seconds': 1200, 'directory_limit_bytes': 10*1024**3,
            'optimization_budget': budget,
            'scope': 'Twenty previously curated drugs, unified twenty-excluded RF priors and frozen algorithms. Original, extension and combined cohorts reported separately; not new blind validation.',
            'baseline': 'evidence_flexible with refreshed twenty-excluded RF inputs; historical 1.494 is not the expected new control score',
            'endpoint_policy': 'Original-reported and observation-derived scalar references never pooled; no invented LLOQ or zero-time concentration; missing original labels remain missing.',
            'bootstrap': {'seed': 20260930, 'resamples': 2000, 'cluster': 'clinical study'},
            'source_sensitivity': {'excluded_drugs': sorted(SOURCE_SENSITIVITY_EXCLUSIONS),
                'basis': 'Prespecified population, source or Digoxin structural assumptions. Generic historical-data and representative-human limitations apply to the main and sensitivity cohorts alike; all flags remain reported.'},
            'limitations': ['All drugs are historical development material; extension is not an unseen final test.',
                'Window-derived scalar references are distinct from original-reported clinical parameters.',
                'All twenty RF identities/scaffolds excluded; some direct reported PK construction inputs are from development studies.',
                'Equal-input deletion/refill controls do not increase the effective robustness denominator.',
                'No parameter or method is selected using target-study performance.'],
            'automatic_promotion': False, 'paper_updated': False}
        runner.verify(pins)
        if runner.paper_pins() != publication: raise ValueError('Publication files changed during preparation')
        save(run/'protocol.json', protocol)
        save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'created_at': runner.now(),
            'protocol': runner.ref(run/'protocol.json'), 'fits': {}, 'computations': {}, 'monitor': {}})
        return {'drugs': 20, 'logical_units': len(units), 'unique_predictions': len(computations),
                'optimization_budget': budget, 'paper_guard_files': len(publication)}

    def load(run, locked=False):
        p, s = original_load(run, locked=locked)
        if (len(p['cases']) != 20 or p['methods'] != list(METHODS) or p['cached_ids']
                or runner.paper_pins() != p['paper_guard']):
            raise ValueError('Expanded cohort/method/publication contract changed')
        if locked:
            if not s.get('formulas_locked_before_target_prediction'): raise ValueError('Formula lock must precede target prediction')
            choices = runner.checked(s['formula_selection'])['choices']
            for cid in p['computations']:
                mappings = [runner.checked(s['fits'][runner.key(cid,f)]['mapping']) for f in runner.FORMULAS]
                best = select_formula(mappings)
                expected = None if best is None else {'formula': best['construction_partition_method'],
                    'mapping': s['fits'][runner.key(cid,best['construction_partition_method'])]['mapping']}
                if choices[cid] != expected: raise ValueError('Formula is not selected by the frozen development-only rule')
                if s['computations'][cid]['status'] == 'predicted' and (expected is None or s['computations'][cid]['mapping'] != expected['mapping']):
                    raise ValueError('Prediction mapping differs from locked formula')
        return p, s

    def report(run):
        from score_expanded_tail import score_case
        p, s = load(run, locked=True)
        if s['status'] == 'complete':
            result = runner.checked(s['results']); runner.verify(result['source_pins']); return result
        store = Store(root); store.ref(s['protocol'])
        for path, digest in p['source_pins'].items(): store.checksum(path, digest)
        store.verify_refs([s['fits'], s['computations'], s['formula_selection']])
        cases = {c['case_id']: c for c in p['cases']}; cache, rows = {}, []
        for unit in p['units']:
            case = cases[unit['case_id']]; comp = s['computations'][unit['computation_id']]
            if unit['computation_id'] not in cache:
                try:
                    if comp['status'] != 'predicted': raise ValueError(str(comp.get('failure')))
                    scored = score_case(case, comp, store); scored.pop('curve', None)
                    scored.setdefault('scoring_status', 'scored' if positive(scored.get('curve_metrics', {}).get('gmfe')) else 'partial')
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    scored = {'scoring_status': 'failed', 'failure': str(exc)}
                cache[unit['computation_id']] = scored
            rows.append({**unit, 'drug': case['drug'], 'cohort': case['cohort'], 'cluster_id': case['cluster_id'],
                'study_id': case['study_id'], 'attempt_status': comp['status'], 'mapping_source': comp.get('mapping'),
                'prediction_source': comp.get('prediction'), **deepcopy(cache[unit['computation_id']])})
        summary, pairs = summarize_records(rows, p['cases'], runner.SCENARIOS, METHODS)
        stability = prediction_stability(rows, p['cases'], store)
        result = {'version': VERSION, 'protocol': runner.ref(run/'protocol.json'), 'scope': p['scope'],
            'units': rows, 'summary': summary, 'candidate_to_current_curve_ratio': pairs, 'stability': stability,
            'nonregression_checks': nonregression_checks(rows, summary, p['cases']),
            'changed_case_ids': p['changed_case_ids'], 'unique_scored_predictions': len(cache),
            'source_pins': dict(store.pins), 'paper_guard_verified': True, 'paper_updated': False,
            'limitations': p['limitations']}
        store.unchanged()
        if runner.paper_pins() != p['paper_guard']: raise ValueError('Publication files changed during scoring')
        save(run/'results.json', result)
        table = [{'cohort': r['cohort'], 'scenario': r['scenario'], 'method': r['method'], 'requested': r['requested'],
            'scored': r['scored'], 'failed': r['failed'], 'curve_gmfe': r['curve_gmfe']['estimate'],
            'ci95_low': (r['curve_gmfe']['ci95'] or [None,None])[0], 'ci95_high': (r['curve_gmfe']['ci95'] or [None,None])[1],
            'within_twofold': r['within_twofold']} for r in summary]
        with (run/'results.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
        return result

    def status(run):
        result = original_status(run); p = runner.read(run/'protocol.json'); counts = Counter()
        for cid in p['computations']:
            for formula in runner.FORMULAS:
                path = run/'work'/cid/formula/'progress.json'
                if not path.exists(): continue
                progress = runner.read(path)
                if 'stages' in progress:
                    for name, stage in progress['stages'].items():
                        kind = 'full' if name == 'full' else 'cv'
                        counts[kind+'_starts_completed'] += len(stage.get('starts', []))
                        counts[kind+'_stages_completed'] += int(stage.get('status') == 'complete')
                else:
                    counts['full_starts_completed'] += len(progress.get('starts', []))
                    counts['full_stages_completed'] += int(progress.get('status') == 'locked')
        result['optimization_progress'] = {'completed': dict(counts), 'planned': p['optimization_budget']}
        return result

    runner.prepare, runner.load, runner.report, runner.status = prepare, load, report, status
    return runner


if __name__ == '__main__':
    create_runner().main()
