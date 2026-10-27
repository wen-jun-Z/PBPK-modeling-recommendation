#!/usr/bin/env python3
"""Twenty-drug restrained exchange refinement; reuse only exact locked controls."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from report_blueprint_validation import Store, positive
from run_expanded_tail import summarize_records, prediction_stability, SOURCE_SENSITIVITY_EXCLUSIONS

VERSION = 'adaptive-twenty-blueprint-1'
SOURCE_NAME = 'expanded_tail_20260930'
DEFAULT_NAME = 'adaptive_twenty_20260930'
METHODS = ('current', 'relaxed_tail', 'adaptive_exchange')
VARIANTS = {'current': 'tail_balanced', 'relaxed_tail': 'relaxed_tail', 'adaptive_exchange': 'adaptive_exchange'}
ENGINE_NAMES = {'current': 'pbpk_kb.blueprint_policy.tail_exchange_calibration',
    'relaxed_tail': 'pbpk_kb.blueprint_policy.adaptive_exchange_calibration',
    'adaptive_exchange': 'pbpk_kb.blueprint_policy.adaptive_exchange_calibration'}
CODE = ('scripts/run_adaptive_twenty.py', 'scripts/run_expanded_tail.py', 'scripts/run_robust_blueprint.py',
    'scripts/run_tail_blueprint.py', 'scripts/run_blueprint_capability.py',
    'scripts/run_blueprint_evidence_policy.py', 'scripts/run_blueprint_learning.py',
    'scripts/run_blueprint_transport.py')


def select_formula(mappings):
    if (len(mappings) != 2 or {m['construction_partition_method'] for m in mappings}
            != {'PKSimStandard', 'RodgersRowland'} or len({m['variant'] for m in mappings}) != 1):
        raise ValueError('Both formulas of one frozen method must be present')
    eligible = [m for m in mappings if m['eligible']]
    for m in eligible:
        value = m['decision']['objective']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite development-only objective required')
    return min(eligible, key=lambda m: (m['decision']['objective'], m['construction_partition_method'])) if eligible else None


def nonregression_checks(rows, summary, cases):
    scores = {(r['cohort'], r['scenario'], r['method']): r for r in summary}
    lookup = {(r['case_id'], r['scenario'], r['method']): r for r in rows}
    output = {}
    for method in METHODS[1:]:
        conditions, failures, regressions, endpoint_regressions = [], [], [], []
        for scenario in ('available', 'missing_fu', 'missing_cl', 'missing_both'):
            a, b = (scores['all_20', scenario, m] for m in (method, 'current'))
            conditions.append(a['scored'] == b['scored'] == len(cases) and
                a['curve_gmfe']['estimate'] <= b['curve_gmfe']['estimate'] + 1e-12)
            for case in cases:
                ra, rb = (lookup[case['case_id'], scenario, m] for m in (method, 'current'))
                va, vb = (r.get('curve_metrics', {}).get('gmfe') for r in (ra, rb))
                item = {'case_id': case['case_id'], 'drug': case['drug'], 'scenario': scenario}
                if not (positive(va) and positive(vb)):
                    failures.append(item)
                elif va > vb * (1 + 1e-10):
                    regressions.append({**item, 'ratio': va/vb})
                for endpoint, ea in ra.get('endpoints', {}).items():
                    eb = rb.get('endpoints', {}).get(endpoint, {})
                    aerr, berr = ea.get('error'), eb.get('error')
                    if (isinstance(aerr, (int, float)) and isinstance(berr, (int, float))
                            and math.isfinite(aerr) and math.isfinite(berr)
                            and ea.get('error_metric') == eb.get('error_metric')
                            and ea.get('reference_kind') == eb.get('reference_kind') and aerr > berr + 1e-10):
                        endpoint_regressions.append({**item, 'endpoint': endpoint,
                            'candidate_error': aerr, 'baseline_error': berr, 'error_metric': ea['error_metric']})
        output[method] = {'all_four_aggregate_errors_nonincreasing': all(conditions),
            'individual_curve_regressions': regressions, 'endpoint_error_regressions': endpoint_regressions,
            'incomplete_pairs': failures, 'automatic_promotion': False}
    return output


def create_runner(*, project_root=ROOT, source_name=SOURCE_NAME, default_name=DEFAULT_NAME,
                  engine_provider=None, controller_class=None):
    root = Path(project_root).resolve()
    if (Path(source_name).name != source_name or not source_name.startswith('expanded_tail_')
            or Path(default_name).name != default_name or not default_name.startswith('adaptive_twenty_')):
        raise ValueError('Use dedicated source and refinement directory names')
    source = root / 'experiments/blueprint_design' / source_name
    default = source.parent / default_name
    spec = importlib.util.spec_from_file_location('_adaptive_frozen_runner_' + uuid.uuid4().hex,
                                                 ROOT / 'scripts/run_robust_blueprint.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    original_signature, original_load, original_status = runner.signature, runner.load, runner.status
    runner.ROOT, runner.VERSION, runner.DEFAULT, runner.METHODS = root, VERSION, default, METHODS
    runner.BASE = root / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
    runner.CAPABILITY = root / 'experiments/blueprint_design/blueprint_capability_20260930'
    runner.__file__, runner.__doc__ = str(root / 'scripts/run_adaptive_twenty.py'), __doc__
    runner.api = runner.previous.transport_runner.create_runner(project_root=root)
    runner.api.choose_formula = select_formula
    for name in ('read', 'ref', 'checked', 'verify'):
        setattr(runner, name, getattr(runner.api, name))
    if controller_class is not None: runner.Controller = controller_class
    runner.engine = engine_provider or (lambda method: importlib.import_module(ENGINE_NAMES[method]))
    runner.variant = lambda method: VARIANTS[method]
    runner.signature = lambda req, tar, tr, method: original_signature(req, tar, tr, VARIANTS[method])
    binding = {'version': VERSION, 'methods': list(METHODS), 'variants': VARIANTS,
        'source_name': source_name, 'default_name': default_name,
        'baseline_source_method': 'tail_balanced',
        'baseline_interpretation': 'current is the locked previous tail_balanced (20-drug GMFE 1.641), not evidence_flexible or the paper model.',
        'selection': 'Full penalized development tail loss within each method; no CV/target-based prior or architecture switches.'}

    def own_run(path):
        raw = Path(path) if Path(path).is_absolute() else root / path
        run = inside(raw, root)
        if raw.is_symlink() or run.parent != default.parent or not run.name.startswith('adaptive_twenty_'):
            raise ValueError('Use a dedicated immediate adaptive_twenty_ directory')
        return run
    runner.own_run = own_run

    def store_input(run, document):
        path = run / 'inputs' / (runner.inputs.digest(document) + '.json')
        if path.exists():
            if runner.read(path) != document: raise ValueError('Frozen input changed')
        else: save(path, document)
        return {**runner.ref(path), 'location': 'Frozen refinement input'}

    def prepare(run, workers=14):
        run = own_run(run)
        if run.exists(): raise FileExistsError('Keep frozen batch and use resume')
        if type(workers) is not int or not 1 <= workers <= 14: raise ValueError('Use 1–14 workers')
        prior_state = runner.read(source/'state.json'); prior = runner.checked(prior_state['protocol'])
        if (prior['version'] != 'expanded-tail-blueprint-1' or prior_state['version'] != prior['version']
                or prior_state['status'] != 'complete' or not prior_state.get('all_attempts_locked_before_scoring')
                or not prior_state.get('monitor_stopped_before_scoring') or not prior_state.get('formulas_locked_before_target_prediction')):
            raise ValueError('Completed and prediction-locked twenty-drug source required')
        cases = prior['cases']
        if (len(cases) != 20 or len({c['case_id'] for c in cases}) != 20 or len({c['compound_id'] for c in cases}) != 20
                or Counter(c['cohort'] for c in cases) != {'original': 10, 'extension': 10}):
            raise ValueError('Exactly twenty unique frozen cases required')
        runner.verify(prior['source_pins']); runner.verify(prior['paper_guard'])
        if runner.paper_pins() != prior['paper_guard']: raise ValueError('Publication file set changed')
        baseline_units = [u for u in prior['units'] if u['method'] == 'tail_balanced']
        expected = {(c['case_id'], s) for c in cases for s in runner.SCENARIOS}
        if len(baseline_units) != 80 or {(u['case_id'], u['scenario']) for u in baseline_units} != expected:
            raise ValueError('All twenty-drug tail controls required')
        pins = dict(prior['source_pins']); runner.api.pin_refs([prior, prior_state], pins)
        for path in (source/'protocol.json', source/'state.json'):
            pins[str(path.relative_to(root))] = sha256(path)
        contracts = {m: {'module': ENGINE_NAMES[m], 'version': runner.engine(m).VERSION,
            'config': deepcopy(runner.engine(m).CONFIG), 'code_pins': runner.engine(m)._code_pins(root)} for m in METHODS}
        for c in contracts.values(): pins.update(c['code_pins'])
        cache, visible = {}, {}
        selections = runner.checked(prior_state['formula_selection'])['choices']
        for unit in baseline_units:
            cid = unit['computation_id']
            if cid in cache: continue
            comp = deepcopy(prior_state['computations'][cid]); source_spec = prior['computations'][cid]
            if comp['status'] != 'predicted' or source_spec['method'] != 'tail_balanced':
                raise ValueError('Every cached control must be a successful tail prediction')
            mapping = runner.checked(comp['mapping']); path = inside(comp['mapping']['path'], root, exists=True)
            runner.engine('current')._validated(mapping, path, root)
            if mapping['variant'] != 'tail_balanced': raise ValueError('Incorrect baseline mapping variant')
            all_mappings = [runner.checked(prior_state['fits'][runner.key(cid,f)]['mapping']) for f in runner.FORMULAS]
            selected = select_formula(all_mappings)
            chosen_ref = prior_state['fits'][runner.key(cid, selected['construction_partition_method'])]['mapping'] if selected else None
            if chosen_ref != comp['mapping'] or selections[cid]['mapping'] != chosen_ref:
                raise ValueError('Cached formula differs from the frozen full-development minimum')
            fixed = runner.read(path.parent/'input.json')
            req, target, training = fixed['request'], runner.checked(comp['request']), runner.checked(mapping['training_source'])
            if runner.signature(req, target, training, 'current') != cid:
                raise ValueError('Cached scientific signature differs from its actual inputs')
            for name in ('source_pins', 'code_pins', 'retained_artifact_pins'):
                runner.verify(mapping[name]); pins.update(mapping[name])
            visible[cid] = req, target, runner.inputs.clean_training(training)
            comp.update(reused_from=runner.ref(source/'state.json'), source_method='tail_balanced',
                original_resources=deepcopy(comp['resources']),
                resources={'total_native_calls': 0, 'linear_solver_calls': 0, 'wall_seconds': 0, 'cache_reused': True})
            cache[cid] = comp
        if len(cache) != 42: raise ValueError('Expected forty-two exact cached tail computations')
        runner.verify(pins); run.mkdir(); (run/'inputs').mkdir()
        computations, units = {}, []
        for old in baseline_units:
            req, target, training = visible[old['computation_id']]
            for method in METHODS:
                cid = runner.signature(req, target, training, method)
                units.append({k: deepcopy(old[k]) for k in ('case_id', 'scenario', 'sampling', 'scientific_inputs_changed')})
                units[-1].update(id='__'.join((old['case_id'], old['scenario'], method)), method=method, computation_id=cid)
                if method != 'current' and cid not in computations:
                    computations[cid] = {'case_id': old['case_id'], 'sampling': 'full', 'method': method,
                        'request': store_input(run, req), 'target_request': store_input(run, target),
                        'training_source': {**store_input(run, training), 'role': 'calibration_data_reconstruction'}}
        if len(units) != 240 or len(computations) != 84 or set(computations) & set(cache):
            raise ValueError('Expected 240 logical units, 84 new and 42 cached computations')
        from score_expanded_tail import CODE_FILES as SCORE_CODE
        for name in set(CODE) | set(SCORE_CODE): pins[name] = sha256(root/name)
        runner.api.pin_refs([computations, cache, cases, prior['input_audit']], pins)
        protocol = {k: deepcopy(prior[k]) for k in ('cases', 'sampling', 'scenarios', 'formulas',
            'input_audit', 'changed_case_ids', 'native_threads', 'native_timeout_seconds', 'monitor_seconds',
            'directory_limit_bytes', 'bootstrap', 'endpoint_policy', 'source_sensitivity', 'limitations')}
        protocol.update(version=VERSION, created_at=runner.now(), methods=list(METHODS), units=units,
            computations=computations, cached_ids=sorted(cache), contracts=contracts, source_pins=pins,
            paper_guard=deepcopy(prior['paper_guard']), source_protocol=runner.ref(source/'protocol.json'),
            source_state=runner.ref(source/'state.json'), workers=workers, runner_binding=deepcopy(binding),
            optimization_budget={'full_fit_stages': 168, 'cv_stages': 0, 'full_optimizer_starts': 2688,
                'cv_optimizer_starts': 0, 'optimizer_starts': 2688},
            scope='Previously viewed twenty-drug development refinement, not new blind validation. Same data, exclusion RF, endpoints and missing-input conditions; no paper update.',
            scientific_change={'relaxed_tail': 'Only partition lower bound .1→.02; unchanged three axes and tail objective.',
                'adaptive_exchange': 'Same widened partition bound, plus one zero-centered regularized tissue-time heterogeneity exponent. No learned per-drug architecture or target-based rule.'},
            automatic_promotion=False, paper_updated=False)
        protocol['limitations'] += ['Very short development support cannot identify a much longer target terminal phase.',
            'The fourth axis adds a structural hypothesis; shrinkage is an engineering choice, not measured permeability or physiological confidence.',
            'All failures and regressions retained; no per-drug choice using target errors.']
        runner.verify(pins)
        if runner.paper_pins() != protocol['paper_guard']: raise ValueError('Publication changed during preparation')
        save(run/'protocol.json', protocol)
        save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'created_at': runner.now(),
            'protocol': runner.ref(run/'protocol.json'), 'fits': {}, 'computations': cache, 'monitor': {}})
        return {'drugs': 20, 'logical_units': 240, 'new_fits': 168, 'new_predictions': 84,
            'cached_predictions': 42, 'optimizer_starts': 2688, 'baseline_source_method': 'tail_balanced',
            'paper_guard_files': len(protocol['paper_guard'])}

    def load(run, locked=False):
        p, s = original_load(run, locked=locked)
        if p.get('runner_binding') != binding or runner.paper_pins() != p['paper_guard']:
            raise ValueError('Frozen runner/publication configuration changed')
        source_state = runner.checked(p['source_state'])
        for cid in p['cached_ids']:
            cached, original = s['computations'][cid], source_state['computations'][cid]
            for field in ('status', 'mapping', 'request', 'prediction', 'completion', 'dense_endpoints', 'selected_formula'):
                if cached.get(field) != original.get(field): raise ValueError('Cached baseline binding changed')
            if cached.get('source_method') != 'tail_balanced': raise ValueError('Cached baseline identity changed')
        if locked:
            if not s.get('formulas_locked_before_target_prediction'): raise ValueError('Formula lock required before prediction')
            selections = runner.checked(s['formula_selection'])['choices']
            for cid in p['computations']:
                best = select_formula([runner.checked(s['fits'][runner.key(cid,f)]['mapping']) for f in runner.FORMULAS])
                expected = None if best is None else {'formula': best['construction_partition_method'],
                    'mapping': s['fits'][runner.key(cid,best['construction_partition_method'])]['mapping']}
                if selections[cid] != expected: raise ValueError('Incorrect locked development formula selection')
                if s['computations'][cid]['status'] == 'predicted' and (expected is None or s['computations'][cid]['mapping'] != expected['mapping']):
                    raise ValueError('Prediction used a different mapping')
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
        transfer_support = []
        for case in p['cases']:
            training = store.ref(case['sampling_sources']['full'])
            last_development = max(r['time_min']/60 for r in training['observations'])
            reference_times = case['nca_policy']['sampling_time_h']
            transfer_support.append({'case_id': case['case_id'], 'drug': case['drug'],
                'last_development_time_h': last_development, 'last_reference_time_h': max(reference_times),
                'reference_points_beyond_development': sum(t > last_development for t in reference_times),
                'reference_points': len(reference_times),
                'horizon_ratio': max(reference_times)/last_development,
                'interpretation': 'Time-support diagnostic only, not a calibrated error probability or a reason to omit a case.'})
        summary, pairs = summarize_records(rows, p['cases'], runner.SCENARIOS, METHODS)
        result = {'version': VERSION, 'protocol': runner.ref(run/'protocol.json'), 'scope': p['scope'],
            'baseline_semantics': binding['baseline_interpretation'], 'units': rows, 'summary': summary,
            'candidate_to_current_curve_ratio': pairs, 'stability': prediction_stability(rows, p['cases'], store),
            'nonregression_checks': nonregression_checks(rows, summary, p['cases']),
            'transfer_support': transfer_support,
            'changed_case_ids': p['changed_case_ids'], 'unique_scored_predictions': len(cache),
            'source_pins': dict(store.pins), 'paper_guard_verified': True, 'paper_updated': False,
            'limitations': p['limitations']}
        store.unchanged()
        if runner.paper_pins() != p['paper_guard']: raise ValueError('Publication changed during scoring')
        save(run/'results.json', result)
        table = [{'cohort': r['cohort'], 'scenario': r['scenario'], 'method': r['method'], 'requested': r['requested'],
            'scored': r['scored'], 'failed': r['failed'], 'curve_gmfe': r['curve_gmfe']['estimate'],
            'ci95_low': (r['curve_gmfe']['ci95'] or [None,None])[0], 'ci95_high': (r['curve_gmfe']['ci95'] or [None,None])[1],
            'within_twofold': r['within_twofold']} for r in summary]
        with (run/'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        return result

    def status(run):
        result = original_status(run); p = runner.read(run/'protocol.json'); counts = Counter()
        for cid in p['computations']:
            for formula in runner.FORMULAS:
                path = run/'work'/cid/formula/'progress.json'
                if path.exists():
                    progress = runner.read(path)
                    counts['full_starts_completed'] += len(progress.get('starts', []))
                    counts['full_stages_completed'] += int(progress.get('status') == 'locked')
        result['optimization_progress'] = {'completed': dict(counts), 'planned': p['optimization_budget']}
        return result
    runner.prepare, runner.load, runner.report, runner.status = prepare, load, report, status
    return runner


if __name__ == '__main__':
    create_runner().main()
