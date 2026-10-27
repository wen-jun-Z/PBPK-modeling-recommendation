"""Native, study-separated development experiments with shared low-level I/O.

The model cache contains only requested native curves, never another method's
candidate scores or visited candidate list. All logical requests are charged.
This runner does not read a final test set and cannot declare a paper benchmark
complete. Native constitutive alternatives are not anatomical topology edits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from numbers import Integral
from pathlib import Path
import threading
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from .design_diagnostics import BudgetExceeded, BudgetLedger, analyze_identifiability
from .design_runtime import execute_snapshot
from .design_scoring import score_calibration_curves
from .parameter_batch import run_parameter_batch


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with target.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if replace:
            target.replace(path)
        else:
            # Same-directory hard link publishes atomically without overwriting
            # an existing frozen artifact. The temporary link is ours alone.
            os.link(target,path)
    finally:
        target.unlink(missing_ok=True)


def validate_case(case):
    if case.get('purpose') not in {'historical_development_reconstruction', 'historical_development_experiment'}:
        raise ValueError('Only explicitly declared historical development cases are supported')
    scenarios = case.get('scenarios', [])
    if not scenarios or any(s.get('role') not in {'calibration', 'selection'} for s in scenarios):
        raise ValueError('Development scenarios must contain only calibration/selection studies')
    if len({s['scenario_id'] for s in scenarios}) != len(scenarios):
        raise ValueError('Scenario identifiers must be unique')
    groups = {role: {s['study_group_id'] for s in scenarios if s['role'] == role}
              for role in ('calibration', 'selection')}
    if not all(groups.values()) or groups['calibration'] & groups['selection']:
        raise ValueError('Calibration and selection must be nonempty disjoint study groups')
    if (groups['calibration'] | groups['selection']) & set(case.get('forbidden_study_group_ids', [])):
        raise ValueError('A forbidden study entered the designer')
    specs = case['parameter_search']['parameters']
    if not specs or len({p['parameter_id'] for p in specs}) != len(specs):
        raise ValueError('Parameter definitions must be unique and nonempty')
    for spec in specs:
        if not all(math.isfinite(spec[k]) for k in ('initial_value', 'lower_bound', 'upper_bound')):
            raise ValueError('Parameter bounds must be finite')
        if not spec['lower_bound'] <= spec['initial_value'] <= spec['upper_bound'] or spec['upper_bound'] <= spec['lower_bound']:
            raise ValueError('Parameter bounds/initial value are inconsistent')
        if spec.get('search_scale') not in {None,'linear','linear_nonnegative','log','log10'}:
            raise ValueError('Unknown parameter search coordinate')
        if spec.get('search_scale') in {'log','log10'} and spec['lower_bound'] <= 0:
            raise ValueError('Log search coordinates require positive lower bounds')
        if spec.get('sensitivity_scale') not in {None,'log','bounded_linear'}:
            raise ValueError('Unknown sensitivity coordinate')
    return groups


def candidate_space(case, *, root):
    from .native_edits import partition_candidates
    specs = case['parameter_search']['parameters']
    ids = [p['parameter_id'] for p in specs]
    mechanism_contract = case.get('mechanism_search')
    if mechanism_contract:
        from .mechanism_edits import native_candidates
        if mechanism_contract.get('version') != 'native-metabolism-search-v1':
            raise ValueError('Unknown native mechanism-search contract')
        variants = native_candidates(case['scenarios'][0]['design_request'], project_root=root,
                                     parameter_specs=specs)
        expected = mechanism_contract.get('expected_structure_ids')
        if not expected or set(expected) != {v['structure_id'] for v in variants}:
            raise ValueError('Compiled mechanisms do not match the preregistered candidate space')
    else:
        variants = partition_candidates(case['scenarios'][0]['design_request'], project_root=root,
                                        parameter_specs=specs)
    candidates = []
    for index, variant in enumerate(variants):
        for flags in itertools.product((False, True), repeat=len(ids)):
            free = [name for name, flag in zip(ids, flags, strict=True) if flag]
            candidate = {
                'candidate_id': variant['structure_id'] + '__' + (''.join('1' if x else '0' for x in flags)),
                'structure_id': variant['structure_id'], 'action': variant['structure_id'],
                'structure_semantic_hash': variant['semantic_hash'],
                'parameterization_hash': digest({'specs': specs, 'free': free}),
                'parameter_ids': ids, 'free_parameter_ids': free,
                'features': [float(index), *map(float, flags)],
                # All currently legal candidates satisfy the same plasma task.
                # Do not invent different evidence/requirement support scores.
                'requirement_score': 1.0, 'evidence_score': 1.0,
                'guidance_is_informative_in_this_case': False,
                'edit_count': variant['edit_count'],
                'structure_edit_count': variant['edit_count'],
                'structure_stage_default': free == ids,
            }
            if mechanism_contract:
                derived_specs = variant['parameter_specs']
                validate_candidate_parameter_coordinates(specs, derived_specs)
                candidate['parameterization_hash'] = digest({'specs':derived_specs, 'free':free})
                candidate['native_mechanism_search'] = True
                candidate['mechanism_manifest'] = variant['edit_manifest']
                # Independent binary factors avoid an artificial ordinal distance
                # between the partition and metabolic constitutive choices.
                candidate['features'] = [float('rodgers_rowland' in variant['structure_id']),
                    float('mm_low_concentration_first_order' in variant['structure_id']), *map(float,flags)]
            candidates.append(candidate)
    return candidates


def validate_candidate_parameter_coordinates(source_specs, derived_specs):
    """Mechanism edits may change native paths/units, never optimizer coordinates."""
    fields = ('parameter_id', 'initial_value', 'lower_bound', 'upper_bound', 'unit',
              'search_scale', 'sensitivity_scale')
    if len(source_specs) != len(derived_specs) or any(
            any(source.get(k) != derived.get(k) for k in fields)
            for source, derived in zip(source_specs, derived_specs, strict=True)):
        raise ValueError('A mechanism edit changed the common optimization coordinates')
    return derived_specs


def aggregate_scores(scenarios, scores):
    """Study-equal, then curve-equal calibration objective; no observed rows dropped."""
    count = len(scores[0]['rows'])
    rows = []
    for index in range(count):
        per_scene = {s['scenario_id']: score['rows'][index] for s, score in zip(scenarios, scores, strict=True)}
        if any(not row['valid'] for row in per_scene.values()):
            rows.append({'valid': False, 'log_rmse': None, 'gmfe': None, 'scenarios': per_scene})
            continue
        grouped = {}
        for scenario in scenarios:
            grouped.setdefault(scenario['study_group_id'], []).append(per_scene[scenario['scenario_id']])
        mse = sum(sum(r['log_mse'] for r in group) / len(group) for group in grouped.values()) / len(grouped)
        absolute = sum(sum(math.log(r['gmfe']) for r in group) / len(group) for group in grouped.values()) / len(grouped)
        rows.append({'valid': True, 'log_rmse': math.sqrt(mse), 'gmfe': math.exp(absolute), 'scenarios': per_scene})
    return rows


def loss(score):
    return score['log_rmse'] if score['valid'] else math.inf


class ScopedLedger:
    """Enforce the search strategy's phase allowance across parallel scenarios."""
    def __init__(self, ledger, allowance):
        if isinstance(allowance, bool) or not isinstance(allowance, Integral) or allowance < 0:
            raise ValueError('Scoped allowance must be a nonnegative integer')
        self.ledger, self.allowance, self.used = ledger, allowance, 0
        self.lock = threading.Lock()

    def _reserve(self, calls, phase):
        if isinstance(calls, bool) or not isinstance(calls, Integral) or calls <= 0:
            raise ValueError('Scenario calls must be a positive integer')
        if phase != 'search':
            return
        with self.lock:
            if calls > self.allowance - self.used:
                error = BudgetExceeded('search', calls, self.allowance - self.used)
                error.budget_scope = 'phase'
                raise error
            self.used += calls

    def _release_unstarted(self, calls, phase):
        if phase == 'search':
            with self.lock:
                self.used -= calls

    def consume(self, scenario_calls=1, *, phase='search', cache_hit=False):
        self._reserve(scenario_calls, phase)
        try:
            return self.ledger.consume(scenario_calls, phase=phase, cache_hit=cache_hit)
        except BaseException:
            self._release_unstarted(scenario_calls, phase)
            raise

    def run(self, callback, *, scenario_calls=1, phase='search'):
        if not callable(callback):
            raise ValueError('callback must be callable')
        self._reserve(scenario_calls, phase)
        entered = False
        def execute():
            nonlocal entered
            entered = True
            return callback()
        try:
            return self.ledger.run(execute, scenario_calls=scenario_calls, phase=phase)
        except BaseException:
            if not entered:
                self._release_unstarted(scenario_calls, phase)
            raise

    def snapshot(self):
        return self.ledger.snapshot()

    def remaining(self, phase='search'):
        inherited = (self.ledger.remaining(phase) if isinstance(self.ledger, ScopedLedger)
                     else self.ledger.snapshot()['phases'][phase]['remaining'])
        if phase != 'search':
            return inherited
        with self.lock:
            return min(inherited, self.allowance - self.used)


def validate_native_parallelism(*, cpu_threads, scenario_workers, drug_workers=1):
    """Bound nested native workers by this process's allowed CPU affinity."""
    for name, value, maximum in [('cpu_threads', cpu_threads, 32),
                                 ('scenario_workers', scenario_workers, 8),
                                 ('drug_workers', drug_workers, 8)]:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'{name} must be an integer in 1..{maximum}')
    available = len(os.sched_getaffinity(0))
    requested = cpu_threads * scenario_workers * drug_workers
    if requested > available:
        raise ValueError(f'Requested native concurrency {requested} exceeds CPU affinity {available}')
    return {'native_cpu_concurrency_limit': requested, 'available_cpu_affinity': available}


class NativeExperiment:
    def __init__(self, *, case, output, project_root, device='cuda', cpu_threads=4,
                 scenario_workers=3, max_output_bytes=1024**3, progress_callback=None,
                 fit_start_strategy='legacy', fit_strategy='coordinate',
                 require_identifiability=False):
        validate_case(case)
        if fit_start_strategy not in {'legacy', 'source_anchor'}:
            raise ValueError('fit_start_strategy must be legacy or source_anchor')
        if fit_strategy not in {'coordinate', 'residual', 'screened_residual'}:
            raise ValueError('fit_strategy must be coordinate, residual, or screened_residual')
        if type(require_identifiability) is not bool:
            raise ValueError('require_identifiability must be a boolean')
        self.fit_start_strategy = fit_start_strategy
        self.fit_strategy = fit_strategy
        self.require_identifiability = require_identifiability
        self.parallelism = validate_native_parallelism(cpu_threads=cpu_threads,
                                                       scenario_workers=scenario_workers)
        self.root = Path(project_root).resolve(strict=True)
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(self.root) or self.output == self.root:
            raise ValueError('Experiment output must remain in the project')
        self.output.mkdir(parents=True, exist_ok=False)
        self.case, self.device = case, device
        self.specs = case['parameter_search']['parameters']
        self.initial = [p['initial_value'] for p in self.specs]
        self.ids = [p['parameter_id'] for p in self.specs]
        self.cpu_threads, self.workers = cpu_threads, scenario_workers
        self.maximum_bytes = max_output_bytes
        self.progress_callback = progress_callback or (lambda **items: None)
        self.models, self.curves, self.rebuilt = {}, {}, {}
        self.sequence, self.lock = 0, threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=scenario_workers)
        self.started = time.monotonic()
        save(self.output / 'case.json', case)

    def close(self):
        self.pool.shutdown(wait=True)

    def disk_bytes(self):
        total = 0
        for path in self.output.rglob('*'):
            if path.is_symlink():
                raise ValueError('Experiment output must not contain symlinks')
            if path.is_file():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
        if total > self.maximum_bytes:
            raise RuntimeError('Experiment output limit exceeded; further execution stopped')
        return total

    def call_limit(self):
        allowance = min(128 * 1024**2, (self.maximum_bytes - self.disk_bytes() - 4*1024**2) // self.workers)
        if allowance < 1024**2:
            raise RuntimeError('Insufficient remaining experiment disk budget')
        return allowance

    def _model(self, candidate, scenario, context, phase):
        from .native_edits import compile_candidate
        if candidate.get('native_mechanism_search'):
            from .mechanism_edits import compile_native_candidate as compile_candidate
        key = (candidate['structure_id'], scenario['scenario_id'])
        if key not in context['model_requests']:
            cached = key in self.models
            if cached:
                context['ledger'].consume(phase=phase, cache_hit=True)
            else:
                variant = compile_candidate(scenario['design_request'], candidate['action'],
                                            project_root=self.root, parameter_specs=self.specs)
                directory = self.output / 'native_models' / digest(key)[:20]
                self.progress_callback(stage='native_model_build', scenario=scenario['scenario_id'], structure=candidate['structure_id'])
                def execute():
                    manifest = execute_snapshot(snapshot=variant['snapshot'], output_dir=directory, project_root=self.root,
                                                timeout_seconds=180, cpu_threads=self.cpu_threads, max_output_bytes=self.call_limit())
                    if not manifest['engineering_accepted']:
                        raise RuntimeError('Native model build failed: ' + str(directory))
                    return manifest
                manifest = context['ledger'].run(execute, phase=phase)
                if len(manifest['pkml_outputs']) != 1:
                    raise RuntimeError('Expected one model per scenario')
                pkml = directory / manifest['pkml_outputs'][0]['path']
                self.models[key] = {'pkml': pkml, 'sha256': file_sha(pkml), 'variant': variant,
                                    'manifest': str((directory/'execution_manifest.json').relative_to(self.root))}
                save(directory / 'edit_manifest.json', variant['edit_manifest'])
            context['model_requests'].add(key)
        return self.models[key]

    def _predict(self, candidate, scenario, points, context, phase):
        model = self._model(candidate, scenario, context, phase)
        ledger = context['ledger']
        remaining = ledger.snapshot()['phases'][phase]['remaining']
        if remaining < len(points):
            raise BudgetExceeded(phase, len(points), remaining)
        keys, pending = [], {}
        for point in points:
            if len(point) != len(self.specs) or any(not math.isfinite(x) or not p['lower_bound'] <= x <= p['upper_bound']
                                                    for x, p in zip(point, self.specs, strict=True)):
                raise ValueError('Native parameter vector is outside its declared bounds')
            key = (model['sha256'], scenario['target_output'], tuple(point))
            keys.append(key)
            if key in self.curves or key in pending:
                ledger.consume(phase=phase, cache_hit=True)
            else:
                pending[key] = point
        if pending:
            values = list(pending.values())
            with self.lock:
                self.sequence += 1
                directory = self.output / 'native_batches' / ('batch_%05d' % self.sequence)
            mapped_specs = validate_candidate_parameter_coordinates(self.specs,
                model['variant']['parameter_specs']) if candidate.get('native_mechanism_search') else self.specs
            native_specs = [{'path': m['runtime_path'], 'unit': m.get('native_unit', spec['unit'])}
                            for spec in mapped_specs for m in spec['runtime_mappings']]
            native_values = [[point[i] * m['multiplier'] for i, spec in enumerate(mapped_specs) for m in spec['runtime_mappings']]
                             for point in values]
            limit = self.call_limit()
            def execute():
                result = run_parameter_batch(pkml_path=model['pkml'], parameter_specs=native_specs,
                        values=native_values, target_output=scenario['target_output'], output_dir=directory,
                        project_root=self.root, timeout_seconds=180, cpu_threads=self.cpu_threads, max_output_bytes=limit)
                if result['status'] != 'completed':
                    raise RuntimeError('Native parameter batch failed: ' + str(directory))
                return result
            result = ledger.run(execute, scenario_calls=len(values), phase=phase)
            for key, curve in zip(pending, result['runs'], strict=True):
                self.curves[key] = {'curve': curve, 'native_results': str((directory/'native_results.json').relative_to(self.root)),
                                    'native_results_sha256': result['native_results_sha256']}
        return [dict(self.curves[key]['curve'], vector_index=i) for i, key in enumerate(keys)]

    def score(self, candidate, points, context, *, role='calibration', phase='search'):
        scenarios = [s for s in self.case['scenarios'] if s['role'] == role]
        futures = [self.pool.submit(self._predict, candidate, s, points, context, phase) for s in scenarios]
        curves, error = [], None
        for future in futures:
            try:
                curves.append(future.result())
            except BaseException as exc:
                curves.append(None)
                error = error or exc
        if error:
            raise error
        scores = [score_calibration_curves(runs=curve, observations=s['observations'], device=self.device)
                  for s, curve in zip(scenarios, curves, strict=True)]
        self.disk_bytes()
        return aggregate_scores(scenarios, scores)

    def diagnose(self, candidate, point, free_ids, context, *, phase='search'):
        if not free_ids:
            return {'status': 'no_free_parameters', 'rank': 0, 'selected_parameter_ids': [], 'step_stability_checked': True}
        indices = [self.ids.index(name) for name in free_ids]
        diagnostics = []
        scenarios = [s for s in self.case['scenarios'] if s['role'] == 'calibration']
        n_groups = len({s['study_group_id'] for s in scenarios})
        for step in (0.001, 0.0005):
            pairs, deltas = [], []
            for index in indices:
                spec, value = self.specs[index], point[index]
                if spec.get('sensitivity_scale') == 'bounded_linear':
                    width = spec['upper_bound'] - spec['lower_bound']
                    minus = max(spec['lower_bound'], value-step*width)
                    plus = min(spec['upper_bound'], value+step*width)
                    delta = (plus-minus)/width
                else:
                    if value <= 0:
                        return {'status': 'not_evaluable', 'reason': 'log_difference_boundary', 'selected_parameter_ids': free_ids}
                    minus = max(spec['lower_bound'], value * math.exp(-step))
                    plus = min(spec['upper_bound'], value * math.exp(step))
                    if minus <= 0 or plus <= minus:
                        return {'status': 'not_evaluable', 'reason': 'log_difference_boundary', 'selected_parameter_ids': free_ids}
                    delta = math.log(plus/minus)
                left, right = list(point), list(point)
                left[index], right[index] = minus, plus
                pairs.extend((left, right)); deltas.append(delta)
            scores = self.score(candidate, pairs, context, phase=phase)
            if any(not row['valid'] for row in scores):
                return {'status': 'not_evaluable', 'reason': 'invalid_perturbation', 'selected_parameter_ids': free_ids}
            jacobian = []
            for scenario in scenarios:
                scene_id = scenario['scenario_id']
                count = len(scenario['observations'])
                same_group = sum(s['study_group_id'] == scenario['study_group_id'] for s in scenarios)
                weight = 1 / math.sqrt(n_groups * same_group * count)
                for row in range(count):
                    jacobian.append([weight * math.log(scores[2*j+1]['scenarios'][scene_id]['predictions_umol_per_l'][row]
                                / scores[2*j]['scenarios'][scene_id]['predictions_umol_per_l'][row]) / deltas[j]
                                for j in range(len(indices))])
            diagnostic = analyze_identifiability(free_ids, jacobian)
            diagnostic.update(observation_source_verified=True, difference_log_step=step)
            diagnostics.append(diagnostic)
        stable = (diagnostics[0]['rank'] == diagnostics[1]['rank']
                  and set(diagnostics[0]['selected_parameter_ids']) == set(diagnostics[1]['selected_parameter_ids']))
        result = dict(diagnostics[0], status='evaluated', step_stability_checked=True,
                      step_stability_passed=stable, second_step_diagnostic=diagnostics[1],
                      scaling='log-prediction; explicit parameter coordinates; study/curve/point-equal weights; no assumed observation noise',
                      parameter_coordinates={self.ids[i]:self.specs[i].get('sensitivity_scale','log') for i in indices})
        if not stable:
            result['selected_parameter_ids'] = list(free_ids)
            result['feedback_action'] = 'retain_requested_subset_due_to_unstable_diagnostic'
        return result

    def _encode(self, point):
        return [(x-p['lower_bound'])/(p['upper_bound']-p['lower_bound']) if self._linear_coordinate(p) else
                math.log(x/p['lower_bound'])/math.log(p['upper_bound']/p['lower_bound'])
                for x, p in zip(point, self.specs, strict=True)]

    def _decode(self, coordinate):
        values = [p['lower_bound'] + x*(p['upper_bound']-p['lower_bound']) if self._linear_coordinate(p) else
                  math.exp(math.log(p['lower_bound']) + x*math.log(p['upper_bound']/p['lower_bound']))
                  for x, p in zip(coordinate, self.specs, strict=True)]
        return [min(p['upper_bound'], max(p['lower_bound'], x)) for x, p in zip(values, self.specs, strict=True)]

    @staticmethod
    def _linear_coordinate(spec):
        return spec['lower_bound'] <= 0 or spec.get('search_scale') in {'linear','linear_nonnegative'}

    @staticmethod
    def _remaining_calls(context, phase='search'):
        ledger = context['ledger']
        return (ledger.remaining(phase) if isinstance(ledger, ScopedLedger)
                else ledger.snapshot()['phases'][phase]['remaining'])

    def _score_call_cost(self, candidate, vector_count, context, *, role='calibration'):
        """Whole-call cost, including each not-yet-requested native model.

        Cache hits are deliberately not deducted: this is the same logical cost
        for a cold or warm method. A whole multi-scenario batch is checked before
        any parallel worker may spend part of its budget.
        """
        scenarios = [s for s in self.case['scenarios'] if s['role'] == role]
        requested = context.get('model_requests', set())
        structure = candidate.get('structure_id', candidate['candidate_id'])
        builds = sum((structure, s['scenario_id']) not in requested for s in scenarios)
        return vector_count * len(scenarios) + builds

    def bounded_diagnose(self, candidate, point, free_ids, context, *, reserve_calls=0, phase='search'):
        """Never consume a partial numerical diagnostic or manufacture a pass."""
        if free_ids:
            required = self._score_call_cost(candidate, 4 * len(free_ids), context)
            remaining = self._remaining_calls(context, phase)
            if required + reserve_calls > remaining:
                return {'status': 'unavailable', 'reason': 'insufficient_diagnostic_budget',
                        'rank': None, 'selected_parameter_ids': list(free_ids),
                        'step_stability_checked': False, 'step_stability_passed': False,
                        'required_solver_calls': required, 'remaining_solver_calls': remaining,
                        'reserved_solver_calls': reserve_calls, 'budget_phase': phase}
        try:
            return self.diagnose(candidate, point, free_ids, context, phase=phase)
        except BudgetExceeded:
            # Retain any real charge, but never reuse a partial-step rank as a
            # completed diagnostic. The caller may still submit its incumbent.
            return {'status': 'unavailable', 'reason': 'diagnostic_budget_exhausted',
                    'rank': None, 'selected_parameter_ids': list(free_ids),
                    'step_stability_checked': False, 'step_stability_passed': False,
                    'budget_phase': phase}

    def fit(self, candidate, free_ids, context):
        if getattr(self, 'fit_strategy', 'coordinate') == 'screened_residual':
            return self.fit_residual(candidate, free_ids, context, screen_starts=True)
        if getattr(self, 'fit_strategy', 'coordinate') == 'residual':
            return self.fit_residual(candidate, free_ids, context)
        strategy = getattr(self, 'fit_start_strategy', 'legacy')
        initial_cost = self._score_call_cost(candidate, 1, context)
        if initial_cost > self._remaining_calls(context):
            raise BudgetExceeded('search', initial_cost, self._remaining_calls(context))
        initial_score = self.score(candidate, [self.initial], context)[0]
        best = {'point': list(self.initial), 'score': initial_score, 'source': 'initial'}
        trace, starts = [], []
        def report(reason=None):
            return {'starts': starts, 'trace': trace, 'initial_score': initial_score,
                    'fit_strategy': 'coordinate',
                    'fit_start_strategy': strategy,
                    'initial_score_reused_as_start': bool(starts) and strategy == 'source_anchor',
                    'optimizer_convergence_proven': False, 'budget_truncated': reason is not None,
                    'budget_stop_stage': reason}
        if not free_ids:
            return best, report()
        axes = [self.ids.index(name) for name in free_ids]
        if strategy == 'source_anchor':
            # The incumbent was already scored and charged above. Reuse that
            # exact point as an optimizer trajectory, without another call.
            # evaluations includes its one reused initial evaluation; unlike
            # legacy, total vector calls therefore equals sum(start evaluations).
            starts = [{'x': self._encode(self.initial), 'point': list(self.initial),
                       'step': .25, 'evaluations': 1, 'score': initial_score,
                       'origin': 'source_anchor', 'initial_point': list(self.initial)}]
        proposed_starts = []
        for fraction in ((0.1, 0.9) if strategy == 'source_anchor' else (0.1, 0.5, 0.9)):
            x = self._encode(self.initial)
            for axis in axes:
                x[axis] = fraction
            point = self._decode(x)
            # Fixed dimensions must retain the exact original float, not a log roundtrip.
            for axis in set(range(len(self.ids))) - set(axes):
                point[axis] = self.initial[axis]
            proposed_starts.append({'x': x, 'point': point, 'step': .25, 'evaluations': 1})
        if self._score_call_cost(candidate, len(proposed_starts), context) > self._remaining_calls(context):
            return best, report('before_multistart_batch')
        try:
            scores = self.score(candidate, [s['point'] for s in proposed_starts], context)
        except BudgetExceeded:
            return best, report('multistart_batch_budget_exhausted')
        starts.extend(proposed_starts)
        for state, score in zip(proposed_starts, scores, strict=True):
            state['score'] = score
            if loss(score) < loss(best['score']):
                best = {'point': state['point'], 'score': score, 'source': 'fit'}
        for iteration in range(25):
            proposals, owners = [], []
            increments = [0] * len(starts)
            for i, state in enumerate(starts):
                if state['step'] < 1e-4:
                    continue
                for axis in axes:
                    for sign in (-1, 1):
                        if state['evaluations'] + increments[i] >= 50:
                            continue
                        x = list(state['x']); x[axis] = max(0., min(1., x[axis] + sign*state['step']))
                        point = self._decode(x)
                        for fixed in set(range(len(self.ids))) - set(axes):
                            point[fixed] = self.initial[fixed]
                        proposals.append(point); owners.append((i, x)); increments[i] += 1
            if not proposals:
                break
            if self._score_call_cost(candidate, len(proposals), context) > self._remaining_calls(context):
                return best, report('before_coordinate_batch')
            try:
                scores = self.score(candidate, proposals, context)
            except BudgetExceeded:
                return best, report('coordinate_batch_budget_exhausted')
            for state, increment in zip(starts, increments, strict=True):
                state['evaluations'] += increment
            improved = set()
            for point, score, (index, x) in zip(proposals, scores, owners, strict=True):
                if loss(score) < loss(starts[index]['score']):
                    starts[index].update(point=point, x=x, score=score); improved.add(index)
                if loss(score) < loss(best['score']):
                    best = {'point': point, 'score': score, 'source': 'fit'}
            for i, state in enumerate(starts):
                if i not in improved:
                    state['step'] /= 2
            trace.append({'round': iteration+1, 'best_log_rmse': best['score']['log_rmse'],
                          'evaluations': [s['evaluations'] for s in starts]})
            self.progress_callback(stage='parameter_fit', candidate=candidate['candidate_id'], fit_round=iteration+1,
                                   calibration_log_rmse=best['score']['log_rmse'], budget=context['ledger'].snapshot())
        return best, report()

    def _calibration_residuals(self, score):
        """Weighted log residuals whose squared norm is the existing objective.

        Source observations and native predictions retain every Dcal point.
        Neither Dsel nor user-requested desired outputs are observations.
        """
        if not score['valid']:
            raise ValueError('Residual fitting requires valid calibration predictions')
        scenarios = [s for s in self.case['scenarios'] if s['role'] == 'calibration']
        groups = {s['study_group_id'] for s in scenarios}
        residuals = []
        for scenario in scenarios:
            observations = scenario['observations']
            predictions = score['scenarios'][scenario['scenario_id']]['predictions_umol_per_l']
            same_group = sum(s['study_group_id'] == scenario['study_group_id'] for s in scenarios)
            weight = 1 / math.sqrt(len(groups) * same_group * len(observations))
            for predicted, observed in zip(predictions, observations, strict=True):
                observed = observed['concentration_umol_per_l']
                if not (math.isfinite(predicted) and predicted > 0 and math.isfinite(observed) and observed > 0):
                    raise ValueError('Residual fitting requires positive finite predictions and observations')
                residuals.append(weight * (math.log(predicted) - math.log(observed)))
        norm = math.sqrt(math.fsum(value * value for value in residuals))
        if not math.isclose(norm, score['log_rmse'], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError('Calibration residuals disagree with the study-equal scoring objective')
        return residuals

    def fit_residual(self, candidate, free_ids, context, *, screen_starts=False):
        """Bounded local joint updates, accepted only after native Dcal scoring.

        At most two sourced parameters are fitted in their declared normalized
        coordinates. Finite differences and line-search proposals are submitted
        as whole batches; a partial budget retains the best actually scored
        point. This is a local optimizer, not global identifiability evidence.
        Optional deterministic screening spends the same candidate allowance on
        a small Dcal-only grid before reusing its best actual score in this same
        local optimizer. No selection score participates in choosing a start.
        """
        import numpy as np

        if len(free_ids) > 2 or len(set(free_ids)) != len(free_ids) or any(name not in self.ids for name in free_ids):
            raise ValueError('Residual fitting requires at most two unique declared free parameters')
        fit_calls_before = context['ledger'].snapshot()['totals']['logical_calls']
        initial_cost = self._score_call_cost(candidate, 1, context)
        if initial_cost > self._remaining_calls(context):
            raise BudgetExceeded('search', initial_cost, self._remaining_calls(context))
        initial_score = self.score(candidate, [self.initial], context)[0]
        best = {'point': list(self.initial), 'score': initial_score, 'source': 'initial'}
        trace, starts = [], []
        stop_reason = None
        screening = None
        screening_budget_stop = None

        def report(budget_stop=None):
            result = {'starts': starts, 'trace': trace, 'initial_score': initial_score,
                    'fit_strategy': 'residual', 'fit_start_strategy': 'source_anchor',
                    'initial_score_reused_as_start': bool(starts),
                    'optimizer_convergence_proven': False,
                    'budget_truncated': budget_stop is not None,
                    'budget_stop_stage': budget_stop, 'stop_reason': stop_reason,
                    'objective': 'study-equal/curve-equal/point-equal squared natural-log residuals',
                    'rank_interpretation': 'local finite-difference sensitivity; not global identifiability'}
            if screen_starts:
                total_calls = context['ledger'].snapshot()['totals']['logical_calls'] - fit_calls_before
                result.update(fit_strategy='screened_residual',
                    fit_start_strategy='calibration_grid_screen', start_screening=screening,
                    initial_score_reused_as_start=bool(starts) and starts[0]['origin'] == 'source_anchor',
                    selected_score_reused_as_start=bool(starts),
                    fit_solver_calls=total_calls, initial_solver_calls=initial_cost,
                    local_solver_calls=total_calls - initial_cost - screening['solver_calls'],
                    budget_truncated=budget_stop is not None or screening_budget_stop is not None,
                    budget_stop_stage=budget_stop or screening_budget_stop)
            return result

        if screen_starts:
            screening = {'status': 'not_applicable' if not free_ids else 'pending',
                'free_parameter_ids': list(free_ids), 'grid_levels': [.1, .5, .9],
                'coordinate_system': 'declared normalized parameter bounds; fixed values unchanged',
                'planned_grid_points': 3 ** len(free_ids) if free_ids else 0,
                'evaluated_grid_points': 0, 'reused_source_grid_points': 0,
                'grid_records': [], 'solver_calls': 0, 'complete': not free_ids,
                'selected_point': list(self.initial), 'selected_origin': 'source_anchor',
                'score_roles': ['calibration'], 'global_optimum_claimed': False}
        if not free_ids or (not initial_score['valid'] and not screen_starts):
            stop_reason = 'no_free_parameters' if not free_ids else 'invalid_source_prediction'
            return best, report()
        axes = [self.ids.index(name) for name in free_ids]
        fixed = set(range(len(self.ids))) - set(axes)
        trust_radius, damping = .25, 1e-3

        def decode(coordinate):
            point = self._decode(coordinate)
            for axis in fixed:
                point[axis] = self.initial[axis]
            return point

        def retain(points, scores):
            nonlocal best
            for point, score in zip(points, scores, strict=True):
                if loss(score) < loss(best['score']):
                    best = {'point': point, 'score': score, 'source': 'residual_fit'}

        if screen_starts:
            anchor = self._encode(self.initial)
            screening['selected_normalized_free_coordinates'] = [anchor[axis] for axis in axes]
            screening['selected_calibration_log_rmse'] = initial_score.get('log_rmse')
            grid_points, grid_coordinates = [], []
            for coordinates in itertools.product(screening['grid_levels'], repeat=len(axes)):
                normalized = list(anchor)
                for axis, value in zip(axes, coordinates, strict=True):
                    normalized[axis] = value
                source_match = all(math.isclose(anchor[axis], value, rel_tol=0., abs_tol=1e-12)
                                   for axis, value in zip(axes, coordinates, strict=True))
                if source_match:
                    screening['grid_records'].append({'normalized_free_coordinates': list(coordinates),
                        'point': list(self.initial), 'origin': 'reused_source_anchor',
                        'valid': initial_score['valid'], 'calibration_log_rmse': initial_score.get('log_rmse')})
                    screening['reused_source_grid_points'] += 1
                else:
                    grid_points.append(decode(normalized))
                    grid_coordinates.append(list(coordinates))
            cost = self._score_call_cost(candidate, len(grid_points), context) if grid_points else 0
            available = self._remaining_calls(context)
            screening.update(required_solver_calls=cost, available_solver_calls=available,
                             evaluated_grid_points=len(screening['grid_records']))
            if cost > available:
                screening.update(status='source_only_insufficient_budget')
                screening_budget_stop = 'before_start_screening_batch'
            else:
                before = context['ledger'].snapshot()['totals']['logical_calls']
                try:
                    scores = self.score(candidate, grid_points, context) if grid_points else []
                except BudgetExceeded:
                    screening.update(status='screening_batch_budget_exhausted',
                        solver_calls=context['ledger'].snapshot()['totals']['logical_calls'] - before)
                    screening_budget_stop = 'start_screening_batch_budget_exhausted'
                    # A partial native batch has no aggregate score to reuse.
                    return best, report(screening_budget_stop)
                screening['solver_calls'] = context['ledger'].snapshot()['totals']['logical_calls'] - before
                for point, coordinates, score in zip(grid_points, grid_coordinates, scores, strict=True):
                    screening['grid_records'].append({'normalized_free_coordinates': coordinates,
                        'point': list(point), 'origin': 'grid', 'valid': score['valid'],
                        'calibration_log_rmse': score.get('log_rmse')})
                    if loss(score) < loss(best['score']):
                        best = {'point': list(point), 'score': score, 'source': 'screened_start'}
                screening.update(status='completed', complete=True)
            screening['evaluated_grid_points'] = len(screening['grid_records'])
            screening['selected_point'] = list(best['point'])
            screening['selected_origin'] = 'grid' if best['source'] == 'screened_start' else 'source_anchor'
            screening['selected_normalized_free_coordinates'] = [self._encode(best['point'])[axis] for axis in axes]
            screening['selected_calibration_log_rmse'] = best['score'].get('log_rmse')
            self.progress_callback(stage='parameter_start_screening', candidate=candidate['candidate_id'],
                screening_status=screening['status'], evaluated_grid_points=screening['evaluated_grid_points'],
                calibration_log_rmse=best['score'].get('log_rmse'), budget=context['ledger'].snapshot())
            if not best['score']['valid']:
                stop_reason = 'no_valid_screened_calibration_prediction'
                return best, report()
        starts.append({'origin': screening['selected_origin'] if screen_starts else 'source_anchor',
                       'initial_point': list(best['point']), 'evaluations': 1})

        for iteration in range(20):
            x = np.asarray(self._encode(best['point']), dtype=float)
            residual = np.asarray(self._calibration_residuals(best['score']), dtype=float)
            previous_loss = loss(best['score'])
            pairs, deltas = [], []
            for axis in axes:
                left, right = x.copy(), x.copy()
                # A projected central difference becomes one-sided at a bound.
                left[axis], right[axis] = max(0., x[axis] - .001), min(1., x[axis] + .001)
                pairs.extend((decode(left), decode(right)))
                deltas.append(float(right[axis] - left[axis]))
            if self._score_call_cost(candidate, len(pairs), context) > self._remaining_calls(context):
                return best, report('before_residual_jacobian_batch')
            try:
                scores = self.score(candidate, pairs, context)
            except BudgetExceeded:
                return best, report('residual_jacobian_batch_budget_exhausted')
            starts[0]['evaluations'] += len(pairs)
            retain(pairs, scores)
            item = {'round': iteration + 1, 'before_log_rmse': previous_loss,
                    'trust_radius': trust_radius, 'damping': damping}
            if any(not score['valid'] for score in scores):
                trust_radius /= 2
                damping *= 10
                item.update(local_rank=None, reason='invalid_finite_difference',
                            best_log_rmse=loss(best['score']))
                trace.append(item)
                if trust_radius < 1e-4:
                    stop_reason = 'invalid_finite_difference_at_minimum_radius'
                    break
                continue
            columns = [(np.asarray(self._calibration_residuals(scores[2*j+1])) -
                        np.asarray(self._calibration_residuals(scores[2*j]))) / deltas[j]
                       for j in range(len(axes))]
            jacobian = np.column_stack(columns)
            rank = analyze_identifiability(free_ids, jacobian)
            item.update(local_rank=rank['rank'], local_not_global=True,
                        singular_values=rank['singular_values'],
                        correlated_parameter_pairs=rank['correlated_pairs'])
            # SVD damped least squares avoids squaring the Jacobian condition
            # number. Bound/radius constraints are applied before true scoring.
            u, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
            scale = max(float(singular[0] ** 2), 1e-12)
            delta = -(vh.T @ ((singular / (singular ** 2 + damping * scale)) * (u.T @ residual)))
            maximum = float(np.max(np.abs(delta)))
            if maximum > trust_radius:
                delta *= trust_radius / maximum
            proposals, normalized_steps = [], []
            for alpha in (1., .5, .25):
                proposed = x.copy()
                proposed[axes] = np.clip(x[axes] + alpha * delta, 0., 1.)
                step = float(np.max(np.abs(proposed[axes] - x[axes])))
                point = decode(proposed)
                if step > 0 and point not in proposals:
                    proposals.append(point)
                    normalized_steps.append(step)
            if not proposals:
                item.update(reason='zero_projected_step', best_log_rmse=loss(best['score']))
                trace.append(item)
                stop_reason = 'zero_projected_step'
                break
            if self._score_call_cost(candidate, len(proposals), context) > self._remaining_calls(context):
                item.update(reason='line_search_budget_unavailable', best_log_rmse=loss(best['score']))
                trace.append(item)
                return best, report('before_residual_line_search_batch')
            try:
                scores = self.score(candidate, proposals, context)
            except BudgetExceeded:
                return best, report('residual_line_search_batch_budget_exhausted')
            starts[0]['evaluations'] += len(proposals)
            retain(proposals, scores)
            improved = loss(best['score']) < previous_loss
            item.update(accepted_improvement=improved, best_log_rmse=loss(best['score']),
                        maximum_proposed_step=max(normalized_steps), evaluations=starts[0]['evaluations'])
            trace.append(item)
            self.progress_callback(stage='parameter_fit', candidate=candidate['candidate_id'], fit_round=iteration+1,
                                   calibration_log_rmse=best['score']['log_rmse'], budget=context['ledger'].snapshot())
            if improved:
                damping = max(1e-8, damping / 3)
            else:
                trust_radius /= 2
                damping = min(1e12, damping * 10)
            if trust_radius < 1e-4 or max(normalized_steps) < 1e-4:
                stop_reason = 'local_step_tolerance'
                break
        else:
            stop_reason = 'iteration_limit'
        return best, report()

    def evaluator(self, context):
        def evaluate(candidate, *, method_config):
            if context.get('infrastructure_failure'):
                raise RuntimeError('Previous native infrastructure failure; stop this method')
            start_calls = context['ledger'].snapshot()['totals']['logical_calls']
            scoped_context = dict(context, ledger=ScopedLedger(context['ledger'], method_config['remaining_solver_calls']))
            directory = context['output'] / 'candidates' / candidate['candidate_id']
            directory.mkdir(parents=True, exist_ok=False)
            free = list(candidate['free_parameter_ids'])
            try:
                require_diagnostic = getattr(self, 'require_identifiability', False)
                selection_reserve = self._score_call_cost(candidate, 1, scoped_context, role='selection')
                # The initial fit builds the Dcal models. Reserve all four
                # perturbations per parameter for its two-step posterior rank
                # check, outside fitting and outside final delivery diagnostics.
                calibration_scenarios = sum(s['role'] == 'calibration' for s in self.case['scenarios'])
                identifiability_reserve = 4 * len(free) * calibration_scenarios if require_diagnostic else 0
                fitting_allowance = self._remaining_calls(scoped_context) - selection_reserve - identifiability_reserve
                if fitting_allowance < 0:
                    raise BudgetExceeded('search', selection_reserve + identifiability_reserve,
                                         self._remaining_calls(scoped_context))
                fit_context = dict(scoped_context, ledger=ScopedLedger(scoped_context['ledger'], fitting_allowance))
                prescreen = None
                if not require_diagnostic and method_config.get('identifiability_prescreen', True) and free:
                    # A prescreen cannot consume the one calibration incumbent
                    # whose selection prediction has already been reserved.
                    calibration_scenarios = sum(s['role'] == 'calibration' for s in self.case['scenarios'])
                    prescreen = self.bounded_diagnose(candidate, self.initial, free, fit_context,
                                                      reserve_calls=calibration_scenarios)
                    if prescreen.get('step_stability_passed'):
                        free = [name for name in free if name in prescreen['selected_parameter_ids']]
                best, fit = self.fit(candidate, free, fit_context)
                diagnostic = self.bounded_diagnose(candidate, best['point'], free,
                    scoped_context if require_diagnostic else fit_context,
                    reserve_calls=selection_reserve if require_diagnostic else 0)
                feedback_applied = False
                if (not require_diagnostic and method_config.get('identifiability_feedback', True) and not method_config.get('fixed_parameterization')
                        and diagnostic.get('step_stability_passed') and set(diagnostic['selected_parameter_ids']) != set(free)):
                    if self._score_call_cost(candidate, 1, fit_context) <= self._remaining_calls(fit_context):
                        free = [name for name in free if name in diagnostic['selected_parameter_ids']]
                        # Refit from common starts, never freeze old fitted values.
                        best, refit = self.fit(candidate, free, fit_context)
                        fit['identifiability_refit'] = refit
                        fit['budget_truncated'] |= refit['budget_truncated']
                        diagnostic = self.bounded_diagnose(candidate, best['point'], free, fit_context)
                        feedback_applied = True
                    else:
                        diagnostic['feedback_action'] = 'retain_current_parameterization_due_to_refit_budget'
                selection_cost = self._score_call_cost(candidate, 1, scoped_context, role='selection')
                if selection_cost > self._remaining_calls(scoped_context):
                    raise BudgetExceeded('search', selection_cost, self._remaining_calls(scoped_context))
                selection = self.score(candidate, [best['point']], scoped_context, role='selection')[0]
                status = 'completed' if best['score']['valid'] and selection['valid'] else 'failed'
                if status == 'completed' and fit['budget_truncated']:
                    status = 'budget_exhausted'
                result = {'status': status, 'candidate_id': candidate['candidate_id'],
                    'structure_id': candidate['structure_id'], 'structure_semantic_hash': candidate['structure_semantic_hash'],
                    'parameterization_hash': digest({'specs': self.specs, 'free': free}),
                    'effective_free_parameter_ids': free, 'fitted_values': best['point'],
                    'calibration_log_rmse': best['score']['log_rmse'], 'selection_log_rmse': selection['log_rmse'],
                    'calibration_scores': best['score'], 'selection_scores': selection,
                    'identifiability': diagnostic, 'prescreen': prescreen,
                    'identifiability_refit_performed': feedback_applied, 'fit': fit,
                    'prediction_available': best['score']['valid'] and selection['valid'],
                    'budget_scope': 'phase' if method_config.get('phase') == 'structure' else 'method',
                    'selection_solver_calls_reserved': selection_reserve,
                    'identifiability_solver_calls_reserved': identifiability_reserve,
                    'identifiability_required': require_diagnostic,
                    'solver_calls': context['ledger'].snapshot()['totals']['logical_calls'] - start_calls,
                    'method_options': method_config, 'independent_test_performed': False, 'scientific_accepted': False}
            except BudgetExceeded as exc:
                result = {'status': 'budget_exhausted', 'candidate_id': candidate['candidate_id'], 'error': str(exc),
                          'budget_scope': ('phase' if method_config.get('phase') == 'structure'
                                           else getattr(exc, 'budget_scope', 'method')),
                          'solver_calls': context['ledger'].snapshot()['totals']['logical_calls'] - start_calls}
            except Exception as exc:
                result = {'status': 'failed', 'candidate_id': candidate['candidate_id'],
                          'error_type': type(exc).__name__, 'error': str(exc), 'infrastructure_failure': True,
                          'solver_calls': context['ledger'].snapshot()['totals']['logical_calls'] - start_calls}
                context['infrastructure_failure'] = result
            save(directory / 'result.json', result)
            return result
        return evaluate

    def rebuild_selected(self, candidate, result, context):
        from .native_edits import compile_candidate, apply_parameter_values
        import csv
        value_file = context['output'] / 'selected_parameters.json'
        save(value_file, {'values': result['fitted_values'], 'parameter_ids': self.ids,
                         'calibration_source': self.case.get('case_id'), 'selection_only_used_for_model_choice': True,
                         'scientific_accepted': False})
        provenance = {'provenance_id': str(uuid5(NAMESPACE_URL, str(value_file) + file_sha(value_file))),
                      'provenance_kind': 'calibration_run', 'source_id': str(value_file.relative_to(self.root)),
                      'source_location': 'development calibration parameter values; Dsel used only to select the candidate',
                      'source_checksum': file_sha(value_file)}
        checks = []
        for scenario in self.case['scenarios']:
            key = (candidate['structure_id'], scenario['scenario_id'], tuple(result['fitted_values']))
            if key in self.rebuilt:
                context['ledger'].consume(phase='diagnostics', cache_hit=True)
                checks.append(dict(self.rebuilt[key], native_artifact_cache_hit=True))
                continue
            if candidate.get('native_mechanism_search'):
                from .mechanism_edits import compile_native_candidate
                rebuilt_variant = compile_native_candidate(scenario['design_request'], candidate['action'],
                    project_root=self.root, parameter_specs=self.specs,
                    fitted_values=result['fitted_values'], parameter_provenance=provenance)
            else:
                variant = compile_candidate(scenario['design_request'], candidate['action'], project_root=self.root, parameter_specs=self.specs)
                request = apply_parameter_values(variant['request'], self.specs, result['fitted_values'], provenance)
                rebuilt_variant = compile_candidate(request, candidate['action'], project_root=self.root, parameter_specs=self.specs)
            directory = self.output / 'selected_native_models' / digest(key)[:20]
            rebuilt = context['ledger'].run(lambda: execute_snapshot(snapshot=rebuilt_variant['snapshot'], output_dir=directory,
                    project_root=self.root, timeout_seconds=180, cpu_threads=self.cpu_threads, max_output_bytes=self.call_limit()), phase='diagnostics')
            if not rebuilt['engineering_accepted']:
                raise RuntimeError('Selected native rebuild failed: ' + str(directory))
            path = directory / rebuilt['curve_outputs'][0]['path']
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            column = scenario['target_output'] + ' [µmol/l]'
            run = {'vector_index': 0, 'time_min': [float(r['Time [min]']) for r in rows],
                   'concentration': [float(r[column]) for r in rows], 'time_unit': 'min', 'unit': 'µmol/l'}
            score = score_calibration_curves(runs=[run], observations=scenario['observations'], device=self.device)['rows'][0]
            expected = result[scenario['role']+'_scores']['scenarios'][scenario['scenario_id']]['predictions_umol_per_l']
            if not score['valid']:
                raise RuntimeError('Selected rebuilt prediction is invalid')
            difference = max(abs(math.log(a/b)) for a, b in zip(score['predictions_umol_per_l'], expected, strict=True))
            check = {'scenario_id': scenario['scenario_id'], 'maximum_absolute_log_difference': difference,
                     'accepted': difference <= 1e-4, 'tolerance': 1e-4, 'execution_manifest': str((directory/'execution_manifest.json').relative_to(self.root))}
            checks.append(check)
            save(directory / 'fit_rebuild_check.json', check)
            if not check['accepted']:
                raise RuntimeError('Selected snapshot does not reproduce the requested parameter-batch curve')
            self.rebuilt[key] = check
        return checks
