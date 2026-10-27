#!/usr/bin/env python3
"""Isolated tail-refinement batch, using the frozen runner without modifying it."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = 'tail-blueprint-refinement-1'
DEFAULT_NAME = 'robust_structure_tail_20260930'
SOURCE_NAME = 'robust_structure_20260930'
METHODS = ('current', 'tail_balanced', 'development_guarded')
ENGINE_NAMES = {'current': 'pbpk_kb.blueprint_policy.robust_exchange_calibration',
    'tail_balanced': 'pbpk_kb.blueprint_policy.tail_exchange_calibration',
    'development_guarded': 'pbpk_kb.blueprint_policy.tail_exchange_calibration'}
VARIANTS = {'current': 'evidence_flexible', 'tail_balanced': 'tail_balanced',
            'development_guarded': 'development_guarded'}
CODE_FILES = ('scripts/run_tail_blueprint.py', 'scripts/run_robust_blueprint.py',
    'scripts/run_blueprint_capability.py', 'scripts/run_blueprint_evidence_policy.py',
    'scripts/run_blueprint_learning.py', 'scripts/run_blueprint_transport.py',
    'scripts/report_blueprint_validation.py')


def create_runner(*, project_root=ROOT, source_name=SOURCE_NAME, default_name=DEFAULT_NAME,
                  engine_provider=None, controller_class=None):
    """Each instance owns all runner globals, including its formula selector.

    The shared frozen module and its api namespace remain untouched. Only this
    new adapter is configured; original execution/reporting function bodies are
    loaded verbatim into a private namespace and included in the source pins.
    """
    root = Path(project_root).resolve()
    for name in (source_name, default_name):
        if Path(name).name != name or not name.startswith('robust_structure_'):
            raise ValueError('Use simple dedicated robust_structure_ directory names')
    if source_name == default_name or not default_name.startswith('robust_structure_tail_'):
        raise ValueError('A distinct tail-refinement output directory is required')
    source_file = ROOT / 'scripts/run_robust_blueprint.py'
    spec = importlib.util.spec_from_file_location('_tail_frozen_runner_' + uuid.uuid4().hex, source_file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    source_version = runner.VERSION
    source = root / 'experiments/blueprint_design' / source_name
    default = source.parent / default_name
    # A fresh api namespace is essential: the frozen module normally shares it
    # through run_blueprint_capability, so mutating that object would leak.
    api = runner.previous.transport_runner.create_runner(project_root=root)
    original_signature, original_load, original_report, original_status = runner.signature, runner.load, runner.report, runner.status
    runner.ROOT, runner.DEFAULT, runner.VERSION = root, default, VERSION
    runner.BASE = root / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
    runner.CAPABILITY = root / 'experiments/blueprint_design/blueprint_capability_20260930'
    runner.METHODS, runner.ENGINE_NAMES, runner.api = METHODS, deepcopy(ENGINE_NAMES), api
    # The frozen background launcher uses __file__. It must spawn this adapter.
    runner.__file__ = str(root / 'scripts/run_tail_blueprint.py')
    for name in ('read', 'ref', 'checked', 'verify'):
        setattr(runner, name, getattr(api, name))
    if controller_class is not None:
        runner.Controller = controller_class
    runner.engine = engine_provider or (lambda method: importlib.import_module(ENGINE_NAMES[method]))
    runner.variant = lambda method: VARIANTS[method]
    binding = {'version': VERSION, 'source_version': source_version,
        'source_run': source_name, 'default_run': default_name,
        'methods': list(METHODS), 'engine_modules': deepcopy(ENGINE_NAMES), 'variant_by_method': deepcopy(VARIANTS),
        'baseline': {'logical_method': 'current', 'source_method': 'evidence_flexible',
            'interpretation': 'current means the previous batch evidence_flexible; it does not mean original flow_exchange.'},
        'formula_selection': {'tail_balanced': 'eligible full development objective',
            'development_guarded': 'eligible formula_selection_score from internal development-held-out predictions; exact score ties prefer stronger selected CL prior, then formula name'},
        'frozen_execution_source': 'scripts/run_robust_blueprint.py'}

    def own_run(path):
        raw = Path(path) if Path(path).is_absolute() else root / path
        result = inside(raw, root)
        if raw.is_symlink() or result.parent != default.parent or not result.name.startswith('robust_structure_tail_'):
            raise ValueError('Use a dedicated immediate robust_structure_tail_ output directory')
        return result

    def signature(request, target, training, method):
        return original_signature(request, target, training, VARIANTS[method])

    def choose_formula(mappings):
        if (len(mappings) != len(runner.FORMULAS)
                or {m['construction_partition_method'] for m in mappings} != set(runner.FORMULAS)
                or len({m['variant'] for m in mappings}) != 1):
            raise ValueError('Both formulas for one variant must finish before selection')
        eligible = [m for m in mappings if m.get('eligible')]
        def score(mapping):
            value = (mapping.get('formula_selection_score') if mapping['variant'] == 'development_guarded'
                     else mapping['decision']['objective'])
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Eligible formula lacks a finite predeclared development selection score')
            if mapping['variant'] == 'development_guarded':
                weight = mapping['clearance_prior_policy']['penalty_weight']
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError('Guarded formula requires a positive finite selected clearance penalty')
                return value, -weight, mapping['construction_partition_method']
            return value, mapping['construction_partition_method']
        return min(eligible, key=score) if eligible else None

    runner.own_run, runner.signature = own_run, signature
    api.choose_formula = choose_formula

    def store_input(run, document):
        path = run / 'inputs' / (runner.inputs.digest(document) + '.json')
        if path.exists():
            if runner.read(path) != document:
                raise ValueError('Immutable visible input changed')
        else:
            save(path, document)
        return {**runner.ref(path), 'location': 'Frozen visible input'}

    def read_source():
        state = runner.read(source / 'state.json')
        protocol = runner.checked(state['protocol'])
        if (state['version'] != source_version or protocol['version'] != source_version
                or state['status'] != 'complete' or not state.get('all_attempts_locked_before_scoring')
                or not state.get('monitor_stopped_before_scoring')):
            raise ValueError('The prior robust batch must be complete and locked')
        runner.verify(protocol['source_pins']); runner.verify(protocol['paper_guard'])
        if runner.paper_pins() != protocol['paper_guard']:
            raise ValueError('Publication files or their file set changed')
        rows = [u for u in protocol['units'] if u['method'] == 'evidence_flexible']
        expected = {(c['case_id'], s) for c in protocol['cases'] for s in runner.SCENARIOS}
        actual = [(u['case_id'], u['scenario']) for u in rows]
        if len(protocol['cases']) != 10 or len(actual) != 40 or set(actual) != expected or any(u['sampling'] != 'full' for u in rows):
            raise ValueError('The complete ten-drug four-scenario evidence_flexible source is required')
        return protocol, state, rows

    def prepare(run, workers=14):
        run = own_run(run)
        if run.exists():
            raise FileExistsError('Preserve existing run; use resume')
        if type(workers) is not int or not 1 <= workers <= 14:
            raise ValueError('Use 1–14 workers')
        previous, previous_state, baseline_units = read_source()
        pins = dict(previous['source_pins'])
        api.pin_refs([previous, previous_state], pins)
        for path in (source / 'protocol.json', source / 'state.json'):
            pins[str(path.relative_to(root))] = sha256(path)
        contracts = {method: {'module': ENGINE_NAMES[method], 'version': runner.engine(method).VERSION,
            'config': deepcopy(runner.engine(method).CONFIG), 'code_pins': runner.engine(method)._code_pins(root)}
            for method in METHODS}
        for contract in contracts.values():
            pins.update(contract['code_pins'])
        cached, visible = {}, {}
        for unit in baseline_units:
            cid = unit['computation_id']
            if cid in cached:
                continue
            computation = deepcopy(previous_state['computations'][cid])
            if computation['status'] != 'predicted':
                raise ValueError('Every frozen evidence_flexible control must have a locked prediction')
            api.pin_refs(computation, pins)
            mapping = runner.checked(computation['mapping'])
            mapping_path = inside(computation['mapping']['path'], root, exists=True)
            runner.engine('current')._validated(mapping, mapping_path, root)
            if mapping['variant'] != 'evidence_flexible':
                raise ValueError('The baseline is evidence_flexible, not another previous method')
            for name in ('source_pins', 'code_pins', 'retained_artifact_pins'):
                runner.verify(mapping[name]); pins.update(mapping[name])
            fixed = runner.read(mapping_path.parent / 'input.json')
            request = fixed['request']; target = runner.checked(computation['request'])
            training = runner.checked(mapping['training_source'])
            if signature(request, target, training, 'current') != cid:
                raise ValueError('Frozen baseline scientific signature does not match its inputs')
            visible[cid] = request, target, runner.inputs.clean_training(training)
            computation.update(reused_from=runner.ref(source / 'state.json'), source_method='evidence_flexible',
                original_resources=deepcopy(computation.get('resources', {})),
                resources={'total_native_calls': 0, 'linear_solver_calls': 0, 'wall_seconds': 0, 'cache_reused': True})
            cached[cid] = computation
        if len(cached) != 20:
            raise ValueError('Expected twenty unique evidence_flexible control predictions')
        # All source/engine contracts are checked before creating the new batch.
        runner.verify(pins)
        run.mkdir(); (run / 'inputs').mkdir()
        specs, units = {}, []
        for baseline in baseline_units:
            request, target, training = visible[baseline['computation_id']]
            for method in METHODS:
                cid = signature(request, target, training, method)
                units.append({**{k: baseline[k] for k in ('case_id', 'scenario', 'sampling', 'scientific_inputs_changed')},
                    'id': '__'.join((baseline['case_id'], baseline['scenario'], method)),
                    'method': method, 'computation_id': cid})
                if method != 'current' and cid not in specs:
                    specs[cid] = {'case_id': baseline['case_id'], 'sampling': 'full', 'method': method,
                        'request': store_input(run, request), 'target_request': store_input(run, target),
                        'training_source': {**store_input(run, training), 'role': 'calibration_data_reconstruction'}}
        if len(units) != 120 or len(specs) != 40 or set(specs) & set(cached):
            raise ValueError('Expected 120 logical units, forty new and twenty reused scientific computations')
        for name in CODE_FILES:
            pins[name] = sha256(root / name)
        for value in (specs, cached, previous['cases'], previous['input_audit']):
            api.pin_refs(value, pins)
        budget = {'full_fit_stages': len(specs) * len(runner.FORMULAS), 'cv_stages': 0,
                  'full_optimizer_starts': 0, 'cv_optimizer_starts': 0}
        for item in specs.values():
            module = runner.engine(item['method']); config = module.CONFIG
            budget['full_optimizer_starts'] += len(runner.FORMULAS) * config['starts']
            if item['method'] == 'development_guarded':
                policy = module.clearance_policy(runner.checked(item['request']))
                weights = config['cv']['rf_penalty_weights' if policy['rule'] == 'qualified_rf' else 'observed_penalty_weights']
                count = len(runner.FORMULAS) * len(config['cv']['folds']) * len(weights)
                budget['cv_stages'] += count
                budget['cv_optimizer_starts'] += count * config['cv_starts']
        budget['optimizer_starts'] = budget['full_optimizer_starts'] + budget['cv_optimizer_starts']
        protocol = {k: deepcopy(previous[k]) for k in ('cases', 'sampling', 'scenarios', 'formulas',
            'input_audit', 'changed_case_ids', 'native_threads', 'native_timeout_seconds', 'monitor_seconds',
            'directory_limit_bytes', 'stability', 'adoption', 'bootstrap', 'limitations')}
        protocol.update(version=VERSION, created_at=runner.now(), methods=list(METHODS), units=units,
            computations=specs, cached_ids=sorted(cached), contracts=contracts, source_pins=pins,
            paper_guard=deepcopy(previous['paper_guard']), source_protocol=runner.ref(source / 'protocol.json'),
            source_state=runner.ref(source / 'state.json'), workers=workers, runner_binding=deepcopy(binding),
            optimization_budget=budget,
            scope='Previously inspected ten-drug refinement. current is the frozen prior evidence_flexible; new candidates use development data only. No blind validation or paper updates.',
            formula_selection_rules=deepcopy(binding['formula_selection']))
        runner.verify(pins)
        if runner.paper_pins() != protocol['paper_guard']:
            raise ValueError('Publication files changed during preparation')
        save(run / 'protocol.json', protocol)
        save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': runner.ref(run / 'protocol.json'),
            'created_at': runner.now(), 'fits': {}, 'computations': cached, 'monitor': {}})
        return {'unique_new_fits': 80, 'new_predictions': 40, 'cached_predictions': 20, 'logical_units': 120,
            'baseline_source_method': 'evidence_flexible', 'paper_guard_files': len(protocol['paper_guard'])}

    def load(run, locked=False):
        protocol, state = original_load(run, locked=locked)
        if protocol.get('runner_binding') != binding:
            raise ValueError('The isolated runner configuration changed')
        if runner.paper_pins() != protocol['paper_guard']:
            raise ValueError('Publication files or their file set changed')
        source_state = runner.checked(protocol['source_state'])
        for cid in protocol['cached_ids']:
            cached, original = state['computations'][cid], source_state['computations'][cid]
            for name in ('status', 'mapping', 'request', 'prediction', 'completion', 'dense_endpoints', 'selected_formula'):
                if cached.get(name) != original.get(name):
                    raise ValueError('A reused baseline prediction binding changed')
            if cached.get('source_method') != 'evidence_flexible':
                raise ValueError('A cached baseline lost its semantic method identity')
        if locked:
            if not state.get('formulas_locked_before_target_prediction'):
                raise ValueError('Formula selection must precede target prediction')
            selection = runner.checked(state['formula_selection'])
            for cid in protocol['computations']:
                mappings = [runner.checked(state['fits'][runner.key(cid, f)]['mapping']) for f in runner.FORMULAS]
                best = choose_formula(mappings)
                expected = None if best is None else {'formula': best['construction_partition_method'],
                    'mapping': state['fits'][runner.key(cid, best['construction_partition_method'])]['mapping']}
                if selection['choices'][cid] != expected:
                    raise ValueError('Locked formula does not minimize its predeclared development score')
                computation = state['computations'][cid]
                if expected is not None and computation.get('status') == 'predicted' and computation.get('mapping') != expected['mapping']:
                    raise ValueError('Target prediction does not use the locked selected mapping')
        return protocol, state

    def report(run):
        result = original_report(run)
        if result.get('baseline_semantics') is None:
            result['baseline_semantics'] = deepcopy(binding['baseline'])
            result['formula_selection_rules'] = deepcopy(binding['formula_selection'])
            save(run / 'results.json', result)
        return result

    def status(run):
        result = original_status(run)
        protocol = runner.read(run / 'protocol.json')
        counts = Counter(); active = Counter()
        for cid in protocol['computations']:
            for formula in runner.FORMULAS:
                folder = run / 'work' / cid / formula
                input_path, progress_path = folder / 'input.json', folder / 'progress.json'
                if not input_path.exists():
                    continue
                fixed = runner.read(input_path)
                if fixed.get('variant') not in METHODS[1:]:
                    raise ValueError('Unexpected variant in tail batch progress')
                counts['initialized_fits'] += 1
                if not progress_path.exists():
                    continue
                progress = runner.read(progress_path)
                for name, stage in progress.get('stages', {}).items():
                    kind = 'full' if name == 'full' else 'cv'
                    counts[kind + '_optimizer_starts_completed'] += len(stage.get('starts', []))
                    counts[kind + '_stages_completed'] += int(stage.get('status') == 'complete')
                if progress.get('incomplete_start'):
                    active[progress['incomplete_start']['stage']] += 1
                if progress.get('status') == 'locked' and 'full' not in progress.get('stages', {}):
                    counts['full_stages_skipped_after_cv_failure'] += 1
        result['optimization_progress'] = {key: counts[key] for key in ('initialized_fits',
            'full_stages_completed', 'cv_stages_completed', 'full_optimizer_starts_completed',
            'cv_optimizer_starts_completed', 'full_stages_skipped_after_cv_failure')}
        result['optimization_progress'].update(planned=protocol['optimization_budget'], active_stages=dict(active),
            note='Counts refer only to this batch; planned full stages/starts are upper bounds if development CV fails.')
        return result

    runner.prepare, runner.load, runner.report, runner.status = prepare, load, report, status
    runner.runner_binding = deepcopy(binding)
    return runner


def main():
    runner = create_runner()
    # This frozen main resolves only globals in this instance, including the
    # new preparation/selection adapters and the new __file__ background entry.
    runner.main()


if __name__ == '__main__':
    main()
