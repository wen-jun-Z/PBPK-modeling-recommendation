"""Versioned, isolated calibration for nested physical-axis search.

The native model, objective, bounds and visible development inputs are unchanged
from axes_calibration. Only the optimizer is replaced. Old experiment modules
and artifacts remain immutable. No prior fit or target outcome initializes a fit.
"""
from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
import shutil
import time

from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, _digest
from pbpk_kb.blueprints.development_curve_blueprint import _training_document
from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature
from pbpk_kb.blueprints.evidence_preserving_calibration import _exclusive, InfrastructurePause
from pbpk_kb.blueprint_learning.native_calibration import _embedded_pins, _verify
from . import axes_calibration as original
from .axes_calibration import _read, _ref, _artifacts, _base, normalize_curve, score_curve
from . import refined_axes_optimizer as axes_optimizer

VERSION = 'native-distribution-calibration-nested-2'
CLEARANCE_POLICY = original.CLEARANCE_POLICY
CONFIG = json.loads(json.dumps({'optimizer': axes_optimizer.CONFIG, 'native_parameters_changed': False,
    'loss_changed': False, 'bounds_changed': False, 'base_included_in_budget': True,
    'external_warm_start_used': False,
    'initialization': 'Same visible development subset; no old fit or target observations',
    'nested_incumbent_guarantee': 'Only within this run; not the old independent 48-attempt two-axis optimum'}, allow_nan=False))


def _code_pins(root):
    pins = original._code_pins(root)
    for path in (Path(__file__), Path(axes_optimizer.__file__)):
        pins[str(inside(path, root, exists=True).relative_to(root))] = sha256(path)
    return pins


def _validate_locked(mapping, path, root):
    fixed = _read(path.parent/'input.json')
    if (mapping.get('version') != VERSION or mapping.get('input_sha256') != sha256(path.parent/'input.json')
            or mapping['source_pins'] != fixed['source_pins'] or mapping['code_pins'] != fixed['code_pins']
            or mapping['active_parameters'] != fixed['active_parameters']
            or mapping.get('optimizer_config') != CONFIG or fixed.get('optimizer_config') != CONFIG
            or mapping['drug_signature'] != _drug_signature(fixed['request'])
            or mapping['construction_partition_method'] != fixed['request']['partition_method']
            or mapping['training_source'] != fixed['training_source']
            or mapping['clearance_policy'] != fixed['clearance_policy']
            or mapping['total_native_calls'] > fixed['maximum_evaluations']):
        raise ValueError('Locked calibration input or policy changed')
    _verify(mapping['source_pins'], root); _verify(mapping['code_pins'], root)
    _verify(mapping['retained_artifact_pins'], root)
    for row in mapping['evaluations']:
        if _read(path.parent/'trials'/row['cache_key']/'trial.json') != row:
            raise ValueError('Locked calibration trial changed')
    best = min((r for r in mapping['evaluations'] if r.get('eligible_for_selection')),
        key=lambda r: (r['objective'], r['evaluation']), default=None)
    if mapping['decision'] != best or mapping['eligible'] != (best is not None):
        raise ValueError('Locked decision differs from actual development trials')


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                active_parameters=axes_optimizer.PARAMETERS, max_evaluations=48, before_trial=None):
    from .native_axes_runtime import run_axes_trial
    root = Path(project_root).resolve(strict=True); output = inside(output_dir, root)
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 2:
        raise ValueError('Budget includes one base solve and at least one trial')
    active = tuple(active_parameters)
    if set(active) != set(axes_optimizer.PARAMETERS) or len(active) != 3:
        raise ValueError('Refinement requires all three predefined physical scales')
    active = tuple(a for a in axes_optimizer.PARAMETERS if a in active)
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode='json', exclude_none=True)
    observed = _training_document(training_source, req, root)
    if (req['dosing']['route'] != 'intravenous' or observed['human'] != req['human']
            or observed['dosing'] != req['dosing']):
        raise ValueError('Development study conditions differ from request')
    pins = _embedded_pins([req, training_source, observed], root); code = _code_pins(root)
    _verify(pins, root); _verify(code, root)
    fixed = {'version': VERSION, 'request': req, 'training_source': deepcopy(training_source),
        'active_parameters': list(active), 'maximum_evaluations': int(max_evaluations),
        'source_pins': pins, 'code_pins': code, 'clearance_policy': CLEARANCE_POLICY,
        'bounds': axes_optimizer.BOUNDS, 'warm_start_used': False,
        'external_warm_start_used': False,
        'initialization': 'Nested two-axis fit from this same visible development subset only',
        'optimizer_config': CONFIG,
        'objective': 'mean(ln(P/O)^2)+0.1*(ln(CLscale)/ln2)^2+0.01*(ln(Kscale)/ln10)^2+0.01*(ln(Pscale)/ln100)^2',
        'native_budget_includes_base': True, 'inactive_parameters_fixed_at': 1.}
    # JSON canon avoids tuple/list drift when resuming.
    fixed = json.loads(json.dumps(fixed, allow_nan=False))
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        input_path = output/'input.json'; mapping_path = output/'mapping_locked.json'
        if input_path.exists():
            if _read(input_path) != fixed:
                raise ValueError('Resume sources, code, budget or parameters changed')
        else:
            if any(p.name != '.calibration.lock' for p in output.iterdir()):
                raise ValueError('Unrecognized existing calibration directory')
            save(input_path, fixed)
        if mapping_path.exists():
            result = _read(mapping_path); _validate_locked(result, mapping_path, root)
            return {**result, 'mapping': _ref(mapping_path, root), 'actual_new_calls': 0}
        started = time.monotonic()
        new_base_calls = int(not (output/'base'/'completion.json').exists())
        base = _base(req, output/'base', root, before_trial)
        trials = output/'trials'; trials.mkdir(exist_ok=True)
        base_ref = base['execution_manifest']
        new_calls = 0

        def attempts():
            return 1 + sum(1 for _ in trials.glob('*/execution_*'))

        def evaluate(parameters, purpose):
            nonlocal new_calls
            if before_trial:
                before_trial()
            key = _digest({'input_sha256': sha256(input_path), 'parameters': parameters})
            folder = trials/key; record = folder/'trial.json'
            if record.exists():
                row = _read(record)
                if row['parameters'] != parameters:
                    raise ValueError('Cached candidate differs from optimizer')
                _verify(row['artifact_pins'], root)
                return row
            if attempts() >= max_evaluations:
                raise axes_optimizer.BudgetExhausted()
            folder.mkdir(exist_ok=True)
            executions = sorted(folder.glob('execution_*'))
            if len(executions) >= 2:
                raise InfrastructurePause('Repeated interrupted candidate; inspect diagnostics')
            execution = folder/f'execution_{len(executions)}'
            # This attempt is charged even if native pre-solve validation rejects it.
            result = run_axes_trial(execution_manifest_path=root/base_ref['path'],
                source_manifest_sha256=base_ref['sha256'], scales=parameters,
                output_dir=execution, project_root=root, timeout_seconds=180, cpu_threads=2)
            new_calls += 1
            if result.get('status') == 'infrastructure_failed':
                save(output/'progress.json', {'status': 'paused', 'total_native_calls': attempts(),
                    'reason': result.get('failure', result.get('error', result.get('reason', 'Native infrastructure failed')))})
                raise InfrastructurePause('Native infrastructure failed; completed candidates remain cached')
            row = {'evaluation': 1+sum(1 for _ in trials.glob('*/trial.json')),
                'parameters': deepcopy(parameters), 'purpose': purpose, 'cache_key': key,
                'native_output_dir': str(execution.relative_to(root)),
                'actual_native_solver_calls': result.get('actual_native_solver_calls', 0),
                'status': 'scientific_failed', 'eligible_for_selection': False,
                'reason': result.get('failure', result.get('error', result.get('reason', result.get('status'))))}
            if result.get('status') == 'predicted':
                try:
                    curve = normalize_curve(result['curve'])
                    row.update(score_curve(curve, observed, parameters))
                except (ValueError, ArithmeticError) as exc:
                    row.update(status='scientific_failed', eligible_for_selection=False, reason=str(exc))
            row['artifact_pins'] = _artifacts(execution, root)
            save(record, row)
            save(output/'progress.json', {'status': 'running', 'total_native_calls': attempts(),
                'maximum_native_calls': max_evaluations, 'completed_trials': row['evaluation'],
                'last_status': row['status'], 'actual_new_calls': new_calls})
            return row

        values = [r['concentration_umol_per_l'] for r in observed['observations']]
        result = axes_optimizer.optimize(evaluate, values, active_parameters=active,
            max_evaluations=max_evaluations-1)
        _verify(pins, root); _verify(code, root)
        retained = dict(base['artifact_pins'])
        retained[str((output/'base'/'completion.json').relative_to(root))] = sha256(output/'base'/'completion.json')
        for row in result['evaluations']:
            record = trials/row['cache_key']/'trial.json'
            retained[str(record.relative_to(root))] = sha256(record)
            if row == result['decision'] or row['status'] != 'ok':
                retained.update(row['artifact_pins'])
        result.update(version=VERSION, input_sha256=sha256(input_path), source_pins=pins, code_pins=code,
            drug_signature=_drug_signature(req), construction_partition_method=req['partition_method'],
            compound_id=req['compound_id'], training_source=deepcopy(training_source),
            reference_human=req['human'], reference_dosing=req['dosing'], clearance_policy=CLEARANCE_POLICY,
            retained_artifact_pins=retained, active_parameters=list(active), total_native_calls=attempts(),
            actual_solver_calls=1+sum(_read(p)['actual_native_solver_calls']
                for p in trials.glob('*/execution_*/solver_receipt.json')),
            native_budget_includes_base=True, base_solver_calls=1, elapsed_seconds=time.monotonic()-started,
            actual_new_calls=new_calls+new_base_calls, development_curve_used=True, comparison_curve_used=False,
            parameter_selection='Fixed three-axis model; nested initialization uses only the same visible development subset',
            optimizer_config=CONFIG,
            limitations=['Previously viewed development cases; not blinded validation.',
                'Fixed physiological and clearance inputs do not establish true patient parameters.',
                'Numerical convergence and local data rank do not establish global identifiability.',
                'Distribution scaling has no automatic tissue-exposure validation.'])
        save(mapping_path, result)
        save(output/'progress.json', {'status': 'locked', 'total_native_calls': attempts(),
            'maximum_native_calls': max_evaluations, 'eligible': result['eligible'],
            'actual_solver_calls': result['actual_solver_calls']})
        return {**result, 'mapping': _ref(mapping_path, root), 'mapping_path': str(mapping_path.relative_to(root))}


def predict_mapping(*, request, mapping_path, output_dir, project_root, before_trial=None):
    from .native_axes_runtime import run_axes_trial
    from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
    root = Path(project_root).resolve(strict=True); output = inside(output_dir, root)
    path = inside(mapping_path, root, exists=True); mapping = _read(path)
    _validate_locked(mapping, path, root)
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
    if not mapping['eligible'] or _drug_signature(req) != mapping['drug_signature']:
        raise ValueError('Prediction request drug evidence or formula differs from locked mapping')
    _training_document(mapping['training_source'], req, root)
    pins = _embedded_pins(req, root); _verify(pins, root)
    construction = {'version': VERSION, 'mapping': _ref(path, root), 'request': req,
        'parameters': mapping['decision']['parameters'], 'target_outcomes_used': False}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        receipt = output/'completion.json'
        if receipt.exists():
            saved = _read(receipt)
            if saved['construction'] != construction:
                raise ValueError('Existing target prediction uses different inputs')
            _verify(saved['artifact_pins'], root)
            return {**_read(output/'result.json'), 'actual_new_calls': 0}
        if any(p.name != '.calibration.lock' for p in output.iterdir()):
            raise InfrastructurePause('Partial target attempt retained; choose an explicit retry directory')
        save(output/'request.json', req); save(output/'construction.json', construction)
        started = time.monotonic()
        base = _base(req, output/'base', root, before_trial); source = base['execution_manifest']
        if before_trial:
            before_trial()
        native = run_axes_trial(execution_manifest_path=root/source['path'], source_manifest_sha256=source['sha256'],
            scales=mapping['decision']['parameters'], output_dir=output/'execution', project_root=root,
            timeout_seconds=180, cpu_threads=2)
        if native['status'] == 'infrastructure_failed':
            raise InfrastructurePause('Target native worker infrastructure failure')
        result = {'status': 'failed', 'native_status': native['status'], 'actual_new_calls': 2,
            'actual_native_solver_calls': 1+native.get('actual_native_solver_calls', 0),
            'elapsed_seconds': time.monotonic()-started, 'reason': native.get('failure', native.get('reason', native.get('error'))),
            'parameters_fitted_to_development_only': True, 'comparison_curve_used': False, 'mass_balance_checked': False}
        if native['status'] == 'predicted':
            curve = normalize_curve(native['curve'])
            save(output/'prediction_locked.json', curve)
            endpoints = summarize_extended_prediction(curve, req)
            endpoints.update(development_curve_used=True, comparison_observations_used=False,
                endpoint_estimator_uses_observations=False, parameters_fitted=True)
            save(output/'endpoints.json', endpoints)
            result.update(status='predicted', prediction=_ref(output/'prediction_locked.json', root),
                prediction_sha256=sha256(output/'prediction_locked.json'))
        _verify(pins, root); _validate_locked(mapping, path, root)
        save(output/'result.json', result)
        save(receipt, {'construction': construction, 'artifact_pins': _artifacts(output, root)})
        return result


def cleanup_unselected(*, output_dir, project_root):
    root = Path(project_root).resolve(strict=True); folder = inside(output_dir, root, exists=True)
    with _exclusive(folder):
        path = folder/'mapping_locked.json'; mapping = _read(path)
        _validate_locked(mapping, path, root)
        selected = (mapping['decision'] or {}).get('cache_key'); removed = []
        for row in mapping['evaluations']:
            if row['status'] != 'ok' or row['cache_key'] == selected:
                continue
            candidate = inside(row['native_output_dir'], root)
            if (candidate.parent != folder/'trials'/row['cache_key']
                    or not candidate.name.startswith('execution_') or candidate.is_symlink()):
                raise ValueError('Unsafe candidate cleanup path')
            if not candidate.exists():
                continue
            if any(p.is_symlink() for p in candidate.rglob('*')):
                raise ValueError('Refuse symlinked candidate cleanup')
            _verify(row['artifact_pins'], root)
            if any((root/p).is_relative_to(candidate) for p in mapping['retained_artifact_pins']):
                raise ValueError('Candidate contains a retained artifact')
            shutil.rmtree(candidate); removed.append(str(candidate.relative_to(root)))
        return {'removed_count': len(removed), 'removed_native_directories': removed,
            'compact_scores_preserved': len(mapping['evaluations']), 'failed_trials_preserved': True}
