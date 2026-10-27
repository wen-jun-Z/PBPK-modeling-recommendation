"""Bounded development calibration of a freshly generated human PBPK instance.

Native PK-Sim/OSPSuite solves on CPU. PyTorch scores actual native curves on the
requested device. This is parameter feedback on a historical development case,
not the M6 structural algorithm or independent paper evaluation.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, UTC
import hashlib
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid5, NAMESPACE_URL

from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.design_diagnostics import BudgetLedger, analyze_identifiability
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.parameter_batch import run_parameter_batch
from pbpk_kb.blueprints.design_scoring import score_calibration_curves
from pbpk_kb.blueprints.development_case import make_caffeine_development_case

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value, *, progress=False):
    if progress:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _project_file(path, root):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Development input must be a file inside this project')
    return resolved


def _artifact_file(directory, relative, root):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Artifact path must be relative to its output directory')
    resolved = _project_file(directory / path, root)
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError('Artifact path leaves its output directory')
    return resolved


def _validated_case(saved, root):
    """Reaudit sources; only two missing, derivable legacy annotations are allowed."""
    expected = make_caffeine_development_case(root)
    normalized = deepcopy(saved)
    annotations = []
    parameters = normalized.get('parameter_search', {}).get('parameters', [])
    for parameter in parameters:
        if not isinstance(parameter, dict):
            raise ValueError('Development parameter specification must be an object')
        scales = {'total_cyp1a2_specific_clearance': 'linear_nonnegative',
                  'specific_intestinal_permeability_transcellular': 'log10'}
        parameter_id = parameter.get('parameter_id')
        if 'search_scale' not in parameter and parameter_id in scales:
            parameter['search_scale'] = scales[parameter_id]
            annotations.append(parameter_id + '.search_scale')
    if _canonical(normalized) != _canonical(expected):
        raise ValueError('Saved case differs from the freshly source-verified development case')
    return expected, annotations


def _validated_baseline_snapshot(case_dir, case, baseline, root):
    source = _project_file(case_dir / 'baseline/snapshot.json', root)
    if sha(source) != baseline['source_sha256']:
        raise ValueError('Baseline snapshot checksum drift')
    design = design_blueprint(BlueprintDesignRequest.model_validate(case['design_request']), project_root=root)
    if not design['candidates'] or design['candidates'][0]['status'] != 'draft_assembled':
        raise ValueError('Source-verified development request has unresolved static gaps')
    expected = design['human_model_binding']['instance']['snapshot']
    if _canonical(json.loads(source.read_text())) != _canonical(expected):
        raise ValueError('Baseline snapshot differs from the source-verified development request')
    return sha(source)


class _OutputBudget:
    """Account for this feedback output only; reserve space for final audit metadata.

    Native workers poll their per-call thresholds, so this is not an OS quota.
    Every call is capped by the remaining aggregate allowance and checked again
    before further writes or work. Existing baseline/probe artifacts are inputs.
    """
    def __init__(self, directory, maximum, metadata_reserve=1024**2):
        self.directory = directory.resolve()
        self.maximum = maximum
        self.metadata_reserve = metadata_reserve

    def size(self):
        total = 0
        for parent, directories, files in os.walk(self.directory, followlinks=False):
            for name in directories + files:
                if (Path(parent) / name).is_symlink():
                    raise ValueError('Feedback output contains a symlink')
            total += sum((Path(parent) / name).stat().st_size for name in files)
        return total

    def check(self):
        used = self.size()
        if used > self.maximum:
            raise RuntimeError('Development output byte limit reached')
        return used

    def call_limit(self, per_call_limit):
        allowance = min(per_call_limit, self.maximum - self.check() - self.metadata_reserve)
        if allowance < 1024**2:
            raise RuntimeError('Insufficient remaining development output budget for a native call')
        return allowance

    def write(self, path, value, *, progress=False):
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError('Feedback artifact leaves its output directory')
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        # A progress replacement temporarily keeps both its old and new content.
        if self.check() + len(encoded) > self.maximum:
            raise RuntimeError('Development output byte limit prevents artifact write')
        write(path, value, progress=progress)
        self.check()


def _loss(score):
    value = score.get('log_rmse')
    if (score.get('valid') is not True or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)):
        return math.inf
    return value


def _best_candidate(candidates):
    valid = [candidate for candidate in candidates if math.isfinite(_loss(candidate['score']))]
    return min(valid, key=lambda s: (_loss(s['score']), s.get('source') != 'baseline', s['point'])) if valid else None


def _best_log_rmse(candidates):
    best = _best_candidate(candidates)
    return None if best is None else best['score']['log_rmse']


def _validate_points(points, specs):
    for point in points:
        if len(point) != len(specs):
            raise ValueError('Parameter vector differs from the locked specification')
        for value, spec in zip(point, specs, strict=True):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not spec['lower_bound'] <= value <= spec['upper_bound']):
                raise ValueError('Parameter value exceeds the finite locked development bounds')


def apply_parameters(request, specs, values, provenance):
    """Only the two registered development drug-parameter mappings are editable."""
    result = deepcopy(request)
    compound = result['instance']['compound']
    for spec, value in zip(specs, values, strict=True):
        if not spec['lower_bound'] <= value <= spec['upper_bound']:
            raise ValueError('Estimated parameter exceeds the locked development bounds')
        if spec['parameter_id'] == 'total_cyp1a2_specific_clearance':
            for mapping in spec['runtime_mappings']:
                prefix = compound['Name'] + '-CYP1A2-'
                path = mapping['runtime_path']
                if not path.startswith(prefix) or not path.endswith('|Specific clearance'):
                    raise ValueError('Unsupported CYP1A2 native mapping')
                source = path.removeprefix(prefix).removesuffix('|Specific clearance')
                matches = [p for p in compound['Processes'] if p.get('Molecule') == 'CYP1A2' and p['DataSource'] == source]
                if len(matches) != 1:
                    raise ValueError('Native process mapping is not unique')
                parameters = [p for p in matches[0]['Parameters'] if p['Name'] == 'Specific clearance']
                if len(parameters) != 1 or parameters[0]['Unit'] != spec['unit']:
                    raise ValueError('Native specific clearance unit differs from fit input')
                parameters[0]['Value'] = value * mapping['multiplier']
            result['instance']['compound_sources']['Processes'] = provenance
        elif spec['parameter_id'] == 'specific_intestinal_permeability_transcellular':
            if spec['unit'] != 'dm/min':
                raise ValueError('Unsupported permeability fit unit')
            parameter = compound['IntestinalPermeability'][0]['Parameters'][0]
            if parameter['Name'] != 'Specific intestinal permeability (transcellular)' or parameter['Unit'] != 'cm/s':
                raise ValueError('Native permeability mapping drift')
            parameter['Value'] = value / 6.0
            result['instance']['compound_sources']['IntestinalPermeability'] = provenance
        else:
            raise ValueError('No reviewed snapshot adapter for parameter ' + spec['parameter_id'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 12 or not 1 <= args.cpu_threads <= 8:
        raise ValueError('Use 1..12 search rounds and 1..8 native CPU threads')
    case_dir, output = args.case_directory.resolve(strict=True), args.output.resolve()
    for path in (case_dir, output):
        if not path.is_relative_to(ROOT) or path == ROOT:
            raise ValueError('Development paths must stay inside this project')
    if output.exists():
        raise FileExistsError('Feedback output exists; preserve prior results')
    case_path = _project_file(case_dir / 'case.json', ROOT)
    case_bytes = case_path.read_bytes()
    case_checksum = hashlib.sha256(case_bytes).hexdigest()
    case, compatible_annotations = _validated_case(json.loads(case_bytes), ROOT)
    if case['purpose'] != 'historical_development_reconstruction' or case['calibration']['role'] != 'calibration':
        raise ValueError('This runner accepts the explicit development/calibration case only')
    specs = case['parameter_search']['parameters']
    if len(specs) != 2 or any(p['unit'] not in {'1/min', 'dm/min'} for p in specs):
        raise ValueError('Expected the reviewed two-parameter development case')
    baseline = json.loads(_project_file(case_dir / 'baseline/execution_manifest.json', ROOT).read_text())
    if not baseline['engineering_accepted'] or len(baseline['pkml_outputs']) != 1:
        raise ValueError('A new native baseline must first pass engineering checks')
    snapshot_checksum = _validated_baseline_snapshot(case_dir, case, baseline, ROOT)
    pkml = _artifact_file(case_dir / 'baseline', baseline['pkml_outputs'][0]['path'], ROOT)
    if sha(pkml) != baseline['pkml_outputs'][0]['sha256']:
        raise ValueError('Baseline PKML checksum drift')
    mapping = json.loads(_project_file(case_dir / 'mapping_probe/parameter_batch_manifest.json', ROOT).read_text())
    probe_path = _project_file(case_dir / 'mapping_probe/native_results.json', ROOT)
    if mapping['status'] != 'completed' or sha(probe_path) != mapping['native_results_sha256'] or mapping['pkml_sha256'] != sha(pkml):
        raise ValueError('Native parameter mapping must be verified against this baseline')
    probe = json.loads(probe_path.read_text())
    native_specs = [{'path': m['runtime_path'], 'unit': p['unit']} for p in specs for m in p['runtime_mappings']]
    initial = [p['initial_value'] for p in specs]
    def native_values(point):
        return [point[i] * m['multiplier'] for i, p in enumerate(specs) for m in p['runtime_mappings']]
    if (mapping['parameter_specs'] != native_specs or len(probe['runs']) != 1
            or mapping['requested_vector_count'] != 1
            or probe['runs'][0]['values'] != native_values(initial)
            or [mapping['target_output']] != case['design_request']['instance']['output_selections']):
        raise ValueError('Probe does not match the explicit development start')
    output.mkdir(parents=True)
    started = time.monotonic()
    ledger = BudgetLedger(250, diagnostic_fraction=0.2)
    ledger.consume(2)  # Newly executed baseline and native mapping probe in this case.
    config = {'case_sha256': case_checksum, 'baseline_pkml_sha256': sha(pkml),
              'baseline_snapshot_sha256': snapshot_checksum,
              'input_validation': {'sources_reverified': True, 'observations_checksum_verified': True,
                                   'baseline_request_matched': True, 'legacy_annotations_added': compatible_annotations},
              'runner_sha256': sha(Path(__file__)),
              'scoring_code_sha256': sha(ROOT/'src/pbpk_kb/blueprints/design_scoring.py'),
              'batch_code_sha256': sha(ROOT/'src/pbpk_kb/blueprints/parameter_batch.py'),
              'purpose': case['purpose'], 'device': args.device, 'precision': 'float64',
              'method': 'bounded_multistart_pattern_search_for_development_calibration',
              'round_limit': args.rounds, 'starts': 3, 'max_evaluations_per_start': 50,
              'cpu_threads': args.cpu_threads, 'logical_solver_budget': 250,
              'diagnostic_fraction': 0.2, 'selection_data_loaded': False, 'test_data_loaded': False,
              'native_backend': 'OSPSuite CPU', 'whole_output_limit_bytes': 512 * 1024**2,
              'whole_output_scope': 'feedback_output_directory_only',
              'native_output_limit_enforcement': 'polled_threshold_with_aggregate_remaining_allowance'}
    disk = _OutputBudget(output, config['whole_output_limit_bytes'])
    persist = disk.write
    persist(output / 'configuration.json', config)
    progress = {'status': 'running', 'stage': 'calibration', 'started_at': datetime.now(UTC).isoformat(),
                'scientific_status': 'development_only', 'gpu_device': args.device}
    def update(**items):
        progress.update(items, elapsed_seconds=time.monotonic()-started, budget=ledger.snapshot())
        persist(output / 'progress.json', progress, progress=True)
        print(json.dumps({k: progress.get(k) for k in ('status','stage','round','best_calibration_log_rmse','elapsed_seconds')}, ensure_ascii=False), flush=True)
    cache = {}
    batches = 0
    trace = []
    observations = case['calibration']['observations']
    target = mapping['target_output']
    try:
        scored = score_calibration_curves(runs=probe['runs'], observations=observations, device=args.device)
        baseline_score = scored['rows'][0]
        if not math.isfinite(_loss(baseline_score)):
            raise RuntimeError('Native baseline does not predict positive values at calibration observations')
        cache[tuple(initial)] = baseline_score
        baseline_candidate = {'point': initial, 'score': baseline_score, 'source': 'baseline'}
        persist(output / 'baseline_score.json', scored)
        def evaluate(points, phase='search'):
            nonlocal batches
            _validate_points(points, specs)
            pending = {}
            for point in points:
                key = tuple(point)
                if key in cache or key in pending:
                    ledger.consume(phase=phase, cache_hit=True)
                else:
                    pending[key] = point
            if pending:
                rows = list(pending.values())
                batches += 1
                batch_dir = output / ('batch_%03d' % batches)
                output_limit = disk.call_limit(128 * 1024**2)
                def execute():
                    result = run_parameter_batch(pkml_path=pkml, parameter_specs=native_specs,
                        values=[native_values(point) for point in rows], target_output=target,
                        output_dir=batch_dir, project_root=ROOT, timeout_seconds=180,
                        cpu_threads=args.cpu_threads, max_output_bytes=output_limit)
                    if result['status'] != 'completed':
                        raise RuntimeError('Native batch failed; see ' + str(batch_dir / 'parameter_batch_manifest.json'))
                    return result
                result = ledger.run(execute, scenario_calls=len(rows), phase=phase)
                disk.check()
                scores = score_calibration_curves(runs=result['runs'], observations=observations, device=args.device)
                for point, row in zip(rows, scores['rows'], strict=True):
                    cache[tuple(point)] = row
                persist(batch_dir / 'calibration_scores.json', scores)
            results = [cache[tuple(point)] for point in points]
            for point, score in zip(points, results, strict=True):
                trace.append({'phase': phase, 'parameters': point, 'valid': score['valid'],
                              'log_rmse': score['log_rmse']})
            disk.check()
            return results
        def encode(point):
            return [(x-p['lower_bound'])/(p['upper_bound']-p['lower_bound']) if p['lower_bound']==0 else
                    (math.log10(x)-math.log10(p['lower_bound']))/math.log10(p['upper_bound']/p['lower_bound'])
                    for x,p in zip(point,specs,strict=True)]
        def decode(point):
            return [p['lower_bound']+x*(p['upper_bound']-p['lower_bound']) if p['lower_bound']==0 else
                    10**(math.log10(p['lower_bound'])+x*math.log10(p['upper_bound']/p['lower_bound']))
                    for x,p in zip(point,specs,strict=True)]
        starts = case['parameter_search']['clearance_starts']
        seed_values = [starts[i]['total_cyp1a2_specific_clearance'] for i in (0,len(starts)//2,len(starts)-1)]
        points = [[value, initial[1]] for value in seed_values]
        scores = evaluate(points)
        states = [{'x': encode(point), 'point': point, 'score': score, 'step': .25, 'evaluations': 1, 'source': 'search'}
                  for point,score in zip(points,scores,strict=True)]
        update(round=0, best_calibration_log_rmse=_best_log_rmse([baseline_candidate, *states]),
               search_has_valid_candidate=_best_candidate(states) is not None)
        for iteration in range(args.rounds):
            proposals, owners = [], []
            for i,state in enumerate(states):
                for axis in range(2):
                    for sign in (-1,1):
                        if state['evaluations'] >= 50:
                            continue
                        point = list(state['x'])
                        point[axis] = min(1.,max(0.,point[axis]+sign*state['step']))
                        proposals.append(decode(point)); owners.append((i,point));state['evaluations'] += 1
            if not proposals:
                break
            values = evaluate(proposals)
            improved = set()
            for point,score,(index,position) in zip(proposals,values,owners,strict=True):
                if _loss(score) < _loss(states[index]['score']):
                    states[index].update(x=position,point=point,score=score);improved.add(index)
            for i,state in enumerate(states):
                if i not in improved:
                    state['step'] /= 2
            update(round=iteration+1,best_calibration_log_rmse=_best_log_rmse([baseline_candidate, *states]),
                   search_has_valid_candidate=_best_candidate(states) is not None)
        best = _best_candidate([baseline_candidate, *states])
        if best is None:
            raise RuntimeError('No positive native calibration curve survived')
        # This is a calibration minimum, never a Dsel-selected structure.
        fit = {'parameter_ids':[p['parameter_id'] for p in specs], 'values':best['point'],
               'units':[p['unit'] for p in specs],
               'origin': 'retained_development_initial_values' if best['source'] == 'baseline' else 'estimated_from_development_calibration',
               'selected_candidate_source': best['source'],
               'calibration_checksum':case['calibration']['checksum'], 'case_sha256':case_checksum,
               'parent_sources':[p['source'] for p in specs], 'calibration_score':best['score'],
               'study_selection_performed':False, 'independent_validation_performed':False,
               'starts':states, 'optimizer_convergence_proven':False}
        persist(output / 'fitted_parameters.json', fit)
        persist(output / 'search_trace.json', trace)
        update(stage='local_identifiability')
        pairs=[];deltas=[]
        for axis,spec in enumerate(specs):
            value=best['point'][axis]
            if value<=0:
                deltas=[];break
            minus=max(spec['lower_bound'],value*math.exp(-.001))
            plus=min(spec['upper_bound'],value*math.exp(.001))
            if minus<=0 or plus<=minus:
                deltas=[];break
            low=list(best['point']);high=list(best['point']);low[axis]=minus;high[axis]=plus
            pairs += [low,high];deltas.append(math.log(plus/minus))
        if deltas:
            perturbations=evaluate(pairs,phase='diagnostics')
            if all(row['valid'] for row in perturbations):
                jacobian=[[ (math.log(perturbations[2*j+1]['predictions_umol_per_l'][i])-math.log(perturbations[2*j]['predictions_umol_per_l'][i]))/deltas[j]
                           for j in range(2)] for i in range(len(observations))]
                diagnostic=analyze_identifiability([p['parameter_id'] for p in specs],jacobian)
                diagnostic.update(observation_source_verified=True,calibration_checksum=case['calibration']['checksum'],
                                  scaling='d log predicted concentration / d log parameter; unweighted observed sampling times',
                                  difference_log_step=.001, step_stability_checked=False)
            else:
                diagnostic={'status':'not_evaluable','reason':'invalid_perturbed_prediction'}
        else:
            diagnostic={'status':'not_evaluable','reason':'parameter_boundary_prevents_log_perturbation'}
        persist(output / 'local_identifiability.json',diagnostic)
        fit_path=output/'fitted_parameters.json'
        provenance={'provenance_id':str(uuid5(NAMESPACE_URL,str(fit_path)+sha(fit_path))),
                    'provenance_kind':'calibration_run','source_id':str(fit_path.relative_to(ROOT)),
                    'source_location': 'development calibration candidate selection: ' + fit['origin'],
                    'source_checksum':sha(fit_path)}
        fitted_request=apply_parameters(case['design_request'],specs,best['point'],provenance)
        fitted_design=design_blueprint(BlueprintDesignRequest.model_validate(fitted_request),project_root=ROOT)
        if fitted_design['candidates'][0]['status']!='draft_assembled':
            raise RuntimeError('Fitted parameter draft has unresolved static gaps')
        persist(output / 'fitted_design.json',fitted_design)
        update(stage='rebuild_fitted_model')
        rebuild_limit = disk.call_limit(256 * 1024**2)
        rebuilt=ledger.run(lambda:execute_snapshot(snapshot=fitted_design['human_model_binding']['instance']['snapshot'],
                    output_dir=output/'fitted_model',project_root=ROOT,timeout_seconds=180,cpu_threads=args.cpu_threads,
                    max_output_bytes=rebuild_limit),phase='diagnostics')
        disk.check()
        if not rebuilt['engineering_accepted']:
            raise RuntimeError('Fitted native rebuild failed; see fitted_model/execution_manifest.json')
        csv_path=_artifact_file(output/'fitted_model', rebuilt['curve_outputs'][0]['path'], ROOT)
        with csv_path.open(encoding='utf-8-sig',newline='') as handle:
            native_rows=list(csv.DictReader(handle))
        concentration_column=target+' [µmol/l]'
        if not native_rows or concentration_column not in native_rows[0]:
            raise RuntimeError('Rebuilt observer or native concentration unit changed')
        rebuilt_run={'vector_index':0,'time_min':[float(r['Time [min]']) for r in native_rows],
                     'concentration':[float(r[concentration_column]) for r in native_rows],
                     'time_unit':'min','unit':'µmol/l'}
        rebuilt_scores=score_calibration_curves(runs=[rebuilt_run],observations=observations,device=args.device)
        checked=rebuilt_scores['rows'][0]
        if not checked['valid']:
            raise RuntimeError('Rebuilt fitted model gives invalid calibration predictions')
        disagreement=max(abs(math.log(a/b)) for a,b in zip(checked['predictions_umol_per_l'],
                              best['score']['predictions_umol_per_l'],strict=True))
        agreement={'maximum_absolute_log_difference':disagreement,'tolerance':1e-4,
                   'accepted':disagreement<=1e-4,'tolerance_reason':'CSV export rounding and deterministic native reconstruction'}
        persist(output/'fitted_rebuild_check.json',{'agreement':agreement,'scores':rebuilt_scores})
        if not agreement['accepted']:
            raise RuntimeError('Fitted snapshot does not reproduce the fitted native parameter-batch prediction')
        summary={'status':'completed','purpose':case['purpose'],'method':config['method'],
                 'gpu_scoring':scored['gpu_used'],'scoring_device':scored['device'],
                 'native_solver_backend':'CPU_CVODES','calibration_points':len(observations),
                 'baseline_calibration_log_rmse':baseline_score['log_rmse'],
                 'fitted_calibration_log_rmse':best['score']['log_rmse'],
                 'selected_candidate_source':best['source'],
                 'native_engineering_accepted':True,'source_engine_equivalence':'not_verified',
                 'fitted_rebuild_verified':True,
                 'independent_validation_performed':False,'structure_search_performed':False,
                 'mass_balance_verified':False,'scientific_accepted':False,
                 'elapsed_seconds':time.monotonic()-started,'budget':ledger.snapshot(),
                 'limitations':case['limitations']}
        persist(output/'result.json',summary)
        update(status='completed',stage='completed')
        disk.check()
        return 0
    except BaseException as exc:
        try:
            update(status='failed',error=type(exc).__name__+': '+str(exc))
        except (OSError, RuntimeError, ValueError) as audit_error:
            exc.add_note('Could not persist bounded failure progress: ' + str(audit_error))
        raise


if __name__=='__main__':
    raise SystemExit(main())
