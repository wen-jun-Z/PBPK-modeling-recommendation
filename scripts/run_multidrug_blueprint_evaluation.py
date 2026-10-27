"""Build multiple drugs from the registered human base, then evaluate locked predictions.

The initial multi-drug cohort has fixed source parameters. B0/B1 are explicitly
declared native constitutive baselines, not renamed M0--M6 search experiments.
Data preparation exports outcomes separately; this prediction runner reads them
only after every prediction in this batch is locked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, UTC
import json
import math
import os
from pathlib import Path
import threading
import time
import zipfile

from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.experiment_runtime import digest, file_sha, save
from pbpk_kb.blueprints.native_edits import compile_candidate

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'B0': 'partition_pksim_standard', 'B1': 'partition_rodgers_rowland'}


def inside(relative, *, exists=True):
    path = (ROOT / relative).resolve(strict=exists)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Resource leaves the project')
    return path


def validate_case(case):
    if case.get('tier') not in {'bottom_up_external', 'reconstruction', 'development', 'main_holdout'}:
        raise ValueError('Each drug must have an explicit evaluation tier')
    if not case.get('case_id') or not case.get('compound_id'):
        raise ValueError('Case and compound identities are required')
    if case.get('parameter_search', {}).get('parameters'):
        raise ValueError('This fixed-parameter cohort does not silently fit unregistered parameters')
    scenarios = case['scenarios']
    if not scenarios or len({s['scenario_id'] for s in scenarios}) != len(scenarios):
        raise ValueError('Unique nonempty prediction scenarios are required')
    for source in case.get('source_audit', []):
        if file_sha(inside(source['path'])) != source['sha256']:
            raise ValueError('A locked drug source changed before prediction')
    for scenario in scenarios:
        # An outcome-bearing case cannot accidentally reach a designer.
        for key in ('observations', 'concentrations', 'predictions_umol_per_l'):
            if key in scenario:
                raise ValueError('External outcomes must be stored separately from prediction cases')
        reference = scenario['holdout_reference']
        path = inside(reference['path'])
        if path.is_symlink() or file_sha(path) != reference['sha256']:
            raise ValueError('External observation artifact changed')
        if not scenario.get('study_group_id') or not scenario.get('target_output'):
            raise ValueError('Study group and exact native output are required')


def read_curve(directory, manifest, target):
    outputs = manifest['curve_outputs']
    if len(outputs) != 1:
        raise ValueError('Expected one native curve export per single scenario')
    path = (directory / outputs[0]['path']).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Curve export leaves the native output directory')
    if file_sha(path) != outputs[0]['sha256']:
        raise ValueError('Native curve checksum differs from execution manifest')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        matches = [name for name in reader.fieldnames or []
                   if name.replace('μ', 'u').replace('µ', 'u') == target + ' [umol/l]']
        if len(matches) != 1:
            raise ValueError('Native plasma concentration column or unit is missing')
        rows = list(reader)
    times = [float(row['Time [min]']) for row in rows]
    values = [float(row[matches[0]]) for row in rows]
    if len(times) < 2 or any(not math.isfinite(x) for x in times + values):
        raise ValueError('Native prediction contains missing or nonfinite values')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Native prediction time grid is not increasing')
    return {'vector_index': 0, 'time_min': times, 'concentration': values,
            'time_unit': 'min', 'unit': 'umol/l'}


def load_observation(scenario, case):
    reference = scenario['holdout_reference']
    path = inside(reference['path'])
    if file_sha(path) != reference['sha256']:
        raise ValueError('Locked observation source changed before evaluation')
    record = json.loads(path.read_text())
    if record.get('scenario_id') != scenario['scenario_id'] or record.get('study_group_id') != scenario['study_group_id']:
        raise ValueError('Observation artifact scenario/study identity mismatch')
    if record.get('compound_id') != case['compound_id']:
        raise ValueError('Observation artifact drug identity mismatch')
    return {**record, 'tier': case['tier'], 'role': record.get('role', 'test'),
            'curve_id': record.get('curve_id', scenario['scenario_id'])}


def run(args):
    from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions

    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError('Use a new output directory inside the project')
    if not 1 <= args.workers <= 4 or not 1 <= args.cpu_threads <= 8 or not 256 <= args.max_output_mib <= 4096:
        raise ValueError('Unsupported bounded worker/core/output settings')
    output.mkdir(parents=True)
    (output/'runner.pid').write_text(str(os.getpid())+'\n')
    start = time.monotonic()
    mutex = threading.Lock()
    state = {'status': 'preparing', 'phase': 'multidrug_fixed_parameter_evaluation',
             'main_method_comparison_started': False, 'completed_native_jobs': 0,
             'failed_native_jobs': 0, 'completed_evaluations': 0}

    def update(**values):
        with mutex:
            state.update(values, elapsed_seconds=time.monotonic()-start)
            save(output/'progress.json', state, replace=True)
            print(json.dumps(state, ensure_ascii=False), flush=True)

    maximum = args.max_output_mib * 1024**2

    def disk_bytes():
        total = 0
        for path in output.rglob('*'):
            if path.is_symlink():
                raise ValueError('Output cannot contain symlinks')
            try:
                if path.is_file(): total += path.stat().st_size
            except FileNotFoundError:
                pass
        if total > maximum:
            raise RuntimeError('Batch output quota reached; stopping further execution')
        return total

    try:
        update(stage='source_preparation')
        input_path = inside(args.input_manifest)
        prepared = json.loads(input_path.read_text())
        if prepared.get('external_observations_in_design_payload') is not False:
            raise ValueError('Use a prepared cohort that explicitly separates external observations')
        cases = prepared['cases']
        if len({c['compound_id'] for c in cases}) < 2:
            raise ValueError('A multi-drug cohort requires at least two distinct compounds')
        if len({c['case_id'] for c in cases}) != len(cases):
            raise ValueError('Case IDs must be unique')
        for case in cases:
            validate_case(case)
            # Hash without decoding outcome files before prediction locking.
            for scenario in case['scenarios']:
                if not scenario['design_request'].get('human_model'):
                    raise ValueError('Each scenario must bind the standard human base')
        # No selection criterion uses the external observations: both baselines
        # are registered before either native predictions or outcomes exist.
        jobs = [(case, method, scenario) for case in cases for method in VARIANTS
                for scenario in case['scenarios']]
        code_paths = sorted({'scripts/run_multidrug_blueprint_evaluation.py',
                            *(str(p.relative_to(ROOT)) for p in (ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
                            'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
                            'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R'})
        with zipfile.ZipFile(output/'runner_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for relative in code_paths: archive.write(ROOT/relative, arcname=relative)
        configuration = {
            'schema_version': 'multidrug-native-evaluation-1',
            'input_manifest_path':str(input_path.relative_to(ROOT)),
            'input_manifest_sha256':file_sha(input_path),
            'drug_count': len({c['compound_id'] for c in cases}),
            'case_ids': [c['case_id'] for c in cases],
            'tiers': {c['case_id']: c['tier'] for c in cases},
            'registered_methods': VARIANTS, 'primary_baseline': 'B0',
            'planned_native_jobs': len(jobs), 'native_backend': 'CPU_CVODES',
            'gpu_native_solver': False, 'scoring_device': args.device,
            'workers': args.workers, 'native_process_environment_threads': args.cpu_threads,
            'timeout_seconds_per_scenario': 180, 'maximum_output_bytes': maximum,
            'parameter_fitting': False, 'm0_m6_comparison': False,
            'outcomes_available_to_prediction_builder': False,
            'test_based_candidate_selection': False, 'anatomical_topology_changed': False,
            'source_code_sha256': {p: file_sha(ROOT/p) for p in code_paths},
            'case_sha256': {c['case_id']: digest(c) for c in cases},
            'limitations': ['Tier-specific external/reconstruction results; not the completed 60-drug main benchmark.',
                            'Constitutive baselines do not establish joint-search or LLM benefit.',
                            'Native execution does not establish mass balance or clinical qualification.']}
        save(output/'configuration.json', configuration)
        save(output/'cohort.json', prepared)
        update(status='running', stage='native_predictions', drug_count=configuration['drug_count'],
               planned_native_jobs=len(jobs), tiers=configuration['tiers'])

        def execute(job):
            case, method, scenario = job
            directory = output/'native_models'/digest([case['case_id'],method,scenario['scenario_id']])[:24]
            available = min(128*1024**2, (maximum-disk_bytes()-8*1024**2)//args.workers)
            if available < 1024**2:
                raise RuntimeError('Insufficient remaining native output budget')
            variant = compile_candidate(scenario['design_request'], VARIANTS[method], project_root=ROOT,
                                        parameter_specs=[])
            manifest = execute_snapshot(snapshot=variant['snapshot'], output_dir=directory,
                        project_root=ROOT, timeout_seconds=180, cpu_threads=args.cpu_threads,
                        max_output_bytes=available)
            save(directory/'edit_manifest.json', variant['edit_manifest'])
            accepted = bool(manifest.get('engineering_accepted'))
            curve = read_curve(directory, manifest, scenario['target_output']) if accepted else None
            result = {'case_id':case['case_id'], 'compound_id':case['compound_id'], 'method':method,
                      'scenario_id':scenario['scenario_id'], 'study_group_id':scenario['study_group_id'],
                      'tier':case['tier'], 'status':'completed' if accepted else 'failed',
                      'run':curve, 'native_manifest':str((directory/'execution_manifest.json').relative_to(ROOT)),
                      'structure_semantic_hash':variant['semantic_hash'],
                      'parameterization_hash':variant['parameterization_hash']}
            save(directory/'prediction.json',result)
            return result

        predictions = []
        # Submit a bounded number of jobs; any native failure stops new work.
        pending = iter(jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = {pool.submit(execute, job): job for job in [next(pending) for _ in range(min(args.workers,len(jobs)))]}
            while active:
                future = next(as_completed(active))
                job = active.pop(future)
                result = future.result()
                predictions.append(result)
                update(completed_native_jobs=len(predictions),
                       failed_native_jobs=sum(p['status']!='completed' for p in predictions),
                       last_compound=result['compound_id'], last_method=result['method'],
                       last_scenario=result['scenario_id'], output_bytes=disk_bytes())
                if result['status'] != 'completed':
                    raise RuntimeError('Native failure: '+result['native_manifest'])
                next_job = next(pending, None)
                if next_job is not None: active[pool.submit(execute,next_job)] = next_job
        # Commit every curve before reading any external outcome values.
        save(output/'predictions.json', predictions)
        locks = []
        for case in cases:
            for method in VARIANTS:
                curves = {p['scenario_id']:p['run'] for p in predictions if p['case_id']==case['case_id'] and p['method']==method}
                locked = {'method':method,'seed':17,'candidate_id':VARIANTS[method],
                          'predictions_sha256':lock_predictions(curves),'status':'completed'}
                locks.append({'case_id':case['case_id'], 'selection':locked})
        save(output/'prediction_locks.json', {'predictions_file_sha256':file_sha(output/'predictions.json'),
             'locked_at_utc':datetime.now(UTC).isoformat(), 'locks':locks,
             'all_native_predictions_finished_before_runner_observation_read':True,
             'data_preparation_reads_observations_separately':True})
        update(stage='locked_external_evaluation')
        summaries=[]
        for case in cases:
            scenarios=[load_observation(s,case) for s in case['scenarios']]
            for method in VARIANTS:
                locked=next(x['selection'] for x in locks if x['case_id']==case['case_id'] and x['selection']['method']==method)
                curves={p['scenario_id']:p['run'] for p in predictions if p['case_id']==case['case_id'] and p['method']==method}
                report=evaluate_locked_predictions(scenarios,curves,locked,device=args.device)
                directory=output/'evaluations'/digest([case['case_id'],method])[:20]
                save(directory/'results.json',report)
                summaries.append({'case_id':case['case_id'],'compound_id':case['compound_id'],
                    'tier':case['tier'],'method':method,'report':report})
                save(output/'results.json',{'evaluations':summaries,'main_benchmark_completed':False},replace=True)
                update(completed_evaluations=len(summaries))
        update(status='completed',stage='finished',output_bytes=disk_bytes())
        save(output/'completion.json',{'status':'completed','elapsed_seconds':time.monotonic()-start,
             'drug_count':configuration['drug_count'],'native_jobs_completed':len(predictions),
             'evaluations_completed':len(summaries),'m0_m6_comparison_completed':False,
             'main_benchmark_completed':False,'scientific_accepted':False})
        return 0
    except BaseException as error:
        update(status='failed',stage='stopped',error_type=type(error).__name__,error=str(error))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--max-output-mib',type=int,default=1024)
    return run(parser.parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
