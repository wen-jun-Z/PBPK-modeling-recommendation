#!/usr/bin/env python3
"""Freeze Furosemide external predictions without opening final-study outcomes.

Four method/budget blocks retain separate denominators. Reference subjects,
food and formulation approximations cannot establish population validation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.experiment_runtime import digest, file_sha, save
from pbpk_kb.blueprints.mechanism_edits import compile_native_candidate
from pbpk_kb.blueprints.external_source_isolation import audit_design_inputs

TOTAL_OUTPUT_BYTES = 2 * 1024**3
NATIVE_OUTPUT_BYTES = 192 * 1024**2
OUTPUT_HEADROOM = 16 * 1024**2
MIN_NATIVE_BYTES = 16 * 1024**2
WORKERS = 2
BLOCKS = {
    'main_5000': (5000, None, ('M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0')),
    'budget_500': (500, None, ('M3','M4','M5','M6')),
    'budget_2000': (2000, None, ('M3','M4','M5','M6')),
    'matched_1': (5000, 1, ('M5','M6')),
}


def local(value, *, exists=True):
    p = (ROOT/value).resolve(strict=exists)
    if p == ROOT or not p.is_relative_to(ROOT):
        raise ValueError('External validation paths must remain in project')
    return p


def read(path, frozen=None):
    path = local(path)
    if path.stat().st_size > 16*1024**2:
        raise ValueError('Unexpectedly large metadata file')
    data = path.read_bytes()
    import hashlib
    checksum = hashlib.sha256(data).hexdigest()
    if frozen is not None:
        frozen[str(path.relative_to(ROOT))] = checksum
    return json.loads(data)


def verify_frozen(frozen):
    for path, checksum in frozen.items():
        if file_sha(local(path)) != checksum:
            raise ValueError('Frozen input changed: '+path)


def ref(path, location, kind):
    checksum = file_sha(local(path))
    return {'provenance_id':str(uuid5(NAMESPACE_URL,str(path)+location+checksum)),
        'provenance_kind':kind,'source_id':str(path.relative_to(ROOT)),
        'source_location':location,'source_checksum':checksum}


def metadata_exclusion_policy(design, design_path):
    """Use curator-registered hashes; never open raw or sealed PK outcomes."""
    ex = design['evidence_exclusion']
    if (design.get('scientific_results_accessed_by_algorithm') is not False or
        ex.get('exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training') is not True):
        raise ValueError('Whole-publication outcome isolation must be explicit')
    paths = list(ex.get('raw_source_files', [])) + [design['sealed_observation_reference']['path']]
    denied_paths = [str(local(p, exists=False).relative_to(ROOT)).lower() for p in paths]
    hashes = [design['source_article_sha256'], design['sealed_observation_reference']['sha256']]
    if any(not re.fullmatch(r'[a-f0-9]{64}', h) for h in hashes):
        raise ValueError('Curated outcome/source checksums required')
    identifiers = [(k,str(v).lower()) for k,v in ex['identifiers'].items() if v]
    identifiers += [('trial',str(v).lower()) for v in ex.get('trial_registry_ids', [])]
    rel, checksum = str(design_path.relative_to(ROOT)), file_sha(design_path)
    return {'schema_version':'pbpk-external-source-isolation-v1',
        'protocols':[{'case_id':design['case_id'],'study_id':design['study_id'],
            'cohort_id':design['cohort_id'],'path':rel,'sha256':checksum}],
        'denied_paths':sorted(set(denied_paths)),'denied_hashes':sorted(set(hashes)),
        'denied_identifiers':[{'kind':k,'value':v} for k,v in sorted(set(identifiers))],
        'curated_metadata_sources':{rel:checksum},'external_outcomes_read':False,
        'checksum_policy':'Curator-registered article/sealed hashes; excluded raw batch paths are never opened here',
        'scope':'Supplied inputs, not unpublished model ancestry or pretraining'}


def expected_jobs(methods):
    return {(m,s) for m in methods for s in ((17,) if m in {'M0','M1','E0'} else (17,43,101))}


def job_set(rows):
    keys = [(r['method'],r['seed']) for r in rows]
    if len(set(keys)) != len(keys) or any(type(s) is not int for _,s in keys):
        raise ValueError('Duplicate or invalid registered method jobs')
    return set(keys)


def load_selections(suite, case, package_path, frozen):
    """Authenticate each registered block without inspecting held-out scores."""
    completed = read(suite/'completion.json', frozen)
    if completed.get('status') != 'finished' or completed.get('halt_reason'):
        raise RuntimeError('The upstream method suite stopped; no silent continuation')
    config = read(suite/'configuration.json', frozen)
    if config.get('version') != 'blueprint-reviewed-development-suite-v1':
        raise ValueError('Unknown upstream suite contract')
    case_id = case['case_id']
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', case_id):
        raise ValueError('Unsafe case identifier')
    tasks = [t for t in config['tasks'] if t['case_id'] == case_id]
    if len(tasks) != 4 or {t['block_id'] for t in tasks} != set(BLOCKS):
        raise ValueError('Expected exactly four frozen blocks for this case')
    package_sha = file_sha(package_path)
    specs = case['parameter_search']['parameters']
    parameter_ids = [p['parameter_id'] for p in specs]
    actions = set(case['mechanism_search']['expected_structure_ids'])
    selections, models = [], {}
    for task in sorted(tasks,key=lambda t:t['block_id']):
        block = task['block_id']
        budget, dimension, methods = BLOCKS[block]
        jobs = expected_jobs(methods)
        if (task['task_id'] != block+'__'+case_id or task['budget'] != budget or
            task['matched_count'] != dimension or task['compound_id'] != case['compound_id'] or
            local(task['bundle_path']) != package_path or task['bundle_sha256'] != package_sha or
            job_set(task['jobs']) != jobs):
            raise ValueError('Registered task identity, budget or parameter dimension mismatch')
        directory = local(suite/'tasks'/task['task_id'])
        tc = read(directory/'configuration.json', frozen)
        task_case = read(directory/'case.json', frozen)
        if (digest(task_case) != digest(case) or tc['case_sha256'] != file_sha(directory/'case.json') or
            tc['case_bundle_sha256'] != package_sha or local(tc['case_bundle_path']) != package_path or
            tc['compound_id'] != case['compound_id'] or tc['max_solver_calls_per_method'] != budget or
            tc['exact_free_parameter_count'] != dimension or job_set(tc['jobs']) != jobs or
            tc.get('external_observations_available_to_optimizer') is not False or
            tc.get('independent_test_loaded') is not False):
            raise ValueError('Frozen task configuration does not match the registered case/block')
        for source, checksum in tc['code_sha256'].items():
            if config['code_sha256'].get(source) != checksum or file_sha(local(source)) != checksum:
                raise ValueError('Prediction compiler/runner no longer matches upstream frozen code')
            frozen[source] = checksum
        candidates = {c['candidate_id']:c for c in tc['candidates']}
        if len(candidates) != len(tc['candidates']):
            raise ValueError('Duplicate candidate IDs')
        result = read(directory/'results.json', frozen)
        if job_set(result['methods']) != jobs or result['completed_jobs'] != len(jobs):
            raise ValueError('Missing or extra method result; retain the registered denominator')
        for row in sorted(result['methods'],key=lambda r:(r['method'],r['seed'])):
            identity = f"{row['method']}_seed_{row['seed']}"
            selection = {'selection_id':block+'__'+identity,'block_id':block,
                'solver_budget':budget,'exact_free_parameter_count':dimension,
                'method':row['method'],'seed':row['seed'],'model_key':None,
                'upstream_status':row['status'],'prediction_available':False,
                'upstream_result_path':str((directory/'results.json').relative_to(ROOT))}
            available, verified = row.get('prediction_available'), row.get('selected_rebuild_verified')
            if type(available) is not bool or type(verified) is not bool or available != verified:
                raise ValueError('Inconsistent upstream prediction/rebuild flags')
            if available:
                if row['status'] not in {'completed','budget_exhausted'}:
                    raise ValueError('Available prediction has an invalid upstream status')
                candidate = candidates.get(row['selected_candidate_id'])
                if not candidate or candidate['action'] not in actions or candidate['action'] != candidate['structure_id']:
                    raise ValueError('Selected candidate is absent from the registered mechanism space')
                free = candidate['free_parameter_ids']
                effective = row['effective_free_parameter_ids']
                if (candidate['parameter_ids'] != parameter_ids or len(set(free)) != len(free) or
                    not set(free) <= set(parameter_ids) or effective is None or
                    not set(effective) <= set(free) or len(set(effective)) != len(effective) or
                    candidate['parameterization_hash'] != digest({'specs':specs,'free':free}) or
                    (dimension is not None and (len(free) != dimension or len(effective) != dimension))):
                    raise ValueError('Selected parameter contract/dimension mismatch')
                path = directory/'methods'/identity/'selected_parameters.json'
                params = read(path, frozen)
                values = params['values']
                if (params['parameter_ids'] != parameter_ids or params['calibration_source'] != case_id or
                    params.get('scientific_accepted') is not False or len(values) != len(specs) or
                    any(type(v) not in (float,int) or not math.isfinite(v) or
                        not spec['lower_bound'] <= v <= spec['upper_bound']
                        for v,spec in zip(values,specs))):
                    raise ValueError('Invalid or foreign fitted parameter vector')
                for spec,value in zip(specs,values):
                    if spec['parameter_id'] not in effective and value != spec['initial_value']:
                        raise ValueError('An unestimated parameter changed from its registered initial value')
                key = digest({'action':candidate['action'],'values':values,
                              'parameter_contract':specs,'case_id':case_id})[:24]
                models.setdefault(key,{'action':candidate['action'],'values':values,
                    'parameters_path':str(path.relative_to(ROOT)),
                    'parameter_contract_sha256':digest(specs)})
                selection.update(model_key=key,prediction_available=True,
                    selected_candidate_id=candidate['candidate_id'],effective_free_parameter_ids=effective,
                    parameters_path=str(path.relative_to(ROOT)),parameters_sha256=file_sha(path))
            else:
                selection['failure'] = row.get('blocked_reason') or 'No verified native prediction submitted upstream'
            selections.append(selection)
    if len(selections) != 57:
        raise ValueError('All 57 registered method jobs must retain their denominator')
    return selections, models


def validate_design(design):
    protocol = design['protocol']
    demo = protocol['reported_demographics']
    if (design.get('compound') != 'furosemide' or design.get('scientific_results_accessed_by_algorithm') is not False or
        protocol['route'] != 'oral' or protocol['dose'] != 40 or protocol['dose_unit'] != 'mg' or
        protocol['administrations'] != [{'time_min':0,'dose':40,'dose_unit':'mg'}] or
        demo['age_year']['mean'] != 39 or demo['body_weight_kg']['mean'] != 77 or
        demo['sex_counts'] != {'male':31,'female':10} or demo['n_enrolled'] != 41):
        raise ValueError('External clinical reference-subject contract changed')


def reference_request(template, source, key, sex):
    if sex not in {'male','female'}:
        raise ValueError('Unexpected sex')
    request = deepcopy(template)
    request['studies'] = []
    request['requirement'].update(dose=40.,dose_unit='mg')
    instance = request['instance']
    protocol = instance['protocol']
    if protocol['ApplicationType'] != 'Oral' or protocol['DosingInterval'] != 'Single':
        raise ValueError('Source template must be a single oral tablet dose')
    if instance['formulation']['FormulationType'] != 'Formulation_Tablet_Weibull':
        raise ValueError('A source tablet cannot be silently replaced by solution')
    parameters = {p['Name']:p for p in protocol['Parameters']}
    if len(parameters) != len(protocol['Parameters']) or not {'InputDose','Start time'} <= parameters.keys():
        raise ValueError('Invalid source dosing parameters')
    parameters['InputDose'].update(Value=40.,Unit='mg')
    parameters['Start time'].update(Value=0.,Unit='h')
    protocol['Name'] = 'PROTO_furosemide_external_reference'
    instance['protocol_source'] = source
    individual = instance['individual']
    individual['Name'] = 'Furosemide_external_reference_'+sex
    individual['OriginData'].update(Age={'Value':39.,'Unit':'year(s)'},Weight={'Value':77.,'Unit':'kg'},Gender=sex.upper())
    individual['OriginData'].pop('Height',None)
    instance['generated_demographics'] = ['Height']
    instance['human_source'] = source
    instance['simulation_name'] = 'furosemide_external_'+key+'_'+sex
    instance['output_schema'] = [{'Parameters':[
        {'Name':'Start time','Value':start,'Unit':'h'}, {'Name':'End time','Value':end,'Unit':'h'},
        {'Name':'Resolution','Value':resolution,'Unit':'pts/h'}]} for start,end,resolution in ((0.,4.,100.),(4.,168.,10.))]
    return request


def validate_curve(times, values):
    if (len(times) != len(values) or len(times) < 4 or times[0] != 0 or abs(times[-1]-168*60)>1e-7 or
        any(not math.isfinite(v) for v in times+values) or any(v<0 for v in values) or
        max(values) <= 0 or any(b<=a for a,b in zip(times,times[1:]))):
        raise ValueError('Invalid native curve (including negative tails); no clipping, replacement or silent repair')


def directory_bytes(directory):
    total = 0
    for base, dirs, files in os.walk(directory, followlinks=False):
        if any((Path(base)/d).is_symlink() for d in dirs):
            raise ValueError('Symlink inside prediction output')
        for name in files:
            path = Path(base)/name
            if path.is_symlink():
                raise ValueError('Symlink inside prediction output')
            try:
                total += path.stat().st_size
            except FileNotFoundError:  # Our atomic JSON writes may remove their temp link.
                continue
    return total


class OutputBudget:
    """Reserve native quotas before dispatch; concurrent jobs cannot overcommit."""
    def __init__(self, directory, limit=TOTAL_OUTPUT_BYTES):
        self.directory, self.limit = directory, limit
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.reserved = 0

    @contextmanager
    def native_quota(self):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError('Prediction dispatch stopped after an earlier failure')
            remaining = self.limit-directory_bytes(self.directory)-self.reserved-OUTPUT_HEADROOM
            allowance = min(NATIVE_OUTPUT_BYTES, remaining//WORKERS)
            if allowance < MIN_NATIVE_BYTES:
                self.stop.set()
                raise RuntimeError('Insufficient remaining output quota under the 2 GiB ceiling')
            self.reserved += allowance
        try:
            yield allowance
        except BaseException:
            self.stop.set()
            raise
        finally:
            with self.lock:
                self.reserved -= allowance
                if directory_bytes(self.directory) > self.limit-OUTPUT_HEADROOM:
                    self.stop.set()

    def check(self):
        if directory_bytes(self.directory) > self.limit-OUTPUT_HEADROOM:
            self.stop.set()
            raise RuntimeError('Output ceiling reached; no further predictions may start')


def run_native(item, *, template, case, source, policy, output, budget):
    key, model, sex = item
    request = reference_request(template,source,key,sex)
    specs = case['parameter_search']['parameters']
    audit_design_inputs({'request':request,'parameters':specs},policy)
    # Fitted original coordinates are applied inside the compiler exactly once;
    # its candidate-specific specs map kcat to CLspec/[Enzyme] for linear variants.
    kwargs = {}
    if model.get('parameters_path'):
        kwargs = {'fitted_values':model['values'], 'parameter_provenance':ref(
            local(model['parameters_path']),'Frozen historical development fitted values','calibration_run')}
    with budget.native_quota() as allowance:
        variant = compile_native_candidate(request,model['action'],project_root=ROOT,
            parameter_specs=specs,**kwargs)
        directory = output/'native'/(key+'_'+sex)
        manifest = execute_snapshot(snapshot=variant['snapshot'],output_dir=directory,project_root=ROOT,
            timeout_seconds=240,cpu_threads=1,max_output_bytes=allowance)
        if not manifest['engineering_accepted'] or len(manifest['curve_outputs']) != 1:
            raise RuntimeError('External native prediction failed: '+str(directory))
        path = local(directory/manifest['curve_outputs'][0]['path'])
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Native output escaped its registered directory')
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows = list(csv.DictReader(stream))
        column = 'Organism|PeripheralVenousBlood|Furosemide|Plasma (Peripheral Venous Blood) [µmol/l]'
        times, values = [float(r['Time [min]']) for r in rows], [float(r[column]) for r in rows]
        validate_curve(times,values)
        budget.check()
        return {'model_key':key,'sex':sex,'time_min':times,'concentration':values,
            'unit':'µmol/l','time_unit':'min','native_csv_path':str(path.relative_to(ROOT)),
            'native_csv_sha256':file_sha(path)}


def run_bounded_jobs(jobs, predict, budget):
    pending, results, iterator = {}, [], iter(jobs)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        try:
            for _ in range(WORKERS):
                job = next(iterator,None)
                if job is not None: pending[pool.submit(predict,job)] = job
            while pending:
                ready,_ = wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    pending.pop(future)
                    results.append(future.result())
                    budget.check()
                    job = next(iterator,None)
                    if job is not None: pending[pool.submit(predict,job)] = job
        except BaseException:
            budget.stop.set()
            for future in pending: future.cancel()
            raise
    return sorted(results,key=lambda c:(c['model_key'],c['sex']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite')
    parser.add_argument('--design', required=True)
    parser.add_argument('--input-package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-hours', type=float, default=0)
    parser.add_argument('--validate-only',action='store_true',help='One source-initial male engineering run; no selection, scoring or final prediction lock')
    args = parser.parse_args()
    if not 0 <= args.wait_hours <= 8:
        raise ValueError('Wait allowance must be bounded to 0..8 hours')
    if not args.validate_only and not args.suite:
        parser.error('--suite is required for formal predictions')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    output = local(args.output,exists=False)
    design_path, package_path = local(args.design), local(args.input_package)
    if output.exists():
        raise FileExistsError('Prediction outputs and locks are immutable')
    frozen = {}
    case = read(package_path,frozen)['case']
    design = read(design_path,frozen)
    validate_design(design)
    template = next(s['design_request'] for s in case['scenarios']
                    if s['design_request']['requirement']['route']=='oral'
                    and s['design_request']['instance']['formulation']['FormulationType']=='Formulation_Tablet_Weibull')
    policy = metadata_exclusion_policy(design,design_path)
    audit_design_inputs({'request':template,'parameters':case['parameter_search']['parameters']},policy)
    mw = next(p['Value'] for p in template['instance']['compound']['Parameters'] if p['Name']=='Molecular weight')
    if args.validate_only:
        selections, models = [], {'source_initial_engineering':{'action':'partition_pksim_standard'}}
    else:
        suite = local(args.suite,exists=False)
        deadline = time.monotonic()+args.wait_hours*3600
        while not (suite/'completion.json').is_file():
            if time.monotonic()>=deadline:
                raise RuntimeError('The frozen method suite has not finished')
            time.sleep(min(15,max(0,deadline-time.monotonic())))
        selections, models = load_selections(suite,case,package_path,frozen)
    frozen[str(Path(__file__).resolve().relative_to(ROOT))] = file_sha(Path(__file__))
    for code in (ROOT/'src/pbpk_kb/blueprints').glob('*.py'):
        frozen[str(code.relative_to(ROOT))] = file_sha(code)
    verify_frozen(frozen)
    output.mkdir(parents=True)
    protocol_path = output/'prediction_protocol.json'
    protocol = {'case_id':design['case_id'],'design_sha256':file_sha(design_path),
        'purpose':'engineering_validation_only' if args.validate_only else 'frozen_external_predictions',
        'scientific_accepted':False,'performance_evaluation':False,
        'molecular_weight_g_mol':mw,'sex_weights':{'male':31/41,'female':10/41},
        'reference_age_year':39,'reference_weight_kg':77.0,'height':'PK-Sim native generation',
        'population_model':'European_ICRP_2002 inherited source; ethnic distribution not reproduced',
        'approximation':{'reference_subjects_only':True,'food_state_matched':False,
            'dissolution':'historical source Weibull; external product equivalence unverified',
            'demographics':'reported enrolled n=41 applied as an approximation to PK subset n=39',
            'group_mean':'sex-weighted reference-subject endpoints; not a virtual individual population',
            'Cmax':'dense-grid maximum; exact clinical sampling schedule unavailable',
            'AUC_infinity':'native AUC0-168h plus prespecified terminal exponential tail; diagnostics required'},
        'tail_policy':{'fit_window_start_h':134.4,'fit_window_end_h':168.,'minimum_points':5,
            'require_negative_log_slope':True,'maximum_tail_fraction_for_primary':.01,
            'minimum_r_squared_for_primary':.99,'do_not_clip_or_replace_failed_tail':True},
        'curve_validity_policy':'Finite strictly increasing time, full 0..168h, finite nonnegative nonempty concentrations; negative tails fail without repair',
        'selections':selections,'models':models,'evidence_exclusion':policy,
        'written_before_native_prediction_at':datetime.now(timezone.utc).isoformat(),
        'input_package_sha256':file_sha(package_path),'frozen_input_sha256':frozen,
        'resource_limits':{'total_output_bytes':TOTAL_OUTPUT_BYTES,'native_max_bytes':NATIVE_OUTPUT_BYTES,
            'native_timeout_seconds':240,'native_workers':WORKERS,'total_cpu_affinity':2,
            'disk_policy':'Reserve remaining capacity divided by two before each native run; stop below 16 MiB and preserve 16 MiB metadata/burst headroom'},
        'external_observations_read':False}
    save(protocol_path,protocol)
    frozen[str(protocol_path.relative_to(ROOT))] = file_sha(protocol_path)
    source = ref(protocol_path,'Prespecified reference subjects and dosing; no PK observations','user_request')
    budget = OutputBudget(output)
    jobs = [(key,model,sex) for key,model in models.items()
            for sex in (('male',) if args.validate_only else ('male','female'))]
    def predict(item):
        return run_native(item,template=template,case=case,source=source,policy=policy,output=output,budget=budget)
    curves = run_bounded_jobs(jobs,predict,budget)
    verify_frozen(frozen)
    budget.check()
    if args.validate_only:
        save(output/'engineering_validation.json',{'status':'engineering_validated',
            'scientific_accepted':False,'performance_evaluation':False,'method_selection_performed':False,
            'external_observations_read':False,'formal_prediction_lock_created':False,
            'native_run_count':len(curves),'protocol_sha256':file_sha(protocol_path),'models':curves,
            'output_bytes_before_report':directory_bytes(output)})
        print(json.dumps({'status':'engineering_validated','native_runs':len(curves),'performance_evaluation':False}),flush=True)
        return
    prediction_path = output/'predictions.json'
    save(prediction_path,{'case_id':design['case_id'],'models':curves,'selections':selections,
        'contains_observed_pk_values':False,'protocol_sha256':file_sha(protocol_path)})
    verify_frozen(frozen)
    budget.check()
    save(output/'prediction_lock.json',{'schema_version':'pbpk-external-predictions-lock-v1',
        'design_sha256':file_sha(design_path),'model_selection_frozen':True,
        'locked_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':{'path':str(protocol_path.relative_to(ROOT)),'sha256':file_sha(protocol_path)},
        'predictions':[{'case_id':design['case_id'],'path':str(prediction_path.relative_to(ROOT)),'sha256':file_sha(prediction_path)}],
        'method_job_count':len(selections),'physical_native_predictions':len(curves)})
    print(json.dumps({'status':'predictions_locked','methods':len(selections),'physical_predictions':len(curves)}),flush=True)


if __name__=='__main__':
    main()
