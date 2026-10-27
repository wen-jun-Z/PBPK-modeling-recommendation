#!/usr/bin/env python3
"""Lock all registered Rosuvastatin source-model method predictions before external scoring.

Independent trial outcomes are never opened, even for hashing. Reference demographics are
approximations, not individual covariates. No optimization or model fallback.
"""
from __future__ import annotations
import argparse
import ast
import bisect
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import NAMESPACE_URL, uuid5
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'runtime/blueprint_rosuvastatin_worktree_20260920'
sys.path.insert(0, str(STAGE/'src'))
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.experiment_runtime import candidate_space, digest, file_sha, save
from pbpk_kb.blueprints.mechanism_edits import compile_native_candidate
from pbpk_kb.blueprints.external_source_isolation import audit_design_inputs

TOTAL_OUTPUT_BYTES = 2*1024**3
NATIVE_OUTPUT_BYTES = 192*1024**2
OUTPUT_HEADROOM = 16*1024**2
MAX_WALL_SECONDS = 4*3600
NATIVE_TIMEOUT_SECONDS = 240
CONCENTRATION_COLUMN = 'Organism|PeripheralVenousBlood|Rosuvastatin|Plasma (Peripheral Venous Blood) [µmol/l]'
BLOCKS = {
 'main_5000':(5000,None,('M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0')),
 'budget_500':(500,None,('M3','M4','M5','M6')),
 'budget_2000':(2000,None,('M3','M4','M5','M6')),
 'matched_1':(5000,1,('M5','M6')),
}
PARTITION_ACTIONS = {'partition_pksim_standard','partition_rodgers_rowland'}
DEFAULT_PACKAGE = 'experiments/blueprint_design/development_inputs/rosuvastatin_reconstruction_20260920/input_package.json'
DEFAULT_SUITE = 'experiments/blueprint_design/rosuvastatin_source_suite_20260920'
DEFAULT_PROTOCOL = 'experiments/blueprint_design/external_validation/concentration_curve_audit_20260920/aggregate_statin_curves/protocols/rosuvastatin_pmc12378308_monotherapy_day_minus4_mean_curve.json'
QUALIFICATION_REASONS = [
 'Target breakfast at minus30 minutes is not implemented by the frozen native meal interface; source default physiology is retained.',
 'Crestor formulation transfer is unverified: historical two-component absorption proxy and Weibull release are preserved exactly, not two clinical administrations.',
 'Two reference-sex uncensored predictions approximate a published arithmetic mean after individual BLQ values were zeroed; joint covariates and individual BLQ counts are unavailable.',
]
QUALIFICATION = {'strict_main_eligible':False,'qualification_tier':'transportability_sensitivity','qualification_reasons':QUALIFICATION_REASONS}

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
    actions = PARTITION_ACTIONS
    expected_candidates = candidate_space(case, root=ROOT)
    if len(expected_candidates) != 8 or {c['action'] for c in expected_candidates} != actions:
        raise ValueError('Expected exactly two partition actions and eight candidates')
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
        if digest(tc['candidates']) != digest(expected_candidates):
            raise ValueError('Candidate space differs from the frozen stage compiler')
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
                    'parameter_contract_sha256':digest(specs),'parameters_sha256':file_sha(path)})
                selection.update(model_key=key,prediction_available=True,
                    selected_candidate_id=candidate['candidate_id'],effective_free_parameter_ids=effective,
                    parameters_path=str(path.relative_to(ROOT)),parameters_sha256=file_sha(path))
            else:
                selection['failure'] = row.get('blocked_reason') or 'No verified native prediction submitted upstream'
            selections.append(selection)
    if len(selections) != 57:
        raise ValueError('All 57 registered method jobs must retain their denominator')
    return selections, models

def original_grid_values(times,values,targets):
    if len(times)!=len(values) or len(times)<2 or any(not math.isfinite(v) for v in times+values):raise ValueError('Nonfinite native curve')
    if any(b<=a for a,b in zip(times,times[1:])) or any(v<0 for v in values) or max(values)<=0:raise ValueError('Negative/empty/invalid native curve; never clip')
    if targets[0]<times[0] or targets[-1]>times[-1]:raise ValueError('No extrapolation beyond native curve')
    sampled=[]
    for target in targets:
        index=bisect.bisect_left(times,target)
        if index<len(times) and times[index]==target:sampled.append(values[index]);continue
        if index==0 or index==len(times):raise ValueError('Target grid is uncovered')
        fraction=(target-times[index-1])/(times[index]-times[index-1])
        sampled.append(values[index-1]+fraction*(values[index]-values[index-1]))
    return sampled

def validate_source_proxy(template):
    instance = template['instance']
    if instance.get('protocol_representation') != 'source_absorption_proxy':
        raise ValueError('Historical absorption must remain an explicitly nonclinical proxy')
    protocol = instance['protocol']
    schema = protocol['Schemas']
    if len(schema) != 1 or protocol['DosingInterval'] != 'Single' or protocol['TimeUnit'] != 'h':
        raise ValueError('Expected original single unrepeated source absorption schema')
    params = {p['Name']:(p['Value'],p.get('Unit')) for p in schema[0]['Parameters']}
    if params != {'Start time':(0.0,'h'),'NumberOfRepetitions':(1.0,None),'TimeBetweenRepetitions':(0.0,'h')}:
        raise ValueError('Source absorption repetitions changed')
    items = schema[0]['SchemaItems']
    if len(items) != 2:raise ValueError('Both original source absorption components are required')
    for item, dose, start in zip(items,(25.36,14.64),(0.0,2.3),strict=True):
        values = {p['Name']:(p['Value'],p.get('Unit')) for p in item['Parameters']}
        if (item['ApplicationType'] != 'Oral' or item['FormulationKey'] != 'Formulation' or
            values != {'Start time':(start,'h'),'InputDose':(dose,'mg'),'Volume of water/body weight':(3.5,'ml/kg')}):
            raise ValueError('Do not replace historical source absorption with invented clinical redosing')
    formulation = instance['formulation']
    values = {p['Name']:(p['Value'],p.get('Unit')) for p in formulation['Parameters']}
    if (formulation['FormulationType'] != 'Formulation_Tablet_Weibull' or
        values != {'Dissolution time (50% dissolved)':(6.25,'min'),'Lag time':(0.0,'min'),
                   'Dissolution shape':(1.77,None),'Use as suspension':(1.0,None)}):
        raise ValueError('Source Weibull formulation must be preserved, not silently replaced by a Crestor fit')


def make_request(template,design,path,sex,model_key='engineering_source_initial'):
    if sex not in {'male','female'}:raise ValueError('Unknown reference sex')
    validate_source_proxy(template)
    request=deepcopy(template);instance=request['instance'];pop=design['population_metadata']
    request['studies']=[]
    request['requirement'].update(route='oral',dose=40.0,dose_unit='mg',formulation=instance['formulation']['Name'])
    instance['human_source']=ref(path,'Curated full-baseline N32 demographic means and sex weights only; no PK observations','user_request')
    individual=instance['individual'];individual['Name']='IND_rosu_reference_'+sex
    individual['OriginData'].update(Gender=sex.upper(),Age={'Value':pop['age_year']['mean'],'Unit':'year(s)'},
        Weight={'Value':pop['body_weight_kg']['mean'],'Unit':'kg'},Height={'Value':pop['height_cm']['mean'],'Unit':'cm'})
    # No meal interface exists in this frozen source builder. Preserve its native
    # physiology/proxy/formulation; protocol qualification declares the mismatch.
    instance['generated_demographics']=[]
    instance['simulation_name']='rosu_external_'+model_key+'_'+sex
    instance['output_schema']=[{'Parameters':[{'Name':'Start time','Value':a,'Unit':'h'},
        {'Name':'End time','Value':b,'Unit':'h'},{'Name':'Resolution','Value':r,'Unit':'pts/h'}]}
        for a,b,r in [(0.,12.,60.),(12.,120.,12.)]]
    for name in ('compound','protocol','protocol_representation','protocol_source','formulation','formulation_source','expression_profiles'):
        if instance[name] != template['instance'][name]:raise ValueError('Historical source anchor changed: '+name)
    if individual['Seed'] != template['instance']['individual']['Seed']:raise ValueError('Source seed changed')
    return BlueprintDesignRequest.model_validate(request).model_dump(mode='json')

def validate_design(design, dose=40):
    protocol=design['protocol'];pop=design['population_metadata']
    if (design.get('schema_version') != 'pbpk-independent-aggregate-curve-design-v1' or
        design.get('scientific_results_accessed_by_algorithm') is not False or design['compound'] != 'rosuvastatin' or
        protocol['dose'] != dose or dose != 40 or protocol['dose_unit'] != 'mg' or protocol['route'] != 'oral' or
        protocol['administrations'] != [{'time_min':0,'dose':40,'dose_unit':'mg'}] or
        protocol['formulation'] != 'Crestor tablet' or protocol['food_state'] != 'fed_standard_breakfast_30_minutes_before_dose' or
        protocol['concomitant_study_drug'] != 'none_at_day_minus4_baseline' or
        pop['n'] != 32 or pop['sex_counts'] != {'male':14,'female':18} or
        pop['sex_weights'] != {'male':14/32,'female':18/32} or
        pop['individual_covariates_or_correlations_available'] is not False or
        design['baseline_curve_n'] != 32 or design['prior_paired_endpoint_n'] != 31 or
        design['publication_cluster_id'] != 'pmid:40569335' or design['trial_cluster_ids'] != ['NCT06081166'] or
        design['primary_aggregation'] != 'arithmetic_mean_of_source_BLQ_zeroed_individual_concentrations' or
        design['censoring']['apply_individual_lloq_filter_to_positive_group_mean'] is not False):
        raise ValueError('Unexpected registered baseline aggregate-curve contract; never substitute paired N31')
    grid=protocol['sampling_times_min']
    if (len(grid) != 22 or grid[0] != 0 or grid[-1] != 7200 or
        grid != [t*60 for t in design['sampling_times_h']] or
        any(type(t) not in (int,float) or not math.isfinite(t) for t in grid) or
        any(b <= a for a,b in zip(grid,grid[1:]))):
        raise ValueError('Preserve all original nominal sampling positions through 120h')


def exclusion_policy(designs, paths):
    design=designs[40];ex=design['evidence_exclusion']
    if not ex.get('exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training') or not ex.get('exclude_entire_trial'):
        raise ValueError('Whole target publication/trial exclusion is mandatory')
    denied_paths=list(ex['raw_source_files'])+list(ex['curator_only_source_paths'])+[design['sealed_observation_reference']['path']]
    hashes=[design[k]['sha256'] for k in ('sealed_observation_reference','source_article','source_pdf')]
    if any(not re.fullmatch('[a-f0-9]{64}',h) for h in hashes):raise ValueError('Curator-registered source exclusions required')
    return {'schema_version':'pbpk-external-source-isolation-v1',
        'protocols':[{'case_id':design['case_id'],'study_id':design['study_id']}],
        'denied_paths':sorted({str(local(p,exists=False).relative_to(ROOT)).lower() for p in denied_paths}),
        'denied_hashes':hashes,
        'denied_identifiers':[{'kind':k,'value':v.lower()} for k,v in ex['identifiers'].items()]+
            [{'kind':'trial','value':v.lower()} for v in ex['trial_registry_ids']],
        'curated_metadata_sources':{str(p.relative_to(ROOT)):file_sha(p) for p in paths.values()},
        'scope':'Whole external publication/trial/supplement excluded from model inputs; excluded files never opened or hashed',
        'external_outcomes_read':False}


def bytes_written(output):
    total = 0
    for path in output.rglob('*'):
        if path.is_symlink():raise ValueError('Output symlinks are forbidden')
        if path.is_file():total += path.stat().st_size
    return total


def archive_pkml(directory, manifest, output, max_bytes=TOTAL_OUTPUT_BYTES):
    """Keep exact native model bytes, including generated GUIDs, losslessly."""
    archives = []
    for item in manifest.get('pkml_outputs',[]):
        path = local(directory/item['path'])
        if not path.is_relative_to(directory) or path.suffix.lower() != '.pkml':
            raise ValueError('Native model escaped its registered directory')
        original_sha, size = file_sha(path), path.stat().st_size
        if original_sha != item['sha256'] or size != item['byte_size']:
            raise ValueError('Native model hash/size mismatch')
        zipped = path.with_suffix(path.suffix+'.gz')
        if zipped.exists():raise FileExistsError('Immutable model archive exists')
        with path.open('rb') as source, zipped.open('xb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as compressed:
                while chunk := source.read(256*1024):
                    compressed.write(chunk)
                    if bytes_written(output) > max_bytes-OUTPUT_HEADROOM:
                        raise RuntimeError('Disk quota during exact native model preservation')
        recovered = hashlib.sha256(); recovered_size = 0
        with gzip.open(zipped,'rb') as handle:
            while chunk := handle.read(256*1024):
                recovered.update(chunk); recovered_size += len(chunk)
        if recovered.hexdigest() != original_sha or recovered_size != size:
            raise ValueError('Lossless model archive verification failed')
        archives.append({'path':str(zipped.relative_to(ROOT)),'sha256':file_sha(zipped),
            'original_path':str(path.relative_to(ROOT)),'original_sha256':original_sha,
            'original_byte_size':size})
        # Only remove this invocation's duplicate after successful exact recovery.
        path.unlink()
    return archives


def registered_jobs(designs, models):
    return [{'case_id':designs[dose]['case_id'],'model_key':key,'sex':sex}
        for dose in designs for key in sorted(models) for sex in ('male','female')]


def terminal_record(job, target, status, error=None):
    if status not in {'completed','native_failed','not_run_resource_limit'}:
        raise ValueError('Unknown native terminal status')
    return {**job,'status':status,'time_min':target,'concentration':None,
        'unit':'µmol/l','time_unit':'min','error':error,'individual_prediction':False,
        'native_execution_invoked':False}


def freeze_sources(case, frozen):
    for item in case['source_audit']:
        path = local(item['path'])
        if file_sha(path) != item['sha256']:raise ValueError('Historical source changed')
        frozen[str(path.relative_to(ROOT))] = item['sha256']
    for module_name in ('design_contracts','design_runtime','experiment_runtime','mechanism_edits','native_edits','source_cotherapy'):
        module = sys.modules.get('pbpk_kb.blueprints.'+module_name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(STAGE/'src'):
            raise ValueError('Wrong blueprint source imported: '+module_name)
    for path in (STAGE/'src/pbpk_kb/blueprints').glob('*.py'):
        frozen[str(path.relative_to(ROOT))] = file_sha(path)
    for path in (ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R'):
        frozen[str(path.relative_to(ROOT))] = file_sha(path)
    script = Path(__file__).resolve()
    frozen[str(script.relative_to(ROOT))] = file_sha(script)


def load_scoring_contract(frozen):
    """Read a literal preregistration; never import a scorer or open observations."""
    path = local('scripts/score_locked_aggregate_curves.py')
    raw = path.read_bytes()
    assignments = [node for node in ast.parse(raw).body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id == 'EVALUATION_CONTRACT' for target in node.targets)]
    if len(assignments) != 1:raise ValueError('One literal evaluation preregistration is required')
    contract = ast.literal_eval(assignments[0].value)
    if not isinstance(contract,dict) or contract.get('version') != 'pbpk-aggregate-curve-evaluation-v1':
        raise ValueError('Unknown scoring preregistration')
    frozen[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return contract


def protocol_record(design, design_path, template, selections, models, evaluation_contract):
    pop=design['population_metadata'];instance=template['instance']
    mw=next(p['Value'] for p in instance['compound']['Parameters'] if p['Name']=='Molecular weight')
    return {'case_id':design['case_id'],'design_sha256':file_sha(design_path),
        'model_keys':sorted(models),'models':models,'selections':selections,'evaluation_contract':evaluation_contract,
        **QUALIFICATION,'molecular_weight_g_mol':mw,'sex_weights':pop['sex_weights'],
        'prediction_aggregation':'sex_weighted_reference_concentration',
        'primary_aggregation':design['primary_aggregation'],
        'native_concentration_column':CONCENTRATION_COLUMN,
        'sampling_time_h':design['sampling_times_h'],'sampling_time_min':design['protocol']['sampling_times_min'],
        'sampling_mapping':'Original nominal 22-position schedule through120h, linear interpolation only; no rounding, shifts or extrapolation',
        'native_output_grid':{'0..12h_points_per_hour':60,'12..120h_points_per_hour':12},
        'target_clinical_protocol':design['protocol'],'target_baseline_curve_n':32,'paired_endpoint_n_not_used':31,
        'target_dose_semantics':'One nominal40mg active-drug Crestor tablet at0, not two clinical administrations',
        'native_administration_representation':'source_absorption_proxy',
        'native_source_protocol':instance['protocol'],'native_source_formulation':instance['formulation'],
        'native_source_protocol_sha256':digest(instance['protocol']),
        'native_source_formulation_sha256':digest(instance['formulation']),
        'target_meal_time_min':-30,'native_meal_event_implemented':False,
        'source_proxy_components':[{'source_input_mg':25.36,'time_h':0.0},{'source_input_mg':14.64,'time_h':2.3}],
        'historical_source_proxy_is_not_clinical_redosing':True,'formulation_transfer_verified':False,
        'model_identity':'Historical Martin2003b calibration/Csonka2019 selection; same frozen source proxy for every method',
        'clinical_coadministration':None,'external_fit_performed':False,
        'approximation':{'reference_subjects_only':True,'group_mean_reference_mismatch':True,
            'source_individual_BLQ_zeroed_mean_vs_uncensored_prediction':True,
            'individual_covariates_available':False,'personalized_prediction':False,
            'demographics':'Male and female references share cohort means49y/76.4kg/166cm; joint distribution and sex-specific covariates unknown',
            'sex_mixture':'14/32 male plus18/32 female; counts describe fullbaseline cohort, not per-time sample sizes',
            'population_category':instance['individual']['OriginData']['Population'],
            'enzyme_profiles':'Historical source expressions retained; no race/genotype adjustments inferred',
            'food':'Target fed meal at-30min unimplemented; frozen native source default physiology retained',
            'formulation':'Original historical absorption proxy and release preserved without a new target-specific fit',
            'assay':'No individual LLOQ threshold imposed on predicted group curve or positive published group means'},
        'statistical_unit':'One compound, one publication, one aggregate curve; seeds and scheduled samples are not independent studies',
        'observation_policy':'Post-lock aggregate scorer only; zero rows retained, no individual-data reconstruction or individual NCA claims',
        'prediction_validity':'Full native curve finite and nonnegative; no clipping or imputation',
        'external_observations_read':False}


def run_predictions(*, output, designs, paths, template, specs, models, selections, frozen, policy, deadline):
    """Single native process, two CPU threads; terminal records retain all jobs."""
    if len(selections) != 57 or len({s['selection_id'] for s in selections}) != 57:
        raise ValueError('All 57 unique selection identities must be frozen')
    if set(models) != {s['model_key'] for s in selections if s['model_key'] is not None}:
        raise ValueError('Unique model registry must match the complete selection list')
    evaluation_contract = load_scoring_contract(frozen)
    output.mkdir(parents=True,exist_ok=False)
    protocols = {dose:protocol_record(d,paths[dose],template,selections,models,evaluation_contract) for dose,d in designs.items()}
    refs = []
    for dose, protocol in protocols.items():
        path = output/f'oral_{dose}mg'/'protocol.json';save(path,protocol)
        refs.append({'case_id':designs[dose]['case_id'],
            'design_path':str(paths[dose].relative_to(ROOT)),'design_sha256':file_sha(paths[dose]),
            'protocol_path':str(path.relative_to(ROOT)),'protocol_sha256':file_sha(path)})
    jobs = registered_jobs(designs,models)
    requests = {}
    for job in jobs:
        dose = next(d for d in designs if designs[d]['case_id']==job['case_id'])
        key = f"{job['case_id']}__{job['model_key']}__{job['sex']}"
        requests[key] = make_request(template,designs[dose],paths[dose],job['sex'],job['model_key'])
        audit_design_inputs({'request':requests[key],'parameters':specs},policy,
            allowed_metadata_sources=list(policy['curated_metadata_sources']))
    verify_frozen(frozen)
    config = output/'config_freeze.json'
    save(config,{'schema_version':'pbpk-rosuvastatin-methods-external-freeze-v1',
        'frozen_before_native_prediction_at':datetime.now(timezone.utc).isoformat(),
        'cases':refs,'models':models,'selections':selections,'expected_native_jobs':jobs,'requests':requests,
        'evaluation_contract':evaluation_contract,**QUALIFICATION,
        'frozen_input_sha256':dict(frozen),'evidence_exclusion':policy,'method_job_count':57,
        'expected_prediction_rows':57,'native_runs_registered':len(jobs),'native_workers':1,'cpu_threads':2,
        'native_timeout_seconds':NATIVE_TIMEOUT_SECONDS,'maximum_total_output_bytes':TOTAL_OUTPUT_BYTES,
        'maximum_wall_seconds':MAX_WALL_SECONDS,'stop_after_consecutive_native_failures':3,
        'retry_count':0,'external_observations_read':False,'external_fit_performed':False,
        'stage_source':str((STAGE/'src').relative_to(ROOT))})
    frozen[str(config.relative_to(ROOT))] = file_sha(config)
    for item in refs:frozen[item['protocol_path']] = item['protocol_sha256']
    outcomes = {dose:[] for dose in designs}; consecutive_failures = 0; stop_reason = None
    for job in jobs:
        verify_frozen(frozen)
        dose = next(d for d in designs if designs[d]['case_id']==job['case_id'])
        target = protocols[dose]['sampling_time_min']
        if stop_reason is None:
            if time.monotonic() >= deadline:stop_reason = 'Four-hour wall limit reached'
            elif bytes_written(output) > TOTAL_OUTPUT_BYTES-NATIVE_OUTPUT_BYTES-OUTPUT_HEADROOM:
                stop_reason = 'Insufficient bounded disk headroom for next native job'
        if stop_reason:
            record = terminal_record(job,target,'not_run_resource_limit',stop_reason)
        else:
            record = terminal_record(job,target,'native_failed')
            key = f"{job['case_id']}__{job['model_key']}__{job['sex']}"
            directory = output/'native'/key
            try:
                model = models[job['model_key']]
                provenance = ref(local(model['parameters_path']),
                    'Frozen historical calibration vector; external study never used for fitting','calibration_run')
                built = compile_native_candidate(requests[key],model['action'],project_root=ROOT,
                    parameter_specs=specs,fitted_values=model['values'],parameter_provenance=provenance)
                record['native_execution_invoked'] = True
                result = execute_snapshot(snapshot=built['snapshot'],output_dir=directory,project_root=ROOT,
                    timeout_seconds=max(1,min(NATIVE_TIMEOUT_SECONDS,int(deadline-time.monotonic()))),
                    cpu_threads=2,max_output_bytes=NATIVE_OUTPUT_BYTES)
                record['native_manifest_path'] = str((directory/'execution_manifest.json').relative_to(ROOT))
                record['native_manifest_sha256'] = file_sha(directory/'execution_manifest.json')
                record['pkml_archives'] = archive_pkml(directory,result,output)
                if not result['engineering_accepted'] or len(result['curve_outputs']) != 1 or len(record['pkml_archives']) != 1:
                    raise ValueError('Native execution did not produce one accepted model/curve pair')
                path = local(directory/result['curve_outputs'][0]['path'])
                if not path.is_relative_to(directory):raise ValueError('Native CSV escaped registered directory')
                if file_sha(path) != result['curve_outputs'][0]['sha256']:
                    raise ValueError('Native CSV hash differs from execution manifest')
                record.update(native_csv_path=str(path.relative_to(ROOT)),native_csv_sha256=file_sha(path),
                    native_time_column='Time [min]',native_concentration_column=CONCENTRATION_COLUMN)
                with path.open(encoding='utf-8-sig',newline='') as handle:rows = list(csv.DictReader(handle))
                times = [float(r['Time [min]']) for r in rows]
                values = [float(r[CONCENTRATION_COLUMN]) for r in rows]
                sampled = original_grid_values(times,values,target)
                record.update(status='completed',error=None,concentration=sampled,native_points=len(times))
                consecutive_failures = 0
            except Exception as exc:
                record['error'] = type(exc).__name__+': '+str(exc)
                consecutive_failures += 1
                if consecutive_failures >= 3:stop_reason = 'Three consecutive native prediction failures; no further dispatch'
            verify_frozen(frozen)
        outcomes[dose].append(record)
        save(output/'native_progress.json',{'completed_terminal_jobs':sum(map(len,outcomes.values())),
            'registered_native_jobs':len(jobs),'last_job':job,'last_status':record['status'],'stop_reason':stop_reason},replace=True)
        print(json.dumps({'status':record['status'],**job}),flush=True)
    predictions = []
    for reference in refs:
        dose = next(d for d in designs if designs[d]['case_id']==reference['case_id'])
        path = output/f'oral_{dose}mg'/'predictions.json'
        save(path,{'case_id':reference['case_id'],'models':outcomes[dose],'selections':selections,
            'protocol_sha256':reference['protocol_sha256'],'contains_observed_pk_values':False})
        predictions.append({**reference,'path':str(path.relative_to(ROOT)),'sha256':file_sha(path)})
    verify_frozen(frozen)
    if bytes_written(output) > TOTAL_OUTPUT_BYTES-OUTPUT_HEADROOM:
        raise RuntimeError('Output quota exceeded; no final lock can be published')
    save(output/'prediction_lock.json',{'schema_version':'pbpk-external-predictions-lock-v1',
        'locked_at_utc':datetime.now(timezone.utc).isoformat(),'model_selection_frozen':True,**QUALIFICATION,
        'config_freeze':{'path':str(config.relative_to(ROOT)),'sha256':file_sha(config)},'predictions':predictions,
        'method_job_count':57,'expected_prediction_rows':57,'all_registered_predictions_terminal':True,
        'registered_native_jobs':len(jobs),
        'native_execution_attempts':sum(r['native_execution_invoked'] for rows in outcomes.values() for r in rows),
        'native_execution_attempts_definition':'Calls entered into execute_snapshot, including failures in its native build/execution; excludes earlier candidate compilation failures and resource-limit skips',
        'completed_native_jobs':sum(r['status']=='completed' for rows in outcomes.values() for r in rows),
        'stop_reason':stop_reason,'external_observations_read':False,'independent_study_count':1,'independent_compound_count':1})
    return output/'prediction_lock.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',default=DEFAULT_SUITE)
    parser.add_argument('--input-package',default=DEFAULT_PACKAGE)
    parser.add_argument('--protocol',default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',required=True)
    parser.add_argument('--wait-hours',type=float,default=0)
    parser.add_argument('--validate-only',action='store_true',help='Compile four source-initial snapshots; no native simulation, external score or fitted-method claim')
    args=parser.parse_args()
    if not 0 <= args.wait_hours <= 4:raise ValueError('Wait allowance must lie in0..4 hours')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    started=time.monotonic();deadline=started+MAX_WALL_SECONDS
    output=local(args.output,exists=False)
    if output.exists():raise FileExistsError('Prediction and engineering outputs are immutable')
    package=local(args.input_package);frozen={};case=read(package,frozen)['case']
    if (case['compound_id'] != 'drug:rosuvastatin' or case['case_id'] != 'rosuvastatin_martin2003b_csonka2019_partition_v1' or
        [p['parameter_id'] for p in case['parameter_search']['parameters']] != ['specific_intestinal_permeability_transcellular','lipophilicity']):
        raise ValueError('Expected frozen Rosuvastatin historical source case')
    template=next(s['design_request'] for s in case['scenarios'] if s['role']=='calibration' and s['design_request']['requirement']['dose']==40)
    validate_source_proxy(template);specs=case['parameter_search']['parameters']
    paths={40:local(args.protocol)};designs={40:read(paths[40],frozen)};validate_design(designs[40])
    policy=exclusion_policy(designs,paths)
    audit_design_inputs({'request':template,'parameters':specs},policy)
    freeze_sources(case,frozen)
    if args.validate_only:
        receipt=[]
        for sex in ('male','female'):
            request=make_request(template,designs[40],paths[40],sex)
            audit_design_inputs({'request':request,'parameters':specs},policy,allowed_metadata_sources=list(policy['curated_metadata_sources']))
            for action in sorted(PARTITION_ACTIONS):
                built=compile_native_candidate(request,action,project_root=ROOT,parameter_specs=specs)
                receipt.append({'sex':sex,'action':action,'snapshot_sha256':digest(built['snapshot']),
                    'source_proxy_sha256':digest(request['instance']['protocol']),
                    'source_formulation_sha256':digest(request['instance']['formulation']),
                    'compound_names':[c['Name'] for c in built['snapshot']['Compounds']]})
        verify_frozen(frozen);output.mkdir(parents=True,exist_ok=False)
        save(output/'engineering_preflight.json',{'status':'compiled','native_simulations_performed':0,'snapshot_contracts':receipt,
            **QUALIFICATION,'external_observations_read':False,'model_fitting_performed':False,'formal_method_predictions':False,
            'frozen_input_sha256':frozen})
        print(json.dumps({'status':'engineering_preflight_passed','contracts':len(receipt),**QUALIFICATION,'output':str(output.relative_to(ROOT))}));return
    suite=local(args.suite,exists=False);wait_deadline=min(deadline,started+args.wait_hours*3600)
    while not (suite/'completion.json').is_file():
        if time.monotonic() >= wait_deadline:raise RuntimeError('Registered method suite has not finished')
        time.sleep(min(15,max(0,wait_deadline-time.monotonic())))
    selections,models=load_selections(suite,case,package,frozen)
    for source,checksum in read(suite/'configuration.json')['code_sha256'].items():
        if file_sha(local(source)) != checksum:raise ValueError('Upstream registered source changed')
        frozen[source]=checksum
    lock=run_predictions(output=output,designs=designs,paths=paths,template=template,specs=specs,models=models,
        selections=selections,frozen=frozen,policy=policy,deadline=deadline)
    print(json.dumps({'status':'predictions_locked','path':str(lock.relative_to(ROOT)),'method_jobs':57,'aggregate_cases':1,**QUALIFICATION}),flush=True)

if __name__ == '__main__':main()
