#!/usr/bin/env python3
"""Freeze all Clari method predictions for the independent repeated-dose study.

Source kinetics and formulation remain intact. No final-study outcomes, external
fitting, fallback model selection, or GPU calls occur in this program.
"""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL_STAGE=ROOT/'runtime/blueprint_clari_external_20260920'
sys.path.insert(0,str(EXTERNAL_STAGE/'scripts'))
from clari_contracts import (STAGE, local, read, ref, file_sha, digest, save, candidate_space,
    load_selections, verify_frozen, audit_design_inputs, original_grid_values,
    archive_pkml, bytes_written, terminal_record, execute_snapshot,
    TOTAL_OUTPUT_BYTES, NATIVE_OUTPUT_BYTES, OUTPUT_HEADROOM)
from clari_protocol import (CASE_ID, PACKAGE_REL, SUITE_REL, DESIGN_REL, COLUMN, QUALIFICATION,
    SAMPLING_MIN, load_design, exclusion_policy, make_request, compile_model, protocol_record)

MAX_WALL_SECONDS=4*3600
NATIVE_TIMEOUT_SECONDS=300


def load_scoring_contract(frozen):
    path=ROOT/'scripts/score_clarithromycin_methods_endpoints.py'
    raw=path.read_bytes()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='EVALUATION_CONTRACT' for t in n.targets)]
    if len(nodes)!=1:raise ValueError('One literal evaluation contract required')
    contract=ast.literal_eval(nodes[0].value)
    if contract['version']!='pbpk-clarithromycin-repeated-endpoints-v1':raise ValueError('Unknown endpoint evaluation contract')
    frozen[str(path.relative_to(ROOT))]=file_sha(path)
    artifact_helper=ROOT/'scripts/score_locked_aggregate_curves.py'
    frozen[str(artifact_helper.relative_to(ROOT))]=file_sha(artifact_helper)
    return contract


def freeze_sources(case,frozen):
    for source in case['source_audit']:
        path=local(source['path'])
        if file_sha(path)!=source['sha256']:raise ValueError('Historical source audit changed')
        frozen[str(path.relative_to(ROOT))]=source['sha256']
    for name in ('design_runtime','human_model_snapshot','native_edits','mechanism_edits','experiment_runtime'):
        module=sys.modules.get('pbpk_kb.blueprints.'+name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(STAGE/'src'):
            raise ValueError('Wrong frozen stage imported: '+name)
    sources=[Path(__file__).resolve(),*(EXTERNAL_STAGE/'scripts').glob('*.py'),
      *(STAGE/'src/pbpk_kb/blueprints').glob('*.py'),
      ROOT/'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
      ROOT/'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R',STAGE/'source_registration.json']
    for path in sources:frozen[str(path.relative_to(ROOT))]=file_sha(path)


def source_inputs(package_path,design_path,frozen):
    package=read(package_path,frozen);case=package['case']
    if case['case_id']!='clarithromycin_chu_source_reconstruction_v1' or package.get('external_scenarios'):
        raise ValueError('Unexpected historical Clari source case')
    design=load_design(design_path,frozen)
    template=next(s['design_request'] for s in case['scenarios'] if s['scenario_id']=='clarithromycin_po_chu_1993_500_mg')
    specs=case['parameter_search']['parameters']
    candidates=candidate_space(case,root=ROOT)
    if len(candidates)!=8 or {c['action'] for c in candidates}!={'partition_pksim_standard','partition_rodgers_rowland'}:
        raise ValueError('Complete two-formula/eight-candidate source contract required')
    freeze_sources(case,frozen)
    return case,design,template,specs,candidates


def validate_source_compilation(case,design,template,specs,candidates,design_path):
    request=make_request(template,design,design_path)
    policy=exclusion_policy(design,design_path)
    audit_design_inputs({'request':request,'parameters':specs},policy,allowed_metadata_sources=list(policy['curated_metadata_sources']))
    compiled=[]
    for candidate in candidates:
        result=compile_model(request,candidate['action'],specs)
        compiled.append({'candidate_id':candidate['candidate_id'],'snapshot_sha256':digest(result['snapshot']),
          'clinical_regimen_audit':result['clinical_regimen_audit']})
    return {'case_id':case['case_id'],'target_case_id':CASE_ID,'candidate_count':len(compiled),
            'candidate_snapshots':compiled,'native_calls':0,'external_outcomes_read':False,**QUALIFICATION}


def authenticated_registry(suite_path,package_path,design_path):
    """Rebuild all source bindings and selections without opening outcomes."""
    frozen={}
    case,design,template,specs,candidates=source_inputs(package_path,design_path,frozen)
    selections,models=load_selections(suite_path,case,package_path,frozen)
    load_scoring_contract(frozen)
    return {'case':case,'design':design,'template':template,'specs':specs,
            'selections':selections,'models':models,'frozen_input_sha256':frozen}


def run_predictions(*,output,design,design_path,template,specs,models,selections,frozen,policy,deadline,
                    suite_path=None,package_path=None):
    if len(selections)!=57 or len({s['selection_id'] for s in selections})!=57:
        raise ValueError('All 57 method selections required')
    if set(models)!={s['model_key'] for s in selections if s['model_key'] is not None}:
        raise ValueError('Unique models must match every available frozen selection')
    evaluation=load_scoring_contract(frozen)
    output.mkdir(parents=True,exist_ok=False)
    protocol=protocol_record(design,design_path,template,selections,models,evaluation)
    protocol.update(suite_path=str(suite_path.relative_to(ROOT)) if suite_path else None,
      input_package_path=str(package_path.relative_to(ROOT)) if package_path else None,
      input_package_sha256=file_sha(package_path) if package_path else None)
    protocol_path=output/'protocol.json';save(protocol_path,protocol)
    reference={'case_id':CASE_ID,'design_path':str(design_path.relative_to(ROOT)),'design_sha256':file_sha(design_path),
               'protocol_path':str(protocol_path.relative_to(ROOT)),'protocol_sha256':file_sha(protocol_path)}
    jobs=[{'case_id':CASE_ID,'model_key':key,'sex':'male'} for key in sorted(models)]
    requests={j['model_key']:make_request(template,design,design_path,j['model_key']) for j in jobs}
    for request in requests.values():
        audit_design_inputs({'request':request,'parameters':specs},policy,allowed_metadata_sources=list(policy['curated_metadata_sources']))
    verify_frozen(frozen)
    config_path=output/'config_freeze.json'
    save(config_path,{'schema_version':'pbpk-clarithromycin-methods-external-freeze-v1',
      'frozen_before_native_prediction_at':datetime.now(timezone.utc).isoformat(),
      'cases':[reference],'models':models,'selections':selections,'expected_native_jobs':jobs,'requests':requests,
      'evaluation_contract':evaluation,'frozen_input_sha256':dict(frozen),'evidence_exclusion':policy,
      'method_job_count':57,'expected_prediction_rows':57,'native_runs_registered':len(jobs),
      'native_workers':1,'cpu_threads':2,'native_timeout_seconds':NATIVE_TIMEOUT_SECONDS,
      'maximum_total_output_bytes':TOTAL_OUTPUT_BYTES,'maximum_wall_seconds':MAX_WALL_SECONDS,
      'stop_after_consecutive_native_failures':3,'retry_count':0,'external_observations_read':False,
      'external_fit_performed':False,'stage_source':str((STAGE/'src').relative_to(ROOT)),
      'suite_path':str(suite_path.relative_to(ROOT)) if suite_path else None,
      'input_package_path':str(package_path.relative_to(ROOT)) if package_path else None,
      'input_package_sha256':file_sha(package_path) if package_path else None,**QUALIFICATION})
    frozen[str(config_path.relative_to(ROOT))]=file_sha(config_path)
    frozen[str(protocol_path.relative_to(ROOT))]=file_sha(protocol_path)
    rows=[];stop_reason=None;consecutive_failures=0
    for job in jobs:
        verify_frozen(frozen)
        if stop_reason is None:
            if time.monotonic()>=deadline:stop_reason='Four-hour dispatch limit reached'
            elif bytes_written(output)>TOTAL_OUTPUT_BYTES-NATIVE_OUTPUT_BYTES-OUTPUT_HEADROOM:
                stop_reason='Insufficient disk headroom for next bounded native job'
        if stop_reason:
            row=terminal_record(job,SAMPLING_MIN,'not_run_resource_limit',stop_reason)
        else:
            row=terminal_record(job,SAMPLING_MIN,'native_failed');key=job['model_key']
            directory=output/'native'/(CASE_ID+'__'+key+'__male')
            try:
                model=models[key]
                provenance=ref(local(model['parameters_path']),'Frozen historical development values; no external fitting','calibration_run')
                compiled=compile_model(requests[key],model['action'],specs,values=model['values'],provenance=provenance)
                audit_path=output/'regimen_audits'/(key+'.json');save(audit_path,compiled['clinical_regimen_audit'])
                row.update(clinical_regimen_audit_path=str(audit_path.relative_to(ROOT)),clinical_regimen_audit_sha256=file_sha(audit_path))
                row['native_execution_invoked']=True
                result=execute_snapshot(snapshot=compiled['snapshot'],output_dir=directory,project_root=ROOT,
                  timeout_seconds=max(1,min(NATIVE_TIMEOUT_SECONDS,int(deadline-time.monotonic()))),
                  cpu_threads=2,max_output_bytes=NATIVE_OUTPUT_BYTES)
                row.update(native_manifest_path=str((directory/'execution_manifest.json').relative_to(ROOT)),
                  native_manifest_sha256=file_sha(directory/'execution_manifest.json'))
                row['pkml_archives']=archive_pkml(directory,result,output)
                if not result['engineering_accepted'] or len(result['curve_outputs'])!=1 or len(row['pkml_archives'])!=1:
                    raise ValueError('Native prediction did not yield one accepted curve/model pair')
                csv_path=local(directory/result['curve_outputs'][0]['path'])
                if not csv_path.is_relative_to(directory) or file_sha(csv_path)!=result['curve_outputs'][0]['sha256']:
                    raise ValueError('Native CSV path or checksum mismatch')
                with csv_path.open(encoding='utf-8-sig',newline='') as handle:native=list(csv.DictReader(handle))
                times=[float(r['Time [min]']) for r in native];values=[float(r[COLUMN]) for r in native]
                if times[0]!=0. or times[-1]!=108.*60:raise ValueError('Incomplete nominal repeated-dose simulation window')
                sampled=original_grid_values(times,values,SAMPLING_MIN)
                row.update(status='completed',error=None,concentration=sampled,native_points=len(times),
                  native_csv_path=str(csv_path.relative_to(ROOT)),native_csv_sha256=file_sha(csv_path),
                  native_time_column='Time [min]',native_concentration_column=COLUMN)
                consecutive_failures=0
            except Exception as exc:
                row['error']=type(exc).__name__+': '+str(exc);consecutive_failures+=1
                if consecutive_failures>=3:stop_reason='Three consecutive native failures; no further dispatch'
        verify_frozen(frozen);rows.append(row)
        save(output/'native_progress.json',{'terminal_jobs':len(rows),'registered_native_jobs':len(jobs),
          'last_job':job,'last_status':row['status'],'stop_reason':stop_reason},replace=True)
        print(json.dumps({**job,'status':row['status']}),flush=True)
    prediction_path=output/'predictions.json'
    save(prediction_path,{'case_id':CASE_ID,'models':rows,'selections':selections,
      'protocol_sha256':reference['protocol_sha256'],'contains_observed_pk_values':False})
    verify_frozen(frozen)
    if bytes_written(output)>TOTAL_OUTPUT_BYTES-OUTPUT_HEADROOM:raise RuntimeError('Final lock would exceed quota')
    lock=output/'prediction_lock.json'
    save(lock,{'schema_version':'pbpk-external-predictions-lock-v1','locked_at_utc':datetime.now(timezone.utc).isoformat(),
      'model_selection_frozen':True,'config_freeze':{'path':str(config_path.relative_to(ROOT)),'sha256':file_sha(config_path)},
      'predictions':[{**reference,'path':str(prediction_path.relative_to(ROOT)),'sha256':file_sha(prediction_path)}],
      'method_job_count':57,'expected_prediction_rows':57,'all_registered_predictions_terminal':True,
      'registered_native_jobs':len(jobs),'native_execution_attempts':sum(r['native_execution_invoked'] for r in rows),
      'native_execution_attempts_definition':'Entered execute_snapshot, including native build/run failures; excludes compile failures and resource skips',
      'completed_native_jobs':sum(r['status']=='completed' for r in rows),'stop_reason':stop_reason,
      'external_observations_read':False,'independent_study_count':1,'independent_compound_count':1,**QUALIFICATION})
    return lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',default=SUITE_REL);parser.add_argument('--input-package',default=PACKAGE_REL)
    parser.add_argument('--protocol',default=DESIGN_REL);parser.add_argument('--output',required=True)
    parser.add_argument('--wait-hours',type=float,default=0);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if not 0<=args.wait_hours<=12:raise ValueError('Bounded wait must be between zero and12h')
    suite,package,design_path=local(args.suite),local(args.input_package),local(args.protocol)
    output=local(args.output,exists=False);frozen={}
    if output.exists():raise FileExistsError('Never overwrite predictions')
    case,design,template,specs,candidates=source_inputs(package,design_path,frozen)
    checked=validate_source_compilation(case,design,template,specs,candidates,design_path)
    if args.validate_only:
        output.mkdir(parents=True,exist_ok=False);save(output/'preflight.json',checked)
        print(json.dumps({'status':'compile_preflight_passed','candidate_count':8,'native_calls':0,'external_outcomes_read':False}));return
    wait_until=time.monotonic()+args.wait_hours*3600
    while not (suite/'completion.json').exists():
        if time.monotonic()>=wait_until:raise RuntimeError('Upstream method suite has not completed')
        verify_frozen(frozen);time.sleep(5)
    selections,models=load_selections(suite,case,package,frozen)
    lock=run_predictions(output=output,design=design,design_path=design_path,template=template,specs=specs,
      models=models,selections=selections,frozen=frozen,policy=exclusion_policy(design,design_path),
      deadline=time.monotonic()+MAX_WALL_SECONDS,suite_path=suite,package_path=package)
    print(json.dumps({'status':'all_predictions_locked','lock_path':str(lock.relative_to(ROOT)),
                      'lock_sha256':file_sha(lock),'external_outcomes_read':False}),flush=True)


if __name__=='__main__':main()
