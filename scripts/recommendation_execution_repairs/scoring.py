"""Register repaired executors with the unchanged clinical statistical scorer.

Only the execution revision/required dependency profile differs. Endpoint
estimators, numerical comparisons, tie handling, independent blocks and outcome
access barriers are inherited unchanged from the frozen original scorer.
"""
from contextlib import contextmanager
from pathlib import Path
import threading
import json

from recommendation_execution_repairs import complete_candidate, snapshot_candidate
import score_recommendation_execution as backend

REVISION='engineering_repair_04'
_LOCK=threading.Lock()


def verify_engineering_barrier(bundle):
    """Offline scoring must obey the same proof barrier as the live controller."""
    from recommendation_execution_repairs.pipeline import engineering_completeness, verify_source_repairs
    f,run=bundle['files'],bundle['run']
    protocol=f.read(run/'protocol.json')
    bound={f.checked(ref) for ref in protocol['inputs']}
    work=run/'engineering_repairs/repair_04'
    required={work/'source_repair_summary.json',work/'engineering_completeness.json'}
    if not required.issubset(bound):raise ValueError('Engineering repair reports are absent from frozen protocol')
    summary=f.read(work/'source_repair_summary.json')
    jobs=f.read(run/'inputs/execution_plan.json')['jobs']
    results=[entry['result'] for entry in bundle['native'].values()]
    computed=engineering_completeness(summary,jobs,results)
    if computed!=f.read(work/'engineering_completeness.json') or computed['accepted'] is not True:
        raise ValueError('Engineering completeness does not authorize clinical outcomes')
    from recommendation_execution_repairs.engineering_gate import assessed_gate_evidence_refs
    if not {f.checked(ref) for ref in assessed_gate_evidence_refs(computed)}.issubset(bound):
        raise ValueError('Unavailable-case evidence is absent from frozen protocol')
    refs=verify_source_repairs(summary,f.rows(run/'inputs/candidate_audit.jsonl'),root=f.root)
    if not {f.checked(ref) for ref in refs}.issubset(bound):
        raise ValueError('Source repair proofs are absent from frozen protocol')
    from recommendation_execution_source_audit import known_overlap_freeze_refs
    if not {f.checked(ref) for ref in known_overlap_freeze_refs(project_root=f.root)}.issubset(bound):
        raise ValueError('Known source-overlap review/evidence is absent from frozen protocol')


def verified_execution_source(f,directory,plan,global_files,prediction_files):
    """Keep catalog JSON identity separate from its proven native execution file."""
    original={'path':plan['source_path'],'sha256':plan['source_sha256']}
    proof=plan.get('json_native_source_proof')
    is_bridge=plan.get('adapter_route')=='verified_json_native_pksim5'
    if not is_bridge:
        if proof is not None:raise ValueError('JSON native proof requires the registered bridge adapter')
        return original
    from recommendation_execution_repairs import json_native_source_proof as bridge
    if not isinstance(proof,dict):raise ValueError('JSON native bridge proof is missing')
    checked=bridge.verify_native_source({'source_path':original['path'],'source_sha256':original['sha256']},proof,project_root=f.root)
    execution={'path':plan['execution_source_path'],'sha256':plan['execution_source_sha256']}
    if checked['source_original']!=original or checked['native_source']!=execution:
        raise ValueError('Catalog or native execution source differs from bridge proof')
    matched=[r for r in checked['allowed_simulations'] if r['id']==plan['source_simulation_id']]
    if len(matched)!=1 or matched[0]['source_blob_sha256']!=plan['source_model_blob_sha256'] or matched[0]['name']!=plan['source_simulation']:
        raise ValueError('Selected native model is absent from source bridge proof')
    proof_path=directory/'json_native_source_proof.json'
    if proof_path not in global_files or proof_path not in prediction_files or f.read(proof_path)!=checked:
        raise ValueError('JSON native bridge proof must be present in both locks')
    protocol=f.read(directory.parent.parent/'protocol.json')
    for ref in checked['proof_refs']:
        f.checked(ref)
        matches=[r for r in protocol.get('inputs',[]) if r.get('path')==ref['path']]
        if matches!=[ref]:
            raise ValueError('JSON native bridge resource is not uniquely frozen in protocol')
    return execution


def verify_application_expansion(f,directory,plan,global_files,prediction_files):
    """Reprove original -> expanded input before permitting clinical outcomes."""
    from recommendation_execution_repairs import repeated_application as expansion
    execution_source=verified_execution_source(f,directory,plan,global_files,prediction_files)
    def locked(ref):
        path=f.checked(ref)
        if path not in global_files or path not in prediction_files:
            raise ValueError('Application expansion evidence must be present in both locks')
        return path
    receipt_path=directory/'repeated_application/receipt.json'
    if receipt_path not in global_files or receipt_path not in prediction_files:
        raise ValueError('Application expansion receipt must be present in both locks')
    receipt=f.read(receipt_path)
    from recommendation_execution_repairs import mixed_application_expansion as mixed
    if plan['application_expansion'].get('version')==mixed.VERSION:
        from recommendation_execution_repairs import historical_mixed_scoring as historical_mixed
        verified=historical_mixed.verify_expansion_receipt(f,plan,receipt_path,expected_policy=plan['application_expansion'])
        for name,digest in receipt['files'].items():locked({'path':name,'sha256':digest})
        locked(receipt['source_pkml']);locked(receipt['pkml'])
        exported=directory/'export/export_manifest.json';request=directory/'registered_native_request.json'
        if not {exported,request}.issubset(global_files & prediction_files):
            raise ValueError('Mixed expansion source and execution must appear in both locks')
        if (verified.get('accepted') is not True or
            f.read(exported).get('pkml')!=receipt['source_pkml'] or f.read(request).get('pkml')!=receipt['pkml'] or
            receipt['source_reference']!={**execution_source,
                'simulation_id':plan['source_simulation_id'],'source_blob_sha256':plan['source_model_blob_sha256']}):
            raise ValueError('Mixed expansion original/actual native input differs')
        return verified
    expected_source={**execution_source,
        'simulation_id':plan['source_simulation_id'],'source_blob_sha256':plan['source_model_blob_sha256']}
    if (receipt.get('version')!=expansion.VERSION or
        receipt.get('status')!='expanded_pending_native_readback_and_ode' or
        receipt.get('source_reference')!=expected_source or
        any(receipt.get(k) is not False for k in ('fitting_performed','ode_executed','sealed_observations_accessed','scientific_applicability_claimed'))):
        raise ValueError('Application expansion original source or outcome-free contract differs')
    for name in expansion.IMPLEMENTATION_FILES:
        if receipt.get('implementation_sha256',{}).get(name)!=plan['implementation_sha256'].get(name):
            raise ValueError('Application expansion implementation differs from frozen plan')
    files={locked({'path':name,'sha256':digest}) for name,digest in receipt['files'].items()}
    required={directory/'repeated_application'/name for name in ('expansion_plan.json','semantic_audit.json','source_audit.json')}
    if not required.issubset(files):raise ValueError('Application expansion audit set incomplete')
    source_path=locked(receipt['source_pkml']);expanded_path=locked(receipt['pkml'])
    export_path=directory/'export/export_manifest.json';request_path=directory/'registered_native_request.json'
    if not {export_path,request_path}.issubset(global_files & prediction_files):
        raise ValueError('Original export and expanded native request must be present in both locks')
    exported=f.read(export_path);request=f.read(request_path)
    if exported.get('pkml')!=receipt['source_pkml'] or request.get('pkml')!=receipt['pkml']:
        raise ValueError('Native execution is not bound to the audited expansion')
    changes=f.read(directory/'repeated_application/expansion_plan.json')
    contract=plan['contract']
    if (changes.get('compound_name')!=plan['native_compound_name'] or
        changes.get('administration_times_min')!=contract['administration_times_min'] or
        changes.get('dose_mg')!=contract['dose_mg'] or
        changes.get('event_prefix')!=plan['application_expansion']['event_prefix']):
        raise ValueError('Application expansion differs from frozen regimen')
    original=expansion._read_xml(source_path);expanded=expansion._read_xml(expanded_path)
    source_audit=expansion._verify_original_source(original,expected_source,f.root)
    audit=expansion.audit_expansion(original,expanded,changes)
    if (source_audit!=f.read(directory/'repeated_application/source_audit.json') or
        audit.get('accepted') is not True or audit!=f.read(directory/'repeated_application/semantic_audit.json')):
        raise ValueError('Application expansion independent preservation audit failed')


def bind_rederived_isolation_policy(f, plan, saved, rederived):
    """Accept old producer hashes only for an authenticated unchanged prediction.

    Re-derive every source/expanded-model proof with current code. All semantic
    fields must remain identical. Historical implementation hashes are then
    authenticated through the exact retained plan and its immutable archive;
    neither the policy nor its original digest is rewritten.
    """
    from recommendation_execution_repairs import coadministration_isolation as isolation
    if not isinstance(saved, dict) or not isinstance(rederived, dict):
        raise ValueError('Invalid isolation policy records')
    if saved == rederived:
        return saved
    if ({k:v for k,v in saved.items() if k!='implementation_sha256'} !=
            {k:v for k,v in rederived.items() if k!='implementation_sha256'}):
        raise ValueError('Re-derived isolation scientific proof differs')
    old = saved.get('implementation_sha256')
    new = rederived.get('implementation_sha256')
    required = set(isolation.IMPLEMENTATION_FILES)
    if not isinstance(old,dict) or not isinstance(new,dict) or set(old)!=required or set(new)!=required:
        raise ValueError('Isolation producer dependency set differs')
    from recommendation_execution_repairs.retained_implementations import binding_profile
    protocol=f.read(f.root/'experiments/model_recommendation/execution_validation_20261001/protocol.json')
    frozen={str(f.checked(ref).relative_to(f.root)):ref['sha256'] for ref in protocol['code']}
    code,_=binding_profile(plan,frozen,
        adapter_path='scripts/recommendation_execution_repairs/complete_candidate.py',
        current_required=complete_candidate.IMPLEMENTATION_FILES,project_root=f.root)
    if any(old[name]!=plan.get('implementation_sha256',{}).get(name) or old[name]!=code.get(name)
           or new[name]!=frozen.get(name) for name in required):
        raise ValueError('Isolation producer is not bound to the retained prediction')
    return saved


def verify_complete_native_binding(f,directory,plan,global_files,prediction_files):
    """Bind every complete-model prediction to the actual worker request/result.

    A correct registration and a different successful native run are not a
    valid chain, even if both artifacts were added to the two outer locks.
    This is an offline check; it does not run a solver or read observations.
    """
    from recommendation_execution_repairs import native_pkml
    execution_source=verified_execution_source(f,directory,plan,global_files,prediction_files) if plan.get('adapter_route')=='verified_json_native_pksim5' else None
    def locked(path):
        path=f.path(path)
        if path not in global_files or path not in prediction_files:
            raise ValueError('Actual native evidence must be present in both locks: '+str(path))
        return path
    def locked_ref(ref):
        return locked(f.checked(ref))
    registered_path=locked(directory/'registered_native_request.json')
    request_path=locked(directory/'native/request.json')
    raw_path=locked(directory/'native/native_result.json')
    result_path=locked(directory/'native/result.json')
    process_path=locked(directory/'native/process_manifest.json')
    artifact_lock_path=locked(directory/'native/artifact_lock.json')
    before=locked(directory/'native/before.pkml');after=locked(directory/'native/after.pkml')
    prediction_path=locked(directory/'prediction.json')
    registered=f.read(registered_path);request=f.read(request_path)
    raw=f.read(raw_path);result=f.read(result_path);process=f.read(process_path)
    native_lock=f.read(artifact_lock_path);prediction=f.read(prediction_path)
    source=locked_ref(registered['pkml']);source_sha=registered['pkml']['sha256']
    if (f.path(request['pkml_path'])!=source or request.get('pkml_sha256')!=source_sha or
        result.get('source_sha256')!=source_sha or raw.get('pkml_sha256')!=source_sha or
        raw.get('request_sha256')!=f.sha(request_path)):
        raise ValueError('Actual native input/request/result is not bound to the registered PKML')
    if (request.get('action')!='run' or result.get('action')!='run' or raw.get('action')!='run' or
        any(record.get('version')!=native_pkml.VERSION for record in (request,result,raw,native_lock)) or
        result.get('status')!='prediction_locked' or result.get('native')!=raw or
        type(raw.get('solver_calls')) is not int or raw['solver_calls']!=1 or
        any(record.get(key) is not False for record in (result,raw)
            for key in ('sealed_observations_accessed','fitting_performed'))):
        raise ValueError('Actual native execution/result contract is incomplete or inconsistent')
    if (native_lock.get('prediction_accepted') is not True or native_lock.get('source')!=registered['pkml']):
        raise ValueError('Native artifact lock is not bound to the registered prediction source')
    native_files={locked_ref({'path':name,'sha256':digest}) for name,digest in native_lock['files'].items()}
    if (not {request_path,raw_path,result_path,process_path,before,after}.issubset(native_files) or
        any(not p.is_relative_to(directory/'native') for p in native_files)):
        raise ValueError('Native artifact lock lacks the actual execution evidence')
    for key,expected in (('result_path',raw_path),('before_pkml_path',before),('after_pkml_path',after)):
        if f.path(request[key])!=expected:raise ValueError('Native output destination differs from this execution: '+key)
    for key in ('patches','output_path','output_times_min'):
        if request.get(key)!=registered.get(key):
            raise ValueError('Actual native request differs from registered '+key)
    if (registered.get('output_path')!=plan['output_path'] or
        registered.get('output_times_min')!=complete_candidate.output_grid(plan['contract']) or
        registered.get('molecular_weight_g_per_mol')!=plan['molecular_weight_g_per_mol'] or
        registered.get('source_metadata_sha256')!=plan['source_metadata_sha256'] or
        raw.get('curve',{}).get('output_path')!=plan['output_path']):
        raise ValueError('Native output/grid/parent differs from the frozen clinical plan')
    native_pkml.validate_request(request)
    from recommendation_execution_repairs import coadministration_isolation as isolation, event_associations, mixed_application_expansion as mixed
    policy=plan.get('coadministration_isolation')
    monitor_keys=('zero_monitor_paths','zero_monitor_dimensions','proof_parameter_requirements','positive_domain_monitor_dimensions')
    if policy:
        metadata=plan['source_metadata']
        if plan.get('source_event_association'):
            proof=event_associations.enrich_source_metadata((execution_source or {'path':plan['source_path']})['path'],
                (execution_source or {'sha256':plan['source_sha256']})['sha256'],
                plan['source_simulation_id'],plan['source_model_blob_sha256'],metadata=metadata,project_root=f.root)
            if proof!=plan['source_event_association']:raise ValueError('Original event association proof differs')
            metadata=proof['enriched_metadata']
        source_ref=policy['source_reference']
        rederived=isolation.prepare_source_isolation(source_ref['path'],source_ref['sha256'],
            source_ref['simulation_id'],source_ref['source_blob_sha256'],metadata,policy['target'],
            plan['contract'],project_root=f.root)
        expected_policy=bind_rederived_isolation_policy(f,plan,policy,rederived)
        if f.read(locked(directory/'coadministration_isolation.json'))!=expected_policy:
            raise ValueError('Frozen original isolation proof differs')
        expansion=plan.get('application_expansion') or {}
        if expansion.get('version')==mixed.VERSION:
            receipt=directory/'repeated_application/receipt.json'
            from recommendation_execution_repairs import historical_mixed_scoring as historical_mixed
            expanded,_=historical_mixed.executable_metadata(f,plan,receipt,plan['source_metadata'],expected_policy=expansion)
            rederived,metadata=historical_mixed.expanded_isolation(f,plan,expected_policy,receipt,expanded,plan['contract'])
            saved_expanded=f.read(locked(directory/'expanded_coadministration_isolation.json'))
            expected_policy=bind_rederived_isolation_policy(f,plan,saved_expanded,rederived)
        expected_monitors=isolation.monitor_request(expected_policy)
        if registered.get('coadministration_isolation')!=expected_policy or registered.get('isolation_monitor_request')!=expected_monitors:
            raise ValueError('Actual isolation policy is not the independently verified one')
        if any(request.get(k)!=expected_monitors[k] for k in monitor_keys):
            raise ValueError('Actual native monitoring omits the original zero-exposure proof requirements')
        complete_candidate.isolation_readback(raw,expected_policy)
    elif any(request.get(k) for k in monitor_keys):
        raise ValueError('Unexpected unregistered isolation monitors')
    native_pkml.validate_native_output(raw,request)
    if raw.get('before_pkml_sha256')!=f.sha(before) or raw.get('after_pkml_sha256')!=f.sha(after):
        raise ValueError('Native serialized model hashes do not match the locked files')
    code=plan['implementation_sha256'];worker=native_pkml.WORKER
    adapter='scripts/recommendation_execution_repairs/native_pkml.py'
    if (result.get('worker_sha256')!=code.get(worker) or result.get('python_sha256')!=code.get(adapter) or
        not code.get(worker) or not code.get(adapter) or process.get('script_sha256')!=code[worker] or
        f.path(process['script'])!=f.path(worker) or f.path(result['process_manifest'])!=process_path or
        process.get('status')!='completed' or type(process.get('returncode')) is not int or process['returncode']!=0 or
        process.get('runtime_unchanged') is not True or process.get('registry_modified') is not False):
        raise ValueError('Native process does not match the frozen worker/runtime binding')
    command=process.get('command',[])
    if (len(command)!=5 or command[1]!='--vanilla' or f.path(command[2])!=f.path(worker) or
        f.path(command[3])!=request_path or command[4]!=f.sha(request_path)):
        raise ValueError('Recorded native process did not receive this exact request')
    if (result.get('source_import_audit',{}).get('accepted') is not True or
        result.get('model_preservation_audit',{}).get('accepted') is not True):
        raise ValueError('Complete model prediction lacks both preservation audits')
    exported=f.read(locked(directory/'export/export_manifest.json'))
    if execution_source is not None:
        proof_path=locked(directory/'json_native_source_proof.json')
        if (f.read(proof_path)!=plan['json_native_source_proof'] or exported.get('source')!=execution_source
                or exported.get('simulation_id')!=plan['source_simulation_id']
                or exported.get('source_blob_sha256')!=plan['source_model_blob_sha256']):
            raise ValueError('Exported native model differs from the catalog JSON bridge')
    if not plan.get('application_expansion') and registered['pkml']!=exported.get('pkml'):
        raise ValueError('Unexpanded native input differs from the original model export')
    if execution_source is not None:
        from recommendation_execution_repairs.json_native_scoring import verify_bridge_semantics
        verify_bridge_semantics(f,directory,plan,registered,before,global_files,prediction_files)
    if plan.get('split_absorption_contract') is not None:
        from recommendation_execution_repairs.json_native_scoring import verify_split_semantics
        verify_split_semantics(f,directory,plan,registered,raw,global_files,prediction_files)
    expected=complete_candidate.prediction_from_curve(complete_candidate.normalized_molar_curve(raw['curve']),plan)
    # The outer adapter changes only the version and adds source/receipt links.
    # Recompute all scientific curve transformations and endpoint estimates.
    for key,value in expected.items():
        if key!='version' and prediction.get(key)!=value:
            raise ValueError('Locked prediction is not derived from the actual native curve: '+key)
    return {'accepted':True,'actual_native_request_sha256':f.sha(request_path),
            'actual_native_result_sha256':f.sha(raw_path),'source_pkml_sha256':source_sha,
            'curve_and_endpoints_recomputed':True,'clinical_outcomes_accessed':False}


def verify_snapshot_native_binding(f,directory,plan,global_files,prediction_files):
    """Recheck the actual JSON input, native run/readback, and every curve point."""
    from pbpk_kb.blueprints import design_runtime, source_adaptation_audit as audit
    def locked(value):
        p=f.path(value)
        if p not in global_files or p not in prediction_files:
            raise ValueError('Snapshot native evidence must be present in both locks: '+str(p))
        if not p.is_relative_to(directory):raise ValueError('Snapshot native evidence belongs to another job')
        return p
    def locked_ref(ref):return locked(f.checked(ref))
    def equal(actual,expected,label):
        if backend.digest(actual)!=backend.digest(expected):
            raise ValueError('Snapshot actual native chain differs: '+label)
    native=directory/'native';readback=directory/'readback'
    input_path=locked(native/'snapshot.json');manifest_path=locked(native/'execution_manifest.json')
    process_path=locked(native/'process_manifest.json');request_path=locked(readback/'request.json')
    raw_path=locked(readback/'native_readback.json');audit_path=locked(readback/'audit.json')
    audit_process_path=locked(readback/'process_manifest.json')
    outer_audit_path=locked(directory/'readback_result.json');prediction_path=locked(directory/'prediction.json')
    manifest=f.read(manifest_path);process=f.read(process_path);request=f.read(request_path)
    raw=f.read(raw_path);report=f.read(audit_path);audit_process=f.read(audit_process_path)
    prediction=f.read(prediction_path);native_snapshot=f.read(input_path)
    equal(native_snapshot,plan['snapshot'],'registered snapshot versus executed input')
    if (backend.digest(native_snapshot)!=plan['snapshot_sha256'] or
        manifest.get('source_sha256')!=f.sha(input_path) or
        manifest.get('source_unchanged') is not True or manifest.get('engineering_accepted') is not True or
        manifest.get('status')!='engineering_passed' or manifest.get('registry_modified') is not False or
        f.path(manifest['manifest_path'])!=manifest_path):
        raise ValueError('Snapshot native manifest does not bind the unchanged successful input')
    f.checked({'path':plan['source_path'],'sha256':plan['source_sha256']})
    f.checked({'path':plan['execution_source_path'],'sha256':plan['execution_source_sha256']})
    equal(manifest.get('process'),process,'embedded execution process')
    equal(report.get('process'),audit_process,'embedded readback process')
    equal(f.read(outer_audit_path),report,'saved readback result')
    equal(manifest.get('runtime_binding'),process.get('runtime_binding'),'execution runtime')
    equal(audit_process.get('runtime_binding'),process.get('runtime_binding'),'readback runtime')
    binding=process.get('runtime_binding') or {}
    resources=binding.get('locked_resources') or []
    if not resources:raise ValueError('Snapshot native runtime lacks locked resources')
    expected_runtime=getattr(f,'_snapshot_native_runtime_binding',None)
    if expected_runtime is None:
        expected_runtime=design_runtime._runtime_binding(f.root)
        f._snapshot_native_runtime_binding=expected_runtime
    equal(binding,expected_runtime,'registered native runtime resources and engine')
    runtime_paths={f.path(ref['path']) for ref in resources}
    rscript=f.path('runtime/ospsuite-env/bin/Rscript')
    if rscript not in runtime_paths:raise ValueError('Snapshot native runtime omits its Rscript')
    for record,worker,arguments,path in (
        (process,design_runtime._RUNNER,[str(input_path),str(native/'outputs')],process_path),
        (audit_process,audit._R_SCRIPT,[str(request_path),f.sha(request_path)],audit_process_path)):
        worker_path=f.path(worker);expected_sha=plan['implementation_sha256'].get(worker)
        if (not expected_sha or record.get('script_sha256')!=expected_sha or
            f.path(record['script'])!=worker_path or record.get('status')!='completed' or
            type(record.get('returncode')) is not int or record['returncode']!=0 or
            record.get('runtime_unchanged') is not True or record.get('registry_modified') is not False or
            record.get('stop_reason') is not None or f.path(record['process_manifest_path'])!=path):
            raise ValueError('Snapshot native worker/process binding differs')
        f.checked({'path':worker,'sha256':expected_sha})
        equal(record.get('command'),[str(rscript),'--vanilla',str(worker_path),*arguments],
              'native worker command/input')
    pkmls=manifest.get('pkml_outputs',[]);curves=manifest.get('curve_outputs',[])
    if len(pkmls)!=1 or len(curves)!=1:raise ValueError('Exactly one native snapshot model and curve required')
    def output(ref):
        p=locked_ref({'path':str(native/ref['path']),'sha256':ref['sha256']})
        if not p.is_relative_to(native/'outputs'):raise ValueError('Snapshot native output outside this run')
        return p
    pkml=output(pkmls[0]);csv=output(curves[0])
    expected_names=[s['Name'] for s in native_snapshot['Simulations']]
    equal(manifest.get('expected_simulations'),expected_names,'native simulation identity')
    if len(expected_names)!=1:raise ValueError('Exactly one selected snapshot simulation required')
    equal(native_snapshot['Simulations'][0].get('OutputSelections'),[plan['output_path']],
          'source native observation output')
    expected_checks=audit.build_expectations(native_snapshot,plan['native_expectations']['expectations'])
    equal(request.get('expectations'),expected_checks,'frozen parameter and regimen readback')
    if (request.get('adapter_version')!=audit.AUDIT_VERSION or
        request.get('source_sha256')!=f.sha(input_path) or request.get('pkml_sha256')!=f.sha(pkml) or
        f.path(request['pkml_path'])!=pkml or f.path(request['result_path'])!=raw_path or
        request.get('expected_event_count')!=plan['native_expectations']['expected_event_count'] or
        raw.get('request_sha256')!=f.sha(request_path) or
        report.get('source_sha256')!=f.sha(input_path) or report.get('pkml_sha256')!=f.sha(pkml) or
        report.get('request_sha256')!=f.sha(request_path) or
        report.get('execution_manifest_sha256')!=f.sha(manifest_path) or
        f.path(report['execution_manifest'])!=manifest_path or
        report.get('native_readback_sha256')!=f.sha(raw_path) or
        f.path(report['native_readback_file'])!=raw_path or
        report.get('accepted') is not True or report.get('status')!='passed' or
        report.get('solver_calls')!=0 or report.get('model_modified') is not False):
        raise ValueError('Snapshot actual readback is not bound to this source/run/request')
    reassessed=audit.assess_native_readback(raw,request)
    if reassessed.get('accepted') is not True:raise ValueError('Snapshot independent native readback failed')
    for key,value in reassessed.items():equal(report.get(key),value,'readback '+key)
    curve=snapshot_candidate.read_curve(native,manifest,plan['output_path'])
    expected=snapshot_candidate.prediction_from_curve(curve,plan)
    # All native output points, sampled points, units, matrix, registration and
    # scalar values are checked, including required null-valued fields.
    equal(prediction,expected,'full native curve and prediction transformation')
    return {'accepted':True,'actual_native_input_sha256':f.sha(input_path),
            'execution_manifest_sha256':f.sha(manifest_path),'curve_sha256':f.sha(csv),
            'curve_and_endpoints_recomputed':True,'clinical_outcomes_accessed':False}


def _emit_verification_progress(pass_number, stage, done, total, job_id=None):
    """Best-effort bounded metadata only; logging cannot change a proof result."""
    try:
        print(json.dumps({'pass': pass_number, 'stage': stage, 'done': done,
                          'total': total, 'job_id': job_id}, separators=(',', ':')), flush=True)
    except (OSError, ValueError, TypeError):
        # A closed stdout must not bypass or replace scientific verification.
        pass


@contextmanager
def repaired_execution_profile():
    # The old scorer used a module-level revision rather than an argument.
    # Keep that configuration local to this single-threaded scoring operation;
    # original files and prior scientific/statistical rules remain untouched.
    with _LOCK:
        old_revision=backend.EXECUTION_REVISION
        old_binding=backend.verify_execution_binding
        old_barrier=backend.verify_barrier
        old_preparation_binding=backend.verify_preparation_binding
        _verification_progress_pass=0
        def verify_binding(record,**kwargs):
            from recommendation_execution_repairs.retained_implementations import binding_profile
            options=dict(kwargs)
            if kwargs.get('expected_revision')!=REVISION:
                raise ValueError('Only the registered repair revision can use this profile')
            implementation=record.get('implementation_sha256',{})
            # Snapshot imports the complete adapter as a helper. Select its own
            # adapter before authenticating historical dependency declarations.
            if record.get('adapter_route') in {'verified_json_native_pksim5','blocked_json_native_bridge'} or 'scripts/recommendation_execution_repairs/json_native_candidate.py' in implementation:
                from recommendation_execution_repairs import json_native_candidate
                route=record.get('adapter_route')
                if route not in {'verified_json_native_pksim5','blocked_json_native_bridge'}:
                    raise ValueError('JSON native adapter marker requires its registered route')
                if route=='blocked_json_native_bridge':
                    if (record.get('status')!='blocked' or record.get('version')!=json_native_candidate.VERSION
                            or any(record.get(name) is not None and record.get(name) is not False
                                   for name in ('native_executed','prediction_locked'))
                            or any(name in record for name in ('prediction_file','predicted_endpoints','prediction_lock'))):
                        raise ValueError('Blocked bridge cannot carry an executed or locked prediction')
                adapter='scripts/recommendation_execution_repairs/json_native_candidate.py'
                required=set(json_native_candidate.IMPLEMENTATION_FILES)
            elif 'scripts/recommendation_execution_repairs/snapshot_candidate.py' in implementation:
                if record.get('adapter_route') == 'complete_original_pksim5':
                    raise ValueError('Complete-model route conflicts with snapshot executor identity')
                adapter='scripts/recommendation_execution_repairs/snapshot_candidate.py'
                required=set(snapshot_candidate.IMPLEMENTATION_FILES)
            elif 'scripts/recommendation_execution_repairs/complete_candidate.py' in implementation:
                adapter='scripts/recommendation_execution_repairs/complete_candidate.py'
                required=set(complete_candidate.IMPLEMENTATION_FILES)
            else:
                raise ValueError('Repaired execution lacks a registered adapter')
            options['frozen_code_sha256'],required=binding_profile(record,options['frozen_code_sha256'],
                adapter_path=adapter,current_required=required)
            result=old_binding(record,**options)
            if not required.issubset(result):raise ValueError('Repaired executor dependencies are incomplete')
            return result
        def verify_preparation_binding(f,run,protocol,frozen_code_sha256):
            from recommendation_execution_repairs.historical_preparation import verify_preparation_binding as verify_history
            return verify_history(f,run,protocol,frozen_code_sha256)
        def verify_barrier(run_dir,**kwargs):
            nonlocal _verification_progress_pass
            _verification_progress_pass+=1
            _verification_progress_done=0
            # This registered experiment has 121 locked predictions; use the
            # actual verified bundle count as soon as the base barrier returns.
            _verification_progress_total=121
            _emit_verification_progress(_verification_progress_pass,'base_before',
                _verification_progress_done,_verification_progress_total)
            bundle=old_barrier(run_dir,**kwargs)
            _verification_progress_total=sum(bool(job['result'].get('prediction_locked'))
                for job in bundle['native'].values())
            _emit_verification_progress(_verification_progress_pass,'base_after',
                _verification_progress_done,_verification_progress_total)
            f=bundle['files'];run=bundle['run']
            # The base barrier has verified its own format but performs no
            # outcome read here. Close access until the added repair proofs are
            # independently bound to both immutable prediction locks.
            f.unsealed=False
            global_lock=f.read(run/'native/predictions_lock.json')
            global_jobs={row['job_id']:row for row in global_lock['jobs']}
            for job in bundle['native'].values():
                result,plan=job['result'],job['plan']
                if not result.get('prediction_locked'):continue
                _emit_verification_progress(_verification_progress_pass,'native_before',
                    _verification_progress_done,_verification_progress_total,job['job_id'])
                directory=run/'native'/job['job_id']
                global_files={f.checked(ref) for ref in global_jobs[job['job_id']]['artifacts']}
                plock=f.read(f.checked({'path':result['prediction_lock'],'sha256':result['prediction_lock_sha256']}))
                prediction_files={f.checked({'path':name,'sha256':digest}) for name,digest in plock['files'].items()}
                if plan.get('adapter_route') in {'complete_original_pksim5','verified_json_native_pksim5'}:
                    audit_path=directory/'native/result.json'
                    global_files={f.checked(ref) for ref in global_jobs[job['job_id']]['artifacts']}
                    plock=f.read(f.checked({'path':result['prediction_lock'],'sha256':result['prediction_lock_sha256']}))
                    prediction_files={f.checked({'path':name,'sha256':digest}) for name,digest in plock['files'].items()}
                    if audit_path not in global_files or audit_path not in prediction_files:
                        raise ValueError('Complete model preservation proof must be present in both locks')
                    native=f.read(audit_path)
                    if (native.get('status')!='prediction_locked' or
                        native.get('source_import_audit',{}).get('accepted') is not True or
                        native.get('model_preservation_audit',{}).get('accepted') is not True):
                        raise ValueError('Complete model prediction lacks both preservation audits')
                    if plan.get('application_expansion'):
                        verify_application_expansion(f,directory,plan,global_files,prediction_files)
                    verify_complete_native_binding(f,directory,plan,global_files,prediction_files)
                else:
                    verify_snapshot_native_binding(f,directory,plan,global_files,prediction_files)
                _verification_progress_done+=1
                _emit_verification_progress(_verification_progress_pass,'native_after',
                    _verification_progress_done,_verification_progress_total,job['job_id'])
            _emit_verification_progress(_verification_progress_pass,'engineering_before',
                _verification_progress_done,_verification_progress_total)
            verify_engineering_barrier(bundle)
            _emit_verification_progress(_verification_progress_pass,'engineering_after',
                _verification_progress_done,_verification_progress_total)
            f.unsealed=True
            return bundle
        backend.EXECUTION_REVISION=REVISION
        backend.verify_execution_binding=verify_binding
        backend.verify_barrier=verify_barrier
        backend.verify_preparation_binding=verify_preparation_binding
        try:yield
        finally:
            backend.EXECUTION_REVISION=old_revision
            backend.verify_execution_binding=old_binding
            backend.verify_barrier=old_barrier
            backend.verify_preparation_binding=old_preparation_binding


def score_run(run_dir,**kwargs):
    with repaired_execution_profile():
        return backend.score_run(run_dir,**kwargs)
