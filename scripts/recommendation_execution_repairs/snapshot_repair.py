"""Source-bound, exact-field repair of legacy native JSON snapshots.

Revalidate authenticated native exports where possible. New conversion is
bounded, never runs ODEs and never reads clinical outcomes. A failed repair is
an operational status, not a scientific negative recommendation label.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import recommendation_candidate_migration as migration
from pbpk_kb.blueprints.design_runtime import run_bounded_r
from .snapshot_schema import map_legacy_expression_schema, OFFICIAL_FILES
from .snapshot_overlay import restore_explicit_simulation_fields
from .snapshot_readback import validate_overlay, OFFICIAL_FILES as READBACK_OFFICIAL_FILES
from .complex_snapshot import repair_complex_snapshot, IMPLEMENTATION_FILES as COMPLEX_IMPLEMENTATION_FILES
from .native_partition_resume import (collect_completed_partitions, resume_partition,
                                     IMPLEMENTATION_FILES as RESUME_IMPLEMENTATION_FILES)

VERSION='source-bound-exact-snapshot-repair-v1'
DEFAULT_REPAIRS=Path(migration.OUT)/'engineering_repairs/snapshot_repairs_11'
R_SCRIPT=migration.R_SCRIPT
IMPLEMENTATION_FILES=[
    'scripts/recommendation_execution_repairs/historical_source_proofs.py',
    'scripts/recommendation_execution_repairs/native_build_cache.py',
    'scripts/recommendation_execution_repairs/snapshot_repair.py',
    'scripts/recommendation_execution_repairs/snapshot_schema.py',
    'scripts/recommendation_execution_repairs/snapshot_overlay.py',
    'scripts/recommendation_execution_repairs/snapshot_readback.py',
    'scripts/recommendation_execution_repairs/source_pkml.py',
    'scripts/recommendation_candidate_migration.py',
    'src/pbpk_kb/blueprints/design_runtime.py',
    'src/pbpk_kb/blueprints/source_adaptation.py',R_SCRIPT,*OFFICIAL_FILES,*READBACK_OFFICIAL_FILES]
IMPLEMENTATION_FILES=sorted(set(IMPLEMENTATION_FILES+COMPLEX_IMPLEMENTATION_FILES+list(RESUME_IMPLEMENTATION_FILES)))
inside,sha,save,reference,checked=(migration.inside,migration.sha,migration.save,
                                  migration.reference,migration.checked)


def exact_comparison(source,converted):
    result=migration.compare_migration(source,converted)
    result['accepted']=result['accepted'] and migration._normalize(migration.clean_snapshot(source))==migration._normalize(migration.clean_snapshot(converted))
    result['numeric_tolerance']={'relative':0,'absolute':0}
    result['exact_normalized_document_equality']=result['accepted']
    if not result['accepted'] and not result['differences']:
        result['differences']=[{'kind':'exact_numeric_identity_failure'}]
        result['difference_count']=1
    return result


def task_for(provenance,root):
    source=inside(provenance['source_path'],root)
    if source.suffix!='.json' or source.stat().st_size>32*1024**2:raise ValueError('BOUNDED_JSON_SOURCE_REQUIRED')
    digest=provenance['source_sha256']
    if sha(source)!=digest:raise ValueError('ORIGINAL_SOURCE_SHA_MISMATCH')
    return {'source_path':str(source.relative_to(root)),'source_sha256':digest,'format':'.json'}


def merge_native_partitions(original,partitions):
    """Lossless union; no averaging, fallback parameter or majority decision."""
    expected=[s['Name'] for s in original['Simulations']]
    if len(expected)!=len(set(expected)) or not partitions:raise ValueError('UNIQUE_NONEMPTY_NATIVE_PARTITIONS_REQUIRED')
    root_document=None;simulations={}
    for raw in partitions:
        converted=migration.clean_snapshot(raw)
        if converted.get('Version')!=80:raise ValueError('ALL_PARTITIONS_REQUIRE_NATIVE_VERSION_80')
        values=converted.pop('Simulations')
        if root_document is None:root_document=converted
        elif not exact_comparison(root_document,converted)['accepted']:
            raise ValueError('NATIVE_PARTITION_GLOBAL_DEFINITIONS_DIFFER')
        for simulation in values:
            name=simulation['Name']
            if name in simulations or name not in expected:raise ValueError('NATIVE_PARTITION_SIMULATION_IDENTITY_DIFFERS')
            simulations[name]=simulation
    if set(simulations)!=set(expected):raise ValueError('NATIVE_PARTITION_COVERAGE_INCOMPLETE')
    return {**root_document,'Simulations':[simulations[name] for name in expected]}


def _native_stage(mode,input_path,directory,*,root,timeout_seconds,cpu_threads,stop_file):
    if stop_file and stop_file.exists():raise InterruptedError('Snapshot repair stop requested')
    if shutil.disk_usage(root).free<20*1024**3:raise OSError('Snapshot repair requires 20 GiB headroom')
    if mode=='snapshot_to_project':
        from .native_build_cache import reuse_build
        reused=reuse_build(input_path,directory,root=root)
        if reused is not None:return reused
    directory.mkdir();inp=directory/'input';out=directory/'output';inp.mkdir();out.mkdir()
    original=reference(input_path,root);local=inp/input_path.name;shutil.copyfile(input_path,local)
    before=sha(local)
    if before!=original['sha256']:raise ValueError('NATIVE_INPUT_COPY_MISMATCH')
    process=run_bounded_r(script=root/R_SCRIPT,args=[mode,str(inp),str(out)],output_dir=directory,
        project_root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,
        max_output_bytes=512*1024**2,stop_file=stop_file)
    receipt={'input':original,'copy_before_sha256':before,'copy_after_sha256':sha(local),
        'original_after_sha256':sha(input_path),'process':reference(directory/'process_manifest.json',root),
        'mode':mode,'status':process['status'],'stop_reason':process.get('stop_reason')}
    save(directory/'stage_receipt.json',receipt)
    if receipt['original_after_sha256']!=original['sha256']:raise ValueError('NATIVE_STAGE_ORIGINAL_CHANGED')
    if input_path.suffix=='.json' and receipt['copy_after_sha256']!=before:raise ValueError('NATIVE_JSON_COPY_CHANGED')
    if process.get('status')!='completed' or process.get('returncode')!=0:
        raise ValueError('NATIVE_STAGE_FAILED:'+mode+':'+str(process.get('stop_reason')))
    outputs=list(out.glob('*.pksim5' if mode=='snapshot_to_project' else '*.json'))
    if len(outputs)!=1:raise ValueError('ONE_NATIVE_STAGE_OUTPUT_REQUIRED')
    # These are newly created private copies/caches, never source artifacts.
    # Keep outputs, original-input refs, manifests and logs for verification.
    removed=[]
    for expendable in (inp,directory/'temporary'):
        if expendable.is_dir() and not expendable.is_symlink():
            removed.append(str(expendable.relative_to(root)));shutil.rmtree(expendable)
    receipt['own_intermediate_directories_removed']=removed
    save(directory/'stage_receipt.json',receipt,replace=True)
    return outputs[0],[reference(directory/'stage_receipt.json',root),receipt['process'],reference(outputs[0],root)]


def _native_base(task,directory,*,root,timeout_seconds,cpu_threads,stop_file):
    """Use only completed, bound native artifacts; never trust a filename."""
    refs=[]
    try:
        prior=migration.revalidate_old_artifact(task,project_root=root)
        refs+=prior['bindings']
        if 'converted' in prior:
            return prior['converted'],refs,'authenticated_prior_native_export'
    except (KeyError,FileNotFoundError) as exc:
        # Supplemental licensed candidates may have no old attempted migration.
        prior=None
    except ValueError as exc:
        if str(exc)!='EXACTLY_ONE_ARCHIVED_SOURCE_REQUIRED':raise
        # The old helper uses this message for zero OR multiple records.
        # Only actual absence permits new conversion; duplicates are not absence.
        archive=json.loads(inside(migration.ARCHIVE,root).read_text())
        entries=[r for r in archive['files'] if r['original_path']==migration.OLD_MIGRATIONS+'/migration_summary.json']
        if len(entries)!=1:raise ValueError('ARCHIVED_SUMMARY_IDENTITY_AMBIGUOUS')
        summary=json.loads(checked({'path':entries[0]['archive_path'],'sha256':entries[0]['sha256']},root).read_text())
        if any(r['source_sha256']==task['source_sha256'] for r in summary['results']):raise
        prior=None
    v3=root/migration.DEFAULT_MIGRATIONS/task['source_sha256']/'receipt.json'
    if v3.exists():
        r=json.loads(v3.read_text());migration.verify_receipt_code(r,root)
        if r.get('source_sha256')!=task['source_sha256'] or r.get('source_path')!=task['source_path']:
            raise ValueError('PRIOR_NATIVE_SOURCE_BINDING_MISMATCH')
        refs.append(reference(v3,root))
        # Even a comparison failure may have a valid completed native export;
        # accepting the artifact is explicitly NOT accepting the comparison.
        stages=r.get('stages',[])
        if len(stages) in (1,2) and stages[-1].get('mode')=='project_to_snapshot' and stages[-1].get('output'):
            for ref in r.get('provenance_refs',[]):checked(ref,root)
            refs+=r.get('provenance_refs',[])
            for stage in stages:
                pp=checked(stage['process_manifest'],root);process=json.loads(pp.read_text())
                if (process.get('status')!='completed' or process.get('returncode')!=0 or
                        process.get('script_sha256')!=sha(root/R_SCRIPT) or process.get('runtime_unchanged') is not True):
                    raise ValueError('PRIOR_NATIVE_STAGE_NOT_COMPLETED')
                refs.extend([reference(pp,root),stage['output']]);checked(stage['output'],root)
                if stage['input']['sha256']!=stage.get('source_input_sha256_after'):
                    raise ValueError('PRIOR_NATIVE_ORIGINAL_DRIFT')
            original=migration.clean_snapshot(json.loads(inside(task['source_path'],root).read_text()))
            if len(stages)==2:
                if stages[0]['mode']!='snapshot_to_project' or stages[1]['input']!=stages[0]['output']:
                    raise ValueError('PRIOR_NATIVE_STAGES_UNBOUND')
                first=checked(stages[0]['input'],root)
                if json.loads(first.read_text())!=original:
                    raise ValueError('PRIOR_NATIVE_INPUT_DIFFERS_FROM_SOURCE_SOLVER_DOCUMENT')
                refs.append(reference(first,root))
            else:
                if (prior is None or r.get('continued_from')!='archived_completed_snapshot_to_project_stage' or
                        r.get('prior_receipt')!=prior['receipt_reference']):
                    raise ValueError('PRIOR_SINGLE_EXPORT_CONTINUATION_UNBOUND')
                old=prior['directory'];old_input=inside(old/'input/model.json',root);old_log=inside(old/'execution.log',root)
                if (json.loads(old_input.read_text())!=original or '\nNULL\n' not in old_log.read_text() or
                        checked(stages[0]['input'],root)!=old/'project/model.pksim5'):
                    raise ValueError('PRIOR_COMPLETED_BUILD_SOURCE_IDENTITY_DIFFERS')
                refs.extend([reference(old_input,root),reference(old_log,root)])
            converted=json.loads(checked(stages[-1]['output'],root).read_text())
            return converted,refs,'completed_prior_v3_native_export_revalidated'
    # No completed chain remains. Partition only the explicit simulation list;
    # every source template/parameter is retained in every input. A merge is
    # allowed only when all exported global definitions are exactly identical.
    original=migration.clean_snapshot(json.loads(inside(task['source_path'],root).read_text()))
    if not 1<=len(original.get('Simulations',[]))<=256:raise ValueError('SOURCE_SIMULATION_COUNT_BOUND_1_TO_256')
    # Nine-compound mechanistic sources exceeded 600 seconds for a four-model
    # build. Reduce only the amount constructed at once; preserve every source
    # simulation and every global definition, and retain strict union checks.
    batch_size=1 if len(original.get('Compounds',[]))>8 else 4
    prior_directories=[root/migration.OUT/'engineering_repairs'/name/task['source_sha256']
                       for name in ('snapshot_repairs_08','snapshot_repairs_07','snapshot_repairs_06','snapshot_repairs_05','native_partition_preflight','snapshot_repairs_partitioned','snapshot_repairs_600s','snapshot_repairs')]
    prior_directories=[p for p in prior_directories if p.is_dir()]
    resumed=collect_completed_partitions(task,prior_directories,project_root=root)
    save(directory/'prior_partition_audit.json',resumed)
    refs.append(reference(directory/'prior_partition_audit.json',root))
    partitions=[];started=time.monotonic()
    for index in range(0,len(original['Simulations']),batch_size):
        if time.monotonic()-started>10800:raise TimeoutError('SOURCE_NATIVE_PREPARATION_10800_SECOND_LIMIT')
        part=directory/('partition_%03d'%(index//batch_size+1));part.mkdir()
        scoped=deepcopy(original);scoped['Simulations']=scoped['Simulations'][index:index+batch_size]
        model_only=part/'source_model_only.json';save(model_only,scoped);refs.append(reference(model_only,root))
        names=[s['Name'] for s in scoped['Simulations']]
        matches=[p for p in resumed['partitions'] if p['simulation_names']==names]
        if matches:
            reuse=resume_partition(matches[0],part/'authenticated_resume',stage_runner=_native_stage,
                project_root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            native=reuse['native_document'];refs+=reuse['provenance_refs']
        else:
            project,stage_refs=_native_stage('snapshot_to_project',model_only,part/'build',
                root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            refs+=stage_refs
            snapshot,stage_refs=_native_stage('project_to_snapshot',project,part/'export',root=root,
                timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            refs+=stage_refs;native=json.loads(snapshot.read_text())
        if {s['Name'] for s in native.get('Simulations',[])}!={s['Name'] for s in scoped['Simulations']}:
            raise ValueError('NATIVE_PARTITION_REQUEST_COVERAGE_DIFFERS')
        partitions.append(native)
    converted=merge_native_partitions(original,partitions)
    save(directory/'partitioned_native_snapshot.json',converted)
    refs.append(reference(directory/'partitioned_native_snapshot.json',root))
    return converted,refs,'partitioned_bounded_native_export_full_coverage_no_ode'


def _verify_receipt(receipt_path,provenance,root,*,require_accepted=True):
    r=json.loads(receipt_path.read_text());task=task_for(provenance,root)
    if r.get('version')!=VERSION or any(r.get(k)!=v for k,v in task.items()):
        raise ValueError('SNAPSHOT_REPAIR_SOURCE_BINDING_MISMATCH')
    if r.get('ode_run') is not False or r.get('sealed_observations_accessed') is not False or r.get('fitting_performed') is not False:
        raise ValueError('SNAPSHOT_REPAIR_SCOPE_DIFFERS')
    expected=[reference(root/p,root) for p in IMPLEMENTATION_FILES]
    verifier=lambda ref:checked(ref,root)
    if r.get('implementation')!=expected:
        from .historical_source_proofs import historical_checker
        try:verifier=historical_checker(receipt_path,r,root)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            raise ValueError('SNAPSHOT_REPAIR_IMPLEMENTATION_DRIFT') from exc
    for ref in r.get('provenance_refs',[]):verifier(ref)
    if require_accepted or r.get('status')=='verified':
        if (r.get('status')!='verified' or r.get('comparison',{}).get('accepted') is not True or
                r['comparison'].get('exact_normalized_document_equality') is not True):
            raise ValueError('SNAPSHOT_REPAIR_NOT_VERIFIED:'+str(r.get('reasons')))
        checked(r['snapshot'],root)
    return r


def repair_snapshot(provenance,output_dir=None,*,project_root=ROOT,timeout_seconds=180,cpu_threads=1,stop_file=None):
    root=Path(project_root).resolve(strict=True);task=task_for(provenance,root)
    if type(timeout_seconds)!=int or not 1<=timeout_seconds<=600:raise ValueError('MAXIMUM_STAGE_TIMEOUT_600_SECONDS')
    if cpu_threads not in (1,2):raise ValueError('ONE_OR_TWO_CPU_THREADS_REQUIRED')
    if stop_file:stop_file=inside(stop_file,root,exists=False)
    if stop_file and stop_file.exists():raise InterruptedError('Snapshot repair stop requested before creation')
    base=inside(output_dir or root/DEFAULT_REPAIRS/task['source_sha256'],root,exists=False)
    if base.exists():
        return _verify_receipt(base/'receipt.json',provenance,root,require_accepted=False)
    from .historical_source_proofs import original_receipt
    historical=original_receipt(provenance,root)
    if historical is not None:
        result=_verify_receipt(historical,provenance,root)
        base.mkdir(parents=True)
        shutil.copyfile(historical,base/'receipt.json')
        return result
    base.mkdir(parents=True)
    result={'version':VERSION,**task,'status':'failed','reasons':[],
        'implementation':[reference(root/p,root) for p in IMPLEMENTATION_FILES],
        'provenance_refs':[],'created_at':datetime.now(timezone.utc).isoformat(),
        'ode_run':False,'sealed_observations_accessed':False,'fitting_performed':False,
        'scientific_inapplicability_inferred':False,'numeric_tolerance':0}
    try:
        if stop_file and stop_file.exists():raise InterruptedError('Snapshot repair stop requested')
        original=migration.clean_snapshot(json.loads(inside(task['source_path'],root).read_text()))
        if original.get('Version')==80:
            converted=original;result['operation']='already_native_version_80'
            result['comparison']=exact_comparison(original,converted)
        else:
            native,refs,operation=_native_base(task,base,root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            result['provenance_refs']=refs;result['operation']=operation
            mapped=map_legacy_expression_schema(original,native)
            result['schema_changes']=mapped['changes']
            force_model_proof=mapped.get('requires_complete_native_model_proof') is True
            result['requires_complete_native_model_proof']=force_model_proof
            result['comparison']=exact_comparison(mapped['mapped_snapshot'],native)
            converted=migration.clean_snapshot(native)
            if not result['comparison']['accepted'] or force_model_proof:
                overlay=restore_explicit_simulation_fields(mapped['mapped_snapshot'],native)
                result['overlay_audit']={k:v for k,v in overlay.items() if k!='snapshot'}
                overlay_verified=False
                if overlay['snapshot'] is not None and not force_model_proof:
                    save(base/'restored_snapshot_pending_readback.json',overlay['snapshot'])
                    result['pending_snapshot']=reference(base/'restored_snapshot_pending_readback.json',root)
                    try:
                        proof,refs=validate_overlay(overlay,base/'native_readback',root=root,stage_runner=_native_stage,
                            timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
                        if proof.get('accepted') is not True:raise ValueError('NATIVE_OVERLAY_READBACK_NOT_ACCEPTED')
                        result['native_readback']={k:v for k,v in proof.items() if k!='readbacks'}
                        result['provenance_refs']+=refs;converted=overlay['snapshot']
                        result['comparison']=exact_comparison(mapped['mapped_snapshot'],converted)
                        if not result['comparison']['accepted']:raise ValueError('FINAL_SOURCE_DOCUMENT_NOT_EXACTLY_PRESERVED')
                        result['operation']+=';restore_explicit_records_with_complete_native_readback'
                        overlay_verified=True
                    except ValueError as exc:
                        # Retain failed engineering evidence. The alternate
                        # path must independently pass whole-model equivalence
                        # and every explicit value, not waive this failure.
                        result['overlay_readback_failure']=str(exc)
                if not overlay_verified:
                    complex_result=repair_complex_snapshot(original,native,base/'complex_source_proof',
                        source_provenance=task,root=root,stage_runner=_native_stage,
                        timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
                    proof=complex_result['proof'];converted=complex_result['snapshot']
                    if proof.get('accepted') is not True or converted is None:raise ValueError('COMPLETE_SOURCE_MODEL_PROOF_NOT_ACCEPTED')
                    result['complex_model_proof']={k:v for k,v in proof.items() if k not in {'simulation_proofs','provenance_refs'}}
                    result['provenance_refs']+=complex_result['provenance_refs']
                    from .complex_snapshot import apply_native_representation_changes
                    expected_document=apply_native_representation_changes(mapped['mapped_snapshot'],
                        proof.get('native_representation_changes',[]),root=root)
                    result['comparison']=exact_comparison(expected_document,converted)
                    result['operation']+=';complete_registered_source_model_and_explicit_parameter_proof'
        if (result.get('comparison',{}).get('accepted') is not True or
                result['comparison'].get('exact_normalized_document_equality') is not True):
            raise ValueError('FINAL_SOURCE_DOCUMENT_NOT_EXACTLY_PRESERVED')
        if converted.get('Version')!=80:raise ValueError('NATIVE_VERSION_80_REQUIRED')
        for ref in result['implementation']:checked(ref,root)
        save(base/'verified_snapshot.json',converted)
        result.update(status='verified',snapshot=reference(base/'verified_snapshot.json',root))
        task_for(provenance,root)
    except (ValueError,KeyError,TypeError,OSError,InterruptedError) as exc:
        result['reasons']=[type(exc).__name__+':'+str(exc)]
        result['failure_location']=[{'path':str(Path(frame.filename).resolve().relative_to(root)),
                                    'line':frame.lineno,'function':frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)[-8:]
            if Path(frame.filename).resolve().is_relative_to(root)]
    result['completed_at']=datetime.now(timezone.utc).isoformat();save(base/'receipt.json',result)
    return result


def resolve_repaired_snapshot(provenance,*,project_root=ROOT,repair_root=None):
    root=Path(project_root).resolve(strict=True);task=task_for(provenance,root)
    receipt=inside(Path(repair_root or root/DEFAULT_REPAIRS)/task['source_sha256']/'receipt.json',root)
    r=_verify_receipt(receipt,provenance,root)
    return {'path':checked(r['snapshot'],root),'sha256':r['snapshot']['sha256'],
        'original_path':inside(task['source_path'],root),'original_sha256':task['source_sha256'],
        'receipt_path':receipt,'receipt_sha256':sha(receipt),'operation':r['operation'],
        'model_identity_preserved':True,'scientific_equivalence_established':False}


def prepare_repairs(audit_rows,*,project_root=ROOT,output_dir=None,workers=2,timeout_seconds=180,stop_file=None):
    """Account for each unique same-parent JSON source, without refitting."""
    root=Path(project_root).resolve(strict=True)
    if type(workers) is not int or workers not in (1,2,3,4):raise ValueError('ONE_TO_FOUR_REPAIR_WORKERS_REQUIRED')
    if stop_file:stop_file=inside(stop_file,root,exists=False)
    if stop_file and stop_file.exists():return {'version':VERSION,'status':'stopped','all_sources_accounted':False,'results':[]}
    base=inside(output_dir or root/DEFAULT_REPAIRS,root,exists=False);base.mkdir(parents=True,exist_ok=True)
    tasks={};skipped=[]
    from .license_audit import audit_candidate_license
    for row in audit_rows:
        if row.get('identity_match') is not True:continue
        prov=row.get('catalog_provenance',row.get('provenance',{}))
        if Path(prov['source_path']).suffix!='.json':continue
        if prov.get('license_status')!='allowed':
            license_result=audit_candidate_license(row,project_root=root)
            if license_result.get('status')!='license_verified':
                skipped.append({'candidate_id':row['candidate_id'],'reason':'LICENSE_UNRESOLVED'});continue
        task_for(prov,root);tasks.setdefault(prov['source_sha256'],prov)
    summary={'version':VERSION,'status':'running','expected_sources':len(tasks),'results':[],
        'skipped_license_unresolved':skipped,'sealed_observations_accessed':False,'ode_run':False,
        'registered_limits':{'batch_seconds':36000,'batch_output_bytes':32*1024**3,
            'native_phase_seconds':timeout_seconds,'cpu_threads_per_worker':1,'workers':workers,
            'native_simulations_per_partition':'1 when source has >8 compounds; otherwise 4',
            'completed_project_reuse_requires_historical_export_input_sha':True,'source_native_preparation_seconds':10800}}
    started=time.monotonic()
    def run(prov):
        if time.monotonic()-started>36000:raise TimeoutError('Snapshot repair batch 36000 second budget')
        if sum(p.stat().st_size for p in base.rglob('*') if p.is_file())>32*1024**3:
            raise OSError('Snapshot repair batch 32 GiB output budget')
        return repair_snapshot(prov,base/prov['source_sha256'],project_root=root,
            timeout_seconds=timeout_seconds,cpu_threads=1,stop_file=stop_file)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run,prov):digest for digest,prov in sorted(tasks.items())}
        for future in as_completed(futures):
            if future.cancelled():continue
            digest=futures[future]
            try:r=future.result()
            except InterruptedError:
                summary['status']='stopped'
                for pending in futures:pending.cancel()
                continue
            receipt=base/digest/'receipt.json'
            summary['results'].append({'source_sha256':digest,'status':r['status'],'reasons':r['reasons'],
                'receipt':reference(receipt,root)})
            summary['results'].sort(key=lambda x:x['source_sha256'])
            save(base/'repair_summary.json',summary,replace=True)
    summary['status']='stopped' if stop_file and stop_file.exists() else 'complete'
    summary['all_sources_accounted']=summary['status']=='complete' and len(summary['results'])==len(tasks)
    summary['verified_sources']=sum(r['status']=='verified' for r in summary['results'])
    save(base/'repair_summary.json',summary,replace=True);return summary
