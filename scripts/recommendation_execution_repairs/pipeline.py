"""Bounded full-cohort rerun with preserved models and audited snapshot repairs.

Original rankings and all prior attempts stay untouched. Clinical values are
inaccessible to this process: scoring requires a separate complete lock barrier.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_recommendation_execution as original
import rank_recommendation_execution as ranking
from recommendation_execution_repairs.native_pkml import inside, sha, save
from recommendation_execution_repairs.license_audit import audit_candidate_license, DEFAULT_RECEIPT

REVISION = 'engineering_repair_04'
OUT = original.OUT
WORK = OUT/'engineering_repairs/repair_04'
RUN = original.RUN/'repair_04'
SCRIPT = Path(__file__).resolve()
PYTHON = ROOT/'runtime/app-env/bin/python'
STOP = RUN/'stop.requested'
SOURCE_PROGRESS = OUT/'engineering_repairs/snapshot_repairs_11/repair_summary.json'
STATISTICAL_PROTOCOL = OUT/'engineering_amendments/attempt_01/frozen_files'/OUT.relative_to(ROOT)/'protocol.json'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference(path):
    p=inside(path)
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p)}


def check(ref):
    p=inside(ref['path'])
    if sha(p)!=ref['sha256']: raise ValueError('Frozen repair resource changed: '+ref['path'])
    return p


def update(path,value):
    original.save(path,value)


def code_files():
    from recommendation_execution_repairs import complete_candidate, snapshot_candidate
    return sorted(set([ROOT/p for p in original.CODE] + list((ROOT/'scripts/recommendation_execution_repairs').glob('*.py')) +
        [ROOT/p for p in (*complete_candidate.IMPLEMENTATION_FILES,*snapshot_candidate.IMPLEMENTATION_FILES)] +
        list((ROOT/'src/pbpk_kb/model_runtime/R').glob('recommendation_*.R')) +
        list((RUN.parent/'source_repairs_11').glob('run.py'))))


def unique_refs(refs):
    """Merge file identities; verify optional byte counts before normalization.

    SHA checks for ordinary references remain with their existing callers.
    A size annotation is checked against the same SHA-bound project file, so
    removing that redundant annotation cannot hide a changed file or size.
    """
    import re
    found = {}
    for ref in refs:
        if not isinstance(ref, dict) or not {'path', 'sha256'} <= set(ref) or set(ref) - {'path', 'sha256', 'bytes'}:
            raise ValueError('Invalid frozen input reference fields')
        name, digest = ref['path'], ref['sha256']
        if (not isinstance(name, str) or name in {'', '.', '..'} or Path(name).is_absolute()
                or Path(name).as_posix() != name or '..' in Path(name).parts
                or any(part.casefold().startswith('sealed') for part in Path(name).parts)
                or not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('Invalid frozen input identity')
        normalized = {'path': name, 'sha256': digest}
        if name in found and found[name] != normalized:
            raise ValueError('Conflicting frozen input binding: '+name)
        if 'bytes' in ref:
            size = ref['bytes']
            if type(size) is not int or size < 0:
                raise ValueError('Invalid frozen input byte count: '+name)
            path = check(normalized)
            if any(part.casefold().startswith('sealed') for part in path.relative_to(ROOT).parts):
                raise ValueError('Sealed reference cannot be normalized')
            if path.stat().st_size != size:
                raise ValueError('Frozen input byte count differs: '+name)
        found[name] = normalized
    return list(found.values())


def freeze():
    WORK.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    p=WORK/'preparation.json'
    if p.exists():
        record=json.loads(p.read_text())
        for ref in record['inputs']+record['code']:check(ref)
        return record
    inputs=[OUT/'inputs'/name for name in ('cohort.json','scenarios.jsonl','candidate_audit.jsonl','declared_tiers.jsonl','declared_tiers_manifest.json')]
    inputs += [ranking.RANK/'inputs'/name for name in ('queries.jsonl','candidates.jsonl','manifest.json')]
    inputs += [ROOT/DEFAULT_RECEIPT,OUT/'preparation.json',STATISTICAL_PROTOCOL]
    from recommendation_execution_repairs.source_bound_dose import LABEL_PATH
    inputs += [ROOT/LABEL_PATH]
    from recommendation_execution_repairs.historical_source_proofs import MANIFEST,MANIFEST_SHA
    inputs += [check({'path':MANIFEST,'sha256':MANIFEST_SHA})]
    from recommendation_execution_repairs.historical_source_proofs import FOLLOWUP_MANIFEST,FOLLOWUP_SHA
    inputs += [check({'path':FOLLOWUP_MANIFEST,'sha256':FOLLOWUP_SHA})]
    from recommendation_execution_source_audit import known_overlap_freeze_refs
    inputs += [check(ref) for ref in known_overlap_freeze_refs(project_root=ROOT)]
    from recommendation_execution_repairs.engineering_gate import known_gate_evidence_refs
    inputs += [check(ref) for ref in known_gate_evidence_refs(project_root=ROOT)]
    from recommendation_execution_repairs.historical_source_proofs import FAILED_MANIFEST,FAILED_SHA
    failed_path=check({'path':FAILED_MANIFEST,'sha256':FAILED_SHA})
    failed_record=json.loads(failed_path.read_text())
    inputs += [failed_path]
    inputs += [check(row['receipt']) for row in failed_record['entries']]
    inputs += [check(row['preserved']) for key in ('implementation_archive','preparation_archive')
               for row in failed_record[key]]
    from recommendation_execution_repairs.retained_implementations import profile_refs
    inputs += [check(ref) for ref in profile_refs(project_root=ROOT)]
    from recommendation_execution_repairs.retained_implementations import MANIFEST as CONTINUATION,MANIFEST_SHA as CONTINUATION_SHA
    continuation=json.loads(check({'path':CONTINUATION,'sha256':CONTINUATION_SHA}).read_text())
    inputs += [check(ref) for ref in continuation.get('engineering_basis',[])]
    record={'execution_revision':REVISION,'frozen_at':original.now(),'inputs':unique_refs([reference(p) for p in inputs]),
        'code':[reference(p) for p in code_files()], 'sealed_outcomes_accessed':False,'fitting_performed':False,
        'ranking_changes':False,'all_same_parent_pairs_required':True,
        'engineering_followup':'json_native_bridge_followup_20261002','prior_predictions_retained_in_place':True,
        'rules':['Preserve original model/source identity and all drug mechanisms',
                 'Repair schema only with field-complete original-to-native proof',
                 'Keep original source Model and Settings; no snapshot rebuild for project models',
                 'Only registered dose/time/output changes; independent native readback',
                 'Unknown execution is never scientific inapplicability'],
        'runtime':{'workers':4,'complete_project_workers':4,'overlapping_source_workers':0,'maximum_overlapping_native_processes':4,'cpu_threads':1,'source_stage_timeout_seconds':600,'native_timeout_seconds':600,'complete_project_native_timeout_seconds':1800,
                   'worker_timeout_seconds':3600,'total_timeout_seconds':12*3600,'output_limit_bytes':48*1024**3,
                   'minimum_free_bytes':20*1024**3,'log_limit_bytes':32*1024**2}}
    save(p,record);return record


def adapter(candidate):
    path=Path(candidate.get('catalog_provenance',candidate.get('provenance',{}))['source_path'])
    if path.suffix=='.pksim5':
        from recommendation_execution_repairs import complete_candidate
        return complete_candidate
    from recommendation_execution_repairs import json_native_candidate
    return json_native_candidate


def job_key(row):
    import re
    from recommendation_execution_repairs.retained_implementations import MANIFEST, MANIFEST_SHA
    record=json.loads(check({'path':MANIFEST,'sha256':MANIFEST_SHA}).read_text())
    if record.get('schema_version')!='pbpk-retained-execution-registry-v2':
        raise ValueError('Registered multi-generation continuation required')
    mapping=record['job_mapping'];suffix=record.get('continuation_suffix')
    if not isinstance(suffix,str) or not re.fullmatch(r'_r4[a-z][0-9]*',suffix):
        raise ValueError('Registered continuation suffix required')
    pairs={(j['scenario_id'],j['candidate_id']):j for j in mapping}
    if len(mapping)!=177 or len(pairs)!=177 or len({j['job_id'] for j in mapping})!=177:
        raise ValueError('Retained continuation omits or duplicates candidate pairs')
    item=pairs.get((row['scenario_id'],row['candidate_id']))
    base=original.pair_key(row).removesuffix('_r2')
    if (item is None or item.get('old_job_id')!=base+'_r4b' or type(item.get('retained')) is not bool
            or not isinstance(item.get('previous_job_id'),str)
            or not re.fullmatch(re.escape(base)+r'_r4[a-z][0-9]*',item['previous_job_id'])):
        raise ValueError('Candidate pair is not registered for continuation')
    wanted=item['previous_job_id'] if item['retained'] else base+suffix
    if item['job_id']!=wanted:raise ValueError('Unregistered continuation attempt identity')
    return wanted


def complete_project_candidates(audits):
    candidates=[r for r in audits if r.get('identity_match') is True and
                Path(r.get('catalog_provenance',r.get('provenance',{}))['source_path']).suffix=='.pksim5']
    if len(candidates)!=64 or len({(r['scenario_id'],r['candidate_id']) for r in candidates})!=64:
        raise ValueError('Complete project phase requires all 64 original candidate pairs')
    return candidates


def verify_complete_project_plan(record,audits):
    candidates=complete_project_candidates(audits)
    expected={job_key(r):r for r in candidates}
    jobs=record['jobs'];by_id={j['job_id']:j for j in jobs}
    if (record.get('execution_revision')!=REVISION or
            record.get('all_complete_project_candidates_included') is not True or
            record.get('selected_using_rank_or_error') is not False or
            record.get('sealed_outcomes_accessed') is not False or
            len(jobs)!=64 or set(by_id)!=set(expected) or
            check(record['candidate_audit'])!=OUT/'inputs/candidate_audit.jsonl'):
        raise ValueError('Complete project phase registration differs')
    bindings={}
    for name,row in expected.items():
        job=by_id[name];plan=job['plan'];prov=row.get('catalog_provenance',row.get('provenance',{}))
        source={'path':prov['source_path'],'sha256':prov['source_sha256']};bindings[name]=source
        if (any(job.get(k)!=row[k] or plan.get(k)!=row[k] for k in ('scenario_id','candidate_id')) or
                plan.get('adapter_route')!='complete_original_pksim5' or
                plan.get('execution_revision')!=REVISION or
                any(k in plan and plan[k]!=source[v] for k,v in (('source_path','path'),('source_sha256','sha256')))):
            raise ValueError('Complete project job identity or original source differs')
    if record.get('source_bindings')!=bindings:raise ValueError('Complete project source bindings differ')
    return jobs


def verify_complete_project_subset(jobs,early_jobs):
    expected={j['job_id']:j for j in early_jobs}
    actual={j['job_id']:j for j in jobs if j['job_id'] in expected}
    if (len(early_jobs)!=64 or len(expected)!=64 or len(jobs)!=len({j['job_id'] for j in jobs}) or
            len(actual)!=len(expected) or any(actual[name]!=job for name,job in expected.items())):
        raise ValueError('Final candidate plan differs from complete project phase')


def alive():
    p=RUN/'pid.json'
    if not p.exists():return False
    d=json.loads(p.read_text());ident=d.get('identity')
    if not ident or ident != original.identity(ident['pid']):return False
    proc=Path('/proc')/str(ident['pid'])
    return proc.joinpath('cwd').resolve()==ROOT and str(SCRIPT).encode() in proc.joinpath('cmdline').read_bytes().split(b'\0')


def registered_source_status():
    directory=RUN.parent/'source_repairs_11';pidfile=directory/'pid.json'
    result={'state':'not_started','owned_process_alive':False}
    if not pidfile.exists():return result
    record=json.loads(pidfile.read_text());script=directory/'run.py';ident=record.get('identity')
    if record.get('script')!=str(script.relative_to(ROOT)) or not ident:
        raise ValueError('Source repair process registration differs')
    state=directory/'status.json'
    if state.exists():
        d=json.loads(state.read_text());result.update(state=d.get('status','unknown'),workers=d.get('workers'))
    if original.identity(ident['pid'])==ident:
        proc=Path('/proc')/str(ident['pid'])
        if (proc.joinpath('cwd').resolve()!=ROOT or
            str(script).encode() not in proc.joinpath('cmdline').read_bytes().split(b'\0')):
            raise ValueError('Registered source repair process scope differs')
        result['owned_process_alive']=True
    return result


class Controller:
    def __init__(self):
        self.preparation=freeze();self.limits=self.preparation['runtime']
        self.started=time.monotonic();self.disk_checked=0.;self.lock=threading.Lock();self.children={}
    def guard(self):
        if STOP.exists():raise InterruptedError('Project repair stop requested')
        if time.monotonic()-self.started>self.limits['total_timeout_seconds']:raise TimeoutError('Repair total time limit')
        with self.lock:
            if time.monotonic()-self.disk_checked>20:
                if shutil.disk_usage(WORK).free<self.limits['minimum_free_bytes']:raise OSError('Repair disk headroom limit')
                if original.tree_size(OUT)>self.limits['output_limit_bytes']:raise OSError('Project experiment output limit')
                if original.tree_size(RUN)>self.limits['log_limit_bytes']:raise OSError('Repair log limit')
                self.disk_checked=time.monotonic()
    def child(self,job,stage="all"):
        self.guard();name=job['job_id'];directory=OUT/'native'/name
        if (directory/'result.json').exists():
            verify_job(job);return
        if directory.exists():raise RuntimeError('Preserve incomplete job before an explicitly registered restart: '+name)
        log_path=RUN/(name+'.log')
        if log_path.exists() and log_path.stat().st_size>8*1024**2:raise OSError('Worker log limit')
        with log_path.open('a') as log:
            p=subprocess.Popen([str(PYTHON),'-B',str(SCRIPT),'worker','--job',name,'--stage',stage],cwd=ROOT,
                               env=original.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with self.lock:
                self.children[name]={'identity':original.identity(p.pid),'started_at':original.now()}
                update(RUN/'children.json',self.children)
            start=time.monotonic()
            try:
                while p.poll() is None:
                    self.guard()
                    if time.monotonic()-start>self.limits['worker_timeout_seconds']:raise TimeoutError('Native worker limit: '+name)
                    time.sleep(1)
                if p.returncode!=0:raise RuntimeError('Repair worker failed: '+name)
                verify_job(job)
            finally:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        p.wait()
                with self.lock:
                    self.children[name].update(returncode=p.returncode,finished_at=original.now())
                    update(RUN/'children.json',self.children)
    def run_complete_projects(self,audits,scenarios):
        path=WORK/'complete_project_execution_plan.json'
        if not path.exists():
            jobs=[];bindings={}
            for row in complete_project_candidates(audits):
                self.guard();plan=adapter(row).prepare_candidate(scenarios[row['scenario_id']],row,project_root=ROOT)
                name=job_key(row);prov=row.get('catalog_provenance',row.get('provenance',{}))
                jobs.append({'job_id':name,'scenario_id':row['scenario_id'],'candidate_id':row['candidate_id'],'plan':plan})
                bindings[name]={'path':prov['source_path'],'sha256':prov['source_sha256']}
                update(RUN/'status.json',{'state':'registering_complete_project_plans','completed':len(jobs),
                    'total':64,'updated_at':original.now(),'sealed_outcomes_accessed':False})
            record={'execution_revision':REVISION,'jobs':jobs,'source_bindings':bindings,
                'candidate_audit':reference(OUT/'inputs/candidate_audit.jsonl'),
                'all_complete_project_candidates_included':True,'selected_using_rank_or_error':False,
                'sealed_outcomes_accessed':False}
            verify_complete_project_plan(record,audits);freeze();save(path,record)
        record=json.loads(path.read_text());jobs=verify_complete_project_plan(record,audits)
        final_path=WORK/'execution_plan.json'
        if final_path.exists():
            verify_complete_project_subset(json.loads(final_path.read_text())['jobs'],jobs)
            # A registered full plan resumes through the shared pool below;
            # long complete-model retries must not delay pending JSON jobs.
            return jobs
        # No overlapping source worker in this revision: join it before four solvers.
        self.await_existing_source_repairs()
        update(RUN/'status.json',{'state':'running_complete_project_predictions','planned':64,
            'workers':4,'execution_plan':reference(path),'updated_at':original.now(),'clinical_outcomes_read':False})
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(self.child,job,stage='complete_projects') for job in jobs]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                STOP.touch();raise
        self.guard();freeze()
        for job in jobs:verify_job(job)
        return jobs

    def await_existing_source_repairs(self):
        """Join only the explicitly registered project source worker."""
        directory=RUN.parent/'source_repairs_11'
        pidfile=directory/'pid.json'
        if not pidfile.exists():return
        record=json.loads(pidfile.read_text());ident=record.get('identity')
        script=directory/'run.py'
        if record.get('script')!=str(script.relative_to(ROOT)) or not ident:
            raise ValueError('Source repair process registration differs')
        while original.identity(ident['pid'])==ident:
            proc=Path('/proc')/str(ident['pid'])
            if (proc.joinpath('cwd').resolve()!=ROOT or
                str(script).encode() not in proc.joinpath('cmdline').read_bytes().split(b'\0')):
                raise ValueError('Registered source repair process scope differs')
            try:self.guard()
            except BaseException:
                (directory/'stop.requested').touch();raise
            time.sleep(1)
        state=json.loads((directory/'status.json').read_text())
        if state.get('status')!='complete':
            raise ValueError('Source preparation stopped or failed; preserve attempts before restart')

    def run(self):
        audits=rows(OUT/'inputs/candidate_audit.jsonl');scenarios={r['scenario_id']:r for r in rows(OUT/'inputs/scenarios.jsonl')}
        same=[r for r in audits if r.get('identity_match') is True]
        from recommendation_execution_repairs.snapshot_repair import prepare_repairs
        early_jobs=self.run_complete_projects(audits,scenarios)
        update(RUN/'status.json',{'state':'repairing_snapshot_schemas','started_at':original.now(),'sealed_outcomes_accessed':False})
        self.await_existing_source_repairs()
        # Supervise the entire preparation stage as well as each bounded R
        # child, so disk/time/stop checks continue between source conversions.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(prepare_repairs,same,workers=self.limits['workers'],
                               timeout_seconds=self.limits['source_stage_timeout_seconds'],stop_file=STOP)
            try:
                while not future.done():
                    self.guard();time.sleep(1)
                source_summary=future.result()
            except BaseException:
                STOP.touch();raise
        self.guard();freeze()
        update(WORK/'source_repair_summary.json',source_summary)
        plan_path=WORK/'execution_plan.json'
        if not plan_path.exists():
            jobs=[]
            for row in same:
                self.guard()
                plan=adapter(row).prepare_candidate(scenarios[row['scenario_id']],row,project_root=ROOT)
                jobs.append({'job_id':job_key(row),'scenario_id':row['scenario_id'],'candidate_id':row['candidate_id'],'plan':plan})
                update(RUN/'status.json',{'state':'registering_all_candidate_plans','completed':len(jobs),'total':len(same),'updated_at':original.now()})
            verify_complete_project_subset(jobs,early_jobs)
            freeze()
            save(plan_path,{'execution_revision':REVISION,'jobs':jobs,'all_same_parent_candidates_included':True,
                            'selected_using_rank_or_error':False,'sealed_outcomes_accessed':False})
        plan=json.loads(plan_path.read_text());jobs=plan['jobs']
        verify_complete_project_subset(jobs,early_jobs)
        if len(jobs)!=len(same) or {(j['scenario_id'],j['candidate_id']) for j in jobs}!={(r['scenario_id'],r['candidate_id']) for r in same}:
            raise ValueError('Repair plans omit original same-parent candidate pairs')
        update(RUN/'status.json',{'state':'running_preserved_candidate_predictions','planned':len(jobs),'updated_at':original.now()})
        with ThreadPoolExecutor(max_workers=self.limits['workers']) as pool:
            futures=[pool.submit(self.child,job) for job in jobs]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                STOP.touch();raise
        self.guard();freeze()
        results=[verify_job(job) for job in jobs]
        records=[{'job_id':job['job_id'],'scenario_id':job['scenario_id'],'candidate_id':job['candidate_id'],
                  'artifacts':[reference(p) for p in sorted((OUT/'native'/job['job_id']).rglob('*')) if p.is_file()]} for job in jobs]
        lock=WORK/'predictions_lock.json'
        lock_value={'execution_revision':REVISION,'preparation':reference(WORK/'preparation.json'),
                    'execution_plan':reference(plan_path),'planned_jobs':len(jobs),'jobs':records,'target_observations_read':False}
        if not lock.exists():save(lock,lock_value)
        elif json.loads(lock.read_text())!=lock_value:raise ValueError('Existing all-predictions lock no longer binds current artifacts')
        update(RUN/'status.json',{'state':'all_candidate_predictions_locked','counts':dict(Counter(r['status'] for r in results)),
                                 'updated_at':original.now(),'predictions_lock':reference(lock),'clinical_outcomes_read':False})
        gate=engineering_completeness(source_summary,jobs,results)
        update(WORK/'engineering_completeness.json',gate)
        if not gate['accepted']:
            update(RUN/'status.json',{'state':'engineering_repairs_required','updated_at':original.now(),
                'unresolved_engineering_issues':gate['issue_count'],'details':reference(WORK/'engineering_completeness.json'),
                'clinical_outcomes_read':False})
            return
        self.score_locked(results)

    def score_locked(self, results):
        jobs=json.loads((WORK/'execution_plan.json').read_text())['jobs']
        gate=engineering_completeness(json.loads((WORK/'source_repair_summary.json').read_text()),jobs,results)
        if not gate['accepted']:raise ValueError('Engineering repair completeness gate rejected clinical scoring')
        while not (ranking.RANK/'ranking_lock.json').exists():
            self.guard()
            progress=original.RUN/'ranking_continuation/status.json'
            if progress.exists() and json.loads(progress.read_text()).get('state')=='failed':
                raise RuntimeError('Frozen ranking continuation failed; predictions remain locked, clinical values unopened')
            update(RUN/'status.json',{'state':'waiting_for_all_frozen_rankings','native_counts':dict(Counter(r['status'] for r in results)),
                                    'updated_at':original.now(),'clinical_outcomes_read':False})
            time.sleep(5)
        self.guard();freeze()
        with (original.RUN/'pipeline.lock').open('a') as main_lock:
            while True:
                self.guard()
                try:fcntl.flock(main_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(1)
            promote_protocol()
            update(RUN/'status.json',{'state':'scoring_all_locked_predictions','updated_at':original.now()})
            from recommendation_execution_repairs.scoring import score_run
            report=score_run(OUT)
            update(RUN/'status.json',{'state':'complete_with_reported_coverage','updated_at':original.now(),
                                     'summary':report['summary'],'results':reference(OUT/'results.json')})


def verify_job(job):
    directory=OUT/'native'/job['job_id'];p=directory/'execution_plan.json';r=json.loads((directory/'result.json').read_text())
    if json.loads(p.read_text())!=job['plan'] or r.get('execution_plan_sha256')!=sha(p):raise ValueError('Frozen native job differs')
    if r.get('execution_revision')!=REVISION or r.get('implementation_sha256')!=job['plan']['implementation_sha256']:raise ValueError('Repair implementation differs')
    if r.get('status') not in {'blocked','prediction_locked','native_failed','readback_failed','failed'}:raise ValueError('Native job lacks terminal state')
    if (r['status']=='prediction_locked') != (r.get('prediction_locked') is True):raise ValueError('Prediction flag/status mismatch')
    if any(r.get(k)!=job[k] for k in ('scenario_id','candidate_id')):raise ValueError('Native identity differs')
    if r.get('sealed_observations_accessed') is not False or r.get('fitting_performed') is not False:raise ValueError('Native outcome/fitting constraint missing')
    if r.get('prediction_locked'):
        prediction=check({'path':r['prediction_file'],'sha256':r['prediction_sha256']})
        lock_path=check({'path':r['prediction_lock'],'sha256':r['prediction_lock_sha256']})
        if not prediction.is_relative_to(directory) or not lock_path.is_relative_to(directory):raise ValueError('Prediction belongs to another attempt')
        lock=json.loads(lock_path.read_text())
        for key in ('execution_revision','implementation_sha256','scenario_sha256','candidate_sha256','source_sha256'):
            if lock.get(key)!=job['plan'].get(key):raise ValueError('Prediction lock provenance differs: '+key)
        if lock.get('execution_verified') is not True or lock.get('sealed_observations_accessed') is not False:raise ValueError('Prediction verification barrier missing')
        files={check({'path':name,'sha256':digest}) for name,digest in lock['files'].items()}
        if not {prediction,p}.issubset(files):raise ValueError('Prediction lock omits plan or prediction')
    return r


def engineering_completeness(source_summary,jobs,results):
    """A recorded engineering failure is not a completed scientific experiment."""
    from recommendation_execution_repairs.engineering_gate import assess_blocked_plan
    from recommendation_execution_repairs.source_unavailability import assess_source_unavailable
    issues=[];unavailable=[];source_unavailable=[]
    records=(source_summary or {}).get('results',[]) if isinstance(source_summary,dict) else []
    valid_records=isinstance(records,list) and all(isinstance(r,dict) for r in records)
    if not valid_records:records=[]
    actual_verified=sum(r.get('status')=='verified' for r in records)
    if (not isinstance(source_summary,dict) or source_summary.get('status')!='complete' or
            source_summary.get('all_sources_accounted') is not True or not valid_records or
            type(source_summary.get('expected_sources')) is not int or source_summary['expected_sources']<0 or
            type(source_summary.get('verified_sources')) is not int or
            source_summary.get('expected_sources')!=len(records) or
            len({r.get('source_sha256') for r in records})!=len(records) or
            source_summary.get('verified_sources')!=actual_verified):
        issues.append({'stage':'source_repair','reason':'Complete unique source inventory and truthful verification counts required'})
    for row in records:
        if row.get('status')=='verified':continue
        review=assess_source_unavailable(row,project_root=ROOT)
        if review['accepted']:
            source_unavailable.append({'stage':'source_review','source_sha256':row['source_sha256'],'assessment':review})
        else:issues.append({'stage':'source_repair',**row,'assessment':review})
    by_pair={(r['scenario_id'],r['candidate_id']):r for r in results}
    expected={(j['scenario_id'],j['candidate_id']) for j in jobs}
    if set(by_pair)!=expected or len(results)!=len(by_pair) or len(expected)!=len(jobs):
        issues.append({'stage':'native','reason':'Complete unique result matrix required'})
    for job in jobs:
        r=by_pair.get((job['scenario_id'],job['candidate_id']))
        if r is None:continue
        plan=job['plan']
        if r.get('status')=='prediction_locked':continue
        if plan.get('status')=='ready' or r.get('status')!='blocked':
            issues.append({'stage':'native','job_id':job['job_id'],'status':r.get('status'),'reasons':r.get('reasons',[])})
        elif r.get('reasons')!=plan.get('reasons'):
            issues.append({'stage':'native','job_id':job['job_id'],'reason':'Terminal blocker differs from registered plan'})
        else:
            review=assess_blocked_plan(plan)
            entry={'stage':'plan_review','job_id':job['job_id'],'assessment':review}
            (unavailable if review['accepted'] else issues).append(entry)
    return {'accepted':not issues,'issue_count':len(issues),'issues':issues,
            'verified_unavailable_cases':unavailable,'verified_unavailable_sources':source_unavailable,
            'source_accounting':{'verified_sources':actual_verified,'verified_unavailable_sources':len(source_unavailable),
                                 'accounted_sources':len(records)},
            'clinical_outcomes_read':False,'engineering_failure_is_not_scientific_inapplicability':True}


def verify_source_repairs(summary,audits,*,root=ROOT):
    """Check the full catalog source set and each proof, not summary counters."""
    from recommendation_execution_repairs.snapshot_repair import _verify_receipt
    expected={}
    for row in audits:
        prov=row.get('catalog_provenance',row.get('provenance',{}))
        if row.get('identity_match') is True and Path(prov['source_path']).suffix=='.json':
            expected[prov['source_sha256']]=prov
    records=summary.get('results',[])
    if (summary.get('status')!='complete' or summary.get('all_sources_accounted') is not True or
            summary.get('skipped_license_unresolved') or len(records)!=len(expected) or
            summary.get('expected_sources')!=len(expected) or
            summary.get('verified_sources')!=sum(r.get('status')=='verified' for r in records) or
            {r['source_sha256'] for r in records}!=set(expected)):
        raise ValueError('Complete original JSON source set has not passed repair')
    refs=[]
    for row in records:
        if row.get('status')!='verified':
            from recommendation_execution_repairs.source_unavailability import assess_source_unavailable
            review=assess_source_unavailable(row,provenance=expected[row['source_sha256']],project_root=root)
            if not review['accepted']:raise ValueError('Unresolved engineering source repair')
            ref=row['receipt'];path=inside(ref['path'],root)
            if sha(path)!=ref['sha256']:raise ValueError('Repair source receipt checksum differs')
            _verify_receipt(path,expected[row['source_sha256']],root,require_accepted=False)
            refs.extend(review['verified_references'])
            continue
        ref=row['receipt'];path=inside(ref['path'],root)
        if sha(path)!=ref['sha256']:raise ValueError('Repair source receipt checksum differs')
        _verify_receipt(path,expected[row['source_sha256']],root,require_accepted=True)
        refs.append(ref)
    return refs


def promote_protocol():
    """Publish one immutable revision after all outcomes-free predictions lock.

    Prior ranking preparation remains intact. The statistical estimators are
    inherited from the protocol bound before this repair began, not chosen after
    seeing native predictions or clinical values.
    """
    frozen=freeze();record=json.loads((WORK/'predictions_lock.json').read_text())
    if record.get('target_observations_read') is not False:raise ValueError('Prediction outcome barrier missing')
    check(record['preparation']);check(record['execution_plan'])
    plan=json.loads((WORK/'execution_plan.json').read_text())
    jobs={j['job_id']:j for j in plan['jobs']}
    if record.get('planned_jobs')!=len(jobs) or {j['job_id'] for j in record['jobs']}!=set(jobs):
        raise ValueError('Global repair lock omits candidate jobs')
    for item in record['jobs']:
        verify_job(jobs[item['job_id']])
        for artifact in item['artifacts']:check(artifact)
    gate=engineering_completeness(json.loads((WORK/'source_repair_summary.json').read_text()),
        list(jobs.values()),[verify_job(job) for job in jobs.values()])
    if not gate['accepted']:raise ValueError('Engineering repair completeness gate rejected publication')
    source_refs=verify_source_repairs(json.loads((WORK/'source_repair_summary.json').read_text()),
                                     rows(OUT/'inputs/candidate_audit.jsonl'))
    from recommendation_execution_repairs.engineering_gate import assessed_gate_evidence_refs
    gate_refs=assessed_gate_evidence_refs(gate)
    for ref in gate_refs:check(ref)
    for method in ranking.METHODS:original.verify_retained_ranking(method)
    ranking_lock=json.loads((ranking.RANK/'ranking_lock.json').read_text())
    if ranking_lock.get('state')!='complete' or ranking_lock.get('scientific_outcomes_read') is not False:
        raise ValueError('All original rankings must be completely locked')
    if (OUT/'results.json').exists() or (OUT/'results.csv').exists():raise ValueError('Existing clinical results must not be overwritten')
    published_plan=OUT/'inputs/execution_plan.json'
    ranking.frozen_write(published_plan,plan)
    prototype=json.loads(STATISTICAL_PROTOCOL.read_text())
    scenarios=rows(OUT/'inputs/scenarios.jsonl');candidates=rows(ranking.RANK/'inputs/candidates.jsonl')
    if prototype['cases']!=len(scenarios) or prototype['candidates']!=len(candidates) or prototype['methods']!=list(ranking.METHODS):
        raise ValueError('Repair cannot change original experiment scope or methods')
    prototype.update(execution_revision=REVISION,frozen_at=frozen['frozen_at'],
        planned_candidate_executions=len(jobs),
        preparation_revision=reference(WORK/'preparation.json'),
        prior_statistical_protocol=reference(STATISTICAL_PROTOCOL),
        engineering_amendment='Complete saved Model/Settings execution, exact snapshot schema restoration and same-commit license evidence; independently verified original input absence remains unknown; runtime failures still block scoring; source identity and ranking unchanged.',
        inputs=unique_refs([*frozen['inputs'],reference(WORK/'preparation.json'),reference(WORK/'execution_plan.json'),
                reference(published_plan),reference(WORK/'predictions_lock.json'),
                reference(WORK/'source_repair_summary.json'),reference(WORK/'engineering_completeness.json'),*source_refs,*gate_refs]),
        code=frozen['code'],runtime=frozen['runtime'],
        outcome_access={'sealed_values_read_by_preparation':False,'all_predictions_and_rankings_locked_before_scoring':True,
                        'previous_source_curation_not_claimed_new_blind_data':True})
    ranking.frozen_write(OUT/'protocol.json',prototype)
    ranking.frozen_write(OUT/'native/predictions_lock.json',{
        'schema_version':'pbpk-all-candidate-predictions-lock-v1','protocol':reference(OUT/'protocol.json'),
        'planned_jobs':len(jobs),'jobs':record['jobs'],'target_observations_read':False})


def worker(job_id,stage='all'):
    def stop(*_):raise KeyboardInterrupt('Owned native repair stopped')
    signal.signal(signal.SIGTERM,stop);preparation=freeze()
    if stage not in ('all','complete_projects'):raise ValueError('Unknown registered worker stage')
    path=WORK/('complete_project_execution_plan.json' if stage=='complete_projects' else 'execution_plan.json')
    record=json.loads(path.read_text());jobs=record['jobs']
    if stage=='complete_projects':jobs=verify_complete_project_plan(record,rows(OUT/'inputs/candidate_audit.jsonl'))
    job=next(j for j in jobs if j['job_id']==job_id)
    scenario=next(r for r in rows(OUT/'inputs/scenarios.jsonl') if r['scenario_id']==job['scenario_id'])
    candidate=next(r for r in rows(OUT/'inputs/candidate_audit.jsonl') if (r['scenario_id'],r['candidate_id'])==(job['scenario_id'],job['candidate_id']))
    executor=adapter(candidate)
    if executor.prepare_candidate(scenario,candidate,project_root=ROOT)!=job['plan']:raise ValueError('Registered plan changed before native execution')
    if job['plan'].get('adapter_route') in {'complete_original_pksim5','verified_json_native_pksim5'}:
        executor.execute_candidate(scenario,candidate,OUT/'native'/job_id,project_root=ROOT,timeout_seconds=preparation['runtime']['complete_project_native_timeout_seconds'],
                                   cpu_threads=1,stop_file=STOP,frozen_plan=job['plan'])
    else:
        executor.execute_candidate(scenario,candidate,OUT/'native'/job_id,project_root=ROOT,timeout_seconds=preparation['runtime']['native_timeout_seconds'])
    freeze();verify_job(job)


def status():
    p=RUN/'status.json';d=json.loads(p.read_text()) if p.exists() else {'state':'not_started'};d['owned_process_alive']=alive()
    d['source_repair_worker']=registered_source_status()
    if SOURCE_PROGRESS.exists():
        source=json.loads(SOURCE_PROGRESS.read_text())
        counts=Counter(r['status'] for r in source.get('results',[]))
        d['source_repairs']={'state':source.get('status'),'planned':source.get('expected_sources',0),
            'completed':sum(counts.values()),'counts':dict(counts)}
    plan=WORK/'execution_plan.json'
    if not plan.exists():plan=WORK/'complete_project_execution_plan.json'
    if plan.exists():
        jobs=json.loads(plan.read_text())['jobs'];counts=Counter()
        for job in jobs:
            p=OUT/'native'/job['job_id']/'result.json'
            if p.exists():counts[json.loads(p.read_text())['status']]+=1
        d['native']={'planned':len(jobs),'completed':sum(counts.values()),'counts':dict(counts)}
    rank_states=[]
    for name in ranking.METHODS:
        p=ranking.RANK/'results'/name/'status.json'
        r=json.loads(p.read_text()) if p.exists() else {}
        rank_states.append({'method':name,'state':r.get('state','pending'),
                            'queries':41 if r.get('state')=='complete' else r.get('completed_query_variants',0)})
    d['ranking']={'complete':sum(r['state']=='complete' for r in rank_states),'planned':len(rank_states),'methods':rank_states}
    return d


def watch():
    def bar(done,total):
        fraction=done/total if total else 0
        return '█'*int(fraction*32)+'░'*(32-int(fraction*32))+f' {done:>3}/{total:<3} {fraction:6.1%}'
    try:
        while True:
            s=status();n=s.get('native',{});r=s['ranking'];p=s.get('source_repairs',{})
            print('\033[2J\033[H'+'PBPK original-model execution validation | repair batch 04\n',end='')
            source_running=s.get('source_repair_worker',{}).get('owned_process_alive')
            label='Running' if s['owned_process_alive'] else ('Source-model repair running' if source_running else 'Stopped')
            print('Status '+s['state']+' | Project process '+label)
            print('Ranking '+bar(r['complete'],r['planned']))
            print('Source model '+bar(p.get('completed',0),p.get('planned',81))+'  '+json.dumps(p.get('counts',{}),ensure_ascii=False))
            print('Simulation '+bar(n.get('completed',0),n.get('planned',177)))
            print('Results '+json.dumps(n.get('counts',{}),ensure_ascii=False))
            print('Ctrl+C exits the display without stopping experiments.',flush=True)
            time.sleep(5)
    except KeyboardInterrupt:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['start','run','worker','status','watch','stop']);parser.add_argument('--job');parser.add_argument('--stage',choices=['all','complete_projects'],default='all');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.action=='worker':worker(args.job,args.stage)
    elif args.action=='watch':watch()
    elif args.action=='status':print(json.dumps(status(),ensure_ascii=False,indent=2))
    elif args.action=='stop':
        STOP.touch()
        if registered_source_status()['owned_process_alive']:
            (RUN.parent/'source_repairs_11/stop.requested').touch()
        if alive():os.kill(json.loads((RUN/'pid.json').read_text())['identity']['pid'],signal.SIGTERM)
        print('Project repair stop requested')
    elif args.action=='start':
        with (RUN/'start.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not alive():
                freeze();STOP.unlink(missing_ok=True)
                with (RUN/'pipeline.log').open('a') as log:
                    p=subprocess.Popen([str(PYTHON),'-B',str(SCRIPT),'run'],cwd=ROOT,env=original.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(RUN/'pid.json',{'identity':original.identity(p.pid),'script':str(SCRIPT.relative_to(ROOT)),'created_at':original.now()})
            print(json.dumps(status(),ensure_ascii=False))
    else:
        with (RUN/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def stop(*_):STOP.touch()
            signal.signal(signal.SIGTERM,stop)
            try:Controller().run()
            except BaseException as exc:
                update(RUN/'status.json',{'state':'stopped' if isinstance(exc,InterruptedError) else 'failed','error':type(exc).__name__+':'+str(exc)[:500],'updated_at':original.now()})
                raise


if __name__=='__main__':main()
