"""Prediction locks fail closed before any clinical observations may be read.

Only synthetic artifacts beneath this project's runtime/tmp are created. Native
execution and snapshot preparation are replaced by stubs in controller tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from recommendation_execution_repairs import pipeline


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


@pytest.fixture
def locked_job(monkeypatch):
    scratch = ROOT / 'runtime/tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preserved-pipeline-', dir=scratch) as directory:
        base = Path(directory)
        out, work, run = base/'out', base/'work', base/'run'
        for p in (out, work, run): p.mkdir()
        for name, value in [('OUT', out), ('WORK', work), ('RUN', run), ('STOP', run/'stop.requested')]:
            monkeypatch.setattr(pipeline, name, value)
        source = base/'source.pksim5'; source.write_bytes(b'synthetic source, no observations')
        code = base/'executor.py'; code.write_text('# Synthetic implementation binding\n')
        implementation = {str(code.relative_to(ROOT)): pipeline.sha(code)}
        plan = {'status': 'ready', 'execution_revision': pipeline.REVISION,
                'scenario_id': 'scenario-1', 'candidate_id': 'candidate-1',
                'scenario_sha256': 'a'*64, 'candidate_sha256': 'b'*64,
                'source_path': str(source.relative_to(ROOT)), 'source_sha256': pipeline.sha(source),
                'implementation_sha256': implementation}
        job = {'job_id': 'job-1_r4', 'scenario_id': 'scenario-1', 'candidate_id': 'candidate-1', 'plan': plan}
        destination = out/'native'/job['job_id']; destination.mkdir(parents=True)
        plan_path = destination/'execution_plan.json'; write(plan_path, plan)
        prediction_path = destination/'prediction.json'
        write(prediction_path, {'scenario_id': job['scenario_id'], 'candidate_id': job['candidate_id'],
                                'execution_revision': pipeline.REVISION, 'test_fixture': True})
        lock = {key: deepcopy(plan[key]) for key in ('execution_revision', 'implementation_sha256', 'scenario_sha256',
                                                    'candidate_sha256', 'source_sha256')}
        lock.update(execution_verified=True, sealed_observations_accessed=False,
                    files={str(p.relative_to(ROOT)): pipeline.sha(p) for p in (plan_path, prediction_path)})
        lock_path = destination/'prediction_lock.json'; write(lock_path, lock)
        result = {'status': 'prediction_locked', 'prediction_locked': True,
                  'execution_revision': pipeline.REVISION, 'implementation_sha256': implementation,
                  'scenario_id': job['scenario_id'], 'candidate_id': job['candidate_id'],
                  'execution_plan_sha256': pipeline.sha(plan_path), 'sealed_observations_accessed': False,
                  'fitting_performed': False, 'prediction_file': str(prediction_path.relative_to(ROOT)),
                  'prediction_sha256': pipeline.sha(prediction_path), 'prediction_lock': str(lock_path.relative_to(ROOT)),
                  'prediction_lock_sha256': pipeline.sha(lock_path)}
        result_path = destination/'result.json'; write(result_path, result)
        yield dict(base=base, out=out, work=work, run=run, directory=destination, job=job, plan=plan,
                   plan_path=plan_path, prediction_path=prediction_path, lock=lock, lock_path=lock_path,
                   result=result, result_path=result_path)


def rewrite_lock(fixture):
    """Rebind only the outer file SHA so tampering reaches the inner lock gate."""
    write(fixture['lock_path'], fixture['lock'])
    fixture['result']['prediction_lock_sha256'] = pipeline.sha(fixture['lock_path'])
    write(fixture['result_path'], fixture['result'])


def test_complete_owned_prediction_and_blocked_terminal_result_are_accepted(locked_job):
    f = locked_job
    assert pipeline.verify_job(f['job'])['status'] == 'prediction_locked'
    f['result'].update(status='blocked', prediction_locked=False)
    for key in ('prediction_file', 'prediction_sha256', 'prediction_lock', 'prediction_lock_sha256'):
        f['result'].pop(key)
    write(f['result_path'], f['result'])
    assert pipeline.verify_job(f['job'])['status'] == 'blocked'


@pytest.mark.parametrize('status,flag', [('prediction_locked', False), ('prediction_locked', None),
                                        ('prediction_locked', 1), ('blocked', True), ('failed', True)])
def test_prediction_status_and_boolean_flag_must_agree(locked_job, status, flag):
    f = locked_job; f['result'].update(status=status, prediction_locked=flag); write(f['result_path'], f['result'])
    with pytest.raises(ValueError, match='flag/status'): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('tamper', ['plan_content', 'result_plan_sha', 'expected_job_plan'])
def test_plan_bytes_content_and_registered_job_are_bound(locked_job, tamper):
    f = locked_job
    if tamper == 'plan_content':
        changed = deepcopy(f['plan']); changed['source_sha256'] = '0'*64; write(f['plan_path'], changed)
    elif tamper == 'result_plan_sha':
        f['result']['execution_plan_sha256'] = '0'*64; write(f['result_path'], f['result'])
    else:
        f['job']['plan'] = deepcopy(f['plan']); f['job']['plan']['source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Frozen native job'): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('field,value', [('execution_revision', 'engineering_repair_old'),
                                        ('implementation_sha256', {'wrong.py': '0'*64})])
def test_result_code_and_revision_must_equal_registered_plan(locked_job, field, value):
    f = locked_job; f['result'][field] = value; write(f['result_path'], f['result'])
    with pytest.raises(ValueError, match='implementation differs'): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('field', ['execution_revision', 'implementation_sha256', 'scenario_sha256',
                                  'candidate_sha256', 'source_sha256'])
def test_prediction_lock_binds_each_original_provenance_field(locked_job, field):
    f = locked_job; f['lock'][field] = {'wrong.py': '0'*64} if field == 'implementation_sha256' else 'tampered'
    rewrite_lock(f)
    with pytest.raises(ValueError, match='lock provenance differs: '+field): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('field,value', [('execution_verified', False), ('sealed_observations_accessed', True)])
def test_lock_cannot_claim_unverified_or_unblinded_execution(locked_job, field, value):
    f = locked_job; f['lock'][field] = value; rewrite_lock(f)
    with pytest.raises(ValueError, match='verification barrier'): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('path_key', ['prediction_path', 'lock_path'])
def test_missing_prediction_or_lock_file_is_rejected(locked_job, path_key):
    f = locked_job; f[path_key].unlink()
    with pytest.raises(FileNotFoundError): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('target', ['prediction', 'plan'])
def test_lock_cannot_omit_prediction_or_plan_even_with_a_valid_outer_sha(locked_job, target):
    f = locked_job; f['lock']['files'].pop(str(f[target+'_path'].relative_to(ROOT))); rewrite_lock(f)
    with pytest.raises(ValueError, match='omits plan or prediction'): pipeline.verify_job(f['job'])


def test_prediction_byte_tampering_is_rejected(locked_job):
    f = locked_job; f['prediction_path'].write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='Frozen repair resource changed'): pipeline.verify_job(f['job'])


def test_another_attempts_prediction_cannot_be_borrowed(locked_job):
    f = locked_job; other = f['out']/'native'/'other_r4'/'prediction.json'
    write(other, {'test_fixture': True})
    f['result']['prediction_file'] = str(other.relative_to(ROOT)); f['result']['prediction_sha256'] = pipeline.sha(other)
    write(f['result_path'], f['result'])
    with pytest.raises(ValueError, match='another attempt'): pipeline.verify_job(f['job'])


@pytest.mark.parametrize('kind', ['source', 'code'])
def test_frozen_preparation_rejects_changed_source_or_implementation_bytes(locked_job, kind):
    f = locked_job
    source = ROOT/f['plan']['source_path']
    code = ROOT/next(iter(f['plan']['implementation_sha256']))
    preparation = {'inputs': [pipeline.reference(source)], 'code': [pipeline.reference(code)]}
    write(f['work']/'preparation.json', preparation)
    assert pipeline.freeze() == preparation
    (source if kind == 'source' else code).write_bytes(b'changed after preparation was frozen')
    with pytest.raises(ValueError, match='Frozen repair resource changed'): pipeline.freeze()


def controller_for_fixture(f, monkeypatch):
    inputs = f['out']/'inputs'; inputs.mkdir()
    (inputs/'candidate_audit.jsonl').write_text(json.dumps({'scenario_id': 'scenario-1', 'candidate_id': 'candidate-1', 'identity_match': True})+'\n')
    (inputs/'scenarios.jsonl').write_text(json.dumps({'scenario_id': 'scenario-1'})+'\n')
    write(f['work']/'execution_plan.json', {'jobs': [f['job']]})
    preparation = {'runtime': {'source_stage_timeout_seconds':600, 'workers':4}, 'test_fixture': True}
    write(f['work']/'preparation.json', preparation)
    monkeypatch.setattr(pipeline, 'freeze', lambda: preparation)
    monkeypatch.setattr(pipeline.Controller, 'guard', lambda self: None)
    # These one-job fixtures exercise final locks and scientific failure gates.
    # Full 64-job dispatch/coverage is tested independently by the staged suite.
    monkeypatch.setattr(pipeline.Controller, 'run_complete_projects', lambda self, audits, scenarios: [f['job']])
    def check_fixture_subset(jobs, early_jobs):
        assert early_jobs == [f['job']]
        assert jobs == early_jobs
    monkeypatch.setattr(pipeline, 'verify_complete_project_subset', check_fixture_subset)
    monkeypatch.setattr(pipeline.Controller, 'child', lambda self, job: pipeline.verify_job(job))
    monkeypatch.setattr(pipeline.Controller, 'score_locked', lambda self, results: None)
    stub = types.ModuleType('recommendation_execution_repairs.snapshot_repair')
    stub.prepare_repairs = lambda *args, **kwargs: {'status':'complete','all_sources_accounted':True,
        'verified_sources':0,'expected_sources':0,'results':[]}
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return pipeline.Controller()


def test_existing_identical_global_lock_is_reusable_without_native_execution(locked_job, monkeypatch):
    f = locked_job; controller = controller_for_fixture(f, monkeypatch)
    controller.run(); path = f['work']/'predictions_lock.json'; original = path.read_bytes()
    controller.run()
    assert path.read_bytes() == original
    assert json.loads((f['run']/'status.json').read_text())['state'] == 'all_candidate_predictions_locked'


@pytest.mark.parametrize('tamper', ['job_set', 'artifact_sha', 'preparation_sha', 'unlocked_extra_artifact'])
def test_existing_global_lock_must_bind_exactly_the_current_inventory(locked_job, monkeypatch, tamper):
    f = locked_job; controller = controller_for_fixture(f, monkeypatch); controller.run()
    path = f['work']/'predictions_lock.json'; lock = json.loads(path.read_text())
    if tamper == 'job_set': lock['jobs'] = []
    elif tamper == 'artifact_sha': lock['jobs'][0]['artifacts'][0]['sha256'] = '0'*64
    elif tamper == 'preparation_sha': lock['preparation']['sha256'] = '0'*64
    else: write(f['directory']/'unexpected_later_artifact.json', {'added_after_lock': True})
    if tamper != 'unlocked_extra_artifact': write(path, lock)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Existing all-predictions lock'): controller.run()
    assert path.read_bytes() == before  # A stale lock must not be silently regenerated.


@pytest.mark.parametrize('failure', ['source_failure', 'native_failure', 'execution_blocked'])
def test_repair_failure_preserves_predictions_but_never_starts_clinical_scoring(locked_job, monkeypatch, failure):
    f=locked_job;controller=controller_for_fixture(f,monkeypatch)
    if failure=='source_failure':
        sys.modules['recommendation_execution_repairs.snapshot_repair'].prepare_repairs=lambda *a,**k: {
            'status':'complete','all_sources_accounted':True,'verified_sources':0,'expected_sources':1,
            'results':[{'source_sha256':'a'*64,'status':'failed','reasons':['source parameter lost']}]}
    else:
        f['result'].update(status='native_failed' if failure=='native_failure' else 'blocked',prediction_locked=False,
                           reasons=['native execution engineering defect'])
        write(f['result_path'],f['result'])
    called=[]
    monkeypatch.setattr(pipeline.Controller,'score_locked',lambda self,results:called.append(results))
    controller.run()
    assert not called
    assert (f['work']/'predictions_lock.json').exists()
    state=json.loads((f['run']/'status.json').read_text())
    assert state['state']=='engineering_repairs_required'
    assert state['clinical_outcomes_read'] is False
    assert json.loads((f['work']/'engineering_completeness.json').read_text())['accepted'] is False


@pytest.mark.parametrize('tamper',['replacement_pair','duplicate_pair','missing_pair'])
def test_completeness_requires_exact_native_pair_set(locked_job,tamper):
    f=locked_job;result=deepcopy(f['result']);results=[result]
    if tamper=='replacement_pair':result['candidate_id']='not-the-registered-candidate'
    elif tamper=='duplicate_pair':results.append(deepcopy(result))
    else:results=[]
    sources={'status':'complete','all_sources_accounted':True,'verified_sources':0,'expected_sources':0,'results':[]}
    assert pipeline.engineering_completeness(sources,[f['job']],results)['accepted'] is False


def test_source_completeness_does_not_trust_success_counters(locked_job,monkeypatch):
    f=locked_job;prov={'source_path':str((f['base']/'model.json').relative_to(ROOT)),'source_sha256':'a'*64}
    audits=[{'identity_match':True,'catalog_provenance':prov}]
    summary={'status':'complete','all_sources_accounted':True,'verified_sources':1,'expected_sources':1,
             'results':[],'skipped_license_unresolved':[]}
    with pytest.raises(ValueError,match='Complete original JSON source set'):
        pipeline.verify_source_repairs(summary,audits)


@pytest.mark.parametrize('accepted',[False,True])
def test_blocked_plan_needs_independent_source_bound_review(locked_job,monkeypatch,accepted):
    from recommendation_execution_repairs import engineering_gate
    f=locked_job;job=deepcopy(f['job']);result=deepcopy(f['result'])
    job['plan'].update(status='blocked',reasons=['Blocked:some reason'])
    result.update(status='blocked',prediction_locked=False,reasons=['Blocked:some reason'])
    monkeypatch.setattr(engineering_gate,'assess_blocked_plan',lambda plan:{'accepted':accepted,'category':'fixture'})
    sources={'status':'complete','all_sources_accounted':True,'verified_sources':0,'expected_sources':0,'results':[]}
    actual=pipeline.engineering_completeness(sources,[job],[result])
    assert actual['accepted'] is accepted
    assert len(actual['verified_unavailable_cases'])==int(accepted)


def test_source_worker_join_is_limited_to_registered_project_process(monkeypatch,locked_job):
    f=locked_job
    source_worker=pipeline.SOURCE_PROGRESS.parent.name.replace('snapshot_repairs_','source_repairs_')
    directory=f['run'].parent/source_worker;directory.mkdir()
    script=directory/'run.py';script.write_text('# registered project-only source worker')
    identity={'pid':987654321,'started':'fixture'}
    write(directory/'pid.json',{'script':str(script.relative_to(ROOT)),'identity':identity})
    write(directory/'status.json',{'status':'complete'})
    monkeypatch.setattr(pipeline.original,'identity',lambda pid:None)
    controller=object.__new__(pipeline.Controller)
    controller.await_existing_source_repairs()
    write(directory/'status.json',{'status':'failed'})
    with pytest.raises(ValueError,match='stopped or failed'):controller.await_existing_source_repairs()
    write(directory/'pid.json',{'script':'other/project/run.py','identity':identity})
    with pytest.raises(ValueError,match='registration differs'):controller.await_existing_source_repairs()
