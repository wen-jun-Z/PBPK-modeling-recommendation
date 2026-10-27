"""Joint-input protection, native reuse identity and scorer isolation contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_joint_evidence_refinement.py'
SPEC = importlib.util.spec_from_file_location('joint_refinement_test', SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def task(independent=True):
    def row(parameter, value, basis='predicted'):
        return {'parameter':parameter, 'compound_id':'c', 'basis':basis, 'value':value, 'unit':'1',
                'context':{'species':'human', 'study_role':'independent'}}
    return {'request':{'compound_id':'c', 'human':{'weight_kg':70}, 'evidence_ledger':{'path':'old'},
        'evidence':[row('total_clearance', 2), row('fraction_unbound', .3, 'measured' if independent else 'predicted'),
                    row('molecular_weight', 200, 'measured')]},
        'sidecars':{'vss':[row('vss',2)], 'half':[row('terminal_half_life',4)]},
        'core':{'design_version':'learned-exchange-design-2'}}


def policy(monkeypatch, change):
    def apply(request, sidecars, artifact, receipt):
        original_request, original_sidecars = deepcopy(request), deepcopy(sidecars)
        change(request, sidecars)
        return request, sidecars, {'request_modified':request!=original_request,
                                  'sidecars_modified':sidecars!=original_sidecars}
    monkeypatch.setattr(r.importlib, 'import_module', lambda name:SimpleNamespace(condition_inputs=apply))


def test_only_joint_directory_is_owned():
    assert r.own(r.DEFAULT)==r.DEFAULT
    for path in (r.BATCH, r.SOURCE, r.BATCH/'conditional', r.BATCH/'soft_half'):
        with pytest.raises(ValueError):
            r.own(path)


def test_predicted_cl_and_sidecars_can_change_without_mutating_original(monkeypatch):
    original = task(); saved = deepcopy(original)
    def change(request, sidecars):
        request['evidence'][0]['value'] = 3.
        sidecars['vss'][0]['value'] = 1.7
    policy(monkeypatch, change)
    result, audit = r.transformed_task(original, {}, {})
    assert original==saved and result['core']==original['core']
    assert result['request']['evidence'][1:]==original['request']['evidence'][1:]
    assert result['request']['evidence_ledger']==original['request']['evidence_ledger']
    assert audit=={'request_modified':True, 'sidecars_modified':True}


@pytest.mark.parametrize('field', ['measured_fu','context','renal','human','basis','measured_sidecar'])
def test_independent_and_nonnumeric_inputs_are_protected(monkeypatch, field):
    original = task()
    original['request']['evidence'].append({'parameter':'renal_clearance','basis':'derived','value':1})
    original['sidecars']['half'][0]['basis']='derived'
    def change(request, sidecars):
        if field=='measured_fu': request['evidence'][1]['value']=.2
        elif field=='context': request['evidence'][0]['context']['study_role']='target'
        elif field=='renal': request['evidence'][-1]['value']=2
        elif field=='human': request['human']['weight_kg']=80
        elif field=='basis': request['evidence'][0]['basis']='measured'
        else: sidecars['half'][0]['value']=2
    policy(monkeypatch, change)
    with pytest.raises(ValueError): r.transformed_task(original, {}, {})


def test_predicted_total_cl_cannot_change_with_independent_renal_split(monkeypatch):
    original=task(); original['request']['evidence'].append({'parameter':'renal_clearance','basis':'derived','value':1})
    policy(monkeypatch, lambda req,side: req['evidence'][0].update(value=3))
    with pytest.raises(ValueError, match='organ split'): r.transformed_task(original, {}, {})


def test_no_independent_input_is_exact_identity_and_cannot_be_shifted(monkeypatch):
    original=task(False)
    policy(monkeypatch, lambda req,side: None)
    result,_=r.transformed_task(original,{},{}); assert result==original
    policy(monkeypatch, lambda req,side: req['evidence'][0].update(value=3))
    with pytest.raises(ValueError, match='Without independent'): r.transformed_task(original,{}, {})


def test_native_reuse_requires_exact_ledger_and_adapter_identity():
    request=task()['request']
    native={'request_sha256':r.value.model._digest(request),'partition_method':'PKSimStandard',
            'adapter_sha256':r.sha256(Path(r.value.model.__file__))}
    assert r.reusable_native(native,request)
    changed=deepcopy(request);changed['evidence_ledger']['path']='new'
    assert not r.reusable_native(native,changed)
    assert not r.reusable_native({**native,'adapter_sha256':'old'},request)


def test_fresh_ledger_contains_changed_evidence_and_prior_receipt(tmp_path,monkeypatch):
    original=task(); original.update(key='a'*64)
    monkeypatch.setattr(r,'own',lambda path:path)
    pins={}; prior={'path':'prior','sha256':'hash'}
    r.fresh_ledger(original,tmp_path,prior,pins)
    receipt=original['request']['evidence_ledger']; data=r.value.old.checked(receipt)
    assert data['evidence']==original['request']['evidence'] and data['conditioning_prior']==prior
    assert pins[receipt['path']]==receipt['sha256']


@pytest.mark.parametrize('native_reused',[True,False])
def test_solver_only_passes_an_existing_native_for_exact_requests(tmp_path,monkeypatch,native_reused):
    (tmp_path/'models').mkdir()
    request=task()['request'];key='f'*64
    native={'request_sha256':r.value.model._digest(request),'partition_method':'PKSimStandard',
            'adapter_sha256':r.sha256(Path(r.value.model.__file__))}
    r.save(tmp_path/'native.json',native); receipt=r.value.old.ref(tmp_path/'native.json')
    work={'key':key,'case_id':'c','request':request,'sidecars':{'vss':[],'half':[]},
          'native_source_ref':receipt if native_reused else None,'reuse_value_receipt':None,'candidate':'joint'}
    monitor=SimpleNamespace(lock=threading.RLock(),state={'outcomes':{}},check=lambda:None,halted=threading.Event())
    monitor.update=lambda **kw:monitor.state.update(kw)
    calls=[]
    def execute(req,vss,half,folder,root,*,native_result=None):
        calls.append(native_result)
        return {'status':'failed','reason':'retained_scientific_failure','native_solver_calls':0 if native_result else 1}
    monkeypatch.setattr(r,'_executor',lambda task:execute)
    r.solve(work,tmp_path,monitor)
    result=r.value.old.checked(monitor.state['outcomes'][key])
    assert (calls[0] is not None)==native_reused
    assert result['new_native_calls']==int(not native_reused)
    assert result['entry']['failure']=='retained_scientific_failure'
    assert result['entry']['target_observations_used'] is False


def test_report_refuses_unlocked_predictions_before_target_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'own',lambda path:path)
    monkeypatch.setattr(r.value.old,'verify',lambda protocol:None)
    r.save(tmp_path/'protocol.json',{'tasks':{'missing':{}}})
    r.save(tmp_path/'state.json',{'protocol':r.value.old.ref(tmp_path/'protocol.json'),
        'all_predictions_locked_before_scoring':False,'outcomes':{}})
    def forbidden(*args): raise AssertionError('Target observation was opened before predictions locked')
    monkeypatch.setattr(r.value.evaluation,'clinical_truth',forbidden)
    with pytest.raises(ValueError,match='Lock every prediction'):
        r.report(tmp_path)
