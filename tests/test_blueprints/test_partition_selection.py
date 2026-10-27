"""Selection isolation, evidence immutability, and complete failure denominators."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import partition_selection as selection


def metric(error=1., auc=1.2, peak=1.3):
    return {'log_rmse':error,'auc':{'fold_error':auc},'sampled_cmax':{'fold_error':peak}}


@pytest.mark.parametrize('candidate,expected', [
    (metric(.99,1.212,1.313),selection.CANDIDATE),
    (metric(.990001),selection.BASELINE),
    (metric(.7,1.213,1.3),selection.BASELINE),
    (metric(.7,1.2,1.314),selection.BASELINE),
    (metric(.7,1.1,1.1),selection.CANDIDATE),
    (None,selection.BASELINE),
    ({},selection.BASELINE),
    (metric(float('nan')),selection.BASELINE),
    (metric(-1),selection.BASELINE),
    (metric(.7,.99),selection.BASELINE),
    (metric(True),selection.BASELINE),
])
def test_declared_three_gates(candidate,expected):
    result=selection.choose_partition(metric(),candidate)
    assert result['selected_partition_method']==expected
    assert result['comparison_used_for_selection'] is False


def test_zero_and_tiny_baseline_errors_do_not_permit_worsening():
    assert selection.choose_partition(metric(0),metric(0))['selected_partition_method']==selection.BASELINE
    assert selection.choose_partition(metric(1e-15),metric(2e-15))['selected_partition_method']==selection.BASELINE


def request(drug):
    return {'entry_reason':'model_missing','compound_id':drug,'compound_name':'Example',
        'human':{'age_years':30.,'weight_kg':70.,'height_cm':175.,'sex':'MALE','population':'European_ICRP_2002'},
        'dosing':{'route':'intravenous','dose_mg':1.,'start_h':0.,'infusion_minutes':1.},
        'target':{'endpoints':['auc','cmax'],'primary_endpoint':'auc','window_start_h':0.,'window_end_h':1.},
        'partition_method':'PKSimStandard','evidence_ledger':{'path':'ledger.json','sha256':'0'*64,'location':'evidence'},'evidence':[]}


def test_only_partition_field_changes():
    original=request('x');before=deepcopy(original)
    candidate=selection.candidate_request(original)
    assert original==before
    candidate['partition_method']='PKSimStandard'
    assert candidate==original
    with pytest.raises(ValueError):selection.candidate_request({**original,'partition_method':'RodgersRowland'})


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def baseline(tmp_path):
    base=tmp_path/'baseline';cases=[];pins={};predictions={}
    for identifier,drug,role in [('fit','drug-a',selection.CALIBRATION),('compare','drug-a',selection.COMPARISON),('no_fit','drug-b',selection.COMPARISON)]:
        op=tmp_path/(identifier+'_observed.json')
        save(op,{'case_id':identifier,'observations':[{'time_min':1.,'concentration_umol_per_l':1.},{'time_min':60.,'concentration_umol_per_l':.5}]})
        pins[str(op.relative_to(tmp_path))]=selection.sha256(op)
        prediction=base/identifier/'prediction_locked.json';save(prediction,{'kind':'baseline','case_id':identifier})
        predictions[identifier]=selection.sha256(prediction)
        cases.append({'case_id':identifier,'compound_id':drug,'compound_name':'Example','study_id':identifier,'role':role,'observations_path':str(op.relative_to(tmp_path)),'request':request(drug)})
    save(base/'protocol.json',{'pins':pins,'cases':cases})
    save(base/'predictions_lock.json',{'locked_before_scoring':True,'predictions':predictions})
    save(base/'summary.json',{'status':'completed','code_and_inputs_unchanged':True,'predictions_unchanged_after_scoring':True,
          'results':'POISON: no metrics are read from this summary'})
    return tmp_path,base,tmp_path/'selection'


def instrument(monkeypatch,root,out,*,fail_comparison=False,change_evidence=False):
    events=[]
    def run(*,request,output_dir,project_root,execute):
        identifier=output_dir.name;events.append(('run',identifier))
        assert request['partition_method']==selection.CANDIDATE
        if identifier!='fit':
            assert (out/'choice_locked.json').is_file()
            lock=json.loads((out/'choice_locked.json').read_text())
            assert lock['choices']['drug-a']['selected_partition_method']==selection.CANDIDATE
            assert lock['choices']['drug-b']['selected_partition_method']==selection.BASELINE
        if change_evidence:
            save(root/'compare_observed.json',{'changed':True})
        if identifier=='compare' and fail_comparison:
            raise RuntimeError('mock candidate native failure')
        save(output_dir/'prediction_locked.json',{'kind':'candidate','case_id':identifier})
        return {'status':'predicted','native_executed':True}
    def score(curve,observed,req):
        identifier=observed['case_id'];events.append(('score',identifier))
        if identifier!='fit':assert (out/'choice_locked.json').is_file()
        # Candidate deliberately overfits: improves calibration, worsens comparison.
        error=1. if curve['kind']=='baseline' else .8 if identifier=='fit' else 5.
        return {**metric(error),'gmfe':1.+error,'c_last':{'fold_error':1.5}}
    monkeypatch.setattr(selection,'run_evidence_blueprint',run)
    monkeypatch.setattr(selection,'score_case',score)
    return events


def test_choices_locked_before_any_comparison_and_no_refitting(monkeypatch,baseline):
    root,base,out=baseline
    events=instrument(monkeypatch,root,out)
    summary=selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root,max_workers=1)
    assert summary['status']=='completed'
    assert summary['attempted_cases']==len(summary['candidate_results'])==2
    assert len(summary['selected_results'])==3
    assert summary['baseline_only_cases']==[{'case_id':'no_fit','status':'not_planned_no_calibration'}]
    assert summary['choices']['drug-a']['selected_partition_method']==selection.CANDIDATE
    assert summary['choices']['drug-b']['selected_partition_method']==selection.BASELINE
    selected={r['case_id']:r for r in summary['selected_results']}
    assert selected['compare']['metrics']['log_rmse']==5. # no comparison-driven rollback
    assert selected['no_fit']['metrics']['log_rmse']==1.
    assert [e for e in events if e[0]=='run']==[('run','fit'),('run','compare')]
    assert (out/'comparison_candidates_prediction_lock.json').is_file()


def test_native_failure_keeps_denominator_and_no_retrospective_fallback(monkeypatch,baseline):
    root,base,out=baseline
    instrument(monkeypatch,root,out,fail_comparison=True)
    summary=selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root,max_workers=2)
    assert summary['status']=='completed_with_failures'
    assert summary['attempted_cases']==2
    assert len(summary['selected_results'])==3
    compare=next(r for r in summary['selected_results'] if r['case_id']=='compare')
    assert compare['selected_partition_method']==selection.CANDIDATE
    assert compare['result']['status']=='failed'


def test_midrun_source_change_stops_before_choice_and_retains_skipped_cases(monkeypatch,baseline):
    root,base,out=baseline
    events=instrument(monkeypatch,root,out,change_evidence=True)
    summary=selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root,max_workers=1)
    assert summary['status']=='completed_with_failures'
    assert summary['code_and_inputs_unchanged'] is False
    assert len(summary['candidate_results'])==2
    assert not (out/'choice_locked.json').exists()
    assert [e for e in events if e[0]=='run']==[('run','fit')]


@pytest.mark.parametrize('issue',['not_completed','changed_observation','changed_prediction','two_calibrations','role_overlap'])
def test_preflight_blocks_incomplete_or_tampered_baseline(monkeypatch,baseline,issue):
    root,base,out=baseline
    if issue=='not_completed':
        save(base/'summary.json',{'status':'running'})
    elif issue=='changed_observation':save(root/'fit_observed.json',{'changed':True})
    elif issue=='changed_prediction':save(base/'fit/prediction_locked.json',{'changed':True})
    else:
        path=base/'protocol.json';protocol=json.loads(path.read_text())
        if issue=='two_calibrations':protocol['cases'][1]['role']=selection.CALIBRATION
        else:protocol['cases'][1]['study_id']='fit'
        save(path,protocol)
    def forbidden(**kwargs):pytest.fail('Native solver must not start')
    monkeypatch.setattr(selection,'run_evidence_blueprint',forbidden)
    with pytest.raises(ValueError):selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root)
    assert not out.exists()


def test_output_overwrite_and_worker_limits(baseline):
    root,base,out=baseline
    for workers in (0,5,True,1.5):
        with pytest.raises(ValueError):selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root,max_workers=workers)
    out.mkdir()
    with pytest.raises(FileExistsError):selection.run_partition_selection(baseline_dir=base,output_dir=out,project_root=root)
