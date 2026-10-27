"""Integration boundaries of the frozen 10-to-30 sampling expansion."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_sampling_expansion as runner


@pytest.fixture(autouse=True)
def restore_runtime_module_context(monkeypatch):
    # Runtime aliases are intentional; tests must not alter legacy runner tests.
    for key in ('RUN', 'data', 'Guard'):
        monkeypatch.setattr(runner.gap, key, getattr(runner.gap, key))


def test_runtime_adapter_routes_only_expansion_context(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'RUN', tmp_path/'new')
    frozen = {path: path.read_bytes() for path in (
        ROOT/'scripts/run_evidence_gap_design.py', ROOT/'scripts/blueprint_gap_design_data.py')}
    runner.configure()
    assert runner.gap.RUN == tmp_path/'new'
    assert runner.gap.data is runner.data
    assert runner.gap.data.reveal_one is runner.split_data.reveal_one
    assert runner.gap.Guard is runner.Guard
    assert all(path.read_bytes() == raw for path, raw in frozen.items())


@pytest.fixture
def synthetic_training(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'RUN', tmp_path)
    features, records = {}, []
    cases = [dict(case_id=f'old{i}') for i in range(10)]
    for case in cases:
        cid=case['case_id']
        for regime in runner.gap.REGIMES:
            key=cid+'__'+regime
            features[key]=dict(case_id=cid,cluster_id=cid,candidate_ids=[4,5],goals={
                g:dict(X=[list(np.arange(8,dtype=float)),list(np.arange(8,dtype=float)+1)]) for g in runner.GOALS})
            for candidate,error in [('baseline',3.),(key+'__point_04',2.),(key+'__point_05',4.)]:
                records.append(dict(case_id=cid,design_key=key,candidate=candidate,
                    endpoints={g:dict(error=error) for g in runner.GOALS}))
    # These tempting foreign labels must never be passed to a learner.
    records.append(dict(case_id='new_drug',design_key='new_drug__late_four',candidate='baseline',
                        endpoints={g:dict(error=99999.) for g in runner.GOALS}))
    docs={'protocol':dict(cases=cases,source_pins={},cached_scores='labels'),
          'features':features,'labels':dict(records=records)}
    state=dict(protocol='protocol',features='features',decisions='old_oof')
    monkeypatch.setattr(runner,'read',lambda path:deepcopy(state))
    monkeypatch.setattr(runner,'checked',lambda name:deepcopy(docs[name]))
    monkeypatch.setattr(runner,'verify',lambda pins:None)
    saved=[]
    monkeypatch.setattr(runner,'save',lambda path,doc:saved.append(deepcopy(doc)))
    return docs,saved


def test_final_rankers_train_only_exact_original_ten(synthetic_training):
    docs,saved=synthetic_training
    result=runner.freeze_rankers()
    assert result['trained_case_ids']==sorted(c['case_id'] for c in docs['protocol']['cases'])
    assert result['added_drug_labels_used'] is False
    assert len(saved)==1
    for model in result['models'].values():
        assert model['train_drugs']==10 and model['train_rows']==40
        assert model['alpha']==1.


def test_ten_rows_from_wrong_drug_set_cannot_train(synthetic_training):
    docs,_=synthetic_training
    docs['protocol']['cases'][0]['case_id']='unseen_original'
    with pytest.raises(ValueError,match='original ten'):
        runner.freeze_rankers()


def test_changed_added_labels_leave_frozen_models_unchanged(synthetic_training):
    docs,_=synthetic_training
    first=runner.freeze_rankers()
    docs['labels']['records'][-1]['endpoints']={'bad':'not even endpoint data'}
    second=runner.freeze_rankers()
    assert first==second


@pytest.fixture
def scored_matrix(monkeypatch):
    cases=[dict(case_id='old',drug='Old',cluster_id='shared',expansion_cohort='previous_ten'),
           dict(case_id='new',drug='New',cluster_id='new_trial',expansion_cohort='new_ten_source_curated')]
    p=dict(cases=cases,original_case_ids=['old']); old={'designs':{}}; state={'designs':{}}
    docs={};records=[];policies={}
    for case in cases:
        for regime in runner.gap.REGIMES:
            key=case['case_id']+'__'+regime
            d=dict(candidate_ids=[4,5],candidate_times_h=[1.,2.])
            docs[key]=d;(old if case['case_id']=='old' else state)['designs'][key]={'result':key}
            policies[key]={g:{m:0 for m in runner.METHODS} for g in runner.GOALS}
            for candidate,error in [('baseline',3.),(key+'__point_04',2.),(key+'__point_05',4.)]:
                eps={ep:dict(value=error,reference=1.,error=error,unit=u,weak_tail=False,
                            reference_kind='observed_curve_nca') for ep,u in runner.data.UNITS.items()}
                records.append(dict(case_id=case['case_id'],design_key=key,candidate=candidate,
                    endpoints=eps,curve_metrics={'gmfe':error}))
    monkeypatch.setattr(runner,'read',lambda path:deepcopy(old))
    monkeypatch.setattr(runner,'checked',lambda name:deepcopy(docs[name]))
    return p,state,records,policies


def test_old_oof_and_new_fixed_policy_all_eleven_endpoints_kept(scored_matrix):
    rows=runner.endpoint_rows(*scored_matrix)
    assert len(rows)==2*2*3*len(runner.METHODS)*11
    assert {r['cohort'] for r in rows}=={'previous_ten','new_ten_source_curated'}
    assert {r['endpoint'] for r in rows}=={'curve',*runner.data.UNITS}
    learned=[r for r in rows if r['method']=='learned_gain']
    assert all(r['selected_time_h']==1. for r in learned)
    assert all(r['error']==2. for r in learned)


def test_exact_random_log_error_and_tmax_arithmetic_error(scored_matrix):
    rows=runner.endpoint_rows(*scored_matrix)
    random=[r for r in rows if r['method']=='uniform_random']
    assert all(r['prediction'] is None and r['selected_time_h'] is None for r in random)
    assert all(r['error']==pytest.approx(3. if r['endpoint']=='tmax' else np.sqrt(8.)) for r in random)


def test_failed_action_prevents_false_exact_random_success(scored_matrix):
    p,state,records,policies=scored_matrix
    records[-1]['endpoints']['cmax']['error']=None
    rows=runner.endpoint_rows(p,state,records,policies)
    failed=[r for r in rows if r['case_id']=='new' and r['regime']=='late_four'
            and r['requested_goal']=='cmax' and r['method']=='uniform_random' and r['endpoint']=='cmax']
    assert len(failed)==1 and failed[0]['error'] is None


def test_missing_tail_reference_does_not_remove_cmax_rows(scored_matrix):
    p,state,records,policies=scored_matrix
    for rec in records:
        for ep in ('terminal_half_life','auc_inf'):
            rec['endpoints'][ep].update(error=None,reference=None)
    rows=runner.endpoint_rows(p,state,records,policies)
    assert all(r['error'] is not None for r in rows if r['endpoint']=='cmax' and r['requested_goal']=='cmax')
    assert all(r['error'] is None for r in rows if r['endpoint']=='terminal_half_life')


def test_report_refuses_unlocked_outcomes_before_reader(monkeypatch):
    state=dict(status='acquiring',protocol='p',units={'x':{}},designs={})
    p=dict(jobs={'x':{}},designs={})
    monkeypatch.setattr(runner,'read',lambda path:state)
    monkeypatch.setattr(runner,'checked',lambda ref:p)
    monkeypatch.setattr(runner,'Guard',lambda p:lambda **kwargs:None)
    def forbidden(*args):
        pytest.fail('Clinical outcomes read before full prediction lock')
    monkeypatch.setattr(runner,'all_records',forbidden)
    with pytest.raises(ValueError,match='must lock'):
        runner.report()


def test_same_count_wrong_job_keys_rejected_before_any_source_read(monkeypatch):
    p=dict(jobs={'expected':{}},designs={'design':{}})
    state=dict(status='predictions_locked',units={'wrong':{}},designs={'design':{}},
               predictions_locked_at=1.,action_lock='lock')
    monkeypatch.setattr(runner,'checked',lambda ref:pytest.fail('Sources read for incomplete attempt matrix'))
    with pytest.raises(ValueError,match='must lock'):
        runner.validate_prediction_lock(p,state)


def test_changed_global_choice_lock_rejected(monkeypatch):
    p=dict(jobs={},designs={})
    state=dict(status='predictions_locked',units={},designs={},protocol='p',predictions_locked_at=1.,action_lock='lock')
    monkeypatch.setattr(runner,'checked',lambda ref:dict(protocol='other',designs={}))
    with pytest.raises(ValueError,match='Global action lock'):
        runner.validate_prediction_lock(p,state)


def test_all_command_never_executes_after_failed_prepare(monkeypatch):
    calls=[]
    def prepare(n):
        calls.append(('prepare',n));raise ValueError('source problem')
    monkeypatch.setattr(runner,'prepare',prepare)
    monkeypatch.setattr(runner,'execute',lambda n:calls.append(('run',n)))
    with pytest.raises(ValueError,match='source problem'):
        runner.run_all(12)
    assert calls==[('prepare',12)]


def test_all_command_prepares_then_runs_with_same_worker_limit(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,'prepare',lambda n:calls.append(('prepare',n)))
    monkeypatch.setattr(runner,'execute',lambda n:calls.append(('run',n)) or {'status':'complete'})
    assert runner.run_all(12)=={'status':'complete'}
    assert calls==[('prepare',12),('run',12)]


def test_real_thirty_request_contract_metadata_preflight():
    """Real source compatibility only; no fitting or clinical error scoring."""
    cohort=runner.read(runner.COHORT)
    assert len({c['compound_id'] for c in cohort['cases']})>=30
    assert cohort['algorithm_outcomes_used_to_select_cases'] is False
    for case in cohort['cases']:
        request=runner.gap.request(case,'target','PKSimStandard')
        contract=runner.data.build_target_contract(case,runner.checked(case['reference_source']),request)
        assert contract['target_observation_values_used'] is False
        assert len(contract['sampling_time_h'])>=2


def test_real_thirty_development_sources_explicitly_declared():
    cohort=runner.read(runner.COHORT)
    for case in cohort['cases']:
        runner.declare_development_source(case)
        request=runner.gap.request(case,'development','PKSimStandard')
        full=runner.base._training_document(case['sampling_sources']['full'],request,ROOT)
        assert full['role']=='calibration_data_reconstruction'


@pytest.mark.parametrize('source_role,reference_role', [('validation',None),
    (None,None), ('calibration_data_reconstruction','validation')])
def test_cannot_relabel_target_as_development(monkeypatch,source_role,reference_role):
    record={'path':'unused','sha256':'unused'}
    if reference_role is not None:
        record['role']=reference_role
    case={'sampling_sources':{'full':record}}
    monkeypatch.setattr(runner,'checked',lambda rec:{'role':source_role})
    with pytest.raises(ValueError,match='explicitly declare'):
        runner.declare_development_source(case)
