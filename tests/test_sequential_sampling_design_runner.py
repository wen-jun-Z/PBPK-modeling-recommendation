"""Synthetic isolation, resume and shared-budget checks; no clinical outcomes."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT / 'scripts'))
    spec = importlib.util.spec_from_file_location(
        'sequential_runner_test', PROJECT / 'scripts/run_sequential_sampling_design.py')
    r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'RUN', tmp_path / 'run'); r.RUN.mkdir()

    def ref(path):
        path = Path(path)
        return dict(path=str(path.relative_to(tmp_path)), sha256=r.base.sha256(path))

    def checked(source):
        path = tmp_path / source['path']
        if r.base.sha256(path) != source['sha256']:
            raise ValueError('Changed source hash: ' + source['path'])
        return r.read(path)

    def verify(pins):
        for name, digest in pins.items():
            if r.base.sha256(tmp_path / name) != digest:
                raise ValueError('Changed source hash: ' + name)

    def pins_in(value, pins):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                checked(value); pins[value['path']] = value['sha256']
            for item in value.values():
                pins_in(item, pins)
        elif isinstance(value, list):
            for item in value:
                pins_in(item, pins)

    def request(case, phase, formula):
        result = deepcopy(case['requests']['evidence'][phase])
        result['partition_method'] = formula
        return result

    monkeypatch.setattr(r, 'ref', ref)
    monkeypatch.setattr(r, 'checked', checked)
    monkeypatch.setattr(r, 'verify', verify)
    monkeypatch.setattr(r.base, 'pins_in', pins_in)
    monkeypatch.setattr(r, 'request', request)
    return r


def setup(r, monkeypatch, *, n=8, feasible=True, layouts=(0,)):
    dev = dict(compound_id='synthetic', human={'weight_kg': 70.},
        dosing=dict(route='intravenous', dose_mg=10., start_h=0., infusion_minutes=30.),
        evidence=[dict(parameter='molecular_weight', value=500., unit='g/mol')])
    target = deepcopy(dev); target['dosing']['dose_mg'] = 20.
    full = dict(compound_id='synthetic', study_id='development', role='calibration_data_reconstruction',
        schema_version='clinical-development-1', compound_name='Synthetic', study_label='Development only',
        matrix='plasma', time_unit='min', unit='umol/l', human=deepcopy(dev['human']),
        dosing=deepcopy(dev['dosing']), molecular_weight_g_mol=500.,
        observations=[dict(time_min=float(i), concentration_umol_per_l=10./i) for i in range(1,n+1)],
        raw_observations={'forbidden': True})
    r.save(r.RUN / 'full.json', full)
    case = dict(case_id='synthetic', sampling_sources={'full': r.ref(r.RUN / 'full.json')},
        requests={'evidence': {'development': dev, 'target': target}})
    specs = {}
    for index in layouts:
        key = f'synthetic__layout{index}'
        split = r.sequential.split_development(full, dev, index)
        r.save(r.RUN / (key + '.json'), split)
        specs[key] = dict(case_id='synthetic', layout_index=index, source=r.ref(r.RUN / (key + '.json')))
    p = dict(version=r.VERSION, source_pins={}, cases=[case], layouts=specs,
        methods=list(r.METHODS), budgets=[0,1,2,3], goal='cmax', fit_budget_per_formula=16384,
        fit_starts=16, physiology_cache={f'synthetic__development__{f}': {'phys': f} for f in r.base.FORMULAS})
    r.base.pins_in(case['sampling_sources'], p['source_pins'])
    for item in specs.values():
        r.base.pins_in(item['source'], p['source_pins'])
    r.save(r.RUN / 'protocol.json', p)
    monkeypatch.setattr(r.prior, 'get_phys', lambda *a: {'synthetic': True})
    monkeypatch.setattr(r.base, '_training_document', lambda source, req, root: r.checked(source))
    calls = []

    def fit(predict, req, training, physiology, *, variant, budget, starts, before_trial):
        # No previous/full fit, candidate concentrations or warm-start hook can
        # enter the real-data multistart fitter through this call contract.
        assert 'raw_observations' not in training
        assert req['dosing']['dose_mg'] == 10.
        assert variant == 'blood_partition' and budget == 16384 and starts == 16
        before_trial()
        calls.append(deepcopy(training))
        return dict(best=dict(objective=float(len(calls)), log_parameters=[0.]*4) if feasible else None,
            calls=3, failures=1, starts=[{'best': None} for _ in range(starts)],
            no_warm_start=True, target_observations_used=False, variant=variant,
            call_limit=budget, visible_points=len(training['observations']))

    monkeypatch.setattr(r.old.optimizer, 'fit', fit)
    monkeypatch.setattr(r.base.engine, 'prediction_grid', lambda req: np.array([.1,1.]))
    monkeypatch.setattr(r.old, 'simulate', lambda *a: dict(concentration_mg_l=np.ones(2), mass_balance={}))
    return p, case, full, calls


def initial(r, p, index=0):
    return r.checked(p['layouts'][f'synthetic__layout{index}']['source'])


def receipt(r, *, status='selected'):
    fitref = dict(path='synthetic-fit', sha256='hash')
    visible = [0,1,3,4]
    menu = dict(candidate_ids=[2,5,6,7], candidate_times_h=[.05,.1,7/60,8/60])
    action = dict(protocol=r.ref(r.RUN / 'protocol.json'), method='time_farthest', fit=fitref,
        visible_ids=visible, candidate_ids=menu['candidate_ids'], candidate_times_h=menu['candidate_times_h'],
        hidden_values_used=False, target_values_used=False, status=status,
        selected_index=0, selected_id=2, selected_time_h=.05, locked_at=1.)
    return action, fitref, visible, menu


def test_cache_is_set_specific_and_cross_method_reusable(runner, monkeypatch):
    r = runner; p, case, full, calls = setup(r, monkeypatch)
    split = initial(r,p); ids = split['design_only']['visible_ids']; training=split['visible_training']
    baseline = r.cached_fit(p,case,training,ids,r.Guard(p))
    assert len(calls)==2
    assert r.cached_fit(p,case,training,list(reversed(ids)),r.Guard(p)) == baseline
    assert len(calls)==2, 'Reordered identical point IDs should share both formula fits'
    new_id = split['design_only']['candidate_ids'][0]
    augmented = r.sequential.reveal(full,training,new_id,case['requests']['evidence']['development'])
    augmented_ref = r.cached_fit(p,case,augmented,ids+[new_id],r.Guard(p))
    assert augmented_ref != baseline and len(calls)==4
    assert [len(x['observations']) for x in calls] == [4,4,5,5]
    result = r.checked(augmented_ref)
    assert result['resources']['objective_calls']==6 and result['resources']['failed_calls']==2


def test_actual_target_simulation_follows_selection_lock(runner, monkeypatch):
    r = runner; p,case,_,calls = setup(r,monkeypatch)
    split=initial(r,p); ids=split['design_only']['visible_ids']
    folder=r.RUN/'fits'/case['case_id']/r.subset_key(case,ids,p)
    def simulate(variant,phys,req,times,parameters):
        selected=r.read(folder/'selection.json')
        assert selected['target_values_used'] is False
        assert req['dosing']['dose_mg']==20.
        assert len(calls)==2 and selected['formula']=='PKSimStandard'
        return dict(concentration_mg_l=np.ones(len(times)),mass_balance={})
    monkeypatch.setattr(r.old,'simulate',simulate)
    assert r.checked(r.cached_fit(p,case,split['visible_training'],ids,r.Guard(p)))['status']=='predicted'


@pytest.mark.parametrize('kind',['ids','request','source','protocol'])
def test_fit_cache_key_binds_all_inputs(runner,monkeypatch,kind):
    r=runner;p,case,_,_=setup(r,monkeypatch);ids=[0,1,3,4]
    before=r.subset_key(case,ids,p); changed=deepcopy(case)
    if kind=='ids': ids=[0,1,3,5]
    elif kind=='request': changed['requests']['evidence']['development']['dosing']['dose_mg']=11.
    elif kind=='source': changed['sampling_sources']['full']['sha256']='different'
    else: r.save(r.RUN/'protocol.json',{**p,'version':'changed'})
    assert r.subset_key(changed,ids,p)!=before


def test_cached_training_changed_at_identical_ids_rejected(runner,monkeypatch):
    r=runner;p,case,_,_=setup(r,monkeypatch);split=initial(r,p)
    training=split['visible_training'];ids=split['design_only']['visible_ids']
    r.cached_fit(p,case,training,ids,r.Guard(p))
    changed=deepcopy(training);changed['observations'][0]['concentration_umol_per_l']*=2
    with pytest.raises(ValueError,match='visible data'):
        r.cached_fit(p,case,changed,ids,r.Guard(p))


@pytest.mark.parametrize('kind',['identity','protocol','artifact'])
def test_finished_fit_resume_rejects_wrong_identity_or_changed_artifact(runner,monkeypatch,kind):
    r=runner;p,case,_,_=setup(r,monkeypatch);split=initial(r,p)
    ids=split['design_only']['visible_ids'];train=split['visible_training']
    source=r.cached_fit(p,case,train,ids,r.Guard(p));folder=(r.ROOT/source['path']).parent
    identity=dict(case_id='synthetic',visible_ids=sorted(ids))
    if kind=='identity': identity['visible_ids']=[0,1,2,3]
    elif kind=='protocol': r.save(r.RUN/'protocol.json',{**p,'version':'changed'})
    else: r.save(folder/'prediction.json',{'changed':True})
    with pytest.raises(ValueError):
        r.fit_predict(p,case,train,folder,r.Guard(p),identity)


def test_partial_training_cannot_be_overwritten(runner,monkeypatch):
    r=runner;p,case,_,_=setup(r,monkeypatch);split=initial(r,p);folder=r.RUN/'partial'
    r.save(folder/'training.json',{'wrong':True})
    with pytest.raises(ValueError,match='partially completed'):
        r.fit_predict(p,case,split['visible_training'],folder,r.Guard(p),{'id':1})
    assert r.read(folder/'training.json')=={'wrong':True}


@pytest.mark.parametrize('field,value',[
    ('request',{}),('no_warm_start',False),('call_limit',2),('visible_points',20),
    ('variant','other'),('target_observations_used',True),('calls',16385),
    ('development_physiology',{}),('training_source',{}),('starts',[]),
])
def test_partial_formula_resume_contract(runner,monkeypatch,field,value):
    r=runner;p,case,_,_=setup(r,monkeypatch);split=initial(r,p);ids=split['design_only']['visible_ids']
    ref=r.cached_fit(p,case,split['visible_training'],ids,r.Guard(p));folder=(r.ROOT/ref['path']).parent
    (folder/'result.json').unlink()
    path=folder/(r.base.FORMULAS[0]+'.json');doc=r.read(path);doc[field]=value;r.save(path,doc)
    with pytest.raises(ValueError,match='resumed formula'):
        r.fit_predict(p,case,split['visible_training'],folder,r.Guard(p),dict(case_id='synthetic',visible_ids=sorted(ids)))


def test_failed_fits_and_prediction_failure_are_retained(runner,monkeypatch):
    r=runner;p,case,_,calls=setup(r,monkeypatch,feasible=False);split=initial(r,p)
    result=r.checked(r.cached_fit(p,case,split['visible_training'],split['design_only']['visible_ids'],r.Guard(p)))
    assert result['status']=='failed' and result['resources']['prediction_calls']==0
    assert result['reason']=='No feasible formula fit' and len(result['fits'])==2 and len(calls)==2


def test_failed_target_prediction_not_silently_removed(runner,monkeypatch):
    r=runner;p,case,_,_=setup(r,monkeypatch);split=initial(r,p)
    def fail(*args): raise ValueError('Synthetic numerical failure')
    monkeypatch.setattr(r.old,'simulate',fail)
    result=r.checked(r.cached_fit(p,case,split['visible_training'],split['design_only']['visible_ids'],r.Guard(p)))
    assert result['status']=='failed' and result['reason']=='Synthetic numerical failure'
    assert result['resources']['prediction_calls']==1 and len(result['fits'])==2


@pytest.mark.parametrize('field,value',[
    ('method','random_0'),('protocol',{}),('fit',{}),('visible_ids',[0,1,2,3]),
    ('candidate_ids',[3]),('candidate_times_h',[5.]),('hidden_values_used',True),
    ('target_values_used',True),('selected_index',True),('selected_index',-1),
    ('selected_id',5),('selected_time_h',1.),('status','arbitrary'),
])
def test_action_resume_rejects_invalid_contract(runner,monkeypatch,field,value):
    r=runner;p,*_=setup(r,monkeypatch);action,fit,visible,menu=receipt(r)
    action[field]=value
    with pytest.raises(ValueError):
        r.verify_decision(p,action,fit,visible,menu,'time_farthest')


def test_source_change_stops_before_visible_fit(runner,monkeypatch):
    r=runner;p,case,_,calls=setup(r,monkeypatch)
    r.save(r.RUN/'full.json',{'changed':True})
    with pytest.raises(ValueError,match='source hash'):
        r.run_case(p,case)
    assert calls==[]


def test_sequential_lock_precedes_vault_reveal_and_all_budgets_retained(runner,monkeypatch):
    r=runner;monkeypatch.setattr(r,'METHODS',('time_farthest','random_0'))
    p,case,full,calls=setup(r,monkeypatch,layouts=(0,1))
    real_reveal=r.sequential.reveal;events=[]
    def reveal(vault,current,candidate,req):
        matching=[]
        for path in (r.RUN/'decisions').glob('*/*/*.json'):
            action=r.read(path)
            if action['status']=='selected' and action['selected_id']==candidate:
                current_times={x['time_min'] for x in current['observations']}
                expected_times={full['observations'][i]['time_min'] for i in action['visible_ids']}
                if current_times==expected_times: matching.append(path)
        assert matching,'No persisted action matched the exact visible subset before reveal'
        events.append(('reveal',len(current['observations']),candidate))
        return real_reveal(vault,current,candidate,req)
    monkeypatch.setattr(r.sequential,'reveal',reveal)
    result=r.checked(r.run_case(p,case))
    assert len(result['sequences'])==4 and len(events)==12
    assert all([s['budget'] for s in x['stages']]==[0,1,2,3] for x in result['sequences'])
    assert all(len(x['actions'])==3 for x in result['sequences'])
    assert len(calls[-1]['observations'])==8,'Full-data reference must be fitted last'
    n=len(calls);assert r.run_case(p,case)==r.ref(r.RUN/'cases/synthetic.json');assert len(calls)==n


def test_repeated_subset_is_reused_between_policies(runner,monkeypatch):
    r=runner;monkeypatch.setattr(r,'METHODS',('random_0','random_1'))
    p,case,_,calls=setup(r,monkeypatch,n=7)
    done=r.checked(r.run_case(p,case));seqs=done['sequences']
    assert seqs[0]['stages'][0]['result']==seqs[1]['stages'][0]['result']
    assert seqs[0]['stages'][3]['result']==seqs[1]['stages'][3]['result']==done['full_reference']
    assert len(calls)<=12,'Common baseline and final point sets must not be refitted per method'
    reference=r.checked(done['full_reference'])
    assert 'study_label' not in r.checked(reference['training'])


def test_known_failed_model_design_is_an_explicit_failure_receipt(runner,monkeypatch):
    r=runner;p,case,_,_=setup(r,monkeypatch,feasible=False);split=initial(r,p)
    fit=r.cached_fit(p,case,split['visible_training'],split['design_only']['visible_ids'],r.Guard(p))
    menu={k:split['design_only'][k] for k in ('candidate_ids','candidate_times_h')}
    decision=r.make_decision(p,case,fit,split['visible_training'],split['design_only']['visible_ids'],menu,'refit_goal',[],r.Guard(p))
    assert decision['status']=='failed' and decision['selected_id'] is None
    assert decision.get('reason') and decision['hidden_values_used'] is False


def test_selection_failure_keeps_all_remaining_budgets(runner,monkeypatch):
    r=runner;monkeypatch.setattr(r,'METHODS',('refit_goal',))
    p,case,_,_=setup(r,monkeypatch,feasible=False)
    done=r.checked(r.run_case(p,case));seq=done['sequences'][0]
    assert [s['budget'] for s in seq['stages']]==[0,1,2,3]
    assert all(s['result'] is None and s['status']=='selection_failed' for s in seq['stages'][1:])
    assert len(seq['actions'])==1 and r.checked(seq['actions'][0])['reason']


def test_finite_derivatives_use_declared_target_goal_only(runner):
    r=runner;coefficient=np.array([1.,2.,3.,4.])
    model=dict(x=np.zeros(4),lower=-np.ones(4),upper=np.ones(4),
        predict=lambda x,times:np.exp(np.outer(np.asarray(times),coefficient)@x),
        goal=lambda x:float(np.exp(coefficient@x)))
    basec,j,g=r.finite_derivatives(model,np.array([1.,2.]),np.array([3.]))
    np.testing.assert_allclose(basec,np.ones(3))
    np.testing.assert_allclose(j,np.outer([1,2,3],coefficient),rtol=1e-9)
    np.testing.assert_allclose(g,coefficient,rtol=1e-9)


def test_posterior_variance_uses_parameter_discrimination_not_raw_amplitude(runner):
    # Equal concentrations provide no member information; separated log
    # concentrations reveal which endpoint is likely, under fixed noise.
    values=runner.posterior_variance([[1.,.1],[1.,10.]],[1.,4.],floor=0.,relative=.1)
    assert values[0]==pytest.approx(np.log(4.)**2/4.)
    assert 0<=values[1]<values[0]*1e-3


def test_interrupted_action_is_hash_committed_before_reveal(runner,monkeypatch):
    r=runner;monkeypatch.setattr(r,'METHODS',('time_farthest',))
    p,case,_,calls=setup(r,monkeypatch)
    real_reveal=r.sequential.reveal
    def interrupt(*args):
        journal=r.read(r.RUN/'cases/synthetic.progress.json')
        assert len(journal['decisions'])==1
        source=next(iter(journal['decisions'].values()))
        assert r.checked(source)['status']=='selected'
        raise RuntimeError('Synthetic interruption after commit before reveal')
    monkeypatch.setattr(r.sequential,'reveal',interrupt)
    with pytest.raises(RuntimeError,match='Synthetic interruption'):
        r.run_case(p,case)
    n=len(calls);assert n==2
    journal=r.read(r.RUN/'cases/synthetic.progress.json')
    committed=next(iter(journal['decisions'].values()))
    monkeypatch.setattr(r.sequential,'reveal',real_reveal)
    done=r.checked(r.run_case(p,case))
    assert done['sequences'][0]['actions'][0]==committed
    assert [s['budget'] for s in done['sequences'][0]['stages']]==[0,1,2,3]


def test_alternate_but_valid_action_after_interruption_is_rejected(runner,monkeypatch):
    r=runner;monkeypatch.setattr(r,'METHODS',('time_farthest',))
    p,case,_,calls=setup(r,monkeypatch)
    def interrupt(*args): raise RuntimeError('Synthetic interruption')
    monkeypatch.setattr(r.sequential,'reveal',interrupt)
    with pytest.raises(RuntimeError): r.run_case(p,case)
    journal=r.read(r.RUN/'cases/synthetic.progress.json');source=next(iter(journal['decisions'].values()))
    path=r.ROOT/source['path'];action=r.read(path)
    alternate=(action['selected_index']+1)%len(action['candidate_ids'])
    action.update(selected_index=alternate,selected_id=action['candidate_ids'][alternate],
        selected_time_h=action['candidate_times_h'][alternate]);r.save(path,action)
    n=len(calls)
    with pytest.raises(ValueError,match='source hash'): r.run_case(p,case)
    assert len(calls)==n


def report_fixture(r,monkeypatch,*,failed=False):
    """A real synthetic sequence manifest with a stub clinical scoring vault."""
    p,case,_,_=setup(r,monkeypatch,layouts=(0,1))
    case.update(drug='Synthetic',cluster_id='one independent synthetic study')
    p['limitations']=['Synthetic fixture'];r.save(r.RUN/'protocol.json',p)
    def choose(p,case,fitref,training,ids,menu,method,random_order,guard):
        fail=failed and method=='refit_goal' and len(ids)==5
        return dict(method=method,fit=fitref,visible_ids=sorted(ids),
            candidate_ids=menu['candidate_ids'],candidate_times_h=menu['candidate_times_h'],
            status='failed' if fail else 'selected',selected_index=None if fail else 0,
            selected_id=None if fail else menu['candidate_ids'][0],
            selected_time_h=None if fail else menu['candidate_times_h'][0],
            reason='Synthetic selection failure' if fail else None,
            hidden_values_used=False,target_values_used=False,locked_at=1.)
    monkeypatch.setattr(r,'make_decision',choose)
    done=r.run_case(p,case)
    state=dict(status='predictions_locked',protocol=r.ref(r.RUN/'protocol.json'),
        completed={'synthetic':done},started_at=0.,locked_at=r.read(r.ROOT/done['path'])['locked_at']+1.)
    r.save(r.RUN/'state.json',state)
    import report_blueprint_validation as scoring
    opened=[];scored=[]
    endpoints={name:dict(value=1.,unit='h' if name=='tmax' else 'synthetic') for name in
        ('auc','cmax','tmax','c_last','terminal_half_life','auc_inf','cl','vz','mrt','vss')}
    class Store:
        def __init__(self,root): pass
        def unchanged(self): pass
    def reference(case,store):
        opened.append(case['case_id'])
        return {},{'unit':'mg/L'},deepcopy(endpoints)
    def score(case,clinical,curve,eps,unit,store):
        scored.append(unit['identity']['visible_ids'])
        return dict(curve_metrics={'gmfe':1.5},endpoints={k:dict(value=1.2,reference=1.,unit=v['unit'],
            error=0. if k=='tmax' else 1.2,weak_tail=False) for k,v in eps.items()}),None,None
    def geomean(values,clusters):
        return dict(estimate=float(np.exp(np.mean(np.log(list(values.values()))))) if values else None,n_cases=len(values))
    def paired(a,b,clusters):
        pairs=[(a[k],b[k]) for k in a if a[k] is not None and b[k] is not None]
        return dict(estimate=float(np.exp(np.mean([np.log(x/y) for x,y in pairs]))) if pairs else None,
            complete_pairs=len(pairs),ratio_below_one=sum(x<y for x,y in pairs))
    monkeypatch.setattr(scoring,'Store',Store)
    monkeypatch.setattr(scoring,'_case_reference',reference)
    monkeypatch.setattr(scoring,'_score_prediction',score)
    monkeypatch.setattr(scoring,'clustered_geomean',geomean)
    monkeypatch.setattr(scoring,'paired_ratio',paired)
    return p,state,opened,scored


@pytest.mark.parametrize('status',['prepared','running','paused'])
def test_target_vault_stays_closed_until_global_prediction_lock(runner,monkeypatch,status):
    r=runner;p,state,opened,scored=report_fixture(r,monkeypatch)
    state['status']=status;r.save(r.RUN/'state.json',state)
    with pytest.raises(ValueError,match='lock'):
        r.report()
    assert opened==[] and scored==[]


@pytest.mark.parametrize('kind',['missing_case','protocol','late_lock','source_changed','missing_sequence','missing_budget'])
def test_incomplete_or_changed_global_manifest_rejected_before_target_read(runner,monkeypatch,kind):
    r=runner;p,state,opened,scored=report_fixture(r,monkeypatch)
    if kind=='missing_case': state['completed']={}
    elif kind=='source_changed': r.save(r.RUN/'full.json',{'changed':True})
    else:
        source=state['completed']['synthetic'];path=r.ROOT/source['path'];case=r.read(path)
        if kind=='protocol': case['protocol']={}
        elif kind=='late_lock': case['locked_at']=state['locked_at']+1
        elif kind=='missing_sequence': case['sequences'].pop()
        elif kind=='missing_budget': case['sequences'][0]['stages'].pop()
        r.save(path,case);state['completed']['synthetic']=r.ref(path)
    r.save(r.RUN/'state.json',state)
    with pytest.raises(ValueError): r.report()
    assert opened==[] and scored==[]


def test_report_scores_each_shared_fit_once_retains_all_endpoint_rows(runner,monkeypatch):
    import csv
    r=runner;p,state,opened,scored=report_fixture(r,monkeypatch)
    r.report()
    assert opened==['synthetic']
    nfits=len(list((r.RUN/'fits/synthetic').glob('*/result.json')))
    assert len(scored)==nfits
    rows=list(csv.DictReader((r.RUN/'endpoints.csv').open()))
    assert len(rows)==(2*7*4+1)*11
    assert {r['budget'] for r in rows}=={'0','1','2','3',''}
    result=r.read(r.RUN/'results.json')
    assert result['summary']['0']['time_farthest']['tmax']['mean']==0.
    assert result['summary']['mean_acquired']['random']['cmax']['n_cases']==1
    assert r.read(r.RUN/'state.json')['status']=='complete'


def test_report_preserves_failed_sequences_and_never_passes_complete_gate(runner,monkeypatch):
    import csv
    r=runner;p,state,opened,scored=report_fixture(r,monkeypatch,failed=True)
    r.report()
    rows=list(csv.DictReader((r.RUN/'endpoints.csv').open()))
    failures=[row for row in rows if row['method']=='refit_goal' and row['budget'] in ('2','3')]
    assert len(failures)==2*2*11 and all(row['status']=='selection_failed' for row in failures)
    assert all(row['prediction']=='' and row['error']=='' for row in failures)
    result=r.read(r.RUN/'results.json')
    assert not result['expansion_checks']['complete'] and not result['expand_to_twenty']
    assert result['summary']['mean_acquired']['refit_goal']['cmax']['n_cases']==0
