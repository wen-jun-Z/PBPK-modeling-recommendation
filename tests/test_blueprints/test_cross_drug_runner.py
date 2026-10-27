"""Cross-drug isolation and checkpoint lifecycle with synthetic, non-native work."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_cross_drug_blueprint as module
from test_robust_runner import batch as old_batch


@pytest.fixture
def batch(old_batch, monkeypatch):
    base = old_batch; counts = Counter(); prior_calls = []
    monkeypatch.setattr(module, 'assemble_evidence_blueprint', lambda *args: {'status':'assembled'})
    monkeypatch.setattr(module, 'summarize_extended_prediction', lambda *args: {'endpoints': {'auc': {'value': 1.}}})
    for name in module.CODE:
        path = base.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# test code\n')
    cases = []
    for index in range(20):
        cid = f'drug{index}'; case = deepcopy(base.cases[index % 10])
        case.update(case_id=cid, compound_id=cid, drug=cid, study_id=cid+'_target',
            cluster_id='shared_study' if index < 2 else cid+'_target',
            cohort='original' if index < 10 else 'extension', reference_kind='observed_curve')
        for family, refs in case['requests'].items():
            for phase, source in list(refs.items()):
                req = base.api.checked(source); req.update(compound_id=cid, compound_name=cid)
                for row in req['evidence']: row['compound_id'] = cid
                refs[phase] = base.document(base.base/f'expanded/{cid}/{family}_{phase}.json', req)
        full = base.api.read(base.base/f'drug{index % 10}/training.json')
        full.update(compound_id=cid, study_id=cid+'_development', role='calibration_data_reconstruction',
            matrix='plasma', time_unit='min', unit='umol/l')
        fullref = base.document(base.base/f'expanded/{cid}/full.json', full)
        case['sampling_sources'] = {'full': {**fullref, 'role':'calibration_data_reconstruction'}}
        cases.append(case)
    source = base.document(base.root/module.SOURCE, {'cases':cases})

    def physiology(request, output, root, check):
        check(); counts['native_export'] += 1
        source = base.document(output/'physiology/perfusion_physiology.json', {
            'compound': request['compound_id'], 'observations_used': False})
        return {**base.api.checked(source), 'manifest':source}

    def teacher_fit(**kwargs):
        request = kwargs['reference_request']; counts['teacher'] += 1
        assert request['partition_method'] == 'PKSimStandard'
        assert not any(e['parameter']=='renal_clearance' for e in request['evidence'])
        full = base.api.checked(kwargs['training_source'])
        assert len(full['observations']) == 6 and 'hidden_reference_metadata' not in full
        physiology(request, kwargs['output_dir'], kwargs['project_root'], kwargs['before_trial'])
        record = {'status':'fitted', 'eligible':True, 'decision':{'log_parameters':[.1,.2,.3]},
            'total_native_calls':1, 'linear_solver_calls':16, 'cost_count_exact':True, 'elapsed_seconds':.01}
        return {**record, 'mapping':base.document(kwargs['output_dir']/'mapping_locked.json', record)}

    def simulate(physiology, req, times, variant, keys, logs):
        counts['prediction'] += 1
        return {'concentration_mg_l': (1-np.exp(-10*np.asarray(times)))*np.exp(-.1*np.asarray(times)),
            'mass_balance':{'synthetic':True}, 'observation':'peripheral_venous',
            'effective_total_clearance_l_h': 1.}

    teacher = SimpleNamespace(VERSION='test-teacher', CONFIG={'starts':16},
        PARAMETERS={'tail_balanced':('clearance_scale','partition_scale','bidirectional_exchange_scale')},
        _code_pins=lambda root:{'engine.py':module.sha256(root/'engine.py')},
        clearance_policy=lambda request:{'rule':'qualified_rf'}, fit_mapping=teacher_fit,
        _validated=lambda *args:None, _physiology=physiology, _simulate=simulate)

    def fit_prior(rows, *, excluded_compound_ids, excluded_study_ids):
        counts['prior'] += 1
        assert not set(excluded_compound_ids).intersection(r['compound_id'] for r in rows)
        assert all(not set(excluded_study_ids).intersection(r['study_ids']) for r in rows)
        prior_calls.append((deepcopy(rows), list(excluded_compound_ids), list(excluded_study_ids)))
        return {'excluded_compound_ids':list(excluded_compound_ids), 'excluded_study_ids':list(excluded_study_ids),
            'training_compound_ids':[r['compound_id'] for r in rows]}

    def prior_prediction(bundle, request, method):
        assert request['compound_id'] in bundle['excluded_compound_ids']
        assert request['compound_id'] not in bundle['training_compound_ids']
        return np.array([{'rf_only':0.,'pooled_mean':.1,'ridge':.2}[method],0.,0.])

    def five(phys, request, doc, center, before_trial=None):
        before_trial(); counts['five'] += 1
        assert len(doc['observations']) == 5 and doc['compound_id'] == request['compound_id']
        assert not {'raw_observations','full_curve','selection','hidden_reference_metadata'}.intersection(doc)
        return {'eligible':True, 'status':'fitted', 'log_parameters':center.tolist(),
            'parameters':{}, 'linear_solver_calls':16, 'failed_evaluation_count':0,
            'starts':[{'index':i} for i in range(16)]}

    learner = SimpleNamespace(VERSION='test-prior', CONFIG={'methods':list(module.METHODS)},
        fit_prior=fit_prior, predict_prior=prior_prediction, fit_five=five)

    def score(run, project_root):
        runner = base.runner; p,s = runner.load(run,locked=True)
        assert s['all_predictions_locked_before_scoring'] and s['monitor_stopped_before_scoring']
        assert len(s['prediction_bundles']) == 20
        runner.checked(base.outcome)  # The sentinel can be opened only here.
        counts['score'] += 1
        result = {'units':p['units'], 'source_pins':{}, 'paper_updated':False}
        module.save(run/'results.json', result)
        return result

    runner = module.create_runner(project_root=base.root, teacher_engine=teacher, learner=learner, scorer=score)
    base.runner, base.run, base.counts, base.prior_calls = runner, runner.DEFAULT, counts, prior_calls
    return base


def test_study_transitive_closure_precedes_any_parameter_loading():
    rows = [dict(case_id='a',compound_id='A',all_study_ids=['s1','s2']),
        dict(case_id='b',compound_id='B',all_study_ids=['s2','s3']),
        dict(case_id='c',compound_id='C',all_study_ids=['s3','s4']),
        dict(case_id='d',compound_id='D',all_study_ids=['s5'])]
    folds = module.study_folds(rows)
    assert len(folds)==2 and folds[0]['heldout_case_ids']==['a','b','c']
    assert folds[0]['excluded_study_ids']==['s1','s2','s3','s4']
    assert folds[0]['training_case_ids']==['d']


def test_prepare_removes_clinical_renal_and_selects_five_without_target_decode(batch,monkeypatch):
    r=batch.runner; read=r.read; opened=[]
    def guarded(path):
        assert (batch.root/path).resolve()!=batch.root/batch.outcome['path']
        opened.append(str(path));return read(path)
    monkeypatch.setattr(r,'read',guarded)
    result=r.prepare(batch.run,workers=1)
    assert result['drugs']==20 and result['folds']==19 and result['units']==120
    p,s=r.load(batch.run)
    assert len(p['removed_renal_case_ids'])==4
    assert len([f for f in p['folds'] if len(f['heldout_case_ids'])==2])==1
    original=r.read(r.source)
    for sourcecase in original['cases']:
        for originalref in sourcecase['requests']['rf'].values():
            assert p['source_pins'][originalref['path']]==originalref['sha256']
    for case in p['cases']:
        assert 'requests' not in case
        request=r.checked(case['cross_drug_requests'])['development']
        rows={e['parameter']:e for e in request['evidence']}
        assert 'renal_clearance' not in rows
        assert rows['elimination_scope']['basis']=='assumption'
        assert rows['elimination_scope']['source']['path'].startswith(str(batch.run.relative_to(batch.root)))
        assert rows['fraction_unbound']['basis']==rows['total_clearance']['basis']=='predicted'
        sparse=r.checked(case['five_training_source'])
        assert len(sparse['observations'])==5 and 'selection' not in sparse
        assert case['five_selection']['concentrations_used_for_selection'] is False
    assert batch.counts=={}


def test_complete_study_blocked_learning_lifecycle_and_idempotent_resume(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.run,workers=1);read=r.read
    def guarded(path):
        actual=(batch.root/path).resolve()
        state=json.loads((batch.run/'state.json').read_text())
        if actual==batch.root/batch.outcome['path']:
            assert state['all_predictions_locked_before_scoring']
        if state['status']=='predicting':
            assert '/teachers/' not in str(actual) or actual.name=='perfusion_physiology.json'
            assert actual.name!='full.json'
        return read(path)
    monkeypatch.setattr(r,'read',guarded)
    result=r.execute(batch.run)
    assert result=={'status':'complete','units':120}
    assert batch.counts=={'teacher':20,'native_export':40,'prior':19,'five':60,'prediction':120,'score':1}
    assert sorted(len(rows) for rows,_,_ in batch.prior_calls)==[18]+[19]*18
    p,s=r.load(batch.run,locked=True)
    for source in s['prediction_bundles'].values():
        bundle=r.checked(source)
        assert len(bundle['entries'])==6 and bundle['status']=='locked'
        for entry in bundle['entries'].values():
            assert entry['same_drug_full_curve_used'] is False
            assert entry['same_drug_teacher_mapping_decoded'] is False
            if entry['condition']=='zero':assert entry['fit'] is None
    before=dict(batch.counts);r.execute(batch.run);r.report(batch.run)
    assert dict(batch.counts)==before
    assert r.cleanup(batch.run)['removed_count']==0


@pytest.mark.parametrize('mutation',['source','paper'])
def test_changed_frozen_input_or_paper_refuses_execution(batch,mutation):
    r=batch.runner;r.prepare(batch.run,workers=1);p,s=r.load(batch.run)
    path=batch.paper if mutation=='paper' else batch.root/p['cases'][0]['five_training_source']['path']
    path.write_text('changed')
    with pytest.raises(ValueError,match='changed'):r.load(batch.run)
    assert batch.counts=={}


def test_incomplete_batch_cannot_score_and_own_run_isolation(batch):
    r=batch.runner;r.prepare(batch.run,workers=1)
    with pytest.raises(ValueError,match='lock'):r.report(batch.run)
    assert batch.counts['score']==0
    for path in [batch.base,batch.root/'cross_drug_wrong_parent',batch.run/'cross_drug_nested']:
        with pytest.raises(ValueError,match='dedicated'):r.own_run(path)


def test_pause_request_prevents_any_new_teacher(batch):
    r=batch.runner;r.prepare(batch.run,workers=1)
    (batch.run/'PAUSE').write_text('pause')
    with pytest.raises(module.Paused):r.execute(batch.run)
    assert batch.counts=={}
    assert r.read(batch.run/'state.json')['status']=='paused'


def test_interrupted_prediction_entry_has_immutable_resume_receipt(batch,monkeypatch):
    r=batch.runner;r.prepare(batch.run,workers=1)
    predict=r.learner.predict_prior;calls=0
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('synthetic interruption after one sealed prediction')
        return predict(*args,**kwargs)
    monkeypatch.setattr(r.learner,'predict_prior',interrupted)
    with pytest.raises(module.Paused):r.execute(batch.run)
    state=r.read(batch.run/'state.json')
    assert state['status']=='paused' and len(state['progress'])==1
    cid=next(iter(state['progress']));progress=state['progress'][cid]
    assert progress['entries_locked']==1
    path=batch.run/'predictions'/(cid+'.json');bundle=r.read(path)
    key=next(iter(progress['entry_digests']))
    assert module.object_digest(bundle['entries'][key])==progress['entry_digests'][key]
    bundle['entries'][key]['curve']['concentration'][1]+=1.
    module.save(path,bundle)
    monkeypatch.setattr(r.learner,'predict_prior',predict)
    before=batch.counts['prediction']
    with pytest.raises(module.Paused,match='Sealed partial prediction entry changed'):
        r.execute(batch.run)
    assert batch.counts['prediction']==before and batch.counts['score']==0
