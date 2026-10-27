"""Wrapper isolation, bounded paths and meaningful-benefit decision contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_blueprint_evidence_refinement.py'
SPEC=importlib.util.spec_from_file_location('evidence_refinement_test',SCRIPT)
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def test_cloned_functions_preserve_keyword_defaults_without_changing_globals():
    source=7
    def original(x=1,*,extra=2):return x+extra+source
    copied=r.clone(original)
    assert copied()==10 and original()==10
    assert copied.__kwdefaults__=={'extra':2} and copied.__globals__ is not original.__globals__


def test_paths_are_limited_to_two_declared_batch_subdirectories():
    assert r.own(r.BATCH/'conditional')==r.BATCH/'conditional'
    assert r.own(r.BATCH/'soft_half')==r.BATCH/'soft_half'
    for path in (r.BATCH,r.SOURCE,r.BATCH/'other',r.ROOT/'runtime'):
        with pytest.raises(ValueError):r.own(path)


def test_prior_checksum_uses_the_training_writers_serialization(tmp_path):
    prior={'audit':{key:False for key in ('target_labels_used','target_observation_files_opened',
           'training_structure_overlap','training_generic_scaffold_overlap')},
           'source_refs':{},'code_pins':{},'half_log_uncertainty_q80':.7}
    prior['result_sha256']=r.training_digest(prior)
    path=tmp_path/'prior.json';path.write_text(json.dumps(prior))
    loaded,_=r._checked_prior(path)
    assert loaded['half_log_uncertainty_q80']==.7
    prior['half_log_uncertainty_q80']=.8;path.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='checksum'):r._checked_prior(path)


def test_conditional_changes_sidecars_but_cannot_touch_request_or_measured_sidecar(monkeypatch):
    task={'request':{'cl':2},'sidecars':{'vss':[{'basis':'predicted','value':1}],
          'half':[{'basis':'measured','value':4}]},'core':{'design_version':'frozen'}}
    original=deepcopy(task)
    def policy(request,sidecars,artifact,receipt):
        sidecars=deepcopy(sidecars);sidecars['vss'][0]['value']=2
        return sidecars,{'applied':True}
    monkeypatch.setattr(r.importlib,'import_module',lambda name:SimpleNamespace(condition_sidecars=policy))
    changed,_=r.transformed_task(task,'conditional',{},{});assert task==original
    assert changed['request']==task['request'] and changed['core']==task['core']
    assert changed['sidecars']['half']==task['sidecars']['half']
    def bad(request,sidecars,artifact,receipt):request['cl']=3;return sidecars,{}
    monkeypatch.setattr(r.importlib,'import_module',lambda name:SimpleNamespace(condition_sidecars=bad))
    with pytest.raises(ValueError):r.transformed_task(task,'conditional',{}, {})


def test_soft_core_changes_without_combining_conditional_inputs(monkeypatch):
    module=SimpleNamespace(VERSION='soft-1',__file__=str(SCRIPT),simulate_uncertain_design=lambda **kw:kw)
    monkeypatch.setattr(r.importlib,'import_module',lambda name:module)
    task={'request':{'cl':2},'sidecars':{'vss':[1],'half':[2]},'core':{'design_version':'frozen','code_pins':{}}}
    changed,_=r.transformed_task(task,'soft_half',{'half_log_uncertainty_q80':.7},{'path':'prior','sha256':'x'})
    assert changed['request']==task['request'] and changed['sidecars']==task['sidecars']
    assert changed['core']['design_version']=='soft-1' and task['core']['design_version']=='frozen'


def test_meaningful_counts_separate_tiny_changes_failures_and_five_percent():
    before={'a':2,'b':2,'c':2,'d':None,'e':2,'f':None}
    after={'a':1.9,'b':1.999,'c':2.1,'d':2,'e':None,'f':None}
    c=r.comparison_counts(after,before,set(before))
    assert c['improved']==2 and c['meaningful_improved_5pct']==1
    assert c['worsened']==1 and c['meaningful_worsened_5pct']==1
    assert c['recovered']==1 and c['new_unscored']==1 and c['both_unscored']==1
    assert c['complete_pairs']==3 and c['requested']==6


def test_absolute_gate_can_pass_when_improved_available_makes_old_ratio_worse():
    ids=[str(i) for i in range(8)]
    def data(candidate=False):
        rows=[]
        for i,cid in enumerate(ids):
            available=1.8 if i<3 else 2.
            if candidate and i==3:available=1.8
            for con in r.CONDITIONS:
                rows.append({'case_id':cid,'condition':con,'curve_metrics':{'gmfe':available if con=='available' else 2.},
                             'changed_from_nominal':con=='missing_clearance' and i<6})
        return {'units':rows,'missingness':[{'scenario':'missing_clearance','prediction_drift':{'estimate':1.1},
            'masked_over_nominal_clinical_gmfe':{'estimate':1.2 if candidate else 1.1}}]}
    protocol={'cases':[{'case_id':cid} for cid in ids],'changed_case_ids':ids,'baseline_results':{},
              'publication_decision':{'expected_selected_meaningful_benefit_count':3}}
    result=r.refinement_statistics(data(True),data(False),protocol)
    assert result['publication_decision']['passed']
    assert not result['publication_decision']['majority_of_eight_benefit']
    assert not result['old_joint_gate_diagnostic']['passed']
    assert result['evidence_benefit_actual_eight']['candidate']['meaningful_improved_5pct']==4
