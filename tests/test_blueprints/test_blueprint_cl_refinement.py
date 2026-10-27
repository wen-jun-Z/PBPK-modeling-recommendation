"""CL-only interventions preserve independent parameters and scoring isolation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_blueprints.test_blueprint_value_runner import rich_request, record

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/run_blueprint_cl_refinement.py'
SPEC=importlib.util.spec_from_file_location('cl_refinement_test',SCRIPT)
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


def test_only_predicted_cl_changes_with_honest_provenance():
    base=rich_request();before=deepcopy(base)
    receipt={'path':'prior.json','sha256':'a'*64}
    result,changed=runner.replace_predicted_cl(base,1.7,receipt,'new-cl-1')
    assert changed and base==before
    assert record(result,'total_clearance')['value']==1.7
    assert record(result,'total_clearance')['prediction_method']['name']=='scaffold_selected_clearance_prior'
    assert record(result,'total_clearance')['source']['sha256']==receipt['sha256']
    assert [r for r in result['evidence'] if r['parameter']!='total_clearance']==[r for r in base['evidence'] if r['parameter']!='total_clearance']


def test_measured_cl_and_renal_split_are_protected():
    base=rich_request();record(base,'total_clearance')['basis']='measured'
    base['evidence'].append({'parameter':'renal_clearance','value':1.1,'basis':'measured'})
    result,changed=runner.replace_predicted_cl(base,20.,{},'new-cl-1')
    assert not changed and result==base
    record(base,'total_clearance')['basis']='predicted'
    with pytest.raises(ValueError,match='independent split'):
        runner.replace_predicted_cl(base,20.,{},'new-cl-1')


def test_unchanged_selected_prediction_reuses_original_input():
    base=rich_request()
    result,changed=runner.replace_predicted_cl(base,record(base,'total_clearance')['value'],{},'new-cl-1')
    assert not changed and result==base


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_invalid_predicted_cl_is_not_silently_clipped(value):
    with pytest.raises(ValueError,match='finite and positive'):
        runner.replace_predicted_cl(rich_request(),value,{},'new-cl-1')


def test_report_rejects_unlocked_state_before_reading_clinical_data(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'own',lambda p:Path(p))
    monkeypatch.setattr(runner.value.old,'checked',lambda r:{'tasks':{}})
    monkeypatch.setattr(runner.value.old,'verify',lambda *a,**k:None)
    monkeypatch.setattr(runner.value.evaluation,'clinical_truth',lambda *a:pytest.fail('Premature clinical access'))
    (tmp_path/'state.json').write_text(json.dumps({'protocol':{},'outcomes':{},'all_predictions_locked_before_scoring':False}))
    with pytest.raises(ValueError,match='Lock every prediction'):
        runner.report(tmp_path)
