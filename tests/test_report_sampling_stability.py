"""Independent report arithmetic and completion boundary checks."""
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope='module')
def report():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import report_sampling_stability
    return report_sampling_stability


def test_no_outcomes_opened_before_completion(report, monkeypatch):
    seen = []
    def read(path):
        seen.append(path)
        assert path == report.RUN/'state.json'
        return dict(status='predictions_locked', results='do-not-read')
    monkeypatch.setattr(report, 'read', read)
    monkeypatch.setattr(report, 'checked', lambda *args: pytest.fail('Uncompleted result opened'))
    with pytest.raises(ValueError, match='incomplete'):
        report.load_complete()
    assert seen == [report.RUN/'state.json']


def test_independent_ridge_has_local_scaling_and_fixed_unit_penalty(report):
    rows = [dict(x=[float(i)]+[0.]*7,y=2.*i,drug='a',design='a1',cluster='s1') for i in range(3)]
    rows += [dict(x=[1e20]*8,y=1e20,drug='excluded',design='b1',cluster='s2')]
    scale, coef, membership = report.independent_fit(rows,[0,1,2])
    assert scale[0] == pytest.approx(np.sqrt(2/3))
    assert coef[0] == pytest.approx(scale[0])
    np.testing.assert_allclose(coef[1:],0,atol=1e-12)
    assert membership['drug_ids'] == ['a']
    assert membership['cluster_ids'] == ['s1']


def test_independent_label_rebuild_ignores_added_outcomes_and_reported_scalars(report):
    features = {'a__late': dict(case_id='a', candidate_ids=[2], goals={'cmax': dict(status='selected',X=[[0.]*8])}),
                'b__late': dict(case_id='b', goals=None)}
    records = [dict(case_id='a',design_key='a__late',candidate='baseline',endpoints={'cmax':{'error':4.}},source_reported_sensitivity={'cmax':{'error':1.}}),
               dict(case_id='a',design_key='a__late',candidate='a__late__point_02',endpoints={'cmax':{'error':2.}},source_reported_sensitivity={'cmax':{'error':100.}}),
               dict(case_id='b',endpoints=None)]
    rows=report.training_rows(features,records,{'a'},'cmax',{'a':{'cluster_id':'s1'}})
    assert len(rows)==1 and rows[0]['y']==pytest.approx(np.log(2.))


@pytest.mark.parametrize('endpoint,prediction,reference,wrong_error', [('cmax',4.,2.,.5),('tmax',4.,2.,1.)])
def test_scalar_error_recomputed_from_predictions_and_references(report,endpoint,prediction,reference,wrong_error):
    protocol=dict(cases=[dict(case_id='a',drug='A',cluster_id='s1')],original_case_ids=['a'])
    key='a__late_four';selection=dict(candidate='baseline',unit_id='unit',time_h=None)
    lock={'choices':{key:{'cmax':{'sources':{'stable':selection}}}}}
    record=dict(design_key=key,candidate='baseline',endpoints={endpoint:dict(value=prediction,reference=reference,error=wrong_error)})
    row=dict(case_id='a',drug='A',cluster_id='s1',cohort='original_ten',regime='late_four',requested_goal='cmax',
             method='stable',endpoint=endpoint,prediction=str(prediction),reference=str(reference),error=str(wrong_error),selected_time_h='')
    with pytest.raises(AssertionError,match='Independent scalar error'):
        report.audit_csv({},protocol,{'repaired_scores':{}},[row],lock,[record])
