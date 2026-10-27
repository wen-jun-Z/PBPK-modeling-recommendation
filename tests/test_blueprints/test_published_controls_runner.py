"""Frozen predictor identity and information isolation for published controls."""
import inspect
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_published_blueprint_controls as runner


@pytest.fixture(scope='module')
def source():
    comparator=runner.read(runner.frozen.OUT/'protocol.json')
    protocol=runner.read(runner.current.RUN/'protocol.json')
    state=runner.read(runner.current.RUN/'state.json')
    native=runner.checked(protocol['source_protocol'])
    choices=runner.checked(protocol['choices'])['choices']
    return comparator,state,native,choices


@pytest.mark.parametrize('case_index',range(30))
def test_exact_current_predictor_for_every_case(case_index,source):
    comparator,state,native,choices=source
    cid,case=list(comparator['cases'].items())[case_index]
    cell=next(c for c in comparator['cells'] if c['case_id']==cid and c['info']=='time_five')
    adapter=runner.build_adapter(case['source_case'],cell['training_source'],native['physiology_cache'])
    selection=choices[cid+'__late_four']['cmax']['sources']['time_repaired']
    audit=runner.verify_adapter(case['source_case'],adapter,state['units'][selection['unit_id']]['result'])
    assert audit['relative_development_objective_difference']<1e-9
    assert not audit['old_parameters_used_for_initialization']
    for inp in adapter['formulas'].values():
        assert len(inp['times_h'])==5
        # All four initial multipliers are nominal evidence values: no fitted warm start.
        assert np.array_equal(inp['initial'],np.zeros(4))
        assert not inp['feasibility']['target_observations_used']
        assert not inp['feasibility']['hidden_development_observations_used']


def test_target_mutation_cannot_change_development_closure(source):
    comparator,state,native,choices=source
    cid,case=next(iter(comparator['cases'].items()))
    cell=next(c for c in comparator['cells'] if c['case_id']==cid and c['info']=='time_five')
    inp=runner.build_adapter(case['source_case'],cell['training_source'],native['physiology_cache'])['formulas']['PKSimStandard']
    mutated=deepcopy(inp)
    mutated['target_request']={'poison':'must not be read'}
    mutated['target_physiology']={'poison':'must not be read'}
    x=np.array([.2,-.2,.7,-.4])
    np.testing.assert_array_equal(runner.predictor(inp)(x),runner.predictor(mutated)(x))
    captured=inspect.getclosurevars(runner.predictor(inp)).nonlocals
    assert set(captured)=={'phys','req','times'}


def test_formula_objective_retains_exact_frozen_regularization():
    x=np.array([7.,.3,-.4,.2]);pred=np.array([1.,2.,3.,4.,5.]);obs=pred+.2
    expected=np.mean((pred-obs)**2)+(.1*x[1]/np.log(10))**2+(.1*x[2]/np.log(100))**2+(.1*x[3]/np.log(10))**2
    assert runner.development_objective(x,pred,obs)==pytest.approx(expected,abs=1e-15)


def test_report_rejects_partial_before_importing_any_target_scoring(monkeypatch):
    state=dict(status='partial',protocol={'path':'unused'},completed={})
    monkeypatch.setattr(runner,'read',lambda _:state)
    monkeypatch.setattr(runner,'checked',lambda _:dict(jobs={'not_done':{}},source_pins={}))
    monkeypatch.setattr(runner,'guard',lambda _:None)
    with pytest.raises(ValueError,match='must lock'):
        runner.report()
