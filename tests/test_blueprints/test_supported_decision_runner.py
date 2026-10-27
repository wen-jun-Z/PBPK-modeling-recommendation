from copy import deepcopy
import json
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_supported_decision_value as mod


@pytest.fixture
def capture(monkeypatch):
    seen = []
    def decide(request, candidates, references, *, primary_endpoint, method):
        seen.append({'references': references, 'method': method, 'primary': primary_endpoint})
        return {'method': method, 'reasons': []}
    monkeypatch.setattr(mod.base, 'decide_value', decide)
    return seen


@pytest.mark.parametrize('family,expected', [('global', [1., 2., 3.]), ('local', [9., 10.])])
def test_calibration_family_controls_reference_not_target(capture, family, expected):
    refs = {'mrt': {'local_log_reference_values': [9., 10.], 'global_log_reference_values': [1., 2., 3.]}}
    old = deepcopy(refs)
    policy = {'endpoints': {'mrt': {'family': family}}}
    result = mod.supported_decision(policy, {}, {}, refs, primary_endpoint='mrt', method='purpose_decision')
    assert capture[0]['references']['mrt']['local_log_reference_values'] == expected
    assert result['method'] == 'purpose_decision'
    assert refs == old


@pytest.mark.parametrize('method,effective', [('purpose_decision', 'local_risk_only'), ('global_decision', 'fixed_capacity')])
def test_equal_clearance_has_no_structural_choice(capture, method, effective):
    pool = {'parameter_only': {'effective_total_clearance_l_h': 5.}, 'combined': {'effective_total_clearance_l_h': 5.}}
    result = mod.supported_decision({'endpoints': {}}, {}, pool, {}, primary_endpoint='clearance', method=method)
    assert capture[0]['method'] == effective
    assert result['method'] == method
    assert 'shared_clearance_input_no_structural_CL_choice' in result['reasons']


def test_different_clearance_preserves_selection(capture):
    pool = {'parameter_only': {'effective_total_clearance_l_h': 5.}, 'combined': {'effective_total_clearance_l_h': 10.}}
    mod.supported_decision({'endpoints': {}}, {}, pool, {}, primary_endpoint='clearance', method='purpose_decision')
    assert capture[0]['method'] == 'purpose_decision'


def test_fixed_control_reference_unchanged(capture):
    refs = {'mrt': {'local_log_reference_values': [9.], 'global_log_reference_values': [1.]}}
    mod.supported_decision({'endpoints': {'mrt': {'family': 'global'}}}, {}, {}, refs, primary_endpoint='mrt', method='fixed_native')
    assert capture[0]['method'] == 'fixed_native'
    assert capture[0]['references'] == refs


def test_unsupported_reference_stays_unassessed_in_underlying_policy(capture):
    mod.supported_decision({'endpoints': {}}, {}, {}, {}, primary_endpoint='auc', method='purpose_decision')
    assert capture[0]['references'] == {}


def test_old_runner_cannot_execute_supported_protocol(tmp_path, monkeypatch):
    run = tmp_path / 'experiments/blueprint_design/decision_value_supported_test'
    run.mkdir(parents=True)
    runner = mod.Runner(tmp_path)
    protocol = {'version': mod.VERSION, 'source_pins': {}, 'code_pins': {}, 'paper_guard': {}}
    (run / 'protocol.json').write_text(json.dumps(protocol))
    (run / 'state.json').write_text(json.dumps({'protocol': runner.ref(run / 'protocol.json')}))
    monkeypatch.setattr(mod.base, 'RefinementRunner', lambda root: SimpleNamespace(paper_pins=lambda: {}))
    assert runner.load(run)[0]['version'] == mod.VERSION
    with pytest.raises(ValueError, match='version mismatch'):
        mod.base.Runner(tmp_path).load(run)
