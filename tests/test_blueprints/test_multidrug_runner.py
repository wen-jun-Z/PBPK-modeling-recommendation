"""Prediction/observation boundary checks; never runs a native solver."""
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location('multidrug_cli', _ROOT/'scripts/run_multidrug_blueprint_evaluation.py')
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    record = {'scenario_id':'drug_iv','compound_id':'drug','study_group_id':'study_one',
              'observations':[{'time_min':1.,'concentration_umol_per_l':1.}]}
    path=tmp_path/'observations.json'
    path.write_text(json.dumps(record))
    return {'case_id':'drug_case','compound_id':'drug','tier':'bottom_up_external',
            'parameter_search':{'parameters':[]},'scenarios':[
                {'scenario_id':'drug_iv','study_group_id':'study_one','target_output':'Exact|Plasma',
                 'holdout_reference':{'path':'observations.json','sha256':runner.file_sha(path)}}]}


def test_preflight_only_hashes_outcomes_without_decoding(case, monkeypatch):
    monkeypatch.setattr(runner.json, 'loads', lambda _: pytest.fail('Designer must not decode external outcomes'))
    runner.validate_case(case)


def test_outcomes_cannot_be_embedded_in_prediction_inputs(case):
    case['scenarios'][0]['observations']=[{'concentration_umol_per_l':10.}]
    with pytest.raises(ValueError, match='stored separately'):
        runner.validate_case(case)


def test_changed_observations_are_rejected_before_evaluation(case):
    runner.validate_case(case)
    path=runner.ROOT/'observations.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        runner.load_observation(case['scenarios'][0],case)


def test_fixed_cohort_cannot_silently_fit_parameters(case):
    case['parameter_search']['parameters']=[{'parameter_id':'unknown'}]
    with pytest.raises(ValueError, match='does not silently fit'):
        runner.validate_case(case)


def test_observation_drug_identity_must_match(case):
    case['compound_id']='another_drug'
    with pytest.raises(ValueError, match='drug identity'):
        runner.load_observation(case['scenarios'][0],case)


@pytest.mark.parametrize('unit',['µmol/l','μmol/l','umol/l'])
def test_native_curve_reads_requested_observer_only(tmp_path,unit):
    (tmp_path/'curve.csv').write_text(f'Time [min],Exact|Plasma [{unit}],Other|Plasma [{unit}]\n0,0,999\n1,2,999\n')
    result=runner.read_curve(tmp_path,{'curve_outputs':[{'path':'curve.csv','sha256':runner.file_sha(tmp_path/'curve.csv')}]},'Exact|Plasma')
    assert result['concentration']==[0.,2.]
    assert result['unit']=='umol/l'


def test_mass_concentration_is_not_silently_relabelled_molar(tmp_path):
    (tmp_path/'curve.csv').write_text('Time [min],Exact|Plasma [mg/l]\n0,0\n1,2\n')
    with pytest.raises(ValueError,match='column or unit'):
        runner.read_curve(tmp_path,{'curve_outputs':[{'path':'curve.csv','sha256':runner.file_sha(tmp_path/'curve.csv')}]},'Exact|Plasma')
