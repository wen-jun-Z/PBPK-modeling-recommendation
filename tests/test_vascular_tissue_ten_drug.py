"""Scientific isolation and condition guards for the frozen-method expansion."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_vascular_tissue_ten_drug as runner


def test_visible_training_excludes_full_curve_estimates_but_keeps_exact_points():
    document = {'observations': [{'time_min': 1, 'concentration_umol_per_l': 2}],
        'study_id': 'development', 'raw_observations': [99], 'full_curve': [98],
        'selection': {'original_indices': [2]}, 'fit': {'clearance': 97}, 'endpoints': {'auc': 96}}
    visible = runner.visible_training(document)
    assert visible == {'observations': document['observations'], 'study_id': 'development'}
    visible['observations'][0]['concentration_umol_per_l'] = 7
    assert document['observations'][0]['concentration_umol_per_l'] == 2


def test_original_ten_subsets_and_input_conditions_are_preserved():
    protocol = runner.api.read(runner.BASE / 'protocol.json')
    for case in protocol['cases']:
        full = runner.api.checked(case['sampling_sources']['full'])
        five = runner.api.checked(case['sampling_sources']['five'])
        assert len(five['observations']) == 5
        assert all(row in full['observations'] for row in five['observations'])
        request = runner.api.checked(case['requests']['evidence']['development'])
        for raw in (full, five):
            visible = runner.visible_training(raw)
            assert visible['observations'] == raw['observations']
            assert visible['human'] == request['human']
            assert visible['dosing'] == request['dosing']
            assert not {'selection', 'fit', 'full_curve', 'raw_observations'} & visible.keys()


@pytest.mark.parametrize('path', [runner.ROOT.parent / 'other', runner.BASE,
    runner.DEFAULT / 'nested', runner.DEFAULT.parent / 'clinical_something'])
def test_output_boundary(path):
    with pytest.raises(ValueError):
        runner.own_run(path)


def test_adapter_does_not_modify_original_default():
    original = runner.scheduler.DEFAULT_RUNNER
    assert original.DEFAULT.name == 'clinical_transport_v1'
    assert original.CONFIG['version'] == 'blueprint-transport-pilot-1'
    assert runner.api is not original
    assert runner.api.report is runner.report
    assert original.report is not runner.report


def test_report_rejects_unlocked_targets_before_outcomes():
    protocol = {'version': runner.VERSION, 'runner_config': runner.CONFIG,
        'methods': list(runner.METHODS), 'sampling': list(runner.SAMPLING),
        'formulas': list(runner.FORMULAS), 'cases': [{'case_id': str(i)} for i in range(10)],
        'source_protocol': {}, 'units': [], 'computations': {}}
    state = {'version': runner.VERSION, 'fits': {}, 'computations': {}, 'status': 'running'}
    with patch.object(runner.api, 'load_batch', return_value=(protocol, state)), \
            patch.object(runner.api, 'checked', return_value={'cases': protocol['cases']}), \
            patch('report_blueprint_validation._case_reference') as clinical:
        with pytest.raises(ValueError):
            runner.report(runner.DEFAULT)
        clinical.assert_not_called()


def test_hypothetical_requests_are_nine_unobserved_conditions():
    protocol = runner.api.read(runner.INPUT_RESPONSE)
    assert len(protocol['cases']) == 9
    assert {(c['dose_multiplier'], c['condition']) for c in protocol['cases']} == {
        (dose, condition) for dose in (.5, 1, 2) for condition in ('C1', 'C2', 'C3')}
    assert all(c['request']['compound_name'] == 'Treprostinil' for c in protocol['cases'])
