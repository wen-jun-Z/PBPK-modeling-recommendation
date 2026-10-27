"""Scientific input isolation of the five-capability experiment adapter."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_blueprint_capability as runner


@pytest.fixture
def example():
    protocol = runner.read(runner.BASE / 'protocol.json')
    case = next(c for c in protocol['cases'] if c['drug'] == 'Moxifloxacin')
    return (case, runner.checked(case['requests']['evidence']['development']),
        runner.checked(case['requests']['evidence']['target']),
        runner.checked(case['requests']['rf']['development']),
        runner.checked(case['sampling_sources']['full']))


def bind(rows):
    return [{**r, 'source': {'path': 'inputs/anonymous.json', 'sha256': 'a' * 64, 'location': 'Visible record'}} for r in rows]


def test_missing_fu_does_not_leak_truth_into_selected_request(example):
    case, base, target, rf, _ = example
    rows, fallback, audit = runner.policy_inputs.scenario_evidence(base, rf, 'missing_fu', case['case_id'])
    assert all(r['parameter'] != 'fraction_unbound' for r in rows)
    assert all('source' not in r for r in rows + fallback)
    request = deepcopy(base); request['evidence'] = bind(rows)
    outcome = runner.flow_decision(request, bind(fallback))
    assert outcome['status'] == 'ready'
    selected_fu = next(e for e in outcome['selected_evidence'] if e['parameter'] == 'fraction_unbound')
    assert selected_fu['value'] == next(e['value'] for e in rf['evidence'] if e['parameter'] == 'fraction_unbound')
    captured = []
    def store(folder, value):
        captured.append(deepcopy(value))
        return {'path': 'inputs/selected.json', 'sha256': 'b' * 64, 'location': 'Selected values'}
    with patch.object(runner.policy_inputs, 'store_input', side_effect=store):
        selected = runner.policy_inputs.selected_request(runner.DEFAULT, base, outcome['selected_evidence'])
    assert all('primary_sources' not in d and 'perturbation' not in d for d in captured)
    assert selected['evidence_ledger']['path'] == 'inputs/selected.json'
    assert selected['evidence_ledger'] != base['evidence_ledger']
    assert 'source_policy_calibration_metadata' not in selected


def test_equal_priority_conflict_blocks_main_and_exposes_unsafe_order(example):
    case, base, target, rf, _ = example
    rows, fallback, _ = runner.policy_inputs.scenario_evidence(base, rf, 'conflicting_fu', case['case_id'])
    request = deepcopy(base); request['evidence'] = bind(rows); fallback = bind(fallback)
    reverse = deepcopy(request); reverse['evidence'].reverse()
    a, b = runner.flow_decision(request, fallback), runner.flow_decision(reverse, fallback)
    assert a['status'] == b['status'] == 'blocked'
    assert runner.selected_values(a) == runner.selected_values(b)
    a = runner.flow_decision(request, fallback, method='no_conflict_guard')
    b = runner.flow_decision(reverse, fallback, method='no_conflict_guard')
    assert a['status'] == b['status'] == 'ready'
    assert runner.selected_values(a) != runner.selected_values(b)
    assert len([r for r in a['selected_evidence'] if r['parameter'] == 'fraction_unbound']) == 1


def test_adapter_declares_actual_axes_without_modifying_source_policy(example):
    _, base, _, rf, _ = example
    fallback = [e for e in rf['evidence'] if e['parameter'] == 'fraction_unbound']
    outcome = runner.flow_decision(base, fallback)
    assert outcome['calibration']['active_parameters'] == list(runner.AXES)
    assert outcome['source_policy_calibration_metadata']['allowed_parameters'] == ['lipophilicity', 'clearance_scale']
    assert outcome['calibration']['protect_net_renal']


def test_exact_alias_ignores_source_spelling_but_not_inputs(example):
    _, base, target, _, train = example
    original = runner.signature(base, target, train, 'always_three')
    variant = deepcopy(base); variant['evidence'].reverse()
    for row in variant['evidence']:
        row['source']['path'] = 'different-visible-path.json'
    assert runner.signature(variant, target, train, 'always_three') == original
    next(r for r in variant['evidence'] if r['parameter'] == 'fraction_unbound')['value'] *= .5
    assert runner.signature(variant, target, train, 'always_three') != original
    assert runner.signature(base, target, train, 'adaptive_axes') != original
    changed = deepcopy(target); changed['dosing']['dose_mg'] *= 2
    assert runner.signature(base, changed, train, 'always_three') != original


def test_five_points_do_not_alias_full_development(example):
    case, base, target, _, train = example
    sparse = runner.checked(case['sampling_sources']['five'])
    assert len(sparse['observations']) == 5
    assert runner.signature(base, target, train, 'always_three') != runner.signature(base, target, sparse, 'always_three')


@pytest.mark.parametrize('path', [runner.BASE, runner.DEFAULT / 'nested', runner.ROOT.parent / 'outside'])
def test_output_scope_guard(path):
    with pytest.raises(ValueError): runner.own_run(path)


def test_applicability_checks_are_executed_for_both_shared_gate_methods(example):
    case, *_ = example
    rows = runner.applicability({'cases': [case]}, {})
    assert len(rows) == 16
    assert sum(r['expected_supported'] for r in rows) == 11
    assert all(r['supported'] == r['expected_supported'] for r in rows)
    assert all(r['fixed_gate_baseline_supported'] == r['expected_supported'] for r in rows)
    assert all(r['reasons'] and r['fixed_gate_baseline_reasons'] for r in rows if not r['expected_supported'])


def test_report_cannot_decode_outcomes_before_complete_lock():
    with patch.object(runner, 'load', side_effect=ValueError('not locked')), \
            patch('report_blueprint_validation._case_reference') as clinical:
        with pytest.raises(ValueError): runner.report(runner.DEFAULT)
        clinical.assert_not_called()
