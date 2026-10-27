from copy import deepcopy
import json
from pathlib import Path
import pytest

from pbpk_kb.blueprints import experiment_runtime as runtime
from pbpk_kb.blueprints.design_diagnostics import BudgetLedger
from pbpk_kb.blueprints.qwen_proposals import build_request, validate_request

ROOT = Path(__file__).resolve().parents[2]


def load_case(drug='alprazolam'):
    return json.loads((ROOT/'experiments/blueprint_design/development_inputs/mechanisms_20260920'/drug/'input_package.json').read_text())['case']


def test_native_space_is_two_independent_structural_choices_and_common_coordinates():
    case = load_case()
    candidates = runtime.candidate_space(case, root=ROOT)
    assert len(candidates) == 16
    assert len({c['structure_semantic_hash'] for c in candidates}) == 4
    assert len({tuple(c['features'][:2]) for c in candidates}) == 4
    assert all(c['guidance_is_informative_in_this_case'] is False for c in candidates)
    bad = deepcopy(case)
    bad['mechanism_search']['expected_structure_ids'].pop()
    with pytest.raises(ValueError, match='preregistered'):
        runtime.candidate_space(bad, root=ROOT)


def test_parameter_mapping_changes_must_not_change_optimizer_units_bounds_or_order():
    specs = load_case()['parameter_search']['parameters']
    changed = deepcopy(specs)
    changed[0]['runtime_mappings'][0]['runtime_path'] = 'new native path'
    assert runtime.validate_candidate_parameter_coordinates(specs, changed) is changed
    changed[0]['unit'] = 'l/µmol/min'
    with pytest.raises(ValueError, match='optimization coordinates'):
        runtime.validate_candidate_parameter_coordinates(specs, changed)


def test_predict_uses_actual_candidate_paths_and_converted_values(tmp_path, monkeypatch):
    case = load_case()
    candidate = next(c for c in runtime.candidate_space(case, root=ROOT)
                     if 'mm_low_concentration' in c['action'] and c['candidate_id'].endswith('__11'))
    from pbpk_kb.blueprints.mechanism_edits import compile_native_candidate
    variant = compile_native_candidate(case['scenarios'][0]['design_request'], candidate['action'],
        project_root=ROOT, parameter_specs=case['parameter_search']['parameters'])
    experiment = runtime.NativeExperiment(case=case, output=tmp_path/'output', project_root=tmp_path,
        cpu_threads=1, scenario_workers=1, device='cpu')
    monkeypatch.setattr(experiment, '_model', lambda *args: {'pkml':tmp_path/'model.pkml','sha256':'a'*64,'variant':variant})
    seen = {}
    def batch(**kwargs):
        seen.update(kwargs)
        return {'status':'completed','native_results_sha256':'b'*64,
                'runs':[{'time_min':[0,1],'concentration':[0,1],'unit':'µmol/l','time_unit':'min'}]}
    monkeypatch.setattr(runtime, 'run_parameter_batch', batch)
    point = [1.25, case['parameter_search']['parameters'][1]['initial_value']]
    experiment._predict(candidate, case['scenarios'][0], [point],
        {'ledger':BudgetLedger(100, diagnostic_fraction=0),'model_requests':set()}, 'search')
    experiment.close()
    assert all(x['path'].endswith('|CLspec/[Enzyme]') for x in seen['parameter_specs'][:2])
    assert all(x['unit']=='l/µmol/min' for x in seen['parameter_specs'][:2])
    assert seen['values'][0][:2] == [1.25*m['multiplier'] for m in variant['parameter_specs'][0]['runtime_mappings']]


def test_qwen_receives_mechanism_condition_without_outcome_payload():
    case = load_case('midazolam')
    candidates = runtime.candidate_space(case, root=ROOT)
    request = build_request(case, candidates)
    assert all('kinetics' in c for c in request['candidates'])
    text = json.dumps(request)
    assert 'observations' not in text and 'calibration_log_rmse' not in text
    assert 'local_unbound_substrate_over_Km <= 0.1' in text
    bad = deepcopy(request)
    bad['candidates'][0]['kinetics']['clinical_applicability_verified'] = True
    with pytest.raises(ValueError, match='kinetic descriptors'):
        validate_request(bad)
