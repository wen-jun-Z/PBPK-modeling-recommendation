"""Independent input provenance and preservation contracts for the new batch."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('expanded_tail_input_test_module', ROOT / 'scripts/prepare_expanded_tail.py')
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def evidence(parameter, value, *, basis='derived', source='empirical.json'):
    return {'parameter': parameter, 'compound_id': 'pubchem:1', 'value': value,
        'unit': 'fraction' if parameter == 'fraction_unbound' else 'ml/min/kg',
        'basis': basis, 'source': {'path': source, 'sha256': 'a' * 64},
        'context': {'study_role': 'development'}, 'notes': 'Original empirical source note'}


PRED = {'predictions': {'human_fup': {'random_forest': .22},
                        'human_CL_mL_min_kg': {'random_forest': 4.}}}
REF = {'path': 'inputs/new_prediction.json', 'sha256': 'b' * 64}


def test_original_prespecified_source_sensitivity_drugs_retained():
    assert set(prep.ORIGINAL_SENSITIVITY_FLAGS) == {'Dexmedetomidine', 'Midazolam', 'Nemonoxacin'}
    assert 'participant_nonoverlap_unconfirmed' in prep.ORIGINAL_SENSITIVITY_FLAGS['Nemonoxacin']


def test_empirical_fu_cl_and_renal_preserved_as_complete_records():
    rows = [evidence('fraction_unbound', .012), evidence('total_clearance', 2.3), evidence('renal_clearance', 1.5)]
    original = deepcopy(rows)
    selected, actions = prep.refresh_evidence(rows, 'Furosemide', 'pubchem:1', REF, PRED)
    assert rows == original == selected
    assert [a['action'] for a in actions] == ['retained_empirical', 'retained_empirical']
    assert actions[1]['source_scope'] == 'independent_third_study'


def test_rf_refresh_replaces_source_and_forbidden_empirical_notes():
    rows = [evidence('fraction_unbound', .1, basis='predicted'), evidence('total_clearance', 2., basis='predicted')]
    selected, actions = prep.refresh_evidence(rows, 'Caffeine', 'pubchem:1', REF, PRED)
    assert [r['value'] for r in selected] == [.22, 4.]
    assert all(r['source'] == REF for r in selected)
    assert all('Original empirical' not in r['notes'] for r in selected)
    assert all(r['prediction_method']['training_overlap'] is False for r in selected)
    assert [a['action'] for a in actions] == ['refreshed_rf', 'refreshed_rf']
    assert rows[0]['value'] == .1


def test_fallback_view_replaces_fu_cl_but_retains_renal():
    rows = [evidence('fraction_unbound', .012), evidence('total_clearance', 2.3), evidence('renal_clearance', 1.5)]
    selected, _ = prep.refresh_evidence(rows, 'Furosemide', 'pubchem:1', REF, PRED, fallback=True)
    assert [r['value'] for r in selected] == [.22, 4., 1.5]
    assert selected[-1] == rows[-1]
    assert all(r['source'] == REF for r in selected[:2])


@pytest.mark.parametrize('drug', ['Caffeine', 'Alfentanil', 'Sufentanil'])
def test_development_curve_clearance_not_accepted_as_empirical_prior(drug):
    rows = [evidence('fraction_unbound', .3), evidence('total_clearance', 2.)]
    with pytest.raises(ValueError, match='unreviewed clinical CL'):
        prep.refresh_evidence(rows, drug, 'pubchem:1', REF, PRED)


def test_existing_moxifloxacin_reported_development_cl_not_claimed_independent():
    rows = [evidence('fraction_unbound', .48), evidence('total_clearance', 2.1), evidence('renal_clearance', .3)]
    selected, actions = prep.refresh_evidence(rows, 'Moxifloxacin', 'pubchem:1', REF, PRED)
    assert selected == rows
    assert actions[1]['source_scope'] == 'directly_reported_development_study_clearance_not_curve_recalculation'


def test_duplicate_primary_and_incompatible_renal_are_errors():
    with pytest.raises(ValueError, match='Conflicting'):
        prep.refresh_evidence([evidence('fraction_unbound', .2)] * 2, 'Furosemide', 'pubchem:1', REF, PRED)
    with pytest.raises(ValueError, match='conflicts with retained renal'):
        prep.refresh_evidence([evidence('renal_clearance', 5.)], 'Furosemide', 'pubchem:1', REF, PRED)


def synthetic_split(monkeypatch):
    def identity(s):
        return {'identity_smiles': s, 'scaffold_group': 'scaffold_' + s, 'scaffold_smiles': s}
    monkeypatch.setattr(prep, 'normalize_structure', identity)
    structures = {'Drug' + str(i): 'external' + str(i) for i in range(20)}
    external = {n: {'input_smiles': s, **identity(s)} for n, s in structures.items()}
    predictions = {n: {**identity(s), **deepcopy(PRED), 'training_structure_overlap': False,
        'training_scaffold_overlap': False} for n, s in structures.items()}
    rows = [{'index': i, **identity('table' + str(i)),
             'status': 'train' if i < 1072 else 'test' if i < 1172 else 'excluded'} for i in range(1283)]
    split = {'external_exclusions': external, 'rows': rows, 'train_indices': list(range(1072)),
        'test_indices': list(range(1072, 1172)), 'train_count': 1072, 'test_count': 100, 'label_audit': {}}
    return split, predictions, structures


def test_all_twenty_exclusions_and_true_denominators(monkeypatch):
    split, predictions, structures = synthetic_split(monkeypatch)
    result = prep.audit_split(split, predictions, structures)
    assert result['row_dispositions'] == {'train': 1072, 'test': 100, 'excluded': 111}
    assert result['unique_drugs'] == 20 and result['rf_refitted'] is False
    assert all(r['train_identity_smiles_overlap'] == r['train_scaffold_group_overlap'] == 0 for r in result['records'])


@pytest.mark.parametrize('kind', ['train', 'test'])
@pytest.mark.parametrize('key', ['identity_smiles', 'scaffold_group'])
def test_structure_and_scaffold_overlap_in_either_partition_rejected(monkeypatch, kind, key):
    split, predictions, structures = synthetic_split(monkeypatch)
    split['rows'][split[kind + '_indices'][0]][key] = split['external_exclusions']['Drug0'][key]
    with pytest.raises(ValueError, match='overlaps RF data'):
        prep.audit_split(split, predictions, structures)


def test_membership_or_prediction_identity_cannot_be_relabelled(monkeypatch):
    split, predictions, structures = synthetic_split(monkeypatch)
    split['rows'][0]['status'] = 'excluded'
    with pytest.raises(ValueError, match='membership'):
        prep.audit_split(split, predictions, structures)
    split, predictions, structures = synthetic_split(monkeypatch)
    predictions['Drug0']['scaffold_group'] = 'other'
    with pytest.raises(ValueError, match='Structure disagrees'):
        prep.audit_split(split, predictions, structures)


def test_append_is_idempotent_and_changed_output_is_refused(tmp_path):
    from pbpk_kb.blueprints.evidence_blueprint import Source
    root = tmp_path; run = root / 'experiments/blueprint_design/expanded_tail_test'; run.mkdir(parents=True)
    store = prep.Sources(root); path = run / 'inputs/unified/one.json'
    first = prep.save_new(store, run, path, {'x': 1})
    assert Source.model_validate(first).location
    assert prep.save_new(store, run, path, {'x': 1}) == first
    with pytest.raises(ValueError, match='Refusing to replace'):
        prep.save_new(store, run, path, {'x': 2})
    assert json.loads(path.read_text()) == {'x': 1}
    with pytest.raises(ValueError, match='only to this run inputs'):
        prep.save_new(store, run, root / 'paper.json', {})


def test_frozen_batch_cannot_have_inputs_reprepared(tmp_path):
    run = tmp_path / 'experiments/blueprint_design/expanded_tail_test'; run.mkdir(parents=True)
    (run / 'state.json').write_text('{}')
    with pytest.raises(ValueError, match='freeze before experiment'):
        prep.prepare_inputs(run, root=tmp_path)


def test_generated_source_and_ledger_pass_real_schema_assembly(tmp_path):
    """Real construction schema catches missing source fields before native work."""
    store = prep.Sources(ROOT)
    protocol = store.read(prep.ORIGINAL)
    case = next(c for c in protocol['cases'] if c['drug'] == 'Treprostinil')
    request = store.read(case['requests']['evidence']['development'])
    source = prep.save_new(store, tmp_path, tmp_path / 'inputs/prediction.json', PRED)
    selected, _ = prep.refresh_evidence(request['evidence'], case['drug'], case['compound_id'], source, PRED)
    ledger = prep.save_new(store, tmp_path, tmp_path / 'inputs/ledger.json', {'evidence': selected})
    request['evidence'] = selected
    request['evidence_ledger'] = ledger
    result = prep.assemble_evidence_blueprint(request, ROOT)
    assert result['status'] == 'assembled'
