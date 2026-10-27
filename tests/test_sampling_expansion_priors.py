"""Scope and identity safety for the expansion's parameter priors."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('sampling_expansion_priors', ROOT / 'scripts/prepare_sampling_expansion_priors.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_standard_molecules_preserve_original_normalization():
    from pbpk_kb.blueprints.pk_parameter_learning import normalize_structure
    expanded = module.complete_external_normalizer(normalize_structure)
    for smiles in ('CCO', 'CC1=C(C(=CC=C1)C)OCC(C)N', 'O=C(O)c1ccccc1'):
        assert expanded(smiles) == normalize_structure(smiles)


def test_invalid_identity_does_not_trigger_budget_fallback():
    from pbpk_kb.blueprints.pk_parameter_learning import normalize_structure
    expanded = module.complete_external_normalizer(normalize_structure)
    for smiles in ('', 'C C', 'invalid'):
        with pytest.raises(ValueError):
            expanded(smiles)


def test_new_chemistry_has_no_copied_clinical_or_fitted_pk():
    source = module.prepare_chemistry()
    assert len(source) == 10
    for ref in source.values():
        doc = module.read(ROOT / ref['path'])
        assert {e['parameter'] for e in doc['evidence']} == {
            'molecular_weight', 'molecular_formula', 'logp', 'pka_types'}
        module.verify_sources(doc)
        assert all(e['context']['target_observations_used'] is False for e in doc['evidence'])
        assert all(e['context']['fitted_compound_parameters_used'] is False for e in doc['evidence'])


def test_project_scope_and_prepared_input_immutability(tmp_path):
    with pytest.raises(ValueError, match='Project boundary'):
        module.ref(ROOT.parent / '__must_reject_before_read__')
    with pytest.raises(ValueError, match='Output must remain'):
        module.save(tmp_path / 'outside.json', {})
    p = module.BATCH / 'inputs/physchem/finerenone.json'
    before = p.read_bytes()
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        module.save(p, {'different': True})
    assert p.read_bytes() == before


def test_full_30_structure_and_scaffold_exclusion():
    folder = module.BATCH / 'inputs/priors'
    data = module.read(folder / 'predictions.json')
    audit = module.read(folder / 'parameter_split.json')
    assert len(data['external_predictions']) == len(audit['external_exclusions']) == 30
    structures = {r['identity_smiles'] for r in audit['external_exclusions'].values()}
    scaffolds = {r['scaffold_group'] for r in audit['external_exclusions'].values()}
    assert len(structures) == 30
    for row in audit['rows']:
        if row['status'] in {'train', 'test'}:
            assert row['identity_smiles'] not in structures
            assert row['scaffold_group'] not in scaffolds
    for pred in data['external_predictions'].values():
        assert not pred['training_structure_overlap']
        assert not pred['training_scaffold_overlap']
        assert pred['predictions']['human_CL_mL_min_kg']['unit'] == 'mL/min/kg'
        assert pred['predictions']['human_VDss_L_kg']['unit'] == 'L/kg'
        assert pred['predictions']['human_fup']['unit'] == 'fraction'
        assert 0 < pred['predictions']['human_fup']['random_forest'] <= 1
    module.verify_sources(data['sources'])
