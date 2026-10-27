"""Cohort choice and human PK unit conversion must not leak outcome values."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_purpose_hundred', ROOT / 'scripts/prepare_purpose_hundred.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_selection_ignores_labels_predictions_and_duplicates():
    rows = [{'identity_smiles': s, 'ht_row': i + 3, 'drug_name': str(i),
             'clinical_label': i, 'prediction_error': i} for i, s in enumerate(['CC', 'CCC', 'CO', 'CCO'])]
    rows.append(dict(rows[0], ht_row=20, drug_name='duplicate', clinical_label=1e9))
    first, _ = prep.freeze_selection(rows, expected=3)
    changed = deepcopy(rows)
    for row in changed:
        row['clinical_label'] = None
        row['prediction_error'] = 0
    second, _ = prep.freeze_selection(list(reversed(changed)), expected=3)
    assert [r['identity_smiles'] for r in first] == [r['identity_smiles'] for r in second]
    assert len({r['identity_smiles'] for r in first}) == 3
    with pytest.raises(ValueError, match='Only 4 unique'):
        prep.freeze_selection(rows, expected=5)


def test_aggregate_references_convert_units_without_inventing_curve_or_labels():
    row = {'compound_id': 'a', 'identity_smiles': 'CC', 'primary_reference_eligible': True,
           'reference_identity_status': 'exact_standardized_stereo_match'}
    labels = {'csv_line': 42, 'labels_log10': {'human_CL_mL_min_kg': math.log10(2),
        'human_VDss_L_kg': math.log10(3), 'human_thalf': math.log10(4), 'human_mrt': None}}
    value = prep.evaluation_record(row, labels, {'path': 'example.whl', 'sha256': 'a'*64})
    assert value['endpoints']['clearance']['value'] == pytest.approx(8.4)
    assert value['endpoints']['vss']['value'] == pytest.approx(210)
    assert value['endpoints']['terminal_half_life']['value'] == pytest.approx(4)
    assert value['endpoints']['mrt']['value'] is None
    assert value['clinical_curve'] is None
    assert value['study_id'] is None
    assert set(value['endpoints']) == {'clearance', 'vss', 'terminal_half_life', 'mrt'}
    assert value['label_role'] == 'evaluation_only'
    row['primary_reference_eligible'] = False
    row['reference_identity_status'] = 'stereo_agreement_unresolved'
    uncertain = prep.evaluation_record(row, labels, {})
    assert all(e['value'] is None for e in uncertain['endpoints'].values())
    assert uncertain['endpoints']['clearance']['connectivity_matched_value_not_scored'] == pytest.approx(8.4)


def test_stereo_reference_matching_does_not_assert_unmarked_isomers_equal():
    assert prep.stereo_parent('C[C@H](O)C(=O)O') != prep.stereo_parent('CC(O)C(=O)O')
    assert prep.stereo_parent('C[C@H](O)C(=O)O') != prep.stereo_parent('C[C@@H](O)C(=O)O')


def test_preparation_never_overwrites_frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'DEFAULT', tmp_path)
    monkeypatch.setattr(prep, 'ROOT', tmp_path)
    path = tmp_path / 'request.json'
    initial = prep.save_new(path, {'input': 1})
    assert prep.save_new(path, {'input': 1}) == initial
    with pytest.raises(FileExistsError):
        prep.save_new(path, {'input': 2})
    assert path.read_text().find('"input": 1') >= 0
