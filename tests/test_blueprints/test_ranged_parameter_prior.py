"""Real small RF fits verify cohort exclusion and range-only calibration."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import ranged_parameter_prior as ranged

ROOT = Path(__file__).resolve().parents[2]
SMILES = [
    'c1ccccc1', 'Cc1ccccc1', 'c1ccncc1', 'CCc1ccncc1',
    'C1CCCCC1', 'C1CCCC1', 'C1CCC1', 'C1CC1',
    'c1ccoc1', 'c1ccsc1', 'c1ncc[nH]1', 'c1ncncc1',
    'C1COCCN1', 'C1CCNCC1', 'C1CCOC1', 'C1COC1',
    'c1ccc2ccccc2c1', 'O=C1CCCCC1', 'O=C1NCCC1',
    'C1CC2CCC1C2', 'c1ccc2[nH]ccc2c1', 'O=C1OC=CC=C1',
]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip('Use --basetemp within the project runtime directory')
    monkeypatch.setattr(ranged, 'CALIBRATION_LIMIT', 3)
    fields = ['smiles_r', *ranged.base.ENDPOINTS]
    rows = []
    for i, smiles in enumerate(SMILES):
        rows.append({'smiles_r': smiles, 'human_VDss_L_kg': .1 + i*.01,
            'human_CL_mL_min_kg': .2 + i*.025, 'human_fup': .15 + i*.01,
            'human_mrt': .4 + i*.015, 'human_thalf': .5 + i*.02})
    def write(name, current):
        p = tmp_path/name
        with p.open('w', newline='') as f:
            out = csv.DictWriter(f, fieldnames=fields); out.writeheader(); out.writerows(current)
        return p
    return rows, write, tmp_path, {'external_a': SMILES[0], 'external_b': SMILES[2]}


def test_fit_excludes_identity_scaffold_and_preserves_artifact_contract(synthetic):
    rows, write, folder, cohort = synthetic
    source = write('data.csv', rows)
    args = dict(exclude_smiles=cohort, output_dir=folder/'fitted', project_root=ROOT, n_jobs=1)
    result = ranged.fit_ranged_prior(source, **args)
    audit = result['audit']
    assert not audit['calibration_is_independent_test']
    assert audit['calibration_count'] == 3
    assert set(audit['train_indices']).isdisjoint(audit['calibration_indices'])
    protected = {r['scaffold_group'] for r in audit['external_exclusions'].values()}
    included = [r for r in audit['rows'] if r['status'] in {'train', 'calibration'}]
    assert not protected & {r['scaffold_group'] for r in included}
    assert all(r['status'] == 'excluded' for r in audit['rows'][:4])
    assert audit['forest_parameters']['n_estimators'] == 200
    assert audit['forest_parameters']['min_samples_leaf'] == 3
    assert audit['forest_parameters']['max_features'] == 'sqrt'
    assert set(result['predictions']) == set(cohort)
    for record in result['predictions'].values():
        assert not record['cohort_labels_used']
        assert not record['training_structure_overlap']
        assert not record['calibration_scaffold_overlap']
        assert set(record['predict_nominal']) == {'total_clearance', 'fraction_unbound'}
        for parameter, value in record['ranges'].items():
            assert 0 < value['lower'] <= value['central'] <= value['upper']
            assert not value['clinical_confidence_interval']
            assert value['calibration_n'] == audit['label_audit'][parameter]['calibration_count']
    assert {p.name for p in (folder/'fitted').iterdir()} == {'parameter_models.joblib', 'audit.json', 'predictions.json'}
    contents = {p.name: p.read_bytes() for p in (folder/'fitted').iterdir()}
    reused = ranged.fit_ranged_prior(source, **args)
    assert reused['reused'] and reused['predictions'] == result['predictions']
    assert {p.name: p.read_bytes() for p in (folder/'fitted').iterdir()} == contents
    with pytest.raises(ValueError, match='different frozen'):
        ranged.fit_ranged_prior(source, **{**args, 'exclude_smiles': {'different': SMILES[0]}})
    p = folder/'fitted'/'predictions.json'; j = json.loads(p.read_text())
    j['predictions']['external_a']['predict_nominal']['total_clearance'] *= 2
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError, match='predictions changed'):
        ranged.fit_ranged_prior(source, **args)


def test_external_labels_never_change_fits_or_ranges_but_calibration_only_changes_ranges(synthetic):
    rows, write, folder, cohort = synthetic
    def fit(name, data):
        return ranged.fit_ranged_prior(write(name+'.csv', data), exclude_smiles=cohort,
            output_dir=folder/name, project_root=ROOT, n_jobs=1)
    original = fit('original', rows)
    modified = deepcopy(rows)
    for i in range(4):
        modified[i]['human_CL_mL_min_kg'] = 10.
        modified[i]['human_fup'] = .000001
    excluded = fit('excluded_labels_changed', modified)
    assert original['predictions'] == excluded['predictions']
    modified = deepcopy(rows)
    index = original['audit']['label_audit']['total_clearance']['calibration_indices'][0]
    modified[index]['human_CL_mL_min_kg'] += 3
    changed = fit('calibration_label_changed', modified)
    for name in cohort:
        assert changed['predictions'][name]['predict_nominal'] == original['predictions'][name]['predict_nominal']
    assert changed['audit']['calibration']['total_clearance']['log10_residual_q90'] > original['audit']['calibration']['total_clearance']['log10_residual_q90']


def test_guard_rejects_cohort_or_train_calibration_leakage(synthetic):
    rows, write, folder, cohort = synthetic
    dataset = ranged.base.load_parameter_dataset(write('dataset.csv', rows))
    split = ranged.base.make_scaffold_split(dataset, exclude_smiles=cohort,
        train_limit=len(rows), test_limit=3, seed=ranged.SEED)
    overlap = deepcopy(split); overlap['train_indices'].append(split['test_indices'][0])
    with pytest.raises(ValueError, match='row leakage'):
        ranged._validate_split(dataset, overlap, cohort)
    external = deepcopy(split); external['train_indices'].append(0)
    with pytest.raises(ValueError, match='cohort labels'):
        ranged._validate_split(dataset, external, cohort)


def test_fu_physical_bound_cl_untouched_and_zero_width_is_honest():
    bound = ranged._stress_range(math.log10(.8), .3, 'fraction_unbound', 25)
    assert bound['upper'] == 1. and bound['upper_clipped_to_one']
    assert bound['upper_before_physical_bound'] > 1.
    assert bound['central'] == pytest.approx(.8)
    cl = ranged._stress_range(math.log10(2.), 1., 'total_clearance', 25)
    assert cl['lower'] == pytest.approx(.2) and cl['upper'] == pytest.approx(20.)
    assert not cl['renal_clearance_used_for_clipping']
    q, residuals = ranged._residual_quantile([0., 1., 2.], [0., 1., 2.])
    assert q == 0. and residuals == [0., 0., 0.]
    exact = ranged._stress_range(0., q, 'total_clearance', 3)
    assert exact['lower'] == exact['central'] == exact['upper'] == 1.
    assert exact['degenerate'] and exact['limited_calibration_count']
    assert ranged._residual_quantile([0., 0., 0.], [1., 2., 3.])[0] == 3.


@pytest.mark.parametrize('observed,predicted', [([], []), ([1.], [float('nan')]),
    ([float('inf')], [1.]), ([1., 2.], [1.])])
def test_invalid_calibration_is_rejected(observed, predicted):
    with pytest.raises(ValueError, match='calibration'):
        ranged._residual_quantile(observed, predicted)


@pytest.mark.parametrize('center,width,n', [(float('nan'), .1, 5), (0., float('inf'), 5),
    (0., -.1, 5), (400., .1, 5), (-400., .1, 5), (0., .1, 0)])
def test_nonfinite_or_invalid_ranges_fail_closed(center, width, n):
    with pytest.raises(ValueError):
        ranged._stress_range(center, width, 'total_clearance', n)


def test_incomplete_directory_is_never_overwritten(synthetic):
    rows, write, folder, cohort = synthetic
    output = folder/'partial'; output.mkdir(); (output/'diagnostic.txt').write_text('keep')
    with pytest.raises(FileExistsError, match='Incomplete'):
        ranged.fit_ranged_prior(write('source.csv', rows), exclude_smiles=cohort,
            output_dir=output, project_root=ROOT)
    assert (output/'diagnostic.txt').read_text() == 'keep'
