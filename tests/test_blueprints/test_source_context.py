from pathlib import Path
import hashlib
import json
import pytest

from pbpk_kb.blueprints.source_adaptation import load_source_model, adapt_source_model
from pbpk_kb.blueprints.source_context import apply_source_context, install_calibrated_parameters

ROOT = Path(__file__).resolve().parents[2]


def source(name, suffix, simulation):
    path = ROOT / f'artifacts/pbpk-model-assets/public_collection_20260909/{name}-Model__{suffix}/source/{name}-Model.json'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    blueprint = adapt_source_model(load_source_model(path, sha, simulation, ROOT), {}, ROOT)
    evidence = {'path': str(path.relative_to(ROOT)), 'sha256': sha, 'locator': 'test-only source identity; not a clinical endorsement'}
    return blueprint, evidence


def test_documented_sex_rebuilds_individual_without_inventing_weight():
    b, e = source('Rifampicin', '1565559bf7b7', 'Rifampicin po 450 mg MD OD (7 days)')
    r = apply_source_context(b, {'human_sex': {'value': 'FEMALE', 'evidence': e}}, ROOT)
    assert r['status'] == 'adapted'
    name = r['snapshot']['Simulations'][0]['Individual']
    individual = next(x for x in r['snapshot']['Individuals'] if x['Name'] == name)
    assert individual['OriginData']['Gender'] == 'FEMALE'
    assert 'Weight' not in individual['OriginData']
    assert b['snapshot']['Compounds'] == r['snapshot']['Compounds']
    assert b['snapshot']['Simulations'][0]['Individual'] != name


def test_known_alternative_preserves_mechanisms_and_marks_hypothesis():
    b, e = source('Midazolam', 'cc916b56d5e3', 'po 15 mg (tablet) - with high-fat breakfast')
    r = apply_source_context(b, {'solubility_alternative': {'value': 'FeSSIF', 'evidence': e}}, ROOT)
    assert r['status'] == 'adapted'
    assert r['snapshot']['Compounds'] == b['snapshot']['Compounds']
    assert r['scientific_validated'] is False


def test_unsupported_demographics_and_unknown_alternatives_are_blocked():
    b, e = source('Midazolam', 'cc916b56d5e3', 'po 15 mg (tablet) - with high-fat breakfast')
    assert apply_source_context(b, {'weight_kg': 52}, ROOT)['status'] == 'blocked'
    assert apply_source_context(b, {'solubility_alternative': {'value': 'invented', 'evidence': e}}, ROOT)['status'] == 'blocked'
    assert apply_source_context(b, {'human_sex': {'value': 'FEMALE', 'evidence': e}}, ROOT)['status'] == 'blocked'


def test_calibration_override_keeps_drug_mechanisms_and_checks_bounds():
    b, _ = source('Rifampicin', '1565559bf7b7', 'Rifampicin po 450 mg MD OD (7 days)')
    spec = {'path': 'Rifampicin|Specific intestinal permeability (transcellular)', 'unit': 'cm/min',
            'lower_bound': 1e-6, 'upper_bound': 1e-4}
    modified = install_calibrated_parameters(b['snapshot'], [spec], [2e-5], 'fit.json')
    assert modified['Compounds'] == b['snapshot']['Compounds']
    assert modified['Simulations'][0]['Parameters'][-1]['Value'] == 2e-5
    with pytest.raises(ValueError):
        install_calibrated_parameters(b['snapshot'], [spec], [2], 'fit.json')
