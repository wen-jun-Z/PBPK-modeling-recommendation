"""Only proven product mass and source parent identity may unblock salt dosing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from scripts.recommendation_execution_repairs import source_bound_dose as m


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    scenarios = project/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    scenario = next(x for x in map(json.loads, scenarios.read_text().splitlines()) if x['case_id'] == m.CASE)
    protocol = tmp_path/'protocol.json'
    protocol.write_bytes((project/scenario['source_protocol']['path']).read_bytes())
    scenario['source_protocol']['path'] = 'protocol.json'
    label = tmp_path/'label.pdf'; label.write_bytes(b'%PDF test source label fixture')
    monkeypatch.setattr(m, 'LABEL_PATH', 'label.pdf')
    monkeypatch.setattr(m, 'LABEL_SHA', hashlib.sha256(label.read_bytes()).hexdigest())
    source = tmp_path/'source.json'
    source.write_text(json.dumps({'Compounds': [{'Name': 'Bupropion', 'Parameters': [
        {'Name': 'Molecular weight', 'Value': 239.741, 'Unit': 'g/mol'}]}],
        # Deliberately misleading historical dose. It must never define a factor.
        'Protocols': [{'Name': '150 mg labelled', 'Parameters': [{'Name': 'InputDose', 'Value': 150.0, 'Unit': 'mg'}]}],
        'Simulations': []}))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(m, 'REGISTERED_CANDIDATES', {'test_candidate': source_sha})
    candidate = {'candidate_id': 'test_candidate', 'compounds': ['bupropion'],
        'catalog_provenance': {'source_path': 'source.json', 'source_sha256': source_sha}}
    m._read_original_parent.cache_clear()
    return scenario, candidate, tmp_path


def test_label_salt_not_historical_dose_ratio_and_no_input_mutation(case):
    scenario, candidate, root = case; original = deepcopy(scenario)
    result = m.normalized_salt_contract(scenario, candidate, project_root=root)
    assert result['dose_mg'] == pytest.approx(150 * 239.741 / 276.2, rel=1e-15)
    assert result['dose_mg'] != 150 and result['dose_mg'] != 130.2
    assert scenario == original
    assert result['salt_conversion']['historical_protocol_ratio_used'] is False
    assert result['dose_normalization']['strict_target_match'] is False
    assert result['salt_conversion']['parent_model_source']['sha256'] == candidate['catalog_provenance']['source_sha256']


@pytest.mark.parametrize('field,value', [('value', 200), ('salt', 'hydrobromide'),
    ('formulation', 'different tablet'), ('active_moiety_dose_mg', 150)])
def test_cannot_relabel_frozen_case(case, field, value):
    scenario, candidate, root = case
    scenario['normalized_design']['dose'][field] = value
    with pytest.raises(m.Blocked, match='SALT_DESIGN_DIFFERS'):
        m.normalized_salt_contract(scenario, candidate, project_root=root)


@pytest.mark.parametrize('target,reason', [('label.pdf', 'FDA_PRODUCT_LABEL_CHECKSUM'),
    ('protocol.json', 'SALT_PROTOCOL_CHECKSUM'), ('source.json', 'SALT_MODEL_SOURCE_CHECKSUM')])
def test_file_drift_fails_even_after_cached_source(case, target, reason):
    scenario, candidate, root = case
    m.normalized_salt_contract(scenario, candidate, project_root=root)
    with (root/target).open('ab') as handle: handle.write(b' ')
    with pytest.raises(m.Blocked, match=reason): m.normalized_salt_contract(scenario, candidate, project_root=root)


@pytest.mark.parametrize('mutation,reason', [('salt_mw', 'SOURCE_PARENT_MW_MISMATCH'),
    ('unit', 'SOURCE_PARENT_MW_UNIT_MISMATCH'), ('boolean', 'SOURCE_PARENT_MW_MISMATCH'),
    ('alias', 'UNIQUE_ORIGINAL_BUPROPION'), ('ambiguous', 'UNIQUE_ORIGINAL_BUPROPION'),
    ('override', 'SOURCE_PARENT_MW_OVERRIDE_MISMATCH')])
def test_native_parent_definitions_cannot_be_guessed(case, monkeypatch, mutation, reason):
    scenario, candidate, root = case; source = root/'source.json'; data = json.loads(source.read_text())
    if mutation == 'salt_mw': data['Compounds'][0]['Parameters'][0]['Value'] = 276.2
    if mutation == 'unit': data['Compounds'][0]['Parameters'][0]['Unit'] = 'mg/mol'
    if mutation == 'boolean': data['Compounds'][0]['Parameters'][0]['Value'] = True
    if mutation == 'alias': data['Compounds'][0]['Name'] = 'Hydroxybupropion'
    if mutation == 'ambiguous': data['Compounds'].append(deepcopy(data['Compounds'][0]))
    if mutation == 'override': data['Simulations'] = [{'Parameters': [
        {'Path': 'Bupropion|Molecular weight', 'Value': 276.2, 'Unit': 'g/mol'}]}]
    source.write_text(json.dumps(data)); source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate['catalog_provenance']['source_sha256'] = source_sha
    monkeypatch.setattr(m, 'REGISTERED_CANDIDATES', {'test_candidate': source_sha})
    with pytest.raises(m.Blocked, match=reason): m.normalized_salt_contract(scenario, candidate, project_root=root)


def test_unregistered_candidate_same_drug_does_not_inherit_conversion(case):
    scenario, candidate, root = case; candidate['candidate_id'] = 'another_candidate'
    with pytest.raises(m.Blocked, match='SALT_CANDIDATE_SOURCE_NOT_REGISTERED'):
        m.normalized_salt_contract(scenario, candidate, project_root=root)


def test_other_salt_or_unknown_product_remains_blocked(case):
    scenario, candidate, root = case; scenario['case_id'] = 'dextromethorphan_yamazaki2016_day1_30mg'
    with pytest.raises(m.Blocked, match='SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN'):
        m.normalized_salt_contract(scenario, candidate, project_root=root)


def test_registered_fm_alias_maps_parent_identity(case, monkeypatch):
    scenario, candidate, root = case; source = root/'source.json'; data = json.loads(source.read_text())
    data['Compounds'][0]['Name'] = 'Bupropion_FM'; source.write_text(json.dumps(data))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate['catalog_provenance']['source_sha256'] = source_sha
    monkeypatch.setattr(m, 'REGISTERED_CANDIDATES', {'test_candidate': source_sha})
    result = m.normalized_salt_contract(scenario, candidate, project_root=root)
    assert result['salt_conversion']['parent_definition_evidence'][0]['compound_name'] == 'Bupropion_FM'


@pytest.mark.parametrize('formula,accepted', [('Dose / MW', True), ('Dose / MW * 0.868', False)])
def test_original_pksim_uses_parent_mass_once_not_a_second_salt_factor(case, monkeypatch, formula, accepted):
    from scripts.recommendation_execution_repairs.source_pkml import compress_model_blob
    scenario, candidate, root = case
    xml = ('<IndividualSimulation id="sim" name="reference"><Model><Root><Children>'
        '<Container id="parent" name="Bupropion"><Children>'
        '<Para id="mw" name="Molecular weight" value="2.39741E-07" dim="Molecular weight" />'
        '</Children></Container><Container id="a" name="Applications"><Children>'
        '<Container id="event" name="Application_1"><Children>'
        '<Container id="item" name="ProtocolSchemaItem"><Children>'
        '<Para id="mass" name="DrugMass" formula="f" dim="Amount" />'
        '</Children></Container></Children></Container></Children></Container>'
        '</Children></Root><FormulaCache><Formulas><Formula id="f" formula="'+formula+'">'
        '<Paths><Path path="..|Dose" as="Dose" dim="Mass" />'
        '<Path path="reference|Bupropion|Molecular weight" as="MW" dim="Molecular weight" />'
        '</Paths></Formula></Formulas></FormulaCache></Model></IndividualSimulation>').encode()
    source = root/'source.pksim5'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE SIMULATIONS (SimulationId TEXT)')
        db.execute('CREATE TABLE BUILDING_BLOCKS (Id TEXT, Name TEXT, ContentId TEXT)')
        db.execute('CREATE TABLE CONTENTS (Id TEXT, Data BLOB)')
        db.execute('INSERT INTO SIMULATIONS VALUES (?)', ('sim',))
        db.execute('INSERT INTO BUILDING_BLOCKS VALUES (?,?,?)', ('sim', 'reference', 'content'))
        db.execute('INSERT INTO CONTENTS VALUES (?,?)', ('content', compress_model_blob(xml)))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate['catalog_provenance'].update(source_path='source.pksim5', source_sha256=source_sha)
    monkeypatch.setattr(m, 'REGISTERED_CANDIDATES', {'test_candidate': source_sha})
    if not accepted:
        with pytest.raises(m.Blocked, match='SOURCE_PARENT_DRUGMASS_FORMULA_NOT_MASS_OVER_MW'):
            m.normalized_salt_contract(scenario, candidate, project_root=root)
    else:
        result = m.normalized_salt_contract(scenario, candidate, project_root=root)
        proof = result['salt_conversion']['parent_definition_evidence'][0]
        assert proof['saved_drugmass_formulas'][0]['expression'] == 'Dose / MW'
        assert result['dose_mg'] == pytest.approx(130.19967414916727)
