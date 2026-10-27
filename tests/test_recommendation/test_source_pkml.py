"""Source-only model extraction and isolated export integrity."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from recommendation_execution_repairs import source_pkml as m
from recommendation_execution_repairs.snapshot_overlay import restore_explicit_simulation_fields


def saved_xml():
    return ET.fromstring('''<IndividualSimulation id="simulation-id" name="Known model" xmlVersion="76">
      <Properties><ModelProperties modelConfiguration="4Comp|Human"/></Properties>
      <Model id="model-id" name="Known model"><Root id="root-id" name="Known model"><Children>
        <EventGroup id="event-id" name="Applications"><Children>
          <Para id="dose-id" name="Dose" value="0.000025" dim="Mass" displayUnit="mg"/>
        </Children></EventGroup>
      </Children></Root><FormulaCache><Formulas/><StringMap/></FormulaCache></Model>
      <SimulationSettings id="settings-id" randomSeed="123"><Solver id="solver-id" name="CVODES"/></SimulationSettings>
      <UsedBuildingBlocks/><UsedObservedData><ObservedData id="forbidden-outcome-reference"/></UsedObservedData>
    </IndividualSimulation>''')


def fixture_project(tmp_path, *, document=None):
    path=tmp_path/'source.pksim5';blob=m.compress_model_blob(ET.tostring(saved_xml() if document is None else document))
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE SIMULATIONS (SimulationId TEXT,SimulationMode TEXT,DataRepositoryId TEXT,SimulationResultsId INT,SimulationAnalysesId INT);
          CREATE TABLE BUILDING_BLOCKS (Id TEXT,Name TEXT,ContentId INT);
          CREATE TABLE CONTENTS (Id INT,Data BLOB);
          CREATE TABLE OBSERVED_DATA (Id TEXT); CREATE TABLE DATA_REPOSITORIES (Id TEXT);
          CREATE TABLE QUANTITY_VALUES (Id TEXT);''')
        db.execute('INSERT INTO SIMULATIONS VALUES (?,?,?,?,?)',('simulation-id','Individual','old-result',2,3))
        db.execute('INSERT INTO BUILDING_BLOCKS VALUES (?,?,?)',('simulation-id','Known model',1))
        db.execute('INSERT INTO CONTENTS VALUES (?,?)',(1,blob))
        db.execute('INSERT INTO CONTENTS VALUES (?,?)',(2,b'FORBIDDEN_OLD_OBSERVATION_PAYLOAD'))
        db.execute('INSERT INTO OBSERVED_DATA VALUES (?)',('old-observation',))
    return path


def test_blob_roundtrip_crc_and_bound_checks():
    raw=ET.tostring(saved_xml());blob=m.compress_model_blob(raw)
    assert m.decompress_model_blob(blob)==raw
    broken=bytearray(blob);broken[14]^=1
    with pytest.raises(ValueError,match='CRC'):m.decompress_model_blob(bytes(broken))
    with pytest.raises(ValueError,match='TRAILING'):m.decompress_model_blob(blob+b'additional-data')


def test_native_utf8_filename_flag_is_supported_but_not_encryption():
    raw=ET.tostring(saved_xml());blob=bytearray(m.compress_model_blob(raw))
    blob[6:8]=(0x0800).to_bytes(2,'little')
    assert m.decompress_model_blob(bytes(blob))==raw
    blob[6:8]=(0x0801).to_bytes(2,'little')
    with pytest.raises(ValueError,match='UNSUPPORTED'):m.decompress_model_blob(bytes(blob))


def test_source_hash_identity_and_selected_blob_are_mandatory(tmp_path):
    path=fixture_project(tmp_path);digest=m.sha(path)
    with pytest.raises(ValueError,match='BINDING'):m.simulation_inventory(path,'0'*64)
    with pytest.raises(ValueError,match='EXACT_SAVED_SIMULATION'):
        m.extract_saved_model(path,digest,'different-id',tmp_path/'bad')
    result=m.extract_saved_model(path,digest,'simulation-id',tmp_path/'extract')
    assert result['audit']['model_and_settings_elements_unchanged']
    assert result['audit']['parameter_xml_count']==1
    assert b'forbidden-outcome-reference' not in (ROOT/result['pkml']['path']).read_bytes()


def test_native_working_copy_clears_only_new_copy_outcomes_and_preserves_model(tmp_path):
    path=fixture_project(tmp_path);digest=m.sha(path)
    result=m.prepare_native_export_copy(path,digest,'simulation-id',tmp_path/'isolated')
    assert m.sha(path)==digest
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as original:
        assert original.execute('SELECT count(*) FROM OBSERVED_DATA').fetchone()==(1,)
    with sqlite3.connect(ROOT/result['working_copy']['path']) as copy:
        assert copy.execute('SELECT count(*) FROM OBSERVED_DATA').fetchone()==(0,)
        assert copy.execute('SELECT DataRepositoryId,SimulationResultsId,SimulationAnalysesId FROM SIMULATIONS').fetchone()==(None,None,None)
        doc=ET.fromstring(m.decompress_model_blob(copy.execute('SELECT Data FROM CONTENTS WHERE Id=1').fetchone()[0]))
        assert doc.find('UsedObservedData') is None
        assert m.element_sha(doc.find('Model'))==m.element_sha(saved_xml().find('Model'))
        assert m.element_sha(doc.find('SimulationSettings'))==m.element_sha(saved_xml().find('SimulationSettings'))


def test_source_metadata_never_interprets_display_unit_as_base_unit(tmp_path):
    path=fixture_project(tmp_path)
    result=m.source_simulation_metadata(path,m.sha(path))['simulations'][0]
    row=result['event_parameters'][0]
    assert row['value']=='0.000025' and row['dimension']=='Mass' and row['display_unit']=='mg'
    assert 'base values' in result['numeric_convention']


def test_imported_compound_identity_comes_from_typed_model_and_assignment_target():
    document=ET.fromstring('''<PopulationSimulation id="s" name="Imported">
      <Model><Root id="r" name="Imported"><Children>
        <Container id="drug" name="Caffeine_template" containerType="Molecule"><Children>
          <Para id="mw" name="Molecular weight" dim="Molecular weight" value="1.94e-7"/>
          <Para id="target" name="Total oral drug mass" value="0"/>
        </Children></Container>
        <EventGroup id="ev" name="Applications" eventGroupType="Oral"><Children>
          <EventAssignment id="a" changedEntity="target"/>
        </Children></EventGroup>
      </Children></Root></Model><UsedBuildingBlocks/>
    </PopulationSimulation>''')
    result=m.describe_saved_simulation(document)
    assert result['compounds']==['Caffeine_template']
    assert result['administrations'][0]['administered_compounds']==['Caffeine_template']
    assert result['administrations'][0]['compound_association_complete']
    assert result['parent_molecular_weight_parameters'][0]['value']=='1.94e-7'


def test_snapshot_overlay_cannot_hide_unrelated_mechanism_difference():
    a={'Version':79,'Compounds':[{'Name':'Drug','Parameters':[{'Name':'CL','Value':1}]}],
       'Simulations':[{'Name':'case','Compounds':[{'Name':'Drug','Alternatives':[{'GroupName':'G','AlternativeName':'chosen'}]}],
                       'Parameters':[{'Path':'Drug|some flag','Value':0}]}]}
    b=deepcopy(a);b['Version']=80;b['Simulations'][0]['Parameters']=[];b['Simulations'][0]['Compounds'][0]['Alternatives']=[]
    r=restore_explicit_simulation_fields(a,b)
    assert r['status']=='restored_pending_native_readback' and r['comparison_after']['accepted']
    assert r['native_execution_approved'] is False
    b['Compounds'][0]['Parameters'][0]['Value']=2
    r=restore_explicit_simulation_fields(a,b)
    assert r['snapshot'] is None and r['status']=='additional_source_changes_unresolved'


def test_real_atorvastatin_and_pitavastatin_original_fields_can_be_restored():
    import recommendation_candidate_migration as migration
    rows=[json.loads(x) for x in (ROOT/migration.OUT/'inputs/candidate_audit.jsonl').open()]
    targets=('7b45f757','86ab3bd9')
    for task in migration.collect_tasks(rows):
        if task['source_sha256'].startswith(targets):
            before=migration.revalidate_old_artifact(task)
            result=restore_explicit_simulation_fields(before['original'],before['converted'])
            assert result['comparison_after']['accepted'] and result['changes']
            assert result['snapshot']['Compounds']==before['converted']['Compounds']


@pytest.mark.parametrize('tag', ['SimulationSettings', 'Settings'])
def test_original_settings_alias_copy_and_wrapper_preserve_full_xml(tmp_path, tag):
    document=saved_xml(); settings=document.find('SimulationSettings'); settings.tag=tag
    original_bytes=ET.tostring(document); original_settings_hash=m.element_sha(settings)
    source=fixture_project(tmp_path,document=document); source_hash=m.sha(source)
    result=m.prepare_native_export_copy(source,source_hash,'simulation-id',tmp_path/'isolated')
    assert result['model_settings_sha256'][tag]==original_settings_hash
    with sqlite3.connect(ROOT/result['working_copy']['path']) as db:
        copied=ET.fromstring(m.decompress_model_blob(db.execute('SELECT Data FROM CONTENTS WHERE Id=1').fetchone()[0]))
    assert m.element_sha(m.saved_settings(copied))==original_settings_hash
    assert m.saved_settings(copied).tag==tag and m.sha(source)==source_hash
    wrapped,audit=m.wrap_preserved_model(document)
    assert audit['settings_xml_sha256']==original_settings_hash
    assert audit['model_and_settings_elements_unchanged']
    configuration=wrapped.find('Simulation/SimulationConfiguration')
    assert m.element_sha(m.saved_settings(configuration))==original_settings_hash
    assert ET.tostring(document)==original_bytes


@pytest.mark.parametrize('tags', [[], ['Settings','SimulationSettings'], ['Settings','Settings'], ['SimulationSettings','SimulationSettings']])
def test_missing_or_ambiguous_settings_rejected_before_copy(tmp_path,tags):
    document=saved_xml(); original=document.find('SimulationSettings');document.remove(original)
    for tag in tags:
        node=deepcopy(original);node.tag=tag;document.append(node)
    source=fixture_project(tmp_path,document=document)
    with pytest.raises(ValueError,match='EXACTLY_ONE_SAVED_SIMULATIONSETTINGS'):
        m.prepare_native_export_copy(source,m.sha(source),'simulation-id',tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
    with pytest.raises(ValueError,match='EXACTLY_ONE_SAVED_SIMULATIONSETTINGS'):
        m.wrap_preserved_model(document)


def test_missing_saved_model_reports_explicit_error_before_copy(tmp_path):
    document=saved_xml();document.remove(document.find('Model'))
    source=fixture_project(tmp_path,document=document)
    with pytest.raises(ValueError,match='EXACTLY_ONE_SAVED_MODEL_REQUIRED'):
        m.prepare_native_export_copy(source,m.sha(source),'simulation-id',tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
