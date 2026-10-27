from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
STAGE = next((p for p in Path(__file__).resolve().parents if p.name == 'staged'), ROOT)
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
NAME = 'recommendation_execution_repairs.json_fixed_process_defaults_proof'
spec = importlib.util.spec_from_file_location(NAME, STAGE/'scripts/recommendation_execution_repairs/json_fixed_process_defaults_proof.py')
m = importlib.util.module_from_spec(spec); sys.modules[NAME] = m; spec.loader.exec_module(m)
REG = json.loads((ROOT/m.REGISTRATION).read_text())


def documents(i):
    e = REG['entries'][i]
    return json.loads((ROOT/e['input']['path']).read_text()), json.loads((ROOT/e['export_snapshot']['path']).read_text())


def first_params(document):
    return m.processes(document)[('DabiEtex', m.TRANSPORT, 'ABCB1', None)]['Parameters']


@pytest.fixture(scope='module')
def models():
    values = []
    for entry in REG['entries'][:2]:
        p = ROOT/entry['native_source']['path']
        inventory = m.source_pkml.simulation_inventory(p, entry['native_source']['sha256'], project_root=ROOT)
        row = next((r for r in inventory if r['name'] == 'Delavenne 2012 - DDI Clarithromycin'), inventory[0])
        with sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            blob = db.execute('SELECT Data FROM CONTENTS WHERE Id=?', (row['content_id'],)).fetchone()[0]
        values.append(ET.fromstring(m.source_pkml.decompress_model_blob(blob)))
    return values


@pytest.mark.parametrize('i', range(3))
def test_real_registered_proofs_replay_all_ten_saved_models(i):
    e = REG['entries'][i]
    p = (ROOT/m.REGISTRATION).parent/(e['batch_id']+'_proof_v2.json')
    proof_ref = {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    result = m.verify(e['input'], e['export_snapshot'], e['native_source'], proof_ref, project_root=ROOT)
    assert result['accepted'] is True
    assert result['exact_normalized_document_equality'] is False
    assert len(result['native_simulations']) == len(e['simulation_names'])
    assert all(len(r['default_consumers']) == 5 and len(r['fixed_processes']) == 4 for r in result['native_simulations'])
    assert result['numeric_tolerance'] == {'relative': 0, 'absolute': 0}


@pytest.mark.parametrize('mutation', ['existing_value', 'origin', 'added_value', 'bool_default', 'unknown', 'duplicate', 'mechanism'])
def test_snapshot_rejects_changes_beyond_exact_defaults(mutation):
    a, b = documents(0); params = first_params(b)
    if mutation == 'existing_value': next(p for p in params if p['Name'] == 'kcat')['Value'] += 1e-12
    elif mutation == 'origin': next(p for p in params if p['Name'] == 'kcat').pop('ValueOrigin')
    elif mutation == 'added_value': next(p for p in params if p['Name'] == 'Vmax')['Value'] = 1e-300
    elif mutation == 'bool_default': next(p for p in params if p['Name'] == 'Vmax')['Value'] = False
    elif mutation == 'unknown': params.append({'Name': 'unknown', 'Value': 0})
    elif mutation == 'duplicate': params.append(deepcopy(params[0]))
    elif mutation == 'mechanism': b['Compounds'][0]['Name'] += ' changed'
    with pytest.raises(ValueError): m.compare_documents(a, b)


def test_comparison_does_not_mutate_source_or_native_export():
    a, b = documents(0); before = deepcopy((a, b))
    changes, remainder, implicit = m.compare_documents(a, b)
    assert (a, b) == before and len(changes) == 4 and implicit == []
    assert remainder['exact_normalized_document_equality'] is True


@pytest.mark.parametrize('mutation', ['notfixed', 'value', 'rhs', 'default', 'units', 'assignment', 'consumer', 'formula'])
def test_native_frontier_rejects_any_live_dependency_or_changed_constant(models, mutation):
    doc = deepcopy(models[0]); ix = m.ModelIndex(doc)
    kcat = ix.node('DabiEtex|ABCB1-FIT|kcat')
    if mutation == 'notfixed': kcat.set('isFixedValue', '0')
    elif mutation == 'value': kcat.set('value', '0.460000000001')
    elif mutation == 'rhs': kcat.set('rhs', kcat.get('formula'))
    elif mutation == 'default': ix.node('DabiEtex|ABCB1-FIT|Vmax').set('value', '1e-300')
    elif mutation == 'units': ix.node('DabiEtex|ABCB1-FIT|Vmax').set('displayUnit', 'mg')
    elif mutation == 'assignment': ET.SubElement(ix.model.find('Root/Children'), 'EventAssignment', id='audit-assignment', name='audit', changedEntity=kcat.get('id'))
    elif mutation == 'consumer':
        f = ET.SubElement(ix.model.find('FormulaCache/Formulas'), 'Formula', id='audit-formula', formula='V')
        ET.SubElement(ET.SubElement(f, 'Paths'), 'Path', path='DabiEtex|ABCB1-FIT|Vmax', **{'as':'V'})
        ET.SubElement(ix.model.find('Root/Children'), 'Para', id='audit-consumer', name='audit', formula='audit-formula', value='0')
    elif mutation == 'formula': ix.formulas[kcat.get('formula')].set('formula', '0')
    changes, _, _ = m.compare_documents(*documents(0))
    with pytest.raises(ValueError): m.validate_model(doc, changes)


@pytest.mark.parametrize('which', ['water', 'lag', 'explicit_water', 'start', 'unknown'])
def test_implicit_import_defaults_are_not_general_protocol_tolerance(which):
    a, b = documents(1)
    protocol = next(p for p in b['Protocols'] if p['Name'].endswith('Delavenne 2012 - Clarithromycin'))
    params = protocol['Schemas'][0]['SchemaItems'][0]['Parameters']
    if which == 'water': next(p for p in params if p['Name'] == 'Volume of water/body weight')['Value'] = 3.6
    elif which == 'lag': next(p for f in b['Formulations'] if f['Name'] == 'DabiEtex_Capsule' for p in f['Parameters'] if p['Name'] == 'Lag time')['Value'] = 0.01
    elif which == 'explicit_water': next(p for p in protocol['Schemas'][0]['SchemaItems'][-1]['Parameters'] if p['Name'] == 'Volume of water/body weight')['Value'] = 3.5
    elif which == 'start': next(p for p in params if p['Name'] == 'Start time')['Value'] = 0.01
    elif which == 'unknown': params.append({'Name':'other parameter','Value':0})
    with pytest.raises(ValueError): m.compare_documents(a, b, allow_protocol_defaults=True)


@pytest.mark.parametrize('suffix', ['Start time', 'Volume of water/body weight'])
def test_native_absolute_schedule_and_water_are_exact(models, suffix):
    a, b = documents(1); _, _, implicit = m.compare_documents(a, b, allow_protocol_defaults=True)
    doc = deepcopy(models[1]); ix = m.ModelIndex(doc)
    path = next(p for p in ix.paths if p.startswith('Events|Clarithromycin DDI, Delavenne 2012 - Clarithromycin|') and p.endswith('|ProtocolSchemaItem|'+suffix))
    ix.node(path).set('value', '0.01')
    with pytest.raises(ValueError): m.validate_implicit_defaults(doc, a, b, implicit)


def test_fixed_rule_cannot_apply_to_other_project_or_source():
    e = REG['entries'][0]; wrong = {**e['native_source'], 'sha256': '0'*64}
    with pytest.raises(ValueError, match='REGISTERED_SOURCE_PARTITION_REQUIRED'):
        m.derive(e['input'], e['export_snapshot'], wrong, project_root=ROOT)
