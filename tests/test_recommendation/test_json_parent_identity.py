"""Exact author-bound names preserve parent identity; never starts native code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import pytest
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs import json_parent_identity as identity

BASE=ROOT/'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_identity_followup_20261002'


def contract(key,binding):
    return {'compound':key[0],'case_id':binding['case_id'],'analyte':'parent','binding':'total','matrix':'plasma'}


@pytest.mark.parametrize('key',list(identity._BINDINGS))
def test_real_author_and_full_chemical_definition_bind_unique_parent(key):
    binding=identity._BINDINGS[key]
    r=identity.source_parent_identity(contract(key,binding),key[1],project_root=ROOT)
    assert r['native_name']==binding['native_name']
    assert r['clinical_parent']==key[0]
    assert r['compound_definition_sha256']==binding['compound_sha256']
    assert r['dose_rescaled'] is r['concentrations_aggregated'] is r['source_compound_or_mechanism_renamed'] is False


@pytest.mark.parametrize('field,value',[('case_id','unregistered'),('analyte','metabolite'),('binding','unbound'),('matrix','serum')])
def test_unregistered_analysis_is_not_enabled(field,value,tmp_path):
    key=next(iter(identity._BINDINGS));c=contract(key,identity._BINDINGS[key]);c[field]=value
    with pytest.raises(ValueError,match='FROZEN_PARENT_ANALYTE'):
        identity.source_parent_identity(c,key[1],project_root=tmp_path)


@pytest.mark.parametrize('parent,source',[('montelukast','0'*64),('pioglitazone','d625e417f439724e54edf6bb03f76ba0bf09057d0a52dda6db1e78f8480e188b'),('omeprazole','0'*64),('posaconazole','77d4db0edcb3811789c0289534b585ee286d3071c21f8a71cc92dd71b0816467')])
def test_no_fuzzy_or_cross_drug_alias(parent,source,tmp_path):
    assert identity.source_parent_identity({'compound':parent},source,project_root=tmp_path) is None


@pytest.fixture
def archive(tmp_path,monkeypatch):
    key=next(iter(identity._BINDINGS));binding=deepcopy(identity._BINDINGS[key])
    for field in ('source','author_document'):
        path=tmp_path/binding[field]['path'];path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/binding[field]['path'],path)
    monkeypatch.setattr(identity,'_BINDINGS',{key:binding})
    return key,binding,tmp_path


@pytest.mark.parametrize('field',['source','author_document'])
def test_source_or_author_evidence_drift_rejected(archive,field):
    key,binding,root=archive;(root/binding[field]['path']).write_text('changed')
    with pytest.raises(ValueError,match='EVIDENCE_CHANGED'):
        identity.source_parent_identity(contract(key,binding),key[1],project_root=root)


@pytest.mark.parametrize('fault',['missing','duplicate','definition','weight'])
def test_parent_definition_remains_unique_and_exact_even_with_new_source_hash(archive,fault):
    key,binding,root=archive;path=root/binding['source']['path'];document=json.loads(path.read_text())
    parent=next(c for c in document['Compounds'] if c['Name']==binding['native_name'])
    if fault=='missing':document['Compounds'].remove(parent)
    elif fault=='duplicate':document['Compounds'].append(deepcopy(parent))
    elif fault=='definition':parent['Processes']=[]
    else:
        next(p for p in parent['Parameters'] if p['Name']=='Molecular weight')['Value']+=1
        binding['compound_sha256']=identity._digest(parent)
    path.write_text(json.dumps(document));binding['source']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='COMPOUND_DEFINITION_CHANGED|MOLECULAR_WEIGHT_CHANGED'):
        identity.source_parent_identity(contract(key,binding),key[1],project_root=root)


def test_two_real_preparations_preserve_source_and_do_not_execute():
    report=json.loads((BASE/'alias_preparation_results.json').read_text())
    assert report['prospective_only'] and not report['native_execution_started'] and not report['sealed_observations_accessed']
    assert len(report['rows'])==2 and all(r['status']=='ready' for r in report['rows'])
    assert all(all(row['preservation_checks'].values()) for row in report['rows'])
    assert {r['identity']['native_name'] for r in report['rows']}=={'Montelukast_Ince et al 2019','Pioglitazone_DT'}
    for row in report['rows']:
        assert hashlib.sha256((ROOT/row['plan']['path']).read_bytes()).hexdigest()==row['plan']['sha256']
    assert 'scripts/recommendation_execution_repairs/json_parent_identity.py' in report['implementation_sha256']
