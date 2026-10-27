from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from scripts.recommendation_execution_repairs import coadministration_isolation as iso
ROOT=Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
spec=importlib.util.spec_from_file_location('cache_existing_fixture',ROOT/'tests/test_recommendation/test_coadministration_isolation.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(iso,'_ZERO_PROOF_CACHE',iso.OrderedDict())
    monkeypatch.setattr(iso,'_ZERO_PROOF_CACHE_BYTES',0)


def tracked(monkeypatch):
    calls=[];original=iso._prove_zero_manifold_uncached
    def run(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(iso,'_prove_zero_manifold_uncached',run)
    return calls


def test_cache_cold_warm_are_exact_and_returns_never_share_mutable_proof(monkeypatch):
    doc,metadata=old.model();expected=iso._prove_zero_manifold_uncached(doc,metadata,'Target')
    before=ET.tostring(doc),deepcopy(metadata);calls=tracked(monkeypatch)
    first=iso.prove_zero_manifold(doc,metadata,'Target');assert first==expected
    first['zero_monitor_paths'].append('caller mutation')
    second=iso.prove_zero_manifold(doc,metadata,'Target');assert second==expected
    second['native_parameter_requirements'].clear()
    assert iso.prove_zero_manifold(doc,metadata,'Target')==expected
    assert len(calls)==1 and (ET.tostring(doc),metadata)==before


@pytest.mark.parametrize('change',['document_outside_model','formula','metadata','target','implementation','sympy'])
def test_every_semantic_and_producer_key_change_invalidates_hit(monkeypatch,change):
    doc,metadata=old.model();calls=tracked(monkeypatch)
    iso.prove_zero_manifold(doc,metadata,'Target');target='Target'
    if change=='document_outside_model': ET.SubElement(doc,'AdditionalEnvelope',identity='changed')
    elif change=='formula': doc.find('./Model/FormulaCache/Formulas/Formula').set('audit','changed')
    elif change=='metadata': metadata['additional_exact_field']='changed'
    elif change=='target': target='Other'
    elif change=='implementation': monkeypatch.setattr(iso,'_proof_implementation_hashes',lambda:(('producer','changed'),))
    elif change=='sympy': monkeypatch.setattr(iso.sy,'__version__',iso.sy.__version__+'-changed')
    try: iso.prove_zero_manifold(doc,metadata,target)
    except ValueError:
        assert change=='target'
    assert len(calls)==2


def test_explicit_racemate_target_key_cannot_collide_with_single_parent():
    doc,metadata=old.model()
    assert iso._zero_proof_cache_key(doc,metadata,'Esomeprazole') != iso._zero_proof_cache_key(doc,metadata,['Esomeprazole','R-omeprazole'])


def test_failed_proofs_are_never_cached(monkeypatch):
    doc,metadata=old.model();calls=tracked(monkeypatch)
    iso.association.ModelIndex(doc).paths['Body|Other'].set('value','1')
    for _ in range(2):
        with pytest.raises(ValueError):iso.prove_zero_manifold(doc,metadata,'Target')
    assert len(calls)==2 and len(iso._ZERO_PROOF_CACHE)==0 and iso._ZERO_PROOF_CACHE_BYTES==0


def test_unverified_results_are_never_cached(monkeypatch):
    doc,metadata=old.model();calls=[]
    def run(*args):calls.append(True);return {'verified':False}
    monkeypatch.setattr(iso,'_prove_zero_manifold_uncached',run)
    for _ in range(2):assert iso.prove_zero_manifold(doc,metadata,'Target')=={'verified':False}
    assert len(calls)==2 and not iso._ZERO_PROOF_CACHE


def test_content_change_during_proof_is_not_cached(monkeypatch):
    doc,metadata=old.model();original=iso._prove_zero_manifold_uncached
    def run(*args):
        result=original(*args);doc.set('changed_during_proof','yes');return result
    monkeypatch.setattr(iso,'_prove_zero_manifold_uncached',run)
    assert iso.prove_zero_manifold(doc,metadata,'Target')['verified']
    assert not iso._ZERO_PROOF_CACHE


def test_lru_entry_bound_evicts_oldest_but_keeps_recent_exact_values(monkeypatch):
    monkeypatch.setattr(iso,'_ZERO_PROOF_CACHE_MAX_ENTRIES',2)
    doc,metadata=old.model();calls=tracked(monkeypatch)
    for version in (1,2,1,3,2):iso.prove_zero_manifold(doc,{**metadata,'version':version},'Target')
    assert len(calls)==4 and len(iso._ZERO_PROOF_CACHE)==2
    assert iso._ZERO_PROOF_CACHE_BYTES==sum(entry[1] for entry in iso._ZERO_PROOF_CACHE.values())


def test_size_bound_skips_large_success_without_changing_result(monkeypatch):
    monkeypatch.setattr(iso,'_ZERO_PROOF_CACHE_MAX_BYTES',1)
    doc,metadata=old.model();calls=tracked(monkeypatch)
    first=iso.prove_zero_manifold(doc,metadata,'Target');second=iso.prove_zero_manifold(doc,metadata,'Target')
    assert first==second and len(calls)==2 and not iso._ZERO_PROOF_CACHE


def test_outer_source_and_policy_verification_remain_outside_cache(monkeypatch):
    calls=[]
    def prepare(*args,**kwargs):calls.append((args,kwargs));return policy
    monkeypatch.setattr(iso,'prepare_source_isolation',prepare)
    policy={'source_reference':{'path':'source','sha256':'sha','simulation_id':'sim','source_blob_sha256':'blob'},'target':'Target'}
    for _ in range(2):assert iso.verify_source_isolation(policy,{}, {},project_root=ROOT) is policy
    assert len(calls)==2
