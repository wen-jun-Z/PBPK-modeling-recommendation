"""The catalog JSON and native execution model must keep separate exact bindings."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
SPEC=importlib.util.spec_from_file_location('json_bridge_scoring',STAGED/'scripts/recommendation_execution_repairs/scoring.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)


@pytest.fixture
def chain(monkeypatch):
    import recommendation_execution_repairs as package
    root=Path('/synthetic');directory=root/'run/native/job'
    original={'path':'original.json','sha256':'a'*64};native={'path':'native.pksim5','sha256':'b'*64}
    proof={'source_original':original,'native_source':native,
           'allowed_simulations':[{'name':'CONTROL','id':'sim-id','source_blob_sha256':'c'*64}],
           'proof_refs':[original,native]}
    plan={'adapter_route':'verified_json_native_pksim5','source_path':original['path'],'source_sha256':original['sha256'],
          'execution_source_path':native['path'],'execution_source_sha256':native['sha256'],
          'json_native_source_proof':proof,'source_simulation':'CONTROL','source_simulation_id':'sim-id','source_model_blob_sha256':'c'*64}
    path=directory/'json_native_source_proof.json';docs={path:deepcopy(proof),directory.parent.parent/'protocol.json':{'inputs':deepcopy(proof['proof_refs'])}}
    class Files:
        unsealed=False
        def __init__(self):self.root=root
        def read(self,path):assert not self.unsealed;return docs[path]
        def checked(self,ref):assert not self.unsealed;return self.root/ref['path']
    module=types.ModuleType('recommendation_execution_repairs.json_native_source_proof')
    def verified(provenance,p,*,project_root):
        assert provenance=={'source_path':original['path'],'source_sha256':original['sha256']}
        return p
    module.verify_native_source=verified
    monkeypatch.setitem(sys.modules,module.__name__,module);monkeypatch.setattr(package,'json_native_source_proof',module,raising=False)
    return {'f':Files(),'directory':directory,'plan':plan,'proof':proof,'docs':docs,'global':{path},'prediction':{path},'native':native,'proof_path':path}


def check(p):return s.verified_execution_source(p['f'],p['directory'],p['plan'],p['global'],p['prediction'])


def test_catalog_and_native_sources_are_both_verified_but_not_relabeled(chain):
    p=chain;assert check(p)==p['native'];assert p['plan']['source_path']=='original.json'
    assert not p['f'].unsealed


@pytest.mark.parametrize('fault',['execution_source','selected_id','blob','name','global','prediction','proof_content','protocol_missing','protocol_hash','protocol_duplicate'])
def test_bridge_fails_closed_on_every_identity_or_lock_boundary(chain,fault):
    p=chain
    if fault=='execution_source':p['plan']['execution_source_sha256']='d'*64
    elif fault=='selected_id':p['plan']['source_simulation_id']='other'
    elif fault=='blob':p['plan']['source_model_blob_sha256']='d'*64
    elif fault=='name':p['plan']['source_simulation']='other'
    elif fault=='global':p['global'].clear()
    elif fault=='prediction':p['prediction'].clear()
    elif fault=='proof_content':p['docs'][p['proof_path']]['altered']=True
    else:
        inputs=p['docs'][p['directory'].parent.parent/'protocol.json']['inputs']
        if fault=='protocol_missing':inputs.pop()
        elif fault=='protocol_hash':inputs[-1]['sha256']='d'*64
        else:inputs.append(dict(inputs[-1]))
    with pytest.raises(ValueError):check(p)
    assert not p['f'].unsealed


def test_native_bridge_proof_cannot_claim_old_adapter_route(chain):
    p=chain;p['plan']['adapter_route']='complete_original_pksim5'
    with pytest.raises(ValueError,match='registered bridge'):check(p)


def test_explicit_bridge_dispatch_precedes_snapshot_and_complete_helpers(monkeypatch):
    import recommendation_execution_repairs as package
    spec=importlib.util.spec_from_file_location('bridge_retention',STAGED/'scripts/recommendation_execution_repairs/retained_implementations.py')
    retained=importlib.util.module_from_spec(spec);spec.loader.exec_module(retained)
    monkeypatch.setitem(sys.modules,'recommendation_execution_repairs.retained_implementations',retained)
    bridge=types.ModuleType('recommendation_execution_repairs.json_native_candidate')
    bridge.IMPLEMENTATION_FILES=tuple(set(s.snapshot_candidate.IMPLEMENTATION_FILES)|set(s.complete_candidate.IMPLEMENTATION_FILES)|{'scripts/recommendation_execution_repairs/json_native_candidate.py','scripts/recommendation_execution_repairs/json_native_source_proof.py'})
    monkeypatch.setitem(sys.modules,bridge.__name__,bridge);monkeypatch.setattr(package,'json_native_candidate',bridge,raising=False)
    code={p:'a'*64 for p in bridge.IMPLEMENTATION_FILES}
    record={'adapter_route':'verified_json_native_pksim5','implementation_sha256':code,'execution_revision':s.REVISION}
    with s.repaired_execution_profile():
        assert s.backend.verify_execution_binding(record,frozen_code_sha256=code,expected_revision=s.REVISION,label='bridge')==code
        wrong={**record,'adapter_route':'complete_original_pksim5'}
        with pytest.raises(ValueError,match='registered route'):s.backend.verify_execution_binding(wrong,frozen_code_sha256=code,expected_revision=s.REVISION,label='bridge')
        missing=deepcopy(record);missing['implementation_sha256'].pop('scripts/recommendation_execution_repairs/json_native_source_proof.py')
        with pytest.raises(ValueError,match='incomplete'):s.backend.verify_execution_binding(missing,frozen_code_sha256=code,expected_revision=s.REVISION,label='bridge')
