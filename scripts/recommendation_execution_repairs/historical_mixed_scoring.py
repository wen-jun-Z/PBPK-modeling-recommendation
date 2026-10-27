"""Scoring-only revalidation of authenticated historical mixed expansion.

The candidate producer and original receipts stay unchanged. Only the registered
native execution wrapper's resource-limit revision may differ; every source,
expanded model, policy, and native metadata proof is recomputed independently.
"""
from copy import deepcopy
import json

from . import mixed_application_expansion as mixed
from . import coadministration_isolation as isolation
from . import source_pkml
from .retained_implementations import binding_profile

base=mixed.base
inside=mixed.inside
sha=mixed.sha
VERSION=mixed.VERSION
ALLOWED_RECEIPT_DRIFT=frozenset({"scripts/recommendation_execution_repairs/native_pkml.py"})


def _require(condition,message):
    if not condition:
        raise ValueError("Historical mixed scoring: "+message)


def _authenticate(f,plan,receipt,expected_policy):
    protocol=f.read(f.root/"experiments/model_recommendation/execution_validation_20261001/protocol.json")
    frozen={}
    for ref in protocol["code"]:
        name=str(f.checked(ref).relative_to(f.root))
        _require(name not in frozen,"duplicate current producer")
        frozen[name]=ref["sha256"]
    route=plan.get("adapter_route")
    if route=="complete_original_pksim5":
        from . import complete_candidate as adapter
        adapter_path="scripts/recommendation_execution_repairs/complete_candidate.py"
    elif route=="verified_json_native_pksim5":
        from . import json_native_candidate as adapter
        adapter_path="scripts/recommendation_execution_repairs/json_native_candidate.py"
    else:
        raise ValueError("Historical mixed scoring: unsupported source adapter")
    code,required=binding_profile(plan,frozen,adapter_path=adapter_path,
        current_required=adapter.IMPLEMENTATION_FILES,project_root=f.root)
    implementation=receipt.get("implementation_sha256")
    producer_set=set(mixed.IMPLEMENTATION_FILES)
    _require(isinstance(implementation,dict) and set(implementation)==producer_set,
             "receipt producer dependency set differs")
    _require(producer_set.issubset(required),"receipt producer is outside authenticated adapter")
    _require(all(h==plan["implementation_sha256"].get(p)==code.get(p)
                 for p,h in implementation.items()),"receipt producer is not bound to original plan")
    changed={p for p,h in implementation.items() if frozen.get(p)!=h}
    _require(changed.issubset(ALLOWED_RECEIPT_DRIFT),"unsupported historical mixed producer drift")
    _require(expected_policy==plan.get("application_expansion"),"expansion policy is not bound to original plan")
    source={"path":plan.get("execution_source_path",plan["source_path"]),
            "sha256":plan.get("execution_source_sha256",plan["source_sha256"]),
            "simulation_id":plan["source_simulation_id"],
            "source_blob_sha256":plan["source_model_blob_sha256"]}
    _require(receipt.get("source_reference")==source,"receipt source differs from original plan")
    f.checked({"path":source["path"],"sha256":source["sha256"]})
    for name in ("ode_executed","fitting_performed","sealed_observations_accessed",
                 "original_applications_modified","scientific_applicability_claimed"):
        _require(receipt.get(name) is False,"original expansion outcome boundary differs")
    for ref in (receipt["source_pkml"],receipt["pkml"]):f.checked(ref)
    for path,digest in receipt["files"].items():f.checked({"path":path,"sha256":digest})


def verify_expansion_receipt(f,plan,receipt_path,*,expected_policy=None):
    root=f.root;path=f.path(receipt_path);receipt=f.read(path)
    expected_policy=plan.get('application_expansion') if expected_policy is None else expected_policy
    _authenticate(f,plan,receipt,expected_policy)
    _require(receipt.get('version')==VERSION and receipt.get('status')=='expanded_pending_native_readback_and_ode','RECEIPT_VERSION_OR_STATUS_DIFFERS')
    for ref in (receipt['source_pkml'],receipt['pkml']):_require(sha(inside(ref['path'],root))==ref['sha256'],'PKML_HASH_CHANGED')
    for name,digest in receipt['files'].items():_require(sha(inside(name,root))==digest,'EVIDENCE_HASH_CHANGED')
    for name in ('policy.json','expansion_plan.json','semantic_audit.json','source_audit.json'):
        _require(str((path.parent/name).relative_to(root)) in receipt['files'],'MISSING_REQUIRED_EVIDENCE')
    policy=json.loads((path.parent/'policy.json').read_text());ref=receipt['source_reference']
    if expected_policy is not None:_require(policy==expected_policy,'EXPECTED_POLICY_DIFFERS')
    saved=base._original_document(ref,root)
    fresh=mixed.prepare_source_expansion(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
        source_pkml.describe_saved_simulation(saved),policy['target'],policy['contract'],project_root=root)
    _require(policy==fresh,'SOURCE_POLICY_CHANGED')
    original=base._read_xml(inside(receipt['source_pkml']['path'],root));expanded=base._read_xml(inside(receipt['pkml']['path'],root))
    source_audit=base._verify_original_source(original,ref,root)
    _require(source_audit==json.loads((path.parent/'source_audit.json').read_text()),'SOURCE_PROOF_CHANGED')
    plan=json.loads((path.parent/'expansion_plan.json').read_text())
    _require(plan['target']==policy['target'] and plan['contract']==policy['contract'],'PLAN_POLICY_DIFFERS')
    proof=mixed.audit_expansion(original,expanded,plan)
    _require(proof['accepted'] and proof==json.loads((path.parent/'semantic_audit.json').read_text()),'FULL_XML_PROOF_CHANGED')
    return {'accepted':True,'receipt':{'path':str(path.relative_to(root)),'sha256':sha(path)},
            'source_model_and_settings_verified':True,'full_expansion_xml_verified':True,
            'source_target_event_count':policy['source_target_event_count'],
            'executable_target_event_count':policy['executable_target_event_count'],
            'executable_total_event_count':policy['executable_total_event_count'],'scientific_applicability_claimed':False}


def executable_metadata(f,plan,receipt_path,original_source_metadata,*,expected_policy):
    root=f.root
    verification=verify_expansion_receipt(f,plan,receipt_path,expected_policy=expected_policy)
    receipt=json.loads(inside(receipt_path,root).read_text());saved=base._original_document(receipt['source_reference'],root)
    _require(source_pkml.describe_saved_simulation(saved)==original_source_metadata,'ORIGINAL_METADATA_CHANGED')
    expanded=base._read_xml(inside(receipt['pkml']['path'],root));saved.remove(saved.find('Model'))
    saved.append(deepcopy(base._models(expanded)));actual=source_pkml.describe_saved_simulation(saved)
    _require(len(actual['administrations'])==expected_policy['executable_total_event_count'],'EXECUTABLE_APPLICATION_COUNT_DIFFERS')
    _require(sum(row.get('administered_compounds')==[expected_policy['target']] for row in actual['administrations'])
             ==expected_policy['executable_target_event_count'],'EXECUTABLE_TARGET_COUNT_DIFFERS')
    actual.update(metadata_scope='actual_expanded_executable_model_only',
                  original_source_metadata_sha256=base._digest(original_source_metadata),
                  expansion_receipt=verification['receipt'],expanded_pkml=receipt['pkml'])
    return actual,verification


def expanded_isolation(f,plan,policy,receipt_path,metadata,contract):
    """Reprove zero invariance on the independently audited expanded model."""
    root=f.root;ref=policy['source_reference']
    verification=verify_expansion_receipt(f,plan,receipt_path)
    source,document=isolation.association._load(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],root)
    receipt=json.loads(source_pkml.inside(receipt_path,root).read_text())
    _require(receipt['source_reference']==ref,'EXPANSION_ORIGINAL_SOURCE_DIFFERS')
    original_document=deepcopy(document)
    expanded=mixed.base._read_xml(source_pkml.inside(receipt['pkml']['path'],root))
    document.remove(document.find('Model'));document.append(deepcopy(mixed.base._models(expanded)))
    raw=source_pkml.describe_saved_simulation(document)
    _require(all(metadata.get(k)==v for k,v in raw.items()),'EXPANSION_METADATA_DIFFERS')
    enriched,_=isolation.association._enrich_document(document,original_document=original_document)
    actual={**metadata,**enriched}
    result=isolation._document_policy(source,ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
        document,actual,policy['target'],contract,root,original_document=original_document)
    result.update(model_origin='audited_expanded_executable_model',
        original_isolation_policy_sha256=isolation._digest(policy),mixed_expansion_receipt=verification['receipt'])
    return result,actual

