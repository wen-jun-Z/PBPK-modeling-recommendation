"""Prove absent target routes in four complete, fixed original JSON sources.

Does not waive multi-drug, process-binding, identity, regimen or parser errors.
Both original and verified migration are inspected without native execution.
"""
from pathlib import Path
import json
from .native_pkml import inside, sha
from recommendation_candidate_executor import _name

ROOT=Path(__file__).resolve().parents[2]
REGISTRY={'path':'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_source_gate_followup_20261002/json_route_sources.json',
          'sha256':'30c46247355116b2a84629f4022a5e6a92fc7e2401e1ed58d8c361e058df39e6'}
ROUTES={'Oral':'Oral','Intravenous':'Intravenous','IntravenousBolus':'Intravenous'}


def checked(ref,root):
    path=inside(ref['path'],root)
    if sha(path)!=ref['sha256']:raise ValueError('JSON_ROUTE_EVIDENCE_CHECKSUM_MISMATCH:'+ref['path'])
    return path


def definitions(root):
    registry=json.loads(checked(REGISTRY,root).read_text())
    if (registry.get('clinical_outcomes_accessed') is not False or registry.get('native_execution_started') is not False):
        raise ValueError('OUTCOMES_FREE_JSON_ROUTE_REGISTRY_REQUIRED')
    return registry['source_definitions']


def _indexed(rows,label):
    if (not isinstance(rows,list) or not rows or
            any(not isinstance(r,dict) or not isinstance(r.get('Name'),str) or not r['Name'] for r in rows)):
        raise ValueError('COMPLETE_NAMED_JSON_INVENTORY_REQUIRED:'+label)
    result={r['Name']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('DUPLICATE_JSON_SOURCE_IDENTITY:'+label)
    return result


def _routes(protocol):
    if protocol.get('ApplicationType'):
        if protocol.get('Schemas'):raise ValueError('AMBIGUOUS_SIMPLE_AND_COMPLEX_PROTOCOL')
        items=[protocol]
    else:
        schemas=protocol.get('Schemas')
        if not isinstance(schemas,list) or not schemas:raise ValueError('EXPLICIT_PROTOCOL_SCHEMAS_REQUIRED')
        items=[]
        for schema in schemas:
            row=schema.get('SchemaItems')
            if not isinstance(row,list) or not row:raise ValueError('EXPLICIT_PROTOCOL_ITEMS_REQUIRED')
            items.extend(row)
    values=[item.get('ApplicationType') for item in items]
    if not values or any(value not in ROUTES for value in values):raise ValueError('UNRESOLVED_NATIVE_APPLICATION_TYPE')
    return sorted({ROUTES[value] for value in values})


def route_inventory(document,compound):
    """Inspect all original simulations, including populations and mixed drugs."""
    if document.get('Version') not in {76,77,78,79,80}:raise ValueError('UNSUPPORTED_JSON_SOURCE_SCHEMA')
    compounds=_indexed(document.get('Compounds'),'Compounds')
    protocols=_indexed(document.get('Protocols'),'Protocols')
    simulations=_indexed(document.get('Simulations'),'Simulations')
    target=[name for name in compounds if _name(name)==compound]
    if len(target)!=1:raise ValueError('EXACT_PARENT_IDENTITY_MUST_BE_ESTABLISHED_BEFORE_ROUTE_ABSENCE')
    target=target[0]
    # Event-driven or custom administration needs an explicit event proof; do
    # not infer absence solely from the normal Protocol field in such models.
    if document.get('Events'):raise ValueError('CUSTOM_SOURCE_EVENTS_REQUIRE_SEPARATE_ROUTE_AUDIT')
    facts=[];target_count=0
    for name,simulation in sorted(simulations.items()):
        if simulation.get('Events'):raise ValueError('CUSTOM_SIMULATION_EVENTS_REQUIRE_SEPARATE_ROUTE_AUDIT')
        selections=_indexed(simulation.get('Compounds'),'Simulation.Compounds')
        if not set(selections).issubset(compounds):raise ValueError('UNBOUND_SELECTED_COMPOUND')
        administrators=[];target_routes=[]
        for selected_name,selection in sorted(selections.items()):
            reference=selection.get('Protocol')
            if not reference:continue
            if not isinstance(reference,dict) or reference.get('Name') not in protocols:
                raise ValueError('UNBOUND_OR_AMBIGUOUS_PROTOCOL')
            routes=_routes(protocols[reference['Name']])
            administrators.append({'compound':selected_name,'protocol':reference['Name'],'routes':routes})
            if selected_name==target:
                target_count+=1;target_routes.extend(routes)
        for parameter in simulation.get('Parameters',[]):
            path=parameter.get('Path','')
            if path.split('|')[-1] in {'ApplicationType','Route'}:
                raise ValueError('SIMULATION_ROUTE_OVERRIDE_REQUIRES_SEPARATE_AUDIT')
        facts.append({'simulation':name,'selected_compounds':sorted(selections),
                      'all_administrations':administrators,'target_routes':sorted(set(target_routes))})
    if not target_count:raise ValueError('PARENT_NOT_ADMINISTERED_IDENTITY_OR_ADAPTER_REPAIR_REQUIRED')
    return facts


def known_json_route_refs(*,project_root=ROOT):
    root=Path(project_root).resolve();refs=[REGISTRY]
    for record in definitions(root).values():refs.extend(record[key] for key in ('source','snapshot','receipt'))
    for ref in refs:checked(ref,root)
    return refs


def prove_json_route_absence(plan,scenario,candidate,*,project_root=ROOT):
    root=Path(project_root).resolve();prov=candidate['catalog_provenance']
    record=definitions(root).get(prov['source_sha256'])
    if not record:raise ValueError('JSON_SOURCE_ROUTE_ABSENCE_NOT_REGISTERED')
    pair={'scenario_id':scenario['scenario_id'],'candidate_id':candidate['candidate_id']}
    if (pair not in record['pairs'] or {k:plan.get(k) for k in pair}!=pair or
            plan.get('reasons')!=['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT']):
        raise ValueError('JSON_ROUTE_ABSENCE_FROZEN_PAIR_REQUIRED')
    if (plan.get('contract',{}).get('compound')!=record['compound'] or
            plan['contract'].get('route')!=record['absent_route'] or _name(scenario.get('compound'))!=record['compound']):
        raise ValueError('JSON_ROUTE_ABSENCE_FROZEN_TARGET_DIFFERS')
    aliases={'oral':'Oral','po':'Oral','intravenous':'Intravenous','iv':'Intravenous'}
    n=scenario['normalized_design'];declared=[n.get('route'),n.get('source_route'),n.get('dose',{}).get('route')]
    declared=[r for r in declared if r is not None]
    if not declared or any(aliases.get(_name(r))!=record['absent_route'] for r in declared):
        raise ValueError('JSON_ROUTE_ABSENCE_CLINICAL_PROTOCOL_DIFFERS')
    if {'path':prov['source_path'],'sha256':prov['source_sha256']}!=record['source']:
        raise ValueError('JSON_ROUTE_ORIGINAL_SOURCE_DIFFERS')
    migration=plan.get('source_migration',{})
    for key,path_key,sha_key in [('snapshot','path','sha256'),('receipt','receipt_path','receipt_sha256'),('source','original_path','original_sha256')]:
        if (inside(migration.get(path_key,''),root)!=checked(record[key],root) or
                migration.get(sha_key)!=record[key]['sha256']):
            raise ValueError('JSON_ROUTE_FROZEN_MIGRATION_DIFFERS')
    from .snapshot_repair import _verify_receipt
    receipt=_verify_receipt(checked(record['receipt'],root),prov,root,require_accepted=True)
    if receipt.get('snapshot')!=record['snapshot']:
        raise ValueError('JSON_ROUTE_RECEIPT_SNAPSHOT_BINDING_DIFFERS')
    original=json.loads(checked(record['source'],root).read_text())
    migrated=json.loads(checked(record['snapshot'],root).read_text())
    facts=route_inventory(original,record['compound'])
    if route_inventory(migrated,record['compound'])!=facts:
        raise ValueError('JSON_ROUTE_ORIGINAL_AND_MIGRATED_INVENTORIES_DIFFER')
    excluded=plan.get('excluded_source_simulations',[])
    if (len(excluded)!=len(facts) or {r.get('simulation') for r in excluded}!={f['simulation'] for f in facts}):
        raise ValueError('JSON_ROUTE_PLAN_OMITS_ORIGINAL_SIMULATIONS')
    if any(record['absent_route'] in f['target_routes'] for f in facts):
        raise ValueError('MATCHING_ORIGINAL_ROUTE_EXISTS_ADAPTER_MUST_BE_REPAIRED')
    refs=[REGISTRY,record['source'],record['snapshot'],record['receipt']]
    for ref in refs:checked(ref,root)
    return refs,[{**f,'reason':'EXACT_PARENT_TARGET_ROUTE_ABSENT_IN_COMPLETE_ORIGINAL_JSON',
                  'scientific_applicability':'unknown'} for f in facts]
