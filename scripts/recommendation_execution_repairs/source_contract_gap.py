"""Three exact source/request gaps, never source-wide failure or negative labels."""
from pathlib import Path
import json

import recommendation_candidate_migration as migration
from recommendation_candidate_executor import _digest

ROOT=migration.ROOT
REGISTRY={'path': 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_bridge_followup_20261002/source_contract_gap_v2_stage/source_contract_gap_registry.json', 'sha256': '22fe8d37ed09b36f7b56167b61f8919e5d73d652a52564c4e2d40468292ca24b'}
SCENARIO='clinical:itraconazole_pmc7526808_15day_sporanox200mg_bid_fed'
SOURCES={
    'e8ab2901-6c4d-549c-b983-b6a2ae29fff2':'7df4ce1ac3793bf0e2c911ee959a23de5c8b8564a4b4c7912c34e5de3e0b26b7',
    'befe8812-16bb-5cd1-a76c-bd93330f2dfd':'6fe6ad17554185b4014e0f7c9ec1efa87f1c36715be4b2fa3353a3f8308f725c',
    '3df9ee3a-afb1-5c3b-a652-dfa3ff9c7390':'6b538bbe36e9104c7a7af0df6b9807b6d8ef9f46dd2f305b374bb0250f5c3a16',
}


def _require(ok,message):
    if not ok:raise ValueError('SOURCE_CONTRACT_GAP_'+message)


def _checked(ref,root):
    path=migration.checked(ref,root)
    _require(not any(p.casefold().startswith('sealed') for p in path.relative_to(root).parts),'SEALED_RESOURCE_FORBIDDEN')
    return path


def _read(ref,root):return json.loads(_checked(ref,root).read_text())


def registered_pair(plan):
    return plan.get('scenario_id')==SCENARIO and plan.get('candidate_id') in SOURCES


def _registry(root):
    reg=_read(REGISTRY,root)
    _require(reg.get('schema_version')=='exact-source-contract-gap-v1'
             and reg.get('clinical_outcomes_accessed') is False
             and reg.get('only_original_source_configuration_asserted') is True
             and reg.get('source_clinical_formulation_not_inferred') is True,'REGISTRY_SCOPE_DIFFERS')
    cases=reg['cases']
    _require(len(cases)==3 and {(e['scenario_id'],e['candidate_id'],e['source_original']['sha256']) for e in cases}
             =={(SCENARIO,c,s) for c,s in SOURCES.items()},'EXACT_PAIR_SOURCE_SET_REQUIRED')
    return reg


def _frozen_pair(entry,reg,root):
    frozen={}
    for key,ref in reg['frozen_inputs'].items():
        frozen[key]=[json.loads(line) for line in _checked(ref,root).read_text().splitlines() if line.strip()]
    scenarios=[s for s in frozen['scenarios'] if s['scenario_id']==entry['scenario_id']]
    candidates=[c for c in frozen['candidates'] if (c['scenario_id'],c['candidate_id'])==
                (entry['scenario_id'],entry['candidate_id']) and c.get('identity_match') is True]
    _require(len(scenarios)==len(candidates)==1,'UNIQUE_FROZEN_EXACT_PARENT_PAIR_REQUIRED')
    scenario,candidate=scenarios[0],candidates[0]
    _require(_digest(scenario)==entry['scenario_sha256'] and _digest(candidate)==entry['candidate_sha256'],
             'FROZEN_CONTEXT_DIFFERS')
    _checked(scenario['source_protocol'],root)
    from .clinical_contract import scenario_contract
    _require(scenario_contract(scenario,project_root=root,candidate=candidate)==entry['expected_contract'],
             'FROZEN_TARGET_CONTRACT_DIFFERS')
    prov=candidate['catalog_provenance']
    _require({'path':prov['source_path'],'sha256':prov['source_sha256']}==entry['source_original'],
             'ORIGINAL_CANDIDATE_SOURCE_DIFFERS')
    return scenario,candidate


def _original_facts(document,entry):
    """Inspect every original simulation, not only the failed selected branch."""
    _require(document.get('Version') in {76,77,78,79,80},'ORIGINAL_SNAPSHOT_VERSION_DIFFERS')
    simulations=document.get('Simulations',[])
    _require(simulations and len({s['Name'] for s in simulations})==len(simulations),'COMPLETE_UNIQUE_SOURCE_SIMULATIONS_REQUIRED')
    target=[s for s in simulations if any(c.get('Name')=='Itraconazole' for c in s.get('Compounds',[]))]
    _require(len(target)==1 and target[0]['Name']==entry['source_simulation'],'OTHER_TARGET_SIMULATION_REQUIRES_REVIEW')
    selections=[c for c in target[0]['Compounds'] if c.get('Name')=='Itraconazole']
    _require(selections==[entry['source_compound_binding']],'SOURCE_SELECTION_CHANGED')
    selection=selections[0]
    _require([a for a in selection.get('Alternatives',[]) if a.get('GroupName')=='COMPOUND_SOLUBILITY']==
             [{'AlternativeName':'Solution fasted (Taupitz et al. 2013)','GroupName':'COMPOUND_SOLUBILITY'}],
             'SOURCE_SOLUBILITY_SELECTION_DIFFERS')
    _require(selection.get('Protocol',{}).get('Formulations')==[{'Name':'Solution','Key':'Formulation'}],
             'SOURCE_RELEASE_SELECTION_DIFFERS')
    protocols=[p for p in document.get('Protocols',[]) if p['Name']==selection['Protocol']['Name']]
    _require(protocols==[entry['source_protocol']],'SOURCE_PROTOCOL_CHANGED')
    formulations=[f for f in document.get('Formulations',[]) if f.get('Name')=='Solution']
    _require(formulations==entry['source_formulations']==[{'Name':'Solution','FormulationType':'Formulation_Dissolved'}],
             'SOURCE_IS_NOT_EXACT_DISSOLVED_FORMULATION')
    compounds=[c for c in document.get('Compounds',[]) if c.get('Name')=='Itraconazole']
    _require(len(compounds)==1,'SOURCE_PARENT_AMBIGUOUS')
    sol=[a for a in compounds[0]['Solubility'] if a['Name']=='Solution fasted (Taupitz et al. 2013)']
    _require(len(sol)==1 and [(p.get('Value'),p.get('Unit')) for p in sol[0]['Parameters']
             if p['Name']=='Solubility at reference pH']==[(8.0,'mg/l')],'SOURCE_FASTED_SOLUBILITY_DIFFERS')
    if entry['candidate_id']=='3df9ee3a-afb1-5c3b-a652-dfa3ff9c7390':
        _require([s['Name'] for s in simulations]==['Prueksaritanont 2017 - DDI Control',
                 'Prueksaritanont 2017 - DDI Itraconazole']
                 and document.get('Formulations')==entry['source_formulations'],
                 'COMPLETE_THIRD_SOURCE_CONFIGURATION_DIFFERS')
    return {'all_original_simulations':[s['Name'] for s in simulations],
        'all_parent_containing_simulations':[s['Name'] for s in target],
        'original_release':'Formulation_Dissolved','original_solubility_scenario':'Solution fasted (Taupitz et al. 2013)',
        'original_solubility_mg_per_l':8.0,'source_clinical_formulation_inferred':False,
        'matching_capsule_fed_source_simulations':[]}


def _native_facts(entry,root):
    from . import json_native_source_proof as bridge,source_pkml,mixed_application_expansion as mixed
    source=entry['source_original'];native=entry['native_source']
    proven=bridge.resolve_native_sources({'source_path':source['path'],'source_sha256':source['sha256']},project_root=root)
    matches=[c for c in proven['candidates'] if c['native_source']=={k:native[k] for k in ('path','sha256')}]
    _require(len(matches)==1,'SOURCE_AUTHENTICATED_NATIVE_PROJECT_REQUIRED')
    selected=[s for s in matches[0]['allowed_simulations'] if s['id']==native['simulation_id']]
    _require(len(selected)==1 and selected[0]['name']==entry['source_simulation']
             and selected[0]['source_blob_sha256']==native['source_blob_sha256'],'SOURCE_NATIVE_SIMULATION_IDENTITY_DIFFERS')
    saved=mixed.base._original_document(native,root)
    document,_=source_pkml.wrap_preserved_model(saved)
    model=mixed.base._models(document);paths,ids,parents=mixed.base._inventory(model)
    branches=[(path,node) for path,node in paths.items() if node.tag=='EventGroup'
        and node.find("Children/Container[@name='ProtocolSchemaItem']") is not None
        and any(n.get('name')=='Itraconazole' for n in node.iter('MoleculeAmount'))]
    third=entry['candidate_id']=='3df9ee3a-afb1-5c3b-a652-dfa3ff9c7390'
    _require(len(branches)==(5 if third else 4),'EXACT_ORIGINAL_ADMINISTRATIONS_REQUIRED')
    expected_destinations={model.find('Root').get('name')+'|'+s for s in (
        'Itraconazole|Total oral drug mass','Organism|Lumen|OralApplicationsEnabled',
        'Organism|Lumen|Stomach|Liquid','Organism|Lumen|Stomach|Itraconazole')}
    events=[]
    for path,node in branches:
        event=list(node.iter('Event'));assignments=list(node.iter('EventAssignment'))
        _require('|Solution|Application_' in path and not list(node.iter('Transport'))
                 and len(event)==1 and event[0].get('name')=='Application_StartEvent'
                 and event[0].get('oneTime')=='1' and len(assignments)==4
                 and {a.get('objectPath') for a in assignments}==expected_destinations,
                 'ORIGINAL_NATIVE_SOLUTION_TOPOLOGY_DIFFERS')
        _require(all(a.get('changedEntity') in ids and paths[a.get('objectPath')].get('id')==a.get('changedEntity')
                     for a in assignments),'NATIVE_ASSIGNMENT_TARGET_IDENTITY_DIFFERS')
        events.append({'path':path,'source_node_sha256':source_pkml.element_sha(node),
                       'release_transport_count':0,'start_events':1,'assignment_count':4})
    if third:
        _require(len(entry.get('expected_native_release_events',[]))==5,'REGISTERED_FIVE_EVENT_PROOF_REQUIRED')
        _,formulas=mixed.base._formula_inventory(model)
        actual=[]
        for path,node in branches:
            actual.append({'path':path,'source_node_sha256':source_pkml.element_sha(node),
                'release_transport_count':len(list(node.iter('Transport'))),
                'start_events':[n.attrib for n in node.iter('Event')],
                'assignments':[n.attrib for n in node.iter('EventAssignment')],
                'native_dose_kg':mixed.base._scalar(paths[path+'|ProtocolSchemaItem|Dose'],formulas),
                'native_start_time_min':mixed.base._scalar(paths[path+'|ProtocolSchemaItem|Start time'],formulas),
                'tablet_active':any(n.get('name')=='TabletIsActive' for n in node.iter())})
        _require(actual==entry['expected_native_release_events']
                 and [r['native_start_time_min'] for r in actual]==[0.,1440.,2880.,4320.,5760.]
                 and all(r['native_dose_kg']==0.00019999999999999998 and r['tablet_active'] is False for r in actual),
                 'EXACT_FIVE_SOLUTION_EVENTS_CHANGED')
    return events,matches[0]['proof_refs']


def _evidence(entry,reg,root):
    scenario,candidate=_frozen_pair(entry,reg,root)
    source=_read(entry['source_original'],root);facts=_original_facts(source,entry)
    events,proof_refs=_native_facts(entry,root)
    refs=[REGISTRY,reg['source_audit'],*reg['author_documents'],*reg['frozen_inputs'].values(),
          scenario['source_protocol'],entry['source_original'],*proof_refs]
    if entry.get('additional_source_audit'):
        refs.append(entry['additional_source_audit'])
    unique={}
    for ref in refs:
        _checked(ref,root)
        _require(ref['path'] not in unique or unique[ref['path']]==ref,'CONFLICTING_EVIDENCE_REFERENCES')
        unique[ref['path']]=ref
    return scenario,candidate,{**facts,'native_release_events':events},[unique[k] for k in sorted(unique)]


def known_contract_gap_refs(*,project_root=ROOT):
    root=Path(project_root).resolve();reg=_registry(root);refs={}
    for entry in reg['cases']:
        _,_,_,found=_evidence(entry,reg,root)
        for ref in found:
            _require(ref['path'] not in refs or refs[ref['path']]==ref,'CONFLICTING_EVIDENCE_REFERENCES')
            refs[ref['path']]=ref
    return [refs[k] for k in sorted(refs)]


def assess_plan_gap(plan,scenario,candidate,*,project_root=ROOT):
    result={'accepted':False,'category':'engineering_repair_required','scientific_applicability':'unknown',
            'verified_references':[],'reasons':[]}
    try:
        _require(registered_pair(plan),'EXACT_REGISTERED_PAIR_REQUIRED')
        root=Path(project_root).resolve();reg=_registry(root)
        entry=next(e for e in reg['cases'] if e['candidate_id']==plan['candidate_id'])
        _require(plan.get('status')=='blocked' and plan.get('reasons')==entry['original_blocked_reasons']
                 and plan.get('sealed_observations_accessed') is False and plan.get('fitting_performed') is False,
                 'ORIGINAL_OUTCOMES_FREE_BLOCKER_REQUIRED')
        _require(plan.get('scenario_sha256')==entry['scenario_sha256']
                 and plan.get('candidate_sha256')==entry['candidate_sha256']
                 and plan.get('contract')==entry['expected_contract'],'EXACT_FROZEN_PLAN_CONTRACT_REQUIRED')
        actual_scenario,actual_candidate,facts,refs=_evidence(entry,reg,root)
        _require(scenario==actual_scenario and candidate==actual_candidate,'CALLER_FROZEN_CONTEXT_DIFFERS')
        result.update(accepted=True,category='original_source_contract_gap',
            reasons=['original_solution_fasted_configuration_has_no_registered_capsule_fed_release_input'],
            verified_references=refs,source_facts=facts,original_engineering_reasons=list(plan['reasons']),
            original_source_retained_for_reference=True,source_execution_verified=False,numerical_prediction_available=False,
            matched_prediction_eligible=False,direct_reuse_eligible=False,scientific_failure_inferred=False,
            inclusion_in_full_candidate_denominator=True)
    except (ValueError,KeyError,TypeError,OSError,StopIteration) as exc:
        result['reasons']=[type(exc).__name__+':'+str(exc)]
    return result
