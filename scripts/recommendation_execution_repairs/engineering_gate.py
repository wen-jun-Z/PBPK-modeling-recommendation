"""Fail-closed separation of genuine unavailable cases from engineering faults.

Only frozen design/catalogue metadata and original model definitions are read.
Accepted unavailable cases remain unknown, never negative scientific labels.
"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_candidate_executor import _digest,_name
from . import license_audit,source_pkml,event_associations
from .native_pkml import inside,sha

FROZEN_INPUTS={
 'scenarios':{'path':'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl',
              'sha256':'c10062f38430c9874bd4d7043da7cc45b2cf6064f14e907e30cbd94187c60fe6'},
 'candidates':{'path':'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl',
               'sha256':'b7926e541c09ef722f9bb549ef03d6dcf9125ccdc8cee9924425226d9ccc86a9'}}
LICENSE_SHA='8e6aa362cd720d0d4666671b951e53baf9f5c58411319a53bf3d4ec58d33ef58'
SOURCE_ABSENCE_REASONS={
 'EXACT_UNIQUE_PARENT_IN_SOURCE_REQUIRED','EXACT_SINGLE_PARENT_COMPOUND_REQUIRED',
 'EXPLICIT_HUMAN_SOURCE_REQUIRED','HUMAN_SOURCE_REQUIRED','SOURCE_ROUTE_DIFFERS'}
VORICONAZOLE_ROUTE_CERTIFICATES={
 'b02b58603aeb75a2e0388fe754ab26f43eabea19a86c10e203fe48991f683ba8': {
  'path':'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/gate_followup_20261002/route_certificates/b02b58603aeb75a2_voriconazole_route_absence.json',
  'sha256':'6434c45b87557f0450ba871e58f89c06b8c7b8602b8e279373ee738310cb5a5b'},
 '57727c9a779c118fdba29486cb134723fdc829b4bdb466e8b6ce501349d24712': {
  'path':'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/gate_followup_20261002/route_certificates/57727c9a779c118f_voriconazole_route_absence.json',
  'sha256':'e21aa2389993b04aab5a2de0c3a5c0c139f2b8a8cf6df5846f55b6deec60b0c1'}}



def _checked(ref,root):
    path=inside(ref['path'],root)
    if 'sealed' in {part.lower() for part in path.parts}:raise ValueError('SEALED_PATH_FORBIDDEN')
    if sha(path)!=ref['sha256']:raise ValueError('EVIDENCE_CHECKSUM_MISMATCH:'+ref['path'])
    return path


def _context(plan,root):
    data={}
    for key,ref in FROZEN_INPUTS.items():
        data[key]=[json.loads(line) for line in _checked(ref,root).read_text().splitlines() if line.strip()]
    scenarios=[s for s in data['scenarios'] if s['scenario_id']==plan.get('scenario_id')]
    pairs=[p for p in data['candidates'] if (p['scenario_id'],p['candidate_id'])==
           (plan.get('scenario_id'),plan.get('candidate_id')) and p.get('identity_match') is True]
    if len(scenarios)!=1 or len(pairs)!=1:raise ValueError('FROZEN_EXACT_PARENT_PAIR_REQUIRED')
    scenario,candidate=scenarios[0],pairs[0]
    if plan.get('scenario_sha256')!=_digest(scenario) or plan.get('candidate_sha256')!=_digest(candidate):
        raise ValueError('PLAN_FROZEN_INPUT_BINDING_DIFFERS')
    _checked(scenario['source_protocol'],root)
    return scenario,candidate,[*FROZEN_INPUTS.values(),scenario['source_protocol']]


def _clinical_gap(scenario,reasons):
    """Prior, explicit curator findings; absence alone is insufficient."""
    case=scenario['case_id'];n=scenario['normalized_design'];p=scenario['protocol_fields']
    if case=='dabigatran_pmc11909909_ctr20231968_fasted_pradaxa150mg_reference':
        if reasons==['EXPLICIT_PARENT_ANALYTE_REQUIRED'] and (
            p.get('assay_qualification',{}).get('native_output_pool_mapping_frozen') is False and
            p.get('assay_qualification',{}).get('target_assay_hydrolysis_or_conjugate_handling_reported') is False and
            n.get('endpoint_contracts') and all(e.get('evaluable_from_source') is False for e in n['endpoint_contracts'])):
            return 'target_assay_and_endpoint_definition_unresolved'
    if case=='efavirenz_pmc6373243_pr105001c_day1_alone600mg':
        if reasons==['EXPLICIT_BOUNDED_SAMPLING_TIMES_REQUIRED'] and (
            n.get('sampling',{}).get('times_h') is None and
            p.get('sampling',{}).get('missing_nominal_grid_not_fabricated') is True and
            p.get('sampling',{}).get('source_sentence_ambiguity')):
            return 'target_sampling_grid_not_reported_and_horizon_analyte_ambiguous'
    if case=='felodipine_pmc5642591_monotherapy_er5mg_bid_day7':
        if reasons==['UNKNOWN_OBSERVATION_ORIGIN_REQUIRES_EXPLICIT_PROTOCOL'] and (
            n.get('regimen',{}).get('observation_origin_h') is None and
            p.get('dose',{}).get('explicit_total_dose_count') is None and
            p.get('protocol_execution_ambiguities')):
            return 'target_last_dose_origin_and_number_ambiguous'
    # These two original clinical protocols explicitly declare the missing
    # salt/hydration convention. The exact product-history review preserves
    # that ambiguity; it is not permission to invent a conversion factor.
    salt_cases = {
        'dextromethorphan_yamazaki2016_day1_30mg':
            'dextromethorphan hydrobromide; hydration state not specified',
        'paroxetine_elshafeey2021_market_tablet_25mg':
            'salt/hydration not stated in clinical protocol',
    }
    if case in salt_cases and len(reasons)==1 and reasons[0].split(':')[0]=='SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN':
        dose=n.get('dose',{})
        if (dose.get('salt')==salt_cases[case] and p.get('dose',{}).get('salt')==salt_cases[case]
                and dose.get('active_moiety_dose_mg') is None and not p.get('salt_conversion')):
            return 'target_product_salt_or_hydration_identity_not_reported; exact_dose_conversion_unresolved'
    return None


def _missing_joint_drug_model(scenario,candidate,root):
    """Verify an absent partner in the complete original candidate inventory.

    A missing original chemical model cannot be supplied by zeroing a dose or
    by combining two catalogue entries and calling them the original candidate.
    This guard does not waive coadministration adapter failures when both drugs
    actually exist in the source.
    """
    cases={'ethinylestradiol_pmc8416862_nct02845219_microgynon_alone_day8',
           'levonorgestrel_pmc8416862_nct02845219_microgynon_alone_day8'}
    if scenario['case_id'] not in cases:raise ValueError('JOINT_SOURCE_ABSENCE_CASE_NOT_REGISTERED')
    n=scenario['normalized_design'];p=scenario['protocol_fields']
    expected=[{'compound':'ethinylestradiol','dose_mg':0.03},{'compound':'levonorgestrel','dose_mg':0.15}]
    if any(value!=expected for value in (n.get('dose',{}).get('coadministered_drugs'),
                                        n.get('regimen',{}).get('coadministered_drugs'),
                                        p.get('dose',{}).get('coadministered_drugs'))):
        raise ValueError('EXACT_FROZEN_JOINT_DRUG_REGIMEN_REQUIRED')
    prov=candidate['catalog_provenance'];ref={'path':prov['source_path'],'sha256':prov['source_sha256']}
    source=_checked(ref,root)
    if source.suffix=='.json':
        document=json.loads(source.read_text())
        names=[r.get('Name') for r in document.get('Compounds',[])]
        simulations=document.get('Simulations',[])
        if (document.get('Version') not in {76,77,78,79,80} or not names or
                any(not isinstance(name,str) or not name for name in names) or len(set(names))!=len(names)
                or not simulations or any(not sim.get('Compounds') for sim in simulations)):
            raise ValueError('COMPLETE_EXPLICIT_SOURCE_COMPOUND_INVENTORY_REQUIRED')
        if any(sel.get('Name') not in names for sim in simulations for sel in sim['Compounds']):
            raise ValueError('SOURCE_SIMULATION_COMPOUND_DEPENDENCY_UNRESOLVED')
        facts=[{'simulation':sim['Name'],'declared_compounds':[sel['Name'] for sel in sim['Compounds']]}
               for sim in simulations]
    elif source.suffix=='.pksim5':
        simulations=source_pkml.source_simulation_metadata(source,ref['sha256'],project_root=root)['simulations']
        if not simulations or any(not sim.get('compounds') for sim in simulations):
            raise ValueError('COMPLETE_ORIGINAL_SAVED_COMPOUND_INVENTORY_REQUIRED')
        names=sorted({name for sim in simulations for name in sim['compounds']})
        facts=[{'simulation_id':sim['simulation_id'],'source_blob_sha256':sim['source_blob_sha256'],
                'declared_compounds':sim['compounds']} for sim in simulations]
    else:raise ValueError('UNSUPPORTED_ORIGINAL_JOINT_MODEL_FORMAT')
    normalized={_name(name) for name in names};required={r['compound'] for r in expected}
    if _name(scenario['compound']) not in normalized:raise ValueError('TARGET_PARENT_INVENTORY_UNRESOLVED')
    missing=sorted(required-normalized)
    if not missing:raise ValueError('JOINT_DRUGS_EXIST_ADAPTER_MUST_BE_REPAIRED')
    _checked(ref,root)
    return [ref],{'required_drugs':expected,'complete_original_compound_names':names,
        'missing_required_parent_models':missing,'all_original_simulations':facts,
        'original_candidate_can_execute_full_joint_regimen':False,
        'new_combined_model_would_change_catalog_candidate_identity':True,
        'scientific_applicability':'unknown'}


def _license_gap(candidate,root):
    ref={'path':str(license_audit.DEFAULT_RECEIPT),'sha256':LICENSE_SHA}
    manifest=json.loads(_checked(ref,root).read_text())
    result=license_audit.audit_candidate_license(candidate,project_root=root)
    if result.get('status')!='unknown' or result.get('reason')!='NO_EXPLICIT_ORIGINAL_SOURCE_GRANT_AT_FROZEN_COMMIT':
        raise ValueError('LICENSE_BLOCKER_NOT_REPRODUCIBLE_FROM_FIXED_SOURCE')
    rows=[r for r in manifest['candidates'] if r['candidate_id']==candidate['candidate_id']]
    if len(rows)!=1 or rows[0].get('status')!='unknown':raise ValueError('LICENSE_ENTRY_NOT_UNKNOWN')
    binding=license_audit._repository_binding(candidate,root)
    if rows[0].get('repository_binding')!=binding:raise ValueError('LICENSE_ORIGINAL_COMMIT_BINDING_DIFFERS')
    docs=[d for d in manifest['documents'] if (d['repository'],d['commit'])==(binding['repository'],binding['commit'])]
    if not docs:raise ValueError('LICENSE_SEARCH_NOT_DOCUMENTED')
    refs=[ref]
    for doc in docs:
        if doc.get('grant'):raise ValueError('LICENSE_GRANT_EXISTS_BLOCKER_INCORRECT')
        if doc.get('path'):
            docref={'path':doc['path'],'sha256':doc['sha256']};_checked(docref,root);refs.append(docref)
    return refs


def _source_absence(plan,scenario,candidate,root):
    """Inspect all saved simulations; mixed adapter failures never pass."""
    prov=candidate['catalog_provenance'];source_ref={'path':prov['source_path'],'sha256':prov['source_sha256']}
    source=_checked(source_ref,root)
    if source.suffix=='.json':
        from .json_source_absence import prove_json_route_absence
        return prove_json_route_absence(plan,scenario,candidate,project_root=root)
    if source.suffix!='.pksim5':raise ValueError('UNSUPPORTED_SOURCE_ABSENCE_FORMAT')
    metadata=source_pkml.source_simulation_metadata(source,prov['source_sha256'],project_root=root)
    simulations=metadata['simulations'];excluded=plan.get('excluded_source_simulations',[])
    if not simulations or len(excluded)!=len(simulations):raise ValueError('ALL_SOURCE_SIMULATION_EXCLUSIONS_REQUIRED')
    by_id={e.get('simulation_id'):e for e in excluded}
    if len(by_id)!=len(excluded) or set(by_id)!={s['simulation_id'] for s in simulations}:
        raise ValueError('SOURCE_SIMULATION_EXCLUSION_COVERAGE_DIFFERS')
    from .complete_candidate import pregnancy_source_policy
    policy=pregnancy_source_policy(candidate,source,metadata,project_root=root)
    alias=policy.get('identity_aliases',{}).get(_name(scenario['compound']))
    expected=_name(alias['native_name']) if alias else _name(scenario['compound'])
    route=plan.get('contract',{}).get('route')
    if route not in {'Oral','Intravenous'}:raise ValueError('EXPLICIT_TARGET_ROUTE_REQUIRED')
    # These original projects contain IV dosing of other drugs. The complete
    # source-bound proof recovers indirect IV associations before checking only
    # Voriconazole's own route; old coadministration/parser exclusion labels do
    # not decide this case. Any unresolved source association still rejects.
    cert_ref=VORICONAZOLE_ROUTE_CERTIFICATES.get(prov['source_sha256'])
    if expected=='voriconazole' and cert_ref is not None:
        n=scenario['normalized_design']
        declared=[n.get('route'),n.get('source_route'),n.get('dose',{}).get('route')]
        declared=[str(r).strip().casefold() for r in declared if r is not None]
        if route!='Intravenous' or not declared or any(r not in {'iv','intravenous'} for r in declared):
            raise ValueError('VORICONAZOLE_TARGET_ROUTE_DIFFERS_FROM_FROZEN_PROTOCOL')
        certificate=json.loads(_checked(cert_ref,root).read_text())
        # Re-parses every original simulation and re-proves all identity links;
        # a stored accepted/status flag, inventory count or filename is not proof.
        verified=event_associations.validate_route_absence_certificate(certificate,
            source_path=source,source_sha256=prov['source_sha256'],compound='Voriconazole',
            required_route='Intravenous',project_root=root)
        if verified.get('accepted') is not True:raise ValueError('COMPLETE_SOURCE_ROUTE_CERTIFICATE_NOT_VERIFIED')
        if {s['simulation_id'] for s in certificate['simulations']}!={s['simulation_id'] for s in simulations}:
            raise ValueError('ROUTE_CERTIFICATE_SOURCE_INVENTORY_DIFFERS')
        facts=[{'simulation_id':s['simulation_id'],'source_blob_sha256':s['source_blob_sha256'],
                'reason':'EXACT_PARENT_INTRAVENOUS_ROUTE_ABSENT_IN_COMPLETE_ORIGINAL_SOURCE',
                'target_compound_declared':s['target_compound_declared'],
                'target_administrations':s['target_administrations'],
                'all_administrations_checked':s['all_administrations_checked'],
                'association_audit':s['association_audit'],
                'scientific_applicability':'unknown'} for s in certificate['simulations']]
        refs=[source_ref,cert_ref,*certificate['implementation']]
        for ref in refs:_checked(ref,root)
        return refs,facts
    facts=[]
    for sim in simulations:
        reason=by_id[sim['simulation_id']]['reason']
        if reason not in SOURCE_ABSENCE_REASONS:raise ValueError('UNRESOLVED_ADAPTER_EXCLUSION:'+reason)
        if reason in {'EXACT_UNIQUE_PARENT_IN_SOURCE_REQUIRED','EXACT_SINGLE_PARENT_COMPOUND_REQUIRED'}:
            if any(_name(c)==expected for c in sim.get('compounds',[])):
                raise ValueError('PARENT_PRESENT_OR_AMBIGUOUS_NOT_SOURCE_ABSENCE')
            # Imported simulations with no identified compound are parser gaps.
            if not sim.get('compounds'):raise ValueError('SOURCE_COMPOUND_INVENTORY_EMPTY')
        elif reason in {'EXPLICIT_HUMAN_SOURCE_REQUIRED','HUMAN_SOURCE_REQUIRED'}:
            species=sim.get('species',[])
            if not species or any(s=='Human' for s in species):raise ValueError('SOURCE_HUMAN_IDENTITY_UNRESOLVED')
        else:
            from .complete_candidate import _metadata_events
            _,assignments=_metadata_events(sim)
            routes={a.get('application_type') for a in assignments.values()
                    if any(_name(c)==expected for c in a.get('administered_compounds',[]))}
            if not routes or None in routes or route in routes:raise ValueError('MATCHING_ROUTE_EXISTS_OR_UNRESOLVED')
        facts.append({'simulation_id':sim['simulation_id'],'reason':reason,
                      'source_blob_sha256':sim['source_blob_sha256']})
    return [source_ref],facts


def assess_blocked_plan(plan,*,project_root=ROOT):
    result={'accepted':False,'category':'engineering_repair_required','reasons':[],
            'scientific_applicability':'unknown','verified_references':[]}
    try:
        if plan.get('status')!='blocked':raise ValueError('BLOCKED_PLAN_REQUIRED')
        if plan.get('sealed_observations_accessed') is not False or plan.get('fitting_performed') is not False:
            raise ValueError('OUTCOMES_FREE_PLAN_REQUIRED')
        reasons=plan.get('reasons',[])
        if not reasons or any(not r.startswith('Blocked:') for r in reasons):raise ValueError('UNEXPECTED_ENGINEERING_EXCEPTION')
        reasons=[r.removeprefix('Blocked:') for r in reasons]
        scenario,candidate,refs=_context(plan,Path(project_root).resolve())
        result['verified_references']=refs
        from .source_contract_gap import registered_pair, assess_plan_gap
        if registered_pair(plan):
            return assess_plan_gap(plan,scenario,candidate,project_root=Path(project_root).resolve())
        from .source_unavailability import PLAN_FAILURE, assess_plan_unavailable
        if reasons==[PLAN_FAILURE]:
            review=assess_plan_unavailable(plan,scenario,candidate,project_root=Path(project_root).resolve())
            review['verified_references']=refs+review['verified_references']
            return review
        clinical=_clinical_gap(scenario,reasons)
        if clinical:
            result.update(accepted=True,category='frozen_clinical_design_unavailable',reasons=[clinical]);return result
        if reasons==['COADMINISTERED_TARGET_DRUGS_REQUIRE_SEPARATE_BOUND_ADAPTER']:
            more,facts=_missing_joint_drug_model(scenario,candidate,Path(project_root).resolve())
            result['verified_references']+=more
            result.update(accepted=True,category='original_source_joint_drug_model_absent',
                          reasons=['required_partner_drug_absent_in_complete_original_candidate'],source_facts=facts)
            return result
        if len(reasons)==1 and reasons[0].split(':')[0]=='CANDIDATE_LICENSE_NOT_RESOLVED':
            result['verified_references']+=_license_gap(candidate,Path(project_root).resolve())
            result.update(accepted=True,category='original_source_license_unresolved',reasons=[reasons[0]]);return result
        if reasons in (['NO_COMPLETE_SOURCE_SIMULATION_MATCHES_FROZEN_CONTRACT'],
                       ['NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT']):
            more,facts=_source_absence(plan,scenario,candidate,Path(project_root).resolve())
            result['verified_references']+=more
            result.update(accepted=True,category='original_source_target_absent',reasons=reasons,source_facts=facts);return result
        raise ValueError('NO_REGISTERED_NONENGINEERING_BLOCKER:'+','.join(reasons))
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result['reasons']=[type(exc).__name__+':'+str(exc)]
        return result


def known_gate_evidence_refs(*,project_root=ROOT):
    """Freeze new unavailable-source and route certificates before execution/scoring."""
    from .source_unavailability import known_unavailability_refs
    root=Path(project_root).resolve();refs=known_unavailability_refs(project_root=root)
    from .json_source_absence import known_json_route_refs
    refs+=known_json_route_refs(project_root=root)
    from .source_contract_gap import known_contract_gap_refs
    refs+=known_contract_gap_refs(project_root=root)
    for ref in VORICONAZOLE_ROUTE_CERTIFICATES.values():
        certificate=json.loads(_checked(ref,root).read_text())
        refs.extend([ref,certificate['source'],*certificate['implementation']])
    unique={}
    for ref in refs:
        _checked(ref,root)
        if ref['path'] in unique and unique[ref['path']]!=ref:
            raise ValueError('GATE_EVIDENCE_HAS_CONFLICTING_CHECKSUMS')
        unique[ref['path']]=ref
    return [unique[name] for name in sorted(unique)]


def assessed_gate_evidence_refs(gate):
    """Every accepted unknown requires the same source evidence in the protocol."""
    refs={}
    for entry in [*gate.get('verified_unavailable_sources',[]),*gate.get('verified_unavailable_cases',[])]:
        review=entry['assessment']
        if review.get('accepted') is not True:
            raise ValueError('UNAVAILABLE_REVIEW_NOT_ACCEPTED')
        for ref in review['verified_references']:
            if ref['path'] in refs and refs[ref['path']]!=ref:
                raise ValueError('GATE_EVIDENCE_HAS_CONFLICTING_CHECKSUMS')
            refs[ref['path']]=ref
    return [refs[name] for name in sorted(refs)]
