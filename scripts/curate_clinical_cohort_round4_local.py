#!/usr/bin/env python3
"""Curate four saved clinical sources only; no network, fitting, or scoring."""
from pathlib import Path
import importlib.util
import json
import os
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


cp = module('round4_checkpoint', ROOT/'scripts/checkpoint_clinical_cohort_round4.py')
old = module('prior_static_helpers', ROOT/'scripts/curate_expanded_clinical_cohort.py')
BASE = cp.BASE / 'curated_local_batch'
ASSETS = ROOT/'artifacts/pbpk-model-assets/public_collection_20260909'


def save(name, value, sealed=False):
    p=BASE/name;blob=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    assert not p.exists(), 'Never overwrite a saved protocol or sealed observation'
    assert cp.footprint()+len(blob)<100*1024**2
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    if sealed:p.chmod(0o600);p.parent.chmod(0o700)
    return cp.ref(p)


def specs():
    return {
        'esomeprazole':dict(pmc='PMC10175790',suffix='fasted_nexium20mg_reference',dose=20,n=32,
            snapshot='Omeprazole-Model__c8b0f70a40c8/source/Omeprazole-Model.json',compound='Esomeprazole',native_formulation='Omeprazole capsule',
            trials=['CTR20171347','CTR20171484'],formulation='Nexium20mg esomeprazole magnesium enteric-coated tablet, AstraZeneca UK, lotMW325',
            dose_basis='20mg labelled active esomeprazole equivalent; not magnesium salt mass',
            population=dict(healthy=True,ethnicity='Chinese',n=32,sex_counts=dict(male=21,female=11),age_mean_year=30.6,height_mean_cm=165.2,weight_mean_kg=62.2,bmi_mean=22.7),
            times=[0,20/60,40/60,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,4,5,6,8,10,12,24],
            food='At least10h overnight fast and at least4h postdose fasting; no drinking water within1h before/after dosing.',
            table='T2',cells=[('Cmax',3,2,'ng/mL'),('AUC_last',4,2,'ng*h/mL'),('AUC_infinity',5,2,'ng*h/mL')],
            paper=ROOT/'experiments/blueprint_design/external_validation/expanded_cohort_20260920/sealed/model_papers/PMC7764797.xml',
            limitations=['Both CTR IDs belong to publication; fasting-versus-fed ID mapping is not invented.',
                'Source native formulation is an omeprazole capsule, not proven equivalent to this enteric tablet.',
                'CYP2C19 individual genotypes absent; mean-population predictions need explicit genotype assumptions.',
                'CorrigendumPMC10267814 changes author list/contributions only; no scientific result correction.']),
        'posaconazole':dict(pmc='PMC10986696',suffix='fasted_noxafil100mg_be_subset',dose=100,n=33,
            snapshot='Posaconazole-Model__45658be5642f/source/Posaconazole.json',compound='Posaconazol ohne supersaturation',
            native_formulation='POS Tablet Delayed-Release Weibull 50% Krishna et al',trials=[],
            formulation='Noxafil100mg gastro-resistant reference tablet',dose_basis='100mg labelled posaconazole active drug',
            population=dict(healthy=True,n_randomized=36,n_evaluable=33,age_range_eligibility=[18,55],bmi_range_eligibility=[18.5,30],site='Canada',sex_counts=None,age_mean_year=None,weight_mean_kg=None,height_mean_cm=None),
            times=None,duration=144,number_of_samples=24,
            food='At least10h overnight fasting;240mL water with dose; no food until4h, supper/snack no earlier than9h.',
            table='tab-51',cells=[('Cmax',3,3,'ng/mL'),('AUC_last',6,3,'ng*h/mL'),('AUC_infinity',8,3,'ng*h/mL')],
            paper=cp.BASE/'sealed/articles/PMC9785192.xml',
            limitations=['The selected endpoints are the first BE comparison R-fasted n33; do not mix with the separate food-effect subset below.',
                'No exact nominal sampling grid;24samples/predose-to144h only. Dense-Cmax and terminal-AUC approximations must be declared before prediction.',
                'Sex counts and group age/weight/height unavailable; reference subjects do not reconstruct the cohort distribution.',
                'Source non-supersaturation compound and DR-tablet are candidate bindings only; source intestinal-solubility adaptations must be preserved and checked.']),
        'risperidone':dict(pmc='PMC3627046',suffix='fasted_risperdal2mg_reference',dose=2,n=24,
            snapshot='Risperidone-Model__aa97230cf125/source/Risperidone-model.json',compound='Risperidone',native_formulation='Solution',trials=[],
            formulation='Risperdal2mg reference tablet, Xian-Janssen, lot080530784',dose_basis='2mg labelled risperidone parent',
            population=dict(healthy=True,ethnicity='Chinese',sex='male',n=24,age_mean_year=22.9,height_mean_cm=171.3,weight_mean_kg=63.2,bmi_mean=21.5,nonsmoking=True),
            times=[0,.33,.67,1,1.5,2,3,4,5,6,8,10,12,14,24,48,72,96],
            food='10h overnight fast;240mL water; standardized meals at4h and9h; no food before4h.',
            table='Tab2',cells=[('Cmax',2,2,'ng/mL'),('AUC_last',4,2,'ng*h/mL'),('AUC_infinity',5,2,'ng*h/mL')],
            paper=None,
            limitations=['Target is parent risperidone, not9-hydroxyrisperidone or their active-moiety sum.',
                'Source oral Solution is not proof of tablet release equivalence.',
                'Individual CYP2D6 genotypes not reported; population/reference approximation must be frozen.',
                'Use online2012 and issue2013 bibliographic aliases in source exclusion.']),
        'ruxolitinib':dict(pmc='PMC6221065',suffix='incb18424139_ir25mg_period1',dose=25,n=9,
            snapshot='Ruxolitinib-Model__d726edba1fcc/source/Ruxolitinib.json',compound='Ruxolitinib',native_formulation='Tablet Jakavi',trials=['INCB18424-139'],
            formulation='Ruxolitinib immediate-release25mg tablet, period1; exclude SR-1/SR-2',dose_basis='25mg source-labelled active ruxolitinib, not phosphate salt mass',
            population=dict(healthy=True,n=9,sex_counts=dict(male=6,female=3),age_median_year=27,age_range_year=[18,53],race_counts=dict(White=8,African_American=1),weight_mean_kg=None,height_mean_cm=None),
            times=[0,.25,.5,1,1.5,2,3,4,6,8,12,16,24,36],
            food='At least10h overnight fast before single dose; postdose meal schedule not stated in inspected supplement.',
            table='hon2544-tbl-0001',cells=[('Cmax',2,1,'nM'),('AUC_last',7,1,'nM*h'),('AUC_infinity',8,1,'nM*h')],
            paper=cp.BASE/'sealed/articles/PMC9785192.xml',
            limitations=['Healthy phase1 is INCB18424-139; NCT01340651 belongs to separate myelofibrosis phase2 in this publication.',
                'Median age27 is not an arithmetic mean; no mean weight/height supplied.',
                'IRperiod1 has9subjects, later SRperiods8; no averaging across formulations.',
                'Native source binding must use immediate release, not the Chen historical ER-labelled formulation.']),
    }


def audit(drug,spec,initial):
    result=json.loads(json.dumps(initial['source_audit_screen']))
    docs=result.pop('documents')
    target=initial['bibliographic_metadata'];ids=target['identifiers']
    tokens=list(ids.values())+[target['title']]+spec['trials']
    added=[]
    if spec['paper']:
        content,nrefs=cp.source_content(spec['paper'])
        added.append((spec['paper'],content,nrefs,None,'original_model_full_article_and_references'))
    if drug=='risperidone':
        path=ROOT/'artifacts/pbpk_raw_runs/compound_runs/run_20260909T083120Z/compounds/paroxetine/pmc/efetch_xml/pmc_oa/pmc_efetch_pmc4165923_pmc9414337_20_46240dcf456b.xml'
        r=ET.parse(path).getroot()
        a=next(a for a in r.iter('article') if any(cp.text(x)=='10.3390/pharmaceutics14081734' for x in a.findall('.//article-meta/article-id')))
        added.append((path,cp.text(a),len(list(a.iter('ref'))),'10.3390/pharmaceutics14081734','selected_original_model_article_in_multiarticle_container'))
    for p,content,nrefs,selector,kind in added:
        if any(x['document']['path']==str(p.relative_to(ROOT)) for x in docs):continue
        normalized=cp.normalize(content)
        matches=[t for t in tokens if cp.normalize(t) in normalized]
        surname=target['authors'][0]['surname'];years={d['components'].get('year') for d in target['publication_dates_as_encoded']}
        flags=[surname+y for y in years if y and cp.normalize(surname+y) in normalized]
        docs.append(dict(document=cp.ref(p),kind=kind,article_doi_selector=selector,references_present=nrefs,target_identifier_or_title_matches=matches,first_author_year_flags=flags))
    assert all(not d['target_identifier_or_title_matches'] and not d['first_author_year_flags'] for d in docs), 'Unresolved source overlap'
    labels=[]
    for d in docs:
        if d['kind']=='registered_same_compound_snapshot':
            model=json.loads((ROOT/d['document']['path']).read_text());labels.extend(o.get('Name','') for o in model.get('ObservedData',[]))
    return dict(schema_version='pbpk-clinical-source-independence-audit-v2',compound=drug,target_bibliography=target,
        clinical_study_identities=spec['trials'],audit_documents=docs,registered_historical_study_labels=sorted(set(labels)),
        known_clinical_overlap=[],clinical_source_independence_supported_at_declared_scope=True,
        scope='Registered same-compound snapshot study identities, repository source documents, original-model full article/reference lists; risperidone also full clinical-source supplement and publisher reference list.',
        limitations=['Clinical-source identity audit cannot prove absence of undocumented participant re-enrollment.',
                    'Not all original-paper supplemental files are locally available; supported by explicit source study labels and main clinical-source citations, not publication date alone.'],
        clinical_cohort_is_not_historical_snapshot_resplit=True,exclude_entire_target_publication_and_registered_trials_from_all_development=True,
        no_target_observations_used_to_select_or_fit_source_model=True)


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    parent=json.loads(cp.PARENT.read_text());checkpoint=json.loads((cp.BASE/'source_receipt.json').read_text())
    initial={r['compound']:r for r in checkpoint['source_candidates']}
    assert not BASE.exists(),'Do not overwrite a completed or partially written curation batch'
    records=[]
    for drug,spec in specs().items():
        source=cp.source_path(spec['pmc']);article=ET.parse(source).getroot()
        a=audit(drug,spec,initial[drug]);aref=save(f'source_audits/{drug}.json',a)
        table=next(t for t in article.iter('table-wrap') if t.get('id')==spec['table'])
        cells=[[cp.text(c) for c in row if c.tag in {'td','th'}] for row in table.iter('tr')]
        case=f"{drug}_{spec['pmc'].lower()}_{spec['suffix']}";endpoints=[]
        for kind,row,col,unit in spec['cells']:
            token=cells[row][col];value=old.scalar(token);assert value>0
            endpoints.append(dict(endpoint=kind,value=value,source_cell=token,row_0based=row,column_0based=col,table_id=spec['table'],
                source_unit=unit,comparable_unit=unit,conversion_multiplier=1,aggregation='arithmetic_mean_of_individual_NCA_endpoints',n_evaluable=spec['n'],evaluable_from_source=True))
        ep={x['endpoint']:x['value'] for x in endpoints};assert ep['AUC_infinity']>=ep['AUC_last']
        eref=save(f'sealed/endpoints/{case}.json',dict(schema_version='pbpk-independent-endpoints-sealed-v1',case_id=case,compound=drug,source_article=cp.ref(source),source_table_xml=ET.tostring(table,encoding='unicode'),endpoints=endpoints),True)
        native=old.native_status(dict(snapshot=spec['snapshot'],compound=spec['compound'],native_formulation=spec['native_formulation']))
        protocol=dict(schema_version='pbpk-independent-clinical-design-v2',case_id=case,compound=drug,
            source_identifiers=a['target_bibliography']['identifiers'],publication_cluster_id=spec['pmc'],cohort_id=f"{spec['pmc']}:{spec['suffix']}",
            trial_registration_ids_at_publication_level=spec['trials'],
            dose=dict(value=spec['dose'],unit='mg',basis=spec['dose_basis'],route='oral',formulation=spec['formulation'],administration_times_h=[0]),
            food=spec['food'],population=spec['population'],matrix='plasma',analyte='Total parent plasma drug; not metabolite or unbound concentration',
            sampling=dict(times_h_from_observation_origin=spec['times'],origin='Single target reference dose; predose sample remains predose',observation_origin_h_after_first_dose=0,
                reported_sampling_duration_h=spec.get('duration',spec['times'][-1] if spec['times'] else None),reported_number_of_samples=spec.get('number_of_samples',len(spec['times']) if spec['times'] else None),missing_nominal_grid_not_fabricated=spec['times'] is None),
            endpoint_contracts=[{k:v for k,v in e.items() if k not in {'value','source_cell'}} for e in endpoints],
            aggregation_preselected_before_predictions=True,auc_definition=dict(AUC_last='Individual NCA to last measurable concentration, not necessarily final scheduled144h/96h point.',AUC_infinity='Individual NCA with terminal extrapolation; cannot substitute truncated AUC silently.'),
            native_binding=native,source_independence_audit=aref,sealed_observation_reference=eref,
            curator_only_source_article=cp.ref(source),evidence_exclusion=dict(entire_publication=spec['pmc'],doi=a['target_bibliography']['identifiers']['doi'],trial_ids=spec['trials'],all_trial_arms_and_derived_results=True),
            source_metadata_and_endpoints_ready=True,independent_source_qualification_supported=True,native_execution_verified=False,run_ready=False,actual_evaluation_completed=False,strict_main_evaluated=False,
            execution_requirements=['Verify native reconstruction and source compound/formulation binding without reading external outcomes.',
                'Freeze population, genotype, food and release approximations before predictions.',
                'Arithmetic mean of individual NCA endpoints is not NCA of a mean concentration curve; freeze compatible aggregation or label reference-subject approximation.',
                'Freeze predictions and hashes before any sealed scoring.']+(['Resolve native process types:'+','.join(native['process_types_not_in_inspected_adapter'])] if native['process_types_not_in_inspected_adapter'] else []),
            limitations=spec['limitations'],observed_values_exposed_to_method_development=False)
        if drug=='ruxolitinib':protocol['sampling']['source_supplement']={**cp.ref(cp.BASE/'sealed/articles/PMC6221065_supplements.zip'),'member':'HON-36-701-s001.docx'}
        if drug=='esomeprazole':protocol['bibliographic_corrigendum']={**cp.ref(cp.BASE/'sealed/articles/PMC10267814.xml'),'correction_type':'Author/contributor list only; scientific findings unchanged.'}
        pref=save(f'protocols/{case}.json',protocol)
        records.append(dict(compound=drug,case_id=case,pmcid=spec['pmc'],protocol=pref,source_audit=aref,sealed_endpoints=eref,endpoint_count=len(endpoints),source_evaluable_endpoint_count=len(endpoints),run_ready=False,native_verified=False,evaluated=False))
    r=save('curation_receipt.json',dict(schema_version='pbpk-expanded-clinical-curation-receipt-v1',status='four_saved_sources_curated_and_static_validated',records=records,
        new_unique_drugs=4,new_protocol_records=4,new_target_publications_in_this_batch=4,target_cohorts_in_this_batch=4,sealed_endpoints=12,source_evaluable_endpoints=12,
        full_independent_numeric_curves_added=0,native_verified_cases=0,actual_evaluated_cases=0,no_network_calls=True,no_model_prediction_or_score_files_read=True,solver_calls=0,gpu_used=False,script=cp.ref(Path(__file__))))
    drugs=sorted(set(parent['drug_names'])|set(specs()));assert len(drugs)==parent['candidate_drugs_total']+4
    h=save('handoff_manifest.json',dict(schema_version='pbpk-new-clinical-cohort-handoff-v1',status='curation_and_static_verification_complete',curation_receipt=r,parent_baseline_summary=cp.ref(cp.PARENT),raw_source_checkpoint=cp.ref(cp.BASE/'handoff_manifest.json'),
        new_drugs=sorted(specs()),new_drug_count=4,candidate_drugs_total=len(drugs),drug_names=drugs,
        clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+4,unique_target_publications_total=parent['unique_target_publications_total']+4,
        new_target_cohorts=4,new_endpoint_records=12,source_evaluable_endpoint_records=12,new_full_numeric_curve_drugs=0,full_curve_drugs_total=3,
        new_cases_native_verified=0,new_cases_evaluated=0,target_independent_evaluation_drugs=60,remaining_candidate_gap_to_60=60-len(drugs),
        clinical_metadata_count_is_not_evaluation_count=True,old_frozen_protocols_modified=False,predictions_or_scores_read=False,
        pending_saved_source_candidate_drugs=sorted(set(initial)-set(specs())),qualifying_protocols_directory=str((BASE/'protocols').relative_to(ROOT))))
    print(json.dumps(dict(handoff=h,curation_receipt=r,drug_count=len(drugs),protocol_count=parent['clinical_protocol_metadata_total']+4,cohort_disk_bytes=cp.footprint(),new_endpoint_packets=4,new_endpoint_records=12,native_verified=0,evaluated=0)))


if __name__=='__main__':main()
