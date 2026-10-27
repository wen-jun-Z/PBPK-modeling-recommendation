#!/usr/bin/env python3
"""Preserve a real clinical protocol with explicit unresolved assay/aggregation.

This adds a candidate dossier, not a scoring-ready or native-verified case.
"""
from pathlib import Path
import importlib.util
import json
import os
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('round4_local',ROOT/'scripts/curate_clinical_cohort_round4_local.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
PARENT=h.cp.BASE/'curated_efavirenz/handoff_manifest.json'
h.BASE=h.cp.BASE/'curated_dabigatran'


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    assert not h.BASE.exists(),'Do not overwrite prior outputs'
    parent=json.loads(PARENT.read_text());raw=json.loads((h.cp.BASE/'source_receipt.json').read_text())
    initial=next(x for x in raw['source_candidates'] if x['compound']=='dabigatran')
    source=h.cp.source_path('PMC11909909');article=ET.parse(source).getroot()
    spec=dict(paper=None,trials=['CTR20231968'],snapshot='Dabigatran-Model__9ea348366a54/source/Dabigatran-Model.json',compound='DabiEtex',native_formulation='DabiEtex_Capsule')
    a=h.audit('dabigatran',spec,initial)
    a['scope']='All5registered same-drug/prodrug snapshots, original full OSP model evaluation report and References.md; explicit development/verification clinical-study list includes internal report identifiers, which differ fromCTR20231968.'
    a['original_model_clinical_study_ids']=['1160-0005','1160-0061','1160-0194','1160-0074','1160-0082','Blech2008','Haertter2012','Delavenne2012']
    a['original_model_journal_fulltext_not_available_locally']='Moj2019 DOI10.1007/s40262-019-00776-y; source report documents its inherited structure and clinical sources.'
    aref=h.save('source_audits/dabigatran.json',a)
    table=next(t for t in article.iter('table-wrap') if t.get('id')=='Tab2')
    cells=[[h.cp.text(c) for c in row if c.tag in {'td','th'}] for row in table.iter('tr')]
    ep=[]
    for pool,col,primary in [('total_dabigatran',2,True),('free_dabigatran_source_term',4,False)]:
        for kind,row,unit in [('Cmax',3,'ng/mL'),('AUC_last',5,'ng*h/mL'),('AUC_infinity',6,'ng*h/mL')]:
            value=h.old.scalar(cells[row][col]);assert value>0
            ep.append(dict(endpoint=kind+'_'+pool,endpoint_type=kind,analyte_source_label=pool,value=value,source_cell=cells[row][col],row_0based=row,column_0based=col,table_id='Tab2',
                source_unit=unit,comparable_unit=unit,conversion_multiplier=1,aggregation='source_reported_mean_and_SD_across_fully_replicated_reference_exposures',
                n_evaluable=None,n_enrolled=50,n_completed_all_periods=47,primary_when_qualified=primary,evaluable_from_source=False,
                source_value_extracted_exactly=True,source_quality_blocker='Assay pool definition and whether repeated reference exposures are averaged within participant are unreported; exact endpoint analysis denominator not stated.'))
    case='dabigatran_pmc11909909_ctr20231968_fasted_pradaxa150mg_reference'
    eref=h.save('sealed/endpoints/'+case+'.json',dict(schema_version='pbpk-independent-endpoints-sealed-v1',case_id=case,compound='dabigatran',source_article=h.cp.ref(source),source_table_xml=ET.tostring(table,encoding='unicode'),endpoints=ep),True)
    native=h.old.native_status(spec)
    protocol=dict(schema_version='pbpk-independent-clinical-design-v2',case_id=case,compound='dabigatran',source_identifiers=a['target_bibliography']['identifiers'],publication_cluster_id='PMC11909909',cohort_id='CTR20231968:fasting:reference_exposures',trial_registration_ids_at_publication_level=['CTR20231968'],
        dose=dict(value=150,unit='mg',basis='150mg labelled dabigatran etexilate prodrug dose; not150mg active dabigatran or mesilate salt mass.',route='oral',formulation='Pradaxa dabigatran etexilate150mg capsule, Boehringer Ingelheim International',administration_times_h=[0]),
        crossover=dict(sequences=['TRTR','RTRT'],periods=4,washout_days=7,reference_exposures_per_completer=2,repeated_exposures_are_not_independent_participants=True,within_person_reference_averaging_reported=False),
        population=dict(healthy=True,ethnicity='Chinese',n_enrolled=50,n_completed_all_periods=47,n_for_endpoint_table=None,sex_counts_enrolled=dict(male=39,female=11),age_mean_year=25.62,height_mean_cm=164.59,weight_mean_kg=58.52,bmi_mean=21.54,demographic_means_describe_enrolled50=True),
        food='Fasting arm with240mL warm boiled water; exact overnight fasting duration not explicitly registered here.',
        sampling=dict(times_h_from_observation_origin=[.25,.5,.75,1,80/60,100/60,2,2.5,3,3.5,4,5,6,8,12,24,48],predose_interval_h=[-1,0],origin='Single dose in each reference period',observation_origin_h_after_first_dose=0,
            exact_predose_time_not_imputed=True,source_rounded_times_1_33_and_1_67_resolved_from_explicit80_and100_minutes=True),
        matrix='plasma',analyte='Source total/free dabigatran labels preserved; free must not be equated with protein-unbound without assay method.',
        assay_qualification=dict(target_assay_hydrolysis_or_conjugate_handling_reported=False,native_active_pool_candidate=['Dabigatran','DabiGluc'],native_prodrug='DabiEtex',native_output_pool_mapping_frozen=False,
            source_report_definition='OSP report defines total as dabigatran plus dabigatran glucuronide, but target assay pre-treatment is unspecified and must be checked before applying this mapping.'),
        endpoint_contracts=[{k:v for k,v in e.items() if k not in {'value','source_cell'}} for e in ep],aggregation_preselected_before_predictions=True,
        primary_policy='Total-dabigatran endpoints preferred once assay/aggregation are qualified; source-free pool retained as auxiliary, never select pool based on model fit.',
        native_binding=native,source_independence_audit=aref,sealed_observation_reference=eref,curator_only_source_article=h.cp.ref(source),
        evidence_exclusion=dict(entire_publication='PMC11909909',doi=a['target_bibliography']['identifiers']['doi'],trial_ids=['CTR20231968'],all_trial_arms_and_derived_results=True),
        source_metadata_and_endpoints_ready=False,source_protocol_and_exact_endpoint_transcription_saved=True,independent_source_qualification_supported=True,
        endpoint_comparison_qualified=False,native_execution_verified=False,run_ready=False,actual_evaluation_completed=False,strict_main_evaluated=False,
        execution_requirements=['Confirm target assay meaning and active+glucuronide mapping without using prediction quality.','Resolve repeated-period analysis weighting and endpoint denominator or register its uncertainty explicitly.','Rebuild prodrug/enzyme/transporter/glucuronide chain; do not simulate active drug as oral150mg.','Freeze all assumptions and predictions before sealed scoring.'],
        limitations=['50enrolled and47completers are not automatically the endpoint analysis sample size.','No observed values or corrections imputed.','No interperiod participant splitting for independent evaluation.'],observed_values_exposed_to_method_development=False)
    pref=h.save('protocols/'+case+'.json',protocol)
    remaining=[]
    by={x['compound']:x for x in raw['source_candidates']}
    for drug,missing,links in [
        ('olanzapine',['Reference suspension NCA values exist only in image Table1(IJPhS-73-678-g004.jpg), which is not downloaded. Main text gives only testODT Cmax; do not substitute it.','18enrolled but16completed/analyzed; use16for future endpoint extraction.','Original Zheng2022 full article/references and snapshot already saved, but full clinical-source audit still needs case-level finalization.'],['https://pmc.ncbi.nlm.nih.gov/articles/PMC3480756/table/T1/','https://pmc.ncbi.nlm.nih.gov/articles/PMC8807508/']),
        ('cabozantinib',['Absolute pooled mean statistic is not clearly arithmetic/geometric; Table2 must not be fed to arithmetic scorer by assumption.','12patients have20/40/60mg doses; individual dose/covariate linkage is missing. This is not a healthy adult validation cohort.','Original Gerner2021 full journal article PMC8224782 is not locally saved; current audit only source JSON+README.','Downloaded2026 DOCXsupplement contains menus/power, not patient-level concentration curves.'],['https://pmc.ncbi.nlm.nih.gov/articles/PMC12960307/','https://pmc.ncbi.nlm.nih.gov/articles/PMC8224782/']),
        ('raltegravir',['Target article has no ordinary NCA endpoint table; registry endpoints describe PK/PD protection thresholds, not Cmax/AUC.','Text and official registry say7days while figure says14days; preserve conflict rather than guess time origin.','Whole supplement ZIP exceeded10MiB and was not written. Named supplementary_data.doc and registry protocolPDF are not saved.','Staggered sampling subgroups require period/last-dose linkage; cannot fabricate one full individual curve.'],['https://pmc.ncbi.nlm.nih.gov/articles/PMC8325523/','https://clinicaltrials.gov/study/NCT03205566'])]:
        initial=by[drug]
        remaining.append(dict(compound=drug,target_bibliography=initial['bibliographic_metadata'],saved_source=initial['source'],saved_source_audit_screen=initial['source_audit_screen'],missing_items=missing,official_source_links=links,completed_safe_protocol=False,completed_sealed_endpoint_packet=False,added_to_candidate_count=False))
    pending=h.save('remaining_saved_sources_audit.json',dict(schema_version='pbpk-pending-saved-clinical-sources-v1',records=remaining,no_further_network_requests=True,
        no_discarded_raw_sources=True,predictions_or_scores_read=False,cumulative_cohort_limit_bytes=100*1024**2))
    rec=h.save('curation_receipt.json',dict(schema_version='pbpk-expanded-clinical-curation-receipt-v1',status='candidate_protocol_saved_assay_and_aggregation_unresolved',
        records=[dict(compound='dabigatran',case_id=case,pmcid='PMC11909909',protocol=pref,source_audit=aref,sealed_endpoints=eref,endpoint_count=6,source_evaluable_endpoint_count=0,run_ready=False,native_verified=False,evaluated=False)],
        new_unique_drugs=1,new_protocol_records=1,new_target_publications_in_this_batch=1,target_cohorts_in_this_batch=1,sealed_endpoints=6,source_evaluable_endpoints=0,remaining_saved_sources=pending,
        no_network_calls=True,no_model_prediction_or_score_files_read=True,native_verified_cases=0,actual_evaluated_cases=0,full_independent_numeric_curves_added=0,solver_calls=0,gpu_used=False,script=h.cp.ref(Path(__file__))))
    drugs=sorted(parent['drug_names']+['dabigatran']);assert len(drugs)==len(set(drugs))
    handoff=h.save('handoff_manifest.json',dict(schema_version='pbpk-new-clinical-cohort-handoff-v1',status='candidate_dossier_saved_with_explicit_endpoint_qualification_blockers',curation_receipt=rec,parent_baseline_summary=h.cp.ref(PARENT),
        new_drugs=['dabigatran'],new_drug_count=1,candidate_drugs_total=len(drugs),drug_names=drugs,clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,
        unique_target_publications_total=parent['unique_target_publications_total']+1,new_target_cohorts=1,new_endpoint_records=6,source_evaluable_endpoint_records=0,new_full_numeric_curve_drugs=0,full_curve_drugs_total=3,
        new_cases_native_verified=0,new_cases_evaluated=0,target_independent_evaluation_drugs=60,remaining_candidate_gap_to_60=60-len(drugs),clinical_metadata_count_is_not_evaluation_count=True,
        new_candidate_is_not_endpoint_qualified=True,old_frozen_protocols_modified=False,predictions_or_scores_read=False,pending_saved_source_candidate_drugs=[r['compound'] for r in remaining],remaining_saved_sources=pending,qualifying_protocols_directory=str((h.BASE/'protocols').relative_to(ROOT))))
    print(json.dumps(dict(handoff=handoff,receipt=rec,cohort_disk_bytes=h.cp.footprint(),candidate_drugs_total=len(drugs),clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,unique_target_publications_total=parent['unique_target_publications_total']+1)))


if __name__=='__main__':main()
