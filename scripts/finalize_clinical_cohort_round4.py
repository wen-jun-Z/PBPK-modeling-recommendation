#!/usr/bin/env python3
"""Finish the directly inspected official Olanzapine table, then close retrieval."""
from pathlib import Path
import importlib.util
import json
import os
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('round4_local',ROOT/'scripts/curate_clinical_cohort_round4_local.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
PARENT=h.cp.BASE/'curated_dabigatran/handoff_manifest.json'
h.BASE=h.cp.BASE/'final_olanzapine_handoff'


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    assert not h.BASE.exists(),'Never overwrite saved handoffs'
    parent=json.loads(PARENT.read_text());raw=json.loads((h.cp.BASE/'source_receipt.json').read_text());by={x['compound']:x for x in raw['source_candidates']}
    source=h.cp.source_path('PMC3480756');image=h.cp.BASE/'sealed/olanzapine_table/IJPhS-73-678-g004.jpg'
    assert h.cp.ref(image)['sha256']=='35dc8aaa8f97f9f6607fbdb4cc4d03be3d0d05d60131bf2a14d5053ce44b702a'
    spec=dict(paper=h.cp.BASE/'sealed/articles/PMC8807508.xml',trials=[],snapshot='Olanzapine-Model__6196cdd8a2cb/source/Olanzapine-Model.json',compound='Olanzapine',native_formulation='Solution')
    a=h.audit('olanzapine',spec,by['olanzapine'])
    for d in a['audit_documents']:
        content,_=h.cp.source_content(ROOT/d['document']['path'])
        assert 'singhal2012' not in h.cp.normalize(content),'Unresolved publication-year alias'
    a['publication_year_aliases_checked']=[2011,2012]
    a['selected_reference_arm']='Olanzapine10mg/5mL suspension; testODT10mg excluded'
    aref=h.save('source_audits/olanzapine.json',a)
    case='olanzapine_pmc3480756_reference_suspension10mg_fasted'
    # Manual transcription of the ORIGINAL SOURCE TABLE, not plot digitization.
    transcribed=[('Cmax',13.564,4.00,'ng/mL','13.564±4.00',1),
                 ('AUC_last',514.537,140.33,'ng*h/mL','514.537±140.33',2),
                 ('AUC_infinity',543.721,149.11,'ng*h/mL','543.721±149.11',3)]
    ep=[dict(endpoint=kind,value=value,sd=sd,source_cell=rawcell,source_table_id='T1',source_row_index_in_image=row,source_column='Olanzapine10mg/5ml Suspension(Mean±SD)',source_unit=unit,comparable_unit=unit,conversion_multiplier=1,
        aggregation='arithmetic_mean_of_individual_NCA_endpoints',n_evaluable=16,evaluable_from_source=True) for kind,value,sd,unit,rawcell,row in transcribed]
    assert all(e['value']>0 and e['sd']>=0 for e in ep) and ep[2]['value']>=ep[1]['value']
    eref=h.save('sealed/endpoints/'+case+'.json',dict(schema_version='pbpk-independent-endpoints-sealed-v1',case_id=case,compound='olanzapine',source_article=h.cp.ref(source),source_table_image=h.cp.ref(image),endpoints=ep,
        transcription_validation=dict(method='Direct visual reading of official raster TABLE1; repeat original-image inspection and textual source/footnote cross-check.',reference_column_preserved=True,test_ODT_column_not_used=True,table_footnote_n16_matches_body16completers=True,
            no_plot_digitization=True,no_inferred_extra_precision=True,second_visual_read_same_curator_not_independent_reviewer=True)),True)
    native=h.old.native_status(spec)
    protocol=dict(schema_version='pbpk-independent-clinical-design-v2',case_id=case,compound='olanzapine',source_identifiers=a['target_bibliography']['identifiers'],publication_cluster_id='PMC3480756',cohort_id='PMC3480756:reference_suspension_crossover',trial_registration_ids_at_publication_level=[],
        dose=dict(value=10,unit='mg',basis='10mg active olanzapine in5mL oral suspension',route='oral',formulation='Reference10mg/5mL oral suspension; brand not specified',administration_times_h=[0]),
        population=dict(healthy=True,sex='male',n_enrolled=18,n_evaluable=16,age_mean_year=28.83,age_range_year=[22,39],weight_mean_kg=62.8,weight_range_kg=[51.3,76.6],height_mean_cm=None,bmi_range=[18.6,26.2],demographic_n=18,demographics_for16completers_not_separately_reported=True),
        food='10h overnight fast; first standardized meal4h after dose.',
        sampling=dict(times_h_from_observation_origin=[0,1,2,3,4,5,6,7,8,9,10,12,16,24,36,48,72,96,120,144],origin='Single reference suspension dose;0h is predose',observation_origin_h_after_first_dose=0,reported_sampling_duration_h=144),
        matrix='plasma',analyte='Total parent olanzapine',endpoint_contracts=[{k:v for k,v in e.items() if k not in {'value','sd','source_cell'}} for e in ep],
        aggregation_preselected_before_predictions=True,source_nca=dict(software='WinNonlin5.0.1',cmax='Observed sample maximum',integration='Linear trapezoidal',auc_infinity='AUC_last plus Clast/terminal elimination rate',individual_NCA_then_arithmetic_mean=True),
        native_binding=native,source_independence_audit=aref,sealed_observation_reference=eref,curator_only_source_article=h.cp.ref(source),curator_only_source_table_image=h.cp.ref(image),
        evidence_exclusion=dict(entire_publication='PMC3480756',doi=a['target_bibliography']['identifiers']['doi'],all_trial_arms_and_derived_results=True),
        source_metadata_and_endpoints_ready=True,independent_source_qualification_supported=True,native_execution_verified=False,run_ready=False,actual_evaluation_completed=False,strict_main_evaluated=False,
        execution_requirements=['Validate nonpregnant native source reconstruction and suspension versus source Solution/Weibull release assumptions.','Freeze missing-height/population/genotype and food approximations before prediction.','Reference-subject NCA must be distinguished from arithmetic mean individual NCA.','Freeze predictions before sealed scoring.'],
        limitations=['18enrolled;16completed both periods and Table1 n16.','Mean age/weight describe enrollment, not linked individual covariates.','No raw numerical concentration-time table; figure is not converted into individual curves.','Source Solution is not proof of exact suspension formulation matching.'],observed_values_exposed_to_method_development=False)
    pref=h.save('protocols/'+case+'.json',protocol)
    # The Cabozantinib source paper was already requested before the stop instruction.
    cabpaper=h.cp.BASE/'sealed/model_papers/PMC8224782.xml';cabroot=ET.parse(cabpaper).getroot();content=h.cp.text(cabroot)
    cab=by['cabozantinib'];tokens=list(cab['bibliographic_metadata']['identifiers'].values())+[cab['bibliographic_metadata']['title'],'NCT05263245']
    matches=[t for t in tokens if h.cp.normalize(t) in h.cp.normalize(content)]
    cabcheck=h.save('source_audits/cabozantinib_original_paper_check.json',dict(compound='cabozantinib',original_model_article=h.cp.ref(cabpaper),target_source=cab['source'],references_checked=len(list(cabroot.iter('ref'))),target_identity_matches=matches,
        source_paper_now_saved=True,protocol_not_qualified=True,pending=['Mixed20/40/60mg patient dose/covariate linkage absent.','Absolute pooled mean statistic unresolved; do not assume arithmetic.','Not a healthy adult cohort.']))
    remaining=h.save('remaining_work.json',dict(schema_version='pbpk-cohort-round4-final-open-items-v1',
        candidate_with_unresolved_endpoint_qualification=dict(compound='dabigatran',protocol_handoff=h.cp.ref(PARENT),issues=['Assay meaning of total/free and conjugate handling.','Repeated-period weighting and Table2 analysis denominator; neither50nor47can be silently used.']),
        saved_sources_not_curated=[dict(compound='cabozantinib',source=cab['source'],original_paper_check=cabcheck,pending=['Mixed doses/patient covariate linkage.','Mean aggregation unresolved.']),
            dict(compound='raltegravir',source=by['raltegravir']['source'],pending=['No ordinary NCA endpoints saved.','Large whole supplement blocked; exact DOC attachment not acquired.','7day text/registry versus14day figure discrepancy and staggered period linkage.'])],
        no_more_downloads_running=True,no_scheduled_background_data_acquisition=True,stop_reason='Parent/user requested close foreground and stop this retrieval round; solver pipelines owned by root are unaffected.'))
    checks=[];transfer=0
    for marker in sorted(h.cp.BASE.rglob('*.acquisition.json')):
        item=json.loads(marker.read_text());transfer+=sum(a.get('bytes_received',0) for a in item.get('attempts',[]))
        p=ROOT/item['path'];success=item['status']=='downloaded'
        if success:
            q=h.cp.ref(p);assert q['sha256']==item['sha256'] and q['bytes']==item['bytes']
        else:assert not p.exists()
        checks.append(dict(acquisition_record=h.cp.ref(marker),status=item['status'],payload_validated=success,bytes=item.get('bytes',0)))
    source_receipt=h.save('source_receipt.json',dict(schema_version='pbpk-cohort-final-source-receipt-v1',records=checks,successful_files=sum(x['payload_validated'] for x in checks),failed_files=sum(not x['payload_validated'] for x in checks),
        recorded_round4_received_bytes_lower_bound=transfer,one_oversize_read_additional_bytes_upper_bound=65536,cumulative_limit_bytes=100*1024**2,
        original_table_image=h.cp.ref(image),image_source_page=h.cp.ref(h.cp.BASE/'sealed/olanzapine_table/T1.html'),background_downloads_running=False,download_loop_stopped=True))
    receipt=h.save('curation_receipt.json',dict(schema_version='pbpk-expanded-clinical-curation-receipt-v1',status='olanzapine_official_source_table_curated_and_round_closed',
        records=[dict(compound='olanzapine',case_id=case,pmcid='PMC3480756',protocol=pref,source_audit=aref,sealed_endpoints=eref,endpoint_count=3,source_evaluable_endpoint_count=3,native_verified=False,run_ready=False,evaluated=False)],
        new_unique_drugs=1,new_protocol_records=1,new_target_publications_in_this_batch=1,target_cohorts_in_this_batch=1,sealed_endpoints=3,source_evaluable_endpoints=3,native_verified_cases=0,actual_evaluated_cases=0,full_independent_numeric_curves_added=0,
        source_receipt=source_receipt,remaining_work=remaining,no_model_prediction_or_score_files_read=True,solver_calls=0,gpu_used=False,script=h.cp.ref(Path(__file__))))
    drugs=sorted(parent['drug_names']+['olanzapine']);assert len(drugs)==len(set(drugs))
    handoff=h.save('handoff_manifest.json',dict(schema_version='pbpk-new-clinical-cohort-handoff-v1',status='curation_and_static_verification_complete_retrieval_round_stopped',curation_receipt=receipt,parent_baseline_summary=h.cp.ref(PARENT),source_receipt=source_receipt,remaining_work=remaining,
        new_drugs=['olanzapine'],new_drug_count=1,candidate_drugs_total=len(drugs),drug_names=drugs,clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,
        unique_target_publications_total=parent['unique_target_publications_total']+1,new_target_cohorts=1,new_endpoint_records=3,source_evaluable_endpoint_records=3,new_full_numeric_curve_drugs=0,full_curve_drugs_total=3,
        new_cases_native_verified=0,new_cases_evaluated=0,target_independent_evaluation_drugs=60,remaining_candidate_gap_to_60=60-len(drugs),clinical_metadata_count_is_not_evaluation_count=True,
        old_frozen_protocols_modified=False,predictions_or_scores_read=False,round4_new_candidate_compounds=7,round4_source_endpoint_qualified_compounds=6,round4_endpoint_qualification_pending_compounds=['dabigatran'],
        pending_saved_source_candidate_drugs=['cabozantinib','raltegravir'],qualifying_protocols_directory=str((h.BASE/'protocols').relative_to(ROOT)),background_data_acquisition_running=False))
    print(json.dumps(dict(handoff=handoff,receipt=receipt,source_receipt=source_receipt,cohort_disk_bytes=h.cp.footprint(),candidate_drugs_total=len(drugs),clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,unique_target_publications_total=parent['unique_target_publications_total']+1,round4_qualified_endpoint_compounds=6,full_curve_drugs_total=3,background_downloads_running=False)))


if __name__=='__main__':main()
