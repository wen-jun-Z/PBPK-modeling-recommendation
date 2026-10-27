#!/usr/bin/env python3
"""Finish one saved-source case, preserving prior immutable batch handoffs."""
from pathlib import Path
import importlib.util
import json
import os
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('round4_local',ROOT/'scripts/curate_clinical_cohort_round4_local.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
PARENT=h.BASE/'handoff_manifest.json'
h.BASE=h.cp.BASE/'curated_efavirenz'


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    assert not h.BASE.exists(),'Do not overwrite prior outputs'
    parent=json.loads(PARENT.read_text())
    initial=next(x for x in json.loads((h.cp.BASE/'source_receipt.json').read_text())['source_candidates'] if x['compound']=='efavirenz')
    source=h.cp.source_path('PMC6373243');article=ET.parse(source).getroot()
    spec=dict(paper=None,trials=['PR-10-5001-C'],snapshot='Efavirenz-Model__9c5dffeee226/source/Efavirenz-Model.json',compound='Efavirenz',native_formulation='Sustiva')
    a=h.audit('efavirenz',spec,initial)
    a['scope']='All5 registered efavirenz-containing snapshots and source documentation including original full OSP model evaluation report, explicit model-building clinical study table, parameterization study list, and complete References.md.'
    a['original_model_clinical_studies']=['Mouly2002','Ogburn2013 as labelled in model report','Xu2013','Dooley2012','Garg2013','Huang2012','Mikus2017 midazolam induction','Katzenmaier2010 midazolam induction']
    a['target_study_identity']={'ethics_protocol':'PR-10-5001-C','site':'Parexel Early Phase Clinical Unit,Baltimore','study_period':'November2011-November2012','cohort':'Efavirenz20subjects;day1 before anyrolapitant'}
    aref=h.save('source_audits/efavirenz.json',a)
    table=next(t for t in article.iter('table-wrap') if t.get('id')=='Tab2')
    cells=[[h.cp.text(c) for c in row if c.tag in {'td','th'}] for row in table.iter('tr')]
    indices=[i for i,row in enumerate(cells) if len(row)>=8 and row[0]=='EFV' and row[1]=='1' and row[2]=='20'];assert len(indices)==1
    i=indices[0];ep=[]
    for kind,col,unit in [('Cmax',3,'ng/mL'),('AUC_last',5,'ng*h/mL'),('AUC_infinity',6,'ng*h/mL')]:
        value=h.old.scalar(cells[i][col]);assert value>0
        ep.append(dict(endpoint=kind,value=value,source_cell=cells[i][col],row_0based=i,column_0based=col,table_id='Tab2',source_unit=unit,comparable_unit=unit,conversion_multiplier=1,aggregation='arithmetic_mean_of_individual_NCA_endpoints',reported_dispersion='coefficient_of_variation_percent',n_evaluable=20,evaluable_from_source=True))
    assert ep[2]['value']>=ep[1]['value']
    case='efavirenz_pmc6373243_pr105001c_day1_alone600mg'
    eref=h.save('sealed/endpoints/'+case+'.json',dict(schema_version='pbpk-independent-endpoints-sealed-v1',case_id=case,compound='efavirenz',source_article=h.cp.ref(source),source_table_xml=ET.tostring(table,encoding='unicode'),endpoints=ep),True)
    native=h.old.native_status(spec)
    protocol=dict(schema_version='pbpk-independent-clinical-design-v2',case_id=case,compound='efavirenz',source_identifiers=a['target_bibliography']['identifiers'],publication_cluster_id='PMC6373243',cohort_id='PR-10-5001-C:efavirenz20:day1',
        ethics_protocol_id='PR-10-5001-C',trial_registration_ids_at_publication_level=[],
        dose=dict(value=600,unit='mg',basis='Labelled active efavirenz',route='oral',formulation='Oral formulation not explicitly named in target article; do not silently assert Sustiva brand.',administration_times_h=[0]),
        population=dict(healthy=True,n=20,sex_counts=dict(male=17,female=3),age_mean_year=32.1,weight_mean_kg=78.6,height_mean_cm=None,bmi_mean=25.8,nonsmoking=True),
        food='Morning dosing after at least10h overnight fast with approximately240mL water. Postdose meal time not reported; semi-recumbent for at least4h.',
        sampling=dict(times_h_from_observation_origin=None,origin='First efavirenz dose on day1 before any rolapitant',observation_origin_h_after_first_dose=0,reported_sampling_duration_h=None,missing_nominal_grid_not_fabricated=True,
            source_sentence_ambiguity='The efavirenz study paragraph reports sampling to120h but labels analyte rolapitant, which was absent on day1. Preserve120h as a reported study horizon, not an exact verified efavirenz sampling grid.',reported_study_sampling_horizon_h=120),
        matrix='plasma',analyte='Total parent efavirenz',endpoint_contracts=[{k:v for k,v in e.items() if k not in {'value','source_cell'}} for e in ep],
        auc_definition=dict(AUC_last='Individual NCA to final quantifiable sample; not automatically120h.',AUC_infinity='Individual NCA total with terminal extrapolation.'),
        aggregation_preselected_before_predictions=True,native_binding=native,source_independence_audit=aref,sealed_observation_reference=eref,curator_only_source_article=h.cp.ref(source),
        evidence_exclusion=dict(entire_publication='PMC6373243',doi=a['target_bibliography']['identifiers']['doi'],ethics_protocol_id='PR-10-5001-C',all_trial_arms_and_derived_results=True),
        source_metadata_and_endpoints_ready=True,independent_source_qualification_supported=True,native_execution_verified=False,run_ready=False,actual_evaluation_completed=False,strict_main_evaluated=False,
        execution_requirements=['Verify native induction and source formulation without opening observations.','Freeze unknown brand, genotype, height, food and sampling approximations before prediction.','Reference-subject NCA is not exact arithmetic mean of individual NCA; label approximation.','Freeze predictions and hashes before sealed scoring.'],
        limitations=['Only efavirenz-alone day1 is selected; day10/17 rolapitant carryover periods excluded.','No CYP2B6 genotype distribution; CYP2D6/2C19 selection in other cohorts must not be applied to efavirenz.','No exact sampling grid or brand; endpoint packet is source-ready, not a verified native simulation.'],observed_values_exposed_to_method_development=False)
    pref=h.save('protocols/'+case+'.json',protocol)
    previous=ROOT/'experiments/blueprint_design/external_validation/remaining_cohort_20260920/protocols/repaglinide_pmc9199882_stage2_day1_monotherapy05mg.json'
    oldrep=json.loads(previous.read_text());assert oldrep['source_identifiers']['pmcid']!='PMC6373243'
    correction=h.save('checkpoint_correction.json',dict(corrected_checkpoint=h.cp.ref(h.cp.BASE/'source_receipt.json'),compound='efavirenz',field='pending publication deduplication statement',
        correction='Checkpoint prematurely said the target publication had already been curated for another probe. Existing Repaglinide protocol isPMC9199882, notPMC6373243; this is a new target publication. Shared probes within this new publication remain one study cluster.',evidence=h.cp.ref(previous)))
    receipt=h.save('curation_receipt.json',dict(schema_version='pbpk-expanded-clinical-curation-receipt-v1',status='saved_efavirenz_source_curated_and_static_validated',
        records=[dict(compound='efavirenz',case_id=case,pmcid='PMC6373243',protocol=pref,source_audit=aref,sealed_endpoints=eref,endpoint_count=3,source_evaluable_endpoint_count=3,native_verified=False,run_ready=False,evaluated=False)],
        new_unique_drugs=1,new_protocol_records=1,new_target_publications_in_this_batch=1,target_cohorts_in_this_batch=1,sealed_endpoints=3,source_evaluable_endpoints=3,native_verified_cases=0,actual_evaluated_cases=0,full_independent_numeric_curves_added=0,
        checkpoint_correction=correction,no_network_calls=True,no_model_prediction_or_score_files_read=True,solver_calls=0,gpu_used=False,script=h.cp.ref(Path(__file__))))
    drug_names=sorted(parent['drug_names']+['efavirenz']);assert len(drug_names)==len(set(drug_names))
    handoff=h.save('handoff_manifest.json',dict(schema_version='pbpk-new-clinical-cohort-handoff-v1',status='curation_and_static_verification_complete',curation_receipt=receipt,parent_baseline_summary=h.cp.ref(PARENT),
        new_drugs=['efavirenz'],new_drug_count=1,candidate_drugs_total=len(drug_names),drug_names=drug_names,clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,
        unique_target_publications_total=parent['unique_target_publications_total']+1,new_target_cohorts=1,new_endpoint_records=3,source_evaluable_endpoint_records=3,new_full_numeric_curve_drugs=0,full_curve_drugs_total=3,
        new_cases_native_verified=0,new_cases_evaluated=0,target_independent_evaluation_drugs=60,remaining_candidate_gap_to_60=60-len(drug_names),clinical_metadata_count_is_not_evaluation_count=True,
        old_frozen_protocols_modified=False,predictions_or_scores_read=False,pending_saved_source_candidate_drugs=['dabigatran','olanzapine','cabozantinib','raltegravir'],qualifying_protocols_directory=str((h.BASE/'protocols').relative_to(ROOT))))
    print(json.dumps(dict(handoff=handoff,receipt=receipt,cohort_disk_bytes=h.cp.footprint(),candidate_drugs_total=len(drug_names),clinical_protocol_metadata_total=parent['clinical_protocol_metadata_total']+1,unique_target_publications_total=parent['unique_target_publications_total']+1)))


if __name__=='__main__':main()
