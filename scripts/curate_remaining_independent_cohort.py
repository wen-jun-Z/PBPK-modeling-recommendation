"""Curate five additional public monotherapy arms without model/scorer execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'experiments/blueprint_design/independent_cohort_20260920'
OUT = ROOT / 'experiments/blueprint_design/external_validation/remaining_cohort_20260920'
ASSETS = ROOT / 'artifacts/pbpk-model-assets/public_collection_20260909'


def ref(p):
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def save(p, value, sealed=False):
    content = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    budget_dirs = [PRIOR, ROOT/'experiments/blueprint_design/external_validation']
    # Include all cohort curation files, excluding native execution outputs.
    total = sum(f.stat().st_size for base in budget_dirs for f in base.rglob('*')
                if f.is_file() and not any(part in ('native_runs', 'native', 'stage', 'snapshots') for part in f.parts))
    if total + len(content) > 100 * 1024**2:
        raise ValueError('100 MiB cumulative curation limit')
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_bytes() != content:
        raise ValueError('Refusing changed output: '+str(p))
    p.write_bytes(content)
    if sealed:
        p.chmod(0o600)
        p.parent.chmod(0o700)
    return ref(p)


def txt(e):
    return ' '.join(''.join(e.itertext()).split()) if e is not None else ''


def specs():
    return {
      'felodipine': dict(pmc='PMC5642591', aliases=['felodipine'], suffix='monotherapy_er5mg_bid_day7',
        snapshot='Felodipine-Model__f94595ca0ae1/source/Felodipine-Model.json', compound='Felodipine', formulation='Felodipine_ER tablet', papers=[],
        trials=[], excluded_background_trials=[], n=12,
        dose=dict(value=5,unit='mg',basis='labelled felodipine dose',route='oral',formulation='Jiangsu Lianhuan extended-release tablet 5mg',frequency='twice daily at08:00 and20:00 for7days',last_dose_day=7,explicit_total_dose_count=None),
        dose_times_h=None, nominal_times_h=[0,.25,.5,.75,1,1.5,2,3,4,5,6,8,12,24,36,48,72],
        origin='last dose on day7; morning/evening placement of that final dose not unambiguously specified in text',
        food='Source says fasted more than10h and250mL water; exact repeated evening meal restriction is not independently documented.',
        population=dict(n=12,healthy=True,sex_counts=dict(male=6,female=6),age_mean_year=24.3,weight_mean_kg=57.3,height_mean_cm=163.2,bmi_mean=21.2,ethnicity='Chinese'),
        aggregation='arithmetic_mean_of_individual_NCA_endpoints', table='T4', endpoints=[('Cmax_ss',2,1,'ng/mL','ng/mL',1,12),('AUC_tau_ss',3,1,'ng*h/mL','ng*h/mL',1,12)],
        auc=dict(definition='steady-state dosing-interval exposure',nominal_interval_h=12,not_AUC_infinity=True),
        protocol_blockers=['Source does not unambiguously state whether final day7 sampling follows the morning or evening dose; no13/14-dose history has been invented. Exact repeated-dose prediction requires resolving or prospectively registering this ambiguity.'],
        limitations=['Generic ER formulation equivalence to source-native Weibull is not established.', 'Original table means±SD differ from reported geometric mean ratios; only absolute arithmetic means are curated.']),
      'clopidogrel': dict(pmc='PMC4422294', aliases=['clopidogrel'], suffix='reference_plavix75mg_fasted',
        snapshot='Clopidogrel-Model__a9667d032264/source/Clopidogrel-Model.json',compound='Clopidogrel',formulation='Clopidogrel_Plavix_Weibull_Valdez et al 2020',papers=['PMC9145019'],
        trials=[],excluded_background_trials=[],n=38,
        dose=dict(value=75,unit='mg',basis='labelled clopidogrel active-moiety dose; bisulfate/hydrogen-sulfate formulation not75mg salt mass',route='oral',formulation='Plavix75mg film-coated tablet',administrations_h=[0]),dose_times_h=[0],
        nominal_times_h=[-1,.25,.5,.75,1,1.33,1.67,2,2.33,2.67,3,3.5,4,5,6,8,10,12,16,24,36,48],origin='single oral dose; predose blank was collected at-1h and is not a0h measurement',
        food='Fasting;240mL water; standard lunch4h postdose; exact fasting duration not stated.',
        population=dict(n=38,healthy=True,ethnicity='Arab/Mediterranean',age_range_year=[19,46],bmi_range=[18.9,30],minimum_weight_kg=52,sex_counts=None,age_mean_year=None,weight_mean_kg=None),
        aggregation='geometric_mean_of_individual_NCA_endpoints',table='t1-dddt-9-2359',
        endpoints=[('Cmax',2,2,'pg/mL','ng/mL',.001,38),('AUC_last',3,2,'pg*h/mL','ng*h/mL',.001,38),('AUC_infinity',4,2,'pg*h/mL','ng*h/mL',.001,38)],
        auc=dict(definition='separate AUC0-last and AUC0-infinity',last_quantifiable_time_not_equated_to_48h=True),protocol_blockers=[],
        limitations=['Sex and mean body-size/age are not reported; prediction needs a declared reference/population approximation.', 'Geometric endpoint aggregation must not be compared with an arithmetic-mean endpoint prediction.', 'Native parent-metabolite pathways must be preserved; parent-only deletion is not scientifically equivalent.']),
      'repaglinide':dict(pmc='PMC9199882',aliases=['repaglinide_dt','repaglinide'],suffix='stage2_day1_monotherapy05mg',
        snapshot='Clopidogrel-Repaglinide-DDI__52112a1cb753/source/Clopidogrel-Repaglinide-DDI.json',compound='Repaglinide_DT',formulation='Repaglinide (Zhu 2014 dissolution profile)_DT',papers=['PMC6885506','PMC9145019'],
        pksim='artifacts/pbpk-model-assets/coverage_expansion_20260907/Repaglinide-Model/source/Repaglinide-Model.pksim5',trials=[],excluded_background_trials=['NCT02081378','NCT03106779'],n=25,
        dose=dict(value=.5,unit='mg',basis='labelled repaglinide active drug',route='oral',formulation='oral tablet;brand not reported',administrations_h=[0]),dose_times_h=[0],
        nominal_times_h=[0,.25,.5,.75,1,1.5,2,3,4,6,8,10,12,24],origin='stage2 day1 morning dose before any asciminib treatment',food='At least10h fasting before and4h after dose;240mL water.',
        population=dict(enrolled=25,monotherapy_pk_n=25,completed_all_periods=24,sex_counts_enrolled=dict(male=21,female=4),age_median_year=41,age_range_year=[26,55],weight_median_kg=85.3,weight_range_kg=[58.4,109.4],bmi_median=25.1,healthy=True,country='Germany'),
        aggregation='geometric_mean_of_individual_NCA_endpoints',table='cts13252-tbl-0002',
        endpoints=[('AUC_infinity',5,7,'ng*h/mL','ng*h/mL',1,23),('AUC_last',8,7,'ng*h/mL','ng*h/mL',1,25),('Cmax',11,7,'ng/mL','ng/mL',1,25)],
        auc=dict(definition='separate AUC0-last and AUC0-infinity',last_quantifiable_time_not_equated_to_24h=True),protocol_blockers=[],
        limitations=['AUCinfinity uses23 participants;AUClast/Cmax use25; no forced common denominator.', 'Demographics are medians, not means; a median reference is only an explicitly registered approximation.', 'Stage1 cocktail and stage2 asciminib-treated periods are excluded; background NCTs are not this trial registration.', 'Repaglinide_DT identity is the source model alias; transporter/MM compound mapping requires native validation.']),
      'itraconazole':dict(pmc='PMC7526808',aliases=['itraconazole'],suffix='15day_sporanox200mg_bid_fed',
        snapshot='Itraconazole-Model__59b6b686ba37/source/Itraconazole-Model.json',compound='Itraconazole',formulation='Capsule fed',papers=['PMC6202474'],trials=[],excluded_background_trials=['NCT03572049'],n=16,
        dose=dict(value=200,unit='mg',basis='itraconazole parent;two100mg capsules',route='oral',formulation='Sporanox conventional capsules',regimen='200mg BID days1–14 then200mg once morning day15',administrations_h=list(range(0,337,12))),dose_times_h=list(range(0,337,12)),
        nominal_times_h=[0,.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12],origin='last morning dose on day15,336h after first dose under nominal12h intervals',food='High-fat high-calorie meals at each dose; high-fat high-calorie breakfast on day15.',
        population=dict(enrolled=24,pk_analyzed=16,sex_counts_analyzed=dict(male=9,female=7),age_mean_year=33,weight_mean_kg=82.3,height_mean_cm=170.7,bmi_mean=28.1,healthy=True,ethnicity_counts_analyzed={'Black/African American':12,'White':1,'Multiracial':2,'American Indian/Alaska Native':1}),
        aggregation='geometric_mean_of_individual_NCA_endpoints',table='T3',endpoints=[('Cmax_ss',3,2,'ng/mL','ng/mL',1,16),('AUC_tau_ss',5,2,'ng*h/mL','ng*h/mL',1,16)],auc=dict(definition='steady-state AUC over12h after day15 dose',nominal_interval_h=12,not_AUC_infinity=True),protocol_blockers=[],
        limitations=['Nominal evenly spaced12h BID schedule must be declared; individual actual administration times are not available.', 'Source sequential hydroxy/keto/N-desalkyl metabolites and feedback inhibition must remain intact.', 'Three-day loading cohort and SUBA formulation are not extra observations for the chosen conventional15-day reference arm.', 'Background NCT03572049 refers to a different efficacy study and is not assigned here.']),
      'pioglitazone':dict(pmc='PMC7591087',aliases=['pioglitazone_dt','pioglitazone'],suffix='nct02753803_day7_monotherapy30mg',
        snapshot='Clopidogrel-Pioglitazone-DDI__0d73e9b3edb3/source/Clopidogrel-Pioglitazone-DDI.json',compound='Pioglitazone_DT',formulation='Pioglitazone (Saha 2013 dissolution profile)_DT',papers=['PMC6885506','PMC9145019'],
        pksim='artifacts/pbpk-model-assets/public_collection_20260909/Pioglitazone-Model__8cb10032891c/source/Pioglitazone-Model.pksim5',trials=['NCT02753803'],excluded_background_trials=['NCT00961025'],n=34,
        dose=dict(value=30,unit='mg',basis='labelled pioglitazone dose;do not reinterpret as hydrochloride salt mass',route='oral',formulation='oral pioglitazone30mg; brand not identified in article text',administrations_h=[0,24,48,72,96,120,144]),dose_times_h=[0,24,48,72,96,120,144],
        nominal_times_h=[0,.5,1,1.5,2,3,4,5,6,8,12,24],origin='last once-daily dose on day7 (144h after first dose)',food='Fasted state;150mL water; precise fasting duration not reported.',
        population=dict(enrolled=36,safety_n=35,pk_analyzed=34,sex='male',healthy=True,ethnicity='Korean',age_mean_year=33,weight_mean_kg=68.8,height_mean_cm=173.7,bmi_mean=22.8,demographic_summary_population='all36enrolled, not separately reported for34PK'),
        aggregation='arithmetic_mean_of_individual_NCA_endpoints',table='t0002',endpoints=[('Cmax_ss',3,1,'microgram/L','ng/mL',1,34),('AUC_tau_ss',4,1,'microgram*h/L','ng*h/mL',1,34)],auc=dict(definition='steady-state dosing interval0–24h after last dose',nominal_interval_h=24,not_AUC_infinity=True),protocol_blockers=[],
        limitations=['Use only PIO monotherapy, not evogliptin coadministration or metabolite endpoints.', 'Demographics of36enrolled approximate34evaluable; distinction retained.', 'Brand/release match to source Saha2013 formulation not verified; any reuse must be declared.', 'The original standalone model is present as PK-Sim SQLite; current JSON source alias Pioglitazone_DT comes from the documented2019 model.'])}


def audit(drug, spec, identifiers, inventory):
    needles = [v.lower() for v in identifiers.values() if v] + [x.lower() for x in spec['trials']]
    evidence, overlaps, source_labels = [], [], []
    source_models = []
    for row in inventory['drug_inventory']:
        if row['compound'] in spec['aliases']:
            source_models += row['models']
            source_labels += [h['source_study_label'] for h in row['historical_clinical_studies']]
    source_paths = {ROOT / m['snapshot_path'] for m in source_models}
    source_paths.add(ASSETS / spec['snapshot'])
    docpaths = set()
    for path in source_paths:
        docpaths.update(path.parent.rglob('*.md'))
        content = path.read_text().lower()
        matched = [n for n in needles if n in content]
        if matched: overlaps.append(dict(path=str(path.relative_to(ROOT)), matched_identifiers=matched))
        evidence.append({**ref(path), 'type':'registered_model_snapshot', 'matched_target_identifiers':matched})
    for path in sorted(docpaths):
        content = path.read_text().lower(); matched = [n for n in needles if n in content]
        if matched: overlaps.append(dict(path=str(path.relative_to(ROOT)),matched_identifiers=matched))
        evidence.append({**ref(path),'type':'repository_source_document','matched_target_identifiers':matched})
    for pmc in spec['papers']:
        path=OUT/'source_audits/model_papers'/f'{pmc}.xml';r=ET.fromstring(path.read_bytes())
        content=txt(r).lower();matched=[n for n in needles if n in content]
        if matched: overlaps.append(dict(path=str(path.relative_to(ROOT)),matched_identifiers=matched))
        evidence.append({**ref(path),'type':'original_model_paper_full_body_and_references','reference_count':len(r.findall('.//ref')),'matched_target_identifiers':matched})
    if spec.get('pksim'):
        path=ROOT/spec['pksim'];db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        # Only source-study names/descriptions, never simulation outcomes/blobs.
        labels=[dict(name=name,description=description) for name,description in db.execute('SELECT Name, Description FROM DATA_REPOSITORIES')]
        db.close(); content=json.dumps(labels,ensure_ascii=False).lower();matched=[n for n in needles if n in content]
        if matched:overlaps.append(dict(path=spec['pksim'],matched_identifiers=matched))
        evidence.append({**ref(path),'type':'original_PKSim_SQLite_readonly_source_metadata','clinical_source_labels':labels,'matched_target_identifiers':matched})
    if overlaps:raise ValueError('Clinical source overlaps discovered for '+drug)
    return dict(compound=drug,target_identifiers=identifiers,target_trial_ids=spec['trials'],excluded_background_trial_ids=spec['excluded_background_trials'],
                native_source_alias=spec['compound'],audit_documents=evidence,registered_historical_labels=sorted(set(source_labels)),overlaps=overlaps,
                clinical_source_disjoint_at_audited_source_level=True,
                source_chain='Original PBPK paper fullbody/reference list, current repository model/source docs, all registered same-compound snapshot identities, and standalone PKSim source-study metadata when available.',
                qualification_limit='No undocumented participant-reuse guarantee. Clinical cohort/source separation is audited; generic physiological references are not treated as clinical overlap.',
                supplement_scope='No model supplement >2MiB downloaded; model original clinical identities were audited through actual registered source-model data repositories and published reference lists.')


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    inventory=json.loads((PRIOR/'cohort_inventory.json').read_text());records=[];validations=[];deny=[]
    for drug,spec in specs().items():
        sources=list(PRIOR.rglob(spec['pmc']+'.xml'));source=sources[0];article=ET.fromstring(source.read_bytes());assert article.find('.//body') is not None
        ids={x.get('pub-id-type'):x.text for x in article.findall('./front/article-meta/article-id')};identifiers={k:ids[k] for k in ['pmcid','pmid','doi']}
        source_audit=audit(drug,spec,identifiers,inventory);audit_ref=save(OUT/'source_audits'/f'{drug}.json',source_audit)
        case_id=f"{drug}_{spec['pmc'].lower()}_{spec['suffix']}";table=next(x for x in article.findall('.//table-wrap') if x.get('id')==spec['table']);rows=table.findall('.//tr')
        endpoints=[];contracts=[]
        for label,row,col,unit,comparable,multiplier,n in spec['endpoints']:
            cells=[x for x in rows[row] if x.tag in ['td','th']];token=txt(cells[col]);m=re.match(r'\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',token)
            if not m:raise ValueError('Cannot parse exact source scalar for '+case_id)
            value=float(m.group(1).replace(',',''));assert value>0
            endpoint=dict(endpoint=label,value=value,source_unit=unit,source_cell=token,row_0based=row,column_0based=col,table_id=spec['table'],source_scalar_number=0,
                          aggregation=spec['aggregation'],n_evaluable=n,comparable_value=value*multiplier,comparable_unit=comparable,unit_conversion_multiplier=multiplier)
            endpoints.append(endpoint);contracts.append({k:v for k,v in endpoint.items() if k not in ['value','source_cell','comparable_value','source_scalar_number']})
        sealed=save(OUT/'sealed'/f'{case_id}.json',dict(schema_version='pbpk-independent-endpoints-sealed-v1',case_id=case_id,compound=drug,identifiers=identifiers,
          source_article=ref(source),source_table_xml=ET.tostring(table,encoding='unicode'),endpoints=endpoints),True)
        modelpath=ASSETS/spec['snapshot'];snapshot=json.loads(modelpath.read_text());compound=next(c for c in snapshot['Compounds'] if c['Name']==spec['compound']);assert spec['formulation'] in [f['Name'] for f in snapshot['Formulations']]
        protocol=dict(schema_version='pbpk-independent-protocol-v1',case_id=case_id,compound=drug,source_identifiers=identifiers,
          publication_cluster_id=spec['pmc'],cohort_id=spec['trials'][0] if spec['trials'] else f"{spec['pmc']}:{spec['suffix']}",
          actual_trial_registration_ids=spec['trials'],background_trial_ids_explicitly_not_assigned=spec['excluded_background_trials'],
          dose=spec['dose'],native_nominal_dose_times_h=spec['dose_times_h'],nominal_sampling_hours_from_observation_origin=spec['nominal_times_h'],observation_origin=spec['origin'],
          food=spec['food'],population=spec['population'],matrix='plasma',analyte='parent total drug',n_evaluable_by_endpoint={e['endpoint']:e['n_evaluable'] for e in endpoints},
          endpoint_contracts=contracts,auc_definition=spec['auc'],source_independence_audit=audit_ref,
          native_binding=dict(snapshot=ref(modelpath),compound_name=spec['compound'],formulation_name=spec['formulation'],source_process_types=sorted(set(p['InternalName'] for p in compound['Processes'])),
                              includes_additional_molecules=len(snapshot['Compounds'])>1,actual_native_execution_verified=False,
                              instructions='Preserve selected compound processes/metabolite dependencies; remove coadministered drug exposure only through validated source-native assembly, not by deleting target clearance/inhibition paths.'),
          sealed_observation_reference=sealed,curator_only_source_article=ref(source),evidence_exclusion=dict(entire_publication=spec['pmc'],all_trial_arms_and_derived_results=True,trial_ids=spec['trials']),
          source_ready=True,protocol_execution_ambiguities=spec['protocol_blockers'],run_ready=False,strict_main_evaluated=False,outcomes_exposed_to_designer=False,
          limitations=spec['limitations'],required_before_prediction=['Freeze source-native model/protocol/population assumptions without target outcomes.','Resolve registered dosing ambiguities, or retain them as explicit unavailable-case reasons.','Lock all predictions before geometric/arithmetic endpoint scoring.'])
        pref=save(OUT/'protocols'/f'{case_id}.json',protocol)
        records.append(dict(compound=drug,case_id=case_id,pmcid=spec['pmc'],protocol=pref,endpoints=len(endpoints),source_ready=True,run_ready=False,protocol_execution_ambiguities=spec['protocol_blockers'],aggregation=spec['aggregation']))
        validations.append(dict(case_id=case_id,source_target_identity_checked=True,exact_cell_and_unit_mapping=True,source_independence_audit_complete_at_declared_scope=True,source_native_compound_and_formulation_exist=True,endpoint_outcomes_absent_from_protocol=True))
        deny.append(dict(pmcid=spec['pmc'],doi=ids['doi'],pmid=ids['pmid'],trial_ids=spec['trials']))
    save(OUT/'evidence_denylist.json',dict(exclude_from_fitting_retrieval_selection=deny))
    save(OUT/'static_validation.json',dict(passed=True,cases=validations,solver_calls=0,gpu_used=False))
    receipt=dict(schema_version='pbpk-independent-curation-receipt-v1',status='five_additional_drugs_source_protocols_and_endpoints_prepared',drugs=5,protocols=5,primary_publications=5,
                 sealed_endpoint_count=sum(r['endpoints'] for r in records),records=records,solver_calls=0,gpu_used=False,strict_main_evaluable_now=0,
                 native_execution_not_claimed=True,script=ref(Path(__file__)))
    save(OUT/'curation_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','drugs','protocols','primary_publications','sealed_endpoint_count']},indent=2))


if __name__=='__main__':main()
