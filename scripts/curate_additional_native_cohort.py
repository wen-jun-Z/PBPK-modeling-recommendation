#!/usr/bin/env python3
"""Bounded, source-auditable acquisition and curation of additional native drugs."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/blueprint_design/external_validation/additional_native_cohort_20260920'
PRIOR = ROOT / 'experiments/blueprint_design/independent_cohort_20260920'
SOURCES = {'pregabalin':'PMC4488182','sildenafil':'PMC10331808',
    'moxifloxacin':'PMC4348051','lisinopril':'PMC7993260','tacrolimus':'PMC13284695',
    'montelukast':'PMC9929805'}
MODELS = {
    'pregabalin':('Pregabalin-Model__47b22b0be8cd/source/Pregabalin.json','capsule'),
    'sildenafil':('Sildenafil-Model__b21240ac0f39/source/Sildenafil-Model.json','Sildenafil Tablet'),
    'moxifloxacin':('Moxifloxacin-Model__51038550e450/source/Moxifloxacin.json','Tablet'),
    'lisinopril':('Lisinopril-model__7fb2806f2a3a/source/Lisinopril-Model.json','Tablet'),
    'tacrolimus':('Tacrolimus-Model__de95cbfaee05/source/Tacrolimus-Model.json','Tacrolimus_ER_Weibull_fasted'),
    'montelukast':('Montelukast-Model__b65eff5296da/source/Montelukast.json','Chewable tablet')}


def sha(data): return hashlib.sha256(data).hexdigest()
def text(e): return ' '.join(' '.join(e.itertext()).split()) if e is not None else ''
def total_bytes():
    return sum(p.stat().st_size for base in [PRIOR, OUT] for p in base.rglob('*') if p.is_file()) + 35071


def save(path, data, sealed=False):
    data = data if isinstance(data,bytes) else (json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if total_bytes() + len(data) > 100*1024**2: raise ValueError('100 MiB cumulative bound')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Refusing to overwrite a changed curation output: '+str(path))
        return {'path':str(path.relative_to(ROOT)),'sha256':sha(data),'byte_size':len(data)}
    with path.open('xb') as f: f.write(data)
    if sealed: os.chmod(path.parent,0o700);os.chmod(path,0o600)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(data),'byte_size':len(data)}


def acquire(provider='europepmc'):
    records=[]
    for drug,pmc in SOURCES.items():
        path=OUT/'sealed/sources'/f'{pmc}.xml'
        url=(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML' if provider=='europepmc'
            else f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc[3:]}')
        if path.exists():
            records.append({'compound':drug,'pmcid':pmc,'url':url,'status':'existing_hashed',
                'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes()),'byte_size':path.stat().st_size});continue
        try:
            with urllib.request.urlopen(url,timeout=30) as response:data=response.read(2*1024**2+1)
            if len(data)>2*1024**2:
                records.append({'compound':drug,'pmcid':pmc,'url':url,'status':'document_oversize_skipped','retry':False});continue
            document=ET.fromstring(data)
            article=document if document.tag=='article' else document.find('.//article')
            if article is None or not {pmc,pmc[3:]} & {e.text for e in article.findall('.//article-id')}:
                raise ValueError('Downloaded article identity mismatch')
            data=ET.tostring(article,encoding='utf-8',xml_declaration=True)
            records.append({'compound':drug,'pmcid':pmc,'url':url,'status':'downloaded',**save(path,data,True)})
        except urllib.error.HTTPError as exc:
            if exc.code==404:
                records.append({'compound':drug,'url':url,'status':'document_not_found_skipped','retry':False});continue
            records.append({'compound':drug,'url':url,'status':'source_error_stopped','error':str(exc)[:200]});break
        except Exception as exc:
            records.append({'compound':drug,'url':url,'status':'source_error_stopped','error':str(exc)[:200]});break
    target=OUT/('acquisition.json' if provider=='europepmc' else 'acquisition_ncbi.json')
    if not target.exists():save(target,{'records':records,'maximum_document_bytes':2*1024**2,
        'maximum_cumulative_bytes':100*1024**2,'cumulative_bytes':total_bytes(),'solver_calls':0})
    print(json.dumps({'source_documents':len(records),'status_counts':{s:sum(r['status']==s for r in records) for s in {r['status'] for r in records}},'cumulative_bytes':total_bytes()}))


def case_specs():
    # Table row numbers count every original tr, including original headers.
    # Values are parsed only from these explicit source coordinates at curation.
    return {
        'pregabalin': dict(suffix='reference_capsule_300mg_fasted', dose=300, formulation='Lyrica 300mg hard capsule',
            native_food='fasted', food='fasted conditions; exact predose fasting duration not explicitly reported', n=39, sex='male', age=[18,45],
            cohort='pmid:25939332:Algorithme_crossover', trials=[],
            times_h=[.25,.33,.5,.67,.83,1,1.25,1.5,1.75,2,3,4,6,9,12,16,24,36],
            aggregation='geometric_mean_of_individual_NCA_endpoints', table='Tab1',
            endpoints=[('Cmax',2,3,'ng/mL',1),('AUC_last',4,3,'ng*h/mL',1),('AUC_infinity',5,3,'ng*h/mL',1)],
            demographics={'enrolled':40,'analyzed':39,'sex_counts_analyzed':{'male':39},'body_weight_mean_kg':None,'age_mean_year':None},
            limitations=['Age and weight means not reported; a source-native reference subject requires a prospective approximation declaration.',
                'Geometric mean is not arithmetic mean; do not use an arithmetic population endpoint scorer.']),
        'sildenafil': dict(suffix='sild_fms_1002_viagra50mg_fasted',dose=50,formulation='Viagra50mg film-coated tablet',
            native_food='fasted',food='fasted overnight at least10h',n=80,sex='male',age=[18,45],
            cohort='study:SILD-FMS-1002',trials=[],study_codes=['SILD-FMS-1002'],
            times_h=[.083,.25,.5,.75,1,1.5,2,3,4,6,8,10,12,14],
            aggregation='arithmetic_mean_of_individual_NCA_endpoints',table='tbl0003',endpoints=[('Cmax',1,2,'ng/mL',1)],
            demographics={'enrolled':80,'analyzed':80,'sex_counts_analyzed':{'male':80},'age_mean_year':31.2,'bmi_mean_kg_m2':23.90,'body_weight_mean_kg':None},
            limitations=['AUC endpoints are not included because original source prints ng/hr/mL, a dimensionally inconsistent AUC unit; no silent repair.',
                'Weight mean not reported; reference-subject weight must be prospectively declared.']),
        'moxifloxacin': dict(suffix='nct01756521_avelox400mg',dose=400,formulation='Avelox400mg tablet',
            native_food='source_default_unknown_external_food',food='not explicitly reported in dosing protocol',n=33,sex='male',age=None,
            cohort='trial:NCT01756521:three_centers',trials=['NCT01756521'],
            times_h=[0,1,2,3,4,6,8,12,16,24],aggregation='arithmetic_mean_of_individual_NCA_endpoints',
            table='t2-dddt-9-1233',endpoints=[('Cmax',2,1,'ng/mL/mg',400),('AUC_last',3,1,'ng*h/mL/mg',400),('AUC_infinity',4,1,'ng*h/mL/mg',400)],
            demographics={'enrolled':38,'analyzed':33,'sex_counts_analyzed':{'male':33},'age_mean_year':26.4,'body_weight_mean_kg':68.3,'height_mean_cm':174.5,'ethnicity':'Korean'},
            limitations=['Table endpoints are explicitly dose-normalized; invert using the reported400mg dose and retain original cells.',
                'External food state not reported; source-default prediction is a declared approximation.',
                'All three centers and400/800mg periods belong to one trial cluster; only400mg is selected before prediction.']),
        'lisinopril': dict(suffix='umin000030894_water_control10mg',dose=10,formulation='Zestril10mg tablet',
            native_food='fasted',food='overnight fast; standardized snack1h and lunch4h postdose',n=10,sex='male_and_female',age=[21,29],
            cohort='trial:UMIN000030894',trials=['UMIN000030894'],times_h=[0,.5,1,2,3,4,6,8,24],
            aggregation='geometric_mean_of_individual_NCA_endpoints',table='cts12905-tbl-0001',
            endpoints=[('Cmax',2,1,'ng/mL',1),('AUC_0_24h',5,1,'ng*h/mL',1),('AUC_infinity',7,1,'ng*h/mL',1)],
            demographics={'enrolled':12,'analyzed':10,'sex_counts_enrolled':{'male':7,'female':5},'sex_counts_analyzed':None,'age_mean_year':None,'body_weight_mean_kg':None},
            limitations=['Water-control monotherapy only; no green-tea extract is included.',
                'Sex counts describe enrolled12, not analyzed10; unknown analyzed mixture cannot be silently reconstructed.',
                'Geometric mean is not arithmetic mean; reference subjects or separately labelled sex scenarios require a geometric-mean comparison contract.']),
        'tacrolimus':dict(suffix='reference1mg_er_fasted',dose=1,formulation='Envarsus XR1mg extended-release tablet',
            native_food='fasted',food='overnight fast at least10h,240mL water',n=85,sex='not_reported',age=[18,45],
            cohort='pmid:42338958:1mg_fasted',trials=[],study_codes=['0094-21','0095-21','0096-21','0097-21'],
            times_h=[0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9,10,11,12,16,24,48,72,96,120,168],
            aggregation='arithmetic_mean_of_observed_period_NCA_endpoints',table='T3',
            endpoints=[('Cmax',3,2,'pg/mL',.001),('AUC_last',7,2,'pg*h/mL',.001),('AUC_infinity',9,2,'pg*h/mL',.001)],
            demographics={'enrolled':90,'analyzed':85,'reference_period_observations':161,'age_mean_year':32.5,'body_weight_mean_kg':62.56,'height_mean_cm':165.27,'bmi_mean_kg_m2':22.86,'sex_counts_analyzed':None},
            limitations=['Whole blood endpoint; plasma predictions are not comparable without a separately validated binding/matrix conversion.',
                '161 reference-period observations are repeated observations from85 subjects, not161 independent participants.',
                'Source ER Weibull formulation is executable but Envarsus formulation equivalence is unverified; report explicit formulation approximation.',
                'Sex not reported; preserve separate sex scenarios or a declared reference subject, never invent a measured sex mixture.',
                'Protocol codes are listed at publication level; individual code-to-arm assignment is not asserted.']),
        'montelukast':dict(suffix='nct05531994_reference5mg_chewable_fed',dose=5,formulation='Singulair5mg chewable tablet',
            native_food='fed',food='high-fat high-calorie breakfast; dose30min after breakfast start',n=30,sex='male_and_female',age=[18,None],
            cohort='trial:NCT05531994',trials=['NCT05528198','NCT05531994'],times_h=[.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,10,12,24],
            aggregation='arithmetic_mean_of_individual_NCA_endpoints',table='t2',
            endpoints=[('Cmax',2,6,'ng/mL',1),('AUC_last',3,6,'ng*h/mL',1),('AUC_infinity',4,6,'ng*h/mL',1)],
            demographics={'enrolled':30,'analyzed':30,'sex_counts_analyzed':{'male':24,'female':6},'age_mean_year':26.0,'body_weight_mean_kg':61.5,'height_mean_cm':166.5,'bmi_mean_kg_m2':22.1},
            limitations=['Fed reference chewable-tablet arm only; do not combine with fasting participants or soluble-film periods.',
                'Publication contains an impossible fasting-weight cell duplicating BMI; it is not used or repaired.',
                'Native source formulation/food transfer must be declared; reference-subject endpoints do not reproduce individual variability.']),
    }


def curate():
    from pbpk_kb.blueprints.human_model_snapshot import _PROCESSES, _MOLECULAR_PROCESS_PARAMETERS
    supported=set(_PROCESSES)|set(_MOLECULAR_PROCESS_PARAMETERS)
    inventory=json.loads((PRIOR/'cohort_inventory.json').read_text())
    model_rows={d['compound']:d for d in inventory['drug_inventory']}
    specs=case_specs();records=[];deny=[]
    for drug,spec in specs.items():
        pmc=SOURCES[drug];source=OUT/'sealed/sources'/(pmc+'.xml');payload=source.read_bytes();article=ET.fromstring(payload)
        if article.find('body') is None:raise ValueError('Primary body absent: '+pmc)
        ids={e.get('pub-id-type'):e.text for e in article.findall('./front/article-meta/article-id')}
        identities={k:ids.get(k) for k in ['pmid','pmcid','doi']}
        assert identities['pmcid']==pmc
        table=next(t for t in article.findall('.//table-wrap') if t.get('id')==spec['table'])
        rows=[[text(c) for c in r] for r in table.findall('.//tr')]
        values=[]
        for endpoint,row,col,rawunit,multiplier in spec['endpoints']:
            cell=rows[row][col]
            match=re.match(r'\s*([0-9,]+(?:\.[0-9]+)?)',cell)
            if not match:raise ValueError('Source endpoint cell is not a positive scalar')
            original=float(match[1].replace(',',''))
            if original<=0:raise ValueError('Invalid clinical endpoint')
            unit='ng/mL' if endpoint=='Cmax' else 'ng*h/mL'
            values.append({'endpoint':endpoint,'value':original*multiplier,'unit':unit,
                'aggregation':spec['aggregation'],'source_table_id':spec['table'],'source_all_tr_row':row,
                'source_physical_column':col,'source_cell':cell,'source_value':original,'source_unit':rawunit,
                'conversion_multiplier':multiplier,'conversion_rule':('invert published dose-normalization by400mg' if drug=='moxifloxacin'
                    else 'pg to ng exactly1e-3' if drug=='tacrolimus' else 'identity'),
                'source_rows_including_headers':rows,'source_footnotes':text(table.find('table-wrap-foot'))})
        case_id=drug+'_'+pmc.lower()+'_'+spec['suffix']
        sealed=save(OUT/'sealed'/ (case_id+'.json'),{'case_id':case_id,'study_id':'pmid:'+ids['pmid'],
            'cohort_id':spec['cohort'],'identifiers':identities,'n_unique_participants':spec['n'],
            'observed_endpoints':values,'source_article':{'path':str(source.relative_to(ROOT)),'sha256':sha(payload)},
            'not_available_to_designer':True},True)
        model_rel,formulation=MODELS[drug]
        model_path=ROOT/'artifacts/pbpk-model-assets/public_collection_20260909'/model_rel
        snapshot=json.loads(model_path.read_text());compound=snapshot['Compounds'][0]
        if len(snapshot['Compounds'])!=1 or formulation not in {f['Name'] for f in snapshot['Formulations']}:
            raise ValueError('Native one-compound/formulation identity contract changed')
        unsupported=sorted({p['InternalName'] for p in compound.get('Processes',[]) if p['InternalName'] not in supported})
        if unsupported or any(p.get('Metabolite') for p in compound.get('Processes',[])):
            raise ValueError('Unsupported native process or named metabolite: '+drug+str(unsupported))
        tokens=[v.lower() for v in identities.values() if v]+[v.lower() for v in spec['trials']]+[s.lower() for s in spec.get('study_codes',[]) if len(s)>8]
        overlaps=[];inspected=[]
        for model in model_rows[drug]['models']:
            path=ROOT/model['snapshot_path'];s=json.loads(path.read_text())
            observed=' '.join(str(v.get('Value','')) for o in s.get('ObservedData',[]) for v in o.get('ExtendedProperties',[])).lower()
            matches=[t for t in tokens if t in observed]
            if matches:overlaps.append({'path':str(path.relative_to(ROOT)),'scope':'snapshot observed study labels','matches':matches})
            for doc in path.parent.rglob('*.md'):
                body=doc.read_text().lower();matches=[t for t in tokens if t in body]
                inspected.append({'path':str(doc.relative_to(ROOT)),'sha256':sha(doc.read_bytes())})
                if matches:overlaps.append({'path':str(doc.relative_to(ROOT)),'scope':'repository model documentation','matches':matches})
        papers=(['PMC6765696','PMC9297990'] if drug=='moxifloxacin' else ['PMC10196430'] if drug=='tacrolimus' else [])
        paper_checks=[]
        for paper in papers:
            path=OUT/'source_audits/model_papers'/(paper+'.xml');a=ET.parse(path).getroot()
            if a.find('body') is None:raise ValueError('Model original paper body unavailable')
            t=text(a).lower();matches=[token for token in tokens if token in t]
            paper_checks.append({'pmcid':paper,'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes()),
                'references_checked':len(a.findall('.//ref-list/ref')),'target_identifiers_found':matches})
            if matches:overlaps.append({'path':str(path.relative_to(ROOT)),'scope':'original model paper fulltext and citations','matches':matches})
        if overlaps:raise ValueError('Clinical-source overlap requires adjudication: '+drug)
        audit=save(OUT/'source_audits'/(drug+'.json'),{'compound':drug,'target_identifiers':identities,
            'target_trial_ids':spec['trials'],'all_registered_snapshots_for_compound_checked':len(model_rows[drug]['models']),
            'repository_source_documents_checked':inspected,'original_model_papers_checked':paper_checks,
            'direct_model_source_description':('Repository publishes explicit original clinical source list; no separate original model research paper is linked.' if not papers else 'Registered source report and original PBPK model papers checked.'),
            'overlaps':overlaps,'clinical_source_disjoint_at_audited_source_level':True,
            'generic_physiology_reference_sharing_is_not_clinical_overlap':True,
            'limitations':'Public-source identity audit cannot establish an absolute guarantee of no undocumented participant reuse.'})
        protocol={'schema_version':'independent-clinical-protocol-4','case_id':case_id,'compound':drug,
            'study_id':'pmid:'+ids['pmid'],'publication_cluster_id':'pmid:'+ids['pmid'],
            'cohort_id':spec['cohort'],'identifiers':identities,'tier':'independent_clinical_candidate',
            'status':'source_disjoint_protocol_and_sealed_endpoints_prepared','source_ready':True,'run_ready':False,
            'n_evaluable':spec['n'],'protocol':{'population':'healthy_adult','sex':spec['sex'],'age_range_year':spec['age'],
                'route':'oral','dose':spec['dose'],'dose_unit':'mg','dose_basis':'reported labelled drug strength; no silent salt mass rescaling',
                'administrations':[{'time_min':0,'dose':spec['dose'],'dose_unit':'mg'}],
                'formulation':spec['formulation'],'food_state':spec['food'],'sampling_times_min':[t*60 for t in spec['times_h']],
                'sampling_time_origin':'administration of selected monotherapy/reference formulation',
                'reported_demographics':spec['demographics'],'trial_registry_ids':spec['trials'],'study_codes':spec.get('study_codes',[])},
            'evaluation_target':{'matrix':'whole_blood' if drug=='tacrolimus' else 'plasma','analyte':'parent','binding':'total',
                'endpoints':[v['endpoint'] for v in values],'aggregation':spec['aggregation'],
                'concentration_curve_evaluable':False,'AUC_last_definition':'last individual quantifiable concentration, not automatically scheduled final sample',
                'AUC_0_24h_definition':'fixed24h window' if drug=='lisinopril' else None,
                'AUC_infinity_definition':'individual NCA total AUC extrapolated to infinity',
                'Cmax_definition':'maximum sampled individual concentration, then reported summary',
                'unit_mappings':[{k:v[k] for k in ['endpoint','source_unit','unit','conversion_multiplier','conversion_rule']} for v in values]},
            'native_binding':{'source_snapshot':{'path':str(model_path.relative_to(ROOT)),'sha256':sha(model_path.read_bytes())},
                'compound_name':compound['Name'],'formulation_name':formulation,
                'process_types':sorted({p['InternalName'] for p in compound.get('Processes',[])}),
                'current_native_adapter_unsupported_types':unsupported,'named_metabolites':[],
                'source_food_binding':spec['native_food'],'original_native_build_executed_in_this_curation':False},
            'source_audit':audit,'source_article_sha256':sha(payload),'sealed_observation_reference':sealed,
            'evidence_exclusion':{'identifiers':identities,'trial_registry_ids':spec['trials'],
                'source_files':[str(source.relative_to(ROOT))], 'exclude_entire_publication_and_cohort':True},
            'prediction_contract':{'freeze_protocol_and_model_selection_before_prediction':True,
                'freeze_prediction_hashes_before_opening_observations':True,'source_reference_subject_prediction_allowed':True,
                'missing_covariates_must_be_declared_approximations':True,'group_statistic_must_match_or_be_labelled_approximation':True,
                'do_not_reuse_arithmetic_scorer_for_geometric_means':True},
            'limitations':spec['limitations'],'strict_main_evaluable_now':False,'scientific_results_accessed_by_algorithm':False}
        reference=save(OUT/'protocols'/(case_id+'.json'),protocol)
        records.append({'compound':drug,'case_id':case_id,'protocol':reference,'endpoints':len(values),'source_ready':True,
            'run_ready':False,'aggregation':spec['aggregation'],'native_types_supported':True})
        deny.append({'identifiers':identities,'cohort_id':spec['cohort'],'trial_registry_ids':spec['trials'],
            'source_files':[str(source.relative_to(ROOT))]})
    save(OUT/'evidence_denylist.json',{'publications':deny,'exclude_from':['fitting','retrieval','designer prompts','selection','training'],
        'all_protocols_share_publication_clusters_across_formulations':True})
    save(OUT/'curation_receipt.json',{'status':'six_source_disjoint_drugs_prepared','drugs':6,'protocols':len(records),
        'primary_publications':6,'records':records,'sealed_endpoint_count':sum(r['endpoints'] for r in records),
        'native_support_qualification':'All parent-process types/formulation names exist in native source adapter; actual new builds remain to be verified.',
        'solver_calls':0,'gpu_used':False,'strict_main_evaluable_now':0,'cumulative_acquisition_bytes':total_bytes(),
        'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':sha(Path(__file__).read_bytes())})
    print(json.dumps({'drugs':6,'protocols':len(records),'sealed_endpoints':sum(r['endpoints'] for r in records),
        'native_supported_drugs':sum(r['native_types_supported'] for r in records),'solver_calls':0,'cumulative_bytes':total_bytes()}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--acquire',action='store_true');parser.add_argument('--curate',action='store_true');parser.add_argument('--provider',choices=['europepmc','ncbi'],default='europepmc');args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if args.acquire:acquire(args.provider)
    if args.curate:curate()


if __name__=='__main__':main()
