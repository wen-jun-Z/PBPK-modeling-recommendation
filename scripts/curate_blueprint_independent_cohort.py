#!/usr/bin/env python3
"""Audit clinical source separation and prepare sealed endpoint evaluation data.

No algorithm, model fitting, native solver, GPU, credentials or existing-file
mutation. Clinical publication/cohort separation is distinguished from model
runtime readiness. Historical model curves can never become independent data.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json'
RUNS = ('run_20260907T134918Z', 'run_20260909T083120Z')
MAX_SOURCE_BYTES = 32 * 1024**2
MAX_NEW_BYTES = 100 * 1024**2
ALIASES = {'acetaminophen': 'paracetamol', 'rifampin': 'rifampicin', 'ciclosporin': 'cyclosporine'}
SELECTED = ('PMC12378308', 'PMC10897852', 'PMC12394267', 'PMC8416862')


def text(e):
    return ' '.join(''.join(e.itertext()).split()) if e is not None else ''


def canonical(name):
    value = str(name).strip().lower()
    return ALIASES.get(value, value)


@lru_cache(maxsize=8192)
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inside(path):
    path = Path(path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('All curator paths must remain inside the project.')
    return path


def identities(value):
    value = str(value)
    return {
        'pmid': sorted(set(re.findall(r'(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/|pubmed[/: =]+|PMID[/: =]+)(\d{6,9})', value, re.I))),
        'doi': sorted({m.rstrip('.,;)]}').lower() for m in re.findall(r'10\.\d{4,9}/[^\s<>"\\]+', value, re.I)}),
        'trial': sorted(set(re.findall(r'\b(?:NCT\d{8}|CTR\d{8}|ISRCTN\d{8}|EudraCT\s*\d{4}-\d{6}-\d{2})\b', value, re.I))),
    }


def article_ids(article):
    ids = {e.attrib.get('pub-id-type'): e.text for e in article.findall('./front/article-meta/article-id')}
    return {'pmid': ids.get('pmid'), 'pmcid': ids.get('pmcid', ids.get('pmc')), 'doi': ids.get('doi')}


class Writer:
    def __init__(self, output):
        self.output = inside(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.bytes = 0

    def raw(self, relative, data, sealed=False):
        if self.bytes + len(data) > MAX_NEW_BYTES:
            raise ValueError('100 MiB total output cap exceeded.')
        p = inside(self.output / relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        if sealed:
            os.chmod(p.parent, 0o700)
        with p.open('xb') as f:
            f.write(data)
        if sealed:
            os.chmod(p, 0o600)
        self.bytes += len(data)
        return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'byte_size': len(data)}

    def json(self, relative, value, sealed=False):
        return self.raw(relative, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode(), sealed)


def source_models():
    inventory = json.loads(INVENTORY.read_text())
    drugs = {}
    repository_ids = {}
    skipped = []
    single_names = {canonical(e['compound_names'][0]) for e in inventory['entries'] if e['compound_count'] == 1}
    for entry in inventory['entries']:
        p = inside(ROOT / entry['path'])
        if p.stat().st_size > MAX_SOURCE_BYTES:
            skipped.append({'path': entry['path'], 'reason': 'per-source size limit'})
            continue
        if sha(p) != entry['sha256']:
            raise ValueError('Model source inventory checksum changed: '+str(p))
        snapshot = json.loads(p.read_text())
        repo = p.parent
        if str(repo) not in repository_ids:
            docs = list(repo.rglob('*.md'))
            doc_records = []
            for doc in docs:
                if doc.stat().st_size > MAX_SOURCE_BYTES:
                    continue
                content = doc.read_text(errors='replace')
                doc_records.append({'path': str(doc.relative_to(ROOT)), 'sha256': sha(doc), 'identifiers': identities(content)})
            repository_ids[str(repo)] = doc_records
        for name in entry['compound_names']:
            key = canonical(name)
            drug = drugs.setdefault(key, {'compound': key, 'display_name': name, 'in_52_single_compound_sources': key in single_names,
                'models': [], 'historical_clinical_studies': [], 'pkdb_studies': [], 'existing_evidence_sources': [], 'primary_article_candidates': []})
            drug['models'].append({'snapshot_path': entry['path'], 'snapshot_sha256': entry['sha256'], 'compound_count': entry['compound_count'],
                'reference_documents': repository_ids[str(repo)]})
            for observed in snapshot.get('ObservedData', []):
                meta = {r['Name']: r.get('Value') for r in observed.get('ExtendedProperties', [])}
                molecule = meta.get('Molecule')
                if molecule and canonical(molecule) != key:
                    continue
                if not molecule and entry['compound_count'] != 1:
                    continue
                ids = identities(meta.get('Reference', ''))
                label = str(meta.get('Study Id', observed.get('Name', '')))
                role = 'used_by_public_model_role_not_distinguished'
                # Source authors' clinical table headings, not our performance, determine roles.
                for doc in repo.rglob('Section2.2_Data.md'):
                    mode = None
                    for line in doc.read_text(errors='replace').splitlines():
                        low = line.lower()
                        if 'model building' in low and ('following' in low or low.startswith('#')):
                            mode = 'model_building'
                        if 'model verification' in low or 'model validation' in low:
                            mode = 'model_verification'
                        if mode and re.sub(r'[^a-z0-9]', '', label.lower()) in re.sub(r'[^a-z0-9]', '', low) and label:
                            role = mode
                columns = observed.get('Columns', [])
                authentic = [c for c in columns if c.get('DataInfo', {}).get('Origin') == 'Observation']
                drug['historical_clinical_studies'].append({'source_study_label': label, 'source_observation_name': observed.get('Name'),
                    'identifiers': ids, 'reference': meta.get('Reference'), 'model_use': role,
                    'species': meta.get('Species'), 'route': meta.get('Route'), 'matrix': meta.get('Compartment'),
                    'data_type': meta.get('Data type'), 'cohort': meta.get('Group', meta.get('Population')),
                    'cohort_description': observed.get('Name'), 'observed_series_present': bool(authentic and observed.get('BaseGrid', {}).get('Values')),
                    'observed_point_count': len(observed.get('BaseGrid', {}).get('Values', [])) if authentic else 0,
                    'snapshot_path': entry['path'], 'independent_candidate': False,
                    'independence_reason': 'Already registered in this public model; never re-label as independent.'})
    return drugs, skipped, len(inventory['entries']), len(single_names)


def ensure_drug(drugs, name):
    key = canonical(name)
    return drugs.setdefault(key, {'compound': key, 'display_name': name, 'in_52_single_compound_sources': False,
        'models': [], 'historical_clinical_studies': [], 'pkdb_studies': [], 'existing_evidence_sources': [], 'primary_article_candidates': []})


def local_evidence(drugs):
    xml_sources = {}
    for run in RUNS:
        p = ROOT / f'data_process/final_evidence/{run}/07_llm_quality_validation/final_auto_validated_evidence.jsonl'
        for line in p.open():
            row = json.loads(line)
            drug = ensure_drug(drugs, row['compound'])
            if row.get('source_id') == 'pmc_oa' and row.get('storage_path', '').endswith('.xml'):
                path = inside(row['storage_path'])
                xml_sources.setdefault(path, set()).add(drug['compound'])
            drug['existing_evidence_sources'].append({'evidence_id': row['evidence_id'], 'source_id': row.get('source_id'),
                'source_location': row.get('source_location'), 'source_url': row.get('source_url'),
                'storage_path': str(inside(row['storage_path']).relative_to(ROOT)) if row.get('storage_path') else None,
                'parameter_type': row.get('parameter'), 'route': row.get('route'), 'population': row.get('population'),
                'independent_clinical_eligibility': 'unadjudicated_not_inferred_from_auto_quality_label'})
    articles = {}
    for path, retrieval_drugs in sorted(xml_sources.items()):
        if path.stat().st_size > MAX_SOURCE_BYTES:
            continue
        for article in ET.parse(path).findall('.//article'):
            ids = article_ids(article)
            if not ids['pmcid']:
                continue
            title = text(article.find('./front/article-meta/title-group/article-title'))
            record = {'identifiers': ids, 'title': title,
                'article_type': article.attrib.get('article-type'), 'source_file': str(path.relative_to(ROOT)),
                'source_file_sha256': sha(path), 'tables': [{'id': t.attrib.get('id'), 'caption': text(t.find('caption'))} for t in article.findall('.//table-wrap')],
                'has_pk_table_candidate': any(re.search(r'pharmacokinetic|plasma PK', text(t.find('caption')), re.I) for t in article.findall('.//table-wrap')),
                'healthy_in_title': bool(re.search(r'healthy|volunteers', title, re.I)),
                'primary_human_status': 'requires_body_protocol_and_outcome_review',
                'clinical_source_separation': 'not_proven_by_title_or_publication_date'}
            articles.setdefault(ids['pmcid'], (article, record))
            for drug in retrieval_drugs:
                # Retrieval query alone does not establish that this article measured this drug.
                if not any(x['identifiers']['pmcid'] == ids['pmcid'] for x in drugs[drug]['primary_article_candidates']):
                    drugs[drug]['primary_article_candidates'].append(record)
    for run in RUNS:
        base = ROOT / f'artifacts/pbpk_raw_runs/compound_runs/{run}/compounds'
        if not base.exists():
            continue
        for directory in base.iterdir():
            if not directory.is_dir():
                continue
            members = directory / 'pkdb/official_bulk_export/pkdb/open_bulk_export_inspection/members'
            if not (members / 'studies.csv').is_file():
                continue
            drug = ensure_drug(drugs, directory.name)
            tables = {}
            for name in ('studies', 'groups', 'interventions', 'outputs', 'timecourses'):
                with (members / (name+'.csv')).open() as f:
                    tables[name] = list(csv.DictReader(f))
            drug.setdefault('pkdb_exports', []).append({'path': str(members.relative_to(ROOT)), 'counts': {k: len(v) for k,v in tables.items()},
                'limitation': 'No outcome data in export' if not tables['outputs'] and not tables['timecourses'] else None})
            for study in tables['studies']:
                sid = study['sid']
                groups = [r for r in tables['groups'] if r.get('study_sid') == sid]
                interventions = [r for r in tables['interventions'] if r.get('study_sid') == sid]
                drug['pkdb_studies'].append({'pkdb_sid': sid, 'study_name': study['name'], 'pmid': study.get('reference_pmid'),
                    'title': study.get('reference_title'), 'licence': study.get('licence'),
                    'cohort_ids': sorted({r.get('group_pk') for r in groups if r.get('group_pk')}),
                    'routes': sorted({r.get('route_label',r.get('route','')) for r in interventions}),
                    'healthy_adult': 'requires_group_characteristics_review',
                    'observed_timecourse_rows': sum(r.get('study_sid')==sid for r in tables['timecourses']),
                    'observed_output_rows': sum(r.get('study_sid')==sid for r in tables['outputs']),
                    'source': str((members/'studies.csv').relative_to(ROOT)), 'source_sha256': sha(members/'studies.csv')})
    return articles


def table_rows(article, table_id):
    table = next(t for t in article.findall('.//table-wrap') if t.attrib.get('id') == table_id)
    return [[text(c) for c in r] for r in table.findall('.//tbody/tr')]


def mean_cell(value):
    match = re.match(r'\s*([0-9]+(?:[,.][0-9]+)*)', value)
    if not match:
        raise ValueError('Expected a published mean at the start of an outcome cell.')
    result = float(match[1].replace(',', ''))
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Nonpositive/nonfinite published endpoint.')
    return result


def overlap_audit(drugs, ids, registry_ids, cohort_note):
    hits = []
    for drug in drugs.values():
        for model in drug['models']:
            for doc in model['reference_documents']:
                d = doc['identifiers']
                if ((ids['pmid'] and ids['pmid'] in d['pmid']) or (ids['doi'] and ids['doi'].lower() in d['doi']) or set(registry_ids)&set(d['trial'])):
                    hits.append({'compound':drug['compound'], 'path':doc['path'], 'type':'model_source_document_reference'})
        for study in drug['historical_clinical_studies']:
            d = study['identifiers']
            if (ids['pmid'] and ids['pmid'] in d['pmid']) or (ids['doi'] and ids['doi'].lower() in d['doi']):
                hits.append({'compound':drug['compound'], 'path':study['snapshot_path'], 'type':'historical_model_observation'})
    return {'source_identifiers':ids, 'trial_registry_ids':registry_ids, 'model_source_overlap':hits,
        'clinical_source_disjoint_in_registered_model_corpus':not hits,
        'cohort_overlap_audit':cohort_note,
        'scope':'All registered model snapshot observations and repository reference documents. Baseline physiological literature sharing is not a clinical overlap.'}


def prepare_case(writer, drugs, articles, pmcid, compound, suffix, protocol, endpoints, table_id, n, cohort_note, exclusions=()):
    article, rec = articles[pmcid]
    ids = rec['identifiers']
    # Trial IDs are identities only; no outcome paragraph enters prediction metadata.
    registries = identities(text(article))['trial']
    audit = overlap_audit(drugs, ids, registries, cohort_note)
    rows = table_rows(article, table_id)
    outcome = []
    for endpoint, row_i, column_i, label_row, units in endpoints:
        source_label = rows[label_row][0]
        if ('Cmax' if endpoint=='Cmax' else 'AUC') not in source_label:
            raise ValueError('Registered endpoint label/table row changed.')
        cell = rows[row_i][column_i]
        outcome.append({'endpoint':endpoint, 'value':mean_cell(cell), 'unit':units, 'aggregation':'arithmetic_mean',
            'source_table_id':table_id,'source_row':row_i,'source_column':column_i,'source_endpoint_label':source_label,'source_cell_text':cell,
            'measurement_basis':'individual observed concentration profiles followed by reported noncompartmental summary',
            'curve_values_available':False})
    case_id = compound+'_'+pmcid.lower()+'_'+suffix
    sealed = writer.json('sealed/'+case_id+'.json',{'case_id':case_id,'study_id':'pmid:'+ids['pmid'],
        'identifiers':ids,'cohort_id':protocol['cohort_id'],'n_evaluable':n,'observed_endpoints':outcome,
        'raw_article_reference':rec['sealed_article_reference'],'not_available_to_designer':True},sealed=True)
    evidence_exclusion = {'identifiers':ids,'trial_registry_ids':registries,
        'raw_source_files':[rec['source_file']],
        'exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training':True,
        'note':'A raw multi-article XML batch is conservatively excluded as a whole until an article-aware evidence mask is enforced.'}
    status = 'source_disjoint_endpoint_package_prepared' if audit['clinical_source_disjoint_in_registered_model_corpus'] else 'blocked_model_source_overlap'
    case = {'schema_version':'independent-clinical-protocol-1','case_id':case_id,'compound':compound,
        'study_id':'pmid:'+ids['pmid'],'cohort_id':protocol['cohort_id'],'identifiers':ids,'title':rec['title'],
        'tier':'independent_clinical_candidate','status':status,'clinical_source_audit':audit,'protocol':protocol,'n_evaluable':n,
        'evaluation_target':{'matrix':'plasma','analyte':'parent','binding':'total','endpoints':[e[0] for e in endpoints],
            'aggregation':'arithmetic_mean_of_individual_PK_endpoints','concentration_curve_evaluable':False,
            'AUC_last_semantics':'last quantifiable sample per individual, not automatically fixed end-of-sampling time',
            'Cmax_semantics':'maximum sampled individual concentration, then published group mean'},
        'evidence_exclusion':evidence_exclusion,'sealed_observation_reference':sealed,
        'source_article_sha256':rec['sealed_article_reference']['sha256'],'scientific_results_accessed_by_algorithm':False,
        'runtime_readiness':'not_executed_requires_registered_compound_and_formulation_adapter',
        'required_before_scoring':['Freeze source-denylist enforcement before generation or calibration.',
            'Register matching formulation, food state, active-moiety dose and population.',
            'Lock predictions before opening sealed outcomes.',
            'Match individual-NCA mean target; a single reference-subject curve is a labelled approximation, not a population prediction.'],
        'limitations':list(exclusions),'strict_main_evaluable_now':False}
    ref = writer.json('prediction_metadata/'+case_id+'.json',case)
    ensure_drug(drugs,compound).setdefault('prepared_independent_candidates',[]).append({'case_id':case_id,'status':status,'protocol_reference':ref})
    return case


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--refresh-selected',action='store_true');args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    writer=Writer(ROOT/args.output)
    drugs,skipped,snapshot_count,single_count=source_models()
    articles=local_evidence(drugs)
    fetches=[];failed_source=False
    for pmcid in SELECTED:
        article,rec=articles[pmcid]
        payload=ET.tostring(article,encoding='utf-8',xml_declaration=True)
        if args.refresh_selected and not failed_source:
            url='https://www.ebi.ac.uk/europepmc/webservices/rest/'+pmcid+'/fullTextXML'
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'PBPK-clinical-source-curator/1.0 (public academic data)'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    downloaded=response.read(2*1024**2+1)
                if len(downloaded)>2*1024**2: raise ValueError('2 MiB per public article download limit')
                fetched=ET.fromstring(downloaded)
                if fetched.tag!='article': fetched=fetched.find('.//article')
                if fetched is None or article_ids(fetched)!=rec['identifiers']: raise ValueError('Refreshed primary article identity mismatch')
                article=fetched;payload=downloaded;articles[pmcid]=(article,rec)
                fetches.append({'pmcid':pmcid,'url':url,'status':'downloaded_identity_verified','bytes':len(payload)})
            except Exception as error:
                failed_source=True
                fetches.append({'pmcid':pmcid,'url':url,'status':'source_stopped_after_error_use_existing_verified_article','error_type':type(error).__name__,'error':str(error)[:300]})
        rec['sealed_article_reference']=writer.raw('sealed/sources/'+pmcid+'.xml',payload,sealed=True)
    cases=[]
    for compound,dose,cohort,n,rows in [('atorvastatin',80,'cohort_1',40,(1,3,5)),('rosuvastatin',40,'cohort_2',31,(22,24,26))]:
        times=([0,.25,.5,.75,1,1.33,1.67,2,2.5,3,3.5,4,5,6,7,9,12,16,24,36,48,72,96,120] if compound=='atorvastatin' else [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,9,12,16,24,36,48,72,96,120])
        cases.append(prepare_case(writer,drugs,articles,'PMC12378308',compound,'monotherapy_day_minus4',
            {'cohort_id':'pmid:40569335:'+cohort,'population':'healthy_adult','age_range_year':[18,65], 'sex':'male_and_female',
             'ethnicity':'non_Asian_required' if compound=='rosuvastatin' else 'not_restricted_in_reported_inclusion',
             'route':'oral','dose':dose,'dose_unit':'mg','dose_basis':'label_active_drug_as_calcium_salt',
             'formulation':'Lipitor tablet' if compound=='atorvastatin' else 'Crestor tablet','food_state':'fed_standard_breakfast_30_minutes_before_dose',
             'administrations':[{'time_min':0,'dose':dose,'dose_unit':'mg'}],'concomitant_study_drug':'none_at_day_minus4_baseline',
             'sampling_times_min':[t*60 for t in times], 'sampling_time_origin':'baseline statin dose on study day minus4',
             'observation_duration_min':7200,'source_protocol_location':'Methods study design and blood sampling'},
            [('Cmax',rows[0],2,rows[0],'ng/mL'),('AUC_last',rows[1],2,rows[1],'ng*h/mL'),('AUC_infinity',rows[2],2,rows[2],'ng*h/mL')],
            'Tab2',n,'Two explicitly distinct sequentially recruited cohorts; same paper retained as one publication cluster, no evidence of cohort reuse in registered development sources.',
            ['Single-drug baseline only; no obicetrapib combination outcomes used.', 'Same publication cannot be split across development and evaluation.']))
    cases.append(prepare_case(writer,drugs,articles,'PMC10897852','paracetamol','reference_suspension',
        {'cohort_id':'pmid:38414712:healthy_adult_crossover','population':'healthy_adult','age_range_year':[18,45],'sex':'male_and_female',
         'country':'Czech Republic','study_dates':['2020-08-23','2021-09-08'],'route':'oral','dose':1000,'dose_unit':'mg',
         'dose_basis':'paracetamol','formulation':'marketed reference oral suspension 42 mL','food_state':'fasted',
         'administrations':[{'time_min':0,'dose':1000,'dose_unit':'mg'}], 'crossover_washout_hours':72,
         'sampling_times_min':[5,10,20,30,40,50,60,80,90,120,150,180,240,300,360,480,600,720,840,960],
         'predose_sample_min':-60,'quantification_limit':{'value':.25,'unit':'ug/mL'},'sampling_time_origin':'oral reference suspension dose',
         'observation_duration_min':960,'source_protocol_location':'Materials and Methods / Study protocol'},
        [('Cmax',5,2,3,'ug/mL'),('AUC_last',20,2,18,'ug*h/mL')],'tbl0003',35,
        'One Czech adult crossover cohort, trial dates given; test and reference are the same people and only reference is retained.',
        ['No numeric concentration curve is supplied; only published observed NCA endpoints.',
         'AUC infinity omitted because the printed table unit is dimensionally inconsistent; no silent repair.',
         'AUC_last is not automatically equivalent to AUC_0_16h.']))
    for food,col in [('fasting',1),('fed',2)]:
        times=[.25,.5,1,1.5,1.75,2,2.25,2.5,3,4,5,6,8,12]+([16] if food=='fasting' else [])+[24]
        cases.append(prepare_case(writer,drugs,articles,'PMC12394267','metformin','reference_'+food,
            {'cohort_id':'trial:CTR20180793:'+food,'population':'healthy_adult','age_range_year':[18,45],'sex':'male_and_female',
             'country':'China','route':'oral','dose':250,'dose_unit':'mg','dose_basis':'metformin_hydrochloride_salt',
             'formulation':'Glycoran reference tablet','food_state':'fasted' if food=='fasting' else 'fed_30_minutes_after_meal',
             'administrations':[{'time_min':0,'dose':250,'dose_unit':'mg','dose_basis':'metformin_hydrochloride_salt'}],
             'crossover_washout_days':7,'sampling_times_min':[t*60 for t in times], 'predose_sample_min':-30,
             'sampling_time_origin':'reference tablet dose','observation_duration_min':1440,
             'source_protocol_location':'Methods / study design fasting and fed trial'},
            [('Cmax',8,col,8,'ng/mL'),('AUC_last',9,col,9,'ng*h/mL'),('AUC_infinity',10,col,10,'ng*h/mL')], 'Tab1',26,
            'Trial CTR20180793 has separately described fasting and fed groups; they remain one publication/trial cluster, not independent drugs.',
            ['Salt-labelled dose must be converted or natively declared, never silently interpreted as free base.',
             'Only reference formulation retained; test/reference periods share participants.',
             'Complete subject IDs are unavailable; no evidence of overlap with development was found, which is an auditable source-level conclusion rather than a guarantee.']))
    cases.append(prepare_case(writer,drugs,articles,'PMC8416862','furosemide','trial2_baseline',
        {'cohort_id':'pmid:33782832:trial2','population':'healthy_adult','age_range_year':[18,65],'sex':'male_and_female',
         'country':'United Kingdom','route':'oral','dose':40,'dose_unit':'mg','dose_basis':'furosemide','formulation':'oral tablet',
         'food_state':'requires_supplement_protocol_confirmation','administrations':[{'time_min':0,'dose':40,'dose_unit':'mg'}],
         'observation_duration_min':720,'sampling_time_origin':'single furosemide dose','concomitant_study_drug':'none_in_baseline_period',
         'source_protocol_location':'Methods / Trial2; first baseline drug, preceding rosuvastatin by 24 h'},
        [('AUC_infinity',2,1,0,'ng*h/mL'),('Cmax',5,1,3,'ng/mL')], 'Tab4',39,
        'Trial2 is the same cohort as this paper rosuvastatin baseline; publication/cohort retained together. No duplicate baseline subset is added.',
        ['Food state and complete sampling times need supplementary-method extraction before execution.',
         'The table baseline subset n=39 is used once; n=40 reanalysis is not a second study.']))
    for drug in drugs.values():
        drug['independent_source_ready_case_count']=sum(x['status']=='source_disjoint_endpoint_package_prepared' for x in drug.get('prepared_independent_candidates',[]))
        if not drug['independent_source_ready_case_count']:
            drug['current_gap']='No source-disjoint clinical outcome package adjudicated yet; model curves are historical and raw paper candidates require protocol/outcome review.'
    report={'schema_version':'clinical-cohort-inventory-1','source_snapshot_count':snapshot_count,'single_compound_source_name_count':single_count,
        'total_covered_source_names':len(drugs),'source_names_are_identity_verified_independent_drug_count':False,
        'single_compound_source_names':sorted(d['compound'] for d in drugs.values() if d['in_52_single_compound_sources']),
        'target_gap':'52 single-compound source names cannot by themselves satisfy 12 development plus 60 disjoint evaluation drugs; this inventory also includes salts, metabolites and placeholders, which are not extra independent drugs.',
        'model_scope':'All registered source inventory snapshots plus their source markdown references',
        'source_inventory':{'path':str(INVENTORY.relative_to(ROOT)),'sha256':sha(INVENTORY)},'large_sources_not_opened':skipped,
        'drug_inventory':sorted(drugs.values(),key=lambda r:r['compound']),
        'prepared_cases':[{'case_id':c['case_id'],'compound':c['compound'],'study_id':c['study_id'],'status':c['status'],
            'clinical_source_disjoint':c['clinical_source_audit']['clinical_source_disjoint_in_registered_model_corpus'],
            'strict_main_evaluable_now':False} for c in cases],
        'source_ready_compounds':sorted({c['compound'] for c in cases if c['clinical_source_audit']['clinical_source_disjoint_in_registered_model_corpus']}),
        'all_observed_outcome_values_sealed':True,'solver_calls':0,'gpu_used':False,'public_downloads':fetches,
        'important_boundary':'Source-ready is not run-ready. Denylist, drug/formulation registration, population target, and prediction locking remain mandatory; source physiology shared across drugs is not itself a disqualifier.'}
    writer.json('cohort_inventory.json',report)
    writer.json('evidence_denylist.json',{'publications':[c['evidence_exclusion'] for c in cases],
        'enforcement_required':True,'actual_algorithm_enforcement_performed':False})
    writer.json('curation_manifest.json',{'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':sha(Path(__file__)),
        'files_bytes_before_manifest':writer.bytes,'maximum_new_bytes':MAX_NEW_BYTES,'maximum_download_bytes_each':2*1024**2,
        'download_error_policy':'Stop EuropePMC source after first failure and preserve verified local primary XML; no retry loop.',
        'cases':len(cases),'source_ready_compounds':report['source_ready_compounds'],'strict_main_evaluable_now':0,'solver_calls':0})
    print(json.dumps({'inventory':str((writer.output/'cohort_inventory.json').relative_to(ROOT)), 'snapshots':snapshot_count,
        'single_compound_source_names':single_count,'covered_source_names_not_drug_count':len(drugs),'prepared_cases':len(cases),
        'source_ready_compounds':report['source_ready_compounds'],'strict_main_evaluable_now':0,'bytes_written':writer.bytes,
        'public_download_statuses':[r['status'] for r in fetches]},ensure_ascii=False))


if __name__=='__main__':
    main()
