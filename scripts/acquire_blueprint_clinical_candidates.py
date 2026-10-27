#!/usr/bin/env python3
"""Finite, resumable public clinical-source acquisition; no inference or GPU.

The explicit parent-drug queue is not a list of qualified evaluation cases.
Only a separately audited protocol plus sealed outcomes can become source-ready.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from curate_blueprint_independent_cohort import ROOT, inside, article_ids, text, sha

PARENTS = '''alfentanil amikacin atazanavir atomoxetine atorvastatin bedaquiline bupropion cabozantinib
carbamazepine cimetidine clarithromycin clopidogrel cyclosporine dabigatran dapagliflozin desipramine
dextromethorphan digoxin efavirenz erythromycin esomeprazole ethinylestradiol felodipine finerenone
fluvoxamine furosemide griseofulvin isoniazid itraconazole ketoconazole levonorgestrel lisinopril
metformin metoprolol mexiletine moclobemide montelukast moxidectin moxifloxacin olanzapine omeprazole
paclitaxel paracetamol paroxetine pioglitazone pitavastatin posaconazole pregabalin probenecid propofol
pyrimethamine quetiapine quinidine raltegravir repaglinide rifampicin risperidone ritonavir rosuvastatin
ruxolitinib sildenafil sufentanil tacrolimus theophylline tizanidine triazolam vancomycin verapamil
voriconazole warfarin apixaban edoxaban'''.split()
DEVELOPMENT = ['alprazolam','caffeine','fluconazole','midazolam']
ALIASES = {'paracetamol':'(paracetamol OR acetaminophen)','rifampicin':'(rifampicin OR rifampin)',
           'cyclosporine':'(cyclosporine OR ciclosporin)','posaconazole':'posaconazole'}


class DocumentSkipped(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--max-drugs',type=int,default=72);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--round',type=int,default=1);parser.add_argument('--previous');args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if len(PARENTS)!=72 or len(set(PARENTS))!=72 or set(PARENTS)&set(DEVELOPMENT):raise ValueError('Parent queue membership contract changed')
    if not 1<=args.max_drugs<=72:raise ValueError('Finite one-round drug cap is 1..72')
    cohort=inside(ROOT/args.cohort);output=inside(ROOT/args.output)
    if not output.is_relative_to(cohort):raise ValueError('Queue output must share the cohort cumulative disk budget')
    inventory=json.loads((cohort/'cohort_inventory.json').read_text())
    config={'schema_version':'bounded-parent-drug-acquisition-1','parent_drugs':PARENTS,'excluded_development_drugs':DEVELOPMENT,
        'parent_definition':'Manually normalized marketed small-molecule active parent drugs, not metabolites, protein drugs, placeholders, formulation variants or duplicated enantiomer entries.',
        'target':'At least60 source-ready independent evaluation drugs; queue membership and text matches are not qualification.',
        'round':args.round,'max_search_results_each':5,'max_new_fulltexts_each':1,'max_search_bytes':512*1024,
        'max_fulltext_bytes':2*1024**2,'max_total_cohort_bytes':100*1024**2,'max_total_seconds':7200,
        'request_timeout_seconds':30,'concurrency':1,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'error_policy':'Oversize/404 skips that document; network/other server failure stops that host. Never retry skipped documents.',
        'planned_reserve':'12 development/60 evaluation split remains to freeze; the4 already developed drugs are excluded, and72 search targets leave capacity to reserve8 additional development drugs.',
        'code_sha256':sha(Path(__file__))}
    if output.exists():
        if not args.resume:raise FileExistsError('Use --resume to skip completed acquisition entries')
        old=json.loads((output/'configuration.json').read_text())
        if old!=config:raise ValueError('Cannot resume with changed queue/configuration/code')
    else:
        output.mkdir(parents=True)
        (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    log=[];host_stops={};start=time.monotonic();previous=inside(ROOT/args.previous) if args.previous else None
    prior_complete=set();blocked_urls=set();prior_articles={}
    if previous:
        for folder in previous.iterdir():
            if folder.is_dir() and (folder/'result.json').exists():
                result=json.loads((folder/'result.json').read_text())
                if result.get('new_fulltexts',0)>0:prior_complete.add(result['compound'])
        for p in (previous/'sealed_articles').glob('*.xml'):
            prior_articles[p.stem]=p
        if (previous/'host_stops.json').exists():
            for record in json.loads((previous/'host_stops.json').read_text()).values():
                if 'byte cap' in record.get('detail','') or '404' in record.get('detail',''):blocked_urls.add(record['url'])
    if (output/'host_stops.json').exists():host_stops=json.loads((output/'host_stops.json').read_text())
    byte_count=sum(p.stat().st_size for p in cohort.rglob('*') if p.is_file())
    def persist(path,value,replace=False,sealed=False):
        nonlocal byte_count
        payload=value if isinstance(value,bytes) else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        if byte_count+len(payload)>config['max_total_cohort_bytes']:raise RuntimeError('Cumulative cohort disk limit reached')
        path.parent.mkdir(parents=True,exist_ok=True)
        if replace:
            old_size=path.stat().st_size if path.exists() else 0
            temporary=path.with_suffix(path.suffix+'.tmp')
            with temporary.open('wb') as f:f.write(payload)
            os.replace(temporary,path);byte_count+=len(payload)-old_size
        else:
            with path.open('xb') as f:f.write(payload)
            byte_count+=len(payload)
        if sealed:
            os.chmod(path.parent,0o700);os.chmod(path,0o600)
        return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'byte_size':len(payload)}
    def download(url,limit):
        host=urllib.parse.urlparse(url).hostname
        if url in blocked_urls:raise DocumentSkipped('Known oversized/404 document; never retried')
        if host in host_stops:raise RuntimeError('Public host is stopped after prior error: '+host)
        if time.monotonic()-start>config['max_total_seconds']:raise RuntimeError('Round wall-clock limit reached')
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PBPK-clinical-candidate-curator/1.0'}),timeout=config['request_timeout_seconds']) as r:
                data=r.read(limit+1)
            if len(data)>limit:
                blocked_urls.add(url)
                raise DocumentSkipped('Response exceeds registered byte cap')
            return data
        except DocumentSkipped:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code==404:
                blocked_urls.add(url)
                raise DocumentSkipped('HTTP404 no full text') from exc
            host_stops[host]={'url':url,'error_type':type(exc).__name__,'detail':str(exc)[:250],'retry':False}
            persist(output/'host_stops.json',host_stops,replace=True)
            raise
        except Exception as exc:
            host_stops[host]={'url':url,'error_type':type(exc).__name__,'detail':str(exc)[:250],'retry':False}
            persist(output/'host_stops.json',host_stops,replace=True)
            raise
    def seal_article(pmc,data):
        if pmc in prior_articles:
            prior=prior_articles[pmc];original=prior.read_bytes()
            return ({'path':str(prior.relative_to(ROOT)),'sha256':hashlib.sha256(original).hexdigest(),
                     'byte_size':len(original),'reused_prior_round_article':True},ET.fromstring(original))
        destination=output/'sealed_articles'/(pmc+'.xml')
        if destination.exists():
            original=destination.read_bytes()
            return ({'path':str(destination.relative_to(ROOT)),'sha256':hashlib.sha256(original).hexdigest(),
                     'byte_size':len(original),'reused_registered_article_version':True},ET.fromstring(original))
        ref=persist(destination,data,sealed=True)
        return ref,ET.fromstring(data)
    # Deduplicated model clinical IDs; these are only prior-source identifiers,
    # and no historical curve is copied into the independent observations area.
    model_pmids=set();model_dois=set()
    for drug in inventory['drug_inventory']:
        for row in drug['historical_clinical_studies']:
            model_pmids.update(row['identifiers']['pmid']);model_dois.update(row['identifiers']['doi'])
        for model in drug['models']:
            for doc in model['reference_documents']:
                model_pmids.update(doc['identifiers']['pmid']);model_dois.update(doc['identifiers']['doi'])
    def assess(article,drug,source):
        ids=article_ids(article);title=text(article.find('./front/article-meta/title-group/article-title'))
        method_sections=[s for s in article.findall('./body/sec') if re.search(r'method|participant|study design',text(s.find('title')),re.I)]
        methods=' '.join(text(s) for s in method_sections)
        tables=[{'id':t.attrib.get('id'),'caption':text(t.find('caption'))} for t in article.findall('.//table-wrap') if re.search(r'pharmacokinetic|plasma PK',text(t.find('caption')),re.I)]
        overlap=bool(ids.get('pmid') in model_pmids or (ids.get('doi') or '').lower() in model_dois)
        animal=bool(re.search(r'\b(rat|rats|dog|dogs|beagle|rabbit|rabbits|mice|mouse)\b',title,re.I))
        simulation=bool(re.search(r'in silico|virtual bioequivalence|physiologically.based.*model|meta.analysis|systematic review',title,re.I))
        drug_title=drug in title.lower() or (drug=='paracetamol' and 'acetaminophen' in title.lower()) or (drug=='rifampicin' and 'rifampin' in title.lower())
        healthy=bool(re.search(r'healthy.{0,30}(volunteer|adult|subject|participant)',methods+' '+title,re.I))
        status=('excluded_historical_model_source' if overlap else 'excluded_animal_title' if animal else
                'excluded_secondary_or_simulation_title' if simulation else
                'requires_manual_protocol_and_cohort_adjudication' if healthy and drug_title and tables else 'insufficient_structured_primary_human_pk_support')
        return {'compound':drug,'identifiers':ids,'study_id':'pmid:'+ids['pmid'] if ids.get('pmid') else 'doi:'+str(ids.get('doi')),
            'title':title,'source':source,'status':status,'clinical_model_source_overlap':overlap,'healthy_reported':healthy,
            'compound_in_title':drug_title,'pk_tables':tables,'source_ready':False,'run_ready':False,
            'observations_visible_to_designer':False,
            'remaining_checks':['actual measured parent and observed endpoint cells','dose/salt/formulation/food and sampling',
                'actual registry/cohort identity and reuse exclusions','study-level split and evidence denylist','sealed extraction with exact table-cell provenance']}
    queue=[]
    for drug in PARENTS:
        query=ALIASES.get(drug,drug)+' AND ("healthy volunteers" OR "healthy subjects" OR "healthy adults" OR bioequivalence) AND (pharmacokinetics OR pharmacokinetic) AND OPEN_ACCESS:Y NOT PUB_TYPE:"Review"'
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':query,'format':'json','resultType':'core','pageSize':5,'sort':'FIRST_PDATE_D desc'})
        queue.append({'compound':drug,'query':query,'search_url':url,'status':'completed' if (output/drug/'result.json').exists() else 'already_downloaded_prior_round' if drug in prior_complete else 'pending'})
    persist(output/'retrieval_queue.json',{'round':args.round,'target_source_ready_drugs':60,'entries':queue},replace=True)
    persist(output/'runner.json',{'pid':os.getpid(),'script':str(Path(__file__).relative_to(ROOT)),'code_sha256':config['code_sha256']},replace=True)
    summary={'status':'running','completed_drugs':sum(e['status']=='completed' for e in queue),'queued_drugs':len(queue),
        'source_ready_not_inferred_from_matches':True}
    for entry in queue[:args.max_drugs]:
        drug=entry['compound'];folder=output/drug
        if entry['status']=='already_downloaded_prior_round':continue
        if (folder/'result.json').exists():continue
        if time.monotonic()-start>config['max_total_seconds']:break
        results=[];errors=[];seen=set();new_fulltexts=0
        local=next((d for d in inventory['drug_inventory'] if d['compound']==drug),None)
        candidates=[] if local is None else local['primary_article_candidates']
        candidates=sorted(candidates,key=lambda r:(not r['healthy_in_title'],not r['has_pk_table_candidate'],r['identifiers'].get('pmid') or ''),reverse=False)
        for record in candidates[:5]:
            pmc=record['identifiers']['pmcid']
            if pmc in seen:continue
            seen.add(pmc)
            path=ROOT/record['source_file']
            if path.stat().st_size>32*1024**2:continue
            try:
                article=next(a for a in ET.parse(path).findall('.//article') if article_ids(a)['pmcid']==pmc)
                ref,article=seal_article(pmc,ET.tostring(article,encoding='utf-8',xml_declaration=True))
                results.append(assess(article,drug,ref))
            except Exception as error:
                errors.append({'source':'local_article','pmcid':pmc,'error_type':type(error).__name__,'detail':str(error)[:200]})
        try:
            payload=download(entry['search_url'],config['max_search_bytes'])
            search_ref=persist(folder/'sealed/search.json',payload,sealed=True)
            search=json.loads(payload)
            records=search.get('resultList',{}).get('result',[])
            if 'resultList' not in search:raise ValueError('EuropePMC response has no resultList')
            for record in records:
                pmc=record.get('pmcid')
                if not pmc or pmc in seen:continue
                if record.get('id') in model_pmids or (record.get('doi') or '').lower() in model_dois:
                    results.append({'compound':drug,'identifiers':{'pmid':record.get('id'),'pmcid':pmc,'doi':record.get('doi')},'title':record.get('title'),
                        'status':'excluded_historical_model_source','source_ready':False,'run_ready':False,'source':search_ref})
                    continue
                if new_fulltexts>=config['max_new_fulltexts_each']:break
                source_url='https://www.ebi.ac.uk/europepmc/webservices/rest/'+pmc+'/fullTextXML'
                try:
                    data=download(source_url,config['max_fulltext_bytes'])
                except DocumentSkipped as error:
                    errors.append({'source':'EuropePMC_document','url':source_url,'status':'skipped_document','detail':str(error),'retry':False})
                    continue
                article=ET.fromstring(data)
                if article.tag!='article':article=article.find('.//article')
                if article is None or article_ids(article)['pmcid']!=pmc:raise ValueError('Downloaded article identity mismatch')
                ref,article=seal_article(pmc,data);seen.add(pmc);new_fulltexts+=1
                results.append(assess(article,drug,ref))
        except Exception as error:
            errors.append({'source':'EuropePMC','error_type':type(error).__name__,'detail':str(error)[:250]})
        result={'compound':drug,'round':args.round,'status':'acquisition_complete_qualification_pending','candidates':results,
            'candidate_count':len(results),'new_fulltexts':new_fulltexts,'source_ready':False,'run_ready':False,'errors':errors}
        persist(folder/'result.json',result)
        entry['status']='completed';entry['candidate_count']=len(results);entry['new_fulltexts']=new_fulltexts
        persist(output/'retrieval_queue.json',{'round':args.round,'target_source_ready_drugs':60,'entries':queue},replace=True)
        completed=sum(e['status']=='completed' for e in queue)
        summary={'status':'running','completed_drugs':completed,'queued_drugs':len(queue),'new_fulltexts_current_drug':new_fulltexts,
            'last_compound':drug,'cumulative_cohort_bytes':byte_count,'host_stops':list(host_stops),'elapsed_seconds':time.monotonic()-start,
            'source_ready_not_inferred_from_matches':True}
        persist(output/'progress.json',summary,replace=True)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    summary.update(status='round_finished' if all(e['status'] in {'completed','already_downloaded_prior_round'} for e in queue) else 'bounded_round_partial',
        skipped_prior_downloaded_drugs=len(prior_complete),document_skip_urls=sorted(blocked_urls))
    persist(output/'completion.json',summary,replace=True)


if __name__=='__main__':main()
