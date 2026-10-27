#!/usr/bin/env python3
"""Bounded primary model acquisition and outcome-free clinical protocol revision."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from curate_blueprint_independent_cohort import ROOT, MAX_NEW_BYTES, inside, identities, article_ids, canonical


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    base=inside(ROOT/args.cohort);out=inside(base/'protocol_revision_20260920')
    out.mkdir(exist_ok=False)
    inventory=json.loads((base/'cohort_inventory.json').read_text())
    written=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    errors=[];stopped={'api.github.com'};events=[]
    def write(relative,payload,sealed=False):
        nonlocal written
        payload=payload if isinstance(payload,bytes) else (json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        if written+len(payload)>MAX_NEW_BYTES:raise RuntimeError('100 MiB cumulative cohort budget exceeded')
        p=inside(out/relative);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(payload)
        if sealed:os.chmod(p,0o600)
        written+=len(payload)
        return {'path':str(p.relative_to(ROOT)),'sha256':sha_bytes(payload),'byte_size':len(payload)}
    def get(url,limit):
        host=urllib.parse.urlparse(url).hostname
        if host in stopped:raise RuntimeError('Source stopped after a previous error: '+host)
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PBPK-public-clinical-curator/1.0'}),timeout=30) as response:
                payload=response.read(limit+1)
            if len(payload)>limit:raise RuntimeError('Per-source byte limit exceeded')
            events.append({'url':url,'bytes':len(payload),'sha256':sha_bytes(payload),'status':'downloaded'})
            return payload
        except Exception as exc:
            stopped.add(host);errors.append({'url':url,'source':host,'error_type':type(exc).__name__,'detail':str(exc)[:250],'retry':False})
            raise
    models=[]
    known={m['snapshot_sha256']:m['snapshot_path'] for d in inventory['drug_inventory'] for m in d['models']}
    for compound in ['Metformin','Paracetamol']:
        repo='Open-Systems-Pharmacology/'+compound+'-Model'
        record={'compound':compound.lower(),'repository':'https://github.com/'+repo,'branch':'master',
            'commit_sha':None,'pinning':'Exact downloaded file SHA256; GitHub unauthenticated API is rate-limited, not retried.', 'files':[]}
        try:
            for name in ['README.md',compound+'-Model.json','Evaluation/Input/Content/Section2.2_Data.md','Evaluation/Input/Content/References.md']:
                url='https://raw.githubusercontent.com/'+repo+'/master/'+name
                data=get(url,24*1024**2 if name.endswith('.json') else 2*1024**2)
                digest=sha_bytes(data)
                ref=({'path':known[digest],'sha256':digest,'byte_size':len(data),'reused_existing_content':True}
                     if digest in known else write('models/'+compound.lower()+'/'+name,data))
                record['files'].append({'url':url,'relative_repository_path':name,**ref})
                if name=='README.md':
                    readme=data.decode()
                    match=re.search(r'(?im)^#+ License\s*\n+([^#]+)',readme)
                    record['license']={'declaration':match[1].strip() if match else None,'source':ref,
                        'separate_license_file_present':False,'note':'Repository declares GPLv2 in README; do not invent a missing LICENSE file.'}
                elif name.endswith('.json'):
                    snapshot=json.loads(data)
                    record['snapshot_compounds']=[c['Name'] for c in snapshot.get('Compounds',[])]
                    record['snapshot_observation_identifiers']=sorted({str(v.get('Value')) for o in snapshot.get('ObservedData',[])
                        for v in o.get('ExtendedProperties',[]) if v.get('Name') in {'Study Id','Reference'}})
            record['status']='acquired_and_hashed'
        except Exception as exc:
            record.update(status='source_stopped_partial',error=str(exc)[:200])
        models.append(record)
    # Only supporting protocol documents are parsed; all downloaded source files
    # stay evaluator/curator-only because registry records can contain results.
    protocol_resources=[];supplement_text='';registry={}
    for trial in ['NCT03010475','NCT06081166']:
        try:
            url='https://clinicaltrials.gov/api/v2/studies/'+trial
            data=get(url,2*1024**2)
            ref=write('curator_only/'+trial+'.json',data,True);protocol_resources.append(ref)
            registry[trial]=json.loads(data)
        except Exception:
            pass
    supplement_url='https://static-content.springer-cdn.com/esm/art%3A10.1007%2Fs40262-020-00976-x/MediaObjects/40262_2020_976_MOESM1_ESM.docx'
    try:
        docx=get(supplement_url,2*1024**2)
        ref=write('curator_only/PMC8416862_supplement.docx',docx,True);protocol_resources.append(ref)
        with zipfile.ZipFile(io.BytesIO(docx)) as z:
            item=z.getinfo('word/document.xml')
            if item.file_size>4*1024**2:raise RuntimeError('Supplement expanded XML limit exceeded')
            document=ET.fromstring(z.read(item))
            supplement_text=' '.join(e.text or '' for e in document.iter() if e.tag.endswith('}t'))
        write('curator_only/PMC8416862_supplement_extracted.txt',supplement_text.encode(),True)
    except Exception:
        pass
    revised=[]
    for old_path in sorted((base/'prediction_metadata').iterdir()):
        case=json.loads(old_path.read_text());drug=case['compound'];pmc=case['identifiers']['pmcid']
        article=ET.parse(base/'sealed/sources'/ (pmc+'.xml')).getroot()
        # Check exact paragraph text so XML adjacent tags cannot hide a registry ID.
        paragraphs=[' '.join(' '.join(p.itertext()).split()) for p in article.iter('p')]
        actual=({'PMC12378308':['NCT06081166'],'PMC8416862':['NCT03010475'],'PMC12394267':['CTR20180793'],'PMC10897852':[]})[pmc]
        if not all(any(identifier in p for p in paragraphs) for identifier in actual):
            raise RuntimeError('Declared trial ID absent from refreshed primary article paragraphs')
        case['supersedes_protocol']={'path':str(old_path.relative_to(ROOT)),'sha256':sha_bytes(old_path.read_bytes())}
        case['schema_version']='independent-clinical-protocol-2'
        case['clinical_source_audit']['trial_registry_ids']=actual
        case['clinical_source_audit']['trial_id_assignment']='Manually adjudicated actual trial from article; background references are not assigned.'
        case['evidence_exclusion']['trial_registry_ids']=actual
        case['source_protocol_resources']=protocol_resources
        if drug in {'atorvastatin','rosuvastatin'}:
            case['protocol']['cohort_id']='trial:NCT06081166:'+('cohort_1' if drug=='atorvastatin' else 'cohort_2')
            case['protocol']['trial_registry_id']='NCT06081166'
        elif drug=='furosemide':
            case['protocol']['cohort_id']='trial:NCT03010475'
            case['protocol']['trial_registry_id']='NCT03010475'
            case['clinical_source_audit']['not_this_trial']=['NCT02845219: oral contraceptive Trial1']
        case['cohort_id']=case['protocol']['cohort_id']
        # The sealed values are not read or rewritten; publication identity stays
        # fixed and this manifest records only cohort-identifier normalization.
        case['sealed_reference_identity_mapping']={'old_cohort_id':json.loads(old_path.read_text())['cohort_id'],
            'normalized_cohort_id':case['cohort_id'],'outcomes_unchanged':True}
        extra=[]
        for model in models:
            if model['compound']!=drug:continue
            for ref in model['files']:
                data=(ROOT/ref['path']).read_text()
                if any(identifier and identifier.lower() in data.lower() for identifier in [case['identifiers']['pmid'],case['identifiers']['doi'],*actual]):
                    extra.append({'type':'new_official_model_source_overlap','path':ref['path']})
        case['clinical_source_audit']['new_official_model_overlap']=extra
        if extra:
            case['status']='blocked_new_official_model_source_overlap'
            case['clinical_source_audit']['clinical_source_disjoint_in_registered_model_corpus']=False
        ref=write('prediction_metadata/'+old_path.name,case)
        revised.append({'case_id':case['case_id'],'compound':drug,'status':case['status'],'protocol_reference':ref})
    # Static adapter inventory: classify types without converting/dropping them.
    from pbpk_kb.blueprints.human_model_snapshot import _PROCESSES,_MOLECULAR_PROCESS_PARAMETERS
    supported=set(_PROCESSES)|set(_MOLECULAR_PROCESS_PARAMETERS)
    adapters=[]
    for drug in ['atorvastatin','rosuvastatin','furosemide','metformin','paracetamol']:
        d=next(d for d in inventory['drug_inventory'] if d['compound']==drug)
        sources=[m['snapshot_path'] for m in d['models'][:1]]
        for model in models:
            if model['compound']==drug:
                sources=[r['path'] for r in model['files'] if r['relative_repository_path'].endswith('.json')] or sources
        for path in sources:
            snapshot=json.loads((ROOT/path).read_text())
            c=next(c for c in snapshot['Compounds'] if canonical(c['Name'])==drug)
            processes=[{'internal_name':p['InternalName'],'molecule':p.get('Molecule'),'data_source':p.get('DataSource'),
                'named_metabolite':p.get('Metabolite'),'supported_type_by_current_snapshot_adapter':p['InternalName'] in supported,
                'parameter_names_units':[(v['Name'],v.get('Unit')) for v in p.get('Parameters',[])]} for p in c.get('Processes',[])]
            adapters.append({'compound':drug,'snapshot_path':path,'snapshot_sha256':sha_bytes((ROOT/path).read_bytes()),
                'processes':processes,'unsupported_types':sorted({p['internal_name'] for p in processes if not p['supported_type_by_current_snapshot_adapter']}),
                'additional_blocker':'Named metabolites need explicit multi-compound assembly; never drop them.' if any(p['named_metabolite'] for p in processes) else None,
                'native_solver_executed':False,'source_organ_parameters_copied':False})
    write('official_model_acquisition.json',{'models':models,'events':events,'errors':errors,
        'github_api_status':'HTTP403 rate limit encountered before this run; API source stopped, no credentials used. Public raw files are exact-hash pinned.',
        'license_sources_preserved':True})
    write('native_adapter_requirements.json',{'current_adapter':'src/pbpk_kb/blueprints/human_model_snapshot.py','drugs':adapters,'solver_calls':0})
    write('protocol_revision_manifest.json',{'cases':revised,'protocol_resources':protocol_resources,
        'supplement_text_obtained':bool(supplement_text),'registry_records_obtained':sorted(registry),
        'sealed_values_modified':False,'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':sha_bytes(Path(__file__).read_bytes()),
        'cumulative_bytes_before_manifest':written,'maximum_cumulative_bytes':MAX_NEW_BYTES,
        'food_or_sampling_not_inferred_from_silence':True})
    print(json.dumps({'prepared_protocols':len(revised),'models':[{k:m.get(k) for k in ['compound','status','snapshot_compounds']} for m in models],
        'registry_records':sorted(registry),'supplement_obtained':bool(supplement_text),'bytes_written_cumulative':written,
        'native_adapter_unsupported':[{k:a[k] for k in ['compound','unsupported_types','additional_blocker']} for a in adapters],
        'errors':errors},ensure_ascii=False))


if __name__=='__main__':main()
