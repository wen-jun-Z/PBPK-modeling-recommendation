#!/usr/bin/env python3
"""Deterministic JATS protocol/table extraction with sealed outcome scalars.

Preserves row/column/header/footnote provenance. Ambiguous dose arms, statistics,
units and simulated values remain explicit machine-processing tasks; no invented
study independence or mandatory human-label dependency is introduced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from curate_blueprint_independent_cohort import ROOT, inside, article_ids


def text(element):
    return ' '.join(' '.join(element.itertext()).split()) if element is not None else ''


def endpoint(value):
    compact=re.sub(r'\s+','',value).lower()
    if re.search(r'c(?:max|maximum)',compact):return 'Cmax'
    if 'auc' in compact or 'areaunderthecurve' in compact:
        return 'AUC_infinity' if 'inf' in compact or '∞' in compact else 'AUC_reported_window'
    return None


def grid(rows):
    carry={};result=[]
    for row in rows[:300]:
        current={};following={}
        for col,(value,count) in carry.items():
            current[col]=value
            if count>1:following[col]=(value,count-1)
        col=0
        for cell in row:
            while col in current:col+=1
            colspan=min(int(cell.attrib.get('colspan','1')),24);rowspan=min(int(cell.attrib.get('rowspan','1')),300)
            value=text(cell)
            for i in range(col,col+colspan):
                current[i]=value
                if rowspan>1:following[i]=(value,rowspan-1)
            col+=colspan
        if current:result.append([current.get(i,'') for i in range(max(current)+1)])
        carry=following
    return result


def scalar(cell):
    normalized=cell.replace('\u2009','').replace('\u00a0',' ').replace(',','')
    m=re.match(r'^\s*(?:[<>≤≥]\s*)?(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',normalized)
    if not m:return None
    if re.search(r'[×x]\s*10',normalized):return {'raw_cell':cell,'parsed_leading_scalar':None,'numeric_parse':'scientific_notation_requires_normalization'}
    return {'raw_cell':cell,'parsed_leading_scalar':float(m[1]),'numeric_parse':'leading_scalar_not_complete_statistical_cell'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--retrieval',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    cohort=inside(ROOT/args.cohort);retrieval=inside(ROOT/args.retrieval);out=inside(ROOT/args.output)
    out.mkdir(parents=True,exist_ok=False)
    original=Path(__file__).read_bytes();maximum=100*1024**2
    def save(relative,value,sealed=False):
        payload=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        current=sum(f.stat().st_size for f in cohort.rglob('*') if f.is_file())
        if current+len(payload)>maximum:raise RuntimeError('Cumulative100MiBcohort limit reached')
        f=out/relative;f.parent.mkdir(parents=True,exist_ok=True)
        with f.open('xb') as h:h.write(payload)
        if sealed:os.chmod(f.parent,0o700);os.chmod(f,0o600)
        return {'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'byte_size':len(payload)}
    candidates=[]
    for folder in retrieval.iterdir():
        if folder.is_dir() and (folder/'result.json').exists():
            row=json.loads((folder/'result.json').read_text())
            candidates.extend(c for c in row['candidates'] if c['status']=='requires_manual_protocol_and_cohort_adjudication')
    seen=set();records=[]
    for candidate in candidates:
        drug=candidate['compound'];pmc=candidate['identifiers']['pmcid'];key=drug+'_'+pmc.lower()
        if key in seen:continue
        seen.add(key)
        source=inside(ROOT/candidate['source']['path']);payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=candidate['source']['sha256']:raise ValueError('Primary article checksum changed')
        article=ET.fromstring(payload);ids=article_ids(article)
        sections=[s for s in article.findall('./body/sec') if re.search(r'method|participant|study design',text(s.find('title')),re.I)]
        paragraphs=[text(p) for s in sections for p in s.iter('p')]
        methods=' '.join(paragraphs)
        dose_mentions=[];sampling_mentions=[]
        for index,para in enumerate(paragraphs):
            phash=hashlib.sha256(para.encode()).hexdigest()
            if (drug in para.lower() or drug=='paracetamol' and 'acetaminophen' in para.lower()) and re.search(r'administ|receiv|dos(?:e|ing)',para,re.I):
                for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(mg|[µμu]g|g)\b(?!\s*/)',para):
                    dose_mentions.append({'value':float(m[1]),'unit':m[2],'source_method_paragraph_index':index,'paragraph_sha256':phash,
                        'arm_assignment':'unresolved_when_multiple_drugs_or_doses_share_paragraph'})
            if re.search(r'blood|plasma|sample',para,re.I) and re.search(r'after|post.?dose|sampling',para,re.I):
                intervals=[{'value':float(m[1]),'unit':m[2]} for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b',para,re.I)]
                if intervals:sampling_mentions.append({'time_mentions':intervals,'source_method_paragraph_index':index,'paragraph_sha256':phash,
                    'complete_ordered_sampling_grid':False})
        routes=sorted(set('intravenous' if v.lower().startswith(('intraven','i.v')) else 'oral' for v in re.findall(r'\b(?:intravenous|i\.v\.|oral)\b',methods,re.I)))
        food=[]
        if re.search(r'fast(?:ed|ing)|overnight fast',methods,re.I):food.append('fasting_mentioned')
        if re.search(r'fed condition|fed state|after.*(?:meal|breakfast)',methods,re.I):food.append('fed_mentioned')
        healthy=bool(re.search(r'healthy.{0,30}(volunteer|adult|subject|participant)',methods,re.I))
        scalars=[];tables=[]
        for table in article.findall('.//table-wrap'):
            caption=text(table.find('caption'))
            if not re.search(r'pharmacokinetic|plasma PK',caption,re.I):continue
            head=grid(table.findall('.//thead/tr'));body=grid(table.findall('.//tbody/tr'))
            foot=text(table.find('table-wrap-foot'));last_endpoint=None;last_label=None
            table_count=0
            for ri,row in enumerate(body):
                rowlabel=row[0] if row else ''
                direct=endpoint(rowlabel)
                if direct:last_endpoint=direct;last_label=rowlabel
                elif re.search(r't.?max|half.life|clearance|t.?½|t.?1/2|lambda',rowlabel,re.I):last_endpoint=None;last_label=None
                for ci,cell in enumerate(row):
                    headers=[r[ci] for r in head if ci<len(r) and r[ci]]
                    column_endpoint=endpoint(' '.join(headers))
                    metric=column_endpoint or last_endpoint
                    if not metric or ci==0:continue
                    value=scalar(cell)
                    if value is None:continue
                    context=' '.join(headers+[rowlabel,foot])
                    stat=('geometric_mean' if re.search(r'geometric|\bGM\b',context,re.I) else
                          'arithmetic_mean' if re.search(r'arithmetic mean',context,re.I) else
                          'mean_unspecified' if re.search(r'mean',context,re.I) else
                          'median' if re.search(r'median',context,re.I) else 'unresolved')
                    contrast=bool(re.search(r'ratio|p.value|confidence|90% CI|95% CI|relative', ' '.join(headers),re.I))
                    predicted=bool(re.search(r'predicted|simulat|model.estimat',' '.join(headers)+caption,re.I))
                    scalars.append({'endpoint':metric,'table_id':table.attrib.get('id'),'body_row_index':ri,'expanded_column_index':ci,
                        'endpoint_label':last_label,'row_label':rowlabel,'column_headers':headers,'caption':caption,'footnotes':foot,
                        **value,'statistical_summary':stat,'is_contrast_or_probability_column':contrast,'possibly_model_predicted':predicted,
                        'unit_status':'explicit_source_label_retained_requires_dimension_normalization',
                        'arm_assignment':'source_headers_retained_not_yet_mapped_to_single_drug_protocol',
                        'clinical_ground_truth_accepted':False})
                    table_count+=1
            if table_count:tables.append({'table_id':table.attrib.get('id'),'caption':caption,'scalar_cell_count':table_count})
        sealed=save('sealed/'+key+'.json',{'compound':drug,'identifiers':ids,'source':candidate['source'],
            'method_paragraphs':paragraphs,'reported_pk_table_scalars':scalars,
            'values_not_visible_to_designer':True,'complete_case_qualification':False},True)
        unresolved=['Map specific monotherapy/control arm and actual cohort identity.', 'Separate measured values from predictions, contrasts and statistical comparison columns.',
            'Normalize endpoint unit and statistic without taking AUC_last as fixed-horizon AUC.',
            'Resolve salt/base dose, formulation and source-based food state.', 'Recover complete sampling grid from primary protocol, not unordered time mentions.']
        meta={'schema_version':'clinical-protocol-extraction-1','case_candidate_id':key,'compound':drug,'identifiers':ids,
            'publication_group_id':'pmid:'+ids['pmid'] if ids['pmid'] else 'doi:'+str(ids['doi']),
            'clinical_source_status':candidate['status'],'healthy_adult_reported':healthy,'route_mentions':routes,'food_mentions':food,
            'dose_mentions':dose_mentions,'sampling_time_mentions':sampling_mentions,
            'endpoint_tables':tables,'sealed_extraction_reference':sealed,'source':candidate['source'],
            'native_prediction_input_complete':False,'source_ready':False,'run_ready':False,
            'unresolved_fields':unresolved,'next_stage':'deterministic_arm_mapping_and_evidence_contract_validation',
            'human_labels_required':False,'solver_calls':0,'outcomes_exposed_to_designer':False}
        ref=save('protocol_candidates/'+key+'.json',meta)
        records.append({'compound':drug,'case_candidate_id':key,'source_protocol_reference':ref,'extracted_scalar_cells':len(scalars),
            'endpoint_tables':len(tables),'status':'protocol_and_endpoint_fields_extracted_pending_arm_contract'})
    save('extraction_manifest.json',{'records':records,'article_drug_pairs':len(records),
        'compounds':sorted({r['compound'] for r in records}),'extracted_scalar_cells':sum(r['extracted_scalar_cells'] for r in records),
        'source_ready_promotions':0,'outcome_values_sealed':True,'mandatory_human_label_dependency':False,
        'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':hashlib.sha256(original).hexdigest(),'solver_calls':0})
    print(json.dumps({'article_drug_pairs':len(records),'compounds':len({r['compound'] for r in records}),
        'extracted_scalar_cells':sum(r['extracted_scalar_cells'] for r in records),'source_ready_promotions':0,'outcome_values_sealed':True}))


if __name__=='__main__':main()
