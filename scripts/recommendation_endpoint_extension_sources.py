#!/usr/bin/env python3
"""Reproduce same-arm clinical endpoint transcription without modifying sealed data.

No model fitting, network access, or prediction-dependent selection is performed.
Physical XML cell indices include header rows and do not expand rowspan/colspan.
New observations are descriptive NCA summaries, not individual concentration data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / 'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
OUTPUT = ROOT / 'results/data/model_recommendation_extended_references.json'
ENDPOINTS = ['Cmax', 'Tmax', 'AUC_window', 'AUC_last', 'AUC_infinity',
             'half_life', 'CL', 'Vz', 'MRT', 'Vss', 'C_last']

# compound: (table id, [(endpoint, physical row, physical column, value, statistic)]).
# Mean and median are transcribed separately; no inversion of aggregate means.
TABLES = {
 'atorvastatin': ('Tab3', [('Tmax',2,3,2.10,'mean'),('half_life',4,3,12.56,'mean')]),
 'rosuvastatin': ('Tab3', [('Tmax',29,3,4.60,'mean'),('half_life',31,3,23.13,'mean')]),
 'bedaquiline': ('t0020', [('Tmax',2,1,4.58,'mean'),('half_life',5,1,30.9,'mean')]),
 'bupropion': ('T0003', [('Tmax',2,6,3.167,'mean'),('half_life',2,5,30.067,'mean')]),
 'clarithromycin': ('T3', [('Tmax',4,3,1.00,'median'),('half_life',5,3,5.90,'mean')]),
 'clopidogrel': ('t1-dddt-9-2359', [('Tmax',6,2,.75,'median'),('half_life',7,2,3.94,'median')]),
 'dabigatran': ('Tab2', [('Tmax',4,2,2.50,'median'),('half_life',7,2,9.77,'mean')]),
 'dapagliflozin': ('t0001', [('Tmax',1,2,.75,'median'),('half_life',5,2,11.46,'mean'),('CL',6,2,19.31,'mean')]),
 'dextromethorphan': ('cpdd281-tbl-0001', [('Tmax',5,7,3.0,'median'),('half_life',6,7,8.2,'mean'),('CL',7,7,3952,'mean')]),
 'digoxin': ('table3-20503121261424728', [('Tmax',4,1,1.5,'median'),('half_life',5,1,38.9,'mean')]),
 'efavirenz': ('Tab2', [('Tmax',15,4,3.0,'median'),('half_life',15,7,87.6,'mean')]),
 'esomeprazole': ('T2', [('Tmax',2,2,2.00,'median'),('half_life',6,2,1.25,'mean')]),
 'ethinylestradiol': ('Tab2', [('Tmax',10,1,1.,'median')]),
 'felodipine': ('T4', [('Tmax',5,1,1.73,'mean'),('half_life',4,1,25.59,'mean'),('CL',6,1,224.15,'mean')]),
 'furosemide': ('Tab4', [('Tmax',11,1,.9,'median')]),
 'levonorgestrel': ('Tab3', [('Tmax',10,1,1.,'median')]),
 'lisinopril': ('cts12905-tbl-0001', [('Tmax',4,1,6.,'median')]),
 'montelukast': ('t2', [('Tmax',5,6,5.,'median'),('half_life',7,6,4.71,'mean')]),
 'moxifloxacin': ('t2-dddt-9-1233', [('Tmax',5,1,1.,'median'),('half_life',6,1,11.87,'mean'),('CL',7,1,12.80,'mean'),('Vz',8,1,214.60,'mean')]),
 'omeprazole': ('Tab2', [('Tmax',2,2,2.33,'median'),('half_life',7,2,1.09,'mean')]),
 'paracetamol': ('tbl0003', [('Tmax',3,4,1.,'median'),('half_life',15,2,3.709,'mean')]),
 'paroxetine': ('pharmaceutics-13-01869-t005', [('Tmax',5,2,3.08,'mean'),('half_life',6,2,22.32,'mean'),('MRT',8,2,34.06,'mean')]),
 'pioglitazone': ('t0002', [('Tmax',2,2,2.,'median'),('half_life',5,1,10.5,'mean'),('CL',6,1,2.6,'mean')]),
 'posaconazole': ('tab-51', [('Tmax',5,3,5.,'median'),('half_life',12,3,23.74,'mean')]),
 'pregabalin': ('Tab1', [('Tmax',3,3,1.25,'median'),('half_life',8,3,6.49,'geometric_mean')]),
 'quetiapine': ('pharmaceutics-13-01573-t003', [('Tmax',11,4,1.17,'mean'),('half_life',11,5,5.24,'mean'),('Vz',11,6,12.83,'mean'),('CL',11,7,1.77,'mean')]),
 'repaglinide': ('cts13252-tbl-0002', [('Tmax',14,7,.75,'median'),('half_life',17,7,2.42,'geometric_mean'),('CL',20,7,46.3,'geometric_mean'),('Vz',23,7,162.,'geometric_mean')]),
 'risperidone': ('Tab2', [('Tmax',3,2,1.09,'mean'),('half_life',6,2,4.94,'mean')]),
 'ruxolitinib': ('hon2544-tbl-0001', [('Tmax',3,1,.9,'mean'),('half_life',6,1,2.8,'mean'),('CL',9,1,22.8,'mean')]),
 'sildenafil': ('tbl0003', [('Tmax',6,2,1.050,'mean'),('half_life',5,2,3.433,'mean')]),
 'tacrolimus': ('T3', [('Tmax',5,2,9.,'median'),('half_life',11,2,36.83,'mean')]),
 'vancomycin': ('antibiotics-13-00993-t003', [('MRT',9,2,5.12,'mean'),('CL',10,2,.146,'mean'),('Vz',11,2,.793,'mean'),('Vss',12,2,.730,'mean'),('half_life',13,2,3.87,'mean')]),
}

# Explicit assay limits only; the numerical calibration lower bound is not
# silently substituted for an unreported LLOQ. Values below are in ng/mL.
# The paragraph number addresses .//p in the corresponding bound article.
LLOQ = {
 'atorvastatin':(.1,27,'0.1 ng/mL'), 'rosuvastatin':(.1,27,'0.1 ng/mL'),
 'bedaquiline':(5.,2,'5 ng/mL'), 'clarithromycin':(100.,11,'100 ng/mL'),
 'clopidogrel':(.024,28,'24.00 pg/mL'), 'felodipine':(.057,23,'0.057 ng/mL'),
 'ethinylestradiol':(.0025,34,'2.5 pg/mL'), 'levonorgestrel':(.025,34,'25 pg/mL'),
 'furosemide':(5.,34,'5 ng/mL'), 'metformin':(5.,25,'5.00 ng/mL'),
 'montelukast':(2.,21,'2.00 ng/mL'), 'moxifloxacin':(100.,10,'100 ng/mL'),
 'omeprazole':(.5,21,'0.500 ng/mL'), 'paracetamol':(250.,15,'0.25 µg/mL'), 'paroxetine':(1.,85,'1 ng/mL'),
 'posaconazole':(2.,73,'2.0 ng/mL'), 'quetiapine':(.5,8,'0.5 ng/mL'),
 'risperidone':(.1,23,'0.1 ng/mL'), 'sildenafil':(4.,32,'4 ng/mL'),
 'voriconazole':(10.,18,'0.010 μg/mL'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def text(node):
    return ' '.join(' '.join(node.itertext()).split())

def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)

def project_path(path):
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Source outside project boundary')
    return resolved

def source(path, **kwargs):
    path = project_path(path)
    return {'path':str(path.relative_to(ROOT)), 'sha256':sha(path), **kwargs}

def article_for(data):
    return next((project_path(p) for p in strings(data)
                 if p.endswith('.xml') and project_path(p).exists()), None)

def first_number(raw):
    match = re.search(r'[-+]?\d+(?:[,.]\d+)*', raw)
    return float(match.group().replace(',','')) if match else None

def xml_cell(path, table, row, column, expected):
    root = ET.parse(path).getroot()
    node = next(x for x in root.findall('.//table-wrap') if x.get('id') == table)
    rows = node.findall('.//tr')
    cells = [c for c in rows[row] if c.tag in ('th','td')]
    raw = text(cells[column])
    if not math.isclose(first_number(raw), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'Transcription mismatch: {path.name} {table} [{row},{column}]: {raw!r} != {expected}')
    return source(path, table=table, row=row, column=column,
                  coordinate_system='physical_XML_cell_zero_based_including_headers',
                  raw_cell=raw, row_cells=[text(c) for c in cells],
                  footnotes=' '.join(text(f) for f in node.findall('.//table-wrap-foot')))

def is_steady(scenario):
    return scenario['compound'] in {'clarithromycin','ethinylestradiol','levonorgestrel',
                                    'felodipine','itraconazole','pioglitazone'}

def record(scenario, endpoint, value, unit, src, **kw):
    route = scenario['normalized_design'].get('route')
    oral = route == 'oral'
    quantity = {'half_life':'apparent_terminal_half_life' if oral else 'terminal_half_life',
                'MRT':'MRT_including_absorption' if oral else 'MRT',
                'CL':'CL/F' if oral else 'CL', 'Vz':'Vz/F' if oral else 'Vz'}.get(endpoint, endpoint)
    body_normalized = scenario['compound'] in {'vancomycin','quetiapine'} and endpoint in {'CL','Vz','Vss'}
    row = {'scenario_id':scenario['scenario_id'], 'case_id':scenario['case_id'],
           'compound':scenario['compound'], 'publication_cluster_id':scenario['publication_cluster_id'],
           'endpoint':endpoint, 'endpoint_key':endpoint, 'quantity':quantity,
           'value':float(value), 'unit':unit, 'window_h':None,
           'steady_state':is_steady(scenario), 'body_weight_normalized':body_normalized,
           'dose_normalization':None, 'route':route, 'dose':scenario['normalized_design'].get('dose'),
           'matrix':scenario['normalized_design'].get('matrix'),
           'analyte':scenario['normalized_design'].get('analyte'),
           'statistic':'arithmetic_mean', 'eligibility':True, 'reason':None,
           'source_ref':src, 'source_quality_flags':[],
           'definition':None, 'terminal_definition':None,
           'source_scope':'same_publication_same_registered_clinical_reference_arm',
           'interpretation':'group_summary_NCA_diagnostic_not_individual_patient_validation'}
    row.update(kw)
    if endpoint == 'half_life':
        row['terminal_definition'] = 'Published NCA terminal slope half-life; an oral apparent terminal half-life may include absorption kinetics.'
    return row

def extra_rows(s, data):
    drug = s['compound']; article = article_for(data); rows=[]
    spec=TABLES.get(drug)
    if drug == 'metformin':
        fed=s['case_id'].endswith('_fed'); col=2 if fed else 1
        spec=('Tab1',[('Tmax',12,col,4. if fed else 2.125,'median'),('half_life',13,col,3.563 if fed else 3.388,'mean')])
    if drug == 'voriconazole':
        high=s['case_id'].endswith('6mgkg')
        spec=('Tab3' if high else 'Tab2',[('Tmax',6,2,2. if high else 1.33,'median'),('half_life',5,2,8.44 if high else 7.91,'mean')])
    if spec:
        table, entries=spec
        for endpoint,r,c,value,stat in entries:
            src=xml_cell(article,table,r,c,value)
            unit='L/h' if endpoint=='CL' else 'L' if endpoint in {'Vz','Vss'} else 'h'
            if drug in {'quetiapine','vancomycin'} and endpoint in {'CL','Vz','Vss'}:unit+='/kg'
            item=record(s,endpoint,value,unit,src,statistic='arithmetic_mean' if stat=='mean' else stat,
                        provenance_kind='new_same_arm_source_transcription')
            if drug=='dabigatran':
                item.update(eligibility=False,reason='Total dabigatran assay includes conjugate contribution; native parent analyte binding not established.',analyte='total_dabigatran')
            if drug=='moxifloxacin' and endpoint=='Vz':
                item.update(eligibility=False,quantity='Vd/F',reason='Source reports apparent Vd/F without an explicit terminal-slope definition; equivalence to Vz/F is not assumed.')
            if drug=='quetiapine' and endpoint=='Vz':
                item['definition']='Methods explicitly define Vd/F as CL/F divided by terminal Ke, corrected for body weight; therefore terminal Vz/F, L/kg.'
            if drug=='vancomycin' and endpoint=='MRT':item['definition']='NCA MRTinf=AUMCinf/AUCinf minus infusion_duration/2 (2 h infusion); not oral MRT.'
            if drug=='repaglinide':item['source_quality_flags'].append('Source exposure units and apparent clearance differ by factor 1000 under the printed dose; independently transcribed quantities are not reconciled by invented unit correction.')
            rows.append(item)
    if drug=='caffeine':
        p=ROOT/'experiments/blueprint_design/source_audits/caffeine_sherman2022/sherman2022.pdf'
        high='250mg' in s['case_id']; column=3 if high else 1
        vals=[('Tmax',1.44 if high else 1.18,'h'),('CL',6.50 if high else 5.58,'L/h'),
              ('Vz',39.5 if high else 43.5,'L'),('half_life',4.22 if high else 5.41,'h'),
              ('MRT',6.85 if high else 7.61,'h')]
        import pymupdf
        assert sha(p)=='f110bbd82e618788c6264168946d26db6c93191ad717a12bd5691a4bf2a93284'
        page=pymupdf.open(p)[2]
        for ep,value,unit in vals:
            y={'Tmax':615.83,'CL':667.25,'Vz':684.37,'half_life':692.93,'MRT':701.49}[ep]
            x=469.98 if high else 379.95
            matches=[w for w in page.get_text('words') if abs(w[0]-x)<.1 and abs(w[1]-y)<.1]
            assert len(matches)==1 and math.isclose(first_number(matches[0][4]),value)
            word=matches[0]
            label={'CL':'CL/F','Vz':'Vz/F','half_life':'t½'}.get(ep,ep)
            rows.append(record(s,ep,value,unit,source(p,page=3,table='1',row_label=label,column=column,
                         column_label='Caffeine 1.29 mmol' if high else 'Caffeine 0.26 mmol',
                         raw_cell=word[4],word_bbox=list(word[:4]),transcription='visually_verified_exact_PDF_word_coordinate'),
                         statistic='geometric_mean',n=9,provenance_kind='new_same_arm_source_transcription',
                         source_quality_flags=['Table labels all values geometric mean (%CV); prose describes median Tmax. Table statistic retained, prose conflict disclosed.'] if ep=='Tmax' else []))
    if drug=='olanzapine':
        p=ROOT/'experiments/blueprint_design/external_validation/clinical_cohort_round4_20260920/sealed/olanzapine_table/IJPhS-73-678-g004.jpg'
        assert sha(p)=='35dc8aaa8f97f9f6607fbdb4cc4d03be3d0d05d60131bf2a14d5053ce44b702a'
        for ep,value,r,raw,stat in [('Tmax',6.,4,'6.0 (2.0–8.0)','median'),('half_life',33.264,5,'33.264±3.92','arithmetic_mean')]:
            rows.append(record(s,ep,value,'h',source(p,table='T1',row=r,column=2,raw_cell=raw,
                     column_label='Olanzapine 10 mg/5 ml Suspension (Mean±SD)',transcription='visually_verified_image_table'),
                     statistic=stat,n=16,provenance_kind='new_same_arm_source_transcription'))
    if drug=='pitavastatin':
        p=ROOT/'experiments/blueprint_design/external_validation/expanded_cohort_20260920/sealed/supplements/PMC9841298/CTS-16-85-s005.docx'
        with zipfile.ZipFile(p) as z: rt=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        table=rt.findall('.//w:tbl',ns)[0];trs=table.findall('w:tr',ns)
        for ep,r,value,stat in [('Tmax',4,.67,'median'),('half_life',5,11.98,'arithmetic_mean')]:
            raw=text(trs[r].findall('w:tc',ns)[2]);assert math.isclose(first_number(raw),value)
            rows.append(record(s,ep,value,'h',source(p,table=0,row=r,column=2,raw_cell=raw,column_label='R (n=36)'),
                     statistic=stat,n=36,provenance_kind='new_same_arm_source_transcription'))
    # These observations are retained, but cannot contribute a numerical score
    # until the printed dimensional inconsistency is independently resolved.
    ambiguous={
        'vancomycin':[('C_last','antibiotics-13-00993-t003',3,2,4.68,'h','source_unit_time_concentration_swap_unresolved')],
        'sildenafil':[('AUC_last','tbl0003',2,2,1074.,'ng/hr/mL','source_AUC_dimension_inverse_time_unresolved'),
                      ('AUC_infinity','tbl0003',3,2,1123.,'ng/hr/mL','source_AUC_dimension_inverse_time_unresolved')],
        'paracetamol':[('AUC_infinity','tbl0003',9,2,50.18,'ug/h/mL','source_AUC_dimension_inverse_time_unresolved')],
    }
    for ep,table,r,c,value,unit,reason in ambiguous.get(drug,[]):
        rows.append(record(s,ep,value,unit,xml_cell(article,table,r,c,value),eligibility=False,
                           reason=reason,provenance_kind='new_same_arm_source_transcription',
                           source_quality_flags=[reason]))
    return rows

def base_rows(s, data, sealed_ref):
    result=[]
    source_rows=next((data[k] for k in ['endpoints','observed_endpoints','endpoint_rows'] if k in data),[])
    for i,e in enumerate(source_rows):
        if e.get('case_id',s['case_id'])!=s['case_id']:continue
        if e.get('counts_as_unique_endpoint') is False:continue
        raw=e['endpoint'];low=raw.lower()
        if low.startswith('cmax'):endpoint='Cmax'
        elif 'inf' in low:endpoint='AUC_infinity'
        elif 'last' in low:endpoint='AUC_last'
        elif low.startswith('auc'):endpoint='AUC_window'
        else:continue
        value=e.get('comparable_value',e.get('value',e.get('mean')))
        if value is None:continue
        unit=e.get('comparable_unit',e.get('unit',e.get('source_unit')))
        eligible=e.get('evaluable_from_source',True) is not False
        reason=e.get('source_quality_blocker')
        src=source(sealed_ref['path'],json_pointer=f'/{next(k for k in ["endpoints","observed_endpoints","endpoint_rows"] if k in data)}/{i}',
                   table=e.get('table_id',e.get('source_table_id',e.get('source_table'))),
                   row=e.get('row_0based',e.get('source_row',e.get('source_all_tr_row'))),
                   column=e.get('column_0based',e.get('source_column',e.get('source_physical_column'))),
                   raw_cell=e.get('source_cell',e.get('source_cell_text',e.get('raw',e.get('cell_native_text')))),
                   original_provenance=e.get('provenance'))
        article=article_for(data)
        if article:src['article']=source(article)
        window=None
        if endpoint=='AUC_window':
            if 'tau' in low:
                window=[0.,12. if s['compound'] in {'clarithromycin','felodipine','itraconazole'} else 24.]
            else:
                match=re.search(r'auc_?0_?(\d+)',low)
                if match:window=[0.,float(match.group(1))]
        # AUC_last is a subject-specific last quantifiable observation, never
        # automatically the last scheduled sample. Keep it distinct from AUC0-t.
        definition='Last quantifiable observation; individual t_last unavailable.' if endpoint=='AUC_last' else ('Terminal extrapolation to infinity.' if endpoint=='AUC_infinity' else None)
        row=record(s,endpoint,value,unit,src,endpoint_key=raw,window_h=window,
                   eligibility=eligible,reason=reason,definition=definition,
                   statistic=e.get('aggregation',e.get('statistic',e.get('population_statistic','unspecified'))),
                   provenance_kind='existing_sealed_endpoint',original_contract=e,
                   dose_normalization='dose_per_body_weight' if 'normalized_by_dose_per_weight' in raw else None)
        if s['compound']=='caffeine' and endpoint=='AUC_last':row['source_quality_flags'].append('Sealed primary_comparable=false because individual last-quantifiable times are absent; retained as descriptive observed AUC_last only.')
        if s['compound']=='repaglinide':row['source_quality_flags'].append('Printed exposure units conflict with independently reported CL/F by factor 1000; not silently corrected.')
        result.append(row)
    return result

def clinical_protocol(s,data):
    article=article_for(data);sample=s['normalized_design'].get('sampling',{})
    row={'scenario_id':s['scenario_id'],'compound':s['compound'],
         'nominal_times_h':sample.get('times_h'),'sampling_source':s['source_protocol'],
         'time_origin':sample.get('time_origin'),
         'route':s['normalized_design'].get('route'),'dose':s['normalized_design'].get('dose'),
         'regimen':s['normalized_design'].get('regimen'),'steady_state':is_steady(s),
         'lloq_ng_ml':None,'lloq_source':None,'lloq_missing_reason':'No unambiguous assay LLOQ transcribed for this exact analyte.'}
    if article and s['compound'] in LLOQ:
        value,index,needle=LLOQ[s['compound']];para=ET.parse(article).getroot().findall('.//p')[index]
        raw=text(para);assert needle in raw,(s['compound'],index,needle,raw)
        row.update(lloq_ng_ml=value,lloq_source=source(article,paragraph=index,text=raw),lloq_missing_reason=None)
    if s['compound']=='caffeine':
        row.update(lloq_ng_ml=10.,lloq_source=source('experiments/blueprint_design/source_audits/caffeine_sherman2022/sherman2022.pdf',page=3,section='2.3 Sampling and analytical procedures',text='The lower limit of quantitation (LLOQ) for caffeine or d9-caffeine was 10 ng/mL.'),lloq_missing_reason=None)
    return row

def build():
    scenarios=[json.loads(line) for line in SCENARIOS.read_text().splitlines()]
    assert len(scenarios)==41 and len({s['scenario_id'] for s in scenarios})==41
    references=[];protocols=[]
    for s in scenarios:
        ref=s['sealed_references'][0]
        assert sha(project_path(ref['path']))==ref['sha256'],ref['path']
        data=json.loads(project_path(ref['path']).read_text())
        references.extend(base_rows(s,data,ref));references.extend(extra_rows(s,data))
        protocols.append(clinical_protocol(s,data))
    for i,r in enumerate(references):
        r['reference_id']=f"{r['scenario_id']}::{r['endpoint_key']}::{i}"
        assert r['value']>0 and math.isfinite(r['value'])
        assert r['endpoint'] in ENDPOINTS and r['source_ref']['sha256']
    missing=[]
    for s in scenarios:
        for endpoint in ENDPOINTS:
            if not any(r['scenario_id']==s['scenario_id'] and r['endpoint']==endpoint for r in references):
                reason='Not reported unambiguously for the registered reference arm in the bound local clinical source.'
                if endpoint=='C_last' and s['compound']=='vancomycin':reason='Table prints concentration C_last with time unit h and time T_last with concentration unit mg/L; ambiguous unit transposition not silently repaired.'
                if endpoint=='CL' and s['compound']=='lisinopril':reason='Only renal clearance is reported; it is not total CL or oral apparent CL/F.'
                missing.append({'scenario_id':s['scenario_id'],'endpoint':endpoint,'value':None,'eligibility':False,'reason':reason})
    return {'schema_version':'pbpk_recommendation_extended_references_v1','scope':'41 frozen execution scenarios; source-bound post hoc endpoint extension; no new study or model fit',
            'source_manifest':source(SCENARIOS),'endpoint_keys':ENDPOINTS,'references':references,'missing':missing,'clinical_protocols':protocols,
            'summary':{'scenarios':len(scenarios),'references':len(references),
                       'new_transcriptions':sum(r['provenance_kind']=='new_same_arm_source_transcription' for r in references),
                       'counts_by_endpoint':dict(Counter(r['endpoint'] for r in references)),
                       'eligible_counts_by_endpoint':dict(Counter(r['endpoint'] for r in references if r['eligibility'])),
                       'protocols_with_lloq':sum(p['lloq_ng_ml'] is not None for p in protocols),
                       'protocols_with_lloq_and_nominal_sampling':sum(p['lloq_ng_ml'] is not None and bool(p['nominal_times_h']) for p in protocols)},
            'limitations':['Aggregate NCA observations do not become individual profiles or independent validation labels.',
                          'AUC_last is not silently converted to a fixed-window AUC; no individual t_last is invented.',
                          'Unknown F is not assumed to be one. CL/F and Vz/F remain distinct from CL and Vz.',
                          'Per-kg source quantities remain per-kg, without multiplying by cohort mean weight.',
                          'Steady-state and single-dose conditions and source statistics are preserved.',
                          'Observed eligibility alone does not imply a matching, successfully executed prediction.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    result=build();encoded=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if args.check:
        assert OUTPUT.read_text()==encoded,'Published reference file is not reproducible.'
    else:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(encoded)
    print(json.dumps(result['summary'],ensure_ascii=False,sort_keys=True))

if __name__=='__main__':main()
