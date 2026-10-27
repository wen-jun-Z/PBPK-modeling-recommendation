#!/usr/bin/env python3
"""Field-level audit of 60 clinical sources. No fitting or frozen writes.

build: regenerate JSON, flat CSV and one LaTeX appendix table from frozen source
chains and locally cached public bibliographic metadata. --fetch supplements the
bounded public metadata cache (at most four concurrent requests, 30 s/request).
Unknown is retained; absence from repository metadata is not absence in a paper.
The separate Edaravone check scores only already locked dense peaks against the
original reported mean individual peak, with the different estimand explicit.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/blueprint_design/sampling_stability_20261009'
OUT=ROOT/'experiments/blueprint_design/reviewer_revision_20261009/sources'
TABLE=ROOT/'results/tables/blueprint_reviewer_sources.tex'
MAX_BYTES=12*1024*1024

def read(p): return json.loads(Path(p).read_text())
def ref(p,pointer='$'):
 p=Path(p).resolve();assert p.is_relative_to(ROOT) and p.is_file()
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pointer=pointer)
def checked(r):
 p=(ROOT/r['path']).resolve();assert p.is_relative_to(ROOT)
 if r.get('sha256') and ref(p)['sha256']!=r['sha256']: raise ValueError('Source hash changed: '+str(p))
 return p,read(p)
def unwrap(d,cid):
 if 'cases' in d:
  m=[x for x in d['cases'] if x.get('case_id')==cid]
  assert len(m)==1;d=m[0]
 return d.get('clinical_reference',d)
def fact(value, source=None, status='documented', note=None):
 if value in (None,'','.'): value=None; status='not_verified'
 f=dict(value=value,status=status,evidence=[] if source is None else [source])
 if note:f['note']=note
 return f

def chain(raw,p):
 """Follow only declared clinical source links, never scan arbitrary data files."""
 q=[(raw,p)];out=[];seen=set()
 while q:
  d,p=q.pop(0)
  if str(p) in seen: continue
  seen.add(str(p));out.append((d,p))
  for key in ('source','source_path','original_source','clinical_protocol_review'):
   s=d.get(key)
   if isinstance(s,str):s={'path':s}
   if isinstance(s,dict) and 'source_id' in s:s={'path':s['source_id'],'sha256':s.get('source_checksum'),'location':s.get('source_location','')}
   if not isinstance(s,dict) or not s.get('path'):continue
   sp=(ROOT/s['path']).resolve()
   if not sp.is_relative_to(ROOT) or not sp.is_file() or sp.suffix!='.json':continue
   sp,sd=checked(s)
   if 'ObservedData' in sd:
    loc=s.get('location',''); name=d.get('source_observation_name')
    if not name and loc.startswith('ObservedData['):name=loc[len('ObservedData['):-1]
    items=[(i,x) for i,x in enumerate(sd['ObservedData']) if x.get('Name')==name]
    if len(items)==1:
     i,x=items[0];meta={v['Name']:v.get('Value') for v in x.get('ExtendedProperties',[])}
     out.append((dict(original_observation_metadata=meta,source_observation_name=name,repository_pointer=f'/ObservedData/{i}',source_columns=[{k:v for k,v in z.items() if k in ('Name','Unit','Dimension','DataInfo','QuantityInfo')} for z in x.get('Columns',[])]),sp))
   else:q.append((sd,sp))
 return out

def ids(text):
 text=str(text);out={}
 m=re.search(r'(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/|pubmed/|pmid[:/ ]+)(\d+)',text,re.I)
 if m:out['pmid']=m.group(1)
 m=re.search(r'10\.\d{4,9}/[^\s"<>?]+',text,re.I)
 if m:out['doi']=re.split(r'/asset/|[&?]',m.group(0))[0].rstrip('.,;)');out['doi']=re.sub(r'\.s\d+$','',out['doi'])
 m=re.search(r'PMC\d+',text)
 if m:out['pmcid']=m.group(0)
 return out

def baseline_record(conf,role):
 key='development_source' if role=='development' else 'reference_source';p,doc=checked(conf[key]);d=unwrap(doc,conf['case_id']);nodes=chain(d,p)
 fields={}
 def setf(k,v,nd,np,ptr,status='documented',note=None):
  if v not in (None,'','.') and (k not in fields or fields[k]['value'] is None):fields[k]=fact(v,ref(np,ptr),status,note)
 for nd,np in nodes:
  meta=nd.get('original_metadata',nd.get('original_observation_metadata',nd.get('metadata',{})));mp=nd.get('repository_pointer','')+'/ExtendedProperties' if nd.get('repository_pointer') else '/original_metadata'
  setf('study_id',nd.get('study_id',nd.get('study_group_id')),nd,np,'/study_id')
  for k,mk in [('n','N'),('curve_kind','Data type'),('analyte','Molecule'),('matrix','Compartment'),('sampling_site','Organ'),('curve_locator','Source')]:setf(k,meta.get(mk) if not (k=='curve_locator' and (':' in str(meta.get(mk,'')) or str(meta.get(mk,''))=='observations')) else None,nd,np,mp)
  for k in ['clinical_reference_url']:
   setf('primary_url',nd.get(k),nd,np,'/'+k)
  setf('primary_url',meta.get('Reference'),nd,np,mp)
  curve=nd.get('curve',{})
  setf('curve_locator',curve.get('location'),nd,np,'/curve/location')
  setf('n',curve.get('sample_size'),nd,np,'/curve/sample_size')
  statistic=curve.get('statistic',nd.get('statistic'))
  if statistic:
   setf('curve_kind','individual' if statistic=='individual' else 'aggregate',nd,np,'/curve/statistic')
   if statistic in ('arithmetic_mean','geometric_mean','median','individual'):setf('mean_type',statistic,nd,np,'/curve/statistic')
  setf('matrix',nd.get('source_matrix',curve.get('matrix',nd.get('matrix'))),nd,np,'/source_matrix' if 'source_matrix'in nd else '/matrix')
  source=nd.get('primary_source',nd.get('source',{}));source=source if isinstance(source,dict) else {}
  if source and not source.get('location','').startswith('ObservedData['):
   setf('primary_url',source.get('url'),nd,np,'/source/url')
   setf('citation',source.get('citation',source.get('title')),nd,np,'/source')
   setf('curve_locator',source.get('figure',source.get('location')) if source.get('location')!='observations' else None,nd,np,'/source')
  pop=nd.get('population_source',{})
  setf('n',pop.get('n',pop.get('sample_size')),nd,np,'/population_source')
  setf('population',nd.get('population',pop.get('study_description')),nd,np,'/population_source')
  dig=curve.get('digitization',nd.get('digitization',{}));dig=dig if isinstance(dig,dict) else {}
  setf('extraction_method',dig.get('method',curve.get('extraction')),nd,np,'/digitization')
  setf('blq_policy',dig.get('blq_policy'),nd,np,'/curve/digitization/blq_policy')
  setf('lloq_ng_ml',dig.get('quantitation_limit_ng_ml',dig.get('LLOQ_ng_ml')),nd,np,'/digitization')
  obsname=nd.get('source_observation_name','')
  if 'arith Mean' in obsname:setf('mean_type','arithmetic_mean',nd,np,'/source_observation_name')
  elif 'geo Mean' in obsname:setf('mean_type','geometric_mean',nd,np,'/source_observation_name')
  if 'arith Mean' in obsname or 'geo Mean' in obsname:setf('curve_kind','aggregate',nd,np,'/source_observation_name')
  if 'indiv.' in obsname or '.PatID.' in obsname:setf('curve_kind','individual',nd,np,'/source_observation_name')
  elif 'agg.' in obsname or 'Mean' in obsname or 'mean' in obsname:setf('curve_kind','aggregate',nd,np,'/source_observation_name')
  mt=re.search(r'n=(\d+)',obsname)
  if mt:setf('n',int(mt.group(1)),nd,np,'/source_observation_name')
  if 'repository_pointer'in nd:
   setf('extraction_method','OSP public observed-data export; original digitization details not verified',nd,np,nd['repository_pointer'])
   for v in nd.get('source_columns',[]):
    data=v.get('DataInfo',{});typ=data.get('Origin')
    qpath=v.get('QuantityInfo',{}).get('Path','')
    if qpath.endswith('ArithmeticMean'):setf('mean_type','arithmetic_mean',nd,np,nd['repository_pointer']+'/Columns/0/QuantityInfo')
    if qpath.endswith('GeometricMean'):setf('mean_type','geometric_mean',nd,np,nd['repository_pointer']+'/Columns/0/QuantityInfo')
    if typ=='Observation' and 'pkdata_pksim' in data.get('Source',''):
     fields['extraction_method']=fact('Sponsor summary-data spreadsheet imported into public OSP snapshot; not figure digitization',ref(np,nd['repository_pointer']))
  # Nested source files carry original bibliography even when the nearest source is only an export.
  for s in [source,*nd.get('source_files',[])]:
   if not isinstance(s,dict):continue
   for k,v in ids(json.dumps({k:v for k,v in s.items() if k in ('url','publication_doi','citation')})).items():setf(k,v,nd,np,'/source')
  for k,v in ids(' '.join(str(nd.get(k,'')) for k in ('study_id','study_group_id','clinical_reference_url'))+' '+str(meta.get('Reference',''))).items():setf(k,v,nd,np,'/source')
 # Request settings are simulation conditions, not observed participant demographics.
 reqref=conf['requests']['evidence'][role];rp,req=checked(reqref)
 for k,v in req['dosing'].items():fields[k]=fact(v,ref(rp,'/dosing/'+k),note='Frozen simulation protocol; original-source confirmation reported separately.')
 fields['simulation_human']=fact(req['human'],ref(rp,'/human'),note='Reference/adapted model conditions; not necessarily observed participant means.')
 curve=d.get('curve',{})
 ts=curve.get('time_h') or [r['time_min']/60 for r in d.get('observations',[])]
 fields['observed_window_h']=fact([min(ts),max(ts)],ref(p,'/curve/time_h' if curve else '/observations'))
 fields['observed_point_count']=fact(len(ts),ref(p,'/curve/time_h' if curve else '/observations'))
 fields['concentration_unit']=fact(curve.get('unit',d.get('unit')),ref(p,'/curve/unit' if curve else '/unit'))
 if role=='target':
  for k in ('lloq_ng_ml','lod_ng_ml','calibration_lower_bound_ng_ml'):
   if conf.get('nca_policy',{}).get(k) is not None:fields[k]=fact(conf['nca_policy'][k],ref(RUN/'protocol.json',f'/cases/{conf["case_id"]}/nca_policy/{k}'))
 for k in ['citation','primary_url','doi','pmid','pmcid','registry','curve_locator','n','curve_kind','mean_type','analyte','matrix','sampling_site','population','lloq_ng_ml','blq_policy','extraction_method']:
  fields.setdefault(k,fact(None,note='Not established in audited source fields; not evidence that the original paper omits it.'))
 if fields['analyte']['value'] is None: fields['analyte']=fact(conf['drug'],ref(p),status='curated_parent_identity',note='Curated parent-drug identity; original assay specificity needs original-source confirmation.')
 if fields['curve_kind']['value']=='aggregated':fields['curve_kind']['value']='aggregate'
 if fields['curve_kind']['value']=='individual':fields['mean_type']=fact('not_applicable_individual',fields['curve_kind']['evidence'][0]);fields['n']=fact(1,fields['curve_kind']['evidence'][0],note='One evaluated participant, not total study enrollment.')
 return dict(case_id=conf['case_id'],drug=conf['drug'],role=role,fields=fields,source_chain=[ref(np) for _,np in nodes],frozen_caveats=conf.get('sensitivity_flags',[]),reported_scalar_endpoints=d.get('endpoints',{}))

def fetch_metadata(rows):
 queries=set()
 for r in rows:
  f=r['fields']
  if f['pmid']['value']:queries.add('EXT_ID:'+str(f['pmid']['value'])+' AND SRC:MED')
  elif f['doi']['value']:queries.add('DOI:"'+f['doi']['value']+'"')
  elif f['pmcid']['value']:queries.add('PMCID:'+f['pmcid']['value'])
 def one(q):
  url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode(dict(query=q,format='json',resultType='core',pageSize=2))
  try:
   with urllib.request.urlopen(url,timeout=30) as r:raw=r.read(MAX_BYTES+1)
   if len(raw)>MAX_BYTES:raise ValueError('oversize')
   data=json.loads(raw);return q,dict(url=url,response=data)
  except Exception as e:return q,dict(url=url,error=type(e).__name__+': '+str(e))
 path=OUT/'bibliographic_metadata.json';cache=read(path) if path.exists() else {}
 with ThreadPoolExecutor(max_workers=4) as ex:
  for q,result in ex.map(one,sorted(queries-{k for k,v in cache.items() if 'error' not in v})):cache[q]=result
 path.write_text(json.dumps(cache,ensure_ascii=False,indent=2)+'\n')

def enrich(rows):
 path=OUT/'bibliographic_metadata.json';cache=read(path) if path.exists() else {}
 extra=OUT/'source_corrections.json';over=read(extra) if extra.exists() else {}
 for r in rows:
  f=r['fields']; amendments=over.get('records',{}).get(r['drug']+'|'+r['role'],{})
  for k,v in amendments.items():
   v=dict(v);v['evidence']=[{**e,**ref(ROOT/e['path'],e.get('pointer','$'))} for e in v.get('evidence',[])];f[k]=v
  options=[]
  for key,doc in cache.items():
   vals=doc.get('response',{}).get('resultList',{}).get('result',[])
   if len(vals)!=1:continue
   m=vals[0]
   if any(f[k]['value'] and str(f[k]['value']).casefold()==str(m.get(k if k!='pmid' else 'id','')).casefold() for k in ('pmid','doi','pmcid')):options.append((key,m))
  if options:
   key,m=options[0];e=ref(path,'/'+key+'/response/resultList/result/0')
   ji=m.get('journalInfo',{});j=ji.get('journal',{}).get('title','');pages=m.get('pageInfo','')
   f['citation']=fact(f"{m.get('authorString','')} {m.get('title','')} {j}. {m.get('pubYear','')};{ji.get('volume','')}({ji.get('issue','')}):{pages}.",e)
   for k,mk in [('pmid','id'),('doi','doi'),('pmcid','pmcid')]:
    if m.get(mk):f[k]=fact(m[mk],e)
   if not f['primary_url']['value']:f['primary_url']=fact('https://doi.org/'+m['doi'] if m.get('doi') else 'https://pubmed.ncbi.nlm.nih.gov/'+m['id']+'/',e)
  # Add any corrected values again so an explicit curated citation is preserved.
  for k,v in amendments.items():
   v=dict(v);v['evidence']=[{**e,**ref(ROOT/e['path'],e.get('pointer','$'))} for e in v.get('evidence',[])];f[k]=v
  if f['curve_kind']['value']=='individual':
   f['mean_type']=fact('not_applicable_individual',f['curve_kind']['evidence'][0]);f['n']=fact(1,f['curve_kind']['evidence'][0],note='One selected participant, not total enrollment.')
 return over

def pair_audit(c,rows,corrections):
 a,b=rows;fa,fb=a['fields'],b['fields'];flags=set(c.get('sensitivity_flags',[]));pair=corrections.get('pairs',{}).get(c['drug'],{})
 ident=lambda f:f['doi']['value'] or f['pmid']['value'] or f['registry']['value'] or (f['primary_url']['value'] if f['primary_url']['value'] and any(z in f['primary_url']['value'] for z in ('accessdata.fda.gov','patents.google.com')) else None)
 knownid=bool(ident(fa) and ident(fb))
 if not knownid:flags.add('primary_identity_incompletely_verified')
 same=knownid and ident(fa)==ident(fb)
 relationship=('same_publication_study_relation_unresolved' if same else 'different_publications_participant_nonoverlap_not_verified') if knownid else 'study_identity_incompletely_verified'
 if c['drug']=='Remdesivir':relationship='same_publication_distinct_registered_studies'
 if not c.get('cross_study_pair',True):relationship='same_study_distinct_parts_or_arms'
 if relationship=='same_publication_study_relation_unresolved':flags.add('same_publication_study_relation_unresolved')
 for r in rows:
  f=r['fields']
  if f['matrix']['value'] and str(f['matrix']['value']).lower()!='plasma':flags.add('matrix_approximation')
  if f['sampling_site']['value'] and 'arterial' in str(f['sampling_site']['value']).lower():flags.add('arterial_peripheral_approximation')
  if f['curve_locator']['value'] is None:flags.add('original_curve_locator_unverified')
  if f['n']['value'] is None:flags.add('curve_sample_size_unverified')
  if f['curve_kind']['value'] is None:flags.add('curve_aggregation_unverified')
  if f['mean_type']['value'] is None:flags.add('mean_type_unverified')
  if f['lloq_ng_ml']['value'] is None:flags.add('lloq_unverified')
 if fa['curve_kind']['value'] and fb['curve_kind']['value'] and fa['curve_kind']['value']!=fb['curve_kind']['value']:flags.add('individual_aggregate_transfer')
 if fb['observed_window_h']['value'][0]>fb['infusion_minutes']['value']/60:flags.add('target_first_sample_after_infusion')
 flags.update(pair.get('add_flags',[]));flags.difference_update(pair.get('remove_flags',[]))
 relationship=pair.get('relationship',relationship)
 # Metadata based filters: never consult any prediction or its error.
 categories={
 'matrix_or_sampling_site_approximation': any(x in flags for x in ['matrix_approximation','arterial_peripheral_approximation','serum_development_plasma_target','therapeutic_to_microdose','peripheral_to_arterial_plasma']),
 'same_study_arms':relationship=='same_study_distinct_parts_or_arms',
 'participant_overlap_flagged':any('overlap' in x for x in flags),
 'source_identity_or_locator_unverified':any(x in flags for x in ['primary_identity_incompletely_verified','same_publication_study_relation_unresolved']),
 'population_or_summary_transfer':any(x in flags for x in ['individual_aggregate_transfer','healthy_to_surgical_population_transfer','recovery_after_myocardial_infarction','oncology_population','surgical_population','male_to_female_conditions']),
 'patent_source': 'patent_target_source' in flags,
 'quantitation_or_summary_metadata_incomplete':any(x in flags for x in ['curve_sample_size_unverified','curve_aggregation_unverified','mean_type_unverified','lloq_unverified']),
 }
 excluded=[k for k,v in categories.items() if v and k!='quantitation_or_summary_metadata_incomplete']
 targetcluster=pair.get('target_cluster_override') or ('doi:'+str(fb['doi']['value']).lower() if fb['doi']['value'] else ('pmid:'+str(fb['pmid']['value']) if fb['pmid']['value'] else 'study:'+c.get('cluster_id',c['study_id'])))
 return dict(case_id=c['case_id'],drug=c['drug'],relationship=relationship,participant_nonoverlap_status=pair.get('participant_nonoverlap_status','not_verified'),participant_overlap_evidence=pair.get('evidence',[]),target_source_cluster=targetcluster,original_cluster=c.get('cluster_id',c['study_id']),risk_flags=sorted(flags),sensitivity_categories=categories,metadata_quality_subset_included=not excluded,metadata_quality_subset_exclusion_reasons=excluded,fully_documented_subset_included=not excluded and not categories['quantitation_or_summary_metadata_incomplete'],selection_used_prediction_error=False,notes=pair.get('notes',[]))

def esc(x):
 s=str(x).replace('–','--').replace('−','-').replace('µ','u').replace('μ','u').replace('±','+/-').replace('≤','<=')
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('$',r'\$')]:s=s.replace(a,b)
 return s

def table(rows,pairs):
 h=r'''\begingroup
\captionsetup[table]{name=Table,singlelinecheck=false}\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}
\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}\setlength{\LTcapwidth}{\linewidth}
\fontsize{7.5}{9.5}\selectfont
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.065\linewidth}>{\raggedright\arraybackslash}p{.405\linewidth}>{\raggedright\arraybackslash}p{.47\linewidth}@{}}
\caption{Clinical-source pairing audit for 30 drugs. D: development; T: target. NV: not verified from available source fields; it does not imply not reported.}\label{tab:blueprint_reviewer_sources}\\
\toprule Side & Original source and clinical curve & Measurement, protocol and source limitations\\\midrule
\endfirsthead
\multicolumn{3}{l}{\textbf{Clinical-source pairing audit (continued)}}\\
\toprule Side & Original source and clinical curve & Measurement, protocol and source limitations\\\midrule
\endhead
\bottomrule\endfoot
'''
 out=[h];by={(r['case_id'],r['role']):r for r in rows}
 def val(f,k):return f[k]['value'] if f[k]['value'] is not None else 'NV'
 for p in pairs:
  out.append(r'\rowcolor{blue!7}\multicolumn{3}{@{}p{\linewidth}@{}}{\strut\textbf{'+esc(p['drug'])+r'}; '+esc(p['relationship'].replace('_',' '))+r'.}\\*'+'\n')
  for role in ['development','target']:
   f=by[p['case_id'],role]['fields'];cite=val(f,'citation');url=f['primary_url']['value'];shortid='DOI '+str(f['doi']['value']) if f['doi']['value'] else ('PMID '+str(f['pmid']['value']) if f['pmid']['value'] else val(f,'study_id'))
   locator=val(f,'curve_locator')
   # A repository array path is not a primary-paper figure/page locator.
   if str(locator).startswith('datasets['):locator='Original figure/page NV; repository locator retained in machine-readable audit.'
   left=esc(cite)+r'\newline '+(r'\href{'+url.replace('%',r'\%')+r'}{'+esc(shortid)+r'}' if url else esc(shortid))+r'\newline '+esc(locator)
   right=esc(f"n={val(f,'n')}; {val(f,'curve_kind')}; {val(f,'mean_type')}. {val(f,'analyte')}; {val(f,'matrix')}; {val(f,'sampling_site')}.")+r'\newline '
   right+=esc(f"IV {val(f,'dose_mg'):.4g} mg / {val(f,'infusion_minutes'):.4g} min; retained observations {val(f,'observed_window_h')[0]:.4g}--{val(f,'observed_window_h')[1]:.4g} h.")+r'\newline '
   right+=esc(f"Population: {val(f,'population')}.")+r'\newline '
   right+=esc(f"LLOQ (ng/mL): {val(f,'lloq_ng_ml')}; BLQ: {val(f,'blq_policy')}.")+r'\newline '+esc(val(f,'extraction_method'))
   out.append(('D' if role=='development' else 'T')+' & '+left+' & '+right+r'\\'+'\n')
  reason=', '.join(p['metadata_quality_subset_exclusion_reasons']) or 'included; participant nonoverlap is not individually certified'
  out.append(r'\multicolumn{3}{@{}p{\linewidth}@{}}{\emph{Metadata-quality sensitivity:} '+esc(reason.replace('_',' '))+r'.}\\\addlinespace[4pt]'+'\n')
 out.append(r'''\end{longtable}
\noindent\footnotesize All cases remain in the full-cohort analysis. Source filters depend only on metadata, not errors. Aggregate-curve NCA is not a mean of individual NCA. Unverified LLOQ or mean type is retained explicitly. Complete source hashes, field-level evidence, original study codes, reported scalar endpoints, proxy human conditions and source-risk flags are provided in the machine-readable pairing audit.
\endgroup
''');TABLE.write_text(''.join(out))

def edaravone_peak_check(protocol):
 """Compare locked dense peaks; never replace sampling-matched references."""
 import xml.etree.ElementTree as ET
 import numpy as np
 case=next(c for c in protocol['cases'] if c['drug']=='Edaravone')
 source=ROOT/'experiments/blueprint_design/clinical_complete_reference_20260928/sources/clinical_other/edaravone_shimizu2021_evaluation.xml'
 tree=ET.parse(source);table3=tree.find('.//table-wrap[@id="cpdd952-tbl-0003"]')
 if table3 is None:
  table3=next(x for x in tree.findall('.//table-wrap') if ''.join(x.find('label').itertext()).strip()=='Table 3')
 rows=[[''.join(x.itertext()).strip() for x in r.findall('td')] for r in table3.findall('.//tbody/tr')]
 iv=next(r for r in rows if r[:2]==['IV (60 mg)','Arithmetic mean']);original_peak=float(iv[3]);assert original_peak==1253
 rp,raw=checked(case['reference_source']);clinical=unwrap(raw,case['case_id']);mw=float(read(ROOT/case['physchem'])['identity']['molecular_weight_g_mol'])
 if 'curve'in clinical:
  curve=clinical['curve'];ts=np.asarray(curve['time_h']);ys=np.asarray(curve.get('concentration',curve.get('concentration_ng_ml',[])),dtype=float)
  if not len(ys):
   for key,v in curve.items():
    if isinstance(v,list) and len(v)==len(ts) and key!='time_h':ys=np.asarray(v,dtype=float);break
  unit=curve['unit']
 else:ts=np.asarray([x['time_min']/60 for x in clinical['observations']]);ys=np.asarray([x['concentration_umol_per_l'] for x in clinical['observations']]);unit='umol/l'
 factor={'umol/l':mw,'ng/mL':1.,'mg/L':1000.,'mg/l':1000.}[unit];ys=ys*factor
 choices=read(RUN/'choices.json')['choices'];outputs=[]
 for layout in ['early_four','late_four']:
  srcs=choices[case['case_id']+'__'+layout]['cmax']['sources']
  for policy in ['no_addition_repaired','time_repaired','stable_repaired']:
   respath,res=checked(srcs[policy]['source']);predpath,pred=checked(res['prediction']);t=np.asarray(pred['time_min'])/60;c=np.asarray(pred['concentration'])*{'umol/l':mw,'ng/mL':1.,'mg/L':1000.,'mg/l':1000.}[pred['unit']]
   dense=float(c.max());matched=float(np.interp(ts,t,c).max());o=float(ys.max())
   outputs.append(dict(layout=layout,policy=policy,prediction=ref(predpath),dense_peak_ng_ml=dense,dense_peak_time_h=float(t[c.argmax()]),reported_peak_ng_ml=original_peak,descriptive_fold_ratio_to_reported_mean_individual_peak=max(dense/original_peak,original_peak/dense),sampling_matched_peak_ng_ml=matched,retained_curve_peak_ng_ml=o,sampling_matched_fold_error=max(matched/o,o/matched)))
 audit=dict(source=ref(source,'Table3: IV60mg arithmetic mean row, Cmax; Methods: IV sampling schedule; Figure1'),curve_reference=ref(rp),n=42,reported_statistic='arithmetic mean of individual Cmax',reported_cv_percent=18.3,first_retained_observation_h=float(ts.min()),target_infusion_duration_h=1.,limitation='Original study sampled 0.25--1.75h; these early crowded points were not digitized in the frozen export. Retained sampling maximum is not actual clinical peak. Dense reference-adult peak versus arithmetic mean individual Cmax is descriptive and not the same estimand; does not replace sampled-curve error or count as independent validation.',rows=outputs)
 (OUT/'edaravone_peak_check.json').write_text(json.dumps(audit,indent=2)+'\n')

def build(fetch=False):
 OUT.mkdir(parents=True,exist_ok=True);p=read(RUN/'protocol.json');rows=[baseline_record(c,r) for c in p['cases'] for r in ['development','target']]
 enrich(rows)
 if fetch:
  fetch_metadata(rows)
  rows=[baseline_record(c,r) for c in p['cases'] for r in ['development','target']]
 corrections=enrich(rows);pairs=[pair_audit(c,rows[i*2:i*2+2],corrections) for i,c in enumerate(p['cases'])]
 assert len(rows)==60 and len(pairs)==30
 for r in rows:
  for f in r['fields'].values():
   assert f['value'] is None or f['evidence'],(r['drug'],f)
   for e in f['evidence']:
    assert ref(ROOT/e['path'])['sha256']==e['sha256']
  assert r['fields']['dose_mg']['value']>0 and r['fields']['infusion_minutes']['value']>=0
 audit=dict(version='blueprint-clinical-source-audit-1',protocol=ref(RUN/'protocol.json'),policy='All frozen observations preserved. Fields unknown unless supported. Source-quality subsets are retrospective sensitivity analyses, not independent validation. No prediction errors used in source categorization.',records=rows,pairs=pairs,summary=dict(drugs=30,source_sides=60,canonical_target_clusters=len({x['target_source_cluster'] for x in pairs}),metadata_quality_subset_n=sum(x['metadata_quality_subset_included'] for x in pairs),fully_documented_subset_n=sum(x['fully_documented_subset_included'] for x in pairs),unknown_field_counts={k:sum(r['fields'][k]['value'] is None for r in rows) for k in rows[0]['fields'] if all(k in r['fields'] for r in rows)}))
 (OUT/'source_pairing.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 names=['case_id','drug','role',*dict.fromkeys(k for r in rows for k in r['fields'])]
 with (OUT/'source_pairing.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=names,extrasaction='ignore');w.writeheader()
  for r in rows:w.writerow({**{k:r[k] for k in ['case_id','drug','role']},**{k:json.dumps(f['value'],ensure_ascii=False) if isinstance(f['value'],(dict,list)) else f['value'] for k,f in r['fields'].items()}})
 table(rows,pairs);edaravone_peak_check(p);print(json.dumps(audit['summary'],indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fetch',action='store_true');args=ap.parse_args();build(args.fetch)
