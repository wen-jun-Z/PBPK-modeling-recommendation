"""Source-bound indirect IV administration identity; no numerical execution.

Only original saved Model/Properties/building-block definitions are read. The
supported chain is rate assignment -> original transport -> same named venous
blood drug state, independently corroborated by DrugMass and protocol origin.
A failed link is an error, never a guessed association or route exclusion.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET
from . import source_pkml

ROOT=source_pkml.ROOT
VERSION='source-bound-indirect-iv-associations-v1'
IMPLEMENTATION_FILES=['scripts/recommendation_execution_repairs/event_associations.py',
                      'scripts/recommendation_execution_repairs/source_pkml.py']


def _require(condition, reason):
    if not condition:raise ValueError('EVENT_ASSOCIATION_'+reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ModelIndex:
    def __init__(self, document):
        self.document=document;self.model=document.find('Model');self.paths={};self.ids={};self.node_paths={}
        _require(self.model is not None,'COMPLETE_MODEL_REQUIRED')
        roots=self.model.findall('Root');_require(len(roots)==1,'UNIQUE_MODEL_ROOT_REQUIRED');root=roots[0]
        self.root_name=root.get('name')
        def walk(node,path):
            if node is not root and node.get('id') and node.get('name'):
                path=path+[node.get('name')];key='|'.join(path)
                _require(key not in self.paths and node.get('id') not in self.ids,'DUPLICATE_ENTITY')
                self.paths[key]=node;self.ids[node.get('id')]=key;self.node_paths[id(node)]=key
            for child in node:walk(child,path)
        walk(root,[])
        cache=self.model.find('FormulaCache');_require(cache is not None,'FORMULA_CACHE_REQUIRED')
        definitions=cache.find('Formulas');_require(definitions is not None,'FORMULAS_REQUIRED')
        self.formulas={x.get('id'):x for x in definitions}
        _require(None not in self.formulas and len(self.formulas)==len(definitions),'UNIQUE_FORMULA_IDS_REQUIRED')
        strings=cache.find('StringMap');self.strings={x.get('id'):x.get('s') for x in strings} if strings is not None else {}
        _require(strings is None or len(strings)==len(self.strings),'UNIQUE_STRING_IDS_REQUIRED')

    def node(self,path,tag=None,dimension=None):
        node=self.paths.get(path);_require(node is not None,'EXACT_ENTITY_MISSING:'+path)
        if tag is not None:_require(node.tag in tag,'ENTITY_TYPE_DIFFERS:'+path)
        if dimension is not None:_require(node.get('dim',node.get('dimension'))==dimension,'ENTITY_DIMENSION_DIFFERS:'+path)
        return node

    def resolve(self,raw,context):
        raw=self.strings.get(raw,raw);_require(isinstance(raw,str) and bool(raw),'FORMULA_PATH_REQUIRED')
        if raw in self.paths:return raw
        root_names={self.root_name,self.document.get('name')}-{None,''}
        absolute=[raw[len(name)+1:] for name in root_names if raw.startswith(name+'|') and raw[len(name)+1:] in self.paths]
        if absolute:
            _require(len(set(absolute))==1,'AMBIGUOUS_ABSOLUTE_PATH');return absolute[0]
        tokens=raw.split('|');_require(tokens[0] in {'.','..'},'EXACT_PATH_NOT_RESOLVED:'+raw)
        path=context.split('|')
        for token in tokens:
            if token=='..':_require(bool(path),'PATH_ESCAPES_MODEL');path.pop()
            elif token!='.':path.append(token)
        key='|'.join(path);_require(key in self.paths,'RELATIVE_PATH_NOT_RESOLVED:'+key);return key

    def formula(self,node,expression,bindings):
        formula=self.formulas.get(node.get('formula'))
        _require(formula is not None and formula.tag=='Formula','EXPLICIT_FORMULA_REQUIRED')
        _require(re.sub(r'\s+','',formula.get('formula',''))==expression,'FORMULA_EXPRESSION_DIFFERS')
        refs={};paths=formula.find('Paths');unused_time_paths=0
        for p in paths if paths is not None else []:
            if p.tag=='TimePath':
                raw_alias=p.get('as',p.get('alias'))
                alias=self.strings.get(raw_alias,raw_alias) or 'Time'
                _require(set(p.attrib)<={'as','alias'} and not ('as' in p.attrib and 'alias' in p.attrib) and alias=='Time' and 'Time' not in re.findall(r'[A-Za-z_][A-Za-z_0-9]*',expression),'UNREFERENCED_TIME_PATH_REQUIRED')
                unused_time_paths+=1;_require(unused_time_paths==1,'DUPLICATE_TIME_PATH');continue
            _require(p.tag=='Path','EXPLICIT_PARAMETER_PATH_REQUIRED')
            alias=self.strings.get(p.get('as'),p.get('as'));_require(alias and alias not in refs,'UNIQUE_FORMULA_ALIAS_REQUIRED')
            refs[alias]=self.resolve(p.get('path'),self.node_paths[id(node)])
        _require(refs==bindings,'FORMULA_BINDINGS_DIFFER')
        return {'formula_id':formula.get('id'),'formula_sha256':source_pkml.element_sha(formula),'expression':expression,'resolved_bindings':refs,'unreferenced_time_path_count':unused_time_paths}


def _prove_indirect_iv(index,administration,metadata,*,original_index=None,original_proof=None):
    prefix=administration['event_prefix'];group=index.node(prefix,{'EventGroup'})
    if original_index is not None:
        _require(original_proof is not None and original_proof.get('verified') is True and
                 original_proof.get('event_prefix')==prefix,'VERIFIED_ORIGINAL_IV_EVENT_REQUIRED')
        original_group=original_index.node(prefix,{'EventGroup'})
        _require(group.get('id')==original_group.get('id'),'ORIGINAL_IV_EVENT_ID_DIFFERS')
    _require(group.get('eventGroupType')==administration.get('application_type')=='Intravenous','ONLY_ORIGINAL_INTRAVENOUS_RATE_CHAIN_SUPPORTED')
    rate_path=prefix+'|Application rate';rate=index.node(rate_path,{'Para','Parameter'},'Amount per time')
    _require(not rate.get('rhs') and not rate.get('rHSFormula'),'RATE_RHS_UNSUPPORTED')
    masses=[n for n in group.iter('MoleculeAmount')]
    _require(len(masses)==1,'UNIQUE_APPLICATION_DRUG_STATE_REQUIRED');drug_state=masses[0]
    drug=drug_state.get('name');_require(drug in metadata['compounds'] and drug_state.get('quantityType')=='Drug' and drug_state.get('dim')=='Amount','EXPLICIT_SOURCE_DRUG_IDENTITY_REQUIRED')
    _require(index.node_paths[id(drug_state)]==prefix+'|'+drug,'APPLICATION_DRUG_STATE_PATH_DIFFERS')
    target_path='Organism|VenousBlood|Plasma|'+drug;target=index.node(target_path,{'MoleculeAmount'},'Amount')
    _require(target.get('quantityType')=='Drug' and target.get('name')==drug,'VENOUS_TARGET_DRUG_IDENTITY_DIFFERS')
    transports=[n for n in index.model.iter('Transport') if n.get('sourceAmount')==drug_state.get('id')]
    _require(len(transports)==1,'UNIQUE_DRUG_OUTGOING_TRANSPORT_REQUIRED');transport=transports[0]
    _require(index.node_paths[id(transport)].startswith(prefix+'|'+drug+'|') and transport.get('targetAmount')==target.get('id'),'TRANSPORT_STATE_BINDING_DIFFERS')
    _require(transport.get('dimension',transport.get('dim'))=='Amount per time','TRANSPORT_DIMENSION_DIFFERS')
    transport_formula=index.formula(transport,'ApplicationRate',{'ApplicationRate':rate_path})
    # A shared rate may not secretly administer a second drug elsewhere.
    consumers=[]
    for other in index.model.iter('Transport'):
        f=index.formulas.get(other.get('formula'))
        for p in f.findall('Paths/Path') if f is not None else []:
            raw=index.strings.get(p.get('path'),p.get('path'))
            if isinstance(raw,str) and raw.endswith('|Application rate'):
                if index.resolve(raw,index.node_paths[id(other)])==rate_path:consumers.append(other.get('id'))
    _require(consumers==[transport.get('id')],'RATE_HAS_AMBIGUOUS_TRANSPORT_CONSUMERS')
    schema=prefix+'|ProtocolSchemaItem|';mass=index.node(schema+'DrugMass',{'Para','Parameter'},'Amount')
    index.node(schema+'Dose',{'Para','Parameter'},'Mass')
    weight=index.node(drug+'|Molecular weight',{'Para','Parameter'},'Molecular weight')
    _require([t.get('value') for t in mass.findall('Tags/Tag')]==[drug],'DRUG_MASS_TAG_DIFFERS')
    mass_formula=index.formula(mass,'Dose/MW',{'Dose':schema+'Dose','MW':drug+'|Molecular weight'})
    start=index.node(schema+'Start time',{'Para','Parameter'},'Time')
    infusion=index.node(schema+'Infusion time',{'Para','Parameter'},'Time')
    origins=[];origin_evidence=[]
    if original_index is not None:
        _require(original_proof['drug']==drug and original_proof['rate_parameter_id']==rate.get('id') and
                 original_proof['source_drug_state_id']==drug_state.get('id') and
                 original_proof['venous_target_state_id']==target.get('id') and
                 original_proof['transport_id']==transport.get('id'),'ORIGINAL_IV_STATE_CHAIN_DIFFERS')
    for parameter in (start,infusion):
        origin=parameter.find('Origin')
        if original_index is not None:
            path=index.node_paths[id(parameter)]
            original_parameter=original_index.node(path,{'Para','Parameter'},'Time')
            source_origin=original_parameter.find('Origin')
            _require(source_origin is not None and source_origin.get('bb'),'PROTOCOL_PARAMETER_ORIGIN_REQUIRED')
            # Exported PKML drops Origin annotations. Bind them to the exact
            # unchanged original event/parameter; never invent an origin on XML.
            _require(all(parameter.get(k)==original_parameter.get(k) for k in ('id','name','dim','value')) and
                     parameter.get('value') is not None and
                     all(not n.get(k) for n in (parameter,original_parameter) for k in ('formula','rhs','rHSFormula')),
                     'ORIGINAL_PROTOCOL_PARAMETER_BINDING_DIFFERS')
            _require(origin is None or source_pkml.element_sha(origin)==source_pkml.element_sha(source_origin),
                     'EXPORTED_PROTOCOL_ORIGIN_CONFLICT')
            origin_evidence.append({'path':path,'id':parameter.get('id'),
                'original_parameter_sha256':source_pkml.element_sha(original_parameter),
                'original_origin':dict(source_origin.attrib),'native_origin_present':origin is not None})
            origin=source_origin
        _require(origin is not None and origin.get('bb'),'PROTOCOL_PARAMETER_ORIGIN_REQUIRED');origins.append(origin.get('bb'))
    _require(len(set(origins))==1,'PROTOCOL_ORIGINS_DIFFER')
    bindings=[r for r in metadata['source_protocol_bindings'] if r.get('protocol_id')==origins[0]]
    _require(len(bindings)==1 and bindings[0].get('compound_name')==drug,'PROTOCOL_COMPOUND_BINDING_DIFFERS')
    if original_index is not None:_require(bindings[0]==original_proof['protocol_binding'],'ORIGINAL_PROTOCOL_BINDING_DIFFERS')
    blocks=[c for w in index.document.findall('UsedBuildingBlocks/*') for c in w]
    # UsedBuildingBlocks children are wrappers containing the original block.
    protocols=[n for n in blocks if n.get('id')==origins[0]]
    _require(len(protocols)==1 and protocols[0].tag in {'SimpleProtocol','AdvancedProtocol'},'ORIGINAL_PROTOCOL_BLOCK_REQUIRED')
    _require(protocols[0].get('applicationType') in {None,'Intravenous'},'PROTOCOL_ROUTE_CONFLICT')
    assignments=list(group.iter('EventAssignment'));_require(len(assignments)==2,'EXACT_START_AND_STOP_RATE_ASSIGNMENTS_REQUIRED')
    assignment_proofs=[];kinds=[]
    for node in assignments:
        _require(node.get('changedEntity')==rate.get('id'),'ASSIGNMENT_RATE_ID_DIFFERS')
        raw=node.get('objectPath');child=node.find('ObjectPath')
        if child is not None:
            _require(raw is None,'AMBIGUOUS_ASSIGNMENT_OBJECT_PATH');raw=child.get('path')
        _require(index.resolve(raw,index.node_paths[id(node)])==rate_path,'ASSIGNMENT_OBJECT_PATH_DIFFERS')
        f=index.formulas.get(node.get('formula'));_require(f is not None,'ASSIGNMENT_FORMULA_MISSING')
        if f.tag=='ConstantFormula':
            _require(len(f)==0 and float(f.get('value','nan'))==0,'RATE_STOP_MUST_BE_ZERO');kind='zero_rate_stop';proof={'formula_id':f.get('id'),'formula_sha256':source_pkml.element_sha(f)}
        else:
            kind='drug_mass_over_infusion';proof=index.formula(node,'DrugMass/InfusionTime',{'DrugMass':schema+'DrugMass','InfusionTime':schema+'Infusion time'})
        kinds.append(kind);assignment_proofs.append({'assignment_id':node.get('id'),'target_id':rate.get('id'),'kind':kind,**proof})
    _require(sorted(kinds)==['drug_mass_over_infusion','zero_rate_stop'],'START_STOP_FORMULA_PAIR_REQUIRED')
    return {'event_prefix':prefix,'drug':drug,'application_type':'Intravenous',
        'rate_parameter_id':rate.get('id'),'rate_parameter_path':rate_path,
        'source_drug_state_id':drug_state.get('id'),'source_drug_state_path':index.node_paths[id(drug_state)],
        'venous_target_state_id':target.get('id'),'venous_target_state_path':target_path,
        'transport_id':transport.get('id'),'transport_formula':transport_formula,
        'drug_mass_formula':mass_formula,'molecular_weight_id':weight.get('id'),
        'protocol_binding':deepcopy(bindings[0]),'assignment_proofs':assignment_proofs,
        'application_node_sha256':source_pkml.element_sha(group),'verified':True,
        **({'original_protocol_origin_evidence':origin_evidence} if original_index is not None else {})}


def _enrich_document(document,*,original_document=None):
    metadata=source_pkml.describe_saved_simulation(document);index=ModelIndex(document);proofs=[]
    original_index=None;original_proofs={}
    if original_document is not None:
        _,verified=_enrich_document(original_document)
        original_index=ModelIndex(original_document);original_proofs={p['event_prefix']:p for p in verified}
    for row in metadata['administrations']:
        if row['compound_association_complete']:continue
        _require(not row['administered_compounds'],'PARTIAL_PREEXISTING_ASSOCIATION_REQUIRES_REVIEW')
        proof=_prove_indirect_iv(index,row,metadata,original_index=original_index,
            original_proof=original_proofs.get(row['event_prefix']));proofs.append(proof)
        row['administered_compounds']=[proof['drug']];row['compound_association_complete']=True
        row['compound_association_basis']='Exact original IV assignment-rate-transport-drug-state, DrugMass/MW and protocol-origin chain'
        row['indirect_association_proof']=proof
    return metadata,proofs


def _load(source_path,source_sha256,simulation_id,source_blob_sha256,project_root):
    root=Path(project_root).resolve(strict=True);source=source_pkml.inside(source_path,root)
    _require(source.suffix=='.pksim5' and source_pkml.sha(source)==source_sha256,'ORIGINAL_SOURCE_SHA_MISMATCH')
    with sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        rows=db.execute('SELECT b.Name,c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?',(simulation_id,)).fetchall()
    _require(len(rows)==1,'UNIQUE_SOURCE_SIMULATION_REQUIRED')
    name,blob=rows[0];digest=hashlib.sha256(blob).hexdigest()
    _require(digest==source_blob_sha256,'ORIGINAL_MODEL_BLOB_SHA_MISMATCH')
    document=ET.fromstring(source_pkml.decompress_model_blob(blob))
    _require(document.get('id')==simulation_id and document.get('name')==name,'SOURCE_SIMULATION_IDENTITY_DIFFERS')
    _require(source_pkml.sha(source)==source_sha256,'SOURCE_CHANGED_DURING_READ')
    return source,document


def enrich_source_metadata(source_path,source_sha256,simulation_id,source_blob_sha256,metadata=None,*,project_root=ROOT):
    """Re-derive and enrich one exact original simulation; caller metadata is checked."""
    root=Path(project_root).resolve(strict=True)
    source,document=_load(source_path,source_sha256,simulation_id,source_blob_sha256,root)
    original=source_pkml.describe_saved_simulation(document)
    if metadata is not None:
        _require(all(metadata.get(k)==v for k,v in original.items()),'PROVIDED_METADATA_DIFFERS_FROM_ORIGINAL')
        if 'source_blob_sha256' in metadata:_require(metadata['source_blob_sha256']==source_blob_sha256,'PROVIDED_BLOB_SHA_DIFFERS')
    enriched,proofs=_enrich_document(document)
    if metadata is not None:enriched={**deepcopy(metadata),**enriched}
    enriched['source_blob_sha256']=source_blob_sha256
    audit={'version':VERSION,'source':source_pkml.reference(source,root),'simulation_id':simulation_id,
        'source_blob_sha256':source_blob_sha256,'source_model_sha256':source_pkml.element_sha(document.find('Model')),
        'original_metadata_sha256':_digest(original),'enriched_metadata_sha256':_digest(enriched),
        'enriched_administrations':proofs,'implementation':[source_pkml.reference(root/p,root) for p in IMPLEMENTATION_FILES],
        'observations_or_results_read':False,'ode_run':False,'parameters_modified':False,
        'scope':'Administration identity only; does not remove coadministration or establish scientific applicability.'}
    _require(source_pkml.sha(source)==source_sha256,'SOURCE_CHANGED_DURING_PROOF')
    return {'enriched_metadata':enriched,'audit':audit}


def create_route_absence_certificate(source_path,source_sha256,compound,required_route,*,project_root=ROOT):
    """Audit every saved simulation before asserting no same-drug target route."""
    root=Path(project_root).resolve(strict=True);source=source_pkml.inside(source_path,root)
    _require(isinstance(compound,str) and bool(compound) and required_route in {'Oral','Intravenous'},'EXACT_COMPOUND_AND_ROUTE_REQUIRED')
    inventory=source_pkml.source_simulation_metadata(source,source_sha256,project_root=root)
    simulations=[];observed_routes=set();target_events=0
    for row in inventory['simulations']:
        result=enrich_source_metadata(source,source_sha256,row['simulation_id'],row['source_blob_sha256'],row,project_root=root)
        enriched=result['enriched_metadata'];events=[]
        for event in enriched['administrations']:
            _require(event['compound_association_complete'],'UNRESOLVED_ADMINISTRATION_PRECLUDES_ROUTE_ABSENCE')
            if compound in event['administered_compounds']:
                _require(event['application_type'] in {'Oral','Intravenous'},'UNREGISTERED_TARGET_ROUTE_PRECLUDES_ABSENCE')
                observed_routes.add(event['application_type']);target_events+=1;events.append(deepcopy(event))
        simulations.append({'simulation_id':row['simulation_id'],'name':row['name'],
            'source_blob_sha256':row['source_blob_sha256'],'target_compound_declared':compound in enriched['compounds'],
            'target_administrations':events,'all_administrations_checked':len(enriched['administrations']),
            'metadata_sha256':_digest(enriched),'association_audit':result['audit']})
    _require(target_events>0,'TARGET_COMPOUND_ADMINISTRATION_NOT_FOUND')
    _require(required_route not in observed_routes,'REQUESTED_COMPOUND_ROUTE_EXISTS')
    certificate={'version':VERSION,'source':source_pkml.reference(source,root),
        'compound':compound,'required_route':required_route,'observed_routes':sorted(observed_routes),
        'simulation_count':len(simulations),'target_administration_count':target_events,
        'simulation_inventory':[{'simulation_id':r['simulation_id'],'name':r['name']} for r in inventory['simulations']],
        'simulations':simulations,'complete_source_inventory_checked':True,'route_absence_verified':True,
        'implementation':[source_pkml.reference(root/p,root) for p in IMPLEMENTATION_FILES],
        'observations_or_results_read':False,'ode_run':False,'scientific_applicability_claimed':False,
        'scope':'Exact named compound in this fixed original project; routes of other drugs and protocol display names are never transferred.'}
    _require(source_pkml.sha(source)==source_sha256,'SOURCE_CHANGED_DURING_CERTIFICATE')
    return certificate


def validate_route_absence_certificate(certificate,*,project_root=ROOT,
        source_path=None,source_sha256=None,compound=None,required_route=None):
    """Recompute full inventory/identity proof, rejecting stale or altered certificates."""
    root=Path(project_root).resolve(strict=True)
    if source_path is not None:_require(source_pkml.inside(source_path,root)==source_pkml.inside(certificate['source']['path'],root),'CERTIFICATE_EXPECTED_SOURCE_DIFFERS')
    for expected,actual in [(source_sha256,certificate['source']['sha256']),
                            (compound,certificate['compound']),(required_route,certificate['required_route'])]:
        if expected is not None:_require(expected==actual,'CERTIFICATE_EXPECTED_SCOPE_DIFFERS')
    _require(certificate.get('version')==VERSION,'CERTIFICATE_VERSION_DIFFERS')
    _require(certificate.get('implementation')==[source_pkml.reference(root/p,root) for p in IMPLEMENTATION_FILES],'CERTIFICATE_IMPLEMENTATION_DIFFERS')
    current=create_route_absence_certificate(certificate['source']['path'],certificate['source']['sha256'],
        certificate['compound'],certificate['required_route'],project_root=root)
    _require(certificate==current,'CERTIFICATE_CONTENT_OR_SOURCE_DIFFERS')
    return {'accepted':True,'certificate_sha256':_digest(certificate),'source':current['source'],
            'compound':current['compound'],'required_route':current['required_route'],
            'simulation_count':current['simulation_count'],'observed_routes':current['observed_routes']}
