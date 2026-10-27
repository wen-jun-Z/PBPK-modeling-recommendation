"""Extract an existing model from a PK-Sim project, without rebuilding it.

Only SIMULATIONS -> BUILDING_BLOCKS.ContentId -> CONTENTS.Data is read.
Results, observed data, fitted outcomes and project history are never queried.
The wrapper is a candidate artifact until the native reader/solver validates it.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import shutil
import sys
import xml.etree.ElementTree as ET
import zlib
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from recommendation_candidate_migration import inside, sha, reference, save

VERSION = "preserved-project-model-extraction-1"
MAX_COMPRESSED_BYTES = 64 * 1024**2
MAX_XML_BYTES = 128 * 1024**2
CORE_EXPORT_SCRIPT = "src/pbpk_kb/model_runtime/R/recommendation_saved_model_core_export.R"


def simulation_inventory(source_path, source_sha256, *, project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    path = inside(source_path, root)
    if path.suffix != ".pksim5" or sha(path) != source_sha256:
        raise ValueError("SOURCE_PROJECT_BINDING_MISMATCH")
    with sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT s.SimulationId,b.Name,s.SimulationMode,b.ContentId,length(c.Data) "
                          "FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId "
                          "JOIN CONTENTS c ON c.Id=b.ContentId ORDER BY b.Name,s.SimulationId").fetchall()
    if sha(path) != source_sha256:
        raise ValueError("SOURCE_PROJECT_CHANGED")
    return [{"simulation_id":r[0],"name":r[1],"mode":r[2],"content_id":r[3],"compressed_bytes":r[4]}
            for r in rows]


def decompress_model_blob(blob):
    """Read the single PK-Sim ZipInputStream entry, including ZIP64 sizes.

    PK-Sim blobs deliberately have no central ZIP directory. Bounds, CRC32,
    local sizes and the deflate end marker are all checked independently.
    """
    if not isinstance(blob,bytes) or not 30 <= len(blob) <= MAX_COMPRESSED_BYTES:
        raise ValueError("MODEL_BLOB_SIZE_INVALID")
    sig,version,flags,method,tm,dt,crc,csize,usize,nlen,xlen=struct.unpack("<IHHHHHIIIHH",blob[:30])
    # Bit 11 only changes filename encoding (UTF-8); it does not encrypt or
    # change the deflate payload. Newer official serializers set this bit.
    if sig!=0x04034b50 or flags not in (0,0x0800) or method!=8 or version>45:
        raise ValueError("UNSUPPORTED_MODEL_BLOB_HEADER")
    if flags==0x0800:
        try:blob[30:30+nlen].decode('utf-8')
        except UnicodeDecodeError as exc:raise ValueError('INVALID_UTF8_ZIP_ENTRY_NAME') from exc
    start=30+nlen+xlen
    if start>=len(blob):raise ValueError("TRUNCATED_MODEL_BLOB")
    extra=blob[30+nlen:start];offset=0
    while offset+4<=len(extra):
        kind,length=struct.unpack("<HH",extra[offset:offset+4]);data=extra[offset+4:offset+4+length];offset+=4+length
        if len(data)!=length:raise ValueError("TRUNCATED_MODEL_BLOB_EXTRA")
        if kind==1:
            cursor=0
            if usize==0xffffffff:
                if len(data)<cursor+8:raise ValueError("MISSING_ZIP64_SIZE")
                usize=struct.unpack("<Q",data[cursor:cursor+8])[0];cursor+=8
            if csize==0xffffffff:
                if len(data)<cursor+8:raise ValueError("MISSING_ZIP64_SIZE")
                csize=struct.unpack("<Q",data[cursor:cursor+8])[0]
    if not 0<usize<=MAX_XML_BYTES or not 0<csize<=MAX_COMPRESSED_BYTES:
        raise ValueError("MODEL_BLOB_DECLARED_SIZE_INVALID")
    if start+csize!=len(blob):raise ValueError("MODEL_BLOB_TRAILING_OR_MISSING_DATA")
    decompressor=zlib.decompressobj(-15)
    data=decompressor.decompress(blob[start:],MAX_XML_BYTES+1)
    if len(data)!=usize or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise ValueError("MODEL_BLOB_DECOMPRESSION_INCOMPLETE")
    if (zlib.crc32(data)&0xffffffff)!=crc:raise ValueError("MODEL_BLOB_CRC_MISMATCH")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML_ENTITIES_FORBIDDEN")
    return data


def element_sha(element):
    return hashlib.sha256(ET.canonicalize(ET.tostring(element,encoding="unicode")).encode()).hexdigest()


def saved_settings(document):
    """Select the one original saved settings node without changing its XML.

    PK-Sim maps the Simulation.Settings property, while legacy projects and
    simulation-transfer XML can use SimulationSettings. Both are the same
    settings object; duplicate or mixed aliases are never guessed between.
    """
    if not isinstance(document, ET.Element):
        raise ValueError('EXACTLY_ONE_SAVED_SIMULATIONSETTINGS_REQUIRED')
    nodes=[child for child in document if child.tag in {'Settings','SimulationSettings'}]
    if len(nodes)!=1:
        raise ValueError('EXACTLY_ONE_SAVED_SIMULATIONSETTINGS_REQUIRED')
    return nodes[0]


def model_parameter_inventory(model):
    """Exact saved parameter paths/current constants, without evaluating formulas.

    PK-Sim and OSPSuite serialize the same parameter classes using different
    element names. IDs and native values remain in the returned records.
    """
    rows={};root=model.find('Root')
    if root is None:raise ValueError('SAVED_MODEL_ROOT_REQUIRED')
    def walk(element,path):
        if element is not root and 'id' in element.attrib and 'name' in element.attrib:
            path=path+[element.get('name')]
            if element.tag in {'Para','DistPara','Parameter','DistributedParameter'}:
                key='|'.join(path)
                if key in rows:raise ValueError('DUPLICATE_MODEL_PARAMETER_PATH:'+key)
                rows[key]={'path':key,'id':element.get('id'),'xml_type':element.tag,
                           'value':element.get('value'),'dimension':element.get('dim'),
                           'display_unit':element.get('displayUnit'),
                           'formula_id':element.get('formula'),
                           'rhs_formula_id':element.get('rhs',element.get('rHSFormula'))}
        for child in element:walk(child,path)
    walk(root,[])
    return rows


def describe_saved_simulation(document):
    """Source-only selection metadata; numeric XML values are base-unit values."""
    parameters=model_parameter_inventory(document.find('Model'))
    blocks=document.find('UsedBuildingBlocks')
    compounds=[];protocols=[];formulations=[];population=[]
    for block in blocks if blocks is not None else []:
        for child in block:
            if child.tag=='Compound':compounds.append(child.get('name'))
            elif child.tag in {'SimpleProtocol','AdvancedProtocol'}:
                protocols.append({'name':child.get('name'),'type':child.tag,
                                  'application_type':child.get('applicationType'),
                                  'dosing_interval':child.get('dosingInterval')})
            elif child.tag=='Formulation':
                formulations.append({'name':child.get('name'),'type':child.get('formulationType')})
            elif child.tag in {'Individual','RandomPopulation','ImportPopulation'}:
                origin=child.find('.//OriginData')
                population.append({'name':child.get('name'),'type':child.tag,
                                   'origin_raw_base_units':dict(origin.attrib) if origin is not None else None})
    compound_inventory_basis='Explicit saved UsedBuildingBlocks/Compound names'
    if not compounds:
        # Imported models can contain the complete mechanistic model without
        # PK-Sim Compound building blocks. Preserve explicit molecule identity
        # from typed root containers; enzyme/metabolite names stay distinguishable.
        compounds=[node.get('name') for node in document.findall('Model/Root/Children/*')
                   if node.get('containerType')=='Molecule' and node.get('name') and
                   node.get('name')+'|Molecular weight' in parameters]
        compound_inventory_basis='Explicit imported Model root containers of type Molecule with their own Molecular weight parameter; includes enzymes/metabolites, administration targets identify administered drug'
    source_protocol_bindings=[]
    all_blocks={child.get('id'):child for wrapper in blocks for child in wrapper
                if child.get('id')} if blocks is not None else {}
    for selection in document.findall('Properties/CompoundPropertiesList/CompoundProperties'):
        compound_block=all_blocks.get(selection.get('compound'))
        protocol=selection.find('ProtocolProperties')
        if protocol is not None:
            protocol_block=all_blocks.get(protocol.get('protocol'))
            source_protocol_bindings.append({'compound_id':selection.get('compound'),
                'compound_name':compound_block.get('name') if compound_block is not None else None,
                'protocol_id':protocol.get('protocol'),
                'protocol_name':protocol_block.get('name') if protocol_block is not None else None,
                'formulation_mappings':[dict(x.attrib) for x in protocol.findall('FormulationMappings/FormulationMapping')]})
    model_root=document.find('Model/Root');entity_paths={};event_groups=[];observer_paths=[]
    def entities(element,path):
        if element is not model_root and 'id' in element.attrib and 'name' in element.attrib:
            path=path+[element.get('name')];entity_paths[element.get('id')]='|'.join(path)
            if element.tag=='Observer':observer_paths.append('|'.join(path))
            if element.tag=='EventGroup' and element.get('eventGroupType'):
                event_groups.append((element,'|'.join(path)))
        for child in element:entities(child,path)
    entities(model_root,[])
    administrations=[]
    for group,prefix in event_groups:
        targets=[];administered=set()
        for assignment in group.iter('EventAssignment'):
            target=entity_paths.get(assignment.get('changedEntity'))
            object_path=assignment.get('objectPath')
            components=(target or object_path or '').split('|')
            matched=sorted(set(compounds)&set(components))
            administered.update(matched)
            targets.append({'assignment_id':assignment.get('id'),'changed_entity_id':assignment.get('changedEntity'),
                'target_entity_path':target,'object_path':object_path,'target_compounds':matched})
        administrations.append({'event_prefix':prefix,'application_type':group.get('eventGroupType'),
            'administered_compounds':sorted(administered),'assignment_targets':targets,
            'compound_association_basis':'Exact saved changedEntity target path segments intersect explicit used Compound names',
            'compound_association_complete':bool(targets) and bool(administered) and all(t['target_entity_path'] for t in targets)})
    events=[]
    for path,row in parameters.items():
        if path.startswith(('Applications|','Events|')) and path.rsplit('|',1)[-1] in {'Dose','Start time','Infusion time'}:
            events.append(row)
    properties=document.find('Properties');model_properties=document.find('Properties/ModelProperties')
    origins=[p['origin_raw_base_units'] for p in population if p['origin_raw_base_units']]
    return {'simulation_id':document.get('id'),'name':document.get('name'),
            'saved_type':document.tag,'saved_xml_version':document.get('xmlVersion'),
            'compounds':compounds,'protocols':protocols,'formulations':formulations,
            'compound_inventory_basis':compound_inventory_basis,
            'population_source':population,'species':sorted({o['species'] for o in origins if o.get('species')}),
            'model_configuration':model_properties.get('modelConfiguration') if model_properties is not None else None,
            'model_origin':properties.get('origin') if properties is not None else None,
            'event_parameters':events,'parameter_count':len(parameters),
            'parent_molecular_weight_parameters':[row for path,row in parameters.items()
                if path.count('|')==1 and path.endswith('|Molecular weight') and path.split('|')[0] in compounds],
            'observer_paths':sorted(observer_paths),
            'pregnancy_model_parameter_paths':[path for path in parameters if any(
                marker in path.casefold() for marker in ('fetus','fetal','placenta','amniotic'))],
            'administrations':administrations,'source_protocol_bindings':source_protocol_bindings,
            'population_simulation_is_reference_model_only':document.tag=='PopulationSimulation',
            'numeric_convention':'XML values are stored base values; displayUnit is presentation only, never a conversion factor'}


def source_simulation_metadata(source_path, source_sha256, *, project_root=ROOT,
                               max_simulations=256, max_seconds=120):
    root=Path(project_root).resolve(strict=True);source=inside(source_path,root)
    inventory=simulation_inventory(source,source_sha256,project_root=root)
    if len(inventory)>max_simulations:raise ValueError('SOURCE_SIMULATION_INVENTORY_BOUND_EXCEEDED')
    started=time.monotonic();rows=[]
    with sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        for item in inventory:
            if time.monotonic()-started>max_seconds:raise TimeoutError('SOURCE_METADATA_INVENTORY_TIMEOUT')
            if not 0<item['compressed_bytes']<=MAX_COMPRESSED_BYTES:raise ValueError('MODEL_BLOB_SIZE_INVALID')
            blob=db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=? AND b.ContentId=?',
                (item['simulation_id'],item['content_id'])).fetchone()[0]
            document=ET.fromstring(decompress_model_blob(blob))
            if document.get('id')!=item['simulation_id'] or document.get('name')!=item['name']:
                raise ValueError('SAVED_SIMULATION_IDENTITY_MISMATCH')
            rows.append({**item,**describe_saved_simulation(document),
                         'source_blob_sha256':hashlib.sha256(blob).hexdigest()})
    if sha(source)!=source_sha256:raise ValueError('SOURCE_PROJECT_CHANGED')
    return {'source':reference(source,root),'simulations':rows,'observations_or_results_read':False}


def compress_model_blob(data):
    """The original PK-Sim stream format, with a checked ZIP local header."""
    if len(data)>MAX_XML_BYTES:raise ValueError("MODEL_XML_TOO_LARGE")
    encoder=zlib.compressobj(level=6,wbits=-15);compressed=encoder.compress(data)+encoder.flush()
    name=b"Name"
    header=struct.pack("<IHHHHHIIIHH",0x04034b50,20,0,8,0,0,zlib.crc32(data)&0xffffffff,
                       len(compressed),len(data),len(name),0)
    blob=header+name+compressed
    if decompress_model_blob(blob)!=data:raise ValueError("MODEL_BLOB_REPACK_FAILED")
    return blob


def prepare_native_export_copy(source_path, source_sha256, simulation_id, output_dir, *, project_root=ROOT):
    """Isolate a model-only native export copy; the original stays unchanged.

    Outcome tables are emptied in the NEW COPY before any native loader sees
    it. No outcome payload is parsed. The selected full model/settings nodes
    are checked before and after removing only outcome-reference XML nodes.
    """
    root=Path(project_root).resolve(strict=True);source=inside(source_path,root)
    output=inside(output_dir,root,exists=False)
    if output.exists():raise FileExistsError("Native export copy needs a new folder")
    inventory=simulation_inventory(source,source_sha256,project_root=root)
    selected=[x for x in inventory if x['simulation_id']==simulation_id]
    if len(selected)!=1:raise ValueError("EXACT_SAVED_SIMULATION_REQUIRED")
    selected=selected[0]
    if not 0<selected['compressed_bytes']<=MAX_COMPRESSED_BYTES:raise ValueError("MODEL_BLOB_SIZE_INVALID")
    with sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1",uri=True) as db:
        blob=db.execute("SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId "
                        "JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=? AND b.ContentId=?",
                        (simulation_id,selected['content_id'])).fetchone()[0]
    document=ET.fromstring(decompress_model_blob(blob))
    if document.get('id')!=simulation_id or document.get('name')!=selected['name']:
        raise ValueError('SAVED_SIMULATION_IDENTITY_MISMATCH')
    models=document.findall('Model')
    if len(models)!=1:raise ValueError('EXACTLY_ONE_SAVED_MODEL_REQUIRED')
    settings=saved_settings(document)
    before={'Model':element_sha(models[0]),settings.tag:element_sha(settings)}
    removed=[]
    for tag in ('UsedObservedData','AucIV','AucDDI','CMaxDDI'):
        for child in document.findall(tag):document.remove(child);removed.append(tag)
    if before!={k:element_sha(document.find(k)) for k in before}:raise ValueError('SAVED_MODEL_CHANGED')
    stripped=compress_model_blob(ET.tostring(document,encoding='utf-8',xml_declaration=True))
    if sha(source)!=source_sha256:raise ValueError('SOURCE_PROJECT_CHANGED')
    output.mkdir(parents=True);copy=output/'model_only.pksim5';shutil.copyfile(source,copy)
    if sha(copy)!=source_sha256:raise ValueError('SOURCE_COPY_MISMATCH')
    statements=[]
    with sqlite3.connect(copy) as db:
        db.execute('PRAGMA foreign_keys=OFF')
        tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # Delete only fresh-copy data/references. Model CONTENTS and building
        # blocks remain intact; orphan outcome CONTENTS are never selected.
        for table in ('USED_OBSERVED_DATA','OBSERVED_DATA','DATA_REPOSITORIES','INDIVIDUAL_RESULTS_QUANTITY_VALUES',
                      'INDIVIDUAL_RESULTS','QUANTITY_VALUES','SIMULATION_RESULTS','SIMULATION_ANALYSES',
                      'SIMULATION_CHARTS','PARAMETER_IDENTIFICATIONS','SENSITIVITY_ANALYSES',
                      'INDIVIDUAL_SIMULATION_COMPARISONS','POPULATION_SIMULATION_COMPARISONS','SIMULATION_COMPARISONS'):
            if table in tables:
                cursor=db.execute('DELETE FROM '+table);statements.append({'table':table,'removed_rows':cursor.rowcount})
        columns={x[1] for x in db.execute('PRAGMA table_info(SIMULATIONS)')}
        nullable=[x for x in ('DataRepositoryId','SimulationResultsId','SimulationAnalysesId') if x in columns]
        if nullable:db.execute('UPDATE SIMULATIONS SET '+','.join(x+'=NULL' for x in nullable))
        db.execute('UPDATE CONTENTS SET Data=? WHERE Id=?',(stripped,selected['content_id']))
        if db.execute('PRAGMA quick_check').fetchone()!=('ok',):raise ValueError('MODEL_ONLY_PROJECT_INTEGRITY_FAILED')
    if sha(source)!=source_sha256:raise ValueError('SOURCE_PROJECT_CHANGED')
    result={'version':VERSION,'status':'model_only_copy_prepared','source':reference(source,root),
            'simulation':selected,'working_copy':reference(copy,root),'model_settings_sha256':before,
            'source_blob_sha256':hashlib.sha256(blob).hexdigest(),
            'model_only_blob_sha256':hashlib.sha256(stripped).hexdigest(),
            'removed_xml_outcome_references':removed,'new_copy_cleared_outcome_tables':statements,
            'model_and_settings_elements_unchanged':True,'original_project_unchanged':True,
            'observed_or_result_payloads_parsed':False,'code':reference(Path(__file__),root)}
    save(output/'model_only_copy.json',result)
    return result


def export_saved_model(source_path, source_sha256, simulation_id, output_dir, *,
                       simulation_name=None, project_root=ROOT, timeout_seconds=180,
                       cpu_threads=1, stop_file=None):
    """Native source-project -> PKML with immutable input and model audit.

    This prepares an artifact, not a clinical prediction. A caller must still
    load/inspect it, adapt only an explicit regimen and validate target outputs.
    """
    from pbpk_kb.blueprints.design_runtime import run_bounded_r
    from recommendation_execution_repairs.model_semantics import compare_saved_model_xml, compare_saved_settings_xml
    root=Path(project_root).resolve(strict=True);output=inside(output_dir,root,exists=False)
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:raise ValueError('Export timeout must be <=180s')
    if cpu_threads not in (1,2):raise ValueError('Export allows at most 2 CPU threads')
    if stop_file and inside(stop_file,root,exists=False).exists():raise InterruptedError('Export stop requested')
    if output.exists():raise FileExistsError('Native preserved model export requires new folder')
    output.mkdir(parents=True)
    record={'version':VERSION,'status':'failed','source':{'path':str(inside(source_path,root).relative_to(root)),
             'sha256':source_sha256},'simulation_id':simulation_id,'simulation_name':simulation_name,
            'ode_run':False,'fitting_performed':False,'observations_or_results_read':False,
            'scientific_accepted':False,'code':reference(Path(__file__),root),
            'worker':reference(root/CORE_EXPORT_SCRIPT,root),'errors':[]}
    try:
        prepared=prepare_native_export_copy(source_path,source_sha256,simulation_id,output/'source',project_root=root)
        selected=prepared['simulation']
        if simulation_name is not None and selected['name']!=simulation_name:
            raise ValueError('SAVED_SIMULATION_NAME_MISMATCH')
        record['simulation_name']=selected['name'];record['model_only_copy']=reference(output/'source/model_only_copy.json',root)
        source=inside(source_path,root)
        with sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            blob=db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                  'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=? AND b.ContentId=?',
                  (simulation_id,selected['content_id'])).fetchone()[0]
        original=ET.fromstring(decompress_model_blob(blob));record['source_metadata']=describe_saved_simulation(original)
        record['source_blob_sha256']=hashlib.sha256(blob).hexdigest()
        native=output/'native';native.mkdir();pkml=native/'preserved_model.pkml'
        process=run_bounded_r(script=root/CORE_EXPORT_SCRIPT,args=[str(root/prepared['working_copy']['path']),
            simulation_id,selected['name'],str(pkml)],output_dir=native,project_root=root,
            timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,max_output_bytes=256*1024**2,
            stop_file=inside(stop_file,root,exists=False) if stop_file else None)
        record['process']=reference(native/'process_manifest.json',root)
        record['working_copy_after_native']=reference(root/prepared['working_copy']['path'],root)
        if sha(source)!=source_sha256:raise ValueError('ORIGINAL_PROJECT_CHANGED_DURING_EXPORT')
        if process.get('status')!='completed' or process.get('returncode')!=0:
            raise ValueError('NATIVE_MODEL_EXPORT_FAILED:'+str(process.get('stop_reason')))
        if process.get('script_sha256')!=record['worker']['sha256'] or sha(root/CORE_EXPORT_SCRIPT)!=record['worker']['sha256']:
            raise ValueError('NATIVE_EXPORT_WORKER_CHANGED')
        native_receipt=Path(str(pkml)+'.native.json')
        core=json.loads(native_receipt.read_text())
        if (core.get('same_model_object') is not True or core.get('same_settings_object') is not True or core.get('model_rebuilt') is not False or
                core.get('solver_calls')!=0 or core.get('imported_flag_changed') is not False):
            raise ValueError('CORE_EXPORT_DID_NOT_PRESERVE_MODEL_OBJECT')
        if not pkml.is_file() or pkml.stat().st_size>MAX_XML_BYTES:raise ValueError('EXPORTED_PKML_SIZE_INVALID')
        exported=ET.parse(pkml).getroot()
        if any(x.tag in {'ObservedData','DataRepository','IndividualResults','SimulationResults'} for x in exported.iter()):
            raise ValueError('EXPORTED_PKML_CONTAINS_OUTCOME_DATA')
        model=exported.find('Simulation/Model')
        if model is None:raise ValueError('EXPORTED_PKML_MODEL_MISSING')
        audit=compare_saved_model_xml(original.find('Model'),model)
        settings_audit=compare_saved_settings_xml(saved_settings(original),
                         exported.find('Simulation/SimulationConfiguration/SimulationSettings'))
        save(output/'model_semantics.json',audit)
        save(output/'settings_semantics.json',settings_audit)
        record.update(pkml=reference(pkml,root),native_core_receipt=reference(native_receipt,root),
                      model_semantics=reference(output/'model_semantics.json',root),
                      settings_semantics=reference(output/'settings_semantics.json',root),
                      source_model_xml_sha256=element_sha(original.find('Model')),
                      source_settings_xml_sha256=element_sha(saved_settings(original)),
                      original_project_unchanged=True)
        if not audit.get('accepted'):raise ValueError('SAVED_MODEL_EXPORT_SEMANTICS_DIFFER')
        if not settings_audit.get('accepted'):raise ValueError('SAVED_SETTINGS_EXPORT_SEMANTICS_DIFFER')
        record['status']='exported_pending_native_load_and_clinical_validation'
    except (ValueError,OSError,KeyError,TypeError,sqlite3.Error,ET.ParseError) as exc:
        record['errors'].append(type(exc).__name__+':'+str(exc))
    save(output/'export_manifest.json',record)
    return record


def wrap_preserved_model(document):
    """Retain the complete saved model and settings; omit outcome references.

    Native loading must still validate legacy XML. This function does not
    claim compatibility or scientific equivalence merely from a wrapper.
    """
    if document.tag not in {"IndividualSimulation","PopulationSimulation"}:
        raise ValueError("UNSUPPORTED_SAVED_SIMULATION_TYPE")
    def exactly_one(tag):
        rows=document.findall(tag)
        if len(rows)!=1:raise ValueError("EXACTLY_ONE_SAVED_"+tag.upper()+"_REQUIRED")
        return rows[0]
    model=exactly_one("Model");settings=saved_settings(document)
    if len(model.findall("Root"))!=1 or len(model.findall("FormulaCache"))!=1:
        raise ValueError("COMPLETE_SAVED_MODEL_REQUIRED")
    if any(x.tag in {"DataRepository","ObservedData","IndividualResults","SimulationResults"} for x in model.iter()):
        raise ValueError("OUTCOME_DATA_INSIDE_MODEL_FORBIDDEN")
    transfer=ET.Element("SimulationTransfer",{"id":document.attrib["id"]+"-preserved-transfer","version":"18"})
    simulation=ET.SubElement(transfer,"Simulation",{k:document.attrib[k] for k in ("id","name","icon") if k in document.attrib})
    configuration=ET.SubElement(simulation,"SimulationConfiguration",{"createAllProcessRateParameters":"0"})
    configuration.append(deepcopy(settings))
    simulation.append(deepcopy(model))
    creation=document.find("Creation")
    if creation is not None:simulation.append(deepcopy(creation))
    ET.SubElement(simulation,"EntitySources")
    ET.SubElement(transfer,"Favorites");ET.SubElement(transfer,"OutputMappings")
    if element_sha(model)!=element_sha(simulation.find("Model")) or element_sha(settings)!=element_sha(saved_settings(configuration)):
        raise ValueError("SAVED_MODEL_OR_SETTINGS_CHANGED")
    return transfer,{"model_xml_sha256":element_sha(model),"settings_xml_sha256":element_sha(settings),
        "saved_xml_version":document.get("xmlVersion"),"saved_simulation_type":document.tag,
        "parameter_xml_count":sum(x.tag in {"Parameter","Para","DistPara","DistributedParameter"} for x in model.iter()),
        "formula_xml_count":sum(x.tag.endswith("Formula") for x in model.iter()),
        "model_and_settings_elements_unchanged":True,"observed_or_result_tables_read":False,
        "population_covariates_applied":False,
        "interpretation":"Original saved model/reference subject only; population sampling is not reconstructed"}


def extract_saved_model(source_path, source_sha256, simulation_id, output_dir, *, project_root=ROOT):
    root=Path(project_root).resolve(strict=True);source=inside(source_path,root)
    output=inside(output_dir,root,exists=False)
    if output.exists():raise FileExistsError("Extraction must use a new output folder")
    inventory=simulation_inventory(source,source_sha256,project_root=root)
    matches=[r for r in inventory if r["simulation_id"]==simulation_id]
    if len(matches)!=1:raise ValueError("EXACT_SAVED_SIMULATION_REQUIRED")
    selected=matches[0]
    if not 0<selected["compressed_bytes"]<=MAX_COMPRESSED_BYTES:raise ValueError("MODEL_BLOB_SIZE_INVALID")
    with sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1",uri=True) as db:
        rows=db.execute("SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId "
                        "JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=? AND b.ContentId=?",
                        (simulation_id,selected["content_id"])).fetchall()
    if len(rows)!=1:raise ValueError("EXACT_SAVED_MODEL_BLOB_REQUIRED")
    data=decompress_model_blob(rows[0][0]);document=ET.fromstring(data)
    if document.get("id")!=simulation_id or document.get("name")!=selected["name"]:
        raise ValueError("SAVED_SIMULATION_IDENTITY_MISMATCH")
    transfer,audit=wrap_preserved_model(document)
    if sha(source)!=source_sha256:raise ValueError("SOURCE_PROJECT_CHANGED")
    output.mkdir(parents=True)
    path=output/"preserved_model.pkml"
    ET.ElementTree(transfer).write(path,encoding="utf-8",xml_declaration=True)
    result={"version":VERSION,"status":"extracted_pending_native_validation","source":reference(source,root),
            "simulation":selected,"blob_sha256":hashlib.sha256(rows[0][0]).hexdigest(),
            "saved_xml_sha256":hashlib.sha256(data).hexdigest(),"pkml":reference(path,root),
            "audit":audit,"code":reference(Path(__file__),root),"ode_run":False,"scientific_accepted":False}
    save(output/"extraction.json",result)
    return result
