"""Append complete target applications while retaining every original DDI node.

This is structural preparation only. Existing target dose/time values and all
other-drug events remain untouched; a separately registered native request must
apply the clinical regimen and prove any coadministration isolation.
"""
from copy import deepcopy
import hashlib
import re
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
import json

from . import repeated_application as base
from . import source_pkml
from .native_pkml import inside, sha, save

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'preserved-mixed-target-application-expansion-1'
IMPLEMENTATION_FILES = tuple(dict.fromkeys([
    'scripts/recommendation_execution_repairs/mixed_application_expansion.py', *base.IMPLEMENTATION_FILES]))


def _require(ok, message):
    if not ok: raise ValueError('MIXED_EXPANSION_' + message)


def _qualify(document, target):
    model = base._models(document)
    base.semantics.model_semantics(model)
    paths, ids, parents = base._inventory(model)
    strings, formulas = base._formula_inventory(model)
    rootname = model.find('Root').get('name')
    applications = [(path, node) for path, node in paths.items()
                    if node.tag == 'EventGroup' and any(
                        child.get('name') == 'ProtocolSchemaItem' for child in node.findall('Children/Container'))]
    selected = []
    for path, node in applications:
        molecules = list(node.iter('MoleculeAmount'))
        if any(n.get('name') == target for n in molecules):
            _require(len(molecules) == 1, 'TARGET_APPLICATION_HAS_MULTIPLE_LOCAL_DRUG_STATES')
            dose = paths.get(path + '|ProtocolSchemaItem|Dose')
            start = paths.get(path + '|ProtocolSchemaItem|Start time')
            _require(dose is not None and start is not None and dose.get('dim') == 'Mass'
                     and start.get('dim') == 'Time', 'EXPLICIT_MASS_AND_TIME_REQUIRED')
            selected.append((base._scalar(start, formulas), path, node, dose, start))
    _require(0 < len(selected) < 30, 'SOURCE_TARGET_EVENT_COUNT_MUST_BE_ONE_TO_29')
    selected.sort(key=lambda row: (row[0], row[1]))
    _require(len({row[0] for row in selected}) == len(selected), 'AMBIGUOUS_SIMULTANEOUS_TARGET_APPLICATIONS')
    _, fullprefix, branch, dose, start = selected[-1]
    _require(branch.get('containerType') == 'Application' and branch.get('eventGroupType') == 'Oral'
             and re.fullmatch(r'Application_[1-9][0-9]*', branch.get('name', '')), 'ORIGINAL_ORAL_APPLICATION_REQUIRED')
    nodes = set(branch.iter()); branchids = {n.get('id') for n in nodes if n.get('id') and n.get('name')}
    _require(len(branchids) == sum(bool(n.get('id') and n.get('name')) for n in nodes), 'DUPLICATE_LOCAL_IDS')
    molecule = list(branch.iter('MoleculeAmount'))[0]
    transports = list(branch.iter('Transport')); events = list(branch.iter('Event'))
    _require(len(transports) == 12 and len(events) == 2 and
             {n.get('name') for n in events} == {'Application_StartEvent', 'Application_StopEvent'} and
             all(n.get('oneTime') == '1' for n in events) and
             any(n.get('name') == 'TabletIsActive' for n in nodes), 'COMPLETE_TABLET_RELEASE_TOPOLOGY_REQUIRED')
    condition = formulas.get(next(n for n in events if n.get('name') == 'Application_StartEvent').get('formula'))
    condition_paths = list(condition.iter('Path')) if condition is not None else []
    _require(condition is not None and condition.get('formula', '').replace(' ', '') == 'Time=StartTime'
             and len(condition_paths) == 1 and len(list(condition.iter('TimePath'))) == 1
             and base._resolve_paths(condition, strings) == ['..|ProtocolSchemaItem|Start time']
             and strings.get(condition_paths[0].get('as'), condition_paths[0].get('as')) == 'StartTime',
             'EXACT_ONCE_ONLY_STARTTIME_TRIGGER_REQUIRED')
    _require(all(t.get('sourceAmount') == molecule.get('id') and t.get('targetAmount') in ids
                 and t.get('targetAmount') not in branchids for t in transports), 'ORIGINAL_EXTERNAL_RELEASE_TARGETS_REQUIRED')
    for node in nodes:
        for field in base.FORMULA_FIELDS:
            if not node.get(field): continue
            formula = formulas.get(node.get(field))
            _require(formula is not None and not any(branch.get('name') in (p or '').split('|')
                      for p in base._resolve_paths(formula, strings)), 'ABSOLUTE_LOCAL_FORMULA_PATH_UNSUPPORTED')
        if node.tag == 'ObjectPath':
            value = strings.get(node.get('path'), node.get('path', ''))
            _require(branch.get('name') not in value.split('|'), 'ABSOLUTE_LOCAL_ASSIGNMENT_PATH_UNSUPPORTED')
    aggregate = paths.get(rootname + '|' + target + '|Total drug mass')
    _require(aggregate is not None and aggregate.get('dim') == 'Amount' and aggregate.get('formula'),
             'UNIQUE_TARGET_TOTAL_DRUG_MASS_REQUIRED')
    formula = formulas.get(aggregate.get('formula'))
    _require(formula is not None and formula.tag == 'Formula', 'EXPLICIT_TARGET_SUM_FORMULA_REQUIRED')
    group = formula.findall('Paths'); count = len(selected)
    expected_paths = {row[1] + '|ProtocolSchemaItem|DrugMass' for row in selected}
    _require(len(group) == 1 and len(group[0]) == count and all(n.tag == 'Path' for n in group[0])
             and len(set(base._resolve_paths(formula, strings))) == count
             and set(base._resolve_paths(formula, strings)) == expected_paths
             and [strings.get(n.get('as'), n.get('as')) for n in group[0]] == [f'P_{i}' for i in range(1,count+1)]
             and formula.get('formula', '').replace(' ', '') == '+'.join(f'P_{i}' for i in range(1,count+1)),
             'TARGET_TOTAL_MASS_MUST_SUM_EACH_ORIGINAL_TARGET_APPLICATION_ONCE')
    paths_by_node={node:path for path,node in paths.items()}
    def points_into_template(raw,node):
        # Other drugs can also have an Application_5. A basename match is not
        # an entity match: resolve the whole original path before blocking it.
        if branch.get('name') not in (raw or '').split('|'):return False
        if raw.startswith(rootname+'|'):
            tokens=[]
        elif raw.startswith(('.', '..')):
            _require(node in paths_by_node,'UNRESOLVED_EXTERNAL_FORMULA_CONTEXT')
            tokens=paths_by_node[node].split('|')
        else:
            candidates=[value for value in (rootname+'|'+raw,paths_by_node.get(node,'')+'|'+raw,
                        paths_by_node.get(node,'').rsplit('|',1)[0]+'|'+raw) if value in paths]
            _require(len(set(candidates))==1,'AMBIGUOUS_EXTERNAL_APPLICATION_PATH')
            return candidates[0]==fullprefix or candidates[0].startswith(fullprefix+'|')
        for token in raw.split('|'):
            if token=='..':
                _require(len(tokens)>1,'EXTERNAL_FORMULA_ESCAPES_ROOT');tokens.pop()
            elif token!='.':tokens.append(token)
        resolved='|'.join(tokens)
        _require(resolved in paths,'UNRESOLVED_EXTERNAL_APPLICATION_PATH')
        return resolved==fullprefix or resolved.startswith(fullprefix+'|')
    for node in model.find('Root').iter():
        if node in nodes: continue
        _require(not any(node.get(k) in branchids for k in base.LINKS), 'EXTERNAL_ENTITY_POINTS_INTO_TEMPLATE')
        for field in base.FORMULA_FIELDS:
            if not node.get(field): continue
            current = formulas.get(node.get(field))
            _require(current is not None, 'MISSING_EXTERNAL_FORMULA')
            _require(node is aggregate or current is not formula, 'TARGET_MASS_FORMULA_SHARED_WITH_OTHER_QUANTITY')
            _require(node is aggregate or not any(points_into_template(p,node)
                     for p in base._resolve_paths(current, strings)), 'UNREGISTERED_EXTERNAL_TEMPLATE_DEPENDENCY')
    sibling_names = [n.get('name', '') for n in parents[branch] if n.tag == 'EventGroup']
    indices = [int(n.rsplit('_',1)[1]) for n in sibling_names if re.fullmatch(r'Application_[1-9][0-9]*', n)]
    return dict(model=model,paths=paths,ids=ids,parents=parents,strings=strings,formulas=formulas,
                rootname=rootname,branch=branch,branchids=branchids,dose=dose,start=start,
                aggregate=aggregate,formula=formula,selected=selected,applications=applications,
                fullprefix=fullprefix,prefix=fullprefix[len(rootname)+1:],next_index=max(indices)+1)


def _spec(contract):
    _require(contract.get('route') == 'Oral', 'ORAL_TARGET_REQUIRED')
    times = contract.get('administration_times_min');dose = contract.get('dose_mg')
    base._validate_spec(times,dose)
    _require(len(times) <= 30, 'MAXIMUM_30_TARGET_APPLICATIONS')
    return times,dose


def expand_document(document, *, target, contract):
    """Append only missing target applications; preserve all originals exactly."""
    times,dose = _spec(contract);src = _qualify(document,target)
    count = len(src['selected']);_require(count < len(times), 'TARGET_DOES_NOT_REQUIRE_EXPANSION')
    result = deepcopy(document);dst = _qualify(result,target)
    seed = hashlib.sha256(ET.tostring(document,encoding='utf-8')).hexdigest()
    records=[];parent=dst['parents'][dst['branch']];position=list(parent).index(dst['branch'])+1
    for offset,time in enumerate(times[count:]):
        index=dst['next_index']+offset;name=f'Application_{index}'
        mapping={old:str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+':'+seed+':'+str(index)+':'+old)) for old in src['branchids']}
        clone=deepcopy(src['branch']);base._rename_clone(clone,mapping,src['branch'].get('name'),name)
        local={n.get('id'):n for n in clone.iter() if n.get('id') and n.get('name')}
        base._set_scalar(local[mapping[src['dose'].get('id')]],dose/1e6)
        base._set_scalar(local[mapping[src['start'].get('id')]],float(time))
        parent.insert(position+offset,clone)
        records.append({'name':name,'prefix':src['prefix'].rsplit('|',1)[0]+'|'+name,
                        'id_mapping':mapping,'start_min':float(time),'dose_kg':dose/1e6})
    added=[];formula=dst['formula'];paths=formula.find('Paths')
    formula.set('formula',' + '.join(f'P_{i}' for i in range(1,len(times)+1)))
    for i,record in enumerate(records,count+1):
        node=deepcopy(src['formula'].find('Paths')[-1])
        node.set('path',base._intern_string(dst['model'],src['rootname']+'|'+record['prefix']+'|ProtocolSchemaItem|DrugMass',added))
        node.set('as',base._intern_string(dst['model'],f'P_{i}',added));paths.append(node)
    plan={'version':VERSION,'target':target,'contract':deepcopy(contract),'template_prefix':src['prefix'],
          'source_target_count':count,'source_total_application_count':len(src['applications']),
          'target_count':len(times),'added_applications':records,'added_string_map_entries':added,
          'target_mass_parameter_id':src['aggregate'].get('id'),'target_mass_formula_id':src['formula'].get('id'),
          'source_xml_sha256':seed,'original_applications_modified':False,
          'other_drug_mechanisms_modified':False,'sealed_observations_accessed':False}
    result_audit=audit_expansion(document,result,plan)
    _require(result_audit['accepted'],'INVERSE_XML_AUDIT_FAILED:'+str(result_audit['reasons']))
    return result,plan


def audit_expansion(original,expanded,plan):
    """Prove added branches independently, then remove edits and compare all XML."""
    failures=[];semantic=None
    try:
        _require(plan.get('version')==VERSION,'UNKNOWN_PLAN')
        times,dose=_spec(plan['contract']);src=_qualify(original,plan['target'])
        count=len(src['selected']);records=plan['added_applications']
        seed=hashlib.sha256(ET.tostring(original,encoding='utf-8')).hexdigest()
        _require(plan['source_xml_sha256']==seed and plan['source_target_count']==count
                 and plan['source_total_application_count']==len(src['applications'])
                 and plan['target_count']==len(times) and len(records)==len(times)-count
                 and len(records)>0 and plan['template_prefix']==src['prefix'], 'PLAN_SOURCE_OR_COUNT_DIFFERS')
        reverse=deepcopy(expanded);model=base._models(reverse)
        semantic=base.semantics.model_semantics(model)
        paths,ids,parents=base._inventory(model);strings,formulas=base._formula_inventory(model)
        new_paths=[];all_added_ids=set()
        template_parent_path=src['fullprefix'].rsplit('|',1)[0]
        for offset,(record,time) in enumerate(zip(records,times[count:])):
            index=src['next_index']+offset;name=f'Application_{index}'
            prefix=src['prefix'].rsplit('|',1)[0]+'|'+name
            expected_mapping={old:str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+':'+seed+':'+str(index)+':'+old)) for old in src['branchids']}
            _require(record['name']==name and record['prefix']==prefix and record['id_mapping']==expected_mapping
                     and record['start_min']==float(time) and record['dose_kg']==dose/1e6,'CLONE_RECIPE_DIFFERS')
            _require(not (set(expected_mapping.values()) & (set(src['ids'])|all_added_ids)), 'CLONE_ID_COLLISION')
            all_added_ids.update(expected_mapping.values())
            branch=paths.get(src['rootname']+'|'+prefix)
            _require(branch is not None and parents[branch] is paths[template_parent_path].find('Children'), 'CLONE_LOCATION_DIFFERS')
            expected=deepcopy(src['branch']);base._rename_clone(expected,expected_mapping,src['branch'].get('name'),name)
            local={n.get('id'):n for n in expected.iter() if n.get('id') and n.get('name')}
            base._set_scalar(local[expected_mapping[src['dose'].get('id')]],dose/1e6)
            base._set_scalar(local[expected_mapping[src['start'].get('id')]],float(time))
            _require(base._canonical(branch)==base._canonical(expected),'CLONED_INTERNAL_OR_EXTERNAL_ENTITY_CHANGED')
            new_paths.append(src['rootname']+'|'+prefix+'|ProtocolSchemaItem|DrugMass');parents[branch].remove(branch)
        aggregate=paths.get(src['rootname']+'|'+plan['target']+'|Total drug mass')
        _require(aggregate is not None and base._canonical(aggregate)==base._canonical(src['aggregate'])
                 and plan['target_mass_parameter_id']==src['aggregate'].get('id')
                 and plan['target_mass_formula_id']==src['formula'].get('id'),'TARGET_MASS_PARAMETER_CHANGED')
        formula=formulas.get(plan['target_mass_formula_id']);expected_formula=deepcopy(src['formula'])
        expected_formula.set('formula',' + '.join(f'P_{i}' for i in range(1,len(times)+1)))
        for index,path in enumerate(new_paths,count+1):
            node=deepcopy(src['formula'].find('Paths')[-1]);node.set('path',path);node.set('as',f'P_{index}')
            expected_formula.find('Paths').append(node)
        for f,mapping in ((expected_formula,src['strings']),):
            for node in f.iter('Path'):
                for key in ('path','as','dim'):
                    if node.get(key) in mapping:node.set(key,mapping[node.get(key)])
        normalized=deepcopy(formula)
        for node in normalized.iter('Path'):
            for key in ('path','as','dim'):
                if node.get(key) in strings:node.set(key,strings[node.get(key)])
        _require(base._canonical(normalized)==base._canonical(expected_formula),'TARGET_MASS_SUM_OR_ALIASES_CHANGED')
        old_map=src['model'].find('FormulaCache/StringMap');new_map=model.find('FormulaCache/StringMap')
        old_rows=list(old_map);new_rows=list(new_map)
        _require([base._canonical(x) for x in new_rows[:len(old_rows)]]==[base._canonical(x) for x in old_rows], 'ORIGINAL_STRING_MAP_CHANGED')
        additions=new_rows[len(old_rows):]
        _require([dict(x.attrib) for x in additions]==plan['added_string_map_entries'],'UNREGISTERED_STRING_ADDITIONS')
        used={node.get(k) for node in formula.iter('Path') for k in ('path','as')}
        allowed=set(new_paths)|{f'P_{i}' for i in range(count+1,len(times)+1)}
        for node in additions:
            _require(node.tag=='Map' and set(node.attrib)=={'id','s'} and not len(node)
                     and node.get('s') in allowed and node.get('id') in used,'UNUSED_OR_NONDOSE_STRING_ADDITION')
            new_map.remove(node)
        group=model.find('FormulaCache/Formulas');position=list(group).index(formula)
        group.remove(formula);group.insert(position,deepcopy(src['formula']))
        _require(base._canonical(reverse)==base._canonical(original),'ORIGINAL_MODEL_OR_SETTINGS_CHANGED')
    except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
        failures.append(type(exc).__name__+':'+str(exc))
    return {'version':VERSION,'accepted':not failures,'reasons':failures,
            'all_original_applications_and_other_drugs_unchanged':not failures,
            'full_inverse_xml_and_settings_verified':not failures,
            'expanded_model_semantics_sha256':semantic['sha256'] if semantic else None,
            'ode_executed':False,'scientific_applicability_claimed':False}


def prepare_source_expansion(source_path,source_sha,simulation_id,blob_sha,metadata,target,contract,*,project_root=ROOT):
    root=Path(project_root).resolve(strict=True);path=inside(source_path,root)
    reference={'path':str(path.relative_to(root)),'sha256':source_sha,
               'simulation_id':simulation_id,'source_blob_sha256':blob_sha}
    saved=base._original_document(reference,root);actual=source_pkml.describe_saved_simulation(saved)
    _require(actual==metadata,'ORIGINAL_METADATA_DIFFERS')
    document,_=source_pkml.wrap_preserved_model(saved);source=_qualify(document,target)
    times,_=_spec(contract);_require(len(source['selected'])<len(times),'TARGET_DOES_NOT_REQUIRE_EXPANSION')
    target_rows=[r for r in actual['administrations'] if r.get('administered_compounds')==[target]
                 and r.get('compound_association_complete') is True and r.get('application_type')=='Oral']
    _require({r['event_prefix'] for r in target_rows}=={r[1][len(source['rootname'])+1:] for r in source['selected']},
             'EXPLICIT_TARGET_APPLICATION_ASSOCIATIONS_REQUIRED')
    return {'version':VERSION,'verified':True,'source_reference':reference,'target':target,
            'contract':deepcopy(contract),'source_metadata_sha256':base._digest(metadata),
            'source_target_event_count':len(source['selected']),'source_total_event_count':len(source['applications']),
            'executable_target_event_count':len(times),
            'executable_total_event_count':len(source['applications'])+len(times)-len(source['selected']),
            'template_prefix':source['prefix'],'template_sha256':source_pkml.element_sha(source['branch']),
            'target_sum_formula_sha256':source_pkml.element_sha(source['formula']),
            'source_model_semantics_sha256':base.semantics.model_semantics(source['model'])['sha256'],
            'original_applications_preserved':True,'clinical_patches_and_isolation_still_required':True,
            'fitting_performed':False,'sealed_observations_accessed':False}


def expand_pkml(source_pkml,policy,output_dir,*,project_root=ROOT):
    root=Path(project_root).resolve(strict=True);source=inside(source_pkml,root);output=inside(output_dir,root,exists=False)
    if output.exists():raise FileExistsError('Preserve every expansion attempt')
    ref=policy['source_reference'];saved=base._original_document(ref,root)
    expected=prepare_source_expansion(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
             source_pkml_module_metadata(saved),policy['target'],policy['contract'],project_root=root)
    _require(expected==policy,'FROZEN_POLICY_DIFFERS')
    document=base._read_xml(source);source_audit=base._verify_original_source(document,ref,root)
    implementation={p:sha(inside(p,ROOT)) for p in IMPLEMENTATION_FILES};source_hash=sha(source)
    expanded,plan=expand_document(document,target=policy['target'],contract=policy['contract'])
    output.mkdir(parents=True);pkml=output/'mixed_application.pkml'
    ET.ElementTree(expanded).write(pkml,encoding='utf-8',xml_declaration=True)
    proof=audit_expansion(document,base._read_xml(pkml),plan);_require(proof['accepted'],'SERIALIZED_PROOF_FAILED')
    for name,value in [('policy.json',policy),('expansion_plan.json',plan),('semantic_audit.json',proof),('source_audit.json',source_audit)]:save(output/name,value)
    _require(source_hash==sha(source) and implementation=={p:sha(inside(p,ROOT)) for p in IMPLEMENTATION_FILES},'SOURCE_OR_CODE_CHANGED')
    receipt={'version':VERSION,'status':'expanded_pending_native_readback_and_ode','source_reference':ref,
             'source_pkml':{'path':str(source.relative_to(root)),'sha256':source_hash},
             'pkml':{'path':str(pkml.relative_to(root)),'sha256':sha(pkml)},'implementation_sha256':implementation,
             'files':{str((output/name).relative_to(root)):sha(output/name) for name in
                      ['policy.json','expansion_plan.json','semantic_audit.json','source_audit.json']},
             'ode_executed':False,'fitting_performed':False,'sealed_observations_accessed':False,
             'original_applications_modified':False,'scientific_applicability_claimed':False}
    save(output/'receipt.json',receipt);return receipt


def source_pkml_module_metadata(saved):
    # Avoid confusion with expand_pkml's explicitly requested source_pkml argument.
    return source_pkml.describe_saved_simulation(saved)


def verify_expansion_receipt(receipt_path,*,expected_policy=None,project_root=ROOT):
    root=Path(project_root).resolve(strict=True);path=inside(receipt_path,root);receipt=json.loads(path.read_text())
    _require(receipt.get('version')==VERSION and receipt.get('status')=='expanded_pending_native_readback_and_ode','RECEIPT_VERSION_OR_STATUS_DIFFERS')
    _require(receipt['implementation_sha256']=={p:sha(inside(p,ROOT)) for p in IMPLEMENTATION_FILES},'IMPLEMENTATION_CHANGED')
    for ref in (receipt['source_pkml'],receipt['pkml']):_require(sha(inside(ref['path'],root))==ref['sha256'],'PKML_HASH_CHANGED')
    for name,digest in receipt['files'].items():_require(sha(inside(name,root))==digest,'EVIDENCE_HASH_CHANGED')
    for name in ('policy.json','expansion_plan.json','semantic_audit.json','source_audit.json'):
        _require(str((path.parent/name).relative_to(root)) in receipt['files'],'MISSING_REQUIRED_EVIDENCE')
    policy=json.loads((path.parent/'policy.json').read_text());ref=receipt['source_reference']
    if expected_policy is not None:_require(policy==expected_policy,'EXPECTED_POLICY_DIFFERS')
    saved=base._original_document(ref,root)
    fresh=prepare_source_expansion(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
        source_pkml.describe_saved_simulation(saved),policy['target'],policy['contract'],project_root=root)
    _require(policy==fresh,'SOURCE_POLICY_CHANGED')
    original=base._read_xml(inside(receipt['source_pkml']['path'],root));expanded=base._read_xml(inside(receipt['pkml']['path'],root))
    source_audit=base._verify_original_source(original,ref,root)
    _require(source_audit==json.loads((path.parent/'source_audit.json').read_text()),'SOURCE_PROOF_CHANGED')
    plan=json.loads((path.parent/'expansion_plan.json').read_text())
    _require(plan['target']==policy['target'] and plan['contract']==policy['contract'],'PLAN_POLICY_DIFFERS')
    proof=audit_expansion(original,expanded,plan)
    _require(proof['accepted'] and proof==json.loads((path.parent/'semantic_audit.json').read_text()),'FULL_XML_PROOF_CHANGED')
    return {'accepted':True,'receipt':{'path':str(path.relative_to(root)),'sha256':sha(path)},
            'source_model_and_settings_verified':True,'full_expansion_xml_verified':True,
            'source_target_event_count':policy['source_target_event_count'],
            'executable_target_event_count':policy['executable_target_event_count'],
            'executable_total_event_count':policy['executable_total_event_count'],'scientific_applicability_claimed':False}


def executable_metadata(receipt_path,original_source_metadata,*,expected_policy,project_root=ROOT):
    root=Path(project_root).resolve(strict=True)
    verification=verify_expansion_receipt(receipt_path,expected_policy=expected_policy,project_root=root)
    receipt=json.loads(inside(receipt_path,root).read_text());saved=base._original_document(receipt['source_reference'],root)
    _require(source_pkml.describe_saved_simulation(saved)==original_source_metadata,'ORIGINAL_METADATA_CHANGED')
    expanded=base._read_xml(inside(receipt['pkml']['path'],root));saved.remove(saved.find('Model'))
    saved.append(deepcopy(base._models(expanded)));actual=source_pkml.describe_saved_simulation(saved)
    _require(len(actual['administrations'])==expected_policy['executable_total_event_count'],'EXECUTABLE_APPLICATION_COUNT_DIFFERS')
    _require(sum(row.get('administered_compounds')==[expected_policy['target']] for row in actual['administrations'])
             ==expected_policy['executable_target_event_count'],'EXECUTABLE_TARGET_COUNT_DIFFERS')
    actual.update(metadata_scope='actual_expanded_executable_model_only',
                  original_source_metadata_sha256=base._digest(original_source_metadata),
                  expansion_receipt=verification['receipt'],expanded_pkml=receipt['pkml'])
    return actual,verification
