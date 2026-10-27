"""Recover registered legacy snapshots with complete model equivalence/readback.

The original source Version is really loaded. A separate native-Version-80
document maps inputs with registered official schema conversions, preserving
raw records and arithmetic provenance. Both build complete models; nothing is
fitted or accepted from JSON equality alone.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import recommendation_candidate_migration as migration
from pbpk_kb.blueprints.design_runtime import run_bounded_r
from .snapshot_schema import map_legacy_expression_schema
from .application_path_schema import normalize_application_paths
from .source_pkml import simulation_inventory
from .model_semantics import compare_saved_model_xml,compare_saved_settings_xml

VERSION='complete-legacy-json-model-readback-v4'
R_SCRIPT='src/pbpk_kb/model_runtime/R/recommendation_snapshot_readback.R'
SOURCE_VERSIONS={'54f5bcdcb19a7beaa67256a09c6846ed0498780cb01c12bbfcc16a02170dfaad':78,
                 '77d4db0edcb3811789c0289534b585ee286d3071c21f8a71cc92dd71b0816467':79,
                 '4103b8280c85b0e7799454b4eab22661a92f750f56838c8387793edf55994303':79}
FROZEN_CANDIDATE_AUDIT={
    'path':'experiments/model_recommendation/execution_validation_20261001/inputs/candidate_audit.jsonl',
    'sha256':'b7926e541c09ef722f9bb549ef03d6dcf9125ccdc8cee9924425226d9ccc86a9'}
IMPLEMENTATION_FILES=[str(Path(__file__).resolve().relative_to(ROOT)),R_SCRIPT,
    'scripts/recommendation_execution_repairs/application_path_schema.py',
    'scripts/recommendation_candidate_migration.py',
    'scripts/recommendation_execution_repairs/snapshot_repair.py',
    'scripts/recommendation_execution_repairs/snapshot_schema.py',
    'scripts/recommendation_execution_repairs/model_semantics.py',
    'scripts/recommendation_execution_repairs/source_pkml.py',
    'src/pbpk_kb/blueprints/design_runtime.py',
    'src/pbpk_kb/model_runtime/R/catalog_snapshot_migration.R']
INACTIVE_MAPPER_EVIDENCE=[
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Mappers/ParameterMapper.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Mappers/SimulationMapper.cs']
SERIALIZATION_EVIDENCE=[
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Mappers/ParameterMapper.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Parameter.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Services/IJsonSerializer.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Json/JsonSerializer.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Json/PKSimJsonSerializerSettings.cs',
    'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Infrastructure/Serialization/Json/NullabeDoubleJsonConverter.cs']
SERIALIZATION_RUNTIME_FILES=[
    'runtime/ospsuite-env/lib/R/library/ospsuite/lib/'+name
    for name in ('PKSim.Core.dll','PKSim.Infrastructure.dll','OSPSuite.Core.dll',
                 'OSPSuite.Utility.dll','Newtonsoft.Json.dll')]


def unique(rows,key):
    result={row[key]:row for row in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate explicit source identity: '+key)
    return result


def registered_source(source_provenance,original,root):
    """Authorize attempts from the frozen exact-parent catalog, not a filename.

    Registration grants only a repair attempt. Complete strict model/settings
    equality and native readback remain mandatory for every simulation.
    """
    source=migration.inside(source_provenance['source_path'],root)
    digest=source_provenance['source_sha256']
    if migration.sha(source)!=digest or original.get('Version') not in {76,77,78,79}:
        raise ValueError('Unregistered exact original JSON source/version')
    audit=migration.inside(FROZEN_CANDIDATE_AUDIT['path'],root)
    if migration.sha(audit)!=FROZEN_CANDIDATE_AUDIT['sha256']:
        raise ValueError('FROZEN_CANDIDATE_AUDIT_CHANGED')
    matched=[]
    for line in audit.open():
        row=json.loads(line);ref=row.get('catalog_provenance',{})
        if not row.get('identity_match') or ref.get('source_sha256')!=digest:continue
        if migration.inside(ref['source_path'],root)!=source:continue
        matched.append({'candidate_id':row['candidate_id'],'scenario_id':row['scenario_id']})
    if not matched:raise ValueError('SOURCE_NOT_IN_FROZEN_EXACT_PARENT_CATALOG')
    if digest in SOURCE_VERSIONS and SOURCE_VERSIONS[digest]!=original['Version']:
        raise ValueError('Registered original version differs')
    return {'catalog':deepcopy(FROZEN_CANDIDATE_AUDIT),'matches':matched,
            'source_version_read_from_exact_hashed_original':original['Version']}


def assert_no_explicit_simulation_seed(snapshot):
    """A declared simulation seed can never be overwritten by registration.

    Individual/population generation seeds are separate source inputs and stay
    byte-for-byte unchanged. Reject ambiguous seed declarations under a
    simulation/settings tree rather than assuming they are irrelevant.
    """
    def walk(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if ''.join(c for c in key.lower() if c.isalnum()) in {'seed','randomseed'}:
                    raise ValueError('EXPLICIT_SIMULATION_SEED_MUST_BE_PRESERVED:'+path+'/'+key)
                walk(item,path+'/'+key)
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,path+'/'+str(i))
    for sim in snapshot.get('Simulations',[]):walk(sim,'Simulations/'+sim['Name'])
    for key in ('Settings','SimulationSettings'):
        if key in snapshot:walk(snapshot[key],key)


def source_partition(snapshot,names):
    """Select simulations and their exact named population dependencies.

    Building an unused source population is expensive and cannot change a
    selected simulation. All other global source definitions remain present.
    Every simulation is covered by a registered batch, independent of scores.
    """
    part=migration.clean_snapshot(snapshot)
    sims=unique(part['Simulations'],'Name')
    part['Simulations']=[deepcopy(sims[n]) for n in names]
    required={s['Population'] for s in part['Simulations'] if s.get('Population')}
    populations=unique(part.get('Populations',[]),'Name')
    if not required<=populations.keys():raise ValueError('SOURCE_POPULATION_DEPENDENCY_MISSING')
    if 'Populations' in part:part['Populations']=[p for p in part['Populations'] if p['Name'] in required]
    return part


def parameter_expectations(snapshot,simulation):
    """Explicit active Parameters records; simulation overrides have precedence.

    Selected expression profiles, individual overrides and each used compound's
    Parameters are checked. Table formulas remain in whole-model comparison;
    they cannot be falsely described as a constant value.
    """
    expected={};tables=[]
    def add(record,path,scope):
        path=path.replace('Applications|','Events|',1) if path.startswith('Applications|') else path
        if 'TableFormula' in record:
            expected.pop(path,None)
            tables.append({'path':path,'source_scope':scope,'source_record':deepcopy(record)});return
        if 'Value' not in record:return
        value=record['Value']
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite explicit source value required')
        tables[:]=[item for item in tables if item['path']!=path]
        expected[path]={'path':path,'value':value,'unit':record.get('Unit'),
                        'source_scope':scope,'source_record':deepcopy(record)}
    profiles={'|'.join(p[k] for k in ('Molecule','Species','Category')):p for p in snapshot.get('ExpressionProfiles',[])}
    individual=None
    if simulation.get('Individual'):
        individual=unique(snapshot.get('Individuals',[]),'Name')[simulation['Individual']]
    elif simulation.get('Population'):
        individual=unique(snapshot.get('Populations',[]),'Name')[simulation['Population']].get('Settings',{}).get('Individual')
    if individual:
        for name in individual.get('ExpressionProfiles',[]):
            for parameter in profiles[name].get('Parameters',[]):add(parameter,parameter['Path'],'expression_profile:'+name)
        for parameter in individual.get('Parameters',[]):add(parameter,parameter['Path'],'individual:'+individual.get('Name','population_reference'))
    compounds=unique(snapshot.get('Compounds',[]),'Name')
    for selected in simulation.get('Compounds',[]):
        name=selected['Name']
        for parameter in compounds[name].get('Parameters',[]):add(parameter,name+'|'+parameter['Name'],'compound:'+name)
    for parameter in simulation.get('Parameters',[]):add(parameter,parameter['Path'],'simulation:'+simulation['Name'])
    return {'parameters':list(expected.values()),'table_formulas':tables,
            'precedence':'selected expression profile, individual, compound, explicit simulation override',
            'all_other_fields_protected_by_whole_model_equivalence':True}


def _record_digest(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _native_record_equal(left,right):
    """Exact JSON-number equality; jsonlite may spell 1.0 as 1, never a tolerance."""
    if type(left) in (int,float) and type(right) in (int,float):
        return math.isfinite(left) and math.isfinite(right) and left==right
    if type(left) is not type(right):return False
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_native_record_equal(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(_native_record_equal(a,b) for a,b in zip(left,right))
    return left==right


def register_preserved_simulation_overrides(original,restored):
    """Only unchanged explicit simulation overrides may be audited as inactive.

    Official ParameterMapper warns and does not apply a nonexistent localized
    parameter; SimulationMapper likewise returns for a missing parameter. This
    is not authority to drop records, infer intended doses or exempt other inputs.
    """
    before=unique(original['Simulations'],'Name');after=unique(restored['Simulations'],'Name')
    if before.keys()!=after.keys():raise ValueError('SOURCE_SIMULATION_SET_CHANGED')
    registered={}
    for name,sim in before.items():
        source=unique(sim.get('Parameters',[]),'Path');mapped=unique(after[name].get('Parameters',[]),'Path')
        if source.keys()!=mapped.keys():raise ValueError('EXPLICIT_SIMULATION_OVERRIDE_SET_CHANGED:'+name)
        registered[name]={}
        for path,record in source.items():
            if _record_digest(record)!=_record_digest(mapped[path]):
                raise ValueError('EXPLICIT_SIMULATION_OVERRIDE_RECORD_CHANGED:'+name+':'+path)
            native_path=path.replace('Applications|','Events|',1) if path.startswith('Applications|') else path
            if native_path in registered[name]:raise ValueError('AMBIGUOUS_NORMALIZED_SIMULATION_OVERRIDE_PATH')
            registered[name][native_path]={'simulation':name,'original_path':path,'native_path':native_path,
                'source_record':deepcopy(record),'source_record_sha256':_record_digest(record),
                'eligibility':'unchanged_explicit_simulation_override_only',
                'requires_absence_in_both_official_builds_and_complete_model_settings_equality':True}
    return registered


def official_snapshot_serialization_equivalent(check,original,*,project_root=ROOT):
    """Audit the running official serializer's artifact; never use epsilon.

    This permits a representation difference only. Both original/restored
    native values and entire models/settings must still be exactly identical.
    The bounded R process and its frozen inputs/runtime bind artifact creation.
    """
    audit=check.get('official_serialization')
    if not isinstance(audit,dict):raise ValueError('OFFICIAL_SERIALIZATION_AUDIT_MISSING')
    unit=check.get('native_unit') if original['unit'] is None else original['unit']
    required={'status':'official_snapshot_serialization_audit',
        'serializer':'PKSim.Core.Services.IJsonSerializer','dto_type':'PKSim.Core.Snapshots.Parameter',
        'parameter_path':original['path'],'actual_native_value':check.get('actual_native_value'),
        'native_unit':check.get('native_unit'),'dimension':check.get('dimension'),
        'declared_unit':unit,'source_unit_was_unspecified':original['unit'] is None,
        'source_value':original['value'],'serialized_value':original['value'],
        'source_value_exactly_reproduced':True}
    if any(not _native_record_equal(audit.get(k),v) for k,v in required.items()):
        raise ValueError('OFFICIAL_SERIALIZATION_SOURCE_OR_NATIVE_BINDING_DIFFERS')
    value=audit.get('source_unit_value_before_serialization')
    if type(value) not in (int,float) or not math.isfinite(value) or not audit.get('serializer_assembly'):
        raise ValueError('OFFICIAL_SERIALIZATION_NATIVE_UNIT_VALUE_MISSING')
    artifact=audit.get('artifact',{})
    try:
        path=migration.inside(artifact['path'],Path(project_root))
        if any('sealed' in part.lower() for part in path.parts):raise ValueError('SEALED_FILE_NOT_A_SERIALIZATION_ARTIFACT')
        if migration.sha(path)!=artifact['sha256']:raise ValueError('OFFICIAL_SERIALIZATION_ARTIFACT_CHANGED')
        encoded=json.loads(path.read_text())
    except (KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('OFFICIAL_SERIALIZATION_ARTIFACT_MISSING_OR_INVALID') from exc
    if (encoded.get('Name')!=original['path'] or encoded.get('Unit')!=unit or
        not _native_record_equal(encoded.get('Value'),original['value']) or
        set(encoded)-{'Name','Unit','Value'}):
        raise ValueError('OFFICIAL_SERIALIZATION_ROUNDTRIP_DIFFERS')
    return {'path':original['path'],'source_value':original['value'],
        'native_value':check['actual_native_value'],'native_unit':check['native_unit'],
        'dimension':check['dimension'],'source_unit_value_before_serialization':value,
        'serialized_value':encoded['Value'],'artifact':deepcopy(artifact),
        'literal_source_to_base_double_equality':False,'numerical_tolerance_used':False}


def validate_parameter_readbacks(row,expectation,*,project_root=ROOT):
    """Validate every returned record against its frozen request, without skips."""
    expected=unique(expectation['parameters'],'path');checks=unique(row['parameter_readbacks'],'path')
    if expected.keys()!=checks.keys():raise ValueError('NATIVE_READBACK_PARAMETER_SET_CHANGED')
    missing=[];present=[];serialized=[]
    for path,check in checks.items():
        original=expected[path]
        if (check.get('source_scope')!=original['source_scope'] or
            not _native_record_equal(check.get('source_record'),original['source_record']) or
            check.get('source_value')!=original['value'] or check.get('source_unit')!=original['unit']):
            raise ValueError('NATIVE_READBACK_SOURCE_RECORD_BINDING_CHANGED:'+path)
        status=check.get('status')
        if status=='missing_in_native_model':
            registration=original.get('inactive_override_registration')
            record_path=original['source_record'].get('Path','')
            normalized_record_path=record_path.replace('Applications|','Events|',1) if record_path.startswith('Applications|') else record_path
            if (not registration or original['source_scope']!='simulation:'+registration['simulation']
                or path!=registration['native_path'] or
                path!=normalized_record_path or registration.get('original_path')!=record_path or
                registration.get('eligibility')!='unchanged_explicit_simulation_override_only' or
                registration.get('requires_absence_in_both_official_builds_and_complete_model_settings_equality') is not True or
                registration['source_record_sha256']!=_record_digest(original['source_record']) or
                _record_digest(registration['source_record'])!=_record_digest(original['source_record']) or
                check.get('accepted') is not False or
                any(k in check for k in ('actual_native_value','expected_native_value','native_unit','absolute_difference'))):
                raise ValueError('UNREGISTERED_OR_NUMERICALLY_MISREPRESENTED_MISSING_PARAMETER:'+path)
            missing.append(path)
        elif status=='present_exact':
            actual=check.get('actual_native_value');wanted=check.get('expected_native_value')
            if (check.get('accepted') is not True or type(actual) not in (int,float) or
                type(wanted) not in (int,float) or not math.isfinite(actual) or not math.isfinite(wanted) or
                actual!=wanted or check.get('absolute_difference')!=0):
                raise ValueError('SOURCE_PARAMETER_NATIVE_DIFFERENCE:'+path)
            present.append(path)
        elif status=='official_snapshot_serialization_equivalent':
            actual=check.get('actual_native_value');wanted=check.get('expected_native_value')
            if (check.get('accepted') is not False or type(actual) not in (int,float) or
                type(wanted) not in (int,float) or not math.isfinite(actual) or not math.isfinite(wanted) or
                actual==wanted or check.get('absolute_difference')!=abs(actual-wanted) or
                not isinstance(check.get('native_unit'),str) or not isinstance(check.get('dimension'),str)):
                raise ValueError('SERIALIZATION_DIFFERENCE_MISREPRESENTED_AS_NUMERIC_EQUALITY:'+path)
            official_snapshot_serialization_equivalent(check,original,project_root=project_root)
            serialized.append(path)
        else:raise ValueError('SOURCE_PARAMETER_NATIVE_DIFFERENCE_OR_UNKNOWN_STATE:'+path)
    if row.get('all_parameters_accepted') is not (not missing and not serialized):
        raise ValueError('NATIVE_NUMERICAL_ACCEPTANCE_SUMMARY_DIFFERS')
    return {'present_exact_paths':present,'missing_pending_bilateral_audit':missing,
            'official_serialization_equivalent_paths':serialized}


def compare_bilateral_readbacks(original,restored,expectation,*,project_root=ROOT):
    left=validate_parameter_readbacks(original,expectation,project_root=project_root)
    right=validate_parameter_readbacks(restored,expectation,project_root=project_root)
    if left!=right:raise ValueError('ORIGINAL_AND_RESTORED_PARAMETER_PRESENCE_DIFFERS')
    checks=unique(expectation['parameters'],'path')
    before=unique(original['parameter_readbacks'],'path');after=unique(restored['parameter_readbacks'],'path')
    for path in left['present_exact_paths']+left['official_serialization_equivalent_paths']:
        for key in ('actual_native_value','expected_native_value','dimension','native_unit'):
            if not _native_record_equal(before[path].get(key),after[path].get(key)):
                raise ValueError('BILATERAL_NATIVE_PARAMETER_DIFFERS:'+path+':'+key)
        if path in left['official_serialization_equivalent_paths']:
            for key in ('source_unit_value_before_serialization','serializer_assembly','declared_unit'):
                if not _native_record_equal(before[path]['official_serialization'].get(key),after[path]['official_serialization'].get(key)):
                    raise ValueError('BILATERAL_NATIVE_SERIALIZATION_DIFFERS:'+path+':'+key)
    return {'active_parameters_exact_per_build':len(left['present_exact_paths']),
        'official_serialization_equivalent_count':len(left['official_serialization_equivalent_paths']),
        'official_serialization_equivalents':[{
            'path':p,'source_scope':checks[p]['source_scope'],'source_record':deepcopy(checks[p]['source_record']),
            'original':official_snapshot_serialization_equivalent(before[p],checks[p],project_root=project_root),
            'restored':official_snapshot_serialization_equivalent(after[p],checks[p],project_root=project_root),
            'native_value_dimension_unit_identical':True,'numerical_tolerance_used':False}
            for p in left['official_serialization_equivalent_paths']],
        'inactive_override_count':len(left['missing_pending_bilateral_audit']),
        'inactive_overrides':[{'path':p,'source_scope':checks[p]['source_scope'],
            **checks[p]['inactive_override_registration'],
            'status':'source_override_inactive_in_both_official_builds',
            'numerical_value_validated':False,'record_preserved':True}
            for p in left['missing_pending_bilateral_audit']],
        'all_source_records_accounted_for':True,
        'scope':'Every frozen scalar parameter expectation; table formulas and other definitions remain protected by full-model equality.',
        'complete_model_and_settings_equality_separately_required':True}


def _constant_parameter_from_pkml(path,parameter_path):
    """Only a named scalar native constant; no formula replacement is allowed."""
    model=ET.parse(path).getroot().find('.//Model')
    if model is None:raise ValueError('SOURCE_NATIVE_MODEL_MISSING')
    found=[]
    def walk(node,prefix):
        children=node.find('Children')
        for child in ([] if children is None else children):
            name=child.get('name')
            current=prefix+([name] if name else [])
            if '|'.join(current)==parameter_path:found.append(child)
            walk(child,current)
    native_root=model.find('Root')
    if native_root is None:raise ValueError('SOURCE_NATIVE_MODEL_ROOT_MISSING')
    walk(native_root,[])
    if len(found)!=1:raise ValueError('UNIQUE_SOURCE_NATIVE_CONSTANT_REQUIRED')
    node=found[0]
    if (node.tag!='Parameter' or 'value' not in node.attrib or
        any('formula' in k.lower() for k in node.attrib) or
        any(x.tag not in {'Info','ValueOrigin'} for x in node)):
        raise ValueError('SOURCE_FORMULA_OR_NONCONSTANT_CANNOT_BE_NORMALIZED')
    value=float(node.get('value'))
    if not math.isfinite(value):raise ValueError('FINITE_SOURCE_NATIVE_CONSTANT_REQUIRED')
    return {'value':value,'dimension':node.get('dim'),'display_unit':node.get('displayUnit'),
            'native_parameter_attributes':dict(node.attrib)}


def apply_native_representation_changes(snapshot,changes,*,root=ROOT):
    """Replay only registered source-unit representations of original constants.

    Raw snapshot records stay in the proof. This reproduces the *original
    effective native model* exactly; it is not an epsilon comparison or fitting.
    Full final native models/settings remain a separate required gate.
    """
    result=deepcopy(snapshot);seen=set()
    profiles={'|'.join(p[k] for k in ('Molecule','Species','Category')):p
              for p in result.get('ExpressionProfiles',[])}
    for change in changes:
        profile=change.get('expression_profile');path=change.get('parameter_path')
        key=(profile,path)
        if key in seen:raise ValueError('DUPLICATE_NATIVE_REPRESENTATION_CHANGE')
        seen.add(key)
        if (change.get('rule')!='original_native_constant_in_unchanged_source_unit' or
            change.get('unit_changed') is not False or change.get('fitting_performed') is not False):
            raise ValueError('NATIVE_REPRESENTATION_RULE_INVALID')
        before=change['source_record'];after=change['restored_record']
        if set(before)!=set(after) or any(not _native_record_equal(before[k],after[k]) for k in before if k!='Value'):
            raise ValueError('NATIVE_REPRESENTATION_MAY_CHANGE_ONLY_VALUE_LITERAL')
        ref=change['original_readback'];readpath=migration.inside(ref['path'],root)
        if migration.sha(readpath)!=ref['sha256']:raise ValueError('ORIGINAL_READBACK_CHANGED')
        report=json.loads(readpath.read_text())
        row=unique(report['simulations'],'name')[change['simulation']]
        check=unique(row['parameter_readbacks'],'path')[path]
        expected={'path':path,'value':before['Value'],'unit':before.get('Unit'),
            'source_scope':'expression_profile:'+profile,'source_record':before}
        if (check.get('status')!='official_snapshot_serialization_equivalent' or
            not _native_record_equal(check.get('source_record'),before) or
            check.get('source_scope')!=expected['source_scope']):
            raise ValueError('SOURCE_BOUND_PROFILE_SERIALIZATION_PROOF_REQUIRED')
        audit=official_snapshot_serialization_equivalent(check,expected,project_root=root)
        target=audit['source_unit_value_before_serialization']
        if (type(after.get('Value')) not in (int,float) or after['Value']!=target or
            after['Value']==before['Value'] or change.get('original_native_value')!=audit['native_value']):
            raise ValueError('NATIVE_REPRESENTATION_RESTORED_VALUE_DIFFERS')
        native_ref=change['original_native_pkml'];native=migration.inside(native_ref['path'],root)
        if (migration.sha(native)!=native_ref['sha256'] or row['pkml_sha256']!=native_ref['sha256'] or
            migration.inside(row['pkml_path'],root)!=native):raise ValueError('SOURCE_NATIVE_PKML_BINDING_DIFFERS')
        constant=_constant_parameter_from_pkml(native,path)
        if constant['value']!=audit['native_value'] or constant['dimension']!=audit['dimension']:
            raise ValueError('ORIGINAL_NATIVE_CONSTANT_DIFFERS_FROM_READBACK')
        if profile not in profiles:raise ValueError('SOURCE_EXPRESSION_PROFILE_MISSING')
        records=unique(profiles[profile].get('Parameters',[]),'Path')
        if path not in records or not _native_record_equal(records[path],before):
            raise ValueError('SOURCE_PROFILE_RECORD_CHANGED_BEFORE_NORMALIZATION')
        records[path]['Value']=after['Value']
    return result


def derive_native_representation_changes(original,restored,expectation,original_readback,*,root=ROOT):
    """Propose a bounded fix only after a real original/restored native drift."""
    validate_parameter_readbacks(original,expectation,project_root=root)
    validate_parameter_readbacks(restored,expectation,project_root=root)
    left=unique(original['parameter_readbacks'],'path');right=unique(restored['parameter_readbacks'],'path')
    expected=unique(expectation['parameters'],'path');changes=[]
    for path,a in left.items():
        b=right[path]
        if a.get('actual_native_value')==b.get('actual_native_value'):continue
        e=expected[path]
        if (a.get('status')!='official_snapshot_serialization_equivalent' or
            b.get('status') not in {'present_exact','official_snapshot_serialization_equivalent'} or
            not e['source_scope'].startswith('expression_profile:') or
            a.get('native_unit')!=b.get('native_unit') or a.get('dimension')!=b.get('dimension')):
            raise ValueError('NATIVE_DRIFT_IS_NOT_REGISTERABLE_SOURCE_PROFILE_REPRESENTATION:'+path)
        audit=official_snapshot_serialization_equivalent(a,e,project_root=root)
        after=deepcopy(e['source_record']);after['Value']=audit['source_unit_value_before_serialization']
        change={'rule':'original_native_constant_in_unchanged_source_unit','simulation':original['name'],
            'expression_profile':e['source_scope'].split(':',1)[1],'parameter_path':path,
            'source_record':deepcopy(e['source_record']),'restored_record':after,
            'original_native_value':a['actual_native_value'],'pre_repair_mapped_native_value':b['actual_native_value'],
            'original_readback':migration.reference(original_readback,root),
            'original_native_pkml':migration.reference(Path(original['pkml_path']),root),
            'unit_changed':False,'fitting_performed':False,
            'scope':'Restore original native constant through an audited JSON representation; raw source retained, no epsilon, full Model/Settings equality required.'}
        changes.append(change)
    if not changes:raise ValueError('NO_REGISTERABLE_NATIVE_REPRESENTATION_CHANGE')
    return changes


def readback(project,expectations,output,*,source_sha256,root,timeout_seconds,cpu_threads,stop_file,_complete_names=None):
    rows=simulation_inventory(project,migration.sha(project),project_root=root)
    full=set(expectations) if _complete_names is None else set(_complete_names)
    if {r['name'] for r in rows}!=full or not set(expectations)<=full:
        raise ValueError('Native source simulation set differs')
    output.mkdir()
    if len(expectations)>1:
        # Keep the complete native build, but bound each readback to one saved
        # simulation. A large four-model export cannot exhaust a shared timer.
        merged={};refs=[]
        for i,name in enumerate(sorted(expectations)):
            values,parts=readback(project,{name:expectations[name]},output/('simulation_%03d'%i),
                source_sha256=source_sha256,root=root,timeout_seconds=timeout_seconds,
                cpu_threads=cpu_threads,stop_file=stop_file,_complete_names=full)
            if set(values)!={name}:raise ValueError('Partitioned readback omitted simulation')
            merged.update(values);refs+=parts
        if set(merged)!=full:raise ValueError('Partitioned readback coverage differs')
        migration.save(output/'readback.json',{'project_sha256':migration.sha(project),
            'simulations':[merged[name] for name in sorted(merged)],'partitioned_readback':True,
            'complete_original_simulation_inventory':sorted(full),'provenance_refs':refs})
        refs.append(migration.reference(output/'readback.json',root))
        return merged,refs
    (output/'preload').mkdir()
    rows=[row for row in rows if row['name'] in expectations]
    config={'project_path':str(project),'project_sha256':migration.sha(project),
            'preload_directory':str(output/'preload'),'result_path':str(output/'readback.json'),'simulations':[],
            'serialization_runtime':[{'path':str(root/p),'sha256':migration.sha(root/p)} for p in SERIALIZATION_RUNTIME_FILES]}
    for i,row in enumerate(rows):
        seed=int(hashlib.sha256((source_sha256+'|'+row['name']).encode()).hexdigest()[:8],16)%2147483646+1
        config['simulations'].append({'name':row['name'],'simulation_id':row['simulation_id'],
            'output_pkml':str(output/('model_%03d.pkml'%i)),'registered_build_seed':seed,**expectations[row['name']]})
    path=output/'request.json';migration.save(path,config)
    process=run_bounded_r(script=root/R_SCRIPT,args=[str(path),migration.sha(path)],output_dir=output,
        project_root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,max_output_bytes=512*1024**2,stop_file=stop_file)
    if process.get('status')!='completed' or process.get('returncode')!=0:raise ValueError('SOURCE_READBACK_NATIVE_FAILED:'+str(output/'execution.log'))
    result=json.loads((output/'readback.json').read_text())
    if result['request_sha256']!=migration.sha(path) or result['project_sha256']!=migration.sha(project):raise ValueError('Readback source binding differs')
    by_name=unique(result['simulations'],'name')
    if set(by_name)!=set(expectations):raise ValueError('Readback result omitted simulation')
    for name,row in by_name.items():
        validate_parameter_readbacks(row,expectations[name],project_root=root)
        if migration.sha(Path(row['pkml_path']))!=row['pkml_sha256']:raise ValueError('Readback PKML changed')
    refs=[migration.reference(p,root) for p in sorted(output.rglob('*')) if p.is_file()]
    return by_name,refs


def repair_complex_snapshot(original,native,output_dir,*,source_provenance,root=ROOT,stage_runner=None,
                            timeout_seconds=180,cpu_threads=1,stop_file=None,max_simulations=None):
    root=Path(root).resolve();out=migration.inside(output_dir,root,exists=False)
    if out.exists():raise FileExistsError('Preserve each complex repair attempt')
    source=migration.inside(source_provenance['source_path'],root);source_sha=source_provenance['source_sha256']
    registration=registered_source(source_provenance,original,root)
    if migration.clean_snapshot(json.loads(source.read_text()))!=migration.clean_snapshot(original):raise ValueError('Original document differs from pinned source')
    if native.get('Version')!=80:raise ValueError('Actual official native version evidence required')
    assert_no_explicit_simulation_seed(original)
    if stage_runner is None:
        from .snapshot_repair import _native_stage
        stage_runner=_native_stage
    mapped=map_legacy_expression_schema(original,native);source_expected=deepcopy(mapped['mapped_snapshot'])
    source_simulations=unique(source_expected['Simulations'],'Name')
    restored=deepcopy(source_expected)
    inactive_registry=register_preserved_simulation_overrides(original,restored)
    restored,path_changes=normalize_application_paths(restored)
    assert_no_explicit_simulation_seed(restored)
    simulations=unique(restored['Simulations'],'Name');names=sorted(simulations)
    if max_simulations is not None:
        if type(max_simulations) is not int or max_simulations<1:raise ValueError('Positive diagnostic subset required')
        names=names[:max_simulations]
    out.mkdir(parents=True)
    batch_size=1 if len(original.get('Compounds',[]))>8 else 4
    config={'administration_path_schema_changes':path_changes,'schema':VERSION,'source':migration.reference(source,root),'source_version':original['Version'],
        'catalog_registration':registration,
        'all_source_simulations':sorted(simulations),'registered_simulations':names,'batch_size':batch_size,
        'batch_rule':'one simulation when global compounds >8, otherwise four; complete source simulation inventory retained',
        'build_seed_rule':'SHA256(original source SHA + | + exact simulation name) first 32 bits modulo 2147483646 + 1; original JSON has no SimulationSettings seed; Individual/Population seeds unchanged',
        'native_readback_unit_conversion':'OSPSuite toBaseUnit; exact double equality OR separate official runtime serializer representation audit; zero numerical tolerance; bilateral native values/dimensions/units and whole Model/Settings strictly identical',
        'native_record_echo':'JSON integer/floating spelling may differ (1.0 to 1); exact numeric equality only, with boolean/string types distinct and no tolerance. Original-to-mapped records remain hash-identical.',
        'simulation_seed_absence_checked':True,
        'inactive_override_official_mapper_evidence':[migration.reference(root/p,root) for p in INACTIVE_MAPPER_EVIDENCE],
        'official_serialization_evidence':[migration.reference(root/p,root) for p in SERIALIZATION_EVIDENCE],
        'official_serialization_runtime':[migration.reference(root/p,root) for p in SERIALIZATION_RUNTIME_FILES],
        'official_serialization_rule':'Running DI IJsonSerializer serializes Parameter.Value nullable double after OSPSuite base-to-declared-unit conversion; artifact Value must reproduce source exactly in each build. No Python/R round or epsilon. Supporting source version is documentation only; actual runtime is pinned separately.',
        'native_representation_recovery_rule':'Only an original source-bound scalar constant in an expression profile, with exact official serialization to the unchanged raw source value, may use the original native value expressed in the SAME original unit in the mapped JSON. Preserve both records; rebuild, require exact bilateral native and whole Model/Settings equality. Formula changes forbidden.',
        'implementation':[migration.reference(root/p,root) for p in IMPLEMENTATION_FILES],
        'observations_read':False,'ode_run':False}
    migration.save(out/'registration.json',config)
    def check_implementation():
        if any(migration.sha(root/r['path'])!=r['sha256'] for r in config['implementation']+config['inactive_override_official_mapper_evidence']+config['official_serialization_evidence']+config['official_serialization_runtime']):
            raise ValueError('COMPLEX_SNAPSHOT_IMPLEMENTATION_CHANGED_DURING_PROOF')
    proofs=[];representation_changes=[]
    refs=[migration.reference(out/'registration.json',root),*config['inactive_override_official_mapper_evidence'],*config['official_serialization_evidence'],*config['official_serialization_runtime']]
    for offset in range(0,len(names),batch_size):
        check_implementation()
        chosen=names[offset:offset+batch_size];group=out/('batch_%02d'%(offset//batch_size+1));group.mkdir()
        # Audit source records in their original representation even after an
        # explicitly registered effective-native-model representation recovery.
        expectations={n:parameter_expectations(source_expected,source_simulations[n]) for n in chosen}
        for name,expectation in expectations.items():
            for check in expectation['parameters']:
                if check['source_scope']=='simulation:'+name:
                    registration=inactive_registry[name].get(check['path'])
                    if not registration or _record_digest(registration['source_record'])!=_record_digest(check['source_record']):
                        raise ValueError('SIMULATION_OVERRIDE_REGISTRATION_MISSING_OR_CHANGED')
                    check['inactive_override_registration']=registration
        outputs={}
        for label,snapshot in [('original',original),('restored',restored)]:
            part=source_partition(snapshot,chosen)
            inp=group/(label+'.json');migration.save(inp,part);refs.append(migration.reference(inp,root))
            project,stage_refs=stage_runner('snapshot_to_project',inp,group/(label+'_build'),root=root,
                timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            refs+=stage_refs
            output,read_refs=readback(project,expectations,group/(label+'_readback'),source_sha256=source_sha,
                root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            outputs[label]=output;refs+=read_refs
        pending={}
        for name in chosen:
            try:
                compare_bilateral_readbacks(outputs['original'][name],outputs['restored'][name],expectations[name],project_root=root)
            except ValueError:
                proposals=derive_native_representation_changes(outputs['original'][name],outputs['restored'][name],
                    expectations[name],group/'original_readback/readback.json',root=root)
                for change in proposals:
                    key=(change['expression_profile'],change['parameter_path'])
                    old=next((c for c in representation_changes if (c['expression_profile'],c['parameter_path'])==key),None)
                    if old is not None:raise ValueError('SOURCE_PROFILE_NORMALIZATION_WOULD_DEPEND_ON_SIMULATION')
                    if key in pending and not _native_record_equal(pending[key]['restored_record'],change['restored_record']):
                        raise ValueError('CONFLICTING_SOURCE_PROFILE_NATIVE_REPRESENTATIONS')
                    pending.setdefault(key,change)
        if pending:
            new_changes=list(pending.values())
            restored=apply_native_representation_changes(source_expected,[*representation_changes,*new_changes],root=root)
            restored,_=normalize_application_paths(restored)
            representation_changes+=new_changes
            recovery=group/'native_representation_recovery.json';migration.save(recovery,{
                'changes':new_changes,'pre_recovery_original':outputs['original'],'pre_recovery_restored':outputs['restored'],
                'no_fitting':True,'requires_complete_final_model_equality':True})
            refs.append(migration.reference(recovery,root))
            inp=group/'restored_native_representation.json';migration.save(inp,source_partition(restored,chosen))
            refs.append(migration.reference(inp,root))
            project,stage_refs=stage_runner('snapshot_to_project',inp,group/'restored_native_representation_build',root=root,
                timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            refs+=stage_refs
            output,read_refs=readback(project,expectations,group/'restored_native_representation_readback',source_sha256=source_sha,
                root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
            outputs['restored']=output;refs+=read_refs
        for name in chosen:
            a=ET.parse(outputs['original'][name]['pkml_path']).getroot();b=ET.parse(outputs['restored'][name]['pkml_path']).getroot()
            models=compare_saved_model_xml(a.find('.//Model'),b.find('.//Model'))
            settings=compare_saved_settings_xml(a.find('.//SimulationSettings'),b.find('.//SimulationSettings'))
            presence=compare_bilateral_readbacks(outputs['original'][name],outputs['restored'][name],expectations[name],project_root=root)
            proof={'simulation':name,'model':models,'settings':settings,
                   'explicit_parameter_count':len(expectations[name]['parameters']),
                   'parameter_presence_audit':presence,
                   'original_readback':outputs['original'][name],'restored_readback':outputs['restored'][name]}
            proofs.append(proof);migration.save(group/('proof_'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.json'),proof)
            if not models['accepted'] or not settings['accepted']:raise ValueError('SOURCE_COMPLETE_MODEL_OR_SETTINGS_CHANGED:'+name)
        print(json.dumps({'complex_snapshot_source':source_sha[:12],'completed_simulations':len(proofs),'total':len(names)}),flush=True)
    check_implementation()
    full=set(names)==set(simulations)
    transforms=[v for c in mapped['changes'] for v in c.get('source_to_target_values',[])]
    numeric_transform=any(v['source_value']!=v['target_value'] for v in transforms)
    proof={'administration_path_schema_changes':path_changes,'accepted':full,'schema':VERSION,'simulations_checked':len(proofs),'expected_simulations':sorted(simulations),
           'explicit_parameter_readbacks_per_build':sum(p['explicit_parameter_count'] for p in proofs),
           'active_parameters_exact_per_build':sum(p['parameter_presence_audit']['active_parameters_exact_per_build'] for p in proofs),
           'official_snapshot_serialization_equivalents_per_build':sum(p['parameter_presence_audit']['official_serialization_equivalent_count'] for p in proofs),
           'bilateral_native_numeric_dimension_unit_equality':True,'numerical_tolerance_used':False,
           'inactive_explicit_simulation_overrides':sum(p['parameter_presence_audit']['inactive_override_count'] for p in proofs),
           'inactive_overrides_are_not_numerical_validation':True,
           'inactive_override_limit':'Preserves the effective official source model only; does not reconstruct missing administrations or infer an intended regimen from names.',
           'all_complete_models_and_settings_equal':True,'source_records_preserved_in_original_and_conversion_provenance':True,
           'source_version':original['Version'],'native_version':80,'version_only_rewrite':False,
           'schema_changes':mapped['changes'],'simulation_proofs':proofs,'ode_run':False,'observations_read':False,
           'fitting_performed':False,'numeric_parameters_modified':numeric_transform or bool(representation_changes),
           'source_json_numeric_literals_changed_by_official_schema_conversion':numeric_transform,
           'source_json_numeric_literals_changed_by_native_representation_recovery':bool(representation_changes),
           'native_representation_changes':representation_changes,
           'source_to_target_numeric_conversions':transforms,'original_native_effective_model_changed':False,
           'literal_numeric_identity_claimed':not numeric_transform and not representation_changes,'provenance_refs':refs}
    migration.save(out/'proof.json',proof);refs.append(migration.reference(out/'proof.json',root))
    return {'snapshot':restored if full else None,'proof':proof,'provenance_refs':refs}
