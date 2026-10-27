"""One registered expression-profile export projection; no epsilon or ODE."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import recommendation_candidate_migration as migration
from . import source_pkml

ROOT=migration.ROOT
RULE='source_bound_native_expression_profile_serialization'
SOURCE_SHA='582e45fc2400f2a7d7e5198b581c7847de7a75765708b73440ee8d9fcda70659'
INPUT_SHA='72515fad44516faad02e5aeff9e9d5731d9fd6ef98198cfc9b4f55c044070247'
NATIVE_SHA='c6787d4350a4759ac762870462e15164b301e54e1c605bda39228ce43c9e5fdb'
EXPORT_SHA='aecffee97ecd0b654cd102d069dec262ec72618dbf6c0f01dc4890e9ad34ba95'
WORKER='src/pbpk_kb/model_runtime/R/recommendation_expression_serialization.R'
EVIDENCE_PREFIX='artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/'
EVIDENCE_FILES=[EVIDENCE_PREFIX+p for p in ('PKSim.Core/Snapshots/Mappers/ExpressionProfileMapper.cs',
    'PKSim.Core/Snapshots/Mappers/ParameterMapper.cs','PKSim.Core/Model/ParameterExtensions.cs',
    'PKSim.Infrastructure/Serialization/Json/NullabeDoubleJsonConverter.cs',
    'PKSim.Infrastructure/Serialization/Json/PKSimJsonSerializerSettings.cs',
    'PKSim.Core/Snapshots/Parameter.cs','PKSim.Core/Services/IJsonSerializer.cs',
    'PKSim.Core/Model/SystemicProcessSelection.cs','PKSim.Core/Snapshots/Mappers/SimulationMapper.cs')]
RUNTIME_FILES=['runtime/ospsuite-env/lib/R/library/ospsuite/lib/'+p for p in (
    'PKSim.Core.dll','PKSim.Infrastructure.dll','OSPSuite.Core.dll','OSPSuite.Utility.dll','Newtonsoft.Json.dll')]


def _require(ok,message):
    if not ok:raise ValueError('Expression-profile proof: '+message)


def _exact(a,b):
    if type(a) in (int,float) and type(b) in (int,float):return math.isfinite(a) and math.isfinite(b) and a==b
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(_exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(_exact(x,y) for x,y in zip(a,b))
    return a==b


def _same(a,b):return _exact(migration._normalize(migration.clean_snapshot(a)),migration._normalize(migration.clean_snapshot(b)))


def _unique(rows,key):
    result={key(r):r for r in rows};_require(len(result)==len(rows),'duplicate explicit identity');return result


def _profile_name(r):return '|'.join(r[k] for k in ('Molecule','Species','Category'))


def _parameters(expression):
    rows={}
    def visit(node,path):
        if node.get('name') and node.get('name')!='ROOT' and node.get('id'):path=path+[node.get('name')]
        if node.tag=='Para':
            key='|'.join(path);_require(key not in rows,'duplicate native parameter path');rows[key]=node
        for child in node:visit(child,path)
    individual=expression.find('Individual');_require(individual is not None,'native expression individual absent')
    visit(individual,[]);return rows


def _native_value(row,parameter):
    _require(parameter is not None and parameter.tag=='Para' and parameter.get('value') is not None
             and not any(parameter.get(k) for k in ('formula','rhs','rHSFormula')), 'source difference requires an explicit native scalar')
    value=float(parameter.get('value'));_require(math.isfinite(value),'nonfinite native scalar')
    if row.get('Unit') is None:
        _require(parameter.get('dim')=='Fraction','unitless expression difference is not a fraction');factor=1.
    else:
        _require(row['Path']=='CYP2C8|t1/2 (liver)' and row['Unit']=='h' and parameter.get('dim')=='Time'
                 and parameter.get('displayUnit')=='h','unregistered dimension or unit');factor=60.
    _require(type(row.get('Value')) in (int,float) and value==row['Value']*factor
             and value/factor==row['Value'],'native parameter does not exactly preserve source value')
    return value,factor


DEFAULT_GLOBALS={'Undefined Liver|Reference concentration':{'Value':1.,'Unit':'µmol/l'},
                 'Undefined Liver|t1/2 (intestine)':{'Value':23.,'Unit':'h'},
                 'Undefined Liver|t1/2 (liver)':{'Value':36.,'Unit':'h'}}


def _simulation_exports(original,export):
    left=migration._normalize(migration.clean_snapshot(original))
    right=migration._normalize(migration.clean_snapshot(export))
    a=_unique(left['Simulations'],lambda r:r['Name']);b=_unique(right['Simulations'],lambda r:r['Name'])
    _require(a.keys()==b.keys(),'simulation identity changed')
    extras={}
    for name,sim in a.items():
        before=_unique(sim.get('Parameters',[]),lambda r:r['Path'])
        after=_unique(b[name].get('Parameters',[]),lambda r:r['Path'])
        _require(before.keys()<=after.keys() and all(_exact(v,after[k]) for k,v in before.items()),
                 'original simulation parameter removed or changed')
        added={k:v for k,v in after.items() if k not in before}
        _require(added=={k:{'Path':k,**v} for k,v in DEFAULT_GLOBALS.items()},
                 'unregistered additional simulation parameters')
        _require(any(p.get('SystemicProcessType')=='Hepatic' for c in sim['Compounds'] for p in c.get('Processes',[])),
                 'original source did not select the undefined hepatic mechanism')
        extras[name]=added
        b[name]['Parameters']=[v for k,v in after.items() if k in before]
    left.pop('ExpressionProfiles');right.pop('ExpressionProfiles')
    _require(_same(left,right),'non-expression/nonregistered-global source data changed')
    return extras


def _model_parameters(document):
    result={}
    def visit(node,path):
        if node.get('name') and node.get('id'):path=path+[node.get('name')]
        if node.tag=='Para':
            key='|'.join(path);_require(key not in result,'duplicate model parameter');result[key]=node
        for child in node:visit(child,path)
    for node in document.find('Model/Root'):visit(node,[])
    return result


def projection(input_ref,export_ref,native_ref,*,project_root=ROOT):
    root=Path(project_root).resolve(strict=True)
    _require((input_ref['sha256'],export_ref['sha256'],native_ref['sha256'])==(INPUT_SHA,EXPORT_SHA,NATIVE_SHA),
             'unregistered partition, export or native source')
    original=json.loads(migration.checked(input_ref,root).read_text());export=json.loads(migration.checked(export_ref,root).read_text())
    native=migration.checked(native_ref,root)
    _require(original.get('Version')==export.get('Version')==80,'native snapshot version differs')
    global_exports=_simulation_exports(original,export)
    left=_unique(original['ExpressionProfiles'],_profile_name);right=_unique(export['ExpressionProfiles'],_profile_name)
    _require(left.keys()==right.keys() and len(left)==90,'expression identity set changed')
    numeric=[];omitted=[];profile_refs=[];used_refs=[];global_refs=[]
    with sqlite3.connect(native.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        templates={}
        for ident,name,blob in db.execute('SELECT b.Id,b.Name,c.Data FROM EXPRESSION_PROFILES e JOIN BUILDING_BLOCKS b ON e.ExpressionProfileId=b.Id JOIN CONTENTS c ON b.ContentId=c.Id'):
            document=ET.fromstring(source_pkml.decompress_model_blob(blob))
            _require(name not in templates and document.tag=='ExpressionProfile' and document.get('id')==ident
                     and document.get('name')==name,'native template identity differs')
            templates[name]=(ident,document,hashlib.sha256(blob).hexdigest())
        _require(templates.keys()==left.keys(),'native expression template coverage changed')
        differences={}
        for name in sorted(left):
            a,b=left[name],right[name]
            _require(_same({k:v for k,v in a.items() if k!='Parameters'},
                           {k:v for k,v in b.items() if k!='Parameters'}),'expression direction/type/localization changed')
            ap=_unique(a.get('Parameters',[]),lambda r:r['Path']);bp=_unique(b.get('Parameters',[]),lambda r:r['Path'])
            _require(bp.keys()<=ap.keys(),'unregistered added expression parameter')
            ident,doc,blob_sha=templates[name];nodes=_parameters(doc);differences[name]=[]
            profile_refs.append({'profile':name,'template_id':ident,'blob_sha256':blob_sha})
            for path,row in sorted(ap.items()):
                out=bp.get(path)
                if _exact(row,out):continue
                parameter=nodes.get(path);value,factor=_native_value(row,parameter)
                common={'profile':name,'path':path,'source_record':deepcopy(row),'export_record':deepcopy(out),
                    'native_parameter_id':parameter.get('id'),'native_parameter_node_sha256':source_pkml.element_sha(parameter),
                    'native_value':value,'unit_factor':factor,'native_is_default':parameter.get('isDefault')}
                if out is None:
                    _require(path.rsplit('|',1)[-1] in ('Fraction expressed apical','Fraction expressed at blood brain barrier')
                             and row['Value'] in (0,1) and parameter.get('isDefault')=='1'
                             and set(row)=={'Path','Value'},'unregistered omitted expression value')
                    omitted.append(common)
                else:
                    _require(_exact({k:v for k,v in row.items() if k!='Value'},{k:v for k,v in out.items() if k!='Value'})
                             and type(out.get('Value')) in (int,float) and math.isfinite(out['Value']),
                             'non-value expression record changed')
                    numeric.append(common)
                differences[name].append(row)
        sims=_unique(original['Simulations'],lambda r:r['Name']);individuals=_unique(original['Individuals'],lambda r:r['Name'])
        actual=list(db.execute('SELECT b.Id,b.Name,c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON s.SimulationId=b.Id JOIN CONTENTS c ON b.ContentId=c.Id'))
        _require({n for _,n,_ in actual}==sims.keys() and len(actual)==4,'native simulation coverage differs')
        for simulation_id,name,blob in actual:
            document=ET.fromstring(source_pkml.decompress_model_blob(blob))
            _require(document.get('id')==simulation_id and document.get('name')==name and document.find('Model') is not None,
                     'saved simulation identity changed')
            actual_parameters=_model_parameters(document)
            for path,row in sorted(global_exports[name].items()):
                node=actual_parameters.get(path);factor=60. if row['Unit']=='h' else 1.
                _require(node is not None and node.get('isDefault')=='1' and node.get('mode')=='Global'
                         and node.get('value') is not None and not any(node.get(k) for k in ('formula','rhs','rHSFormula'))
                         and node.get('displayUnit')==row['Unit']
                         and node.get('dim')==('Time' if row['Unit']=='h' else 'Concentration (molar)')
                         and float(node.get('value'))==row['Value']*factor,
                         'exported global is not the unchanged native source default')
                global_refs.append({'simulation':name,'simulation_id':simulation_id,'path':path,'export_record':row,
                    'native_value':float(node.get('value')),'unit_factor':factor,'native_parameter_id':node.get('id'),
                    'native_parameter_node_sha256':source_pkml.element_sha(node)})
            blocks=document.find('UsedBuildingBlocks');_require(blocks is not None,'native used-block chain absent')
            used={n.get('name'):n for n in blocks if n.find('ExpressionProfile') is not None}
            expected=set(individuals[sims[name]['Individual']]['ExpressionProfiles']) & left.keys()
            _require(expected<=used.keys(),'explicit expression profile missing from native simulation')
            for profile in sorted(expected):
                block=used[profile];child=block.find('ExpressionProfile');template_id=templates[profile][0]
                db_rows=db.execute('SELECT Id,TemplateId FROM USED_BUILDING_BLOCKS WHERE SimulationId=? AND Name=? AND BuildingBlockType=?',(simulation_id,profile,'ExpressionProfile')).fetchall()
                _require(db_rows==[(block.get('id'),template_id)] and block.get('templateId')==template_id
                         and child.get('id')==block.get('id') and child.get('name')==profile and block.get('altered')=='0',
                         'native simulation substituted or altered the original expression template')
                params=_parameters(child)
                for row in differences[profile]:_native_value(row,params.get(row['Path']))
                used_refs.append({'simulation_id':simulation_id,'simulation':name,'profile':profile,
                    'used_id':child.get('id'),'template_id':template_id,'used_profile_sha256':source_pkml.element_sha(child),
                    'all_changed_export_parameters_exactly_preserved':True})
    _require(migration.sha(native)==native_ref['sha256'],'native project changed while read')
    _require(len(numeric)==57 and len(omitted)==156,'registered difference census changed')
    return {'rule':RULE,'source_sha256':SOURCE_SHA,'input':input_ref,'export':export_ref,'native':native_ref,
        'profile_count':len(left),'numeric_export_differences':numeric,'default_omissions':omitted,
        'native_expression_templates':profile_refs,'simulation_expression_bindings':used_refs,'simulation_global_exports':global_refs,
        'all_other_snapshot_data_exactly_unchanged':True,'original_legacy_complete_model_comparison_required':True,'native_source_values_exactly_preserved':True,
        'numerical_tolerance_used':False,'clinical_outcomes_accessed':False}


def serialization_items(projected):
    return [{'profile':r['profile'],'path':r['path'],'value':r['source_record']['Value'],
             'unit':r['source_record'].get('Unit'),'expected_serialized_value':r['export_record']['Value']}
            for r in projected['numeric_export_differences']]


def verify(input_ref,export_ref,native_ref,proof_ref,*,project_root=ROOT):
    root=Path(project_root).resolve(strict=True);proof=json.loads(migration.checked(proof_ref,root).read_text())
    expected=projection(input_ref,export_ref,native_ref,project_root=root)
    _require(proof.get('projection')==expected and proof.get('accepted') is True,'registered expression proof differs')
    refs={r['path']:r for r in proof['evidence_refs']}
    _require(len(refs)==len(proof['evidence_refs']),'duplicate proof evidence')
    for r in refs.values():migration.checked(r,root)
    def read(ref):
        _require(refs.get(ref['path'])==ref,'serialization evidence is not registered')
        return json.loads(migration.checked(ref,root).read_text())
    registration=read(proof['registration']);request=read(proof['request']);process=read(proof['process']);result=read(proof['result'])
    baseline=proof['original_equivalence']
    baseline_registration=read(baseline['registration']);baseline_result=read(baseline['build_result'])
    original_partition=json.loads(migration.checked(baseline_registration['registered_original_partition'],root).read_text())
    source=json.loads(migration.checked(baseline_registration['source'],root).read_text())
    from . import complex_snapshot,json_native_source_proof as bridge
    _require(baseline_registration['source']['sha256']==SOURCE_SHA
             and baseline_registration['target_verified_partition']==input_ref
             and _exact(original_partition,complex_snapshot.source_partition(source,baseline_registration['simulation_names'])),
             'original baseline partition differs from literal source')
    stage_refs=[r for r in baseline_result['refs'] if r['path'].endswith('/build/stage_receipt.json')]
    _require(len(stage_refs)==1,'unique original-source build receipt required')
    stage=read(stage_refs[0]);original_ref=baseline_registration['registered_original_partition']
    _require(stage.get('input')==original_ref and stage.get('mode')=='snapshot_to_project' and stage.get('status')=='completed'
             and all(stage.get(k)==original_ref['sha256'] for k in ('copy_before_sha256','copy_after_sha256','original_after_sha256')),
             'literal original-source input changed during native build')
    original_process,out,runtime=bridge._process(stage['process'],root=root,refs=refs,mode='snapshot_to_project',
        script_sha=refs[migration.R_SCRIPT]['sha256'])
    original_native=baseline_result['native']
    _require(migration.checked(original_native,root).parent==out
             and Path(original_native['path']).stem==Path(original_ref['path']).stem,'original native output identity differs')
    actual_baseline=original_model_equivalence(baseline_registration['source'],input_ref,original_native,native_ref,project_root=root)
    _require(proof['original_model_equivalence']==actual_baseline,'complete original-native model proof differs')
    _require(registration['input']==input_ref and registration['export']==export_ref and registration['native']==native_ref
             and request['items']==serialization_items(expected),'official serialization input differs')
    _require(request['serialization_runtime']==registration['serialization_runtime']
             and {r['path'] for r in registration['serialization_runtime']}==set(RUNTIME_FILES)
             and {r['path'] for r in registration['official_source_evidence']}==set(EVIDENCE_FILES),
             'serializer runtime or official mapper evidence differs')
    for r in registration['serialization_runtime']+registration['official_source_evidence']:_require(refs.get(r['path'])==r,'mapper/runtime not frozen')
    worker=registration['worker'];_require(worker['path']==WORKER and refs.get(worker['path'])==worker,'serialization worker not registered')
    command=process.get('command',[])
    _require(process.get('status')=='completed' and type(process.get('returncode')) is int and process['returncode']==0
             and process.get('runtime_unchanged') is True and process.get('registry_modified') is False
             and process.get('script_sha256')==worker['sha256'] and migration.inside(process['script'],root)==root/WORKER
             and len(command)==5 and command[1]=='--vanilla' and migration.inside(command[2],root)==root/WORKER
             and migration.inside(command[3],root)==root/proof['request']['path'] and command[4]==proof['request']['sha256'],
             'official serializer process/request/runtime differs')
    _require(result.get('request_sha256')==proof['request']['sha256'] and type(result.get('solver_calls')) is int
             and result['solver_calls']==0 and result.get('observations_read') is False
             and result.get('parameter_values_modified') is False
             and result.get('serializer')=='PKSim.Core.Services.IJsonSerializer'
             and isinstance(result.get('serializer_assembly'),str) and len(result.get('parameters',[]))==len(request['items']),
             'official serializer result contract differs')
    for item,answer in zip(request['items'],result['parameters']):
        artifact=read(answer['artifact'])
        _require(answer['item']==item and artifact.get('Name')==item['path'] and artifact.get('Unit')==item['unit']
                 and artifact.get('Value')==item['expected_serialized_value'] and set(artifact)<= {'Name','Unit','Value'},
                 'official serialized value does not exactly reproduce native export')
    return {'accepted':True,'rule':RULE,'native_source_values_exactly_preserved':True,'numerical_tolerance_used':False,
        'official_serialized_values_reproduced':57,'native_default_omissions_verified':156,'clinical_outcomes_accessed':False}


_BASELINE_CACHE={}


def original_model_equivalence(source_ref,partition_ref,original_native_ref,verified_native_ref,*,project_root=ROOT):
    """Complete native model equality, with only declared build-seed control.

    The saved projects are read-only. Native generated simulation seeds are
    recorded and changed only in comparison copies using the existing registered
    complex_snapshot convention. Every source Individual/Population seed and all
    other solver/output settings remain strict.
    """
    from . import model_semantics,complex_snapshot
    root=Path(project_root).resolve(strict=True)
    source=json.loads(migration.checked(source_ref,root).read_text())
    partition=json.loads(migration.checked(partition_ref,root).read_text())
    _require(source_ref['sha256']==SOURCE_SHA and partition_ref['sha256']==INPUT_SHA,'original-model source binding differs')
    complex_snapshot.assert_no_explicit_simulation_seed(source)
    complex_snapshot.assert_no_explicit_simulation_seed(partition)
    names=[r['Name'] for r in partition['Simulations']]
    for collection in ('Individuals','Populations'):
        a={r['Name']:r.get('Seed') for r in source.get(collection,[])}
        b={r['Name']:r.get('Seed') for r in partition.get(collection,[])}
        _require(all(name in a and a[name]==seed for name,seed in b.items()),'source subject/population seed changed')
    originals=migration.checked(original_native_ref,root);verified=migration.checked(verified_native_ref,root)
    key=(original_native_ref['sha256'],verified_native_ref['sha256'],migration.sha(Path(__file__)),
         migration.sha(Path(model_semantics.__file__)),migration.sha(Path(complex_snapshot.__file__)))
    if key in _BASELINE_CACHE:return deepcopy(_BASELINE_CACHE[key])
    def documents(path):
        with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            rows=list(db.execute('SELECT b.Name,c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON s.SimulationId=b.Id JOIN CONTENTS c ON b.ContentId=c.Id'))
        result={}
        for name,blob in rows:
            _require(name not in result,'duplicate original native simulation')
            result[name]=ET.fromstring(source_pkml.decompress_model_blob(blob))
        _require(set(result)==set(names),'original native simulation coverage differs')
        return result
    a,b=documents(originals),documents(verified);checks=[]
    for name in sorted(a):
        model=model_semantics.compare_saved_model_xml(a[name].find('Model'),b[name].find('Model'))
        sa,sb=deepcopy(a[name].find('Settings')),deepcopy(b[name].find('Settings'))
        _require(sa is not None and sb is not None and sa.get('randomSeed') and sb.get('randomSeed'),
                 'explicit native-generated simulation seed required')
        before={'original':sa.get('randomSeed'),'verified':sb.get('randomSeed')}
        seed=int(hashlib.sha256((SOURCE_SHA+'|'+name).encode()).hexdigest()[:8],16)%2147483646+1
        sa.set('randomSeed',str(seed));sb.set('randomSeed',str(seed))
        settings=model_semantics.compare_saved_settings_xml(sa,sb)
        _require(model['accepted'] is True and settings['accepted'] is True,'original and verified full model/settings differ')
        checks.append({'simulation':name,'complete_model':model,'controlled_seed_settings':settings,
            'actual_saved_simulation_seeds':before,'registered_comparison_seed':seed,
            'seed_rule':'complex_snapshot.readback: SHA256(source SHA + | + simulation name) first 32 bits modulo 2147483646 plus 1',
            'saved_projects_changed':False,'source_individual_population_seeds_unchanged':True})
    _require(migration.sha(originals)==original_native_ref['sha256'] and migration.sha(verified)==verified_native_ref['sha256'],
             'original or verified project changed during comparison')
    _BASELINE_CACHE[key]=deepcopy(checks)
    return checks
