"""Read source-explicit omissions from newly built, unsimulated native models."""
from copy import deepcopy
from decimal import Decimal
import sqlite3
import xml.etree.ElementTree as ET
from .snapshot_overlay import unique,migration
from .source_pkml import simulation_inventory,decompress_model_blob,model_parameter_inventory

OFFICIAL_FILES=['artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Mappers/'+p
    for p in ('CompoundPropertiesMapper.cs','SimulationMapper.cs','ParameterMapper.cs')]


def omitted_fields(overlay):
    checks={}
    for change in overlay['changes']:
        name=change['simulation_name'];checks.setdefault(name,[])
        if change['field']=='Parameters':
            old=unique(migration._normalize(change['source'] or [],'Parameters'),'Path')
            native=unique(migration._normalize(change['native_before'] or [],'Parameters'),'Path')
            if not native.keys()<=old.keys() or any(old[k]!=v for k,v in native.items()):
                raise ValueError('READBACK_CANNOT_HIDE_CHANGED_PARAMETER_OR_NEW_OVERRIDE')
            for path in old.keys()-native.keys():
                p=old[path]
                if p.get('Unit') or p.get('TableFormula') or set(p)-{'Path','Value','ValueOrigin'}:
                    raise ValueError('ONLY_EXPLICIT_DIMENSIONLESS_CONSTANT_OMISSION_AUDITED')
                if not isinstance(p.get('Value'),(float,int)) or not Decimal(str(p['Value'])).is_finite():
                    raise ValueError('FINITE_EXPLICIT_PARAMETER_VALUE_REQUIRED')
                checks[name].append({'kind':'dimensionless_parameter','path':path,'record':p})
        elif change['field']=='Alternatives':
            old={p['GroupName']:p for p in change['source'] or []}
            native={p['GroupName']:p for p in change['native_before'] or []}
            if len(old)!=len(change['source']) or not native.keys()<=old.keys() or any(old[k]!=v for k,v in native.items()):
                raise ValueError('READBACK_CANNOT_HIDE_CHANGED_ALTERNATIVE')
            for group in old.keys()-native.keys():
                checks[name].append({'kind':'unique_alternative','compound':change['compound_name'],
                    'group':group,'alternative':old[group]['AlternativeName']})
        else:raise ValueError('UNREVIEWED_OVERLAY_FIELD')
    if not checks or any(not c for c in checks.values()):raise ValueError('REAL_OMISSION_REQUIRED')
    return checks


def inspect_native_project(project,expected,*,root):
    rows=simulation_inventory(project,migration.sha(project),project_root=root)
    if {r['name'] for r in rows}!=set(expected):raise ValueError('NATIVE_READBACK_SIMULATION_SET_CHANGED')
    report=[]
    with sqlite3.connect(project.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        for row in rows:
            d=ET.fromstring(decompress_model_blob(db.execute('SELECT Data FROM CONTENTS WHERE Id=?',(row['content_id'],)).fetchone()[0]))
            if d.get('id')!=row['simulation_id'] or d.get('name')!=row['name']:raise ValueError('NATIVE_SIMULATION_ID_CHANGED')
            parameters=model_parameter_inventory(d.find('Model'))
            compounds={c.get('name'):c for c in d.findall('UsedBuildingBlocks/UsedBuildingBlock/Compound')}
            selections={c.get('compound'):c for c in d.findall('Properties/CompoundPropertiesList/CompoundProperties')}
            for check in expected[row['name']]:
                proof={'simulation':row['name'],**check}
                if check['kind']=='dimensionless_parameter':
                    p=parameters.get(check['path'])
                    if not p or p['dimension'] not in (None,'Dimensionless') or p['formula_id'] or p['rhs_formula_id']:
                        raise ValueError('SOURCE_EXPLICIT_PARAMETER_NOT_NATIVE_DIMENSIONLESS_CONSTANT')
                    if Decimal(p['value'])!=Decimal(str(check['record']['Value'])):
                        raise ValueError('SOURCE_EXPLICIT_NATIVE_VALUE_DIFFERENCE')
                    proof['native_parameter']=p
                else:
                    compound=compounds[check['compound']];selection=selections[compound.get('id')]
                    groups=[g for g in compound.iter('ParameterAlternativeGroup') if g.get('name')==check['group']]
                    if len(groups)!=1:raise ValueError('NATIVE_ALTERNATIVE_GROUP_NOT_UNIQUE')
                    alternatives=list(groups[0].find('Children'))
                    if len(alternatives)!=1 or alternatives[0].get('name')!=check['alternative']:
                        raise ValueError('NATIVE_OMITTED_ALTERNATIVE_NOT_SOLE_AVAILABLE_CHOICE')
                    actual=[x for x in selection.findall('CompoundGroupSelections/CompoundGroupSelection') if x.get('groupName')==check['group']]
                    if len(actual)!=1 or actual[0].get('alternativeName')!=check['alternative']:
                        raise ValueError('NATIVE_SELECTED_ALTERNATIVE_DIFFERS')
                    proof.update(native_selection=dict(actual[0].attrib),
                        native_alternative_count=1,alternative_definition_sha256=__import__('hashlib').sha256(ET.canonicalize(ET.tostring(alternatives[0],encoding='unicode')).encode()).hexdigest())
                    # Non-calculated alternatives have explicit source values;
                    # their model parameter must retain that selected value.
                    if check['group']=='COMPOUND_LIPOPHILICITY':
                        p=parameters[check['compound']+'|Lipophilicity']
                        q=alternatives[0].find("Children/Para[@name='Lipophilicity']")
                        if q is None or Decimal(q.get('value'))!=Decimal(p['value']):raise ValueError('SELECTED_LIPOPHILICITY_MODEL_DIFFERS')
                        proof['native_model_parameter']=p
                report.append(proof)
    return report


def validate_overlay(overlay,output_dir,*,root,stage_runner,timeout_seconds,cpu_threads,stop_file):
    checks=omitted_fields(overlay);snapshot=overlay['snapshot'];all_sims=unique(snapshot['Simulations'],'Name')
    names=sorted(checks);refs=[];readbacks=[]
    output_dir.mkdir()
    for index in range(0,len(names),4):
        chosen=names[index:index+4];part=deepcopy(snapshot);part['Simulations']=[deepcopy(all_sims[n]) for n in chosen]
        inp=output_dir/('scope_%02d.json'%(index//4+1));migration.save(inp,part)
        project,stage_refs=stage_runner('snapshot_to_project',inp,output_dir/('build_%02d'%(index//4+1)),
            root=root,timeout_seconds=timeout_seconds,cpu_threads=cpu_threads,stop_file=stop_file)
        refs.extend([migration.reference(inp,root),*stage_refs])
        readbacks.extend(inspect_native_project(project,{n:checks[n] for n in chosen},root=root))
    proof={'accepted':True,'simulations_checked':len(names),'expected_simulations':names,
        'readbacks':readbacks,'ode_run':False,'numeric_tolerance':0,
        'other_fields_protected_by_complete_exact_snapshot_comparison':True,
        'official_export_behavior':'Native snapshots omit parameters unchanged from their building block and alternative selections where only one alternative exists',
        'provenance_refs':refs}
    migration.save(output_dir/'readback.json',proof)
    return proof,[*refs,migration.reference(output_dir/'readback.json',root)]
