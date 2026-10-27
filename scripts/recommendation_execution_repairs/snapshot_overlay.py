"""Restore explicitly declared simulation settings after a native upgrade.

The native-produced Version 80 document is the base. Only original Parameters
and compound Alternatives can be restored, with exact identity joins. Complete
strict equivalence must then pass; native readback remains a separate gate.
"""
from copy import deepcopy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import recommendation_candidate_migration as migration


def unique(rows,key):
    result={}
    for row in rows:
        identity=row[key]
        if identity in result:raise ValueError('AMBIGUOUS_SOURCE_IDENTITY:'+identity)
        result[identity]=row
    return result


def restore_explicit_simulation_fields(original,native_converted):
    if native_converted.get('Version')!=80:raise ValueError('NATIVE_VERSION_80_BASE_REQUIRED')
    source=migration.clean_snapshot(original);result=migration.clean_snapshot(native_converted)
    before=migration.compare_migration(source,result)
    originals=unique(source.get('Simulations',[]),'Name');converted=unique(result.get('Simulations',[]),'Name')
    if originals.keys()!=converted.keys():raise ValueError('SIMULATION_SET_CHANGED')
    changes=[];expectations=[]
    for name,old in originals.items():
        new=converted[name]
        if migration._normalize(old.get('Parameters'),'Parameters')!=migration._normalize(new.get('Parameters'),'Parameters'):
            native_before=deepcopy(new.get('Parameters'))
            parameters=deepcopy(old.get('Parameters',[]))
            unique(parameters,'Path')
            for parameter in parameters:
                path=parameter['Path']
                if path.startswith('Applications|'):parameter['Path']='Events|'+path.removeprefix('Applications|')
                expectations.append({'simulation_name':name,'native_path':parameter['Path'],
                    'expected_value':parameter.get('Value'),'expected_unit':parameter.get('Unit',''),
                    'original_record':deepcopy(parameter),'require_exact_native_unit_conversion':True})
            if 'Parameters' in old:new['Parameters']=parameters
            else:new.pop('Parameters',None)
            changes.append({'simulation_name':name,'field':'Parameters',
                            'source':deepcopy(old.get('Parameters')),'native_before':native_before})
        old_compounds=unique(old.get('Compounds',[]),'Name');new_compounds=unique(new.get('Compounds',[]),'Name')
        if old_compounds.keys()!=new_compounds.keys():raise ValueError('SIMULATION_COMPOUND_SET_CHANGED')
        for compound_name,selection in old_compounds.items():
            target=new_compounds[compound_name]
            if selection.get('Alternatives')!=target.get('Alternatives'):
                changes.append({'simulation_name':name,'compound_name':compound_name,'field':'Alternatives',
                                'source':deepcopy(selection.get('Alternatives')),'native_before':deepcopy(target.get('Alternatives'))})
                if 'Alternatives' in selection:target['Alternatives']=deepcopy(selection['Alternatives'])
                else:target.pop('Alternatives',None)
    after=migration.compare_migration(source,result)
    return {'status':'restored_pending_native_readback' if after['accepted'] else 'additional_source_changes_unresolved',
            'snapshot':result if after['accepted'] else None,'source_version':source.get('Version'),
            'native_base_version':native_converted.get('Version'),'changes':changes,'native_parameter_expectations':expectations,
            'comparison_before':before,'comparison_after':after,
            'version_only_rewrite':False,'native_execution_approved':False,
            'compound_parameters_fitted':False,'source_preservation_required':True,
            'remaining_gate':'Native parameter readback, selected alternative values/mechanisms and model execution; no acceptance from JSON identity alone'}
