"""Source-preserving legacy expression conversion with native equivalence gates."""
from copy import deepcopy
import math
from .snapshot_overlay import migration, unique

OFFICIAL_ROOT='artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/'
OFFICIAL_FILES=[OFFICIAL_ROOT+p for p in ('Snapshots/Services/LocalizationConverter.cs',
    'Snapshots/Mappers/ExpressionProfileMapper.cs','Snapshots/Mappers/ExpressionContainerMapper.cs',
    'Model/SchemaFactory.cs','Snapshots/Mappers/SchemaMapper.cs','Model/Individual.cs','CoreConstants.cs',
    'Snapshots/Services/MembraneLocationConverter.cs','Commands/NormalizeRelativeExpressionCommand.cs',
    'Model/TransportDirection.cs','Services/IndividualTransporterFactory.cs','Services/IndividualMoleculeFactory.cs')]

_SMALL={'Duodenum','UpperJejunum','LowerJejunum','UpperIleum','LowerIleum'}
_LARGE={'Caecum','ColonAscendens','ColonDescendens','ColonSigmoid','ColonTransversum','Rectum'}
_ORGANS={'Brain','Gonads','Kidney','Lung','SmallIntestine','Bone','Fat','Heart','Muscle','Skin','Spleen','Stomach','LargeIntestine','Pancreas'}


def _organ_path(region):
    if region in _SMALL:return 'Organism|SmallIntestine|Mucosa|'+region
    if region in _LARGE:return 'Organism|LargeIntestine|Mucosa|'+region
    if region in {'Pericentral','Periportal'}:return 'Organism|Liver|'+region
    if region in _ORGANS:return 'Organism|'+region
    raise ValueError('LEGACY_EXPRESSION_REGION_NOT_AUDITED:'+region)


def _transport_direction(kind,region,compartment):
    """TransportDirections.DefaultDirectionFor and IndividualTransporterFactory."""
    index={'Influx':0,'Efflux':1,'BiDirectional':2,'PgpLike':3}.get(kind)
    if index is None:raise ValueError('LEGACY_TRANSPORT_TYPE_NOT_AUDITED:'+str(kind))
    if region=='BloodCells':
        values=['InfluxPlasmaToBloodCells','EffluxBloodCellsToPlasma','BiDirectionalBloodCellsPlasma','EffluxBloodCellsToPlasma']
    elif region=='VascularEndothelium':
        values=['InfluxPlasmaToInterstitial','EffluxInterstitialToPlasma','BiDirectionalPlasmaInterstitial','EffluxInterstitialToPlasma']
    elif region=='Brain':
        if compartment=='Plasma':values=['InfluxBrainPlasmaToInterstitial','EffluxBrainInterstitialToPlasma','BiDirectionalBrainPlasmaInterstitial','PgpBrainInterstitialToPlasma']
        elif compartment=='Interstitial':values=['InfluxBrainInterstitialToTissue','EffluxBrainTissueToInterstitial','BiDirectionalBrainInterstitialTissue','PgpBrainTissueToInterstitial']
        else:raise ValueError('UNREGISTERED_BRAIN_TRANSPORT_COMPARTMENT')
    elif region in {'Kidney','Pericentral','Periportal'} and compartment=='Intracellular':
        return 'ExcretionKidney' if region=='Kidney' else 'ExcretionLiver'
    elif region in _SMALL|_LARGE and compartment=='Intracellular':
        values=['InfluxLumenToMucosaIntracellular','EffluxMucosaIntracellularToLumen','BiDirectionalLumenMucosaIntracellular','PgpMucosaIntracellularToLumen']
    else:
        _organ_path(region)
        if compartment not in {'Intracellular','Interstitial'}:raise ValueError('UNREGISTERED_TRANSPORT_COMPARTMENT')
        values=['InfluxInterstitialToIntracellular','EffluxIntracellularToInterstitial','BiDirectionalInterstitialIntracellular','PgpIntracellularToInterstitial']
    return values[index]


def _legacy_transporter(molecule,native_profile,changes,location):
    """Preserve old transporter inputs via the official v9->v10 semantics.

    Native export supplies only the constructed compartment identities. Every
    direction is independently derived, every numeric input comes from source.
    Whole native Model comparison remains mandatory for normalized expressions.
    """
    if molecule.get('Type')!='Transporter':return molecule
    old_expressions=molecule.get('Expression',[])
    legacy=any('Value' in e or 'MembraneLocation' in e for e in old_expressions)
    named=any('Name' in p and 'Path' not in p for p in molecule.get('Parameters',[]))
    if not legacy and not named:return molecule
    if not legacy:raise ValueError('LEGACY_TRANSPORTER_GLOBAL_PARAMETERS_WITHOUT_EXPRESSION_CONTEXT')
    if native_profile.get('Type')!='Transporter' or native_profile.get('TransportType')!=molecule.get('TransportType'):
        raise ValueError('NATIVE_TRANSPORTER_IDENTITY_OR_TYPE_DIFFERS')
    source_molecule=deepcopy(molecule)
    molecule=deepcopy(molecule);name=molecule['Name'];parameters=[]
    for old in molecule.get('Parameters',[]):
        p=deepcopy(old)
        if 'Name' not in p or 'Path' in p:raise ValueError('MIXED_LEGACY_TRANSPORTER_PARAMETER_IDENTITY')
        p['Path']=name+'|'+p.pop('Name');parameters.append(p)
    expressions=[];seen=set()
    for old in native_profile.get('Expression',[]):
        if set(old)-{'Name','CompartmentName','TransportDirection'}:raise ValueError('UNEXPECTED_NATIVE_TRANSPORT_EXPRESSION_FIELDS')
        key=(old['Name'],old.get('CompartmentName'))
        if key in seen:raise ValueError('DUPLICATE_NATIVE_TRANSPORT_COMPARTMENT')
        seen.add(key)
        expected=_transport_direction(molecule['TransportType'],*key)
        if old.get('TransportDirection')!=expected:raise ValueError('NATIVE_TRANSPORT_DIRECTION_DIFFERS_FROM_SOURCE_TYPE')
        expressions.append(deepcopy(old))
    if not expressions:raise ValueError('NATIVE_TRANSPORT_COMPARTMENTS_MISSING')
    values=[]
    for old in old_expressions:
        if set(old)-{'Name','MembraneLocation','Value','Unit','ValueOrigin'}:raise ValueError('UNREGISTERED_LEGACY_TRANSPORT_EXPRESSION_FIELDS')
        value=old.get('Value')
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('FINITE_NONNEGATIVE_LEGACY_EXPRESSION_REQUIRED')
        values.append(value)
    maximum=max(values,default=0)
    source_records=[];value_transforms=[]
    for old in old_expressions:
        region=old['Name'];source_records.append(deepcopy(old));p=deepcopy(old)
        p.pop('Name');membrane=p.pop('MembraneLocation','Basolateral')
        p['Value']=0 if maximum==0 else old['Value']/maximum
        if region in {'BloodCells','VascularEndothelium'}:
            p['Path']=name+'|Relative expression in '+('blood cells' if region=='BloodCells' else 'vascular endothelium')
        else:p['Path']=_organ_path(region)+'|Intracellular|'+name+'|Relative expression'
        value_transforms.append({'source_record':deepcopy(old),'target_path':p['Path'],
            'source_value':old['Value'],'formula':'0 if source maximum == 0 else source value / source maximum',
            'denominator':maximum,'target_value':p['Value'],'fitted':False})
        parameters.append(p)
        if region=='Brain':
            if membrane not in {'BloodBrainBarrier','Tissue'}:raise ValueError('UNREGISTERED_LEGACY_BRAIN_MEMBRANE')
            parameters.append({'Path':_organ_path(region)+'|Plasma|'+name+'|Fraction expressed at blood brain barrier',
                               'Value':1 if membrane=='BloodBrainBarrier' else 0})
        elif region in _SMALL|_LARGE|{'Kidney','Periportal','Pericentral'}:
            if membrane not in {'Apical','Basolateral'}:raise ValueError('UNREGISTERED_LEGACY_LUMEN_MEMBRANE')
            parameters.append({'Path':_organ_path(region)+'|Intracellular|'+name+'|Fraction expressed apical',
                               'Value':1 if membrane=='Apical' else 0})
        elif membrane!='Basolateral':raise ValueError('UNREGISTERED_LEGACY_TISSUE_MEMBRANE')
    unique(parameters,'Path')
    molecule['Parameters']=parameters;molecule['Expression']=expressions
    changes.append({'location':location,'molecule':name,'operation':'official_legacy_transporter_expression_and_membrane_conversion',
        'source_expression_records':source_records,'source_molecule_record':source_molecule,
        'normalization':'official NormalizeRelativeExpressionCommand: divide each expression by maximum; zero maximum remains zero',
        'normalization_denominator':maximum,'native_numeric_values_copied':False,'numeric_tolerance':0,
        'source_to_target_values':value_transforms,
        'mandatory_gate':'entire original-version and restored-version native Model/Settings equality plus explicit native readback'})
    return molecule


def _legacy_protein(molecule, changes, location):
    """Audited v7/v9 IndividualProtein schemas; no transporter inference."""
    if 'Expression' not in molecule or 'TissueLocation' not in molecule:return molecule
    molecule=deepcopy(molecule)
    if molecule.get('Type') not in {'Enzyme','OtherProtein'}:raise ValueError('LEGACY_EXPRESSION_TYPE_NOT_AUDITED')
    name=molecule['Name'];tissue=molecule.pop('TissueLocation');membrane=molecule.pop('MembraneLocation')
    endo=molecule.pop('IntracellularVascularEndoLocation')
    if endo!='Endosomal' or membrane!='Apical':
        raise ValueError('LEGACY_LOCALIZATION_COMBINATION_NOT_AUDITED')
    if tissue=='Intracellular':
        molecule['Localization']='Intracellular, BloodCellsIntracellular, VascEndosome'
    elif tissue=='ExtracellularMembrane' and molecule['Type']=='OtherProtein':
        molecule['Localization']='Interstitial, BloodCellsMembrane, VascMembranePlasmaSide'
    else:raise ValueError('LEGACY_LOCALIZATION_COMBINATION_NOT_AUDITED')
    parameters=[]
    for old in molecule.get('Parameters',[]):
        p=deepcopy(old);p['Path']=name+'|'+p.pop('Name');parameters.append(p)
    small={'Duodenum','UpperJejunum','LowerJejunum','UpperIleum','LowerIleum'}
    large={'ColonAscendens','ColonDescendens','ColonSigmoid','ColonTransversum'}
    organs={'Brain','Gonads','Kidney','Lung','SmallIntestine','Bone','Fat','Heart','Muscle','Skin','Spleen','Stomach','LargeIntestine','Pancreas'}
    for old in molecule.pop('Expression'):
        p=deepcopy(old);region=p.pop('Name')
        if region in {'BloodCells','Plasma'}:
            p['Path']=name+'|Relative expression in '+('blood cells' if region=='BloodCells' else 'plasma')
            parameters.append(p);continue
        if region in small:prefix='Organism|SmallIntestine|Mucosa|'+region
        elif region in large:prefix='Organism|LargeIntestine|Mucosa|'+region
        elif region in {'Pericentral','Periportal'}:prefix='Organism|Liver|'+region
        elif region in organs:prefix='Organism|'+region
        else:raise ValueError('LEGACY_EXPRESSION_REGION_NOT_AUDITED:'+region)
        p['Path']=prefix+'|Intracellular|'+name+'|Relative expression';parameters.append(p)
    molecule['Parameters']=parameters
    changes.append({'location':location,'molecule':name,'operation':'legacy_protein_localization_and_expression_paths',
        'source_localization':{'TissueLocation':tissue,'MembraneLocation':membrane,'IntracellularVascularEndoLocation':endo},
        'numeric_records_unchanged':True,'normalization_or_rescaling_performed':False})
    return molecule


def map_legacy_expression_schema(original, native_converted):
    """Convert explicit molecule inputs using registered official schema rules.

    Native exports supply profile and transport-compartment identities, never
    numerical values. Legacy transport expressions use the official maximum
    normalization; raw records and every arithmetic transformation are retained.
    Such conversions require complete strict native Model/Settings proof.
    """
    if native_converted.get('Version')!=80:raise ValueError('NATIVE_VERSION_80_BASE_REQUIRED')
    source=migration.clean_snapshot(original);expected=deepcopy(source)
    expected['Version']=80
    profiles=deepcopy(expected.get('ExpressionProfiles',[]));native_profiles={}
    for profile in native_converted.get('ExpressionProfiles',[]):
        key='|'.join(profile[k] for k in ('Molecule','Species','Category'))
        if key in native_profiles:raise ValueError('DUPLICATE_NATIVE_EXPRESSION_PROFILE')
        native_profiles[key]=profile
    changes=[];expectations=[]
    def individual(old,new_native,location):
        molecules=old.get('Molecules')
        if molecules is None:
            if 'ExpressionProfiles' not in old and new_native.get('ExpressionProfiles')==[]:
                old['ExpressionProfiles']=[]
                changes.append({'location':location,'operation':'explicit_empty_expression_profile_list'})
            return
        molecule_map=unique(molecules,'Name');params=deepcopy(old.get('Parameters',[]))
        unique(params,'Path');refs=new_native.get('ExpressionProfiles',[]);matched={}
        for ref in refs:
            if ref not in native_profiles:raise ValueError('NATIVE_EXPRESSION_PROFILE_REFERENCE_MISSING')
            p=native_profiles[ref];name=p['Molecule']
            if name not in molecule_map or name in matched:raise ValueError('EXPRESSION_MOLECULE_IDENTITY_AMBIGUOUS')
            if p['Species']!=old.get('OriginData',{}).get('Species'):raise ValueError('EXPRESSION_SPECIES_CHANGED')
            matched[name]=(ref,p)
        if matched.keys()!=molecule_map.keys():raise ValueError('EXPRESSION_MOLECULE_SET_CHANGED')
        assignments={name:[] for name in molecule_map};remaining=[]
        for parameter in params:
            names=set(parameter['Path'].split('|'))&molecule_map.keys()
            if len(names)>1:raise ValueError('EXPRESSION_PARAMETER_SCOPE_AMBIGUOUS')
            if names:
                name=next(iter(names));assignments[name].append(parameter)
                expectations.append({'individual_location':location,'molecule':name,'parameter':deepcopy(parameter)})
            else:remaining.append(parameter)
        new_refs=[]
        for molecule in molecules:
            molecule=_legacy_transporter(molecule,matched[molecule['Name']][1],changes,location)
            molecule=_legacy_protein(molecule,changes,location)
            name=molecule['Name'];ref,native_profile=matched[name]
            profile={k:deepcopy(v) for k,v in molecule.items() if k!='Name'}
            profile.update(Molecule=name,Species=native_profile['Species'],Category=native_profile['Category'])
            if assignments[name]:profile['Parameters']=profile.get('Parameters',[])+assignments[name]
            profiles.append(profile);new_refs.append(ref)
        old.pop('Molecules')
        old['ExpressionProfiles']=new_refs
        if remaining:old['Parameters']=remaining
        else:old.pop('Parameters',None)
        changes.append({'location':location,'operation':'relocate_legacy_molecules_to_expression_profiles',
                        'molecule_count':len(molecules),'parameter_count':len(params)-len(remaining),
                        'source_molecule_records_retained_in_source_or_conversion_provenance':True,
                        'relocated_individual_parameter_records_copied_exactly':True})
    native_individuals=unique(native_converted.get('Individuals',[]),'Name')
    for old in expected.get('Individuals',[]):
        individual(old,native_individuals[old['Name']],'Individuals/'+old['Name'])
    native_populations=unique(native_converted.get('Populations',[]),'Name')
    for population in expected.get('Populations',[]):
        old=population.get('Settings',{}).get('Individual')
        if old is not None:
            new=native_populations[population['Name']].get('Settings',{}).get('Individual')
            if new is None:raise ValueError('POPULATION_REFERENCE_INDIVIDUAL_REMOVED')
            individual(old,new,'Populations/'+population['Name']+'/Settings/Individual')
    if profiles:
        merged={}
        for profile in profiles:
            key='|'.join(profile[k] for k in ('Molecule','Species','Category'))
            if key in merged and migration._normalize(merged[key])!=migration._normalize(profile):
                raise ValueError('MAPPED_PROFILE_IDENTITY_COLLISION:'+key)
            if key not in merged:merged[key]=profile
            else:changes.append({'profile':key,'operation':'share_exactly_identical_declared_profile'})
        expected['ExpressionProfiles']=list(merged.values())
    # Older snapshots could omit the explicit GFR enum while naming the exact
    # source-defined GlomerularFiltration process. Derive only from that typed
    # definition and its exact data-source-qualified name, never a name guess.
    compounds=unique(expected.get('Compounds',[]),'Name')
    for simulation in expected.get('Simulations',[]):
        for selection in simulation.get('Compounds',[]):
            for process in selection.get('Processes',[]):
                if 'SystemicProcessType' in process or not process.get('Name'):continue
                matches=[p for p in compounds[selection['Name']].get('Processes',[]) if
                         p.get('InternalName')=='GlomerularFiltration' and p.get('DataSource') and
                         process['Name']=='Glomerular Filtration-'+p['DataSource']]
                if len(matches)==1:
                    process['SystemicProcessType']='GFR'
                    changes.append({'simulation':simulation['Name'],'compound':selection['Name'],
                        'operation':'make_explicit_source_defined_glomerular_filtration_type',
                        'source_process':deepcopy(matches[0])})
    # SchemaFactory defines these explicit defaults (0, 1, 0). Only absent
    # values are filled, and their exact native record must match the defaults.
    native_protocols=unique(native_converted.get('Protocols',[]),'Name')
    for protocol in expected.get('Protocols',[]):
        ns=unique(native_protocols[protocol['Name']].get('Schemas',[]),'Name')
        for schema in protocol.get('Schemas',[]):
            current=unique(schema.get('Parameters',[]),'Name')
            for parameter in ns[schema['Name']].get('Parameters',[]):
                name=parameter['Name']
                if name in current:continue
                defaults={'Start time':0,'NumberOfRepetitions':1,'TimeBetweenRepetitions':0}
                if name not in defaults or parameter.get('Value')!=defaults[name]:continue
                allowed={'Name','Value'}|({'Unit'} if name!='NumberOfRepetitions' else set())
                if set(parameter)!=allowed:raise ValueError('UNEXPECTED_NATIVE_DEFAULT_METADATA')
                if name!='NumberOfRepetitions' and parameter['Unit']!=protocol.get('TimeUnit','min'):
                    raise ValueError('SCHEMA_DEFAULT_TIME_UNIT_DIFFERS')
                schema.setdefault('Parameters',[]).append(deepcopy(parameter))
                changes.append({'operation':'explicit_official_schema_factory_default','protocol':protocol['Name'],
                    'schema':schema['Name'],'parameter':deepcopy(parameter)})
    return {'mapped_snapshot':expected,'changes':changes,'native_parameter_expectations':expectations,
            'requires_complete_native_model_proof':any(c['operation']=='official_legacy_transporter_expression_and_membrane_conversion' for c in changes),
            'comparison':migration.compare_migration(expected,native_converted),
            'original_source_retained':True,'native_version_evidence_required':True}
