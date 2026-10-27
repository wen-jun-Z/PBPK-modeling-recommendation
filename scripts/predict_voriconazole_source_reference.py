#!/usr/bin/env python3
"""Frozen Xia2020-source reference predictions for independent Li2023 IV curves.

No fitting, model choice, individual covariate inference or sealed outcome I/O.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from uuid import NAMESPACE_URL,uuid5
import bisect,csv,hashlib,json,math,os,sys
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runtime/blueprint_newdrugs_stage_20260920'
sys.path.insert(0,str(STAGE/'src'))
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.experiment_runtime import file_sha,save,digest
from pbpk_kb.blueprints.external_source_isolation import audit_design_inputs

PACKAGE=ROOT/'experiments/blueprint_design/development_inputs/newdrugs_native_20260920/voriconazole/input_package.json'
PROTOCOLS=ROOT/'experiments/blueprint_design/external_validation/concentration_curve_audit_20260920/protocols'
OUTPUT=ROOT/'experiments/blueprint_design/external_validation/voriconazole_li2023_source_reference_20260920'
MODEL_KEY='published_source_reference_Xia2020'
MAX_BYTES=512*1024**2


def local(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT) or path==ROOT:raise ValueError('Project-local paths required')
    return path


def load(path,frozen):
    path=local(path)
    if path.stat().st_size>16*1024**2:raise ValueError('Unexpected metadata size')
    frozen[str(path.relative_to(ROOT))]=file_sha(path)
    return json.loads(path.read_text())


def checked_reference(path,where):
    path=local(path);checksum=file_sha(path)
    return {'provenance_id':str(uuid5(NAMESPACE_URL,str(path)+where+checksum)),
        'provenance_kind':'user_request','source_id':str(path.relative_to(ROOT)),
        'source_location':where,'source_checksum':checksum}


def verify(frozen):
    for name,checksum in frozen.items():
        if file_sha(local(ROOT/name))!=checksum:raise ValueError('Frozen source/config changed: '+name)


def bytes_written():
    result=0
    for path in OUTPUT.rglob('*'):
        if path.is_symlink():raise ValueError('No output symlinks permitted')
        if path.is_file():result+=path.stat().st_size
    return result


def original_grid_values(times,values,targets):
    if len(times)!=len(values) or len(times)<2 or any(not math.isfinite(v) for v in times+values):raise ValueError('Nonfinite native curve')
    if any(b<=a for a,b in zip(times,times[1:])) or any(v<0 for v in values) or max(values)<=0:raise ValueError('Negative/empty/invalid native curve; never clip')
    if targets[0]<times[0] or targets[-1]>times[-1]:raise ValueError('No extrapolation beyond native curve')
    sampled=[]
    for target in targets:
        index=bisect.bisect_left(times,target)
        if index<len(times) and times[index]==target:sampled.append(values[index]);continue
        if index==0 or index==len(times):raise ValueError('Target grid is uncovered')
        fraction=(target-times[index-1])/(times[index]-times[index-1])
        sampled.append(values[index-1]+fraction*(values[index]-values[index-1]))
    return sampled


def make_request(template,design,path,sex):
    request=deepcopy(template);instance=request['instance'];pop=design['population'];dose=design['dose']
    request['studies']=[];request['requirement'].update(route='intravenous',dose=float(dose['value']),dose_unit='mg/kg',formulation=None)
    provenance=checked_reference(path,'Curated protocol demographics and dosing only; no concentrations or endpoints')
    instance['human_source']=provenance;instance['protocol_source']=provenance
    individual=instance['individual'];individual['Name']='IND_vori_li2023_reference_'+str(dose['value'])+'mgkg_'+sex
    individual['OriginData'].update(Gender=sex.upper(),Age={'Value':pop['age_mean_years'],'Unit':'year(s)'},
        Weight={'Value':pop['weight_mean_kg'],'Unit':'kg'},Height={'Value':pop['height_mean_cm'],'Unit':'cm'})
    # Preserve source population category, expression profiles, seed and drug
    # kinetics. Nationality does not supply CYP2C19 genotype or an expression fit.
    instance['generated_demographics']=[]
    instance['protocol']={'Name':'PROTO_vori_li2023_'+str(dose['value'])+'mgkg','ApplicationType':'Intravenous','DosingInterval':'Single',
        'Parameters':[{'Name':'Start time','Value':0.,'Unit':'h'},
                      {'Name':'InputDose','Value':float(dose['value']),'Unit':'mg/kg'},
                      {'Name':'Infusion time','Value':float(dose['infusion_duration_min']),'Unit':'min'}]}
    instance['simulation_name']='vori_Xia2020_reference_'+str(dose['value'])+'mgkg_'+sex
    instance['output_schema']=[{'Parameters':[{'Name':'Start time','Value':a,'Unit':'h'},
        {'Name':'End time','Value':b,'Unit':'h'},{'Name':'Resolution','Value':r,'Unit':'pts/h'}]} for a,b,r in [(0.,4.,600.),(4.,48.,12.)]]
    assert instance['compound']==template['instance']['compound']
    assert instance['expression_profiles']==template['instance']['expression_profiles']
    assert individual['Seed']==template['instance']['individual']['Seed']
    return BlueprintDesignRequest.model_validate(request).model_dump(mode='json')


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if OUTPUT.exists():raise FileExistsError('Immutable prediction directory already exists')
    frozen={};case=load(PACKAGE,frozen)['case']
    if (case['purpose']!='fixed_source_reference_engineering' or case['parameter_search']['fit_permitted'] is not False
        or case['parameter_search']['selection_permitted'] is not False or case['parameter_search']['parameters']):raise ValueError('A fixed-source-only reference is required')
    template=case['design_request']
    if template['instance']['compound']['CalculationMethods'][0]!='Cellular partition coefficient method - Poulin and Theil':raise ValueError('Preserve predeclared source constitutive method')
    for item in case['source_audit']:
        target=local(ROOT/item['path'])
        if file_sha(target)!=item['sha256']:raise ValueError('Source model changed')
        frozen[item['path']]=item['sha256']
    for path in (STAGE/'src/pbpk_kb/blueprints').glob('*.py'):frozen[str(path.relative_to(ROOT))]=file_sha(path)
    runner=ROOT/'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R'
    frozen[str(runner.relative_to(ROOT))]=file_sha(runner)
    frozen[str(Path(__file__).relative_to(ROOT))]=file_sha(Path(__file__))
    designs={};design_paths={}
    for dose in (4,6):
        path=PROTOCOLS/f'voriconazole_li2023_reference_iv_{dose}mgkg_curves.json';design=load(path,frozen)
        if (design['outcomes_exposed_to_designer'] is not False or design['dose']['value']!=dose or
            design['dose']['unit']!='mg/kg' or design['dose']['route']!='IV' or
            design['dose']['administrations_h']!=[0] or design['dose']['infusion_duration_min']!={4:80,6:120}[dose] or
            design['population']['n']!=24 or design['population']['male_n']+design['population']['female_n']!=24 or
            design['demographic_linkage']['individual_covariates_available'] is not False or
            design['registry_id']!='NCT05330000' or design['publication_cluster_id']!='PMC9985189'):
            raise ValueError('Unexpected clinical protocol contract')
        if dose==4 and design['sampling']['time_h'][3]!=1.33:raise ValueError('Keep the published 1.33h nominal sample unchanged')
        designs[dose]=design;design_paths[dose]=path
    policy={'schema_version':'pbpk-external-source-isolation-v1',
        'protocols':[{'case_id':d['case_id'],'study_id':d['study_id']} for d in designs.values()],
        'denied_paths':[d['sealed_observation_reference']['path'].lower() for d in designs.values()],
        'denied_hashes':[d['sealed_observation_reference']['sha256'] for d in designs.values()],
        'denied_identifiers':[{'kind':k,'value':v.lower()} for k,v in designs[4]['source_identifiers'].items()]+[{'kind':'trial','value':'nct05330000'}],
        'curated_metadata_sources':{str(p.relative_to(ROOT)):file_sha(p) for p in design_paths.values()},
        'scope':'Exclude whole external publication/trial from supplied model inputs; no sealed file opened even for hashing'}
    requests={};snapshots={};protocols={}
    mw=next(p['Value'] for p in template['instance']['compound']['Parameters'] if p['Name']=='Molecular weight')
    for dose,design in designs.items():
        pop=design['population']
        protocols[dose]={'case_id':design['case_id'],'design_sha256':file_sha(design_paths[dose]),'model_key':MODEL_KEY,
            'molecular_weight_g_mol':mw,'sex_weights':{'male':pop['male_n']/24,'female':pop['female_n']/24},
            'prediction_aggregation':'sex_weighted_reference_concentration',
            'sampling_time_h':design['sampling']['time_h'],'sampling_time_min':[t*60. for t in design['sampling']['time_h']],
            'sampling_mapping':'Original nominal time_h times60 without rounding or shifting; linear interpolation of native output only',
            'native_output_grid':{'0..4h_points_per_hour':600,'4..48h_points_per_hour':12},
            'dose_mg_kg':dose,'infusion_duration_min':design['dose']['infusion_duration_min'],
            'model_identity':'Fixed published-source reference reconstructed from Xia2020 DDI asset with no coadministered drugs',
            'not_a_completed_M0_M6_comparison':True,'fit_performed':False,'model_selection_performed':False,
            'approximation':{'reference_subjects_only':True,'individual_covariates_available':False,'personalized_prediction':False,
                'demographics':'Both sex reference subjects share published cohort mean age/weight/height; individual covariates and actual doses unavailable',
                'population_category':'European_ICRP_2002 inherited source; Chinese ancestry not modeled',
                'enzyme_profiles':'Source legacy protein expression profiles retained; no CYP2C19 genotype inferred from nationality',
                'fixed_source_parameters':'Source historical parameters and Poulin-Theil partition retained; no outcome-driven tuning',
                'crossover':'Predict reference treatment only; all doses/treatments/periods excluded from fitting',
                'subject_mapping':'Each cohort subject is compared to the same prespecified sex-weighted reference curve; no invented subject sex assignment'},
            'statistical_unit':'Two dose cohorts in one study/drug; never count individual subjects/doses as independent studies',
            'observation_policy':'Preserve censoring/missingness in separate scorer; do not replace BQL by invented positive observations',
            'prediction_validity':'Finite nonnegative native curves required; no concentration clipping or imputation',
            'external_observations_read':False}
        for sex in ('male','female'):
            request=make_request(template,design,design_paths[dose],sex)
            audit_design_inputs({'request':request},policy,allowed_metadata_sources=list(policy['curated_metadata_sources']))
            built=design_blueprint(BlueprintDesignRequest.model_validate(request),project_root=ROOT)
            if not built['human_model_binding'].get('instance'):raise RuntimeError(str(built['human_model_binding'].get('gaps')))
            key=f'iv_{dose}mgkg_{sex}';requests[key]=request;snapshots[key]=built['human_model_binding']['instance']['snapshot']
    verify(frozen);OUTPUT.mkdir(parents=True)
    protocol_refs={}
    for dose,protocol in protocols.items():
        path=OUTPUT/f'iv_{dose}mgkg'/'protocol.json';save(path,protocol)
        protocol_refs[dose]={'path':str(path.relative_to(ROOT)),'sha256':file_sha(path)}
    config_path=OUTPUT/'config_freeze.json'
    save(config_path,{'schema_version':'pbpk-fixed-source-reference-freeze-v1','model_key':MODEL_KEY,
        'frozen_before_native_prediction_at':datetime.now(timezone.utc).isoformat(),'requests':requests,
        'snapshot_sha256':{k:digest(v) for k,v in snapshots.items()},'protocols':protocol_refs,'frozen_input_sha256':frozen,
        'evidence_exclusion':policy,'native_runs_registered':4,'cpu_threads':2,'native_timeout_seconds':240,
        'maximum_total_output_bytes':MAX_BYTES,'external_observations_read':False,'fit_permitted':False,'model_selection_permitted':False})
    frozen[str(config_path.relative_to(ROOT))]=file_sha(config_path)
    for reference in protocol_refs.values():frozen[reference['path']]=reference['sha256']
    all_models={4:[],6:[]}
    for dose in (4,6):
        for sex in ('male','female'):
            verify(frozen)
            remaining=MAX_BYTES-bytes_written()-8*1024**2
            quota=min(128*1024**2,remaining)
            if quota<16*1024**2:raise RuntimeError('Insufficient disk quota; no more native runs')
            key=f'iv_{dose}mgkg_{sex}';directory=OUTPUT/'native'/key
            result=execute_snapshot(snapshot=snapshots[key],output_dir=directory,project_root=ROOT,
                timeout_seconds=240,cpu_threads=2,max_output_bytes=quota)
            if not result['engineering_accepted'] or len(result['curve_outputs'])!=1:raise RuntimeError('Native reference prediction failed: '+key)
            path=local(directory/result['curve_outputs'][0]['path'])
            if not path.is_relative_to(directory):raise ValueError('Native curve leaves registered directory')
            with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
            column='Organism|PeripheralVenousBlood|Voriconazole|Plasma (Peripheral Venous Blood) [µmol/l]'
            times=[float(r['Time [min]']) for r in rows];values=[float(r[column]) for r in rows]
            target=protocols[dose]['sampling_time_min'];sampled=original_grid_values(times,values,target)
            all_models[dose].append({'model_key':MODEL_KEY,'sex':sex.upper(),'time_min':target,
                'concentration':sampled,'unit':'µmol/l','time_unit':'min',
                'native_csv_path':str(path.relative_to(ROOT)),'native_csv_sha256':file_sha(path),
                'native_points':len(times),'individual_prediction':False})
            print(json.dumps({'status':'native_reference_done','dose_mg_kg':dose,'sex':sex,'native_points':len(times),'nominal_points':len(target)}),flush=True)
    verify(frozen)
    predictions=[]
    for dose,models in all_models.items():
        path=OUTPUT/f'iv_{dose}mgkg'/'predictions.json';design=designs[dose];protocol=protocol_refs[dose]
        save(path,{'case_id':design['case_id'],'models':models,'contains_observed_pk_values':False,'protocol_sha256':protocol['sha256']})
        predictions.append({'case_id':design['case_id'],'path':str(path.relative_to(ROOT)),'sha256':file_sha(path),
            'design_path':str(design_paths[dose].relative_to(ROOT)),'design_sha256':file_sha(design_paths[dose]),
            'protocol_path':protocol['path'],'protocol_sha256':protocol['sha256']})
    verify(frozen)
    if bytes_written()>MAX_BYTES-4*1024**2:raise RuntimeError('Prediction output quota exceeded')
    save(OUTPUT/'prediction_lock.json',{'schema_version':'pbpk-external-predictions-lock-v1','model_selection_frozen':True,
        'model_key':MODEL_KEY,'locked_at_utc':datetime.now(timezone.utc).isoformat(),
        'config_freeze':{'path':str(config_path.relative_to(ROOT)),'sha256':file_sha(config_path)},'predictions':predictions,
        'external_observations_read':False,'physical_native_predictions':4,'independent_study_count':1,
        'source_reference_only':True,'main_method_jobs_completed':0})
    print(json.dumps({'status':'predictions_locked','output':str(OUTPUT.relative_to(ROOT)),'native_runs':4,'dose_cohorts':2,'study_count':1}),flush=True)

if __name__=='__main__':main()
