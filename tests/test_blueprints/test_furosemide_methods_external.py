"""Only synthetic observations; frozen MM/FO contracts and all 57 rows survive."""
import csv
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('new_furo_wrapper', ROOT/'scripts/score_furosemide_methods_endpoints.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
predictor = wrapper.PREDICTOR


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def source_specs():
    return [{'parameter_id':name, 'initial_value':1., 'lower_bound':.1, 'upper_bound':10.,
        'unit':'dimensionless', 'search_scale':'log', 'sensitivity_scale':1.,
        'runtime':{'path':'kcat' if name == 'capacity' else 'Lipophilicity'}}
        for name in ('lipophilicity', 'capacity')]


def spaces():
    specs = source_specs()
    derived = deepcopy(specs)
    derived[1]['runtime'] = {'path':'CLspec/[Enzyme]', 'multiplier':.5, 'native_unit':'l/µmol/min'}
    contracts = {'mm':specs, 'fo':derived}
    rows = []
    for action, contract in contracts.items():
        for flags in ((0,0),(0,1),(1,0),(1,1)):
            free = [s['parameter_id'] for s,flag in zip(specs,flags) if flag]
            rows.append({'candidate_id':action+'__'+''.join(map(str,flags)), 'structure_id':action,
                'action':action, 'parameter_ids':[s['parameter_id'] for s in specs],
                'free_parameter_ids':free, 'parameterization_hash':predictor.digest({'specs':contract,'free':free}),
                'structure_semantic_hash':'semantic_'+action})
    return specs, contracts, rows


@pytest.fixture
def suite(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, 'ROOT', tmp_path)
    monkeypatch.setattr(wrapper, 'ROOT', tmp_path)
    monkeypatch.setattr(wrapper.BASE, 'ROOT', tmp_path)
    specs, contracts, candidates = spaces()
    monkeypatch.setattr(predictor, 'candidate_contracts', lambda case:(deepcopy(candidates),deepcopy(contracts)))
    case = {'case_id':'synthetic_case', 'compound_id':'furosemide',
        'parameter_search':{'parameters':specs}, 'mechanism_search':{'expected_structure_ids':['mm','fo']}}
    package = write(tmp_path/'package.json', {'case':case})
    out = tmp_path/'suite'
    tasks = []
    for block,(budget,dimension,methods) in predictor.BLOCKS.items():
        jobs = [{'method':m,'seed':s} for m,s in sorted(predictor.expected_jobs(methods))]
        task = {'task_id':block+'__synthetic_case','case_id':case['case_id'],'compound_id':'furosemide',
            'block_id':block,'budget':budget,'matched_count':dimension,
            'bundle_path':'package.json','bundle_sha256':predictor.file_sha(package),'jobs':jobs}
        tasks.append(task)
        directory = out/'tasks'/task['task_id']
        case_path = write(directory/'case.json',case)
        write(directory/'configuration.json',{'case_sha256':predictor.file_sha(case_path),
            'case_bundle_sha256':predictor.file_sha(package),'case_bundle_path':'package.json',
            'compound_id':'furosemide','max_solver_calls_per_method':budget,
            'exact_free_parameter_count':dimension,'jobs':jobs,'external_observations_available_to_optimizer':False,
            'independent_test_loaded':False,'code_sha256':{},'candidates':deepcopy(candidates)})
        rows = []
        for job in jobs:
            # FO free subset retains shared optimizer coordinates but a different
            # native path; full MM and one-dimensional matched models coexist.
            action = 'fo' if job['method'] == 'M6' else 'mm'
            free = ['capacity'] if dimension == 1 else ['lipophilicity','capacity']
            candidate = next(c for c in candidates if c['action'] == action and c['free_parameter_ids'] == free)
            rows.append({**job,'status':'budget_exhausted' if budget == 500 else 'completed',
                'prediction_available':True,'selected_rebuild_verified':True,
                'selected_candidate_id':candidate['candidate_id'],'effective_free_parameter_ids':free})
            write(directory/'methods'/f"{job['method']}_seed_{job['seed']}"/'selected_parameters.json',{
                'parameter_ids':[s['parameter_id'] for s in specs], 'calibration_source':case['case_id'],
                'scientific_accepted':False,'values':[1.,2.]})
        write(directory/'results.json',{'methods':rows,'completed_jobs':len(rows)})
    write(out/'configuration.json',{'version':'blueprint-reviewed-development-suite-v1','tasks':tasks,'code_sha256':{}})
    write(out/'completion.json',{'status':'finished','halt_reason':None})
    return tmp_path,out,case,package,contracts,candidates


def test_actual_derived_fo_contract_accepts_all_registered_blocks(suite):
    _,out,case,package,_,_=suite
    rows,models=predictor.load_selections(out,case,package,{})
    assert len(rows)==57 and all(r['prediction_available'] for r in rows)
    assert len(models)==2
    fo=next(v for v in models.values() if v['action']=='fo')
    assert fo['native_parameter_contract_sha256']!=fo['parameter_contract_sha256']
    assert fo['native_parameter_specs'][1]['runtime']['path']=='CLspec/[Enzyme]'
    assert {r['solver_budget'] for r in rows}=={500,2000,5000}
    assert all(len(r['effective_free_parameter_ids'])==1 for r in rows if r['block_id']=='matched_1')


@pytest.mark.parametrize('alter', ['hash','mapping','missing','order'])
def test_entire_space_tamper_is_rejected(suite,alter):
    _,out,case,package,_,_=suite
    path=out/'tasks'/'main_5000__synthetic_case'/'configuration.json'
    config=json.loads(path.read_text())
    if alter=='hash':config['candidates'][4]['parameterization_hash']='0'*64
    elif alter=='mapping':config['candidates'][4]['action']='mm'
    elif alter=='missing':config['candidates'].pop()
    else:config['candidates'].reverse()
    write(path,config)
    with pytest.raises(ValueError,match='Complete frozen candidate space'):
        predictor.load_selections(out,case,package,{})


@pytest.mark.parametrize('effective', [[],['capacity','lipophilicity'],['capacity','capacity'],None])
def test_matched_effective_dimension_cannot_drift(suite,effective):
    _,out,case,package,_,_=suite
    path=out/'tasks'/'matched_1__synthetic_case'/'results.json'
    result=json.loads(path.read_text());result['methods'][0]['effective_free_parameter_ids']=effective
    write(path,result)
    with pytest.raises(ValueError,match='contract/dimension'):
        predictor.load_selections(out,case,package,{})


def test_original_hash_cannot_impersonate_converted_fo_contract():
    specs,contracts,candidates=spaces()
    candidate=deepcopy(next(c for c in candidates if c['candidate_id']=='fo__11'))
    candidate['parameterization_hash']=predictor.digest({'specs':specs,'free':candidate['free_parameter_ids']})
    with pytest.raises(ValueError,match='contract/dimension'):
        predictor.validate_selection_coordinates(candidate,candidate['free_parameter_ids'],specs,contracts['fo'],None)


def test_candidate_cannot_change_coordinate_bounds():
    specs,contracts,candidates=spaces()
    contract=deepcopy(contracts['fo']);contract[1]['upper_bound']=20.
    with pytest.raises(ValueError,match='optimization coordinates'):
        predictor.validate_selection_coordinates(candidates[-1],['lipophilicity','capacity'],specs,contract,None)


def test_unestimated_source_coordinate_cannot_change(suite):
    _,out,case,package,_,_=suite
    path=out/'tasks'/'matched_1__synthetic_case'/'methods'/'M5_seed_17'/'selected_parameters.json'
    values=json.loads(path.read_text());values['values'][0]=1.5;write(path,values)
    with pytest.raises(ValueError,match='unestimated parameter'):
        predictor.load_selections(out,case,package,{})


@pytest.fixture
def frozen_predictions(suite,monkeypatch):
    root,out,case,package,_,_=suite
    monkeypatch.setattr(predictor,'freeze_prediction_sources',lambda frozen:None)
    # Synthetic-only sealed endpoints are readable in these contract tests.
    sealed=write(root/'synthetic_sealed.json',{'case_id':'external_synthetic','study_id':'synthetic_study',
        'observed_endpoints':[{'endpoint':name,'unit':unit,'aggregation':'arithmetic_mean','value':100.,
            'source_table_id':'synthetic','source_row':1,'source_column':1}
            for name,unit in [('Cmax','ng/mL'),('AUC_infinity','ng*h/mL')]]})
    design={'case_id':'external_synthetic','study_id':'synthetic_study','cohort_id':'synthetic_cohort',
        'compound':'furosemide','scientific_results_accessed_by_algorithm':False,
        'source_article_sha256':'0'*64,
        'sealed_observation_reference':{'path':sealed.name,'sha256':predictor.file_sha(sealed)},
        'evidence_exclusion':{'exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training':True,
            'raw_source_files':[],'identifiers':{'doi':'synthetic-doi'},'trial_registry_ids':[]},
        'protocol':{'route':'oral','dose':40,'dose_unit':'mg','administrations':[{'time_min':0,'dose':40,'dose_unit':'mg'}],
            'reported_demographics':{'age_year':{'mean':39},'body_weight_kg':{'mean':77},
                'sex_counts':{'male':31,'female':10},'n_enrolled':41}}}
    design_path=write(root/'design.json',design)
    frozen={};predictor.read(package,frozen);predictor.read(design_path,frozen)
    selections,models=predictor.load_selections(out,case,package,frozen)
    pred_dir=root/'predictions';pred_dir.mkdir()
    tail={'fit_window_start_h':134.4,'fit_window_end_h':168.,'minimum_points':5,
        'require_negative_log_slope':True,'maximum_tail_fraction_for_primary':.01,
        'minimum_r_squared_for_primary':.99,'do_not_clip_or_replace_failed_tail':True}
    protocol={'case_id':design['case_id'],'design_sha256':predictor.file_sha(design_path),
        'sex_weights':{'male':31/41,'female':10/41},'molecular_weight_g_mol':100.,
        'approximation':{'reference_subject':True},'tail_policy':tail,'selections':selections,'models':models,
        'input_package_path':package.name,'suite_path':'suite','input_package_sha256':predictor.file_sha(package),
        'frozen_input_sha256':frozen,'strict_main_eligible':False,'qualification_tier':'transportability_sensitivity',
        'qualification_reasons':['Synthetic reference subjects'],'external_observations_read':False,
        'evidence_exclusion':predictor.metadata_exclusion_policy(design,design_path)}
    protocol_path=write(pred_dir/'prediction_protocol.json',protocol)
    curves=[]
    for key in models:
        for sex in ('male','female'):
            times=list(range(0,168*60+1,60));concentration=[math.exp(-t/600) for t in times]
            csvpath=pred_dir/'native'/(key+'_'+sex)/'curve.csv';csvpath.parent.mkdir(parents=True)
            with csvpath.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['Time [min]',
                    'Organism|PeripheralVenousBlood|Furosemide|Plasma (Peripheral Venous Blood) [µmol/l]'])
                writer.writerows(zip(times,concentration))
            curves.append({'model_key':key,'sex':sex,'time_min':times,'concentration':concentration,
                'unit':'µmol/l','time_unit':'min','native_csv_path':str(csvpath.relative_to(root)),
                'native_csv_sha256':predictor.file_sha(csvpath)})
    prediction=write(pred_dir/'predictions.json',{'case_id':design['case_id'],'models':curves,'selections':selections,
        'protocol_sha256':predictor.file_sha(protocol_path)})
    lock=write(pred_dir/'prediction_lock.json',{'schema_version':wrapper.BASE.SCHEMA,'model_selection_frozen':True,
        'design_sha256':predictor.file_sha(design_path),'protocol':{'path':str(protocol_path.relative_to(root)),
        'sha256':predictor.file_sha(protocol_path)},'predictions':[{'case_id':design['case_id'],
        'path':str(prediction.relative_to(root)),'sha256':predictor.file_sha(prediction)}],
        'method_job_count':57,'physical_native_predictions':len(curves),'strict_main_eligible':False,
        'qualification_tier':'transportability_sensitivity'})
    return root,lock,design_path,sealed


def test_full_synthetic_prediction_contract_unseals_only_after_57_validation(frozen_predictions):
    root,lock,design,sealed=frozen_predictions
    checked=wrapper.validate_locked_inputs(lock,design)
    assert len(checked['selections'])==57
    receipt=wrapper.score_and_seal(lock,design,root/'evaluation')
    assert receipt['status']=='scored_and_sealed' and receipt['method_rows_scored']==57
    assert receipt['endpoint_counts']['Cmax']['evaluable']==57
    assert receipt['independent_drugs']==receipt['independent_studies']==1
    assert receipt['metrics_exposed_to_method_development'] is False
    assert 'fold_error' not in json.dumps(receipt)
    assert (root/'evaluation'/'sealed'/'endpoint_scores.json').is_file()


def test_existing_output_rejected_before_validation_or_unsealing(frozen_predictions,monkeypatch):
    root,lock,design,sealed=frozen_predictions
    target=root/'evaluation';target.mkdir();sealed.unlink()
    monkeypatch.setattr(wrapper,'validate_locked_inputs',lambda *_:pytest.fail('Must reject output first'))
    with pytest.raises(FileExistsError,match='before opening'):
        wrapper.score_and_seal(lock,design,target)


def test_upstream_tamper_rejected_before_any_sealed_read(frozen_predictions):
    root,lock,design,sealed=frozen_predictions;sealed.unlink()
    path=root/'suite/tasks/main_5000__synthetic_case/methods/M6_seed_17/selected_parameters.json'
    params=json.loads(path.read_text());params['values'][1]=3;write(path,params)
    with pytest.raises(ValueError,match='Frozen selections'):
        wrapper.score_and_seal(lock,design,root/'evaluation')


def test_native_csv_tamper_rejected_before_sealed_read(frozen_predictions):
    root,lock,design,sealed=frozen_predictions;sealed.unlink()
    csvpath=next((root/'predictions/native').glob('*/*.csv'));csvpath.write_text('changed')
    with pytest.raises(ValueError,match='CSV checksum'):
        wrapper.score_and_seal(lock,design,root/'evaluation')


def test_all_method_identities_required_even_when_count_is_57(suite):
    _,out,case,package,_,_=suite
    rows,_=predictor.load_selections(out,case,package,{})
    rows[-1]=deepcopy(rows[0])
    with pytest.raises(ValueError,match='duplicate'):
        wrapper.validate_job_denominator(rows)


def test_native_failure_never_returns_partial_prediction_lock():
    from threading import Event
    class Budget:
        stop=Event()
        def check(self):pass
    budget=Budget()
    def fail(job):
        raise RuntimeError('synthetic native failure')
    with pytest.raises(RuntimeError,match='native failure'):
        predictor.run_bounded_jobs([1,2,3],fail,budget)
    assert budget.stop.is_set()


def test_missing_model_sex_rejected_before_sealed_read(frozen_predictions):
    root,lock,design,sealed=frozen_predictions;sealed.unlink()
    path=root/'predictions/predictions.json';data=json.loads(path.read_text());data['models'].pop();write(path,data)
    locked=json.loads(lock.read_text());locked['predictions'][0]['sha256']=predictor.file_sha(path);write(lock,locked)
    with pytest.raises(ValueError,match='lacks exactly'):
        wrapper.score_and_seal(lock,design,root/'evaluation')
