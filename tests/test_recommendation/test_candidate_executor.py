"""Scientific boundary and numeric checks for fixed-catalog native execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("candidate_executor", ROOT / "scripts/recommendation_candidate_executor.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


@pytest.fixture
def case():
    path = ROOT / "experiments/blueprint_design/external_validation/additional_native_cohort_20260920/protocols/lisinopril_pmc7993260_umin000030894_water_control10mg.json"
    return json.loads(path.read_text())


@pytest.fixture
def candidate():
    rows = ROOT / "experiments/model_recommendation/frontier_core_145_20260921/inputs/candidates.jsonl"
    return next(json.loads(line) for line in rows.open() if json.loads(line)["candidate_id"] == "16b753ad-835b-5c90-815d-bd3b025647bb")


def test_true_clinical_protocol_normalizes_without_outcome_file(case):
    c = ex.scenario_contract(case)
    assert c["dose_mg"] == 10
    assert c["route"] == "Oral"
    assert c["sampling_times_min"] == [0, 30, 60, 120, 180, 240, 360, 480, 1440]


def test_retains_entire_drug_and_fitted_local_parameters(case, candidate):
    p = ex.prepare_candidate(case, candidate)
    assert p["status"] == "ready", p["reasons"]
    original = json.loads((ROOT / candidate["provenance"]["source_path"]).read_text())
    assert p["snapshot"]["Compounds"] == original["Compounds"]
    assert p["snapshot"].get("ExpressionProfiles") == original.get("ExpressionProfiles")
    source_sim = next(s for s in original["Simulations"] if s["Name"] == p["source_simulation"])
    before = [x for x in source_sim["Parameters"] if not x["Path"].startswith("Events|")]
    after = [x for x in p["snapshot"]["Simulations"][0]["Parameters"] if not x["Path"].startswith("Events|")]
    assert before == after
    assert "ObservedData" not in p["snapshot"]
    assert p["fitting_performed"] is False
    assert p["compatibility"]["strict_target_match"] is False
    assert p["source_individual"]["OriginData"]["Age"]["Value"] == 30
    assert p["contract"]["population"]["age_range_year"] == [21, 29]
    assert any(x["path"].endswith("|Dose") and x["value"] == 10 for x in p["native_expectations"]["expectations"])


@pytest.mark.parametrize("mutate,reason", [
    (lambda c: c["protocol"].update(route="subcutaneous"), "UNSUPPORTED_TARGET_ROUTE"),
    (lambda c: c["protocol"].update(dose_unit="mg/kg"), "ABSOLUTE_MASS_DOSE_REQUIRED"),
    (lambda c: c["protocol"].update(dose=-1), "POSITIVE_NUMBER_REQUIRED"),
    (lambda c: c["protocol"].update(administrations=[{"time_min": 60,"dose":10,"dose_unit":"mg"}]), "TARGET_ADMINISTRATION_MUST_START_AT_ZERO"),
    (lambda c: c["protocol"].update(administrations=[{"time_min":0},{"time_min":60},{"time_min":180}]), "NONREGULAR_TARGET_REGIMEN"),
    (lambda c: c["protocol"].update(administrations=[{"time_min":0,"dose":20,"dose_unit":"mg"}]), "CONFLICTING_TARGET_DOSES"),
    (lambda c: c["evaluation_target"].update(matrix="serum"), "PARENT_TOTAL_PLASMA_OR_WHOLE_BLOOD_OUTPUT_ONLY"),
    (lambda c: c["evaluation_target"].update(analyte="metabolite"), "EXPLICIT_PARENT_ANALYTE_REQUIRED"),
    (lambda c: c["evaluation_target"].update(binding="unbound"), "TOTAL_PARENT_OUTPUT_ONLY"),
    (lambda c: c["protocol"].update(sampling_times_min=[0,60,60]), "SAMPLING_TIMES_MUST_INCREASE"),
    (lambda c: c.update(observations=[{"concentration":1}]), "OUTCOMES_FORBIDDEN"),
])
def test_invalid_or_unsupported_design_is_explicitly_blocked(case, candidate, mutate, reason):
    mutate(case)
    p = ex.prepare_candidate(case, candidate)
    assert p["status"] == "blocked"
    assert reason in p["reasons"][0]
    assert p["scientific_inapplicability_inferred"] is False


def test_reference_subject_not_silently_population(case, candidate):
    case["prediction_contract"]["source_reference_subject_prediction_allowed"] = False
    case['execution_policy'] = {'allow_reference_subject_approximation':True}
    p = ex.prepare_candidate(case, candidate)
    assert p["status"] == "blocked"
    assert "APPROXIMATION_REQUIRES_EXPLICIT_PROTOCOL_PERMISSION" in p["reasons"][0]


def test_wrong_drug_and_drift_never_run(case, candidate):
    c = deepcopy(candidate)
    c["compounds"] = ["midazolam"]
    assert "TARGET_COMPOUND_NOT_REGISTERED" in ex.prepare_candidate(case,c)["reasons"][0]
    c = deepcopy(candidate)
    c["provenance"]["source_sha256"] = "0"*64
    assert "SOURCE_CHECKSUM_MISMATCH" in ex.prepare_candidate(case,c)["reasons"][0]


def test_sealed_and_outside_paths_rejected_without_opening():
    with pytest.raises(ValueError, match="sealed"):
        ex._inside(ROOT / "experiments/sealed_answers/missing.json", exists=False)
    with pytest.raises(ValueError, match="inside"):
        ex._inside(ROOT / "../outside.json", exists=False)


def test_molar_to_mass_and_window_trapezoid_are_dimensionally_correct():
    curve = {"time_min":[0.,60.,120.],"concentration":[0.,2.,0.],"time_unit":"min","unit":"umol/l"}
    p = {"candidate_id":"c","scenario_id":"q","output_path":"parent",
        "molecular_weight_g_per_mol":100.,"contract":{"sampling_times_min":[0.,30.,60.]},"compatibility":{}}
    out = ex.prediction_from_curve(curve,p)
    assert out["sampled_concentration_ng_ml"] == [0.,100.,200.]
    assert out["endpoints"]["sampled_Cmax"]["value"] == 200
    assert out["endpoints"]["AUC_fixed_sampling_window"]["value"] == 100
    assert "AUClast" in out["uncomputed_endpoints"]
    assert "AUCinf" in out["uncomputed_endpoints"]


@pytest.mark.parametrize("curve", [
    {"time_min":[0.,60.],"concentration":[0.,-1.],"time_unit":"min","unit":"umol/l"},
    {"time_min":[0.,60.],"concentration":[0.,1.],"time_unit":"h","unit":"umol/l"},
    {"time_min":[0.,60.],"concentration":[0.,1.],"time_unit":"min","unit":"mg/l"},
    {"time_min":[0.,0.],"concentration":[0.,1.],"time_unit":"min","unit":"umol/l"},
])
def test_nonphysical_or_ambiguous_curves_fail(curve):
    p = {"candidate_id":"c","scenario_id":"q","output_path":"p","molecular_weight_g_per_mol":100.,
        "contract":{"sampling_times_min":[0.,60.]},"compatibility":{}}
    with pytest.raises(ValueError):
        ex.prediction_from_curve(curve,p)


def test_no_extrapolation():
    with pytest.raises(ValueError,match="extrapolation"):
        ex._interpolate([0.,60.],[0.,1.],61.)


def test_target_and_source_food_conflict_excluded(case,candidate):
    raw = json.loads((ROOT/candidate["provenance"]["source_path"]).read_text())
    opts, rejected = ex._source_options(raw, ex.scenario_contract(case), source_sha256=candidate['provenance']['source_sha256'])
    assert opts[0]["name"] == "Mojaverian PO fasted 20mg"
    assert any(r["simulation"] == "Mojaverian PO fed 20mg" and "CONFLICT" in r["reason"] for r in rejected)


def test_explicit_negative_analyte_qualifier_is_not_positive_unbound(case):
    case['evaluation_target']['analyte'] = 'Total parent plasma drug; not metabolite or unbound concentration'
    assert ex.scenario_contract(case)['binding'] == 'total'


def test_predose_samples_excluded_and_disclosed(case):
    case['protocol']['sampling_times_min'] = [-60,30,60,1440]
    c=ex.scenario_contract(case)
    assert c['sampling_times_min'] == [30,60,1440]
    assert c['predose_sampling_times_min_not_simulated'] == [-60]


def test_nonzero_observation_origin_never_treated_as_dose_time(case):
    case['normalized_design']={'regimen':{'observation_origin_h':96}}
    with pytest.raises(ValueError,match='OBSERVATION_ORIGIN'):
        ex.scenario_contract(case)


def test_native_binding_from_other_source_does_not_waive_formulation_mismatch(case,candidate):
    raw=json.loads((ROOT/candidate['provenance']['source_path']).read_text())
    c=ex.scenario_contract(case)
    c['formulation']='tablet'
    opts,rejected=ex._source_options(raw,c,source_sha256='0'*64)
    assert not opts
    assert any('FORMULATION_CLASS_DIFFERS' in r['reason'] for r in rejected)


def test_frozen_metformin_salt_conversion_changes_model_dose_only():
    path=ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    s=next(json.loads(line) for line in path.open() if json.loads(line)['compound']=='metformin')
    c=ex.scenario_contract(s)
    assert c['reported_labelled_dose_mg']==250
    assert c['dose_mg']==195
    assert c['salt_conversion']['factor']==.78


def test_unresolved_salt_mapping_blocked_before_source(case):
    case['dose']={'value':10,'unit':'mg','conversion_rule':'Freeze native salt-to-active mapping'}
    with pytest.raises(ValueError,match='MAPPING_NOT_FROZEN'):
        ex.scenario_contract(case)


def test_active_dose_without_source_or_conflicting_factor_rejected(case):
    case['protocol']['active_moiety_dose_mg']=5
    with pytest.raises(ValueError,match='CONVERSION_SOURCE'):
        ex.scenario_contract(case)
    case['protocol']['salt_conversion']={'factor':.2,'source_file':'x','source_file_sha256':'0'*64}
    with pytest.raises(ValueError,match='DOSE_CONFLICT'):
        ex.scenario_contract(case)


@pytest.mark.parametrize('compound,count,interval_h,origin_h', [
    ('clarithromycin', 10, 12, 96), ('itraconazole', 29, 12, 336), ('pioglitazone', 7, 24, 144)])
def test_frozen_repeated_protocol_uses_exact_times_and_observation_origin(compound,count,interval_h,origin_h):
    path=ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    s=next(json.loads(line) for line in path.open() if json.loads(line)['compound']==compound)
    c=ex.scenario_contract(s)
    assert c['administration_times_min']==[i*interval_h*60 for i in range(count)]
    assert c['observation_origin_min']==origin_h*60
    assert c['absolute_sampling_times_min']==[origin_h*60+t for t in c['sampling_times_min']]


def test_repeated_clarithromycin_preserves_model_and_checks_every_target_event():
    path=ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    s=next(json.loads(line) for line in path.open() if json.loads(line)['compound']=='clarithromycin')
    rows=ROOT/'experiments/model_recommendation/frontier_core_145_20260921/inputs/candidates.jsonl'
    c=next(json.loads(line) for line in rows.open() if json.loads(line)['candidate_id']=='b49f0fa2-fc0e-5663-b4bf-c93fa00e0213')
    p=ex.prepare_candidate(s,c)
    assert p['status']=='ready',p['reasons']
    assert p['execution_revision']=='engineering_repair_02'
    assert p['implementation_sha256']==ex.implementation_hashes()
    original=json.loads((ROOT/c['provenance']['source_path']).read_text())
    assert p['snapshot']['Compounds']==original['Compounds']
    assert p['snapshot']['ExpressionProfiles']==original['ExpressionProfiles']
    assert p['native_expectations']['expected_event_count']==10
    checks=p['native_expectations']['expectations']
    assert [r['value'] for r in checks if r['path'].endswith('|Start time')]==[i*720 for i in range(10)]
    assert [r['value'] for r in checks if r['path'].endswith('|Dose')]==[500]*10
    assert p['compatibility']['strict_target_match'] is False


def test_repeat_origin_shifts_export_and_auc_without_discarding_native_history():
    curve={'time_min':[0.,60.,120.,150.,180.], 'concentration':[0.,20.,1.,2.,1.],
           'time_unit':'min','unit':'umol/l'}
    p={'candidate_id':'c','scenario_id':'q','output_path':'parent','molecular_weight_g_per_mol':100.,
       'contract':{'sampling_times_min':[0.,30.,60.], 'observation_origin_min':120.,
                   'absolute_sampling_times_min':[120.,150.,180.]},'compatibility':{}}
    out=ex.prediction_from_curve(curve,p)
    assert out['time_h']==[-2.,-1.,0.,.5,1.]
    assert out['sampled_concentration_ng_ml']==[100.,200.,100.]
    assert out['endpoints']['sampled_Cmax']['value']==200
    assert out['endpoints']['AUC_fixed_sampling_window']['value']==150
    assert out['endpoints']['AUC_fixed_sampling_window']['window_h']==[0.,1.]
    assert out['native_time_h_after_first_dose']==[0.,1.,2.,2.5,3.]


def test_breakfast_does_not_select_fasted_itraconazole():
    path=ROOT/'experiments/model_recommendation/execution_validation_20261001/inputs/scenarios.jsonl'
    s=next(json.loads(line) for line in path.open() if json.loads(line)['compound']=='itraconazole')
    source=ROOT/'artifacts/pbpk-model-assets/public_collection_20260909/Itraconazole-Model__59b6b686ba37/source/Itraconazole-Model.json'
    options,rejected=ex._source_options(json.loads(source.read_text()),ex.scenario_contract(s))
    assert options
    assert all(not ex.re.search(r'\bfast(?:ed)?\b', row['name'].casefold()) for row in options)
    assert any('SOURCE_FASTED_TARGET_FED_CONFLICT' in row['reason'] for row in rejected)
