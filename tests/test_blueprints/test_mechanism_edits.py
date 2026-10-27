from copy import deepcopy
import json
import math
from pathlib import Path
import pytest

from pbpk_kb.blueprints.mechanism_edits import (ACTION, NATIVE_UNIT, MechanismNotApplicable,
    compile_native_candidate, native_candidates, km_umol_per_l, _branches)
from pbpk_kb.blueprints.native_edits import apply_parameter_values, reviewed_process_parameter_path

ROOT = Path(__file__).resolve().parents[2]
ACTION_ID = 'partition_pksim_standard__' + ACTION


def case(drug='alprazolam'):
    directory = {'alprazolam': 'alprazolam_reconstruction_20260920_v2',
                 'midazolam': 'midazolam_reconstruction_20260920'}[drug]
    return json.loads((ROOT/'experiments/blueprint_design/development_inputs'/directory/'input_package.json').read_text())['case']


def spec_midazolam(request):
    source = request['instance']['compound_sources']['Processes']
    identity = {'internal_name': 'MetabolizationLiverMicrosomes_MM', 'molecule': 'CYP3A4',
                'data_source': 'Optimized', 'parameter_name': 'kcat'}
    return {'parameter_id': 'single_process_kcat', 'unit': '1/min', 'initial_value': 7.8703703704,
            'lower_bound': 1., 'upper_bound': 20., 'native_process': identity, 'source': source,
            'runtime_mappings': [{'runtime_path': reviewed_process_parameter_path('Midazolam', identity),
                                  'multiplier': 1., 'native_unit': '1/min'}]}


@pytest.mark.parametrize('unit,value', [('pmol/l',4e6),('nmol/l',4000),('µmol/l',4),('µM',4),('mM',.004),('mmol/l',.004)])
def test_source_molar_unit_conversion(unit,value):
    assert km_umol_per_l(value,unit) == 4.


@pytest.mark.parametrize('value,unit', [(0,'µM'),(-1,'µM'),(True,'µM'),(float('nan'),'µM'),(1,'mg/l'),(1,'umol/l')])
def test_nonphysical_or_unreviewed_km_rejected(value,unit):
    with pytest.raises(ValueError): km_umol_per_l(value,unit)


def test_actual_source_branches_and_composable_partition_candidates():
    data=case(); request=data['design_request']; specs=data['parameter_search']['parameters']
    original=deepcopy(request)
    candidates=native_candidates(request,project_root=ROOT,parameter_specs=specs)
    assert len(candidates)==4
    assert len({c['semantic_hash'] for c in candidates})==4
    base,_,linear,rrlinear=candidates
    assert request==original
    assert linear['snapshot']['Individuals']==base['snapshot']['Individuals']
    assert linear['snapshot']['Protocols']==base['snapshot']['Protocols']
    assert linear['edit_manifest']['applicability_condition']['qualified_at_all_doses'] is False
    assert linear['edit_manifest']['applicability_condition']['requires_scenario_specific_check'] is True
    assert linear['edit_manifest']['source_binding_verified'] is True
    assert linear['parameter_specs'][1]==specs[1]
    for b,m in zip(linear['edit_manifest']['branches'],linear['parameter_specs'][0]['runtime_mappings']):
        assert m['runtime_path'].endswith('|CLspec/[Enzyme]')
        assert m['native_unit']==NATIVE_UNIT
        assert m['multiplier']==b['kcat_anchor']/b['km_umol_per_l']
        process=linear['request']['instance']['compound']['Processes'][b['process_index']]
        assert process['InternalName']=='MetabolizationLiverMicrosomes_FirstOrder'
        assert not any(p['Name'] in {'kcat','Km'} for p in process['Parameters'])
    assert rrlinear['edit_count']==2


def test_fit_rebuild_converts_once_and_preserves_original_branches():
    data=case();request=data['design_request'];specs=data['parameter_search']['parameters']
    source=specs[0]['source'];values=[1.5,specs[1]['initial_value']]
    candidate=compile_native_candidate(request,ACTION_ID,project_root=ROOT,parameter_specs=specs,
                                       fitted_values=values,parameter_provenance=source)
    assert candidate['edit_manifest']['fitted_values_materialized']
    for b in candidate['edit_manifest']['branches']:
        p=candidate['request']['instance']['compound']['Processes'][b['process_index']]
        assert next(x['Value'] for x in p['Parameters'] if x['Name']=='CLspec/[Enzyme]')==1.5*b['linear_anchor']
    with pytest.raises(ValueError,match='source anchor'):
        apply_parameter_values(candidate['request'],candidate['parameter_specs'],values,source)


def test_single_enzyme_kcat_updates_no_other_enzyme_in_mm_and_linear_candidates():
    data=case('midazolam');request=data['design_request'];spec=spec_midazolam(request)
    before=deepcopy(request['instance']['compound']['Processes'])
    changed=apply_parameter_values(request,[spec],[10.],spec['source'])
    for old,new in zip(before,changed['instance']['compound']['Processes']):
        if old.get('Molecule')!='CYP3A4': assert new==old
        else: assert next(p['Value'] for p in new['Parameters'] if p['Name']=='kcat')==10.
    linear=compile_native_candidate(request,ACTION_ID,project_root=ROOT,parameter_specs=[spec],
                                    fitted_values=[10.],parameter_provenance=spec['source'])
    assert linear['parameter_specs'][0]['unit']=='1/min'
    assert linear['parameter_specs'][0]['runtime_mappings'][0]['multiplier']==.25
    for b in linear['edit_manifest']['branches']:
        proc=linear['request']['instance']['compound']['Processes'][b['process_index']]
        actual=next(p['Value'] for p in proc['Parameters'] if p['Name']=='CLspec/[Enzyme]')
        assert actual==(2.5 if proc['Molecule']=='CYP3A4' else b['linear_anchor'])


def test_no_mm_source_cannot_invent_reverse_mm_or_fake_a_mechanism():
    data=case();request=data['design_request'];specs=data['parameter_search']['parameters']
    linear=compile_native_candidate(request,ACTION_ID,project_root=ROOT,parameter_specs=specs)
    candidates=native_candidates(linear['request'],project_root=ROOT,parameter_specs=linear['parameter_specs'])
    assert len(candidates)==2
    with pytest.raises(MechanismNotApplicable):
        compile_native_candidate(linear['request'],ACTION_ID,project_root=ROOT,parameter_specs=linear['parameter_specs'])


def test_same_enzyme_interaction_requires_separate_audit():
    compound=case()['design_request']['instance']['compound']
    compound['Processes'].append({'InternalName':'CompetitiveInhibition','Molecule':'CYP3A4'})
    with pytest.raises(MechanismNotApplicable,match='interaction'):_branches(compound)


@pytest.mark.parametrize('mutate', [
    lambda s:s['runtime_mappings'][0].update(multiplier=1.),
    lambda s:s['runtime_mappings'][0].update(native_unit='1/min'),
    lambda s:s['runtime_mappings'][0].update(runtime_path='Alprazolam|Lipophilicity'),
    lambda s:s['mechanism_conversion']['branches'][0].update(km_umol_per_l=1.),
    lambda s:s['native_processes'][0].update(molecule='CYP1A2'),
])
def test_transformed_parameter_cannot_escape_typed_mapping(mutate):
    data=case();specs=data['parameter_search']['parameters']
    linear=compile_native_candidate(data['design_request'],ACTION_ID,project_root=ROOT,parameter_specs=specs)
    transformed=linear['parameter_specs'];mutate(transformed[0])
    with pytest.raises(ValueError):apply_parameter_values(linear['request'],transformed,[1.,2.],specs[0]['source'])


def test_low_concentration_analytic_limit_and_high_concentration_failure():
    for ratio in [1e-8,.01,.1,1.,10.]:
        enzyme_amount,kcat,km=3.,7.,4.; concentration=ratio*km
        native_mm=enzyme_amount*kcat*concentration/(km+concentration)
        native_linear=enzyme_amount*(kcat/km)*concentration
        assert math.isclose((native_linear-native_mm)/native_mm,ratio,rel_tol=1e-7)
    assert ratio==10.  # Linear law differs; it is not an identity at finite C.


def test_tiny_source_km_must_not_overflow_native_clearance():
    compound=case()['design_request']['instance']['compound']
    for process in compound['Processes']:
        for p in process['Parameters']:
            if p['Name']=='Km':p.update(Value=1e-320,Unit='µmol/l')
    with pytest.raises(ValueError,match='overflow'):_branches(compound)


def test_shared_factor_cannot_be_repurposed_to_cross_enzyme_scaling():
    data=case();specs=data['parameter_search']['parameters']
    specs[0]['native_processes'][1]['molecule']='UGT1A4'
    with pytest.raises(ValueError,match='outside the transformed'):
        compile_native_candidate(data['design_request'],ACTION_ID,project_root=ROOT,parameter_specs=specs)


def test_fitted_values_require_recorded_application_provenance():
    data=case()
    with pytest.raises(ValueError,match='calibration provenance'):
        compile_native_candidate(data['design_request'],ACTION_ID,project_root=ROOT,
                                 parameter_specs=data['parameter_search']['parameters'],fitted_values=[1.,2.])
