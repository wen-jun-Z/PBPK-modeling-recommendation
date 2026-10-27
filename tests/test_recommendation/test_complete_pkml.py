"""Source model preservation and fixed clinical-administration boundaries."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.recommendation_execution_repairs import native_pkml as native


def request():
    return {'version': native.VERSION, 'action': 'run', 'output_path': 'Organism|Drug|Plasma',
            'output_times_min': [0., 1., 60.], 'patches': [
                {'path': 'Events|Oral|Application_1|ProtocolSchemaItem|Dose', 'value': 100., 'unit': 'mg',
                 'expected_before_value': .05, 'expected_before_unit': 'g'}]}


@pytest.mark.parametrize('key,value', [('value', float('nan')), ('unit', 'mg/kg'),
    ('path', 'Drug|Clearance'), ('expected_before_unit', None)])
def test_unresolved_dose_and_drug_parameter_changes_are_rejected(key, value):
    r = request(); r['patches'][0][key] = value
    with pytest.raises(ValueError): native.validate_request(r)


def test_duplicated_and_nonfinite_times_are_rejected():
    for times in [[0., 1., 1.], [1., 2.], [0., float('inf')]]:
        r = request(); r['output_times_min'] = times
        with pytest.raises(ValueError): native.validate_request(r)


def test_explicit_administration_request_is_valid_without_fitting():
    native.validate_request(request())


def model(path, *, dose='1', clearance='2', mapping='7', path_value='Organism|Liver'):
    path.write_text(f'''<SimulationTransfer><Simulation><SimulationConfiguration>
      <Individual name="source"/><SimulationSettings randomSeed="42"><Solver tolerance="0.000001"/>
      <OutputSchema end="100"/></SimulationSettings></SimulationConfiguration><Model id="model"><Root><Children>
      <Parameter id="dose" name="Dose" value="{dose}" isFixedValue="0" dim="Mass" />
      <Parameter id="kinetic" name="CLint" value="{clearance}" />
      <OutputSchema><Parameter value="100" /></OutputSchema>
      </Children></Root><FormulaCache><Formula id="f"><Paths><Path path="{mapping}" /></Paths></Formula>
      <StringMap><Map id="{mapping}" s="{path_value}" /></StringMap></FormulaCache>
      </Model></Simulation></SimulationTransfer>''')
    return path


def test_full_model_audit_allows_only_registered_dose_and_serialization_index(tmp_path):
    before = model(tmp_path/'before.pkml')
    after = model(tmp_path/'after.pkml', dose='3', mapping='18')
    changes = [{'id':'dose', 'path':request()['patches'][0]['path'], 'before':{'value':1.}, 'expected_native_value':3.}]
    assert native.audit_preserved_model(before, after, changes)['accepted']


@pytest.mark.parametrize('kwargs', [{'clearance':'3'}, {'path_value':'Organism|Kidney'}])
def test_drug_kinetics_and_formula_bindings_cannot_be_hidden(tmp_path, kwargs):
    before = model(tmp_path/'before.pkml')
    after = model(tmp_path/'after.pkml', dose='3', **kwargs)
    changes = [{'id':'dose', 'path':request()['patches'][0]['path'], 'before':{'value':1.}, 'expected_native_value':3.}]
    assert not native.audit_preserved_model(before, after, changes)['accepted']


def test_unregistered_dose_change_fails_preservation(tmp_path):
    before=model(tmp_path/'before.pkml'); after=model(tmp_path/'after.pkml',dose='3')
    assert not native.audit_preserved_model(before,after,[])['accepted']


def test_clinical_outcome_files_cannot_be_input(tmp_path):
    path=tmp_path/'sealed_values';path.mkdir();(path/'case.pkml').write_text('private')
    with pytest.raises(ValueError,match='unsealed'): native.inside(path/'case.pkml',tmp_path)


def test_existing_attempt_is_never_overwritten(tmp_path):
    pkml=model(tmp_path/'source.pkml');out=tmp_path/'existing';out.mkdir()
    with pytest.raises(FileExistsError): native.execute_complete_pkml(pkml,out,project_root=tmp_path)


def test_wrong_inline_dose_cannot_hide_in_the_allowed_edit(tmp_path):
    before=model(tmp_path/'before.pkml');after=model(tmp_path/'after.pkml',dose='4')
    changes=[{'id':'dose','path':request()['patches'][0]['path'],'before':{'value':1.},'expected_native_value':3.}]
    with pytest.raises(ValueError,match='Inline administration'):
        native.audit_preserved_model(before,after,changes)


def test_solver_and_individual_configuration_are_protected(tmp_path):
    before=model(tmp_path/'before.pkml');after=model(tmp_path/'after.pkml')
    assert native.audit_configuration(before,after,allow_output_changes=True)
    after.write_text(after.read_text().replace('randomSeed="42"','randomSeed="43"'))
    assert not native.audit_configuration(before,after,allow_output_changes=True)


def test_import_encoding_normalization_is_narrow_and_does_not_hide_values(tmp_path):
    before=model(tmp_path/'before.pkml');after=model(tmp_path/'after.pkml')
    before.write_text(before.read_text().replace('<Individual name="source"/>',
        '<Individual><IndividualParameter name="Percentile" value="0.5" displayUnit="kg"/></Individual><CoreCalculationMethod name="same"/>'))
    after.write_text(after.read_text().replace('<Individual name="source"/>',
        '<Individual><IndividualParameter name="Percentile" value="0.5"/></Individual><CoreCalculationMethod name="same" id="generated"/>'))
    assert not native.audit_configuration(before,after,allow_output_changes=False)
    assert native.audit_configuration(before,after,allow_output_changes=False,allow_import_encoding=True)
    after.write_text(after.read_text().replace('value="0.5"','value="0.6"'))
    assert not native.audit_configuration(before,after,allow_output_changes=False,allow_import_encoding=True)


def test_only_replaced_output_constants_may_leave_settings_cache(tmp_path):
    before=model(tmp_path/'before.pkml');after=model(tmp_path/'after.pkml')
    before.write_text(before.read_text().replace('<OutputSchema end="100"/>',
        '<OutputSchema><Parameter name="End time" formula="end"/></OutputSchema>'
        '<FormulaCache><Formulas><ConstantFormula id="end" value="100"/></Formulas><StringMap/></FormulaCache>'))
    assert native.audit_configuration(before,after,allow_output_changes=True)
    before.write_text(before.read_text().replace('<Solver tolerance="0.000001"/>',
        '<Solver tolerance="0.000001" formula="end"/>'))
    assert not native.audit_configuration(before,after,allow_output_changes=True)


def valid_native():
    r=request();p=r['patches'][0]
    before={'path':p['path'],'id':'dose','value':.05,'unit':'g'}
    after={**before,'value':.1}
    n={'parameter_changes':[{'id':'dose','path':p['path'],'before':before,'after':after,
       'requested_value':100.,'requested_unit':'mg','expected_native_value':.1}],
       'output_grid_verified':True,'curve':{'time_min':[0.,1.,60.],'concentration':[0.,2.,1.],'time_unit':'min'}}
    return n,r


def test_native_values_and_time_grid_have_independent_python_validation():
    n,r=valid_native();native.validate_native_output(n,r)
    n['curve']['time_min'][1]=2.
    with pytest.raises(ValueError,match='grid'):native.validate_native_output(n,r)


def test_self_consistent_but_wrong_native_mass_conversion_fails():
    n,r=valid_native();n['parameter_changes'][0]['after']['value']=100.
    n['parameter_changes'][0]['expected_native_value']=100.
    with pytest.raises(ValueError,match='unit conversion'):native.validate_native_output(n,r)


def test_native_single_precision_grid_is_exactly_bound_not_loosely_tolerated():
    import struct
    n,r=valid_native()
    r['output_times_min'][1]=19.8
    n['curve']['time_min'][1]=19.8
    n['native_output_time_min']=[0.,struct.unpack('f',struct.pack('f',19.8))[0],60.]
    native.validate_native_output(n,r)
    n['native_output_time_min'][1]+=1e-6
    with pytest.raises(ValueError,match='grid'):native.validate_native_output(n,r)


def test_fixed_dose_override_preserves_the_original_default_constant(tmp_path):
    before=model(tmp_path/'before.pkml',dose='1');after=model(tmp_path/'after.pkml',dose='3')
    after.write_text(after.read_text().replace('value="3" isFixedValue="0" dim="Mass"',
        'value="3" isFixedValue="1" dim="Mass" formula="old-dose"').replace('<FormulaCache>',
        '<FormulaCache><Formulas><ConstantFormula id="old-dose" dim="Mass" value="1"/></Formulas>'))
    changes=[{'id':'dose','path':request()['patches'][0]['path'],'before':{'value':1.},'expected_native_value':3.}]
    # An originally absent empty Formulas cache and a now-empty added cache
    # differ structurally; give the original its actual serialized empty group.
    before.write_text(before.read_text().replace('<FormulaCache>','<FormulaCache><Formulas/>'))
    assert native.audit_preserved_model(before,after,changes)['accepted']
    after.write_text(after.read_text().replace('dim="Mass" value="1"','dim="Mass" value="2"'))
    with pytest.raises(ValueError,match='unchanged original formula'):native.audit_preserved_model(before,after,changes)


def monitor_pair():
    r = request(); r['patches'] = []
    r.update(zero_monitor_paths=['Organism|Other|Mass'],zero_monitor_dimensions={'Organism|Other|Mass':'Amount'},
             proof_parameter_requirements=[{'path':'Organism|Volume','id':'v','dimension':'Volume','finite_required':True,'strictly_positive_required':True}],
             positive_domain_monitor_dimensions={'Organism|Liquid':'Volume'})
    n = {'parameter_changes':[], 'curve':{'time_min':[0.,1.,60.],'concentration':[0.,1.,.5],'time_unit':'min'},
         'native_output_time_min':[0.,1.,60.],'output_grid_verified':True,
         'zero_monitors':[{'path':'Organism|Other|Mass','dimension':'Amount','unit':'µmol','values':[0.,0.,0.]}],
         'zero_monitor_time_min':[0.,1.,60.],'zero_monitor_native_time_min':[0.,1.,60.],'zero_monitor_grid_verified':True,
         'proof_parameter_readback':[{'path':'Organism|Volume','id':'v','dimension':'Volume','unit':'l','value':1.}],
         'positive_domain_monitors':[{'path':'Organism|Liquid','dimension':'Volume','unit':'l','values':[1.,1.,1.]}],
         'domain_monitor_time_min':[0.,1.,60.],'domain_monitor_native_time_min':[0.,1.,60.],'domain_monitor_grid_verified':True}
    return r,n


def test_zero_exposure_and_dynamic_domains_are_replayed_without_second_solver():
    r,n=monitor_pair(); native.validate_request(r); native.validate_native_output(n,r)


@pytest.mark.parametrize('change',['missing','extra','duplicate','positive','nan','short','unit','dimension','time','rawtime','flag',
                                  'coefficient_id','coefficient_zero','domain_zero','domain_missing','domain_time'])
def test_monitor_or_domain_tampering_cannot_pass(change):
    r,n=monitor_pair(); row=n['zero_monitors'][0]
    if change=='missing': n['zero_monitors']=[]
    elif change=='extra': n['zero_monitors'].append({**row,'path':'Unexpected'})
    elif change=='duplicate': n['zero_monitors'].append(deepcopy(row))
    elif change=='positive': row['values'][1]=1e-30
    elif change=='nan': row['values'][1]=float('nan')
    elif change=='short': row['values'].pop()
    elif change=='unit': row['unit']='mg/ml'
    elif change=='dimension': row['dimension']='Concentration'
    elif change=='time': n['zero_monitor_time_min'][1]=2.
    elif change=='rawtime': n['zero_monitor_native_time_min'][1]=2.
    elif change=='flag': n['zero_monitor_grid_verified']=False
    elif change=='coefficient_id': n['proof_parameter_readback'][0]['id']='other'
    elif change=='coefficient_zero': n['proof_parameter_readback'][0]['value']=0.
    elif change=='domain_zero': n['positive_domain_monitors'][0]['values'][1]=0.
    elif change=='domain_missing': n['positive_domain_monitors']=[]
    else: n['domain_monitor_time_min'][1]=2.
    with pytest.raises(ValueError): native.validate_native_output(n,r)
