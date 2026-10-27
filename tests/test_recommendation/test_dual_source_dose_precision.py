from copy import deepcopy
from pathlib import Path
import json
import math
import pytest
from recommendation_execution_repairs import json_racemate_contract as m, complete_candidate as complete
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
JOB=ROOT/'experiments/model_recommendation/execution_validation_20261001/native/7e518bea38c1999f987924b8_r4g'

@pytest.fixture
def original_actual_dose():
    plan=json.loads((JOB/'execution_plan.json').read_text())
    channel=plan['dual_parent_contract']['channels'][0]
    meta=json.loads((JOB/'executable_metadata.json').read_text())['metadata']
    inv=json.loads((JOB/'inspect/result.json').read_text())['native']['inventory']
    source=next(g['Dose'] for key,g in complete._metadata_events(meta)[0].items() if key.startswith('Events|'+channel['protocol_name']+'|'))
    native=next(p for p in inv['parameters'] if p.get('id')==source['id'])
    return channel,source,native


def test_actual_double_conversion_is_not_exact_decimal_division(original_actual_dose):
    channel,source,native=original_actual_dose
    assert native['value']==float(source['value'])==20.0*1e-6
    assert native['value']!=20.0/1e6
    audit=m.validate_source_channel_dose(channel,source,native,project_root=ROOT)
    assert audit['source_base_value_hex']==audit['expected_base_value_hex']==audit['native_value_hex']
    assert audit['allowed_ulp_difference']==0


@pytest.mark.parametrize('field', ['native','saved','both'])
@pytest.mark.parametrize('direction', [-math.inf,math.inf])
def test_one_ulp_any_side_is_rejected(original_actual_dose,field,direction):
    channel,source,native=deepcopy(original_actual_dose)
    altered=math.nextafter(native['value'],direction)
    if field in ['native','both']:native['value']=altered
    if field in ['saved','both']:source['value']=repr(altered)
    with pytest.raises(ValueError):m.validate_source_channel_dose(channel,source,native,project_root=ROOT)


@pytest.mark.parametrize('mutation', ['half','double','wrong_unit','native_dimension','source_dimension','source_formula','source_rhs','native_nan','saved_nan','saved_bool','native_bool','unknown_unit','forged_original','forged_contract'])
def test_real_parameter_or_binding_changes_are_rejected(original_actual_dose,mutation):
    channel,source,native=deepcopy(original_actual_dose)
    if mutation=='half':native['value']/=2;source['value']=str(native['value'])
    elif mutation=='double':native['value']*=2;source['value']=str(native['value'])
    elif mutation=='wrong_unit':native['unit']='mg'
    elif mutation=='native_dimension':native['dimension']='Amount'
    elif mutation=='source_dimension':source['dimension']='Amount'
    elif mutation=='source_formula':source['formula_id']='other'
    elif mutation=='source_rhs':source['rhs_formula_id']='other'
    elif mutation=='native_nan':native['value']=math.nan
    elif mutation=='saved_nan':source['value']='nan'
    elif mutation=='saved_bool':source['value']=True
    elif mutation=='native_bool':native['value']=True
    elif mutation=='unknown_unit':native['unit']='mg/kg'
    elif mutation=='forged_original':next(p for p in channel['protocol']['Parameters'] if p['Name']=='InputDose')['Value']=10.
    elif mutation=='forged_contract':channel['source_dose_rational_mg']=[10,1]
    with pytest.raises(ValueError):m.validate_source_channel_dose(channel,source,native,project_root=ROOT)


def test_exact_display_unit_readback_is_allowed(original_actual_dose):
    channel,source,native=deepcopy(original_actual_dose)
    native['unit']='mg';native['value']=float(source['value'])/1e-6
    assert m.validate_source_channel_dose(channel,source,native,project_root=ROOT)['native_unit']=='mg'


def test_changed_registered_official_evidence_is_rejected(original_actual_dose,monkeypatch):
    refs=deepcopy(m.UNIT_CONVERSION_EVIDENCE)
    refs[-1]['sha256']='0'*64
    monkeypatch.setattr(m,'UNIT_CONVERSION_EVIDENCE',refs)
    with pytest.raises(ValueError,match='EVIDENCE_CHANGED'):
        m.validate_source_channel_dose(*original_actual_dose,project_root=ROOT)
