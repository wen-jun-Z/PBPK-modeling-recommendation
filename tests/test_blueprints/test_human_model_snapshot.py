"""Static synthetic fixtures only; these tests never load or execute PK-Sim."""
from __future__ import annotations

import copy
import json

import pytest

from pbpk_kb.blueprints.human_model_snapshot import (
    HumanModelSnapshotError,
    build_human_model_snapshot,
)


def _source() -> dict:
    return {"provenance_id": "00000000-0000-0000-0000-000000000001",
            "provenance_kind": "user_request", "source_id": "synthetic-unit-fixture",
            "source_location": "toy values only", "source_checksum": "a" * 64}


def _parameter(name: str, value: float, unit: str | None = None) -> dict:
    result = {"Name": name, "Value": value}
    if unit is not None:
        result["Unit"] = unit
    return result


def _request(*, oral: bool = False) -> dict:
    compound = {
        "Name": "ToyDrug", "IsSmallMolecule": True, "PlasmaProteinBindingPartner": "Albumin",
        "Parameters": [_parameter("Molecular weight", 200.0, "g/mol"),
                       *[_parameter(name, 0) for name in ("F", "Cl", "Br", "I")]],
        "Lipophilicity": [{"Name": "toy-logp", "Parameters": [_parameter("Lipophilicity", 1.0, "Log Units")]}],
        "FractionUnbound": [{"Name": "toy-fu", "Species": "Human", "Parameters": [_parameter("Fraction unbound (plasma, reference value)", 0.5)]}],
        "Solubility": [{"Name": "toy-solubility", "Parameters": [_parameter("Solubility at reference pH", 1.0, "mg/ml"), _parameter("Reference pH", 7.0)]}],
        "PkaTypes": [],
        "Processes": [{"InternalName": "KidneyClearance", "DataSource": "toy-clearance", "Species": "Human",
                       "Parameters": [_parameter("Fraction unbound (experiment)", 0.5), _parameter("Plasma clearance", 1.0, "ml/min/kg")]}],
        "CalculationMethods": ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"],
    }
    protocol = {"Name": "ToyDose", "ApplicationType": "Oral" if oral else "IntravenousBolus",
                "DosingInterval": "Single", "Parameters": [_parameter("InputDose", 10.0, "mg"), _parameter("Start time", 0.0, "h")]}
    if oral:
        compound["IntestinalPermeability"] = [{"Name": "toy-permeability", "Parameters": [_parameter("Specific intestinal permeability (transcellular)", 1e-5, "cm/s")]}]
        protocol["Parameters"].append(_parameter("Volume of water/body weight", 3.0, "ml/kg"))
    request = {
        "individual": {"Name": "ToyHuman", "Seed": 17, "OriginData": {"Species": "Human", "Population": "European_ICRP_2002", "Gender": "MALE",
                        "Age": {"Value": 30, "Unit": "year(s)"}, "Weight": {"Value": 70, "Unit": "kg"}, "Height": {"Value": 175, "Unit": "cm"}}},
        "compound": compound, "protocol": protocol, "simulation_name": "ToyDesign", "snapshot_version": 80,
        "human_source": _source(), "protocol_source": _source(),
        "compound_sources": {key: _source() for key in compound if key not in {"Name", "Description", "IsSmallMolecule"}},
        "expected_compound_name": "ToyDrug",
        "output_schema": [{"Parameters": [_parameter("Start time", 0, "h"), _parameter("End time", 12, "h"), _parameter("Resolution", 4, "pts/h")]}],
        "output_selections": ["Organism|PeripheralVenousBlood|ToyDrug|Plasma (Peripheral Venous Blood)"],
    }
    if oral:
        request["formulation"] = {"Name": "ToySolution", "FormulationType": "Formulation_Dissolved", "Parameters": []}
        request["formulation_source"] = _source()
    return request


def test_iv_input_binds_standard_human_without_copying_a_model() -> None:
    request = _request()
    before = copy.deepcopy(request)
    result = build_human_model_snapshot(**request)
    snapshot, manifest = result["snapshot"], result["manifest"]
    assert request == before
    assert snapshot["Version"] == 80
    simulation = snapshot["Simulations"][0]
    assert simulation["Model"] == "4Comp"
    assert "ModelProperties" not in simulation
    assert simulation["Individual"] == "ToyHuman"
    assert simulation["Compounds"][0]["Name"] == "ToyDrug"
    assert simulation["Compounds"][0]["Processes"] == [{"Name": "Renal Clearances-toy-clearance", "SystemicProcessType": "Renal"}]
    assert snapshot["Compounds"][0] == request["compound"]
    assert manifest["configuration"] == "4Comp|Human"
    assert manifest["clearance_mechanisms"] == ["renal"]
    assert manifest["elimination_configured"] is True
    assert manifest["generation_mode"] == "standard_model_instantiation"
    assert manifest["compile_status"] == manifest["simulation_status"] == "not_run"
    assert manifest["source_binding_verified"] is False
    assert manifest["execution"] == {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0}
    assert manifest["normalized_administration"]["route"] == "intravenous"
    assert manifest["missing_execution_configuration"] == []
    json.dumps(result, allow_nan=False)
    snapshot["Individuals"][0]["Name"] = "modified output"
    assert request == before


def test_explicit_oral_formulation_is_mapped() -> None:
    result = build_human_model_snapshot(**_request(oral=True))
    protocol = result["snapshot"]["Simulations"][0]["Compounds"][0]["Protocol"]
    assert protocol == {"Name": "ToyDose", "Formulations": [{"Key": "Formulation", "Name": "ToySolution"}]}
    assert result["manifest"]["normalized_administration"] == {
        "route": "oral", "dose": 10.0, "unit": "mg", "start_time_h": 0.0,
        "application_type": "Oral", "formulation_name": "ToySolution",
    }


@pytest.mark.parametrize('duration,unit', [(10, 'min'), (1 / 6, 'h')])
def test_explicit_iv_infusion_preserves_native_protocol_and_normalizes_duration(duration, unit):
    request = _request()
    request['protocol']['ApplicationType'] = 'Intravenous'
    request['protocol']['Parameters'].append(_parameter('Infusion time', duration, unit))
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Protocols'][0] == request['protocol']
    administration = result['manifest']['normalized_administration']
    assert administration['route'] == 'intravenous'
    assert administration['application_type'] == 'Intravenous'
    assert administration['infusion_duration_min'] == pytest.approx(10)
    assert administration['formulation_name'] is None


@pytest.mark.parametrize('duration,unit', [(0, 'min'), (-10, 'min'), (10, 's'), (float('inf'), 'min')])
def test_iv_infusion_rejects_invalid_duration(duration, unit):
    request = _request()
    request['protocol']['ApplicationType'] = 'Intravenous'
    request['protocol']['Parameters'].append(_parameter('Infusion time', duration, unit))
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


def test_iv_infusion_has_no_default_duration_or_oral_formulation():
    request = _request()
    request['protocol']['ApplicationType'] = 'Intravenous'
    with pytest.raises(HumanModelSnapshotError, match='Infusion time'):
        build_human_model_snapshot(**request)
    request['protocol']['Parameters'].append(_parameter('Infusion time', 10, 'min'))
    request['formulation'] = {'Name': 'Unneeded', 'FormulationType': 'Formulation_Dissolved', 'Parameters': []}
    request['formulation_source'] = _source()
    with pytest.raises(HumanModelSnapshotError, match='IV administration'):
        build_human_model_snapshot(**request)


def test_bolus_cannot_silently_ignore_infusion_duration():
    request = _request()
    request['protocol']['Parameters'].append(_parameter('Infusion time', 10, 'min'))
    with pytest.raises(HumanModelSnapshotError, match='UNSUPPORTED_PARAMETER'):
        build_human_model_snapshot(**request)


def test_same_explicit_inputs_have_stable_checksum() -> None:
    request = _request()
    first = build_human_model_snapshot(**request)
    reordered = dict(reversed(list(request.items())))
    assert build_human_model_snapshot(**reordered) == first


@pytest.mark.parametrize("field", ["Parameters", "Lipophilicity", "FractionUnbound", "PkaTypes", "Processes", "CalculationMethods"])
def test_missing_drug_data_blocks_build_without_defaults(field: str) -> None:
    request = _request()
    del request["compound"][field]
    with pytest.raises(HumanModelSnapshotError) as error:
        build_human_model_snapshot(**request)
    assert error.value.code == "MISSING_FIELDS"
    assert "compound." + field in error.value.missing_fields


def test_missing_compound_source_is_reported() -> None:
    request = _request()
    del request["compound_sources"]["Lipophilicity"]
    with pytest.raises(HumanModelSnapshotError) as error:
        build_human_model_snapshot(**request)
    assert "compound_sources.Lipophilicity" in error.value.missing_fields


@pytest.mark.parametrize("version", [79, 81, True, "80"])
def test_unknown_snapshot_versions_are_rejected(version: object) -> None:
    request = _request()
    request["snapshot_version"] = version
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_SNAPSHOT_VERSION"):
        build_human_model_snapshot(**request)


def test_nonhuman_and_old_compound_id_are_rejected() -> None:
    request = _request()
    request["individual"]["OriginData"]["Species"] = "Rat"
    with pytest.raises(HumanModelSnapshotError, match="NON_HUMAN"):
        build_human_model_snapshot(**request)
    request = _request()
    request["expected_compound_name"] = "DifferentDrug"
    with pytest.raises(HumanModelSnapshotError, match="COMPOUND_IDENTITY_MISMATCH"):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize("target,field,value", [
    ("individual", "Parameters", [{"Path": "Organism|OldDrug|CL", "Value": 1}]),
    ("individual", "ExpressionProfiles", ["OldDrugEnzyme"]),
    ("compound", "Simulations", [{"Name": "OldDrug"}]),
    ("protocol", "Schemas", []),
])
def test_arbitrary_overrides_and_existing_model_blocks_are_rejected(target: str, field: str, value: object) -> None:
    request = _request()
    request[target][field] = value
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_FIELD"):
        build_human_model_snapshot(**request)


def test_duplicate_parameter_ids_are_rejected() -> None:
    request = _request()
    request["compound"]["Parameters"].append(copy.deepcopy(request["compound"]["Parameters"][0]))
    with pytest.raises(HumanModelSnapshotError, match="DUPLICATE_IDENTIFIER"):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1.0])
def test_invalid_molecular_weight_is_rejected(bad: object) -> None:
    request = _request()
    request["compound"]["Parameters"][0]["Value"] = bad
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


def test_inherited_derived_clearance_values_are_rejected() -> None:
    request = _request()
    request["compound"]["Processes"][0]["Parameters"].append(_parameter("Body weight", 72, "kg"))
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_PARAMETER"):
        build_human_model_snapshot(**request)


def test_enzyme_and_metabolite_mechanisms_cannot_be_silently_dropped() -> None:
    request = _request()
    request["compound"]["Processes"][0]["Molecule"] = "CYP3A4"
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_FIELD"):
        build_human_model_snapshot(**request)


def test_different_drug_output_and_duplicate_output_are_rejected() -> None:
    request = _request()
    request["output_selections"] = ["Organism|PeripheralVenousBlood|OldDrug|Plasma (Peripheral Venous Blood)"]
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_OUTPUT_PATH"):
        build_human_model_snapshot(**request)
    request = _request()
    request["output_selections"] *= 2
    with pytest.raises(HumanModelSnapshotError, match="DUPLICATE_OR_INVALID_OUTPUT"):
        build_human_model_snapshot(**request)


def test_oral_permeability_and_formulation_must_be_explicit() -> None:
    request = _request(oral=True)
    del request["compound"]["IntestinalPermeability"]
    with pytest.raises(HumanModelSnapshotError) as error:
        build_human_model_snapshot(**request)
    assert "compound.IntestinalPermeability" in error.value.missing_fields
    request = _request(oral=True)
    request["formulation"]["FormulationType"] = "Formulation_Tablet_Weibull"
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_FORMULATION"):
        build_human_model_snapshot(**request)


def test_repeat_dose_is_rejected() -> None:
    request = _request()
    request["protocol"]["DosingInterval"] = "DI_24"
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_ADMINISTRATION"):
        build_human_model_snapshot(**request)


def test_conflicting_source_checksum_is_rejected() -> None:
    request = _request()
    request["compound_sources"]["Parameters"]["source_checksum"] = "b" * 64
    with pytest.raises(HumanModelSnapshotError, match="CONFLICTING_PROVENANCE"):
        build_human_model_snapshot(**request)


def test_missing_execution_schedule_is_explicit_and_never_invented() -> None:
    request = _request()
    request.pop("output_schema")
    request.pop("output_selections")
    result = build_human_model_snapshot(**request)
    assert result["manifest"]["missing_execution_configuration"] == ["output_schema", "output_selections"]
    assert "OutputSchema" not in result["snapshot"]["Simulations"][0]
    assert "OutputSelections" not in result["snapshot"]["Simulations"][0]


@pytest.mark.parametrize("field,value", [("ApplicationType", []), ("DosingInterval", {})])
def test_malformed_administration_is_a_controlled_rejection(field: str, value: object) -> None:
    request = _request()
    request["protocol"][field] = value
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


def test_invalid_nested_unit_is_a_controlled_rejection() -> None:
    request = _request()
    request["compound"]["Parameters"][0]["Unit"] = []
    with pytest.raises(HumanModelSnapshotError, match="UNSUPPORTED_UNIT"):
        build_human_model_snapshot(**request)


def test_zero_explicit_clearance_does_not_claim_elimination_is_configured() -> None:
    request = _request()
    request["compound"]["Processes"][0]["Parameters"][1]["Value"] = 0.0
    result = build_human_model_snapshot(**request)
    assert result["manifest"]["clearance_mechanisms"] == ["renal"]
    assert result["manifest"]["elimination_configured"] is False


def _gfr_request():
    request = _request()
    request['compound']['Processes'] = [{
        'InternalName': 'GlomerularFiltration', 'Species': 'Human', 'DataSource': 'toy-filtration',
        'Parameters': [_parameter('GFR fraction', 1)],
    }]
    return request


@pytest.mark.parametrize('fraction', [0., .8, 1.])
def test_explicit_gfr_selects_native_filtration_and_preserves_sourced_fraction(fraction):
    request = _gfr_request()
    request['compound']['Processes'][0]['Parameters'][0]['Value'] = fraction
    before = copy.deepcopy(request)
    result = build_human_model_snapshot(**request)
    assert request == before
    assert result['snapshot']['Compounds'][0]['Processes'] == before['compound']['Processes']
    assert result['snapshot']['Simulations'][0]['Compounds'][0]['Processes'] == [{
        'Name': 'Glomerular Filtration-toy-filtration', 'SystemicProcessType': 'GFR',
    }]
    assert result['manifest']['clearance_mechanisms'] == ['renal']
    assert result['manifest']['elimination_configured'] is (fraction > 0)
    assert result['manifest']['compound_sources']['Processes'] == request['compound_sources']['Processes']
    assert result['manifest']['execution'] == {'solver_calls': 0, 'llm_calls': 0, 'gpu_jobs': 0}


@pytest.mark.parametrize('fraction', [-.1, 1.01, True, float('nan'), float('inf')])
def test_gfr_fraction_requires_finite_nonboolean_fraction(fraction):
    request = _gfr_request()
    request['compound']['Processes'][0]['Parameters'][0]['Value'] = fraction
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize('change', [
    lambda request: request['compound']['Processes'][0]['Parameters'][0].update(Unit='ml/min/kg'),
    lambda request: request['compound']['Processes'][0]['Parameters'].clear(),
    lambda request: request['compound']['Processes'][0]['Parameters'].append(_parameter('Plasma clearance', 1, 'ml/min/kg')),
    lambda request: request['compound']['Processes'][0].update(Species='Rat'),
    lambda request: request['compound_sources'].pop('Processes'),
    lambda request: request['compound']['Processes'].append(copy.deepcopy(request['compound']['Processes'][0])),
])
def test_gfr_rejects_units_missing_sources_or_duplicate_processes(change):
    request = _gfr_request()
    change(request)
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize('reverse', [False, True])
def test_gfr_and_systemic_renal_clearance_cannot_double_count(reverse):
    request = _gfr_request()
    request['compound']['Processes'].append(_request()['compound']['Processes'][0])
    if reverse:
        request['compound']['Processes'].reverse()
    with pytest.raises(HumanModelSnapshotError, match='OVERLAPPING_CLEARANCE'):
        build_human_model_snapshot(**request)


def test_individual_seed_is_required_to_avoid_native_tick_count_randomization() -> None:
    request = _request()
    del request["individual"]["Seed"]
    with pytest.raises(HumanModelSnapshotError) as error:
        build_human_model_snapshot(**request)
    assert error.value.missing_fields == ["individual.Seed"]


@pytest.mark.parametrize("seed", [None, True, -1, 2147483648, 1.5])
def test_individual_seed_requires_explicit_valid_int32(seed: object) -> None:
    request = _request()
    request["individual"]["Seed"] = seed
    with pytest.raises(HumanModelSnapshotError, match="INVALID_SEED"):
        build_human_model_snapshot(**request)


def test_all_supplied_alternatives_are_explicitly_selected_and_seed_is_recorded() -> None:
    result = build_human_model_snapshot(**_request(oral=True))
    selection = result["snapshot"]["Simulations"][0]["Compounds"][0]
    assert selection["Alternatives"] == [
        {"GroupName": "COMPOUND_LIPOPHILICITY", "AlternativeName": "toy-logp"},
        {"GroupName": "COMPOUND_FRACTION_UNBOUND", "AlternativeName": "toy-fu"},
        {"GroupName": "COMPOUND_SOLUBILITY", "AlternativeName": "toy-solubility"},
        {"GroupName": "COMPOUND_INTESTINAL_PERMEABILITY", "AlternativeName": "toy-permeability"},
    ]
    assert result["manifest"]["individual_seed"] == 17
    assert [item["SystemicProcessType"] for item in selection["Processes"]] == ["Renal"]


def _enzyme_request():
    request = _request()
    category = 'toy-profile'
    identifier = 'CYP1A2|Human|' + category
    request['individual']['ExpressionProfiles'] = [identifier]
    request['expression_profiles'] = [{
        'Type': 'Enzyme', 'Species': 'Human', 'Molecule': 'CYP1A2',
        'Category': category, 'Localization': 'Intracellular', 'Ontogeny': {'Name': 'CYP1A2'},
        'Parameters': [
            {'Path': 'CYP1A2|Reference concentration', 'Value': 1.8, 'Unit': 'µmol/l'},
            {'Path': 'CYP1A2|t1/2 (intestine)', 'Value': 23, 'Unit': 'h'},
            {'Path': 'CYP1A2|t1/2 (liver)', 'Value': 39, 'Unit': 'h'},
            {'Path': 'Organism|Liver|Pericentral|Intracellular|CYP1A2|Relative expression', 'Value': 1},
            {'Path': 'Organism|Liver|Periportal|Intracellular|CYP1A2|Relative expression', 'Value': 1},
        ],
    }]
    request['expression_profile_sources'] = {identifier: _source()}
    for branch, clearance in [('first', .1), ('second', .02)]:
        request['compound']['Processes'].append({
            'InternalName': 'MetabolizationIntrinsic_FirstOrder', 'Molecule': 'CYP1A2',
            'Species': 'Human', 'DataSource': branch,
            'Parameters': [_parameter('Intrinsic clearance', 0, 'l/min'),
                           _parameter('Specific clearance', clearance, '1/min')],
        })
    return request


def _cyp3a4_request():
    request = _gfr_request()
    identifier = 'CYP3A4|Human|toy-standard-profile'
    request['individual']['ExpressionProfiles'] = [identifier]
    containers = ['Brain', 'Gonads', 'Kidney', 'Liver|Pericentral', 'Liver|Periportal', 'Lung',
                  'SmallIntestine', 'SmallIntestine|Mucosa|Duodenum', 'SmallIntestine|Mucosa|LowerIleum',
                  'SmallIntestine|Mucosa|LowerJejunum', 'SmallIntestine|Mucosa|UpperIleum',
                  'SmallIntestine|Mucosa|UpperJejunum']
    request['expression_profiles'] = [{
        'Type': 'Enzyme', 'Species': 'Human', 'Molecule': 'CYP3A4', 'Category': 'toy-standard-profile',
        'Localization': 'Intracellular, BloodCellsIntracellular, VascEndosome', 'Ontogeny': {'Name': 'CYP3A4'},
        'Parameters': [
            {'Path': 'CYP3A4|Reference concentration', 'Value': 4.32, 'Unit': 'µmol/l'},
            {'Path': 'CYP3A4|t1/2 (intestine)', 'Value': 23., 'Unit': 'h'},
            {'Path': 'CYP3A4|t1/2 (liver)', 'Value': 36., 'Unit': 'h'},
            *[{'Path': f'Organism|{container}|Intracellular|CYP3A4|Relative expression', 'Value': .1}
              for container in containers],
        ],
    }]
    request['expression_profile_sources'] = {identifier: _source()}
    request['compound']['Processes'].append({
        'InternalName': 'MetabolizationIntrinsic_FirstOrder', 'Molecule': 'CYP3A4',
        'Species': 'Human', 'DataSource': 'toy-intrinsic',
        'Parameters': [_parameter('Intrinsic clearance', .504, 'l/min')],
    })
    return request


def test_cyp3a4_intrinsic_input_keeps_native_mode_and_complete_expression_profile():
    request = _cyp3a4_request()
    before = copy.deepcopy(request)
    result = build_human_model_snapshot(**request)
    assert request == before
    assert result['snapshot']['ExpressionProfiles'] == request['expression_profiles']
    assert len(result['snapshot']['ExpressionProfiles'][0]['Parameters']) == 15
    assert result['snapshot']['Simulations'][0]['Compounds'][0]['Processes'] == [
        {'Name': 'Glomerular Filtration-toy-filtration', 'SystemicProcessType': 'GFR'},
        {'Name': 'CYP3A4-toy-intrinsic', 'MoleculeName': 'CYP3A4'},
    ]
    assert result['snapshot']['Compounds'][0]['Processes'][1]['Parameters'] == [
        {'Name': 'Intrinsic clearance', 'Value': .504, 'Unit': 'l/min'}]
    assert result['manifest']['clearance_mechanisms'] == ['metabolism', 'renal']
    assert result['manifest']['elimination_configured'] is True


@pytest.mark.parametrize('clearance', [0., .504])
def test_cyp3a4_positive_intrinsic_clearance_controls_elimination_flag(clearance):
    request = _cyp3a4_request()
    request['compound']['Processes'][0]['Parameters'][0]['Value'] = 0
    request['compound']['Processes'][1]['Parameters'][0]['Value'] = clearance
    assert build_human_model_snapshot(**request)['manifest']['elimination_configured'] is (clearance > 0)


@pytest.mark.parametrize('change', [
    lambda r: r['expression_profiles'][0].update(Localization='UnregisteredLocation'),
    lambda r: r['expression_profiles'][0]['Parameters'][3].update(Path='Organism|Liver|Volume'),
    lambda r: r['expression_profiles'][0]['Parameters'].pop(0),
    lambda r: r['expression_profiles'][0]['Parameters'][0].update(Unit='mg/l'),
    lambda r: r['expression_profile_sources'].clear(),
    lambda r: r['compound']['Processes'][1].update(Molecule='CYP1A2'),
    lambda r: r['compound']['Processes'][1]['Parameters'][0].update(Unit='1/min'),
    lambda r: r['compound']['Processes'][1]['Parameters'][0].update(Value=-1),
    lambda r: r['compound']['Processes'][1]['Parameters'].append(_parameter('Km', .1, 'µmol/l')),
    lambda r: r['compound']['Processes'].append(copy.deepcopy(r['compound']['Processes'][1])),
])
def test_cyp3a4_rejects_unreviewed_profiles_parameters_and_unsupported_kinetics(change):
    request = _cyp3a4_request()
    change(request)
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize('molecule', ['UGT2B7', 'CYP2D6', 'SULT2A1'])
def test_source_backed_other_enzymes_keep_source_expression_and_downstream_clearance(molecule):
    request = _cyp3a4_request()
    profile = request['expression_profiles'][0]
    profile['Molecule'] = molecule
    profile['Ontogeny'] = {'Name': molecule}
    for parameter in profile['Parameters']:
        parameter['Path'] = parameter['Path'].replace('CYP3A4', molecule)
    identifier = molecule + '|Human|' + profile['Category']
    request['individual']['ExpressionProfiles'] = [identifier]
    request['expression_profile_sources'] = {identifier: _source()}
    process = request['compound']['Processes'][1]
    process['Molecule'] = molecule
    process['Parameters'].append(_parameter('Specific clearance', .002, '1/min'))
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Compounds'][0]['Processes'][1]['Parameters'] == process['Parameters']
    audit = result['manifest']['source_clearance_parameterization'][1]
    assert audit['explicit_downstream_input'] == 'Specific clearance'
    assert audit['native_derivation_runtime_verified'] is False


@pytest.mark.parametrize('internal,parameters', [
    ('rCYP450_FirstOrder', [_parameter('In vitro CL/recombinant enzyme', .058, 'µl/min/pmol rec. enzyme'),
                           _parameter('CLspec/[Enzyme]', .0212, 'l/µmol/min')]),
    ('MetabolizationLiverMicrosomes_FirstOrder', [_parameter('In vitro CL for liver microsomes', 36, 'µl/min/mg mic. protein')]),
    ('MetabolizationLiverMicrosomes_FirstOrder', [_parameter('In vitro CL for liver microsomes', .39, 'µl/min/mg mic. protein'),
                                                _parameter('Content of CYP proteins in liver microsomes', 96, 'pmol/mg mic. protein')]),
])
def test_reviewed_native_first_order_forms_preserve_types_and_source_units(internal, parameters):
    request = _cyp3a4_request()
    request['compound']['Processes'][1].update(InternalName=internal, Parameters=parameters)
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Compounds'][0]['Processes'][1] == request['compound']['Processes'][1]
    assert result['manifest']['elimination_configured'] is True
    wrong = copy.deepcopy(request)
    wrong['compound']['Processes'][1]['Parameters'][0]['Unit'] = 'l/min'
    with pytest.raises(HumanModelSnapshotError, match='UNSUPPORTED_UNIT'):
        build_human_model_snapshot(**wrong)


@pytest.mark.parametrize('internal', ['KidneyClearance', 'LiverClearance', 'BiliaryClearance'])
def test_systemic_specific_clearance_preserves_both_sourced_inputs_without_double_role(internal):
    request = _request()
    process = request['compound']['Processes'][0]
    process['InternalName'] = internal
    if internal != 'KidneyClearance':
        process['Parameters'].append(_parameter('Lipophilicity (experiment)', 1, 'Log Units'))
    process['Parameters'].append(_parameter('Specific clearance', .1, '1/h'))
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Compounds'][0]['Processes'][0] == process
    process['Parameters'][-1]['Value'] = 0
    assert build_human_model_snapshot(**request)['manifest']['elimination_configured'] is False


def test_multiple_complete_enzyme_profiles_require_each_to_be_used():
    request = _cyp3a4_request()
    extra = _enzyme_request()
    request['expression_profiles'] += extra['expression_profiles']
    request['expression_profile_sources'].update(extra['expression_profile_sources'])
    request['individual']['ExpressionProfiles'] += extra['individual']['ExpressionProfiles']
    request['compound']['Processes'] += extra['compound']['Processes'][1:]
    assert len(build_human_model_snapshot(**request)['snapshot']['ExpressionProfiles']) == 2
    request['compound']['Processes'] = request['compound']['Processes'][:2]
    with pytest.raises(HumanModelSnapshotError, match='UNUSED_EXPRESSION_PROFILE'):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize('internal,capacity', [
    ('rCYP450_MM', _parameter('In vitro Vmax/recombinant enzyme', 1, 'pmol/min/pmol rec. enzyme')),
    ('MetabolizationLiverMicrosomes_MM', _parameter('In vitro Vmax for liver microsomes', 1, 'nmol/min/mg mic. protein')),
    ('MetabolizationSpecific_MM', _parameter('Vmax', 1, 'pmol/ml/min')),
    ('MetabolizationIntrinsic_MM', _parameter('Vmax (liver tissue)', 1, 'pmol/min/g tissue')),
])
def test_sourced_native_mm_requires_capacity_km_and_preserves_original_type(internal, capacity):
    request = _cyp3a4_request()
    process = request['compound']['Processes'][1]
    process.update(InternalName=internal, Parameters=[capacity, _parameter('Km', 2, 'µmol/l')])
    process.pop('Species')  # This optional native DTO field is absent in source MM processes.
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Compounds'][0]['Processes'][1] == process
    assert result['manifest']['elimination_configured'] is True
    for index in (0, 1):
        invalid = copy.deepcopy(request)
        invalid['compound']['Processes'][1]['Parameters'].pop(index)
        with pytest.raises(HumanModelSnapshotError, match='MISSING_FIELDS'):
            build_human_model_snapshot(**invalid)
    process['Parameters'][1]['Unit'] = 'mg'
    with pytest.raises(HumanModelSnapshotError, match='UNSUPPORTED_UNIT'):
        build_human_model_snapshot(**request)


def test_sourced_noncompetitive_inhibition_is_preserved_without_claiming_clearance():
    request = _cyp3a4_request()
    process = request['compound']['Processes'][1]
    process.update(InternalName='NoncompetitiveInhibition', Parameters=[_parameter('Ki', 13.1, 'µmol/l')])
    process.pop('Species')
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['Compounds'][0]['Processes'][1] == process
    simulation = result['snapshot']['Simulations'][0]
    assert len(simulation['Compounds'][0]['Processes']) == 1
    assert simulation['Interactions'] == [{'Name': 'CYP3A4-toy-intrinsic', 'MoleculeName': 'CYP3A4',
                                           'CompoundName': 'ToyDrug'}]
    assert result['manifest']['clearance_mechanisms'] == ['renal']
    request['compound']['Processes'] = [process]
    assert build_human_model_snapshot(**request)['manifest']['elimination_configured'] is False


def test_specific_binding_requires_sourced_other_protein_and_is_not_elimination():
    request = _cyp3a4_request()
    profile = request['expression_profiles'][0]
    profile.update(Type='OtherProtein', Localization='Interstitial, BloodCellsMembrane, VascMembranePlasmaSide')
    process = request['compound']['Processes'][1]
    process.update(InternalName='SpecificBinding', Parameters=[_parameter('koff', 1, '1/min'), _parameter('Kd', 1, 'µmol/l')])
    request['compound']['Processes'] = [process]
    result = build_human_model_snapshot(**request)
    assert result['snapshot']['ExpressionProfiles'] == request['expression_profiles']
    assert result['manifest']['clearance_mechanisms'] == []
    assert result['manifest']['elimination_configured'] is False
    assert 'Interactions' not in result['snapshot']['Simulations'][0]
    assert len(result['snapshot']['Simulations'][0]['Compounds'][0]['Processes']) == 1
    process['Metabolite'] = 'not-supplied'
    with pytest.raises(HumanModelSnapshotError, match='UNBOUND_METABOLITE'):
        build_human_model_snapshot(**request)


@pytest.mark.parametrize('internal,parameters', [
    ('CompetitiveInhibition', [_parameter('Ki', 2, 'µmol/l')]),
    ('IrreversibleInhibition', [_parameter('kinact', .1, '1/min'), _parameter('K_kinact_half', 2, 'µmol/l')]),
    ('Induction', [_parameter('EC50', 2, 'µmol/l'), _parameter('Emax', 3)]),
])
def test_native_interactions_are_bound_at_simulation_level_without_mutating_drug(internal, parameters):
    request = _cyp3a4_request()
    request['compound']['Processes'][1].update(InternalName=internal, Parameters=parameters)
    before = copy.deepcopy(request)
    result = build_human_model_snapshot(**request)
    assert request == before
    assert result['snapshot']['Compounds'][0] == request['compound']
    simulation = result['snapshot']['Simulations'][0]
    assert simulation['Interactions'] == [{'Name': 'CYP3A4-toy-intrinsic', 'MoleculeName': 'CYP3A4',
                                           'CompoundName': 'ToyDrug'}]
    assert simulation['Compounds'][0]['Processes'] == [
        {'Name': 'Glomerular Filtration-toy-filtration', 'SystemicProcessType': 'GFR'}]


def test_missing_demographics_require_explicit_native_generation_policy():
    request = _request()
    del request['individual']['OriginData']['Weight']
    with pytest.raises(HumanModelSnapshotError, match='MISSING_FIELDS'):
        build_human_model_snapshot(**request)
    request['generated_demographics'] = ['Weight']
    result = build_human_model_snapshot(**request)
    assert 'Weight' not in result['snapshot']['Individuals'][0]['OriginData']
    assert result['manifest']['generated_demographics'] == {'Weight': 'to_be_generated_by_pksim_not_observed'}
    request['individual']['OriginData']['Weight'] = {'Value': 70, 'Unit': 'kg'}
    with pytest.raises(HumanModelSnapshotError, match='AMBIGUOUS_DEMOGRAPHICS'):
        build_human_model_snapshot(**request)


def test_sourced_first_order_enzyme_branches_preserve_native_selection():
    request = _enzyme_request()
    result = build_human_model_snapshot(**request)
    selected = result['snapshot']['Simulations'][0]['Compounds'][0]['Processes']
    assert {'MoleculeName': 'CYP1A2', 'Name': 'CYP1A2-first'} in selected
    assert {'MoleculeName': 'CYP1A2', 'Name': 'CYP1A2-second'} in selected
    assert result['manifest']['clearance_mechanisms'] == ['metabolism', 'renal']
    assert result['snapshot']['ExpressionProfiles'] == request['expression_profiles']


@pytest.mark.parametrize('change', ['missing_source', 'wrong_binding', 'anatomical_override', 'overlap'])
def test_enzyme_extension_rejects_unreviewed_or_ambiguous_inputs(change):
    request = _enzyme_request()
    if change == 'missing_source':
        request['expression_profile_sources'] = {}
    elif change == 'wrong_binding':
        request['individual']['ExpressionProfiles'] = ['CYP3A4|Human|other']
    elif change == 'anatomical_override':
        request['expression_profiles'][0]['Parameters'][0]['Path'] = 'Organism|Liver|Volume'
    else:
        request['compound']['Processes'].append({
            'InternalName': 'LiverClearance', 'Species': 'Human', 'DataSource': 'toy-overlap',
            'Parameters': [_parameter('Plasma clearance', 1, 'ml/min/kg'),
                           _parameter('Fraction unbound (experiment)', .5),
                           _parameter('Lipophilicity (experiment)', 1, 'Log Units')],
        })
    with pytest.raises(HumanModelSnapshotError):
        build_human_model_snapshot(**request)
