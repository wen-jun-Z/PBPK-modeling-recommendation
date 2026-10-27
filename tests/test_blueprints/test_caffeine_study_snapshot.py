from __future__ import annotations

import copy
from datetime import UTC, datetime

from pbpk_kb.blueprints.caffeine_dataset import (
    ObservationPoint,
    ObservationProvenance,
    PKMetric,
)
from pbpk_kb.blueprints.caffeine_parameters import (
    CALIBRATION_DATASETS,
    CALIBRATION_OBJECTIVE,
    CaffeineParameterManifest,
    CaffeineParameterRecord,
    ParameterProvenance,
    ParameterSourceArtifact,
)
from pbpk_kb.blueprints.caffeine_study_snapshot import (
    RENAL_PROCESS_SELECTION_NAME,
    _normalize_snapshot_numeric_serialization,
    frozen_individuals,
    frozen_protocols,
    frozen_simulations,
    transform_study_snapshot,
)
from pbpk_kb.model_assets.execution_plan import canonical_sha256


PARAMETER_VALUES = {
    "caffeine_molecular_weight": (194.19, "g/mol"),
    "caffeine_pka_base": (0.7, "dimensionless"),
    "caffeine_water_solubility_25c": (21600.0, "mg/l"),
    "caffeine_fraction_unbound_plasma": (0.68, "fraction"),
    "caffeine_renal_plasma_clearance": (0.010330512664107847, "ml/min/kg"),
    "caffeine_cyp1a2_km_n3_demethylation": (500.0, "umol/l"),
    "caffeine_cyp1a2_vmax_effective_n3": (
        250.0,
        "pmol/min/mg mic. protein",
    ),
    "caffeine_effective_lipophilicity": (-0.1, "Log Units"),
}


def _parameter_manifest() -> CaffeineParameterManifest:
    provenance = [
        ParameterProvenance(
            artifact_id="source",
            source_location="fixture:field",
            extraction_method="structured_json",
            source_value_text="fixture",
            source_value_text_sha256="a" * 64,
        )
    ]
    parameters: list[CaffeineParameterRecord] = []
    for parameter_id, (value, unit) in PARAMETER_VALUES.items():
        estimated = parameter_id in {
            "caffeine_cyp1a2_vmax_effective_n3",
            "caffeine_effective_lipophilicity",
        }
        if parameter_id == "caffeine_cyp1a2_vmax_effective_n3":
            bounds = (115.0, 450.0)
        elif parameter_id == "caffeine_effective_lipophilicity":
            bounds = (-1.1, 0.9)
        else:
            bounds = (None, None)
        parameters.append(
            CaffeineParameterRecord(
                parameter_id=parameter_id,
                name=parameter_id,
                pksim_path=f"Compounds|Caffeine|{parameter_id}",
                origin="estimated" if estimated else "observed",
                value=value,
                unit=unit,
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                calibrate=estimated,
                calibration_dataset_ids=(
                    CALIBRATION_DATASETS if estimated else []
                ),
                objective_function=(CALIBRATION_OBJECTIVE if estimated else None),
                provenance=provenance,
            )
        )
    artifacts = [
        ParameterSourceArtifact(
            artifact_id=("caffeine_snapshot_v2" if index == 0 else f"source_{index}"),
            role="fixture",
            source_url="https://example.test/source",
            source_location=f"fixture/source_{index}",
            sha256=f"{index + 1:064x}"[-64:],
            byte_size=1,
            license="test",
            redistribution="public_reference",
        )
        for index in range(10)
    ]
    draft = CaffeineParameterManifest(
        generated_at=datetime.now(UTC),
        dataset_manifest_checksum="d" * 64,
        source_artifacts=artifacts,
        parameters=parameters,
        structural_assumptions=["a", "b", "c"],
        checks={"fixture": True},
        accepted=True,
        checksum="0" * 64,
    )
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def _source_snapshot() -> dict:
    return {
        "Version": 80,
        "Compounds": [
            {
                "Name": "Caffeine",
                "Parameters": [
                    {"Name": "Molecular weight", "Value": 194.2, "Unit": "g/mol"}
                ],
                "PkaTypes": [{"Type": "Base", "Pka": 0.8}],
                "Solubility": [
                    {
                        "Name": "Water solubility",
                        "Parameters": [
                            {
                                "Name": "Solubility at reference pH",
                                "Value": 1.0,
                                "Unit": "mg/l",
                            },
                            {"Name": "Reference pH", "Value": 7.0},
                        ],
                    }
                ],
                "FractionUnbound": [
                    {
                        "Name": "old",
                        "Species": "Human",
                        "Parameters": [
                            {
                                "Name": "Fraction unbound (plasma, reference value)",
                                "Value": 0.7,
                            }
                        ],
                    }
                ],
                "Lipophilicity": [],
                "IntestinalPermeability": [{"Name": "fitted"}],
                "Processes": [
                    {
                        "InternalName": "MetabolizationLiverMicrosomes_MM",
                        "DataSource": "MM",
                        "Molecule": "CYP1A2",
                        "Parameters": [
                            {
                                "Name": "In vitro Vmax for liver microsomes",
                                "Value": 109.0,
                            },
                            {"Name": "Km", "Value": 14.7},
                        ],
                    },
                    {
                        "InternalName": "KidneyClearance",
                        "DataSource": "old",
                        "Species": "Human",
                        "Parameters": [
                            {"Name": "Fraction unbound (experiment)", "Value": 0.7},
                            {"Name": "Plasma clearance", "Value": 0.1},
                        ],
                    },
                ],
            }
        ],
        "ExpressionProfiles": [
            {
                "Type": "Enzyme",
                "Species": "Human",
                "Molecule": "CYP3A4",
                "Category": "old",
            },
            {
                "Type": "Enzyme",
                "Species": "Human",
                "Molecule": "CYP1A2",
                "Category": "PBPK-KB-Caffeine-Adult-v1",
            },
        ],
        "Individuals": [{"Name": "old"}],
        "Formulations": [{"Name": "old"}],
        "Protocols": [{"Name": "old"}],
        "Simulations": [{"Name": "old"}],
    }


def _observations() -> list[ObservationPoint]:
    datasets = [
        ("blanchard_1983_iv_5mgkg", "calibration", 17, "intravenous_infusion"),
        ("blanchard_1983_oral_5mgkg", "calibration", 11, "oral"),
        ("wahllander_1989_iv_3mgkg", "internal_holdout", 7, "intravenous_infusion"),
        (
            "eudract_2011_002291_16_oral_50mg",
            "external_validation",
            9,
            "oral",
        ),
        ("pmc5350996_synthetic_200mg", "external_validation", 6, "oral"),
    ]
    rows: list[ObservationPoint] = []
    index = 0
    for dataset_id, split, count, route in datasets:
        for row in range(count):
            excluded = dataset_id.startswith("eudract") and row == 0
            dose_value = 5.0
            dose_unit = "mg/kg"
            if dataset_id.startswith("wahllander"):
                dose_value = 3.0
            elif dataset_id.startswith("eudract"):
                dose_value, dose_unit = 50.0, "mg"
            elif dataset_id.startswith("pmc535"):
                dose_value, dose_unit = 200.0, "mg"
            rows.append(
                ObservationPoint(
                    observation_id=f"obs_{index:024x}",
                    dataset_id=dataset_id,
                    study_id=dataset_id,
                    split=split,
                    matrix="plasma",
                    route=route,
                    formulation="fixture",
                    dose_value=dose_value,
                    dose_unit=dose_unit,
                    infusion_duration_min=(
                        30.0
                        if dataset_id.startswith("blanchard_1983_iv")
                        else 10.0
                        if route == "intravenous_infusion"
                        else None
                    ),
                    time_min=float(row * 30),
                    source_value=1.0,
                    source_unit="umol/l",
                    canonical_value_umol_per_l=1.0,
                    statistic="arithmetic_mean",
                    subject_count=10,
                    use_in_objective=not excluded,
                    exclusion_reason=("predose" if excluded else None),
                    provenance=ObservationProvenance(
                        artifact_id="fixture",
                        source_location="fixture",
                        row_index=row,
                    ),
                )
            )
            index += 1
    return rows


def _metrics() -> list[PKMetric]:
    endpoints = ["auc_0_t", "auc_0_inf", "cmax", "tmax", "half_life"]
    return [
        PKMetric(
            metric_id=f"metric_{index:024x}",
            dataset_id="pmc5724581_beverage_50_65mg",
            study_id="pmc5724581_caffeinated_beverage",
            split="external_validation_secondary",
            endpoint=endpoint,
            value=1.0,
            unit="fixture",
            statistic=("median" if endpoint == "tmax" else "geometric_mean"),
            subject_count=16,
            provenance=ObservationProvenance(
                artifact_id="fixture",
                source_location="fixture",
                row_index=index,
            ),
        )
        for index, endpoint in enumerate(endpoints)
    ]


def test_frozen_scenario_contract_is_exact() -> None:
    individuals = frozen_individuals()
    protocols = frozen_protocols()
    simulations = frozen_simulations(_observations(), _metrics())

    assert len(individuals) == 7
    assert individuals[0].seed == individuals[1].seed
    assert len(protocols) == 6
    assert len(simulations) == 7
    assert sum(item.observation_count for item in simulations) == 50
    assert sum(
        item.metric_count * item.aggregation_weight
        for item in simulations
        if item.split == "external_validation_secondary"
    ) == 5
    assert {
        item.dataset_id
        for item in simulations
        if item.use_in_calibration_objective
    } == set(CALIBRATION_DATASETS)


def test_protocols_preserve_true_infusion_and_declared_capsule_assumption() -> None:
    protocols = {item.name: item for item in frozen_protocols()}

    assert protocols["PROTO_CAL_BLANCHARD_IV"].application_type == "Intravenous"
    assert protocols["PROTO_CAL_BLANCHARD_IV"].infusion_duration_min == 30.0
    assert protocols["PROTO_HOLD_WAHLLANDER_IV"].infusion_duration_min == 10.0
    assert "CAPSULE_DISSOLUTION_DATA_MISSING" in protocols[
        "PROTO_EXT_EUDRACT_PO"
    ].assumption_codes
    assert all(
        item.formulation_type == "Formulation_Dissolved"
        for item in protocols.values()
        if item.application_type == "Oral"
    )


def test_transform_applies_locked_parameters_and_replaces_probe_scenarios() -> None:
    source = _source_snapshot()
    original = copy.deepcopy(source)
    transformed = transform_study_snapshot(
        source_snapshot=source,
        parameter_manifest=_parameter_manifest(),
        individuals=frozen_individuals(),
        protocols=frozen_protocols(),
        simulations=frozen_simulations(_observations(), _metrics()),
    )

    assert source == original
    compound = transformed["Compounds"][0]
    assert "IntestinalPermeability" not in compound
    assert compound["Parameters"][0]["Value"] == 194.19
    assert compound["PkaTypes"][0]["Pka"] == 0.7
    assert compound["Lipophilicity"][0]["Parameters"][0]["Value"] == -0.1
    cyp = next(
        item
        for item in compound["Processes"]
        if item["InternalName"] == "MetabolizationLiverMicrosomes_MM"
    )
    assert {item["Name"]: item["Value"] for item in cyp["Parameters"]} == {
        "In vitro Vmax for liver microsomes": 250.0,
        "Km": 500.0,
    }
    assert len(transformed["Individuals"]) == 7
    assert len(transformed["Protocols"]) == 6
    assert {item["Name"] for item in transformed["Simulations"]} == {
        "CAL_BLANCHARD_IV",
        "CAL_BLANCHARD_PO",
        "HOLD_WAHLLANDER_IV",
        "EXT_EUDRACT_PO",
        "EXT_PMC535_PO",
        "EXT2_PMC572_PO_MALE",
        "EXT2_PMC572_PO_FEMALE",
    }
    renal = next(
        item
        for item in compound["Processes"]
        if item["InternalName"] == "KidneyClearance"
    )
    expected_renal_selection = f"Renal Clearances-{renal['DataSource']}"
    assert expected_renal_selection == RENAL_PROCESS_SELECTION_NAME
    assert all(
        expected_renal_selection
        in {
            process["Name"]
            for process in simulation["Compounds"][0]["Processes"]
        }
        for simulation in transformed["Simulations"]
    )
    individuals = {
        item["Name"]: item for item in transformed["Individuals"]
    }
    for name in ("IND_EXT2_PMC572_PO_MALE", "IND_EXT2_PMC572_PO_FEMALE"):
        assert "BMI" not in individuals[name]["OriginData"]
    simulations = {item["Name"]: item for item in transformed["Simulations"]}
    assert simulations["HOLD_WAHLLANDER_IV"]["Parameters"] == [
        {
            "Path": (
                "Events|PROTO_HOLD_WAHLLANDER_IV|Application_1|"
                "ProtocolSchemaItem|Infusion time"
            ),
            "Value": 0.1666666667,
            "Unit": "h",
        }
    ]
    assert simulations["EXT_PMC535_PO"]["Parameters"] == [
        {
            "Path": (
                "Events|PROTO_EXT_PMC535_PO|FORM_PMC535_LIQUID_DISSOLVED|"
                "Application_1|ProtocolSchemaItem|Volume of water/body weight"
            ),
            "Value": 3.5,
            "Unit": "ml/kg",
        }
    ]
    assert all(
        "Alternatives" not in simulation["Compounds"][0]
        for simulation in transformed["Simulations"]
    )
    assert transformed["ExpressionProfiles"][0]["Molecule"] == "CYP1A2"


def test_snapshot_numeric_normalization_is_strict_and_audited() -> None:
    notes: list[str] = []
    normalized = _normalize_snapshot_numeric_serialization(
        {"value": 0.010330512664107847, "dose": 5.0},
        {"value": 0.0103305127, "dose": 5.01},
        notes=notes,
    )

    assert normalized["value"] == 0.0103305127
    assert normalized["dose"] == 5.0
    assert notes == [
        "pksim_snapshot_numeric_serialization: "
        "/value 0.010330512664107847 -> 0.0103305127"
    ]
