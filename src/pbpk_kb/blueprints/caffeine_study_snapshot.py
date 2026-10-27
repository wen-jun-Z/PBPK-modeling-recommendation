from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from pbpk_kb.blueprints.caffeine_dataset import (
    CaffeineDatasetManifest,
    ObservationPoint,
    PKMetric,
)
from pbpk_kb.blueprints.caffeine_parameters import (
    CALIBRATION_DATASETS,
    CaffeineParameterManifest,
)
from pbpk_kb.blueprints.pksim_snapshot_builder import (
    _diff_paths,
    _semantic_projection,
)
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


STUDY_SNAPSHOT_BUILDER_VERSION = "caffeine-study-snapshot-builder-1.2.0"
STUDY_SCENARIO_VERSION = "caffeine_study_scenarios_v3"
CAFFEINE_OUTPUT_PATH = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
CYP1A2_PROFILE_NAME = "CYP1A2|Human|PBPK-KB-Caffeine-Adult-v1"
CALCULATION_METHODS = [
    "Cellular partition coefficient method - PK-Sim Standard",
    "Cellular permeability - PK-Sim Standard",
]
RENAL_PROCESS_DATA_SOURCE = "Birkett 1991 Table 1 rederived PBPK-KB-v1"
RENAL_PROCESS_SELECTION_NAME = f"Renal Clearances-{RENAL_PROCESS_DATA_SOURCE}"


class CaffeineStudySnapshotError(RuntimeError):
    pass


class StudyInputProvenance(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class StudyIndividualSpec(StrictBlueprintModel):
    name: str = Field(min_length=1)
    seed_key: str = Field(min_length=1)
    seed: int = Field(gt=0)
    gender: Literal["MALE", "FEMALE"]
    age_years: float = Field(gt=0)
    weight_kg: float | None = Field(default=None, gt=0)
    height_m: float | None = Field(default=None, gt=0)
    reported_bmi_kg_per_m2: float | None = Field(default=None, gt=0)
    context_only_fields: list[Literal["bmi"]] = Field(default_factory=list)
    population: Literal["European_ICRP_2002"] = "European_ICRP_2002"
    system_generated_fields: list[Literal["weight", "height"]]
    provenance: list[StudyInputProvenance] = Field(min_length=1)

    @model_validator(mode="after")
    def generated_fields_are_missing_inputs(self) -> "StudyIndividualSpec":
        expected: set[str] = set()
        if self.weight_kg is None:
            expected.add("weight")
        if self.height_m is None:
            expected.add("height")
        if set(self.system_generated_fields) != expected:
            raise ValueError(
                "System-generated anthropometry fields do not match missing inputs."
            )
        expected_context = {"bmi"} if self.reported_bmi_kg_per_m2 else set()
        if set(self.context_only_fields) != expected_context:
            raise ValueError(
                "Context-only fields do not match reported non-executable inputs."
            )
        return self


class StudyProtocolSpec(StrictBlueprintModel):
    name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    application_type: Literal["Intravenous", "Oral"]
    dose_value: float = Field(gt=0)
    dose_unit: Literal["mg", "mg/kg"]
    infusion_duration_min: float | None = Field(default=None, gt=0)
    formulation_name: str | None = None
    formulation_type: Literal["Formulation_Dissolved"] | None = None
    assumption_codes: list[str] = Field(default_factory=list)
    provenance: list[StudyInputProvenance] = Field(min_length=1)

    @model_validator(mode="after")
    def route_contract(self) -> "StudyProtocolSpec":
        if self.application_type == "Intravenous":
            if self.infusion_duration_min is None:
                raise ValueError("Intravenous scenarios require infusion duration.")
            if self.formulation_name is not None or self.formulation_type is not None:
                raise ValueError("IV scenarios cannot select an oral formulation.")
        else:
            if self.infusion_duration_min is not None:
                raise ValueError("Oral scenarios cannot define infusion duration.")
            if not self.formulation_name or not self.formulation_type:
                raise ValueError("Oral scenarios require a formulation.")
        return self


class OutputIntervalSpec(StrictBlueprintModel):
    start_h: float = Field(ge=0)
    end_h: float = Field(gt=0)
    resolution_points_per_h: float = Field(gt=0)

    @model_validator(mode="after")
    def end_after_start(self) -> "OutputIntervalSpec":
        if self.end_h <= self.start_h:
            raise ValueError("Output interval end must be after start.")
        return self


class StudySimulationSpec(StrictBlueprintModel):
    name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: Literal[
        "calibration",
        "internal_holdout",
        "external_validation",
        "external_validation_secondary",
    ]
    individual_name: str = Field(min_length=1)
    protocol_name: str = Field(min_length=1)
    aggregation_weight: float = Field(gt=0, le=1)
    observation_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    max_observation_time_h: float | None = Field(default=None, ge=0)
    output_intervals: list[OutputIntervalSpec] = Field(min_length=1)
    use_in_calibration_objective: bool

    @model_validator(mode="after")
    def split_contract(self) -> "StudySimulationSpec":
        if self.use_in_calibration_objective != (self.split == "calibration"):
            raise ValueError("Only calibration scenarios may enter the objective.")
        if self.split == "external_validation_secondary":
            if self.observation_count != 0 or self.metric_count <= 0:
                raise ValueError("Secondary validation must map metrics, not points.")
        elif self.observation_count <= 0 or self.metric_count != 0:
            raise ValueError("Primary scenarios must map concentration observations.")
        if self.max_observation_time_h is not None:
            if self.output_intervals[-1].end_h < self.max_observation_time_h:
                raise ValueError("Output grid does not cover the last observation.")
        return self


class CaffeineStudyScenarioManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[STUDY_SNAPSHOT_BUILDER_VERSION] = (
        STUDY_SNAPSHOT_BUILDER_VERSION
    )
    scenario_version: Literal[STUDY_SCENARIO_VERSION] = STUDY_SCENARIO_VERSION
    generated_at: datetime
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_semantic_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    dataset_version: Literal["caffeine_dataset_v3"] = "caffeine_dataset_v3"
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_version: Literal["caffeine_parameter_manifest_v1"] = (
        "caffeine_parameter_manifest_v1"
    )
    parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    individuals: list[StudyIndividualSpec] = Field(min_length=7, max_length=7)
    protocols: list[StudyProtocolSpec] = Field(min_length=6, max_length=6)
    simulations: list[StudySimulationSpec] = Field(min_length=7, max_length=7)
    system_assumptions: list[str] = Field(min_length=4)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def references_are_exact(self) -> "CaffeineStudyScenarioManifest":
        individual_names = [item.name for item in self.individuals]
        protocol_names = [item.name for item in self.protocols]
        simulation_names = [item.name for item in self.simulations]
        if len(set(individual_names)) != 7 or len(set(simulation_names)) != 7:
            raise ValueError("Individuals and simulations must have unique names.")
        if len(set(protocol_names)) != 6:
            raise ValueError("Protocols must have unique names.")
        if {item.individual_name for item in self.simulations} != set(
            individual_names
        ):
            raise ValueError("Every individual must be used by one simulation.")
        if {item.protocol_name for item in self.simulations} != set(protocol_names):
            raise ValueError("Every protocol must be used by a simulation.")
        return self


class CaffeineStudyRoundTripAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[STUDY_SNAPSHOT_BUILDER_VERSION] = (
        STUDY_SNAPSHOT_BUILDER_VERSION
    )
    generated_at: datetime
    scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_differences: list[str]
    accepted_normalizations: list[str]
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _seed(value: str) -> int:
    seed = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
    return seed % 2_147_483_647 or 1


def _provenance(
    artifact_id: str,
    source_location: str,
    fields: list[str],
    interpretation: str,
) -> list[StudyInputProvenance]:
    return [
        StudyInputProvenance(
            artifact_id=artifact_id,
            source_location=source_location,
            fields=fields,
            interpretation=interpretation,
        )
    ]


def frozen_individuals() -> list[StudyIndividualSpec]:
    blanchard_seed_key = "blanchard_1983_absolute_bioavailability"
    return [
        StudyIndividualSpec(
            name="IND_CAL_BLANCHARD_IV",
            seed_key=blanchard_seed_key,
            seed=_seed(blanchard_seed_key),
            gender="MALE",
            age_years=21.8,
            system_generated_fields=["weight", "height"],
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3, Blanchard 1983 population",
                ["gender", "age"],
                "Weight and height were not reported and remain PK-Sim generated.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_CAL_BLANCHARD_PO",
            seed_key=blanchard_seed_key,
            seed=_seed(blanchard_seed_key),
            gender="MALE",
            age_years=21.8,
            system_generated_fields=["weight", "height"],
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3, Blanchard 1983 population",
                ["gender", "age"],
                "Crossover arms use identical demographics and deterministic seed.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_HOLD_WAHLLANDER_IV",
            seed_key="wahllander_1989_caffeine_iv",
            seed=_seed("wahllander_1989_caffeine_iv"),
            gender="MALE",
            age_years=26.0,
            weight_kg=74.0,
            system_generated_fields=["height"],
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3, Wahllander 1989 population",
                ["gender", "age", "weight"],
                "Height was not reported and remains PK-Sim generated.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_EXT_EUDRACT_PO",
            seed_key="eudract_2011_002291_16_base_model",
            seed=_seed("eudract_2011_002291_16_base_model"),
            gender="MALE",
            age_years=30.0,
            weight_kg=73.0,
            height_m=1.76,
            system_generated_fields=[],
            provenance=_provenance(
                "pmc8302243_nxml",
                "Methods, PBPK model personalization, base model individual",
                ["gender", "age", "height", "weight"],
                "Exact base-model demographics reported by the OA article.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_EXT_PMC535_PO",
            seed_key="pmc5350996_synthetic_caffeine",
            seed=_seed("pmc5350996_synthetic_caffeine"),
            gender="MALE",
            age_years=24.33,
            weight_kg=78.91,
            system_generated_fields=["height"],
            provenance=_provenance(
                "pmc5350996_nxml",
                "Methods/Subjects, arithmetic mean demographics",
                ["gender", "age", "weight"],
                "Height was not reported and remains PK-Sim generated.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_EXT2_PMC572_PO_MALE",
            seed_key="pmc5724581_beverage_male",
            seed=_seed("pmc5724581_beverage_male"),
            gender="MALE",
            age_years=35.0,
            reported_bmi_kg_per_m2=26.0,
            context_only_fields=["bmi"],
            system_generated_fields=["weight", "height"],
            provenance=_provenance(
                "pmc5724581_nxml",
                "Results/Demographics, 8 males and 8 females",
                ["gender proportion", "mean age", "mean BMI"],
                "Mean BMI is context only; PK-Sim generates weight and height.",
            ),
        ),
        StudyIndividualSpec(
            name="IND_EXT2_PMC572_PO_FEMALE",
            seed_key="pmc5724581_beverage_female",
            seed=_seed("pmc5724581_beverage_female"),
            gender="FEMALE",
            age_years=35.0,
            reported_bmi_kg_per_m2=26.0,
            context_only_fields=["bmi"],
            system_generated_fields=["weight", "height"],
            provenance=_provenance(
                "pmc5724581_nxml",
                "Results/Demographics, 8 males and 8 females",
                ["gender proportion", "mean age", "mean BMI"],
                "Mean BMI is context only; PK-Sim generates weight and height.",
            ),
        ),
    ]


def frozen_protocols() -> list[StudyProtocolSpec]:
    return [
        StudyProtocolSpec(
            name="PROTO_CAL_BLANCHARD_IV",
            dataset_id="blanchard_1983_iv_5mgkg",
            application_type="Intravenous",
            dose_value=5.0,
            dose_unit="mg/kg",
            infusion_duration_min=30.0,
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3 and PMID 6832208",
                ["dose", "route", "infusion duration"],
                "5 mg/kg resolves the repository filename discrepancy.",
            ),
        ),
        StudyProtocolSpec(
            name="PROTO_CAL_BLANCHARD_PO",
            dataset_id="blanchard_1983_oral_5mgkg",
            application_type="Oral",
            dose_value=5.0,
            dose_unit="mg/kg",
            formulation_name="FORM_BLANCHARD_AQUEOUS_DISSOLVED",
            formulation_type="Formulation_Dissolved",
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3 and PMID 6832208",
                ["dose", "route", "aqueous solution"],
                "Aqueous oral solution maps to PK-Sim dissolved formulation.",
            ),
        ),
        StudyProtocolSpec(
            name="PROTO_HOLD_WAHLLANDER_IV",
            dataset_id="wahllander_1989_iv_3mgkg",
            application_type="Intravenous",
            dose_value=3.0,
            dose_unit="mg/kg",
            infusion_duration_min=10.0,
            provenance=_provenance(
                "pmc8302243_supinfo_s2_docx",
                "Supinfo S2 Table S3, Wahllander 1989",
                ["dose", "route", "infusion duration"],
                "Internal holdout protocol; never used for estimation.",
            ),
        ),
        StudyProtocolSpec(
            name="PROTO_EXT_EUDRACT_PO",
            dataset_id="eudract_2011_002291_16_oral_50mg",
            application_type="Oral",
            dose_value=50.0,
            dose_unit="mg",
            formulation_name="FORM_EUDRACT_CAPSULE_DISSOLVED_ASSUMPTION",
            formulation_type="Formulation_Dissolved",
            assumption_codes=["CAPSULE_DISSOLUTION_DATA_MISSING"],
            provenance=_provenance(
                "pmc8302243_nxml",
                "Clinical study description, 50 mg trimethyl-13C3 caffeine capsule",
                ["dose", "route", "gelatin capsule"],
                "Dissolved immediate-release structure; no dissolution values invented.",
            ),
        ),
        StudyProtocolSpec(
            name="PROTO_EXT_PMC535_PO",
            dataset_id="pmc5350996_synthetic_200mg",
            application_type="Oral",
            dose_value=200.0,
            dose_unit="mg",
            formulation_name="FORM_PMC535_LIQUID_DISSOLVED",
            formulation_type="Formulation_Dissolved",
            provenance=_provenance(
                "pmc5350996_nxml",
                "Methods, synthetic caffeine treatment consumed within 5 min",
                ["dose", "route", "liquid formulation"],
                "Oral liquid maps to PK-Sim dissolved formulation.",
            ),
        ),
        StudyProtocolSpec(
            name="PROTO_EXT2_PMC572_PO",
            dataset_id="pmc5724581_beverage_50_65mg",
            application_type="Oral",
            dose_value=50.65,
            dose_unit="mg",
            formulation_name="FORM_PMC572_BEVERAGE_DISSOLVED",
            formulation_type="Formulation_Dissolved",
            provenance=_provenance(
                "pmc5724581_nxml",
                "Table 1, caffeinated beverage actual mean dose",
                ["actual mean dose", "route", "beverage formulation"],
                "Secondary validation only; oral beverage is dissolved.",
            ),
        ),
    ]


def _output_intervals(end_h: float) -> list[OutputIntervalSpec]:
    intervals = [
        OutputIntervalSpec(
            start_h=0.0,
            end_h=2.0,
            resolution_points_per_h=60.0,
        ),
        OutputIntervalSpec(
            start_h=2.0,
            end_h=24.0,
            resolution_points_per_h=10.0,
        ),
    ]
    if end_h > 24.0:
        intervals.append(
            OutputIntervalSpec(
                start_h=24.0,
                end_h=end_h,
                resolution_points_per_h=4.0,
            )
        )
    return intervals


_SIMULATION_INPUTS = [
    (
        "CAL_BLANCHARD_IV",
        "blanchard_1983_iv_5mgkg",
        "calibration",
        "IND_CAL_BLANCHARD_IV",
        "PROTO_CAL_BLANCHARD_IV",
        1.0,
        24.0,
    ),
    (
        "CAL_BLANCHARD_PO",
        "blanchard_1983_oral_5mgkg",
        "calibration",
        "IND_CAL_BLANCHARD_PO",
        "PROTO_CAL_BLANCHARD_PO",
        1.0,
        24.0,
    ),
    (
        "HOLD_WAHLLANDER_IV",
        "wahllander_1989_iv_3mgkg",
        "internal_holdout",
        "IND_HOLD_WAHLLANDER_IV",
        "PROTO_HOLD_WAHLLANDER_IV",
        1.0,
        24.0,
    ),
    (
        "EXT_EUDRACT_PO",
        "eudract_2011_002291_16_oral_50mg",
        "external_validation",
        "IND_EXT_EUDRACT_PO",
        "PROTO_EXT_EUDRACT_PO",
        1.0,
        24.0,
    ),
    (
        "EXT_PMC535_PO",
        "pmc5350996_synthetic_200mg",
        "external_validation",
        "IND_EXT_PMC535_PO",
        "PROTO_EXT_PMC535_PO",
        1.0,
        24.0,
    ),
    (
        "EXT2_PMC572_PO_MALE",
        "pmc5724581_beverage_50_65mg",
        "external_validation_secondary",
        "IND_EXT2_PMC572_PO_MALE",
        "PROTO_EXT2_PMC572_PO",
        0.5,
        48.0,
    ),
    (
        "EXT2_PMC572_PO_FEMALE",
        "pmc5724581_beverage_50_65mg",
        "external_validation_secondary",
        "IND_EXT2_PMC572_PO_FEMALE",
        "PROTO_EXT2_PMC572_PO",
        0.5,
        48.0,
    ),
]


def frozen_simulations(
    observations: list[ObservationPoint],
    metrics: list[PKMetric],
) -> list[StudySimulationSpec]:
    result: list[StudySimulationSpec] = []
    for (
        name,
        dataset_id,
        split,
        individual,
        protocol,
        aggregation_weight,
        output_end_h,
    ) in _SIMULATION_INPUTS:
        dataset_observations = [
            item for item in observations if item.dataset_id == dataset_id
        ]
        dataset_metrics = [item for item in metrics if item.dataset_id == dataset_id]
        result.append(
            StudySimulationSpec(
                name=name,
                dataset_id=dataset_id,
                split=split,
                individual_name=individual,
                protocol_name=protocol,
                aggregation_weight=aggregation_weight,
                observation_count=len(dataset_observations),
                metric_count=len(dataset_metrics),
                max_observation_time_h=(
                    max(item.time_min for item in dataset_observations) / 60.0
                    if dataset_observations
                    else None
                ),
                output_intervals=_output_intervals(output_end_h),
                use_in_calibration_objective=split == "calibration",
            )
        )
    return result


def _find_unique(
    values: list[dict[str, Any]],
    *,
    key: str,
    expected: str,
    kind: str,
) -> dict[str, Any]:
    matches = [item for item in values if item.get(key) == expected]
    if len(matches) != 1:
        raise CaffeineStudySnapshotError(
            f"Expected one {kind} {expected!r}; found {len(matches)}."
        )
    return matches[0]


def _set_named_parameter(
    values: list[dict[str, Any]],
    *,
    name: str,
    value: float,
    unit: str | None = None,
) -> None:
    parameter = _find_unique(values, key="Name", expected=name, kind="parameter")
    parameter["Value"] = value
    if unit is not None:
        parameter["Unit"] = unit


def _parameter_values(manifest: CaffeineParameterManifest) -> dict[str, float]:
    values = {item.parameter_id: item.value for item in manifest.parameters}
    expected = {
        "caffeine_molecular_weight",
        "caffeine_pka_base",
        "caffeine_water_solubility_25c",
        "caffeine_fraction_unbound_plasma",
        "caffeine_renal_plasma_clearance",
        "caffeine_cyp1a2_km_n3_demethylation",
        "caffeine_cyp1a2_vmax_effective_n3",
        "caffeine_effective_lipophilicity",
    }
    if set(values) != expected:
        raise CaffeineStudySnapshotError("Parameter manifest IDs drifted.")
    return values


def _apply_compound_parameters(
    compound: dict[str, Any],
    manifest: CaffeineParameterManifest,
) -> None:
    values = _parameter_values(manifest)
    _set_named_parameter(
        compound.get("Parameters", []),
        name="Molecular weight",
        value=values["caffeine_molecular_weight"],
        unit="g/mol",
    )
    pka = _find_unique(
        compound.get("PkaTypes", []),
        key="Type",
        expected="Base",
        kind="pKa type",
    )
    pka["Pka"] = values["caffeine_pka_base"]

    solubility = _find_unique(
        compound.get("Solubility", []),
        key="Name",
        expected="Water solubility",
        kind="solubility",
    )
    _set_named_parameter(
        solubility.get("Parameters", []),
        name="Solubility at reference pH",
        value=values["caffeine_water_solubility_25c"],
        unit="mg/l",
    )

    fraction_unbound = _find_unique(
        compound.get("FractionUnbound", []),
        key="Species",
        expected="Human",
        kind="fraction-unbound record",
    )
    fraction_unbound["Name"] = "Lelo 1986 PBPK-KB-v1"
    _set_named_parameter(
        fraction_unbound.get("Parameters", []),
        name="Fraction unbound (plasma, reference value)",
        value=values["caffeine_fraction_unbound_plasma"],
    )

    compound["Lipophilicity"] = [
        {
            "Name": "PubChem-XLogP-Initial-v1",
            "Parameters": [
                {
                    "Name": "Lipophilicity",
                    "Value": values["caffeine_effective_lipophilicity"],
                    "Unit": "Log Units",
                }
            ],
        }
    ]
    compound.pop("IntestinalPermeability", None)
    compound["CalculationMethods"] = copy.deepcopy(CALCULATION_METHODS)

    cyp = _find_unique(
        compound.get("Processes", []),
        key="InternalName",
        expected="MetabolizationLiverMicrosomes_MM",
        kind="CYP1A2 process",
    )
    if cyp.get("Molecule") != "CYP1A2":
        raise CaffeineStudySnapshotError("CYP process molecule drifted.")
    _set_named_parameter(
        cyp.get("Parameters", []),
        name="In vitro Vmax for liver microsomes",
        value=values["caffeine_cyp1a2_vmax_effective_n3"],
        unit="pmol/min/mg mic. protein",
    )
    _set_named_parameter(
        cyp.get("Parameters", []),
        name="Km",
        value=values["caffeine_cyp1a2_km_n3_demethylation"],
        unit="µmol/l",
    )

    renal = _find_unique(
        compound.get("Processes", []),
        key="InternalName",
        expected="KidneyClearance",
        kind="renal process",
    )
    renal["DataSource"] = RENAL_PROCESS_DATA_SOURCE
    renal["Parameters"] = [
        {
            "Name": "Fraction unbound (experiment)",
            "Value": values["caffeine_fraction_unbound_plasma"],
        },
        {
            "Name": "Plasma clearance",
            "Value": values["caffeine_renal_plasma_clearance"],
            "Unit": "ml/min/kg",
        },
    ]
    compound["Processes"] = [cyp, renal]


def _individual_snapshot(spec: StudyIndividualSpec) -> dict[str, Any]:
    origin: dict[str, Any] = {
        "CalculationMethods": [
            "Body surface area - Mosteller",
            "SurfaceAreaPlsInt_VAR1",
        ],
        "Species": "Human",
        "Population": spec.population,
        "Gender": spec.gender,
        "Age": {"Value": spec.age_years, "Unit": "year(s)"},
    }
    if spec.weight_kg is not None:
        origin["Weight"] = {"Value": spec.weight_kg, "Unit": "kg"}
    if spec.height_m is not None:
        origin["Height"] = {"Value": spec.height_m, "Unit": "m"}
    return {
        "Name": spec.name,
        "Seed": spec.seed,
        "OriginData": origin,
        "ExpressionProfiles": [CYP1A2_PROFILE_NAME],
    }


def _protocol_snapshot(spec: StudyProtocolSpec) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = [
        {"Name": "Start time", "Value": 0.0, "Unit": "h"},
        {
            "Name": "InputDose",
            "Value": spec.dose_value,
            "Unit": spec.dose_unit,
        },
    ]
    if spec.infusion_duration_min is not None:
        parameters.append(
            {
                "Name": "Infusion time",
                "Value": spec.infusion_duration_min,
                "Unit": "min",
            }
        )
    elif spec.application_type == "Oral":
        parameters.append(
            {
                "Name": "Volume of water/body weight",
                "Value": 3.5,
                "Unit": "ml/kg",
            }
        )
    return {
        "Name": spec.name,
        "ApplicationType": spec.application_type,
        "DosingInterval": "Single",
        "Parameters": parameters,
    }


def _output_schema(spec: StudySimulationSpec) -> list[dict[str, Any]]:
    return [
        {
            "Parameters": [
                {"Name": "Start time", "Value": item.start_h, "Unit": "h"},
                {"Name": "End time", "Value": item.end_h, "Unit": "h"},
                {
                    "Name": "Resolution",
                    "Value": item.resolution_points_per_h,
                    "Unit": "pts/h",
                },
            ]
        }
        for item in spec.output_intervals
    ]


def _simulation_snapshot(
    spec: StudySimulationSpec,
    protocol: StudyProtocolSpec,
) -> dict[str, Any]:
    protocol_selection: dict[str, Any] = {"Name": spec.protocol_name}
    if protocol.formulation_name:
        protocol_selection["Formulations"] = [
            {"Key": "Formulation", "Name": protocol.formulation_name}
        ]
    if protocol.application_type == "Intravenous":
        if protocol.infusion_duration_min is None:
            raise CaffeineStudySnapshotError("IV protocol lost infusion duration.")
        simulation_parameters = [
            {
                "Path": (
                    f"Events|{protocol.name}|Application_1|"
                    "ProtocolSchemaItem|Infusion time"
                ),
                "Value": round(protocol.infusion_duration_min / 60.0, 10),
                "Unit": "h",
            }
        ]
    else:
        if protocol.formulation_name is None:
            raise CaffeineStudySnapshotError("Oral protocol lost formulation.")
        simulation_parameters = [
            {
                "Path": (
                    f"Events|{protocol.name}|{protocol.formulation_name}|"
                    "Application_1|ProtocolSchemaItem|"
                    "Volume of water/body weight"
                ),
                "Value": 3.5,
                "Unit": "ml/kg",
            }
        ]
    return {
        "Name": spec.name,
        "Individual": spec.individual_name,
        "Model": "4Comp",
        "Solver": {},
        "HasResults": False,
        "OutputSchema": _output_schema(spec),
        "OutputSelections": [CAFFEINE_OUTPUT_PATH],
        "Parameters": simulation_parameters,
        "Compounds": [
            {
                "Name": "Caffeine",
                "CalculationMethods": copy.deepcopy(CALCULATION_METHODS),
                "Processes": [
                    {
                        "Name": RENAL_PROCESS_SELECTION_NAME,
                        "SystemicProcessType": "Renal",
                    },
                    {"Name": "CYP1A2-MM", "MoleculeName": "CYP1A2"},
                ],
                "Protocol": protocol_selection,
            }
        ],
    }


def transform_study_snapshot(
    *,
    source_snapshot: dict[str, Any],
    parameter_manifest: CaffeineParameterManifest,
    individuals: list[StudyIndividualSpec],
    protocols: list[StudyProtocolSpec],
    simulations: list[StudySimulationSpec],
) -> dict[str, Any]:
    transformed = copy.deepcopy(source_snapshot)
    compound = _find_unique(
        transformed.get("Compounds", []),
        key="Name",
        expected="Caffeine",
        kind="compound",
    )
    _apply_compound_parameters(compound, parameter_manifest)

    profiles = [
        item
        for item in transformed.get("ExpressionProfiles", [])
        if item.get("Molecule") == "CYP1A2"
        and item.get("Species") == "Human"
        and item.get("Category") == "PBPK-KB-Caffeine-Adult-v1"
    ]
    if len(profiles) != 1:
        raise CaffeineStudySnapshotError("Locked CYP1A2 profile is missing.")
    transformed["ExpressionProfiles"] = profiles
    transformed["Individuals"] = [_individual_snapshot(item) for item in individuals]
    transformed["Formulations"] = [
        {
            "Name": item.formulation_name,
            "FormulationType": item.formulation_type,
        }
        for item in protocols
        if item.formulation_name is not None
    ]
    transformed["Protocols"] = [_protocol_snapshot(item) for item in protocols]
    protocol_lookup = {item.name: item for item in protocols}
    transformed["Simulations"] = [
        _simulation_snapshot(item, protocol_lookup[item.protocol_name])
        for item in simulations
    ]
    return transformed


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    values: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(model.model_validate_json(line))
            except Exception as exc:
                raise CaffeineStudySnapshotError(
                    f"Invalid {path.name}:{line_number}: {exc}"
                ) from exc
    return values


def _manifest_checksum_valid(model: Any) -> bool:
    return canonical_sha256(model.model_dump(mode="json", exclude={"checksum"})) == (
        model.checksum
    )


def _snapshot_checks(
    *,
    transformed: dict[str, Any],
    parameter_manifest: CaffeineParameterManifest,
    dataset_manifest: CaffeineDatasetManifest,
    observations: list[ObservationPoint],
    metrics: list[PKMetric],
    individuals: list[StudyIndividualSpec],
    protocols: list[StudyProtocolSpec],
    simulations: list[StudySimulationSpec],
) -> dict[str, bool]:
    simulation_snapshots = {
        item["Name"]: item for item in transformed.get("Simulations", [])
    }
    protocol_snapshots = {
        item["Name"]: item for item in transformed.get("Protocols", [])
    }
    protocol_specs = {item.name: item for item in protocols}

    route_exact = True
    process_exact = True
    event_parameters_exact = True
    for simulation_spec in simulations:
        simulation = simulation_snapshots.get(simulation_spec.name, {})
        compound_matches = [
            item
            for item in simulation.get("Compounds", [])
            if item.get("Name") == "Caffeine"
        ]
        if len(compound_matches) != 1:
            process_exact = False
            route_exact = False
            continue
        selected = compound_matches[0]
        process_exact &= {
            (
                item.get("Name"),
                item.get("MoleculeName"),
                item.get("SystemicProcessType"),
            )
            for item in selected.get("Processes", [])
        } == {
            ("CYP1A2-MM", "CYP1A2", None),
            (RENAL_PROCESS_SELECTION_NAME, None, "Renal"),
        }
        protocol_spec = protocol_specs[simulation_spec.protocol_name]
        protocol = protocol_snapshots.get(simulation_spec.protocol_name, {})
        params = {item.get("Name"): item for item in protocol.get("Parameters", [])}
        simulation_parameters = simulation.get("Parameters", [])
        if protocol_spec.application_type == "Intravenous":
            expected_event_parameter = {
                "Path": (
                    f"Events|{protocol_spec.name}|Application_1|"
                    "ProtocolSchemaItem|Infusion time"
                ),
                "Value": round(protocol_spec.infusion_duration_min / 60.0, 10),
                "Unit": "h",
            }
        else:
            expected_event_parameter = {
                "Path": (
                    f"Events|{protocol_spec.name}|"
                    f"{protocol_spec.formulation_name}|Application_1|"
                    "ProtocolSchemaItem|Volume of water/body weight"
                ),
                "Value": 3.5,
                "Unit": "ml/kg",
            }
        event_parameters_exact &= simulation_parameters == [
            expected_event_parameter
        ]
        route_exact &= (
            protocol.get("ApplicationType") == protocol_spec.application_type
            and params.get("InputDose", {}).get("Value") == protocol_spec.dose_value
            and params.get("InputDose", {}).get("Unit") == protocol_spec.dose_unit
            and (
                params.get("Infusion time", {}).get("Value")
                == protocol_spec.infusion_duration_min
                if protocol_spec.application_type == "Intravenous"
                else "Infusion time" not in params
            )
            and (selected.get("Protocol") or {}).get("Name")
            == simulation_spec.protocol_name
        )

    mapped_primary = sum(
        item.observation_count
        for item in simulations
        if item.split != "external_validation_secondary"
    )
    mapped_secondary = sum(
        item.metric_count * item.aggregation_weight
        for item in simulations
        if item.split == "external_validation_secondary"
    )
    calibration_ids = {
        item.dataset_id
        for item in simulations
        if item.use_in_calibration_objective
    }
    pmc572_weights = sum(
        item.aggregation_weight
        for item in simulations
        if item.dataset_id == "pmc5724581_beverage_50_65mg"
    )

    compound = _find_unique(
        transformed.get("Compounds", []),
        key="Name",
        expected="Caffeine",
        kind="compound",
    )
    values = _parameter_values(parameter_manifest)
    cyp = _find_unique(
        compound.get("Processes", []),
        key="InternalName",
        expected="MetabolizationLiverMicrosomes_MM",
        kind="CYP process",
    )
    renal = _find_unique(
        compound.get("Processes", []),
        key="InternalName",
        expected="KidneyClearance",
        kind="renal process",
    )
    cyp_parameters = {item["Name"]: item["Value"] for item in cyp["Parameters"]}
    renal_parameters = {
        item["Name"]: item["Value"] for item in renal["Parameters"]
    }
    renal_process_reference_resolves = (
        renal.get("DataSource") == RENAL_PROCESS_DATA_SOURCE
        and all(
            any(
                process.get("Name")
                == f"Renal Clearances-{renal.get('DataSource')}"
                and process.get("SystemicProcessType") == "Renal"
                for selected_compound in simulation.get("Compounds", [])
                if selected_compound.get("Name") == "Caffeine"
                for process in selected_compound.get("Processes", [])
            )
            for simulation in transformed.get("Simulations", [])
        )
    )
    parameter_values_exact = (
        cyp_parameters["Km"]
        == values["caffeine_cyp1a2_km_n3_demethylation"]
        and cyp_parameters["In vitro Vmax for liver microsomes"]
        == values["caffeine_cyp1a2_vmax_effective_n3"]
        and renal_parameters["Fraction unbound (experiment)"]
        == values["caffeine_fraction_unbound_plasma"]
        and renal_parameters["Plasma clearance"]
        == values["caffeine_renal_plasma_clearance"]
    )
    individual_snapshots = {
        item.get("Name"): item for item in transformed.get("Individuals", [])
    }
    contextual_bmi_not_encoded = all(
        (
            item.reported_bmi_kg_per_m2 is None
            or item.context_only_fields == ["bmi"]
        )
        and "BMI"
        not in individual_snapshots.get(item.name, {}).get("OriginData", {})
        for item in individuals
    )
    return {
        "dataset_manifest_accepted": dataset_manifest.accepted,
        "parameter_manifest_accepted": parameter_manifest.accepted,
        "dataset_manifest_checksum_reproducible": _manifest_checksum_valid(
            dataset_manifest
        ),
        "parameter_manifest_checksum_reproducible": _manifest_checksum_valid(
            parameter_manifest
        ),
        "seven_individuals_exact": len(individuals) == 7
        and len(transformed.get("Individuals", [])) == 7,
        "six_protocols_exact": len(protocols) == 6
        and len(transformed.get("Protocols", [])) == 6,
        "seven_simulations_exact": len(simulations) == 7
        and set(simulation_snapshots) == {item.name for item in simulations},
        "route_dose_and_infusion_exact": bool(route_exact),
        "canonical_event_parameters_exact": bool(event_parameters_exact),
        "cyp1a2_and_renal_selected_in_all": bool(process_exact),
        "renal_process_reference_resolves": renal_process_reference_resolves,
        "parameter_values_exact": parameter_values_exact,
        "fitted_intestinal_permeability_removed": (
            "IntestinalPermeability" not in compound
        ),
        "contextual_bmi_not_encoded": contextual_bmi_not_encoded,
        "all_oral_formulations_dissolved": all(
            item.formulation_type == "Formulation_Dissolved"
            for item in protocols
            if item.application_type == "Oral"
        ),
        "capsule_missing_dissolution_declared": any(
            "CAPSULE_DISSOLUTION_DATA_MISSING" in item.assumption_codes
            for item in protocols
        ),
        "all_50_observations_mapped": len(observations) == 50
        and mapped_primary == 50,
        "all_5_secondary_metrics_mapped": len(metrics) == 5
        and math.isclose(mapped_secondary, 5.0),
        "secondary_sex_weights_sum_to_one": math.isclose(pmc572_weights, 1.0),
        "calibration_datasets_exact": calibration_ids
        == set(CALIBRATION_DATASETS),
        "holdout_external_excluded_from_calibration": all(
            not item.use_in_calibration_objective
            for item in simulations
            if item.split != "calibration"
        ),
        "output_grids_cover_observations": all(
            item.max_observation_time_h is None
            or item.output_intervals[-1].end_h >= item.max_observation_time_h
            for item in simulations
        ),
    }


def build_caffeine_study_snapshot_package(
    *,
    source_snapshot_path: Path,
    parameter_manifest_path: Path,
    dataset_manifest_path: Path,
    observations_path: Path,
    metrics_path: Path,
    output_directory: Path,
) -> CaffeineStudyScenarioManifest:
    if output_directory.exists():
        raise CaffeineStudySnapshotError(
            f"Immutable output directory exists: {output_directory}"
        )
    source_snapshot = json.loads(
        source_snapshot_path.read_text(encoding="utf-8-sig")
    )
    parameter_manifest = CaffeineParameterManifest.model_validate_json(
        parameter_manifest_path.read_text(encoding="utf-8")
    )
    dataset_manifest = CaffeineDatasetManifest.model_validate_json(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    observations = _read_jsonl(observations_path, ObservationPoint)
    metrics = _read_jsonl(metrics_path, PKMetric)

    source_artifact = next(
        (
            item
            for item in parameter_manifest.source_artifacts
            if item.artifact_id == "caffeine_snapshot_v2"
        ),
        None,
    )
    if source_artifact is None or source_artifact.sha256 != sha256_file(
        source_snapshot_path
    ):
        raise CaffeineStudySnapshotError(
            "Source snapshot does not match the parameter-manifest lock."
        )
    if parameter_manifest.dataset_manifest_checksum != dataset_manifest.checksum:
        raise CaffeineStudySnapshotError(
            "Parameter and dataset manifests are not cryptographically bound."
        )

    individuals = frozen_individuals()
    protocols = frozen_protocols()
    simulations = frozen_simulations(observations, metrics)
    transformed = transform_study_snapshot(
        source_snapshot=source_snapshot,
        parameter_manifest=parameter_manifest,
        individuals=individuals,
        protocols=protocols,
        simulations=simulations,
    )
    checks = _snapshot_checks(
        transformed=transformed,
        parameter_manifest=parameter_manifest,
        dataset_manifest=dataset_manifest,
        observations=observations,
        metrics=metrics,
        individuals=individuals,
        protocols=protocols,
        simulations=simulations,
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise CaffeineStudySnapshotError(
            "Study snapshot acceptance failed: " + ", ".join(failed)
        )

    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise CaffeineStudySnapshotError(f"Stale temporary output: {temporary}")
    temporary.mkdir(parents=True)
    snapshot_file_sha256 = write_json_atomic(
        temporary / "study_snapshot.json",
        transformed,
        canonical=False,
    )
    payload = {
        "schema_version": "1.0.0",
        "builder_version": STUDY_SNAPSHOT_BUILDER_VERSION,
        "scenario_version": STUDY_SCENARIO_VERSION,
        "generated_at": datetime.now(UTC),
        "source_snapshot_sha256": sha256_file(source_snapshot_path),
        "transformed_snapshot_file_sha256": snapshot_file_sha256,
        "transformed_snapshot_semantic_sha256": canonical_sha256(transformed),
        "dataset_version": dataset_manifest.dataset_version,
        "dataset_manifest_checksum": dataset_manifest.checksum,
        "parameter_manifest_version": parameter_manifest.manifest_version,
        "parameter_manifest_checksum": parameter_manifest.checksum,
        "individuals": individuals,
        "protocols": protocols,
        "simulations": simulations,
        "system_assumptions": [
            (
                "European_ICRP_2002 is retained from the licensed PK-Sim source "
                "structure where a study-specific population was not reported."
            ),
            (
                "Missing height or weight is generated by PK-Sim from locked "
                "demographics and deterministic seeds; it is not observed evidence."
            ),
            (
                "Oral protocol water volume 3.5 ml/kg is the locked PK-Sim source "
                "default and is not a reported study administration volume."
            ),
            (
                "Water-solubility reference pH 7 is retained from the licensed "
                "source structure as a model-system setting, not observed evidence."
            ),
            (
                "The EudraCT capsule uses a separately named dissolved immediate-"
                "release approximation because no dissolution curve was reported."
            ),
            (
                "PMC572 mean BMI is retained as context only because PK-Sim 12.3 "
                "individual snapshots require height or weight to constrain "
                "anthropometry; neither study mean was reported."
            ),
        ],
        "checks": checks,
        "accepted": True,
    }
    draft = CaffeineStudyScenarioManifest(**payload, checksum="0" * 64)
    manifest = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    write_json_atomic(
        temporary / "study_scenario_manifest.json",
        manifest,
        canonical=False,
    )
    files = [
        {
            "file": item.name,
            "sha256": sha256_file(item),
            "byte_size": item.stat().st_size,
        }
        for item in sorted(temporary.iterdir(), key=lambda value: value.name)
        if item.is_file()
    ]
    write_json_atomic(
        temporary / "files.sha256.json",
        {"files": files},
        canonical=True,
    )
    temporary.replace(output_directory)
    return manifest


def _normalize_population_mean_elisions(
    transformed: dict[str, Any],
    roundtrip: dict[str, Any],
    manifest: CaffeineStudyScenarioManifest,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    left = copy.deepcopy(transformed)
    right = copy.deepcopy(roundtrip)
    notes: list[str] = []
    left_individuals = {
        item.get("Name"): item for item in left.get("Individuals", [])
    }
    right_individuals = {
        item.get("Name"): item for item in right.get("Individuals", [])
    }
    for spec in manifest.individuals:
        left_origin = left_individuals.get(spec.name, {}).get("OriginData", {})
        right_origin = right_individuals.get(spec.name, {}).get("OriginData", {})
        for key, expected_value, expected_unit in (
            ("Weight", spec.weight_kg, "kg"),
            ("Height", spec.height_m, "m"),
        ):
            if expected_value is None or key in right_origin:
                continue
            candidate = left_origin.get(key)
            if candidate != {"Value": expected_value, "Unit": expected_unit}:
                continue
            left_origin.pop(key)
            notes.append(
                "population_mean_elision: "
                f"{spec.name}.OriginData.{key}={expected_value} {expected_unit}"
            )
    return left, right, notes


def _normalize_snapshot_numeric_serialization(
    left: Any,
    right: Any,
    *,
    path: str = "",
    notes: list[str],
) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: _normalize_snapshot_numeric_serialization(
                value,
                right[key],
                path=f"{path}/{key}",
                notes=notes,
            )
            if key in right
            else value
            for key, value in left.items()
        }
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return left
        return [
            _normalize_snapshot_numeric_serialization(
                value,
                right[index],
                path=f"{path}/{index}",
                notes=notes,
            )
            for index, value in enumerate(left)
        ]
    numeric = (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    )
    if numeric and left != right and math.isclose(
        float(left),
        float(right),
        rel_tol=5e-9,
        abs_tol=1e-12,
    ):
        notes.append(
            "pksim_snapshot_numeric_serialization: "
            f"{path or '/'} {left!r} -> {right!r}"
        )
        return right
    return left


def audit_caffeine_study_roundtrip(
    *,
    transformed_snapshot_path: Path,
    roundtrip_snapshot_path: Path,
    scenario_manifest_path: Path,
) -> CaffeineStudyRoundTripAudit:
    manifest = CaffeineStudyScenarioManifest.model_validate_json(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    if not _manifest_checksum_valid(manifest):
        raise CaffeineStudySnapshotError("Scenario manifest checksum is invalid.")
    transformed = json.loads(
        transformed_snapshot_path.read_text(encoding="utf-8-sig")
    )
    roundtrip = json.loads(
        roundtrip_snapshot_path.read_text(encoding="utf-8-sig")
    )
    transformed, roundtrip, pair_notes = _normalize_population_mean_elisions(
        transformed,
        roundtrip,
        manifest,
    )
    left, left_notes = _semantic_projection(transformed)
    right, right_notes = _semantic_projection(roundtrip)
    numeric_notes: list[str] = []
    left = _normalize_snapshot_numeric_serialization(
        left,
        right,
        notes=numeric_notes,
    )
    differences = sorted(_diff_paths(left, right))
    expected_simulations = {item.name for item in manifest.simulations}
    expected_protocols = {
        item.name: item.application_type for item in manifest.protocols
    }
    roundtrip_protocols = {
        item.get("Name"): item for item in roundtrip.get("Protocols", [])
    }
    process_checks: list[bool] = []
    for simulation in roundtrip.get("Simulations", []):
        compounds = [
            item
            for item in simulation.get("Compounds", [])
            if item.get("Name") == "Caffeine"
        ]
        if len(compounds) != 1:
            process_checks.append(False)
            continue
        processes = compounds[0].get("Processes", [])
        process_checks.append(
            any(
                item.get("Name") == "CYP1A2-MM"
                and item.get("MoleculeName") == "CYP1A2"
                for item in processes
            )
            and any(item.get("SystemicProcessType") == "Renal" for item in processes)
        )
    checks = {
        "complete_semantic_snapshot_equal": not differences,
        "seven_simulations_preserved": {
            item.get("Name") for item in roundtrip.get("Simulations", [])
        }
        == expected_simulations,
        "seven_individuals_preserved": len(roundtrip.get("Individuals", [])) == 7,
        "routes_preserved": all(
            roundtrip_protocols.get(name, {}).get("ApplicationType") == route
            for name, route in expected_protocols.items()
        ),
        "cyp1a2_and_renal_preserved": len(process_checks) == 7
        and all(process_checks),
        "source_transformed_snapshot_matches_manifest": sha256_file(
            transformed_snapshot_path
        )
        == manifest.transformed_snapshot_file_sha256,
    }
    payload = {
        "schema_version": "1.0.0",
        "builder_version": STUDY_SNAPSHOT_BUILDER_VERSION,
        "generated_at": datetime.now(UTC),
        "scenario_manifest_checksum": manifest.checksum,
        "transformed_snapshot_sha256": sha256_file(transformed_snapshot_path),
        "roundtrip_snapshot_sha256": sha256_file(roundtrip_snapshot_path),
        "transformed_semantic_sha256": canonical_sha256(left),
        "roundtrip_semantic_sha256": canonical_sha256(right),
        "semantic_differences": differences,
        "accepted_normalizations": sorted(
            set(left_notes) | set(right_notes) | set(pair_notes) | set(numeric_notes)
        ),
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineStudyRoundTripAudit(**payload, checksum="0" * 64)
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or audit the frozen caffeine study snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-snapshot", type=Path, required=True)
    build.add_argument("--parameter-manifest", type=Path, required=True)
    build.add_argument("--dataset-manifest", type=Path, required=True)
    build.add_argument("--observations", type=Path, required=True)
    build.add_argument("--metrics", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--transformed-snapshot", type=Path, required=True)
    audit.add_argument("--roundtrip-snapshot", type=Path, required=True)
    audit.add_argument("--scenario-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_caffeine_study_snapshot_package(
            source_snapshot_path=args.source_snapshot,
            parameter_manifest_path=args.parameter_manifest,
            dataset_manifest_path=args.dataset_manifest,
            observations_path=args.observations,
            metrics_path=args.metrics,
            output_directory=args.output_dir,
        )
    else:
        result = audit_caffeine_study_roundtrip(
            transformed_snapshot_path=args.transformed_snapshot,
            roundtrip_snapshot_path=args.roundtrip_snapshot,
            scenario_manifest_path=args.scenario_manifest,
        )
        write_json_atomic(args.output, result, canonical=False)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
