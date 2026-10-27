from __future__ import annotations

from dataclasses import dataclass
from typing import Final


ONTOLOGY_VERSION: Final = "pbpk-recommendation-ontology-1.1.0"


@dataclass(frozen=True)
class RelationSpec:
    subject_types: frozenset[str]
    object_types: frozenset[str]
    scientific_fact: bool = True
    event_role: bool = False


NODE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Compound",
        "ActiveMoiety",
        "SaltForm",
        "Metabolite",
        "Formulation",
        "DosageForm",
        "AdministrationRoute",
        "DoseRegimen",
        "Enzyme",
        "Transporter",
        "Protein",
        "Gene",
        "GeneticVariant",
        "Organ",
        "Tissue",
        "CellType",
        "Species",
        "Population",
        "Disease",
        "Study",
        "StudyArm",
        "ClinicalTrial",
        "Publication",
        "SourceDocument",
        "Assay",
        "Observation",
        "PKParameter",
        "ParameterValue",
        "Unit",
        "PBPKModel",
        "ModelVersion",
        "ModelStructure",
        "Compartment",
        "ModelParameter",
        "Software",
        "SimulationScenario",
        "QualificationDataset",
        "ValidationMetric",
        "Evidence",
        "Conflict",
        "PKMeasurement",
        "DDIEvent",
        "MetabolismEvent",
        "TransportEvent",
        "AbsorptionEvent",
        "ExcretionEvent",
        "ModelCalibrationEvent",
        "ModelValidationEvent",
    }
)

COMPOUND_LIKE: Final = frozenset({"Compound", "ActiveMoiety", "SaltForm", "Metabolite"})
BIOLOGICAL_TARGET: Final = frozenset({"Enzyme", "Transporter", "Protein", "Gene"})
ANATOMY: Final = frozenset({"Organ", "Tissue", "CellType"})
EVENT_TYPES: Final = frozenset(
    {
        "PKMeasurement",
        "DDIEvent",
        "MetabolismEvent",
        "TransportEvent",
        "AbsorptionEvent",
        "ExcretionEvent",
        "ModelCalibrationEvent",
        "ModelValidationEvent",
    }
)
PARAMETER_TYPES: Final = frozenset({"PKParameter", "ModelParameter"})
DATASET_TYPES: Final = frozenset({"QualificationDataset", "SourceDocument", "Publication", "Evidence"})


RELATIONS: Final[dict[str, RelationSpec]] = {
    "HAS_ACTIVE_MOIETY": RelationSpec(COMPOUND_LIKE, frozenset({"ActiveMoiety"})),
    "HAS_SALT_FORM": RelationSpec(COMPOUND_LIKE, frozenset({"SaltForm"})),
    "HAS_METABOLITE": RelationSpec(COMPOUND_LIKE, frozenset({"Metabolite"})),
    "SUBSTRATE_OF": RelationSpec(COMPOUND_LIKE, BIOLOGICAL_TARGET),
    "INHIBITS": RelationSpec(COMPOUND_LIKE, BIOLOGICAL_TARGET),
    "INDUCES": RelationSpec(COMPOUND_LIKE, BIOLOGICAL_TARGET),
    "TRANSPORTED_BY": RelationSpec(COMPOUND_LIKE, frozenset({"Transporter"})),
    "METABOLIZED_TO": RelationSpec(COMPOUND_LIKE, frozenset({"Metabolite"})),
    "EXPRESSED_IN": RelationSpec(BIOLOGICAL_TARGET, ANATOMY),
    "ABSORBED_VIA": RelationSpec(COMPOUND_LIKE, frozenset({"AbsorptionEvent"})),
    "EXCRETED_VIA": RelationSpec(COMPOUND_LIKE, frozenset({"ExcretionEvent"})),
    "PERPETRATOR_IN": RelationSpec(COMPOUND_LIKE, frozenset({"DDIEvent"}), event_role=True),
    "VICTIM_IN": RelationSpec(COMPOUND_LIKE, frozenset({"DDIEvent"}), event_role=True),
    "MEDIATED_BY": RelationSpec(EVENT_TYPES, BIOLOGICAL_TARGET),
    "AFFECTS_PARAMETER": RelationSpec(EVENT_TYPES, PARAMETER_TYPES),
    "REPORTS": RelationSpec(frozenset({"Evidence"}), EVENT_TYPES, event_role=True),
    "OF_COMPOUND": RelationSpec(EVENT_TYPES, COMPOUND_LIKE, event_role=True),
    "MEASURES": RelationSpec(EVENT_TYPES | frozenset({"Observation"}), frozenset({"PKParameter"}), event_role=True),
    "HAS_MEASUREMENT": RelationSpec(
        frozenset({"Study", "StudyArm", "Evidence"}),
        frozenset({"PKMeasurement", "Observation"}),
        event_role=True,
    ),
    "HAS_VALUE": RelationSpec(EVENT_TYPES | frozenset({"Observation"}), frozenset({"ParameterValue"})),
    "HAS_UNIT": RelationSpec(frozenset({"ParameterValue"}), frozenset({"Unit"})),
    "UNDER_REGIMEN": RelationSpec(EVENT_TYPES, frozenset({"DoseRegimen"}), event_role=True),
    "STUDIED_IN": RelationSpec(
        EVENT_TYPES | frozenset({"Evidence", "Study", "StudyArm"}),
        frozenset({"Population", "Species"}),
        event_role=True,
    ),
    "IN_STUDY_ARM": RelationSpec(EVENT_TYPES, frozenset({"StudyArm"}), event_role=True),
    "HAS_FORMULATION": RelationSpec(
        frozenset({"DoseRegimen", "Compound"}),
        frozenset({"Formulation", "DosageForm"}),
    ),
    "HAS_ROUTE": RelationSpec(frozenset({"DoseRegimen"}), frozenset({"AdministrationRoute"})),
    "DESCRIBED_BY_MODEL": RelationSpec(COMPOUND_LIKE, frozenset({"PBPKModel"})),
    "HAS_MODEL_VERSION": RelationSpec(frozenset({"PBPKModel"}), frozenset({"ModelVersion"})),
    "HAS_SIMULATION": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"SimulationScenario"}),
    ),
    "HAS_VALIDATION_RUN": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"ModelValidationEvent"}),
    ),
    "USES_STRUCTURE": RelationSpec(frozenset({"PBPKModel", "ModelVersion"}), frozenset({"ModelStructure"})),
    "HAS_COMPARTMENT": RelationSpec(
        frozenset({"PBPKModel", "ModelVersion", "ModelStructure"}),
        frozenset({"Compartment"}),
    ),
    "USES_PARAMETER": RelationSpec(
        frozenset({"PBPKModel", "ModelVersion", "SimulationScenario"}),
        frozenset({"ModelParameter", "PKParameter"}),
    ),
    "USES_ENZYME": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"Enzyme"}),
    ),
    "USES_TRANSPORTER": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"Transporter"}),
    ),
    "SUPPORTS_ROUTE": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"AdministrationRoute"}),
    ),
    "SUPPORTS_FORMULATION": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"Formulation"}),
    ),
    "APPLIES_TO": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"Population", "Species"}),
    ),
    "CALIBRATED_WITH": RelationSpec(
        frozenset({"PBPKModel", "ModelVersion", "ModelCalibrationEvent"}),
        DATASET_TYPES,
    ),
    "VALIDATED_WITH": RelationSpec(
        frozenset({"PBPKModel", "ModelVersion", "ModelValidationEvent"}),
        DATASET_TYPES,
    ),
    "HAS_VALIDATION_METRIC": RelationSpec(
        frozenset({"ModelValidationEvent"}),
        frozenset({"ValidationMetric"}),
    ),
    "EVALUATES_DATASET": RelationSpec(
        frozenset({"ValidationMetric"}),
        frozenset({"QualificationDataset"}),
    ),
    "SUPPORTED_BY": RelationSpec(NODE_TYPES - frozenset({"SourceDocument"}), frozenset({"Evidence"})),
    "DERIVED_FROM": RelationSpec(frozenset({"Evidence"}), frozenset({"SourceDocument", "Publication"}), False),
    "CONTRADICTS": RelationSpec(frozenset({"Evidence", "Conflict"}), frozenset({"Evidence", "Conflict"})),
    "HAS_CONFLICT_MEMBER": RelationSpec(frozenset({"Conflict"}), frozenset({"Evidence"}), False),
}

RELATION_GLOSSES: Final[dict[str, str]] = {
    "HAS_ACTIVE_MOIETY": "links a drug product or substance to its pharmacologically active moiety",
    "HAS_SALT_FORM": "links a compound to a specific salt form",
    "HAS_METABOLITE": "links a parent compound to an identified metabolite",
    "SUBSTRATE_OF": "compound is transported or metabolized by the target protein",
    "INHIBITS": "compound suppresses enzyme or transporter activity",
    "INDUCES": "compound increases enzyme or transporter abundance or activity",
    "TRANSPORTED_BY": "compound movement is mediated by a membrane transporter",
    "METABOLIZED_TO": "parent compound is biotransformed into a metabolite",
    "EXPRESSED_IN": "protein or gene is expressed in an organ, tissue, or cell type",
    "ABSORBED_VIA": "compound enters systemic circulation through an absorption process",
    "EXCRETED_VIA": "compound or metabolite leaves the body through an excretion pathway",
    "PERPETRATOR_IN": "compound causes or modifies a drug-drug interaction",
    "VICTIM_IN": "compound exposure or response is affected in a drug-drug interaction",
    "MEDIATED_BY": "event is mechanistically mediated by an enzyme, transporter, protein, or gene",
    "AFFECTS_PARAMETER": "event changes or is evaluated by a pharmacokinetic parameter",
    "REPORTS": "evidence record reports a biological or pharmacokinetic event",
    "OF_COMPOUND": "event or measurement is attributed to the compound",
    "MEASURES": "measurement or observation quantifies a pharmacokinetic parameter",
    "HAS_MEASUREMENT": "study or evidence contains a measurement or observation",
    "HAS_VALUE": "measurement has a numerical or categorical parameter value",
    "HAS_UNIT": "parameter value is expressed using the linked measurement unit",
    "UNDER_REGIMEN": "event occurred under a defined dose, route, formulation, and schedule",
    "STUDIED_IN": "event or evidence was observed in a defined population, species, or disease context",
    "IN_STUDY_ARM": "event belongs to a specific intervention or comparator study arm",
    "HAS_FORMULATION": "dose regimen or compound uses a defined formulation or dosage form",
    "HAS_ROUTE": "dose regimen uses a defined administration route",
    "DESCRIBED_BY_MODEL": "compound is represented by a public physiologically based pharmacokinetic model",
    "HAS_MODEL_VERSION": "PBPK model has a versioned implementation",
    "HAS_SIMULATION": "model version contains a checksum-locked executable simulation scenario",
    "HAS_VALIDATION_RUN": "model version was evaluated in a versioned validation run",
    "USES_STRUCTURE": "model uses a whole-body or reduced PBPK structural design",
    "HAS_COMPARTMENT": "model structure contains the linked organ or compartment",
    "USES_PARAMETER": "model or simulation uses the linked physiological or drug parameter",
    "USES_ENZYME": "model version explicitly represents the linked enzyme",
    "USES_TRANSPORTER": "model version explicitly represents the linked transporter",
    "SUPPORTS_ROUTE": "model version or simulation supports the linked administration route",
    "SUPPORTS_FORMULATION": "model version or simulation supports the linked formulation",
    "APPLIES_TO": "model version or simulation represents the linked population or species",
    "CALIBRATED_WITH": "model parameters were fitted using the linked dataset or evidence",
    "VALIDATED_WITH": "model predictions were evaluated against the linked independent data",
    "HAS_VALIDATION_METRIC": "model validation event reports the linked performance metric",
    "EVALUATES_DATASET": "validation metric was calculated for the linked qualification dataset",
    "SUPPORTED_BY": "scientific fact or event is supported by a provenance-bearing evidence record",
    "DERIVED_FROM": "evidence was derived from the linked source document or publication",
    "CONTRADICTS": "two evidence records or conflict assertions disagree under comparable context",
    "HAS_CONFLICT_MEMBER": "conflict group contains the linked evidence record",
}

RELATION_SEARCH_TERMS: Final[dict[str, str]] = {
    "HAS_ACTIVE_MOIETY": "active drug moiety",
    "HAS_SALT_FORM": "chemical salt form",
    "HAS_METABOLITE": "parent drug metabolite",
    "SUBSTRATE_OF": "enzyme or transporter substrate",
    "INHIBITS": "enzyme or transporter activity inhibition",
    "INDUCES": "enzyme or transporter induction",
    "TRANSPORTED_BY": "membrane transporter mediated movement",
    "METABOLIZED_TO": "biotransformation to metabolite",
    "EXPRESSED_IN": "tissue organ cell expression",
    "ABSORBED_VIA": "drug absorption into systemic circulation",
    "EXCRETED_VIA": "renal biliary fecal urinary excretion",
    "PERPETRATOR_IN": "DDI causal perpetrator drug",
    "VICTIM_IN": "DDI affected victim drug",
    "MEDIATED_BY": "mechanistic enzyme or transporter mediator",
    "AFFECTS_PARAMETER": "changed pharmacokinetic endpoint",
    "REPORTS": "evidence reports event",
    "OF_COMPOUND": "measurement attributed to compound",
    "MEASURES": "quantified pharmacokinetic parameter",
    "HAS_MEASUREMENT": "study measurement observation",
    "HAS_VALUE": "numerical parameter value",
    "HAS_UNIT": "measurement unit",
    "UNDER_REGIMEN": "dose route formulation schedule",
    "STUDIED_IN": "population species disease study context",
    "IN_STUDY_ARM": "intervention comparator study arm",
    "HAS_FORMULATION": "formulation dosage form",
    "HAS_ROUTE": "administration route",
    "DESCRIBED_BY_MODEL": "public reusable PBPK model implementation",
    "HAS_MODEL_VERSION": "versioned model implementation",
    "HAS_SIMULATION": "executable simulation scenario",
    "HAS_VALIDATION_RUN": "versioned reproduced validation run",
    "USES_STRUCTURE": "whole-body or reduced PBPK structure",
    "HAS_COMPARTMENT": "organ tissue model compartment",
    "USES_PARAMETER": "model physiological drug parameter",
    "USES_ENZYME": "model represented enzyme",
    "USES_TRANSPORTER": "model represented transporter",
    "SUPPORTS_ROUTE": "supported administration route",
    "SUPPORTS_FORMULATION": "supported formulation",
    "APPLIES_TO": "applicable population or species",
    "CALIBRATED_WITH": "parameter calibration dataset",
    "VALIDATED_WITH": "independent model validation dataset",
    "HAS_VALIDATION_METRIC": "prediction performance metric",
    "EVALUATES_DATASET": "metric evaluated qualification dataset",
    "SUPPORTED_BY": "provenance evidence support",
    "DERIVED_FROM": "source document publication provenance",
    "CONTRADICTS": "context-compatible evidence disagreement",
    "HAS_CONFLICT_MEMBER": "evidence conflict group member",
}

RELATION_COMPACT_GLOSSES: Final[dict[str, str]] = {
    name: search_terms
    for name, search_terms in RELATION_SEARCH_TERMS.items()
}

PK_PARAMETER_SEMANTICS: Final[dict[str, tuple[tuple[str, ...], str]]] = {
    "auc": (
        ("area under the concentration-time curve", "systemic exposure"),
        "integral of drug concentration over time and a measure of total systemic exposure",
    ),
    "systemic exposure": (
        ("AUC", "area under the concentration-time curve"),
        "total circulating drug exposure integrated over time",
    ),
    "cmax": (
        ("maximum concentration", "peak concentration"),
        "highest observed or predicted drug concentration after dosing",
    ),
    "tmax": (
        ("time to maximum concentration", "time to peak concentration"),
        "elapsed time from dosing until the maximum concentration is reached",
    ),
    "half life": (
        ("elimination half-life", "terminal half-life"),
        "time required for drug concentration to decrease by one half",
    ),
    "clearance": (
        ("systemic clearance", "CL"),
        "volume of plasma or blood cleared of drug per unit time",
    ),
    "volume of distribution": (
        ("apparent volume of distribution", "Vd"),
        "apparent volume relating the amount of drug in the body to measured concentration",
    ),
    "bioavailability": (
        ("absolute bioavailability", "relative bioavailability", "F"),
        "fraction of an administered dose that reaches systemic circulation unchanged",
    ),
    "fraction absorbed": (
        ("fraction of dose absorbed", "Fa"),
        "proportion of the administered dose absorbed across the administration-site barrier",
    ),
    "fraction unbound": (
        ("unbound fraction", "free fraction", "fu", "protein binding"),
        "fraction of circulating drug not bound to plasma proteins",
    ),
    "protein binding": (
        ("fraction unbound", "unbound fraction", "fu"),
        "reversible association of drug with plasma proteins that determines free fraction",
    ),
    "renal clearance": (
        ("kidney clearance", "CLR"),
        "plasma clearance attributable to renal filtration, secretion, and reabsorption",
    ),
    "intrinsic clearance": (
        ("unbound intrinsic clearance", "CLint"),
        "enzyme-mediated metabolic clearance capacity independent of blood flow limitations",
    ),
    "absorption rate constant": (
        ("Ka", "first-order absorption rate"),
        "rate constant governing transfer of drug from the dosing site into systemic circulation",
    ),
    "molecularweight": (
        ("molecular weight", "molar mass", "formula weight"),
        "mass of one mole of the chemical compound",
    ),
    "full mwt": (
        ("molecular weight", "molar mass", "formula weight"),
        "mass of one mole of the chemical compound",
    ),
    "xlogp": (
        ("logP", "octanol-water partition coefficient", "lipophilicity"),
        "logarithm of the neutral compound partition coefficient between octanol and water",
    ),
    "alogp": (
        ("logP", "octanol-water partition coefficient", "lipophilicity"),
        "calculated logarithmic octanol-water partition coefficient describing lipophilicity",
    ),
    "tpsa": (
        ("topological polar surface area", "polar surface area", "PSA"),
        "molecular surface area contributed by polar atoms and their attached hydrogens",
    ),
    "psa": (
        ("polar surface area", "topological polar surface area", "TPSA"),
        "molecular surface area contributed by polar atoms and their attached hydrogens",
    ),
}


def normalized_term(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def pk_parameter_semantics(value: str) -> tuple[tuple[str, ...], str] | None:
    normalized = normalized_term(value)
    direct = PK_PARAMETER_SEMANTICS.get(normalized)
    if direct is not None:
        return direct
    for aliases, definition in PK_PARAMETER_SEMANTICS.values():
        if normalized in {normalized_term(alias) for alias in aliases}:
            return aliases, definition
    return None


def validate_node_type(node_type: str) -> str | None:
    if node_type not in NODE_TYPES:
        return f"unknown_node_type:{node_type}"
    return None


def validate_relation(predicate: str, subject_type: str, object_type: str) -> str | None:
    spec = RELATIONS.get(predicate)
    if spec is None:
        return f"unknown_predicate:{predicate}"
    if subject_type not in spec.subject_types:
        return f"invalid_subject_type:{predicate}:{subject_type}"
    if object_type not in spec.object_types:
        return f"invalid_object_type:{predicate}:{object_type}"
    return None
