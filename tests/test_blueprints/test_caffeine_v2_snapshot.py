from __future__ import annotations

from types import SimpleNamespace

from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    PERMEABILITY_ALTERNATIVE,
    _normalize_inherited_individual_elisions,
    _path_allowed,
    _structural_checks,
    _transformed_snapshot,
)


def _parameter(parameter_id: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(parameter_id=parameter_id, value=value)


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        parameters=[
            _parameter("caffeine_fraction_n3_paraxanthine", 0.839),
            _parameter("caffeine_fraction_n1_theobromine", 0.121),
            _parameter("caffeine_fraction_n7_theophylline", 0.040),
            _parameter("total_cyp1a2_specific_clearance", 0.1),
            SimpleNamespace(
                parameter_id="specific_intestinal_permeability_transcellular",
                value=1e-5,
                unit="dm/min",
            ),
        ]
    )


def _source_snapshot() -> dict:
    simulations = []
    for name in EXPECTED_SIMULATIONS:
        simulations.append(
            {
                "Name": name,
                "Compounds": [
                    {
                        "Name": "Caffeine",
                        "CalculationMethods": ["PK-Sim Standard"],
                        "Processes": [
                            {"MoleculeName": "CYP1A2", "Name": "CYP1A2-MM"},
                            {
                                "Name": (
                                    "Renal Clearances-Birkett 1991 Table 1 "
                                    "rederived PBPK-KB-v1"
                                ),
                                "SystemicProcessType": "Renal",
                            },
                        ],
                    }
                ],
            }
        )
    return {
        "Version": 23,
        "ExpressionProfiles": [],
        "Individuals": [{"Name": f"I{i}"} for i in range(7)],
        "Compounds": [
            {
                "Name": "Caffeine",
                "Description": "v1",
                "Lipophilicity": [
                    {
                        "Name": "PubChem-XLogP-Initial-v1",
                        "Parameters": [
                            {
                                "Name": "Lipophilicity",
                                "Value": -0.1,
                                "Unit": "Log Units",
                            }
                        ],
                    }
                ],
                "Processes": [
                    {
                        "DataSource": "MM",
                        "InternalName": "MetabolizationLiverMicrosomes_MM",
                        "Molecule": "CYP1A2",
                        "Parameters": [],
                    },
                    {
                        "DataSource": (
                            "Birkett 1991 Table 1 rederived PBPK-KB-v1"
                        ),
                        "InternalName": "KidneyClearance",
                        "Species": "Human",
                        "Parameters": [],
                    },
                ],
            }
        ],
        "Formulations": [{"Name": "F"}],
        "Protocols": [{"Name": f"P{i}"} for i in range(6)],
        "Simulations": simulations,
    }


def test_transforms_only_the_registered_v2_model_structure() -> None:
    transformed, specs = _transformed_snapshot(_source_snapshot(), _manifest())

    assert [item.process_name for item in specs] == [
        "CYP1A2-N3-Paraxanthine",
        "CYP1A2-N1-Theobromine",
        "CYP1A2-N7-Theophylline",
    ]
    compound = transformed["Compounds"][0]
    assert [item["InternalName"] for item in compound["Processes"][:3]] == [
        "MetabolizationIntrinsic_FirstOrder",
        "MetabolizationIntrinsic_FirstOrder",
        "MetabolizationIntrinsic_FirstOrder",
    ]
    assert compound["IntestinalPermeability"][0]["Name"] == (
        PERMEABILITY_ALTERNATIVE
    )
    assert all(_structural_checks(transformed, _manifest()).values())


def test_changed_path_allowlist_blocks_dose_or_individual_changes() -> None:
    assert _path_allowed("/Compounds/0/Processes")
    assert _path_allowed("/Simulations/0/Compounds/0/Alternatives")
    assert not _path_allowed("/Protocols/0/Dose")
    assert not _path_allowed("/Individuals/0/Weight")


def test_roundtrip_normalization_requires_exact_inherited_value_and_unit() -> None:
    transformed = {
        "Individuals": [
            {
                "Name": "I1",
                "OriginData": {
                    "Weight": {"Value": 73.0, "Unit": "kg"},
                    "Height": {"Value": 1.76, "Unit": "m"},
                },
            }
        ]
    }
    roundtrip = {"Individuals": [{"Name": "I1", "OriginData": {}}]}
    inherited = [
        {"name": "I1", "weight_kg": 73.0, "height_m": 1.75}
    ]

    left, _, notes = _normalize_inherited_individual_elisions(
        transformed,
        roundtrip,
        inherited,
    )

    assert "Weight" not in left["Individuals"][0]["OriginData"]
    assert left["Individuals"][0]["OriginData"]["Height"] == {
        "Value": 1.76,
        "Unit": "m",
    }
    assert notes == ["population_mean_elision: I1.OriginData.Weight=73.0 kg"]
