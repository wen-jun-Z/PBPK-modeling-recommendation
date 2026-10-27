from __future__ import annotations

from types import SimpleNamespace

from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    PERMEABILITY_GROUP,
)
from pbpk_kb.blueprints.caffeine_v3_snapshot import (
    PERMEABILITY_ALTERNATIVE,
    SOURCE_PERMEABILITY_ALTERNATIVE,
    _path_allowed,
    _structural_checks,
    _transform_v4_snapshot,
)


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        parameters=[
            SimpleNamespace(
                parameter_id="specific_intestinal_permeability_transcellular",
                value=1e-5,
                unit="dm/min",
            )
        ]
    )


def _snapshot() -> dict:
    simulations = [
        {
            "Name": name,
            "Compounds": [
                {
                    "Name": "Caffeine",
                    "Processes": [
                        {"MoleculeName": "CYP1A2", "Name": "CYP1A2-N3"},
                        {"Name": "Renal", "SystemicProcessType": "Renal"},
                    ],
                    "Alternatives": [
                        {
                            "AlternativeName": SOURCE_PERMEABILITY_ALTERNATIVE,
                            "GroupName": PERMEABILITY_GROUP,
                        }
                    ],
                }
            ],
        }
        for name in EXPECTED_SIMULATIONS
    ]
    return {
        "Version": 23,
        "Individuals": [{"Name": f"I{index}"} for index in range(7)],
        "Protocols": [{"Name": f"P{index}"} for index in range(6)],
        "Formulations": [{"Name": "F"}],
        "ExpressionProfiles": [{"Name": "CYP1A2"}],
        "Compounds": [
            {
                "Name": "Caffeine",
                "Description": "v2",
                "CalculationMethods": ["PK-Sim Standard"],
                "Processes": [{"InternalName": "KidneyClearance"}],
                "IntestinalPermeability": [
                    {
                        "Name": SOURCE_PERMEABILITY_ALTERNATIVE,
                        "IsDefault": False,
                        "Parameters": [
                            {
                                "Name": (
                                    "Specific intestinal permeability "
                                    "(transcellular)"
                                ),
                                "Value": 1.155e-5,
                                "Unit": "dm/min",
                            }
                        ],
                    }
                ],
            }
        ],
        "Simulations": simulations,
    }


def test_v3_transform_changes_only_permeability_semantics() -> None:
    source = _snapshot()
    transformed = _transform_v4_snapshot(source, _manifest())

    compound = transformed["Compounds"][0]
    assert compound["IntestinalPermeability"][0]["Name"] == (
        PERMEABILITY_ALTERNATIVE
    )
    assert compound["IntestinalPermeability"][0]["Parameters"][0]["Value"] == (
        1e-5
    )
    assert compound["Processes"] == source["Compounds"][0]["Processes"]
    assert transformed["Individuals"] == source["Individuals"]
    assert all(
        simulation["Compounds"][0]["Alternatives"]
        == [
            {
                "AlternativeName": PERMEABILITY_ALTERNATIVE,
                "GroupName": PERMEABILITY_GROUP,
            }
        ]
        for simulation in transformed["Simulations"]
    )
    assert all(
        _structural_checks(
            source=source,
            transformed=transformed,
            manifest=_manifest(),
        ).values()
    )


def test_v3_changed_path_allowlist_is_narrow() -> None:
    assert _path_allowed("/Compounds/0/Description")
    assert _path_allowed("/Compounds/0/IntestinalPermeability/0/Name")
    assert _path_allowed("/Simulations/0/Compounds/0/Alternatives/0")
    assert not _path_allowed("/Compounds/0/Processes/0")
    assert not _path_allowed("/Protocols/0/Dose")
