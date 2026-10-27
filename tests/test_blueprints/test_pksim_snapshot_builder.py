from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pbpk_kb.blueprints.pksim_snapshot_builder import (
    CaffeineSnapshotTransformationSpec,
    PKSimSnapshotBuildError,
    RemovedSimulation,
    audit_snapshot_roundtrip,
    build_pksim_system_parameter_manifest,
    transform_caffeine_snapshot,
)
from pbpk_kb.model_assets.execution_plan import write_json_atomic
from pbpk_kb.model_assets.license_policy import sha256_file


SOURCE_COMMIT = "b8f13cc6b2317488e9fa3a969701aef4b2132e5c"
OUTPUT_PATH = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
ORIGINAL_WATER_PATH = (
    "Applications|Oral|Weibull|Application_1|ProtocolSchemaItem|"
    "Volume of water/body weight"
)
CANONICAL_WATER_PATH = (
    "Events|Oral|Weibull|Application_1|ProtocolSchemaItem|"
    "Volume of water/body weight"
)


def _source_snapshot() -> dict:
    compound = {
        "Name": "Caffeine",
        "Processes": [
            {
                "InternalName": "MetabolizationLiverMicrosomes_MM",
                "DataSource": "MM",
                "Molecule": "CYP1A2",
                "Parameters": [
                    {
                        "Name": "In vitro Vmax for liver microsomes",
                        "Value": 109.0,
                        "Unit": "pmol/min/mg mic. protein",
                    },
                    {"Name": "Km", "Value": 14.7, "Unit": "umol/l"},
                ],
            },
            {
                "InternalName": "KidneyClearance",
                "DataSource": "Birkett and Miners 1991",
                "Species": "Human",
                "Parameters": [
                    {"Name": "Body weight", "Value": 73.0, "Unit": "kg"},
                    {
                        "Name": "Volume (kidney)",
                        "Value": 0.4377,
                        "Unit": "l",
                    },
                    {
                        "Name": "Blood flow rate (kidney)",
                        "Value": 1.325,
                        "Unit": "l/min",
                    },
                    {"Name": "Fraction unbound (experiment)", "Value": 0.7},
                    {
                        "Name": "Plasma clearance",
                        "Value": 0.01033051,
                        "Unit": "ml/min/kg",
                    },
                ],
            },
        ],
    }
    simulation_compound = {
        "Name": "Caffeine",
        "Processes": [
            {
                "Name": "Renal Clearances-Birkett and Miners 1991",
                "SystemicProcessType": "Renal",
            }
        ],
    }
    return {
        "Version": 80,
        "Individuals": [
            {"Name": "Ind", "OriginData": {"Species": "Human"}},
            {"Name": "Kid", "ExpressionProfiles": ["CYP3A4|Human|Kid"]},
        ],
        "ExpressionProfiles": [
            {
                "Type": "Enzyme",
                "Species": "Human",
                "Molecule": "CYP3A4",
                "Category": "Kid",
            }
        ],
        "Compounds": [compound],
        "Formulations": [
            {
                "Name": "Weibull",
                "FormulationType": "Formulation_Tablet_Weibull",
                "Parameters": [
                    {
                        "Name": "Dissolution time (50% dissolved)",
                        "Value": 240.0,
                        "Unit": "min",
                    }
                ],
            }
        ],
        "Protocols": [
            {
                "Name": "Oral",
                "ApplicationType": "Oral",
                "DosingInterval": "Single",
                "Parameters": [
                    {"Name": "InputDose", "Value": 1.0, "Unit": "mg/kg"}
                ],
            },
            {
                "Name": "IV",
                "ApplicationType": "IntravenousBolus",
                "DosingInterval": "Single",
                "Parameters": [
                    {"Name": "InputDose", "Value": 1.0, "Unit": "mg/kg"}
                ],
            },
        ],
        "Simulations": [
            {
                "Name": "S1",
                "Individual": "Ind",
                "Model": "4Comp",
                "Solver": {},
                "OutputSchema": [
                    {
                        "Parameters": [
                            {"Name": "Start time", "Value": 0.0, "Unit": "h"},
                            {"Name": "End time", "Value": 24.0, "Unit": "h"},
                        ]
                    }
                ],
                "Parameters": [
                    {"Path": ORIGINAL_WATER_PATH, "Value": 3.5, "Unit": "ml/kg"}
                ],
                "Compounds": [
                    {
                        **json.loads(json.dumps(simulation_compound)),
                        "Protocol": {
                            "Name": "Oral",
                            "Formulations": [
                                {"Key": "Formulation", "Name": "Weibull"}
                            ],
                        },
                    }
                ],
            },
            {
                "Name": "S2",
                "Individual": "Ind",
                "Model": "4Comp",
                "Solver": {},
                "OutputSchema": [],
                "Compounds": [
                    {
                        **json.loads(json.dumps(simulation_compound)),
                        "Protocol": {"Name": "IV"},
                    }
                ],
            },
            {
                "Name": "S3",
                "Individual": "Kid",
                "Compounds": [
                    {
                        **json.loads(json.dumps(simulation_compound)),
                        "Protocol": {"Name": "IV"},
                    }
                ],
            },
        ],
    }


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE tab_molecule_parameters (
          molecule TEXT,
          parameter_name TEXT,
          distribution_type TEXT,
          mean_value REAL,
          standard_deviation REAL,
          min_value REAL,
          min_isallowed INTEGER,
          max_value REAL,
          max_isallowed INTEGER
        );
        CREATE TABLE tab_parameters (
          parameter_name TEXT,
          display_name TEXT,
          description TEXT,
          dimension TEXT,
          default_unit TEXT
        );
        CREATE TABLE tab_calculation_method_rates (
          calculation_method TEXT,
          formula_rate TEXT,
          formula TEXT,
          dimension TEXT
        );
        CREATE TABLE tab_ontogenies (
          molecule TEXT,
          species TEXT,
          group_name TEXT,
          postmenstrual_age REAL,
          ontogeny_factor REAL,
          deviation REAL
        );
        CREATE TABLE tab_container_parameter_rates (
          container_type TEXT,
          container_name TEXT,
          parameter_name TEXT,
          calculation_method TEXT,
          formula_rate TEXT,
          min_value REAL,
          min_isallowed INTEGER,
          max_value REAL,
          max_isallowed INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO tab_molecule_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "CYP1A2",
                "Reference concentration",
                "Normal",
                1.8,
                0.0,
                None,
                1,
                None,
                1,
            ),
            (
                "CYP1A2",
                "t1/2 (liver)",
                "Normal",
                2340.0,
                1440.0,
                None,
                1,
                None,
                1,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO tab_parameters VALUES (?, ?, ?, ?, ?)",
        [
            (
                "Reference concentration",
                "Reference concentration",
                "Source: Rodrigues (1999), Table 2",
                "Concentration (molar)",
                None,
            ),
            (
                "t1/2 (intestine)",
                "t1/2 (intestine)",
                "Protein turnover half-life",
                "Time",
                "min",
            ),
            (
                "t1/2 (liver)",
                "t1/2 (liver)",
                "Protein turnover half-life",
                "Time",
                "min",
            ),
        ],
    )
    connection.execute(
        "INSERT INTO tab_calculation_method_rates VALUES (?, ?, ?, ?)",
        ("CompoundCommon", "PARAM_HalfLife_Intestine", "1380", "Time"),
    )
    connection.execute(
        "INSERT INTO tab_ontogenies VALUES (?, ?, ?, ?, ?, ?)",
        ("CYP1A2", "Human", "ONTOGENY_LIVER_ALL", 1.93, 0.99, 1.52),
    )
    connection.execute(
        "INSERT INTO tab_container_parameter_rates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "GENERAL",
            "PROTEIN",
            "Relative expression",
            "ExpressionParameters",
            "Zero_Rate",
            0.0,
            1,
            None,
            1,
        ),
    )
    connection.commit()
    connection.close()


def _manifest(tmp_path: Path):
    source_project = tmp_path / "SimpleProject_730.pksim5"
    source_snapshot = tmp_path / "SimpleProject_730.json"
    database = tmp_path / "PKSimDB.sqlite"
    protein_factory = tmp_path / "IndividualProteinFactory.cs"
    license_path = tmp_path / "OSP-Suite-License.pdf"
    source_project.write_bytes(b"project")
    source_snapshot.write_text(
        json.dumps(_source_snapshot()), encoding="utf-8"
    )
    _database(database)
    protein_factory.write_text("Localization.Intracellular", encoding="utf-8")
    license_path.write_bytes(b"license")
    return build_pksim_system_parameter_manifest(
        source_project_path=source_project,
        source_snapshot_path=source_snapshot,
        pksim_database_path=database,
        protein_factory_source_path=protein_factory,
        license_path=license_path,
        source_root=tmp_path,
        source_commit=SOURCE_COMMIT,
        pksim_version="12.3.173",
    )


def _spec(source_snapshot_path: Path, manifest) -> CaffeineSnapshotTransformationSpec:
    return CaffeineSnapshotTransformationSpec(
        transformation_id="test-caffeine-cyp1a2",
        created_at=datetime.now(UTC),
        source_commit=SOURCE_COMMIT,
        pksim_version="12.3.173",
        source_project_sha256="1" * 64,
        source_snapshot_sha256=sha256_file(source_snapshot_path),
        system_parameter_manifest_sha256=manifest.checksum,
        expression_profile_category="PBPK-KB-Caffeine-Adult-v1",
        target_individual_names=["Ind"],
        target_simulation_names=["S1", "S2"],
        expected_simulation_protocols={"S1": "Oral", "S2": "IV"},
        removed_simulations=[
            RemovedSimulation(
                name="S3",
                reason_code="OUT_OF_SCOPE_PEDIATRIC_CYP3A4_TEST_FIXTURE",
            )
        ],
        simulation_parameter_path_migrations=[
            {
                "simulation_name": "S1",
                "original_path": ORIGINAL_WATER_PATH,
                "canonical_path": CANONICAL_WATER_PATH,
                "reason_code": (
                    "PKSIM_12_3_SNAPSHOT_IMPORT_CANONICAL_EVENT_PATH"
                ),
            }
        ],
        output_selections=[OUTPUT_PATH],
    )


def test_snapshot_transformation_is_whitelisted_and_traceable(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "SimpleProject_730.json"
    source_path.write_text(json.dumps(_source_snapshot()), encoding="utf-8")
    manifest = _manifest(tmp_path)
    spec = _spec(source_path, manifest)

    transformed, audit = transform_caffeine_snapshot(
        source_snapshot=_source_snapshot(),
        spec=spec,
        manifest=manifest,
    )

    assert audit.accepted
    assert audit.changed_paths == [
        "/ExpressionProfiles",
        "/Individuals/0/ExpressionProfiles",
        "/Simulations",
    ]
    assert [item["Name"] for item in transformed["Simulations"]] == [
        "S1",
        "S2",
    ]
    profile = transformed["ExpressionProfiles"][-1]
    assert profile["Molecule"] == "CYP1A2"
    profile_parameters = {
        item["Path"]: item for item in profile["Parameters"]
    }
    assert profile_parameters["CYP1A2|Reference concentration"]["Value"] == 1.8
    assert profile_parameters["CYP1A2|t1/2 (intestine)"]["Value"] == 23.0
    assert profile_parameters["CYP1A2|t1/2 (liver)"]["Value"] == 39.0
    assert transformed["Simulations"][0]["Parameters"][0]["Path"] == (
        CANONICAL_WATER_PATH
    )
    for simulation in transformed["Simulations"]:
        assert {"Name": "CYP1A2-MM", "MoleculeName": "CYP1A2"} in (
            simulation["Compounds"][0]["Processes"]
        )
        assert OUTPUT_PATH in simulation["OutputSelections"]


def test_snapshot_transformation_rejects_missing_source_process(
    tmp_path: Path,
) -> None:
    source = _source_snapshot()
    source["Compounds"][0]["Processes"] = []
    source_path = tmp_path / "SimpleProject_730.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    manifest = _manifest(tmp_path)
    spec = _spec(source_path, manifest)

    with pytest.raises(PKSimSnapshotBuildError, match="process definition"):
        transform_caffeine_snapshot(
            source_snapshot=source,
            spec=spec,
            manifest=manifest,
        )


def test_roundtrip_audit_compares_model_semantics(tmp_path: Path) -> None:
    source_path = tmp_path / "SimpleProject_730.json"
    source_path.write_text(json.dumps(_source_snapshot()), encoding="utf-8")
    manifest = _manifest(tmp_path)
    spec = _spec(source_path, manifest)
    transformed, _ = transform_caffeine_snapshot(
        source_snapshot=_source_snapshot(),
        spec=spec,
        manifest=manifest,
    )
    transformed_path = tmp_path / "transformed.json"
    roundtrip_path = tmp_path / "roundtrip.json"
    write_json_atomic(transformed_path, transformed)
    roundtrip = json.loads(json.dumps(transformed))
    roundtrip["Compounds"].reverse()
    roundtrip["Protocols"].reverse()
    roundtrip["Simulations"].reverse()
    kidney = next(
        process
        for process in roundtrip["Compounds"][0]["Processes"]
        if process["InternalName"] == "KidneyClearance"
    )
    kidney["Parameters"] = [
        item
        for item in kidney["Parameters"]
        if item["Name"]
        not in {
            "Blood flow rate (kidney)",
            "Body weight",
            "Volume (kidney)",
        }
    ]
    write_json_atomic(roundtrip_path, roundtrip)

    audit = audit_snapshot_roundtrip(
        transformed_snapshot_path=transformed_path,
        roundtrip_snapshot_path=roundtrip_path,
        spec=spec,
    )

    assert audit.accepted
    assert audit.semantic_differences == []
    assert audit.accepted_normalizations
    assert (
        audit.transformed_semantic_sha256
        == audit.roundtrip_semantic_sha256
    )


def test_roundtrip_audit_rejects_dose_drift(tmp_path: Path) -> None:
    source_path = tmp_path / "SimpleProject_730.json"
    source_path.write_text(json.dumps(_source_snapshot()), encoding="utf-8")
    manifest = _manifest(tmp_path)
    spec = _spec(source_path, manifest)
    transformed, _ = transform_caffeine_snapshot(
        source_snapshot=_source_snapshot(),
        spec=spec,
        manifest=manifest,
    )
    changed = json.loads(json.dumps(transformed))
    oral_protocol = next(
        item for item in changed["Protocols"] if item["Name"] == "Oral"
    )
    input_dose = next(
        item
        for item in oral_protocol["Parameters"]
        if item["Name"] == "InputDose"
    )
    input_dose["Value"] = 2.0
    transformed_path = tmp_path / "transformed.json"
    roundtrip_path = tmp_path / "roundtrip.json"
    write_json_atomic(transformed_path, transformed)
    write_json_atomic(roundtrip_path, changed)

    audit = audit_snapshot_roundtrip(
        transformed_snapshot_path=transformed_path,
        roundtrip_snapshot_path=roundtrip_path,
        spec=spec,
    )

    assert not audit.accepted
    assert not audit.checks["complete_scientific_snapshot_equal"]
    assert any("Protocols" in path for path in audit.semantic_differences)
