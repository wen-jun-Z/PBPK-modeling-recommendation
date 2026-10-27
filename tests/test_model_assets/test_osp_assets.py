from __future__ import annotations

import binascii
import json
import sqlite3
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from pbpk_kb.model_assets.compression import OSPContentError, decompress_osp_content
from pbpk_kb.model_assets.gate1_audit import _process_participant_errors
from pbpk_kb.model_assets.inventory import build_repository_inventory
from pbpk_kb.model_assets.license_policy import (
    resolve_repository_license,
    sha256_file,
)
from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseStatus,
    ParseStatus,
)
from pbpk_kb.model_assets.osp_project_parser import parse_osp_project
from pbpk_kb.model_assets.osp_pkml_parser import (
    parse_pkml_simulation,
    parse_result_csv,
)
from pbpk_kb.model_assets.report_parser import parse_markdown_report
from pbpk_kb.model_assets.repository_adapter import inspect_repository


def _osp_blob(xml: str, *, filename: str = "Name") -> bytes:
    data = xml.encode("utf-8")
    compressor = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(data) + compressor.flush()
    filename_bytes = filename.encode("utf-8")
    crc = binascii.crc32(data) & 0xFFFFFFFF
    extra = struct.pack("<HHQQ", 0x0001, 16, len(data), len(compressed))
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        45,
        0x0800,
        8,
        0,
        0,
        crc,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(filename_bytes),
        len(extra),
    )
    return header + filename_bytes + extra + compressed


def _create_project(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE PROJECTS (
            Id INTEGER PRIMARY KEY, Name TEXT, Description TEXT,
            Version INTEGER, ContentId INTEGER
        );
        CREATE TABLE CONTENTS (Id INTEGER PRIMARY KEY, Data BLOB);
        CREATE TABLE BUILDING_BLOCKS (
            Id TEXT PRIMARY KEY, Name TEXT, Description TEXT, Icon TEXT,
            Version INTEGER, StructureVersion INTEGER, ContentId INTEGER,
            PropertiesId INTEGER, ProjectId INTEGER
        );
        CREATE TABLE SIMULATIONS (
            SimulationId TEXT PRIMARY KEY, SimulationMode TEXT,
            DataRepositoryId INTEGER, SimulationResultsId INTEGER,
            SimulationAnalysesId INTEGER
        );
        CREATE TABLE USED_BUILDING_BLOCKS (
            Id INTEGER PRIMARY KEY, Name TEXT, TemplateId TEXT, Version INTEGER,
            StructureVersion INTEGER, BuildingBlockType TEXT, Altered INTEGER,
            SimulationId TEXT
        );
        CREATE TABLE USED_OBSERVED_DATA (SimulationId TEXT, ObservedDataId INTEGER);
        CREATE TABLE DATA_REPOSITORIES (
            Id INTEGER PRIMARY KEY, Name TEXT, Description TEXT, ContentId INTEGER
        );
        CREATE TABLE PARAMETER_IDENTIFICATIONS (
            Id INTEGER PRIMARY KEY, Name TEXT, Description TEXT,
            ContentId INTEGER, PropertiesId INTEGER, ProjectId INTEGER
        );
        """
    )
    compound = """<?xml version="1.0" encoding="utf-8"?>
    <Compound id="compound-1" name="Midazolam" icon="Compound" xmlVersion="80">
      <Root>
        <Para id="mw" name="Molecular weight" value="325.8"
              dim="Molecular weight" displayUnit="g/mol" isDefault="0"/>
        <EnzymaticProcess id="p1" name="CYP3A4 metabolism"/>
      </Root>
    </Compound>"""
    protocol = """<?xml version="1.0" encoding="utf-8"?>
    <SimpleProtocol id="protocol-1" name="po 5 mg" icon="Protocol"
      applicationType="Oral" dosingInterval="Single" xmlVersion="80">
      <Para id="dose" name="InputDose" value="5" dim="Mass" displayUnit="mg"
        isDefault="0"/>
    </SimpleProtocol>"""
    simulation = """<?xml version="1.0" encoding="utf-8"?>
    <IndividualSimulation id="simulation-1" name="po 5 mg" icon="Simulation"
      xmlVersion="80">
      <Organ id="liver" name="Liver"/>
      <Container id="drug" name="Midazolam" containerType="Molecule"/>
      <Reaction id="rxn" name="Midazolam-CYP3A4" icon="Metabolism"/>
      <Transport id="t1" name="PassiveDiffusion"/>
      <Observer id="o1" name="Plasma"/>
      <Para id="p1" name="Dose" value="5" dim="Mass" displayUnit="mg"/>
      <Formula id="f1" name="Mass balance"/>
    </IndividualSimulation>"""
    rows = [
        (1, _osp_blob(compound)),
        (2, _osp_blob(protocol)),
        (3, _osp_blob(simulation)),
    ]
    connection.executemany("INSERT INTO CONTENTS (Id, Data) VALUES (?, ?)", rows)
    connection.execute("INSERT INTO CONTENTS (Id, Data) VALUES (4, NULL)")
    connection.execute(
        "INSERT INTO PROJECTS VALUES (1, 'Midazolam-Model', NULL, 80, 1)"
    )
    connection.executemany(
        """
        INSERT INTO BUILDING_BLOCKS
        (Id, Name, Description, Icon, Version, StructureVersion, ContentId,
         PropertiesId, ProjectId)
        VALUES (?, ?, NULL, ?, 0, 0, ?, NULL, 1)
        """,
        [
            ("compound-1", "Midazolam", "Compound", 1),
            ("protocol-1", "po 5 mg", "Protocol", 2),
            ("simulation-1", "po 5 mg", "Simulation", 3),
        ],
    )
    connection.execute(
        "INSERT INTO SIMULATIONS VALUES ('simulation-1', 'Individual', NULL, 1, NULL)"
    )
    connection.execute(
        """
        INSERT INTO USED_BUILDING_BLOCKS
        VALUES (1, 'Midazolam', NULL, 0, 0, 'Compound', 0, 'simulation-1')
        """
    )
    connection.commit()
    connection.close()


def test_decompress_osp_content_validates_size_and_crc() -> None:
    blob = _osp_blob("<Root answer='42'/>")
    decoded = decompress_osp_content(blob)

    assert decoded.filename == "Name"
    assert decoded.data == b"<Root answer='42'/>"
    assert decoded.storage_format == "zip64_deflate"
    assert decoded.crc_valid is True
    assert decoded.uncompressed_size == len(decoded.data)


def test_decompress_osp_content_accepts_inline_utf8_xml() -> None:
    blob = b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><RandomPopulation/>'
    decoded = decompress_osp_content(blob)

    assert decoded.filename == "inline.xml"
    assert decoded.data == blob
    assert decoded.storage_format == "utf8_xml"
    assert decoded.crc_valid is None
    assert decoded.compressed_size == len(blob)


def test_decompress_osp_content_rejects_truncation() -> None:
    with pytest.raises(OSPContentError, match="Compressed size mismatch"):
        decompress_osp_content(_osp_blob("<Root/>")[:-1])


def _classic_osp_blob(xml: str) -> bytes:
    original = _osp_blob(xml)
    header = list(struct.unpack_from("<IHHHHHIIIHH", original))
    filename = original[30:30+header[9]]
    compressed = original[30+header[9]+header[10]:]
    header[1], header[7], header[8], header[10] = 20, len(compressed), len(xml.encode()), 0
    return struct.pack("<IHHHHHIIIHH", *header) + filename + compressed


def test_classic_zip_model_content_keeps_crc_verification():
    blob = _classic_osp_blob("<LegacyModel/>")
    result = decompress_osp_content(blob)
    assert result.storage_format == "zip_deflate"
    assert result.data == b"<LegacyModel/>"
    assert result.crc_valid is True
    damaged = bytearray(blob)
    damaged[14] ^= 1  # CRC, without changing the deflate payload.
    assert decompress_osp_content(bytes(damaged)).crc_valid is False
    with pytest.raises(OSPContentError, match="Compressed size mismatch"):
        decompress_osp_content(blob[:-1])


def test_model_decompression_stops_at_the_declared_size():
    blob = bytearray(_classic_osp_blob("<LegacyModel>" + "a"*10000 + "</LegacyModel>"))
    struct.pack_into("<I", blob, 22, 1)
    with pytest.raises(OSPContentError, match="oversized"):
        decompress_osp_content(bytes(blob))


def test_missing_repository_license_is_unresolved(tmp_path: Path) -> None:
    decision = resolve_repository_license(tmp_path)

    assert decision.status == LicenseStatus.UNRESOLVED
    assert decision.reason_code == "LICENSE_FILE_MISSING"


def test_repository_license_normalizes_gpl_whitespace(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE"
    license_path.write_text(
        "GNU GENERAL PUBLIC LICENSE\n\nVersion 2, June 1991",
        encoding="utf-8",
    )

    decision = resolve_repository_license(tmp_path)

    assert decision.status == LicenseStatus.ALLOWED
    assert decision.spdx_id == "GPL-2.0-only"
    assert decision.source_location == "LICENSE"
    assert decision.source_checksum == sha256_file(license_path)


def test_inventory_accepts_checksum_locked_file_snapshot(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "snapshot"
    repository.mkdir()
    _create_project(repository / "Caffeine.pksim5")
    (repository / "LICENSE").write_text(
        "GNU General Public License Version 2, June 1991",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema_version": "1.0.0",
                "source_id": "licensed_snapshot",
                "repository_url": "https://example.test/locked.git",
                "repository_commit": "a" * 40,
                "assets": [
                    {
                        "relative_path": path.name,
                        "sha256": sha256_file(path),
                        "source_location": (
                            "GITHUB_BLOB:" + "a" * 40 + f"/{path.name}"
                        ),
                    }
                    for path in sorted(repository.iterdir())
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_repository_inventory(
        repository_root=repository,
        output_directory=tmp_path / "inventory",
        source_id="licensed_snapshot",
        repository_url="https://example.test/locked.git",
        repository_commit="a" * 40,
        snapshot_manifest_path=manifest_path,
    )

    assert summary.gate1_static_acceptance is True
    assert summary.repository_commit == "a" * 40
    assert summary.repository_license.status == LicenseStatus.ALLOWED
    assert summary.metadata["repository_snapshot_mode"] == "pinned_files"


def test_repository_inspection_uses_only_local_git_metadata(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "-b", "main", str(repository)], check=True)
    (repository / "README.md").write_text("model library\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "add",
            "origin",
            "https://invalid.example.org/models.git",
        ],
        check=True,
    )
    snapshot = inspect_repository(
        repository,
        source_id="test",
        repository_url="https://invalid.example.org/models.git",
    )

    assert snapshot.default_branch == "main"
    assert snapshot.is_clean is True


def test_parse_osp_project_reads_sqlite_and_compressed_xml(tmp_path: Path) -> None:
    project = tmp_path / "Midazolam-Model.pksim5"
    _create_project(project)

    parsed = parse_osp_project(project, repository_commit="a" * 40)

    assert parsed.sqlite_integrity == "ok"
    assert parsed.project_name == "Midazolam-Model"
    assert parsed.project_version == 80
    assert parsed.compounds == ["Midazolam"]
    assert parsed.compound_details[0].process_names == ["CYP3A4 metabolism"]
    assert parsed.compound_details[0].parameters[0].display_unit == "g/mol"
    assert parsed.protocols[0].application_type == "Oral"
    assert parsed.protocols[0].parameters["InputDose"] == 5.0
    assert parsed.simulations[0].organ_names == ["Liver"]
    assert parsed.simulations[0].reaction_names == ["Midazolam-CYP3A4"]
    assert parsed.simulations[0].parameter_count == 1
    assert parsed.simulations[0].equation_count == 1
    assert parsed.parse_status == ParseStatus.REGISTERED_NOT_EXECUTABLE
    assert (
        parsed.execution_probe_status
        == ExecutionProbeStatus.BLOCKED_RUNTIME_UNAVAILABLE
    )
    assert len(parsed.compressed_content_audit) == 3
    assert parsed.null_content_ids == [4]
    assert all(
        item.crc_valid is not False for item in parsed.compressed_content_audit
    )


def test_parse_pkml_and_csv_preserves_scientific_provenance(
    tmp_path: Path,
) -> None:
    pkml = tmp_path / "simulation.pkml"
    pkml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<SimulationTransfer id="transfer" version="18">
  <Simulation id="embedded-simulation" name="Study, Midazolam po 5 mg">
    <SimulationConfiguration>
      <Individual id="individual-1" name="European adult" icon="Human">
        <Builders>
          <IndividualParameter id="weight" name="Weight" value="70"
            dim="Mass" displayUnit="kg" isDefault="0">
            <ValueOrigin source="Publication" description="Study table 1"/>
          </IndividualParameter>
        </Builders>
      </Individual>
      <MoleculeBuildingBlock>
        <MoleculeBuilder id="drug-1" name="Midazolam" icon="DRUG"
          quantityType="Drug" dim="Amount" displayUnit="µmol"/>
        <MoleculeBuilder id="metabolite-1" name="1-OH-Midazolam"
          quantityType="Metabolite" dim="Amount"/>
        <MoleculeBuilder id="enzyme-1" name="CYP3A4" icon="ENZYME"
          quantityType="Enzyme" dim="Amount"/>
        <MoleculeBuilder id="transporter-1" name="ABCB1" icon="TRANSPORTER"
          quantityType="Transporter" dim="Amount"/>
      </MoleculeBuildingBlock>
      <ReactionBuildingBlock>
        <ReactionBuilder id="reaction-1" name="Midazolam-CYP3A4"
          icon="Metabolism" dimension="Amount per time" formula="formula-1">
          <Children>
            <Parameter id="kcat" name="kcat" value="2.5"
              dim="Inversed time" displayUnit="1/min" isDefault="0"/>
          </Children>
          <Educts>
            <ReactionPartnerBuilder partner="drug-1"
              stoichiometricCoefficient="1"/>
          </Educts>
          <Products>
            <ReactionPartnerBuilder partner="metabolite-1"
              stoichiometricCoefficient="1"/>
          </Products>
          <ModifierNames>
            <ModifierNamesChild>enzyme-1</ModifierNamesChild>
          </ModifierNames>
        </ReactionBuilder>
      </ReactionBuildingBlock>
      <EventGroupBuildingBlock>
        <EventGroupBuilder id="dose-1" name="5 mg PO"
          containerType="EventGroup">
          <Children>
            <EventGroupBuilder id="formulation-1" name="Solution"
              containerType="Formulation">
              <Children>
                <ApplicationBuilder id="application-1" name="Application_1"
                  icon="Oral" eventGroupType="Oral"
                  moleculeName="Midazolam"/>
              </Children>
            </EventGroupBuilder>
          </Children>
        </EventGroupBuilder>
      </EventGroupBuildingBlock>
    </SimulationConfiguration>
    <Model>
      <Container id="liver" name="Liver" containerType="Organ"/>
      <Observer id="observer-1" name="Plasma concentration"/>
      <Solver id="solver-1" name="CVODES"/>
    </Model>
  </Simulation>
</SimulationTransfer>
""",
        encoding="utf-8",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        '"IndividualId","Time [min]","Plasma [µmol/l]"\n'
        '"1","0","0"\n'
        '"1","60","0.5"\n',
        encoding="utf-8",
    )
    csv_summary = parse_result_csv(
        csv_path,
        artifact_relative_path="projects/p/exports/s0001/results.csv",
        source_relative_path="Study, Midazolam po 5 mg-Results.csv",
    )
    parsed = parse_pkml_simulation(
        pkml,
        artifact_relative_path="projects/p/exports/s0001/simulation.pkml",
        artifact_sha256=sha256_file(pkml),
        job_id="osp-exec-" + "a" * 32,
        project_key="b" * 32,
        simulation_id="project-simulation-id",
        simulation_name="Study, Midazolam po 5 mg",
        csv_results=[csv_summary],
    )

    assert parsed.summary.parse_status == "parsed"
    assert parsed.summary.embedded_simulation_id == "embedded-simulation"
    assert parsed.summary.compounds == ["Midazolam"]
    assert parsed.summary.metabolites == ["1-OH-Midazolam"]
    assert parsed.summary.enzymes == ["CYP3A4"]
    assert parsed.summary.transporters == ["ABCB1"]
    assert parsed.summary.species == ["Human"]
    assert parsed.summary.populations == ["European adult"]
    assert parsed.summary.routes == ["Oral"]
    assert parsed.summary.formulations == ["Solution"]
    assert parsed.summary.compartments == ["Liver"]
    assert parsed.summary.solver == "CVODES"
    assert len(parsed.summary.csv_results) == 1
    assert parsed.summary.csv_results[0].row_count == 2
    assert parsed.summary.csv_results[0].time_max == 60
    assert parsed.summary.csv_results[0].non_finite_value_count == 0
    weight = next(item for item in parsed.parameters if item.name == "Weight")
    assert weight.value == 70
    assert weight.value_origin_source == "Publication"
    assert "#element_path=" in weight.source.source_location
    reaction = next(item for item in parsed.processes if item.name == "Midazolam-CYP3A4")
    assert reaction.process_type == "metabolism"
    assert reaction.educts == ["Midazolam"]
    assert reaction.products == ["1-OH-Midazolam"]
    assert reaction.modifiers == ["CYP3A4"]
    assert reaction.educt_identifiers == ["drug-1"]
    assert reaction.product_identifiers == ["metabolite-1"]
    assert reaction.modifier_identifiers == ["enzyme-1"]
    assert reaction.educt_participants[0].component_type == "compound"
    assert reaction.product_participants[0].component_type == "metabolite"
    assert reaction.modifier_participants[0].component_type == "enzyme"
    assert reaction.educt_participants[0].stoichiometric_coefficient == 1
    assert reaction.product_participants[0].stoichiometric_coefficient == 1
    assert reaction.modifier_participants[0].stoichiometric_coefficient is None
    assert reaction.unresolved_partner_identifiers == []
    assert _process_participant_errors(reaction) == []
    assert parsed.administrations[0].formulation == "Solution"


def test_pkml_process_participants_resolve_after_process_closes(
    tmp_path: Path,
) -> None:
    pkml = tmp_path / "late-components.pkml"
    pkml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<SimulationTransfer id="transfer" version="18">
  <Simulation id="embedded-simulation" name="Late component definitions">
    <SimulationConfiguration>
      <ReactionBuildingBlock>
        <ReactionBuilder id="reaction-1" name="Drug metabolism"
          icon="Metabolism">
          <Educts><ReactionPartnerBuilder partner="drug-1"/></Educts>
          <Products><ReactionPartnerBuilder partner="metabolite-1"/></Products>
          <ModifierNames>
            <ModifierNamesChild>enzyme-1</ModifierNamesChild>
            <ModifierNamesChild>missing-component</ModifierNamesChild>
          </ModifierNames>
        </ReactionBuilder>
        <Transport id="transport-1" name="Drug transport"
          sourceAmount="drug-amount" targetAmount="metabolite-amount"/>
      </ReactionBuildingBlock>
      <MoleculeBuildingBlock>
        <MoleculeBuilder id="drug-1" name="Example drug"
          quantityType="Drug"/>
        <MoleculeBuilder id="metabolite-1" name="Example metabolite"
          quantityType="Metabolite"/>
        <MoleculeBuilder id="enzyme-1" name="CYP3A4"
          quantityType="Enzyme"/>
        <MoleculeAmount id="drug-amount" name="Example drug"
          quantityType="Drug"/>
        <MoleculeAmount id="metabolite-amount" name="Example metabolite"
          quantityType="Metabolite"/>
        <MoleculeAmount id="protein-amount" name="Example receptor"
          quantityType="OtherProtein"/>
        <MoleculeAmount id="complex-amount" name="Example complex"
          quantityType="Complex"/>
      </MoleculeBuildingBlock>
    </SimulationConfiguration>
  </Simulation>
</SimulationTransfer>
""",
        encoding="utf-8",
    )

    parsed = parse_pkml_simulation(
        pkml,
        artifact_relative_path="projects/p/exports/s0001/simulation.pkml",
        artifact_sha256=sha256_file(pkml),
        job_id="osp-exec-" + "a" * 32,
        project_key="b" * 32,
        simulation_id="project-simulation-id",
        simulation_name="Late component definitions",
    )

    process = parsed.processes[0]
    assert process.educts == ["Example drug"]
    assert process.products == ["Example metabolite"]
    assert process.modifiers == ["CYP3A4"]
    assert process.unresolved_partner_identifiers == ["missing-component"]
    assert process.modifier_participants[-1].resolution_status == "unresolved"
    assert "UNRESOLVED_PROCESS_PARTICIPANTS" in parsed.summary.warnings
    assert _process_participant_errors(process) == []
    transport = next(
        item for item in parsed.processes if item.name == "Drug transport"
    )
    assert transport.educts == ["Example drug"]
    assert transport.products == ["Example metabolite"]
    component_types = {
        item.name: item.component_type for item in parsed.components
    }
    assert component_types["Example receptor"] == "protein"
    assert component_types["Example complex"] == "molecular_complex"


def test_result_csv_parser_distinguishes_population_and_pk_analysis(
    tmp_path: Path,
) -> None:
    population = tmp_path / "population.csv"
    population.write_text(
        "#Project: Digoxin_Pediatrics\n"
        "#PK-Sim version: 12.3.173\n"
        '"IndividualId","Gender","Population","Weight [kg]"\n'
        '0,MALE,European,8.9\n',
        encoding="utf-8",
    )
    pk_analysis = tmp_path / "pk-analysis.csv"
    pk_analysis.write_text(
        '"IndividualId","QuantityPath","Parameter","Value","Unit"\n'
        '0,"Organism|Blood|Drug","C_max",0.0062,"µmol/l"\n',
        encoding="utf-8",
    )

    population_summary = parse_result_csv(
        population,
        artifact_relative_path="s0001/results_002.csv",
        source_relative_path="Study-Population.csv",
    )
    analysis_summary = parse_result_csv(
        pk_analysis,
        artifact_relative_path="s0001/results_001.csv",
        source_relative_path="Study-PK-Analyses.csv",
    )

    assert population_summary.parse_status == "parsed"
    assert population_summary.result_kind == "population"
    assert population_summary.metadata["Project"] == "Digoxin_Pediatrics"
    assert population_summary.row_count == 1
    assert population_summary.text_value_count == 2
    assert analysis_summary.parse_status == "parsed"
    assert analysis_summary.result_kind == "pk_analysis"
    assert analysis_summary.row_count == 1
    assert analysis_summary.non_finite_value_count == 0


def test_markdown_report_parser_extracts_traceable_tables_and_datasets(
    tmp_path: Path,
) -> None:
    report = tmp_path / "Midazolam_evaluation_report.md"
    report.write_text(
        """# Midazolam model evaluation

## Model Building

| Publication | Arm / Treatment |
| --- | --- |
| Smith 2020 | Oral 5 mg in healthy adults |

## Model Verification

| Scenario | Data reference |
| --- | --- |
| iv 1 mg | Jones 2021 |

## Diagnostics Plots

| Group | GMFE |
| --- | --- |
| All | 1.46 |
""",
        encoding="utf-8",
    )
    parsed = parse_markdown_report(
        report,
        relative_path="Midazolam/Midazolam_evaluation_report.md",
    )

    assert parsed.title == "Midazolam model evaluation"
    assert parsed.table_count == 3
    assert len(parsed.sections) == 4
    assert [item.role for item in parsed.dataset_references] == [
        "calibration",
        "validation",
    ]
    assert parsed.dataset_references[0].source_location.endswith("#L7")
    assert len(parsed.metrics) == 1
    assert parsed.metrics[0].metric_name == "GMFE"
    assert parsed.metrics[0].value == 1.46
