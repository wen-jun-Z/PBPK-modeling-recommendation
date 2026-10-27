from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import UUID

import pytest

from pbpk_kb.blueprints.builder import (
    derive_compartments,
    derive_equation_templates,
    derive_processes,
)
from pbpk_kb.blueprints.schemas import (
    BlueprintTemplate,
    ParameterSlot,
    TargetRuntime,
)
from pbpk_kb.blueprints.template_catalog import audit_template_skeleton
from pbpk_kb.blueprints.template_registration import (
    BlueprintTemplateRegistrationSpec,
    TemplateRegistrationError,
    register_blueprint_template,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import LicenseDecision, LicenseStatus
from pbpk_kb.model_assets.osp_pkml_parser import parse_pkml_simulation


COMPOUND_ID = UUID("6ef60faa-3f99-559d-bc2a-7d9da4764fce")


def _write_skeleton(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<SimulationTransfer id="transfer" version="18">
  <Simulation id="simulation-1" name="Caffeine blueprint test">
    <SimulationConfiguration>
      <Individual id="individual-1" name="European adult" icon="Human"/>
      <MoleculeBuildingBlock>
        <MoleculeBuilder id="caffeine-builder" name="Caffeine"
          icon="DRUG" quantityType="Drug"/>
        <MoleculeBuilder id="metabolite-builder" name="Caffeine metabolite"
          quantityType="Metabolite"/>
        <MoleculeBuilder id="enzyme-builder" name="CYP1A2"
          icon="ENZYME" quantityType="Enzyme"/>
      </MoleculeBuildingBlock>
      <ReactionBuildingBlock>
        <ReactionBuilder id="builder-only" name="Builder metabolism"
          icon="Metabolism" formula="builder-formula">
          <Educts>
            <ReactionPartnerBuilder partner="caffeine-builder"
              stoichiometricCoefficient="1"/>
          </Educts>
          <Products>
            <ReactionPartnerBuilder partner="metabolite-builder"
              stoichiometricCoefficient="1"/>
          </Products>
        </ReactionBuilder>
      </ReactionBuildingBlock>
      <EventGroupBuildingBlock>
        <EventGroupBuilder id="formulation" name="Solution"
          containerType="Formulation">
          <ApplicationBuilder id="dose" name="Dose" eventGroupType="Oral"
            moleculeName="Caffeine"/>
        </EventGroupBuilder>
      </EventGroupBuildingBlock>
    </SimulationConfiguration>
    <Model id="model">
      <Container id="lumen" name="Lumen">
        <MoleculeAmount id="lumen-caffeine" name="Caffeine"
          quantityType="Drug"/>
      </Container>
      <Container id="liver" name="Liver">
        <Container id="cell" name="Intracellular">
          <MoleculeAmount id="cell-caffeine" name="Caffeine"
            quantityType="Drug"/>
          <MoleculeAmount id="cell-metabolite" name="Caffeine metabolite"
            quantityType="Metabolite"/>
          <MoleculeAmount id="cell-enzyme" name="CYP1A2"
            quantityType="Enzyme"/>
          <Reaction id="metabolism-1" name="Caffeine-CYP1A2-MM"
            icon="Metabolism" dimension="Amount per time"
            formula="metabolism-formula">
            <Educts>
              <ReactionPartner partner="cell-caffeine"
                stoichiometricCoefficient="1"/>
            </Educts>
            <Products>
              <ReactionPartner partner="cell-metabolite"
                stoichiometricCoefficient="1"/>
            </Products>
            <ModifierNames>
              <ModifierNamesChild>cell-enzyme</ModifierNamesChild>
            </ModifierNames>
          </Reaction>
        </Container>
      </Container>
      <Transport id="absorption-1" name="DrugAbsorption"
        dimension="Amount per time" formula="absorption-formula"
        sourceAmount="lumen-caffeine" targetAmount="cell-caffeine"/>
    </Model>
  </Simulation>
</SimulationTransfer>
""",
        encoding="utf-8",
    )


def _records(path: Path):
    checksum = sha256_file(path)
    return parse_pkml_simulation(
        path,
        artifact_relative_path=path.name,
        artifact_sha256=checksum,
        job_id="template-test",
        project_key=checksum[:32],
        simulation_id="simulation-1",
        simulation_name="Caffeine blueprint test",
    )


def _template(path: Path, process_counts: dict[str, int]) -> BlueprintTemplate:
    records = _records(path)
    compartments = derive_compartments(records)
    return BlueprintTemplate(
        template_id="caffeine-test",
        template_version="1.0.0",
        target_runtime=TargetRuntime(
            adapter="ospsuite_r",
            ospsuite_r_version="12.4.3",
            package_sha256="a" * 64,
            r_version="4.4.3",
            dotnet_runtime_version="8.0.29",
            dotnet_release_metadata_sha256="e" * 64,
            platform="linux-x86_64",
        ),
        source_skeleton_relative_path=path.name,
        source_skeleton_sha256=sha256_file(path),
        source_project_relative_path="source.pksim5",
        source_project_sha256="b" * 64,
        source_repository="https://example.test/repository",
        source_tag="v1",
        source_commit="c" * 40,
        license_id="GPL-2.0-only",
        license_relative_path="LICENSE",
        license_sha256="d" * 64,
        license_decision=LicenseDecision(
            status=LicenseStatus.ALLOWED,
            spdx_id="GPL-2.0-only",
            source_location="LICENSE",
            source_checksum="d" * 64,
            reason_code="REPOSITORY_LICENSE_EXPLICIT",
        ),
        species="Human",
        skeleton_simulation_id="simulation-1",
        skeleton_simulation_name="Caffeine blueprint test",
        skeleton_compound_names=["Caffeine"],
        skeleton_instantiated_process_count=sum(process_counts.values()),
        skeleton_process_type_counts=process_counts,
        compound_semantic_ids=[f"compound:{COMPOUND_ID}"],
        model_structure="whole_body",
        supported_objectives=["base_pk"],
        supported_routes=["Oral"],
        supported_process_types=list(process_counts),
        required_compartment_ids=[
            item.compartment_id for item in compartments
        ],
        parameter_slots=[
            ParameterSlot(
                slot_id="weight",
                semantic_id="parameter:body_weight",
                model_path="Organism|Weight",
                dimension="mass",
                canonical_unit="kg",
                template_value=70,
                source_location="PKML:skeleton#parameter=weight",
                source_checksum=sha256_file(path),
                minimum_value=1,
                maximum_value=300,
                compound_specific=False,
            )
        ],
        equation_templates=derive_equation_templates(records),
        validation_scenario_ids=["simulation-1"],
        validation_status="structural_template_runtime_verified",
    )


def test_full_topology_uses_only_instantiated_processes(tmp_path: Path) -> None:
    path = tmp_path / "skeleton.pkml"
    _write_skeleton(path)
    records = _records(path)

    equations = derive_equation_templates(records)
    compartments = derive_compartments(records)
    processes = derive_processes(
        records=records,
        compound_semantic_id=f"compound:{COMPOUND_ID}",
        compound_names={"caffeine"},
        route="Oral",
    )

    assert len(equations) == 2
    assert {item.process_id for item in processes} == {
        "absorption-1",
        "metabolism-1",
    }
    assert "builder-only" not in {item.process_id for item in processes}
    metabolism = next(
        item for item in processes if item.process_type == "metabolism"
    )
    assert (
        metabolism.input_compartment_id
        == metabolism.output_compartment_id
        == "cell"
    )
    assert metabolism.enzyme_semantic_id == "enzyme:cyp1a2"
    assert {item.compartment_id for item in compartments} == {
        "lumen",
        "liver",
        "cell",
    }


def test_template_audit_rejects_claimed_builder_capability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skeleton.pkml"
    _write_skeleton(path)
    actual_counts = Counter(
        item.process_type
        for item in derive_processes(
            records=_records(path),
            compound_semantic_id=f"compound:{COMPOUND_ID}",
            compound_names={"caffeine"},
            route="Oral",
        )
    )
    accepted = audit_template_skeleton(
        template=_template(path, dict(actual_counts)),
        asset_root=tmp_path,
    )
    assert accepted.accepted

    false_counts = {"absorption": 1, "metabolism": 2}
    rejected = audit_template_skeleton(
        template=_template(path, false_counts),
        asset_root=tmp_path,
    )
    assert not rejected.accepted
    assert "PROCESS_INVENTORY_MISMATCH" in rejected.errors


def test_template_registration_derives_capabilities_from_pkml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skeleton.pkml"
    _write_skeleton(path)
    checksum = sha256_file(path)
    (tmp_path / "source.pksim5").write_bytes(b"project")
    (tmp_path / "LICENSE").write_text(
        "GNU General Public License version 2",
        encoding="utf-8",
    )
    spec = BlueprintTemplateRegistrationSpec(
        template_id="registered-caffeine",
        template_version="1.0.0",
        target_runtime=TargetRuntime(
            adapter="ospsuite_r",
            ospsuite_r_version="12.4.3",
            package_sha256="a" * 64,
            r_version="4.4.3",
            dotnet_runtime_version="8.0.29",
            dotnet_release_metadata_sha256="e" * 64,
            platform="linux-x86_64",
        ),
        source_skeleton_relative_path=path.name,
        source_project_relative_path="source.pksim5",
        source_repository="https://example.test/repository",
        source_tag="v1",
        source_commit="c" * 40,
        license_id="GPL-2.0-only",
        license_relative_path="LICENSE",
        species="Human",
        skeleton_simulation_id="simulation-1",
        skeleton_simulation_name="Caffeine blueprint test",
        compound_semantic_ids=["compound:test-caffeine"],
        model_structure="whole_body",
        supported_objectives=["base_pk"],
        parameter_slots=[
            ParameterSlot(
                slot_id="weight",
                semantic_id="parameter:body-weight",
                model_path="Organism|Weight",
                dimension="mass",
                canonical_unit="kg",
                template_value=73.0,
                source_location=(
                    "PKML:skeleton.pkml#parameter=Organism%7CWeight"
                ),
                source_checksum=checksum,
                compound_specific=False,
            )
        ],
        validation_scenario_ids=["simulation-1"],
    )

    unlicensed_source = tmp_path / "unlicensed-source"
    unlicensed_source.mkdir()
    with pytest.raises(
        TemplateRegistrationError,
        match="LICENSE_FILE_MISSING",
    ):
        register_blueprint_template(
            spec=spec,
            asset_root=tmp_path,
            source_repository_root=unlicensed_source,
        )

    template, audit = register_blueprint_template(
        spec=spec,
        asset_root=tmp_path,
        source_repository_root=tmp_path,
    )

    assert audit.accepted is True
    assert template.skeleton_instantiated_process_count == 2
    assert template.supported_process_types == [
        "absorption",
        "metabolism",
    ]
