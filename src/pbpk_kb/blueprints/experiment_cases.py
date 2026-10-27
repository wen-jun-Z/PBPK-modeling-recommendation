"""Source-locked historical development scenarios, never a blinded benchmark.

Blanchard IV and PO are one calibration study/cohort. Wahllander is a separate
study for development selection, previously called an internal holdout. Their
observations share a published PK-Sim container: per-study exports govern which
records may be read; different export checksums do not establish independence.
Only the selected Wahllander repository is decoded. Final-test repositories,
mixed observation files, old fitted results and executable structures are never
loaded. No compiler, solver, fitting process or GPU is invoked here.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as SafeET

from pbpk_kb.model_assets.compression import decompress_osp_content
from .caffeine_dataset import (
    DATASET_VERSION, _UnitTable, _canonical_concentration, _source_unit,
    _stable_id, decode_nrbf_single_array,
)
from .development_case import (
    DevelopmentCaseError, _ASSETS, _INPUTS, _canonical_checksum, _one,
    _plain, _reference, _verify, make_caffeine_development_case,
)


_DIRECTORY = "experiments/blueprint_design/development_inputs/caffeine_studies_v1"
_CAL_STUDY = "blanchard_1983_absolute_bioavailability"
_SEL_STUDY = "wahllander_1989_caffeine_iv"
_TARGET = "caffeine_parent_plasma"
_OUTPUT = "Organism|PeripheralVenousBlood|Caffeine|Plasma (Peripheral Venous Blood)"
_SPECS = (
    ("blanchard_1983_iv_5mgkg", _CAL_STUDY, "calibration", "CAL_BLANCHARD_IV", 17, "F-rS1HLuukmHRkKxQeGHlw", 273),
    ("blanchard_1983_oral_5mgkg", _CAL_STUDY, "calibration", "CAL_BLANCHARD_PO", 11, "HzJXqJlvoU2p5p0GiaE-8g", 274),
    ("wahllander_1989_iv_3mgkg", _SEL_STUDY, "selection", "HOLD_WAHLLANDER_IV", 7, "6fUaQcC6TkG1PU6ROd4zQA", 282),
)
_FORBIDDEN = ("eudract_2011_002291_16", "pmc5350996_synthetic_caffeine", "pmc5724581_caffeinated_beverage")


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _check_observations(points: list[dict], dataset: dict, count: int) -> None:
    if (len(points) != count or len({p["observation_id"] for p in points}) != count
            or {p["observation_id"] for p in points} != set(dataset["observation_ids"])):
        raise DevelopmentCaseError("Study observation identities or point count changed.")
    for point in points:
        for field in ("time_min", "concentration_umol_per_l"):
            value = point[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise DevelopmentCaseError("Development observations must be finite and positive.")
    if any(a["time_min"] >= b["time_min"] for a, b in zip(points, points[1:])):
        raise DevelopmentCaseError("Study times must be strictly increasing.")


def _selection_observations(root: Path, container: dict, dimensions: dict) -> list[dict]:
    """Decode exactly the seven Wahllander records, never other repositories."""
    container_path = _ASSETS + container["source_location"]
    dimensions_path = _ASSETS + dimensions["source_location"]
    _verify(root, container_path, container["sha256"])
    _verify(root, dimensions_path, dimensions["sha256"])
    units = _UnitTable(root / dimensions_path)
    connection = sqlite3.connect((root / container_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            "SELECT d.Id,d.Name,d.ContentId,c.Data FROM DATA_REPOSITORIES d "
            "JOIN CONTENTS c ON c.Id=d.ContentId WHERE d.Id=?",
            (_SPECS[2][5],),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1 or tuple(rows[0][:3]) != (_SPECS[2][5], "Wahlländer_1989_IV_3mgkg", 282):
        raise DevelopmentCaseError("The selected Wahllander repository identity changed.")
    row = rows[0]
    decoded = decompress_osp_content(bytes(row[3]))
    if decoded.crc_valid is False:
        raise DevelopmentCaseError("Wahllander source CRC failed.")
    tree = SafeET.fromstring(decoded.data)
    columns = tree.find("Columns")
    if columns is None:
        raise DevelopmentCaseError("Wahllander source has no columns.")
    grids = [c for c in columns if c.tag == "McBaseGrid"]
    values = [c for c in columns if c.tag == "McDataColumn" and c.find("DataInfo") is not None
              and c.find("DataInfo").get("origin") == "Observation"]
    if len(grids) != 1 or len(values) != 1:
        raise DevelopmentCaseError("Ambiguous Wahllander concentration or time column.")
    grid, value_column = grids[0], values[0]
    raw_times = decode_nrbf_single_array(grid.findtext("Values", ""))
    raw_values = decode_nrbf_single_array(value_column.findtext("Values", ""))
    if len(raw_times) != 7 or len(raw_values) != 7:
        raise DevelopmentCaseError("Wahllander repository must contain exactly seven points.")
    time_unit, value_unit = _source_unit(grid, units), _source_unit(value_column, units)
    location = (f"DATA_REPOSITORIES.Id={row[0]};CONTENTS.Id={row[2]};"
                f"McDataRepository/Columns/McDataColumn[@id='{value_column.attrib['id']}']/Values")
    result = []
    for index, (time, value) in enumerate(zip(raw_times, raw_values, strict=True)):
        result.append({
            "observation_id": _stable_id("obs", DATASET_VERSION, _SPECS[2][0], index),
            "time_min": units.from_base("Time", "min", time),
            "concentration_umol_per_l": _canonical_concentration(
                units.from_base(value_column.attrib["dimension"], value_unit, value), value_unit),
            "unit": "umol/l", "statistic": "arithmetic_mean",
            "provenance": [_reference(container_path, container["sha256"], location + f";row_index={index}")],
            "extraction": {
                "repository_id": row[0], "sqlite_content_id": row[2], "row_index": index,
                "decoded_xml_sha256": hashlib.sha256(decoded.data).hexdigest(),
                "time_column_id": grid.attrib["id"], "value_column_id": value_column.attrib["id"],
                "time_values_sha256": hashlib.sha256(base64.b64decode(grid.findtext("Values", ""), validate=True)).hexdigest(),
                "values_sha256": hashlib.sha256(base64.b64decode(value_column.findtext("Values", ""), validate=True)).hexdigest(),
                "source_time_unit": time_unit, "source_concentration_unit": value_unit,
                "dimensions_source": _reference(dimensions_path, dimensions["sha256"], "Dimension unit conversion"),
            },
        })
    return result


def _study_documents(root: Path) -> tuple[dict, dict, dict]:
    """Reconstruct only the allowed records from their locked original lineage."""
    config = _verify(root, *_INPUTS["calibration_config"], parse=True)
    manifest = _verify(root, *_INPUTS["dataset_manifest"], parse=True)
    container = _one(manifest["source_artifacts"], "artifact_id", "pmc8302243_caffeine_pksim5")
    dimensions = _one(manifest["source_artifacts"], "artifact_id", "ospsuite_dimensions_v12_4_3")
    if any(s.get("split") != "calibration" for s in config["calibration_scenarios"]):
        raise DevelopmentCaseError("Non-calibration records in the calibration-only source.")
    docs: dict[str, dict] = {}
    for dataset_id, study_id, role, _, count, repository_id, content_id in _SPECS:
        dataset = _one(manifest["datasets"], "dataset_id", dataset_id)
        if dataset["study_id"] != study_id or dataset["split"] != ("calibration" if role == "calibration" else "internal_holdout"):
            raise DevelopmentCaseError("Source study grouping or historical role changed.")
        if role == "calibration":
            old = _one(config["calibration_scenarios"], "dataset_id", dataset_id)
            points = []
            for item in old["observations"]:
                expected_location = f"DATA_REPOSITORIES.Id={repository_id};CONTENTS.Id={content_id};"
                if (item["dataset_id"] != dataset_id or item["split"] != "calibration"
                        or item["source_artifact_id"] != container["artifact_id"]
                        or not item["use_in_objective"] or item.get("exclusion_reason") is not None
                        or expected_location not in item["source_location"]):
                    raise DevelopmentCaseError("Calibration point has an unexpected source or exclusion.")
                points.append({
                    "observation_id": item["observation_id"], "time_min": item["time_min"],
                    "concentration_umol_per_l": item["value_umol_per_l"], "unit": "umol/l",
                    "statistic": "arithmetic_mean",
                    "provenance": [
                        _reference(*_INPUTS["calibration_config"], f"calibration_scenarios[{dataset_id}].observations[{item['observation_id']}]"),
                        _reference(_ASSETS + container["source_location"], container["sha256"], item["source_location"]),
                    ],
                })
        else:
            points = _selection_observations(root, container, dimensions)
        _check_observations(points, dataset, count)
        doc = docs.setdefault(study_id, {
            "schema_version": "caffeine-development-study-1.0.0", "study_group_id": study_id,
            "cohort_ids": [study_id], "role": role, "purpose": "historical_development_reconstruction",
            "source_governance": {
                "original_container_shared": True, "independence_unit": "study_and_cohort_not_file_checksum",
                "original_container": {"path": _ASSETS + container["source_location"], "sha256": container["sha256"],
                                       "license": container["license"], "redistribution": container["redistribution"]},
                "derived_export_establishes_independence": False, "blinded_independent_test": False,
                "role_change": "historical_internal_holdout_to_development_selection" if role == "selection" else None,
            }, "datasets": [],
        })
        doc["datasets"].append({
            "dataset_id": dataset_id, "study_group_id": study_id, "publication_id": dataset["publication_id"],
            "population": dataset["population"], "subject_count": dataset["subject_count"],
            "route": dataset["route"], "regimen": dataset["regimen"],
            "repository_id": repository_id, "content_id": content_id,
            "discrepancy_codes": dataset["discrepancy_codes"], "observations": points,
        })
    return docs, manifest, dimensions


def prepare_caffeine_experiment_inputs(project_root: Path) -> dict[str, str]:
    """Register two deterministic study exports, refusing to overwrite changes.

    This explicit preparation writes only the fixed project development-input
    directory. It is idempotent for identical bytes and does not run a model.
    """
    root = Path(project_root).resolve()
    docs, _, _ = _study_documents(root)
    directory = (root / _DIRECTORY).resolve()
    if not directory.is_relative_to(root):
        raise DevelopmentCaseError("Study export directory escapes the project.")
    directory.mkdir(parents=True, exist_ok=True)
    registered = {}
    for study_id, doc in docs.items():
        relative = f"{_DIRECTORY}/{study_id}.json"
        path = root / relative
        if path.is_symlink():
            raise DevelopmentCaseError("Study export must not be a symlink.")
        encoded = _encoded(doc)
        checksum = hashlib.sha256(encoded).hexdigest()
        if path.exists():
            _verify(root, relative, checksum)
        else:
            with path.open("xb") as stream:
                stream.write(encoded)
        registered[relative] = checksum
    return registered


def make_caffeine_experiment_case(project_root: Path) -> dict[str, Any]:
    """Return joint IV/PO calibration and separate-study selection; read only.

    Run ``prepare_caffeine_experiment_inputs`` once to register the derived input
    records. Every call checks their bytes against re-extraction from the locked
    sources. Selection observations are returned only in the selection scenario,
    never in the top-level design request or parameter-search configuration.
    """
    root = Path(project_root).resolve()
    common = make_caffeine_development_case(root)
    docs, manifest, dimensions = _study_documents(root)
    snapshot = _verify(root, *_INPUTS["snapshot"], parse=True)
    references, audit = {}, copy.deepcopy(common["source_audit"])
    for study_id, doc in docs.items():
        relative, checksum = f"{_DIRECTORY}/{study_id}.json", hashlib.sha256(_encoded(doc)).hexdigest()
        _verify(root, relative, checksum)
        references[study_id] = {
            "study_id": study_id, "source_id": relative, "source_checksum": checksum,
            "role": doc["role"], "cohort_ids": [study_id], "observed_target_ids": [_TARGET],
        }
        audit.append({"artifact_id": study_id, "path": relative, "sha256": checksum,
                      "verification": "passed", "operation": "verified_study_record_export",
                      "source_governance": copy.deepcopy(doc["source_governance"])})
    audit.append({"artifact_id": dimensions["artifact_id"], "path": _ASSETS + dimensions["source_location"],
                  "sha256": dimensions["sha256"], "verification": "passed", "operation": "unit_conversion"})
    scenarios = []
    for dataset_id, study_id, role, sim_name, count, _, _ in _SPECS:
        request = copy.deepcopy(common["design_request"])
        instance = request["instance"]
        individual = _plain(_one(snapshot["Individuals"], "Name", "IND_" + sim_name))
        individual["OriginData"].pop("CalculationMethods", None)
        protocol = _plain(_one(snapshot["Protocols"], "Name", "PROTO_" + sim_name))
        simulation = _one(snapshot["Simulations"], "Name", sim_name)
        is_oral = dataset_id == _SPECS[1][0]
        expected_dose, expected_duration = (3.0, 10.0) if role == "selection" else (5.0, 30.0)
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        if dose["Value"] != expected_dose or dose["Unit"] != "mg/kg":
            raise DevelopmentCaseError("Scenario dose differs from its verified study.")
        if not is_oral:
            duration = _one(protocol["Parameters"], "Name", "Infusion time")
            if protocol["ApplicationType"] != "Intravenous" or duration != {"Name": "Infusion time", "Value": expected_duration, "Unit": "min"}:
                raise DevelopmentCaseError("Scenario infusion differs from its verified study.")
            instance["formulation"] = instance["formulation_source"] = None
        instance.update(
            individual=individual, protocol=protocol,
            generated_demographics=[key for key in ("Weight", "Height") if key not in individual["OriginData"]],
            simulation_name="Caffeine_" + sim_name + "_Development",
            human_source=_reference(*_INPUTS["snapshot"], f"Individuals[{individual['Name']}]; unreported demographics generated by locked PK-Sim"),
            protocol_source=_reference(*_INPUTS["snapshot"], f"Protocols[{protocol['Name']}]; PMC8302243 Supinfo S2 Table S3; oral water volume is a model assumption"),
            output_schema=_plain(simulation["OutputSchema"]),
            output_selections=copy.deepcopy(simulation["OutputSelections"]),
        )
        if instance["output_selections"] != [_OUTPUT]:
            raise DevelopmentCaseError("Unexpected scenario output target.")
        request["requirement"].update(route="oral" if is_oral else "intravenous", dose=expected_dose,
                                      formulation=instance["formulation"]["Name"] if is_oral else None)
        request["studies"] = [copy.deepcopy(references[study_id])]
        dataset = _one(docs[study_id]["datasets"], "dataset_id", dataset_id)
        points = copy.deepcopy(dataset["observations"])
        scenarios.append({
            "scenario_id": dataset_id, "dataset_id": dataset_id, "study_group_id": study_id,
            "role": role, "design_request": request, "target_output": _OUTPUT,
            "time_unit": "min", "unit": "umol/l", "matrix": "plasma", "target_id": _TARGET,
            "observation_count": count, "observations": points, "checksum": _canonical_checksum(points),
            "source_audit": {"study_reference": references[study_id],
                             "source_governance": copy.deepcopy(docs[study_id]["source_governance"]),
                             "discrepancy_codes": dataset["discrepancy_codes"],
                             "subject_count": dataset["subject_count"], "statistic": "arithmetic_mean"},
        })
    search = copy.deepcopy(common["parameter_search"])
    search["dataset_ids"] = [spec[0] for spec in _SPECS if spec[2] == "calibration"]
    search["selection_dataset_ids"] = [_SPECS[2][0]]
    search["objective"] = "study_curve_equal_log_rmse"
    search["aggregation"] = "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))"
    search["gmfe_aggregation"] = "exp(mean_study(mean_curve(mean_point(abs(log(Cpred/Cobs))))))"
    search["selection_usage"] = "rank_completed_candidates_only; never fit parameters to selection observations"
    return {
        "schema_version": "caffeine-development-experiment-1.0.0", "case_id": "caffeine_blanchard_joint_wahllander_selection",
        "purpose": "historical_development_reconstruction", "compound_id": common["design_request"]["requirement"]["compound_semantic_id"],
        "requirement": copy.deepcopy(scenarios[1]["design_request"]["requirement"]),
        "design_request": copy.deepcopy(scenarios[1]["design_request"]),
        "parameter_search": search, "scenarios": scenarios, "source_audit": audit,
        "forbidden_study_group_ids": list(_FORBIDDEN),
        "study_counts": {"calibration": 1, "selection": 1, "test_loaded": 0},
        "observation_counts": {"calibration": 28, "selection": 7, "test_loaded": 0},
        "limitations": [
            "Historical development reconstruction, not a newly blinded or independent paper benchmark.",
            "Blanchard IV17 and PO11 are the same study/cohort and always remain together in calibration.",
            "Wahllander IV7 is a separate study used for development selection, not final test; source supplement reports ten points but repository provides seven.",
            "Both studies were present in a historical model container; exporting separate files does not remove prior model-development knowledge.",
            "Blanchard repository labels incorrectly say 0.5 mg/kg; locked supplement/primary-source reconciliation establishes 5 mg/kg.",
            "Blanchard weight/height and Wahllander height are unreported and generated from sourced demographics by PK-Sim.",
            "Drug/enzyme expression starts, branch fractions and search bounds are inherited development assumptions; old fitted optima and simulation structure are not copied.",
            "Albumin, solubility reference pH, oral water volume and default enzyme expression/turnover include explicit historical assumptions.",
            "Joint IV/PO data may improve separation of absorption and clearance; identifiability is not established by this input preparation.",
            "Observed study means are not individual-subject records; do not interpret points as independent subjects.",
            "Final-test observations remain unread. No scientific, mass-balance or external-validation acceptance is established here.",
            "Source hashes verify lineage, not biological truth or permission to redistribute restricted derived data.",
        ],
        "compile_status": "not_run", "simulation_status": "not_run", "scientific_status": "not_evaluated",
    }
