"""Bind designs to the actual PK-Sim standard human small-molecule model sources.

Registration reads immutable project-local assets and SQLite in read-only mode.
It does not run PK-Sim, evaluate equations, compile a model, simulate, or import
any drug-specific experiment. The registered object is a reusable *standard model
definition*: model 4Comp, Human, original structural/formula resources and pinned
construction code. Drug and individual values still require explicit instantiation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "pksim-standard-human-model-1.0.0"
STANDARD_HUMAN_MODEL_ID = "pksim-human-standard-small-molecule"
REGISTRY_PATH = "artifacts/pbpk-model-assets/human_base/standard_small_molecule.json"
SOURCE_ROOT = "artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173"
SOURCE_VERSION = "12.3.173"
SOURCE_COMMIT = "b8f13cc6b2317488e9fa3a969701aef4b2132e5c"
MODEL_ID = "4Comp"
SPECIES = "Human"
REFERENCE_POPULATION = "European_ICRP_2002"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
# Previously locked upstream distribution artifacts, not caffeine parameters.
_EXPECTED_DB_SHA = "aec61433ddddf39ec5c832bda90a6977a25ea5427e77edc08f2d97a751fe1469"
_EXPECTED_LICENSE_SHA = "9a1a8fdc9b5c7d1f9c6963c4751fd9f7e9c55b27da25da357953c78886d1a737"
_RESOURCES = {
    "database": "src/Db/PKSimDB.sqlite",
    "license": "Open Systems Pharmacology Suite License.pdf",
    "individual_factory": "src/PKSim.Core/Model/IndividualFactory.cs",
    "individual_model_task": "src/PKSim.Core/Services/IndividualModelTask.cs",
    "model_properties": "src/PKSim.Core/Model/ModelProperties.cs",
    "model_configuration": "src/PKSim.Core/Model/ModelConfiguration.cs",
    "model_properties_task": "src/PKSim.Core/Services/ModelPropertiesTask.cs",
    "spatial_structure_factory": "src/PKSim.Core/Model/PKSimSpatialStructureFactory.cs",
    "species_container_query": "src/PKSim.Infrastructure/ORM/Queries/SpeciesContainerQuery.cs",
    "model_container_query": "src/PKSim.Infrastructure/ORM/Queries/ModelContainerQuery.cs",
    "model_neighborhood_query": "src/PKSim.Infrastructure/ORM/Queries/ModelNeighborhoodQuery.cs",
    "neighborhood_finalizer": "src/PKSim.Infrastructure/Services/NeighborhoodFinalizer.cs",
    "passive_transport_query": "src/PKSim.Infrastructure/ORM/Queries/ModelPassiveTransportQuery.cs",
    "formula_factory": "src/PKSim.Core/Model/FormulaFactory.cs",
    "simulation_configuration": "src/PKSim.Core/Services/SimulationConfigurationTask.cs",
    "simulation_model_creator": "src/PKSim.Core/Services/SimulationModelCreator.cs",
}
_TABLES = (
    "tab_models", "tab_model_species", "tab_species", "tab_populations",
    "tab_model_containers", "tab_population_containers", "tab_containers",
    "tab_neighborhoods", "tab_model_calculation_methods", "tab_species_calculation_methods",
    "tab_calculation_methods", "tab_calculation_method_rates", "tab_process_rates",
    "tab_rate_container_parameters", "tab_rate_generic_parameters",
    "tab_rate_container_molecules", "tab_rate_generic_molecules",
    "tab_container_parameter_values", "tab_container_parameter_curves",
    "tab_container_parameter_fcurves", "tab_container_parameter_rates",
)


class HumanModelResourceError(ValueError):
    """A required standard-model source or its integrity cannot be established."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_path(project_root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise HumanModelResourceError("INVALID_PROJECT_RESOURCE_PATH")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"..", "."} for part in relative_path.split("/")):
        raise HumanModelResourceError("RESOURCE_PATH_OUTSIDE_PROJECT")
    root = project_root.resolve(strict=True)
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise HumanModelResourceError("RESOURCE_PATH_OUTSIDE_PROJECT")
    return path


@lru_cache(maxsize=128)
def _cached_file_sha(resolved: str, device: int, inode: int, size: int,
                     mtime_ns: int, ctime_ns: int) -> str:
    path = Path(resolved)
    expected = (device, inode, size, mtime_ns, ctime_ns)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    current = path.stat()
    if (current.st_dev, current.st_ino, current.st_size,
        current.st_mtime_ns, current.st_ctime_ns) != expected:
        raise HumanModelResourceError("RESOURCE_CHANGED_DURING_HASH")
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return _cached_file_sha(str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
                            stat.st_mtime_ns, stat.st_ctime_ns)


def _read_json(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise HumanModelResourceError("MODEL_DEFINITION_TOO_LARGE")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise HumanModelResourceError("MODEL_DEFINITION_MUST_BE_OBJECT")
    return value


def _selection(connection: sqlite3.Connection, sql: str, parameters: tuple,
               *, include_rows: bool = True) -> dict:
    rows = [dict(row) for row in connection.execute(sql, parameters)]
    result = {"resource": "database", "sql": sql, "parameters": list(parameters),
              "row_count": len(rows), "rows_sha256": _digest(rows)}
    result["rows" if include_rows else "examples"] = rows if include_rows else rows[:6]
    return result


def _container_path(connection: sqlite3.Connection, container_id: int) -> str:
    names, seen = [], set()
    while container_id is not None:
        if container_id in seen:
            raise HumanModelResourceError("SOURCE_CONTAINER_CYCLE")
        seen.add(container_id)
        rows = connection.execute(
            "SELECT container_name,parent_container_id FROM tab_containers WHERE container_id=?",
            (container_id,),
        ).fetchall()
        if len(rows) != 1:
            raise HumanModelResourceError("UNRESOLVED_SOURCE_CONTAINER")
        names.append(rows[0]["container_name"])
        container_id = rows[0]["parent_container_id"]
    names.reverse()
    if names and names[0] == "ROOT":
        names.pop(0)
    return "|".join(names)


def _database_definition(database: Path) -> dict:
    # URI mode=ro and query_only protect the existing reference database.
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        model = _selection(connection, "SELECT * FROM tab_models WHERE model=?", (MODEL_ID,))
        human = _selection(connection, "SELECT * FROM tab_model_species WHERE model=? AND species=?",
                           (MODEL_ID, SPECIES))
        population = _selection(connection,
                                "SELECT * FROM tab_populations WHERE population=? AND species=?",
                                (REFERENCE_POPULATION, SPECIES))
        if model["row_count"] != 1 or human["row_count"] != 1 or population["row_count"] != 1:
            raise HumanModelResourceError("STANDARD_HUMAN_MODEL_SELECTION_NOT_UNIQUE")
        if model["rows"][0]["display_name"] != "Standard model for small molecules":
            raise HumanModelResourceError("STANDARD_MODEL_NAME_MISMATCH")
        containers = _selection(connection, (
            "SELECT c.*,m.usage_in_individual FROM tab_containers c "
            "JOIN tab_model_containers m USING(container_id,container_type,container_name) "
            "JOIN tab_population_containers p USING(container_id,container_type,container_name) "
            "WHERE m.model=? AND p.population=? ORDER BY c.container_id"),
            (MODEL_ID, REFERENCE_POPULATION))
        neighborhoods = _selection(connection, (
            "SELECT n.*,m.usage_in_individual FROM tab_neighborhoods n "
            "JOIN tab_model_containers m ON m.container_id=n.neighborhood_id "
            "AND m.container_type=n.neighborhood_type AND m.container_name=n.neighborhood_name "
            "JOIN tab_population_containers p ON p.container_id=n.neighborhood_id "
            "AND p.container_type=n.neighborhood_type AND p.container_name=n.neighborhood_name "
            "WHERE m.model=? AND p.population=? ORDER BY n.neighborhood_id"),
            (MODEL_ID, REFERENCE_POPULATION))
        if not containers["row_count"] or not neighborhoods["row_count"]:
            raise HumanModelResourceError("EMPTY_STANDARD_MODEL_STRUCTURE")
        # Paths are derived from the original container table, not from drug outputs.
        neighborhood_paths = [{
            "neighborhood_id": row["neighborhood_id"],
            "first_neighbor_path": _container_path(connection, row["first_neighbor_id"]),
            "second_neighbor_path": _container_path(connection, row["second_neighbor_id"]),
        } for row in neighborhoods["rows"]]
        methods = _selection(connection, (
            "SELECT m.calculation_method,m.is_default,m.sequence,c.category,c.display_name "
            "FROM tab_model_calculation_methods m JOIN tab_calculation_methods c USING(calculation_method) "
            "JOIN tab_species_calculation_methods s USING(calculation_method) "
            "WHERE m.model=? AND s.species=? ORDER BY c.category,m.sequence,m.calculation_method"),
            (MODEL_ID, SPECIES))
        formulas = _selection(connection, (
            "SELECT r.* FROM tab_calculation_method_rates r "
            "JOIN tab_model_calculation_methods m USING(calculation_method) "
            "JOIN tab_species_calculation_methods s USING(calculation_method) "
            "WHERE m.model=? AND s.species=? ORDER BY r.calculation_method,r.formula_rate"),
            (MODEL_ID, SPECIES), include_rows=False)
        bindings = {}
        for table in ("tab_process_rates", "tab_rate_container_parameters", "tab_rate_generic_parameters",
                      "tab_rate_container_molecules", "tab_rate_generic_molecules"):
            bindings[table] = _selection(connection, (
                f"SELECT r.* FROM {table} r JOIN tab_model_calculation_methods m USING(calculation_method) "
                "JOIN tab_species_calculation_methods s USING(calculation_method) "
                "WHERE m.model=? AND s.species=? ORDER BY " +
                ",".join("r." + row[1] for row in connection.execute(f"PRAGMA table_info({table})"))),
                (MODEL_ID, SPECIES), include_rows=False)
        defaults: dict[str, list[str]] = {}
        for row in methods["rows"]:
            if row["is_default"]:
                defaults.setdefault(row["category"], []).append(row["calculation_method"])
        return {
            "model_selection": model, "species_selection": human, "population_selection": population,
            "reference_structure": {
                "population": REFERENCE_POPULATION,
                "scope": "Standard-model/reference-population structural rows; not individualized parameter values.",
                "containers": containers, "neighborhoods": neighborhoods,
                "resolved_neighborhood_paths": neighborhood_paths,
                "container_type_counts": dict(Counter(row["container_type"] for row in containers["rows"])),
                "connection_semantics": "Neighborhood adjacency; not every edge is blood flow or an active transport process.",
            },
            "calculation_methods": methods,
            "model_properties_policy": {
                "configuration_id": f"{MODEL_ID}|{SPECIES}",
                "default_candidates_by_category": defaults,
                "categories_with_multiple_default_candidates": {
                    key: values for key, values in defaults.items() if len(values) > 1},
                "selection": "Use ModelPropertiesTask.DefaultFor(originData, '4Comp'); retain explicit origin-data and compound calculation choices. Database default flags alone are not final selections.",
            },
            "formula_sources": {
                "rates": formulas, "bindings": bindings,
                "semantics": "Original generic standard-model expressions remain in the locked database. Examples are verbatim source rows, not newly derived equations or a complete numerical model.",
            },
            "database_schema": {table: [dict(row) for row in connection.execute(f"PRAGMA table_info({table})")]
                                for table in _TABLES},
        }
    finally:
        connection.close()


def make_standard_human_model(project_root: Path) -> dict:
    """Collect the real local standard-model resources without writing or running them."""
    resources = []
    for role, suffix in _RESOURCES.items():
        relative = f"{SOURCE_ROOT}/{suffix}"
        path = _safe_path(project_root, relative)
        if not path.is_file():
            raise HumanModelResourceError(f"MISSING_MODEL_RESOURCE: {role}")
        resources.append({"role": role, "path": relative, "sha256": _file_sha(path),
                          "byte_size": path.stat().st_size})
    by_role = {resource["role"]: resource for resource in resources}
    for role, expected in (("database", _EXPECTED_DB_SHA), ("license", _EXPECTED_LICENSE_SHA)):
        if by_role[role]["sha256"] != expected:
            raise HumanModelResourceError(f"UPSTREAM_LOCK_MISMATCH: {role}")
    database = _safe_path(project_root, by_role["database"]["path"])
    definition = {
        "schema_version": SCHEMA_VERSION, "registry_id": STANDARD_HUMAN_MODEL_ID,
        "human_model_id": STANDARD_HUMAN_MODEL_ID,
        "status": "source_bound_standard_model_definition", "executable": False,
        "model": {
            "model_id": MODEL_ID, "species": SPECIES, "configuration_id": f"{MODEL_ID}|{SPECIES}",
            "display_name": "Standard model for small molecules",
            "source_version": SOURCE_VERSION, "source_commit": SOURCE_COMMIT,
            "scope": "PK-Sim standard whole-body small-molecule framework; not a reduced four-organ model.",
        },
        "resources": resources,
        "construction_entrypoints": [
            {"resource": "individual_factory", "symbol": "IndividualFactory.CreateAndOptimizeFor",
             "purpose": "Instantiate supplied Human OriginData using system physiology rules."},
            {"resource": "model_properties_task", "symbol": "ModelPropertiesTask.DefaultFor(OriginData, string)",
             "purpose": "Select the explicit 4Comp/Human configuration and calculation methods."},
            {"resource": "spatial_structure_factory", "symbol": "PKSimSpatialStructureFactory.CreateFor",
             "purpose": "Reuse standard anatomical containers, neighborhoods and their parameter formulas."},
            {"resource": "simulation_configuration", "symbol": "SimulationConfigurationTask.CreateFor",
             "purpose": "Combine the existing standard structure with the requested compound and administration."},
            {"resource": "simulation_model_creator", "symbol": "SimulationModelCreator.CreateModelFor",
             "purpose": "Delegate actual construction to OSPSuite IModelConstructor when separately authorized."},
        ],
        "instantiation_requirements": [
            "Explicit Human individual with source-backed demographics and PK-Sim population identifier.",
            "Explicit drug properties and compound calculation choices; no caffeine or other drug defaults inherited.",
            "Explicit dose, route, protocol, formulation if needed, and target observations.",
            "Pinned PK-Sim/OSPSuite runtime and successful construction checks before executable status.",
            "Independent scientific validation before claiming prediction quality for the requested use.",
        ],
        "limitations": [
            "Registration verifies sources and database selections only; no compilation, simulation or solver validation was performed.",
            "Reference anatomy uses European_ICRP_2002; this is not a healthy_adult alias or a qualification of all populations.",
            "Source commit/version follow the existing project distribution lock; current source files are individually hash-pinned, not re-attested against a remote Git repository.",
            "The selected source files are construction entrypoints, not a vendored closure of OSPSuite runtime dependencies.",
        ],
    }
    definition.update(_database_definition(database))
    definition["checksum"] = _digest(definition)
    return definition


def validate_human_model_resources(definition: dict, project_root: Path) -> dict:
    """Fail closed on source drift, unsafe paths, or tampered model selections.

    Replays only bounded local metadata queries; no expressions are evaluated.
    The canonical comparison also prevents an edited manifest from inventing
    connectivity or formula claims while retaining valid source checksums.
    """
    errors, verified = [], []
    try:
        if not isinstance(definition, dict):
            raise HumanModelResourceError("MODEL_DEFINITION_MUST_BE_OBJECT")
        supplied_sha = definition.get("checksum")
        if not isinstance(supplied_sha, str) or not _SHA.fullmatch(supplied_sha):
            raise HumanModelResourceError("DEFINITION_CHECKSUM_REQUIRED")
        body = {key: value for key, value in definition.items() if key != "checksum"}
        if _digest(body) != supplied_sha:
            raise HumanModelResourceError("DEFINITION_CHECKSUM_MISMATCH")
        resources = definition.get("resources")
        if not isinstance(resources, list) or len(resources) != len(_RESOURCES):
            raise HumanModelResourceError("MODEL_RESOURCE_SET_MISMATCH")
        seen = set()
        for resource in resources:
            role, relative = resource.get("role"), resource.get("path")
            path = _safe_path(project_root, relative)
            if role not in _RESOURCES or role in seen or relative != f"{SOURCE_ROOT}/{_RESOURCES[role]}":
                raise HumanModelResourceError("MODEL_RESOURCE_SET_MISMATCH")
            seen.add(role)
            if not path.is_file() or _file_sha(path) != resource.get("sha256") or path.stat().st_size != resource.get("byte_size"):
                raise HumanModelResourceError(f"MODEL_RESOURCE_CHECKSUM_MISMATCH: {role}")
            verified.append(role)
        expected = make_standard_human_model(project_root)
        if expected != definition:
            raise HumanModelResourceError("MODEL_DEFINITION_DOES_NOT_MATCH_SOURCES")
    except (HumanModelResourceError, OSError, ValueError, TypeError, AttributeError, sqlite3.Error) as exc:
        errors.append(str(exc))
    return {"accepted": not errors, "errors": errors, "verified_resources": verified,
            "human_model_id": STANDARD_HUMAN_MODEL_ID,
            "checksum": definition.get("checksum") if isinstance(definition, dict) else None,
            "scope": "Static source integrity and original model-definition selections only.",
            "compiled": False, "simulated": False}


def register_standard_human_model(project_root: Path) -> dict:
    """Write the single deterministic final registration; never overwrite a different one."""
    definition = make_standard_human_model(project_root)
    target = _safe_path(project_root, REGISTRY_PATH)
    if target.exists():
        if _read_json(target) != definition:
            raise HumanModelResourceError("EXISTING_HUMAN_MODEL_REGISTRATION_DIFFERS")
        return definition
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves concurrent/user-owned data; no legacy assets change.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(definition, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return definition


def load_standard_human_model(project_root: Path) -> dict:
    """Read and validate the fixed registered standard-model definition."""
    definition = _read_json(_safe_path(project_root, REGISTRY_PATH))
    validation = validate_human_model_resources(definition, project_root)
    if not validation["accepted"]:
        raise HumanModelResourceError("; ".join(validation["errors"]))
    return definition
