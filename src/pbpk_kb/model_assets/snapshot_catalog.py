"""Read PK-Sim JSON snapshots into reference-only catalogue descriptors.

Names, administration and scope come from explicit snapshot fields. This reader
does not execute models, infer numerical validation or create evidence records.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import UTC, datetime

from pbpk_kb.model_assets.license_policy import resolve_repository_license
from pbpk_kb.model_assets.public_collection import PROJECT_ROOT, MIB, verify_manifest, write_json

# Source-specific bindings reviewed against the repository's model description.
# Keep stereochemistry explicit; these bindings do not assign external chemical IDs.
REVIEWED_TARGET_NAMES = {
    "Aripiprazole-Model": {"APZ": "Aripiprazole"},
    "Cyclosporine-Model": {"CsA": "Cyclosporine"},
    "Clomiphene-Model": {"(E)-clomiphene": "(E)-clomiphene"},
    "Coproporphyrin-I": {"Coproporphyrin I": "Coproporphyrin I"},
    "Nicotine-Cotinine-Model": {"Nicotine": "Nicotine"},
    "Metoprolol-Model": {"R-Metoprolol": "R-Metoprolol", "S-Metoprolol": "S-Metoprolol"},
    "Oxycodone-Model": {"OCD": "Oxycodone"},
    "Posaconazole-Model": {"Posaconazol": "Posaconazole", "Posaconazol mit supersaturation": "Posaconazole", "Posaconazol ohne supersaturation": "Posaconazole"},
}

REVIEWED_MODEL_TARGETS = {
    # A model of a racemic drug may represent its two enantiomers separately.
    # These are model coverage relationships, never chemical identity aliases.
    "Warfarin-Model": ("Warfarin", {"R-Warfarin", "S-Warfarin"}),
    "Verapamil-Model": ("Verapamil", {"R-Verapamil", "S-Verapamil"}),
    "Verapamil-Dabigatran-DDI": ("Verapamil", {"R-Verapamil", "S-Verapamil"}),
    "Omeprazole-Model": ("Omeprazole", {"Esomeprazole", "R-omeprazole"}),
    "Metoprolol-Model": ("Metoprolol", {"R-Metoprolol", "S-Metoprolol"}),
}


def name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def resolve_target_names(raw_names, canonical_names, repository_name, *, small_molecule_flags=None, source_format=".json"):
    bindings = []
    for raw_name in raw_names:
        matches = _unique(n for n in canonical_names if name_key(n) == name_key(raw_name)
                          or raw_name.casefold().startswith(n.casefold() + "_")
                          or raw_name.casefold().startswith(n.casefold() + " "))
        reviewed = REVIEWED_TARGET_NAMES.get(repository_name, {}).get(raw_name)
        if reviewed:
            matches = [reviewed]
        if len(matches) == 1:
            bindings.append({"snapshot_name": raw_name, "canonical_name": matches[0],
                             "basis": ("reviewed_repository_model_description" if reviewed else "official_drug_repository_target_and_" + ("snapshot_administration" if source_format == ".json" else "compound_building_block")),
                             "is_small_molecule": (small_molecule_flags or {}).get(raw_name)})
    return bindings


def _unique(values):
    return sorted({v for v in values if isinstance(v, str) and v.strip()}, key=str.casefold)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _named(data, key, *, required=False):
    rows = data.get(key, [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not r.get("Name") for r in rows):
        raise ValueError(f"Invalid named {key} collection")
    names = [r["Name"] for r in rows]
    if len(set(names)) != len(names) or (required and not names):
        raise ValueError(f"Empty or duplicate {key} names")
    return {r["Name"]: r for r in rows}


def describe_snapshot(data: dict) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("Version"), int):
        raise ValueError("Not a PK-Sim project snapshot")
    compounds = _named(data, "Compounds", required=True)
    simulations = _named(data, "Simulations", required=True)
    individuals = _named(data, "Individuals")
    populations = _named(data, "Populations")
    protocols = _named(data, "Protocols")
    formulations = _named(data, "Formulations")
    used_compounds, administered, used_individuals, used_protocols, used_formulations = set(), set(), set(), set(), set()
    metabolite_names = set()
    for simulation in simulations.values():
        if not simulation.get("Model"):
            raise ValueError("Simulation has no model structure")
        subject = simulation.get("Individual")
        if isinstance(subject, str):
            if subject not in individuals:
                raise ValueError("Unresolved simulation individual")
            used_individuals.add(subject)
        population = simulation.get("Population")
        if isinstance(population, str):
            if population not in populations:
                raise ValueError("Unresolved simulation population")
            individual = populations[population].get("Individual")
            if isinstance(individual, str) and individual in individuals:
                used_individuals.add(individual)
        simulated = simulation.get("Compounds")
        if not isinstance(simulated, list) or not simulated:
            raise ValueError("Simulation has no compounds")
        for compound in simulated:
            if compound.get("Name") not in compounds:
                raise ValueError("Unresolved simulation compound")
            used_compounds.add(compound["Name"])
            for process in compound.get("Processes", []):
                if process.get("MetaboliteName"):
                    metabolite_names.add(process["MetaboliteName"])
            protocol = compound.get("Protocol")
            if protocol:
                protocol_name = protocol.get("Name") if isinstance(protocol, dict) else protocol
                if protocol_name not in protocols:
                    raise ValueError("Unresolved dosing protocol")
                administered.add(compound["Name"])
                used_protocols.add(protocol_name)
                for formulation in protocol.get("Formulations", []) if isinstance(protocol, dict) else []:
                    if formulation.get("Name") not in formulations:
                        raise ValueError("Unresolved formulation")
                    used_formulations.add(formulation["Name"])
    if not administered:
        raise ValueError("No explicitly administered compound; requires adapter review")
    origins = [individuals[n].get("OriginData", {}) for n in used_individuals]
    expression = data.get("ExpressionProfiles", [])
    outputs = _unique(p for s in simulations.values() for p in s.get("OutputSelections", []))
    routes = _unique(d.get("ApplicationType") for name in used_protocols for d in _walk(protocols[name]))
    semantic = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {
        "parser_version": "pksim-snapshot-reference-1.0", "snapshot_schema_version": data["Version"],
        "software_version": "unspecified (snapshot schema " + str(data["Version"]) + ")",
        "semantic_sha256": semantic,
        "compounds": _unique(used_compounds), "administered_compounds": _unique(administered),
        "small_molecule_flags": {n: compounds[n].get("IsSmallMolecule") for n in administered},
        "metabolites": _unique(metabolite_names),
        "species": _unique(d.get("Species") for d in origins),
        "populations": _unique(d.get("Population") for d in origins),
        "routes": routes, "formulations": _unique(used_formulations),
        "enzymes": _unique(p.get("Molecule") for p in expression if p.get("Type") == "Enzyme"),
        "transporters": _unique(p.get("Molecule") for p in expression if p.get("Type") == "Transporter"),
        "compartments": _unique(p.split("|")[1] for p in outputs if p.startswith("Organism|") and len(p.split("|")) > 2),
        "process_types": [], "output_names": outputs,
        "simulation_count": len(simulations), "simulation_names": list(simulations),
        "dataset_reference_count": len(data.get("ObservedData", [])),
        "source_pointers": {"compounds": "#/Compounds", "scope": "#/Individuals/*/OriginData",
                            "administration": "#/Simulations/*/Compounds/*/Protocol", "simulations": "#/Simulations"},
    }


def describe_project_file(path: Path, commit: str) -> dict:
    """Reuse the established SQLite/XML parser in a bounded child process."""
    command = r'''
import json, resource, sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
from pbpk_kb.model_assets.osp_project_parser import parse_osp_project
p = parse_osp_project(Path(sys.argv[1]), repository_commit=sys.argv[2])
print(json.dumps({
    "parser_version": p.parser_version, "snapshot_schema_version": None,
    "software_version": "unspecified (project schema " + str(p.project_version) + ")",
    "semantic_sha256": p.project_sha256,
    "compounds": p.compounds, "administered_compounds": [], "identity_candidate_names": p.compounds,
    "small_molecule_flags": {}, "metabolites": [], "species": [], "populations": [],
    "routes": sorted({x.application_type for x in p.protocols if x.application_type}),
    "formulations": p.formulations, "enzymes": [], "transporters": [],
    "compartments": sorted({x for s in p.simulations for x in s.organ_names}),
    "process_types": [], "output_names": sorted({x for s in p.simulations for x in s.observer_names}),
    "simulation_count": len(p.simulations), "simulation_names": [s.name for s in p.simulations],
    "dataset_reference_count": len(p.observed_datasets),
    "source_pointers": {"compounds": "SQLITE:BUILDING_BLOCKS[Icon=Compound]", "simulations": "SQLITE:SIMULATIONS"},
    "parser_warnings": p.warnings,
}))
'''
    result = subprocess.run([sys.executable, "-c", command, str(path), commit], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["resource limit or no parser output"]
        raise ValueError("PK-Sim parser rejected file: " + detail[0][-400:])
    descriptor = json.loads(result.stdout)
    if not descriptor["compounds"] or not descriptor["simulation_count"]:
        raise ValueError("PK-Sim project has no compounds or simulations")
    return descriptor


def prepare_snapshot_catalog(output: Path) -> dict:
    rows = json.loads((output / "collected_models.json").read_text())
    recovered = PROJECT_ROOT / ("artifacts/pbpk-model-assets/public_collection_" + output.name.rsplit("_", 1)[-1])
    by_repository = {row["repository"]: row for row in rows}
    for manifest in recovered.glob("*__selected_sources/asset_manifest.json"):
        row = json.loads(manifest.read_text())
        by_repository[row["repository"]] = row
    rows = sorted(by_repository.values(), key=lambda r: r["repository"])
    cached_path = output / "catalog_candidates.json"
    cached = {c["project_sha256"]: c for c in json.loads(cached_path.read_text())} if cached_path.exists() else {}
    candidates, exclusions = [], []
    semantic_seen = {}
    inventory = json.loads((output / "repository_inventory.json").read_text())
    canonical_names = []
    for repo in inventory:
        name = repo["name"]
        if name.lower().endswith("-model") and "-DDI" not in name and "-DDGI" not in name:
            canonical_names.append(name[:-6])
        elif name.startswith("Example_"):
            canonical_names.append(name.removeprefix("Example_"))
        elif name in {"Coproporphyrin-I", "Levonorgestrel", "Valproic-Acid", "IFN-alpha"}:
            canonical_names.append(name)
    for repository_index, repository in enumerate(rows):
        write_json(output / "catalog_preparation.json", {"status": "preparing", "repositories_total": len(rows),
                   "repositories_processed": repository_index, "current_repository": repository["repository"],
                   "updated_at": datetime.now(UTC).isoformat(), "candidate_versions": len(candidates)})
        if repository["collection_status"] == "failed":
            exclusions.append({"repository": repository["repository"], "reason": "download_failed"})
            continue
        verify_manifest(repository)
        source = PROJECT_ROOT / repository["source_directory"]
        license = resolve_repository_license(source).model_dump(mode="json")
        repo_name = repository["repository"].split("/")[-1]
        family = re.sub(r"-model$", "", repo_name, flags=re.I).removeprefix("Example_")
        if repo_name == "Clomiphene-Model":
            family = "(E)-clomiphene"
        kind = "interaction" if "-DDI" in repo_name or "-DDGI" in repo_name else (
            "teaching_example" if repo_name.startswith("Example_") else "drug_or_specialized_model")
        found = 0
        items = sorted(repository["files"], key=lambda f: (Path(f["path"]).suffix.lower() != ".json", f["path"]))
        parsed_json = False
        for item in items:
            path = source / item["path"]
            project_family = family
            target_names = canonical_names
            project_kind = kind
            if repo_name == "Pregnancy-Models" and Path(item["path"]).parts[:1] == ("Models",):
                drug = Path(item["path"]).parts[1]
                project_family = drug + (" Nonpregnant" if "nonpregnant" in path.name.casefold() else " Pregnancy")
                project_kind = "special_population"
                target_names = [*canonical_names, drug] if drug not in canonical_names else canonical_names
            suffix = path.suffix.casefold()
            if suffix not in {".json", ".pksim5"}:
                continue
            if suffix == ".json":
                if item["bytes"] > 80 * MIB:
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                except (json.JSONDecodeError, UnicodeError):
                    exclusions.append({"repository": repository["repository"], "path": item["path"], "reason": "invalid_json"})
                    continue
                if not isinstance(data, dict) or "Compounds" not in data or "Simulations" not in data:
                    continue
            elif parsed_json:
                # Binary/JSON exports of the same repository may be equivalent.
                # Do not inflate the pool by registering both without comparison.
                exclusions.append({"repository": repository["repository"], "path": item["path"], "reason": "json_representation_preferred_pending_cross_format_comparison"})
                continue
            try:
                descriptor = (describe_snapshot(data) if suffix == ".json" else
                              dict(cached[item["sha256"]]) if item["sha256"] in cached else
                              describe_project_file(path, repository["commit"]))
            except (ValueError, TypeError, KeyError, AttributeError, subprocess.TimeoutExpired) as exc:
                exclusions.append({"repository": repository["repository"], "path": item["path"], "reason": str(exc)})
                continue
            found += 1
            checksum = descriptor["semantic_sha256"]
            if checksum in semantic_seen:
                exclusions.append({"repository": repository["repository"], "path": item["path"],
                                   "reason": "duplicate_snapshot_content", "same_as": semantic_seen[checksum]})
                continue
            semantic_seen[checksum] = str(path.relative_to(PROJECT_ROOT))
            assets = [{"asset_kind": "pksim_snapshot" if suffix == ".json" else "pksim_project", "relative_path": str(path.relative_to(PROJECT_ROOT)),
                       "sha256": item["sha256"], "source_location": repository["source_url"] + "/" + item["path"]},
                      {"asset_kind": "source_snapshot", "relative_path": str((source.parent / "source.tar.gz").relative_to(PROJECT_ROOT)),
                       "sha256": repository["source_archive_sha256"], "source_location": repository["source_url"]}]
            reports = []
            for file in repository["files"]:
                suffix = Path(file["path"]).suffix.lower()
                if suffix in {".md", ".pdf"}:
                    rel = str((source / file["path"]).relative_to(PROJECT_ROOT))
                    reports.append(rel)
                    assets.append({"asset_kind": "repository_document", "relative_path": rel, "sha256": file["sha256"],
                                   "source_location": repository["source_url"] + "/" + file["path"]})
            descriptor.update(repository=repository["repository"], repository_url="https://github.com/" + repository["repository"],
                              repository_commit=repository["commit"], model_family=project_family, candidate_kind=project_kind,
                              project_relative_path=str(path.relative_to(PROJECT_ROOT)), project_sha256=item["sha256"],
                              source_snapshot_sha256=repository["source_archive_sha256"],
                              source_collection_scope=repository.get("collection_scope", "complete_repository_archive"),
                              license=license, assets=assets, report_paths=reports,
                              static_parse_status="registered_not_executable", pkml_parse_status="registered_not_executable",
                              execution_probe_status="not_run", reusable_candidate=False,
                              blocking_reasons=["SNAPSHOT_NOT_EXECUTED", "PKML_NOT_EXPORTED", "INDEPENDENT_VALIDATION_NOT_RUN"])
            # Parameter-set suffixes are not new chemical identities. Preserve
            # every raw name, and index only unambiguous official drug targets.
            bindings = resolve_target_names(
                descriptor.get("identity_candidate_names", descriptor["administered_compounds"]), target_names, repo_name,
                small_molecule_flags=descriptor["small_molecule_flags"], source_format=suffix)
            reviewed_model_target = REVIEWED_MODEL_TARGETS.get(repo_name)
            if reviewed_model_target:
                target, components = reviewed_model_target
                if components.issubset(set(descriptor["compounds"])):
                    descriptor["model_coverage_target"] = target
                    bindings.append({"snapshot_name": " + ".join(sorted(components)), "canonical_name": target,
                                     "basis": "reviewed_readme_model_coverage; not_chemical_identity_alias",
                                     "component_names": sorted(components), "is_small_molecule": True})
            descriptor["identity_bindings"] = bindings
            descriptor["unresolved_identity_names"] = [n for n in descriptor["administered_compounds"]
                                                       if n not in {b["snapshot_name"] for b in bindings}]
            descriptor["compounds"] = _unique([*descriptor["compounds"], *(b["canonical_name"] for b in bindings)])
            if not bindings:
                exclusions.append({"repository": repository["repository"], "path": item["path"],
                                   "reason": "no_resolved_drug_target; generic or ambiguous compound names",
                                   "compound_names": descriptor["compounds"]})
                continue
            parsed_json = parsed_json or suffix == ".json"
            if license["status"] != "allowed":
                descriptor["blocking_reasons"].append("LICENSE_NOT_CLEARED")
            candidates.append(descriptor)
        if not found:
            model_files = [f["path"] for f in repository["files"] if Path(f["path"]).suffix.lower() in {".pksim5", ".mobi", ".pkml"}]
            exclusions.append({"repository": repository["repository"], "reason": "no_complete_supported_snapshot",
                               "other_model_assets": model_files})
    write_json(output / "catalog_candidates.json", candidates)
    write_json(output / "catalog_exclusions.json", exclusions)
    summary = {"status": "prepared", "repository_snapshots": len(rows), "candidate_versions": len(candidates),
               "candidate_families": len({name_key(c["model_family"]) for c in candidates}),
               "administered_compound_names": _unique(n for c in candidates for n in c["administered_compounds"]),
               "execution_verified_new_versions": 0, "curation_status": "machine_extracted",
               "exclusion_count": len(exclusions)}
    write_json(output / "catalog_preparation.json", summary)
    return summary
