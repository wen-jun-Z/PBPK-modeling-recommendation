"""Offline, bounded assembly of evidence-grounded human PBPK design drafts.

This entry point reads the locked local PK-Sim definition but never loads an
LLM, compiles a model, or runs a solver. Static preference is NOT a scientific
model ranking. Numerical
identifiability, calibration and independent prediction remain explicit gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pbpk_kb.blueprints.design_contracts import (
    BlueprintDesignRequest,
    DesignEvidence,
    MechanismModule,
    ModuleParameterSpec,
)

DESIGN_VERSION = "human-pbpk-joint-design-0.2.0"


def _checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def _canonical_collections(value: Any) -> Any:
    # Normalize metadata sets only. The caller preserves native instance arrays
    # because PK-Sim input ordering must not be silently changed.
    if isinstance(value, dict):
        return {key: _canonical_collections(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [_canonical_collections(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return value


def _gap(code: str, *, module_id: str | None = None,
         parameter_id: str | None = None, detail: str = "") -> dict[str, Any]:
    return {"code": code, "module_id": module_id,
            "parameter_id": parameter_id, "detail": detail}


def _context_matches(evidence: DesignEvidence, request: BlueprintDesignRequest,
                     *, route_independent: bool = False) -> bool:
    requirement = request.requirement
    return (
        evidence.compound_semantic_id == requirement.compound_semantic_id
        and evidence.species == requirement.species
        and evidence.population in (None, requirement.population)
        and (evidence.route in (None, requirement.route) or route_independent)
    )


def _resolve_parameter(spec: ModuleParameterSpec,
                       request: BlueprintDesignRequest) -> dict[str, Any]:
    facts = [e for e in request.evidence if e.kind == "parameter"
             and e.parameter_semantic_id == spec.semantic_id
             and _context_matches(e, request, route_independent=spec.route_independent)
             and (request.requirement.formulation is None or spec.formulation_independent
                  or e.formulation == request.requirement.formulation)]
    exact = [e for e in facts if e.unit == spec.unit]
    errors: list[str] = []
    if facts and not exact:
        errors.append("PARAMETER_UNIT_MISMATCH")
    values = [e.value for e in exact if e.value is not None]
    value = values[0] if values else None
    if value is not None and any(
        not math.isclose(value, x, rel_tol=1e-9, abs_tol=0.0) for x in values[1:]
    ):
        errors.append("CONFLICTING_PARAMETER_VALUES")
        value = None
    bounded = [e for e in exact if e.lower_bound is not None and e.upper_bound is not None]
    lower = max((e.lower_bound for e in bounded), default=None)
    upper = min((e.upper_bound for e in bounded), default=None)
    if lower is not None and upper is not None and lower > upper:
        errors.append("CONFLICTING_PARAMETER_BOUNDS")
    if value is not None and lower is not None and not lower <= value <= upper:
        errors.append("PARAMETER_VALUE_OUTSIDE_EVIDENCE_BOUNDS")
    calibration = sorted({study.study_id for study in request.studies
                          if study.role == "calibration"
                          and set(study.observed_target_ids) & set(spec.observation_target_ids)})
    can_estimate = bool(spec.estimate_allowed and calibration and bounded
                        and lower < upper and not errors)
    if errors:
        value = None
    origin = "fixed_from_evidence" if value is not None else (
        "to_estimate" if can_estimate else "missing"
    )
    if origin == "missing" and spec.required and not errors:
        errors.append("PARAMETER_NOT_RESOLVED")
    return {
        "semantic_id": spec.semantic_id, "dimension": spec.dimension,
        "unit": spec.unit, "required": spec.required, "mode": origin,
        "value": value, "lower_bound": lower, "upper_bound": upper,
        "initial_value": (value if value is not None else
                          lower / 2 + upper / 2 if can_estimate else None),
        "estimation_allowed_by_inputs": can_estimate,
        "calibration_study_ids": calibration,
        "observation_target_ids": sorted(spec.observation_target_ids),
        "evidence_ids": sorted(e.evidence_id for e in exact),
        "provenance": [e.provenance.model_dump(mode="json")
                       for e in sorted(exact, key=lambda e: e.evidence_id)],
        "errors": errors,
    }


def _dependency_closure(ids: frozenset[str], modules: dict[str, MechanismModule]) -> frozenset[str]:
    result = set(ids)
    pending = list(ids)
    while pending:
        for dependency in modules[pending.pop()].requires_modules:
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return frozenset(result)


def _compatible(ids: frozenset[str], modules: dict[str, MechanismModule]) -> bool:
    groups: set[str] = set()
    for module_id in sorted(ids):
        module = modules[module_id]
        if set(module.excludes_modules) & ids:
            return False
        if module.clearance_group:
            if module.clearance_group in groups:
                return False
            groups.add(module.clearance_group)
    return True


def _parameter_sets(ids: frozenset[str], modules: dict[str, MechanismModule],
                    request: BlueprintDesignRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions: dict[str, ModuleParameterSpec] = {}
    gaps: list[dict[str, Any]] = []
    for module_id in sorted(ids):
        for spec in modules[module_id].parameter_specs:
            previous = definitions.get(spec.semantic_id)
            if previous is not None and previous.model_dump() != spec.model_dump():
                gaps.append(_gap("PARAMETER_CONTRACT_CONFLICT", module_id=module_id,
                                 parameter_id=spec.semantic_id,
                                 detail="Shared parameter definitions must have identical semantics and units."))
            else:
                definitions[spec.semantic_id] = spec
    return [_resolve_parameter(definitions[key], request) for key in sorted(definitions)], gaps


def _variants(parameters: list[dict[str, Any]], max_free: int) -> list[list[dict[str, Any]]]:
    mandatory = {p["semantic_id"] for p in parameters if p["mode"] == "to_estimate"}
    optional = [p["semantic_id"] for p in parameters
                if p["estimation_allowed_by_inputs"] and p["mode"] == "fixed_from_evidence"]
    # Keep evidence-fixed, individually released, and joint subsets; numerical
    # identifiability will later determine which releases data really support.
    subsets = [mandatory]
    subsets.extend(mandatory | {key} for key in optional if len(mandatory) < max_free)
    if optional and len(mandatory) + len(optional) <= max_free:
        subsets.append(mandatory | set(optional))
    seen: set[tuple[str, ...]] = set()
    variants = []
    for subset in subsets:
        key = tuple(sorted(subset))
        if key in seen:
            continue
        seen.add(key)
        variants.append([dict(p, mode="to_estimate", value=None)
                         if p["semantic_id"] in subset else dict(p) for p in parameters])
    return variants


def _human_model_context(request: BlueprintDesignRequest, project_root: Path) -> dict[str, Any]:
    """Resolve the registered numerical model definition, never execute it."""
    context: dict[str, Any] = {
        "status": "unbound", "model_reference": None, "inherited_roles": [],
        "inherited_targets": [], "compartment_paths": [], "instance": None, "gaps": [],
    }
    if request.human_model is None:
        context["gaps"].append(_gap("BASE_MODEL_UNBOUND"))
        return context
    from pbpk_kb.blueprints.human_model import (
        load_standard_human_model, validate_human_model_resources,
    )
    try:
        definition = load_standard_human_model(project_root)
        audit = validate_human_model_resources(definition, project_root)
    except (OSError, ValueError) as exc:
        context["gaps"].append(_gap("BASE_MODEL_RESOURCE_UNAVAILABLE", detail=str(exc)))
        return context
    if not audit["accepted"]:
        context["gaps"].append(_gap("BASE_MODEL_RESOURCE_CHECK_FAILED",
                                   detail=json.dumps(audit["errors"], ensure_ascii=False)))
        return context
    if definition["checksum"] != request.human_model.definition_checksum:
        context["gaps"].append(_gap("BASE_MODEL_DEFINITION_CHECKSUM_MISMATCH"))
        return context
    blocked_hashes = {s.source_checksum for s in request.studies
                      if s.role in {"selection", "test"} or s.study_id in request.forbidden_study_ids
                      or s.source_id in request.forbidden_source_ids}
    if any(resource["sha256"] in blocked_hashes or resource["path"] in request.forbidden_source_ids
           for resource in definition["resources"]):
        context["gaps"].append(_gap("BASE_MODEL_SOURCE_ISOLATION_FAILED"))
        return context
    context.update(
        status="source_bound_standard_model", model_reference={
            **request.human_model.model_dump(mode="json"), "model": definition["model"],
            "resource_validation": "passed", "core_topology_policy": "preserve_registered_standard_model",
            "core_parameter_policy": "to_be_generated_from_individual_by_locked_pksim",
        },
        inherited_roles=["distribution"],
        inherited_targets=[t.target_id for t in request.requirement.targets
                           if t.analyte == "parent" and t.matrix == "plasma"
                           and t.quantity == "concentration" and t.binding == "total"],
    )
    rows = definition["reference_structure"]["containers"]["rows"]
    by_id = {row["container_id"]: row for row in rows}
    for row in rows:
        # Neighborhoods are connections, and Events is a logical container;
        # neither is an anatomical compartment that a drug module may target.
        if row["container_type"] not in {"ORGANISM", "ORGAN", "COMPARTMENT"}:
            continue
        names, current, seen = [], row, set()
        while current is not None:
            if current["container_id"] in seen:
                raise ValueError("Registered source contains a container hierarchy cycle.")
            seen.add(current["container_id"])
            if current["container_name"] == "ROOT":
                break
            names.append(current["container_name"])
            parent = by_id.get(current["parent_container_id"])
            if parent is None and current["parent_container_name"] != "ROOT":
                context["gaps"].append(_gap("BASE_MODEL_COMPARTMENT_PATH_UNRESOLVED"))
                return context
            current = parent
        if names:
            context["compartment_paths"].append("|".join(reversed(names)))
    context["compartment_paths"] = sorted(set(context["compartment_paths"]))
    # These are standard-model capabilities, not predictions, observed data or
    # proof that its generated instance has passed numerical checks.
    if request.instance is None:
        context["gaps"].append(_gap("HUMAN_DRUG_INSTANCE_NOT_PROVIDED"))
        return context
    from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot
    values = request.instance.model_dump(mode="json")
    population_context = values.pop("population_context")
    try:
        package = build_human_model_snapshot(**values, expected_compound_name=request.requirement.name)
    except ValueError as exc:
        context["gaps"].append(_gap("HUMAN_DRUG_INSTANCE_INVALID", detail=str(exc)))
        return context
    age = request.instance.individual["OriginData"]["Age"]["Value"]
    if population_context != "healthy_adult" or age < 18:
        context["gaps"].append(_gap("INSTANCE_POPULATION_UNSUPPORTED",
                                   detail="Current design adapter requires an explicitly declared healthy adult, age >= 18 years."))
        return context
    package["manifest"]["population_context"] = population_context
    package["manifest"]["population_source_verified"] = False
    regimen = package["manifest"]["normalized_administration"]
    route_matches = regimen["route"] == request.requirement.route
    dose_matches = (regimen["unit"] == request.requirement.dose_unit
                    and regimen["dose"] == request.requirement.dose)
    if not route_matches or not dose_matches:
        context["gaps"].append(_gap("INSTANCE_REGIMEN_MISMATCH"))
        return context
    supplied_formulation = request.instance.formulation
    if request.requirement.formulation is not None and (
        supplied_formulation is None or supplied_formulation.get("Name") != request.requirement.formulation
    ):
        context["gaps"].append(_gap("INSTANCE_FORMULATION_MISMATCH"))
        return context
    context["instance"] = package
    missing_execution = package["manifest"]["missing_execution_configuration"]
    if missing_execution:
        context["gaps"].append(_gap("EXECUTION_CONFIGURATION_MISSING",
                                   detail=", ".join(missing_execution)))
    selections = request.instance.output_selections
    plasma_observer = (f"Organism|PeripheralVenousBlood|{request.requirement.name}|"
                       "Plasma (Peripheral Venous Blood)")
    if selections is not None and any(
        target.required and target.target_id in context["inherited_targets"]
        for target in request.requirement.targets
    ) and plasma_observer not in selections:
        context["gaps"].append(_gap("INSTANCE_OUTPUT_MAPPING_MISMATCH", detail=plasma_observer))
    context["inherited_roles"].append("input")
    if package["manifest"].get("elimination_configured"):
        context["inherited_roles"].append("elimination")
    context["status"] = "standard_model_snapshot_prepared"
    return context


def _candidate(ids: frozenset[str], parameters: list[dict[str, Any]],
               extra_gaps: list[dict[str, Any]], request: BlueprintDesignRequest,
               modules: dict[str, MechanismModule], module_gaps: dict[str, list[dict[str, Any]]],
               context_checksum: str, depth: int, baseline: dict[str, Any]) -> dict[str, Any]:
    selected = [modules[key] for key in sorted(ids)]
    provided = {target for m in selected for target in m.provides_targets} | set(baseline["inherited_targets"])
    required = {t.target_id for t in request.requirement.targets if t.required}
    gaps = [*extra_gaps, *baseline["gaps"]]
    for module in selected:
        gaps.extend(module_gaps[module.module_id])
    roles = {m.role for m in selected} | set(baseline["inherited_roles"])
    for role in ("input", "distribution"):
        if role not in roles:
            gaps.append(_gap("REQUIRED_ROLE_MISSING", detail=role))
    if not {"elimination", "metabolism"} & roles:
        gaps.append(_gap("REQUIRED_ROLE_MISSING", detail="elimination_or_metabolism"))
    for target in sorted(required - provided):
        gaps.append(_gap("TARGET_NOT_MAPPED", detail=target))
    free_ids = sorted(p["semantic_id"] for p in parameters if p["mode"] == "to_estimate")
    if len(free_ids) > request.budget.max_free_parameters:
        gaps.append(_gap("FREE_PARAMETER_BUDGET_EXCEEDED"))
    for parameter in parameters:
        if parameter["required"]:
            gaps.extend(_gap(code, parameter_id=parameter["semantic_id"])
                        for code in parameter["errors"])
    if baseline["model_reference"] is None and (request.base is None or not request.base.parameters):
        gaps.append(_gap("PHYSIOLOGY_PARAMETERS_NOT_PROVIDED"))
    if request.requirement.formulation is not None and baseline["instance"] is None and not any(
        m.role == "input" and request.requirement.formulation in m.formulations for m in selected
    ):
        gaps.append(_gap("FORMULATION_CONTEXT_UNVERIFIED", detail=request.requirement.formulation))
    for parameter in request.base.parameters if request.base is not None else []:
        if parameter.blocking and (
            parameter.origin == "missing"
            or (parameter.origin in {"derived", "estimated"} and parameter.value is None)
        ):
            gaps.append(_gap("PHYSIOLOGY_PARAMETER_UNRESOLVED", parameter_id=parameter.parameter_id))
    if request.base is not None and request.requirement.population not in request.base.supported_populations:
        gaps.append(_gap("HUMAN_BASE_POPULATION_UNSUPPORTED",
                         detail=request.requirement.population))
    gaps = sorted({json.dumps(g, sort_keys=True): g for g in gaps}.values(),
                  key=lambda g: (g["code"], g["module_id"] or "", g["parameter_id"] or "", g["detail"]))
    content = {
        "input_checksum": context_checksum, "module_ids": sorted(ids),
        "base_model_reference": baseline["model_reference"],
        "inherited_roles": baseline["inherited_roles"],
        "inherited_targets": baseline["inherited_targets"],
        "change_scope": "drug_and_optional_mechanisms_only",
        "parameters": parameters, "estimated_parameter_ids": free_ids,
        "target_mapping": {t.target_id: sorted(
            [m.module_id for m in selected if t.target_id in m.provides_targets]
            + (["@base:4Comp|Human"] if t.target_id in baseline["inherited_targets"] else [])
        ) for t in request.requirement.targets},
        "gaps": gaps,
    }
    return {
        "candidate_id": _checksum(content), **content,
        "edit_depth": depth, "status": "draft_with_gaps" if gaps else "draft_assembled",
        "module_edges": [{"from": dep, "to": m.module_id, "relation": "requires"}
                         for m in selected for dep in sorted(m.requires_modules)],
        "edits": [{"action": "configure_drug_mechanism_on_base", "module_id": m.module_id,
                   "version": m.version,
                   "evidence_ids": sorted(e.evidence_id for e in request.evidence
                                          if e.kind == "mechanism" and e.mechanism == m.mechanism
                                          and _context_matches(e, request)),
                   "provenance": [p.model_dump(mode="json") for p in m.provenance]}
                  for m in selected],
        "patch_application_status": "not_applied" if selected else "no_additional_patches",
        "identifiability_status": "not_run",
        "compile_status": "not_run", "simulation_status": "not_run",
        "scientific_status": "not_evaluated",
    }


def _static_priority(candidate: dict[str, Any]) -> tuple[Any, ...]:
    gaps = candidate["gaps"]
    unmet = sum(g["code"] in {"REQUIRED_ROLE_MISSING", "TARGET_NOT_MAPPED"} for g in gaps)
    return (unmet, len(gaps), len(candidate["estimated_parameter_ids"]),
            len(candidate["module_ids"]), candidate["candidate_id"])


def design_blueprint(request: BlueprintDesignRequest, *, project_root: Path | None = None) -> dict[str, Any]:
    """Assemble bounded, source-linked candidates without any runtime side effects."""
    # Revalidate nested objects too, guarding callers that mutated Pydantic models.
    raw_payload = request.model_dump(mode="json")
    native_instance = raw_payload.pop("instance")
    request_payload = _canonical_collections(raw_payload)
    request_payload["instance"] = native_instance
    request = BlueprintDesignRequest.model_validate(request_payload)
    context_checksum = _checksum(request_payload)
    baseline = _human_model_context(request, project_root or Path(__file__).resolve().parents[3])
    modules = {m.module_id: m for m in sorted(request.modules, key=lambda m: m.module_id)}
    compartments = (set(baseline["compartment_paths"]) if baseline["model_reference"] is not None
                    else {c.compartment_id for c in request.base.compartments} if request.base else set())
    module_gaps: dict[str, list[dict[str, Any]]] = {}
    unavailable: list[dict[str, Any]] = []
    available: dict[str, MechanismModule] = {}
    for module_id, module in modules.items():
        gaps = []
        if module.module_id.startswith("@base:") or any(
            spec.semantic_id.startswith(("physiology:", "@base:")) for spec in module.parameter_specs
        ):
            unavailable.append({"module_id": module_id, "reason": "PROTECTED_HUMAN_BASE_MUTATION"})
            continue
        # Generic metadata cannot replace a native core or silently add another
        # clearance/input process. A verified native replacement adapter is required.
        if module.role in baseline["inherited_roles"] or (
            module.role == "metabolism" and "elimination" in baseline["inherited_roles"]
        ):
            unavailable.append({"module_id": module_id, "reason": "BASE_PROCESS_REPLACEMENT_ADAPTER_REQUIRED"})
            continue
        if request.requirement.route not in module.routes:
            unavailable.append({"module_id": module_id, "reason": "ROUTE_UNSUPPORTED"})
            continue
        if module.requires_evidence and not any(
            e.kind == "mechanism" and e.mechanism == module.mechanism
            and _context_matches(e, request) for e in request.evidence
        ):
            unavailable.append({"module_id": module_id, "reason": "MECHANISM_EVIDENCE_MISSING"})
            continue
        for compartment in sorted(set(module.required_compartments) - compartments):
            gaps.append(_gap("PHYSIOLOGY_COMPARTMENT_MISSING", module_id=module_id, detail=compartment))
        module_gaps[module_id] = gaps
        available[module_id] = module
    # Remove a supported module if one of its dependencies is unsupported.
    changed = True
    while changed:
        changed = False
        for module_id, module in list(available.items()):
            if not set(module.requires_modules) <= available.keys():
                unavailable.append({"module_id": module_id, "reason": "DEPENDENCY_UNAVAILABLE"})
                del available[module_id]
                changed = True
    # Bounded rule branches keep an early incompatible input from locking out
    # a complete assembly. These are warm starts, not fitted/validated models.
    # A set here contains only modifications. The bound whole-body core is
    # always inherited and is never a removable/searchable module.
    seeds = [frozenset()]
    needed_targets = ({t.target_id for t in request.requirement.targets if t.required}
                      - set(baseline["inherited_targets"]))
    goals = [("role", role) for role in ("input", "distribution", "elimination")
             if role not in baseline["inherited_roles"]]
    goals += [("target", target) for target in sorted(needed_targets)]
    for kind, goal in goals:
        def meets_goal(ids: frozenset[str]) -> bool:
            if kind == "target":
                return any(goal in available[key].provides_targets for key in ids)
            roles = {available[key].role for key in ids}
            return bool(roles & {"elimination", "metabolism"}) if goal == "elimination" else goal in roles
        choices = set()
        for seed in seeds:
            if meets_goal(seed):
                choices.add(seed)
                continue
            for module_id in available:
                proposal = _dependency_closure(seed | {module_id}, available)
                if _compatible(proposal, available) and meets_goal(proposal):
                    choices.add(proposal)
        if choices:
            seeds = sorted(choices, key=lambda ids: (len(ids), tuple(sorted(ids))))[:request.budget.beam_width]
    candidates: dict[str, dict[str, Any]] = {}
    expanded: set[frozenset[str]] = set()
    beam = [*seeds, frozenset()]
    stopped = "search_space_exhausted"
    for depth in range(request.budget.max_rounds + 1):
        next_beam: list[tuple[dict[str, Any], frozenset[str]]] = []
        proposals = set(beam) if depth == 0 else set()
        if depth:
            for ids in beam:
                for module_id, module in available.items():
                    if module_id in ids:
                        continue
                    removed = {key for key in ids if key in module.excludes_modules
                               or module_id in available[key].excludes_modules
                               or (module.clearance_group is not None
                                   and module.clearance_group == available[key].clearance_group)}
                    changed_removal = True
                    while changed_removal:
                        dependents = {key for key in ids if set(available[key].requires_modules) & removed}
                        changed_removal = bool(dependents - removed)
                        removed |= dependents
                    proposals.add(_dependency_closure((ids - removed) | {module_id}, available))
        # Expand the most useful static contracts first, with stable tie breaking.
        def proposal_priority(ids: frozenset[str]) -> tuple[Any, ...]:
            provided = {t for m in ids for t in available[m].provides_targets}
            roles = {available[m].role for m in ids}
            required = {t.target_id for t in request.requirement.targets if t.required}
            return (-len(provided & required), -len(roles & {"input", "distribution", "elimination", "metabolism"}),
                    len(ids), tuple(sorted(ids)))
        for ids in sorted(proposals, key=proposal_priority):
            if ids in expanded or not _compatible(ids, available):
                continue
            expanded.add(ids)
            params, gaps = _parameter_sets(ids, available, request)
            group = []
            for variant in _variants(params, request.budget.max_free_parameters):
                if len(candidates) >= request.budget.max_candidates:
                    stopped = "candidate_budget_exhausted"
                    break
                candidate = _candidate(ids, variant, gaps, request, available, module_gaps,
                                       context_checksum, depth, baseline)
                candidates.setdefault(candidate["candidate_id"], candidate)
                group.append(candidate)
            if group:
                next_beam.append((min(group, key=_static_priority), ids))
            if stopped == "candidate_budget_exhausted":
                break
        if stopped == "candidate_budget_exhausted":
            break
        if not next_beam:
            break
        next_beam.sort(key=lambda item: _static_priority(item[0]))
        beam = [ids for _, ids in next_beam[:request.budget.beam_width]]
        if depth == request.budget.max_rounds:
            stopped = "round_budget_exhausted"
    ordered = sorted(candidates.values(), key=_static_priority)
    preferred = ordered[0] if ordered else None
    result = {
        "schema_version": "1.0.0", "designer_version": DESIGN_VERSION,
        "mode": "offline_design_only", "input_checksum": context_checksum,
        "requirement": request.requirement.model_dump(mode="json"),
        "human_base": request.base.model_dump(mode="json") if request.base else None,
        "physiology_status": "individual_metadata_separate_from_standard_model",
        "human_model_binding": baseline,
        "module_catalog": [m.model_dump(mode="json") for m in modules.values()],
        "candidate_count": len(ordered), "candidates": ordered,
        "preferred_draft_id": preferred["candidate_id"] if preferred else None,
        "selection_basis": "static_contracts_only_not_predictive_performance",
        "unavailable_modules": sorted(unavailable, key=lambda item: item["module_id"]),
        "data_roles": [s.model_dump(mode="json") for s in sorted(request.studies, key=lambda s: s.study_id)],
        "execution": {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0,
                      "compile_status": "not_run", "scientific_status": "not_evaluated"},
        "search": {"termination": stopped, "budget": request.budget.model_dump(mode="json")},
        "next_required_steps": ["generate_instance_with_registered_pksim", "compile_supported_drug_patches", "numerical_mass_balance",
                                "calibration_observation_identifiability", "development_selection",
                                "lock_artifact_then_independent_validation"],
    }
    result["checksum"] = _checksum(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create PBPK design drafts without LLM, GPU or simulation.")
    parser.add_argument("--input", type=Path, required=True, help="Structured design request JSON")
    parser.add_argument("--output", type=Path, required=True, help="New output JSON; existing files are not overwritten")
    parser.add_argument("--bind-standard-human-model", action="store_true",
                        help="Pin the validated project-local PK-Sim human model when input has no reference")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.bind_standard_human_model and payload.get("human_model") is None:
        from pbpk_kb.blueprints.human_model import load_standard_human_model
        definition = load_standard_human_model(Path(__file__).resolve().parents[3])
        payload["human_model"] = {"registry_id": definition["registry_id"],
                                  "definition_checksum": definition["checksum"]}
    request = BlueprintDesignRequest.model_validate(payload)
    result = design_blueprint(request)
    # Exclusive creation keeps old evidence/results safe when paths are reused.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
