"""Build a new drug PBPK snapshot from curated evidence and registered physiology.

No same-drug snapshot, fitted model, or observation curve is accepted here.
Source hashes establish integrity, not the truth of a curated extraction.
The initial implementation supports single-dose adult plasma predictions with
explicit organ plasma-clearance inputs; it does not infer CYP/DDI mechanisms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .endpoint_prediction import predict_requested_endpoints
from .human_model import load_standard_human_model
from .human_model_snapshot import build_human_model_snapshot
from .source_workflow import inside, read_curve, save, sha256

VERSION = "evidence-blueprint-1.1-endpoints"
# Endpoint reporting changes only; the drug construction and parameter rules
# remain the fixed 1.0 baseline for the multi-drug development evaluation.
ENDPOINTS = {"plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
             "terminal_half_life", "auc_inf", "clearance", "apparent_clearance",
             "vz", "apparent_vz", "mrt", "vss"}
UNITS = {"molecular_weight": "g/mol", "molecular_formula": "text", "logp": "Log Units",
         "fraction_unbound": "fraction", "pka_types": "1", "solubility": "mg/ml",
         "solubility_ph": "1", "binding_partner": "text", "elimination_scope": "text",
         "total_clearance": "ml/min/kg", "hepatic_clearance": "ml/min/kg",
         "renal_clearance": "ml/min/kg", "intestinal_permeability": "cm/s"}
COMMON = set(UNITS) - {"total_clearance", "hepatic_clearance", "renal_clearance", "intestinal_permeability"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True, strict=True)


class Source(Strict):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    location: str = Field(min_length=1)


class Evidence(Strict):
    parameter: str
    compound_id: str = Field(min_length=1)
    value: Any
    unit: str
    basis: Literal["measured", "derived", "predicted", "assumption"]
    source: Source
    context: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    prediction_method: dict[str, Any] | None = None


class Human(Strict):
    age_years: float = Field(ge=18, le=80)
    weight_kg: float = Field(gt=0, le=300)
    height_cm: float = Field(gt=0, le=250)
    sex: Literal["MALE", "FEMALE"]
    population: str
    seed: int = Field(default=17, ge=0, le=2147483647, strict=True)


class Dosing(Strict):
    route: Literal["intravenous", "oral"]
    dose_mg: float = Field(gt=0)
    start_h: float = Field(default=0, ge=0)
    infusion_minutes: float = Field(default=0, ge=0)
    formulation: Literal["solution"] | None = None
    water_ml_per_kg: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def route_inputs(self):
        if self.route == "oral":
            if self.formulation != "solution" or self.water_ml_per_kg is None or self.infusion_minutes:
                raise ValueError("Oral construction requires explicit solution, water and no IV infusion")
        elif self.formulation is not None or self.water_ml_per_kg is not None:
            raise ValueError("IV construction cannot carry oral formulation settings")
        return self


class TerminalFit(Strict):
    start_h: float = Field(ge=0)
    end_h: float = Field(gt=0)
    min_points: int = Field(default=4, ge=3)
    min_adjusted_r2: float = Field(default=0.98, gt=0, le=1)
    min_span_half_lives: float = Field(default=1., gt=0)
    max_auc_extrapolated_fraction: float = Field(default=0.2, gt=0, lt=1)
    max_aumc_extrapolated_fraction: float = Field(default=0.2, gt=0, lt=1)


class Target(Strict):
    endpoints: list[str] = Field(min_length=1, max_length=13)
    primary_endpoint: str
    window_start_h: float = Field(ge=0)
    window_end_h: float = Field(gt=0, le=168)
    terminal_fit: TerminalFit | None = None

    @model_validator(mode="after")
    def endpoints_and_window(self):
        if (len(set(self.endpoints)) != len(self.endpoints) or not set(self.endpoints) <= ENDPOINTS
                or self.primary_endpoint not in self.endpoints or self.window_end_h <= self.window_start_h):
            raise ValueError("Select distinct supported endpoints, one selected primary endpoint and an increasing window")
        if self.terminal_fit and not (
                self.window_start_h <= self.terminal_fit.start_h < self.terminal_fit.end_h == self.window_end_h):
            raise ValueError("Terminal fit must lie inside the requested window and end at its final time")
        return self


class EvidenceBlueprintRequest(Strict):
    entry_reason: Literal["model_missing", "unsupported_use"]
    compound_id: str = Field(min_length=1)
    compound_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9 _-]{0,79}$")
    human: Human
    dosing: Dosing
    target: Target
    partition_method: Literal["PKSimStandard", "RodgersRowland"] = "PKSimStandard"
    evidence_ledger: Source
    evidence: list[Evidence] = Field(max_length=32)
    assumptions: list[str] = Field(default_factory=list, max_length=32)
    forbidden_sources: list[Source] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def consistent_request(self):
        keys = [e.parameter for e in self.evidence]
        if len(keys) != len(set(keys)) or not set(keys) <= UNITS.keys():
            raise ValueError("Each supported evidence parameter must appear at most once")
        for e in self.evidence:
            if e.compound_id != self.compound_id or e.unit != UNITS[e.parameter]:
                raise ValueError("Evidence drug identity or canonical unit mismatch: " + e.parameter)
            json.dumps(e.value, allow_nan=False)
            if e.basis == "predicted" and (not e.prediction_method or not all(
                    key in e.prediction_method for key in ("name", "version", "training_overlap", "applicability"))):
                raise ValueError("Predicted inputs need method/version/training-overlap/applicability records")
        if self.dosing.start_h >= self.target.window_end_h:
            raise ValueError("Prediction window must extend beyond the dose")
        if self.dosing.start_h + self.dosing.infusion_minutes / 60 > self.target.window_end_h:
            raise ValueError("The complete infusion must fit within the simulated interval")
        return self


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _number(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(name + " requires a finite positive value" + (" or zero" if zero else ""))
    return float(value)


def _reject_native_document(value):
    pending, count = [value], 0
    while pending:
        node = pending.pop()
        count += 1
        if count > 100000:
            raise ValueError("Evidence JSON is too deeply structured for bounded source inspection")
        if isinstance(node, dict):
            if ({"Simulations", "Compounds", "Individuals"} & node.keys()
                    or isinstance(node.get("snapshot"), dict)):
                raise ValueError("A complete/native model is not an independent drug evidence source")
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def _verify_sources(request, root):
    forbidden_hashes = {r.sha256 for r in request.forbidden_sources}
    forbidden_paths = {inside(r.path, root) for r in request.forbidden_sources}
    ledger_ref = request.evidence_ledger
    ledger_path = inside(ledger_ref.path, root, exists=True)
    if (ledger_path in forbidden_paths or ledger_ref.sha256 in forbidden_hashes
            or ledger_path.suffix.lower() != ".json" or not ledger_path.is_file()
            or ledger_path.stat().st_size > 2 * 1024**2 or sha256(ledger_path) != ledger_ref.sha256):
        raise ValueError("Curated evidence ledger is forbidden, invalid, or changed")
    ledger = json.loads(ledger_path.read_text())
    _reject_native_document(ledger)
    rows = ledger.get("evidence") if isinstance(ledger, dict) else ledger
    if not isinstance(rows, list):
        raise ValueError("Curated evidence ledger must provide an evidence array")
    bound = [Evidence.model_validate(row).model_dump(mode="json") for row in rows]
    supplied = [e.model_dump(mode="json") for e in request.evidence]
    if sorted(bound, key=lambda row: row["parameter"]) != sorted(supplied, key=lambda row: row["parameter"]):
        raise ValueError("Requested values differ from the checksum-bound curated evidence ledger")
    verified = {ledger_path: ledger_ref.sha256}
    for evidence in request.evidence:
        ref = evidence.source
        path = inside(ref.path, root, exists=True)
        if path in forbidden_paths or ref.sha256 in forbidden_hashes:
            raise ValueError("Evidence refers to a forbidden/test source: " + evidence.parameter)
        if path in verified:
            if verified[path] != ref.sha256:
                raise ValueError("Conflicting source checksums")
            continue
        if not path.is_file() or path.suffix.lower() not in {".json", ".pdf", ".xml", ".csv", ".txt", ".md", ".html"}:
            raise ValueError("Evidence must reference a project-local readable document, not a native model")
        if path.stat().st_size > 64 * 1024**2 or sha256(path) != ref.sha256:
            raise ValueError("Evidence source size or checksum mismatch")
        if path.suffix.lower() == ".json":
            _reject_native_document(json.loads(path.read_text()))
        if path.suffix.lower() == ".xml":
            from defusedxml import ElementTree
            document = ElementTree.parse(path)
            if any(element.tag.rsplit("}", 1)[-1] in {"Simulation", "SimulationTransfer", "PKSimProject"}
                   for element in document.iter()):
                raise ValueError("A native XML model is not an independent drug evidence source")
        verified[path] = ref.sha256
    return [{"path": str(p.relative_to(root)), "sha256": digest} for p, digest in verified.items()]


def _parameter(name, value, unit=None):
    return {"Name": name, "Value": value, **({"Unit": unit} if unit is not None else {})}


def _provenance(identity, value, kind, location):
    return {"provenance_id": str(uuid5(NAMESPACE_URL, identity)), "provenance_kind": kind,
            "source_id": identity, "source_location": location, "source_checksum": _digest(value)}


def assemble_evidence_blueprint(request: dict | EvidenceBlueprintRequest, project_root: Path) -> dict:
    """Construct new native input, or a draft listing absent required evidence."""
    req = EvidenceBlueprintRequest.model_validate(request.model_dump(mode="json")
        if isinstance(request, EvidenceBlueprintRequest) else request)
    root = project_root.resolve(strict=True)
    sources = _verify_sources(req, root)
    evidence = {e.parameter: e for e in req.evidence}
    required = COMMON | ({"intestinal_permeability"} if req.dosing.route == "oral" else set())
    missing = sorted(required - evidence.keys())
    manifest = {"algorithm_version": VERSION, "request_sha256": _digest(req.model_dump(mode="json")),
                "entry_reason": req.entry_reason, "used_complete_drug_model": False,
                "source_integrity_verified": True, "source_value_extraction": "curated_not_automatically_verified",
                "values_match_curated_evidence_ledger": True,
                "sources": sources, "primary_endpoint": req.target.primary_endpoint,
                "scientific_status": "not_independently_validated", "assumptions": req.assumptions,
                "parameter_origins": {k: e.model_dump(mode="json") for k, e in evidence.items()},
                "limitations": ["adult_single_drug_single_dose_plasma_only",
                    "organ_clearance_input_not_identified_enzyme_or_transporter_mechanism",
                    "no_DDI_disease_population_or_metabolite_claims",
                    "fixed_declared_partition_method_not_selected_using_test_endpoints",
                    "no_new_parameter_fit_or_endpoint_specific_optimizer"]}
    if missing:
        return {"status": "draft_with_gaps", "gaps": missing, "manifest": manifest}
    values = {k: e.value for k, e in evidence.items()}
    scope = values["elimination_scope"]
    if scope not in {"hepatic_only", "renal_only", "hepatic_and_renal"}:
        raise ValueError("Elimination requires an explicitly sourced organ allocation")
    present = set(values) & {"hepatic_clearance", "renal_clearance", "total_clearance"}
    choices = {"hepatic_only": ({"total_clearance"}, {"hepatic_clearance"}),
               "renal_only": ({"total_clearance"}, {"renal_clearance"}),
               "hepatic_and_renal": ({"total_clearance", "renal_clearance"}, {"hepatic_clearance", "renal_clearance"})}
    if present not in choices[scope]:
        if not present or any(present < option for option in choices[scope]):
            return {"status": "draft_with_gaps", "gaps": ["clearance_inputs_for_" + scope], "manifest": manifest}
        raise ValueError("Ambiguous or overlapping total/organ clearances")
    for key in present:
        e = evidence[key]
        if e.context.get("species") != "human" or e.context.get("matrix") != "plasma" or e.context.get("route") not in {"iv", "intravenous"}:
            raise ValueError("Organ plasma clearance requires human IV plasma evidence, not oral CL/F")
        _number(e.value, key, zero=True)
    fu = _number(values["fraction_unbound"], "fraction_unbound")
    if fu > 1 or evidence["fraction_unbound"].context.get("species") != "human" or evidence["fraction_unbound"].context.get("matrix") != "plasma":
        raise ValueError("Fraction unbound must be human plasma fraction in (0,1]")
    for key in ("molecular_weight", "solubility"):
        _number(values[key], key)
    logp = values["logp"]
    if isinstance(logp, bool) or not isinstance(logp, (float, int)) or not math.isfinite(logp):
        raise ValueError("logp must be a finite numeric value")
    formula = values["molecular_formula"]
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula) if isinstance(formula, str) else []
    if not tokens or "".join(a + n for a, n in tokens) != formula or any(int(n or 1) <= 0 for _, n in tokens):
        raise ValueError("A single unsolvated molecular formula is required for halogen counts")
    halogens = {a: sum(int(n or 1) for atom, n in tokens if atom == a) for a in ("F", "Cl", "Br", "I")}
    renal = values.get("renal_clearance", values.get("total_clearance", 0) if scope == "renal_only" else 0)
    hepatic = values.get("hepatic_clearance", values.get("total_clearance", 0) - renal if scope != "renal_only" else 0)
    if hepatic < 0 or renal < 0 or hepatic + renal <= 0:
        raise ValueError("Renal clearance cannot exceed total clearance; elimination must be positive")
    manifest["clearance_allocation"] = {"scope": scope, "hepatic_ml_min_kg": hepatic, "renal_ml_min_kg": renal,
        "total_not_added_as_separate_process": True, "renal_includes_filtration": renal > 0,
        "allocation_basis": evidence["elimination_scope"].basis}
    if scope == "hepatic_only" and "total_clearance" in values:
        manifest["limitations"].append("total_clearance_localized_to_liver_is_explicit_lumped_approximation")
    definition = load_standard_human_model(root)
    manifest["human_definition_sha256"] = _digest(definition)
    p = _parameter
    processes = []
    for internal, clearance in (("LiverClearance", hepatic), ("KidneyClearance", renal)):
        if clearance == 0:
            continue
        pars = [p("Fraction unbound (experiment)", fu), p("Plasma clearance", clearance, "ml/min/kg")]
        if internal == "LiverClearance":
            pars.append(p("Lipophilicity (experiment)", logp, "Log Units"))
        processes.append({"InternalName": internal, "Species": "Human", "DataSource": "EvidenceAssembly", "Parameters": pars})
    method = "PK-Sim Standard" if req.partition_method == "PKSimStandard" else "Rodgers and Rowland"
    compound = {"Name": req.compound_name, "IsSmallMolecule": True,
        "PlasmaProteinBindingPartner": values["binding_partner"],
        "Parameters": [p("Molecular weight", values["molecular_weight"], "g/mol"), *[p(k, v) for k, v in halogens.items()]],
        "Lipophilicity": [{"Name": "EvidenceLogP", "Parameters": [p("Lipophilicity", logp, "Log Units")]}],
        "FractionUnbound": [{"Name": "EvidenceFu", "Species": "Human", "Parameters": [p("Fraction unbound (plasma, reference value)", fu)]}],
        "Solubility": [{"Name": "EvidenceSolubility", "Parameters": [p("Solubility at reference pH", values["solubility"], "mg/ml"), p("Reference pH", values["solubility_ph"])]}],
        "PkaTypes": values["pka_types"], "Processes": processes,
        "CalculationMethods": ["Cellular partition coefficient method - " + method, "Cellular permeability - PK-Sim Standard"]}
    human = req.human
    individual = {"Name": "EvidenceHuman", "Seed": human.seed, "OriginData": {"Species": "Human", "Population": human.population,
        "Gender": human.sex, "Age": {"Value": human.age_years, "Unit": "year(s)"},
        "Weight": {"Value": human.weight_kg, "Unit": "kg"}, "Height": {"Value": human.height_cm, "Unit": "cm"}}}
    dose = req.dosing
    application = "Oral" if dose.route == "oral" else "Intravenous" if dose.infusion_minutes else "IntravenousBolus"
    protocol = {"Name": "EvidenceDose", "ApplicationType": application, "DosingInterval": "Single",
                "Parameters": [p("InputDose", dose.dose_mg, "mg"), p("Start time", dose.start_h, "h")]}
    formulation = None
    if application == "Intravenous":
        protocol["Parameters"].append(p("Infusion time", dose.infusion_minutes, "min"))
    if application == "Oral":
        _number(values["intestinal_permeability"], "intestinal_permeability")
        if evidence["intestinal_permeability"].context.get("definition") != "pksim_specific_intestinal_permeability":
            raise ValueError("Caco2/MDCK permeability cannot be directly substituted for native intestinal permeability")
        compound["IntestinalPermeability"] = [{"Name": "EvidencePint", "Parameters": [p("Specific intestinal permeability (transcellular)", values["intestinal_permeability"], "cm/s")]}]
        protocol["Parameters"].append(p("Volume of water/body weight", dose.water_ml_per_kg, "ml/kg"))
        formulation = {"Name": "EvidenceSolution", "FormulationType": "Formulation_Dissolved", "Parameters": []}
    target_path = "Organism|PeripheralVenousBlood|" + req.compound_name + "|Plasma (Peripheral Venous Blood)"
    ref = _provenance("evidence-input:" + manifest["request_sha256"], req.model_dump(mode="json"), "user_request", "Curated evidence ledger and declared request; source hashes separately verified")
    physiology = _provenance("human-definition:" + manifest["human_definition_sha256"], definition, "model_asset", "Registered PK-Sim physiology; no drug values")
    # Explicitly include boundaries/events so requested windows are never extrapolated.
    times = sorted({0., dose.start_h, dose.start_h + dose.infusion_minutes / 60,
                    req.target.window_start_h, req.target.window_end_h})
    schema = [{"Parameters": [p("Start time", a, "h"), p("End time", b, "h"), p("Resolution", 60., "pts/h")]}
              for a, b in zip(times, times[1:]) if b > a]
    built = build_human_model_snapshot(individual=individual, compound=compound, protocol=protocol,
        simulation_name="EvidencePrediction", snapshot_version=80, human_source=physiology,
        compound_sources={k: ref for k in compound if k not in {"Name", "IsSmallMolecule", "Description"}},
        protocol_source=ref, formulation=formulation, formulation_source=ref if formulation else None,
        output_schema=schema, output_selections=[target_path], expected_compound_name=req.compound_name)
    manifest.update(native_builder=built["manifest"], target_output=target_path,
                    snapshot_sha256=_digest(built["snapshot"]))
    return {"status": "assembled", "snapshot": built["snapshot"], "manifest": manifest, "gaps": []}


def run_evidence_blueprint(*, request: dict, output_dir: Path, project_root: Path, execute=False) -> dict:
    from .design_runtime import execute_snapshot
    from .evidence_native_audit import audit_evidence_execution
    from .source_adaptation_audit import audit_execution, derive_administration_expectations
    req = EvidenceBlueprintRequest.model_validate(request)
    root = project_root.resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Use a new output directory; previous results are never overwritten")
    blueprint = assemble_evidence_blueprint(req, root)
    output.mkdir(parents=True)
    save(output / "request.json", req.model_dump(mode="json"))
    save(output / "blueprint.json", blueprint)
    result = {"status": blueprint["status"], "scientific_status": "not_independently_validated",
              "used_complete_drug_model": False, "native_executed": False, "gaps": blueprint["gaps"],
              "mass_balance_checked": False}
    if blueprint["status"] == "assembled":
        save(output / "snapshot.json", blueprint["snapshot"])
    if execute and blueprint["status"] == "assembled":
        execution = execute_snapshot(snapshot=blueprint["snapshot"], output_dir=output / "native", project_root=root,
                                     timeout_seconds=240, cpu_threads=2, max_output_bytes=128 * 1024**2)
        result.update(native_executed=True, status="native_failed", execution_manifest=execution["manifest_path"])
        if execution.get("engineering_accepted"):
            expectations = derive_administration_expectations(blueprint["snapshot"])
            audit = audit_execution(snapshot=blueprint["snapshot"], execution_manifest=execution, output_dir=output / "readback",
                                    project_root=root, **expectations)
            save(output / "readback_result.json", audit)
            result.update(status="readback_failed", native_readback_passed=bool(audit.get("accepted")))
            if audit.get("accepted"):
                drug_audit = audit_evidence_execution(blueprint["snapshot"], execution["manifest_path"], output / "native_parameter_audit", root)
                result.update(native_parameter_audit_passed=bool(drug_audit.get("accepted")),
                              status="native_parameter_audit_failed")
                if not drug_audit.get("accepted"):
                    save(output / "result.json", result)
                    return result
                manifest = json.loads(Path(execution["manifest_path"]).read_text())
                curve = read_curve(Path(execution["manifest_path"]).parent, manifest, blueprint["manifest"]["target_output"])
                # All user endpoints are derived from this one persisted trajectory.
                save(output / "prediction_locked.json", curve)
                mw = next(e.value for e in req.evidence if e.parameter == "molecular_weight")
                context = {"route": req.dosing.route, "dose_mg": req.dosing.dose_mg,
                           "start_h": req.dosing.start_h, "infusion_minutes": req.dosing.infusion_minutes,
                           "single_dose": True}
                endpoints = predict_requested_endpoints(curve, req.target.model_dump(exclude_none=True), mw, context=context)
                save(output / "endpoints.json", endpoints)
                result.update(status="predicted", prediction_sha256=sha256(output / "prediction_locked.json"),
                              endpoint_file=str((output / "endpoints.json").relative_to(root)), primary_endpoint=req.target.primary_endpoint,
                              endpoint_statuses={name: item.get("status", "completed")
                                                 for name, item in endpoints["endpoints"].items()})
    save(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Construct a new PBPK drug model from evidence, without a same-drug snapshot")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    path = inside(args.request, root, exists=True)
    if path.stat().st_size > 2 * 1024**2:
        parser.error("Request exceeds 2 MiB")
    result = run_evidence_blueprint(request=json.loads(path.read_text()), output_dir=args.output, project_root=root, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"assembled", "predicted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
