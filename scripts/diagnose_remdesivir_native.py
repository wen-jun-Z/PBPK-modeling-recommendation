#!/usr/bin/env python3
"""One bounded development-model solve; no fit, target outcomes, or parameter edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprints.design_runtime import run_bounded_r


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def discover(pkml):
    model = ET.parse(pkml).getroot().find("Simulation/Model/Root")
    rows, paths, ids = [], {}, {}

    def walk(node, prefix=()):
        for child in node.findall("./Children/*"):
            path = prefix + (child.get("name"),)
            if None in path or path in paths:
                raise ValueError("Missing or duplicate native path")
            paths[path] = child
            ids[child.get("id")] = path
            walk(child, path)

    walk(model)
    blood = [
        "Organism|PeripheralVenousBlood|Remdesivir|Plasma (Peripheral Venous Blood)",
        "Organism|VenousBlood|Plasma|Remdesivir|Concentration in container",
        "Organism|ArterialBlood|Plasma|Remdesivir|Concentration in container",
    ]
    for text in blood:
        node = paths[tuple(text.split("|"))]
        if node.tag != "Observer" or node.get("dim") != "Concentration (molar)":
            raise ValueError("Blood observer identity/quantity mismatch")
        rows.append(dict(path=text, kind="blood_plasma", dimension="Concentration (molar)"))
    parent_ids, product_ids = set(), set()
    for path, node in paths.items():
        if node.tag != "MoleculeAmount":
            continue
        if path[-1] == "Remdesivir":
            parent_ids.add(node.get("id"))
            kind = "parent_reservoir" if path[0] == "Events" else "parent_amount"
        elif path[-1] == "Remdesivir-Undefined Liver Metabolite":
            product_ids.add(node.get("id"))
            kind = "hepatic_product_counter"
        else:
            continue
        if node.get("dim") != "Amount":
            raise ValueError("Drug amount dimension differs")
        rows.append(dict(path="|".join(path), kind=kind, dimension="Amount"))
    reactions = []
    for path, node in paths.items():
        if node.tag != "Reaction":
            continue
        educts, products = node.findall("Educts/ReactionPartner"), node.findall("Products/ReactionPartner")
        if len(educts) != 1 or len(products) != 1:
            raise ValueError("Unreviewed reaction stoichiometry")
        educt, product = educts[0], products[0]
        if (educt.get("partner") not in parent_ids or product.get("partner") not in product_ids
                or educt.get("stoichiometricCoefficient") != "1" or product.get("stoichiometricCoefficient") != "1"):
            raise ValueError("Unreviewed metabolite transfer")
        reactions.append({"path": "|".join(path), "educt": "|".join(ids[educt.get("partner")]),
                          "product": "|".join(ids[product.get("partner")]), "stoichiometry": "1:1"})
    if len(product_ids) != 2 or len(reactions) != 2:
        raise ValueError("Expected two hepatic bookkeeping products")
    # These products and the urine accumulator must have no outgoing process.
    urine_id = paths[("Organism", "Kidney", "Urine", "Remdesivir")].get("id")
    for node in paths.values():
        if node.tag == "Transport" and node.get("sourceAmount") in product_ids | {urine_id}:
            raise ValueError("A proposed accumulator has an outgoing transport")
    protocol = ["|".join(p) for p, n in paths.items() if n.tag == "Parameter"
                and p[0] == "Events" and p[-2] == "ProtocolSchemaItem" and p[-1] in {"Dose", "Start time", "Infusion time"}]
    if len(protocol) != 3:
        raise ValueError("Requires one single infusion")
    return rows, reactions, protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mapping = (ROOT / args.mapping).resolve(strict=True)
    base = ROOT / "experiments/blueprint_design/architecture_20260930/clinical_pilot/work"
    if not mapping.is_relative_to(base) or "__full__extent_exchange_clearance" not in str(mapping):
        raise ValueError("Only locked full three-axis development mappings are in scope")
    locked = json.loads(mapping.read_text())
    if locked.get("compound_id") != "pubchem:121304016" or not locked.get("development_curve_used"):
        raise ValueError("Requires the locked Remdesivir development fit")
    pkml = (ROOT / locked["decision"]["native_output_dir"] / "candidate.pkml").resolve(strict=True)
    if not pkml.is_relative_to(mapping.parent / "trials"):
        raise ValueError("Source must be selected development trial, never target model")
    if locked["decision"]["artifact_pins"].get(str(pkml.relative_to(ROOT))) != sha(pkml):
        raise ValueError("Locked selected model hash differs")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "experiments/blueprint_design/architecture_20260930/clinical_refinement_v2_diagnostics"):
        raise ValueError("Output outside dedicated diagnostic namespace")
    output.mkdir(parents=True, exist_ok=False)
    quantities, reactions, protocol = discover(pkml)
    worker = ROOT / "src/pbpk_kb/model_runtime/R/remdesivir_native_diagnostic.R"
    cfg = dict(version="remdesivir-native-diagnostic-1", pkml_path=str(pkml), pkml_sha256=sha(pkml),
               mapping_path=str(mapping), mapping_sha256=sha(mapping), quantities=quantities,
               reactions=reactions, protocol=protocol, output_dir=str(output), native_cpu_cores=2)
    request = output / "request.json"
    write(request, cfg)
    runtime = run_bounded_r(script=worker, args=[str(request), sha(request)], output_dir=output,
                            project_root=ROOT, timeout_seconds=180, cpu_threads=2,
                            max_output_bytes=128 * 1024**2)
    result = {"runtime": runtime, "source_unchanged": sha(pkml) == cfg["pkml_sha256"],
              "mapping_unchanged": sha(mapping) == cfg["mapping_sha256"],
              "parameter_changes": [], "clinical_target_used": False, "model_kind": "selected_development_fit"}
    native_path = output / "native_result.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else {}
    result["actual_native_solver_calls"] = native.get("actual_native_solver_calls", int((output / "solver_receipt.json").exists()))
    result["native_status"] = native.get("status", "interrupted")
    if native.get("status") == "completed":
        frame = pd.read_csv(output / "outputs.csv")
        t = frame.Time.to_numpy(dtype=float)
        mass = [r["path"] for r in quantities if r["dimension"] == "Amount"]
        total = frame[mass].sum(axis=1).to_numpy(dtype=float)
        dose_umol = native["dose_mg"] * 1000 / native["molecular_weight_g_mol"]
        result["bookkeeping"] = {"expected_dose_umol": dose_umol,
            "sum_definition": "all native parent molecule amounts including dose reservoir/excreta + two 1:1 cumulative hepatic bookkeeping products",
            "max_absolute_residual_umol": float(np.max(np.abs(total - dose_umol))),
            "max_relative_residual": float(np.max(np.abs(total - dose_umol)) / dose_umol),
            "does_not_establish_biological_metabolism": True}
        result["snapshots"] = []
        for hour in [.5, 1, 2, 4, 6, 12, 24, 48, 72, 168, 1344]:
            values = {r["path"]: float(np.interp(hour * 60, t, frame[r["path"]])) for r in quantities}
            organs = {}
            for r in quantities:
                if r["kind"] == "parent_amount":
                    organ = r["path"].split("|")[1]
                    organs[organ] = organs.get(organ, 0.) + values[r["path"]]
            result["snapshots"].append({"time_h": hour, "organs_parent_umol": organs,
                "blood_plasma_umol_l": {r["path"]: values[r["path"]] for r in quantities if r["kind"] == "blood_plasma"},
                "hepatic_cumulative_parent_removed_umol": sum(values[r["path"]] for r in quantities if r["kind"] == "hepatic_product_counter"),
                "urine_cumulative_parent_umol": values["Organism|Kidney|Urine|Remdesivir"],
                "parent_amounts_umol": {r["path"]: values[r["path"]] for r in quantities if r["kind"].startswith("parent_")}})
        result["artifacts"] = {str(p.relative_to(ROOT)): sha(p) for p in (request, worker, native_path, output / "outputs.csv")}
    write(output / "diagnosis.json", result)
    print(json.dumps({"status": result["native_status"], "solver_calls": result["actual_native_solver_calls"],
                      "output": str(output), "bookkeeping": result.get("bookkeeping")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
