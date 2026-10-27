#!/usr/bin/env python3
"""Render Figures 5/6 from one source-locked, uniformly defined ten-drug method.

Existing renderers and historical results are not edited. The manifest is
results/data/blueprint_current.json: method_version, source_pins, run_summary,
comparison (five-method display JSON), comparison_input (clinical report JSON),
and ten cases with config plus SHA-bound mapping, development_prediction,
target_prediction, development_source, physchem and target_reference records.
Use --validate-only to verify all numerical sources without creating figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import render_ten_drug_curves as curves
import render_clinical_method_figure as comparison
import render_pbpk_method_comparison as clinical_table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/data/blueprint_current.json"
LEGACY_COMPARISON = ROOT / "results/tables/pbpk_ours_vs_rf/additional_comparison.json"


def equal(a, b, context):
    if not np.allclose(a, b, rtol=1e-10, atol=1e-13, equal_nan=False):
        raise ValueError("Current figure data disagree: " + context)


class CurrentData:
    def __init__(self, manifest):
        self.pins = {}
        self.path = curves.path_of(manifest)
        self.document = curves.load(self.path, self.pins)
        d = self.document
        if not isinstance(d.get("method_version"), str) or not d["method_version"].strip():
            raise ValueError("A single explicit method_version is required")
        for path, checksum in d.get("source_pins", {}).items():
            resolved = curves.path_of(path)
            if curves.digest(resolved) != checksum:
                raise ValueError("Unified source pin changed: " + path)
            self.pins[str(resolved.relative_to(ROOT))] = checksum
        self.summary = self.ref(d["run_summary"])
        self.run = curves.path_of(d["run_summary"]["path"]).parent
        self.report = self.ref(d["comparison_input"])
        self.display = self.ref(d["comparison"])
        self.comparison_path = curves.path_of(d["comparison"]["path"])
        self.cases = d.get("cases", [])
        if len(self.cases) != 10 or len({c["case_id"] for c in self.cases}) != 10:
            raise ValueError("The current method must contain all ten unique drugs")
        if len({c["drug"] for c in self.cases}) != 10:
            raise ValueError("Duplicate drug in unified manifest")
        expected = [c["case_id"] for c in self.cases]
        for label, data in (("comparison", self.display), ("clinical report", self.report)):
            if [c["case_id"] for c in data.get("cases", [])] != expected:
                raise ValueError(label + " differs from the complete current ten-drug cohort/order")
        self.report_cases = {c["case_id"]: c for c in self.report["cases"]}
        qualified, excluded = clinical_table.qualify_document(self.report)
        if excluded or len(qualified) != 10:
            raise ValueError("Current clinical references failed qualification: " + repr(excluded))
        self.qualified = {c["case_id"]: c for c in qualified}
        self.panels, self.observed = self.collect_curves()
        self.validate_comparison()

    def ref(self, reference):
        data = curves.load_ref(reference, self.pins)
        pointer = reference.get("json_pointer")
        if pointer is not None:
            if not isinstance(pointer, str) or pointer and not pointer.startswith("/"):
                raise ValueError("Invalid SHA-bound reference JSON pointer")
            for token in pointer.split("/")[1:]:
                key = token.replace("~1", "/").replace("~0", "~")
                data = data[int(key)] if isinstance(data, list) else data[key]
        return data

    def load(self, path):
        p = curves.path_of(path)
        expected = self.document.get("source_pins", {}).get(str(p.relative_to(ROOT)))
        return curves.load(p, self.pins, expected)

    def native_context(self, reference):
        """Bind each curve to native CSV, snapshot and parameter readback."""
        from pbpk_kb.blueprints.source_workflow import read_curve
        path = curves.path_of(reference["path"])
        native = self.ref(reference)
        folder = path.parent
        receipt = self.load(folder / "native/execution_manifest.json")
        if not receipt.get("engineering_accepted") or not receipt.get("source_unchanged"):
            raise ValueError("Native execution did not pass: " + str(folder))
        snapshot = curves.load(folder / "native/snapshot.json", self.pins, receipt["source_sha256"])
        if not self.load(folder / "parameter_audit_result.json").get("accepted"):
            raise ValueError("Drug input/native capacity audit failed: " + str(folder))
        target = snapshot["Simulations"][0]["OutputSelections"][0]
        csv_curve = read_curve(folder / "native", receipt, target)
        equal(native["time_min"], csv_curve["time_min"], "native CSV times")
        equal(native["concentration"], csv_curve["concentration"], "native CSV concentrations")
        for item in receipt["curve_outputs"]:
            csv_path = curves.path_of(folder / "native" / item["path"])
            if curves.digest(csv_path) != item["sha256"]:
                raise ValueError("Native curve CSV changed")
            self.pins[str(csv_path.relative_to(ROOT))] = item["sha256"]
        return native, snapshot

    def collect_curves(self):
        panels, rows = [], []
        for case in self.cases:
            cid, drug, config = case["case_id"], case["drug"], case["config"]
            target = self.report_cases[cid]
            if config["case_id"] != cid or config["drug"] != drug or target["drug"] != drug:
                raise ValueError("Case/drug identity mismatch")
            target_path = curves.path_of(case["target_prediction"]["path"])
            if not target_path.is_relative_to(self.run):
                raise ValueError("Every target prediction must be newly executed in the common run")
            reference = self.ref(case["target_reference"])
            if "cases" in reference:
                matching = [c for c in reference["cases"] if c.get("case_id") == cid]
                if len(matching) != 1:
                    raise ValueError("Clinical source must contain exactly one matching case: " + cid)
                reference = matching[0]["clinical_reference"]
            for key in ("case_id", "study_id", "curve", "endpoints"):
                if reference.get(key) != target["clinical_reference"].get(key):
                    raise ValueError("Original clinical reference changed: " + cid + "/" + key)
            dev = self.ref(case["development_source"])
            physical = self.ref(case["physchem"])
            mw = float(physical["identity"]["molecular_weight_g_mol"])
            mapping = self.ref(case["mapping"])
            mapping_path = curves.path_of(case["mapping"]["path"])
            fit_input = self.load(mapping_path.parent / "input.json")
            if (not mapping.get("eligible") or mapping.get("warm_start_source") is not None
                    or fit_input.get("warm_start_source") is not None
                    or fit_input.get("maximum_evaluations") != 24):
                raise ValueError("Expected uniform eligible two-parameter fitting, 24 calls, without warm starts")
            if mapping["reference_human"] != dev["human"] or mapping["reference_dosing"] != dev["dosing"]:
                raise ValueError("Development study inputs do not match selected mapping")
            if any(mapping["training_source"].get(k) != case["development_source"].get(k) for k in ("path", "sha256")):
                raise ValueError("Mapping uses a different development study")
            if dev["study_id"] == config["study_id"]:
                raise ValueError("Development and evaluation study identifiers coincide")
            selected = mapping["decision"]
            attempt = mapping_path.parent / f"attempt_{selected['evaluation']:02d}"
            if curves.path_of(case["development_prediction"]["path"]) != attempt / "prediction_locked.json":
                raise ValueError("Development curve is not the chosen fit evaluation")
            trial = self.load(attempt / "result.json")
            if trial.get("status") != "predicted" or trial.get("prediction_sha256") != case["development_prediction"]["sha256"]:
                raise ValueError("Chosen development prediction did not complete unchanged")
            native_dev, dev_snapshot = self.native_context(case["development_prediction"])
            native_target, target_snapshot = self.native_context(case["target_prediction"])
            if dev_snapshot["Compounds"] != target_snapshot["Compounds"]:
                raise ValueError("Selected drug parameters differ between development and target studies")
            if dev_snapshot["Simulations"][0]["Compounds"][0]["CalculationMethods"] != target_snapshot["Simulations"][0]["Compounds"][0]["CalculationMethods"]:
                raise ValueError("Selected partition method changed between studies")
            d = curves.series(native_dev, [v["time_min"]/60 for v in dev["observations"]],
                [v["concentration_umol_per_l"]*mw/1000 for v in dev["observations"]], mw)
            evaluation = mapping["evaluations"][selected["evaluation"]-1]
            equal(d["sampled_prediction"]*1000/mw, evaluation["predicted_at_observed_times"], "selected development predictions")
            equal(d["gmfe"], selected["curve_metrics"]["gmfe"], "selected development GMFE")
            clinical = reference["curve"]
            if clinical["unit"] != "ng/mL":
                raise ValueError("Expected source-bound clinical curve units ng/mL")
            if not (clinical["matrix"] == "plasma" or clinical["matrix"] == "serum" and "serum_as_plasma" in target["approximations"]):
                raise ValueError("Missing explicit serum/plasma approximation")
            e = curves.series(native_target, clinical["time_h"], np.asarray(clinical["concentration"])/1000, mw)
            equal(e["gmfe"], self.qualified[cid]["curve_metrics"]["ours"]["gmfe"], "clinical-protocol report GMFE")
            reported = self.ref(target["predictions"]["ours"])
            q = np.interp(e["time"], reported["curve"]["time_h"], np.asarray(reported["curve"]["concentration"])/1000)
            equal(e["sampled_prediction"], q, "target native versus reported predictions")
            meta = []
            for label, item, human, dosing, study in (
                ("Development", d, dev["human"], dev["dosing"], dev["study_id"]),
                ("Evaluation", e, config["target_request_metadata"]["human"], config["target_request_metadata"]["dosing"], config["study_id"])):
                meta.append({"stage": label, "study_id": study, "human_model_inputs": human, "dosing": dosing,
                    "observed_points": len(item["time"]), "display_window_h": [float(item["time"].min()), float(item["time"].max())],
                    "gmfe": item["gmfe"], "ln_rmse": item["ln_rmse"], "within_twofold_fraction": item["within_twofold_fraction"],
                    "dense_prediction_points_displayed": len(item["plot_time"]),
                    "observed_time_h": item["time"].tolist(), "observed_mg_l": item["observed"].tolist(),
                    "prediction_at_observations_mg_l": item["sampled_prediction"].tolist()})
                rows.extend({"drug": drug, "stage": label, "time_h": t, "observed_mg_l": o, "predicted_mg_l": p}
                            for t,o,p in zip(item["time"], item["observed"], item["sampled_prediction"]))
            detail = {"case_id": cid, "drug": drug, "method_version": self.document["method_version"],
                "formula": mapping["construction_partition_method"], "same_selected_parameters_for_both_studies": True,
                "selected_development_evaluation": selected["evaluation"],
                "selected_parameters": {k: selected[k] for k in ("lipophilicity", "clearance_scale")},
                "mapping": case["mapping"], "development_native": case["development_prediction"],
                "evaluation_native": case["target_prediction"], "target_newly_executed_in_common_run": True,
                "clinical_evaluation_matrix": clinical["matrix"], "clinical_evaluation_statistic": clinical["statistic"],
                "clinical_evaluation_sample_size": clinical.get("sample_size"),
                "development_assumptions": dev.get("demographic_assumptions", []),
                "development_limitations": dev.get("limitations", []), "independence": dev.get("independence"),
                "evaluation_assumptions": config["target_request_metadata"].get("demographic_assumptions", []),
                "clinical_reference_quality_notes": target.get("clinical_reference_quality_notes", []), "series": meta}
            panels.append({"drug": drug, "series": [d,e], "detail": detail})
        return panels, pd.DataFrame(rows)

    def validate_comparison(self):
        methods = self.display.get("methods", [])
        if [m.get("id") for m in methods] != comparison.METHOD_ORDER:
            raise ValueError("Expected the same five explicitly identified comparator methods")
        for method in methods:
            if not all(isinstance(method.get(k), str) and method[k].strip() for k in ("label", "information")):
                raise ValueError("Method labels and information sets must be explicit")
        old = self.load(LEGACY_COMPARISON)
        old_cases = {c["case_id"]: c for c in old["cases"]}
        for case in self.display["cases"]:
            cid, qualified = case["case_id"], self.qualified[case["case_id"]]
            if set(case["endpoints"]) != set(comparison.ENDPOINTS) or set(case["gmfe"]) != set(comparison.METHOD_ORDER):
                raise ValueError("Incomplete current endpoint/method matrix")
            if case["drug"] != qualified["drug"] or case["observed_points"] != qualified["curve_metrics"]["ours"]["n"]:
                raise ValueError("Curve observation identity/count differs")
            equal(case["gmfe"]["ours"], qualified["curve_metrics"]["ours"]["gmfe"], "Figure 6 current Ours GMFE")
            for method in comparison.METHOD_ORDER[1:]:
                equal(case["gmfe"][method], old_cases[cid]["gmfe"][method], "Frozen comparator GMFE")
            for row in qualified["rows"]:
                ep = row["endpoint"]
                cell = case["endpoints"][ep]
                factor = row["display_value_factor"]
                if cell["unit"] != row["display_unit"] or cell["source_unit"] != row["unit"] or cell["display_factor"] != factor:
                    raise ValueError("Display unit conversion changed")
                observed = row["clinical_reference"]
                if observed is None:
                    if cell["observed"] is not None:
                        raise ValueError("Clinical reference was invented")
                else:
                    equal(cell["observed"], observed*factor, "Clinical reference display value")
                if cell["observed"] != old_cases[cid]["endpoints"][ep]["observed"]:
                    raise ValueError("Original clinical observation changed")
                for method in comparison.METHOD_ORDER:
                    pred = cell["predictions"][method]
                    if not isinstance(pred.get("weak_tail"), bool):
                        raise ValueError("Prediction quality flag must be boolean")
                    if pred.get("value") is None:
                        if not isinstance(pred.get("reason"), str) or not pred["reason"].strip():
                            raise ValueError("Unavailable prediction requires its retained reason")
                    else:
                        comparison._finite(pred["value"], cid+"/"+method+"/"+ep, allow_zero=ep == "tmax")
                    if method in ("ours", "rf"):
                        current = row[method+"_prediction"]
                        if current is None:
                            if pred["value"] is not None:
                                raise ValueError("Unavailable protocol endpoint was filled")
                        else:
                            equal(pred["value"], current*factor, "Protocol endpoint display value")
                        if pred["weak_tail"] != row[method+"_weak_tail_estimate"]:
                            raise ValueError("Protocol endpoint quality flag changed")
                    if method != "ours":
                        original = old_cases[cid]["endpoints"][ep]["predictions"][method]
                        if pred["value"] != original["value"] or pred["weak_tail"] != original["weak_tail"]:
                            raise ValueError("Historical comparator prediction/quality changed")

    def verify_unchanged(self):
        for path, expected in self.pins.items():
            if curves.digest(curves.path_of(path)) != expected:
                raise ValueError("Figure input changed during rendering: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--figure", choices=("all", "5", "6"), default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    data = CurrentData(args.manifest)
    if args.validate_only:
        data.verify_unchanged()
        print(json.dumps({"validated_cases": len(data.cases), "observed_curve_points": len(data.observed),
                          "method_version": data.document["method_version"]}))
        return
    if args.figure in ("all", "5"):
        previous = sys.argv
        try:
            curves.collect = lambda: (data.panels, data.observed, data.pins)
            curves.SOURCE = data.run
            sys.argv = [str(Path(curves.__file__)), "--final"]
            curves.main()
        finally:
            sys.argv = previous
    if args.figure in ("all", "6"):
        comparison.load_comparison = lambda: (data.display["methods"], data.display["cases"], data.comparison_path)
        comparison.main()
    data.verify_unchanged()
    for name in (["figure_audit.json"] if args.figure == "5" else ["comparison_audit.json"] if args.figure == "6"
                 else ["figure_audit.json", "comparison_audit.json"]):
        path = curves.OUT/name
        audit = json.loads(path.read_text())
        audit.update(current_manifest={"path": str(data.path.relative_to(ROOT)), "sha256": curves.digest(data.path)},
            method_version=data.document["method_version"], input_sha256=data.pins,
            plotting_wrapper={"path": str(Path(__file__).relative_to(ROOT)), "sha256": curves.digest(Path(__file__))},
            all_ten_targets_newly_executed=True, uniform_fit_policy="Two parameters, two formula candidates, 24 calls each, no warm starts; eligible identical-policy fit cache allowed")
        if name == "figure_audit.json":
            audit["data_checks"] = ["Twenty native curves checked against source-bound CSV/snapshot receipts",
                "Selected development prediction and GMFE match stored optimizer evaluation",
                "Identical drug parameters and partition method in development and newly executed target snapshots",
                "All original observations retained without extrapolation", "Target GMFE agrees with current clinical report and Figure 6"]
        else:
            audit.pop("baseline_csv_sha256", None)
            audit["clinical_report_source"] = data.document["comparison_input"]
            audit["unchanged_comparator_source"] = {"path": str(LEGACY_COMPARISON.relative_to(ROOT)), "sha256": curves.digest(LEGACY_COMPARISON)}
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str)+"\n")


if __name__ == "__main__":
    main()
