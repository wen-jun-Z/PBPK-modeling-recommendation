#!/usr/bin/env python3
"""Prove two explicit serializer additions for one pinned pregabalin snapshot.

Does not edit source/converted snapshots or the generic migration comparator.
The reviewed differences are empty expression-profile declarations and an explicit
GFR type already uniquely determined by the original glomerular filtration process.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from recommendation_candidate_migration import compare_migration

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/pbpk-model-assets/public_collection_20260909/Pregabalin-Model__47b22b0be8cd/source/Pregabalin.json"
CONVERTED = ROOT / "experiments/model_recommendation/execution_validation_20261001/migrations/0a4fbffe1e6adae54c58344ced780be8eff0a2547a099d80cf3e3ebd2e23588f/snapshot/model.json"
SOURCE_SHA = "0a4fbffe1e6adae54c58344ced780be8eff0a2547a099d80cf3e3ebd2e23588f"
CONVERTED_SHA = "de3c7aaf46adbd5be6575d5c8a9149bd6995cc18849e0b13ae8035a149ebbf62"


def by_name(rows):
    out = {r["Name"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("ambiguous duplicate record names")
    return out


def review(source, converted):
    left = deepcopy(source)
    right = deepcopy(converted)
    counts = {"empty_expression_profile_added": 0, "explicit_gfr_process_type_added": 0}
    individuals = by_name(left["Individuals"])
    for person in right["Individuals"]:
        before = individuals[person["Name"]]
        if "ExpressionProfiles" not in before and "ExpressionProfiles" in person:
            if person["ExpressionProfiles"] != []:
                raise ValueError("new expression profile is not empty")
            del person["ExpressionProfiles"]
            counts["empty_expression_profile_added"] += 1
    originals = by_name(left["Simulations"])
    definitions = by_name(left["Compounds"])
    for simulation in right["Simulations"]:
        old_compounds = by_name(originals[simulation["Name"]]["Compounds"])
        for compound in simulation["Compounds"]:
            previous = by_name(old_compounds[compound["Name"]]["Processes"])
            definition = definitions[compound["Name"]]
            for process in compound["Processes"]:
                old_process = previous[process["Name"]]
                if "SystemicProcessType" not in old_process and "SystemicProcessType" in process:
                    if process["SystemicProcessType"] != "GFR":
                        raise ValueError("new systemic process is not GFR")
                    matches = [p for p in definition["Processes"]
                               if p.get("InternalName") == "GlomerularFiltration"
                               and process["Name"] == "Glomerular Filtration-" + p["DataSource"]]
                    if len(matches) != 1:
                        raise ValueError("original process does not uniquely establish GFR")
                    del process["SystemicProcessType"]
                    counts["explicit_gfr_process_type_added"] += 1
    comparison = compare_migration(left, right)
    if not comparison["accepted"]:
        raise ValueError("unreviewed parameter or structural difference remains")
    return counts, comparison


def ref(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    source_ref, converted_ref = ref(SOURCE), ref(CONVERTED)
    if source_ref["sha256"] != SOURCE_SHA or converted_ref["sha256"] != CONVERTED_SHA:
        raise ValueError("this review only applies to the two exact pinned artifacts")
    source, converted = json.loads(SOURCE.read_text()), json.loads(CONVERTED.read_text())
    counts, comparison = review(source, converted)
    if counts != {"empty_expression_profile_added": 18, "explicit_gfr_process_type_added": 44}:
        raise ValueError("reviewed source-specific change counts differ")
    payload = {"schema": "reviewer-source-specific-migration-proof-v1", "source": source_ref, "converted": converted_ref,
               "candidate_id": "cb5e5b4a-0504-5714-9e93-f7101084ddc2", "accepted_at_declared_representation_scope": True,
               "counts": counts, "comparison_after_reviewed_metadata_normalization": comparison,
               "original_internal_process": source["Compounds"][0]["Processes"],
               "semantics": ["No expression profiles in original JSON and an explicit empty list describe the same absence.",
                             "The original uniquely selected GlomerularFiltration process with matching data source already declares the GFR mechanism; native serialization makes its type explicit."],
               "raw_snapshot_values_changed": False, "global_comparison_rules_changed": False,
               "numerical_observation_or_error_used": False, "predictive_or_scientific_accuracy_established": False,
               "checker": ref(Path(__file__)), "generic_comparator": ref(ROOT / "scripts/recommendation_candidate_migration.py")}
    out = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/external_validation/pregabalin_snapshot_representation_review.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"review": ref(out), "counts": counts, "numeric_checks": comparison["numeric_checks"]}))


if __name__ == "__main__":
    main()
