"""Source-specific migration resolver for the reviewer external experiment.

This resolver accepts only the exact reviewed pregabalin source/converted pair;
all other candidates must continue through the original migration resolver.
"""
from pathlib import Path
import json

import recommendation_candidate_migration as migration
from reviewer_revalidate_pregabalin_snapshot import SOURCE_SHA, CONVERTED_SHA, SOURCE, CONVERTED, review

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = "experiments/model_recommendation/reviewer_revision_20261009/external_validation/pregabalin_snapshot_representation_review.json"
SIDECAR_SHA = "5facf7b5e09419c4e113daaccc7e3b05255a45d21a5829e7b966461919fa3d17"
SOURCE_REL = str(SOURCE.relative_to(ROOT))
CONVERTED_REL = str(CONVERTED.relative_to(ROOT))


def resolve_reviewed(provenance, *, project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    if provenance.get("source_sha256") != SOURCE_SHA or provenance.get("source_path") != SOURCE_REL:
        raise ValueError("reviewed resolver is restricted to one exact pregabalin source")
    original = migration.checked({"path": SOURCE_REL, "sha256": SOURCE_SHA}, root)
    converted = migration.checked({"path": CONVERTED_REL, "sha256": CONVERTED_SHA}, root)
    sidecar = migration.checked({"path": SIDECAR, "sha256": SIDECAR_SHA}, root)
    receipt = json.loads(sidecar.read_text())
    if (receipt.get("schema") != "reviewer-source-specific-migration-proof-v1"
            or receipt.get("accepted_at_declared_representation_scope") is not True
            or receipt.get("raw_snapshot_values_changed") is not False
            or receipt.get("numerical_observation_or_error_used") is not False
            or receipt.get("global_comparison_rules_changed") is not False):
        raise ValueError("reviewed proof scope differs")
    migration.checked(receipt["checker"], root)
    migration.checked(receipt["generic_comparator"], root)
    if receipt["source"] != {"path": SOURCE_REL, "sha256": SOURCE_SHA} or receipt["converted"] != {"path": CONVERTED_REL, "sha256": CONVERTED_SHA}:
        raise ValueError("reviewed proof artifact binding differs")
    prior = migration.archived_attempt(provenance, project_root=root)
    process = prior["process"]
    if process.get("status") != "completed" or process.get("returncode") != 0 or process.get("runtime_unchanged") is not True:
        raise ValueError("original native export execution not authenticated")
    if migration.one_json(prior["directory"] / "snapshot").resolve() != converted.resolve():
        raise ValueError("converted artifact is not the original native output")
    counts, comparison = review(json.loads(original.read_text()), json.loads(converted.read_text()))
    if counts != {"empty_expression_profile_added": 18, "explicit_gfr_process_type_added": 44} or comparison != receipt["comparison_after_reviewed_metadata_normalization"]:
        raise ValueError("live exact-source review differs from bound proof")
    return {"path": converted, "sha256": CONVERTED_SHA, "receipt_path": SIDECAR, "receipt_sha256": SIDECAR_SHA,
            "original_path": SOURCE_REL, "original_sha256": SOURCE_SHA,
            "comparison_scope": "Exact pinned source:18 empty expression declarations and44 explicit GFR types already determined by original process; remaining structures and592 numeric fields unchanged.",
            "operation": "reviewed_exact_source_metadata_completion", "native_invocations": 0,
            "prior_receipt": prior["receipt_reference"], "reviewed_sidecar": {"path": SIDECAR, "sha256": SIDECAR_SHA},
            "provenance_refs": prior["bindings"], "resolver": migration.reference(Path(__file__), root)}
