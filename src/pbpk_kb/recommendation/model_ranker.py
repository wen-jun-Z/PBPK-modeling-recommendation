from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from pbpk_kb.model_assets.models import OSPModelVersionSnapshot
from pbpk_kb.recommendation.abstention import abstention_reasons
from pbpk_kb.recommendation.calibration import (
    disabled_calibration_manifest,
    wilson_interval,
)
from pbpk_kb.recommendation.model_candidate_retrieval import (
    retrieve_model_candidates,
)
from pbpk_kb.recommendation.schemas import (
    RankedModelCandidate,
    RankedModelRecommendation,
    RequirementProfile,
    ranked_model_recommendation_checksum,
)


RANKER_VERSION = "pbpk-public-model-ranker-1.1.0"
RANKING_SCHEMA_VERSION = "1.1.0"
RANKING_RUN_NAMESPACE = UUID("ed9ae5ae-cd44-46f9-a9d3-c2d0fd98cb88")
MODEL_RUN_MANIFEST_NAMESPACE = UUID(
    "11257e37-c5a4-48ce-822b-c069ad353eb8"
)


class ModelRankingError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ModelRankingError(
                    f"Expected object at {path}:{line_number}."
                )
            yield value


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_gzip_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="\n",
            ) as text:
                for row in rows:
                    text.write(_canonical_json(row))
                    text.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return _sha256_file(path)


def model_run_manifest_uuid(path: Path) -> UUID:
    return uuid5(
        MODEL_RUN_MANIFEST_NAMESPACE,
        _sha256_file(path),
    )


def build_ranking_recommendation(
    *,
    profile: RequirementProfile,
    candidates: list[RankedModelCandidate],
    model_run_manifest_id: UUID,
    auto_reuse_enabled: bool,
    created_at: datetime | None = None,
) -> RankedModelRecommendation:
    if not candidates:
        raise ModelRankingError("Model candidate list is empty.")
    grade_3_candidates = [
        candidate
        for candidate in candidates
        if candidate.candidate_class == "exact_compound_model"
        and candidate.eligibility == "eligible"
        and candidate.run_label_grade == 3
        and candidate.ranking_label == "reuse_positive"
        and candidate.validation_run_id is not None
    ]
    direct_reuse = (
        auto_reuse_enabled
        and len(grade_3_candidates) == 1
        and not profile.critical_gaps
        and not any(
            conflict.severity == "hard"
            for conflict in profile.conflicts
        )
    )
    reasons = (
        []
        if direct_reuse
        else abstention_reasons(
            profile=profile,
            candidates=candidates,
            auto_reuse_enabled=auto_reuse_enabled,
        )
    )
    if not direct_reuse and len(grade_3_candidates) > 1:
        reasons = sorted({*reasons, "AMBIGUOUS_TOP_CANDIDATES"})
    if not direct_reuse and not reasons:
        reasons = ["SCORE_BELOW_THRESHOLD"]
    ranking_run_id = uuid5(
        RANKING_RUN_NAMESPACE,
        _canonical_json(
            {
                "profile_id": str(profile.profile_id),
                "profile_checksum": profile.checksum,
                "model_run_manifest_id": str(model_run_manifest_id),
                "candidate_ids": [
                    str(candidate.model_version_id)
                    for candidate in candidates
                ],
                "ranker_version": RANKER_VERSION,
            }
        ),
    )
    payload: dict[str, Any] = {
        "ranking_run_id": ranking_run_id,
        "ranking_schema_version": RANKING_SCHEMA_VERSION,
        "profile_id": profile.profile_id,
        "decision": "reuse" if direct_reuse else "abstain",
        "candidates": candidates,
        "selected_model_version_id": (
            grade_3_candidates[0].model_version_id
            if direct_reuse
            else None
        ),
        "adaptation_plan_id": None,
        "abstention_reason_codes": reasons,
        "decision_certainty": 1.0,
        "confidence_semantics": "deterministic_gate",
        "model_run_manifest_id": model_run_manifest_id,
        "auto_reuse_enabled": auto_reuse_enabled,
        "created_at": created_at or datetime.now(UTC),
    }
    draft = RankedModelRecommendation.model_construct(
        checksum="0" * 64,
        **payload,
    )
    payload["checksum"] = ranked_model_recommendation_checksum(
        draft.model_dump(mode="json")
    )
    return RankedModelRecommendation.model_validate(payload)


def build_split_manifest(
    *,
    profile_index: list[dict[str, Any]],
    profiles: list[RequirementProfile],
    models: list[OSPModelVersionSnapshot],
) -> dict[str, Any]:
    profile_ids = {str(profile.profile_id) for profile in profiles}
    index_by_profile: dict[str, dict[str, Any]] = {}
    for row in profile_index:
        profile_id = str(row.get("profile_id") or "")
        if profile_id in index_by_profile:
            raise ModelRankingError(
                f"Duplicate profile index row: {profile_id}"
            )
        index_by_profile[profile_id] = row
    missing = sorted(profile_ids - set(index_by_profile))
    extra = sorted(set(index_by_profile) - profile_ids)
    split_compounds: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    invalid_splits: list[str] = []
    for profile_id in sorted(profile_ids):
        row = index_by_profile.get(profile_id)
        if row is None:
            continue
        split = str(row.get("benchmark_split") or "")
        if split not in split_compounds:
            invalid_splits.append(f"{profile_id}:{split}")
            continue
        split_compounds[split].add(str(row.get("compound_id") or ""))
    overlap_counts = {
        "train_validation": len(
            split_compounds["train"]
            & split_compounds["validation"]
        ),
        "train_test": len(
            split_compounds["train"]
            & split_compounds["test"]
        ),
        "validation_test": len(
            split_compounds["validation"]
            & split_compounds["test"]
        ),
    }
    compound_disjoint = (
        not missing
        and not extra
        and not invalid_splits
        and all(split_compounds.values())
        and not any(overlap_counts.values())
    )
    source_ids = sorted({model.source_id for model in models})
    repositories = sorted({model.repository_url for model in models})
    commits = sorted({model.repository_commit for model in models})
    return {
        "split_schema_version": "1.0.0",
        "profile_count": len(profiles),
        "model_count": len(models),
        "compound_disjoint": {
            "status": (
                "eligible" if compound_disjoint else "failed"
            ),
            "split_compound_counts": {
                key: len(value)
                for key, value in split_compounds.items()
            },
            "overlap_counts": overlap_counts,
            "missing_profile_ids": missing,
            "extra_profile_ids": extra,
            "invalid_splits": invalid_splits,
        },
        "model_family_disjoint": {
            "status": "ineligible_no_grade_2_3_labels",
            "model_family_count": len(
                {model.model_family for model in models}
            ),
        },
        "source_repository_disjoint": {
            "status": (
                "ineligible_single_source_repository"
                if len(source_ids) == 1 and len(repositories) == 1
                else "not_executed_no_grade_2_3_labels"
            ),
            "source_ids": source_ids,
            "repositories": repositories,
        },
        "time_split": {
            "status": "ineligible_single_repository_commit",
            "repository_commits": commits,
        },
        "learned_ranking_evaluation_eligible": False,
        "generalization_claim_eligible": False,
    }


def audit_ranking_run(
    *,
    profiles: list[RequirementProfile],
    recommendations: list[RankedModelRecommendation],
    model_count: int,
    split_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_by_id = {
        profile.profile_id: profile for profile in profiles
    }
    recommendation_profile_ids = [
        recommendation.profile_id
        for recommendation in recommendations
    ]
    hard_constraint_violations: list[str] = []
    provenance_errors: list[str] = []
    candidate_coverage_errors: list[str] = []
    hard_negative_errors: list[str] = []
    for recommendation in recommendations:
        profile = profile_by_id.get(recommendation.profile_id)
        if profile is None:
            candidate_coverage_errors.append(
                f"{recommendation.ranking_run_id}:unknown_profile"
            )
            continue
        if len(recommendation.candidates) != model_count:
            candidate_coverage_errors.append(
                f"{recommendation.ranking_run_id}:"
                f"{len(recommendation.candidates)}:{model_count}"
            )
        if not any(
            any(
                check.severity == "hard"
                and check.result in {"fail", "unknown"}
                for check in candidate.constraint_checks
            )
            for candidate in recommendation.candidates
        ):
            hard_negative_errors.append(
                str(recommendation.ranking_run_id)
            )
        if recommendation.decision in {"reuse", "adapt"}:
            selected = next(
                (
                    candidate
                    for candidate in recommendation.candidates
                    if candidate.model_version_id
                    == recommendation.selected_model_version_id
                ),
                None,
            )
            if selected is None or any(
                check.severity == "hard"
                and check.result in {"fail", "unknown"}
                for check in selected.constraint_checks
            ):
                hard_constraint_violations.append(
                    str(recommendation.ranking_run_id)
                )
        for candidate in recommendation.candidates:
            for check in candidate.constraint_checks:
                if (
                    check.result != "not_applicable"
                    and not check.evidence_ids
                    and not check.model_asset_locations
                ):
                    provenance_errors.append(check.constraint_id)

    compound_count = len(
        {
            profile.compound_identity.compound_id
            for profile in profiles
        }
    )
    objective_counts = Counter(
        next(
            requirement.expected_value
            for requirement in profile.requirements
            if requirement.category == "validation_context"
        )
        for profile in profiles
    )
    special_population_count = sum(
        objective_counts.get(value, 0)
        for value in (
            "renal_impairment",
            "hepatic_impairment",
            "pediatric",
            "geriatric",
            "pregnancy",
            "special_population",
        )
    )
    reuse_count = sum(
        recommendation.decision == "reuse"
        for recommendation in recommendations
    )
    false_reuse_count = sum(
        recommendation.decision == "reuse"
        and not any(
            candidate.model_version_id
            == recommendation.selected_model_version_id
            and candidate.run_label_grade == 3
            and candidate.ranking_label == "reuse_positive"
            and candidate.eligibility == "eligible"
            for candidate in recommendation.candidates
        )
        for recommendation in recommendations
    )
    false_reuse_interval = wilson_interval(
        false_reuse_count,
        len(recommendations),
    )
    abstain_count = sum(
        recommendation.decision == "abstain"
        for recommendation in recommendations
    )
    checks = {
        "profile_coverage_complete": (
            len(profiles) == len(recommendations)
            and set(profile_by_id) == set(recommendation_profile_ids)
        ),
        "recommendation_profile_ids_unique": (
            len(recommendation_profile_ids)
            == len(set(recommendation_profile_ids))
        ),
        "minimum_request_count_30": len(profiles) >= 30,
        "minimum_compound_count_10": compound_count >= 10,
        "compound_disjoint_split_valid": (
            split_manifest is not None
            and (
                split_manifest.get("compound_disjoint") or {}
            ).get("status")
            == "eligible"
        ),
        "objective_coverage_base_pk_ddi_special_population": (
            objective_counts.get("base_pk", 0) > 0
            and (
                objective_counts.get("ddi_victim", 0)
                + objective_counts.get("ddi_perpetrator", 0)
            )
            > 0
            and special_population_count > 0
        ),
        "each_request_has_hard_negative": not hard_negative_errors,
        "minimum_abstain_fraction_20_percent": (
            abstain_count / len(recommendations) >= 0.2
            if recommendations
            else False
        ),
        "candidate_model_coverage_complete": (
            not candidate_coverage_errors
        ),
        "critical_hard_constraint_violation_zero": (
            not hard_constraint_violations
        ),
        "constraint_provenance_complete": not provenance_errors,
        "automatic_reuse_requires_grade_3_run_label": all(
            recommendation.decision != "reuse"
            or any(
                candidate.model_version_id
                == recommendation.selected_model_version_id
                and candidate.run_label_grade == 3
                and candidate.ranking_label == "reuse_positive"
                and candidate.validation_run_id is not None
                for candidate in recommendation.candidates
            )
            for recommendation in recommendations
        ),
        "false_reuse_count_zero": false_reuse_count == 0,
    }
    return {
        "audit_schema_version": "1.0.0",
        "ranker_version": RANKER_VERSION,
        "benchmark_level": "silver_requests_with_run_derived_labels",
        "gold_claim": False,
        "generalization_claim_eligible": False,
        "accepted": all(checks.values()),
        "checks": checks,
        "counts": {
            "requests": len(profiles),
            "compounds": compound_count,
            "models_per_request": model_count,
            "recommendations": len(recommendations),
            "reuse": reuse_count,
            "adapt": sum(
                recommendation.decision == "adapt"
                for recommendation in recommendations
            ),
            "abstain": abstain_count,
            "hard_constraint_violations": len(
                hard_constraint_violations
            ),
            "false_reuse": false_reuse_count,
        },
        "objective_counts": dict(sorted(objective_counts.items())),
        "false_reuse_rate": (
            false_reuse_count / len(recommendations)
            if recommendations
            else None
        ),
        "false_reuse_95_percent_wilson_interval": {
            "lower": false_reuse_interval[0],
            "upper": false_reuse_interval[1],
        },
        "errors": {
            "candidate_coverage": candidate_coverage_errors[:100],
            "hard_negative": hard_negative_errors[:100],
            "hard_constraint_violations": (
                hard_constraint_violations[:100]
            ),
            "constraint_provenance": provenance_errors[:100],
        },
        "limitations": [
            "A grade 3 label permits deterministic reuse only for the exact "
            "validated model and request scope.",
            "The run-label corpus is still too small for learned ranking, "
            "probability calibration, or a generalization claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic candidate matrices and forced-abstain "
            "public PBPK model decisions."
        )
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--model-run-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--profile-index",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = [
        RequirementProfile.model_validate(row)
        for row in _iter_jsonl(args.profiles)
    ]
    models = [
        OSPModelVersionSnapshot.model_validate(row)
        for row in _iter_jsonl(args.models)
    ]
    if not profiles or not models:
        raise ModelRankingError("Profiles and models must be non-empty.")
    profile_index = list(_iter_jsonl(args.profile_index))
    split_manifest = build_split_manifest(
        profile_index=profile_index,
        profiles=profiles,
        models=models,
    )
    run_manifest_id = model_run_manifest_uuid(
        args.model_run_manifest
    )
    grade_3_count = sum(
        model.validation_run_label is not None
        and model.validation_run_label.run_label_grade == 3
        and model.validation_run_label.ranking_label == "reuse_positive"
        for model in models
    )
    calibration_manifest = disabled_calibration_manifest(
        request_count=len(profiles),
        grade_2_count=0,
        grade_3_count=grade_3_count,
    )
    recommendations: list[RankedModelRecommendation] = []
    for index, profile in enumerate(profiles, 1):
        candidates = retrieve_model_candidates(
            profile=profile,
            models=models,
        )
        auto_reuse_enabled = any(
            candidate.candidate_class == "exact_compound_model"
            and candidate.eligibility == "eligible"
            and candidate.run_label_grade == 3
            and candidate.ranking_label == "reuse_positive"
            for candidate in candidates
        )
        recommendations.append(
            build_ranking_recommendation(
                profile=profile,
                candidates=candidates,
                model_run_manifest_id=run_manifest_id,
                auto_reuse_enabled=auto_reuse_enabled,
            )
        )
        if index % 10 == 0:
            print(
                _canonical_json(
                    {
                        "profiles_ranked": index,
                        "profile_total": len(profiles),
                    }
                ),
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=False)
    output_checksums = {
        "ranked_recommendations.jsonl.gz": _atomic_gzip_jsonl(
            args.output / "ranked_recommendations.jsonl.gz",
            [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ],
        ),
        "ranked_model_recommendation.schema.json": _atomic_json(
            args.output
            / "ranked_model_recommendation.schema.json",
            RankedModelRecommendation.model_json_schema(),
        ),
        "calibration_manifest.json": _atomic_json(
            args.output / "calibration_manifest.json",
            calibration_manifest,
        ),
        "split_manifest.json": _atomic_json(
            args.output / "split_manifest.json",
            split_manifest,
        ),
    }
    audit = audit_ranking_run(
        profiles=profiles,
        recommendations=recommendations,
        model_count=len(models),
        split_manifest=split_manifest,
    )
    output_checksums["audit.json"] = _atomic_json(
        args.output / "audit.json",
        audit,
    )
    build_manifest = {
        "ranking_build_schema_version": "1.0.0",
        "ranker_version": RANKER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "accepted": audit["accepted"],
        "model_run_manifest_id": str(run_manifest_id),
        "input_artifacts": {
            "profiles": {
                "path": str(args.profiles),
                "sha256": _sha256_file(args.profiles),
            },
            "models": {
                "path": str(args.models),
                "sha256": _sha256_file(args.models),
            },
            "model_run_manifest": {
                "path": str(args.model_run_manifest),
                "sha256": _sha256_file(args.model_run_manifest),
            },
            "profile_index": {
                "path": str(args.profile_index),
                "sha256": _sha256_file(args.profile_index),
            },
        },
        "output_checksums": output_checksums,
    }
    _atomic_json(args.output / "build_manifest.json", build_manifest)
    if not audit["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
