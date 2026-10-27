from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pbpk_kb.recommendation.model_matching.benchmark import (
    build_benchmark,
    relevance_grade,
    run_experiment,
)
from pbpk_kb.recommendation.model_matching.checkpoint_selection import (
    select_checkpoint,
)
from pbpk_kb.recommendation.model_matching.graphs import (
    GraphEdge,
    GraphNode,
    TypedGraph,
    build_execution_requirement_graph,
    build_model_graph,
    endpoint_categories,
    requirement_graph_from_model,
)
from pbpk_kb.recommendation.model_matching.model_kg_graphs import (
    condense_model_graph,
)
from pbpk_kb.recommendation.model_matching.metrics import _ece, _pairwise_accuracy, evaluate_rankings
from pbpk_kb.recommendation.model_matching.molecular_features import _molecule_data
from pbpk_kb.recommendation.model_matching.promotion import evaluate_promotion
from pbpk_kb.recommendation.model_matching.serving import score_for_serving
from pbpk_kb.recommendation.model_matching.neural import (
    AUXILIARY_NODE_FEATURE_DIMENSIONS,
    build_matcher,
    tensorize_graph,
)
from pbpk_kb.recommendation.model_matching.simulation_validation import (
    evaluate_simulation_rows,
    evaluate_validated_artifacts,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    apply_use_tier_gate,
    assess_candidate_use,
    score_candidate,
)
from pbpk_kb.recommendation.model_matching.statistics import compare_research_suite
from pbpk_kb.recommendation.model_matching.training import (
    _batched_encoded_outputs,
    _decision_class_weighting,
    _decision_classification,
    _decision_label,
    _validation_joint_score,
    default_variants,
    select_variants,
)


def test_pairwise_score_ties_are_half_correct_in_both_label_orders() -> None:
    assert _pairwise_accuracy([0, 3], [0.5, 0.5]) == 0.5
    assert _pairwise_accuracy([3, 0], [0.5, 0.5]) == 0.5


def test_ece_uses_probability_intervals_not_interleaved_ranks() -> None:
    # Opposite errors in different confidence bins must not cancel each other.
    assert _ece([0.1, 0.1, 0.9, 0.9], [1, 1, 0, 0], bins=2) == pytest.approx(0.9)
    assert _ece([0.0, 1.0], [0, 1], bins=2) == 0.0
    with pytest.raises(ValueError):
        _ece([1.1], [1])


def _model(
    model_id: str,
    family: str,
    *,
    route: str = "Oral",
    enzyme: str = "CYP3A4",
) -> dict[str, object]:
    return {
        "model_id": f"model:{family}",
        "model_version_id": model_id,
        "model_family": family,
        "compounds": [family],
        "species": ["Human"],
        "populations": ["Adult"],
        "routes": [route],
        "formulations": ["Solution"],
        "process_types": ["absorption", "distribution", "metabolism"],
        "enzymes": [enzyme],
        "transporters": [],
        "compartments": ["Liver", "Kidney", "Plasma"],
        "output_names": ["Plasma concentration"],
        "software": "PK-Sim",
        "license": {"status": "allowed"},
        "execution_probe_status": "passed",
        "pkml_parse_status": "parsed",
        "simulation_count": 1,
        "parameter_record_count": 10,
        "dataset_reference_count": 1,
        "validation_metric_count": 1,
    }


def test_typed_hierarchical_match_finds_missing_mechanism() -> None:
    source = _model("version-a", "Midazolam")
    candidate = _model("version-b", "Midazolam", enzyme="CYP2D6")
    requirement = requirement_graph_from_model(
        source,
        query_id="query-midazolam",
    )

    result = score_candidate(requirement, build_model_graph(candidate))

    assert result.hard_gate_passed is False
    assert "Enzyme:cyp3a4" in result.hard_gate_failures
    assert result.adaptation_cost > 0
    assert any(
        operation.node_type == "Enzyme"
        for operation in result.adaptation_operations
    )


def test_use_assessment_separates_ranking_from_reuse_gate() -> None:
    source = _model("version-a", "Midazolam")
    candidate = _model("version-b", "Midazolam")
    candidate.update(
        {
            "engineering_accepted": True,
            "scientific_accepted": False,
            "reusable_candidate": False,
        }
    )
    requirement = requirement_graph_from_model(
        source,
        query_id="query-midazolam",
    )
    model_graph = build_model_graph(candidate)
    match = score_candidate(requirement, model_graph)

    assessment = assess_candidate_use(
        requirement,
        model_graph,
        match=match,
    )

    assert match.score > 0
    assert assessment.structural_gate_passed is True
    assert assessment.direct_reuse_gate_passed is False
    assert assessment.maximum_use_tier == "controlled_adaptation"
    assert "ScientificReuse:not_validated" in (
        assessment.direct_reuse_gate_failures
    )


def test_use_tier_gate_caps_claim_without_changing_rank_class() -> None:
    source = _model("version-a", "Midazolam")
    candidate = _model("version-b", "Midazolam")
    candidate.update(
        {
            "engineering_accepted": True,
            "scientific_accepted": False,
            "reusable_candidate": False,
        }
    )
    requirement = requirement_graph_from_model(source, query_id="query")
    graph = build_model_graph(candidate)
    assessment = assess_candidate_use(requirement, graph)

    assert apply_use_tier_gate(0, assessment, enabled=True) == 1
    assert apply_use_tier_gate(0, assessment, enabled=False) == 0


def test_execution_requirement_graph_preserves_mechanism_requirements() -> None:
    graph = build_execution_requirement_graph(
        {
            "task_id": "task-a",
            "active_moieties": ["midazolam"],
            "observed_analytes": ["midazolam"],
            "species": "human",
            "process_types": ["metabolism"],
            "enzymes": ["CYP3A4"],
            "transporters": ["ABCB1"],
            "ddi_roles": ["victim"],
            "scenario": {"route": "oral"},
            "required_outputs": ["plasma_concentration_time"],
        }
    )

    node_types = {node.node_type for node in graph.nodes}
    assert {"ADMEProcess", "Enzyme", "Transporter", "DDIRole"}.issubset(
        node_types
    )


def test_different_compound_with_mechanism_overlap_is_reference_only() -> None:
    source = _model("version-a", "Midazolam")
    candidate = _model("version-b", "Erythromycin")
    candidate.update(
        {
            "engineering_accepted": True,
            "scientific_accepted": True,
            "reusable_candidate": True,
        }
    )
    requirement = requirement_graph_from_model(source, query_id="query")
    graph = build_model_graph(candidate)

    assessment = assess_candidate_use(requirement, graph)

    assert assessment.same_active_moiety is False
    assert assessment.direct_reuse_gate_passed is False
    assert assessment.maximum_use_tier == "mechanistic_reference"


def test_execution_decision_labels_preserve_all_four_use_tiers() -> None:
    query = {"task": "plasma_concentration_time_prediction"}

    assert [_decision_label(query, grade) for grade in (3, 2, 1, 0)] == [
        0,
        1,
        2,
        3,
    ]


def test_decision_class_weights_use_only_supplied_training_scope() -> None:
    queries = [{"query_id": "train-q"}]
    candidates = ["reuse", "adapt", "reference", "reject-a", "reject-b"]
    labels = {
        ("train-q", "reuse"): 3,
        ("train-q", "adapt"): 2,
        ("train-q", "reference"): 1,
        ("train-q", "reject-a"): 0,
        ("train-q", "reject-b"): 0,
        ("validation-q", "reuse"): 0,
    }

    weighting = _decision_class_weighting(
        queries=queries,
        candidate_ids=candidates,
        labels=labels,
    )

    assert weighting["class_counts"] == [1, 1, 1, 2]
    assert sum(weighting["class_weights"]) / 4 == pytest.approx(1.0)
    assert weighting["class_weights"][3] < weighting["class_weights"][0]
    assert weighting["scope"] == "train_query_x_train_candidate_only"


def test_validation_joint_score_penalizes_decision_collapse() -> None:
    high_ndcg_collapsed = _validation_joint_score(0.95, 0.20)
    balanced = _validation_joint_score(0.80, 0.75)

    assert balanced > high_ndcg_collapsed


def test_batched_heads_match_serial_pair_outputs() -> None:
    torch = pytest.importorskip("torch")
    source = _model("version-a", "Midazolam")
    requirement = requirement_graph_from_model(source, query_id="query")
    candidates = {
        "same": build_model_graph(_model("version-b", "Midazolam")),
        "other": build_model_graph(_model("version-c", "Erythromycin")),
    }
    feature_dimensions = 128 + AUXILIARY_NODE_FEATURE_DIMENSIONS
    matcher = build_matcher(
        "cg_hhgm",
        feature_dimensions=feature_dimensions,
    ).eval()
    query_tensor = tensorize_graph(
        requirement,
        feature_dimensions=128,
    )
    model_tensors = {
        key: tensorize_graph(graph, feature_dimensions=128)
        for key, graph in candidates.items()
    }
    encoded_query = matcher.encode_graph(query_tensor)
    encoded_models = {
        key: matcher.encode_graph(value) for key, value in model_tensors.items()
    }
    serial = [
        matcher.forward_encoded(
            query_tensor,
            model_tensors[key],
            encoded_query,
            encoded_models[key],
        )
        for key in model_tensors
    ]
    batched = _batched_encoded_outputs(
        model=matcher,
        query_tensor=query_tensor,
        candidate_ids=list(model_tensors),
        model_tensors=model_tensors,
        encoded_query=encoded_query,
        encoded_models=encoded_models,
    )

    for serial_output, batched_output in zip(serial, batched, strict=True):
        for key in serial_output:
            assert torch.allclose(
                serial_output[key],
                batched_output[key],
                atol=1e-6,
                rtol=0.0,
            ), key
        assert int(serial_output["decision_logits"].argmax()) == int(
            batched_output["decision_logits"].argmax()
        )


def test_decision_classification_reports_confusion_and_macro_f1() -> None:
    result = _decision_classification(
        [0, 1, 2, 3],
        [0, 1, 2, 2],
    )

    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"][3][2] == 1
    assert result["macro_f1"] < 1.0


def test_candidate_retrieval_sanity_labels_are_graded() -> None:
    source = _model("version-a", "Alfentanil")
    same_family = _model("version-b", "Alfentanil")
    mechanism_reference = _model("version-c", "Midazolam")
    unrelated = _model("version-d", "Amikacin", route="Intravenous", enzyme="UGT1A1")

    assert relevance_grade(source, source) == 3
    assert relevance_grade(source, same_family) == 2
    assert relevance_grade(source, mechanism_reference) == 1
    assert relevance_grade(source, unrelated) == 0


def test_metrics_report_safety_and_ranking() -> None:
    metrics = evaluate_rankings(
        [
            {
                "query_id": "q1",
                "candidate_id": "positive",
                "score": 0.9,
                "relevance_grade": 3,
                "hard_gate_passed": True,
                "reuse_decision": True,
            },
            {
                "query_id": "q1",
                "candidate_id": "negative",
                "score": 0.1,
                "relevance_grade": 0,
                "hard_gate_passed": False,
                "reuse_decision": False,
            },
        ]
    )["aggregate"]

    assert metrics["ndcg@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["false_reuse_rate"] == 0.0
    assert metrics["hard_constraint_violation_rate"] == 0.0


def test_metrics_report_raw_and_gated_decision_quality() -> None:
    metrics = evaluate_rankings(
        [
            {
                "query_id": "q1",
                "candidate_id": "reuse",
                "score": 1.0,
                "relevance_grade": 3,
                "raw_decision_class": 0,
                "decision_class": 0,
                "predicted_use_tier": "direct_reuse",
                "reuse_decision": True,
                "hard_gate_passed": True,
            },
            {
                "query_id": "q1",
                "candidate_id": "reject",
                "score": 0.0,
                "relevance_grade": 0,
                "raw_decision_class": 2,
                "decision_class": 3,
                "predicted_use_tier": "reject",
                "reuse_decision": False,
                "hard_gate_passed": False,
            },
        ]
    )["aggregate"]

    assert metrics["raw_decision_accuracy"] == 0.5
    assert metrics["gated_decision_accuracy"] == 1.0


def test_mechanistic_reference_is_not_an_unsafe_top1_claim() -> None:
    metrics = evaluate_rankings(
        [
            {
                "query_id": "q1",
                "candidate_id": "reference",
                "score": 0.9,
                "relevance_grade": 1,
                "hard_gate_passed": False,
                "predicted_use_tier": "mechanistic_reference",
                "reuse_decision": False,
                "adaptation_decision": False,
            },
            {
                "query_id": "q1",
                "candidate_id": "negative",
                "score": 0.1,
                "relevance_grade": 0,
                "hard_gate_passed": False,
                "predicted_use_tier": "reject",
                "reuse_decision": False,
                "adaptation_decision": False,
            },
        ]
    )["aggregate"]

    assert metrics["unsafe_top1_rate"] == 0.0
    assert metrics["false_reuse_rate"] == 0.0


def test_invalid_adaptation_claim_is_counted_as_unsafe() -> None:
    metrics = evaluate_rankings(
        [
            {
                "query_id": "q1",
                "candidate_id": "reference-only",
                "score": 0.9,
                "relevance_grade": 1,
                "hard_gate_passed": False,
                "predicted_use_tier": "controlled_adaptation",
                "reuse_decision": False,
                "adaptation_decision": True,
            }
        ]
    )["aggregate"]

    assert metrics["unsafe_top1_rate"] == 1.0
    assert metrics["false_adaptation_rate"] == 1.0


def test_metrics_report_real_adaptation_cost_and_operation_quality() -> None:
    metrics = evaluate_rankings(
        [
            {
                "query_id": "q1",
                "candidate_id": "adapted",
                "score": 0.9,
                "relevance_grade": 2,
                "hard_gate_passed": True,
                "reuse_decision": False,
                "predicted_adaptation_cost": 1.2,
                "adaptation_cost_label": 1.0,
                "predicted_adaptation_operations": ["dose"],
                "adaptation_operation_labels": ["dose"],
            },
            {
                "query_id": "q1",
                "candidate_id": "rejected",
                "score": 0.1,
                "relevance_grade": 0,
                "hard_gate_passed": False,
                "reuse_decision": False,
            },
        ]
    )["aggregate"]

    assert metrics["adaptation_cost_mae"] == pytest.approx(0.2)
    assert metrics["adaptation_operation_micro_f1"] == 1.0
    assert metrics["adaptation_operation_macro_f1"] == 1.0
    assert metrics["adaptation_operation_f1_by_type"] == {"dose": 1.0}


def test_metrics_separate_reusable_and_reference_only_query_cohorts() -> None:
    result = evaluate_rankings(
        [
            {
                "query_id": "reuse-query",
                "candidate_id": "reuse",
                "score": 0.9,
                "relevance_grade": 3,
                "predicted_use_tier": "direct_reuse",
                "hard_gate_passed": True,
                "reuse_decision": True,
                "adaptation_decision": False,
            },
            {
                "query_id": "reuse-query",
                "candidate_id": "negative",
                "score": 0.1,
                "relevance_grade": 0,
                "predicted_use_tier": "reject",
                "hard_gate_passed": False,
                "reuse_decision": False,
                "adaptation_decision": False,
            },
            {
                "query_id": "reference-query",
                "candidate_id": "reference",
                "score": 0.8,
                "relevance_grade": 1,
                "predicted_use_tier": "mechanistic_reference",
                "hard_gate_passed": False,
                "reuse_decision": False,
                "adaptation_decision": False,
            },
            {
                "query_id": "reference-query",
                "candidate_id": "unrelated",
                "score": 0.2,
                "relevance_grade": 0,
                "predicted_use_tier": "reject",
                "hard_gate_passed": False,
                "reuse_decision": False,
                "adaptation_decision": False,
            },
        ]
    )

    assert result["positive_candidate_query_aggregate"]["query_count"] == 1
    assert result["positive_candidate_query_aggregate"]["recall@1"] == 1.0
    assert result["cohort_aggregate"]["direct_reuse_available"][
        "query_count"
    ] == 1
    assert result["cohort_aggregate"]["mechanistic_reference_only"][
        "reference_hitrate@1"
    ] == 1.0
    assert result["aggregate"]["reuse_adapt_availability_f1"] == 1.0
    assert result["aggregate"]["reuse_adapt_abstention_specificity"] == 1.0


def test_proposed_method_has_all_preregistered_ablation_variants() -> None:
    variants = {row.name: row for row in default_variants(["cg_hhgm"])}

    assert variants["cg_hhgm_no_adaptation_head_or_cost"].adaptation_weight == 0
    assert (
        variants["cg_hhgm_no_molecular_representation"]
        .use_molecular_representation
        is False
    )
    assert (
        variants["cg_hhgm_no_simulation_verification"]
        .use_simulation_verification
        is False
    )


def test_training_variant_selection_is_preregistered_and_ordered() -> None:
    selected = select_variants(
        ["gmn", "rgcn_dual_encoder"],
        ["rgcn_dual_encoder", "gmn"],
    )

    assert [variant.name for variant in selected] == [
        "gmn",
        "rgcn_dual_encoder",
    ]
    with pytest.raises(ValueError, match="non-preregistered"):
        select_variants(["gmn"], ["gmn_unknown"])


def test_concentration_time_output_supports_derived_auc_and_cmax() -> None:
    assert endpoint_categories(["Plasma concentration over time"]) == [
        "auc",
        "cmax",
        "concentration_time",
    ]


def test_experiment_manifest_keeps_unrun_methods_explicit(tmp_path: Path) -> None:
    model_path = tmp_path / "models.jsonl"
    model_path.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                _model("version-a", "Midazolam"),
                _model("version-b", "Alfentanil"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "experiment"

    manifest = run_experiment(
        model_paths=[model_path],
        output_directory=output,
    )

    assert manifest["benchmark_level"] == "candidate_retrieval_sanity"
    assert manifest["evaluation_scope"] == "candidate_retrieval_and_representation_sanity_only"
    assert manifest["scientific_claim_eligible"] is False
    assert manifest["gpu_or_external_methods"]["gmn"]["status"] == "not_executed"
    assert "test_comparison" in manifest
    assert (output / "model_gate_graphs.jsonl").is_file()
    assert (output / "experiment_manifest.json").is_file()


def test_benchmark_split_is_model_family_stable() -> None:
    records = [
        _model("version-a", "Alfentanil"),
        _model("version-b", "Alfentanil"),
        _model("version-c", "Midazolam"),
    ]
    queries, _, _ = build_benchmark(records)
    alfentanil_splits = {
        query["split"]
        for query in queries
        if query["source_model_family"] == "Alfentanil"
    }
    assert len(alfentanil_splits) == 1


def test_semantic_condensation_retains_source_ids() -> None:
    graph = TypedGraph(
        graph_id="version-a",
        graph_kind="executable_model_full_kg",
        nodes=(
            GraphNode("version-a", "ModelVersion", "ModelVersion:a", "a", "root"),
            GraphNode(
                "enzyme-1",
                "Enzyme",
                "Enzyme:cyp3a4",
                "CYP3A4",
                "mechanism",
                attributes={"source_node_id": "enzyme-1", "occurrence_count": 2},
            ),
            GraphNode(
                "enzyme-2",
                "Enzyme",
                "Enzyme:cyp3a4",
                "CYP3A4",
                "mechanism",
                attributes={"source_node_id": "enzyme-2", "occurrence_count": 3},
            ),
        ),
        edges=(
            GraphEdge(
                "version-a",
                "IMPLEMENTS_ENZYME",
                "enzyme-1",
                attributes={"source_edge_id": "edge-1"},
            ),
            GraphEdge(
                "version-a",
                "IMPLEMENTS_ENZYME",
                "enzyme-2",
                attributes={"source_edge_id": "edge-2"},
            ),
        ),
    )

    condensed = condense_model_graph(graph)

    enzyme = next(node for node in condensed.nodes if node.node_type == "Enzyme")
    assert enzyme.attributes["source_node_ids"] == ["enzyme-1", "enzyme-2"]
    assert enzyme.attributes["occurrence_count"] == 5
    assert len(condensed.edges) == 1
    assert condensed.edges[0].attributes["source_edge_ids"] == ["edge-1", "edge-2"]


def test_serving_without_checkpoint_is_explicit_fallback(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PBPK_MODEL_MATCHER_SERVING_MANIFEST", raising=False)
    requirement = requirement_graph_from_model(
        _model("version-a", "Midazolam"),
        query_id="query-midazolam",
    )

    result = score_for_serving(
        requirement,
        build_model_graph(_model("version-a", "Midazolam")),
    )

    assert result["learned"] is False
    assert result["matching_method"] == "cg_hhgm_deterministic_fallback"
    assert result["serving_scope"] == "deterministic_fallback"


def test_simulation_validation_reports_fold_error_and_hard_failures() -> None:
    result = evaluate_simulation_rows(
        [
            {
                "run_id": "run-1",
                "endpoint": "AUC",
                "observed": 100.0,
                "predicted": 150.0,
                "compile_success": True,
                "execution_success": True,
                "mass_balance_passed": True,
                "dimensional_check_passed": True,
                "reproducibility_passed": True,
            },
            {
                "run_id": "run-2",
                "endpoint": "Cmax",
                "observed": 10.0,
                "predicted": 30.0,
                "compile_success": True,
                "execution_success": True,
                "mass_balance_passed": False,
                "dimensional_check_passed": True,
                "reproducibility_passed": True,
            },
        ]
    )

    assert result["aggregate"]["fraction_within_two_fold"] == 0.5
    assert result["scientific_gate_failure_count"] == 1


def test_validated_artifact_adapter_checks_checksum(tmp_path: Path) -> None:
    import hashlib

    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_name": "external validation",
                "positive_observed_mol_per_litre": [1.0, 2.0],
                "positive_predicted_mol_per_litre": [1.1, 1.8],
                "dataset_guard_passed": True,
                "source_location": "dataset.json#/0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "protocol_id": "run-1",
                "validation_grade": "external_study_validated",
                "ranking_label": "reuse_positive",
                "verification_dataset_metrics_sha256": hashlib.sha256(
                    metrics_path.read_bytes()
                ).hexdigest(),
                "engineering_checks": {
                    "all_artifacts_verified": True,
                    "all_simulations_passed": True,
                    "csv_repeatability_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_validated_artifacts(
        audit_path=audit_path,
        dataset_metrics_path=metrics_path,
    )

    assert result["dataset_count"] == 1
    assert result["aggregate"]["positive_pair_count"] == 2
    assert result["recommendation_quality_claim_eligible"] is False


def test_molclr_graph_mapping_matches_official_indices() -> None:
    graph = _molecule_data("CCO")

    assert graph.x.tolist() == [[5, 0], [5, 0], [7, 0]]
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.tolist() == [[0, 0], [0, 0], [0, 0], [0, 0]]


def test_numeric_regimen_features_are_appended_without_overwriting_embedding() -> None:
    graph = TypedGraph(
        graph_id="query-dose",
        graph_kind="requirement",
        nodes=(
            GraphNode(
                "dose",
                "Regimen",
                "Regimen:single dose",
                "single_dose",
                "context",
                attributes={"dose_value": 40.0, "dose_unit": "mg"},
            ),
        ),
        edges=(),
    )
    stored = {"Regimen:single dose": [0.25] * 8}

    tensor = tensorize_graph(
        graph,
        feature_dimensions=8,
        node_embeddings=stored,
    )

    assert tensor.x.shape == (1, 8 + AUXILIARY_NODE_FEATURE_DIMENSIONS)
    assert tensor.x[0, :8].tolist() == pytest.approx([0.25] * 8)
    assert tensor.x[0, 8 + 4] == pytest.approx(math.log1p(40.0))


def test_statistics_discovers_supplemental_molclr_metrics(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    suite = tmp_path / "suite"
    molecular = suite / "molecular_molclr"
    benchmark.mkdir()
    molecular.mkdir(parents=True)
    metric = {
        "aggregate": {
            "ndcg@10": 1.0,
            "false_reuse_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
            "unsafe_top1_rate": 0.0,
            "aurc": None,
        },
        "per_query": [{"query_id": "query-a", "ndcg@10": 1.0}],
    }
    (benchmark / "metrics_uniform_rule_test.json").write_text(
        json.dumps(metric), encoding="utf-8"
    )
    (benchmark / "perturbation_metrics_uniform_rule_test.json").write_text(
        json.dumps(
            {
                "details": [
                    {
                        "base_query_id": "query-a",
                        "perturbed_query_id": "query-a-plus-1pct",
                        "top1_stable": True,
                        "topk_jaccard": 1.0,
                        "kendall_tau": 1.0,
                        "reuse_decision_stable": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (molecular / "metrics_official_molclr_pretrained_gin_test.json").write_text(
        json.dumps(metric), encoding="utf-8"
    )

    result = compare_research_suite(
        benchmark_directory=benchmark,
        suite_directory=suite,
    )

    assert "official_molclr_pretrained_gin" in result["methods"]
    assert result["methods"]["uniform_rule"]["perturbation"]["pair_count"] == 1
    assert (
        result["methods"]["uniform_rule"]["perturbation"]["metrics"]
        ["kendall_tau"]["mean"]
        == 1.0
    )


def test_promotion_blocks_unstable_local_perturbation_ranking(tmp_path: Path) -> None:
    import torch

    checkpoint = tmp_path / "seed_17.pt"
    torch.save(
        {
            "variant": {
                "architecture": "cg_hhgm",
                "adaptation_weight": 0.25,
            }
        },
        checkpoint,
    )
    seed_metrics = tmp_path / "seed_17_metrics.json"
    seed_metrics.write_text(
        json.dumps(
            {
                "aggregate": {
                    "ndcg@10": 1.0,
                    "false_reuse_rate": 0.0,
                    "hard_constraint_violation_rate": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "experiment_manifest.json"
    benchmark.write_text(
        json.dumps(
            {
                "benchmark_level": "execution_derived",
                "scientific_claim_eligible": True,
                "perturbed_query_count": 1,
                "test_comparison": {
                    "uniform_rule": {
                        "ndcg@10": 0.9,
                        "false_reuse_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    feature_store = tmp_path / "features"
    feature_store.mkdir()
    (feature_store / "manifest.json").write_text(
        json.dumps({"node_embeddings_sha256": "a" * 64}),
        encoding="utf-8",
    )
    perturbation = tmp_path / "seed_17_perturbation_metrics.json"
    perturbation.write_text(
        json.dumps(
            {
                "top1_stability_rate": 0.5,
                "mean_topk_jaccard": 1.0,
                "mean_kendall_tau": 1.0,
                "reuse_layer_stability_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_promotion(
        checkpoint=checkpoint,
        seed_metrics=seed_metrics,
        benchmark_manifest=benchmark,
        feature_store=feature_store,
        perturbation_metrics=perturbation,
        output_directory=tmp_path / "promotion",
        method="CG-HHGM",
    )

    assert result["status"] == "rejected"
    assert "PERTURBATION_TOP1_STABILITY_RATE_BELOW_THRESHOLD" in result["blockers"]


def test_checkpoint_selection_uses_validation_only(tmp_path: Path) -> None:
    variant = tmp_path / "cg_hhgm"
    variant.mkdir()
    rows = (
        (17, 0.80, 0.75, 0.99),
        (29, 0.90, 0.20, 0.20),
    )
    for seed, validation, decision_f1, test in rows:
        (variant / f"seed_{seed}_history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 1,
                        "validation_ndcg@10": validation,
                        "validation_raw_decision_macro_f1": decision_f1,
                        "validation_joint_score": _validation_joint_score(
                            validation, decision_f1
                        ),
                    }
                ]
            ),
            encoding="utf-8",
        )
        (variant / f"seed_{seed}.pt").write_bytes(str(seed).encode())
        (variant / f"seed_{seed}_metrics.json").write_text(
            json.dumps({"aggregate": {"ndcg@10": test}}),
            encoding="utf-8",
        )

    selection = select_checkpoint(
        variant_directory=variant,
        output_directory=tmp_path / "selection",
    )

    assert selection["selected"]["seed"] == 17
    assert selection["test_metrics_used_for_selection"] is False
