"""Recompute the blueprint experiment design and conditional sample-size analysis.

This is a planning artifact, not a benchmark launcher or a source of observed
performance. It reads no held-out concentrations and runs no PBPK solver/LLM.
The authoritative rationale and eligibility rules are in docs/研究计划.md §9.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

import scipy
from scipy.stats import nct, t


ROOT = Path(__file__).resolve().parents[1]


def paired_t_power(n: int, mean_difference: float, sd_difference: float,
                   alpha: float = 0.05) -> float:
    """Conditional power for independent normal drug-level paired differences."""
    if n < 2 or sd_difference <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid independent-drug power inputs")
    critical = t.ppf(1 - alpha / 2, n - 1)
    shift = math.sqrt(n) * mean_difference / sd_difference
    return float(nct.sf(critical, n - 1, shift) + nct.cdf(-critical, n - 1, shift))


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_plan() -> dict:
    hold_path = ROOT / 'runtime/deployment/blueprint_experiment_hold.json'
    if hold_path.is_file() and json.loads(hold_path.read_text()).get('development_status') == 'algorithm_redesign':
        raise ValueError('旧蓝图实验方案已退役；当前方法与验证安排见 docs/推荐策略.md 和 docs/研究计划.md 第 9 节。')
    development_drugs, evaluation_drugs, requests_per_drug = 12, 60, 2
    requests = evaluation_drugs * requests_per_drug
    mechanism_requests, synthetic_tasks, seeds = 32, 120, [17, 43, 101]
    methods = [
        ("M0", "固定人体底座及药物参数，统一拟合", 1),
        ("M1", "需求规则装配，统一拟合", 1),
        ("M2", "同一 Qwen 单次提案，统一拟合", 3),
        ("M3", "合法空间随机搜索", 3),
        ("M4", "随机森林代理辅助搜索", 3),
        ("M5", "先结构后参数，两阶段选择", 3),
        ("M6", "需求与证据引导的结构和参数子集联合搜索", 3),
    ]
    counts = {
        "main_comparisons": requests * sum(item[2] for item in methods),
        "ablations_A1_A2_A3": requests * 3 * len(seeds),
        "matched_parameter_count_M5_M6": mechanism_requests * 2 * len(seeds),
        "robustness_M4_M6": requests * 4 * 2 * len(seeds),
        "synthetic_M1_M5_M6": synthetic_tasks * (1 + 2 * len(seeds)),
    }
    power_rows = []
    for sd in (0.20, 0.25, 0.30, 0.35):
        n_min = next(n for n in range(2, 1001) if paired_t_power(n, 0.10, sd) >= 0.80)
        power_rows.append({
            "assumed_sd_drug_level_difference": sd,
            "minimum_independent_drugs_for_80_percent_power": n_min,
            "power_at_60_drugs": paired_t_power(60, 0.10, sd),
            "power_at_84_drugs": paired_t_power(84, 0.10, sd),
        })
    baseline_result = ROOT / (
        "experiments/blueprint_design/development_20260918/"
        "caffeine_blanchard_po/numerical_feedback/result.json")
    current = json.loads(baseline_result.read_text(encoding="utf-8"))
    if (current["purpose"] != "historical_development_reconstruction"
            or current["independent_validation_performed"] is not False
            or current["structure_search_performed"] is not False):
        raise ValueError("Development status changed; review the experiment readiness statements")
    audited_sources = [
        "src/pbpk_kb/blueprints/design.py",
        "src/pbpk_kb/blueprints/human_model_snapshot.py",
        "src/pbpk_kb/blueprints/design_contracts.py",
        "scripts/run_blueprint_development.py",
        "docs/研究计划.md",
        "docs/推荐策略.md",
    ]
    return {
        "artifact_kind": "experiment_design_and_conditional_power",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_status": "draft_not_frozen",
        "benchmark_results_generated": False,
        "solver_llm_or_gpu_launched": False,
        "scope": "adult_human_small_molecule_requirement_conditioned_pbpk_design",
        "cohort_targets_not_observed_counts": {
            "development_drugs": development_drugs,
            "evaluation_drugs": evaluation_drugs,
            "requests_per_evaluation_drug": requests_per_drug,
            "main_requests": requests,
            "external_test_study_groups_minimum_per_drug": 3,
            "calibration_and_selection_study_groups_minimum_per_drug": 2,
            "external_test_drug_study_memberships_minimum": evaluation_drugs * 3,
            "external_test_unique_study_groups_minimum": 180,
            "all_roles_drug_study_memberships_minimum": evaluation_drugs * 5,
            "study_count_note": "Shared cross-drug study groups are deduplicated; memberships are not necessarily globally independent studies.",
            "strict_new_evaluation_drugs_minimum": 20,
            "additional_qualified_drug_packages_relative_to_current_30": 42,
            "mechanism_evaluation_drugs_minimum": 16,
            "mechanism_requests": mechanism_requests,
            "robustness_variants_per_request": 4,
            "synthetic_design_groups": 20,
            "synthetic_tasks": synthetic_tasks,
        },
        "methods": [{"id": method, "description": description, "runs_per_request": repetitions}
                    for method, description, repetitions in methods],
        "ablations": {"A1": "remove_requirement_guidance", "A2": "remove_evidence_guidance",
                      "A3": "remove_identifiability_prescreen_and_feedback",
                      "M5_also_serves_as": "remove_joint_structure_parameter_search"},
        "controls": {
            "matched_parameter_count": "Prespecified mechanism tasks; M5 and M6 share feasible parameter cardinality and common fitter. Failure to meet the constraint remains in the denominator.",
            "enumeration_reference": "Enumerate the entire native non-equivalent legal space only when <=24 states and within the same budget, with Dsel selection and no Dtest oracle.",
            "enumeration_additional_run_count": None,
            "enumeration_included_in_mandatory_total": False,
        },
        "seeds_for_stochastic_methods": seeds,
        "planned_design_run_counts": counts,
        "planned_mandatory_design_runs_total": sum(counts.values()),
        "budget": {
            "maximum_native_scenario_equivalents_per_design": 5000,
            "mandatory_design_worst_case_scenario_equivalents": 5000 * sum(counts.values()),
            "diagnostic_reserved_fraction": 0.20,
            "maximum_unique_candidate_states": 24,
            "maximum_free_parameters": 6,
            "maximum_starts_per_candidate": 3,
            "maximum_evaluations_per_start": 50,
            "search_checkpoints": [500, 2000, 5000],
            "budget_note": "Counts logical calls including failures, multi-scenario objectives, Dsel and diagnostics; cached physical savings are separate. Caps are not required expenditure.",
            "excluded_costs": ["development_tuning", "final_Dtest_prediction",
                               "independent_uncertainty_evaluation", "conditional_enumeration_reference"],
            "wall_time_prediction": None,
        },
        "primary_analysis": {
            "comparison": "M6_minus_M5",
            "endpoint": "independent_study_plasma_natural_log_RMSE",
            "benefit_direction": "negative_difference",
            "unit": "drug_level_equal_weight_paired_difference",
            "aggregation": "seed_errors_then_study_equal_weight_then_drug_equal_weight; deduplicate studies shared by requests",
            "secondary_comparisons": "M6 versus M0/M1/M2/M3/M4 with Holm correction",
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": 20260918,
            "resampling_unit": "drug_or_connected_drug_shared_study_component; retain all nested observations",
            "confidence_interval": "percentile_0.025_0.975",
            "two_sided_p": "(1+count(abs(delta_boot-delta_observed)>=abs(delta_observed)))/(B+1)",
            "calibration_required": "Simulate coverage/type-I error using development-based clustering/fallback before freeze",
            "ablation_comparison_family": "M6_vs_A1_A2_A3_exploratory_Holm",
            "failures": "prelocked_engineering_fallback; never switch based on Dtest error",
            "interpretation": "Report effect and interval with native-success coverage; no clinical acceptability or novelty is implied by significance alone.",
        },
        "conditional_power_not_empirical_evidence": {
            "method": "two_sided_noncentral_t_independent_normal_drug_differences",
            "alpha": 0.05,
            "target_power": 0.80,
            "assumed_mean_log_rmse_improvement": 0.10,
            "rows": power_rows,
            "limitations": [
                "No study-level or clinical response values were read for this calculation.",
                "SD inputs are hypothetical drug-level paired SDs, not values estimated from the caffeine calibration.",
                "This is not the power of the full clustered bootstrap/fallback analysis.",
                "Final sample size requires development-based simulation of study correlation, heterogeneity and failure fallback.",
                "N=84 is not sufficient for every assumed SD; do not add seeds or dose arms as independent drugs.",
            ],
        },
        "readiness_review_20260918": {
            "completed_development_artifact": str(baseline_result.relative_to(ROOT)),
            "completed_development_artifact_sha256": checksum(baseline_result),
            "executed_generic_native_structure_edit_actions": 0,
            "current_fixed_structure_fitted_parameters": 2,
            "full_M0_to_M6_benchmark_ready": False,
            "blocking_implementation_work": [
                "native_module_edits_with_semantic_difference_and_mass_balance_checks",
                "joint_state_search_with_identifiability_feedback_and_refitting",
                "shared_multiscenario_calibrator_and_Dsel_selector",
                "genuine_method_and_ablation_implementations",
                "study_group_and_parameter_lineage_audit_then_locked_Dtest_evaluator",
            ],
            "code_review_source_sha256": {name: checksum(ROOT / name) for name in audited_sources},
            "review_note": "Manual code audit at the recorded hashes, not an automatically refreshed readiness certificate.",
        },
        "literature": [
            "https://link.springer.com/article/10.1007/s00204-024-03764-9",
            "https://www.nature.com/articles/s43856-025-01054-8",
            "https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11904618/",
        ],
        "reproducibility": {"script_sha256": checksum(Path(__file__)), "scipy_version": scipy.__version__},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Planning output must remain inside this project")
    if output.exists():
        raise FileExistsError("Preserve the existing plan; use a new output path")
    plan = build_plan()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": plan["protocol_status"], "output": str(output.relative_to(ROOT)),
                      "planned_mandatory_design_runs": plan["planned_mandatory_design_runs_total"],
                      "experiments_launched": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
