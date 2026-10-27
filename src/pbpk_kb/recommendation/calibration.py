from __future__ import annotations

import math
from typing import Any


CALIBRATION_POLICY_VERSION = "pbpk-ranking-calibration-policy-1.0.0"


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Invalid binomial counts.")
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def disabled_calibration_manifest(
    *,
    request_count: int,
    grade_2_count: int,
    grade_3_count: int,
) -> dict[str, Any]:
    training_eligible = (
        request_count >= 30
        and grade_2_count > 0
        and grade_3_count > 0
    )
    return {
        "calibration_policy_version": CALIBRATION_POLICY_VERSION,
        "training_eligible": training_eligible,
        "auto_reuse_enabled": False,
        "reuse_threshold": None,
        "adapt_threshold": None,
        "calibration_method": None,
        "label_counts": {
            "grade_2_adapt_positive": grade_2_count,
            "grade_3_reuse_positive": grade_3_count,
        },
        "request_count": request_count,
        "reason_codes": (
            []
            if training_eligible
            else ["TRAINING_LABELS_INSUFFICIENT"]
        ),
        "limitations": [
            "No reuse/adapt threshold is estimated without real "
            "run-derived grade 2 and grade 3 labels.",
            "Rule compatibility values remain diagnostic and are not "
            "calibrated reuse probabilities.",
        ],
    }
