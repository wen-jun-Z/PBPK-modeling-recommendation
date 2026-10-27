from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pbpk_kb.model_assets.models import (
    ModelValidationScenario,
    ModelValidationScope,
)


VALIDATION_SCOPE_BUILDER_VERSION = "pbpk-validation-scope-builder-1.0.0"


def _list(value: Any) -> list[str]:
    return sorted(
        {str(item) for item in (value or ()) if str(item).strip()},
        key=str.casefold,
    )


def _dose_from_name(name: str) -> tuple[float | None, str | None]:
    match = re.search(
        r"(?<![A-Za-z0-9])(?:po|iv)?\s*(\d+(?:\.\d+)?)\s*"
        r"(mg|mcg|ug|µg|g)(?![A-Za-z])",
        name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    unit = match.group(2).casefold().replace("µ", "u")
    return float(match.group(1)), unit


def _frequency_from_name(name: str) -> tuple[str | None, str]:
    normalized = " ".join(name.casefold().split())
    for token, frequency in (
        (" qd", "once_daily"),
        (" once daily", "once_daily"),
        (" bid", "twice_daily"),
        (" twice daily", "twice_daily"),
        (" tid", "three_times_daily"),
        (" three times daily", "three_times_daily"),
    ):
        if token in f" {normalized}":
            return frequency, "multiple"
    if "single dose" in normalized:
        return "single_dose", "single"
    return None, "unknown"


def _outputs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    csv_results = row.get("csv_results") or ()
    if csv_results:
        return [
            output
            for result in csv_results
            for output in (result.get("outputs") or ())
            if isinstance(output, dict)
        ]
    metadata = row.get("metadata_json") or {}
    return [
        output
        for output in (metadata.get("outputs") or ())
        if isinstance(output, dict)
    ]


def _evaluated_endpoints(row: Mapping[str, Any]) -> list[str]:
    dataset = str(row.get("validation_dataset_name") or "").casefold()
    output_text = " ".join(
        str(output.get("name") or "").casefold()
        for output in _outputs(row)
    )
    endpoints: set[str] = set()
    if any(
        token in dataset
        for token in ("plasma", "blood", "serum", "concentration")
    ) and any(
        token in output_text
        for token in ("plasma", "blood", "concentration")
    ):
        endpoints.add("concentration_time_profile")
    validation_metrics = row.get("validation_metrics") or {}
    if validation_metrics.get("auc0_last_ratio") is not None:
        endpoints.add("AUC")
    if validation_metrics.get("cmax_ratio") is not None:
        endpoints.add("Cmax")
    return sorted(endpoints)


def build_model_validation_scope(
    simulations: Iterable[Mapping[str, Any]],
) -> ModelValidationScope | None:
    scenarios: list[ModelValidationScenario] = []
    for row in simulations:
        if str(row.get("protocol_role") or "") != "model_verification":
            continue
        dataset_name = str(row.get("validation_dataset_name") or "").strip()
        if not dataset_name:
            continue
        name = str(
            row.get("simulation_name") or row.get("name") or ""
        ).strip()
        dose, dose_unit = _dose_from_name(name)
        frequency, mode = _frequency_from_name(name)
        pkml = row.get("pkml_source") or {}
        source_location = str(
            row.get("source_location")
            or pkml.get("source_location")
            or ""
        ).strip()
        scenarios.append(
            ModelValidationScenario(
                simulation_id=str(
                    row.get("external_simulation_id")
                    or row.get("simulation_id")
                    or ""
                ),
                simulation_name=name,
                dataset_name=dataset_name,
                source_location=source_location,
                species=_list(row.get("species")),
                populations=_list(row.get("populations")),
                routes=_list(row.get("routes")),
                formulations=_list(row.get("formulations")),
                evaluated_endpoints=_evaluated_endpoints(row),
                dose=dose,
                dose_unit=dose_unit,
                frequency=frequency,
                single_or_multiple=mode,
            )
        )
    if not scenarios:
        return None
    scenarios.sort(
        key=lambda item: (
            item.simulation_name.casefold(),
            item.simulation_id,
        )
    )

    def aggregate(field: str) -> list[str]:
        return sorted(
            {
                value
                for scenario in scenarios
                for value in getattr(scenario, field)
            },
            key=str.casefold,
        )

    return ModelValidationScope(
        species=aggregate("species"),
        populations=aggregate("populations"),
        routes=aggregate("routes"),
        formulations=aggregate("formulations"),
        evaluated_endpoints=aggregate("evaluated_endpoints"),
        dataset_names=sorted(
            {scenario.dataset_name for scenario in scenarios},
            key=str.casefold,
        ),
        scenarios=scenarios,
    )
