from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pbpk_kb.model_runtime.caffeine_study_runtime import (
    _individual_demographics,
    _instantiated_renal_processes,
    _log_issues,
)


def test_log_audit_accepts_zero_error_summary(tmp_path: Path) -> None:
    path = tmp_path / "export.log"
    path.write_text("Export complete\n0 warnings\n0 errors\n", encoding="utf-8")

    assert _log_issues(path) == []


def test_log_audit_reports_real_warning_and_error(tmp_path: Path) -> None:
    path = tmp_path / "export.log"
    path.write_text(
        "warning: route changed\nERROR: simulation failed\n",
        encoding="utf-8",
    )

    issues = _log_issues(path)

    assert len(issues) == 2
    assert "route changed" in issues[0]
    assert "simulation failed" in issues[1]


def test_individual_demographics_are_read_from_pkml_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "simulation.pkml"
    path.write_text(
        """<SimulationTransfer><ExtendedProperties>
        <StringExtendedProperty name="Age" value="30.00 year(s)" />
        <StringExtendedProperty name="Height" value="176.00 cm" />
        <StringExtendedProperty name="Weight" value="73.00 kg" />
        </ExtendedProperties></SimulationTransfer>""",
        encoding="utf-8",
    )

    result = _individual_demographics(path)

    assert result["age_years"] == 30.0
    assert result["height_m"] == 1.76
    assert result["weight_kg"] == 73.0
    assert abs(result["bmi_kg_per_m2"] - 23.5666) < 1e-4


def test_renal_audit_uses_instantiated_transport_not_builder() -> None:
    name = "Renal Clearances-Birkett 1991 Table 1 rederived PBPK-KB-v1-Caffeine"
    processes = [
        SimpleNamespace(
            element_type="TransportBuilder",
            process_type="renal_excretion",
            name=name,
        ),
        SimpleNamespace(
            element_type="Transport",
            process_type="renal_excretion",
            name=name,
        ),
    ]

    selected = _instantiated_renal_processes(processes)

    assert len(selected) == 1
    assert selected[0].element_type == "Transport"
