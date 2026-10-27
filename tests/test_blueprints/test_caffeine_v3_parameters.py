from __future__ import annotations

import math

from pbpk_kb.blueprints.caffeine_v3_parameters import (
    AbsorptionSetProtocol,
    _permeability_grid,
)


def test_permeability_grid_is_complete_deterministic_and_log_spaced() -> None:
    grid = _permeability_grid()

    assert len(grid) == 71
    assert grid[0].candidate_id == 1
    assert grid[-1].candidate_id == 71
    assert math.isclose(grid[0].value_dm_per_min, 1e-9)
    assert math.isclose(grid[-1].value_dm_per_min, 1e-2)
    assert [item.log10_dm_per_min for item in grid[:3]] == [
        -9.0,
        -8.9,
        -8.8,
    ]
    assert all(
        math.isclose(
            right.log10_dm_per_min - left.log10_dm_per_min,
            0.1,
            abs_tol=1e-12,
        )
        for left, right in zip(grid, grid[1:])
    )


def test_absorption_protocol_preserves_set_identification_semantics() -> None:
    protocol = AbsorptionSetProtocol()

    assert protocol.identification_type == "set_identified_nuisance_parameter"
    assert protocol.reference_selection == "minimum_admissible_candidate"
    assert protocol.upper_bound_interpretation == (
        "right_censored_numerical_guard"
    )
    assert protocol.bioavailability_lower == 0.99
    assert protocol.tmax_lower_min == 11.5
