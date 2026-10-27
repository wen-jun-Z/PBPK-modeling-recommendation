"""Cache compatibility must protect both acquired observations and fit rules."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import repair_sampling_policy_candidates as candidate


class CandidateRepairTests(unittest.TestCase):
    def setUp(self):
        self.source = {"path": "original.json", "sha256": "a" * 64}
        self.training = {"path": "training.json", "sha256": "b" * 64}
        self.audits = {f: {"repair_required": True, "version": candidate.repair.VERSION}
                       for f in candidate.base.FORMULAS}
        self.unit = dict(status="predicted", original_source=self.source,
                         training=self.training, feasibility=self.audits)
        self.fits = [dict(formula=f, fit=dict(starts=[{}] * 16, call_limit=16384,
            target_observations_used=False, no_warm_start=True, version=candidate.repair.VERSION,
            partition_feasibility=self.audits[f])) for f in candidate.base.FORMULAS]

    def check(self, unit=None, fits=None):
        return candidate.reusable_repair(unit or self.unit, self.source, self.training,
                                         self.audits, fits or self.fits)

    def test_exact_rule_and_observations_can_be_reused(self):
        self.assertTrue(self.check())

    def test_other_acquired_point_cannot_be_reused(self):
        unit = deepcopy(self.unit)
        unit["training"]["sha256"] = "c" * 64
        self.assertFalse(self.check(unit=unit))

    def test_different_repair_rule_is_rejected(self):
        fits = deepcopy(self.fits)
        fits[0]["fit"]["partition_feasibility"]["version"] = "another-rule"
        self.assertFalse(self.check(fits=fits))

    def test_target_fitted_or_warm_started_result_is_rejected(self):
        for field in ("target_observations_used", "no_warm_start"):
            fits = deepcopy(self.fits)
            fits[0]["fit"][field] = not fits[0]["fit"][field]
            self.assertFalse(self.check(fits=fits))


if __name__ == "__main__":
    unittest.main()
