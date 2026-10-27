"""Reviewer checks cover scientific qualification and protected input isolation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import blueprint_reviewer_diagnostics as diagnostics
import blueprint_reviewer_rf_ablation as ablation


class ReviewerDiagnosticsTests(unittest.TestCase):
    def test_poor_sufentanil_slope_is_not_strictly_qualified(self):
        raw=dict(selected_tail=dict(n=8,adjusted_r_squared=.492153),terminal_span_in_half_lives=1.7145)
        result=diagnostics.strict_quality(raw)
        self.assertFalse(result['terminal_half_life_qualified'])
        self.assertIn('adjusted_r_squared_below_0.98_or_missing',result['terminal_reasons'])
        self.assertFalse(result['infinity_qualified'])

    def test_good_half_life_does_not_invent_unobserved_initial_exposure(self):
        raw=dict(selected_tail=dict(n=4,adjusted_r_squared=.999),terminal_span_in_half_lives=2.)
        result=diagnostics.strict_quality(raw)
        self.assertTrue(result['terminal_half_life_qualified'])
        self.assertFalse(result['infinity_qualified'])
        self.assertFalse(result['moment_qualified'])

    def test_dense_gate_uses_four_original_points(self):
        raw=dict(terminal_fit=dict(fit_points=3,adjusted_r_squared=.999,span_half_lives=2.,
            auc_extrapolated_fraction=.01,aumc_extrapolated_fraction=.01))
        self.assertFalse(diagnostics.strict_quality(raw,dense=True)['terminal_half_life_qualified'])

    def test_ablation_preserves_independent_evidence_and_input(self):
        request=dict(evidence=[dict(parameter='total_clearance',basis='measured',value=8.,unit='ml/min/kg'),
            dict(parameter='fraction_unbound',basis='predicted',value=.02,unit='fraction',
                 prediction_method={'name':'random_forest'},source={},notes='')])
        before=deepcopy(request)
        changed,receipt=ablation.modified_request(request,'median_both',
            {'human_CL_mL_min_kg':3.,'human_fup':.3},{'path':'x','sha256':'0'*64})
        self.assertEqual(request,before)
        self.assertEqual(changed['evidence'][0],request['evidence'][0])
        self.assertEqual(changed['evidence'][1]['value'],.3)
        self.assertEqual([r['parameter'] for r in receipt],['fraction_unbound'])

    def test_other_prediction_methods_not_silently_replaced(self):
        request=dict(evidence=[dict(parameter='fraction_unbound',basis='predicted',value=.02,
            prediction_method={'name':'independent_external_model'})])
        changed,receipt=ablation.modified_request(request,'median_fu',{'human_fup':.3},{})
        self.assertEqual(changed,request);self.assertEqual(receipt,[])


if __name__=='__main__':unittest.main()
