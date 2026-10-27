"""Score a pre-existing numeric tail gate; never feed outcomes into the gate.

The signal is the dense prediction's frozen numerical-validity status. Its
association with the clinical-protocol (P*) fold error is exploratory: those
estimators use different windows. It is not a clinical acceptance classifier.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from pbpk_kb.blueprints.source_workflow import inside, sha256

VERSION = 'blueprint-tail-gate-evaluation-1'
TAIL_ENDPOINTS = ('terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
CONFIG = {
    'gate': "locked dense endpoint status != 'completed'",
    'error_signal': 'matched clinical-protocol fold error > 2',
    'error_threshold': 2., 'threshold_is_clinical_acceptance': False,
    'methods': ['always_three', 'adaptive_axes'], 'scenario': 'calibration',
    'missing_reference': 'not scored; never treat as zero error or a rejection',
    'warning_source_uses_target_observations': False,
    'limitation': 'Dense numerical-tail quality and matched clinical NCA differ; this is an association audit, not calibrated clinical confidence.',
}


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def score_tail_rows(rows):
    """Count both warning errors and missed errors, with all denominators."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['sampling'])].append(row)
    output = {}
    for (method, sampling), records in sorted(groups.items()):
        eligible = [r for r in records if r['reference_available'] and _positive(r['matched_fold_error'])]
        large = [r for r in eligible if r['matched_fold_error'] > CONFIG['error_threshold']]
        flagged = [r for r in eligible if r['warning']]
        passed = [r for r in eligible if not r['warning']]
        hit = sum(r['warning'] for r in large)
        false_warning = sum(r['matched_fold_error'] <= CONFIG['error_threshold'] for r in flagged)
        small = len(eligible) - len(large)
        output[method + '__' + sampling] = {
            'requested_tail_endpoints': len(records),
            'reference_available': sum(r['reference_available'] for r in records),
            'matched_error_available': len(eligible),
            'warnings_with_reference_and_matched_error': len(flagged),
            'passed_with_reference_and_matched_error': len(passed),
            'large_matched_errors': len(large), 'flagged_large_errors': hit,
            'missed_large_errors': len(large)-hit, 'false_warnings': false_warning,
            'large_error_detection_rate': hit/len(large) if large else None,
            'false_warning_rate': false_warning/small if small else None,
            'matched_error_above_two_among_passed': (
                sum(r['matched_fold_error'] > CONFIG['error_threshold'] for r in passed)/len(passed)
                if passed else None),
            'coverage_of_scored_references': len(passed)/len(eligible) if eligible else None,
        }
    return output


def evaluate_tail_gate(records, state_computations, project_root):
    root = Path(project_root).resolve(strict=True)
    rows, pins = [], {}
    for record in records:
        if record['scenario'] != CONFIG['scenario'] or record['method'] not in CONFIG['methods']:
            continue
        computation = state_computations.get(record.get('computation_id'), {})
        prediction = computation.get('prediction')
        if computation.get('status') != 'predicted' or not prediction:
            # Scientific failures remain in the experiment's main coverage
            # results; do not invent a tail warning for a missing simulation.
            continue
        path = inside(root/prediction['path'], root, exists=True)
        if sha256(path) != prediction['sha256']:
            raise ValueError('Locked prediction changed before tail-gate audit')
        receipt_ref, dense_ref = computation['completion'], computation['dense_endpoints']
        receipt_path = inside(root/receipt_ref['path'], root, exists=True)
        dense_path = inside(root/dense_ref['path'], root, exists=True)
        if (receipt_path != path.parent/'completion.json' or dense_path != path.parent/'endpoints.json'
                or sha256(receipt_path) != receipt_ref['sha256'] or sha256(dense_path) != dense_ref['sha256']):
            raise ValueError('Endpoint gate or receipt differs from the pre-scoring lock')
        receipt = json.loads(receipt_path.read_text())
        relative = str(dense_path.relative_to(root))
        expected = receipt['artifact_pins'][relative]
        if sha256(dense_path) != expected:
            raise ValueError('Locked endpoint gate changed before scoring')
        pins[relative] = expected
        pins[str(receipt_path.relative_to(root))] = sha256(receipt_path)
        dense = json.loads(dense_path.read_text())
        if (dense.get('endpoint_estimator_uses_observations') is not False
                or dense.get('comparison_observations_used') is not False):
            raise ValueError('Gate is not independent of target outcomes')
        for endpoint in TAIL_ENDPOINTS:
            signal = dense['endpoints'][endpoint]
            scored = record.get('endpoints', {}).get(endpoint, {})
            rows.append({
                'case_id': record['case_id'], 'drug': record['drug'],
                'method': record['method'], 'sampling': record['sampling'], 'endpoint': endpoint,
                'dense_gate_status': signal['status'], 'warning': signal['status'] != 'completed',
                'reference_available': _positive(scored.get('reference')),
                'matched_fold_error': scored.get('error') if scored.get('error_metric') == 'fold_error' else None,
                'gate_source': {'path': relative, 'sha256': expected},
            })
    return {'version': VERSION, 'config': CONFIG, 'records': rows,
            'by_method_sampling': score_tail_rows(rows), 'source_pins': pins}
