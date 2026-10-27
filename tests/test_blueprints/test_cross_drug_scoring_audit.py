"""Whole-run label-access gate, independent of clinical scoring fixtures."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import score_cross_drug_blueprint as scoring


class LabelsReached(RuntimeError):
    pass


@pytest.mark.parametrize('late_fault', [None, 'unfinished', 'missing_unit', 'bad_units', 'bundle_hash'])
def test_every_prediction_is_validated_before_first_label_access(monkeypatch, late_fault):
    cases = [{'case_id': str(i), 'compound_id': 'compound:'+str(i), 'drug': str(i),
              'cohort': 'original', 'cluster_id': str(i), 'reference_kind': 'original_reported'}
             for i in range(20)]
    protocol = {'cases': cases, 'methods': list(scoring.METHODS),
                'conditions': list(scoring.CONDITIONS), 'source_pins': {}, 'paper_guard': {}}
    state = {'protocol': {'path': 'protocol.json', 'sha256': 'protocol-hash'},
             'all_predictions_locked_before_scoring': True,
             'prediction_bundles': {c['case_id']: {'path': 'bundle-'+c['case_id'], 'sha256': 'bundle-hash'}
                                    for c in cases}}
    bundles = {}
    for case in cases:
        entries = []
        for method in scoring.METHODS:
            for condition in scoring.CONDITIONS:
                entries.append({'method': method, 'condition': condition, 'status': 'predicted',
                    'request': {'compound_id': case['compound_id'],
                                'dosing': {'route': 'intravenous', 'start_h': 0},
                                'evidence': [{'parameter': 'molecular_weight', 'unit': 'g/mol', 'value': 100.}]},
                    'curve': {'unit': 'umol/l', 'time_unit': 'min', 'time_min': [0., 30., 60.],
                              'concentration': [0., 2., 1.]},
                    'endpoints': {'comparison_observations_used': False,
                                  'endpoint_estimator_uses_observations': False}})
        bundles['bundle-'+case['case_id']] = {'case_id': case['case_id'], 'entries': entries}
    last = bundles['bundle-19']['entries']
    if late_fault == 'unfinished':
        last[-1]['status'] = 'running'
    elif late_fault == 'missing_unit':
        last.pop()
    elif late_fault == 'bad_units':
        last[-1]['curve']['unit'] = 'mg/l'

    bundle_reads, label_reads = [], []

    class MemoryStore:
        def __init__(self, root):
            self.pins = {}

        def read(self, name):
            return deepcopy(protocol if Path(name).name == 'protocol.json' else state)

        def ref(self, ref):
            if ref['path'] == 'protocol.json':
                return deepcopy(protocol)
            bundle_reads.append(ref['path'])
            if late_fault == 'bundle_hash' and ref['path'] == 'bundle-19':
                raise ValueError('Source lock mismatch: bundle-19')
            return deepcopy(bundles[ref['path']])

    def first_label(*args):
        label_reads.append(True)
        raise LabelsReached('All prediction units passed; scorer can now open labels')

    monkeypatch.setattr(scoring.legacy, 'Store', MemoryStore)
    monkeypatch.setattr(scoring.legacy, '_case_reference', first_label)
    expected = LabelsReached if late_fault is None else ValueError
    with pytest.raises(expected):
        scoring.score(ROOT / 'scripts', project_root=ROOT)
    assert len(bundle_reads) == 20
    assert bool(label_reads) == (late_fault is None)
