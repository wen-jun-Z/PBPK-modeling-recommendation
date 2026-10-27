"""Training ownership, common references, numerical ties, and scoring locks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope='module')
def runner():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    import run_sampling_stability
    return run_sampling_stability


def inputs():
    cases = [dict(case_id=f'd{i}', cluster_id='shared' if i < 2 else f'study{i}') for i in range(30)]
    original = {c['case_id'] for c in cases[:10]}
    features, records = {}, []
    for case in cases:
        cid = case['case_id']; key = cid + '__late_four'
        features[key] = dict(case_id=cid, candidate_ids=[2, 5], goals={
            'cmax': dict(status='selected', X=[[0.] * 8, [1.] * 8])})
        for candidate, error, source_error in [('baseline', 4., 1.), (key+'__point_02', 2., 100.), (key+'__point_05', 1., 1000.)]:
            records.append(dict(case_id=cid, design_key=key, candidate=candidate,
                endpoints={'cmax': dict(error=error, reference_kind='observed_curve_nca')},
                source_reported_sensitivity={'cmax': dict(error=source_error)}))
    return cases, original, features, records


def test_training_rows_are_only_original_ten_and_use_common_matched_errors(runner):
    rows = runner.label_rows(*inputs(), 'cmax')
    assert len(rows['X']) == 20
    assert set(rows['drug_ids']) == {f'd{i}' for i in range(10)}
    assert rows['y'] == pytest.approx([math.log(2.), math.log(4.)] * 10)
    assert {cluster for drug, cluster in zip(rows['drug_ids'], rows['cluster_ids']) if drug in {'d0', 'd1'}} == {'shared'}


def test_new_twenty_outcomes_not_read_and_reported_sensitivity_not_used(runner):
    data = inputs(); before = runner.label_rows(*data, 'cmax')
    cases, original, features, records = deepcopy(data)
    for record in records:
        if record['case_id'] not in original:
            # Numeric outcomes and even endpoint dictionaries may be absent.
            record.pop('endpoints')
            record.pop('source_reported_sensitivity')
        else:
            record['source_reported_sensitivity']['cmax']['error'] = float('nan')
    for cid in {c['case_id'] for c in cases} - original:
        features[cid+'__late_four']['goals'] = None
    assert runner.label_rows(cases, original, features, records, 'cmax') == before


def test_whitelist_size_enforced_and_missing_reference_not_filled(runner):
    cases, original, features, records = inputs()
    with pytest.raises(ValueError, match='original ten'):
        runner.label_rows(cases, original | {'d10'}, features, records, 'cmax')
    records[1]['endpoints']['cmax']['error'] = None
    rows = runner.label_rows(cases, original, features, records, 'cmax')
    assert 'd0' not in rows['drug_ids']
    assert len(rows['X']) == 18


def test_montelukast_scale_roundoff_is_tie_not_regression(runner):
    assert runner.counts({'a': 1., 'montelukast': 1.+2e-14, 'c': 1.-3e-13,
                          'better': .95, 'worse': 1.05}) == dict(improved=1, worse=1, tied=3)


@pytest.mark.parametrize('state_status,units,jobs', [
    ('prepared', {}, {}), ('fitting', {'u': {}}, {'u': {}}),
    ('predictions_locked', {}, {'u': {}}), ('complete', {'unexpected': {}}, {}),
])
def test_scoring_requires_prediction_lock_and_exact_complete_job_set(runner, monkeypatch, state_status, units, jobs):
    state = dict(status=state_status, units=units, protocol='protocol-reference')
    reads = []
    def read(path):
        reads.append(path)
        if path == runner.RUN/'state.json': return state
        raise AssertionError('Clinical or prior result read before lock check')
    def checked(reference):
        assert reference == 'protocol-reference', 'A choice/result was read before lock check'
        return dict(jobs=jobs)
    monkeypatch.setattr(runner, 'read', read)
    monkeypatch.setattr(runner, 'checked', checked)
    monkeypatch.setattr(runner.Guard, '__call__', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.data, 'score_case', lambda *args, **kwargs: pytest.fail('Unfrozen target scoring'))
    with pytest.raises(ValueError, match='lock before scoring'): runner.report()
    assert reads == [runner.RUN/'state.json']


def test_frozen_sources_remain_hash_bound_after_pure_label_and_count_operations(runner):
    protocol = json.loads((runner.OLD/'protocol.json').read_text())
    paths = [
        'scripts/run_sampling_expansion.py',
        'src/pbpk_kb/blueprint_policy/learned_sampling_design.py',
        'src/pbpk_kb/blueprint_policy/distribution_calibration_optimizer.py',
    ]
    def digest(relative): return hashlib.sha256((runner.ROOT/relative).read_bytes()).hexdigest()
    before = {p: digest(p) for p in paths}
    assert all(before[p] == protocol['source_pins'][p] for p in paths)
    runner.label_rows(*inputs(), 'cmax'); runner.counts({'drug': 1.})
    assert {p: digest(p) for p in paths} == before
