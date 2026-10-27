"""Resume contracts use synthetic process registrations, never real outcomes."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('furo_external_resume', ROOT/'scripts/run_furosemide_external_validation.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def previous(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'checksum', lambda path: 'synthetic-sha')
    directory = tmp_path/'runtime/deployment'
    directory.mkdir(parents=True)
    registration = {'pid': 100, 'start_ticks': '1'}
    status = {'pid': 100, 'status': 'stopped', 'steps': [
        {'pid': 200, 'start_ticks': '2', 'output_bytes': 0}]}
    (directory/'blueprint_furosemide_validation_pipeline_20260920_seed_fix_v2.json').write_text(json.dumps(registration))
    target = directory/'blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json'
    target.write_text(json.dumps(status))
    return target, status


@pytest.mark.parametrize('live_pid', [None, 100, 200])
def test_old_method_or_predictor_must_exit_before_resume(tmp_path, monkeypatch, live_pid):
    previous(tmp_path, monkeypatch)
    class Scope:
        def state(self, pid, ticks):
            assert (pid, ticks) in {(100, '1'), (200, '2')}
            return {'pid': pid} if pid == live_pid else None
    if live_pid:
        with pytest.raises(RuntimeError):
            pipeline.check_previous_attempt(Scope())
    else:
        assert pipeline.check_previous_attempt(Scope())['previous_prediction_output_bytes'] == 0


@pytest.mark.parametrize('damage', ['status', 'identity'])
def test_changed_or_active_previous_registration_rejected(tmp_path, monkeypatch, damage):
    path, status = previous(tmp_path, monkeypatch)
    status['status' if damage == 'status' else 'pid'] = 'running' if damage == 'status' else 999
    path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError):
        pipeline.check_previous_attempt(None)


@pytest.mark.parametrize('key,value', [
    (None, None), ('method_jobs_expected', 56), ('method_rows_scored', 56),
    ('strict_main_eligible', True), ('strict_main_eligible', 0),
    ('independent_drugs', 57), ('metrics_exposed_to_method_development', True),
    ('lock_sha256', 'changed'),
])
def test_receipt_cannot_drop_failed_methods_or_upgrade_qualification(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(pipeline, 'checksum', lambda path: 'synthetic-sha')
    receipt = {'status': 'scored_and_sealed', 'method_jobs_expected': 57, 'method_rows_scored': 57,
               'independent_drugs': 1, 'independent_studies': 1, 'strict_main_eligible': False,
               'metrics_exposed_to_method_development': False, 'lock_sha256': 'synthetic-sha'}
    if key:
        receipt[key] = value
        with pytest.raises(RuntimeError):
            pipeline.validate_receipt(receipt, tmp_path/'lock.json')
    else:
        pipeline.validate_receipt(receipt, tmp_path/'lock.json')
