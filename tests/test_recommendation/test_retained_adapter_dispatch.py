"""A helper import must not change which original adapter is being verified."""
import importlib.util
from pathlib import Path
import sys
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
SPEC=importlib.util.spec_from_file_location('staged_adapter_scoring',STAGED/'scripts/recommendation_execution_repairs/scoring.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)


def record(kind):
    producer=s.snapshot_candidate if kind=='snapshot' else s.complete_candidate
    code={p:'a'*64 for p in producer.IMPLEMENTATION_FILES}
    value={'execution_revision':s.REVISION,'implementation_sha256':code}
    if kind=='complete': value['adapter_route']='complete_original_pksim5'
    return value,code


def check(value,code):
    with s.repaired_execution_profile():
        return s.backend.verify_execution_binding(value,frozen_code_sha256=code,
                    expected_revision=s.REVISION,label='Unit adapter')


def test_snapshot_own_entrypoint_wins_over_complete_helper_dependency():
    value,code=record('snapshot')
    assert 'scripts/recommendation_execution_repairs/complete_candidate.py' in code
    assert 'scripts/recommendation_execution_repairs/native_pkml.py' not in code
    assert check(value,code)==code


def test_original_complete_adapter_still_checks_its_full_producer_set():
    value,code=record('complete')
    assert 'scripts/recommendation_execution_repairs/snapshot_candidate.py' not in code
    assert check(value,code)==code


@pytest.mark.parametrize('kind,missing',[
 ('snapshot','src/pbpk_kb/blueprints/source_adaptation_audit.py'),
 ('snapshot','src/pbpk_kb/blueprints/design_runtime.py'),
 ('snapshot','src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R'),
 ('complete','src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R'),
 ('complete','scripts/recommendation_execution_repairs/native_pkml.py'),
 ('complete','scripts/recommendation_execution_repairs/coadministration_isolation.py'),
])
def test_correct_adapter_does_not_waive_missing_runtime_or_proof_dependencies(kind,missing):
    value,code=record(kind);value['implementation_sha256']=dict(code);value['implementation_sha256'].pop(missing)
    with pytest.raises(ValueError,match='incomplete'): check(value,code)


def test_complete_route_cannot_masquerade_as_a_snapshot():
    value,code=record('snapshot');value['adapter_route']='complete_original_pksim5'
    with pytest.raises(ValueError,match='conflicts'): check(value,code)


def test_removing_snapshot_entrypoint_does_not_make_its_helper_a_complete_adapter():
    value,code=record('snapshot');value['implementation_sha256']=dict(code)
    value['implementation_sha256'].pop('scripts/recommendation_execution_repairs/snapshot_candidate.py')
    with pytest.raises(ValueError,match='incomplete'): check(value,code)


@pytest.fixture(autouse=True)
def use_staged_retention(monkeypatch):
    spec=importlib.util.spec_from_file_location('dependency_staged_retention', STAGED/'scripts/recommendation_execution_repairs/retained_implementations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,'recommendation_execution_repairs.retained_implementations',module)
