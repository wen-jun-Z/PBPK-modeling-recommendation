"""Staged JSON/resource fixes preserve values and reject unsafe boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/runtime_limits_followup_20261002'
LIMIT = 16 * 1024**2
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]


def staged(name, relative, package=''):
    source = BASE / 'staged' / relative
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / relative)
    module.__package__ = package
    exec(compile(source.read_text(), module.__file__, 'exec'), module.__dict__)
    assert module.ROOT == ROOT
    return module


@pytest.fixture(scope='module')
def writer():
    return staged('_staged_json_followup', 'scripts/recommendation_candidate_executor.py')


@pytest.fixture(scope='module')
def complete():
    return staged('recommendation_execution_repairs._staged_complete_limits',
                  'scripts/recommendation_execution_repairs/complete_candidate.py',
                  'recommendation_execution_repairs')


@pytest.fixture(scope='module')
def native():
    return staged('recommendation_execution_repairs._staged_native_limits',
                  'scripts/recommendation_execution_repairs/native_pkml.py',
                  'recommendation_execution_repairs')


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch, tmp_path):
    assert tmp_path.is_relative_to(BASE) or tmp_path.is_relative_to(ROOT/'runtime/tmp'), 'Use a project-owned test directory'
    def forbidden(*args, **kwargs):
        raise AssertionError('These staged tests must never start a subprocess')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def test_large_legal_json_uses_compact_without_losing_fields(writer, tmp_path):
    value = {'metadata': {'nested': {'rows': [{'name': 'µ', 'value': 0}] * 350000}}}
    pretty = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    compact = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(pretty.encode()) > LIMIT > len(compact.encode())
    destination = tmp_path / 'large-valid.json'
    writer._save(destination, value)
    assert destination.read_bytes() == compact.encode()
    assert json.loads(destination.read_text()) == value


def test_small_json_preserves_pretty_format_values_and_fsync(writer, tmp_path, monkeypatch):
    value = {'unicode': '\u4e2d\u6587 µ', 'nested': [None, True, False, {'integer': 2**75}],
             'numbers': [-0.0, 1.2345678901234567, 1e-250]}
    destination = tmp_path / 'small.json'
    expected = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    synced = []
    real_fsync = os.fsync
    def record_sync(fd):
        assert os.fstat(fd).st_size == len(expected)
        assert destination.read_bytes() == expected
        synced.append(fd)
        real_fsync(fd)
    monkeypatch.setattr(os, 'fsync', record_sync)
    writer._save(destination, value)
    decoded = json.loads(destination.read_text())
    assert decoded == value and len(synced) == 1
    assert [number.hex() for number in decoded['numbers']] == [number.hex() for number in value['numbers']]


def test_utf8_exact_byte_limit_remains_hard(writer, tmp_path):
    value = 'é' * ((LIMIT - 3) // 2) + 'x'
    assert len((json.dumps(value, ensure_ascii=False) + '\n').encode()) == LIMIT
    destination = tmp_path / 'exact-limit.json'
    writer._save(destination, value)
    assert destination.stat().st_size == LIMIT
    with pytest.raises(ValueError, match='exceeds 16 MiB'):
        writer._save(tmp_path / 'one-byte-over.json', value + 'x')
    assert not (tmp_path / 'one-byte-over.json').exists()


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_are_rejected_before_file_creation(writer, tmp_path, nonfinite):
    destination = tmp_path / 'nonfinite.json'
    with pytest.raises(ValueError, match='Out of range float'):
        writer._save(destination, {'nested': [nonfinite]})
    assert not destination.exists()


def test_exclusive_write_preserves_prior_result(writer, tmp_path):
    destination = tmp_path / 'prior.json'
    prior = b'{"preserved":true}\n'
    destination.write_bytes(prior)
    with pytest.raises(FileExistsError):
        writer._save(destination, {'replacement': 1})
    assert destination.read_bytes() == prior


class BoundaryReached(RuntimeError):
    pass


@pytest.mark.parametrize('seconds', [1, 600, 601, 1799, 1800])
def test_complete_accepts_registered_timeout_boundaries(complete, tmp_path, monkeypatch, seconds):
    calls = []
    def preparation(*args, **kwargs):
        calls.append(kwargs)
        raise BoundaryReached('Validated timeout before outcome-free preparation')
    monkeypatch.setattr(complete, 'prepare_candidate', preparation)
    monkeypatch.setattr(complete.shutil, 'disk_usage', lambda path: SimpleNamespace(free=4 * 1024**3))
    with pytest.raises(BoundaryReached):
        complete.execute_candidate({}, {}, tmp_path / 'complete', timeout_seconds=seconds, project_root=ROOT)
    assert len(calls) == 1 and not (tmp_path / 'complete').exists()


@pytest.mark.parametrize('seconds', [-1, 0, 1801, True, 1800.0, '1800', None])
def test_complete_rejects_invalid_timeout_before_preparation(complete, tmp_path, monkeypatch, seconds):
    def unexpected(*args, **kwargs):
        raise AssertionError('Invalid timeout reached preparation')
    monkeypatch.setattr(complete, 'prepare_candidate', unexpected)
    with pytest.raises(ValueError, match='timeout_seconds must be 1..1800'):
        complete.execute_candidate({}, {}, tmp_path / 'complete', timeout_seconds=seconds, project_root=ROOT)
    assert not (tmp_path / 'complete').exists()


@pytest.mark.parametrize('seconds', [1, 600, 601, 1799, 1800])
def test_native_passes_accepted_timeout_unchanged_to_bounded_runner(native, tmp_path, monkeypatch, seconds):
    pkml = tmp_path / 'model.pkml'
    pkml.write_text('<Model/>')
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        raise BoundaryReached('Native runner intercepted; no R process')
    monkeypatch.setattr(native, 'run_bounded_r', runner)
    with pytest.raises(BoundaryReached):
        native.execute_complete_pkml(pkml, tmp_path / 'native', action='inspect',
                                     timeout_seconds=seconds, cpu_threads=1, project_root=ROOT)
    assert len(calls) == 1 and calls[0]['timeout_seconds'] == seconds
    assert calls[0]['cpu_threads'] == 1
    request_path = Path(calls[0]['args'][0])
    assert request_path == tmp_path/'native/request.json'
    assert calls[0]['args'][1] == native.sha(request_path)


@pytest.mark.parametrize('seconds', [-1, 0, 1801, True, 1800.0, '1800', None])
def test_native_rejects_invalid_timeout_before_writing_or_running(native, tmp_path, monkeypatch, seconds):
    pkml = tmp_path / 'model.pkml'
    pkml.write_text('<Model/>')
    def unexpected(**kwargs):
        raise AssertionError('Invalid timeout reached native runner')
    monkeypatch.setattr(native, 'run_bounded_r', unexpected)
    with pytest.raises(ValueError, match='Execution timeout must be 1..1800 seconds'):
        native.execute_complete_pkml(pkml, tmp_path / 'native', timeout_seconds=seconds, project_root=ROOT)
    assert not (tmp_path / 'native').exists()
