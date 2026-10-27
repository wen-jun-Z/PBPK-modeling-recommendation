"""Clinical query boundaries and safe process-record handling; no GPU/subprocess."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rank_recommendation_execution as rank
from scripts import run_recommendation_execution as driver


def scenario():
    return {"scenario_id":"clinical_case_1", "compound":"midazolam",
            "publication_ids":{"pmc":"PMC999999991"}, "sealed_reference":{"observed_auc":987654321},
            "normalized_design":{"dose":{"value":10,"unit":"mg","route":"oral",
                "source_cell":"PMC999999991 table 1", "observed_auc":987654321},
                "population":{"healthy":True,"age_mean_year":30,"source_cell":"PMC999999991"},
                "food":"fasted", "matrix":"plasma", "analyte":"midazolam",
                "endpoint_contracts":[{"endpoint":"auc", "observed":987654321,"source_cell":"PMC999999991"}]}}


def test_query_uses_conditions_not_target_answers_or_citation_identity():
    s = scenario()
    query = rank.build_query(s)
    visible = json.dumps({k:query[k] for k in ["text","graph","features"]})
    assert "987654321" not in visible
    assert "PMC999999991" not in visible
    assert "source_cell" not in visible
    assert "midazolam" in query["text"] and "fasted" in query["text"]
    assert query["input_evidence_ids"] == []
    assert query["variants"] == {}
    assert s == scenario()  # construction must not mutate the original contract


@pytest.mark.parametrize("section,field", [("dose","value"),("population","age_mean_year")])
def test_nested_answers_cannot_pass_a_shallow_clinical_field_allowlist(section, field):
    s = scenario()
    s["normalized_design"][section][field] = {"value":10,"observed_auc":987654321,"source_cell":"PMC999999991"}
    try:
        query = rank.build_query(s)
    except (ValueError, TypeError):
        return  # rejecting malformed typed input is valid
    text = json.dumps(query)
    assert "987654321" not in text and "PMC999999991" not in text


def test_project_path_guard_rejects_parent_and_root():
    with pytest.raises(ValueError):
        rank.inside(rank.ROOT.parent)
    with pytest.raises(ValueError):
        rank.inside(rank.ROOT)


@pytest.mark.parametrize("record", [{}, {"identity":None}, {"identity":{}},
                                   {"identity":{"pid":-1}}, {"identity":{"pid":"invalid"}}])
def test_malformed_pid_record_never_reports_an_alive_project(tmp_path, monkeypatch, record):
    record = {**record, "script": str(driver.SCRIPT.relative_to(driver.ROOT))}
    (tmp_path / "pid.json").write_text(json.dumps(record))
    monkeypatch.setattr(driver, "RUN", tmp_path)
    monkeypatch.setattr(driver.ranking, "read", lambda _: record)
    monkeypatch.setattr(driver, "identity", lambda _: None)
    assert driver.alive() is False


def test_start_ticks_mismatch_is_stale_not_owned_alive(tmp_path, monkeypatch):
    (tmp_path / "pid.json").write_text("{}")
    saved={"identity":{"pid":123,"uid":os.getuid(),"start_ticks":"100"},
           "script":str(driver.SCRIPT.relative_to(driver.ROOT))}
    monkeypatch.setattr(driver, "RUN", tmp_path)
    monkeypatch.setattr(driver.ranking,"read",lambda _:saved)
    monkeypatch.setattr(driver,"identity",lambda _: {"pid":123,"uid":os.getuid(),"start_ticks":"101"})
    assert driver.alive() is False


def test_worker_environment_contains_no_inherited_api_keys(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_TOKEN", "unit-test-only")
    env = driver.environment()
    assert "TEST_SECRET_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(env["TMPDIR"]).is_relative_to(rank.ROOT)


def test_wrong_project_script_does_not_inspect_the_recorded_pid(tmp_path, monkeypatch):
    (tmp_path / "pid.json").write_text("{}")
    saved = {"identity":{"pid":123,"uid":os.getuid(),"start_ticks":"100"}, "script":"other_project.py"}
    monkeypatch.setattr(driver,"RUN",tmp_path)
    monkeypatch.setattr(driver.ranking,"read",lambda _:saved)
    monkeypatch.setattr(driver,"identity",lambda _:pytest.fail("Must not inspect a PID bound to another script"))
    assert driver.alive() is False


@pytest.fixture
def ranking_fixture(tmp_path, monkeypatch):
    """Tiny score matrices exercise binding; recovery/model adapters are mocked."""
    monkeypatch.setattr(rank,"ROOT",tmp_path)
    monkeypatch.setattr(rank,"RANK",tmp_path / "rankings")
    monkeypatch.setattr(rank,"METHODS",("bm25_text",))
    monkeypatch.setattr(rank,"sha",lambda path:"fixture-hash:" + str(path))
    monkeypatch.setitem(sys.modules,"frontier_result_audit",SimpleNamespace(refresh_output_recovery=lambda _:{}))
    queries=[{"query_id":"q1"},{"query_id":"q2"}]
    candidates=[{"candidate_id":"c1","text":"one"},{"candidate_id":"c2","text":"two"}]
    hashes={"queries":"qhash","candidates":"chash"}
    monkeypatch.setattr(rank,"load_inputs",lambda:(queries,candidates,hashes))
    folder=rank.RANK / "results/bm25_text"
    folder.mkdir(parents=True)
    status={"state":"complete", "input_sha256":dict(hashes),
            "clinical_wrapper_sha256":rank.sha(rank.__file__),
            "implementation_sha256":{name:rank.sha(tmp_path / "scripts" / name) for name in
                                     ("run_frontier_comparison.py","frontier_external_models.py")}}
    index={"query_ids":["q1","q2"],"candidate_ids":["c1","c2"],"variants":["original"]}
    (folder / "status.json").write_text(json.dumps(status))
    (folder / "index.json").write_text(json.dumps(index))
    np.save(folder / "scores.npy",np.array([[[1.,2.],[3.,4.]]]),allow_pickle=False)
    return folder,status,index


def test_correctly_bound_small_rankings_can_be_locked(ranking_fixture):
    r=rank.finalize_scores()
    assert r["query_ids"] == ["q1","q2"]
    assert r["candidate_ids"] == ["c1","c2"]
    assert r["all_finite"] is True


@pytest.mark.parametrize("corruption",["query_order","candidate_order","input_hash","wrapper_hash","implementation_hash"])
def test_same_shape_scores_cannot_be_resigned_to_different_inputs(ranking_fixture,corruption):
    folder,status,index=ranking_fixture
    if corruption=="query_order":
        index["query_ids"].reverse()
    elif corruption=="candidate_order":
        index["candidate_ids"].reverse()
    elif corruption=="input_hash":
        status["input_sha256"]["queries"]="old-hash"
    elif corruption=="wrapper_hash":
        status["clinical_wrapper_sha256"]="old-wrapper"
    else:
        status["implementation_sha256"]["run_frontier_comparison.py"]="old-implementation"
    (folder / "status.json").write_text(json.dumps(status))
    (folder / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        rank.finalize_scores()
    assert not (rank.RANK / "ranking_lock.json").exists()


@pytest.fixture
def resume_fixture(ranking_fixture, monkeypatch):
    folder, state, index = ranking_fixture
    monkeypatch.setattr(driver, 'ranking', rank)
    monkeypatch.setattr(driver, 'ROOT', rank.ROOT)
    state['clinical_wrapper_sha256'] = rank.sha(rank.ROOT / 'scripts/rank_recommendation_execution.py')
    (folder / 'status.json').write_text(json.dumps(state))
    np.save(folder / 'query_seconds.npy', np.array([[1., 2.]]), allow_pickle=False)
    return folder, state, index


def test_resume_checks_index_before_any_rank_runner_write(resume_fixture):
    folder, state, index = resume_fixture
    assert driver.verify_retained_ranking('bm25_text')['complete_rows'] == 2
    original_scores = (folder / 'scores.npy').read_bytes()
    index['candidate_ids'].reverse()
    (folder / 'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='identity order'):
        driver.verify_retained_ranking('bm25_text')
    assert (folder / 'scores.npy').read_bytes() == original_scores


def test_partial_resume_requires_whole_score_row_and_timing(resume_fixture):
    folder, state, _ = resume_fixture
    state.update(state='scoring')
    state.pop('clinical_wrapper_sha256')
    (folder / 'status.json').write_text(json.dumps(state))
    np.save(folder / 'query_seconds.npy', np.array([[1., np.nan]]), allow_pickle=False)
    assert driver.verify_retained_ranking('bm25_text')['complete_rows'] == 1
    np.save(folder / 'scores.npy', np.array([[[1., np.nan], [3., 4.]]]), allow_pickle=False)
    assert driver.verify_retained_ranking('bm25_text')['complete_rows'] == 0


@pytest.mark.parametrize('field', ['input_sha256', 'implementation_sha256', 'clinical_wrapper_sha256'])
def test_resume_rejects_provenance_mismatch_before_overwrite(resume_fixture, field):
    folder, state, _ = resume_fixture
    state[field] = 'changed'
    (folder / 'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):
        driver.verify_retained_ranking('bm25_text')


def test_completed_resume_rejects_partial_timing(resume_fixture):
    folder, _, _ = resume_fixture
    np.save(folder / 'query_seconds.npy', np.array([[1., np.nan]]), allow_pickle=False)
    with pytest.raises(ValueError, match='incomplete rows'):
        driver.verify_retained_ranking('bm25_text')


def test_missing_status_cannot_relabel_existing_scores(resume_fixture):
    folder, _, _ = resume_fixture
    (folder / 'status.json').unlink()
    with pytest.raises(ValueError, match='without provenance'):
        driver.verify_retained_ranking('bm25_text')


@pytest.fixture
def transaction_fixture(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'AMENDMENT', tmp_path / 'amendment.json')
    monkeypatch.setattr(driver.ranking, 'ROOT', tmp_path)
    monkeypatch.setattr(driver.ranking, 'sha', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    monkeypatch.setattr(driver, 'freeze_preparation', lambda: {'frozen': True})
    moves, archived = [], []
    for name in ('protocol', 'plan'):
        p = tmp_path / name
        p.write_text(name)
        sha = driver.ranking.sha(p)
        moves.append({'source': name, 'destination': 'history/' + name, 'sha256': sha})
        archived.append({'path': 'history/' + name, 'sha256': sha})
    transaction = tmp_path / 'pending.json'
    transaction.write_text(json.dumps({'moves': moves, 'code': [], 'unchanged_inputs': [],
                                       'amendment': {'archived_files': archived}}))
    return tmp_path, transaction


def test_amendment_can_resume_after_one_file_was_retired(transaction_fixture):
    root, transaction = transaction_fixture
    (root / 'history').mkdir()
    (root / 'protocol').rename(root / 'history/protocol')
    assert driver.finish_amendment(transaction) == {'frozen': True}
    assert (root / 'history/plan').read_text() == 'plan'
    assert not (root / 'plan').exists()
    assert (root / 'amendment.json').exists()


def test_amendment_never_overwrites_a_conflicting_historical_file(transaction_fixture):
    root, transaction = transaction_fixture
    (root / 'history').mkdir()
    (root / 'history/protocol').write_text('different history')
    with pytest.raises(ValueError, match='Conflicting'):
        driver.finish_amendment(transaction)
    assert (root / 'protocol').read_text() == 'protocol'
    assert (root / 'history/protocol').read_text() == 'different history'


def test_retirement_transaction_commits_before_new_preparation_is_created(transaction_fixture, monkeypatch):
    root, transaction = transaction_fixture
    def interrupted_preparation():
        (root / 'protocol').write_text('new active preparation at original pathname')
        raise OSError('simulated power loss during preparation')
    monkeypatch.setattr(driver, 'freeze_preparation', interrupted_preparation)
    with pytest.raises(OSError, match='simulated power loss'):
        driver.finish_amendment(transaction)
    assert not transaction.exists()
    assert (root / 'completed.json').exists()
    assert (root / 'amendment.json').exists()
    assert (root / 'history/protocol').read_text() == 'protocol'


def test_amendment_refuses_a_still_owned_child_even_if_parent_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'RUN', tmp_path)
    saved = {'pid': 123, 'uid': os.getuid(), 'start_ticks': 'abc'}
    (tmp_path / 'children.json').write_text(json.dumps({'old_ranker': {'identity': saved}}))
    monkeypatch.setattr(driver, 'identity', lambda pid: saved)
    monkeypatch.setattr(driver, '_amend_engineering_locked', lambda: pytest.fail('Do not edit while a child is live'))
    with pytest.raises(ValueError, match='child is still running'):
        driver.amend_engineering()


def test_amendment_and_start_share_the_same_mutex(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(driver, 'RUN', tmp_path)
    monkeypatch.setattr(driver, '_amend_engineering_locked', lambda: pytest.fail('No mutation while start lock is held'))
    with (tmp_path / 'start.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            driver.amend_engineering()


def test_cpu_migration_and_rankings_overlap_and_scoring_waits_for_both(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(driver, 'RUN', tmp_path)
    monkeypatch.setattr(driver, 'OUT', tmp_path / 'experiment')
    monkeypatch.setattr(driver, 'MIGRATIONS', tmp_path / 'experiment/migrations')
    monkeypatch.setattr(driver.signal, 'signal', lambda *a: None)
    initial = {'migration_workers': 2, 'migration_seconds_per_stage': 600,
               'migration_output_budget_bytes': 100000, 'migration_total_timeout_seconds': 600}
    monkeypatch.setattr(driver, 'freeze_preparation', lambda: initial)
    monkeypatch.setattr(driver.ranking, 'rows', lambda _: [])
    monkeypatch.setattr(driver, 'ref', lambda p: {'path': str(p), 'sha256': 'fixture'})
    monkeypatch.setattr(driver, 'save', lambda *a: None)
    ranked, migrated, prepared, native = (threading.Event() for _ in range(4))
    finished = []
    class FakeSupervisor:
        def __init__(self, _): self.stop_event = threading.Event()
        def rankings(self, *, during_preparation):
            assert during_preparation
            ranked.set()
            assert native.wait(2), 'Ranking must overlap migration/native preparation'
            finished.append('ranking')
        def native(self):
            assert migrated.is_set() and prepared.is_set()
            native.set()
            finished.append('native')
        def guard(self): assert not self.stop_event.is_set()
    monkeypatch.setattr(driver, 'Supervisor', FakeSupervisor)
    def migrate(*a, **kw):
        assert ranked.wait(2), 'CPU migration must not hold up GPU ranking startup'
        migrated.set()
        return {'stop_reason': None}
    def prepare():
        assert migrated.is_set()
        prepared.set()
    def score(_):
        assert sorted(finished) == ['native', 'ranking']
        return {'summary': {'fixture': 'all branches completed before scoring'}}
    monkeypatch.setattr(driver, 'prepare', prepare)
    monkeypatch.setattr(driver, 'verify_protocol', lambda: prepared.is_set())
    monkeypatch.setitem(sys.modules, 'recommendation_candidate_migration', SimpleNamespace(prepare_migrations=migrate))
    monkeypatch.setitem(sys.modules, 'score_recommendation_execution', SimpleNamespace(score_run=score))
    driver.run()
    assert sorted(finished) == ['native', 'ranking']


@pytest.fixture
def migration_amendment_fixture(tmp_path, monkeypatch):
    import hashlib
    import shutil
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'OUT', tmp_path / 'experiment')
    monkeypatch.setattr(driver.ranking, 'ROOT', tmp_path)
    monkeypatch.setattr(driver.ranking, 'RANK', tmp_path / 'rankings')
    monkeypatch.setattr(driver, 'AMENDMENT', driver.OUT / 'engineering_amendments/repair_03.json')
    monkeypatch.setattr(driver, 'MIGRATIONS', driver.OUT / 'migrations_repair_03')
    monkeypatch.setattr(driver, 'CODE', ['scripts/run_recommendation_execution.py',
                                        'scripts/recommendation_candidate_executor.py'])
    monkeypatch.setattr(driver.ranking, 'METHODS', ('bm25_text',))
    monkeypatch.setattr(driver, 'verify_retained_ranking', lambda *a, **kw: {'method': 'bm25_text', 'complete_rows': 41})
    monkeypatch.setattr(driver, 'freeze_preparation', lambda: {'frozen': True})
    monkeypatch.setattr(driver.ranking, 'sha', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    files = [tmp_path / p for p in driver.CODE]
    files += [driver.OUT / 'inputs/frozen.json', driver.OUT / 'engineering_amendments/repair_02.json']
    for p in files:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    preparation = driver.OUT / 'preparation.json'
    preparation.write_text(json.dumps({'code': [driver.ref(tmp_path / p) for p in driver.CODE],
                                       'inputs': [driver.ref(p) for p in files[2:]]}))
    files.append(preparation)
    archive = driver.OUT / 'engineering_amendments/attempt_02'
    rows = []
    for p in files:
        saved = archive / 'frozen_files' / p.relative_to(tmp_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, saved)
        rows.append({'original_path': str(p.relative_to(tmp_path)), 'archive_path': str(saved.relative_to(tmp_path)),
                     'sha256': driver.ranking.sha(p)})
    (archive / 'archive_manifest.json').write_text(json.dumps({'files': rows, 'preparation': driver.ref(preparation)}))
    (tmp_path / driver.CODE[0]).write_text('repaired migration directory and transaction')
    return tmp_path


def test_migration_only_amendment_preserves_previous_preparation_and_rankings(migration_amendment_fixture):
    assert driver.amend_migration_recovery() == {'frozen': True}
    assert not (driver.OUT / 'preparation.json').exists()
    assert (driver.OUT / 'engineering_amendments/attempt_02/retired_active_files/preparation.json').exists()
    value = driver.ranking.read(driver.AMENDMENT)
    assert value['retained_rankings'][0]['complete_rows'] == 41
    assert value['no_clinical_observations_accessed'] is True
    assert (driver.OUT / 'engineering_amendments/completed_repair_03_transaction.json').exists()


def test_migration_amendment_rejects_changed_frozen_inputs(migration_amendment_fixture):
    (driver.OUT / 'inputs/frozen.json').write_text('changed clinical design')
    with pytest.raises(ValueError, match='Frozen reference changed'):
        driver.amend_migration_recovery()
    assert (driver.OUT / 'preparation.json').exists()


def test_migration_amendment_cannot_change_prediction_implementation(migration_amendment_fixture):
    (driver.ROOT / 'scripts/recommendation_candidate_executor.py').write_text('changed simulator')
    with pytest.raises(ValueError, match='Non-migration code changed'):
        driver.amend_migration_recovery()
    assert (driver.OUT / 'preparation.json').exists()
