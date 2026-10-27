"""Append imports must survive retries without altering previous evidence."""
import json
import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest

from pbpk_kb.storage.import_processed_evidence import EvidenceImporter, psycopg_dsn


@pytest.mark.skipif(os.getenv("PBPK_RUN_PROJECT_DB_TESTS") != "1", reason="Explicit project database test only")
def test_model_identity_enrichment_preserves_ids_and_human_decisions():
    import psycopg
    from pbpk_kb.config.settings import get_settings
    from pbpk_kb.storage.import_processed_evidence import enrich_bound_compound_identities

    ids = [uuid4(), uuid4()]
    with psycopg.connect(psycopg_dsn(get_settings().database_url)) as connection:
        try:
            rows = []
            for index, (cid, status) in enumerate(zip(ids, ["machine_extracted", "reviewed_accepted"])):
                name = "identity-test-" + cid.hex
                connection.execute("INSERT INTO compound_identity (compound_id,preferred_name,curation_status) VALUES (%s,%s,%s)",
                                   (cid, name, status))
                rows.append({"compound_id": str(cid), "preferred_name": name,
                             "existing_registry_binding": {"compound_id": str(cid)},
                             "pubchem_cid": str(123 + index), "pubchem_cids": [str(123 + index)],
                             "identity_provenance": [{"source_id": "pubchem", "checksum": "fixture"}]})
            assert enrich_bound_compound_identities(connection, rows, "test-run") == 1
            assert enrich_bound_compound_identities(connection, rows, "test-run") == 0
            result = connection.execute("SELECT compound_id,pubchem_cid,curation_status FROM compound_identity WHERE compound_id=ANY(%s)", (ids,)).fetchall()
            assert {r[0]: r[1:] for r in result} == {ids[0]: ("123", "machine_extracted"), ids[1]: (None, "reviewed_accepted")}
            events = connection.execute("SELECT field_changes FROM curation_event WHERE evidence_table='compound_identity' AND evidence_id=ANY(%s)", (ids,)).fetchall()
            assert len(events) == 1 and events[0][0]["compound_id_unchanged"] is True
        finally:
            connection.rollback()


def test_empty_validated_partition_is_a_valid_count(tmp_path):
    output = tmp_path / "07_llm_quality_validation"
    output.mkdir()
    (output / "final_auto_validated_evidence.jsonl").write_text("")
    summary = output / "quality_summary.json"
    summary.write_text(json.dumps({"accepted": True, "final_auto_validated_count": 0}))
    importer = object.__new__(EvidenceImporter)
    importer.final_evidence_dir = tmp_path
    assert importer._load_auto_validated_rows() == {}
    summary.write_text(json.dumps({"accepted": True}))
    with pytest.raises(RuntimeError, match="valid serving evidence count"):
        importer._load_auto_validated_rows()


@pytest.mark.skipif(os.getenv("PBPK_RUN_PROJECT_DB_TESTS") != "1", reason="Explicit project database test only")
def test_native_append_is_idempotent_and_rolls_back_partial_input():
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from pbpk_kb.config.settings import get_settings

    root = Path(__file__).resolve().parents[2]
    url = get_settings().database_url
    address = make_url(url)
    ports = json.loads((root / "runtime/deployment/ports.json").read_text())
    assert address.host == "127.0.0.1" and address.port == ports["postgres"] and address.database == "pbpk_kb"
    table = "pbpk_evidence_import_probe_" + uuid4().hex
    importer = object.__new__(EvidenceImporter)
    importer.native_database_url = url
    importer.counts = Counter()
    with psycopg.connect(psycopg_dsn(url), autocommit=True) as control:
        control.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY, quote text NOT NULL)").format(sql.Identifier(table)))
        try:
            importer._copy_rows(table, ["id", "quote"], [{"id": 1, "quote": "original"}])
            importer._copy_rows(table, ["id", "quote"], [{"id": 1, "quote": "must not overwrite"}, {"id": 2, "quote": "new"}])
            def interrupted_rows():
                yield {"id": 3, "quote": "partial"}
                raise RuntimeError("source stream interrupted")
            with pytest.raises(RuntimeError, match="source stream interrupted"):
                importer._copy_rows(table, ["id", "quote"], interrupted_rows())
            result = control.execute(sql.SQL("SELECT id, quote FROM {} ORDER BY id").format(sql.Identifier(table))).fetchall()
            assert result == [(1, "original"), (2, "new")]
            assert importer.counts[table + ":inserted"] == 2
            assert importer.counts[table + ":already_present"] == 1
        finally:
            control.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))



@pytest.mark.parametrize("failure", ["source", "normalization", "writer_creation", None])
def test_multi_table_evidence_import_aborts_all_writers_on_failure(monkeypatch, failure):
    from types import SimpleNamespace
    from pbpk_kb.storage import import_processed_evidence as module

    original_failure = RuntimeError("deliberate " + str(failure))
    writers = []

    class MemoryWriter:
        def __init__(self, importer, table, columns):
            if failure == "writer_creation" and len(writers) == 2:
                raise original_failure
            self.table = table
            self.rows = []
            self.closed_with = []
            writers.append(self)

        def write(self, row):
            self.rows.append(row)

        def close(self, error=None):
            self.closed_with.append(error)
            if error is not None:
                self.rows.clear()

    def rows(path):
        yield {"evidence_id": "one", "evidence_type": "clinical_pk"}
        if failure == "source":
            raise original_failure

    def normalize(row):
        if failure == "normalization":
            raise original_failure
        return row

    importer = object.__new__(EvidenceImporter)
    importer.paths = SimpleNamespace(path=lambda name: Path(name))
    importer.limit = None
    importer.final_evidence_dir = None
    importer.auto_validated_rows = {}
    importer.seen_candidate_ids = set()
    importer.seen_typed_ids = {}
    importer.counts = Counter()
    importer._normalize_evidence_row = normalize
    importer._candidate_row = lambda row: row
    importer._typed_row = lambda row: ("clinical_pk_evidence", row)
    monkeypatch.setattr(module, "CopyWriter", MemoryWriter)
    monkeypatch.setattr(module, "read_jsonl", rows)
    if failure is not None:
        with pytest.raises(RuntimeError) as caught:
            importer._import_evidence()
        assert caught.value is original_failure
        assert writers
        assert all(writer.closed_with == [original_failure] for writer in writers)
        assert all(writer.rows == [] for writer in writers)
    else:
        importer._import_evidence()
        assert len(writers) == 8
        assert all(writer.closed_with == [None] for writer in writers)
        assert sum(len(writer.rows) for writer in writers) == 2


def test_multi_table_abort_continues_after_a_cleanup_error(monkeypatch):
    from types import SimpleNamespace
    from pbpk_kb.storage import import_processed_evidence as module

    failure = RuntimeError("source failure")
    calls = []

    class MemoryWriter:
        def __init__(self, importer, table, columns):
            self.table = table

        def close(self, error=None):
            calls.append((self.table, error))
            if len(calls) == 1:
                raise RuntimeError("cleanup failure")

    def rows(path):
        raise failure
        yield

    importer = object.__new__(EvidenceImporter)
    importer.paths = SimpleNamespace(path=lambda name: Path(name))
    monkeypatch.setattr(module, "CopyWriter", MemoryWriter)
    monkeypatch.setattr(module, "read_jsonl", rows)
    with pytest.raises(RuntimeError) as caught:
        importer._import_evidence()
    assert caught.value is failure
    assert len(calls) == 8
    assert all(error is failure for _, error in calls)


@pytest.mark.skipif(os.getenv("PBPK_RUN_PROJECT_DB_TESTS") != "1", reason="Explicit project database test only")
def test_serving_status_reconciliation_preserves_facts_and_human_decisions():
    from sqlalchemy import create_engine,text
    from sqlalchemy.engine import make_url
    from pbpk_kb.config.settings import get_settings
    from pbpk_kb.storage.import_processed_evidence import reconcile_auto_validated_statuses
    root=Path(__file__).resolve().parents[2];url=get_settings().database_url;address=make_url(url)
    assert address.host=='127.0.0.1' and address.port==json.loads((root/'runtime/deployment/ports.json').read_text())['postgres'] and address.database=='pbpk_kb'
    engine=create_engine(url)
    with engine.connect() as c:
        transaction=c.begin()
        try:
            c.execute(text("CREATE TEMP TABLE processed_evidence_candidate (evidence_id uuid PRIMARY KEY,run_id text,curation_status curation_status,updated_at timestamptz,source_record jsonb,raw_asset_id text,source_text text,checksum text,parameter text,value numeric,unit text,compound_id uuid,compound_name text,evidence_type text)"))
            c.execute(text("CREATE TEMP TABLE evidence_layer_membership (evidence_id uuid,run_id text,evidence_layer text)"))
            c.execute(text("CREATE TEMP TABLE evidence_compound_attribution (evidence_id uuid,compound_id uuid,compound_role text)"))
            c.execute(text("CREATE TEMP TABLE curation_event (evidence_table text,evidence_id uuid,previous_status curation_status,new_status curation_status,curator text,curator_comment text,field_changes jsonb)"))
            c.execute(text("CREATE TEMP VIEW auto_validated_evidence AS SELECT evidence_id FROM processed_evidence_candidate WHERE curation_status='auto_validated'"))
            a,b=str(uuid4()),str(uuid4());compound=str(uuid4())
            common={'run_id':'new-run','quality_status':'auto_validated','source_text':'Caffeine Cmax was 10 mg/L.',
                    'checksum':'a'*64,'parameter':'Cmax','value':10.0,'unit':'mg/L','compound_id':compound,
                    'compound_name':'caffeine','evidence_type':'clinical_pk','compound_role':'target_compound',
                    'raw_asset_id':'new-acquisition','validation_method':'test'}
            incoming={eid:{**common,'evidence_id':eid} for eid in [a,b]}
            for eid in [a,b]:
                c.execute(text("INSERT INTO processed_evidence_candidate VALUES (CAST(:eid AS uuid),'old-run','needs_review',now(),'{}','old-acquisition',:source_text,:checksum,:parameter,:value,:unit,CAST(:compound_id AS uuid),:compound_name,:evidence_type)"),{'eid':eid,**common})
                c.execute(text("INSERT INTO evidence_layer_membership VALUES (CAST(:eid AS uuid),'new-run','auto_validated')"),{'eid':eid})
                c.execute(text("INSERT INTO evidence_compound_attribution VALUES (CAST(:eid AS uuid),CAST(:compound AS uuid),'target_compound')"),{'eid':eid,'compound':compound})
            bad={a:incoming[a],b:{**incoming[b],'value':20}}
            with pytest.raises(RuntimeError,match='scientific fields differ'):reconcile_auto_validated_statuses(c,bad)
            assert c.execute(text("SELECT count(*) FROM curation_event")).scalar_one()==0
            assert c.execute(text("SELECT count(*) FROM processed_evidence_candidate WHERE curation_status='needs_review'")).scalar_one()==2
            result=reconcile_auto_validated_statuses(c,{a:incoming[a]})
            assert result=={'verified_count':1,'promoted_ids':[a]}
            assert c.execute(text("SELECT raw_asset_id FROM processed_evidence_candidate WHERE evidence_id::text=:id"),{'id':a}).scalar_one()=='old-acquisition'
            event=c.execute(text('SELECT curator,field_changes FROM curation_event')).mappings().one()
            assert event['curator']=='system:final_auto_validation' and event['field_changes']['scientific_fields_unchanged']
            assert reconcile_auto_validated_statuses(c,{a:incoming[a]})['promoted_ids']==[]
            c.execute(text("INSERT INTO curation_event (evidence_table,evidence_id,previous_status,new_status,curator) VALUES ('processed_evidence_candidate',CAST(:id AS uuid),'needs_review','needs_review','test reviewer')"),{'id':b})
            with pytest.raises(RuntimeError,match='untouched machine evidence'):reconcile_auto_validated_statuses(c,{b:incoming[b]})
            assert c.execute(text("SELECT curation_status::text FROM processed_evidence_candidate WHERE evidence_id::text=:id"),{'id':b}).scalar_one()=='needs_review'
        finally:
            transaction.rollback()
    engine.dispose()
