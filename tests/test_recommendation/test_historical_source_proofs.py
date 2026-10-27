"""A code update must not turn an unverified or modified artifact into a pass."""
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs import historical_source_proofs as h
from recommendation_execution_repairs import snapshot_repair as s


def test_exact_historical_source_retains_original_implementation_identity():
    record=h.checkpoint(ROOT);assert record['count']==75
    entry=record['entries'][0];path=ROOT/entry['receipt']['path']
    result=s._verify_receipt(path,entry,ROOT)
    assert result['status']=='verified'
    assert result['implementation']==entry['implementation']


def test_modified_historical_receipt_is_not_relabelled_as_verified(tmp_path):
    entry=h.checkpoint(ROOT)['entries'][0]
    original=ROOT/entry['receipt']['path'];value=json.loads(original.read_text())
    value['operation']='modified';p=tmp_path/'receipt.json';p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Unregistered'):
        h.historical_checker(p,value,ROOT)


def test_failed_source_is_not_in_checkpoint():
    assert all(not x['source_sha256'].startswith('4a011318') for x in h.checkpoint(ROOT)['entries'])


def test_all_eighty_sources_keep_their_original_verified_receipts():
    entries=[row for record in h.records(ROOT) for row in record["entries"]]
    assert len(entries)==80 and len({row["source_sha256"] for row in entries})==80
    recent=[row for row in entries if row["source_sha256"].startswith("77d4db0")][0]
    result=s._verify_receipt(ROOT/recent["receipt"]["path"],recent,ROOT)
    assert result["status"]=="verified" and result["implementation"]==recent["implementation"]
