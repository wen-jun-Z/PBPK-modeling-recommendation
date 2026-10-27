"""Reuse only the 75 fully verified, immutable source proofs frozen before maintenance.

This preserves their original implementation identity. It does not relabel old
proofs as runs of new code or reuse clinical predictions.
"""
import json
from pathlib import Path

MANIFEST = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/verified_source_checkpoint_20261002/manifest.json'
MANIFEST_SHA = '178548bb69c16aafb2cb2d96e19d772fa9e0caa9f8458b291bd26fbbc4b811d9'


def checkpoint(root):
    from recommendation_candidate_migration import checked
    return json.loads(checked({'path':MANIFEST,'sha256':MANIFEST_SHA},root).read_text())


EXTRA_MANIFEST = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/verified_source_checkpoint_20261002_after08/manifest.json'
EXTRA_SHA = 'e339f5adced56963769a0acc137bff89a7b360a26a0952fcf8967b8ec4a3d4ce'

FOLLOWUP_MANIFEST = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/native_runtime_followup_20261002/manifest.json'
FOLLOWUP_SHA = '1b2cb895c856d6c10c7e6f4209378064847bdf4901c2aa53d1aea3849ee72f01'


def records(root):
    from recommendation_candidate_migration import checked
    result=[checkpoint(root)] if (Path(root)/MANIFEST).exists() else []
    if (Path(root)/EXTRA_MANIFEST).exists():
        result.append(json.loads(checked({'path':EXTRA_MANIFEST,'sha256':EXTRA_SHA},root).read_text()))
    if (Path(root)/FOLLOWUP_MANIFEST).exists():
        result.append(json.loads(checked({'path':FOLLOWUP_MANIFEST,'sha256':FOLLOWUP_SHA},root).read_text()))
    return result


def original_receipt(provenance,root):
    from recommendation_candidate_migration import checked
    rows=[r for record in records(root) for r in record['entries'] if r['source_sha256']==provenance['source_sha256']
          and r['source_path']==provenance['source_path']]
    if not rows:return None
    if len(rows)!=1:raise ValueError('Historical source identity is ambiguous')
    return checked(rows[0]['receipt'],root)


FAILED_MANIFEST = 'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/snapshot_input_followup_20261002/failed_source_checkpoint/manifest.json'
FAILED_SHA = 'af4d897fd22b907a5cc6b0c88b7e8f51fd65735ca5f2e16d60dc87d635972731'


def historical_checker(receipt_path,receipt,root):
    from recommendation_candidate_migration import checked,sha
    integrity_records=records(root)
    if (Path(root)/FAILED_MANIFEST).exists():
        integrity_records.append(json.loads(checked({'path':FAILED_MANIFEST,'sha256':FAILED_SHA},root).read_text()))
    candidates=[(record,r) for record in integrity_records for r in record['entries']
                if r['receipt']['sha256']==sha(receipt_path) and r['source_path']==receipt['source_path']
                and r['source_sha256']==receipt['source_sha256']]
    matches=[r for _,r in candidates]
    record=candidates[0][0] if len(candidates)==1 else {}
    if len(matches)!=1 or receipt['implementation']!=matches[0]['implementation']:
        raise ValueError('Unregistered historical source proof or implementation')
    registered_receipt=checked(matches[0]['receipt'],root)
    if record.get('failed_receipt_integrity_only') is True:
        if (receipt.get('status')!='failed' or receipt!=json.loads(registered_receipt.read_text())
                or record.get('verified_source_reuse_allowed') is not False):
            raise ValueError('Historical failed receipt cannot become a verified source proof')
        for entry in record['preparation_archive']:
            if entry['preserved']['sha256']!=entry['original']['sha256']:
                raise ValueError('Historical failed preparation archive differs')
            checked(entry['preserved'],root)
    archived={r['original']['path']:r for r in record['implementation_archive']}
    for ref in receipt['implementation']:
        entry=archived[ref['path']]
        if entry['original']!=ref or entry['preserved']['sha256']!=ref['sha256']:
            raise ValueError('Historical implementation archive differs')
        checked(entry['preserved'],root)
    def verify(ref):
        entry=archived.get(ref['path'])
        if entry and entry['original']==ref:return checked(entry['preserved'],root)
        return checked(ref,root)
    return verify
