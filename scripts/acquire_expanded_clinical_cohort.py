#!/usr/bin/env python3
"""Bounded public-source retrieval for new independent clinical cohorts.

Output is curator-only. This is not automatic scientific qualification.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import sys
from urllib.parse import urlencode
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/blueprint_design/external_validation/expanded_cohort_20260920"
MAX_FILE = 10 * 1024**2
MAX_TOTAL = 80 * 1024**2
MAX_COHORT = 100 * 1024**2
LOCK = threading.Lock()
SPENT = 0
TRANSFER_THIS_RUN = 0
COHORT_DIRS = [ROOT / 'experiments/blueprint_design/independent_cohort_20260920'] + [ROOT / 'experiments/blueprint_design/external_validation' / name for name in ('newstudy_bundle_20260920', 'additional_native_cohort_20260920', 'concentration_curve_audit_20260920', 'remaining_cohort_20260920', 'expanded_cohort_20260920')]
DISK_BASE = sum(p.stat().st_size for d in COHORT_DIRS for p in d.rglob('*') if p.is_file())
DRUGS = ['atomoxetine', 'dapagliflozin', 'digoxin', 'finerenone', 'omeprazole', 'pitavastatin', 'quetiapine', 'raltegravir', 'isoniazid', 'atazanavir', 'rifampicin', 'fluvoxamine']


def fetch(url, destination):
    global SPENT, TRANSFER_THIS_RUN
    marker = destination.with_suffix(destination.suffix + '.acquisition.json')
    if marker.exists():
        return json.loads(marker.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    event = {'url': url, 'path': str(destination.relative_to(ROOT)), 'attempts': [], 'maximum_file_bytes': MAX_FILE}
    for attempt in range(2):
        data = bytearray()
        start = time.monotonic()
        try:
            with requests.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                if int(response.headers.get('content-length', 0)) > MAX_FILE:
                    raise ValueError('document_size_limit')
                for chunk in response.iter_content(65536):
                    if time.monotonic() - start > 30:
                        raise TimeoutError('total_request_duration_30s')
                    with LOCK:
                        if SPENT + len(chunk) > MAX_TOTAL:
                            raise ValueError('round_total_limit')
                        if DISK_BASE + TRANSFER_THIS_RUN + len(chunk) + 131072 > MAX_COHORT:
                            raise ValueError('cumulative_cohort_100MiB_limit')
                        SPENT += len(chunk)
                        TRANSFER_THIS_RUN += len(chunk)
                    if len(data) + len(chunk) > MAX_FILE:
                        raise ValueError('document_size_limit')
                    data.extend(chunk)
            destination.write_bytes(data)
            destination.chmod(0o600)
            event.update(status='downloaded', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            event['attempts'].append({'attempt': attempt + 1, 'status': 'downloaded', 'bytes': len(data)})
            break
        except Exception as exc:
            event['attempts'].append({'attempt': attempt + 1, 'status': 'failed', 'bytes_received': len(data), 'reason': str(exc)[:300]})
            event.update(status='failed_no_further_automatic_retry', reason=str(exc)[:300])
            # Only transient network failures get one retry. 403/404 and caps stop.
            if not isinstance(exc, (requests.Timeout, requests.ConnectionError, TimeoutError)):
                break
    marker.write_text(json.dumps(event, indent=2) + '\n')
    return event


def search(drug):
    targeted = '--targeted' in sys.argv
    query = f'TITLE:{drug} AND ("healthy" OR "bioequivalence" OR "pharmacokinetics") AND OPEN_ACCESS:Y NOT PUB_TYPE:"Review" AND FIRST_PDATE:[2018-01-01 TO 2026-09-20]'
    if targeted:
        query = f'TITLE:{drug} AND (TITLE:pharmacokinetics OR TITLE:pharmacokinetic OR TITLE:bioequivalence OR TITLE:exposure) AND (ABSTRACT:healthy OR TITLE:healthy OR TITLE:bioequivalence) AND OPEN_ACCESS:Y NOT PUB_TYPE:"Review" AND FIRST_PDATE:[2018-01-01 TO 2026-09-20]'
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({'query': query, 'format': 'json', 'resultType': 'core', 'pageSize': 15, 'sort': 'FIRST_PDATE_D desc'})
    p = OUT / ('sealed/search_targeted' if targeted else 'sealed/search') / f'{drug}.json'
    e = fetch(url, p)
    return {'compound': drug, 'query': query, 'event': e}


def main():
    global SPENT
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / ('targeted_search_manifest.json' if '--targeted' in sys.argv else 'search_manifest.json')
    if marker.exists():
        print('Existing completed search manifest; no duplicate retrieval.')
        return
    (OUT / 'sealed').mkdir(exist_ok=True)
    (OUT / 'sealed').chmod(0o700)
    SPENT = sum(sum(a.get('bytes', a.get('bytes_received', 0)) for a in json.loads(p.read_text()).get('attempts', [])) for p in OUT.glob('sealed/**/*.acquisition.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(search, DRUGS))
    marker.write_text(json.dumps({'schema_version': 'bounded-expanded-clinical-source-retrieval-v1', 'records': records, 'download_bytes': SPENT, 'maximum_total_bytes': MAX_TOTAL, 'maximum_file_bytes': MAX_FILE, 'maximum_parallel_downloads': 3, 'maximum_request_seconds': 30, 'maximum_retries': 1, 'candidate_status_is_not_qualification': True}, indent=2) + '\n')
    print(json.dumps({'completed_drug_queries': len(records), 'download_bytes': SPENT, 'failures': sum(r['event']['status'] != 'downloaded' for r in records)}))


if __name__ == '__main__':
    main()
