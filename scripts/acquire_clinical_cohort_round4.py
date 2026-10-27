#!/usr/bin/env python3
"""Bounded round4 curator retrieval; public documents only, no qualification inference."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/blueprint_design/external_validation'
OUT = BASE / 'clinical_cohort_round4_20260920'
DIRECTORIES = [ROOT/'experiments/blueprint_design/independent_cohort_20260920'] + [BASE/n for n in (
    'newstudy_bundle_20260920','additional_native_cohort_20260920','concentration_curve_audit_20260920',
    'remaining_cohort_20260920','expanded_cohort_20260920','clinical_cohort_round4_20260920')]
LIMIT = 100*1024**2
FILE_LIMIT = 10*1024**2
MUTEX = threading.Lock()
TRANSFER = 0
BASE_BYTES = sum(p.stat().st_size for d in DIRECTORIES for p in d.rglob('*') if p.is_file())


def fetch(url, destination):
    global TRANSFER
    marker = destination.with_suffix(destination.suffix+'.acquisition.json')
    if marker.exists():
        return json.loads(marker.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    event = dict(url=url,path=str(destination.relative_to(ROOT)),attempts=[])
    for attempt in range(2):
        data = bytearray()
        # curl's total timeout covers DNS/connect/slow-body, unlike a per-read timeout.
        proc = subprocess.Popen(['curl','--fail','--silent','--show-error','--location',
            '--connect-timeout','10','--max-time','30','--max-filesize',str(FILE_LIMIT),url],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT)
        failure = None
        try:
            while True:
                chunk=proc.stdout.read(65536)
                if not chunk:break
                with MUTEX:
                    TRANSFER += len(chunk)
                    if BASE_BYTES+TRANSFER+262144 >= LIMIT:
                        raise ValueError('cumulative_curation_100MiB_limit')
                if len(data)+len(chunk)>FILE_LIMIT:
                    raise ValueError('document_10MiB_limit')
                data.extend(chunk)
            stderr=proc.stderr.read(8192).decode(errors='replace')
            returncode=proc.wait(timeout=3)
            if returncode:
                failure=f'curl_exit_{returncode}: {stderr[:250]}'
        except Exception as exc:
            failure=str(exc)
            proc.kill()
            proc.wait(timeout=3)
            returncode=-1
        finally:
            proc.stdout.close();proc.stderr.close()
        event['attempts'].append(dict(attempt=attempt+1,bytes_received=len(data),failure=failure))
        if failure is None:
            destination.write_bytes(data);destination.chmod(0o600)
            event.update(status='downloaded',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            break
        event.update(status='failed_no_further_automatic_retry',reason=failure)
        if returncode not in {5,6,7,28,52,56}:break
    marker.write_text(json.dumps(event,indent=2)+'\n')
    return event


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    (OUT/'sealed').mkdir(parents=True,exist_ok=True);(OUT/'sealed').chmod(0o700)
    drugs=['esomeprazole','risperidone','olanzapine','moxidectin','amikacin','cabozantinib','posaconazole',
           'raltegravir','efavirenz','ruxolitinib','atazanavir','finerenone','warfarin','cyclosporine',
           'dabigatran','tizanidine','theophylline','carbamazepine']
    def search(drug):
        query=f'TITLE:{drug} AND (TITLE:pharmacokinetic OR TITLE:pharmacokinetics OR TITLE:bioequivalence OR TITLE:bioavailability) AND (healthy OR volunteer OR bioequivalence) AND OPEN_ACCESS:Y NOT PUB_TYPE:"Review"'
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode(dict(query=query,format='json',resultType='core',pageSize=12,sort='FIRST_PDATE_D desc'))
        return dict(compound=drug,query=query,event=fetch(url,OUT/'sealed/search'/f'{drug}.json'))
    dest=OUT/'search_manifest.json'
    if dest.exists():
        print('Completed search exists; duplicate retrieval skipped.');return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(search,drugs))
    dest.write_text(json.dumps(dict(records=results,transfer_bytes=TRANSFER,maximum_total_curation_bytes=LIMIT,
        maximum_file_bytes=FILE_LIMIT,maximum_parallel_downloads=3,maximum_total_request_seconds=30,
        maximum_transient_retries=1),indent=2)+'\n')
    print(json.dumps({'queries':len(results),'bytes_received':TRANSFER,'failures':sum(x['event']['status']!='downloaded' for x in results)}))


if __name__=='__main__':main()
