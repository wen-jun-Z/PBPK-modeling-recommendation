#!/usr/bin/env python3
"""Bounded identity retrieval for the predeclared clinical expansion candidates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/blueprint_design/expanded_validation_20261007/pubchem_properties.json'
DRUGS = ('Amikacin', 'Clarithromycin', 'Ethinylestradiol', 'Finerenone', 'Fluvoxamine',
         'Griseofulvin', 'Levonorgestrel', 'Lisinopril', 'Mexiletine', 'Moclobemide',
         'Montelukast', 'Oxycodone', 'Paclitaxel', 'Paroxetine', 'Pitavastatin',
         'Probenecid', 'Propofol', 'Rifampicin', 'Rosuvastatin', 'Vancomycin')
LIMIT = 128 * 1024


def fetch(drug):
    url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(drug) +
           '/property/MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES,IsomericSMILES/JSON')
    record = {'drug': drug, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    begin = time.monotonic()
    try:
        data = bytearray()
        with requests.get(url, stream=True, timeout=(5, 20)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(16384):
                data.extend(chunk)
                if len(data) > LIMIT or time.monotonic() - begin > 30:
                    raise ValueError('Bounded download exceeded bytes/time budget')
        body = json.loads(data)
        properties = body['PropertyTable']['Properties']
        if len(properties) != 1 or not (properties[0].get('SMILES') or properties[0].get('IsomericSMILES')):
            raise ValueError('Require one unambiguous molecular identity')
        record.update(status='downloaded', bytes=len(data), response_sha256=hashlib.sha256(data).hexdigest(),
                      response=body)
    except (requests.RequestException, ValueError, KeyError) as error:
        record.update(status='failed_no_automatic_retry', reason=type(error).__name__ + ': ' + str(error)[:250])
    return record


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        result = json.loads(OUT.read_text())
        if result['candidate_names'] != list(DRUGS):
            raise ValueError('Frozen candidate list changed')
    else:
        result = {'version': 'expanded-clinical-identity-retrieval-1', 'candidate_names': list(DRUGS),
                  'per_response_limit_bytes': LIMIT, 'max_total_bytes': LIMIT * len(DRUGS),
                  'concurrency': 3, 'automatic_retries': 0, 'records': {}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch, drug): drug for drug in DRUGS if drug not in result['records']}
        for job in as_completed(pending):
            row = job.result(); result['records'][row['drug']] = row
            tmp = OUT.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            tmp.replace(OUT)
            print(row['drug'], row['status'], flush=True)
    print(json.dumps({'complete': len(result['records']),
                      'downloaded': sum(r['status'] == 'downloaded' for r in result['records'].values())}))


if __name__ == '__main__':
    main()
