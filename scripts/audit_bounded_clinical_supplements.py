"""Fetch only six explicitly published clinical supplements, with strict caps."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/blueprint_design'
OUT = BASE/'external_validation/concentration_curve_audit_20260920/supplement_round2'
TARGETS = {
    'PMC13284695': ['DataSheet1.docx'],
    'PMC10897852': ['mmc1.docx'],
    'PMC5297975': ['CPDD-6-54-s001.docx'],
    'PMC7993260': ['CTS-14-476-s002.docx'],
    'PMC12378308': ['40256_2025_740_MOESM1_ESM.pdf'],
    'PMC9929805': ['atm-11-02-93-rc.pdf','atm-11-02-93-tp.pdf','atm-11-02-93-dss.pdf','atm-11-02-93-coif.pdf'],
}
MAX_NEW = 10*1024**2
MAX_DOCUMENT = 2*1024**2


def cohort_bytes():
    dirs = [BASE/'independent_cohort_20260920'] + [BASE/'external_validation'/x for x in (
        'newstudy_bundle_20260920','additional_native_cohort_20260920','concentration_curve_audit_20260920','remaining_cohort_20260920')]
    return sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'acquisition.json'
    if manifest.exists():
        print('Existing completed audit; no automatic re-download/retry.')
        return
    records=[];spent=0
    for pmc,names in TARGETS.items():
        remaining=min(MAX_NEW-spent,100*1024**2-cohort_bytes())
        if remaining<=0:break
        limit=min(MAX_DOCUMENT,remaining)
        url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/supplementaryFiles'
        row={'pmcid':pmc,'explicit_attachment_names':names,'url':url,'maximum_package_bytes':limit,'files':[]}
        records.append(row);data=bytearray()
        try:
            with requests.get(url,stream=True,timeout=(15,45)) as response:
                if response.status_code==404:
                    row.update(status='no_repository_supplement_package',retry=False);continue
                response.raise_for_status()
                while len(data)<limit:
                    chunk=response.raw.read(min(65536,limit-len(data)))
                    if not chunk:break
                    data.extend(chunk)
            spent+=len(data);row['download_bytes']=len(data)
            if len(data)>=limit:
                row.update(status='document_package_limit_reached_skipped',retry=False);continue
            archive=zipfile.ZipFile(io.BytesIO(data));row['package_member_names']=archive.namelist()
            for name in names:
                if name not in archive.namelist():
                    row['files'].append({'name':name,'status':'explicit_attachment_absent_in_package'});continue
                info=archive.getinfo(name)
                if info.file_size>MAX_DOCUMENT:
                    row['files'].append({'name':name,'status':'uncompressed_document_oversize_skipped'});continue
                if cohort_bytes()+info.file_size>100*1024**2:raise ValueError('100MiB cumulative directory cap')
                content=archive.read(info);p=OUT/'sealed'/pmc/name;p.parent.mkdir(parents=True,exist_ok=True);p.parent.chmod(0o700)
                p.write_bytes(content);p.chmod(0o600)
                row['files'].append({'name':name,'status':'downloaded','path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
            row['status']='explicit_supplements_acquired'
        except Exception as exc:
            if 'download_bytes' not in row:
                spent+=len(data);row['download_bytes']=len(data)
            row.update(status='source_error_stopped',error=str(exc)[:250],retry=False)
            break
    payload={'schema_version':'bounded-clinical-supplement-audit-v1','records':records,'new_download_bytes':spent,
             'maximum_new_download_bytes':MAX_NEW,'maximum_document_bytes':MAX_DOCUMENT,
             'cumulative_cohort_bytes':cohort_bytes(),'maximum_cohort_bytes':100*1024**2,
             'source_documents_not_model_predictions':True,'solver_calls':0,'gpu_used':False}
    manifest.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'articles_attempted':len(records),'download_bytes':spent,'status_counts':{status:sum(r['status']==status for r in records) for status in set(r['status'] for r in records)}},indent=2))


if __name__=='__main__':main()
