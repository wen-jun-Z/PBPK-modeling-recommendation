"""Same-origin web workspace. Reuses the existing PBPK API and live database."""
from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from uuid import UUID
from urllib.parse import urlsplit

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pbpk_kb.api.main import app as api
from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.storage.db import get_db_session
from pbpk_kb.web.provenance import document_provenance, resolve_pdf

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / 'static'
RUN = ROOT / 'runtime/deployment/workspace'
COOKIE = 'pbpk_workspace'
app = FastAPI(title='PBPK Workspace', docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])
_attempts: list[float] = []


def secret() -> str:
    return (RUN / 'access.key').read_text().strip()


def signed_session(expiry: int) -> str:
    stamp = str(expiry)
    return stamp + '.' + hmac.new(secret().encode(), stamp.encode(), hashlib.sha256).hexdigest()


def authenticated(request: Request) -> bool:
    token = request.cookies.get(COOKIE, '')
    try:
        expiry = int(token.split('.')[0])
        return time.time() < expiry <= time.time() + 86460 and hmac.compare_digest(token, signed_session(expiry))
    except (ValueError, OSError):
        return False


@app.middleware('http')
async def private_access(request: Request, call_next):
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': 'Cross-site request rejected'}, status_code=403)
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail': 'Cross-site request rejected'}, status_code=403)
    if request.url.path.startswith('/api') and not authenticated(request):
        return JSONResponse({'detail': 'Enter your personal access code first'}, status_code=401)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers['Cache-Control'] = ('public, max-age=86400' if request.url.path == '/static/workspace-font.otf' else 'no-store')
    return response


class Login(BaseModel):
    access_code: str = Field(max_length=256)


@app.post('/auth/login')
def login(body: Login):
    now = time.time()
    _attempts[:] = [t for t in _attempts if now - t < 60]
    if len(_attempts) >= 10:
        raise HTTPException(429, 'Too many attempts. Try again in one minute.')
    if not hmac.compare_digest(body.access_code, secret()):
        _attempts.append(now)
        raise HTTPException(401, 'Incorrect access code')
    response = JSONResponse({'ok': True})
    response.set_cookie(COOKIE, signed_session(int(now) + 86400), max_age=86400,
                        httponly=True, samesite='strict')
    return response


@app.post('/auth/logout')
def logout():
    response = JSONResponse({'ok': True})
    response.delete_cookie(COOKIE)
    return response


@app.get('/static/workspace-font.otf')
def workspace_font():
    return FileResponse(ROOT / 'runtime/fonts/NotoSansCJKsc-Regular.otf', media_type='font/otf')


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'pbpk-workspace'}


@api.get('/workspace/summary')
def summary(db: Session = Depends(get_db_session)):
    counts = db.execute(text('''SELECT
      (SELECT count(*) FROM auto_validated_evidence) AS evidence,
      (SELECT count(DISTINCT attributed_compound_id) FROM auto_validated_evidence) AS covered_compounds,
      (SELECT count(*) FROM compound_identity) AS compounds,
      (SELECT count(*) FROM public_pbpk_model_registry) AS families,
      (SELECT count(*) FROM public_pbpk_model_version) AS versions,
      (SELECT count(DISTINCT source_id) FROM auto_validated_evidence) AS evidence_sources
    ''')).mappings().one()
    def rows(sql):
        return [serialize_row(dict(r)) for r in db.execute(text(sql)).mappings()]
    return {'counts': dict(counts),
        'evidence_types': rows('SELECT evidence_type AS label, count(*) AS count FROM auto_validated_evidence GROUP BY 1 ORDER BY 2 DESC'),
        'coverage': rows('SELECT attributed_compound_name AS label, count(*) AS count FROM auto_validated_evidence GROUP BY 1 ORDER BY 2 DESC'),
        'sources': rows('SELECT source_id AS label, count(*) AS count FROM auto_validated_evidence GROUP BY 1 ORDER BY 2 DESC'),
        'component_generation': False,
        'recommendation_method': 'PBPK-Match light research inference and applicability checks',
        'blueprint_design': True}


MODEL_FROM = '''FROM public_pbpk_model_version v
 JOIN public_pbpk_model_registry m ON m.model_id=v.model_id'''
MODEL_WHERE = ''' WHERE (:q='' OR m.model_name ILIKE :pattern OR v.compounds::text ILIKE :pattern
 OR v.version_label ILIKE :pattern)
 AND (:software='' OR v.software=:software)
 AND (:status='' OR v.execution_probe_status=:status)'''


@api.get('/workspace/models')
def models(q: str = '', software: str = '', status: str = '',
           limit: int = Query(24, ge=1, le=1000), offset: int = Query(0, ge=0),
           db: Session = Depends(get_db_session)):
    params = dict(q=q, pattern=f'%{q}%', software=software, status=status, limit=limit, offset=offset)
    total = db.execute(text('SELECT count(*) ' + MODEL_FROM + MODEL_WHERE), params).scalar_one()
    rows = db.execute(text('''SELECT v.model_version_id::text, v.model_id::text, m.model_name,
      v.version_label, v.software, v.compounds, v.species, v.routes, v.populations,
      v.repository_url, v.repository_commit, v.license_spdx_id, v.static_parse_status,
      v.execution_probe_status, v.reusable_candidate, v.simulation_count,
      EXISTS(SELECT 1 FROM public_pbpk_model_validation_run r
       WHERE r.model_version_id=v.model_version_id AND r.scientific_accepted IS TRUE) AS scientific_accepted
    ''' + MODEL_FROM + MODEL_WHERE + ' ORDER BY lower(m.model_name), v.version_label, v.model_version_id LIMIT :limit OFFSET :offset'), params).mappings()
    return {'total': total, 'items': [serialize_row(dict(r)) for r in rows]}


@api.get('/workspace/models/{version_id}')
def model_detail(version_id: UUID, db: Session = Depends(get_db_session)):
    r = db.execute(text('SELECT v.*, m.model_name ' + MODEL_FROM +
                        ' WHERE v.model_version_id=:id'), {'id': version_id}).mappings().first()
    if not r:
        raise HTTPException(404, 'Model version not found')
    checks = db.execute(text('''SELECT validation_run_id::text, engineering_accepted,
        scientific_accepted, pooled_gmfe, verification_dataset_count, validated_at
        FROM public_pbpk_model_validation_run WHERE model_version_id=:id ORDER BY validated_at DESC'''), {'id': version_id}).mappings()
    return {'version': serialize_row(dict(r)), 'validation': [serialize_row(dict(x)) for x in checks]}


@api.get('/workspace/evidence/{evidence_type}/{evidence_id}')
def workspace_evidence(evidence_type: str, evidence_id: UUID, db: Session = Depends(get_db_session)):
    from pbpk_kb.api.evidence_routes import evidence_detail
    record = evidence_detail(evidence_type, str(evidence_id), db)
    record['provenance'] = document_provenance(db, evidence=record)
    return record


@api.get('/workspace/documents/{document_id}')
def workspace_document(document_id: UUID, db: Session = Depends(get_db_session)):
    exists = db.execute(text('SELECT 1 FROM parsed_document WHERE parsed_document_id=:id'),
                        {'id': document_id}).scalar_one_or_none()
    if not exists:
        raise HTTPException(404, 'Source document not found')
    return document_provenance(db, document_id)


def redirect_to_source(provenance):
    priority = {'article': 0, 'pdf_local': 1, 'full_text': 2, 'publisher': 3, 'pdf_remote': 4}
    links = sorted(provenance.get('links', []), key=lambda link: priority.get(link['kind'], 99))
    if not links:
        raise HTTPException(404, 'This evidence record has no verified accessible source page or PDF')
    return RedirectResponse(links[0]['url'], status_code=307)


@api.get('/workspace/evidence/{evidence_type}/{evidence_id}/source')
def evidence_source(evidence_type: str, evidence_id: UUID, db: Session = Depends(get_db_session)):
    return redirect_to_source(workspace_evidence(evidence_type, evidence_id, db)['provenance'])


@api.get('/workspace/documents/{document_id}/source')
def document_source(document_id: UUID, db: Session = Depends(get_db_session)):
    return redirect_to_source(workspace_document(document_id, db))


@api.get('/workspace/source-pdf/{document_id}')
def source_pdf(document_id: UUID, db: Session = Depends(get_db_session)):
    record = db.execute(text('''SELECT raw_storage_path,raw_checksum FROM raw_document
        WHERE raw_document_id=:id'''), {'id': document_id}).mappings().first()
    path = resolve_pdf(dict(record)) if record else None
    if not path:
        raise HTTPException(404, 'Local PDF is missing or failed source verification')
    return FileResponse(path, media_type='application/pdf', filename=path.name,
                        content_disposition_type='inline')


@api.get('/workspace/exports/{job_id}/download')
def download_export(job_id: UUID, db: Session = Depends(get_db_session)):
    path = db.execute(text('SELECT output_path FROM export_job WHERE export_job_id=:id AND status IN (\'completed\', \'succeeded\')'), {'id':job_id}).scalar_one_or_none()
    if not path:
        raise HTTPException(404, 'Export file has not been generated')
    target = (ROOT / path).resolve()
    if not target.is_relative_to((ROOT / 'exports').resolve()) or not target.is_file():
        raise HTTPException(404, 'Export file is unavailable')
    return FileResponse(target, filename=target.name)


from pbpk_kb.web.blueprints import router as workspace_blueprint_router
from pbpk_kb.web.research import router as workspace_research_router
from pbpk_kb.web.recommendations import router as workspace_recommendation_router

api.include_router(workspace_blueprint_router)
api.include_router(workspace_research_router)
api.include_router(workspace_recommendation_router)

app.mount('/api', api)
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')
