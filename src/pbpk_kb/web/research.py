"""Read completed recommendation publications; expose only fixed project artifacts."""
from __future__ import annotations

import csv
from functools import lru_cache
import json
import math
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix='/workspace/research', tags=['workspace-research'])
TABLES = ROOT / 'results/tables'
CURVES = ROOT / 'results/data/pbpk_match_light_curves.json'
FAMILIES = {
    'curve': 'Concentration profile', 'finite_AUC': 'Finite-window AUC', 'Cmax': 'Cmax',
    'Tmax': 'Tmax', 'C_last': 'Last concentration Clast', 'C_last': 'Clast', 'half_life': 'Half-life',
    'AUC_infinity': 'AUC∞', 'CL': 'Clearance', 'Vz': 'Volume of distribution Vz',
    'MRT': 'Mean residence time', 'Vss': 'Steady-state distribution volume',
}
DOWNLOADS = {
    'quality': ('Model recommendation document evaluation CSV', 'results/tables/model_recommendation_quality.csv'),
    'endpoints': ('All endpoints and coverage CSV', 'results/tables/model_recommendation_eleven_endpoints.csv'),
    'curves': ('Concentration curves and rankings JSON', 'results/data/pbpk_match_light_curves.json'),
    'runtime': ('Runtime cost and stability CSV', 'results/tables/model_recommendation_runtime.csv'),
    'paper': ('Model recommendation manuscript PDF', 'results/manuscript/pbpk_paper_zh.pdf'),
}
LIMITATIONS = [
    'Document-ranking metrics cover the 40 of 145 requests with known positives; unknown candidates are not treated as negatives.',
    'The proposed method uses grouped training, while some comparators remain frozen and untrained. Scores are not clinical prediction accuracy.',
    'Concentration curves cover two scenarios and six candidates from one study. Missing results are not substituted, and coverage is reported separately.',
    'New requests use the saved small network for research inference; component augmentation and error correction are disabled by default.',
]


def scoped_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise HTTPException(404, 'The result file is unavailable')
    return path


@lru_cache(maxsize=12)
def _json_cached(name: str, size: int, modified: int):
    return json.loads(scoped_file(name).read_text())


def read_json(path: Path):
    relative = str(path.relative_to(ROOT))
    checked = scoped_file(relative)
    stat = checked.stat()
    return _json_cached(relative, stat.st_size, stat.st_mtime_ns)


def scalar(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        number = float(value)
        return (int(number) if number.is_integer() else number) if math.isfinite(number) else None
    except (ValueError, TypeError):
        return value


def table(name: str):
    with scoped_file('results/tables/' + name).open(newline='') as source:
        return [{key: scalar(value) for key, value in row.items()} for row in csv.DictReader(source)]


def endpoint_families(rows):
    definitions = {}
    for row in rows:
        definitions.setdefault(row['family'], set()).add(row['endpoint_key'])
    return [{'id': key, 'label': FAMILIES.get(key, key), 'definition_count': len(values)}
            for key, values in definitions.items()]


@router.get('/summary')
def summary():
    quality = table('model_recommendation_quality.csv')
    endpoints = table('model_recommendation_eleven_endpoints.csv')
    cases = read_json(CURVES)['cases']
    return {
        'status': 'complete',
        'counts': {'document_requests': 145, 'candidates': 280, 'execution_scenarios': 41,
                   'curve_cases': len(cases), 'methods': len(quality)},
        'quality_rows': quality,
        'endpoint_families': endpoint_families(endpoints),
        'curve_cases': [{**{k: case.get(k) for k in ('compound', 'scenario_id', 'dose_mg', 'route', 'formulation')},
                         'candidate_count': len(case['curves']),
                         'observation_count': len(case['observation']['time_h'])} for case in cases],
        'downloads': [{'key': key, 'label': label, 'url': '/api/workspace/research/download/' + key}
                      for key, (label, _) in DOWNLOADS.items()],
        'limitations': LIMITATIONS,
    }


@router.get('/endpoints')
def endpoints(method_id: str = Query(default='', max_length=100), family: str = Query(default='', max_length=100)):
    rows = table('model_recommendation_eleven_endpoints.csv')
    return {'items': [r for r in rows if (not method_id or r['method_id'] == method_id)
                     and (not family or r['family'] == family)],
            'families': endpoint_families(rows),
            'methods': sorted({r['method_id'] for r in rows})}


@router.get('/curves/{compound}')
def curves(compound: str):
    case = next((c for c in read_json(CURVES)['cases'] if c['compound'] == compound), None)
    if case is None:
        raise HTTPException(404, 'No saved comparable concentration curves exist for this compound')
    obs = case['observation']
    article = obs.get('source_article')
    source_url = article.get('url') if isinstance(article, dict) else article
    if not isinstance(source_url, str) or not source_url.startswith(('https://', 'http://')):
        publication = case.get('publication_cluster_id', '')
        source_url = 'https://pubmed.ncbi.nlm.nih.gov/' + publication.split(':')[1] + '/' if publication.startswith('pmid:') else None
    return {
        **{k: case.get(k) for k in ('compound', 'scenario_id', 'dose_mg', 'route', 'food', 'formulation', 'matrix', 'analyte')},
        'observation': {**{k: obs.get(k) for k in ('time_h', 'concentration_ng_ml', 'sd_ng_ml', 'population_n')},
                        'source_url': source_url},
        'curves': [{**{k: curve[k] for k in ('label', 'candidate_id', 'name', 'time_h', 'concentration_ng_ml', 'gmfe', 'log_rmse')},
                    'model_version_id': curve['candidate_id']} for curve in case['curves']],
        'rankings': case['rankings'],
        'limitations': ['Clinical observations are study-group means; predictions use a model reference individual. These curves were not refitted to the target observations.',
                        'This page shows completed fixed study cases, not simulations for a new request.'],
    }


@router.get('/download/{key}')
def download(key: str):
    if key not in DOWNLOADS:
        raise HTTPException(404, 'Download file not found')
    _, relative = DOWNLOADS[key]
    path = scoped_file(relative)
    return FileResponse(path, filename=path.name, media_type={'csv': 'text/csv', 'pdf': 'application/pdf', 'json': 'application/json'}[path.suffix[1:]])
