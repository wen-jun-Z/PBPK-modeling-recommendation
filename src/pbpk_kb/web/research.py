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
    'curve': '浓度曲线', 'finite_AUC': '有限时间窗 AUC', 'Cmax': 'Cmax',
    'Tmax': 'Tmax', 'C_last': '末次浓度 Clast', 'C_last': 'Clast', 'half_life': '半衰期',
    'AUC_infinity': 'AUC∞', 'CL': '清除率', 'Vz': '分布容积 Vz',
    'MRT': '平均滞留时间', 'Vss': '稳态分布容积',
}
DOWNLOADS = {
    'quality': ('模型推荐文档评价 CSV', 'results/tables/model_recommendation_quality.csv'),
    'endpoints': ('全部终点与覆盖率 CSV', 'results/tables/model_recommendation_eleven_endpoints.csv'),
    'curves': ('浓度曲线与排名 JSON', 'results/data/pbpk_match_light_curves.json'),
    'runtime': ('运行成本与稳定性 CSV', 'results/tables/model_recommendation_runtime.csv'),
    'paper': ('模型推荐论文 PDF', 'results/manuscript/pbpk_paper_zh.pdf'),
}
LIMITATIONS = [
    '文档排序指标基于 145 条请求中有已知正例的 40 条；未知候选不当作负例。',
    '自建方法使用分组训练，对照含未训练的冻结方法；分数不等于临床预测准确率。',
    '浓度曲线只覆盖同一研究的 2 个场景、6 个候选；缺失结果不递补，覆盖率单独报告。',
    '新请求使用已保存的小网络进行研究推理，组件增强和误差修正默认关闭。',
]


def scoped_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise HTTPException(404, '当前结果文件不可用')
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
        raise HTTPException(404, '该化合物没有已保存的可比浓度曲线')
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
        'limitations': ['临床观测为研究组均值，预测为模型参考个体；这些曲线没有按目标观测重新拟合。',
                        '本页是已完成的固定研究案例，不是当前新请求的仿真结果。'],
    }


@router.get('/download/{key}')
def download(key: str):
    if key not in DOWNLOADS:
        raise HTTPException(404, '没有该下载文件')
    _, relative = DOWNLOADS[key]
    path = scoped_file(relative)
    return FileResponse(path, filename=path.name, media_type={'csv': 'text/csv', 'pdf': 'application/pdf', 'json': 'application/json'}[path.suffix[1:]])
