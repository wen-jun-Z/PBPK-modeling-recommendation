"""Private, bounded web access to the completed evidence-to-PBPK workflow.

The frozen time-coverage branch supplies a registered evidence/native-physiology
bundle. Optional *development* observations fit the existing four-axis model;
target-study outcomes are never opened by this module. All new output is kept
under the workspace runtime directory, never in the frozen experiments.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
import threading
import time
from typing import Literal
from urllib.parse import urlsplit, quote
from uuid import UUID, uuid4
import zipfile

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pbpk_kb.blueprint_policy import blood_partition_calibration as model
from pbpk_kb.blueprint_policy import visible_partition_feasibility as fitter
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.web.blueprint_human import rebuild_physiology
from pbpk_kb.web.provenance import blueprint_evidence_url

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'experiments/blueprint_design/sampling_stability_20261009'
RUN = ROOT / 'runtime/deployment/workspace/blueprints'
METHOD = '人体模板＋药物证据＋四参数校准＋时间覆盖补证与可行性修复'
LIMITATIONS = [
    '当前交互范围为已整理的 30 药、成人、单药、单次静脉给药；默认使用案例的预测条件。',
    '新人体条件必须随对应开发观测数据一起提供；18–80 岁仅为人体构建的输入边界，不代表已有校准或验证覆盖。',
    '结果为研究模型预测，改变人体或给药条件不等于完成了新场景的独立科学验证。',
    '开发曲线校准与独立评价分开；未上传开发数据时复用已保存的校准参数。',
    '终末参数仅在数值质量检查通过时报告；缺少独立临床参考的终点不声称已验证。',
]
router = APIRouter(prefix='/workspace', tags=['workspace-blueprints'])
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='pbpk-blueprint')
_slots = threading.BoundedSemaphore(2)
_guard = threading.Lock()
_incarnation = str(uuid4())
Regime = Literal['late_four', 'early_four']


def _path(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raw = ROOT / raw
    # Reject lexically before following symlinks outside this project.
    if not raw.is_relative_to(ROOT) or '..' in raw.parts:
        raise ValueError('来源路径不在本项目范围内')
    path = raw.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('来源路径超出项目范围')
    return path


def _bytes(path: str, modified: int, size: int) -> bytes:
    if size > 64 * 1024 * 1024:
        raise ValueError('输入文件超过交互读取上限')
    return Path(path).read_bytes()


def _read(path, digest=None):
    path = _path(path)
    st = path.stat()
    data = _bytes(str(path), st.st_mtime_ns, st.st_size)
    if digest and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('来源文件校验失败：' + path.name)
    return json.loads(data)


def _ref(reference):
    return _read(reference['path'], reference['sha256'])


def _source_bytes(reference):
    path = _path(reference['path']); st = path.stat()
    content = _bytes(str(path), st.st_mtime_ns, st.st_size)
    if hashlib.sha256(content).hexdigest() != reference['sha256']:
        raise ValueError('证据来源校验失败：' + path.name)
    return content


def _source(reference):
    content = _source_bytes(reference)
    if content.startswith(b'%PDF-'):
        return {'local_pdf': True}
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}



def _save(path, data):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def _catalog():
    state = _read(FROZEN / 'state.json')
    if state['status'] != 'complete':
        raise ValueError('蓝图主线结果尚未完成，不能作为交互基础')
    protocol = _ref(state['protocol'])
    source = _ref(protocol['source_protocol'])
    choices = _ref(protocol['choices'])['choices']
    return state, protocol, source, choices


def _case(case_id):
    state, protocol, source, choices = _catalog()
    matches = [c for c in protocol['cases'] if c['case_id'] == case_id]
    if len(matches) != 1:
        raise HTTPException(404, '未找到该蓝图案例')
    return matches[0], state, source, choices


def _bundle(case_id, regime='late_four'):
    case, state, source, choices = _case(case_id)
    choice = choices[case_id + '__' + regime]['cmax']['sources']['time_repaired']
    reference = state['units'][choice['unit_id']]['result']
    fitted = _ref(reference)
    if fitted['status'] != 'predicted':
        raise ValueError('该案例没有可用的已锁定参数')
    request = EvidenceBlueprintRequest.model_validate(_ref(fitted['request'])).model_dump(mode='json', exclude_none=True)
    formula = fitted['formula']
    physref = source['physiology_cache'][f'{case_id}__target__{formula}']
    physiology = _ref(physref)
    # Audit the evidence ledger and original native inputs without launching PK-Sim.
    for item in request['evidence']:
        _source(item['source'])
    _ref(request['evidence_ledger'])
    for relative, sha in physiology['source_pins'].items():
        p = _path(relative); st = p.stat()
        if hashlib.sha256(_bytes(str(p), st.st_mtime_ns, st.st_size)).hexdigest() != sha:
            raise ValueError('人体生理来源校验失败：' + p.name)
    return dict(case=case, request=request, physiology=physiology, fitted=fitted,
                choice=choice, result_reference=reference, physiology_reference=physref,
                saved_curve=_ref(fitted['prediction']), training=_ref(fitted['training']))


def _safe_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    return value.strip() if parsed.scheme in ('https', 'http') and parsed.hostname and not parsed.username and not parsed.password else None


def _canonical_source_url(url, bundle):
    parsed = urlsplit(url)
    if parsed.hostname == 'pubchem.ncbi.nlm.nih.gov':
        match = re.search(r'/compound/(?:cid/)?(\d+)(?:/|$)', parsed.path)
        if match:
            identity = match.group(1)
        elif bundle['request']['compound_id'].startswith('pubchem:'):
            identity = bundle['request']['compound_id'].split(':', 1)[1]
        else:
            identity = quote(bundle['request']['compound_name'], safe='')
        return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + identity
    return url


def _source_link(bundle, row):
    reference = row['source']
    content = _source_bytes(reference)
    if content.startswith(b'%PDF-'):
        return f'/api/workspace/blueprint-cases/{bundle["case"]["case_id"]}/source/{reference["sha256"]}'
    url = blueprint_evidence_url(content, row, bundle['request']['compound_name'],
                                 bundle['request']['compound_id'])
    return _canonical_source_url(url, bundle) if url else None


def _sources(bundle):
    result = []
    seen = set()
    for e in bundle['request']['evidence']:
        url = _source_link(bundle, e)
        if url and url not in seen:
            result.append({'label': '参数证据原文 PDF' if url.startswith('/') else '参数来源', 'url': url})
            seen.add(url)
    return result[:30]


def _evidence_rows(bundle, request=None):
    return [{k: row.get(k) for k in ('parameter', 'value', 'unit', 'basis', 'notes')} |
            {'source_url': _source_link(bundle, row)}
            for row in (request or bundle['request'])['evidence']]


def _display_curve(curve, mw, maximum=700):
    n = len(curve['time_min'])
    # Keep the actual peak as well as evenly spaced samples for the display.
    index = sorted(set(np.linspace(0, n-1, min(n, maximum), dtype=int).tolist() +
                       [int(np.argmax(curve['concentration']))]))
    return {'time_h': [curve['time_min'][i]/60 for i in index],
            'concentration_ng_ml': [curve['concentration'][i]*mw for i in index],
            'unit': 'ng/mL', 'full_point_count': n}


def _presentation(bundle, request=None, curve=None, parameters=None, calibration=None):
    req = request or bundle['request']
    curve = curve or bundle['saved_curve']
    pars = parameters or bundle['fitted']['parameters']
    mw = next(float(e['value']) for e in req['evidence'] if e['parameter'] == 'molecular_weight')
    phys = bundle['physiology']
    names = [('clearance_scale','非肾清除率倍率'),('partition_scale','组织分配倍率'),
             ('bidirectional_exchange_scale','双向交换倍率'),('blood_partition_scale','血细胞分配倍率')]
    evidence = _evidence_rows(bundle, req)
    summary = summarize_extended_prediction(curve, req)
    visible = summary['endpoints']['plasma_concentration_time']
    display_curve = {'time_min': [t*60 for t in visible['time_h']],
                     'concentration': visible['concentration_umol_per_l']}
    return {'case_id': bundle['case']['case_id'], 'drug': bundle['case']['drug'],
            'method': METHOD, 'request': {k: req[k] for k in ('human','dosing','target','partition_method')},
            'structure': {'organ_count': len(phys['organs']),
                'organs': [{k: r[k] for k in ('name','volume_l','flow_l_h')} for r in phys['organs']],
                'processes': ['动脉、静脉与门静脉循环','15 个器官的血管—组织交换','组织与血细胞分配',
                              '肾清除及非肾清除','静脉给药与血浆浓度输出']},
            'parameters': [{'name': k,'label': label,'value': pars['parameters'][k],
                            'unit': '倍','basis': 'development_calibrated'} for k,label in names],
            'evidence': evidence, 'sources': _sources(bundle),
            'curve': _display_curve(display_curve, mw), 'endpoints': summary['endpoints'],
            'endpoint_summary': summary,
            'calibration': calibration or {'mode': 'existing','visible_points': len(bundle['training']['observations']),
                'added_time_h': bundle['choice']['time_h'], 'development_gmfe': pars.get('development_gmfe'),
                'human': deepcopy(bundle['training']['human']), 'dosing': deepcopy(bundle['training']['dosing']),
                'study_id': bundle['training'].get('study_id'),
                'source_label': bundle['training'].get('source_label') or bundle['training'].get('study_id') or '已有开发研究',
                'source': '已保存的时间覆盖补证与容量修复分支；非新训练'},
            'human_context': {'changed': False, 'reference_human': deepcopy(req['human']),
                'reference_kind': 'case_target',
                'active_human': deepcopy(req['human']), 'physiology_source': 'case',
                'calibration_mode': 'case_saved'},
            'limitations': LIMITATIONS, 'scientific_accepted': False}


class Point(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    time_h: float = Field(ge=0, le=168)
    concentration_ng_ml: float = Field(gt=0, le=1e9)


class HumanConditions(BaseModel):
    """Declared observation demographics; population and seed stay case-defined."""
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    age_years: float = Field(ge=18, le=80)
    sex: Literal['MALE', 'FEMALE']
    weight_kg: float = Field(gt=0, le=300)
    height_cm: float = Field(gt=0, le=250)


class DevelopmentCurve(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    source_label: str = Field(min_length=3, max_length=300)
    source_url: str | None = Field(default=None, max_length=2000)
    human: HumanConditions
    dose_mg: float = Field(gt=0, le=100000)
    infusion_minutes: float = Field(ge=0, le=10080)
    observations: list[Point] = Field(min_length=5, max_length=128)

    @model_validator(mode='after')
    def validate_curve(self):
        if self.source_url and not _safe_url(self.source_url):
            raise ValueError('开发曲线来源仅接受 HTTP(S) 网页链接')
        if any(b.time_h <= a.time_h for a,b in zip(self.observations, self.observations[1:])):
            raise ValueError('开发曲线时间必须严格递增')
        if self.infusion_minutes / 60 >= self.observations[-1].time_h:
            raise ValueError('开发曲线必须包含输注完成后的观测')
        if self.infusion_minutes and self.observations[0].time_h == 0:
            raise ValueError('输注前零时刻不能有正浓度')
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    case_id: str = Field(min_length=1, max_length=140, pattern=r'^[a-zA-Z0-9_-]+$')
    regime: Regime = 'late_four'
    dose_mg: float = Field(gt=0, le=100000)
    infusion_minutes: float = Field(ge=0, le=10080)
    window_end_h: float = Field(gt=0, le=168)
    human: HumanConditions | None = None
    development_curve: DevelopmentCurve | None = None

    @model_validator(mode='after')
    def valid_window(self):
        if self.infusion_minutes / 60 >= self.window_end_h:
            raise ValueError('预测时间窗必须超过输注完成时间')
        if self.human is not None and self.development_curve is not None and self.human != self.development_curve.human:
            raise ValueError('新预测人体必须与开发观测数据声明的人体条件一致')
        return self


def _effective_human(body, bundle):
    human = deepcopy(bundle['request']['human'])
    declared = body.development_curve.human if body.development_curve else body.human
    if declared is not None:
        human.update(declared.model_dump())
    if human != bundle['request']['human'] and body.development_curve is None:
        raise ValueError('修改人体条件需提供该人体对应的开发观测曲线；没有新观测数据时请使用案例默认条件')
    return human


def _run_prediction(body, bundle, update, output=None):
    begin = time.monotonic()
    deadline = begin + 180
    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('超过本阶段计算时限；任务已停止，不会后台持续重试')
    req = deepcopy(bundle['request'])
    req['human'] = _effective_human(body, bundle)
    human_changed = req['human'] != bundle['request']['human']
    # Preserve the formula selected before this interactive request. Some frozen
    # request documents predate formula selection and carry a different default.
    req['partition_method'] = bundle['fitted']['formula']
    req['dosing'].update(dose_mg=body.dose_mg, infusion_minutes=body.infusion_minutes)
    req['target'].update(window_start_h=0., window_end_h=body.window_end_h)
    req['target'].pop('terminal_fit', None)
    parameters = deepcopy(bundle['fitted']['parameters'])
    physiology = bundle['physiology']
    if human_changed:
        if output is None:
            raise ValueError('自定义人体需要独立的项目运行目录')
        # Each native subprocess has its own timeout and byte limit. Construction,
        # administration audit and physiology export together get 600 s; fitting
        # and prediction get a separate 180 s budget after construction completes.
        deadline = begin + 600
        update('building_human', '正在按新人体条件计算器官体积、血流量及分配参数')
        physiology = rebuild_physiology(req, _path(output), ROOT, before_trial=guard)
        guard()
        deadline = time.monotonic() + 180
    calibration = None
    human_context = {'changed': human_changed,
        'reference_human': deepcopy(bundle['request']['human']),
        'reference_kind': 'case_target',
        'active_human': deepcopy(req['human']),
        'physiology_source': 'rebuilt' if human_changed else 'case',
        'calibration_mode': 'case_saved'}
    if body.development_curve:
        update('calibrating', '正在用上传的开发曲线校准四参数')
        dev = body.development_curve
        fit_req = deepcopy(req)
        # Metadata is explicitly declared with the observations and validated
        # against the prediction human. Never relabel an old curve as new data.
        fit_req['human'].update(dev.human.model_dump())
        fit_req['dosing'].update(dose_mg=dev.dose_mg, infusion_minutes=dev.infusion_minutes)
        mw = next(float(e['value']) for e in fit_req['evidence'] if e['parameter']=='molecular_weight')
        training = {'compound_id': fit_req['compound_id'], 'human': fit_req['human'],
            'dosing': fit_req['dosing'], 'matrix':'plasma','time_unit':'min','unit':'umol/l',
            'role':'user_declared_development', 'source_label':dev.source_label, 'source_url':dev.source_url,
            'observations':[{'time_min': p.time_h*60,'concentration_umol_per_l':p.concentration_ng_ml/mw}
                            for p in dev.observations]}
        # Formula and rebuilt human physiology are fixed before reading outcomes.
        # No target-study observation or historical best-fit warm start is used.
        fitted = fitter.fit(lambda x,t: model.simulate(physiology, fit_req, t, x),
            fit_req, training, physiology, budget=4096, starts=4, before_trial=guard)
        if not fitted['best']:
            raise ValueError('在固定范围和计算预算内未得到可行的开发曲线拟合')
        parameters = fitted['best']
        calibration = {'mode':'user_development','visible_points':len(dev.observations),
            'source_label':dev.source_label,'source_url':dev.source_url,'source': '用户提供的开发曲线',
            'human': deepcopy(training['human']), 'dosing': deepcopy(training['dosing']),
            'development_gmfe':parameters['development_gmfe'], 'calls':fitted['calls'],
            'max_calls':4096,'starts':4,'formula_selection':'预先固定，未根据目标结果选择',
            'target_observations_used':False,'fit':fitted}
        human_context['calibration_mode'] = 'development_refit'
    guard(); update('predicting', '正在构建目标给药方案并求解浓度曲线')
    end = body.window_end_h
    # Bounded dense grid with early distribution and exact infusion discontinuity.
    times = np.unique(np.r_[0., np.geomspace(min(1e-5,end/10000),end,700),
                            np.linspace(0,end,401),body.infusion_minutes/60])
    prediction = model.simulate(physiology, req, times, parameters['log_parameters'])
    guard()
    mw = next(float(e['value']) for e in req['evidence'] if e['parameter']=='molecular_weight')
    curve = {'time_min':(times*60).tolist(),
             'concentration':(np.asarray(prediction['concentration_mg_l'])*1000/mw).tolist(),
             'time_unit':'min','unit':'umol/l','observation':prediction['observation']}
    result = _presentation({**bundle, 'physiology': physiology}, req, curve, parameters, calibration)
    result['human_context'] = human_context
    result['mass_balance'] = prediction['mass_balance']
    result['elapsed_seconds'] = time.monotonic()-begin
    result['scenario_changed'] = human_changed or req['dosing'] != bundle['request']['dosing'] or req['target'] != bundle['request']['target']
    source_physiology = (physiology.get('manifest') or {'embedded_in': 'physiology',
        'source_execution_manifest_sha256': physiology['source_execution_manifest_sha256']}
        if human_changed else bundle['physiology_reference'])
    package = {'version':'workspace-blueprint-2','method':METHOD,'request':req,'parameters':parameters,
               'physiology':physiology,'curve':curve,'calibration':result['calibration'],
               'human_context':human_context,
               'source_result':bundle['result_reference'],'source_physiology':source_physiology,
               'input':body.model_dump(),'scientific_accepted':False,'limitations':LIMITATIONS}
    return result, package


def _job_path(job_id):
    return RUN / str(UUID(str(job_id))) / 'job.json'


def _worker(job_id, body, bundle):
    path = _job_path(job_id)
    def update(stage, label):
        doc = _read(path); doc.update(status='running', stage=stage, stage_label=label)
        _save(path, doc)
    try:
        result, package = _run_prediction(body,bundle,update,output=path.parent/'human')
        _save(path.parent/'blueprint.json', package)
        doc = _read(path)
        result['download_url'] = f'/api/workspace/blueprint-predictions/{job_id}/download'
        doc.update(status='completed',stage='completed',stage_label='蓝图与预测已生成',
                   completed_at=time.time(),result=result,download_url=result['download_url'])
        _save(path,doc)
    except Exception as exc:
        doc = _read(path)
        message = str(exc) if isinstance(exc,(ValueError,TimeoutError,OSError,HTTPException)) else '计算失败，请检查输入与本项目服务日志'
        doc.update(status='failed',stage='failed',stage_label='生成未完成',error=message,completed_at=time.time())
        _save(path,doc)
    finally:
        _slots.release()


def _load_job(job_id):
    path = _job_path(job_id)
    if not path.is_file():
        raise HTTPException(404,'蓝图任务不存在')
    result = _read(path)
    if result['status'] in ('queued','running') and result.get('service_instance') != _incarnation:
        result.update(status='interrupted',stage='interrupted',error='服务重启中断了该任务，请重新提交')
    result.pop('service_instance',None)
    return result


@router.get('/blueprint-cases')
def cases():
    try:
        _,p,_,_ = _catalog()
        items = []
        for c in p['cases']:
            req = _ref(c['requests']['evidence']['target'])
            items.append({'case_id':c['case_id'],'drug':c['drug'],'study_id':c['study_id'],
                'human':req['human'],'route':req['dosing']['route'],
                'dose_mg':req['dosing']['dose_mg'],'infusion_minutes':req['dosing']['infusion_minutes'],
                'window_end_h':req['target']['window_end_h'],'status':'ready'})
        return {'items':sorted(items,key=lambda r:r['drug'].lower()),'total':len(items),'method':METHOD,
                'scope':LIMITATIONS[0],'regimes':['late_four','early_four'],'development_calibration':True}
    except (ValueError,OSError,KeyError) as exc:
        raise HTTPException(503,str(exc)) from exc


@router.get('/blueprint-cases/{case_id}')
def case_detail(case_id: str, regime: Regime='late_four'):
    try:
        result = _presentation(_bundle(case_id,regime))
        result['download_url'] = f'/api/workspace/blueprint-cases/{case_id}/download?regime={regime}'
        return result
    except (ValueError,OSError,KeyError) as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/blueprint-predictions',status_code=202)
def create_prediction(body: PredictionRequest):
    if not _slots.acquire(blocking=False):
        raise HTTPException(429,'已有两个蓝图任务运行中，完成后可继续提交')
    try:
        bundle = _bundle(body.case_id,body.regime)
        try:
            _effective_human(body, bundle)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        with _guard:
            RUN.mkdir(parents=True,exist_ok=True,mode=0o700)
            if sum(1 for _ in RUN.glob('*/job.json')) >= 200:
                raise HTTPException(409,'已保存 200 个交互任务；请先导出并整理项目蓝图运行目录')
            job_id = str(uuid4())
            doc = {'job_id':job_id,'status':'queued','stage':'queued','stage_label':'等待开始',
                   'drug':bundle['case']['drug'],'created_at':time.time(),'input':body.model_dump(),
                   'service_instance':_incarnation,'status_url':f'/api/workspace/blueprint-predictions/{job_id}'}
            _save(_job_path(job_id),doc)
        _pool.submit(_worker,job_id,body,bundle)
        return {k:v for k,v in doc.items() if k!='service_instance'}
    except (ValueError, OSError, KeyError) as exc:
        _slots.release()
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        _slots.release()
        raise


@router.get('/blueprint-predictions')
def prediction_history(limit:int=Query(30,ge=1,le=200)):
    paths = sorted(RUN.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:limit]
    rows=[]
    for p in paths:
        row=_load_job(p.parent.name)
        rows.append({k:v for k,v in row.items() if k!='result'})
    return {'items':rows,'total':sum(1 for _ in RUN.glob('*/job.json'))}


@router.get('/blueprint-predictions/{job_id}')
def prediction_status(job_id: UUID):
    doc = _load_job(job_id)
    if doc['status'] == 'completed' and doc.get('result'):
        # Refresh presentation links in historical jobs without changing saved
        # parameters, predictions, provenance assets or their original files.
        body = doc['input']
        bundle = _bundle(body['case_id'], body.get('regime', 'late_four'))
        doc['result']['evidence'] = _evidence_rows(bundle)
        doc['result']['sources'] = _sources(bundle)
    return doc


def _csv(package):
    out=io.StringIO();writer=csv.writer(out);writer.writerow(['time_h','concentration_ng_ml'])
    mw=next(float(e['value']) for e in package['request']['evidence'] if e['parameter']=='molecular_weight')
    visible=summarize_extended_prediction(package['curve'],package['request'])['endpoints']['plasma_concentration_time']
    for t,c in zip(visible['time_h'],visible['concentration_umol_per_l']):
        writer.writerow([t,c*mw])
    return out.getvalue()


def _download(package, stem, format):
    blob=json.dumps(package,ensure_ascii=False,allow_nan=False,indent=2).encode()
    if format=='json': data,media=blob,'application/json'
    elif format=='csv': data,media=_csv(package).encode('utf-8-sig'),'text/csv; charset=utf-8'
    else:
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('blueprint.json',blob)
            archive.writestr('concentration.csv',_csv(package))
            archive.writestr('README.txt','研究用 PBPK 蓝图包\n在本项目环境中执行 python replay.py 可重算浓度曲线。\n参数、证据、生理结构和输入保存在 blueprint.json。\n本包复用既有四参数求解器；不表示独立科学验证或临床适用许可。\n')
            archive.writestr('replay.py',"from pathlib import Path\nimport json\nfrom pbpk_kb.blueprint_policy.blood_partition_calibration import simulate\np=json.loads(Path('blueprint.json').read_text())\nr=simulate(p['physiology'],p['request'],[t/60 for t in p['curve']['time_min']],p['parameters']['log_parameters'])\nPath('replayed_prediction.json').write_text(json.dumps(r, default=lambda x: x.tolist(), allow_nan=False))\n")
            for name in ('blood_partition_calibration.py','permeability_model.py','perfusion_model.py',
                         'perfusion_calibration.py','distribution_calibration_optimizer.py','visible_partition_feasibility.py'):
                archive.writestr('method_sources/'+name,(ROOT/'src/pbpk_kb/blueprint_policy'/name).read_bytes())
        data,media=out.getvalue(),'application/zip'
    return Response(data,media_type=media,headers={'Content-Disposition':f'attachment; filename="{stem}.{format}"'})


@router.get('/blueprint-predictions/{job_id}/download')
def download_prediction(job_id:UUID,format:Literal['json','csv','zip']='zip'):
    job=_load_job(job_id)
    if job['status']!='completed':
        raise HTTPException(409,'蓝图任务尚未完成')
    return _download(_read(_job_path(job_id).parent/'blueprint.json'),f'blueprint-{job_id}',format)


@router.get('/blueprint-cases/{case_id}/download')
def download_case(case_id:str,regime:Regime='late_four',format:Literal['json','csv','zip']='zip'):
    b=_bundle(case_id,regime)
    package={'version':'workspace-blueprint-1','method':METHOD,'request':b['request'],
             'physiology':b['physiology'],'parameters':b['fitted']['parameters'],'curve':b['saved_curve'],
             'source_result':b['result_reference'],'source_physiology':b['physiology_reference'],
             'scientific_accepted':False,'limitations':LIMITATIONS}
    return _download(package,'blueprint-'+case_id,format)


@router.get('/blueprint-cases/{case_id}/source/{source_hash}')
def case_source_pdf(case_id: str, source_hash: str):
    b = _bundle(case_id)
    refs = [row['source'] for row in b['request']['evidence']]
    for ref in refs:
        value = _source(ref) if ref['sha256'] == source_hash else {}
        if isinstance(value, dict) and value.get('local_pdf'):
            path = _path(ref['path'])
            return FileResponse(path, media_type='application/pdf', filename=path.name,
                                content_disposition_type='inline')
    raise HTTPException(404, '该案例未登记此 PDF 来源')
