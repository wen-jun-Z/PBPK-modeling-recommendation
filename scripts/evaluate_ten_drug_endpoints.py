#!/usr/bin/env python3
"""Evaluate the frozen evidence builder across ten drugs and declared IV scenarios.

Clinical observations and hypothetical scenario predictions are separate. No
clinical outcome selects a parameter, partition method, or simulation horizon.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, run_evidence_blueprint
from pbpk_kb.blueprints.evidence_evaluation import aggregate_cases, observed_rows, score_case
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/blueprint_design/multidrug_endpoints_20260923'
BATCH = ROOT / 'experiments/blueprint_design/ten_drug_endpoints_20260923'
DRUGS = ('caffeine', 'midazolam', 'fluconazole', 'alfentanil', 'alprazolam',
         'sufentanil', 'moxifloxacin', 'amikacin', 'vancomycin', 'furosemide')
ENDPOINTS = ['plasma_concentration_time', 'auc', 'cmax', 'tmax', 'c_last',
             'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss']
TERMINAL = ENDPOINTS[5:]
FIELDS = {'auc': ('value_mg_h_per_l', 'mg*h/l'), 'cmax': ('value_mg_per_l', 'mg/l'),
          'tmax': ('value_h', 'h'), 'c_last': ('value_mg_per_l', 'mg/l'),
          'terminal_half_life': ('value_h', 'h'), 'auc_inf': ('value_mg_h_per_l', 'mg*h/l'),
          'clearance': ('value_l_per_h', 'l/h'), 'vz': ('value_l', 'l'),
          'mrt': ('value_h', 'h'), 'vss': ('value_l', 'l')}


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(path.relative_to(ROOT))


def slim(value):
    if isinstance(value, dict):
        return {k: slim(v) for k, v in value.items()
                if k not in {'fit_times_h', 'concentration_umol_per_l', 'concentration_mg_per_l', 'time_h'}}
    if isinstance(value, list):
        return [slim(v) for v in value]
    return value


def prepare():
    choice_path = OLD / 'partition_selection/choice_locked.json'
    choices = load(choice_path)['choices']
    cases, files = [], [choice_path]
    for drug in DRUGS:
        cohort_path = (OLD if drug in DRUGS[:4] else BATCH) / 'inputs' / drug / 'cohort.json'
        cohort = load(cohort_path)
        files.append(cohort_path)
        clinical = []
        for record in cohort['cases']:
            request_path = inside(record['request_path'], ROOT, exists=True)
            observation_path = inside(record['observations_path'], ROOT, exists=True)
            request = deepcopy(load(request_path))
            if request['compound_id'] != cohort['compound_id'] or request['dosing']['route'] != 'intravenous':
                raise ValueError('Single-dose IV identity mismatch: ' + drug)
            times = [r['time_min'] for r in load(observation_path)['observations']]
            if len(times) < 2 or times[-1] <= 0 or any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError('Invalid locked sampling schedule')
            if record['role'] not in {'calibration_data_reconstruction', 'separate_study_development_comparison'}:
                raise ValueError('Clinical case needs an explicit developmental role')
            request['partition_method'] = choices.get(cohort['compound_id'], {}).get('selected_partition_method', 'PKSimStandard')
            request['target'] = {'endpoints': ENDPOINTS.copy(), 'primary_endpoint': 'plasma_concentration_time',
                                 'window_start_h': request['dosing'].get('start_h', 0.), 'window_end_h': times[-1] / 60}
            row = {**record, 'drug': drug, 'compound_id': cohort['compound_id'], 'compound_name': cohort['compound_name'],
                   'scenario_kind': 'clinical_study', 'condition': 'reported_study_with_declared_assumptions',
                   'request': request, 'observation_times_min': times}
            clinical.append(row)
            files.extend([request_path, observation_path, inside(request['evidence_ledger']['path'], ROOT, exists=True)])
            files.extend(inside(e['source']['path'], ROOT, exists=True) for e in request['evidence'])
            for reference in record.get('endpoint_references', {}).values():
                source = reference['source']
                path = inside(source['path'], ROOT, exists=True)
                if sha256(path) != source['sha256']:
                    raise ValueError('Published endpoint source hash mismatch')
                files.append(path)
        if not clinical:
            raise ValueError('Every drug requires at least one clinical curve: ' + drug)
        evidence_sets = {json.dumps(r['request']['evidence'], sort_keys=True) for r in clinical}
        if len(evidence_sets) != 1:
            raise ValueError('Keep one fixed drug evidence ledger across scenarios: ' + drug)
        forbidden = []
        for case in clinical:
            if case['role'] == 'separate_study_development_comparison':
                op = ROOT / case['observations_path']
                forbidden.append({'path': rel(op), 'sha256': sha256(op), 'location': 'comparison outcomes; not construction evidence'})
        for case in clinical:
            refs = {r['path']: r for r in case['request'].get('forbidden_sources', [])}
            refs.update({r['path']: r for r in forbidden})
            case['request']['forbidden_sources'] = list(refs.values())
        cases.extend(clinical)
        reference = next((c for c in clinical if c['role'] == 'calibration_data_reconstruction'), clinical[0])
        for factor in (.5, 1., 2.):
            for condition in ('reference_male', 'longer_infusion_male', 'reference_female'):
                req = deepcopy(reference['request'])
                req['human'].update(age_years=30., weight_kg=70., height_cm=175., sex='MALE')
                if condition == 'reference_female':
                    req['human'].update(weight_kg=60., height_cm=165., sex='FEMALE')
                req['dosing']['dose_mg'] *= factor
                req['dosing']['start_h'] = 0.
                if condition == 'longer_infusion_male':
                    old = req['dosing'].get('infusion_minutes', 0.)
                    req['dosing']['infusion_minutes'] = 2 * old if old > 0 else 30.
                req['target'].update(window_start_h=0., window_end_h=24.)
                req['assumptions'] = [*req.get('assumptions', []),
                    'Hypothetical single-dose scenario, not a clinical dosing recommendation or measured cohort.',
                    'Explicit scenario human fields override any reference-person demographics mentioned in source assumptions.',
                    'Three dose multiples and three input profiles are declared before running; no observation is transferred from another condition.']
                case_id = f'{drug}_grid_d{str(factor).replace(".", "p")}_{condition}'
                cases.append({'case_id': case_id, 'drug': drug, 'compound_id': cohort['compound_id'],
                              'compound_name': cohort['compound_name'], 'role': 'prediction_only',
                              'study_id': None, 'scenario_kind': 'hypothetical_grid', 'condition': condition,
                              'dose_multiplier': factor, 'reference_case_id': reference['case_id'],
                              'observations_path': None, 'request': req})
    ids, roles = set(), {}
    for case in cases:
        cid = case['case_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', cid) or cid in ids:
            raise ValueError('Duplicate or unsafe case identifier')
        ids.add(cid)
        if case['study_id'] is not None:
            key = (case['compound_id'], case['study_id'])
            if key in roles and roles[key] != case['role']:
                raise ValueError('A clinical study cannot cross calibration and comparison roles')
            roles[key] = case['role']
        req = EvidenceBlueprintRequest.model_validate(case['request']).model_dump(mode='json', exclude_none=True)
        assembled = assemble_evidence_blueprint(req, ROOT)
        if assembled['status'] != 'assembled':
            raise ValueError(f'{cid}: construction gaps {assembled["gaps"]}')
        case['request'] = req
    files.extend((ROOT / 'src/pbpk_kb/blueprints').glob('*.py'))
    files.extend((ROOT / 'src/pbpk_kb/model_runtime/R').glob('*.R'))
    files.append(Path(__file__).resolve())
    return cases, {rel(p): sha256(p) for p in set(files)}


def export_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def base_columns(case):
    req = case['request']
    return {'case_id': case['case_id'], 'drug': case['drug'], 'role': case['role'],
            'study_id': case['study_id'], 'scenario_kind': case['scenario_kind'], 'condition': case['condition'],
            'dose_mg': req['dosing']['dose_mg'], 'infusion_minutes': req['dosing'].get('infusion_minutes', 0.),
            'age_years': req['human']['age_years'], 'sex': req['human']['sex'],
            'weight_kg': req['human']['weight_kg'], 'height_cm': req['human']['height_cm'],
            'partition_method': req['partition_method'], 'report_start_h': req['target']['window_start_h'],
            'report_end_h': req['target']['window_end_h']}


def make_reports(output, protocol, rows):
    """Decode clinical values only after all predictions are locked."""
    cases = {c['case_id']: c for c in protocol['cases']}
    wide, endpoint_rows, comparison_rows = [], [], []
    for row in rows:
        case = cases[row['case_id']]
        base = base_columns(case)
        predictions = row.get('endpoint_values', {})
        curve_path = row.get('prediction_path')
        case_wide = {**base, 'prediction_status': row['status'], 'curve_json': curve_path,
                     'curve_csv': row.get('curve_csv'), 'simulation_end_h': row.get('simulation_end_h')}
        for ep, (field, unit) in FIELDS.items():
            item = predictions.get(ep, {})
            case_wide[f'{ep} [{unit}]'] = item.get(field)
            case_wide[f'{ep}_status'] = item.get('status', 'completed' if field in item else 'unavailable')
        wide.append(case_wide)
        metrics, scoring_error = None, None
        if curve_path and sha256(ROOT / curve_path) != row['prediction_sha256']:
            raise ValueError('Locked prediction changed before scoring')
        if curve_path and case['observations_path'] and case.get('curve_evaluation_eligible', True):
            scoring_request = deepcopy(case['request'])
            times = case['observation_times_min']
            if len(times) >= 4:
                scoring_request['target']['terminal_fit'] = {
                    'start_h': times[-4] / 60, 'end_h': times[-1] / 60, 'min_points': 4,
                    'min_adjusted_r2': .98, 'min_span_half_lives': 1.,
                    'max_auc_extrapolated_fraction': .2, 'max_aumc_extrapolated_fraction': .2}
            try:
                metrics = score_case(load(ROOT / curve_path), load(ROOT / case['observations_path']), scoring_request)
                save(output / 'cases' / row['case_id'] / 'clinical_comparison.json', metrics)
            except Exception as exc:
                scoring_error = type(exc).__name__ + ': ' + str(exc)
        row['metrics'] = metrics or {}
        row['scoring_status'] = 'completed' if metrics else ('not_applicable' if not case['observations_path'] or not case.get('curve_evaluation_eligible', True) else 'failed')
        row['scoring_error'] = scoring_error
        if metrics:
            comparison_rows.append({**base, 'point_count': metrics['point_count'],
                'curve_gmfe': metrics['gmfe'], 'curve_log_rmse': metrics['log_rmse'],
                'auc_fold_error': metrics['auc']['fold_error'], 'sampled_cmax_fold_error': metrics['sampled_cmax']['fold_error'],
                'sampled_tmax_absolute_error_h': metrics['sampled_tmax']['absolute_error_h'],
                'c_last_fold_error': metrics['c_last']['fold_error'],
                'terminal_half_life_fold_error': metrics['terminal_half_life'].get('fold_error')})
        mw_factor = next(e['value'] for e in case['request']['evidence'] if e['parameter'] == 'molecular_weight') / 1000
        for ep in ENDPOINTS:
            field, unit = FIELDS.get(ep, ('', 'curve'))
            item = predictions.get(ep, {})
            out = {**base, 'endpoint': ep, 'unit': unit,
                   'predicted_value': item.get(field) if field else None,
                   'predicted_curve_csv': row.get('curve_csv') if ep == ENDPOINTS[0] else None,
                   'prediction_status': item.get('status', 'completed' if (field in item or ep == ENDPOINTS[0] and curve_path) else 'unavailable'),
                   'evaluation_prediction': None, 'observed_value': None, 'error': None, 'error_metric': None,
                   'evaluation_start_h': None, 'evaluation_end_h': None,
                   'evaluation_status': 'no_observation' if not case['observations_path'] else 'no_qualified_endpoint_reference',
                   'blank_reason': 'No matched clinical observation' if not case['observations_path'] else 'No independently qualified endpoint reference',
                   'prediction_missing_reason': item.get('reason'), 'reference_source': None,
                   'evaluation_definition': None}
            if metrics and ep == 'plasma_concentration_time':
                out.update(error=metrics['gmfe'], error_metric='GMFE', evaluation_status='scored', blank_reason=None)
            elif metrics and ep in {'auc', 'cmax', 'tmax', 'c_last'}:
                key = {'cmax': 'sampled_cmax', 'tmax': 'sampled_tmax'}.get(ep, ep)
                m = metrics[key]
                if ep == 'auc':
                    p, o = m['predicted_umol_h_l'] * mw_factor, m['observed_umol_h_l'] * mw_factor
                elif ep == 'tmax':
                    p, o = m['predicted_h'], m['observed_h']
                else:
                    p, o = m['predicted_umol_l'] * mw_factor, m['observed_umol_l'] * mw_factor
                out.update(evaluation_prediction=p, observed_value=o,
                           error=m.get('fold_error', m.get('absolute_error_h')),
                           error_metric='absolute_error_h' if ep == 'tmax' else 'fold_error',
                           evaluation_status='scored_at_observation_times', blank_reason=None)
                out['evaluation_definition'] = m['definition']
            elif metrics and ep == 'terminal_half_life':
                m = metrics[ep]
                out.update(evaluation_prediction=m['predicted'].get('value_h'), observed_value=m['observed'].get('value_h'))
                if m['status'] == 'completed':
                    out.update(error=m['fold_error'], error_metric='fold_error', evaluation_status='scored_at_observation_times', blank_reason=None)
                else:
                    out.update(evaluation_status='terminal_reference_not_qualified', blank_reason='Observed and matched predicted terminal estimates must both pass fixed last-four-point gates')
            if metrics and out['error'] is not None:
                out['evaluation_start_h'], out['evaluation_end_h'] = metrics['observed_window_h']
            if not case.get('curve_evaluation_eligible', True):
                out.update(evaluation_status='curve_reference_excluded',
                           blank_reason=case.get('curve_evaluation_ineligible_reason', 'Curated reference curve discrepancy'))
            reference = case.get('endpoint_references', {}).get(ep)
            if reference:
                source = reference['source']
                out['reference_source'] = source['path'] + '#' + source['location']
                out['evaluation_definition'] = reference.get('definition')
                if reference.get('independent_of_input') is not True:
                    out.update(evaluation_status='reference_used_as_input', blank_reason='Endpoint reference also informed model input; not independent accuracy evidence',
                               evaluation_prediction=None, observed_value=None, error=None, error_metric=None,
                               evaluation_start_h=None, evaluation_end_h=None)
                elif ep not in TERMINAL or reference.get('scenario_matched') is not True:
                    out.update(evaluation_status='reference_definition_not_matched',
                               blank_reason='Published reference must match dose, administration and population; dense versus sampled peak/window equivalence is not presumed',
                               evaluation_prediction=None, observed_value=None, error=None, error_metric=None,
                               evaluation_start_h=None, evaluation_end_h=None)
                elif reference.get('unit') != unit:
                    raise ValueError('Published endpoint unit mismatch for ' + case['case_id'] + ':' + ep)
                else:
                    p, o = out['predicted_value'], reference['value']
                    out['observed_value'] = o
                    out['evaluation_prediction'] = p
                    if isinstance(p, (float, int)) and isinstance(o, (float, int)) and p > 0 and o > 0 and math.isfinite(p + o):
                        out.update(error=abs(p - o) if ep == 'tmax' else max(p / o, o / p),
                                   error_metric='absolute_error_h' if ep == 'tmax' else 'fold_error',
                                   evaluation_status='scored_against_published_endpoint', blank_reason=None,
                                   evaluation_start_h=None, evaluation_end_h=None)
                    else:
                        out.update(evaluation_status='prediction_endpoint_unavailable', blank_reason='Published endpoint exists but model endpoint did not meet numerical gates',
                                   error=None, error_metric=None, evaluation_start_h=None, evaluation_end_h=None)
            if scoring_error and out['error'] is None:
                out.update(evaluation_status='scoring_failed', blank_reason=scoring_error)
            endpoint_rows.append(out)
    export_csv(output / 'predictions.csv', wide)
    export_csv(output / 'endpoint_evaluation.csv', endpoint_rows)
    export_csv(output / 'clinical_comparison.csv', comparison_rows)
    coverage = {ep: {'requested': len(rows),
                    'predicted': sum(r['endpoint'] == ep and r['prediction_status'] == 'completed' for r in endpoint_rows),
                    'evaluated': sum(r['endpoint'] == ep and r['error'] is not None for r in endpoint_rows)} for ep in ENDPOINTS}
    unchanged = all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in protocol['pins'].items())
    clinical_count = sum(c['observations_path'] is not None for c in protocol['cases'])
    eligible_clinical = sum(c['observations_path'] is not None and c.get('curve_evaluation_eligible', True) for c in protocol['cases'])
    predictions_unchanged = all(sha256(ROOT / r['prediction_path']) == r['prediction_sha256'] for r in rows if r.get('prediction_path'))
    summary = {'status': 'completed' if all(r['status'] == 'predicted' for r in rows) and unchanged and predictions_unchanged and len(comparison_rows) == eligible_clinical else 'completed_with_failures',
               'compound_count': len(DRUGS), 'case_count': len(rows), 'clinical_cases': clinical_count,
               'clinical_curve_eligible_cases': eligible_clinical, 'clinical_scored_cases': len(comparison_rows), 'hypothetical_cases': len(rows) - clinical_count,
               'native_attempts': sum(len(r.get('attempts', [])) for r in rows),
               'predicted_cases': sum(r['status'] == 'predicted' for r in rows),
               'independent_blinded_validation': False, 'parameters_optimized': False,
               'source_and_code_pins_unchanged': unchanged, 'endpoint_coverage': coverage,
               'predictions_unchanged_after_scoring': predictions_unchanged,
               'stratified_metrics': aggregate_cases(rows), 'results': rows}
    save(output / 'summary.json', summary)
    render_readme(output, summary, protocol, wide, comparison_rows)
    return summary


def render_readme(output, summary, protocol, wide, comparison_rows):
    chinese = {'caffeine': '咖啡因', 'midazolam': '咪达唑仑', 'fluconazole': '氟康唑',
               'alfentanil': '阿芬太尼', 'alprazolam': '阿普唑仑', 'sufentanil': '舒芬太尼',
               'moxifloxacin': '莫西沙星', 'amikacin': '阿米卡星', 'vancomycin': '万古霉素', 'furosemide': '呋塞米'}
    ep_names = dict(zip(ENDPOINTS, ['浓度曲线', 'AUC', 'Cmax', 'Tmax', '末次浓度', '半衰期', 'AUC∞', 'CL', 'Vz', 'MRT', 'Vss']))
    lines = ['# 十药、多剂量与多条件终点评测', '',
        f"已运行 **10 种药物、{summary['case_count']} 个场景**：{summary['clinical_cases']} 个临床研究场景，90 个假设预测场景。",
        f"成功预测 {summary['predicted_cases']}/{summary['case_count']}；原生求解共 {summary['native_attempts']} 次（包括延长模拟）。", '',
        '**本轮评估当前算法，没有调参。** 人体基础 PBPK＋药物证据构建；阿芬太尼沿用上轮已选定的 Rodgers–Rowland，其余使用 PK-Sim Standard。没有用完整同药拟合模型作为输入。', '',
        '**预测覆盖率不等于准确率。** 没有匹配观测或参考不合格时，误差留空；不把其他剂量/人群的观测复制过来。临床案例为历史开发资料，不能称为新药独立盲测。', '',
        '## 下载结果', '',
        '- [每场景全部预测值](predictions.csv)：每行一个场景，10 个数值终点及完整曲线路径。',
        '- [逐终点评测](endpoint_evaluation.csv)：预测值、匹配观测时点的预测值、观测值、误差、单位与空值原因。',
        '- [临床曲线比较](clinical_comparison.csv)：只包含通过资料检查且可评分的研究场景。',
        '- [完整结果](summary.json) · [冻结协议与来源校验](protocol.json) · [预测锁定记录](predictions_lock.json)', '',
        '## 场景与曲线结果', '',
        '每药额外运行：参考剂量的 0.5、1、2 倍，分别组合参考男性、延长输注的男性、参考女性。男性为30岁/70kg/175cm，女性为30岁/60kg/165cm；这些是声明的模拟条件，不是实测人群或用药建议。单次IV给药，网格场景有限窗口为0–24h。', '',
        '曲线误差只汇总另研究比较，不混入推导输入参数的研究；同研究剂量臂先平均，再按研究、药物汇总。', '',
        '| 药物 | 总场景 | 临床场景 | 可评分曲线 | 另研究曲线 GMFE ↓ |',
        '|---|---:|---:|---:|---:|']
    by_drug = summary['stratified_metrics'].get('separate_study_development_comparison', {}).get('metrics', {}).get('gmfe', {}).get('by_compound', {})
    for drug in DRUGS:
        cases = [c for c in protocol['cases'] if c['drug'] == drug]
        val = by_drug.get(cases[0]['compound_id'])
        lines.append(f"| {chinese[drug]} | {len(cases)} | {sum(c['observations_path'] is not None for c in cases)} | {sum(c['drug']==drug for c in comparison_rows)} | {f'{val:.3f}' if val else ''} |")
    lines += ['', '## 11 类终点覆盖', '', '| 终点 | 有预测值/场景数 | 有误差/场景数 |', '|---|---:|---:|']
    for ep, counts in summary['endpoint_coverage'].items():
        lines.append(f"| {ep_names[ep]} | {counts['predicted']}/{counts['requested']} | {counts['evaluated']}/{counts['requested']} |")
    lines += ['', '## 如何看空值和误差', '',
        '- 曲线本身保存在每场景的 `predicted_curve.csv`，不压成一个预测数字。',
        '- AUC、Cmax、Tmax、末次浓度的主预测对应请求窗口；临床误差使用相同观测时点和观测窗口。两列含义不同，不能互换。采样峰值不等于真实连续峰值。',
        '- 半衰期及AUC∞/CL/Vz/MRT/Vss使用长轨迹计算；临床稀疏曲线不能可靠提供这些参考时，误差空白。有核实的独立论文终点时，单独标注来源和比较定义。',
        '- 末次浓度是有限窗口末端浓度；不代表稳态谷浓度。Vz/Vss是表观分布容积。',
        '- 终末计算不放宽门槛：拟合调整R²≥0.98、跨度≥1个半衰期，AUC/AUMC外推比例≤20%。模型时域至多168h。仍不满足时，预测也会标明无法可靠估计，不伪造数值。',
        '- 保留每次模拟、失败记录和来源假设。历史资料曾用于开发；各药证据质量和人群匹配程度不同。没有用这些结果证明SOTA或临床可用。', '',
        '## 运行与进度', '', '在项目根目录执行：', '', '```bash',
        'runtime/app-env/bin/python -B scripts/evaluate_ten_drug_endpoints.py --workers 6',
        '# 相同代码和输入中断后继续：',
        'runtime/app-env/bin/python -B scripts/evaluate_ten_drug_endpoints.py --workers 6 --resume',
        './scripts/progress.sh', '```', '',
        '原生求解使用CPU CVODES，每场景2线程，最多6场景并行；不是GPU神经网络训练。未启动旧实验队列或网页服务。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    from pbpk_kb.blueprints.extended_prediction import MODEL_TAIL_POLICY, summarize_extended_prediction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BATCH / 'evaluation')
    parser.add_argument('--workers', type=int, default=6, choices=range(1, 9))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output = inside(args.output, ROOT)
    if args.resume:
        protocol = load(output / 'protocol.json')
        if not all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in protocol['pins'].items()):
            raise ValueError('Frozen code or inputs changed; use a new run instead of resuming')
        cases = protocol['cases']
    else:
        cases, pins = prepare()
        if args.prepare_only:
            print(json.dumps({'drug_count': len(DRUGS), 'cases': len(cases), 'clinical_cases': sum(c['observations_path'] is not None for c in cases), 'assembled': len(cases)}, indent=2))
            return 0
        if output.exists():
            raise FileExistsError('Preserve existing results; use --resume for an identical frozen run')
        output.mkdir(parents=True)
        protocol = {'started_at': now(), 'cases': cases, 'pins': pins,
                    'builder': 'current_evidence_blueprint_with_previously_locked_partition_choices',
                    'new_drug_partition_method': 'PKSimStandard', 'parameter_optimization': False,
                    'grid': {'dose_multipliers': [.5, 1., 2.], 'conditions': ['30y male 70kg 175cm; reference infusion', 'same male; doubled infusion (30min if reference bolus)', '30y female 60kg 165cm; reference infusion'], 'report_window_h': [0., 24.]},
                    'initial_simulation_end_h': 'max(24, requested_report_end)',
                    'maximum_simulation_end_h': 168., 'maximum_attempts_per_case': 4,
                    'model_tail_policy': MODEL_TAIL_POLICY,
                    'extension_policy': 'model-only diagnostics; fixed numerical rules; never observed outcomes',
                    'endpoint_reference_policy': 'Same observation times for clinical comparisons; no synthetic truth or cross-condition imputation',
                    'workers': args.workers, 'cpu_threads_per_job': 2, 'solver': 'CPU_CVODES',
                    'max_total_output_bytes': 4 * 1024**3,
                    'output_quota_reserve_per_worker_bytes': 256 * 1024**2,
                    'output_quota_is_bounded_admission_guard_not_filesystem_hard_quota': True,
                    'independent_blinded_validation': False}
        save(output / 'protocol.json', protocol)
    started = time.monotonic()
    stat_fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(output / 'runner.json', {'pid': os.getpid(), 'start_ticks': int(stat_fields[19]), 'started_at': now()})
    save(ROOT / 'runtime/deployment/source_adaptation_pilot.json', {'output': rel(output), 'registered_at': now()})
    rows, mutex = [], threading.Lock()

    def progress(stage, status='running'):
        with mutex:
            save(output / 'progress.json', {'mode': 'evidence_blueprint', 'cohort_mode': 'multidrug_endpoints',
                 'stage': stage, 'status': status, 'started_at': protocol['started_at'],
                 'elapsed_seconds': round(time.monotonic() - started, 2), 'compound_count': 10,
                 'planned_native_jobs': len(cases), 'completed_native_jobs': len(rows),
                 'successful_native_jobs': sum(r['status'] == 'predicted' for r in rows),
                 'study_count': len({(c['compound_id'], c['study_id']) for c in cases if c['study_id']}),
                 'planned_jobs_are_cases_with_bounded_numerical_extensions': True})

    def run(case):
        case_dir = output / 'cases' / case['case_id']
        result_path = case_dir / 'case_result.json'
        if result_path.exists():
            old = load(result_path)
            if old.get('prediction_path') and sha256(ROOT / old['prediction_path']) != old['prediction_sha256']:
                raise ValueError('Previously locked prediction changed')
            if old['status'] == 'predicted':
                return old
            result_path.rename(case_dir / ('previous_failed_case_' + str(time.time_ns()) + '.json'))
        row = {k: v for k, v in case.items() if k not in {'request', 'observation_times_min'}}
        row.update(status='failed', attempts=[])
        try:
            case_dir.mkdir(parents=True, exist_ok=True)
            horizon = max(24., case['request']['target']['window_end_h'])
            for attempt in range(4):
                reserve = args.workers * protocol['output_quota_reserve_per_worker_bytes']
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > protocol['max_total_output_bytes'] - reserve:
                    raise RuntimeError('Project output quota reached; no further native execution')
                req = deepcopy(case['request'])
                req['target']['window_end_h'] = horizon
                req['target']['terminal_fit'] = {'start_h': horizon * 2 / 3, 'end_h': horizon}
                attempt_dir = case_dir / f'native_{attempt + 1}'
                if attempt_dir.exists():
                    previous = load(attempt_dir / 'result.json') if (attempt_dir / 'result.json').exists() else None
                    if not previous or previous.get('status') != 'predicted':
                        attempt_dir.rename(case_dir / f'interrupted_native_{attempt + 1}_{time.time_ns()}')
                    else:
                        result = previous
                if not attempt_dir.exists():
                    result = run_evidence_blueprint(request=req, output_dir=attempt_dir, project_root=ROOT, execute=True)
                row['attempts'].append({'path': rel(attempt_dir), 'horizon_h': horizon, 'status': result['status']})
                if result['status'] != 'predicted':
                    raise RuntimeError('Native prediction failed: ' + json.dumps(result))
                curve_path = attempt_dir / 'prediction_locked.json'
                curve = load(curve_path)
                endpoints = summarize_extended_prediction(curve, case['request'])
                save(case_dir / f'endpoint_attempt_{attempt + 1}.json', slim(endpoints))
                row.update(status='predicted', prediction_path=rel(curve_path), prediction_sha256=sha256(curve_path),
                           simulation_end_h=horizon, endpoint_values=slim(endpoints['endpoints']),
                           endpoint_diagnostics=slim({k: v for k, v in endpoints.items() if k != 'endpoints'}))
                missing = [e for e in TERMINAL if row['endpoint_values'].get(e, {}).get('status') != 'completed']
                if not missing or horizon >= 168. or not endpoints.get('extension', {}).get('recommended', False):
                    break
                suggestion = endpoints.get('extension', {}).get('suggested_end_h') or 0.
                horizon = min(168., max(horizon * 2, horizon + 24., suggestion))
            times, concentrations = curve['time_min'], curve['concentration']
            mw = next(e['value'] for e in case['request']['evidence'] if e['parameter'] == 'molecular_weight')
            curve_csv = case_dir / 'predicted_curve.csv'
            export_csv(curve_csv, [{'time_h': t / 60, 'concentration_umol_l': c, 'concentration_mg_l': c * mw / 1000}
                                    for t, c in zip(times, concentrations)])
            row['curve_csv'] = rel(curve_csv)
        except Exception as exc:
            row.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        save(result_path, row)
        return row

    progress('native_prediction')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, c) for c in cases]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            progress('native_prediction')
            print(json.dumps({'case': row['case_id'], 'status': row['status'], 'completed': len(rows), 'total': len(cases),
                              'attempts': len(row['attempts'])}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: r['case_id'])
    save(output / 'predictions_lock.json', {'locked_at': now(), 'before_clinical_scoring': True,
        'predictions': {r['case_id']: r['prediction_sha256'] for r in rows if r.get('prediction_sha256')}})
    progress('scoring')
    summary = make_reports(output, protocol, rows)
    progress('completed', summary['status'])
    print(json.dumps({k: v for k, v in summary.items() if k not in {'results', 'stratified_metrics'}}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
