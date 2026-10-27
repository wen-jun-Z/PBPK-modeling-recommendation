#!/usr/bin/env python3
"""Merge a frozen main run and its numerical supplement into a separate report."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time

from evaluate_ten_drug_endpoints import BATCH, DRUGS, ROOT, make_reports, now
from pbpk_kb.blueprints.source_workflow import save, sha256


def load(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def retry_failed_supplement():
    """Recover operational failures serially; no progress writer races quota scans."""
    from supplement_ten_drug_horizons import run_case
    source, output = BATCH / 'horizon_supplement', BATCH / 'horizon_retry'
    summary, protocol = load(source / 'summary.json'), load(source / 'protocol.json')
    failed = {r['case_id'] for r in summary['results'] if r['horizon_supplement']['status'] == 'failed'}
    if not failed:
        print('No operational horizon failures to retry')
        return 0
    if output.exists():
        raise FileExistsError('Preserve existing retry results')
    if not all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in protocol['pins'].items()):
        raise ValueError('Frozen source changed before recovery')
    cases = [c for c in protocol['cases'] if c['case_id'] in failed]
    pins = {**protocol['pins'], relative(source / 'summary.json'): sha256(source / 'summary.json'),
            relative(Path(__file__).resolve()): sha256(Path(__file__).resolve())}
    output.mkdir()
    save(output / 'protocol.json', {'cases': cases, 'pins': pins, 'workers': 1, 'started_at': now(),
         'reason': 'Serial recovery of operational failures; original failures retained; no concurrent progress writer during quota scan',
         'maximum_horizon_h': 1344., 'parameters_changed': False, 'quality_gates_changed': False})
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(output / 'runner.json', {'pid': os.getpid(), 'start_ticks': int(fields[19]), 'started_at': now()})
    rows, started = [], time.monotonic()
    def progress(status):
        save(output / 'progress.json', {'status': status, 'stage': 'serial_operational_recovery',
             'planned_native_jobs': len(cases), 'completed_native_jobs': len(rows),
             'elapsed_seconds': time.monotonic() - started})
    progress('running')
    for case in cases:
        rows.append(run_case(case, output, 1024**3))
        progress('running')
    unchanged = all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items())
    status = 'completed' if unchanged and all(r['horizon_supplement']['status'] == 'completed' for r in rows) else 'completed_with_failures'
    save(output / 'summary.json', {'status': status, 'source_and_code_pins_unchanged': unchanged,
         'new_native_attempts': sum(len(r['horizon_supplement']['attempts']) for r in rows), 'results': rows})
    save(output / 'predictions_lock.json', {'locked_at': now(), 'predictions': {r['case_id']: r['prediction_sha256'] for r in rows}})
    progress(status)
    print(json.dumps({'status': status, 'recovered_cases': len(rows)}))
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-horizon-failures', action='store_true')
    args = parser.parse_args()
    if args.retry_horizon_failures:
        return retry_failed_supplement()
    baseline, supplement, output = BATCH / 'evaluation', BATCH / 'horizon_supplement', BATCH / 'final_results'
    if output.exists():
        raise FileExistsError('The final report already exists; preserve it')
    base_protocol, extra_protocol = load(baseline / 'protocol.json'), load(supplement / 'protocol.json')
    base, extra = load(baseline / 'summary.json'), load(supplement / 'summary.json')
    if not base['status'].startswith('completed') or not extra['status'].startswith('completed'):
        raise ValueError('Finish both numerical phases first')
    pins = {**base_protocol['pins'], **extra_protocol['pins']}
    for directory in (baseline, supplement):
        for name in ('protocol.json', 'summary.json', 'predictions_lock.json'):
            p = directory / name
            pins[relative(p)] = sha256(p)
    retry_path = BATCH / 'horizon_retry'
    retry = load(retry_path / 'summary.json') if (retry_path / 'summary.json').exists() else None
    if retry:
        if retry['status'] != 'completed':
            raise ValueError('Operational recovery is not complete')
        pins.update(load(retry_path / 'protocol.json')['pins'])
        for name in ('protocol.json', 'summary.json', 'predictions_lock.json'):
            p = retry_path / name
            pins[relative(p)] = sha256(p)
        retry_rows = {r['case_id']: r for r in retry['results']}
        original_failures = {r['case_id'] for r in extra['results'] if r['horizon_supplement']['status'] == 'failed'}
        if set(retry_rows) != original_failures:
            raise ValueError('Recovery must correspond exactly to recorded failures')
        extra['results'] = [retry_rows.get(r['case_id'], r) for r in extra['results']]
    pins[relative(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    if not all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items()):
        raise ValueError('Frozen source or result changed before final reporting')
    rows = {r['case_id']: deepcopy(r) for r in base['results']}
    expected = {r['case_id'] for r in base['results'] if r['status'] == 'predicted' and any(
        r.get('endpoint_values', {}).get(e, {}).get('status') != 'completed'
        for e in ('terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss'))}
    if set(r['case_id'] for r in extra['results']) != expected:
        raise ValueError('Supplement does not cover the declared missing-endpoint cases')
    for row in extra['results']:
        old = rows[row['case_id']]
        if (row['compound_id'], row['role'], row['study_id']) != (old['compound_id'], old['role'], old['study_id']):
            raise ValueError('Supplement changed case identity or evaluation role')
        if row['prediction_path'] != old['prediction_path']:
            attempts = row['horizon_supplement']['attempts']
            accepted = [a for a in attempts if a['status'] == 'predicted']
            if not accepted or accepted[-1]['original_window_consistency']['status'] != 'passed':
                raise ValueError('No original-window consistency proof for supplemental prediction')
        rows[row['case_id']] = deepcopy(row)
    protocol = deepcopy(base_protocol)
    protocol.update(pins=pins, final_report_created_at=now(), baseline_protocol=relative(baseline / 'protocol.json'),
                    supplement_protocol=relative(supplement / 'protocol.json'),
                    maximum_simulation_end_h=1344., maximum_attempts_per_case=7,
                    extension_policy='Original <=168h run retained; missing terminal predictions supplemented at 336/672/1344h without changing parameters or quality gates',
                    numerical_supplement_cap_overrides_module_suggestion_only=True)
    output.mkdir()
    for cid in rows:
        (output / 'cases' / cid).mkdir(parents=True)
    save(output / 'protocol.json', protocol)
    save(output / 'predictions_lock.json', {'locked_at': now(), 'before_final_report_scoring': True,
         'main_batch_was_previously_scored': True, 'no_selection_on_clinical_scores': True,
         'predictions': {cid: row['prediction_sha256'] for cid, row in rows.items() if row.get('prediction_sha256')}})
    summary = make_reports(output, protocol, sorted(rows.values(), key=lambda row: row['case_id']))
    summary['main_native_attempts'] = base['native_attempts']
    summary['supplement_native_attempts'] = extra['new_native_attempts']
    summary['retry_native_attempts'] = retry['new_native_attempts'] if retry else 0
    summary['recorded_operational_failures_recovered'] = len(retry['results']) if retry else 0
    summary['original_horizon_supplement_status'] = extra['status']
    summary['horizon_supplement_status'] = 'completed_after_recorded_recovery' if retry else extra['status']
    summary['all_eleven_predictions_available'] = all(c['predicted'] == c['requested'] for c in summary['endpoint_coverage'].values())
    save(output / 'summary.json', summary)
    readme_path = output / 'README.md'
    text = readme_path.read_text().replace('模型时域至多168h。', '主批时域上限168h；22个缺失终末指标的场景另按336/672/1344h追加时域。原时间网格、原窗口预测、模型参数和质量门槛保持不变。')
    text = text.replace('./scripts/progress.sh\n```',
        'runtime/app-env/bin/python -B scripts/supplement_ten_drug_horizons.py --workers 6\n'
        '# 仅当补充阶段出现操作失败时串行恢复：\n'
        'runtime/app-env/bin/python -B scripts/summarize_ten_drug_evaluation.py --retry-horizon-failures\n'
        'runtime/app-env/bin/python -B scripts/summarize_ten_drug_evaluation.py\n./scripts/progress.sh\n```')
    names = {'caffeine': '咖啡因', 'midazolam': '咪达唑仑', 'fluconazole': '氟康唑', 'alfentanil': '阿芬太尼',
             'alprazolam': '阿普唑仑', 'sufentanil': '舒芬太尼', 'moxifloxacin': '莫西沙星',
             'amikacin': '阿米卡星', 'vancomycin': '万古霉素', 'furosemide': '呋塞米'}
    requests = {c['case_id']: c['request'] for c in protocol['cases']}
    def value(row, endpoint, field):
        number = row['endpoint_values'].get(endpoint, {}).get(field)
        return '' if number is None else f'{number:.4g}'
    lines = ['', '## 预测值示例：每药一个参考场景', '',
        '下表是模型预测，不是实测结果。统一30岁男性、70kg、175cm，使用各药参考剂量和输注时长；曲线、AUC、峰值和末次浓度对应0–24h。全部109场景见CSV。', '',
        '| 药物 | 剂量 mg | 输注 min | AUC₀–₂₄ mg·h/L | Cmax mg/L | Tmax h | C₂₄ mg/L |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for drug in DRUGS:
        row = rows[f'{drug}_grid_d1p0_reference_male']
        dosing = requests[row['case_id']]['dosing']
        vals = [value(row, *pair) for pair in [('auc', 'value_mg_h_per_l'), ('cmax', 'value_mg_per_l'),
                ('tmax', 'value_h'), ('c_last', 'value_mg_per_l')]]
        lines.append('| ' + ' | '.join([names[drug], f"{dosing['dose_mg']:g}", f"{dosing.get('infusion_minutes', 0):g}", *vals]) + ' |')
    lines += ['', '| 药物 | 半衰期 h | AUC∞ mg·h/L | CL L/h | Vz L | MRT h | Vss L |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for drug in DRUGS:
        row = rows[f'{drug}_grid_d1p0_reference_male']
        vals = [value(row, *pair) for pair in [('terminal_half_life', 'value_h'), ('auc_inf', 'value_mg_h_per_l'),
                ('clearance', 'value_l_per_h'), ('vz', 'value_l'), ('mrt', 'value_h'), ('vss', 'value_l')]]
        lines.append('| ' + ' | '.join([names[drug], *vals]) + ' |')
    lines += ['', '当前清除和结合过程采用线性假设。同一人体与输注条件下，剂量变化时AUC/Cmax近似按比例变化，CL、半衰期、Vz、MRT、Vss可能基本不变；这不能证明真实药物在这些剂量下没有饱和现象。', '',
              '## 本轮效果与限制', '',
              '另研究曲线比较仅覆盖8药、12场景、10个研究组；阿米卡星和莫西沙星目前只有开发重建数据，不能把其结果充作跨研究泛化证据。', '',
              '全部误差分终点报告。AUC接近观测并不保证曲线形状、峰值或末段准确；人口资料缺失、手术背景与通用生理不匹配、多电离位点近似等限制见各药输入记录。呋塞米Haegeli曲线未评分，论文报告终点对照仍保留。', '',
              '[原主批结果](../evaluation/summary.json) · [追加时域结果](../horizon_supplement/summary.json) · [独立数值复核](../review.json)']
    if retry:
        lines += ['', '补充阶段1例在原生求解前因进度临时文件原子替换与目录计数竞争而失败，已在无并发进度写入的目录串行恢复；原失败和[恢复记录](../horizon_retry/summary.json)均保留。']
    readme_path.write_text(text + '\n'.join(lines) + '\n')
    (BATCH / 'README.md').write_text('# 十药终点评测\n\n'
        '最终汇总见 [报告与预测值](final_results/README.md)。\n\n'
        '- [全部109场景预测](final_results/predictions.csv)\n'
        '- [11类终点评测及空值原因](final_results/endpoint_evaluation.csv)\n'
        '- [临床曲线误差](final_results/clinical_comparison.csv)\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'results', 'stratified_metrics'}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
