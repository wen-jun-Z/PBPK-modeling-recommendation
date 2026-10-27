#!/usr/bin/env python3
"""Read bounded project progress only; never inspect sealed scientific scores."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unicodedata

from watch_blueprint_suite import ProcessScope, inside, read_json

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ('历史三药', 'reviewed_development_suite_20260920'),
    ('机制两药', 'mechanism_suite_20260920'),
    ('呋塞米', 'furosemide_mechanism_suite_20260920_seed_fix_v2'),
    ('克拉霉素原尝试', 'clarithromycin_source_suite_20260920'),
    ('克拉霉素窗口修复', 'clarithromycin_window_fix_suite_20260920'),
    ('克拉霉素重建修复', 'clarithromycin_rebuild_fix_suite_20260920'),
    ('瑞舒伐他汀', 'rosuvastatin_source_suite_20260920'),
    ('伏立康唑', 'voriconazole_source_suite_20260920'),
    ('伏立康唑重建修复', 'voriconazole_rebuild_fix_suite_20260920'),
    ('阿托伐他汀', 'atorvastatin_source_suite_20260920'),
)
PIPELINES = (
    ('克拉霉素方法接续', 'blueprint_followup_clarithromycin_20260920_status.json'),
    ('克拉霉素修复接续', 'blueprint_followup_clarithromycin_window_fix_20260920_status.json'),
    ('克拉霉素重建修复接续', 'blueprint_followup_clarithromycin_rebuild_fix_20260920_status.json'),
    ('瑞舒伐他汀方法接续', 'blueprint_followup_rosuvastatin_20260920_status.json'),
    ('伏立康唑方法接续', 'blueprint_followup_voriconazole_20260920_status.json'),
    ('伏立康唑修复后方法接续', 'blueprint_followup_voriconazole_rebuild_fix_20260920_status.json'),
    ('呋塞米外部验证', 'blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json'),
    ('呋塞米外部验证接口修复', 'blueprint_furosemide_external_contract_fix_20260920_status.json'),
    ('伏立康唑外部验证', 'blueprint_voriconazole_validation_followup_status.json'),
    ('伏立康唑修复后外部验证', 'blueprint_voriconazole_rebuild_validation_followup_status.json'),
    ('瑞舒伐他汀外部验证', 'blueprint_rosuvastatin_validation_followup_status.json'),
    ('阿托伐他汀方法接续', 'blueprint_followup_atorvastatin_20260920_status.json'),
    ('阿托伐他汀外部验证', 'blueprint_atorvastatin_validation_followup_status.json'),
    ('阿托伐他汀提案接续', 'blueprint_atorvastatin_after_qwen_status.json'),
)
HISTORICAL_ATTEMPTS = {'克拉霉素原尝试', '克拉霉素窗口修复', '伏立康唑'}
EXTERNAL_PIPELINES = {
    '呋塞米': '呋塞米外部验证接口修复',
    '瑞舒伐他汀': '瑞舒伐他汀外部验证',
    '伏立康唑重建修复': '伏立康唑修复后外部验证',
    '阿托伐他汀': '阿托伐他汀外部验证',
}
ACTIVE_STATES = {'running', 'starting', 'waiting_for_methods', 'waiting_for_qwen', 'waiting_for_previous_suite'}
LABELS = {
    'running': '运行中', 'completed': '已结束', 'stopped': '已停止',
    'completed_with_failures': '已结束（保留失败记录）',
    'stopped_or_stale': '进程已退出', 'not_started': '未启动', 'not_registered': '未登记',
    'waiting_for_methods': '自动等待方法完成', 'waiting_for_qwen': '等待 Qwen',
    'waiting_for_previous_suite': '等待前置队列', 'predictions': '独立预测中', 'scoring': '独立评分中',
    'handed_off': '已交接', 'starting': '启动中',
}


def load(relative):
    path = inside(relative, root=ROOT)
    return read_json(path) if path.exists() else None


def alive(record, scope):
    if not record or type(record.get('pid')) is not int or 'start_ticks' not in record:
        return None
    return scope.state(record['pid'], record['start_ticks']) is not None


def snapshot():
    scope = ProcessScope()
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'sealed_scores_read': False, 'suites': [], 'pipelines': []}
    source_pilot = load('runtime/deployment/source_adaptation_pilot.json')
    if source_pilot:
        relative = source_pilot['output']
        state = load(relative + '/progress.json') or {}
        runner = load(relative + '/runner.json') or {}
        report['status'] = 'source_adaptation_pilot'
        report['source_adaptation_pilot'] = {
            'path': relative, 'registered_runner_alive': alive(runner, scope), **state}
        auxiliary = load('runtime/deployment/blueprint_auxiliary_jobs.json') or {}
        report['auxiliary_jobs'] = []
        for job in auxiliary.get('jobs', []):
            auxiliary_state = load(job['output'] + '/progress.json') or {'status': 'not_started'}
            report['auxiliary_jobs'].append({'name': job['name'], 'path': job['output'], **auxiliary_state})
        if state.get('status') == 'running' and state.get('started_at'):
            report['source_adaptation_pilot']['elapsed_seconds'] = max(
                0, (datetime.now(timezone.utc) - datetime.fromisoformat(state['started_at'])).total_seconds())
        if state.get('status') == 'running' and report['source_adaptation_pilot']['registered_runner_alive'] is False:
            report['source_adaptation_pilot']['status'] = 'stopped_or_stale'
        return report
    hold = load('runtime/deployment/blueprint_experiment_hold.json')
    if hold and hold.get('development_status') == 'algorithm_redesign':
        report['experiment_hold'] = hold
        report['status'] = 'algorithm_redesign'
        return report
    pilot = load('runtime/deployment/native_design_pilot.json')
    if pilot:
        relative = pilot['output']
        state = load(relative + '/progress.json') or {}
        runner = load(relative + '/runner.json') or {}
        report['native_design_pilot'] = {
            'path': relative, 'registered_runner_alive': alive(runner, scope),
            **{key: state.get(key) for key in ('status', 'stage', 'completed_designs',
                    'planned_designs', 'completed_predictions', 'planned_predictions',
                    'successful_predictions', 'elapsed_seconds', 'reason')},
            'active_designs': []}
        for job in state.get('jobs', {}).values():
            drug, method = job['drug'], job['method']
            progress = load(f'{relative}/{drug}/{method}/design/progress.json') or {}
            if progress.get('status') == 'running':
                report['native_design_pilot']['active_designs'].append({
                    'drug': drug, 'method': method, 'stage': progress.get('stage'),
                    'budget': progress.get('budget', {}).get('totals', {})})
        if state.get('status') == 'running' and report['native_design_pilot']['registered_runner_alive'] is False:
            report['native_design_pilot']['status'] = 'stopped_or_stale'
    hold = load('runtime/deployment/blueprint_experiment_hold.json')
    if hold:
        report['experiment_hold'] = {key: hold.get(key) for key in
            ('status', 'requested_at', 'checked_at', 'reason', 'results_deleted',
             'development_status', 'current_algorithm', 'next_step')}
    for name, directory in SUITES:
        relative = 'experiments/blueprint_design/' + directory
        progress = load(relative + '/progress.json')
        if progress is None:
            report['suites'].append({'name': name, 'status': 'not_started'})
            continue
        row = {'name': name, 'path': relative,
               **{key: progress.get(key) for key in (
                   'status', 'stage', 'planned_method_jobs', 'method_status_counts',
                   'elapsed_seconds', 'halt_reason', 'strict_main_eligible_drugs')}}
        runner = load(relative + '/runner.json')
        row['registered_runner_alive'] = alive(runner, scope)
        row['runner_pid'] = runner.get('pid') if runner else None
        if row['status'] == 'running' and row['registered_runner_alive'] is False:
            row['recorded_status'] = row['status']
            row['status'] = 'stopped_or_stale'
        if row['status'] == 'running':
            queue = load(relative + '/queue.json') or {}
            row['active_method_progress'] = []
            for task in queue.get('tasks', []):
                if task.get('status') != 'running':
                    continue
                task_progress = load(str(Path(task['output']) / 'progress.json')) or {}
                row['active_method_progress'].append({
                    'task_id': task['task_id'],
                    **{key: task_progress.get(key) for key in
                       ('method', 'seed', 'stage', 'completed_jobs', 'planned_jobs')},
                    'budget': task_progress.get('budget', {}).get('totals', {}),
                })
        report['suites'].append(row)
    for name, filename in PIPELINES:
        record = load('runtime/deployment/' + filename)
        if record is None:
            report['pipelines'].append({'name': name, 'status': 'not_registered'})
            continue
        process = record
        if filename == 'blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json':
            registration = load('runtime/deployment/blueprint_furosemide_validation_pipeline_20260920_seed_fix_v2.json')
            if (registration and registration.get('pid') == record.get('pid')
                    and registration.get('status_file') == 'runtime/deployment/' + filename):
                process = registration
        pipeline = {'name': name, 'status_path': 'runtime/deployment/' + filename,
            **{key: record.get(key) for key in ('status', 'stage', 'reason')},
            'registered_process_alive': alive(process, scope),
            **{key: process.get(key) for key in ('pid', 'start_ticks')}}
        if pipeline['status'] in ACTIVE_STATES and pipeline['registered_process_alive'] is False:
            pipeline['recorded_status'] = pipeline['status']
            pipeline['status'] = 'stopped_or_stale'
        report['pipelines'].append(pipeline)
    qwen = load('experiments/blueprint_design/qwen_atorvastatin_source_proposals_20260920/progress.json')
    if qwen:
        registration = load('runtime/deployment/blueprint_qwen_atorvastatin_20260920.json')
        report['qwen_atorvastatin'] = {key: qwen.get(key) for key in ('state', 'completed', 'total')}
        report['qwen_atorvastatin']['registered_timeout_alive'] = alive(registration, scope)
    acquisition = load('experiments/blueprint_design/independent_cohort_20260920/acquisition_summary.json')
    if acquisition:
        coverage = acquisition.get('curation_coverage', {})
        curves = acquisition.get('concentration_curve_coverage', {})
        report['data'] = {
            'candidate_drugs': coverage.get('independent_clinical_candidate_drugs'),
            'protocol_records': coverage.get('prepared_protocols'),
            'target_publications': coverage.get('target_publications'),
            'numerical_curve_drugs': curves.get('unique_compounds'),
            'numerical_curve_studies': curves.get('unique_studies'),
            'candidates_are_not_completed_independent_evaluations': True,
            'target_independent_drugs': 60,
        }
    handoff_paths = [ROOT/'experiments/blueprint_design/external_validation/expanded_cohort_20260920/handoff_manifest.json']
    round4 = ROOT/'experiments/blueprint_design/external_validation/clinical_cohort_round4_20260920'
    handoff_paths.extend(sorted(round4.glob('curated_*/handoff_manifest.json')))
    final_handoff = round4/'final_olanzapine_handoff/handoff_manifest.json'
    if final_handoff.exists():
        handoff_paths.append(final_handoff)
    verified_handoffs = []
    for handoff_file in handoff_paths:
        handoff_path = str(handoff_file.relative_to(ROOT))
        handoff = load(handoff_path)
        if not handoff or handoff.get('status') not in {
                'curation_and_static_verification_complete',
                'curation_and_static_verification_complete_retrieval_round_stopped'}:
            continue
        for key in ('curation_receipt', 'parent_baseline_summary', 'final_static_validation'):
            if key == 'final_static_validation' and key not in handoff:
                continue
            ref = handoff[key]
            path = inside(ref['path'], root=ROOT)
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != ref['sha256']:
                raise ValueError('Clinical cohort handoff resource changed: ' + ref['path'])
        drugs = handoff['drug_names']
        if len(drugs) != len(set(drugs)) or len(drugs) != handoff['candidate_drugs_total']:
            raise ValueError('Candidate drugs must be deduplicated in the verified handoff')
        verified_handoffs.append((handoff['candidate_drugs_total'], handoff_path, handoff))
    if verified_handoffs:
        _, handoff_path, handoff = max(verified_handoffs, key=lambda item: item[0])
        report.setdefault('data', {}).update(
            candidate_drugs=handoff['candidate_drugs_total'],
            protocol_records=handoff['clinical_protocol_metadata_total'],
            target_publications=handoff['unique_target_publications_total'],
            numerical_curve_drugs=handoff['full_curve_drugs_total'],
            latest_verified_handoff=handoff_path,
            candidates_are_not_completed_independent_evaluations=True,
            target_independent_drugs=handoff['target_independent_evaluation_drugs'],
            latest_round_endpoint_qualification_pending=handoff.get('round4_endpoint_qualification_pending_compounds', []),
            background_data_acquisition_running=handoff.get('background_data_acquisition_running'),
        )
    plan = load('experiments/blueprint_design/experiment_plan.json')
    if plan:
        targets = plan['cohort_targets_not_observed_counts']
        report['research_plan'] = {
            'protocol_status': plan['protocol_status'],
            'development_drugs_target': targets['development_drugs'],
            'independent_evaluation_drugs_target': targets['evaluation_drugs'],
            'planned_design_runs_target': plan['planned_mandatory_design_runs_total'],
            'all_planned_experiments_queued': False,
            'requires_later_preparation_and_launch': [
                '克拉霉素新研究的重复给药原生验证及独立评分接续',
                '其余药物的资料资格核验、原生适配与完整方法对照',
                '全规模独立主实验、用途/机制目标、稳健性与合成机制挑战',
            ],
            'counts_are_targets_not_registered_task_counts': True,
        }
    return report


def label(value):
    return LABELS.get(value, value or '-')


def duration(seconds):
    if not isinstance(seconds, (int, float)):
        return '-'
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(text))


def table(rows):
    widths = [max(width(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join(' | '.join(str(value) + ' '*(widths[i]-width(value)) for i, value in enumerate(row)) for row in rows)


def render(report, *, history=False, details=False):
    lines = ['PBPK 实验进度  ' + report['checked_at_utc'], '']
    if report.get('status') == 'source_adaptation_pilot':
        pilot = report['source_adaptation_pilot']
        counts = []
        if 'cases' in pilot and 'adaptations_accepted' in pilot:
            counts.append(f"模型适配 {pilot['adaptations_accepted']}/{pilot['cases']}")
        counts.append(f"原生执行与读回 {pilot.get('completed_native_jobs', 0)}/{pilot.get('planned_native_jobs', 0)}")
        if 'successful_native_jobs' in pilot:
            counts.append(f"通过 {pilot['successful_native_jobs']}")
        if 'planned_pk_checks' in pilot:
            counts.append(f"PK 检查完成 {pilot.get('completed_pk_checks', 0)}/{pilot['planned_pk_checks']}")
        evidence_mode = pilot.get('mode') == 'evidence_blueprint'
        title = '人体模板与药物证据构建' if evidence_mode else '完整来源模型适配'
        lines += [f"{title}：{label(pilot.get('status'))}；阶段={pilot.get('stage')}。",
                  '；'.join(counts) + '。',
                  f"用时 {duration(pilot.get('elapsed_seconds'))}；进程存活={pilot.get('registered_runner_alive')}。",
                  ((f"本轮覆盖 {pilot.get('compound_count', '-')} 药、{pilot.get('study_count', '-')} 研究；固定构建规则，扩展终点评测；不代表独立盲测。"
                    if pilot.get('cohort_mode') == 'multidrug_endpoints' else
                    '本轮是一药两研究的证据构建开发检查；不代表独立预测验证，旧大规模实验没有恢复。')
                   if evidence_mode else '本轮是两药小规模工程检查与历史观测回归；旧大规模实验没有恢复。'),
                  '结果目录：' + str(ROOT / pilot['path'])]
        if pilot.get('reason'):
            lines.append('原因：' + pilot['reason'])
        for job in report.get('auxiliary_jobs', []):
            lines.append(f"后续小评估：{job['name']}；{label(job.get('status'))}；阶段={job.get('stage', '-')}；"
                         f"执行 {job.get('completed_native_jobs', 0)}/{job.get('planned_native_jobs', '-')}；"
                         f"用时 {duration(job.get('elapsed_seconds'))}。")
        if details:
            lines.extend(f"  {key}: {value.get('status')}" for key, value in pilot.get('jobs', {}).items())
        return '\n'.join(lines)
    if report.get('status') == 'algorithm_redesign':
        lines += ['当前阶段：重新设计基于人体生理与完整药物模型的蓝图算法。',
                  '旧蓝图实验结果已按要求删除；没有安排新的实验。',
                  '原始模型、药物证据、临床观测与必要的构建输入保留。',
                  '方法见 docs/推荐策略.md 的蓝图设计部分；验证安排见 docs/研究计划.md 第 9 节。']
        return '\n'.join(lines)
    pilot = report.get('native_design_pilot')
    if pilot:
        lines.append(f"当前小规模评估：{label(pilot.get('status'))}；阶段={pilot.get('stage')}；"
                     f"设计 {pilot.get('completed_designs', 0)}/{pilot.get('planned_designs', 8)}；"
                     f"预测尝试 {pilot.get('completed_predictions', 0)}/{pilot.get('planned_predictions', 28)}；"
                     f"成功 {pilot.get('successful_predictions', 0)}。")
        for item in pilot['active_designs']:
            lines.append(f"  {item['drug']} / {item['method']}：{item['stage']}；调用={item['budget'].get('logical_calls', '-')}。")
        lines.append('旧全量队列保持暂停；小规模评估单独登记。\n')
    if report.get('experiment_hold', {}).get('status') == 'paused':
        if any(row['status'] in ACTIVE_STATES for row in report['suites'] + report['pipelines']):
            lines.append('存在暂停记录，但检测到活动队列；以以下实时状态为准。\n')
        elif pilot:
            pass
        elif report['experiment_hold'].get('development_status') == 'code_verified_waiting_for_small_evaluation':
            lines.append('旧实验已暂停：算法开发与代码检查已完成，下一步为小规模真实评估；旧结果保留。\n')
        else:
            lines.append('实验已暂停：先完成需求与证据引导算法；结果保留，剩余任务需另行启动。\n')
    pipelines = {row['name']: row for row in report['pipelines']}
    displayed = [row for row in report['suites'] if history or row['name'] not in HISTORICAL_ATTEMPTS]
    rows = [['队列', '状态', '已结束/计划', '完整', '预算耗尽', '其他结束', '用时', '独立评价/接续']]
    for row in displayed:
        counts = row.get('method_status_counts') or {}
        state = label(row['status'])
        if row['name'] == '阿托伐他汀' and row['status'] == 'not_started':
            followers = [pipelines.get(n, {}) for n in ('阿托伐他汀方法接续', '阿托伐他汀提案接续')]
            live = next((p for p in followers if p.get('registered_process_alive') is True), None)
            state = label(live['status']) if live else '未启动/需处理'
        external = pipelines.get(EXTERNAL_PIPELINES.get(row['name']))
        if external:
            evaluation = label(external.get('stage') if external['status'] == 'running' else external['status'])
            if external['status'] == 'completed':
                evaluation = '独立评价已结束'
        elif row['name'] == '克拉霉素重建修复':
            evaluation = '独立接续尚未接通'
        else:
            evaluation = '见既有结果'
        rows.append([row['name'], state, f"{sum(counts.values())}/{row.get('planned_method_jobs', 57)}",
                     counts.get('completed', 0), counts.get('budget_exhausted', 0),
                     sum(v for k, v in counts.items() if k not in {'completed', 'budget_exhausted'}),
                     duration(row.get('elapsed_seconds')), evaluation])
    lines.append(table(rows))
    lines.append('已结束含预算耗尽和其他终止；不代表全部搜索完成或预测质量合格。')
    if 'qwen_atorvastatin' in report:
        qwen = report['qwen_atorvastatin']
        lines.append(f"阿托伐他汀 Qwen：{label(qwen['state'])}，{qwen['completed']}/{qwen['total']} 条提案。")
    if details or history:
        lines.append('\n进程与自动接续：')
        for row in report['pipelines']:
            if not history and row['name'] in {'克拉霉素方法接续', '克拉霉素修复接续', '伏立康唑方法接续', '伏立康唑外部验证', '呋塞米外部验证'}:
                continue
            if (not history and row['name'] == '阿托伐他汀提案接续' and row['status'] == 'stopped'
                    and any(s['name'] == '阿托伐他汀' and s['status'] in {'running', 'completed'} for s in displayed)):
                continue
            lines.append(f"  {row['name']}: {label(row['status'])}; PID={row.get('pid', '-')}; 存活={row.get('registered_process_alive', '-')}")
            if row.get('reason'):
                lines.append('    原因：' + row['reason'])
            if row.get('status_path'):
                lines.append('    状态文件：' + str(ROOT/row['status_path']))
        for row in displayed:
            if row.get('registered_runner_alive'):
                lines.append(f"  {row['name']} 实验主进程 PID={row['runner_pid']}")
            for task in row.get('active_method_progress', []):
                budget = task['budget']
                lines.append(f"  {row['name']} {task['method']}/seed {task['seed']} {task['stage']}: "
                    f"逻辑调用={budget.get('logical_calls', '-')}, 实际调用={budget.get('physical_calls', '-')}, "
                    f"求解失败={budget.get('failed_calls', '-')}")
    data = report.get('data', {})
    lines.append(f"\n资料：{data.get('candidate_drugs', '-')} 种候选药物，{data.get('protocol_records', '-')} 份协议资料，"
                 f"{data.get('numerical_curve_drugs', '-')} 药有完整数值曲线；候选数不等于已完成评测数。")
    if data.get('background_data_acquisition_running') is False:
        lines.append('资料下载：本轮已收尾，没有持续下载进程。')
    plan = report.get('research_plan')
    if plan:
        if report.get('experiment_hold', {}).get('development_status') == 'code_verified_waiting_for_small_evaluation':
            lines.append('当前下一步：新算法的 2–3 药小规模对照；以下为暂缓的历史完整目标。')
        lines.append(f"完整计划尚未全部排队：{plan['development_drugs_target']} 种开发药物 + "
            f"{plan['independent_evaluation_drugs_target']} 种独立评测药物，{plan['planned_design_runs_target']:,} 次目标运行。")
        lines.extend('  后续需准备并启动：' + text for text in plan['requires_later_preparation_and_launch'])
    lines.append('仅查看登记进程和进度，不读取封存观测或评分；退出查看不会停止实验。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='只读查看 PBPK 实验状态；不调用 LLM，不控制实验进程。')
    parser.add_argument('--json', action='store_true', help='机器可读输出；监看模式每行一个 JSON')
    parser.add_argument('--watch', type=float, nargs='?', const=10, metavar='秒', help='自动刷新，默认 10 秒，Ctrl+C 只退出查看')
    parser.add_argument('--history', action='store_true', help='同时显示已被修复版本替代的旧尝试')
    parser.add_argument('--details', action='store_true', help='显示 PID、当前方法、调用数及停止原因')
    parser.add_argument('--count', type=int, metavar='次数', help='监看若干次后自动退出')
    args = parser.parse_args(argv)
    if args.watch is not None and not 2 <= args.watch <= 3600:
        parser.error('--watch 间隔须在 2..3600 秒之间')
    if args.count is not None and (args.count < 1 or args.watch is None):
        parser.error('--count 须为正整数，并与 --watch 一起使用')
    try:
        iteration = 0
        while True:
            try:
                report = snapshot()
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                print('无法可靠读取当前状态：' + str(exc), file=sys.stderr)
                if args.watch is None:
                    return 1
            else:
                if args.watch is not None and not args.json and sys.stdout.isatty():
                    print('\033[2J\033[H', end='')
                if args.json:
                    print(json.dumps(report, ensure_ascii=False, indent=None if args.watch is not None else 2), flush=True)
                else:
                    print(render(report, history=args.history, details=args.details), flush=True)
            iteration += 1
            if args.watch is None or args.count is not None and iteration >= args.count:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        if not args.json:
            print('\n已退出进度查看，未改变实验状态。')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
