#!/usr/bin/env python3
"""Analyze locked policy predictions; optionally recover reporting without a live monitor.

Adds no simulations, fits, thresholds, or changes to the frozen experiment.
Post-hoc endpoint comparisons and four-drug input-effect diagnostics are
exploratory. Summary is stored in existing state.json and experiments/README.md.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
from datetime import datetime

import run_blueprint_evidence_policy as runner
from report_blueprint_evidence_policy import report, _all_refs
from report_blueprint_validation import Store, paired_ratio

ROOT = runner.ROOT
BEGIN = '<!-- evidence-policy-analysis:start -->'
END = '<!-- evidence-policy-analysis:end -->'


def finalize(run):
    """Never overlap a monitor's mutable state with the report's integrity read."""
    run = runner.own_run(run)
    with (run/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if runner.status(run).get('alive'):
            raise RuntimeError('Live runner: wait for it to exit before independent finalization')
        state = runner.read(run/'state.json')
        if state['status'] == 'complete':
            return report(run, project_root=ROOT)
        protocol = runner.checked(state['protocol'])
        if not state.get('all_attempts_locked_before_scoring') or not state.get('locked_at'):
            raise RuntimeError('No complete prediction lock; finalization cannot bypass experiments')
        if set(state['computations']) != set(protocol['computations']):
            raise RuntimeError('Incomplete prediction computations')
        runner.verify(protocol['source_pins'])
        store = Store(ROOT)
        _all_refs(protocol, store); _all_refs(state, store); store.unchanged()
        state.setdefault('report_recovery', []).append({'time':runner.now(),
            'previous_status':state['status'], 'previous_reason':state.get('pause_reason'),
            'action':'Standalone report under exclusive lock with no monitor; all model inputs and predictions unchanged'})
        state['status'] = 'predictions_locked'; state.pop('pause_reason', None)
        runner.save(run/'state.json', state)
        try:
            result = report(run, project_root=ROOT)
        except Exception as exc:
            state.update(status='paused',pause_reason=type(exc).__name__+': '+str(exc))
            runner.save(run/'state.json',state)
            raise
        state.update(status='complete',completed_at=runner.now(),results=runner.ref(run/'results.json'))
        runner.save(run/'state.json',state)
        return result


def analyze(run):
    run = runner.own_run(run)
    state = runner.read(run/'state.json')
    if state['status'] != 'complete': raise ValueError('Finalize all locked predictions before analysis')
    result = report(run, project_root=ROOT)  # Verify locks; completed reports are not rewritten.
    protocol = runner.checked(state['protocol'])
    records = {(u['case_id'],u['scenario'],u['sampling'],u['method']):u for u in result['units']}
    cases = protocol['cases']; ids = [c['case_id'] for c in cases]
    clusters = {c['case_id']:c['cluster_id'] for c in cases}
    changed = []
    for cid in ids:
        evidence = {}
        for method in ('evidence_policy','rf_fill'):
            d = next(d for d in protocol['decisions'] if (d['case_id'],d['scenario'],d['method'])==(cid,'available',method))
            evidence[method] = {e['parameter']:e['value'] for e in d['decision']['selected_evidence']}
        differences = {k for k in evidence['evidence_policy'] if evidence['evidence_policy'][k]!=evidence['rf_fill'][k]}
        if differences - {'fraction_unbound'}: raise ValueError('This batch was intended to change RF fu only')
        if differences: changed.append(cid)
    def scores(sampling,method,endpoint=None,subset=ids,nonweak=False):
        output={}
        for cid in subset:
            row=records[cid,'available',sampling,method]
            item=row.get('endpoints',{}).get(endpoint,{}) if endpoint else {}
            value=item.get('error') if endpoint else row['curve_metrics'].get('gmfe')
            if value is not None and (not nonweak or not item.get('weak_tail')): output[cid]=value
        return output
    pairs={}
    for sampling in ('full','five'):
        pairs[sampling]={}
        for endpoint in result['statistics']['groups']['available'][sampling]['evidence_policy']['endpoints']:
            if endpoint=='tmax':continue
            a=scores(sampling,'evidence_policy',endpoint);b=scores(sampling,'two_compartment',endpoint)
            clean_a=scores(sampling,'evidence_policy',endpoint,nonweak=True)
            clean_b=scores(sampling,'two_compartment',endpoint,nonweak=True)
            pairs[sampling][endpoint]={'all_scored':paired_ratio(a,b,clusters,expected_ids=ids),
                'both_predictions_nonweak':paired_ratio(clean_a,clean_b,clusters,expected_ids=ids)}
    effective={sampling:paired_ratio(scores(sampling,'evidence_policy',subset=changed),
        scores(sampling,'rf_fill',subset=changed),clusters,expected_ids=changed) for sampling in ('full','five')}
    trials=Counter(); reasons=Counter(); axes=Counter(); calls=0
    for fit in state['fits'].values():
        mapping=runner.checked(fit['mapping']);calls+=mapping['total_native_calls']
        axes[','.join(mapping['screening']['selected_parameters'])]+=1
        for row in mapping['evaluations']:
            trials[row['status']]+=1
            if row['status']!='ok':reasons[str(row.get('reason','unspecified'))]+=1
    metrics=result['statistics']['groups']['available']
    rows=[]
    for c in cases:
        cid=c['case_id']
        rows.append({'drug':c['drug'],'case_id':cid,**{m:{s:records[cid,'available',s,m]['curve_metrics']['gmfe'] for s in ('full','five')}
            for m in ('evidence_policy','two_compartment','rf_fill')}})
    summary={'analyzed_at':runner.now(),'script':runner.ref(Path(__file__).resolve()),
        'source_result':runner.ref(run/'results.json'),
        'analysis_role':'Post-hoc diagnosis of a previously viewed development cohort; no new hypothesis confirmation',
        'unique_predictions':len(state['computations']),'logical_units':len(protocol['units']),
        'scoring_statuses':dict(Counter(r['scoring_status'] for r in result['units'])),
        'native_fit_calls':calls,'native_target_calls':38,
        'fit_statuses':dict(Counter(f['status'] for f in state['fits'].values())),
        'screened_axes':dict(axes),'trial_statuses':dict(trials),'trial_failures':dict(reasons),
        'calculation_seconds':(datetime.fromisoformat(state['locked_at'])-datetime.fromisoformat(state['started_at'])).total_seconds(),
        'effective_fu_case_ids':changed,'effective_fu_policy_over_rf':effective,
        'endpoint_policy_over_two_compartment':pairs,'per_drug_curve_gmfe':rows,
        'available_curve_summary':{s:{m:values['curve'] for m,values in metrics[s].items()} for s in ('full','five')},
        'interpretation':['Missing-fu policy outputs equal RF-fu outputs and do not outperform two-compartment PK.',
            'Policy and fixed-template predictions are exactly shared under available evidence; clinical policy superiority is not established there.',
            'All engineering acceptance labels follow prescribed rules and are not independent clinical accuracy labels.',
            'Endpoint contrasts are exploratory without multiplicity correction; AUC infinity and clearance are mathematically coupled.',
            'A diagnostic probe consumes native budget; current results do not isolate a causal effect of sensitivity screening.']}
    return result,summary


def markdown(result,summary):
    out=[BEGIN,'### 当前：证据构建与用途小试（2026-09-29，已完成分析）','',
        '78/78组实际计算成功：38组PBPK和40组两室。76个PBPK公式拟合完成；720个逻辑单元中440个有临床评分，160个因证据规则阻塞，120个DDI请求只评价能力边界。它们来自10药、9个研究簇，不能算作720项独立临床实验。',
        '', '计算约62分钟，1,808次校准调用加38次目标预测。试算中46次失败（42次清除能力/公式检查、4次浓度非正），均保留；最终预测全部成功。22/76公式拟合达到数值收敛，54/76用满预算。报告首次因监控并发写入state触发完整性检查，已在进程退出后独占锁恢复评分，未改预测和源数据。','',
        '**曲线GMFE（按药等权，越低越好）：**','',
        '| 方法/条件 | 完整开发曲线 | 固定五点 |','|---|---:|---:|']
    for method,label in [('evidence_policy','本方法：可用证据'),('fixed_template','固定输入PBPK：可用证据'),('rf_fill','RF fu优先PBPK'),('two_compartment','两室PK'),('two_compartment_prior','两室PK＋CL先验')]:
        vals=[result['statistics']['groups']['available'][s][method]['curve']['estimate'] for s in ('full','five')]
        out.append(f'| {label} | {vals[0]:.3f} | {vals[1]:.3f} |')
    out.extend(['',
        '本方法相对两室的GMFE比为1.114 [0.950, 1.381]和1.050 [0.922, 1.232]；区间为2,000次研究聚类bootstrap。点估计仍落后，区间跨1不证明等效。五点下6/10药物误差更低，但药物等权总体仍未超过两室。正常输入时本方法和固定输入对照复用相同计算，数值相同是预期结果。','',
        '**证据收益与代价：** 四药实际改变fu，证据优先均改善曲线，四药配对GMFE比为0.731 [0.551, 0.971]（完整）和0.731 [0.481, 0.987]（五点）。其余六药输入相同，不算新增消融证据。缺fu或剔除错误物种fu后，本方法用RF补缺，GMFE为1.708/1.695，与RF fu对照完全相同；固定输入流程阻塞10/10，两室仍可运行。收益是可继续构建，不是缺信息后精度优于两室。','',
        '**工程处置：** 本方法60个请求放行30个、阻塞30个，按预设规则无误放行/误阻塞。固定输入流程额外阻塞20个可补缺请求；RF fu优先和去掉冲突保护均放行10个矛盾fu请求。两室不需要fu和分子式，50个血浆请求均可执行、10个DDI请求阻塞。这是人工扰动与预设规则的一致性检验，不能写成临床决策准确率100%。','',
        '**终点分析：** 五点下本方法的AUC和Cmax平均倍数误差为1.087、1.367，两室为1.159、1.444；对应两倍内比例为6/6、9/10与6/6、8/10。但半衰期为1.859对1.447，Vz为1.845对1.470，仍落后。终点逐项比较属于探索性分析，没有多重检验校正；CL与AUC∞相关，不能当作两份独立优势。弱尾部预测继续标记，详见结果表。','',
        '**主要问题：** Remdesivir曲线GMFE为2.992/2.255，是唯一仍超过2的药物。五点下AUC误差仅1.031倍，但半衰期5.482倍、Vz 5.218倍，后两项有弱尾标记，说明暴露积分改善没有解决末端曲线偏差。剔除预先指定的Nemonoxacin、Dexmedetomidine、Midazolam后，七药GMFE为1.604/1.598，两室为1.326/1.427，结论没有逆转。','',
        '**后续方向：** 先诊断Remdesivir的分布/尾段与开发拟合误差，区分参数化限制和搜索未充分；不根据目标观测调参后重新宣称独立验证。保留证据补缺及冲突保护，补真实缺失/冲突案例；局部灵敏度是否有独立贡献需要另作相同预算消融。当前不自动启动新一轮训练，也不覆盖论文PDF。','',
        '[最终逐项结果CSV](blueprint_design/evidence_policy_20260929/results.csv) · [完整评分JSON](blueprint_design/evidence_policy_20260929/results.json) · [后续诊断](blueprint_design/evidence_policy_20260929/state.json) · [复核入口](../scripts/analyze_blueprint_evidence_policy.py)','',
        '```bash','runtime/app-env/bin/python -B scripts/analyze_blueprint_evidence_policy.py --write-summary',
        '# 若已全部锁定、仅因报告状态冲突暂停，使用独立恢复入口（不会重新模拟）：',
        'runtime/app-env/bin/python -B scripts/analyze_blueprint_evidence_policy.py --finalize --write-summary','```',END])
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=runner.DEFAULT)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--write-summary',action='store_true')
    args=parser.parse_args(); run=runner.own_run(args.run_dir)
    if args.finalize: finalize(run)
    result,summary=analyze(run)
    if args.write_summary:
        if run != runner.DEFAULT: raise ValueError('README summary is scoped to the frozen 20260929 batch')
        with (run/'.run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if runner.status(run).get('alive'):raise RuntimeError('Cannot edit summary while runner is active')
            state=runner.read(run/'state.json');state['analysis']=summary;runner.save(run/'state.json',state)
            path=ROOT/'experiments/README.md';text=path.read_text();block=markdown(result,summary)
            if BEGIN in text:
                start=text.index(BEGIN);end=text.index(END,start)+len(END);text=text[:start]+block+text[end:]
            else:
                anchor='## 4. 实际仿真与蓝图\n';assert anchor in text
                text=text.replace(anchor,anchor+'\n'+block+'\n',1)
            path.write_text(text)
    print(json.dumps({k:summary[k] for k in ('unique_predictions','scoring_statuses','fit_statuses','trial_statuses','effective_fu_policy_over_rf')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
