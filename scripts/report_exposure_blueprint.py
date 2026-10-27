#!/usr/bin/env python3
"""Compact report adapter for the frozen exposure experiment, without refitting."""
from pathlib import Path
import fcntl
import json
import math

import run_exposure_blueprint as batch
from report_blueprint_validation import paired_ratio


def main():
    state=batch.read(batch.RUN/'state.json')
    if state['status'] not in {'predictions_locked','complete'}:
        raise ValueError('All predictions must be locked before reporting')
    protocol=batch.checked(state['protocol'])
    units={**state['units'],**protocol['controls']}
    original_save=batch.save
    def compact_save(path,value):
        if Path(path)==batch.RUN/'results.json':
            for record in value['records']:
                curve=record.pop('curve',None)
                if curve is not None:
                    key=f"{record['case_id']}__{record['sampling']}__{record['method']}"
                    record['prediction_source']=units[key]['prediction']
                    record['curve_storage']='Full locked prediction referenced, not duplicated in aggregate JSON'
            value['report_adapter']=batch.ref(Path(__file__))
            value['report_note']='Formatting-only correction after all predictions locked: omit duplicated dense curves; scoring, all endpoints and NCA diagnostics unchanged.'
            clusters={case['case_id']:case['cluster_id'] for case in protocol['cases']}
            value['development_method_choice']={'method':'direct_relaxed',
                'analysis_type':'Post-hoc method choice in an already viewed development cohort; not independent confirmation.',
                'scope':'A drug-specific development curve remains required; no claim of improved zero-curve prediction.',
                'reason':'Lower cohort GMFE in both full and five-point conditions than fixed_prior and exposure_constrained; retain all regressions and the negative exposure-constraint comparison.',
                'direct_over_fixed_prior':{},'excluding_ertapenem_exploratory':{},'ertapenem_share_of_net_log_improvement':{}}
            for sampling in batch.SAMPLINGS:
                scores={method:{r['case_id']:r['curve_metrics'].get('gmfe') for r in value['records'] if r['method']==method and r['sampling']==sampling} for method in ('direct_relaxed','fixed_prior')}
                value['development_method_choice']['direct_over_fixed_prior'][sampling]=paired_ratio(scores['direct_relaxed'],scores['fixed_prior'],clusters)
                ertapenem=next(c['case_id'] for c in protocol['cases'] if c['drug']=='Ertapenem')
                subset={k:v for k,v in clusters.items() if k!=ertapenem}
                value['development_method_choice']['excluding_ertapenem_exploratory'][sampling]=paired_ratio(scores['direct_relaxed'],scores['fixed_prior'],subset)
                ratios=value['development_method_choice']['direct_over_fixed_prior'][sampling]['per_case_ratio']
                value['development_method_choice']['ertapenem_share_of_net_log_improvement'][sampling]=math.log(ratios[ertapenem])/sum(map(math.log,ratios.values()))
        original_save(path,value)
    with (batch.RUN/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batch.save=compact_save
        result=batch.report()
        document=batch.read(batch.RUN/'results.json')
        comparisons=document['development_method_choice']['direct_over_fixed_prior']
        lines=['','## 本轮结论','',
            '当前有开发曲线分支采用direct_relaxed作为下一轮研发基线：保留人体双区和固定净肾输入，去掉非肾CL的强先验惩罚，扩大数值搜索区间，仍只拟合CL/K/PS三个参数。来源CL仅作初值和倍率参照，原始证据账本不改写。',
            '这是查看本轮开发结果后的方法选择，不是新盲测确认；没有改动原论文、无本药曲线分支或逐药拼接最优方法。', '',
            '| 条件 | direct / fixed GMFE比 | 95%研究簇区间 | 改善 / 退步 |','|---|---:|---|---:|']
        for sampling in batch.SAMPLINGS:
            c=comparisons[sampling];lines.append(f'| {sampling} | {c["estimate"]:.4f} | [{c["ci95"][0]:.4f}, {c["ci95"][1]:.4f}] | {c["ratio_below_one"]} / {c["ratio_above_one"]} |')
        lines+=['',
            '直接放宽CL优于强制开发窗面积相等；后者在Treprostinil的完整/五点GMFE为4.304/4.227，直接校准为2.375/2.264。面积一致不等于目标研究预测准确，不采用该硬约束为主方法。',
            '改进并非所有药物、所有终点一致获益：Cefepime、Midazolam、Nemonoxacin曲线误差上升；不能把平均改善解释为临床可靠。两室冻结参照总体仍更低，预算不同。',
            '两种采样的配对区间都跨1。Ertapenem贡献约81%的净log-GMFE改善；仅探索性排除它后，其余九药平均改善约1.9%，区间仍跨1。因此不是已经证明普遍优胜。',
            '末次浓度仅2药有参考，完整/五点终点GMFE由2.915/3.041退步到3.827/3.173。Remdesivir半衰期仍约6.34/6.42倍并带弱尾标记；其余各终点覆盖和弱尾统计见results.json，不将相关尾部指标视为独立重复证据。',
            '五点相对完整曲线的GMFE增幅约1.15%，原法约0.97%，未证明少点稳定性进一步提升；减少采样点也不等同于缺CL或fu的证据缺失实验。',
            '', '## 验收与计算记录','',
            '60个新单元、120次公式拟合全部得到可用的单元预测；连同40个冻结参照，共100个比较单元、1,100条终点记录。两次单公式无解由另一公式完成，失败原因保留。',
            '32项工程/报告检查通过。独立算术验收通过：831项文件哈希、1,040个曲线点比较、800个标量误差；2000次研究簇bootstrap、固定肾输入及48项体重迁移检查通过。这里的1,040点/800项包含多方法重复比较，不是独立样本数。',
            '正式拟合300,161次数学求解；仅开发预检21,999次另计。5组新原生配置按共享缓存首次构建归属，不作为方法速度比较；本轮未发现原生构建失败。reference资源是旧运行记录，非本批新增；并发unit wall_seconds包含等待，不能相加当总运行时长。',
            '报告曾因重复嵌入稠密预测曲线超过32 MiB上限而中止；仅增加格式适配器，保留原预测文件引用，评分数值、冻结预测和校准代码均未改变。',
            '', '复算报告：`runtime/app-env/bin/python -B scripts/report_exposure_blueprint.py`。原`run_exposure_blueprint.py`保留冻结代码，用于status、完成状态检查和来源追溯。']
        readme=batch.RUN/'README.md'
        text=readme.read_text().replace('，`report`重算结果。','；结果复算使用下方紧凑报告入口。')
        readme.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
