#!/usr/bin/env python3
"""Publish compact reviewer-diagnostic tables, preserving original manuscript."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/model_recommendation/reviewer_revision_20261009'
PAPER=BASE/'paper'
def read(p):return json.loads(p.read_text())
def write(name,s):PAPER.mkdir(parents=True,exist_ok=True);(PAPER/name).write_text(s)

def component_table(source):
    """Format saved component diagnostics without updating other audit outputs."""
    rows={r['method']:r for r in source['summary']}
    names=[('original','Original request'),('components','Full components'),('text_only','Text only'),
           ('structure_only','Structured fields only'),('structure_without_process_types','Structured, no generic ADME'),
           ('guarded_components','Conservative fallback')]
    lines=[r'\begingroup',r'\fontsize{9.3}{11.8}\selectfont',
        r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.12}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrr@{}}',r'\toprule',
        r'Input intervention & NDCG@10 ($\uparrow$) & Top-1 Cmax GMFE ($\downarrow$) \\',r'\midrule']
    for key,name in names:
        if key in ('components', 'structure_without_process_types'):
            lines.append(r'\addlinespace[4pt]')
        r=rows[key];c=r['cmax'][0]
        if key=='original':name=r'\textbf{'+name+'}'
        lines.append(f"{name} & {r['quality']['metrics']['ndcg@10']:.4f} & {c['gmfe']:.3f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular*}',r'\endgroup']
    return '\n'.join(lines)+'\n'

def component():
    source=read(BASE/'component_diagnostics/evaluation.json')
    write('component_diagnostics.tex',component_table(source))
    text='为定位组件增补的退步，固定原有留出权重，分别只改变请求文本或结构化字段。仅追加文本使已判子集NDCG@10由0.9095降至0.9053，首位Cmax GMFE保持1.635；仅改变结构字段时NDCG为0.9115，但GMFE升至1.900。撤去新增的泛ADME过程字段后，两项指标均恢复原值。地高辛的首位更替同样由结构字段触发：原首位倍数误差1.095，替代候选37.773。桥接将软建议并入普通条件特征，而旧排序头未区分两者；其中“distribution”的引用仅为分子量，其他引文还涉及诱导或胎儿条件，来源可定位不等于支持当前用途。\n\n另以不读取标签或实测误差的保守规则回退：只有新增具体酶、转运体或器官、来源条件无显式冲突且首位适用概率不低于0.5时才保留增强。50条增强请求中仅2条文档请求通过，9条执行请求全部回退，故首位误差恢复是限制启用范围的结果，不能解释为新的预测改善。组件增强与误差修正均保留为显式可选项；本诊断不构成独立验证。\n'
    write('component_diagnostics_zh.txt',text)

def graph():
    source=read(BASE/'graph_diagnostics/summary.json')
    if source['status']!='complete':return
    comparisons={r['before']:r for r in source['comparisons'] if r['metric']=='ndcg_at_10'}
    sorted_p=sorted(comparisons,key=lambda k:comparisons[k]['exact_group_sign_flip_p'])
    adjusted={};last=0.
    for i,k in enumerate(sorted_p):
        last=max(last,min(1.,comparisons[k]['exact_group_sign_flip_p']*(len(sorted_p)-i)));adjusted[k]=last
    names=[('no_gnn','NoGNN'),('hgt','HGT (full)'),('no_message_passing','HGT, no message passing'),
           ('no_teacher_initialization','HGT, no teacher initialization'),
           ('no_teacher_initialization_or_anchor','HGT, no initialization or anchor'),
           ('hgt_no_residual_at_original_checkpoint','HGT, no residual at fixed checkpoint')]
    lines=[r'\begingroup',r'\fontsize{7.6}{9.6}\selectfont',r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & NDCG@10 & Recall@10 & $\Delta$NDCG [95\% CI] & $p_{\mathrm{Holm}}$ \\',r'\midrule']
    for key,name in names:
        m=source['methods'][key];r=comparisons.get(key)
        difference='---' if r is None else f"{r['difference']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        p='---' if r is None else f'{adjusted[key]:.3f}'
        lines.append(f"{name} & {m['ndcg_at_10']:.4f} & {m['recall_at_10']:.4f} & {difference} & {p} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\endgroup']
    write('graph_diagnostics.tex','\n'.join(lines)+'\n')
    a=comparisons['no_gnn'];b=comparisons['no_message_passing']
    write('graph_diagnostics_zh.txt',f"在相同银标签与原有划分上，补充3种子控制训练，并将38条测试请求按16个语义关联组进行配对重采样；种子先在每条查询内平均，不视为新增样本。HGT相对NoGNN的NDCG增量为{a['difference']:.4f}（95%区间[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]，五项NDCG比较Holm校正p={adjusted['no_gnn']:.3f}）。保留教师初始化、锚定及融合而关闭图消息传递后，NDCG为{source['methods']['no_message_passing']['ndcg_at_10']:.4f}；其与完整HGT的增量区间为[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]。这些结果用于分解当前实现的贡献，不能替代独立数据验证。去残差一项只在原选定checkpoint上关闭融合，未重新按无融合验证分数早停，故仅解释为固定权重推理消融。\n")
    write('graph_statistics.json',json.dumps({'family':'five NDCG contrasts: full HGT minus comparison',
        'holm_p':adjusted,'bootstrap':'10000 semantic-group resamples; query-weighted difference; three seeds averaged within query',
        'test':'exact two-sided sign flips of 16 semantic groups'},indent=2)+'\n')

if __name__=='__main__':component();graph()
