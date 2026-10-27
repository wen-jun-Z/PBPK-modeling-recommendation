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
    text='To diagnose deterioration after component augmentation, original held-out weights were fixed while only request text or structured fields changed. Text augmentation reduced judged-subset NDCG@10 from 0.9095 to 0.9053 while top-1 Cmax GMFE stayed 1.635. Changing only structured fields gave NDCG 0.9115 but GMFE 1.900. Removing added generic ADME-process fields restored both metrics. The Digoxin top-1 change also came from structured fields: fold error was 1.095 for the original candidate and 37.773 for the replacement. The bridge inserted soft suggestions into ordinary condition features, which the old ranking head did not distinguish. The distribution citation supported only molecular weight; other citations concerned induction or fetal conditions. A traceable source does not necessarily support the current purpose.\n\nA conservative fallback without labels or measured errors retains augmentation only when it adds a specific enzyme, transporter, or organ, has no explicit source-condition conflict, and top-1 applicability probability is at least 0.5. Only two document requests passed among 50 augmented requests; all nine execution requests fell back. Restored top-1 error therefore reflects restricted activation, not new predictive improvement. Component augmentation and error correction remain explicit options; this diagnosis is not independent validation.\n'
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
    write('graph_diagnostics_zh.txt',f"Three-seed controls reuse the same silver labels and split. Paired resampling groups 38 test requests into 16 semantic groups, averaging seeds within queries rather than treating them as new samples. HGT minus NoGNN NDCG is {a['difference']:.4f} (95% interval [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]; Holm-adjusted p across five NDCG comparisons={adjusted['no_gnn']:.3f}). Keeping teacher initialization, anchoring, and fusion but disabling message passing gives NDCG {source['methods']['no_message_passing']['ndcg_at_10']:.4f}; its difference interval against full HGT is [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]. This decomposes the current implementation, not independent-data validation. No-residual disables fusion at the originally selected checkpoint without new early stopping on no-fusion validation scores, so it is a fixed-weight inference ablation only.\n")
    write('graph_statistics.json',json.dumps({'family':'five NDCG contrasts: full HGT minus comparison',
        'holm_p':adjusted,'bootstrap':'10000 semantic-group resamples; query-weighted difference; three seeds averaged within query',
        'test':'exact two-sided sign flips of 16 semantic groups'},indent=2)+'\n')

if __name__=='__main__':component();graph()
