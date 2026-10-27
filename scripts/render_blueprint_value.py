#!/usr/bin/env python3
"""Render one source-bound clinical-value figure and one English result table.

The model and evaluation remain frozen. No training, simulation, case selection,
or score changes occur here. A preview is temporary; final assets live in the
experiment's paper directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))
import numpy as np
import run_blueprint_value as runner
from pbpk_kb.blueprints.source_workflow import save, sha256

SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill')
VERSION = 'clinical-blueprint-value-publication-1'
CONDITIONS = ('all_ml', 'available', 'missing_clearance', 'missing_fu', 'missing_both')
LABELS = {'all_ml': 'Predicted inputs only', 'available': 'Evidence priority',
          'missing_clearance': 'Remove CL evidence', 'missing_fu': 'Remove fu evidence',
          'missing_both': 'Remove both'}
COLORS = ['#0072B2', '#D55E00', '#009E73']


def gm(values):
    values = list(values)
    return math.exp(math.fsum(map(math.log, values)) / len(values)) if values else None


def number(value, digits=3):
    return '---' if value is None else f'{value:.{digits}f}'


def verify_data(run):
    state = runner.old.read(run / 'state.json')
    if state['status'] != 'complete' or not state['all_predictions_locked_before_scoring']:
        raise ValueError('A completed prediction-first experiment is required')
    protocol = runner.old.checked(state['protocol'])
    runner.verify(protocol)
    result = runner.old.checked(state['results'])
    for ref in state['outcomes'].values():
        runner.old.checked(ref)
    rows = result['units']
    by = {(r['case_id'], r['condition']): r for r in rows}
    ids = {c['case_id'] for c in protocol['cases']}
    if len(by) != 100 or set(by) != {(cid, con) for cid in ids for con in CONDITIONS}:
        raise ValueError('All twenty cases and five conditions must remain')
    changed = set(protocol['changed_case_ids'])
    if len(changed) != 8:
        raise ValueError('Unexpected evidence-intervention population')
    for summary in result['summaries']:
        if summary['group'] not in {'all_20', 'changed_8'}:
            continue
        group = ids if summary['group'] == 'all_20' else changed
        vals = [by[cid, summary['condition']]['curve_metrics'].get('gmfe') for cid in group]
        valid = [v for v in vals if v is not None]
        if len(valid) != summary['scored'] or not math.isclose(gm(valid), summary['curve']['gmfe'], rel_tol=1e-12):
            raise ValueError('Curve summary does not reproduce per-drug scores')
        for threshold in (2, 3, 5):
            if sum(v < threshold for v in valid) != summary['curve']['within_fold'][str(threshold)]['count']:
                raise ValueError('Fold-threshold count changed')
    return state, protocol, result, by


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    if not fonts.resolve().is_relative_to(ROOT):
        raise ValueError('Use the project-local plotting environment')
    for font in fonts.glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
        'axes.titlesize': 8.5, 'legend.fontsize': 7.5, 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'text.color': '#243B4A', 'axes.labelcolor': '#243B4A',
        'axes.edgecolor': '#778593', 'xtick.color': '#43525D', 'ytick.color': '#43525D'})


def draw(protocol, result, by):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig = plt.figure(figsize=(5.5, 5.55), facecolor='white')
    ax = fig.add_axes([.23, .52, .74, .34])
    drugs = [c for c in protocol['cases'] if c['case_id'] in protocol['changed_case_ids']]
    for y, case in enumerate(drugs):
        cid = case['case_id']
        vals = [by[cid, cond]['curve_metrics'].get('gmfe') for cond in ('all_ml', 'available')]
        if all(v is not None for v in vals):
            ax.plot(vals, [y, y], color='#ABB6BD', linewidth=1.1, zorder=1)
        for value, color, marker in zip(vals, COLORS, ('o', '^')):
            if value is not None:
                ax.scatter(value, y, s=27, marker=marker, facecolor=color,
                           edgecolor='white', linewidth=.45, zorder=3)
    ax.set_yticks(range(len(drugs)), [c['drug'] for c in drugs])
    ax.set_ylim(len(drugs)-.5, -.5)
    vals = [by[c['case_id'], con]['curve_metrics'].get('gmfe') for c in drugs for con in ('all_ml', 'available')]
    valid = [v for v in vals if v is not None]
    ax.set_xscale('log'); ax.set_xlim(.95, max(valid)*1.15)
    ticks = [v for v in (1, 2, 3, 5, 10, 20, 30, 50, 100) if v <= max(valid)*1.15]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Curve GMFE (lower is better; log scale)', labelpad=5)
    ax.grid(axis='x', color='#E3E9EE', linewidth=.6)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.025, .978, 'A  Does independent evidence improve prediction?', fontsize=9.2, fontweight='bold', va='top')
    fig.text(.025, .940, 'Eight drugs with changed inputs; identical PBPK core and predicted Vss / half-life', fontsize=7.2, color='#5D6D78')
    handles = [Line2D([], [], color=c, marker=m, linestyle='none', markersize=5)
               for c, m in zip(COLORS, ('o','^'))]
    fig.legend(handles, ['Predicted inputs only', 'Evidence priority'], loc='upper center',
               bbox_to_anchor=(.58,.924), ncol=2, frameon=False, handletextpad=.35, columnspacing=1.3)
    specs = [(.115, 'B  Clinical error after removal', 'GMFE ratio to available inputs', False),
             (.60, 'C  Prediction change', 'Prediction drift (fold)', True)]
    for left, title, ylabel, drift in specs:
        sub = fig.add_axes([left, .16, .345, .225])
        allvals = []
        counts = []
        for i, condition in enumerate(CONDITIONS[2:]):
            rows = [r for r in result['units'] if r['condition'] == condition and r['changed_from_nominal']]
            values = []
            for r in rows:
                if drift:
                    value = r['prediction_drift']['curve_geometric_fold_change']
                else:
                    base = by[r['case_id'], 'available']['curve_metrics'].get('gmfe')
                    changed = r['curve_metrics'].get('gmfe')
                    value = changed/base if base and changed else None
                if value is not None:
                    values.append(value)
            allvals.extend(values); counts.append(len(values))
            offsets = np.linspace(-.17, .17, len(values))
            sub.scatter(i+offsets, values, color=COLORS[i], s=15, alpha=.85,
                        marker=('o','^','s')[i], edgecolor='white', linewidth=.3, zorder=3)
            if values:
                sub.scatter(i, gm(values), marker='D', s=32, facecolor='white', edgecolor='#243B4A', linewidth=1, zorder=4)
        sub.axhline(1, color='#788590', linestyle='--', linewidth=.7, zorder=1)
        sub.set_yscale('log')
        low = min([1., *allvals]); high = max([1., *allvals]); span = max(math.log(high/low), .5)
        sub.set_ylim(low*math.exp(-.10*span), high*math.exp(.16*span))
        ticks = [v for v in (.1,.2,.5,1,2,5,10,20,50,100) if sub.get_ylim()[0] <= v <= sub.get_ylim()[1]]
        sub.yaxis.set_major_locator(FixedLocator(ticks)); sub.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:g}'))
        sub.yaxis.set_minor_locator(NullLocator())
        sub.set_xticks(range(3), [f'{label}\n(n={n})' for label,n in zip(('CL','fu','Both'),counts)])
        sub.set_xlim(-.45,2.45); sub.set_xlabel('Evidence removed', labelpad=4)
        sub.set_ylabel(ylabel, labelpad=5)
        sub.grid(axis='y', color='#E3E9EE', linewidth=.55)
        sub.spines[['top','right']].set_visible(False)
        fig.text(left-.05, .435, title, fontsize=8.5, fontweight='bold', va='top')
    fig.text(.5,.023,'Dots: individual drugs. Open diamonds: geometric mean. Only changed inputs counted.',
             ha='center', fontsize=6.7, color='#5D6D78')
    return fig


def make_table(result):
    lines = [r'\begin{table}[!htbp]', r'\captionsetup{name=Table}',
        r'\caption{Clinical value of evidence selection without same-drug curve fitting. The PBPK core and learned priors are identical across conditions. GMFE is drug-equal; threshold counts retain failed requests in the denominator. Repeated inputs are cached, not independent experiments.}',
        r'\label{tab:blueprint_value}', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabularx}{\linewidth}{@{}Xrrrrr@{}}', r'\toprule',
        r'Input policy & Scored & GMFE & $<2$ & $<3$ & $<5$\\', r'\midrule']
    for group, label in (('all_20','All 20 requested drugs'), ('changed_8','8 drugs with independent-evidence interventions')):
        lines.append(r'\multicolumn{6}{@{}l}{\textbf{'+label+r'}}\\')
        for con in CONDITIONS:
            row = next(s for s in result['summaries'] if s['group']==group and s['condition']==con)
            n = row['attempted']
            counts = [str(row['curve']['within_fold'][str(t)]['count'])+'/'+str(n) for t in (2,3,5)]
            lines.append(f"{LABELS[con]} & {row['scored']}/{n} & {number(row['curve']['gmfe'])} & " + ' & '.join(counts)+r'\\')
        lines.append(r'\midrule' if group=='all_20' else r'\bottomrule')
    lines += [r'\end{tabularx}', r'\vspace{3pt}', r'\begin{minipage}{\linewidth}\footnotesize',
        r'CL removal deletes total, hepatic and renal evidence together. Missing evidence uses the same frozen predictor. These descriptive fold thresholds do not establish clinical suitability. The 20-drug cohort was previously viewed during development.',
        r'\end{minipage}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=runner.DEFAULT)
    parser.add_argument('--preview-only',action='store_true')
    args = parser.parse_args(); run = runner.own(args.run_dir)
    state, protocol, result, by = verify_data(run)
    style(); sys.path.insert(0,str(SKILL/'scripts'))
    from visual_qa import audit_layout, render_preview
    fig = draw(protocol,result,by)
    issues = audit_layout(fig)
    preview = ROOT/'runtime/cache/paper_figures/blueprint_value_preview.png'
    render_preview(fig,str(preview),dpi=220)
    from PIL import Image
    with Image.open(preview) as im:
        im.convert('L').save(preview.with_name('blueprint_value_gray.png'))
    if any(level in {'FAIL','WARN'} for level, _ in issues):
        raise ValueError('Figure requires layout review: '+repr(issues))
    print(json.dumps({'preview':str(preview),'layout_issues':issues},ensure_ascii=False))
    if args.preview_only:
        return
    out = run/'paper'; out.mkdir(exist_ok=True)
    pdf = out/'clinical_value.pdf'; table = out/'clinical_value.tex'
    fig.savefig(pdf,metadata={'Title':'Clinical value of evidence-preserving PBPK construction','Creator':VERSION})
    table.write_text(make_table(result))
    manifest = {'version':VERSION,'protocol':state['protocol'],'results':state['results'],
        'renderer':runner.old.ref(Path(__file__)), 'assets':[runner.old.ref(pdf),runner.old.ref(table)],
        'clinical_population':20,'changed_input_population':8,'logical_rows':100,
        'width_inches':5.5,'figure_language':'English','layout_audit':issues,
        'new_training_calls':0,'new_solver_calls':0}
    manifest['failure_diagnostics'] = []
    for receipt in state['outcomes'].values():
        item = runner.old.checked(receipt)
        entry = item['entry']
        if entry['status'] == 'failed':
            audit = entry.get('design_audit', {})
            manifest['failure_diagnostics'].append({'case_id': item['case_id'],
                'prediction_receipt': receipt, 'original_wrapper_code': entry.get('failure'),
                'root_causes': audit.get('reasons', []),
                'target_vss_l': audit.get('target_vss_l'),
                'trajectory_calls': entry.get('linear_trajectory_calls'),
                'interpretation': 'Read root_causes and design_audit; the original wrapper code alone does not diagnose an NCA failure'})
    save(out/'publication_manifest.json',manifest)
    print(json.dumps({'figure':str(pdf),'table':str(table)}))


if __name__=='__main__': main()
