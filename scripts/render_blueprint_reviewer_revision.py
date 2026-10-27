#!/usr/bin/env python3
"""Publish the separate reviewer diagnostics; never fit or rewrite frozen scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))
RUN = ROOT / 'experiments/blueprint_design/reviewer_revision_20261009'
DIAG = RUN / 'diagnostics'
TABLE = ROOT / 'results/tables/blueprint_reviewer_diagnostics.tex'
FIG = ROOT / 'results/figures/experiments/blueprint_reviewer_tails'
AXES = ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale', 'blood_partition_scale')


def read(p):
    return json.loads(Path(p).read_text())


def number(v, digits=3):
    return '---' if v is None else f'{v:.{digits}g}'


def esc(s):
    return str(s).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


def longtable(caption, label, cols, headers, rows, note, *, column_padding_pt=2.5,
              continued_caption=False):
    head = '\\toprule\n' + ' & '.join(headers) + r' \\' + '\n\\midrule\n'
    continued = '\\caption[]{Continued.}\\\\\n' if continued_caption else ''
    return ('\\begingroup\\captionsetup[table]{name=Table}\n'
        f'\\fontsize{{7.5}}{{9.5}}\\selectfont\\setlength{{\\tabcolsep}}{{{column_padding_pt:g}pt}}\n'
        '\\setlength{\\LTleft}{\\fill}\\setlength{\\LTright}{\\fill}\\setlength{\\LTcapwidth}{\\linewidth}\n'
        '\\renewcommand{\\arraystretch}{1.15}\n'
        f'\\begin{{longtable}}{{{cols}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n'
        + head + '\\endfirsthead\n' + continued + head + '\\endhead\n'
        + '\\bottomrule\n\\endfoot\n' + '\n'.join(rows) + '\n\\end{longtable}\n'
        + '\\noindent\\parbox{\\linewidth}{' + note + '}\\par\\endgroup\n\n')


def compact_table(caption, label, cols, headers, rows, note, *, column_padding_pt=2.5):
    """Keep short comparison tables and their notes together before the next topic."""
    return ('\\begin{table}[!htbp]\n\\centering\n'
        '\\begingroup\\captionsetup{name=Table}\n'
        f'\\fontsize{{7.5}}{{9.5}}\\selectfont\\setlength{{\\tabcolsep}}{{{column_padding_pt:g}pt}}\n'
        '\\renewcommand{\\arraystretch}{1.15}\n'
        f'\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n'
        + '\\toprule\n' + ' & '.join(headers) + r' \\' + '\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\par\\medskip\n'
        + '\\noindent\\parbox{\\linewidth}{' + note + '}\\par\\endgroup\n'
        '\\end{table}\n\\FloatBarrier\n\n')


def tables(quality, report, rf):
    profiles = {(r['case_id'], m): r for r in report['profile_comparisons'] for m in r['methods']}
    out, rows = [], []
    for entry in quality['parameters']:
        selected = [x for x in entry['fits'] if x['selected']]
        if len(selected) != 1:
            raise ValueError('Exactly one published formula must be selected')
        fit = selected[0]
        p = profiles[entry['case_id'], entry['method']]
        tag = 'T' if entry['method'] == 'time_repaired' else 'L'
        row = [esc(entry['drug']) + f' ({tag})', 'S' if fit['formula'] == 'PKSimStandard' else 'R']
        for i, axis in enumerate(AXES):
            star = r'$^{*}$' if fit['near_boundary_1pct'][i] else ''
            val = number(fit['best']['parameters'][axis]) + star
            lo, hi = number(p['parameter_min_five'][i]), number(p['parameter_max_five'][i])
            row.append(r'\shortstack{' + val + r'\\{[' + lo + ',' + hi + ']}}')
        row += [number(fit['final_total_clearance_l_h']), str(p['members_five'])]
        rows.append(' & '.join(row) + r' \\')
    out.append(longtable(
        'Published five-point parameter multipliers and finite near-equivalent profile ranges. All thirty time-rule fits and four distinct learned-point fits are retained.',
        'tab:blueprint_reviewer_parameters',
        # Scientific-notation ranges need wider cells at the 5.5-inch text width.
        # Recover that width from padding rather than reducing the 7.5-point font.
        '@{}p{.18\\linewidth}c*{4}{>{\\centering\\arraybackslash}p{.152\\linewidth}}cc@{}',
        ['Drug / points', 'Form.', '$s_{\\rm CL}$', '$s_K$', '$s_{\\rm ex}$', '$s_{\\rm RBC}$', r'\shortstack{CL\\L/h}', '$m$'], rows,
        r'T: time coverage; L: a different learned point (the other 26 reuse T). S/R: PK-Sim Standard/Rodgers--Rowland. Upper number: published penalized optimum; brackets: data-compatible finite set across both formulas, using MSE$_{\min}+[\ln(1.1)]^2$. These are not confidence intervals. *: within 1\% of a log-boundary. CL is final systemic clearance under development human conditions; protected renal input is retained. $m$: retained profile members. Edaravone and Digoxin exceed the data-MSE tolerance; finite profile nodes do not enumerate every admissible fit, including the published optimum. Montelukast also changes the RBC bound and retains only one node: its zero sampled width does not establish certainty or a pure sampling effect.',
        column_padding_pt=2.1, continued_caption=True))
    rows = []
    for entry in quality['parameters']:
        p = profiles[entry['case_id'], entry['method']]
        tag = 'T' if entry['method'] == 'time_repaired' else 'L'
        row = [esc(entry['drug']) + f' ({tag})', str(p['members_five'])]
        for ep in ('auc', 'cmax', 'c_last', 'terminal_half_life'):
            v = p['five_matched_endpoint_ranges'].get(ep, {})
            row.append(number(v.get('max_over_min')))
        for ep in ('auc_inf', 'vss'):
            v = p['five_dense_endpoint_ranges'].get(ep, {})
            row.append(r'\shortstack{' + number(v.get('max_over_min')) + r'\\(' +
                str(v.get('estimable_members', 0)) + '/' + str(p['members_five']) + ')}')
        row += [str(sum(v['converged_nodes'] for v in p['five_profile_status'].values())) + '/' +
                str(sum(v['nodes'] for v in p['five_profile_status'].values()))]
        rows.append(' & '.join(row) + r' \\')
    out.append(longtable(
        'Prediction variability across near-equivalent five-point fits: maximum/minimum ratio, not prediction error or a calibrated interval.',
        'tab:blueprint_reviewer_prediction_ranges', '@{}p{.24\\linewidth}*{8}{c}@{}',
        ['Drug / points', '$m$', 'AUC', '$C_{\\max}$', '$C_{\\rm last}$', '$t_{1/2}$', 'AUC$_\\infty$', '$V_{ss}$', r'\shortstack{Profile\\nodes}'], rows,
        r'AUC, Cmax, Clast and half-life use target sampling times. AUC$_\infty$ and Vss use extended model trajectories without clinical reference; parentheses give estimable/all members. Full ranges and availability for all endpoints remain in the result JSON. Profile nodes gives converged/attempted nodes. Finite profiling is incomplete when nodes fail; no bound is presented as proof of biological identifiability. A narrow range within these retained fits does not exclude other compatible parameter combinations.',
        continued_caption=True))
    rows = []
    for entry in quality['cases']:
        tag = 'T' if entry['method'] == 'time_repaired' else 'L'
        pred, obs = entry['matched']['selected_tail'], entry['observed']['selected_tail']
        vals = [esc(entry['drug']) + f' ({tag})']
        for side in (pred, obs):
            vals.append(r'\shortstack{' + number(side.get('start_h')) + '--' + number(side.get('end_h')) + r'\\$n=' + str(side.get('n', 0)) + '$}')
        vals.extend([number(pred.get('adjusted_r_squared'), 4), number(obs.get('adjusted_r_squared'), 4)])
        vals.append(('Y' if entry['strict_matched']['terminal_half_life_qualified'] else 'N') + '/' +
                    ('Y' if entry['strict_observed']['terminal_half_life_qualified'] else 'N'))
        rows.append(' & '.join(vals) + r' \\')
    out.append(longtable(
        'Actual sampled terminal-regression windows and additional strict quality audit. Historical estimates remain unchanged.',
        'tab:blueprint_reviewer_tail_quality', '@{}p{.26\\linewidth}*{2}{>{\\centering\\arraybackslash}p{.19\\linewidth}}ccc@{}',
        ['Drug / points', r'\shortstack{Prediction\\window (h)}', r'\shortstack{Reference\\window (h)}', r'$R^2_{\rm adj,P}$', r'$R^2_{\rm adj,O}$', 'Pass P/O'], rows,
        r'Strict sensitivity requires at least three sampled tail points, adjusted $R^2\geq0.98$, and a span of at least one fitted half-life on both sides. This additional rule does not replace the frozen descriptive NCA definition. O is NCA of a retained clinical curve, not the mean of individual half-lives. An unknown assay LLOQ remains unknown.',
        continued_caption=True))
    rows = []
    for batch, data in rf['batches'].items():
        label = 'Original 10' if batch == 'original_ten' else 'Added 20'
        for key, ep in [('human_CL_mL_min_kg', 'CL (used)'), ('human_fup', '$f_u$ (used)')]:
            m = data['metrics'][key]
            a, b = m['random_forest'], m['training_median']
            rows.append(' & '.join([label, ep, str(m['train_n']), str(a['n']), number(a['gmfe']),
                f"{100*a['within_2fold']:.1f}", number(b['gmfe']), f"{100*b['within_2fold']:.1f}"]) + r' \\')
    out.append(longtable(
        'Scaffold-isolated parameter test sets: frozen RF versus a training-only median. Only CL and unbound fraction enter this blueprint.',
        'tab:blueprint_reviewer_rf', '@{}llrrrrrr@{}',
        ['RF batch', 'Endpoint', 'Train', 'Test', r'\shortstack{RF\\GMFE}', r'\shortstack{RF\\2-fold \%}', r'\shortstack{Median\\GMFE}', r'\shortstack{Median\\2-fold \%}'], rows,
        r'Only CL and unbound fraction, the RF-derived construction inputs used by this blueprint, are shown. Each frozen split holds out 100 structures; endpoint counts depend on available labels. All parameters are evaluated in their recorded units, with no clinical-curve refitting in this table. Tests across batches are not asserted to be independent or pooled as 200 independent compounds. RF is a gap filler, not a measurement or a calibrated uncertainty distribution. The separate post-calibration ablation tests whether the parameter predictor still changes target predictions.'))
    rf_result = report['rf_ablation']
    base = rf_result['summary']['median_cl']
    rows = [' & '.join(['RF gap inputs', '0'] + [f"{base[e]['all_drugs']['rf_input_gmfe']:.6f}" for e in ('curve','cmax','auc')] + ['Reference']) + r' \\']
    for method, name in [('median_cl','CL median'), ('median_fu',r'$f_u$ median'), ('median_both',r'CL + $f_u$ medians')]:
        s = rf_result['summary'][method]
        ci = rf_result['paired_ratios'][method]['all_drugs']['curve']['ci95']
        rows.append(' & '.join([name, str(s['curve']['input_actually_changed']['n'])] +
            [f"{s[e]['all_drugs']['median_input_gmfe']:.6f}" for e in ('curve','cmax','auc')] +
            [f"{s['curve']['all_drugs']['error_ratio_median_over_rf']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]"]) + r' \\')
    out.append(compact_table(
        'Does RF gap filling retain predictive benefit after the same five-point calibration? All thirty cases are retained.',
        'tab:blueprint_reviewer_rf_ablation', '@{}p{.20\\linewidth}rccc>{\\centering\\arraybackslash}p{.28\\linewidth}@{}',
        ['Gap input', 'Changed $n$', 'Curve', '$C_{\\max}$', 'AUC', r'\shortstack{Curve ratio to RF\\{[95\% CI]}}'], rows,
        r'The three error columns are drug-equal GMFE. Replacement uses the frozen training-only median for RF-derived inputs; independent qualified evidence stays unchanged. Eighty native exports and readbacks, 64 new fits and 10 unchanged-input reuses yield 90 logical comparison units. Each fit sees identical time-rule five points, equations, absolute clearance bounds, feasibility-repair rules and fitting budget; multiplier bounds may change with construction inputs. Tiny floating-point changes are not clinical wins. CIs use 2,000 study-cluster draws. Changed-input-only statistics, all endpoint values and failure accounting are retained in the machine-readable report.'))
    TABLE.write_text(''.join(out))


def plot(quality, report, curves, export, *, caption_notes=False, output_base=None,
         legend_columns=2):
    import sys
    skill = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')
    sys.path.insert(0, str(skill))
    from setup_style import setup_style
    from profile_data import profile_data
    from visual_qa import audit_layout, render_preview
    import matplotlib
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    destination = Path(output_base) if output_base is not None else FIG
    selected = [('Remdesivir', 'time_repaired'), ('Sufentanil', 'time_repaired')]
    profile_data(pd.DataFrame([dict(drug=drug, time_h=t, prediction=y) for drug, method in selected
        for t, y in zip(curves['cases'][drug+'__'+method]['time_h'], curves['cases'][drug+'__'+method]['prediction_ng_ml'])]), group_cols=['drug'])
    setup_style(journal='general', lang='en', use_sciplots=False)
    for font in (Path(matplotlib.get_data_path()) / 'fonts/ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.titlesize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.constrained_layout.use': False, 'axes.linewidth': .6})
    fig, ax = plt.subplots(2, 2, figsize=(5.5, 4.7))
    fig.subplots_adjust(left=.115, right=.98, top=.865, bottom=.12, wspace=.37, hspace=.50)
    for i, (drug, method) in enumerate(selected):
        d = curves['cases'][drug+'__'+method]
        entry = next(x for x in quality['cases'] if x['drug'] == drug and x['method'] == method)
        t, y = np.array(d['time_h']), np.array(d['prediction_ng_ml'])
        ot, oy = np.array(d['observed_time_h']), np.array(d['observed_concentration_ng_ml'])
        a, b = ax[i]
        profile = next(x for x in report['profile_comparisons'] if x['case_id'] == entry['case_id'] and method in x['methods'])
        er = profile['target_curve_envelope']['artifact']
        ep = ROOT / er['path']
        if hashlib.sha256(ep.read_bytes()).hexdigest() != er['sha256']:
            raise ValueError('Changed profile-envelope artifact')
        env = np.load(ep)
        et = env['time_h']; lo = env['minimum_mg_l']*1000; hi = env['maximum_mg_l']*1000
        keep = (t >= 0) & (t <= ot[-1]*1.03) & (y > 0)
        clinical = (et >= 0) & (et <= ot[-1]*1.03) & (lo > 0)
        a.fill_between(et[clinical], lo[clinical], hi[clinical], color='#0072B2', alpha=.16, lw=0)
        a.plot(t[keep], y[keep], color='#0072B2', lw=1.2)
        a.scatter(ot, oy, facecolors='none', edgecolors='#E66100', marker='s', s=17, lw=.8, zorder=4)
        for side, color in [('matched_tail', '#0072B2'), ('observed_tail', '#E66100')]:
            f = d[side]
            x = np.linspace(f['start_h'], f['end_h'], 80)
            a.plot(x, np.exp(f['intercept_ln_ng_ml']-f['lambda_z_per_h']*x), color=color, ls='--', lw=1.0)
        a.set_yscale('log'); a.set_xlabel('Clinical window (h)')
        a.set_xlim(0, ot[-1]*1.06)
        a.set_ylim(min(oy[oy > 0].min(), y[keep][-1])*.5, max(oy.max(), y[keep].max())*1.8)
        a.set_title(chr(65+i*2)+'  '+drug, loc='left', fontweight='bold', pad=6)
        a.set_ylabel('Concentration (ng/mL)')
        show = (t >= ot[-1]) & (y > 0)
        extension = (et >= ot[-1]) & (lo > 0)
        b.fill_between(et[extension], lo[extension], hi[extension], color='#0072B2', alpha=.16, lw=0)
        b.plot(t[show], y[show], color='#0072B2', lw=1.2)
        windows = next(x for x in report['terminal_regression_windows'] if x['case_id'] == entry['case_id'] and x['method'] == method)
        w = windows['model_dense_actual_sample_window_h']
        b.axvspan(w[0], w[1], color='#0072B2', alpha=.13)
        b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(ot[-1], t[-1]*1.06)
        lloq = d['clinical_lloq_ng_ml']
        if lloq and lloq > 0:
            b.axhline(lloq, color='#E66100', ls=':', lw=1)
        if not caption_notes:
            if lloq and lloq > 0:
                b.text(.03, .91, f'Assay LLOQ {lloq:g} ng/mL', transform=b.transAxes, color='#B65300', fontsize=6.5,
                    bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
            else:
                b.text(.03, .91, 'Assay LLOQ: not verified', transform=b.transAxes, color='#596A73', fontsize=6.5,
                    bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
            h = np.log(2)/entry['dense']['terminal_fit']['lambda_z_per_h']
            b.text(.04, .055, f'Model tail: {h:.1f} h\nFit window {w[0]:.2f}–{w[1]:.0f} h',
                transform=b.transAxes, fontsize=6.5, va='bottom', bbox=dict(fc='white', ec='none', alpha=.85,pad=1))
        b.set_xlabel('Model extension (h; log scale)')
        b.set_title(chr(66+i*2)+'  Unobserved long tail', loc='left', fontweight='bold', pad=6)
        for panel in (a,b):
            panel.grid(axis='y', color='#E6EBEE', lw=.4)
            panel.tick_params(length=2, width=.5)
    handles = [Line2D([],[],color='#0072B2',label='Locked prediction'),
        Line2D([],[],marker='s',color='#E66100',mfc='none',ls='',label='Target observations',markersize=4),
        Line2D([],[],color='#596A73',ls='--',label='Clinical-window tail fits')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52,.992), ncol=legend_columns,
        frameon=False,fontsize=7,handlelength=2,columnspacing=1.3)
    render_preview(fig,str(destination.with_suffix('.png')),dpi=300)
    issues = audit_layout(fig)
    if issues:
        raise ValueError('Layout audit: '+str(issues))
    if export:
        fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', action='store_true')
    a = p.parse_args()
    quality, report, rf, curves = [read(DIAG/n) for n in ('nca_quality.json','report.json','rf_heldout.json','tail_case_curves.json')]
    tables(quality,report,rf)
    qa = plot(quality,report,curves,a.export)
    manifest = dict(version='blueprint-reviewer-publication-2-time-coverage',
        frozen_prediction_values_changed=False,
        tail_example_method='time_repaired',
        retained_parameter_rows={'time_repaired':30, 'distinct_stable_repaired':4},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={str((DIAG/n).relative_to(ROOT)):hashlib.sha256((DIAG/n).read_bytes()).hexdigest()
            for n in ('nca_quality.json','report.json','rf_heldout.json','tail_case_curves.json')}, layout_issues=qa)
    (RUN/'publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published reviewer tables and tail figure; no model fit or source edits.')


if __name__=='__main__':
    main()
