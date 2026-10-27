#!/usr/bin/env python3
"""Publish selected and complete locked curves and clinical endpoints; never fit.

Use --export only after reviewing the PNG. The experiment and its
SHA-bound source artifacts remain read-only. P* and O come from the frozen
common-curve endpoint CSV; dense P is a model-only summary of the same curve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill')
sys.path.insert(0, str(SKILL / 'scripts'))
import numpy as np
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from blueprint_sampling_expansion_data import _clinical, _curve_view, _curve_ng_ml
from summarize_additional_clinical_baselines import prediction_in_hours
from render_current_clinical_table import native_value

RUN = ROOT / 'experiments/blueprint_design/sampling_stability_20261009'
FIG = ROOT / 'results/figures/experiments/blueprint_sampling_curves'
ALL_FIG = ROOT / 'results/figures/experiments/blueprint_sampling_curves_all'
TABLE = ROOT / 'results/tables/blueprint_sampling_clinical.tex'
MANIFEST = ROOT / 'results/tables/blueprint_sampling_clinical_sources.json'
EPS = ('auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life')
LABELS = (r'AUC (window)', r'$C_{\max}$', r'$T_{\max}$',
          r'$C_{\mathrm{last}}$', 'Terminal half-life')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(ref):
    path = (ROOT / ref['path']).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != ref['sha256']:
        raise ValueError('Source/hash mismatch: ' + str(path))
    return read(path)


def same(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f'Independent recomputation differs: {a} versus {b}')


def load():
    state = read(RUN / 'state.json')
    if state['status'] != 'complete':
        raise ValueError('Only completed experiments can be published')
    protocol, result = checked(state['protocol']), checked(state['results'])
    for path, expected in protocol['source_pins'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    choices = checked(protocol['choices'])['choices']
    if digest(ROOT / result['endpoints']['path']) != result['endpoints']['sha256']:
        raise ValueError('Endpoint CSV changed')
    with (ROOT / result['endpoints']['path']).open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['regime'] == 'late_four'
                and r['requested_goal'] == 'cmax'
                and r['method'] in ('time_repaired', 'no_addition_repaired')]
    by = {(r['case_id'], r['method'], r['endpoint']): r for r in rows}
    cases, source_list = [], []
    quality_path = ROOT / 'experiments/blueprint_design/reviewer_revision_20261009/diagnostics/nca_quality.json'
    quality = read(quality_path)
    quality_by = {(x['case_id'], x['method']): x for x in quality['cases']}
    for config in protocol['cases']:
        cid = config['case_id']
        ch = choices[cid + '__late_four']['cmax']['sources']
        curves, units, requests = {}, {}, {}
        sources = {}
        for method in ('time_repaired', 'no_addition_repaired'):
            unitref = state['units'][ch[method]['unit_id']]['result']
            unit = checked(unitref)
            if unit['status'] != 'predicted':
                raise ValueError('Do not silently discard failed predictions')
            units[method] = unit
            curves[method] = checked(unit['prediction'])
            requests[method] = checked(unit['request'])
            sources[method] = {'result': unitref, 'prediction': unit['prediction'],
                               'request': unit['request']}
        req = requests['time_repaired']
        if req['dosing'] != requests['no_addition_repaired']['dosing']:
            raise ValueError('Prediction conditions differ')
        contract = checked(config['target_contract'])
        mw = contract['molecular_weight_g_mol']
        raw = _clinical(checked(config['reference_source']), cid)
        obs_t, obs_y = _curve_ng_ml(_curve_view(raw), mw)
        if obs_t != contract['sampling_time_h']:
            raise ValueError('Target sampling contract differs')
        processed, metrics = {}, {}
        for method, native in curves.items():
            t, y, _ = prediction_in_hours(native, mw)
            processed[method] = (np.asarray(t), np.asarray(y))
            pred = np.interp(obs_t, t, y)
            valid = np.array(obs_y) > 0
            logs = np.log(pred[valid] / np.array(obs_y)[valid])
            gmfe = float(np.exp(np.mean(np.abs(logs))))
            same(gmfe, by[cid, method, 'curve']['error'])
            metrics[method] = dict(gmfe=gmfe, ln_rmse=float(np.sqrt(np.mean(logs**2))),
                                   n=int(valid.sum()))
        # Remove only exact duplicate timestamps introduced by float conversion.
        # Values must agree; do not smooth or resimulate the frozen trajectory.
        t, y = processed['time_repaired']
        clean_t, clean_c = [], []
        for tm, c in zip(t * 60, y / mw):
            if clean_t and tm == clean_t[-1]:
                same(c, clean_c[-1])
                continue
            clean_t.append(float(tm)); clean_c.append(float(c))
        clean = dict(time_min=clean_t, concentration=clean_c, time_unit='min', unit='umol/l')
        dense = summarize_extended_prediction(clean, req)
        endpoint_rows = []
        for endpoint in EPS:
            r = by[cid, 'time_repaired', endpoint]
            if not r['reference']:
                raise ValueError('This observed endpoint unexpectedly has no reference')
            pv, ov, er = float(r['prediction']), float(r['reference']), float(r['error'])
            same(abs(pv - ov) if endpoint == 'tmax' else max(pv / ov, ov / pv), er)
            v, _, reason = native_value(dense['endpoints'][endpoint], endpoint, r['unit'], req['human']['weight_kg'])
            endpoint_rows.append(dict(endpoint=endpoint, unit=r['unit'], dense=v,
                prediction=pv, reference=ov, error=er, weak_tail=r['weak_tail'] == 'True',
                dense_unavailable_reason=reason))
        # Clinical-reference weak-tail flags are recorded in the frozen score.
        unit = units['time_repaired']
        repaired = result['repaired_scores'].get(ch['time_repaired']['unit_id'])
        if repaired is None:
            # All other units are already scored in the parent expansion.
            if 'parent_records' not in locals():
                parent_records = read(ROOT / 'experiments/blueprint_design/sampling_expansion_20261009/results.json')['records']
                parent_index = {r['source']['path']: r for r in parent_records}
            score = parent_index[sources['time_repaired']['result']['path']]
        else:
            score = repaired
        for r in endpoint_rows:
            r['reference_weak_tail'] = score['endpoints'][r['endpoint']]['reference_weak_tail']
        # Add separate reviewer quality annotations; do not alter frozen values.
        qr = quality_by.get((cid, 'time_repaired'))
        if qr is None:
            raise ValueError('Missing additional tail-quality audit for '+cid)
        if qr['source']['sha256'] != sources['time_repaired']['result']['sha256']:
            raise ValueError('Tail-quality audit belongs to a different model for '+cid)
        for r in endpoint_rows:
            # The separate frozen NCA audit provides a second scalar source,
            # including dense P. Never relabel the learned selector's values.
            audited = qr['original_endpoints'][r['endpoint']]
            for field, other in (('prediction', 'value'), ('reference', 'reference'),
                                 ('error', 'error')):
                same(r[field], audited[other])
            audited_dense, _, _ = native_value(qr['dense_endpoints'][r['endpoint']],
                r['endpoint'], r['unit'], req['human']['weight_kg'])
            if r['dense'] is None or audited_dense is None:
                if r['dense'] is not None or audited_dense is not None:
                    raise ValueError('Dense endpoint availability differs from the frozen audit')
            else:
                same(r['dense'], audited_dense)
            r['strict_prediction_failed'] = r['endpoint'] == 'terminal_half_life' and not qr['strict_matched']['terminal_half_life_qualified']
            r['strict_reference_failed'] = r['endpoint'] == 'terminal_half_life' and not qr['strict_observed']['terminal_half_life_qualified']
        cases.append(dict(config=config, request=req, observed=(np.array(obs_t), np.array(obs_y)),
            processed=processed, metrics=metrics, endpoints=endpoint_rows,
            selected_time_h=ch['time_repaired']['time_h'], dense=dense,
            formula=unit.get('formula', 'recorded'),
            matrix=contract['clinical_matrix']))
        source_list.append(dict(case_id=cid, drug=config['drug'], study_id=config['study_id'],
            target_reference=config['reference_source'], target_contract=config['target_contract'],
            **sources, endpoints=endpoint_rows, target_metrics=metrics,
            selected_development_time_h=ch['time_repaired']['time_h'],
            selected_candidate=ch['time_repaired']['candidate'],
            quality_method='time_repaired',
            source_caveats=config.get('sensitivity_flags', [])))
    if len(cases) != 30 or len({c['config']['drug'] for c in cases}) != 30:
        raise ValueError('Require all thirty unique drugs')
    return cases, dict(version='blueprint-sampling-clinical-publication-5-nine-compact-curves',
        experiment=state['results'], csv=result['endpoints'], cases=source_list,
        prediction_lock=result['prediction_lock'], scored_at=result['scored_at'],
        publication_parameters_fitted=False, source_pins_checked=len(protocol['source_pins']),
        scalar_verification={'rows': 150, 'fields': ['dense P', 'matched P*', 'reference O', 'error'],
            'sources': 'Locked endpoint CSV and independent frozen reviewer NCA audit',
            'relative_tolerance': 1e-9, 'absolute_tolerance': 1e-12},
        scalar_rows=150, clinical_curve_points=sum(c['metrics']['time_repaired']['n'] for c in cases),
        reference_semantics='NCA of original observed individual or aggregate curves, as identified in source audit; not original reported scalar PK',
        dense_policy='Existing deterministic-model-prefix-1; finite endpoints on declared request window',
        additional_quality_source={'path': str(quality_path.relative_to(ROOT)), 'sha256': digest(quality_path)},
        frozen_prediction_and_reference_values_changed=False,
        shown_regime='late_four', requested_goal='cmax', method='time_repaired',
        before_method='no_addition_repaired',
        selection='Frozen deterministic time-coverage candidate, not the learned stable selector')


def fmt(value):
    return '\\textemdash{}' if value is None else f'\\({value:.4g}\\)'


def esc(s):
    return str(s).replace('\\', '/').replace('&', r'\&').replace('_', r'\_').replace('%', r'\%').replace('#', r'\#')


def table(cases):
    h = r'''\begingroup\captionsetup[table]{name=Table}
\definecolor{BPSInk}{HTML}{233A4B}\definecolor{BPSFaint}{HTML}{EAF0F3}\definecolor{BPSStripe}{HTML}{F6F9FA}
\setlength{\tabcolsep}{3pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}
\setlength{\LTcapwidth}{\linewidth}\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}
\renewcommand{\arraystretch}{1.10}\fontsize{8.5}{11}\selectfont
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.19\linewidth}>{\centering\arraybackslash}p{.085\linewidth}*{3}{>{\centering\arraybackslash}p{.127\linewidth}}>{\centering\arraybackslash}p{.10\linewidth}>{\centering\arraybackslash}p{.13\linewidth}@{}}
\caption{Locked blueprint predictions under target-study conditions: 150 observed-curve scalar comparisons across 30 drugs. Calibration uses the shared five-observation set; its construction from four observations and one time-coverage observation belongs to the additional sampling analysis.}\label{tab:blueprint_clinical_metrics}\\
'''
    head = r'''\toprule
\rowcolor{BPSStripe}\textbf{Endpoint} & \textbf{Unit} & \shortstack{Dense model\\P} & \shortstack{Matched\\P$^*$} & \shortstack{Observed-curve\\reference O} & \shortstack{Fold error\\FE} & \shortstack{Absolute error\\h ($T_{\max}$)} \\
\midrule
'''
    notes = r'''\par\noindent\begin{minipage}{\linewidth}\begingroup\fontsize{7.5}{9.5}\selectfont
\setlength{\parindent}{0pt}\setlength{\parskip}{2pt}
\textbf{Notes to Table~\ref{tab:blueprint_clinical_metrics}.}\par
P: model-only dense-grid endpoints on the stated request window (half-life uses the fixed model-tail rule). P$^*$: same locked curve sampled at target times. O: NCA of that study's individual or aggregate concentration curve (see source audit), not a source-reported scalar or mean individual NCA. Only P$^*$ is scored.\par
FE=$\max(P^*/O,O/P^*)$; Tmax uses absolute hours. $\dagger/\ddagger$: historical predicted/reference weak-tail flags. $\S$: fails the added strict tail-quality rule (at least 3 points, adjusted $R^2\geq0.98$, span at least one half-life). Historical values are retained. Missing dense P is ---; never filled from P$^*$. AUC$_\infty$, CL, Vz, MRT and Vss lack common observed time-zero references.\par
Dense half-life P is a model-only long-tail estimate, which can use simulation beyond the clinical window; it is not a measured terminal half-life. Actual regression windows and quality are reported in the reviewer-diagnostics appendix.\par
\endgroup\end{minipage}\par
'''
    continued_foot = r'''\midrule
\multicolumn{7}{@{}p{\linewidth}@{}}{\fontsize{7.5}{9.5}\selectfont Continued on the next page; endpoint definitions and quality-flag explanations follow the table.}\\
\bottomrule
'''
    out = [h, head, '\\endfirsthead\n',
        r'\multicolumn{7}{@{}l@{}}{\bfseries Table~\thetable\quad Thirty-drug target predictions (continued)}\\'+'\n',
        head, '\\endhead\n', continued_foot, '\\endfoot\n', '\\bottomrule\n', '\\endlastfoot\n']
    for i, c in enumerate(cases):
        conf, req, met = c['config'], c['request'], c['metrics']['time_repaired']
        d, p = req['dosing'], req['human']
        obs_t = c['observed'][0]
        study_label = 'Heinig 2018; Bayer 16535 (OSP F B)' if conf['drug'] == 'Finerenone' else conf['study_id']
        title = f"{i+1:02d}. {esc(conf['drug'])} / {esc(study_label)}"
        out.append(r'\rowcolor{BPSInk}\multicolumn{7}{@{}p{\linewidth}@{}}{\strut\hspace{1mm}\color{white}\bfseries '+title+r'} \\*'+'\n')
        info = (f"Target dose {d['dose_mg']:.4g} mg; infusion {d['infusion_minutes']:.4g} min; "
                f"{p['weight_kg']:.4g} kg. Dense P: {req['target']['window_start_h']:.4g}--{req['target']['window_end_h']:.4g} h; "
                f"matched: {obs_t[0]:.4g}--{obs_t[-1]:.4g} h. "
                f"Sampling-analysis added time {c['selected_time_h']:.4g} h. "
                f"Curve GMFE {met['gmfe']:.3f}; ln-RMSE {met['ln_rmse']:.3f}; n_points={met['n']}.")
        out.append(r'\rowcolor{BPSFaint}\multicolumn{7}{@{}p{\linewidth}@{}}{\fontsize{8}{10}\selectfont '+esc(info)+r'} \\*'+'\n')
        for j, (r, label) in enumerate(zip(c['endpoints'], LABELS)):
            unit = r'ng$\cdot$h/mL' if r['unit'] == 'ng.h/mL' else r['unit']
            pred = fmt(r['prediction']) + (r'$^{\dagger}$' if r['weak_tail'] else '')
            observed = fmt(r['reference']) + (r'$^{\ddagger}$' if r['reference_weak_tail'] else '')
            err = fmt(r['error']) + (r'$^{\dagger}$' if r['weak_tail'] else '') + (r'$^{\ddagger}$' if r['reference_weak_tail'] else '')
            if r['strict_prediction_failed']: pred += r'$^{\S}$'
            if r['strict_reference_failed']: observed += r'$^{\S}$'
            if r['strict_prediction_failed'] or r['strict_reference_failed']: err += r'$^{\S}$'
            vals = [label, unit, fmt(r['dense']), pred, observed,
                    r'\textemdash{}' if r['endpoint'] == 'tmax' else err,
                    err if r['endpoint'] == 'tmax' else r'\textemdash{}']
            out.append((r'\rowcolor{BPSStripe}' if j % 2 == 0 else '') + ' & '.join(vals) + (r' \\*' if j < 4 else r' \\') + '\n')
        caveat = []
        if not conf.get('cross_study_pair', True): caveat.append('Same-study dose-arm transfer, not cross-study validation.')
        if c['matrix'] != 'plasma': caveat.append('Clinical '+c['matrix']+' is compared with predicted plasma under the declared approximation.')
        if conf['drug'] == 'Midazolam': caveat.append('Target source is a public patent.')
        if conf['drug'] == 'Nemonoxacin': caveat.append('Participant overlap between sources is unresolved.')
        if conf['drug'] == 'Dexmedetomidine': caveat.append('Target observation is a surgical participant; population confounding remains.')
        if conf['drug'] == 'Edaravone': caveat.append('Retained digitized points start at 2 h, after infusion. Earlier samples exist in the source; this sampled peak is not the clinical peak.')
        if conf['drug'] == 'Levonorgestrel': caveat.append('Raw Bayer dataset A229 identifies serum; the frozen plasma label is retained only as an audited source discrepancy.')
        if caveat:
            out.append(r'\multicolumn{7}{@{}p{\linewidth}@{}}{\fontsize{7.5}{9.5}\selectfont '+esc(' '.join(caveat))+r'} \\'+'\n')
        out.append(r'\addlinespace[4pt]'+'\n')
    out.extend(['\\end{longtable}\n', notes, '\\endgroup\n'])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(''.join(out))


def select_illustrative_cases(cases):
    """Choose an explicit effect-spanning display, never a performance estimate.

    Select order statistics nearest 0, 25, 50, 75 and 100 percent of the
    after/before curve-GMFE ratio, then the remaining case nearest ratio one.
    Retain these six cases, adding the 12.5, 37.5 and 87.5 percent order
    statistics to show nine cases. Ties are resolved by drug name,
    independently of visual appearance.
    """
    ordered = sorted(cases, key=lambda c: (
        c['metrics']['time_repaired']['gmfe'] /
        c['metrics']['no_addition_repaired']['gmfe'], c['config']['drug']))
    indices = [int(np.rint(q * (len(ordered) - 1))) for q in (0, .25, .5, .75, 1)]
    neutral = min((i for i in range(len(ordered)) if i not in indices),
                  key=lambda i: (abs(np.log(ordered[i]['metrics']['time_repaired']['gmfe'] /
                                           ordered[i]['metrics']['no_addition_repaired']['gmfe'])),
                                 ordered[i]['config']['drug']))
    original_indices = sorted(indices + [neutral])
    added_indices = [int(np.rint(q * (len(ordered) - 1))) for q in (.125, .375, .875)]
    indices = sorted(set(original_indices + added_indices))
    if len(indices) != 9:
        raise ValueError('The fixed thirty-drug display must contain nine unique cases')
    selected = [ordered[i] for i in indices]
    return selected, dict(
        purpose='Outcome-spanning illustration only; not a representative sample or overall estimate',
        rule='Nearest order statistics at 0%, 25%, 50%, 75%, 100% of after/before curve-GMFE; '
             'plus remaining case nearest ratio 1 on the log scale; retain these six and add '
             '12.5%, 37.5%, 87.5% order statistics; ties by drug name',
        retained_original_drugs=[ordered[i]['config']['drug'] for i in original_indices],
        added_drugs=[ordered[i]['config']['drug'] for i in added_indices],
        sorted_rank_one_based=[i + 1 for i in indices],
        drugs=[c['config']['drug'] for c in selected],
        case_ids=[c['config']['case_id'] for c in selected],
        ratios=[c['metrics']['time_repaired']['gmfe'] /
                c['metrics']['no_addition_repaired']['gmfe'] for c in selected],
        observed_points=sum(len(c['observed'][0]) for c in selected),
        cohort_drugs=len(cases),
        complete_curves=str(ALL_FIG.with_suffix('.pdf').relative_to(ROOT)))


def plot(cases, export=False, *, main_only=False, spacious=False, output_base=None):
    from setup_style import setup_style
    from visual_qa import audit_layout, render_preview
    from profile_data import profile_data
    import pandas as pd
    import matplotlib
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter, NullLocator
    if output_base is not None and not main_only:
        raise ValueError('A separate output base requires main_only=True')
    main_figure = FIG if output_base is None else Path(output_base)
    # EDA: units/groups/missingness/range precede chart selection.
    data = pd.DataFrame([dict(drug=c['config']['drug'], time_h=float(t), observed_ng_ml=float(y))
                         for c in cases for t, y in zip(*c['observed'])])
    profile_data(data, group_cols=['drug'])
    if data.isna().any().any() or (data['observed_ng_ml'] <= 0).any():
        raise ValueError('Do not hide missing or nonpositive observations on log axes')
    print('Observed curves:', len(data), 'points;', data['drug'].nunique(), 'drugs;',
          'range', data['observed_ng_ml'].min(), data['observed_ng_ml'].max())
    selected, selection = select_illustrative_cases(cases)
    print('Illustrative cases:', json.dumps(selection))
    # A compact 3 x 3 main figure shows outcome heterogeneity. Complete curves
    # remain in protocol order on three readable 5 x 2 appendix pages.
    setup_style(journal='general', lang='en', use_sciplots=False)
    for font in (Path(matplotlib.get_data_path()) / 'fonts/ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                         'axes.titlesize': 8.5, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'figure.constrained_layout.use': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#929A9D',
                         'axes.linewidth': .6, 'xtick.color': '#39434A',
                         'ytick.color': '#39434A', 'text.color': '#202C34',
                         'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans',
                         'savefig.bbox': None, 'savefig.pad_inches': 0})
    blue, clay, ink = '#3B6B8E', '#B96B4D', '#36434A'
    handles = [Line2D([], [], color=blue, ls='-', lw=1.4, label='Limited data (4 obs.)'),
               Line2D([], [], color=clay, ls=(0, (3.5, 1.8)), lw=1.4, label='Augmented data (5 obs.)'),
               Line2D([], [], color=ink, marker='o', markerfacecolor='white', linestyle='',
                      markeredgewidth=.7, markersize=3.3, label='Observed target')]
    qa, figs = [], []

    def make_page(page_cases, rows, cols, size, *, full=False, page=1):
        fig, axes = plt.subplots(rows, cols, figsize=size)
        if full:
            fig.subplots_adjust(left=.13, right=.976, bottom=.065, top=.885,
                                hspace=.92, wspace=.34)
        elif spacious:
            fig.subplots_adjust(left=.115, right=.978, bottom=.11, top=.86,
                                hspace=.96, wspace=.43)
        else:
            fig.subplots_adjust(left=.115, right=.978, bottom=.115, top=.825,
                                hspace=.72, wspace=.43)
        panel_headings = []
        for local_i, (ax, c) in enumerate(zip(axes.flat, page_cases)):
            ot, oy = c['observed']; maximum = ot[-1]; minimum = ot[0]
            vals = list(oy)
            for method, color, ls in [
                    ('no_addition_repaired', blue, '-'),
                    ('time_repaired', clay, (0, (3.5, 1.8)))]:
                t, y = c['processed'][method]
                keep = (t >= minimum) & (t <= maximum) & (y > 0)
                vals.extend(y[keep])
                ax.plot(t[keep], y[keep], color=color, ls=ls, lw=1.3,
                        solid_capstyle='round', zorder=2)
            ax.scatter(ot, oy, s=11, marker='o', facecolor='white', edgecolor=ink,
                       linewidth=.65, zorder=4)
            ax.set_yscale('log')
            ax.set_xlim(minimum-.035*(maximum-minimum), maximum+.035*(maximum-minimum))
            lo, hi = min(vals), max(vals)
            # Padding is defined in log space so every curve and point is visible.
            pad = max(.10, .08 * np.log10(hi / lo))
            ax.set_ylim(lo / 10**pad, hi * 10**pad)
            ymin, ymax = ax.get_ylim()
            candidates = [m * 10.**e for e in range(int(np.floor(np.log10(ymin))),
                           int(np.ceil(np.log10(ymax)))+1) for m in (1, 2, 5)]
            decades = [10.**e for e in range(int(np.floor(np.log10(ymin))),
                       int(np.ceil(np.log10(ymax)))+1) if ymin <= 10.**e <= ymax]
            ticks = decades if len(decades) >= 2 else [v for v in candidates if ymin <= v <= ymax]
            if len(ticks) < 2:
                ticks = [v for v in MaxNLocator(nbins=3).tick_values(ymin, ymax)
                         if ymin <= v <= ymax and v > 0]
            if len(ticks) > 3:
                ticks = [ticks[i] for i in np.rint(np.linspace(0, len(ticks)-1, 3)).astype(int)]
            def scientific_tick(value, _):
                exponent = int(np.floor(np.log10(value)))
                mantissa = value / 10.**exponent
                return (rf'$10^{{{exponent}}}$' if np.isclose(mantissa, 1) else
                        rf'${mantissa:g}\!\times\!10^{{{exponent}}}$')
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(scientific_tick))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10],
                                                   min_n_ticks=2))
            ax.grid(axis='y', color='#E7EAEC', linewidth=.5, zorder=0)
            ax.tick_params(axis='both', length=2.4, width=.5, pad=2)
            a = c['metrics']['no_addition_repaired']['gmfe']
            b = c['metrics']['time_repaired']['gmfe']
            label = f'{(page-1)*10+local_i+1:02d}' if full else chr(97+local_i)
            if spacious and not full:
                # Point offsets preserve a consistent title/metric separation
                # independently of panel height or concentration range.
                panel_letter = ax.annotate(label, xy=(-.12, 1), xycoords='axes fraction',
                    xytext=(0, 13.5), textcoords='offset points', fontsize=8,
                    fontweight='bold', ha='left', va='bottom', color=ink)
                panel_title = ax.annotate(c['config']['drug'], xy=(.5, 1),
                    xycoords='axes fraction', xytext=(0, 13.5), textcoords='offset points',
                    fontsize=8, fontweight='bold', ha='center', va='bottom')
                panel_metric = ax.annotate(f'GMFE {a:.2f} → {b:.2f}', xy=(.5, 1),
                    xycoords='axes fraction', xytext=(0, 2.5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, color='#53616B')
                panel_headings.append((panel_letter, panel_title, panel_metric))
            else:
                ax.text(-.12, 1.25, label, transform=ax.transAxes, fontsize=8.5 if full else 8,
                        fontweight='bold', ha='left', va='bottom', color=ink)
                ax.text(.5, 1.25, c['config']['drug'], transform=ax.transAxes,
                        fontsize=8.4 if full else 8, fontweight='bold', ha='center', va='bottom')
                ax.text(.5, 1.035, f'GMFE {a:.2f} → {b:.2f}', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=7.2 if full else 7, color='#53616B')
        fig.supylabel('Concentration (ng/mL; log scale)', x=.015, fontsize=8)
        fig.supxlabel('Time after target dose (h)', y=.009, fontsize=8)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55,.998),
                   ncol=3, fontsize=7.4, frameon=False, handlelength=1.9,
                   columnspacing=1.45, handletextpad=.5)
        if full:
            fig.text(.55, .950, f'All 30 drugs · {1+(page-1)*10:02d}–{page*10:02d}',
                     ha='center', va='center', fontsize=8.5, fontweight='bold')
        path = ALL_FIG.with_name(ALL_FIG.name+f'_{page}.png') if full else main_figure.with_suffix('.png')
        render_preview(fig, str(path), dpi=300)
        # Keep the final PNG at the same declared size as its vector PDF.
        fig.savefig(path, dpi=300)
        issues = audit_layout(fig)
        row_gaps = []
        if spacious and not full:
            renderer = fig.canvas.get_renderer()
            for row in range(rows-1):
                above = [label.get_window_extent(renderer).y0
                    for ax in axes[row] for label in ax.get_xticklabels()
                    if label.get_visible() and label.get_text()]
                below = [label.get_window_extent(renderer).y1
                    for group in panel_headings[(row+1)*cols:(row+2)*cols]
                    for label in group]
                row_gaps.append((min(above)-max(below))*72/fig.dpi)
            if min(row_gaps) < 10:
                raise ValueError('Insufficient visible row gap: '+str(row_gaps))
        data_hash = hashlib.sha256()
        for ax in axes.flat:
            for line in ax.lines:
                data_hash.update(np.asarray(line.get_xydata(), dtype='<f8').tobytes())
            for collection in ax.collections:
                data_hash.update(np.asarray(collection.get_offsets(), dtype='<f8').tobytes())
        qa.append({'display': 'complete_appendix' if full else 'illustrative_main',
                   'page': page, 'rows': rows, 'columns': cols, 'drug_panels': len(page_cases),
                   'drug_names': [c['config']['drug'] for c in page_cases],
                   'observations': sum(len(c['observed'][0]) for c in page_cases),
                   'size_inches': list(size), 'minimum_font_pt': 7,
                   'plotted_data_sha256': data_hash.hexdigest(),
                   'visible_row_gaps_pt': row_gaps,
                   'issues': [str(x) for x in issues]})
        return fig

    main_figure.parent.mkdir(parents=True, exist_ok=True)
    figs.append(make_page(selected, 3, 3, (5.5, 4.3 if spacious else 3.55)))
    if not main_only:
        for page in range(3):
            figs.append(make_page(cases[page*10:(page+1)*10], 5, 2, (5.5, 6.85),
                                  full=True, page=page+1))
    print('Layout QA:', json.dumps(qa))
    if any(r['issues'] for r in qa):
        for fig in figs:
            plt.close(fig)
        raise ValueError('Visual layout audit failed; inspect previews')
    if not main_only and sum(r['observations'] for r in qa if r['display'] == 'complete_appendix') != len(data):
        raise ValueError('Complete appendix lost observations')
    if export:
        figs[0].savefig(main_figure.with_suffix('.pdf'))
        figs[0].savefig(main_figure.with_suffix('.svg'))
        if not main_only:
            with PdfPages(ALL_FIG.with_suffix('.pdf')) as pdf:
                for fig in figs[1:]:
                    pdf.savefig(fig)
    for fig in figs:
        plt.close(fig)
    return qa, selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--main-only', action='store_true',
                        help='Update only the main curve figure; preserve appendix assets and table')
    parser.add_argument('--spacious', action='store_true',
                        help='Use wider vertical spacing for the nine-panel English figure')
    parser.add_argument('--output-base', type=Path,
                        help='Separate figure path without suffix; requires --main-only and leaves shared assets unchanged')
    args=parser.parse_args()
    if args.output_base is not None and not args.main_only:
        parser.error('--output-base requires --main-only')
    cases, manifest=load()
    if not args.main_only:
        table(cases)
    qa, selection = plot(cases, args.export, main_only=args.main_only,
                         spacious=args.spacious, output_base=args.output_base)
    if args.output_base is not None:
        report = dict(layout_audit=qa, main_figure_selection=selection,
                      frozen_prediction_and_reference_values_changed=False)
        args.output_base.with_suffix('.layout.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Rendered separate figure; shared figures, tables and manifest unchanged')
        return
    if args.main_only:
        qa += [r for r in read(MANIFEST)['layout_audit'] if r['display'] == 'complete_appendix']
    manifest['layout_audit'], manifest['main_figure_selection'] = qa, selection
    manifest['script_sha256']=digest(Path(__file__))
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Published', len(cases), 'drugs;', manifest['scalar_rows'], 'scalar rows;',
          manifest['clinical_curve_points'], 'target points; no fit performed')


if __name__=='__main__':
    main()
