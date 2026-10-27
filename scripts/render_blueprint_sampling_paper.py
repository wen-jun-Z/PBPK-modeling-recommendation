#!/usr/bin/env python3
"""Render publication assets from the locked 30-drug sampling evaluation.

No fitting, clinical extraction, source mutation, or experimental re-selection is
performed. Run ``preview`` first, inspect the PNGs, then run ``export`` to create
the matching PDFs. Tables retain every prespecified layout and requested goal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from report_sampling_stability import METHODS, RUN, load_complete
from report_sampling_expansion import bootstrap

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'results/figures/experiments'
TABLES = ROOT / 'results/tables'
PUBLICATION = ROOT / 'results/data/blueprint_sampling_publication.json'
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')
LABELS = {
    'previous_learned': 'Legacy sampling (H, 5)',
    'aligned_only': 'Aligned sampling (H, 5)',
    'stable': 'Learned sampling (H, 5)',
    'time': 'Time coverage (5)',
    'stable_repaired': 'Learned + repair (H, 5)',
    'time_repaired': 'Time + repair (5)',
    'no_addition': 'Limited fit (4)',
    'no_addition_repaired': 'Limited fit + repair (4)',
}
PRIMARY_METHODS = ('time_repaired', 'no_addition_repaired',
                   'stable_repaired', 'previous_learned')
COMPARISON_METHOD_LABELS = ('Time\ncoverage\n(5 obs.)', 'Limited\ndata\n(4 obs.)',
                            'Learned\nsampling\n(5 obs.)', 'Legacy\nsampling\n(5 obs.)')
COLORS = {'no_addition_repaired': '#A47D3D', 'previous_learned': '#4B7A65',
          'time_repaired': '#527F9E', 'stable_repaired': '#8064A0'}
PALES = {'no_addition_repaired': '#FCF7E9', 'previous_learned': '#F1F7F2',
         'time_repaired': '#F1F7FB', 'stable_repaired': '#F5F1FA'}
MARKERS = {'no_addition_repaired': 'x', 'previous_learned': 'v',
           'time_repaired': 's', 'stable_repaired': 'o'}
COMPARISONS = ('time_repaired/no_addition_repaired',
               'time_repaired/time',
               'stable_repaired/time_repaired')
COMPARISON_LABELS = ('Observation augmentation\n(5 vs 4 observations)',
                     'Capacity repair\n(same 5 observations)',
                     'Learned vs time sampling\n(same 5 observations)')
COHORTS = ('original_ten', 'added_twenty', 'all_thirty')
COHORT_NAMES = ('Original 10 drugs / 9 clusters', 'Added 20 drugs / 20 clusters',
                'All 30 drugs / 29 clusters')
CAPTIONS = {
    'blueprint_sampling_comparison':
        'Sampling-design extension: thirty-drug clinical prediction errors under the '
        'frozen late-four-point protocol. Time coverage (outlined first column) adds '
        'one development observation; Limited data retains four. Both use the same '
        'PBPK and feasibility-repair policy. Historical consensus-based Learned sampling '
        'and Legacy sampling each use five observations and remain separate columns; '
        'Legacy sampling has no repair. Every cell '
        'is one drug/method error. Blue denotes Cmax and purple denotes curve GMFE; '
        'both sequential palettes share the same logarithmic range, with darker colors indicating larger errors and no clipped '
        'outliers. The horizontal line separates the original 10 from the added 20 drugs; '
        'asterisks mark within-study arm pairs. The bottom row gives drug-equal GMFE. '
        'Sampled Cmax is evaluated only at target observation times. All data were '
        'historically viewed; these are development results, not a new blind test.',
    'blueprint_sampling_validation':
        'Additional sampling-design and update analyses under the frozen late-four/Cmax '
        'protocol. Rows compare time-coverage augmentation with the limited-observation '
        'fit (five versus four observations, same repair policy); capacity repair with '
        'no repair (identical five observations); and historical Learned sampling with '
        'time coverage (identical five-observation budget and repair policy). Points '
        'are drug-equal geometric means of paired error ratios; values below one favor '
        'the first-named method. Bars show 95% target-study-cluster bootstrap intervals '
        '(2,000 resamples). Parentheses give improved/worse/tied drug counts. The dashed '
        'line denotes equal error. Added 20 remains the prespecified primary cohort '
        'of this sampling experiment; Original 10 and All 30 are secondary summaries. '
        'These contrasts do not replace the equal-observation classical-model controls.',
}


def number(value):
    return float(value) if value not in ('', None) else None


def geometric(values):
    if not values:
        return None
    if any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError('Finite positive fold errors required')
    return math.exp(sum(math.log(v) for v in values) / len(values))


def stat(rows, method, regime, goal, endpoint, cohort=None):
    values = [number(r['error']) for r in rows if r['method'] == method
              and r['regime'] == regime and r['requested_goal'] == goal
              and r['endpoint'] == endpoint and (cohort is None or r['cohort'] == cohort)
              and r['error'] != '']
    return geometric(values)


def load():
    state, protocol, result, rows = load_complete()
    audit = json.loads((RUN / 'audit.json').read_text())
    if audit.get('status') != 'passed' or audit['results'] != state['results']:
        raise ValueError('Publication requires a matching independent numerical audit')
    if len(rows) != 15840 or result['counts']['drugs'] != 30:
        raise ValueError('Unexpected evaluation scope')
    for scope in COHORTS:
        for method in METHODS:
            for ep in ('cmax', 'curve'):
                observed = stat(rows, method, 'late_four', 'cmax', ep,
                                None if scope == 'all_thirty' else scope)
                expected = result['summary'][scope][method][ep]['estimate']
                if not math.isclose(observed, expected, rel_tol=1e-12):
                    raise ValueError('Publication aggregate disagrees with frozen result')
    return state, protocol, result, rows, audit


def publication_statistics(protocol, rows):
    """Recompute the declared time-rule view; never relabel learned predictions."""
    cases = {c['case_id']: c for c in protocol['cases']}
    original = set(protocol['original_case_ids'])
    scopes = dict(original_ten=original, added_twenty=set(cases)-original, all_thirty=set(cases))
    primary = {(r['case_id'], r['method'], r['endpoint']): number(r['error'])
               for r in rows if r['regime'] == 'late_four' and r['requested_goal'] == 'cmax'}

    def aggregate(values):
        values = {k: v for k, v in values.items() if v is not None and v > 0}
        return bootstrap(values, {k: cases[k]['cluster_id'] for k in values}, **protocol['bootstrap'])

    def paired(ids, numerator, denominator, ep):
        ratios = {k: primary[k, numerator, ep]/primary[k, denominator, ep] for k in sorted(ids)
                  if primary.get((k, numerator, ep)) is not None
                  and primary.get((k, denominator, ep)) is not None}
        values = list(ratios.values())
        return aggregate(ratios) | dict(
            improved=sum(math.log(x) < -1e-10 for x in values),
            worse=sum(math.log(x) > 1e-10 for x in values),
            tied=sum(abs(math.log(x)) <= 1e-10 for x in values))

    comparisons, deletions = {}, {}
    for scope, ids in scopes.items():
        comparisons[scope] = {pair: {ep: paired(ids, *pair.split('/'), ep)
                                    for ep in ('cmax', 'curve')} for pair in COMPARISONS}
        if scope == 'original_ten':
            continue
        subsets = dict(all=ids, without_Montelukast={k for k in ids if cases[k]['drug'] != 'Montelukast'})
        subsets.update({'without_study:'+cluster: {k for k in ids if cases[k]['cluster_id'] != cluster}
                        for cluster in sorted({cases[k]['cluster_id'] for k in ids})})
        deletions[scope] = {}
        for group, keep in subsets.items():
            item = dict(included=sorted(keep), excluded=sorted(ids-keep))
            for ep in ('cmax', 'curve'):
                item[ep] = {m: aggregate({k: primary[k, m, ep] for k in keep})
                            for m in ('no_addition_repaired', 'time_repaired', 'stable_repaired')}
                for pair in ('time_repaired/no_addition_repaired', 'stable_repaired/time_repaired'):
                    item[ep][pair] = paired(keep, *pair.split('/'), ep)
            deletions[scope][group] = item
    coverage = []
    for scope, ids in scopes.items():
        for regime in ('early_four', 'late_four'):
            for goal in ('cmax', 'terminal_half_life', 'auc_inf'):
                for ep in ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life',
                           'auc_inf', 'clearance', 'vz', 'mrt', 'vss'):
                    selected = [r for r in rows if r['case_id'] in ids and r['method'] == 'time_repaired'
                                and r['regime'] == regime and r['requested_goal'] == goal and r['endpoint'] == ep]
                    valid = [number(r['error']) for r in selected if number(r['error']) is not None]
                    value = (sum(valid)/len(valid) if valid else None) if ep == 'tmax' else geometric(valid)
                    coverage.append(dict(cohort=scope, regime=regime, requested_goal=goal, endpoint=ep,
                        n_drugs=len(ids), n_scored=len(valid),
                        n_predictions=len(valid) if ep == 'curve' else sum(r['prediction'] != '' for r in selected),
                        n_references=len(valid) if ep == 'curve' else sum(r['reference'] != '' for r in selected),
                        metric='mean_absolute_error_h' if ep == 'tmax' else 'drug_equal_gmfe', estimate=value))
    return dict(primary_method='time_repaired', comparisons=comparisons, deletions=dict(groups=deletions),
                all_prespecified_time_endpoints=coverage, bootstrap=protocol['bootstrap'],
                selection_of_main_method='Post-development simplification; frozen historical methods remain unchanged.')


def style():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    # Use the active project environment's fonts even if an older font cache
    # still points at a relocated virtual environment.
    for font in (Path(matplotlib.get_data_path()) / 'fonts/ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    sys.path.insert(0, str(SKILL))
    from setup_style import setup_style
    setup_style(journal='nature', lang='en', use_sciplots=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.labelsize': 7.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 7,
        'ytick.labelsize': 7.2, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None,
        'figure.constrained_layout.use': False, 'axes.linewidth': .6})


def comparison(protocol, result, rows):
    """Separate method columns expose all 240 errors, without overplotting."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import NullLocator
    from matplotlib.font_manager import FontProperties
    method_font = FontProperties(family='DejaVu Sans', size=7)
    original = set(protocol['original_case_ids'])
    cases = sorted(protocol['cases'], key=lambda c: (c['case_id'] not in original, c['drug']))
    subset = [r for r in rows if r['regime'] == 'late_four'
              and r['requested_goal'] == 'cmax' and r['method'] in PRIMARY_METHODS
              and r['endpoint'] in ('cmax', 'curve')]
    lookup = {(r['case_id'], r['method'], r['endpoint']): number(r['error']) for r in subset}
    if len(lookup) != 240:
        raise ValueError('All 30 x 4 x 2 primary results must be displayed')
    if any(v is None or not math.isfinite(v) or v < 1. - 1e-12 for v in lookup.values()):
        raise ValueError('Every primary error must be a finite fold error')
    # Fixed manuscript width; 9.3-point row pitch keeps all 30 drugs readable
    # without reserving a nearly full manuscript page for one numeric matrix.
    fig = plt.figure(figsize=(5.5, 5.2))
    axes = [fig.add_axes([.257, .134, .350, .747]),
            fig.add_axes([.630, .134, .350, .747])]
    # Hue identifies the endpoint; matched sequential ramps share one log scale.
    # The maximum is the actual maximum: the large Montelukast error is retained.
    palettes = {
        'cmax': ['#E4EFF6', '#BCD3E3', '#82ABC7', '#477B9F', '#234E70'],
        'curve': ['#EEE8F6', '#CFC0E0', '#AE92C8', '#8561A7', '#563770'],
    }
    accents = {'cmax': ('#E4EFF6', '#477B9F', '#82ABC7'),
               'curve': ('#EEE8F6', '#8561A7', '#AE92C8')}
    fig._pbpk_endpoint_palettes = palettes
    norm = LogNorm(vmin=1., vmax=max(lookup.values()))
    images = []
    short_labels = COMPARISON_METHOD_LABELS
    fig._pbpk_method_headers = []
    for ax, ep, panel in zip(axes, ('cmax', 'curve'), ('a  Sampled Cmax FE', 'b  Curve GMFE')):
        cmap = LinearSegmentedColormap.from_list('paper_fold_error_' + ep, palettes[ep])
        pale, accent, outline = accents[ep]
        values = np.array([[lookup[c['case_id'], m, ep] for m in PRIMARY_METHODS] for c in cases])
        image = ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
        images.append(image)
        for row in range(30):
            for col in range(4):
                value = values[row, col]
                rgb = np.asarray(cmap(norm(value))[:3])
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
                luminance = float(linear @ np.array([.2126, .7152, .0722]))
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=7.,
                        color='white' if luminance < .179 else 'black')
        # Method identity remains explicit text; a narrow outline marks the
        # time-coverage reference without introducing method-specific color scales.
        for col, (method, text) in enumerate(zip(PRIMARY_METHODS, short_labels)):
            current = method == 'time_repaired'
            ax.add_patch(Rectangle((col/4, 1.004), .25, .090,
                transform=ax.transAxes, facecolor=pale if current else '#F2F4F5',
                edgecolor='white', linewidth=.7, clip_on=False))
            ax.plot([col/4+.008, (col+1)/4-.008], [1.095, 1.095],
                    transform=ax.transAxes, color=accent if current else '#AAB6BE',
                    lw=1.3 if current else .65, clip_on=False)
            header = ax.text((col+.5)/4, 1.049, text, transform=ax.transAxes, ha='center',
                             va='center', fontsize=7., color='#111111',
                             fontproperties=method_font, linespacing=1.0)
            fig._pbpk_method_headers.append(header)
            ax.text(col, 30.78, f'{result["summary"]["all_thirty"][method][ep]["estimate"]:.3f}',
                    ha='center', va='center', fontsize=7.2,
                    weight='bold' if current else 'normal', color='#203340', clip_on=False)
        ax.text(0, 1.123, panel, transform=ax.transAxes, fontsize=8.3,
                weight='bold', color='#17232C', va='bottom')
        ax.set_yticks(range(30), [c['drug'] + ('*' if not c.get('cross_study_pair', True) else '')
                                 for c in cases] if ep == 'cmax' else ['']*30)
        ax.tick_params(axis='y', length=0, pad=4, labelsize=7.)
        ax.set_xticks([])
        ax.set_xticks(np.arange(-.5, 4., 1.), minor=True)
        ax.set_yticks(np.arange(-.5, 30., 1.), minor=True)
        ax.grid(which='minor', color='white', lw=.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.axhline(9.5, color=accent, lw=.85)
        ax.add_patch(Rectangle((-.5, -.5), 1., 30., facecolor='none',
                              edgecolor=outline, linewidth=.65, zorder=4))
        ax.plot([-.5, 3.5], [30.02, 30.02], color='#879CA9', lw=.6, clip_on=False)
        ax.set_ylim(29.5, -.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    # Aggregate row is separated from drug values, and is not a new observation.
    axes[0].text(-.60, 30.78, 'Drug-equal GMFE', ha='right', va='center',
                 fontsize=7., color='#17232C', weight='bold', clip_on=False)
    # Cohort markers remain legible in grayscale and do not encode performance.
    for start, stop, label in [(0, 10, 'Original 10'), (10, 30, 'Added 20')]:
        bottom = .134 + .747*(1-stop/30)
        top = .134 + .747*(1-start/30)
        fig.lines.append(plt.Line2D([.031, .031], [bottom+.006, top-.006],
                                   transform=fig.transFigure, color='#9AAEBB', lw=.8))
        fig.text(.017, (bottom+top)/2, label, rotation=90, ha='center', va='center',
                 fontsize=7., color='#4B6372')
    for left, image in zip((.257, .630), images):
        cax = fig.add_axes([left, .042, .350, .012])
        bar = fig.colorbar(image, cax=cax, orientation='horizontal',
                          ticks=[1, 2, 5, 10, 50, 100, 300])
        bar.ax.set_xticklabels(['1', '2', '5', '10', '50', '100', '300'])
        bar.ax.xaxis.set_minor_locator(NullLocator())
        bar.ax.tick_params(labelsize=7, length=2.5, pad=1.8)
        bar.outline.set_visible(False)
    fig.text(.245, .048, 'Fold error (log)', ha='right', va='center', fontsize=7., color='#4B6372')
    return fig


def validation(result):
    """Paired contribution contrasts, stratified by the declared cohort."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig = plt.figure(figsize=(5.5, 5.4))
    for index, (scope, name) in enumerate(zip(COHORTS, COHORT_NAMES)):
        bottom = .709-index*.282
        fig.text(.012, bottom+.197, name, fontsize=7.7, color='#17232C', weight='bold')
        for col, ep in enumerate(('cmax', 'curve')):
            ax = fig.add_axes([.305+col*.348, bottom, .306, .174])
            color = '#0072B2' if ep == 'cmax' else '#E66100'
            for j, pair in enumerate(COMPARISONS):
                stats = result['comparisons'][scope][pair][ep]
                low, high = stats['ci95']
                ax.hlines(j, low, high, color=color, lw=1.3)
                ax.vlines([low, high], j-.07, j+.07, color=color, lw=.8)
                ax.plot(stats['estimate'], j, 'o' if ep == 'cmax' else 's',
                        color=color, markerfacecolor='white', markeredgewidth=1.1,
                        ms=4, zorder=3)
                # Each annotation is adjacent to its row, separated from the CI.
                text = f'{stats["estimate"]:.3f}  ({stats["improved"]}/{stats["worse"]}/{stats["tied"]})'
                ax.text(.98, j+.36, text, transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=7., color='#4D606B',
                        bbox=dict(facecolor='white', edgecolor='none', pad=.2))
            ax.axvline(1, color='#85959E', linestyle='--', lw=.7)
            ax.set_ylim(2.64, -.48)
            ax.set_xscale('log')
            # Shared scales within each endpoint retain every interval.
            all_ci = [v for s in COHORTS for pair in COMPARISONS
                      for v in result['comparisons'][s][pair][ep]['ci95']]
            ax.set_xlim(min(.30, min(all_ci)*.95), max(1.18, max(all_ci)*1.04))
            ax.xaxis.set_major_locator(FixedLocator([.3, .5, 1.] if ep == 'cmax' else [.3, .5, 1.]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.set_yticks(range(3), COMPARISON_LABELS if col == 0 else ['']*3)
            ax.tick_params(axis='y', length=0, pad=5, labelsize=7.)
            ax.tick_params(axis='x', pad=2)
            ax.grid(axis='x', color='#E0E7EC', lw=.45)
            if index == 0:
                ax.set_title(('a  Sampled Cmax FE' if col == 0 else 'b  Curve GMFE'), loc='left', pad=37,
                             fontsize=8.4, color='#17232C', weight='bold')
            if index == 2:
                ax.set_xlabel('Paired error ratio (log scale)', labelpad=4)
    return fig


def tex_number(value):
    return r'\textemdash{}' if value is None else f'{value:.3f}'


def summary_table(rows, result):
    lines = [r'% Generated only from completed sampling_stability_20261009.',
        r'\begingroup\captionsetup[table]{name=Table}',
        r'\definecolor{BPSInk}{HTML}{233F4D}\definecolor{BPSPale}{HTML}{EDF3F5}',
        r'\setlength{\tabcolsep}{2.8pt}\renewcommand{\arraystretch}{1.10}',
        r'\fontsize{8}{9.6}\selectfont',
        r'\newlength{\BPSMethodWidth}\newlength{\BPSNumericWidth}',
        r'\setlength{\BPSMethodWidth}{.27\linewidth}',
        r'\setlength{\BPSNumericWidth}{\dimexpr.73\linewidth-12\tabcolsep\relax}',
        r'\setlength{\BPSNumericWidth}{\dimexpr\BPSNumericWidth/6\relax}',
        # Retain the five configurations used in the current comparisons.
        # One tabular avoids
        # longtable's transient first-pass multicolumn width warnings, while
        # fixed, equal numeric columns retain an exactly linewidth-wide grid.
        r'\captionof{table}{Additional sampling-design analysis for the five configurations retained in the current comparisons. All 30 drugs, both prespecified layouts and all three requests are included. Geometric mean fold errors are drug-equal; lower is better.}\label{tab:blueprint_sampling_summary}\par\vspace{3pt}',
        r'\noindent\begin{tabular}{@{}>{\raggedright\arraybackslash}p{\BPSMethodWidth}*{6}{>{\centering\arraybackslash}p{\BPSNumericWidth}}@{}}',
        r'\toprule']
    order = ('time_repaired', 'no_addition_repaired', 'time',
             'stable_repaired', 'previous_learned')
    conditions = [(regime, goal) for regime in ('early_four', 'late_four')
                  for goal in ('cmax', 'terminal_half_life', 'auc_inf')]
    for letter, ep, title in [('A', 'cmax', 'Cmax GMFE'), ('B', 'curve', 'Curve GMFE'),
                             ('C', 'requested', 'Requested scalar GMFE')]:
        lines.extend([r'\addlinespace[5pt]',
            r'\rowcolor{BPSPale}\multicolumn{7}{@{}l}{\textbf{'+letter+'. All 30 drugs: '+title+r'}}\\',
            r'Method & E--C & E--H & E--A & L--C & L--H & L--A\\\midrule'])
        for method in order:
            values = [stat(rows, method, regime, goal, goal if ep == 'requested' else ep)
                      for regime, goal in conditions]
            if method == 'time_repaired': lines.append(r'\rowcolor{BPSPale}')
            lines.append(LABELS[method]+' & '+' & '.join(map(tex_number, values))+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\par\vspace{6pt}',
        # Notes are ordinary paragraphs: they do not participate in TeX's
        # multicolumn width allocation and cannot distort the numeric columns.
        r'\noindent\textbf{Reading the table.} E/L: early/late four-point layout. C/H/A: requested sampled Cmax, terminal half-life, or AUC$_\infty$. All entries use 30 drugs. Curve GMFE evaluates the entire positive observed target curve; sampled Cmax uses actual target sampling times.\par',
        r'\noindent\textbf{Missing references.} AUC$_\infty$ requests remain in the experiment, but scalar errors are unavailable under the common clinical-curve contract (0/30). Half-life errors include weak-tail cases and are descriptive. Missing observations are not imputed.\par',
        r'\noindent\textbf{Configurations.} Parenthetic 4/5 gives the development-observation count. Time + repair (5) adds the point farthest from visible observations in log time, then applies the declared visible-evidence feasibility repair. Time coverage (5) uses the same point without repair. H marks historical learned selectors; Learned + repair (H, 5) uses the frozen consensus selector with the same repair, while Legacy sampling (H, 5) has no repair. Limited fit + repair (4) adds no observation. These are sampling and update variants, not separate PBPK architectures.\par',
        r'\noindent\textbf{Scope.} The five configurations correspond to the current sampling, observation-augmentation and repair comparisons; they were not chosen by prediction accuracy. Intermediate historical versions and the repeated cohort summary remain in the frozen machine-readable results. All drug regressions and unavailable references are retained. Reused fits and repeated references are not independent experiments; no predictions were refitted for this presentation.\par',
        r'\endgroup'])
    return '\n'.join(lines)+'\n'


def audit_table(state, protocol, result, rows, audit, publication):
    subset = [r for r in rows if r['method'] == 'time_repaired' and r['regime'] == 'late_four'
              and r['requested_goal'] == 'cmax']
    flags = {ep: sum(r['weak_tail'] == 'True' for r in subset if r['endpoint'] == ep)
             for ep in ('terminal_half_life', 'auc_inf', 'mrt')}
    scalar = {ep: sum(bool(r['prediction']) for r in subset if r['endpoint'] == ep)
              for ep in ('auc_inf', 'clearance', 'vz', 'mrt', 'vss')}
    if len(set(scalar.values())) != 1:
        raise ValueError('Do not compress unequal endpoint availability')
    repaired = sum(not unit['reused'] for unit in state['units'].values())
    if repaired != 4:
        raise ValueError('Unexpected repair-fit count')
    benefit = publication['comparisons']['all_thirty']['time_repaired/no_addition_repaired']['cmax']
    historic = publication['comparisons']['all_thirty']['stable_repaired/time_repaired']['cmax']
    without = publication['deletions']['groups']['added_twenty']['without_Montelukast']['cmax']['time_repaired/no_addition_repaired']
    def description(st):
        return (f'{st["improved"]} better, {st["worse"]} worse, {st["tied"]} tied; '
                f'ratio {st["estimate"]:.3f} [{st["ci95"][0]:.3f}, {st["ci95"][1]:.3f}].')
    entries = [
        ('Cohort and design', '30 drugs; 60 four-point layouts; 180 goal-specific decisions; 28 cross-study pairs and 2 within-study arm pairs.',
         'Historical viewed data. Drug counts, layouts and decisions are distinct units; none imply 180 independent drugs.'),
        ('Sampling extension', 'Time coverage chooses the largest nearest-visible log-time distance; ties choose the earlier time. No learned sampling model is required.',
         'Four visible observations still calibrate the drug-specific PBPK parameters; an acquired fifth point is then added. Selection does not use its hidden value.'),
        ('Historical learner audit', f'{audit["independent_model_refits"]} full/nested ridge fits reconstructed; original 10 only; {audit["replayed_decisions"]}/180 historical decisions replayed.',
         'Learning variants remain historical controls. Their added 20 drugs did not train the selector. All clinical datasets were already viewed; this is not a new blind test.'),
        ('Source and lock integrity', f'{audit["source_hashes"]:,} pinned source hashes checked; selections locked before fits; predictions locked before scoring.',
         'Runtime ordering and unchanged inputs are verified. Historical viewing of studies is not undone by the locks.'),
        ('Computation and reuse', f'{result["counts"]["checked_units"]} required prediction units verified: 161 reused and 4 refitted; 30/30 drugs retained.',
         'Cache reuse is not a new fit. Completion is an execution measure, not clinical validity or an independent success-rate experiment.'),
        ('Physical feasibility repair', 'Only Montelukast triggers repair. Late-four Cmax FE: 332.50 to 1.34; curve GMFE: 52.03 to 2.08.',
         'Visible mass-bound conflict permits an unmeasured RBC bound change. Current time coverage and the historical repaired learner share this policy; its gain is not a learning-specific contribution.'),
        ('Arithmetic reproduction', f'{audit["endpoint_rows"]:,} endpoint rows; {audit["scalar_error_recomputations"]:,} scalar errors; {audit["statistic_checks"]} primary aggregates and paired statistics independently checked.',
         'Arithmetic consistency does not establish biological adequacy or clinically calibrated uncertainty.'),
        ('Clinical reference coverage', 'Curve and five scalars: window AUC, sampled Cmax/Tmax, last concentration and half-life (30/30 each). Five other scalar references: 0/30.',
         'AUC$_\\infty$, CL, $V_z$, MRT and $V_{\\rm ss}$ have no common qualified reference here. Eleven output categories do not mean eleven validated categories.'),
        ('Output and tail limitations', f'AUC$_\\infty$/CL/$V_z$/MRT/$V_{{\\rm ss}}$ computable in {next(iter(scalar.values()))}/30; weak-tail flags: half-life {flags["terminal_half_life"]}/30, AUC$_\\infty$ {flags["auc_inf"]}/27, MRT {flags["mrt"]}/27.',
         'Availability and warnings refer to the primary blueprint predictions. Missing values stay missing, and flagged comparisons remain descriptive.'),
        ('Benefit of one observation', 'Time-coverage augmentation (5) / limited fit (4), all 30 Cmax: '+description(benefit),
         'Supports an average benefit of time-coverage acquisition in the late-four setting; some drugs still worsen. Both arms use the same repair policy.'),
        ('Influential-case sensitivity', 'Added 19 after deleting Montelukast, time augmentation (5) / limited fit (4), Cmax: '+description(without),
         'This sensitivity uses the time-rule predictions, not the historical learner. The main analysis retains all 30 drugs; no fitting or selection is repeated.'),
        ('Historical selection benefit', 'Repaired learned sampling / time coverage, both 5 observations, all 30 Cmax: '+description(historic),
         'No general learning-specific superiority. This historical contrast does not make a learned selector necessary for the current blueprint.'),
    ]
    lines = [r'% Current sampling validation audit; not the old capability experiment.',
        r'\begingroup\captionsetup[table]{name=Table}',
        r'\definecolor{BPAuditPale}{HTML}{EDF3F5}',
        r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.14}\fontsize{8.3}{11.2}\selectfont',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.19\linewidth}>{\raggedright\arraybackslash}p{.385\linewidth}>{\raggedright\arraybackslash}p{.365\linewidth}@{}}',
        r'\caption{Audit and claim boundaries for the 30-drug PBPK workflow and its additional sampling-design experiment. Counts are logical audit units, not additional independent clinical trials.}\label{tab:blueprint_capability}\\\toprule',
        r'\rowcolor{BPAuditPale}\textbf{Check} & \textbf{Observed result} & \textbf{Interpretation boundary}\\\midrule\endfirsthead',
        r'\toprule\rowcolor{BPAuditPale}\textbf{Check} & \textbf{Observed result} & \textbf{Interpretation boundary}\\\midrule\endhead',
        r'\bottomrule\endfoot\bottomrule\endlastfoot']
    for i, row in enumerate(entries):
        if i % 2 == 0: lines.append(r'\rowcolor{BPAuditPale}')
        lines.append(' & '.join(row)+r'\\')
        # A trailing addlinespace can force longtable to emit an otherwise
        # empty continuation page with only the repeated header.
        if i < len(entries)-1:
            lines.append(r'\addlinespace[3pt]')
    lines.append(r'\end{longtable}\endgroup')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preview', 'export', 'tables'])
    args = parser.parse_args()
    state, protocol, result, rows, audit = load()
    publication = publication_statistics(protocol, rows)
    TABLES.mkdir(exist_ok=True, parents=True)
    (TABLES / 'blueprint_sampling_summary.tex').write_text(summary_table(rows, result))
    def reference(path):
        return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    document = dict(version='time-coverage-blueprint-publication-1', source_result=state['results'],
        source_protocol=state['protocol'], source_endpoints=result['endpoints'],
        numerical_audit=reference(RUN/'audit.json'), renderer=reference(Path(__file__).resolve()),
        new_fits=0, new_training=0, historical_results_unchanged=True,
        main_prediction_method='time_repaired', historical_learned_method='stable_repaired',
        captions=CAPTIONS, statistics=publication, figures=[],
        scope='Post-development presentation of the frozen time-rule method; no outcome-based drug exclusions in the primary analysis.')
    def publish():
        document['tables'] = [reference(TABLES/name) for name in
                              ('blueprint_sampling_summary.tex',)]
        PUBLICATION.parent.mkdir(parents=True, exist_ok=True)
        PUBLICATION.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if args.command == 'tables':
        publish()
        return
    style()
    from profile_data import profile_data
    from visual_qa import audit_layout, render_preview
    import pandas as pd
    import matplotlib.pyplot as plt
    profile = profile_data(pd.DataFrame([{'drug': r['drug'], 'cohort': r['cohort'],
        'method': r['method'], 'endpoint': r['endpoint'], 'error': number(r['error'])}
        for r in rows if r['regime'] == 'late_four' and r['requested_goal'] == 'cmax'
        and r['endpoint'] in ('cmax', 'curve')]), group_cols=['cohort', 'endpoint'])
    FIGURES.mkdir(exist_ok=True, parents=True)
    for name, fig in [('blueprint_sampling_comparison', comparison(protocol, result, rows)),
                      ('blueprint_sampling_validation', validation(publication))]:
        render_preview(fig, str(FIGURES / (name + '.png')), dpi=300)
        qa = audit_layout(fig)
        if any(level == 'FAIL' for level, _ in qa):
            raise ValueError('Figure audit failed: ' + str(qa))
        # Keep the same final physical canvas in the PNG and the vector PDF.
        fig.savefig(FIGURES / (name + '.png'), dpi=300, bbox_inches=None)
        if args.command == 'export':
            fig.savefig(FIGURES / (name + '.pdf'), format='pdf', dpi=300)
        from PIL import Image
        with Image.open(FIGURES/(name+'.png')) as preview:
            preview.convert('L').save(FIGURES/(name+'_grayscale.png'), dpi=(300,300))
        document['figures'].append(dict(name=name, png=reference(FIGURES/(name+'.png')),
            pdf=reference(FIGURES/(name+'.pdf')) if args.command == 'export' else None,
            layout_qa=qa, inches=list(fig.get_size_inches()), dpi=300,
            visual_review='Inspect color and grayscale previews before accepting exported artifacts.'))
        print(json.dumps({'figure': name, 'qa': qa, 'inches': list(fig.get_size_inches()),
                          'pdf_exported': args.command == 'export'}, ensure_ascii=False))
        plt.close(fig)
    publish()
    print(json.dumps({'data_rows': len(rows), 'profile_columns': list(profile),
                      'source_integrity': 'passed', 'all_drugs': 30,
                      'prespecified_layout_goal_combinations': 6,
                      'method_variants': len(METHODS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
