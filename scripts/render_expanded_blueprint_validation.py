#!/usr/bin/env python3
"""One paired drug-level comparison figure for the expanded validation batch."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'experiments/blueprint_design/expanded_validation_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--source-qualified', action='store_true')
    args = parser.parse_args()
    result_dir = BATCH / 'source_qualified' if args.source_qualified else BATCH
    result = json.loads((result_dir / 'results.json').read_text())
    if not result['all_predictions_locked_before_scoring']:
        raise ValueError('Uncommitted predictions cannot be plotted')
    protocol = json.loads((ROOT / result['protocol']['path']).read_text())
    import hashlib
    if hashlib.sha256((ROOT / result['protocol']['path']).read_bytes()).hexdigest() != result['protocol']['sha256']:
        raise ValueError('Protocol checksum changed')
    lookup = {(r['case_id'], r['condition']): r for r in result['units']}
    methods = ('all_ml', 'available')
    labels = ('Predicted PK inputs', 'Source-qualified evidence' if args.source_qualified else 'Available evidence first')
    colors = ('#0072B2', '#D55E00')
    markers = ('o', 'D')
    groups = [('old_20', 'A  Original 20 compounds'), ('new_compounds', 'B  Added clinical compounds')]
    values = [r['curve_metrics'].get('gmfe') for r in result['units'] if r['condition'] in methods]
    valid = [x for x in values if x is not None and math.isfinite(x)]
    if not valid or min(valid) < 1 - 1e-10:
        raise ValueError('Expected positive fold errors at least one')
    xmax = max(valid) ** 1.04
    ticks = [x for p in range(8) for x in (10**p, 2*10**p, 5*10**p) if x <= xmax]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.25))
    fig.subplots_adjust(left=.175, right=.98, bottom=.23, top=.84, wspace=.88)
    nmax = max(sum(c['validation_stratum'] == name for c in protocol['cases']) for name, _ in groups)
    for ax, (name, title) in zip(axes, groups):
        cases = sorted((c for c in protocol['cases'] if c['validation_stratum'] == name), key=lambda c: c['drug'])
        ax.set_xscale('log'); ax.set_xlim(.96, xmax)
        ax.set_ylim(nmax-.4, -.7)
        ax.set_yticks(range(len(cases)), [c['drug'] for c in cases])
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis='y', length=0, pad=6)
        ax.grid(axis='x', color='#d8e0e6', linewidth=.5, zorder=0)
        ax.axvline(2, color='#607483', linewidth=.8, linestyle='--', zorder=1)
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        ax.set_xlabel('Curve GMFE (log scale)\nLower is better', labelpad=9)
        for i, case in enumerate(cases):
            pair = [lookup[(case['case_id'], method)]['curve_metrics'].get('gmfe') for method in methods]
            if all(v is not None for v in pair):
                ax.plot(pair, [i, i], color='#95a4af', linewidth=.9, zorder=2)
            for j, val in enumerate(pair):
                if val is not None:
                    ax.plot(val, i, marker=markers[j], markersize=5.5 if j == 0 else 3.6,
                        color=colors[j], markerfacecolor='white' if j == 0 else colors[j],
                        markeredgewidth=1, linestyle='none', zorder=3+j)
            if all(v is None for v in pair):
                ax.text(.03, i, 'Construction unavailable', transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color='#687784', va='center')
        summaries = [next(s for s in result['summaries'] if s['group'] == name and s['condition'] == m) for m in methods]
        a, b = summaries
        total = a['attempted']
        pairs = [[lookup[(c['case_id'], m)]['curve_metrics'].get('gmfe') for m in methods] for c in cases]
        pairs = [p for p in pairs if all(v is not None for v in p)]
        means = [math.exp(sum(math.log(p[j]) for p in pairs)/len(pairs)) for j in range(2)]
        summary = f"Scored: {a['scored']}/{total} → {b['scored']}/{total} compounds\nPaired GMFE: {means[0]:.2f} → {means[1]:.2f} (n={len(pairs)})"
        ax.text(0, -.16, summary, transform=ax.transAxes, ha='left', va='top', fontsize=8, linespacing=1.5)
    handles = [Line2D([], [], color=c, marker=m, markersize=5, linestyle='none',
        markerfacecolor='white' if i == 0 else c, label=label)
        for i, (c, m, label) in enumerate(zip(colors, markers, labels))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .928), ncol=2, frameon=False)
    fig.text(.04, .985, 'Does evidence help when the drug range expands?', ha='left', va='top',
        fontsize=11, fontweight='bold', color='#243746')
    fig.text(.04, .95, 'Fixed selected algorithm; common retrained weights; no target-curve fitting',
        ha='left', va='top', fontsize=8, color='#506474')
    fig.text(.04, .035,
        'Each row is one clinical scenario. Missing predictions remain visible. The dashed line marks 2-fold error.\n'
        'This is development validation; repeated evidence conditions are not independent samples.',
        ha='left', va='bottom', fontsize=7, color='#506474', linespacing=1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    cropped = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            cropped.append(artist.get_text())
    if cropped:
        raise ValueError('Figure text exceeds canvas: '+str(cropped))
    if args.preview:
        fig.savefig(BATCH / '_comparison_preview.png', dpi=180)
    fig.savefig(BATCH / 'comparison.svg')
    plt.close(fig)
    print(json.dumps({'figure': str(BATCH / 'comparison.svg'), 'scored_pairs':len(valid)//2,
        'gmfe_min':min(valid), 'gmfe_max':max(valid), 'failed_values':sum(x is None for x in values),
        'text_bounds_passed':True}))


if __name__ == '__main__':
    main()
