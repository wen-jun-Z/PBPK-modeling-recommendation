"""Preview full-window candidate curves and untouched Top-3 recommendation maps.

Reads an exported, outcome-independent candidate set; never executes models,
changes recommendation scores, drops poor curves, resolves ties, or backfills.
The default output is deliberately separate from the manuscript figure assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np
from PIL import Image

from render_recommendation_curve_cases import figure_methods, load_figure_data

DATA = ROOT / 'results/data/model_recommendation_curve_cases_extended.json'
OUT = ROOT / 'results/figures/experiments/previews/recommendation_curve_extension'
SIZE = (5.5, 6.95)  # Exact 139.7-mm manuscript body width; do not crop on export.
Y_OFFSET = 0
INK = '#252F38'
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')
LINESTYLES = ('-', (0, (4.4, 1.8)), (0, (1.0, 1.8)),
              (0, (4.4, 1.6, 1.0, 1.6)), (0, (7.0, 2.2)), (0, (2.2, 1.4)))
MARKERS = ('o', 's', '^', 'D', 'v', 'P')
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7,
        'axes.labelsize': 7.2, 'axes.linewidth': .6,
        'xtick.labelsize': 6.7, 'ytick.labelsize': 6.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'figure.constrained_layout.use': False,
        'savefig.facecolor': 'white',
    })


def label_key(curve):
    return tuple(int(s) if s.isdigit() else s for s in re.split(r'(\d+)', curve['label']))


def validated_cases(data):
    """Validate complete exported data without applying quality-based selection."""
    cases = sorted(data['cases'], key=lambda c: c['compound'])
    assert len(cases) == 2, 'This two-column publication layout requires two cases.'
    methods = figure_methods(data)
    assert len(methods) == 14, 'Expected the frozen fourteen-method publication view.'
    profile = []
    for case in cases:
        ot = np.asarray(case['observation']['time_h'], dtype=float)
        oy = np.asarray(case['observation']['concentration_ng_ml'], dtype=float)
        assert len(ot) == len(oy) and len(ot) > 1
        assert np.isfinite(ot).all() and np.isfinite(oy).all()
        assert np.all(np.diff(ot) > 0) and np.all(ot > 0) and np.all(oy > 0)
        assert 1 <= len(case['curves']) <= len(PALETTE)
        curves = {c['label']: c for c in case['curves']}
        assert len(curves) == len(case['curves'])
        assert len({c['candidate_id'] for c in case['curves']}) == len(curves)
        curve_stats = []
        for curve in case['curves']:
            t = np.asarray(curve['time_h'], dtype=float)
            y = np.asarray(curve['concentration_ng_ml'], dtype=float)
            assert len(t) == len(y) and np.isfinite(t).all() and np.isfinite(y).all()
            assert np.all(np.diff(t) > 0) and t.min() <= ot.min() <= ot.max() <= t.max()
            tx = np.unique(np.r_[ot.min(), t[(t >= ot.min()) & (t <= ot.max())], ot.max()])
            cy = np.interp(tx, t, y)
            assert np.all(cy > 0), 'Nonpositive concentrations cannot be hidden by a log axis.'
            error = np.log(np.interp(ot, t, y) / oy)
            assert np.isclose(np.exp(np.mean(np.abs(error))), curve['gmfe'], rtol=1e-9)
            assert np.isclose(np.sqrt(np.mean(error ** 2)), curve['log_rmse'], rtol=1e-9)
            curve_stats.append({'label': curve['label'], 'candidate_id': curve['candidate_id'],
                'gmfe': curve['gmfe'], 'visible_min': float(cy.min()),
                'visible_max': float(cy.max()), 'full_window_h': [float(tx[0]), float(tx[-1])]})
        assert {r['method_id'] for r in case['rankings']} == {m[0] for m in methods}
        for ranking in case['rankings']:
            assert [p['position'] for p in ranking['positions']] == [1, 2, 3]
            for p in ranking['positions']:
                assert len(p['members']) == p['tie_group_size']
                assert len({m['candidate_id'] for m in p['members']}) == len(p['members'])
                assert np.isclose(sum(m['weight'] for m in p['members']), 1)
                assert all(np.isclose(m['weight'], 1 / p['tie_group_size']) for m in p['members'])
                known = [m for m in p['members'] if m['curve_label']]
                assert np.isclose(sum(m['weight'] for m in known), p['curve_coverage'])
                for m in known:
                    assert curves[m['curve_label']]['candidate_id'] == m['candidate_id']
        profile.append({'compound': case['compound'], 'observations': len(ot),
            'observation_window_h': [float(ot.min()), float(ot.max())],
            'observation_range_ng_ml': [float(oy.min()), float(oy.max())],
            'curves': curve_stats})
    return cases, profile


def curve_style(curve, curves):
    # Number-based colors preserve the original R1–R5 palette across exports.
    match = re.search(r'(\d+)$', curve['label'])
    i = int(match[1]) - 1 if match else list(curves).index(curve)
    return PALETTE[i % len(PALETTE)], LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)]


def candidate_name(curve, case):
    name = curve.get('display_name') or curve['name']
    if 'display_name' not in curve:
        name = name.removesuffix('-DDI').replace('-', '–')
        # Plain models need a version to distinguish otherwise identical names.
        if sum(c['name'] == curve['name'] for c in case['curves']) > 1:
            name += f" [{curve['version'][:7]}]"
    return name


def fig_text(fig, x, y, text, **kwargs):
    return fig.text(x / SIZE[0], (y - Y_OFFSET) / SIZE[1], text, color=INK,
                    va='center', **kwargs)


def add_axis(fig, x, y, w, h):
    return fig.add_axes((x / SIZE[0], (y - Y_OFFSET) / SIZE[1], w / SIZE[0], h / SIZE[1]))


def observed_time_ticks(ot):
    lo, hi = float(ot.min()), float(ot.max())
    available = [v for v in (.25, .5, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 168)
                 if lo * 1.7 < v < hi / 1.7]
    anchors = [lo, hi]
    for target in np.geomspace(lo, hi, 4)[1:-1]:
        if available:
            choice = min(available, key=lambda v: abs(math.log(v / target)))
            anchors.append(choice); available.remove(choice)
    return sorted(anchors)


def draw_curves(fig, cases):
    """Extend the original two-column small multiples: one curve per panel."""
    panel_bounds = []
    fig_text(fig, .12, 6.83, '(a) Individual PBPK predictions',
             fontsize=8.2, fontweight='bold')
    handles = [Line2D([], [], color='#606D78', lw=1.3, label='Prediction'),
               Line2D([], [], color=INK, marker='o', linestyle='none', markersize=3.2,
                      label='Observed mean')]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.58, 6.83 / SIZE[1]),
               borderaxespad=0, frameon=False, ncol=2, fontsize=6.3,
               handlelength=1.3, handletextpad=.4, columnspacing=.8)
    limits = {}
    for case in cases:
        ot = np.asarray(case['observation']['time_h'], dtype=float)
        oy = np.asarray(case['observation']['concentration_ng_ml'], dtype=float)
        full_values = [oy]
        for curve in case['curves']:
            t = np.asarray(curve['time_h']); y = np.asarray(curve['concentration_ng_ml'])
            tx = np.unique(np.r_[ot.min(), t[(t >= ot.min()) & (t <= ot.max())], ot.max()])
            full_values.append(np.interp(tx, t, y))
        vals = np.concatenate(full_values); lo, hi = vals.min(), vals.max()
        padding = max(.12, math.log10(hi / lo) * .06)
        limits[case['compound']] = (10 ** (math.log10(lo) - padding),
                                    10 ** (math.log10(hi) + padding))
    panels = [(case, curve) for case in cases for curve in sorted(case['curves'], key=label_key)]
    assert len(panels) <= 8, 'More than eight candidates needs another page, not smaller text.'
    for i, (case, curve) in enumerate(panels):
        row, col = divmod(i, 2)
        x, bottom = .54 + col * 2.64, 5.76 - row * 1.10
        ax = add_axis(fig, x, bottom, 2.02, .60)
        ot = np.asarray(case['observation']['time_h'], dtype=float)
        oy = np.asarray(case['observation']['concentration_ng_ml'], dtype=float)
        t = np.asarray(curve['time_h']); y = np.asarray(curve['concentration_ng_ml'])
        tx = np.unique(np.r_[ot.min(), t[(t >= ot.min()) & (t <= ot.max())], ot.max()])
        cy = np.interp(tx, t, y)
        color = curve_style(curve, case['curves'])[0]
        ax.plot(tx, cy, color=color, linewidth=1.3, zorder=2)
        ax.scatter(ot, oy, s=11, color=INK, edgecolors='white', linewidths=.25, zorder=3)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(ot.min() / 1.15, ot.max() * 1.12)
        ax.set_ylim(limits[case['compound']])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ticks = sorted(set([float(ot.min()), 2., 12., float(ot.max())]))
        ticks = [v for v in ticks if ot.min() <= v <= ot.max()]
        ax.set_xticks(ticks, [f'{v:g}' for v in ticks])
        ax.minorticks_off(); ax.tick_params(length=2.5, pad=2, labelsize=6.5)
        ax.grid(axis='y', color='#E3E8ED', lw=.4, zorder=0)
        ax.set_title(f"{curve['label']}  {candidate_name(curve, case)}", loc='left',
                     fontsize=6.8, fontweight='bold', pad=16, color=INK)
        dose = f"{case['dose_mg']:g} mg" + (' HCl' if case['compound'] == 'metformin' else '')
        ax.text(0, 1.045, f"{dose}  ·  GMFE {curve['gmfe']:.3f}", transform=ax.transAxes,
                fontsize=6.8, color=color, va='bottom')
        panel_bounds.append({'compound': case['compound'], 'candidate': curve['label'],
                             'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()),
                             'full_window_h': [float(tx[0]), float(tx[-1])],
                             'observations_drawn': len(ot), 'all_curves_visible': True})
    fig_text(fig, .075, 4.51, 'Plasma concentration (ng/mL; log scale)',
             rotation=90, fontsize=7, ha='center')
    fig_text(fig, 2.88, 2.24, 'Time after dose (h; log scale)', fontsize=6.8, ha='center')
    return panel_bounds


def rank_cell(position, color_by_label):
    known = [m['curve_label'] for m in position['members'] if m['curve_label']]
    if not known:
        return '--', '#F0F1F3'
    if position['tie_group_size'] == 1:
        return known[0], color_by_label[known[0]]
    # Exact exported tie group, including non-executable members, is preserved.
    return f"T{position['tie_group_size']}", '#687587'


def tie_definitions(cases):
    definitions = {}
    for case in cases:
        for ranking in case['rankings']:
            for p in ranking['positions']:
                if p['tie_group_size'] <= 1:
                    continue
                known = tuple(sorted((m['curve_label'] for m in p['members'] if m['curve_label'])))
                if known:
                    definitions[(p['tie_group_size'], known)] = p['tie_group_size'] - len(known)
    lines = []
    for (n, known), missing in sorted(definitions.items()):
        line = f"T{n}: {', '.join(known)}, each 1/{n} at that position"
        if missing:
            line += f'; {missing} other candidates lack curves'
        lines.append(line + '.')
    return lines


def draw_rank_mapping(fig, data, cases):
    methods = figure_methods(data)
    fig_text(fig, .12, 2.10, '(b) Top-3 recommendations',
             fontsize=8.2, fontweight='bold')
    ax = add_axis(fig, .12, .14, 5.22, 1.84)
    ax.set_xlim(0, 8.6); ax.set_ylim(0, 16.6); ax.axis('off')
    first, step = 3.0, .98
    color_by_label = {c['label']: curve_style(c, case['curves'])[0]
                      for case in cases for c in case['curves']}
    mappings = []
    for ci, case in enumerate(cases):
        center = first + (ci * 3 + 1) * step
        ax.text(center, 16.1, case['compound'].capitalize(), fontsize=7.3,
                fontweight='bold', ha='center', va='center')
        ax.plot([center - step * 1.42, center + step * 1.42], [15.62] * 2,
                color='#B5BDC6', lw=.5)
    ax.text(0, 15.05, 'Recommendation method', fontsize=6.8, va='center')
    for k in range(6):
        ax.text(first + k * step, 15.05, str(1 + k % 3), fontsize=7.0,
                ha='center', va='center')
    indexed = [{r['method_id']: r for r in c['rankings']} for c in cases]
    for row, (mid, name, group) in enumerate(methods):
        yy = 14 - row
        if row in (9, 12):
            ax.plot([0, 8.5], [yy + .52] * 2, color='#ADB6C0', lw=.5)
        ax.text(0, yy, name, fontsize=6.55 if len(name) > 22 else 6.8,
                va='center', fontweight='bold' if group == 'own' else 'normal')
        for ci, case in enumerate(cases):
            for p in indexed[ci][mid]['positions']:
                xx = first + (3 * ci + p['position'] - 1) * step
                txt, color = rank_cell(p, color_by_label)
                ax.add_patch(Rectangle((xx - step * .45, yy - .38), step * .90, .76,
                    facecolor=color, alpha=1 if txt == '--' else .15, edgecolor='none'))
                ax.text(xx, yy, txt, fontsize=7, fontweight='normal' if txt == '--' else 'bold',
                        va='center', ha='center', color=INK)
                mappings.append({'compound': case['compound'], 'method_id': mid,
                                 'position': p['position'], 'display': txt,
                                 'members': p['members'], 'tie_group_size': p['tie_group_size']})
    notes = ['Candidate IDs link curves and ranks. No unavailable positions are backfilled.']
    notes.extend(tie_definitions(cases))
    fig_text(fig, .12, .065, '--: unavailable curve; Tn: n-way tie.', fontsize=6.3)
    assert len(mappings) == 6 * len(methods)
    return mappings, notes


def layout_audit(fig):
    """Check all visible text bounds, text collisions and axis tick collisions."""
    issues = []
    if (SKILL / 'visual_qa.py').exists():
        spec = importlib.util.spec_from_file_location('scipilot_visual_qa', SKILL / 'visual_qa.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        issues.extend(module.audit_layout(fig))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always'); fig.canvas.draw()
    issues.extend(('FAIL', str(w.message)) for w in recorded if 'Glyph' in str(w.message))
    renderer = fig.canvas.get_renderer(); boxes = []
    undrawn_tick_ids = {id(tick.label1) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                       for tick in axis.get_major_ticks()
                       if not ax.axison or not (axis.get_view_interval().min()
                       <= tick.get_loc() <= axis.get_view_interval().max())}
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bb = text.get_window_extent(renderer)
        # LogLocator may create ticks beyond the axes. Matplotlib does not draw them.
        if id(text) in undrawn_tick_ids:
            continue
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
            issues.append(('FAIL', 'Text outside canvas: ' + text.get_text()))
        boxes.append((text, bb))
    collisions = []
    for i, (a, aa) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if min(aa.x1, bb.x1) - max(aa.x0, bb.x0) > .7 and min(aa.y1, bb.y1) - max(aa.y0, bb.y0) > .7:
                collisions.append([a.get_text(), b.get_text()])
    return {'issues': issues, 'text_collisions': collisions,
            'min_font_pt': min(t.get_fontsize() for t, _ in boxes),
            'visible_text_count': len(boxes), 'canvas_inches': list(fig.get_size_inches())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--outdir', type=Path, default=OUT)
    parser.add_argument('--existing-data-preview', action='store_true',
                        help='Layout test with the existing unmodified atorvastatin/rosuvastatin data.')
    args = parser.parse_args()
    data = load_figure_data() if args.existing_data_preview else json.loads(args.data.read_text())
    cases, profile = validated_cases(data)
    style(); fig = plt.figure(figsize=SIZE)
    bounds = draw_curves(fig, cases)
    mappings, notes = draw_rank_mapping(fig, data, cases)
    qa = layout_audit(fig)
    out = args.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    stem = 'layout_test_existing_data' if args.existing_data_preview else 'recommendation_curve_extension'
    base = out / stem
    audit = {'preview_only': True, 'data_path': None if args.existing_data_preview else str(args.data.resolve()),
        'source_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        'data_file_sha256': None if args.existing_data_preview else hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'unchanged_ranking_positions': True, 'no_backfilling': True,
        'all_exported_curves_displayed': True, 'profile': profile, 'axis_bounds': bounds,
        'mapping': mappings, 'notes': notes, 'layout': qa}
    caption = ('Candidate concentration curves and recommendation positions. '
        'Black points denote observed means; each panel shows one exported candidate '
        'over the complete positive-observation window. Candidate IDs link curves to '
        'the fourteen-method Top-3 map; recommendation position is not a GMFE rank. '
        'Lower GMFE indicates closer agreement. ' + ' '.join(notes))
    if any(c['compound'] == 'metformin' for c in cases):
        caption += (' The metformin reference dose is 250 mg hydrochloride '
                    '(195 mg active moiety in the source simulation).')
    base.with_suffix('.caption_draft.txt').write_text(caption + '\n')
    base.with_suffix('.audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    fig.savefig(base.with_suffix('.png'), dpi=300)
    with Image.open(base.with_suffix('.png')) as im:
        im.convert('L').save(out / (stem + '_grayscale.png'), dpi=(300, 300))
    assert not qa['issues'] and not qa['text_collisions'], qa
    assert qa['min_font_pt'] >= 6
    fig.savefig(base.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(base.with_suffix('.svg'), metadata={'Date': None})
    # Type-0 fonts keep their FontFile in a descendant, not on the parent font.
    from pypdf import PdfReader
    pdf = PdfReader(base.with_suffix('.pdf'))
    fonts = []
    for ref in pdf.pages[0]['/Resources']['/Font'].values():
        font = ref.get_object()
        for descendant in font.get('/DescendantFonts', [ref]):
            descriptor = descendant.get_object()['/FontDescriptor'].get_object()
            embedded = any(k in descriptor for k in ('/FontFile', '/FontFile2', '/FontFile3'))
            fonts.append({'name': str(font['/BaseFont']), 'embedded': embedded})
    assert all(f['embedded'] for f in fonts)
    audit['pdf_fonts'] = fonts
    audit['pdf_size_inches'] = [float(pdf.pages[0].mediabox.width) / 72,
                               float(pdf.pages[0].mediabox.height) / 72]
    assert np.allclose(audit['pdf_size_inches'], SIZE)
    base.with_suffix('.audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    plt.close(fig)
    print(json.dumps({'base': str(base), 'size_inches': SIZE, 'qa': qa,
                      'curves': sum(len(c['curves']) for c in cases),
                      'observations': sum(len(c['observation']['time_h']) for c in cases)}))


if __name__ == '__main__':
    main()
