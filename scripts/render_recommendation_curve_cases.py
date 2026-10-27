"""Compact frozen candidate curves with the original pastel Top-3 model mapping.

Default: combined and mapping-only PNG previews. --export: a one-page PDF and SVGs.
No model execution, fitting, ranking changes or outcome-dependent selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from model_recommendation_paper import METHODS

DATA = ROOT / 'results/data/model_recommendation_curve_cases.json'
DATA_SHA = 'e5982a64918d2badd108790deb5f474343b8090f4ea340c2f029b6bc0a28a5ab'
EXTENSION = ROOT / 'results/data/pbpk_match_light_curves.json'
COMPONENT_RUN = ROOT / 'experiments/model_recommendation/component_integration_20261007'
COMPONENT_METHOD = ('pbpk_match_components', 'PBPK-Match + components', 'own')
OWN_METHODS = (('pbpk_match_light', 'PBPK-Match', 'own'),
               ('pbpk_match_light_document', 'PBPK-Match (document)', 'own'))
BASE = ROOT / 'results/figures/experiments/model_recommendation_curve_cases'
AUDIT = ROOT / 'results/data/model_recommendation_curve_figure_audit.json'
RANK_BASE = BASE.with_name(BASE.name + '_top3')
CURVE_HEIGHT, RANK_HEIGHT = 4.1, 2.9
SIZE = (5.5, CURVE_HEIGHT + RANK_HEIGHT)
COLORS = {'A1': '#0072B2', 'R1': '#0072B2', 'R2': '#D55E00',
          'R3': '#009E73', 'R4': '#CC79A7', 'R5': '#E69F00'}
BLUE, INK = '#216B8A', '#252F38'
PANEL_TITLE_X = .015


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'figure.constrained_layout.use': False,
        'savefig.facecolor': 'white',
    })


def validated_cases(data):
    cases = sorted(data['cases'], key=lambda c: c['compound'])
    assert [c['compound'] for c in cases] == ['atorvastatin', 'rosuvastatin']
    assert [len(c['curves']) for c in cases] == [1, 5]
    assert sum(len(c['observation']['time_h']) for c in cases) == 43
    for case in cases:
        ot = np.asarray(case['observation']['time_h'])
        oy = np.asarray(case['observation']['concentration_ng_ml'])
        assert len(ot) == len(oy) and np.all(np.diff(ot) > 0) and np.all(oy > 0)
        for curve in case['curves']:
            t, y = np.asarray(curve['time_h']), np.asarray(curve['concentration_ng_ml'])
            assert np.all(np.diff(t) > 0) and ot.min() >= t.min() and ot.max() <= t.max()
            error = np.log(np.interp(ot, t, y) / oy)
            assert np.isclose(np.exp(np.mean(np.abs(error))), curve['gmfe'], rtol=1e-10)
            assert np.isclose(np.sqrt(np.mean(error ** 2)), curve['log_rmse'], rtol=1e-10)
        curves = {c['label']: c for c in case['curves']}
        assert {r['method_id'] for r in case['rankings']} == {m[0] for m in figure_methods(data)}
        for ranking in case['rankings']:
            assert [p['position'] for p in ranking['positions']] == [1, 2, 3]
            for p in ranking['positions']:
                assert len(p['members']) == p['tie_group_size']
                assert np.isclose(sum(m['weight'] for m in p['members']), 1)
                known = [m for m in p['members'] if m['curve_label']]
                assert np.isclose(sum(m['weight'] for m in known), p['curve_coverage'])
                for m in known:
                    assert curves[m['curve_label']]['candidate_id'] == m['candidate_id']
    return cases


def figure_methods(data):
    if data.get('component_view'):
        return METHODS + (OWN_METHODS[0], COMPONENT_METHOD)
    return METHODS + (OWN_METHODS if data.get('own_methods') else ())


def component_view(extended):
    """Bind a publication-only view to completed, held-out component scores.

    Keep the frozen scientific exports intact. The document-only curve row is
    omitted only after verifying identical candidate identities and tie weights;
    no positions are backfilled and no curve or observation is refitted.
    """
    report_path = COMPONENT_RUN / 'report/summary.json'
    manifest_path = COMPONENT_RUN / 'predictions/manifest.json'
    scores_path = COMPONENT_RUN / 'predictions/component_scores.npz'
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert report['state'] == manifest['state'] == 'complete'
    assert report['validation']['all_280_candidates']
    assert report['validation']['baseline_metrics_reproduced']
    assert manifest['original_request_semantics_preserved']
    assert manifest['components_are_soft_requirements']
    for key in ('labels_used_as_features', 'observations_used_as_features',
                'weights_retrained', 'alpha_retuned', 'temperature_refitted'):
        assert manifest[key] is False, key
    refs = {r['path']: r for r in report['sources']}
    relevant = [r for r in report['sources'] if str(COMPONENT_RUN.relative_to(ROOT)) in r['path']]
    for ref in relevant:
        assert hashlib.sha256((ROOT / ref['path']).read_bytes()).hexdigest() == ref['sha256']
    assert str(manifest_path.relative_to(ROOT)) in refs
    assert str(scores_path.relative_to(ROOT)) in refs
    assert hashlib.sha256(scores_path.read_bytes()).hexdigest() == manifest['predictions_sha256']
    baseline_path = ROOT / 'experiments/model_recommendation/pbpk_match_light/protocol.json'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == manifest['baseline_protocol_sha256']
    protocol = json.loads(baseline_path.read_text())
    candidate_ids = manifest['candidate_ids']
    assert candidate_ids == protocol['candidate_ids'] and len(set(candidate_ids)) == 280
    rankings = {(r['method_id'], r['scenario_id']): r for r in report['curves']['rankings']}
    rows = {(r['method_id'], r['scenario_id'], r['position']): r
            for r in report['curves']['position_rows']}
    result = json.loads(json.dumps(extended))
    from pbpk_match_light_curves import curve_errors, score_position
    audit_cases = []
    with np.load(scores_path, allow_pickle=False) as arrays:
        assert arrays['scores'].shape == (len(manifest['query_ids']), 280)
        assert np.isfinite(arrays['scores']).all()
        for case in result['cases']:
            sid = case['scenario_id']
            old = {r['method_id']: r for r in case['rankings']}
            # Scores may differ, but the complete selected tie groups must not.
            identities = lambda r: [{k: v for k, v in p.items() if k != 'score'}
                                    for p in r['positions']]
            assert identities(old[OWN_METHODS[0][0]]) == identities(old[OWN_METHODS[1][0]])
            assert old[OWN_METHODS[0][0]]['positions'] == rankings[OWN_METHODS[0][0], sid]['positions']
            ranking = json.loads(json.dumps(rankings[COMPONENT_METHOD[0], sid]))
            ranking['method'] = COMPONENT_METHOD[1]
            qi = manifest['query_ids'].index(sid)
            global_qi = protocol['query_ids'].index(sid)
            assert int(arrays['indices'][qi]) == global_qi
            folds = [i for i, f in enumerate(protocol['folds']) if global_qi in f['outer_test']]
            assert len(folds) == 1 and folds[0] == old[OWN_METHODS[0][0]]['outer_fold']
            fold = protocol['folds'][folds[0]]
            assert all(protocol['groups'][i] != protocol['groups'][global_qi] for i in fold['outer_train'])
            scores = arrays['scores'][qi]
            metrics = {c['candidate_id']: curve_errors(case, c) for c in case['curves']}
            for position in ranking['positions']:
                score = np.sort(scores)[-position['position']]
                tied = np.flatnonzero(scores == score)
                assert score == position['score']
                assert {m['candidate_id'] for m in position['members']} == {candidate_ids[i] for i in tied}
                assert position['tie_group_size'] == len(tied)
                assert all(np.isclose(m['weight'], 1 / len(tied)) for m in position['members'])
                expected = rows[COMPONENT_METHOD[0], sid, position['position']]
                actual = score_position(case, ranking, position, metrics)
                for key in ('curve_gmfe', 'curve_log_rmse', 'curve_coverage', 'within_twofold_fraction'):
                    assert actual[key] == expected[key], (sid, position['position'], key)
            case['rankings'] = [r for r in case['rankings'] if r['method_id'] != OWN_METHODS[1][0]] + [ranking]
            audit_cases.append({'scenario_id': sid, 'outer_fold': folds[0],
                'component_input_changed': bool(arrays['changed'][qi]),
                'before': [rows[OWN_METHODS[0][0], sid, p] for p in (1, 2, 3)],
                'after': [rows[COMPONENT_METHOD[0], sid, p] for p in (1, 2, 3)]})
    result['component_view'] = {'schema': 'component-curve-publication-view-v1',
        'figure_sources': [{'path': str(p.relative_to(ROOT)),
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in (DATA, EXTENSION, Path(__file__).resolve())],
        'source_report': {'path': str(report_path.relative_to(ROOT)),
                          'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()},
        'source_bindings': relevant,
        'omitted_redundant_row': OWN_METHODS[1][0],
        'omitted_row_has_identical_candidates_and_tie_weights': True,
        'scientific_sources_modified': False, 'independent_publications': 1,
        'scenarios': 2, 'qualified_observations': 43,
        'displayed_methods': [m[0] for m in METHODS] + [OWN_METHODS[0][0], COMPONENT_METHOD[0]],
        'size_inches': list(SIZE),
        'cases': audit_cases}
    return result


def load_figure_data():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA, 'Frozen curve export changed'
    original = json.loads(DATA.read_text())
    if not EXTENSION.exists():
        return original
    extended = json.loads(EXTENSION.read_text())
    assert extended['source']['sha256'] == DATA_SHA
    assert extended['validation']['outer_held_out_predictions_verified'] is True
    for before, after in zip(original['cases'], extended['cases']):
        assert {k: v for k, v in before.items() if k != 'rankings'} == {k: v for k, v in after.items() if k != 'rankings'}
        assert after['rankings'][:-len(OWN_METHODS)] == before['rankings']
    # Bind the figure to actual frozen scores; rendering never scores a model.
    for ref in extended['provenance']:
        if ref['path'].endswith(('predictions/seed17.npz', 'pbpk_match_light/protocol.json')):
            assert hashlib.sha256((ROOT / ref['path']).read_bytes()).hexdigest() == ref['sha256']
    return component_view(extended) if (COMPONENT_RUN / 'report/summary.json').exists() else extended


def title(curve):
    label = curve['label']
    if label == 'A1':
        return 'A1  Atorvastatin'
    if label in ('R1', 'R4'):
        return f"{label}  Rosuvastatin [{curve['version'][:7]}]"
    return f"{label}  {curve['name'].split('-')[0]}–rosuvastatin"


def coordinates(box):
    left, bottom, width, height = box
    return (lambda x, y: (left + x * width, bottom + y * height),
            lambda x, y, w, h: [left + x * width, bottom + y * height, w * width, h * height])


def draw_curves(fig, data, box):
    """Compact six-panel display; keep the complete window of every candidate."""
    cases = validated_cases(data)
    point, rect = coordinates(box)
    fig.text(*point(PANEL_TITLE_X, .985), 'a  Individual PBPK predictions', fontsize=8.2,
             fontweight='bold', color=INK, ha='left', va='top')
    handles = [Line2D([], [], color='#606D78', lw=1.3, label='Prediction'),
               Line2D([], [], color=INK, marker='o', linestyle='none', markersize=3.2,
                      label='Observed mean')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=point(.57, .99),
               borderaxespad=0, frameon=False, ncol=2, fontsize=6.2,
               handlelength=1.3, handletextpad=.4, columnspacing=.8)
    panels = [(case, curve) for case in cases for curve in case['curves']]
    for i, (case, curve) in enumerate(panels):
        row, col = divmod(i, 2)
        ax = fig.add_axes(rect(.11 + .485 * col, .685 - .30 * row, .365, .16))
        ot = np.asarray(case['observation']['time_h'])
        oy = np.asarray(case['observation']['concentration_ng_ml'])
        t, y = np.asarray(curve['time_h']), np.asarray(curve['concentration_ng_ml'])
        tx = np.unique(np.r_[ot.min(), t[(t >= ot.min()) & (t <= ot.max())], ot.max()])
        cy = np.interp(tx, t, y)
        assert np.all(cy > 0)
        color = COLORS[curve['label']]
        ax.plot(tx, cy, color=color, lw=1.3, zorder=2)
        ax.scatter(ot, oy, s=13, facecolors=INK, edgecolors='white', linewidths=.3, zorder=3)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(ot.min() / 1.15, ot.max() * 1.12)
        ax.set_ylim((.0025, 40) if case['compound'] == 'atorvastatin' else (.014, 40))
        assert min(oy.min(), cy.min()) > ax.get_ylim()[0]
        assert max(oy.max(), cy.max()) < ax.get_ylim()[1]
        ticks = [ot.min(), 2, 12, ot.max()]
        ax.set_xticks(ticks, [f'{x:g}' for x in ticks])
        ax.tick_params(axis='both', labelsize=6.5, pad=2, length=2.5)
        ax.minorticks_off()
        ax.grid(axis='y', color='#E3E8ED', lw=.4)
        ax.set_title(title(curve), loc='left', fontsize=6.8, fontweight='bold', pad=16, color=INK)
        ax.text(0, 1.045, f"{case['dose_mg']:g} mg  ·  GMFE {curve['gmfe']:.3f}",
                transform=ax.transAxes, fontsize=6.8, color=color, va='bottom')
    fig.text(*point(.012, .51), 'Plasma concentration (ng/mL; log scale)',
             rotation=90, va='center', fontsize=7, color=INK)
    fig.text(*point(.54, .012), 'Time after dose (h; log scale)',
             ha='center', fontsize=6.8, color=INK)


def rank_cell(position):
    """Restore the original candidate-ID mapping, including unresolved tie groups."""
    known = [m['curve_label'] for m in position['members'] if m['curve_label']]
    if not known:
        return '--', '#F0F1F3'
    if position['tie_group_size'] == 1:
        return known[0], COLORS[known[0]]
    if position['tie_group_size'] == 2:
        assert set(known) == {'R1', 'R5'}
    elif position['tie_group_size'] == 20:
        assert set(known) == {'R1', 'R2', 'R3', 'R5'}
    else:
        raise ValueError('Undocumented curve-ranking tie')
    return f"T{position['tie_group_size']}", '#687587'


def draw_rank_mapping(fig, data, box, *, show_notes=True):
    """The original pastel candidate-code grid, compact enough to share a page."""
    cases = validated_cases(data)
    indexed = [{r['method_id']: r for r in c['rankings']} for c in cases]
    point, rect = coordinates(box)
    methods = figure_methods(data)
    n = len(methods)
    fig.text(*point(PANEL_TITLE_X, .99), 'b  Models at the first three recommendation positions',
             fontsize=8.2, fontweight='bold', ha='left', va='top', color=INK)
    ax = fig.add_axes(rect(.015, .20, .97, .73))
    ax.set_xlim(0, 8); ax.set_ylim(-.15, n + 2.5); ax.axis('off')
    first_x, step = 2.90, .84
    for g, case in enumerate(cases):
        center = first_x + (3 * g + 1) * step
        ax.text(center, n + 2.00, case['compound'].capitalize(), ha='center',
                va='center', fontsize=7.5, fontweight='bold')
        ax.plot([center - step * 1.45, center + step * 1.45], [n + 1.57, n + 1.57],
                lw=.5, color='#B5BDC6')
    ax.text(.06, n + .97, 'Recommendation method', fontsize=7, va='center')
    for k in range(6):
        ax.text(first_x + k * step, n + .97, f'Rank {1 + k % 3}',
                ha='center', va='center', fontsize=6.7)
    used = 0
    for row, (mid, name, group) in enumerate(methods):
        yy = n - row
        if row in (9, 12):
            ax.plot([.06, 7.77], [yy + .52, yy + .52], color='#A3ADB8', lw=.6)
        ax.text(.06, yy, name, fontsize=6.4 if mid == COMPONENT_METHOD[0] else 6.8,
                va='center', fontweight='bold' if group == 'own' else 'normal')
        for ci in range(2):
            for p in indexed[ci][mid]['positions']:
                used += 1
                xx = first_x + (3 * ci + p['position'] - 1) * step
                text, color = rank_cell(p)
                ax.add_patch(Rectangle((xx - step * .46, yy - .38), step * .92, .76,
                                      facecolor=color, alpha=1 if text == '--' else .15,
                                      edgecolor='none'))
                ax.text(xx, yy, text, ha='center', va='center', fontsize=7.2,
                        fontweight='normal' if text == '--' else 'bold', color='#283442')
    assert used == 6 * n
    if show_notes:
        fig.text(*point(.02, .143), 'A1 / R1–R5: candidate IDs.  --: no evaluable curve.',
                 fontsize=6.5, va='top')
        fig.text(*point(.02, .095), 'T2: R1 and R5, each with probability 1/2 at that position.',
                 fontsize=6.5, va='top')
        fig.text(*point(.02, .047), 'T20: R1, R2, R3 and R5, each 1/20; 16 other candidates have no matched curve.',
                 fontsize=6.5, va='top')


def make_figure(data, *, caption_notes=False):
    style()
    # Remove the note area when definitions are supplied by the manuscript caption.
    # Keep all remaining content at its original physical size and position from the top.
    trim = .20 * RANK_HEIGHT if caption_notes else 0
    height = SIZE[1] - trim
    fig = plt.figure(figsize=(SIZE[0], height))
    # Render at final physical sizes, rather than scaling finished figures in LaTeX.
    draw_curves(fig, data, (0, (RANK_HEIGHT - trim) / height, 1, CURVE_HEIGHT / height))
    draw_rank_mapping(fig, data, (0, -trim / height, 1, RANK_HEIGHT / height),
                      show_notes=not caption_notes)
    return fig


def make_rank_figure(data):
    style()
    fig = plt.figure(figsize=(SIZE[0], RANK_HEIGHT))
    draw_rank_mapping(fig, data, (0, 0, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', help='Also export one-page PDF and SVGs.')
    args = parser.parse_args()
    data = load_figure_data()
    figures = [(make_figure(data), BASE), (make_rank_figure(data), RANK_BASE)]
    BASE.parent.mkdir(parents=True, exist_ok=True)
    for fig, base in figures:
        fig.savefig(base.with_suffix('.png'), dpi=300)
        gray = ROOT / 'runtime/tmp' / (base.name + '_grayscale.png')
        gray.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(base.with_suffix('.png')) as image:
            image.convert('L').save(gray, dpi=(300, 300))
    if args.export:
        figures[0][0].savefig(BASE.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        for fig, base in figures:
            fig.savefig(base.with_suffix('.svg'), metadata={'Date': None})
    if data.get('component_view'):
        AUDIT.write_text(json.dumps(data['component_view'], ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')
    print(json.dumps({'previews': [str(base.with_suffix('.png').relative_to(ROOT)) for _, base in figures],
                      'size_inches': SIZE, 'data_sha256': DATA_SHA, 'exported': args.export,
                      'curve_count': 6, 'unique_observed_points': 43,
                      'rank_position_groups': 6 * len(figure_methods(data)),
                      'extension_sha256': hashlib.sha256(EXTENSION.read_bytes()).hexdigest() if EXTENSION.exists() else None}))
    for fig, _ in figures:
        plt.close(fig)


if __name__ == '__main__':
    main()
