"""Render all 34 Cmax scenarios and fourteen methods' exact first three positions.

Reads completed publication data only; no source outcomes, native runs, or new
statistical tests. Color is exp(E[absolute ln(P/O) | this position is scorable]),
not the error of an averaged concentration, and never best-of-Top-K.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/data/model_recommendation_execution.json'
CSV_SOURCE = ROOT / 'results/tables/model_recommendation_execution_rank_cases.csv'
EXPECTED_SOURCE_SHA = 'f9f84b7bf01ee13fafb13659f09da9ac616f410c86542b921e6173fc2c26eb08'
FIGURE = ROOT / 'results/figures/experiments/model_recommendation_rank_cases'
DATA = ROOT / 'results/data/model_recommendation_rank_figure.json'
LIGHT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
LIGHT_CASES = ROOT / 'results/data/pbpk_match_light_rank_cases.json'
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')
sys.path.insert(0, str(SKILL))
from profile_data import profile_data
from setup_style import setup_style
from visual_qa import audit_layout, render_preview
from check_figure import check_figure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    path = path.resolve()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': sha(path), 'bytes': path.stat().st_size}


def require(value, message):
    if not value:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def short_label(row):
    compound, sid = row['compound'], row['scenario_id']
    label = compound.capitalize()
    if compound == 'caffeine':
        label += ' 250 mg' if '_250mg' in sid else ' 50 mg'
    elif compound == 'metformin':
        label += ' fasted' if '_fasting' in sid else ' fed'
    elif compound == 'voriconazole':
        label += ' 4 mg/kg' if '_4mgkg' in sid else ' 6 mg/kg'
    return label


def embedded_pdf_fonts(path):
    from pypdf import PdfReader
    fonts = []
    for page in PdfReader(path).pages:
        for value in page['/Resources']['/Font'].get_object().values():
            font = value.get_object()
            children = font.get('/DescendantFonts', [value])
            for child in children:
                child = child.get_object()
                descriptor = child.get('/FontDescriptor')
                descriptor = descriptor.get_object() if descriptor else {}
                fonts.append({'font': str(font.get('/BaseFont')), 'subtype': str(child.get('/Subtype')),
                              'embedded': any(k in descriptor for k in ('/FontFile', '/FontFile2', '/FontFile3'))})
    require(fonts and all(x['embedded'] and x['subtype'] != '/Type3' for x in fonts), 'PDF font embedding failed')
    return fonts


def light_cases(scenarios):
    """Reproduce held-out ranks from scores and compare with the completed report."""
    sys.path.insert(0, str(ROOT / 'src'))
    from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
    from summarize_pbpk_match_light import load_inputs
    from build_model_execution_results import exact_position
    data = load_dataset()
    protocol, _, arrays, refs, _, _ = load_inputs(data)
    report_path = LIGHT / 'report/summary.json'
    completion = json.loads((LIGHT / 'report/completion.json').read_text())
    report_ref = next(x for x in completion['outputs'] if x['path'] == str(report_path.relative_to(ROOT)))
    require(sha(report_path) == report_ref['sha256'], 'Completed light report changed')
    report = json.loads(report_path.read_text())
    require(report['state'] == 'complete' and report['protocol_sha256'] == sha(LIGHT / 'protocol.json'),
            'Light report is not bound to the completed protocol')
    native_ref = next(x for x in report['input_refs'] if x['path'].endswith('followup_20261003/primary/results.json'))
    native_path = ROOT / native_ref['path']
    require(sha(native_path) == native_ref['sha256'], 'Frozen candidate execution results changed')
    native = json.loads(native_path.read_text())
    native_rows = {(r['scenario_id'], r['candidate_id']): r for r in native['candidate_results']}
    methods = [('pbpk_match_light', 'PBPK-Match', 'scores'),
               ('pbpk_match_light_document', 'PBPK-Match (document)', 'base_scores')]
    rows = []
    for method, _, array in methods:
        report_rows = {(r['scenario_id'], r['rank']): r for r in report['reports'][method]['execution']['position_rows']
                       if r['endpoint'] == 'Cmax_reference_vs_group'}
        require(len(report_rows) == len(scenarios) * 3, 'Own Cmax scenario denominator changed')
        for sid in scenarios:
            qi = protocol['query_ids'].index(sid)
            scores = arrays[array][0, qi]
            endpoints = {cid: native_rows[sid, cid]['reference_diagnostic']['endpoints']['Cmax_reference_vs_group']
                         for cid in protocol['candidate_ids']
                         if 'Cmax_reference_vs_group' in native_rows[sid, cid]['reference_diagnostic']['endpoints']}
            for position in (1, 2, 3):
                row = exact_position(scores, protocol['candidate_ids'], endpoints, position)
                saved = report_rows[sid, position]
                require(math.isclose(row['evaluable_probability'], saved['coverage'], abs_tol=1e-12),
                        'Own coverage differs from completed report')
                for key, saved_key in [('conditional_absolute_log_error', 'absolute_log_error'),
                                       ('conditional_signed_log_error', 'signed_log_error')]:
                    actual, expected = row[key], saved[saved_key]
                    require(actual is expected if actual is None or expected is None else
                            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                            'Own Cmax error differs from completed report')
                require(row['tie_group_size'] == saved['tie_size'], 'Own exact tie group differs')
                for member in row['members']:
                    candidate = native_rows[sid, member['candidate_id']]
                    member.update(family_id=candidate['family_id'], identity_match=candidate['identity_match'],
                        native_execution_status=candidate['reference_diagnostic']['execution_status'])
                    if not member['scorable']:
                        member['reason'] = 'endpoint_unavailable: ' + candidate['reference_diagnostic'].get('reason', 'unavailable')
                row.update(method_id=method, scenario_id=sid, compound=saved['compound'],
                    publication_cluster_id=saved['publication_cluster_id'], endpoint='Cmax_reference_vs_group',
                    stratum='reference_diagnostic')
                rows.append(row)
    payload = {'schema': 'pbpk-match-light-cmax-rank-cases-v1', 'state': 'complete', 'cases': rows,
        'methods': [{'method_id': m, 'method': name} for m, name, _ in methods],
        'validation': {'all_original_methods_unchanged': True, 'held_out_scores_verified': True,
                       'completed_report_reproduced': True, 'candidate_count': 280,
                       'scenario_count': len(scenarios), 'no_backfill': True},
        'sources': refs + [ref(report_path), ref(LIGHT / 'report/completion.json'), ref(native_path)]}
    LIGHT_CASES.write_text(json.dumps(clean(payload), ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False) + '\n')
    return rows, [(m, name) for m, name, _ in methods]


def load_data():
    require(sha(SOURCE) == EXPECTED_SOURCE_SHA, 'Approved rank-position publication data changed')
    source = json.loads(SOURCE.read_text())
    require(source['state'] == 'complete' and source['checks']['all_504_rank1_summaries_reproduce_frozen_metrics'],
            'Rank-position checks are incomplete')
    cases = [r for r in source['rank_positions']['cases']
             if r['endpoint'] == 'Cmax_reference_vs_group' and r['stratum'] == 'reference_diagnostic']
    by_key = {(r['method_id'], r['scenario_id'], r['position']): r for r in cases}
    require(len(cases) == len(by_key) == 34 * 12 * 3, 'Incomplete/duplicate figure scope')
    with CSV_SOURCE.open(newline='') as stream:
        csv_rows = [r for r in csv.DictReader(stream)
                    if r['endpoint'] == 'Cmax_reference_vs_group' and r['stratum'] == 'reference_diagnostic']
    csv_rows = [r for r in csv_rows if (r['method_id'], r['scenario_id'], int(r['position'])) in by_key]
    require(len(csv_rows) == len(cases), 'CSV scope differs')
    for row in csv_rows:
        saved = by_key[row['method_id'], row['scenario_id'], int(row['position'])]
        require(json.loads(row['members']) == saved['members'], 'CSV member identity/value differs')
        require(float(row['evaluable_probability']) == saved['evaluable_probability'], 'CSV coverage differs')
    methods = [(r['method_id'], r['method']) for r in source['main_cmax_table']]
    scenarios = sorted({r['scenario_id'] for r in cases})
    require(len(methods) == 12 and len(scenarios) == 34, 'Figure axes changed')
    extension, extra_methods = light_cases(scenarios)
    for row in extension:
        key = row['method_id'], row['scenario_id'], row['position']
        require(key not in by_key, 'New method overwrites an original method')
        by_key[key] = row
    methods.extend(extra_methods)
    rows, cells = [], []
    for sid in scenarios:
        first = by_key[methods[0][0], sid, 1]
        rows.append({'scenario_id': sid, 'compound': first['compound'], 'label': short_label(first),
                     'publication_cluster_id': first['publication_cluster_id']})
        for method, _ in methods:
            for position in (1, 2, 3):
                raw = by_key[method, sid, position]
                value = raw['conditional_absolute_log_error']
                probability = raw['evaluable_probability']
                color = math.exp(value) if value is not None else None
                require((color is None) == (probability == 0), 'Color availability contradicts coverage')
                require(color is None or math.isfinite(color) and color >= 1, 'Invalid conditional fold error')
                known = [m for m in raw['members'] if m['scorable']]
                require(len(known) / len(raw['members']) == probability, 'Tie coverage mismatch')
                if known:
                    error = math.fsum(abs(math.log(m['predicted']) - math.log(m['observed'])) for m in known) / len(known)
                    require(math.isclose(error, value, rel_tol=1e-12, abs_tol=1e-12), 'Cell error rederivation differs')
                cells.append({'method_id': method, 'scenario_id': sid, 'position': position,
                              'conditional_fold_error': color, 'coverage_probability': probability,
                              'tie_group_size': raw['tie_group_size'], 'fully_evaluable': raw['fully_evaluable'],
                              'partial_tie': 0 < probability < 1, 'members': [
                                  {k: v for k, v in m.items() if k in ('candidate_id', 'family_id', 'weight', 'scorable',
                                    'native_execution_status', 'identity_match', 'reason', 'predicted', 'observed')}
                                  for m in raw['members']]})
    require(len({r['label'] for r in rows}) == 34, 'Scenario display labels collide')
    return source, methods, rows, cells


def build():
    sources = [ref(SOURCE), ref(CSV_SOURCE), ref(Path(__file__))]
    source, methods, rows, cells = load_data()
    sources.append(ref(LIGHT_CASES))
    columns = len(methods) * 3
    frame = pd.DataFrame([{k: c[k] for k in ('method_id', 'position', 'conditional_fold_error',
                                           'coverage_probability', 'tie_group_size')} for c in cells])
    profile = profile_data(frame, group_cols=['method_id', 'position'])
    matrix = np.array([c['conditional_fold_error'] if c['conditional_fold_error'] is not None else np.nan
                       for c in cells]).reshape(34, columns)
    setup_style(journal='general', lang='en', use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.2,
                         'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'model-recommendation-top3-cmax-v1', 'hatch.linewidth': .35})
    fig = plt.figure(figsize=(5.5, 7.0), dpi=150)
    ax = fig.add_axes([.255, .15, .727, .65])
    masked = np.ma.masked_invalid(matrix)
    cmap = matplotlib.colormaps['cividis_r'].copy()
    cmap.set_bad('#dddddd')
    mesh = ax.pcolormesh(np.arange(columns + 1), np.arange(35), masked,
                         cmap=cmap, norm=LogNorm(vmin=1, vmax=64),
                         edgecolors='#ffffff', linewidth=.18, rasterized=False)
    ax.set_xlim(0, columns)
    ax.set_ylim(34, 0)
    ax.set_yticks(np.arange(34) + .5, [r['label'] for r in rows])
    ax.set_xticks(np.arange(columns) + .5, [str(p) for _ in methods for p in (1, 2, 3)])
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=3)
    for side in ax.spines.values():
        side.set_visible(False)
    for i, (method, name) in enumerate(methods):
        plot_name = name.replace(' (document)', ' (doc.)')
        ax.text(i * 3 + 1.5, 1.037, plot_name, transform=ax.get_xaxis_transform(),
                rotation=90, ha='center', va='bottom', fontsize=6.5, clip_on=False,
                fontweight='bold' if method.startswith('pbpk_match_light') else 'normal')
        if i:
            ax.axvline(i * 3, color='#666666', lw=.45, zorder=4)
    for i, cell in enumerate(cells):
        if cell['partial_tie']:
            ax.add_patch(Rectangle((i % columns, i // columns), 1, 1, fill=False,
                                   hatch='////', edgecolor='#111111', linewidth=0, zorder=5))
    # Optional three-way distinctions remain in the data; chart never invents values for unknowns.
    cax = fig.add_axes([.29, .076, .65, .017])
    bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[1, 2, 4, 8, 16, 32, 64])
    bar.ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    bar.set_label('Conditional geometric fold error (lower is better)', fontsize=7, labelpad=3)
    bar.outline.set_linewidth(.5)
    ax.legend(handles=[Patch(facecolor='#dddddd', edgecolor='none', label='No scorable candidate'),
                       Patch(facecolor='white', edgecolor='#111111', hatch='////', label='Partly scorable tie')],
              loc='upper center', bbox_to_anchor=(.5, -.015), ncol=2, frameon=False,
              fontsize=6.5, handlelength=1.25, handleheight=1.0, columnspacing=1.5)
    fig.text(.025, .978, 'Clinical Cmax scenarios', fontsize=7, ha='left', va='top')
    fig.text(.985, .984, 'Exact ranks 1 / 2 / 3', fontsize=7, ha='right', va='top')
    layout_issues = audit_layout(fig)
    require(not any(level in ('FAIL', 'WARN') for level, _ in layout_issues), 'Layout audit requires revision: ' + str(layout_issues))
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    require(min(t.get_fontsize() for t in texts) >= 6, 'Figure text below 6 pt')
    renderer = fig.canvas.get_renderer()
    clipped = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
               or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    require(not clipped, 'Text outside exact figure canvas: ' + str(clipped))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    png, pdf, svg = [FIGURE.with_suffix('.' + ext) for ext in ('png', 'pdf', 'svg')]
    # Use the skill preview renderer, then save at the required exact physical size.
    render_preview(fig, str(png), dpi=300)
    fig.savefig(png, dpi=300, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(svg, bbox_inches=None, metadata={'Date': None})
    font_report = embedded_pdf_fonts(pdf)
    checks = {}
    for path in (png, pdf, svg):
        issues, info = check_figure(str(path), min_dpi=300, target_inches=(5.5, 7.0))
        require(not any(level == 'FAIL' for level, _ in issues), 'Figure format check failed: ' + str(issues))
        checks[path.suffix[1:]] = {'issues': issues, 'info': info}
    with Image.open(png) as image:
        require(image.size == (1650, 2100), 'PNG has unexpected physical dimensions')
        gray = image.convert('L')
        gray_range = gray.getextrema()
        gray.save(ROOT / 'runtime/tmp/model_recommendation_rank_cases_grayscale.png', dpi=(300, 300))
    require(gray_range[1] - gray_range[0] >= 150, 'Insufficient grayscale contrast')
    plt.close(fig)
    require(sha(SOURCE) == EXPECTED_SOURCE_SHA, 'Source changed while rendering')
    for item in sources:
        require(sha(ROOT / item['path']) == item['sha256'], 'Publication input changed while rendering')
    metadata = {
        'schema_version': 'pbpk-top3-cmax-figure-v1', 'state': 'complete',
        'analysis_role': source['rank_positions']['analysis_role'], 'sources': sources,
        'skill_helpers': [ref(SKILL / name) for name in ('profile_data.py', 'setup_style.py', 'visual_qa.py', 'check_figure.py')],
        'definition': 'exp(mean absolute ln(P/O) over scorable candidates in the exact-score block containing the position)); every tie member has weight 1/block_size before conditioning.',
        'population_estimand': 'Locked reference-individual sampled Cmax versus reported clinical-group Cmax; not direct scientific validation.',
        'scope': {'methods': len(methods), 'requested_cmax_scenarios': 34, 'positions': [1, 2, 3], 'cells': len(cells),
                  'scorable_cells': sum(c['coverage_probability'] > 0 for c in cells),
                  'fully_scorable_cells': sum(c['fully_evaluable'] for c in cells),
                  'partial_tie_cells': sum(c['partial_tie'] for c in cells),
                  'unknown_cells': sum(c['coverage_probability'] == 0 for c in cells),
                  'scenarios_with_any_numeric_top3': len({c['scenario_id'] for c in cells if c['coverage_probability'] > 0})},
        'methods': [{'method_id': method, 'label': label} for method, label in methods],
        'scenarios': rows, 'cells': cells,
        'layout': {'size_inches': [5.5, 7.0], 'dpi': 300, 'color_scale': 'logarithmic',
                   'color_limits': [1, 64], 'actual_numeric_limits': [float(masked.min()), float(masked.max())],
                   'colormap': 'cividis_r', 'gray': 'no scorable candidate',
                   'hatch': '0 < evaluable probability < 1', 'rank_font_pt': 6.2,
                   'method_font_pt': 6.5, 'scenario_font_pt': 7},
        'data_profile': profile,
        'qa': {'layout_issues': layout_issues, 'all_text_within_canvas': True,
               'minimum_font_pt': min(t.get_fontsize() for t in texts), 'format_checks': checks,
               'pdf_fonts': font_report, 'grayscale_dynamic_range': gray_range,
               'all_cells_rederived_and_csv_crosschecked': True, 'extreme_values_clipped': False,
               'new_scientific_tests_or_model_runs': False, 'sealed_sources_opened': False,
               'visual_review': {'state': 'pending', 'preview_sha256': sha(png)}},
        'artifacts': [ref(path) for path in (png, pdf, svg)]}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(clean(metadata), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    return {'state': 'complete', 'scope': metadata['scope'], 'layout_issues': layout_issues,
            'artifacts': metadata['artifacts'], 'visual_review': metadata['qa']['visual_review']}


def review(preview_sha):
    """Attach the caller's visual-review attestation only to the exact checked preview."""
    metadata = json.loads(DATA.read_text())
    require(metadata['state'] == 'complete', 'No completed figure to review')
    for item in metadata['sources'] + metadata['artifacts']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Reviewed artifact/source changed')
    require(sha(FIGURE.with_suffix('.png')) == preview_sha == metadata['qa']['visual_review']['preview_sha256'],
            'Visual review refers to another preview')
    metadata['qa']['visual_review'] = {'state': 'passed', 'preview_sha256': preview_sha,
        'review': 'Rendered preview inspected at full resolution: labels readable, no clipping/overlap, unknown and partial-tie cells distinct, all methods/scenarios/extreme values retained.'}
    DATA.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    return {'state': 'visual_review_passed', 'metadata': ref(DATA)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-preview-sha256', help='Record completed visual inspection of this exact rendered PNG; do not regenerate.')
    args = parser.parse_args()
    print(json.dumps(clean(review(args.review_preview_sha256) if args.review_preview_sha256 else build()), ensure_ascii=False))
