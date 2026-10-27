#!/usr/bin/env python3
"""Plot clinical O and absolute predictions; curve GMFE is the only score.

Prefer additional_comparison.json when present; otherwise retain the original
two-method CSV report. The additional JSON contains methods[{id,label,information}]
and cases[{case_id,drug,observed_points,gmfe:{method_id:value},endpoints:{endpoint:
{observed,unit,source_unit,display_factor,predictions:{method_id:{value,weak_tail}}}}}].
All JSON quantities are ALREADY in display units: do not apply the factor again.
An unavailable prediction uses value=null and a nonempty reason; an unavailable
curve score uses gmfe[method_id]=null (optional gmfe_missing_reasons maps reasons).
An invalid existing JSON fails explicitly rather than falling back to old results.
Five methods are split across two vector-PDF pages, five drugs per page; the two
PNGs are previews of the same Figure 6 and its continuation. No models are run.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))

import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

SOURCE = ROOT / 'results/tables/pbpk_ours_vs_rf/comparison.csv'
ADDITIONAL = SOURCE.with_name('additional_comparison.json')
OUT = ROOT / 'results/figures/experiments/blueprint_ten_drug'
ENDPOINTS = ['auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss']
LABELS = ['AUC', '$C_{max}$', '$T_{max}$', '$C_{last}$', '$t_{1/2}$', '$AUC_\\infty$', 'CL', '$V_z$', 'MRT', '$V_{ss}$']
WIDTH_MM, HEIGHT_MM = 185, 222
# Restore the original teal palette; shade encodes row identity, never error.
ROW_COLORS = ['#f1f6f5', '#d9ebe7', '#b3d2ca']
EXPANDED_COLORS = ['#f1f6f5', '#e1efeb', '#cfe4df', '#b8d7ce', '#a2c9bf', '#8bb9af']
METHOD_ORDER = ['ours', 'rf', 'svr', 'xgboost', 'two_compartment']
SHORT_LABELS = {'ours': 'Ours', 'rf': 'RF', 'svr': 'SVR', 'xgboost': 'XGB',
                'two_compartment': '2-comp.'}
INK, HEADER = '#173e50', '#173e50'


def display_record(row):
    """Convert absolute values, retaining original per-study weight conventions."""
    factor = float(row['display_value_factor'])
    units = (row['unit'], row['display_unit'])
    conversion = {('ng/mL', 'mg/L'): .001, ('ng.h/mL', 'mg.h/L'): .001}
    assert factor == (1.0 if units[0] == units[1] else conversion[units]), units
    o = float(row['clinical_reference']) * factor if row['clinical_reference'] else None
    return {'observed': o, 'ours': float(row['ours_prediction']) * factor,
            'rf': float(row['rf_prediction']) * factor, 'unit': units[1], 'source_unit': units[0],
            'display_factor': factor, 'ours_weak': row['ours_weak_tail_estimate'] == 'True',
            'rf_weak': row['rf_weak_tail_estimate'] == 'True'}


def number(value):
    return '—' if value is None else np.format_float_positional(float(f'{value:.4g}'), trim='-')


def _finite(value, context, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{context}: expected a finite numeric value')
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'{context}: nonfinite or nonpositive value')
    return float(value)


def load_comparison():
    """Validate all values before plotting, including unchanged O/Ours/RF data."""
    rows = list(csv.DictReader(SOURCE.open()))
    drugs = list(dict.fromkeys(r['drug'] for r in rows))
    assert len(drugs) == 10 and len(rows) == 100
    baseline_cases = []
    for drug in drugs:
        by_endpoint = {r['endpoint']: r for r in rows if r['drug'] == drug}
        assert set(by_endpoint) == set(ENDPOINTS)
        endpoints = {}
        for ep in ENDPOINTS:
            d = display_record(by_endpoint[ep])
            endpoints[ep] = {k: d[k] for k in ['observed', 'unit', 'source_unit']}
            endpoints[ep]['display_factor'] = d['display_factor']
            endpoints[ep]['predictions'] = {
                m: {'value': d[m], 'weak_tail': d[m+'_weak']} for m in ['ours', 'rf']}
        baseline_cases.append({'case_id': by_endpoint['auc']['case_id'], 'drug': drug,
            'observed_points': int(by_endpoint['auc']['curve_scored_points']),
            'gmfe': {m: float(by_endpoint['auc'][m+'_curve_gmfe']) for m in ['ours', 'rf']},
            'endpoints': endpoints})
    if ADDITIONAL.exists():
        data = json.loads(ADDITIONAL.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('methods'), list) or not isinstance(data.get('cases'), list):
            raise ValueError('additional_comparison.json: expected methods and cases lists')
        methods, cases = data['methods'], data['cases']
        if [m.get('id') for m in methods] != METHOD_ORDER:
            raise ValueError(f'Expected methods in order {METHOD_ORDER}')
        source = ADDITIONAL
    else:
        methods = [{'id': 'ours', 'label': 'Ours', 'information': 'Separate development curve'},
                   {'id': 'rf', 'label': 'RF + PBPK', 'information': 'Structure-derived PK parameters'}]
        cases, source = baseline_cases, SOURCE
    ids = [m['id'] for m in methods]
    for m in methods:
        if not all(isinstance(m.get(k), str) and m[k].strip() for k in ['label', 'information']):
            raise ValueError(f'{m["id"]}: method label and information must be nonempty strings')
    if len(cases) != 10 or [c.get('case_id') for c in cases] != [c['case_id'] for c in baseline_cases]:
        raise ValueError('Expected the same ten clinical case IDs and order as the frozen CSV')
    for case, baseline in zip(cases, baseline_cases):
        context = case['case_id']
        if case.get('drug') != baseline['drug'] or case.get('observed_points') != baseline['observed_points']:
            raise ValueError(f'{context}: drug identity or clinical point count differs from CSV')
        if set(case.get('endpoints', {})) != set(ENDPOINTS) or set(case.get('gmfe', {})) != set(ids):
            raise ValueError(f'{context}: incomplete endpoint or GMFE keys')
        for m in ids:
            if case['gmfe'][m] is None:
                continue
            score = _finite(case['gmfe'][m], f'{context}/{m}/GMFE')
            if score < 1-1e-12:
                raise ValueError(f'{context}/{m}: GMFE cannot be below one')
        for ep, cell in case['endpoints'].items():
            original = baseline['endpoints'][ep]
            for key in ['unit', 'source_unit', 'display_factor', 'observed']:
                if cell.get(key) != original[key]:
                    raise ValueError(f'{context}/{ep}: clinical reference or display convention changed ({key})')
            if cell['observed'] is not None:
                _finite(cell['observed'], f'{context}/{ep}/O', allow_zero=ep == 'tmax')
            if set(cell.get('predictions', {})) != set(ids):
                raise ValueError(f'{context}/{ep}: missing method predictions')
            for m in ids:
                pred = cell['predictions'][m]
                if pred.get('value') is None:
                    if not isinstance(pred.get('reason'), str) or not pred['reason'].strip():
                        raise ValueError(f'{context}/{ep}/{m}: unavailable estimate needs a reason')
                else:
                    _finite(pred['value'], f'{context}/{ep}/{m}', allow_zero=ep == 'tmax')
                if not isinstance(pred.get('weak_tail'), bool):
                    raise ValueError(f'{context}/{ep}/{m}: weak_tail must be boolean')
            for m in ['ours', 'rf']:
                if any(cell['predictions'][m][k] != original['predictions'][m][k]
                       for k in ['value', 'weak_tail']):
                    raise ValueError(f'{context}/{ep}/{m}: original prediction or quality flag changed')
        for m in ['ours', 'rf']:
            if not np.isclose(case['gmfe'][m], baseline['gmfe'][m], rtol=1e-12, atol=1e-12):
                raise ValueError(f'{context}/{m}: original curve GMFE changed')
    return methods, cases, source


def draw_page(methods, cases, *, part, parts, start_index):
    expanded = len(methods) > 2
    ids = [m['id'] for m in methods]
    colors = EXPANDED_COLORS if expanded else ROW_COLORS
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
    # The wider row-label gutter fits "2-comp." without reducing the data font.
    ax = fig.add_axes([.097 if expanded else .080, .080 if expanded else .055,
                       .893 if expanded else .910, .860 if expanded else .885])
    header_h, observed_h, pred_h, gap_h = (1.6 if expanded else .85), 1.50, 1.00, .18
    block_h = header_h + observed_h + len(methods) * pred_h + gap_h
    total_h = len(cases) * block_h - gap_h
    ax.set_xlim(-.5, 9.5); ax.set_ylim(total_h, 0)
    ax.set_xticks(range(10), LABELS)
    ax.xaxis.tick_top(); ax.tick_params(axis='both', length=0, pad=6, labelcolor=INK)
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    for i, case in enumerate(cases):
        y = block_h*i
        left = -1.50 if expanded else -1.33
        ax.add_patch(Rectangle((left, y), 9.5-left, header_h, color=HEADER,
                               zorder=0, lw=0, clip_on=False))
        ax.text(left+.18, y+(.42 if expanded else header_h/2),
                f'{start_index+i+1:02d}  {case["drug"]}', ha='left', va='center',
                fontsize=7.5, weight='bold', color='white', clip_on=False)
        scores = '  |  '.join(SHORT_LABELS[m]+' '+('—' if case['gmfe'][m] is None
                                                else f"{case['gmfe'][m]:.2f}") for m in ids)
        ax.text(left+.18 if expanded else 9.30, y+(1.12 if expanded else header_h/2),
                'Curve GMFE   '+scores, ha='left' if expanded else 'right', va='center',
                fontsize=7.2, color='white', clip_on=False)
        row_start = y + header_h
        labels = ['O']+[SHORT_LABELS[m] for m in ids]
        for m, (label, height) in enumerate(zip(labels, [observed_h]+[pred_h]*len(methods))):
            center = row_start + height/2
            ax.add_patch(Rectangle((left, row_start), 9.5-left, height, color=colors[m],
                                   zorder=0, lw=0, clip_on=False))
            ax.text(-.64, center, label, ha='right', va='center', fontsize=7.3,
                    weight='bold', color=INK, clip_on=False)
            for j, ep in enumerate(ENDPOINTS):
                cell = case['endpoints'][ep]
                if m == 0:
                    ax.text(j, center-.29, number(cell['observed']), ha='center', va='center',
                            fontsize=7.3, weight='bold', color=INK)
                    # Units apply to both prediction rows even when O is absent.
                    ax.text(j, center+.33, cell['unit'].replace('.', '·'), ha='center', va='center',
                            fontsize=7.0, color='#50686d')
                else:
                    prediction = cell['predictions'][ids[m-1]]
                    value = number(prediction['value']) + ('†' if prediction['value'] is not None
                                                          and prediction['weak_tail'] else '')
                    ax.text(j, center, value, ha='center', va='center', fontsize=7.3, color=INK)
            for x in np.arange(-.5, 9.6, 1):
                ax.plot([x, x], [row_start, row_start+height], color='white', lw=.55, zorder=1)
            row_start += height
    if expanded:
        fig.text(.010, .983, 'O: clinical reference     Model rows: absolute predictions', fontsize=7.5, color=INK)
        fig.text(.99, .983, f'Part {part} / {parts}'+(' (continued)' if part > 1 else ''),
                 ha='right', fontsize=7, color=INK)
        fig.text(.010, .060, 'RF: random forest + PBPK; SVR: support vector regression + PBPK; XGB: XGBoost + PBPK.',
                 fontsize=7, color='#50686d')
        fig.text(.010, .043, '2-comp.: two-compartment PK. Different information sets are specified in the figure caption.',
                 fontsize=7, color='#50686d')
    else:
        fig.text(.010, .983, 'O: clinical reference     Ours / RF: absolute prediction', fontsize=7.5, color=INK)
    fig.text(.010, .026 if expanded else .034,
             'Units below O apply to all rows.  — Reference/estimate unavailable.  † Weak prediction NCA tail.',
             fontsize=7, color='#50686d')
    fig.text(.010, .009 if expanded else .017,
             'Teal shades identify rows; they do not encode accuracy. Curve GMFE is the only accuracy score.',
             fontsize=7, color='#50686d')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    clipped=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:clipped.append(t.get_text())
    assert not clipped,clipped
    overlaps=[]
    cell_boxes=[(t.get_text(),t.get_window_extent(renderer)) for t in ax.texts]
    for i,(label,box) in enumerate(cell_boxes):
        for other,other_box in cell_boxes[i+1:]:
            if box.overlaps(other_box): overlaps.append([label,other])
    assert not overlaps,overlaps
    assert not any('error' in t.get_text().lower() or ' AE' in t.get_text() or ' FE' in t.get_text() for t in texts)
    return fig, {'part': part, 'case_ids': [c['case_id'] for c in cases],
                 'minimum_font_pt': min(t.get_fontsize() for t in texts),
                 'clipped_text': clipped, 'overlapping_cell_text': overlaps}


def main():
    for font in (Path(matplotlib.get_data_path()) / 'fonts/ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    methods, cases, source = load_comparison()
    ids = [m['id'] for m in methods]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                         'svg.fonttype': 'path', 'axes.linewidth': .6,
                         'savefig.facecolor': 'white'})
    per_page = 5 if len(methods) > 2 else 10
    parts = (len(cases)+per_page-1)//per_page
    figures, page_audits = [], []
    for start in range(0, len(cases), per_page):
        fig, qa = draw_page(methods, cases[start:start+per_page],
                            part=start//per_page+1, parts=parts, start_index=start)
        figures.append(fig); page_audits.append(qa)
    # All pages must pass before any existing formal figure is overwritten.
    import pymupdf
    merged = pymupdf.open()
    for fig in figures:
        svg = io.BytesIO(); fig.savefig(svg, format='svg')
        pdf = cairosvg.svg2pdf(bytestring=svg.getvalue())
        with pymupdf.open(stream=pdf, filetype='pdf') as page:
            merged.insert_pdf(page)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'method_comparison'
    merged.save(target.with_suffix('.pdf')); merged.close()
    for index, fig in enumerate(figures):
        path = target.with_suffix('.png') if index == 0 else OUT/'method_comparison_continued.png'
        fig.savefig(path, dpi=300); plt.close(fig)
    if parts == 1:
        (OUT/'method_comparison_continued.png').unlink(missing_ok=True)
    # Preserve the established flattened audit values for existing Ours/RF consumers.
    cells, curves = {}, {}
    for case in cases:
        cells[case['drug']] = {}
        curves[case['drug']] = {'observed_points': case['observed_points'],
                               **{m+'_gmfe': case['gmfe'][m] for m in ids}}
        for ep, cell in case['endpoints'].items():
            cells[case['drug']][ep] = {k: cell[k] for k in ['observed', 'unit', 'source_unit', 'display_factor']}
            for m in ids:
                cells[case['drug']][ep].update({m: cell['predictions'][m]['value'],
                    m+'_weak': cell['predictions'][m]['weak_tail'],
                    m+'_missing_reason': cell['predictions'][m].get('reason', '')
                    if cell['predictions'][m]['value'] is None else ''})
    colors = EXPANDED_COLORS if len(methods) > 2 else ROW_COLORS
    audit={'source':str(source.relative_to(ROOT)), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'baseline_csv_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'methods':methods,
           'size_mm':[WIDTH_MM,HEIGHT_MM], 'display_mode':'observations_and_absolute_predictions',
           'scalar_prediction_basis':'Stored clinical-protocol prediction values; not FE, not terminal native P from Table 10. JSON already uses display units.',
           'observed_scalar_values':sum(c['observed'] is not None for d in cells.values() for c in d.values()),
           'scalar_predictions':sum(c[m] is not None for d in cells.values() for c in d.values() for m in ids),
           'attempted_scalar_predictions':len(cases)*len(ENDPOINTS)*len(methods),
           'missing_clinical_scalars':sum(c['observed'] is None for d in cells.values() for c in d.values()),
           'accuracy_metric':'curve_GMFE_only', 'scalar_error_cells':0,
           'gmfe_cells':sum(c['gmfe'][m] is not None for c in cases for m in ids),
           'attempted_gmfe_cells':len(cases)*len(methods),
           'clinical_points':sum(c['observed_points'] for c in cases), 'values_displayed':cells, 'curves':curves,
           'weak_prediction_cells':{m:sum(c[m] is not None and c[m+'_weak'] for d in cells.values() for c in d.values()) for m in ids},
           'unavailable_predictions':[{'case_id':c['case_id'], 'endpoint':ep, 'method':m,
                 'reason':cell['predictions'][m]['reason']}
                 for c in cases for ep,cell in c['endpoints'].items() for m in ids
                 if cell['predictions'][m]['value'] is None],
           'unavailable_curve_scores':[{'case_id':c['case_id'], 'method':m,
                 'reason':c.get('gmfe_missing_reasons',{}).get(m)}
                 for c in cases for m in ids if c['gmfe'][m] is None],
           'colors':'Original teal palette; categorical row backgrounds, no accuracy-dependent scalar colors.',
           'row_colors':dict(zip(['observed']+ids, colors)), 'header_color':HEADER,
           'pages':page_audits, 'page_count':parts,
           'minimum_font_pt':min(p['minimum_font_pt'] for p in page_audits),
           'clipped_text':[], 'overlapping_cell_text':[],
           'plotting_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'comparison_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['size_mm','display_mode','observed_scalar_values','scalar_predictions','gmfe_cells','scalar_error_cells','weak_prediction_cells','clipped_text']},indent=2))


if __name__ == '__main__': main()
