#!/usr/bin/env python3
"""Publish the sealed ten-drug vascular--tissue experiment without fitting.

Only five publication files and one manifest are written under RUN/paper.
All model predictions, formulas, source references and endpoint rules are
checked before rendering. Legacy results and renderers remain read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))
import numpy as np

import render_clinical_method_figure as matrix_plot
import render_current_clinical_table as table_helpers
import render_pbpk_method_comparison as clinical_helpers
import render_ten_drug_curves as curve_helpers
import render_ten_drug_table as scenario_helpers
from render_blueprint_architecture_paper import pinned_references, require_completed, same_number, json_plain
from report_blueprint_validation import Store, _case_reference, _score_prediction

VERSION = 'vascular-tissue-ten-drug-publication-1'
DEFAULT = ROOT / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
BASELINES = 'results/tables/pbpk_ours_vs_rf/additional_comparison.json'
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill')
METHOD = 'flow_exchange'
EPS = matrix_plot.ENDPOINTS


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def exact_arrays(left, right, label):
    if np.shape(left) != np.shape(right) or not np.allclose(left, right, rtol=1e-9, atol=1e-12):
        raise ValueError('Numerical data disagree: ' + label)


def checked_complete_matrix(protocol, state, result, choices):
    cases = {c['case_id']: c for c in protocol['cases']}
    if len(cases) != 10 or len({c['drug'] for c in cases.values()}) != 10:
        raise ValueError('All ten unique drugs are required')
    methods = {METHOD, 'two_compartment', 'two_compartment_prior'}
    expected = {(cid, s, m) for cid in cases for s in ('full', 'five') for m in methods}
    rows = {(u['case_id'], u['sampling'], u['method']): u for u in result['units']}
    if len(rows) != 60 or set(rows) != expected or set(protocol['methods']) != methods:
        raise ValueError('Incomplete ten-drug, two-sampling, three-method matrix')
    if set(state['computations']) != {u['unit_id'] for u in rows.values()}:
        raise ValueError('Locked attempt set differs from results')
    if choices.get('basis') != 'development_objective_only':
        raise ValueError('Formula selection must use development outcomes only')
    for key, row in rows.items():
        config = cases[key[0]]
        attempt = state['computations'][row['unit_id']]
        if any(row[k] != config[k] for k in ('drug', 'study_id', 'cluster_id')):
            raise ValueError('Drug/study identity changed')
        if row['attempt_status'] != attempt['status']:
            raise ValueError('Prediction completion status differs')
        if attempt['status'] == 'predicted' and row['prediction_source'] != attempt['prediction']:
            raise ValueError('Scored prediction differs from lock')
        if key[2] == METHOD and attempt['status'] == 'predicted':
            choice = choices['choices'][row['unit_id']]
            if choice['mapping'] != attempt['mapping'] or choice['formula'] != attempt['selected_formula']:
                raise ValueError('Formula differs from development selection')
    return cases, rows


class Data:
    def __init__(self, run):
        self.store = store = Store(ROOT)
        self.run = store.path(str(run))
        self.state = store.read(str(self.run / 'state.json'))
        require_completed(self.state)
        self.protocol = store.ref(self.state['protocol'])
        self.result = store.ref(self.state['results'])
        if any(x.get('version') != self.protocol.get('version') for x in (self.state, self.result)):
            raise ValueError('Experiment versions differ')
        if self.result.get('all_attempts_locked_before_scoring') is not True:
            raise ValueError('Results must retain the prediction-lock declaration')
        for key, filename in (('protocol', 'protocol.json'), ('results', 'results.json')):
            if store.path(self.state[key]['path']) != self.run / filename:
                raise ValueError('State points outside the selected run: ' + key)
        choices = store.ref(self.state['formula_selection'])
        for item in (self.protocol, self.state, self.result, choices):
            pinned_references(item, store)
        self.cases, self.units = checked_complete_matrix(self.protocol, self.state, self.result, choices)
        self.aux = store.ref(self.result['auxiliary_source'])
        if self.aux != self.result['auxiliary'] or self.aux.get('all_29_attempts_locked_before_scoring') is not True:
            raise ValueError('Auxiliary predictions must be sealed before scoring')
        pinned_references(self.aux, store)
        self.references = {cid: _case_reference(c, store) for cid, c in self.cases.items()}
        self.scored = {}
        for key, row in self.units.items():
            if row['attempt_status'] != 'predicted':
                continue
            cid = key[0]
            clinical, observed, endpoints = self.references[cid]
            computed, _, _ = _score_prediction(self.cases[cid], clinical, observed, endpoints,
                self.state['computations'][row['unit_id']], store)
            for name in ('gmfe', 'ln_rmse', 'n'):
                same_number(computed['curve_metrics'].get(name), row['curve_metrics'].get(name), name)
            for ep in EPS:
                for name in ('value', 'error', 'reference'):
                    same_number(computed['endpoints'][ep][name], row['endpoints'][ep][name], ep + '/' + name)
                for name in ('unit', 'weak_tail', 'error_metric'):
                    if computed['endpoints'][ep][name] != row['endpoints'][ep][name]:
                        raise ValueError('Endpoint rule differs: ' + ep + '/' + name)
            self.scored[key] = computed
        self.baseline = store.read(BASELINES)
        pinned_references(self.baseline, store)
        self.baseline_cases = {c['case_id']: c for c in self.baseline['cases']}
        if set(self.baseline_cases) != set(self.cases):
            raise ValueError('Background cohort changed')
        self.panels, self.profile_rows, self.table_cases, self.table_rows = [], [], [], []
        self.comparison = []
        self.prepare()
        self.scenarios = self.prepare_scenarios()
        store.checksum(str(Path(__file__)))
        for module in (matrix_plot, table_helpers, clinical_helpers, curve_helpers, scenario_helpers):
            store.checksum(module.__file__)
        store.unchanged()

    def attempt(self, cid, sampling='full', method=METHOD):
        row = self.units[cid, sampling, method]
        attempt = self.state['computations'][row['unit_id']]
        if attempt['status'] != 'predicted':
            raise ValueError('Publication requires this prediction; retain failed result separately: ' + row['unit_id'])
        return row, attempt

    def dense(self, attempt):
        p = self.store.path(attempt['prediction']['path']).parent / 'endpoints.json'
        # Completion receipts bind endpoint files even if state stores only the curve.
        receipt = self.store.read(str(p.parent / 'completion.json'))
        pinned_references(receipt, self.store)
        pins = receipt['artifact_pins']
        for name, expected in pins.items():
            self.store.checksum(name, expected)
        dense = self.store.read(str(p), pins[relative(p)])
        if dense.get('endpoint_estimator_uses_observations') is not False:
            raise ValueError('Dense endpoints must exclude target observations')
        return dense

    def prepare(self):
        for cid, case in self.cases.items():
            row, attempt = self.attempt(cid)
            uid = row['unit_id']
            development = self.aux['development'][uid]
            if development['mapping'] != attempt['mapping']:
                raise ValueError('Development and target mappings differ')
            mapping = self.store.ref(attempt['mapping'])
            if mapping['version'] != 'physiological-flow-exchange-calibration-1' or mapping['variant'] != METHOD:
                raise ValueError('Unexpected adopted model algorithm')
            dev = self.store.ref(case['sampling_sources']['full'])
            if dev['study_id'] == case['study_id']:
                raise ValueError('Development and target studies coincide')
            req = self.store.ref(attempt['request'])
            devreq = self.store.ref(development['request'])
            if devreq['human'] != dev['human'] or devreq['dosing'] != dev['dosing']:
                raise ValueError('Development simulation uses different human/dosing inputs')
            mw = self.store.read(case['physchem'])['identity']['molecular_weight_g_mol']
            clinical, observed, refs = self.references[cid]
            devcurve = self.store.ref(development['prediction'])
            targetcurve = self.store.ref(attempt['prediction'])
            d = curve_helpers.series(devcurve, [o['time_min']/60 for o in dev['observations']],
                [o['concentration_umol_per_l']*mw/1000 for o in dev['observations']], mw)
            if observed['unit'] != 'ng/mL':
                raise ValueError('Clinical concentration must retain ng/mL source unit')
            e = curve_helpers.series(targetcurve, observed['time_h'], np.asarray(observed['concentration'])/1000, mw)
            same_number(e['gmfe'], row['curve_metrics']['gmfe'], 'Figure 5 clinical GMFE')
            infos = []
            for label, series, request in (('Development', d, devreq), ('Evaluation', e, req)):
                infos.append({'stage': label, 'dosing': request['dosing'], 'human_model_inputs': request['human'],
                    'observed_points': len(series['time']), 'gmfe': series['gmfe'], 'ln_rmse': series['ln_rmse']})
                self.profile_rows.extend({'drug': case['drug'], 'stage': label, 'time_h': float(t),
                    'observed_mg_l': float(o), 'predicted_mg_l': float(p)}
                    for t, o, p in zip(series['time'], series['observed'], series['sampled_prediction']))
            self.panels.append({'drug': case['drug'], 'case_id': cid, 'series': [d, e], 'detail': {'series': infos}})
            dense = self.dense(attempt)
            table_helpers.scalar_native_audit(dense, targetcurve, mw, req['dosing']['dose_mg'])
            table_case = {'case_id': cid, 'drug': case['drug'], 'curve_metrics': {'ours': row['curve_metrics']},
                'native_context': self.context(case, req, mapping, dense),
                'missing_clinical_endpoints': [], 'native_missing': [], 'endpoint_protocol': case['nca_policy'],
                'clinical_reference_quality_notes': clinical.get('clinical_reference_quality_notes') or [],
                'approximations': clinical.get('approximations') or [],
                'citation': clinical.get('citation') or case['study_id']}
            base = deepcopy(self.baseline_cases[cid])
            scored = {'ours': self.scored[cid, 'full', METHOD],
                      'two_compartment': self.scored[cid, 'full', 'two_compartment']}
            for method in ('rf', 'svr', 'xgboost'):
                # This request supplies only the matched dose, weight and infusion
                # for scoring; no prediction is generated or altered here.
                comparison_attempt = {'prediction': base['prediction_sources'][method], 'request': attempt['request']}
                scored[method], _, _ = _score_prediction(case, clinical, observed, refs, comparison_attempt, self.store)
                same_number(scored[method]['curve_metrics']['gmfe'], base['gmfe'][method], 'Frozen ' + method + ' GMFE')
            base['gmfe'] = {m: scored[m]['curve_metrics']['gmfe'] for m in matrix_plot.METHOD_ORDER}
            for ep in EPS:
                endpoint = row['endpoints'][ep]
                cell = base['endpoints'][ep]
                factor = cell['display_factor']
                observed_value = refs[ep]['value']
                same_number(None if observed_value is None else observed_value*factor, cell['observed'], 'Clinical O')
                for method in matrix_plot.METHOD_ORDER:
                    fresh = scored[method]['endpoints'][ep]
                    if fresh['unit'] != cell['source_unit']:
                        raise ValueError('Frozen display unit differs from common clinical scoring')
                    value = None if fresh['value'] is None else fresh['value']*factor
                    if method in ('rf', 'svr', 'xgboost'):
                        same_number(value, cell['predictions'][method]['value'], 'Frozen ' + method + '/' + ep)
                        if fresh['weak_tail'] != cell['predictions'][method]['weak_tail']:
                            raise ValueError('Frozen background NCA quality differs')
                    cell['predictions'][method] = {'value': value, 'weak_tail': fresh['weak_tail'], 'reason': fresh['reason']}
                if observed_value is None:
                    table_case['missing_clinical_endpoints'].append(ep)
                    continue
                value, _, reason = table_helpers.native_value(dense['endpoints'][ep], ep, endpoint['unit'], req['human']['weight_kg'])
                if value is None:
                    table_case['native_missing'].append(ep)
                self.table_rows.append({'case_id': cid, 'endpoint': ep, 'endpoint_label': clinical_helpers.ENDPOINTS[ep],
                    'display_unit': cell['unit'], 'display_value_factor': factor, 'predicted_value': value,
                    'evaluation_prediction': endpoint['value'], 'observed_value': observed_value,
                    'matched_weak_tail': endpoint['weak_tail'], 'fold_error': endpoint['error'] if ep != 'tmax' else None,
                    'absolute_error_h': endpoint['error'] if ep == 'tmax' else None, 'dense_missing_reason': reason})
            self.table_cases.append(table_case)
            self.comparison.append(base)
        if len(self.table_rows) != 80:
            raise ValueError('Original scalar reference count changed from 80')

    @staticmethod
    def context(case, req, mapping, dense):
        d, h = req['dosing'], req['human']
        window = dense['main_window']
        return (f"Dose {d['dose_mg']:.5g} mg; infusion {d['infusion_minutes']:g} min; "
                f"{h['sex'][0]} {h['age_years']:g} y, {h['weight_kg']:g} kg, {h['height_cm']:g} cm; "
                f"P window {window['start_h']:g}-{window['end_h']:g} h; "
                f"{mapping['construction_partition_method']}; development-selected parameters, target prediction.")

    def prepare_scenarios(self):
        scenarios = self.aux['input_response']
        if len(scenarios) != 9:
            raise ValueError('Exactly nine prediction-only input scenarios are required')
        row, target = self.attempt(scenario_helpers.CASE_ID)
        rows = []
        seen = set()
        for cid, record in scenarios.items():
            if record['mapping'] != target['mapping']:
                raise ValueError('Input-response scenario uses another mapping')
            request, dense = self.store.ref(record['request']), self.store.ref(record['endpoints'])
            curve = self.store.ref(record['prediction'])
            if request['compound_name'] != 'Treprostinil':
                raise ValueError('Wrong input-response compound')
            mw = next(e['value'] for e in request['evidence'] if e['parameter'] == 'molecular_weight')
            table_helpers.scalar_native_audit(dense, curve, mw, request['dosing']['dose_mg'])
            key = (record['dose_multiplier'], record['condition'])
            seen.add(key)
            if key == (1., 'C1'):
                clinical_curve = self.store.ref(target['prediction'])
                for field in ('time_min', 'concentration'):
                    exact_arrays(curve[field], clinical_curve[field], 'Repeated Treprostinil reference input')
            rows.append({'case_id': cid, **record, 'request_data': request, 'dense': dense})
        if seen != {(d, c) for d in (.5, 1., 2.) for c in ('C1', 'C2', 'C3')}:
            raise ValueError('Input-response dose and condition matrix changed')
        return sorted(rows, key=lambda r: (r['dose_multiplier'], r['condition']))


def figure_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    sys.path.insert(0, str(SKILL / 'scripts'))
    from setup_style import setup_style
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    if not fonts.resolve().is_relative_to(ROOT):
        raise ValueError('Use project-local plotting dependencies')
    for font in fonts.glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    setup_style(journal='general', lang='en', use_sciplots=False)
    plt.rcParams.update({'font.family': ['DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.linewidth': .5,
        'lines.linewidth': .9, 'text.color': '#253447', 'axes.labelcolor': '#253447',
        'axes.edgecolor': '#798391', 'xtick.color': '#475467', 'ytick.color': '#475467',
        'svg.fonttype': 'path', 'figure.constrained_layout.use': False,
        'axes.unicode_minus': False, 'savefig.bbox': None})


def draw_curves(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, FixedLocator, NullFormatter, FuncFormatter
    from visual_qa import audit_layout
    fig = plt.figure(figsize=(185/25.4, 215/25.4), facecolor='white')
    colors, styles, markers = curve_helpers.COLORS, curve_helpers.STYLES, curve_helpers.MARKERS
    handles = [Line2D([], [], color=c, linestyle=s, marker=m, markersize=3.6,
        markerfacecolor='white', markeredgewidth=.75, linewidth=.9) for c, s, m in zip(colors, styles, markers)]
    fig.legend(handles, ['Development: fit + observations', 'Evaluation: prediction + observations'],
        loc='upper left', bbox_to_anchor=(.076, 1.005), frameon=False, ncol=2, columnspacing=1.8, handlelength=2.4)
    fig.text(.015, .51, 'Concentration (mg/L, log scale)', rotation=90, va='center', fontsize=7.5)
    fig.text(.535, .014, 'Time after dose (h, log scale)', ha='center', fontsize=7.5)
    for k, panel in enumerate(data.panels):
        row, col = divmod(k, 2)
        left, top, width = .09+col*.48, .955-row*.184, .395
        fig.text(left, top, f"{chr(65+k)}  {panel['drug']}" + ('*' if panel['drug'] == 'Ceftriaxone' else ''),
            fontsize=8.5, fontweight='bold', va='top')
        for i, info in enumerate(panel['detail']['series']):
            dosing = info['dosing']
            label = 'Dev' if i == 0 else 'Eval'
            fig.text(left, top-.017-i*.013, f"{label}  {dosing['dose_mg']:.5g} mg / {dosing['infusion_minutes']:g} min",
                va='top', fontsize=7, color=colors[i])
            fig.text(left+width, top-.017-i*.013, f"GMFE {panel['series'][i]['gmfe']:.2f}",
                ha='right', va='top', fontsize=7, color=colors[i])
        values = np.concatenate([np.r_[s['observed'], s['plot_prediction']] for s in panel['series']])
        times = np.concatenate([s['time'] for s in panel['series']])
        if np.any(times <= 0) or np.any(values <= 0):
            raise ValueError('Cannot omit nonpositive observations from logarithmic figure')
        span = np.log10(max(values)/min(values)); pad = max(span*.08, .085)
        ylim = (10**(np.log10(min(values))-pad), 10**(np.log10(max(values))+pad))
        tpad = max(.035, np.log10(max(times)/min(times))*.04)
        xlim = (10**(np.log10(min(times))-tpad), 10**(np.log10(max(times))+tpad))
        ax = fig.add_axes([left, top-.161, width, .115])
        for i, s in enumerate(panel['series']):
            ax.plot(s['plot_time'], s['plot_prediction'], color=colors[i], linestyle=styles[i], zorder=2)
            ax.scatter(s['time'], s['observed'], marker=markers[i], s=16 if i == 0 else 9,
                facecolor='white', edgecolor=colors[i], linewidth=.65, zorder=3+i)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(ylim); ax.set_xlim(xlim)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        if span < 1.6:
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator(curve_helpers.log_time_ticks(*xlim)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='major', length=2.5, width=.5, pad=1.8)
        ax.tick_params(which='minor', length=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', color='#E7EBEF', linewidth=.45, zorder=0)
    fig.canvas.draw()
    issues = audit_layout(fig)
    renderer = fig.canvas.get_renderer()
    labels = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.texts]
    labels.append(('Legend', fig.legends[0].get_window_extent(renderer)))
    for i, (name, box) in enumerate(labels):
        for other, otherbox in labels[i+1:]:
            if box.overlaps(otherbox):
                raise ValueError('Figure 5 labels overlap: ' + name + '/' + other)
    if any(level == 'FAIL' for level, _ in issues):
        raise ValueError('Figure 5 layout failed: ' + repr(issues))
    return fig, issues


def export_figures(figures, destination, cache):
    import matplotlib.pyplot as plt
    import cairosvg
    import pymupdf
    from PIL import Image
    from check_figure import check_figure
    doc = pymupdf.open()
    records = []
    for index, fig in enumerate(figures, 1):
        name = destination.stem + ('_' + str(index) if len(figures) > 1 else '')
        png = cache / (name + '.png')
        gray = cache / (name + '_gray.png')
        fig.savefig(png, dpi=300, facecolor='white')
        with Image.open(png) as image:
            image.convert('L').save(gray, dpi=(300, 300))
        svg = io.BytesIO(); fig.savefig(svg, format='svg')
        pdf = cairosvg.svg2pdf(bytestring=svg.getvalue())
        part = pymupdf.open(stream=pdf, filetype='pdf')
        if len(part) != 1 or part[0].get_fonts() or part[0].get_images():
            raise ValueError('Publication PDF must preserve pure vector glyphs and curves')
        pix = part[0].get_pixmap(dpi=150, alpha=False)
        if not pix.width or not pix.height:
            raise ValueError('PDF cannot be rendered')
        doc.insert_pdf(part)
        issues, info = check_figure(str(png), min_dpi=300, target_inches=tuple(fig.get_size_inches()))
        if any(level == 'FAIL' for level, _ in issues):
            raise ValueError('Figure file check failed: ' + repr(issues))
        records.append({'preview': relative(png), 'grayscale_preview': relative(gray),
            'page_size_points': list(part[0].rect)[2:], 'file_checks': issues,
            'vector_paths': len(part[0].get_drawings()), 'raster_images': 0, 'font_resources': 0})
        part.close(); plt.close(fig)
    doc.save(destination); doc.close()
    return records


def draw_sampling(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from visual_qa import audit_layout
    from report_blueprint_validation import clustered_geomean
    methods = (METHOD, 'two_compartment', 'two_compartment_prior')
    labels = ('Vascular--tissue', 'Two-compartment', 'Two-comp. + CL prior')
    colors = ('#0072B2', '#D55E00', '#009E73')
    fig, axes = plt.subplots(1, 2, figsize=(185/25.4, 116/25.4), sharey=True)
    fig.subplots_adjust(left=.245, right=.97, top=.82, bottom=.21, wspace=.24)
    clusters = {cid: c['cluster_id'] for cid, c in data.cases.items()}
    details = {}
    for ax, sampling, title in zip(axes, ('full', 'five'), ('Full development curve', 'Five development points')):
        estimates = []
        for index, (method, color) in enumerate(zip(methods, colors)):
            values = {cid: data.units[cid, sampling, method].get('curve_metrics', {}).get('gmfe') for cid in data.cases}
            values = {cid: v for cid, v in values.items() if v is not None}
            report = clustered_geomean(values, {cid: clusters[cid] for cid in values})
            details[sampling+'__'+method] = report
            if not values:
                ax.text(1.04, index, 'No scored predictions', fontsize=7)
                continue
            offsets = np.linspace(-.2, .2, len(data.cases))
            for offset, cid in zip(offsets, data.cases):
                if cid in values:
                    ax.scatter(values[cid], index+offset, s=17, facecolor='white', edgecolor=color, linewidth=.7, zorder=3)
            estimate, ci = report['estimate'], report['ci95']
            if ci is not None:
                ax.plot(ci, [index, index], color=color, lw=2.2, alpha=.75, zorder=2)
            ax.scatter(estimate, index, marker='D', s=30, facecolor=color, edgecolor='white', linewidth=.5, zorder=4)
            estimates.append((labels[index], estimate, len(values)))
        ax.set_title(title, fontsize=9, loc='left', pad=12)
        ax.set_xscale('log'); ax.set_ylim(2.5, -.5)
        ax.set_yticks(range(3), labels)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(1, lw=.7, color='#89939D', zorder=0)
        ax.grid(axis='x', color='#E7EBEF', linewidth=.5)
        ax.set_xlabel('Target-curve GMFE (log scale)', fontsize=8)
        # Summary values stay outside the point cloud rather than label each drug.
        text = '  /  '.join(f'{x[1]:.3f}' for x in estimates)
        ax.text(0, 1.14, 'Drug-equal GMFE: '+text, transform=ax.transAxes, fontsize=7, color='#4B5966')
    all_values = [r.get('curve_metrics', {}).get('gmfe') for r in data.units.values()]
    high = max(v for v in all_values if v is not None)*1.2
    for ax in axes:
        ax.set_xlim(.94, high)
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ticks = [v for v in (1, 1.5, 2, 3, 5, 10, 20, 50) if v <= high]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.legend([Line2D([], [], marker='o', color='#536570', markerfacecolor='white', linestyle='none'),
                Line2D([], [], marker='D', color='#536570', linewidth=2)],
        ['One drug / target study', 'Drug-equal geometric mean and 95% study-cluster bootstrap interval'],
        loc='lower left', bbox_to_anchor=(.02, .04), ncol=1, frameon=False, fontsize=7.5)
    fig.text(.025, .95, 'Same target studies; only the development sampling is reduced', fontsize=10, weight='bold')
    fig.text(.025, .017, '10 drugs in 9 study clusters; 2,000 bootstrap resamples. Previously viewed development evaluation.',
        fontsize=7, color='#56636E')
    fig.canvas.draw(); issues = audit_layout(fig)
    if any(level == 'FAIL' for level, _ in issues):
        raise ValueError('Sampling figure layout failed: ' + repr(issues))
    return fig, {'layout': issues, 'statistics': details}


def clinical_table(data):
    h = table_helpers
    notes = [
        r'\textbf{Values.} P: dense-model endpoints on the request/model-tail windows. P$^*$: the same prediction with the frozen clinical sampling and cutoff rules; linear integration for Pletz 2004, linear-up/log-down otherwise. Only P$^*$ is scored. O: original study-reported scalars, never reconstructed; missing-O rows are omitted.',
        r'\textbf{Errors.} FE=$\max(P^*/O,O/P^*)$ (ideal 1); $T_{\max}$ uses absolute hours (ideal 0). Curve GMFE and ln-RMSE use positive observed sampling times and natural logs; $n$ counts points. --- means unavailable, not zero. Four significant figures are displayed.',
        r'\textbf{Quality.} $\dagger$ flags endpoint-specific weak-tail screens: span $\geq1$ half-life; additionally AUC extrapolation $\leq20\%$ for AUC$_\infty$/CL/$V_z$, and AUMC extrapolation $\leq20\%$ for MRT/$V_{\rm ss}$. Flagged errors remain descriptive. All ten drugs use the same vascular--tissue method. These viewed cases are development evaluations, not blinded validation.'
    ]
    tex = ['% Generated from the sealed adopted finite-exchange ten-drug run.',
        r'\begingroup\captionsetup[table]{name=Table}',
        r'\definecolor{BPClinicalInk}{HTML}{233A4B}\definecolor{BPClinicalFaint}{HTML}{EAF0F3}',
        r'\definecolor{BPClinicalStripe}{HTML}{F6F9FA}\definecolor{BPClinicalGray}{HTML}{56636E}',
        r'\setlength{\tabcolsep}{2.5pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}',
        r'\setlength{\LTcapwidth}{250mm}\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}',
        r'\renewcommand{\arraystretch}{1.08}\fontsize{10}{13}\selectfont',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{39.8mm}>{\centering\arraybackslash}p{20mm}*{3}{>{\centering\arraybackslash}p{37.55mm}}>{\centering\arraybackslash}p{32mm}>{\centering\arraybackslash}p{35mm}@{}}',
        r'\caption{Vascular--tissue blueprint: 80 scalar comparisons against original clinical references from 10 drugs; 104 clinical curve points.}\label{tab:blueprint_clinical_metrics}\\', r'\toprule']
    header = h.row([r'\textbf{Endpoint}', r'\textbf{Unit}', r'\shortstack{Dense model prediction\\P}',
        r'\shortstack{Matched prediction\\P$^*$}', r'\shortstack{Observation-based\\reference O}',
        r'\shortstack{Fold error\\FE}', r'\shortstack{Absolute error\\h ($T_{\max}$ only)}'], keep=False, stripe=True)
    tex += [header, r'\midrule\endfirsthead',
        r'\multicolumn{7}{@{}l@{}}{\textcolor{BPClinicalGray}{\bfseries Table~\thetable\quad Vascular--tissue blueprint: clinical endpoint comparisons (continued)}}\\[3pt]',
        r'\toprule', header, r'\midrule\endhead', r'\midrule']
    tex += [h.note(x) for x in notes] + [r'\bottomrule\endfoot', r'\midrule']
    tex += [h.note(x) for x in notes] + [r'\bottomrule\endlastfoot']
    for i, case in enumerate(data.table_cases, 1):
        if i > 1:
            tex.append(r'\addlinespace[5pt]')
        m = case['curve_metrics']['ours']
        title = clinical_helpers.escape(f"{i:02d}. {case['drug']} / {h.STUDIES[case['drug']]}")
        title += rf"\hfill GMFE {h.number(m['gmfe'])}; ln-RMSE {h.number(m['ln_rmse'])}; $n={m['n']}$\hspace{{1mm}}"
        tex += [h.band(title, dark=True), h.band(clinical_helpers.escape(case['native_context']))]
        for index, record in enumerate(r for r in data.table_rows if r['case_id'] == case['case_id']):
            ep, f = record['endpoint'], record['display_value_factor']
            scale = lambda v: None if v is None else v*f
            tex.append(h.row([clinical_helpers.TEX_ENDPOINTS[ep], clinical_helpers.escape(record['display_unit'].replace('.h/', '·h/')),
                h.number(scale(record['predicted_value'])), h.number(scale(record['evaluation_prediction']), record['matched_weak_tail']),
                h.number(scale(record['observed_value'])), h.number(record['fold_error'], record['matched_weak_tail']),
                h.number(record['absolute_error_h'])], stripe=index % 2 == 0))
        missing = ', '.join(clinical_helpers.ENDPOINTS[k] for k in case['missing_clinical_endpoints'])
        detail = ['Source: '+case['citation']+'.']
        if missing:
            detail.append('Original scalar reference not reported/qualified (row omitted): '+missing+'.')
        detail += case['clinical_reference_quality_notes']
        for key in case['approximations']:
            if key not in {'representative_adult_vs_group_summary', 'standard_nca_from_reported_sampling'}:
                detail.append(clinical_helpers.APPROXIMATIONS[key])
        if case['native_missing']:
            detail.append('Dense P unavailable under the numerical-tail quality gate: '+', '.join(clinical_helpers.ENDPOINTS[k] for k in case['native_missing'])+'. P* is retained separately without filling P.')
        if case['endpoint_protocol'].get('mrt_infusion_correction') == 'none':
            detail.append('P* MRT/Vss use the source convention without infusion correction; dense P retains its infusion-corrected definition.')
        for j, note in enumerate(detail):
            tex.append(h.note(clinical_helpers.escape(note), keep=j<len(detail)-1))
    return '\n'.join(tex+[r'\end{longtable}\endgroup', ''])


def input_table(data):
    helpers = scenario_helpers
    rows = []
    horizons = set()
    for scenario in data.scenarios:
        req, endpoints = scenario['request_data'], scenario['dense']['endpoints']
        curve = data.store.ref(scenario['prediction'])
        horizons.add(max(curve['time_min'])/60)
        cells = [f"{helpers.num(req['dosing']['dose_mg']*1000)} / {scenario['condition']} / {helpers.num(req['dosing']['infusion_minutes'])}"]
        for ep, _, _, field, factor in helpers.ENDPOINTS:
            if ep == 'plasma_concentration_time':
                # One source link, without duplicate curve CSVs.
                value = (r'\href{\detokenize{../../' + scenario['prediction']['path']
                    + r'}}{\textcolor{BPTeal}{Curve data}}')
            else:
                scalar = endpoints.get(ep, {}).get(field)
                value = helpers.num(None if scalar is None else scalar*factor)
            cells.append(value)
        rows.append(cells)
    out = [r'% Frozen finite-exchange input response; no refitting or clinical scoring.',
        r'\begingroup\captionsetup[table]{name=Table}',
        r'\definecolor{BPInk}{HTML}{233A4B}\definecolor{BPTeal}{HTML}{287C86}',
        r'\definecolor{BPFaint}{HTML}{EAF0F3}\definecolor{BPStripe}{HTML}{F6F9FA}\definecolor{BPGray}{HTML}{56636E}',
        r'\newcommand{\BPmissing}{\textcolor{BPGray}{---}}',
        r'\setlength{\tabcolsep}{2.5pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}',
        r'\setlength{\LTcapwidth}{250mm}\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}',
        r'\renewcommand{\arraystretch}{1.12}\fontsize{9.5}{12.5}\selectfont',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{37mm}*{11}{>{\centering\arraybackslash}p{17.6mm}}@{}}',
        r'\caption{Treprostinil input-response example: three doses, three input conditions and eleven predicted endpoints.}\label{tab:blueprint_ten_drug}\\', r'\toprule']
    header = helpers.row([r'\shortstack[l]{Dose / input / infusion\\$\mu$g / C1--C3 / min}'] +
        [rf'\shortstack{{{label}\\{unit}}}' for _, label, unit, _, _ in helpers.ENDPOINTS], 'BPFaint')
    out += [header, r'\midrule\endfirsthead', r'\toprule', header, r'\midrule\endhead',
        r'\bottomrule\endfoot\bottomrule\endlastfoot',
        r'\rowcolor{BPInk}\multicolumn{12}{@{}p{250mm}@{}}{\strut\hspace{1.5mm}\color{white}\bfseries Treprostinil\hfill Locked vascular--tissue reversible exchange\hspace{1.5mm}}\\',
        r'\rowcolor{BPFaint}\multicolumn{12}{@{}p{250mm}@{}}{\strut\hspace{1.5mm}Predictions P only; the same drug mapping is reused for all nine inputs without refitting.}\\']
    out += [helpers.row(cells, 'BPStripe' if i//3 % 2 == 0 else None) for i, cells in enumerate(rows)]
    out += [r'\addlinespace[4pt]',
        helpers.note(r'\textbf{Inputs.} C1: representative male adult, 30 y, 69.5 kg, 175 cm; 60-min IV infusion. C2: the same adult and total dose, infused over 120 min. C3: hypothetical female adult, 30 y, 60 kg, 165 cm; 60-min IV infusion. C3 changes several physiological inputs together, not sex alone.'),
        helpers.note(r'\textbf{Doses.} 31.275, 62.55 and 125.1 $\mu$g are 0.5, 1 and 2 times the reference total dose. Absolute dose is unchanged across C1--C3 at each level. Source weight is retained; age, sex and height are representative assumptions.'),
        helpers.note(r'\textbf{Endpoints.} Curves, AUC and peak endpoints use 0--9 h; $C_{9\mathrm{h}}$ is the value at exactly 9 h, not an assay-limited measurement. Terminal endpoints use the unchanged dense-trajectory numerical-quality policy (simulation horizon: '
            + ', '.join(f'{h:g}' for h in sorted(horizons)) + r' h). MRT subtracts half the infusion duration; $V_{\rm ss}=\mathrm{CL}\times\mathrm{MRT}$.'),
        helpers.note(r'\textbf{Interpretation.} This is an input-response demonstration, not nine clinical validations. No reference O or error is assigned. The 1$\times$ C1 prediction reproduces the target input and adds no clinical sample. Dense terminal estimates differ from clinically sampled NCA.'),
        helpers.note(r'\textbf{Reading.} Values have three significant figures. --- indicates an unavailable endpoint, not zero. Model linearity does not establish clinical dose proportionality. Curve data link to the sealed concentration trajectories in minutes and $\mu$mol/L.'),
        r'\end{longtable}\endgroup', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    data = Data(args.run_dir)
    if args.validate_only:
        print(json.dumps({'validated': True, 'drugs': 10, 'comparison_units': 60,
            'clinical_scalar_references': len(data.table_rows), 'scenarios': 9, 'source_pins': len(data.store.pins)}))
        return
    output = data.run / 'paper'; output.mkdir(exist_ok=True)
    cache = ROOT / 'runtime/cache/vascular_tissue_ten_drug'; cache.mkdir(parents=True, exist_ok=True)
    figure_style()
    import pandas as pd
    from profile_data import profile_data
    profile = profile_data(pd.DataFrame(data.profile_rows), group_cols=['drug', 'stage'])
    fig, curve_layout = draw_curves(data)
    curve_audit = export_figures([fig], output / 'clinical_curves.pdf', cache)
    methods = [dict(m) for m in data.baseline['methods']]
    methods[0]['label'] = 'Vascular--tissue blueprint'
    methods[0]['information'] = 'Evidence-first inputs and full development curve; three calibrated parameters'
    figures, matrix_layout = [], []
    for i, subset in enumerate((data.comparison[:5], data.comparison[5:])):
        figure, audit = matrix_plot.draw_page(methods, subset, part=i+1, parts=2, start_index=i*5)
        figures.append(figure); matrix_layout.append(audit)
    matrix_audit = export_figures(figures, output / 'method_comparison.pdf', cache)
    sampling_figure, sampling_audit = draw_sampling(data)
    sampling_audit['pages'] = export_figures([sampling_figure], output / 'sampling_comparison.pdf', cache)
    (output / 'clinical_endpoints.tex').write_text(clinical_table(data))
    (output / 'input_response.tex').write_text(input_table(data))
    data.store.unchanged()
    files = [output / name for name in ('clinical_curves.pdf', 'method_comparison.pdf', 'sampling_comparison.pdf', 'clinical_endpoints.tex', 'input_response.tex')]
    manifest = {'version': VERSION, 'method': METHOD, 'method_version': 'physiological-flow-exchange-calibration-1',
        'run': relative(data.run), 'main_sampling_condition': 'full', 'source_pins': data.store.pins,
        'output_sha256': {relative(p): curve_helpers.digest(p) for p in files},
        'observed_curve_points': len(data.profile_rows), 'clinical_reference_count': len(data.table_rows),
        'missing_scalar_reference_count': 100-len(data.table_rows), 'input_response_scenarios': 9,
        'curve_figure': {'layout': curve_layout, 'pages': curve_audit, 'profile': profile},
        'sampling_figure': sampling_audit,
        'comparison_figure': {'layout': matrix_layout, 'pages': matrix_audit, 'methods': methods,
            'frozen_rf_svr_xgboost_recomputed_with_common_nca': True,
            'two_compartment_source': 'Frozen validation_20260929, full development curve; common ten-drug runner'},
        'table_definition': {'P': 'Dense-model prediction; no clinical observations used',
            'P_star': 'Prediction-only NCA at frozen clinical schedule/cutoff; scored against original clinical scalar O',
            'O': 'Original study-reported values only; missing references never reconstructed'},
        'clinical_table_rows': data.table_rows,
        'checks': {'predictions_locked_before_scoring': True, 'prediction_units_rescored': len(data.scored),
            'development_target_same_selected_mapping': True, 'original_clinical_references_preserved': True,
            'target_requests_never_fit': True, 'original_results_not_modified': True},
        'interpretation': 'Previously viewed ten-drug development evaluation, not new blinded or unseen-compound validation',
        'cleanup': 'Only publication PDF/TeX and this manifest under paper; preview PNGs stay in project cache'}
    (output / 'publication_manifest.json').write_text(json.dumps(json_plain(manifest), indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'published': relative(output), 'files': len(files)+1, 'clinical_reference_count': len(data.table_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
