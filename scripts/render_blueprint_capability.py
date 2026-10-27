#!/usr/bin/env python3
"""Publish the sealed ten-drug blueprint-capability experiment.

This publication module does not train, calibrate or select a model.  It keeps
engineering decisions, clinical prediction errors and numerical reproduction
as separate outcomes.  Its only final products are a vector multipanel figure,
an English LaTeX table and a source-bound publication manifest under RUN/paper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/cache/paper_figures'))

import numpy as np

from report_blueprint_validation import Store
from render_blueprint_architecture_paper import pinned_references, json_plain
from render_vascular_tissue_ten_drug import export_figures, figure_style
from report_blueprint_validation import _case_reference, _score_prediction, clustered_geomean, paired_ratio

VERSION = 'blueprint-capability-publication-1'
DEFAULT = ROOT / 'experiments/blueprint_design/blueprint_capability_20260930'
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill')
COLORS = {'ink': '#233A4B', 'blue': '#0072B2', 'orange': '#D55E00',
          'teal': '#009E73', 'purple': '#CC79A7', 'gray': '#788590',
          'light': '#EDF2F5', 'grid': '#E3E9ED'}
POLICIES = ('evidence_policy', 'fixed_template', 'rf_fill', 'no_conflict_guard')
SCENARIOS = ('available', 'missing_fu', 'wrong_species_fu', 'conflicting_fu',
             'missing_required_pbpk_formula', 'unsupported_ddi')
SHORT_POLICY = {'evidence_policy': 'Evidence\npolicy', 'fixed_template': 'Fixed\ninputs',
                'rf_fill': 'RF-first', 'no_conflict_guard': 'No conflict\nguard'}
SHORT_SCENARIO = {'available': 'Available', 'missing_fu': 'Missing fu',
    'wrong_species_fu': 'Nonhuman fu', 'conflicting_fu': 'Conflicting fu',
    'missing_required_pbpk_formula': 'No molecular formula', 'unsupported_ddi': 'DDI request'}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def require_unique(rows, fields, label):
    identities = [tuple(row[field] for field in fields) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate publication records: ' + label)
    return identities


def finite(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A finite measured value is required: ' + label)
    if nonnegative and value < 0:
        raise ValueError('A nonnegative measured value is required: ' + label)
    return float(value)


def number(value, digits=3):
    if value is None:
        return '---'
    return format(finite(value, 'publication number'), f'.{digits}g')


def tex_escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
        '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(text))


def panel_title(ax, letter, title, subtitle=None):
    ax.set_title(title, loc='left', fontsize=8.5, color=COLORS['ink'], pad=12)
    box = ax.get_position()
    ax.figure.text(box.x0 - .028, box.y1 + .019, letter, fontsize=10,
                   fontweight='bold', color=COLORS['ink'], ha='left', va='bottom')
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=6.4,
                color=COLORS['gray'], ha='left', va='bottom')


def clean_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7, length=2.5, width=.5, pad=2)
    ax.grid(axis='x', color=COLORS['grid'], linewidth=.5, zorder=0)


def cell_background(ax, cells):
    """Draw small categorical matrices as true vector rectangles."""
    from matplotlib.patches import Rectangle
    rows, columns = len(cells), len(cells[0])
    for i, row in enumerate(cells):
        for j, color in enumerate(row):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=color,
                                   edgecolor='white', linewidth=1.0))
    ax.set_xlim(-.5, columns - .5); ax.set_ylim(rows - .5, -.5)


class Data:
    """Verify source and completion locks before reading any scored outcomes."""
    def __init__(self, run):
        self.store = store = Store(ROOT)
        self.run = store.path(str(run))
        self.state = store.read(str(self.run / 'state.json'))
        if self.state.get('status') != 'complete' or any(self.state.get(k) is not True for k in
                ('all_attempts_locked_before_scoring', 'monitor_stopped_before_scoring')):
            raise ValueError('Publication requires all completed predictions and pre-scoring locks')
        self.protocol = store.ref(self.state['protocol'])
        self.result = store.ref(self.state['results'])
        if self.protocol.get('version') != 'blueprint-capability-1':
            raise ValueError('Unexpected capability experiment version')
        for field, filename in (('protocol', 'protocol.json'), ('results', 'results.json')):
            if store.path(self.state[field]['path']) != self.run / filename:
                raise ValueError('Completion state points to another experiment directory')
        for doc in (self.state, self.result):
            if doc.get('version') != self.protocol.get('version'):
                raise ValueError('Frozen experiment versions differ')
        if self.result.get('all_attempts_locked_before_scoring') is not True:
            raise ValueError('Scored results have no prediction-first lock')
        for doc in (self.protocol, self.state, self.result):
            pinned_references(doc, store)
        self.cases = {c['case_id']: c for c in self.protocol['cases']}
        if len(self.cases) != 10 or len({c['drug'] for c in self.cases.values()}) != 10:
            raise ValueError('The complete ten-drug cohort is required')
        self.drugs = [c['drug'] for c in self.cases.values()]
        self.decisions = self.result['decisions']
        if self.decisions != self.protocol['decisions'] or len(self.decisions) != 240:
            raise ValueError('Engineering decisions changed after preparation')
        identities = require_unique(self.decisions, ('case_id', 'scenario', 'method'), 'engineering decisions')
        if set(identities) != {(c, s, m) for c in self.cases for s in SCENARIOS for m in POLICIES}:
            raise ValueError('Engineering scenario matrix is incomplete')
        self.units = self.result['units']
        if len(self.units) != 520:
            raise ValueError('All 520 logical comparison units must remain present')
        require_unique(self.units, ('case_id', 'scenario', 'sampling', 'method'), 'logical clinical units')
        self.lookup = {(u['case_id'], u['scenario'], u['sampling'], u['method']): u for u in self.units}
        declared = {u['id']: u for u in self.protocol['units']}
        if set(declared) != {u['id'] for u in self.units}:
            raise ValueError('Scored clinical units differ from the declared matrix')
        self.references = {cid: _case_reference(case, store) for cid, case in self.cases.items()}
        self.scored = {}
        for row in self.units:
            unit = declared[row['id']]
            if unit['status'] == 'blocked':
                if row['attempt_status'] != 'blocked' or row['scoring_status'] != 'not_attempted':
                    raise ValueError('A blocked construction was silently removed or scored')
                continue
            compid = unit['computation_id']
            computation = self.state['computations'][compid]
            if row.get('computation_id') != compid or row['attempt_status'] != computation['status']:
                raise ValueError('The deduplicated computation identity changed')
            if computation['status'] != 'predicted':
                continue
            if row['prediction_source'] != computation['prediction']:
                raise ValueError('Scored curve differs from the locked prediction')
            if compid not in self.scored:
                case = self.cases[row['case_id']]
                self.scored[compid], _, _ = _score_prediction(case, *self.references[row['case_id']], computation, store)
            fresh = self.scored[compid]
            for key in ('gmfe', 'ln_rmse', 'n'):
                self.same(fresh['curve_metrics'].get(key), row.get('curve_metrics', {}).get(key), key)
            for endpoint, scalar in fresh['endpoints'].items():
                for key in ('value', 'reference', 'error'):
                    self.same(scalar[key], row['endpoints'][endpoint][key], endpoint + '/' + key)
                for key in ('unit', 'weak_tail', 'error_metric'):
                    if scalar[key] != row['endpoints'][endpoint][key]:
                        raise ValueError('Endpoint interpretation changed: ' + endpoint + '/' + key)
        self.protection = self.result['protection']
        require_unique(self.protection, ('case_id', 'sampling', 'method', 'parameter'), 'protected construction inputs')
        self.replay = self.result['replay']
        if len(self.replay) != 40:
            raise ValueError('All forty trajectory replays across both calibration rules must be reported')
        require_unique(self.replay, ('case_id', 'sampling', 'method'), 'trajectory replay')
        for row in self.replay:
            if row.get('status') == 'prediction_failed':
                if row['passed'] is not False:
                    raise ValueError('An unavailable trajectory cannot pass replay')
                continue
            for key in ('max_absolute_difference_umol_l', 'max_relative_difference', 'wall_seconds'):
                finite(row[key], key, nonnegative=True)
        self.calibration = self.result['calibration']
        if len(self.calibration) != 40:
            raise ValueError('Both calibration rules require all ten drugs and both sampling conditions')
        require_unique(self.calibration, ('case_id', 'sampling', 'method'), 'calibration diagnostics')
        self.n_only_equivalence = {'checked_formula_fits': 0, 'matching_formula_fits': 0,
                                   'interpretation': 'Arithmetic rule comparison; no additional experiment or independent sample'}
        for fit in self.state['fits'].values():
            mapping = store.ref(fit['mapping'])
            if mapping['version'] != 'capability-adaptive-flow-exchange-1':
                continue
            selected = mapping['axis_selection']
            config = selected['config']
            cap = min(len(config['priority']), selected['observation_count'] // config['minimum_points_per_axis'])
            expected = config['priority'][:cap]
            self.n_only_equivalence['checked_formula_fits'] += 1
            self.n_only_equivalence['matching_formula_fits'] += selected['active_parameters'] == expected
        self.applicability = self.result['applicability']
        if len(self.applicability) != 160:
            raise ValueError('All 110 plasma requests and 50 out-of-scope requests must be retained')
        require_unique(self.applicability, ('case_id', 'request_type'), 'request scope')
        self.permutations = self.result['permutations']
        if self.permutations != self.protocol['permutations'] or len(self.permutations) != 80:
            raise ValueError('All eighty input-order checks must be retained')
        self.tail_gate = self.result['tail_gate']
        pinned_references(self.tail_gate, store)
        require_unique(self.tail_gate['records'], ('case_id', 'sampling', 'method', 'endpoint'), 'tail-gate association')
        self.validate_tail_gate()
        self.summary = {(r['scenario'], r['sampling'], r['method']): r for r in self.result['clinical_summary']}
        if len(self.summary) != 52:
            raise ValueError('Clinical summaries must cover every scenario and calibration comparison')
        for key, reported in self.summary.items():
            rows = [u for u in self.units if (u['scenario'], u['sampling'], u['method']) == key]
            valid = {u['case_id']: u['curve_metrics']['gmfe'] for u in rows if u.get('curve_metrics', {}).get('gmfe') is not None}
            if reported['attempted'] != len(rows) or reported['scored'] != len(valid):
                raise ValueError('Clinical summary denominator changed')
            for field, expected in [('blocked', sum(u['attempt_status'] == 'blocked' for u in rows)),
                    ('failed', sum(u['attempt_status'] == 'failed' or u['scoring_status'] == 'failed' for u in rows))]:
                if reported[field] != expected:
                    raise ValueError('Clinical coverage count changed: ' + field)
            bootstrap = self.protocol['bootstrap']
            fresh = clustered_geomean(valid, {c: self.cases[c]['cluster_id'] for c in valid},
                seed=bootstrap['seed'], resamples=bootstrap['resamples'])
            self.same(fresh['estimate'], reported['curve_gmfe']['estimate'], 'drug-equal GMFE')
            if fresh['ci95'] != reported['curve_gmfe']['ci95']:
                if fresh['ci95'] is None or reported['curve_gmfe']['ci95'] is None or not np.allclose(
                        fresh['ci95'], reported['curve_gmfe']['ci95'], rtol=1e-10, atol=1e-12):
                    raise ValueError('Clinical bootstrap interval changed')
        for sampling in ('full', 'five'):
            def values(method):
                return {cid: self.lookup[cid, 'calibration', sampling, method].get('curve_metrics', {}).get('gmfe')
                        for cid in self.cases}
            fresh = paired_ratio(values('adaptive_axes'), values('always_three'),
                {cid: case['cluster_id'] for cid, case in self.cases.items()}, expected_ids=list(self.cases),
                seed=self.protocol['bootstrap']['seed'], resamples=self.protocol['bootstrap']['resamples'])
            reported = self.result['adaptive_to_always_three_curve_ratio'][sampling]
            self.same(fresh['estimate'], reported['estimate'], 'Adaptive/fixed paired GMFE ratio')
            if fresh['ci95'] is not None and not np.allclose(fresh['ci95'], reported['ci95'], rtol=1e-10, atol=1e-12):
                raise ValueError('Paired calibration bootstrap interval differs')
        for sampling, policies in self.result['missing_fu_effect'].items():
            for method, groups in policies.items():
                for name, field in (('changed_four_drugs', 'changed_fu_case_ids'),
                                    ('unchanged_six_controls', 'unchanged_fu_case_ids')):
                    ids = self.protocol[field]
                    def values(scenario):
                        return {cid: self.lookup[cid, scenario, sampling, method].get('curve_metrics', {}).get('gmfe')
                                for cid in ids}
                    fresh = paired_ratio(values('missing_fu'), values('available'),
                        {cid: self.cases[cid]['cluster_id'] for cid in ids}, expected_ids=ids,
                        seed=self.protocol['bootstrap']['seed'], resamples=self.protocol['bootstrap']['resamples'])
                    reported = groups[name]
                    self.same(fresh['estimate'], reported['estimate'], 'Missing-fu paired curve effect')
                    if (fresh['ci95'] is None) != (reported['ci95'] is None) or (fresh['ci95'] is not None and
                            not np.allclose(fresh['ci95'], reported['ci95'], rtol=1e-10, atol=1e-12)):
                        raise ValueError('Missing-fu subgroup bootstrap interval differs')
        self.fu_recovery = self.hidden_fu_recovery()
        store.checksum(__file__)
        store.unchanged()

    @staticmethod
    def same(left, right, label):
        if (left is None) != (right is None) or (left is not None and not math.isclose(
                float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError('Publication arithmetic disagrees: ' + label)

    def estimates(self, scenario, sampling, method):
        return self.summary[scenario, sampling, method]

    def hidden_fu_recovery(self):
        records = []
        for cid in self.protocol['changed_fu_case_ids']:
            case = self.cases[cid]
            values, refs = {}, {}
            for kind in ('evidence', 'rf'):
                refs[kind] = case['requests'][kind]['development']
                request = self.store.ref(refs[kind])
                rows = [r for r in request['evidence'] if r['parameter'] == 'fraction_unbound']
                if len(rows) != 1 or rows[0]['unit'] not in ('1', 'fraction'):
                    raise ValueError('Hidden-fu recovery requires one unitless construction value')
                pinned_references(rows[0], self.store)
                value = finite(rows[0]['value'], 'hidden fu')
                if not 0 < value <= 1:
                    raise ValueError('Hidden-fu recovery values must lie in (0,1]')
                values[kind] = value
            records.append({'case_id': cid, 'drug': case['drug'], 'qualified_fu': values['evidence'],
                'rf_fu': values['rf'], 'fold_error': max(values['evidence'] / values['rf'],
                    values['rf'] / values['evidence']), 'unit': '1', 'request_sources': refs})
        return {'records': records, 'n_drugs': len(records),
            'gmfe': float(np.exp(np.mean(np.log([r['fold_error'] for r in records])))),
            'within_twofold': sum(r['fold_error'] <= 2 for r in records),
            'interpretation': 'Recovery of four hidden existing construction-evidence values; not new clinical observations or new independent samples'}

    def validate_tail_gate(self):
        for name, reported in self.tail_gate['by_method_sampling'].items():
            method, sampling = name.split('__')
            rows = [r for r in self.tail_gate['records'] if r['method'] == method and r['sampling'] == sampling]
            eligible = [r for r in rows if r['reference_available'] and r['matched_fold_error'] is not None
                        and r['matched_fold_error'] > 0]
            large = [r for r in eligible if r['matched_fold_error'] > 2]
            flagged = [r for r in eligible if r['warning']]
            passed = [r for r in eligible if not r['warning']]
            hit = sum(r['warning'] for r in large)
            false = sum(r['matched_fold_error'] <= 2 for r in flagged)
            counts = {'requested_tail_endpoints': len(rows), 'matched_error_available': len(eligible),
                'large_matched_errors': len(large), 'flagged_large_errors': hit,
                'missed_large_errors': len(large) - hit, 'false_warnings': false,
                'warnings_with_reference_and_matched_error': len(flagged),
                'passed_with_reference_and_matched_error': len(passed)}
            if any(reported[k] != v for k, v in counts.items()):
                raise ValueError('Tail-gate association counts changed: ' + name)
            for row in rows:
                original = self.lookup[row['case_id'], 'calibration', sampling, method]['endpoints'][row['endpoint']]
                self.same(row['matched_fold_error'], original['error'], 'Tail gate / matched FE')
                dense = self.store.ref(row['gate_source'])
                status = dense['endpoints'][row['endpoint']]['status']
                if row['dense_gate_status'] != status or row['warning'] != (status != 'completed'):
                    raise ValueError('Tail warning differs from its locked numerical status')

    def profile(self):
        import pandas as pd
        from profile_data import profile_data
        clinical = [{'drug': u['drug'], 'scenario': u['scenario'], 'sampling': u['sampling'],
            'method': u['method'], 'status': u['attempt_status'],
            'gmfe': u.get('curve_metrics', {}).get('gmfe')} for u in self.units]
        replay = [{k: row.get(k) for k in ('drug', 'sampling', 'passed', 'max_absolute_difference_umol_l',
                  'max_relative_difference', 'wall_seconds')} for row in self.replay]
        return {'clinical': profile_data(pd.DataFrame(clinical), group_cols=['scenario', 'sampling', 'method']),
            'replay': profile_data(pd.DataFrame(replay), group_cols=['sampling'])}


def draw_protection(ax, data):
    columns = [('fraction_unbound', 'always_three'), ('fraction_unbound', 'adaptive_axes'),
               ('renal_clearance', 'always_three'), ('renal_clearance', 'adaptive_axes')]
    matrix = np.full((10, 4), np.nan)
    groups = {}
    for i, drug in enumerate(data.drugs):
        for j, (parameter, method) in enumerate(columns):
            rows = [r for r in data.protection if r['drug'] == drug and r['parameter'] == parameter and r['method'] == method]
            groups[drug, parameter, method] = rows
            if rows:
                matrix[i, j] = sum(r['passed'] for r in rows) / len(rows)
    cell_background(ax, [['#F1F3F5' if not np.isfinite(v) else '#C9E2EC' if v == 1 else '#F2DDCA'
                          for v in row] for row in matrix])
    for i, drug in enumerate(data.drugs):
        for j, (parameter, method) in enumerate(columns):
            rows = groups[drug, parameter, method]
            label = f"{sum(r['passed'] for r in rows)}/{len(rows)}" if rows else '---'
            ax.text(j, i, label, ha='center', va='center', fontsize=7.5, color=COLORS['ink'])
    ax.set_xticks(range(4), ['fu\n3 axes', 'fu\nSelected', 'Renal CL\n3 axes', 'Renal CL\nSelected'], fontsize=6.5)
    ax.set_yticks(range(10), data.drugs, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    panel_title(ax, 'a', 'Qualified inputs remain fixed', 'Passed / eligible; --- no qualified input')
    return {'eligible_mapping_checks': len(data.protection), 'matched': sum(r['passed'] for r in data.protection),
            'clinical_interpretation': 'Preserved construction inputs, not verified target-patient parameter values'}


def draw_decisions(ax, data):
    matrix = np.zeros((len(SCENARIOS), len(POLICIES)), dtype=int)
    mismatch = np.zeros_like(matrix)
    for i, scenario in enumerate(SCENARIOS):
        for j, policy in enumerate(POLICIES):
            rows = [r for r in data.decisions if r['scenario'] == scenario and r['method'] == policy]
            if len(rows) != 10:
                raise ValueError('Each engineering scenario/policy must retain ten requests')
            matrix[i, j] = sum(r['status'] == 'ready' for r in rows)
            mismatch[i, j] = sum((r['status'] == 'ready') != r['expected_ready'] for r in rows)
    # Discrete states, with exact counts in every cell, rather than a continuous
    # color scale that could imply a performance score.
    states = np.where(matrix == 0, 0, np.where(matrix == 10, 2, 1))
    palette = ['#F0F3F6', '#B9D9E7', '#307F9F']
    cell_background(ax, [[palette[v] for v in row] for row in states])
    for (i, j), value in np.ndenumerate(matrix):
        text = f'{value}/10' + (' !' if mismatch[i, j] else '')
        ax.text(j, i, text, ha='center', va='center', fontsize=7,
                color='white' if value >= 7 else COLORS['ink'])
    ax.set_xticks(range(4), [SHORT_POLICY[p] for p in POLICIES], fontsize=6.6)
    ax.set_yticks(range(6), [SHORT_SCENARIO[s] for s in SCENARIOS], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    panel_title(ax, 'b', 'Completion and refusal', 'Ready / 10; ! engineering rule mismatch')
    return {'ready_counts': matrix.tolist(), 'rule_mismatches': mismatch.tolist(),
            'row_order': list(SCENARIOS), 'column_order': list(POLICIES),
            'hidden_fu_recovery': data.fu_recovery,
            'annotation': 'Each cell reports the full denominator; engineered checks are not clinical validations'}


def draw_calibration(ax, data):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    methods = ('always_three', 'adaptive_axes')
    x_positions = {'full': -.12, 'five': .12}
    points, diagnostics = [], {}
    for m, method in enumerate(methods):
        for sampling, marker, color in [('full', 'o', COLORS['blue']), ('five', '^', COLORS['orange'])]:
            rows = [data.lookup[cid, 'calibration', sampling, method] for cid in data.cases]
            valid = [r for r in rows if r.get('curve_metrics', {}).get('gmfe') is not None]
            offsets = dict(zip(data.cases, np.linspace(-.10, .10, 10)))
            for row in valid:
                value = row['curve_metrics']['gmfe']
                y = m + x_positions[sampling] + offsets[row['case_id']] * .7
                ax.scatter(value, y, s=20, marker=marker, facecolor='white', edgecolor=color,
                           linewidth=.8, zorder=3)
                points.append(value)
            summary = data.estimates('calibration', sampling, method)
            estimate = summary['curve_gmfe']['estimate']
            interval = summary['curve_gmfe']['ci95']
            if interval is not None:
                ax.plot(interval, [m + x_positions[sampling]] * 2, color=color, linewidth=1.7, alpha=.7, zorder=1)
            if estimate is not None:
                ax.scatter(estimate, m + x_positions[sampling], marker='D', s=25, color=color,
                           edgecolor='white', linewidth=.5, zorder=4)
            diagnostics[method + '__' + sampling] = summary
    ax.set_xscale('log'); ax.set_ylim(1.6, -.55)
    high = max(points, default=2) * 1.2
    ticks = [x for x in (1, 1.5, 2, 3, 5, 10, 20, 50) if x <= high]
    ax.set_xlim(.96, high); ax.set_yticks([0, 1], ['Three scales', 'Selected scales'], fontsize=7)
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Target-curve GMFE (log scale)', fontsize=7.5)
    clean_axes(ax)
    ax.legend(handles=[Line2D([], [], color=c, marker=marker, markerfacecolor='white', linestyle='none',
        markersize=4, label=label) for marker, c, label in [('o', COLORS['blue'], 'Full'), ('^', COLORS['orange'], 'Five points')]],
        frameon=False, loc='lower left', bbox_to_anchor=(0, -.40), ncol=2, fontsize=7,
        handletextpad=.4, columnspacing=1.2)
    panel_title(ax, 'c', 'Restricted calibration', 'Drug points and drug-equal GMFE [95% CI]')
    diagnostics['point_count_rule_equivalence'] = data.n_only_equivalence
    return diagnostics


def draw_scope(ax, data):
    groups = [('always_three', 'full'), ('always_three', 'five'),
              ('adaptive_axes', 'full'), ('adaptive_axes', 'five')]
    # Fixed semantic column shading, never a color-coded composite score.
    cell_background(ax, [['#C9E2EC', '#F0DFD0', '#EDF1F4'] for _ in groups])
    details = {}
    for i, (method, sampling) in enumerate(groups):
        row = data.tail_gate['by_method_sampling'][method + '__' + sampling]
        large = row['large_matched_errors']
        small = row['matched_error_available'] - large
        cells = [(row['flagged_large_errors'], large), (row['missed_large_errors'], large),
                 (row['false_warnings'], small)]
        for j, (count, total) in enumerate(cells):
            ax.text(j, i, f'{count}/{total}' if total else '---', ha='center', va='center',
                    fontsize=8, color=COLORS['ink'])
        details[method + '__' + sampling] = row
    ax.set_xticks(range(3), ['Flagged\nFE > 2', 'Missed\nFE > 2', 'Warned\nFE ≤ 2'], fontsize=7)
    ax.set_yticks(range(4), ['3 axes / full', '3 axes / five', 'Selected / full', 'Selected / five'], fontsize=6.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    admitted = [r for r in data.applicability if r['expected_supported']]
    negative = [r for r in data.applicability if not r['expected_supported']]
    annotation = (f'Scope: {sum(r["supported"] for r in admitted)}/{len(admitted)} plasma admitted\n'
                  f'{sum(not r["supported"] for r in negative)}/{len(negative)} negative requests blocked')
    ax.text(0, -.35, annotation, transform=ax.transAxes, fontsize=6.2, color=COLORS['gray'])
    panel_title(ax, 'd', 'Numerical tail warnings', 'Dense gate vs P*: NCA windows differ')
    return {'tail_gate_groups': details, 'scope_annotation': annotation,
        'clinical_interpretation': 'Exploratory association, not a calibrated clinical-confidence classifier. Scope gate shared by fixed baseline.'}


def draw_replay(ax, data):
    ax.set_axis_off()
    passed = sum(r['passed'] for r in data.replay)
    readback = sum(r.get('native_readback_verified', False) for r in data.replay)
    maximum = max((r['max_absolute_difference_umol_l'] for r in data.replay
                   if r.get('max_absolute_difference_umol_l') is not None), default=None)
    ax.text(0, .79, f'{passed}/{len(data.replay)} trajectories reproduced', fontsize=8.5, color=COLORS['ink'], transform=ax.transAxes)
    ax.text(.48, .79, f'Max. absolute difference: {number(maximum)} µmol/L', fontsize=8.2, color=COLORS['ink'], transform=ax.transAxes)
    ax.text(0, .59, f'{readback}/{len(data.replay)} frozen native readbacks verified; no native rerun or refit',
            fontsize=7.5, color=COLORS['gray'], transform=ax.transAxes)
    centers = np.linspace(.30, .92, 4)
    for x, method in zip(centers, POLICIES):
        ax.text(x, .37, SHORT_POLICY[method].replace('\n', ' '), fontsize=7, ha='center', transform=ax.transAxes)
    counts = {}
    for scenario, y, label in [('available', .19, 'Normal order reversal'), ('conflicting_fu', .015, 'Conflict order reversal')]:
        ax.text(0, y, label, fontsize=7.1, transform=ax.transAxes, va='center')
        for x, method in zip(centers, POLICIES):
            rows = [r for r in data.permutations if r['scenario'] == scenario and r['method'] == method]
            good = sum(r['status_equal'] and r['selected_values_equal'] for r in rows)
            counts[scenario + '__' + method] = {'matched': good, 'checked': len(rows)}
            ax.text(x, y, f'{good}/{len(rows)}', fontsize=8, ha='center', va='center', transform=ax.transAxes,
                    color=COLORS['blue'] if good == len(rows) else COLORS['orange'])
    panel_title(ax, 'e', 'Reproducibility and input-order sensitivity')
    return {'replays_passed': passed, 'replays_checked': len(data.replay),
            'max_absolute_difference_umol_l': maximum, 'native_readbacks_verified': readback,
            'permutation_checks': counts, 'reproduction_scope': 'Locked physiology and solver; no fresh native rebuild'}


def draw(data):
    import matplotlib.pyplot as plt
    from visual_qa import audit_layout
    fig = plt.figure(figsize=(185 / 25.4, 212 / 25.4), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.16, right=.97, top=.89, bottom=.065,
        hspace=.58, wspace=.85, height_ratios=(1.15, .82, .55))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :])]
    details = {name: function(ax, data) for name, function, ax in zip(
        ('protection', 'decisions', 'calibration', 'scope', 'reproduction'),
        (draw_protection, draw_decisions, draw_calibration, draw_scope, draw_replay), axes)}
    fig.text(.035, .982, 'Evidence-constrained blueprint capabilities', fontsize=11, fontweight='bold',
             color=COLORS['ink'], va='top')
    fig.text(.035, .956, 'Ten development drugs | Engineering checks and clinical accuracy are reported separately',
             fontsize=7.5, color=COLORS['gray'], va='top')
    fig.canvas.draw()
    qa = audit_layout(fig)
    if any(level == 'FAIL' for level, _ in qa):
        raise ValueError('Capability figure layout failed: ' + repr(qa))
    details['layout_checks'] = qa
    return fig, details


def result_table(rows):
    """Five capability rows; counts and clinical estimates keep their own units."""
    expected = ['Evidence protection', 'Missing-input completion',
                'Restricted calibration', 'Requested-use scope', 'Reproducibility']
    if [row['capability'] for row in rows] != expected:
        raise ValueError('The five capability dimensions must be reported separately')
    text = [r'% Values come only from the completed capability experiment.',
        r'\begingroup\captionsetup[table]{name=Table}',
        r'\definecolor{CapInk}{HTML}{233A4B}\definecolor{CapPale}{HTML}{EDF2F5}',
        r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.2}',
        r'\fontsize{8.5}{11}\selectfont',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{29mm}'
        r'>{\raggedright\arraybackslash}p{35mm}'
        r'>{\raggedright\arraybackslash}p{57mm}'
        r'>{\raggedright\arraybackslash}p{48mm}@{}}',
        r'\caption{Five distinct capabilities of evidence-constrained blueprint construction: engineering checks, clinical prediction and computational reproduction.}'
        r'\label{tab:blueprint_capability}\\\toprule',
        r'\rowcolor{CapPale}\textbf{Capability} & \textbf{Evaluation unit} & '
        r'\textbf{Observed result} & \textbf{Interpretation boundary}\\\midrule\endfirsthead',
        r'\toprule\rowcolor{CapPale}\textbf{Capability} & \textbf{Evaluation unit} & '
        r'\textbf{Observed result} & \textbf{Interpretation boundary}\\\midrule\endhead',
        r'\bottomrule\endfoot\bottomrule\endlastfoot']
    for index, row in enumerate(rows):
        if index % 2 == 0:
            text.append(r'\rowcolor{CapPale}')
        text.append(' & '.join(tex_escape(row[k]) for k in
            ('capability', 'unit', 'result', 'boundary')) + r'\\\addlinespace[2pt]')
    text += [r'\end{longtable}\endgroup', '']
    return '\n'.join(text)


def table_rows(data):
    protection = data.protection
    fu = [r for r in protection if r['parameter'] == 'fraction_unbound']
    renal = [r for r in protection if r['parameter'] == 'renal_clearance']
    def gmfes(scenario, method):
        return '/'.join(format(finite(data.estimates(scenario, sampling, method)['curve_gmfe']['estimate'], 'GMFE'), '.3f')
                        for sampling in ('full', 'five'))
    completion = [r for r in data.decisions if r['scenario'] in ('missing_fu', 'wrong_species_fu')
                  and r['method'] == 'evidence_policy']
    fixed = [r for r in data.decisions if r['scenario'] in ('missing_fu', 'wrong_species_fu')
             and r['method'] == 'fixed_template']
    active = Counter(len(r.get('active_parameters', [])) for r in data.calibration
                     if r['method'] == 'adaptive_axes' and r['status'] == 'predicted')
    active_text = ', '.join(f'{n}: {count}/20' for n, count in sorted(active.items()))
    scope = data.applicability
    admitted = [r for r in scope if r['expected_supported']]
    negative = [r for r in scope if not r['expected_supported']]
    maxdiff = max((r['max_absolute_difference_umol_l'] for r in data.replay
                   if r.get('max_absolute_difference_umol_l') is not None), default=None)
    policy_order = [r for r in data.permutations if r['method'] == 'evidence_policy']
    noguard = [r for r in data.permutations if r['method'] == 'no_conflict_guard' and r['scenario'] == 'conflicting_fu']
    stable = lambda rows: sum(r['status_equal'] and r['selected_values_equal'] for r in rows)
    risk = data.tail_gate['by_method_sampling']
    warning_text = '; '.join(f'{label} ' + ', '.join(
        f'{risk[method + "__" + sampling]["missed_large_errors"]}/'
        f'{risk[method + "__" + sampling]["large_matched_errors"]}'
        for sampling in ('full', 'five')) for method, label in
        [('always_three', 'three-scale'), ('adaptive_axes', 'selected-scale')])
    return [
        {'capability': 'Evidence protection', 'unit': f'{len({r["case_id"] for r in fu})} drugs with qualified fu; '
            f'{len({r["case_id"] for r in renal})} with renal CL; {len(protection)} mapping checks',
         'result': f'fu preserved {sum(r["passed"] for r in fu)}/{len(fu)}; renal CL preserved '
                   f'{sum(r["passed"] for r in renal)}/{len(renal)} after nonrenal calibration.',
         'boundary': 'Construction input preservation does not establish the true target-patient CL or validate renal excretion.'},
        {'capability': 'Missing-input completion', 'unit': '20 engineered missing/nonhuman-fu requests; full and five-point predictions',
         'result': f'Evidence policy ready {sum(r["status"] == "ready" for r in completion)}/{len(completion)}; '
                   f'fixed inputs ready {sum(r["status"] == "ready" for r in fixed)}/{len(fixed)}. '
                   f'Missing-fu GMFE {gmfes("missing_fu", "evidence_policy")} versus '
                   f'available-input GMFE {gmfes("available", "evidence_policy")} (full/five). '
                   f'Hidden-fu recovery GMFE {data.fu_recovery["gmfe"]:.3f}; '
                   f'{data.fu_recovery["within_twofold"]}/{data.fu_recovery["n_drugs"]} within twofold.',
         'boundary': 'Four existing fu evidence values are hidden; six drugs are unchanged controls. Similar curve error does not imply accurate recovered fu or improved clinical accuracy.'},
        {'capability': 'Restricted calibration', 'unit': '20 adaptive mappings versus 20 frozen three-scale mappings',
         'result': f'Active parameter count (mappings): {active_text}. Target GMFE '
                   f'{gmfes("calibration", "adaptive_axes")} versus {gmfes("calibration", "always_three")} (full/five).',
         'boundary': f'Development-only selection equals the point-count cap in '
            f'{data.n_only_equivalence["matching_formula_fits"]}/{data.n_only_equivalence["checked_formula_fits"]} formula fits. '
            'No separate Jacobian-screening benefit or global identifiability is established.'},
        {'capability': 'Requested-use scope', 'unit': '110 plasma requests; 50 scope negatives; locked numerical-tail warnings',
         'result': f'Admissible plasma requests {sum(r["supported"] for r in admitted)}/{len(admitted)}; '
                   f'out-of-scope requests blocked {sum(not r["supported"] for r in negative)}/{len(negative)}. '
                   f'Missed matched FE > 2 events: {warning_text} (full, five).',
         'boundary': 'The fixed scope gate is shared by the comparator. Dense-tail and clinical-NCA windows differ; warning association is not clinical-confidence validation.'},
        {'capability': 'Reproducibility', 'unit': f'{len(data.replay)} locked-trajectory replays across both calibration rules; '
            '80 order reversals across four policies',
         'result': f'Replay passed {sum(r["passed"] for r in data.replay)}/{len(data.replay)}; maximum absolute '
                   f'difference {number(maxdiff)} micromol/L. Evidence-policy decision/value order consistency '
                   f'{stable(policy_order)}/{len(policy_order)}; conflict-guard ablation '
                   f'{stable(noguard)}/{len(noguard)} on conflicts.',
         'boundary': 'The solver uses frozen physiology and parameters, without refitting or a new native build. Numerical agreement is not proof of biological validity.'},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    data = Data(args.run_dir)
    if args.validate_only:
        print(json.dumps({'validated': True, 'drugs': len(data.cases), 'engineering_decisions': len(data.decisions),
            'logical_clinical_units': len(data.units), 'unique_scored_predictions': len(data.scored),
            'request_scope_checks': len(data.applicability), 'trajectory_replays': len(data.replay),
            'source_pins': len(data.store.pins)}))
        return
    figure_style()
    profile = data.profile()
    figure, details = draw(data)
    output = data.run / 'paper'; output.mkdir(exist_ok=True)
    cache = ROOT / 'runtime/cache/blueprint_capability'; cache.mkdir(parents=True, exist_ok=True)
    exports = export_figures([figure], output / 'capability_figure.pdf', cache)
    rows = table_rows(data)
    (output / 'capability_table.tex').write_text(result_table(rows))
    data.store.unchanged()
    for source in (Path(__file__), Path(__file__).with_name('render_vascular_tissue_ten_drug.py')):
        data.store.checksum(str(source))
    manifest = {'version': VERSION, 'experiment': relative(data.run),
        'scope': data.result['scope'], 'source_pins': data.store.pins,
        'profile': profile, 'figure_data': details, 'table_rows': rows,
        'figure_export': exports, 'outputs': []}
    for name in ('capability_figure.pdf', 'capability_table.tex'):
        path = output / name
        data.store.checksum(str(path))
        manifest['outputs'].append({'path': relative(path), 'sha256': data.store.pins[relative(path)]})
    (output / 'publication_manifest.json').write_text(json.dumps(json_plain(manifest), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'figure': relative(output / 'capability_figure.pdf'), 'table': relative(output / 'capability_table.tex'),
        'manifest': relative(output / 'publication_manifest.json'), 'source_pins': len(data.store.pins)}, indent=2))


if __name__ == '__main__':
    main()
