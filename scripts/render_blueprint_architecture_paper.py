#!/usr/bin/env python3
"""Publish one completed four-drug architecture experiment without fitting.

Only paper/main_figure.pdf, paper/endpoints.tex and
paper/publication_manifest.json are published. Preview images and a temporary
standalone table compilation stay in runtime/cache/architecture_paper.
The candidate is explicit; methods and formulas are never selected by target
error. All source comparisons remain in the sealed experiment results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from report_blueprint_validation import Store, _case_reference, _score_prediction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'experiments/blueprint_design/architecture_20260930/clinical_transport_v1'
VERSION = 'four-drug-architecture-publication-3'
DRUGS = ('Cefepime', 'Edaravone', 'Moxifloxacin', 'Remdesivir')
SAMPLING = ('full', 'five')
LABELS = {'extent_exchange_clearance': 'Native PBPK, three axes',
          'flow_exchange': 'Vascular-tissue exchange',
          'hepatic_exchange': 'Hepatic-route PBPK',
          'residence_exchange': 'Shared-residence PBPK',
          'two_compartment': 'Two-compartment',
          'two_compartment_prior': 'Two-compartment + CL prior'}
ENDPOINTS = {'auc': r'AUC', 'cmax': r'$C_{\max}$', 'tmax': r'$T_{\max}$',
             'c_last': r'$C_{\mathrm{last}}$', 'terminal_half_life': r'$t_{1/2}$',
             'auc_inf': r'AUC$_\infty$', 'clearance': 'CL', 'vz': r'$V_z$',
             'mrt': 'MRT', 'vss': r'$V_{\mathrm{ss}}$'}
FIGSIZE = (7.2, 8.5)
SKILL = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_plain(value):
    if isinstance(value, dict):
        return {key: json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def same_number(actual, expected, context):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('Missing value differs: ' + context)
    elif (isinstance(actual, bool) or isinstance(expected, bool)
          or not math.isfinite(float(actual)) or not math.isfinite(float(expected))
          or not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError('Numerical source mismatch: ' + context)


def pinned_references(value, store):
    """Hash nested artifacts without opening their JSON contents."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            if not re.fullmatch('[0-9a-f]{64}', str(value['sha256'])):
                raise ValueError('Invalid SHA256 reference')
            store.checksum(value['path'], value['sha256'])
        for key, child in value.items():
            if key in ('source_pins', 'code_pins', 'retained_artifact_pins'):
                if not isinstance(child, dict):
                    raise ValueError('Expected a source pin mapping')
                for path, expected in child.items():
                    store.checksum(path, expected)
            else:
                pinned_references(child, store)
    elif isinstance(value, list):
        for child in value:
            pinned_references(child, store)


def require_completed(state):
    if state.get('status') != 'complete' or any(state.get(k) is not True for k in (
            'formulas_locked_before_target_prediction', 'all_attempts_locked_before_scoring',
            'monitor_stopped_before_scoring')):
        raise ValueError('Completed experiment and every pre-scoring lock are required')


def validate_matrix(protocol, state, result, selection):
    """No target observations are opened until this structural gate passes."""
    if not protocol.get('version') or any(x.get('version') != protocol['version'] for x in (state, result)):
        raise ValueError('Experiment versions disagree')
    if result.get('all_attempts_locked_before_scoring') is not True:
        raise ValueError('Results lack the prediction-lock declaration')
    cases = protocol['cases']
    if (len(cases) != 4 or sorted(c['drug'] for c in cases) != sorted(DRUGS)
            or len({c['case_id'] for c in cases}) != 4 or len({c['cluster_id'] for c in cases}) != 4):
        raise ValueError('This publication is restricted to the four declared drug/study cases')
    if protocol.get('sampling') != list(SAMPLING):
        raise ValueError('Full and five sampling conditions are required')
    methods = protocol['methods']
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Duplicate or empty method list')
    expected = {(c['case_id'], s, m) for c in cases for s in SAMPLING for m in methods}
    declared = {(u['case_id'], u['sampling'], u['method']): u['id'] for u in protocol['units']}
    if len(declared) != len(protocol['units']) or set(declared) != expected or len(set(declared.values())) != len(expected):
        raise ValueError('Protocol comparison matrix is incomplete or duplicated')
    if set(state['computations']) != set(declared.values()):
        raise ValueError('Every declared prediction attempt must be retained')
    lookup = {(u['case_id'], u['sampling'], u['method']): u for u in result['units']}
    if len(lookup) != len(result['units']) or set(lookup) != expected:
        raise ValueError('Results matrix is incomplete or duplicated')
    by_id = {c['case_id']: c for c in cases}
    for key, record in lookup.items():
        cid, _, _ = key
        case = by_id[cid]
        if record['unit_id'] != declared[key] or any(record[k] != case[k] for k in ('drug', 'study_id', 'cluster_id')):
            raise ValueError('Result case or study identity differs')
        computation = state['computations'][record['unit_id']]
        if computation['status'] not in ('predicted', 'failed') or record['attempt_status'] != computation['status']:
            raise ValueError('Prediction attempt status differs')
        if computation['status'] == 'predicted' and record['prediction_source'] != computation['prediction']:
            raise ValueError('Scored and locked prediction sources differ')
        if (record.get('curve_metrics', {}).get('gmfe') is not None
                and (record['attempt_status'] != 'predicted' or record.get('scoring_status') != 'scored')):
            raise ValueError('A failed or unscored attempt cannot have a curve score')
    new_units = set(protocol['computations'])
    new_methods = {c['method'] for c in protocol['computations'].values()}
    planned = {(c['case_id'], c['sampling'], c['method']) for c in protocol['computations'].values()}
    if (len(planned) != len(new_units) or planned != {(cid, s, m) for cid in by_id for s in SAMPLING for m in new_methods}
            or new_units != {declared[key] for key in planned}):
        raise ValueError('New methods must retain the entire four-drug sampling matrix')
    expected_fits = {uid + '__' + f for uid in new_units for f in protocol['formulas']}
    if set(state['fits']) != expected_fits:
        raise ValueError('Every planned formula fit must be retained')
    if selection.get('basis') != 'development_objective_only' or set(selection['choices']) != new_units:
        raise ValueError('Development-only formula selection is incomplete')
    for uid, choice in selection['choices'].items():
        computation = state['computations'][uid]
        if choice is None:
            if computation['status'] != 'failed':
                raise ValueError('A missing formula choice cannot produce a prediction')
        elif (choice['formula'] not in protocol['formulas']
              or choice['mapping'] != state['fits'][uid + '__' + choice['formula']]['mapping']
              or computation.get('mapping') != choice['mapping']):
            raise ValueError('Target mapping differs from the locked development choice')
    return by_id, lookup


@dataclass
class PublicationData:
    root: Path
    run: Path
    store: Store
    protocol: dict
    state: dict
    result: dict
    cases: dict
    lookup: dict
    references: dict
    curves: dict
    summaries: dict
    methods: tuple
    labels: dict
    candidate: str
    eda: list
    candidate_role: str = 'new_candidate'
    adopted_source: dict | None = None


def validate_display_role(protocol, state, lookup, store, candidate, role):
    """Adoption changes the display, never the experiment or its selected formulas."""
    if role not in ('new_candidate', 'adopted_baseline'):
        raise ValueError('Unknown publication candidate role')
    new_methods = {c['method'] for c in protocol['computations'].values()}
    if role == 'new_candidate':
        if candidate not in new_methods:
            raise ValueError('Candidate must belong to the declared new experiment methods')
        return None
    if candidate != 'flow_exchange' or candidate in new_methods:
        raise ValueError('Adopted baseline must be the explicitly retained finite-exchange method')
    original = protocol
    visited = set()
    while candidate not in {c['method'] for c in protocol['computations'].values()}:
        refs = {k: protocol['source_' + k] for k in ('protocol', 'state', 'results')}
        token = refs['protocol']['sha256']
        if token in visited:
            raise ValueError('Cyclic frozen-baseline source chain')
        visited.add(token)
        protocol, source_state, result = (store.ref(refs[k]) for k in ('protocol', 'state', 'results'))
        require_completed(source_state)
        if source_state['protocol'] != refs['protocol'] or source_state['results'] != refs['results']:
            raise ValueError('Frozen baseline source-state references differ')
        for source in (protocol, source_state, result):
            pinned_references(source, store)
        _, source_lookup = validate_matrix(protocol, source_state, result,
                                           store.ref(source_state['formula_selection']))
    if protocol['version'] != 'blueprint-transport-pilot-1':
        raise ValueError('Adopted finite-exchange method must originate in the frozen transport experiment')
    if protocol['cases'] != original['cases'] or protocol['sampling'] != original['sampling']:
        raise ValueError('Adopted baseline case matrix changed')
    for key, record in lookup.items():
        if key[2] != candidate:
            continue
        source_record = source_lookup[key]
        current = state['computations'][record['unit_id']]
        source = source_state['computations'][source_record['unit_id']]
        for field in ('status', 'mapping', 'request', 'selected_formula', 'prediction'):
            if current.get(field) != source.get(field):
                raise ValueError('Adopted baseline differs from locked original: ' + field)
        for field in ('curve_metrics', 'endpoints', 'clinical_references', 'prediction_source'):
            if record.get(field) != source_record.get(field):
                raise ValueError('Adopted baseline differs from original scoring: ' + field)
    return refs


def display_methods(protocol, candidate):
    """Publish the declared method and fixed controls, not every later experiment."""
    methods = tuple(dict.fromkeys(('extent_exchange_clearance', 'flow_exchange', candidate,
                                  'two_compartment', 'two_compartment_prior')))
    if any(m not in protocol['methods'] for m in methods):
        raise ValueError('A required frozen comparator or candidate is absent')
    return methods


def load_data(run, *, root=ROOT, candidate='flow_exchange', candidate_label=None,
              candidate_role='new_candidate'):
    root = Path(root).resolve()
    store = Store(root)
    run = store.path(str(run))
    if run.is_symlink() or not run.is_dir():
        raise ValueError('A completed project experiment directory is required')
    state = store.read(str(run / 'state.json'))
    require_completed(state)
    for key, name in (('protocol', 'protocol.json'), ('results', 'results.json')):
        if store.path(state[key]['path']) != run / name:
            raise ValueError('State points to another experiment ' + key)
    protocol = store.ref(state['protocol'])
    # Hash all frozen sources before decoding any target observations.
    pinned_references(protocol, store)
    pinned_references(state, store)
    result = store.ref(state['results'])
    pinned_references(result, store)
    selection = store.ref(state['formula_selection'])
    cases, lookup = validate_matrix(protocol, state, result, selection)
    labels = dict(LABELS)
    if candidate_label:
        if not candidate_label.isascii() or not candidate_label.strip():
            raise ValueError('Candidate label must be nonempty English text')
        labels[candidate] = candidate_label
    elif candidate not in labels:
        raise ValueError('A new candidate requires an explicit --candidate-label')
    methods = display_methods(protocol, candidate)
    adopted_source = validate_display_role(protocol, state, lookup, store, candidate, candidate_role)
    references = {cid: _case_reference(case, store) for cid, case in cases.items()}
    curves = {}
    for cid in cases:
        clinical, observed, endpoints = references[cid]
        for sampling in SAMPLING:
            for method in methods:
                key = (cid, sampling, method)
                record = lookup[key]
                if record['clinical_references'] != endpoints:
                    raise ValueError('Recorded clinical references differ from the original source')
                if record['attempt_status'] != 'predicted':
                    continue
                computation = state['computations'][record['unit_id']]
                scored, _, _ = _score_prediction(cases[cid], clinical, observed, endpoints, computation, store)
                for field in ('gmfe', 'ln_rmse', 'n'):
                    same_number(scored['curve_metrics'].get(field), record['curve_metrics'].get(field), field)
                for endpoint in ENDPOINTS:
                    fresh, saved = scored['endpoints'][endpoint], record['endpoints'][endpoint]
                    for field in ('value', 'error', 'reference'):
                        same_number(fresh[field], saved[field], endpoint + '/' + field)
                    for field in ('unit', 'error_metric', 'weak_tail'):
                        if fresh[field] != saved[field]:
                            raise ValueError('Endpoint protocol metadata differs: ' + endpoint + '/' + field)
                if scored['curve_metrics'].get('gmfe') is not None and record['scoring_status'] != 'scored':
                    raise ValueError('Scoring status is inconsistent with the locked curve')
                curves[key] = scored['curve']
    summaries, eda = {}, []
    # Check the entire comparison matrix, including methods omitted from the figure.
    for sampling in SAMPLING:
        for method in protocol['methods']:
            records = [lookup[cid, sampling, method] for cid in cases]
            values = [u.get('curve_metrics', {}).get('gmfe') for u in records]
            good = [float(v) for v in values if v is not None]
            if any(not math.isfinite(v) or v < 1 - 1e-12 for v in good):
                raise ValueError('Invalid curve GMFE')
            mean = math.exp(math.fsum(map(math.log, good)) / len(good)) if good else None
            summary = result['statistics']['by_sampling'][sampling][method]['curve']
            same_number(mean, summary['estimate'], 'drug-equal mean')
            if summary['attempted'] != 4 or summary['scored'] != len(good):
                raise ValueError('Aggregate denominator disagrees with the four cases')
            summaries[sampling, method] = mean if len(good) == 4 else None
            eda.extend({'drug': u['drug'], 'sampling': sampling, 'method': method,
                        'curve_gmfe': v} for u, v in zip(records, values))
    store.unchanged()
    return PublicationData(root, run, store, protocol, state, result, cases, lookup,
                           references, curves, summaries, methods, labels, candidate, eda,
                           candidate_role, adopted_source)


def tex_escape(value):
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}.get(c, c) for c in str(value))


def tex_number(value, weak=False):
    if value is None:
        return r'\textemdash{}'
    if not math.isfinite(float(value)):
        raise ValueError('Cannot publish a nonfinite number')
    number = format(float(value), '.4g')
    if 'e' in number:
        base, exponent = number.split('e')
        number = base + r'\!\times\!10^{' + str(int(exponent)) + '}'
    return r'\(' + number + (r'^{\dagger}' if weak else '') + r'\)'


def table_text(data):
    candidate = tex_escape(data.labels[data.candidate])
    method_column = 'Adopted' if getattr(data, 'candidate_role', 'new_candidate') == 'adopted_baseline' else 'Candidate'
    lines = [r'% Generated only from the sealed four-drug experiment; no fitted parameters are changed.',
             r'\begingroup', r'\captionsetup[table]{name=Table}',
             r'\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.12}',
             r'\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}',
             r'\fontsize{8.5}{10.5}\selectfont',
             r'\begin{longtable}{@{}p{31mm}p{22mm}*{5}{>{\raggedleft\arraybackslash}p{22mm}}@{}}',
             r'\caption{Four-drug exploratory architecture assessment: ' + candidate
             + r' and frozen two-compartment predictions under full and five-point development inputs.}'
             + r'\label{tab:blueprint_architecture_endpoints}\\',
             r'\toprule Endpoint & Unit & Clinical $O$ & ' + method_column + r' $P^*$ & Error & Two-comp. $P^*$ & Error \\',
             r'\midrule\endfirsthead',
             r'\multicolumn{7}{@{}l}{Four-drug architecture assessment (continued)}\\',
             r'\toprule Endpoint & Unit & Clinical $O$ & ' + method_column + r' $P^*$ & Error & Two-comp. $P^*$ & Error \\',
             r'\midrule\endhead\bottomrule\endfoot\bottomrule\endlastfoot']
    for drug in DRUGS:
        cid = next(cid for cid, c in data.cases.items() if c['drug'] == drug)
        case = data.cases[cid]
        for sampling in SAMPLING:
            new = data.lookup[cid, sampling, data.candidate]
            old = data.lookup[cid, sampling, 'two_compartment']
            title = drug + ' / ' + ('Full development curve' if sampling == 'full' else 'Five development points')
            lines += [r'\addlinespace[4pt]\multicolumn{7}{@{}l}{\textbf{' + tex_escape(title) + r'}}\\*',
                      r'\multicolumn{7}{@{}l}{\footnotesize Target study: ' + tex_escape(case['study_id']) + r'}\\*']
            gmfe = lambda record: tex_number(record.get('curve_metrics', {}).get('gmfe'))
            n = len(data.references[cid][1]['time_h'])
            lines.append(r'Curve GMFE & dimensionless & ' + tex_escape(str(n) + ' points')
                         + ' & ' + r'\textemdash{} & ' + gmfe(new)
                         + ' & ' + r'\textemdash{} & ' + gmfe(old) + r'\\*')
            for endpoint, name in ENDPOINTS.items():
                reference = data.references[cid][2][endpoint]
                a = new.get('endpoints', {}).get(endpoint, {})
                b = old.get('endpoints', {}).get(endpoint, {})
                # Missing references stay blank, notably C_last; predictions do not create labels.
                cells = [name, tex_escape(reference.get('unit') or a.get('unit') or b.get('unit') or ''),
                         tex_number(reference['value'])]
                for estimate in (a, b):
                    weak = estimate.get('weak_tail', False)
                    cells += [tex_number(estimate.get('value'), weak),
                              tex_number(estimate.get('error') if reference['value'] is not None else None, weak)]
                lines.append(' & '.join(cells) + r'\\*')
            missing = [ENDPOINTS[e] for e in ENDPOINTS if data.references[cid][2][e]['value'] is None]
            if missing:
                lines.append(r'\multicolumn{7}{@{}p{174mm}@{}}{\footnotesize No qualified reported scalar reference: '
                             + ', '.join(missing) + r'. Values are not imputed.}\\')
            else:
                lines.append(r'\multicolumn{7}{@{}l}{}\\')
    notes = [
        r'$P^*$ denotes prediction-only clinical-protocol NCA; it is not a long-horizon native PBPK endpoint.',
        r'Error is $\max(P^*/O,O/P^*)$ except $T_{\max}$, which uses absolute error in the stated time unit. Curve error is GMFE.',
        r'$\dagger$ marks a weak predicted terminal estimate under the frozen protocol. It remains visible and scored when a reference exists.',
        r'Four previously viewed development cases; not blinded validation and not an update of all ten historical drugs.',
        r'Full/five refer to development-curve sampling plus the same frozen PK evidence. Moxifloxacin retains a prior from its development study.',
        r'Historical predictions are reused. The source results retain every method, failure and source reference. Within-run parameters and partition formulas use development data only.']
    if getattr(data, 'candidate_role', 'new_candidate') == 'adopted_baseline':
        notes.append(r'Finite tissue exchange is provisionally adopted for continued development across all four drugs. This post-development choice is not independent validation or per-drug model selection.')
    for note in notes:
        lines.append(r'\multicolumn{7}{@{}p{174mm}@{}}{\footnotesize ' + note + r'}\\')
    lines += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(lines)


def figure(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
    fig = plt.figure(figsize=FIGSIZE)
    colors = {'full': '#0072B2', 'five': '#D55E00'}
    styles = {'full': '-', 'five': '--'}
    markers = ('o', 's', '^', 'P')
    fig.text(.5, .99, 'Four-drug architecture assessment', ha='center', va='top', fontsize=11, weight='bold')
    fig.text(.5, .95, data.labels[data.candidate] + ': locked target-study predictions', ha='center', fontsize=8.5)
    handles = [Line2D([], [], color='black', marker='o', ls='', mfc='white', label='Observed target')]
    handles += [Line2D([], [], color=colors[s], ls=styles[s], label='Prediction: ' + s) for s in SAMPLING]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False, fontsize=8)
    for i, drug in enumerate(DRUGS):
        cid = next(cid for cid, c in data.cases.items() if c['drug'] == drug)
        left, bottom = (.095 if i % 2 == 0 else .595), (.70 if i < 2 else .405)
        ax = fig.add_axes([left, bottom, .36, .185])
        observed = data.references[cid][1]
        if observed['unit'] != 'ng/mL':
            raise ValueError('This figure currently requires original curves in ng/mL')
        window = data.cases[cid]['target_request_metadata']['window_end_h']
        for sampling in SAMPLING:
            curve = data.curves.get((cid, sampling, data.candidate))
            if curve is not None:
                y = np.asarray(curve['concentration'], float)
                time = np.asarray(curve['time_h'], float)
                shown = (time > 0) & (time <= window)
                ax.plot(time[shown], np.where(y[shown] > 0, y[shown], np.nan), styles[sampling],
                        color=colors[sampling], lw=1.3)
            else:
                ax.text(.98, .1 if sampling == 'full' else .2, sampling + ': FAIL/NA',
                        transform=ax.transAxes, ha='right', fontsize=8, color=colors[sampling])
        t, y = np.asarray(observed['time_h']), np.asarray(observed['concentration'])
        ax.scatter(t[y > 0], y[y > 0], s=17, facecolors='white', edgecolors='black', lw=.7, zorder=4)
        ax.set_yscale('log')
        ax.set_xlim(0, window * 1.03)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.grid(axis='y', which='major', color='.88', lw=.4)
        ax.set_xlabel('Time (h)', fontsize=8)
        ax.set_ylabel('Plasma concentration (ng/mL, log)', fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=7)
        dose = data.cases[cid]['target_request_metadata']['dosing']
        ax.set_title(drug + '\n' + f"{dose['dose_mg']:g} mg IV; {dose['infusion_minutes']:g}-min infusion", fontsize=8, pad=7)
        fig.text(left-.075, bottom+.20, 'abcd'[i], fontsize=10, weight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    # n=4: show each drug, with a separate diamond for its equal-weight geometric mean.
    for i, sampling in enumerate(SAMPLING):
        left = .29 if i == 0 else .665
        ax = fig.add_axes([left, .14, .275, .16])
        for j, method in enumerate(data.methods):
            for k, drug in enumerate(DRUGS):
                cid = next(cid for cid, c in data.cases.items() if c['drug'] == drug)
                value = data.lookup[cid, sampling, method].get('curve_metrics', {}).get('gmfe')
                if value is not None:
                    ax.scatter(value, j + (k-1.5)*.085, marker=markers[k], s=20,
                               facecolors='none', edgecolors=colors[sampling], linewidths=.8)
            mean = data.summaries[sampling, method]
            if mean is None:
                ax.text(1.01, j, 'NA', va='center', fontsize=7)
            else:
                ax.scatter(mean, j, marker='D', s=16, color='black', zorder=5)
        ax.axvline(2, color='.65', ls=':', lw=.7)
        ax.set_xscale('log')
        vals = [u['curve_metrics']['gmfe'] for key, u in data.lookup.items()
                if key[2] in data.methods and u.get('curve_metrics', {}).get('gmfe') is not None]
        upper = max(3, max(vals)*1.08)
        ax.set_xlim(.97, upper)
        ticks = [v for v in (1, 1.5, 2, 3, 4, 6, 8, 12, 16) if v <= upper]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_yticks(range(len(data.methods)), [data.labels[m] for m in data.methods] if i == 0 else [])
        ax.set_ylim(len(data.methods)-.5, -.5)
        ax.tick_params(axis='y', length=0, labelsize=7.5)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(('Full' if sampling == 'full' else 'Five-point') + ' development inputs', fontsize=8)
        ax.set_xlabel('Target-curve GMFE (log; lower is better)', fontsize=7.5)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.text(.04 if i == 0 else .61, .32, 'ef'[i], fontsize=10, weight='bold')
    drug_handles = [Line2D([], [], marker=m, ls='', color='.25', mfc='none', markersize=4, label=d)
                    for d, m in zip(DRUGS, markers)]
    drug_handles += [Line2D([], [], marker='D', ls='', color='black', markersize=4, label='Geomean (n = 4)')]
    fig.legend(handles=drug_handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=5,
               columnspacing=1.0, handletextpad=.3, frameon=False, fontsize=7)
    fig.text(.5, .033, 'Previously viewed cohort; not blinded validation. Earlier target results informed model development.',
             ha='center', fontsize=7)
    fig.text(.5, .015, 'Full/five share frozen PK evidence; Moxifloxacin retains a prior from its development study.',
             ha='center', fontsize=7)
    return fig


def publish(data, *, preview_only=False):
    cache = data.root / 'runtime/cache/architecture_paper' / data.run.name
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache / 'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import pandas as pd
    from PIL import Image
    sys.path.insert(0, str(SKILL / 'scripts'))
    from profile_data import profile_data
    from setup_style import setup_style
    from visual_qa import audit_layout, render_preview
    profile = profile_data(pd.DataFrame(data.eda), group_cols=['sampling', 'method'])
    eda = {'comparison_units': len(data.eda), 'independent_drug_study_clusters': 4,
           'observed_curve_points': {data.cases[c]['drug']: len(v[1]['time_h']) for c, v in data.references.items()},
           'curve_gmfe': json_plain(profile['columns']['curve_gmfe']),
           'display': 'Four concentration-time panels and individual-drug GMFE dot plots; no significance test.'}
    setup_style(journal='nature', lang='en', use_sciplots=False)
    # A reused global font cache can contain paths from a removed environment.
    # Register the installed Matplotlib fonts without changing that shared cache.
    for font in (Path(matplotlib.get_data_path()) / 'fonts/ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.constrained_layout.use': False,
                         'pdf.fonttype': 42, 'savefig.bbox': None, 'font.size': 8})
    fig = figure(data)
    preview = cache / 'main_figure_preview.png'
    render_preview(fig, str(preview), dpi=180)
    # Preserve the exact final page frame; the skill's tight preview can crop
    # font ascenders and hides the physical margins used by the PDF.
    fig.savefig(preview, dpi=180, bbox_inches=None)
    with Image.open(preview) as source:
        source.convert('L').save(cache / 'main_figure_grayscale.png')
    issues = audit_layout(fig)
    data.store.unchanged()
    if any(level == 'FAIL' for level, _ in issues):
        plt.close(fig)
        raise ValueError('Figure QA failed: ' + repr(issues))
    qa = {'layout_issues': issues, 'preview': str(preview.relative_to(data.root)),
          'preview_sha256': digest(preview), 'figure_size_inches': list(FIGSIZE),
          'manual_visual_review': 'Inspect the color and grayscale previews before manuscript use.'}
    if preview_only:
        plt.close(fig)
        return {'EDA': eda, 'QA': qa, 'published': False}
    output = data.run / 'paper'
    if output.is_symlink():
        raise ValueError('Publication output cannot be a symbolic link')
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.publication-', dir=output) as temporary:
        temporary = Path(temporary)
        fig.savefig(temporary / 'main_figure.pdf', format='pdf', bbox_inches=None,
                    metadata={'Title': 'Four-drug architecture assessment',
                              'Subject': 'Previously viewed cohort; not blinded validation'})
        (temporary / 'endpoints.tex').write_text(table_text(data))
        plt.close(fig)
        from check_figure import check_figure
        file_qa, file_info = check_figure(str(temporary / 'main_figure.pdf'), target_inches=FIGSIZE)
        if any(level == 'FAIL' for level, _ in file_qa):
            raise ValueError('Published PDF QA failed: ' + repr(file_qa))
        qa['file_checks'] = file_qa
        qa['file_info'] = {**file_info, 'path': str((output/'main_figure.pdf').relative_to(data.root))}
        # The skill's font checker does not descend into Type0 CID fonts.
        # Independently extract every font program instead of ignoring its warning.
        import pymupdf
        with pymupdf.open(temporary / 'main_figure.pdf') as pdf:
            embedded = [{'name': font[3], 'type': font[2],
                         'embedded_bytes': len(pdf.extract_font(font[0])[-1])}
                        for font in pdf.get_page_fonts(0)]
            if len(pdf) != 1 or not embedded or any(
                    f['embedded_bytes'] == 0 or f['type'] == 'Type3' for f in embedded):
                raise ValueError('Independent PDF font/page verification failed')
            qa['embedded_font_verification'] = embedded
            qa['font_warning_resolution'] = (
                'Every font program is embedded; the skill warning concerns Type0 parent '
                'dictionaries whose font descriptors reside in their CID descendants.')
        data.store.unchanged()
        manifest = {'schema_version': VERSION, 'experiment_version': data.protocol['version'],
                    'run': str(data.run.relative_to(data.root)), 'candidate_method': data.candidate,
                    'candidate_role': data.candidate_role, 'adopted_original_source': data.adopted_source,
                    'method_selection_context': ('User-selected provisional development baseline; same method for every drug; no new fitting.'
                                                 if data.candidate_role == 'adopted_baseline' else 'Declared new experiment candidate'),
                    'candidate_label': data.labels[data.candidate], 'display_methods': list(data.methods),
                    'all_methods_retained_in_results': data.protocol['methods'],
                    'cases': [{k: c[k] for k in ('case_id', 'drug', 'study_id', 'cluster_id')}
                              for c in data.cases.values()],
                    'sampling': list(SAMPLING), 'target_observations_read_after_completion_and_lock_checks': True,
                    'prediction_or_parameter_fitting_performed': False,
                    'source_pins': dict(data.store.pins),
                    'renderer': {'path': str(Path(__file__).resolve().relative_to(data.root)), 'sha256': digest(__file__)},
                    'source_results': data.state['results'], 'source_protocol': data.state['protocol'],
                    'outputs': {name: {'path': str((output/name).relative_to(data.root)), 'sha256': digest(temporary/name)}
                                for name in ('main_figure.pdf', 'endpoints.tex')},
                    'data_profile': eda, 'figure_qa': qa,
                    'limitations': data.result['limitations'] + ['Four previously viewed drugs only; historical ten-drug figures are not updated.',
                        'P* is the frozen clinical-protocol estimate; native extended-horizon P is not supplied.',
                        'Missing clinical endpoints remain missing, including C_last; weak tails are retained.']}
        (temporary / 'publication_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        data.store.unchanged()
        for name in ('main_figure.pdf', 'endpoints.tex', 'publication_manifest.json'):
            (temporary / name).replace(output / name)
    return {'EDA': eda, 'QA': qa, 'published': str(output.relative_to(data.root))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--candidate-method', default='flow_exchange')
    parser.add_argument('--candidate-label')
    parser.add_argument('--candidate-role', choices=('new_candidate', 'adopted_baseline'), default='new_candidate')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args(argv)
    data = load_data(args.run_dir, candidate=args.candidate_method, candidate_label=args.candidate_label,
                     candidate_role=args.candidate_role)
    if args.validate_only:
        print(json.dumps({'validated_units': len(data.lookup), 'source_pins': len(data.store.pins),
                          'candidate': data.candidate, 'display_methods': data.methods}))
    else:
        print(json.dumps(publish(data, preview_only=args.preview_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
