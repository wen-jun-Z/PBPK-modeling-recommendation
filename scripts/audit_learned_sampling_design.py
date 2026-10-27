#!/usr/bin/env python3
"""Independent algebra/CSV/bootstrap checks and one compact comparison figure."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import run_learned_sampling_design as run

ROOT, RUN = run.ROOT, run.RUN


def close(a, b, label):
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(label + ': null mismatch')
    elif not np.allclose(a, b, rtol=1e-9, atol=1e-11):
        raise AssertionError(label + f': {a} != {b}')


def expected(values, arithmetic=False):
    if any(v is None for v in values):
        return None
    return float(np.mean(values)) if arithmetic else math.exp(float(np.log(values).mean()))


def independent_fit(rows, features, raw, train_drugs, goal, alpha):
    """Augmented least squares, separately implemented from the normal-equation learner."""
    xs, ys, ws = [], [], []
    for drug in sorted(train_drugs):
        keys = sorted(k for k, d in features.items() if d['case_id'] == drug)
        for key in keys:
            d = features[key]
            x = np.asarray(d['goals'][goal]['X']); y = []
            for cid in d['candidate_ids']:
                record = raw[(key, key + f'__point_{cid:02d}')]
                baseline = raw[(key, 'baseline')]
                y.append(math.log(baseline['endpoints'][goal]['error']) - math.log(record['endpoints'][goal]['error']))
            xs.extend(x - x.mean(axis=0)); ys.extend(np.asarray(y) - np.mean(y))
            ws.extend([1 / len(train_drugs) / len(keys) / len(x)] * len(x))
    x, y, w = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    scale = np.sqrt(np.sum(w[:, None] * x*x, axis=0)); scale[scale < 1e-12] = 1.
    z = x / scale
    a = np.vstack((np.sqrt(w)[:, None] * z, np.sqrt(alpha) * np.eye(x.shape[1])))
    b = np.r_[np.sqrt(w) * y, np.zeros(x.shape[1])]
    return scale, np.linalg.lstsq(a, b, rcond=None)[0]


def bootstrap(values, clusters, seed, count):
    names = sorted(set(clusters.values())); keys = sorted(values)
    logs = {k: math.log(values[k]) for k in keys}
    samples = np.random.default_rng(seed).integers(0, len(names), size=(count, len(names)))
    sums = np.array([sum(logs[k] for k in keys if clusters[k] == name) for name in names])
    sizes = np.array([sum(clusters[k] == name for k in keys) for name in names])
    draws = np.exp(np.sum(sums[samples], axis=1) / np.sum(sizes[samples], axis=1))
    return math.exp(np.mean(list(logs.values()))), np.quantile(draws, [.025, .975])


def load():
    s = run.read(RUN / 'state.json')
    if s['status'] != 'complete':
        raise ValueError('Complete choices and report required before audit')
    p = run.checked(s['protocol']); run.verify(p['source_pins'])
    r = run.checked(s['results']); f = run.checked(s['features']); d = run.checked(s['decisions'])
    with (RUN / 'endpoints.csv').open(newline='') as h:
        rows = list(csv.DictReader(h))
    if run.ref(RUN / 'endpoints.csv') != r['endpoint_csv']:
        raise AssertionError('Changed endpoint CSV')
    return s, p, r, f, d, rows


def audit():
    s, p, r, features, decisions, rows = load()
    raw = {(v['design_key'], v['candidate']): v for v in run.checked(p['cached_scores'])['records']}
    assert len(features) == 20 and len(decisions['folds']) == 9 and len(rows) == 5940
    assert s['choices_locked_at'] <= r['scored_at']
    assert run.base.sha256(ROOT / p['paper_before']['path']) == p['paper_before']['sha256']
    all_drugs = {c['case_id'] for c in p['cases']}; drugclusters = {c['case_id']: c['cluster_id'] for c in p['cases']}
    model_count = 0
    for cluster, fold in decisions['folds'].items():
        test = {drug for drug in all_drugs if drugclusters[drug] == cluster}
        train = all_drugs - test
        assert set(fold['heldout_case_ids']) == test and set(fold['train_case_ids']) == train
        assert fold['locked_at'] <= s['choices_locked_at']
        for goal, model in fold['models'].items():
            scale, coef = independent_fit(rows, features, raw, train, goal, p['ridge_alpha'])
            close(model['scale'], scale, 'Training-only scale'); close(model['coef'], coef, 'Augmented least squares')
            for key, d in features.items():
                if d['case_id'] not in test:
                    continue
                scores = np.asarray(d['goals'][goal]['X']) / scale @ coef
                idx = min(np.flatnonzero(scores >= max(scores) - 1e-12), key=lambda i: d['times'][i])
                choice = decisions['choices'][key][goal]
                close(choice['scores'], scores, 'Held-out scores'); assert choice['index'] == idx
            model_count += 1
    copt_count = 0
    for key, d in features.items():
        jv = np.asarray(d['visible_jacobian_whitened']); jc = np.asarray(d['candidate_jacobian_whitened'])
        fisher = jv.T @ jv + np.diag(d['regularization_precision'])
        for goal, info in d['goals'].items():
            g = np.asarray(info['goal_gradient'])
            before = float(g @ np.linalg.solve(fisher, g))
            after = [float(g @ np.linalg.solve(fisher + np.outer(j, j), g)) for j in jc]
            close(before, info['c_optimal_before'], 'Prior local variance')
            close(after, info['c_optimal']['scores'], 'Official criterion vs independent Fisher solve')
            copt_count += 1
    seen = set()
    for row in rows:
        key = row['case_id'] + '__' + row['regime']; d = features[key]
        ident = (key, row['requested_goal'], row['method'], row['endpoint'])
        assert ident not in seen; seen.add(ident)
        method, ep, goal = row['method'], row['endpoint'], row['requested_goal']
        candidates = [raw[(key, key + f'__point_{i:02d}')] for i in d['candidate_ids']]
        if method == 'no_addition':
            picked = [raw[(key, 'baseline')]]; idx = None
        elif method == 'uniform_random':
            picked = candidates; idx = None
        else:
            if method == 'learned_gain':
                idx = decisions['choices'][key][goal]['index']
            elif method == 'oracle':
                idx = int(np.argmin([v['endpoints'][goal]['error'] for v in candidates]))
            else:
                idx = d['goals'][goal]['choices'][method]
            picked = [candidates[idx]]
        assert row['candidate_index'] == ('' if idx is None else str(idx))
        error = expected([v['curve_metrics']['gmfe'] if ep == 'curve' else v['endpoints'][ep]['error'] for v in picked], ep == 'tmax')
        close(None if not row['error'] else float(row['error']), error, 'CSV error')
        if ep != 'curve':
            close(None if not row['reference'] else float(row['reference']), picked[0]['endpoints'][ep]['reference'], 'Clinical reference')
            close(None if not row['prediction'] else float(row['prediction']), None if len(picked) > 1 else picked[0]['endpoints'][ep]['value'], 'Actual predicted endpoint')
    def values(regime, goal, method, ep):
        return {row['case_id']: float(row['error']) for row in rows if row['regime'] == regime
                and row['requested_goal'] == goal and row['method'] == method and row['endpoint'] == ep and row['error']}
    stats_count = 0
    for regime in run.previous.REGIMES:
        for goal in run.GOALS:
            for method in run.METHODS:
                for label, ep in [('endpoint', goal), ('curve', 'curve')]:
                    v = values(regime, goal, method, ep)
                    est, ci = bootstrap(v, {k: drugclusters[k] for k in v}, p['bootstrap']['seed'], p['bootstrap']['resamples'])
                    out = r['summary'][regime][goal][method][label]
                    close(out['estimate'], est, 'GMFE'); close(out['ci95'], ci, 'Study bootstrap')
                    stats_count += 1
    v = values('late_four', 'cmax', 'learned_gain', 'cmax')
    timev = values('late_four', 'cmax', 'time_farthest', 'cmax')
    ratios = {k: v[k] / timev[k] for k in v}
    est, ci = bootstrap(ratios, drugclusters, p['bootstrap']['seed'], p['bootstrap']['resamples'])
    close(r['comparisons']['late_four']['cmax']['time_farthest']['estimate'], est, 'Paired effect')
    close(r['comparisons']['late_four']['cmax']['time_farthest']['ci95'], ci, 'Paired interval')
    report = dict(status='passed', audited_at=time.time(), protocol=s['protocol'], results=s['results'],
        source_hashes=len(p['source_pins']), feature_designs=20, independent_models=model_count,
        official_criteria_checked=copt_count, endpoint_rows=len(rows), independent_statistics=stats_count,
        paper_unchanged=True, cached_prediction_refits=0,
        main_drug_outcomes={c['drug']: dict(learned_cmax_fe=v[c['case_id']], time_cmax_fe=timev[c['case_id']],
                                         ratio=ratios[c['case_id']], outcome='improved' if ratios[c['case_id']] < 1 else 'worse' if ratios[c['case_id']] > 1 else 'same') for c in p['cases']},
        limits=['Checks implementation and arithmetic, not scientific generalizability.',
                'Existing cached PBPK clinical scores remain source-hash locked; no new native fitting or prospective observation.'])
    run.save(RUN / 'audit.json', report)
    return report


def plot(export=False):
    s, p, r, features, decisions, rows = load()
    audit_doc = run.read(RUN / 'audit.json')
    if audit_doc['results'] != s['results'] or audit_doc['status'] != 'passed':
        raise ValueError('Current numerical audit required')
    skill = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')
    sys.path.insert(0, str(skill))
    from setup_style import setup_style
    from profile_data import profile_data
    from visual_qa import audit_layout, render_preview
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    df = pd.DataFrame([{**row, 'error': float(row['error'])} for row in rows
                      if row['regime'] == 'late_four' and row['requested_goal'] == 'cmax' and row['endpoint'] == 'cmax'])
    profile = profile_data(df[['drug', 'method', 'error']], group_cols=['method'])
    setup_style(journal='nature', lang='en')
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.constrained_layout.use': False,
                         'savefig.bbox': None})
    # Primary task only; point/interval + drug-paired ratios, no mean bars.
    names = ['learned_gain', 'time_farthest', 'maximum_variance', 'c_optimal', 'request_directed',
             'linear_covariance', 'uniform_random', 'no_addition', 'oracle']
    labels = ['Learned gain (held out)', 'Time coverage', 'Concentration variance', 'Official c-optimal',
              'Nonlinear utility', 'Linear utility', 'Exact random expectation', 'No addition', 'Hindsight oracle']
    colors = ['#009E73', '#0072B2', '#757D85', '#D55E00', '#CC79A7', '#8A6E38', '#555555', '#999999', '#333333']
    fig = plt.figure(figsize=(7.2, 7.1)); a = fig.add_axes([.29, .57, .64, .29]); b = fig.add_axes([.29, .17, .64, .30])
    for y, (method, color) in enumerate(zip(names, colors)):
        st = r['summary']['late_four']['cmax'][method]['endpoint']; center = st['estimate']; low, high = st['ci95']
        a.errorbar(center, y, xerr=[[center-low], [high-center]], fmt='s' if method == 'learned_gain' else 'o',
                   color=color, capsize=2, markersize=4.5, linewidth=1.1)
    a.set_yticks(range(len(names)), labels); a.invert_yaxis(); a.set_xscale('log'); a.set_xlabel('Cmax GMFE (lower is better; log scale)')
    a.xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6])); a.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    a.minorticks_off(); a.grid(axis='x', color='#E3E8EC', linewidth=.5); a.set_axisbelow(True)
    a.set_title('a  Does learned sampling beat the same-budget controls?', loc='left', fontsize=9, pad=12)
    outcomes = audit_doc['main_drug_outcomes']; drugs = sorted(outcomes)
    for y, drug in enumerate(drugs):
        ratio = outcomes[drug]['ratio']; color = '#009E73' if ratio < 1 else '#D55E00' if ratio > 1 else '#777777'
        b.plot([1, ratio], [y, y], color=color, linewidth=1.1)
        b.scatter([ratio], [y], marker='s' if ratio < 1 else '^' if ratio > 1 else 'o', s=23, color=color, zorder=3)
    b.set_yticks(range(len(drugs)), drugs); b.invert_yaxis(); b.set_xscale('log')
    b.axvline(1, color='#555555', linewidth=.8, linestyle='--'); b.set_xlabel('Cmax fold-error ratio: learned / time (log scale)')
    b.set_xlim(min(.6, min(v['ratio'] for v in outcomes.values())*.93), max(1.2, max(v['ratio'] for v in outcomes.values())*1.07))
    b.xaxis.set_major_locator(FixedLocator([.5, .6, .7, .8, 1, 1.2, 1.4])); b.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    b.minorticks_off(); b.grid(axis='x', color='#E3E8EC', linewidth=.5); b.set_axisbelow(True)
    b.set_title('b  Benefit is not consistent across drugs', loc='left', fontsize=9, pad=12)
    fig.text(.04, .965, 'Learning which observation to add', fontsize=12, weight='bold')
    fig.text(.04, .935, '10 development drugs | 9 held-out study groups | 4 late observations + 1 acquired point', fontsize=8)
    fig.text(.04, .075, 'a: Drug-equal GMFE and 95% study-cluster bootstrap interval (2,000 draws).', fontsize=7)
    fig.text(.04, .055, 'b: One paired value per drug; <1 favors learned selection. 3 improve, 5 tie, 2 worsen.', fontsize=7)
    fig.text(.04, .035, 'Retrospective development screening; no prospective or zero-curve accuracy claim.', fontsize=7)
    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
    preview = RUN / 'comparison.png'
    render_preview(fig, str(preview), dpi=300)
    issues = audit_layout(fig)
    if any(level == 'FAIL' for level, message in issues):
        raise ValueError('Figure layout failed: ' + repr(issues))
    if export:
        fig.savefig(RUN / 'comparison.pdf', dpi=300)
        fig.savefig(preview, dpi=300)
    from PIL import Image
    gray = ROOT / 'runtime/test_tmp/learned_sampling_20261009/comparison_grayscale.png'
    gray.parent.mkdir(parents=True, exist_ok=True)
    Image.open(preview).convert('L').save(gray)
    audit_doc['figure'] = dict(rows=len(df), drugs=df['drug'].nunique(), methods=len(names),
                              missing_error=int(df['error'].isna().sum()),
                              error_range=[float(df['error'].min()), float(df['error'].max())],
                              layout_issues=issues, preview=run.ref(preview), exported=export,
                              script=run.ref(Path(__file__)))
    run.save(RUN / 'audit.json', audit_doc)
    plt.close(fig)
    return dict(preview=str(preview), grayscale=str(gray), issues=issues, exported=export)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'preview', 'export'], default='audit', nargs='?')
    args = parser.parse_args()
    result = audit() if args.command == 'audit' else plot(args.command == 'export')
    print(json.dumps(result, indent=2, ensure_ascii=False))
