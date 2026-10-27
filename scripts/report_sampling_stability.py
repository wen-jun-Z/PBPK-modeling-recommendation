#!/usr/bin/env python3
"""Independent completed-run audit and one thirty-drug stability figure.

Only state.json is opened before completion. This module imports no runner,
PBPK fitter, clinical scorer, or production sampling learner. Inspect the PNG
preview before requesting its matching PDF with ``plot --export``.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np

from report_sampling_expansion import (ROOT, SKILL, bootstrap, checked, digest,
                                       inside, json_native, read, ref, same)

RUN = ROOT / 'experiments/blueprint_design/sampling_stability_20261009'
METHODS = ('previous_learned', 'aligned_only', 'stable', 'time', 'stable_repaired',
           'time_repaired', 'no_addition', 'no_addition_repaired')
LABELS = ('Previous learned rule', 'Aligned labels only', 'Stable selector', 'Time rule',
          'Stable + feasible fitting', 'Time + feasible fitting', 'No added point',
          'No point + feasible fitting')
GOALS = ('cmax', 'auc_inf', 'terminal_half_life')
ENDPOINTS = ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life', 'auc_inf',
             'clearance', 'vz', 'mrt', 'vss')
TOLERANCE = 1e-10
MARGIN = 1e-12


def save(path, document):
    import json
    p = inside(path)
    if not p.is_relative_to(RUN):
        raise ValueError('Audit artifacts must remain in their experiment directory')
    p.write_text(json.dumps(json_native(document), ensure_ascii=False, indent=2,
                            allow_nan=False) + '\n')


def load_complete():
    try:
        state = read(RUN / 'state.json')
    except FileNotFoundError as exc:
        raise ValueError('Experiment state unavailable; no outcome reads permitted') from exc
    if state.get('status') != 'complete':
        raise ValueError('Experiment incomplete; no outcome reads permitted')
    protocol = checked(state['protocol'])
    for path, expected in protocol['source_pins'].items():
        if digest(path) != expected:
            raise ValueError('Changed frozen source: ' + path)
    result = checked(state['results'])
    if result['protocol'] != state['protocol']:
        raise ValueError('Result/protocol identity mismatch')
    if (set(state['units']) != set(protocol['jobs'])
            or result['prediction_lock'] != state['predictions_locked_at']
            or not state['predictions_locked_at'] <= result['scored_at'] <= state['completed_at']):
        raise ValueError('Prediction completion/scoring lock mismatch')
    path = inside(result['endpoints']['path'])
    if digest(path) != result['endpoints']['sha256']:
        raise ValueError('Changed endpoint CSV')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return state, protocol, result, rows


def training_rows(features, records, originals, goal, cases):
    """Rebuild eligible labels independently, including their original row order."""
    lookup = {}
    for r in records:
        if r['case_id'] not in originals:
            continue
        key = (r['design_key'], r['candidate'])
        if key in lookup:
            raise ValueError('Duplicate original-cohort outcome')
        lookup[key] = r
    rows = []
    for key, f in sorted(features.items()):
        cid = f['case_id']
        if cid not in originals:
            continue
        d = f['goals'][goal]
        if d.get('status') != 'selected':
            continue
        baseline = lookup[key, 'baseline']['endpoints'][goal]['error']
        outcomes = [lookup[key, key+f'__point_{i:02d}']['endpoints'][goal]['error'] for i in f['candidate_ids']]
        if baseline is None or any(value is None for value in outcomes):
            continue
        if len(d['X']) != len(outcomes):
            raise ValueError('Training menu length mismatch')
        for x, error in zip(d['X'], outcomes):
            if not all(math.isfinite(v) and v > 0 for v in (baseline, error)):
                raise ValueError('Invalid matched training error')
            rows.append(dict(x=x, y=math.log(baseline/error), design=key,
                             drug=cid, cluster=cases[cid]['cluster_id']))
    return rows


def independent_fit(rows, indices):
    """Fold-local augmented least squares, independent of production normal equations."""
    chosen = [rows[i] for i in indices]
    drugs = sorted({r['drug'] for r in chosen})
    x = np.asarray([r['x'] for r in chosen], dtype=float)
    y = np.asarray([r['y'] for r in chosen], dtype=float)
    w = np.empty(len(chosen))
    for drug in drugs:
        designs = sorted({r['design'] for r in chosen if r['drug'] == drug})
        for design in designs:
            menu = np.array([r['design'] == design for r in chosen])
            x[menu] -= x[menu].mean(axis=0)
            y[menu] -= y[menu].mean()
            w[menu] = 1/len(drugs)/len(designs)/menu.sum()
    scale = np.sqrt(np.sum(w[:, None]*x*x, axis=0))
    scale[scale < 1e-12] = 1.
    matrix = np.vstack((np.sqrt(w)[:, None]*x/scale, np.eye(x.shape[1])))
    response = np.r_[np.sqrt(w)*y, np.zeros(x.shape[1])]
    coef = np.linalg.lstsq(matrix, response, rcond=None)[0]
    membership = dict(row_indices=list(indices), drug_ids=drugs,
                      design_ids=sorted({r['design'] for r in chosen}),
                      cluster_ids=sorted({r['cluster'] for r in chosen}))
    return scale, coef, membership


def audit_models(protocol, lock, features, records):
    cases = {c['case_id']: c for c in protocol['cases']}
    originals = set(protocol['original_case_ids'])
    if len(cases) != 30 or len(originals) != 10 or not originals <= cases.keys():
        raise ValueError('Expected all thirty drugs and exactly original ten')
    bundles = checked(protocol['models'])
    if set(bundles) != {'cmax', 'terminal_half_life'}:
        raise ValueError('Only prespecified supported label goals may train')
    if not (lock['new_twenty_labels_used'] is False
            and lock['only_visible_before_acquisition'] is True
            and lock['predictor_repair_after_fixed_choices'] is True):
        raise ValueError('Locked selection isolation contract differs')
    trained, decisions, switches = 0, 0, 0
    for goal, by_exclusion in bundles.items():
        rows = training_rows(features, records, originals, goal, cases)
        if {r['drug'] for r in rows} != originals:
            raise ValueError('Incomplete original-ten model training')
        clusters = {r['cluster'] for r in rows}
        if set(by_exclusion) != clusters | {'__all__'}:
            raise ValueError('Missing nested whole-study folds')
        for outer, bundle in by_exclusion.items():
            excluded = None if outer == '__all__' else outer
            if bundle['excluded_cluster'] != excluded or bundle['alpha'] != 1.:
                raise ValueError('Changed outer exclusion/regularization')
            eligible = [i for i, r in enumerate(rows) if r['cluster'] != excluded]
            groups = sorted({rows[i]['cluster'] for i in eligible})
            folds = bundle['jackknife_models']
            if [f['excluded_cluster'] for f in folds] != groups:
                raise ValueError('Missing or duplicated inner study exclusions')
            todo = [(bundle['full_model'], bundle['training_membership'], eligible)]
            todo += [(f['model'], f['training_membership'],
                      [i for i in eligible if rows[i]['cluster'] != f['excluded_cluster']]) for f in folds]
            for model, membership, indices in todo:
                scale, coef, members = independent_fit(rows, indices)
                if membership != members or not set(membership['drug_ids']) <= originals:
                    raise ValueError('Training membership or original-cohort isolation mismatch')
                same(model['scale'], scale, 'Fold-local feature scale')
                same(model['coef'], coef, 'Independent ridge solution')
                if model['train_rows'] != len(indices) or model['alpha'] != 1.:
                    raise ValueError('Training row count or penalty differs')
                trained += 1
    for key, goal_choices in lock['choices'].items():
        cid = protocol['designs'][key]['case_id']
        source = checked(protocol['designs'][key]['source'])
        f = features[key]
        for goal, item in goal_choices.items():
            decision = item['decision']
            time_index = source['choices']['time_farthest'][goal]
            if decision['time_index'] != time_index:
                raise ValueError('Time comparator changed')
            if goal not in bundles or f['goals'][goal].get('status') != 'selected':
                expected, proposal, adopted = time_index, time_index, False
                if decision['margins']:
                    raise ValueError('Unsupported goal cannot acquire learned margins')
            else:
                bundle = bundles[goal][cases[cid]['cluster_id'] if cid in originals else '__all__']
                x = np.asarray(f['goals'][goal]['X'])
                models = [bundle['full_model'], *[j['model'] for j in bundle['jackknife_models']]]
                scores = [x/np.asarray(m['scale'])@np.asarray(m['coef']) for m in models]
                times = source['candidate_times_h']
                proposal = int(min(np.flatnonzero(scores[0] >= scores[0].max()-MARGIN), key=lambda i: times[i]))
                margins = [float(s[proposal]-s[time_index]) for s in scores]
                same(decision['full_scores'], scores[0], 'Frozen full-model scores')
                same([r['margin'] for r in decision['margins']], margins, 'Every study-deletion margin')
                adopted = proposal != time_index and all(m > MARGIN for m in margins)
                expected = proposal if adopted else time_index
            if (decision['selected_index'] != expected or decision['proposed_index'] != proposal
                    or decision['adopted'] != adopted):
                raise ValueError('Fixed selection replay differs')
            for method, index in [('stable', expected), ('stable_repaired', expected),
                                  ('aligned_only', proposal), ('time', time_index), ('time_repaired', time_index)]:
                selected = item['sources'][method]
                if index is None:
                    if selected is not None: raise ValueError('Unsupported action was filled')
                else:
                    candidate = key+f'__point_{source["candidate_ids"][index]:02d}'
                    if selected['candidate'] != candidate:
                        raise ValueError('Selected prediction identity mismatch')
                    same(selected['time_h'], source['candidate_times_h'][index], 'Locked candidate time')
            decisions += 1
            switches += adopted
    return dict(independent_model_refits=trained, replayed_decisions=decisions,
                adopted_decisions=switches, training_drugs=10, added_twenty_labels_used=False)


def audit_csv(state, protocol, result, rows, lock, records):
    cases = {c['case_id']: c for c in protocol['cases']}
    originals = set(protocol['original_case_ids'])
    raw = {(r['design_key'], r['candidate']): r for r in records}
    if len(raw) != len(records): raise ValueError('Duplicate old prediction result')
    repaired = result['repaired_scores']; seen = set(); scalar_checks = 0
    for row in rows:
        cid, regime, goal, method, ep = (row[k] for k in ('case_id', 'regime', 'requested_goal', 'method', 'endpoint'))
        ident = (cid, regime, goal, method, ep)
        if ident in seen: raise ValueError('Duplicate endpoint cell')
        seen.add(ident)
        case = cases[cid]; key = cid+'__'+regime
        if (row['drug'] != case['drug'] or row['cluster_id'] != case['cluster_id']
                or row['cohort'] != ('original_ten' if cid in originals else 'added_twenty')):
            raise ValueError('Changed drug/cohort/study identity')
        selection = lock['choices'][key][goal]['sources'][method]
        record = None
        if selection is not None:
            record = raw[key, selection['candidate']]
            if method.endswith('_repaired') and selection['unit_id'] in repaired:
                record = repaired[selection['unit_id']]
            same(number(row, 'selected_time_h'), selection['time_h'], 'CSV selection time')
        else:
            same(number(row, 'selected_time_h'), None, 'Unsupported selection remains absent')
        expected_error = None if record is None else record['curve_metrics']['gmfe'] if ep == 'curve' else record['endpoints'][ep]['error']
        same(number(row, 'error'), expected_error, 'CSV selected result error')
        if ep == 'curve': continue
        endpoint = {} if record is None else record['endpoints'][ep]
        p, o, error = number(row, 'prediction'), number(row, 'reference'), number(row, 'error')
        same(p, endpoint.get('value'), 'Prediction copied exactly')
        same(o, endpoint.get('reference'), 'Observation reference copied exactly')
        if p is None or o is None:
            same(error, None, 'Missing reference/prediction cannot have error')
        else:
            computed = abs(p-o) if ep == 'tmax' else max(p/o, o/p)
            same(error, computed, 'Independent scalar error from P/O')
            scalar_checks += 1
    expected = {(c, r, g, m, e) for c in cases for r in ('early_four', 'late_four')
                for g in GOALS for m in METHODS for e in ENDPOINTS}
    if seen != expected or len(rows) != 30*2*3*8*11 or result['endpoint_rows'] != len(rows):
        raise ValueError('All thirty drugs and all endpoint cells must be retained')
    if set(repaired) != {key for key, unit in state['units'].items() if not unit['reused']}:
        raise ValueError('Missing repaired-unit score')
    for key, unit in state['units'].items():
        if unit['reused']:
            if unit['result'] != protocol['jobs'][key]['source']:
                raise ValueError('Reused result changed')
        else:
            artifact = checked(unit['result'])
            if artifact['protocol'] != state['protocol'] or artifact['original_source'] != protocol['jobs'][key]['source']:
                raise ValueError('Repaired prediction provenance differs')
            for path, expected_hash in artifact['output_pins'].items():
                if digest(path) != expected_hash: raise ValueError('Changed repaired prediction artifact')
    return dict(endpoint_rows=len(rows), scalar_error_recomputations=scalar_checks,
                all_thirty_retained=True)


def number(row, key):
    return None if row[key] == '' else float(row[key])


def summarize(protocol, result, rows):
    cases = {c['case_id']: c for c in protocol['cases']}; original = set(protocol['original_case_ids'])
    values = {(r['case_id'], r['method'], r['endpoint']): float(r['error']) for r in rows
              if r['regime'] == 'late_four' and r['requested_goal'] == 'cmax' and r['error'] != ''}
    scopes = dict(original_ten=original, added_twenty=cases.keys()-original, all_thirty=set(cases))
    statistic_checks = 0
    for scope, members in scopes.items():
        for method in METHODS:
            for endpoint in ('cmax', 'curve'):
                v = {cid: values[cid, method, endpoint] for cid in members if (cid, method, endpoint) in values}
                stats = bootstrap(v, {cid: cases[cid]['cluster_id'] for cid in v}, **protocol['bootstrap'])
                reported = result['summary'][scope][method][endpoint]
                for field in ('estimate', 'ci95'): same(stats[field], reported[field], 'Independent CSV aggregate')
                if stats['n_cases'] != reported['n_cases'] or stats['n_clusters'] != reported['n_clusters']:
                    raise ValueError('Aggregate denominators differ')
                statistic_checks += 1
        for pair, by_endpoint in result['comparisons'][scope].items():
            numerator, denominator = pair.split('/')
            for endpoint, reported in by_endpoint.items():
                ratios = {cid: values[cid, numerator, endpoint]/values[cid, denominator, endpoint] for cid in members
                          if (cid, numerator, endpoint) in values and (cid, denominator, endpoint) in values}
                stats = bootstrap(ratios, {cid: cases[cid]['cluster_id'] for cid in ratios}, **protocol['bootstrap'])
                for field in ('estimate', 'ci95'): same(stats[field], reported[field], 'Independent paired CSV statistic')
                for cid, ratio in ratios.items(): same(ratio, reported['per_case_ratio'][cid], 'Drug-specific ratio')
                if set(ratios) != set(reported['per_case_ratio']): raise ValueError('Drug removed from paired results')
                independent_counts = dict(improved=sum(v < 1-TOLERANCE for v in ratios.values()),
                                          worse=sum(v > 1+TOLERANCE for v in ratios.values()),
                                          tied=sum(abs(v-1) <= TOLERANCE for v in ratios.values()))
                if any(reported[k] != v for k, v in independent_counts.items()):
                    raise ValueError('Floating-point-tolerant outcome counts differ')
                statistic_checks += 1
    outcomes = {}
    for cid, case in cases.items():
        outcomes[cid] = dict(drug=case['drug'], original=cid in original,
                             cross_study=case.get('cross_study_pair', True), endpoints={})
        for ep in ('cmax', 'curve'):
            p = values.get((cid, 'previous_learned', ep)); s = values.get((cid, 'stable_repaired', ep)); t = values.get((cid, 'time_repaired', ep))
            outcomes[cid]['endpoints'][ep] = dict(previous=p, stable_repaired=s, time_repaired=t,
                total_ratio=s/p if s is not None and p is not None else None,
                selector_ratio=s/t if s is not None and t is not None else None)
    return dict(statistic_checks=statistic_checks, drug_outcomes=outcomes)


def audit():
    state, protocol, result, rows = load_complete()
    lock = checked(protocol['choices'])
    if lock['locked_at'] > state['predictions_locked_at']:
        raise ValueError('Selections must precede predictions/scoring')
    features = checked(protocol['features'])['designs']
    previous_state = checked(protocol['source_state'])
    records = checked(previous_state['results'])['records']
    models = audit_models(protocol, lock, features, records)
    cells = audit_csv(state, protocol, result, rows, lock, records)
    statistics = summarize(protocol, result, rows)
    document = dict(status='passed', audited_at=time.time(), protocol=state['protocol'], results=state['results'],
        audit_code=ref(Path(__file__)), source_hashes=len(protocol['source_pins']), **models, **cells, **statistics,
        scope='Arithmetic, frozen source integrity, historical-only training and study isolation; not blinded clinical validation.')
    save(RUN/'audit.json', document)
    return document


def make_figure(protocol, result, audited):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from setup_style import setup_style
    setup_style(journal='nature', lang='en', use_sciplots=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7.1, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'figure.constrained_layout.use': False, 'axes.spines.right': False,
        'axes.spines.top': False, 'savefig.bbox': None})
    fig = plt.figure(figsize=(7.2, 10.6))
    top = fig.add_axes([.32, .715, .60, .175])
    left = fig.add_axes([.23, .19, .28, .41]); right = fig.add_axes([.64, .19, .28, .41])
    estimates = []
    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        stat = result['summary']['added_twenty'][method]['cmax']
        color = '#009E73' if method == 'stable_repaired' else '#263F4C'
        if stat['estimate'] is not None:
            estimates.append(stat['estimate'])
            if stat['ci95']:
                low, high = stat['ci95']; estimates.extend([low, high])
                top.hlines(i, low, high, color=color, linewidth=1.2)
            top.plot(stat['estimate'], i, 's' if method == 'stable_repaired' else 'o', markersize=4.2, color=color)
        else: top.text(.04, i, 'NA', transform=top.get_yaxis_transform(), va='center')
        top.text(1.02, i, f'{stat["n_cases"]}/20', transform=top.get_yaxis_transform(), va='center', fontsize=7)
    top.set_yticks(range(len(METHODS)), LABELS); top.invert_yaxis()
    top.set_title('a  Added 20 drugs: primary Cmax comparison', loc='left', pad=11)
    top.set_xscale('log'); top.minorticks_off(); top.set_xlabel('Drug-equal GMFE (log scale; lower is better)')
    lo, hi = min([1., *estimates])/.98, max(estimates)*1.12
    top.set_xlim(min(.95,lo),hi)
    top.xaxis.set_major_locator(FixedLocator(np.geomspace(min(.95,lo), hi, 4)))
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
    top.grid(axis='x', color='#E0E7EB', linewidth=.5)
    outcomes = audited['drug_outcomes']
    ids = sorted(outcomes, key=lambda cid: (not outcomes[cid]['original'], outcomes[cid]['drug'].lower()))
    all_ratios = [entry['endpoints'][ep][ratio] for entry in outcomes.values() for ep in ('cmax','curve')
                  for ratio in ('total_ratio','selector_ratio') if entry['endpoints'][ep][ratio] is not None]
    lower = min([1., *all_ratios])/1.18; upper = max([1., *all_ratios])*1.18
    ticks = sorted(set([10.**e for e in range(math.ceil(math.log10(lower)), math.floor(math.log10(upper))+1)] + [1.]))
    for axis, ep, title in ((left, 'cmax', 'b  Cmax fold error'), (right, 'curve', 'c  Curve GMFE')):
        for i, cid in enumerate(ids):
            for name, offset, marker, color in [('total_ratio',-.14,'o','#0072B2'),('selector_ratio',.14,'s','#D55E00')]:
                value = outcomes[cid]['endpoints'][ep][name]
                if value is None: axis.text(1., i+offset, 'NA', va='center', ha='center', fontsize=6, color=color)
                else:
                    axis.plot([1.,value],[i+offset,i+offset],color=color,linewidth=.5,alpha=.65)
                    axis.plot(value,i+offset,marker,color=color,markersize=3.3,markeredgewidth=.4)
        axis.axvline(1., color='#84929A', linestyle='--', linewidth=.8)
        axis.axhline(9.5,color='#BEC9D0',linewidth=.7)
        axis.set_xscale('log'); axis.set_xlim(lower,upper); axis.set_ylim(len(ids)-.3,-.7)
        axis.xaxis.set_major_locator(FixedLocator(ticks)); axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'))
        axis.set_yticks(range(len(ids)), [outcomes[cid]['drug']+('*' if not outcomes[cid]['cross_study'] else '') for cid in ids] if axis is left else ['']*len(ids))
        axis.tick_params(axis='y',length=0,pad=4)
        axis.set_title(title,loc='left',pad=11); axis.set_xlabel('Error ratio (log scale)\n<1 favors the revised pipeline')
        axis.grid(axis='x',color='#E0E7EB',linewidth=.5)
    fig.text(.035,.974,'Can aligned training and feasible fitting reduce regressions?',fontsize=10.6,weight='bold',color='#263F4C',va='top')
    fig.text(.035,.948,'Historical 30-drug evaluation | Original 10 train the selector | Added 20 assess transfer',fontsize=7.5,color='#4B5B65')
    outcomes_count = result['comparisons']['added_twenty']['stable_repaired/previous_learned']['cmax']
    fig.text(.035,.924,
        f'Added-drug Cmax versus previous: {outcomes_count["improved"]} improved, '
        f'{outcomes_count["worse"]} worse, {outcomes_count["tied"]} unchanged. '
        'An average gain does not mean every drug improves.',
        fontsize=7,color='#4B5B65')
    fig.legend(handles=[Line2D([],[],color='#0072B2',marker='o',linestyle='None',markersize=4,label='Total change: revised / previous learned'),
                        Line2D([],[],color='#D55E00',marker='s',linestyle='None',markersize=4,label='Selector effect: revised / repaired time')],
               loc='upper center',bbox_to_anchor=(.54,.667),frameon=False,ncol=1,fontsize=7.6)
    notes = ['a: GMFE with 95% study-cluster bootstrap intervals (2,000 draws); available/enrolled counts at right.',
             'b–c: Every drug is retained, including extreme errors; original 10 above the divider and added 20 below.',
             'Blue combines selector and fitting changes. Orange compares selectors with the same feasibility repair.',
             '* Finerenone and Montelukast use different arms within one study; no independent-study claim.',
             'Cmax uses the maximum at the actual observation times. Missing comparisons remain NA.',
             'Study consensus is a stability guard, not a guarantee of clinical improvement. No manuscript overwrite.']
    for i,line in enumerate(notes): fig.text(.035,.106-i*.0145,line,fontsize=6.5,color='#4B5B65',va='bottom')
    return fig


def plot(export=False):
    state, protocol, result, rows = load_complete()
    audited = read(RUN/'audit.json')
    if audited.get('status') != 'passed' or audited['results'] != state['results']:
        raise ValueError('Current independent numerical audit required')
    sys.path.insert(0, str(SKILL))
    from profile_data import profile_data
    from visual_qa import audit_layout, render_preview
    import pandas as pd
    frame = pd.DataFrame([{**row, 'error': number(row,'error')} for row in rows
        if row['regime']=='late_four' and row['requested_goal']=='cmax' and row['endpoint'] in ('cmax','curve')])
    profile = profile_data(frame[['drug','cohort','method','endpoint','error']],group_cols=['cohort','endpoint'])
    fig = make_figure(protocol,result,audited)
    render_preview(fig,str(RUN/'comparison.png'),dpi=220)
    qa = audit_layout(fig)
    audited['figure'] = dict(size_inches=[7.2,10.6], layout_qa=qa, profile=profile,
        png=ref(RUN/'comparison.png'), all_thirty_shown=True, extreme_values_retained=True,
        vector_exported=bool(export), visual_review_required=True)
    if export:
        if any(level=='FAIL' for level,message in qa): raise ValueError('Layout QA failed; PDF export blocked')
        fig.savefig(RUN/'comparison.pdf',format='pdf',dpi=300)
        audited['figure']['pdf']=ref(RUN/'comparison.pdf')
    save(RUN/'audit.json',audited)
    path=RUN/'README.md'
    if path.exists():
        start,stop='<!-- sampling-stability-figure:start -->','<!-- sampling-stability-figure:end -->'
        text=path.read_text()
        if start in text or stop in text:
            if text.count(start)!=1 or text.count(stop)!=1 or text.index(start)>text.index(stop):
                raise ValueError('Malformed figure section')
            before,tail=text.split(start);_,after=tail.split(stop);text=before.rstrip()+'\n'+after.lstrip()
        text += ('\n'+start+'\n\n![Thirty-drug stability evaluation](comparison.png)\n\n'
                 'The upper panel shows primary additional-20 results with study-cluster 95% intervals; the lower panel retains all 30 per-drug error ratios, including extremes.'
                 'Blue combines selection and fitting changes; orange compares selection under identical fitting repair. Fitting gains must not be attributed to the selection algorithm.'
                 'Original-ten drugs use study holdout; additional-20 drugs do not train the scorer. All data were inspected and this remains development evaluation.'
                 'Independent checks recalculated models, fixed selections, every scalar error, and aggregate and paired statistics.\n\n'+stop+'\n')
        path.write_text(text)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return audited['figure']


if __name__ == '__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','plot','all'])
    parser.add_argument('--export',action='store_true',help='After PNG visual review, export matching PDF')
    args=parser.parse_args()
    if args.command in ('audit','all'):
        report=audit();print(json.dumps({k:report[k] for k in ('status','endpoint_rows','independent_model_refits','replayed_decisions','statistic_checks')}))
    if args.command in ('plot','all'):
        figure=plot(args.export);print(json.dumps({k:figure[k] for k in ('png','layout_qa','vector_exported')}))
