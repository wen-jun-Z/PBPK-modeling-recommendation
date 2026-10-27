#!/usr/bin/env python3
"""One frozen grouped-CV test of learned sampling value; zero new PBPK fits."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import math
import os
from pathlib import Path
import time

import numpy as np
import run_evidence_gap_design as previous
from pbpk_kb.blueprint_policy import learned_sampling_design as learner
from pbpk_kb.blueprint_policy import classical_sampling_design as classical

ROOT = previous.ROOT
RUN = ROOT / 'experiments/blueprint_design/learned_sampling_20261009'
GOALS = previous.GOALS
METHODS = ('learned_gain', 'time_farthest', 'maximum_variance', 'c_optimal',
           'request_directed', 'linear_covariance', 'uniform_random', 'no_addition', 'oracle')
old, base, prior = previous.old, previous.base, previous.prior
read, save, ref, checked, verify = previous.read, previous.save, previous.ref, previous.checked, previous.verify


class Guard:
    def __init__(self, p):
        self.p, self.last = p, 0.

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Paused; resolve PAUSE before resuming')
        if force or time.monotonic() - self.last >= 1200:
            verify(self.p['source_pins'])
            size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
            if size > 10 * 1024**3:
                raise RuntimeError('Batch exceeds 10 GiB')
            self.last = time.monotonic()


def prepare():
    if (RUN / 'protocol.json').exists():
        raise ValueError('Frozen protocol already exists; cannot overwrite')
    pp = read(previous.RUN / 'protocol.json'); s = read(previous.RUN / 'state.json')
    if s['status'] != 'complete':
        raise ValueError('The exhaustive action experiment must be complete')
    verify(pp['source_pins']); pins = deepcopy(pp['source_pins'])
    for item in list(s['designs'].values()) + list(s['units'].values()):
        obj = checked(item['result']); base.pins_in(item, pins)
        pins.update(obj['output_pins'])
    files = [Path(__file__), ROOT / 'src/pbpk_kb/blueprint_policy/learned_sampling_design.py',
             ROOT / 'src/pbpk_kb/blueprint_policy/classical_sampling_design.py',
             ROOT / 'scripts/report_blueprint_validation.py', previous.RUN / 'protocol.json',
             previous.RUN / 'state.json', previous.RUN / 'results.json', *classical.source_files()]
    for f in files:
        pins[str(f.relative_to(ROOT))] = base.sha256(f)
    paper = ROOT / 'results/manuscript/pbpk_paper_zh.pdf'
    p = dict(version='learned-acquisition-grouped-development-1', created_at=time.time(),
        previous_protocol=ref(previous.RUN / 'protocol.json'),
        previous_state=ref(previous.RUN / 'state.json'), cached_scores=ref(previous.RUN / 'results.json'),
        source_pins=pins, paper_before=ref(paper), cases=pp['cases'],
        designs=pp['designs'], goals=list(GOALS), methods=list(METHODS),
        features=list(learner.FEATURE_NAMES), ridge_alpha=1.,
        learner='Separate fixed-alpha ridge for each requested endpoint, trained on both early/late layouts. Within-menu X/y centering, training-only weighted RMS scaling, drug-equal/layout-equal/menu-equal row weights. Minimize weighted mean squared residual plus alpha*sum(coefficients squared). No tuning, nonlinear learner or identity features.',
        label='log(baseline clinical FE) - log(acquired clinical FE). Within-menu centering removes the baseline constant. Relative ranking only; not an absolute gain or online stopping probability.',
        isolation='Nine outer target-study folds, every candidate and layout of each held-out drug excluded. Ertapenem/Ceftriaxone jointly held out. Feature builder has no candidate observations or target values. Learner receives only training rows/labels. All fold choices saved before held-out scores are joined. Historical outcomes already viewed; this is retrospective development screening, not a new blind test.',
        classical='Official PyApprox COptimalCriterion, same old 4+1 layouts. Selected visible-data formula; log-coordinate derivatives 1e-4; constant log-noise SD 0.1 matching the old profile experiment; precision=(4+1)*PENALTY^2/0.1^2+1e-8. Finite engineering regularization, not clinical uncertainty. Nine development and eight target simulations per layout; all three model-only endpoints computed together. No new fitting.',
        primary=dict(regime='late_four', goal='cmax'),
        secondary=[dict(regime='early_four', goal=g) for g in GOALS[1:]],
        bootstrap=dict(seed=20261009, resamples=2000, cluster='target study; drug-equal point estimate'),
        gate='Primary GMFE <=0.99 times time, maximum variance, exact random AND official c-opt separately; >=6/10 directional wins versus time. Primary curve <=1.01 times time and no-addition. Early-four AUC infinity and half-life <=1.05 times no-addition. Complete ten drugs and every action; no failed outcomes. Same old gates plus official strong control. Engineering development gate, not significance/clinical acceptance/publication criterion.',
        stop_rule='One frozen feature set and alpha only. If gate fails, stop complex acquisition redesign on this ten-drug set; retain the simple time rule as a provisional comparator, not universally optimal. Keep negative evidence and PBPK predictor; do not select a favorable endpoint/regime after scoring.',
        monitor_seconds=1200, directory_limit_gib=10,
        limitations=['Ten previously viewed drugs, nine target-study clusters; 140 actions are not 140 independent drugs.',
                    'Four same-drug development concentrations required. No zero-curve prediction claim.',
                    'Training labels use previous drugs clinical targets; held-out targets do not enter that fold learner.',
                    'Finite profile solutions and local Fisher information omit shared model error.',
                    'Bootstrap conditions on these cross-validated predictions; it does not refit models or remove development selection bias.',
                    'All 11 output categories retained; missing clinical references stay blank. Random is exact expected log FE, not a synthetic averaged prediction.'])
    verify(pins); save(RUN / 'protocol.json', p)
    save(RUN / 'state.json', dict(status='prepared', protocol=ref(RUN / 'protocol.json'), feature_designs=0, folds=0))
    return status()


def design_features(p, pp, ps, key, guard):
    start = time.monotonic(); spec = pp['designs'][key]
    case = previous.get_case(pp, spec['case_id'])
    d = checked(ps['designs'][key]['result']); training = checked(d['visible_source'])['visible_training']
    result = checked(d['baseline']); formula = result['formula']
    fit = checked(next(f['fit'] for f in result['fits'] if f['formula'] == formula))
    dev = previous.request(case, 'development', formula); target = previous.request(case, 'target', formula)
    devphys = prior.get_phys(pp, case, 'development', formula, dev)
    tarphys = prior.get_phys(pp, case, 'target', formula, target)
    contract = checked(case['target_contract']); grid = base.engine.prediction_grid(target)
    calls = dict(development=0, target=0)
    def predict(x, times):
        guard(); calls['development'] += 1
        return old.simulate('blood_partition', devphys, dev, times, x)['concentration_mg_l']
    def goals(x):
        guard(); calls['target'] += 1
        pred = old.simulate('blood_partition', tarphys, target, grid, x)
        values = previous.data.model_goals(contract, prior.curve_document(pred, grid, target), target)['values']
        return [values[g] for g in GOALS]
    ts = np.array([o['time_min'] / 60 for o in training['observations']])
    ys = np.array([o['concentration_umol_per_l'] * training['molecular_weight_g_mol'] / 1000 for o in training['observations']])
    times = d['candidate_times_h']
    _, jac, grads = learner.local_derivatives(fit['best']['log_parameters'], fit['parameter_lower'],
        fit['parameter_upper'], ts, times, predict, goals)
    penalty = np.array([0., .1 / math.log(10), .1 / math.log(100), .1 / math.log(10)])
    precision = (len(ts) + 1) * penalty**2 / .1**2 + 1e-8
    jv, jc = jac[:len(ts)] / .1, jac[len(ts):] / .1
    before = jv.T @ jv + np.diag(precision)
    out = dict(design_key=key, case_id=case['case_id'], drug=case['drug'], cluster_id=case['cluster_id'],
               regime=spec['regime'], times=times, candidate_ids=d['candidate_ids'],
               baseline=d['baseline'], decision_source=ps['designs'][key]['result'], goals={},
               visible_values_only=True, selected_formula=formula, model_calls=calls,
               visible_jacobian_whitened=jv.tolist(), candidate_jacobian_whitened=jc.tolist(),
               regularization_precision=precision.tolist())
    for gi, goal in enumerate(GOALS):
        classic = classical.c_optimal_scores(jv, jc, grads[gi], precision)
        vbefore = float(grads[gi] @ np.linalg.solve(before, grads[gi]))
        info = d['details'][goal]['sensitivity']['0.1']
        features = learner.features(times=times, visible_times=ts, visible_max=float(max(ys)),
            infusion_end=training['dosing']['start_h'] + training['dosing']['infusion_minutes'] / 60,
            concentrations=d['concentrations_mg_l'],
            linear_utility=info['goals'][goal]['utility_log_variance_reduction'],
            nonlinear_utility=info['nonlinear_goals'][goal]['utility_log_variance_reduction'],
            goal_variance=info['goals'][goal]['variance_log'], c_optimal_scores=classic['scores'], c_optimal_before=vbefore)
        choices = {m: d['choices'][m][goal] for m in d['choices']}
        choices['c_optimal'] = classic['selected_index']
        if any(i is None for i in choices.values()):
            raise ValueError('Incomplete cached comparator selection')
        out['goals'][goal] = dict(X=features.tolist(), choices=choices, c_optimal=classic,
                                  c_optimal_before=vbefore, goal_gradient=grads[gi].tolist())
    out['wall_seconds'] = time.monotonic() - start
    return out


def labels_for_training(cached, feature_rows, permitted_case_ids, goal):
    """Steward API: return only this fold's training labels in the supplied order."""
    if any(row['case_id'] not in permitted_case_ids for row in feature_rows):
        raise ValueError('Held-out row requested from training-label steward')
    records = {(r['design_key'], r['candidate']): r for r in cached['records']
               if r['case_id'] in permitted_case_ids}
    ys = []
    for row in feature_rows:
        key = row['design_key']
        b = records[(key, 'baseline')]['endpoints'][goal]['error']
        a = records[(key, row['candidate'])]['endpoints'][goal]['error']
        if not all(v is not None and math.isfinite(v) and v > 0 for v in (a, b)):
            raise ValueError('Missing/invalid training endpoint; no silent imputation')
        ys.append(math.log(b) - math.log(a))
    return ys


def feature_rows(features):
    return [dict(design_key=key, case_id=d['case_id'], cluster_id=d['cluster_id'],
                 candidate=key + f'__point_{cid:02d}', candidate_index=i)
            for key, d in features.items() for i, cid in enumerate(d['candidate_ids'])]


def run():
    s = read(RUN / 'state.json'); p = checked(s['protocol']); guard = Guard(p); guard(force=True)
    if s['status'] == 'complete':
        return status()
    start = time.monotonic(); s.update(status='features', pid=os.getpid(), started_at=s.get('started_at', time.time()))
    save(RUN / 'state.json', s)
    try:
        pp, ps = checked(p['previous_protocol']), checked(p['previous_state'])
        feature_path = RUN / 'features.json'
        features = checked(s['features']) if s.get('features') else {}
        for key in p['designs']:
            if key not in features:
                features[key] = design_features(p, pp, ps, key, guard)
                save(feature_path, features)
                s.update(feature_designs=len(features), features=ref(feature_path)); save(RUN / 'state.json', s)
                print(f'feature designs {len(features)}/{len(p["designs"])}', flush=True)
        guard(force=True)
        # First outcome-file read in this runner. Models below receive training subsets only.
        cached = checked(p['cached_scores']); rows = feature_rows(features)
        decisions = checked(s['decisions']) if s.get('decisions') else dict(folds={}, choices={}, protocol=s['protocol'], features=s['features'])
        s['status'] = 'grouped_training'; save(RUN / 'state.json', s)
        for group, train, test in learner.cluster_folds(rows):
            if group in decisions['folds']:
                continue
            training = [rows[i] for i in train]; testing = [rows[i] for i in test]
            allowed = {r['case_id'] for r in training}; heldout = {r['case_id'] for r in testing}
            if allowed & heldout:
                raise ValueError('Drug crossed a fold')
            fold = dict(train_case_ids=sorted(allowed), heldout_case_ids=sorted(heldout), models={})
            for goal in GOALS:
                x = np.array([features[r['design_key']]['goals'][goal]['X'][r['candidate_index']] for r in training])
                y = labels_for_training(cached, training, allowed, goal)
                model = learner.fit_ranker(x, y, [r['design_key'] for r in training], [r['case_id'] for r in training], alpha=p['ridge_alpha'])
                fold['models'][goal] = model
                for key in sorted({r['design_key'] for r in testing}):
                    d = features[key]; scores = learner.predict_scores(model, d['goals'][goal]['X'])
                    index = learner.choose_index(scores, d['times'])
                    decisions['choices'].setdefault(key, {})[goal] = dict(index=index, scores=scores.tolist(),
                        selected_time_h=d['times'][index], fold=group, heldout_labels_used=False)
            fold['locked_at'] = time.time(); decisions['folds'][group] = fold
            save(RUN / 'decisions.json', decisions)
            s.update(folds=len(decisions['folds']), decisions=ref(RUN / 'decisions.json')); save(RUN / 'state.json', s)
            print(f'held-out study folds {len(decisions["folds"])}/9', flush=True)
        if len(decisions['choices']) != len(p['designs']):
            raise ValueError('Incomplete choice lock')
        guard(force=True)
        s.update(status='choices_locked', choices_locked_at=time.time(), elapsed_seconds=s.get('elapsed_seconds', 0) + time.monotonic() - start)
        save(RUN / 'state.json', s)
        return report()
    except BaseException as exc:
        save(RUN / 'PAUSE', dict(reason=str(exc))); s.update(status='paused', reason=str(exc)); save(RUN / 'state.json', s)
        raise


def gm(values):
    if not values or any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        return None
    return math.exp(float(np.mean(np.log(values))))


def selected_rows(p, features, decisions, cached):
    records = {(r['design_key'], r['candidate']): r for r in cached['records']}
    rows = []
    for key, d in features.items():
        candidates = [records[(key, key + f'__point_{i:02d}')] for i in d['candidate_ids']]
        for goal in GOALS:
            for method in METHODS:
                index = None
                if method == 'no_addition':
                    chosen = [records[(key, 'baseline')]]
                elif method == 'uniform_random':
                    chosen = candidates
                else:
                    if method == 'learned_gain':
                        index = decisions['choices'][key][goal]['index']
                    elif method == 'oracle':
                        # Hindsight bound is never offered to the learner.
                        index = int(np.argmin([r['endpoints'][goal]['error'] for r in candidates]))
                    else:
                        index = d['goals'][goal]['choices'][method]
                    chosen = [candidates[index]]
                if any(r['status'] != 'predicted' for r in chosen):
                    raise ValueError('Failed cached acquisition; retain failure and pause')
                for ep in ('curve', *chosen[0]['endpoints']):
                    if ep == 'curve':
                        error = gm([r['curve_metrics']['gmfe'] for r in chosen]); pred = obs = None
                        unit = ''; weak = None; metric = 'curve_gmfe'; count = chosen[0]['curve_metrics']['n']
                    else:
                        entries = [r['endpoints'][ep] for r in chosen]
                        errors = [e['error'] for e in entries]
                        if ep == 'tmax':
                            error = float(np.mean(errors)) if all(v is not None and math.isfinite(v) for v in errors) else None
                        else:
                            error = gm(errors)
                        pred = entries[0]['value'] if len(entries) == 1 else None
                        obs, unit = entries[0]['reference'], entries[0]['unit']
                        weak = float(np.mean([e.get('weak_tail', False) for e in entries]))
                        metric = entries[0]['error_metric']; count = None
                    rows.append(dict(case_id=d['case_id'], drug=d['drug'], cluster_id=d['cluster_id'],
                        regime=d['regime'], requested_goal=goal, method=method, endpoint=ep,
                        candidate_index=index, selected_time_h=None if index is None else d['times'][index],
                        prediction=pred, reference=obs, unit=unit, error=error, error_metric=metric,
                        weak_tail_fraction=weak, curve_points=count,
                        semantics='exact_expected_error_no_single_prediction' if method == 'uniform_random' else 'cached_actual_prediction'))
    return rows


def report():
    from report_blueprint_validation import clustered_geomean, paired_ratio
    s = read(RUN / 'state.json'); p = checked(s['protocol']); Guard(p)(force=True)
    if s['status'] not in ('choices_locked', 'complete'):
        raise ValueError('All held-out choices must lock before scoring')
    features, decisions, cached = checked(s['features']), checked(s['decisions']), checked(p['cached_scores'])
    if decisions['protocol'] != s['protocol'] or decisions['features'] != s['features']:
        raise ValueError('Changed decision input receipts')
    rows = selected_rows(p, features, decisions, cached)
    path = RUN / 'endpoints.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    clusters = {c['case_id']: c['cluster_id'] for c in p['cases']}
    stats_args = dict(resamples=p['bootstrap']['resamples'], seed=p['bootstrap']['seed'])
    summary, comparisons = {}, {}
    for regime in previous.REGIMES:
        summary[regime], comparisons[regime] = {}, {}
        for goal in GOALS:
            summary[regime][goal], comparisons[regime][goal] = {}, {}
            by = {}
            for method in METHODS:
                values = {ep: {r['case_id']: r['error'] for r in rows if r['regime'] == regime and r['requested_goal'] == goal
                    and r['method'] == method and r['endpoint'] == ep and r['error'] is not None} for ep in (goal, 'curve')}
                by[method] = values
                summary[regime][goal][method] = {name: clustered_geomean(values[ep], {c: clusters[c] for c in values[ep]}, **stats_args)
                     for name, ep in (('endpoint', goal), ('curve', 'curve'))}
            for method in METHODS:
                if method != 'learned_gain':
                    comparisons[regime][goal][method] = paired_ratio(by['learned_gain'][goal], by[method][goal], clusters, **stats_args)
    primary = summary['late_four']['cmax']; pair = comparisons['late_four']['cmax']
    checks = dict(all_ten_primary=all(primary[m]['endpoint']['n_cases'] == 10 for m in METHODS),
        one_percent_better_controls=all(pair[m]['estimate'] is not None and pair[m]['estimate'] <= .99 for m in ('time_farthest', 'maximum_variance', 'uniform_random', 'c_optimal')),
        six_cmax_wins=pair['time_farthest']['ratio_below_one'] >= 6,
        curve_not_harmed=all(primary['learned_gain']['curve']['estimate'] <= 1.01 * primary[m]['curve']['estimate'] for m in ('time_farthest', 'no_addition')),
        secondary_not_harmed=all(summary['early_four'][g]['learned_gain']['endpoint']['estimate'] <= 1.05 * summary['early_four'][g]['no_addition']['endpoint']['estimate'] for g in GOALS[1:]),
        all_cached_actions_complete=len(cached['records']) == 160 and all(r['status'] == 'predicted' for r in cached['records']))
    # All endpoints, including missing-reference coverage and Tmax in hours.
    endpoint_summary = {}
    main_rows = [r for r in rows if r['regime'] == 'late_four' and r['requested_goal'] == 'cmax']
    for method in METHODS:
        endpoint_summary[method] = {}
        for ep in sorted({r['endpoint'] for r in rows}):
            rr = [r for r in main_rows if r['method'] == method and r['endpoint'] == ep]
            vals = [r['error'] for r in rr if r['error'] is not None]
            endpoint_summary[method][ep] = dict(n=len(vals), attempted=10,
                mean_absolute_hours=float(np.mean(vals)) if ep == 'tmax' and vals else None,
                gmfe=gm(vals) if ep != 'tmax' else None,
                within_two=sum(v <= 2 for v in vals) if ep != 'tmax' else None,
                weak_tail_fraction=float(np.mean([r['weak_tail_fraction'] for r in rr if r['weak_tail_fraction'] is not None])) if ep != 'curve' else None)
    result = dict(version=p['version'], protocol=s['protocol'], choices=s['decisions'],
        choices_locked_at=s['choices_locked_at'], scored_at=time.time(), summary=summary, comparisons=comparisons,
        expansion_checks=checks, expand_to_more_drugs=all(checks.values()), primary_all_endpoints=endpoint_summary,
        data=dict(drugs=10, study_clusters=9, layouts=20, cached_acquisitions=140, training_endpoint_labels=420,
                  fold_models=27, newly_fitted_pbpk_models=0, endpoint_rows=len(rows)),
        resources=dict(development_calls=sum(d['model_calls']['development'] for d in features.values()),
                       target_calls=sum(d['model_calls']['target'] for d in features.values()),
                       feature_seconds=sum(d['wall_seconds'] for d in features.values()), elapsed_seconds=s['elapsed_seconds']),
        limitations=p['limitations'], endpoint_csv=ref(path))
    save(RUN / 'results.json', result)
    s.update(status='complete', completed_at=time.time(), results=ref(RUN / 'results.json')); save(RUN / 'state.json', s)
    write_readme(result)
    return status()


def write_readme(r):
    names = dict(learned_gain='Learned realized gain (group-held-out)', time_farthest='Time coverage', maximum_variance='Concentration variance',
        c_optimal='Official c-optimal', request_directed='Original nonlinear utility', linear_covariance='Original linear utility',
        uniform_random='Exact random expectation', no_addition='No additional point', oracle='Post-hoc oracle (not deployable)')
    lines = ['# Learning additional-point utility: ten-drug development evaluation', '',
        'The original human PBPK predictor is unchanged; only sampling-point selection is learned. All 140 actual additional-point results are reused, with zero new PBPK fits. Ten drugs are held out in nine target-study groups; Ertapenem and Ceftriaxone are held out together. Each fold trains fixed-regularization linear scorers for Cmax, AUC infinity, and half-life, for 27 small models. All drugs were used in development; this is not a new blinded test.', '',
        '**Conclusion: ' + ('Passed the predefined development expansion gate; independent validation after freezing is still required.' if r['expand_to_more_drugs'] else 'Did not pass the predefined expansion gate. Stop adding complex selection variants on these ten drugs; retain the original PBPK predictor and simple time rule for subsequent validation.') + '**', '',
        '| Method | Missing early point: Cmax GMFE (primary) | Missing late point: AUC infinity GMFE | Missing late point: half-life GMFE | Primary-task curve GMFE |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        a = r['summary']['late_four']['cmax'][m]
        b = r['summary']['early_four']
        lines.append(f"| {names[m]} | {a['endpoint']['estimate']:.4f} | {b['auc_inf'][m]['endpoint']['estimate']:.4f} | {b['terminal_half_life'][m]['endpoint']['estimate']:.4f} | {a['curve']['estimate']:.4f} |")
    pair = r['comparisons']['late_four']['cmax']['time_farthest']
    lines += ['', f"Primary Cmax error ratio, learned/time rule: **{pair['estimate']:.4f}**, study-cluster 95% interval **[{pair['ci95'][0]:.4f}, {pair['ci95'][1]:.4f}]**; {pair['ratio_below_one']}/10 drugs improved and {pair['ratio_above_one']}/10 worsened. Only ratios below 1 indicate improvement.", '',
        'Adding versus not adding a point tests the value of more data; equal-budget rule comparisons test the selection algorithm. The post-hoc oracle bounds attainable performance and cannot select online samples. Training labels use clinical errors of other drugs; held-out drug answers do not enter that fold. Scores provide relative order only and cannot establish clinical safety or a stopping rule from their sign.', '',
        'All original 11 outputs are in one [endpoints.csv](endpoints.csv), leaving absent clinical references empty. Tmax uses absolute error in hours. The random method reports exact expected candidate errors, not a fabricated mean trajectory. Both layouts, three endpoints, all controls, and intervals are in [results.json](results.json).', '',
        'Eight pre-observation features and ridge alpha=1 are fixed, without hyperparameter or feature selection. Rules are frozen in [protocol.json](protocol.json), with selections in [decisions.json](decisions.json). Scaling uses training folds only, with equal drug and layout weights. The 140 actions are not 140 independent drugs. The 95% intervals condition on these CV predictions, without bootstrap retraining or removal of repeated-development bias.', '',
        f"New compute is sensitivity only: {r['resources']['development_calls']} development-model solves and {r['resources']['target_calls']} target-condition solves, without target observations; elapsed {r['resources']['elapsed_seconds']:.1f} seconds. The manuscript and PBPK base remain unchanged.", '',
        'Commands from the project root:', '```bash',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 runtime/app-env/bin/python -B scripts/run_learned_sampling_design.py status',
        '# The same entry supports prepare / run / report / cleanup; existing protocols cannot be overwritten.', '```', '']
    (RUN / 'README.md').write_text('\n'.join(lines))


def status():
    s = read(RUN / 'state.json')
    out = {k: s.get(k) for k in ('status', 'feature_designs', 'folds', 'elapsed_seconds', 'reason')}
    if s['status'] == 'complete':
        r = checked(s['results']); out.update(expand=r['expand_to_more_drugs'], primary=r['summary']['late_four']['cmax']['learned_gain']['endpoint'])
    return out


def cleanup():
    s = read(RUN / 'state.json')
    if s['status'] != 'complete':
        raise ValueError('Do not clean an incomplete run')
    Guard(checked(s['protocol']))(force=True)
    removed = 0
    for path in RUN.glob('*.tmp'):
        if path.is_file() and not path.is_symlink():
            path.unlink(); removed += 1
    return dict(removed_transient_files=removed, scientific_outputs_retained=True)


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'report', 'cleanup'])
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](), ensure_ascii=False, indent=2))
