#!/usr/bin/env python3
"""Frozen 10-drug-trained sampling policy evaluated on twenty added drugs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import run_evidence_gap_design as gap
import run_learned_sampling_design as learned
import blueprint_gap_design_data as split_data
import blueprint_sampling_expansion_data as data
from pbpk_kb.blueprint_policy import classical_sampling_design as classical
from pbpk_kb.blueprint_policy import learned_sampling_design as learner

ROOT = gap.ROOT
RUN = ROOT / 'experiments/blueprint_design/sampling_expansion_20261009'
COHORT = RUN / 'inputs/cohort_manifest.json'
TRAIN = ROOT / 'experiments/blueprint_design/learned_sampling_20261009'
OLD = ROOT / 'experiments/blueprint_design/evidence_gap_design_20261008'
GOALS = gap.GOALS
METHODS = learned.METHODS
base, old, prior = gap.base, gap.old, gap.prior
read, save, ref, checked, verify = gap.read, gap.save, gap.ref, gap.checked, gap.verify


def configure():
    """Adapt execution context only; frozen on-disk implementations are untouched."""
    gap.RUN = RUN
    data.split_development = split_data.split_development
    data.reveal_one = split_data.reveal_one
    gap.data = data
    gap.Guard = Guard


class Guard:
    def __init__(self, protocol):
        self.p, self.last = protocol, 0.

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Batch paused; inspect PAUSE before resuming')
        if force or time.monotonic() - self.last >= 1200:
            verify(self.p['source_pins'])
            size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
            if size > 10 * 1024**3:
                raise RuntimeError('Batch directory exceeds 10 GiB')
            self.last = time.monotonic()


def freeze_rankers():
    source_state = read(TRAIN / 'state.json'); original = checked(source_state['protocol'])
    verify(original['source_pins'])
    features = checked(source_state['features']); cached = checked(original['cached_scores'])
    rows = learned.feature_rows(features); allowed = {r['case_id'] for r in rows}
    expected = {c['case_id'] for c in original['cases']}
    if len(allowed) != 10 or allowed != expected:
        raise ValueError('Only original ten drugs may train the fixed score models')
    models = {}
    for goal in GOALS:
        x = [features[r['design_key']]['goals'][goal]['X'][r['candidate_index']] for r in rows]
        y = learned.labels_for_training(cached, rows, allowed, goal)
        models[goal] = learner.fit_ranker(x, y, [r['design_key'] for r in rows], [r['case_id'] for r in rows], alpha=1.)
    document = dict(models=models, trained_case_ids=sorted(allowed), features=source_state['features'],
        labels=original['cached_scores'], original_oof_decisions=source_state['decisions'],
        alpha=1., feature_names=list(learner.FEATURE_NAMES), added_drug_labels_used=False)
    path = RUN / 'rankers.json'
    if path.exists() and read(path) != document:
        raise ValueError('Frozen score model differs on resume')
    save(path, document)
    return document


def export_one(job):
    case_id, phase, formula, request = job
    folder = RUN / 'native_cache' / f'{case_id}__{phase}__{formula}'
    folder.mkdir(parents=True, exist_ok=True)
    def guard():
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Native construction paused')
    base.engine._physiology(request, folder, ROOT, guard)
    return ref(folder / 'physiology/perfusion_physiology.json')


def declare_development_source(case):
    """Copy the source's declared role into its reference; never infer study role."""
    record = case['sampling_sources']['full']
    role = 'calibration_data_reconstruction'
    document = checked(record)
    if document.get('role') != role or record.get('role', role) != role:
        raise ValueError('Development source must explicitly declare calibration role')
    case['sampling_sources']['full'] = dict(record, role=role)


def prepare(workers):
    if (RUN / 'protocol.json').exists():
        return status()
    cohort = read(COHORT)
    verify(cohort['source_pins'])
    ids = [c['case_id'] for c in cohort['cases']]
    compounds = [c['compound_id'] for c in cohort['cases']]
    drugs = [c['drug'] for c in cohort['cases']]
    if (cohort['unique_drugs'] != len(set(compounds)) or len(set(compounds)) < 30
            or len(set(drugs)) != len(compounds) or len(set(ids)) != len(ids)
            or any(c.get('requests_pending_RF_priors') for c in cohort['cases'])):
        raise ValueError('At least thirty curated, request-ready distinct drugs required')
    rankers = freeze_rankers(); original_ids = set(rankers['trained_case_ids'])
    cases = deepcopy(cohort['cases'])
    original_p = read(OLD / 'protocol.json')
    # Retain original ten exact construction inputs/cache, with new scoring-only contract.
    for i, case in enumerate(cases):
        if case['case_id'] in original_ids:
            oldcase = next(c for c in original_p['cases'] if c['case_id'] == case['case_id'])
            cases[i] = deepcopy(oldcase) | dict(expansion_cohort='previous_ten', cross_study_pair=True,
                sensitivity_flags=case.get('sensitivity_flags', []), original_metadata=case)
    for case in cases:
        declare_development_source(case)
        base._training_document(case['sampling_sources']['full'],
                                gap.request(case, 'development', 'PKSimStandard'), ROOT)
    prep_path = RUN / 'preparation.json'
    prep = read(prep_path) if prep_path.exists() else dict(cohort=ref(COHORT), status='native_export', completed={}, failures={})
    if prep['cohort'] != ref(COHORT):
        raise ValueError('Cohort changed during construction')
    prep.update(pid=os.getpid(), started_at=prep.get('started_at', time.time())); save(prep_path, prep)
    jobs = {}
    for case in cases:
        if case['case_id'] in original_ids:
            continue
        for phase in ('development', 'target'):
            for formula in base.FORMULAS:
                key = f'{case["case_id"]}__{phase}__{formula}'
                jobs[key] = (case['case_id'], phase, formula, gap.request(case, phase, formula))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(export_one, job): key for key, job in jobs.items() if key not in prep['completed']}
        for future in as_completed(pending):
            key = pending[future]
            try:
                prep['completed'][key] = future.result()
                prep['failures'].pop(key, None)
            except Exception as exc:
                prep['failures'][key] = str(exc)
            save(prep_path, prep)
            print(f'native exports {len(prep["completed"])}/{len(jobs)}, failures={len(prep["failures"])}', flush=True)
    if prep['failures']:
        prep['status'] = 'construction_blocked'; save(prep_path, prep)
        raise ValueError('Native construction failed; inspect preparation.json, no clinical scoring performed')
    caches = dict(original_p['physiology_cache']) | prep['completed']
    designs, jobs = {}, {}
    for case in cases:
        target = gap.request(case, 'target', 'PKSimStandard')
        contract = data.build_target_contract(case, checked(case['reference_source']), target)
        path = RUN / 'inputs/contracts' / (case['case_id'] + '.json'); save(path, contract)
        case['target_contract'] = ref(path)
        if case['case_id'] in original_ids:
            continue
        dev = gap.request(case, 'development', 'PKSimStandard')
        full = base._training_document(case['sampling_sources']['full'], dev, ROOT)
        for regime in gap.REGIMES:
            key = case['case_id'] + '__' + regime
            split = split_data.split_development(full, dev, regime)
            path = RUN / 'inputs/layouts' / (key + '.json'); save(path, split)
            designs[key] = dict(case_id=case['case_id'], regime=regime, visible_source=ref(path))
            for ix, t in zip(split['design_only']['candidate_ids'], split['design_only']['candidate_times_h']):
                jobs[key + f'__point_{ix:02d}'] = dict(design_key=key, case_id=case['case_id'], regime=regime, candidate_id=ix, time_h=t)
    pins = deepcopy(original_p['source_pins'])
    pins.update(cohort['source_pins'])
    for item in cases:
        base.pins_in(item, pins)
        for phase in ('development', 'target'):
            base.pins_in(checked(item['requests']['evidence'][phase]), pins)
    base.pins_in(caches, pins); base.pins_in(designs, pins)
    for reference in caches.values():
        physiology = checked(reference)
        pins.update(physiology['source_pins']); pins.update(physiology['artifact_pins'])
        completion = (ROOT/reference['path']).parent.parent/'base/completion.json'
        base.pins_in(ref(completion), pins)
        pins.update(read(completion)['artifact_pins'])
    files = [Path(__file__), ROOT / 'scripts/blueprint_sampling_expansion_data.py',
        ROOT / 'scripts/prepare_sampling_expansion_cohort.py', ROOT / 'scripts/prepare_sampling_expansion_priors.py',
        ROOT / 'src/pbpk_kb/blueprint_policy/learned_sampling_design.py',
        ROOT / 'src/pbpk_kb/blueprint_policy/classical_sampling_design.py',
        ROOT / 'scripts/run_learned_sampling_design.py', COHORT, RUN / 'rankers.json',
        TRAIN / 'protocol.json', TRAIN / 'features.json', TRAIN / 'decisions.json', OLD / 'results.json', OLD / 'state.json',
        *classical.source_files()]
    for f in files:
        pins[str(f.relative_to(ROOT))] = base.sha256(f)
    p = dict(version='frozen-sampling-policy-thirty-drugs-1', created_at=time.time(), cases=cases,
        original_case_ids=sorted(original_ids), source_pins=pins, physiology_cache=caches,
        designs=designs, jobs=jobs, goals=list(GOALS), regimes=list(gap.REGIMES), rankers=ref(RUN / 'rankers.json'),
        fit_budget_per_formula=16384, fit_starts=16, profile_grid_points=7,
        profile_budget_per_formula=65536, log_tolerance=math.log(1.1), log_noise_sd=.1,
        noise_sensitivity=[.05, .1, .2], quadrature_nodes=31, monitor_seconds=1200, directory_limit_gib=10,
        worker_count=workers, native_threads=2, native_timeout_seconds=180,
        paper_before=ref(ROOT / 'results/manuscript/pbpk_paper_zh.pdf'),
        primary='Added twenty drugs: late-four Cmax fold error against matched observed-curve sampled Cmax; curve GMFE reported separately. Original ten OOF and all thirty are secondary scope summaries, not forty or thirty independent trials.',
        frozen_policy='Original eight features, alpha=1; three models fitted only to original-ten historical action labels before new outcomes. No new-drug outcome tuning. Added-drug RF parameter priors exclude all thirty structures/scaffolds; original-ten caches unchanged and separately identified.',
        bolus='Physical zero-duration bolus retained. Log-time feature uses 1 h reference; this adds only a common menu offset so linear ranking differences are unchanged. After-infusion flag constant one. Never simulate a fake infusion.',
        reference_policy='All-thirty common scores are NCA of actual clinical curves. Missing t0 means full AUC∞/CL/Vz/MRT/Vss references stay absent; model predictions retained. Published scalars stay in a separate sensitivity field. Never manufacture missing clinical measurements.',
        isolation='Freeze ten-drug models; lock all added-drug selections before revealing candidate concentrations; lock every target prediction before common clinical scoring. Curation reads original documents but does not supply outcomes to selectors.',
        strata='Original ten; previously curated additional ten; new ten source-curated; added twenty; all thirty; cross-study vs within-study. Same-study Finerenone/Montelukast arms not claimed independent studies. Matrix/population/coadministration caveats retained.',
        bootstrap=dict(seed=20261009, resamples=2000),
        expansion_authorization='User explicitly requested >=30 despite ten-drug gate failure; historical gate remains failed. This is a frozen scale assessment, not posthoc reversal of the old conclusion.',
        limitations=['All project clinical sources are historical/developer-viewed; no prospective or untouched blind-test claim.',
            'All sampling methods require four same-drug observations; no zero-curve or new-drug-parameter accuracy claim.',
            'Concentration matrix, human defaults, coadministration and nonlinear-PK concerns require stratified interpretation.',
            'Original-ten cached constructions and RF sources differ from added-drug thirty-excluded RF priors; pooled statistics do not erase this difference.',
            'Exact random uses every candidate; oracle is hindsight only; neither changes the locked selector.'])
    verify(pins); save(RUN / 'protocol.json', p)
    save(RUN / 'state.json', dict(status='prepared', protocol=ref(RUN / 'protocol.json'), designs={}, units={}, features={}))
    prep['status'] = 'complete'; save(prep_path, prep)
    return status()


def make_features(p, key, entry):
    """One local derivative calculation; unavailable tail goals cannot erase Cmax."""
    d = checked(entry['result']); case = gap.get_case(p, p['designs'][key]['case_id'])
    visible = checked(d['visible_source'])['visible_training']; baseline = checked(d['baseline'])
    formula = baseline['formula']; fit = checked(next(f['fit'] for f in baseline['fits'] if f['formula'] == formula))
    dev = gap.request(case, 'development', formula); target = gap.request(case, 'target', formula)
    dp = prior.get_phys(p, case, 'development', formula, dev); tp = prior.get_phys(p, case, 'target', formula, target)
    contract = checked(case['target_contract']); grid = base.engine.prediction_grid(target)
    ts = np.array([v['time_min']/60 for v in visible['observations']]); times = np.array(d['candidate_times_h'])
    ys = np.array([v['concentration_umol_per_l'] * visible['molecular_weight_g_mol']/1000 for v in visible['observations']])
    x = np.asarray(fit['best']['log_parameters']); lower = np.asarray(fit['parameter_lower']); upper = np.asarray(fit['parameter_upper'])
    combined = np.r_[ts, times]; order = np.argsort(combined); inverse = np.argsort(order)
    jac, grads = [], []
    for axis in range(4):
        a, b = x.copy(), x.copy(); a[axis] = max(lower[axis], x[axis]-1e-4); b[axis] = min(upper[axis], x[axis]+1e-4)
        width = b[axis]-a[axis]
        ca = np.asarray(old.simulate('blood_partition', dp, dev, combined[order], a)['concentration_mg_l'])[inverse]
        cb = np.asarray(old.simulate('blood_partition', dp, dev, combined[order], b)['concentration_mg_l'])[inverse]
        jac.append((np.log(cb)-np.log(ca))/width)
        ga = data.model_goals(contract, prior.curve_document(old.simulate('blood_partition', tp, target, grid, a), grid, target), target)['values']
        gb = data.model_goals(contract, prior.curve_document(old.simulate('blood_partition', tp, target, grid, b), grid, target), target)['values']
        grads.append([(math.log(gb[g])-math.log(ga[g]))/width if ga[g] and gb[g] else np.nan for g in GOALS])
    jac = np.asarray(jac).T/.1; grads = np.asarray(grads).T
    penalty = np.array([0, .1/math.log(10), .1/math.log(100), .1/math.log(10)])
    precision = 5*penalty**2/.1**2+1e-8; fisher = jac[:4].T@jac[:4]+np.diag(precision)
    models = checked(p['rankers'])['models']; result = dict(design_key=key, goals={}, locked_at=time.time(), hidden_values_used=False,
        model_calls=dict(development=8, target=8), candidate_times_h=times.tolist(), candidate_ids=d['candidate_ids'])
    for gi, goal in enumerate(GOALS):
        choices = {m: d['choices'][m][goal] for m in d['choices']}
        info = d['details'][goal]
        if not np.isfinite(grads[gi]).all() or info['status'] != 'selected':
            result['goals'][goal] = dict(status='unsupported', choices=choices | dict(learned_gain=None, c_optimal=None), reason='Missing matched model-only endpoint or finite profile support')
            continue
        classic = classical.c_optimal_scores(jac[:4], jac[4:], grads[gi], precision)
        vbefore = float(grads[gi] @ np.linalg.solve(fisher, grads[gi])); info = info['sensitivity']['0.1']
        xfeatures = data.acquisition_features(times=times, visible_times=ts, visible_max=float(max(ys)),
            infusion_end=dev['dosing']['infusion_minutes']/60, concentrations=d['concentrations_mg_l'],
            linear_utility=info['goals'][goal]['utility_log_variance_reduction'],
            nonlinear_utility=info['nonlinear_goals'][goal]['utility_log_variance_reduction'],
            goal_variance=info['goals'][goal]['variance_log'], c_optimal_scores=classic['scores'], c_optimal_before=vbefore)
        scores = learner.predict_scores(models[goal], xfeatures)
        choices.update(learned_gain=learner.choose_index(scores, times), c_optimal=classic['selected_index'])
        result['goals'][goal] = dict(status='selected', choices=choices, X=xfeatures.tolist(), scores=scores.tolist(), c_optimal=classic)
    return result


def design_worker(p, key):
    configure(); entry = gap.design_one(p, key)
    feature = RUN / 'designs' / key / 'learned_choices.json'
    if feature.exists():
        doc = read(feature)
        if doc.get('protocol') != ref(RUN/'protocol.json') or doc.get('decision') != entry['result']:
            raise ValueError('Changed resumed sampling decision')
    else:
        doc = make_features(p, key, entry); doc.update(protocol=ref(RUN/'protocol.json'), decision=entry['result'])
        save(feature, doc)
    return entry | dict(features=ref(feature))


def action_worker(p, key, job, lock):
    configure()
    return gap.execute(p, key, job, lock)


def execute(workers):
    configure()
    s = read(RUN / 'state.json'); p = checked(s['protocol']); guard = Guard(p); guard(force=True)
    if s['status'] == 'complete':
        return status()
    s.update(status='designing', pid=os.getpid(), started_at=s.get('started_at', time.time())); save(RUN/'state.json', s)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(design_worker, p, key): key for key in p['designs'] if key not in s['designs']}
            for future in as_completed(futures):
                key = futures[future]; s['designs'][key] = future.result(); save(RUN/'state.json', s)
                print(f'added-drug layouts {len(s["designs"])}/{len(p["designs"])}', flush=True)
        guard(force=True)
        lockpath = RUN/'actions_locked.json'
        doc = dict(protocol=s['protocol'], designs=s['designs'], locked_at=time.time())
        if lockpath.exists():
            locked = read(lockpath)
            if locked['protocol'] != doc['protocol'] or locked['designs'] != doc['designs']:
                raise ValueError('Changed global selection lock')
        else:
            save(lockpath, doc)
        lock = ref(lockpath); s.update(status='acquiring', action_lock=lock); save(RUN/'state.json', s)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(action_worker, p, key, job, lock): key for key, job in p['jobs'].items() if key not in s['units']}
            for future in as_completed(futures):
                key = futures[future]; s['units'][key] = future.result(); save(RUN/'state.json', s)
                print(f'added-drug actual acquisition fits {len(s["units"])}/{len(p["jobs"])}', flush=True)
        guard(force=True); s.update(status='predictions_locked', predictions_locked_at=time.time()); save(RUN/'state.json', s)
        return report()
    except BaseException as exc:
        save(RUN/'PAUSE', dict(reason=str(exc), time=time.time())); s.update(status='paused', reason=str(exc)); save(RUN/'state.json', s)
        raise


def all_records(p, state):
    """Read clinical labels only at this post-lock scoring stage."""
    old_state = read(OLD/'state.json'); old_choices = read(TRAIN/'decisions.json')['choices']
    old_features = read(TRAIN/'features.json')
    records, policies, failures = [], {}, []
    for case in p['cases']:
        cid = case['case_id']; original = cid in p['original_case_ids']
        for regime in gap.REGIMES:
            key = cid+'__'+regime
            decision_ref = (old_state if original else state)['designs'][key]['result']; d = checked(decision_ref)
            if original:
                policies[key] = {g: old_features[key]['goals'][g]['choices'] | dict(learned_gain=old_choices[key][g]['index']) for g in GOALS}
            else:
                ch = checked(state['designs'][key]['features'])
                policies[key] = {g: ch['goals'][g]['choices'] for g in GOALS}
            sources = [('baseline', d['baseline'])]
            sources += [(key+f'__point_{i:02d}', (old_state if original else state)['units'][key+f'__point_{i:02d}']['result']) for i in d['candidate_ids']]
            for candidate, source in sources:
                unit = checked(source)
                rec = dict(case_id=cid, drug=case['drug'], design_key=key, regime=regime, candidate=candidate,
                    cohort=case['expansion_cohort'], status=unit['status'], source=source)
                if unit['status'] == 'predicted':
                    try:
                        rec.update(data.score_case(case, unit))
                    except (ValueError, ArithmeticError, KeyError) as exc:
                        rec.update(status='scoring_failed', reason=str(exc))
                if rec['status'] != 'predicted':
                    failures.append(rec)
                    rec.update(endpoints={ep:dict(value=None, reference=None, error=None, unit=u, reason=rec.get('reason','Prediction failed'),reference_kind='unavailable') for ep,u in data.UNITS.items()},curve_metrics=dict(gmfe=None,n=0))
                records.append(rec)
    return records, policies, failures


def endpoint_rows(p, state, records, policies):
    lookup = {(r['design_key'],r['candidate']):r for r in records}; rows=[]
    original = read(OLD/'state.json')
    for case in p['cases']:
        for regime in gap.REGIMES:
            key=case['case_id']+'__'+regime
            d=checked((original if case['case_id'] in p['original_case_ids'] else state)['designs'][key]['result'])
            menu=[lookup[(key,key+f'__point_{i:02d}')] for i in d['candidate_ids']]
            for goal in GOALS:
                for method in METHODS:
                    idx=None; chosen=[]; reason=''
                    if method=='no_addition': chosen=[lookup[(key,'baseline')]]
                    elif method=='uniform_random': chosen=menu
                    elif method=='oracle':
                        es=[r['endpoints'][goal]['error'] for r in menu]
                        if all(v is not None for v in es): idx=int(np.argmin(es)); chosen=[menu[idx]]
                        else: reason='No complete endpoint reference/actions for hindsight bound'
                    else:
                        idx=policies[key][goal].get(method)
                        if idx is not None: chosen=[menu[idx]]
                        else: reason='Selector endpoint unsupported'
                    for ep in ('curve',*data.UNITS):
                        val=obs=err=None; unit=''; weak=None; kind='unavailable'
                        if chosen:
                            if ep=='curve':
                                err=learned.gm([r['curve_metrics']['gmfe'] for r in chosen]); kind='observed_curve'
                            else:
                                entries=[r['endpoints'][ep] for r in chosen]; vals=[r['error'] for r in entries]
                                err=(float(np.mean(vals)) if all(v is not None for v in vals) else None) if ep=='tmax' else learned.gm(vals)
                                val=entries[0]['value'] if len(chosen)==1 else None; obs=entries[0]['reference']; unit=entries[0]['unit']; kind=entries[0].get('reference_kind','unavailable')
                                weak=float(np.mean([bool(e.get('weak_tail')) for e in entries]))
                        rows.append(dict(case_id=case['case_id'],drug=case['drug'],cluster_id=case['cluster_id'],cohort=case['expansion_cohort'],
                            cross_study_pair=case.get('cross_study_pair',True),regime=regime,requested_goal=goal,method=method,endpoint=ep,
                            selected_time_h=d['candidate_times_h'][idx] if idx is not None else None,prediction=val,reference=obs,unit=unit,error=err,
                            error_metric='absolute_error_h' if ep=='tmax' else 'curve_gmfe' if ep=='curve' else 'fold_error',
                            reference_kind=kind,weak_tail_fraction=weak,reason=reason))
    return rows


def report():
    from report_blueprint_validation import clustered_geomean, paired_ratio
    s=read(RUN/'state.json'); p=checked(s['protocol']); Guard(p)(force=True)
    validate_prediction_lock(p, s)
    records, policies, failures=all_records(p,s); rows=endpoint_rows(p,s,records,policies)
    with (RUN/'endpoints.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    original=set(p['original_case_ids']); added={c['case_id'] for c in p['cases']}-original
    subsets=dict(original_ten=original,added_twenty=added,all_thirty=original|added,
        added_cross_study={c['case_id'] for c in p['cases'] if c['case_id'] in added and c.get('cross_study_pair',True)},
        added_within_study={c['case_id'] for c in p['cases'] if c['case_id'] in added and not c.get('cross_study_pair',True)},
        prior_extension={c['case_id'] for c in p['cases'] if c['expansion_cohort']=='previous_twenty_extension'},
        new_source_extension={c['case_id'] for c in p['cases'] if c['expansion_cohort']=='new_ten_source_curated'})
    clusters={c['case_id']:c['cluster_id'] for c in p['cases']}; summary={}; comparisons={}
    for subset,ids in subsets.items():
        summary[subset]={}; comparisons[subset]={}
        for regime in gap.REGIMES:
            summary[subset][regime]={}; comparisons[subset][regime]={}
            for goal in GOALS:
                data_by={}; out={}
                for method in METHODS:
                    out[method]={}; data_by[method]={}
                    for label,ep in (('endpoint',goal),('curve','curve')):
                        vals={r['case_id']:r['error'] for r in rows if r['case_id'] in ids and r['regime']==regime and r['requested_goal']==goal and r['method']==method and r['endpoint']==ep and r['error'] is not None}
                        data_by[method][label]=vals
                        stat=clustered_geomean(vals,{k:clusters[k] for k in vals},**p['bootstrap']);stat['attempted_drugs']=len(ids)
                        out[method][label]=stat
                summary[subset][regime][goal]=out
                comparisons[subset][regime][goal]={m:paired_ratio(data_by['learned_gain']['endpoint'],data_by[m]['endpoint'],{k:clusters[k] for k in ids},expected_ids=ids,**p['bootstrap']) for m in METHODS if m!='learned_gain'}
    primary=comparisons['added_twenty']['late_four']['cmax']['time_farthest']
    result=dict(version=p['version'],protocol=s['protocol'],summary=summary,comparisons=comparisons,
        primary_scope='added_twenty/late_four/cmax',failures=failures,
        counts=dict(enrolled_drugs=len(p['cases']),original_drugs=10,added_drugs=len(added),new_layouts=len(p['designs']),new_acquisitions=len(p['jobs']),all_cached_and_new_prediction_records=len(records),endpoint_rows=len(rows)),
        predictions_locked_at=s['predictions_locked_at'],scored_at=time.time(),
        endpoints=ref(RUN/'endpoints.csv'),records=records,
        conclusion=dict(primary_error_ratio=primary['estimate'],ci95=primary['ci95'],improved=primary['ratio_below_one'],worse=primary['ratio_above_one'],complete_pairs=primary['complete_pairs']),
        limitations=p['limitations'])
    save(RUN/'results.json',result);s.update(status='complete',results=ref(RUN/'results.json'),completed_at=time.time());save(RUN/'state.json',s)
    write_report(result)
    return status()


def validate_prediction_lock(p, state):
    """Verify every attempt and fixed choice before opening clinical outcomes."""
    if (state.get('status') not in ('predictions_locked', 'complete')
            or set(state.get('units', {})) != set(p['jobs'])
            or set(state.get('designs', {})) != set(p['designs'])
            or not state.get('predictions_locked_at') or not state.get('action_lock')):
        raise ValueError('Every added-drug target prediction must lock before clinical scoring')
    configure()
    locked = checked(state['action_lock'])
    if locked.get('protocol') != state['protocol'] or locked.get('designs') != state['designs']:
        raise ValueError('Global action lock differs from completed designs')
    for key, entry in state['designs'].items():
        decision = checked(entry['result'])
        gap.validate_decision(p, key, decision)
        feature = checked(entry['features'])
        if (feature.get('protocol') != state['protocol'] or feature.get('decision') != entry['result']
                or feature.get('design_key') != key or feature.get('hidden_values_used') is not False
                or set(feature.get('goals', {})) != set(GOALS)):
            raise ValueError('Learned choice lock identity or visibility contract differs')
        for goal in GOALS:
            for index in feature['goals'][goal]['choices'].values():
                if index is not None and (isinstance(index, bool) or not isinstance(index, int)
                        or not 0 <= index < len(decision['candidate_ids'])):
                    raise ValueError('Learned choice is outside its registered time menu')
    for key, entry in state['units'].items():
        gap.validate_unit(p, key, checked(entry['result']), state['action_lock'])


def write_report(r):
    lines=['# Thirty-drug expansion of sampling-decision evaluation','','The original ten-drug scorer is frozen before evaluation on 20 additional drugs. New-drug errors are not used for training, feature selection, or regularization tuning. Original drugs retain their prior held-out results. Human PBPK is unchanged; RF gap filling for the additional drugs excludes all 30 structures and scaffolds. These remain historical development data, not prospective blinded validation.','',
        'Common primary references are actual clinical sampled peaks and curves; reported scalar endpoints remain separate. AUC infinity, CL, Vz, MRT, and Vss references remain empty when early or terminal observations are missing, while predictions remain. Thirty drugs are not 30 independent trials; Finerenone and Montelukast use different dose arms of the same study.','',
        '| Scope/method | Cmax GMFE | Scorable/enrolled drugs | Curve GMFE |','|---|---:|---:|---:|']
    for scope in ('original_ten','added_twenty','all_thirty','added_cross_study','added_within_study'):
        for method in ('learned_gain','time_farthest','c_optimal','uniform_random','no_addition'):
            e=r['summary'][scope]['late_four']['cmax'][method]; ep=e['endpoint']; cv=e['curve']
            fmt=lambda v:'—' if v is None else f'{v:.4f}'
            lines.append(f"| {scope} / {method} | {fmt(ep['estimate'])} | {ep['n_cases']}/{ep['attempted_drugs']} | {fmt(cv['estimate'])} |")
    c=r['conclusion']; lines+=['',f"Primary conclusion uses the additional 20 drugs: error ratio against time coverage {c['primary_error_ratio']}, study-cluster 95% interval {c['ci95']}; {c['improved']} improved, {c['worse']} worsened, {c['complete_pairs']-c['improved']-c['worse']} tied, and {20-c['complete_pairs']} lacked paired results. Original-ten or pooled results cannot replace this conclusion.",'',
        '[All 11 predictions, references, and errors](endpoints.csv) · [Full results and failures](results.json) · [Frozen protocol](protocol.json) · [30-drug source audit](inputs/cohort_manifest.json)','',
        '```bash','runtime/app-env/bin/python -B scripts/run_sampling_expansion.py status','```','',
        'The interface supports prepare / run / status / report / cleanup. At most 14 workers run; each native build uses two threads and a 180-second timeout. Batch source hashes and the 10 GiB limit are checked every 20 minutes and at stage boundaries. The manuscript is not overwritten.']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')


def status():
    path=RUN/'state.json'
    if not path.exists():
        prep=read(RUN/'preparation.json') if (RUN/'preparation.json').exists() else {}
        return dict(status=prep.get('status','waiting_for_preparation'),native_complete=len(prep.get('completed',{})),native_failures=prep.get('failures',{}))
    s=read(path);p=checked(s['protocol'])
    out=dict(status=s['status'],enrolled_drugs=len(p['cases']),original_cached_drugs=10,added_drugs=len(p['cases'])-10,
        layouts_completed=len(s['designs']),layouts_total=len(p['designs']),actions_completed=len(s['units']),actions_total=len(p['jobs']),reason=s.get('reason'))
    if s['status']=='complete': out['conclusion']=checked(s['results'])['conclusion']
    return out


def cleanup():
    s=read(RUN/'state.json')
    if s['status']!='complete': raise ValueError('Cleanup only after completion')
    n=0
    for path in RUN.rglob('*.tmp'):
        if path.is_file() and not path.is_symlink():path.unlink();n+=1
    return dict(temporary_files_removed=n,scientific_inputs_and_outputs_retained=True)


def run_all(workers):
    """One locked background invocation completes construction and evaluation."""
    prepare(workers)
    return execute(workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','all','status','report','cleanup'])
    parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    if not 1<=args.workers<=14:parser.error('workers must be between 1 and 14')
    if args.command=='status':out=status()
    else:
        RUN.mkdir(parents=True,exist_ok=True)
        with (RUN/'batch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            out=(prepare(args.workers) if args.command=='prepare' else execute(args.workers)
                 if args.command=='run' else run_all(args.workers) if args.command=='all' else globals()[args.command]())
    print(json.dumps(out,ensure_ascii=False,indent=2))
