#!/usr/bin/env python3
"""Frozen early/late evidence-gap pilot with matched clinical endpoint design.

Four visible development observations; one acquired value per action. All
actions lock before revealing values, all predictions lock before scoring.
The finite profile union is not a calibrated posterior or a new PBPK solver.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import math
import os
from pathlib import Path
import time

import numpy as np
import run_endpoint_evidence_design as previous
import blueprint_gap_design_data as data
from pbpk_kb.blueprint_policy import multiaxis_evidence_profile as profiles
from pbpk_kb.blueprint_policy import nonlinear_evidence_design as design

old, base, prior = previous.old, previous.base, previous.prior
ROOT = previous.ROOT
RUN = ROOT / 'experiments/blueprint_design/evidence_gap_design_20261008'
VERSION = 'matched-endpoint-evidence-gap-design-1'
REGIMES = ('early_four', 'late_four')
GOALS = ('cmax', 'auc_inf', 'terminal_half_life')
POLICIES = ('no_addition', 'request_directed', 'linear_covariance',
            'time_farthest', 'maximum_variance', 'uniform_random', 'oracle')
read, ref, checked, verify = previous.read, previous.ref, previous.checked, previous.verify


def save(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    previous.save(path, document)


def request(case, phase, formula):
    req = base.normalized(checked(case['requests']['evidence'][phase]))
    req['partition_method'] = formula
    return req


class Guard:
    def __init__(self, protocol):
        self.protocol, self.last = protocol, 0.

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Evidence-gap batch paused')
        if not force and time.monotonic() - self.last < 1200:
            return
        verify(self.protocol['source_pins'])
        size = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if size > 10 * 1024**3:
            raise RuntimeError('Batch exceeds 10 GiB')
        self.last = time.monotonic()


def get_case(p, cid):
    return next(c for c in p['cases'] if c['case_id'] == cid)


def profile_members(profile, threshold):
    """Equal-weight distinct visible-data solutions, not repeated axis nodes."""
    lower, upper = np.asarray(profile['lower4']), np.asarray(profile['upper4'])
    chosen = {}
    for index, row in enumerate(profile['profile']):
        if not row['successful'] or row['mse'] > threshold:
            continue
        point = np.asarray(row['best']['log_parameters'])
        key = tuple(np.round((point - lower) / (upper - lower), 6))
        if key not in chosen or (row['mse'], index) < (chosen[key][1]['mse'], chosen[key][0]):
            chosen[key] = (index, row)
    return sorted(chosen.values())


def _clinical_metadata_source(case):
    """Steward reads a source but only an explicit metadata whitelist leaves it."""
    document = checked(case['reference_source'])
    if 'cases' in document:
        matches = [c for c in document['cases'] if c.get('case_id') == case['case_id']]
        if len(matches) != 1:
            raise ValueError('Ambiguous clinical protocol metadata')
        return matches[0]
    return document


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists():
        raise ValueError('Frozen protocol exists; cannot overwrite')
    pp = read(previous.RUN / 'protocol.json')
    verify(pp['source_pins'])
    pins = deepcopy(pp['source_pins'])
    cases, designs, jobs = deepcopy(pp['cases']), {}, {}
    for case in cases:
        cid = case['case_id']
        dev = request(case, 'development', 'PKSimStandard')
        full = base._training_document(case['sampling_sources']['full'], dev, ROOT)
        target = request(case, 'target', 'PKSimStandard')
        mw = read(ROOT / case['physchem'])['identity']['molecular_weight_g_mol']
        contract = data.build_target_contract(case, _clinical_metadata_source(case), target,
                                              molecular_weight_g_mol=mw)
        folder = RUN / 'inputs' / cid
        save(folder / 'target_contract.json', contract)
        case['target_contract'] = ref(folder / 'target_contract.json')
        base.pins_in(case['target_contract'], pins)
        for phase in ('development', 'target'):
            for formula in base.FORMULAS:
                prior.get_phys(pp, case, phase, formula, request(case, phase, formula))
        for regime in REGIMES:
            key = cid + '__' + regime
            split = data.split_development(full, dev, regime=regime)
            save(folder / (regime + '.json'), split)
            source = ref(folder / (regime + '.json'))
            base.pins_in(source, pins)
            designs[key] = dict(case_id=cid, regime=regime, visible_source=source)
            for ix, t in zip(split['design_only']['candidate_ids'], split['design_only']['candidate_times_h']):
                jobs[key + f'__point_{ix:02d}'] = dict(design_key=key, case_id=cid,
                    regime=regime, candidate_id=ix, time_h=t)
    for path in (Path(__file__), ROOT / 'scripts/blueprint_gap_design_data.py',
                 ROOT / 'src/pbpk_kb/blueprint_policy/nonlinear_evidence_design.py',
                 ROOT / 'src/pbpk_kb/blueprint_policy/multiaxis_evidence_profile.py',
                 previous.RUN / 'protocol.json'):
        pins[str(path.relative_to(ROOT))] = base.sha256(path)
    p = dict(version=VERSION, created_at=time.time(), cases=cases, designs=designs, jobs=jobs,
        source_pins=pins, physiology_cache=pp['physiology_cache'], goals=list(GOALS),
        regimes=list(REGIMES), visible_points=4, acquired_points=1,
        fit_budget_per_formula=16384, fit_starts=16, profile_grid_points=7,
        profile_budget_per_formula=65536, log_tolerance=math.log(1.1),
        log_noise_sd=.1, noise_sensitivity=[.05, .1, .2], quadrature_nodes=31,
        method='Union of all four coordinate profiles in both partition formulas; uniform finite feasible set, nonlinear expected log-endpoint variance reduction under a declared log-Gaussian measurement model.',
        collection='Converged coordinate profile rows within global visible MSE minimum + ln(1.1)^2. Deduplicate each formula by normalized log-coordinates rounded to six decimals, retain smallest visible MSE with index tie-break. Not a full four-dimensional posterior.',
        fitting='Original four-axis calibration with fixed independent initialization, 16 starts and 16384 actual calls per formula; no full-curve or previous-fit warm start.',
        goal_estimation='Exact same sampled-prediction NCA and units as final clinical scoring, using whitelisted study protocol metadata only.',
        data_isolation='Steward validates development sources and extracts target protocol metadata; action API sees four development concentrations, candidate times, model outputs and metadata, never hidden development or target concentration/endpoint values.',
        primary=dict(goal='cmax', regime='late_four'),
        secondary=[dict(goal='auc_inf', regime='early_four'), dict(goal='terminal_half_life', regime='early_four')],
        reverse_regimes='All three endpoints in both regimes reported; reverse cases are sensitivity analyses, never replace primary.',
        expansion_gate='Primary Cmax late-four GMFE <=0.99x time, maximum-variance and exact random separately; >=6/10 directional wins against time; curve <=1.01x both time and no-addition; both prespecified early-four secondary endpoints <=1.05x no-addition; all ten scored in required comparisons, no failed acquisitions. Keep all failures. Engineering feasibility only, not significance or clinical acceptance.',
        controls='Same visible data, one added observation and identical fit budgets. Linear-covariance ablation uses the same new collection and matched estimator; it is not an exact rerun of the previous five-point method. Random averages log errors over all possible single actions; oracle is hindsight only.',
        limits=['Ten repeatedly viewed development drugs; target-study isolation does not make these new blinded cases.',
                'Early/late truncation removes concentration information, not drug-component parameters. No missing-CL or zero-curve claim.',
                'Finite profile union omits joint directions and shared model discrepancy; assumed noise is not calibrated clinical uncertainty.',
                'Retrospective acquisition of a hidden point is not a newly conducted patient experiment.'],
        monitor_seconds=1200, directory_limit_gib=10)
    verify(pins)
    save(RUN / 'protocol.json', p)
    save(RUN / 'state.json', dict(status='prepared', protocol=ref(RUN / 'protocol.json'), designs={}, units={}))
    return dict(status='prepared', drugs=len(cases), designs=len(designs), acquisitions=len(jobs))


def fit_predict(p, case, training, folder, guard, identity):
    """Same frozen fitter for a four-point baseline and every five-point action."""
    result_path = folder / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if result.get('protocol_sha256') != base.sha256(RUN / 'protocol.json') or result.get('identity') != identity:
            raise ValueError('Changed resumed prediction identity')
        verify(result['output_pins'])
        if checked(result['training']) != training:
            raise ValueError('Changed resumed visible data')
        return result
    start = time.monotonic()
    if (folder / 'training.json').exists() and read(folder / 'training.json') != training:
        raise ValueError('Changed partially completed visible data')
    save(folder / 'training.json', training)
    fits, calls, failures = [], 0, 0
    for formula in base.FORMULAS:
        req = request(case, 'development', formula)
        phys = prior.get_phys(p, case, 'development', formula, req)
        path = folder / (formula + '.json')
        if path.exists():
            fit = read(path)
            if (fit.get('protocol_sha256') != base.sha256(RUN / 'protocol.json')
                    or fit.get('training_source') != ref(folder / 'training.json')
                    or fit.get('request') != req or fit.get('no_warm_start') is not True
                    or fit.get('call_limit') != p['fit_budget_per_formula']
                    or len(fit.get('starts', [])) != p['fit_starts']
                    or fit.get('visible_points') != len(training['observations'])
                    or fit.get('variant') != 'blood_partition'
                    or fit.get('target_observations_used') is not False
                    or not 0 <= fit.get('calls', -1) <= p['fit_budget_per_formula']
                    or fit.get('development_physiology') != p['physiology_cache'][case['case_id'] + '__development__' + formula]):
                raise ValueError('Changed resumed formula fit')
        else:
            fit = old.optimizer.fit(lambda x, t: old.simulate('blood_partition', phys, req, t, x),
                req, training, phys, variant='blood_partition', budget=p['fit_budget_per_formula'],
                starts=p['fit_starts'], before_trial=guard)
            fit.update(protocol_sha256=base.sha256(RUN / 'protocol.json'), request=req,
                training_source=ref(folder / 'training.json'), development_physiology=p['physiology_cache'][case['case_id'] + '__development__' + formula])
            save(path, fit)
        fits.append(dict(formula=formula, fit=ref(path), best=fit['best']))
        calls += fit['calls']; failures += fit['failures']
    result = dict(identity=identity, protocol_sha256=base.sha256(RUN / 'protocol.json'),
        status='failed', training=ref(folder / 'training.json'), fits=fits,
        resources=dict(objective_calls=calls, failed_calls=failures, prediction_calls=0))
    valid = [f for f in fits if f['best'] is not None]
    if valid:
        best = min(valid, key=lambda f: f['best']['objective'])
        formula = best['formula']; target = request(case, 'target', formula)
        save(folder / 'selection.json', dict(formula=formula, parameters=best['best'],
             locked_at=time.time(), target_values_used=False))
        phys = prior.get_phys(p, case, 'target', formula, target)
        grid = base.engine.prediction_grid(target)
        guard(); result['resources']['prediction_calls'] = 1
        try:
            pred = old.simulate('blood_partition', phys, target, grid, best['best']['log_parameters'])
            save(folder / 'prediction.json', prior.curve_document(pred, grid, target))
            save(folder / 'request.json', target)
            result.update(status='predicted', formula=formula, parameters=best['best'],
                prediction=ref(folder / 'prediction.json'), request=ref(folder / 'request.json'),
                selection=ref(folder / 'selection.json'), mass_balance=pred['mass_balance'])
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            result['reason'] = str(exc)
    else:
        result['reason'] = 'No feasible formula fit'
    result['resources']['wall_seconds'] = time.monotonic() - start
    result['output_pins'] = {}
    base.pins_in({k: v for k, v in result.items() if k != 'output_pins'}, result['output_pins'])
    guard(force=True); save(result_path, result)
    return result


def validate_decision(p, key, receipt):
    spec = p['designs'][key]; menu = checked(spec['visible_source'])['design_only']
    if (receipt.get('protocol_sha256') != base.sha256(RUN / 'protocol.json')
            or receipt.get('design_key') != key or receipt.get('visible_source') != spec['visible_source']
            or receipt.get('target_contract') != get_case(p, spec['case_id'])['target_contract']
            or receipt.get('candidate_ids') != menu['candidate_ids']
            or receipt.get('candidate_times_h') != menu['candidate_times_h']
            or receipt.get('hidden_values_used') is not False or receipt.get('target_values_used') is not False):
        raise ValueError('Decision resume or isolation mismatch')
    if set(receipt.get('choices', {})) != {'request_directed', 'linear_covariance', 'time_farthest', 'maximum_variance'}:
        raise ValueError('Changed decision policy set')
    for mapping in receipt['choices'].values():
        if set(mapping) != set(GOALS):
            raise ValueError('Changed requested endpoint set')
        for index in mapping.values():
            if index is not None and (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(menu['candidate_ids'])):
                raise ValueError('Action outside time-only candidate menu')
    verify(receipt['output_pins'])
    baseline = checked(receipt['baseline'])
    if baseline.get('identity') != dict(design_key=key, candidate_id=None) or baseline.get('protocol_sha256') != receipt['protocol_sha256']:
        raise ValueError('Baseline belongs to a different visible-data regime')


def validate_unit(p, key, result, lock):
    if (result.get('identity') != dict(job=p['jobs'][key], action_lock=lock)
            or result.get('protocol_sha256') != base.sha256(RUN / 'protocol.json')
            or result.get('status') not in ('predicted', 'failed')):
        raise ValueError('Changed predicted action identity')
    verify(result['output_pins'])


def design_one(p, key):
    guard = Guard(p); guard(force=True)
    spec = p['designs'][key]; case = get_case(p, spec['case_id'])
    folder = RUN / 'designs' / key; path = folder / 'locked.json'
    if path.exists():
        receipt = read(path); validate_decision(p, key, receipt)
        return dict(result=ref(path))
    visible = checked(spec['visible_source']); training = visible['visible_training']
    baseline = fit_predict(p, case, training, folder / 'baseline', guard, dict(design_key=key, candidate_id=None))
    profs, calls = {}, 0
    for formula in base.FORMULAS:
        req = request(case, 'development', formula)
        phys = prior.get_phys(p, case, 'development', formula, req)
        protection = prior.blood._evidence_policy(phys, req, 0.)
        prior.blood._native_source(phys)
        if protection['explicit_measured_records']:
            raise ValueError('Protected measured blood partition is not eligible for this profile-union pilot')
        tt, yy = old.optimizer._visible(req, training)
        lower, upper, _, _, _ = old.optimizer._initialization(req, phys, 'blood_partition', 2)
        fp = folder / ('profile_' + formula + '.json')
        if fp.exists():
            prof = read(fp)
            if (prof.get('protocol_sha256') != base.sha256(RUN / 'protocol.json')
                    or prof.get('visible_source') != spec['visible_source'] or prof.get('formula') != formula
                    or prof.get('max_calls') != p['profile_budget_per_formula']
                    or prof.get('requested_grid_points') != p['profile_grid_points']
                    or prof.get('visible_points') != 4 or prof.get('external_warm_start_used') is not False
                    or prof.get('lower4') != lower.tolist() or prof.get('upper4') != upper.tolist()):
                raise ValueError('Changed resumed profile')
        else:
            prof = profiles.analyze(lambda x, t: old.simulate('blood_partition', phys, req, t, x),
                tt, yy, lower, upper, grid_points=p['profile_grid_points'],
                max_calls=p['profile_budget_per_formula'], before_trial=guard)
            prof.update(protocol_sha256=base.sha256(RUN / 'protocol.json'), visible_source=spec['visible_source'], formula=formula)
            save(fp, prof)
        profs[formula] = prof; calls += prof['calls']
    eligible = [r['mse'] for prof in profs.values() for r in prof['profile'] if r['successful']]
    if not eligible:
        raise ValueError('No converged visible-data profile members')
    threshold = min(eligible) + p['log_tolerance']**2
    members, concentrations, values, failed, simulation_calls = [], [], {g: [] for g in GOALS}, [], 0
    menu = visible['design_only']; contract = checked(case['target_contract'])
    for formula, prof in profs.items():
        dev = request(case, 'development', formula); target = request(case, 'target', formula)
        dp = prior.get_phys(p, case, 'development', formula, dev)
        tp = prior.get_phys(p, case, 'target', formula, target)
        grid = base.engine.prediction_grid(target)
        for ri, row in profile_members(prof, threshold):
            x = np.asarray(row['best']['log_parameters'])
            try:
                guard(); simulation_calls += 1
                concentrations_at_menu = old.simulate('blood_partition', dp, dev, menu['candidate_times_h'], x)['concentration_mg_l']
                if not np.isfinite(concentrations_at_menu).all() or np.any(np.asarray(concentrations_at_menu) <= 0):
                    raise ValueError('Nonpositive candidate model concentration')
                guard(); simulation_calls += 1
                pred = old.simulate('blood_partition', tp, target, grid, x)
                goal = data.model_goals(contract, prior.curve_document(pred, grid, target), target)
                members.append(dict(formula=formula, profile_index=ri, log_parameters=x.tolist(), mse=row['mse'], goal_diagnostics=goal))
                concentrations.append(concentrations_at_menu)
                for g in GOALS:
                    values[g].append(goal['values'][g])
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failed.append(dict(formula=formula, profile_index=ri, reason=str(exc)))
    if len(members) < 2:
        raise ValueError('Insufficient finite model collection')
    cc = np.asarray(concentrations)
    generic = previous.design.choose(menu['candidate_times_h'], menu['observed_times_h'], cc, {'constant': np.ones(len(cc))})
    choices = {m: {} for m in ('request_directed', 'linear_covariance', 'time_farthest', 'maximum_variance')}
    details = {}
    for g in GOALS:
        choices['time_farthest'][g] = generic['policies']['time_farthest']['selected_index']
        choices['maximum_variance'][g] = generic['policies']['max_variance']['selected_index']
        good = np.asarray([v is not None and math.isfinite(v) and v > 0 for v in values[g]])
        if good.sum() < 2:
            choices['request_directed'][g] = choices['linear_covariance'][g] = None
            details[g] = dict(status='unsupported', members=int(good.sum()))
            continue
        vv = np.asarray([v for v, yes in zip(values[g], good) if yes])
        sensitivity = {}
        for sigma in p['noise_sensitivity']:
            sensitivity[str(sigma)] = design.choose(menu['candidate_times_h'], menu['observed_times_h'],
                cc[good], {g: vv}, log_noise_sd=sigma, quadrature_nodes=p['quadrature_nodes'])
        main = sensitivity[str(p['log_noise_sd'])]
        choices['request_directed'][g] = main['policies']['nonlinear_goal_directed'][g]['selected_index']
        choices['linear_covariance'][g] = main['policies']['goal_directed'][g]['selected_index']
        details[g] = dict(status='selected', members=int(good.sum()), missing_members=int((~good).sum()), sensitivity=sensitivity)
    receipt = dict(protocol_sha256=base.sha256(RUN / 'protocol.json'), design_key=key, locked_at=time.time(),
        visible_source=spec['visible_source'], target_contract=case['target_contract'], candidate_ids=menu['candidate_ids'],
        candidate_times_h=menu['candidate_times_h'], baseline=ref(folder / 'baseline/result.json'),
        profile_sources={f: ref(folder / ('profile_' + f + '.json')) for f in base.FORMULAS},
        threshold=threshold, members=members, concentrations_mg_l=cc.tolist(), goal_values=values,
        choices=choices, details=details, member_failures=failed,
        resources=dict(profile_calls=calls, committee_simulations=simulation_calls),
        hidden_values_used=False, target_values_used=False, output_pins={})
    for field in ('baseline', 'profile_sources'):
        base.pins_in(receipt[field], receipt['output_pins'])
    receipt['output_pins'].update(baseline['output_pins'])
    validate_decision(p, key, receipt); guard(force=True); save(path, receipt)
    return dict(result=ref(path))


def execute(p, key, job, lock):
    guard = Guard(p); guard(force=True)
    receipt = checked(lock)
    if receipt['protocol'] != ref(RUN / 'protocol.json') or set(receipt['designs']) != set(p['designs']):
        raise ValueError('All actions must lock before revealing any concentration')
    if p['jobs'].get(key) != job:
        raise ValueError('Changed acquisition job')
    d = checked(receipt['designs'][job['design_key']]['result'])
    validate_decision(p, job['design_key'], d)
    if (job['candidate_id'], job['time_h']) not in list(zip(d['candidate_ids'], d['candidate_times_h'])):
        raise ValueError('Action is not registered')
    case = get_case(p, job['case_id'])
    full = base._training_document(case['sampling_sources']['full'], request(case, 'development', 'PKSimStandard'), ROOT)
    visible = checked(p['designs'][job['design_key']]['visible_source'])['visible_training']
    training = data.reveal_one(full, visible, job['time_h'])
    if len(training['observations']) != 5:
        raise ValueError('An action must reveal exactly one observation')
    result = fit_predict(p, case, training, RUN / 'work' / key, guard, dict(job=job, action_lock=lock))
    return dict(result=ref(RUN / 'work' / key / 'result.json'), status=result['status'])


def _parallel(state, p, keys, worker, count, extra=()):
    with ProcessPoolExecutor(max_workers=count) as pool:
        tasks = {pool.submit(worker, p, k, *extra): k for k in keys}
        try:
            for future in as_completed(tasks):
                key = tasks[future]
                state['designs'][key] = future.result()
                save(RUN / 'state.json', state)
                print('designs', len(state['designs']), len(p['designs']), flush=True)
        except BaseException as exc:
            save(RUN / 'PAUSE', dict(reason=str(exc)))
            for f in tasks:
                f.cancel()
            raise


def run(workers):
    state = read(RUN / 'state.json'); p = checked(state['protocol']); Guard(p)(force=True)
    if state['status'] == 'complete':
        return status()
    state.update(status='designing', pid=os.getpid(), started_at=state.get('started_at', time.time()))
    save(RUN / 'state.json', state)
    try:
        for key, entry in state['designs'].items():
            validate_decision(p, key, checked(entry['result']))
        _parallel(state, p, [k for k in p['designs'] if k not in state['designs']], design_one, workers)
        if set(state['designs']) != set(p['designs']):
            raise ValueError('Incomplete global action lock')
        lockpath = RUN / 'actions_locked.json'
        if not lockpath.exists():
            save(lockpath, dict(protocol=state['protocol'], designs=state['designs'], locked_at=time.time()))
        lock = ref(lockpath); locked = checked(lock)
        if locked['protocol'] != state['protocol'] or locked['designs'] != state['designs']:
            raise ValueError('Changed global action receipt')
        state.update(status='fitting', action_lock=lock); save(RUN / 'state.json', state)
        for key, entry in state['units'].items():
            if key not in p['jobs']:
                raise ValueError('Unknown resumed acquisition')
            validate_unit(p, key, checked(entry['result']), lock)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(execute, p, k, j, lock): k for k, j in p['jobs'].items() if k not in state['units']}
            try:
                for future in as_completed(tasks):
                    state['units'][tasks[future]] = future.result()
                    save(RUN / 'state.json', state)
                    print('acquisitions', len(state['units']), len(p['jobs']), flush=True)
            except BaseException as exc:
                save(RUN / 'PAUSE', dict(reason=str(exc)))
                for f in tasks:
                    f.cancel()
                raise
        Guard(p)(force=True)
        state.update(status='predictions_locked', locked_at=time.time()); save(RUN / 'state.json', state)
        return report()
    except BaseException as exc:
        save(RUN / 'PAUSE', dict(reason=str(exc)))
        state.update(status='paused', reason=str(exc)); save(RUN / 'state.json', state)
        raise


def expansion_checks(summary, comparisons, units):
    primary = summary['late_four']['cmax']; pair = comparisons['late_four']['cmax']
    def value(v):
        return v is not None and math.isfinite(v) and v > 0
    def no_harm(a, b, tolerance, field):
        return a[field]['n_cases'] == b[field]['n_cases'] == 10 and value(a[field]['estimate']) and value(b[field]['estimate']) and a[field]['estimate'] <= tolerance * b[field]['estimate']
    return dict(all_ten_primary=all(primary[m]['endpoint']['n_cases'] == 10 for m in POLICIES),
        one_percent_better_all_generic=all(value(pair[m]['estimate']) and pair[m]['estimate'] <= .99 for m in ('time_farthest', 'maximum_variance', 'uniform_random')),
        six_cmax_wins=pair['time_farthest']['ratio_below_one'] >= 6,
        curve_not_harmed=all(no_harm(primary['request_directed'], primary[m], 1.01, 'curve') for m in ('no_addition', 'time_farthest')),
        early_secondary_not_harmed=all(no_harm(summary['early_four'][g]['request_directed'], summary['early_four'][g]['no_addition'], 1.05, 'endpoint') for g in ('auc_inf', 'terminal_half_life')),
        no_failed_acquisitions=bool(units) and all(u['status'] == 'predicted' for u in units.values()))


def report():
    from report_blueprint_validation import Store, _case_reference, _score_prediction, clustered_geomean, paired_ratio
    state = read(RUN / 'state.json'); p = checked(state['protocol']); verify(p['source_pins'])
    if state['status'] not in ('predictions_locked', 'complete') or set(state['units']) != set(p['jobs']) or set(state['designs']) != set(p['designs']):
        raise ValueError('All actions and predictions must be locked')
    lock = checked(state['action_lock'])
    if lock['designs'] != state['designs'] or lock['protocol'] != state['protocol']:
        raise ValueError('Changed action lock')
    decisions = {k: checked(v['result']) for k, v in state['designs'].items()}
    units = {k: checked(v['result']) for k, v in state['units'].items()}
    for key, d in decisions.items():
        validate_decision(p, key, d)
    for key, u in units.items():
        validate_unit(p, key, u, state['action_lock'])
    store = Store(ROOT); records, goal_records, csv_rows = [], [], []
    references = {c['case_id']: _case_reference(c, store) for c in p['cases']}
    for key, spec in p['designs'].items():
        case = get_case(p, spec['case_id']); clinical, curve, endpoints = references[case['case_id']]
        decision = decisions[key]; baseline = checked(decision['baseline'])
        relevant = [('baseline', baseline)] + [(k, u) for k, u in units.items() if p['jobs'][k]['design_key'] == key]
        scores = {}
        for candidate, unit in relevant:
            rec = dict(design_key=key, case_id=case['case_id'], drug=case['drug'], regime=spec['regime'], candidate=candidate, status=unit['status'])
            if unit['status'] == 'predicted':
                score, _, _ = _score_prediction(case, clinical, curve, endpoints, unit, store)
                score.pop('curve', None); rec.update(score)
            else:
                rec.update(reason=unit.get('reason'), curve_metrics={'gmfe': None}, endpoints={ep: dict(value=None, reference=v['value'], unit=v['unit'], error=None) for ep, v in endpoints.items()})
            scores[candidate] = rec; records.append(rec)
            common = {k: rec[k] for k in ('drug', 'regime', 'candidate', 'status')}
            csv_rows.append(common | dict(endpoint='curve', prediction=None, reference=None, error=rec['curve_metrics'].get('gmfe'), unit=curve['unit'], weak_tail=None))
            csv_rows.extend(common | dict(endpoint=ep, prediction=v.get('value'), reference=v.get('reference'), error=v.get('error'), unit=v.get('unit'), weak_tail=v.get('weak_tail')) for ep, v in rec['endpoints'].items())
        keys = [key + f'__point_{ix:02d}' for ix in decision['candidate_ids']]
        for goal in GOALS:
            errors = [scores[k]['endpoints'][goal].get('error') for k in keys]
            curves = [scores[k]['curve_metrics'].get('gmfe') for k in keys]
            complete = all(v is not None and math.isfinite(v) and v > 0 for v in errors)
            row = dict(design_key=key, case_id=case['case_id'], drug=case['drug'], regime=spec['regime'], goal=goal, policies={})
            for policy in POLICIES:
                index = None
                if policy == 'no_addition':
                    rec = scores['baseline']
                elif policy in decision['choices']:
                    index = decision['choices'][policy][goal]
                    if index is None:
                        row['policies'][policy] = dict(error=None, curve_gmfe=None, reason='Insufficient matched endpoint members')
                        continue
                    rec = scores[keys[index]]
                elif policy == 'oracle':
                    if not complete:
                        row['policies'][policy] = dict(error=None, curve_gmfe=None, reason='Incomplete candidate errors')
                        continue
                    index = int(np.argmin(errors)); rec = scores[keys[index]]
                else:
                    row['policies'][policy] = dict(error=math.exp(float(np.mean(np.log(errors)))) if complete else None,
                        curve_gmfe=math.exp(float(np.mean(np.log(curves)))) if all(v is not None and math.isfinite(v) and v > 0 for v in curves) else None,
                        semantics='Exact expected log error for one uniformly selected observation')
                    continue
                ep = rec['endpoints'][goal]
                row['policies'][policy] = dict(error=ep.get('error'), prediction=ep.get('value'), reference=ep.get('reference'), weak_tail=ep.get('weak_tail'),
                    curve_gmfe=rec['curve_metrics'].get('gmfe'), selected_index=index, selected_time_h=None if index is None else decision['candidate_times_h'][index])
            goal_records.append(row)
    clusters = {c['case_id']: c['cluster_id'] for c in p['cases']}
    summary, comparisons = {}, {}
    for regime in REGIMES:
        summary[regime] = {}; comparisons[regime] = {}
        for goal in GOALS:
            selected = [r for r in goal_records if r['regime'] == regime and r['goal'] == goal]
            summary[regime][goal] = {}; comparisons[regime][goal] = {}
            scores = {m: {r['case_id']: r['policies'][m]['error'] for r in selected} for m in POLICIES}
            for m in POLICIES:
                v = {k: x for k, x in scores[m].items() if x is not None and math.isfinite(x) and x > 0}
                cr = {r['case_id']: r['policies'][m]['curve_gmfe'] for r in selected}
                cr = {k: x for k, x in cr.items() if x is not None and math.isfinite(x) and x > 0}
                summary[regime][goal][m] = dict(endpoint=clustered_geomean(v, {k: clusters[k] for k in v}), curve=clustered_geomean(cr, {k: clusters[k] for k in cr}), missing=10-len(v), over_twofold=sum(x > 2 for x in v.values()))
                if m != 'request_directed':
                    comparisons[regime][goal][m] = paired_ratio(scores['request_directed'], scores[m], clusters)
    gates = expansion_checks(summary, comparisons, units)
    all_units = list(units.values()) + [checked(d['baseline']) for d in decisions.values()]
    result = dict(version=VERSION, scored_at=time.time(), prediction_lock=state['locked_at'],
        summary=summary, comparisons=comparisons, expansion_checks=gates, expand_to_twenty=all(gates.values()),
        records=records, goal_records=goal_records,
        resources=dict(fit_calls=sum(u['resources']['objective_calls'] for u in all_units),
            target_predictions=sum(u['resources']['prediction_calls'] for u in all_units),
            profile_calls=sum(d['resources']['profile_calls'] for d in decisions.values()),
            committee_simulations=sum(d['resources']['committee_simulations'] for d in decisions.values()),
            elapsed_seconds=time.time()-state['started_at']), limitations=p['limits'])
    save(RUN / 'results.json', result)
    with (RUN / 'endpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    lines = ['# Matched-endpoint evidence-gap pilot', '',
        'Ten development drugs; four early or four late observations plus one acquired point. Lower GMFE is better.', '',
        '| Regime | Goal | No addition | Nonlinear goal | Linear goal | Time | Max variance | Random | Oracle |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for regime in REGIMES:
        for goal in GOALS:
            vals = [summary[regime][goal][m]['endpoint']['estimate'] for m in POLICIES]
            lines.append('| ' + regime + ' | ' + goal + ' | ' + ' | '.join('NA' if x is None else f'{x:.4f}' for x in vals) + ' |')
    lines += ['', f'Expansion permitted: **{result["expand_to_twenty"]}**. Prespecified checks: `{gates}`.',
        '', 'Primary: late-four Cmax. Secondary: early-four AUC infinity and terminal half-life. Reverse regimes are sensitivity analyses; no favorable endpoint substitution.',
        '', 'All 11 endpoint categories, failures and unavailable references are retained in `endpoints.csv`. Random is expected log error; oracle uses outcomes and cannot be deployed.',
        '', '```bash', 'runtime/app-env/bin/python -B scripts/run_evidence_gap_design.py status', '```']
    (RUN / 'README.md').write_text('\n'.join(lines) + '\n')
    state.update(status='complete', completed_at=time.time(), results=ref(RUN / 'results.json'))
    save(RUN / 'state.json', state)
    return dict(status='complete', expansion_checks=gates, summary=summary)


def status():
    state = read(RUN / 'state.json'); p = checked(state['protocol'])
    return dict(status=state['status'], designs=len(state['designs']), total_designs=len(p['designs']),
        acquisitions=len(state['units']), total_acquisitions=len(p['jobs']), pid=state.get('pid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 14:
        parser.error('workers must be 1..14')
    print({'prepare': prepare, 'run': lambda: run(args.workers), 'status': status, 'report': report}[args.command](), flush=True)
