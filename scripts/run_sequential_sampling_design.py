#!/usr/bin/env python3
"""Sequential, outcome-isolated PBPK sampling pilot on ten development drugs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import run_evidence_gap_design as previous
import blueprint_sequential_design_data as sequential
from pbpk_kb.blueprint_policy import refit_sampling_design as acquisition
from pbpk_kb.blueprint_policy import classical_sampling_design as classical

ROOT = previous.ROOT
RUN = ROOT / 'experiments/blueprint_design/sequential_sampling_20261008'
VERSION = 'sequential-refit-endpoint-design-1'
old, base, prior = previous.old, previous.base, previous.prior
read, save, ref, checked, verify = previous.read, previous.save, previous.ref, previous.checked, previous.verify
request = previous.request
METHODS = ('refit_goal', 'posterior_variance', 'c_optimal', 'time_farthest',
           'random_0', 'random_1', 'random_2')
REPORT_METHODS = METHODS[:4] + ('random',)
GOAL = 'cmax'


class Guard:
    def __init__(self, protocol):
        self.p, self.last = protocol, 0.

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Batch paused; inspect PAUSE before resuming')
        if not force and time.monotonic()-self.last < 1200:
            return
        verify(self.p['source_pins'])
        size = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if size > 10*1024**3:
            raise RuntimeError('Batch exceeds 10 GiB')
        self.last = time.monotonic()


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise ValueError('Frozen protocol already exists')
    previous_p = read(previous.RUN/'protocol.json')
    verify(previous_p['source_pins'])
    pins = deepcopy(previous_p['source_pins'])
    cases = deepcopy(previous_p['cases'])
    layouts = {}
    for case in cases:
        cid = case['case_id']; req = request(case, 'development', 'PKSimStandard')
        full = base._training_document(case['sampling_sources']['full'], req, ROOT)
        for li in (0, 1):
            split = sequential.split_development(full, req, layout_index=li)
            path = RUN/'inputs'/f'{cid}__layout{li}.json'; save(path, split)
            layouts[f'{cid}__layout{li}'] = dict(case_id=cid, layout_index=li, source=ref(path))
            base.pins_in(ref(path), pins)
    new_sources = [Path(__file__), ROOT/'scripts/blueprint_sequential_design_data.py',
        ROOT/'src/pbpk_kb/blueprint_policy/refit_sampling_design.py',
        ROOT/'src/pbpk_kb/blueprint_policy/classical_sampling_design.py', previous.RUN/'protocol.json']
    for path in new_sources:
        pins[str(path.relative_to(ROOT))] = base.sha256(path)
    # Include the actual author-code dependency, its license and provenance.
    for path in classical.source_files():
        pins[str(Path(path).relative_to(ROOT))] = base.sha256(path)
    p = dict(version=VERSION, created_at=time.time(), cases=cases, layouts=layouts,
        source_pins=pins, physiology_cache=previous_p['physiology_cache'],
        methods=list(METHODS), budgets=[0,1,2,3], goal=GOAL,
        fit_budget_per_formula=16384, fit_starts=16, local_refit_calls_per_formula=128,
        proportional_noise_sd=.1, additive_floor_fraction=.01,
        local_scenarios_per_formula=2, scenario_objective_tolerance=math.log(1.1)**2,
        formula_selection='Visible penalized log loss; independent real-data initialization.',
        proposal='At most two diverse near-optimal multistart solutions per formula. Simulate three fixed log-noise nodes; locally refit both formulas and minimize squared log Cmax error to generating scenario. Approximation, not posterior or exact full-refit value.',
        measurement_noise='Engineering assumption: log SD=sqrt(log(1+0.1^2+(0.01*max(visible concentration)/predicted concentration)^2)). No invented assay precision or target-study LLOQ reused as development evidence.',
        classical='Official PyApprox COptimalCriterion applied to whitened log-concentration Jacobians and target log-Cmax gradient. Noise is frozen at local predictions, omitting variance derivatives. Prespecified engineering precision maps to the fitter only in the homoscedastic limit; not its exact heteroscedastic Hessian. Candidate added once; all criteria recorded.',
        variance_ablation='Expected posterior log-Cmax variance on the same finite scenarios, Gaussian mixture quadrature; same mixed-noise assumption. This is not an exact repeat of the old profile-union algorithm.',
        layout='Two seeded mixed four-point layouts using times alone; both halves represented; seed 20261008/20261009; duplicate second layout rejected.',
        random='Three fixed random permutations per drug/layout; expected log error estimated over these three, not exact all-sequence expectation.',
        isolation='Lock each action before revealing its development concentration. Fit cache keys contain exact visible IDs, request, source and protocol. All target predictions lock before target values are loaded for scoring. No previous/full fit warm starts.',
        primary='Cmax: drug-equal average log fold error over two layouts and budgets 1,2,3. Report every budget separately as well. Neither layouts nor budgets create additional independent drugs.',
        secondary='Target curve, AUC infinity, terminal half-life and all original 11 endpoint categories; no claim that Cmax-selected actions optimize other requested endpoints.',
        efficiency='Posthoc budget needed for Cmax FE<=2 and <=1.10 times full-development-data FE; failures/not-reached retained. Full-data fit is an information reference, not guaranteed best and not a deployable stopping rule.',
        expansion_gate='Primary Cmax GMFE (log error averaged within drug across layouts and acquired budgets) <=0.99 times time, classical c-opt and random separately; at least 6/10 directional wins versus time; curve GMFE <=1.01 times time; AUC infinity and half-life GMFE <=1.05 times no-addition; all cases/policies complete. Engineering gate, not significance or clinical acceptance.',
        limitations=['Ten repeatedly viewed development drugs; not new blinded validation.',
            'Retrospective hidden clinical development points, not newly collected patients.',
            'Four observed points are required; no zero-curve/new-drug accuracy claim.',
            'Finite near-optimal solutions are not a calibrated posterior; local hypothetical refit may disagree with final multistart refit.',
            'Common structure error and assay misspecification can defeat all design scores.'],
        monitor_seconds=1200, directory_limit_gib=10)
    verify(pins); save(RUN/'protocol.json', p)
    save(RUN/'state.json', dict(status='prepared', protocol=ref(RUN/'protocol.json'), completed={}))
    return dict(status='prepared', drugs=len(cases), layouts=len(layouts), sequences=len(layouts)*len(METHODS), target_slots=len(layouts)*(1+3*len(METHODS))+len(cases))


def subset_key(case, ids, p):
    content = dict(case_id=case['case_id'], visible_ids=sorted(ids),
        request=case['requests']['evidence']['development'], source=case['sampling_sources']['full'],
        protocol=base.sha256(RUN/'protocol.json'))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:24]


def cached_fit(p, case, training, ids, guard):
    key = subset_key(case, ids, p)
    folder = RUN/'fits'/case['case_id']/key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        fit_predict(p, case, training, folder, guard, dict(case_id=case['case_id'], visible_ids=sorted(ids)))
    return ref(folder/'result.json')


def models_for(p, case, fit, training, guard, counters=None):
    models = []; contract = checked(case['target_contract'])
    for fr in fit['fits']:
        f = checked(fr['fit'])
        if f['best'] is None:
            continue
        formula = fr['formula']; dev = request(case, 'development', formula); target = request(case, 'target', formula)
        devphys = prior.get_phys(p, case, 'development', formula, dev)
        targetphys = prior.get_phys(p, case, 'target', formula, target)
        grid = base.engine.prediction_grid(target)
        x = np.asarray(f['best']['log_parameters']); lower=np.asarray(f['parameter_lower']); upper=np.asarray(f['parameter_upper'])
        pool = [np.asarray(s['best']['log_parameters']) for s in f['starts']
                if s['best'] is not None and s['best']['objective'] <= f['best']['objective']+p['scenario_objective_tolerance']]
        # Two fixed model formulas receive the same scenario cap; no clinical-outcome weights.
        members = [x]
        if pool:
            far = max(pool, key=lambda v: float(np.linalg.norm((v-x)/(upper-lower))))
            if np.linalg.norm((far-x)/(upper-lower)) > 1e-6:
                members.append(far)
        def predict(v, times, phys=devphys, req=dev):
            guard()
            if counters is not None:
                counters['development']+=1
            return old.simulate('blood_partition', phys, req, times, v)['concentration_mg_l']
        def goal(v, phys=targetphys, req=target, times=grid):
            guard()
            if counters is not None:
                counters['target']+=1
            pred = old.simulate('blood_partition', phys, req, times, v)
            val = previous.data.model_goals(contract, prior.curve_document(pred, times, req), req)['values'][GOAL]
            if val is None:
                raise ValueError('Protocol-matched Cmax unavailable')
            return val
        models.append(dict(label=formula, x=x, lower=lower, upper=upper, members=members, predict=predict, goal=goal))
    selected = next(i for i,m in enumerate(models) if m['label'] == fit['formula'])
    return models, selected


def finite_derivatives(model, visible_times, candidate_times):
    x=np.asarray(model['x']); lower=model['lower']; upper=model['upper']; times=np.r_[visible_times,candidate_times]
    order=np.argsort(times); inverse=np.argsort(order)
    def predicted(v):
        return np.asarray(model['predict'](v,times[order]))[inverse]
    basec=predicted(x); js=[]; gs=[]
    for axis in range(4):
        a=x.copy(); b=x.copy(); a[axis]=max(lower[axis],x[axis]-1e-4); b[axis]=min(upper[axis],x[axis]+1e-4)
        width=b[axis]-a[axis]
        js.append((np.log(predicted(b))-np.log(predicted(a)))/width)
        gs.append((math.log(model['goal'](b))-math.log(model['goal'](a)))/width)
    return basec, np.asarray(js).T, np.asarray(gs)


def posterior_variance(scenario_conc, scenario_goals, floor, relative=.1):
    """Mixture-integrated conditional variance with member-specific log SD."""
    from scipy.special import logsumexp
    c=np.asarray(scenario_conc); e=np.log(scenario_goals); logs=np.log(c)
    sig=np.sqrt(np.log1p(relative**2+(floor/c)**2)); nodes,weights=np.polynomial.hermite.hermgauss(31)
    values=[]
    for j in range(c.shape[1]):
        total=0.
        for h in range(len(e)):
            ys=logs[h,j]+math.sqrt(2)*sig[h,j]*nodes
            ll=-.5*((ys[:,None]-logs[:,j])/sig[:,j])**2-np.log(sig[:,j])
            prob=np.exp(ll-logsumexp(ll,axis=1)[:,None]); mean=prob@e
            var=np.sum(prob*(e[None,:]-mean[:,None])**2,axis=1)
            total+=float(weights@var)/math.sqrt(math.pi)/len(e)
        values.append(total)
    return values


def make_decision(p, case, fitref, training, visible_ids, menu, method, random_order, guard):
    start=time.monotonic(); counters={'development':0,'target':0}
    fit=checked(fitref); ts=np.array([r['time_min']/60 for r in training['observations']])
    ys=np.array([r['concentration_umol_per_l']*training['molecular_weight_g_mol']/1000 for r in training['observations']])
    candidates=np.asarray(menu['candidate_times_h']); ids=menu['candidate_ids']
    result=dict(method=method, fit=fitref, visible_ids=sorted(visible_ids), candidate_ids=ids,
        candidate_times_h=candidates.tolist(), hidden_values_used=False, target_values_used=False)
    if method.startswith('random_'):
        choice=next(i for i in random_order if i in ids); ix=ids.index(choice)
        result['permutation']=random_order
    elif method == 'time_farthest':
        scores=np.min(np.abs(np.log(candidates[:,None])-np.log(ts[None,:])),axis=1)
        ix=int(np.flatnonzero(scores>=scores.max()-1e-12)[0]); result['scores']=scores.tolist()
    else:
        if fit['status'] != 'predicted':
            return dict(result,status='failed',selected_index=None,selected_id=None,selected_time_h=None,
                        reason='No feasible visible fit for model-dependent selection',locked_at=time.time())
        models,selected=models_for(p,case,fit,training,guard,counters)
        if method == 'refit_goal':
            detail=acquisition.choose(models,selected,ts,ys,candidates,
                proportional_sd=p['proportional_noise_sd'],floor_fraction=p['additive_floor_fraction'],
                local_budget=p['local_refit_calls_per_formula'],guard=guard)
            result['detail']=detail; ix=detail['selected_index']
        elif method == 'c_optimal':
            c,j,g=finite_derivatives(models[selected],ts,candidates)
            sd=np.sqrt(np.log1p(p['proportional_noise_sd']**2+(p['additive_floor_fraction']*max(ys)/c)**2))
            # Fixed weak regularization in log coordinates; transparent numerical CL ridge.
            precision=(len(ts)+1)*acquisition.PENALTY**2 / p['proportional_noise_sd']**2 + 1e-8
            detail=classical.c_optimal_scores(j[:len(ts)]/sd[:len(ts),None],j[len(ts):]/sd[len(ts):,None],g,precision)
            detail.update(log_coordinate_difference_step=1e-4,noise_frozen_at_predictions=True,
                          structural_uncertainty_ignored=True,engineering_precision=precision.tolist())
            result['detail']=detail; ix=detail['selected_index']
        else:
            scenarioc=[]; scenariog=[]
            for m in models:
                for x in m['members']:
                    scenarioc.append(m['predict'](x,candidates)); scenariog.append(m['goal'](x))
            scores=posterior_variance(scenarioc,scenariog,p['additive_floor_fraction']*max(ys),p['proportional_noise_sd'])
            ix=int(np.argmin(scores)); result['scores']=scores
    result.update(status='selected' if ix is not None else 'failed', selected_index=ix,
        selected_id=None if ix is None else ids[ix], selected_time_h=None if ix is None else float(candidates[ix]), locked_at=time.time(),
        model_calls=counters,wall_seconds=time.monotonic()-start)
    return result


def verify_decision(p, receipt, fitref, visible_ids, menu, method):
    if (receipt['protocol'] != ref(RUN/'protocol.json') or receipt['method'] != method
            or receipt['fit'] != fitref or receipt['visible_ids'] != sorted(visible_ids)
            or receipt['candidate_ids'] != menu['candidate_ids']
            or receipt['candidate_times_h'] != menu['candidate_times_h']
            or receipt['hidden_values_used'] is not False or receipt['target_values_used'] is not False):
        raise ValueError('Changed decision or visible-data identity')
    if receipt.get('status') not in ('selected','failed'):
        raise ValueError('Unknown decision status')
    if receipt['status']=='selected':
        ix=receipt['selected_index']
        if isinstance(ix,bool) or not isinstance(ix,int) or not 0<=ix<len(menu['candidate_ids']):
            raise ValueError('Invalid locked action')
        if receipt['selected_id']!=menu['candidate_ids'][ix] or receipt['selected_time_h']!=menu['candidate_times_h'][ix]:
            raise ValueError('Action ID/time mismatch')


def run_case(p, case):
    """One worker owns a drug, avoiding duplicate concurrent subset fits."""
    guard=Guard(p); guard(force=True); cid=case['case_id']; req=request(case,'development','PKSimStandard')
    output=RUN/'cases'/f'{cid}.json'
    if output.exists():
        done=read(output)
        if done['protocol']!=ref(RUN/'protocol.json'):
            raise ValueError('Changed completed case protocol')
        verify(done['output_pins']); return ref(output)
    journal_path=RUN/'cases'/f'{cid}.progress.json'
    journal=read(journal_path) if journal_path.exists() else dict(protocol=ref(RUN/'protocol.json'),decisions={})
    if journal['protocol']!=ref(RUN/'protocol.json'):
        raise ValueError('Changed decision journal protocol')
    for source in journal['decisions'].values():
        checked(source)
    sequences=[]
    for key,spec in p['layouts'].items():
        if spec['case_id']!=cid:
            continue
        initial=checked(spec['source']); original_menu=initial['design_only']
        baseline=cached_fit(p,case,initial['visible_training'],original_menu['visible_ids'],guard)
        for method in METHODS:
            training=deepcopy(initial['visible_training']); visible_ids=list(original_menu['visible_ids'])
            fitref=baseline; stages=[dict(budget=0,result=fitref)]; actions=[]
            rs=int(method[-1]) if method.startswith('random_') else 0
            seed=20261008+100*spec['layout_index']+rs
            random_order=np.random.default_rng(seed).permutation(original_menu['candidate_ids']).tolist()
            for budget in (1,2,3):
                pairs=[(i,t) for i,t in zip(original_menu['candidate_ids'],original_menu['candidate_times_h']) if i not in visible_ids]
                menu=dict(candidate_ids=[i for i,t in pairs],candidate_times_h=[t for i,t in pairs])
                decision_path=RUN/'decisions'/key/method/f'{budget}.json'
                decision_key=str(decision_path.relative_to(RUN))
                if decision_key in journal['decisions']:
                    receipt=checked(journal['decisions'][decision_key])
                else:
                    receipt=make_decision(p,case,fitref,training,visible_ids,menu,method,random_order,guard)
                    receipt['protocol']=ref(RUN/'protocol.json'); save(decision_path,receipt)
                    journal['decisions'][decision_key]=ref(decision_path); save(journal_path,journal)
                verify_decision(p,receipt,fitref,visible_ids,menu,method)
                actions.append(ref(decision_path))
                if receipt['status']!='selected':
                    stages.extend(dict(budget=b,result=None,status='selection_failed') for b in range(budget,4)); break
                # Only the steward can open the full development vault, after this action exists.
                full=base._training_document(case['sampling_sources']['full'],req,ROOT)
                training=sequential.reveal(full,training,receipt['selected_id'],req)
                visible_ids.append(receipt['selected_id'])
                fitref=cached_fit(p,case,training,visible_ids,guard)
                stages.append(dict(budget=budget,result=fitref))
                print(cid, key.rsplit('__',1)[-1],method,budget,flush=True)
            sequences.append(dict(layout=key,method=method,stages=stages,actions=actions))
    # Full information reference is fitted last and is never used by any selector.
    full=base._training_document(case['sampling_sources']['full'],req,ROOT)
    clean=sequential._validated(full,req,'Full development reference')
    positive=[(i,r) for i,r in enumerate(clean) if r['time_min']>0]
    fullref=cached_fit(p,case,sequential._safe(full,[r for i,r in positive]),[i for i,r in positive],guard)
    done=dict(protocol=ref(RUN/'protocol.json'),case_id=cid,sequences=sequences,full_reference=fullref,
              journal=ref(journal_path),locked_at=time.time())
    pins={}; base.pins_in(done,pins)
    for fitpath in (RUN/'fits'/cid).glob('*/result.json'):
        pins.update(read(fitpath)['output_pins'])
    done['output_pins']=pins; verify(pins); save(output,done)
    return ref(output)


def run(workers):
    with (RUN/'.run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(RUN/'state.json'); p=checked(state['protocol']); Guard(p)(force=True)
        if state['status']=='complete':
            return status()
        state.update(status='running',pid=os.getpid(),started_at=state.get('started_at',time.time())); save(RUN/'state.json',state)
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                todo={pool.submit(run_case,p,c):c['case_id'] for c in p['cases'] if c['case_id'] not in state['completed']}
                for f in as_completed(todo):
                    state['completed'][todo[f]]=f.result(); save(RUN/'state.json',state)
            Guard(p)(force=True)
            if len(state['completed'])!=len(p['cases']):
                raise ValueError('Missing completed drug')
            state.update(status='predictions_locked',locked_at=time.time()); save(RUN/'state.json',state)
            return report()
        except BaseException as exc:
            save(RUN/'PAUSE',dict(reason=str(exc))); state.update(status='paused',reason=str(exc)); save(RUN/'state.json',state)
            raise


def status():
    s=read(RUN/'state.json')
    fits=list((RUN/'fits').glob('*/*/result.json')); decisions=list((RUN/'decisions').glob('*/*/*.json'))
    return dict(status=s['status'],completed_drugs=len(s['completed']),total_drugs=10,
        unique_fits=len(fits),decisions=len(decisions),total_sequential_actions=420,
        elapsed_seconds=None if 'started_at' not in s else (s.get('completed_at',time.time())-s['started_at']),reason=s.get('reason'))


def cleanup():
    s=read(RUN/'state.json')
    if s['status']!='complete':
        raise ValueError('Cleanup requires complete results')
    verify(checked(s['protocol'])['source_pins'])
    for source in s['completed'].values():
        verify(checked(source)['output_pins'])
    removed=[]
    for path in (RUN/'fits').glob('*/*/.lock'):
        path.unlink(); removed.append(str(path.relative_to(RUN)))
    return dict(removed_lock_files=len(removed),retained='Compact fit scores, final curves, decisions, source hashes and failures; no per-trial directories created.')


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


def report():
    from report_blueprint_validation import Store, _case_reference, _score_prediction, clustered_geomean, paired_ratio
    state=read(RUN/'state.json'); p=checked(state['protocol']); verify(p['source_pins'])
    if state['status'] not in ('predictions_locked','complete') or set(state['completed'])!={c['case_id'] for c in p['cases']}:
        raise ValueError('All drug predictions must lock before target scoring')
    done={cid:checked(source) for cid,source in state['completed'].items()}
    for cid,d in done.items():
        if d['protocol']!=state['protocol'] or d['case_id']!=cid or d['locked_at']>state['locked_at']:
            raise ValueError('Case predictions do not match the global lock')
        verify(d['output_pins'])
        expected={(key,method) for key,spec in p['layouts'].items() if spec['case_id']==cid for method in METHODS}
        actual=[(s['layout'],s['method']) for s in d['sequences']]
        if set(actual)!=expected or len(actual)!=len(expected):
            raise ValueError('Incomplete or duplicated sequence manifest')
        for seq in d['sequences']:
            if [s['budget'] for s in seq['stages']]!=[0,1,2,3]:
                raise ValueError('Incomplete budget manifest')
            for stage in seq['stages']:
                if stage['result'] is None and stage.get('status')!='selection_failed':
                    raise ValueError('Missing prediction without recorded failure')
    store=Store(ROOT); score_by_fit={}; records=[]; csv_rows=[]
    for case in p['cases']:
        cid=case['case_id']; clinical,curve,endpoints=_case_reference(case,store)
        entries=[]
        for seq in done[cid]['sequences']:
            entries.extend(dict(layout=seq['layout'],method=seq['method'],**st) for st in seq['stages'])
        entries.append(dict(layout='full',method='full_reference',budget=None,result=done[cid]['full_reference']))
        for entry in entries:
            reference=entry['result']; unit=None if reference is None else checked(reference)
            key=None if reference is None else reference['path']
            if key not in score_by_fit:
                if unit is not None and unit['status']=='predicted':
                    score,_,_=_score_prediction(case,clinical,curve,endpoints,unit,store)
                    score.pop('curve',None)
                else:
                    score=dict(curve_metrics={'gmfe':None},endpoints={ep:dict(value=None,reference=v['value'],unit=v['unit'],error=None,weak_tail=None) for ep,v in endpoints.items()})
                if key is not None:
                    score_by_fit[key]=score
            else:
                score=score_by_fit[key]
            common=dict(case_id=cid,drug=case['drug'],layout=entry['layout'],method=entry['method'],budget=entry['budget'],
                status=entry.get('status',unit['status'] if unit else 'failed'))
            record=dict(common,score=score); records.append(record)
            csv_rows.append(dict(common,endpoint='curve',prediction=None,reference=None,error=score['curve_metrics'].get('gmfe'),unit=curve['unit'],weak_tail=None))
            csv_rows.extend(dict(common,endpoint=ep,prediction=v.get('value'),reference=v.get('reference'),error=v.get('error'),unit=v.get('unit'),weak_tail=v.get('weak_tail')) for ep,v in score['endpoints'].items())
    clusters={c['case_id']:c['cluster_id'] for c in p['cases']}
    endpoints_list=['curve']+list(next(iter(score_by_fit.values()))['endpoints'])
    def get_error(record,endpoint):
        score=record['score']
        return score['curve_metrics'].get('gmfe') if endpoint=='curve' else score['endpoints'][endpoint].get('error')
    def valid(x):
        return x is not None and math.isfinite(x) and x>0
    def drug_values(method,budgets,endpoint):
        values={}
        for cid in clusters:
            candidates=[r for r in records if r['case_id']==cid and r['budget'] in budgets and
                        (r['method'].startswith('random_') if method=='random' else r['method']==method)]
            vals=[get_error(r,endpoint) for r in candidates]
            expected=(6 if method=='random' else 2)*len(budgets) if method!='full_reference' else 1
            values[cid]=math.exp(float(np.mean(np.log(vals)))) if len(vals)==expected and all(valid(v) for v in vals) else None
        return values
    summary={}; values={}
    for b in (0,1,2,3,'mean_acquired','full'):
        summary[str(b)]={}; values[b]={}
        methods=('full_reference',) if b=='full' else REPORT_METHODS
        budgets=[None] if b=='full' else ([1,2,3] if b=='mean_acquired' else [b])
        for method in methods:
            summary[str(b)][method]={}; values[b][method]={}
            for endpoint in endpoints_list:
                vals=drug_values(method,budgets,endpoint); values[b][method][endpoint]=vals
                usable={k:v for k,v in vals.items() if valid(v)}
                # Tmax uses an absolute error and can be zero, so do not report GMFE for it.
                if endpoint=='tmax':
                    raw=[get_error(r,endpoint) for r in records if r['budget'] in budgets and (r['method'].startswith('random_') if method=='random' else r['method']==method)]
                    raw=[v for v in raw if v is not None and math.isfinite(v)]
                    summary[str(b)][method][endpoint]=dict(metric='absolute_error_h',mean=float(np.mean(raw)) if raw else None,n_scenarios=len(raw))
                else:
                    summary[str(b)][method][endpoint]=dict(clustered_geomean(usable,{k:clusters[k] for k in usable}),missing_drugs=10-len(usable),
                        within_twofold_drug_averages=sum(v<=2 for v in usable.values()))
    comparisons={}
    for b in (1,2,3,'mean_acquired'):
        comparisons[str(b)]={}
        for endpoint in ('cmax','curve','auc_inf','terminal_half_life'):
            comparisons[str(b)][endpoint]={m:paired_ratio(values[b]['refit_goal'][endpoint],values[b][m][endpoint],clusters) for m in REPORT_METHODS if m!='refit_goal'}
            comparisons[str(b)][endpoint]['no_addition']=paired_ratio(values[b]['refit_goal'][endpoint],values[0]['refit_goal'][endpoint],clusters)
    efficiency=[]
    for case in p['cases']:
        cid=case['case_id']; full=values['full']['full_reference']['cmax'][cid]
        for seq in done[cid]['sequences']:
            rows=[r for r in records if r['case_id']==cid and r['layout']==seq['layout'] and r['method']==seq['method']]
            reached=[r['budget'] for r in rows if valid(get_error(r,'cmax')) and valid(full) and get_error(r,'cmax')<=2 and get_error(r,'cmax')<=1.1*full]
            efficiency.append(dict(case_id=cid,layout=seq['layout'],method=seq['method'],first_budget=min(reached) if reached else None,
                full_information_error=full,interpretation='Retrospective criterion; not a stopping rule available before clinical scoring'))
    primary=comparisons['mean_acquired']['cmax']
    gates=dict(complete=all(r['status']=='predicted' for r in records),
        one_percent_better_controls=all(primary[m]['complete_pairs']==10 and primary[m]['estimate']<=.99 for m in ('time_farthest','c_optimal','random')),
        six_directional_wins=primary['time_farthest']['ratio_below_one']>=6,
        curve_no_harm=comparisons['mean_acquired']['curve']['time_farthest']['complete_pairs']==10 and comparisons['mean_acquired']['curve']['time_farthest']['estimate']<=1.01,
        secondary_no_harm=all(comparisons['mean_acquired'][ep]['no_addition']['complete_pairs']==10 and comparisons['mean_acquired'][ep]['no_addition']['estimate']<=1.05 for ep in ('auc_inf','terminal_half_life')))
    fitdocs=[read(f) for f in (RUN/'fits').glob('*/*/result.json')]
    decisiondocs=[read(f) for f in (RUN/'decisions').glob('*/*/*.json')]
    resources=dict(unique_fits=len(fitdocs),actual_formula_fits=sum(len(f['fits']) for f in fitdocs),
        objective_calls=sum(f['resources']['objective_calls'] for f in fitdocs),
        lookahead_calls=sum(d.get('detail',{}).get('resources',{}).get('refit_calls',0) for d in decisiondocs),
        hypothetical_budget_exhaustions=sum(d.get('detail',{}).get('resources',{}).get('budget_exhaustions',0) for d in decisiondocs),
        final_target_predictions=sum(f['resources']['prediction_calls'] for f in fitdocs),
        decision_costs={m:dict(development_calls=sum(d.get('model_calls',{}).get('development',0) for d in decisiondocs if d['method']==m),
                              target_calls=sum(d.get('model_calls',{}).get('target',0) for d in decisiondocs if d['method']==m),
                              wall_seconds=sum(d.get('wall_seconds',0) for d in decisiondocs if d['method']==m)) for m in METHODS},
        elapsed_seconds=time.time()-state['started_at'])
    result=dict(version=VERSION,prediction_lock=state['locked_at'],scored_at=time.time(),summary=summary,comparisons=comparisons,
        efficiency=efficiency,expansion_checks=gates,expand_to_twenty=all(gates.values()),resources=resources,limitations=p['limitations'])
    save(RUN/'results.json',result)
    with (RUN/'endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines=['# Sequential endpoint-directed PBPK sampling','','Ten development drugs; two mixed four-point layouts per drug. Each step locks a sampling time before revealing its concentration. Lower GMFE is better.','',
        '| Added points | Endpoint | Refit goal | Posterior variance | Official c-opt | Time coverage | Random (3 seeds) |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for b in (0,1,2,3,'mean_acquired'):
        for ep in ('cmax','curve','auc_inf','terminal_half_life'):
            vals=[summary[str(b)][m][ep]['estimate'] for m in REPORT_METHODS]
            lines.append('| '+str(b)+' | '+ep+' | '+' | '.join('NA' if v is None else f'{v:.4f}' for v in vals)+' |')
    lines+=['',f'Expansion gate passed: **{result["expand_to_twenty"]}**. Checks: `{gates}`.',
        '', 'Primary endpoint: Cmax, averaging log error over the two layouts and budgets 1–3 within each drug. Other endpoints are secondary; the selector did not optimize them. Ten drugs remain ten, not 60 independent cases.',
        '', 'The complete table is `endpoints.csv`; all 11 endpoint categories, predictions, references, errors, unavailable references and failures are retained. Tmax is an absolute error in hours. Random uses three fixed permutations, not exhaustive expectation.',
        '', 'The full-data fit is an information reference, not guaranteed optimal. Data-efficiency thresholds are evaluated after scoring and cannot be used as prospective stopping criteria.',
        '', 'The finite generating scenarios, measurement-noise floor and bounded local refits are engineering approximations. This is a retrospective development pilot, not a clinical-risk or zero-curve validation.',
        '', '```bash','runtime/app-env/bin/python -B scripts/run_sequential_sampling_design.py status','```']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')
    store.unchanged(); state.update(status='complete',completed_at=time.time(),results=ref(RUN/'results.json')); save(RUN/'state.json',state)
    return dict(status='complete',expansion_checks=gates,resources=resources)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run','status','report','cleanup'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=14:
        parser.error('workers must be 1..14')
    print({'prepare':prepare,'run':lambda:run(args.workers),'status':status,'report':report,'cleanup':cleanup}[args.command](),flush=True)
