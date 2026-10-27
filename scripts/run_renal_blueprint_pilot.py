#!/usr/bin/env python3
"""Small evidence-update pilot; fit development curves, lock, then score.

Existing ten-drug results and code are immutable. Target renal references are
only decoded by --score. This is exploratory development, not a blinded trial.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import math
from pathlib import Path
import shutil
import time

from run_complete_reference_pilot import ROOT, read, now, ref, verify
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

BASE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug'
PILOT = ROOT / 'experiments/blueprint_design/renal_pilot_20260929'
RUN = PILOT / 'run'
FINAL = ROOT / 'results/data/renal_blueprint_pilot.json'
FORMULAS = ('PKSimStandard', 'RodgersRowland')


def embedded_pins(value):
    """Pin locally available primary sources without decoding target outcomes."""
    pins = {}
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = inside(value['path'], ROOT, exists=True)
            pins[str(path.relative_to(ROOT))] = value['sha256']
        for v in value.values(): pins.update(embedded_pins(v))
    elif isinstance(value, list):
        for v in value: pins.update(embedded_pins(v))
    return pins


def prepare():
    if RUN.exists():
        raise FileExistsError('Existing pilot is immutable; inspect its summary')
    source = read(PILOT / 'inputs.json')
    old = read(BASE / 'optimized_run/summary.json')
    verify(old['protocol']['pins']); verify(old['locks']['inputs'])
    cases = {c['case_id']: c for c in old['protocol']['cases']}
    RUN.mkdir()
    protocol = {'version': 'evidence-renal-pilot-1', 'prepared_at': now(),
        'max_calls_per_formula': 24, 'workers': 4, 'cases': [],
        'selection': 'Development log-curve objective only; no target outcome enters fitting or formula choice',
        'interpretation': 'Previously viewed drugs; evidence-based exploratory extension, not blind validation',
        'limitations': ['Two construction parameters and fixed physiology; no enzyme-specific or disease extrapolation',
            'Nonrenal clearance is lumped in the liver module, not measured liver metabolism',
            'Renal clearance includes all net processes; filtration must not be added separately',
            'Urine is unchanged parent drug; metabolites and total radioactivity are not substitutes'],
        'source_pins': {str(p.relative_to(ROOT)): sha256(p) for p in
            [PILOT/'inputs.json', PILOT/'references.json', Path(__file__),
             BASE/'optimized_run/comparison_input.json',
             ROOT/'results/tables/pbpk_ours_vs_rf/additional_comparison.json',
             ROOT/'scripts/summarize_additional_clinical_baselines.py',
             ROOT/'scripts/run_classical_pk_comparator.py']}}
    # Preserve original dependencies without including target outcome values.
    protocol['source_pins'].update(old['protocol']['pins'])
    protocol['source_pins'].update(embedded_pins(source))
    verify(protocol['source_pins'])
    module = ROOT / 'src/pbpk_kb/blueprints/renal_extension.py'
    if module.exists(): protocol['source_pins'][str(module.relative_to(ROOT))] = sha256(module)
    for update in source['cases']:
        cid = update['case_id']; case = deepcopy(cases[cid])
        folder = RUN / 'inputs' / cid; folder.mkdir(parents=True)
        req = read(BASE/'run/inputs'/cid/'development_request.json')
        updates = update['updates']; weight = update.get('normalization_weight_kg')
        pairs = {'total_clearance_ml_min': ('total_clearance', 'ml/min/kg'),
                 'renal_clearance_ml_min': ('renal_clearance', 'ml/min/kg'),
                 'fraction_unbound': ('fraction_unbound', 'fraction')}
        evidence = {e['parameter']: deepcopy(e) for e in req['evidence']}
        for name, value in updates.items():
            if name not in pairs or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid independent input: '+name)
            parameter, unit = pairs[name]
            if parameter.endswith('clearance'):
                if not isinstance(weight, (int, float)) or weight <= 0: raise ValueError('Explicit reference body weight required')
                value /= weight
            item = update['evidence'][name]
            evidence[parameter] = {'parameter': parameter, 'compound_id': req['compound_id'],
                'value': value, 'unit': unit, 'basis': 'derived',
                'source': ref(PILOT/'inputs.json', cid+'/'+name+': '+str(item.get('location', 'curated independent source'))),
                'context': {'species': 'human', 'matrix': 'plasma', 'study_role': 'development',
                            **({'route': 'intravenous'} if parameter.endswith('clearance') else {})},
                'notes': str(item)+'; absolute CL divided by declared source/reference body weight when applicable'}
        renal = 'renal_clearance_ml_min' in updates
        if renal:
            if not 0 < updates['renal_clearance_ml_min'] < updates['total_clearance_ml_min']:
                raise ValueError('Independent renal and total clearance must be compatible')
            evidence['elimination_scope'] = {'parameter': 'elimination_scope', 'compound_id': req['compound_id'],
                'value': 'hepatic_and_renal', 'unit': 'text', 'basis': 'derived',
                'source': ref(PILOT/'inputs.json', cid+': total minus renal clearance, same source arm'),
                'notes': 'Nonrenal remainder is localized to liver as a lumped approximation; not enzyme or liver-specific evidence', 'context': {}}
        ledger = folder/'evidence.json'; save(ledger, {'evidence': list(evidence.values())})
        req['evidence'] = list(evidence.values()); req['evidence_ledger'] = ref(ledger, 'Updated independent construction evidence')
        req['forbidden_sources'].append(ref(PILOT/'references.json', 'Held-out renal outcomes; post-lock scoring only'))
        req['assumptions'] = [x for x in req['assumptions'] if not x.startswith('Total clearance is lumped')]
        req['assumptions'] += update.get('limitations', []) + protocol['limitations']
        req['assumptions'] = list(dict.fromkeys(req['assumptions']))
        save(folder/'development.json', req)
        target = deepcopy(req)
        original = read(BASE/'run/inputs'/cid/'target_request.json')
        for name in ('human', 'dosing', 'target'): target[name] = original[name]
        save(folder/'target.json', target)
        case.update(renal_enabled=renal, updated_inputs=updates,
                    source_reference_weight_kg=weight,
                    renal_fraction_prior=(updates['renal_clearance_ml_min']/updates['total_clearance_ml_min'] if renal else None))
        protocol['cases'].append(case)
        classical = read(BASE/'additional_baselines/two_compartment/summary.json')['results'][cid]
        for name in ('fit', 'prediction'):
            protocol['source_pins'][classical[name]['path']] = classical[name]['sha256']
        for p in (ledger, folder/'development.json', folder/'target.json', ROOT/case['development_source']['path']):
            protocol['source_pins'][str(p.relative_to(ROOT))] = sha256(p)
    save(RUN/'protocol.json', protocol)
    save(RUN/'summary.json', {'status': 'prepared', 'fits': {}, 'results': {}, 'locks': {}})
    return protocol


def execute():
    from pbpk_kb.blueprints.curve_pattern_blueprint import fit_pattern_mapping
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    from pbpk_kb.blueprints.pk_anchored_blueprint import mapped_snapshot, _run, _drug_signature
    from pbpk_kb.blueprints.renal_extension import add_renal_outputs, read_renal_outputs
    protocol = prepare() if not RUN.exists() else read(RUN/'protocol.json')
    summary = read(RUN/'summary.json')
    if summary['status'] != 'prepared': raise ValueError('Only an unstarted pilot may execute')
    verify(protocol['source_pins']); start = time.monotonic()

    def progress(status):
        summary.update(status=status, updated_at=now(), elapsed_seconds=round(time.monotonic()-start, 1))
        save(RUN/'summary.json', summary)

    def fit(case, formula):
        cid = case['case_id']; req = read(RUN/'inputs'/cid/'development.json'); req['partition_method'] = formula
        folder = RUN/'fits'/cid/formula
        try:
            result = fit_pattern_mapping(reference_request=req, training_source=case['development_source'],
                output_dir=folder, project_root=ROOT, max_evaluations=protocol['max_calls_per_formula'])
            return {'status': result['status'], 'eligible': result['eligible'], 'decision': result['decision'],
                    'converged': result['converged'], 'calls': result['evaluation_count'],
                    'failures': result['failed_evaluation_count'], 'mapping': ref(folder/'mapping_locked.json', 'Development-only mapping')}
        except Exception as exc:
            return {'status': 'failed', 'eligible': False, 'reason': type(exc).__name__+': '+str(exc)}

    try:
        progress('fitting_development')
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            jobs = {pool.submit(fit, c, f): (c['case_id'], f) for c in protocol['cases'] for f in FORMULAS}
            for job in as_completed(jobs):
                cid, formula = jobs[job]; result = job.result()
                summary['fits'].setdefault(cid, {})[formula] = result
                print(cid, formula, result['status'], result.get('decision', {}).get('objective') if result.get('decision') else '', flush=True)
                progress('fitting_development')
        verify(protocol['source_pins'])
        summary['selection'] = {}
        for c in protocol['cases']:
            eligible = [(f, r) for f, r in summary['fits'][c['case_id']].items() if r['eligible']]
            chosen = min(eligible, key=lambda x: (x[1]['decision']['objective'], x[0] != 'PKSimStandard')) if eligible else None
            summary['selection'][c['case_id']] = None if chosen is None else {'formula': chosen[0], 'mapping': chosen[1]['mapping'], 'locked_at': now()}
        progress('predicting')
        for case in protocol['cases']:
            cid = case['case_id']; chosen = summary['selection'][cid]
            if chosen is None:
                summary['results'][cid] = {'status': 'failed', 'reason': 'No feasible development mapping'}; continue
            folder = RUN/'predictions'/cid; folder.mkdir(parents=True)
            req = read(RUN/'inputs'/cid/'target.json'); req['partition_method'] = chosen['formula']
            verify({chosen['mapping']['path']: chosen['mapping']['sha256']})
            mapping = read(ROOT/chosen['mapping']['path'])
            if mapping['drug_signature'] != _drug_signature(req): raise ValueError('Drug signature changed between development and target')
            blueprint = assemble_evidence_blueprint(req, ROOT)
            if blueprint['status'] != 'assembled': raise ValueError('Incomplete target blueprint')
            native = mapped_snapshot(blueprint, mapping['decision']['lipophilicity'], mapping['decision']['clearance_scale'])
            if case['renal_enabled']: native = add_renal_outputs(native)
            result = _run(native, req, folder, ROOT, audit=True)
            if result['status'] == 'predicted':
                summary['locks'][cid] = {'plasma': ref(folder/'prediction_locked.json', 'Target plasma prediction before scoring')}
                if case['renal_enabled']:
                    renal = read_renal_outputs(folder/'native/execution_manifest.json', native, ROOT)
                    save(folder/'renal_prediction.json', renal)
                    summary['locks'][cid]['renal'] = ref(folder/'renal_prediction.json', 'Native parent-drug renal prediction before scoring')
            summary['results'][cid] = result; progress('predicting')
        verify(protocol['source_pins'])
        summary.update(finished_at=now(), all_attempts_locked_before_scoring=True, target_values_used_for_selection=False)
        progress('predictions_locked')
    except BaseException as exc:
        summary['failure'] = type(exc).__name__+': '+str(exc); progress('failed'); raise


def score():
    import numpy as np
    from scipy.integrate import quad
    from run_classical_pk_comparator import concentration_mg_l, dosing_kwargs, PARAMETERS
    from run_clinical_method_comparison import predicted_nca, clinical_unit_factor
    from render_pbpk_method_comparison import checked_curve, curve_metrics
    from summarize_additional_clinical_baselines import prediction_in_hours, weak_endpoints
    from pbpk_kb.blueprints.renal_extension import window_renal_endpoints
    protocol, summary = read(RUN/'protocol.json'), read(RUN/'summary.json')
    if summary['status'] != 'predictions_locked': raise ValueError('Score only after every prediction attempt is locked')
    verify(protocol['source_pins'])
    for fits in summary['fits'].values():
        for fit in fits.values():
            if fit.get('mapping'): verify({fit['mapping']['path']: fit['mapping']['sha256']})
    for locks in summary['locks'].values(): verify({r['path']: r['sha256'] for r in locks.values()})
    # First decoding of held-out outcome documents occurs here, after locks.
    renal_refs = {r['case_id']: r for r in read(PILOT/'references.json')['cases']}
    clinical = {c['case_id']: c for c in read(BASE/'optimized_run/comparison_input.json')['cases']}
    baseline = {c['case_id']: c for c in read(ROOT/'results/tables/pbpk_ours_vs_rf/additional_comparison.json')['cases']}
    rows = []
    for case in protocol['cases']:
        cid = case['case_id']; old = baseline[cid]; target = clinical[cid]; policy = case['nca_policy']
        item = {'case_id': cid, 'drug': case['drug'], 'status': summary['results'][cid]['status'],
                'updated_inputs': case['updated_inputs'], 'conditions': target['conditions'], 'methods': {},
                'references': case['reference_source'], 'baseline_curve_gmfe_for_audit': old['gmfe']['ours'],
                'limitations': protocol['limitations']}
        if item['status'] != 'predicted':
            item['failure'] = summary['results'][cid]; rows.append(item); continue
        native = read(ROOT/summary['locks'][cid]['plasma']['path'])
        mw = read(ROOT/case['physchem'])['identity']['molecular_weight_g_mol']
        times, values, merged = prediction_in_hours(native, mw)
        nca = predicted_nca(policy=policy, time_h=times, concentration_ng_ml=values,
            dose_ng=policy['dose_ng'], infusion_h=policy['infusion_h'], nominal_times_h=policy['sampling_time_h'],
            lloq_ng_ml=policy['lloq_ng_ml'], weight_kg=policy['weight_kg'])
        observed_curve = checked_curve(target['clinical_reference']['curve'], 'clinical', clinical=True)
        factor = clinical_unit_factor('curve', observed_curve['unit'], policy['weight_kg'])
        curve = {'time_h': times, 'concentration': [v*factor for v in values], 'unit': observed_curve['unit'],
                 'matrix': 'plasma', 'statistic': 'representative_adult'}
        ours = {'curve_metrics': curve_metrics(curve, observed_curve, set(target.get('approximations', []))),
                'endpoints': {}, 'nca_quality': nca['diagnostics'], 'prediction': summary['locks'][cid]['plasma']}
        two = {'curve_metrics': old['curve_metrics']['two_compartment'], 'endpoints': {},
               'prediction': old['prediction_sources']['two_compartment']}
        weak = weak_endpoints(nca['diagnostics'])
        for ep, e in old['endpoints'].items():
            p = nca['endpoints'][ep]['value']; o = e['observed']
            if p is not None: p *= clinical_unit_factor(ep, e['source_unit'], policy['weight_kg'])*e['display_factor']
            error = None if p is None or o is None or (ep != 'tmax' and min(p, o) <= 0) else (abs(p-o) if ep == 'tmax' else max(p/o, o/p))
            ours['endpoints'][ep] = {'predicted': p, 'observed': o, 'unit': e['unit'], 'error': error,
                'error_metric': 'absolute_error_h' if ep == 'tmax' else 'fold_error', 'weak_tail': ep in weak}
            two['endpoints'][ep] = {**e['predictions']['two_compartment'], 'observed': o, 'unit': e['unit']}
        if case['renal_enabled']:
            renal = read(ROOT/summary['locks'][cid]['renal']['path'])
            ours['renal_checks'] = renal['checks']; ours['renal'] = {}; two['renal'] = {}
            fit = read(BASE/'additional_baselines/two_compartment'/cid/'fit.json')
            pars = [fit['target_parameters'][k] for k in PARAMETERS]
            kwargs = dosing_kwargs(fit['target_dosing']); fraction = case['renal_fraction_prior']
            for label, endpoint in renal_refs[cid]['endpoints'].items():
                interval = endpoint.get('window_h', renal_refs[cid]['conditions']['urine_collection_window_h'])
                if len(interval) != 2 or interval[0] != 0 or interval[1] <= 0:
                    raise ValueError('This pilot supports only urine collections from dosing start')
                window = interval[1]; observed = endpoint['value']
                key = {'renal_clearance': 'renal_clearance_l_h', 'urinary_fraction': 'fraction_percent'}[label]
                unit_factor = {'L/h': 1., 'mL/min': 1000/60, 'percent_of_administered_dose': 1.}[endpoint['unit']]
                pred = window_renal_endpoints(renal, native, window, mw)
                auc = quad(lambda t: float(concentration_mg_l(pars, [t], **kwargs)[0]), 0, window,
                           points=[kwargs['infusion_h']], epsabs=1e-9, epsrel=1e-9)[0]
                ae = fraction*pars[0]*auc
                if not 0 <= ae <= kwargs['dose_mg']*(1+1e-6): raise ValueError('Classical urine amount violates dose balance')
                simple = {'amount_mg': ae, 'fraction_percent': ae/kwargs['dose_mg']*100,
                          'renal_clearance_l_h': fraction*pars[0]}
                for method, value in ((ours, pred[key]*unit_factor), (two, simple[key]*unit_factor)):
                    method['renal'][label] = {'prediction': value, 'observed': observed,
                        'unit': endpoint['unit'], 'window_h': window,
                        'fold_error': max(value/observed, observed/value),
                        'source': {**renal_refs[cid]['source'], 'location': endpoint['location']},
                        'observation_statistic': endpoint['statistic']}
            two['renal_policy'] = ('Same independent CLr/CL fraction attached to classical total elimination; not a PBPK-exclusive capability. '
                                   'PBPK additionally receives total CL/fu priors; this remains an unequal-information comparison, not an isolated architecture ablation.')
        item['methods'] = {'ours': ours, 'two_compartment': two}; rows.append(item)
    verify(protocol['source_pins']); verify(embedded_pins(list(renal_refs.values())))
    document = {'version': protocol['version'], 'scored_at': now(), 'cases': rows,
        'interpretation': protocol['interpretation'], 'clinical_qualification': False,
        'target_values_used_for_fitting_or_selection': False,
        'source_pins': {**protocol['source_pins'], str((RUN/'summary.json').relative_to(ROOT)): sha256(RUN/'summary.json')},
        'summary': {}}
    for method in ('ours', 'two_compartment'):
        scores = [r['methods'][method]['curve_metrics']['gmfe'] for r in rows if method in r['methods']]
        document['summary'][method] = {'drug_equal_curve_gmfe': math.exp(sum(map(math.log, scores))/len(scores)) if scores else None,
                                     'completed': len(scores), 'within_twofold': sum(s <= 2 for s in scores)}
    FINAL.parent.mkdir(parents=True, exist_ok=True); save(FINAL, document)
    print(document['summary'])


def cleanup():
    protocol, summary = read(RUN/'protocol.json'), read(RUN/'summary.json')
    if not FINAL.exists() or summary['status'] != 'predictions_locked': raise ValueError('Cleanup follows successful scoring')
    verify(protocol['source_pins']); removed = 0; nbytes = 0
    for case in protocol['cases']:
        for formula, fit in summary['fits'][case['case_id']].items():
            if not fit.get('eligible'): continue
            mapping = read(ROOT/fit['mapping']['path'])
            successful = {r['evaluation'] for r in mapping['evaluations'] if r.get('status') == 'ok'}
            for path in (RUN/'fits'/case['case_id']/formula).glob('attempt_[0-9][0-9]'):
                number = int(path.name.split('_')[-1])
                if number == mapping['decision']['evaluation'] or number not in successful: continue
                if path.is_symlink(): raise ValueError('Refuse to clean symlink')
                inside(path, RUN, exists=True)
                files = [p for p in path.rglob('*') if p.is_file()]
                removed += len(files); nbytes += sum(p.stat().st_size for p in files)
                shutil.rmtree(path)
    # Store cleanup counts in the existing final record, no extra report file.
    document = read(FINAL); document['cleanup'] = {'removed_files': removed, 'removed_bytes': nbytes,
        'scope': 'Only this pilot: unselected successful native development trials. Inputs, all trial scores, chosen runs, failures and final predictions retained'}
    save(FINAL, document); print(document['cleanup'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ('run', 'score', 'cleanup', 'status'): mode.add_argument('--'+name, action='store_true')
    args = p.parse_args()
    if args.run: execute()
    elif args.score: score()
    elif args.cleanup: cleanup()
    else:
        summary = read(RUN/'summary.json'); print({k: summary.get(k) for k in ('status', 'elapsed_seconds', 'selection', 'results', 'failure')})
        for path in sorted((RUN/'fits').glob('*/*/progress.json')): print(path.parent.parent.name, path.parent.name, read(path))


if __name__ == '__main__': main()
