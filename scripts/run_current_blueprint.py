#!/usr/bin/env python3
"""Apply the evidence-priority blueprint consistently to the ten-drug cohort.

No clinical target outcomes are decoded by prepare/execute. Exact existing
no-warm-start fits may be reused; every final target prediction is executed
again. Reporting is a separate post-lock step. All paths are project-local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from run_complete_reference_pilot import ROOT, now, read, ref, verify
from run_renal_blueprint_pilot import embedded_pins, FORMULAS
from pbpk_kb.blueprints.source_workflow import save, sha256

BASE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug'
PILOT = ROOT / 'experiments/blueprint_design/renal_pilot_20260929'
CURRENT = ROOT / 'experiments/blueprint_design/evidence_priority_20260929'
RUN = CURRENT / 'run'
VERSION = 'evidence-priority-ten-drug-1'


def prepare(workers):
    from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature
    if RUN.exists():
        raise FileExistsError('Existing run is immutable; use --run to resume completed fits')
    source = read(CURRENT / 'inputs.json')
    old = read(BASE / 'optimized_run/summary.json')
    cached = read(PILOT / 'run/summary.json')
    cached_inputs = {c['case_id']: c for c in read(PILOT / 'inputs.json')['cases']}
    cached_protocol = read(PILOT / 'run/protocol.json')
    verify(old['protocol']['pins']); verify(old['locks']['inputs'])
    verify(cached_protocol['source_pins'])
    updates = {c['case_id']: c for c in source['cases']}
    if set(updates) != {c['case_id'] for c in old['protocol']['cases']}:
        raise ValueError('Complete unchanged ten-drug cohort required')
    RUN.mkdir()
    protocol = {
        'version': VERSION, 'prepared_at': now(), 'workers': workers,
        'max_calls_per_formula': 24, 'warm_start_source': None, 'cases': [],
        'selection': 'Development log-curve objective only; no target outcomes enter fitting or formula selection',
        'interpretation': 'Previously viewed drugs; exploratory development, not blinded validation',
        'cache_policy': 'Exact same inputs, development data, core code, 24-call budget and no warm starts; final targets always rerun',
        'limitations': cached_protocol['limitations'] + [
            'One common clearance scale still changes renal and nonrenal processes together',
            'Independent source data are prioritized only when first-hand and compatible; missing inputs retain RF priors',
            'No organ tissue concentration/AUC prediction or validation is added'],
        'pins': deepcopy(old['protocol']['pins']),
    }
    protocol['pins'].update(embedded_pins(source))
    for p in [CURRENT/'inputs.json', Path(__file__), PILOT/'run/protocol.json',
              PILOT/'inputs.json', PILOT/'references.json',
              ROOT/'src/pbpk_kb/blueprints/renal_extension.py']:
        protocol['pins'][str(p.relative_to(ROOT))] = sha256(p)
    summary = {'status': 'prepared', 'protocol': protocol, 'fits': {}, 'results': {},
               'selection': {}, 'locks': {'inputs': {}, 'mappings': {}, 'predictions': {}}}
    for original in old['protocol']['cases']:
        case = deepcopy(original); cid = case['case_id']; update = updates[cid]
        folder = RUN/'inputs'/cid; folder.mkdir(parents=True)
        cache = cid in cached_inputs
        if cache:
            if update != cached_inputs[cid]:
                raise ValueError('Cached independent inputs changed: '+cid)
            development = read(PILOT/'run/inputs'/cid/'development.json')
            target = read(PILOT/'run/inputs'/cid/'target.json')
        else:
            development = read(BASE/'run/inputs'/cid/'development_request.json')
            target = read(BASE/'run/inputs'/cid/'target_request.json')
            if update['updates']:
                evidence = {e['parameter']: deepcopy(e) for e in development['evidence']}
                pairs = {'total_clearance_ml_min': ('total_clearance', 'ml/min/kg'),
                         'renal_clearance_ml_min': ('renal_clearance', 'ml/min/kg'),
                         'fraction_unbound': ('fraction_unbound', 'fraction')}
                weight = update.get('normalization_weight_kg')
                for name, value in update['updates'].items():
                    if name not in pairs or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError('Invalid independent parameter: '+name)
                    parameter, unit = pairs[name]
                    if parameter.endswith('clearance'):
                        if not isinstance(weight, (int, float)) or weight <= 0:
                            raise ValueError('Explicit normalization weight required')
                        value /= weight
                    item = update['evidence'][name]
                    evidence[parameter] = {'parameter': parameter, 'compound_id': development['compound_id'],
                        'value': value, 'unit': unit, 'basis': 'derived',
                        'source': ref(CURRENT/'inputs.json', cid+'/'+name+': '+str(item.get('location', 'independent input'))),
                        'context': {'species': 'human', 'matrix': 'plasma', 'study_role': 'development',
                                    **({'route': 'intravenous'} if parameter.endswith('clearance') else {})},
                        'notes': str(item)+'; absolute CL divided by declared source/reference body weight when applicable'}
                if 'renal_clearance_ml_min' in update['updates']:
                    raise ValueError('New renal partitions require an explicit independently reviewed scope')
                ledger = folder/'evidence.json'; save(ledger, {'evidence': list(evidence.values())})
                for req in (development, target):
                    req['evidence'] = deepcopy(list(evidence.values()))
                    req['evidence_ledger'] = ref(ledger, 'Independent construction inputs, RF fallback for missing values')
                    req['assumptions'] = list(dict.fromkeys(req['assumptions'] + update.get('limitations', [])))
        save(folder/'development.json', development); save(folder/'target.json', target)
        protocol['pins'].update(embedded_pins(development)); protocol['pins'].update(embedded_pins(target))
        for p in folder.glob('*.json'):
            summary['locks']['inputs'][str(p.relative_to(ROOT))] = sha256(p)
        case.update(updated_inputs=update['updates'], source_reference_weight_kg=update.get('normalization_weight_kg'),
                    renal_enabled='renal_clearance_ml_min' in update['updates'], reused_exact_fit=cache)
        protocol['cases'].append(case)
        summary['results'][cid] = {'rf': deepcopy(old['results'][cid]['rf'])}
        summary['locks']['predictions'][cid] = {'rf': deepcopy(old['locks']['predictions'][cid]['rf'])}
        if cache:
            summary['fits'][cid] = {}
            for formula in FORMULAS:
                prior = deepcopy(cached['fits'][cid][formula]); lock = prior['mapping']
                verify({lock['path']: lock['sha256']})
                path = ROOT/lock['path']; mapping = read(path); fit_input = read(path.parent/'input.json')
                request = {**development, 'partition_method': formula}
                if (mapping.get('warm_start_source') is not None or fit_input.get('warm_start_source') is not None
                        or fit_input['maximum_evaluations'] != 24 or mapping['evaluation_count'] != 24
                        or mapping['drug_signature'] != _drug_signature(request)
                        or mapping['training_source'] != case['development_source']):
                    raise ValueError('Cached fit protocol or request mismatch: '+cid+'/'+formula)
                prior.update(mapping_path=lock['path'], reused_exact_fit=True)
                summary['fits'][cid][formula] = prior
    protocol['source_pins'] = deepcopy(protocol['pins'])
    verify(protocol['pins']); verify(summary['locks']['inputs'])
    save(RUN/'protocol.json', protocol); save(RUN/'summary.json', summary)
    return summary


def execute(workers):
    from pbpk_kb.blueprints.curve_pattern_blueprint import fit_pattern_mapping
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    from pbpk_kb.blueprints.pk_anchored_blueprint import mapped_snapshot, _run, _drug_signature
    from pbpk_kb.blueprints.renal_extension import add_renal_outputs, read_renal_outputs
    summary = read(RUN/'summary.json') if RUN.exists() else prepare(workers)
    if summary['status'] not in {'prepared', 'fitting_development', 'predicting', 'failed'}:
        raise ValueError('Run already completed: '+summary['status'])
    protocol = summary['protocol']; verify(protocol['pins']); verify(summary['locks']['inputs'])
    started = time.monotonic(); prior_seconds = summary.get('elapsed_seconds', 0)

    def progress(status):
        summary.update(status=status, updated_at=now(), elapsed_seconds=round(prior_seconds+time.monotonic()-started, 1))
        save(RUN/'summary.json', summary)

    def fit(case, formula):
        cid = case['case_id']; req = read(RUN/'inputs'/cid/'development.json'); req['partition_method'] = formula
        folder = RUN/'fits'/cid/formula
        if (folder/'mapping_locked.json').exists():
            result = read(folder/'mapping_locked.json')
        elif folder.exists():
            raise ValueError('Incomplete fit requires inspection; will not overwrite '+str(folder))
        else:
            result = fit_pattern_mapping(reference_request=req, training_source=case['development_source'],
                output_dir=folder, project_root=ROOT, max_evaluations=24)
        return {'status': result['status'], 'eligible': result['eligible'], 'decision': result['decision'],
                'converged': result['converged'], 'calls': result['evaluation_count'],
                'failures': result['failed_evaluation_count'], 'reused_exact_fit': False,
                'mapping': ref(folder/'mapping_locked.json', 'Uniform no-warm-start development-only mapping'),
                'mapping_path': str((folder/'mapping_locked.json').relative_to(ROOT))}

    def predict(case):
        cid = case['case_id']; selected = summary['selection'][cid]
        folder = RUN/'predictions'/cid/'ours'
        if folder.exists():
            if (folder/'result.json').exists():
                result = read(folder/'result.json')
                if result['status'] == 'predicted':
                    verify({result['prediction_path']: result['prediction_sha256']})
                    return result
            raise ValueError('Incomplete target requires inspection: '+cid)
        folder.mkdir(parents=True)
        req = read(RUN/'inputs'/cid/'target.json'); req['partition_method'] = selected['formula']
        mapping = read(ROOT/selected['mapping_path'])
        if mapping['drug_signature'] != _drug_signature(req):
            raise ValueError('Drug construction changed from fit to target: '+cid)
        blueprint = assemble_evidence_blueprint(req, ROOT)
        if blueprint['status'] != 'assembled': raise ValueError('Incomplete target blueprint: '+cid)
        native = mapped_snapshot(blueprint, mapping['decision']['lipophilicity'], mapping['decision']['clearance_scale'])
        if case['renal_enabled']: native = add_renal_outputs(native)
        save(folder/'request.json', req)
        result = _run(native, req, folder, ROOT, audit=True)
        result.update(partition_method=selected['formula'], method_version=VERSION,
                      development_curve_used=True, comparison_curve_used=False)
        if result['status'] == 'predicted':
            result.update(prediction_path=str((folder/'prediction_locked.json').relative_to(ROOT)),
                          endpoints_path=str((folder/'endpoints.json').relative_to(ROOT)))
            if case['renal_enabled']:
                renal = read_renal_outputs(folder/'native/execution_manifest.json', native, ROOT)
                save(folder/'renal_prediction.json', renal)
                result['renal_prediction'] = ref(folder/'renal_prediction.json', 'Native parent-drug renal output')
        save(folder/'result.json', result)
        return result

    try:
        progress('fitting_development')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(fit, c, f): (c['case_id'], f) for c in protocol['cases'] for f in FORMULAS
                    if f not in summary['fits'].get(c['case_id'], {})}
            for future in as_completed(jobs):
                cid, formula = jobs[future]; result = future.result()
                summary['fits'].setdefault(cid, {})[formula] = result
                print(cid, formula, result['status'], result.get('decision', {}).get('objective'), flush=True)
                progress('fitting_development')
        for case in protocol['cases']:
            cid = case['case_id']; fits = summary['fits'][cid]
            eligible = [(f, r) for f, r in fits.items() if r['eligible']]
            if not eligible: raise ValueError('No eligible mapping: '+cid)
            formula, chosen = min(eligible, key=lambda pair: (pair[1]['decision']['objective'], pair[0] != 'PKSimStandard'))
            summary['selection'][cid] = {'formula': formula, 'mapping_path': chosen['mapping_path'], 'locked_at': now()}
            for fit_result in fits.values():
                lock = fit_result['mapping']; summary['locks']['mappings'][lock['path']] = lock['sha256']
        verify(protocol['pins']); verify(summary['locks']['mappings']); progress('predicting')
        with ThreadPoolExecutor(max_workers=min(workers, 10)) as pool:
            jobs = {pool.submit(predict, c): c['case_id'] for c in protocol['cases']}
            for future in as_completed(jobs):
                cid = jobs[future]; result = future.result(); summary['results'][cid]['ours'] = result
                if result['status'] == 'predicted':
                    summary['locks']['predictions'][cid]['ours'] = ref(result['prediction_path'], 'New native target prediction before scoring')
                print(cid, 'target', result['status'], flush=True); progress('predicting')
        verify(protocol['pins']); verify(summary['locks']['mappings'])
        if any(c['ours']['status'] != 'predicted' for c in summary['results'].values()):
            raise ValueError('A target prediction failed; retained diagnostics require review')
        summary.update(all_attempts_locked_before_scoring=True, target_values_used_for_selection=False,
                       finished_at=now(), fresh_fit_calls=sum(f['calls'] for fits in summary['fits'].values() for f in fits.values() if not f['reused_exact_fit']),
                       cached_fit_calls=sum(f['calls'] for fits in summary['fits'].values() for f in fits.values() if f['reused_exact_fit']),
                       fresh_target_predictions=10)
        summary.pop('failure', None); progress('predictions_locked')
    except BaseException as exc:
        summary['failure'] = type(exc).__name__+': '+str(exc); progress('failed'); raise


def status():
    s = read(RUN/'summary.json')
    counts = {}
    for c in s['protocol']['cases']:
        cid = c['case_id']; counts[cid] = {}
        for formula in FORMULAS:
            fit = s['fits'].get(cid, {}).get(formula)
            if fit:
                counts[cid][formula] = {'calls': fit['calls'], 'complete': True, 'cached': fit['reused_exact_fit']}
            else:
                p = RUN/'fits'/cid/formula/'progress.json'
                counts[cid][formula] = read(p) if p.exists() else {'status': 'not_started'}
    print(json.dumps({'status': s['status'], 'elapsed_seconds': s.get('elapsed_seconds'), 'fits': counts,
                      'targets': {k: v.get('ours', {}).get('status') for k,v in s['results'].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true'); action.add_argument('--status', action='store_true')
    action.add_argument('--report', action='store_true')
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()
    if not 1 <= args.workers <= 14: parser.error('workers must be between 1 and 14')
    if args.run: execute(args.workers)
    elif args.report:
        from report_current_blueprint import report
        report(RUN)
    else: status()


if __name__ == '__main__': main()
