#!/usr/bin/env python3
"""Refine a completed clinical batch without retraining RF or changing cases.

Only development observations enter fitting. All new predictions are locked
before the existing scorer reads clinical references. Earlier source pins and
artifacts remain valid; optimization is a separate exploratory stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_clinical_method_comparison import FORMULAS, report
from run_complete_reference_pilot import ROOT, read, now, ref, verify
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

BASE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug'


def prepare(baseline, output, workers=12):
    baseline = inside(baseline, ROOT, exists=True)
    output = inside(output, ROOT)
    if output.exists():
        raise FileExistsError('Refinement outputs are immutable; choose a new directory')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError('Native worker count must be an integer from 1 through 16')
    base = read(baseline / 'summary.json')
    if base['status'] != 'reported' or not base.get('all_prediction_attempts_locked_before_scoring'):
        raise ValueError('Complete and report the baseline before refinement')
    verify(base['protocol']['pins'])
    for key in ('inputs', 'mappings'):
        verify(base['locks'][key])
    for methods in base['locks']['predictions'].values():
        verify({r['path']: r['sha256'] for r in methods.values()})
    protocol = deepcopy(base['protocol'])
    protocol.update(version='clinical-pattern-refinement-1.0', prepared_at=now(), workers=workers,
        fit_calls_per_formula=24, optimizer='curve-pattern-input-prior-1.0', recover_failed_trials=True,
        baseline_summary=ref(baseline / 'summary.json', 'Completed immutable original-algorithm batch'),
        baseline_comparison=ref(baseline / 'comparison_input.json', 'Prior clinical scoring; never read by fitting'),
        warm_start_policy='Re-evaluate each eligible same-formula baseline development decision with the new loss; count every native call',
        interpretation='Exploratory development refinement after viewing the baseline; not a new blinded test or equal-compute ablation',
        rf_policy='Reuse exact baseline native prediction and parameter-training artifacts; no retraining')
    for path in [Path(__file__), ROOT / 'scripts/render_pbpk_method_comparison.py',
                 ROOT / 'src/pbpk_kb/blueprints/curve_pattern_blueprint.py']:
        protocol['pins'][str(path.relative_to(ROOT))] = sha256(path)
    protocol['pins'].update({protocol[key]['path']: protocol[key]['sha256']
                             for key in ('baseline_summary', 'baseline_comparison')})
    protocol['pins'].update(base['locks']['mappings'])
    for methods in base['locks']['predictions'].values():
        protocol['pins'].update({r['path']: r['sha256'] for r in methods.values()})
    # The baseline's construction evidence and RF models are reused, not copied.
    output.mkdir(parents=True)
    save(output / 'summary.json', {'status': 'prepared', 'protocol': protocol,
        'fits': {}, 'results': {cid: {'rf': deepcopy(result['rf'])}
                              for cid, result in base['results'].items()},
        'locks': {'inputs': deepcopy(base['locks']['inputs'])},
        'parameter_training_reused': True})
    return protocol


def execute(output):
    from pbpk_kb.blueprints.curve_pattern_blueprint import fit_pattern_mapping
    from pbpk_kb.blueprints.development_curve_blueprint import predict_with_development_mapping

    output = inside(output, ROOT, exists=True)
    summary = read(output / 'summary.json')
    if summary['status'] != 'prepared':
        raise ValueError('Only an unstarted refinement may run')
    protocol = summary['protocol']; verify(protocol['pins']); verify(summary['locks']['inputs'])
    base = read(ROOT / protocol['baseline_summary']['path'])
    baseline = (ROOT / protocol['baseline_summary']['path']).parent
    started = time.monotonic(); pid = output / 'runner.pid'
    with pid.open('x') as handle:
        handle.write(str(os.getpid()) + '\n')

    def progress(stage):
        summary.update(status=stage, updated_at=now(), elapsed_seconds=round(time.monotonic()-started, 1))
        save(output / 'summary.json', summary)

    def fit(case, formula):
        cid = case['case_id']
        req = read(baseline / 'inputs' / cid / 'development_request.json')
        req['partition_method'] = formula
        previous = base['fits'][cid][formula]
        warm = ({'path': previous['mapping_path'],
                 'sha256': base['locks']['mappings'][previous['mapping_path']],
                 'location': 'Previous same-formula development fit; re-evaluation required'}
                if previous.get('eligible') else None)
        folder = output / 'fits' / cid / formula
        try:
            result = fit_pattern_mapping(reference_request=req, training_source=case['development_source'],
                output_dir=folder, project_root=ROOT, max_evaluations=protocol['fit_calls_per_formula'],
                warm_start_source=warm)
            result['mapping_path'] = str((folder / 'mapping_locked.json').relative_to(ROOT))
            return result
        except Exception as exc:
            return {'status': 'failed', 'eligible': False, 'reason': type(exc).__name__ + ': ' + str(exc)}

    def predict(case):
        cid = case['case_id']; selected = summary['selection'][cid]
        if selected['formula'] is None:
            return {'status': 'failed', 'reason': 'No feasible development fit in either partition formula'}
        req = read(baseline / 'inputs' / cid / 'target_request.json')
        req['partition_method'] = selected['formula']
        folder = output / 'predictions' / cid / 'ours'
        try:
            result = predict_with_development_mapping(request=req, mapping_path=ROOT / selected['mapping_path'],
                output_dir=folder, project_root=ROOT)
            result.update(prediction_path=str((folder / 'prediction_locked.json').relative_to(ROOT)),
                          endpoints_path=str((folder / 'endpoints.json').relative_to(ROOT)),
                          partition_method=selected['formula'])
            return result
        except Exception as exc:
            return {'status': 'failed', 'reason': type(exc).__name__ + ': ' + str(exc)}

    try:
        progress('development_fits')
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            jobs = {pool.submit(fit, case, formula): (case['case_id'], formula)
                    for case in protocol['cases'] for formula in FORMULAS}
            for job in as_completed(jobs):
                cid, formula = jobs[job]; result = job.result()
                summary['fits'].setdefault(cid, {})[formula] = result
                progress('development_fits')
                print(cid, formula, result['status'], flush=True)
        verify(protocol['pins']); verify(summary['locks']['inputs'])
        summary['selection'] = {}; summary['locks']['mappings'] = {}
        for case in protocol['cases']:
            cid = case['case_id']; fits = summary['fits'][cid]
            for result in fits.values():
                if result.get('mapping_path'):
                    summary['locks']['mappings'][result['mapping_path']] = sha256(ROOT / result['mapping_path'])
            eligible = [(formula, result) for formula, result in fits.items() if result.get('eligible')]
            chosen = min(eligible, key=lambda item: (item[1]['decision']['objective'], item[0] != 'PKSimStandard')) if eligible else None
            summary['selection'][cid] = {'locked_at': now(), 'formula': chosen[0] if chosen else None,
                                        'mapping_path': chosen[1]['mapping_path'] if chosen else None}
        progress('predicting_ours')
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            jobs = {pool.submit(predict, case): case['case_id'] for case in protocol['cases']}
            for job in as_completed(jobs):
                summary['results'][jobs[job]]['ours'] = job.result(); progress('predicting_ours')
        summary['locks']['predictions'] = {}
        for cid, results in summary['results'].items():
            locks = summary['locks']['predictions'][cid] = {}
            if 'rf' in base['locks']['predictions'][cid]:
                locks['rf'] = deepcopy(base['locks']['predictions'][cid]['rf'])
            if results['ours'].get('status') == 'predicted':
                locks['ours'] = ref(results['ours']['prediction_path'], 'Refined native prediction locked before re-scoring')
        verify(protocol['pins']); verify(summary['locks']['inputs']); verify(summary['locks']['mappings'])
        removed = []
        for cid, fits in summary['fits'].items():
            for formula, result in fits.items():
                if not result.get('eligible'):
                    continue
                chosen = result['decision']['evaluation']
                successful = {r['evaluation'] for r in result['evaluations']
                              if r.get('status') == 'ok' and r.get('eligible_for_selection')}
                for trial in sorted((output / 'fits' / cid / formula).glob('attempt_[0-9][0-9]')):
                    number = int(trial.name.split('_')[-1])
                    if number != chosen and number in successful and not trial.is_symlink():
                        inside(trial, output, exists=True)
                        removed.append(str(trial.relative_to(ROOT))); shutil.rmtree(trial)
        summary.update(finished_at=now(), evaluation_outcomes_read_during_fitting=False,
            all_prediction_attempts_locked_before_scoring=True, removed_unselected_trial_directories=removed,
            baseline_evaluation_already_viewed=True)
        progress('predictions_locked')
    except BaseException as exc:
        summary.update(failure={'type': type(exc).__name__, 'reason': str(exc), 'stage': summary.get('status')})
        progress('failed'); raise
    finally:
        if pid.is_file() and pid.read_text().strip() == str(os.getpid()):
            pid.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASE / 'run')
    parser.add_argument('--output', type=Path, default=BASE / 'optimized_run')
    parser.add_argument('--workers', type=int, default=12)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--report', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args(); output = inside(args.output, ROOT)
    if args.prepare:
        p = prepare(args.baseline, output, args.workers)
        print('Prepared', len(p['cases']), 'drugs; RF reused; 24 new calls/formula')
    elif args.run:
        execute(output)
    elif args.report:
        report(output)
        p = read(output / 'summary.json')['protocol']
        try:
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/render_pbpk_method_comparison.py'),
                '--baseline-input', str(ROOT / p['baseline_comparison']['path']),
                '--input', str(output / 'comparison_input.json'), '--preview'], cwd=ROOT, check=True, timeout=180)
        except (subprocess.SubprocessError, OSError) as exc:
            s = read(output / 'summary.json')
            s.update(status='report_failed', report_failure={'type': type(exc).__name__, 'reason': str(exc)})
            save(output / 'summary.json', s)
            raise
        s = read(output / 'summary.json'); s['report_mode'] = 'paired_stages'; save(output / 'summary.json', s)
    else:
        s = read(output / 'summary.json')
        progress_files = [read(path) for path in (output / 'fits').glob('*/*/progress.json')]
        total = len(s['protocol']['cases'])*len(FORMULAS)
        print(s['status'], 'formula fits:', sum(map(len, s['fits'].values())), f'/{total};',
              'new fit calls:', sum(p.get('native_evaluations', 0) for p in progress_files),
              f"/{total*s['protocol']['fit_calls_per_formula']}")


if __name__ == '__main__':
    main()
