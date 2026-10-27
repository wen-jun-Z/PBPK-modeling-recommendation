#!/usr/bin/env python3
"""Two explicitly historical development cases for the evidence-only builder."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from pbpk_kb.blueprints.evidence_blueprint import run_evidence_blueprint
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path('experiments/blueprint_design/evidence_assembly_20260922/inputs')
BLANCHARD = Path('experiments/blueprint_design/development_inputs/caffeine_studies_v1/blanchard_1983_absolute_bioavailability.json')


def score_observed_times(curve, observations):
    if curve.get('time_unit') != 'min' or curve.get('unit') != 'umol/l':
        raise ValueError('Prediction must be in minutes and umol/l')
    pt, pv = np.asarray(curve['time_min'], dtype=float), np.asarray(curve['concentration'], dtype=float)
    if (len(pt) < 2 or len(pt) != len(pv) or np.any(~np.isfinite(pt)) or np.any(~np.isfinite(pv))
            or np.any(np.diff(pt) <= 0) or np.any(pt < 0) or np.any(pv < 0)):
        raise ValueError('Malformed prediction trajectory')
    times = np.asarray([r['time_min'] for r in observations], dtype=float)
    observed = np.asarray([r['concentration_umol_per_l'] for r in observations], dtype=float)
    if (len(times) < 2 or np.any(~np.isfinite(times)) or np.any(~np.isfinite(observed))
            or np.any(observed <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError('Positive, ordered observations are required')
    if times[0] < curve['time_min'][0] or times[-1] > curve['time_min'][-1]:
        raise ValueError('Observation times exceed frozen prediction coverage')
    predicted = np.interp(times, curve['time_min'], curve['concentration'])
    if np.any(predicted <= 0) or np.any(~np.isfinite(predicted)):
        raise ValueError('Invalid predictions at observed times')
    errors = np.log(predicted / observed)
    auc = lambda values: float(np.sum(np.diff(times / 60) * (values[:-1] + values[1:]) / 2))
    fold = lambda a, b: float(max(a / b, b / a))
    return {'point_count': len(times), 'gmfe': math.exp(float(np.mean(np.abs(errors)))),
            'log_rmse': math.sqrt(float(np.mean(errors ** 2))),
            'observed_window_h': [float(times[0] / 60), float(times[-1] / 60)],
            'auc': {'observed_umol_h_l': auc(observed), 'predicted_umol_h_l': auc(predicted),
                    'fold_error': fold(auc(predicted), auc(observed)),
                    'definition': 'linear trapezoids at the same observed times only'},
            'sampled_cmax': {'observed_umol_l': float(max(observed)), 'predicted_umol_l': float(max(predicted)),
                             'fold_error': fold(max(predicted), max(observed)),
                             'definition': 'maximum at observed times, not true early Cmax'},
            'time_min': times.tolist(), 'observed_umol_l': observed.tolist(), 'predicted_umol_l': predicted.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = inside(args.output, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cases = [('blanchard', 'case_request.json', 'calibration_data_reconstruction'),
             ('wahllander', 'wahllander_request.json', 'separate_study_development_comparison')]
    files = [p for p in (ROOT / INPUTS).iterdir() if p.is_file()]
    files.append(ROOT / BLANCHARD)
    for _, filename, _ in cases:
        request = json.loads((ROOT / INPUTS / filename).read_text())
        files += [inside(e['source']['path'], ROOT, exists=True) for e in request['evidence']]
    files += [Path(__file__), ROOT / 'scripts/design_evidence_pbpk.py']
    for name in ('evidence_blueprint', 'endpoint_prediction', 'human_model_snapshot', 'human_model',
                 'design_runtime', 'source_workflow', 'source_adaptation_audit', 'evidence_native_audit'):
        files.append(ROOT / 'src/pbpk_kb/blueprints' / (name + '.py'))
    files += [ROOT / 'src/pbpk_kb/model_runtime/R' / name for name in
              ('blueprint_snapshot_run.R', 'source_adaptation_audit.R', 'evidence_native_audit.R')]
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    save(output / 'protocol.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'pins': pins,
        'cases': cases, 'independent_validation': False, 'parameter_selection_from_comparison': False,
        'no_complete_drug_model_input': True, 'endpoint_policy': 'Score equal observation windows and sampled peaks; no cross-study endpoint substitution'})
    save(output / 'progress.json', {'status': 'running', 'stage': 'native_prediction', 'mode': 'evidence_blueprint',
         'planned_native_jobs': 2, 'completed_native_jobs': 0, 'started_at': datetime.now(timezone.utc).isoformat()})

    def run(case):
        name, filename, role = case
        try:
            request = json.loads((ROOT / INPUTS / filename).read_text())
            result = run_evidence_blueprint(request=request, output_dir=output / name, project_root=ROOT, execute=True)
            return {'case': name, 'role': role, 'result': result}
        except Exception as error:
            return {'case': name, 'role': role, 'result': {'status': 'failed',
                    'native_executed': (output / name / 'native/execution_manifest.json').is_file(),
                    'error': type(error).__name__ + ': ' + str(error)}}
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, cases))
    prediction_pins = {}
    for row in results:
        if row['result']['status'] == 'predicted':
            try:
                prediction_pins[row['case']] = sha256(output / row['case'] / 'prediction_locked.json')
            except OSError as error:
                row['result'].update(status='prediction_lock_failed', error=str(error))
    save(output / 'predictions_lock.json', {'locked_before_scoring': True, 'predictions': prediction_pins})
    # Scoring data are opened only after every attempted prediction is persisted.
    for row in results:
        if row['result']['status'] == 'predicted':
            try:
                path = output / row['case'] / 'prediction_locked.json'
                if sha256(path) != prediction_pins[row['case']]:
                    raise ValueError('Frozen prediction changed')
                observed_path = BLANCHARD if row['case'] == 'blanchard' else INPUTS / 'wahllander_observations.json'
                if sha256(ROOT / observed_path) != pins[str(observed_path)]:
                    raise ValueError('Frozen observation source changed')
                data = json.loads((ROOT / observed_path).read_text())
                rows = (next(d['observations'] for d in data['datasets'] if d['dataset_id'] == 'blanchard_1983_iv_5mgkg')
                        if row['case'] == 'blanchard' else data['observations'])
                row['metrics'] = score_observed_times(json.loads(path.read_text()), rows)
                row['scoring_status'] = 'completed'
            except Exception as error:
                row.update(scoring_status='score_failed', scoring_error=type(error).__name__ + ': ' + str(error))
    unchanged = all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items())
    success = sum(r['result']['status'] == 'predicted' for r in results)
    scored = sum(r.get('scoring_status') == 'completed' for r in results)
    summary = {'status': 'completed' if unchanged and success == scored == len(cases) else 'completed_with_failures',
               'elapsed_seconds': round(time.monotonic() - started, 2), 'code_and_inputs_unchanged': unchanged,
               'attempted_cases': len(cases), 'scored_cases': scored, 'planned_native_jobs': 2,
               'completed_native_jobs': sum(bool(r['result'].get('native_executed')) for r in results), 'successful_native_jobs': success,
               'independent_validation': False, 'compound_count': 1, 'study_count': 2,
               'no_reliable_improvement_claim': True, 'results': results}
    save(output / 'summary.json', summary)
    save(output / 'progress.json', {'mode': 'evidence_blueprint', 'stage': 'completed', **{k: summary[k] for k in
         ('status', 'elapsed_seconds', 'planned_native_jobs', 'completed_native_jobs', 'successful_native_jobs')}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
