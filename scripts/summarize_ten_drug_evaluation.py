#!/usr/bin/env python3
"""Merge a frozen main run and its numerical supplement into a separate report."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time

from evaluate_ten_drug_endpoints import BATCH, DRUGS, ROOT, make_reports, now
from pbpk_kb.blueprints.source_workflow import save, sha256


def load(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def retry_failed_supplement():
    """Recover operational failures serially; no progress writer races quota scans."""
    from supplement_ten_drug_horizons import run_case
    source, output = BATCH / 'horizon_supplement', BATCH / 'horizon_retry'
    summary, protocol = load(source / 'summary.json'), load(source / 'protocol.json')
    failed = {r['case_id'] for r in summary['results'] if r['horizon_supplement']['status'] == 'failed'}
    if not failed:
        print('No operational horizon failures to retry')
        return 0
    if output.exists():
        raise FileExistsError('Preserve existing retry results')
    if not all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in protocol['pins'].items()):
        raise ValueError('Frozen source changed before recovery')
    cases = [c for c in protocol['cases'] if c['case_id'] in failed]
    pins = {**protocol['pins'], relative(source / 'summary.json'): sha256(source / 'summary.json'),
            relative(Path(__file__).resolve()): sha256(Path(__file__).resolve())}
    output.mkdir()
    save(output / 'protocol.json', {'cases': cases, 'pins': pins, 'workers': 1, 'started_at': now(),
         'reason': 'Serial recovery of operational failures; original failures retained; no concurrent progress writer during quota scan',
         'maximum_horizon_h': 1344., 'parameters_changed': False, 'quality_gates_changed': False})
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(output / 'runner.json', {'pid': os.getpid(), 'start_ticks': int(fields[19]), 'started_at': now()})
    rows, started = [], time.monotonic()
    def progress(status):
        save(output / 'progress.json', {'status': status, 'stage': 'serial_operational_recovery',
             'planned_native_jobs': len(cases), 'completed_native_jobs': len(rows),
             'elapsed_seconds': time.monotonic() - started})
    progress('running')
    for case in cases:
        rows.append(run_case(case, output, 1024**3))
        progress('running')
    unchanged = all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items())
    status = 'completed' if unchanged and all(r['horizon_supplement']['status'] == 'completed' for r in rows) else 'completed_with_failures'
    save(output / 'summary.json', {'status': status, 'source_and_code_pins_unchanged': unchanged,
         'new_native_attempts': sum(len(r['horizon_supplement']['attempts']) for r in rows), 'results': rows})
    save(output / 'predictions_lock.json', {'locked_at': now(), 'predictions': {r['case_id']: r['prediction_sha256'] for r in rows}})
    progress(status)
    print(json.dumps({'status': status, 'recovered_cases': len(rows)}))
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-horizon-failures', action='store_true')
    args = parser.parse_args()
    if args.retry_horizon_failures:
        return retry_failed_supplement()
    baseline, supplement, output = BATCH / 'evaluation', BATCH / 'horizon_supplement', BATCH / 'final_results'
    if output.exists():
        raise FileExistsError('The final report already exists; preserve it')
    base_protocol, extra_protocol = load(baseline / 'protocol.json'), load(supplement / 'protocol.json')
    base, extra = load(baseline / 'summary.json'), load(supplement / 'summary.json')
    if not base['status'].startswith('completed') or not extra['status'].startswith('completed'):
        raise ValueError('Finish both numerical phases first')
    pins = {**base_protocol['pins'], **extra_protocol['pins']}
    for directory in (baseline, supplement):
        for name in ('protocol.json', 'summary.json', 'predictions_lock.json'):
            p = directory / name
            pins[relative(p)] = sha256(p)
    retry_path = BATCH / 'horizon_retry'
    retry = load(retry_path / 'summary.json') if (retry_path / 'summary.json').exists() else None
    if retry:
        if retry['status'] != 'completed':
            raise ValueError('Operational recovery is not complete')
        pins.update(load(retry_path / 'protocol.json')['pins'])
        for name in ('protocol.json', 'summary.json', 'predictions_lock.json'):
            p = retry_path / name
            pins[relative(p)] = sha256(p)
        retry_rows = {r['case_id']: r for r in retry['results']}
        original_failures = {r['case_id'] for r in extra['results'] if r['horizon_supplement']['status'] == 'failed'}
        if set(retry_rows) != original_failures:
            raise ValueError('Recovery must correspond exactly to recorded failures')
        extra['results'] = [retry_rows.get(r['case_id'], r) for r in extra['results']]
    pins[relative(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    if not all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items()):
        raise ValueError('Frozen source or result changed before final reporting')
    rows = {r['case_id']: deepcopy(r) for r in base['results']}
    expected = {r['case_id'] for r in base['results'] if r['status'] == 'predicted' and any(
        r.get('endpoint_values', {}).get(e, {}).get('status') != 'completed'
        for e in ('terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss'))}
    if set(r['case_id'] for r in extra['results']) != expected:
        raise ValueError('Supplement does not cover the declared missing-endpoint cases')
    for row in extra['results']:
        old = rows[row['case_id']]
        if (row['compound_id'], row['role'], row['study_id']) != (old['compound_id'], old['role'], old['study_id']):
            raise ValueError('Supplement changed case identity or evaluation role')
        if row['prediction_path'] != old['prediction_path']:
            attempts = row['horizon_supplement']['attempts']
            accepted = [a for a in attempts if a['status'] == 'predicted']
            if not accepted or accepted[-1]['original_window_consistency']['status'] != 'passed':
                raise ValueError('No original-window consistency proof for supplemental prediction')
        rows[row['case_id']] = deepcopy(row)
    protocol = deepcopy(base_protocol)
    protocol.update(pins=pins, final_report_created_at=now(), baseline_protocol=relative(baseline / 'protocol.json'),
                    supplement_protocol=relative(supplement / 'protocol.json'),
                    maximum_simulation_end_h=1344., maximum_attempts_per_case=7,
                    extension_policy='Original <=168h run retained; missing terminal predictions supplemented at 336/672/1344h without changing parameters or quality gates',
                    numerical_supplement_cap_overrides_module_suggestion_only=True)
    output.mkdir()
    for cid in rows:
        (output / 'cases' / cid).mkdir(parents=True)
    save(output / 'protocol.json', protocol)
    save(output / 'predictions_lock.json', {'locked_at': now(), 'before_final_report_scoring': True,
         'main_batch_was_previously_scored': True, 'no_selection_on_clinical_scores': True,
         'predictions': {cid: row['prediction_sha256'] for cid, row in rows.items() if row.get('prediction_sha256')}})
    summary = make_reports(output, protocol, sorted(rows.values(), key=lambda row: row['case_id']))
    summary['main_native_attempts'] = base['native_attempts']
    summary['supplement_native_attempts'] = extra['new_native_attempts']
    summary['retry_native_attempts'] = retry['new_native_attempts'] if retry else 0
    summary['recorded_operational_failures_recovered'] = len(retry['results']) if retry else 0
    summary['original_horizon_supplement_status'] = extra['status']
    summary['horizon_supplement_status'] = 'completed_after_recorded_recovery' if retry else extra['status']
    summary['all_eleven_predictions_available'] = all(c['predicted'] == c['requested'] for c in summary['endpoint_coverage'].values())
    save(output / 'summary.json', summary)
    readme_path = output / 'README.md'
    text = readme_path.read_text().replace('The model horizon is at most 168 h.', 'The primary batch horizon is 168 h; 22 scenarios missing terminal endpoints additionally extend to 336/672/1344 h. Original time grids, window predictions, parameters, and quality thresholds remain unchanged.')
    text = text.replace('./scripts/progress.sh\n```',
        'runtime/app-env/bin/python -B scripts/supplement_ten_drug_horizons.py --workers 6\n'
        '# Resume serially only if the supplemental stage has operational failures:\n'
        'runtime/app-env/bin/python -B scripts/summarize_ten_drug_evaluation.py --retry-horizon-failures\n'
        'runtime/app-env/bin/python -B scripts/summarize_ten_drug_evaluation.py\n./scripts/progress.sh\n```')
    names = {'caffeine': 'Caffeine', 'midazolam': 'Midazolam', 'fluconazole': 'Fluconazole', 'alfentanil': 'Alfentanil',
             'alprazolam': 'Alprazolam', 'sufentanil': 'Sufentanil', 'moxifloxacin': 'Moxifloxacin',
             'amikacin': 'Amikacin', 'vancomycin': 'Vancomycin', 'furosemide': 'Furosemide'}
    requests = {c['case_id']: c['request'] for c in protocol['cases']}
    def value(row, endpoint, field):
        number = row['endpoint_values'].get(endpoint, {}).get(field)
        return '' if number is None else f'{number:.4g}'
    lines = ['', '## Prediction examples: one reference scenario per drug', '',
        'These are model predictions, not measurements. All use a 30-year-old male, 70 kg and 175 cm, with drug-specific reference dose and infusion duration. Curves, AUC, peaks, and last concentration cover 0–24 h. CSV includes all 109 scenarios.', '',
        '| Drug | Dose mg | Infusion min | AUC 0–24 mg·h/L | Cmax mg/L | Tmax h | C24 mg/L |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for drug in DRUGS:
        row = rows[f'{drug}_grid_d1p0_reference_male']
        dosing = requests[row['case_id']]['dosing']
        vals = [value(row, *pair) for pair in [('auc', 'value_mg_h_per_l'), ('cmax', 'value_mg_per_l'),
                ('tmax', 'value_h'), ('c_last', 'value_mg_per_l')]]
        lines.append('| ' + ' | '.join([names[drug], f"{dosing['dose_mg']:g}", f"{dosing.get('infusion_minutes', 0):g}", *vals]) + ' |')
    lines += ['', '| Drug | Half-life h | AUC infinity mg·h/L | CL L/h | Vz L | MRT h | Vss L |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for drug in DRUGS:
        row = rows[f'{drug}_grid_d1p0_reference_male']
        vals = [value(row, *pair) for pair in [('terminal_half_life', 'value_h'), ('auc_inf', 'value_mg_h_per_l'),
                ('clearance', 'value_l_per_h'), ('vz', 'value_l'), ('mrt', 'value_h'), ('vss', 'value_l')]]
        lines.append('| ' + ' | '.join([names[drug], *vals]) + ' |')
    lines += ['', 'Clearance and binding currently assume linearity. With the same human and infusion conditions, AUC/Cmax change approximately in proportion to dose while CL, half-life, Vz, MRT, and Vss may remain nearly constant. This does not establish absence of saturation in real drugs at these doses.', '',
              '## Results and limitations', '',
              'Other-study curve comparisons cover only eight drugs, 12 scenarios, and ten study groups. Amikacin and Moxifloxacin currently have only development reconstruction data, not evidence of cross-study generalization.', '',
              'Errors are reported separately by endpoint. Matching AUC does not guarantee accurate curve shape, peak, or tail. Drug inputs document missing demographics, surgical contexts mismatched to generic physiology, and multiple-ionization-site approximations. The Furosemide Haegeli curve was not scored; reported-endpoint comparisons remain.', '',
              '[Original primary results](../evaluation/summary.json) · [Extended horizons](../horizon_supplement/summary.json) · [Independent numerical review](../review.json)']
    if retry:
        lines += ['', 'One supplemental case failed before native solving because atomic progress-file replacement raced with directory counting. It was recovered serially without concurrent progress writes; the original failure and [recovery record](../horizon_retry/summary.json) remain.']
    readme_path.write_text(text + '\n'.join(lines) + '\n')
    (BATCH / 'README.md').write_text('# Ten-drug endpoint evaluation\n\n'
        'Final summary: [report and predictions](final_results/README.md).\n\n'
        '- [Predictions for all 109 scenarios](final_results/predictions.csv)\n'
        '- [Eleven endpoint types and missing-value reasons](final_results/endpoint_evaluation.csv)\n'
        '- [Clinical curve errors](final_results/clinical_comparison.csv)\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'results', 'stratified_metrics'}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
