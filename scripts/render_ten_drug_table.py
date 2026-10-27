#!/usr/bin/env python3
"""Render Table 11; --run first predicts nine inputs from one frozen drug model."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import math
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.development_curve_blueprint import predict_with_development_mapping
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

SOURCE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/optimized_run'
DATA = ROOT / 'experiments/blueprint_design/one_drug_input_response'
DEST = ROOT / 'results/tables'
CASE_ID = 'treprostinil_lrx_triumph_ba001_iv'
ENDPOINTS = [
    ('plasma_concentration_time', r'Plasma $C(t)$', 'ng/mL', None, 1.),
    ('auc', r'$\mathrm{AUC}_{0-9}$', r'ng$\cdot$h/mL', 'value_mg_h_per_l', 1000.),
    ('cmax', r'$C_{\max}$', 'ng/mL', 'value_mg_per_l', 1000.),
    ('tmax', r'$T_{\max}$', 'h', 'value_h', 1.),
    ('c_last', r'$C_{9\mathrm{h}}$', 'ng/mL', 'value_mg_per_l', 1000.),
    ('terminal_half_life', r'$t_{1/2}$', 'h', 'value_h', 1.),
    ('auc_inf', r'$\mathrm{AUC}_{\infty}$', r'ng$\cdot$h/mL', 'value_mg_h_per_l', 1000.),
    ('clearance', 'CL', 'L/h', 'value_l_per_h', 1.),
    ('vz', r'$V_z$', 'L', 'value_l', 1.),
    ('mrt', 'MRT', 'h', 'value_h', 1.),
    ('vss', r'$V_{\mathrm{ss}}$', 'L', 'value_l', 1.),
]

def now():
    return datetime.now(timezone.utc).isoformat()

def rel(path):
    return str(Path(path).relative_to(ROOT))

def verify(pins):
    for path, expected in pins.items():
        if sha256(inside(path, ROOT, exists=True)) != expected:
            raise ValueError('Frozen input or output changed: ' + path)

def prepare():
    if (DATA / 'protocol.json').exists() or (DATA / 'predictions').exists():
        raise FileExistsError('Existing input-response predictions are immutable; default command only renders')
    upstream = json.loads((SOURCE / 'summary.json').read_text())
    selected = upstream['selection'][CASE_ID]
    mapping_path = selected['mapping_path']
    mapping = json.loads(inside(mapping_path, ROOT, exists=True).read_text())
    if not mapping.get('eligible') or mapping.get('algorithm_version') != 'curve-pattern-input-prior-1.0':
        raise ValueError('Expected an eligible locked optimized mapping')
    expected = upstream['locks']['mappings'][mapping_path]
    verify({mapping_path: expected})
    request_path = SOURCE / 'predictions' / CASE_ID / 'ours/request.json'
    reference = json.loads(request_path.read_text())
    if reference['compound_name'] != 'Treprostinil' or reference['partition_method'] != selected['formula']:
        raise ValueError('Locked source identity or partition does not match')
    pins = {mapping_path: expected, rel(request_path): sha256(request_path)}
    for source in [reference['evidence_ledger'], mapping['training_source'], *[e['source'] for e in reference['evidence']]]:
        pins[source['path']] = source['sha256']
    for name in ('development_curve_blueprint', 'pk_anchored_blueprint', 'evidence_blueprint',
                 'extended_prediction', 'endpoint_prediction', 'evidence_native_audit', 'design_runtime'):
        path = ROOT / f'src/pbpk_kb/blueprints/{name}.py'
        pins[rel(path)] = sha256(path)
    cases = []
    for multiplier, label in ((.5, 'd0p5'), (1., 'd1'), (2., 'd2')):
        for condition in ('C1', 'C2', 'C3'):
            request = deepcopy(reference)
            request['dosing']['dose_mg'] = reference['dosing']['dose_mg'] * multiplier
            if condition == 'C2':
                request['dosing']['infusion_minutes'] *= 2.
            if condition == 'C3':
                request['human'].update(sex='FEMALE', age_years=30., weight_kg=60., height_cm=165.)
                request['assumptions'] = reference['assumptions'][2:] + [
                    'Hypothetical female adult: 30 years, 60 kg and 165 cm; not a measured study subject.']
            request['assumptions'] += ['Input-response demonstration only; no clinical observations or error scores assigned.',
                'Absolute dose is held fixed across the three input conditions at each dose level.',
                'The drug mapping is reused unchanged without any refitting.']
            request = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
            cases.append({'case_id': f'treprostinil_{label}_{condition}', 'dose_multiplier': multiplier,
                'condition': condition, 'request': request,
                'repeats_clinical_representative_input': multiplier == 1. and condition == 'C1'})
    verify(pins)
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 'one-drug-input-response-1.0', 'created_at': now(), 'compound': 'Treprostinil',
        'mapping_path': mapping_path, 'mapping_sha256': expected, 'source_summary': rel(SOURCE / 'summary.json'),
        'partition_method': selected['formula'], 'mapping_algorithm': mapping['algorithm_version'],
        'source_case_id': CASE_ID, 'cases': cases, 'input_pins': pins, 'native_prediction_count': 9,
        'generator_sha256_at_launch': sha256(Path(__file__)),
        'parameters_refitted': False, 'clinical_observations_used': False,
        'interpretation': 'Prediction-only input response, not an additional clinical validation dataset',
        'notes': ['C1 age, sex and height are representative assumptions; weight is the source study mean.',
            'C2 doubles infusion duration at the same total dose, halving the infusion rate.',
            'C3 changes sex, weight and height together; it does not isolate a causal sex effect.',
            'The 1x C1 representative inputs were evaluated elsewhere and add no clinical sample.']}
    save(DATA / 'protocol.json', protocol)
    return protocol

def run_native(workers):
    protocol = prepare()
    started = time.monotonic()
    summary = {'status': 'running', 'started_at': now(), 'results': {},
        'protocol_sha256': sha256(DATA / 'protocol.json'), 'prediction_pins': {}}
    save(DATA / 'summary.json', summary)
    def execute(case):
        output = DATA / 'predictions' / case['case_id']
        try:
            result = predict_with_development_mapping(request=case['request'], mapping_path=protocol['mapping_path'],
                output_dir=output, project_root=ROOT)
            result['output_dir'] = rel(output)
        except Exception as exc:
            result = {'status': 'failed', 'reason': type(exc).__name__ + ': ' + str(exc), 'output_dir': rel(output)}
        return case['case_id'], result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(execute, c) for c in protocol['cases']]):
            cid, result = future.result()
            summary['results'][cid] = result
            summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
            save(DATA / 'summary.json', summary)
            print(json.dumps({'case': cid, 'status': result['status'], 'completed': len(summary['results'])}), flush=True)
    verify(protocol['input_pins'])
    if sha256(DATA / 'protocol.json') != summary['protocol_sha256']:
        raise ValueError('Protocol changed during prediction')
    for case in protocol['cases']:
        directory = DATA / 'predictions' / case['case_id']
        for name in ('request.json', 'prediction_locked.json', 'endpoints.json', 'result.json',
                     'construction.json', 'parameter_audit_result.json', 'readback_result.json'):
            path = directory / name
            if path.exists():
                summary['prediction_pins'][rel(path)] = sha256(path)
    summary.update(status='predictions_locked', finished_at=now())
    save(DATA / 'summary.json', summary)

def num(value):
    if value is None or value == '':
        return r'\BPmissing'
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite table value')
    text = f'{value:.3g}'
    if 'e' in text:
        base, power = text.split('e'); text = rf'{base}\!\cdot\!10^{{{int(power)}}}'
    return rf'\({text}\)'

def row(cells, shade=None):
    if len(cells) != 12:
        raise ValueError('Retained table structure has 12 columns')
    return (rf'\rowcolor{{{shade}}}' if shade else '') + ' & '.join(cells) + r' \\'

def note(text):
    return rf'\multicolumn{{12}}{{@{{}}p{{250mm}}@{{}}}}{{\fontsize{{8.5}}{{11.5}}\selectfont\color{{BPGray}}{text}}} \\'

def render():
    protocol = json.loads((DATA / 'protocol.json').read_text())
    summary = json.loads((DATA / 'summary.json').read_text())
    if summary['status'] not in {'predictions_locked', 'reported'}:
        raise ValueError('All nine prediction attempts must finish before rendering')
    verify(protocol['input_pins']); verify(summary['prediction_pins'])
    if sha256(DATA / 'protocol.json') != summary['protocol_sha256']:
        raise ValueError('Protocol checksum changed')
    records, case_rows, missing, receipts = [], [], [], []
    curves, finite_scalars = 0, 0
    for case in protocol['cases']:
        cid, request = case['case_id'], case['request']
        result = summary['results'][cid]; directory = ROOT / result['output_dir']
        endpoints = json.loads((directory / 'endpoints.json').read_text())['endpoints'] if result['status'] == 'predicted' else {}
        curve_path = directory / 'curve.csv'; curve = endpoints.get('plasma_concentration_time', {})
        if curve.get('time_h'):
            values = curve.get('concentration_mg_per_l')
            if values is None:
                mw = next(e['value'] for e in request['evidence'] if e['parameter'] == 'molecular_weight')
                values = [v * mw / 1000. for v in curve['concentration_umol_per_l']]
            with curve_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(['time_h', 'concentration_ng_ml'])
                for t, c in zip(curve['time_h'], values, strict=True):
                    if not math.isfinite(t) or not math.isfinite(c) or t < 0 or c < 0:
                        raise ValueError('Invalid model concentration')
                    writer.writerow([t, c * 1000.])
            curves += 1; summary['prediction_pins'][rel(curve_path)] = sha256(curve_path)
        receipt = directory / 'native/execution_manifest.json'
        if receipt.exists():
            receipts.append(rel(receipt))
        cells = [f"{num(request['dosing']['dose_mg'] * 1000.)} / {case['condition']} / {num(request['dosing']['infusion_minutes'])}"]
        for endpoint, _, unit, key, factor in ENDPOINTS:
            output = endpoints.get(endpoint, {}); value = None if key is None else output.get(key)
            state = output.get('status', 'completed' if output else 'failed')
            if value is not None:
                value = float(value) * factor
                if not math.isfinite(value):
                    raise ValueError('Nonfinite endpoint')
                finite_scalars += 1
            if endpoint == 'plasma_concentration_time':
                shown = (rf'\href{{\detokenize{{../../{rel(curve_path)}}}}}{{\textcolor{{BPTeal}}{{Curve CSV}}}}' if curve_path.exists() else r'\BPmissing')
            else:
                shown = num(value)
                if value is None:
                    missing.append({'case_id': cid, 'endpoint': endpoint, 'reason': output.get('reason', state)})
            cells.append(shown)
            records.append({'case_id': cid, 'compound': 'Treprostinil', 'dose_multiplier': case['dose_multiplier'],
                'dose_mg': request['dosing']['dose_mg'], 'condition': case['condition'],
                'infusion_minutes': request['dosing']['infusion_minutes'], **request['human'],
                'endpoint': endpoint, 'unit': unit.replace(r'$\cdot$', '.'), 'predicted_value': value, 'prediction_status': state,
                'curve_csv': rel(curve_path) if curve_path.exists() else '', 'observed_value': None, 'error': None,
                'blank_reason': 'Input-response illustration only; clinical references and errors are intentionally not assigned'})
        case_rows.append(cells)
    if len(records) != 99 or len(case_rows) != 9:
        raise ValueError('Expected one drug, nine inputs, eleven endpoint types')
    csv_path = DATA / 'endpoint_predictions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    out = [r'% Nine native input-response predictions; no refitting or clinical scoring.',
        r'\begingroup', r'\captionsetup[table]{name=Table}',
        r'\definecolor{BPInk}{HTML}{233A4B}', r'\definecolor{BPTeal}{HTML}{287C86}',
        r'\definecolor{BPFaint}{HTML}{EAF0F3}', r'\definecolor{BPStripe}{HTML}{F6F9FA}',
        r'\definecolor{BPGray}{HTML}{56636E}', r'\newcommand{\BPmissing}{\textcolor{BPGray}{---}}',
        r'\setlength{\tabcolsep}{2.5pt}', r'\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}',
        r'\setlength{\LTcapwidth}{250mm}', r'\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}',
        r'\renewcommand{\arraystretch}{1.12}', r'\fontsize{9.5}{12.5}\selectfont',
        r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{37mm}*{11}{>{\centering\arraybackslash}p{17.6mm}}@{}}',
        r'\caption{Treprostinil input-response example: three doses, three input conditions and eleven predicted endpoints.}\label{tab:blueprint_ten_drug}\\', r'\toprule']
    header = row([r'\shortstack[l]{Dose / input / infusion\\$\mu$g / C1--C3 / min}'] +
        [rf'\shortstack{{{label}\\{unit}}}' for _, label, unit, _, _ in ENDPOINTS], 'BPFaint')
    out += [header, r'\midrule', r'\endfirsthead', r'\toprule', header, r'\midrule', r'\endhead',
        r'\bottomrule', r'\endfoot', r'\bottomrule', r'\endlastfoot',
        r'\rowcolor{BPInk}\multicolumn{12}{@{}p{250mm}@{}}{\strut\hspace{1.5mm}\color{white}\bfseries Treprostinil\hfill Locked optimized blueprint; Rodgers--Rowland\hspace{1.5mm}}\\',
        r'\rowcolor{BPFaint}\multicolumn{12}{@{}p{250mm}@{}}{\strut\hspace{1.5mm}Predictions P only; the same drug mapping is reused for all nine inputs without refitting.}\\']
    out += [row(cells, 'BPStripe' if i // 3 % 2 == 0 else None) for i, cells in enumerate(case_rows)]
    out += [r'\addlinespace[4pt]',
        note(r'\textbf{Inputs.} C1: representative male adult, 30 y, 69.5 kg, 175 cm; 60-min IV infusion. C2: the same adult and total dose, infused over 120 min. C3: hypothetical female adult, 30 y, 60 kg, 165 cm; 60-min IV infusion. C3 changes several physiological inputs together, not sex alone.'),
        note(r'\textbf{Doses.} 31.275, 62.55 and 125.1 $\mu$g are 0.5, 1 and 2 times the reference total dose. Absolute dose is unchanged across C1--C3 at each level. The source study supplied the reference mean weight; age, sex and height remain declared representative assumptions.'),
        note(r'\textbf{Endpoints.} Curves, AUC and peak endpoints use 0--9 h; $C_{9\mathrm{h}}$ is the value at exactly 9 h, not an assay-limited last measurement. Terminal endpoints use the unchanged numerical-quality policy on the extended trajectory (simulated to 1344 h). MRT corrects for half the infusion duration; $V_{\rm ss}=\mathrm{CL}\times\mathrm{MRT}$.'),
        note(r'\textbf{Interpretation.} This illustrates input response, not nine clinical validations. Clinical references O and errors E are intentionally blank in the data. The 1$\times$ C1 configuration repeats representative inputs evaluated separately and adds no clinical sample. Model-prefix terminal estimates here differ from clinically sampled NCA estimands.'),
        note(r'\textbf{Reading.} Values are rounded to three significant figures. --- means a quality gate did not yield an endpoint, not zero. Near-constant dose-normalized parameters follow the current linear model assumptions and do not establish clinical dose proportionality.'),
        r'\end{longtable}', r'\endgroup', '']
    table = '\n'.join(out)
    if re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', table):
        raise ValueError('Table must remain English')
    DEST.mkdir(parents=True, exist_ok=True); tex_path = DEST / 'blueprint_ten_drug.tex'; tex_path.write_text(table)
    audit = {'source_directory': rel(DATA), 'output': rel(tex_path), 'output_sha256': sha256(tex_path),
        'protocol_sha256': sha256(DATA / 'protocol.json'), 'csv_path': rel(csv_path), 'csv_sha256': sha256(csv_path),
        'drug_count': 1, 'scenario_count': 9, 'endpoint_types': 11, 'endpoint_records': len(records),
        'actual_native_executions': len(receipts), 'native_receipts': receipts,
        'successful_predictions': sum(r['status'] == 'predicted' for r in summary['results'].values()),
        'curve_prediction_links': curves, 'scalar_predictions_displayed': finite_scalars, 'unavailable_scalars': missing,
        'observed_scalar_values_displayed': 0, 'scored_endpoint_records': 0, 'parameters_refitted': False,
        'mapping_sha256': protocol['mapping_sha256'], 'renderer_sha256': sha256(Path(__file__)), 'language': 'English'}
    save(DEST / 'blueprint_ten_drug_audit.json', audit)
    summary.update(status='reported', reported_at=now(), actual_native_executions=len(receipts), scalar_predictions=finite_scalars,
        curve_predictions=curves, unavailable_scalars=missing, csv_sha256=sha256(csv_path), table_sha256=sha256(tex_path))
    save(DATA / 'summary.json', summary)
    print(json.dumps({k: audit[k] for k in ('scenario_count', 'actual_native_executions', 'successful_predictions',
        'curve_prediction_links', 'scalar_predictions_displayed', 'output')}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('workers must be between 1 and 6')
    if args.run:
        run_native(args.workers)
    render()
