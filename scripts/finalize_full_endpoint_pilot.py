#!/usr/bin/env python3
"""Model-only horizon completion and transparent reporting of the clinical pilot.

Keep matched-window failures immutable. Native terminal/infinite-time predictions
are an explicitly separate estimator, not replacement scores or a model upgrade.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pymupdf

from pbpk_kb.blueprints.evidence_blueprint import run_evidence_blueprint
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.full_endpoint_evaluation import SCALAR_FIELDS
from pbpk_kb.blueprints.pk_anchor_moments import model_integral_pk
from pbpk_kb.blueprints.source_workflow import sha256, save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/blueprint_design/full_endpoint_pilot_20260927'
TABLE = ROOT/'results/tables/blueprint_full_endpoint_pilot'
NAMES = {'auc': r'AUC ($t_{\rm first}$--$t_{\rm last}$)', 'cmax': r'$C_{\max}$ (sampled)',
    'tmax': r'$T_{\max}$ (sampled)', 'c_last': r'$C_{\rm last}$',
    'terminal_half_life': 'Terminal half-life', 'auc_inf': r'AUC$_\infty$',
    'clearance': 'CL', 'vz': r'$V_z$', 'mrt': 'MRT', 'vss': r'$V_{\rm ss}$'}


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    return r'\textcolor{Gray}{---}' if value is None else f'{value:.4g}'


def main():
    clinical_reports = read(BASE/'inputs/observations/published_clinical_reference_review.json')
    original = BASE/'evaluation/alfentanil/cases/alfentanil_kharasch2012_control'
    supplement = BASE/'evaluation/alfentanil_horizon_completion'
    request = read(original/'request.json')
    # A fixed existing policy cap, selected without using the clinical endpoints.
    horizon_request = deepcopy(request)
    horizon_request['target']['window_end_h'] = 168.
    horizon_request['target'].pop('terminal_fit', None)
    if not supplement.exists():
        supplement.mkdir()
        save(supplement/'protocol.json', {
            'purpose': 'Numerical completion of native terminal/infinite-time endpoints; no accuracy optimization',
            'trigger': 'Matched 7-h model AUMC failed its fixed 20% tail-area gate',
            'fixed_horizon_h': 168., 'horizon_basis': 'existing MODEL_TAIL_POLICY maximum_horizon_h',
            'original_request_sha256': sha256(original/'request.json'),
            'original_prediction_sha256': sha256(original/'prediction_locked.json'),
            'original_scores_sha256': sha256(BASE/'evaluation/alfentanil/scores.json'),
            'model_parameters_changed': False, 'fit_performed': False,
            'observations_used_for_model_tail_selection': False,
            'orchestration_sha256': sha256(Path(__file__)),
            'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                (ROOT/'src/pbpk_kb/blueprints').glob('*.py')}})
        result = run_evidence_blueprint(request=horizon_request, output_dir=supplement/'native_prediction',
                                       project_root=ROOT, execute=True)
        if result['status'] != 'predicted':
            raise RuntimeError('Numerical completion failed: '+str(result))
    protocol = read(supplement/'protocol.json')
    for name, digest in protocol['source_hashes'].items():
        assert sha256(ROOT/name) == digest, 'Supplement code changed: '+name
    assert sha256(original/'request.json') == protocol['original_request_sha256']
    assert sha256(original/'prediction_locked.json') == protocol['original_prediction_sha256']
    assert sha256(BASE/'evaluation/alfentanil/scores.json') == protocol['original_scores_sha256']
    old_snapshot = read(original/'snapshot.json')
    new_snapshot = read(supplement/'native_prediction/snapshot.json')
    for snapshot in (old_snapshot, new_snapshot):
        for simulation in snapshot['Simulations']:
            simulation.pop('OutputSchema')
    assert old_snapshot == new_snapshot, 'Extending output time must not change the model'
    curve = read(supplement/'native_prediction/prediction_locked.json')
    old = read(original/'prediction_locked.json')
    sampled = np.interp(old['time_min'], curve['time_min'], curve['concentration'])
    scaled_delta = float(np.max(np.abs(sampled-np.array(old['concentration'])))/max(old['concentration']))
    assert scaled_delta < 1e-5, 'Old prediction window changed materially'
    native = summarize_extended_prediction(curve, request)
    save(supplement/'native_endpoints.json', native)
    integral = model_integral_pk(curve, request)
    save(supplement/'integral_convergence.json', integral)
    native_available = native['endpoint_counts']['completed']
    print('Native extended endpoints:', native_available, '/11; original-window peak-scaled difference:', scaled_delta, flush=True)
    rows, cases = [], []
    for drug in ('alfentanil', 'cefepime'):
        cohort = read(BASE/f'inputs/{drug}_cohort.json')['cases'][0]
        cid = cohort['case_id']
        score = read(BASE/f'evaluation/{drug}/scores.json')[cid]
        obs = read(ROOT/cohort['observations_path'])
        req = read(ROOT/cohort['request_path'])
        group = []
        for endpoint, (field, unit) in SCALAR_FIELDS.items():
            matched = score['endpoints'][endpoint]
            assert matched['reference_status'] == 'qualified' and matched['reference'] is not None
            native_terminal = endpoint in ('terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
            p, reason = matched['prediction'], matched['prediction_reason']
            basis = 'matched_observation_times'
            native_p = native_reason = None
            if drug == 'alfentanil' and native_terminal:
                item = native['endpoints'][endpoint]
                native_p = item.get(field) if item['status'] == 'completed' else None
                native_reason = item.get('reason')
            if drug == 'cefepime':
                reason = 'Construction blocked: permanent positive charge not represented; pka_types absent'
            o = matched['reference']
            fe = max(p/o,o/p) if p is not None and endpoint != 'tmax' and p>0 and o>0 else None
            row = {'case_id':cid, 'drug':drug, 'study':cohort['study'], 'dose_mg':req['dosing']['dose_mg'],
                'infusion_minutes':req['dosing']['infusion_minutes'], 'endpoint':endpoint, 'unit':unit,
                'prediction_P':p, 'recomputed_digitized_mean_curve_reference':o, 'fold_error_vs_recomputed_curve':fe,
                'published_clinical_reference':clinical_reports[drug]['endpoints'][endpoint]['value'],
                'published_clinical_reference_sd':clinical_reports[drug]['endpoints'][endpoint]['sd'],
                'published_clinical_reference_unit':clinical_reports[drug]['endpoints'][endpoint]['unit'],
                'published_clinical_reference_location':clinical_reports[drug]['endpoints'][endpoint]['location'],
                'absolute_error_h':abs(p-o) if p is not None and endpoint=='tmax' else None,
                'prediction_basis':basis, 'reference_basis':matched['reference_type'],
                'prediction_missing_reason':reason,
                'supplement_extended_native_prediction':native_p,
                'supplement_extended_native_reason':native_reason,
                'supplement_extended_native_fold_error':max(native_p/o,o/native_p) if native_p else None,
                'matched_prediction_P_star':matched['prediction'], 'matched_fold_error':matched['fold_error'],
                'matched_status':matched['status'], 'matched_missing_reason':matched['prediction_reason'],
                'curve_gmfe':score['endpoints']['plasma_concentration_time']['gmfe'],
                'curve_ln_rmse':score['endpoints']['plasma_concentration_time']['log_rmse'],
                'source_url':obs.get('source_url',obs['source'].get('primary_url',obs['source'].get('url')))}
            rows.append(row); group.append(row)
        cases.append({'id':cid,'drug':drug,'cohort':cohort,'request':req,'observation':obs,'rows':group,
                      'curve':score['endpoints']['plasma_concentration_time']})
    with TABLE.with_suffix('.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    save(BASE/'final_results.json', {'case_count':2,'drug_count':2,
        'all_11_clinical_reported_references_per_case':False,
        'all_11_digitized_curve_reconstructions_per_case':True,
        'clinical_reference_requirement_met':False,
        'reference_precision_note':'Cefepime tail passes at fixed digitized points, but +/-1-pixel uncertainty can cross the 0.98 adjusted-R2 gate; qualification is provisional with respect to digitization precision.',
        'clinical_observed_points':30,'scalar_reference_count':20,
        'scalar_predictions_available':sum(r['prediction_P'] is not None for r in rows),
        'native_model_executed_cases':1,'construction_failed_cases':1,
        'primary_endpoint_comparisons_available':sum(r['prediction_P'] is not None for r in rows)+1,
        'primary_endpoint_comparisons_expected':22,
        'horizon_completion':{'horizon_h':168.,'native_endpoints_completed':native_available,
            'model_snapshot_identical_except_output_schema':True,'peak_scaled_early_window_difference':scaled_delta,
            'matched_window_failures_preserved':True,'parameters_changed':False},
        'all_cases_retained':True,'rows':rows})
    # Keep the original matched-window estimator as primary, including failures.
    # The post-hoc extended simulation is a separately disclosed numerical check.
    lines=[r'\begingroup',r'\definecolor{Ink}{HTML}{233A4B}',r'\definecolor{Pale}{HTML}{EAF0F3}',
        r'\definecolor{Stripe}{HTML}{F6F9FA}',r'\definecolor{Gray}{HTML}{56636E}',
        r'\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{3pt}\fontsize{10}{13}\selectfont',
        r'\begin{longtable}{@{}p{45mm}*{5}{>{\centering\arraybackslash}p{35mm}}@{}}',
        r'\caption{Exploratory comparison with reconstructed digitized-mean-curve endpoints. This is NOT a complete 11-endpoint clinical-reported validation set. Published clinical values are listed separately below.}\\',
        r'\toprule\rowcolor{Pale} Endpoint & Unit & Prediction P* & Recomputed curve & FE vs. recomputed & Abs. error (h) \\\midrule\endhead']
    for i,case in enumerate(cases,1):
        curve_score=case['curve'];req=case['request']
        stat=(f"GMFE {fmt(curve_score['gmfe'])}; ln-RMSE {fmt(curve_score['log_rmse'])}"
              if curve_score['gmfe'] is not None else 'Construction blocked')
        lines += [r'\rowcolor{Ink}\multicolumn{6}{@{}p{237mm}@{}}{\color{white}\bfseries '+
                  f" {i:02d}. {case['cohort']['study']} / {case['drug'].capitalize()}"+r'\hfill '+stat+r'}\\*',
                  r'\rowcolor{Pale}\multicolumn{6}{@{}p{237mm}@{}}{'+
                  f" Dose {req['dosing']['dose_mg']:.4g} mg; IV "+
                  (f"infusion {req['dosing']['infusion_minutes']:.4g} min" if req['dosing']['infusion_minutes'] else 'bolus')+
                  f"; {len(case['observation']['observations'])} digitized clinical points. "+
                  ('9/11 endpoints scored; MRT and Vss predictions fail tail-quality gates.' if case['drug']=='alfentanil' else 'Permanent-charge representation is missing from the current constructor.')+r'}\\*']
        for j,row in enumerate(case['rows']):
            unit=row['unit'].replace('mg*h/L',r'mg$\cdot$h/L')
            line=' & '.join([NAMES[row['endpoint']],unit,fmt(row['prediction_P']),fmt(row['recomputed_digitized_mean_curve_reference']),fmt(row['fold_error_vs_recomputed_curve']),fmt(row['absolute_error_h'])])
            lines.append((r'\rowcolor{Stripe}' if j%2==0 else '')+line+(r'\\*' if j<9 else r'\\'))
        lines.append(r'\addlinespace[7pt]')
    lines.append(r'\rowcolor{Ink}\multicolumn{6}{@{}p{237mm}@{}}{\color{white}\bfseries Published clinical study results, verified from each article Table 1}\\*')
    lines.append(r'\rowcolor{Pale} Endpoint & \multicolumn{2}{c}{Alfentanil / IV control} & \multicolumn{2}{c}{Cefepime / 2000 mg DAY 1} & Velsecorat\\*')
    for endpoint in ('auc_inf','clearance','vz','mrt','vss'):
        entries=[]
        for drug in ('alfentanil','cefepime'):
            item=clinical_reports[drug]['endpoints'][endpoint]
            value=(r'\textit{Not reported}' if item['value'] is None else
                   fmt(item['value'])+r' $\pm$ '+fmt(item['sd'])+' '+item['unit'].replace('mg*h/L',r'mg$\cdot$h/L'))
            entries.append(value)
        candidate=clinical_reports['velsecorat']['endpoints'][endpoint]
        candidate_value=fmt(candidate['value'])+' '+candidate['unit'].replace('pmol*h/L',r'pmol$\cdot$h/L')
        lines.append(NAMES[endpoint]+r' & \multicolumn{2}{c}{'+entries[0]+r'} & \multicolumn{2}{c}{'+entries[1]+r'} & '+candidate_value+r'\\*')
    notes=[
        r'\textbf{Reference meaning.} Recomputed values above are project NCA on digitized group means, not study-reported endpoints. They must not fill missing clinical references. Published alfentanil/cefepime values are mean $\pm$ SD; neither study reports Vz. Velsecorat values are geometric means (Holmberg 2022, Table 1): six adults received 720 $\mu$g inhaled drug then 30 $\mu$g IV microtracer over 1 h. It is a new data candidate only; no prediction was run and Clast remains unverified. AUC units differ. No error against a published group mean is inferred from reconstruction error.',
        r'\textbf{Predictions and errors.} P* uses the actual clinical sampling times and the same NCA rules as O. FE = max(P*/O,O/P*); sampled Tmax uses absolute error. Original 7-h model MRT and Vss fail the 20\% AUMC-tail gate (39.01\% extrapolated). A separate model-only 168-h extension also fails its fixed tail-fit gate (adjusted R$^2$=0.9616); it does not replace any primary score. Blank predictions are failures, not absent references.',
        r'\textbf{NCA boundaries and quality.} Dose-time values are explicit NCA assumptions: two-point log back-extrapolation for alfentanil bolus, zero for first-dose cefepime infusion. They are excluded from observed sample counts, curve errors and sampled peaks. Fixed four-point observed tails require adjusted R$^2\geq0.98$, at least one half-life and AUC/AUMC extrapolation at most 20\%. Cefepime MRT subtracts half the 30-min infusion duration.',
        r'\textbf{Scope.} Historical pilot, not blinded clinical validation. Alfentanil inputs and partition choice come from Phimmasone 2001; outcomes come from Kharasch 2012. A generic male represents a mixed-sex mean; ondansetron pretreatment is not modeled and assay LLOQ is unreported in accessible sources. Cefepime inputs use an independent 1990 study; 1992 evaluation data are withheld from construction. Its digitized tail is provisional: $\pm$1-pixel perturbations can cross the 0.98 fit-quality gate. No parameter was fitted to these evaluation curves.',
    ]
    lines.append(r'\midrule')
    for note in notes:lines.append(r'\multicolumn{6}{@{}p{237mm}@{}}{\fontsize{8.5}{11}\selectfont '+note+r'}\\[3pt]')
    lines += [r'\bottomrule\end{longtable}\endgroup']
    TABLE.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    wrapper=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=270mm,paperheight=305mm,margin=15mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic]
\usepackage{amsmath,booktabs,longtable,array}
\usepackage[table]{xcolor}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=6pt}
\pagestyle{empty}
\begin{document}
\input{results/tables/blueprint_full_endpoint_pilot.tex}
\end{document}
'''
    temp=Path(tempfile.mkdtemp(prefix='full-endpoint-pilot.',dir=ROOT/'runtime/latex/tmp'))
    (temp/'table.tex').write_text(wrapper)
    result=subprocess.run([str(ROOT/'runtime/latex/tectonic-project'),'--only-cached','--keep-logs',
        '-Z','search-path='+str(ROOT),'--outdir',str(temp),str(temp/'table.tex')],cwd=ROOT,capture_output=True,text=True,timeout=180)
    log=result.stdout+result.stderr
    if result.returncode or any(s in log for s in ('Overfull','Missing character','error:')):
        raise RuntimeError(str(temp)+'\n'+log[-2400:])
    pdf=pymupdf.open(temp/'table.pdf')
    assert len(pdf)==1, 'One-page preview expected'
    text=pdf[0].get_text();assert text.count('Terminal half-life')==2
    shutil.copy2(temp/'table.pdf',TABLE.with_suffix('.pdf'))
    pdf[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(TABLE.with_suffix('.png'))
    pdf.close();shutil.rmtree(temp)
    print('Final CSV and PDF:',TABLE,flush=True)


if __name__ == '__main__':
    main()
