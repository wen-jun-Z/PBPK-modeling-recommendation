#!/usr/bin/env python3
"""Publish the selected complementary-CL value study using frozen plot helpers.

This replaces the earlier value figure/table in the manuscript. It performs no
training, native simulation, score fitting, or manuscript edit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import render_blueprint_value as drawing
import run_blueprint_cl_refinement as runner
from pbpk_kb.blueprints.source_workflow import save

VERSION = 'selected-complementary-blueprint-value-publication-1'
DEFAULT = ROOT/'experiments/blueprint_design/blueprint_cl_refinement_complementary_20261004'
SELECTED = 'retained_animal_krr_equal_log_mean'
FIGURE_CAPTION = (
    'Evidence value and missing-input stability with the selected fixed complementary CL prior. '
    '(A) Paired curve GMFE for the eight drugs whose available independent evidence changes construction inputs; '
    'the predicted-only policy uses the same frozen predictor but ignores the available independent evidence. '
    '(B) Clinical curve-error ratio after removing evidence, relative to available inputs. '
    '(C) Prediction drift relative to available inputs. Removal affects six drugs for CL, six for fu and eight for both. '
    'Dots denote individual drugs and open diamonds the geometric mean; the dashed line is unity. '
    'Axes are logarithmic; no error bars or significance symbols are shown. Across the full cohort, 19 of 20 drugs '
    'were scored in every condition; the Treprostinil failure remains in request denominators. '
    'The evidence-priority/predicted-only paired ratio for the eight changed drugs is 0.626 '
    '(95% study-cluster bootstrap interval 0.324--1.043; 2,000 resamples). '
    'These previously viewed development data do not establish clinical suitability or uniform endpoint improvement.'
)


def verify_data(run):
    run = runner.own(run)
    if run != DEFAULT:
        raise ValueError('This publication replaces assets with the frozen selected complementary experiment only')
    state, protocol, result, by = drawing.verify_data(run)
    if protocol['version'] != runner.VERSION or set(state['outcomes']) != set(protocol['tasks']):
        raise ValueError('Unexpected clinical-refinement protocol or incomplete outcomes')
    runner.value.old.verify(protocol)
    prior = runner.value.old.checked(protocol['cl_prior'])
    if prior['selection']['selected'] != SELECTED or not prior['selected_using_training_only']:
        raise ValueError('Publication requires the selected fixed complementary CL prior')
    if not state.get('adoption_decision', {}).get('predeclared_joint_gate_passed'):
        raise ValueError('The predeclared joint development decision has not passed')
    failures = []
    for key, receipt in state['outcomes'].items():
        item = runner.value.old.checked(receipt); task = protocol['tasks'][key]; entry = item['entry']
        if (item.get('key') != key or item['case_id'] != task['case_id'] or not item.get('locked_at')
                or entry.get('target_observations_used') is not False
                or entry['status'] not in {'predicted', 'failed'}
                or runner.key_for(entry['request'], task['sidecars'], task['core']) != key):
            raise ValueError('Prediction identity, scientific inputs or information isolation changed')
        runner.value.old.collect_refs(item, {})
        if entry['status'] == 'predicted':
            runner.value._native_fields(entry)
        else:
            audit = entry.get('design_audit', {})
            failures.append({'case_id': item['case_id'], 'prediction_receipt': receipt,
                'original_wrapper_code': entry.get('failure'), 'root_causes': audit.get('reasons', []),
                'target_vss_l': audit.get('target_vss_l'), 'trajectory_calls': entry.get('linear_trajectory_calls')})
    for condition in drawing.CONDITIONS:
        summary = next(s for s in result['summaries'] if s['group'] == 'all_20' and s['condition'] == condition)
        if (summary['attempted'], summary['scored'], summary['failed']) != (20, 19, 1):
            raise ValueError('Publication coverage changed')
    for row, expected in zip(result['missingness'], (6, 6, 8)):
        actual = [r for r in result['units'] if r['condition'] == row['scenario'] and r['changed_from_nominal']]
        if row['actually_changed_drugs'] != expected or len(actual) != expected:
            raise ValueError('Missingness population differs from the fixed interventions')
        drift = [r['prediction_drift']['curve_geometric_fold_change'] for r in actual]
        ratios = [r['curve_metrics']['gmfe']/by[r['case_id'], 'available']['curve_metrics']['gmfe'] for r in actual]
        if not math.isclose(drawing.gm(drift), row['prediction_drift']['estimate'], rel_tol=1e-12):
            raise ValueError('Prediction-drift summary does not reproduce individual drugs')
        if not math.isclose(drawing.gm(ratios), row['masked_over_nominal_clinical_gmfe']['estimate'], rel_tol=1e-12):
            raise ValueError('Missingness curve ratio does not reproduce individual drugs')
    return state, protocol, result, by, failures


def profile(result):
    import pandas as pd
    sys.path.insert(0, str(drawing.SKILL/'scripts'))
    from profile_data import profile_data
    frame = pd.DataFrame([{'case_id': r['case_id'], 'condition': r['condition'],
        'changed_from_available': r['changed_from_nominal'], 'curve_gmfe': r['curve_metrics'].get('gmfe'),
        'prediction_drift': r['prediction_drift'].get('curve_geometric_fold_change')}
        for r in result['units']])
    for col in ('case_id', 'condition'):
        frame[col] = frame[col].astype('category')
    report = profile_data(frame, group_cols=['condition'])
    return json.loads(json.dumps(report, default=lambda value: value.item()),
                      parse_constant=lambda value: None)


def make_table(result):
    table = drawing.make_table(result)
    caption = (
        r'\caption{Clinical value of evidence selection with the selected fixed complementary CL prior '
        r'(equal log-weight fusion of the retained predictor, animal-assisted predictor and Tanimoto kernel regressor). '
        r'No same-drug concentration curve is fitted. The PBPK core and frozen parameter predictors are identical '
        r'across input policies. GMFE gives equal weight to each scored drug; threshold counts use all requested drugs, '
        r'including failures.}'
    )
    lines = table.splitlines()
    lines = [caption if line.startswith(r'\caption{') else line for line in lines]
    footnote = (
        r'All conditions score 19/20 drugs; Treprostinil remains a structural-capacity failure. '
        r'CL removal deletes total, hepatic and renal evidence together; missing inputs use the same selected predictor. '
        r'Only 6, 6 and 8 drugs change inputs under CL, fu and joint removal, respectively; the 8-drug block retains '
        r'all evidence-intervention drugs. The evidence-priority/predicted-only GMFE ratio for these eight drugs is '
        r'0.626 (95\% study-cluster bootstrap CI 0.324--1.043; 2,000 resamples). '
        r'These previously viewed development cases and descriptive thresholds do not establish clinical suitability. '
        r'Repeated scientific inputs are cached, not independent experiments.'
    )
    lines = [footnote if line.startswith('CL removal deletes') else line for line in lines]
    return '\n'.join(lines) + '\n'


def verify_pdf_fonts(path):
    """Resolve Type0 descendant descriptors, which the generic checker misses."""
    from pypdf import PdfReader
    fonts = []
    for page in PdfReader(path).pages:
        for reference in page['/Resources']['/Font'].values():
            font = reference.get_object()
            descendants = font.get('/DescendantFonts', [reference])
            for child in descendants:
                actual = child.get_object()
                descriptor = actual.get('/FontDescriptor')
                descriptor = descriptor.get_object() if descriptor else {}
                embedded = [k for k in ('/FontFile', '/FontFile2', '/FontFile3') if k in descriptor]
                if not embedded or actual.get('/Subtype') == '/Type3':
                    raise ValueError('PDF font is unembedded or Type3')
                fonts.append({'font': str(font.get('/BaseFont')), 'parent_type': str(font.get('/Subtype')),
                              'descendant_type': str(actual.get('/Subtype')), 'embedded_stream': embedded})
    return {'passed': True, 'fonts': fonts,
            'generic_warning_resolution': 'All three Type0 fonts have CIDFontType2 descendants with embedded FontFile2 streams'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    state, protocol, result, by, failures = verify_data(DEFAULT)
    data_profile = profile(result)
    drawing.style()
    sys.path.insert(0, str(drawing.SKILL/'scripts'))
    from visual_qa import audit_layout, render_preview
    from check_figure import check_figure
    fig = drawing.draw(protocol, result, by)
    fig.texts[1].set_text('Selected complementary CL prior; independent evidence retained when available')
    issues = audit_layout(fig)
    preview = ROOT/'runtime/cache/paper_figures/blueprint_selected_value_preview.png'
    render_preview(fig, str(preview), dpi=240)
    from PIL import Image
    with Image.open(preview) as image:
        image.convert('L').save(preview.with_name('blueprint_selected_value_gray.png'))
    if any(level in {'FAIL', 'WARN'} for level, message in issues):
        raise ValueError('Figure layout requires review: '+repr(issues))
    print(json.dumps({'preview': str(preview), 'layout_issues': issues,
                      'logical_units': len(result['units']), 'coverage': '19/20 per condition'}))
    if args.preview_only:
        return
    out = DEFAULT/'paper'; out.mkdir(exist_ok=True)
    figure = out/'clinical_value.pdf'; table = out/'clinical_value.tex'
    fig.savefig(figure, metadata={'Title': 'Clinical value with selected complementary CL prior', 'Creator': VERSION})
    table.write_text(make_table(result))
    export_issues, export_info = check_figure(str(figure), target_inches=(5.5, 5.55))
    if any(level == 'FAIL' for level, message in export_issues):
        raise ValueError('Publication figure compliance failed: '+repr(export_issues))
    font_audit = verify_pdf_fonts(figure)
    manifest = {'version': VERSION, 'protocol': state['protocol'], 'results': state['results'],
        'selected_prior': protocol['cl_prior'], 'renderer': runner.value.old.ref(Path(__file__)),
        'drawing_helpers': runner.value.old.ref(ROOT/'scripts/render_blueprint_value.py'),
        'clinical_runner': runner.value.old.ref(ROOT/'scripts/run_blueprint_cl_refinement.py'),
        'scientific_source_pins': protocol['source_pins'], 'scientific_code_pins': protocol['code_pins'],
        'assets': [runner.value.old.ref(figure), runner.value.old.ref(table)],
        'replaces': {'figure': 'previous clinical_value.pdf', 'table_label': 'tab:blueprint_value'},
        'figure_caption': FIGURE_CAPTION, 'figure_language': 'English', 'width_inches': 5.5,
        'height_inches': 5.55, 'clinical_population': 20, 'scored_population': 19,
        'changed_input_population': 8, 'actual_missingness_populations': [6, 6, 8],
        'logical_units': 100, 'curve_aggregation': 'geometric mean of per-drug curve GMFE',
        'threshold_comparison': '<', 'threshold_denominator': 'all attempted requests including failures',
        'failure_diagnostics': failures, 'data_profile': data_profile,
        'chart_design': 'Paired drug-level dots and missingness strip plots, logarithmic axes, no means-only bars',
        'layout_audit': issues, 'export_audit': export_issues, 'export_info': export_info,
        'independent_pdf_font_audit': font_audit,
        'visual_review': 'Color and grayscale previews inspected: labels and legends readable, panels aligned, no clipping or legend/data occlusion; marker shapes preserve policy identification',
        'grayscale_preview': str(preview.with_name('blueprint_selected_value_gray.png').relative_to(ROOT)),
        'new_training_calls': 0, 'new_solver_calls': 0}
    save(out/'publication_manifest.json', manifest)
    print(json.dumps({'figure': str(figure), 'table': str(table), 'export_audit': export_issues}))


if __name__ == '__main__':
    main()
