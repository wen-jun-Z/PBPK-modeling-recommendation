"""Publish source-bound reviewer analyses without changing frozen experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/model_recommendation/reviewer_revision_20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path):
    data = json.loads(path.read_text())
    for name, expected in data.get('sources', {}).items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or sha(source) != expected:
            raise ValueError('Changed source: ' + name)
    return data


def full_list_table(data):
    rows = {r['method_id']: r for r in data['rows'] if r['scope'] == 'all'}
    frozen = json.loads((ROOT / 'results/data/model_recommendation_quality.json').read_text())
    order = [r['method_id'] for r in frozen['rows']] + ['pbpk_match_light', 'pbpk_match_components']
    lines = [r'\begingroup', r'\fontsize{9}{11.3}\selectfont',
             r'\definecolor{ReviewerSage}{HTML}{526B59}',
             r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}', r'\toprule',
             r'& \multicolumn{4}{c}{Top-1 grade (\%)} & Coverage (\%) & Sensitivity (\%) \\',
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(l){7-7}',
             r'Method & $g\geq2$ & $g=1$ & $g=0$ & Unknown & Known@10 & Y-sens. \\',
             r'\midrule',
             r'\multicolumn{7}{@{}l}{\textit{Frozen external configurations}} \\[1pt]']
    for method in order:
        row, binary = rows[method], data['binary_applicability'][method]
        if method == 'pbpk_match_light':
            lines += [r'\addlinespace[5pt]',
                      r'\multicolumn{7}{@{}l}{\textit{Grouped supervised outputs}} \\[1pt]']
        values = [row[k] for k in ('top1_supported', 'top1_limited', 'top1_unsupported',
                                  'top1_unknown', 'top10_known_fraction')]
        values.append(binary['sensitivity'])
        if abs(sum(values[:4]) - 1) > 1e-10:
            raise ValueError('First-rank states must partition all requests')
        label = row['method']
        if method in ('pbpk_match_light', 'pbpk_match_components'):
            label = r'\textcolor{ReviewerSage}{\textbf{' + label + '}}'
        lines.append(label + ' & ' + ' & '.join(f'{100*x:.1f}' for x in values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def purpose_table(data):
    rows = {r['scope']: r for r in data['rows'] if r['method_id'] == 'pbpk_match_light'}
    purposes = [('direct_prediction', 'Direct prediction'),
                ('mechanism_reference', 'Mechanism reference'),
                ('parameterization_template', 'Parameterization template')]
    lines = [r'\begingroup', r'\fontsize{9.5}{12}\selectfont',
             r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.2}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Conditional ranking} & \multicolumn{3}{c}{Full-list support (\%)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(l){4-6}',
             r'Purpose & Scorable / all & NDCG@10 & \shortstack{Top-1\\$g\geq2$} & \shortstack{Top-1\\unknown} & Known@10 \\',
             r'\midrule']
    for key, label in purposes:
        row = rows[key]
        lines.append(f"{label} & {row['ranking_eligible_requests']}/{row['requests']} & {row['ndcg@10']:.3f} & "
                     + ' & '.join(f'{100*row[k]:.1f}' for k in ('top1_supported', 'top1_unknown', 'top10_known_fraction')) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def input_reference_tables():
    """Publish input/rule references, keeping the complete experiment intact.

    Manuscript claims concern domain recommendation, not superiority of a
    fusion architecture. The text-only and conditions-only controls have
    different visible inputs; their table must not claim information matching.
    """
    directory = RUN / 'controls/paper'
    names = {'table.tex': 'model_recommendation_input_controls.tex',
             'clinical_table.tex': 'model_recommendation_input_controls_execution.tex'}
    sources, outputs = {}, {}
    for source_name, target_name in names.items():
        source = directory / source_name
        rows = source.read_text().splitlines()
        excluded = [row for row in rows if row.startswith('Text + conditions (concat) & ')]
        if len(excluded) != 1:
            raise ValueError('Expected exactly one concatenation comparison in the complete source table')
        target = ROOT / 'results/tables' / target_name
        target.write_text('\n'.join(row for row in rows if row not in excluded) + '\n')
        sources[str(source.relative_to(ROOT))] = sha(source)
        outputs[str(target.relative_to(ROOT))] = sha(target)
    return sources, outputs


def main():
    source = RUN / 'full_list/results.json'
    data = read_bound(source)
    out = RUN / 'paper'
    out.mkdir(exist_ok=True)
    outputs = {'full_list.tex': full_list_table(data), 'purpose_strata.tex': purpose_table(data)}
    for name, text in outputs.items():
        (out / name).write_text(text)
    # The frozen display formerly mixed boundary-inclusive known coverage with
    # the exact-ten estimand above. Preserve raw data, show coverage only once.
    quality_path = ROOT / 'results/tables/model_recommendation_quality.tex'
    quality = quality_path.read_text()
    if 'Known (\\%)' in quality:
        lines = []
        for line in quality.splitlines():
            if ' & ' in line and line.rstrip().endswith(r'\\'):
                cells = line.rsplit(' & ', 1)
                if len(cells) != 2:
                    raise ValueError('Unexpected historical quality table row')
                line = cells[0] + r' \\'
            line = line.replace('lrrrrrr@{}}', 'lrrrrr@{}}').replace(r'\multicolumn{7}', r'\multicolumn{6}')
            lines.append(line)
        quality_path.write_text('\n'.join(lines) + '\n')
    input_sources, input_outputs = input_reference_tables()
    receipt = {'schema': 'reviewer-publication-v1', 'source': {'path': str(source.relative_to(ROOT)), 'sha256': sha(source)},
               'producer': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(__file__)},
               'input_reference_sources': input_sources,
               'input_reference_scope': 'Single-input learned and deterministic references; not an exhaustive architecture comparison or evidence of interaction superiority. Complete experimental results are retained at the bound source paths.',
               'outputs': {**{str((out / name).relative_to(ROOT)): sha(out / name) for name in outputs},
                           str(quality_path.relative_to(ROOT)): sha(quality_path), **input_outputs}}
    (out / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'state': 'complete', 'tables': list(outputs)}))


if __name__ == '__main__':
    main()
