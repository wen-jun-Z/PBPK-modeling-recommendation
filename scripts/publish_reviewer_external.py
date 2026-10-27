"""Publish all new-study cases, retaining explicit exposure/condition strata."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reviewer_external_validation import OUT, read, ref, checked

NAMES = {'bm25_text': 'BM25', 'qwen3_embedding_06b': 'Qwen3-Emb-0.6B',
         'identity_condition_rule': 'Identity/condition rule', 'service_typed_rule': 'Typed-match rule',
         'pbpk_match_light_document': 'PBPK-Match'}


def main():
    report = read(OUT / 'report/results.json')
    for source in report['sources']:
        checked(source)
    membership = {r['scenario_id']: r for r in report['membership']}
    rows = {(r['scenario_id'], r['method'], r['endpoint']): r for r in report['positions'] if r['rank'] == 1}
    order = [('external:pregabalin_pmc12511804_armA80mg', 'Pregabalin 80 mg; newly selected study'),
             ('external:sildenafil_pmc12640539_studyA_fct100mg', 'Sildenafil 100 mg, fasted; outcomes visible during screening'),
             ('external:sildenafil_pmc12640539_studyB_fct100mg', 'Sildenafil 100 mg, fed; outcomes visible during screening'),
             ('external:rosuvastatin_pmc13104799_day_minus4', 'Rosuvastatin 10 mg; prior metadata exposure, execution unavailable'),
             ('external:sildenafil_pmc13104799_day_minus4', 'Sildenafil, explicit 20 mg assumption; source dose unresolved')]
    lines = [r'\begingroup', r'\fontsize{8.2}{10}\selectfont', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
             r'Method & Cmax FE & AUC$_\infty$ FE & Tmax AE (h) & $t_{1/2}$ FE & Coverage (\%) \\']
    def number(value):
        return '--' if value is None else f'{value:.3f}'
    for sid, title in order:
        m = membership[sid]
        lines += [r'\midrule', r'\multicolumn{6}{@{}l@{}}{\textit{' + title + r'}} \\']
        for method in NAMES:
            values = [rows[(sid, method, endpoint)] for endpoint in ('Cmax', 'AUC_inf', 'Tmax', 'terminal_half_life')]
            coverages = {round(r['coverage'], 9) for r in values}
            if len(coverages) != 1:
                raise ValueError('Endpoint coverage must be displayed separately if it differs')
            fields = [number(v['tmax_mae_h'] if v['endpoint'] == 'Tmax' else v['gmfe']) for v in values]
            lines.append(NAMES[method] + ' & ' + ' & '.join(fields + [f"{100*values[0]['coverage']:.1f}"]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    dest = OUT.parent / 'paper/new_study_cases.tex'
    dest.write_text('\n'.join(lines) + '\n')
    (OUT.parent / 'paper/new_study_manifest.json').write_text(json.dumps({'source': ref(OUT / 'report/results.json'),
        'producer': ref(Path(__file__)), 'output': ref(dest)}, indent=2) + '\n')
    bib = ROOT / 'results/manuscript/references.bib'
    old = bib.read_text()
    extra = (OUT / 'sources/publications.bib').read_text()
    if 'external12511804' not in old:
        bib.write_text(old.rstrip() + '\n\n% Frozen reviewer new-study sources\n' + extra + '\n')
    print(dest)


if __name__ == '__main__':
    main()
