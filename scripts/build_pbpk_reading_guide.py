#!/usr/bin/env python3
"""Build the main-text figure/table guide from reviewed local content.

Only the guide is published. The paper and experimental artifacts are read-only.
LaTeX intermediates are removed after a successful build.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / 'results/manuscript'
SOURCE = PAPER_DIR / 'pbpk_reading_guide_zh.json'
OUTPUT = PAPER_DIR / 'pbpk_reading_guide_zh.pdf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tex(text):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(text))


def expected_references(data):
    refs = {ref for group in ('component', 'model', 'blueprint')
            for item in data[group] for ref in item['refs']
            if re.fullmatch('[\u56fe\u8868]\\d+', ref)}
    figures = {ref for ref in refs if ref.startswith('\u56fe')}
    if figures != {f'\u56fe{i}' for i in range(1, 10)}:
        raise ValueError('The guide must cover all nine main-text figures')
    table_numbers = sorted(int(ref[1:]) for ref in refs if ref.startswith('\u8868'))
    if not table_numbers or table_numbers != list(range(1, table_numbers[-1] + 1)):
        raise ValueError('Guide table references must be consecutively numbered from 1')
    return refs


def reference_scope(refs):
    figure_count = sum(ref.startswith('\u56fe') for ref in refs)
    table_count = sum(ref.startswith('\u8868') for ref in refs)
    return f'Figures 1–{figure_count}, Tables 1–{table_count}'


def paper_map(path, wanted):
    found = {}
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            for match in re.finditer(r'(?m)^(Figure|Table)\s+(\d+)\b', page.get_text()):
                kind = '\u56fe' if match[1] == 'Figure' else '\u8868'
                found.setdefault(kind + match[2], i + 1)
    if set(found) != wanted:
        raise ValueError('Main figure/table scope changed: ' + str(set(found) ^ wanted))
    return found


def display_reference(ref):
    # Preserve historical source keys while rendering English figure/table labels.
    if ref.startswith('\u56fe'):
        return 'Figure ' + ref[1:]
    if ref.startswith('\u8868'):
        return 'Table ' + ref[1:]
    return ref


def module(item, pages):
    refs = '; '.join(f'{display_reference(ref)}: paper page {pages[ref]}' for ref in item['refs'] if ref in pages)
    out = [r'\Block{' + tex(item['title']) + '}{' + tex(refs) + '}\n',
           r'\textbf{Purpose}\quad ' + tex(item['purpose']) + r'\par' + '\n']
    if item.get('intro'):
        out.append(tex(item['intro']) + r'\par' + '\n')
    if 'context_table' in item:
        context = item['context_table']
        out.append(table(context['headers'], context['rows'], context.get('widths', [.28, .67])))
    out.append(table(item.get('headers', ['Metric', 'Plain-language meaning']), item['metrics'], [.28, .67]))
    paragraphs = item['result'] if isinstance(item['result'], list) else [item['result']]
    out.append(r'\textbf{' + tex(item.get('result_label', 'Results')) + r'}\quad ')
    out.extend(tex(paragraph) + r'\par' + '\n' for paragraph in paragraphs)
    out.append(r'\vspace{5pt}' + '\n')
    return ''.join(out)


def chapter(title, subtitle, anchor):
    return (r'\clearpage\hypertarget{' + anchor + '}{}\n' +
            r'\pdfbookmark[0]{' + tex(title) + '}{' + anchor + '-bookmark}\n' +
            r'\Chapter{' + tex(title) + '}{' + tex(subtitle) + '}\n')


def table(headers, rows, widths):
    spec = '@{}' + ''.join(r'>{\raggedright\arraybackslash}p{' + str(w) + r'\linewidth}' for w in widths) + '@{}'
    out = [r'\begin{GuideTable}\begin{tabular}{' + spec + '}', r'\toprule',
           r'\rowcolor{Pale}' + ' & '.join(r'\textbf{' + tex(h) + '}' for h in headers) + r'\\\midrule']
    for row in rows:
        out.append(' & '.join(tex(c) for c in row) + r'\\[3pt]')
    out.extend([r'\bottomrule\end{tabular}\end{GuideTable}', ''])
    return '\n'.join(out)


def build_tex(data, pages, paper_page_count):
    fonts = ROOT / 'runtime/fonts'
    scope = reference_scope(expected_references(data))
    year, month, day = map(int, data['edition'].split('-'))
    edition = f'Edition {year:04d}-{month:02d}-{day:02d}'
    preamble = r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec,xeCJK}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setCJKmainfont[Path=FONTROOT/noto_serif_cjk/,BoldFont=NotoSerifCJKsc-Bold.otf,AutoFakeSlant=.15]{NotoSerifCJKsc-Regular.otf}
\setCJKsansfont[Path=FONTROOT/,AutoFakeBold=2]{NotoSansCJKsc-Regular.otf}
\usepackage[margin=18mm,top=20mm,bottom=18mm,headheight=14pt,headsep=8pt,footskip=20pt]{geometry}
\usepackage{amsmath,amssymb,array,booktabs,tabularx}
\usepackage[table]{xcolor}
\usepackage{fancyhdr}
\usepackage[hidelinks,bookmarksdepth=1]{hyperref}
\definecolor{Ink}{HTML}{111111}
\definecolor{Accent}{HTML}{355D76}
\definecolor{Muted}{HTML}{536875}
\definecolor{Pale}{HTML}{ECF2F6}
\hypersetup{pdftitle={PBPK main-text figure and table guide},pdfsubject={Purpose, metric definitions, and interpretation of the current main-text REFERENCE_SCOPE}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\fontsize{8.5}{11}\selectfont\color{Muted}PBPK main-text figure and table guide}
\fancyhead[R]{\fontsize{8.5}{11}\selectfont\color{Muted}GUIDE_EDITION · Main text only}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{.3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.14}
\newcommand{\Chapter}[2]{{\sffamily\bfseries\color{Accent}\fontsize{17}{23}\selectfont #1\par}\vspace{2pt}{\color{Muted}\small #2\par}\vspace{6pt}}
\newcommand{\Block}[2]{\vspace{6pt}{\sffamily\bfseries\color{Accent}\fontsize{12}{17}\selectfont #1\par}{\fontsize{8.8}{12}\selectfont\color{Muted}#2\par}\vspace{2pt}}
\newcommand{\Mini}[1]{\par\vspace{2pt}\textbf{#1}\par\vspace{-2pt}}
\newcommand{\ReadNote}[1]{\par{\small\textbf{Reading notes}\quad #1\par}\vspace{6pt}}
\newenvironment{GuideItems}{\begin{list}{\textcolor{Accent}{\textbullet}}{\setlength{\leftmargin}{12pt}\setlength{\labelsep}{5pt}\setlength{\topsep}{1pt}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}\setlength{\partopsep}{0pt}}}{\end{list}}
\newenvironment{GuideTable}{\par\vspace{3pt}\begingroup\fontsize{10}{14}\selectfont}{\endgroup\par\vspace{5pt}}
\begin{document}\raggedbottom\color{Ink}\fontsize{10.5}{15}\selectfont
'''.replace('FONTROOT', str(fonts)).replace('REFERENCE_SCOPE', tex(scope)).replace('GUIDE_EDITION', tex(edition))
    items = {item['id']: item for group in ('component', 'model', 'blueprint') for item in data[group]}
    out = [preamble]
    for index, spec in enumerate(data['sections']):
        if index:
            out.append(chapter(spec['title'], spec['subtitle'], spec['id']))
        else:
            out += [r'\hypertarget{guide-start}{}\pdfbookmark[0]{Overall methods and experimental aims}{guide-start-bookmark}',
                    r'\Chapter{PBPK main-text figure and table guide}{Purpose at a glance · Metric definitions · Brief result interpretation}',
                    tex(f'This guide covers only {scope} in the current paper main text. Page numbers refer to the {paper_page_count}-page paper PDF.') + r'\par']
        for part in spec['parts']:
            if part == 'pk_terms':
                out += [r'\Mini{PK metrics in Table 8}',
                        table(['Metric', 'Plain-language meaning'], data['pk_terms'], [.28, .67])]
            elif part == 'prediction_terms':
                out += [r'\Mini{Distinguish predictions from measured references}',
                        table(['Symbol', 'Plain-language meaning'], data['prediction_terms'], [.13, .82])]
            else:
                out.append(module(items[part], pages))
    out.append(r'\end{document}')
    return '\n'.join(out)


def main():
    data = json.loads(SOURCE.read_text())
    paper = ROOT / data['paper']
    paper_sha = digest(paper)
    with pymupdf.open(paper) as paper_doc:
        paper_page_count = paper_doc.page_count
    wanted = expected_references(data)
    pages = paper_map(paper, wanted)
    figure_count = sum(ref.startswith('\u56fe') for ref in pages)
    table_count = sum(ref.startswith('\u8868') for ref in pages)
    all_items = data['component'] + data['model'] + data['blueprint']
    item_ids = {item['id'] for item in all_items}
    placed = [part for section in data['sections'] for part in section['parts']
              if part not in ('pk_terms', 'prediction_terms')]
    assert len(placed) == len(set(placed)) and set(placed) == item_ids, 'Each module must appear once'
    covered = {r for item in all_items for r in item['refs'] if r in pages}
    assert covered == set(pages), 'The guide must explain every main-text figure and table'
    items_by_id = {item['id']: item for item in all_items}
    reference_sequence = [ref for part in placed for ref in items_by_id[part]['refs']
                          if ref in pages]
    assert reference_sequence == list(pages), 'Guide order must follow paper caption order'
    temp_root = ROOT / 'runtime/latex/tmp'
    temp_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='reading-guide.', dir=temp_root))
    source = temp / 'reading_guide.tex'
    source.write_text(build_tex(data, pages, paper_page_count))
    command = [str(ROOT / 'runtime/latex/tectonic-project'), '--only-cached', '--keep-logs',
               '--outdir', str(temp), str(source)]
    run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (temp/'build.log').write_text(run.stdout)
    problems = re.findall(r'^.*(?:Overfull|Missing character|undefined references|undefined citations|error:).*$'
                          , run.stdout, flags=re.M)
    if run.returncode or problems:
        print(run.stdout[-10000:])
        raise RuntimeError('Guide build needs review: '+str(temp))
    produced = temp / 'reading_guide.pdf'
    with pymupdf.open(produced) as doc:
        titles = ['PBPK main-text figure and table guide'] + [section['title'] for section in data['sections'][1:]]
        if len(doc) != len(titles):
            print('Page counts:', len(doc), '; expected:', len(titles))
            print([(i+1, p.get_text()[:110]) for i,p in enumerate(doc)])
            raise RuntimeError('Unexpected guide pagination; review '+str(temp))
        for page, title in zip(doc,titles):
            normalized=re.sub(r'\s+','',page.get_text())
            assert re.sub(r'\s+','',title) in normalized, (page.number+1,title)
        for i,p in enumerate(doc):
            for block in p.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for s in line['spans']:
                        box=pymupdf.Rect(s['bbox'])
                        assert box.x0 >= 35 and box.x1 <= p.rect.width-35, (i+1,s['text'],box)
                        assert box.y0 >= 20 and box.y1 <= p.rect.height-20, (i+1,s['text'],box)
    assert digest(paper)==paper_sha, 'The paper must remain unchanged'
    shutil.copyfile(produced, OUTPUT)
    audit={'scope':data['scope'],'date':data['edition'],'source':str(SOURCE.relative_to(ROOT)),
           'source_sha256':digest(SOURCE),'paper':str(paper.relative_to(ROOT)),'paper_sha256':paper_sha,
           'paper_pages':paper_page_count,'guide_pages':len(data['sections']),
           'covered_figures':figure_count,'covered_tables':table_count,
           'reference_pages':pages,'guide_sha256':digest(OUTPUT),'latex_checks':'passed',
           'reference_sequence':reference_sequence,'reference_order':'matches paper caption order',
           'text_bounds':'passed','experimental_data_changed':False,
           'visual_review':'Pending rendered-page inspection.'}
    (PAPER_DIR/'pbpk_reading_guide_zh_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    shutil.rmtree(temp)
    print('Published:', OUTPUT,
          f"({len(data['sections'])} pages; {figure_count} figures and {table_count} tables covered)")


if __name__ == '__main__':
    main()
