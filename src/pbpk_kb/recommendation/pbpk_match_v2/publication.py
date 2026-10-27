"""Add only complete, reproducible v2 statistics to the existing manuscript."""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess
from .experiment import OUT, ROOT, sha, write


def insert_before(text, section, name, content):
    begin, end = f'% PBPK_MATCH_V2_{name}_BEGIN', f'% PBPK_MATCH_V2_{name}_END'
    block = begin + '\n' + content.strip() + '\n' + end + '\n\n'
    pattern = re.compile(re.escape(begin) + r'.*?' + re.escape(end) + r'\n*', re.S)
    if begin in text:
        if len(pattern.findall(text)) != 1:
            raise ValueError('Ambiguous existing manuscript block: ' + name)
        return pattern.sub(lambda _: block, text)
    if text.count(section) != 1:
        raise ValueError('Manuscript section has moved or is ambiguous: ' + section)
    return text.replace(section, block + section)


def publish():
    report = OUT / 'report'
    completion = json.loads((report / 'completion.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    if not (completion.get('state') == 'complete' and summary.get('state') == 'complete'):
        raise ValueError('Incomplete v2 results cannot be published')
    # Revalidate source and model artifacts before editing any manuscript text.
    from .data import load_dataset
    from .reporting import summarize
    summarize(load_dataset(), require_complete=True)
    tex = ROOT / 'results/manuscript/main.tex'
    original, original_sha = tex.read_text(), sha(tex)
    text = original
    sections = [('DOCUMENT', '\\subsection{\u63a8\u8350\u6a21\u578b\u6267\u884c\u540e\u7684\u9884\u6d4b\u8bef\u5dee\u4e0e\u9057\u6f0f}', 'paper_document.tex'),
                ('EXECUTION', '\\subsection{\u6a21\u578b\u6392\u5e8f\u7684\u7a33\u5b9a\u6027\u4e0e\u5e76\u5217\u8bca\u65ad}', 'paper_execution.tex'),
                ('STABILITY', '\\subsection{\u6a21\u578b\u6392\u5e8f\u7684\u8fd0\u884c\u6210\u672c\u4e0e\u751f\u6210\u6062\u590d}', 'paper_stability.tex')]
    for name, section, filename in sections:
        path = report / filename
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('Missing verified paper fragment: ' + filename)
        text = insert_before(text, section, name, path.read_text())
    text = insert_before(text, '\\subsection{PBPK\u6a21\u578b\u84dd\u56fe\u8bbe\u8ba1}', 'METHOD', '\n\\paragraph{Joint applicability and error learning in PBPK-Match v2}\nPBPK-Match v2 jointly encodes raw requests and candidate descriptions with Qwen3-Reranker-0.6B. Dose, route, interval, infusion, food, formulation, and endpoint-window compatibility features enter a 128-dimensional fusion network. Rank-16 LoRA is trained only in the final eight layers, jointly learning document ranking, ordinal support grades, and applicable/inapplicable/insufficient-information probabilities. The representation is then frozen, and training-group execution results learn endpoint-specific absolute log error $|\\ln(\\hat y/y)|$; inference receives only requests and candidate information. Each endpoint requires at least three independent training studies; otherwise document scores remain. Predicted error is a nonnegative ranking penalty. Applicability probability does not establish validated direct reuse.\n')
    text = insert_before(text, '\\subsection{\u84dd\u56fe\u8bbe\u8ba1\u5b9e\u9a8c}', 'DESIGN', '\n\\paragraph{Grouped internal validation of PBPK-Match v2}\nThe experiment reuses 145 document requests, 41 execution scenarios, and 772 stability requests, combined into 37 groups by target drug, shared study, and duplicate source. It uses five outer folds, three inner folds, and seeds 17, 23, and 41. Document and execution labels remain bound to their own requests; unknown grades are not negatives. Inner folds select early stopping, temperature calibration, and error-penalty strength $\\{0,0.1,0.3,1\\}$. Eligible strengths must reduce document NDCG@10 by no more than 0.02 without decreasing coverage; selection minimizes top-1 error on scenarios scorable for every eligible strength. Outer folds evaluate only. Individual seeds and the three-seed ensemble are separate; MLP and LambdaMART use the same supervision. This is internal validation on existing data, not new independent validation. PBPK parameters are not refitted to evaluation observations.\n')
    destination = OUT / 'publication'
    destination.mkdir(parents=True, exist_ok=True)
    if text != original:
        backup = destination / ('main_before_' + original_sha[:12] + '.tex')
        if not backup.exists():
            backup.write_text(original)
        if sha(tex) != original_sha:
            raise RuntimeError('Manuscript changed concurrently; review before insertion')
        temporary = tex.with_name('main.v2.tmp')
        temporary.write_text(text)
        temporary.replace(tex)
    edited_sha = sha(tex)
    result = subprocess.run(['bash', str(ROOT / 'results/manuscript/build.sh')], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    (destination / 'build.log').write_text(result.stdout)
    if result.returncode:
        if sha(tex) == edited_sha:
            tex.write_text(original)
        raise RuntimeError('Paper compilation failed; original manuscript restored, project build.log retained')
    write(destination / 'completion.json', {'state': 'complete', 'original_tex_sha256': original_sha,
          'final_tex_sha256': sha(tex), 'pdf_sha256': sha(ROOT / 'results/manuscript/pbpk_paper_zh.pdf'),
          'report_sha256': sha(report / 'summary.json')})
