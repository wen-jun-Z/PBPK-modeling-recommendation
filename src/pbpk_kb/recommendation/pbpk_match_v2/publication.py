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
    sections = [('DOCUMENT', r'\subsection{推荐模型执行后的预测误差与遗漏}', 'paper_document.tex'),
                ('EXECUTION', r'\subsection{模型排序的稳定性与并列诊断}', 'paper_execution.tex'),
                ('STABILITY', r'\subsection{模型排序的运行成本与生成恢复}', 'paper_stability.tex')]
    for name, section, filename in sections:
        path = report / filename
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('Missing verified paper fragment: ' + filename)
        text = insert_before(text, section, name, path.read_text())
    text = insert_before(text, r'\subsection{PBPK模型蓝图设计}', 'METHOD', r'''
\paragraph{PBPK-Match v2的用途与误差联合学习}
PBPK-Match v2将原始需求文本与候选描述联合输入Qwen3-Reranker-0.6B，并将剂量、途径、间隔、输注、进食、制剂及终点窗口的匹配特征接入128维融合网络。仅在最后8层训练秩16的LoRA参数，联合学习文档排序、有序支持等级及适用／不适用／资料不足概率。随后冻结该表示，用训练组的执行结果学习逐终点绝对对数误差$|\ln(\hat y/y)|$；推理仅接收需求和候选资料。各终点至少需要3项独立训练研究，不满足时保留文档分数。预计误差作为非负惩罚修正文档排序；适用概率不等同于已验证可直接复用。
''')
    text = insert_before(text, r'\subsection{蓝图设计实验}', 'DESIGN', r'''
\paragraph{PBPK-Match v2的内部分组验证}
新增实验复用145条文档请求、41个执行场景及772条稳定性请求，按目标药物、共享研究和重复来源合为37组，采用外层五折、内层三折及种子17、23、41。文档与执行标签分别绑定自身请求；未知等级不补为负例。内层决定早停、温度校准及误差惩罚强度$\{0,0.1,0.3,1\}$：文档NDCG@10下降不超过0.02、覆盖率不降后，在所有合格强度共同可计分的场景选择较低首位误差。外层只评价，单种子与三种子集成分开报告；MLP与LambdaMART使用相同监督信息。此为已有资料上的内部验证，不是新增独立验证，不按评估观测重新拟合PBPK参数。
''')
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
