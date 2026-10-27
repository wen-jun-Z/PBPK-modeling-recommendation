#!/usr/bin/env python3
"""Report the user-selected full model against already completed baselines.

The original frozen code, protocol, and reports are never edited. Integrity
validation still uses the complete original registry; only the report's declared
comparison family changes. This does not complete the original 300-fit protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from pbpk_kb.recommendation.pbpk_match_v2 import reporting as r

DEFERRED = ('no_numeric', 'frozen_text', 'mlp', 'lambdamart')
DEFAULT_SCOPE = ROOT / 'runtime/deployment/pbpk_match_v2/scope.json'


def completion_gate(external, trained, pending):
    """Only four explicitly deferred method families may be missing."""
    allowed = {f'{method}/seed{seed}' for method in DEFERRED for seed in r.SEEDS}
    required = set(pending) - allowed
    required.update('external/' + m for m in set(r.EXTERNAL) - set(external))
    required.update(f'full/seed{s}' for s in set(r.SEEDS) - set(trained.get('full', {})))
    if set(external) - set(r.EXTERNAL):
        raise ValueError('Unregistered external method in own-only report')
    if set(trained.get('full', {})) - set(r.SEEDS):
        raise ValueError('Unregistered full-model seed')
    if required:
        raise RuntimeError('Own-model experiment incomplete: ' + ', '.join(sorted(required)))
    return sorted(set(pending) & allowed)


def paired(reports, data):
    """The same frozen estimators, with a new explicitly declared family."""
    expected = set(r.EXTERNAL) | {'full', r.NO_ERROR}
    if set(reports) != expected:
        raise ValueError('Own-only comparison family is incomplete')
    methods = list(reports)
    groups = {row['query_id']: data['groups'][i] for i, row in enumerate(data['query_records'])}
    quality = r.ev.paired_quality({m: v['quality'] for m, v in reports.items()}, groups, baseline=r.BASELINE)
    execution = []
    for rank in (1, 2, 3):
        rows = [row for report in reports.values() for row in report['execution']['position_rows']
                if row['rank'] == rank]
        result = r.ev.paired_method_statistics(rows, methods, baseline=r.BASELINE)
        result.update(rank=rank, comparison_family=methods,
            multiplicity='Holm over the own-only declared family versus F2LLM within each rank/stratum/endpoint; unavailable comparisons count as p=1.')
        execution.append(result)
    return {'state': 'computed', 'baseline': r.BASELINE, 'comparison_family': methods,
            'quality': quality, 'execution': execution,
            'interpretation': 'Full three-seed model and completed external baselines; document-only scores reuse the same fitted network. Seeds are not independent scientific studies.'}


def validate_scope(supplied, protocol_sha256):
    expected = {'schema': 'pbpk-match-v2-execution-scope-1', 'mode': 'own_model_only',
        'original_protocol_sha256': protocol_sha256, 'active_methods': ['full'],
        'deferred_methods': list(DEFERRED), 'seeds': list(r.SEEDS),
        'outer_folds': 5, 'inner_folds': 3, 'expected_fit_jobs': 60,
        'external_methods': list(r.EXTERNAL)}
    for name, value in expected.items():
        if supplied.get(name) != value:
            raise ValueError('Own-only scope differs: ' + name)


def summarize(data=None, output=r.OUT, *, scope_path=None):
    """Require all full-model folds/seeds and baselines before writing results."""
    output = r._scope(output)
    if data is None:
        from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
        data = load_dataset(output)
    # Do not change r.METHODS: this original reader verifies the original full
    # registry, source hashes, training/OOF memberships and every score receipt.
    external, trained, pending, refs, integrity = r._load_inputs(data, output)
    deferred_pending = completion_gate(external, trained, pending)
    refs.append(r._ref(Path(__file__)))
    scope = {'active_trained_methods': ['full'], 'deferred_trained_methods': list(DEFERRED),
             'reason': 'User requested training only the proposed model; completed baselines are reused.',
             'original_protocol_sha256': r.sha(output / 'protocol.json')}
    path = r._scope(scope_path if scope_path is not None else DEFAULT_SCOPE)
    supplied = json.loads(path.read_text())
    validate_scope(supplied, scope['original_protocol_sha256'])
    control = r._scope(r.ROOT / 'scripts/pbpk_match_v2_own_scope.py')
    if supplied.get('control_script_sha256') != r.sha(control):
        raise ValueError('Own-only scope supervisor source changed')
    scope['supervisor_scope'] = supplied
    refs.extend([r._ref(path), r._ref(control)])
    contract = r.ev.clinical_contract()
    refs.append(r._ref(r.ev.PRIMARY))
    reports, per_seed, variability, calibration = {}, {}, {}, {}
    for method in r.EXTERNAL:
        scores = external[method]
        probabilities, calibration[method] = r.external_probabilities(data, scores[0])
        reports[method] = r._evaluate(data, scores, probabilities, method, contract)
    seeds = trained['full']
    probabilities = np.mean([seeds[s]['probabilities'] for s in r.SEEDS], axis=0)
    for method, key in (('full', 'scores'), (r.NO_ERROR, 'document_scores')):
        per_seed[method] = {str(s): r._evaluate(data, seeds[s][key], seeds[s]['probabilities'], method, contract)
                            for s in r.SEEDS}
        variability[method] = r.seed_variability(list(per_seed[method].values()))
        scores = np.mean([r.percentile_ranks(seeds[s][key]) for s in r.SEEDS], axis=0)
        reports[method] = r._evaluate(data, scores, probabilities, method, contract)
    paired_seed = {}
    for seed in r.SEEDS:
        selected = {m: reports[m] for m in r.EXTERNAL}
        selected.update({m: rows[str(seed)] for m, rows in per_seed.items()})
        paired_seed[str(seed)] = paired(selected, data)
    result = {'schema': 'pbpk-match-v2-own-only-report-1', 'completed': True,
        'state': 'complete_own_only', 'original_protocol_completed': False, 'scope': scope,
        'active_trained_methods': ['full'], 'deferred_trained_methods': list(DEFERRED),
        'deferred_pending': deferred_pending, 'expected_external_methods': list(r.EXTERNAL),
        'expected_trained_methods': ['full'], 'seeds': list(r.SEEDS),
        'query_count': len(data['query_records']), 'candidate_count': len(data['candidate_records']),
        'document_request_count': len(data['core_indices']), 'execution_request_count': len(data['execution_indices']),
        'comparison_family': list(reports),
        'ensemble': {'method': 'equal seed mean of within-query exact-tie midpoint percentile ranks',
                    'probabilities': 'arithmetic mean of outer-calibrated seed probabilities',
                    'no_error_branch': 'Document scores from the same full fitted model; no additional training'},
        'reports': reports, 'single_seed_reports': per_seed, 'seed_mean_sd': variability,
        'external_calibration': calibration,
        'inference_integrity': {m: v for m, v in integrity.items() if m in reports},
        'inference_integrity_scope': 'Original complete artifact-bound audits: two held-out requests per full outer checkpoint; both full and document-only score checks retained. New external methods audit one core and one clinical request.',
        'paired_ensemble': paired(reports, data), 'paired_by_seed': paired_seed,
        'limitations': ['Internal grouped validation of existing observations, not independent prospective validation.',
            'The original five-trained-method 300-fit protocol remains incomplete; four method families are deferred.',
            'MLP, LambdaMART and separately trained ablations are not compared here.',
            'Document support and PK prediction errors remain separate evaluation targets.',
            'Exact ranks 1/2/3 use uniform ties, no backfill or best-of-three selection; endpoint windows remain separate.',
            'Coverage and conditional errors must be interpreted together; unknown and unfavorable outcomes are retained.',
            'Three seeds measure optimization variability, not independent scientific sample size.',
            'Existing concentration curves support case analysis only.'],
        'input_refs': refs}
    # The frozen registry may be large; verify every consumed source before
    # declaring completion. No skipped-family missing file counts as poor scores.
    for ref in refs:
        if r.sha(r._scope(r.ROOT / ref['path'])) != ref['sha256']:
            raise ValueError('Own-only report input changed: ' + ref['path'])
    destination = output / 'report_own_only'
    destination.mkdir(parents=True, exist_ok=True)
    r.write(destination / 'summary.json', result)
    compact = r._compact(reports)
    names = ['summary.json', 'README.md', 'tables.tex']
    for name, rows in compact.items():
        filename = name + '.csv'
        (destination / filename).write_text(r._csv(rows))
        names.append(filename)
    (destination / 'tables.tex').write_text('% Complete only for the declared own-model scope.\n' +
                                            r._tex(compact['quality'], True))
    for filename, text in r._paper_tables(reports, variability).items():
        (destination / filename).write_text('% Own-only scope; not automatically inserted into the manuscript.\n' + text)
        names.append(filename)
    (destination / 'README.md').write_text(
        '# PBPK-Match v2：自建模型与已完成对照\n\n'
        '自建模型的五折外层、三折内层和三个种子已完成；保留并复用已完成的9个外部对照。'
        '本目录只表示这一缩减范围完成，**不表示原300个拟合任务全部完成**。\n\n'
        '暂缓：去数值条件、冻结编码器、MLP、LambdaMART。去误差分支直接使用同一模型的文档分数，没有另外训练。\n\n'
        'summary.json包含逐请求结果、单种子结果、三种子波动与集成、校准与覆盖率、'
        '前三位各终点GMFE和遗漏、稳定性和配对统计。quality.csv、execution.csv、stability.csv为简表。'
        '各名次不递补，不取前三位最佳值；不同AUC窗口不合并。\n\n'
        '这是现有数据的分组内部验证。文档支持和执行预测误差分开报告；GMFE须结合覆盖率比较。'
        '论文表格片段已导出，但没有自动修改论文；未完成的消融不能作为已验证结论。\n')
    r.write(destination / 'completion.json', {'completed': True, 'state': 'complete_own_only',
        'original_protocol_completed': False, 'deferred_trained_methods': list(DEFERRED),
        'input_count': len(refs), 'outputs': [r._ref(destination / name) for name in names]})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, default=None)
    args = parser.parse_args()
    result = summarize(scope_path=args.scope)
    print(json.dumps({'state': result['state'], 'output': str(r.OUT / 'report_own_only'),
                      'deferred_trained_methods': result['deferred_trained_methods']}, ensure_ascii=False))
