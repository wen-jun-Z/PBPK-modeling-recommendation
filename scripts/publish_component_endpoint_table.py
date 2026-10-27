#!/usr/bin/env python3
"""Append completed component-input results to the locked endpoint comparison.

This publication step never runs inference, changes reference qualification or
recomputes native PK predictions. Original fourteen-method rows are byte-value
preserved; only the independently recorded component-input configuration is added.
Author-confirmed aggregate corrections are applied at publication time, without
rewriting the frozen experiment reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import build_recommendation_endpoint_table as table

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'pbpk_match_components'
BASE = 'pbpk_match_light'
RUN = 'experiments/model_recommendation/component_integration_20261007'
FROZEN = 'results/data/model_recommendation_extended_metrics.json'
FROZEN_SHA = '0236c2c9f82d10a37b793080cb13a446d75b2b3bc3df3949bc0f99f15a1fd489'
AUTHOR_STATISTICS = 'results/data/component_input_publication.json'


def apply_author_statistics(rows, root=ROOT):
    """Keep the component Top-1 Cmax aggregate consistent with Table 12."""
    root = Path(root)
    path = root / AUTHOR_STATISTICS
    if not path.exists():
        return None
    statistics = table.read(path)
    value = float(statistics['diagnostics']['full_components'][1])
    coverage = statistics['cmax_coverage']
    if not math.isfinite(value) or value < 1:
        raise ValueError('Invalid author-confirmed Cmax GMFE')
    diagnostic_table = (root / 'results/tables/component_diagnostics.tex').read_text()
    match = re.search(r'Full components\s*&\s*[\d.]+\s*&\s*([\d.]+)', diagnostic_table)
    if match is None:
        raise ValueError('Table 12 full-component aggregate is missing')
    table.equal(value, float(match[1]), 'author statistics versus Table 12')
    selected = [r for r in rows if r['method_id'] == METHOD and r['rank'] == 1
                and r['endpoint_key'] == 'legacy:Cmax_reference_vs_group'
                and r['source_stratum'] == 'legacy:reference_individual']
    if len(selected) != 1:
        raise ValueError('Expected exactly one component Top-1 Cmax aggregate')
    row = selected[0]
    if [row['covered_scenarios'], row['denominator']] != coverage:
        raise ValueError('Author aggregate and endpoint table use different coverage')
    previous = row['gmfe']
    row['gmfe'] = value
    values = {'gmfe': value, 'coverage': coverage}
    return {'source': AUTHOR_STATISTICS, 'source_type': statistics['source_type'],
            'statistics_sha256': table.canonical_hash(values),
            'method_id': METHOD, 'endpoint_key': row['endpoint_key'], 'rank': 1,
            'field': 'gmfe', 'previous_published_value': previous, 'value': value,
            'coverage': coverage, 'underlying_experiments_recomputed': False}


def verify_experiment_sources(report, root, source_path):
    """Reject controls, stale reports and source paths outside the project."""
    root = Path(root).resolve()
    source_path = Path(source_path).resolve()
    if report.get('purpose') != 'paired_component_input_intervention':
        raise ValueError('Only the actual paired component intervention can be published; controls are excluded')

    def scoped_path(reference):
        if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
            raise ValueError('Malformed component source reference')
        path = (root / reference['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Component source is outside the project')
        digest = reference.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Malformed component source hash')
        return path

    completion_path = source_path.with_name('completion.json')
    if not source_path.is_relative_to(root) or not completion_path.resolve().is_relative_to(root):
        raise ValueError('Component report is outside the project')
    completion = table.read(completion_path)
    if completion.get('state') != 'complete':
        raise ValueError('Component evaluation completion receipt is incomplete')
    bindings = [r for r in completion.get('outputs', []) if scoped_path(r) == source_path]
    if len(bindings) != 1 or bindings[0]['sha256'] != table.digest(source_path):
        raise ValueError('Completion receipt does not bind the exact component summary')
    references = report.get('sources', [])
    if not isinstance(references, list) or not references:
        raise ValueError('Component evaluation has no source bindings')
    seen = {}
    for reference in references:
        path = scoped_path(reference)
        expected = reference['sha256']
        if path in seen:
            if seen[path] != expected:
                raise ValueError('Conflicting component source hashes')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('Component evaluation source changed: ' + str(path.relative_to(root)))
        seen[path] = expected
    return {'source_references_verified': len(references), 'unique_sources_verified': len(seen),
            'completion': {'path': str(completion_path.relative_to(root)), 'sha256': table.digest(completion_path)}}


def curve_summary(report, method, rank):
    """Use the original per-time-point twofold definition, not GMFE <= 2."""
    rows = [r for r in report['curves']['position_rows']
            if r['method_id'] == method and r['position'] == rank]
    if len(rows) != 2 or len({r['scenario_id'] for r in rows}) != 2:
        raise ValueError('Curve table requires exactly the two registered cases')
    measured = [r for r in rows if r['curve_coverage'] > 0]
    return {'coverage': sum(r['curve_coverage'] for r in rows) / len(rows),
            'covered_scenarios': len(measured), 'scenarios': len(rows),
            'gmfe': math.exp(sum(r['mean_abs_log_error'] for r in measured) / len(measured)) if measured else None,
            'within_twofold': sum(r['within_twofold_fraction'] for r in measured) / len(measured) if measured else None}


def component_rows(report, original, method):
    """Map 28 qualified extension panels and 8 original panels exactly."""
    extended = table.unique([r for r in report['extended_endpoints']['publication_rows']
                             if r['method_id'] == method], ('endpoint_key', 'source_stratum', 'rank'))
    scalar = table.unique(report['reports'][method]['execution']['position_summary'], ('endpoint', 'rank'))
    rows = []
    for template in original['publication_rows']:
        if template['method_id'] != BASE:
            continue
        row = deepcopy(template)
        row['method_id'] = method
        if row['source_stratum'] == 'legacy:reference_individual':
            source = (curve_summary(report, method, row['rank']) if row['family'] == 'curve'
                      else scalar[row['endpoint'], row['rank']])
            table.equal(source['scenarios'], row['denominator'], 'original-panel denominator')
            for field in ('gmfe', 'within_twofold', 'coverage', 'covered_scenarios'):
                row[field] = source[field]
            row['coverage_eligible_reference'] = source['coverage']
            row['coverage_total41'] = source['coverage'] * row['denominator'] / 41
        else:
            source = extended[row['endpoint_key'], row['source_stratum'], row['rank']]
            for field in ('endpoint', 'family', 'quantity', 'route', 'steady_state', 'window_h',
                          'body_weight_normalized', 'dose_normalization', 'metric_kind',
                          'eligible_reference_scenarios', 'requested_reference_scenarios', 'total_scenarios'):
                if source[field] != row[field]:
                    raise ValueError('Changed endpoint definition or qualification: ' + field)
            row.update(source)
            row['coverage'] = source['coverage_eligible_reference']
            row['denominator'] = source['eligible_reference_scenarios']
        rows.append(row)
    if len(rows) != len(original['endpoint_panels']) * 3:
        raise ValueError('Component endpoint grid is incomplete')
    return rows


def build(root=ROOT):
    root = Path(root)
    frozen_path = root / FROZEN
    if table.digest(frozen_path) != FROZEN_SHA:
        raise ValueError('The original locked fourteen-method metrics changed')
    original = table.read(frozen_path)
    source_path = root / RUN / 'report/summary.json'
    report = table.read(source_path)
    if report['state'] != 'complete':
        raise ValueError('Component experiment is incomplete')
    source_validation = verify_experiment_sources(report, root, source_path)
    # A zero-change replay must reproduce the old values before new results can
    # be combined. In particular, this catches wrong stratum/coverage mappings.
    replay = component_rows(report, original, BASE)
    expected = [r for r in original['publication_rows'] if r['method_id'] == BASE]
    for before, after in zip(expected, replay):
        for field in ('gmfe', 'mae_h', 'within_twofold', 'coverage', 'denominator',
                      'covered_scenarios', 'coverage_total41'):
            table.equal(before[field], after[field], 'baseline replay:' + field)
    combined = deepcopy(original)
    added = component_rows(report, original, METHOD)
    author_update = apply_author_statistics(added, root)
    combined['publication_rows'] += added
    combined['publication_method_count'] = 15
    methods = deepcopy(original['methods']) + [{'method_id': METHOD, 'method': 'PBPK-Match + components'}]
    combined['methods'] = methods
    overview = table.overview_data(combined, methods)
    old_overview = table.overview_data(original, original['methods'])
    if overview['selected_definitions'] != old_overview['selected_definitions']:
        raise ValueError('Adding a configuration changed the endpoint selection')
    overview.update(schema='pbpk-component-endpoint-publication-v1',
                    publication_rows=combined['publication_rows'],
                    source_files={FROZEN: table.digest(frozen_path), str(source_path.relative_to(root)): table.digest(source_path)},
                    producer={'path': 'scripts/publish_component_endpoint_table.py', 'sha256': table.digest(__file__)},
                    renderer={'path': 'scripts/build_recommendation_endpoint_table.py',
                              'sha256': table.digest(table.__file__)},
                    source_validation=source_validation,
                    author_aggregate_update=author_update,
                    replay={'rows_checked': len(replay), 'original_rows_preserved': len(original['publication_rows']),
                            'new_rows': len(added), 'baseline_values_unchanged': True, 'inference_performed': False,
                            'curve_twofold_definition': 'Mean per-case fraction of qualified concentration points within twofold.'})
    return combined, methods, overview


def publish(root=ROOT):
    root = Path(root)
    combined, methods, overview = build(root)
    target = root / 'results/tables/model_recommendation_eleven_endpoints'
    target.with_suffix('.tex').write_text(table.overview_latex(combined, methods))
    target.with_name(target.name + '_all_methods').with_suffix('.tex').write_text(table.additional_matrices_latex(combined, methods))
    rows = combined['publication_rows']
    with target.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    destination = root / 'results/data/component_model_recommendation_endpoint_overview.json'
    destination.write_text(json.dumps(overview, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return {'method_count': len(methods), 'endpoint_definitions': len(combined['endpoint_panels']),
            'csv_rows': len(rows), 'baseline_values_unchanged': True,
            'output': str(destination.relative_to(root))}


if __name__ == '__main__':
    print(json.dumps(publish(), ensure_ascii=False, indent=2))
