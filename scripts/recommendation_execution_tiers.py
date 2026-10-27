"""Freeze the existing deterministic engineering-use policy before PK scoring.

This is an additional declared rule policy, not a classifier inferred from the
12 rankers. Its rejection is lack of existing-rule clearance, never clinical
inapplicability. No service, trained checkpoint, outcomes or LLM are accessed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.recommendation.model_matching.graphs import (
    build_model_graph, build_execution_requirement_graph, endpoint_categories,
    GRAPH_SCHEMA_VERSION,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    score_candidate, assess_candidate_use, deterministic_use_class, SCORER_VERSION,
)

OUT = ROOT / 'experiments/model_recommendation/execution_validation_20261001'
CATALOG = ROOT / 'experiments/model_recommendation/frontier_core_145_20260921/inputs/candidates.jsonl'
METADATA = [
    'data_process/model_catalog/osp_v12_3_1/model_versions.jsonl',
    'experiments/model_recommendation/model_expansion_20260909/catalog_candidates.json',
    'data_process/model_assets/rosuvastatin_v1_grade3_registration/model_versions.jsonl.gz',
]
CODE = ['scripts/recommendation_execution_tiers.py',
        'src/pbpk_kb/recommendation/model_matching/graphs.py',
        'src/pbpk_kb/recommendation/model_matching/scoring.py']
CLASS_NAMES = {0: 'direct_reuse', 2: 'reference_only', 3: 'not_cleared_by_existing_rules'}
VERSION = 'existing-engineering-use-decision-v1'


def inside(path):
    p = (ROOT / path).resolve(strict=True)
    if p == ROOT or not p.is_relative_to(ROOT) or any(x.casefold().startswith('sealed') for x in p.relative_to(ROOT).parts):
        raise ValueError('Unsealed project resource required')
    return p


def sha(path):
    with inside(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def reference(path):
    p = inside(path)
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def rows(path):
    p = inside(path)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if p.suffix == '.json':
        return json.loads(p.read_text())
    return [json.loads(line) for line in p.open() if line.strip()]


def frozen_write(path, value, *, jsonl=False):
    path = ROOT / path
    parent = inside(path.parent)
    path = parent / path.name
    payload = (''.join(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n' for x in value)
               if jsonl else json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if path.exists():
        if path.read_text() != payload:
            raise ValueError('Frozen declared-use input changed: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(payload)


def clinical_task(scenario):
    """Map only outcome-free clinical fields supported by the existing builder."""
    design = scenario['normalized_design']
    if str(design.get('species', '')).casefold() != 'human':
        raise ValueError('This frozen clinical cohort requires explicit human species')
    dose = design.get('dose') or {}
    regimen = design.get('regimen') or {}
    population = design.get('population') or {}
    outputs = sorted({c.get('endpoint') or c.get('paper_target') for c in design.get('endpoint_contracts') or []
                      if c.get('endpoint') or c.get('paper_target')})
    times = regimen.get('dose_times_h')
    frequency = dose.get('frequency') or ('single_dose' if isinstance(times, list) and len(times) == 1 else
                                         'multiple_dose_schedule' if isinstance(times, list) and len(times) > 1 else 'unknown_frequency')
    return {'task_id': scenario['scenario_id'], 'active_moieties': [scenario['compound']],
            'species': design['species'], 'required_outputs': outputs,
            'scenario': {'compound': scenario['compound'], 'route': design.get('route') or dose.get('route'),
                         'formulation': dose.get('formulation') or design.get('formulation'),
                         'dose_value': dose.get('value'), 'dose_unit': dose.get('unit'), 'dose_basis': dose.get('basis'),
                         'frequency': frequency, 'start_time_value': min(times) if times else None,
                         'end_time_value': max(times) if times else None, 'start_time_unit': 'h', 'end_time_unit': 'h',
                         'infusion_duration_value': regimen.get('infusion_duration_h'), 'infusion_duration_unit': 'h',
                         'population': population.get('population') or ('healthy' if population.get('healthy') is True else None),
                         'sex': population.get('sex')}}


def metadata_match(candidate, records):
    """A matching name/family cannot replace the exact original-source hash."""
    source = candidate['provenance']['source_sha256']
    matches = [(r, ref) for r, ref in records if source in {r.get('project_sha256'), r.get('source_snapshot_sha256')}]
    if len(matches) != 1:
        return None, None, 'metadata_missing' if not matches else 'metadata_ambiguous'
    record, ref = matches[0]
    model = deepcopy(record)
    model['model_version_id'] = candidate['candidate_id']
    return model, ref, None


def stable_model_metadata(record):
    """Choose a fixed representative for case-equivalent catalog strings.

    The existing builder sorts sets by casefold only, so equal keys otherwise
    depend on the interpreter hash seed. Its semantic keys and use-rule node
    matching already ignore case; only presentation/graph hashing is fixed.
    """
    value=deepcopy(record)
    for key, items in value.items():
        if isinstance(items,list) and all(isinstance(x,str) for x in items):
            groups={}
            for item in sorted(items):
                groups.setdefault(item.casefold(),item)
            value[key]=[groups[k] for k in sorted(groups)]
    return value


def prepare_declared_tiers():
    scenarios_path = OUT / 'inputs/scenarios.jsonl'
    scenarios, candidates = rows(scenarios_path), rows(CATALOG)
    if len({r['scenario_id'] for r in scenarios}) != len(scenarios) or len({r['candidate_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate scenario/candidate identity')
    records = [(r, {'file': path, 'row_1based': i}) for path in METADATA for i, r in enumerate(rows(path), 1)]
    models, metadata_refs, gaps, source_refs = {}, {}, {}, {}
    for candidate in candidates:
        cid = candidate['candidate_id']
        prov = candidate['provenance']
        asset = reference(prov['source_path'])
        if asset['sha256'] != prov['source_sha256']:
            raise ValueError('Frozen candidate original source changed: ' + cid)
        source_refs[cid] = asset
        record, ref, reason = metadata_match(candidate, records)
        metadata_refs[cid] = ref
        if reason:
            gaps[cid] = reason
        else:
            models[cid] = build_model_graph(stable_model_metadata(record))
    output, query_refs = [], []
    for scenario in scenarios:
        task = clinical_task(scenario)
        query = build_execution_requirement_graph(task)
        query_refs.append({'scenario_id': scenario['scenario_id'], 'scenario_sha256': digest(scenario),
            'mapped_task': task, 'graph_sha256': query.checksum,
            'endpoint_categories': endpoint_categories(task['required_outputs'])})
        for candidate in candidates:
            cid = candidate['candidate_id']
            row = {'scenario_id': scenario['scenario_id'], 'candidate_id': cid,
                'prediction_kind': 'existing_engineering_use_decision',
                'query_graph_sha256': query.checksum, 'candidate_record_sha256': digest(candidate),
                'source_sha256': candidate['provenance']['source_sha256'], 'metadata_reference': metadata_refs[cid],
                'scientific_inapplicability_asserted': False, 'class_probabilities': None}
            if cid in gaps:
                row.update(status='unknown', original_policy_class=None, declared_use='unknown', reasons=[gaps[cid]])
            else:
                model = models[cid]
                match = score_candidate(query, model)
                assessment = assess_candidate_use(query, model, match=match)
                value = deterministic_use_class(assessment)
                row.update(status='declared', original_policy_class=value, declared_use=CLASS_NAMES[value],
                    model_graph_sha256=model.checksum, ranking_score=match.score,
                    maximum_use_tier=assessment.maximum_use_tier,
                    direct_reuse_gate_passed=assessment.direct_reuse_gate_passed,
                    reasons=list(assessment.direct_reuse_gate_failures))
            output.append(row)
    result_path = OUT / 'inputs/declared_tiers.jsonl'
    frozen_write(result_path, output, jsonl=True)
    manifest = {'schema_version': VERSION, 'prediction_kind': 'existing_engineering_use_decision',
        'algorithm': 'score_candidate -> assess_candidate_use -> deterministic_use_class',
        'scorer_version': SCORER_VERSION, 'graph_schema_version': GRAPH_SCHEMA_VERSION,
        'inputs': [reference(scenarios_path), reference(CATALOG), *[reference(p) for p in METADATA]],
        'code': [reference(p) for p in CODE], 'candidate_original_sources': source_refs,
        'output': reference(result_path), 'scenarios': len(scenarios), 'candidates': len(candidates),
        'pairs': len(output), 'metadata_matched_candidates': len(models), 'metadata_gaps': gaps,
        'class_names': CLASS_NAMES, 'counts': dict(Counter(r['declared_use'] for r in output)),
        'mapped_queries': query_refs, 'trained': False, 'llm_used': False, 'sealed_outcomes_accessed': False,
        'score_threshold_added': False, 'external_rankers_assigned_classes': False,
        'metadata_case_equivalence': 'Fixed lexicographic representative for case-equivalent string values before the existing builder; no change to case-insensitive semantic keys or rule decisions.',
        'claim_boundary': 'Historical deterministic engineering/use policy; not verified clinical applicability or a calibrated classifier.',
        'known_representation_limits': [
            'Existing matching skips Regimen/Objective; numerical dose, timing, origin and sampling are not fully enforced by tier decisions.',
            'Matrix, analyte identity details, food schedule, demographic numeric distributions and salt mapping are not enforced by this graph policy.',
            'Required outputs are categorized by the existing builder; AUC windows and individual aggregation contracts are not distinguished.',
            'Historical unparsed/unexecuted/license-unresolved sources yield not-cleared actions, not scientific negative labels.',
            'These are an additional single existing policy; no three-class probabilities or F1 are fabricated for the 12 rankers.']}
    frozen_write(OUT / 'inputs/declared_tiers_manifest.json', manifest)
    return manifest


def verify_declared_tiers(*, check_original_sources=True):
    """Authenticate the declaration and its complete cross-product, no outcomes.

    The default verifies original binary/model files too. A worker may skip
    only those expensive asset rehashes after the coordinator checked them;
    metadata inputs, implementation and declared results are always checked.
    """
    manifest = json.loads(inside(OUT / 'inputs/declared_tiers_manifest.json').read_text())
    if manifest.get('schema_version') != VERSION or manifest.get('prediction_kind') != 'existing_engineering_use_decision':
        raise ValueError('Unexpected declared-use schema')
    if any(manifest.get(k) is not False for k in ('trained','llm_used','sealed_outcomes_accessed','external_rankers_assigned_classes')):
        raise ValueError('Declared-use boundary changed')
    expected_inputs = {str(inside(p).relative_to(ROOT)) for p in [OUT/'inputs/scenarios.jsonl',CATALOG,*METADATA]}
    expected_code = set(CODE)
    if {r['path'] for r in manifest['inputs']} != expected_inputs or {r['path'] for r in manifest['code']} != expected_code:
        raise ValueError('Declared-use input/code inventory differs')
    for ref in [*manifest['inputs'],*manifest['code'],manifest['output']]:
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Declared-use frozen file changed: '+ref['path'])
    if inside(manifest['output']['path']) != inside(OUT/'inputs/declared_tiers.jsonl'):
        raise ValueError('Declared-use result path differs')
    scenarios, candidates = rows(OUT/'inputs/scenarios.jsonl'), rows(CATALOG)
    sids, cids = {r['scenario_id'] for r in scenarios}, {r['candidate_id'] for r in candidates}
    expected = {(sid,cid) for sid in sids for cid in cids}
    declared = rows(manifest['output']['path'])
    keys = [(r['scenario_id'],r['candidate_id']) for r in declared]
    if (len(sids)!=len(scenarios) or len(cids)!=len(candidates) or len(set(keys))!=len(keys)
            or set(keys)!=expected or manifest['pairs']!=len(declared)
            or manifest['scenarios']!=len(scenarios) or manifest['candidates']!=len(candidates)):
        raise ValueError('Declared-use scenario/candidate cross-product incomplete')
    sources = manifest['candidate_original_sources']
    if set(sources)!=cids:
        raise ValueError('Declared-use original source inventory differs')
    for candidate in candidates:
        ref = sources[candidate['candidate_id']]
        prov = candidate['provenance']
        if ref['sha256']!=prov['source_sha256'] or inside(ref['path'])!=inside(prov['source_path']):
            raise ValueError('Declared-use source/candidate binding differs')
        if check_original_sources and sha(ref['path'])!=ref['sha256']:
            raise ValueError('Declared-use original model source changed')
    for row in declared:
        if row.get('prediction_kind')!='existing_engineering_use_decision' or row.get('scientific_inapplicability_asserted') is not False:
            raise ValueError('Declared-use interpretation differs')
        value=row.get('original_policy_class')
        if ((value is None and not (row['status']=='unknown' and row['declared_use']=='unknown')) or
                (value is not None and (value not in CLASS_NAMES or row['declared_use']!=CLASS_NAMES[value]))):
            raise ValueError('Declared-use class mapping differs')
    return {'manifest':manifest,'rows':declared}


if __name__ == '__main__':
    result = prepare_declared_tiers()
    print(json.dumps({k: result[k] for k in ('pairs', 'metadata_matched_candidates', 'counts')}, ensure_ascii=False))
