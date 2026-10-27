"""Frozen clinical-condition inputs and isolated reuse of the 12 ranking adapters.

No PK answers, previous quality labels, fitting, external API, or shared service.
The old experiment's files and entry points are not modified.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_frontier_comparison as frontier

OLD = ROOT / 'experiments/model_recommendation/frontier_comparison_20260914'
OUT = ROOT / 'experiments/model_recommendation/execution_validation_20261001'
RANK = OUT / 'rankings'
RUN = ROOT / 'runtime/deployment/recommendation_execution'
METHODS = tuple(frontier.METHODS)


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT) or p == ROOT:
        raise ValueError('Resource must remain inside this project')
    return p


def read(path):
    return json.loads(inside(path).read_text())


def rows(path):
    return [json.loads(line) for line in inside(path).read_text().splitlines() if line.strip()]


def sha(path):
    with inside(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def frozen_write(path, value, *, jsonl=False):
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (''.join(json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for x in value)
               if jsonl else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    if p.exists():
        if p.read_text() != payload:
            raise ValueError('Frozen input changed: ' + str(p.relative_to(ROOT)))
    else:
        with p.open('x') as handle:
            handle.write(payload)


def metadata_value(value):
    """Only primitive clinical values: nested objects cannot smuggle answer fields."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [metadata_value(v) for v in value]
    raise ValueError('Clinical inference fields require primitive metadata, not nested result objects')


def build_query(scenario):
    """Only clinical conditions enter model text; citation identifiers stay out."""
    design = scenario['normalized_design']
    dose = design.get('dose') or {}
    population = design.get('population') or {}
    endpoints = sorted({x.get('endpoint') or x.get('paper_target') for x in design.get('endpoint_contracts') or []
                        if x.get('endpoint') or x.get('paper_target')})
    graph_endpoints = sorted({('auc' if str(e).casefold().startswith('auc') else
                              'cmax' if str(e).casefold().startswith('cmax') else
                              'concentration_time' if 'concentration' in str(e).casefold() else
                              str(e).casefold()) for e in endpoints})
    populations = []
    if population.get('healthy') is True:
        populations.append('healthy')
    if str(population.get('sex', '')).lower() in {'male', 'female'}:
        populations.append(population['sex'].lower())
    route = design.get('route') or dose.get('route') or 'unspecified'
    features = dict(species=['human'], routes=[route], populations=populations,
                    endpoints=graph_endpoints, enzymes=[], transporters=[], process_types=[])
    # An allowlist excludes source_cell, source tables, output values and reviews.
    safe_dose = {k: dose[k] for k in ('value', 'unit', 'route', 'formulation', 'administration_times_h',
                 'infusion_duration_h', 'coadministered_drugs') if k in dose}
    safe_dose['route'] = route
    safe_dose['formulation'] = design.get('formulation') or dose.get('formulation') or 'unspecified'
    if isinstance(safe_dose.get('coadministered_drugs'), list):
        normalized = []
        for item in safe_dose['coadministered_drugs']:
            if isinstance(item, dict):
                if set(item) - {'compound', 'drug', 'name', 'dose', 'value', 'dose_mg', 'unit', 'dose_unit'}:
                    raise ValueError('Unexpected field in clinical coadministration metadata')
                for val in item.values():
                    metadata_value(val)
                normalized.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
            else:
                normalized.append(metadata_value(item))
        safe_dose['coadministered_drugs'] = normalized
    regimen = design.get('regimen') or {}
    safe_regimen = {k: regimen[k] for k in ('dose_times_h', 'infusion_duration_h', 'observation_origin_h') if k in regimen}
    sampling_times = (design.get('sampling') or {}).get('times_h')
    safe_population = {k: population[k] for k in ('healthy', 'sex', 'ethnicity', 'age_mean_year',
                       'age_range_year', 'weight_mean_kg', 'weight_range_kg', 'height_mean_cm',
                       'bmi_range', 'n', 'n_evaluable') if k in population}
    for fields in (safe_dose, safe_regimen, safe_population):
        for key, val in fields.items():
            metadata_value(val)
    metadata_value(sampling_times)
    food = design.get('food')
    if isinstance(food, dict):
        food = food.get('status') or food.get('description') or 'unspecified'
    metadata_value(food)
    lines = [f"Find PBPK model candidates for {scenario['compound']}.",
             'Intended task: predict pharmacokinetics under the following clinical conditions without fitting target observations.',
             'Species context: human.',
             'Administration: ' + json.dumps(safe_dose, ensure_ascii=False, sort_keys=True),
             'Regimen and observation origin: ' + json.dumps(safe_regimen, ensure_ascii=False, sort_keys=True),
             'Nominal sampling hours: ' + json.dumps(sampling_times),
             'Population context: ' + json.dumps(safe_population, ensure_ascii=False, sort_keys=True),
             'Food context: ' + str(food or 'unspecified'),
             'Matrix: ' + str(design.get('matrix') or 'unspecified'),
             'Analyte: ' + str(design.get('analyte') or 'unspecified'),
             'Target endpoints: ' + ', '.join(endpoints or ['plasma concentration time profile']),
             'Study outcomes and prediction errors are unavailable to the recommender.']
    text = '\n'.join(lines)
    qid = scenario['scenario_id']
    node_root = qid + ':root'
    nodes = [{'node_id': node_root, 'node_type': 'ModelingRequest',
              'label': 'PBPK modeling request', 'required': False}]
    edges = []
    terms = [('Compound', [scenario['compound']]), ('Species', features['species']),
             ('Route', features['routes']), ('Population', populations), ('Endpoint', graph_endpoints),
             ('Objective', ['plasma_concentration_time_prediction'])]
    for kind, labels in terms:
        for i, label in enumerate(labels):
            nid = f'{qid}:{kind}:{i}'
            nodes.append(dict(node_id=nid, node_type=kind, label=label, required=kind == 'Compound'))
            edges.append(dict(source_id=node_root, target_id=nid, predicate='REQUIRES_' + kind.upper(),
                              required=kind == 'Compound'))
    return dict(query_id=qid, text=text, compounds=[scenario['compound']],
                task_type='plasma_concentration_time_prediction', features=features,
                graph=dict(graph_id=qid, nodes=nodes, edges=edges), variants={}, input_evidence_ids=[],
                request_origin='clinical_protocol_conditions_only',
                graph_limitation='Existing graph schema represents drug/species/route/population/endpoints; numerical regimen remains in common text.')


def prepare_inputs():
    scenarios = rows(OUT / 'inputs/scenarios.jsonl')
    candidates = rows(OLD / 'inputs/candidates.jsonl')
    queries = [build_query(s) for s in scenarios]
    if not queries or len(candidates) != 280 or len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('Invalid clinical input coverage')
    # Raw literature and sealed references cannot reach text/graph inference.
    identifiers = set()
    for s in scenarios:
        def visit(v):
            if isinstance(v, dict):
                for val in v.values():
                    visit(val)
            elif isinstance(v, list):
                for val in v:
                    visit(val)
            elif isinstance(v, str) and (v.upper().startswith(('PMC', 'NCT')) or v.startswith('10.')):
                identifiers.add(v.casefold())
        visit(s.get('publication_ids', {}))
    for c in candidates:
        text = json.dumps({k: c[k] for k in ('family_name', 'text', 'graph')}, ensure_ascii=False).casefold()
        if any(re.search(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', text) for term in identifiers):
            raise ValueError('Target publication identity found in candidate inference fields')
    for name, data in [('queries.jsonl', queries), ('candidates.jsonl', candidates)]:
        frozen_write(RANK / 'inputs' / name, data, jsonl=True)
    manifest = dict(schema_version='pbpk-clinical-ranking-inputs-v1', queries=len(queries), candidates=len(candidates),
                    variants=['original'], target_outcomes_accessed=False,
                    source_files={str(p.relative_to(ROOT)): sha(p) for p in
                                  [OUT / 'inputs/scenarios.jsonl', OLD / 'inputs/candidates.jsonl', Path(__file__)]},
                    outputs={name: {'sha256': sha(RANK / 'inputs' / name)}
                             for name in ['queries.jsonl', 'candidates.jsonl']},
                    differences_from_prior_batch='New explicit clinical requests; same masked catalog, no old scores reused.')
    manifest_path = RANK / 'inputs/manifest.json'
    if manifest_path.exists() and 'preparation_code_revisions' in read(manifest_path):
        manifest['preparation_code_revisions'] = read(manifest_path)['preparation_code_revisions']
    if manifest_path.exists() and read(manifest_path) != manifest and not (OUT / 'preparation.json').exists():
        previous = read(manifest_path)
        if previous.get('outputs') != manifest['outputs']:
            raise ValueError('Prepared clinical content changed; explicit new input version required')
        # Only preparation-code provenance may change during implementation,
        # before any official ranking/prediction run or preparation freeze.
        if any((RANK / 'results' / m / 'status.json').exists() for m in METHODS):
            raise ValueError('Ranking has started; preserve its original manifest')
        for key, value in previous.items():
            if key not in {'source_files', 'preparation_code_revisions'} and manifest.get(key) != value:
                raise ValueError('Prepared manifest semantics changed')
        manifest['preparation_code_revisions'] = previous.get('preparation_code_revisions', []) + [
            {'previous_manifest_sha256': sha(manifest_path), 'previous_source_files': previous['source_files']}]
        frontier.write_json(manifest_path, manifest)
    else:
        # Preserve the pre-freeze change receipt on later deterministic calls.
        if manifest_path.exists() and 'preparation_code_revisions' in read(manifest_path):
            manifest['preparation_code_revisions'] = read(manifest_path)['preparation_code_revisions']
        if manifest_path.exists():
            # Preparation amendments use the existing atomic JSON writer, whose
            # key order differs from frozen_write. Equal content must preserve
            # the original bytes/SHA; do not rewrite a frozen manifest merely
            # to change formatting.
            if read(manifest_path) != manifest:
                raise ValueError('Frozen ranking manifest changed')
        else:
            frozen_write(manifest_path, manifest)
    return manifest


def load_inputs():
    q, c = rows(RANK / 'inputs/queries.jsonl'), rows(RANK / 'inputs/candidates.jsonl')
    m = read(RANK / 'inputs/manifest.json')
    for name in ('queries.jsonl', 'candidates.jsonl'):
        if sha(RANK / 'inputs' / name) != m['outputs'][name]['sha256']:
            raise ValueError('Frozen ranking input changed')
    if len(q) != m['queries'] or len(c) != 280:
        raise ValueError('Unexpected ranking shape')
    return q, c, {str(p.relative_to(ROOT)): sha(p) for p in
                  [RANK / 'inputs' / n for n in ('queries.jsonl', 'candidates.jsonl', 'manifest.json')]}


def local_weights(method, repo_override=None):
    repo = repo_override or frontier.METHODS[method]['repo']
    filename = 'weights.json' if repo_override is None else 'dependency_weights_' + repo.rsplit('/', 1)[-1] + '.json'
    source = OLD / 'results' / method / filename
    # G-reasoner uses the already pinned Qwen embedding dependency through its own method.
    record = read(source)
    if record.get('repo') != repo or not record.get('revision') or not record.get('sha256'):
        raise ValueError('Missing pinned local weights')
    base = inside(record['path'])
    for name, expected in record['sha256'].items():
        if sha(base / name) != expected:
            raise ValueError('Pinned model weight changed: ' + name)
    frozen_write(RANK / 'results' / method / filename, record)
    return record


def report():
    states = {m: frontier.read_json(RANK / 'results' / m / 'status.json', {}) for m in METHODS}
    complete = [m for m, s in states.items() if s.get('state') == 'complete']
    value = dict(state='complete' if len(complete) == len(METHODS) else 'running',
                 completed_methods=complete, total_methods=len(METHODS),
                 completed_pairs=sum(s.get('completed_pairs', 0) for s in states.values()),
                 states={m: s.get('state', 'pending') for m, s in states.items()})
    frontier.write_json(RANK / 'status.json', value)
    return value


def run_method(method):
    if method not in METHODS:
        raise ValueError('Unknown frozen ranking method')
    RUN.mkdir(parents=True, exist_ok=True)
    frontier.OUT, frontier.RUN = RANK, RUN / 'ranking'
    frontier.RUN.mkdir(parents=True, exist_ok=True)
    frontier.VARIANTS = ('original',)
    frontier.load_inputs = load_inputs
    frontier.download = local_weights
    frontier.report = report
    frontier.init()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if method == 'bm25_top20_reasonrank_7b_nf4':
        import frontier_external_models as adapters
        original = adapters.load_model
        def official_budget(*args, **kwargs):
            model = original(*args, **kwargs)
            model.max_new_tokens = 3172
            return model
        adapters.load_model = official_budget
    batches = {'bm25_top20_reasonrank_7b_nf4': 4, 'bm25_top20_rearank_7b_nf4': 4,
               'bm25_top20_rank_r1_3b': 8}
    frontier.run_method(method, batch_size=32, max_length=2048, query_batch_size=batches.get(method, 1))
    # Bind the clinical wrapper too; original upstream runner binds its own sources.
    path = RANK / 'results' / method / 'status.json'
    state = read(path)
    state['clinical_wrapper_sha256'] = sha(__file__)
    state['official_reasonrank_budget'] = 3172 if method == 'bm25_top20_reasonrank_7b_nf4' else None
    frontier.write_json(path, state)


def finalize_scores():
    """Retain raw scores; apply the same documented recovery/equivalence rules."""
    import numpy as np
    import frontier_result_audit as audit
    audit.VARIANTS = ('original',)
    recovery = audit.refresh_output_recovery(RANK)
    q, c, input_hashes = load_inputs()
    expected_index = {'query_ids': [x['query_id'] for x in q], 'candidate_ids': [x['candidate_id'] for x in c],
                      'variants': ['original']}
    groups = defaultdict(list)
    for i, row in enumerate(c):
        groups[row['text']].append(i)
    interchangeable = {'qwen3_embedding_06b', 'qwen3_embedding_4b', 'f2llm_v2_06b', 'qwen3_reranker_06b'}
    effective = {}
    for method in METHODS:
        base = RANK / 'results' / method
        state = read(base / 'status.json')
        if state.get('state') != 'complete':
            raise ValueError('Ranking barrier requires every method')
        if state.get('input_sha256') != input_hashes or read(base / 'index.json') != expected_index:
            raise ValueError('Ranking identities or input hashes differ from clinical freeze')
        if state.get('clinical_wrapper_sha256') != sha(__file__):
            raise ValueError('Clinical ranking wrapper changed')
        expected_implementation = {name: sha(ROOT / 'scripts' / name) for name in
                                   ('run_frontier_comparison.py', 'frontier_external_models.py')}
        if frontier.METHODS[method]['kind'] in {'g_reasoner', 'gfm_rag'}:
            expected_implementation['frontier_graph_models.py'] = sha(ROOT / 'scripts/frontier_graph_models.py')
        if state.get('implementation_sha256') != expected_implementation:
            raise ValueError('Ranking implementation hashes changed')
        source = base / 'scores.npy'
        if method in recovery.get('recovered_scores_by_method', {}):
            source = RANK / recovery['recovered_scores_by_method'][method]['path']
        values = np.load(source, allow_pickle=False)
        if values.shape != (1, len(q), len(c)) or not np.isfinite(values).all():
            raise ValueError('Incomplete final score matrix')
        if method in interchangeable:
            values = values.copy()
            for indices in groups.values():
                values[:, :, indices] = values[:, :, indices[:1]]
            source = base / 'text_equivalent_scores.npy'
            if source.exists():
                if not np.array_equal(values, np.load(source, allow_pickle=False)):
                    raise ValueError('Derived score matrix changed')
            else:
                with source.open('xb') as handle:
                    np.save(handle, values, allow_pickle=False)
        effective[method] = {'path': str(source.relative_to(ROOT)), 'sha256': sha(source),
                             'original_scores': {'path': str((base / 'scores.npy').relative_to(ROOT)), 'sha256': sha(base / 'scores.npy')},
                             'index': {'path': str((base / 'index.json').relative_to(ROOT)), 'sha256': sha(base / 'index.json')},
                             'status': {'path': str((base / 'status.json').relative_to(ROOT)), 'sha256': sha(base / 'status.json')}}
    result = dict(schema_version='pbpk-clinical-ranking-lock-v1', state='complete', methods=effective,
                  query_ids=[x['query_id'] for x in q], candidate_ids=[x['candidate_id'] for x in c],
                  manifest={'path': str((RANK / 'inputs/manifest.json').relative_to(ROOT)), 'sha256': sha(RANK / 'inputs/manifest.json')},
                  scientific_outcomes_read=False, all_finite=True,
                  score_semantics='Ordering only; no native three-tier predictions or applicability probabilities.')
    frozen_write(RANK / 'ranking_lock.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker', 'finalize', 'status'])
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare_inputs(), ensure_ascii=False))
    elif args.action == 'worker':
        run_method(args.method)
    elif args.action == 'finalize':
        print(json.dumps(finalize_scores(), ensure_ascii=False))
    else:
        print(json.dumps(report(), ensure_ascii=False))
