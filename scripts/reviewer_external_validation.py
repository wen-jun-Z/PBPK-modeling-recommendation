"""Freeze and execute new-study rankings and native PBPK predictions.

The prediction phase does not open observation tables or article XML. Frozen
clinical Methods and the original public catalogue are the only new inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
OUT = ROOT / 'experiments/model_recommendation/reviewer_revision_20261009/external_validation'
LIGHT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
CATALOG = ROOT / 'experiments/model_recommendation/execution_validation_20261001/rankings/inputs/candidates.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve().relative_to(ROOT)), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
        return
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def status(phase, **values):
    path = OUT / 'status.json'
    value = {'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(), **values}
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
    print(json.dumps(value), flush=True)


def checked(reference):
    path = (ROOT / reference['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != reference['sha256']:
        raise ValueError('Changed source binding: ' + str(path))
    return path


def prepare():
    from rank_recommendation_execution import build_query
    from recommendation_candidate_executor import scenario_contract
    selection = read(OUT / 'selection_manifest.json')
    candidates = [json.loads(line) for line in CATALOG.read_text().splitlines()]
    scenarios, queries = [], []
    for raw in selection['selected_scenarios']:
        checked(raw['methods_source_ref'])
        scenario = deepcopy(raw)
        design = scenario['normalized_design']
        # The legacy query builder accepts structured public population metadata.
        # Do not invent age/body-weight means from ranges or alter source subjects.
        population = design['population']
        design['population'] = {'healthy': True} if 'healthy' in population.casefold() else {}
        if 'healthy men' in population.casefold():
            design['population']['sex'] = 'male'
        scenario['reported_population'] = population
        scenario['execution_policy'] = {'allow_reference_subject_approximation': True}
        scenario['case_id'] = scenario['scenario_id']
        scenario['catalog_candidate_ids'] = [c['candidate_id'] for c in candidates if scenario['compound'] in c['compounds']]
        scenario_contract(scenario)
        scenarios.append(scenario)
        queries.append(build_query(scenario))
    write_once(OUT / 'inputs/scenarios.json', scenarios)
    write_once(OUT / 'inputs/queries.json', queries)
    binding = {'schema': 'reviewer-new-study-inputs-v1', 'selection': ref(OUT / 'selection_manifest.json'),
               'master_protocol': ref(OUT.parent / 'protocol.json'), 'catalog': ref(CATALOG),
               'scenarios': ref(OUT / 'inputs/scenarios.json'), 'queries': ref(OUT / 'inputs/queries.json'),
               'population_mapping': 'Only explicit healthy/male fields enter the unchanged query builder; no target individual is reconstructed.',
               'methods_sampling_discrepancies': 'Retained in each scenario; no invented fixed-window observation.',
               'native_execution': 'Every same-parent candidate; original mechanisms and parameters; reference-subject approximation.',
               'outcomes_opened': False, 'producer': ref(Path(__file__))}
    write_once(OUT / 'inputs/manifest.json', binding)
    return scenarios, queries, candidates


def document_head(vector, features, embeddings, directory):
    """Load only the frozen document/applicability heads; no error labels/head."""
    import torch
    from pbpk_match_light_training import LightMatch, _normalized, _canonical
    directory = Path(directory)
    meta = read(directory / 'metadata.json')
    selection = read(directory.parent / 'selection.json')
    state = torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True)
    model = LightMatch(meta['embedding_size'], len(meta['feature_mean']), meta['width'], meta['dropout']).eval()
    model.load_state_dict(state['heads'])
    x = _normalized(features, meta)
    order, restore = _canonical(embeddings, x)
    with torch.inference_mode():
        output = model(torch.as_tensor(vector), torch.as_tensor(embeddings[order]), torch.as_tensor(x[order]))
    scores = output['scores'].numpy()[restore]
    logits = output['app_logits'].numpy()[restore] / selection['calibration']['temperature']
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    return scores, probs


def rank(scenarios, queries, candidates, device):
    from pbpk_match_light_inputs import load_cache, _encode_missing
    from run_component_model_integration import verify_baseline
    from pbpk_kb.recommendation.pbpk_match_v2.data import pair_feature_row, feature_names
    from reviewer_controls_models import identity_condition_rule, visible_graph
    from pbpk_kb.recommendation.model_matching.scoring import score_candidate
    from run_frontier_comparison import BM25
    import torch
    torch.set_num_threads(4)
    protocol = verify_baseline()
    final = OUT / 'rankings/manifest.json'
    if final.exists():
        receipt = read(final)
        for source in receipt['sources'] + [receipt['predictions']]:
            checked(source)
        return receipt
    cache = load_cache()
    ids = [c['candidate_id'] for c in candidates]
    if ids != protocol['candidate_ids'] or ids != [c['candidate_id'] for c in cache['candidate_records']]:
        raise ValueError('Catalogue identity/order changed')
    texts = [q['text'] for q in queries]
    embeddings = cache['light_candidate_embeddings'].astype(np.float32)
    vectors = np.zeros((len(texts), embeddings.shape[-1]), np.float16)
    completed = np.zeros(len(texts), bool)
    (OUT / 'cache').mkdir(exist_ok=True)
    _encode_missing(texts, list(range(len(texts))), vectors, completed, OUT / 'cache',
                    {'encoder': cache['binding']['encoder']}, device=device, batch_size=8,
                    callback=lambda event: status('encoding', event=event))
    if not completed.all():
        raise ValueError('Incomplete query embeddings')
    vectors = vectors.astype(np.float32)
    features = np.asarray([[pair_feature_row(q, c, design=s['normalized_design']) for c in candidates]
                           for s, q in zip(scenarios, queries)], np.float32)
    bm25 = BM25([c['text'] for c in candidates])
    outputs = {'bm25_text': np.asarray([bm25.score_query(q['text'], None) for q in queries]),
               'qwen3_embedding_06b': vectors @ embeddings.T,
               'identity_condition_rule': identity_condition_rule(features, feature_names())}
    graphs = [visible_graph(c, candidate=True) for c in candidates]
    outputs['service_typed_rule'] = np.asarray([[score_candidate(visible_graph(q), c).score for c in graphs] for q in queries], np.float32)
    own, probabilities, used_folds = [], [], []
    original = cache['query_records']
    for i, (s, q) in enumerate(zip(scenarios, queries)):
        matched = [j for j, r in enumerate(original) if s['compound'] in r['compounds']]
        folds = [j for j, fold in enumerate(protocol['folds']) if set(matched) & set(fold['outer_test'])]
        if matched and (len(folds) != 1 or not set(matched) <= set(protocol['folds'][folds[0]]['outer_test'])):
            raise ValueError('Target compound was not isolated to one frozen outer group')
        folds = folds if matched else list(range(5))
        results = [document_head(vectors[i], features[i], embeddings, LIGHT / f'runs/seed17/fold{f}/outer') for f in folds]
        if len(results) == 1:
            scores, probs = results[0]
        else:
            # Rank-based combination avoids mixing differently scaled fold logits.
            from scipy.stats import rankdata
            scores = np.mean([rankdata(r[0], method='average') / len(ids) for r in results], axis=0)
            probs = np.mean([r[1] for r in results], axis=0)
        own.append(scores); probabilities.append(probs)
        used_folds.append({'query_id': q['query_id'], 'folds': folds, 'same_compound_training_records_excluded': matched})
    outputs['pbpk_match_light_document'] = np.asarray(own, np.float32)
    if any(a.shape != (len(queries), 280) or not np.isfinite(a).all() for a in outputs.values()):
        raise ValueError('Every method must retain complete finite 280-candidate scores')
    (OUT / 'rankings').mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'rankings/scores.npz', **outputs, probabilities=np.asarray(probabilities))
    sources = [ref(OUT / 'inputs/manifest.json'), ref(LIGHT / 'protocol.json'), ref(LIGHT / 'predictions/seed17.json'),
               ref(LIGHT / 'cache/manifest.json'), ref(LIGHT / 'cache/binding.json'), ref(Path(__file__)),
               ref(ROOT / 'scripts/reviewer_controls_models.py')]
    for f in range(5):
        for name in ('heads.pt', 'metadata.json', 'complete.json'):
            sources.append(ref(LIGHT / f'runs/seed17/fold{f}/outer' / name))
    receipt = {'schema': 'reviewer-new-study-rankings-v1', 'state': 'locked',
               'locked_at': datetime.now(timezone.utc).isoformat(), 'query_ids': [q['query_id'] for q in queries],
               'candidate_ids': ids, 'methods': list(outputs), 'fold_assignments': used_folds,
               'sources': sources, 'predictions': ref(OUT / 'rankings/scores.npz'),
               'clinical_outcome_inputs': False, 'new_training': False, 'error_correction': False, 'components': False}
    write_once(final, receipt)
    return receipt


def native(scenarios, candidates):
    from recommendation_candidate_executor import execute_candidate
    lookup = {c['candidate_id']: c for c in candidates}
    jobs = [(s, lookup[cid]) for s in scenarios for cid in s['catalog_candidate_ids']]
    results = []
    def one(s, c):
        key = hashlib.sha256((s['scenario_id'] + '|' + c['candidate_id']).encode()).hexdigest()[:20]
        output = OUT / 'native' / key
        if (output / 'result.json').exists():
            result = read(output / 'result.json')
            plan = read(output / 'execution_plan.json')
            from recommendation_candidate_executor import _digest, implementation_hashes
            if plan['scenario_sha256'] != _digest(s) or plan['candidate_sha256'] != _digest(c) or plan['implementation_sha256'] != implementation_hashes():
                raise ValueError('Resumed native job has changed inputs')
            if result.get('prediction_locked'):
                checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']})
        else:
            result = execute_candidate(s, c, output, project_root=ROOT, timeout_seconds=300)
        return {'scenario_id': s['scenario_id'], 'candidate_id': c['candidate_id'], 'result': ref(output / 'result.json'),
                'status': result['status'], 'reasons': result.get('reasons', [])}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, s, c) for s, c in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            status('native_execution', completed=len(results), total=len(jobs), last_status=results[-1]['status'])
    results.sort(key=lambda r: (r['scenario_id'], r['candidate_id']))
    receipt = {'schema': 'reviewer-new-study-native-v1', 'tasks': results, 'complete': True,
               'observations_opened': False, 'parameter_refitting': False,
               'input_manifest': ref(OUT / 'inputs/manifest.json')}
    write_once(OUT / 'native/manifest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'rank', 'native', 'predict'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    scenarios, queries, candidates = prepare()
    if args.action in ('rank', 'predict'):
        rank(scenarios, queries, candidates, args.device)
    if args.action in ('native', 'predict'):
        result = native(scenarios, candidates)
        status('predictions_complete', tasks=len(result['tasks']),
               locked=sum(r['status'] == 'prediction_locked' for r in result['tasks']))


if __name__ == '__main__':
    main()
