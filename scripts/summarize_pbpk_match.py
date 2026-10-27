"""Summarize every completed development configuration, without selecting test results."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from pbpk_kb.recommendation.pbpk_match.data import OUTPUT, build_data, sha, write_json, scoped
from pbpk_kb.recommendation.pbpk_match.diagnostics import compare_reports_paired, summarize_grade_support
from pbpk_kb.recommendation.pbpk_match.evaluation import compare_stability
from run_pbpk_match import evaluate, baseline_reports


def percentile_ensemble(score_arrays):
    """Equal-weight full-catalog ranks, preserving exact half-integer rank sums."""
    arrays = [np.asarray(values, dtype=float) for values in score_arrays]
    if (not arrays or arrays[0].ndim != 3 or arrays[0].shape[0] != 3
            or not arrays[0].shape[1] or not arrays[0].shape[2]):
        raise ValueError('Ensemble requires nonempty [3 variants,queries,candidates] arrays')
    if any(values.shape != arrays[0].shape or not np.isfinite(values).all() for values in arrays):
        raise ValueError('Ensemble scores must be finite and share the same shape')
    # Average ranks are integers or half-integers, exactly represented in float.
    # Dividing each seed first can break mathematically identical rank-sum ties.
    ranks = np.stack([rankdata(values, method='average', axis=-1) for values in arrays])
    return ranks.sum(axis=0) / (len(arrays) * arrays[0].shape[-1])


def seed_completion(expected_seeds, completed_seeds):
    """Report progress against the frozen seed list, never a lucky subset."""
    expected, completed = list(expected_seeds), list(completed_seeds)
    if (not expected or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in expected + completed)
            or len(set(expected)) != len(expected) or len(set(completed)) != len(completed)):
        raise ValueError('Protocol and completed seeds must be unique integer IDs')
    if not set(completed) <= set(expected):
        raise ValueError('Completed seed was not declared in the frozen protocol')
    missing = [seed for seed in expected if seed not in completed]
    return {'state': 'complete' if not missing else ('partial' if completed else 'not_started'),
            'expected_seeds': expected, 'completed_seeds': completed, 'missing_seeds': missing,
            'expected_seed_count': len(expected), 'completed_seed_count': len(completed)}


def _prediction_arrays(predictions, data):
    scores = np.asarray(predictions['scores'])
    probabilities = np.asarray(predictions['probabilities'])
    shape = (len(data['query_records']), len(data['candidate_records']))
    if scores.shape != (3, *shape) or not np.isfinite(scores).all():
        raise ValueError('Saved scores require all three finite, complete frozen query/candidate matrices')
    if (probabilities.shape != (*shape, 3) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-7)):
        raise ValueError('Saved probabilities require normalized finite [query,candidate,YNU] values')
    return scores, probabilities


def _seed_metric(values):
    defined = [float(value) for value in values if value is not None]
    if any(not np.isfinite(value) for value in defined):
        raise ValueError('Seed metrics cannot contain nonfinite values')
    return {'mean': float(np.mean(defined)) if defined else None,
            'seed_sd': float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
            'defined_seed_count': len(defined), 'undefined_seed_count': len(values) - len(defined)}


def summarize(output=OUTPUT):
    output=scoped(output)
    data=build_data(output,device='cpu')
    bases=baseline_reports(data,output)
    reports={}; groups=defaultdict(list); declared_groups={}; incomplete=[]; source_files={}
    changed=[q['variants']['missing_evidence']['text_changed'] for q in data['query_records']]
    for run in sorted((output/'runs').iterdir()):
        if not (run/'protocol.json').exists():continue
        protocol=json.loads((run/'protocol.json').read_text())
        if protocol['snapshot_sha256']!=sha(output/'development_snapshot.json'):
            raise ValueError('Cannot compare different label snapshots')
        seed_completion(protocol['seeds'], [])
        if not protocol['methods'] or len(set(protocol['methods'])) != len(protocol['methods']):
            raise ValueError('Frozen methods must be nonempty and unique')
        source_files[str((run/'protocol.json').relative_to(ROOT))] = sha(run/'protocol.json')
        for method in protocol['methods']:
            declared_groups[(run.name, method)] = protocol['seeds']
            for seed in protocol['seeds']:
                key=f'{run.name}/{method}_seed{seed}'
                source=run/f'{method}_seed{seed}'
                if not (source/'summary.json').exists() or not (source/'out_of_fold_predictions.npz').exists():
                    incomplete.append(key); continue
                report=json.loads((source/'summary.json').read_text())
                if report.get('fold_count') != 3 or report.get('method') != method or report.get('seed') != seed:
                    raise ValueError('Saved summary lacks three completed folds or mismatches method/seed')
                with np.load(source/'out_of_fold_predictions.npz',allow_pickle=False) as pred:
                    scores, probabilities = _prediction_arrays(pred, data)
                # Re-evaluation binds identical judgment pools and adds risk summaries.
                values=evaluate(data,scores[0],probabilities)
                report.update(values)
                report['run']=run.name
                report['configuration_state']='complete'
                report['result_origin']='saved out-of-fold predictions; no fitting in summary'
                report['legacy_numerical_tie_issue']=run.name=='pilot_v1'
                report['paired_against_jina']=compare_reports_paired(report,bases['jina_reranker_v35'])
                reports[key]=report
                groups[(run.name,method)].append((seed,report,scores,probabilities))
                source_files[str((source/'out_of_fold_predictions.npz').relative_to(ROOT))]=sha(source/'out_of_fold_predictions.npz')
    grouped={}; ensembles={}
    for (run,method), expected_seeds in declared_groups.items():
        rows = groups[(run, method)]
        completion = seed_completion(expected_seeds, [row[0] for row in rows])
        metrics={}
        for name in ['ndcg@10','recall@10','mrr']:
            vals=[r[1]['judged_only']['metrics'][name] for r in rows]
            metrics[name]=_seed_metric(vals)
        grouped[f'{run}/{method}']={**completion, 'seeds':[r[0] for r in rows], 'metrics':metrics,
             'interpretation':'Variability across training seeds on the same development split, not a confidence interval'}
        if completion['state'] == 'complete' and len(rows)>=3:
            # Equal-weight percentile ranks avoid mixing arbitrary neural score scales.
            score=percentile_ensemble([row[2] for row in rows])
            p=np.mean([r[3] for r in rows],axis=0);p=p/p.sum(-1,keepdims=True)
            rep=evaluate(data,score[0],p)
            rep['method']=method;rep['run']=run;rep['seeds']=[r[0] for r in rows]
            rep['seed_completion']=completion
            rep['ensemble_rule']='equal-weight average full-catalog percentile ranks; exact within-seed ties preserved'
            rep['inference_cost_multiplier']=len(rows)
            rep['stability']={'reordered':compare_stability(score[0],score[1]),
                'missing_evidence':compare_stability(score[0],score[2],'missing_evidence',changed)}
            rep['calibration_status']='uncalibrated probability ensemble'
            rep['paired_external_comparisons']={name:compare_reports_paired(rep,base) for name,base in bases.items()}
            ensembles[f'{run}/{method}']=rep
    supervised_comparisons = {}
    for key, report in ensembles.items():
        baseline_key = f"{report['run']}/mlp"
        baseline = ensembles.get(baseline_key)
        if report['method'] == 'mlp' or baseline is None:
            continue
        if report['seeds'] != baseline['seeds']:
            raise ValueError('Within-run supervised comparison requires identical declared seeds')
        supervised_comparisons[key] = {
            'baseline': baseline_key,
            'scope': 'Same run, frozen labels, splits, training configuration and declared seeds; both ensembles complete',
            'raw_ranking': compare_reports_paired(report, baseline),
            'interpretation': 'Exploratory development comparison; not independent confirmation after architecture selection',
        }
    result={'scope':'partial-primary-judgment development; no locked test quality evaluation',
        'completion_state':'partial' if incomplete else 'complete',
        'scientific_gold':False,
        'grade_support':summarize_grade_support(data['grades'],data['query_records']),
        'incomplete_configurations':incomplete,'result_hashes':source_files,
        'completed':reports,'seed_summaries':grouped,'ensembles':ensembles,
        'paired_supervised_comparisons': supervised_comparisons,
        'external_baselines':bases,'selection_warning':'All configurations reported; any choice is development selection, not independent confirmation'}
    write_json(output/'results.json',result)
    rows=[]
    for category,collection in [('external',bases),('single',reports),('ensemble',ensembles)]:
        for key,r in collection.items():
            met=r['judged_only']['metrics']; app=r.get('applicability',{})
            policy=r.get('shared_identity_policy',{}).get('metrics',{})
            stability=r.get('stability',{})
            rows.append({'category':category,'method':key,'eligible_queries':r['judged_only']['eligible_queries'],
                **met,'known_grade_fraction':r['coverage']['known_grade_fraction'],
                'top1_unknown_fraction':r['coverage']['full_catalog_top1_unknown_rate'],
                'applicability_macro_f1':app.get('macro_f1'),
                'applicability_Y_f1':app.get('per_class_f1',{}).get('Y'),
                'brier_known_YN':app.get('brier'),'ece_known_YN':app.get('ece'),
                'same_drug_registered_family_pair_accuracy':r.get('same_drug_different_family_preference',{}).get('query_macro_accuracy'),
                'reordered_top1_jaccard':stability.get('reordered',{}).get('top1_set_jaccard'),
                'missing_evidence_top1_jaccard':stability.get('missing_evidence',{}).get('top1_set_jaccard'),
                'shared_identity_policy_ndcg10':policy.get('ndcg@10'),
                'shared_identity_policy_recall10':policy.get('recall@10'),
                'shared_identity_policy_mrr':policy.get('mrr'),
                'legacy_numerical_tie_issue':r.get('legacy_numerical_tie_issue',False)})
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'completed_configurations':len(reports),'ensembles':len(ensembles),'incomplete':incomplete},ensure_ascii=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    summarize(parser.parse_args().output)
