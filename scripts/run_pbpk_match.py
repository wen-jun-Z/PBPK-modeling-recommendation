"""Train and audit PBPK-Match on development compounds only; never unlock test labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np

from pbpk_kb.recommendation.pbpk_match.data import (
    CORE, OUTPUT, VARIANTS, build_data, fitting_indices, scoped, sha, write_json,
)
from pbpk_kb.recommendation.pbpk_match.evaluation import (
    evaluate_ranking, compare_stability, fit_probability_calibrator,
    apply_probability_calibrator,
)
from pbpk_kb.recommendation.pbpk_match.diagnostics import (
    bind_label_pools, compare_reports_paired, risk_coverage_from_predictions,
    summarize_grade_support,
)
from pbpk_kb.recommendation.pbpk_match.policy_evaluation import evaluate_identity_policy


def probability_normalize(values):
    values = np.asarray(values, np.float64)
    values = np.maximum(values, 0)
    return values / np.maximum(values.sum(-1, keepdims=True), 1e-20)


def compound_summary(report):
    grouped = defaultdict(list)
    for row in report["per_query"]:
        if row["judged_only"] is not None:
            grouped[row["compound_id"]].append(row["judged_only"])
    metrics = {}
    for name in ("ndcg@10", "recall@10", "mrr"):
        values = [np.mean([r[name] for r in rows]) for rows in grouped.values()]
        metrics[name] = float(np.mean(values)) if values else None
    report["compound_macro_judged_only"] = {"eligible_compounds": len(grouped), "metrics": metrics}
    return report


def evaluate(data, scores, probabilities=None, indices=None):
    ids = list(range(len(data["query_records"]))) if indices is None else list(indices)
    report = evaluate_ranking(scores, data["grades"][ids], data["applicability"][ids],
                 [data["query_records"][i] for i in ids], data["candidate_records"], probabilities)
    report = bind_label_pools(report, data['grades'][ids],
        [data['query_records'][i] for i in ids], data['candidate_records'])
    report['shared_identity_policy'] = evaluate_identity_policy(scores,data['grades'][ids],
        data['applicability'][ids],[data['query_records'][i] for i in ids],data['candidate_records'])
    if probabilities is not None:
        report['selective_risk'] = risk_coverage_from_predictions(scores, probabilities, data['applicability'][ids])
        app = report['applicability']
        matrix = app.get('confusion_true_rows_predicted_columns')
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            ratio = lambda a,b: float(a/b) if b else None
            app['decision_rates'] = {
                'false_recommendation_among_documented_N':ratio(matrix[1,0],matrix[1].sum()),
                'explicit_false_rejection_among_documented_Y':ratio(matrix[0,1],matrix[0].sum()),
                'abstention_among_documented_Y':ratio(matrix[0,2],matrix[0].sum()),
                'predicted_Y_fraction_of_valid_pairs':ratio(matrix[:,0].sum(),matrix.sum()),
                'predicted_U_fraction_of_valid_pairs':ratio(matrix[:,2].sum(),matrix.sum()),
                'scope':'Classification on valid automatic Y/N/U labels; distinct from full-catalog request recommendation coverage',
            }
    return compound_summary(report)


def fold_partition(data, fold):
    lookup = data["snapshot"]["fold_by_compound"]
    q = data["query_records"]
    outer = [i for i,r in enumerate(q) if lookup[r["compound_id"]] == fold]
    remaining = [cid for cid in data["snapshot"]["development_compounds"] if lookup[cid] != fold]
    # Chosen only by frozen identifiers, never by labels or observed performance.
    calibration_cid, inner_val_cid = remaining[0], remaining[1]
    calibration = [i for i,r in enumerate(q) if r["compound_id"] == calibration_cid]
    validation = [i for i,r in enumerate(q) if r["compound_id"] == inner_val_cid]
    train = [i for i,r in enumerate(q) if r["compound_id"] in remaining[2:]]
    calibration = fitting_indices(data, calibration, outer)
    validation = fitting_indices(data, validation, outer + calibration)
    train = fitting_indices(data, train, outer + calibration + validation)
    refit = fitting_indices(data, train + validation, outer + calibration)
    if not all((outer, calibration, validation, train, refit)):
        raise ValueError("A frozen group is empty after source-overlap exclusion")
    return dict(outer=outer, calibration=calibration, inner_validation=validation, train=train, refit=refit)


def source_signature():
    paths = [Path(__file__), *sorted((ROOT / "src/pbpk_kb/recommendation/pbpk_match").glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze_implementation(output, signature):
    target = output / 'implementation.zip'
    if target.exists():
        with zipfile.ZipFile(target) as saved:
            if {name: hashlib.sha256(saved.read(name)).hexdigest() for name in saved.namelist()} != signature:
                raise ValueError('Implementation archive differs from the run protocol')
        return
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as saved:
        for name, digest in signature.items():
            content = (ROOT / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Source changed while freezing implementation')
            saved.writestr(name, content)


def write_status(**values):
    payload = {"pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), **values}
    def save(path):
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        temp.replace(path)
    save(ROOT / "runtime/deployment/pbpk_match/status.json")
    if values.get('run'):
        save(ROOT / "runtime/deployment/pbpk_match" / values['run'] / 'status.json')


def baseline_reports(data, output):
    wanted = [q["query_id"] for q in data["query_records"]]
    candidates = [c["candidate_id"] for c in data["candidate_records"]]
    reports = {}
    for directory in sorted((CORE / "results").iterdir()):
        if not (directory / "index.json").is_file():
            continue
        index = json.loads((directory / "index.json").read_text())
        if index["candidate_ids"] != candidates:
            raise ValueError("Baseline candidate ordering differs")
        loc = {qid: i for i,qid in enumerate(index["query_ids"])}
        scores = np.load(directory / "scores.npy", allow_pickle=False)[:, [loc[qid] for qid in wanted]]
        report = evaluate(data, scores[0])
        report["method"] = directory.name
        report["training"] = "Frozen zero-shot external configuration; no PBPK-Match labels used for fitting"
        report["score_sha256"] = sha(directory / "scores.npy")
        changed = [q["variants"]["missing_evidence"]["text_changed"] for q in data["query_records"]]
        report["stability"] = {
            "reordered": compare_stability(scores[0],scores[1],"reordered"),
            "missing_evidence": compare_stability(scores[0],scores[2],"missing_evidence",changed)}
        reports[directory.name] = report
    write_json(output / "external_baselines.json", reports)
    return reports


def run_experiment(data, args):
    from pbpk_kb.recommendation.pbpk_match.training import (
        train_neural, predict_neural, train_tree, predict_tree,
    )
    import torch
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        if torch.cuda.mem_get_info(args.device)[0] < 5 * 1024**3:
            raise RuntimeError("Not enough GPU headroom; primary scoring remains untouched")
        torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / torch.cuda.mem_get_info(args.device)[1], args.device)
    output = scoped(args.output) / "runs" / args.run_name
    output.mkdir(parents=True,exist_ok=True)
    config = {"epochs": args.epochs, "patience": args.patience, "hidden_dim": args.hidden_dim,
              "lr": args.lr,"dropout":.1,"weight_decay":.001,"chunk_size":64,"batch_queries":2,
              "consistency_weight":args.consistency_weight,"n_jobs":2,"n_estimators":200,
              "max_depth":3,"tree_patience":15,"ordinal_balance":args.ordinal_balance,
              "validation_tie_break":args.validation_tie_break}
    protocol = {"version":"pbpk-match-development-training-v1", "run_name":args.run_name,
         "seeds":args.seeds,"methods":args.methods,"config":config,"device":args.device,
         "snapshot_sha256":sha(scoped(args.output)/"development_snapshot.json"),
         "source_sha256":source_signature(),"outer_folds":3,
         "inner_design":"4 drugs train, 1 early-stop, 1 independent calibration; refit 5 drugs at selected epoch; 3 outer holdout",
         "locked_test_quality_labels_used_for_fit_or_selection":False,
         "mixed_journal_scanned_for_development_export":True,
         "quality_scope":"Partial-judgment development study; unknowns masked, not full-catalog quality",
         "optimization_scope":"All model/config changes use development only; outer scores are development selection, not final test"}
    if (output/"protocol.json").exists():
        if json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Run is frozen; changed implementation/config requires a new --run-name")
    else:write_json(output/"protocol.json",protocol)
    freeze_implementation(output, protocol['source_sha256'])
    baselines = baseline_reports(data,output)
    write_json(output/'grade_support.json', summarize_grade_support(data['grades'],data['query_records']))
    summaries={}
    for method in args.methods:
        for seed in args.seeds:
            key=f"{method}_seed{seed}"
            folder=output/key;folder.mkdir(exist_ok=True)
            scores=np.full((3,len(data['query_records']),len(data['candidate_records'])),np.nan,np.float32)
            probabilities=np.full((*scores.shape[1:],3),np.nan,np.float64)
            calibrated=np.full_like(probabilities,np.nan)
            all_calibrated=True; fold_reports=[]
            for fold in range(3):
                target=folder/f"fold{fold}"
                target.mkdir(exist_ok=True)
                if (target/"report.json").exists() and (target/"predictions.npz").exists():
                    rec=json.loads((target/"report.json").read_text()); pred=np.load(target/"predictions.npz",allow_pickle=False)
                    outer=rec['partition']['outer'];scores[:,outer]=pred['scores'];probabilities[outer]=pred['probabilities']
                    if rec['calibrator']['state']=='fitted':calibrated[outer]=pred['calibrated_probabilities']
                    else:all_calibrated=False
                    fold_reports.append(rec);continue
                partition=fold_partition(data,fold);outer=partition['outer']
                write_status(state="training",run=args.run_name,method=method,seed=seed,fold=fold)
                print(f"[{args.run_name}] {key} fold {fold+1}/3 training ({len(partition['train'])} queries)",flush=True)
                start=time.monotonic()
                if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
                trainer,predictor=(train_tree,predict_tree) if method=='tree' else (train_neural,predict_neural)
                options=dict(seed=seed,device=args.device,config=config)
                if method!='tree':options['model_kind']=method
                model,history=trainer(data,partition['train'],partition['inner_validation'],**options)
                fixed_config={**config}
                if method=='tree':fixed_config['fixed_n_estimators']=history['best_iteration']+1
                else:fixed_config['fixed_epochs']=history['best_epoch']
                del model;gc.collect()
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
                options['config']=fixed_config
                model,refit_history=trainer(data,partition['refit'],[],**options)
                _,cal_p=predictor(model,data,partition['calibration'],device=args.device)
                cal_p=probability_normalize(cal_p)
                ids=lambda ix:sorted({data['query_records'][i]['compound_id'] for i in ix})
                calibrator=fit_probability_calibrator(np.log(np.clip(cal_p,1e-12,1)),data['applicability'][partition['calibration']],
                    [data['query_records'][i] for i in partition['calibration']],allowed_compound_ids=ids(partition['calibration']+partition['refit']),
                    excluded_compound_ids=ids(outer),model_fit_compound_ids=ids(partition['refit']))
                local=[];prediction_seconds={}
                for variant in VARIANTS:
                    before=time.monotonic()
                    val,probs=predictor(model,data,outer,variant=variant,device=args.device)
                    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
                    prediction_seconds[variant]=time.monotonic()-before
                    local.append(val)
                    if variant=='original':original_p=probability_normalize(probs)
                local=np.stack(local);scores[:,outer]=local;probabilities[outer]=original_p
                cp=None
                if calibrator['state']=='fitted':
                    cp=apply_probability_calibrator(np.log(np.clip(original_p,1e-12,1)),calibrator)
                    calibrated[outer]=cp
                else:all_calibrated=False
                rec={'partition':partition,'method':method,'seed':seed,'fold':fold,
                     'history':history,'refit_history':refit_history,'calibrator':calibrator,
                     'wall_seconds':time.monotonic()-start,'prediction_seconds':prediction_seconds,
                     'parameter_count':getattr(model,'parameter_count',None),
                     'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated(args.device)/1024**3 if args.device.startswith('cuda') else None,
                     'report':evaluate(data,local[0],original_p,outer)}
                if cp is not None:rec['calibrated_report']=evaluate(data,local[0],cp,outer)
                if method=='tree':
                    import joblib
                    joblib.dump(model,target/'model.joblib')
                else:
                    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                'model_kind':method,'config':fixed_config,'embedding_dim':data['q_embeddings'].shape[-1],
                                'pair_feature_dim':data['pair_features'].shape[-1],'num_types':data['num_types'],
                                'num_tasks':data['num_tasks'],'snapshot_sha256':protocol['snapshot_sha256']},target/'model.pt')
                np.savez_compressed(target/'predictions.npz',scores=local,probabilities=original_p,
                                    calibrated_probabilities=cp if cp is not None else np.empty((0,)))
                write_json(target/'report.json',rec);fold_reports.append(rec)
                print(f"[{key}] fold {fold+1} NDCG={rec['report']['judged_only']['metrics']['ndcg@10']} time={rec['wall_seconds']:.1f}s",flush=True)
                del model;gc.collect()
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
            assert np.isfinite(scores).all() and np.isfinite(probabilities).all()
            report=evaluate(data,scores[0],probabilities)
            report['method']=method;report['seed']=seed;report['fold_count']=3
            report['calibration_status']='all_folds_fitted' if all_calibrated else 'some_folds_unavailable'
            if all_calibrated:report['calibrated_applicability']=evaluate(data,scores[0],calibrated)['applicability']
            changed=[q['variants']['missing_evidence']['text_changed'] for q in data['query_records']]
            report['stability']={'reordered':compare_stability(scores[0],scores[1]),
                    'missing_evidence':compare_stability(scores[0],scores[2],'missing_evidence',changed)}
            report['total_wall_seconds']=sum(r['wall_seconds'] for r in fold_reports)
            report['parameter_counts']=sorted({r['parameter_count'] for r in fold_reports if r['parameter_count'] is not None})
            report['paired_external_comparisons']={name:compare_reports_paired(report, base)
                for name,base in baselines.items()}
            np.savez_compressed(folder/'out_of_fold_predictions.npz',scores=scores,probabilities=probabilities)
            write_json(folder/'summary.json',report);summaries[key]=report
            write_json(output/'summary.json',summaries)
    write_status(state='completed',run=args.run_name,methods=args.methods,seeds=args.seeds)
    print('Completed development run:',output,flush=True)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','pilot','status'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--run-name',default='pilot_v1')
    parser.add_argument('--methods',nargs='+',choices=['tree','mlp','full','no_type','no_null','hybrid'],default=['tree','mlp','full'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[17])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--hidden-dim',type=int,default=128)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--consistency-weight',type=float,default=.05)
    parser.add_argument('--ordinal-balance',choices=['none','sqrt_inverse'],default='none')
    parser.add_argument('--validation-tie-break',choices=['none','ordinal_loss'],default='none')
    args=parser.parse_args()
    if args.command=='status':
        for run in sorted((scoped(args.output)/'runs').glob('*')):
            if not (run/'protocol.json').exists():continue
            protocol=json.loads((run/'protocol.json').read_text())
            total=len(protocol['methods'])*len(protocol['seeds'])*3
            done=sum((run/f'{method}_seed{seed}'/f'fold{fold}'/'report.json').is_file()
                for method in protocol['methods'] for seed in protocol['seeds'] for fold in range(3))
            bar='█'*int(24*done/total)+'░'*(24-int(24*done/total))
            print(f'{run.name:26s} {bar} {done:2d}/{total:2d} completed folds')
        print('Counts refer to saved fold results; they do not prove a worker is currently running.')
        return
    data=build_data(args.output,device=args.device)
    if args.command=='prepare':
        print(json.dumps({'queries':len(data['query_records']),'candidates':len(data['candidate_records']),
                          'known_grades':int((data['grades']>=0).sum()),'positive_pairs':int((data['grades']>=2).sum())}));return
    try:run_experiment(data,args)
    except Exception as exc:
        write_status(state='failed',run=args.run_name,error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
