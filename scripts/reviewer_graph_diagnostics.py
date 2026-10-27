#!/usr/bin/env python3
"""Source-bound post-development graph controls; never change frozen experiments.

Reuse original splits, silver labels, tensors, seeds, loss and stopping rule.
New controls disable only message passing, shared teacher initialization, or
teacher initialization plus anchoring. Residual removal is an inference ablation
at the original selected checkpoint, not a separately selected no-blend model.
"""
from __future__ import annotations
import argparse, csv, hashlib, inspect, json, sys, time, types
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'experiments/component_recommendation'
OLD = BASE / '80_gate4_v3_gnn_training'
BENCH = BASE / '79_gate4_v3_locked_benchmark'
OUT = ROOT / 'experiments/model_recommendation/reviewer_revision_20261009/graph_diagnostics'
SEEDS = (17, 29, 43)
METRICS = ('ndcg_at_10', 'recall_at_10', 'reciprocal_rank')

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def paired_cluster(before, after, groups, seed=20261009, replicates=10000):
    """Average seeds per query; resample semantic groups, retaining query weights."""
    query_ids=sorted(before)
    assert query_ids==sorted(after) and all(q in groups for q in query_ids)
    by=defaultdict(list)
    for q in query_ids: by[groups[q]].append(after[q]-before[q])
    sums=np.asarray([sum(by[g]) for g in sorted(by)])
    counts=np.asarray([len(by[g]) for g in sorted(by)])
    sample=np.random.default_rng(seed).integers(len(sums),size=(replicates,len(sums)))
    boot=sums[sample].sum(1)/counts[sample].sum(1)
    signs=1-2*((np.arange(2**len(sums),dtype=np.uint32)[:,None]>>np.arange(len(sums)))&1).astype(float)
    null=np.abs(signs@sums/counts.sum())
    observed=float(abs(sums.sum()/counts.sum()))
    return {'difference':float(sums.sum()/counts.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'semantic_groups':len(by),'queries':len(query_ids),'replicates':replicates,
            'exact_group_sign_flip_p':float(np.mean(null>=observed-1e-14)),
            'groups_with_nonzero_mean_difference':int((np.abs(sums)>1e-12).sum()),
            'estimand':'query-weighted paired difference, seeds averaged before group resampling'}

def summarize():
    groups={r['query_id']:r['semantic_group_id'] for r in rows(BENCH/'semantic_split.jsonl') if r['split']=='test'}
    reports={};source_paths=[BENCH/'semantic_split.jsonl',BENCH/'gnn_benchmark_manifest.json',Path(__file__)]
    for name in ('no_gnn','hgt'):
        paths=[OLD/name/f'seed_{s}_metrics.json' for s in SEEDS]
        for p in paths:
            assert sha(p.with_name(p.name.replace('_metrics.json','.pt')))==read(p)['checkpoint_sha256']
        source_paths+=paths
        reports[name]=[read(p)['test'] for p in paths]
    reports['hgt_no_residual_at_original_checkpoint']=[read(OLD/'hgt'/f'seed_{s}_metrics.json')['raw_test'] for s in SEEDS]
    for control in ('no_message_passing','no_teacher_initialization','no_teacher_initialization_or_anchor'):
        files=[OUT/control/'hgt'/f'seed_{s}_metrics.json' for s in SEEDS]
        if all(p.exists() for p in files):
            for p in files:
                assert sha(p.with_name(p.name.replace('_metrics.json','.pt')))==read(p)['checkpoint_sha256']
            source_paths+=files
            reports[control]=[read(p)['test'] for p in files]
    means={};flat=[]
    for name,reps in reports.items():
        means[name]={}
        for metric in METRICS:
            query_values=defaultdict(list)
            for seed,rep in zip(SEEDS,reps):
                for r in rep['per_query']:
                    query_values[r['query_id']].append(r[metric])
                    flat.append({'method':name,'seed':seed,'query_id':r['query_id'],'semantic_group':groups[r['query_id']], 'metric':metric,'value':r[metric]})
            means[name][metric]={q:float(np.mean(v)) for q,v in query_values.items()}
    comparisons=[]
    pairs=[('no_gnn','hgt'),('hgt_no_residual_at_original_checkpoint','hgt')]
    pairs += [(c,'hgt') for c in reports if c.startswith('no_') and c!='no_gnn']
    for a,b in pairs:
        for metric in METRICS:
            comparisons.append({'before':a,'after':b,'metric':metric,**paired_cluster(means[a][metric],means[b][metric],groups)})
    summary={'schema':'reviewer-graph-controls-v1','status':'complete' if len(reports)==6 else 'in_progress',
        'methods':{m:{k:float(np.mean(list(v.values()))) for k,v in vals.items()} for m,vals in means.items()},
        'comparisons':comparisons,'queries':len(groups),'semantic_groups':len(set(groups.values())),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        'interpretation':'Post-development silver-label diagnostics, not independent scientific validation; seeds are not independent queries.',
        'original_training_mechanisms':{'teacher_shared_initialization':True,'teacher_anchor_weight':.1,'validation_selected_category_residual':True},
        'control_definitions':{'no_message_passing':'HGT encoder returns its node feature encoder before graph layers; same teacher init/anchor, optimizer groups and residual grid.',
            'no_teacher_initialization':'Do not copy NoGNN shared weights; retain teacher anchor and residual.',
            'no_teacher_initialization_or_anchor':'Do not copy NoGNN shared weights and disable teacher anchor; retain residual.',
            'hgt_no_residual_at_original_checkpoint':'Use saved raw HGT scores; original checkpoint selected with blending.'}}
    write(OUT/'summary.json',summary)
    with (OUT/'per_query.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(summary['methods']),flush=True)
    return summary

def train(device):
    import torch
    import pbpk_kb.recommendation.gnn_benchmark as g
    torch.set_num_threads(4)
    manifest=read(BENCH/'gnn_benchmark_manifest.json')
    graphdir=BASE/'107_runtime_graph_gate5_locked';index=BASE/'73_gate3_multiretriever_production_index'
    assert sha(graphdir/'kg_nodes.jsonl')==manifest['input_hashes']['nodes']
    assert sha(graphdir/'kg_edges.jsonl')==manifest['input_hashes']['edges']
    for key,filename in {'query_records':'query_records.jsonl','query_embeddings':'query_embeddings.npy','semantic_split':'semantic_split.jsonl','query_subgraph_pairs':'query_subgraph_pairs.jsonl','relation_split':'relation_split.jsonl','subgraph_index':'subgraph_index.jsonl','acceptance':'preregistered_acceptance.json'}.items():
        assert sha(BENCH/filename)==manifest['output_hashes'][key]
    dev=torch.device(device)
    graph=g.build_graph_tensors(graph_dir=graphdir,embedding_index_dir=index)
    qr={r['query_id']:r['query_row'] for r in rows(BENCH/'query_records.jsonl')}
    sub={r['subgraph_id']:r for r in rows(BENCH/'subgraph_index.jsonl')}
    allpairs=rows(BENCH/'query_subgraph_pairs.jsonl')
    pairs={s:g.build_pair_tensors(rows=[r for r in allpairs if r['split']==s],query_row_by_id=qr,subgraph_by_id=sub,
        node_index={n:i for i,n in enumerate(graph.node_ids)},relation_vocab=graph.relation_vocab).to(dev) for s in ('train','validation','test')}
    ri=g._relation_indices(graph,rows(BENCH/'relation_split.jsonl'))
    context=g._subset_graph(graph,ri['context']).to(dev);graph=graph.to(dev)
    rel={s:(graph.forward_edge_index[:,torch.tensor(ri[s],device=dev)],
        torch.tensor([graph.relation_ids[i] for i in ri[s]],device=dev)) for s in ('train','validation','test')}
    queries=torch.from_numpy(np.load(BENCH/'query_embeddings.npy')).to(dev)
    input_hash=g._model_input_hash(manifest,BENCH/'preregistered_acceptance.json')
    source=inspect.getsource(g.train_one)
    assert source.count('model.load_state_dict(shared_state, strict=False)')==1
    source=source.replace('model.load_state_dict(shared_state, strict=False)',
                          'model.load_state_dict(shared_state, strict=False) if REVIEWER_INITIALIZE else None')
    original_factory=g.build_ranker
    namespace=dict(g.__dict__)
    exec(compile(source,str(OUT/'controlled_train_one.py'),'exec'),namespace)
    controlled=namespace['train_one']
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'controlled_train_one.py').write_text(source)
    write(OUT/'protocol.json',{'frozen_source':str(Path(g.__file__).relative_to(ROOT)),'frozen_source_sha256':sha(g.__file__),
        'controlled_source_sha256':sha(OUT/'controlled_train_one.py'),'seeds':SEEDS,'training_config':g.TRAINING_CONFIG,
        'input_hash':input_hash,'fit_count':9,'development_only':True,'test_used_for_stopping':False})
    for control in ('no_message_passing','no_teacher_initialization','no_teacher_initialization_or_anchor'):
        namespace['REVIEWER_INITIALIZE']=control=='no_message_passing'
        namespace['LOSS_WEIGHTS']=dict(g.LOSS_WEIGHTS)
        # _losses resolves the module's loss weights; restore after each run.
        g.LOSS_WEIGHTS['teacher_anchor']=0. if control=='no_teacher_initialization_or_anchor' else .1
        namespace['LOSS_WEIGHTS']=g.LOSS_WEIGHTS
        def factory(*,architecture,graph,category_count):
            m=original_factory(architecture=architecture,graph=graph,category_count=category_count)
            if control=='no_message_passing' and architecture=='hgt':
                m.encode_nodes=types.MethodType(lambda self,cg:self._feature_encoder(cg),m)
            return m
        namespace['build_ranker']=factory
        for seed in SEEDS:
            result=OUT/control/'hgt'/f'seed_{seed}_metrics.json'
            if result.exists(): continue
            print(json.dumps({'phase':'training','control':control,'seed':seed}),flush=True)
            start=time.monotonic();torch.cuda.reset_peak_memory_stats(dev)
            controlled(architecture='hgt',seed=seed,graph=graph,context_graph=context,query_embeddings=queries,
                pairs_by_split=pairs,relation_splits=rel,output_dir=OUT/control,device=dev,input_hash=input_hash,
                teacher_checkpoint_path=OLD/'no_gnn'/f'seed_{seed}.pt')
            write(OUT/control/f'seed_{seed}_resource.json',{'seconds':time.monotonic()-start,
                  'peak_allocated_gib':torch.cuda.max_memory_allocated(dev)/1024**3})
            print(json.dumps({'phase':'finished','control':control,'seed':seed,'seconds':time.monotonic()-start}),flush=True)
        summarize()
    g.LOSS_WEIGHTS['teacher_anchor']=.1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    summarize()
    if a.train: train(a.device)
