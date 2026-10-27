#!/usr/bin/env python3
"""Frozen, grouped evaluation of one visible-context sampling policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from pbpk_kb.blueprint_policy import contextual_sampling_design as policy
from blueprint_sampling_policy_data import load_policy_inputs

RUN=ROOT/'experiments/blueprint_design/sampling_policy_20261009'
STABLE=ROOT/'experiments/blueprint_design/sampling_stability_20261009'
ALPHAS=(.1,1.,10.)
METHODS=('time','contextual','original_features','without_curve_guard','without_study_guard','frozen_stable','no_addition')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))
def checked(r):
    p=(ROOT/r['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p)!=r['sha256']:raise ValueError('Source mismatch: '+str(p))
    return read(p)
def save(path,obj):
    p=Path(path);assert p.resolve().is_relative_to(RUN)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def guard(p):
    for name,h in p['source_pins'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed frozen source: '+name)
    if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())>10*1024**3:raise ValueError('10 GiB batch limit')


def prepare():
    if (RUN/'protocol.json').exists():raise ValueError('Protocol already frozen; use run/status')
    raw=load_policy_inputs(allow_model_replay=True)
    designs=[]
    for d in raw['designs']:
        x=policy.contextual_features(d['goals']['cmax']['X'],d['candidate_times_h'],d['visible']['time_h'],
            d['visible']['concentration_mg_l'],d['baseline']['visible_prediction_mg_l'],
            d['committee']['concentrations_mg_l'],d['committee']['cmax_ng_ml'])
        distances=np.asarray(x)[:,1];eligible=np.flatnonzero(distances>=distances.max()-1e-12)
        j=min(eligible,key=lambda i:d['candidate_times_h'][i])
        designs.append(dict(design_key=d['design_key'],case_id=d['case_id'],drug=d['drug'],regime=d['regime'],
            drug_study_group=d['drug_study_group'],cluster_id=d['old_target_cluster'],
            candidate_ids=d['candidate_ids'],times=d['candidate_times_h'],X=x.tolist(),time_index=int(j)))
    save(RUN/'inputs.json',dict(raw=raw,designs=designs))
    source=read(STABLE/'protocol.json');orig=set(source['original_case_ids'])
    folds={}
    for group in sorted({d['drug_study_group'] for d in designs}):
        hold={d['case_id'] for d in designs if d['drug_study_group']==group}
        train=sorted({d['case_id'] for d in designs}-hold)
        folds['grouped30__'+group]=dict(evaluation='grouped30',heldout_case_ids=sorted(hold),train_case_ids=train)
    added=sorted({d['case_id'] for d in designs}-orig)
    testgroups={d['drug_study_group'] for d in designs if d['case_id'] in added}
    allowed=sorted({d['case_id'] for d in designs if d['case_id'] in orig and d['drug_study_group'] not in testgroups})
    folds['original10_to_added20']=dict(evaluation='original10_to_added20',heldout_case_ids=added,train_case_ids=allowed)
    records=RUN/'candidate_repairs/records.json'
    if not records.exists():raise ValueError('Complete uniformly repaired candidate predictions required')
    pins={r['path']:r['sha256'] for r in raw['source_manifest']}
    for path in (Path(__file__),ROOT/'src/pbpk_kb/blueprint_policy/contextual_sampling_design.py',
                 ROOT/'scripts/blueprint_sampling_policy_data.py',records,RUN/'inputs.json',STABLE/'protocol.json'):
        pins[str(path.relative_to(ROOT))]=sha(path)
    p=dict(version=policy.VERSION,created_at=time.time(),inputs=ref(RUN/'inputs.json'),
        outcome_store=ref(records),previous_protocol=ref(STABLE/'protocol.json'),previous_choices=source['choices'],
        source_pins=pins,original_case_ids=sorted(orig),folds=folds,methods=list(METHODS),
        algorithm=dict(features=list(policy.FEATURE_NAMES),alpha_grid=list(ALPHAS),label_cap_log=policy.LABEL_CAP,
            kernel_bandwidth_log=policy.LOG_TIME_BANDWIDTH,log_noise_sd=.1,
            relative_gain='ln(time-rule error / candidate error), Cmax and curve fitted separately',
            choice='maximize minimum Cmax gain over full and leave-training-study models, with minimum curve gain >= -ln(1.05); otherwise time',
            inner_selection='Leave one training drug-study group out; minimize late-four Cmax GMFE ratio among alpha choices whose early and late curve ratios <=1.01. Time policy is included. Ties choose stronger regularization.',
            half_life='No half-life selection training: quality insufficient. Historical endpoint errors retained as secondary diagnostics.'),
        primary=dict(evaluation='original10_to_added20',regime='late_four',endpoint='cmax',reference='time'),
        development=dict(evaluation='grouped30',note='All 30 already viewed; outer held-out predictions are retrospective development CV, not fresh clinical validation'),
        support_rule='Additional predictive support only if primary Cmax error-ratio 95% bootstrap upper bound <1 and curve GMFE ratio <=1.01. Early-four is reported separately, no endpoint switching.',
        bootstrap=dict(seed=20261009,resamples=2000),new_fits=0,monitor_seconds=1200,directory_limit_gib=10,
        source_repair='586 complete uniformly treated candidate records; fifth-point feasibility belongs to acquisition outcomes, never features',
        isolation='All candidates/layouts and shared development/target study components removed from outer training. Inner regularization selection repeats grouping; only training labels passed to learner. Candidate and target concentrations never selection features.',
        limitations=['Current 30 drugs already viewed during previous development; no blind-test claim.',
            'Study deletion scores are not calibrated confidence intervals or clinical risk probabilities.',
            'Cmax is sampled peak, not necessarily true continuous peak; endpoints missing reference remain missing.',
            'No test-effect-based feature or threshold search; feature/guard ablations are all retained.',
            'Proposal benefits include downstream feasibility update under the same policy for every action.'])
    guard(p);save(RUN/'protocol.json',p);save(RUN/'state.json',dict(status='prepared',protocol=ref(RUN/'protocol.json'),folds={}))
    return status()


def labels_for(designs, records, permitted):
    """Whitelist before looking up any clinical outcome or repaired action."""
    if any(d['case_id'] not in permitted for d in designs):raise ValueError('Held-out label requested')
    out={}
    for d in designs:
        r=[records[d['design_key'],d['design_key']+f'__point_{i:02d}'] for i in d['candidate_ids']]
        out[d['design_key']]=dict(cmax=[v['endpoints']['cmax']['error'] for v in r],curve=[v['curve_metrics']['gmfe'] for v in r])
    return out


def tune(designs, labels, dimensions):
    groups=sorted({d['drug_study_group'] for d in designs}); trials=[]
    for alpha in ALPHAS:
        out=[]
        for g in groups:
            train=[d for d in designs if d['drug_study_group']!=g];held=[d for d in designs if d['drug_study_group']==g]
            # Labels for the inner holdout are not in model training arguments.
            train_labels={d['design_key']:labels[d['design_key']] for d in train}
            model=policy.train(train,train_labels,dimensions=dimensions,alpha=alpha)
            choices=[(d,policy.choose(model,d['X'],d['times'],d['time_index'])) for d in held]
            for d,c in choices:
                y=labels[d['design_key']];i,j=c['selected_index'],d['time_index']
                out.append(dict(case_id=d['case_id'],group=g,regime=d['regime'],index=i,
                    cmax_log_ratio=math.log(y['cmax'][i]/y['cmax'][j]),curve_log_ratio=math.log(y['curve'][i]/y['curve'][j])))
        means={reg:{e:float(np.mean([r[e+'_log_ratio'] for r in out if r['regime']==reg])) for e in ('cmax','curve')} for reg in ('late_four','early_four')}
        eligible=all(means[r]['curve']<=math.log(1.01) for r in means)
        trials.append(dict(alpha=alpha,means=means,eligible=eligible,heldout_decisions=out))
    eligible=[t for t in trials if t['eligible'] and t['means']['late_four']['cmax'] < -1e-12]
    selected=min(eligible,key=lambda t:(t['means']['late_four']['cmax'],-t['alpha'])) if eligible else None
    return dict(alpha=selected['alpha'] if selected else None,reason='training_only_grouped_selection' if selected else 'training_CV_prefers_time',
        trials=trials,train_case_ids=sorted({d['case_id'] for d in designs}),train_groups=groups)


def fold_job(key,spec):
    p=read(RUN/'protocol.json');guard(p);data=checked(p['inputs'])['designs']
    train=[d for d in data if d['case_id'] in spec['train_case_ids']]
    held=[d for d in data if d['case_id'] in spec['heldout_case_ids']]
    if {d['drug_study_group'] for d in train}&{d['drug_study_group'] for d in held}:raise ValueError('Study leakage')
    # Outcome store is steward-only. Held-out values do not enter either model.
    records={(r['design_key'],r['candidate']):r for r in checked(p['outcome_store'])['records']}
    labels=labels_for(train,records,set(spec['train_case_ids']))
    tuning={str(dim):tune(train,labels,dim) for dim in (8,12)}
    bundles={str(dim):policy.train(train,labels,dimensions=dim,alpha=tuning[str(dim)]['alpha'] or 1.) for dim in (8,12)}
    decisions={}
    for d in held:
        methods={}
        for method,dim,cg,sg in [('contextual',12,True,True),('original_features',8,True,True),
                               ('without_curve_guard',12,False,True),('without_study_guard',12,True,False)]:
            if tuning[str(dim)]['alpha'] is None:
                methods[method]=dict(selected_index=d['time_index'],changed=False,reason='training_CV_prefers_time')
            else:
                methods[method]=policy.choose(bundles[str(dim)],d['X'],d['times'],d['time_index'],curve_guard=cg,study_guard=sg)
        decisions[d['design_key']]=methods
    out=dict(fold=key,**spec,tuning=tuning,models=bundles,choices=decisions,locked_at=time.time(),heldout_outcomes_used=False)
    path=RUN/'folds'/f'{key}.json';save(path,out)
    return ref(path)


def run(workers):
    if not 1<=workers<=6:raise ValueError('1–6 workers')
    s=read(RUN/'state.json');p=checked(s['protocol']);guard(p)
    if s['status']=='complete':return status()
    s.update(status='grouped_training',pid=os.getpid(),started_at=s.get('started_at',time.time()));save(RUN/'state.json',s)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(fold_job,k,v):k for k,v in p['folds'].items() if k not in s['folds']}
            for f in as_completed(pending):
                k=pending[f];s['folds'][k]=f.result();save(RUN/'state.json',s)
                print(f'Locked study folds {len(s["folds"])}/{len(p["folds"])}',flush=True)
        guard(p);s.update(status='choices_locked',choices_locked_at=time.time());save(RUN/'state.json',s)
        return report()
    except BaseException as e:
        s.update(status='paused',reason=str(e));save(RUN/'state.json',s);raise


def bootstrap(rows,method,endpoint):
    by={(r['case_id'],r['method']):r for r in rows};ids=sorted({r['case_id'] for r in rows})
    vals=[];groups=[]
    for cid in ids:
        a=by[cid,method]['errors'][endpoint];b=by[cid,'time']['errors'][endpoint]
        if a is not None and b is not None and a>0 and b>0:
            vals.append(math.log(a/b));groups.append(by[cid,method]['study_group'])
    v=np.array(vals);gs=sorted(set(groups));indices=[np.flatnonzero(np.array(groups)==g) for g in gs]
    rng=np.random.default_rng(20261009)
    draws=[float(np.exp(v[np.concatenate([indices[i] for i in take])].mean())) for take in rng.integers(0,len(gs),(2000,len(gs)))]
    return dict(n=len(v),clusters=len(gs),ratio=float(np.exp(v.mean())),ci95=np.quantile(draws,[.025,.975]).tolist(),
        improved=int(sum(v < -1e-10)),worse=int(sum(v > 1e-10)),tied=int(sum(abs(v)<=1e-10)))


def report():
    s=read(RUN/'state.json');p=checked(s['protocol']);guard(p)
    if s['status']=='complete':return status()
    if s['status']!='choices_locked':raise ValueError('Choices must lock before held-out scoring')
    records={(r['design_key'],r['candidate']):r for r in checked(p['outcome_store'])['records']}
    designs={d['design_key']:d for d in checked(p['inputs'])['designs']};old=checked(p['previous_choices'])['choices']
    rows=[]
    for fr in s['folds'].values():
        fold=checked(fr)
        for key,choices in fold['choices'].items():
            d=designs[key]
            selected=dict(time=d['time_index'],no_addition=None,
                frozen_stable=old[key]['cmax']['decision']['selected_index'],
                **{m:choices[m]['selected_index'] for m in choices})
            for method,i in selected.items():
                cid='baseline' if i is None else key+f'__point_{d["candidate_ids"][i]:02d}'
                record=records[key,cid]
                errors={ep:v['error'] for ep,v in record['endpoints'].items()};errors['curve']=record['curve_metrics']['gmfe']
                rows.append(dict(evaluation=fold['evaluation'],regime=d['regime'],case_id=d['case_id'],drug=d['drug'],
                    cohort='original10' if d['case_id'] in p['original_case_ids'] else 'added20',study_group=d['drug_study_group'],
                    method=method,candidate=cid,time_h=None if i is None else d['times'][i],changed=i!=d['time_index'],
                    errors=errors,endpoints=record['endpoints'],source=record['source']))
    summaries={}
    for ev in ('grouped30','original10_to_added20'):
        for reg in ('late_four','early_four'):
            rr=[r for r in rows if r['evaluation']==ev and r['regime']==reg]
            out={}
            for m in METHODS:
                mr=[r for r in rr if r['method']==m]
                eps={}
                for ep in ('curve','cmax','auc','c_last','terminal_half_life'):
                    v=[r['errors'][ep] for r in mr if r['errors'][ep] is not None]
                    eps[ep]=dict(n=len(v),gmfe=float(np.exp(np.mean(np.log(v)))),within_two=sum(x<=2 for x in v),paired_vs_time=bootstrap(rr,m,ep))
                out[m]=dict(n=len(mr),changed=sum(r['changed'] for r in mr),endpoints=eps)
            summaries[ev+'__'+reg]=out
    primary=summaries['original10_to_added20__late_four']['contextual']['endpoints']
    supported=primary['cmax']['paired_vs_time']['ci95'][1]<1 and primary['curve']['paired_vs_time']['ratio']<=1.01
    result=dict(version=policy.VERSION,protocol=s['protocol'],choices_locked_at=s['choices_locked_at'],scored_at=time.time(),
        summaries=summaries,rows=rows,primary_incremental_support=supported,
        conclusion='Frozen retrospective primary gate passed; independent prospective validation still required' if supported else 'No primary evidence of extra learned-selection benefit; do not replace the paper winner',
        all_cases_retained=True,unknown_endpoints_imputed=False)
    save(RUN/'results.json',result)
    with (RUN/'comparisons.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=['evaluation','regime','drug','method','candidate','time_h','changed','curve','cmax','auc','c_last','terminal_half_life'])
        writer.writeheader()
        for r in rows:writer.writerow({**{k:r[k] for k in ('evaluation','regime','drug','method','candidate','time_h','changed')},**{k:r['errors'][k] for k in ('curve','cmax','auc','c_last','terminal_half_life')}})
    s.update(status='complete',results=ref(RUN/'results.json'),completed_at=time.time());save(RUN/'state.json',s)
    return status()


def status():
    s=read(RUN/'state.json');p=checked(s['protocol'])
    return dict(status=s['status'],folds=len(s['folds']),total_folds=len(p['folds']),results=s.get('results'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run','status','report']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    out=prepare() if args.command=='prepare' else run(args.workers) if args.command=='run' else report() if args.command=='report' else status()
    print(json.dumps(out,ensure_ascii=False))
