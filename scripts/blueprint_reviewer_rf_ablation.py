#!/usr/bin/env python3
"""Training-only median ablations with fixed time-rule five-point observations.

Only RF-imputed CL and/or fu may change. Qualified independent inputs remain
fixed. Native partitions are freshly exported for altered fu, while CL affects
the same aggregate clearance sink as the frozen four-axis implementation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import blueprint_reviewer_diagnostics as common
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair

RUN=common.RUN
OUT=RUN/'rf_ablation'
frozen=common.frozen
read,checked,ref,save=common.read,common.checked,common.ref,common.save
VERSION='reviewer-rf-median-ablation-1'
MODES={'median_cl':{'total_clearance'},'median_fu':{'fraction_unbound'},'median_both':{'total_clearance','fraction_unbound'}}
ENDPOINT={'total_clearance':'human_CL_mL_min_kg','fraction_unbound':'human_fup'}


def modified_request(request, mode, medians, source):
    result=deepcopy(request);changes=[]
    for e in result['evidence']:
        if e['parameter'] not in MODES[mode]:continue
        method=(e.get('prediction_method') or {}).get('name','').lower().replace(' ','_')
        if e['basis']!='predicted' or 'random_forest' not in method:continue
        endpoint=ENDPOINT[e['parameter']];old=e['value'];new=medians[endpoint]
        e['value']=float(new)
        e['source']={**source,'location':'training-only medians/'+endpoint}
        e['prediction_method']={'name':'training_median','version':VERSION,'training_overlap':'same frozen clinical structure/scaffold exclusions',
            'applicability':'Fixed training-only median ablation; no target concentration or PK labels used'}
        e['notes']='Reviewer ablation of RF-imputed value only; independently measured/derived records preserved.'
        changes.append(dict(parameter=e['parameter'],previous=old,replacement=new,unit=e['unit']))
    return result,changes


def bind_ablation_ledger(request):
    """Bind the changed evidence explicitly without touching its original ledger."""
    import hashlib
    document=dict(version=VERSION,original_ledger=request['evidence_ledger'],
        intervention='Replace only declared RF gaps with same-split training medians',evidence=request['evidence'])
    digest=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path=OUT/'ledgers'/(digest+'.json')
    if not path.exists():save(path,document)
    result=deepcopy(request);result['evidence_ledger']={**ref(path),'location':'evidence'}
    return result


def prepare():
    if (OUT/'protocol.json').exists():return status()
    s,p,basep,ch=common.source_bundle(); rf=read(RUN/'rf_heldout.json');tasks={};native={};comparisons=[]
    for c in p['cases']:
        cid=c['case_id'];batch='original_ten' if cid in p['original_case_ids'] else 'added_twenty'
        medians=rf['batches'][batch]['medians']; choice=ch[cid+'__late_four']['cmax']['sources']['time_repaired']
        source=s['units'][choice['unit_id']]['result'];unit=checked(source);training=checked(unit['training'])
        if len(training['observations'])!=5:raise ValueError('Require exact five-point time-rule observations')
        for mode in MODES:
            req=frozen.gap.request(c,'development','PKSimStandard');_,changes=modified_request(req,mode,medians,ref(RUN/'rf_heldout.json'))
            signature='+'.join(sorted(x['parameter'] for x in changes)) or 'original'
            key=cid+'__'+signature
            comparisons.append(dict(case_id=cid,drug=c['drug'],mode=mode,key=key,changed_parameters=signature,
                source=source,time_h=choice['time_h'],batch=batch))
            if key in tasks:tasks[key]['modes'].append(mode);continue
            tasks[key]=dict(case_id=cid,drug=c['drug'],mode=mode,modes=[mode],batch=batch,changes=changes,source=source,
                training=unit['training'],source_selection=unit['selection'],medians=medians)
            if any(x['parameter']=='fraction_unbound' for x in changes):
                for phase in ('development','target'):
                    for formula in frozen.base.FORMULAS:
                        nr=frozen.gap.request(c,phase,formula)
                        nr,cc=modified_request(nr,'median_fu',medians,ref(RUN/'rf_heldout.json'))
                        nk=cid+'__'+phase+'__'+formula
                        native[nk]=dict(case_id=cid,phase=phase,formula=formula,request=nr,changes=cc)
    pins=deepcopy(p['source_pins'])
    for f in [Path(__file__),Path(common.__file__),RUN/'rf_heldout.json']:
        pins[str(f.relative_to(ROOT))]=common.sha(f)
    protocol=dict(version=VERSION,source_protocol=p['source_protocol'],source_pins=pins,created_at=time.time(),
        tasks=tasks,native_tasks=native,comparisons=comparisons,fit=dict(starts=16,calls_per_formula=16384),
        clinical_values_used_for_modification=False,parameter_forest_retrained=False,
        source_selection='Frozen time-rule + same visible-only feasibility repair; exactly the same five observations',
        caches='Fresh native fu partitions; unchanged-fu anatomy and partitions reused for aggregate-CL-only intervention',
        limitations=['Development rerun on previously viewed drugs, not an independent new test',
            'RF CL influence may be absorbed by unconstrained clearance calibration; report this outcome rather than infer usefulness from initial accuracy'])
    save(OUT/'protocol.json',protocol);save(OUT/'state.json',dict(status='prepared',native_done={},fits_done={}))
    return status()


class Guard(common.Guard):
    def __call__(self,force=False):
        if (OUT/'PAUSE').exists():raise RuntimeError('RF ablation paused')
        super().__call__(force=force)


def native_task(key,job):
    """Same project-native construction and audit, explicitly one CPU thread."""
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    from pbpk_kb.blueprints.design_runtime import execute_snapshot
    from pbpk_kb.blueprints.source_adaptation_audit import audit_execution,derive_administration_expectations
    from pbpk_kb.blueprint_policy.axes_calibration import extended_snapshot,_artifacts
    from pbpk_kb.blueprint_policy.perfusion_export import export_physiology
    p=read(OUT/'protocol.json');guard=Guard(p);guard(force=True)
    folder=OUT/'native'/key;path=folder/'result.json'
    if path.exists():
        out=read(path);frozen.verify(out['artifact_pins']);return ref(path)
    start=time.monotonic();request=bind_ablation_ledger(job['request']);completion=folder/'construction.json'
    if completion.exists():execution=checked(read(completion)['execution'])
    else:
        blueprint=assemble_evidence_blueprint(request,ROOT)
        if blueprint['status']!='assembled':raise ValueError('Incomplete median-input construction')
        native=extended_snapshot(blueprint['snapshot'])
        execution=execute_snapshot(snapshot=native,output_dir=folder/'execution',project_root=ROOT,
            timeout_seconds=180,cpu_threads=1,max_output_bytes=128*1024**2)
        if not execution.get('engineering_accepted'):raise RuntimeError('Project-native median-fu construction failed; diagnosis retained')
        audit=audit_execution(snapshot=native,execution_manifest=execution,output_dir=folder/'readback',project_root=ROOT,
            **derive_administration_expectations(native))
        if not audit.get('accepted'):raise RuntimeError('Native administration readback failed')
        save(completion,dict(request=request,execution=ref(execution['manifest_path']),audit=audit))
    source=ref(execution.get('manifest_path',ROOT/read(completion)['execution']['path']))
    phys=export_physiology(execution_manifest_path=ROOT/source['path'],source_manifest_sha256=source['sha256'],
        output_dir=folder/'physiology',project_root=ROOT)
    out=dict(job=job,physiology=ref(folder/'physiology/perfusion_physiology.json'),wall_seconds=time.monotonic()-start,
        cpu_threads=1,artifact_pins=_artifacts(folder,ROOT))
    save(path,out);return ref(path)


def get_phys(p,case,phase,formula,original_req,changed_req,native_state):
    oldfu=next(e['value'] for e in original_req['evidence'] if e['parameter']=='fraction_unbound')
    newfu=next(e['value'] for e in changed_req['evidence'] if e['parameter']=='fraction_unbound')
    if oldfu==newfu:
        return frozen.prior.get_phys(checked(p['source_protocol']),case,phase,formula,original_req)
    key=case['case_id']+'__'+phase+'__'+formula
    out=checked(native_state[key]);return checked(out['physiology'])


def fit_task(key,job):
    p=read(OUT/'protocol.json');guard=Guard(p);guard(force=True);s=read(OUT/'state.json')
    if not job['changes']:return dict(reused=True,result=job['source'],reason='No RF-imputed selected input; independent evidence protected')
    path=OUT/'fits'/key/'result.json'
    if path.exists():return dict(reused=False,result=ref(path))
    start=time.monotonic();basep=checked(p['source_protocol']);case=frozen.gap.get_case(basep,job['case_id'])
    training=checked(job['training']);fits=[];sources=ref(RUN/'rf_heldout.json')
    for formula in frozen.base.FORMULAS:
        original=frozen.gap.request(case,'development',formula)
        request,changes=modified_request(original,job['mode'],job['medians'],sources)
        request=bind_ablation_ledger(request)
        phys=get_phys(p,case,'development',formula,original,request,s['native_done'])
        fp=path.parent/(formula+'.json')
        if fp.exists():f=read(fp)
        else:
            f=repair.fit(lambda x,t:frozen.old.simulate('blood_partition',phys,request,t,x),request,training,phys,
                budget=16384,starts=16,before_trial=guard)
            f.update(request=request,training_source=job['training'],input_changes=changes)
            save(fp,f)
        fits.append(dict(formula=formula,fit=ref(fp),best=f['best']))
    good=[f for f in fits if f['best']]
    result=dict(version=VERSION,job=job,status='failed',fits=fits,training=job['training'],
        source=job['source'],target_observations_used=False,output_pins={})
    if good:
        chosen=min(good,key=lambda r:r['best']['objective']);formula=chosen['formula'];best=chosen['best']
        original=frozen.gap.request(case,'target',formula)
        request,changes=modified_request(original,job['mode'],job['medians'],sources)
        request=bind_ablation_ledger(request)
        phys=get_phys(p,case,'target',formula,original,request,s['native_done'])
        grid=frozen.base.engine.prediction_grid(request)
        save(path.parent/'selection.json',dict(formula=formula,parameters=best,locked_at=time.time(),target_values_used=False))
        pred=frozen.old.simulate('blood_partition',phys,request,grid,best['log_parameters'])
        save(path.parent/'prediction.json',frozen.prior.curve_document(pred,grid,request));save(path.parent/'request.json',request)
        result.update(status='predicted',formula=formula,parameters=best,prediction=ref(path.parent/'prediction.json'),
            request=ref(path.parent/'request.json'),selection=ref(path.parent/'selection.json'),mass_balance=pred['mass_balance'])
    else:result['reason']='No feasible formula candidate'
    result['resources']=dict(wall_seconds=time.monotonic()-start,calls=sum(checked(f['fit'])['calls'] for f in fits),
        failures=sum(checked(f['fit'])['failures'] for f in fits))
    frozen.base.pins_in({k:v for k,v in result.items() if k!='output_pins'},result['output_pins'])
    save(path,result);return dict(reused=False,result=ref(path))


def run(workers):
    if not 1<=workers<=6:raise ValueError('Maximum six processes')
    # One group at a time; do not exceed the diagnostic group's assigned slots.
    if (RUN/'state.json').exists() and read(RUN/'state.json')['status']=='profiles_running':
        raise RuntimeError('Five-point profiles still use the six diagnostic slots; run RF after they lock')
    p=read(OUT/'protocol.json');guard=Guard(p);guard(force=True);s=read(OUT/'state.json')
    for kind,tasks,done,fn in [('native',p['native_tasks'],'native_done',native_task),('fit',p['tasks'],'fits_done',fit_task)]:
        s.update(status=kind+'_running',pid=os.getpid());save(OUT/'state.json',s)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(fn,k,j):k for k,j in tasks.items() if k not in s[done]}
            for future in as_completed(futures):
                key=futures[future];s[done][key]=future.result();save(OUT/'state.json',s)
                print(json.dumps(dict(kind=kind,done=len(s[done]),total=len(tasks),key=key)),flush=True)
    s.update(status='predictions_locked',locked_at=time.time());save(OUT/'state.json',s)
    return report()


def report():
    p=read(OUT/'protocol.json');s=read(OUT/'state.json')
    if s['status'] not in ('predictions_locked','complete'):raise ValueError('All predictions must lock before target scoring')
    basep=checked(p['source_protocol']);rows=[];scores={}
    for comparison in p['comparisons']:
        case=frozen.gap.get_case(basep,comparison['case_id']);entry=s['fits_done'][comparison['key']];unit=checked(entry['result'])
        if entry['result']['path'] not in scores:
            scores[entry['result']['path']]=frozen.data.score_case(case,unit) if unit['status']=='predicted' else None
        baseline=checked(comparison['source'])
        if comparison['source']['path'] not in scores:scores[comparison['source']['path']]=frozen.data.score_case(case,baseline)
        score=scores[entry['result']['path']];orig=scores[comparison['source']['path']]
        rows.append(dict(**comparison,reused=entry['reused'],result=entry['result'],status=unit['status'],
            score=score,baseline_score=orig,parameters=common.summarize_fit(unit),baseline_parameters=common.summarize_fit(baseline)))
    summary={}
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode]
        summary[mode]={}
        for ep in ('curve','cmax','auc','terminal_half_life'):
            vals=[];changed=[]
            for row in rr:
                if row['score'] is None:continue
                a=row['score']['curve_metrics']['gmfe'] if ep=='curve' else row['score']['endpoints'][ep]['error']
                b=row['baseline_score']['curve_metrics']['gmfe'] if ep=='curve' else row['baseline_score']['endpoints'][ep]['error']
                if a is None or b is None:continue
                vals.append((a,b));
                if row['changed_parameters']!='original':changed.append((a,b))
            def aggregate(v):
                if not v:return dict(n=0)
                gm=lambda x:math.exp(sum(math.log(z) for z in x)/len(x))
                ratios=[a/b for a,b in v]
                return dict(n=len(v),median_input_gmfe=gm([a for a,b in v]),rf_input_gmfe=gm([b for a,b in v]),
                    error_ratio_median_over_rf=gm(ratios),median_better=sum(x<1-1e-10 for x in ratios),
                    median_worse=sum(x>1+1e-10 for x in ratios),ties=sum(abs(x-1)<=1e-10 for x in ratios))
            summary[mode][ep]=dict(all_drugs=aggregate(vals),input_actually_changed=aggregate(changed))
    save(OUT/'results.json',dict(version=VERSION,protocol=ref(OUT/'protocol.json'),predictions_locked_at=s['locked_at'],
        scored_at=time.time(),rows=rows,summary=summary,independent_evidence_protected=True))
    s.update(status='complete',results=ref(OUT/'results.json'));save(OUT/'state.json',s);return status()


def status():
    p=read(OUT/'protocol.json');s=read(OUT/'state.json')
    return dict(status=s['status'],native_done=len(s['native_done']),native_total=len(p['native_tasks']),
        fits_done=len(s['fits_done']),unique_fits=len(p['tasks']),comparison_units=len(p['comparisons']))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run','status','report']);a.add_argument('--workers',type=int,default=6);args=a.parse_args()
    value=prepare() if args.command=='prepare' else run(args.workers) if args.command=='run' else report() if args.command=='report' else status()
    print(json.dumps(value,ensure_ascii=False))
