#!/usr/bin/env python3
"""Independent completion audit; never reads clinical outcomes while running.

The frozen clinical scorer is replayed for every unique prediction. Aggregates
and cluster bootstrap are calculated separately here, not through the report's
aggregation functions. Three predetermined prediction units are simulated
again without optimization. This audit checks integrity, not scientific success.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'experiments/blueprint_design/sequential_sampling_20261008'
VERSION = 'sequential-sampling-independent-audit-1'
TRAIN_FIELDS = {'compound_id','study_id','role','matrix','time_unit','unit',
                'molecular_weight_g_mol','human','dosing','observations'}
SEEDS = (20261008,20261009)
BOOTSTRAP_SEED, RESAMPLES = 20260929, 2000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(source):
    path=(ROOT/source['path']).resolve()
    require(path.is_relative_to(ROOT),'Source outside project boundary')
    require(sha(path)==source['sha256'],'Source hash mismatch: '+source['path'])
    return read(path)


def check_pins(pins):
    for name,digest in pins.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT),'Pinned source outside project')
        require(sha(path)==digest,'Pinned source changed: '+name)


def same(a,b,label,*,rtol=1e-10,atol=1e-12):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),label+': dictionary fields differ')
        for key in b:
            same(a[key],b[key],label+'.'+str(key),rtol=rtol,atol=atol)
    elif isinstance(b,(list,tuple)):
        require(isinstance(a,(list,tuple)) and len(a)==len(b),label+': array size differs')
        for i,(av,bv) in enumerate(zip(a,b)):
            same(av,bv,label+f'[{i}]',rtol=rtol,atol=atol)
    elif isinstance(b,(int,float)) and not isinstance(b,bool):
        require(isinstance(a,(int,float)) and not isinstance(a,bool)
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a,b,rel_tol=rtol,abs_tol=atol),label+': numerical mismatch')
    else:
        require(type(a) is type(b) and a==b,label+': value/type differs')


def valid(value):
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and value>0)


def independently_seeded_layouts(full):
    times=[row['time_min'] for row in full['observations']]
    require(all(t>=0 and math.isfinite(t) for t in times)
            and all(a<b for a,b in zip(times,times[1:])),'Unordered source times')
    ids=[i for i,t in enumerate(times) if t>0]
    require(len(ids)>=7,'Fewer than seven positive development times')
    early=set(ids[:len(ids)//2]);late=set(ids[len(ids)//2:]);answer=[]
    for seed in SEEDS:
        rng=random.Random(seed)
        for _ in range(10000):
            candidate=sorted(rng.sample(ids,4))
            if early.intersection(candidate) and late.intersection(candidate) and candidate not in answer:
                answer.append(candidate);break
        else:
            raise ValueError('Cannot recreate a distinct mixed layout')
    return answer


def training_for(full,ids):
    result={key:deepcopy(full[key]) for key in TRAIN_FIELDS-{'observations'}}
    result['observations']=[dict(time_min=float(full['observations'][i]['time_min']),
        concentration_umol_per_l=float(full['observations'][i]['concentration_umol_per_l'])) for i in sorted(ids)]
    return result


def independent_cluster_bootstrap(values,clusters):
    """Multiplicity matrix over whole study groups; no production aggregator."""
    ids=sorted(values);groups=sorted({clusters[k] for k in ids})
    if not ids:
        return dict(estimate=None,ci95=None,n_cases=0,n_clusters=0)
    logs=np.log(np.asarray([values[k] for k in ids],float))
    require(np.isfinite(logs).all(),'Nonfinite bootstrap log values')
    answer=dict(estimate=float(np.exp(np.mean(logs))),n_cases=len(ids),n_clusters=len(groups))
    if len(groups)<2:
        return dict(answer,ci95=None,ci_reason='Fewer than two independent study clusters')
    membership=np.asarray([[int(clusters[k]==group) for k in ids] for group in groups],int)
    draws=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(groups),size=(RESAMPLES,len(groups)))
    multiplicities=np.asarray([np.bincount(row,minlength=len(groups)) for row in draws],int)
    drug_multiplicities=multiplicities@membership
    estimates=np.exp((drug_multiplicities@logs)/drug_multiplicities.sum(axis=1))
    answer.update(ci95=np.quantile(estimates,[.025,.975]).tolist(),resamples=RESAMPLES,seed=BOOTSTRAP_SEED)
    return answer


def independent_pair(numerator,denominator,clusters):
    ids=sorted(clusters)
    ratios={k:numerator[k]/denominator[k] for k in ids if valid(numerator.get(k)) and valid(denominator.get(k))}
    answer=independent_cluster_bootstrap(ratios,clusters)
    answer.update(attempted_pairs=len(ids),complete_pairs=len(ratios),
        incomplete_case_ids=[k for k in ids if k not in ratios],per_case_ratio=ratios,
        ratio_below_one=sum(v<1 for v in ratios.values()),ratio_above_one=sum(v>1 for v in ratios.values()),
        ci_condition='Complete paired cases; failed attempts remain in the stated denominator')
    return answer


def validate_layouts_sequences(run,p,state,runner):
    cases={c['case_id']:c for c in p['cases']};methods=p['methods'];layouts=p['layouts']
    require(len(cases)==10 and len(layouts)==20 and len(methods)==7,'Frozen cohort/method size differs')
    require(p['budgets']==[0,1,2,3] and p['goal']=='cmax','Frozen budgets/request differ')
    require(set(state['completed'])==set(cases),'Global prediction lock lacks cases')
    units={};entries=[];fulls={};done={};action_sources={};subset_paths={};first_reveal={};failure_count=0
    protocol_hash=sha(run/'protocol.json')

    def unit_for(case,source,ids,full):
        value=checked(source);key=source['path'];canonical=sorted(ids)
        require(value['protocol_sha256']==protocol_hash,'Unit protocol mismatch')
        same(value['identity'],dict(case_id=case['case_id'],visible_ids=canonical),'Unit identity')
        check_pins(value['output_pins'])
        same(checked(value['training']),training_for(full,canonical),'Whitelisted visible training')
        payload=dict(case_id=case['case_id'],visible_ids=canonical,
            request=case['requests']['evidence']['development'],source=case['sampling_sources']['full'],protocol=protocol_hash)
        expected=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
        require(Path(key).parent.name==expected,'Subset cache key differs from exact scientific inputs')
        sk=(case['case_id'],tuple(canonical))
        require(sk not in subset_paths or subset_paths[sk]==key,'Identical subset fitted separately by policy')
        require(key not in units or units[key]['identity']==value['identity'],'Different subsets share one fit')
        subset_paths[sk]=key;units[key]=value
        require(value['status'] in ('predicted','failed'),'Unknown prediction status')
        require(len(value['fits'])==2,'Both fixed formulas must have recorded fit outcomes')
        if value['status']=='failed':
            require(bool(value.get('reason')),'Failed fit lacks diagnostics')
        else:
            selection=checked(value['selection'])
            require(selection['target_values_used'] is False,'Target outcomes used for formula selection')
            same(selection['parameters'],value['parameters'],'Selection parameters')
            feasible=[x for x in value['fits'] if x['best'] is not None]
            require(bool(feasible),'Successful prediction without feasible fit')
            best=min(feasible,key=lambda x:x['best']['objective'])
            require(selection['formula']==value['formula']==best['formula'],'Formula not selected by visible objective')
            same(best['best'],value['parameters'],'Minimum visible objective parameters')
        return value

    for cid,case in cases.items():
        req=runner.request(case,'development','PKSimStandard')
        full=runner.base._training_document(case['sampling_sources']['full'],req,ROOT);fulls[cid]=full
        expected_layouts=independently_seeded_layouts(full)
        positive=[i for i,row in enumerate(full['observations']) if row['time_min']>0]
        case_layouts=[key for key,spec in layouts.items() if spec['case_id']==cid]
        require(len(case_layouts)==2,'Drug must have both prespecified layouts')
        for key in case_layouts:
            spec=layouts[key];li=spec['layout_index'];visible=expected_layouts[li]
            source=checked(spec['source']);hidden=[i for i in positive if i not in visible]
            expected=dict(visible_training=training_for(full,visible),design_only=dict(visible_ids=visible,
                candidate_ids=hidden,candidate_times_h=[full['observations'][i]['time_min']/60 for i in hidden],
                visible_times_h=[full['observations'][i]['time_min']/60 for i in visible]))
            same(source,expected,'Time-only layout '+key)
        value=checked(state['completed'][cid]);done[cid]=value
        require(value['case_id']==cid and value['protocol']==state['protocol']
                and value['locked_at']<=state['locked_at'],'Case/global lock mismatch')
        check_pins(value['output_pins']);journal=checked(value['journal'])
        require(journal['protocol']==state['protocol'],'Decision journal protocol differs')
        expected_sequences={(key,m) for key in case_layouts for m in methods}
        actual=[(s['layout'],s['method']) for s in value['sequences']]
        require(len(actual)==14 and set(actual)==expected_sequences,'Missing/duplicated policy sequence')
        used_journal={}
        for seq in value['sequences']:
            key=seq['layout'];li=layouts[key]['layout_index'];ids=list(expected_layouts[li]);method=seq['method']
            initial=checked(layouts[key]['source']);menu0=initial['design_only']
            require([s['budget'] for s in seq['stages']]==[0,1,2,3],'Missing stage budget')
            require(1<=len(seq['actions'])<=3,'Sequence action count is invalid')
            prev=seq['stages'][0]['result'];unit_for(case,prev,ids,full)
            entries.append(dict(case_id=cid,drug=case['drug'],layout=key,method=method,budget=0,result=prev))
            stopped=False
            for budget in (1,2,3):
                stage=seq['stages'][budget]
                if stopped:
                    require(stage['result'] is None and stage.get('status')=='selection_failed','Failure continuation lost')
                else:
                    require(len(seq['actions'])>=budget,'Stage lacks its action receipt')
                    source=seq['actions'][budget-1];action=checked(source)
                    relative=str(Path(source['path']).relative_to(run.relative_to(ROOT)))
                    require(journal['decisions'].get(relative)==source,'Action was not hash-committed in journal')
                    used_journal[relative]=source;action_sources[source['path']]=source
                    require(action['protocol']==state['protocol'] and action['method']==method and action['fit']==prev,
                            'Action basis does not match previous visible fit')
                    require(action['hidden_values_used'] is False and action['target_values_used'] is False,'Outcome isolation flag differs')
                    same(action['visible_ids'],sorted(ids),'Action visible IDs')
                    remaining=[i for i in menu0['candidate_ids'] if i not in ids]
                    times=[full['observations'][i]['time_min']/60 for i in remaining]
                    same(action['candidate_ids'],remaining,'Remaining action menu')
                    same(action['candidate_times_h'],times,'Remaining action times')
                    require(action['locked_at']<=value['locked_at'],'Action created after case lock')
                    if action['status']=='failed':
                        require(bool(action.get('reason') or action.get('detail')),'Failure diagnostics absent')
                        require(action['selected_id'] is None and stage['result'] is None
                                and stage.get('status')=='selection_failed','Failed acquisition was discarded or relabelled')
                        require(len(seq['actions'])==budget,'New action attempted after stopped sequence')
                        stopped=True;failure_count+=1
                    else:
                        require(action['status']=='selected','Unknown action status')
                        ix=action['selected_index']
                        require(type(ix) is int and 0<=ix<len(remaining),'Invalid chosen index')
                        require(action['selected_id']==remaining[ix] and action['selected_time_h']==times[ix],'Chosen ID/time mismatch')
                        if method.startswith('random_'):
                            seed=20261008+100*li+int(method[-1])
                            permutation=np.random.default_rng(seed).permutation(menu0['candidate_ids']).tolist()
                            same(action['permutation'],permutation,'Fixed random permutation')
                            require(action['selected_id']==next(x for x in permutation if x not in ids),'Random policy changed order')
                        elif method=='time_farthest':
                            visible_times=np.asarray([full['observations'][i]['time_min']/60 for i in ids])
                            distances=np.min(np.abs(np.log(np.asarray(times)[:,None])-np.log(visible_times[None,:])),axis=1)
                            chosen=int(np.flatnonzero(distances>=distances.max()-1e-12)[0])
                            require(ix==chosen,'Time-only policy differs from declared tie rule')
                        ids.append(action['selected_id']);require(len(ids)==4+budget,'Observation budget violated')
                        current=unit_for(case,stage['result'],ids,full)
                        path=stage['result']['path']
                        first_reveal[path]=min(first_reveal.get(path,math.inf),action['locked_at'])
                        prev=stage['result']
                entries.append(dict(case_id=cid,drug=case['drug'],layout=key,method=method,**stage))
            if not stopped:
                require(len(seq['actions'])==3,'Successful sequence lacks three actions')
        require(journal['decisions']==used_journal,'Unaccounted action journal entries')
        unit_for(case,value['full_reference'],positive,full)
        entries.append(dict(case_id=cid,drug=case['drug'],layout='full',method='full_reference',budget=None,result=value['full_reference']))
    disk={str(path.relative_to(ROOT)) for path in (run/'fits').glob('*/*/result.json')}
    require(disk==set(units),'Unreported fit artifacts or missing shared fits')
    # A cached subset may predate a later policy's action. Its original fit
    # must follow at least the earliest committed action that revealed it.
    for path,locked_at in first_reveal.items():
        if units[path]['status']=='predicted':
            require(checked(units[path]['selection'])['locked_at']>=locked_at,
                    'Augmented target prediction predates every matching acquisition lock')
    require(len(entries)==570,'Expected 560 policy stages plus ten full-information references')
    return cases,done,fulls,units,entries,action_sources,failure_count


def validate_initializations(p,cases,units,runner):
    count=0
    for source,unit in units.items():
        case=cases[unit['identity']['case_id']]
        calls=failed=0
        for formula in unit['fits']:
            fit=checked(formula['fit']);req=runner.request(case,'development',formula['formula'])
            phys=runner.prior.get_phys(p,case,'development',formula['formula'],req)
            require(fit['no_warm_start'] is True and fit['target_observations_used'] is False,'Fit used a warm start/target')
            require(fit['variant']=='blood_partition' and fit['call_limit']==p['fit_budget_per_formula'],'Fit rules changed')
            require(fit['visible_points']==len(unit['identity']['visible_ids']),'Fit visible count differs')
            same(fit['request'],req,'Fit request');same(fit['training_source'],unit['training'],'Fit training source')
            cl=runner.base.engine.clearance_inputs(req);renal=cl['renal_clearance_l_h'] or 0.;nonrenal=cl['total_clearance_l_h']-renal
            lo=np.log([1e-4/nonrenal,.1,.01,.05]);hi=np.log([10*phys['circulation']['cardiac_flow_l_h']/nonrenal,10.,100.,20.])
            expected=np.column_stack([
                np.r_[np.clip(0.,lo[0],hi[0]),np.random.default_rng(20260930).uniform(lo[0],hi[0],15)],
                np.vstack([np.zeros(2),np.random.default_rng(20260930).uniform(lo[1:3],hi[1:3],size=(15,2))]),
                np.r_[0.,np.random.default_rng(20261008).uniform(lo[3],hi[3],15)]])
            require(len(fit['starts'])==p['fit_starts']==16,'Fit lacks sixteen independent starts')
            same([s['initial_log_parameters'] for s in fit['starts']],expected.tolist(),'Independent initialization')
            same(fit['parameter_lower'],lo.tolist(),'Fit lower bounds');same(fit['parameter_upper'],hi.tolist(),'Fit upper bounds')
            same(fit['clearance_input'],cl,'Clearance input')
            require(fit['calls']==sum(s['calls'] for s in fit['starts']) and 0<=fit['calls']<=fit['call_limit'],'Call accounting differs')
            require(fit['failures']==sum(s['failures'] for s in fit['starts']),'Failed-call accounting differs')
            for i,start in enumerate(fit['starts']):
                quota=p['fit_budget_per_formula']//16+int(i<p['fit_budget_per_formula']%16)
                require(start['index']==i and start['call_limit']==quota and 0<=start['calls']<=quota,'Start budget differs')
            same(formula['best'],fit['best'],'Stored formula optimum')
            calls+=fit['calls'];failed+=fit['failures'];count+=1
        require(unit['resources']['objective_calls']==calls and unit['resources']['failed_calls']==failed,'Unit resource totals differ')
    return count


def replay_scores(run,p,cases,units,entries,runner):
    from report_blueprint_validation import Store,_case_reference,_score_prediction
    store=Store(ROOT);references={};scores={};matched=0
    for cid,case in cases.items():
        references[cid]=_case_reference(case,store)
    for key,unit in units.items():
        cid=unit['identity']['case_id'];case=cases[cid];clinical,curve,endpoints=references[cid]
        if unit['status']=='predicted':
            score,native,_=_score_prediction(case,clinical,curve,endpoints,unit,store)
            goal=runner.previous.data.model_goals(checked(case['target_contract']),native,checked(unit['request']))
            for endpoint in ('cmax','auc_inf','terminal_half_life'):
                same(goal['values'][endpoint],score['endpoints'][endpoint]['value'],'Matched clinical scoring '+endpoint)
            matched+=1
        else:
            score=dict(curve_metrics={'gmfe':None},endpoints={ep:dict(value=None,reference=v['value'],unit=v['unit'],error=None,weak_tail=None) for ep,v in endpoints.items()})
        scores[key]=score
    records=[];expected={}
    for entry in entries:
        ref=entry['result'];unit=None if ref is None else units[ref['path']]
        cid=entry['case_id'];_,curve,endpoints=references[cid]
        score=scores[ref['path']] if ref else dict(curve_metrics={'gmfe':None},endpoints={ep:dict(value=None,reference=v['value'],unit=v['unit'],error=None,weak_tail=None) for ep,v in endpoints.items()})
        common={k:entry[k] for k in ('case_id','drug','layout','method','budget')}
        common['status']=entry.get('status',unit['status'] if unit else 'failed')
        records.append(dict(common,score=score))
        rows=[dict(common,endpoint='curve',prediction=None,reference=None,error=score['curve_metrics'].get('gmfe'),unit=curve['unit'],weak_tail=None)]
        rows.extend(dict(common,endpoint=ep,prediction=v.get('value'),reference=v.get('reference'),error=v.get('error'),unit=v.get('unit'),weak_tail=v.get('weak_tail')) for ep,v in score['endpoints'].items())
        require(len(rows)==11,'Each stage must retain all eleven endpoint categories')
        for row in rows:
            key=tuple(row[k] for k in ('case_id','layout','method','budget','endpoint'))
            require(key not in expected,'Duplicate reconstructed endpoint row');expected[key]=row
    with (run/'endpoints.csv').open(newline='') as handle:
        actual=list(csv.DictReader(handle))
    require(len(actual)==len(expected)==6270,'CSV endpoint rows missing/duplicated')
    seen=set()
    for row in actual:
        budget=None if row['budget']=='' else int(row['budget'])
        key=(row['case_id'],row['layout'],row['method'],budget,row['endpoint'])
        require(key in expected and key not in seen,'Unknown or duplicated CSV endpoint row');seen.add(key)
        target=expected[key]
        for field in ('prediction','reference','error'):
            value=None if row[field]=='' else float(row[field]);same(value,target[field],'CSV '+str(key)+' '+field)
        for field in ('case_id','drug','layout','method','endpoint','status','unit','weak_tail'):
            want='' if target[field] is None else str(target[field])
            require(row[field]==want,'CSV categorical mismatch '+str(key)+' '+field)
    store.unchanged()
    return records,matched,len(expected)


def aggregates(p,records):
    clusters={c['case_id']:c['cluster_id'] for c in p['cases']};methods=p['methods'][:4]+['random']
    require(len(set(clusters.values()))==9,'Ten drugs should represent nine study clusters')
    pletz=[c['cluster_id'] for c in p['cases'] if c['drug'] in ('Ertapenem','Ceftriaxone')]
    require(len(pletz)==2 and pletz[0]==pletz[1],'The two Pletz drugs must share their study cluster')
    endpoint_names=['curve']+list(records[0]['score']['endpoints'])
    def error(row,endpoint):
        return row['score']['curve_metrics'].get('gmfe') if endpoint=='curve' else row['score']['endpoints'][endpoint]['error']
    def matching(method,budgets):
        return [r for r in records if r['budget'] in budgets and (r['method'].startswith('random_') if method=='random' else r['method']==method)]
    values={};summary={}
    for label in ('0','1','2','3','mean_acquired','full'):
        budgets=[None] if label=='full' else [1,2,3] if label=='mean_acquired' else [int(label)]
        summary[label]={};values[label]={}
        for method in (['full_reference'] if label=='full' else methods):
            rows=matching(method,budgets);summary[label][method]={};values[label][method]={}
            expected_count=1 if method=='full_reference' else len(budgets)*2*(3 if method=='random' else 1)
            for ep in endpoint_names:
                values[label][method][ep]={}
                for cid in clusters:
                    errors=[error(r,ep) for r in rows if r['case_id']==cid]
                    values[label][method][ep][cid]=float(np.exp(np.mean(np.log(errors)))) if len(errors)==expected_count and all(valid(x) for x in errors) else None
                usable={cid:v for cid,v in values[label][method][ep].items() if valid(v)}
                if ep=='tmax':
                    arr=[error(r,ep) for r in rows if error(r,ep) is not None and math.isfinite(error(r,ep))]
                    summary[label][method][ep]=dict(metric='absolute_error_h',mean=float(np.mean(arr)) if arr else None,n_scenarios=len(arr))
                else:
                    summary[label][method][ep]=dict(independent_cluster_bootstrap(usable,clusters),
                        missing_drugs=10-len(usable),within_twofold_drug_averages=sum(v<=2 for v in usable.values()))
    comparisons={}
    for label in ('1','2','3','mean_acquired'):
        comparisons[label]={}
        for ep in ('cmax','curve','auc_inf','terminal_half_life'):
            comparisons[label][ep]={method:independent_pair(values[label]['refit_goal'][ep],values[label][method][ep],clusters) for method in methods if method!='refit_goal'}
            comparisons[label][ep]['no_addition']=independent_pair(values[label]['refit_goal'][ep],values['0']['refit_goal'][ep],clusters)
    primary=comparisons['mean_acquired']['cmax']
    gates=dict(complete=all(r['status']=='predicted' for r in records),
        one_percent_better_controls=all(primary[m]['complete_pairs']==10 and primary[m]['estimate']<=.99 for m in ('time_farthest','c_optimal','random')),
        six_directional_wins=primary['time_farthest']['ratio_below_one']>=6,
        curve_no_harm=comparisons['mean_acquired']['curve']['time_farthest']['complete_pairs']==10 and comparisons['mean_acquired']['curve']['time_farthest']['estimate']<=1.01,
        secondary_no_harm=all(comparisons['mean_acquired'][ep]['no_addition']['complete_pairs']==10 and comparisons['mean_acquired'][ep]['no_addition']['estimate']<=1.05 for ep in ('auc_inf','terminal_half_life')))
    efficiency=[]
    for case in p['cases']:
        cid=case['case_id'];full=values['full']['full_reference']['cmax'][cid]
        for layout in p['layouts']:
            if p['layouts'][layout]['case_id']!=cid: continue
            for method in p['methods']:
                candidates=[r for r in records if r['case_id']==cid and r['layout']==layout and r['method']==method]
                successful=[r['budget'] for r in candidates if valid(error(r,'cmax')) and valid(full) and error(r,'cmax')<=2 and error(r,'cmax')<=1.1*full]
                efficiency.append(dict(case_id=cid,layout=layout,method=method,first_budget=min(successful) if successful else None,
                    full_information_error=full,interpretation='Retrospective criterion; not a stopping rule available before clinical scoring'))
    return summary,comparisons,gates,efficiency


def simulate_three(p,cases,done,units,runner):
    audits=[]
    # Predetermined by protocol order and budget, never chosen by FE or success.
    for case,budget in zip(p['cases'][:3],(0,1,3)):
        cid=case['case_id'];layout=next(k for k,s in p['layouts'].items() if s['case_id']==cid and s['layout_index']==0)
        sequence=next(s for s in done[cid]['sequences'] if s['layout']==layout and s['method']=='refit_goal')
        source=sequence['stages'][budget]['result']
        audit=dict(case_id=cid,layout=layout,method='refit_goal',budget=budget)
        if source is None or units[source['path']]['status']!='predicted':
            audits.append(dict(audit,status='unavailable',reason='Predetermined unit failed; not replaced by a favorable unit'));continue
        unit=units[source['path']];formula=unit['formula'];x=np.asarray(unit['parameters']['log_parameters'])
        training=checked(unit['training']);dev=runner.request(case,'development',formula)
        devphys=runner.prior.get_phys(p,case,'development',formula,dev)
        ts=np.asarray([row['time_min']/60 for row in training['observations']]);mw=training['molecular_weight_g_mol']
        ys=np.asarray([row['concentration_umol_per_l']*mw/1000 for row in training['observations']])
        development=runner.old.simulate('blood_partition',devphys,dev,ts,x)
        e=np.log(development['concentration_mg_l'])-np.log(ys)
        objective=float(np.mean(e*e)+(.1*x[1]/math.log(10))**2+(.1*x[2]/math.log(100))**2+(.1*x[3]/math.log(10))**2)
        same(objective,unit['parameters']['objective'],'Replayed visible penalized loss',rtol=2e-7,atol=2e-9)
        target=checked(unit['request']);phys=runner.prior.get_phys(p,case,'target',formula,target)
        native=checked(unit['prediction']);times=np.asarray(native['time_min'])/60
        predicted=runner.old.simulate('blood_partition',phys,target,times,x)
        replay=runner.prior.curve_document(predicted,times,target)
        same(replay['concentration'],native['concentration'],'Replayed target curve',rtol=2e-7,atol=2e-9)
        clearance=runner.base.engine.clearance_inputs(target);renal=clearance['renal_clearance_l_h'] or 0.
        component=predicted['model']['clearance_components_l_h']
        same(component['qualified_renal_fixed'],renal,'Qualified renal input preserved')
        same(component['scaled_nonrenal'],(clearance['total_clearance_l_h']-renal)*math.exp(x[0]),'Only nonrenal clearance scaled')
        same(predicted['effective_total_clearance_l_h'],renal+component['scaled_nonrenal'],'Total clearance accounting')
        for name,pred,req in (('development',development,dev),('target',predicted,target)):
            balance=pred['mass_balance'];tolerance=1e-8*max(req['dosing']['dose_mg'],1.)
            require(balance['maximum_absolute_error_mg']<=tolerance,'Replayed mass conservation failed '+name)
            require(abs(balance['final_in_body_mg']+balance['final_eliminated_mg']-balance['final_delivered_mg'])<=tolerance,'Final mass ledger differs '+name)
        goal=runner.previous.data.model_goals(checked(case['target_contract']),replay,target)
        audits.append(dict(audit,status='passed',formula=formula,visible_ids=unit['identity']['visible_ids'],
            objective=objective,matched_goals=goal['values'],renal_clearance_fixed_l_h=renal,
            mass_balance=predicted['mass_balance'],max_curve_absolute_difference=float(np.max(np.abs(np.asarray(replay['concentration'])-np.asarray(native['concentration']))))))
    return audits


def posthoc_diagnostics(p,fulls,units,action_sources):
    """Evaluate frozen engineering scenarios only after experiment completion.

    Actual added development concentrations are now used as diagnostic labels.
    No design, score, threshold, gate or model is changed by this calculation.
    Mixture quantiles integrate the predeclared median-preserving lognormal
    observation noise; they are not bounds of a calibrated clinical posterior.
    """
    from scipy.optimize import brentq
    from scipy.special import ndtr

    rows=[];local_fits=exhaustions=failed_calls=0
    for path,source in sorted(action_sources.items()):
        decision=checked(source)
        if decision['method']!='refit_goal':
            continue
        require(decision['status']=='selected','Posthoc diagnostic requires a recorded selected action')
        detail=decision['detail'];index=decision['selected_index'];chosen=detail['candidates'][index]
        basis=units[decision['fit']['path']];cid=basis['identity']['case_id'];full=fulls[cid]
        actual_row=full['observations'][decision['selected_id']]
        actual=float(actual_row['concentration_umol_per_l']*full['molecular_weight_g_mol']/1000.)
        require(valid(actual),'Added development observation is not positive and finite')
        members=detail['generating_scenarios']
        log_goals=np.asarray([member['log_goal'] for member in members],float)
        concentrations=np.asarray([member['concentration_mg_l'][index] for member in members],float)
        require(np.isfinite(log_goals).all() and np.isfinite(concentrations).all()
                and np.all(concentrations>0),'Invalid frozen generating scenarios')
        relative=detail['proportional_sd'];floor=detail['additive_floor_mg_l']
        same(relative,p['proportional_noise_sd'],'Frozen diagnostic noise assumption')
        visible=checked(basis['training'])
        visible_values=[r['concentration_umol_per_l']*visible['molecular_weight_g_mol']/1000 for r in visible['observations']]
        same(floor,p['additive_floor_fraction']*max(visible_values),'Visible-only additive noise floor')
        logs=np.log(concentrations)
        sd=np.sqrt(np.log1p(relative**2+(floor/concentrations)**2))
        require(np.isfinite(sd).all() and np.all(sd>0),'Invalid frozen lognormal noise SD')
        def cdf(log_value):
            return float(np.mean(ndtr((log_value-logs)/sd)))
        # Ten SDs beyond every component brackets the fixed .025/.975
        # quantiles without guessing a concentration-space range.
        lo=float(np.min(logs-10.*sd));hi=float(np.max(logs+10.*sd))
        quantiles=[math.exp(brentq(lambda value:cdf(value)-q,lo,hi,xtol=1e-12)) for q in (.025,.5,.975)]
        pit=cdf(math.log(actual))
        classification='below' if pit<.025 else 'above' if pit>.975 else 'inside'
        before=float(detail['initial_scenario_loss']);after=float(chosen['expected_log_endpoint_loss'])
        reduction=before-after;same(reduction,chosen['expected_reduction'],'Expected hypothetical loss reduction')
        spread=float(math.exp(np.max(log_goals)-np.min(log_goals)))
        resource=detail['resources'];local_fits+=resource['local_fits']
        exhaustions+=resource['budget_exhaustions'];failed_calls+=resource['failed_calls']
        relative_path=Path(path)
        rows.append(dict(case_id=cid,layout=relative_path.parent.parent.name,
            budget=int(relative_path.stem),selected_id=decision['selected_id'],time_h=decision['selected_time_h'],
            generating_scenarios=len(members),cmax_scenario_max_over_min=spread,
            expected_log_loss_before=before,expected_log_loss_after=after,expected_gain=reduction,
            expected_positive_gain=reduction>0.,observed_added_concentration_mg_l=actual,
            scenario_concentration_range_mg_l=[float(concentrations.min()),float(concentrations.max())],
            inside_scenario_envelope=bool(concentrations.min()<=actual<=concentrations.max()),
            engineering_mixture_q025_median_q975_mg_l=quantiles,
            engineering_mixture_cdf_at_observation=pit,engineering_interval_class=classification,
            observed_over_mixture_median=actual/quantiles[1],
            proportional_sd=relative,additive_floor_mg_l=floor))
    require(len(rows)==60,'Expected sixty endpoint-directed sequential decisions')
    spreads=np.asarray([row['cmax_scenario_max_over_min'] for row in rows])
    def counts(selected):
        return dict(decisions=len(selected),inside=sum(r['engineering_interval_class']=='inside' for r in selected),
            below=sum(r['engineering_interval_class']=='below' for r in selected),
            above=sum(r['engineering_interval_class']=='above' for r in selected),
            inside_scenario_envelope=sum(r['inside_scenario_envelope'] for r in selected))
    summary=counts(rows)
    summary.update(cmax_scenario_spread_median=float(np.median(spreads)),
        cmax_scenario_spread_q25_q75=np.quantile(spreads,[.25,.75]).tolist(),
        cmax_scenario_spread_min_max=[float(spreads.min()),float(spreads.max())],
        cmax_scenario_spread_below_1_01=int(np.sum(spreads<1.01)),
        expected_positive_gain=sum(r['expected_positive_gain'] for r in rows),
        expected_nonpositive_gain=sum(not r['expected_positive_gain'] for r in rows),
        local_fits=local_fits,local_budget_exhaustions=exhaustions,failed_local_calls=failed_calls,
        engineering_interval_inside_fraction=summary['inside']/len(rows))
    return dict(analysis_role='posthoc_diagnostics_only',computed_after_prediction_scoring=True,
        decisions_changed=False,clinical_scores_changed=False,expansion_gate_changed=False,
        summary=summary,by_budget={str(b):counts([r for r in rows if r['budget']==b]) for b in (1,2,3)},
        by_drug={cid:counts([r for r in rows if r['case_id']==cid]) for cid in fulls},decisions=rows,
        interpretation=['The target-Cmax scenario spread measures separation of a finite engineering set, not calibrated epistemic uncertainty.',
            'The interval checks the sixty selected new development concentrations against pre-action scenario-plus-noise mixtures; target clinical outcomes are not used in this diagnostic.',
            'These are sixty dependent sequential decisions in twenty layouts of ten previously viewed drugs, not sixty independent validation samples.',
            'Being inside a nominal 95% engineering interval does not establish a calibrated posterior, clinical coverage or clinical risk reliability.',
            'Positive expected gain is relative to the finite simulated scenarios and bounded hypothetical refits; it is not observed clinical improvement.',
            'Narrow scenarios or interval misses can support a mismatch diagnosis, but cannot by themselves establish the main causal reason for the selector failing its prespecified comparison.'])


def audit(run):
    run=Path(run).resolve();require(run.is_relative_to(ROOT),'Audit run outside project')
    state=read(run/'state.json')
    # This check deliberately precedes source/target imports and source reads.
    require(state.get('status')=='complete','Experiment is not complete; target outcomes were not opened')
    import run_sequential_sampling_design as runner
    require(run==runner.RUN.resolve(),'Audit applies only to this frozen sequential experiment')
    result=dict(version=VERSION,status='running',started_at=time.time(),audit_code_sha256=sha(__file__),
        protocol=state['protocol'],limitations=['Integrity audit, not a claim of algorithm superiority or clinical calibration.',
            'Frozen clinical scorer replay is shared; cluster aggregation/bootstrap and receipt traversal are independent.',
            'Ten previously viewed development drugs remain ten drugs in nine study clusters; shared fits are not additional cases.'])
    try:
        p=checked(state['protocol']);check_pins(p['source_pins']);reported=checked(state['results'])
        require(state['locked_at']<=reported['scored_at']<=state['completed_at'],'Scoring precedes prediction lock')
        cases,done,fulls,units,entries,actions,failures=validate_layouts_sequences(run,p,state,runner)
        result['source_and_sequence_checks']=dict(passed=True,source_pins=len(p['source_pins']),drugs=10,study_clusters=9,
            layouts=20,sequences=140,stage_slots=len(entries),unique_fits=len(units),committed_actions=len(actions),failed_acquisitions=failures)
        result['fit_initializations']=dict(passed=True,formula_fits=validate_initializations(p,cases,units,runner),starts_per_fit=16)
        records,matched,csvrows=replay_scores(run,p,cases,units,entries,runner)
        result['clinical_scoring_replay']=dict(passed=True,unique_matched_predictions=matched,csv_rows=csvrows,endpoint_categories=11,
            requested_goal='cmax',secondary_matched_goals=['auc_inf','terminal_half_life'])
        summary,comparisons,gates,efficiency=aggregates(p,records)
        same(reported['summary'],summary,'Independent summary')
        same(reported['comparisons'],comparisons,'Independent paired study bootstrap')
        same(reported['expansion_checks'],gates,'Independent expansion checks')
        require(reported['expand_to_twenty']==all(gates.values()),'Expansion gate conclusion differs')
        index=lambda rows:{(r['case_id'],r['layout'],r['method']):r for r in rows}
        same(index(reported['efficiency']),index(efficiency),'Independent data-efficiency thresholds')
        result['aggregate_checks']=dict(passed=True,resamples=RESAMPLES,seed=BOOTSTRAP_SEED,expansion_checks=gates,
            primary=summary['mean_acquired']['refit_goal']['cmax'],paired_primary=comparisons['mean_acquired']['cmax'])
        result['predetermined_solver_replays']=simulate_three(p,cases,done,units,runner)
        require(all(x['status']=='passed' for x in result['predetermined_solver_replays']),'A predetermined numerical replay is unavailable')
        require(reported['resources']['unique_fits']==len(units),'Unique fit count differs')
        require(reported['resources']['objective_calls']==sum(u['resources']['objective_calls'] for u in units.values()),'Actual fit calls differ')
        require(reported['resources']['actual_formula_fits']==sum(len(u['fits']) for u in units.values()),'Formula fit count differs')
        require(reported['resources']['final_target_predictions']==sum(u['resources']['prediction_calls'] for u in units.values()),'Prediction call count differs')
        decisions=[checked(source) for source in actions.values()]
        require(reported['resources']['lookahead_calls']==sum(d.get('detail',{}).get('resources',{}).get('refit_calls',0) for d in decisions),'Hypothetical refit count differs')
        for method in p['methods']:
            own=[d for d in decisions if d['method']==method]
            expected=dict(development_calls=sum(d.get('model_calls',{}).get('development',0) for d in own),
                target_calls=sum(d.get('model_calls',{}).get('target',0) for d in own),
                wall_seconds=math.fsum(d.get('wall_seconds',0.) for d in own))
            same(reported['resources']['decision_costs'][method],expected,'Decision resource costs '+method)
        result['posthoc_diagnostics']=posthoc_diagnostics(p,fulls,units,actions)
        pdf=ROOT/'results/manuscript/pbpk_paper_zh.pdf'
        if pdf.exists():
            result['paper_pdf_current_sha256']=sha(pdf)
            result['paper_hash_interpretation']='Current hash only; no pre-run PDF hash was supplied to this audit.'
        check_pins(p['source_pins'])
        require(read(run/'state.json')==state,'Experiment state changed during audit')
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    result['finished_at']=time.time()
    (run/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    try:
        result=audit(args.run)
        print(json.dumps({k:result[k] for k in ('status','error') if k in result}))
        sys.exit(0 if result['status']=='passed' else 1)
    except (ValueError,OSError,KeyError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)
