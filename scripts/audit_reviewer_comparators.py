#!/usr/bin/env python3
"""Read-only independent checks of the locked same-observation classical controls.

The sole output is comparators/audit.json. No optimizer is called. Independent
matrix exponentials check the analytic forward solution; scalar NCA and GMFE
are recalculated from actual reference curves and locked concentrations.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from scipy.stats import linregress
import blueprint_reviewer_comparators as r
from blueprint_sampling_expansion_data import _clinical, _curve_view, _curve_ng_ml


def scalar_endpoints(t,c,infusion,cutoff):
    t,c=np.asarray(t,float),np.asarray(c,float)
    area=0.
    for i in range(len(t)-1):
        a,b=c[i],c[i+1];dt=t[i+1]-t[i]
        area+=dt*(a-b)/math.log(a/b) if 0<b<a and abs(math.log(a/b))>1e-10 else dt*(a+b)/2
    peak=int(np.argmax(c));out=dict(auc=area,cmax=float(c[peak]),tmax=float(t[peak]),c_last=float(c[-1]),terminal_half_life=None)
    ids=np.flatnonzero((c>0)&(c>=cutoff)&(t>infusion)&(t>t[peak]));fits=[]
    for n in range(3,len(ids)+1):
        use=ids[-n:];reg=linregress(t[use],np.log(c[use]))
        if reg.slope<0:
            adjusted=1-(1-max(0.,min(1.,reg.rvalue**2)))*(n-1)/(n-2)
            fits.append(dict(n=n,adjusted=adjusted,half=math.log(2)/-reg.slope))
    if fits:
        best=max(f['adjusted'] for f in fits)
        out['terminal_half_life']=max([f for f in fits if f['adjusted']>=best-.0001],key=lambda f:f['n'])['half']
    return out


def independent_matrix(parameters, times, dose, infusion):
    p=np.array(parameters);n=1 if len(p)==2 else 2
    matrix=np.array([[-p[0]/p[1]]]) if n==1 else np.array([[-(p[0]+p[2])/p[1],p[2]/p[3]],[p[2]/p[1],-p[2]/p[3]]])
    out=[]
    for t in times:
        if infusion==0:
            y=expm(matrix*t)@np.r_[dose,np.zeros(n-1)]
        else:
            augmented=np.zeros((n+1,n+1));augmented[:n,:n]=matrix;augmented[0,n]=dose/infusion
            y=(expm(augmented*min(t,infusion))@np.r_[np.zeros(n),1.])[:n]
            y=expm(matrix*max(t-infusion,0.))@y
        out.append(y[0]/p[1])
    return np.array(out)


def close(x,y,label,*,rtol=1e-7):
    if x is None or y is None:
        if x is not None or y is not None:raise ValueError(label+': null mismatch')
    elif not np.isclose(x,y,rtol=rtol,atol=1e-10):
        raise ValueError(f'{label}: {x} != {y}')


def main():
    state=r.read(r.OUT/'state.json');p=r.checked(state['protocol']);results=r.checked(state['results']);r.verify(p['source_pins'])
    if state['status']!='complete' or not state['predictions_locked_at']<=results['scored_at']<=state['completed_at']:
        raise ValueError('Uncompleted experiment or invalid prediction-scoring lock')
    with (r.OUT/'endpoints.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    look={(x['case_id'],x['info'],x['method'],x['transfer'],x['endpoint']):x for x in rows}
    if len(rows)!=len(look):raise ValueError('Duplicate comparison cell')
    counts=dict(jobs=0,matrix_points=0,scalar_checks=0,curve_checks=0,subset_isolation_checks=0,objective_checks=0)
    scorecache={};boundary_counts={};convergence={};allometry={}
    for key,ref in state['completed'].items():
        doc=r.checked(ref);job=p['jobs'][key];case=p['cases'][job['case_id']]
        if doc['status']!='predicted':continue
        if doc['protocol']!=state['protocol'] or doc['parameter_locked_at']>doc['prediction_locked_at'] or doc['prediction_locked_at']>state['predictions_locked_at']:
            raise ValueError('Fit parameter/prediction lock invalid')
        train=r.checked(job['training_source']);dev=r.checked(job['development_request']);target=r.checked(job['target_request']);contract=r.checked(job['target_contract'])
        ts=np.array([x['time_min']/60 for x in train['observations']]);cs=np.array([x['concentration_umol_per_l']*train['molecular_weight_g_mol']/1000 for x in train['observations']])
        fit=doc['fit'];best=fit['selected'];model=job['model']
        if fit['initialization']['observed_times_h']!=ts.tolist() or fit['initialization']['observations']!=len(ts) or len(ts) not in (4,5):raise ValueError('Visible row isolation failed')
        counts['subset_isolation_checks']+=1
        f=r.concentration(best['parameters'],ts,**r.dosing(train))
        obj=float(np.mean(np.log(f/cs)**2))
        if model.endswith('_cl_prior'):obj+=.1*(math.log(best['parameters'][0]/job['clearance_prior']['value_L_h'])/math.log(2))**2
        close(obj,best['objective'],'fitted objective');counts['objective_checks']+=1
        finite=[a for a in fit['attempts'] if 'objective' in a]
        if len(fit['attempts'])!=16 or any(a.get('nfev',0)>400 for a in fit['attempts']):raise ValueError('Optimization cap differs')
        close(best['objective'],min(a['objective'] for a in finite),'selected minimum')
        boundary_counts.setdefault(model,[0,0]);boundary_counts[model][0]+=any(best['active_bounds']);boundary_counts[model][1]+=1
        convergence.setdefault(model,[0,0]);convergence[model][0]+=best['converged'];convergence[model][1]+=1
        # Original cohort references may share a containing document; select its exact case.
        source=r.checked(case['reference_source']);raw=_clinical(source,job['case_id']);ocurve=_curve_view(raw)
        times,observed=_curve_ng_ml(ocurve,contract['molecular_weight_g_mol']);onca=scalar_endpoints(times,observed,target['dosing']['infusion_minutes']/60,contract['cutoff_ng_ml'])
        scorecache[key]={}
        for transfer,variant in doc['predictions'].items():
            parameters=np.array(variant['parameters']);pred=r.concentration(parameters,times,**r.dosing(target))*1000
            independent=independent_matrix(parameters,times,target['dosing']['dose_mg'],target['dosing']['infusion_minutes']/60)*1000
            np.testing.assert_allclose(pred,independent,rtol=5e-7,atol=1e-9)
            counts['matrix_points']+=len(times)
            curve=variant['prediction'];interp=np.interp(times,np.asarray(curve['time_min'])/60,np.asarray(curve['concentration'])*contract['molecular_weight_g_mol'])
            np.testing.assert_allclose(pred,interp,rtol=1e-8,atol=1e-9)
            pnca=scalar_endpoints(times,pred,target['dosing']['infusion_minutes']/60,contract['cutoff_ng_ml'])
            positives=np.array(observed)>0
            gmfe=float(np.exp(np.mean(np.abs(np.log(pred[positives]/np.array(observed)[positives])))))
            scorecache[key][transfer]=dict(prediction=pnca,observed=onca,curve=gmfe)
        counts['jobs']+=1
    for cell in p['cells']:
        if cell['job'] not in scorecache:continue
        job=p['jobs'][cell['job']]
        a=r.checked(cell['training_source'])['observations'];b=r.checked(job['training_source'])['observations']
        if a!=b:raise ValueError('Deduplicated fit changed its visible concentration rows')
        for transfer,data in scorecache[cell['job']].items():
            row=look[cell['case_id'],cell['info'],cell['method'],transfer,'curve'];close(float(row['error']),data['curve'],'curve GMFE');counts['curve_checks']+=1
            for e in ('auc','cmax','tmax','c_last','terminal_half_life'):
                row=look[cell['case_id'],cell['info'],cell['method'],transfer,e]
                pp,oo=data['prediction'][e],data['observed'][e]
                close(float(row['prediction']) if row['prediction'] else None,pp,'predicted '+e)
                close(float(row['reference']) if row['reference'] else None,oo,'observed '+e)
                error=None if pp is None or oo is None else abs(pp-oo) if e=='tmax' else max(pp/oo,oo/pp)
                close(float(row['error']) if row['error'] else None,error,'error '+e);counts['scalar_checks']+=3
    for cohort in ('all_thirty','added_twenty'):
        ids={cid for cid,c in p['cases'].items() if cohort=='all_thirty' or c['cohort']=='added_twenty'}
        allometry[cohort]={}
        for info in r.INFO:
            allometry[cohort][info]={}
            for model in r.MODELS:
                allometry[cohort][info][model]={}
                for transfer in ('linear_weight','allometric_sensitivity'):
                    allometry[cohort][info][model][transfer]={}
                    for ep in ('curve','cmax'):
                        values={x['case_id']:float(x['error']) for x in rows if x['case_id'] in ids and x['info']==info and x['method']==model and x['transfer']==transfer and x['endpoint']==ep and x['error']}
                        stat=r.bootstrap(values,{k:p['cases'][k]['cluster_id'] for k in values},seed=r.SEED,resamples=2000)
                        allometry[cohort][info][model][transfer][ep]=stat
                        if transfer=='linear_weight':close(stat['estimate'],results['summary'][cohort][info]['methods'][model][ep]['estimate'],'aggregate GMFE')
    out=dict(status='passed',audited_at=time.time(),audit_code=r.ref(Path(__file__)),results=state['results'],counts=counts,
        numerical_method='Independent matrix-exponential curves, regression-tail suffixes, direct linear-up/log-down NCA and GMFE',
        boundary_hit_counts=boundary_counts,selected_start_convergence=convergence,transfer_sensitivity=allometry,
        limitations=['Boundary hits and multistart convergence are not proof of parameter identifiability.',
            'Old PBPK values copied without refitting; this audit does not assert clinical validity of disputed source metadata.'])
    r.save(r.OUT/'audit.json',out);print(out['status'],counts)

if __name__=='__main__':main()
