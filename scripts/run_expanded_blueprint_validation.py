#!/usr/bin/env python3
"""Validate the selected PBPK recipe on old and additional clinical compounds.

This runner never trains, fits clinical curves, selects an improved candidate,
or changes the manuscript. The union-excluded parameter artifact is supplied
separately; all predictions are locked before clinical observations are parsed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_joint_evidence_refinement as execution
from pbpk_kb.blueprint_learning.clinical_shared_prior import _digest as training_digest, normalize_structure
from pbpk_kb.blueprints.source_workflow import save, sha256

value = execution.value
retained = execution.retained
clone = execution.clone
VERSION = 'expanded-blueprint-validation-1'
PRIOR_VERSION = 'expanded-selected-parameter-prior-1'
PREDICTOR = 'expanded_fixed_selected_parameter_prior'
CANDIDATE = 'expanded_selected'
DEFAULT = ROOT/'experiments/blueprint_design/expanded_validation_20261007'
SELECTED = ROOT/'results/data/blueprint_selected.json'
CONDITIONS = retained.CONDITIONS
SEED, RESAMPLES = 20261007, 2000
PARAMETER_UNITS = {'total_clearance':'ml/min/kg', 'fraction_unbound':'fraction',
                   'vss':'l/kg', 'terminal_half_life':'h'}


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT:
        raise ValueError('Use only expanded_validation_20261007')
    return path


def merge_pins(destination,incoming):
    for name,digest in incoming.items():
        if name in destination and destination[name]!=digest:
            raise ValueError('Conflicting frozen source hashes: '+name)
        if sha256(value.inside(name,ROOT,exists=True))!=digest:
            raise ValueError('Frozen source changed: '+name)
        destination[name]=digest


def checked_prior(path, case_ids, selected):
    receipt = value.old.ref(value.inside(path, ROOT, exists=True))
    prior = value.old.checked(receipt)
    if prior.get('version')!=PRIOR_VERSION or prior.get('result_sha256')!=training_digest(
            {k:v for k,v in prior.items() if k!='result_sha256'}):
        raise ValueError('Parameter prior version or checksum differs')
    if prior.get('mode') not in {'refit_union_exclusion','reuse_frozen_weights'}:
        raise ValueError('Explicit union-refit or verified frozen-weight inference mode required')
    if not set(case_ids)<=set(prior['parameters']) or not set(case_ids)<=set(prior['per_compound_audit']):
        raise ValueError('Every clinical compound needs a certified prior; screened-out targets may remain excluded')
    for flag in ('target_labels_used','target_observation_files_opened','training_structure_overlap','training_generic_scaffold_overlap'):
        if prior.get('audit',{}).get(flag) is not False:
            raise ValueError('Training isolation audit missing: '+flag)
    for cid in case_ids:
        audit = prior['per_compound_audit'][cid]
        if any(audit.get(k) is not False for k in ('human_training_overlap','animal_training_overlap')):
            raise ValueError('Human/animal target overlap remains: '+cid)
        for name,unit in PARAMETER_UNITS.items():
            if prior.get('units',{}).get(name,'').lower()!=unit:
                raise ValueError('Parameter unit differs: '+name)
            val = prior['parameters'][cid][name]
            if not value.scores.positive(val) or (name=='fraction_unbound' and val>1):
                raise ValueError('Invalid parameter prediction: '+cid+'/'+name)
    for name,digest in {**prior['source_refs'],**prior['code_pins']}.items():
        if sha256(value.inside(name,ROOT,exists=True))!=digest:
            raise ValueError('Frozen parameter dependency changed: '+name)
    value.old.collect_refs(prior['model_receipts'],{})
    if prior.get('selected_receipt') != value.old.ref(SELECTED):
        raise ValueError('Parameter model does not bind the selected research version')
    if prior['mode']=='reuse_frozen_weights':
        original = value.old.checked(selected['core']['parameter_prior'])
        if (prior['model_receipts']['shared']['sha256']!=original['model_sha256']
                or prior['model_receipts']['complementary']!=selected['clearance_model']):
            raise ValueError('A cache reuse claim requires the exact selected model weights')
    return prior,receipt


def parameter_record(cid, name, val, source):
    return {'parameter':name, 'compound_id':cid, 'value':float(val), 'unit':PARAMETER_UNITS[name],
        'basis':'predicted', 'source':{**source,'location':cid+'/'+name},
        'context':{'species':'human','matrix':'plasma','route':'intravenous',
                   'study_role':'independent_parameters','independent_of_target':True},
        'prediction_method':{'name':PREDICTOR,'version':PRIOR_VERSION,'training_overlap':False,
            'applicability':{'human_IV_parameter_model':True,'target_structure_and_scaffold_excluded':True}},
        'notes':'Fixed selected recipe; all union target identities and scaffolds excluded; no clinical curve fitting'}


def replace_parameters(request, sidecars, parameters, receipt):
    request,sidecars = deepcopy(request),deepcopy(sidecars)
    for name in ('total_clearance','fraction_unbound'):
        found = [r for r in request['evidence'] if r['parameter']==name]
        if len(found)>1:
            raise ValueError('Only one already-selected construction parameter is allowed')
        if not found or found[0]['basis']=='predicted':
            if name=='total_clearance' and any(r['parameter'] in {'renal_clearance','hepatic_clearance'} for r in request['evidence']):
                raise ValueError('Do not replace predicted total CL while retaining an organ split')
            request['evidence']=[r for r in request['evidence'] if r['parameter']!=name]
            request['evidence'].append(parameter_record(request['compound_id'],name,parameters[name],receipt))
    for key,name in (('vss','vss'),('half','terminal_half_life')):
        if not any(r['basis']!='predicted' for r in sidecars.get(key,[])):
            sidecars[key]=[parameter_record(request['compound_id'],name,parameters[name],receipt)]
    return request,sidecars


def qualified(record):
    method=record.get('prediction_method',{}); app=method.get('applicability',{})
    return (record.get('basis')=='predicted' and method.get('name')==PREDICTOR
        and method.get('version')==PRIOR_VERSION and method.get('training_overlap') is False
        and app.get('human_IV_parameter_model') is True and app.get('target_structure_and_scaffold_excluded') is True)


def input_policy(request,fallback,*,purpose=None):
    return clone(value.policy,rich_qualified=qualified)(request,fallback,purpose=purpose)


def select_request(base,independent,condition,scope_source):
    if condition=='all_ml':
        return deepcopy(base),{'status':'ready','selected_evidence':deepcopy(base['evidence'])}
    return clone(value.select_request,policy=input_policy)(base,independent,condition,scope_source=scope_source)


def complete_cache_match(original,parameters,mode):
    """Reuse the entire old input only with identical weights and values."""
    if mode!='reuse_frozen_weights':
        return False
    records=original['request']['evidence']+original['sidecars']['vss']+original['sidecars']['half']
    for row in records:
        name=row['parameter']
        if row['basis']=='predicted' and name in PARAMETER_UNITS:
            if row['unit'].lower()!=PARAMETER_UNITS[name] or row['value']!=parameters[name]:
                return False
    return True


def _metadata_case(case,chemical,assumptions_ref):
    case=deepcopy(case)
    if set(case)&{'observations','curve','clinical_reference','reference_values'}:
        raise ValueError('Cohort preparation metadata must not contain observations')
    for name in ('case_id','drug','study_id','cluster_id','human','dosing','target','reference_locator','nca_policy'):
        if name not in case:
            raise ValueError('New cohort metadata missing '+name)
    if chemical.get('construction_status') not in {'ready','blocked'}:
        raise ValueError('Chemical construction status must be explicit')
    case.update(compound_id=chemical['compound_id'],smiles=chemical['smiles'],
        construction_status=chemical['construction_status'],construction_missing_reasons=chemical['missing_reasons'],
        independent_evidence=deepcopy(chemical.get('independent_evidence',[])),independent_evidence_reviewed=True,
        evidence_review_scope='Curated source and construction suitability; no claim that unknown underlying clinical studies are independent of the evaluation study')
    common=[('solubility',1.,'mg/ml'),('solubility_ph',7.4,'1'),('binding_partner','Albumin','text'),
            ('elimination_scope','hepatic_only','text')]
    evidence=deepcopy(chemical['evidence'])
    if any(r['parameter'] not in {'molecular_weight','molecular_formula','logp','pka_types'} for r in evidence):
        raise ValueError('Only independently sourced molecular identity/properties belong in chemical evidence')
    for name,val,unit in common:
        evidence.append({'parameter':name,'compound_id':case['compound_id'],'value':val,'unit':unit,'basis':'assumption',
            'source':{**assumptions_ref,'location':'Frozen common dissolved-IV template assumptions'},
            'notes':'Common IV template convention, not drug-specific measured evidence or demonstrated elimination mechanism'})
    target={**deepcopy(case['target']),'endpoints':['plasma_concentration_time',*value.UNITS],
            'primary_endpoint':'plasma_concentration_time'}
    request={'entry_reason':'model_missing','compound_id':case['compound_id'],'compound_name':case['drug'],
        'human':deepcopy(case['human']),'dosing':deepcopy(case['dosing']),'target':deepcopy(case['target']),
        'partition_method':'PKSimStandard','evidence':evidence,'forbidden_sources':[],
        'evidence_ledger':{**assumptions_ref,'location':'Temporary schema binding; exact evidence ledger written before native assembly'},
        'assumptions':['Fixed dissolved-IV construction conventions; no clinical outcomes, PK-Sim fitted drug parameters or clinical curve fit.']}
    request['target']=target
    case['request_template']=request
    if request['compound_id']!=case['compound_id'] or 'terminal_fit' in request['target']:
        raise ValueError('New request identity or outcome-derived terminal-fit input differs')
    locator=case['reference_locator']; ref=locator['source']
    case.update(cohort='expanded',dataset=case.get('dataset','osp_observed_expansion'),reference_kind='observed_curve',
        reference_source={**ref,'expanded_case_id':case['case_id'],'location':'Frozen raw observed-data locator'},
        forbidden_clinical_sources=[*case.get('forbidden_clinical_sources',[]),ref])
    return case


def prepare(run=DEFAULT,cohort_path=None,prior_path=None,workers=14,chemical_path=None):
    run=own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Protocol is frozen; resume with run')
    if type(workers) is not int or not 1<=workers<=14:
        raise ValueError('workers must be 1..14')
    selected=value.old.read(SELECTED); selected_ref=value.old.ref(SELECTED)
    source=value.old.checked(selected['protocol']);value.verify(source)
    source_dir=value.inside(selected['protocol']['path'],ROOT).parent
    source_state=value.old.read(source_dir/'state.json')
    if source_state['status']!='complete' or not source_state['all_predictions_locked_before_scoring']:
        raise ValueError('Completed selected baseline required')
    cohort_ref=value.old.ref(value.inside(cohort_path or run/'cohort.json',ROOT,exists=True))
    metadata=value.old.checked(cohort_ref)
    chemical_ref=value.old.ref(value.inside(chemical_path or run/'chemical_sources.json',ROOT,exists=True))
    chemicals=value.old.checked(chemical_ref)
    for name,digest in chemicals['source_refs'].items():
        if sha256(value.inside(name,ROOT,exists=True))!=digest:raise ValueError('Chemical source changed: '+name)
    run.mkdir(parents=True,exist_ok=True);(run/'inputs').mkdir(exist_ok=True)
    assumptions=run/'inputs/selection_rules.json'
    save(assumptions,{'version':VERSION,'policy':'Independent CL/fu first; missing CL removes total, renal and hepatic evidence together. A qualified net renal split sets hepatic_and_renal; otherwise hepatic_only remains a construction assumption. No calibration.',
        'common_IV_assumptions':{'solubility_mg_ml':1.,'solubility_pH':7.4,'binding_partner':'Albumin','elimination_scope':'hepatic_only'},
        'basis':'Previously used standard dissolved-IV template; placeholders and binding/allocation assumptions are not measured properties of added compounds.'})
    assumptions_ref=value.old.ref(assumptions)
    added=[_metadata_case(c,chemicals['compounds'][c['drug']],assumptions_ref) for c in metadata['cases']]
    if not added:
        raise ValueError('At least one clinically eligible additional compound is required')
    cases=deepcopy(source['cases'])
    for c in cases:c['validation_stratum']='old_20'
    old_ids={c['compound_id'] for c in cases}
    if old_ids&{c['compound_id'] for c in added}:
        raise ValueError('Additional compounds must differ from the old cohort')
    for c in added:c['validation_stratum']='new_compounds'
    cases+=added
    if len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Case identifiers must be unique')
    prior,prior_ref=checked_prior(prior_path or run/'prior/predictions.json',{c['compound_id'] for c in cases},selected)
    for case in cases:
        if normalize_structure(case['smiles'])['identity_smiles']!=prior['per_compound_audit'][case['compound_id']]['normalized_identity']:
            raise ValueError('Clinical chemical identity differs from parameter inference: '+case['case_id'])
    pins=deepcopy(source['source_pins']);code=deepcopy(source['code_pins'])
    for ref in (selected_ref,selected['protocol'],selected['results'],cohort_ref,chemical_ref,assumptions_ref,prior_ref):
        if sha256(value.inside(ref['path'],ROOT,exists=True))!=ref['sha256']:raise ValueError('Source changed')
        pins[ref['path']]=ref['sha256']  # Results are never decoded during preparation.
    for incoming in (prior['source_refs'],chemicals['source_refs'],metadata.get('source_pins',{})):
        merge_pins(pins,incoming)
    for incoming in (prior['code_pins'],metadata.get('code_pins',{})):
        merge_pins(code,incoming)
    if chemicals.get('code_ref'):
        ref=chemicals['code_ref'];merge_pins(code,{ref['path']:ref['sha256']})
    value.old.collect_refs(prior['model_receipts'],pins)
    for path in (Path(__file__),Path(execution.__file__),Path(execution.separate.__file__)):
        code[str(path.resolve().relative_to(ROOT))]=sha256(path)
    scope_source={**assumptions_ref,'location':'Explicit rule-derived organ allocation, not observed organ mechanism'}
    pins[scope_source['path']]=scope_source['sha256']
    common_core={**deepcopy(source['core']),'parameter_prior':prior_ref}
    tasks,units,baselines={},[],{}
    def add_task(case,condition,request,sidecars,core,reuse=None,blocked=None):
        key=retained.key_for(request,sidecars,core)
        if key not in tasks:
            task={'key':key,'case_id':case['case_id'],'request':request,'sidecars':sidecars,'core':core,
                'reuse_value_receipt':reuse,'native_source_ref':None,'candidate':CANDIDATE,'construction_blocked':blocked}
            if reuse is None:
                ledger=run/'inputs'/(key[:20]+'.json');save(ledger,{'evidence':request['evidence']})
                task['request']['evidence_ledger']={**value.old.ref(ledger),'location':'Exact expanded validation construction inputs'}
                pins[str(ledger.relative_to(ROOT))]=sha256(ledger)
                value.model._check_request(request)
                for rows in sidecars.values():value.model._check_sidecars(request,rows,ROOT)
                if not blocked:
                    built=value.model.assemble_evidence_blueprint(request,ROOT)
                    if built['status']!='assembled':raise ValueError(case['case_id']+': '+str(built.get('gaps')))
            else:
                item=value.old.checked(reuse);value.old.collect_refs(item,pins);pins[reuse['path']]=reuse['sha256']
                if item['entry']['request']!=request or retained.key_for(item['entry']['request'],sidecars,core)!=key:
                    raise ValueError('Cached scientific inputs are not identical')
            tasks[key]=task
        elif tasks[key]['request']!=request and value.semantic_request(tasks[key]['request'])!=value.semantic_request(request):
            raise ValueError('Prediction fingerprint collision')
        units.append({'case_id':case['case_id'],'condition':condition,'key':key})
        if condition=='all_ml':baselines[case['case_id']]={'fingerprint':key}
    for case in cases:
        cid=case['compound_id'];params=prior['parameters'][cid]
        value.old.collect_refs(case,pins)
        if case['validation_stratum']=='old_20':
            old_units={u['condition']:u['key'] for u in source['units'] if u['case_id']==case['case_id']}
            for condition in CONDITIONS:
                oldkey=old_units[condition];original=source['tasks'][oldkey]
                if complete_cache_match(original,params,prior['mode']):
                    add_task(case,condition,deepcopy(original['request']),deepcopy(original['sidecars']),
                             deepcopy(original['core']),source_state['outcomes'][oldkey])
                else:
                    request,sidecars=replace_parameters(original['request'],original['sidecars'],params,prior_ref)
                    add_task(case,condition,request,sidecars,deepcopy(common_core))
        else:
            base=deepcopy(case['request_template'])
            independent=deepcopy(case.get('independent_evidence',[]))
            if any(r['basis'] not in {'measured','derived'} or r['parameter'] not in value.ALLOW for r in independent):
                raise ValueError('New independent inputs require reviewed CL/fu/renal evidence')
            if independent and case.get('independent_evidence_reviewed') is not True:
                raise ValueError('Independent new sources must be reviewed before preparation')
            base['evidence']=[r for r in base['evidence'] if r['parameter'] not in value.CLEARANCES|{'fraction_unbound'}]
            base['forbidden_sources']=[{**r,'location':r.get('location','Sealed clinical observations')} for r in case['forbidden_clinical_sources']]
            base['partition_method']='PKSimStandard'
            base,sidecars=replace_parameters(base,{'vss':[],'half':[]},params,prior_ref)
            for condition in CONDITIONS:
                blocked=case['construction_missing_reasons'] if case['construction_status']=='blocked' else None
                if blocked:
                    # Clinically eligible but chemically unsupported requests
                    # remain attempts; do not run an invalid native construction.
                    request=deepcopy(base)
                else:
                    request,decision=select_request(base,independent,condition,scope_source)
                    if decision['status']!='ready':raise ValueError(case['case_id']+'/'+condition+': '+str(decision['reasons']))
                add_task(case,condition,request,deepcopy(sidecars),deepcopy(common_core),blocked=blocked)
    expected={(c['case_id'],cond) for c in cases for cond in CONDITIONS}
    if len(units)!=len(expected) or {(u['case_id'],u['condition']) for u in units}!=expected:
        raise ValueError('Incomplete case/condition matrix')
    changed=sorted({u['case_id'] for u in units if u['condition']=='available' and u['key']!=baselines[u['case_id']]['fingerprint']})
    protocol={'version':VERSION,'created_at':value.old.now(),'selected_receipt':selected_ref,
        'source_protocol':selected['protocol'],'historical_results':selected['results'],'cohort_metadata':cohort_ref,'chemical_sources':chemical_ref,
        'parameter_prior':prior_ref,'parameter_mode':prior['mode'],'parameter_target_count':len(prior['parameters']),
        'cases':cases,'tasks':tasks,'units':units,'baselines':baselines,'changed_case_ids':changed,
        'source_pins':pins,'code_pins':code,'paper_guard':{},'publication_snapshot':value.old.paper_pins(),
        'workers':workers,'native_threads':2,'native_timeout_seconds':180,'directory_limit_bytes':10*1024**3,
        'monitor_seconds':1200,'new_unique_predictions':sum(t['reuse_value_receipt'] is None for t in tasks.values()),
        'expected_new_native_calls':sum(t['reuse_value_receipt'] is None and not t['construction_blocked'] for t in tasks.values()),
        'seed':SEED,'bootstrap_resamples':RESAMPLES,'logical_comparisons':len(units),
        'endpoints':['plasma_concentration_time',*value.UNITS], 'no_clinical_curve_fitting':True,
        'limitations':['Fixed selected recipe; no architecture, hyperparameter or clinical score based selection',
            'Old twenty compounds are previously viewed development data; new-compound source and training exclusions are reported separately',
            'All scored strata use the same frozen parameter weights; historical selected results are not pooled',
            'Source-reported scalar endpoints and finite-window curve-derived endpoints remain separate',
            'Added general drug-binding sources may have unknown underlying-study overlap and are not target-patient binding truth; original evidence contexts remain unchanged',
            'No clinical safety threshold or publication decision is inferred from this expansion']}
    value.verify(protocol,publication=True);save(run/'protocol.json',protocol)
    save(run/'state.json',{'version':VERSION,'protocol':value.old.ref(run/'protocol.json'),'status':'prepared',
        'created_at':value.old.now(),'outcomes':{},'all_predictions_locked_before_scoring':False})
    return {'status':'prepared','clinical_compounds':len({c['compound_id'] for c in cases}),'clinical_cases':len(cases),
        'new_cases':len(added),'parameter_exclusion_targets':len(prior['parameters']),'logical_units':len(units),
        'unique_predictions':len(tasks),'new_native_calls_expected':protocol['expected_new_native_calls'],'parameter_mode':prior['mode']}


class ClinicalStore(value.scores.Store):
    """Raw clinical columns are decoded only inside the post-lock report."""
    def __init__(self,root,cases):
        super().__init__(root);self.cases={c['case_id']:c for c in cases};self.curves={}

    def ref(self,reference):
        cid=reference.get('expanded_case_id')
        if cid is None:return super().ref(reference)
        if cid not in self.curves:
            case=self.cases[cid];locator=case['reference_locator']
            if any(reference.get(k)!=locator['source'][k] for k in ('path','sha256')):
                raise ValueError('Expanded clinical locator identity differs')
            raw=super().ref(locator['source'])
            self.curves[cid]=decode_curve(case,raw)
        return self.curves[cid]


def decode_curve(case,snapshot):
    locator=case['reference_locator'];observed=snapshot['ObservedData'][locator['observation_index']]
    column=observed['Columns'][locator['column_index']];grid=observed['BaseGrid']
    if observed.get('Name')!=locator['observation_name'] or column['QuantityInfo']['Path']!=locator['quantity_path']:
        raise ValueError('Frozen observation identity differs')
    if grid['Unit']!=locator['time_unit'] or column['Unit']!=locator['concentration_unit']:
        raise ValueError('Frozen clinical units differ')
    if 'concentration_dimension' in locator and column.get('Dimension')!=locator['concentration_dimension']:
        raise ValueError('Frozen clinical concentration dimension differs')
    request=case['request_template']; mwrows=[r for r in request['evidence'] if r['parameter']=='molecular_weight']
    if len(mwrows)!=1 or mwrows[0]['unit']!='g/mol':raise ValueError('Independent molecular weight required')
    mw=mwrows[0]['value'];unit=column['Unit'].lower().replace('μ','µ').replace(' ','')
    factors={'mg/l':1000/mw,'µg/ml':1000/mw,'ug/ml':1000/mw,'µg/l':1/mw,'ug/l':1/mw,
             'ng/ml':1/mw,'ng/l':.001/mw,'µmol/l':1.,'umol/l':1.,'nmol/l':.001,'mol/l':1e6}
    times={'h':60.,'min':1.}
    if unit not in factors or grid['Unit'] not in times:raise ValueError('Unsupported clinical unit conversion')
    t=np.asarray(grid['Values'],dtype=float)*times[grid['Unit']]
    y=np.asarray(column['Values'],dtype=float)*factors[unit]
    if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0) or np.any(t<0) or np.any(y<0):
        raise ValueError('Clinical times or concentrations are invalid')
    schedule=case['nca_policy']['sampling_time_h']
    if len(schedule)!=len(t) or not np.allclose(np.array(schedule)*60,t,rtol=1e-12,atol=1e-12):
        raise ValueError('Clinical schedule differs from frozen metadata')
    return {'compound_id':case['compound_id'],'study_id':case['study_id'],'role':'historical_cross_study_reference',
        'matrix':'plasma','unit':'umol/l','time_unit':'min','human':deepcopy(request['human']),
        'dosing':deepcopy(request['dosing']),'molecular_weight_g_mol':mw,
        'observations':[{'time_min':float(a),'concentration_umol_per_l':float(b)} for a,b in zip(t,y)],
        'raw_source':locator['source'],'interpretation':'Frozen concentration column; no related SD columns or reported scalar labels inferred'}


def geometric(values):
    values=list(values)
    return math.exp(math.fsum(math.log(x) for x in values)/len(values)) if values else None


def drug_equal(values,cases,*,resamples=RESAMPLES,seed=SEED):
    """First average log error within drug, then across drugs; bootstrap studies."""
    mapping={c['case_id']:c for c in cases};valid={k:v for k,v in values.items() if value.scores.positive(v)}
    if not set(valid)<=set(mapping):raise ValueError('Unknown scored case')
    druglogs=defaultdict(list); clusters=defaultdict(list)
    for key,val in valid.items():
        druglogs[mapping[key]['compound_id']].append(math.log(val));clusters[mapping[key]['cluster_id']].append(key)
    perdrug={cid:math.exp(math.fsum(v)/len(v)) for cid,v in druglogs.items()}
    result={'estimate':geometric(perdrug.values()),'per_drug':perdrug,'n_cases':len(valid),'n_drugs':len(perdrug),
        'requested_cases':len(cases),'requested_drugs':len({c['compound_id'] for c in cases}),
        'n_clusters':len(clusters),'ci95':None,
        'weighting':'Equal drug means of log errors; fixed inverse study-count case weights; bootstrap cluster multiplicity retained'}
    if len(clusters)>1:
        names=sorted(clusters)
        weights={key:1/len(druglogs[mapping[key]['compound_id']]) for key in valid}
        weighted_sums=np.array([math.fsum(weights[k]*math.log(valid[k]) for k in clusters[name]) for name in names])
        weight_sums=np.array([math.fsum(weights[k] for k in clusters[name]) for name in names])
        indices=np.random.default_rng(seed).integers(0,len(names),size=(resamples,len(names)))
        draws=np.exp(weighted_sums[indices].sum(axis=1)/weight_sums[indices].sum(axis=1))
        result.update(ci95=[float(v) for v in np.quantile(draws,[.025,.975])],resamples=resamples,seed=seed)
    return result


def paired(numerator,denominator,cases):
    valid={c['case_id']:numerator[c['case_id']]/denominator[c['case_id']] for c in cases
        if value.scores.positive(numerator.get(c['case_id'])) and value.scores.positive(denominator.get(c['case_id']))}
    result=drug_equal(valid,cases)
    return {**result,'per_case_ratio':valid,'complete_pairs':len(valid),
        'meaningful_improved_drugs_5pct':sum(v<=.95+1e-12 for v in result['per_drug'].values()),
        'meaningful_worsened_drugs_5pct':sum(v>=1.05-1e-12 for v in result['per_drug'].values()),
        'ci_condition':'Complete paired cases; failures remain in requested denominators'}


def summarize(rows,protocol):
    cases=protocol['cases'];groups={'all':cases,'old_20':[c for c in cases if c['validation_stratum']=='old_20'],
        'new_compounds':[c for c in cases if c['validation_stratum']=='new_compounds'],
        'evidence_changed':[c for c in cases if c['case_id'] in protocol['changed_case_ids']]}
    summaries,pairs,missing=[],{},[]
    for name,group in groups.items():
        ids={c['case_id'] for c in group};values={}
        for condition in CONDITIONS:
            selected=[r for r in rows if r['case_id'] in ids and r['condition']==condition]
            values[condition]={r['case_id']:r['curve_metrics'].get('gmfe') for r in selected}
            curve=drug_equal(values[condition],group)
            curve['threshold_statistics']=value.evaluation.fold_statistics(curve['per_drug'].values(),
                attempted=curve['requested_drugs'],reference_available=curve['requested_drugs'])
            endpoints={}
            for family in value.evaluation.REFERENCE_FAMILIES:
                familyrows=[r for r in selected if r['reference_family']==family]
                if not familyrows:continue
                endpoints[family]={}
                for endpoint in value.UNITS:
                    items=[r['endpoints'][endpoint] for r in familyrows]
                    errors=[x['error'] for x in items if x['error'] is not None]
                    stats={'scored':len(errors),'reference_available':sum(x['reference'] is not None for x in items),'attempted':len(items)}
                    if endpoint=='tmax':
                        perdrug=defaultdict(list)
                        for row in familyrows:
                            err=row['endpoints'][endpoint]['error']
                            if err is not None:
                                cid=next(c['compound_id'] for c in group if c['case_id']==row['case_id'])
                                perdrug[cid].append(err)
                        stats['case_weighted_mae_h']=math.fsum(errors)/len(errors) if errors else None
                        stats['mae_h']=math.fsum(math.fsum(v)/len(v) for v in perdrug.values())/len(perdrug) if perdrug else None
                        stats['scored_drugs']=len(perdrug)
                    else:
                        stats=value.evaluation.fold_statistics(errors,attempted=len(items),reference_available=stats['reference_available'])
                        stats['drug_equal']=drug_equal({r['case_id']:r['endpoints'][endpoint]['error'] for r in familyrows},
                            [c for c in group if c['case_id'] in {r['case_id'] for r in familyrows}])
                        stats['case_weighted_gmfe']=stats['gmfe'];stats['gmfe']=stats['drug_equal']['estimate']
                    stats['weak_scored']=sum(x.get('weak_tail',False) and x['error'] is not None for x in items)
                    endpoints[family][endpoint]=stats
            summaries.append({'group':name,'condition':condition,'attempted':len(selected),
                'failed':sum(r['attempt_status']=='failed' for r in selected),'scored':curve['n_cases'],
                'curve':curve,'endpoints':endpoints})
        pairs[name]=paired(values['available'],values['all_ml'],group)
        for condition in CONDITIONS[2:]:
            relevant=[r for r in rows if r['case_id'] in ids and r['condition']==condition and r['changed_from_nominal']]
            changed={r['case_id'] for r in relevant};actual=[c for c in group if c['case_id'] in changed]
            missing.append({'group':name,'condition':condition,'actually_changed_cases':len(actual),
                'actually_changed_drugs':len({c['compound_id'] for c in actual}),
                'masked_over_available_error':paired(values[condition],values['available'],actual),
                'prediction_drift':drug_equal({r['case_id']:r['prediction_drift']['curve_geometric_fold_change'] for r in relevant},actual),
                'not_clinical_safety_claim':True})
    return summaries,pairs,missing


def report(run=DEFAULT):
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.verify(protocol)
        if not state.get('all_predictions_locked_before_scoring') or set(state['outcomes'])!=set(protocol['tasks']):
            raise ValueError('Lock every prediction/failure before clinical observations are opened')
        loaded={}
        for key,receipt in state['outcomes'].items():
            item=value.old.checked(receipt);task=protocol['tasks'][key];entry=item['entry']
            if (item['key']!=key or not item.get('locked_at') or entry.get('target_observations_used') is not False
                    or entry['status'] not in {'predicted','failed'} or entry['request']!=task['request']
                    or retained.key_for(entry['request'],task['sidecars'],task['core'])!=key):
                raise ValueError('Prediction input/identity/lock differs')
            value.old.collect_refs(item,{})
            if entry['status']=='predicted':value._native_fields(entry)
            loaded[key]=item
        store=ClinicalStore(ROOT,protocol['cases']);rows=[]
        for case in protocol['cases']:
            clinical,curve,truth=value.evaluation.clinical_truth(case,store)
            units=[u for u in protocol['units'] if u['case_id']==case['case_id']]
            nominal=next(u['key'] for u in units if u['condition']=='available');base=protocol['baselines'][case['case_id']]['fingerprint']
            scored={}
            for unit in units:
                key=unit['key'];item=loaded[key];entry=item['entry']
                if key not in scored:
                    scored[key]=value.score_one(case,state['outcomes'][key],entry,store,clinical,curve,truth)
                row=deepcopy(scored[key]);row.update(condition=unit['condition'],method=unit['condition'],
                    validation_stratum=case['validation_stratum'],fingerprint=key,prediction_reused=item['reused'],
                    historical_development_used=case.get('historical_development_used',case['validation_stratum']=='old_20'),
                    qualification_flags=deepcopy(case.get('qualification_flags',[])),
                    prediction_receipt=state['outcomes'][key],source_hash_verified=True,changed_from_nominal=key!=nominal,
                    input_changed_from_all_ml=key!=base,prediction_drift=value.drift(case,loaded[nominal]['entry'],entry))
                if entry['status']=='failed':row['failure_root_causes']=entry.get('design_audit',{}).get('reasons',[])
                rows.append(row)
        summaries,pairs,missing=summarize(rows,protocol)
        store.unchanged();value.verify(protocol)
        result={'version':VERSION,'protocol':state['protocol'],'selected_receipt':protocol['selected_receipt'],
            'parameter_prior':protocol['parameter_prior'],'parameter_mode':protocol['parameter_mode'],
            'all_predictions_locked_before_scoring':True,'predictions_locked_at':state['predictions_locked_at'],
            'units':rows,'summaries':summaries,'available_over_all_ml':pairs,'missingness':missing,
            'historical_results':protocol['historical_results'],'historical_results_pooled':False,
            'new_native_calls':sum(x['new_native_calls'] for x in loaded.values()),
            'new_linear_trajectory_calls':sum(x['new_linear_trajectory_calls'] for x in loaded.values()),
            'logical_units':len(rows),'unique_predictions':len(loaded),'threshold_comparison':'<',
            'bootstrap':{'resamples':RESAMPLES,'seed':SEED,'unit':'study_cluster','aggregation':'drug_equal'},
            'paper_updated':False,'clinical_safety_established':False,'limitations':protocol['limitations']}
        save(run/'results.json',result)
        clone(value.evaluation.write_csv,LABELS={k:k for k in CONDITIONS})(run/'results.csv',rows)
        lines=['# Expanded validation of the selected PBPK recipe','',
            'All included clinical compounds use the same parameter-model weights. Old twenty-drug development results and new compounds are reported separately; previous selected results are preserved as historical records and are not pooled.',
            '', '| Group | Input condition | Scored cases / attempted | Drug-equal curve GMFE | Failed predictions |',
            '|---|---|---:|---:|---:|']
        for s in summaries:
            val=s['curve']['estimate'];fmt='—' if val is None else f'{val:.3f}'
            lines.append(f"| {s['group']} | {s['condition']} | {s['scored']}/{s['attempted']} | {fmt} | {s['failed']} |")
        lines+=['','The available/all-ML paired comparison and 5% benefit counts are descriptive, not clinical qualification or an adoption threshold. Masked conditions are repeated interventions, not extra independent compounds.',
            'All eleven outputs remain in results.csv. Original-reported scalar references are kept separate from curve-derived finite-window AUC and sampled peaks; missing scalar references remain empty. Missing tails and construction failures are retained.',
            'New compounds means additional to the selected twenty-drug cohort. Some were used in earlier development; per-case historical_development_used and metadata qualification flags are preserved in results.json. This is not described as a blinded test.',
            'General drug-binding evidence is separate from the target curve source, but its underlying-study overlap can be unknown. The available-input comparison does not establish fully independent clinical evidence or patient-specific binding.',
            'Prediction drift measures sensitivity to missing inputs, not clinical safety. No clinical curve fitting, new method selection, production service, or manuscript update is performed.','']
        (run/'README.md').write_text('\n'.join(lines))
        state.update(status='complete',completed_at=value.old.now(),results=value.old.ref(run/'results.json'),result_table=value.old.ref(run/'results.csv'))
        save(run/'state.json',state)
        return {'status':'complete','logical_units':len(rows),'unique_predictions':len(loaded),'new_native_calls':result['new_native_calls']}


def solve(task,run,monitor):
    if task.get('construction_blocked'):
        key=task['key']
        with monitor.lock:
            if key in monitor.state['outcomes']:
                value.old.checked(monitor.state['outcomes'][key]);return key
        monitor.check()
        entry={'status':'failed','reason':'construction_blocked','failure':'; '.join(task['construction_blocked']),
            'design_audit':{'reasons':task['construction_blocked']},'request':task['request'],
            'target_observations_used':False,'curve':None,'endpoints':None}
        item={'key':key,'case_id':task['case_id'],'entry':entry,'reused':False,'new_native_calls':0,
              'new_linear_trajectory_calls':0,'elapsed_seconds':0.,'locked_at':value.old.now()}
        path=run/'models'/f'{key[:20]}.json';save(path,item)
        with monitor.lock:
            monitor.state['outcomes'][key]=value.old.ref(path)
            monitor.update(completed_unique=len(monitor.state['outcomes']))
        return key
    return clone(execution.solve,CANDIDATE=CANDIDATE)(task,run,monitor)


def execute(run=DEFAULT):
    return clone(retained.execute,own=own,solve=solve,Monitor=retained.Monitor,report=report)(own(run))


def status(run=DEFAULT):
    run=own(run)
    if not (run/'state.json').exists():return {'status':'not_prepared','run_dir':str(run)}
    state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.verify(protocol)
    for ref in state['outcomes'].values():value.old.checked(ref)
    return {'status':state['status'],'completed_unique':len(state['outcomes']),'total_unique':len(protocol['tasks']),
        'logical_units':len(protocol['units']),'parameter_mode':protocol['parameter_mode'],
        'monitor':state.get('monitor'),'pause_reason':state.get('pause_reason'),'results':state.get('results')}


def cleanup(run=DEFAULT):
    """No discarded fit trials exist: retain receipts, native outputs and failures."""
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.verify(protocol)
        if state['status']!='complete':raise ValueError('Cleanup requires a completed experiment')
        for receipt in state['outcomes'].values():
            value.old.collect_refs(value.old.checked(receipt),{})
        paths=list(run.rglob('*'))
        if any(p.is_symlink() for p in paths):raise ValueError('Refuse symlink cleanup')
        return {'status':'no_op','deleted_files':0,'bytes':sum(p.stat().st_size for p in paths if p.is_file()),
            'reason':'No unselected clinical-fit trials were created. Native sources, model weights, score inputs and failure diagnostics remain required for reproduction.'}


def background(run=DEFAULT):
    run=own(run)
    with (run/'run.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--run-dir',str(run)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    save(run/'pid.json',{'pid':p.pid,'project':str(ROOT),'created_at':value.old.now()})
    return {'status':'started','pid':p.pid,'log':str(run/'run.log')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','status','report','cleanup'))
    p.add_argument('--run-dir',type=Path,default=DEFAULT);p.add_argument('--cohort',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--chemicals',type=Path)
    p.add_argument('--workers',type=int,default=14);p.add_argument('--background',action='store_true');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.run_dir,a.cohort,a.prior,a.workers,a.chemicals)
    elif a.command=='run':result=background(a.run_dir) if a.background else execute(a.run_dir)
    else:result=globals()[a.command](a.run_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
