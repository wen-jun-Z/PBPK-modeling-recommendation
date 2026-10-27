#!/usr/bin/env python3
"""Cache-only evaluation of one declared plasma-fu source qualification rule.

The base expansion remains immutable. All clinical cases and five conditions
remain present, including construction failures and source-rejected evidence.
No training, native simulation, or score-driven per-drug selection occurs here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_expanded_blueprint_validation as base
from pbpk_kb.blueprint_policy import qualified_binding as policy
from pbpk_kb.blueprints.source_workflow import save,sha256

value=base.value
clone=base.clone
VERSION='source-qualified-binding-validation-1'
SOURCE=base.DEFAULT
DEFAULT=SOURCE/'source_qualified'
CONDITIONS=base.CONDITIONS


def own(path):
    path=value.inside(path,ROOT)
    if path!=DEFAULT:raise ValueError('Use only the source_qualified subdirectory')
    return path


def exact_input_digest(task):
    return value.model._digest({k:task[k] for k in ('request','sidecars','core')})


def without_fu_or_ledger(request):
    result=deepcopy(request);result.pop('evidence_ledger',None)
    result['evidence']=sorted((r for r in result['evidence'] if r['parameter']!='fraction_unbound'),
                              key=lambda r:r['parameter'])
    return result


def bind_existing_request(original, all_ml):
    """Prove fu-only change, then adopt the exact already-locked ledger/order."""
    records=[r for r in all_ml['request']['evidence'] if r['parameter']=='fraction_unbound']
    if len(records)!=1:raise ValueError('The all-ML reference must contain exactly one fu')
    transformed,audit=policy.qualify_binding(original['request'],records[0])
    if not audit['changed']:return deepcopy(original),audit
    if original['sidecars']!=all_ml['sidecars'] or original['core']!=all_ml['core']:
        raise ValueError('Needs new native assessment: sidecars/core differ; cannot reuse all-ML input')
    if without_fu_or_ledger(transformed)!=without_fu_or_ledger(all_ml['request']):
        raise ValueError('Needs new native assessment: other evidence, clearance, physiology or dosing would change')
    # Every full record, including its provenance and context, must now agree.
    normalized=deepcopy(transformed);normalized.pop('evidence_ledger',None)
    normalized['evidence']=sorted(normalized['evidence'],key=lambda r:r['parameter'])
    target=deepcopy(all_ml['request']);target.pop('evidence_ledger',None)
    target['evidence']=sorted(target['evidence'],key=lambda r:r['parameter'])
    if normalized!=target:raise ValueError('The qualified request is not the existing all-ML scientific input')
    result=deepcopy(all_ml)
    audit.update(exact_cached_input_digest=exact_input_digest(result),
        ledger_and_order='Bound to the exact existing all-ML request after full non-fu evidence preservation checks',
        non_fu_evidence_preserved=True,request_before_sha256=value.model._digest(original['request']),
        request_after_sha256=value.model._digest(result['request']))
    return result,audit


def prepare(run=DEFAULT):
    run=own(run)
    if (run/'protocol.json').exists():raise FileExistsError('Protocol already frozen; resume with run')
    state=value.old.read(SOURCE/'state.json');source=value.old.checked(state['protocol']);value.verify(source)
    if state['status']!='complete' or not state.get('all_predictions_locked_before_scoring'):
        raise ValueError('Completed expansion with locked predictions is required')
    pins=deepcopy(source['source_pins']);code=deepcopy(source['code_pins'])
    for receipt in (state['protocol'],state['results']):
        if sha256(value.inside(receipt['path'],ROOT,exists=True))!=receipt['sha256']:raise ValueError('Base source changed')
        pins[receipt['path']]=receipt['sha256']  # Do not decode clinical results here.
    for path in (Path(__file__),Path(policy.__file__)):
        code[str(path.resolve().relative_to(ROOT))]=sha256(path)
    tasks,units,audits={},[],[]
    for unit in source['units']:
        original=source['tasks'][unit['key']]
        allkey=source['baselines'][unit['case_id']]['fingerprint'];all_ml=source['tasks'][allkey]
        selected,audit=bind_existing_request(original,all_ml)
        key=selected['key']
        if key not in source['tasks'] or exact_input_digest(selected)!=exact_input_digest(source['tasks'][key]):
            raise ValueError('New native prediction would be needed; this cache-only candidate must stop')
        if base.retained.key_for(selected['request'],selected['sidecars'],selected['core'])!=key:
            raise ValueError('Input fingerprint differs from the frozen source')
        if key not in tasks:
            receipt=state['outcomes'][key];outcome=value.old.checked(receipt)
            value.old.collect_refs(outcome,pins);pins[receipt['path']]=receipt['sha256']
            if (outcome['key']!=key or outcome['entry']['request']!=selected['request']
                    or outcome['entry'].get('target_observations_used') is not False
                    or outcome['entry']['status'] not in {'predicted','failed'} or not outcome.get('locked_at')):
                raise ValueError('Source prediction is not a locked exact-input terminal result')
            tasks[key]={**selected,'reuse_value_receipt':receipt,'source_exact_input_digest':exact_input_digest(selected)}
        units.append({**deepcopy(unit),'key':key,'base_key':unit['key'],'binding_policy_changed':audit['changed']})
        audits.append({'case_id':unit['case_id'],'condition':unit['condition'],**audit})
    matrix={(c['case_id'],condition) for c in source['cases'] for condition in CONDITIONS}
    if len(units)!=len(matrix) or {(u['case_id'],u['condition']) for u in units}!=matrix:
        raise ValueError('Retain every clinical case and condition')
    protocol=deepcopy(source)
    actual=sorted({u['case_id'] for u in units if u['condition']=='available'
        and u['key']!=source['baselines'][u['case_id']]['fingerprint']})
    protocol.update(version=VERSION,created_at=value.old.now(),tasks=tasks,units=units,
        policy_version=policy.VERSION,policy_audits=audits,source_pins=pins,code_pins=code,
        source_expansion_protocol=state['protocol'],source_expansion_results=state['results'],
        current_evidence_changed_case_ids=actual,original_evidence_changed_case_ids=source['changed_case_ids'],
        changed_case_ids=source['changed_case_ids'],
        evidence_group_contract='The original expansion evidence-eligible group remains the comparison denominator, including source-rejected cases',
        clinical_cases_dropped=0,workers=1,new_unique_predictions=0,expected_new_native_calls=0,
        new_independent_experiments=0,cache_only=True,publication_snapshot=value.old.paper_pins(),paper_guard={},
        limitations=[*source['limitations'],
            'A single source-qualification rule was proposed after the expansion results; this is development exploration, not external confirmation',
            'General binding adopted as plasma fu is not a hard input; exact previously locked all-ML predictions are reused',
            'All original evidence-eligible cases remain in the benefit denominator; rejected evidence and every failure remain reported',
            'No new drug, patient, fit, or independent simulation is introduced by this cache remapping'])
    value.verify(protocol,publication=True)
    run.mkdir(parents=True,exist_ok=True);save(run/'protocol.json',protocol)
    save(run/'state.json',{'version':VERSION,'status':'prepared','created_at':value.old.now(),
        'protocol':value.old.ref(run/'protocol.json'),'outcomes':{},'all_predictions_locked_before_scoring':False})
    changed=[a for a in audits if a['changed']]
    return {'status':'prepared','clinical_cases':len(source['cases']),'logical_units':len(units),
        'cached_unique_predictions':len(tasks),'changed_logical_units':len(changed),
        'changed_cases':sorted({a['case_id'] for a in changed}),'original_evidence_group':len(source['changed_case_ids']),
        'current_evidence_changed_cases':len(actual),'new_native_calls':0,'new_training_calls':0,'new_independent_experiments':0}


def lock_predictions(run=DEFAULT):
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.verify(protocol)
        if state.get('all_predictions_locked_before_scoring'):
            if set(state['outcomes'])!=set(protocol['tasks']):raise ValueError('Locked cache coverage differs')
            return {'status':state['status'],'cached_unique_predictions':len(state['outcomes'])}
        (run/'models').mkdir(exist_ok=True)
        monitor=base.retained.Monitor(run,protocol,state);monitor.check(force=True)
        for key,task in protocol['tasks'].items():
            monitor.check()
            if task['source_exact_input_digest']!=exact_input_digest(task):raise ValueError('Scientific inputs changed')
            original=value.old.checked(task['reuse_value_receipt'])
            if original['entry']['request']!=task['request']:raise ValueError('Exact cached request differs')
            base.retained.solve(task,run,monitor)
        monitor.check(force=True)
        if set(state['outcomes'])!=set(protocol['tasks']):raise ValueError('Incomplete cache-lock matrix')
        monitor.update(status='predictions_locked',all_predictions_locked_before_scoring=True,
                       predictions_locked_at=value.old.now(),new_native_calls=0,new_training_calls=0)
        return {'status':'predictions_locked','cached_unique_predictions':len(state['outcomes']),'new_native_calls':0}


def comparison_statistics(result,previous,protocol):
    groups={'all':protocol['cases'],
        'old_20':[c for c in protocol['cases'] if c['validation_stratum']=='old_20'],
        'new_compounds':[c for c in protocol['cases'] if c['validation_stratum']=='new_compounds'],
        'original_evidence_eligible':[c for c in protocol['cases'] if c['case_id'] in protocol['original_evidence_changed_case_ids']]}
    comparisons={}
    for name,cases in groups.items():
        comparisons[name]={}
        for condition in CONDITIONS:
            after={r['case_id']:r['curve_metrics'].get('gmfe') for r in result['units'] if r['condition']==condition}
            before={r['case_id']:r['curve_metrics'].get('gmfe') for r in previous['units'] if r['condition']==condition}
            comparisons[name][condition]=base.paired(after,before,cases)
    original_ids=set(protocol['original_evidence_changed_case_ids'])
    group=[c for c in protocol['cases'] if c['case_id'] in original_ids]
    def evidence_benefit(data):
        a={r['case_id']:r['curve_metrics'].get('gmfe') for r in data['units'] if r['condition']=='available'}
        b={r['case_id']:r['curve_metrics'].get('gmfe') for r in data['units'] if r['condition']=='all_ml'}
        return base.paired(a,b,group)
    return {'candidate_over_base_expansion':comparisons,
        'evidence_benefit_original_group':{'base':evidence_benefit(previous),'qualified':evidence_benefit(result)},
        'original_evidence_group_case_ids':sorted(original_ids),
        'current_evidence_changed_case_ids':protocol['current_evidence_changed_case_ids'],
        'denominator_reduced_for_benefit_claim':False,'clinical_safety_established':False,
        'new_independent_experiments':0,'no_per_drug_score_selection':True}


def report(run=DEFAULT):
    run=own(run)
    info=clone(base.report,own=own,VERSION=VERSION)(run)
    state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol'])
    result=value.old.checked(state['results']);previous=value.old.checked(protocol['source_expansion_results'])
    result.update(policy_version=policy.VERSION,policy_audits=protocol['policy_audits'],
        source_expansion_results=protocol['source_expansion_results'],
        evidence_group_contract=protocol['evidence_group_contract'],
        source_qualification=comparison_statistics(result,previous,protocol),
        new_training_calls=0,new_independent_experiments=0,cache_only=True)
    value.verify(protocol);save(run/'results.json',result);state['results']=value.old.ref(run/'results.json');save(run/'state.json',state)
    with (run/'README.md').open('a') as stream:
        stream.write('\nSource-qualification candidate: only explicitly assumed general-binding-to-plasma fu is rejected. Rejected source records and fallback receipts are preserved. Unknown underlying-study overlap alone does not reject evidence.\n\n')
        stream.write(f"The evidence comparison retains the original {len(protocol['original_evidence_changed_case_ids'])}-case eligible group; the {len(protocol['current_evidence_changed_case_ids'])} cases with a current numerical input change are reported separately. This smaller group is not used to claim a larger benefit proportion.\n\n")
        stream.write('All predictions are exact copies of already locked input-identical results. There are no new fits, native calls, subjects or independent experiments. The rule was developed after reviewing the expansion, so this remains exploratory development evidence.\n')
    return {**info,'cache_only':True,'new_training_calls':0,'new_independent_experiments':0}


def run(run_dir=DEFAULT):
    lock_predictions(run_dir)
    return report(run_dir)


def status(run=DEFAULT):
    return clone(base.status,own=own)(own(run))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','report'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT);args=parser.parse_args()
    result=globals()[args.command](args.run_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
