#!/usr/bin/env python3
"""Analyze locked policy predictions; optionally recover reporting without a live monitor.

Adds no simulations, fits, thresholds, or changes to the frozen experiment.
Post-hoc endpoint comparisons and four-drug input-effect diagnostics are
exploratory. Summary is stored in existing state.json and experiments/README.md.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
from datetime import datetime

import run_blueprint_evidence_policy as runner
from report_blueprint_evidence_policy import report, _all_refs
from report_blueprint_validation import Store, paired_ratio

ROOT = runner.ROOT
BEGIN = '<!-- evidence-policy-analysis:start -->'
END = '<!-- evidence-policy-analysis:end -->'


def finalize(run):
    """Never overlap a monitor's mutable state with the report's integrity read."""
    run = runner.own_run(run)
    with (run/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if runner.status(run).get('alive'):
            raise RuntimeError('Live runner: wait for it to exit before independent finalization')
        state = runner.read(run/'state.json')
        if state['status'] == 'complete':
            return report(run, project_root=ROOT)
        protocol = runner.checked(state['protocol'])
        if not state.get('all_attempts_locked_before_scoring') or not state.get('locked_at'):
            raise RuntimeError('No complete prediction lock; finalization cannot bypass experiments')
        if set(state['computations']) != set(protocol['computations']):
            raise RuntimeError('Incomplete prediction computations')
        runner.verify(protocol['source_pins'])
        store = Store(ROOT)
        _all_refs(protocol, store); _all_refs(state, store); store.unchanged()
        state.setdefault('report_recovery', []).append({'time':runner.now(),
            'previous_status':state['status'], 'previous_reason':state.get('pause_reason'),
            'action':'Standalone report under exclusive lock with no monitor; all model inputs and predictions unchanged'})
        state['status'] = 'predictions_locked'; state.pop('pause_reason', None)
        runner.save(run/'state.json', state)
        try:
            result = report(run, project_root=ROOT)
        except Exception as exc:
            state.update(status='paused',pause_reason=type(exc).__name__+': '+str(exc))
            runner.save(run/'state.json',state)
            raise
        state.update(status='complete',completed_at=runner.now(),results=runner.ref(run/'results.json'))
        runner.save(run/'state.json',state)
        return result


def analyze(run):
    run = runner.own_run(run)
    state = runner.read(run/'state.json')
    if state['status'] != 'complete': raise ValueError('Finalize all locked predictions before analysis')
    result = report(run, project_root=ROOT)  # Verify locks; completed reports are not rewritten.
    protocol = runner.checked(state['protocol'])
    records = {(u['case_id'],u['scenario'],u['sampling'],u['method']):u for u in result['units']}
    cases = protocol['cases']; ids = [c['case_id'] for c in cases]
    clusters = {c['case_id']:c['cluster_id'] for c in cases}
    changed = []
    for cid in ids:
        evidence = {}
        for method in ('evidence_policy','rf_fill'):
            d = next(d for d in protocol['decisions'] if (d['case_id'],d['scenario'],d['method'])==(cid,'available',method))
            evidence[method] = {e['parameter']:e['value'] for e in d['decision']['selected_evidence']}
        differences = {k for k in evidence['evidence_policy'] if evidence['evidence_policy'][k]!=evidence['rf_fill'][k]}
        if differences - {'fraction_unbound'}: raise ValueError('This batch was intended to change RF fu only')
        if differences: changed.append(cid)
    def scores(sampling,method,endpoint=None,subset=ids,nonweak=False):
        output={}
        for cid in subset:
            row=records[cid,'available',sampling,method]
            item=row.get('endpoints',{}).get(endpoint,{}) if endpoint else {}
            value=item.get('error') if endpoint else row['curve_metrics'].get('gmfe')
            if value is not None and (not nonweak or not item.get('weak_tail')): output[cid]=value
        return output
    pairs={}
    for sampling in ('full','five'):
        pairs[sampling]={}
        for endpoint in result['statistics']['groups']['available'][sampling]['evidence_policy']['endpoints']:
            if endpoint=='tmax':continue
            a=scores(sampling,'evidence_policy',endpoint);b=scores(sampling,'two_compartment',endpoint)
            clean_a=scores(sampling,'evidence_policy',endpoint,nonweak=True)
            clean_b=scores(sampling,'two_compartment',endpoint,nonweak=True)
            pairs[sampling][endpoint]={'all_scored':paired_ratio(a,b,clusters,expected_ids=ids),
                'both_predictions_nonweak':paired_ratio(clean_a,clean_b,clusters,expected_ids=ids)}
    effective={sampling:paired_ratio(scores(sampling,'evidence_policy',subset=changed),
        scores(sampling,'rf_fill',subset=changed),clusters,expected_ids=changed) for sampling in ('full','five')}
    trials=Counter(); reasons=Counter(); axes=Counter(); calls=0
    for fit in state['fits'].values():
        mapping=runner.checked(fit['mapping']);calls+=mapping['total_native_calls']
        axes[','.join(mapping['screening']['selected_parameters'])]+=1
        for row in mapping['evaluations']:
            trials[row['status']]+=1
            if row['status']!='ok':reasons[str(row.get('reason','unspecified'))]+=1
    metrics=result['statistics']['groups']['available']
    rows=[]
    for c in cases:
        cid=c['case_id']
        rows.append({'drug':c['drug'],'case_id':cid,**{m:{s:records[cid,'available',s,m]['curve_metrics']['gmfe'] for s in ('full','five')}
            for m in ('evidence_policy','two_compartment','rf_fill')}})
    summary={'analyzed_at':runner.now(),'script':runner.ref(Path(__file__).resolve()),
        'source_result':runner.ref(run/'results.json'),
        'analysis_role':'Post-hoc diagnosis of a previously viewed development cohort; no new hypothesis confirmation',
        'unique_predictions':len(state['computations']),'logical_units':len(protocol['units']),
        'scoring_statuses':dict(Counter(r['scoring_status'] for r in result['units'])),
        'native_fit_calls':calls,'native_target_calls':38,
        'fit_statuses':dict(Counter(f['status'] for f in state['fits'].values())),
        'screened_axes':dict(axes),'trial_statuses':dict(trials),'trial_failures':dict(reasons),
        'calculation_seconds':(datetime.fromisoformat(state['locked_at'])-datetime.fromisoformat(state['started_at'])).total_seconds(),
        'effective_fu_case_ids':changed,'effective_fu_policy_over_rf':effective,
        'endpoint_policy_over_two_compartment':pairs,'per_drug_curve_gmfe':rows,
        'available_curve_summary':{s:{m:values['curve'] for m,values in metrics[s].items()} for s in ('full','five')},
        'interpretation':['Missing-fu policy outputs equal RF-fu outputs and do not outperform two-compartment PK.',
            'Policy and fixed-template predictions are exactly shared under available evidence; clinical policy superiority is not established there.',
            'All engineering acceptance labels follow prescribed rules and are not independent clinical accuracy labels.',
            'Endpoint contrasts are exploratory without multiplicity correction; AUC infinity and clearance are mathematically coupled.',
            'A diagnostic probe consumes native budget; current results do not isolate a causal effect of sensitivity screening.']}
    return result,summary


def markdown(result,summary):
    out=[BEGIN,'### Current: evidence construction and purpose pilot (2026-09-29; analysis complete)','',
        'All 78/78 groups completed real calculations: 38 PBPK and 40 two-compartment groups. There were 76 PBPK formula fits. Of 720 logical units, 440 had clinical scores, 160 were blocked by evidence rules, and 120 DDI requests assessed capability boundaries only. They came from ten drugs and nine study clusters, not 720 independent clinical experiments.',
        '', 'Computation took about 62 minutes, with 1,808 calibration calls and 38 target predictions. All 46 trial failures were retained: 42 clearance-capacity/formula checks and four nonpositive concentrations. Every final prediction succeeded. Numerical convergence was reached in 22/76 formula fits; 54/76 exhausted the budget. Reporting initially hit an integrity check because monitoring wrote state concurrently. Scoring resumed under an exclusive lock after the process exited; predictions and source data were unchanged.','',
        '**Curve GMFE, equally weighted by drug; lower is better:**','',
        '| Method/condition | Full development curve | Fixed five points |','|---|---:|---:|']
    for method,label in [('evidence_policy','Proposed method: available evidence'),('fixed_template','Fixed-input PBPK: available evidence'),('rf_fill','RF-fu-first PBPK'),('two_compartment','Two-compartment PK'),('two_compartment_prior','Two-compartment PK + CL prior')]:
        vals=[result['statistics']['groups']['available'][s][method]['curve']['estimate'] for s in ('full','five')]
        out.append(f'| {label} | {vals[0]:.3f} | {vals[1]:.3f} |')
    out.extend(['',
        'GMFE ratios against two-compartment PK were 1.114 [0.950, 1.381] and 1.050 [0.922, 1.232], using 2,000 study-cluster bootstrap replicates. Point estimates remained worse; intervals crossing 1 do not establish equivalence. With five points, 6/10 drugs had lower error, but drug-equally-weighted performance did not beat two-compartment PK. Under normal inputs, the proposed and fixed-input controls reuse the same calculations, so identical values are expected.','',
        '**Evidence benefits and costs:** fu actually changed for four drugs; evidence priority improved all four curves. Paired GMFE ratios were 0.731 [0.551, 0.971] with full curves and 0.731 [0.481, 0.987] with five points. The other six drugs used identical inputs and are not new ablation evidence. With fu absent or wrong-species fu excluded, RF gap filling gave GMFE 1.708/1.695, identical to the RF-fu control. Fixed inputs blocked 10/10 drugs while two-compartment PK remained executable. The benefit is continued construction, not superior accuracy over two-compartment PK after information loss.','',
        '**Engineering decisions:** of 60 requests, the method admitted 30 and blocked 30, with no false decisions under predefined rules. Fixed inputs additionally blocked 20 fillable requests; RF-fu priority and removal of conflict protection each admitted ten contradictory-fu requests. Two-compartment PK needs neither fu nor molecular formula, so all 50 plasma requests ran and ten DDI requests were blocked. This checks consistency with artificial perturbations and predefined rules, not 100% clinical decision accuracy.','',
        '**Endpoints:** with five points, AUC/Cmax mean fold errors were 1.087/1.367 versus 1.159/1.444 for two-compartment PK. Corresponding within-twofold counts were 6/6 and 9/10 versus 6/6 and 8/10. Half-life remained worse at 1.859 versus 1.447, as did Vz at 1.845 versus 1.470. Endpoint-wise comparisons are exploratory without multiple-testing adjustment; CL and AUC infinity are correlated, not independent advantages. Weak-tail predictions remain flagged in the result table.','',
        '**Main issue:** Remdesivir curve GMFE was 2.992/2.255, the only drug still above 2. With five points, AUC error was only 1.031-fold, but half-life and Vz errors were 5.482-fold and 5.218-fold, both with weak-tail flags. Improved exposure integration did not resolve terminal-shape bias. Excluding the prespecified Nemonoxacin, Dexmedetomidine, and Midazolam gave seven-drug GMFE 1.604/1.598 versus 1.326/1.427 for two-compartment PK; the conclusion did not reverse.','',
        '**Next direction:** diagnose Remdesivir distribution, terminal behavior, and development-fit errors to distinguish parameterization limits from incomplete search. Do not tune on target observations and relabel the result independent validation. Retain evidence gap filling and conflict protection, add real missing/conflicting cases, and test the independent contribution of local sensitivity with equal-budget ablation. No new training starts automatically and the manuscript PDF is not overwritten.','',
        '[Final item-level CSV](blueprint_design/evidence_policy_20260929/results.csv) · [Full scoring JSON](blueprint_design/evidence_policy_20260929/results.json) · [Follow-up diagnostics](blueprint_design/evidence_policy_20260929/state.json) · [Audit entry point](../scripts/analyze_blueprint_evidence_policy.py)','',
        '```bash','runtime/app-env/bin/python -B scripts/analyze_blueprint_evidence_policy.py --write-summary',
        '# If everything is locked and only a reporting-state conflict paused the run, use independent recovery without resimulation:',
        'runtime/app-env/bin/python -B scripts/analyze_blueprint_evidence_policy.py --finalize --write-summary','```',END])
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=runner.DEFAULT)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--write-summary',action='store_true')
    args=parser.parse_args(); run=runner.own_run(args.run_dir)
    if args.finalize: finalize(run)
    result,summary=analyze(run)
    if args.write_summary:
        if run != runner.DEFAULT: raise ValueError('README summary is scoped to the frozen 20260929 batch')
        with (run/'.run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if runner.status(run).get('alive'):raise RuntimeError('Cannot edit summary while runner is active')
            state=runner.read(run/'state.json');state['analysis']=summary;runner.save(run/'state.json',state)
            path=ROOT/'experiments/README.md';text=path.read_text();block=markdown(result,summary)
            if BEGIN in text:
                start=text.index(BEGIN);end=text.index(END,start)+len(END);text=text[:start]+block+text[end:]
            else:
                anchor='## 4. \u5b9e\u9645\u4eff\u771f\u4e0e\u84dd\u56fe\n';assert anchor in text
                text=text.replace(anchor,anchor+'\n'+block+'\n',1)
            path.write_text(text)
    print(json.dumps({k:summary[k] for k in ('unique_predictions','scoring_statuses','fit_statuses','trial_statuses','effective_fu_policy_over_rf')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
