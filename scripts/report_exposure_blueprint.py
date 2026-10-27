#!/usr/bin/env python3
"""Compact report adapter for the frozen exposure experiment, without refitting."""
from pathlib import Path
import fcntl
import json
import math

import run_exposure_blueprint as batch
from report_blueprint_validation import paired_ratio


def main():
    state=batch.read(batch.RUN/'state.json')
    if state['status'] not in {'predictions_locked','complete'}:
        raise ValueError('All predictions must be locked before reporting')
    protocol=batch.checked(state['protocol'])
    units={**state['units'],**protocol['controls']}
    original_save=batch.save
    def compact_save(path,value):
        if Path(path)==batch.RUN/'results.json':
            for record in value['records']:
                curve=record.pop('curve',None)
                if curve is not None:
                    key=f"{record['case_id']}__{record['sampling']}__{record['method']}"
                    record['prediction_source']=units[key]['prediction']
                    record['curve_storage']='Full locked prediction referenced, not duplicated in aggregate JSON'
            value['report_adapter']=batch.ref(Path(__file__))
            value['report_note']='Formatting-only correction after all predictions locked: omit duplicated dense curves; scoring, all endpoints and NCA diagnostics unchanged.'
            clusters={case['case_id']:case['cluster_id'] for case in protocol['cases']}
            value['development_method_choice']={'method':'direct_relaxed',
                'analysis_type':'Post-hoc method choice in an already viewed development cohort; not independent confirmation.',
                'scope':'A drug-specific development curve remains required; no claim of improved zero-curve prediction.',
                'reason':'Lower cohort GMFE in both full and five-point conditions than fixed_prior and exposure_constrained; retain all regressions and the negative exposure-constraint comparison.',
                'direct_over_fixed_prior':{},'excluding_ertapenem_exploratory':{},'ertapenem_share_of_net_log_improvement':{}}
            for sampling in batch.SAMPLINGS:
                scores={method:{r['case_id']:r['curve_metrics'].get('gmfe') for r in value['records'] if r['method']==method and r['sampling']==sampling} for method in ('direct_relaxed','fixed_prior')}
                value['development_method_choice']['direct_over_fixed_prior'][sampling]=paired_ratio(scores['direct_relaxed'],scores['fixed_prior'],clusters)
                ertapenem=next(c['case_id'] for c in protocol['cases'] if c['drug']=='Ertapenem')
                subset={k:v for k,v in clusters.items() if k!=ertapenem}
                value['development_method_choice']['excluding_ertapenem_exploratory'][sampling]=paired_ratio(scores['direct_relaxed'],scores['fixed_prior'],subset)
                ratios=value['development_method_choice']['direct_over_fixed_prior'][sampling]['per_case_ratio']
                value['development_method_choice']['ertapenem_share_of_net_log_improvement'][sampling]=math.log(ratios[ertapenem])/sum(map(math.log,ratios.values()))
        original_save(path,value)
    with (batch.RUN/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batch.save=compact_save
        result=batch.report()
        document=batch.read(batch.RUN/'results.json')
        comparisons=document['development_method_choice']['direct_over_fixed_prior']
        lines=['','## Conclusions from this batch','',
            'The development-curve branch adopts direct_relaxed as the next development baseline: retain human two-region physiology and fixed net renal inputs, remove the strong nonrenal-CL prior penalty, widen numerical search bounds, and still fit only CL/K/PS. Source CL supplies initialization and multiplier reference only; the original evidence ledger is unchanged.',
            'This is method selection after inspecting development results, not confirmation by a new blinded test. The original manuscript and no-drug-curve branch are unchanged; methods are not stitched together per drug.', '',
            '| Condition | direct/fixed GMFE ratio | 95% study-cluster interval | Improved/worsened |','|---|---:|---|---:|']
        for sampling in batch.SAMPLINGS:
            c=comparisons[sampling];lines.append(f'| {sampling} | {c["estimate"]:.4f} | [{c["ci95"][0]:.4f}, {c["ci95"][1]:.4f}] | {c["ratio_below_one"]} / {c["ratio_above_one"]} |')
        lines+=['',
            'Direct CL relaxation outperformed forced development-window area equality. For Treprostinil, full/five-point GMFE was 4.304/4.227 with the latter versus 2.375/2.264 with direct calibration. Equal area does not establish accurate target-study prediction; this hard constraint is not adopted as the main method.',
            'Benefits were not uniform across drugs or endpoints: Cefepime, Midazolam, and Nemonoxacin curve errors increased. Mean improvement is not clinical reliability. Frozen two-compartment references remained lower overall, with different budgets.',
            'Paired intervals for both sampling settings crossed 1. Ertapenem contributed about 81% of net log-GMFE improvement. Exploratory exclusion left about 1.9% mean improvement among nine drugs, still with an interval crossing 1. General superiority is not established.',
            'Only two drugs had last-concentration references; full/five-point endpoint GMFE worsened from 2.915/3.041 to 3.827/3.173. Remdesivir half-life error remained about 6.34/6.42-fold with weak-tail flags. Other endpoint coverage and weak-tail counts are in results.json; correlated tail metrics are not independent repeated evidence.',
            'Five-point GMFE exceeded full-curve GMFE by about 1.15%, versus about 0.97% previously. Greater sparse-point stability was not established. Fewer sampling points are not equivalent to missing-CL or missing-fu experiments.',
            '', '## Verification and compute record','',
            'All 60 new units and 120 formula fits yielded usable unit predictions. With 40 frozen references, there were 100 comparison units and 1,100 endpoint records. Two individual formulas had no solution and the alternate formula completed the unit; failures remain.',
            'All 32 engineering/report checks passed. Independent arithmetic checks covered 831 file hashes, 1,040 curve-point comparisons, 800 scalar errors, 2000 study-cluster bootstrap replicates, fixed renal inputs, and 48 weight-transfer checks. The 1,040 points and 800 items include repeated method comparisons, not independent samples.',
            'Formal fitting used 300,161 mathematical solves; 21,999 development-precheck solves are separate. Five new native configurations are attributed to first shared-cache construction and are not method-speed comparisons; no native-construction failure was found. Reference resources are historical runs, not new work. Concurrent unit wall_seconds includes waiting and cannot be summed as total elapsed time.',
            'Reporting previously stopped when duplicate dense curves exceeded the 32 MiB limit. Only a format adapter was added, retaining prediction-file references; scores, frozen predictions, and calibration code were unchanged.',
            '', 'Recalculate with `runtime/app-env/bin/python -B scripts/report_exposure_blueprint.py`. The frozen run_exposure_blueprint.py remains for status, completion checks, and provenance.']
        readme=batch.RUN/'README.md'
        text=readme.read_text().replace('; `report` recalculates results.','; use the compact reporting entry point below for recalculation.')
        readme.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
