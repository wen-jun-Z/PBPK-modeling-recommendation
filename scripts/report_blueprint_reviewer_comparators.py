#!/usr/bin/env python3
"""Render compact manuscript tables and revised-source sensitivity from locked fits.

No fitting, selection, or manuscript mutation. Newly audited metadata defines
source exclusions independently of error; original frozen-flag analyses remain
separate and unchanged. Outputs record the exact source metadata checksum.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import math
import re
import time
import blueprint_reviewer_comparators as r

SOURCE = r.OUT.parent/'sources/source_pairing.json'
TABLE = r.ROOT/'results/tables/blueprint_reviewer_comparators.tex'
TABLE_FRAGMENTS = (
    ('blueprint_reviewer_model_controls.tex', 'tab:blueprint_reviewer_comparators'),
    ('blueprint_reviewer_deletions.tex', 'tab:blueprint_reviewer_deletions'),
    ('blueprint_reviewer_source_sensitivity.tex', 'tab:blueprint_reviewer_source_sensitivity'),
)
PUBLICATION = r.ROOT/'results/data/blueprint_sampling_publication.json'
CATEGORY_LABELS = {
 'matrix_or_sampling_site_approximation':'Matrix / sampling site',
 'same_study_arms':'Same-study dose arms',
 'participant_overlap_flagged':'Potential participant overlap',
 'source_identity_or_locator_unverified':'Unresolved source identity',
 'population_or_summary_transfer':'Population / summary transfer',
 'patent_source':'Patent source',
 'quantitation_or_summary_metadata_incomplete':'Quantitation / summary gaps',
}
PAIRS = {
 'time_addition':('time_five','pbpk','four','pbpk'),
 'learned_addition':('learned_five','pbpk','four','pbpk'),
 'selector':('learned_five','pbpk','time_five','pbpk'),
 'pbpk_vs_two_compartment':('time_five','pbpk','time_five','two_compartment'),
}


def source_sensitivity(p, rows):
    metadata_ref=r.ref(SOURCE); source=r.checked(metadata_ref)
    pairs={x['case_id']:x for x in source['pairs']}
    if pairs.keys()!=p['cases'].keys():raise ValueError('Source cohort and comparison cohort differ')
    categories=set(CATEGORY_LABELS)
    if any(set(x['sensitivity_categories'])!=categories for x in pairs.values()):
        raise ValueError('Source sensitivity definitions changed')
    canonical={k:x['target_source_cluster'] for k,x in pairs.items()}
    old_members={cluster:tuple(sorted(k for k,c in p['cases'].items() if c['cluster_id']==cluster)) for cluster in {c['cluster_id'] for c in p['cases'].values()}}
    new_members={cluster:tuple(sorted(k for k,v in canonical.items() if v==cluster)) for cluster in set(canonical.values())}
    old_by_members={members:cluster for cluster,members in old_members.items()}
    draw_labels={cluster:old_by_members.get(members,'new-cluster:'+members[0]) for cluster,members in new_members.items()}
    # A DOI renaming must not change which fixed-seed bootstrap group is drawn.
    cases={k:dict(cluster_id=draw_labels[canonical[k]]) for k in pairs}
    same_partition=set(old_members.values())==set(new_members.values())
    values={(x['case_id'],x['info'],x['method'],x['endpoint']):float(x['error']) for x in rows
            if x['transfer']=='linear_weight' and x['error']!=''}
    results={}; records=[]
    for scope in ('all_thirty','added_twenty'):
        ids={cid for cid,c in p['cases'].items() if scope=='all_thirty' or c['cohort']=='added_twenty'}
        sets={'all':ids,'strict_metadata_subset':{k for k in ids if pairs[k]['metadata_quality_subset_included']}}
        sets.update({'without_'+cat:{k for k in ids if not pairs[k]['sensitivity_categories'][cat]} for cat in categories})
        # Same investigator-defined metadata groups are applied to every method.
        results[scope]={}
        for group,keep in sets.items():
            result=dict(n_drugs=len(keep),n_clusters=len({cases[k]['cluster_id'] for k in keep}),
                included=sorted(p['cases'][k]['drug'] for k in keep),excluded=sorted(p['cases'][k]['drug'] for k in ids-keep),comparisons={})
            for name,(ni,nm,di,dm) in PAIRS.items():
                result['comparisons'][name]={}
                for ep in ('cmax','curve'):
                    a={k:values[k,ni,nm,ep] for k in keep if (k,ni,nm,ep) in values}
                    b={k:values[k,di,dm,ep] for k in keep if (k,di,dm,ep) in values}
                    st=r.pair_stats(a,b,cases)
                    result['comparisons'][name][ep]=st
                    records.append(dict(scope=scope,group=group,comparison=name,endpoint=ep,n=st['n_cases'],clusters=st['n_clusters'],
                        ratio=st['estimate'],ci_low=st['ci95'][0] if st['ci95'] else None,ci_high=st['ci95'][1] if st['ci95'] else None,
                        improved=st['improved'],worse=st['worse'],tied=st['tied'],excluded=';'.join(result['excluded'])))
            results[scope][group]=result
    if r.ref(SOURCE)!=metadata_ref:raise ValueError('Source metadata changed during analysis; retry final version')
    out=dict(version='reviewer-source-metadata-sensitivity-1',metadata=metadata_ref,
        source_clusters=canonical,bootstrap_draw_labels={k:v['cluster_id'] for k,v in cases.items()},
        canonical_group_membership_equals_frozen=same_partition,
        bootstrap_order='Original study-group order preserved when canonical IDs only rename identical memberships; avoids arbitrary Monte Carlo changes.',generated_at=time.time(),
        distinction='New source audit categories; NOT the original frozen sensitivity_flags. Original analyses retained separately.',
        primary_reference='All30 and added20 main estimates retain every drug and the frozen target-study grouping.',
        strict_definition='Exclude union of first six source categories, excluding quantitation/summary missingness alone; inclusion never based on prediction error.',
        clinical_interpretation='Retrospective source-quality sensitivity, not independent or newly blinded validation; unknown source fields remain unknown.',
        bootstrap=dict(seed=r.SEED,resamples=2000,unit='audited canonical target-source cluster',drug_equal=True),groups=results)
    r.save(r.OUT/'source_metadata_sensitivity.json',out);r.csv_save(r.OUT/'source_metadata_sensitivity.csv',records)
    return out


def fmt(x):return '--' if x is None else f'{x:.3f}'


def ci(stat):
    if stat['estimate'] is None:return '--'
    return f"{stat['estimate']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]" if stat['ci95'] else fmt(stat['estimate'])


def esc(x):
    return str(x).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def begin(caption,label):
    return [r'\begin{table}[!htbp]',r'\captionsetup{name=Table,font=footnotesize}',r'\centering',
        r'\caption{'+caption+r'}\label{'+label+r'}',r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}\fontsize{8}{10.4}\selectfont']


def export_table_fragments(combined_table: Path = TABLE) -> tuple[Path, ...]:
    """Copy each float byte-for-byte, with its own shared color/group wrapper."""
    content = combined_table.read_bytes()
    tables = list(re.finditer(rb'\\begin\{table\}.*?\\end\{table\}', content, re.DOTALL))
    if len(tables) != len(TABLE_FRAGMENTS):
        raise ValueError('Expected exactly three reviewer comparator tables')
    prefix, suffix = content[:tables[0].start()], content[tables[-1].end():]
    if (prefix.count(b'\\begingroup') != 1
            or b'\\definecolor{BPRPale}{HTML}{EDF3F5}' not in prefix
            or suffix.strip() != b'\\endgroup'):
        raise ValueError('Expected the scoped BPRPale wrapper around reviewer tables')
    if any(content[left.end():right.start()].strip()
           for left, right in zip(tables, tables[1:])):
        raise ValueError('Unexpected content between reviewer comparator tables')
    outputs = []
    for table, (filename, label) in zip(tables, TABLE_FRAGMENTS, strict=True):
        body = table.group()
        if ('\\label{' + label + '}').encode('utf-8') not in body:
            raise ValueError('Unexpected reviewer comparator table order')
        outputs.append((combined_table.with_name(filename), prefix + body + suffix))
    for path, fragment in outputs:
        path.write_bytes(fragment)
    return tuple(path for path, _ in outputs)


def render(data,sens,new):
    source = r.checked(new['metadata'])
    unknown_locators = source['summary']['unknown_field_counts']['curve_locator']
    source_sides = len(source['records'])
    lines=[r'% Generated from locked reviewer comparators; no fitted values are changed.',r'\begingroup',r'\definecolor{BPRPale}{HTML}{EDF3F5}']
    lines+=begin('Complete blueprint prediction and classical model controls. Every configuration uses the same five development observations, target dosing and scoring times. The six controls span three compartment structures and three shared-CL constraint variants. Values are drug-equal GMFE (lower is better).','tab:blueprint_reviewer_comparators')
    methods=[('pbpk','Evidence-constrained PBPK (ours)'),
             ('one_compartment','One-compartment PK'),('two_compartment','Two-compartment PK'),
             ('three_compartment','Three-compartment PK'),
             ('one_compartment_cl_prior','One-compartment + CL prior'),
             ('two_compartment_cl_prior','Two-compartment + CL prior'),
             ('two_compartment_fixed_cl','Two-compartment, fixed CL')]
    lines += [r'\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}\toprule',
        r'\rowcolor{BPRPale} & \multicolumn{2}{c}{All 30 drugs} & \multicolumn{2}{c}{Added 20 drugs}\\',
        r'Method & Sampled $C_{\max}$ & Curve & Sampled $C_{\max}$ & Curve\\\midrule']
    for m,label in methods:
        values=[]
        for scope in ('all_thirty','added_twenty'):
            for e in ('cmax','curve'):values.append(fmt(data['summary'][scope]['time_five']['methods'][m][e]['estimate']))
        if m=='pbpk':lines.append(r'\rowcolor{BPRPale}')
        if m=='one_compartment_cl_prior':lines.append(r'\addlinespace[3pt]')
        lines.append(esc(label)+' & '+' & '.join(values)+r'\\')
    lines += [r'\midrule',r'\multicolumn{5}{@{}l}{\textbf{Paired PBPK / control error ratio [95\% CI], all 30 drugs}}\\',
              r'Control & \multicolumn{2}{c}{Sampled $C_{\max}$} & \multicolumn{2}{c}{Curve}\\']
    for model,label in methods[1:]:
        v=data['summary']['all_thirty']['time_five']['comparisons']['pbpk/'+model]
        lines.append(esc(label)+r' & \multicolumn{2}{c}{'+ci(v['cmax'])+r'} & \multicolumn{2}{c}{'+ci(v['curve'])+r'}\\')
    diagnostics=data.get('diagnostics',[])
    pending=sum(d.get('selected_converged') is False for d in diagnostics) if isinstance(diagnostics,list) else 0
    lines += [r'\bottomrule\end{tabularx}',r'\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{7.4}{9.7}\selectfont',
        r'A ratio below 1 favors PBPK. Pointwise descriptive CIs use 2,000 target-study cluster bootstrap draws (29 clusters; seed 20261009); they are not multiplicity-corrected tests. PBPK additionally uses drug evidence and human physiology. CL constraints share the development-condition input: 24 drugs use predicted CL and six use independently sourced measured/derived CL. Soft priors use the same prespecified engineering penalty; fixed CL is a constraint sensitivity, not clinical ground truth. Free parameters: one-compartment 2, two-compartment 4, three-compartment 6, fixed-CL two-compartment 3. Five observations do not identify six physiological parameters. All controls preserve bolus/infusion and dose; CL and volumes transfer linearly with weight. The added controls required 90 fits with no missing final predictions; '+str(pending)+r' selected new fits reached the optimization limit and remain included. Historical nonconvergence and allometric transfer are reported separately. All outcomes were historically viewed.',
        r'\end{minipage}\end{table}']
    lines += begin('Study influence in the additional observation-augmentation analysis. All primary cases remain in the main analysis; deletions assess influence and do not select better-performing drugs.','tab:blueprint_reviewer_deletions')
    lines += [r'\begin{tabularx}{\linewidth}{@{}Xrrrrr@{}}\toprule',r'\rowcolor{BPRPale} & & \multicolumn{2}{c}{Sampled $C_{\max}$ GMFE} & \multicolumn{2}{c}{Curve GMFE}\\',
        r'Population & $n$ & Limited data & Augmented & Limited data & Augmented\\\midrule']
    for scope,sub,label in [('all_thirty','all','All drugs'),('all_thirty','without_Montelukast','Without Montelukast'),('added_twenty','all','Added drugs'),('added_twenty','without_Montelukast','Added, without Montelukast')]:
        b=sens['groups'][scope][sub];n=b['cmax']['time_repaired']['n_cases'];values=[]
        for e in ('cmax','curve'):
            values.extend(fmt(b[e][m]['estimate']) for m in ('no_addition_repaired','time_repaired'))
        lines.append(label+f' & {n} & '+' & '.join(values)+r'\\')
    lines += [r'\bottomrule\end{tabularx}\par\vspace{7pt}',
        r'\begin{tabularx}{\linewidth}{@{}Xrrr@{}}\toprule',r'\rowcolor{BPRPale}Deletion analysis & Retained $n$ & $C_{\max}$ ratio range & Curve ratio range\\\midrule']
    for scope,label in [('all_thirty','All 30'),('added_twenty','Added 20')]:
        deleted=[v for k,v in sens['groups'][scope].items() if k.startswith('without_study:')]
        n=[v['cmax']['time_repaired']['n_cases'] for v in deleted];nr=str(min(n)) if min(n)==max(n) else f'{min(n)}--{max(n)}'
        vals=[]
        for e in ('cmax','curve'):
            rr=[v[e]['time_repaired/no_addition_repaired']['estimate'] for v in deleted]
            vals.append(f'{min(rr):.3f}--{max(rr):.3f}')
        lines.append(label+', augmented vs limited & '+nr+' & '+' & '.join(vals)+r'\\')
    lines += [r'\bottomrule\end{tabularx}',r'\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{7.4}{9.7}\selectfont',
        r'Limited data uses four development observations; Augmented uses five, with the extra time selected by time coverage. Both arms use the same feasibility-repair policy. Ratios are augmented/limited-data GMFE after removing each target-study cluster; ranges are not confidence intervals. Ertapenem and Ceftriaxone are removed together. No model is refitted. After removing Montelukast, added-19 ratios are '+ci(sens['groups']['added_twenty']['without_Montelukast']['cmax']['time_repaired/no_addition_repaired'])+r' for Cmax and '+ci(sens['groups']['added_twenty']['without_Montelukast']['curve']['time_repaired/no_addition_repaired'])+r' for curve GMFE. CIs use 2,000 paired cluster-bootstrap draws (seed 20261009). This outcome-level deletion is distinct from the historical learner\textquotesingle s training jackknife.',
        r'\end{minipage}\end{table}']
    lines+=begin('Source sensitivity of the additional observation-augmentation analysis. Ratios compare augmented-data PBPK (five observations) with limited-data PBPK (four observations). Exclusions are defined by source attributes, not prediction error.','tab:blueprint_reviewer_source_sensitivity')
    lines += [r'\begin{tabularx}{\linewidth}{@{}Xrrr@{}}\toprule',r'\rowcolor{BPRPale}Source subset & $n$ & $C_{\max}$ ratio [95\% CI] & Curve ratio [95\% CI]\\\midrule']
    entries=[('all_thirty','all','All 30, audited clustering')]
    entries += [('all_thirty','without_'+cat,'Exclude: '+label) for cat,label in CATEGORY_LABELS.items() if cat!='quantitation_or_summary_metadata_incomplete']
    entries += [('all_thirty','strict_metadata_subset','Combined source-quality subset'),('added_twenty','strict_metadata_subset','Added-drug source-quality subset')]
    for scope,sub,label in entries:
        v=new['groups'][scope][sub];pair=v['comparisons']['time_addition']
        if sub=='strict_metadata_subset':lines.append(r'\rowcolor{BPRPale}')
        lines.append(esc(label)+f" & {v['n_drugs']} & "+ci(pair['cmax'])+' & '+ci(pair['curve'])+r'\\')
    lines += [r'\bottomrule\end{tabularx}',r'\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{7.4}{9.7}\selectfont',
        r'Each single-category exclusion is applied separately; rows overlap. The combined subset excludes the union of the six named categories. Quantitation-limit or summary-metadata gaps alone do not remove a clinical curve. All 30 original curves remain in the main analysis. CIs use the audited canonical target-source clusters (2,000 bootstrap draws, fixed seed). These categories come from the revised source audit and are distinct from the original frozen sensitivity flags. Bibliographic identity is now resolved for every case, while '+str(unknown_locators)+' of '+str(source_sides)+r' source-side curve locators remain unverified. Unknown metadata remain unknown; the subset is not a newly blinded validation cohort. Exact inclusion lists and additional selector/classical-control sensitivities accompany the machine-readable results.',
        r'\end{minipage}\end{table}',r'\endgroup']
    TABLE.write_text('\n'.join(lines)+'\n')
    export_table_fragments()


def main():
    state=r.read(r.OUT/'state.json');p=r.checked(state['protocol']);data=r.checked(state['results'])
    if state['status']!='complete':raise ValueError('Complete locked experiment required')
    with (r.OUT/'endpoints.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    previous=r.OUT/'source_metadata_sensitivity.json'
    new=r.read(previous) if previous.exists() else source_sensitivity(p,rows)
    if new['metadata'] != r.ref(SOURCE):
        new=source_sensitivity(p,rows)
    publication=r.read(PUBLICATION)
    if publication['main_prediction_method']!='time_repaired':raise ValueError('Current time-rule publication required')
    for key in ('source_result','source_protocol','source_endpoints'):
        reference=publication[key]
        if r.sha(r.ROOT/reference['path']) != reference['sha256']:raise ValueError('Time-rule publication source changed')
    extension=r.ROOT/'experiments/blueprint_design/model_controls_extension_20261011'
    ext_state=r.read(extension/'state.json')
    if ext_state['status']!='complete' or len(ext_state['completed'])!=90:
        raise ValueError('All 90 added predictions must complete before publication')
    extended=r.checked(ext_state['results'])
    for scope in ('all_thirty','added_twenty','original_ten'):
        for method in ('pbpk',*r.MODELS):
            for endpoint in ('cmax','curve'):
                old_value=data['summary'][scope]['time_five']['methods'][method][endpoint]['estimate']
                new_value=extended['summary'][scope]['time_five']['methods'][method][endpoint]['estimate']
                if not math.isclose(old_value,new_value,rel_tol=1e-12):
                    raise ValueError('Historical control changed during extension')
    sens=publication['statistics']['deletions'];render(extended,sens,new)
    print(TABLE)
    for scope in ('all_thirty','added_twenty'):
        v=new['groups'][scope]['strict_metadata_subset'];print(scope,v['n_drugs'],v['included'])
        for name,items in v['comparisons'].items():print(name,{e:ci(st) for e,st in items.items()})

if __name__=='__main__':main()
