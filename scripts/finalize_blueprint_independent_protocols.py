#!/usr/bin/env python3
"""Create outcome-free active protocol contracts without reopening PK outcomes.

Only protocol/demographic source fields are parsed. Original sealed endpoint
files are checksum-verified, never decoded or rewritten. This command does not
run a solver and does not promote approximate predictions to exact validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from curate_blueprint_independent_cohort import ROOT, inside, MAX_NEW_BYTES


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text(element):
    return ' '.join(' '.join(element.itertext()).split()) if element is not None else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    cohort = inside(ROOT / args.cohort)
    previous = cohort / 'protocol_revision_20260920'
    output = cohort / 'active_protocols_v3'
    output.mkdir(exist_ok=False)

    def save(relative, content):
        payload = (json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        if sum(p.stat().st_size for p in cohort.rglob('*') if p.is_file()) + len(payload) > MAX_NEW_BYTES:
            raise RuntimeError('Cumulative cohort byte limit exceeded')
        target = inside(output / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(payload)
        return {'path': str(target.relative_to(ROOT)), 'sha256': digest(payload), 'byte_size': len(payload)}

    # Chemical dose equivalence, not the source study's PK observations.
    salt_file = inside(ROOT / 'artifacts/pbpk_raw_runs/compound_runs/run_20260907T134918Z/compounds/metformin/pmc/efetch_xml_finalizer/pmc_oa/pmc_efetch_finalizer_pmc8359581_pmc8791222_20_dcc32b6f7585.xml')
    salt_article = next(a for a in ET.parse(salt_file).getroot().iter('article')
        if {'PMC8604249', '8604249'} & {e.text for e in a.findall('.//article-id')})
    equivalence = next(text(p) for p in salt_article.iter('p')
        if text(p).startswith('A: A single oral dose of 1 g metformin hydrochloride') and '780 mg of free base' in text(p))
    chemical_source = {'pmcid': 'PMC8604249', 'pmid': '34268884', 'doi': '10.1111/cts.13107',
        'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8604249/',
        'source_file': str(salt_file.relative_to(ROOT)), 'source_file_sha256': digest(salt_file.read_bytes()),
        'paragraph_sha256': digest(equivalence.encode()),
        'source_role': 'chemical salt-to-free-base dose equivalence only; no PK outcomes used',
        'reported_salt_mg': 1000, 'reported_free_base_mg': 780,
        'factor': 0.78, 'precision': 'rounded equivalence reported in primary source, not exact molecular-weight ratio'}

    furo_source = cohort / 'sealed/sources/PMC8416862.xml'
    furo_article = ET.parse(furo_source).getroot()
    demo_table = next(t for t in furo_article.findall('.//table-wrap') if t.get('id') == 'Tab1')
    demo_rows = [[text(cell) for cell in row] for row in demo_table.findall('.//tbody/tr')]
    demo_values = {row[0]: row[2] for row in demo_rows if len(row) == 3}
    expected = {'Mean age, years (min; max)': '39 (18; 65)', 'Female': '10 (24.4)', 'Male': '31 (75.6)',
        'White': '31 (75.6)', 'Black or African American': '8 (19.5)', 'Other': '2 (4.9)',
        'Mean body weight, kg (min; max)': '77.0 (50.7; 95.7)', 'Mean BMI, kg/m 2 (min; max)': '25.0 (20.1; 29.8)'}
    if any(demo_values.get(label) != value for label, value in expected.items()):
        raise ValueError('Furosemide demographic source changed; do not silently remap columns')
    demographics = {'population': 'Trial2 enrolled cohort', 'n_enrolled': 41,
        'age_year': {'mean': 39, 'minimum': 18, 'maximum': 65},
        'body_weight_kg': {'mean': 77.0, 'minimum': 50.7, 'maximum': 95.7},
        'bmi_kg_m2': {'mean': 25.0, 'minimum': 20.1, 'maximum': 29.8},
        'sex_counts': {'female': 10, 'male': 31},
        'race_counts': {'White': 31, 'Black_or_African_American': 8, 'Other': 2},
        'source': {'pmcid': 'PMC8416862', 'table_id': 'Tab1', 'expanded_column_index': 2,
            'source_article_sha256': digest(furo_source.read_bytes())},
        'limitation': 'Demographics of the n39 paired PK subset are not separately reported; never label enrolled n41 as that subset.'}

    records = []
    cases = []
    checks = []
    for old in sorted((previous / 'prediction_metadata').glob('*.json')):
        case = json.loads(old.read_text())
        sealed = case['sealed_observation_reference']
        # Binary integrity verification only: sealed endpoint JSON is not parsed.
        sealed_bytes = inside(ROOT / sealed['path']).read_bytes()
        if digest(sealed_bytes) != sealed['sha256'] or len(sealed_bytes) != sealed['byte_size']:
            raise ValueError('Sealed endpoint checksum changed')
        case['schema_version'] = 'independent-clinical-protocol-3'
        case['supersedes_protocol'] = {'path': str(old.relative_to(ROOT)), 'sha256': digest(old.read_bytes())}
        case['clinical_source_audit']['trial_id_assignment'] = 'Exact actual-trial primary-source curation; background trial references excluded.'
        case['publication_cluster_id'] = case['study_id']
        case['trial_cluster_ids'] = case['clinical_source_audit']['trial_registry_ids']
        # Source resources remain curator-only and must not be automatically fed
        # to the designer just because a metadata file references their hashes.
        case['protocol_resources_are_curator_only'] = True
        case['prediction_contract'] = {
            'source_reference_subject_prediction_allowed': True,
            'must_register_approximations_before_prediction': True,
            'missing_food_or_sampling_may_use_explicit_sensitivity_scenarios': True,
            'missing_protocol_must_not_be_filled_as_observed_fact': True,
            'reference_subject_is_not_group_mean_of_individual_NCA': True,
            'AUC_infinity': 'Match total plasma parent AUC extrapolated to infinity; document integration horizon and residual-tail treatment.',
            'AUC_last': 'Do not equate the last individual quantifiable sample to the scheduled final sample without explicit approximation.',
            'Cmax': 'Use source sampling grid when known; a dense-grid maximum is an approximation to published sampled Cmax.',
            'population_target_exactly_matched_now': False,
            'prediction_freeze_required_before_scoring': True,
        }
        if case['compound'] == 'metformin':
            case['protocol']['active_moiety_dose_mg'] = 195.0
            case['protocol']['active_moiety_dose_basis'] = 'free base converted from the reported 250mg hydrochloride salt'
            case['protocol']['salt_conversion'] = chemical_source
            case['limitations'] = [s for s in case['limitations'] if not s.startswith('Salt-labelled dose must')]
            case['limitations'].append('250mg hydrochloride is approximated as 195mg free base using the primary source rounded 0.78 equivalence; original salt-labelled dose is retained.')
        if case['compound'] == 'furosemide':
            case['protocol']['reported_demographics'] = demographics
            case['protocol']['food_state'] = 'not_reported_in_retrieved_protocol_sources'
            case['protocol']['sampling_times_min'] = None
            case['protocol']['sampling_grid_status'] = 'not_reported; source reports a 12-hour profile duration only'
            case['protocol']['protocol_missingness_audit'] = {
                'checked': ['primary article PMC8416862', 'Online Resource 1 supplemental DOCX', 'ClinicalTrials.gov NCT03010475', 'FDA 2019 clinical pharmacology review pages45-47'],
                'fda_source': 'https://www.accessdata.fda.gov/drugsatfda_docs/nda/2019/213051Orig1s000ClinPharmR.pdf',
                'food_state_resolved': False, 'sampling_grid_resolved': False,
                'no_assumption_from_semaglutide_prescribing_instructions': True}
            case['runtime_readiness'] = 'reference_subject_or_explicit_sensitivity_prediction_allowed_after_adapter_registration; exact_protocol_match_not_established'
            case['limitations'].append('Trial2 enrolled-cohort means can define a preregistered reference subject; they do not recover individual covariates or the n39 PK subgroup.')
        reference = save('prediction_metadata/' + old.name, case)
        records.append({'case_id': case['case_id'], 'compound': case['compound'], 'tier': case['tier'],
            'source_ready': True, 'strict_main_evaluable_now': False, 'protocol_reference': reference})
        cases.append(case)
        checks.append({'case_id': case['case_id'], 'sealed_hash_verified_without_decoding': True,
            'observed_endpoint_values_added_to_metadata': False,
            'actual_trial_assignment_verified': True, 'strict_eligibility_not_promoted': True})

    denylist = {'schema_version': 'clinical-evidence-denylist-2',
        'pmids': sorted({c['identifiers']['pmid'] for c in cases}),
        'pmcids': sorted({c['identifiers']['pmcid'] for c in cases}),
        'dois': sorted({c['identifiers']['doi'] for c in cases}),
        'trial_registry_ids': sorted({i for c in cases for i in c['trial_cluster_ids']}),
        'cohort_ids': sorted({c['cohort_id'] for c in cases}),
        'raw_source_files': sorted({f for c in cases for f in c['evidence_exclusion']['raw_source_files']}),
        'exclude_from': ['designer prompts', 'retrieval', 'parameter sources', 'fitting', 'training', 'search feedback'],
        'shared_generic_physiology_literature_is_not_clinical_overlap': True,
        'whole_raw_batch_exclusion_until_article_aware_mask': True,
        'supersedes': str((cohort / 'evidence_denylist.json').relative_to(ROOT))}
    denyref = save('evidence_denylist.json', denylist)
    checkref = save('static_validation.json', {'passed': True, 'cases': checks, 'solver_calls': 0,
        'sealed_outcomes_modified': False, 'no_manual_label_dependency': True})
    manifest = {'status': 'source_protocols_ready_with_explicit_approximation_contracts',
        'independent_clinical_candidate_drugs': 5, 'protocols': len(records),
        'source_ready_means': 'primary observed endpoints sealed plus outcome-free actual-protocol metadata; does not imply executable adapter or frozen predictions',
        'strict_main_evaluable_now': 0, 'records': records,
        'evidence_denylist': denyref, 'validation': checkref,
        'script': str(Path(__file__).relative_to(ROOT)), 'script_sha256': digest(Path(__file__).read_bytes()),
        'sealed_outcomes_decoded_by_this_step': False, 'solver_calls': 0}
    save('active_protocol_manifest.json', manifest)
    print(json.dumps({'protocols': len(records), 'drugs': 5, 'static_validation': 'passed',
        'outcomes_unchanged': True, 'metformin_salt_conversion_bound': True,
        'furosemide_demographics_bound': True, 'strict_main_evaluable_now': 0}))


if __name__ == '__main__':
    main()
