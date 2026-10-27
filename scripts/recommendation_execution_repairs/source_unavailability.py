"""Independently prove one frozen author's missing input; never invent a model.

The original failed preservation receipt remains failed. This separate finding
permits complete coverage accounting, not execution success or applicability.
No clinical outcome is opened and no unregistered failure is waived.
"""
from pathlib import Path
import json
from .native_pkml import inside, sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = {
    'path': 'artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Repaglinide-DDI__52112a1cb753/source/Clopidogrel-Repaglinide-DDI.json',
    'sha256': '4a01131810370d7a6712721ecb28fca8575b758442457d9de6865d8e35bf7ba1'}
AUDIT = {
    'path': 'runtime/deployment/source_failure_diagnostics_20261002/missing_formulation_audit.json',
    'sha256': '3cb13c2c0b1b7303cbc0beacfa227c3cb8f2d539d4e7b6724537983a15abbdb9'}
RECEIPT = {
    'path': 'experiments/model_recommendation/execution_validation_20261001/engineering_repairs/snapshot_repairs_11/' + SOURCE['sha256'] + '/receipt.json',
    'sha256': '6f63d6a78ef33a51a477f6905062fe0924820ecccc5d28dbd66d35175b1596ae'}
FORMULATION = {'Name': 'Repaglinide (Zhu 2014 dissolution profile)_DT', 'FormulationType': 'Formulation_Table'}
# Keep the author spelling exact; do not silently normalize identifiers.
SIMULATIONS = frozenset({'3.2.R_Tornio et al 2014_male_DDI', '3.3.R_Tornio et al 2014_male_CONTROL'})
CANDIDATE = '58ffb154-0d9c-53b1-8702-cc7e0f18d3ab'
PAIRS = frozenset({
    ('clinical:clopidogrel_pmc4422294_reference_plavix75mg_fasted', CANDIDATE),
    ('clinical:repaglinide_pmc9199882_stage2_day1_monotherapy05mg', CANDIDATE)})
SOURCE_FAILURE = 'ValueError:BOUNDED_NONEMPTY_SIMULATION_LIST_REQUIRED'
PLAN_FAILURE = 'VERIFIED_NATIVE_MIGRATION_UNAVAILABLE:SNAPSHOT_REPAIR_NOT_VERIFIED:' + repr([SOURCE_FAILURE])


def _checked(ref, root):
    path = inside(ref['path'], root)
    if sha(path) != ref['sha256']:
        raise ValueError('SOURCE_UNAVAILABILITY_EVIDENCE_CHECKSUM_MISMATCH:' + ref['path'])
    return path


def _verified_evidence(root):
    refs = [SOURCE, AUDIT, RECEIPT]
    audit = json.loads(_checked(AUDIT, root).read_text())
    source = json.loads(_checked(SOURCE, root).read_text())
    receipt = json.loads(_checked(RECEIPT, root).read_text())
    if (audit.get('original_source') != SOURCE or audit.get('missing_formulation') != FORMULATION or
            audit.get('status') != 'original_formulation_input_missing; exact_recovery_not_established' or
            audit.get('exact_original_table_identity_verified') is not False or
            audit.get('reference_curve_substituted') is not False or
            any(audit.get(key) is not False for key in ('clinical_outcomes_accessed', 'fitting_performed', 'failure_is_scientific_inapplicability'))):
        raise ValueError('ORIGINAL_MISSING_INPUT_AUDIT_SCOPE_DIFFERS')
    audit_pairs = audit.get('affected_frozen_pairs', [])
    if (len(audit_pairs) != len(PAIRS) or
            {(p.get('scenario_id'), p.get('candidate_id')) for p in audit_pairs} != PAIRS):
        raise ValueError('ORIGINAL_MISSING_INPUT_PAIR_SET_DIFFERS')
    names = audit.get('source_simulations_requiring_missing_formulation', [])
    if len(names) != len(SIMULATIONS) or set(names) != SIMULATIONS:
        raise ValueError('ORIGINAL_MISSING_INPUT_AUDIT_SIMULATIONS_DIFFER')
    formulations = source.get('Formulations', [])
    matches = [f for f in formulations if f.get('Name') == FORMULATION['Name']]
    if source.get('Version') != 77 or matches != [FORMULATION]:
        raise ValueError('EXACT_ORIGINAL_EMPTY_FORMULATION_REQUIRED')
    simulations = source.get('Simulations', [])
    if (len(simulations) != len(SIMULATIONS) or
            {s.get('Name') for s in simulations} != SIMULATIONS):
        raise ValueError('COMPLETE_ORIGINAL_MISSING_INPUT_SIMULATION_SET_REQUIRED')
    facts = []
    for simulation in simulations:
        dependencies = [(c.get('Name'), f) for c in simulation.get('Compounds', [])
                        for f in c.get('Protocol', {}).get('Formulations', [])
                        if f.get('Name') == FORMULATION['Name']]
        if dependencies != [('Repaglinide_DT', {'Name': FORMULATION['Name'], 'Key': 'Formulation'})]:
            raise ValueError('EVERY_ORIGINAL_SIMULATION_MUST_REQUIRE_THE_MISSING_INPUT')
        facts.append({'simulation': simulation['Name'], 'compound': 'Repaglinide_DT',
                      'required_formulation': FORMULATION['Name']})
    if (receipt.get('source_path') != SOURCE['path'] or receipt.get('source_sha256') != SOURCE['sha256'] or
            receipt.get('status') != 'failed' or receipt.get('reasons') != [SOURCE_FAILURE] or
            any(receipt.get(key) is not False for key in ('ode_run', 'sealed_observations_accessed',
                                                         'fitting_performed', 'scientific_inapplicability_inferred'))):
        raise ValueError('ORIGINAL_FAILED_SOURCE_RECEIPT_DIFFERS')
    # Keep all existing acquisition and native causal evidence bound. A similar
    # curve in another model is documented evidence, never a substitute input.
    for key in ('native_diagnostic', 'same_commit_tree', 'published_releases', 'available_reference_formulation'):
        ref = audit[key]
        _checked(ref, root)
        refs.append(ref)
    log = _checked(audit['native_diagnostic'], root).read_text()
    if ("Table formulation '" + FORMULATION['Name'] + "' requires at least one point" not in log or
            "Cannot load Simulation '3.2.R_Tornio et al 2014_male_DDI'" not in log or
            'DIRECT_SNAPSHOT_LOAD_DONE' not in log):
        raise ValueError('ORIGINAL_MISSING_INPUT_NATIVE_DIAGNOSIS_NOT_ESTABLISHED')
    for ref in refs:
        _checked(ref, root)
    return refs, facts


def known_unavailability_refs(*, project_root=ROOT):
    refs, _ = _verified_evidence(Path(project_root).resolve())
    return [dict(ref) for ref in refs]


def assess_source_unavailable(row, *, provenance=None, project_root=ROOT):
    result = {'accepted': False, 'category': 'engineering_repair_required',
              'scientific_applicability': 'unknown', 'verified_references': [], 'reasons': []}
    try:
        if (row.get('source_sha256') != SOURCE['sha256'] or row.get('status') != 'failed' or
                row.get('reasons') != [SOURCE_FAILURE] or row.get('receipt') != RECEIPT):
            raise ValueError('UNREGISTERED_SOURCE_FAILURE')
        if provenance is not None and any(provenance.get('source_' + key) != SOURCE[key] for key in ('path', 'sha256')):
            raise ValueError('ORIGINAL_SOURCE_PROVENANCE_DIFFERS')
        refs, facts = _verified_evidence(Path(project_root).resolve())
        result.update(accepted=True, category='original_source_required_input_absent',
                      verified_references=refs, source_facts=facts,
                      original_preservation_receipt_status='failed',
                      source_execution_verified=False, numerical_prediction_available=False,
                      reasons=['author_snapshot_omits_required_dissolution_table_points'])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result['reasons'] = [type(exc).__name__ + ':' + str(exc)]
    return result


def assess_plan_unavailable(plan, scenario, candidate, *, project_root=ROOT):
    pair = (scenario.get('scenario_id'), candidate.get('candidate_id'))
    if (pair not in PAIRS or (plan.get('scenario_id'), plan.get('candidate_id')) != pair or
            plan.get('reasons') != ['Blocked:' + PLAN_FAILURE]):
        raise ValueError('UNREGISTERED_ORIGINAL_INPUT_ABSENCE_PLAN')
    provenance = candidate['catalog_provenance']
    row = {'source_sha256': SOURCE['sha256'], 'status': 'failed', 'reasons': [SOURCE_FAILURE], 'receipt': RECEIPT}
    result = assess_source_unavailable(row, provenance=provenance, project_root=project_root)
    if not result['accepted']:
        raise ValueError('ORIGINAL_INPUT_ABSENCE_NOT_VERIFIED:' + str(result['reasons']))
    return result
