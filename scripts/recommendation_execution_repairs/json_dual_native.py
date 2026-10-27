"""Native selection/readback for a source-bound two-parent racemate contract."""
from copy import deepcopy
import math
from . import json_racemate_contract as racemate
from . import event_associations, coadministration_isolation, source_pkml


def selection_options(metadata, provenance, contract, source_policy, source, source_sha, *, project_root):
    from . import complete_candidate as complete
    options, failures = [], []
    for row in metadata['simulations']:
        try:
            dual = racemate.prepare_dual_parent_contract(provenance, row['name'], contract, project_root=project_root)
            isolation = None
            if dual['other_administered_drugs']:
                if dual['other_administered_drugs'] != ['Quinidine']:
                    raise ValueError('DUAL_PARENT_ONLY_REGISTERED_QUINIDINE_ISOLATION_ALLOWED')
                isolation = coadministration_isolation.prepare_source_isolation(source, source_sha,
                    row['simulation_id'], row['source_blob_sha256'], row, list(racemate.PARENTS), contract, project_root=project_root)
            administered = {c for a in row['administrations'] for c in a['administered_compounds']}
            if administered != set(racemate.PARENTS) | set(dual['other_administered_drugs']):
                raise ValueError('DUAL_PARENT_EXACT_NATIVE_TWO_CHANNELS_REQUIRED')
            parts, suppressions = [], {}
            for channel in dual['channels']:
                parent = channel['parent']; view = coadministration_isolation.target_metadata(row, parent)
                target = {**contract, 'dose_mg': channel['target_dose_mg']}
                policy = {**source_policy, 'identity_aliases': {contract['compound']: {'native_name': parent,
                    'basis': 'source-bound racemate channel; not an equivalent single-parent alias'}}}
                choices, rejected = complete.select_source_simulation({'simulations': [view]}, target, source_policy=policy)
                if not choices:
                    raise ValueError('DUAL_PARENT_CHANNEL_REJECTED:' + repr(rejected))
                part = choices[0]
                if part.get('application_expansion'):
                    raise ValueError('DUAL_PARENT_EXPANSION_REQUIRES_SEPARATE_PROOF')
                if part['source_event_count'] > len(contract['administration_times_min']):
                    suppressions[parent] = complete.source_suppression_policy(source, source_sha, row['simulation_id'],
                        row['source_blob_sha256'], complete._original_metadata(view), target, project_root=project_root)
                parts.append(part)
            first, second = parts
            if first['source_event_count'] != second['source_event_count']:
                raise ValueError('DUAL_PARENT_NATIVE_CHANNEL_EVENT_COUNTS_DIFFER')
            # Source timing/vehicle equality was independently proved by the JSON
            # contract; native per-channel identities and units are checked again
            # when patches are constructed from the actual native inventory.
            chosen = deepcopy(first); chosen['metadata'] = row
            chosen['identity_alias'] = None
            chosen['dual_parent_contract'] = dual
            chosen['dual_parent_suppression'] = suppressions
            if isolation: chosen['coadministration_isolation'] = isolation
            chosen['sort_key'] = tuple(first['sort_key'][:-3]) + (
                abs(math.log(dual['source_total_dose_mg'] / contract['dose_mg'])), row['name'], row['simulation_id'])
            options.append(chosen)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append({'simulation_id': row.get('simulation_id'), 'simulation': row.get('name'),
                             'reason': type(exc).__name__ + ':' + str(exc)})
    options.sort(key=lambda x: (bool(x.get('coadministration_isolation')), x['sort_key']))
    return options, failures


def source_output_contract(source, source_sha, simulation_id, source_blob_sha, dual, *, project_root):
    from . import complete_candidate as complete
    _, document = event_associations._load(source, source_sha, simulation_id, source_blob_sha, project_root)
    weights = {parent: complete.source_output_contract(source, source_sha, simulation_id, source_blob_sha,
        parent, project_root=project_root)['molecular_weight_g_per_mol'] for parent in racemate.PARENTS}
    proof = racemate.validate_native_observer(document, dual, molecular_weights=weights, project_root=project_root)
    return {**proof, 'observer_node_sha256': proof['native_observer_sha256'],
            'source_model_blob_sha256': source_blob_sha}


def native_output(inventory, plan):
    from . import complete_candidate as complete
    dual = plan['dual_parent_contract']; output = dual['output_path']
    if inventory.get('observer_paths', []).count(output) != 1:
        raise ValueError('DUAL_PARENT_EXACT_NATIVE_TOTAL_OBSERVER_REQUIRED')
    for parent in racemate.PARENTS:
        rows = [p for p in inventory.get('parameters', []) if p.get('path') == parent + '|Molecular weight']
        if len(rows) != 1 or rows[0].get('unit') not in complete.MW_G_MOL:
            raise ValueError('DUAL_PARENT_EXACT_NATIVE_PARENT_MOLECULAR_WEIGHT_REQUIRED')
        mw = rows[0]['value'] * complete.MW_G_MOL[rows[0]['unit']]
        if not math.isfinite(mw) or not math.isclose(mw, dual['molecular_weight_g_per_mol'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('DUAL_PARENT_NATIVE_MOLECULAR_WEIGHT_CHANGED')
    return output, dual['molecular_weight_g_per_mol']


def verify_plan(plan, candidate, *, project_root):
    from . import complete_candidate as complete
    dual = plan['dual_parent_contract']
    racemate.revalidate_contract(dual, plan['contract'], project_root=project_root)
    source = complete.execution_source_reference(plan, candidate, project_root=project_root)
    expected = source_output_contract(source['path'], source['sha256'], plan['source_simulation_id'],
        plan['source_model_blob_sha256'], dual, project_root=project_root)
    if expected != plan['dual_parent_observer'] or expected != plan['source_output_contract']:
        raise ValueError('DUAL_PARENT_FROZEN_NATIVE_OBSERVER_DIFFERS')
    return expected


def evidence_refs(dual):
    return [deepcopy(dual[key]) for key in ('registry', 'source', 'author_document', 'source_receipt', 'verified_snapshot')]


def build_patches(inventory, plan, source_metadata, *, project_root):
    dual = plan['dual_parent_contract']; clinical = plan['contract']
    verify_suppression(plan, source_metadata, project_root=project_root)
    isolation = plan.get('coadministration_isolation')
    kwargs = {'suppression_policies_by_parent':plan['dual_parent_suppression'], 'project_root':project_root}
    if not isolation:
        return racemate.build_dual_parent_patches(inventory, clinical, source_metadata, dual, **kwargs)
    if isolation.get('target') != list(racemate.PARENTS) or dual['other_administered_drugs'] != ['Quinidine']:
        raise ValueError('DUAL_PARENT_EXACT_RACEMATE_QUINIDINE_ISOLATION_REQUIRED')
    coadministration_isolation.verify_source_isolation(isolation, source_metadata, clinical, project_root=project_root)
    def target_builder(filtered, contract, retained_metadata, **unused):
        return racemate.build_dual_parent_patches(filtered, contract, retained_metadata, dual,
            source_isolation=isolation, full_source_metadata=source_metadata, **kwargs)
    return coadministration_isolation.build_isolated_patches(inventory, clinical, source_metadata, isolation,
        target_patch_builder=target_builder)


def verify_suppression(plan, source_metadata, *, project_root):
    from . import complete_candidate as complete
    dual = plan['dual_parent_contract']; contract = plan['contract']; expected = {}
    for channel in dual['channels']:
        parent = channel['parent']; view = coadministration_isolation.target_metadata(source_metadata, parent)
        groups, _ = complete._metadata_events(view)
        if len(groups) > len(contract['administration_times_min']):
            expected[parent] = complete.source_suppression_policy(plan['execution_source_path'], plan['execution_source_sha256'],
                plan['source_simulation_id'], plan['source_model_blob_sha256'], complete._original_metadata(view),
                {**contract, 'dose_mg':channel['target_dose_mg']}, project_root=project_root)
    if expected != plan['dual_parent_suppression']:
        raise ValueError('DUAL_PARENT_FROZEN_SURPLUS_EVENT_POLICY_DIFFERS')
    return expected
