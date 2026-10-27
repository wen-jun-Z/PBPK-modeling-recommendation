"""Clinical-design execution of a complete, original PK-Sim candidate project.

Preparation is outcome-free and does not export or solve a model. Execution
exports the exact saved simulation and changes only registered administration
scalars and output times. Extra source doses are zeroed and their proven time
triggers are moved beyond the frozen output window.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from recommendation_candidate_executor import (
    prediction_from_curve, _digest, _form, _name, Blocked,
    IMPLEMENTATION_FILES as LEGACY_IMPLEMENTATION_FILES,
)
from recommendation_execution_repairs import source_pkml, native_pkml, repeated_application, event_associations, coadministration_isolation
from recommendation_execution_repairs.license_audit import audit_candidate_license, DEFAULT_RECEIPT
from .clinical_contract import scenario_contract
from . import mixed_application_expansion as mixed_application
from .surplus_event_suppression import source_suppression_policy

VERSION = 'complete-original-project-candidate-v1'
EXECUTION_REVISION = 'engineering_repair_04'
_RECENT_PREPARATIONS = {}
IMPLEMENTATION_FILES = tuple(dict.fromkeys(LEGACY_IMPLEMENTATION_FILES + repeated_application.IMPLEMENTATION_FILES + mixed_application.IMPLEMENTATION_FILES + coadministration_isolation.IMPLEMENTATION_FILES + tuple(event_associations.IMPLEMENTATION_FILES) + (
    'scripts/recommendation_execution_repairs/complete_candidate.py',
    'scripts/recommendation_execution_repairs/split_absorption.py',
    'scripts/recommendation_execution_repairs/clinical_contract.py',
    'scripts/recommendation_execution_repairs/source_bound_dose.py',
    'scripts/recommendation_execution_repairs/surplus_event_suppression.py',
    'scripts/recommendation_execution_repairs/source_pkml.py',
    'scripts/recommendation_execution_repairs/native_pkml.py',
    'scripts/recommendation_execution_repairs/model_semantics.py',
    'scripts/recommendation_execution_repairs/license_audit.py',
    'scripts/recommendation_candidate_executor.py',
    'scripts/recommendation_candidate_migration.py',
    'src/pbpk_kb/blueprints/design_runtime.py',
    'src/pbpk_kb/model_runtime/R/recommendation_saved_model_core_export.R',
    'src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R',
)))
MASS_MG = {'kg': 1e6, 'g': 1e3, 'mg': 1., 'ug': .001, 'µg': .001}
TIME_MIN = {'min': 1., 'h': 60., 's': 1/60., 'day(s)': 1440.}
MW_G_MOL = {'kg/µmol': 1e9, 'kg/umol': 1e9, 'g/mol': 1., 'kg/mol': 1000.}
_METADATA_CACHE = {}
PREGNANCY_REGISTRY = 'artifacts/pbpk-model-assets/execution_repairs_20261001/pregnancy_native_models/registry.json'
PREGNANCY_MARKERS = {'AmnioticFluid', 'Fetus', 'PlacentaFetal', 'PlacentaMaternal'}
SEROQUEL_RELEASE_SOURCE = 'https://www.ema.europa.eu/en/medicines/human/referrals/seroquel-seroquel-xr-associated-names'
SOURCE_BOUND_ALIASES = {
    'c42ae88dd0fc45df975395c1f2504097199b1344484e5cde6cf2a4fd4f1b01e6': {
        'compound': 'caffeine', 'native_name': 'Caffeine_template',
        'commit': '9fbb41b59b52497faffefbbec0c50de45103d84f',
        'readme': 'artifacts/pbpk-model-assets/public_collection_20260909/Pregnancy-Models__9fbb41b59b52__selected_sources/source/README.md',
        'readme_sha256': 'fde6008c2e4a7dc4bbfaf382a7fea7a1694278a13fc125d12798cf62c753f985',
        'basis': 'Author README identifies Caffeine in Models/Caffeine; exact original saved parent is Caffeine_template.'},
    '5b842bbc36eb315477266e339999ffa760b7a3d3c91a30af46e120d28a86012a': {
        'compound': 'repaglinide', 'native_name': 'Repaglinide_2018-07-02',
        'commit': 'bdbc59b20bb6da39572ce75856fa41bd33a4648b',
        'readme': 'artifacts/pbpk-model-assets/public_collection_20260909/Itraconazole-Repaglinide-DDI__bdbc59b20bb6/source/README.md',
        'readme_sha256': '1a6b77d028a580498593a7a06bf50c630168ec67aa037d4aa9e15409702d56a8',
        'basis': 'Author README identifies itraconazole-repaglinide clinical DDI/control project; exact original parent in repaglinide-only control is Repaglinide_2018-07-02.'},
}


def source_bound_aliases(provenance, root):
    rule = SOURCE_BOUND_ALIASES.get(provenance['source_sha256'])
    if not rule: return {}
    path = inside(rule['readme'], root)
    if provenance.get('repository_commit') != rule['commit'] or sha(path) != rule['readme_sha256']:
        raise Blocked('AUTHOR_PARENT_IDENTITY_SOURCE_OR_COMMIT_MISMATCH')
    return {rule['compound']: {'native_name': rule['native_name'], 'source_sha256': provenance['source_sha256'],
                              'author_project_documentation': reference(path, root), 'basis': rule['basis']}}


def _release(text, compound=None):
    value = str(text or '').lower()
    if re.search(r'\b(?:xr|er|sr|cr|mr|xl|extended[ -]release|prolonged[ -]release|sustained[ -]release|controlled[ -]release|modified[ -]release)\b', value):
        return 'modified_release'
    if re.search(r'\b(?:ir|immediate[ -]release)\b', value): return 'immediate_release'
    if compound == 'quetiapine' and re.search(r'\bseroquel\b', value):
        return 'immediate_release'  # EMA differentiates named Seroquel and Seroquel XR.
    if re.search(r'\b(?:solution|dissolved)\b', value): return 'solution'
    return 'unknown'


def _food_state(text):
    value = str(text or '').lower()
    if re.search(r'not (?:explicitly )?(?:stated|reported|known|available)|unknown|do not infer|unreported|unspecified', value):
        return 'unknown'
    fast = bool(re.search(r'\b(?:fasted|fasting)\b', value))
    fed = bool(re.search(r'\bfed\b', value))
    return 'fasted' if fast and not fed else 'fed' if fed and not fast else 'unknown'


def pregnancy_source_policy(candidate, source, metadata, *, project_root=ROOT):
    """Preserve pregnancy identity using exact original model and population XML.

    Registry/proof references select the frozen original version, while the
    Human/Pregnant population and structural markers are read again from its
    original saved simulation. A same-project nonpregnant control is ineligible.
    """
    root = Path(project_root); prov = candidate.get('provenance', candidate.get('catalog_provenance', {}))
    registry_path = inside(PREGNANCY_REGISTRY, root, exists=False)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'records': []}
    rows = [r for r in registry['records'] if r['candidate_id'] == candidate.get('candidate_id')]
    if not rows:
        if 'pregnan' in source.name.lower(): raise Blocked('PREGNANCY_CANDIDATE_EXACT_SOURCE_BINDING_REQUIRED')
        return {'pregnancy_candidate': False, 'simulations': {}, 'identity_aliases': source_bound_aliases(prov, root)}
    source_ref = {'path': str(source.relative_to(root)), 'sha256': prov['source_sha256']}
    simulations = {}; aliases = {}
    source_rows = {s['simulation_id']: s for s in metadata['simulations']}
    with sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        for row in rows:
            if row['source'] != source_ref or row['repository_commit'] != prov.get('repository_commit'):
                raise Blocked('PREGNANCY_REGISTRY_SOURCE_OR_COMMIT_MISMATCH')
            sid = row['simulation']['simulation_id']; registered = source_rows.get(sid)
            if registered is None: raise Blocked('PREGNANCY_REGISTERED_SIMULATION_NOT_IN_ORIGINAL_SOURCE')
            proof_path = inside(row['proof']['path'], root)
            if sha(proof_path) != row['proof']['sha256']: raise Blocked('PREGNANCY_PROOF_CHECKSUM_MISMATCH')
            proof = json.loads(proof_path.read_text())
            if proof.get('source') != source_ref or proof.get('simulation', {}).get('simulation_id') != sid or proof.get('source_blob_sha256') != registered['source_blob_sha256']:
                raise Blocked('PREGNANCY_PROOF_ORIGINAL_SIMULATION_MISMATCH')
            blobs = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                               'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?', (sid,)).fetchall()
            if len(blobs) != 1 or hashlib.sha256(blobs[0][0]).hexdigest() != registered['source_blob_sha256']:
                raise Blocked('PREGNANCY_ORIGINAL_MODEL_BLOB_MISMATCH')
            original = ET.fromstring(source_pkml.decompress_model_blob(blobs[0][0]))
            origins = [dict(o.attrib) for o in original.findall('./UsedBuildingBlocks//OriginData')]
            model = original.find('Model')
            organisms = [x for x in model.iter() if x.get('name') == 'Organism']
            if len(organisms) != 1: raise Blocked('EXACT_ORIGINAL_ORGANISM_REQUIRED')
            names = {x.get('name') for x in organisms[0].findall('./Children/*')}
            markers = sorted(PREGNANCY_MARKERS.intersection(names))
            pregnant = set(markers) == PREGNANCY_MARKERS
            human_pregnant = bool(origins) and all(o.get('species') == 'Human' and o.get('speciesPopulation') == 'Pregnant' for o in origins)
            if markers != row['pregnancy_markers']: raise Blocked('PREGNANCY_REGISTRY_STRUCTURE_MISMATCH')
            simulations[sid] = {'pregnancy_structure_preserved': pregnant, 'human_pregnant_origin': human_pregnant,
                'original_origin_data': origins, 'pregnancy_markers': markers,
                'source_model_blob_sha256': registered['source_blob_sha256'], 'proof': row['proof']}
    aliases = source_bound_aliases(prov, root)
    if sha(source) != source_ref['sha256']: raise Blocked('PREGNANCY_SOURCE_CHANGED_DURING_IDENTITY_READ')
    return {'pregnancy_candidate': True, 'registry': reference(registry_path, root),
            'simulations': simulations, 'identity_aliases': aliases,
            'population_application': 'Original saved reference only; no covariate reconstruction or gestational-age inference.'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def inside(path, root=ROOT, *, exists=True):
    return native_pkml.inside(path, root, exists=exists)


def save(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    if len(data.encode()) > 32 * 1024**2: raise ValueError('COMPLETE_CANDIDATE_JSON_SIZE_LIMIT')
    with Path(path).open('x') as f: f.write(data)


def reference(path, root):
    p = inside(path, root)
    return {'path': str(p.relative_to(root)), 'sha256': sha(p)}


def implementation_hashes(project_root=ROOT, *, verified_json_source=False):
    paths = IMPLEMENTATION_FILES
    if verified_json_source:
        from .json_native_candidate import IMPLEMENTATION_FILES as BRIDGE_FILES
        paths = BRIDGE_FILES
    return {path: sha(inside(path, project_root)) for path in paths}


def _execution_implementation_hashes(root, proof):
    return implementation_hashes(root, verified_json_source=True) if proof is not None else implementation_hashes(root)


def execution_source_reference(plan, candidate, *, project_root=ROOT):
    """Revalidate both source identity and the exact authenticated native bridge."""
    root = Path(project_root)
    prov = candidate.get('provenance', candidate.get('catalog_provenance', {}))
    original = {'path': prov['source_path'], 'sha256': prov['source_sha256']}
    if (plan.get('source_path') != original['path'] or plan.get('source_sha256') != original['sha256']
            or sha(inside(original['path'], root)) != original['sha256']):
        raise Blocked('ORIGINAL_CATALOGUE_SOURCE_IDENTITY_CHANGED')
    if plan.get('adapter_route') != 'verified_json_native_pksim5':
        if plan.get('json_native_source_proof') or plan.get('execution_source_path'):
            raise Blocked('UNREGISTERED_NATIVE_SOURCE_OVERRIDE')
        return original
    from .json_native_source_proof import verify_native_source
    verified = verify_native_source(prov, plan.get('json_native_source_proof'), project_root=root)
    native = verified['native_source']
    if (verified != plan['json_native_source_proof'] or
            native['path'] != plan.get('execution_source_path') or
            native['sha256'] != plan.get('execution_source_sha256')):
        raise Blocked('JSON_NATIVE_SOURCE_PROOF_CHANGED')
    allowed = [r for r in verified['allowed_simulations'] if r['id'] == plan['source_simulation_id']]
    if (len(allowed) != 1 or allowed[0]['name'] != plan['source_simulation'] or
            allowed[0]['source_blob_sha256'] != plan['source_model_blob_sha256']):
        raise Blocked('JSON_NATIVE_SELECTED_SIMULATION_NOT_PROVEN')
    return native


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _event_prefix(path):
    match = re.fullmatch(r'((?:Events|Applications)\|.+\|Application_[1-9][0-9]*)\|ProtocolSchemaItem\|(Dose|Start time|Infusion time)', path)
    if not match: raise Blocked('UNREGISTERED_ADMINISTRATION_PATH:' + str(path))
    return match[1], match[2]


def _normalized_event(path):
    return re.sub(r'^Applications\|', 'Events|', path)


def _preserved_meal_parameters(metadata):
    rows = []
    for row in metadata.get('event_parameters', []):
        if re.fullmatch(r'Events\|Standard Meal\|Standard Meal_[1-9][0-9]*\|Start time', row['path']):
            if row.get('dimension') != 'Time': raise Blocked('ORIGINAL_MEAL_TIME_DIMENSION_REQUIRED')
            if any(a['event_prefix'] == row['path'].rsplit('|', 1)[0] for a in metadata.get('administrations', [])):
                raise Blocked('MEAL_NAME_CANNOT_HIDE_DRUG_ADMINISTRATION')
            rows.append(row)
    return rows


def _metadata_events(metadata, preserved_physiology=None):
    groups = {}
    preserved_meal_paths = {row['path'] for row in _preserved_meal_parameters(metadata)}
    if preserved_physiology:
        if preserved_physiology.get('verified') is not True or preserved_physiology.get('simulation_id')!=metadata['simulation_id']:
            raise Blocked('SOURCE_PHYSIOLOGY_PROOF_IDENTITY_DIFFERS')
        by_path={r['path']:r for r in metadata['event_parameters']}
        for row in preserved_physiology['preserved_event_parameters']:
            if row['path'] in by_path and by_path[row['path']]!=row:raise Blocked('ORIGINAL_PHYSIOLOGY_PARAMETER_DIFFERS')
            preserved_meal_paths.add(row['path'])
    for row in metadata.get('event_parameters', []):
        if row['path'] in preserved_meal_paths: continue
        prefix, field = _event_prefix(row['path'])
        key = _normalized_event(prefix)
        group = groups.setdefault(key, {})
        if field in group: raise Blocked('DUPLICATE_SOURCE_ADMINISTRATION_PARAMETER')
        group[field] = row
    if not groups or any(not {'Dose', 'Start time'}.issubset(x) for x in groups.values()):
        raise Blocked('COMPLETE_SOURCE_ADMINISTRATION_DOSE_AND_TIME_REQUIRED')
    assignments = {}
    for row in metadata.get('administrations', []):
        key = _normalized_event(row['event_prefix'])
        if key in assignments: raise Blocked('DUPLICATE_SOURCE_ADMINISTRATION_ASSIGNMENT')
        assignments[key] = row
    if set(assignments) != set(groups): raise Blocked('SOURCE_EVENT_COMPOUND_ASSOCIATION_INCOMPLETE')
    for key, assignment in assignments.items():
        if assignment.get('compound_association_complete') is not True:
            raise Blocked('SOURCE_EVENT_COMPOUND_ASSOCIATION_INCOMPLETE')
        if not assignment.get('assignment_targets'):
            raise Blocked('SOURCE_EVENT_ASSIGNMENT_TARGETS_REQUIRED')
    return groups, assignments


def select_source_simulation(metadata, contract, *, source_policy=None):
    """Select using fixed identity/route/design metadata, never performance."""
    target_count = len(contract['administration_times_min'])
    source_policy = source_policy or {'pregnancy_candidate': False, 'simulations': {}, 'identity_aliases': {}}
    alias = source_policy['identity_aliases'].get(contract['compound'])
    expected_native_parent = _name(alias['native_name']) if alias else contract['compound']
    options, excluded = [], []
    for source in metadata.get('simulations', []):
        try:
            if source.get('saved_type') not in {'IndividualSimulation', 'PopulationSimulation'}:
                raise Blocked('UNSUPPORTED_ORIGINAL_SIMULATION_TYPE')
            if source.get('species') not in ([], ['Human']):
                raise Blocked('EXPLICIT_HUMAN_SOURCE_REQUIRED')
            pregnancy = source_policy['simulations'].get(source['simulation_id'], {})
            if source_policy['pregnancy_candidate']:
                if pregnancy.get('pregnancy_structure_preserved') is not True:
                    raise Blocked('PREGNANCY_CANDIDATE_CANNOT_USE_NONPREGNANT_CONTROL')
                if pregnancy.get('human_pregnant_origin') is not True:
                    raise Blocked('ORIGINAL_HUMAN_PREGNANT_ORIGIN_REQUIRED')
                species_evidence = 'Original UsedBuildingBlocks OriginData Human/Pregnant and all four pregnancy Model markers'
            elif not str(source.get('model_configuration', '')).startswith('4Comp|Human'):
                raise Blocked('ORIGINAL_HUMAN_4COMP_MODEL_REQUIRED')
            else:
                species_evidence = 'Original complete model_configuration=4Comp|Human; OriginData when present agrees'
            compound_names = [c for c in source.get('compounds', []) if _name(c) == expected_native_parent]
            if len(compound_names) != 1: raise Blocked('EXACT_UNIQUE_PARENT_IN_SOURCE_REQUIRED')
            physiology=source_policy.get('preserved_physiology_policies',{}).get(source['simulation_id'])
            groups, assignments = _metadata_events(source,physiology)
            for key, row in assignments.items():
                if [_name(c) for c in row.get('administered_compounds', [])] != [expected_native_parent]:
                    raise Blocked('COADMINISTERED_OR_UNRESOLVED_SOURCE_DRUG_NOT_ALLOWED')
                if row.get('application_type') != contract['route']:
                    raise Blocked('SOURCE_ROUTE_DIFFERS')
                if contract['route'] == 'Intravenous' and 'Infusion time' not in groups[key]:
                    raise Blocked('SOURCE_INFUSION_DURATION_PARAMETER_REQUIRED')
            expansion = None
            if len(groups) < target_count:
                expansion = source_policy.get('application_expansion_policies', {}).get(source['simulation_id'])
                mixed_ok=(expansion and expansion.get('version')==mixed_application.VERSION and
                    expansion.get('verified') is True and expansion.get('target')==compound_names[0] and
                    expansion.get('contract')==contract and expansion.get('source_target_event_count')==len(groups) and
                    expansion.get('executable_target_event_count')==target_count and
                    expansion.get('source_reference',{}).get('simulation_id')==source['simulation_id'] and
                    expansion.get('source_reference',{}).get('source_blob_sha256')==source['source_blob_sha256'])
                if not mixed_ok and (not expansion or expansion.get('verified') is not True or len(groups) != 1 or
                        contract['route'] != 'Oral' or expansion.get('original_source_event_count') != 1 or
                        expansion.get('executable_event_count') != target_count or
                        expansion.get('compound_name') != compound_names[0] or
                        expansion.get('administration_times_min') != contract['administration_times_min'] or
                        expansion.get('dose_mg') != contract['dose_mg'] or
                        expansion.get('source_reference', {}).get('simulation_id') != source['simulation_id'] or
                        expansion.get('source_reference', {}).get('source_blob_sha256') != source['source_blob_sha256']):
                    raise Blocked('SOURCE_HAS_FEWER_EVENTS_THAN_FROZEN_TARGET_REGIMEN')
            if len(groups) * 3 > 256: raise Blocked('SOURCE_EVENT_PATCH_BOUND_EXCEEDED')
            # Source XML values use native base units (kg and minutes), never
            # displayUnit. Formula-bound values are ranked after explicit ones.
            doses = []
            for group in groups.values():
                dose = group['Dose']
                try: value = float(dose.get('value'))
                except (TypeError, ValueError): continue
                if dose.get('dimension') == 'Mass' and math.isfinite(value) and value > 0:
                    doses.append(value * 1e6)
            distance = min((abs(math.log(d / contract['dose_mg'])) for d in doses), default=1e6)
            source_forms = sorted({f for item in source.get('formulations', [])
                                   if (f := _form(item.get('name')) or _form(item.get('type')))})
            wanted_form = _form(contract.get('formulation'))
            form_penalty = 0 if wanted_form and source_forms == [wanted_form] else 1
            target_release = _release(contract.get('formulation'), contract['compound'])
            source_releases = sorted({_release(item.get('name')) for item in source.get('formulations', [])})
            source_release = source_releases[0] if len(source_releases) == 1 else 'unknown'
            # Known nonmodified and modified release have distinct absorption
            # behavior. A solution is an explicit approximation to IR, whereas
            # selecting XR solely because both names say tablet is not justified.
            release_penalty = (0 if target_release == 'unknown' or target_release == source_release else
                               1 if {target_release, source_release} <= {'immediate_release', 'solution'} else
                               2 if source_release == 'unknown' else 3)
            target_food = _food_state(contract.get('food')); source_food = _food_state(source['name'])
            if target_food == 'fasted' and source_food == 'fed': raise Blocked('EXPLICIT_SOURCE_FED_TARGET_FASTED_CONFLICT')
            if target_food == 'fed' and source_food == 'fasted': raise Blocked('EXPLICIT_SOURCE_FASTED_TARGET_FED_CONFLICT')
            food_penalty = 0 if target_food != 'unknown' and target_food == source_food else 1
            options.append({'metadata': source, 'native_compound_name': compound_names[0],
                            'source_event_count': len(groups), 'source_formulation_classes': source_forms,
                            'application_expansion': expansion,
                            'target_formulation_class': wanted_form, 'source_species_evidence': species_evidence,
                            'identity_alias': alias, 'pregnancy_evidence': pregnancy if source_policy['pregnancy_candidate'] else None,
                            'source_release': source_release, 'target_release': target_release,
                            'target_food_state': target_food, 'source_food_state': source_food,
                            'sort_key': (bool(expansion), release_penalty, form_penalty, food_penalty, abs(len(groups)-target_count), distance, source['name'], source['simulation_id'])})
            if physiology:options[-1]['preserved_physiology_policy']=physiology
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({'simulation_id': source.get('simulation_id'), 'simulation': source.get('name'), 'reason': str(exc)})
    options.sort(key=lambda r: r['sort_key'])
    return options, excluded


def source_output_contract(source, source_sha, simulation_id, source_blob_sha, compound_name, *, project_root=ROOT):
    """Freeze a parent observer and MW from the exact original model XML only.

    Saved PK-Sim Model scalar values use kg/µmol for Molecular weight. The
    independent native readback must subsequently report the same g/mol value;
    displayUnit is never treated as the unit of the serialized scalar.
    """
    root = Path(project_root).resolve(strict=True); path = inside(source, root)
    if sha(path) != source_sha: raise Blocked('ORIGINAL_SOURCE_OUTPUT_CHECKSUM_MISMATCH')
    with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                          'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=?', (simulation_id,)).fetchall()
    if len(rows) != 1 or hashlib.sha256(rows[0][0]).hexdigest() != source_blob_sha:
        raise Blocked('EXACT_SELECTED_SOURCE_MODEL_BLOB_REQUIRED')
    document = ET.fromstring(source_pkml.decompress_model_blob(rows[0][0]))
    if document.get('id') != simulation_id: raise Blocked('SOURCE_MODEL_SIMULATION_ID_MISMATCH')
    model = document.find('Model')
    parameters = source_pkml.model_parameter_inventory(model)
    parameter_path = compound_name+'|Molecular weight'
    if parameter_path not in parameters: raise Blocked('EXPLICIT_ORIGINAL_PARENT_MOLECULAR_WEIGHT_REQUIRED')
    row = parameters[parameter_path]
    if row.get('dimension') != 'Molecular weight': raise Blocked('EXACT_MOLECULAR_WEIGHT_DIMENSION_REQUIRED')
    if row.get('rhs_formula_id'): raise Blocked('DYNAMIC_MOLECULAR_WEIGHT_NOT_SUPPORTED')
    raw = row.get('value')
    if row.get('formula_id'):
        formulas = [f for f in model.iter() if f.get('id') == row['formula_id']]
        if len(formulas) != 1 or formulas[0].tag != 'ConstantFormula' or len(formulas[0]):
            raise Blocked('NONCONSTANT_ORIGINAL_MOLECULAR_WEIGHT_NOT_SUPPORTED')
        formula_value = formulas[0].get('value')
        if raw is not None and float(raw) != float(formula_value): raise Blocked('CONFLICTING_ORIGINAL_MOLECULAR_WEIGHT_VALUES')
        raw = formula_value
    try: base_value = float(raw)
    except (TypeError, ValueError): raise Blocked('EXPLICIT_ORIGINAL_MOLECULAR_WEIGHT_VALUE_REQUIRED')
    if not _finite(base_value, positive=True): raise Blocked('POSITIVE_ORIGINAL_MOLECULAR_WEIGHT_REQUIRED')
    observers = {}
    model_root = model.find('Root')
    def walk(node, names):
        if node is not model_root and 'id' in node.attrib and 'name' in node.attrib:
            names = names + [node.get('name')]
            if node.tag in {'Observer', 'Obs'}:
                key = '|'.join(names)
                if key in observers: raise Blocked('DUPLICATE_ORIGINAL_OBSERVER_PATH')
                observers[key] = node
        for child in node: walk(child, names)
    walk(model_root, [])
    choices = [f'Organism|PeripheralVenousBlood|{compound_name}|Plasma (Peripheral Venous Blood)',
               f'Organism|VenousBlood|Plasma|{compound_name}|Concentration in container']
    matches = [p for p in choices if p in observers]
    if not matches: raise Blocked('EXACT_ORIGINAL_PARENT_TOTAL_PLASMA_OBSERVER_REQUIRED')
    selected = matches[0]
    if sha(path) != source_sha: raise Blocked('SOURCE_CHANGED_DURING_OUTPUT_CONTRACT_READ')
    return {'output_path': selected, 'molecular_weight_g_per_mol': base_value * 1e9,
            'molecular_weight_source': {**row, 'native_base_unit': 'kg/µmol', 'base_value': base_value,
                                       'conversion': 'kg/µmol * 1e9 = g/mol; independent native unit readback required'},
            'observer_node_sha256': source_pkml.element_sha(observers[selected]),
            'source_model_blob_sha256': source_blob_sha}


def _original_metadata(row):
    return {k: v for k, v in row.items() if k not in {'mode', 'content_id', 'compressed_bytes', 'source_blob_sha256'}}


def enriched_selection_metadata(metadata, source, source_sha, *, project_root=ROOT):
    """Prove missing indirect IV identities without changing the source record."""
    selection = deepcopy(metadata); proofs = {}; failures = []
    for row in selection.get('simulations', []):
        if not any(a.get('compound_association_complete') is not True for a in row.get('administrations', [])):
            continue
        try:
            proof = event_associations.enrich_source_metadata(source, source_sha, row['simulation_id'],
                row['source_blob_sha256'], metadata=_original_metadata(row), project_root=project_root)
            proofs[row['simulation_id']] = proof
            row.update(proof['enriched_metadata'])
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, ET.ParseError) as exc:
            failures.append({'simulation_id': row.get('simulation_id'), 'reason': str(exc)})
    return selection, proofs, failures


def source_expansion_policies(metadata, excluded, contract, source_policy, source, source_sha, *, project_root=ROOT):
    """Qualify only otherwise eligible short-regimen sources by their full XML."""
    allowed = {r['simulation_id'] for r in excluded
               if r.get('reason') == 'SOURCE_HAS_FEWER_EVENTS_THAN_FROZEN_TARGET_REGIMEN'}
    policies = {}; failures = []
    if contract['route'] != 'Oral' or len(contract['administration_times_min']) <= 1:
        return policies, failures
    alias = source_policy.get('identity_aliases', {}).get(contract['compound'])
    expected = _name(alias['native_name']) if alias else contract['compound']
    for row in metadata.get('simulations', []):
        if row['simulation_id'] not in allowed:
            continue
        try:
            groups, _ = _metadata_events(row)
            if len(groups) != 1: raise Blocked('REPEATED_APPLICATION_REQUIRES_EXACTLY_ONE_ORIGINAL_APPLICATION')
            parents = [c for c in row.get('compounds', []) if _name(c) == expected]
            if len(parents) != 1: raise Blocked('REPEATED_APPLICATION_EXACT_PARENT_REQUIRED')
            group = next(iter(groups.values()))
            prefix, _ = _event_prefix(group['Dose']['path'])
            ref = {'path': str(Path(source).relative_to(Path(project_root))), 'sha256': source_sha,
                   'simulation_id': row['simulation_id'], 'source_blob_sha256': row['source_blob_sha256']}
            policies[row['simulation_id']] = repeated_application.prepare_source_expansion(ref,
                _original_metadata(row), compound_name=parents[0], event_prefix=prefix,
                administration_times_min=contract['administration_times_min'], dose_mg=contract['dose_mg'],
                project_root=project_root)
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, ET.ParseError) as exc:
            failures.append({'simulation_id': row['simulation_id'], 'reason': str(exc)})
    return policies, failures



def isolated_source_options(metadata, excluded, contract, source_policy, source, source_sha, *, project_root=ROOT):
    allowed = {r['simulation_id'] for r in excluded if r.get('reason') == 'COADMINISTERED_OR_UNRESOLVED_SOURCE_DRUG_NOT_ALLOWED'}
    options, failures = [], []
    alias = source_policy.get('identity_aliases',{}).get(contract['compound'])
    expected = _name(alias['native_name']) if alias else contract['compound']
    for row in metadata.get('simulations',[]):
        if row['simulation_id'] not in allowed: continue
        try:
            names=[c for c in row.get('compounds',[]) if _name(c)==expected]
            if len(names)!=1: raise Blocked('ISOLATION_EXACT_TARGET_PARENT_REQUIRED')
            view=coadministration_isolation.target_metadata(row,names[0])
            candidates, rejected=select_source_simulation({'simulations':[view]},contract,source_policy=source_policy)
            if not candidates and any(r['reason']=='SOURCE_HAS_FEWER_EVENTS_THAN_FROZEN_TARGET_REGIMEN' for r in rejected):
                ref={'path':str(Path(source).relative_to(Path(project_root))),'sha256':source_sha,
                     'simulation_id':row['simulation_id'],'source_blob_sha256':row['source_blob_sha256']}
                saved=repeated_application._original_document(ref,Path(project_root))
                raw=source_pkml.describe_saved_simulation(saved)
                expansion=mixed_application.prepare_source_expansion(source,source_sha,row['simulation_id'],row['source_blob_sha256'],
                    raw,names[0],contract,project_root=project_root)
                extended={**source_policy,'application_expansion_policies':{row['simulation_id']:expansion}}
                candidates,rejected=select_source_simulation({'simulations':[view]},contract,source_policy=extended)
            if not candidates: raise Blocked('ISOLATION_TARGET_DESIGN_REQUIRES_ENGINEERING_REPAIR:' + ';'.join(r['reason'] for r in rejected))
            policy=coadministration_isolation.prepare_source_isolation(source,source_sha,row['simulation_id'],row['source_blob_sha256'],
                _original_metadata(row),names[0],contract,project_root=project_root)
            for candidate in candidates:
                candidate['coadministration_isolation']=policy
                options.append(candidate)
        except (ValueError,KeyError,TypeError,OSError,sqlite3.Error,ET.ParseError) as exc:
            failures.append({'simulation_id':row['simulation_id'],'reason':str(exc)})
    options.sort(key=lambda r:r['sort_key'])
    return options,failures


def prepare_candidate(scenario, candidate, *, project_root=ROOT, execution_revision=EXECUTION_REVISION,
                      license_receipt=DEFAULT_RECEIPT, verified_json_source=None):
    """Frozen plan only: read source metadata/licence; no native export or ODE."""
    root = Path(project_root).resolve(strict=True)
    plan = {'version': VERSION, 'adapter_route': 'complete_original_pksim5', 'status': 'blocked',
            'execution_revision': execution_revision, 'implementation_sha256': _execution_implementation_hashes(root, verified_json_source),
            'candidate_id': candidate.get('candidate_id'), 'scenario_id': scenario.get('scenario_id', scenario.get('case_id')),
            'scenario_sha256': _digest(scenario), 'candidate_sha256': _digest(candidate), 'reasons': [],
            'scientific_inapplicability_inferred': False, 'sealed_observations_accessed': False,
            'fitting_performed': False, 'replacement_blueprint_constructed': False}
    try:
        if execution_revision != EXECUTION_REVISION: raise Blocked('EXECUTION_REVISION_NOT_REGISTERED')
        contract = scenario_contract(scenario, project_root=root, candidate=candidate)
        plan['contract'] = contract
        if contract.get('matrix') != 'plasma':
            raise Blocked('COMPLETE_PKSIM5_EXACT_PLASMA_OUTPUT_REQUIRED')
        if contract.get('salt_conversion'):
            conversion = contract['salt_conversion']
            if sha(inside(conversion['source_file'], root)) != conversion['source_file_sha256']:
                raise Blocked('DOSE_CONVERSION_SOURCE_CHECKSUM_MISMATCH')
        if not contract.get('case_id') or not contract.get('compound'): raise Blocked('CASE_AND_COMPOUND_ID_REQUIRED')
        registered = candidate.get('compounds', candidate.get('registered_compounds', []))
        if contract['compound'] not in {_name(c) for c in registered}: raise Blocked('TARGET_COMPOUND_NOT_REGISTERED_IN_CANDIDATE')
        if 'provenance' in candidate and 'catalog_provenance' in candidate and candidate['provenance'] != candidate['catalog_provenance']:
            raise Blocked('CONFLICTING_CANDIDATE_PROVENANCE_FIELDS')
        prov = candidate.get('provenance', candidate.get('catalog_provenance', {}))
        original_source = inside(prov['source_path'], root)
        if sha(original_source) != prov['source_sha256']: raise Blocked('SOURCE_CHECKSUM_MISMATCH')
        plan.update(source_path=str(original_source.relative_to(root)), source_sha256=prov['source_sha256'])
        source, source_sha = original_source, prov['source_sha256']
        native_binding = None
        if verified_json_source is not None:
            if original_source.suffix != '.json': raise Blocked('JSON_BRIDGE_REQUIRES_ORIGINAL_JSON_SOURCE')
            if 'pregnan' in original_source.name.lower(): raise Blocked('JSON_PREGNANCY_BRIDGE_REQUIRES_SEPARATE_REGISTERED_IDENTITY')
            from .json_native_source_proof import verify_native_source
            native_binding = verify_native_source(prov, verified_json_source, project_root=root)
            source = inside(native_binding['native_source']['path'], root)
            source_sha = native_binding['native_source']['sha256']
            if source.suffix != '.pksim5' or sha(source) != source_sha:
                raise Blocked('JSON_NATIVE_PROJECT_IDENTITY_DIFFERS')
            plan.update(adapter_route='verified_json_native_pksim5', execution_source_path=str(source.relative_to(root)),
                        execution_source_sha256=source_sha, json_native_source_proof=native_binding)
        elif source.suffix != '.pksim5':
            raise Blocked('ORIGINAL_PKSIM5_REQUIRED_NO_JSON_COMPANION_SUBSTITUTION')
        if prov.get('license_status') == 'allowed':
            license_result = {'status': 'allowed_in_frozen_catalogue', 'source_sha256': prov['source_sha256'],
                              'catalogue_record_sha256': plan['candidate_sha256']}
        else:
            license_result = audit_candidate_license(candidate, project_root=root, receipt_path=license_receipt)
            if license_result['status'] != 'license_verified':
                plan['license_audit'] = license_result
                raise Blocked('CANDIDATE_LICENSE_NOT_RESOLVED:' + license_result.get('reason', 'unknown'))
        plan['license_audit'] = license_result
        permission = contract.get('prediction_contract', {}).get('source_reference_subject_prediction_allowed')
        if permission is False or not (permission is True or scenario.get('execution_policy', {}).get('allow_reference_subject_approximation') is True):
            raise Blocked('REFERENCE_SUBJECT_APPROXIMATION_REQUIRES_EXPLICIT_PROTOCOL_PERMISSION')
        cache_key = (str(source), source_sha, plan['implementation_sha256']['scripts/recommendation_execution_repairs/source_pkml.py'])
        if cache_key not in _METADATA_CACHE:
            _METADATA_CACHE[cache_key] = source_pkml.source_simulation_metadata(source, source_sha, project_root=root)
        metadata = _METADATA_CACHE[cache_key]
        if native_binding is not None:
            allowed = native_binding['allowed_simulations']
            ids = {r['id'] for r in allowed}
            if len(ids) != len(allowed) or not ids: raise Blocked('JSON_NATIVE_ALLOWED_SIMULATIONS_AMBIGUOUS')
            selected = [r for r in metadata['simulations'] if r['simulation_id'] in ids]
            if len(selected) != len(ids): raise Blocked('JSON_NATIVE_ALLOWED_SIMULATION_ABSENT')
            for row in selected:
                binding = next(r for r in allowed if r['id'] == row['simulation_id'])
                if binding['name'] != row['name'] or binding['source_blob_sha256'] != row['source_blob_sha256']:
                    raise Blocked('JSON_NATIVE_SIMULATION_BINDING_DIFFERS')
            metadata = {**metadata, 'simulations': selected}
        source_policy = pregnancy_source_policy(candidate, source, metadata, project_root=root)

        source_design = None
        if native_binding is not None:
            from . import json_source_design
            source_design = json_source_design.prepare_source_design_amendment(prov, contract, project_root=root)
            if source_design:
                metadata = {**metadata, 'simulations': [r for r in metadata['simulations']
                    if r['name'] == source_design['source_simulation']]}
                if len(metadata['simulations']) != 1:
                    raise Blocked('SOURCE_DESIGN_AMENDMENT_EXACT_NATIVE_SIMULATION_REQUIRED')
                source_policy = {**source_policy, 'identity_aliases': {**source_policy['identity_aliases'],
                    contract['compound']: source_design}}
                plan['source_design_amendment'] = source_design
        plan['source_identity_policy'] = source_policy
        selection_metadata, association_proofs, association_failures = enriched_selection_metadata(
            metadata, source, source_sha, project_root=root)
        plan['event_association_assessment_failures'] = association_failures
        physiology_policies={}
        for source_row in selection_metadata.get('simulations',[]):
            if not any(r['path'].startswith('Events|Urinary bladder emptying|') for r in source_row.get('event_parameters',[])):continue
            _,original=event_associations._load(source,source_sha,source_row['simulation_id'],source_row['source_blob_sha256'],root)
            original_metadata,_=event_associations._enrich_document(original)
            physiology=coadministration_isolation.preserved_urinary_events(original,original_metadata)
            if physiology:physiology_policies[source_row['simulation_id']]=physiology
        if physiology_policies:source_policy={**source_policy,'preserved_physiology_policies':physiology_policies}

        plan['source_identity_policy'] = source_policy
        dual_mode = native_binding is not None and contract['compound'] == 'omeprazole'
        split_mode = contract['compound'] == 'rosuvastatin'
        if split_mode:
            from . import split_absorption
            options, excluded = split_absorption.selection_options(selection_metadata, prov, contract,
                source_policy, source, source_sha, project_root=root)
        elif dual_mode:
            from . import json_dual_native
            options, excluded = json_dual_native.selection_options(selection_metadata, prov, contract,
                source_policy, source, source_sha, project_root=root)
        else:
            options, excluded = select_source_simulation(selection_metadata, contract, source_policy=source_policy)
        if not options and not dual_mode and not split_mode:
            policies, failures = source_expansion_policies(selection_metadata, excluded, contract, source_policy,
                                                           source, source_sha, project_root=root)
            plan['application_expansion_assessment_failures'] = failures
            if policies:
                # Existing adequate source simulations are always preferred;
                # expansion is a structural fallback, never an error/PK-driven choice.
                source_policy = {**source_policy, 'application_expansion_policies': policies}
                options, excluded = select_source_simulation(selection_metadata, contract, source_policy=source_policy)
        if not options and not dual_mode and not split_mode:
            options, isolation_failures = isolated_source_options(selection_metadata, excluded, contract, source_policy,
                source, source_sha, project_root=root)
            plan['coadministration_isolation_assessment_failures'] = isolation_failures
        plan['excluded_source_simulations'] = excluded
        if not options: raise Blocked('NO_COMPLETE_SOURCE_SIMULATION_MATCHES_FROZEN_CONTRACT')
        chosen = options[0]
        original_rows = [row for row in metadata['simulations'] if row['simulation_id'] == chosen['metadata']['simulation_id']]
        if len(original_rows) != 1: raise Blocked('EXACT_ORIGINAL_SELECTION_METADATA_REQUIRED')
        source_metadata = _original_metadata(original_rows[0])
        association = association_proofs.get(source_metadata['simulation_id'])
        expansion = chosen.get('application_expansion')
        isolation = chosen.get('coadministration_isolation')
        split = chosen.get('split_absorption_contract')
        if split:
            if expansion or isolation: raise Blocked('SPLIT_ABSORPTION_REJECTS_GENERIC_ISOLATION_OR_EXPANSION')
            plan.update(split_absorption_contract=split, internal_absorption_event_count=2)
        dual = chosen.get('dual_parent_contract')
        if dual:
            output_contract = json_dual_native.source_output_contract(source, source_sha,
                source_metadata['simulation_id'], chosen['metadata']['source_blob_sha256'], dual, project_root=root)
            plan.update(dual_parent_contract=dual, dual_parent_observer=output_contract,
                dual_parent_suppression=chosen['dual_parent_suppression'])
        else:
            output_contract = source_output_contract(source, source_sha, source_metadata['simulation_id'],
                                chosen['metadata']['source_blob_sha256'], chosen['native_compound_name'], project_root=root)
        suppression = None
        if not dual and not split and chosen['source_event_count'] > len(contract['administration_times_min']):
            suppression = source_suppression_policy(source, source_sha, source_metadata['simulation_id'],
                chosen['metadata']['source_blob_sha256'], _original_metadata(chosen['metadata']), contract, project_root=root)
        approximate = ['source_reference_subject_is_not_clinical_group_statistic',
                       'target_covariates_not_fitted_or_silently_substituted',
                       'original_source_drug_model_and_settings_preserved']
        approximate += contract.get('dose_normalization', {}).get('approximations', [])
        approximate += contract.get('sampling_grid_registration', {}).get('approximations', [])
        if split:
            approximate += ['original single-dose biphasic absorption preserved as two source-defined internal pulses; no clinical dose omitted']
        if dual:
            approximate += ['both original racemate parent channels retained at original source dose fractions; original total molar observer unchanged']
        if source_design:
            approximate += ['explicit source-bound execution design amendment; no clinical formulation or parameter-variant equivalence claimed']
        if isolation:
            approximate += ['source_coadministration_removed_only_by_registered_zero_dose_and_future_start; all_shared_mechanisms_retained',
                            'conditional_zero_invariance_and_registered_output_grid_domain_checks; not_every_internal_solver_step',
                            'single_drug_reference_adaptation_not_direct_reuse']
        if contract['route'] == 'Oral':
            approximate += ['source_formulation_not_verified_identical_to_clinical_product', 'target_food_schedule_not_reconstructed']
            if chosen['source_formulation_classes'] != [chosen['target_formulation_class']]:
                approximate.append('source_formulation_class_differs_or_is_unresolved; reference_only')
        if source_metadata.get('population_simulation_is_reference_model_only'):
            approximate.append('source_population_distribution_not_reconstructed; saved_reference_model_only')
        if source_policy['pregnancy_candidate']:
            approximate += ['pregnancy_candidate_retains_original_pregnant_model; nonpregnant_controls_are_not_substituted',
                            'pregnancy_reference_population_not_matched_to_target; not_direct_reuse']
        if chosen['source_release'] != chosen['target_release'] or chosen['target_release'] == 'unknown':
            approximate.append('source_and_target_release_behavior_differs_or_is_unknown; reference_only')
        preserved_meals = _preserved_meal_parameters(source_metadata)
        if chosen.get('preserved_physiology_policy'):
            preserved_meals += chosen['preserved_physiology_policy']['preserved_event_parameters']
        if _preserved_meal_parameters(source_metadata):
            approximate.append('original_standard_meal_events_retained_unchanged; target_meal_schedule_not_applied')
        if chosen.get('preserved_physiology_policy'):
            approximate.append('original urinary bladder emptying triggers and urine-zero resets retained unchanged')
        plan.update(status='ready', source_simulation=source_metadata['name'], source_simulation_id=source_metadata['simulation_id'],
                    source_metadata=source_metadata, source_metadata_sha256=_digest(source_metadata),
                    source_model_blob_sha256=chosen['metadata']['source_blob_sha256'],
                    source_output_contract=output_contract, output_path=output_contract['output_path'],
                    molecular_weight_g_per_mol=output_contract['molecular_weight_g_per_mol'],
                    source_species_evidence=chosen['source_species_evidence'],
                    parent_identity_alias=chosen['identity_alias'], source_pregnancy_evidence=chosen['pregnancy_evidence'],
                    preserved_non_drug_event_parameters=preserved_meals,
                    native_compound_name=chosen['native_compound_name'],
                    source_event_count=isolation['source_event_count'] if isolation else chosen['source_event_count'],
                    source_target_event_count=chosen['source_event_count'],
                    executable_event_count=(len(contract['administration_times_min']) if expansion else isolation['source_event_count'] if isolation else chosen['source_event_count']),
                    target_event_count=len(contract['administration_times_min']),
                    surplus_source_doses_to_zero=0 if split else max(0, chosen['source_event_count']-len(contract['administration_times_min'])),
                    application_expansion=expansion, source_event_association=association, coadministration_isolation=isolation,
                    surplus_event_suppression=suppression,
                    source_formulation_classes=chosen['source_formulation_classes'], target_formulation_class=chosen['target_formulation_class'],
                    release_design={'target': chosen['target_release'], 'source': chosen['source_release'],
                                    'seroquel_brand_rule_source': SEROQUEL_RELEASE_SOURCE if contract['compound']=='quetiapine' and re.search(r'\bseroquel\b', str(contract.get('formulation','')).lower()) else None},
                    food_design={'target': chosen['target_food_state'], 'source': chosen['source_food_state']},
                    selection_rule='exact human parent/route; unmodified single-drug source preferred; fully proved zero-exposure coadministration isolation only if no single-drug source qualifies; preserve pregnancy candidate; prefer adequate original applications; full-XML-proven single tablet expansion only when none qualify; release, formulation, food, surplus events, source dose distance, name/id',
                    source_individual=source_metadata.get('population_source'),
                    compatibility={'stratum': 'reference_subject_approximation', 'strict_target_match': False,
                                   'approximations': approximate, 'drug_identity_match': True, 'route_match': True, 'dose_match_planned': True},
                    administration_policy=('repeat the complete original tablet application under the frozen full-XML policy; preserve drug/physiology/Settings and audit executable identities before dose/start patches' if expansion else
                        'assign ordered original events to frozen target times; zero surplus doses and move proven time triggers beyond output end; no added events or model rebuild'),
                    output_grid_policy='union of fixed clinical times, target administration times, observation origin and uniform <=6 min reference grid (max10000 uniform points)')
        if split:
            plan['administration_policy'] = 'preserve both original internal absorption pulses at source-bound dose fractions and lag; one clinical administration'
        if native_binding is not None:
            plan['source_selection_sort_key'] = list(chosen['sort_key'])
            execution_source_reference(plan, candidate, project_root=root)
            plan['compatibility']['approximations'].append('exact_original_JSON_model_authenticated_in_native_project; original catalogue identity retained')
        if sha(source) != source_sha or sha(original_source) != prov['source_sha256']:
            raise Blocked('SOURCE_CHANGED_DURING_PREPARATION')
        if _execution_implementation_hashes(root, verified_json_source) != plan['implementation_sha256']:
            raise Blocked('IMPLEMENTATION_CHANGED_DURING_PREPARATION')
    except (ValueError, KeyError, TypeError, OSError, TimeoutError, sqlite3.Error, ET.ParseError) as exc:
        plan['status'] = 'blocked'; plan['reasons'].append(type(exc).__name__ + ':' + str(exc))
    if plan['status']=='ready' and inside(license_receipt,root,exists=False).is_file():
        license_path=inside(license_receipt,root)
        key=_digest(plan)
        _RECENT_PREPARATIONS[key]={'plan':deepcopy(plan),'root':str(root),
            'scenario':_digest(scenario),'candidate':_digest(candidate),
            'license_path':str(license_path),'license_sha256':sha(license_path)}
        while len(_RECENT_PREPARATIONS)>4:_RECENT_PREPARATIONS.pop(next(iter(_RECENT_PREPARATIONS)))
    return plan


def output_grid(contract):
    end = contract['absolute_sampling_times_min'][-1]
    count = min(9999, max(1, math.ceil(end / 6.)))
    points = {end*i/count for i in range(count+1)}
    points.update(contract['absolute_sampling_times_min'])
    points.update(t for t in contract['administration_times_min'] if t <= end)
    points.add(contract['observation_origin_min'])
    return sorted(points)


def build_patches(inventory, contract, source_metadata, *, native_compound_name=None, suppression_policy=None):
    """Bind every real native event to the preserved source and frozen regimen."""
    original_groups, assignments = _metadata_events(source_metadata)
    groups = {}
    for parameter in inventory.get('parameters', []):
        if not native_pkml.ADMINISTRATION.fullmatch(parameter.get('path', '')): continue
        prefix, field = _event_prefix(parameter['path']); key = _normalized_event(prefix)
        if field in groups.setdefault(key, {}): raise Blocked('DUPLICATE_NATIVE_ADMINISTRATION_PARAMETER')
        if not _finite(parameter.get('value')): raise Blocked('NONFINITE_OR_NEGATIVE_NATIVE_ADMINISTRATION')
        groups[key][field] = parameter
    if set(groups) != set(original_groups): raise Blocked('NATIVE_EVENT_SET_DIFFERS_FROM_SAVED_MODEL')
    for key, group in groups.items():
        if set(group) != set(original_groups[key]): raise Blocked('NATIVE_EVENT_FIELDS_DIFFER_FROM_SAVED_MODEL')
        if [_name(x) for x in assignments[key]['administered_compounds']] != [_name(native_compound_name) if native_compound_name else contract['compound']]:
            raise Blocked('COADMINISTERED_SOURCE_EVENT_NOT_ALLOWED')
        if assignments[key]['application_type'] != contract['route']: raise Blocked('NATIVE_SOURCE_ROUTE_CONFLICT')
        if group['Dose']['unit'] not in MASS_MG or group['Start time']['unit'] not in TIME_MIN:
            raise Blocked('EXPLICIT_NATIVE_BASE_DOSE_AND_TIME_UNITS_REQUIRED')
        if 'Infusion time' in group and group['Infusion time']['unit'] not in TIME_MIN:
            raise Blocked('EXPLICIT_NATIVE_INFUSION_UNIT_REQUIRED')
        for field, row in group.items():
            if row['id'] != original_groups[key][field]['id']: raise Blocked('NATIVE_EVENT_ID_DIFFERS_FROM_SAVED_MODEL')
    ordered = sorted(groups, key=lambda key: (groups[key]['Start time']['value'] * TIME_MIN[groups[key]['Start time']['unit']], key))
    target_times = contract['administration_times_min']
    if len(ordered) < len(target_times): raise Blocked('NATIVE_EVENT_COUNT_INSUFFICIENT')
    suppression_rows = {}
    if len(ordered) > len(target_times):
        policy = suppression_policy or {}
        if (policy.get('verified') is not True or policy.get('ordered_source_event_prefixes') != ordered
                or policy.get('target_event_count') != len(target_times)
                or policy.get('output_end_min') != output_grid(contract)[-1]):
            raise Blocked('SOURCE_PROVEN_SURPLUS_EVENT_SUPPRESSION_REQUIRED')
        rows = policy.get('events', [])
        suppression_rows = {row['event_prefix']: row for row in rows}
        if len(suppression_rows) != len(rows) or set(suppression_rows) != set(ordered[len(target_times):]):
            raise Blocked('SURPLUS_EVENT_SUPPRESSION_COVERAGE_DIFFERS')
        for key, row in suppression_rows.items():
            if (row.get('start_parameter_id') != groups[key]['Start time']['id']
                    or row.get('start_parameter_path') != _normalized_event(groups[key]['Start time']['path'])
                    or row.get('dose_mg') != 0.0 or row.get('barrier', {}).get('verified') is not True
                    or row['barrier'].get('all_application_events_accounted') is not True
                    or row['barrier'].get('application_path') != key
                    or not _finite(row.get('new_start_time_min'))
                    or row['new_start_time_min'] <= policy['output_end_min']):
                raise Blocked('SURPLUS_EVENT_SUPPRESSION_BINDING_OR_TIME_DIFFERS')
    patches, events = [], []
    for i, key in enumerate(ordered):
        group = groups[key]; active = i < len(target_times)
        desired = {'Dose': (contract['dose_mg'] if active else 0., 'mg')}
        if active:
            desired['Start time'] = (target_times[i], 'min')
            if contract['route'] == 'Intravenous':
                if 'Infusion time' not in group: raise Blocked('NATIVE_INFUSION_PARAMETER_MISSING')
                desired['Infusion time'] = (contract['infusion_minutes'], 'min')
        else:
            desired['Start time'] = (suppression_rows[key]['new_start_time_min'], 'min')
        for field, (value, unit) in desired.items():
            original = group[field]
            patches.append({'path': original['path'], 'expected_before_value': original['value'],
                            'expected_before_unit': original['unit'], 'value': value, 'unit': unit})
        events.append({'event_prefix': key, 'target_administration_index': i if active else None,
                       'disposition': 'frozen_target_administration' if active else 'surplus_zero_dose_event_moved_beyond_output_window',
                       'dose_mg': contract['dose_mg'] if active else 0.,
                       'start_time_min': target_times[i] if active else suppression_rows[key]['new_start_time_min'],
                       'suppression_proof': None if active else suppression_rows[key]['barrier'],
                       'original_source_parameters': group})
    return patches, events



def monitor_request(policy):
    return coadministration_isolation.monitor_request(policy)


def build_isolated_patches(inventory, contract, source_metadata, *, native_compound_name=None,
                           suppression_policy=None, isolation_policy=None):
    if not isolation_policy or native_compound_name != isolation_policy['target']:
        raise Blocked('EXACT_COADMINISTRATION_ISOLATION_POLICY_REQUIRED')
    return coadministration_isolation.build_isolated_patches(inventory, contract, source_metadata, isolation_policy,
        target_patch_builder=build_patches, suppression_policy=suppression_policy)


def isolation_readback(native, policy):
    if policy is None: return None
    native_pkml.validate_zero_monitors(native, {'output_times_min':native['curve']['time_min'], **monitor_request(policy)})
    native_pkml.validate_positive_domains(native, {'output_times_min':native['curve']['time_min'], **monitor_request(policy)})
    suppression = suppression_readback(native, policy)
    proof=policy['zero_manifold_proof']
    discrete=coadministration_isolation.verify_discrete_parameter_readback(native,proof)
    result = {'accepted':True, 'source_policy_sha256':_digest(policy), 'administration_readback':suppression,
            'zero_trajectory_count':len(proof['zero_monitor_paths']),
            'positive_dynamic_domain_count':len(proof['positive_domain_monitor_dimensions']),
            'static_parameter_readback_count':len(proof['native_parameter_requirements']),
            'registered_output_grid_only':True, 'all_shared_mechanisms_and_initial_physiology_retained':True,
            'limits':policy['limits'], 'scientific_validity_established':False}
    if discrete:result['exact_static_discrete_readback']=discrete
    return result

def suppression_readback(native, policy):
    """Independently confirm every surplus trigger is beyond the solved window."""
    rows = []
    changes = {_normalized_event(row['path']): row for row in native.get('parameter_changes', [])}
    for event in (policy or {}).get('events', []):
        start = changes.get(event['start_parameter_path'], {})
        dose = changes.get(event['event_prefix'] + '|ProtocolSchemaItem|Dose', {})
        after = start.get('after', {})
        dose_after = dose.get('after', {})
        if (start.get('id') != event['start_parameter_id']
                or after.get('unit') not in TIME_MIN or not _finite(after.get('value'))
                or dose_after.get('unit') not in MASS_MG or dose_after.get('value') != 0.0):
            raise Blocked('SURPLUS_NATIVE_PARAMETER_READBACK_MISSING_OR_DIFFERS')
        actual_min = after['value'] * TIME_MIN[after['unit']]
        if actual_min != event['new_start_time_min'] or actual_min <= policy['output_end_min']:
            raise Blocked('SURPLUS_NATIVE_EVENT_TIME_NOT_BEYOND_OUTPUT')
        rows.append({'event_prefix': event['event_prefix'], 'dose_native_value': 0.0,
                     'start_time_min': actual_min, 'output_end_min': policy['output_end_min'],
                     'trigger_outside_frozen_output_window': True})
    return {'accepted': True, 'native_readback': rows, 'scientific_validity_established': False}


def expansion_readback(native, policy, executable_metadata, verification):
    """Verify registered doses/starts against the actual expanded native IDs."""
    if policy is None:
        return None
    if policy.get('version')==mixed_application.VERSION:
        if verification.get('accepted') is not True or executable_metadata.get('metadata_scope')!='actual_expanded_executable_model_only':
            raise Blocked('MIXED_EXPANSION_NATIVE_METADATA_PROOF_REQUIRED')
        view=coadministration_isolation.target_metadata(executable_metadata,policy['target'])
        view.update(original_source_event_count=1,executable_event_count=policy['executable_target_event_count'])
        keys={_normalized_event(r['path']) for r in view['event_parameters']}
        selected={**native,'parameter_changes':[r for r in native['parameter_changes'] if _normalized_event(r['path']) in keys]}
        audit=expansion_readback(selected,{'administration_times_min':policy['contract']['administration_times_min'],
            'dose_mg':policy['contract']['dose_mg'],'executable_event_count':policy['executable_target_event_count']},view,verification)
        audit['source_event_count']=policy['source_target_event_count']
        audit['original_total_event_count']=policy['source_total_event_count']
        return audit
    if (verification.get('accepted') is not True or
            executable_metadata.get('metadata_scope') != 'actual_expanded_executable_model_only' or
            executable_metadata.get('original_source_event_count') != 1 or
            executable_metadata.get('executable_event_count') != policy['executable_event_count']):
        raise Blocked('EXPANSION_NATIVE_METADATA_PROOF_REQUIRED')
    groups, _ = _metadata_events(executable_metadata)
    ordered = sorted(groups, key=lambda key: (float(groups[key]['Start time']['value']), key))
    changes = {_normalized_event(row['path']): row for row in native.get('parameter_changes', [])}
    if len(changes) != len(native.get('parameter_changes', [])) or len(changes) != 2 * len(ordered):
        raise Blocked('EXPANSION_NATIVE_READBACK_COUNT_DIFFERS')
    rows = []
    for key, time_min in zip(ordered, policy['administration_times_min']):
        for field, value, units in [('Dose', policy['dose_mg'], MASS_MG), ('Start time', time_min, TIME_MIN)]:
            change = changes.get(key + '|ProtocolSchemaItem|' + field, {})
            actual = change.get('after', {})
            if (change.get('id') != groups[key][field]['id'] or actual.get('unit') not in units or
                    not _finite(actual.get('value')) or
                    not math.isclose(actual['value'] * units[actual['unit']], value, rel_tol=1e-12, abs_tol=1e-12)):
                raise Blocked('EXPANSION_NATIVE_DOSE_OR_START_READBACK_DIFFERS')
            rows.append({'event_prefix': key, 'field': field, 'native_id': change['id'],
                         'value': value, 'unit': 'mg' if field == 'Dose' else 'min', 'verified': True})
    if len(ordered) != policy['executable_event_count']:
        raise Blocked('EXPANSION_NATIVE_EVENT_COUNT_DIFFERS')
    return {'accepted': True, 'source_event_count': 1, 'executable_event_count': len(ordered),
            'full_source_and_expansion_audit': verification, 'native_readback': rows,
            'source_metadata_preserved': True, 'scientific_validity_established': False}


def _native_output(inventory, plan):
    if plan.get('dual_parent_contract'):
        from .json_dual_native import native_output
        return native_output(inventory, plan)
    drug = plan['native_compound_name']
    candidates = [f'Organism|PeripheralVenousBlood|{drug}|Plasma (Peripheral Venous Blood)',
                  f'Organism|VenousBlood|Plasma|{drug}|Concentration in container']
    outputs = inventory.get('observer_paths', [])
    matched = [p for p in candidates if outputs.count(p) == 1]
    if not matched: raise Blocked('EXACT_PARENT_TOTAL_PLASMA_OBSERVER_UNAVAILABLE')
    parameters = [p for p in inventory.get('parameters', []) if p.get('path') == drug+'|Molecular weight']
    if len(parameters) != 1 or parameters[0].get('unit') not in MW_G_MOL or not _finite(parameters[0].get('value'), positive=True):
        raise Blocked('EXACT_PARENT_MOLECULAR_WEIGHT_REQUIRED')
    return matched[0], parameters[0]['value'] * MW_G_MOL[parameters[0]['unit']]


def normalized_molar_curve(curve):
    """Normalize only literal micro-symbol aliases, without numerical scaling."""
    unit = curve.get('unit'); dimension = curve.get('dimension')
    if unit not in {'umol/l', 'µmol/l', 'μmol/l'}:
        raise Blocked('EXACT_NATIVE_MICROMOLAR_CONCENTRATION_UNIT_REQUIRED')
    if dimension not in (None, 'Concentration (molar)') or (unit != 'umol/l' and dimension != 'Concentration (molar)'):
        raise Blocked('EXACT_NATIVE_MOLAR_CONCENTRATION_DIMENSION_REQUIRED')
    normalized = deepcopy(curve)
    if unit != 'umol/l':
        normalized.update(unit='umol/l', original_native_unit=unit,
                          unit_alias_normalization='literal micro-symbol alias only; numerical scale=1')
    return normalized


def execute_candidate(scenario, candidate, output_dir, *, project_root=ROOT, timeout_seconds=300,
                      cpu_threads=1, stop_file=None, execution_revision=EXECUTION_REVISION,
                      license_receipt=DEFAULT_RECEIPT, frozen_plan=None, verified_json_source=None):
    """One preserved attempt; no fallback choice based on execution or error."""
    root = Path(project_root).resolve(strict=True); output = inside(output_dir, root, exists=False)
    if output.exists(): raise FileExistsError('Never overwrite complete candidate attempts')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800: raise ValueError('timeout_seconds must be 1..1800')
    if cpu_threads not in (1, 2): raise ValueError('One or two CPU threads required')
    if shutil.disk_usage(root).free < 2*1024**3: raise OSError('At least 2 GiB free space required')
    cached=_RECENT_PREPARATIONS.get(_digest(frozen_plan)) if frozen_plan is not None else None
    if (cached and cached['root']==str(root) and cached['scenario']==_digest(scenario) and
        cached['candidate']==_digest(candidate) and cached['plan']['execution_revision']==execution_revision and
        cached['plan']['implementation_sha256']==_execution_implementation_hashes(root, verified_json_source) and
        sha(inside(cached['plan']['source_path'],root))==cached['plan']['source_sha256'] and
        str(inside(license_receipt,root))==cached['license_path'] and
        sha(inside(license_receipt,root))==cached['license_sha256']):
        plan=deepcopy(cached['plan'])
    else:
        plan = prepare_candidate(scenario, candidate, project_root=root, execution_revision=execution_revision, license_receipt=license_receipt, verified_json_source=verified_json_source)
    if frozen_plan is not None and _digest(plan) != _digest(frozen_plan):
        raise ValueError('FROZEN_COMPLETE_CANDIDATE_PLAN_DIFFERS')
    output.mkdir(parents=True); save(output/'execution_plan.json', plan)
    result = {'version': VERSION, 'adapter_route': plan['adapter_route'], 'scenario_id': plan['scenario_id'], 'candidate_id': plan['candidate_id'],
              'execution_revision': plan['execution_revision'], 'implementation_sha256': plan['implementation_sha256'],
              'status': plan['status'], 'reasons': list(plan['reasons']), 'native_executed': False, 'prediction_locked': False,
              'sealed_observations_accessed': False, 'fitting_performed': False, 'scientific_inapplicability_inferred': False,
              'execution_plan_sha256': sha(output/'execution_plan.json')}
    receipt = {'execution_revision': plan['execution_revision'], 'implementation_sha256': plan['implementation_sha256'],
               'scenario_sha256': plan['scenario_sha256'], 'candidate_sha256': plan['candidate_sha256'],
               'execution_plan_sha256': result['execution_plan_sha256'], 'stages': {}, 'observations_read': False}
    def stop_check():
        if stop_file and inside(stop_file, root, exists=False).exists(): raise Blocked('PROJECT_STOP_REQUESTED')
    try:
        if plan['status'] == 'ready':
            stop_check()
            native_source = execution_source_reference(plan, candidate, project_root=root)
            if plan.get('json_native_source_proof'):
                save(output/'json_native_source_proof.json', plan['json_native_source_proof'])
                receipt['stages']['json_native_source_proof'] = reference(output/'json_native_source_proof.json', root)
            split = plan.get('split_absorption_contract')
            if split:
                from . import split_absorption
                split_absorption.revalidate_contract(split, plan['contract'], plan['source_metadata'], project_root=root)
                save(output/'split_absorption_contract.json', split)
                receipt['stages']['split_absorption_contract'] = reference(output/'split_absorption_contract.json', root)
            dual = plan.get('dual_parent_contract')
            if dual:
                from . import json_dual_native, json_racemate_contract
                json_dual_native.verify_plan(plan, candidate, project_root=root)
                for field in ('dual_parent_contract', 'dual_parent_observer'):
                    save(output/(field+'.json'), plan[field]); receipt['stages'][field] = reference(output/(field+'.json'), root)
            if plan.get('source_design_amendment'):
                from .json_source_design import prepare_source_design_amendment
                current = prepare_source_design_amendment(candidate.get('provenance', candidate.get('catalog_provenance', {})),
                    plan['contract'], project_root=root)
                if current != plan['source_design_amendment']: raise Blocked('SOURCE_DESIGN_AMENDMENT_CHANGED')
                save(output/'source_design_amendment.json', current)
                receipt['stages']['source_design_amendment'] = reference(output/'source_design_amendment.json', root)
            exported = source_pkml.export_saved_model(native_source['path'], native_source['sha256'], plan['source_simulation_id'], output/'export',
                         simulation_name=plan['source_simulation'], project_root=root, timeout_seconds=min(timeout_seconds, 180),
                         cpu_threads=cpu_threads, stop_file=stop_file)
            receipt['stages']['export'] = reference(output/'export/export_manifest.json', root)
            if exported.get('status') != 'exported_pending_native_load_and_clinical_validation':
                raise Blocked('ORIGINAL_SAVED_MODEL_EXPORT_NOT_VERIFIED:'+';'.join(exported.get('errors', [])))
            if _digest(exported.get('source_metadata')) != plan['source_metadata_sha256']:
                raise Blocked('EXPORTED_SIMULATION_METADATA_DIFFERS_FROM_PLAN')
            if exported.get('source_blob_sha256') != plan['source_model_blob_sha256']:
                raise Blocked('EXPORTED_ORIGINAL_MODEL_BLOB_DIFFERS_FROM_PLAN')
            pkml = inside(exported['pkml']['path'], root)
            if sha(pkml) != exported['pkml']['sha256']: raise Blocked('EXPORTED_PKML_CHECKSUM_DIFFERS')
            execution_metadata = plan['source_metadata']
            association = plan.get('source_event_association')
            if association:
                current = event_associations.enrich_source_metadata(native_source['path'], native_source['sha256'],
                    plan['source_simulation_id'], plan['source_model_blob_sha256'],
                    metadata=plan['source_metadata'], project_root=root)
                if _digest(current) != _digest(association): raise Blocked('FROZEN_EVENT_ASSOCIATION_PROOF_DIFFERS')
                save(output/'source_event_association.json', current)
                receipt['stages']['source_event_association'] = reference(output/'source_event_association.json', root)
                execution_metadata = current['enriched_metadata']
            isolation = plan.get('coadministration_isolation')
            if isolation:
                verified_isolation = coadministration_isolation.verify_source_isolation(isolation, execution_metadata, plan['contract'], project_root=root)
                save(output/'coadministration_isolation.json', verified_isolation)
                receipt['stages']['coadministration_isolation'] = reference(output/'coadministration_isolation.json', root)
            expansion = plan.get('application_expansion'); expansion_verification = None
            if expansion:
                stop_check()
                if expansion.get('version')==mixed_application.VERSION:
                    expanded=mixed_application.expand_pkml(pkml,expansion,output/'repeated_application',project_root=root)
                    execution_metadata,expansion_verification=mixed_application.executable_metadata(
                        output/'repeated_application/receipt.json',plan['source_metadata'],expected_policy=expansion,project_root=root)
                    if not isolation:raise Blocked('MIXED_EXPANSION_REQUIRES_ZERO_EXPOSURE_ISOLATION')
                    isolation,execution_metadata=coadministration_isolation.prepare_expanded_isolation(isolation,
                        output/'repeated_application/receipt.json',execution_metadata,plan['contract'],project_root=root)
                    save(output/'expanded_coadministration_isolation.json',isolation)
                    receipt['stages']['expanded_coadministration_isolation']=reference(output/'expanded_coadministration_isolation.json',root)
                else:
                    expanded = repeated_application.expand_applications(pkml, output/'repeated_application',
                    source_reference=expansion['source_reference'], compound_name=expansion['compound_name'],
                    event_prefix=expansion['event_prefix'], administration_times_min=expansion['administration_times_min'],
                    dose_mg=expansion['dose_mg'], project_root=root)
                receipt['stages']['application_expansion'] = reference(output/'repeated_application/receipt.json', root)
                if expansion.get('version')!=mixed_application.VERSION:
                    execution_metadata, expansion_verification = repeated_application.executable_metadata(
                        output/'repeated_application/receipt.json', plan['source_metadata'], expected_policy=expansion, project_root=root)
                pkml = inside(expanded['pkml']['path'], root)
                if sha(pkml) != expanded['pkml']['sha256']: raise Blocked('EXPANDED_PKML_CHECKSUM_DIFFERS')
            metadata_record = {'scope': ('expanded_executable_model' if expansion else
                                        'original_model_with_source_proven_event_identity' if association else 'original_source_model'),
                               'original_source_metadata_sha256': plan['source_metadata_sha256'],
                               'executable_metadata_sha256': _digest(execution_metadata), 'pkml': reference(pkml, root),
                               'metadata': execution_metadata}
            save(output/'executable_metadata.json', metadata_record)
            receipt['executable_metadata'] = reference(output/'executable_metadata.json', root)
            stop_check()
            inspection = native_pkml.execute_complete_pkml(pkml, output/'inspect', project_root=root, timeout_seconds=min(timeout_seconds, 120),
                                                          cpu_threads=cpu_threads, stop_file=stop_file)
            receipt['stages']['inspect'] = reference(output/'inspect/result.json', root)
            if inspection.get('status') != 'inspected': raise Blocked('COMPLETE_NATIVE_INSPECTION_FAILED')
            inventory = inspection['native']['inventory']
            if split:
                if isolation or expansion or plan.get('surplus_event_suppression'):
                    raise Blocked('SPLIT_ABSORPTION_FORBIDS_GENERIC_EVENT_MODIFICATIONS')
                patches, events = split_absorption.build_split_patches(inventory, plan['contract'], execution_metadata, split, project_root=root)
            elif dual:
                if expansion: raise Blocked('DUAL_PARENT_EXPANSION_NOT_REGISTERED')
                patches, events = json_dual_native.build_patches(inventory, plan, execution_metadata, project_root=root)
            else:
                patch_builder = build_isolated_patches if isolation else build_patches
                patches, events = patch_builder(inventory, plan['contract'], execution_metadata,
                    native_compound_name=plan['native_compound_name'], suppression_policy=plan.get('surplus_event_suppression'),
                    **({'isolation_policy':isolation} if isolation else {}))
            isolation_monitor_request = monitor_request(isolation)
            if expansion:
                for event in events:
                    event['inspected_executable_parameters'] = event.pop('original_source_parameters')
                    event['parameter_origin'] = 'full-XML-audited original-application expansion; not an original source event'
            output_path, mw = _native_output(inventory, plan)
            if output_path != plan['output_path'] or not math.isclose(mw, plan['molecular_weight_g_per_mol'], rel_tol=1e-12, abs_tol=1e-12):
                raise Blocked('NATIVE_PARENT_OUTPUT_OR_MOLECULAR_WEIGHT_DIFFERS_FROM_FROZEN_ORIGINAL_MODEL')
            times = output_grid(plan['contract'])
            execution_details = {'patches': patches, 'administrations': events, 'output_path': output_path,
                                 'surplus_event_suppression': plan.get('surplus_event_suppression'),
                                 'coadministration_isolation': isolation, 'isolation_monitor_request': isolation_monitor_request,
                                 'application_expansion': receipt['stages'].get('application_expansion'),
                                 'executable_metadata': receipt['executable_metadata'],
                                 'executable_metadata_sha256': _digest(execution_metadata),
                                 'molecular_weight_g_per_mol': mw, 'output_times_min': times,
                                 'pkml': reference(pkml, root), 'source_metadata_sha256': plan['source_metadata_sha256']}
            if split:
                execution_details['split_absorption_contract'] = receipt['stages']['split_absorption_contract']
            if dual:
                execution_details.update(dual_parent_contract=receipt['stages']['dual_parent_contract'],
                    dual_parent_observer=receipt['stages']['dual_parent_observer'], dual_parent_suppression=plan['dual_parent_suppression'])
            if plan.get('source_design_amendment'):
                execution_details['source_design_amendment'] = receipt['stages']['source_design_amendment']
            save(output/'registered_native_request.json', execution_details)
            receipt['registered_native_request'] = reference(output/'registered_native_request.json', root)
            stop_check()
            native = native_pkml.execute_complete_pkml(pkml, output/'native', action='run', patches=patches, output_path=output_path,
                         output_times_min=times, timeout_seconds=timeout_seconds, cpu_threads=cpu_threads, project_root=root, stop_file=stop_file, **isolation_monitor_request)
            receipt['stages']['native'] = reference(output/'native/result.json', root)
            result['native_executed'] = native.get('native', {}).get('solver_calls') == 1 if native.get('native') else False
            if native.get('status') != 'prediction_locked': raise Blocked('COMPLETE_NATIVE_PREDICTION_NOT_VERIFIED:'+';'.join(native.get('reasons', [])))
            suppression_audit = suppression_readback(native['native'], plan.get('surplus_event_suppression'))
            isolation_audit = isolation_readback(native['native'], isolation)
            expansion_audit = expansion_readback(native['native'], expansion, execution_metadata, expansion_verification)
            curve = native['native']['curve']
            prediction = prediction_from_curve(normalized_molar_curve(curve), plan)
            prediction.update(version=VERSION, adapter_route=plan['adapter_route'], original_model_export=receipt['stages']['export'],
                              registered_native_request=receipt['registered_native_request'],
                              application_expansion=receipt['stages'].get('application_expansion'))
            save(output/'prediction.json', prediction)
            readback = {'accepted': True, 'adapter_route': plan['adapter_route'], 'source_model_export_verified': True,
                        'native_model_preservation': native['model_preservation_audit'],
                        'native_administration_changes': native['native']['parameter_changes'], 'administrations': events,
                        'surplus_event_suppression': suppression_audit,
                        'coadministration_isolation': isolation_audit,
                        'application_expansion': expansion_audit,
                        'source_event_association': receipt['stages'].get('source_event_association'),
                        'executable_metadata': receipt['executable_metadata'],
                        'output_grid_verified': native['native'].get('output_grid_verified'), 'scientific_validity_established': False}
            if split:
                readback['split_absorption_contract'] = receipt['stages']['split_absorption_contract']
                readback['split_absorption'] = split_absorption.verify_native_readback(native['native'], split)
            if dual:
                readback['dual_parent_contract'] = receipt['stages']['dual_parent_contract']
                readback['dual_parent_observer'] = receipt['stages']['dual_parent_observer']
                readback['dual_parent_suppression'] = {parent: suppression_readback(native['native'], policy)
                    for parent, policy in plan['dual_parent_suppression'].items()}
            if plan.get('source_design_amendment'):
                readback['source_design_amendment'] = receipt['stages']['source_design_amendment']
            save(output/'readback_result.json', readback)
            execution_source_reference(plan, candidate, project_root=root)
            if _execution_implementation_hashes(root, verified_json_source) != plan['implementation_sha256']: raise Blocked('EXECUTION_IMPLEMENTATION_CHANGED')
            receipt.update(status='prediction_locked', original_project_unchanged=True)
            save(output/'execution_receipt.json', receipt)
            # Include all returned native evidence, not only a success flag.
            files = [p for p in sorted(output.rglob('*')) if p.is_file()]
            lock = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'execution_revision': plan['execution_revision'], 'implementation_sha256': plan['implementation_sha256'],
                    'scenario_id': plan['scenario_id'], 'candidate_id': plan['candidate_id'],
                    'scenario_sha256': plan['scenario_sha256'], 'candidate_sha256': plan['candidate_sha256'],
                    'source_sha256': plan['source_sha256'], 'files': {str(p.relative_to(root)): sha(p) for p in files},
                    'sealed_observations_accessed': False, 'execution_verified': True, 'prediction_accuracy_evaluated': False,
                    'compatibility': plan['compatibility'], 'adapter_route': plan['adapter_route']}
            save(output/'prediction_lock.json', lock)
            result.update(status='prediction_locked', prediction_locked=True, compatibility=plan['compatibility'], strict_target_match=False,
                          approximation_reason=plan['compatibility']['approximations'], molecular_weight_g_per_mol=mw,
                          source={'path': plan['source_path'], 'sha256': plan['source_sha256'], 'simulation': plan['source_simulation']},
                          prediction_file=str((output/'prediction.json').relative_to(root)), prediction_sha256=sha(output/'prediction.json'),
                          prediction_lock=str((output/'prediction_lock.json').relative_to(root)), prediction_lock_sha256=sha(output/'prediction_lock.json'),
                          execution_manifest=str((output/'execution_receipt.json').relative_to(root)), curve=curve, predicted_endpoints=prediction['endpoints'])
    except (ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
        result.update(status='blocked' if isinstance(exc, Blocked) else 'failed', reasons=[type(exc).__name__+':'+str(exc)])
    if not (output/'execution_receipt.json').exists():
        receipt.update(status=result['status'], reasons=result['reasons']); save(output/'execution_receipt.json', receipt)
    save(output/'result.json', result)
    return result
