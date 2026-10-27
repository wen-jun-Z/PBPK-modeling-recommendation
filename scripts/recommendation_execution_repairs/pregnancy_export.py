"""Export every saved simulation of the three exact pregnancy catalog projects.

This performs no fitting, outcome lookup, simulation selection by performance or
ODE execution. Every source simulation stays in the task inventory, including
nonpregnant controls and DDI models. A population project's preserved reference
model is never reported as execution of its original population distribution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from recommendation_execution_repairs import source_pkml
from recommendation_execution_repairs.pregnancy_source_assets import CANDIDATES, CATALOG, inside, sha_file, write_json

DEFAULT_OUTPUT = ROOT / 'artifacts/pbpk-model-assets/execution_repairs_20261001/pregnancy_exports'
PREGNANCY_MARKERS = {'Fetus', 'PlacentaMaternal', 'PlacentaFetal', 'AmnioticFluid'}


def publish_native_models(batch_manifest, output):
    """Publish only verified full models/proofs; do not publish temporary projects.

    Every saved simulation is retained irrespective of treatment or results. This
    is a source-model registry, never a claim of cohort execution or applicability.
    """
    batch_path = inside(batch_manifest)
    batch = json.loads(batch_path.read_text())
    if batch.get('summary') != {'catalog_simulations': 16, 'attempted': 16, 'passed': 16, 'failed': 0}:
        raise ValueError('All 16 original simulations must have passed before publication')
    output = inside(output)
    if output.exists():
        raise FileExistsError('Publication never overwrites an earlier registry')
    def checked(ref):
        path = inside(ROOT / ref['path'])
        if sha_file(path) != ref['sha256']:
            raise ValueError('Publication reference changed: ' + ref['path'])
        return path
    extracted = json.loads(checked(batch['input_inventory']).read_text())
    by_id = {(r['candidate_id'], r['simulation']['simulation_id']): r for r in extracted['records']}
    keys = {(r['candidate_id'], r['simulation']['simulation_id']) for r in batch['records']}
    if len(keys) != 16 or keys != set(by_id):
        raise ValueError('Native publication inventory differs from the full source inventory')
    catalog = {r['candidate_id']: r for r in map(json.loads, CATALOG.open())}
    staged = []
    for row in batch['records']:
        key = row['candidate_id'], row['simulation']['simulation_id']
        old = by_id[key]
        source_ref = old['extraction']['source']; checked(source_ref)
        checked(old['original_population_inputs'])
        provenance = catalog[row['candidate_id']]['provenance']
        if (source_ref['path'] != provenance['source_path'] or source_ref['sha256'] != provenance['source_sha256']
                or row['repository_commit'] != provenance['repository_commit']):
            raise ValueError('Exact frozen catalog candidate binding differs')
        if (row['status'] != 'native_serialization_passed' or not row['model_semantics']['accepted']
                or not row['settings_semantics']['accepted'] or not row['native_audit']['same_model_object']
                or not row['native_audit']['same_settings_object'] or row['native_audit']['solver_calls'] != 0):
            raise ValueError('Native model/settings preservation proof is incomplete')
        staged.append((row, old, checked(row['pkml'])))
    output.mkdir(parents=True)
    registry = []
    for row, old, native in staged:
        entry = output / row['compound'] / hashlib.sha256(row['simulation']['simulation_id'].encode()).hexdigest()[:16]
        entry.mkdir(parents=True)
        target = entry / 'model.pkml'; shutil.copyfile(native, target)
        if sha_file(target) != row['pkml']['sha256']:
            raise ValueError('Native model copy checksum differs')
        proof = {'schema': 'complete-pregnancy-source-preservation-proof-v1',
                 'candidate_id': row['candidate_id'], 'simulation': row['simulation'],
                 'repository': old['repository'], 'repository_commit': row['repository_commit'],
                 'source': old['extraction']['source'], 'source_blob_sha256': old['extraction']['blob_sha256'],
                 'native_model': {'path': str(target.relative_to(ROOT)), 'sha256': sha_file(target)},
                 'model_semantics': row['model_semantics'], 'settings_semantics': row['settings_semantics'],
                 'native_serializer_audit': row['native_audit'],
                 'structure': row['structure'], 'original_population_inputs': old['original_population_inputs'],
                 'batch_manifest': {'path': str(batch_path.relative_to(ROOT)), 'sha256': sha_file(batch_path)},
                 'publication_implementation': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha_file(Path(__file__))},
                 'population_covariates_applied': False, 'scientific_applicability_inferred': False,
                 'native_solver_validation_claimed': False, 'sealed_observations_accessed': False}
        proof_path = entry / 'proof.json'; write_json(proof_path, proof)
        registry.append({'candidate_id': row['candidate_id'], 'compound': row['compound'], 'simulation': row['simulation'],
                         'source': proof['source'], 'repository_commit': row['repository_commit'],
                         'native_model': proof['native_model'], 'proof': {'path': str(proof_path.relative_to(ROOT)), 'sha256': sha_file(proof_path)},
                         'pregnancy_markers': row['pregnancy_markers_preserved'], 'source_model_preservation_verified': True,
                         'population_execution_claimed': False, 'scientific_applicability_inferred': False})
    result = {'schema': 'pregnancy-complete-native-model-registry-v1',
              'catalog': {'path': str(CATALOG.relative_to(ROOT)), 'sha256': sha_file(CATALOG)},
              'source_batch': {'path': str(batch_path.relative_to(ROOT)), 'sha256': sha_file(batch_path)},
              'records': registry, 'counts': {'candidates': 3, 'simulations': 16, 'pregnancy_simulations': sum(bool(r['pregnancy_markers']) for r in registry)},
              'frozen_experiment_modified': False, 'original_sources_modified': False,
              'selection_basis': 'All saved simulations, including original controls and DDI; never outcome-based.',
              'limitations': ['Native source preservation alone does not establish clinical applicability or predictive accuracy.',
                             'Population distributions/cache remain retained in the original inputs; no population sampling is claimed.']}
    write_json(output / 'registry.json', result)
    return result


def read_document(source, simulation):
    """Use only the exact saved simulation content, never observation/results tables."""
    with sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT c.Data FROM SIMULATIONS s JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId '
                          'JOIN CONTENTS c ON c.Id=b.ContentId WHERE s.SimulationId=? AND b.ContentId=?',
                          (simulation['simulation_id'], simulation['content_id'])).fetchall()
    if len(rows) != 1:
        raise ValueError('Exactly one immutable simulation blob is required')
    root = ET.fromstring(source_pkml.decompress_model_blob(rows[0][0]))
    if root.get('id') != simulation['simulation_id'] or root.get('name') != simulation['name']:
        raise ValueError('Saved simulation identity differs')
    return root


def structural_inventory(document):
    model = document.find('Model')
    if model is None:
        raise ValueError('Complete saved Model required')
    names = {node.get('name') for node in model.iter('Container')}
    marker_names = sorted(PREGNANCY_MARKERS & names)
    organs = []
    for node in model.iter('Container'):
        if node.get('name') == 'Organism':
            children = node.find('Children')
            if children is not None:
                organs.extend(child.get('name') for child in children if child.tag == 'Container')
    populations = []
    for node in document.findall('./UsedBuildingBlocks/UsedBuildingBlock/RandomPopulation'):
        settings = node.find('Settings')
        populations.append({'id': node.get('id'), 'name': node.get('name'), 'seed': node.get('seed'),
                            'declared_number_of_individuals': settings.get('numberOfIndividuals') if settings is not None else None,
                            'original_population_xml_sha256': source_pkml.element_sha(node)})
    return {'saved_simulation_type': document.tag, 'saved_xml_version': document.get('xmlVersion'),
            'pregnancy_specific_structure_present': bool(marker_names),
            'complete_pregnancy_marker_set_present': PREGNANCY_MARKERS <= names,
            'pregnancy_markers': marker_names, 'organism_direct_container_names': organs,
            'parameter_element_count': sum(node.tag in {'Para', 'DistPara', 'Parameter', 'DistributedParameter'} for node in model.iter()),
            'model_xml_sha256': source_pkml.element_sha(model), 'populations': populations,
            'population_covariates_applied_to_native_instance': False,
            'scientific_applicability_inferred': False}


def preserve_population_inputs(document, output):
    """Retain source population definitions/cache separately from the native instance.

    Only source physiology, distributions and simulation bindings are copied;
    UsedObservedData, results, analyses and reaction diagrams are not included.
    """
    selected = ['Root', 'Properties', 'UsedBuildingBlocks', 'AgingData', 'ParameterValuesCache']
    wrapper = ET.Element('OriginalSimulationPopulationInputs', {
        'sourceSimulationId': document.get('id'), 'sourceSimulationName': document.get('name'),
        'sourceSimulationType': document.tag, 'notAppliedToNativeReferenceInstance': 'true'})
    fingerprints = {}
    for tag in selected:
        node = document.find(tag)
        if node is not None:
            if any(child.tag in {'ObservedData', 'DataRepository', 'SimulationResults', 'IndividualResults'} for child in node.iter()):
                raise ValueError('Outcome data cannot enter retained population inputs')
            wrapper.append(deepcopy(node))
            fingerprints[tag] = source_pkml.element_sha(node)
    path = output / 'original_population_inputs.xml'
    if path.exists():
        raise FileExistsError('Never overwrite original population inputs')
    ET.ElementTree(wrapper).write(path, encoding='utf-8', xml_declaration=True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha_file(path),
            'original_element_sha256': fingerprints, 'population_execution_claimed': False}


def export_all(output=DEFAULT_OUTPUT):
    output = inside(output)
    if output.exists():
        raise FileExistsError('Complete export needs a fresh destination')
    catalog = {r['candidate_id']: r for r in map(json.loads, CATALOG.open())}
    output.mkdir(parents=True)
    records = []
    for candidate_id, (compound, repository) in CANDIDATES.items():
        candidate = catalog[candidate_id]
        provenance = candidate['provenance']
        source = inside(ROOT / provenance['source_path'])
        inventory = source_pkml.simulation_inventory(source, provenance['source_sha256'])
        for simulation in inventory:
            destination = output / compound / hashlib.sha256(simulation['simulation_id'].encode()).hexdigest()[:16]
            row = {'candidate_id': candidate_id, 'compound': compound, 'repository': repository,
                   'repository_commit': provenance['repository_commit'], 'simulation': simulation}
            try:
                result = source_pkml.extract_saved_model(source, provenance['source_sha256'],
                                                        simulation['simulation_id'], destination)
                document = read_document(source, simulation)
                row['structure'] = structural_inventory(document)
                if row['structure']['model_xml_sha256'] != result['audit']['model_xml_sha256']:
                    raise ValueError('Saved model structural checksum differs after extraction')
                row['original_population_inputs'] = preserve_population_inputs(document, destination)
                row.update(status='exported_pending_native_validation', extraction=result,
                           sealed_observations_accessed=False, ode_executed=False)
            except Exception as exc:
                row.update(status='blocked', reasons=[type(exc).__name__ + ':' + str(exc)])
            records.append(row)
        if sha_file(source) != provenance['source_sha256']:
            raise ValueError('Original catalog project changed during extraction')
    summary = {'schema': 'complete-pregnancy-catalog-export-v1',
        'implementation': [{'path': str(path.relative_to(ROOT)), 'sha256': sha_file(path)}
                           for path in [Path(__file__), Path(source_pkml.__file__)]],
        'catalog_sha256': sha_file(CATALOG), 'records': records,
        'counts': {'catalog_candidates': len(CANDIDATES), 'saved_simulations': len(records),
            'exported': sum(r['status'] == 'exported_pending_native_validation' for r in records),
            'blocked': sum(r['status'] == 'blocked' for r in records),
            'pregnancy_structures': sum(r.get('structure', {}).get('complete_pregnancy_marker_set_present', False) for r in records)},
        'clinical_observations_accessed': False, 'native_solver_calls': 0,
        'candidate_substitution_performed': False, 'frozen_experiment_modified': False,
        'limitations': ['Complete extracted PKML requires native import, parameter readback and solver verification.',
            'Saved population definitions are retained but their sampling has not been executed.',
            'Original reference-instance execution is distinct from a population prediction.',
            'Pregnancy structure is audited by retained elements, never inferred solely from repository name.']}
    write_json(output / 'pregnancy_export_manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = export_all(args.output)
    print(json.dumps(result['counts'], ensure_ascii=False))
    raise SystemExit(0 if result['counts']['blocked'] == 0 else 1)
