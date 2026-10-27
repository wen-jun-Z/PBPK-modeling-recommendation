#!/usr/bin/env python3
"""Independent chemistry and structure/scaffold-excluded RF for the 30-drug batch.

Reads curated chemical inputs, not archived fitted Compounds or clinical curves.
Uses the existing fixed PKSmart-data RF recipe without hyperparameter search.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BATCH = ROOT / 'experiments/blueprint_design/sampling_expansion_20261009'
OLD = ROOT / 'experiments/blueprint_design/expanded_validation_20261007'
OLD20 = ROOT / 'experiments/blueprint_design/expanded_tail_20260930/inputs/unified_manifest.json'
WHEEL = ROOT / 'artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl'
DRUGS = ('Ethinylestradiol', 'Levonorgestrel', 'Mexiletine', 'Moclobemide',
         'Paclitaxel', 'Probenecid', 'Rosuvastatin', 'Rifampicin', 'Montelukast', 'Finerenone')
VERSION = 'sampling-expansion-independent-priors-1'


def complete_external_normalizer(original):
    """Extend only a failed enumeration budget, never accept partial tautomers.

    Rifampicin has 2,098 enumerated tautomers and exceeds the old 256/512
    safeguards. Both training and excluded structures use the same fallback.
    The unchanged feature definition still operates on completed canonicals.
    """
    def normalize(smiles):
        try:
            return original(smiles)
        except ValueError as exc:
            if not str(exc).startswith('Incomplete bounded tautomer normalization:'):
                raise
        from rdkit import Chem, rdBase
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
        with rdBase.BlockLogs():
            mol = rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smiles))
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetMaxTautomers(4096)
            enumerator.SetMaxTransforms(16384)
            candidates = enumerator.Enumerate(mol)
            if candidates.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
                raise ValueError('Incomplete bounded expanded tautomer normalization: ' + str(candidates.status))
            mol = enumerator.PickCanonical(candidates)
        Chem.RemoveStereochemistry(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        identity = Chem.MolToSmiles(mol, isomericSmiles=False)
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
        return {'identity_smiles': identity,
                'scaffold_group': 'MURCKO:' + scaffold if scaffold else 'ACYCLIC',
                'scaffold_smiles': scaffold}
    return normalize


def read(path):
    return json.loads(Path(path).read_text())


def ref(path, location='$'):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Project boundary')
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'location': location}


def verify_sources(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            if ref(ROOT / value['path'])['sha256'] != value['sha256']:
                raise ValueError('Changed source: ' + value['path'])
        for item in value.values():
            verify_sources(item)
    elif isinstance(value, list):
        for item in value:
            verify_sources(item)


def save(path, data):
    path = Path(path).resolve()
    if not path.is_relative_to(BATCH):
        raise ValueError('Output must remain in this batch')
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != payload:
        raise ValueError('Refusing to overwrite different prepared data: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return ref(path)


def prepare_chemistry():
    properties = read(OLD / 'pubchem_properties.json')['records']
    curated = read(OLD / 'chemical_sources.json')['compounds']
    facts_path = BATCH / 'inputs/physchem/supplemental_chemical_facts.json'
    facts = read(facts_path)
    output = {}
    for drug in DRUGS:
        record = properties[drug]
        if record['status'] != 'downloaded':
            raise ValueError('Missing verified molecular identity: ' + drug)
        p = record['response']['PropertyTable']['Properties'][0]
        cid = 'pubchem:' + str(p['CID'])
        if drug in curated:
            c = curated[drug]
            if c['construction_status'] != 'ready':
                raise ValueError('Chemically unsupported compound: ' + drug)
            evidence = deepcopy(c['evidence'])
        else:
            evidence = []
            for parameter, value, unit, field in (
                    ('molecular_weight', float(p['MolecularWeight']), 'g/mol', 'MolecularWeight'),
                    ('molecular_formula', p['MolecularFormula'], 'text', 'MolecularFormula'),
                    ('logp', float(p['XLogP']), 'Log Units', 'XLogP')):
                e = {'parameter': parameter, 'value': value, 'unit': unit, 'compound_id': cid,
                     'basis': 'predicted' if parameter == 'logp' else 'derived',
                     'source': ref(OLD / 'pubchem_properties.json', '/records/' + drug + '/response/PropertyTable/Properties/0/' + field),
                     'context': {'target_observations_used': False, 'fitted_compound_parameters_used': False},
                     'notes': 'PubChem independent parent chemical property, not a fitted PBPK input.'}
                if e['basis'] == 'predicted':
                    e['prediction_method'] = {'name': 'PubChem XLogP', 'version': 'frozen-20261007',
                        'training_overlap': 'unknown', 'applicability': 'Physicochemical estimate, not clinical evidence'}
                evidence.append(e)
            fact = facts['records'][drug]
            e = {'parameter': 'pka_types', 'compound_id': cid, 'value': fact['pka_types'],
                 'unit': '1', 'basis': fact['basis'], 'notes': fact['notes'],
                 'source': ref(facts_path, '/records/' + drug),
                 'context': {'target_observations_used': False, 'fitted_compound_parameters_used': False,
                             'url': fact['url'], 'source_location': fact['source_location']}}
            if fact['basis'] == 'predicted':
                e['prediction_method'] = {'name': 'ChemAxon pKa as reported by public database',
                    'version': 'unreported_by_source', 'training_overlap': 'unknown',
                    'applicability': 'Chemical prediction only; not an experimental pKa measurement'}
            evidence.append(e)
        # Independent binding may be useful separately, but a uniform RF fu is
        # requested in this batch. Do not silently import a different policy.
        evidence = [e for e in evidence if e['parameter'] in
                    {'molecular_weight', 'molecular_formula', 'logp', 'pka_types'}]
        verify_sources(evidence)
        vals = {e['parameter']: e['value'] for e in evidence}
        if len(vals['pka_types']) > 3:
            raise ValueError('Unsupported polyprotic representation')
        physical = {'version': VERSION, 'drug': drug, 'compound_id': cid,
            'identity': {'pubchem_cid': p['CID'], 'molecular_weight_g_mol': float(p['MolecularWeight']),
                'molecular_formula': p['MolecularFormula'], 'isomeric_smiles': p.get('SMILES', p.get('IsomericSMILES'))},
            'logp': vals['logp'], 'pka_types': vals['pka_types'], 'evidence': evidence,
            'source_files': [ref(OLD / 'pubchem_properties.json'),
                ref(OLD / 'chemical_sources.json') if drug in curated else ref(facts_path)],
            'construction_status': 'ready',
            'limitations': ['Independent physicochemical inputs do not prove clinical prediction accuracy.',
                'XLogP and marked pKa predictions are not measured clinical evidence.',
                'No archived fitted drug parameters, CL, Vss, target concentration or target scalar endpoint copied.']}
        output[drug] = save(BATCH / 'inputs/physchem' / (drug.lower() + '.json'), physical)
    return output


def train_priors():
    import joblib
    from pbpk_kb.blueprints import pk_parameter_learning as learning
    chemistry = prepare_chemistry()
    old = read(OLD20)
    structures = {c['drug']: read(ROOT / c['physchem'])['identity']['isomeric_smiles'] for c in old['cases']}
    structures.update({name: read(ROOT / source['path'])['identity']['isomeric_smiles']
                       for name, source in chemistry.items()})
    if len(structures) != 30:
        raise ValueError('Expected exactly 30 unique named drugs')
    folder = BATCH / 'inputs/priors'
    if (folder / 'parameter_models.joblib').exists():
        prior = read(folder / 'parameter_split.json')
        if {n: r['input_smiles'] for n, r in prior['external_exclusions'].items()} != structures:
            raise ValueError('Existing RF external exclusions differ')
        verify_sources(read(folder / 'predictions.json')['sources'])
        return read(folder / 'predictions.json')
    original_normalizer = learning.normalize_structure
    learning.normalize_structure = complete_external_normalizer(original_normalizer)
    try:
        learned = learning.run_parameter_pilot(WHEEL, exclude_smiles=structures, train_limit=1283,
                                              test_limit=100, seed=20260929, n_jobs=4)
        predictions = {name: learned.predict(smiles) for name, smiles in structures.items()}
    finally:
        learning.normalize_structure = original_normalizer
    learned.audit['feature_specification'] = deepcopy(learned.audit['feature_specification'])
    learned.audit['feature_specification']['tautomer_limits'] = {
        'initial': {'max_tautomers': 256, 'max_transforms': 512},
        'fallback': {'max_tautomers': 4096, 'max_transforms': 16384},
        'incomplete_enumeration': 'exclude; external unresolved structure blocks training',
        'applies_to': 'training table and all external structures equally'}
    learned.audit['normalization_adapter'] = ref(Path(__file__))
    if any(p['training_structure_overlap'] or p['training_scaffold_overlap'] for p in predictions.values()):
        raise ValueError('External compound leakage')
    identities = {p['identity_smiles'] for p in predictions.values()}
    if len(identities) != 30:
        raise ValueError('Duplicate standardized external parent identity')
    folder.mkdir(parents=True, exist_ok=True)
    split = save(folder / 'parameter_split.json', learned.audit)
    data = {'version': VERSION, 'external_predictions': predictions,
        'held_out_parameter_metrics': learned.metrics, 'external_drugs': 30,
        'clinical_curves_used_for_RF_training': False,
        'all_thirty_drug_structures_and_scaffolds_excluded': True,
        'sources': [ref(WHEEL), ref(OLD20), ref(Path(__file__)), split,
                    *[ref(ROOT / c['physchem']) for c in old['cases']], *chemistry.values()],
        'unit_conversion': {'CL_L_h': 'human_CL_mL_min_kg * actual_request_weight_kg * 60 / 1000',
            'Vss_L': 'human_VDss_L_kg * actual_request_weight_kg', 'fu': 'human_fup (fraction)'},
        'limitations': ['RF predictions are prior estimates, not measured or PBPK-fitted parameters.',
            'Vss is provided as a prior prediction; do not silently map it into tissue partition factors.',
            'Old results using prior20 and new30 priors must not be pooled as one unchanged predictor.']}
    save(folder / 'predictions.json', data)
    joblib.dump(learned.models, folder / 'parameter_models.joblib', compress=3)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['chemistry', 'train'])
    args = parser.parse_args()
    out = prepare_chemistry() if args.command == 'chemistry' else train_priors()
    print(json.dumps({'command': args.command, 'status': 'complete',
        'drugs': list(out) if args.command == 'chemistry' else list(out['external_predictions'])}))
