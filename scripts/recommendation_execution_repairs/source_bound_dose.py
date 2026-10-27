"""Product-label mass to source-model parent mass; never fit an observed result.

The sole registered transformation is the frozen Teaima Wellbutrin SR case.
Other salt/product identities deliberately retain the original normalizer's block.
The 2011 FDA description defines tablet strength as hydrochloride mass; source
models define bupropion with MW 239.741 g/mol. Historical model protocol names
and apparent dose ratios are not evidence for this chemical conversion.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from recommendation_candidate_executor import Blocked, _digest, _inside, scenario_contract as original_contract
from .source_pkml import decompress_model_blob, model_parameter_inventory

VERSION = 'source-bound-product-salt-dose-v1'
CASE = 'bupropion_teaima2020_wellbutrin_sr_150mg'
PROTOCOL_SHA = '573a303a4d124c67a95b501fb0842d44793f33ac31204072764ea3257286cff4'
LABEL_PATH = 'artifacts/pbpk-model-assets/execution_repairs_20261001/dose_convention_sources/fda_wellbutrin_sr_2011_020358s050lbl.pdf'
LABEL_SHA = 'b2b8993f8a45e08344f492e464df232ef9ce14190fea37ff561258e48f6230aa'
LABEL_URL = 'https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/020358s050lbl.pdf'
PARENT_MW = Decimal('239.741')
SALT_MW = Decimal('276.2')
ALIASES = frozenset({'Bupropion', 'Bupropion_FM'})
REGISTERED_CANDIDATES = {
    '242ab5e4-ed60-5e72-b9e6-ba9e5505e4fa': '612342dd63ca5ed043a61c2104be5e4f41d03fed236b98aa6393f56236cfcb30',
    '402489ae-2d1f-5839-8ab6-7ea12b0bee69': 'e5913f5e0880e163e20e4a0796a4d671136d333c1ffdff1b77627821f5bdeb01',
    '5825b60f-7009-5b2a-a113-494cb73c2c03': 'b78938d7ca6b8fa03e5c57bc3515b403a6bd245d86120c93a7e0efaa8fbcdf8a',
    '673404c9-fe74-5197-bea1-bac985d54667': '74b3619ececc6583ece1246f5d3c48b54b60bf97b58b128db587ba53eadba4b2',
    '6847a701-cafb-5257-812f-47efbef91313': '195c2fc857b67b14eea34b60a24ef3ae87b0c618e650aa241a05549ab69aa4d4',
    'c15b47af-4289-5faa-934d-145e14ac68b7': '0fd82b504ca67bf55ccdee00e4b69043eea0df58f572481ac20224f7583da10d',
    'c47a5d8e-7ccc-592e-95c5-a5459e8eb8a5': '57727c9a779c118fdba29486cb134723fdc829b4bdb466e8b6ce501349d24712',
    'cc6a1133-a39d-5692-bbdd-72a1a1d7ce3b': '0635dcc6fd3f9f45c91b0f5ca480159926f0f57772c9af4e37ed78def0e5ea02',
}


def _sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _exact_number(value, expected, reason):
    if isinstance(value, bool): raise Blocked(reason)
    try: actual = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): raise Blocked(reason)
    if not actual.is_finite() or actual != expected: raise Blocked(reason)


def _formula_definition(model, identifier):
    cache = model.find('FormulaCache')
    if cache is None: raise Blocked('SOURCE_DOSE_FORMULA_CACHE_REQUIRED')
    choices = [node for node in cache.iter() if node.get('id') == identifier and node.tag != 'Map']
    if len(choices) != 1: raise Blocked('UNIQUE_SOURCE_DOSE_FORMULA_REQUIRED')
    node = choices[0]
    strings = {row.get('id'): row.get('s') for row in cache.findall('StringMap/Map')}
    paths = [{key: strings.get(value, value) for key, value in row.attrib.items()}
             for row in node.findall('Paths/Path')]
    return {'expression': node.get('formula'), 'paths': paths, 'id': identifier}


@lru_cache(maxsize=16)
def _read_original_parent(path_text, source_sha):
    """Only original definitions; PK-Sim result/observed tables are not queried."""
    path = Path(path_text)
    records = []
    if path.suffix == '.json':
        document = json.loads(path.read_text(), parse_float=str, parse_int=str)
        compounds = [(index, c) for index, c in enumerate(document.get('Compounds', [])) if c.get('Name') in ALIASES]
        if len(compounds) != 1: raise Blocked('UNIQUE_ORIGINAL_BUPROPION_PARENT_REQUIRED')
        index, compound = compounds[0]
        rows = [(i, p) for i, p in enumerate(compound.get('Parameters', [])) if p.get('Name') == 'Molecular weight']
        if len(rows) != 1: raise Blocked('UNIQUE_ORIGINAL_PARENT_MW_REQUIRED')
        i, mw = rows[0]
        if mw.get('Unit') != 'g/mol': raise Blocked('SOURCE_PARENT_MW_UNIT_MISMATCH')
        _exact_number(mw.get('Value'), PARENT_MW, 'SOURCE_PARENT_MW_MISMATCH')
        # A snapshot-local override must not quietly change the model species.
        for simulation in document.get('Simulations', []):
            for parameter in simulation.get('Parameters', []):
                if parameter.get('Path') == compound['Name']+'|Molecular weight':
                    if parameter.get('Unit') != 'g/mol': raise Blocked('SOURCE_PARENT_OVERRIDE_UNIT_MISMATCH')
                    _exact_number(parameter.get('Value'), PARENT_MW, 'SOURCE_PARENT_MW_OVERRIDE_MISMATCH')
        records.append({'compound_name': compound['Name'], 'pointer': f'/Compounds/{index}/Parameters/{i}',
                        'raw_numeric': mw['Value'], 'raw_unit': 'g/mol', 'molecular_weight_g_per_mol': str(PARENT_MW),
                        'saved_drugmass_formula': None,
                        'formula_scope': 'Snapshot defines parent molecular weight and mass InputDose; complete ODE formula is not serialized in JSON.'})
    elif path.suffix == '.pksim5':
        with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            rows = db.execute('SELECT s.SimulationId,b.Name,c.Data FROM SIMULATIONS s '
                'JOIN BUILDING_BLOCKS b ON b.Id=s.SimulationId JOIN CONTENTS c ON c.Id=b.ContentId '
                'ORDER BY s.SimulationId').fetchall()
        for simulation_id, simulation_name, blob in rows:
            document = ET.fromstring(decompress_model_blob(blob))
            if document.get('id') != simulation_id: raise Blocked('SOURCE_SIMULATION_IDENTITY_MISMATCH')
            model = document.find('Model')
            parameters = model_parameter_inventory(model)
            matches = [(name, parameters[name+'|Molecular weight']) for name in ALIASES if name+'|Molecular weight' in parameters]
            if not matches: continue
            if len(matches) != 1: raise Blocked('UNIQUE_ORIGINAL_BUPROPION_PARENT_REQUIRED')
            name, mw = matches[0]
            if mw.get('dimension') != 'Molecular weight' or mw.get('formula_id') or mw.get('rhs_formula_id'):
                raise Blocked('SOURCE_PARENT_MW_MUST_BE_EXPLICIT_CONSTANT')
            _exact_number(mw['value'], PARENT_MW/Decimal(10)**9, 'SOURCE_PARENT_MW_MISMATCH')
            formulas = []
            for p in parameters.values():
                if not p['path'].endswith('|ProtocolSchemaItem|DrugMass'): continue
                formula = _formula_definition(model, p.get('formula_id'))
                aliases = {x.get('as'): x.get('path') for x in formula['paths']}
                if aliases.get('MW') != simulation_name+'|'+name+'|Molecular weight': continue
                if formula['expression'] != 'Dose / MW' or aliases.get('Dose') != '..|Dose':
                    raise Blocked('SOURCE_PARENT_DRUGMASS_FORMULA_NOT_MASS_OVER_MW')
                formulas.append({'path': p['path'], **formula})
            if not formulas: raise Blocked('SOURCE_PARENT_DRUGMASS_FORMULA_REQUIRED')
            records.append({'simulation_id': simulation_id, 'simulation_name': simulation_name,
                'compound_name': name, 'parameter_path': mw['path'], 'parameter_id': mw['id'],
                'raw_numeric': mw['value'], 'raw_unit': 'kg/umol',
                'molecular_weight_g_per_mol': str(PARENT_MW), 'saved_drugmass_formulas': formulas,
                'source_blob_sha256': hashlib.sha256(blob).hexdigest()})
    else: raise Blocked('ORIGINAL_JSON_OR_PKSIM5_DOSE_SOURCE_REQUIRED')
    if not records: raise Blocked('ORIGINAL_PARENT_MW_EVIDENCE_REQUIRED')
    if _sha(path) != source_sha: raise Blocked('DOSE_SOURCE_CHANGED_DURING_READ')
    return records


def normalized_salt_contract(scenario, candidate, *, project_root=ROOT):
    """Return original_contract(normalized copy), with auditable dose conversion.

    No shared caller is modified. Only the exact registered case/source is
    admitted. This fixes mass identity only, not release kinetics, population,
    food, coadministration, or candidate scientific applicability.
    """
    if scenario.get('case_id') != CASE:
        return original_contract(scenario)
    root = Path(project_root).resolve(strict=True)
    ref = scenario.get('source_protocol', {})
    if ref.get('sha256') != PROTOCOL_SHA: raise Blocked('SALT_PROTOCOL_NOT_REGISTERED')
    protocol_path = _inside(ref.get('path', ''), root)
    if _sha(protocol_path) != PROTOCOL_SHA: raise Blocked('SALT_PROTOCOL_CHECKSUM_MISMATCH')
    protocol = json.loads(protocol_path.read_text())
    dose = scenario.get('normalized_design', {}).get('dose', {})
    if protocol.get('case_id') != CASE or protocol.get('compound') != 'bupropion' or scenario.get('compound') != 'bupropion':
        raise Blocked('SALT_PROTOCOL_IDENTITY_MISMATCH')
    if dose != protocol.get('dose') or scenario.get('protocol_fields') != protocol:
        raise Blocked('SALT_DESIGN_DIFFERS_FROM_FROZEN_PROTOCOL')
    _exact_number(dose.get('value'), Decimal('150'), 'REGISTERED_150MG_LABEL_DOSE_REQUIRED')
    if dose.get('unit') != 'mg' or dose.get('formulation') != 'Wellbutrin SR sustained-release tablet':
        raise Blocked('EXACT_REGISTERED_WELLBUTRIN_SR_PRODUCT_REQUIRED')
    if dose.get('active_moiety_dose_mg') is not None: raise Blocked('SALT_DOSE_ALREADY_NORMALIZED')
    label = _inside(LABEL_PATH, root)
    if _sha(label) != LABEL_SHA: raise Blocked('FDA_PRODUCT_LABEL_CHECKSUM_MISMATCH')
    provenance = candidate.get('provenance', candidate.get('catalog_provenance', {}))
    source_sha = provenance.get('source_sha256')
    if REGISTERED_CANDIDATES.get(candidate.get('candidate_id')) != source_sha or not source_sha:
        raise Blocked('SALT_CANDIDATE_SOURCE_NOT_REGISTERED')
    if 'bupropion' not in candidate.get('compounds', candidate.get('registered_compounds', [])):
        raise Blocked('CANDIDATE_PARENT_IDENTITY_NOT_REGISTERED')
    source = _inside(provenance.get('source_path', ''), root)
    if _sha(source) != source_sha: raise Blocked('SALT_MODEL_SOURCE_CHECKSUM_MISMATCH')
    parent = deepcopy(_read_original_parent(str(source), source_sha))
    if _sha(source) != source_sha: raise Blocked('DOSE_SOURCE_CHANGED_DURING_READ')
    factor = PARENT_MW/SALT_MW
    converted = Decimal('150')*factor
    conversion = {'factor': float(factor), 'factor_decimal': str(factor),
        'source_file': LABEL_PATH, 'source_file_sha256': LABEL_SHA,
        'source_url': LABEL_URL, 'source_pages_1based': [36, 61],
        'rule': 'labelled bupropion hydrochloride mg * original parent MW / labelled hydrochloride MW',
        'salt_formula': 'C13H18ClNO.HCl', 'salt_molecular_weight_g_per_mol': str(SALT_MW),
        'parent_molecular_weight_g_per_mol': str(PARENT_MW),
        'parent_model_source': {'path': str(source.relative_to(root)), 'sha256': source_sha},
        'parent_definition_evidence': parent, 'historical_protocol_ratio_used': False}
    normalized = deepcopy(scenario)
    normalized['normalized_design']['dose']['active_moiety_dose_mg'] = float(converted)
    normalized['protocol_fields']['salt_conversion'] = conversion
    result = original_contract(normalized)
    result['dose_normalization'] = {'version': VERSION,
        'operation': 'labelled_hydrochloride_mass_to_source_parent_mass',
        'original_reported_dose': deepcopy(dose), 'registered_dose_mg': float(converted),
        'registered_dose_decimal_mg': str(converted), 'dimension_check': 'mg * (g/mol)/(g/mol) = mg',
        'source_protocol': deepcopy(ref), 'original_scenario_sha256': _digest(scenario),
        'salt_conversion': deepcopy(conversion), 'target_product_salt_identity_verified': True,
        'strict_target_match': False, 'remaining_scope': 'Only chemical mass is normalized. Original-model release, physiology, population, food and coadministration gates remain required.',
        'fitting_performed': False, 'clinical_outcomes_accessed': False}
    return result
