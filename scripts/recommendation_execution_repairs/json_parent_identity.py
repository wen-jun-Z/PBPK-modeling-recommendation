"""Two author-documented JSON parent names, bound to complete source chemistry.

This resolves names only. It does not merge drugs, rename model records, change
molecular weight/dose, switch formulations, or generate a total-analyte formula.
"""
from pathlib import Path
import hashlib
import json

VERSION = 'json-author-parent-identity-1'
_BINDINGS = {
    ('montelukast', 'd625e417f439724e54edf6bb03f76ba0bf09057d0a52dda6db1e78f8480e188b'): {
        'case_id': 'montelukast_pmc9929805_nct05531994_reference5mg_chewable_fed',
        'native_name': 'Montelukast_Ince et al 2019',
        'source': {'path': 'artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Montelukast-DDI__70eb2e5bbfc3/source/Clopidogrel-Montelukast-DDI.json',
                   'sha256': 'd625e417f439724e54edf6bb03f76ba0bf09057d0a52dda6db1e78f8480e188b'},
        'author_document': {'path': 'artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Montelukast-DDI__70eb2e5bbfc3/source/README.md',
                            'sha256': 'dec191ac075ee1c5805b0721fbb3496c7d2b7fe43ea7755fab118d3a1d3c5086'},
        'compound_sha256': 'bd3c53c8b2079bd1a9ce5bc18c1c70a74460bcec1a72497a5a0b77c5ca09d999',
        'molecular_weight_g_per_mol': 586.2,
        'basis': 'Author README identifies the applied published Montelukast model and links OSP Montelukast-Model v1.1; the unique original native parent and entire definition are bound by SHA256.'},
    ('pioglitazone', '704519ec69a7be0caf0affe3af2129ada41f8f069fdcb8c1d2a276171954c226'): {
        'case_id': 'pioglitazone_pmc7591087_nct02753803_day7_monotherapy30mg',
        'native_name': 'Pioglitazone_DT',
        'source': {'path': 'artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Pioglitazone-DDI__0d73e9b3edb3/source/Clopidogrel-Pioglitazone-DDI.json',
                   'sha256': '704519ec69a7be0caf0affe3af2129ada41f8f069fdcb8c1d2a276171954c226'},
        'author_document': {'path': 'artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Pioglitazone-DDI__0d73e9b3edb3/source/README.md',
                            'sha256': '0bcf09d9f2752c1ccbaa1d1f10e1f0c6c7ad8fff1a7c3c4e255fec085b846c21'},
        'compound_sha256': '7feed8bcd2fa9f4653ccb99f55c2736ff7379eef56453bdc4309c769753a07b1',
        'molecular_weight_g_per_mol': 356.4,
        'basis': 'Author README explicitly identifies the published pioglitazone model (PMID31129789); the unique original native parent and entire definition are bound by SHA256.'},
}


def _checked(ref, root):
    path=(root/ref['path']).resolve(strict=True)
    if not path.is_relative_to(root) or 'sealed' in {part.lower() for part in path.parts}:
        raise ValueError('AUTHOR_PARENT_IDENTITY_PATH_OUTSIDE_PROJECT')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
        raise ValueError('AUTHOR_PARENT_IDENTITY_EVIDENCE_CHANGED:'+ref['path'])
    return path


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_parent_identity(contract, source_sha256, *, project_root):
    """Return an exact reviewed alias, or None so ordinary strict identity applies."""
    binding=_BINDINGS.get((contract.get('compound'),source_sha256))
    if binding is None:return None
    if (contract.get('case_id')!=binding['case_id'] or contract.get('analyte')!='parent'
            or contract.get('binding')!='total' or contract.get('matrix') not in {'plasma','whole_blood'}):
        raise ValueError('AUTHOR_PARENT_IDENTITY_REQUIRES_FROZEN_PARENT_ANALYTE')
    root=Path(project_root).resolve(strict=True)
    source=_checked(binding['source'],root);_checked(binding['author_document'],root)
    document=json.loads(source.read_text())
    matches=[row for row in document.get('Compounds',[]) if row.get('Name')==binding['native_name']]
    if len(matches)!=1 or _digest(matches[0])!=binding['compound_sha256']:
        raise ValueError('AUTHOR_PARENT_IDENTITY_COMPOUND_DEFINITION_CHANGED')
    weight=[row for row in matches[0].get('Parameters',[]) if row.get('Name')=='Molecular weight']
    if (len(weight)!=1 or weight[0].get('Unit')!='g/mol'
            or weight[0].get('Value')!=binding['molecular_weight_g_per_mol']):
        raise ValueError('AUTHOR_PARENT_IDENTITY_MOLECULAR_WEIGHT_CHANGED')
    return {'version':VERSION,'native_name':binding['native_name'],'clinical_parent':contract['compound'],
        'source_sha256':source_sha256,'source':dict(binding['source']),
        'author_document':dict(binding['author_document']),'compound_definition_sha256':binding['compound_sha256'],
        'original_molecular_weight_g_per_mol':binding['molecular_weight_g_per_mol'],
        'basis':binding['basis'],'analyte_mapping':'unchanged original named parent concentration',
        'source_compound_or_mechanism_renamed':False,'dose_rescaled':False,'concentrations_aggregated':False}
