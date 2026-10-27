"""Exclude final-study outcomes from design, fitting priors and LLM evidence.

Clinical dose/demographic metadata can be supplied separately after curation;
this exception never authorizes the raw article, sealed outcomes or their hash.
This checks supplied inputs, not unpublished ancestry or the model's pretraining.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re


def _file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusion_policy(protocol_paths, *, project_root):
    root = Path(project_root).resolve(strict=True)
    denied_paths, denied_hashes, terms, protocols = set(), set(), set(), []
    metadata_sources = {}
    for relative in protocol_paths:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or path == root:
            raise ValueError('External protocol must remain in project')
        document = json.loads(path.read_text())
        if document.get('scientific_results_accessed_by_algorithm') is not False:
            raise ValueError('Protocol has no explicit outcome-blinding assertion')
        exclusion = document['evidence_exclusion']
        if not exclusion.get('exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training'):
            raise ValueError('Whole publication/cohort exclusion must be explicit')
        for kind, value in exclusion['identifiers'].items():
            if value:
                terms.add((kind, str(value).lower()))
        terms.update(('trial', str(value).lower()) for value in exclusion.get('trial_registry_ids', []))
        raw_paths = list(exclusion.get('raw_source_files', [])) + [document['sealed_observation_reference']['path']]
        for source in raw_paths:
            target = (root / source).resolve(strict=True)
            if target == root or not target.is_relative_to(root):
                raise ValueError('Excluded study artifact leaves project')
            denied_paths.add(str(target.relative_to(root)).lower())
            denied_hashes.add(_file_hash(target))
        denied_hashes.add(document['source_article_sha256'])
        denied_hashes.add(document['sealed_observation_reference']['sha256'])
        checksum = _file_hash(path)
        metadata_sources[str(path.relative_to(root))] = checksum
        protocols.append({'case_id':document['case_id'], 'study_id':document['study_id'],
            'cohort_id':document['cohort_id'], 'path':str(path.relative_to(root)), 'sha256':checksum})
    return {'schema_version':'pbpk-external-source-isolation-v1', 'protocols':protocols,
        'denied_paths':sorted(denied_paths), 'denied_hashes':sorted(denied_hashes),
        'denied_identifiers':[{'kind':kind,'value':value} for kind,value in sorted(terms)],
        'curated_metadata_sources':metadata_sources,
        'scope':'supplied design/parameter/evidence/LLM inputs; external protocol metadata is separated from raw outcomes'}


def audit_design_inputs(inputs, policy, *, allowed_metadata_sources=()):
    """Fail closed on a denied source even when embedded in a nested reference.

    Pass design requests and parameter contracts, not the final-study audit
    itself. No filesystem, database, retrieval or external outcome is read here.
    """
    allowed = set(allowed_metadata_sources)
    if allowed - policy['curated_metadata_sources'].keys():
        raise ValueError('Only checksummed curated metadata can be allowed')
    denied_paths = set(policy['denied_paths'])
    denied_hashes = set(policy['denied_hashes'])
    violations, references = [], 0
    def inspect(value, location):
        nonlocal references
        if isinstance(value, dict):
            metadata_reference = False
            if 'source_id' in value and 'source_checksum' in value:
                references += 1
                source, checksum = value['source_id'], value['source_checksum']
                if source in allowed and checksum == policy['curated_metadata_sources'][source]:
                    # A protocol's study name is allowed only through its exact
                    # metadata provenance. Denied raw hashes always win.
                    if checksum in denied_hashes:
                        violations.append(location + ':denied_raw_metadata_hash')
                    metadata_reference = True
            for key,item in value.items():
                if metadata_reference and key in {'source_id','source_checksum','source_location','provenance_id','provenance_kind'}:
                    continue
                if key not in {'forbidden_study_ids','forbidden_source_ids'}:
                    inspect(item, location + '.' + str(key))
        elif isinstance(value, list):
            for i,item in enumerate(value):
                inspect(item, location + '[' + str(i) + ']')
        elif isinstance(value, str):
            text = value.lower()
            if text in denied_hashes or any(path in text for path in denied_paths):
                violations.append(location + ':denied_source_file_or_hash')
            for identifier in policy['denied_identifiers']:
                token = identifier['value']
                if re.search(r'(?<![a-z0-9])' + re.escape(token) + r'(?![a-z0-9])', text):
                    violations.append(location + ':denied_' + identifier['kind'])
    inspect(inputs, '$')
    if violations:
        raise ValueError('External study entered design inputs: ' + '; '.join(sorted(set(violations))[:12]))
    return {'accepted':True, 'checked_provenance_references':references,
        'excluded_study_count':len({p['study_id'] for p in policy['protocols']}),
        'external_outcomes_read':False, 'allowed_metadata_sources':sorted(allowed)}
