"""Recheck exact candidate/study source audits without parsing clinical outcomes.

This proves only the stated public-document nonoverlap scope. It does not prove
participant independence, target-population validity, or scientific usability.
Only the frozen protocol and audit JSON are parsed; referenced model/paper files
are streamed for SHA verification, never interpreted as observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = "candidate-source-audit-recheck-v2"
PASS_STATUS = "verified_disjoint_within_audited_clinical_sources"
PASS_FLAGS = ("clinical_source_disjoint_at_audited_source_level",
              "clinical_source_independence_supported_at_declared_scope")
MATCH_FIELDS = {"overlaps", "known_clinical_overlap", "matching_target_sources",
                "target_identity_matches", "matched_target_identifiers",
                "target_identifier_or_title_matches", "matching_references",
                "target_identifiers_found", "first_author_year_flags"}
KNOWN_OVERLAP_REVIEWS = {
    ("clinical:dextromethorphan_yamazaki2016_day1_30mg", "6023dcee-888b-56fe-a7c6-d3ca0ab75e20"): {
        "path": "experiments/model_recommendation/execution_validation_20261001/source_preservation_repairs/dose_historical_product_audit_20261001.json",
        "sha256": "98983f65cac582109b18be679407862fc487aac52f9090c34e50068b5b3f7902",
    },
}


def _refs(value):
    if isinstance(value, dict):
        if "path" in value:
            if not isinstance(value["path"], str) or not isinstance(value.get("sha256"), str):
                raise ValueError("audit_document_reference_lacks_path_or_sha256")
            yield value
        for item in value.values():
            yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _matches(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in MATCH_FIELDS and item:
                yield key
            yield from _matches(item)
    elif isinstance(value, list):
        for item in value:
            yield from _matches(item)


def _identifier(key, value):
    value = str(value).strip().casefold()
    if key == "doi":
        value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", "", value)
    return value


def known_overlap_freeze_refs(*, project_root=ROOT):
    """Return exact review/repository evidence refs for the pre-outcome barrier.

    These are additional inputs, not replacements for frozen scenarios/sources.
    A missing or changed registered attachment is fatal here, never ``unknown``.
    """
    root = Path(project_root).resolve(strict=True)
    refs = {}
    def read(ref):
        p = (root / ref['path']).resolve(strict=True)
        if (not p.is_relative_to(root) or not p.is_file()
                or any(part.casefold().startswith('sealed') for part in p.relative_to(root).parts)):
            raise ValueError('known_overlap_freeze_resource_outside_project_or_sealed')
        if p.stat().st_size > 2 * 1024**2:
            raise ValueError('known_overlap_freeze_resource_exceeds_2MiB')
        data = p.read_bytes()
        if len(data) > 2 * 1024**2 or hashlib.sha256(data).hexdigest() != ref.get('sha256'):
            raise ValueError('known_overlap_freeze_resource_sha256_changed')
        normalized = {'path': str(p.relative_to(root)), 'sha256': ref['sha256']}
        if normalized['path'] in refs and refs[normalized['path']] != normalized:
            raise ValueError('known_overlap_freeze_conflicting_reference')
        refs[normalized['path']] = normalized
        return p, data
    for (sid, cid), review_ref in KNOWN_OVERLAP_REVIEWS.items():
        _, payload = read(review_ref)
        review = json.loads(payload)
        overlap = review['source_study_overlap']
        original = review['dextromethorphan']['original_author_case_mapping']
        if (overlap.get('scenario_id') != sid or overlap.get('candidate_id') != cid
                or overlap.get('verified_overlap') is not True
                or original.get('candidate_id') != cid
                or overlap.get('source_sha256') != original['source']['sha256']):
            raise ValueError('known_overlap_freeze_identity_binding_differs')
        doi = _identifier('doi', overlap.get('target_doi', ''))
        doi_found = role_found = False
        for evidence in original['study_identity_evidence']:
            p, payload = read(evidence)
            if p.suffix.casefold() != '.md':
                raise ValueError('known_overlap_freeze_requires_repository_markdown')
            lines = payload.decode('utf-8').splitlines()
            start, end = evidence['lines_1based']
            if (type(start) is not int or type(end) is not int
                    or not 1 <= start <= end <= len(lines)):
                raise ValueError('known_overlap_freeze_invalid_line_range')
            excerpt = '\n'.join(lines[start - 1:end])
            if excerpt != evidence['text']:
                raise ValueError('known_overlap_freeze_excerpt_differs')
            doi_found |= bool(doi) and doi in [v.rstrip('.,;)') for v in
                re.findall(r'10\.\d{4,9}/[^\s<>]+', excerpt.casefold())]
            role_found |= ('used for model verification' in excerpt.casefold()
                           and 'yamazaki 2017' in excerpt.casefold())
        if not doi_found or not role_found:
            raise ValueError('known_overlap_freeze_doi_or_verification_role_unconfirmed')
    return [refs[key] for key in sorted(refs)]


def _known_overlap(review_ref, scenario, audit, protocol, source_ref, *, metadata, verify, path):
    """Validate a separately registered review; never mutate the frozen audit."""
    review = metadata(review_ref)
    overlap = review["source_study_overlap"]
    if (overlap.get("verified_overlap") is not True
            or overlap.get("scenario_id") != scenario["scenario_id"]
            or overlap.get("candidate_id") != audit["candidate_id"]
            or overlap.get("source_sha256") != source_ref["sha256"]):
        raise ValueError("known_overlap_review_candidate_scenario_or_source_mismatch")
    if not any(ref.get("sha256") == scenario["source_protocol"]["sha256"]
               and path(ref["path"]) == path(scenario["source_protocol"]["path"])
               for ref in review.get("clinical_design_sources", [])):
        raise ValueError("known_overlap_review_protocol_not_bound")
    identifiers = protocol.get("identifiers") or protocol.get("source_identifiers") or {}
    doi = _identifier("doi", overlap.get("target_doi", ""))
    if not doi or doi != _identifier("doi", identifiers.get("doi", "")):
        raise ValueError("known_overlap_target_doi_mismatch")
    if (overlap.get("target_pmcid") and _identifier("pmcid", overlap["target_pmcid"])
            != _identifier("pmcid", identifiers.get("pmcid", ""))):
        raise ValueError("known_overlap_target_pmcid_mismatch")
    original = review["dextromethorphan"]["original_author_case_mapping"]
    if (original.get("candidate_id") != audit["candidate_id"]
            or original["source"]["sha256"] != source_ref["sha256"]
            or path(original["source"]["path"]) != path(source_ref["path"])):
        raise ValueError("known_overlap_original_source_binding_mismatch")
    verify(original["source"])
    verify(overlap["existing_frozen_audit_ref"])
    evidence = original.get("study_identity_evidence", [])
    doi_found = role_found = False
    verified_excerpts = []
    for ref in evidence:
        # These are bounded public repository documents, never clinical tables.
        p = path(ref["path"], parse_metadata=True)
        if p.suffix.casefold() != ".md" or p.stat().st_size > 2 * 1024**2:
            raise ValueError("known_overlap_evidence_must_be_bounded_repository_markdown")
        verify(ref)
        payload = p.read_bytes()
        if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != ref["sha256"]:
            raise ValueError("known_overlap_evidence_sha256_changed_during_read")
        lines = payload.decode("utf-8").splitlines()
        start, end = ref["lines_1based"]
        if (type(start) is not int or type(end) is not int
                or start < 1 or end < start or end > len(lines)):
            raise ValueError("known_overlap_evidence_line_range_invalid")
        excerpt = "\n".join(lines[start - 1:end])
        if excerpt != ref["text"]:
            raise ValueError("known_overlap_repository_excerpt_mismatch")
        found = re.findall(r"10\.\d{4,9}/[^\s<>]+", excerpt.casefold())
        doi_found |= doi in [value.rstrip(".,;)") for value in found]
        role_found |= ("used for model verification" in excerpt.casefold()
                       and "yamazaki 2017" in excerpt.casefold())
        verified_excerpts.append({"path": ref["path"], "sha256": ref["sha256"],
                                  "lines_1based": [start, end]})
    if not doi_found or not role_found or overlap.get("original_model_role") != "DGI Model Verification":
        raise ValueError("known_overlap_exact_reference_doi_or_verification_role_unconfirmed")
    return {
        "status": "documented_source_study_overlap",
        "source_study_overlap_verified": True,
        "strict_unseen_study_stratum_eligible": False,
        "study_usage_role": "verification",
        "used_for_parameter_fitting": "not_established_from_reviewed_metadata",
        "allowed_reporting_stratum": "original_author_verification_study_reexecution",
        "scope": "The exact target publication was explicitly used to verify this source model; this pair is not an unseen-study validation.",
        "review_overlay": {"path": review_ref["path"], "sha256": review_ref["sha256"],
                           "target_doi": doi, "source_sha256": source_ref["sha256"],
                           "frozen_candidate_audit_changed": False,
                           "prior_independence_status": audit.get("independence_status"),
                           "verified_repository_excerpts": verified_excerpts},
    }


def audit_candidate_source(scenario, audit, *, project_root=ROOT):
    """Return scoped pass, documented overlap, or unknown, with limitations.

    The caller must already have frozen the scenario and candidate audit rows.
    Eligibility is deliberately restricted to their 44 pre-bound audit pairs;
    same-drug/family membership or an inline assertion cannot confer eligibility.
    Call after the complete experiment lock barrier if model-paper paths reside
    under a sealed directory. No clinical target outcome is opened or parsed.
    """
    result = {"schema_version": VERSION, "status": "unknown", "reasons": [],
              "candidate_id": audit.get("candidate_id"), "scenario_id": scenario.get("scenario_id"),
              "source_independence_confirmed_at_declared_scope": False,
              "scientific_applicability_established": False, "strict_target_match_established": False,
              "clinical_outcome_values_parsed": False, "limitations": [],
              "scope": "No target clinical study identified in the audited public model-development sources; not universal participant-level independence.",
              "verified_references": [], "audit_results": []}
    checked, verified = {}, {}
    try:
        root = Path(project_root).resolve(strict=True)

        def path(value, *, parse_metadata=False):
            p = (root / value).resolve(strict=True)
            if p == root or not p.is_relative_to(root) or not p.is_file():
                raise ValueError("source_audit_resource_outside_project_or_not_file")
            if parse_metadata and any(part.casefold().startswith("sealed") for part in p.relative_to(root).parts):
                raise ValueError("sealed_resource_cannot_be_parsed_as_audit_metadata")
            return p

        def verify(ref):
            expected = ref.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid_declared_source_audit_sha256")
            p = path(ref["path"])
            if p not in checked:
                with p.open("rb") as stream:
                    checked[p] = hashlib.file_digest(stream, "sha256").hexdigest()
            if checked[p] != expected:
                raise ValueError("source_audit_reference_sha256_changed:" + str(p.relative_to(root)))
            verified[p] = expected
            return p

        def metadata(ref):
            # Reject sealed metadata before verify() opens its bytes.
            p = path(ref["path"], parse_metadata=True)
            if p.stat().st_size > 2 * 1024**2:
                raise ValueError("source_audit_metadata_exceeds_2MiB")
            payload = p.read_bytes()
            if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != ref.get("sha256"):
                raise ValueError("source_audit_metadata_sha256_changed:" + str(p.relative_to(root)))
            checked[p] = verified[p] = ref["sha256"]
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("source_audit_metadata_must_be_object")
            return value

        known_review = KNOWN_OVERLAP_REVIEWS.get((scenario.get("scenario_id"), audit.get("candidate_id")))
        if (audit.get("identity_match") is not True or (known_review is None and (
                audit.get("bound_snapshot_exact_match") is not True
                or audit.get("independence_status") != "previously_audited_pending_recheck"))):
            raise ValueError("candidate_has_no_previously_frozen_exact_source_audit_binding")
        if (audit.get("scenario_id") != scenario.get("scenario_id")
                or audit.get("case_id") != scenario.get("case_id")
                or audit.get("compound") != scenario.get("compound")):
            raise ValueError("candidate_audit_scenario_case_or_compound_mismatch")
        if not scenario.get("case_id") or not audit.get("candidate_id"):
            raise ValueError("explicit_candidate_and_case_identity_required")
        if "catalog_candidate_ids" in scenario and audit["candidate_id"] not in scenario["catalog_candidate_ids"]:
            raise ValueError("candidate_not_in_frozen_scenario_catalog_binding")
        prov = audit["catalog_provenance"]
        source_ref = {"path": prov["source_path"], "sha256": prov["source_sha256"]}
        source = verify(source_ref)
        asset = audit.get("source_asset", {})
        if (asset.get("state") != "verified" or asset.get("sha256_matches") is not True
                or asset.get("actual_sha256") != source_ref["sha256"] or path(asset["path"]) != source):
            raise ValueError("candidate_source_asset_not_verified_at_bound_hash")
        if known_review is not None:
            protocol = metadata(scenario["source_protocol"])
            if protocol.get("case_id") != scenario["case_id"] or protocol.get("compound") != scenario["compound"]:
                raise ValueError("frozen_protocol_case_or_compound_mismatch")
            result.update(_known_overlap(known_review, scenario, audit, protocol, source_ref,
                                         metadata=metadata, verify=verify, path=path))
            result["limitations"].append("Original model verification reused the target study; parameter fitting to it has not been established.")
            result["verified_references"] = [{"path": str(p.relative_to(root)), "sha256": sha}
                                             for p, sha in sorted(verified.items())]
            return result
        independence = scenario.get("source_independence", {})
        for bound_refs in (audit.get("exact_bound_references", []), independence.get("bound_snapshot_references", [])):
            if not any(ref.get("sha256") == source_ref["sha256"] and path(ref["path"]) == source for ref in bound_refs):
                raise ValueError("candidate_source_not_in_exact_frozen_audit_bindings")
        protocol = metadata(scenario["source_protocol"])
        if protocol.get("case_id") != scenario["case_id"] or protocol.get("compound") != scenario["compound"]:
            raise ValueError("frozen_protocol_case_or_compound_mismatch")
        protocol_ids = protocol.get("identifiers") or protocol.get("source_identifiers") or {}
        if not any(protocol_ids.get(k) for k in ("doi", "pmcid", "pmid")):
            raise ValueError("frozen_protocol_has_no_explicit_publication_identity")
        audit_refs = independence.get("audit_references", [])
        if not audit_refs:
            raise ValueError("inline_source_assertion_is_not_a_bound_document_audit")
        links = [protocol[k] for k in ("source_audit", "source_independence_audit") if isinstance(protocol.get(k), dict)]
        for ref in audit_refs:
            if not any(link.get("sha256") == ref.get("sha256") and path(link["path"]) == path(ref["path"]) for link in links):
                raise ValueError("protocol_to_audit_path_or_sha256_mismatch")
            document = metadata(ref)
            if document.get("compound") not in (None, scenario["compound"]):
                raise ValueError("source_audit_compound_mismatch")
            if (any(document.get(k) is False for k in PASS_FLAGS)
                    or not (document.get("status") == PASS_STATUS or any(document.get(k) is True for k in PASS_FLAGS))):
                raise ValueError("source_audit_has_no_uncontradicted_scoped_pass")
            if any(_matches(document)):
                raise ValueError("source_audit_has_unresolved_target_match_or_overlap")
            target = document.get("target_identifiers") or document.get("target_bibliography", {}).get("identifiers") or {}
            if target:
                meaningful = {k: v for k, v in target.items() if k in {"doi", "pmcid", "pmid"} and v}
                if not meaningful or any(_identifier(k, v) != _identifier(k, protocol_ids.get(k, "")) for k, v in meaningful.items()):
                    raise ValueError("source_audit_target_publication_mismatch")
            references = list(_refs(document))
            if not references:
                raise ValueError("source_audit_has_no_hash_bound_reviewed_documents")
            for reference in references:
                verify(reference)
            for name in ("limitations", "unknowns", "qualification_limit", "supplement_scope",
                         "original_model_journal_fulltext_not_available_locally"):
                value = document.get(name)
                if value:
                    result["limitations"].extend(value if isinstance(value, list) else [value])
            result["audit_results"].append({"path": ref["path"], "sha256": ref["sha256"],
                "target_binding": "explicit_publication_identifiers" if target else "frozen_protocol_to_legacy_audit_link",
                "scoped_pass_field": "status" if document.get("status") == PASS_STATUS else next(k for k in PASS_FLAGS if document.get(k) is True),
                "reviewed_reference_count": len(references),
                "declared_scope": document.get("audited_scope", document.get("scope", document.get("source_qualification_scope", document.get("source_chain"))))})
        result.update(status="scoped_source_audit_pass", source_independence_confirmed_at_declared_scope=True)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result["reasons"].append(type(exc).__name__ + ":" + str(exc))
    result["limitations"].append("Unreported participant reuse and undocumented model-development material cannot be excluded.")
    result["verified_references"] = [{"path": str(p.relative_to(root)), "sha256": sha} for p, sha in sorted(verified.items())]
    return result
