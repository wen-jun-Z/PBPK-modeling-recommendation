#!/usr/bin/env python3
"""Render current tables, normalizing the two explicit native metadata schemas.

The simulation adapter is itself frozen by the nine-scenario protocol. This
small reporting entry leaves that file and every native prediction unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import sys

import render_unified_blueprint_tables as tables
from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.pk_anchored_blueprint import mapped_snapshot
from pbpk_kb.blueprints.renal_extension import add_renal_outputs
from pbpk_kb.blueprints.source_workflow import read_curve


def native_audit(manifest_path):
    manifest, _, _ = tables.selected_context(manifest_path)
    records = {}
    for case in manifest['cases']:
        prediction = tables.source(case['target_prediction'])
        request_path = tables.source(case['target_request'])
        mapping_path = tables.source(case['mapping'])
        request, mapping = tables.read(request_path), tables.read(mapping_path)
        folder = prediction.parent
        snapshot_path, execution_path = (folder / 'native' / name for name in ('snapshot.json', 'execution_manifest.json'))
        snapshot, execution = tables.read(snapshot_path), tables.read(execution_path)
        if execution['source_sha256'] != tables.sha256(snapshot_path) or not execution['engineering_accepted']:
            raise ValueError('Native snapshot hash/execution checks failed')
        blueprint = assemble_evidence_blueprint(request, tables.ROOT)
        if blueprint['status'] != 'assembled':
            raise ValueError('Current frozen request no longer assembles')
        expected = mapped_snapshot(blueprint, mapping['decision']['lipophilicity'], mapping['decision']['clearance_scale'])
        if case['config']['renal_enabled']:
            expected = add_renal_outputs(expected)
        if snapshot != expected:
            raise ValueError('Executed native snapshot differs from the current mapping and request')
        curve = tables.read(prediction)
        original = read_curve(folder / 'native', execution, snapshot['Simulations'][0]['OutputSelections'][0])
        if original != curve:
            raise ValueError('Locked concentration curve differs from the hashed native CSV')
        endpoint_path = folder / 'endpoints.json'
        saved = tables.read(endpoint_path)
        recomputed = summarize_extended_prediction(curve, request)
        for key in ('endpoints', 'main_window', 'diagnostics', 'extension', 'observations_used'):
            if saved[key] != recomputed[key]:
                raise ValueError('Native endpoint recomputation differs: ' + case['case_id'] + '/' + key)
        records[case['case_id']] = {'status': 'passed',
            'hashes': {tables.rel(path): tables.sha256(path) for path in
                (mapping_path, request_path, snapshot_path, execution_path, prediction, endpoint_path)},
            'snapshot_reassembled_identically': True, 'native_csv_equals_locked_curve': True,
            'all_native_endpoints_and_quality_diagnostics_recomputed_identically': True}
    return records


def main():
    # This compatibility applies to the native endpoint estimator only, not to
    # upstream parameter fitting, which did use a development clinical curve.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--manifest', type=tables.Path, default=tables.MANIFEST)
    args, _ = parser.parse_known_args()
    full_native_audit = {} if any(arg in sys.argv for arg in ('--help', '-h')) else native_audit(args.manifest)
    original_read = tables.clinical.comparison.read_json
    normalized = {}

    def read_compatible(path):
        record = original_read(path)
        resolved = tables.inside(path, tables.ROOT, exists=True)
        if resolved.name != 'endpoints.json' or 'endpoint_estimator_uses_observations' in record:
            return record
        if (record.get('observations_used') is not False
                or record.get('endpoint_estimator_fitted_pbpk_parameters') is not False):
            raise ValueError('Native endpoint estimation lacks explicit prediction-only flags')
        result_path = resolved.parent / 'result.json'
        result = original_read(result_path)
        if result.get('comparison_curve_used') is not False or result.get('native_audits_passed') is not True:
            raise ValueError('Native result must reject comparison-curve fitting and pass its physical audits')
        normalized[tables.rel(resolved)] = {
            'sha256': tables.sha256(resolved),
            'observations_used': False,
            'endpoint_estimator_fitted_pbpk_parameters': False,
            'normalized_endpoint_estimator_uses_observations': False,
            'result_path': tables.rel(result_path), 'result_sha256': tables.sha256(result_path),
            'comparison_curve_used': False, 'native_audits_passed': True}
        compatible = deepcopy(record)
        compatible['endpoint_estimator_uses_observations'] = False
        return compatible

    tables.clinical.comparison.read_json = read_compatible
    try:
        tables.main()
    finally:
        tables.clinical.comparison.read_json = original_read
    if normalized:
        audit_path = tables.clinical.DEST / 'blueprint_clinical_audit.json'
        audit = tables.read(audit_path)
        audit['native_metadata_compatibility'] = {
            'schema': 'Explicit observations_used=false and endpoint_estimator_fitted_pbpk_parameters=false',
            'scope': 'Endpoint calculation only; upstream development fitting is retained and disclosed',
            'files_changed': False, 'files': normalized}
        audit['native_source_reconstruction'] = full_native_audit
        audit['source_sha256'][tables.rel(__file__)] = tables.sha256(tables.Path(__file__))
        tables.save(audit_path, audit)


if __name__ == '__main__':
    main()
