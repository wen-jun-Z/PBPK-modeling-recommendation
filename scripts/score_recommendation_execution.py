"""Score frozen native PBPK predictions only after the complete experiment locks.

The reference-subject stratum describes numerical agreement, never a direct-reuse
label. Published group NCA Cmax and a sampled reference-individual peak differ in
their estimand; their ratio is explicitly diagnostic. No fixed-window integral is
substituted for AUClast, AUCinf, or steady-state AUC. No fitting or outcome-directed
candidate selection occurs in this module.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recommendation_execution_metrics as metrics
import recommendation_execution_statistics as statistics

VERSION = 'pbpk-clinical-execution-scoring-v1'
TERMINAL = {'blocked', 'prediction_locked', 'native_failed', 'readback_failed', 'failed'}
EXECUTION_REVISION = 'engineering_repair_02'
EXECUTION_IMPLEMENTATION_REQUIRED = frozenset({
    'scripts/recommendation_candidate_executor.py',
    'src/pbpk_kb/blueprints/source_adaptation.py',
    'src/pbpk_kb/blueprints/source_adaptation_audit.py',
    'src/pbpk_kb/blueprints/source_output_grid.py',
    'src/pbpk_kb/blueprints/source_workflow.py',
    'src/pbpk_kb/blueprints/design_runtime.py',
    'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
    'src/pbpk_kb/model_runtime/R/source_adaptation_audit.R',
})


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Files:
    """An explicit outcome gate; resolving symlinks cannot bypass it."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.unsealed = False
        self.outcome_paths = set()
        self.opened_outcomes = {}

    def path(self, value):
        p = (self.root / value).resolve()
        if not p.is_relative_to(self.root) or p == self.root:
            raise ValueError('Resource must remain inside this project')
        if not self.unsealed and (p in self.outcome_paths or any(
                part.lower().startswith('sealed') for part in p.relative_to(self.root).parts)):
            raise ValueError('Sealed outcome access before the complete lock barrier')
        return p

    def sha(self, path):
        with self.path(path).open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()

    def read(self, path):
        return json.loads(self.path(path).read_text())

    def rows(self, path):
        return [json.loads(s) for s in self.path(path).read_text().splitlines() if s.strip()]

    def checked(self, reference):
        p = self.path(reference['path'])
        if not reference.get('sha256') or self.sha(p) != reference['sha256']:
            raise ValueError('Frozen SHA differs: ' + str(p.relative_to(self.root)))
        return p

    def outcome(self, reference):
        if not self.unsealed:
            raise ValueError('The entire ranking/native barrier is required before outcomes')
        p = self.checked(reference)
        self.opened_outcomes[str(p.relative_to(self.root))] = reference['sha256']
        return self.read(p)


def unique(rows, key, expected=None):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows) or (expected is not None and set(result) != set(expected)):
        raise ValueError('Duplicate, missing, or unexpected ' + key)
    return result


def verify_execution_binding(record, *, frozen_code_sha256, expected_revision,
                             label, expected_implementation=None):
    """Tie every native artifact, even a blocked result, to one frozen revision.

    Protocol code has already been hashed against current project files. Explicit
    required dependencies prevent an otherwise matching one-file receipt from
    concealing an old solver, adaptation, grid, or readback implementation.
    """
    if record.get('execution_revision') != expected_revision:
        raise ValueError(label + ' execution revision differs or is missing')
    implementation = record.get('implementation_sha256')
    if not isinstance(implementation, dict) or not EXECUTION_IMPLEMENTATION_REQUIRED.issubset(implementation):
        raise ValueError(label + ' execution implementation binding is incomplete')
    for path, sha in implementation.items():
        if (not isinstance(path, str) or Path(path).is_absolute() or Path(path).as_posix() != path
                or '..' in Path(path).parts or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha)
                or frozen_code_sha256.get(path) != sha):
            raise ValueError(label + ' implementation is not bound to current protocol code: ' + str(path))
    if expected_implementation is not None and implementation != expected_implementation:
        raise ValueError(label + ' implementation differs from frozen native plan')
    return implementation


def verify_preparation_binding(f, run, protocol, frozen_code_sha256):
    """Strict default preparation checks; repaired runs must authenticate history."""
    for name in ('preparation.json',):
        if (run / name).exists():
            preparation = f.read(run / name)
            for ref in [*preparation.get('inputs', []), *preparation.get('code', [])]:
                f.checked(ref)
            if preparation.get('sealed_outcomes_accessed') is not False:
                raise ValueError('Preparation outcome access is not false')


def verify_barrier(run_dir, *, project_root=ROOT, expected_methods=None, allow_missing_declared_tiers=False):
    """Read no outcomes; verify all units including blocked/failed candidates.

    The optional root/method arguments support isolated synthetic barrier tests.
    Production additionally requires the complete registered method/code lists.
    """
    import numpy as np
    f = Files(project_root)
    if allow_missing_declared_tiers and f.root == ROOT:
        raise ValueError('Production cannot omit the frozen engineering-policy declarations')
    run = f.path(run_dir)
    protocol = f.read(run / 'protocol.json')
    if protocol.get('schema_version') != 'pbpk-recommendation-execution-protocol-v1':
        raise ValueError('Unrecognized protocol')
    if protocol.get('execution_revision') != EXECUTION_REVISION:
        raise ValueError('Protocol execution revision differs or is missing')
    scenarios = f.rows(run / 'inputs/scenarios.jsonl')
    by_scenario = unique(scenarios, 'scenario_id')
    for scenario in scenarios:
        for ref in scenario.get('sealed_references', []):
            p = (f.root / ref['path']).resolve()
            if not p.is_relative_to(f.root):
                raise ValueError('Outcome reference outside project')
            f.outcome_paths.add(p)
    frozen = {str(f.checked(ref).relative_to(f.root)) for ref in protocol['inputs']}
    frozen_code_sha256 = {str(f.checked(ref).relative_to(f.root)): ref['sha256'] for ref in protocol['code']}
    frozen_code = set(frozen_code_sha256)
    required = ['inputs/cohort.json', 'inputs/scenarios.jsonl', 'inputs/candidate_audit.jsonl',
                'inputs/execution_plan.json', 'rankings/inputs/queries.jsonl',
                'rankings/inputs/candidates.jsonl', 'rankings/inputs/manifest.json']
    if not allow_missing_declared_tiers:
        required.extend(['inputs/declared_tiers.jsonl', 'inputs/declared_tiers_manifest.json'])
    if not {str((run / x).relative_to(f.root)) for x in required}.issubset(frozen):
        raise ValueError('Protocol did not freeze every required input')
    if f.root == ROOT:
        import rank_recommendation_execution as ranking
        import run_recommendation_execution as driver
        expected_methods = list(ranking.METHODS)
        if not set(driver.CODE).issubset(frozen_code):
            raise ValueError('Protocol did not freeze all execution/scoring code')
    if expected_methods is not None and protocol['methods'] != list(expected_methods):
        raise ValueError('Protocol omitted or reordered registered methods')
    if not {'scripts/score_recommendation_execution.py',
            'scripts/recommendation_execution_metrics.py'}.issubset(frozen_code):
        raise ValueError('Scoring code was not frozen')
    access = protocol.get('outcome_access', {})
    if access.get('sealed_values_read_by_preparation') is not False or access.get(
            'all_predictions_and_rankings_locked_before_scoring') is not True:
        raise ValueError('Protocol lacks the complete outcome barrier')
    cohort = f.read(run / 'inputs/cohort.json')
    for path, reference in cohort.get('source_files', {}).items():
        ref = reference if isinstance(reference, dict) else {'path': path, 'sha256': reference}
        f.checked(ref)
    verify_preparation_binding(f, run, protocol, frozen_code_sha256)
    rank = run / 'rankings'
    queries = f.rows(rank / 'inputs/queries.jsonl')
    candidates = f.rows(rank / 'inputs/candidates.jsonl')
    qids, cids = [q['query_id'] for q in queries], [c['candidate_id'] for c in candidates]
    unique(queries, 'query_id', by_scenario)
    unique(candidates, 'candidate_id')
    if len(queries) != protocol['cases'] or len(candidates) != protocol['candidates']:
        raise ValueError('Clinical ranking cardinality differs from protocol')
    manifest = f.read(rank / 'inputs/manifest.json')
    for path, sha in manifest.get('source_files', {}).items():
        f.checked({'path': path, 'sha256': sha})
    for name in ('queries.jsonl', 'candidates.jsonl'):
        f.checked({'path': str(rank / 'inputs' / name), 'sha256': manifest['outputs'][name]['sha256']})
    hashes = {str((rank / 'inputs' / n).relative_to(f.root)): f.sha(rank / 'inputs' / n)
              for n in ('queries.jsonl', 'candidates.jsonl', 'manifest.json')}
    lock = f.read(rank / 'ranking_lock.json')
    if lock.get('state') != 'complete' or lock.get('scientific_outcomes_read') is not False:
        raise ValueError('Ranking lock is incomplete or outcomes were read')
    if lock.get('query_ids') != qids or lock.get('candidate_ids') != cids or set(
            lock.get('methods', {})) != set(protocol['methods']):
        raise ValueError('Ranking lock has incomplete identities/methods')
    if f.checked(lock['manifest']) != rank / 'inputs/manifest.json':
        raise ValueError('Ranking lock manifest differs')
    matrices = {}
    for method, entry in lock['methods'].items():
        base = rank / 'results' / method
        index = f.read(f.checked(entry['index']))
        status = f.read(f.checked(entry['status']))
        if index != {'query_ids': qids, 'candidate_ids': cids, 'variants': ['original']}:
            raise ValueError('Frozen score index differs')
        if status.get('state') != 'complete' or status.get('input_sha256') != hashes:
            raise ValueError('Ranking method is incomplete or used different inputs')
        if status.get('clinical_wrapper_sha256') != f.sha(f.root / 'scripts/rank_recommendation_execution.py'):
            raise ValueError('Ranking wrapper changed')
        implementation = status.get('implementation_sha256', {})
        if not {'run_frontier_comparison.py', 'frontier_external_models.py'}.issubset(implementation):
            raise ValueError('Ranking implementation not recorded')
        for name, sha in implementation.items():
            f.checked({'path': 'scripts/' + name, 'sha256': sha})
        f.checked(entry['original_scores'])
        if f.checked(entry['index']) != base / 'index.json' or f.checked(entry['status']) != base / 'status.json':
            raise ValueError('Ranking lock points to another method')
        scores = np.load(f.checked(entry), allow_pickle=False)
        if scores.shape != (1, len(qids), len(cids)) or not np.isfinite(scores).all():
            raise ValueError('Ranking matrix incomplete or nonfinite')
        matrices[method] = scores[0]
    audits = f.rows(run / 'inputs/candidate_audit.jsonl')
    pairs = {(r['scenario_id'], r['candidate_id']): r for r in audits}
    expected_pairs = {(q, c) for q in qids for c in cids}
    if len(pairs) != len(audits) or set(pairs) != expected_pairs or len(audits) != protocol['complete_scope_audit_pairs']:
        raise ValueError('Full catalog identity audit is incomplete')
    declared_tiers, tiers_manifest = {}, None
    if not allow_missing_declared_tiers:
        import recommendation_execution_tiers as tiers
        if Path(tiers.ROOT).resolve() != f.root or Path(tiers.OUT).resolve() != run:
            raise ValueError('Declared-tier verifier is bound to a different frozen run')
        declared = tiers.verify_declared_tiers(check_original_sources=True)
        tiers_manifest = declared['manifest']
        if tiers_manifest != f.read(run / 'inputs/declared_tiers_manifest.json'):
            raise ValueError('Declared-tier verifier manifest differs from protocol-bound manifest')
        if not {ref['path'] for ref in tiers_manifest['code']}.issubset(frozen_code):
            raise ValueError('Protocol did not freeze every engineering-policy implementation')
        if f.checked(tiers_manifest['output']) != run / 'inputs/declared_tiers.jsonl':
            raise ValueError('Declared-tier output is not the frozen run output')
        declared_tiers = {(r['scenario_id'], r['candidate_id']): r for r in declared['rows']}
        if len(declared_tiers) != len(declared['rows']) or set(declared_tiers) != expected_pairs:
            raise ValueError('Declared tiers do not cover the complete frozen ranking index')
        originals = unique(tiers.rows(tiers.CATALOG), 'candidate_id', cids)
        mapped = unique(tiers_manifest['mapped_queries'], 'scenario_id', qids)
        for sid, row in mapped.items():
            if row.get('scenario_sha256') != digest(by_scenario[sid]):
                raise ValueError('Declared-tier clinical scenario changed')
        for (sid, cid), row in declared_tiers.items():
            source = originals[cid]['provenance']
            if row.get('candidate_record_sha256') != digest(originals[cid]) or row.get('source_sha256') != source['source_sha256']:
                raise ValueError('Declared-tier candidate/source binding changed')
            if row.get('query_graph_sha256') != mapped[sid]['graph_sha256']:
                raise ValueError('Declared-tier graph binding changed')
            asset = pairs[(sid, cid)]['source_asset']
            if asset.get('actual_sha256') != source['source_sha256'] or f.path(asset['path']) != f.path(source['source_path']):
                raise ValueError('Declared-tier source is not the audited execution candidate')
    asset_checks = set()
    for row in audits:
        asset = row.get('source_asset', {})
        if asset.get('readable') and asset.get('actual_sha256'):
            key = (asset['path'], asset['actual_sha256'])
            if key not in asset_checks:
                f.checked({'path': key[0], 'sha256': key[1]})
                asset_checks.add(key)
    plan = f.read(run / 'inputs/execution_plan.json')
    jobs = unique(plan['jobs'], 'job_id')
    same = {key for key, row in pairs.items() if row.get('identity_match') is True}
    job_pairs = {(j['scenario_id'], j['candidate_id']) for j in jobs.values()}
    if len(jobs) != protocol['planned_candidate_executions'] or len(job_pairs) != len(jobs) or job_pairs != same:
        raise ValueError('All same-parent plans, including blocked ones, are required')
    native_lock = f.read(run / 'native/predictions_lock.json')
    if f.checked(native_lock['protocol']) != run / 'protocol.json' or native_lock.get('target_observations_read') is not False:
        raise ValueError('Native lock protocol/outcome state differs')
    locked_jobs = unique(native_lock['jobs'], 'job_id', jobs)
    if native_lock.get('planned_jobs') != len(jobs):
        raise ValueError('Native lock did not include every planned unit')
    native = {}
    for jid, job in jobs.items():
        row = locked_jobs[jid]
        if any(row.get(k) != job[k] for k in ('scenario_id', 'candidate_id')):
            raise ValueError('Native lock identity differs')
        directory = run / 'native' / jid
        artifacts = {f.checked(ref) for ref in row['artifacts']}
        if not {directory / 'result.json', directory / 'execution_plan.json'}.issubset(artifacts):
            raise ValueError('Every native plan/result must be locked')
        if any(not p.is_relative_to(directory) for p in artifacts):
            raise ValueError('Native lock includes another execution unit')
        saved_plan = f.read(directory / 'execution_plan.json')
        result = f.read(directory / 'result.json')
        pair = (job['scenario_id'], job['candidate_id'])
        if saved_plan != job['plan'] or result.get('execution_plan_sha256') != f.sha(directory / 'execution_plan.json'):
            raise ValueError('Native plan changed')
        implementation = verify_execution_binding(saved_plan, frozen_code_sha256=frozen_code_sha256,
            expected_revision=protocol['execution_revision'], label='Native plan')
        verify_execution_binding(result, frozen_code_sha256=frozen_code_sha256,
            expected_revision=protocol['execution_revision'], label='Native result', expected_implementation=implementation)
        if saved_plan.get('scenario_sha256') != digest(by_scenario[pair[0]]) or saved_plan.get('candidate_sha256') != digest(pairs[pair]):
            raise ValueError('Plan no longer binds the exact scenario/candidate audit')
        if saved_plan.get('source_path'):
            f.checked({'path': saved_plan['source_path'], 'sha256': saved_plan['source_sha256']})
        migration = saved_plan.get('source_migration')
        if migration:
            for path_key, sha_key in [('receipt_path', 'receipt_sha256'),
                                      ('original_path', 'original_sha256'), ('path', 'sha256')]:
                f.checked({'path': migration[path_key], 'sha256': migration[sha_key]})
        if any(result.get(k) != job[k] for k in ('scenario_id', 'candidate_id')) or result.get('status') not in TERMINAL:
            raise ValueError('Native result lacks the correct terminal disposition')
        if result.get('sealed_observations_accessed') is not False or result.get('fitting_performed') is not False:
            raise ValueError('Native result does not assert outcome-free execution')
        prediction = None
        if result.get('prediction_locked'):
            if result['status'] != 'prediction_locked':
                raise ValueError('Native prediction flag/status mismatch')
            pp = f.checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']})
            lp = f.checked({'path': result['prediction_lock'], 'sha256': result['prediction_lock_sha256']})
            plock = f.read(lp)
            verify_execution_binding(plock, frozen_code_sha256=frozen_code_sha256,
                expected_revision=protocol['execution_revision'], label='Prediction lock', expected_implementation=implementation)
            if not {pp, lp}.issubset(artifacts):
                raise ValueError('Native prediction and lock missing from global lock')
            for key in ('scenario_sha256', 'candidate_sha256', 'source_sha256'):
                if plock.get(key) != saved_plan.get(key):
                    raise ValueError('Prediction lock/plan differs: ' + key)
            if plock.get('sealed_observations_accessed') is not False or plock.get('execution_verified') is not True:
                raise ValueError('Native execution/readback not verified')
            locked_files = {f.checked({'path': path, 'sha256': sha}) for path, sha in plock.get('files', {}).items()}
            if not {pp, directory / 'execution_plan.json'}.issubset(locked_files):
                raise ValueError('Prediction lock lacks prediction/plan')
            source = f.checked({'path': saved_plan['source_path'], 'sha256': saved_plan['source_sha256']})
            if result.get('source', {}).get('sha256') != saved_plan['source_sha256'] or f.path(result['source']['path']) != source:
                raise ValueError('Result/source/plan mismatch')
            prediction = f.read(pp)
            verify_execution_binding(prediction, frozen_code_sha256=frozen_code_sha256,
                expected_revision=protocol['execution_revision'], label='Prediction', expected_implementation=implementation)
            if any(prediction.get(k) != job[k] for k in ('scenario_id', 'candidate_id')):
                raise ValueError('Prediction identity mismatch')
            if prediction.get('compatibility') != saved_plan.get('compatibility') or result.get('compatibility') != saved_plan.get('compatibility'):
                raise ValueError('Prediction compatibility changed')
            if result.get('strict_target_match') != saved_plan.get('compatibility', {}).get('strict_target_match'):
                raise ValueError('Native result strict-target declaration differs from plan')
            if prediction.get('molecular_weight_g_per_mol') != saved_plan.get('molecular_weight_g_per_mol'):
                raise ValueError('Prediction molecular weight differs from frozen candidate')
            if result.get('predicted_endpoints') != prediction.get('endpoints'):
                raise ValueError('Result and prediction endpoint values differ')
        elif result['status'] == 'prediction_locked':
            raise ValueError('Prediction terminal state lacks locked prediction')
        native[pair] = {'job_id': jid, 'result': result, 'prediction': prediction, 'plan': saved_plan}
    # This is the only transition that permits any sealed file access.
    f.unsealed = True
    return {'files': f, 'run': run, 'protocol': protocol, 'scenarios': scenarios,
            'qids': qids, 'cids': cids, 'matrices': matrices, 'audits': pairs,
            'native': native, 'ranking_lock_sha256': f.sha(rank / 'ranking_lock.json'),
            'declared_tiers': declared_tiers, 'declared_tiers_manifest': tiers_manifest,
            'predictions_lock_sha256': f.sha(run / 'native/predictions_lock.json')}


def convert_unit(value, unit, molecular_weight, *, exposure=False):
    """Convert only explicitly dimensioned concentration/exposure; never dose normalize."""
    if not finite(value) or value <= 0:
        raise ValueError('nonpositive_or_missing_observation')
    u = str(unit or '').lower().replace('μ', 'u').replace('µ', 'u')
    u = u.replace('·', '*').replace(' ', '').replace('hr', 'h').replace('hours', 'h')
    time_factor = 1.
    if exposure:
        if u.startswith('h*'):
            u = u[2:]
        elif '*h/' in u:
            u = u.replace('*h/', '/')
        elif u.startswith('min*'):
            u, time_factor = u[4:], 1 / 60
        elif '*min/' in u:
            u, time_factor = u.replace('*min/', '/'), 1 / 60
        elif u.endswith('*h'):
            u = u[:-2]
        else:
            raise ValueError('exposure_unit_has_no_explicit_time_dimension')
    factors = {'ng/ml': 1., 'ug/l': 1., 'ug/ml': 1000., 'mg/l': 1000.,
               'mg/ml': 1.e6, 'ng/l': .001, 'pg/ml': .001}
    molar = {'nmol/l': .001, 'umol/l': 1., 'mol/l': 1.e6, 'mmol/l': 1000., 'nm': .001}
    if u in molar:
        if not finite(molecular_weight) or molecular_weight <= 0:
            raise ValueError('molar_conversion_requires_candidate_parent_molecular_weight')
        factor = molar[u] * molecular_weight
    elif u in factors:
        factor = factors[u]
    else:
        raise ValueError('unsupported_or_dose_normalized_unit:' + str(unit))
    converted = value * factor * time_factor
    if not finite(converted) or converted <= 0:
        raise ValueError('unit_conversion_nonfinite')
    return converted


def interpolate(times, values, wanted):
    if len(times) != len(values) or len(times) < 2 or not all(finite(v) for v in times + values):
        raise ValueError('invalid_native_curve')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('native_times_not_increasing')
    if wanted < times[0] - 1e-8 or wanted > times[-1] + 1e-8:
        raise ValueError('native_curve_extrapolation_forbidden')
    i = bisect.bisect_left(times, wanted)
    if i < len(times) and math.isclose(times[i], wanted, abs_tol=1e-8):
        return values[i]
    if not i or i == len(times):
        raise ValueError('native_curve_extrapolation_forbidden')
    w = (wanted - times[i - 1]) / (times[i] - times[i - 1])
    return values[i - 1] * (1 - w) + values[i] * w


def outcome_rows(document, scenario):
    """Do not duplicate caffeine's mass/molar representations or mix trial arms."""
    if 'endpoint_rows' in document:
        rows = [r for r in document['endpoint_rows'] if r.get('case_id') == scenario['case_id']
                and r.get('canonical_unit_representation') is True and r.get('counts_as_unique_endpoint') is True]
    else:
        if document.get('case_id') != scenario['case_id']:
            return []
        rows = document.get('observed_endpoints', document.get('endpoints', []))
    names = [r.get('endpoint') for r in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate scalar endpoints in one case/source')
    return rows


def endpoint_key(name):
    lower = re.sub(r'[^a-z0-9]', '', str(name).lower())
    window = re.fullmatch(r'auc_?0_?(\d+(?:\.\d+)?)h?', str(name).lower())
    if window:
        return 'AUC_0_' + format(float(window.group(1)), 'g') + 'h_reference_vs_group'
    if lower.startswith('cmax'):
        return 'Cmax_ss_reference_vs_group' if 'ss' in lower else 'Cmax_reference_vs_group'
    return str(name)


def scalar_comparison(row, scenario, prediction):
    name = str(row.get('endpoint') or '')
    out = {'endpoint': name, 'state': 'unavailable', 'reason': None,
           'source_aggregation': row.get('aggregation', row.get('statistic', row.get('population_statistic'))),
           'comparison_stratum': 'reference_diagnostic', 'scientific_status': 'unknown'}
    contracts = scenario['normalized_design'].get('endpoint_contracts', [])
    contract = next((c for c in contracts if c.get('endpoint', c.get('paper_target')) == name), {})
    if row.get('evaluable_from_source') is False or contract.get('evaluable_from_source') is False:
        return {**out, 'reason': 'source_endpoint_not_qualified'}
    matrix = str(scenario['normalized_design'].get('matrix') or '').lower()
    analyte = str(scenario['normalized_design'].get('analyte') or '').lower()
    # Curated descriptions explicitly say "not metabolite or unbound"; that is
    # positive parent-total evidence, not an analyte conflict.
    analyte = analyte.split('; not ')[0].split(', not ')[0]
    expected_matrix = {'plasma': 'plasma', 'blood plasma': 'plasma', 'whole_blood': 'whole_blood'}.get(matrix)
    if expected_matrix is None or 'parent' not in analyte or 'free' in analyte or 'unbound' in analyte or 'metabolite' in analyte:
        return {**out, 'reason': 'analyte_or_matrix_not_matched_to_total_parent_plasma_or_whole_blood'}
    if prediction.get('matrix') != expected_matrix or prediction.get('analyte') != 'parent' or prediction.get('binding') != 'total':
        return {**out, 'reason': 'native_prediction_matrix_analyte_binding_mismatch'}
    lower = re.sub(r'[^a-z0-9]', '', name.lower())
    auc_window = re.fullmatch(r'auc_?0_?(\d+(?:\.\d+)?)h?', name.lower())
    if lower.startswith('auc'):
        if not auc_window:
            # AUClast, infinity and tau cannot be recovered from a fixed requested grid.
            return {**out, 'reason': 'auc_definition_not_matched_fixed_sampling_window_is_not_last_infinity_or_tau',
                    'native_available_definition': 'AUC_fixed_sampling_window'}
        target_end = float(auc_window.group(1))
        endpoint = prediction.get('endpoints', {}).get('AUC_fixed_sampling_window')
        if not endpoint or endpoint.get('unit') != 'ng*h/mL' or endpoint.get('window_h') != [0, target_end]:
            return {**out, 'reason': 'explicit_fixed_AUC_window_not_equal_to_locked_prediction_window'}
        output_unit = 'ng*h/mL'
        endpoint_key = 'AUC_0_' + format(target_end, 'g') + 'h_reference_vs_group'
        warning = 'Dense native integral on the same explicit fixed window versus published group NCA AUC; reference-person/group estimand difference remains.'
    elif not lower.startswith('cmax'):
        return {**out, 'reason': 'endpoint_definition_not_registered_for_native_output'}
    else:
        endpoint = prediction.get('endpoints', {}).get('sampled_Cmax')
        dense = prediction.get('endpoints', {}).get('dense_grid_Cmax')
        registration = prediction.get('sampling_grid_registration') or {}
        from recommendation_execution_repairs.clinical_contract import _DENSE_REFERENCE_CASES, dense_reference_contract
        if scenario.get('case_id') in _DENSE_REFERENCE_CASES:
            expected_registration = dense_reference_contract(scenario)['sampling_grid_registration']
            if endpoint is not None or dense is None or registration != expected_registration:
                return {**out, 'reason': 'frozen_scenario_dense_Cmax_registration_or_definition_differs'}
        if dense is not None:
            if (endpoint is not None or registration.get('version') != 'frozen-horizon-dense-reference-grid-v1'
                    or registration.get('reported_clinical_sampling_grid_known') is not False
                    or registration.get('grid_kind') != 'registered_numerical_mesh_not_clinical_sampling'
                    or registration.get('selection_depends_on_clinical_outcomes') is not False):
                return {**out, 'reason': 'dense_Cmax_approximation_not_exclusively_registered'}
            endpoint = dense
        if not endpoint or endpoint.get('unit') != 'ng/mL':
            return {**out, 'reason': 'native_sampled_Cmax_missing_or_unit_mismatch'}
        output_unit = 'ng/mL'
        endpoint_key = 'Cmax_ss_reference_vs_group' if 'ss' in lower else 'Cmax_reference_vs_group'
        warning = 'Sampled reference-individual peak versus published group summary of individual NCA peaks; not equivalent population estimands.'
        if dense is not None:
            warning = 'Registered dense numerical-grid maximum versus published sampled group Cmax; original clinical sampling times are unknown; reference-individual/group estimands also differ.'
    try:
        if 'comparable_value' in row:
            value, unit = row['comparable_value'], row.get('comparable_unit')
        else:
            value, unit = row.get('value', row.get('mean')), row.get('unit', row.get('source_unit'))
        observed = convert_unit(value, unit, prediction.get('molecular_weight_g_per_mol'), exposure=bool(auc_window))
    except ValueError as exc:
        return {**out, 'reason': str(exc)}
    comparison = metrics.endpoint_error(endpoint.get('value'), observed, endpoint=name)
    return {**out, **comparison, 'unit': output_unit, 'source_unit': unit,
            'endpoint_key': endpoint_key, 'definition_warning': warning,
            'prediction_window_h': endpoint.get('window_h'),
            'scientific_status': 'unknown', 'candidate_source_audit_reported_separately': True}


def curve_comparison(document, prediction, *, scenario):
    """Explicitly align native predictions; no BQL substitution or hidden deletion.

    Log errors use qualified positive observations with coverage reported. Window
    integrals require every internal point quantified; leading/trailing zeros are
    outside that explicitly restricted positive-observation window. Individual R
    curves are separate within-study records, never independent study samples.
    """
    schema = document.get('schema_version', '')
    if schema not in {'pbpk-independent-concentration-curves-sealed-v1', 'pbpk-independent-aggregate-curves-sealed-v1'}:
        return []
    allowed = {v.get('protocol_fields', {}).get('case_id') for v in scenario.get('curve_views', [])}
    if document.get('case_id') not in allowed:
        raise ValueError('Curve document does not match a frozen view')
    declared_unit = str(document.get('concentration_unit', document.get('unit', ''))).lower()
    if declared_unit != 'ng/ml' or document.get('time_unit', 'h') != 'h':
        return [{'state': 'unavailable', 'reason': 'source_curve_units_not_explicitly_ng_ml_and_hours'}]
    if document.get('compound') != scenario['compound']:
        raise ValueError('Curve source compound does not match frozen scenario')
    if prediction.get('matrix') != 'plasma' or prediction.get('analyte') != 'parent' or prediction.get('binding') != 'total':
        return [{'state': 'unavailable', 'reason': 'native_curve_analyte_matrix_not_matched'}]
    if schema.endswith('aggregate-curves-sealed-v1'):
        series = [{'subject_id': 'published_arithmetic_mean', 'observations': [
            {'nominal_time_h': r.get('time_h'), 'concentration_ng_ml': r.get('arithmetic_mean_ng_ml'),
             'observation_status': 'source_zero_censored_mean' if r.get('published_arithmetic_mean_is_zero') else 'quantified'}
            for r in document.get('records', [])]}]
    else:
        series = [r for r in document.get('records', []) if r.get('treatment') == 'R']
        if len({r.get('subject_id') for r in series}) != len(series):
            raise ValueError('Duplicate R participant curves within one clinical cohort')
    result = []
    for record in series:
        points, gaps, prediction_failures = [], [], []
        original = record.get('observations', [])
        for i, row in enumerate(original):
            t, obs = row.get('nominal_time_h'), row.get('concentration_ng_ml')
            why = ('unquantified_or_source_missing' if row.get('observation_status') != 'quantified' else
                   'invalid_observation_time' if not finite(t) or t < 0 else
                   'nonpositive_or_missing_observation' if not finite(obs) or obs <= 0 else None)
            if why:
                gaps.append({'index': i, 'time_h': t, 'reason': why})
                continue
            # This membership depends only on observed eligibility, identically
            # for every candidate. A bad prediction cannot shorten the target.
            point = {'index': i, 'time_h': t, 'observed': obs, 'predicted': None}
            try:
                predicted = interpolate(prediction['time_h'], prediction['concentration_ng_ml'], t)
                if predicted <= 0:
                    raise ValueError('nonpositive_prediction')
                point['predicted'] = predicted
            except (ValueError, KeyError) as exc:
                prediction_failures.append({'index': i, 'time_h': t, 'reason': str(exc)})
            points.append(point)
        if any(b['time_h'] <= a['time_h'] for a, b in zip(points, points[1:])):
            result.append({'state': 'unavailable', 'subject_id': record.get('subject_id'),
                           'reason': 'qualified_observation_times_duplicate_or_not_increasing',
                           'source_point_count': len(original), 'qualified_point_count': len(points),
                           'excluded_points': gaps, 'curve': {'state': 'unavailable'},
                           'window_metrics': None, 'scientific_status': 'unknown'})
            continue
        errors = [metrics.endpoint_error(r['predicted'], r['observed'], endpoint='concentration') for r in points]
        summary = metrics.summarize_errors(errors)
        valid_errors = [r for r in errors if r['state'] == 'valid']
        summary['log_rmse'] = math.sqrt(math.fsum(r['signed_log_error'] ** 2 for r in valid_errors) / len(valid_errors)) if valid_errors else None
        summary['scope'] = 'All source-qualified positive observations; source gaps remain excluded and counted.'
        complete = bool(points) and not prediction_failures
        row = {'state': 'computed' if complete else 'unavailable', 'subject_id': record.get('subject_id'),
               'period': record.get('period'), 'treatment': record.get('treatment', 'aggregate'),
               'source_point_count': len(original), 'qualified_point_count': len(points), 'excluded_points': gaps,
               'qualified_point_fraction': len(points) / len(original) if original else None,
               'prediction_failures': prediction_failures,
               'prediction_coverage_on_qualified_observations': len(valid_errors) / len(points) if points else None,
               'qualified_observed_time_h': [p['time_h'] for p in points],
               'curve': summary if complete else {'state': 'unavailable', 'reason': 'not_all_fixed_qualified_points_predicted'},
               'partial_prediction_diagnostic': summary if not complete else None,
               'window_metrics': None, 'window_reason': 'insufficient_positive_points',
               'curve_kind': 'arithmetic_mean' if schema.endswith('aggregate-curves-sealed-v1') else 'individual_R',
               'comparison_stratum': 'reference_diagnostic', 'scientific_status': 'unknown',
               'time_alignment': 'linear interpolation of frozen native curve; no extrapolation',
               'subject_matching': 'One reference person, not fitted or personalized to observed participants.',
               'publication_cluster_id': scenario.get('publication_cluster_id')}
        if prediction_failures:
            row['window_reason'] = 'not_all_fixed_qualified_points_predicted_no_model_specific_window'
        elif len(points) >= 2:
            if points[-1]['index'] - points[0]['index'] + 1 != len(points):
                row['window_reason'] = 'internal_missing_or_censored_points_no_integral_across_gaps'
            else:
                times = [p['time_h'] for p in points]
                window = metrics.curve_metrics(times, [p['observed'] for p in points], times, [p['predicted'] for p in points])
                row.update(window_metrics=window, window_reason=window.get('reason'),
                           window_definition='Only the explicitly bounded, contiguous, quantified positive observation span; not AUClast/AUCinf.')
        result.append(row)
    return result


def summarize_candidate_curves(curves):
    """Equal R participants within a cohort, not pooled concentration points."""
    complete = [r for r in curves if r.get('state') == 'computed' and r.get('curve', {}).get('state') == 'computed']
    if not curves or len(complete) != len(curves):
        return {'state': 'unavailable', 'reason': 'not_all_fixed_subject_curves_evaluable' if curves else 'no_complete_source_curves',
                'source_series': len(curves), 'complete_series': len(complete)}
    absolute = sum(r['curve']['mean_absolute_log_error'] for r in curves) / len(curves)
    signed = sum(r['curve']['mean_signed_log_error'] for r in curves) / len(curves)
    return {'state': 'computed', 'source_series': len(curves), 'complete_series': len(curves),
            'mean_absolute_log_error': absolute, 'mean_signed_log_error': signed,
            'gmfe': math.exp(absolute), 'afe': math.exp(signed),
            'log_rmse': math.sqrt(sum(r['curve']['log_rmse'] ** 2 for r in curves) / len(curves)),
            'weighting': 'equal source R-participant series within clinical cohort, or one published arithmetic-mean series; not independent studies',
            'scientific_status': 'unknown'}


def candidate_comparison(scenario, audit, native, documents, *, source_assessment=None):
    strict = {'candidate_id': audit['candidate_id'], 'track': 'direct', 'execution_status': 'not_run',
              'scientific_status': 'unknown', 'endpoints': {}, 'reason': 'candidate_not_scientifically_validated'}
    diagnostic = {**strict, 'track': 'reference_diagnostic', 'endpoints': {}}
    source_assessment = source_assessment or {'status': 'unknown', 'reason': 'candidate_specific_source_audit_not_supplied'}
    scoped_pass = source_assessment.get('status') == 'scoped_source_audit_pass'
    result = {'scenario_id': scenario['scenario_id'], 'case_id': scenario['case_id'], 'compound': scenario['compound'],
              'publication_cluster_id': scenario.get('publication_cluster_id'), 'candidate_id': audit['candidate_id'],
              'family_id': audit.get('family_id'), 'identity_match': audit.get('identity_match'),
              'previous_independence_status': audit.get('independence_status'),
              'source_audit': source_assessment, 'scoped_source_audit_passed': scoped_pass,
              'absolute_source_independence_established': False,
              'strict': strict, 'reference_diagnostic': diagnostic, 'scalars': [], 'curves': []}
    if audit.get('identity_match') is not True:
        strict.update(scientific_status='incompatible', reason='direct_parent_identity_scope_exclusion_only')
        diagnostic['reason'] = 'cross_drug_reference_value_unassessed_not_a_negative'
        return result
    if native is None:
        raise ValueError('Same-parent candidate missing from verified native results')
    status = native['result']['status']
    result['native_status'] = status
    if not native['prediction']:
        execution = 'not_run' if status == 'blocked' else 'failed'
        for item in (strict, diagnostic):
            item.update(execution_status=execution, reason='; '.join(native['result'].get('reasons', [])) or status)
        return result
    prediction = native['prediction']
    for item in (strict, diagnostic):
        item.update(execution_status='succeeded', reason='reference_person_group_estimand_mismatch_prevents_strict_validation'
                    if scoped_pass else 'reference_person_approximation_and_candidate_specific_source_audit_unconfirmed')
    for doc in documents:
        for row in outcome_rows(doc, scenario):
            comparison = scalar_comparison(row, scenario, prediction)
            result['scalars'].append(comparison)
            if comparison['state'] == 'valid':
                key = comparison['endpoint_key']
                if key in diagnostic['endpoints']:
                    raise ValueError('Duplicate scalar comparison across sealed sources')
                diagnostic['endpoints'][key] = {k: comparison[k] for k in ('predicted', 'observed')}
        result['curves'].extend(curve_comparison(doc, prediction, scenario=scenario))
    aggregate = [r for r in result['curves'] if r.get('curve_kind') == 'arithmetic_mean' and (r.get('window_metrics') or {}).get('state') == 'computed']
    if len(aggregate) > 1:
        raise ValueError('Multiple aggregate curve targets require explicit endpoint separation')
    if aggregate:
        wm = aggregate[0]['window_metrics']
        for key, source in [('curve_observed_window_AUC', 'observed_window_auc'), ('curve_sampled_peak', 'sampled_peak')]:
            error = wm[source]
            diagnostic['endpoints'][key] = {k: error[k] for k in ('predicted', 'observed')}
    result['curve_summary'] = summarize_candidate_curves(result['curves'])
    # Even excellent numeric agreement cannot establish population/source validity.
    return result


def aggregate_method_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['stratum'], row['endpoint'])].append(row)
    result = []
    for (method, stratum, endpoint), group in sorted(grouped.items()):
        valid = [r for r in group if r['top1_absolute_log_error'] is not None]
        study = defaultdict(list)
        for r in valid:
            study[(r['compound'], r['publication_cluster_id'])].append(r)
        compounds = defaultdict(list)
        for (compound, _), cases in study.items():
            compounds[compound].append(sum(c['top1_absolute_log_error'] for c in cases) / len(cases))
        vals = [sum(studies) / len(studies) for studies in compounds.values()]
        def hierarchical(field):
            by_study = defaultdict(list)
            for r in group:
                if r.get(field) is not None:
                    by_study[(r['compound'], r['publication_cluster_id'])].append(r[field])
            by_compound = defaultdict(list)
            for (drug, _), values in by_study.items():
                by_compound[drug].append(sum(values) / len(values))
            averages = [sum(v) / len(v) for v in by_compound.values()]
            return sum(averages) / len(averages) if averages else None
        signed = hierarchical('top1_signed_log_error')
        result.append({'method': method, 'stratum': stratum, 'endpoint': endpoint,
                       'scenario_count': len(group), 'top1_numerically_covered_scenarios': len(valid),
                       'top1_mean_evaluable_probability': sum(r['top1_evaluable_probability'] for r in group) / len(group),
                       'top1_gmfe_conditioned_on_known_selection': math.exp(sum(vals) / len(vals)) if vals else None,
                       'top1_afe_conditioned_on_known_selection': math.exp(signed) if signed is not None else None,
                       'top1_fraction_within_1_5_fold_conditioned_on_known_selection': hierarchical('top1_fraction_within_1_5_fold'),
                       'top1_fraction_within_2_fold_conditioned_on_known_selection': hierarchical('top1_fraction_within_2_fold'),
                       'covered_compounds': len(compounds), 'covered_publication_compound_groups': len(study),
                       'weighting': 'equal cases within publication/compound, equal publications within compound, equal covered compounds',
                       'measured_pool_choice_scenarios': sum(r['measured_pool_candidate_count'] >= 2 for r in group),
                       'mean_measured_pool_regret_conditioned_on_known_choice': hierarchical('measured_pool_regret'),
                       'scientific_status': 'unknown', 'exact_recall': None,
                       'inference': {'state': 'unavailable',
                                     'reason': 'No exact-population scientifically validated labels; diagnostic coverage differs across rankings. Scientific F1 is unavailable; paired numerical inference is reported separately.'}})
    return result


def audit_declared_policy(comparisons, manifest=None):
    """Audit the existing engineering gate without manufacturing clinical labels."""
    counts = {'same_parent': Counter(), 'other_parent': Counter()}
    cross = defaultdict(Counter)
    signals = []
    omitted = 0
    for row in comparisons:
        decision = row.get('existing_policy_decision')
        if decision is None:
            omitted += 1
            continue
        action = decision['declared_use']
        same = row.get('identity_match') is True
        counts['same_parent' if same else 'other_parent'][action] += 1
        if not same:
            continue
        cross[action][row.get('native_status', 'unavailable')] += 1
        if action == 'direct_reuse':
            continue
        for endpoint, pair in row['reference_diagnostic']['endpoints'].items():
            error = metrics.endpoint_error(pair.get('predicted'), pair.get('observed'), endpoint=endpoint)
            if error['state'] != 'valid' or error['absolute_log_error'] > math.log(2.):
                continue
            signals.append({'scenario_id': row['scenario_id'], 'case_id': row['case_id'],
                            'candidate_id': row['candidate_id'], 'compound': row['compound'],
                            'family_id': row.get('family_id'), 'endpoint': endpoint,
                            'existing_action': action, 'original_policy_class': decision.get('original_policy_class'),
                            'policy_reasons': decision.get('reasons', []),
                            'fold_error': error['fold_error'], 'absolute_log_error': error['absolute_log_error'],
                            'within_1_5_fold': error['absolute_log_error'] <= math.log(1.5), 'within_2_fold': True,
                            'comparison_stratum': 'reference_diagnostic',
                            'scientific_false_negative_established': False,
                            'interpretation': 'Potential engineering-policy overrestriction clue; numerical agreement does not validate direct reuse.'})
    return {'state': 'computed' if not omitted else 'unavailable_for_explicitly_omitted_synthetic_declarations',
            'policy': 'Existing deterministic engineering/use policy, separately frozen from all 12 ordering methods.',
            'declared_pairs': sum(sum(v.values()) for v in counts.values()), 'missing_declaration_pairs': omitted,
            'action_counts_by_parent_scope': {k: dict(v) for k, v in counts.items()},
            'same_parent_action_by_native_status': {k: dict(v) for k, v in sorted(cross.items())},
            'not_direct_cleared_but_numerically_within_threshold': signals,
            'signal_candidate_pairs': len({(r['scenario_id'], r['candidate_id']) for r in signals}),
            'signal_endpoint_rows': len(signals),
            'scientific_false_negative_rate': None, 'scientific_F1': None,
            'representation_limits': manifest.get('known_representation_limits', []) if manifest else [],
            'interpretation': 'Reference-only and unknown are retained as their original actions. Threshold signals prompt review; they are not false-negative truth or applicability labels.'}


def score_run(run_dir, *, project_root=ROOT, expected_methods=None, allow_missing_declared_tiers=False):
    bundle = verify_barrier(run_dir, project_root=project_root, expected_methods=expected_methods,
                            allow_missing_declared_tiers=allow_missing_declared_tiers)
    from recommendation_execution_source_audit import audit_candidate_source
    f, run = bundle['files'], bundle['run']
    comparisons, selections, csv_rows, document_cache, inventories, curve_selections = [], [], [], {}, [], []
    by_scenario = {s['scenario_id']: s for s in bundle['scenarios']}
    for qi, sid in enumerate(bundle['qids']):
        scenario = by_scenario[sid]
        documents = []
        for ref in scenario.get('sealed_references', []):
            key = (ref['path'], ref['sha256'])
            if key not in document_cache:
                document_cache[key] = f.outcome(ref)
            documents.append(document_cache[key])
        outcomes = []
        for cid in bundle['cids']:
            audit = bundle['audits'][(sid, cid)]
            source_assessment = (audit_candidate_source(scenario, audit, project_root=project_root)
                                 if audit.get('identity_match') is True else
                                 {'status': 'unknown', 'reason': 'cross_parent_candidate_source_review_not_performed'})
            comparison = candidate_comparison(scenario, audit, bundle['native'].get((sid, cid)), documents,
                                              source_assessment=source_assessment)
            comparison['existing_policy_decision'] = bundle['declared_tiers'].get((sid, cid))
            outcomes.append(comparison)
        comparisons.extend(outcomes)
        observed_rows = [r for doc in documents for r in outcome_rows(doc, scenario)]
        requested = {endpoint_key(r['endpoint']) for r in observed_rows}
        if any(d.get('schema_version') == 'pbpk-independent-aggregate-curves-sealed-v1' for d in documents):
            requested.update(['curve_observed_window_AUC', 'curve_sampled_peak'])
        requested = sorted(requested)
        inventories.append({'scenario_id': sid, 'case_id': scenario['case_id'],
                            'scalar_endpoint_count': len(observed_rows), 'requested_numerical_endpoints': requested,
                            'no_scalar_source_packet': not observed_rows,
                            'source_exclusions': [r['endpoint'] for r in observed_rows if r.get('evaluable_from_source') is False],
                            'same_parent_candidates': sum(r['identity_match'] is True for r in outcomes),
                            'reference_numerically_evaluable_candidates': sum(bool(r['reference_diagnostic']['endpoints']) for r in outcomes)})
        for method, scores in bundle['matrices'].items():
            ranking = dict(zip(bundle['cids'], (float(v) for v in scores[qi])))
            for stratum, track in [('strict', 'direct'), ('reference_diagnostic', 'reference_diagnostic')]:
                evaluated = metrics.evaluate_selection(bundle['cids'], [r[stratum] for r in outcomes], ranking,
                                                       track=track, endpoint_names=requested)
                evaluated.update(method=method, scenario_id=sid, case_id=scenario['case_id'], compound=scenario['compound'],
                                 stratum=stratum, publication_cluster_id=scenario.get('publication_cluster_id'))
                # Avoid repeating 280 inclusion weights three times in each JSON result;
                # the retained top-1 weights still reproduce conditional endpoint errors.
                for k, entry in evaluated['by_k'].items():
                    if k != '1':
                        entry.pop('inclusion_probabilities', None)
                if stratum == 'reference_diagnostic':
                    curve_valid = {r['candidate_id']: r['curve_summary'] for r in outcomes
                                   if r.get('curve_summary', {}).get('state') == 'computed'}
                    weights = evaluated['by_k']['1']['inclusion_probabilities']
                    cc = sum(weights[c] for c in curve_valid)
                    error = sum(weights[c] * r['mean_absolute_log_error'] for c, r in curve_valid.items()) / cc if cc else None
                    best = min((r['mean_absolute_log_error'] for r in curve_valid.values()), default=None)
                    curve_selections.append({'method': method, 'scenario_id': sid, 'stratum': stratum,
                        'compound': scenario['compound'], 'publication_cluster_id': scenario.get('publication_cluster_id'),
                        'top1_evaluable_probability': cc, 'measured_pool_candidate_count': len(curve_valid),
                        'top1_conditional_mean_absolute_log_error': error,
                        'top1_conditional_log_rmse': math.sqrt(sum(weights[c] * r['log_rmse'] ** 2 for c, r in curve_valid.items()) / cc) if cc else None,
                        'best_measured_mean_absolute_log_error': best,
                        'measured_pool_regret': max(0., error - best) if error is not None and len(curve_valid) >= 2 else None,
                        'scientific_status': 'unknown', 'exact_full_pool_regret': None,
                        'reason': None if cc else 'selected_candidate_complete_curve_unavailable'})
                for endpoint, e in evaluated['endpoints'].items():
                    weights = evaluated['by_k']['1']['inclusion_probabilities']
                    valid = {}
                    for r in outcomes:
                        pair = r[stratum]['endpoints'].get(endpoint)
                        if pair:
                            error = metrics.endpoint_error(pair.get('predicted'), pair.get('observed'), endpoint=endpoint)
                            if error['state'] == 'valid':
                                valid[r['candidate_id']] = error
                    coverage = e['top1_evaluable_probability']
                    signed = sum(weights[c] * error['signed_log_error'] for c, error in valid.items()) / coverage if coverage else None
                    fractions = {threshold: sum(weights[c] for c, error in valid.items() if error['absolute_log_error'] <= math.log(threshold)) / coverage
                                 if coverage else None for threshold in (1.5, 2.0)}
                    family_ids = {r.get('family_id') for r in outcomes if r['candidate_id'] in valid and r.get('family_id')}
                    e.update(measured_pool_catalog_family_count=len(family_ids),
                             independent_model_sources_confirmed=False,
                             top1_signed_log_error_conditioned_on_known_selection=signed,
                             numerical_candidates_within_1_5_fold=sum(v['absolute_log_error'] <= math.log(1.5) for v in valid.values()),
                             numerical_candidates_within_2_fold=sum(v['absolute_log_error'] <= math.log(2.) for v in valid.values()),
                             numerical_thresholds_are_not_scientific_labels=True)
                    if stratum == 'reference_diagnostic':
                        e['numerical_omission_audit'] = statistics.numerical_miss_audit(
                            bundle['cids'], [r['candidate_id'] for r in outcomes if r['identity_match'] is True],
                            ranking, {cid: error['absolute_log_error'] for cid, error in valid.items()})
                    csv_rows.append({'method': method, 'stratum': stratum, 'scenario_id': sid,
                                     'compound': scenario['compound'], 'publication_cluster_id': scenario.get('publication_cluster_id'),
                                     'endpoint': endpoint, 'top1_evaluable_probability': e['top1_evaluable_probability'],
                                     'top1_absolute_log_error': e['selected_known_conditional_absolute_log_error'],
                                     'top1_signed_log_error': signed,
                                     'top1_fraction_within_1_5_fold': fractions[1.5], 'top1_fraction_within_2_fold': fractions[2.],
                                     'measured_pool_candidate_count': e['valid_candidate_count'],
                                     'measured_pool_catalog_family_count': len(family_ids),
                                     'measured_pool_regret': e['known_pool_conditional_regret'] if e['valid_candidate_count'] >= 2 else None,
                                     'exact_full_pool_regret': e['exact_full_pool_regret'], 'exact_recall_at_1': evaluated['by_k']['1']['exact_recall'],
                                     'scientific_status': 'unknown'})
                    if e['valid_candidate_count'] < 2:
                        e['known_pool_conditional_regret'] = None
                        e['choice_comparison_reason'] = 'fewer_than_two_numerically_comparable_candidates'
                selections.append(evaluated)
    status_counts = Counter(n['result']['status'] for n in bundle['native'].values())
    scalar_counts = Counter(r.get('reason') or 'valid' for c in comparisons for r in c['scalars'])
    summary = {'scenarios': len(bundle['qids']), 'compounds': len({s['compound'] for s in bundle['scenarios']}),
               'candidates': len(bundle['cids']), 'methods': len(bundle['matrices']),
               'ranking_pairs_per_method': len(bundle['qids']) * len(bundle['cids']),
               'all_planned_native_jobs_accounted_for': len(bundle['native']), 'native_status_counts': dict(status_counts),
               'all_candidate_scope_rows': len(comparisons), 'opened_sealed_files': len(f.opened_outcomes),
               'numerically_evaluable_reference_pairs': sum(bool(c['reference_diagnostic']['endpoints']) for c in comparisons),
               'direct_validated_pairs': sum(c['strict']['scientific_status'] == 'validated' for c in comparisons),
               'scoped_source_audit_pass_pairs': sum(c['scoped_source_audit_passed'] for c in comparisons),
               'source_audit_status_counts': dict(Counter(c['source_audit']['status'] for c in comparisons if c['identity_match'])),
               'absolute_source_independence_claimed': False,
               'scalar_dispositions': dict(scalar_counts),
               'exact_recall': None, 'missed_scientifically_suitable_candidates': None,
               'missed_recommendation_gap': 'Every same-parent candidate is executed or explicitly blocked; full scientific recall remains unavailable until source/population validity and acceptance are established.',
               'complete_execution_accounting': True, 'complete_scientific_validation': False}
    report = {'schema_version': VERSION, 'summary': summary,
              'locks': {'protocol_sha256': f.sha(run / 'protocol.json'), 'ranking_lock_sha256': bundle['ranking_lock_sha256'],
                        'predictions_lock_sha256': bundle['predictions_lock_sha256'], 'sealed_sources_read_after_all_locks': f.opened_outcomes},
              'definitions': {'strict': 'Only exact target/source-qualified predictions; none currently established.',
                              'reference_diagnostic': 'Frozen reference-individual numerical comparison; does not assert a fitted adaptation or scientific applicability.',
                              'scalar_Cmax': 'Reference-person sampled peak versus reported group NCA peak summary; diagnostic estimand mismatch is retained.',
                              'curve_AUC': 'Trapezoid on common, explicitly bounded, contiguous positive observed times; never AUClast/AUCinf.',
                              'thresholds': '1.5/2 fold are exploratory numerical screens, never automatic applicability labels.',
                              'independence': 'Previously curated clinical material. Exact-bound candidate reviews may establish no target-study overlap within recorded audited materials; no undisclosed participant overlap or absolute independence guarantee.'},
              'method_endpoint_summary': aggregate_method_rows(csv_rows), 'selection': selections,
              'paired_comparisons': statistics.paired_method_statistics(csv_rows, bundle['matrices'].keys())
                  if 'bm25_text' in bundle['matrices'] else {'state': 'unavailable', 'reason': 'registered_baseline_absent'},
              'curve_selection': curve_selections,
              'declared_policy_audit': audit_declared_policy(comparisons, bundle['declared_tiers_manifest']),
              'observation_inventory': inventories,
              'candidate_results': comparisons,
              'limitations': ['No scientific ground truth is created from execution success or reference-subject accuracy.',
                              'Raw ordering methods lack frozen applicability probabilities/decisions; F1, Brier and ECE are not inferred.',
                              'All failures, unknowns, direct identity exclusions and unmeasured candidates remain in denominators.',
                              'Same-family versions are not independent models; multiple cases/timepoints from one study are not independent studies.']}
    # Recheck public locks/inputs after reading observations, before committing outputs.
    verify_barrier(run, project_root=project_root, expected_methods=expected_methods,
                   allow_missing_declared_tiers=allow_missing_declared_tiers)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fields = list(csv_rows[0]) if csv_rows else ['method', 'stratum', 'scenario_id', 'endpoint']
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
    for name, payload in [('results.json', text), ('results.csv', buffer.getvalue())]:
        p = run / name
        if p.exists() and p.read_text() != payload:
            raise ValueError('Existing score result differs; retain original and investigate: ' + name)
        if not p.exists():
            with p.open('x') as handle:
                handle.write(payload)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir)['summary'], ensure_ascii=False, indent=2))
