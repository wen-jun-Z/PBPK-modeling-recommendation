"""Read-only, pre-acquisition inputs for a new sampling-policy experiment.

``load_policy_inputs(allow_model_replay=True)`` returns ordinary Python data in
memory. It never fits parameters, writes files, reveals a fifth development
concentration, reads a target concentration, or reads historical policy scores.
The optional replay evaluates locked four-point parameters at four visible
times, plus committee members whose common target-time predictions were not
cached. Target *times and conditions* are permitted metadata, not outcomes.

The source audit is used solely for clinical identity/grouping, never predictors.
Source-document SHA verification does not imply independent clinical validation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'preacquisition-policy-inputs-1'
STABILITY = 'experiments/blueprint_design/sampling_stability_20261009'
SOURCE_AUDIT = 'experiments/blueprint_design/reviewer_revision_20261009/sources/source_pairing.json'
FEATURE_NAMES = (
    'log_time_over_infusion', 'nearest_visible_log_time_distance',
    'mean_log_concentration_over_visible_max', 'concentration_log_variance',
    'linear_explained_goal_variance', 'nonlinear_explained_goal_variance',
    'local_c_optimal_fractional_reduction', 'after_infusion',
)


class _Reader:
    """Follow explicit whitelisted references, not arbitrary provenance links."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.documents = {}
        self.manifest = {}

    def pin(self, source, role):
        if isinstance(source, str):
            source = {'path': source}
        path = (self.root / source['path']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('Missing or out-of-project source: ' + str(path))
        key = str(path.relative_to(self.root))
        if key not in self.manifest:
            raw = path.read_bytes()
            if len(raw) > 64 * 1024**2:
                raise ValueError('Unexpectedly large input: ' + key)
            digest = hashlib.sha256(raw).hexdigest()
            self.manifest[key] = {'path': key, 'sha256': digest, 'roles': []}
        record = self.manifest[key]
        if source.get('sha256') and source['sha256'] != record['sha256']:
            raise ValueError('Source hash changed: ' + key)
        if role not in record['roles']:
            record['roles'].append(role)
        return path

    def read(self, source, role):
        path = self.pin(source, role)
        if path not in self.documents:
            self.documents[path] = json.loads(path.read_text())
        return self.documents[path]

    def records(self):
        return [dict(x, roles=sorted(x['roles'])) for _, x in sorted(self.manifest.items())]


def _array(value, name, ndim=None, positive=False):
    a = np.asarray(value, dtype=float)
    if (ndim is not None and a.ndim != ndim) or not np.isfinite(a).all():
        raise ValueError('Invalid finite array: ' + name)
    if positive and np.any(a <= 0):
        raise ValueError('Nonpositive values: ' + name)
    return a


def _observed_four(training):
    if training.get('time_unit') != 'min' or training.get('unit') != 'umol/l':
        raise ValueError('Visible concentration/time units differ from the frozen contract')
    observations = training['observations']
    if len(observations) != 4:
        raise ValueError('Only four pre-acquisition development observations are allowed')
    t = _array([o['time_min'] / 60 for o in observations], 'visible times', 1, True)
    mw = float(training['molecular_weight_g_mol'])
    if not np.isfinite(mw) or mw <= 0:
        raise ValueError('Invalid molecular weight')
    c = _array([o['concentration_umol_per_l'] * mw / 1000 for o in observations],
               'visible concentrations in mg/L', 1, True)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Four visible times must be ordered and unique')
    return t, c


def _study_groups(audit, cases):
    """Conservative connected components across either clinical source side.

    Publication identities are intentionally conservative proxies for shared
    participants. Reusing a publication connects cases even if its arms could
    be separate. Unverified nonoverlap never becomes an independence claim.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    case_map = {c['case_id']: c for c in cases}
    edges, side_ids = [], {}
    for cid, case in case_map.items():
        join('case:' + cid, 'compound:' + case['compound_id'])
    for row in audit['records']:
        cid = row['case_id']
        if cid not in case_map:
            continue
        fields = row['fields']
        value = lambda key: (fields.get(key) or {}).get('value')
        identities = []
        for field in ('doi', 'pmid', 'pmcid', 'registry'):
            v = value(field)
            if v:
                identities.append((field + ':' + re.sub(r'\s+', ' ', str(v).strip().lower()), field))
        sid = value('study_id')
        # F A/F B and similar bare repository letters are not clinical IDs.
        if sid and len(re.sub(r'[^a-z0-9]', '', str(sid).lower())) > 3:
            identities.append(('study:' + re.sub(r'\s+', ' ', str(sid).strip().lower()), 'study_id'))
        if not identities:
            identities.append(('unresolved:' + cid + ':' + row['role'], 'unresolved'))
        side_ids[cid + '|' + row['role']] = [key for key, _ in identities]
        for identity, field in identities:
            join('case:' + cid, identity)
            edges.append({'case_id': cid, 'role': row['role'], 'identity': identity,
                          'identity_field': field,
                          'evidence': deepcopy((fields.get(field) or {}).get('evidence', []))})
    components = defaultdict(list)
    for cid in sorted(case_map):
        components[find('case:' + cid)].append(cid)
    groups, case_to_group = [], {}
    for members in sorted(components.values(), key=lambda v: v[0]):
        group = 'drugstudy:' + hashlib.sha256('|'.join(members).encode()).hexdigest()[:16]
        groups.append({'group_id': group, 'case_ids': members,
                       'drugs': [case_map[c]['drug'] for c in members]})
        case_to_group.update({cid: group for cid in members})
    pairs = {p['case_id']: p for p in audit['pairs']}
    return {
        'groups': groups, 'case_to_group': case_to_group, 'side_identities': side_ids,
        'edges': edges, 'old_target_clusters': {c: case_map[c]['cluster_id'] for c in case_map},
        'audited_target_clusters': {c: pairs[c]['target_source_cluster'] for c in case_map},
        'limitations': [
            'Uses development AND target source identities, and compound identity, without outcomes.',
            'Publication-level links conservatively merge studies/arms from the same paper.',
            'Different identifiers do not prove participant nonoverlap; unknown relations remain unknown.',
            'Registry strings are used only when established by the source audit; repository F A/F B is not a registry.',
        ],
    }


def _simulate(reader, physiology, request, times, parameters):
    """Pure in-memory evaluation; never calls the native exporter or a fitter."""
    source_root = str(reader.root / 'src')
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from pbpk_kb.blueprint_policy import blood_partition_calibration
    for name in ('blood_partition_calibration', 'permeability_model', 'perfusion_model',
                 'perfusion_calibration'):
        reader.pin('src/pbpk_kb/blueprint_policy/' + name + '.py', 'model_only_replay_code')
    result = blood_partition_calibration.simulate(physiology, request, times, parameters)
    concentration = _array(result['concentration_mg_l'], 'replayed concentrations', 1, True)
    if len(concentration) != len(times):
        raise ValueError('Replay returned a different number of concentrations')
    return concentration


def _common_cmax(reader, feature, decision, case, protocol, allow_replay, replay_log):
    """Retrieve exact cached model samples under the current sampling contract.

    The old ten-drug goal_values are NOT copied: their schedule can differ.
    No interpolation of old sparse model diagnostics is allowed. If exact
    samples are absent, explicit replay uses the fixed model at metadata times.
    """
    contract_ref = feature.get('feature_rebuild_receipt', {}).get('target_contract',
                                                                  decision['target_contract'])
    contract = reader.read(contract_ref, 'target_metadata_only_contract')
    if (contract.get('source_kind') != 'target_protocol_metadata_only'
            or contract.get('target_observation_values_used') is not False
            or contract.get('version') != 'sampling-expansion-observed-curve-nca-1'):
        raise ValueError('Current common metadata-only endpoint contract is required')
    wanted = _array(contract['sampling_time_h'], 'common target times', 1)
    values, origins, target_inputs = [], [], {}
    for member_index, member in enumerate(decision['members']):
        diagnostics = member.get('goal_diagnostics', {}).get('diagnostics', {})
        old_t = _array(diagnostics.get('sampled_time_h', []), 'cached model times', 1)
        old_c = _array(diagnostics.get('sampled_concentration_ng_ml', []), 'cached model concentrations', 1)
        indices = [np.flatnonzero(np.isclose(old_t, t, rtol=0, atol=1e-10)) for t in wanted]
        if len(old_t) == len(old_c) and len(old_t) and all(len(ix) == 1 for ix in indices):
            values.append(float(max(old_c[ix[0]] for ix in indices)))
            origins.append('exact_cached_model_samples_at_common_times')
            continue
        if not allow_replay:
            values.append(None)
            origins.append('common_model_samples_not_cached')
            continue
        formula = member['formula']
        if formula not in target_inputs:
            if 'document' not in protocol:
                protocol['document'] = reader.read(protocol['reference'], 'frozen_physiology_cache_index')
            p = protocol['document']
            request = deepcopy(reader.read(case['requests']['evidence']['target'], 'target_request_no_observations'))
            if any(k in request for k in ('observations', 'clinical_reference', 'curve')):
                raise ValueError('Target request must not contain observed concentrations')
            request['partition_method'] = formula
            phys_ref = p['physiology_cache'][case['case_id'] + '__target__' + formula]
            physiology = reader.read(phys_ref, 'target_unfitted_physiology')
            if request['dosing'] != contract['dosing'] or request['human'] != contract['human']:
                raise ValueError('Target replay conditions differ from the common contract')
            target_inputs[formula] = request, physiology
        request, physiology = target_inputs[formula]
        replay_log.append({'design_key': feature['design_key'], 'kind': 'committee_common_cmax',
                           'member_index': member_index, 'formula': formula,
                           'times_h': wanted.tolist(), 'target_observations_used': False})
        c = _simulate(reader, physiology, request, wanted, member['log_parameters'])
        values.append(float(c.max() * 1000))
        origins.append('model_only_replay_at_common_metadata_times')
    return values, origins, contract_ref


def load_policy_inputs(root=None, *, allow_model_replay=False, case_ids=None):
    """Return 60 menus, source manifests and both-side drug-study groups.

    No outcome labels are returned. ``case_ids`` limits numeric input loading,
    but grouping is always built from all 30 cases to avoid splitting a shared
    study merely because its other compound was omitted from a local subset.
    ``allow_model_replay`` defaults to False and must be an explicit bool.
    """
    if type(allow_model_replay) is not bool:
        raise TypeError('allow_model_replay must be an explicit boolean')
    reader = _Reader(root or ROOT)
    reader.pin(str(Path(__file__).resolve().relative_to(reader.root)), 'adapter_code')
    p = reader.read(STABILITY + '/protocol.json', 'frozen_protocol_metadata')
    features = reader.read(p['features'], 'frozen_preacquisition_features')['designs']
    audit = reader.read(SOURCE_AUDIT, 'clinical_identity_fields_only_not_predictors')
    case_map = {c['case_id']: c for c in p['cases']}
    allowed = set(case_map) if case_ids is None else set(case_ids)
    if not allowed <= case_map.keys():
        raise ValueError('Unknown requested case IDs')
    grouping = _study_groups(audit, p['cases'])
    replay_log, designs = [], []
    replay_protocol = {'reference': p['source_protocol']}
    for key, feature in sorted(features.items()):
        cid = feature['case_id']
        if cid not in allowed:
            continue
        case = case_map[cid]
        spec = p['designs'][key]
        decision_ref = feature.get('decision_source', feature.get('decision'))
        decision = reader.read(decision_ref, 'preacquisition_committee')
        if (decision.get('design_key') != key or decision.get('hidden_values_used') is not False
                or decision.get('target_values_used') is not False
                or feature.get('hidden_values_used') is not False):
            raise ValueError('Pre-acquisition isolation receipt differs: ' + key)
        if decision_ref != spec['source']:
            raise ValueError('Feature committee differs from locked protocol: ' + key)
        visible = reader.read(decision['visible_source'], 'only_four_visible_development_points')
        training, menu = visible['visible_training'], visible['design_only']
        t, observed = _observed_four(training)
        times = _array(decision['candidate_times_h'], 'candidate times', 1, True)
        ids = decision['candidate_ids']
        if (len(ids) != len(times) or len(set(ids)) != len(ids)
                or ids != feature['candidate_ids'] or ids != menu['candidate_ids']
                or not np.array_equal(times, feature['candidate_times_h'])
                or not np.array_equal(times, menu['candidate_times_h'])
                or not np.array_equal(t, menu['observed_times_h'])
                or np.any(np.diff(times) <= 0)):
            raise ValueError('Candidate/visible menu alignment failed: ' + key)
        baseline_ref = decision['baseline']
        baseline = reader.read(baseline_ref, 'four_point_baseline_fit_metadata')
        baseline_training = reader.read(baseline['training'], 'verify_four_point_baseline_training_only')
        bt, bc = _observed_four(baseline_training)
        if not np.array_equal(bt, t) or not np.allclose(bc, observed, rtol=1e-13, atol=0):
            raise ValueError('Baseline used different or additional observations: ' + key)
        formula = baseline['formula']
        fit_ref = next(f['fit'] for f in baseline['fits'] if f['formula'] == formula)
        fit = reader.read(fit_ref, 'four_point_fit_summary_and_locked_parameters')
        if fit.get('target_observations_used') is not False or fit.get('visible_points') != 4:
            raise ValueError('Baseline fit isolation failed: ' + key)
        x = _array(fit['best']['log_parameters'], 'locked parameters', 1)
        lower = _array(fit['parameter_lower'], 'parameter lower', 1)
        upper = _array(fit['parameter_upper'], 'parameter upper', 1)
        if x.shape != (4,) or lower.shape != x.shape or np.any(upper <= lower):
            raise ValueError('Unexpected parameter axes')
        prediction, residual = None, None
        if allow_model_replay:
            physiology = reader.read(fit['development_physiology'], 'development_unfitted_physiology')
            request = fit['request']
            if request['dosing'] != training['dosing']:
                raise ValueError('Baseline replay dosing differs from visible training')
            replay_log.append({'design_key': key, 'kind': 'baseline_visible', 'times_h': t.tolist(),
                               'visible_points': 4, 'target_observations_used': False})
            pred = _simulate(reader, physiology, request, t, x)
            prediction = pred.tolist()
            residual = (np.log(observed) - np.log(pred)).tolist()
            if not np.isclose(np.sqrt(np.mean(np.square(residual))),
                              fit['best']['development_ln_rmse'], rtol=2e-7, atol=1e-9):
                raise ValueError('Locked baseline replay does not reproduce four-point fit RMSE: ' + key)
        cmax, cmax_origins, contract_ref = _common_cmax(
            reader, feature, decision, case, replay_protocol, allow_model_replay, replay_log)
        committee_c = _array(decision['concentrations_mg_l'], 'model committee', 2, True)
        if committee_c.shape != (len(decision['members']), len(ids)):
            raise ValueError('Committee/member/candidate alignment differs')
        goals = {}
        for goal, info in feature['goals'].items():
            item = {'status': info['status'], 'X': None,
                    'known_time_choice': info.get('choices', {}).get('time_farthest')}
            if info['status'] == 'selected':
                matrix = _array(info['X'], 'original eight features', 2)
                if matrix.shape != (len(ids), len(FEATURE_NAMES)):
                    raise ValueError('Original feature dimensions differ')
                item['X'] = matrix.tolist()
            goals[goal] = item
        logt, logc = np.log(t), np.log(observed)
        slope = float(np.dot(logt - logt.mean(), logc - logc.mean()) / np.sum((logt - logt.mean())**2))
        designs.append({
            'design_key': key, 'case_id': cid, 'drug': case['drug'], 'regime': spec['regime'],
            'old_target_cluster': case['cluster_id'], 'drug_study_group': grouping['case_to_group'][cid],
            'candidate_ids': list(ids), 'candidate_times_h': times.tolist(), 'goals': goals,
            'visible': {'time_h': t.tolist(), 'concentration_mg_l': observed.tolist(),
                        'dosing': deepcopy(training['dosing']), 'source': deepcopy(decision['visible_source'])},
            'baseline': {
                'formula': formula, 'development_ln_rmse': float(fit['best']['development_ln_rmse']),
                'visible_prediction_mg_l': prediction,
                'visible_log_residual_observed_minus_predicted': residual,
                'prediction_status': 'locked_parameter_replay' if allow_model_replay else 'not_cached',
                'log_parameters': x.tolist(), 'parameter_lower': lower.tolist(), 'parameter_upper': upper.tolist(),
                'fractional_distance_to_nearest_bound': (np.minimum(x-lower, upper-x)/(upper-lower)).tolist(),
                'no_warm_start': fit.get('no_warm_start'), 'source': deepcopy(fit_ref),
            },
            'committee': {
                'concentrations_mg_l': committee_c.tolist(), 'cmax_ng_ml': cmax,
                'cmax_source_per_member': cmax_origins, 'common_target_contract': deepcopy(contract_ref),
                'member_count': len(decision['members']),
                'members': [{k: deepcopy(m[k]) for k in ('formula', 'profile_index', 'log_parameters', 'mse') if k in m}
                            for m in decision['members']],
                'candidate_concentrations_are_predictions_not_acquired_measurements': True,
            },
            'context': {'visible_log_log_slope': slope, 'infusion_end_h':
                        training['dosing']['start_h'] + training['dosing']['infusion_minutes']/60,
                        'visible_jacobian_whitened': deepcopy(feature.get('visible_jacobian_whitened'))},
        })
    return {
        'version': VERSION, 'feature_names': list(FEATURE_NAMES), 'designs': designs,
        'grouping': grouping, 'source_manifest': reader.records(),
        'replay': {'enabled': allow_model_replay, 'solver_calls': len(replay_log),
                   'counts': dict(Counter(x['kind'] for x in replay_log)), 'calls': replay_log,
                   'fits': 0, 'native_exports': 0},
        'isolation': {'hidden_values_used': False, 'target_observations_used': False,
                      'labels_returned': False, 'clinical_audit_used_only_for_grouping': True,
                      'writes': 0, 'solver_calls': len(replay_log)},
    }
