"""Re-express frozen sampling profiles under the common target-NCA contract.

No refitting, clinical outcome reads, policy learning, or filesystem writes occur
here. The frozen development concentration committee and parameter vectors are
reused. Only model-derived target endpoints and their derivatives are rebuilt.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import FunctionType

import numpy as np

import run_sampling_expansion as expansion

VERSION = 'common-reference-sampling-features-1'
GOALS = expansion.GOALS


def _record_pin(reference, pins):
    if not isinstance(reference, dict) or not reference.get('path') or not reference.get('sha256'):
        raise ValueError('A source with a hash is required')
    existing = pins.setdefault(reference['path'], reference['sha256'])
    if existing != reference['sha256']:
        raise ValueError('Conflicting source hashes')


def _refresh_decision(p, key, entry, spec):
    """Preserve all committee members; unavailable goals stay unsupported."""
    pins = {}

    def checked(reference):
        _record_pin(reference, pins)
        return expansion.checked(reference)

    source = checked(entry['result'])
    if (source.get('design_key') != key
            or source.get('hidden_values_used') is not False
            or source.get('target_values_used') is not False
            or source.get('visible_source') != spec['visible_source']):
        raise ValueError('Frozen committee identity or isolation differs')
    expansion.verify(source['output_pins'])
    pins.update(source['output_pins'])
    visible = checked(spec['visible_source'])
    menu = visible['design_only']
    if (source['candidate_ids'] != menu['candidate_ids']
            or source['candidate_times_h'] != menu['candidate_times_h']):
        raise ValueError('Frozen candidate menu differs')
    case = expansion.gap.get_case(p, spec['case_id'])
    contract = checked(case['target_contract'])
    if (contract.get('source_kind') != 'target_protocol_metadata_only'
            or contract.get('target_observation_values_used') is not False):
        raise ValueError('Target contract must contain metadata only')
    concentrations = np.asarray(source['concentrations_mg_l'], dtype=float)
    if (concentrations.shape != (len(source['members']), len(menu['candidate_ids']))
            or len(concentrations) < 2 or not np.isfinite(concentrations).all()
            or np.any(concentrations <= 0)):
        raise ValueError('Frozen committee has invalid candidate concentrations')

    refreshed = deepcopy(source)
    refreshed['target_contract'] = case['target_contract']
    values = {goal: [] for goal in GOALS}
    members = []
    # Requests/physiology are source-checked. Reuse them across committee rows.
    target_inputs = {}
    for formula in sorted({m['formula'] for m in source['members']}):
        _record_pin(case['requests']['evidence']['target'], pins)
        req = expansion.gap.request(case, 'target', formula)
        physref = p['physiology_cache'][case['case_id'] + '__target__' + formula]
        _record_pin(physref, pins)
        phys = expansion.prior.get_phys(p, case, 'target', formula, req)
        target_inputs[formula] = (req, phys, expansion.base.engine.prediction_grid(req))
    for member in source['members']:
        req, phys, grid = target_inputs[member['formula']]
        predicted = expansion.old.simulate('blood_partition', phys, req, grid, member['log_parameters'])
        goals = expansion.data.model_goals(contract,
            expansion.prior.curve_document(predicted, grid, req), req)
        if goals.get('target_observation_values_used') is not False:
            raise ValueError('Model-only target endpoint contract violated')
        members.append(dict(member, goal_diagnostics=goals))
        for goal in GOALS:
            values[goal].append(goals['values'][goal])
    refreshed['members'] = members
    refreshed['goal_values'] = values
    refreshed['details'] = {}
    # Generic time/variance rules depend only on unchanged times/concentrations.
    refreshed['choices'] = deepcopy(source['choices'])
    for goal in GOALS:
        finite = np.asarray([v is not None and math.isfinite(v) and v > 0 for v in values[goal]])
        if finite.sum() < 2:
            refreshed['details'][goal] = dict(status='unsupported', members=int(finite.sum()))
            for method in ('request_directed', 'linear_covariance'):
                refreshed['choices'][method][goal] = None
            continue
        selected = expansion.gap.design.choose(menu['candidate_times_h'], menu['observed_times_h'],
            concentrations[finite], {goal: np.asarray(values[goal], dtype=float)[finite]},
            log_noise_sd=.1, quadrature_nodes=31)
        refreshed['details'][goal] = dict(status='selected', members=int(finite.sum()),
            missing_members=int((~finite).sum()), sensitivity={'0.1': selected})
        refreshed['choices']['request_directed'][goal] = selected['policies']['nonlinear_goal_directed'][goal]['selected_index']
        refreshed['choices']['linear_covariance'][goal] = selected['policies']['goal_directed'][goal]['selected_index']
    return refreshed, pins


def _feature_builder(refreshed, decision_ref, pins):
    """Use the existing builder without mutating its module-global reader."""
    original = expansion.make_features
    original_checked = expansion.checked

    def checked(reference):
        _record_pin(reference, pins)
        if reference == decision_ref:
            return deepcopy(refreshed)
        return original_checked(reference)

    globals_copy = dict(original.__globals__, checked=checked)
    derived = FunctionType(original.__code__, globals_copy, original.__name__,
        original.__defaults__, original.__closure__)
    derived.__kwdefaults__ = original.__kwdefaults__
    return derived


def rebuild_original_features(expansion_protocol, original_state):
    """Return twenty feature documents, each with hash-bound rebuild receipts.

    ``original_state`` accepts the exhaustive evidence-gap state or the learned
    ten-drug state (which points to it). New-drug outcomes and feature files are
    neither read nor altered. The caller owns persistence and policy training.
    """
    state = original_state
    initial_pins = {}
    if 'designs' not in state:
        _record_pin(state['protocol'], initial_pins)
        learned_protocol = expansion.checked(state['protocol'])
        _record_pin(learned_protocol['previous_state'], initial_pins)
        state = expansion.checked(learned_protocol['previous_state'])
    if state.get('status') != 'complete':
        raise ValueError('Original exhaustive study must be complete')
    _record_pin(state['protocol'], initial_pins)
    original_protocol = expansion.checked(state['protocol'])
    ids = set(expansion_protocol['original_case_ids'])
    if len(ids) != 10 or ids != {c['case_id'] for c in original_protocol['cases']}:
        raise ValueError('Only the frozen original ten compounds may be rebuilt')
    specs = original_protocol['designs']
    if (len(specs) != 20 or set(state['designs']) != set(specs)
            or {v['case_id'] for v in specs.values()} != ids):
        raise ValueError('Exactly twenty original sampling layouts are required')
    p = dict(expansion_protocol, designs={**expansion_protocol['designs'], **specs})
    out = {}
    for key, spec in specs.items():
        entry = state['designs'][key]
        refreshed, pins = _refresh_decision(p, key, entry, spec)
        pins.update(initial_pins)
        features = _feature_builder(refreshed, entry['result'], pins)(p, key, entry)
        case = expansion.gap.get_case(p, spec['case_id'])
        for phase in ('development', 'target'):
            _record_pin(case['requests']['evidence'][phase], pins)
            for formula in expansion.base.FORMULAS:
                _record_pin(p['physiology_cache'][case['case_id'] + '__' + phase + '__' + formula], pins)
        # Existing builder scores use the old ranker only as an implementation
        # byproduct. Do not expose these as selections under the revised rule.
        for info in features['goals'].values():
            info.pop('scores', None)
            info['choices'].pop('learned_gain', None)
        features.update(case_id=case['case_id'], drug=case['drug'], cluster_id=case['cluster_id'],
            regime=spec['regime'], times=refreshed['candidate_times_h'], baseline=refreshed['baseline'],
            decision_source=entry['result'], visible_values_only=True,
            source_pins=pins, feature_rebuild_receipt=dict(version=VERSION,
                old_decision=entry['result'], target_contract=case['target_contract'],
                committee_members=len(refreshed['members']), committee_parameters_unchanged=True,
                candidate_concentrations_unchanged=True, refit_calls=0,
                committee_target_simulations=len(refreshed['members']),
                log_noise_sd=.1, quadrature_nodes=31, hidden_values_used=False,
                target_observation_values_used=False, old_ranker_scores_discarded=True))
        expansion.verify(pins)
        out[key] = features
    return out
