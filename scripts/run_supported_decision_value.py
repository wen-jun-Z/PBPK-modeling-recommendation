#!/usr/bin/env python3
"""Run the registered decision comparison with calibration-supported locality.

The old experiment and code remain frozen.  Only an independently selected
reference family changes; the native candidate pool and scoring are identical.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_decision_value as base
from pbpk_kb.blueprints.source_workflow import save

DEFAULT = ROOT / 'experiments/blueprint_design/decision_value_supported_20261001'
ORIGINAL_PRIOR = ROOT / 'experiments/blueprint_design/decision_value_20261001/reference_prior.json'
VERSION = 'decision-value-supported-runner-1'


def supported_decision(policy, request, candidates, references, *, primary_endpoint, method):
    data = deepcopy(references)
    if (method in {'local_risk_only', 'purpose_decision'}
            and primary_endpoint in policy['endpoints'] and primary_endpoint in data):
        family = policy['endpoints'][primary_endpoint]['family']
        if family == 'global':
            data[primary_endpoint]['local_log_reference_values'] = list(data[primary_endpoint]['global_log_reference_values'])
        data[primary_endpoint]['calibration_selected_family'] = family
    chosen_method = method
    # Distribution does not alter the same supplied elimination rate. Avoid
    # treating a minute NCA quadrature difference as a CL selection benefit.
    if primary_endpoint == 'clearance' and method in {'global_decision', 'purpose_decision'}:
        a = candidates.get('parameter_only', {}).get('effective_total_clearance_l_h')
        b = candidates.get('combined', {}).get('effective_total_clearance_l_h')
        if (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(a) and math.isfinite(b)
                and a > 0 and b > 0 and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)):
            chosen_method = 'local_risk_only' if method == 'purpose_decision' else 'fixed_capacity'
    result = base.decide_value(request, candidates, data, primary_endpoint=primary_endpoint, method=chosen_method)
    if chosen_method != method:
        result['reasons'].append('shared_clearance_input_no_structural_CL_choice')
    result['method'] = method
    return result


class Runner(base.Runner):
    def load(self, run):
        namespace = dict(base.Runner.load.__globals__)
        namespace['VERSION'] = VERSION
        loader = FunctionType(base.Runner.load.__code__, namespace, argdefs=base.Runner.load.__defaults__)
        return loader(self, run)

    def prepare(self, run):
        run = self.path(run)
        policy_path = run / 'calibration_policy.json'
        from pbpk_kb.blueprint_learning.calibration_supported_policy import fit_calibration_policy
        if not policy_path.is_file():
            raise ValueError('Prepare requires frozen independent calibration model selection')
        policy = fit_calibration_policy(self.root, policy_path)
        # Base preparation verifies the original reference artifact in place.
        # The project-local temporary link is removed after references resolve.
        link = run / 'reference_prior.json'
        if link.exists() or link.is_symlink():
            raise FileExistsError('Preserve existing reference file')
        link.symlink_to(ORIGINAL_PRIOR)
        try:
            result = super().prepare(run)
        finally:
            link.unlink()
        p = self.read(run / 'protocol.json'); s = self.read(run / 'state.json')
        p['version'] = VERSION
        ref = self.ref(policy_path)
        p['calibration_policy'] = ref
        p['reference_families'] = {ep: info['family'] for ep, info in policy['endpoints'].items()}
        p['source_pins'].update(policy['source_pins']); p['source_pins'][ref['path']] = ref['sha256']
        for name in ('scripts/run_supported_decision_value.py', 'src/pbpk_kb/blueprint_learning/calibration_supported_policy.py'):
            p['code_pins'][name] = self.ref(self.root / name)['sha256']
        p['locality_selection'] = 'Whole-calibration-scaffold leave-out; local only if mean CRPS improvement exceeds one standard error'
        p['clearance_selection'] = 'Equal effective clearance inputs: retain fixed-capacity choice, do not select NCA numerical noise'
        save(run / 'protocol.json', p); s['protocol'] = self.ref(run / 'protocol.json'); save(run / 'state.json', s)
        return dict(result, reference_families=p['reference_families'])

    def run(self, run):
        p, _ = self.load(run)
        policy = base.checked(p['calibration_policy'], self.root)
        def decide(request, candidates, references, **kwargs):
            return supported_decision(policy, request, candidates, references, **kwargs)
        namespace = dict(base.Runner.run.__globals__)
        namespace['decide_value'] = decide
        namespace['VERSION'] = VERSION
        execute = FunctionType(base.Runner.run.__code__, namespace, argdefs=base.Runner.run.__defaults__)
        return execute(self, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report', 'status'])
    parser.add_argument('--run', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(getattr(Runner(), args.command)(args.run), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
