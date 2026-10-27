#!/usr/bin/env python3
"""Plan one source-bound PBPK request; numerical design requires --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def inside(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Inputs must be files inside the project')
    return path


def main(argv=None):
    from pbpk_kb.blueprints.native_design import run_native_design, policy_from_template
    from pbpk_kb.blueprints.experiment_runtime import file_sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', required=True, type=Path, help='Existing calibration/selection case bundle')
    parser.add_argument('--request', required=True, type=Path, help='Compact pbpk-template-request-v1 or full native-request-design-v1 JSON')
    parser.add_argument('--output', required=True, type=Path, help='New project-local output directory')
    parser.add_argument('--execute', action='store_true', help='Explicitly fit candidates and generate a native model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--solver-budget', type=int, default=500)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--scenario-workers', type=int, default=2)
    args = parser.parse_args(argv)
    payload = json.loads(inside(args.case_file).read_text())
    case = payload['case']
    # Deliberately do not resolve/read any external_scenarios or holdout_reference.
    request_path = inside(args.request)
    policy = json.loads(request_path.read_text())
    if policy.get('version') == 'pbpk-template-request-v1':
        source = {'provenance_id': str(uuid5(NAMESPACE_URL, str(request_path) + file_sha(request_path))),
            'provenance_kind': 'user_request', 'source_id': str(request_path.relative_to(ROOT)),
            'source_location': 'Structured user regimen and design constraints; not observed pharmacokinetics',
            'source_checksum': file_sha(request_path)}
        policy = policy_from_template(case, policy, request_source=source)
    result = run_native_design(case=case, policy=policy, output=args.output, project_root=ROOT,
        execute=args.execute, device=args.device, max_solver_calls=args.solver_budget,
        cpu_threads=args.cpu_threads, scenario_workers=args.scenario_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in {'planned', 'completed'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
