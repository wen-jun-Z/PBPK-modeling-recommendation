"""Design one PBPK model from a registered template and a structured request.

Default: constraints and snapshot only. --execute enables bounded calibration
and native prediction. This entry point never starts a benchmark or reads Dtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.blueprints.experiment_runtime import file_sha
from pbpk_kb.blueprints.native_design import policy_from_template, run_native_design

ROOT = Path(__file__).resolve().parents[1]


def input_file(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError('Inputs must be project files no larger than 16 MiB')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', required=True, help='Development input package containing case and registered source references')
    parser.add_argument('--request', required=True, help='TemplateRequest JSON or full NativeDesignPolicy JSON')
    parser.add_argument('--output', required=True, help='New directory inside this project')
    parser.add_argument('--execute', action='store_true', help='Fit candidates and generate the requested native model and curve')
    parser.add_argument('--max-solver-calls', type=int, default=500)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--scenario-workers', type=int, default=2)
    parser.add_argument('--budget-strategy', choices=('fit_priority', 'equal'), default='fit_priority',
                        help='Improved allocation/anchored fitting, or the previous equal-share comparator')
    args = parser.parse_args(argv)
    try:
        case_file, request_file = input_file(args.case_file), input_file(args.request)
        package = json.loads(case_file.read_text())
        case = package.get('case', package)
        request = json.loads(request_file.read_text())
        if request.get('version') == 'pbpk-template-request-v1':
            source = {'provenance_id': str(uuid5(NAMESPACE_URL, str(request_file))),
                'provenance_kind': 'user_request', 'source_id': str(request_file.relative_to(ROOT)),
                'source_checksum': file_sha(request_file), 'source_location': 'structured user request'}
            policy = policy_from_template(case, request, request_source=source)
        else:
            policy = request
        result = run_native_design(case=case, policy=policy, output=args.output, project_root=ROOT,
            execute=args.execute, max_solver_calls=args.max_solver_calls, device=args.device,
            cpu_threads=args.cpu_threads, scenario_workers=args.scenario_workers,
            budget_strategy=args.budget_strategy)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'PBPK design input rejected: {exc}\n')
    summary = {key: result.get(key) for key in (
        'status', 'stage', 'candidate_count', 'selected_candidate_id', 'model_executed',
        'delivery_complete', 'search_complete', 'native_model', 'prediction_file', 'reasons', 'reason') if key in result}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.get('status') == 'planned' and not args.execute or result.get('delivery_complete'):
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
