#!/usr/bin/env python3
"""Compare two fixed partition formulas after the complete frozen baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pbpk_kb.blueprints.partition_selection import run_partition_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    summary = run_partition_selection(baseline_dir=args.baseline, output_dir=args.output,
        project_root=Path(__file__).resolve().parents[1], max_workers=args.workers)
    print(json.dumps({k:v for k,v in summary.items() if k not in
        {'candidate_results','baseline_results','selected_results','stratified_metrics'}}, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
