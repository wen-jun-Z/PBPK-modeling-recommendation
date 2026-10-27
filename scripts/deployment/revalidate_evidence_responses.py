"""Recheck saved project Qwen responses after a validator fix, preserving audits."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stop-paused-pipeline', action='store_true')
    parser.add_argument('--responses-from-validator-archive', help='Replay the same original responses from a preserved validator archive.')
    args = parser.parse_args()
    if not re.fullmatch(r'run_\d{8}T\d{6}Z', args.run_id):
        raise ValueError('Invalid project run ID')
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    from pbpk_kb.processing import run_llm_extraction as m
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    spec = importlib.util.spec_from_file_location('watchdog', root/'scripts/deployment/evidence_watchdog.py')
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    if args.stop_paused_pipeline:
        state = json.loads((root/f'runtime/deployment/evidence_monitor/{args.run_id}/status.json').read_text())
        target = watchdog.same_process(state['target'])
        if not target or target['state'] != 'T':
            raise RuntimeError('Only the registered, already paused project pipeline may be stopped')
        os.killpg(target['pid'], signal.SIGTERM)
        os.killpg(target['pid'], signal.SIGCONT)
    lock = (root/'runtime/deployment/evidence_pipeline.lock').open('a')
    for _ in range(100):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.1)
    else:
        raise RuntimeError('Project pipeline has not released its lock')
    base = root/f'data_process/raw_processing/{args.run_id}'
    layout = raw_processing_layout(base)
    archive = root/f'artifacts/pbpk_raw_runs/compound_runs/{args.run_id}/finalization/source_recovery/llm_before_{m.VALIDATOR_VERSION}'
    responses = list(m._iter_jsonl(layout.path('llm_extraction_responses_live.jsonl')))
    replay_source = 'live_response_ledger'
    if args.responses_from_validator_archive:
        version = args.responses_from_validator_archive
        if not re.fullmatch(r'pbpk-llm-validator-v[0-9]+', version):
            raise ValueError('Invalid validator archive version')
        source = archive.parent/f'llm_before_{version}'/'05_llm_extraction'/'llm_extraction_responses_live.jsonl'
        saved = list(m._iter_jsonl(source))
        identity_fields = ('request_id', 'input_hash', 'output_hash', 'attempt_number', 'status',
                           'prompt_template_version', 'chunk_id', 'completed_at')
        identities = lambda rows: [tuple(row.get(key) for key in identity_fields) for row in rows]
        if identities(saved) != identities(responses):
            raise RuntimeError('Replay archive does not contain exactly the same recorded model attempts')
        responses = saved
        replay_source = str(source.relative_to(root))
    archive.mkdir(exist_ok=False)
    for name in ('05_llm_extraction', '04_evidence_candidates', '00_run_manifest'):
        shutil.copytree(base/name, archive/name)
    requests = {(r['request_id'], r['input_hash']): r for r in m._iter_jsonl(layout.path('llm_extraction_requests_live.jsonl'))}
    ids = {r['chunk_id'] for r in responses}
    chunks = {r['chunk_id']: r for r in m._iter_jsonl(layout.path('text_chunks.jsonl')) if r['chunk_id'] in ids}
    upstream = m._load_upstream_index(layout)
    counts = Counter()
    failures = [r for r in m._iter_jsonl(layout.path('processing_failures.jsonl'))
                if r.get('failure_type') != 'llm_candidate_validation_rejection']
    revised, tasks = [], []
    for response in responses:
        request = requests[response['request_id'], response['input_hash']]
        task = m.PendingTask(chunk=chunks[response['chunk_id']], target_compound=request['target_compound'],
                             status_record={}, request_record=request, input_hash=request['input_hash'])
        raw = [*response.get('candidates', []), *[r['candidate'] for r in response.get('rejected_candidates', [])]]
        previous_repairs = {}
        for candidate in raw:
            repairs = candidate.pop('validation_repairs', [])
            previous_repairs[json.dumps(candidate, sort_keys=True)] = repairs
        valid, rejected = m._validate_candidates(task, raw, upstream)
        for candidate in valid:
            new = candidate.pop('validation_repairs', [])
            old = previous_repairs.get(json.dumps(candidate, sort_keys=True), [])
            if old or new:
                candidate['validation_repairs'] = list(dict.fromkeys(old+new))
        revised.append({**response, 'candidates': valid, 'candidate_count': len(valid),
                        'rejected_candidates': rejected, 'rejected_candidate_count': len(rejected),
                        'raw_candidate_count': len(raw), 'validator_version': m.VALIDATOR_VERSION,
                        'revalidated_from_validator_version': response.get('validator_version'),
                        'schema_validation_status': ('partial_rejected' if valid and rejected else 'rejected' if rejected else 'valid') if response.get('status') in m.PROCESSED_LLM_STATUSES else response.get('schema_validation_status', 'not_validated')})
        tasks.append((task, valid))
        counts.update(real_requests=1, accepted_candidates=len(valid), rejected_candidates=len(rejected))
        failures.extend(m._candidate_failure(task, r) for r in rejected)

    def write(path, rows):
        temporary = path.with_suffix(path.suffix+'.revalidate.tmp')
        with temporary.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        temporary.replace(path)

    write(layout.path('llm_extraction_responses_live.jsonl'), revised)
    write(layout.path('processing_failures.jsonl'), failures)
    llm, merged = layout.path('evidence_candidates_llm.jsonl'), layout.path('evidence_candidates.jsonl')
    llm.write_text('')
    shutil.copyfile(layout.path('evidence_candidates_deterministic.jsonl'), merged)
    existing = m._load_existing_evidence_ids(merged)
    for task, valid in tasks:
        m._append_evidence_rows(task.chunk, task.target_compound, valid, llm_evidence_path=llm,
                                merged_evidence_path=merged, existing_evidence_ids=existing, counts=Counter())
    audit = watchdog.evidence_audit({r['request_id']: r for r in requests.values()}, revised, historical=True)
    result = {'status': 'passed' if not audit['reasons'] else 'failed', 'validator_version': m.VALIDATOR_VERSION,
              'routing_version': m.LLM_ROUTING_VERSION, 'counts': dict(counts), 'audit': audit,
              'archive': str(archive.relative_to(root)), 'generated_at': m._utc_now(),
              'stage_acceptance_unchanged': True, 'replay_source': replay_source}
    (base/'00_run_manifest'/f'revalidation_{m.VALIDATOR_VERSION}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if audit['reasons']:
        raise RuntimeError('Saved response audit did not pass')


if __name__ == '__main__':
    main()
