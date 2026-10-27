"""Bounded new-request inference with frozen PBPK-Match light research heads.

The worker consumes observable requests, frozen catalogue representations and
saved heads. It never loads outcome arrays, grades or hidden source judgments.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import sys
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from pbpk_kb.recommendation.schemas import RecommendationRequest

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runtime/deployment/workspace/recommendations'
LIGHT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
router = APIRouter(prefix='/workspace/research', tags=['workspace-recommendation'])
JOB_SECONDS = 240
ACTIVE = {'queued', 'running'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def scoped(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Artifact escapes project scope')
    return path


def read(path):
    return json.loads(scoped(path).read_text())


def write(path, value):
    path = scoped(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with scoped(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def job_directory(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise HTTPException(404, 'Job not found')
    path = scoped(RUN / job_id)
    if not path.is_dir():
        raise HTTPException(404, 'Job not found')
    return path


def process_ticks(pid):
    # Only a PID originally saved by this project's worker is inspected.
    info = Path('/proc') / str(int(pid))
    if info.stat().st_uid != os.getuid() or (info / 'cwd').resolve() != ROOT:
        return None
    text = (info / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return None if fields[0] == 'Z' else fields[19]


def state(directory):
    result = read(directory / 'status.json')
    if result['status'] in ACTIVE:
        try:
            alive = bool(result.get('pid')) and process_ticks(result['pid']) == result.get('start_ticks')
        except (OSError, ValueError):
            alive = False
        if not alive and time.time() - result['created_epoch'] > 10:
            result.update(status='failed', stage='interrupted', message='The inference process ended before completion. You can submit again.', updated_at=stamp())
            write(directory / 'status.json', result)
    return result


def clean_environment():
    name = '_workspace_project_environment'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/deployment/service_runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.environment()
    values.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return values


def valid_payload(request):
    raw = request.model_dump(mode='json')
    if len(json.dumps(raw, ensure_ascii=False).encode()) > 20000:
        raise HTTPException(422, 'Request is too long; simplify the conditions.')
    for value in raw.values():
        if isinstance(value, str) and len(value) > 2048:
            raise HTTPException(422, 'A request field is too long.')
    return raw


@router.post('/recommend', status_code=202)
def recommend(request: RecommendationRequest):
    payload = valid_payload(request)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    RUN.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (RUN / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read(RUN / 'current.json') if (RUN / 'current.json').exists() else None
        if current:
            previous = state(job_directory(current['job_id']))
            if previous['status'] in ACTIVE:
                if previous['request_sha256'] == digest:
                    return JSONResponse(public_state(previous), status_code=202)
                raise HTTPException(409, {'message': 'A model recommendation job is already running. Review it before submitting another.', **public_state(previous)})
        job_id = uuid.uuid4().hex
        directory = RUN / job_id
        directory.mkdir(mode=0o700)
        write(directory / 'request.json', payload)
        record = {'job_id': job_id, 'status': 'queued', 'stage': 'queued', 'message': 'Preparing model recommendations',
                  'created_at': stamp(), 'created_epoch': time.time(), 'updated_at': stamp(),
                  'request_sha256': digest}
        write(directory / 'status.json', record)
        with (directory / 'worker.log').open('wb') as log:
            process = subprocess.Popen([str(ROOT / 'runtime/app-env/bin/python'), '-B', '-m',
                'pbpk_kb.web.recommendations', job_id], cwd=ROOT, env=clean_environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=process.pid, start_ticks=process_ticks(process.pid))
        write(directory / 'status.json', record)
        write(RUN / 'current.json', {'job_id': job_id})
        threading.Thread(target=supervise, args=(process, directory), daemon=True).start()
    return JSONResponse(public_state(record), status_code=202)


def supervise(process, directory):
    """Reap our own child and enforce a second wall-clock bound."""
    try:
        process.wait(timeout=JOB_SECONDS + 20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        record = read(directory / 'status.json')
        if record['status'] in ACTIVE:
            record.update(status='failed', stage='timeout', message='The recommendation timed out and this job was stopped.', updated_at=stamp())
            write(directory / 'status.json', record)


def public_state(record):
    return {**{key: record.get(key) for key in ('job_id', 'status', 'stage', 'message', 'created_at', 'updated_at')},
            'status_url': '/api/workspace/research/jobs/' + record['job_id']}


@router.get('/jobs/current')
def current_job():
    if not (RUN / 'current.json').exists():
        return {'status': 'idle', 'job_id': None}
    return get_job(read(RUN / 'current.json')['job_id'])


@router.get('/jobs/{job_id}')
def get_job(job_id: str):
    directory = job_directory(job_id)
    record = state(directory)
    value = public_state(record)
    if record['status'] == 'complete':
        value['result'] = display_result(read(directory / 'result.json'))
        value['download_url'] = '/api/workspace/research/jobs/' + job_id + '/download'
    return value


@router.get('/jobs/{job_id}/download')
def download_job(job_id: str):
    directory = job_directory(job_id)
    if state(directory)['status'] != 'complete':
        raise HTTPException(409, 'Job is not complete')
    return JSONResponse(display_result(read(directory / 'result.json')), headers={'Content-Disposition': 'attachment; filename=pbpk-model-recommendation.json'})


DISPLAY_LAYERS = (
    ('validated_reuse', 'Passed current applicability checks'),
    ('controlled_adaptation_candidate', 'Controlled adaptation required'),
    ('same_compound_qualification_candidate', 'Same compound; further validation needed'),
    ('mechanistic_reference', 'Cross-compound mechanism reference'),
    ('blocked', 'Not currently applicable'),
    ('not_checked', 'Applicability not yet confirmed'),
)


def display_result(result):
    """Project frozen scores through use qualifications without rewriting ranks.

    ``ranking`` is always the original 280-entry research score order.
    ``top_candidates`` has a separate display rank and never includes blocked or
    unassessed models. Cross-drug mechanism references have their own list.
    """
    labels = dict(DISPLAY_LAYERS)
    groups = {key: [] for key, _ in DISPLAY_LAYERS}
    for raw in result.get('ranking', []):
        gate = raw.get('gate') or {}
        layer = gate.get('candidate_layer')
        blockers = gate.get('hard_blockers') or []
        reasons = sorted({str(item['reason_code']) for item in blockers if item.get('reason_code')}
                         | set(gate.get('technical_blocker_codes') or []))
        if layer not in groups:
            layer = 'not_checked'
        if layer == 'validated_reuse' and (blockers or gate.get('direct_reuse_allowed') is False):
            layer = 'blocked'
            reasons = sorted({*reasons, 'INCONSISTENT_REUSE_GATE'})
        if layer in dict(DISPLAY_LAYERS[:3]) and raw.get('policy_eligible') is not True:
            layer = 'not_checked'
            reasons = sorted({*reasons, 'CATALOG_IDENTITY_NOT_CONFIRMED'})
        if layer == 'not_checked' and not reasons:
            reasons = ['USE_NOT_ASSESSED']
        groups[layer].append({**raw, 'rank': None, 'rank_end': None, 'display_rank': None,
            'display_rank_end': None, 'display_layer': layer, 'decision_label': labels[layer],
            'reason_codes': reasons, 'direct_reuse_allowed': layer == 'validated_reuse',
            'qualification_actions': gate.get('qualification_actions') or [],
            'raw_rank': raw.get('raw_rank', raw.get('rank'))})
    offset = 0
    for layer, _ in DISPLAY_LAYERS:
        rows = groups[layer]
        rows.sort(key=lambda row: (-row['score'], row['candidate_id']))
        if layer in dict(DISPLAY_LAYERS[:4]):
            # Cross-drug references use their own numbering, not main-model ranks.
            base = 0 if layer == 'mechanistic_reference' else offset
            for row in rows:
                higher = sum(other['score'] > row['score'] for other in rows)
                tied = sum(other['score'] == row['score'] for other in rows)
                row.update(rank=base + higher + 1, rank_end=base + higher + tied,
                           display_rank=base + higher + 1, display_rank_end=base + higher + tied)
            if layer != 'mechanistic_reference':
                offset += len(rows)
    primary = [row for layer, _ in DISPLAY_LAYERS[:3] for row in groups[layer]]
    top = [row for row in primary if row['display_rank'] <= 10]
    references = [row for row in groups['mechanistic_reference'] if row['display_rank'] <= 10]
    status = ('reuse_candidate_found' if groups['validated_reuse'] else
              'adaptation_required' if groups['controlled_adaptation_candidate'] else
              'qualification_required' if top else 'reference_only' if references else 'no_cleared_candidate')
    return {**result, 'status': status, 'display_policy': 'qualified-use-layers-then-learned-score-v1',
            'top_candidates': top, 'reference_candidates': references,
            'blocked_candidates': groups['blocked'], 'unassessed_count': len(groups['not_checked']),
            'raw_top_candidates': result.get('raw_top_candidates', result.get('top_candidates', [])),
            'candidate_groups': [{'id': layer, 'label': label, 'count': len(groups[layer])}
                                 for layer, label in DISPLAY_LAYERS],
            'display_notice': 'Candidates are grouped by applicability, then ranked within each group. Pending validation does not permit direct reuse. The download preserves the full original ranking.'}


def input_query(request: RecommendationRequest):
    """Map the public form to the already trained text/condition contract."""
    raw = request.model_dump(mode='json')
    compounds = sorted({x.strip().casefold() for x in (request.compound, request.victim_drug,
                        request.perpetrator_drug, request.interacting_drug) if x})
    objective = request.objective
    task = ('ddi_exposure_prediction' if objective.startswith('ddi') else
            'special_population_prediction' if objective in {'special_population', 'pediatric', 'geriatric', 'pregnancy', 'renal_impairment', 'hepatic_impairment'}
            else 'plasma_concentration_time_prediction')
    route = {'po': 'oral', 'iv': 'intravenous', 'iv_bolus': 'intravenous', 'iv_infusion': 'intravenous'}.get(request.route, request.route)
    endpoints = [{'concentration_time_profile': 'concentration_time', 'half_life': 'half_life'}.get(x, x.casefold()) for x in request.prediction_endpoints]
    features = {'species': [request.species], 'routes': [route] if route else [],
                'populations': [x for x in (request.population, request.age_range, request.disease_state, request.renal_function, request.hepatic_function, request.genotype, request.pregnancy_status) if x],
                'endpoints': endpoints, 'enzymes': sorted(set(request.enzyme + request.required_enzymes)),
                'transporters': sorted(set(request.transporter + request.required_transporters)),
                'formulations': [x for x in (request.formulation, request.dosage_form, request.release_type) if x],
                'model_structure': [request.whole_body_or_reduced_pbpk] if request.whole_body_or_reduced_pbpk else [],
                'software': request.software_preference, 'compartments': request.required_organs, 'process_types': []}
    regimen = {}
    if request.single_or_multiple == 'single':
        regimen['dose_times_h'] = [0]
    frequency = (request.frequency or '').strip().casefold()
    interval = {'qd': 24, 'daily': 24, 'once daily': 24, 'bid': 12, 'twice daily': 12, 'tid': 8, 'qid': 6}.get(frequency)
    match = re.fullmatch(r'(?:q|every\s*)(\d+(?:\.\d+)?)\s*(?:h|hours?)', frequency)
    if interval is None and match:
        interval = float(match.group(1))
    if interval:
        regimen['dosing_interval_h'] = interval
    infusion = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?)\s*', request.infusion_duration or '', re.I)
    if infusion:
        regimen['infusion_duration_h'] = float(infusion[1]) / (60 if infusion[2].lower().startswith('min') else 1)
    dose = {k: v for k, v in {'value': request.dose, 'unit': request.dose_unit, 'route': route, 'formulation': request.formulation}.items() if v is not None}
    design = {'dose': dose, 'regimen': regimen, 'food': request.fed_or_fasted,
              'formulation': request.formulation, 'route': route}
    descriptions = {'plasma_concentration_time_prediction': 'predict the plasma concentration-time profile',
                    'ddi_exposure_prediction': 'predict drug interaction exposure changes',
                    'special_population_prediction': 'predict pharmacokinetics in the requested population'}
    lines = ['Find PBPK model candidates for ' + ', '.join(compounds) + '.',
             'Intended task: ' + descriptions[task] + '.', 'Species context: ' + request.species + '.',
             'Administration: ' + json.dumps(dose, ensure_ascii=False, sort_keys=True),
             'Regimen and observation origin: ' + json.dumps(regimen, ensure_ascii=False, sort_keys=True),
             'Population context: ' + ', '.join(features['populations'] or ['unspecified']),
             'Food context: ' + str(request.fed_or_fasted or 'unspecified'),
             'Target endpoints: ' + ', '.join(endpoints or ['unspecified']),
             'Enzyme mentions in supplied evidence: ' + ', '.join(features['enzymes'] or ['unspecified']),
             'Transporter mentions in supplied evidence: ' + ', '.join(features['transporters'] or ['unspecified'])]
    # Additional requested constraints remain visible to the encoder; no output
    # fields, labels or measured values are admitted by RecommendationRequest.
    extras = {key: raw[key] for key in ('objective', 'constraints', 'frequency', 'infusion_duration', 'body_weight_kg',
              'sex', 'ethnicity', 'age_range', 'formulation', 'dosage_form', 'release_type', 'whole_body_or_reduced_pbpk',
              'parent_or_parent_metabolite', 'required_organs', 'software_preference', 'interaction_mechanism',
              'victim_drug', 'perpetrator_drug', 'victim_regimen', 'perpetrator_regimen') if raw[key] not in (None, [], '')}
    lines.append('Additional requested conditions: ' + json.dumps(extras, ensure_ascii=False, sort_keys=True))
    key = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {'query_id': 'workspace:' + key, 'compounds': compounds, 'task_type': task, 'features': features,
            'text': '\n'.join(lines), 'variants': {}}, design


def select_folds(compounds, query_records, folds):
    matched = {i for i, row in enumerate(query_records) if set(compounds) & {c.casefold() for c in row['compounds']}}
    selected = [i for i, fold in enumerate(folds) if not matched or matched <= set(fold['outer_test'])]
    if not selected:
        raise ValueError('This multi-drug request spans training groups, and no saved weights exclude all these drugs together. Use applicability checks or inspect each drug separately.')
    return selected


def infer(request: RecommendationRequest, progress=lambda stage, message: None):
    # Imports are local: loading the UI never imports Torch or allocates VRAM.
    import numpy as np
    import torch
    from scipy.stats import rankdata
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from pbpk_match_light_training import LightMatch, _canonical, _normalized
    from pbpk_kb.recommendation.pbpk_match.serving import encode_new_query
    from pbpk_kb.recommendation.pbpk_match_v2.data import pair_feature_row, feature_names
    from pbpk_kb.recommendation.pbpk_match.policy import catalog_identity_policy, select_policy_topk
    from pbpk_kb.recommendation.pbpk_match.data import fields_for
    from pbpk_kb.storage.db import SessionLocal
    from pbpk_kb.recommendation.public_model_service import recommend_public_model, PublicModelRecommendationError
    from sqlalchemy import text

    progress('loading', 'Verifying the model and candidate library')
    torch.set_num_threads(2)
    protocol = read(LIGHT / 'protocol.json')
    manifest = read(LIGHT / 'cache/manifest.json')
    binding = read(LIGHT / 'cache/binding.json')
    if not manifest.get('completed') or sha(LIGHT / 'cache/manifest.json') != protocol['cache_manifest']['sha256']:
        raise ValueError('Frozen candidate representations do not match the model protocol')
    for name in ('scripts/pbpk_match_light_training.py', 'src/pbpk_kb/recommendation/pbpk_match_v2/data.py'):
        if sha(ROOT / name) != protocol['implementation_sha256'][name]:
            raise ValueError('Inference dependencies differ from the frozen training versions')
    if sha(LIGHT / 'cache/binding.json') != manifest['binding_sha256'] or sha(LIGHT / 'cache/inputs.npz') != manifest['array_sha256']:
        raise ValueError('Frozen candidate representation verification failed')
    candidates = manifest['candidate_records']
    ids = [c['candidate_id'] for c in candidates]
    if ids != protocol['candidate_ids'] or len(ids) != 280 or manifest['feature_names'] != feature_names():
        raise ValueError('Candidate or condition features differ from the training version')
    with np.load(LIGHT / 'cache/inputs.npz', allow_pickle=False) as arrays:
        embeddings = arrays['raw_candidate_embeddings'].astype(np.float32)
    query, design = input_query(request)
    folds = select_folds(query['compounds'], manifest['query_records'], protocol['folds'])
    progress('encoding', 'Encoding the request on the server GPU')
    vectors, encoder = encode_new_query([query['text']], binding['encoder'], device='cuda:0', batch_size=1)
    features = np.asarray([pair_feature_row(query, c, design=design) for c in candidates], np.float32)
    progress('ranking', 'Ranking all 280 model versions')
    outputs = []
    for fold in folds:
        directory = LIGHT / f'runs/seed17/fold{fold}/outer'
        receipt = read(directory / 'complete.json')
        for name in ('heads.pt', 'metadata.json', 'binding.json'):
            if sha(directory / name) != receipt['artifacts_sha256'][name]:
                raise ValueError('Small-network weight verification failed')
        metadata = read(directory / 'metadata.json')
        job_binding = read(directory / 'binding.json')
        if job_binding['protocol_sha256'] != sha(LIGHT / 'protocol.json') or job_binding['configuration']['selection_sha256'] != sha(directory.parent / 'selection.json'):
            raise ValueError('Inference weights do not match the selected training group')
        selection = read(directory.parent / 'selection.json')
        saved = torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True)
        model = LightMatch(metadata['embedding_size'], len(metadata['feature_mean']), metadata['width'], metadata['dropout']).to('cuda:0').eval()
        model.load_state_dict(saved['heads'])
        normalized = _normalized(features, metadata)
        order, restore = _canonical(embeddings, normalized)
        with torch.inference_mode():
            output = model(torch.as_tensor(vectors[0], device='cuda:0'),
                           torch.as_tensor(embeddings[order], device='cuda:0'),
                           torch.as_tensor(normalized[order], device='cuda:0'))
            scores = output['scores'].cpu().numpy()[restore]
            probs = (output['app_logits'] / selection['calibration']['temperature']).softmax(-1).cpu().numpy()[restore]
        outputs.append((scores, probs))
        del model, output, saved
    torch.cuda.empty_cache()
    if len(outputs) == 1:
        scores, probabilities = outputs[0]
    else:
        scores = np.mean([rankdata(value[0], method='average') / len(ids) for value in outputs], axis=0)
        probabilities = np.mean([value[1] for value in outputs], axis=0)
    if scores.shape != (280,) or probabilities.shape != (280, 3) or not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise ValueError('Full-candidate inference is incomplete')
    identity = catalog_identity_policy(query, candidates)
    selected = select_policy_topk(scores, identity, k=10)
    progress('checking', 'Checking applicability and catalog model versions')
    policy, policy_error = None, None
    with SessionLocal() as db:
        mapping = {str(row['model_version_id']): dict(row) for row in db.execute(text('''
            SELECT v.model_version_id::text, v.external_model_version_id, v.version_label,
                   m.model_name, m.repository_url
            FROM public_pbpk_model_version v JOIN public_pbpk_model_registry m ON m.model_id=v.model_id
        ''')).mappings()}
        try:
            policy = recommend_public_model(session=db, request=request)
        except PublicModelRecommendationError:
            policy_error = 'Applicability checks could not be completed; all results remain unverified.'
    gates = {str(c['model_version_id']): c for c in (policy or {}).get('candidates', [])}
    ordered = []
    for decision in selected['all_candidates']:
        i = decision['catalog_index']
        candidate = candidates[i]
        catalog = mapping.get(ids[i], {})
        gate = gates.get(ids[i])
        fields = fields_for(candidate, candidate=True)
        missing = [{'field': key, 'requested_values': values, 'catalog_values': fields.get(key, [])}
                   for key, values in query['features'].items() if values and not set(x.casefold() for x in values) <= set(x.casefold() for x in fields.get(key, []))]
        ordered.append({'candidate_id': ids[i], 'model_version_id': catalog.get('model_version_id'),
            'name': catalog.get('model_name') or candidate.get('family_name'), 'family_id': candidate.get('family_id'),
            'version': catalog.get('version_label'), 'source_url': catalog.get('repository_url'),
            'compounds': candidate.get('compounds'), 'rank': decision['policy_rank'],
            'raw_rank': decision['raw_rank'], 'rank_end': decision['policy_rank_end'], 'score': float(scores[i]),
            'probabilities': dict(zip(('Y', 'N', 'U'), map(float, probabilities[i]))),
            'policy_eligible': decision['eligible'], 'field_gaps': missing,
            'gate': {key: gate.get(key) for key in ('candidate_layer', 'hard_blockers', 'recommendation_tier', 'scenario_transferability_class', 'direct_reuse_allowed', 'technical_blocker_codes', 'qualification_actions', 'permitted_use')} if gate else
                    {'candidate_layer': 'not_checked', 'hard_blockers': [], 'recommendation_tier': 'qualification_required'},
            'scientific_applicability_verified': False})
    ordered.sort(key=lambda row: (-row['score'], row['candidate_id']))
    selected_ids = {r['candidate_id'] for r in selected['items']}
    top = [row for row in ordered if row['candidate_id'] in selected_ids]
    result = {'status': 'needs_review' if top else 'no_catalog_match', 'method': 'PBPK-Match light',
            'method_id': 'pbpk_match_light_document', 'mode': 'research_inference',
            'candidate_count': len(ids), 'top_candidates': top, 'ranking': ordered,
            'request': request.model_dump(mode='json'), 'request_gaps': request.request_gaps(),
            'policy_summary': {key: (policy or {}).get(key) for key in ('status', 'decision', 'recommended_use', 'candidate_summary')},
            'policy_error': policy_error,
            'checkpoint_use': {'folds': folds, 'rule': 'Known compounds use a held-out-compound research head; unseen compounds use mean fold percentile ranks.',
                               'new_training': False, 'component_enhancement': False, 'error_correction': False},
            'encoder': {'model': encoder['repo'], 'device': encoder['device'], 'peak_gpu_gib': encoder['gpu_peak_allocated_gib']},
            'limitations': ['This is PBPK-Match light research inference; saved grouped weights are not claimed to be independently validated production models.',
                'Scores rank candidates; Y/N/U represent document-support probabilities, not actual prediction accuracy.',
                'Applicability checks are retained separately; research rankings do not override license, scope, or execution-validation requirements.',
                'No PBPK simulation was performed for this request. The curve page shows fixed cases from completed research.']}
    return display_result(result)


def worker(job_id):
    directory = job_directory(job_id)
    # Parent records PID before worker updates state, avoiding initialization races.
    for _ in range(100):
        if read(directory / 'status.json').get('pid') == os.getpid():
            break
        time.sleep(.02)
    def progress(stage, message):
        record = read(directory / 'status.json')
        record.update(status='running', stage=stage, message=message, updated_at=stamp())
        write(directory / 'status.json', record)
    def expired(signum, frame):
        raise TimeoutError('The recommendation exceeded 240 seconds; it stopped and released its resources.')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(JOB_SECONDS)
    try:
        request = RecommendationRequest.model_validate(read(directory / 'request.json'))
        result = infer(request, progress)
        write(directory / 'result.json', result)
        record = read(directory / 'status.json')
        record.update(status='complete', stage='complete', message='Model recommendations completed', updated_at=stamp())
        write(directory / 'status.json', record)
    except Exception as error:
        record = read(directory / 'status.json')
        # No arbitrary provider/network exception text is exposed to the browser.
        detail = str(error) if isinstance(error, (ValueError, TimeoutError)) else 'Model inference did not complete (' + type(error).__name__ + '); job resources were released. Check this project deployment.'
        record.update(status='failed', stage='failed', message=detail[:500], updated_at=stamp())
        write(directory / 'status.json', record)
        print(type(error).__name__ + ': ' + detail[:500], flush=True)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    worker(sys.argv[1])
