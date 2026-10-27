#!/usr/bin/env python3
"""Study-blocked cross-drug parameter learning; no publication files are changed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_variable, '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprint_policy.perfusion_calibration import prediction_grid
from run_blueprint_validation import select_five

VERSION = 'cross-drug-blueprint-1'
METHODS = ('rf_only', 'pooled_mean', 'ridge')
CONDITIONS = ('zero', 'five')
SOURCE = 'experiments/blueprint_design/expanded_tail_20260930/protocol.json'
DEFAULT_NAME = 'cross_drug_20260930'
CODE = ('scripts/run_cross_drug_blueprint.py', 'scripts/score_cross_drug_blueprint.py',
        'scripts/run_blueprint_validation.py', 'src/pbpk_kb/blueprint_learning/shared_exchange_prior.py',
        'src/pbpk_kb/blueprint_learning/curve_prior.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(case_id, method, condition):
    return '__'.join((case_id, method, condition))


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def clean_training(document):
    keys = ('schema_version', 'compound_id', 'compound_name', 'study_id', 'study_label',
            'role', 'matrix', 'time_unit', 'unit', 'molecular_weight_g_mol', 'human', 'dosing')
    result = {key: deepcopy(document[key]) for key in keys if key in document}
    result['role'] = 'calibration_data_reconstruction'
    result['observations'] = [{key: float(row[key]) for key in ('time_min', 'concentration_umol_per_l')}
                              for row in document['observations']]
    return result


def study_folds(cases):
    """Connected components of compound/study metadata; no outcomes are read."""
    by_id = {case['case_id']: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError('Duplicate clinical case identity')
    remaining, folds = set(by_id), []
    while remaining:
        members = {min(remaining)}
        while True:
            compounds = {by_id[c]['compound_id'] for c in members}
            studies = set().union(*(set(by_id[c]['all_study_ids']) for c in members))
            linked = {c for c in remaining if by_id[c]['compound_id'] in compounds
                      or studies.intersection(by_id[c]['all_study_ids'])}
            expanded = members | linked
            if expanded == members:
                break
            members = expanded
        remaining -= members
        folds.append({'fold_id': f'fold_{len(folds)+1:02d}', 'heldout_case_ids': sorted(members),
            'excluded_compound_ids': sorted({by_id[c]['compound_id'] for c in members}),
            'excluded_study_ids': sorted(set().union(*(set(by_id[c]['all_study_ids']) for c in members))),
            'training_case_ids': sorted(set(by_id)-members)})
    return folds


class Paused(RuntimeError):
    pass


class Monitor:
    def __init__(self, runner, run, protocol, state):
        self.runner, self.run, self.protocol, self.state = runner, run, protocol, state
        self.lock, self.closed, self.halted = threading.RLock(), threading.Event(), threading.Event()
        self.last_size = self.last_hash = 0.

    def persist(self, **updates):
        with self.lock:
            self.state.update(updates, updated_at=now())
            save(self.run/'state.json', self.state)

    def pause(self, reason):
        self.halted.set()
        self.persist(status='paused', pause_reason=str(reason))

    def check(self, force=False):
        with self.lock:
            if self.halted.is_set() or (self.run/'PAUSE').exists():
                raise Paused(self.state.get('pause_reason', 'Pause requested'))
            stamp = time.monotonic(); changed = False
            if force or stamp-self.last_size >= 60:
                size = 0
                for path in self.run.rglob('*'):
                    if path.is_symlink():
                        raise Paused('Unexpected symlink in experiment')
                    try:
                        if path.is_file():
                            size += path.stat().st_size
                    except FileNotFoundError:
                        pass
                monitor = self.state['monitor']
                monitor.update(bytes=size, growth_bytes=size-monitor.get('bytes', 0), checked_at=now())
                self.last_size = stamp; changed = True
                if size > self.protocol['directory_limit_bytes']:
                    raise Paused('Experiment exceeds 10 GiB; stop submitting work')
            if force or stamp-self.last_hash >= self.protocol['monitor_seconds']:
                self.runner.verify(self.protocol['source_pins'])
                self.runner.checked(self.state['protocol'])
                if self.runner.paper_pins() != self.protocol['paper_guard']:
                    raise Paused('Publication files changed')
                self.state['monitor']['hashes_checked_at'] = now()
                self.last_hash = stamp; changed = True
            if changed:
                self.persist()

    def watch(self):
        while not self.closed.wait(15):
            try:
                import psutil
                parent = psutil.Process(os.getpid())
                processes = [parent] + parent.children(recursive=True)
                rss = count = 0
                for process in processes:
                    try:
                        if process.uids().real == os.getuid():
                            rss += process.memory_info().rss; count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                with self.lock:
                    self.state['monitor'].update(own_processes=count, own_rss_bytes=rss)
                self.check()
            except Exception as exc:
                self.pause(str(exc))
                return


class Runner:
    def __init__(self, project_root=ROOT, teacher_engine=None, learner=None, scorer=None,
                 controller_class=Monitor, source=SOURCE):
        self.root = Path(project_root).resolve()
        self.DEFAULT = self.root/'experiments/blueprint_design'/DEFAULT_NAME
        self.source = source
        self.teacher = teacher_engine or importlib.import_module('pbpk_kb.blueprint_policy.tail_exchange_calibration')
        self.learner = learner or importlib.import_module('pbpk_kb.blueprint_learning.shared_exchange_prior')
        self.scorer, self.controller_class = scorer, controller_class

    def read(self, path):
        return json.loads(inside(path, self.root, exists=True).read_text())

    def ref(self, path, **extra):
        path = inside(path, self.root, exists=True)
        return {'path': str(path.relative_to(self.root)), 'sha256': sha256(path),
                'location': 'Frozen cross-drug experiment artifact', **extra}

    def verify(self, pins):
        for name, digest in pins.items():
            if sha256(inside(name, self.root, exists=True)) != digest:
                raise ValueError('Frozen source or artifact changed: '+name)

    def checked(self, reference):
        self.verify({reference['path']: reference['sha256']})
        return self.read(reference['path'])

    def pin_refs(self, value, pins):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.verify({value['path']: value['sha256']}); pins[value['path']] = value['sha256']
            for key, child in value.items():
                if key in ('source_pins', 'code_pins', 'artifact_pins', 'retained_artifact_pins') and isinstance(child, dict):
                    self.verify(child); pins.update(child)
                else:
                    self.pin_refs(child, pins)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.pin_refs(child, pins)

    def own_run(self, path):
        raw = Path(path) if Path(path).is_absolute() else self.root/path
        run = inside(raw, self.root)
        if raw.is_symlink() or run.parent != self.DEFAULT.parent or not run.name.startswith('cross_drug_'):
            raise ValueError('Use a dedicated immediate cross_drug_ experiment directory')
        return run

    def paper_pins(self):
        folders = [self.root/'results', self.root/'experiments/blueprint_design/vascular_tissue_ten_drug_20260930/paper',
                   self.root/'experiments/blueprint_design/blueprint_capability_20260930/paper']
        return {str(p.relative_to(self.root)): sha256(p) for folder in folders
                for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}

    def prepare(self, run, workers=14):
        run = self.own_run(run)
        if (run/'protocol.json').exists() or (run/'state.json').exists():
            raise FileExistsError('Preserve frozen experiment; use resume')
        if type(workers) is not int or not 1 <= workers <= 14:
            raise ValueError('Use one to fourteen workers')
        original = self.read(self.source); source_ref = self.ref(self.source)
        if len(original['cases']) != 20 or len({c['compound_id'] for c in original['cases']}) != 20:
            raise ValueError('Exactly twenty distinct clinical compounds are required')
        publication = self.paper_pins(); pins = {source_ref['path']: source_ref['sha256']}
        run.mkdir(parents=True, exist_ok=True)
        for folder in ('inputs', 'priors', 'predictions', 'teachers', 'target_physiology'):
            (run/folder).mkdir(exist_ok=True)
        policy = {'version': VERSION, 'elimination_scope': 'hepatic_only',
            'interpretation': 'Native assembly uses one lumped total plasma clearance; not a biological hepatic-clearance claim.',
            'clinical_net_renal_inputs_removed': True, 'fu_and_total_clearance': 'Existing twenty-compound/scaffold-excluded RF only',
            'partition_method': 'PKSimStandard', 'same_drug_observed_pk_not_supplied_as_construction_parameters': True}
        save(run/'inputs/build_policy.json', policy); policy_ref = self.ref(run/'inputs/build_policy.json')
        cases, removed_renal = [], []
        for item in original['cases']:
            metadata = ('case_id', 'compound_id', 'drug', 'study_id', 'cluster_id', 'cohort',
                'reference_kind', 'reference_statistic', 'physchem', 'reference_source', 'nca_policy',
                'transfer_group', 'sensitivity_flags')
            case = {key: deepcopy(item[key]) for key in metadata if key in item}
            cid = case['case_id']; folder = run/'inputs'/cid
            original_requests = [item['requests']['rf']['development'],
                item['requests']['rf'].get('target', item['requests']['evidence']['target'])]
            self.pin_refs(original_requests, pins)
            if isinstance(case.get('physchem'), str):
                self.pin_refs(self.ref(case['physchem']), pins)
            source_request = self.checked(original_requests[0])
            target_request = self.checked(original_requests[1])
            rows = [deepcopy(e) for e in source_request['evidence'] if e['parameter'] != 'renal_clearance']
            if len(rows) != len(source_request['evidence']):
                removed_renal.append(cid)
            rows = [e for e in rows if e['parameter'] != 'elimination_scope']
            rows.append({'parameter': 'elimination_scope', 'compound_id': case['compound_id'],
                'value': 'hepatic_only', 'unit': 'text', 'basis': 'assumption', 'source': policy_ref,
                'context': {}, 'notes': policy['interpretation']})
            values = {e['parameter']: e for e in rows}
            for name in ('fraction_unbound', 'total_clearance'):
                if values[name]['basis'] != 'predicted':
                    raise ValueError('Clinical fu/CL cannot enter a cross-drug request')
            folder.mkdir(parents=True, exist_ok=True)
            save(folder/'evidence.json', {'evidence': rows}); ledger = self.ref(folder/'evidence.json')
            development = deepcopy(source_request)
            development.update(evidence=rows, evidence_ledger=ledger, partition_method='PKSimStandard')
            development['assumptions'] = [policy['interpretation'],
                'fu and total plasma CL are RF predictions excluding all twenty compounds and scaffolds.',
                'Clinical renal split and old evidence-preservation claims are not used in this experiment.',
                'Representative adult and nominal IV protocol retain source-specific limitations; not individualized population reconstruction.',
                'No organ exposure, renal disease, active transport, DDI or oral validation is claimed.',
                'Cross-drug parameter learning excludes the heldout compound and all study-connected compounds.']
            target = deepcopy(development)
            for field in ('human', 'dosing', 'target'):
                target[field] = deepcopy(target_request[field])
            if self.teacher.clearance_policy(development)['rule'] != 'qualified_rf':
                raise ValueError('Cross-drug experiment requires qualified RF clearance')
            for request in (development, target):
                if assemble_evidence_blueprint(request, self.root)['status'] != 'assembled':
                    raise ValueError('RF-only native construction rejected: '+cid)
            save(folder/'requests.json', {'development': development, 'target': target})
            full_ref = item['sampling_sources']['full']; full = clean_training(self.checked(full_ref))
            if full['human'] != development['human'] or full['dosing'] != development['dosing']:
                raise ValueError('Development request does not match its clinical curve')
            indices = select_five(full); five = deepcopy(full)
            five['observations'] = [full['observations'][i] for i in indices]
            save(folder/'full.json', full); save(folder/'five.json', five)
            studies = sorted({str(s) for s in (full.get('study_id'), case['study_id'], case['cluster_id']) if s})
            if len(studies) < 2:
                raise ValueError('Development and target study separation must be explicit')
            case.update(all_study_ids=studies, cross_drug_requests=self.ref(folder/'requests.json'),
                full_training_source=self.ref(folder/'full.json', role='calibration_data_reconstruction'),
                five_training_source=self.ref(folder/'five.json', role='calibration_data_reconstruction'),
                five_selection={'rule': 'frozen time-only select_five', 'original_indices': indices,
                    'concentrations_used_for_selection': False, 'original_development_source': full_ref})
            # Discard the old per-method request map: no accidental use of a
            # measured renal prior or an old winner is possible in later stages.
            case.pop('requests', None)
            self.pin_refs([case, development, target, full_ref], pins)
            cases.append(case)
        folds = study_folds(cases)
        if len(folds) != 19:
            raise ValueError('Expected nineteen connected-study folds; review metadata before fitting')
        by_case = {cid: f['fold_id'] for f in folds for cid in f['heldout_case_ids']}
        for case in cases:
            case['cross_drug_cluster_id'] = by_case[case['case_id']]
        units = [{'unit_id': uid(c['case_id'], m, k), 'case_id': c['case_id'], 'method': m,
                  'condition': k, 'fold_id': by_case[c['case_id']]} for c in cases for m in METHODS for k in CONDITIONS]
        teacher_contract = {'version': self.teacher.VERSION, 'config': deepcopy(self.teacher.CONFIG),
                            'variant': 'tail_balanced', 'code_pins': self.teacher._code_pins(self.root)}
        pins.update(teacher_contract['code_pins'])
        for name in CODE:
            pins[name] = sha256(self.root/name)
        if hasattr(self.learner, '_code_pins'):
            pins.update(self.learner._code_pins(self.root))
        scorer_module = importlib.import_module('score_cross_drug_blueprint') if self.scorer is None else None
        for name in getattr(scorer_module, 'CODE_FILES', ()):
            pins[name] = sha256(self.root/name)
        protocol = {'version': VERSION, 'created_at': now(), 'source_protocol': source_ref,
            'cases': cases, 'folds': folds, 'units': units, 'methods': list(METHODS), 'conditions': list(CONDITIONS),
            'formula': 'PKSimStandard', 'teacher_contract': teacher_contract,
            'learner_contract': {'version': self.learner.VERSION, 'config': deepcopy(self.learner.CONFIG)},
            'source_pins': pins, 'paper_guard': publication, 'workers': workers,
            'native_threads': 2, 'native_timeout_seconds': 180, 'monitor_seconds': 1200,
            'directory_limit_bytes': 10*1024**3, 'removed_renal_case_ids': removed_renal,
            'budget': {'teacher_fits': 20, 'five_point_fits': 60, 'target_predictions': 120,
                'native_development_models': 20, 'native_target_models': 20, 'fixed_optimizer_starts_per_fit': 16,
                'max_evaluations_per_start': 400},
            'teacher_label_kind': 'Full-development fitted log scale factors, not measured PK labels',
            'learning_features': 'Static preclinical/chemical/RF input features only; no same-drug curve features',
            'resume': 'Completed teacher/prior/entry reuse; interrupted five-point fit restarts with fixed starts and explicitly unknown partial cost',
            'scope': 'Previously viewed twenty-drug development material, leave-compound-and-connected-study-out learning; not a new blind clinical test',
            'baseline_semantics': 'All three methods receive RF fu/CL, no measured renal split and one fixed formula; not numerically comparable to previous evidence-rich 1.64 baseline',
            'target_observations_used_for_selection': False, 'paper_updated': False, 'automatic_promotion': False}
        self.verify(pins)
        if self.paper_pins() != publication:
            raise ValueError('Publication changed during preparation')
        save(run/'protocol.json', protocol)
        save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': self.ref(run/'protocol.json'),
            'teachers': {}, 'target_physiology': {}, 'fold_priors': {}, 'prediction_bundles': {},
            'progress': {}, 'monitor': {}, 'all_predictions_locked_before_scoring': False})
        return {'status': 'prepared', 'drugs': 20, 'folds': 19, 'units': len(units), 'budget': protocol['budget']}

    def load(self, run, locked=False):
        run = self.own_run(run); state = self.read(run/'state.json'); protocol = self.checked(state['protocol'])
        if state['version'] != VERSION or protocol['version'] != VERSION:
            raise ValueError('Cross-drug version changed')
        self.verify(protocol['source_pins'])
        if self.paper_pins() != protocol['paper_guard']:
            raise ValueError('Publication files changed')
        cases = {c['case_id']: c for c in protocol['cases']}
        expected = {uid(cid, m, k) for cid in cases for m in METHODS for k in CONDITIONS}
        if (len(cases) != 20 or len(protocol['units']) != 120 or {u['unit_id'] for u in protocol['units']} != expected
                or protocol['folds'] != study_folds(protocol['cases'])):
            raise ValueError('Cross-drug matrix or study isolation changed')
        for key in ('teachers', 'target_physiology', 'prediction_bundles'):
            if not set(state[key]) <= set(cases):
                raise ValueError('Unexpected case in checkpoint')
            self.pin_refs(state[key], {})
        if not set(state['fold_priors']) <= {f['fold_id'] for f in protocol['folds']}:
            raise ValueError('Unexpected training fold in checkpoint')
        self.pin_refs(state['fold_priors'], {})
        if locked:
            if (not state.get('all_predictions_locked_before_scoring') or not state.get('monitor_stopped_before_scoring')
                    or set(state['prediction_bundles']) != set(cases) or state['status'] not in ('predictions_locked', 'complete')):
                raise ValueError('All 120 predictions must lock before clinical scoring')
            for cid, source in state['prediction_bundles'].items():
                bundle = self.checked(source)
                if (bundle['case_id'] != cid or bundle['status'] != 'locked'
                        or set(bundle['entries']) != {m+'__'+k for m in METHODS for k in CONDITIONS}
                        or any(e['status'] not in ('predicted', 'failed') for e in bundle['entries'].values())):
                    raise ValueError('Incomplete or altered prediction bundle')
        return protocol, state

    def execute(self, run):
        run = self.own_run(run)
        with (run/'.run.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol, state = self.load(run)
            if state['status'] == 'complete':
                return {'status': 'complete', 'reused': True}
            if (self.teacher.VERSION != protocol['teacher_contract']['version']
                    or self.teacher.CONFIG != protocol['teacher_contract']['config']
                    or self.learner.VERSION != protocol['learner_contract']['version']
                    or self.learner.CONFIG != protocol['learner_contract']['config']):
                raise ValueError('Learning/teacher algorithm contract changed')
            controller = self.controller_class(self, run, protocol, state)
            watcher = threading.Thread(target=controller.watch, daemon=True); watcher.start()
            cases = {c['case_id']: c for c in protocol['cases']}

            def phase(name, items, function):
                controller.persist(status=name)
                with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                    futures = [pool.submit(function, item) for item in items]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            if not controller.halted.is_set():
                                controller.pause(str(exc))
                controller.check(force=True)

            def teacher_one(case):
                controller.check(); cid = case['case_id']
                if cid in state['teachers']:
                    return
                req = self.checked(case['cross_drug_requests'])['development']
                folder = run/'teachers'/cid
                fit = self.teacher.fit_mapping(reference_request=req, training_source=case['full_training_source'],
                    output_dir=folder, project_root=self.root, variant='tail_balanced', before_trial=controller.check)
                self.teacher._validated(self.checked(fit['mapping']), self.root/fit['mapping']['path'], self.root)
                physiology = self.ref(folder/'physiology/perfusion_physiology.json')
                item = {'status': fit['status'], 'eligible': fit['eligible'], 'mapping': fit['mapping'],
                    'development_physiology': physiology, 'request_source': case['cross_drug_requests'],
                    'total_native_calls': fit['total_native_calls'], 'linear_solver_calls': fit['linear_solver_calls'],
                    'cost_count_exact': fit['cost_count_exact'], 'elapsed_seconds': fit['elapsed_seconds']}
                with controller.lock:
                    state['teachers'][cid] = item; controller.persist()

            def target_one(case):
                controller.check(); cid = case['case_id']
                if cid in state['target_physiology']:
                    return
                req = self.checked(case['cross_drug_requests'])['target']
                folder = run/'target_physiology'/cid
                # A partial native export is preserved for diagnosis; only one
                # explicit retry directory is used, never silent overwriting.
                if folder.exists() and not (folder/'physiology/perfusion_physiology.json').exists():
                    folder = folder.with_name(cid+'_retry_1')
                    if folder.exists() and not (folder/'physiology/perfusion_physiology.json').exists():
                        raise Paused('Two partial native target exports retained: '+cid)
                physiology = self.teacher._physiology(req, folder, self.root, controller.check)
                source = physiology.get('manifest') or self.ref(folder/'physiology/perfusion_physiology.json')
                with controller.lock:
                    state['target_physiology'][cid] = {'physiology': source,
                        'source_request': case['cross_drug_requests'], 'native_calls': 1,
                        'prior_partial_cost_unknown': folder.name.endswith('_retry_1')}
                    controller.persist()

            def prior_one(fold):
                controller.check(); fid = fold['fold_id']
                if fid in state['fold_priors']:
                    return
                rows, sources, excluded_failed = [], [], []
                excluded_compounds, excluded_studies = set(fold['excluded_compound_ids']), set(fold['excluded_study_ids'])
                # Filter identifiers BEFORE any mapping or teacher request is
                # decoded. A heldout teacher can never enter fit_prior inputs.
                allowed = [case for case in protocol['cases'] if case['compound_id'] not in excluded_compounds
                           and not excluded_studies.intersection(case['all_study_ids'])]
                if {c['case_id'] for c in allowed} != set(fold['training_case_ids']):
                    raise ValueError('Study closure differs from frozen training membership')
                for case in allowed:
                    record = state['teachers'][case['case_id']]
                    if not record['eligible']:
                        excluded_failed.append(case['case_id']); continue
                    mapping = self.checked(record['mapping'])
                    req = self.checked(case['cross_drug_requests'])['development']
                    rows.append({'compound_id': case['compound_id'], 'study_ids': case['all_study_ids'],
                                 'request': req, 'log_parameters': mapping['decision']['log_parameters']})
                    sources.extend([record['mapping'], case['cross_drug_requests']])
                learned = self.learner.fit_prior(rows, excluded_compound_ids=fold['excluded_compound_ids'],
                                                  excluded_study_ids=fold['excluded_study_ids'])
                bundle = {'version': VERSION, 'fold': fold, 'model': learned, 'source_refs': sources,
                    'attempted_teacher_count': len(allowed), 'successful_teacher_count': len(rows),
                    'failed_teacher_case_ids': excluded_failed, 'metadata_filtered_before_teacher_decode': True}
                path = run/'priors'/(fid+'.json'); save(path, bundle)
                with controller.lock:
                    state['fold_priors'][fid] = self.ref(path); controller.persist()

            def predict_case(case):
                controller.check(); cid = case['case_id']
                if cid in state['prediction_bundles']:
                    return
                fold = next(f for f in protocol['folds'] if cid in f['heldout_case_ids'])
                prior_ref = state['fold_priors'][fold['fold_id']]; prior = self.checked(prior_ref)
                requests = self.checked(case['cross_drug_requests'])
                dev_req, target_req = requests['development'], requests['target']
                target_source = state['target_physiology'][cid]['physiology']
                target_physiology = self.checked(target_source)
                context = {'version': VERSION, 'case_id': cid, 'prior_source': prior_ref,
                    'request_source': case['cross_drug_requests'], 'target_physiology_source': target_source,
                    'five_training_source': case['five_training_source'],
                    'development_physiology_source': state['teachers'][cid]['development_physiology'],
                    'target_outcomes_used': False, 'formula': 'PKSimStandard'}
                path = run/'predictions'/(cid+'.json')
                if path.exists():
                    bundle = self.read(path)
                    if bundle['construction'] != context:
                        raise ValueError('Incomplete bundle construction changed')
                    seals = state['progress'].get(cid, {}).get('entry_digests', {})
                    if not set(seals) <= set(bundle['entries']):
                        raise ValueError('Previously sealed prediction entry disappeared')
                    for key, digest in seals.items():
                        if object_digest(bundle['entries'][key]) != digest:
                            raise ValueError('Sealed partial prediction entry changed: '+cid+'/'+key)
                    # A shutdown can occur after writing an entry but before
                    # its state receipt. Preserve that unsealed diagnostic and
                    # recompute it; never reuse an unbound prediction.
                    for key in set(bundle['entries'])-set(seals):
                        bundle.setdefault('unsealed_entry_diagnostics', {})[key] = bundle['entries'].pop(key)
                        bundle['interrupted_entry_cost_unknown'] = True
                    if bundle.get('incomplete_entry'):
                        bundle['interrupted_entry_cost_unknown'] = True
                else:
                    bundle = {'version': VERSION, 'case_id': cid, 'status': 'running', 'construction': context,
                        'entries': {}, 'source_refs': [prior_ref, case['cross_drug_requests'], target_source,
                            case['five_training_source'], state['teachers'][cid]['development_physiology']],
                        'interrupted_entry_cost_unknown': False}
                for condition in CONDITIONS:
                    for method in METHODS:
                        controller.check(); key = method+'__'+condition
                        if key in bundle['entries']:
                            continue
                        bundle['incomplete_entry'] = key; save(path, bundle)
                        begun = time.monotonic()
                        center = self.learner.predict_prior(prior['model'], dev_req, method=method)
                        entry = {'unit_id': uid(cid, method, condition), 'method': method, 'condition': condition,
                            'status': 'failed', 'request': target_req, 'prior_source': prior_ref,
                            'prior_center': np.asarray(center, float).tolist(), 'target_observations_used': False,
                            'same_drug_full_curve_used': False, 'same_drug_teacher_mapping_decoded': False,
                            'native_calls': 0, 'linear_solver_calls': 0}
                        if condition == 'five':
                            # No full-development document or teacher mapping is
                            # opened in this branch; only five selected values.
                            sparse = self.checked(case['five_training_source'])
                            dev_phys = self.checked(state['teachers'][cid]['development_physiology'])
                            fitted = self.learner.fit_five(dev_phys, dev_req, sparse, center,
                                                          before_trial=controller.check)
                            entry['fit'] = fitted; entry['linear_solver_calls'] += fitted['linear_solver_calls']
                            eligible = fitted['eligible']; logs = fitted.get('log_parameters')
                        else:
                            eligible = True; logs = np.asarray(center, float).tolist()
                            entry['fit'] = None
                        if eligible:
                            try:
                                times = prediction_grid(target_req)
                                predicted = self.teacher._simulate(target_physiology, target_req, times,
                                    'tail_balanced', self.teacher.PARAMETERS['tail_balanced'], logs)
                                mw = next(float(e['value']) for e in target_req['evidence'] if e['parameter']=='molecular_weight')
                                curve = {'time_min': (times*60).tolist(),
                                    'concentration': (np.asarray(predicted['concentration_mg_l'])*1000/mw).tolist(),
                                    'time_unit': 'min', 'unit': 'umol/l', 'observation': predicted['observation']}
                                endpoints = summarize_extended_prediction(curve, target_req)
                                endpoints.update(development_curve_used=condition=='five',
                                    comparison_observations_used=False, endpoint_estimator_uses_observations=False,
                                    cross_drug_learning=method!='rf_only', same_drug_full_curve_used=False)
                                entry.update(status='predicted', curve=curve, endpoints=endpoints,
                                    log_parameters=list(logs), parameters=dict(zip(self.teacher.PARAMETERS['tail_balanced'], np.exp(logs).tolist())),
                                    mass_balance=predicted['mass_balance'],
                                    effective_total_clearance_l_h=predicted['effective_total_clearance_l_h'])
                            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                                entry['failure'] = {'category': 'scientific_failed', 'reason': str(exc)}
                            finally:
                                entry['linear_solver_calls'] += 1
                        else:
                            entry['failure'] = {'category': 'five_point_fit_failed', 'reason': 'No eligible fixed-start fit'}
                        entry['elapsed_seconds'] = time.monotonic()-begun
                        bundle['entries'][key] = entry; bundle['incomplete_entry'] = None; save(path, bundle)
                        with controller.lock:
                            state['progress'][cid] = {'entries_locked': len(bundle['entries']), 'latest': key,
                                'entry_digests': {name: object_digest(value) for name, value in bundle['entries'].items()}}
                            controller.persist()
                bundle.update(status='locked', locked_at=now()); save(path, bundle)
                with controller.lock:
                    state['prediction_bundles'][cid] = self.ref(path); controller.persist()

            try:
                phase('teacher_fitting', list(cases.values()), teacher_one)
                phase('target_physiology', list(cases.values()), target_one)
                phase('learning_priors', protocol['folds'], prior_one)
                phase('predicting', list(cases.values()), predict_case)
                controller.closed.set(); watcher.join(timeout=30)
                if watcher.is_alive():
                    raise Paused('Monitor did not stop before clinical scoring')
                controller.persist(status='predictions_locked', all_predictions_locked_before_scoring=True,
                    monitor_stopped_before_scoring=True, predictions_locked_at=now())
                self.report(run)
                controller.persist(status='complete', completed_at=now(), results=self.ref(run/'results.json'))
                return {'status': 'complete', 'units': 120}
            except Exception as exc:
                if not controller.halted.is_set():
                    controller.pause(str(exc))
                raise
            finally:
                controller.closed.set(); watcher.join(timeout=30)

    def report(self, run):
        run = self.own_run(run); protocol, state = self.load(run, locked=True)
        if state['status'] == 'complete':
            result = self.checked(state['results'])
            self.verify(result.get('source_pins', {}))
            return result
        score = self.scorer or importlib.import_module('score_cross_drug_blueprint').score
        result = score(run, project_root=self.root)
        if self.paper_pins() != protocol['paper_guard']:
            raise ValueError('Publication changed during scoring')
        if not (run/'results.json').exists():
            save(run/'results.json', result)
        return result

    def status(self, run):
        run = self.own_run(run); p, s = self.load(run)
        return {'status': s['status'], 'teachers_completed': len(s['teachers']), 'teachers_total': 20,
            'teacher_failures': sum(not r['eligible'] for r in s['teachers'].values()),
            'target_exports_completed': len(s['target_physiology']), 'folds_completed': len(s['fold_priors']),
            'folds_total': len(p['folds']), 'drug_bundles_locked': len(s['prediction_bundles']),
            'predictions_completed': sum(v['entries_locked'] for v in s['progress'].values()),
            'predictions_total': 120, 'monitor': s['monitor'], 'pause_reason': s.get('pause_reason'),
            'paper_updated': False}

    def background(self, run):
        run = self.own_run(run); self.load(run)
        pidfile = run/'pid.json'
        if pidfile.exists():
            import psutil
            old = self.read(pidfile)
            try:
                process = psutil.Process(old['pid'])
                if process.uids().real == os.getuid() and process.create_time() == old['create_time'] and process.is_running():
                    raise ValueError('This experiment already has a live recorded process')
            except psutil.NoSuchProcess:
                pass
        with (run/'run.log').open('ab') as log:
            environment = os.environ.copy()
            environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
            child = subprocess.Popen([sys.executable, '-B', str(self.root/'scripts/run_cross_drug_blueprint.py'),
                'run', '--run', str(run)], cwd=self.root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True, env=environment)
        import psutil
        save(pidfile, {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(),
            'script': 'scripts/run_cross_drug_blueprint.py', 'run': str(run.relative_to(self.root)), 'started_at': now()})
        return {'status': 'started', 'pid': child.pid, 'log': str(run/'run.log')}

    def cleanup(self, run):
        _, state = self.load(run, locked=True)
        if state['status'] != 'complete':
            raise ValueError('Cleanup requires completed locked results')
        return {'removed_count': 0, 'per_trial_files_generated': False,
            'reason': 'Compact teacher/start/prior/prediction bundles and native provenance are retained for reproducibility.'}

    def main(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=['prepare','run','status','report','pause','resume','cleanup'])
        parser.add_argument('--run', type=Path, default=self.DEFAULT)
        parser.add_argument('--workers', type=int, default=14)
        parser.add_argument('--background', action='store_true')
        args = parser.parse_args(); run = self.own_run(args.run)
        if args.command == 'prepare':
            result = self.prepare(run, args.workers)
        elif args.command == 'pause':
            self.load(run); (run/'PAUSE').write_text('User requested pause\n'); result = {'status':'pause_requested'}
        elif args.command in ('run','resume'):
            if args.command == 'resume' and (run/'PAUSE').exists():
                (run/'PAUSE').unlink()
            result = self.background(run) if args.background else self.execute(run)
        else:
            result = getattr(self, args.command)(run)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


def create_runner(**kwargs):
    return Runner(**kwargs)


if __name__ == '__main__':
    create_runner().main()
