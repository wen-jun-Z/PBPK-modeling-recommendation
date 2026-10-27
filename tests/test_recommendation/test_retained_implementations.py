"""V2 migrates all 24 V1 invariants to registry/profile fixtures.

Producer changes use the registered explicit path allowlist. Archive, actual
record, identity, protocol and numerical-worker rejection checks are retained.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

STAGED = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("retained_under_test", STAGED / "scripts/recommendation_execution_repairs/retained_implementations.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    def write(path, value):
        data = json.dumps(value, ensure_ascii=False, indent=2).encode() if not isinstance(value, bytes) else value
        target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        return {"path": path, "sha256": sha(data)}
    old = {}; current = {}; archive = []; changes = []
    for name in sorted(m.PRODUCER_PATHS | {"src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R"}):
        previous = ("original " + name).encode(); old[name] = sha(previous)
        after = ("new " + name).encode() if name in m.PRODUCER_PATHS else previous
        current[name] = sha(after); write(name, after)
        preserved = write("archive/r4b/code/" + name, previous)
        archive.append({"original": {"path": name, "sha256": old[name]}, "preserved": preserved})
        if name in m.PRODUCER_PATHS:
            changes.append({"path": name, "before_sha256": old[name], "after_sha256": current[name]})
    prep = write("archive/r4b/preparation.json", {"execution_revision": m.REVISION,
        "sealed_outcomes_accessed": False, "code": [x["original"] for x in archive]})
    base = {"scenario_id": "clinical:drug", "candidate_id": "candidate", "execution_revision": m.REVISION,
            "implementation_sha256": old, "sealed_observations_accessed": False}
    plan = {**deepcopy(base), "status": "blocked", "role": "plan"}
    result = {**deepcopy(base), "status": "blocked", "role": "result", "prediction_locked": False}
    job = {"job_id": "unit_r4b", "scenario_id": base["scenario_id"], "candidate_id": base["candidate_id"], "records": {}}
    def record_ref(role, record, jid="unit_r4b"):
        ref = write(m.OUT + "/native/" + jid + "/" + m.RECORD_NAMES[role], record)
        return {**ref, "canonical_sha256": m.canonical_sha256(record)}
    job["records"] = {"plan": record_ref("plan", plan), "result": record_ref("result", result)}
    manifest = {"schema_version": m.SCHEMA, "profile_id": "r4b", "execution_revision": m.REVISION,
        "sealed_observations_accessed": False, "producer_changes": changes, "code_archive": archive,
        "preparation": prep, "retained_jobs": [job]}
    profiles = [manifest]
    def profile_path(pid):
        return m.OUT + "/engineering_amendments/readback_path_followup_20261002/profiles/" + pid + ".json"
    def freeze(value=manifest, protocol=True):
        entries = [{"profile_id": profile["profile_id"],
                    "manifest": write(profile_path(profile["profile_id"]), profile)} for profile in profiles]
        registry = {"schema_version": m.REGISTRY_SCHEMA, "execution_revision": m.REVISION,
                    "sealed_observations_accessed": False, "profiles": entries}
        ref = write(m.REGISTRY, registry); monkeypatch.setattr(m, "REGISTRY_SHA", ref["sha256"])
        if protocol:
            write(m.OUT + "/protocol.json", {"execution_revision": m.REVISION,
                "inputs": [ref, *(entry["manifest"] for entry in entries)]})
        return ref
    freeze()
    return {"root": root, "manifest": manifest, "profiles": profiles, "plan": plan, "result": result,
            "old": old, "current": current, "write": write, "freeze": freeze,
            "record_ref": record_ref, "base": base, "profile_path": profile_path}


def bind(p, record=None):
    return m.binding_code(p["plan"] if record is None else record, p["current"], project_root=p["root"])


def test_current_implementation_needs_no_historical_registration(tmp_path):
    code = {"scripts/current.py": "a" * 64}
    assert m.binding_code({"implementation_sha256": code}, code, project_root=tmp_path) is code


def test_retained_plan_and_result_keep_their_exact_original_identity(prepared):
    p = prepared; old = deepcopy(p["plan"]); current = dict(p["current"])
    assert bind(p) == p["old"] and bind(p, p["result"]) == p["old"]
    assert p["plan"] == old and p["current"] == current


def test_successful_prediction_and_lock_are_registered_separately(prepared):
    p = prepared; job = p["manifest"]["retained_jobs"][0]
    for role in ("prediction", "prediction_lock"):
        record = {**deepcopy(p["base"]), "role": role}
        job["records"][role] = p["record_ref"](role, record)
    p["result"].update(status="prediction_locked", prediction_locked=True)
    job["records"]["result"] = p["record_ref"]("result", p["result"])
    p["freeze"]()
    assert bind(p, {**deepcopy(p["base"]), "role": "prediction"}) == p["old"]
    assert bind(p, {**deepcopy(p["base"]), "role": "prediction_lock"}) == p["old"]


def test_unregistered_record_is_rejected(prepared):
    record = {**prepared["plan"], "new_unregistered_field": True}
    with pytest.raises(ValueError, match="not uniquely registered"): bind(prepared, record)


def test_actual_retained_file_tampering_is_rejected(prepared):
    p = prepared; ref = p["manifest"]["retained_jobs"][0]["records"]["plan"]
    p["write"](ref["path"], {**p["plan"], "status": "ready"})
    with pytest.raises(ValueError, match="checksum differs"): bind(p)


def test_rehashing_tampered_file_without_matching_canonical_record_is_rejected(prepared):
    p = prepared; ref = p["manifest"]["retained_jobs"][0]["records"]["plan"]
    ref.update(p["write"](ref["path"], {**p["plan"], "status": "ready"})); p["freeze"]()
    with pytest.raises(ValueError, match="actual retained record content differs"): bind(p)


def test_archived_producer_tampering_is_rejected(prepared):
    p = prepared; p["write"](p["manifest"]["code_archive"][0]["preserved"]["path"], b"tampered")
    with pytest.raises(ValueError, match="checksum differs"): bind(p)


def test_current_admitted_producer_hash_drift_is_rejected(prepared):
    p = prepared; p["current"][next(iter(m.PRODUCER_PATHS))] = "b" * 64
    with pytest.raises(ValueError, match="after-hash differs"): bind(p)


def test_numerical_worker_change_cannot_use_retained_results(prepared):
    p = prepared; p["current"]["src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R"] = "b" * 64
    with pytest.raises(ValueError, match="unregistered producer drift"): bind(p)


@pytest.mark.parametrize("mode", ["missing", "wrong_hash", "duplicate"])
def test_missing_or_conflicting_protocol_registration_is_rejected(prepared, mode):
    p = prepared; ref = {"path": m.REGISTRY, "sha256": m.REGISTRY_SHA}
    refs = [] if mode == "missing" else [{**ref, "sha256": "b" * 64}] if mode == "wrong_hash" else [ref, ref]
    p["write"](m.OUT + "/protocol.json", {"execution_revision": m.REVISION, "inputs": refs})
    with pytest.raises(ValueError, match="uniquely bound"): bind(p)


def test_manifest_tampering_is_rejected(prepared):
    p = prepared; p["write"](p["profile_path"]("r4b"), {**p["manifest"], "unauthorized": True})
    with pytest.raises(ValueError, match="checksum differs"): bind(p)


def test_additional_producer_change_is_rejected(prepared):
    p = prepared; p["manifest"]["producer_changes"].append({"path": "native.R", "before_sha256": "a"*64, "after_sha256": "b"*64}); p["freeze"]()
    with pytest.raises(ValueError, match="bounded registered"): bind(p)


def test_missing_old_producer_archive_is_rejected(prepared):
    p = prepared; p["manifest"]["code_archive"] = [x for x in p["manifest"]["code_archive"] if x["original"]["path"] in m.PRODUCER_PATHS]; p["freeze"]()
    with pytest.raises(ValueError, match="absent from original archive"): bind(p)


def test_archive_before_hash_must_match_old_preparation(prepared):
    p = prepared; row = p["manifest"]["code_archive"][0]
    row["preserved"] = p["write"](row["preserved"]["path"], b"forged original"); row["original"]["sha256"] = row["preserved"]["sha256"]; p["freeze"]()
    with pytest.raises(ValueError, match="historical preparation"): bind(p)


def test_retained_record_must_remain_in_exact_job_directory(prepared):
    p = prepared; p["manifest"]["retained_jobs"][0]["records"]["plan"]["path"] = "arbitrary.json"; p["freeze"]()
    with pytest.raises(ValueError, match="record path/checksum"): bind(p)


def test_retained_identity_mismatch_is_rejected(prepared):
    p = prepared; p["manifest"]["retained_jobs"][0]["candidate_id"] = "different"; p["freeze"]()
    with pytest.raises(ValueError, match="record identity"): bind(p)


def test_duplicate_retained_job_is_rejected(prepared):
    p = prepared; p["manifest"]["retained_jobs"].append(deepcopy(p["manifest"]["retained_jobs"][0])); p["freeze"]()
    with pytest.raises(ValueError, match="duplicate retained job"): bind(p)


def test_incomplete_prediction_registration_is_rejected(prepared):
    p = prepared; p["manifest"]["retained_jobs"][0]["records"]["prediction"] = p["record_ref"]("prediction", p["base"]); p["freeze"]()
    with pytest.raises(ValueError, match="record set is incomplete"): bind(p)


def test_profile_refs_available_before_new_protocol_exists(prepared):
    p = prepared; (p["root"] / m.OUT / "protocol.json").unlink()
    refs = m.profile_refs(project_root=p["root"])
    assert refs[0] == {"path": m.REGISTRY, "sha256": m.REGISTRY_SHA}
    registry = json.loads((p["root"] / m.REGISTRY).read_text())
    expected = [{"path": m.REGISTRY, "sha256": m.REGISTRY_SHA},
                registry["profiles"][0]["manifest"], p["manifest"]["preparation"],
                *(x["preserved"] for x in p["manifest"]["code_archive"])]
    assert len(refs) == len(expected)
    assert {(x["path"], x["sha256"]) for x in refs} == {(x["path"], x["sha256"]) for x in expected}


def test_sealed_archive_reference_is_rejected(prepared):
    p = prepared; row = p["manifest"]["code_archive"][0]
    data = (p["root"] / row["preserved"]["path"]).read_bytes()
    row["preserved"] = p["write"]("sealed/never_open.py", data); p["freeze"]()
    with pytest.raises(ValueError, match="sealed resources"): bind(p)


def test_preparation_tampering_is_rejected(prepared):
    p = prepared; p["write"](p["manifest"]["preparation"]["path"], {"altered": True})
    with pytest.raises(ValueError, match="checksum differs"): bind(p)


def test_canonical_digest_is_independent_of_json_formatting(prepared):
    p = prepared; ref = p["manifest"]["retained_jobs"][0]["records"]["plan"]
    ref.update(p["write"](ref["path"], json.dumps(p["plan"], separators=(",", ":")).encode())); p["freeze"]()
    assert bind(p) == p["old"]


@pytest.fixture
def two_generations(prepared):
    p = prepared; old = {}; archive = []; changes = []
    changed = {'src/pbpk_kb/blueprints/design_runtime.py', 'src/pbpk_kb/blueprints/source_adaptation_audit.py', 'scripts/recommendation_execution_repairs/historical_source_proofs.py', 'src/pbpk_kb/blueprints/source_adaptation.py'}
    for name in sorted(p['old']):
        data = (('original ' if name in changed or name not in m.PRODUCER_PATHS else 'new ') + name).encode()
        old[name] = sha(data)
        saved = p['write']('archive/r4c/code/' + name, data)
        archive.append({'original': {'path': name, 'sha256': old[name]}, 'preserved': saved})
        if old[name] != p['current'][name]:
            changes.append({'path': name, 'before_sha256': old[name], 'after_sha256': p['current'][name]})
    prep = p['write']('archive/r4c/preparation.json', {'execution_revision': m.REVISION,
        'sealed_outcomes_accessed': False, 'code': [x['original'] for x in archive]})
    base = {**deepcopy(p['base']), 'scenario_id': 'clinical:other_drug', 'candidate_id': 'other_candidate',
            'implementation_sha256': old}
    plan = {**deepcopy(base), 'status': 'blocked', 'role': 'plan'}
    result = {**deepcopy(base), 'status': 'blocked', 'role': 'result', 'prediction_locked': False}
    job = {'job_id': 'unit_r4c', 'scenario_id': base['scenario_id'], 'candidate_id': base['candidate_id'],
        'records': {role: p['record_ref'](role, record, 'unit_r4c') for role, record in [('plan', plan), ('result', result)]}}
    profile = {'schema_version': m.SCHEMA, 'profile_id': 'r4c', 'execution_revision': m.REVISION,
        'sealed_observations_accessed': False, 'producer_changes': changes, 'code_archive': archive,
        'preparation': prep, 'retained_jobs': [job]}
    p['profiles'].append(profile); p['second'] = {'profile': profile, 'base': base, 'plan': plan, 'result': result, 'old': old}
    p['freeze']()
    return p


def test_two_generations_return_distinct_exact_original_producers(two_generations):
    p = two_generations
    assert bind(p) == p['old'] and bind(p, p['second']['plan']) == p['second']['old']
    assert p['old'] != p['second']['old']
    assert len(p['manifest']['producer_changes']) == len(m.PRODUCER_PATHS)
    assert len(p['second']['profile']['producer_changes']) == 4


def test_both_generations_can_retain_successful_predictions_without_relabelling(two_generations):
    p = two_generations; records = []
    for profile, base, old in [(p['manifest'], p['base'], p['old']),
                               (p['second']['profile'], p['second']['base'], p['second']['old'])]:
        job = profile['retained_jobs'][0]
        for role in m.RECORD_NAMES:
            record = {**deepcopy(base), 'role': role}
            if role == 'result': record.update(status='prediction_locked', prediction_locked=True)
            if role == 'plan': record['status'] = 'ready'
            job['records'][role] = p['record_ref'](role, record, job['job_id'])
            records.append((record, old))
    p['freeze']()
    for record, old in records:
        before = deepcopy(record)
        assert bind(p, record) == old and record == before


def test_cross_generation_producer_substitution_is_rejected_even_when_record_is_registered(two_generations):
    p = two_generations; record = {**p['plan'], 'implementation_sha256': p['second']['old']}
    p['manifest']['retained_jobs'][0]['records']['plan'] = p['record_ref']('plan', record); p['freeze']()
    with pytest.raises(ValueError, match='absent from original archive'): bind(p, record)


def test_cross_generation_archive_substitution_is_rejected(two_generations):
    p = two_generations; second = p['second']['profile']
    name = 'scripts/recommendation_candidate_executor.py'
    old_row = next(x for x in p['manifest']['code_archive'] if x['original']['path'] == name)
    row = next(x for x in second['code_archive'] if x['original']['path'] == name)
    row['preserved'] = deepcopy(old_row['preserved']); p['freeze']()
    with pytest.raises(ValueError, match='preserve original bytes'): bind(p, p['second']['plan'])


def test_duplicate_logical_candidate_across_profiles_is_rejected(two_generations):
    p = two_generations; second = p['second']; job = second['profile']['retained_jobs'][0]
    job.update(scenario_id=p['base']['scenario_id'], candidate_id=p['base']['candidate_id'])
    for role in ('plan', 'result'):
        second[role].update(scenario_id=job['scenario_id'], candidate_id=job['candidate_id'])
        job['records'][role] = p['record_ref'](role, second[role], job['job_id'])
    p['freeze']()
    with pytest.raises(ValueError, match='duplicated across profiles'): bind(p)


def test_profile_current_hashes_must_agree(two_generations):
    p = two_generations; p['second']['profile']['producer_changes'][0]['after_sha256'] = 'c'*64; p['freeze']()
    with pytest.raises(ValueError, match='disagree on current producer'): bind(p)


def test_omitted_current_change_is_not_implicitly_allowed(two_generations):
    p = two_generations; p['second']['profile']['producer_changes'].pop(); p['freeze']()
    with pytest.raises(ValueError, match='unregistered producer drift'): bind(p, p['second']['plan'])


def test_numerical_worker_cannot_be_added_to_two_change_profile(two_generations):
    p = two_generations
    p['second']['profile']['producer_changes'].append({'path': 'src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R',
        'before_sha256': p['old']['src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R'], 'after_sha256': 'c'*64})
    p['freeze']()
    with pytest.raises(ValueError, match='unregistered or duplicate producer change'): bind(p, p['second']['plan'])


def test_each_selected_profile_requires_its_own_protocol_binding(two_generations):
    p = two_generations; protocol_path = m.OUT+'/protocol.json'
    protocol = json.loads((p['root']/protocol_path).read_text())
    protocol['inputs'] = [r for r in protocol['inputs'] if r['path'] != p['profile_path']('r4c')]
    p['write'](protocol_path, protocol)
    assert bind(p) == p['old']
    with pytest.raises(ValueError, match='uniquely bound'): bind(p, p['second']['plan'])


def test_registry_tampering_is_rejected(two_generations):
    p = two_generations; path = p['root']/m.REGISTRY; value = json.loads(path.read_text());value['altered'] = True
    p['write'](m.REGISTRY, value)
    with pytest.raises(ValueError, match='checksum differs'): bind(p)


def test_one_profile_tampering_invalidates_the_registered_set(two_generations):
    p = two_generations; p['write'](p['profile_path']('r4c'), {**p['second']['profile'], 'altered': True})
    with pytest.raises(ValueError, match='checksum differs'): bind(p)


def test_registry_cannot_swap_profile_names(two_generations, monkeypatch):
    p = two_generations; path = p['root']/m.REGISTRY; value = json.loads(path.read_text())
    value['profiles'][0]['profile_id'], value['profiles'][1]['profile_id'] = 'r4c', 'r4b'
    ref = p['write'](m.REGISTRY, value); monkeypatch.setattr(m, 'REGISTRY_SHA', ref['sha256'])
    with pytest.raises(ValueError, match='profile identity differs'): bind(p)


def test_profile_refs_include_both_old_code_generations(two_generations):
    p = two_generations; refs = m.profile_refs(project_root=p['root'])
    registry = json.loads((p['root'] / m.REGISTRY).read_text())
    expected = [{'path': m.REGISTRY, 'sha256': m.REGISTRY_SHA},
                *(entry['manifest'] for entry in registry['profiles'])]
    for profile in p['profiles']:
        expected.append(profile['preparation'])
        expected.extend(row['preserved'] for row in profile['code_archive'])
    assert len(refs) == len(expected)
    assert {(x['path'], x['sha256']) for x in refs} == {(x['path'], x['sha256']) for x in expected}


@pytest.mark.parametrize('status', ['failed', 'native_failed', 'readback_failed', 'running'])
def test_unsuccessful_native_attempts_cannot_be_retained(prepared, status):
    p = prepared; p['result']['status'] = status
    p['manifest']['retained_jobs'][0]['records']['result'] = p['record_ref']('result', p['result']);p['freeze']()
    with pytest.raises(ValueError, match='cannot be retained'): bind(p)


def test_success_flag_cannot_override_failed_terminal_state(prepared):
    p = prepared; p['result']['prediction_locked'] = True
    p['manifest']['retained_jobs'][0]['records']['result'] = p['record_ref']('result', p['result']);p['freeze']()
    with pytest.raises(ValueError, match='cannot be retained'): bind(p)


def test_profile_cannot_retain_empty_change_set(prepared):
    p = prepared; p['manifest']['producer_changes'].clear();p['freeze']()
    with pytest.raises(ValueError, match='bounded registered'): bind(p)


def test_failure_receipt_control_change_requires_its_own_explicit_registration(two_generations):
    p = two_generations
    p['second']['profile']['producer_changes'] = [x for x in p['second']['profile']['producer_changes']
        if x['path'] != 'scripts/recommendation_execution_repairs/historical_source_proofs.py']
    p['freeze']()
    with pytest.raises(ValueError, match='unregistered producer drift'):
        bind(p, p['second']['plan'])

def test_source_adapter_change_requires_its_own_explicit_registration(two_generations):
    p=two_generations
    p['second']['profile']['producer_changes']=[x for x in p['second']['profile']['producer_changes']
        if x['path']!='src/pbpk_kb/blueprints/source_adaptation.py']
    p['freeze']()
    with pytest.raises(ValueError,match='unregistered producer drift'):
        bind(p,p['second']['plan'])


def test_third_generation_can_register_only_the_remaining_source_adapter_change(two_generations):
    p=two_generations; name='src/pbpk_kb/blueprints/source_adaptation.py'; archive=[]; old={}
    for path in sorted(p['current']):
        original=(('original ' if path==name or path not in m.PRODUCER_PATHS else 'new ')+path).encode()
        old[path]=sha(original)
        archive.append({'original':{'path':path,'sha256':old[path]},
                        'preserved':p['write']('archive/r4d/code/'+path,original)})
    prep=p['write']('archive/r4d/preparation.json',{'execution_revision':m.REVISION,
        'sealed_outcomes_accessed':False,'code':[row['original'] for row in archive]})
    base={**deepcopy(p['base']),'scenario_id':'clinical:third_drug','candidate_id':'third_candidate','implementation_sha256':old}
    job={'job_id':'unit_r4d','scenario_id':base['scenario_id'],'candidate_id':base['candidate_id'],'records':{}}
    rows=[]
    for role in m.RECORD_NAMES:
        row={**deepcopy(base),'role':role}
        if role=='plan':row['status']='ready'
        if role=='result':row.update(status='prediction_locked',prediction_locked=True)
        job['records'][role]=p['record_ref'](role,row,job['job_id']);rows.append(row)
    p['profiles'].append({'schema_version':m.SCHEMA,'profile_id':'r4d','execution_revision':m.REVISION,
        'sealed_observations_accessed':False,'producer_changes':[{'path':name,'before_sha256':old[name],'after_sha256':p['current'][name]}],
        'code_archive':archive,'preparation':prep,'retained_jobs':[job]})
    p['freeze']()
    assert bind(p)==p['old']
    assert bind(p,p['second']['plan'])==p['second']['old']
    for row in rows:assert bind(p,row)==old


ADAPTER = "scripts/recommendation_execution_repairs/snapshot_candidate.py"
NEW_HELPER = "scripts/recommendation_execution_repairs/json_parent_identity.py"


def declare_dependencies(p):
    p["manifest"]["adapter_dependencies"] = {ADAPTER: {
        "producer_sha256": p["old"][ADAPTER], "implementation_sha256": dict(p["old"])}}
    p["current"][NEW_HELPER] = "c" * 64
    p["freeze"]()


def profile_bind(p, record=None):
    return m.binding_profile(p["plan"] if record is None else record, p["current"],
        adapter_path=ADAPTER, current_required=set(p["current"]), project_root=p["root"])


def test_registered_historical_dependencies_survive_new_helper_without_forged_hashes(prepared):
    p=prepared; declare_dependencies(p)
    code,required=profile_bind(p)
    assert required == set(p["old"]) and NEW_HELPER not in required
    assert all(code[k] == v for k,v in p["old"].items())
    assert NEW_HELPER not in p["plan"]["implementation_sha256"]


def test_new_helper_cannot_be_waived_without_fixed_historical_dependency_declaration(prepared):
    p=prepared;p["current"][NEW_HELPER]="c"*64
    with pytest.raises(ValueError,match="dependencies are incomplete"):profile_bind(p)


@pytest.mark.parametrize("fault",["drop_dependency","wrong_hash","wrong_producer","unarchived","wrong_adapter","extra_field"])
def test_historical_dependency_declaration_cannot_weaken_or_change_original_set(prepared,fault):
    p=prepared;declare_dependencies(p);declaration=p["manifest"]["adapter_dependencies"][ADAPTER]
    if fault=="drop_dependency":declaration["implementation_sha256"].pop("src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R")
    elif fault=="wrong_hash":declaration["implementation_sha256"]["src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R"]="b"*64
    elif fault=="wrong_producer":declaration["producer_sha256"]="b"*64
    elif fault=="unarchived":declaration["implementation_sha256"][NEW_HELPER]="c"*64
    elif fault=="wrong_adapter":p["manifest"]["adapter_dependencies"]={"scripts/recommendation_execution_repairs/complete_candidate.py":declaration}
    else:declaration["bypass"]=True
    p["freeze"]()
    with pytest.raises(ValueError,match="dependency|dependencies|producer hash|identity"):profile_bind(p)


def test_registered_dependency_set_does_not_accept_modified_plan(prepared):
    p=prepared;declare_dependencies(p);value=deepcopy(p["plan"])
    value["implementation_sha256"].pop("src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R")
    with pytest.raises(ValueError,match="not uniquely registered"):profile_bind(p,value)


def test_even_registered_plan_must_equal_dependency_set(prepared):
    p=prepared;declare_dependencies(p);p["plan"]["implementation_sha256"].pop("src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R")
    p["manifest"]["retained_jobs"][0]["records"]["plan"]=p["record_ref"]("plan",p["plan"])
    p["freeze"]()
    with pytest.raises(ValueError,match="retained record dependency set differs"):profile_bind(p)


def test_current_records_still_require_current_full_dependency_set(prepared):
    p=prepared;declare_dependencies(p)
    value={"implementation_sha256":{k:v for k,v in p["current"].items() if k!=NEW_HELPER}}
    with pytest.raises(ValueError,match="dependencies are incomplete"):profile_bind(p,value)
    value["implementation_sha256"][NEW_HELPER]=p["current"][NEW_HELPER]
    assert profile_bind(p,value)==(p["current"],set(p["current"]))
