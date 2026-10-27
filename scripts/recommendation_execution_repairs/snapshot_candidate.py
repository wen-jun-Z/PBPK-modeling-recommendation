"""Execute rigorously repaired JSON snapshots with the frozen clinical adapter.

This retains the registered adaptation/grid/readback procedure. Supplemental
same-commit license grants and audited schema repairs never change a catalog
candidate or choose a different source version. Complete project models use the
separate saved-model executor, avoiding snapshot reconstruction entirely.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import recommendation_candidate_executor as original
from recommendation_candidate_executor import (
    Blocked, _digest, _inside, _sha, _save, _name, _named, _parameters,
    _number, _source_options, load_source_model,
    adapt_source_model, apply_explicit_regimen, ensure_output_times,
    derive_administration_expectations, execute_snapshot, audit_execution,
    read_curve, prediction_from_curve,
)
from .license_audit import audit_candidate_license
from .snapshot_repair import IMPLEMENTATION_FILES as REPAIR_IMPLEMENTATION_FILES
from .complete_candidate import _food_state, _release, SEROQUEL_RELEASE_SOURCE
from .clinical_contract import scenario_contract

VERSION = 'catalog-repaired-snapshot-executor-v1'
EXECUTION_REVISION = 'engineering_repair_04'
IMPLEMENTATION_FILES = tuple(dict.fromkeys((*original.IMPLEMENTATION_FILES, *REPAIR_IMPLEMENTATION_FILES,
    'scripts/recommendation_candidate_migration.py',
    'src/pbpk_kb/model_runtime/R/catalog_snapshot_migration.R',
    'scripts/recommendation_execution_repairs/snapshot_candidate.py',
    'scripts/recommendation_execution_repairs/complete_candidate.py',
    'scripts/recommendation_execution_repairs/clinical_contract.py',
    'scripts/recommendation_execution_repairs/source_bound_dose.py',
    'scripts/recommendation_execution_repairs/json_parent_identity.py',
    'scripts/recommendation_execution_repairs/snapshot_repair.py',
    'scripts/recommendation_execution_repairs/snapshot_overlay.py',
    'scripts/recommendation_execution_repairs/snapshot_schema.py',
    'scripts/recommendation_execution_repairs/license_audit.py',
)))


def implementation_hashes():
    return {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}


def source_parent_identity(contract, source_sha256, *, project_root=ROOT):
    """Author-documented parent names, tied to immutable source and chemistry."""
    from .json_parent_identity import source_parent_identity as registered_identity
    identity=registered_identity(contract,source_sha256,project_root=project_root)
    if identity is not None:return identity
    if source_sha256 == '77d4db0edcb3811789c0289534b585ee286d3071c21f8a71cc92dd71b0816467':
        if contract.get('compound') != 'posaconazole': return None
        binding = contract.get('native_binding', {})
        source = binding.get('snapshot', {})
        name = 'Posaconazol ohne supersaturation'
        registration = contract.get('sampling_grid_registration', {})
        if (contract.get('case_id') != 'posaconazole_pmc10986696_fasted_noxafil100mg_be_subset'
                or source.get('sha256') != source_sha256 or binding.get('compound_name') != name
                or registration.get('source_protocol', {}).get('sha256') !=
                '24edd777280d91d929f7c836ae1bd8d8ffc1457bce8b0d3380393e1be0676e7e'
                or _sha(_inside(source.get('path'), project_root)) != source_sha256):
            raise Blocked('FROZEN_POSACONAZOLE_NATIVE_PARENT_BINDING_REQUIRED')
        return {'native_name': name, 'clinical_parent': 'posaconazole',
            'source_sha256': source_sha256, 'source_protocol': registration['source_protocol'],
            'basis': 'Exact frozen protocol names this source-specific non-supersaturation parent; all absorption and solubility mechanisms remain original. This alias is not a formulation-equivalence claim.',
            'source_compound_or_mechanism_renamed': False}
    if source_sha256 != '0635dcc6fd3f9f45c91b0f5ca480159926f0f57772c9af4e37ed78def0e5ea02':
        return None
    if contract.get('compound') != 'bupropion':return None
    conversion=contract.get('salt_conversion') or {}
    if (contract.get('case_id')!='bupropion_teaima2020_wellbutrin_sr_150mg' or
        conversion.get('parent_model_source',{}).get('sha256')!=source_sha256 or
        conversion.get('parent_molecular_weight_g_per_mol')!='239.741' or
        {r.get('compound_name') for r in conversion.get('parent_definition_evidence',[])}!={'Bupropion_FM'}):
        raise Blocked('SOURCE_BOUND_PARENT_ALIAS_CHEMISTRY_PROOF_REQUIRED')
    name='artifacts/pbpk-model-assets/public_collection_20260909/Clopidogrel-Bupropion-DDI__c44f59af9b46/source/README.md'
    digest='fbcbd84946d22db1220f5804c8b505c66c9d12be6de10fcaf286452b740c6219'
    if _sha(_inside(name,project_root))!=digest:raise Blocked('AUTHOR_PARENT_ALIAS_DOCUMENT_CHANGED')
    return {'native_name':'Bupropion_FM','clinical_parent':'bupropion','source_sha256':source_sha256,
        'author_document':{'path':name,'sha256':digest},
        'basis':'Author documents the published bupropion model; unique original Bupropion_FM parent and its239.741 g/mol definition verified before dose normalization.',
        'source_compound_or_mechanism_renamed':False}


def reference_source_options(snapshot, contract, *, source_sha256=None, reference_allowed=False):
    """Use the complete-model design policy without changing any source record.

    The legacy discovery function combines identity/route checks with a hard
    formulation-class gate. Only for explicitly permitted reference prediction
    is that discovery gate disabled; the actual target is retained below for
    release/class preference and explicit mismatch accounting. This never
    authorizes a formulation switch or makes the candidate directly applicable.
    """
    discovery = deepcopy(contract)
    identity=source_parent_identity(contract,source_sha256)
    if identity:discovery['compound']=original._name(identity['native_name'])
    target_food = _food_state(contract.get('food'))
    discovery['food'] = target_food
    if reference_allowed:
        discovery['formulation'] = None
    options, rejected = _source_options(snapshot, discovery, source_sha256=source_sha256)
    target_form = original._form(contract.get('formulation'))
    target_release = _release(contract.get('formulation'), contract['compound'])
    for option in options:
        simulation = _named(snapshot.get('Simulations'), option['name'], 'Simulations')
        selection = original._one_administered_selection(simulation)
        references = selection.get('Protocol', {}).get('Formulations', [])
        source_forms = [_named(snapshot.get('Formulations'), r['Name'], 'Formulations') for r in references]
        releases = sorted({_release(f.get('Name')) for f in source_forms})
        source_release = releases[0] if len(releases) == 1 else 'unknown'
        source_form = option['source_formulation_class']
        release_penalty = (0 if target_release == 'unknown' or target_release == source_release else
                           1 if {target_release, source_release} <= {'immediate_release', 'solution'} else
                           2 if source_release == 'unknown' else 3)
        source_food = _food_state(option['name'])
        form_penalty = 0 if target_form and source_form == target_form else 1
        food_penalty = 0 if target_food != 'unknown' and target_food == source_food else 1
        approximations = []
        if contract['route'] == 'Oral':
            if not target_form or source_form != target_form:
                approximations.append('source_formulation_class_differs_or_is_unresolved; reference_only')
            if target_release == 'unknown' or target_release != source_release:
                approximations.append('source_and_target_release_behavior_differs_or_is_unknown; reference_only')
        option['source_selection_design'] = {
            'reference_subject_approximation_explicitly_permitted': reference_allowed,
            'source_parent_identity': identity,
            'source_reported_dose': option.get('source_reported_dose'),
            'source_dose_basis': option.get('source_dose_basis'),
            'source_dose_distance_used_for_selection': option['source_dose_mg'] is not None,
            'original_source_formulations': deepcopy(source_forms),
            'target_formulation': deepcopy(contract.get('formulation')),
            'source_formulation_class': source_form, 'target_formulation_class': target_form,
            'source_release': source_release, 'target_release': target_release,
            'source_food_state': source_food, 'target_food_state': target_food,
            'source_formulation_changed_by_selection': False, 'strict_target_match': False,
            'approximations': approximations,
            'seroquel_brand_rule_source': SEROQUEL_RELEASE_SOURCE if contract['compound'] == 'quetiapine' and
                re.search(r'\bseroquel\b', str(contract.get('formulation', '')).lower()) else None,
            'selection_rule': 'exact identity/route and source regimen; release behavior, formulation class, food designation, log dose distance, source name',
        }
        option['sort_key'] = (release_penalty, form_penalty, food_penalty,
                              abs(math.log(option['source_dose_mg']/contract['dose_mg'])) if option['source_dose_mg'] is not None else math.inf, option['name'])
    return sorted(options, key=lambda row: row['sort_key']), rejected


def prepare_candidate(scenario, candidate, *, project_root=ROOT):
    """Produce a frozen executable plan or explicit operational blockers."""
    root = Path(project_root).resolve(strict=True)
    plan = {"version": VERSION, "status": "blocked", "candidate_id": candidate.get("candidate_id"),
        "execution_revision": EXECUTION_REVISION, "implementation_sha256": implementation_hashes(),
        "scenario_id": scenario.get("scenario_id", scenario.get("case_id")),
        "scenario_sha256": _digest(scenario), "candidate_sha256": _digest(candidate),
        "reasons": [], "scientific_inapplicability_inferred": False,
        "sealed_observations_accessed": False, "fitting_performed": False,
        "replacement_blueprint_constructed": False}
    try:
        contract = scenario_contract(scenario, project_root=root, candidate=candidate)
        plan["contract"] = contract
        if contract.get("salt_conversion"):
            conversion = contract["salt_conversion"]
            if _sha(_inside(conversion["source_file"], root)) != conversion["source_file_sha256"]:
                raise Blocked("DOSE_CONVERSION_SOURCE_CHECKSUM_MISMATCH")
        if not contract["case_id"] or not contract["compound"]:
            raise Blocked("CASE_AND_COMPOUND_ID_REQUIRED")
        registered = candidate.get("compounds", candidate.get("registered_compounds", []))
        if contract["compound"] not in {_name(c) for c in registered}:
            raise Blocked("TARGET_COMPOUND_NOT_REGISTERED_IN_CANDIDATE")
        prov = candidate.get("provenance", candidate.get("catalog_provenance", {}))
        if prov.get("license_status") != "allowed":
            license_check = audit_candidate_license(candidate, project_root=root)
            plan["license_audit"] = license_check
            if license_check.get("status") != "license_verified":
                raise Blocked("CANDIDATE_LICENSE_NOT_RESOLVED")
        original_path = _inside(prov.get("source_path", ""), root)
        if _sha(original_path) != prov.get("source_sha256"):
            raise Blocked("SOURCE_CHECKSUM_MISMATCH")
        path, execution_sha = original_path, prov["source_sha256"]
        if path.suffix == ".json" and path.stat().st_size <= 32 * 1024**2:
            snapshot = json.loads(path.read_text())
        else:
            snapshot = {}
        if path.suffix != ".json" or snapshot.get("Version") != 80:
            try:
                from recommendation_execution_repairs.snapshot_repair import resolve_repaired_snapshot
                migration = resolve_repaired_snapshot(prov, project_root=root)
            except (ValueError, OSError) as exc:
                raise Blocked("VERIFIED_NATIVE_MIGRATION_UNAVAILABLE:" + str(exc)) from exc
            path, execution_sha = migration["path"], migration["sha256"]
            plan["source_migration"] = {k:str(v) if isinstance(v,Path) else v for k,v in migration.items()}
            snapshot = json.loads(path.read_text())
        if path.stat().st_size > 32 * 1024**2:
            raise Blocked("SOURCE_EXCEEDS_32_MIB")
        if snapshot.get("Version") != 80:
            raise Blocked("NATIVE_SOURCE_NOT_VERSION_80")
        permission = contract['prediction_contract'].get('source_reference_subject_prediction_allowed')
        reference_allowed = permission is True or (permission is not False and
            scenario.get('execution_policy', {}).get('allow_reference_subject_approximation') is True)
        options, rejected = reference_source_options(snapshot, contract, source_sha256=prov["source_sha256"],
                                                      reference_allowed=reference_allowed)
        plan["excluded_source_simulations"] = rejected
        if not options:
            raise Blocked("NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT")
        chosen = options[0]
        plan['source_selection_design'] = chosen['source_selection_design']
        source = load_source_model(path, execution_sha, chosen["name"], root)
        request = {"simulation_name": "candidate_" + _digest([contract["case_id"], candidate["candidate_id"]])[:16],
            "compound_name": chosen["compound_name"], "dose": {"value": contract["dose_mg"], "unit": "mg"},
            "output_selections": [chosen["output_path"]]}
        if contract["infusion_minutes"] is not None:
            request["infusion_minutes"] = contract["infusion_minutes"]
        # A prior native binding may select an existing source formulation only
        # when it names this exact immutable source; never invent dissolution.
        binding = contract["native_binding"]
        bound_source = binding.get("source_snapshot", {})
        if (contract["route"] == "Oral" and bound_source.get("sha256") == prov["source_sha256"]
                and binding.get("formulation_name")):
            request["formulation_name"] = binding["formulation_name"]
        adapted = adapt_source_model(source, request, root)
        if adapted["status"] != "adapted":
            raise Blocked("SOURCE_ADAPTER_BLOCKED:" + ";".join(adapted.get("reasons", [])))
        compound = _named(adapted["snapshot"].get("Compounds"), chosen["compound_name"], "Compounds")
        mw = _parameters(compound).get("Molecular weight", {})
        if mw.get("Unit") != "g/mol":
            raise Blocked("EXPLICIT_PARENT_MOLECULAR_WEIGHT_G_PER_MOL_REQUIRED")
        molecular_weight = _number(mw.get("Value"), "molecular weight", positive=True)
        prediction_contract = contract["prediction_contract"]
        # Source subjects remain exact source subjects. Clinical aggregate
        # populations are never reconstructed by guessing unknown covariates.
        old_permission = prediction_contract.get("source_reference_subject_prediction_allowed")
        approximation_allowed = old_permission is True
        if old_permission is not False:
            approximation_allowed |= scenario.get("execution_policy", {}).get("allow_reference_subject_approximation") is True
        if not approximation_allowed:
            raise Blocked("REFERENCE_SUBJECT_APPROXIMATION_REQUIRES_EXPLICIT_PROTOCOL_PERMISSION")
        approximate = ["source_reference_subject_is_not_clinical_group_statistic",
            "target_covariates_not_fitted_or_silently_substituted"]
        approximate += contract.get('sampling_grid_registration', {}).get('approximations', [])
        approximate += chosen['source_selection_design']['approximations']
        approximate += contract.get('dose_normalization', {}).get('approximations', [])
        if contract["route"] == "Oral":
            approximate += ["source_formulation_not_verified_identical_to_clinical_product",
                "target_food_schedule_not_reconstructed"]
        regimen = apply_explicit_regimen(adapted['snapshot'], contract)
        if regimen['manifest']['changed']:
            approximate.append('source_administration_vehicle_parameters_reused_at_explicit_target_doses')
        # A fixed design-only budget retains all original source output points,
        # including dense single-dose schedules. Never thin or refit the grid.
        point_limit = 16384
        gridded = ensure_output_times(regimen["snapshot"], contract["absolute_sampling_times_min"], root,
                                      max_points=point_limit)
        final, grid = gridded["snapshot"], gridded["manifest"]
        if final["Compounds"] != source["snapshot"]["Compounds"]:
            raise ValueError("Compound model was modified")
        # Target values come from the clinical design; only native paths come
        # from the selected model. These checks are independent of outcomes.
        expectations = derive_administration_expectations(final)
        target_times = contract['administration_times_min']
        if expectations["expected_event_count"] != len(target_times):
            raise Blocked("NATIVE_EVENT_COUNT_DIFFERS_FROM_EXPLICIT_CLINICAL_SCHEDULE")
        for row in expectations["expectations"]:
            field = row["path"].split("|")[-1]
            match = re.search(r'\|Application_([1-9][0-9]*)\|ProtocolSchemaItem\|', row['path'])
            if not match or int(match[1]) > len(target_times):
                raise Blocked('NATIVE_EVENT_PATH_NOT_BOUND_TO_TARGET_SCHEDULE')
            if field == "Dose":
                row.update(value=contract["dose_mg"], unit="mg")
            elif field == "Start time":
                row.update(value=target_times[int(match[1]) - 1], unit="min")
            elif field == "Infusion time":
                row.update(value=contract["infusion_minutes"], unit="min")
        plan.update(status="ready", source_path=str(original_path.relative_to(root)), source_sha256=prov["source_sha256"],
            execution_source_path=str(path.relative_to(root)), execution_source_sha256=execution_sha,
            source_simulation=chosen["name"], selection_rule=chosen['source_selection_design']['selection_rule'],
            request=request, output_path=chosen["output_path"], molecular_weight_g_per_mol=molecular_weight,
            compound_definition_sha256=_digest(source["snapshot"]["Compounds"]),
            source_individual=chosen["individual"], compatibility={"stratum": "reference_subject_approximation",
                "strict_target_match": False, "approximations": approximate,
                "drug_identity_match": True, "route_match": True, "dose_match_planned": True},
            snapshot=final, snapshot_sha256=_digest(final), grid_audit=grid,
            explicit_regimen_registration=regimen['manifest'], regimen_diff=regimen['diff'],
            native_expectations=expectations, source_adaptation_checks=adapted["checks"],
            source_adaptation_diff=adapted["diff"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        plan["reasons"].append(type(exc).__name__ + ":" + str(exc))
    return plan


def execute_candidate(scenario, candidate, output_dir, *, project_root=ROOT, timeout_seconds=300):
    """Execute once and lock outputs, preserving every blocked/failed attempt."""
    root = Path(project_root).resolve(strict=True)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be 1..600")
    output = _inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError("Never overwrite candidate prediction attempts")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError("At least 2 GiB free space required before candidate execution")
    output.mkdir(parents=True)
    plan = prepare_candidate(scenario, candidate, project_root=root)
    _save(output / "execution_plan.json", plan)
    result = {"version": VERSION, "scenario_id": plan["scenario_id"], "candidate_id": plan["candidate_id"],
        "execution_revision": plan['execution_revision'], "implementation_sha256": plan['implementation_sha256'],
        "status": plan["status"], "reasons": plan["reasons"], "native_executed": False,
        "prediction_locked": False, "sealed_observations_accessed": False,
        "fitting_performed": False, "scientific_inapplicability_inferred": False,
        "execution_plan_sha256": _sha(output / "execution_plan.json")}
    try:
        if plan["status"] == "ready":
            execution = execute_snapshot(snapshot=plan["snapshot"], output_dir=output / "native", project_root=root,
                timeout_seconds=timeout_seconds, max_output_bytes=128 * 1024**2, cpu_threads=2)
            result.update(native_executed=True, native_status=execution["status"],
                execution_manifest=execution["manifest_path"])
            if not execution.get("engineering_accepted"):
                result.update(status="native_failed", reasons=[str(execution.get("stop_reason", "native engineering verification failed"))])
            else:
                audit = audit_execution(snapshot=plan["snapshot"], execution_manifest=execution,
                    output_dir=output / "readback", project_root=root, **plan["native_expectations"])
                _save(output / "readback_result.json", audit)
                if audit.get("accepted") is not True:
                    result.update(status="readback_failed", reasons=audit.get("reasons", []))
                else:
                    curve = read_curve(output / "native", execution, plan["output_path"])
                    prediction = prediction_from_curve(curve, plan)
                    _save(output / "prediction.json", prediction)
                    files = [output / "execution_plan.json", output / "prediction.json", output / "readback_result.json",
                        _inside(execution["manifest_path"], root)]
                    for collection in ("pkml_outputs", "curve_outputs"):
                        files.extend(_inside(output / "native" / item["path"], root) for item in execution[collection])
                    # Bind the entire actual native/readback chain as well as
                    # its summary. The scoring gate replays these source,
                    # request, process and numeric-readback bindings.
                    for private_directory in (output / 'native', output / 'readback'):
                        files.extend(_inside(p, root) for p in sorted(private_directory.rglob('*')) if p.is_file())
                    files = sorted(set(files))
                    lock = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                        "execution_revision": plan['execution_revision'], "implementation_sha256": plan['implementation_sha256'],
                        "scenario_id": plan["scenario_id"], "candidate_id": plan["candidate_id"],
                        "scenario_sha256": plan["scenario_sha256"], "candidate_sha256": plan["candidate_sha256"],
                        "source_sha256": plan["source_sha256"], "files": {str(p.relative_to(root)): _sha(p) for p in files},
                        "sealed_observations_accessed": False, "execution_verified": True,
                        "prediction_accuracy_evaluated": False, "compatibility": plan["compatibility"]}
                    if _sha(_inside(plan["source_path"], root)) != plan["source_sha256"]:
                        raise ValueError("Source changed during execution")
                    if implementation_hashes() != plan['implementation_sha256']:
                        raise ValueError('Execution implementation changed during prediction')
                    _save(output / "prediction_lock.json", lock)
                    result.update(status="prediction_locked", prediction_locked=True,
                        compatibility=plan["compatibility"], prediction_file=str((output / "prediction.json").relative_to(root)),
                        prediction_sha256=_sha(output / "prediction.json"),
                        prediction_lock=str((output / "prediction_lock.json").relative_to(root)),
                        prediction_lock_sha256=_sha(output / "prediction_lock.json"),
                        molecular_weight_g_per_mol=plan["molecular_weight_g_per_mol"],
                        source={"path": plan["source_path"], "sha256": plan["source_sha256"],
                            "simulation": plan["source_simulation"]},
                        strict_target_match=False, approximation_reason=plan["compatibility"]["approximations"],
                        curve=curve, predicted_endpoints=prediction["endpoints"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="failed", reasons=[type(exc).__name__ + ":" + str(exc)])
    _save(output / "result.json", result)
    return result
