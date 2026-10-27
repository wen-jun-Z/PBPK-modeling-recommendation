"""Bounded native migration of catalog sources, before prediction-plan freeze.

JSON is loaded and exported by PK-Sim, never promoted by rewriting Version.
PK-Sim projects are exported then round-tripped without simulation or fitting.
Only immutable original-source and converted-source receipts enter execution.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprints.design_runtime import run_bounded_r
from pbpk_kb.blueprints.source_adaptation import _ROOT_NON_SOLVER, _SIM_NON_SOLVER

VERSION = "catalog-source-native-migration-v3"
R_SCRIPT = "src/pbpk_kb/model_runtime/R/catalog_snapshot_migration.R"
RUNTIME_ADAPTER = "src/pbpk_kb/blueprints/design_runtime.py"
OUT = "experiments/model_recommendation/execution_validation_20261001"
DEFAULT_MIGRATIONS = OUT + "/migrations_repair_03"
OLD_MIGRATIONS = OUT + "/migrations"
ARCHIVE = OUT + "/engineering_amendments/attempt_01/archive_manifest.json"


def inside(value, root=ROOT, *, exists=True):
    root = Path(root).resolve(strict=True)
    p = (root / value).resolve(strict=exists)
    if p == root or not p.is_relative_to(root) or any(x.casefold().startswith("sealed") for x in p.relative_to(root).parts):
        raise ValueError("Migration resource must be unsealed and inside this project")
    return p


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_receipt_code(receipt, root):
    if (receipt.get("python_adapter_sha256") != sha(Path(__file__)) or
            receipt.get("worker_script_sha256") != sha(root / R_SCRIPT) or
            receipt.get("runtime_adapter_sha256") != sha(root / RUNTIME_ADAPTER)):
        raise ValueError("MIGRATION_IMPLEMENTATION_CHANGED")


def save(path, value, *, replace=False):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if replace:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    else:
        with path.open("x") as handle:
            handle.write(text)


def clean_snapshot(snapshot):
    document = deepcopy({k:v for k,v in snapshot.items() if k not in _ROOT_NON_SOLVER})
    for simulation in document.get("Simulations", []):
        for key in _SIM_NON_SOLVER:
            simulation.pop(key, None)
        simulation["HasResults"] = False
    return document


def _normalize(value, key=""):
    """Canonicalize identified unordered records; never discard duplicates.

    Parameter values/units/mechanism selections remain fully compared. Lists
    without an established identity (dose schedules, curves, etc.) retain order.
    """
    if isinstance(value, dict):
        return {k:_normalize(v,k) for k,v in value.items() if k != "Version"}
    if isinstance(value, list):
        values = [_normalize(x) for x in value]
        identity_keys = None
        if values and all(isinstance(x, dict) and isinstance(x.get("Name"), str) for x in values):
            identity_keys = ("Name",)
        elif key == "Parameters" and values and all(isinstance(x, dict) and isinstance(x.get("Path"), str) for x in values):
            identity_keys = ("Path",)
        elif key == "Processes" and values and all(isinstance(x, dict) and any(
                x.get(k) is not None for k in ("Name", "MoleculeName", "SystemicProcessType", "InternalName", "Molecule")) for x in values):
            identity_keys = ("Name", "MoleculeName", "SystemicProcessType", "InternalName", "Molecule", "DataSource")
        elif key == "Alternatives" and values and all(isinstance(x, dict) and "GroupName" in x and "AlternativeName" in x for x in values):
            identity_keys = ("GroupName", "AlternativeName")
        elif key == "ExpressionProfiles" and values and all(isinstance(x, dict) and all(k in x for k in ("Molecule", "Species", "Category")) for x in values):
            identity_keys = ("Molecule", "Species", "Category", "Type")
        if identity_keys:
            # Sorting is multiset normalization, not dictionary conversion:
            # identical and conflicting duplicates retain their multiplicity.
            identities={}
            for row in values:
                identity=json.dumps([row.get(k) for k in identity_keys],sort_keys=True)
                identities.setdefault(identity,set()).add(json.dumps(row,sort_keys=True))
            # Different overrides at one parameter path may be order-sensitive
            # (last write wins). Do not normalize ambiguous duplicate identities.
            if all(len(records)==1 for records in identities.values()):
                values = sorted(values, key=lambda x: json.dumps([x.get(k) for k in identity_keys],sort_keys=True))
        elif key == "CalculationMethods" and all(isinstance(x, str) for x in values):
            values = sorted(values)
        return values
    if key == "Path" and isinstance(value,str) and value.startswith("Applications|"):
        return "Events|" + value.removeprefix("Applications|")
    return value


def compare_migration(source, converted):
    """Fail closed on unreviewed structural/numeric differences."""
    a, b = _normalize(clean_snapshot(source)), _normalize(clean_snapshot(converted))
    differences, scalar_checks, numeric_checks = [], 0, 0
    max_relative_error = 0.
    def compare(x,y,path):
        nonlocal scalar_checks,numeric_checks,max_relative_error
        if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
            numeric_checks += 1
            if not math.isfinite(x) or not math.isfinite(y):
                differences.append({"path":path,"kind":"nonfinite"});return
            error=abs(x-y)/max(abs(x),abs(y),1e-300)
            max_relative_error=max(max_relative_error,error)
            if not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-15):
                differences.append({"path":path,"kind":"numeric","source":x,"converted":y})
        elif isinstance(x,dict) and isinstance(y,dict):
            for k in sorted(x.keys()|y.keys()):
                if k not in x or k not in y:
                    differences.append({"path":path+"/"+k,"kind":"key_added_or_removed"})
                else:compare(x[k],y[k],path+"/"+k)
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):differences.append({"path":path,"kind":"list_length","source":len(x),"converted":len(y)})
            for i,(xx,yy) in enumerate(zip(x,y)):compare(xx,yy,path+"/"+str(i))
        else:
            scalar_checks+=1
            if type(x)!=type(y) or x!=y:differences.append({"path":path,"kind":"scalar","source":str(x)[:180],"converted":str(y)[:180]})
    compare(a,b,"")
    return {"accepted":not differences,"difference_count":len(differences),"differences":differences[:100],
        "scalar_checks":scalar_checks,"numeric_checks":numeric_checks,"max_numeric_relative_error":max_relative_error,
        "allowed_transformations":["native Version upgrade","Applications to Events path mapping",
            "multiset ordering of named records, path parameters, typed process selections and alternatives; multiplicity preserved",
            "ordering of calculation-method identifiers and identified expression profiles"],
        "numeric_tolerance":{"relative":1e-12,"absolute":1e-15},
        "compound_parameters_fitted":False,"scientific_equivalence_established":False}


def collect_tasks(audit_rows, *, project_root=ROOT):
    root=Path(project_root).resolve(strict=True)
    tasks={}
    for row in audit_rows:
        if row.get("identity_match") is not True:continue
        p=row.get("catalog_provenance",row.get("provenance",{}))
        if p.get("license_status")!="allowed":continue
        source=inside(p["source_path"],root)
        if source.suffix not in {".json",".pksim5"}:continue
        if source.suffix==".json":
            document=json.loads(source.read_text())
            if document.get("Version")==80:continue
        digest=p["source_sha256"]
        if len(digest)!=64 or any(x not in "0123456789abcdef" for x in digest):raise ValueError("Invalid source SHA")
        task=tasks.setdefault(digest,{"source_sha256":digest,"source_path":str(source.relative_to(root)),
            "format":source.suffix,"candidate_ids":[],"scenario_ids":[]})
        if row.get("candidate_id") not in task["candidate_ids"]:task["candidate_ids"].append(row.get("candidate_id"))
        sid=row.get("scenario_id",row.get("case_id"))
        if sid not in task["scenario_ids"]:task["scenario_ids"].append(sid)
    return [tasks[k] for k in sorted(tasks)]


def reference(path, root=ROOT):
    path=inside(path,root)
    return {"path":str(path.relative_to(root)),"sha256":sha(path)}


def checked(ref, root=ROOT):
    path=inside(ref["path"],root)
    if sha(path)!=ref["sha256"]:raise ValueError("MIGRATION_BOUND_RESOURCE_CHANGED:"+ref["path"])
    return path


def archived_attempt(task, *, project_root=ROOT):
    """Authenticate attempt 01 against its preserved preparation/code snapshot."""
    root=Path(project_root).resolve(strict=True)
    archive_path=inside(ARCHIVE,root)
    archive=json.loads(archive_path.read_text())
    if archive.get("outcome_results_absent") is not True or archive.get("ranking_lock_absent") is not True:
        raise ValueError("ARCHIVE_NOT_PRE_OUTCOME_REPAIR")
    files={r["original_path"]:r for r in archive["files"]}
    if len(files)!=len(archive["files"]):raise ValueError("ARCHIVE_DUPLICATE_PATH")
    bindings=[reference(archive_path,root)]
    def archived(name):
        item=files[name]
        path=checked({"path":item["archive_path"],"sha256":item["sha256"]},root)
        bindings.append(reference(path,root));return path
    old_python=archived("scripts/recommendation_candidate_migration.py")
    old_r=archived(R_SCRIPT)
    archived("src/pbpk_kb/blueprints/source_adaptation.py")
    archived("src/pbpk_kb/blueprints/design_runtime.py")
    summary_path=archived(OLD_MIGRATIONS+"/migration_summary.json")
    summary=json.loads(summary_path.read_text())
    entries=[r for r in summary["results"] if r["source_sha256"]==task["source_sha256"]]
    if len(entries)!=1:raise ValueError("EXACTLY_ONE_ARCHIVED_SOURCE_REQUIRED")
    row=entries[0]
    receipt_path=checked({"path":row["receipt_path"],"sha256":row["receipt_sha256"]},root)
    r=json.loads(receipt_path.read_text());bindings.append(reference(receipt_path,root))
    if (r.get("version")!="catalog-source-native-migration-v1" or r.get("source_sha256")!=task["source_sha256"]
            or r.get("source_path")!=task["source_path"] or r.get("python_adapter_sha256")!=sha(old_python)
            or r.get("worker_script_sha256")!=sha(old_r) or r.get("ode_run") is not False
            or r.get("fitting_performed") is not False or r.get("sealed_observations_accessed") is not False):
        raise ValueError("ARCHIVED_RECEIPT_BINDING_MISMATCH")
    source=inside(task["source_path"],root)
    if sha(source)!=task["source_sha256"]:raise ValueError("ARCHIVED_ORIGINAL_SOURCE_CHANGED")
    process_path=checked({"path":r["process_manifest"],"sha256":r["process_manifest_sha256"]},root)
    process=json.loads(process_path.read_text());bindings.append(reference(process_path,root))
    if process.get("script_sha256")!=sha(old_r):raise ValueError("ARCHIVED_NATIVE_SCRIPT_MISMATCH")
    if process.get("runtime_unchanged") is not True:raise ValueError("ARCHIVED_NATIVE_RUNTIME_CHANGED")
    directory=receipt_path.parent
    command=process.get("command",[])
    expected=[str(directory/"input"),str(directory/"project"),str(directory/"snapshot")]
    if command[-3:]!=expected:raise ValueError("ARCHIVED_NATIVE_DIRECTORIES_DIFFER")
    return {"receipt":r,"process":process,"directory":directory,"bindings":bindings,
        "receipt_reference":reference(receipt_path,root),"archived_python_reference":reference(old_python,root),
        "archived_worker_reference":reference(old_r,root)}


def one_json(directory):
    paths=list(directory.glob("*.json"))
    if len(paths)!=1:raise ValueError("Exactly one native snapshot required: "+str(directory))
    return paths[0]


def revalidate_old_artifact(task, *, project_root=ROOT):
    """Pure read-only revalidation; no native call and no change to old receipts.

    Return the actual before/after documents when retained. For a previously
    accepted binary source whose temporary first export was already removed,
    the verified archived receipt supplies that earlier equivalence chain.
    """
    root=Path(project_root).resolve(strict=True)
    prior=archived_attempt(task,project_root=root);r=prior["receipt"];p=prior["process"];d=prior["directory"]
    if r.get("status")=="converted":
        if (r.get("structural_comparison_passed") is not True or r.get("comparison",{}).get("accepted") is not True
                or p.get("status")!="completed" or p.get("returncode")!=0):
            raise ValueError("ARCHIVED_SUCCESS_NOT_ACCEPTED")
        path=checked({"path":r["converted_source_path"],"sha256":r["converted_source_sha256"]},root)
        prior["bindings"].append(reference(path,root));converted=clean_snapshot(json.loads(path.read_text()))
        if task["format"]==".json":
            original=clean_snapshot(json.loads(inside(task["source_path"],root).read_text()))
            scope="original JSON versus archived native converted snapshot, rechecked without a new conversion"
        else:
            original=deepcopy(converted)
            scope="previous original-project equivalence established by authenticated archived receipt; current artifact identity revalidation only"
        prior.update(original=original,converted=converted,comparison=compare_migration(original,converted),
                     disposition="revalidated_prior_success",comparison_scope=scope,
                     inherited_original_comparison=r["comparison"] if task["format"]==".pksim5" else None)
        return prior
    if p.get("status")=="completed" and p.get("returncode")==0 and r.get("comparison"):
        path=one_json(d/"snapshot");prior["bindings"].append(reference(path,root))
        if task["format"]==".json":original=clean_snapshot(json.loads(inside(task["source_path"],root).read_text()))
        else:
            first=one_json(d/"first_native_export");prior["bindings"].append(reference(first,root))
            original=clean_snapshot(json.loads(first.read_text()))
        converted=clean_snapshot(json.loads(path.read_text()))
        prior.update(original=original,converted=converted,comparison=compare_migration(original,converted),
                     disposition="rechecked_prior_comparison_failure",
                     comparison_scope="retained original/native snapshot documents rechecked with strict identified-collection multiset equivalence")
        return prior
    prior["disposition"]="retry_staged_timeout" if p.get("stop_reason")=="EXECUTION_TIMEOUT" else "previous_export_failure_not_retried"
    return prior


def _native_stage(mode, input_path, directory, result, *, root, timeout_seconds, deadline, stop_file=None):
    if stop_file and stop_file.exists():raise InterruptedError("Migration stop requested between native stages")
    remaining=int(deadline-time.monotonic())
    if remaining<1:raise ValueError("MIGRATION_TOTAL_SOURCE_TIMEOUT")
    if shutil.disk_usage(root).free<20*1024**3:raise OSError("Migration free disk below 20 GiB")
    index=len(result["stages"])+1
    stage=directory/("stage_%02d_"%index+mode);stage.mkdir()
    inp=stage/"input";out=stage/"output";inp.mkdir();out.mkdir()
    suffix=".json" if mode=="snapshot_to_project" else ".pksim5"
    if input_path.suffix!=suffix:raise ValueError("Native stage input format differs")
    input_ref=reference(input_path,root);local=inp/("model"+suffix);shutil.copyfile(input_path,local)
    copied_before=sha(local)
    if copied_before!=input_ref["sha256"]:raise ValueError("Native stage input copy changed")
    process=run_bounded_r(script=root/R_SCRIPT,args=[mode,str(inp),str(out)],output_dir=stage,
        project_root=root,timeout_seconds=min(timeout_seconds,remaining),cpu_threads=2,max_output_bytes=512*1024**2,
        stop_file=stop_file)
    copied_after=sha(local);source_after=sha(input_path)
    row={"mode":mode,"input":input_ref,"copied_input_sha256":copied_after,
         "copied_input_sha256_before":copied_before,"copied_input_sha256_after":copied_after,
         "copied_input_changed":copied_before!=copied_after,
         "source_input_sha256_after":source_after,"source_input_unchanged":source_after==input_ref["sha256"],
         "working_copy_policy":"Native PK-Sim may maintain its isolated SQLite project copy; original inputs remain immutable and full snapshot equivalence is still required",
         "process_manifest":reference(stage/"process_manifest.json",root),"output":None,
         "status":process["status"],"stop_reason":process.get("stop_reason")}
    result["stages"].append(row);result["native_invocations"]+=1
    result["process_manifest"]=str((stage/"process_manifest.json").relative_to(root))
    result["process_manifest_sha256"]=sha(stage/"process_manifest.json")
    if source_after!=input_ref["sha256"]:raise ValueError("Native stage source changed")
    if suffix==".json" and copied_before!=copied_after:raise ValueError("Native stage snapshot input copy changed")
    if process["status"]!="completed" or process.get("returncode")!=0:
        raise ValueError("NATIVE_STAGE_FAILED:"+mode+":"+str(process.get("stop_reason")))
    outputs=list(out.glob("*.pksim5" if mode=="snapshot_to_project" else "*.json"))
    if len(outputs)!=1:raise ValueError("Native stage must produce exactly one output")
    row["output"]=reference(outputs[0],root)
    return outputs[0]


def _continue_or_retry(task, prior, directory, result, *, root, timeout_seconds, stop_file=None):
    """At most 600 seconds per phase and 1,200 seconds total per source."""
    deadline=time.monotonic()+1200
    source=inside(task["source_path"],root);old=prior["directory"]
    if sha(source)!=task["source_sha256"]:raise ValueError("ORIGINAL_SOURCE_CHANGED_BEFORE_NATIVE_RETRY")
    bindings=prior["bindings"]
    result["comparison_scope"]=prior["receipt"].get("comparison_scope","native staged source-preserving migration")
    if source.suffix==".json":
        original=clean_snapshot(json.loads(source.read_text()))
    else:
        exports=list((old/"first_native_export").glob("*.json"))
        reuse_export=False
        if len(exports)==1:
            copied=inside(old/"input/model.pksim5",root)
            copied_ref=reference(copied,root)
            reuse_export=copied_ref["sha256"]==task["source_sha256"]
            result["archived_first_export_review"]={
                "working_copy":copied_ref,"original_source_sha256":task["source_sha256"],
                "working_copy_matches_original":reuse_export,
                "old_export_reused":reuse_export,
                "decision":"reuse_exact_original_copy_export" if reuse_export else "fresh_native_export_from_verified_original",
                "reason":None if reuse_export else "Archived PK-Sim working copy differs; do not infer its export provenance from the changed copy"}
            # Native opening may maintain the SQLite work copy. A changed copy
            # neither proves original-source corruption nor authenticates the
            # archived export: regenerate from an exact fresh original copy.
            bindings.append(copied_ref)
            if reuse_export:
                first=exports[0];bindings.append(reference(first,root))
                result["continued_from"]="archived_first_project_export"
        if not reuse_export:
            first=_native_stage("project_to_snapshot",source,directory,result,root=root,
                                timeout_seconds=timeout_seconds,deadline=deadline,stop_file=stop_file)
            result["continued_from"]="fresh_verified_original_project_export"
        original=clean_snapshot(json.loads(first.read_text()))
    project_files=list((old/"project").glob("*.pksim5"))
    log=old/"execution.log"
    phase_complete=source.suffix==".json" and len(project_files)==1 and log.exists() and "\nNULL\n" in log.read_text()
    if phase_complete:
        project=project_files[0]
        # Integrity checking is read-only; no model data table is queried.
        with sqlite3.connect(project.as_uri()+"?mode=ro",uri=True) as database:
            if database.execute("PRAGMA quick_check").fetchone()!=('ok',):raise ValueError("ARCHIVED_PROJECT_INTEGRITY_FAILED")
        old_input=inside(old/"input/model.json",root)
        if json.loads(old_input.read_text())!=original:raise ValueError("ARCHIVED_PROJECT_INPUT_DIFFERS_FROM_SOURCE")
        bindings.extend([reference(project,root),reference(log,root),reference(old_input,root)])
        result["continued_from"]="archived_completed_snapshot_to_project_stage"
    else:
        clean=directory/"source_solver_snapshot.json";save(clean,original)
        project=_native_stage("snapshot_to_project",clean,directory,result,root=root,
                              timeout_seconds=timeout_seconds,deadline=deadline,stop_file=stop_file)
    converted_path=_native_stage("project_to_snapshot",project,directory,result,root=root,
                                timeout_seconds=timeout_seconds,deadline=deadline,stop_file=stop_file)
    converted=clean_snapshot(json.loads(converted_path.read_text()))
    return original,converted


def migrate_one(task, directory, *, project_root=ROOT, timeout_seconds=600, stop_file=None):
    root=Path(project_root).resolve(strict=True);directory=inside(directory,root,exists=False)
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=600:raise ValueError("At most 600 seconds per native phase")
    if directory.exists():
        receipt=directory/"receipt.json"
        if not receipt.exists():raise ValueError("Existing migration attempt is incomplete; preserve it before another attempt")
        old=json.loads(receipt.read_text())
        if old.get("source_sha256")!=task["source_sha256"]:raise ValueError("Migration source changed")
        verify_receipt_code(old,root)
        if old.get("status")=="converted":resolve_migrated_source(task,project_root=root,migration_root=directory.parent)
        return old
    if shutil.disk_usage(root).free<20*1024**3:raise OSError("Migration requires 20 GiB free project filesystem space")
    directory.mkdir(parents=True)
    result={"version":VERSION,**task,"status":"failed","reasons":[],"ode_run":False,"fitting_performed":False,
        "sealed_observations_accessed":False,"created_at":datetime.now(timezone.utc).isoformat(),
        "worker_script_sha256":sha(root/R_SCRIPT),"python_adapter_sha256":sha(Path(__file__)),
        "runtime_adapter_sha256":sha(root/RUNTIME_ADAPTER),
        "native_invocations":0,"stages":[],"provenance_refs":[],"operation":"archived_artifact_revalidation",
        "timeout_seconds_per_stage":timeout_seconds,"max_seconds_per_source":1200}
    try:
        prior=revalidate_old_artifact(task,project_root=root)
        result["provenance_refs"]=prior["bindings"]
        result["prior_receipt"]=prior["receipt_reference"]
        result["prior_disposition"]=prior["disposition"]
        result["process_manifest"]=prior["receipt"]["process_manifest"]
        result["process_manifest_sha256"]=prior["receipt"]["process_manifest_sha256"]
        if "comparison" in prior:
            original,converted=prior["original"],prior["converted"]
            result["comparison"]=prior["comparison"];result["comparison_scope"]=prior["comparison_scope"]
            if prior.get("inherited_original_comparison"):
                result["inherited_original_comparison"]=prior["inherited_original_comparison"]
        elif prior["disposition"]=="retry_staged_timeout":
            result["operation"]="bounded_native_stage_retry_or_continuation"
            original,converted=_continue_or_retry(task,prior,directory,result,root=root,
                timeout_seconds=timeout_seconds,stop_file=stop_file)
            result["comparison"]=compare_migration(original,converted)
        else:
            raise ValueError("ARCHIVED_NATIVE_EXPORT_FAILURE_NOT_BLINDLY_RETRIED")
        if not result["comparison"]["accepted"]:raise ValueError("UNREVIEWED_NATIVE_MIGRATION_DIFFERENCES")
        if converted.get("Version")!=80:raise ValueError("Native artifact must be Version 80")
        save(directory/"converted_snapshot.json",converted)
        result.update(status="converted",converted_source_path=str((directory/"converted_snapshot.json").relative_to(root)),
            converted_source_sha256=sha(directory/"converted_snapshot.json"),source_version=original.get("Version"),
            converted_version=converted.get("Version"),source_simulation_names=[x["Name"] for x in original["Simulations"]],
            native_roundtrip_simulation_names=[x["Name"] for x in converted["Simulations"]],structural_comparison_passed=True)
        if sha(inside(task["source_path"],root))!=task["source_sha256"]:raise ValueError("Original source changed during migration")
        # New stage metadata/logs and raw outputs remain bound. Only redundant
        # own input copies are removed; old attempts and original assets stay.
        removed=[]
        for path in sorted(directory.glob("stage_*/input")):
            if path.is_dir() and not path.is_symlink():removed.append(str(path.relative_to(directory)));shutil.rmtree(path)
        result["own_intermediate_input_folders_removed"]=removed
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error,InterruptedError) as exc:
        result.update(status="failed",reasons=[type(exc).__name__+":"+str(exc)])
    result["completed_at"]=datetime.now(timezone.utc).isoformat()
    save(directory/"receipt.json",result)
    return result


def prepare_migrations(cohort_scenarios, audit_rows, output_dir, *, project_root=ROOT,
                       workers=2, timeout_seconds=600, max_total_bytes=4*1024**3,
                       max_seconds=7200, stop_file=None):
    """Account for every source, with at most two bounded native workers."""
    root=Path(project_root).resolve(strict=True)
    if workers not in {1,2} or type(timeout_seconds) is not int or not 1<=timeout_seconds<=600:
        raise ValueError("At most two workers and 600 seconds per native stage are permitted")
    output=inside(output_dir,root,exists=False);output.mkdir(parents=True,exist_ok=True)
    stop=inside(stop_file,root,exists=False) if stop_file else None
    tasks=collect_tasks(audit_rows,project_root=root)
    started=time.monotonic();summary={"version":VERSION,"status":"running","expected_sources":len(tasks),
        "workers":workers,"timeout_seconds_per_stage":timeout_seconds,"max_seconds_per_source":1200,"max_total_bytes":max_total_bytes,
        "max_seconds":max_seconds,"scenario_count":len(cohort_scenarios),"results":[],
        "stop_reason":None,"ode_run":False,"sealed_observations_accessed":False}
    pending=list(tasks);active={}
    def used():
        total=0
        for path in output.rglob("*"):
            try:
                if path.is_symlink():raise ValueError("Migration output symlink is forbidden")
                if path.is_file():total+=path.stat().st_size
            except FileNotFoundError:
                # Bounded workers may remove their own redundant input copies.
                continue
        return total
    def receipt_row(task,result):
        path=output/task["source_sha256"]/"receipt.json"
        return {"source_sha256":task["source_sha256"],"candidate_ids":task["candidate_ids"],
            "status":result["status"],"reasons":result.get("reasons",[]),
            "receipt_path":str(path.relative_to(root)) if path.is_file() else None,
            "receipt_sha256":sha(path) if path.is_file() else None}
    save(output/"migration_summary.json",summary,replace=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            reason=("stop_requested" if stop and stop.exists() else
                "total_time_limit" if time.monotonic()-started>max_seconds else
                "total_output_limit" if used()>max_total_bytes else
                "free_disk_below_20GiB" if shutil.disk_usage(root).free<20*1024**3 else None)
            if reason and summary["stop_reason"] is None:summary["stop_reason"]=reason
            while pending and len(active)<workers and not summary["stop_reason"]:
                task=pending.pop(0)
                future=pool.submit(migrate_one,task,output/task["source_sha256"],project_root=root,timeout_seconds=timeout_seconds,stop_file=stop)
                active[future]=task
            if not active:break
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)
                try:result=future.result()
                except Exception as exc:result={"status":"failed","reasons":[type(exc).__name__+":"+str(exc)]}
                summary["results"].append(receipt_row(task,result))
                summary.update(completed_sources=len(summary["results"]),elapsed_seconds=time.monotonic()-started,output_bytes=used())
                save(output/"migration_summary.json",summary,replace=True)
    for task in pending:
        summary["results"].append(receipt_row(task,{"status":"not_started","reasons":[summary["stop_reason"]]}))
    summary.update(status="accounted" if not pending else "stopped",completed_sources=sum(r["status"]!="not_started" for r in summary["results"]),
        successful_sources=sum(r["status"]=="converted" for r in summary["results"]),
        failed_sources=sum(r["status"]=="failed" for r in summary["results"]),
        elapsed_seconds=time.monotonic()-started,output_bytes=used())
    save(output/"migration_summary.json",summary,replace=True)
    return summary


def resolve_migrated_source(provenance, *, project_root=ROOT, migration_root=None):
    """Read and authenticate a completed receipt; never perform conversion."""
    root=Path(project_root).resolve(strict=True)
    base=migration_root or root/DEFAULT_MIGRATIONS
    receipt=inside(Path(base)/provenance["source_sha256"]/"receipt.json",root)
    r=json.loads(receipt.read_text())
    verify_receipt_code(r, root)
    if (r.get("version")!=VERSION or r.get("status")!="converted" or r.get("source_sha256")!=provenance["source_sha256"]
            or r.get("structural_comparison_passed") is not True or r.get("comparison",{}).get("accepted") is not True
            or r.get("ode_run") is not False or r.get("fitting_performed") is not False):
        raise ValueError("MIGRATION_RECEIPT_NOT_ACCEPTED")
    original=inside(provenance["source_path"],root)
    converted=inside(r["converted_source_path"],root)
    process_path=inside(r["process_manifest"],root)
    if sha(original)!=r["source_sha256"] or sha(converted)!=r["converted_source_sha256"] or sha(process_path)!=r["process_manifest_sha256"]:
        raise ValueError("MIGRATION_ARTIFACT_CHECKSUM_MISMATCH")
    for ref in r.get("provenance_refs",[]):checked(ref,root)
    process=json.loads(process_path.read_text())
    if process.get("status")!="completed" or process.get("returncode")!=0:
        raise ValueError("MIGRATION_PROCESS_NOT_SUCCESSFUL")
    if r.get("native_invocations")==0:
        prior=archived_attempt(provenance,project_root=root)
        if r.get("operation")!="archived_artifact_revalidation" or r.get("prior_receipt")!=prior["receipt_reference"]:
            raise ValueError("MIGRATION_REVALIDATION_CHAIN_DIFFERS")
        if process.get("script_sha256")!=prior["receipt"]["worker_script_sha256"]:
            raise ValueError("MIGRATION_ARCHIVED_PROCESS_SCRIPT_DIFFERS")
    else:
        if r.get("native_invocations")!=len(r.get("stages",[])) or not r.get("stages"):
            raise ValueError("MIGRATION_STAGE_COUNT_DIFFERS")
        for stage in r["stages"]:
            stage_process=json.loads(checked(stage["process_manifest"],root).read_text())
            if (stage_process.get("status")!="completed" or stage_process.get("returncode")!=0
                    or stage_process.get("script_sha256")!=r["worker_script_sha256"]):
                raise ValueError("MIGRATION_STAGE_NOT_SUCCESSFUL")
            if (stage.get("copied_input_sha256_before")!=stage["input"]["sha256"]
                    or stage.get("source_input_sha256_after")!=stage["input"]["sha256"]
                    or stage.get("source_input_unchanged") is not True
                    or stage.get("copied_input_sha256_after")!=stage.get("copied_input_sha256")
                    or not isinstance(stage.get("copied_input_sha256_after"),str)
                    or len(stage["copied_input_sha256_after"])!=64
                    or stage.get("copied_input_changed")!=(stage["copied_input_sha256_before"]!=stage["copied_input_sha256_after"])):
                raise ValueError("MIGRATION_STAGE_INPUT_BINDING_DIFFERS")
            if stage.get("mode")=="snapshot_to_project" and stage["copied_input_changed"]:
                raise ValueError("MIGRATION_STAGE_SNAPSHOT_INPUT_CHANGED")
            checked(stage["input"],root);checked(stage["output"],root)
    return {"path":converted,"sha256":r["converted_source_sha256"],"receipt_path":str(receipt.relative_to(root)),
        "receipt_sha256":sha(receipt),"original_path":str(original.relative_to(root)),"original_sha256":r["source_sha256"],
        "comparison_scope":r["comparison_scope"],"operation":r.get("operation"),
        "native_invocations":r.get("native_invocations"),"prior_receipt":r.get("prior_receipt")}
