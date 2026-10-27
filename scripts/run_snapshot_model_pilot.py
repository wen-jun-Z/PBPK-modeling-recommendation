"""Bounded execution of one acquired PBPK JSON snapshot using project OSPSuite.

Produces engineering outputs only. Observations are not assumed independent, and
successful simulation does not change registry reuse flags or scientific labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def project_ospsuite_environment():
    prefix = ROOT/"runtime/ospsuite-env"
    library = prefix/"lib/R/library"
    return {**os.environ, "DOTNET_ROOT":str(ROOT/"runtime/ospsuite-dotnet"),
        "PATH":str(ROOT/"runtime/ospsuite-dotnet")+":"+str(prefix/"bin")+":/usr/bin:/bin",
        "R_LIBS_USER":str(library),"R_LIBS_SITE":str(library),
        "LD_LIBRARY_PATH":":".join(map(str,[library/"ospsuite/lib",prefix/"lib",prefix/"lib/R/lib"])),
        "DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1",
        "DOTNET_PROCESSOR_COUNT":"2","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
        "TMPDIR":str(ROOT/"runtime/cache"),"LC_ALL":"C.UTF-8"}


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    # RLIMIT_FSIZE makes the CLR host exit with SIGXFSZ even during package load.
    # Enforce the total project-output limit from the supervising parent instead.



def validate_snapshot_outputs(output, snapshot, source_sha256):
    outputs=output/"outputs"
    expected=[r["Name"] for r in snapshot["Simulations"]]
    index=outputs/"outputs.csv"
    indexed=[]
    if index.exists():
        with index.open(encoding="utf-8-sig",newline="") as handle:
            indexed=[r.get("Sim_id","") for r in csv.DictReader(handle)]
    reports=[]
    for path in sorted(outputs.rglob("*-Results.csv")):
        with path.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.reader(handle);header=next(reader,[]);data=list(reader)
        bad=0;positive_signals=0;times={};monotone=True
        for texts in data:
            if len(texts)!=len(header):bad+=1;continue
            try:values=[float(v) for v in texts]
            except ValueError:bad+=1;continue
            if not all(math.isfinite(v) for v in values):bad+=1;continue
            if len(values)>=3:
                person,t=values[:2]
                if person in times and t<times[person]:monotone=False
                times[person]=t
                positive_signals+=sum(v>0 for v in values[2:])
        header_valid=(len(header)>=3 and header[0]=="IndividualId" and header[1].startswith("Time ["))
        reports.append({"path":str(path.relative_to(output)),"rows":len(data),"columns":len(header),
            "invalid_numeric_rows":bad,"positive_signal_values":positive_signals,
            "time_monotone_within_individual":monotone,"header_valid":header_valid,
            "sha256":digest(path),"pkml_exists":path.with_name(path.name.removesuffix("-Results.csv")+".pkml").is_file()})
    pkml=[{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(outputs.rglob("*.pkml"))]
    original=json.loads((output/"execution_summary.json").read_text())
    source_verified=original["source_sha256"]==source_sha256 and original["source_unchanged"]
    index_matches=(len(indexed)==len(set(indexed))==len(expected) and set(indexed)==set(expected))
    accepted=(original["returncode"]==0 and original["stop_reason"] is None and source_verified
        and index_matches and len(reports)==len(pkml)==len(expected)
        and all(r["rows"]>1 and r["header_valid"] and r["invalid_numeric_rows"]==0
            and r["positive_signal_values"]>0 and r["time_monotone_within_individual"] and r["pkml_exists"] for r in reports))
    return {"validator_version":"pbpk-json-execution-output-1.0.0","engineering_accepted":accepted,
        "scientific_accepted":False,"independent_observations_verified":False,"registry_modified":False,
        "source_sha256":source_sha256,"source_verified":source_verified,"simulation_index_matches_snapshot":index_matches,
        "expected_simulations":len(expected),"csv_result_count":len(reports),"pkml_count":len(pkml),
        "index_csv_role":"outputs.csv lists simulation IDs; it is not a concentration curve",
        "simulation_outputs":reports,"pkml_outputs":pkml,"original_summary_sha256":digest(output/"execution_summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=300)
    parser.add_argument("--verify-existing",action="store_true",help="Validate an existing execution without rerunning simulations")
    args=parser.parse_args()
    if not 1 <= args.timeout_seconds <= 1800:raise ValueError("Timeout must be 1..1800 seconds")
    catalog=ROOT/"experiments/model_recommendation/model_expansion_20260909/catalog_candidates.json"
    rows=json.loads(catalog.read_text())
    matches=[r for r in rows if r["model_family"]==args.model_family and r["project_relative_path"].endswith(".json")]
    if len(matches)!=1:raise ValueError("Choose a model family with exactly one registered JSON snapshot.")
    row=matches[0];source=(ROOT/row["project_relative_path"]).resolve();source.relative_to(ROOT)
    if row["license"]["status"]!="allowed":raise ValueError("Source license not allowed")
    if digest(source)!=row["project_sha256"]:raise ValueError("Snapshot checksum drift")
    output=args.output.resolve();output.relative_to(ROOT)
    if args.verify_existing:
        validation=validate_snapshot_outputs(output,json.loads(source.read_text()),row["project_sha256"])
        validation_path=output/"output_validation.json"
        if validation_path.exists():
            if json.loads(validation_path.read_text()) != validation:
                raise ValueError("Existing validation differs; preserve it and investigate changed inputs or outputs.")
        else:
            with validation_path.open("x") as handle:json.dump(validation,handle,indent=2)
        print(json.dumps({k:v for k,v in validation.items() if k not in ("simulation_outputs","pkml_outputs")},ensure_ascii=False))
        raise SystemExit(0 if validation["engineering_accepted"] else 1)
    output.mkdir(parents=True,exist_ok=False)
    runner=output/"run_snapshot.R"
    runner.write_text('''args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(ospsuite))
cat("ospsuite=",as.character(packageVersion("ospsuite")),"; rSharp=",as.character(packageVersion("rSharp")),"\\n")
runSimulationsFromSnapshot(args[[1]],output=args[[2]],exportCSV=TRUE,exportPKML=TRUE)
cat("SNAPSHOT_RUN_RETURNED\\n")
''')
    outputs=output/"outputs";outputs.mkdir()
    temporary=output/"temporary";temporary.mkdir()
    child_environment=project_ospsuite_environment()
    child_environment["TMPDIR"]=str(temporary)
    manifest={"model_family":row["model_family"],"repository":row["repository"],"commit":row["repository_commit"],
        "source":row["project_relative_path"],"source_sha256":row["project_sha256"],
        "expected_simulations":row["simulation_count"],"source_observed_dataset_count":row["dataset_reference_count"],
        "scientific_accepted":False,"independent_observations_verified":False,"registry_modified":False,
        "created_at":datetime.now(UTC).isoformat(),"timeout_seconds":args.timeout_seconds,
        "output_limit_bytes":2*1024**3,"process_cpu_limit_seconds":600,
        "claim_boundary":"Engineering simulation probe; no new scientific grade, reuse claim, or independent predictive accuracy."}
    started=time.monotonic();reason=None
    with (output/"execution.log").open("x") as log:
        child=subprocess.Popen([str(ROOT/"runtime/ospsuite-env/bin/Rscript"),"--vanilla",str(runner),str(source),str(outputs)],
            cwd=ROOT,env=child_environment,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=child_limits)
        (output/"pid.json").write_text(json.dumps({"pid":child.pid,"uid":os.getuid(),"cwd":str(ROOT),"script":str(runner)}))
        try:
            while child.poll() is None:
                size=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if size>manifest["output_limit_bytes"] or time.monotonic()-started>args.timeout_seconds:
                    reason="OUTPUT_SIZE_LIMIT" if size>manifest["output_limit_bytes"] else "EXECUTION_TIMEOUT"
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if reason is None and child.returncode is not None and child.returncode < 0:
        reason="PROCESS_SIGNAL:"+signal.Signals(-child.returncode).name
    manifest.update(returncode=child.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-started)
    csv_reports=[]
    for path in sorted(outputs.rglob("*-Results.csv")):
        with path.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.reader(handle);header=next(reader,[]);data=list(reader)
        finite=0;positive=0;numeric_failures=0
        for values in data:
            for text in values:
                try:value=float(text)
                except ValueError:continue
                if math.isfinite(value):finite+=1;positive+=value>0
                else:numeric_failures+=1
        csv_reports.append({"path":str(path.relative_to(output)),"rows":len(data),"columns":len(header),
            "finite_numeric_values":finite,"positive_numeric_values":positive,"nonfinite_values":numeric_failures,
            "sha256":digest(path)})
    pkml=[{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(outputs.rglob("*.pkml"))]
    manifest.update(csv_outputs=csv_reports,pkml_outputs=pkml)
    manifest["engineering_accepted"]=(child.returncode==0 and reason is None
        and len(csv_reports)==row["simulation_count"] and len(pkml)==row["simulation_count"]
        and all(r["rows"]>1 and r["finite_numeric_values"]>0 and r["nonfinite_values"]==0 for r in csv_reports))
    manifest["source_unchanged"]=digest(source)==row["project_sha256"]
    if not manifest["source_unchanged"]:raise RuntimeError("Source changed during execution")
    with (output/"execution_summary.json").open("x") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("csv_outputs","pkml_outputs")},ensure_ascii=False))
    print(json.dumps({"csv_count":len(csv_reports),"pkml_count":len(pkml)},ensure_ascii=False))
    validation=validate_snapshot_outputs(output,json.loads(source.read_text()),row["project_sha256"])
    with (output/"output_validation.json").open("x") as handle:json.dump(validation,handle,indent=2)
    print(json.dumps({"engineering_accepted":validation["engineering_accepted"],"validated_simulations":validation["csv_result_count"]}))
    raise SystemExit(0 if validation["engineering_accepted"] else 1)


if __name__=="__main__":main()
