# Engineering timing only: source the unchanged registered native worker.
# The supervisor supplies a separate request with only output paths redirected.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))

native_timeout_diagnostic <- function() {
  args <- commandArgs(trailingOnly=TRUE)
  if (length(args)!=2L) stop("Expected diagnostic request and SHA256")
  root <- normalizePath(getwd(), mustWork=TRUE)
  inside <- function(path) {
    value <- normalizePath(path, mustWork=TRUE)
    if (!startsWith(value,paste0(root,"/"))) stop("Diagnostic path escapes project")
    value
  }
  sha256 <- function(path) {
    out <- system2("sha256sum",shQuote(path),stdout=TRUE,stderr=TRUE)
    if (!is.null(attr(out,"status")) || length(out)!=1L) stop("Cannot hash diagnostic input")
    strsplit(out[[1]],"[[:space:]]+")[[1]][[1]]
  }
  request_path <- inside(args[[1]])
  directory <- dirname(request_path)
  amendment <- paste0(root,"/experiments/model_recommendation/execution_validation_20261001/engineering_amendments/native_timeout_diagnostic_20261002/")
  if (!startsWith(directory,paste0(amendment)) || basename(request_path)!="request.json")
    stop("A separate diagnostic request directory is required")
  registration_path <- inside(file.path(directory,"registration.json"))
  registration_sha <- sha256(registration_path)
  registration <- fromJSON(registration_path,simplifyVector=FALSE)
  worker <- inside(file.path(root,"src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R"))
  worker_sha <- "42e2e9d09abc63eb09d4a56610a809b0d3894265306b4b2b31bcffe93e52dad1"
  if (!isTRUE(registration$diagnostic_only) ||
      !identical(registration$formal_prediction_lock_created,FALSE) ||
      registration$timeout_seconds!=1800L ||
      !identical(registration$native_worker_sha256,worker_sha) ||
      sha256(worker)!=worker_sha || sha256(request_path)!=args[[2]] ||
      !identical(registration$diagnostic_request_sha256,args[[2]]))
    stop("Diagnostic registration or original worker differs")
  original_path <- inside(file.path(root,registration$original_request$path))
  original_sha <- registration$original_request$sha256
  if (!startsWith(original_path,paste0(root,"/experiments/model_recommendation/execution_validation_20261001/native/")) ||
      sha256(original_path)!=original_sha) stop("Original request binding differs")
  request <- fromJSON(request_path,simplifyVector=FALSE)
  original <- fromJSON(original_path,simplifyVector=FALSE)
  outputs <- c(result_path="native_result.json",before_pkml_path="before.pkml",after_pkml_path="after.pkml")
  if (!setequal(names(request),names(original)) || request$action!="run" || request$cpu_threads!=1L ||
      !setequal(unlist(registration$changed_fields),names(outputs))) stop("Diagnostic request scope differs")
  for (key in setdiff(names(original),names(outputs))) {
    if (!isTRUE(all.equal(request[[key]],original[[key]],tolerance=0,check.attributes=TRUE)))
      stop(paste("Diagnostic changed original input:",key))
  }
  for (key in names(outputs)) {
    expected <- file.path(directory,outputs[[key]])
    if (!identical(request[[key]],expected) || file.exists(expected))
      stop("Diagnostic outputs must be new files in the registered directory")
  }
  timing_path <- file.path(directory,"stages.jsonl")
  if (file.exists(timing_path)) stop("Never replace diagnostic timing evidence")
  timing <- file(timing_path,open="wt")
  on.exit(close(timing),add=TRUE)
  started <- proc.time()[["elapsed"]]
  emit <- function(phase,edge,call_index=0L,duration=NULL) {
    row <- list(diagnostic_only=TRUE,formal_prediction_eligible=FALSE,
      phase=phase,edge=edge,call_index=call_index,
      utc=format(Sys.time(),"%Y-%m-%dT%H:%M:%OS6Z",tz="UTC"),
      elapsed_seconds=unname(proc.time()[["elapsed"]]-started),duration_seconds=duration)
    line <- toJSON(row,auto_unbox=TRUE,digits=17,null="null")
    cat(line,"\n",file=timing,sep="");flush(timing)
    cat("NATIVE_TIMEOUT_DIAGNOSTIC ",line,"\n",file=stderr(),sep="");flush(stderr())
    invisible(NULL)
  }
  install_trace <- function(name,where,label) {
    call_index <- 0L; entered <- NULL; phase <- label
    before <- function() {
      call_index <<- call_index+1L
      phase <<- if (label=="loadSimulation" && call_index==2L) "reload" else label
      entered <<- proc.time()[["elapsed"]]
      emit(phase,"before",call_index)
    }
    after <- function() {
      emit(phase,"after",call_index,unname(proc.time()[["elapsed"]]-entered))
    }
    invisible(trace(name,tracer=as.call(list(before)),exit=as.call(list(after)),where=where,print=FALSE))
  }
  worker_environment <- new.env(parent=.GlobalEnv)
  for (name in c("loadSimulation","saveSimulation","runSimulations","getOutputValues")) {
    assign(name,getExportedValue("ospsuite",name),envir=worker_environment)
    install_trace(name,worker_environment,name)
  }
  # The original source defines this function itself. Trace that exact function
  # when it is assigned, without editing or replacing its numerical body.
  proof_store <- new.env(parent=worker_environment)
  makeActiveBinding("proof_parameters",function(value) {
    if (missing(value)) return(get("implementation",envir=proof_store,inherits=FALSE))
    if (!is.function(value) || exists("implementation",envir=proof_store,inherits=FALSE))
      stop("Unexpected original proof function definition")
    assign("implementation",value,envir=proof_store)
    install_trace("implementation",proof_store,"proof_parameters")
    invisible(value)
  },worker_environment)
  emit("original_worker","before")
  failure <- tryCatch({
    source(worker,local=worker_environment,echo=FALSE,print.eval=FALSE,keep.source=FALSE)
    NULL
  },error=function(error) error)
  if (sha256(worker)!=worker_sha || sha256(original_path)!=original_sha ||
      sha256(request_path)!=args[[2]] || sha256(registration_path)!=registration_sha)
    stop("Diagnostic input changed during execution")
  if (!is.null(failure)) {
    emit("original_worker","error")
    stop(failure)
  }
  emit("original_worker","after",duration=unname(proc.time()[["elapsed"]]-started))
  invisible(NULL)
}
native_timeout_diagnostic()
