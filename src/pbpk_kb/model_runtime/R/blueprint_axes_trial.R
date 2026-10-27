# One native solve after independent parameter/readback and clearance gates.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request path and hash")
sha <- function(path) {
  text <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  status <- attr(text, "status")
  if ((!is.null(status) && status != 0) || length(text)!=1L) stop("Hash failed")
  strsplit(text[[1]], "[[:space:]]+")[[1]][[1]]
}
request_path <- normalizePath(args[[1]], mustWork=TRUE)
if (sha(request_path) != args[[2]]) stop("Request hash mismatch")
cfg <- fromJSON(request_path, simplifyVector=FALSE)
if (cfg$version != "native-three-axes-trial-1" || sha(cfg$pkml_path)!=cfg$pkml_sha256) stop("Version/source mismatch")
if (file.exists(cfg$result_path)) stop("Output already exists")
receipt_path <- file.path(cfg$output_dir, "solver_receipt.json")
write_receipt <- function(count) {
  json <- list(request_sha256=args[[2]], actual_native_solver_calls=count)
  temporary <- paste0(receipt_path, ".tmp")
  write_json(json, temporary, auto_unbox=TRUE)
  if (!file.rename(temporary, receipt_path)) stop("Cannot persist native solver receipt")
}
write_receipt(0L)
result <- list(version=cfg$version, request_sha256=args[[2]], pkml_sha256=cfg$pkml_sha256,
  source_manifest_sha256=cfg$source_manifest_sha256, target_output=cfg$target_output,
  scales=cfg$scales, status="infrastructure_failed", actual_native_solver_calls=0L)
close <- function(a,b) abs(a-b) <= 1e-12 + 1e-9*abs(a)
scientific <- function(message) stop(structure(list(message=message, call=NULL), class=c("scientific_rejection", "error", "condition")))
read_values <- function(simulation, specs, canonical=FALSE) {
  lapply(specs, function(spec) {
    p <- getParameter(spec$path, simulation)
    value <- as.numeric(p$value)
    unit <- if (is.null(p$unit)) "" else p$unit
    if (length(value)!=1L || !is.finite(value)) stop(paste("Nonfinite parameter",spec$path))
    if (canonical) {
      if (spec$unit == "fraction") {
        if (!unit %in% c("", "1", "dimensionless", "fraction")) stop("Dimensionless unit mismatch")
      } else value <- as.numeric(toUnit(p, value, spec$unit))
      unit <- spec$unit
    }
    if (length(value)!=1L || !is.finite(value)) stop("Canonical unit conversion failed")
    list(path=spec$path, value=value, unit=unit, dimension=p$dimension)
  })
}
pair_values <- function(before, after) {
  lapply(seq_along(before), function(i) {
    if (before[[i]]$path!=after[[i]]$path || before[[i]]$unit!=after[[i]]$unit) stop("Readback binding changed")
    list(path=before[[i]]$path, before=before[[i]]$value, after=after[[i]]$value,
         unit=after[[i]]$unit, dimension=after[[i]]$dimension)
  })
}
capacity <- function(rows, field) {
  values <- setNames(vapply(rows,function(x)x[[field]],numeric(1)), vapply(rows,function(x)x$path,character(1)))
  lapply(cfg$plan$clearance_processes, function(process) {
    v <- function(name) unname(values[[paste0(process$stem,"|",name)]])
    factor <- switch(process$clearance_unit, "ml/min/kg"=.001, "l/h/kg"=1/60, "ml/h/kg"=.001/60, stop("Unknown CL unit"))
    cl <- v("Plasma clearance")*factor*v("Body weight")
    fu <- v("Fraction unbound (experiment)")
    if (process$organ == "hepatic") {
      flow <- v("Blood flow rate (liver)")+v("Blood flow rate (portal vein)")
      ratio <- v("Blood/Plasma concentration ratio")
      denominator <- if (ratio>0) flow-cl/ratio else -1
      volume <- v("Volume (liver)")*v("Fraction intracellular (liver)")
      fraction_valid <- v("Fraction intracellular (liver)")>0 && v("Fraction intracellular (liver)")<=1
    } else {
      flow <- v("Blood flow rate (kidney)")*(1-v("Hematocrit"))
      denominator <- flow-cl
      volume <- v("Volume (kidney)")
      fraction_valid <- v("Hematocrit")>=0 && v("Hematocrit")<1
    }
    actual <- v("Specific clearance")
    valid <- denominator>0 && flow>0 && volume>0 && fu>0 && fu<=1 && cl>0 && fraction_valid && v("Body weight")>0
    expected <- if (valid) cl*flow/(fu*denominator*volume) else NA_real_
    accepted <- valid && is.finite(expected) && actual>0 && actual<1e6 && abs(actual-expected)<=1e-10+1e-6*abs(expected)
    list(organ=process$organ, accepted=accepted)
  })
}
phase <- "load"
tryCatch({
  if (length(cfg$plan$changed)!=48L) stop("Unexpected distribution plan")
  if (any(!vapply(cfg$scales,function(x)is.numeric(x)&&length(x)==1L&&is.finite(x)&&x>0,logical(1)))) stop("Invalid scales")
  if (!cfg$plan$clearance_axis_available && cfg$scales$clearance_scale!=1) scientific("No nonrenal clearance axis; net renal input is protected")
  simulation <- loadSimulation(cfg$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
  changed_paths <- vapply(cfg$plan$changed,function(x)x$path,character(1))
  native_paths <- getAllParameterPathsIn(simulation)
  all_requested <- unique(c(changed_paths, vapply(cfg$plan$protected,function(x)x$path,character(1)), vapply(cfg$plan$clearance_parameters,function(x)x$path,character(1))))
  if (anyDuplicated(native_paths) || !all(all_requested %in% native_paths)) stop("Native exact path plan does not match source")
  before <- read_values(simulation,cfg$plan$changed)
  protected_before <- read_values(simulation,cfg$plan$protected)
  clearance_before <- read_values(simulation,cfg$plan$clearance_parameters,TRUE)
  for (i in seq_along(clearance_before)) {
    spec <- cfg$plan$clearance_parameters[[i]]
    if (!is.null(spec$expected) && abs(clearance_before[[i]]$value-spec$expected)>1e-12+spec$relative_tolerance*abs(spec$expected)) stop("Source native input does not equal raw evidence")
  }
  phase <- "modify"
  for (i in seq_along(changed_paths)) {
    if (!before[[i]]$unit %in% c("", "1", "fraction", "dimensionless")) stop("Distribution dimension mismatch")
    value <- before[[i]]$value*cfg$scales[[cfg$plan$changed[[i]]$axis]]
    if (!is.finite(value) || value<0) scientific("Nonfinite or negative distribution candidate")
    setParameterValues(getParameter(changed_paths[[i]],simulation),value)
  }
  for (spec in cfg$plan$clearance_changes) {
    p <- getParameter(spec$path,simulation)
    value <- as.numeric(p$value)*cfg$scales$clearance_scale
    if (!is.finite(value) || value<=0) scientific("Nonfinite or nonpositive clearance candidate")
    setParameterValues(p,value)
  }
  setOutputs(cfg$target_output,simulation,stopIfNotFound=TRUE)
  exported <- file.path(cfg$output_dir,"candidate.pkml")
  saveSimulation(simulation,exported)
  result$exported_pkml <- list(path=basename(exported),sha256=sha(exported))
  simulation <- loadSimulation(exported,loadFromCache=FALSE,addToCache=FALSE,resetIds=FALSE)
  phase <- "readback"
  result$changed <- pair_values(before,read_values(simulation,cfg$plan$changed))
  result$protected <- pair_values(protected_before,read_values(simulation,cfg$plan$protected))
  result$clearance_readback <- pair_values(clearance_before,read_values(simulation,cfg$plan$clearance_parameters,TRUE))
  for (i in seq_along(result$changed)) {
    row <- result$changed[[i]]
    if (!close(row$after,row$before*cfg$scales[[cfg$plan$changed[[i]]$axis]])) stop("Distribution readback mismatch")
  }
  for (row in result$protected) if (!close(row$before,row$after)) stop(paste("Protected quantity changed",row$path))
  changing <- vapply(cfg$plan$clearance_changes,function(x)x$path,character(1))
  derived <- vapply(Filter(function(x)x$organ=="hepatic",cfg$plan$clearance_processes),function(x)paste0(x$stem,"|Specific clearance"),character(1))
  for (row in result$clearance_readback) {
    if (row$path %in% derived) next
    expected <- row$before * if (row$path %in% changing) cfg$scales$clearance_scale else 1
    if (!close(row$after,expected)) stop("Clearance readback/protection mismatch")
  }
  source_checks <- capacity(result$clearance_readback,"before")
  trial_checks <- capacity(result$clearance_readback,"after")
  # A raw evidence input can exceed native capacity; a lower CL candidate may
  # repair it. Record the source diagnostic, but gate the actual trial model.
  if (!all(vapply(trial_checks,function(x)x$accepted,logical(1)))) scientific("Native clearance capacity/inversion gate rejected candidate")
  phase <- "solve"
  options <- SimulationRunOptions$new(numberOfCores=as.integer(cfg$native_cpu_cores),showProgress=FALSE)
  result$actual_native_solver_calls <- 1L
  write_receipt(1L)
  solved <- runSimulations(simulation,simulationRunOptions=options,silentMode=TRUE,stopIfFails=TRUE)[[1]]
  phase <- "extract"
  values <- getOutputValues(solved,quantitiesOrPaths=cfg$target_output,stopIfNotFound=TRUE,addMetaData=TRUE)
  metadata <- values$metaData[[cfg$target_output]]
  if (values$metaData$Time$unit!="min" || !grepl("concentration",metadata$dimension,ignore.case=TRUE) || length(unique(values$data$IndividualId))!=1L) stop("Invalid output metadata")
  time <- as.numeric(values$data$Time)
  concentration <- as.numeric(values$data[[cfg$target_output]])
  if (length(time)<2L || length(time)!=length(concentration) || any(!is.finite(time)) || any(!is.finite(concentration)) || any(diff(time)<=0)) scientific("Invalid native curve")
  result$curve <- list(time_min=time,concentration=concentration,time_unit="min",unit=metadata$unit,dimension=metadata$dimension)
  result$status <- "predicted"
}, scientific_rejection=function(e) {
  result$status <<- "scientific_failed";result$failure <<- conditionMessage(e)
}, error=function(e) {
  result$status <<- if (phase=="solve") "scientific_failed" else "infrastructure_failed"
  result$failure <<- paste(phase,conditionMessage(e))
})
if (sha(request_path)!=args[[2]] || sha(cfg$pkml_path)!=cfg$pkml_sha256) stop("Frozen source changed")
write_json(result,cfg$result_path,auto_unbox=TRUE,pretty=TRUE,digits=17,na="null",null="null")
cat("NATIVE_AXES_TRIAL_FINISHED\n")
