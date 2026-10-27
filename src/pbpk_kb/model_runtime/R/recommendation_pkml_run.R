# Execute a complete source PKML; only explicit administration values may change.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request and SHA256")
sha <- function(path) {
  out <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  if (!is.null(attr(out,"status")) || !length(out)) stop("Cannot hash input")
  strsplit(out[[1]], "[[:space:]]+")[[1]][[1]]
}
if (sha(args[[1]]) != args[[2]]) stop("Request checksum differs")
request <- fromJSON(args[[1]], simplifyVector=FALSE)
if (request$version != "recommendation-complete-pkml-v1") stop("Unknown request")
if (!request$action %in% c("inspect", "run")) stop("Unknown action")
if (sha(request$pkml_path) != request$pkml_sha256) stop("Original model checksum differs")
if (file.exists(request$result_path)) stop("Never replace a result")
simulation <- loadSimulation(request$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
containers <- getAllContainerPathsIn(simulation)
if (anyDuplicated(paths) || anyDuplicated(containers)) stop("Ambiguous native paths")
read_parameter <- function(path) {
  if (sum(paths == path) != 1L) stop("Exactly one native parameter is required")
  p <- getParameter(path, simulation)
  value <- as.numeric(p$value)
  if (length(value) != 1L || !is.finite(value)) stop(paste("Parameter is not finite:",path))
  list(path=path, id=p$id, value=value, unit=if (is.null(p$unit)) "" else p$unit,
       dimension=if (is.null(p$dimension)) "" else p$dimension)
}
selected <- paths[grepl("^(Events|Applications)\\|.*\\|(Dose|Start time|Infusion time)$", paths) |
                  grepl("(^Organism\\|(Weight|Age)$|\\|Molecular weight$)", paths)]
if (length(selected)>4096L) stop("Inspection inventory exceeds bound")
inventory <- list(parameter_count=length(paths), container_count=length(containers),
                  observer_paths=unname(as.list(getAllObserverPathsIn(simulation))),
                  organism_containers=unname(as.list(containers[grepl("^Organism($|\\|)", containers)])),
                  parameters=lapply(sort(selected), function(path) {
                    tryCatch(c(read_parameter(path),list(status="evaluated")),
                      error=function(e)list(path=path,status="not_evaluable",error=conditionMessage(e)))
                  }))
result <- list(version=request$version, action=request$action, pkml_sha256=request$pkml_sha256,
               request_sha256=args[[2]], simulation_name=simulation$name, inventory=inventory,
               sealed_observations_accessed=FALSE, fitting_performed=FALSE, solver_calls=0L,
               ospsuite_version=as.character(packageVersion("ospsuite")), parameter_changes=list())
proof_parameters <- function() {
  requirements <- request$proof_parameter_requirements
  if (is.null(requirements)) requirements <- list()
  if (length(requirements)>4096L) stop("Too many static proof parameters")
  out <- lapply(requirements, function(requirement) {
    row <- read_parameter(requirement$path)
    if (row$id!=requirement$id || row$dimension!=requirement$dimension ||
        !isTRUE(requirement$finite_required) ||
        (isTRUE(requirement$nonnegative_required) && row$value<0) ||
        (isTRUE(requirement$strictly_positive_required) && row$value<=0))
      stop(paste("Static proof coefficient domain differs:", requirement$path))
    row
  })
  if (anyDuplicated(vapply(out, function(x)x$path, character(1)))) stop("Duplicate static proof parameter")
  out
}
if (request$action == "inspect") result$proof_parameter_readback <- proof_parameters()
if (request$action == "run") {
  if (length(request$patches)>256L) stop("Too many administration changes")
  saveSimulation(simulation, request$before_pkml_path)
  changes <- lapply(request$patches, function(patch) {
    path <- patch$path
    if (!grepl("^(Events|Applications)\\|.*\\|ProtocolSchemaItem\\|(Dose|Start time|Infusion time)$", path))
      stop("Only explicit administration scalar values may be changed")
    before <- read_parameter(path)
    if (before$unit != patch$expected_before_unit ||
        abs(before$value-patch$expected_before_value)>1e-12*max(abs(before$value),1e-15))
      stop("Source administration value differs from its frozen expectation")
    p <- getParameter(path, simulation)
    value <- as.numeric(toBaseUnit(p, patch$value, patch$unit))
    if (length(value)!=1L || !is.finite(value) || value<0) stop("Invalid converted administration value")
    setParameterValues(p, value)
    after <- read_parameter(path)
    if (abs(after$value-value)>1e-12*max(abs(value),1e-15)) stop("Administration readback differs")
    list(path=path, id=before$id, before=before, after=after, requested_value=patch$value,
         requested_unit=patch$unit, expected_native_value=value)
  })
  if (anyDuplicated(vapply(changes,function(x)x$path,character(1)))) stop("Duplicate administration changes")
  times <- as.numeric(unlist(request$output_times_min))
  if (length(times)<2L || length(times)>50000L || any(!is.finite(times)) ||
      times[[1]]!=0 || any(diff(times)<=0)) stop("Invalid frozen output grid")
  simulation$outputSchema$clear()
  # PK-Sim 12.3 does not persist lone timePoints when saving PKML. Register
  # exact piecewise-uniform intervals, then solve the reloaded serialized grid.
  i <- 1L; interval_count <- 0L
  while (i < length(times)) {
    step <- times[[i+1L]]-times[[i]]; j <- i+1L
    while (j < length(times) && abs((times[[j+1L]]-times[[j]])-step)<=1e-12+1e-9*abs(step)) j <- j+1L
    interval_count <- interval_count+1L
    if (interval_count>2048L) stop("Too many distinct output intervals")
    addOutputInterval(simulation, times[[i]], times[[j]], (j-i)/(times[[j]]-times[[i]]))
    i <- j
  }
  monitors <- as.character(unlist(request$zero_monitor_paths))
  if (length(monitors)>4096L || anyDuplicated(monitors) || request$output_path %in% monitors)
    stop("Invalid zero-exposure monitor registration")
  domains <- names(request$positive_domain_monitor_dimensions)
  if (length(domains)>128L || anyDuplicated(domains) || length(intersect(domains,monitors)) || request$output_path %in% domains)
    stop("Invalid dynamic domain monitors")
  output_paths <- c(request$output_path, monitors, domains)
  setOutputs(output_paths, simulation, stopIfNotFound=TRUE)
  saveSimulation(simulation, request$after_pkml_path)
  # Reload the exact exported model before solving it.
  simulation <- loadSimulation(request$after_pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
  for (change in changes) {
    actual <- read_parameter(change$path)
    if (abs(actual$value-change$expected_native_value)>1e-12*max(abs(change$expected_native_value),1e-15))
      stop("Reloaded administration differs")
  }
  result$proof_parameter_readback <- proof_parameters()
  options <- SimulationRunOptions$new(numberOfCores=as.integer(request$cpu_threads), showProgress=FALSE)
  runs <- runSimulations(simulation, simulationRunOptions=options, silentMode=TRUE, stopIfFails=TRUE)
  if (length(runs)!=1L || is.null(runs[[1]])) stop("Exactly one native result is required")
  values <- getOutputValues(runs[[1]], quantitiesOrPaths=output_paths, stopIfNotFound=TRUE, addMetaData=TRUE)
  meta <- values$metaData[[request$output_path]]; data <- values$data
  if (values$metaData$Time$unit!="min" || length(unique(data$IndividualId))!=1L) stop("Unexpected time or population output")
  if (is.null(meta$unit) || !grepl("concentration",meta$dimension,ignore.case=TRUE)) stop("Concentration metadata required")
  time <- as.numeric(data$Time); concentration <- as.numeric(data[[request$output_path]])
  if (length(time)<2L || length(time)!=length(concentration) || any(!is.finite(time)) ||
      any(!is.finite(concentration)) || any(diff(time)<=0) || any(concentration<0)) stop("Invalid native curve")
  # SimulationResults exports single-precision time values. Verify the exact
  # registered grid after that documented numeric encoding, not a loose timing
  # tolerance; retain both raw output and the registered double-precision grid.
  encoded_times <- readBin(writeBin(as.double(times), raw(), size=4L), what="double", n=length(times), size=4L)
  if (length(time)!=length(times) || any(pmin(abs(time-times),abs(time-encoded_times))>1e-10)) stop("Native grid differs from registered output times")
  result$output_grid_verified <- TRUE
  result$native_output_time_min <- time
  result$time_encoding <- "registered float64 grid with independently checked float32 native serialization"
  result$curve <- list(time_min=times, concentration=concentration, time_unit="min", unit=meta$unit,
                       dimension=meta$dimension, output_path=request$output_path)
  if (length(monitors)) {
    if (!setequal(names(request$zero_monitor_dimensions),monitors)) stop("Monitor dimensions differ")
    result$zero_monitors <- lapply(monitors,function(path) {
      m <- values$metaData[[path]]; v <- as.numeric(data[[path]])
      expected_dimension <- request$zero_monitor_dimensions[[path]]
      allowed_units <- if (expected_dimension=="Amount") c("µmol","μmol","umol","mol","mmol","nmol") else if (expected_dimension=="Mass") c("kg","g","mg","µg","μg","ug","ng") else if (expected_dimension=="Amount per time") c("µmol/min","μmol/min","umol/min","mol/min","mmol/min") else if (expected_dimension=="Dimensionless" && endsWith(path,"|TabletIsActive")) "" else character()
      if (is.null(m) || is.null(m$unit) || m$dimension!=expected_dimension || !m$unit %in% allowed_units ||
          length(v)!=length(times) || any(!is.finite(v)) || any(v!=0))
        stop(paste("Removed drug mass is not exactly zero on the complete frozen grid:",path))
      list(path=path,dimension=m$dimension,unit=m$unit,values=unname(as.list(v)))
    })
    result$zero_monitor_time_min <- unname(as.list(times))
    result$zero_monitor_native_time_min <- unname(as.list(time))
    result$zero_monitor_grid_verified <- TRUE
  } else result$zero_monitors <- list()
  if (length(domains)) {
    result$positive_domain_monitors <- lapply(domains,function(path) {
      m <- values$metaData[[path]]; v <- as.numeric(data[[path]])
      if (is.null(m) || m$dimension!="Volume" || request$positive_domain_monitor_dimensions[[path]]!="Volume" ||
          !m$unit %in% c("l","L","ml","mL","µl","ul") || length(v)!=length(times) || any(!is.finite(v)) || any(v<=0))
        stop(paste("Dynamic volume denominator domain fails on registered grid:",path))
      list(path=path,dimension=m$dimension,unit=m$unit,values=unname(as.list(v)))
    })
    result$domain_monitor_time_min <- unname(as.list(times))
    result$domain_monitor_native_time_min <- unname(as.list(time))
    result$domain_monitor_grid_verified <- TRUE
  } else result$positive_domain_monitors <- list()
  result$parameter_changes <- changes
  result$solver_calls <- 1L
  result$before_pkml_sha256 <- sha(request$before_pkml_path)
  result$after_pkml_sha256 <- sha(request$after_pkml_path)
}
if (sha(request$pkml_path)!=request$pkml_sha256 || sha(args[[1]])!=args[[2]]) stop("Original input changed")
write_json(result, request$result_path, auto_unbox=TRUE, pretty=TRUE, digits=17, null="null", na="null")
cat("COMPLETE_PKML_NATIVE_FINISHED\n")
