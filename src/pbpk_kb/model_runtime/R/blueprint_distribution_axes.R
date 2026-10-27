# Separate equilibrium partition and symmetric cellular-exchange hypotheses.
# Every candidate reloads an immutable source. No clinical observations or fit.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request path and SHA256")
sha <- function(path) {
  text <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  status <- attr(text, "status")
  if ((!is.null(status) && status != 0) || length(text) != 1L) stop("Hash failed")
  strsplit(text[[1]], "[[:space:]]+")[[1]][[1]]
}
request_path <- normalizePath(args[[1]], mustWork=TRUE)
if (sha(request_path) != args[[2]]) stop("Request hash mismatch")
cfg <- fromJSON(request_path, simplifyVector=FALSE)
if (cfg$version != "native-distribution-axes-1") stop("Unknown version")
if (sha(cfg$pkml_path) != cfg$pkml_sha256) stop("Source PKML hash mismatch")
if (file.exists(cfg$result_path)) stop("Result already exists")
if (length(cfg$plan$changed) != 48L || length(cfg$candidates) < 1L || length(cfg$candidates) > 8L) stop("Invalid candidate/parameter count")
paths <- vapply(cfg$plan$changed, function(x) x$path, character(1))
protected_paths <- vapply(cfg$plan$protected, function(x) x$path, character(1))
if (anyDuplicated(c(paths, protected_paths))) stop("Overlapping or duplicate native paths")
read_values <- function(simulation, requested) {
  lapply(requested, function(path) {
    p <- getParameter(path, simulation)
    value <- as.numeric(p$value)
    if (length(value) != 1L || !is.finite(value)) stop(paste("Nonfinite native value", path))
    list(path=path, value=value, unit=if (is.null(p$unit)) "" else p$unit, dimension=p$dimension)
  })
}
close <- function(a, b) abs(a-b) <= 1e-12 + 1e-9 * abs(a)
runs <- list()
for (index in seq_along(cfg$candidates)) {
  scales <- cfg$candidates[[index]]
  if (any(!vapply(scales, function(x) is.numeric(x) && length(x)==1L && is.finite(x) && x>0, logical(1)))) stop("Invalid positive scales")
  simulation <- loadSimulation(cfg$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
  native_paths <- getAllParameterPathsIn(simulation)
  if (anyDuplicated(native_paths) || !all(c(paths, protected_paths) %in% native_paths)) stop("Native path binding differs")
  before <- read_values(simulation, paths)
  protected_before <- read_values(simulation, protected_paths)
  expected <- vapply(seq_along(paths), function(i) {
    if (!before[[i]]$unit %in% c("", "1", "fraction", "dimensionless")) stop("Axis targets dimensional parameter")
    value <- before[[i]]$value * scales[[cfg$plan$changed[[i]]$axis]]
    if (!is.finite(value) || value < 0) stop("Invalid scaled native value")
    value
  }, numeric(1))
  parameters <- lapply(paths, function(path) getParameter(path, simulation))
  setParameterValues(parameters, expected)
  setOutputs(cfg$target_output, simulation, stopIfNotFound=TRUE)
  output_path <- file.path(cfg$output_dir, sprintf("candidate_%02d.pkml", index-1L))
  if (file.exists(output_path)) stop("Candidate PKML already exists")
  saveSimulation(simulation, output_path)
  # Read exported artifact and run that exact reloaded model.
  simulation <- loadSimulation(output_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
  after <- read_values(simulation, paths)
  protected_after <- read_values(simulation, protected_paths)
  changed <- lapply(seq_along(paths), function(i) {
    if (!close(expected[[i]], after[[i]]$value) || before[[i]]$unit != after[[i]]$unit) stop("Changed parameter readback mismatch")
    list(path=paths[[i]], before=before[[i]]$value, after=after[[i]]$value,
         unit=after[[i]]$unit, dimension=after[[i]]$dimension)
  })
  protected <- lapply(seq_along(protected_paths), function(i) {
    if (!close(protected_before[[i]]$value, protected_after[[i]]$value) || protected_before[[i]]$unit != protected_after[[i]]$unit) stop(paste("Protected parameter changed", protected_paths[[i]]))
    list(path=protected_paths[[i]], before=protected_before[[i]]$value,
         after=protected_after[[i]]$value, unit=protected_after[[i]]$unit,
         dimension=protected_after[[i]]$dimension)
  })
  options <- SimulationRunOptions$new(numberOfCores=as.integer(cfg$native_cpu_cores), showProgress=FALSE)
  solved <- runSimulations(simulation, simulationRunOptions=options, silentMode=TRUE, stopIfFails=TRUE)[[1]]
  values <- getOutputValues(solved, quantitiesOrPaths=cfg$target_output, stopIfNotFound=TRUE, addMetaData=TRUE)
  metadata <- values$metaData[[cfg$target_output]]
  data <- values$data
  if (values$metaData$Time$unit != "min" || !grepl("concentration", metadata$dimension, ignore.case=TRUE) || length(unique(data$IndividualId)) != 1L) stop("Unexpected plasma output metadata")
  time <- as.numeric(data$Time)
  concentration <- as.numeric(data[[cfg$target_output]])
  if (length(time)<2L || length(time)!=length(concentration) || any(!is.finite(time)) || any(!is.finite(concentration)) || any(diff(time)<=0)) stop("Invalid native curve")
  runs[[index]] <- list(candidate_index=index-1L, scales=scales, changed=changed, protected=protected,
      pkml_file=basename(output_path), pkml_sha256=sha(output_path),
      curve=list(time_min=time, concentration=concentration, time_unit="min", unit=metadata$unit, dimension=metadata$dimension))
}
if (sha(request_path) != args[[2]] || sha(cfg$pkml_path) != cfg$pkml_sha256) stop("Source changed during execution")
result <- list(version=cfg$version, request_sha256=args[[2]], pkml_sha256=cfg$pkml_sha256,
    source_manifest_sha256=cfg$source_manifest_sha256, target_output=cfg$target_output,
    ospsuite_version=as.character(packageVersion("ospsuite")), runs=runs)
write_json(result, cfg$result_path, auto_unbox=TRUE, pretty=TRUE, digits=17, na="null", null="null")
cat("NATIVE_DISTRIBUTION_AXES_FINISHED\n")
