# One native CPU batch. No observation data, optimizer, interpolation or GPU.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Expected request JSON and its SHA256.")
sha256 <- function(path) {
  output <- system2("sha256sum", shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) stop("Could not verify input SHA256.")
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}
config_path <- normalizePath(args[[1]], mustWork = TRUE)
if (sha256(config_path) != args[[2]]) stop("Request JSON checksum mismatch.")
config <- fromJSON(config_path, simplifyVector = FALSE)
if (config$adapter_version != "osp-parameter-batch-1.1.0") stop("Unsupported batch schema.")
cores <- config$native_cpu_cores
if (!is.numeric(cores) || length(cores) != 1 || !is.finite(cores) ||
    cores != as.integer(cores) || cores < 1 || cores > 32) stop("Invalid explicit native core limit.")
if (sha256(config$pkml_path) != config$pkml_sha256) stop("PKML checksum mismatch.")
if (length(config$values) < 1 || length(config$values) > 64) stop("Batch vector limit exceeded.")
if (file.exists(config$result_path)) stop("Native result output already exists.")
specs <- config$parameter_specs
paths <- vapply(specs, function(spec) spec$path, character(1))
if (length(paths) < 1 || anyDuplicated(paths)) stop("Invalid variable parameter paths.")

simulation <- loadSimulation(config$pkml_path, loadFromCache = FALSE, addToCache = FALSE, resetIds = FALSE)
setOutputs(config$target_output, simulation, stopIfNotFound = TRUE)
parameters <- lapply(paths, function(path) getParameter(path, simulation))
names(parameters) <- paths
specs_by_path <- specs
names(specs_by_path) <- paths
batch <- createSimulationBatch(simulation = simulation, parametersOrPaths = paths)
native_order <- batch$getVariableParameters()
if (!is.character(native_order) || anyDuplicated(native_order) || !setequal(paths, native_order)) {
  stop("Native batch variable parameter mapping differs from the request.")
}
to_native <- function(parameter, value, unit) {
  if (!is.numeric(value) || length(value) != 1 || !is.finite(value)) stop("Nonfinite parameter value.")
  if (unit %in% c("dimensionless", "fraction", "1")) {
    native_unit <- parameter$unit
    if (!is.null(native_unit) && !native_unit %in% c("", "dimensionless", "fraction", "1")) {
      stop("Dimensionless request cannot target a dimensional native parameter.")
    }
    return(as.numeric(value))
  }
  converted <- toBaseUnit(parameter, as.numeric(value), unit)
  if (length(converted) != 1 || !is.finite(converted)) stop("Parameter unit conversion failed.")
  converted
}
run_ids <- character(length(config$values))
for (index in seq_along(config$values)) {
  requested <- config$values[[index]]
  if (length(requested) != length(paths)) stop("Parameter vector length differs from specification.")
  names(requested) <- paths
  native_values <- vapply(native_order, function(path) {
    to_native(parameters[[path]], requested[[path]], specs_by_path[[path]]$unit)
  }, numeric(1))
  id <- batch$addRunValues(parameterValues = native_values)
  if (!is.character(id) || length(id) != 1 || !nzchar(id)) stop("Native run ID is invalid.")
  run_ids[[index]] <- id
}
if (anyDuplicated(run_ids)) stop("Native run IDs are not unique.")
run_options <- SimulationRunOptions$new(numberOfCores = as.integer(cores), showProgress = FALSE)
results <- runSimulationBatches(list(batch), silentMode = TRUE, stopIfFails = TRUE,
                                simulationRunOptions = run_options)
# Keep inner run IDs directly: flattening would prepend the outer batch ID.
if (length(results) != 1) stop("Native batch result grouping differs from request.")
by_run <- results[[1]]
if (length(by_run) != length(run_ids) || anyDuplicated(names(by_run)) ||
    !setequal(names(by_run), run_ids)) stop("Native run IDs do not map one-to-one to parameter vectors.")
runs <- lapply(seq_along(run_ids), function(index) {
  values <- getOutputValues(by_run[[run_ids[[index]]]], quantitiesOrPaths = config$target_output,
                            stopIfNotFound = TRUE, addMetaData = TRUE)
  metadata <- values$metaData[[config$target_output]]
  if (is.null(metadata) || !is.character(metadata$unit) || !nzchar(metadata$unit) ||
      !grepl("concentration", metadata$dimension, ignore.case = TRUE)) {
    stop("Requested native output does not have explicit concentration metadata.")
  }
  if (values$metaData$Time$unit != "min") stop("Native time metadata is not minutes.")
  data <- values$data
  if (length(unique(data$IndividualId)) != 1) stop("Batch output unexpectedly contains multiple individuals.")
  time <- as.numeric(data$Time)
  concentration <- as.numeric(data[[config$target_output]])
  if (length(time) < 2 || length(time) != length(concentration) ||
      any(!is.finite(time)) || any(!is.finite(concentration)) || any(diff(time) <= 0)) {
    stop("Native curve is missing, nonfinite or has a non-increasing time grid.")
  }
  list(vector_index = index - 1L, run_id = run_ids[[index]],
       values = unname(config$values[[index]]), time_min = time,
       concentration = concentration, time_unit = "min", unit = metadata$unit,
       dimension = metadata$dimension)
})
result <- list(request_checksum = config$request_checksum, pkml_sha256 = config$pkml_sha256,
               native_cpu_cores = as.integer(run_options$numberOfCores),
               target_output = config$target_output, ospsuite_version = as.character(packageVersion("ospsuite")),
               runs = runs)
temporary <- paste0(config$result_path, ".tmp")
write_json(result, temporary, auto_unbox = TRUE, digits = 17, na = "null", null = "null")
if (!file.rename(temporary, config$result_path)) stop("Could not commit native batch results.")
