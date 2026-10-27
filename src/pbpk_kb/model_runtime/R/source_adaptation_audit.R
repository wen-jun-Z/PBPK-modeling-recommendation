# Read actual exported PKML values. This worker never runs or modifies a model.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected audit request JSON and its SHA256")
sha256 <- function(path) {
  output <- system2("sha256sum", shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if ((!is.null(status) && status != 0) || length(output) < 1L) stop("Could not hash audit input")
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}
request_path <- normalizePath(args[[1]], mustWork = TRUE)
if (sha256(request_path) != args[[2]]) stop("Audit request checksum mismatch")
request <- fromJSON(request_path, simplifyVector = FALSE)
if (request$adapter_version != "source-adaptation-audit-1.0") stop("Unsupported audit version")
if (sha256(request$pkml_path) != request$pkml_sha256) stop("PKML checksum mismatch")
if (file.exists(request$result_path)) stop("Audit output already exists")
if (length(request$expectations) < 1L || length(request$expectations) > 2048L) stop("Invalid expectation count")

simulation <- loadSimulation(request$pkml_path, loadFromCache = FALSE, addToCache = FALSE, resetIds = FALSE)
paths <- getAllParameterPathsIn(simulation)
containers <- getAllContainerPathsIn(simulation)
if (anyDuplicated(paths) || anyDuplicated(containers)) stop("Native model paths are ambiguous")
read_parameter <- function(path) {
  if (sum(paths == path) != 1L) stop("Expected exactly one native parameter at the requested path")
  p <- getParameter(path, simulation)
  if (is.null(p) || !identical(p$path, path)) stop("Native parameter lookup did not preserve the exact path")
  value <- as.numeric(p$value)
  if (length(value) != 1L || !is.finite(value)) stop("Native parameter value is not finite")
  unit <- p$unit
  if (is.null(unit)) unit <- ""
  dimension <- p$dimension
  if (is.null(dimension)) dimension <- ""
  list(parameter = p, value = value, unit = as.character(unit), dimension = as.character(dimension))
}
to_native <- function(p, value, unit) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) stop("Expected value is not finite")
  if (!is.character(unit) || length(unit) != 1L) stop("An explicit expected unit is required")
  if (unit %in% c("", "dimensionless", "fraction", "1")) {
    native_unit <- p$unit
    if (!is.null(native_unit) && !native_unit %in% c("", "dimensionless", "fraction", "1")) {
      stop("A dimensionless expectation cannot target a dimensional native parameter")
    }
    return(as.numeric(value))
  }
  converted <- toBaseUnit(p, as.numeric(value), unit)
  if (length(converted) != 1L || !is.finite(converted)) stop("Native unit conversion failed")
  as.numeric(converted)
}
checks <- lapply(seq_along(request$expectations), function(index) {
  expected <- request$expectations[[index]]
  result <- list(check_id = expected$check_id, path = expected$path,
                 expected_value = expected$value, expected_unit = expected$unit,
                 origin = expected$origin, label = expected$label,
                 status = "failed", native_value = NULL, native_unit = NULL,
                 native_dimension = NULL, expected_native_value = NULL, error = NULL)
  tryCatch({
    actual <- read_parameter(expected$path)
    result$native_value <- actual$value
    result$native_unit <- actual$unit
    result$native_dimension <- actual$dimension
    result$expected_native_value <- to_native(actual$parameter, expected$value, expected$unit)
    result$status <- "evaluated"
    result
  }, error = function(e) {
    result$error <- conditionMessage(e)
    result
  })
})

event_paths <- paths[grepl("^Events\\|.*\\|ProtocolSchemaItem\\|(Dose|Start time|Infusion time)$", paths)]
event_values <- lapply(sort(event_paths), function(path) {
  result <- list(path = path, event_path = sub("\\|[^|]+$", "", path),
                 field = sub("^.*\\|", "", path), status = "failed",
                 value = NULL, unit = NULL, dimension = NULL, error = NULL)
  tryCatch({
    actual <- read_parameter(path)
    result$value <- actual$value
    result$unit <- actual$unit
    result$dimension <- actual$dimension
    result$status <- "evaluated"
    result
  }, error = function(e) {
    result$error <- conditionMessage(e)
    result
  })
})
organism_containers <- containers[grepl("^Organism($|\\|)", containers)]
organism_parameters <- paths[grepl("^Organism\\|", paths)]
top_level <- sort(unique(sub("\\|.*$", "", containers)))
process_paths <- sort(paths[grepl("\\|(kcat|Km|Vmax|CLint|kinact|K_kinact_half|Ki)$", paths)])
if (length(event_paths) > 4096L || length(process_paths) > 16384L) stop("Native audit inventory exceeds bounded scope")
result <- list(adapter_version = request$adapter_version,
               request_sha256 = args[[2]], pkml_sha256 = request$pkml_sha256,
               simulation_name = simulation$name,
               parameter_checks = checks, events = event_values,
               inventory = list(parameter_count = length(paths), container_count = length(containers),
                 organism_parameter_count = length(organism_parameters),
                 organism_container_count = length(organism_containers),
                 top_level_containers = unname(as.list(top_level)),
                 organism_top_level_containers = unname(as.list(sort(organism_containers[lengths(strsplit(organism_containers, "|", fixed = TRUE)) <= 2L]))),
                 process_parameter_paths = unname(as.list(process_paths)),
                 process_parameter_count = length(process_paths),
                 process_count_interpretation = "Selected kinetic parameter paths, not a count of independent mechanisms"),
               ospsuite_version = as.character(packageVersion("ospsuite")),
               solver_calls = 0L, model_modified = FALSE)
if (sha256(request$pkml_path) != request$pkml_sha256 || sha256(request_path) != args[[2]]) {
  stop("Audit input changed during native inspection")
}
temporary <- paste0(request$result_path, ".tmp")
write_json(result, temporary, auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null")
if (!file.rename(temporary, request$result_path)) stop("Could not commit audit results")
cat("SOURCE_ADAPTATION_NATIVE_AUDIT_FINISHED\n")
